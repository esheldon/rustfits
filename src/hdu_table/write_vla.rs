// VLA write paths: per-column input extraction, heap-layout planning,
// per-cell serialization, the strip writer + descriptor builder,
// write_vla_aware + append_vla_aware + repack_table_heap.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::Ordering;

use crate::common::{
    check_not_tainted, lock_file, parse_keyword,
    shift_file_tail_and_update_offsets,
    shift_file_tail_backward_and_update_offsets, zero_fill_range,
    FileHandle, RawBuffer,
};
use crate::hdu::HDU;
use crate::hdu_image::{round_up_to_block, serialize_header_to_disk_bytes};
use crate::header::card_int;

use super::columns::{
    bytes_per_element, byteswap_unit, parse_columns, Column,
};
use super::read::{heap_base_in_data, read_descriptor};
use super::write_fixed::{
    apply_transform_cell, set_pcount_in_cards,
};
use super::write_setup::{
    column_expected_shape, column_transform, WriteTransform,
};

// True iff any column is variable-length (P/Q).  Dispatches the write
// path: fixed-only tables take the existing fast/slow strip writer;
// tables with any VLA column take the heap-aware path below.
pub(crate) fn any_var_column(columns: &[Column]) -> bool {
    columns.iter().any(|c| c.var_kind.is_some())
}

// Pull per-column input ndarrays out of any of the three accepted
// input forms (structured ndarray / dict / list+names), in column
// order.  Used by the VLA write path because structured-ndarray
// shared-buffer addressing breaks down once Object fields appear:
// every column needs its own per-row source array for the slow path.
//
// Validates the per-form structural constraints (extras / missing /
// duplicates / wrong length) but does NOT validate per-cell dtypes —
// that's per-column work the caller does next.
pub(crate) fn extract_per_column_inputs<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    names: Option<&Bound<'py, PyAny>>,
    columns: &[Column],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let column_names: std::collections::HashSet<&str> =
        columns.iter().map(|c| c.name.as_str()).collect();
    if data.is_instance_of::<PyDict>() {
        if names.is_some() {
            return Err(PyValueError::new_err(
                "names= is only valid with list/tuple data, not dict"));
        }
        let d = data.cast::<PyDict>()?;
        for k in d.keys() {
            let key: String = k.extract().map_err(|_| {
                PyValueError::new_err("dict keys must be strings")
            })?;
            if !column_names.contains(key.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "dict has extra key '{}' not in table columns", key)));
            }
        }
        let mut out = Vec::with_capacity(columns.len());
        for col in columns {
            let val = d.get_item(col.name.as_str())?
                .ok_or_else(|| PyValueError::new_err(format!(
                    "dict is missing column '{}'", col.name)))?;
            out.push(val);
        }
        Ok(out)
    } else if data.is_instance_of::<PyList>()
        || data.is_instance_of::<PyTuple>()
    {
        let names_obj = names.ok_or_else(|| PyValueError::new_err(
            "when data is a list/tuple, names= is required"))?;
        let arrays: Vec<Bound<'_, PyAny>> = data.try_iter()?
            .collect::<PyResult<Vec<_>>>()?;
        let provided_names: Vec<String> = names_obj.extract().map_err(|_| {
            PyValueError::new_err(
                "names= must be a sequence of strings")
        })?;
        if arrays.len() != provided_names.len() {
            return Err(PyValueError::new_err(format!(
                "len(data)={} != len(names)={}",
                arrays.len(), provided_names.len())));
        }
        let mut name_to_arr: std::collections::HashMap<String, Bound<'_, PyAny>> =
            std::collections::HashMap::with_capacity(provided_names.len());
        for (n, a) in provided_names.iter().zip(arrays.iter()) {
            if !column_names.contains(n.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "names list has extra entry '{}' not in table columns", n)));
            }
            if name_to_arr.insert(n.clone(), a.clone()).is_some() {
                return Err(PyValueError::new_err(format!(
                    "duplicate name '{}' in names list", n)));
            }
        }
        let mut out = Vec::with_capacity(columns.len());
        for col in columns {
            let val = name_to_arr.remove(&col.name)
                .ok_or_else(|| PyValueError::new_err(format!(
                    "column '{}' is missing from names list", col.name)))?;
            out.push(val);
        }
        Ok(out)
    } else {
        if names.is_some() {
            return Err(PyValueError::new_err(
                "names= is only valid with list/tuple data, not a \
                 structured ndarray"));
        }
        let np = py.import("numpy")?;
        let ndarray = np.getattr("ndarray")?;
        if !data.is_instance(&ndarray)? {
            return Err(PyValueError::new_err(
                "data must be a structured numpy ndarray, a dict \
                 {name: ndarray}, or a list/tuple of ndarrays with \
                 names=[...]"));
        }
        let dtype = data.getattr("dtype")?;
        let names_attr = dtype.getattr("names")?;
        if names_attr.is_none() {
            return Err(PyValueError::new_err(
                "structured input must have named fields"));
        }
        let input_names: Vec<String> = names_attr.extract()?;
        let input_names_set: std::collections::HashSet<&str> =
            input_names.iter().map(|s| s.as_str()).collect();
        for col in columns {
            if !input_names_set.contains(col.name.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "input dtype is missing field '{}' (table column)",
                    col.name)));
            }
        }
        // arr[col_name] on a structured ndarray returns a per-column
        // VIEW with stride == record itemsize, not stride == field
        // itemsize.  So for any input with more than one field, the
        // view is non-contiguous (RawBuffer.acquire would reject it)
        // because the write loop assumes tight packing — it indexes
        // `buffer[row * per_cell_bytes ..]` to get row N.  Calling
        // np.ascontiguousarray here materializes a compacted copy
        // when needed and is a no-op when the view is already
        // contiguous.  Cost: one memcpy per fixed column per write,
        // sized to the column's actual bytes.  For Object (VLA)
        // columns, the copy shuffles 8-byte pointers; the heap cells
        // themselves are untouched.
        //
        // FUTURE: a stride-aware FixedColInfo (carrying src_stride
        // alongside per_cell_bytes and indexing rows by stride) would
        // avoid this copy entirely.  Worth doing if profiling shows
        // the copy as a hot path for large structured + VLA inputs.
        let ascontiguousarray = np.getattr("ascontiguousarray")?;
        let mut out = Vec::with_capacity(columns.len());
        for col in columns {
            let view = data.get_item(col.name.as_str())?;
            out.push(ascontiguousarray.call1((view,))?);
        }
        Ok(out)
    }
}

// Maps an inner FITS letter to the numpy dtype kind/itemsize tuple
// that a VLA cell must have.  Mirrors classify_var_numpy_field but
// in the inverse direction (write-time validation against the on-disk
// column type rather than dtype → letter mapping).
fn vla_cell_expected_dtype(inner_letter: char) -> (&'static str, usize) {
    match inner_letter {
        // X (bit-packed) VLA: each cell is a 1-D numpy bool array of
        // length equal to the bit count.  Same dtype kind/size as L.
        'L' | 'X' => ("b", 1),
        'B' => ("u", 1),
        'I' => ("i", 2),
        'J' => ("i", 4),
        'K' => ("i", 8),
        'E' => ("f", 4),
        'D' => ("f", 8),
        'C' => ("c", 8),
        'M' => ("c", 16),
        _ => unreachable!(
            "vla_cell_expected_dtype called with unsupported inner '{}'",
            inner_letter),
    }
}

// Per-row VLA cell metadata captured during the validation pass.
// nelements is the cell's logical length; bytes_offset_in_heap is
// the cell's start position in the planned heap layout.  Caller
// can compute byte_count = nelements * elem_size (and the heap-
// builder uses the cell's stored ndarray bytes to do the actual
// big-endian serialization).
#[derive(Clone, Copy)]
pub(crate) struct VlaCellPlan {
    pub(crate) nelements: usize,
    pub(crate) bytes_offset_in_heap: usize,
}

// Encode a single ASCII string VLA cell (Python str / bytes / numpy
// str-scalar) to its on-disk bytes.  Mirrors the read side: 'A' cells
// hold ASCII text; non-ASCII bytes in a str input are rejected with
// the same message shape the read side raises.  Used by both the
// validate pass (length only) and the serialize pass (full bytes).
//
// Explicit isinstance checks rather than extract() because numpy
// scalars subclass str/bytes (good — they fall through to the right
// branch) but a numpy ndarray of integers would otherwise extract
// successfully as Vec<u8> (each int coerced), silently mis-typing
// the cell.
fn extract_string_vla_cell_bytes(
    cell: &Bound<'_, PyAny>,
    col_name: &str,
    row_idx: usize,
) -> PyResult<Vec<u8>> {
    // Python `bytes` / numpy.bytes_ scalar — take verbatim.
    if cell.is_instance_of::<PyBytes>() {
        return Ok(cell.extract::<Vec<u8>>()?);
    }
    // Python `str` / numpy.str_ scalar — ASCII-encode.
    if cell.is_instance_of::<PyString>() {
        let s: String = cell.extract()?;
        for (i, &b) in s.as_bytes().iter().enumerate() {
            if !b.is_ascii() {
                return Err(PyValueError::new_err(format!(
                    "column '{}' row {}: VLA string cell contains \
                     non-ASCII byte 0x{:02X} at position {}; pass \
                     bytes instead of str to store arbitrary bytes",
                    col_name, row_idx, b, i)));
            }
        }
        return Ok(s.into_bytes());
    }
    Err(PyValueError::new_err(format!(
        "column '{}' row {}: VLA string cell must be a Python str \
         (ASCII) or bytes; got {}",
        col_name, row_idx, cell.get_type().name()?)))
}

// Validate one VLA cell + return its element count.  Numeric cells
// must be a 1-D C-contiguous numpy ndarray with dtype matching the
// inner letter; 'A' string cells must be a Python str (ASCII) or
// bytes.  Empty cells (nelements == 0) are accepted (descriptor is
// just (0, current_heap_offset)).
pub(crate) fn validate_vla_cell(
    cell: &Bound<'_, PyAny>,
    ndarray: &Bound<'_, PyAny>,
    inner_letter: char,
    col_name: &str,
    row_idx: usize,
) -> PyResult<usize> {
    if inner_letter == 'A' {
        return Ok(extract_string_vla_cell_bytes(cell, col_name, row_idx)?.len());
    }
    if !cell.is_instance(ndarray)? {
        return Err(PyValueError::new_err(format!(
            "column '{}' row {}: VLA cell must be a numpy ndarray",
            col_name, row_idx)));
    }
    let shape: Vec<usize> = cell.getattr("shape")?.extract()?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "column '{}' row {}: VLA cell must be 1-D, got shape {:?}",
            col_name, row_idx, shape)));
    }
    let nelements = shape[0];
    let dtype = cell.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    let (expected_kind, expected_size) = vla_cell_expected_dtype(inner_letter);
    if kind != expected_kind || itemsize != expected_size {
        return Err(PyValueError::new_err(format!(
            "column '{}' row {}: VLA cell dtype kind '{}' itemsize {} \
             does not match expected inner type '{}' (kind '{}' \
             itemsize {})",
            col_name, row_idx, kind, itemsize, inner_letter,
            expected_kind, expected_size)));
    }
    if nelements > 0 {
        let flags = cell.getattr("flags")?;
        let c_contig: bool = flags.getattr("c_contiguous")?.extract()?;
        if !c_contig {
            return Err(PyValueError::new_err(format!(
                "column '{}' row {}: VLA cell ndarray must be C-contiguous",
                col_name, row_idx)));
        }
    }
    Ok(nelements)
}

// Plan the heap layout: walk every VLA cell in row-major order (per
// row, walk every VLA column) and assign each cell a heap offset.
// Returns per-column per-row plans plus the TOTAL heap size after
// this batch (== heap_start_offset + sum of cell bytes).
// `heap_start_offset` lets the caller start the layout at a non-zero
// position (used for VLA append, where new cells extend the existing
// heap rather than replacing it).
pub(crate) fn plan_vla_heap_layout(
    columns: &[Column],
    per_col: &[Bound<'_, PyAny>],
    nrows: usize,
    ndarray: &Bound<'_, PyAny>,
    heap_start_offset: usize,
) -> PyResult<(Vec<Vec<VlaCellPlan>>, usize)> {
    let mut plans: Vec<Vec<VlaCellPlan>> = columns.iter()
        .map(|c| if c.var_kind.is_some() {
            Vec::with_capacity(nrows)
        } else {
            Vec::new()
        })
        .collect();
    let mut cursor = heap_start_offset;
    for row_idx in 0..nrows {
        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_none() {
                continue;
            }
            let cell = per_col[col_idx].get_item(row_idx)?;
            let nelements = validate_vla_cell(
                &cell, ndarray, col.tform_letter, &col.name, row_idx)?;
            // X (bit-packed) VLA: nelements is the bit count; the
            // heap holds ceil(nelements/8) bytes per cell.  All
            // other inner letters have a fixed element width.
            let bytes = if col.tform_letter == 'X' {
                nelements.div_ceil(8)
            } else {
                let elem_size = bytes_per_element(col.tform_letter)
                    .unwrap_or(0);
                nelements * elem_size
            };
            plans[col_idx].push(VlaCellPlan {
                nelements,
                bytes_offset_in_heap: cursor,
            });
            cursor = cursor.checked_add(bytes).ok_or_else(|| {
                PyValueError::new_err("heap size overflow")
            })?;
        }
    }
    Ok((plans, cursor))
}

// Write one VLA cell's bytes into `dst`, byteswapping inner elements
// (numpy → big-endian on disk).  `dst.len()` must equal
// `nelements * elem_size`.
pub(crate) fn serialize_vla_cell(
    cell: &Bound<'_, PyAny>,
    inner_letter: char,
    nelements: usize,
    dst: &mut [u8],
) -> PyResult<()> {
    if nelements == 0 {
        return Ok(());
    }
    if inner_letter == 'A' {
        let bytes = extract_string_vla_cell_bytes(cell, "<vla>", 0)?;
        if bytes.len() != nelements {
            return Err(PyValueError::new_err(format!(
                "VLA string cell length {} differs from planned \
                 nelements {} (input changed between validate and \
                 serialize passes)",
                bytes.len(), nelements)));
        }
        dst[..nelements].copy_from_slice(&bytes);
        return Ok(());
    }
    if inner_letter == 'X' {
        // X (bit-packed) VLA cell: pack `nelements` bool source
        // bytes (one per element in numpy) into ceil(nelements/8)
        // MSB-first bytes; trailing bits in the last byte are
        // zeroed per the FITS spec.  Inverse of read.rs::
        // build_var_cell_value's X branch.
        let buf = RawBuffer::acquire(cell)?;
        let src = buf.as_slice();
        if src.len() < nelements {
            return Err(PyValueError::new_err(format!(
                "VLA X cell buffer length {} smaller than expected \
                 {} bools", src.len(), nelements)));
        }
        let n_bytes = nelements.div_ceil(8);
        for b in dst[..n_bytes].iter_mut() {
            *b = 0;
        }
        for i in 0..nelements {
            if src[i] != 0 {
                dst[i / 8] |= 1u8 << (7 - (i % 8));
            }
        }
        return Ok(());
    }
    let buf = RawBuffer::acquire(cell)?;
    let src = buf.as_slice();
    let elem_size = bytes_per_element(inner_letter).unwrap();
    let total = nelements * elem_size;
    if src.len() < total {
        return Err(PyValueError::new_err(format!(
            "VLA cell buffer length {} smaller than expected {}",
            src.len(), total)));
    }
    let swap_w = byteswap_unit(inner_letter);
    if inner_letter == 'L' {
        // numpy bool 0/1 → FITS L 'T'/'F'.  No byteswap.
        for i in 0..nelements {
            dst[i] = if src[i] == 0 { b'F' } else { b'T' };
        }
    } else if swap_w == 1 {
        dst[..total].copy_from_slice(&src[..total]);
    } else {
        let units = total / swap_w;
        for u in 0..units {
            let s = &src[u * swap_w..(u + 1) * swap_w];
            let d = &mut dst[u * swap_w..(u + 1) * swap_w];
            for k in 0..swap_w {
                d[k] = s[swap_w - 1 - k];
            }
        }
    }
    Ok(())
}

// Write a P or Q descriptor (nelements, heap_offset) into `dst` as
// big-endian.  P descriptors are 2 × i32 = 8 bytes; Q descriptors
// are 2 × i64 = 16 bytes.
pub(crate) fn write_descriptor(
    descriptor_kind: char,
    nelements: usize,
    heap_offset: usize,
    dst: &mut [u8],
) {
    match descriptor_kind {
        'P' => {
            let n = (nelements as i32).to_be_bytes();
            let off = (heap_offset as i32).to_be_bytes();
            dst[0..4].copy_from_slice(&n);
            dst[4..8].copy_from_slice(&off);
        }
        'Q' => {
            let n = (nelements as i64).to_be_bytes();
            let off = (heap_offset as i64).to_be_bytes();
            dst[0..8].copy_from_slice(&n);
            dst[8..16].copy_from_slice(&off);
        }
        _ => unreachable!(),
    }
}

// Per-column write info for the VLA-aware path.  Fixed and VLA
// columns are kept in parallel Vec<Option<...>>s indexed by column
// position so the strip-builder can dispatch without re-classifying.
pub(crate) struct FixedColInfo {
    pub(crate) buffer: RawBuffer,
    pub(crate) per_cell_bytes: usize,
    pub(crate) transform: WriteTransform,
}

pub(crate) struct VlaColInfo<'py> {
    // Per-row (nelements, heap_offset) plans, indexed by input row.
    pub(crate) plans: Vec<VlaCellPlan>,
    // The 1-D Object ndarray (held so the heap-serialization pass
    // can `arr[i]` to get each row's cell).
    pub(crate) per_col_array: Bound<'py, PyAny>,
}

// Validate a fixed column's per-column input ndarray against the on-
// disk column and acquire its raw buffer.  Mirrors
// acquire_per_column_array but the inputs are already extracted into
// per-column ndarrays by extract_per_column_inputs, so this function
// can borrow the buffer directly into the per-column FixedColInfo
// without going through the shared Vec<RawBuffer> indirection that
// the prepare_*_input functions need.
pub(crate) fn build_fixed_col_info(
    arr: &Bound<'_, PyAny>,
    ndarray: &Bound<'_, PyAny>,
    col: &Column,
    nrows: usize,
) -> PyResult<FixedColInfo> {
    if !arr.is_instance(ndarray)? {
        return Err(PyValueError::new_err(format!(
            "column '{}': value must be a numpy ndarray", col.name)));
    }
    let arr_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if arr_shape.is_empty() || arr_shape[0] != nrows {
        return Err(PyValueError::new_err(format!(
            "column '{}': array shape {:?} does not have first axis \
             == nrows ({})", col.name, arr_shape, nrows)));
    }
    let per_cell_shape: Vec<usize> = arr_shape[1..].to_vec();
    let expected_shape = column_expected_shape(col);
    if per_cell_shape != expected_shape {
        return Err(PyValueError::new_err(format!(
            "column '{}': per-cell shape {:?} does not match table \
             column expected shape {:?}",
            col.name, per_cell_shape, expected_shape)));
    }
    let arr_dtype = arr.getattr("dtype")?;
    let kind: String = arr_dtype.getattr("kind")?.extract()?;
    let elem_size: usize = arr_dtype.getattr("itemsize")?.extract()?;
    let transform = column_transform(col, &kind, elem_size)?;
    let cell_elements: usize =
        per_cell_shape.iter().product::<usize>().max(1);
    let per_cell_bytes = elem_size * cell_elements;
    let flags = arr.getattr("flags")?;
    let c_contig: bool = flags.getattr("c_contiguous")?.extract()?;
    if !c_contig {
        return Err(PyValueError::new_err(format!(
            "column '{}': ndarray must be C-contiguous", col.name)));
    }
    let buffer = RawBuffer::acquire(arr)?;
    let expected_bytes = nrows.checked_mul(per_cell_bytes)
        .ok_or_else(|| PyValueError::new_err("input size overflow"))?;
    if buffer.as_slice().len() < expected_bytes {
        return Err(PyValueError::new_err(format!(
            "column '{}': buffer length {} smaller than expected {}",
            col.name, buffer.as_slice().len(), expected_bytes)));
    }
    Ok(FixedColInfo { buffer, per_cell_bytes, transform })
}

// Fill one main-data row in `row_buf` from per-column inputs.  Fixed
// columns copy+transform from their per-column buffer; VLA columns
// write a descriptor pointing into the planned heap layout, using
// each column's own var_kind (different VLA columns may use
// different descriptor sizes in principle, though our writer emits
// a uniform descriptor= choice at create time).
fn fill_main_row(
    columns: &[Column],
    fixed: &[Option<FixedColInfo>],
    vla: &[Option<VlaColInfo<'_>>],
    input_row: usize,
    row_buf: &mut [u8],
) -> PyResult<()> {
    for (col_idx, col) in columns.iter().enumerate() {
        let dst = &mut row_buf
            [col.byte_offset..col.byte_offset + col.byte_width];
        if let Some(vci) = &vla[col_idx] {
            let plan = vci.plans[input_row];
            let kind = col.var_kind.unwrap();
            write_descriptor(
                kind, plan.nelements, plan.bytes_offset_in_heap, dst);
        } else if let Some(fci) = &fixed[col_idx] {
            let src_off = input_row * fci.per_cell_bytes;
            let src = &fci.buffer.as_slice()
                [src_off..src_off + fci.per_cell_bytes];
            apply_transform_cell(
                &fci.transform, src, dst, &col.name, input_row)?;
        } else {
            unreachable!(
                "column '{}' is neither fixed nor VLA", col.name);
        }
    }
    Ok(())
}

// Heart of the VLA-aware write path.  Writes `input_nrows` rows of
// main-table data (with embedded descriptors) starting at
// `main_start_offset` and the corresponding heap bytes starting at
// `heap_start_offset` in the file.  Returns the total bytes added
// to the heap (so the caller can update PCOUNT).
//
// The caller is responsible for everything OUTSIDE the bytes this
// function writes:
//  - File growth (set_len / shift_file_tail) to make room.
//  - Header rewrites (PCOUNT, NAXIS2).
//  - Old-heap relocation for append.
//
// Mid-write I/O failures taint the file.
#[allow(clippy::too_many_arguments)]
// Build the per-write VLA heap buffer in RAM.  Walks every VLA
// column's per-row plan and serializes each cell's bytes into the
// right offset within the buffer.  Shared between write_vla_data_range
// (contiguous main-row write) and write_vla_data_strided (per-row
// seek + write) — the heap layout depends only on the per-input-row
// plans, not on where each main row lands on disk.
//
// `added_heap_bytes = total_heap_bytes - heap_start_offset_in_heap`.
// X (bit-packed) cells contribute ceil(nelements/8) bytes; other
// inner letters use a fixed element width.  No file I/O.
fn build_vla_heap_buf(
    columns: &[Column],
    vla: &[Option<VlaColInfo<'_>>],
    total_heap_bytes: usize,
    heap_start_offset_in_heap: usize,
    input_nrows: usize,
) -> PyResult<Vec<u8>> {
    let added_heap_bytes = total_heap_bytes - heap_start_offset_in_heap;
    let mut heap_buf: Vec<u8> = vec![0u8; added_heap_bytes];
    for (col_idx, col) in columns.iter().enumerate() {
        let Some(vci) = &vla[col_idx] else { continue; };
        let is_x = col.tform_letter == 'X';
        let elem_size = if is_x {
            0
        } else {
            bytes_per_element(col.tform_letter).unwrap_or(0)
        };
        for input_row in 0..input_nrows {
            let plan = vci.plans[input_row];
            if plan.nelements == 0 { continue; }
            let cell = vci.per_col_array.get_item(input_row)?;
            let local_off =
                plan.bytes_offset_in_heap - heap_start_offset_in_heap;
            let n_bytes = if is_x {
                plan.nelements.div_ceil(8)
            } else {
                plan.nelements * elem_size
            };
            let dst = &mut heap_buf[local_off..local_off + n_bytes];
            serialize_vla_cell(&cell, col.tform_letter, plan.nelements, dst)?;
        }
    }
    Ok(heap_buf)
}

// Write the in-memory heap buffer at its absolute position, then
// flush.  Caller passes the already-locked file ref so this runs
// under the same lock as the main-row writes that precede it.
// Mid-write failures taint per the standard discipline.
fn write_heap_and_flush(
    f: &mut std::fs::File,
    heap_buf: &[u8],
    heap_start_offset_in_file: u64,
    tainted: &crate::common::TaintFlag,
) -> PyResult<()> {
    if !heap_buf.is_empty() {
        f.seek(SeekFrom::Start(heap_start_offset_in_file))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(heap_buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "write error during VLA heap write: {}", e)));
        }
    }
    if let Err(e) = f.flush() {
        tainted.store(true, Ordering::Release);
        return Err(PyIOError::new_err(format!(
            "flush error during VLA write: {}", e)));
    }
    Ok(())
}

pub(crate) fn write_vla_data_range(
    columns: &[Column],
    fixed: &[Option<FixedColInfo>],
    vla: &[Option<VlaColInfo<'_>>],
    total_heap_bytes: usize,
    heap_start_offset_in_heap: usize,
    file: &FileHandle,
    main_start_offset: u64,
    heap_start_offset_in_file: u64,
    input_nrows: usize,
    row_width: usize,
    tainted: &crate::common::TaintFlag,
) -> PyResult<usize> {
    if input_nrows == 0 {
        return Ok(0);
    }
    let heap_buf = build_vla_heap_buf(
        columns, vla, total_heap_bytes,
        heap_start_offset_in_heap, input_nrows,
    )?;
    let added_heap_bytes = heap_buf.len();

    // Main data strip writer.  Same strip sizing as the fixed path;
    // each row is built one at a time via fill_main_row (which mixes
    // fixed-column transforms with VLA descriptor writes).
    let strip_target_bytes: usize = 1 << 20;
    let strip_nrows = (strip_target_bytes / row_width.max(1))
        .max(1).min(input_nrows);
    let mut strip_buf: Vec<u8> = vec![0u8; strip_nrows * row_width];

    let mut guard = lock_file(file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    f.seek(SeekFrom::Start(main_start_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    let mut row_start = 0usize;
    while row_start < input_nrows {
        let chunk = (input_nrows - row_start).min(strip_nrows);
        let want = chunk * row_width;
        if want < strip_buf.len() {
            strip_buf.truncate(want);
        }
        for b in strip_buf.iter_mut() { *b = 0; }
        for r in 0..chunk {
            let off = r * row_width;
            fill_main_row(
                columns, fixed, vla, row_start + r,
                &mut strip_buf[off..off + row_width])?;
        }
        if let Err(e) = f.write_all(&strip_buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "write error during VLA main-data write: {}", e)));
        }
        row_start += chunk;
    }

    write_heap_and_flush(f, &heap_buf, heap_start_offset_in_file, tainted)?;
    Ok(added_heap_bytes)
}

// Strided / fancy-row variant of write_vla_data_range: instead of
// writing `input_nrows` CONTIGUOUS main rows starting at one
// offset, walk a flat list of disk-row indices and seek+write each
// row's main bytes individually.  The heap pass is the same (one
// bulk write at the heap end) because the heap layout is per-input-
// row regardless of where each row's main bytes land on disk.
//
// Used by setitem_row_slice_vla_aware (step != 1) and
// setitem_fancy_rows_vla_aware.  Caller's `disk_rows.len()` must
// equal `input_nrows` (each input row maps to one disk row).
//
// Per-row seek+write cost: O(input_nrows) syscalls.  Acceptable
// because strided/fancy VLA writes are uncommon and each row's
// main bytes are typically a few tens of bytes.  If a hot-path
// workload demands it, a "bucket-by-tile + write-contiguous-
// strips" optimization is possible but adds complexity for
// negligible win on typical tables.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_vla_data_strided(
    columns: &[Column],
    fixed: &[Option<FixedColInfo>],
    vla: &[Option<VlaColInfo<'_>>],
    total_heap_bytes: usize,
    heap_start_offset_in_heap: usize,
    file: &FileHandle,
    data_offset: u64,
    heap_start_offset_in_file: u64,
    disk_rows: &[usize],
    row_width: usize,
    tainted: &crate::common::TaintFlag,
) -> PyResult<usize> {
    let input_nrows = disk_rows.len();
    if input_nrows == 0 {
        return Ok(0);
    }
    let heap_buf = build_vla_heap_buf(
        columns, vla, total_heap_bytes,
        heap_start_offset_in_heap, input_nrows,
    )?;
    let added_heap_bytes = heap_buf.len();

    // Per-row build + write.  No strip buffer (rows are non-
    // contiguous on disk); one row_width buffer reused per row.
    let mut row_buf: Vec<u8> = vec![0u8; row_width];
    let mut guard = lock_file(file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    for (input_row, &disk_row) in disk_rows.iter().enumerate() {
        for b in row_buf.iter_mut() { *b = 0; }
        fill_main_row(columns, fixed, vla, input_row, &mut row_buf)?;
        let off = data_offset + (disk_row as u64) * row_width as u64;
        f.seek(SeekFrom::Start(off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&row_buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "write error during VLA strided/fancy row write: {}", e)));
        }
    }

    write_heap_and_flush(f, &heap_buf, heap_start_offset_in_file, tainted)?;
    Ok(added_heap_bytes)
}

// Bulk write path for tables with at least one VLA (P/Q) column.
// Validates fixed + VLA columns, plans the heap layout from scratch
// (a full overwrite resets the heap to start at offset 0), grows the
// data section if needed, writes main rows + heap, then updates
// PCOUNT in the header.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_vla_aware(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
    nrows: usize,
    row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let per_col = extract_per_column_inputs(
        py, data, names, columns)?;
    // Per-column input length check: each input must have nrows rows.
    for (col_idx, col) in columns.iter().enumerate() {
        let shape: Vec<usize> =
            per_col[col_idx].getattr("shape")?.extract()?;
        if shape.first().copied().unwrap_or(0) != nrows {
            return Err(PyValueError::new_err(format!(
                "column '{}': input has {} rows but table NAXIS2={}",
                col.name, shape.first().copied().unwrap_or(0),
                nrows)));
        }
    }
    let mut fixed: Vec<Option<FixedColInfo>> =
        columns.iter().map(|_| None).collect();
    for (col_idx, col) in columns.iter().enumerate() {
        if col.var_kind.is_none() {
            fixed[col_idx] = Some(build_fixed_col_info(
                &per_col[col_idx], &ndarray, col, nrows)?);
        }
    }
    let (plans, total_heap_bytes) = plan_vla_heap_layout(
        columns, &per_col, nrows, &ndarray, 0)?;
    let vla: Vec<Option<VlaColInfo>> = columns.iter().enumerate()
        .map(|(col_idx, col)| {
            if col.var_kind.is_some() {
                Some(VlaColInfo {
                    plans: plans[col_idx].clone(),
                    per_col_array: per_col[col_idx].clone(),
                })
            } else {
                None
            }
        }).collect();

    let current_pcount = parse_keyword(cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    let main_bytes = (nrows * row_width) as u64;
    let current_data_bytes = main_bytes + current_pcount;
    let new_data_bytes = main_bytes + total_heap_bytes as u64;
    let current_padded = round_up_to_block(current_data_bytes);
    let new_padded = round_up_to_block(new_data_bytes);
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;

    if new_hdu_end > current_hdu_end {
        let delta = new_hdu_end - current_hdu_end;
        let file_len = {
            let g = lock_file(&super_.file)?;
            let f = g.as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            shift_file_tail_and_update_offsets(
                &super_.file, &super_.layout,
                current_hdu_end, delta, &super_.tainted)?;
            zero_fill_range(
                &super_.file, current_hdu_end, delta,
                &super_.tainted)?;
        } else {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    let heap_offset_in_file = data_offset + main_bytes;
    write_vla_data_range(
        columns, &fixed, &vla, total_heap_bytes, 0,
        &super_.file, data_offset, heap_offset_in_file,
        nrows, row_width, &super_.tainted)?;

    // PCOUNT update — disk-write-before-commit ordering.  Also
    // refresh the TFORMn `(maxlen)` hint for any PX/QX columns so
    // astropy's strict TFORM parser will accept the file (other
    // VLA letters round-trip fine without the hint).
    let mut cards_guard = super_.header.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards_guard.clone();
    set_pcount_in_cards(&mut new_cards, total_heap_bytes as u64);
    for (col_idx, col) in columns.iter().enumerate() {
        if col.tform_letter != 'X' { continue; }
        let Some(desc) = col.var_kind else { continue; };
        let Some(vci) = &vla[col_idx] else { continue; };
        let max_bits = vci.plans.iter()
            .map(|p| p.nelements).max().unwrap_or(0);
        super::write_fixed::set_x_vla_tform_maxlen_in_cards(
            &mut new_cards, col_idx + 1, desc, max_bits,
        );
    }
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = data_offset - header_bytes.len() as u64;
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "PCOUNT header write failed: {}; close + reopen", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "PCOUNT header flush failed: {}; close + reopen", e))
        })?;
    }
    *cards_guard = new_cards;
    Ok(())
}

// VLA-aware append path.  Mirrors the fixed append flow but
// additionally:
//   - Plans the heap layout starting at the current PCOUNT (so
//     descriptors for new rows point to offsets after the existing
//     heap).
//   - Relocates the existing heap forward (within the data section)
//     to sit after the appended main rows.
//   - Updates PCOUNT alongside NAXIS2 in the header rewrite.
// Reads the old heap into memory once, before any byte movement,
// and writes it back to its new position after the new main rows
// are in place.  For very large heaps this could chunk; MVP is
// in-RAM for clarity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_vla_aware(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
    current_nrows: usize,
    append_nrows: usize,
    new_nrows: usize,
    row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let per_col = extract_per_column_inputs(py, data, names, columns)?;
    // Per-column length sanity (each input must have append_nrows rows).
    for (col_idx, col) in columns.iter().enumerate() {
        let shape: Vec<usize> =
            per_col[col_idx].getattr("shape")?.extract()?;
        if shape.first().copied().unwrap_or(0) != append_nrows {
            return Err(PyValueError::new_err(format!(
                "column '{}': input has {} rows but append_nrows={}",
                col.name, shape.first().copied().unwrap_or(0),
                append_nrows)));
        }
    }
    let mut fixed: Vec<Option<FixedColInfo>> =
        columns.iter().map(|_| None).collect();
    for (col_idx, col) in columns.iter().enumerate() {
        if col.var_kind.is_none() {
            fixed[col_idx] = Some(build_fixed_col_info(
                &per_col[col_idx], &ndarray, col, append_nrows)?);
        }
    }
    let current_pcount = parse_keyword(cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    let (plans, total_heap_bytes_after) = plan_vla_heap_layout(
        columns, &per_col, append_nrows, &ndarray,
        current_pcount as usize)?;
    let new_pcount = total_heap_bytes_after as u64;
    let vla: Vec<Option<VlaColInfo>> = columns.iter().enumerate()
        .map(|(col_idx, col)| {
            if col.var_kind.is_some() {
                Some(VlaColInfo {
                    plans: plans[col_idx].clone(),
                    per_col_array: per_col[col_idx].clone(),
                })
            } else {
                None
            }
        }).collect();

    let old_main_bytes = (current_nrows * row_width) as u64;
    let new_main_bytes = (new_nrows * row_width) as u64;
    let current_data_bytes = old_main_bytes + current_pcount;
    let new_data_bytes = new_main_bytes + new_pcount;
    let current_padded = round_up_to_block(current_data_bytes);
    let new_padded = round_up_to_block(new_data_bytes);
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;

    // Read OLD heap before any byte movement: the upcoming new-main
    // write may overwrite part of the old heap's region in place.
    let old_heap_bytes: Vec<u8> = if current_pcount > 0 {
        let mut buf = vec![0u8; current_pcount as usize];
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset + old_main_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut buf)
            .map_err(|e| PyIOError::new_err(format!(
                "read error capturing old heap during append: {}", e)))?;
        buf
    } else {
        Vec::new()
    };

    // Grow data section if needed.  Last-HDU branch uses set_len;
    // non-last branch shifts and zero-fills the gap (mirrors the
    // fixed-table append path).
    if new_hdu_end > current_hdu_end {
        let delta = new_hdu_end - current_hdu_end;
        let file_len = {
            let g = lock_file(&super_.file)?;
            let f = g.as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            shift_file_tail_and_update_offsets(
                &super_.file, &super_.layout,
                current_hdu_end, delta, &super_.tainted)?;
            zero_fill_range(
                &super_.file, current_hdu_end, delta, &super_.tainted)?;
        } else {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // Write the appended main rows + new heap bytes.  Heap goes at
    // the NEW heap position, AFTER where the relocated old heap will
    // sit (descriptors already encode offsets >= current_pcount).
    write_vla_data_range(
        columns, &fixed, &vla, total_heap_bytes_after,
        current_pcount as usize,
        &super_.file,
        data_offset + old_main_bytes,
        data_offset + new_main_bytes + current_pcount,
        append_nrows, row_width, &super_.tainted)?;

    // Relocate the captured old heap bytes into their new slot
    // between the new main rows and the new heap content.
    if !old_heap_bytes.is_empty() {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset + new_main_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&old_heap_bytes) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "write error during old-heap relocation: {}", e)));
        }
        if let Err(e) = f.flush() {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "flush error during old-heap relocation: {}", e)));
        }
    }

    // Update NAXIS2 + PCOUNT cards (disk-write-before-commit).
    let mut cards_guard = super_.header.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards_guard.clone();
    let naxis2_card = card_int(
        "NAXIS2", new_nrows as i64, "number of rows in table");
    let naxis2_idx = new_cards.iter()
        .position(|c| c.len() >= 6 && c[..6].trim() == "NAXIS2")
        .ok_or_else(|| PyValueError::new_err("header missing NAXIS2"))?;
    new_cards[naxis2_idx] = naxis2_card.trim_end().to_string();
    set_pcount_in_cards(&mut new_cards, new_pcount);
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = data_offset - header_bytes.len() as u64;
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "header write failed during VLA append: {}", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "header flush failed during VLA append: {}", e))
        })?;
    }
    *cards_guard = new_cards;
    Ok(())
}

// Rebuild the heap with only the bytes that live descriptors point at,
// dropping orphan bytes left behind by VLA __setitem__.  Reads main
// rows + old heap into RAM, builds the compact new heap, writes them
// back, and shrinks the file if the new padded extent is smaller than
// the old.
//
// Validate-then-mutate: pre-write failures (THEAP nonstandard, lock,
// metadata) don't taint.  Once any byte movement starts, failures
// taint.  No-op for non-VLA tables and for already-compact heaps.
pub(crate) fn repack_table_heap(super_: &HDU) -> PyResult<()> {
    check_not_tainted(&super_.tainted)?;
    let cards = super_.header_snapshot()?;
    let columns = parse_columns(&cards)?;
    if !any_var_column(&columns) {
        return Ok(());
    }
    let nrows = parse_keyword(&cards, "NAXIS2")
        .unwrap_or(0).max(0) as usize;
    let row_width = parse_keyword(&cards, "NAXIS1")
        .unwrap_or(0).max(0) as usize;
    let current_pcount = parse_keyword(&cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    let data_offset = super_.offsets.data_offset();
    if current_pcount == 0 || nrows == 0 {
        return Ok(());
    }

    // Reject non-default THEAP — the read side would point at a heap
    // start that differs from where repack writes the new heap.  Files
    // rustfits creates never set THEAP, so this only blocks repack on
    // files written by other tools with a non-default layout.
    let theap_raw = parse_keyword(&cards, "THEAP").unwrap_or(0);
    let main_bytes = (nrows as u64).saturating_mul(row_width as u64);
    if theap_raw > 0 && (theap_raw as u64) != main_bytes {
        return Err(PyValueError::new_err(format!(
            "repack: file has non-default THEAP={} (main rows end at \
             {}); repack would write the new heap at the default \
             position and corrupt the file.  Workaround: rewrite the \
             file through a fresh create_table_hdu + write",
            theap_raw, main_bytes)));
    }

    // Read the main table + old heap into RAM under a single file lock.
    let mut main_buf = vec![0u8; nrows * row_width];
    let mut old_heap = vec![0u8; current_pcount as usize];
    let heap_base = heap_base_in_data(&cards);
    let old_heap_off = data_offset + heap_base;
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut main_buf)
            .map_err(|e| PyIOError::new_err(format!(
                "repack: read main table failed: {}", e)))?;
        f.seek(SeekFrom::Start(old_heap_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut old_heap)
            .map_err(|e| PyIOError::new_err(format!(
                "repack: read heap failed: {}", e)))?;
    }

    // Walk rows × VLA columns, copy live cells from old_heap into a
    // new compact buffer, and rewrite each descriptor in main_buf to
    // point at its new location.
    let mut new_heap: Vec<u8> = Vec::new();
    for r in 0..nrows {
        let row_off = r * row_width;
        for col in &columns {
            let Some(descriptor_kind) = col.var_kind else { continue; };
            let desc_off = row_off + col.byte_offset;
            let desc = &main_buf[desc_off..desc_off + col.byte_width];
            let (nelements_s, old_off_s) =
                read_descriptor(descriptor_kind, desc);
            // Negative descriptor values indicate a bad file; reject
            // up front rather than passing them through.
            if nelements_s < 0 || old_off_s < 0 {
                return Err(PyValueError::new_err(format!(
                    "repack: column '{}' row {}: descriptor has \
                     negative field (nelements={}, offset={})",
                    col.name, r, nelements_s, old_off_s)));
            }
            let nelements = nelements_s as u64;
            let old_off = old_off_s as u64;
            // X (bit-packed) VLA: nelements is the bit count;
            // heap bytes per cell = ceil(nelements/8).  Other
            // letters use a fixed element width.
            let n_bytes = if col.tform_letter == 'X' {
                nelements.div_ceil(8)
            } else {
                let elem_size = bytes_per_element(col.tform_letter)
                    .unwrap_or(0) as u64;
                nelements * elem_size
            };
            if old_off + n_bytes > current_pcount {
                return Err(PyValueError::new_err(format!(
                    "repack: column '{}' row {}: descriptor points \
                     past heap end (offset+bytes={} > PCOUNT={})",
                    col.name, r, old_off + n_bytes, current_pcount)));
            }
            let new_off = new_heap.len() as u64;
            if n_bytes > 0 {
                new_heap.extend_from_slice(
                    &old_heap[old_off as usize
                        ..(old_off + n_bytes) as usize]);
            }
            let dst = &mut main_buf[desc_off..desc_off + col.byte_width];
            write_descriptor(
                descriptor_kind, nelements as usize, new_off as usize, dst);
        }
    }
    drop(old_heap);
    let new_pcount = new_heap.len() as u64;
    if new_pcount == current_pcount {
        // Already compact; nothing to do.
        return Ok(());
    }

    let current_data_bytes = main_bytes + current_pcount;
    let new_data_bytes = main_bytes + new_pcount;
    let current_padded = round_up_to_block(current_data_bytes);
    let new_padded = round_up_to_block(new_data_bytes);
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;

    // Write the rebuilt main table + new heap (the new heap sits
    // immediately after main; if the old heap was at default position
    // we may be partially overwriting old-heap bytes, which is fine
    // since they're in RAM as old_heap was already dropped above —
    // sorry, the read snapshot in main_buf+new_heap is what we write
    // out).  Pad the heap region within the current padded extent so
    // any trailing bytes between new heap end and current end are
    // zeroed (they won't be reachable from any descriptor regardless).
    let heap_off_in_file = data_offset + main_bytes;
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&main_buf) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: write main failed: {}; close + reopen", e)));
        }
        f.seek(SeekFrom::Start(heap_off_in_file))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&new_heap) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: write heap failed: {}; close + reopen", e)));
        }
        if let Err(e) = f.flush() {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: flush failed: {}; close + reopen", e)));
        }
    }

    // Shrink the file extent.  For non-last HDUs, shift the tail
    // backward to fill the gap and bump every later HDU's offset down;
    // for the last HDU, a plain set_len reclaims the trailing block(s).
    if new_hdu_end < current_hdu_end {
        let delta = current_hdu_end - new_hdu_end;
        let file_len = {
            let g = lock_file(&super_.file)?;
            let f = g.as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            shift_file_tail_backward_and_update_offsets(
                &super_.file, &super_.layout,
                current_hdu_end, delta, &super_.tainted)?;
        } else {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "repack: set_len failed: {}; close + reopen", e))
            })?;
        }
    }

    // PCOUNT update — disk-write-before-commit ordering.
    let mut cards_guard = super_.header.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards_guard.clone();
    set_pcount_in_cards(&mut new_cards, new_pcount);
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = data_offset - header_bytes.len() as u64;
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack: PCOUNT header write failed: {}; close + reopen",
                e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack: PCOUNT header flush failed: {}; close + reopen",
                e))
        })?;
    }
    *cards_guard = new_cards;
    Ok(())
}
