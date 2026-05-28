// Fixed-column write paths: input preparation, the strip writer,
// strided/per-column writers, write_fixed_only + append_fixed_only,
// PCOUNT card patcher, dispatch_write_input + determine_input_nrows.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::Ordering;

use crate::common::{
    lock_file, shift_file_tail_and_update_offsets, zero_fill_range,
    FileHandle, RawBuffer, TaintFlag,
};
use crate::hdu::HDU;
use crate::hdu_image::{round_up_to_block, serialize_header_to_disk_bytes};
use crate::header::card_int;

use super::columns::Column;
use super::write_setup::{
    column_expected_shape, column_transform, WriteTransform,
};

// Per-column source view for the strip writer.
//
// For structured-array input, all columns share the same `src_bytes`
// (the array's raw buffer) and `src_row_stride` (the array's itemsize),
// with each column's `src_offset` set to its field offset within a row.
//
// For dict / list+names input, each column has its OWN `src_bytes`
// (its own ndarray's raw buffer), `src_offset` is 0, and
// `src_row_stride == src_total_size` (the per-cell byte count).
pub(crate) struct ColumnSource<'a> {
    pub(crate) src_bytes: &'a [u8],
    pub(crate) src_offset: usize,
    pub(crate) src_row_stride: usize,
    pub(crate) src_total_size: usize,
}

// Per-column source metadata that doesn't carry the borrowed bytes,
// so the per-input-form preparation functions can build it before the
// final Vec<ColumnSource> is assembled.  buffer_idx indexes into a
// per-call Vec<RawBuffer> owned by the write pymethod.
pub(crate) struct ColumnSourceMeta {
    pub(crate) buffer_idx: usize,
    pub(crate) src_offset: usize,
    pub(crate) src_row_stride: usize,
    pub(crate) src_total_size: usize,
}

// Result of input preparation, common across all input forms.
pub(crate) struct PreparedInput {
    pub(crate) transforms: Vec<WriteTransform>,
    pub(crate) metas: Vec<ColumnSourceMeta>,
    // True iff the fast-path bulk-memcpy is safe: all sources share
    // the same buffer, each src_offset == col.byte_offset, each
    // src_total_size == col.byte_width, and src_row_stride == row_width.
    pub(crate) layout_matches: bool,
}

// Validate one input field's shape + dtype against the HDU column and
// return the per-cell WriteTransform.  Shared by all input forms.
// `field_dtype` may be a subarray dtype (carrying numpy shape) for
// structured-array input, or a synthetic per-cell dtype derived from
// a per-column ndarray's shape for dict/list input.
fn validate_field_for_column(
    col: &Column,
    field_dtype: &Bound<'_, PyAny>,
) -> PyResult<WriteTransform> {
    let input_shape: Vec<usize> = field_dtype.getattr("shape")?.extract()?;
    let expected_shape = column_expected_shape(col);
    if input_shape != expected_shape {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: column '{}' per-cell shape {:?} does not \
             match table column expected shape {:?}",
            col.name, input_shape, expected_shape)));
    }
    let base = field_dtype.getattr("base")?;
    let input_kind: String = base.getattr("kind")?.extract()?;
    let input_elem_size: usize = base.getattr("itemsize")?.extract()?;
    column_transform(col, &input_kind, input_elem_size)
}

// Structured ndarray input.  Allows field-order normalization: the
// HDU is the authoritative ordering, and the input dtype just needs
// to contain a field for every HDU column (extras, missing, or
// duplicates are rejected).  layout_matches is true iff (a) names
// are in HDU order with no reordering, (b) per-field offsets and
// widths match the FITS row layout, (c) input itemsize == row_width.
pub(crate) fn prepare_structured_input(
    data: &Bound<'_, PyAny>,
    columns: &[Column],
    nrows: usize,
    row_width: usize,
    buffers: &mut Vec<RawBuffer>,
) -> PyResult<PreparedInput> {
    let dtype = data.getattr("dtype")?;
    let names_attr = dtype.getattr("names")?;
    if names_attr.is_none() {
        return Err(PyValueError::new_err(
            "TableHDU.write: structured input must have named fields"));
    }
    let input_names: Vec<String> = names_attr.extract()?;
    if input_names.len() != columns.len() {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: input has {} columns, table has {}",
            input_names.len(), columns.len())));
    }
    // Build a name -> input-index map for cross-check, also catches
    // duplicate field names in the input dtype.
    let mut name_seen = std::collections::HashSet::with_capacity(
        input_names.len());
    for n in &input_names {
        if !name_seen.insert(n.clone()) {
            return Err(PyValueError::new_err(format!(
                "TableHDU.write: input dtype has duplicate field name '{}'", n)));
        }
    }
    // Every HDU column must be present by exact name.
    for col in columns {
        if !name_seen.contains(&col.name) {
            return Err(PyValueError::new_err(format!(
                "TableHDU.write: input dtype is missing field '{}' \
                 (table column)", col.name)));
        }
    }
    let data_len: usize = data.len()?;
    if data_len != nrows {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: input has {} rows but table NAXIS2={}",
            data_len, nrows)));
    }
    let flags = data.getattr("flags")?;
    let c_contig: bool = flags.getattr("c_contiguous")?.extract()?;
    if !c_contig {
        return Err(PyValueError::new_err(
            "TableHDU.write: input ndarray must be C-contiguous"));
    }
    let input_itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    let buf = RawBuffer::acquire(data)?;
    let expected_bytes = data_len.checked_mul(input_itemsize)
        .ok_or_else(|| PyValueError::new_err("input size overflow"))?;
    if buf.as_slice().len() < expected_bytes {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: source buffer length {} smaller than \
             expected {}", buf.as_slice().len(), expected_bytes)));
    }
    let buffer_idx = buffers.len();
    buffers.push(buf);

    // Walk HDU columns in order; for each, look up the input field
    // (which may be at a different position in the input dtype).
    let fields = dtype.getattr("fields")?;
    let mut transforms = Vec::with_capacity(columns.len());
    let mut metas = Vec::with_capacity(columns.len());
    let mut layout_matches = input_itemsize == row_width;
    for (i, col) in columns.iter().enumerate() {
        let entry = fields.get_item(col.name.as_str())?;
        let entry_tup = entry.cast::<PyTuple>()?;
        let field_dtype = entry_tup.get_item(0)?;
        let src_offset: usize = entry_tup.get_item(1)?.extract()?;
        let src_total_size: usize =
            field_dtype.getattr("itemsize")?.extract()?;
        transforms.push(validate_field_for_column(col, &field_dtype)?);
        metas.push(ColumnSourceMeta {
            buffer_idx,
            src_offset,
            src_row_stride: input_itemsize,
            src_total_size,
        });
        // Order check: input dtype's field at position i must be col.
        let input_name_at_i = &input_names[i];
        if input_name_at_i != &col.name {
            layout_matches = false;
        }
        if src_offset != col.byte_offset
            || src_total_size != col.byte_width
        {
            layout_matches = false;
        }
        // X (bit-packed) columns always route through the slow path:
        // src width = num_bits bytes (one per bool), dst width =
        // ceil(num_bits/8) bytes.  Even when those happen to match
        // (1-bit scalar), the bulk-memcpy fast path doesn't know how
        // to bit-pack — only the per-cell apply_transform_cell does.
        if col.tform_letter == 'X' {
            layout_matches = false;
        }
    }
    Ok(PreparedInput { transforms, metas, layout_matches })
}

// Dict input: keys are column names, values are per-column ndarrays.
// Each ndarray contributes its own buffer; layout_matches is always
// false (per-column buffers cannot share a contiguous strip).
fn prepare_dict_input(
    py: Python<'_>,
    data: &Bound<'_, PyDict>,
    columns: &[Column],
    nrows: usize,
    buffers: &mut Vec<RawBuffer>,
) -> PyResult<PreparedInput> {
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let hdu_names: std::collections::HashSet<&str> =
        columns.iter().map(|c| c.name.as_str()).collect();
    // Reject extras up front.
    for key_obj in data.keys() {
        let key: String = key_obj.extract().map_err(|_| {
            PyValueError::new_err("TableHDU.write: dict keys must be strings")
        })?;
        if !hdu_names.contains(key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "TableHDU.write: dict has extra key '{}' not in table \
                 columns", key)));
        }
    }
    let mut transforms = Vec::with_capacity(columns.len());
    let mut metas = Vec::with_capacity(columns.len());
    for col in columns {
        let val = data.get_item(col.name.as_str())?
            .ok_or_else(|| PyValueError::new_err(format!(
                "TableHDU.write: dict is missing column '{}'", col.name)))?;
        let (transform, src_total_size, buffer_idx) =
            acquire_per_column_array(&val, &ndarray, col, nrows, buffers)?;
        transforms.push(transform);
        metas.push(ColumnSourceMeta {
            buffer_idx,
            src_offset: 0,
            src_row_stride: src_total_size,
            src_total_size,
        });
    }
    Ok(PreparedInput { transforms, metas, layout_matches: false })
}

// List+names input: parallel sequences of arrays and column names.
// Same per-column model as dict; just a different surface API.
fn prepare_list_names_input(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    names_obj: &Bound<'_, PyAny>,
    columns: &[Column],
    nrows: usize,
    buffers: &mut Vec<RawBuffer>,
) -> PyResult<PreparedInput> {
    let arrays: Vec<Bound<'_, PyAny>> = data.try_iter()?
        .collect::<PyResult<Vec<_>>>()?;
    let names: Vec<String> = names_obj.extract().map_err(|_| {
        PyValueError::new_err(
            "TableHDU.write: names= must be a sequence of strings")
    })?;
    if arrays.len() != names.len() {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: len(data)={} != len(names)={}",
            arrays.len(), names.len())));
    }
    let hdu_names: std::collections::HashSet<&str> =
        columns.iter().map(|c| c.name.as_str()).collect();
    let mut name_to_arr: std::collections::HashMap<String, &Bound<'_, PyAny>> =
        std::collections::HashMap::with_capacity(names.len());
    for (n, a) in names.iter().zip(arrays.iter()) {
        if !hdu_names.contains(n.as_str()) {
            return Err(PyValueError::new_err(format!(
                "TableHDU.write: names list has extra entry '{}' not in \
                 table columns", n)));
        }
        if name_to_arr.insert(n.clone(), a).is_some() {
            return Err(PyValueError::new_err(format!(
                "TableHDU.write: duplicate name '{}' in names list", n)));
        }
    }
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let mut transforms = Vec::with_capacity(columns.len());
    let mut metas = Vec::with_capacity(columns.len());
    for col in columns {
        let arr = name_to_arr.get(col.name.as_str())
            .ok_or_else(|| PyValueError::new_err(format!(
                "TableHDU.write: column '{}' is missing from names list",
                col.name)))?;
        let (transform, src_total_size, buffer_idx) =
            acquire_per_column_array(arr, &ndarray, col, nrows, buffers)?;
        transforms.push(transform);
        metas.push(ColumnSourceMeta {
            buffer_idx,
            src_offset: 0,
            src_row_stride: src_total_size,
            src_total_size,
        });
    }
    Ok(PreparedInput { transforms, metas, layout_matches: false })
}

// Per-column ndarray validation + buffer acquisition for dict/list
// inputs.  arr.shape[0] must equal nrows; arr.shape[1:] is the
// per-cell numpy shape and must match the column's expected shape.
// Returns (transform, src_total_size, buffer_idx).
pub(crate) fn acquire_per_column_array(
    arr: &Bound<'_, PyAny>,
    ndarray: &Bound<'_, PyAny>,
    col: &Column,
    nrows: usize,
    buffers: &mut Vec<RawBuffer>,
) -> PyResult<(WriteTransform, usize, usize)> {
    if !arr.is_instance(ndarray)? {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: column '{}': value must be a numpy ndarray",
            col.name)));
    }
    let arr_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if arr_shape.is_empty() || arr_shape[0] != nrows {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: column '{}': array shape {:?} does not have \
             first axis == nrows ({})", col.name, arr_shape, nrows)));
    }
    let per_cell_shape: Vec<usize> = arr_shape[1..].to_vec();
    let expected_shape = column_expected_shape(col);
    if per_cell_shape != expected_shape {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: column '{}': per-cell shape {:?} does not \
             match table column expected shape {:?}",
            col.name, per_cell_shape, expected_shape)));
    }
    let arr_dtype = arr.getattr("dtype")?;
    let kind: String = arr_dtype.getattr("kind")?.extract()?;
    let elem_size: usize = arr_dtype.getattr("itemsize")?.extract()?;
    let transform = column_transform(col, &kind, elem_size)?;
    let cell_elements: usize =
        per_cell_shape.iter().product::<usize>().max(1);
    let src_total_size = elem_size * cell_elements;
    let flags = arr.getattr("flags")?;
    let c_contig: bool = flags.getattr("c_contiguous")?.extract()?;
    if !c_contig {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: column '{}': ndarray must be C-contiguous",
            col.name)));
    }
    let buf = RawBuffer::acquire(arr)?;
    let expected_bytes = nrows.checked_mul(src_total_size)
        .ok_or_else(|| PyValueError::new_err("input size overflow"))?;
    if buf.as_slice().len() < expected_bytes {
        return Err(PyValueError::new_err(format!(
            "TableHDU.write: column '{}': buffer length {} smaller than \
             expected {}", col.name, buf.as_slice().len(), expected_bytes)));
    }
    let buffer_idx = buffers.len();
    buffers.push(buf);
    Ok((transform, src_total_size, buffer_idx))
}

// Apply one cell-worth of a WriteTransform from `src` to `dst`.
// `src` and `dst` may differ in length only for UnicodeToAscii
// (src.len() == 4 × dst.len()); for all other variants the lengths
// are equal.  Used by the slow path; the fast path applies the same
// transforms in place on a pre-bulk-copied strip buffer.
pub(crate) fn apply_transform_cell(
    transform: &WriteTransform,
    src: &[u8],
    dst: &mut [u8],
    col_name: &str,
    row_in_strip: usize,
) -> PyResult<()> {
    match *transform {
        WriteTransform::Identity { elem_w, num_elems } => {
            if elem_w == 1 {
                dst.copy_from_slice(src);
            } else {
                for e in 0..num_elems {
                    let s = &src[e * elem_w..(e + 1) * elem_w];
                    let d = &mut dst[e * elem_w..(e + 1) * elem_w];
                    for k in 0..elem_w {
                        d[k] = s[elem_w - 1 - k];
                    }
                }
            }
        }
        WriteTransform::UnsignedXor { elem_w, num_elems } => {
            for e in 0..num_elems {
                let s = &src[e * elem_w..(e + 1) * elem_w];
                let d = &mut dst[e * elem_w..(e + 1) * elem_w];
                for k in 0..elem_w {
                    d[k] = s[elem_w - 1 - k];
                }
                d[0] ^= 0x80;
            }
        }
        WriteTransform::BoolToLogical { num_bytes } => {
            for i in 0..num_bytes {
                dst[i] = if src[i] == 0 { b'F' } else { b'T' };
            }
        }
        WriteTransform::BytesCopy { num_bytes } => {
            dst[..num_bytes].copy_from_slice(&src[..num_bytes]);
        }
        WriteTransform::UnicodeToAscii { num_chars } => {
            for i in 0..num_chars {
                let cp_bytes: [u8; 4] =
                    src[i * 4..i * 4 + 4].try_into().unwrap();
                let cp = u32::from_le_bytes(cp_bytes);
                if cp > 0x7F {
                    return Err(PyValueError::new_err(format!(
                        "TableHDU.write: column '{}' row {} char {}: \
                         non-ASCII Unicode codepoint U+{:04X}; FITS A \
                         columns are restricted to 7-bit ASCII",
                        col_name, row_in_strip, i, cp)));
                }
                dst[i] = cp as u8;
            }
        }
        WriteTransform::BitsPackMsb { num_bits } => {
            // Pack `num_bits` source bytes (one per numpy bool) into
            // ceil(num_bits/8) destination bytes, MSB-first within
            // each byte.  Bit i of the cell goes to byte (i/8), bit
            // position (7 - i%8).  Trailing bits in the last byte
            // (when num_bits % 8 != 0) are left zero per the FITS
            // spec.  Inverse of read.rs::convert_x_cell.
            let n_bytes = num_bits.div_ceil(8);
            for b in dst[..n_bytes].iter_mut() {
                *b = 0;
            }
            for i in 0..num_bits {
                if src[i] != 0 {
                    dst[i / 8] |= 1u8 << (7 - (i % 8));
                }
            }
        }
    }
    Ok(())
}

// Apply the in-place transform variants to a strip buffer that has
// already been bulk-filled by a layout-matched memcpy.  Only Identity,
// UnsignedXor, and BoolToLogical can run in place — they preserve byte
// width.  BytesCopy is also in-place safe (it's a memcpy that
// happened to already happen via the bulk copy).  UnicodeToAscii is
// only valid on the slow path and never reaches this function.
fn apply_in_place_transform(
    strip_buf: &mut [u8],
    transform: &WriteTransform,
    col: &Column,
    chunk: usize,
    row_width: usize,
) {
    match *transform {
        WriteTransform::Identity { elem_w, num_elems } => {
            if elem_w == 1 { return; }
            for r in 0..chunk {
                let row_off = r * row_width + col.byte_offset;
                for e in 0..num_elems {
                    let beg = row_off + e * elem_w;
                    strip_buf[beg..beg + elem_w].reverse();
                }
            }
        }
        WriteTransform::UnsignedXor { elem_w, num_elems } => {
            for r in 0..chunk {
                let row_off = r * row_width + col.byte_offset;
                for e in 0..num_elems {
                    let beg = row_off + e * elem_w;
                    strip_buf[beg..beg + elem_w].reverse();
                    strip_buf[beg] ^= 0x80;
                }
            }
        }
        WriteTransform::BoolToLogical { num_bytes } => {
            for r in 0..chunk {
                let row_off = r * row_width + col.byte_offset;
                for b in 0..num_bytes {
                    let pos = row_off + b;
                    strip_buf[pos] =
                        if strip_buf[pos] == 0 { b'F' } else { b'T' };
                }
            }
        }
        WriteTransform::BytesCopy { .. } => {
            // No-op: the bulk copy already placed the bytes correctly.
        }
        WriteTransform::UnicodeToAscii { .. } => {
            unreachable!(
                "UnicodeToAscii in fast-path; validate should have routed \
                 through the slow path");
        }
        WriteTransform::BitsPackMsb { .. } => {
            unreachable!(
                "BitsPackMsb in fast-path; X columns always force \
                 layout_matches=false so the slow path runs");
        }
    }
}

// Strip-based bulk write into the table's data section.
//
// Two paths:
//   - FAST: layout_matches=true.  All ColumnSources share the same
//     buffer and offsets/widths exactly match the FITS row layout.
//     Per strip: one memcpy of `chunk * row_width` bytes from the
//     shared buffer into the strip buffer, then per-column in-place
//     transform.
//   - SLOW: layout_matches=false.  Each ColumnSource is read
//     independently using its own src_bytes + src_offset +
//     src_row_stride.  Used for U columns (which break the row
//     layout because numpy U is UTF-32-LE), for structured arrays
//     with reordered fields, and for dict / list+names input.
//     Strip is pre-zeroed so short strings end up null-padded.
//
// Peak memory ~1 MiB regardless of nrows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_table_data(
    columns: &[Column],
    transforms: &[WriteTransform],
    sources: &[ColumnSource<'_>],
    layout_matches: bool,
    file: &FileHandle,
    start_offset: u64,
    nrows: usize,
    row_width: usize,
    tainted: &TaintFlag,
) -> PyResult<()> {
    if nrows == 0 {
        return Ok(());
    }
    let strip_target_bytes: usize = 1 << 20;
    let strip_nrows = (strip_target_bytes / row_width.max(1)).max(1).min(nrows);
    let mut strip_buf: Vec<u8> = vec![0u8; strip_nrows * row_width];

    let mut guard = lock_file(file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    f.seek(SeekFrom::Start(start_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    let mut row_start = 0usize;
    while row_start < nrows {
        let chunk = (nrows - row_start).min(strip_nrows);
        let want = chunk * row_width;
        if want < strip_buf.len() {
            strip_buf.truncate(want);
        }

        if layout_matches {
            // Fast path: bulk copy strip bytes from the shared source
            // buffer (all sources point to it), then per-column
            // in-place transform.  sources[0] carries the shared
            // src_bytes + row stride; layout_matches=true guarantees
            // every other ColumnSource agrees.
            let shared = &sources[0];
            let src_start = row_start * shared.src_row_stride;
            strip_buf.copy_from_slice(
                &shared.src_bytes[src_start..src_start + want]);
            for (col, transform) in columns.iter().zip(transforms.iter()) {
                apply_in_place_transform(
                    &mut strip_buf, transform, col, chunk, row_width);
            }
        } else {
            // Slow path: zero-init the strip (so partial / short
            // fields end up null-padded), then per-column per-row
            // strided copy + transform from each column's own source.
            for b in strip_buf.iter_mut() { *b = 0; }
            for ((col, transform), source) in
                columns.iter().zip(transforms.iter()).zip(sources.iter())
            {
                for r in 0..chunk {
                    let src_off = (row_start + r) * source.src_row_stride
                        + source.src_offset;
                    let dst_off = r * row_width + col.byte_offset;
                    let src = &source.src_bytes
                        [src_off..src_off + source.src_total_size];
                    let dst = &mut strip_buf
                        [dst_off..dst_off + col.byte_width];
                    apply_transform_cell(transform, src, dst, &col.name, r)?;
                }
            }
        }

        if let Err(e) = f.write_all(&strip_buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "write error during table write: {}", e)));
        }
        row_start += chunk;
    }
    if let Err(e) = f.flush() {
        tainted.store(true, Ordering::Release);
        return Err(PyIOError::new_err(format!(
            "flush error during table write: {}", e)));
    }
    Ok(())
}

// Per-row write for strided slice assignment.  Each row is built from
// the per-strip machinery (fast-path bulk memcpy + in-place transform
// when layout matches, otherwise zero-pad + per-column strided copy)
// then written at a custom file offset.  No read-modify-write: every
// column is being overwritten so the prior on-disk bytes are discarded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_table_strided(
    columns: &[Column],
    transforms: &[WriteTransform],
    sources: &[ColumnSource<'_>],
    layout_matches: bool,
    file: &FileHandle,
    data_offset: u64,
    row_indices: &[i64],
    row_width: usize,
    tainted: &TaintFlag,
) -> PyResult<()> {
    if row_indices.is_empty() {
        return Ok(());
    }
    let mut row_buf: Vec<u8> = vec![0u8; row_width];
    let mut guard = lock_file(file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    for (input_row, &disk_row) in row_indices.iter().enumerate() {
        if layout_matches {
            let shared = &sources[0];
            let src_start = input_row * shared.src_row_stride;
            row_buf.copy_from_slice(
                &shared.src_bytes[src_start..src_start + row_width]);
            for (col, transform) in columns.iter().zip(transforms.iter()) {
                apply_in_place_transform(
                    &mut row_buf, transform, col, 1, row_width);
            }
        } else {
            for b in row_buf.iter_mut() { *b = 0; }
            for ((col, transform), source) in
                columns.iter().zip(transforms.iter()).zip(sources.iter())
            {
                let src_off = input_row * source.src_row_stride
                    + source.src_offset;
                let src = &source.src_bytes
                    [src_off..src_off + source.src_total_size];
                let dst = &mut row_buf
                    [col.byte_offset..col.byte_offset + col.byte_width];
                apply_transform_cell(transform, src, dst, &col.name, input_row)?;
            }
        }
        let file_off = data_offset
            + (disk_row as u64) * row_width as u64;
        f.seek(SeekFrom::Start(file_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&row_buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "write error during strided row write: {}", e)));
        }
    }
    if let Err(e) = f.flush() {
        tainted.store(true, Ordering::Release);
        return Err(PyIOError::new_err(format!(
            "flush error during strided row write: {}", e)));
    }
    Ok(())
}

// Whole-column write: per-row seek + write of just this column's
// byte_width bytes.  No read-modify-write — the other columns' bytes
// in each row are preserved by virtue of never being touched.  Cost
// is O(nrows) seek+write syscalls of byte_width each; this dominates
// over the alternative strip RMW (which would read/write ~2× the
// full table) whenever byte_width << row_width, which is the common
// case for "fix one column" assignments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_table_one_column(
    col: &Column,
    transform: &WriteTransform,
    source: &ColumnSource<'_>,
    file: &FileHandle,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    tainted: &TaintFlag,
) -> PyResult<()> {
    if nrows == 0 {
        return Ok(());
    }
    let mut cell_buf: Vec<u8> = vec![0u8; col.byte_width];
    let mut guard = lock_file(file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    for r in 0..nrows {
        for b in cell_buf.iter_mut() { *b = 0; }
        let src_off = r * source.src_row_stride + source.src_offset;
        let src = &source.src_bytes
            [src_off..src_off + source.src_total_size];
        apply_transform_cell(transform, src, &mut cell_buf, &col.name, r)?;
        let file_off = data_offset
            + (r * row_width + col.byte_offset) as u64;
        f.seek(SeekFrom::Start(file_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&cell_buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "write error during column write: {}", e)));
        }
    }
    if let Err(e) = f.flush() {
        tainted.store(true, Ordering::Release);
        return Err(PyIOError::new_err(format!(
            "flush error during column write: {}", e)));
    }
    Ok(())
}

// Bulk write path for tables with no VLA columns.  Validates input
// against the table schema, then dispatches to write_table_data,
// which writes contiguous main-section rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_fixed_only(
    py: Python<'_>,
    super_: &HDU,
    columns: &[Column],
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
    nrows: usize,
    row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    let mut buffers: Vec<RawBuffer> = Vec::new();
    let prep = dispatch_write_input(
        py, data, names, columns, nrows, row_width, &mut buffers)?;
    let sources = build_sources(&prep.metas, &buffers);
    write_table_data(
        columns, &prep.transforms, &sources, prep.layout_matches,
        &super_.file, data_offset, nrows, row_width, &super_.tainted,
    )
}

// Append rows to a table with no VLA columns.  Validates input, grows
// the data section if needed (last-HDU branch uses set_len; non-last
// branch shifts the file tail and zero-fills the gap), rewrites
// NAXIS2 on disk, then writes the appended rows at the end of the
// existing data section.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_fixed_only(
    py: Python<'_>,
    super_: &HDU,
    columns: &[Column],
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
    current_nrows: usize,
    append_nrows: usize,
    new_nrows: usize,
    row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    let mut buffers: Vec<RawBuffer> = Vec::new();
    let prep = dispatch_write_input(
        py, data, names, columns, append_nrows, row_width,
        &mut buffers)?;

    let current_data_bytes = (current_nrows * row_width) as u64;
    let new_data_bytes = (new_nrows * row_width) as u64;
    let current_padded = round_up_to_block(current_data_bytes);
    let new_padded = round_up_to_block(new_data_bytes);
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;

    if new_hdu_end > current_hdu_end {
        let delta = new_hdu_end - current_hdu_end;
        let file_len = {
            let guard = lock_file(&super_.file)?;
            let file = guard.as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            file.len()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        };
        if file_len > current_hdu_end {
            shift_file_tail_and_update_offsets(
                &super_.file, &super_.layout,
                current_hdu_end, delta, &super_.tainted,
            )?;
            zero_fill_range(
                &super_.file, current_hdu_end, delta, &super_.tainted,
            )?;
        } else {
            let mut guard = lock_file(&super_.file)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            file.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // Disk-write-before-commit ordering with taint on mid-write
    // failure, same as the header- and image-grow paths.
    let new_card = card_int(
        "NAXIS2", new_nrows as i64, "number of rows in table");
    let cards_guard = super_.cards_write_lock()?;
    let mut new_cards = cards_guard.clone_cards();
    let card_idx = new_cards.iter()
        .position(|c| c.len() >= 6 && c[..6].trim() == "NAXIS2")
        .ok_or_else(|| PyValueError::new_err(
            "header missing NAXIS2"))?;
    new_cards[card_idx] = new_card.trim_end().to_string();

    {
        let mut guard = lock_file(&super_.file)?;
        let file = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = data_offset - header_bytes.len() as u64;
        file.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        file.write_all(&header_bytes).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "header write failed during append: {}; close + \
                 reopen the file to recover", e))
        })?;
        file.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "header flush failed during append: {}; close + \
                 reopen the file to recover", e))
        })?;
    }
    cards_guard.commit(new_cards);

    // A write failure here taints — header already advertises the
    // larger NAXIS2 but the new rows are partly or wholly stale.
    let sources = build_sources(&prep.metas, &buffers);
    let append_offset = data_offset
        + (current_nrows * row_width) as u64;
    write_table_data(
        columns, &prep.transforms, &sources, prep.layout_matches,
        &super_.file, append_offset, append_nrows, row_width,
        &super_.tainted,
    ).inspect_err(|_e| {
        super_.tainted.store(true, Ordering::Release);
    })
}

// Rewrite (or insert) the PCOUNT card in `new_cards` to `new_pcount`.
// PCOUNT is mandatory in BINTABLE headers so we expect it to exist;
// fall back to inserting it just before TFIELDS if it's missing,
// which keeps things sane for hand-built headers.
// Rewrite the TFORMn card for a PX/QX VLA column so its optional
// `(maxlen)` hint reflects at least `new_max_bits`.  The FITS spec
// treats `(maxlen)` as informational (the per-cell descriptor is
// the actual length), but astropy's TFORM parser strictly rejects
// `1PX` without it, so we always emit the hint for X-inner VLA
// columns.  Monotonic: if the existing TFORM already has a larger
// hint, keep the larger value (so append/setitem never shrink it
// below a previously-recorded peak).
pub(crate) fn set_x_vla_tform_maxlen_in_cards(
    new_cards: &mut [String],
    column_index_1based: usize,
    descriptor_kind: char,
    new_max_bits: usize,
) {
    use crate::header::card_string;
    let key = format!("TFORM{}", column_index_1based);
    let kw_len = key.len();
    let Some(idx) = new_cards.iter().position(|c|
        c.len() >= kw_len && c[..kw_len].trim_end() == key
    ) else {
        return;  // No TFORMn card to update — shouldn't happen for
                 // a column the caller is actively writing.
    };
    // Parse the existing maxlen from "1{P|Q}X(<n>)" if present.
    let existing = new_cards[idx].as_str();
    let mut prev_max = 0usize;
    if let (Some(lp), Some(rp)) = (existing.find('('), existing.rfind(')')) {
        if rp > lp {
            if let Ok(n) = existing[lp + 1..rp].trim().parse::<usize>() {
                prev_max = n;
            }
        }
    }
    let max_bits = prev_max.max(new_max_bits);
    let tform = format!("1{}X({})", descriptor_kind, max_bits);
    new_cards[idx] = card_string(
        &key, &tform, "data format of column");
}

pub(crate) fn set_pcount_in_cards(new_cards: &mut Vec<String>, new_pcount: u64) {
    let card = card_int(
        "PCOUNT", new_pcount as i64, "size of special data area");
    let trimmed = card.trim_end().to_string();
    if let Some(idx) = new_cards.iter().position(|c|
        c.len() >= 6 && c[..6].trim() == "PCOUNT")
    {
        new_cards[idx] = trimmed;
    } else {
        let tfields_idx = new_cards.iter().position(|c|
            c.len() >= 7 && c[..7].trim() == "TFIELDS")
            .unwrap_or(new_cards.len() - 1);
        new_cards.insert(tfields_idx, trimmed);
    }
}

// Inspect the input + names= kwarg and return the row count it
// describes, without doing any per-column validation (which would
// require the columns Vec).  Used by append() before any file
// mutation so the grow + header-update can be sized correctly.
//
// For a structured ndarray: data.len() (== shape[0]).
// For a dict: shape[0] of the first value (per-column consistency
//   is enforced later by acquire_per_column_array).
// For a list/tuple: shape[0] of the first element.
pub(crate) fn determine_input_nrows(
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    if data.is_instance_of::<PyDict>() {
        if names.is_some() {
            return Err(PyValueError::new_err(
                "names= is only valid with list/tuple data, not dict"));
        }
        let d = data.cast::<PyDict>()?;
        let values = d.values();
        if values.is_empty() {
            return Err(PyValueError::new_err("data dict is empty"));
        }
        let first = values.get_item(0)?;
        let shape: Vec<usize> = first.getattr("shape")?.extract()?;
        Ok(shape.first().copied().unwrap_or(0))
    } else if data.is_instance_of::<PyList>()
        || data.is_instance_of::<PyTuple>()
    {
        if names.is_none() {
            return Err(PyValueError::new_err(
                "when data is a list/tuple, names= is required"));
        }
        if data.len()? == 0 {
            return Err(PyValueError::new_err(
                "data list/tuple is empty"));
        }
        let first = data.get_item(0)?;
        let shape: Vec<usize> = first.getattr("shape")?.extract()?;
        Ok(shape.first().copied().unwrap_or(0))
    } else {
        if names.is_some() {
            return Err(PyValueError::new_err(
                "names= is only valid with list/tuple data, not a \
                 structured ndarray"));
        }
        Ok(data.len()?)
    }
}

// Run the input-form dispatch + per-column validation shared by
// TableHDU.write and TableHDU.append.  Caller passes the row count
// it wants to validate against (NAXIS2 for write, append count for
// append) and the buffer Vec that will outlive the returned
// PreparedInput's source references.
pub(crate) fn dispatch_write_input(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
    columns: &[Column],
    expected_nrows: usize,
    row_width: usize,
    buffers: &mut Vec<RawBuffer>,
) -> PyResult<PreparedInput> {
    if data.is_instance_of::<PyDict>() {
        if names.is_some() {
            return Err(PyValueError::new_err(
                "names= is only valid with list/tuple data, not dict"));
        }
        let d = data.cast::<PyDict>()?;
        prepare_dict_input(py, d, columns, expected_nrows, buffers)
    } else if data.is_instance_of::<PyList>()
        || data.is_instance_of::<PyTuple>()
    {
        let names_obj = names.ok_or_else(|| PyValueError::new_err(
            "when data is a list/tuple, names= is required"))?;
        prepare_list_names_input(
            py, data, names_obj, columns, expected_nrows, buffers)
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
        prepare_structured_input(
            data, columns, expected_nrows, row_width, buffers)
    }
}

// Build a Vec<ColumnSource> by walking PreparedInput.metas and the
// per-call Vec<RawBuffer>.  Same pattern used by the bulk write entry
// point; factored out so the setitem helpers share it.
pub(crate) fn build_sources<'a>(
    metas: &[ColumnSourceMeta],
    buffers: &'a [RawBuffer],
) -> Vec<ColumnSource<'a>> {
    metas.iter()
        .map(|m| ColumnSource {
            src_bytes: buffers[m.buffer_idx].as_slice(),
            src_offset: m.src_offset,
            src_row_stride: m.src_row_stride,
            src_total_size: m.src_total_size,
        })
        .collect()
}
