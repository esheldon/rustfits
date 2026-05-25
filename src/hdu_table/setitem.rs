// __setitem__ helpers for TableHDU + SingleColumnSubset + ColumnSubset:
// single-row, slice, fancy-rows, single-column, multi-columns, cell,
// and the VLA-aware variants of each.

use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PySlice, PyString, PyTuple};
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::Ordering;

use crate::common::{
    lock_file, parse_keyword, shift_file_tail_and_update_offsets,
    zero_fill_range, FileHandle, RawBuffer, TaintFlag,
};
use crate::hdu::HDU;
use crate::hdu_image::{round_up_to_block, serialize_header_to_disk_bytes};

use super::columns::{bytes_per_element, Column};
use super::read::field_dtype_and_shape;
use super::write_fixed::{
    acquire_per_column_array, apply_transform_cell, build_sources,
    set_pcount_in_cards, write_table_data, write_table_one_column,
    write_table_strided, ColumnSource, prepare_structured_input,
};
use super::write_setup::column_expected_shape;
use super::write_vla::{
    any_var_column, build_fixed_col_info, extract_per_column_inputs,
    plan_vla_heap_layout, serialize_vla_cell, validate_vla_cell,
    write_descriptor, write_vla_data_range, write_vla_data_strided,
    FixedColInfo, VlaCellPlan, VlaColInfo,
};

// Normalize a possibly-negative row index against nrows; reject
// out-of-range.  Mirrors numpy/structured-array indexing semantics.
fn normalize_row_index(i: i64, nrows: usize) -> PyResult<usize> {
    let n = nrows as i64;
    let r = if i < 0 { i + n } else { i };
    if r < 0 || r >= n {
        return Err(PyIndexError::new_err(format!(
            "row index {} out of bounds for {} rows", i, nrows)));
    }
    Ok(r as usize)
}

// Locate a column by name, case-insensitively (matches read-side
// lookup conventions).
fn find_column_by_name<'a>(
    columns: &'a [Column],
    name: &str,
) -> PyResult<&'a Column> {
    let name_u = name.to_uppercase();
    for c in columns.iter() {
        if c.name.to_uppercase() == name_u {
            return Ok(c);
        }
    }
    Err(PyValueError::new_err(format!(
        "TableHDU[name] = value: no column named '{}'", name)))
}

// Coerce a single-row value into a length-1 structured ndarray that
// prepare_structured_input can consume.  Accepts numpy.void (0-d
// structured scalar) or a structured ndarray with shape `()` or `(1,)`.
// Everything else (tuple, dict, plain ndarray, etc.) is rejected with
// a clear message — those forms can be added later if requested.
pub(crate) fn coerce_to_len1_record<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let void = np.getattr("void")?;
    if !value.is_instance(&ndarray)? && !value.is_instance(&void)? {
        return Err(PyValueError::new_err(
            "TableHDU[i] = value: value must be a structured numpy record \
             (numpy.void) or a structured ndarray with one row"));
    }
    let arr = np.call_method1("asarray", (value,))?;
    let names = arr.getattr("dtype")?.getattr("names")?;
    if names.is_none() {
        return Err(PyValueError::new_err(
            "TableHDU[i] = value: value's dtype must be a structured \
             dtype with named fields"));
    }
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape.is_empty() {
        arr.call_method1("reshape", ((1usize,),))
    } else if shape == [1usize] {
        Ok(arr)
    } else {
        Err(PyValueError::new_err(format!(
            "TableHDU[i] = value: expected scalar record or shape-(1,) \
             ndarray, got shape {:?}", shape)))
    }
}

// hdu[i] = record: overwrite a single row.  The value is coerced into
// a length-1 structured ndarray and validated against the HDU columns
// the same way bulk write validates; the write then targets the byte
// range [data_offset + i*row_width, +row_width).
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_single_row(
    py: Python<'_>,
    columns: &[Column],
    file: &FileHandle,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    i: i64,
    value: &Bound<'_, PyAny>,
    tainted: &TaintFlag,
) -> PyResult<()> {
    let r = normalize_row_index(i, nrows)?;
    let arr = coerce_to_len1_record(py, value)?;
    let mut buffers: Vec<RawBuffer> = Vec::new();
    let prep = prepare_structured_input(
        &arr, columns, 1, row_width, &mut buffers)?;
    let sources = build_sources(&prep.metas, &buffers);
    let start_offset = data_offset + (r as u64) * row_width as u64;
    write_table_data(
        columns, &prep.transforms, &sources, prep.layout_matches,
        file, start_offset, 1, row_width, tainted)
}

// hdu[a:b[:s]] = arr: overwrite a range of rows.  Step-1 slices fall
// through to write_table_data with the strip-write fast path; non-unit
// steps go through write_table_strided (per-row seek + write).  Length
// validation is delegated to prepare_structured_input.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_row_slice(
    py: Python<'_>,
    columns: &[Column],
    file: &FileHandle,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    slice_py: &Bound<'_, PySlice>,
    value: &Bound<'_, PyAny>,
    tainted: &TaintFlag,
) -> PyResult<()> {
    let indices = slice_py.indices(nrows as isize)?;
    if indices.step <= 0 {
        return Err(PyValueError::new_err(
            "TableHDU[slice] = value: negative or zero step is not supported"));
    }
    let count = indices.slicelength as usize;
    let start = indices.start as i64;
    let step = indices.step as i64;

    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(
            "TableHDU[slice] = value: value must be a structured numpy \
             ndarray with one element per selected row"));
    }
    if count == 0 {
        let v_len: usize = value.len().unwrap_or(0);
        if v_len != 0 {
            return Err(PyValueError::new_err(format!(
                "TableHDU[slice] = value: slice selects 0 rows but value \
                 has length {}", v_len)));
        }
        return Ok(());
    }

    let mut buffers: Vec<RawBuffer> = Vec::new();
    let prep = prepare_structured_input(
        value, columns, count, row_width, &mut buffers)?;
    let sources = build_sources(&prep.metas, &buffers);

    if step == 1 {
        let start_offset = data_offset
            + (start as u64) * row_width as u64;
        write_table_data(
            columns, &prep.transforms, &sources, prep.layout_matches,
            file, start_offset, count, row_width, tainted)
    } else {
        let row_indices: Vec<i64> = (0..count as i64)
            .map(|r| start + r * step)
            .collect();
        write_table_strided(
            columns, &prep.transforms, &sources, prep.layout_matches,
            file, data_offset, &row_indices, row_width, tainted)
    }
}

// hdu["col"] = arr: overwrite a single column across all rows.  The
// per-column ndarray is validated the same way dict/list+names input
// validates one column, then handed to write_table_one_column.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_single_column(
    py: Python<'_>,
    columns: &[Column],
    file: &FileHandle,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    name: &str,
    value: &Bound<'_, PyAny>,
    tainted: &TaintFlag,
) -> PyResult<()> {
    let col = find_column_by_name(columns, name)?;
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let mut buffers: Vec<RawBuffer> = Vec::new();
    let (transform, src_total_size, buffer_idx) =
        acquire_per_column_array(value, &ndarray, col, nrows, &mut buffers)?;
    let source = ColumnSource {
        src_bytes: buffers[buffer_idx].as_slice(),
        src_offset: 0,
        src_row_stride: src_total_size,
        src_total_size,
    };
    write_table_one_column(
        col, &transform, &source, file, data_offset, nrows, row_width, tainted)
}

// ---------------------------------------------------------------------------
// Multi-column / fancy-row / cell setitem helpers
// ---------------------------------------------------------------------------

// hdu[[i, j, k]] = arr: write a structured ndarray to a non-contiguous
// set of row indices.  Each input row maps to one disk row; the per-
// row strided writer (write_table_strided) handles the actual I/O.
// VLA tables are rejected — strided VLA writes would need per-row
// heap layouts; defer until requested.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_fancy_rows(
    py: Python<'_>,
    columns: &[Column],
    file: &FileHandle,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    row_indices_signed: &[i64],
    value: &Bound<'_, PyAny>,
    tainted: &TaintFlag,
) -> PyResult<()> {
    // VLA-bearing tables go through the VLA-aware fancy-row helper
    // (which routes through write_vla_data_strided).  The dispatcher
    // in hdu.rs picks the right entry point per HDU; reaching this
    // branch with any VLA column is an internal routing error.
    debug_assert!(!any_var_column(columns),
        "setitem_fancy_rows called on a VLA-bearing table");
    let count = row_indices_signed.len();
    let row_indices: Vec<i64> = row_indices_signed.iter()
        .map(|&i| normalize_row_index(i, nrows).map(|r| r as i64))
        .collect::<PyResult<_>>()?;

    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(
            "TableHDU[[rows]] = value: value must be a structured \
             numpy ndarray of length equal to the row list"));
    }
    if count == 0 {
        let v_len: usize = value.len().unwrap_or(0);
        if v_len != 0 {
            return Err(PyValueError::new_err(format!(
                "TableHDU[[rows]] = value: row list is empty but value \
                 has length {}", v_len)));
        }
        return Ok(());
    }
    let v_len: usize = value.len()?;
    if v_len != count {
        return Err(PyValueError::new_err(format!(
            "TableHDU[[rows]] = value: row list has {} entries but \
             value has length {}", count, v_len)));
    }

    let mut buffers: Vec<RawBuffer> = Vec::new();
    let prep = prepare_structured_input(
        value, columns, count, row_width, &mut buffers)?;
    let sources = build_sources(&prep.metas, &buffers);
    write_table_strided(
        columns, &prep.transforms, &sources, prep.layout_matches,
        file, data_offset, &row_indices, row_width, tainted)
}

// hdu[[name1, name2]] = arr: rewrite a subset of columns across all
// rows.  Routes each named column individually to the existing
// per-column writers (fixed → setitem_single_column,
// VLA → setitem_single_column_vla); the other columns' bytes stay
// untouched.  Value must be a structured ndarray with at least the
// named fields (extras tolerated for forward compatibility), length
// equal to NAXIS2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_multi_columns(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    nrows: usize,
    row_width: usize,
    names: &[String],
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    if names.is_empty() {
        return Err(PyValueError::new_err(
            "TableHDU[[names]] = value: empty column list"));
    }
    // Validate value shape: structured ndarray of length nrows.
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(
            "TableHDU[[names]] = value: value must be a structured \
             numpy ndarray with one element per row"));
    }
    let v_len: usize = value.len()?;
    if v_len != nrows {
        return Err(PyValueError::new_err(format!(
            "TableHDU[[names]] = value: value has {} rows but table \
             NAXIS2={}", v_len, nrows)));
    }
    let dtype = value.getattr("dtype")?;
    let value_names_attr = dtype.getattr("names")?;
    if value_names_attr.is_none() {
        return Err(PyValueError::new_err(
            "TableHDU[[names]] = value: value must be a structured \
             ndarray with named fields"));
    }
    let value_names: Vec<String> = value_names_attr.extract()?;
    let value_names_upper: std::collections::HashSet<String> =
        value_names.iter().map(|n| n.to_uppercase()).collect();

    // Resolve names case-insensitively + duplicate-check.
    let mut col_indices: Vec<usize> = Vec::with_capacity(names.len());
    let mut seen_upper: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for name in names {
        let name_u = name.to_uppercase();
        if !seen_upper.insert(name_u.clone()) {
            return Err(PyValueError::new_err(format!(
                "TableHDU[[names]] = value: duplicate column name '{}'",
                name)));
        }
        let idx = columns.iter()
            .position(|c| c.name.to_uppercase() == name_u)
            .ok_or_else(|| PyValueError::new_err(format!(
                "TableHDU[[names]] = value: no column named '{}'", name)))?;
        if !value_names_upper.contains(&name_u) {
            return Err(PyValueError::new_err(format!(
                "TableHDU[[names]] = value: value structured dtype is \
                 missing field '{}'", name)));
        }
        col_indices.push(idx);
    }

    for (name, &col_idx) in names.iter().zip(col_indices.iter()) {
        // Extract the per-column ndarray as a contiguous array (the
        // structured-field view has record-itemsize strides which fail
        // RawBuffer's c-contig check).
        let field_view = value.get_item(name.as_str())?;
        let per_col = np.call_method1("ascontiguousarray", (field_view,))?;
        if columns[col_idx].var_kind.is_some() {
            setitem_single_column_vla(
                py, super_, cards, columns, col_idx, nrows, row_width,
                &per_col, data_offset)?;
        } else {
            setitem_single_column(
                py, columns, &super_.file, data_offset, nrows, row_width,
                name, &per_col, &super_.tainted)?;
        }
    }
    Ok(())
}

// hdu[row, "col"] = value: single-cell write.  For a fixed-width
// column: convert the value to the column's expected per-cell shape,
// encode to bytes, write byte_width bytes at row_offset +
// col.byte_offset.  For a VLA column: append cell bytes to the heap
// end, rewrite the row's descriptor, update PCOUNT.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_cell(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    nrows: usize,
    row_width: usize,
    row_idx_signed: i64,
    col_name: &str,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let r = normalize_row_index(row_idx_signed, nrows)?;
    let name_u = col_name.to_uppercase();
    let col_idx = columns.iter()
        .position(|c| c.name.to_uppercase() == name_u)
        .ok_or_else(|| PyValueError::new_err(format!(
            "TableHDU[(row, col)] = value: no column named '{}'",
            col_name)))?;
    let col = &columns[col_idx];

    if col.var_kind.is_some() {
        return setitem_cell_vla(
            py, super_, cards, columns, col_idx, nrows, row_width,
            r, value, data_offset);
    }

    // Fixed cell: promote `value` to a length-1 per-column ndarray
    // matching the column's expected dtype and shape.  np.asarray
    // with the column dtype coerces Python ints / floats to the right
    // width (NEP 50 raises OverflowError on out-of-range int → int
    // narrowing).  np.broadcast_to handles 0-d scalars + pre-shaped
    // ndarrays uniformly; shape mismatches surface as numpy ValueError.
    let np = py.import("numpy")?;
    let expected_shape: Vec<usize> = column_expected_shape(col);
    let full_shape: Vec<usize> = std::iter::once(1)
        .chain(expected_shape.iter().copied()).collect();
    let (dtype_str, _) = field_dtype_and_shape(col, false)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", &dtype_str)?;
    let arr = np.call_method("asarray", (value,), Some(&kwargs))?;
    let broadcast = np.getattr("broadcast_to")?
        .call1((arr, full_shape))?;
    let promoted = np.call_method1("ascontiguousarray", (broadcast,))?;

    let mut buffers: Vec<RawBuffer> = Vec::new();
    let ndarray = np.getattr("ndarray")?;
    let (transform, src_total_size, buffer_idx) =
        acquire_per_column_array(&promoted, &ndarray, col, 1, &mut buffers)?;
    let source = ColumnSource {
        src_bytes: buffers[buffer_idx].as_slice(),
        src_offset: 0,
        src_row_stride: src_total_size,
        src_total_size,
    };
    let mut cell_buf = vec![0u8; col.byte_width];
    apply_transform_cell(
        &transform, source.src_bytes, &mut cell_buf, &col.name, 0)?;

    let file_off = data_offset
        + (r * row_width + col.byte_offset) as u64;
    let mut guard = lock_file(&super_.file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    f.seek(SeekFrom::Start(file_off))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    f.write_all(&cell_buf).map_err(|e| {
        super_.tainted.store(true, Ordering::Release);
        PyIOError::new_err(format!(
            "cell write failed: {}; close + reopen", e))
    })?;
    f.flush().map_err(|e| {
        super_.tainted.store(true, Ordering::Release);
        PyIOError::new_err(format!(
            "cell flush failed: {}; close + reopen", e))
    })?;
    Ok(())
}

// hdu[row, "vla_col"] = value: append the new cell bytes at the heap
// end, rewrite the row's descriptor in place, update PCOUNT.  Same
// orphan-and-append model as setitem_single_column_vla but for one
// row.
#[allow(clippy::too_many_arguments)]
fn setitem_cell_vla(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    col_idx: usize,
    nrows: usize,
    row_width: usize,
    row_idx: usize,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let col = &columns[col_idx];
    let descriptor_kind = col.var_kind.unwrap();
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let nelements = validate_vla_cell(
        value, &ndarray, col.tform_letter, &col.name, row_idx)?;
    // X (bit-packed) VLA: descriptor nelements is the bit count,
    // heap bytes per cell = ceil(nelements/8).  Other letters use
    // a fixed element width.
    let n_bytes = if col.tform_letter == 'X' {
        nelements.div_ceil(8)
    } else {
        let elem_size = bytes_per_element(col.tform_letter).unwrap_or(0);
        nelements * elem_size
    };

    let current_pcount = parse_keyword(cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    let new_pcount = current_pcount + n_bytes as u64;
    let nrows_u64 = nrows as u64;
    let main_bytes = nrows_u64 * row_width as u64;
    let current_padded = round_up_to_block(main_bytes + current_pcount);
    let new_padded = round_up_to_block(main_bytes + new_pcount);
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
                &super_.file, current_hdu_end, delta, &super_.tainted)?;
        } else {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // Build heap bytes + descriptor bytes.
    let mut heap_bytes = vec![0u8; n_bytes];
    if n_bytes > 0 {
        serialize_vla_cell(
            value, col.tform_letter, nelements, &mut heap_bytes)?;
    }
    let mut desc_bytes = vec![0u8; col.byte_width];
    write_descriptor(
        descriptor_kind, nelements,
        current_pcount as usize, &mut desc_bytes);

    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        if n_bytes > 0 {
            let heap_off = data_offset + main_bytes + current_pcount;
            f.seek(SeekFrom::Start(heap_off))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.write_all(&heap_bytes).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "VLA cell heap write failed: {}; close + reopen", e))
            })?;
        }
        let desc_off = data_offset
            + (row_idx as u64) * row_width as u64
            + col.byte_offset as u64;
        f.seek(SeekFrom::Start(desc_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&desc_bytes).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "VLA cell descriptor write failed: {}; close + reopen", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "VLA cell flush failed: {}; close + reopen", e))
        })?;
    }

    // PCOUNT update via disk-write-before-commit.
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

// Resolve a __setitem__ rows key (int / slice / iterable of ints) to
// a normalized Vec<usize>.  Used by SingleColumnSubset.__setitem__
// and ColumnSubset.__setitem__ to flatten any of the three row
// shapes into a single per-row loop.  Returns (rows_vec, was_single):
// `was_single` is True when the user passed a bare int (so callers
// can validate the value as a scalar / record rather than an ndarray
// of length 1).
fn resolve_rows_key(
    rows: &Bound<'_, PyAny>,
    nrows: usize,
) -> PyResult<(Vec<usize>, bool)> {
    if rows.is_instance_of::<PySlice>() {
        let slice_py = rows.cast::<PySlice>()?;
        let indices = slice_py.indices(nrows as isize)?;
        let count = indices.slicelength as i64;
        if count <= 0 {
            return Ok((Vec::new(), false));
        }
        let start = indices.start as i64;
        let step = indices.step as i64;
        let mut out = Vec::with_capacity(count as usize);
        for k in 0..count {
            let r = start + k * step;
            if r < 0 || r as i64 >= nrows as i64 {
                return Err(PyIndexError::new_err(format!(
                    "row index {} out of bounds for {} rows", r, nrows)));
            }
            out.push(r as usize);
        }
        return Ok((out, false));
    }
    if !rows.is_instance_of::<PyBool>() {
        if let Ok(i) = rows.extract::<i64>() {
            let r = normalize_row_index(i, nrows)?;
            return Ok((vec![r], true));
        }
    }
    // Iterable of ints.
    let iter = rows.try_iter().map_err(|_| PyValueError::new_err(
        "row key must be an int, slice, or iterable of ints"))?;
    let items: Vec<Bound<'_, PyAny>> = iter.collect::<PyResult<_>>()?;
    let mut out: Vec<usize> = Vec::with_capacity(items.len());
    for item in items.iter() {
        if item.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(
                "row iterable contains a bool"));
        }
        let i: i64 = item.extract().map_err(|_| PyValueError::new_err(
            "row iterable contains a non-int element"))?;
        out.push(normalize_row_index(i, nrows)?);
    }
    Ok((out, false))
}

// hdu["name"][rows] = value: write to one column at the specified
// rows.  Handles all three row-key shapes by flattening through
// resolve_rows_key and looping per-cell.  Cards are re-snapshotted
// between cells so VLA writes see fresh PCOUNT.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_one_column_at_rows(
    py: Python<'_>,
    super_: &HDU,
    columns: &[Column],
    nrows: usize,
    row_width: usize,
    col_name: &str,
    rows: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let (row_indices, was_single) = resolve_rows_key(rows, nrows)?;
    if row_indices.is_empty() {
        return Ok(());
    }
    if was_single {
        // [int] = v shortcut.
        let cards = super_.header_snapshot()?;
        return setitem_cell(
            py, super_, &cards, columns, nrows, row_width,
            row_indices[0] as i64, col_name, value, data_offset);
    }
    // Slice / fancy: value must be a length-matching ndarray (or an
    // Object ndarray for VLA columns).  Validate length up front, then
    // walk rows.
    let count = row_indices.len();
    let v_len: usize = value.len().map_err(|_| PyValueError::new_err(
        "value must be indexable (ndarray, list, ...) for slice/fancy \
         row writes"))?;
    if v_len != count {
        return Err(PyValueError::new_err(format!(
            "value has length {} but row selector picks {} rows",
            v_len, count)));
    }
    for (i, &r) in row_indices.iter().enumerate() {
        let cell_value = value.get_item(i)?;
        let cards = super_.header_snapshot()?;
        setitem_cell(
            py, super_, &cards, columns, nrows, row_width,
            r as i64, col_name, &cell_value, data_offset)?;
    }
    Ok(())
}

// hdu[["a","b"]][rows] = value: write a column subset at the
// specified rows.  For int row: value is a structured scalar
// (numpy.void) or shape-(1,) ndarray; we walk each column and
// extract its field as the cell.  For slice/fancy: value is a
// structured ndarray of length len(row indices); per row, per
// column, extract the cell and forward to setitem_cell.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_column_subset_at_rows(
    py: Python<'_>,
    super_: &HDU,
    columns: &[Column],
    nrows: usize,
    row_width: usize,
    col_names: &[String],
    rows: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    if col_names.is_empty() {
        return Err(PyValueError::new_err(
            "column subset is empty"));
    }
    let (row_indices, was_single) = resolve_rows_key(rows, nrows)?;
    if row_indices.is_empty() {
        return Ok(());
    }

    // For single-row writes, coerce value to a length-1 structured
    // record so we can index value[name][0] uniformly.  For
    // slice/fancy, value must already be a structured ndarray of the
    // matching length.
    let one_row_view = if was_single {
        coerce_to_len1_record(py, value)?
    } else {
        let np = py.import("numpy")?;
        let ndarray = np.getattr("ndarray")?;
        if !value.is_instance(&ndarray)? {
            return Err(PyValueError::new_err(
                "value must be a structured numpy ndarray for slice/\
                 fancy-row column-subset writes"));
        }
        let v_len: usize = value.len()?;
        if v_len != row_indices.len() {
            return Err(PyValueError::new_err(format!(
                "value has length {} but row selector picks {} rows",
                v_len, row_indices.len())));
        }
        value.clone()
    };

    // Validate field presence up front (before any I/O).
    let dtype = one_row_view.getattr("dtype")?;
    let field_names_attr = dtype.getattr("names")?;
    if field_names_attr.is_none() {
        return Err(PyValueError::new_err(
            "value dtype must be a structured dtype with named fields"));
    }
    let field_names: Vec<String> = field_names_attr.extract()?;
    let field_set: std::collections::HashSet<String> =
        field_names.iter().map(|n| n.to_uppercase()).collect();
    for name in col_names {
        if !field_set.contains(&name.to_uppercase()) {
            return Err(PyValueError::new_err(format!(
                "value is missing field '{}'", name)));
        }
    }

    for (i, &r) in row_indices.iter().enumerate() {
        for name in col_names {
            let field_view = one_row_view.get_item(name.as_str())?;
            let cell = field_view.get_item(i)?;
            let cards = super_.header_snapshot()?;
            setitem_cell(
                py, super_, &cards, columns, nrows, row_width,
                r as i64, name, &cell, data_offset)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// VLA __setitem__ helpers
// ---------------------------------------------------------------------------
//
// All three forms follow the same heap model: new cells are appended
// at the end of the existing heap (heap_start_offset = current PCOUNT)
// and the old cells become orphans.  PCOUNT grows monotonically.
// Mirrors the compressed-image __setitem__ pattern; a future repack()
// can compact the heap when a workload demands it.
//
// Validate-then-mutate: input is fully validated before any file or
// header bytes are touched, so a dtype/shape error leaves the file
// unchanged.  Mid-write I/O failures taint the file (close + reopen
// to recover) — same semantics as every other write path.

// Shared core for the single-row and row-slice VLA setitem paths.
// Writes `input_nrows` contiguous rows of main-table data starting at
// row index `first_row`, planning the new heap to start at the
// current PCOUNT.  Caller has already extracted the per-column input
// ndarrays and validated their length == input_nrows.
#[allow(clippy::too_many_arguments)]
// Row-selection shape passed to the VLA-aware inner helper.
// Contiguous(first_row, count) → write_vla_data_range (strip-walk
// fast path).  Strided(&[disk_rows]) → write_vla_data_strided
// (per-row seek+write).  Both produce the same on-disk result for
// the contiguous case; the split is purely a performance choice.
enum VlaRowSpec<'a> {
    Contiguous { first_row: usize, count: usize },
    Strided { disk_rows: &'a [usize] },
}

impl<'a> VlaRowSpec<'a> {
    fn input_nrows(&self) -> usize {
        match self {
            VlaRowSpec::Contiguous { count, .. } => *count,
            VlaRowSpec::Strided { disk_rows } => disk_rows.len(),
        }
    }
}

fn setitem_rows_vla_aware_inner(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    per_col: Vec<Bound<'_, PyAny>>,
    rows: VlaRowSpec<'_>,
    row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    let input_nrows = rows.input_nrows();
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;

    // Build per-column fixed info up front so dtype/shape errors
    // surface before any file mutation.
    let mut fixed: Vec<Option<FixedColInfo>> =
        columns.iter().map(|_| None).collect();
    for (col_idx, col) in columns.iter().enumerate() {
        if col.var_kind.is_none() {
            fixed[col_idx] = Some(build_fixed_col_info(
                &per_col[col_idx], &ndarray, col, input_nrows)?);
        }
    }

    let nrows_total = parse_keyword(cards, "NAXIS2")
        .unwrap_or(0).max(0) as u64;
    let current_pcount = parse_keyword(cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    let (plans, total_heap_bytes) = plan_vla_heap_layout(
        columns, &per_col, input_nrows, &ndarray,
        current_pcount as usize)?;
    let new_pcount = total_heap_bytes as u64;

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

    let main_bytes = nrows_total * row_width as u64;
    let current_data_bytes = main_bytes + current_pcount;
    let new_data_bytes = main_bytes + new_pcount;
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
                &super_.file, current_hdu_end, delta, &super_.tainted)?;
        } else {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // New heap bytes start right after the existing heap end.
    let heap_start_offset_in_file =
        data_offset + main_bytes + current_pcount;
    match rows {
        VlaRowSpec::Contiguous { first_row, count } => {
            let main_start_offset =
                data_offset + (first_row as u64) * row_width as u64;
            write_vla_data_range(
                columns, &fixed, &vla, total_heap_bytes,
                current_pcount as usize,
                &super_.file, main_start_offset, heap_start_offset_in_file,
                count, row_width, &super_.tainted)?;
        }
        VlaRowSpec::Strided { disk_rows } => {
            write_vla_data_strided(
                columns, &fixed, &vla, total_heap_bytes,
                current_pcount as usize,
                &super_.file, data_offset, heap_start_offset_in_file,
                disk_rows, row_width, &super_.tainted)?;
        }
    }

    // PCOUNT update — disk-write-before-commit.  No NAXIS2 change
    // (row count unchanged).
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

// hdu[i] = record on a VLA-bearing table.  Coerces value to a length-1
// structured ndarray, extracts per-column inputs, and dispatches to
// the shared inner helper.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_single_row_vla_aware(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    nrows: usize,
    row_width: usize,
    i: i64,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let r = normalize_row_index(i, nrows)?;
    let arr = coerce_to_len1_record(py, value)?;
    let per_col = extract_per_column_inputs(
        py, &arr, None, columns)?;
    setitem_rows_vla_aware_inner(
        py, super_, cards, columns, per_col,
        VlaRowSpec::Contiguous { first_row: r, count: 1 },
        row_width, data_offset)
}

// hdu[a:b[:s]] = arr on a VLA-bearing table.  step=1 uses the
// contiguous strip-walk writer; step>1 routes to the strided per-
// row writer.  Negative or zero step is rejected (parity with the
// fixed-column slice path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_row_slice_vla_aware(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    nrows: usize,
    row_width: usize,
    slice_py: &Bound<'_, PySlice>,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let indices = slice_py.indices(nrows as isize)?;
    if indices.step <= 0 {
        return Err(PyValueError::new_err(
            "TableHDU[slice] = value: negative or zero step is not \
             supported"));
    }
    let count = indices.slicelength as usize;
    let start = indices.start as usize;
    let step = indices.step as usize;
    if count == 0 {
        let v_len: usize = value.len().unwrap_or(0);
        if v_len != 0 {
            return Err(PyValueError::new_err(format!(
                "TableHDU[slice] = value: slice selects 0 rows but value \
                 has length {}", v_len)));
        }
        return Ok(());
    }
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(
            "TableHDU[slice] = value: value must be a structured numpy \
             ndarray with one element per selected row"));
    }
    let v_len: usize = value.len()?;
    if v_len != count {
        return Err(PyValueError::new_err(format!(
            "TableHDU[slice] = value: slice selects {} rows but value \
             has length {}", count, v_len)));
    }
    let per_col = extract_per_column_inputs(
        py, value, None, columns)?;
    let rows = if step == 1 {
        VlaRowSpec::Contiguous { first_row: start, count }
    } else {
        let disk_rows: Vec<usize> =
            (0..count).map(|r| start + r * step).collect();
        // Build owned Vec, then borrow it for the call.  Lifetimes
        // work because the helper takes the slice by reference for
        // the duration of the call.
        return setitem_rows_vla_aware_inner(
            py, super_, cards, columns, per_col,
            VlaRowSpec::Strided { disk_rows: &disk_rows },
            row_width, data_offset);
    };
    setitem_rows_vla_aware_inner(
        py, super_, cards, columns, per_col, rows, row_width, data_offset)
}

// hdu[[i, j, k]] = arr on a VLA-bearing table.  Routes a flat
// list of disk-row indices through the strided per-row writer.
// Same heap-append-orphan model as the contiguous case; duplicate
// row indices in the input list follow numpy fancy-assignment
// semantics (last write wins).
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_fancy_rows_vla_aware(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    nrows: usize,
    row_width: usize,
    row_indices_signed: &[i64],
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let count = row_indices_signed.len();
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(
            "TableHDU[[rows]] = value: value must be a structured \
             numpy ndarray of length equal to the row list"));
    }
    if count == 0 {
        let v_len: usize = value.len().unwrap_or(0);
        if v_len != 0 {
            return Err(PyValueError::new_err(format!(
                "TableHDU[[rows]] = value: row list is empty but value \
                 has length {}", v_len)));
        }
        return Ok(());
    }
    let v_len: usize = value.len()?;
    if v_len != count {
        return Err(PyValueError::new_err(format!(
            "TableHDU[[rows]] = value: row list has {} entries but \
             value has length {}", count, v_len)));
    }
    let disk_rows: Vec<usize> = row_indices_signed.iter()
        .map(|&i| normalize_row_index(i, nrows))
        .collect::<PyResult<_>>()?;
    let per_col = extract_per_column_inputs(
        py, value, None, columns)?;
    setitem_rows_vla_aware_inner(
        py, super_, cards, columns, per_col,
        VlaRowSpec::Strided { disk_rows: &disk_rows },
        row_width, data_offset)
}

// hdu["vla_col"] = arr where the named column is variable-length.
// Writes only the column's descriptor bytes at each row (the other
// columns' bytes are untouched), appending the new cell bytes at the
// end of the heap.  Old cells become orphans.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_single_column_vla(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    col_idx: usize,
    nrows: usize,
    row_width: usize,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let col = &columns[col_idx];
    let descriptor_kind = col.var_kind.unwrap();
    let is_x = col.tform_letter == 'X';
    let elem_size = if is_x {
        0
    } else {
        bytes_per_element(col.tform_letter).unwrap_or(0)
    };

    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(format!(
            "column '{}': value must be a numpy ndarray", col.name)));
    }
    let shape: Vec<usize> = value.getattr("shape")?.extract()?;
    if shape.len() != 1 || shape[0] != nrows {
        return Err(PyValueError::new_err(format!(
            "column '{}': value must have shape ({},); got {:?}",
            col.name, nrows, shape)));
    }
    let dtype_kind: String = value.getattr("dtype")?
        .getattr("kind")?.extract()?;
    if dtype_kind != "O" {
        return Err(PyValueError::new_err(format!(
            "column '{}': VLA column write requires an Object-dtype \
             ndarray (one inner ndarray per row); got dtype kind '{}'",
            col.name, dtype_kind)));
    }

    // Plan + validate every cell up front.  X (bit-packed) cells
    // contribute ceil(nelements/8) bytes per cell to the heap;
    // other letters contribute nelements * elem_size.
    let current_pcount = parse_keyword(cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    let mut plans: Vec<VlaCellPlan> = Vec::with_capacity(nrows);
    let mut cursor = current_pcount as usize;
    for r in 0..nrows {
        let cell = value.get_item(r)?;
        let nelements = validate_vla_cell(
            &cell, &ndarray, col.tform_letter, &col.name, r)?;
        plans.push(VlaCellPlan {
            nelements, bytes_offset_in_heap: cursor,
        });
        let cell_bytes = if is_x {
            nelements.div_ceil(8)
        } else {
            nelements * elem_size
        };
        cursor = cursor.checked_add(cell_bytes)
            .ok_or_else(|| PyValueError::new_err("heap size overflow"))?;
    }
    let new_pcount = cursor as u64;
    let added_heap_bytes = (new_pcount - current_pcount) as usize;

    let nrows_u64 = nrows as u64;
    let main_bytes = nrows_u64 * row_width as u64;
    let current_data_bytes = main_bytes + current_pcount;
    let new_data_bytes = main_bytes + new_pcount;
    let current_padded = round_up_to_block(current_data_bytes);
    let new_padded = round_up_to_block(new_data_bytes);
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;

    // Grow file if heap end pushes past the padded extent.
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

    // Build the heap-bytes buffer + per-row descriptor bytes.  X
    // (bit-packed) cells contribute ceil(nelements/8) bytes;
    // other letters use the fixed elem_size.
    let mut heap_buf = vec![0u8; added_heap_bytes];
    let desc_width = col.byte_width;
    let mut desc_bytes = vec![0u8; nrows * desc_width];
    for r in 0..nrows {
        let plan = plans[r];
        let cell = value.get_item(r)?;
        if plan.nelements > 0 {
            let local_off = plan.bytes_offset_in_heap
                - current_pcount as usize;
            let n_bytes = if is_x {
                plan.nelements.div_ceil(8)
            } else {
                plan.nelements * elem_size
            };
            serialize_vla_cell(
                &cell, col.tform_letter, plan.nelements,
                &mut heap_buf[local_off..local_off + n_bytes])?;
        }
        let dst = &mut desc_bytes[r * desc_width..(r + 1) * desc_width];
        write_descriptor(
            descriptor_kind, plan.nelements,
            plan.bytes_offset_in_heap, dst);
    }

    // Write new heap bytes first (no descriptors yet refer to them),
    // then walk rows and overwrite each descriptor in place.
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        if added_heap_bytes > 0 {
            let heap_off = data_offset + main_bytes + current_pcount;
            f.seek(SeekFrom::Start(heap_off))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.write_all(&heap_buf).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "heap write failed during VLA column setitem: {}", e))
            })?;
        }
        for r in 0..nrows {
            let off = data_offset
                + (r as u64) * row_width as u64
                + col.byte_offset as u64;
            f.seek(SeekFrom::Start(off))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.write_all(
                &desc_bytes[r * desc_width..(r + 1) * desc_width]
            ).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "descriptor write failed during VLA column setitem: {}",
                    e))
            })?;
        }
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "flush failed during VLA column setitem: {}", e))
        })?;
    }

    // PCOUNT update (disk-write-before-commit).
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

// What kind of selection the user passed to TableHDU.__setitem__.
// Mirrors the read-side TableKey plus an extra `Cell` variant for the
// (row, column) tuple form.  Mixed-tuple keys like (slice, [names])
// are rejected — those are deferable surface extensions.
pub(crate) enum SetItemKey {
    SingleRow(i64),
    RowSlice,
    SingleColumn(String),
    FancyRows(Vec<i64>),
    MultiColumns(Vec<String>),
    Cell(i64, String),
}

pub(crate) fn classify_setitem_key(key: &Bound<'_, PyAny>) -> PyResult<SetItemKey> {
    if key.is_instance_of::<PySlice>() {
        return Ok(SetItemKey::RowSlice);
    }
    if let Some(name) = try_extract_column_name(key)? {
        return Ok(SetItemKey::SingleColumn(name));
    }
    if !key.is_instance_of::<PyBool>() {
        if let Ok(idx) = key.extract::<i64>() {
            return Ok(SetItemKey::SingleRow(idx));
        }
    }
    // Two-element tuple `(row, col)` — single cell write.  Other tuple
    // shapes (slice/list rows × list of cols, etc.) are rejected for
    // now; users can chain the existing forms instead.
    if let Ok(tup) = key.cast::<PyTuple>() {
        if tup.len() == 2 {
            let row_obj = tup.get_item(0)?;
            let col_obj = tup.get_item(1)?;
            let row_is_int = !row_obj.is_instance_of::<PyBool>()
                && row_obj.extract::<i64>().is_ok();
            let col_name = try_extract_column_name(&col_obj)?;
            if row_is_int && col_name.is_some() {
                let idx: i64 = row_obj.extract()?;
                return Ok(SetItemKey::Cell(idx, col_name.unwrap()));
            }
            return Err(PyValueError::new_err(
                "TableHDU[(row, col)] = value requires (int row, \
                 str column name); other tuple shapes are not \
                 supported"));
        }
    }
    // Iterable: classify by first element (matches __getitem__'s
    // classify_table_key shape).
    let iter = key.try_iter().map_err(|_| PyValueError::new_err(
        "TableHDU[key] = value: key must be an int, slice, column \
         name, two-tuple (row, col), iterable of ints (fancy rows), \
         or iterable of str (column subset)"
    ))?;
    let items: Vec<Bound<'_, PyAny>> = iter.collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Err(PyValueError::new_err(
            "TableHDU[key] = value: empty sequence is ambiguous \
             (rows or columns?)"));
    }
    let first = &items[0];
    if try_extract_column_name(first)?.is_some() {
        let names: Vec<String> = items.iter()
            .map(|i| try_extract_column_name(i)?.ok_or_else(|| {
                PyValueError::new_err(
                    "TableHDU[key] = value: column-name sequence \
                     contains non-string elements")
            }))
            .collect::<PyResult<_>>()?;
        Ok(SetItemKey::MultiColumns(names))
    } else if !first.is_instance_of::<PyBool>() && first.extract::<i64>().is_ok() {
        let rows: Vec<i64> = items.iter()
            .map(|i| {
                if i.is_instance_of::<PyBool>() {
                    return Err(PyValueError::new_err(
                        "TableHDU[key] = value: row-index sequence \
                         contains a bool"));
                }
                i.extract::<i64>().map_err(|_| PyValueError::new_err(
                    "TableHDU[key] = value: row-index sequence mixes \
                     ints and non-ints"))
            })
            .collect::<PyResult<_>>()?;
        Ok(SetItemKey::FancyRows(rows))
    } else {
        Err(PyValueError::new_err(
            "TableHDU[key] = value: sequence must be all int (rows) \
             or all str (columns)"))
    }
}

// Try to extract `obj` as a string-like column name: str, bytes,
// numpy.str_, or numpy.bytes_.  Returns Ok(None) for anything else.
//
// Type checks are explicit (PyString / PyBytes instance checks) rather
// than relying on extract::<String>() / extract::<Vec<u8>>() — the
// latter is generic over iterables, so a Python list of small ints
// like [2, 0] would silently succeed as Vec<u8>=[2,0] and be
// mis-routed to a column lookup with control-char "name".
pub(crate) fn try_extract_column_name(obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(None);
    }
    if obj.is_instance_of::<PyString>() {
        // numpy.str_ is a str subclass, so this catches it too.
        return Ok(Some(obj.extract::<String>()?));
    }
    if obj.is_instance_of::<PyBytes>() {
        // numpy.bytes_ is a bytes subclass, so this catches it too.
        let b: Vec<u8> = obj.extract()?;
        if !b.iter().all(|c| c.is_ascii()) {
            return Err(PyValueError::new_err(
                "bytes-like column name contains non-ASCII bytes",
            ));
        }
        return Ok(Some(String::from_utf8(b).unwrap()));
    }
    Ok(None)
}
