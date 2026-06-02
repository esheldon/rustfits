// __setitem__ helpers for AsciiTableHDU + AsciiSingleColumnSubset +
// AsciiColumnSubset: single-row, slice, fancy-rows, single-column,
// multi-columns, and cell writes.
//
// Mirrors `hdu_table/setitem.rs` minus all VLA and WriteTransform
// machinery — ASCII tables have no heap, and every cell is text-
// formatted (no fast-path memcpy possible).  All partial-write paths
// pass `pad_to_block=false` to `write_ascii_table_data` so the data
// section's existing trailing block-pad is left untouched.

use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PySlice, PyString};
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::Ordering;

use crate::common::lock_file;
use crate::hdu::HDU;

use super::columns::AsciiColumn;
use super::write_fixed::{
    extract_per_column_arrays, format_one_cell, write_ascii_table_data,
    write_ascii_table_one_column, write_ascii_table_strided,
};

// Normalize a possibly-negative row index against nrows; reject
// out-of-range.  Same semantics as numpy / BINTABLE row indexing.
fn normalize_row_index(i: i64, nrows: usize) -> PyResult<usize> {
    let n = nrows as i64;
    let r = if i < 0 { i + n } else { i };
    if r < 0 || r >= n {
        return Err(PyIndexError::new_err(format!(
            "row index {} out of bounds for {} rows", i, nrows)));
    }
    Ok(r as usize)
}

// Locate a column by name, case-insensitively.
fn find_column_by_name<'a>(
    columns: &'a [AsciiColumn], name: &str,
) -> PyResult<&'a AsciiColumn> {
    let name_u = name.to_uppercase();
    for c in columns.iter() {
        if c.name.to_uppercase() == name_u {
            return Ok(c);
        }
    }
    Err(PyValueError::new_err(format!(
        "AsciiTableHDU[name] = value: no column named '{}'", name)))
}

// Coerce a single-row value into a length-1 structured ndarray that
// `extract_per_column_arrays` can consume.  Accepts numpy.void (0-d
// structured scalar) or a structured ndarray with shape `()` or `(1,)`.
// Mirrors `hdu_table::coerce_to_len1_record`.
fn coerce_to_len1_record<'py>(
    py: Python<'py>, value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let void = np.getattr("void")?;
    if !value.is_instance(&ndarray)? && !value.is_instance(&void)? {
        return Err(PyValueError::new_err(
            "AsciiTableHDU[i] = value: value must be a structured numpy \
             record (numpy.void) or a structured ndarray with one row"));
    }
    let arr = np.call_method1("asarray", (value,))?;
    let names = arr.getattr("dtype")?.getattr("names")?;
    if names.is_none() {
        return Err(PyValueError::new_err(
            "AsciiTableHDU[i] = value: value's dtype must be a structured \
             dtype with named fields"));
    }
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape.is_empty() {
        arr.call_method1("reshape", ((1usize,),))
    } else if shape == [1usize] {
        Ok(arr)
    } else {
        Err(PyValueError::new_err(format!(
            "AsciiTableHDU[i] = value: expected scalar record or shape-(1,) \
             ndarray, got shape {:?}", shape)))
    }
}

// hdu[i] = record.  Coerce value to a length-1 structured ndarray,
// extract per-column views, write 1 row at the targeted byte range.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_single_row(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
    nrows: usize,
    row_width: usize,
    i: i64,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let r = normalize_row_index(i, nrows)?;
    let arr = coerce_to_len1_record(py, value)?;
    let per_column =
        extract_per_column_arrays(py, &arr, None, columns, 1)?;
    let start_offset = data_offset + (r as u64) * row_width as u64;
    write_ascii_table_data(
        py, super_, columns, &per_column, row_width, 1,
        start_offset, /* pad_to_block = */ false)
}

// hdu[a:b[:s]] = arr.  step=1 routes to the contiguous strip writer;
// step>1 routes to the per-row strided writer.  Negative / zero step
// rejected (parity with BINTABLE).
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_row_slice(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
    nrows: usize,
    row_width: usize,
    slice_py: &Bound<'_, PySlice>,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let indices = slice_py.indices(nrows as isize)?;
    if indices.step <= 0 {
        return Err(PyValueError::new_err(
            "AsciiTableHDU[slice] = value: negative or zero step is not \
             supported"));
    }
    let count = indices.slicelength;
    let start = indices.start as i64;
    let step = indices.step as i64;

    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(
            "AsciiTableHDU[slice] = value: value must be a structured \
             numpy ndarray with one element per selected row"));
    }
    if count == 0 {
        let v_len: usize = value.len().unwrap_or(0);
        if v_len != 0 {
            return Err(PyValueError::new_err(format!(
                "AsciiTableHDU[slice] = value: slice selects 0 rows but \
                 value has length {}", v_len)));
        }
        return Ok(());
    }
    let v_len: usize = value.len()?;
    if v_len != count {
        return Err(PyValueError::new_err(format!(
            "AsciiTableHDU[slice] = value: slice selects {} rows but \
             value has length {}", count, v_len)));
    }
    let per_column =
        extract_per_column_arrays(py, value, None, columns, count)?;

    if step == 1 {
        let start_offset =
            data_offset + (start as u64) * row_width as u64;
        write_ascii_table_data(
            py, super_, columns, &per_column, row_width, count,
            start_offset, /* pad_to_block = */ false)
    } else {
        let row_indices: Vec<usize> = (0..count as i64)
            .map(|k| (start + k * step) as usize)
            .collect();
        write_ascii_table_strided(
            py, super_, columns, &per_column, row_width,
            &row_indices, data_offset)
    }
}

// hdu["col"] = arr.  Whole-column write: per-row direct write of just
// the column's byte_width bytes at col.byte_offset.  Other columns'
// bytes are preserved by never being touched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_single_column(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
    nrows: usize,
    row_width: usize,
    name: &str,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let col = find_column_by_name(columns, name)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("asanyarray", (value,))?;
    let v_len: usize = arr.len().map_err(|_| PyValueError::new_err(
        "AsciiTableHDU['col'] = value: value must be array-like"))?;
    if v_len != nrows {
        return Err(PyValueError::new_err(format!(
            "AsciiTableHDU['{}'] = value: value has length {} but table \
             NAXIS2={}", col.name, v_len, nrows)));
    }
    write_ascii_table_one_column(
        py, super_, col, &arr, nrows, row_width, data_offset)
}

// hdu[[i, j, k]] = arr.  Each input row maps to one disk row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_fancy_rows(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
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
            "AsciiTableHDU[[rows]] = value: value must be a structured \
             numpy ndarray of length equal to the row list"));
    }
    if count == 0 {
        let v_len: usize = value.len().unwrap_or(0);
        if v_len != 0 {
            return Err(PyValueError::new_err(format!(
                "AsciiTableHDU[[rows]] = value: row list is empty but \
                 value has length {}", v_len)));
        }
        return Ok(());
    }
    let v_len: usize = value.len()?;
    if v_len != count {
        return Err(PyValueError::new_err(format!(
            "AsciiTableHDU[[rows]] = value: row list has {} entries but \
             value has length {}", count, v_len)));
    }
    let row_indices: Vec<usize> = row_indices_signed.iter()
        .map(|&i| normalize_row_index(i, nrows))
        .collect::<PyResult<_>>()?;
    let per_column =
        extract_per_column_arrays(py, value, None, columns, count)?;
    write_ascii_table_strided(
        py, super_, columns, &per_column, row_width,
        &row_indices, data_offset)
}

// hdu[[name1, name2]] = arr.  Per named column, route the field-view
// to the whole-column writer; other columns untouched.  Case-
// insensitive name lookup; duplicate name in the list raises.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_multi_columns(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
    nrows: usize,
    row_width: usize,
    names: &[String],
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    if names.is_empty() {
        return Err(PyValueError::new_err(
            "AsciiTableHDU[[names]] = value: empty column list"));
    }
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(
            "AsciiTableHDU[[names]] = value: value must be a structured \
             numpy ndarray with one element per row"));
    }
    let v_len: usize = value.len()?;
    if v_len != nrows {
        return Err(PyValueError::new_err(format!(
            "AsciiTableHDU[[names]] = value: value has {} rows but \
             table NAXIS2={}", v_len, nrows)));
    }
    let dtype = value.getattr("dtype")?;
    let value_names_attr = dtype.getattr("names")?;
    if value_names_attr.is_none() {
        return Err(PyValueError::new_err(
            "AsciiTableHDU[[names]] = value: value must be a structured \
             ndarray with named fields"));
    }
    let value_names: Vec<String> = value_names_attr.extract()?;
    let value_names_upper: std::collections::HashSet<String> =
        value_names.iter().map(|n| n.to_uppercase()).collect();

    // Validate names: case-insensitive, no duplicates, all present.
    let mut col_indices: Vec<usize> = Vec::with_capacity(names.len());
    let mut seen_upper: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for name in names {
        let name_u = name.to_uppercase();
        if !seen_upper.insert(name_u.clone()) {
            return Err(PyValueError::new_err(format!(
                "AsciiTableHDU[[names]] = value: duplicate column name '{}'",
                name)));
        }
        let idx = columns.iter()
            .position(|c| c.name.to_uppercase() == name_u)
            .ok_or_else(|| PyValueError::new_err(format!(
                "AsciiTableHDU[[names]] = value: no column named '{}'",
                name)))?;
        if !value_names_upper.contains(&name_u) {
            return Err(PyValueError::new_err(format!(
                "AsciiTableHDU[[names]] = value: value structured dtype \
                 is missing field '{}'", name)));
        }
        col_indices.push(idx);
    }

    // Walk named columns and route each through the per-column writer.
    // Pull the field view through ascontiguousarray so the iteration
    // stride matches the column dtype rather than the parent record
    // itemsize — same defensive pattern as BINTABLE multi-column.
    for (name, &col_idx) in names.iter().zip(col_indices.iter()) {
        let field_view = value.get_item(name.as_str())?;
        let per_col =
            np.call_method1("ascontiguousarray", (field_view,))?;
        write_ascii_table_one_column(
            py, super_, &columns[col_idx], &per_col, nrows, row_width,
            data_offset)?;
    }
    Ok(())
}

// One cell at (row, col).  Used by subset paths (`hdu["col"][row] = v`
// and `hdu[["a","b"]][row] = record`).  Promote value to a length-1
// ndarray so format_one_cell can index it; write byte_width bytes
// at row_offset + col.byte_offset.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_cell(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
    nrows: usize,
    row_width: usize,
    row_idx_signed: i64,
    col_name: &str,
    value: &Bound<'_, PyAny>,
    data_offset: u64,
) -> PyResult<()> {
    let r = normalize_row_index(row_idx_signed, nrows)?;
    let col = find_column_by_name(columns, col_name)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (value,))?;
    let arr_1d = np.call_method1("atleast_1d", (arr,))?;
    let shape: Vec<usize> = arr_1d.getattr("shape")?.extract()?;
    if shape != [1usize] {
        return Err(PyValueError::new_err(format!(
            "AsciiTableHDU cell write: expected scalar value, got shape \
             {:?}", shape)));
    }

    let mut cell_buf = vec![b' '; col.byte_width];
    format_one_cell(py, col, &arr_1d, 0, &mut cell_buf)?;

    let file_off = data_offset
        + (r as u64) * row_width as u64
        + col.byte_offset as u64;
    let mut guard = lock_file(&super_.file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let io_err = |e: std::io::Error| PyIOError::new_err(e.to_string());
    f.seek(SeekFrom::Start(file_off)).map_err(io_err)?;
    if let Err(e) = f.write_all(&cell_buf) {
        super_.tainted.store(true, Ordering::Release);
        return Err(io_err(e));
    }
    if let Err(e) = f.flush() {
        super_.tainted.store(true, Ordering::Release);
        return Err(io_err(e));
    }
    Ok(())
}

// Resolve a __setitem__ rows key (int / slice / iterable of ints) to
// a normalized Vec<usize>.  Used by the subset paths to flatten any
// of the three row-key shapes into one per-row loop.  Second tuple
// element is True when the caller passed a bare int (so the value
// must be a scalar / record rather than an ndarray of length 1).
fn resolve_rows_key(
    rows: &Bound<'_, PyAny>, nrows: usize,
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
            if r < 0 || r >= nrows as i64 {
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

// hdu["name"][rows] = value.  Per-cell loop through setitem_cell.
// Cards / meta are re-snapshotted in the dispatcher (the parent's
// header changes between meta calls are cheap via the version cache).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_one_column_at_rows(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
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
        return setitem_cell(
            py, super_, columns, nrows, row_width,
            row_indices[0] as i64, col_name, value, data_offset);
    }
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
        setitem_cell(
            py, super_, columns, nrows, row_width,
            r as i64, col_name, &cell_value, data_offset)?;
    }
    Ok(())
}

// hdu[["a","b"]][rows] = value.  Per-row × per-column loop through
// setitem_cell.  For int row: value is a structured scalar / shape-(1,)
// ndarray.  For slice/fancy: value is a structured ndarray of length
// = len(row indices), with fields containing each subset column.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_column_subset_at_rows(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
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
            setitem_cell(
                py, super_, columns, nrows, row_width,
                r as i64, name, &cell, data_offset)?;
        }
    }
    Ok(())
}

// What kind of selection the user passed to AsciiTableHDU.__setitem__.
// Mirrors the read-side AsciiTableKey + BINTABLE's SetItemKey.
pub(crate) enum AsciiSetItemKey {
    SingleRow(i64),
    RowSlice,
    SingleColumn(String),
    FancyRows(Vec<i64>),
    MultiColumns(Vec<String>),
}

pub(crate) fn classify_ascii_setitem_key(
    key: &Bound<'_, PyAny>,
) -> PyResult<AsciiSetItemKey> {
    if key.is_instance_of::<PySlice>() {
        return Ok(AsciiSetItemKey::RowSlice);
    }
    if let Some(name) = try_extract_column_name(key)? {
        return Ok(AsciiSetItemKey::SingleColumn(name));
    }
    if !key.is_instance_of::<PyBool>() {
        if let Ok(idx) = key.extract::<i64>() {
            return Ok(AsciiSetItemKey::SingleRow(idx));
        }
    }
    let iter = key.try_iter().map_err(|_| PyValueError::new_err(
        "AsciiTableHDU[key] = value: key must be an int, slice, column \
         name, iterable of ints (fancy rows), or iterable of str \
         (column subset)"
    ))?;
    let items: Vec<Bound<'_, PyAny>> = iter.collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Err(PyValueError::new_err(
            "AsciiTableHDU[key] = value: empty sequence is ambiguous \
             (rows or columns?)"));
    }
    let first = &items[0];
    if try_extract_column_name(first)?.is_some() {
        let names: Vec<String> = items.iter()
            .map(|i| try_extract_column_name(i)?.ok_or_else(|| {
                PyValueError::new_err(
                    "AsciiTableHDU[key] = value: column-name sequence \
                     contains non-string elements")
            }))
            .collect::<PyResult<_>>()?;
        Ok(AsciiSetItemKey::MultiColumns(names))
    } else if !first.is_instance_of::<PyBool>()
        && first.extract::<i64>().is_ok()
    {
        let rows: Vec<i64> = items.iter()
            .map(|i| {
                if i.is_instance_of::<PyBool>() {
                    return Err(PyValueError::new_err(
                        "AsciiTableHDU[key] = value: row-index sequence \
                         contains a bool"));
                }
                i.extract::<i64>().map_err(|_| PyValueError::new_err(
                    "AsciiTableHDU[key] = value: row-index sequence mixes \
                     ints and non-ints"))
            })
            .collect::<PyResult<_>>()?;
        Ok(AsciiSetItemKey::FancyRows(rows))
    } else {
        Err(PyValueError::new_err(
            "AsciiTableHDU[key] = value: sequence must be all int (rows) \
             or all str (columns)"))
    }
}

// Try to extract `obj` as a string-like column name: str, bytes,
// numpy.str_, or numpy.bytes_.  Returns Ok(None) for anything else.
// Same shape as `crate::hdu_table::try_extract_column_name`.
fn try_extract_column_name(
    obj: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(None);
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(Some(obj.extract::<String>()?));
    }
    if obj.is_instance_of::<PyBytes>() {
        let b: Vec<u8> = obj.extract()?;
        if !b.iter().all(|c| c.is_ascii()) {
            return Err(PyValueError::new_err(
                "bytes-like column name contains non-ASCII bytes"));
        }
        return Ok(Some(String::from_utf8(b).unwrap()));
    }
    Ok(None)
}
