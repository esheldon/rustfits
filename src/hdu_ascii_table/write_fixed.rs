// ASCII-table bulk write + append.
//
// Strategy: extract per-column numpy arrays from the input (structured
// / dict / list+names), validate each against the expected width and
// dtype, then iterate rows in ~1 MiB chunks formatting each cell into
// a text buffer.  Write each chunk; pad the final block with ASCII
// spaces (NOT NUL — TABLE-extension padding is space per FITS spec).
//
// Per-cell access goes through Python-level indexing rather than
// RawBuffer + bulk memcpy because every cell needs text formatting
// anyway.  Performance is not the primary goal — ASCII tables are
// rare and intrinsically slower than BINTABLE.  If a real workload
// shows a hot path, the int / float / S / U paths can switch to
// RawBuffer-backed per-element loops without changing the public API.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::Ordering;

use crate::common::{
    lock_file, shift_file_tail_and_update_offsets, zero_fill_range,
    BLOCK_SIZE,
};
use crate::hdu::HDU;
use crate::header::card_int;
use crate::hdu_image::{round_up_to_block, serialize_header_to_disk_bytes};

use super::columns::AsciiColumn;
use super::format::{
    format_a_field, format_d_field, format_e_field, format_f_field,
    format_int_field,
};

// Streaming row-chunk budget — same convention as the read path.
const WRITE_CHUNK_TARGET_BYTES: usize = 1 << 20;

// Determine the number of rows in the user's input.  Mirrors the
// hdu_table::determine_input_nrows contract: structured -> len(data);
// dict -> length of first value; list/tuple -> length of first element.
pub(crate) fn determine_input_nrows(
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    if data.is_instance_of::<PyDict>() {
        if names.is_some() {
            return Err(PyValueError::new_err(
                "names= is only valid with list/tuple data, not dict",
            ));
        }
        let d = data.cast::<PyDict>()?;
        let values = d.values();
        if values.is_empty() {
            return Err(PyValueError::new_err("data dict is empty"));
        }
        let first = values.get_item(0)?;
        Ok(first.len()?)
    } else if data.is_instance_of::<PyList>()
        || data.is_instance_of::<PyTuple>()
    {
        if names.is_none() {
            return Err(PyValueError::new_err(
                "when data is a list/tuple, names= is required",
            ));
        }
        if data.len()? == 0 {
            return Err(PyValueError::new_err("data list/tuple is empty"));
        }
        let first = data.get_item(0)?;
        Ok(first.len()?)
    } else {
        if names.is_some() {
            return Err(PyValueError::new_err(
                "names= is only valid with list/tuple data, not a \
                 structured ndarray",
            ));
        }
        Ok(data.len()?)
    }
}

// Extract one numpy ndarray per column from the user's input,
// preserving on-disk column order.  Length is validated against
// expected_nrows.  Field-name match is case-insensitive (matches the
// read-side convention); missing or extra fields raise.
pub(crate) fn extract_per_column_arrays<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    names: Option<&Bound<'py, PyAny>>,
    columns: &[AsciiColumn],
    expected_nrows: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let np = py.import("numpy")?;
    let asarray = |v: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        np.call_method1("asanyarray", (v,))
    };

    // Dict input: {name: ndarray}.
    if let Ok(d) = data.cast::<PyDict>() {
        if names.is_some() {
            return Err(PyValueError::new_err(
                "names= is not valid with dict input",
            ));
        }
        let mut by_name_upper: std::collections::HashMap<
            String,
            Bound<'py, PyAny>,
        > = std::collections::HashMap::new();
        for (k, v) in d.iter() {
            let key: String = k.extract().map_err(|_| {
                PyValueError::new_err(
                    "dict keys must be strings (column names)",
                )
            })?;
            by_name_upper.insert(key.to_ascii_uppercase(), v);
        }
        let mut out = Vec::with_capacity(columns.len());
        let mut matched: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for col in columns {
            let key = col.name.to_ascii_uppercase();
            let arr_val = by_name_upper.get(&key).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "data dict missing column '{}'", col.name
                ))
            })?;
            matched.insert(key);
            let arr = asarray(arr_val.clone())?;
            check_column_len(&arr, &col.name, expected_nrows)?;
            out.push(arr);
        }
        // Extra dict keys are user error (typo or stale name).
        if matched.len() != by_name_upper.len() {
            for k in by_name_upper.keys() {
                if !matched.contains(k) {
                    return Err(PyValueError::new_err(format!(
                        "data dict has extra column '{}' not in the \
                         table schema", k
                    )));
                }
            }
        }
        return Ok(out);
    }

    // List/tuple input + names=.
    if data.is_instance_of::<PyList>()
        || data.is_instance_of::<PyTuple>()
    {
        let names_obj = names.ok_or_else(|| {
            PyValueError::new_err(
                "when data is a list/tuple, names= is required",
            )
        })?;
        let name_list: Vec<String> = names_obj.extract().map_err(|_| {
            PyValueError::new_err("names= must be a list/tuple of str")
        })?;
        let data_seq: Vec<Bound<'py, PyAny>> = data.extract()?;
        if data_seq.len() != name_list.len() {
            return Err(PyValueError::new_err(format!(
                "names= length {} does not match data length {}",
                name_list.len(), data_seq.len(),
            )));
        }
        // Reorder to match write_columns; same case-insensitive
        // lookup as the dict path.
        let mut by_name_upper: std::collections::HashMap<
            String,
            Bound<'py, PyAny>,
        > = std::collections::HashMap::new();
        for (n, v) in name_list.iter().zip(data_seq.iter()) {
            by_name_upper.insert(n.to_ascii_uppercase(), v.clone());
        }
        let mut out = Vec::with_capacity(columns.len());
        for col in columns {
            let key = col.name.to_ascii_uppercase();
            let arr_val = by_name_upper.get(&key).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "names= missing column '{}'", col.name
                ))
            })?;
            let arr = asarray(arr_val.clone())?;
            check_column_len(&arr, &col.name, expected_nrows)?;
            out.push(arr);
        }
        return Ok(out);
    }

    // Structured ndarray input.
    let ndarray = np.getattr("ndarray")?;
    if !data.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(
            "data must be a structured numpy ndarray, a dict \
             {name: ndarray}, or a list/tuple of ndarrays with \
             names=[...]",
        ));
    }
    let dtype = data.getattr("dtype")?;
    let names_attr = dtype.getattr("names")?;
    if names_attr.is_none() {
        return Err(PyValueError::new_err(
            "data ndarray must be structured (dtype must have named \
             fields)",
        ));
    }
    let field_names: Vec<String> = names_attr.extract()?;
    let mut field_set_upper: std::collections::HashSet<String> =
        field_names.iter().map(|n| n.to_ascii_uppercase()).collect();
    let mut out = Vec::with_capacity(columns.len());
    for col in columns {
        let key = col.name.to_ascii_uppercase();
        // Find by case-insensitive match against the input field
        // names (Python dict access is case-sensitive).
        let actual_name = field_names.iter().find(|n| {
            n.to_ascii_uppercase() == key
        }).ok_or_else(|| PyValueError::new_err(format!(
            "data ndarray missing column '{}'", col.name
        )))?;
        field_set_upper.remove(&key);
        let arr = data.get_item(actual_name.as_str())?;
        check_column_len(&arr, &col.name, expected_nrows)?;
        out.push(arr);
    }
    // Reject extras (consistent with BINTABLE's strict mode).
    if !field_set_upper.is_empty() {
        let extras: Vec<&String> = field_set_upper.iter().collect();
        return Err(PyValueError::new_err(format!(
            "data ndarray has extra columns not in the table schema: {:?}",
            extras,
        )));
    }
    Ok(out)
}

fn check_column_len(
    arr: &Bound<'_, PyAny>, col_name: &str, expected_nrows: usize,
) -> PyResult<()> {
    let n = arr.len()?;
    if n != expected_nrows {
        return Err(PyValueError::new_err(format!(
            "column '{}': expected {} rows, got {}",
            col_name, expected_nrows, n,
        )));
    }
    Ok(())
}

// Detect whether a column declares the unsigned-int trick on the
// write side: I letter, TSCAL=1, TZERO=2^63.  cfitsio uses the same
// 2^63 bias regardless of the user's int width (rustfits always maps
// Iw -> i8 on read, so the trick targets i8/i64-range).
pub(crate) fn is_unsigned_trick(col: &AsciiColumn) -> bool {
    col.tform_letter == 'I' && col.tscal == 1.0
        && col.tzero == 9223372036854775808.0
}

// Format one column's cell from per-column ndarray `arr` at
// `row_index` into `cell_dst` (length = col.byte_width).  Dispatches
// on `col.tform_letter`.  For A columns, inspects the cell's Python
// type (bytes vs str) rather than pre-classifying the input dtype.
//
// Shared by `format_row` (bulk write loop) and the single-column /
// single-cell writers in setitem.rs — all three walk over rows and
// need identical per-letter dispatch.
pub(crate) fn format_one_cell(
    py: Python<'_>,
    col: &AsciiColumn,
    arr: &Bound<'_, PyAny>,
    row_index: usize,
    cell_dst: &mut [u8],
) -> PyResult<()> {
    let cell = arr.get_item(row_index)?;
    match col.tform_letter {
        'A' => {
            // Accept bytes or str.  numpy.bytes_ is a bytes subclass;
            // numpy.str_ is a str subclass; both cast cleanly.  Call
            // format_a_field inside each branch so the bytes / str
            // borrow lives long enough.
            if let Ok(b) = cell.cast::<PyBytes>() {
                format_a_field(
                    b.as_bytes(), cell_dst, &col.name, row_index,
                )?;
            } else if let Ok(s) = cell.cast::<PyString>() {
                format_a_field(
                    s.to_str()?.as_bytes(), cell_dst, &col.name,
                    row_index,
                )?;
            } else {
                return Err(PyValueError::new_err(format!(
                    "column '{}' row {}: A field cell must be bytes \
                     or str, got {}",
                    col.name, row_index,
                    cell.get_type().name()?,
                )));
            }
            let _ = py;
        }
        'I' => {
            let value: i64 = if is_unsigned_trick(col) {
                let u: u64 = cell.extract()?;
                (u ^ (1u64 << 63)) as i64
            } else {
                cell.extract()?
            };
            format_int_field(value, cell_dst, &col.name, row_index)?;
        }
        'F' => {
            let v: f64 = cell.extract()?;
            let d = col.decimals.unwrap_or(0);
            format_f_field(v, d, cell_dst, &col.name, row_index)?;
        }
        'E' => {
            let v: f64 = cell.extract()?;
            let d = col.decimals.unwrap_or(0);
            format_e_field(v, d, cell_dst, &col.name, row_index)?;
        }
        'D' => {
            let v: f64 = cell.extract()?;
            let d = col.decimals.unwrap_or(0);
            format_d_field(v, d, cell_dst, &col.name, row_index)?;
        }
        other => unreachable!(
            "unsupported ASCII TFORM letter '{}' on write", other,
        ),
    }
    Ok(())
}

// Format one row's worth of bytes into `dst` (length = row_width).
// Loops over columns and dispatches each cell to `format_one_cell`.
fn format_row(
    py: Python<'_>,
    columns: &[AsciiColumn],
    per_column: &[Bound<'_, PyAny>],
    row_index: usize,
    dst: &mut [u8],
) -> PyResult<()> {
    debug_assert_eq!(
        dst.len(),
        columns.iter().map(|c| c.byte_width).sum::<usize>(),
    );
    for (col, arr) in columns.iter().zip(per_column.iter()) {
        let cell_dst = &mut dst
            [col.byte_offset..col.byte_offset + col.byte_width];
        format_one_cell(py, col, arr, row_index, cell_dst)?;
    }
    Ok(())
}

// Bulk-write `n_rows` rows into the file at `start_offset`.  Caller
// is responsible for the header NAXIS2 card update and any
// file-extent grow — this just formats + writes the row bytes.
//
// On success the last row's tail of the data section is padded to a
// 2880-byte block boundary with ASCII spaces (NOT NUL) per the FITS
// spec for ASCII tables.  The padding is only written when this call
// covers all rows in the data section (start_offset == data_offset
// AND n_rows == total NAXIS2) — partial-section writes leave the
// existing pad untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ascii_table_data(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
    per_column: &[Bound<'_, PyAny>],
    row_width: usize,
    n_rows: usize,
    start_offset: u64,
    pad_to_block: bool,
) -> PyResult<()> {
    if row_width == 0 || n_rows == 0 {
        return Ok(());
    }
    let rows_per_chunk =
        std::cmp::max(1, WRITE_CHUNK_TARGET_BYTES / row_width);
    let mut chunk_buf = vec![b' '; rows_per_chunk * row_width];

    let mut guard = lock_file(&super_.file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let io_err = |e: std::io::Error| PyIOError::new_err(e.to_string());

    f.seek(SeekFrom::Start(start_offset)).map_err(io_err)?;

    let mut row_cursor: usize = 0;
    while row_cursor < n_rows {
        let this_rows = std::cmp::min(rows_per_chunk, n_rows - row_cursor);
        for r in 0..this_rows {
            let row_dst = &mut chunk_buf[r * row_width..(r + 1) * row_width];
            format_row(py, columns, per_column, row_cursor + r, row_dst)?;
        }
        let pre_taint_ok = f.write_all(&chunk_buf[..this_rows * row_width]);
        if let Err(e) = pre_taint_ok {
            // Bytes have been written -> file is now inconsistent.
            super_.tainted.store(true, std::sync::atomic::Ordering::Release);
            return Err(io_err(e));
        }
        row_cursor += this_rows;
    }

    if pad_to_block {
        let total = n_rows * row_width;
        let pad_n = (BLOCK_SIZE - total % BLOCK_SIZE) % BLOCK_SIZE;
        if pad_n > 0 {
            let pad = vec![b' '; pad_n];
            if let Err(e) = f.write_all(&pad) {
                super_.tainted.store(true, std::sync::atomic::Ordering::Release);
                return Err(io_err(e));
            }
        }
    }

    if let Err(e) = f.flush() {
        super_.tainted.store(true, std::sync::atomic::Ordering::Release);
        return Err(io_err(e));
    }
    Ok(())
}

// One-call wrapper: validate input, extract per-column arrays, write
// all rows starting at the data section.  Used by AsciiTableHDU.write
// (which sets pad_to_block=true).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ascii_table_full(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
    n_rows: usize,
    row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    let per_column = extract_per_column_arrays(
        py, data, names, columns, n_rows,
    )?;
    write_ascii_table_data(
        py, super_, columns, &per_column, row_width, n_rows,
        data_offset, /* pad_to_block = */ true,
    )
}

// Append `append_nrows` rows to the end of an existing ASCII table.
// Mirrors hdu_table::append_fixed_only:
//   1. Validate-then-mutate: extract + check per-column ndarrays
//      before any file mutation.
//   2. Grow the file extent if the new data section crosses a block
//      boundary.  For non-last HDUs, shift the trailing tail forward
//      via the shared primitive (bumps later-HDU offsets in lockstep).
//   3. Rewrite the NAXIS2 card to disk first (taint on failure),
//      then commit the in-memory cards.
//   4. Format + write the new rows at `data_offset +
//      current_nrows * row_width`; final-block pad on the last write.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_ascii_table(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
    current_nrows: usize,
    append_nrows: usize,
    new_nrows: usize,
    row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    let per_column = extract_per_column_arrays(
        py, data, names, columns, append_nrows,
    )?;

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

    // Disk-write-before-commit ordering on the NAXIS2 card.  Same
    // taint discipline as the BINTABLE append path.
    let new_card = card_int(
        "NAXIS2", new_nrows as i64, "number of rows in table",
    );
    let cards_guard = super_.cards_write_lock()?;
    let mut new_cards = cards_guard.clone_cards();
    let card_idx = new_cards.iter()
        .position(|c| c.len() >= 6 && c[..6].trim() == "NAXIS2")
        .ok_or_else(|| PyValueError::new_err("header missing NAXIS2"))?;
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
                 reopen the file to recover", e,
            ))
        })?;
        file.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "header flush failed during append: {}; close + \
                 reopen the file to recover", e,
            ))
        })?;
    }
    cards_guard.commit(new_cards);

    // Write the new rows at the tail of the data section.  Pad the
    // final block to the 2880-byte boundary (this call covers
    // everything new).
    let append_offset = data_offset + current_data_bytes;
    write_ascii_table_data(
        py, super_, columns, &per_column, row_width, append_nrows,
        append_offset, /* pad_to_block = */ true,
    )
}

// Strided per-row writer.  Used by __setitem__ for stepped slices
// (`hdu[a:b:s] = arr` with s != 1) and fancy-row writes
// (`hdu[[i,j,k]] = arr`).  For each input row, formats the full row
// into a temp buffer, seeks to `data_offset + disk_row * row_width`,
// writes the bytes.  No final-block pad: this is a partial overwrite
// of an already-padded data section.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ascii_table_strided(
    py: Python<'_>,
    super_: &HDU,
    columns: &[AsciiColumn],
    per_column: &[Bound<'_, PyAny>],
    row_width: usize,
    row_indices: &[usize],
    data_offset: u64,
) -> PyResult<()> {
    if row_indices.is_empty() {
        return Ok(());
    }
    let mut row_buf = vec![b' '; row_width];
    let mut guard = lock_file(&super_.file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let io_err = |e: std::io::Error| PyIOError::new_err(e.to_string());

    for (input_row, &disk_row) in row_indices.iter().enumerate() {
        for b in row_buf.iter_mut() { *b = b' '; }
        format_row(py, columns, per_column, input_row, &mut row_buf)?;
        let file_off = data_offset + (disk_row as u64) * row_width as u64;
        f.seek(SeekFrom::Start(file_off)).map_err(io_err)?;
        if let Err(e) = f.write_all(&row_buf) {
            super_.tainted.store(true, Ordering::Release);
            return Err(io_err(e));
        }
    }
    if let Err(e) = f.flush() {
        super_.tainted.store(true, Ordering::Release);
        return Err(io_err(e));
    }
    Ok(())
}

// Whole-column writer.  Per-row seek + write of just this column's
// byte_width bytes; the other columns' bytes are preserved by virtue
// of never being touched.  Cost is O(nrows) seek+write syscalls of
// byte_width each; same trade-off as BINTABLE's write_table_one_column
// (better than strip RMW when byte_width << row_width, which is
// typical).
pub(crate) fn write_ascii_table_one_column(
    py: Python<'_>,
    super_: &HDU,
    col: &AsciiColumn,
    arr: &Bound<'_, PyAny>,
    nrows: usize,
    row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    if nrows == 0 {
        return Ok(());
    }
    let mut cell_buf = vec![b' '; col.byte_width];
    let mut guard = lock_file(&super_.file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let io_err = |e: std::io::Error| PyIOError::new_err(e.to_string());

    for r in 0..nrows {
        for b in cell_buf.iter_mut() { *b = b' '; }
        format_one_cell(py, col, arr, r, &mut cell_buf)?;
        let file_off =
            data_offset + (r * row_width + col.byte_offset) as u64;
        f.seek(SeekFrom::Start(file_off)).map_err(io_err)?;
        if let Err(e) = f.write_all(&cell_buf) {
            super_.tainted.store(true, Ordering::Release);
            return Err(io_err(e));
        }
    }
    if let Err(e) = f.flush() {
        super_.tainted.store(true, Ordering::Release);
        return Err(io_err(e));
    }
    Ok(())
}
