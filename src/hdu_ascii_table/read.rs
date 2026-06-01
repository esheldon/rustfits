// ASCII-table read path: build the numpy structured dtype, plan + walk
// the requested rows, parse each field's text into the matching numpy
// slot.
//
// Phase 2 surface:
//   * read_ascii_table(rows=, columns=, scale=, mask_null=) for the
//     full structured-ndarray read with optional row + column subsets
//     and MaskedArray returns.
//   * read_one_column(name, rows=, scale=, mask_null=) for single-column
//     plain-ndarray reads.
//   * Shared run planner mirroring hdu_table/read.rs::plan_runs +
//     process_runs (duplicated, not generalized — see CLAUDE.md
//     "ASCII tables" plan for why).
//
// Re-uses `crate::hdu_table::resolve_rows` directly: row-index semantics
// are identical and the helper is purely arithmetic over the row count.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::io::{Read, Seek, SeekFrom};

use crate::common::{lock_file, FileHandle, RawBuffer};
use crate::hdu_table::resolve_rows;

use super::columns::{
    ascii_scaled_output_dtype, ascii_scaling_kind, AsciiColumn,
    AsciiScalingKind,
};
use super::format::{matches_tnull, parse_float_field, parse_int_field, trim_ascii};
use super::meta::AsciiTableMeta;

// Maximum decimal places a 'F' / 'E' column carries while still
// fitting in f4.  f4 has a 24-bit mantissa ~= 7.2 significant decimal
// digits; columns with up to 7 decimal places land in f4.  Anything
// wider — including cfitsio's `E26.17` default emission for an f8
// numpy column written via `fits_write_col`'s automatic-format path
// (which fitsio.write_table uses for `table_type='ascii'`) — promotes
// to f8 so the precision survives.  This rule keeps astropy's narrower
// `E10.4` columns at f4 while handling cfitsio's wider emission.
const F_E_F4_MAX_DECIMALS: usize = 7;

// Map an ASCII column to its numpy structured-field dtype string and
// (currently empty) shape.  ASCII tables have no per-cell shape —
// every field is a scalar, including A which is U<width>.
fn ascii_field_dtype(col: &AsciiColumn, scale: bool) -> String {
    let kind = if scale { ascii_scaling_kind(col) } else { AsciiScalingKind::None };
    match kind {
        AsciiScalingKind::None => match col.tform_letter {
            'A' => format!("U{}", col.byte_width),
            'I' => "i8".to_string(),
            'F' | 'E' => {
                let d = col.decimals.unwrap_or(0);
                if d <= F_E_F4_MAX_DECIMALS { "f4" } else { "f8" }.to_string()
            }
            'D' => "f8".to_string(),
            _ => unreachable!(
                "unsupported ASCII TFORM letter '{}'", col.tform_letter
            ),
        },
        AsciiScalingKind::UnsignedTrick | AsciiScalingKind::General => {
            ascii_scaled_output_dtype(col.tform_letter, kind).to_string()
        }
    }
}

// Build the numpy structured dtype for an ASCII table.
pub(crate) fn build_ascii_numpy_dtype(
    py: Python<'_>,
    columns: &[AsciiColumn],
    scale: bool,
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let np_dtype = numpy.getattr("dtype")?;
    let fields = PyList::empty(py);
    for col in columns {
        let dtype_str = ascii_field_dtype(col, scale);
        let tuple = PyTuple::new(py, [
            col.name.clone().into_pyobject(py)?.into_any(),
            dtype_str.into_pyobject(py)?.into_any(),
        ])?;
        fields.append(tuple)?;
    }
    Ok(np_dtype.call1((fields,))?.unbind())
}

// Per-column numpy field layout: (byte offset within record, bytes
// within record).  numpy may pad fields; we trust numpy to tell us
// where each one lives.
fn numpy_field_layout(
    py: Python<'_>,
    dtype: &Bound<'_, PyAny>,
    columns: &[AsciiColumn],
) -> PyResult<Vec<(usize, usize)>> {
    let fields = dtype.getattr("fields")?;
    let mut out = Vec::with_capacity(columns.len());
    for col in columns {
        let key = col.name.clone().into_pyobject(py)?;
        let info = fields.get_item(key)?;
        let sub_dtype = info.get_item(0)?;
        let offset: usize = info.get_item(1)?.extract()?;
        let sub_itemsize: usize = sub_dtype.getattr("itemsize")?.extract()?;
        out.push((offset, sub_itemsize));
    }
    Ok(out)
}

// Pack one ASCII string (`src`) into a numpy U<n> slot.  Each codepoint
// is written as 4 native-endian bytes (UCS-4).  `dst` is pre-zeroed by
// np.zeros (whole-array) or by the per-row scratch buffer, so we only
// write the live codepoints — trailing zero bytes are already the
// numpy null-terminator.  Leading spaces preserved; trailing spaces
// stripped (matches FITS A semantics + BINTABLE convert_a_cell).
fn pack_u_cell_from_ascii(
    src: &[u8], dst: &mut [u8], col_name: &str, row_index: usize,
) -> PyResult<()> {
    // rstrip ASCII spaces.
    let mut end = src.len();
    while end > 0 && src[end - 1] == b' ' {
        end -= 1;
    }
    let trimmed = &src[..end];
    if dst.len() < trimmed.len() * 4 {
        // Should not happen — dst was sized to col.byte_width * 4.
        return Err(PyValueError::new_err(format!(
            "internal: U slot too small for column '{}' row {}",
            col_name, row_index,
        )));
    }
    for (i, &b) in trimmed.iter().enumerate() {
        if !b.is_ascii() {
            return Err(PyValueError::new_err(format!(
                "column '{}' row {}: A field contains non-ASCII byte \
                 0x{:02X} at position {}",
                col_name, row_index, b, i,
            )));
        }
        let cp_bytes = (b as u32).to_ne_bytes();
        dst[i * 4..i * 4 + 4].copy_from_slice(&cp_bytes);
    }
    Ok(())
}

// Pack one ASCII string into a numpy S<n> (bytes) slot — used by
// `read_one_column(as_bytes=True)`.  Verbatim copy of the on-disk
// bytes (no trim, no validation).  numpy treats trailing NUL bytes
// as string terminator.
fn pack_s_cell_from_ascii(src: &[u8], dst: &mut [u8]) {
    // Pre-zero (caller passed an np.zeros'd buffer slice).  Copy as-is.
    let n = src.len().min(dst.len());
    dst[..n].copy_from_slice(&src[..n]);
}

// Apply scaling to a parsed float value and write the result into
// `dst`.  Caller picked the kind upfront (constant per column).
fn write_scaled_float(
    kind: AsciiScalingKind, tscal: f64, tzero: f64, value: f64,
    dst: &mut [u8],
) {
    match kind {
        AsciiScalingKind::General => {
            let physical = tscal * value + tzero;
            // Always f8 destination for General (8 bytes).
            dst[0..8].copy_from_slice(&physical.to_ne_bytes());
        }
        AsciiScalingKind::None => {
            // Numeric dst is either f4 (E/F input) or f8 (D input).
            if dst.len() == 4 {
                let v32 = value as f32;
                dst.copy_from_slice(&v32.to_ne_bytes());
            } else {
                dst.copy_from_slice(&value.to_ne_bytes());
            }
        }
        AsciiScalingKind::UnsignedTrick => unreachable!(
            "unsigned trick applies to ASCII I only, not float"
        ),
    }
}

// Apply scaling to a parsed integer value and write the result into
// `dst`.  Caller picked the kind upfront (constant per column).
fn write_scaled_int(
    kind: AsciiScalingKind, tscal: f64, tzero: f64, value: i64,
    dst: &mut [u8],
) {
    match kind {
        AsciiScalingKind::None => {
            // Default I dest is i8 (8 bytes).
            dst.copy_from_slice(&value.to_ne_bytes());
        }
        AsciiScalingKind::UnsignedTrick => {
            // Reverse the K-style sign bias (TZERO=2^63).  Output dtype
            // is u8 (8 bytes).
            let physical: u64 = (value as u64).wrapping_add(1u64 << 63);
            dst.copy_from_slice(&physical.to_ne_bytes());
        }
        AsciiScalingKind::General => {
            // Promote to f8 (8 bytes).
            let physical = tscal * (value as f64) + tzero;
            dst.copy_from_slice(&physical.to_ne_bytes());
        }
    }
}

// Convert one ASCII-table field's bytes into the matching numpy slot.
fn convert_ascii_cell(
    col: &AsciiColumn,
    kind: AsciiScalingKind,
    src: &[u8],
    dst: &mut [u8],
    row_index: usize,
) -> PyResult<()> {
    match col.tform_letter {
        'A' => pack_u_cell_from_ascii(src, dst, &col.name, row_index),
        'I' => {
            let v = parse_int_field(src, &col.name, row_index)?;
            write_scaled_int(kind, col.tscal, col.tzero, v, dst);
            Ok(())
        }
        'F' | 'E' | 'D' => {
            let v = parse_float_field(src, &col.name, row_index)?;
            write_scaled_float(kind, col.tscal, col.tzero, v, dst);
            Ok(())
        }
        _ => unreachable!(
            "unsupported ASCII TFORM letter '{}'", col.tform_letter
        ),
    }
}

// Resolve a user-supplied list of column names against the full column
// list parsed from the header.  Case-insensitive lookup; duplicates and
// unknown names rejected up front.  Returns the matching AsciiColumns
// in the user's requested order (byte_offset / byte_width still point
// at this column's slot in the on-disk row, so the per-row converter
// can subset directly).
pub(crate) fn resolve_ascii_columns(
    all: &[AsciiColumn],
    requested: &[String],
) -> PyResult<Vec<AsciiColumn>> {
    if requested.is_empty() {
        return Err(PyValueError::new_err(
            "columns= requested an empty list; pass None for all columns",
        ));
    }
    let mut seen: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(requested.len());
    let mut out = Vec::with_capacity(requested.len());
    for name in requested {
        let key = name.trim().to_ascii_uppercase();
        if !seen.insert(key.clone()) {
            return Err(PyValueError::new_err(format!(
                "duplicate column name in request: '{}'", name
            )));
        }
        let matched = all.iter()
            .find(|c| c.name.eq_ignore_ascii_case(name.trim()));
        match matched {
            Some(col) => out.push(col.clone()),
            None => {
                let available: Vec<&str> =
                    all.iter().map(|c| c.name.as_str()).collect();
                return Err(PyValueError::new_err(format!(
                    "unknown column name: '{}'.  Available columns: {:?}",
                    name, available,
                )));
            }
        }
    }
    Ok(out)
}

// Target chunk size for streaming reads — same convention as the
// BINTABLE planner (~1 MiB across rows_per_chunk * row_width).
const READ_CHUNK_TARGET_BYTES: usize = 1 << 20;

// One contiguous span of disk rows to read in a single I/O.
// `output_indices[i]` is the position in the output array for the i-th
// row of this run.  When rows=None there is one run covering everything
// with output_indices = [0, 1, ..., n_rows-1]; with rows=, runs come
// from coalescing the sorted-unique disk indices.
struct RunPlan {
    start_disk_row: usize,
    len: usize,
    output_indices: Vec<usize>,
}

// Build the run plan from a `rows=` argument.  When rows is None, one
// run covers the whole table.  Otherwise: sort the user-order-deduped
// indices, group contiguous runs, and carry the output-position list
// per run so each row read knows where to land in the user's
// requested order.  Mirrors hdu_table/read.rs::plan_runs exactly.
fn plan_runs(
    rows_arg: Option<&Bound<'_, PyAny>>,
    n_rows: usize,
) -> PyResult<(usize, Vec<RunPlan>)> {
    match rows_arg {
        None => {
            if n_rows == 0 {
                return Ok((0, Vec::new()));
            }
            Ok((n_rows, vec![RunPlan {
                start_disk_row: 0,
                len: n_rows,
                output_indices: (0..n_rows).collect(),
            }]))
        }
        Some(arg) => {
            let user_unique = resolve_rows(arg, n_rows)?;
            let n_out = user_unique.len();
            let mut indexed: Vec<(usize, usize)> = user_unique.iter()
                .enumerate()
                .map(|(i, &r)| (r, i))
                .collect();
            indexed.sort_by_key(|&(r, _)| r);
            let mut runs = Vec::new();
            let mut i = 0;
            while i < indexed.len() {
                let mut j = i + 1;
                while j < indexed.len()
                    && indexed[j].0 == indexed[j - 1].0 + 1
                {
                    j += 1;
                }
                let start = indexed[i].0;
                let len = j - i;
                let output_indices: Vec<usize> =
                    indexed[i..j].iter().map(|&(_, o)| o).collect();
                runs.push(RunPlan { start_disk_row: start, len, output_indices });
                i = j;
            }
            Ok((n_out, runs))
        }
    }
}

// Walk the run plan, doing one seek + one chunked sequential read per
// run, invoking `on_row` once per row with (src_row_bytes, disk_row,
// output_row).  Owns the file handle and the chunk buffer.  Mirrors
// hdu_table/read.rs::process_runs.
fn process_runs<F>(
    file_handle: &FileHandle,
    runs: &[RunPlan],
    data_offset: u64,
    row_width: usize,
    rows_per_chunk: usize,
    mut on_row: F,
) -> PyResult<()>
where
    F: FnMut(&[u8], usize, usize) -> PyResult<()>,
{
    let mut chunk_buf = vec![0u8; rows_per_chunk * row_width];
    let mut guard = lock_file(file_handle)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    for run in runs {
        let run_offset_bytes =
            data_offset + (run.start_disk_row * row_width) as u64;
        f.seek(SeekFrom::Start(run_offset_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let mut local_offset = 0usize;
        while local_offset < run.len {
            let this_rows =
                std::cmp::min(rows_per_chunk, run.len - local_offset);
            f.read_exact(&mut chunk_buf[..this_rows * row_width])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            for r_local in 0..this_rows {
                let in_run = local_offset + r_local;
                let disk_row = run.start_disk_row + in_run;
                let output_row = run.output_indices[in_run];
                let src_row = &chunk_buf
                    [r_local * row_width..(r_local + 1) * row_width];
                on_row(src_row, disk_row, output_row)?;
            }
            local_offset += this_rows;
        }
    }
    Ok(())
}

// MaskedArray helpers ------------------------------------------------

// Wrap `data` in numpy.ma.MaskedArray.  None for mask passes np.ma.nomask.
fn wrap_masked(
    py: Python<'_>,
    data: Bound<'_, PyAny>,
    mask: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let ma = py.import("numpy")?.getattr("ma")?;
    let mask_obj = match mask {
        Some(m) => m,
        None => ma.getattr("nomask")?,
    };
    Ok(ma.call_method1("MaskedArray", (data, mask_obj))?.unbind())
}

// Build a numpy structured bool dtype mirroring the data dtype's
// per-column shapes but with '?' for every field.
fn build_mask_dtype(
    py: Python<'_>,
    columns: &[AsciiColumn],
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let np_dtype = numpy.getattr("dtype")?;
    let fields = PyList::empty(py);
    for col in columns {
        let tuple = PyTuple::new(py, [
            col.name.clone().into_pyobject(py)?.into_any(),
            "?".into_pyobject(py)?.into_any(),
        ])?;
        fields.append(tuple)?;
    }
    Ok(np_dtype.call1((fields,))?.unbind())
}

// Allocated mask array + pre-computed layout for the row loop.
// Returned by allocate_mask_array; None when mask_null=False OR no
// selected column carries TNULL (the caller still wraps the data in
// MaskedArray with nomask for consistent return type — zero allocation).
struct MaskArray<'py> {
    arr: Bound<'py, PyAny>,
    itemsize: usize,
    field_layout: Vec<(usize, usize)>,
}

fn allocate_mask_array<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    columns: &[AsciiColumn],
    n_out: usize,
    mask_null: bool,
) -> PyResult<Option<MaskArray<'py>>> {
    if !mask_null || !columns.iter().any(|c| c.tnull.is_some()) {
        return Ok(None);
    }
    let mask_dtype = build_mask_dtype(py, columns)?;
    let arr = np.call_method1("zeros", (n_out, mask_dtype.bind(py)))?;
    let mdt = arr.getattr("dtype")?;
    let itemsize: usize = mdt.getattr("itemsize")?.extract()?;
    // The mask layout uses the same field names as the data; reuse
    // numpy_field_layout since AsciiColumn carries the name.
    let field_layout = numpy_field_layout(py, &mdt, columns)?;
    Ok(Some(MaskArray { arr, itemsize, field_layout }))
}

// Write per-element TNULL masks for one row across all selected
// columns.  Columns without TNULL are skipped (their bytes were
// pre-zeroed by np.zeros, so False is already correct).
fn write_row_mask(
    columns: &[AsciiColumn],
    field_layout: &[(usize, usize)],
    src_row: &[u8],
    m_row: &mut [u8],
) {
    for (col_idx, col) in columns.iter().enumerate() {
        if let Some(tnull) = &col.tnull {
            let src = &src_row[col.byte_offset
                ..col.byte_offset + col.byte_width];
            let (off, _w) = field_layout[col_idx];
            // Mask field is a single bool byte per column.
            m_row[off] = matches_tnull(trim_ascii(src), tnull) as u8;
        }
    }
}

// Read a structured ndarray from the ASCII table.  All four args
// (rows=, columns=, scale=, mask_null=) accepted; matches the
// hdu_table::read_table contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_ascii_table(
    py: Python<'_>,
    meta: &AsciiTableMeta,
    data_offset: u64,
    file_handle: &FileHandle,
    rows_arg: Option<&Bound<'_, PyAny>>,
    columns_requested: Option<Vec<String>>,
    scale: bool,
    mask_null: bool,
) -> PyResult<Py<PyAny>> {
    let n_rows = meta.nrows as usize;
    let row_width = meta.row_width as usize;
    let columns: Vec<AsciiColumn> = match columns_requested {
        None => meta.columns.clone(),
        Some(names) => resolve_ascii_columns(&meta.columns, &names)?,
    };

    let (n_out, runs) = plan_runs(rows_arg, n_rows)?;

    let scaling_kinds: Vec<AsciiScalingKind> = columns.iter()
        .map(|c| if scale { ascii_scaling_kind(c) } else { AsciiScalingKind::None })
        .collect();

    let dtype = build_ascii_numpy_dtype(py, &columns, scale)?;
    let np = py.import("numpy")?;
    // np.zeros so U fields start null-terminated and numeric fields
    // default to 0 — matches FITS "undefined value" for blank fields
    // without us having to write anything per cell.
    let arr = np.call_method1("zeros", (n_out, dtype.bind(py)))?;
    let mask = allocate_mask_array(py, &np, &columns, n_out, mask_null)?;

    if n_out == 0 || row_width == 0 {
        return if mask_null {
            wrap_masked(py, arr, mask.map(|m| m.arr))
        } else {
            Ok(arr.unbind())
        };
    }

    let arr_dtype = arr.getattr("dtype")?;
    let itemsize: usize = arr_dtype.getattr("itemsize")?.extract()?;
    let field_layout = numpy_field_layout(py, &arr_dtype, &columns)?;

    let rows_per_chunk = std::cmp::max(1, READ_CHUNK_TARGET_BYTES / row_width);

    {
        let mut buf = RawBuffer::acquire_writable(&arr)?;
        if buf.len() != n_out * itemsize {
            return Err(PyValueError::new_err(format!(
                "numpy buffer size {} != expected {}",
                buf.len(), n_out * itemsize,
            )));
        }
        let mut mbuf_opt = mask.as_ref()
            .map(|m| RawBuffer::acquire_writable(&m.arr))
            .transpose()?;
        let out = buf.as_mut_slice();
        let mut mout_opt: Option<&mut [u8]> =
            mbuf_opt.as_mut().map(|m| m.as_mut_slice());
        let mask_itemsize = mask.as_ref().map(|m| m.itemsize).unwrap_or(0);
        let mask_field_layout: Option<&[(usize, usize)]> =
            mask.as_ref().map(|m| m.field_layout.as_slice());

        process_runs(
            file_handle, &runs, data_offset, row_width, rows_per_chunk,
            |src_row, disk_row, output_row| {
                let dst_row = &mut out
                    [output_row * itemsize..(output_row + 1) * itemsize];
                for (col_idx, col) in columns.iter().enumerate() {
                    let src = &src_row
                        [col.byte_offset..col.byte_offset + col.byte_width];
                    let (dst_off, dst_w) = field_layout[col_idx];
                    let dst = &mut dst_row[dst_off..dst_off + dst_w];
                    // TNULL short-circuit: when the field text matches
                    // the user-declared null sentinel, skip parsing
                    // (which would raise on a non-numeric sentinel like
                    // "NA" or "NULL") and leave the pre-zeroed default.
                    // The mask gets set below by write_row_mask if
                    // mask_null=True; otherwise the cell silently reads
                    // as the dtype's zero.
                    if let Some(tnull) = &col.tnull {
                        if matches_tnull(trim_ascii(src), tnull) {
                            continue;
                        }
                    }
                    convert_ascii_cell(
                        col, scaling_kinds[col_idx], src, dst, disk_row,
                    )?;
                }
                if let Some(m) = mout_opt.as_deref_mut() {
                    let m_row = &mut m
                        [output_row * mask_itemsize
                            ..(output_row + 1) * mask_itemsize];
                    write_row_mask(
                        &columns, mask_field_layout.unwrap(),
                        src_row, m_row,
                    );
                }
                Ok(())
            },
        )?;
    }

    if mask_null {
        wrap_masked(py, arr, mask.map(|m| m.arr))
    } else {
        Ok(arr.unbind())
    }
}

// Read one column into a plain (non-structured) ndarray of shape
// `(n_selected_rows,)`.  Mirrors hdu_table::read_one_column.
//
// `as_bytes` is meaningful only for A columns: when true, on-disk
// bytes are placed into an S<n> field with no decoding, null-truncation,
// or trailing-space stripping.  Useful for non-ASCII byte content
// (the default U decode would reject it).  Rejected on non-A columns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_one_column(
    py: Python<'_>,
    meta: &AsciiTableMeta,
    data_offset: u64,
    file_handle: &FileHandle,
    name: &str,
    rows_arg: Option<&Bound<'_, PyAny>>,
    as_bytes: bool,
    scale: bool,
    mask_null: bool,
) -> PyResult<Py<PyAny>> {
    let n_rows_total = meta.nrows as usize;
    let row_width = meta.row_width as usize;

    let col = meta.columns.iter()
        .find(|c| c.name.eq_ignore_ascii_case(name.trim()))
        .ok_or_else(|| {
            let available: Vec<&str> =
                meta.columns.iter().map(|c| c.name.as_str()).collect();
            PyValueError::new_err(format!(
                "unknown column name: '{}'.  Available columns: {:?}",
                name, available,
            ))
        })?
        .clone();

    if as_bytes && col.tform_letter != 'A' {
        return Err(PyValueError::new_err(format!(
            "as_bytes=True is only meaningful for character (A) columns; \
             column '{}' has TFORM type '{}'",
            col.name, col.tform_letter,
        )));
    }

    let kind = if scale { ascii_scaling_kind(&col) } else { AsciiScalingKind::None };

    let (n_out, runs) = plan_runs(rows_arg, n_rows_total)?;

    // Pick the output dtype:
    //   - as_bytes on A:  S<width>
    //   - everything else: ascii_field_dtype(col, scale)
    let dtype_str = if as_bytes {
        format!("S{}", col.byte_width)
    } else {
        ascii_field_dtype(&col, scale)
    };

    let np = py.import("numpy")?;
    let arr = np.call_method1("zeros", ((n_out,), &dtype_str))?;
    let mask_arr: Option<Bound<'_, PyAny>> = if mask_null && col.tnull.is_some() {
        Some(np.call_method1("zeros", ((n_out,), "?"))?)
    } else {
        None
    };

    if n_out == 0 || row_width == 0 || col.byte_width == 0 {
        return if mask_null {
            wrap_masked(py, arr, mask_arr)
        } else {
            Ok(arr.unbind())
        };
    }

    let dt = arr.getattr("dtype")?;
    let dst_bytes_per_row: usize = dt.getattr("itemsize")?.extract()?;

    let rows_per_chunk = std::cmp::max(1, READ_CHUNK_TARGET_BYTES / row_width);

    {
        let mut buf = RawBuffer::acquire_writable(&arr)?;
        if buf.len() != n_out * dst_bytes_per_row {
            return Err(PyValueError::new_err(format!(
                "numpy buffer size {} != expected {}",
                buf.len(), n_out * dst_bytes_per_row,
            )));
        }
        let mut mbuf_opt = mask_arr.as_ref()
            .map(RawBuffer::acquire_writable)
            .transpose()?;
        let out = buf.as_mut_slice();
        let mut mout_opt: Option<&mut [u8]> =
            mbuf_opt.as_mut().map(|m| m.as_mut_slice());

        process_runs(
            file_handle, &runs, data_offset, row_width, rows_per_chunk,
            |src_row, disk_row, output_row| {
                let src = &src_row[col.byte_offset
                    ..col.byte_offset + col.byte_width];
                let dst_start = output_row * dst_bytes_per_row;
                let dst = &mut out[dst_start..dst_start + dst_bytes_per_row];
                // TNULL short-circuit (see read_ascii_table for the
                // rationale): skip the per-cell parse when the field
                // matches the user-declared sentinel.
                let is_null = match &col.tnull {
                    Some(tnull) => matches_tnull(trim_ascii(src), tnull),
                    None => false,
                };
                if !is_null {
                    if as_bytes {
                        pack_s_cell_from_ascii(src, dst);
                    } else {
                        convert_ascii_cell(&col, kind, src, dst, disk_row)?;
                    }
                }
                if let Some(m) = mout_opt.as_deref_mut() {
                    m[output_row] = is_null as u8;
                }
                Ok(())
            },
        )?;
    }

    if mask_null {
        wrap_masked(py, arr, mask_arr)
    } else {
        Ok(arr.unbind())
    }
}

// Used by __repr__ on the pyclass.
pub(crate) fn ascii_repr_dtype_str(col: &AsciiColumn) -> String {
    ascii_field_dtype(col, true)
}
