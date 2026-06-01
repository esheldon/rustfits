// ASCII-table read path: build the numpy structured dtype, walk the
// rows, parse each field's text into the matching numpy slot.
//
// Phase 1 ships whole-table read only.  rows= / columns= subsets,
// per-column read, MaskedArray return for `mask_null=True`, and the
// strip-walk planner go in Phase 2.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::io::{Read, Seek, SeekFrom};

use crate::common::{lock_file, FileHandle, RawBuffer};

use super::columns::{
    ascii_scaled_output_dtype, ascii_scaling_kind, AsciiColumn,
    AsciiScalingKind,
};
use super::format::{parse_float_field, parse_int_field};
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

// Read the whole ASCII table into a numpy structured ndarray.  Phase 1:
// rows= and columns= subsets, MaskedArray return, single-column read
// all come later.
//
// Strategy: allocate the output via np.zeros (so U fields start
// zero-terminated and not-touched scalar fields default to 0 / 0.0 /
// "" which matches FITS "undefined" semantics for blank fields).  Walk
// the rows in ~1 MiB chunks; for each row, parse every column into the
// matching field bytes via the Py_buffer (no Python-level indexing).
pub(crate) fn read_ascii_table(
    py: Python<'_>,
    meta: &AsciiTableMeta,
    data_offset: u64,
    file_handle: &FileHandle,
    scale: bool,
) -> PyResult<Py<PyAny>> {
    let n_rows = meta.nrows as usize;
    let row_width = meta.row_width as usize;
    let columns: &[AsciiColumn] = &meta.columns;

    let dtype = build_ascii_numpy_dtype(py, columns, scale)?;
    let np = py.import("numpy")?;
    // np.zeros so U fields start null-terminated and numeric fields
    // default to 0 — matches FITS "undefined value" for blank fields
    // without us having to write anything per cell.
    let arr = np.call_method1("zeros", (n_rows, dtype.bind(py)))?;

    if n_rows == 0 || row_width == 0 {
        return Ok(arr.unbind());
    }

    let arr_dtype = arr.getattr("dtype")?;
    let itemsize: usize = arr_dtype.getattr("itemsize")?.extract()?;
    let field_layout = numpy_field_layout(py, &arr_dtype, columns)?;

    // Pre-classify scaling per column so the per-row loop is just an
    // enum match.
    let scaling_kinds: Vec<AsciiScalingKind> = columns.iter()
        .map(|c| if scale { ascii_scaling_kind(c) } else { AsciiScalingKind::None })
        .collect();

    // Streaming chunk budget: ~1 MiB across `rows_per_chunk * row_width`.
    // Matches the BINTABLE READ_CHUNK_TARGET_BYTES convention.
    const READ_CHUNK_TARGET_BYTES: usize = 1 << 20;
    let rows_per_chunk = std::cmp::max(1, READ_CHUNK_TARGET_BYTES / row_width);
    let mut chunk_buf = vec![0u8; rows_per_chunk * row_width];

    {
        let mut buf = RawBuffer::acquire_writable(&arr)?;
        if buf.len() != n_rows * itemsize {
            return Err(PyValueError::new_err(format!(
                "numpy buffer size {} != expected {}",
                buf.len(), n_rows * itemsize,
            )));
        }
        let out = buf.as_mut_slice();

        let mut guard = lock_file(file_handle)?;
        let f = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let mut row_cursor = 0usize;
        while row_cursor < n_rows {
            let this_rows = std::cmp::min(rows_per_chunk, n_rows - row_cursor);
            let read_bytes = this_rows * row_width;
            f.read_exact(&mut chunk_buf[..read_bytes])
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            for r in 0..this_rows {
                let disk_row = row_cursor + r;
                let src_row = &chunk_buf[r * row_width..(r + 1) * row_width];
                let dst_row = &mut out
                    [disk_row * itemsize..(disk_row + 1) * itemsize];
                for (col_idx, col) in columns.iter().enumerate() {
                    let src = &src_row
                        [col.byte_offset..col.byte_offset + col.byte_width];
                    let (dst_off, dst_w) = field_layout[col_idx];
                    let dst = &mut dst_row[dst_off..dst_off + dst_w];
                    convert_ascii_cell(
                        col, scaling_kinds[col_idx], src, dst, disk_row,
                    )?;
                }
            }
            row_cursor += this_rows;
        }
    }

    // A columns need Py-level processing (creating Python str objects
    // would defeat the structured-array shape).  Instead we wrote into
    // U slots above via pack_u_cell_from_ascii — those are valid numpy
    // strings and need no post-processing.  Done.
    drop(np);
    Ok(arr.unbind())
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

// Used by __repr__ on the pyclass — small wrapper so hdu.rs doesn't
// need to import the column types directly for display.
pub(crate) fn ascii_repr_dtype_str(col: &AsciiColumn) -> String {
    // Prefer scaled dtype; fall back to unscaled if scaling produces
    // an unreachable case (no current cases do, but mirror the
    // tolerant TableHDU pattern).
    ascii_field_dtype(col, true)
}

