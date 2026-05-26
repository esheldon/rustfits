// Table read path: read_table, read_one_column, plus the cell
// converters, heap-pass plumbing, run planner, and mask helpers.

use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PySlice, PyTuple};
use std::io::{Read, Seek, SeekFrom};

use crate::common::{lock_file, parse_keyword, FileHandle, RawBuffer};

use super::columns::{
    bytes_per_element, byteswap_unit, scaled_output_dtype, scaling_kind,
    Column, ScalingKind, TableMeta,
};

// Unsigned-trick conversion for one column-cell's worth of elements.
// All four cases preserve byte width (u8↔i8, i16↔u16, i32↔u32, i64↔u64);
// the conversion is equivalent to flipping the on-disk sign bit.
fn convert_unsigned_trick_cell(
    letter: char, repeat: usize, src: &[u8], dst: &mut [u8],
) {
    match letter {
        'B' => {
            // FITS B is u8; physical = stored - 128 yields signed i8.
            // dst[k] = src[k] - 128 reinterpreted as i8 bit pattern.
            for k in 0..repeat {
                dst[k] = src[k].wrapping_sub(128);
            }
        }
        'I' => {
            for k in 0..repeat {
                let stored = i16::from_be_bytes(
                    src[2 * k..2 * k + 2].try_into().unwrap()
                );
                let physical: u16 = (stored as i32 + 32768) as u16;
                dst[2 * k..2 * k + 2]
                    .copy_from_slice(&physical.to_ne_bytes());
            }
        }
        'J' => {
            for k in 0..repeat {
                let stored = i32::from_be_bytes(
                    src[4 * k..4 * k + 4].try_into().unwrap()
                );
                let physical: u32 = (stored as i64 + 2147483648) as u32;
                dst[4 * k..4 * k + 4]
                    .copy_from_slice(&physical.to_ne_bytes());
            }
        }
        'K' => {
            for k in 0..repeat {
                let stored = i64::from_be_bytes(
                    src[8 * k..8 * k + 8].try_into().unwrap()
                );
                // 2^63 doesn't fit in i64; do the add as u64
                // (wrapping_add is the correct unsigned-bias map).
                let physical: u64 =
                    (stored as u64).wrapping_add(1u64 << 63);
                dst[8 * k..8 * k + 8]
                    .copy_from_slice(&physical.to_ne_bytes());
            }
        }
        _ => unreachable!(),
    }
}

// Write one byte (0 or 1) per element into `mask_dst`, set to 1 where
// the stored big-endian element equals `tnull`.  The comparison is in
// stored (pre-scaling) space per the FITS spec — TNULLn is the raw
// on-disk sentinel.  Called only for integer columns (B/I/J/K); the
// caller is responsible for the letter check.
fn write_cell_mask(
    letter: char, repeat: usize, tnull: i64, src: &[u8], mask_dst: &mut [u8],
) {
    match letter {
        'B' => {
            for k in 0..repeat {
                let stored = src[k] as i64;
                mask_dst[k] = (stored == tnull) as u8;
            }
        }
        'I' => {
            for k in 0..repeat {
                let stored = i16::from_be_bytes(
                    src[2 * k..2 * k + 2].try_into().unwrap()
                ) as i64;
                mask_dst[k] = (stored == tnull) as u8;
            }
        }
        'J' => {
            for k in 0..repeat {
                let stored = i32::from_be_bytes(
                    src[4 * k..4 * k + 4].try_into().unwrap()
                ) as i64;
                mask_dst[k] = (stored == tnull) as u8;
            }
        }
        'K' => {
            for k in 0..repeat {
                let stored = i64::from_be_bytes(
                    src[8 * k..8 * k + 8].try_into().unwrap()
                );
                mask_dst[k] = (stored == tnull) as u8;
            }
        }
        _ => unreachable!(
            "write_cell_mask called with non-integer letter '{}'", letter
        ),
    }
}

// General scaling: physical = tscal * stored + tzero, in f64, output
// as f8.  Loses precision for i64 inputs whose magnitude exceeds 2^53.
fn convert_general_scaling_cell(
    letter: char, repeat: usize, tscal: f64, tzero: f64,
    src: &[u8], dst: &mut [u8],
) {
    for k in 0..repeat {
        let stored: f64 = match letter {
            'B' => src[k] as f64,
            'I' => i16::from_be_bytes(
                src[2 * k..2 * k + 2].try_into().unwrap()
            ) as f64,
            'J' => i32::from_be_bytes(
                src[4 * k..4 * k + 4].try_into().unwrap()
            ) as f64,
            'K' => i64::from_be_bytes(
                src[8 * k..8 * k + 8].try_into().unwrap()
            ) as f64,
            'E' => f32::from_be_bytes(
                src[4 * k..4 * k + 4].try_into().unwrap()
            ) as f64,
            'D' => f64::from_be_bytes(
                src[8 * k..8 * k + 8].try_into().unwrap()
            ),
            _ => unreachable!(
                "general scaling on unsupported letter '{}'", letter
            ),
        };
        let physical = tscal * stored + tzero;
        dst[8 * k..8 * k + 8].copy_from_slice(&physical.to_ne_bytes());
    }
}

// Map a column to its numpy field dtype string + shape (in numpy axis
// order, i.e. slowest-varying first).  TFORM repeat for non-A columns is
// the element count; for A columns it is the per-row byte width (== total
// string length).  TDIMn is in FITS (FORTRAN) order with fastest first,
// so we reverse it for numpy.
//
// Shape conventions:
//   - variable (P/Q): scalar Object — one ndarray (or str/bytes) per row
//   - no TDIM, numeric, repeat == 1: scalar (empty shape)
//   - no TDIM, numeric, repeat  > 1: shape = (repeat,)
//   - TDIM present, numeric: shape = reversed(tdim)
//   - no TDIM, A: scalar U<repeat>
//   - TDIM present, A: U<tdim[0]>, shape = reversed(tdim[1..])
//   - X (bit), repeat == 1, no TDIM: scalar bool
//   - X (bit), repeat  > 1, no TDIM: shape = (repeat,) of bool
//   - X (bit), TDIM present: shape = reversed(tdim) of bool
//
// Numpy structured fields with shape (1,) are NOT equivalent to scalar
// fields (they add a trailing axis of length 1).  We deliberately use
// scalar for repeat==1, no-TDIM to keep the read-back shape natural.
pub(crate) fn field_dtype_and_shape(
    col: &Column,
    scale: bool,
) -> PyResult<(String, Vec<usize>)> {
    if col.var_kind.is_some() {
        return Ok(("O".to_string(), Vec::new()));
    }
    if col.tform_letter == 'X' {
        let shape: Vec<usize> = match &col.tdim {
            Some(tdim) => tdim.iter().rev().copied().collect(),
            None => if col.repeat > 1 { vec![col.repeat] } else { Vec::new() },
        };
        return Ok(("?".to_string(), shape));
    }
    if col.tform_letter == 'A' {
        return Ok(match &col.tdim {
            Some(tdim) => {
                let str_len = tdim[0];
                let shape: Vec<usize> = tdim[1..].iter().rev().copied().collect();
                (format!("U{}", str_len), shape)
            }
            None => (format!("U{}", col.repeat), Vec::new()),
        });
    }
    // Numeric letters.  All native-endian; the byte swap from on-disk
    // big-endian happens in the row reader.  When TSCAL/TZERO are
    // active, the dtype string may be promoted by scaled_output_dtype.
    let kind = if scale { scaling_kind(col)? } else { ScalingKind::None };
    let dtype_str: &str = match kind {
        ScalingKind::None => match col.tform_letter {
            'L' => "?",
            'B' => "u1",
            'I' => "i2",
            'J' => "i4",
            'K' => "i8",
            'E' => "f4",
            'D' => "f8",
            'C' => "c8",
            'M' => "c16",
            _ => unreachable!("unsupported TFORM letter '{}'", col.tform_letter),
        },
        ScalingKind::UnsignedTrick | ScalingKind::General => {
            scaled_output_dtype(col.tform_letter, kind)
        }
    };
    let shape: Vec<usize> = match &col.tdim {
        Some(tdim) => tdim.iter().rev().copied().collect(),
        None => if col.repeat > 1 { vec![col.repeat] } else { Vec::new() },
    };
    Ok((dtype_str.to_string(), shape))
}

// Build a numpy structured dtype matching the table layout.  The dtype is
// always native-endian; the on-disk big-endian bytes are swapped at read
// time.  Cell shapes are reversed from TDIMn so that numpy (row-major)
// iteration walks the same elements as FITS (column-major) iteration
// would in the original file.  `scale=true` promotes columns with
// TSCAL/TZERO to their scaled dtype (e.g. u2 for the unsigned-int trick,
// f8 for general scaling).
pub(crate) fn build_numpy_dtype(
    py: Python<'_>,
    columns: &[Column],
    scale: bool,
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let np_dtype = numpy.getattr("dtype")?;
    let fields = PyList::empty(py);
    for col in columns {
        let (dtype_str, shape) = field_dtype_and_shape(col, scale)?;
        let tuple = if shape.is_empty() {
            PyTuple::new(py, [
                col.name.clone().into_pyobject(py)?.into_any(),
                dtype_str.into_pyobject(py)?.into_any(),
            ])?
        } else {
            let shape_tuple = PyTuple::new(py, &shape)?;
            PyTuple::new(py, [
                col.name.clone().into_pyobject(py)?.into_any(),
                dtype_str.into_pyobject(py)?.into_any(),
                shape_tuple.into_any(),
            ])?
        };
        fields.append(tuple)?;
    }
    Ok(np_dtype.call1((fields,))?.unbind())
}

// Wrap `data` in a numpy.ma.MaskedArray.  When `mask` is None we pass
// np.ma.nomask explicitly.  For plain ndarrays this gives a true
// nomask (`.mask is np.ma.nomask`).  For STRUCTURED data numpy.ma
// always materializes an all-False structured bool mask regardless of
// what's passed — the "zero overhead" path is unavailable in that
// case, but the rustfits-side allocation is still skipped, and
// callers still get a consistent MaskedArray return type.
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

// Reject mask_null=True when any selected column is variable-length
// AND carries TNULL.  VLA mask support is deferred (per-row Object
// mask ndarrays); this is a clean up-front rejection so the user sees
// a useful error before any I/O.
fn reject_var_tnull(columns: &[Column]) -> PyResult<()> {
    for col in columns {
        if col.tnull.is_some() && col.var_kind.is_some() {
            return Err(PyValueError::new_err(format!(
                "column '{}' is variable-length and carries TNULL; \
                 mask_null=True on variable-length columns is not yet \
                 supported.  Use mask_null=False, or read this column \
                 separately.",
                col.name
            )));
        }
    }
    Ok(())
}

// Allocated mask array + pre-computed layout the row loop needs.
// Returned by allocate_mask_array; absent (None) when mask_null=False
// or when no selected column carries TNULL (in which case the caller
// still wraps the data in MaskedArray with nomask, for consistent
// return type — no allocation overhead).
struct MaskArray<'py> {
    arr: Bound<'py, PyAny>,
    itemsize: usize,
    field_layout: Vec<(usize, usize)>,
}

fn allocate_mask_array<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    columns: &[Column],
    n_out: usize,
    mask_null: bool,
) -> PyResult<Option<MaskArray<'py>>> {
    if !mask_null || !columns.iter().any(|c| c.tnull.is_some()) {
        return Ok(None);
    }
    let mask_dtype = build_mask_dtype(py, columns)?;
    // np.zeros so non-null elements (and non-int columns) stay False
    // without any per-cell mask writes.
    let arr = np.call_method1("zeros", (n_out, mask_dtype.bind(py)))?;
    let mdt = arr.getattr("dtype")?;
    let itemsize: usize = mdt.getattr("itemsize")?.extract()?;
    let field_layout = numpy_field_layout(py, &mdt, columns)?;
    Ok(Some(MaskArray { arr, itemsize, field_layout }))
}

// Write per-element TNULL masks for one row across all selected
// columns.  Columns without TNULL are skipped (their bytes were
// pre-zeroed by np.zeros, so False is already correct).  Assumes
// VLA+TNULL columns have been rejected upstream.
fn write_row_mask(
    columns: &[Column],
    field_layout: &[(usize, usize)],
    src_row: &[u8],
    m_row: &mut [u8],
) {
    for (col_idx, col) in columns.iter().enumerate() {
        if let Some(tnull) = col.tnull {
            let src = &src_row[col.byte_offset
                ..col.byte_offset + col.byte_width];
            let (off, w) = field_layout[col_idx];
            write_cell_mask(
                col.tform_letter, col.repeat, tnull,
                src, &mut m_row[off..off + w],
            );
        }
    }
}

// Build a numpy structured bool dtype that mirrors the data dtype's
// per-field shapes but uses '?' (bool) for every field.  Used when
// `mask_null=True` to allocate the parallel mask array that
// np.ma.MaskedArray wraps around the data.  Each field's shape matches
// the data field exactly so per-element TNULL masking lines up with
// the per-element data layout (repeat>1, TDIM reshape, A array, X
// reshape).  Object (variable-length) fields get a scalar bool slot;
// callers refuse `mask_null=True` on VLA columns that actually carry
// TNULL before reaching here.
fn build_mask_dtype(
    py: Python<'_>,
    columns: &[Column],
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let np_dtype = numpy.getattr("dtype")?;
    let fields = PyList::empty(py);
    for col in columns {
        // scale=false: we only need the shape, and that path never
        // errors (it skips scaling_kind() entirely).  Shape is
        // identical with scale=true; the dtype string would just be
        // different (which we override to '?' here anyway).
        let (_, shape) = field_dtype_and_shape(col, /* scale = */ false)?;
        let tuple = if shape.is_empty() {
            PyTuple::new(py, [
                col.name.clone().into_pyobject(py)?.into_any(),
                "?".into_pyobject(py)?.into_any(),
            ])?
        } else {
            let shape_tuple = PyTuple::new(py, &shape)?;
            PyTuple::new(py, [
                col.name.clone().into_pyobject(py)?.into_any(),
                "?".into_pyobject(py)?.into_any(),
                shape_tuple.into_any(),
            ])?
        };
        fields.append(tuple)?;
    }
    Ok(np_dtype.call1((fields,))?.unbind())
}

// Copy `src` to `dst`, reversing each `elem_size`-byte chunk if the host
// is little-endian.  FITS numeric values are big-endian on disk; numpy
// fields here are native-endian, so on little-endian hosts we swap.
fn copy_with_byteswap(src: &[u8], dst: &mut [u8], elem_size: usize) {
    if cfg!(target_endian = "big") || elem_size <= 1 {
        dst.copy_from_slice(src);
        return;
    }
    let n = src.len() / elem_size;
    for k in 0..n {
        let base = k * elem_size;
        for i in 0..elem_size {
            dst[base + i] = src[base + elem_size - 1 - i];
        }
    }
}

// Per-row converter for one column.  `src` is the on-disk bytes for this
// column in one row; `dst` is the numpy field's bytes for the same row.
// Layouts and sizes match for numerics (modulo byte order); for A columns
// the numpy U field is 4x larger than the on-disk A bytes.  When `kind`
// is non-None, the scaling converter handles the cell instead and may
// produce a wider dst (general scaling → f8).
pub(crate) fn convert_column_cell(
    col: &Column,
    src: &[u8],
    dst: &mut [u8],
    row_index: usize,
    kind: ScalingKind,
) -> PyResult<()> {
    match kind {
        ScalingKind::None => {}
        ScalingKind::UnsignedTrick => {
            convert_unsigned_trick_cell(
                col.tform_letter, col.repeat, src, dst,
            );
            return Ok(());
        }
        ScalingKind::General => {
            convert_general_scaling_cell(
                col.tform_letter, col.repeat, col.tscal, col.tzero, src, dst,
            );
            return Ok(());
        }
    }
    match col.tform_letter {
        // FITS L is one byte: ASCII 'T' (true) or 'F'/anything-else (false).
        // numpy bool is one byte: 0 or 1.  Convert per byte.
        'L' => {
            for (i, &b) in src.iter().enumerate() {
                dst[i] = if b == b'T' { 1 } else { 0 };
            }
            Ok(())
        }
        'B' => {
            dst.copy_from_slice(src);
            Ok(())
        }
        'I' => { copy_with_byteswap(src, dst, 2); Ok(()) }
        // E (f4) and the f4 halves of C (c8) are 4 bytes; J (i4) is too.
        'J' | 'E' | 'C' => { copy_with_byteswap(src, dst, 4); Ok(()) }
        // K (i8), D (f8), and the f8 halves of M (c16) are 8 bytes.
        'K' | 'D' | 'M' => { copy_with_byteswap(src, dst, 8); Ok(()) }
        'A' => convert_a_cell(col, src, dst, row_index),
        'X' => { convert_x_cell(col, src, dst); Ok(()) }
        // parse_columns rejects unsupported letters up front.
        _ => unreachable!("unsupported TFORM letter '{}'", col.tform_letter),
    }
}

// Unpack a row's FITS X (bit) cell into numpy bool bytes.  On disk:
// `n_bits` packed bits in ceil(n_bits/8) bytes, MSB-first within each
// byte (so the first bit of the cell is the high bit of the first
// byte).  In numpy: one byte per bit (0 or 1).  The bit-walk order
// matches the FITS in-cell order, so a TDIM reshape ends up correctly
// transposed by the standard "reversed(tdim) on the numpy side" rule.
fn convert_x_cell(col: &Column, src: &[u8], dst: &mut [u8]) {
    let n_bits = col.repeat;
    for i in 0..n_bits {
        let byte_idx = i / 8;
        let bit_idx = 7 - (i % 8);
        dst[i] = (src[byte_idx] >> bit_idx) & 1;
    }
}

// Convert FITS A bytes to numpy U cells.  When TDIM is present, the on-disk
// A bytes hold `total / str_len` strings of `str_len` chars each; each
// becomes one U<str_len> slot in the numpy field.  For each string:
//   1. truncate at first null byte (C-string semantics)
//   2. rstrip ASCII spaces
//   3. validate each remaining byte is ASCII; raise if not, naming the
//      column and pointing at read_column(..., as_bytes=True) as the
//      escape hatch
//   4. write codepoints into the U slot as 4-byte native-endian UCS-4,
//      zero-padding the rest
fn convert_a_cell(
    col: &Column,
    src: &[u8],
    dst: &mut [u8],
    row_index: usize,
) -> PyResult<()> {
    let str_len = match &col.tdim {
        Some(tdim) => tdim[0],
        None => col.repeat,
    };
    if str_len == 0 {
        return Ok(());
    }
    let num_strings = col.repeat / str_len;
    let u_bytes_per_str = str_len * 4;

    // Pre-zero the whole destination; any unwritten codepoints stay null,
    // which numpy treats as string terminator.
    for b in dst.iter_mut() { *b = 0; }

    for s in 0..num_strings {
        let src_str = &src[s * str_len..(s + 1) * str_len];
        let dst_str = &mut dst[s * u_bytes_per_str..(s + 1) * u_bytes_per_str];

        let null_pos = src_str.iter()
            .position(|&b| b == 0)
            .unwrap_or(src_str.len());
        let mut eff_len = null_pos;
        while eff_len > 0 && src_str[eff_len - 1] == b' ' {
            eff_len -= 1;
        }

        for i in 0..eff_len {
            let b = src_str[i];
            if !b.is_ascii() {
                return Err(PyValueError::new_err(format!(
                    "column '{}' row {} contains non-ASCII byte 0x{:02X} \
                     at position {} (read this column with \
                     table.read_column('{}', as_bytes=True) to get raw bytes)",
                    col.name, row_index, b, i, col.name,
                )));
            }
            let cp_bytes = (b as u32).to_ne_bytes();
            dst_str[i * 4..i * 4 + 4].copy_from_slice(&cp_bytes);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Variable-length (P/Q) heap-pass helpers
// ---------------------------------------------------------------------------

// One variable-length cell descriptor, captured during the main pass.
// Sorted by heap_offset before the heap pass so that sequential heap
// access is cache-friendly.
//
// col_idx: index into the `columns` slice the caller is using (so the
//   heap pass can look up the inner element letter, column name for
//   error messages, and the destination column-name for structured
//   assignment).
// output_row: row in the output ndarray (already in user-requested
//   order — the main pass's run-planner does the on-disk → output
//   shuffle for us).
struct VarCell {
    output_row: usize,
    col_idx: usize,
    nelements: i64,
    heap_offset: i64,
}

// Parse THEAP and compute the heap base offset (relative to the start
// of the data section).  THEAP is allowed to be 0 / missing, in which
// case the heap starts immediately after the main row block at offset
// NAXIS1 * NAXIS2.
pub(crate) fn heap_base_in_data(cards: &[String]) -> u64 {
    let naxis1 = parse_keyword(cards, "NAXIS1").unwrap_or(0).max(0) as u64;
    let naxis2 = parse_keyword(cards, "NAXIS2").unwrap_or(0).max(0) as u64;
    let theap = parse_keyword(cards, "THEAP").unwrap_or(0);
    if theap > 0 { theap as u64 } else { naxis1.saturating_mul(naxis2) }
}

// Pull the big-endian (nelements, heap_offset) descriptor from a row's
// bytes for one P or Q column.  P is two i32s, Q is two i64s.  The
// signed types are per the FITS standard; values < 0 indicate a bad
// file and are rejected at heap-read time.
pub(crate) fn read_descriptor(kind: char, src: &[u8]) -> (i64, i64) {
    match kind {
        'P' => {
            let n = i32::from_be_bytes(src[0..4].try_into().unwrap()) as i64;
            let off = i32::from_be_bytes(src[4..8].try_into().unwrap()) as i64;
            (n, off)
        }
        'Q' => {
            let n = i64::from_be_bytes(src[0..8].try_into().unwrap());
            let off = i64::from_be_bytes(src[8..16].try_into().unwrap());
            (n, off)
        }
        _ => unreachable!("read_descriptor called with kind '{}'", kind),
    }
}

// Build the Python object for one variable-length cell from already-
// read big-endian heap bytes.  For numeric/L: a numpy 1-D ndarray of
// native-endian dtype matching the inner element letter.  For A: a
// Python `str` (strict ASCII, with the same kind of helpful error as
// the fixed-A path) — unless `as_bytes` is set, in which case raw
// bytes are returned verbatim (matches read_column(as_bytes=True)).
pub(crate) fn build_var_cell_value(
    py: Python<'_>,
    col: &Column,
    src_bytes: &[u8],
    nelements: usize,
    row_idx: usize,
    as_bytes: bool,
    kind: ScalingKind,
) -> PyResult<Py<PyAny>> {
    let inner_letter = col.tform_letter;
    if inner_letter == 'A' {
        if as_bytes {
            return Ok(PyBytes::new(py, src_bytes).into_any().unbind());
        }
        // Strict ASCII validate; FITS A is supposed to be printable
        // ASCII.  No null-truncation / rstrip applied — variable A
        // cells store exactly `nelements` bytes with no implicit
        // padding, and the user gets all of them as-is.
        for (i, &b) in src_bytes.iter().enumerate() {
            if !b.is_ascii() {
                return Err(PyValueError::new_err(format!(
                    "column '{}' row {} contains non-ASCII byte 0x{:02X} \
                     at position {} (read this column with \
                     table.read_column('{}', as_bytes=True) to get raw bytes)",
                    col.name, row_idx, b, i, col.name,
                )));
            }
        }
        let s = std::str::from_utf8(src_bytes).unwrap();
        return Ok(s.into_pyobject(py)?.into_any().unbind());
    }
    let np = py.import("numpy")?;
    // X (bit) VLA cell: descriptor nelements is the bit count; the
    // heap holds ceil(nelements/8) MSB-packed bytes.  Build a bool
    // ndarray of length nelements (one byte per bool in numpy) and
    // unpack MSB-first — the inverse of write_vla.rs::serialize
    // for X and a per-cell version of read.rs::convert_x_cell.
    if inner_letter == 'X' {
        let arr = np.call_method1("empty", (nelements, "?"))?;
        if nelements == 0 {
            return Ok(arr.unbind());
        }
        let expected_bytes = nelements.div_ceil(8);
        if src_bytes.len() != expected_bytes {
            return Err(PyIOError::new_err(format!(
                "variable X cell heap read length mismatch: got {} \
                 bytes, expected {} (ceil({} bits / 8))",
                src_bytes.len(), expected_bytes, nelements,
            )));
        }
        let mut buf = RawBuffer::acquire_writable(&arr)?;
        let dst = buf.as_mut_slice();
        for i in 0..nelements {
            dst[i] = (src_bytes[i / 8] >> (7 - (i % 8))) & 1;
        }
        return Ok(arr.unbind());
    }
    let dtype_str: &str = match kind {
        ScalingKind::None => match inner_letter {
            'L' => "?",
            'B' => "u1",
            'I' => "i2",
            'J' => "i4",
            'K' => "i8",
            'E' => "f4",
            'D' => "f8",
            'C' => "c8",
            'M' => "c16",
            _ => unreachable!(
                "unsupported variable inner letter '{}'", inner_letter
            ),
        },
        ScalingKind::UnsignedTrick | ScalingKind::General => {
            scaled_output_dtype(inner_letter, kind)
        }
    };
    let arr = np.call_method1("empty", (nelements, dtype_str))?;
    if nelements == 0 {
        return Ok(arr.unbind());
    }
    let elem_size = bytes_per_element(inner_letter).unwrap();
    if src_bytes.len() != nelements * elem_size {
        return Err(PyIOError::new_err(format!(
            "variable cell heap read length mismatch: got {} bytes, \
             expected {} ({} elements × {})",
            src_bytes.len(), nelements * elem_size, nelements, elem_size,
        )));
    }
    let mut buf = RawBuffer::acquire_writable(&arr)?;
    let dst = buf.as_mut_slice();
    match kind {
        ScalingKind::None => {
            if inner_letter == 'L' {
                for (i, &b) in src_bytes.iter().enumerate() {
                    dst[i] = if b == b'T' { 1 } else { 0 };
                }
            } else {
                // Swap by the base float/int width — for C and M that
                // means swapping each half (real, imag) independently,
                // not the whole element.  See `byteswap_unit` docs.
                copy_with_byteswap(
                    src_bytes, dst, byteswap_unit(inner_letter),
                );
            }
        }
        ScalingKind::UnsignedTrick => {
            convert_unsigned_trick_cell(inner_letter, nelements, src_bytes, dst);
        }
        ScalingKind::General => {
            convert_general_scaling_cell(
                inner_letter, nelements, col.tscal, col.tzero, src_bytes, dst,
            );
        }
    }
    drop(buf);
    Ok(arr.unbind())
}

// Walk the captured variable-cell descriptors, read each cell's bytes
// from the heap (sorted by heap offset so seeks are forward-moving),
// build the corresponding Python object, and assign it into the output
// array.  For structured arrays use `arr[col_name][row]`; for the
// single-column 1-D Object case use `arr[row]`.  Empty cells (nelements
// == 0) are still assigned so they overwrite numpy's default None with
// an explicit empty ndarray / "" / b"".
fn heap_pass(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    file_handle: &FileHandle,
    columns: &[Column],
    data_offset: u64,
    theap: u64,
    mut var_cells: Vec<VarCell>,
    as_bytes: bool,
    single_column: bool,
    scaling_kinds: &[ScalingKind],
) -> PyResult<()> {
    if var_cells.is_empty() {
        return Ok(());
    }
    let heap_base_file = data_offset + theap;
    var_cells.sort_by_key(|c| c.heap_offset);

    let mut guard = lock_file(file_handle)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    let mut buf: Vec<u8> = Vec::new();
    for cell in &var_cells {
        if cell.nelements < 0 || cell.heap_offset < 0 {
            return Err(PyIOError::new_err(format!(
                "variable cell descriptor (nelements={}, heap_offset={}) \
                 has negative values",
                cell.nelements, cell.heap_offset,
            )));
        }
        let n = cell.nelements as usize;
        let col = &columns[cell.col_idx];
        let inner = col.tform_letter;
        // X (bit-packed) VLA: descriptor `nelements` is the BIT count
        // (per the FITS spec), and the on-disk heap holds
        // ceil(nelements/8) bytes per cell.  All other inner letters
        // have a fixed element width on disk.
        let read_len = if inner == 'X' {
            n.div_ceil(8)
        } else {
            let elem_size = bytes_per_element(inner).unwrap();
            n * elem_size
        };
        let abs_offset = heap_base_file + cell.heap_offset as u64;
        f.seek(SeekFrom::Start(abs_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        buf.resize(read_len, 0);
        if read_len > 0 {
            f.read_exact(&mut buf)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
        let value = build_var_cell_value(
            py, col, &buf, n, cell.output_row, as_bytes,
            scaling_kinds[cell.col_idx],
        )?;
        if single_column {
            arr.set_item(cell.output_row, value)?;
        } else {
            arr.get_item(&col.name)?.set_item(cell.output_row, value)?;
        }
    }
    Ok(())
}

// Per-column numpy field layout: (offset within record, bytes within
// record).  numpy may pad fields; we trust numpy to tell us where each
// field lives rather than recomputing it.
pub(crate) fn numpy_field_layout(
    py: Python<'_>,
    dtype: &Bound<'_, PyAny>,
    columns: &[Column],
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

// Resolve a user-supplied list of column names against the full column
// list parsed from the header.  Matching is case-insensitive (per the
// project convention — column names preserve case on disk but lookup is
// case-insensitive).  Reject duplicates and unknown names up front so we
// don't start reading just to fail mid-stream.  Returns the matching
// Columns in the user's requested order — `byte_offset`/`byte_width` on
// each Column still point at this column's slot in the on-disk row, so
// the per-row converter can subset directly.
pub(crate) fn resolve_columns(
    all: &[Column],
    requested: &[String],
) -> PyResult<Vec<Column>> {
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
                // Build a helpful "did you mean" message listing available
                // columns; useful for typos and case mistakes.
                let available: Vec<&str> =
                    all.iter().map(|c| c.name.as_str()).collect();
                return Err(PyValueError::new_err(format!(
                    "unknown column name: '{}'.  Available columns: {:?}",
                    name, available
                )));
            }
        }
    }
    Ok(out)
}

// Target chunk size for streaming reads.  Picked to absorb syscall
// overhead on tables with many small rows while keeping peak overhead
// (over and above the numpy output array) small enough to ignore.
const READ_CHUNK_TARGET_BYTES: usize = 1 << 20;  // 1 MiB

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

// Parse a Python `rows=` argument (slice OR iterable of ints) into a
// list of disk-row indices in the user's requested order, with negatives
// normalized and duplicates removed (first occurrence kept).  Validates
// range up front so a bad index in the middle of a large request fails
// before any I/O.
pub(crate) fn resolve_rows(
    rows_arg: &Bound<'_, PyAny>,
    n_rows: usize,
) -> PyResult<Vec<usize>> {
    let mut requested: Vec<usize> = Vec::new();
    if let Ok(slice) = rows_arg.cast::<PySlice>() {
        let indices = slice.indices(n_rows as isize)?;
        let step = indices.step;
        if step == 0 {
            return Err(PyValueError::new_err("rows= slice has zero step"));
        }
        // Match Python slice semantics: walk start..stop with step (which
        // may be negative).  Empty result when start == stop.
        let mut i = indices.start;
        let stop = indices.stop;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            if i < 0 || i >= n_rows as isize {
                // Should not happen — PySlice.indices clamps to [0, n_rows].
                break;
            }
            requested.push(i as usize);
            i += step;
        }
    } else {
        let iter = rows_arg.try_iter().map_err(|_| {
            PyValueError::new_err(
                "rows= must be a slice or an iterable of integers"
            )
        })?;
        for item in iter {
            let item = item?;
            let v: i64 = item.extract().map_err(|_| PyValueError::new_err(
                "rows= entries must be integers"
            ))?;
            let normalized = if v < 0 { n_rows as i64 + v } else { v };
            if normalized < 0 || normalized >= n_rows as i64 {
                return Err(PyIndexError::new_err(format!(
                    "row index {} out of range for table with {} rows",
                    v, n_rows
                )));
            }
            requested.push(normalized as usize);
        }
    }
    if requested.is_empty() {
        return Err(PyValueError::new_err(
            "rows= selected zero rows; pass None for all rows"
        ));
    }
    // Dedup preserving first-occurrence order.
    let mut seen: std::collections::HashSet<usize> =
        std::collections::HashSet::with_capacity(requested.len());
    let mut deduped = Vec::with_capacity(requested.len());
    for r in requested {
        if seen.insert(r) {
            deduped.push(r);
        }
    }
    Ok(deduped)
}

// Build the run plan from a `rows=` argument.  When rows is None, one
// run covers the whole table.  Otherwise: sort the user-order-deduped
// indices, group contiguous runs (consecutive disk rows differ by 1),
// and carry the output-position list per run so each row read knows
// where to land in the user's requested order.
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
            // Pair (disk_row, output_position) and sort by disk_row.
            let mut indexed: Vec<(usize, usize)> = user_unique.iter()
                .enumerate()
                .map(|(i, &r)| (r, i))
                .collect();
            indexed.sort_by_key(|&(r, _)| r);
            // Coalesce runs of consecutive disk rows.
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
// output_row).  The callback decides what to do with the bytes;
// `process_runs` owns the file handle, the chunk buffer, and all the
// run/chunk bookkeeping.  Shared by `read_table` (multi-column) and
// `read_one_column` (single-column).
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

// Read a BINTABLE into a freshly-allocated numpy structured array of
// native-endian dtype.  Returns the array.  The output shape is
// `(n_selected_rows,)`, where the selection comes from `rows_arg`:
//   - rows_arg = None: every row in file order (shape `(NAXIS2,)`).
//   - rows_arg = Some(slice or iterable): deduped user-requested order.
//
// I/O strategy: the run planner sorts + coalesces the requested disk
// indices into contiguous runs and reads each run with one seek + one
// chunked sequential read (chunked to bound peak memory to ~1 MiB
// above the output array).  Within each run, each row is converted to
// the output position recorded in the plan, so the final array is in
// the user's requested order.
//
// `columns_requested = None` selects every column in file order;
// passing a list selects + reorders to the user's request.  The full
// on-disk row is still read; only the per-row conversion loop is
// restricted to selected columns.
pub(crate) fn read_table(
    py: Python<'_>,
    meta: &TableMeta,
    data_offset: u64,
    file_handle: &FileHandle,
    rows_arg: Option<&Bound<'_, PyAny>>,
    columns_requested: Option<Vec<String>>,
    scale: bool,
    mask_null: bool,
) -> PyResult<Py<PyAny>> {
    let n_rows = meta.nrows as usize;
    let row_width = meta.row_width as usize;
    let columns = match columns_requested {
        None => meta.columns.clone(),
        Some(names) => resolve_columns(&meta.columns, &names)?,
    };
    if mask_null {
        reject_var_tnull(&columns)?;
    }

    // Pre-classify scaling per column so the per-cell loop is just a
    // ScalingKind match — no f64 comparisons or TZERO checks per row.
    // Also surfaces a C/M-with-scaling error before any I/O.
    let scaling_kinds: Vec<ScalingKind> = columns.iter()
        .map(|c| if scale { scaling_kind(c) } else { Ok(ScalingKind::None) })
        .collect::<PyResult<Vec<_>>>()?;

    let (n_out, runs) = plan_runs(rows_arg, n_rows)?;

    let dtype = build_numpy_dtype(py, &columns, scale)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("empty", (n_out, dtype.bind(py)))?;
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
    let mut var_cells: Vec<VarCell> = Vec::new();
    {
        let mut buf = RawBuffer::acquire_writable(&arr)?;
        if buf.len() != n_out * itemsize {
            return Err(PyValueError::new_err(format!(
                "numpy buffer size {} != expected {}",
                buf.len(), n_out * itemsize
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
                    let src = &src_row[col.byte_offset
                        ..col.byte_offset + col.byte_width];
                    if let Some(kind) = col.var_kind {
                        // Skip the Object pointer slot — its bytes were
                        // initialized to None by np.empty and must not
                        // be touched via raw memory.  Just capture the
                        // descriptor for the heap pass.
                        let (n, off) = read_descriptor(kind, src);
                        var_cells.push(VarCell {
                            output_row, col_idx,
                            nelements: n, heap_offset: off,
                        });
                    } else {
                        let (dst_off, dst_w) = field_layout[col_idx];
                        let dst = &mut dst_row[dst_off..dst_off + dst_w];
                        convert_column_cell(
                            col, src, dst, disk_row, scaling_kinds[col_idx],
                        )?;
                    }
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
    }  // drop RawBuffers here, before heap pass touches arrays via Python.

    heap_pass(
        py, &arr, file_handle, &columns, data_offset, meta.theap,
        var_cells, /* as_bytes = */ false, /* single_column = */ false,
        &scaling_kinds,
    )?;

    if mask_null {
        wrap_masked(py, arr, mask.map(|m| m.arr))
    } else {
        Ok(arr.unbind())
    }
}

// Read one column of a BINTABLE into a freshly-allocated ndarray of
// shape `(n_selected_rows,) + field_shape`.  Output is a plain ndarray,
// not a structured array.
//
// `as_bytes` is meaningful only for A (character) columns: when true,
// the on-disk bytes are placed into an S<n> field with no decoding,
// null-truncation, or trailing-space stripping — exactly the bytes from
// the file.  This is the escape hatch for rows that contain non-ASCII
// data, which the default (strict) U decode would reject.  Rejected
// with a clear error on any non-A column.
//
// `rows_arg` semantics are identical to `read_table`.
pub(crate) fn read_one_column(
    py: Python<'_>,
    meta: &TableMeta,
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
                name, available
            ))
        })?
        .clone();

    if as_bytes && col.tform_letter != 'A' {
        return Err(PyValueError::new_err(format!(
            "as_bytes=True is only meaningful for character (A) columns; \
             column '{}' has TFORM type '{}'",
            col.name, col.tform_letter
        )));
    }
    if mask_null {
        reject_var_tnull(std::slice::from_ref(&col))?;
    }

    // Pre-classify scaling for this one column (errors early for C/M
    // with non-default TSCAL/TZERO, before any I/O).
    let kind = if scale { scaling_kind(&col)? } else { ScalingKind::None };

    let (n_out, runs) = plan_runs(rows_arg, n_rows_total)?;

    // Element dtype + per-row "field" shape (excluding the leading row
    // axis).  Cases:
    //   variable (P/Q):           dtype = 'O', shape = ()  — heap fills.
    //   fixed A, as_bytes=true:   dtype = 'S<n>'           — raw bytes per cell.
    //   fixed otherwise:          field_dtype_and_shape()  — same as structured.
    let (dtype_str, field_shape) = if col.var_kind.is_some() {
        ("O".to_string(), Vec::new())
    } else if as_bytes {
        let str_len = match &col.tdim {
            Some(tdim) => tdim[0],
            None => col.repeat,
        };
        let array_shape: Vec<usize> = match &col.tdim {
            Some(tdim) => tdim[1..].iter().rev().copied().collect(),
            None => Vec::new(),
        };
        (format!("S{}", str_len), array_shape)
    } else {
        field_dtype_and_shape(&col, scale)?
    };

    let mut arr_shape: Vec<usize> = Vec::with_capacity(1 + field_shape.len());
    arr_shape.push(n_out);
    arr_shape.extend_from_slice(&field_shape);

    let np = py.import("numpy")?;
    let arr = np.call_method1("empty", (arr_shape.clone(), &dtype_str))?;
    // The mask is a plain ndarray of the same shape as the data array;
    // dtype "?" (one byte per element).  Only allocated when this
    // column actually carries TNULL — otherwise nomask is fine.
    let mask_arr: Option<Bound<'_, PyAny>> =
        if mask_null && col.tnull.is_some() {
            Some(np.call_method1("zeros", (arr_shape, "?"))?)
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

    // dst_bytes_per_row is what numpy actually laid out; reading
    // itemsize from the dtype (rather than recomputing) keeps us honest
    // if numpy adds alignment we didn't anticipate.
    let dt = arr.getattr("dtype")?;
    let elem_size: usize = dt.getattr("itemsize")?.extract()?;
    let elements_per_row: usize = field_shape.iter().product::<usize>().max(1);
    let dst_bytes_per_row = elem_size * elements_per_row;
    // Mask buffer is "?" (1 byte per element), so byte stride is the
    // element count.  Only meaningful when mask_arr is Some.
    let mask_bytes_per_row = elements_per_row;

    let rows_per_chunk = std::cmp::max(1, READ_CHUNK_TARGET_BYTES / row_width);
    let mut var_cells: Vec<VarCell> = Vec::new();
    let is_variable = col.var_kind.is_some();
    {
        let mut buf = RawBuffer::acquire_writable(&arr)?;
        if buf.len() != n_out * dst_bytes_per_row {
            return Err(PyValueError::new_err(format!(
                "numpy buffer size {} != expected {}",
                buf.len(), n_out * dst_bytes_per_row
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
                if let Some(kind) = col.var_kind {
                    // Variable: do not write the Object pointer slot;
                    // capture the descriptor for the heap pass.
                    let (n, off) = read_descriptor(kind, src);
                    var_cells.push(VarCell {
                        output_row, col_idx: 0,
                        nelements: n, heap_offset: off,
                    });
                } else {
                    let dst_start = output_row * dst_bytes_per_row;
                    let dst = &mut out[dst_start..dst_start + dst_bytes_per_row];
                    if as_bytes {
                        // No decode, no null-truncate, no rstrip — give
                        // the caller exactly the bytes from disk.
                        dst.copy_from_slice(src);
                    } else {
                        convert_column_cell(&col, src, dst, disk_row, kind)?;
                    }
                }
                if let Some(m) = mout_opt.as_deref_mut() {
                    // mask_arr is Some only when col.tnull is Some and
                    // the column is fixed-width B/I/J/K.
                    let tnull = col.tnull.unwrap();
                    let mdst_start = output_row * mask_bytes_per_row;
                    write_cell_mask(
                        col.tform_letter, col.repeat, tnull, src,
                        &mut m[mdst_start..mdst_start + mask_bytes_per_row],
                    );
                }
                Ok(())
            },
        )?;
    }  // drop RawBuffers before heap pass.

    if is_variable {
        // For read_one_column the heap pass uses col_idx=0 against a
        // single-element columns slice.
        let columns_slice = std::slice::from_ref(&col);
        let scaling_kinds = [kind];
        heap_pass(
            py, &arr, file_handle, columns_slice, data_offset, meta.theap,
            var_cells, as_bytes, /* single_column = */ true,
            &scaling_kinds,
        )?;
    }

    if mask_null {
        wrap_masked(py, arr, mask_arr)
    } else {
        Ok(arr.unbind())
    }
}
