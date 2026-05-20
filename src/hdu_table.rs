// TableHDU: BINTABLE extension HDU.  Read API is being built up in stages;
// the column-metadata parser lives here so that downstream layers (dtype
// builder, row reader) can operate on a typed Vec<Column> rather than
// re-walking the header.
//
// Fixed-length column types supported: L (logical), B (uint8), I (int16),
// J (int32), K (int64), A (character), E (float32), D (float64),
// C (complex64), M (complex128), X (bit, MSB-packed).  TDIMn multi-dim
// cells respected.
//
// Variable-length (P/Q descriptor) columns supported for the same inner
// element letters except X.  Each row's main-data slot holds an 8-byte
// (P) or 16-byte (Q) big-endian descriptor (nelements, heap_offset); the
// actual data lives in the heap section at file offset
// (data_offset + THEAP + heap_offset).  Read returns numpy Object dtype,
// one ndarray (or str/bytes for A) per row.
//
// Not yet supported (rejected at parse time so downstream code stays
// simple): P/Q with TFORM repeat > 1, TDIM on P/Q, variable-length bit
// (P/Q with inner X).

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PySlice, PyString, PyTuple};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use crate::common::{
    lock_file, parse_keyword, parse_keyword_float, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, RawBuffer, TaintFlag,
};
use crate::hdu::HDU;

// All the per-column metadata needed downstream.  byte_offset is the
// offset of this column's bytes within a single row; byte_width is the
// total bytes the column occupies in each row.
//
// For variable-length (P/Q) columns: `tform_letter` is the INNER element
// letter (e.g. 'E' for "1PE(100)"), `repeat` is always 1 (multi-
// descriptor not yet supported), `byte_width` is the descriptor size
// (8 for P, 16 for Q), and `var_kind` is Some('P') or Some('Q').  The
// actual data is in the heap; the row's bytes carry only the descriptor.
#[derive(Debug, Clone)]
pub(crate) struct Column {
    pub(crate) name: String,
    pub(crate) tform_letter: char,
    // For most types: number of values per row.  For 'A' it is the total
    // string length in bytes per row (per FITS standard).  For P/Q: 1
    // (a single descriptor per row).
    pub(crate) repeat: usize,
    // From TDIMn, in FITS (FORTRAN) order: fastest-varying axis first.
    // For 'A' columns, the first dim is the per-string length; the rest
    // are array dims.  None means flat (treat as 1-D of `repeat`).
    pub(crate) tdim: Option<Vec<usize>>,
    pub(crate) byte_offset: usize,
    pub(crate) byte_width: usize,
    // Variable-length descriptor kind, if this is a P or Q column.
    pub(crate) var_kind: Option<char>,
    // TSCAL/TZERO scaling: physical = tscal * stored + tzero.
    // Defaults to no-op (1.0 / 0.0).  Only meaningful for numeric
    // columns; ignored for L/A/X and raised-on for C/M (see
    // scaling_kind()).
    pub(crate) tscal: f64,
    pub(crate) tzero: f64,
    // TNULLn integer sentinel — only populated for integer columns
    // (fixed B/I/J/K and VLA inner B/I/J/K).  For non-integer letters
    // the header keyword is silently ignored (it's meaningless per
    // the FITS spec).  The mask compare happens in stored-integer
    // space, before TSCAL/TZERO, so this value is the raw on-disk
    // sentinel regardless of any scaling.
    pub(crate) tnull: Option<i64>,
    // TUNITn column units string (e.g. "Jy", "deg", "s").  Purely
    // informational; nothing in the read/write path consumes it.
    // Surfaced via TableHDU.units and shown in the repr.
    pub(crate) tunit: Option<String>,
}

// Bytes per single element for each supported TFORM letter.  'A' is 1
// byte per character (no decoding done at this layer); the repeat count
// already encodes the per-row total.
fn bytes_per_element(letter: char) -> Option<usize> {
    match letter {
        'L' | 'B' | 'A' => Some(1),
        'I' => Some(2),
        'J' | 'E' => Some(4),
        'K' | 'D' | 'C' => Some(8),
        'M' => Some(16),
        _ => None,
    }
}

// Width (in bytes) of the smallest unit that must be byte-reversed when
// going from FITS big-endian to native-endian.  Note this differs from
// bytes_per_element for the complex types: a C (complex64) element is
// 8 bytes total but is two 4-byte float halves, each byteswapped
// independently; an M (complex128) is 16 bytes total but two 8-byte
// float halves.  Reversing the whole element would swap real↔imaginary.
fn byteswap_unit(letter: char) -> usize {
    match letter {
        'L' | 'B' | 'A' => 1,
        'I' => 2,
        'J' | 'E' | 'C' => 4,
        'K' | 'D' | 'M' => 8,
        // Unreachable: parse_columns rejects unsupported letters up front.
        _ => unreachable!("byteswap_unit called with unsupported letter '{}'", letter),
    }
}

// ---------------------------------------------------------------------------
// TSCAL/TZERO scaling
// ---------------------------------------------------------------------------

// Classification of how to apply TSCAL/TZERO to a column on read.
// Pre-computed once per column at read entry so the per-cell loop is a
// trivial enum match — columns without scaling stay on the same fast
// copy_with_byteswap path they always used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalingKind {
    // No scaling — pass stored values through unchanged.  Applied when
    // TSCAL == 1 AND TZERO == 0, when scale=False is requested, or for
    // types where scaling is meaningless (L, A, X).
    None,
    // The "unsigned-int trick": TSCAL=1 plus TZERO equal to the type's
    // sign-bias (2^15 for I, 2^31 for J, 2^63 for K, -128 for B).
    // Output preserves integer semantics — promoted to the matching
    // unsigned dtype (or i1 for B) — with no precision loss.
    UnsignedTrick,
    // Anything else: physical = TSCAL * stored + TZERO computed in f64
    // and output as f8.  i64 inputs may lose precision (53-bit mantissa).
    General,
}

// Inspect a column's TSCAL/TZERO and decide which scaling path to use.
// Raises for C/M with non-default scaling, because the FITS spec is
// silent on whether scaling applies to the real half, the imaginary
// half, or both — and silently picking one would corrupt data.
fn scaling_kind(col: &Column) -> PyResult<ScalingKind> {
    if col.tscal == 1.0 && col.tzero == 0.0 {
        return Ok(ScalingKind::None);
    }
    match col.tform_letter {
        // Scaling is meaningless on these types; treat as no-op
        // (matches astropy / fitsio).
        'L' | 'A' | 'X' => return Ok(ScalingKind::None),
        'C' | 'M' => return Err(PyValueError::new_err(format!(
            "column '{}' has TSCAL/TZERO set on a complex column \
             (TFORM='{}'); FITS does not unambiguously specify how to \
             apply scaling to complex values.  Use scale=False to read \
             raw stored values.",
            col.name, col.tform_letter,
        ))),
        _ => {}
    }
    if col.tscal == 1.0 {
        let trick = matches!(
            (col.tform_letter, col.tzero),
            ('B', t) if t == -128.0
        ) || matches!(
            (col.tform_letter, col.tzero),
            ('I', t) if t == 32768.0
        ) || matches!(
            (col.tform_letter, col.tzero),
            ('J', t) if t == 2147483648.0
        ) || matches!(
            (col.tform_letter, col.tzero),
            ('K', t) if t == 9223372036854775808.0
        );
        if trick {
            return Ok(ScalingKind::UnsignedTrick);
        }
    }
    Ok(ScalingKind::General)
}

// numpy dtype string the column reads into after applying scaling
// (only valid when kind != None).
fn scaled_output_dtype(letter: char, kind: ScalingKind) -> &'static str {
    match kind {
        ScalingKind::UnsignedTrick => match letter {
            'B' => "i1",
            'I' => "u2",
            'J' => "u4",
            'K' => "u8",
            _ => unreachable!(
                "unsigned-trick scaling on unexpected letter '{}'", letter
            ),
        },
        ScalingKind::General => "f8",
        ScalingKind::None => unreachable!("scaled_output_dtype called with None"),
    }
}

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

// Parsed pieces of a TFORM string.  For "8A": repeat=8, letter='A',
// inner_letter=None.  For "1PE(100)": repeat=1, letter='P',
// inner_letter=Some('E') (the "(100)" maxlen hint is informational and
// ignored at read time).
struct TformInfo {
    repeat: usize,
    letter: char,
    inner_letter: Option<char>,
}

// Split a TFORM string into its pieces.  P/Q descriptors have the
// trailing syntax `rPt(maxlen)` / `rQt(maxlen)` where `t` is the inner
// element type letter; the `(maxlen)` hint is parsed leniently (just
// validates the shape and discards the content).  Other letters must
// not carry any trailing characters.
fn parse_tform(tform: &str, col_index: usize) -> PyResult<TformInfo> {
    let trimmed = tform.trim();
    let (digits, rest) = trimmed
        .find(|c: char| c.is_ascii_alphabetic())
        .map(|i| trimmed.split_at(i))
        .ok_or_else(|| PyValueError::new_err(format!(
            "column {}: TFORM='{}' has no type letter", col_index, tform
        )))?;
    let letter = rest.chars().next().unwrap();
    let trailing = &rest[1..];
    let repeat: usize = if digits.is_empty() {
        1
    } else {
        digits.parse().map_err(|_| PyValueError::new_err(format!(
            "column {}: TFORM='{}' repeat count is not an integer",
            col_index, tform
        )))?
    };
    let inner_letter = if letter == 'P' || letter == 'Q' {
        Some(parse_variable_inner_letter(trailing, tform, col_index)?)
    } else {
        if !trailing.trim().is_empty() {
            return Err(PyValueError::new_err(format!(
                "column {}: TFORM='{}' has unsupported trailing modifier '{}'",
                col_index, tform, trailing
            )));
        }
        None
    };
    Ok(TformInfo { repeat, letter, inner_letter })
}

// Pull the inner element letter out of the trailing portion of a P/Q
// TFORM (e.g. "E" or "E(100)").  The "(maxlen)" hint is ignored.
fn parse_variable_inner_letter(
    trailing: &str,
    tform: &str,
    col_index: usize,
) -> PyResult<char> {
    let t = trailing.trim();
    if t.is_empty() {
        return Err(PyValueError::new_err(format!(
            "column {}: variable-length TFORM='{}' is missing the inner \
             element type letter (e.g. 'PE', 'PD', '1PJ(100)')",
            col_index, tform,
        )));
    }
    let inner = t.chars().next().unwrap();
    if !inner.is_ascii_alphabetic() {
        return Err(PyValueError::new_err(format!(
            "column {}: variable-length TFORM='{}' inner element '{}' \
             is not a letter", col_index, tform, inner,
        )));
    }
    let after = t[inner.len_utf8()..].trim();
    if !after.is_empty()
        && !(after.starts_with('(') && after.ends_with(')'))
    {
        return Err(PyValueError::new_err(format!(
            "column {}: variable-length TFORM='{}' has invalid trailer \
             '{}' after the inner letter", col_index, tform, after,
        )));
    }
    Ok(inner)
}

// Parse a TDIMn value like "(3,3)" or "(10,5,2)" into a Vec of positive
// dimensions in FORTRAN order (fastest first, as written).  Empty
// parentheses or non-positive dims are rejected.
fn parse_tdim(tdim: &str, col_index: usize) -> PyResult<Vec<usize>> {
    let trimmed = tdim.trim();
    if !(trimmed.starts_with('(') && trimmed.ends_with(')')) {
        return Err(PyValueError::new_err(format!(
            "column {}: TDIM='{}' must be parenthesized", col_index, tdim
        )));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let n: usize = part.trim().parse().map_err(|_| PyValueError::new_err(
            format!("column {}: TDIM='{}' contains non-integer dimension '{}'",
                col_index, tdim, part)
        ))?;
        if n == 0 {
            return Err(PyValueError::new_err(format!(
                "column {}: TDIM='{}' contains zero dimension", col_index, tdim
            )));
        }
        dims.push(n);
    }
    if dims.is_empty() {
        return Err(PyValueError::new_err(format!(
            "column {}: TDIM='{}' has no dimensions", col_index, tdim
        )));
    }
    Ok(dims)
}

// Walk the header cards and produce a Vec<Column> describing each
// column.  Fixed-width types L/B/I/J/K/A/E/D/C/M and X (bit) are
// supported.  Variable-length P/Q descriptors are supported for those
// same inner element letters except X (output via Object dtype, filled
// from the heap on read).
pub(crate) fn parse_columns(cards: &[String]) -> PyResult<Vec<Column>> {
    let tfields = parse_keyword(cards, "TFIELDS").ok_or_else(|| {
        PyValueError::new_err("BINTABLE missing required TFIELDS keyword")
    })?;
    if tfields < 0 {
        return Err(PyValueError::new_err(format!(
            "BINTABLE TFIELDS={} is negative", tfields
        )));
    }
    let n = tfields as usize;

    let mut columns = Vec::with_capacity(n);
    let mut offset = 0usize;

    for i in 1..=n {
        let tform_key = format!("TFORM{}", i);
        let tform = parse_string_keyword(cards, &tform_key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "BINTABLE missing required {} keyword", tform_key
            ))
        })?;
        let TformInfo { repeat, letter, inner_letter } =
            parse_tform(&tform, i)?;

        let name = parse_string_keyword(cards, &format!("TTYPE{}", i))
            .unwrap_or_else(|| format!("COL{}", i));

        let column = if letter == 'P' || letter == 'Q' {
            // Variable-length descriptor.  Only repeat=1 supported for
            // now (multi-descriptor P/Q is rare and would complicate
            // the dtype + per-cell descriptor extraction).
            if repeat != 1 {
                return Err(PyValueError::new_err(format!(
                    "column {}: variable-length TFORM='{}' with repeat>1 \
                     not yet supported (only one descriptor per row)",
                    i, tform
                )));
            }
            let inner = inner_letter.unwrap();
            if inner == 'X' {
                return Err(PyValueError::new_err(format!(
                    "column {}: variable-length TFORM='{}' inner bit type \
                     (X) not yet supported", i, tform
                )));
            }
            bytes_per_element(inner).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "column {}: variable-length TFORM='{}' has unsupported \
                     inner element letter '{}'", i, tform, inner
                ))
            })?;
            // TDIM on P/Q would mean "reshape each cell to these dims",
            // which is a useful feature but adds a heap-side reshape step.
            // Reject for now so behavior is predictable.
            if parse_string_keyword(cards, &format!("TDIM{}", i)).is_some() {
                return Err(PyValueError::new_err(format!(
                    "column {}: TDIM on variable-length column not yet \
                     supported", i
                )));
            }
            let descriptor_size = if letter == 'P' { 8 } else { 16 };
            let tscal = parse_keyword_float(cards, &format!("TSCAL{}", i))
                .unwrap_or(1.0);
            let tzero = parse_keyword_float(cards, &format!("TZERO{}", i))
                .unwrap_or(0.0);
            // Track TNULLn for VLA columns whose inner type is integer
            // so the read path can refuse mask_null=True (VLA TNULL
            // masking is not yet implemented).  For non-int inner
            // letters the keyword has no meaning; leave as None.
            let tnull = if matches!(inner, 'B' | 'I' | 'J' | 'K') {
                parse_keyword(cards, &format!("TNULL{}", i))
            } else {
                None
            };
            let tunit = parse_string_keyword(cards, &format!("TUNIT{}", i));
            Column {
                name,
                tform_letter: inner,
                repeat: 1,
                tdim: None,
                byte_offset: offset,
                byte_width: descriptor_size,
                var_kind: Some(letter),
                tscal,
                tzero,
                tnull,
                tunit,
            }
        } else {
            // X is a bit column: `repeat` is the bit count and the
            // on-disk row width is ceil(repeat/8).  Other fixed types
            // have a whole-byte element size.
            let byte_width = if letter == 'X' {
                repeat.div_ceil(8)
            } else {
                let elem_size = bytes_per_element(letter).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "column {}: TFORM='{}' uses unsupported type letter '{}'",
                        i, tform, letter
                    ))
                })?;
                repeat * elem_size
            };
            let tdim = match parse_string_keyword(cards, &format!("TDIM{}", i)) {
                Some(s) => {
                    let dims = parse_tdim(&s, i)?;
                    // Validate: product of dims must match the TFORM
                    // repeat.  For A columns this is the total byte
                    // count (A's repeat is per-row string length).
                    // For X columns this is the total bit count.
                    let product: usize = dims.iter().product();
                    if product != repeat {
                        return Err(PyValueError::new_err(format!(
                            "column {}: TDIM dims {:?} have product {} but \
                             TFORM repeat is {}", i, dims, product, repeat
                        )));
                    }
                    Some(dims)
                }
                None => None,
            };
            let tscal = parse_keyword_float(cards, &format!("TSCAL{}", i))
                .unwrap_or(1.0);
            let tzero = parse_keyword_float(cards, &format!("TZERO{}", i))
                .unwrap_or(0.0);
            // TNULLn only applies to integer columns (B/I/J/K) per the
            // FITS standard.  For other letters the keyword is silently
            // ignored if present.
            let tnull = if matches!(letter, 'B' | 'I' | 'J' | 'K') {
                parse_keyword(cards, &format!("TNULL{}", i))
            } else {
                None
            };
            let tunit = parse_string_keyword(cards, &format!("TUNIT{}", i));
            Column {
                name,
                tform_letter: letter,
                repeat,
                tdim,
                byte_offset: offset,
                byte_width,
                var_kind: None,
                tscal,
                tzero,
                tnull,
                tunit,
            }
        };
        offset += column.byte_width;
        columns.push(column);
    }

    // The accumulated row width should equal NAXIS1; if it doesn't, the
    // header is internally inconsistent (TFORM*s don't sum to NAXIS1).
    let naxis1 = parse_keyword(cards, "NAXIS1").unwrap_or(0);
    if naxis1 as usize != offset {
        return Err(PyValueError::new_err(format!(
            "BINTABLE row width {} bytes from TFORM*s does not match \
             NAXIS1={}", offset, naxis1
        )));
    }

    Ok(columns)
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
fn field_dtype_and_shape(
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
fn build_numpy_dtype(
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
fn convert_column_cell(
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
fn heap_base_in_data(cards: &[String]) -> u64 {
    let naxis1 = parse_keyword(cards, "NAXIS1").unwrap_or(0).max(0) as u64;
    let naxis2 = parse_keyword(cards, "NAXIS2").unwrap_or(0).max(0) as u64;
    let theap = parse_keyword(cards, "THEAP").unwrap_or(0);
    if theap > 0 { theap as u64 } else { naxis1.saturating_mul(naxis2) }
}

// Pull the big-endian (nelements, heap_offset) descriptor from a row's
// bytes for one P or Q column.  P is two i32s, Q is two i64s.  The
// signed types are per the FITS standard; values < 0 indicate a bad
// file and are rejected at heap-read time.
fn read_descriptor(kind: char, src: &[u8]) -> (i64, i64) {
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
fn build_var_cell_value(
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
    cards: &[String],
    mut var_cells: Vec<VarCell>,
    as_bytes: bool,
    single_column: bool,
    scaling_kinds: &[ScalingKind],
) -> PyResult<()> {
    if var_cells.is_empty() {
        return Ok(());
    }
    let heap_base_file = data_offset + heap_base_in_data(cards);
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
        let elem_size = bytes_per_element(inner).unwrap();
        let read_len = n * elem_size;
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
fn numpy_field_layout(
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
fn resolve_columns(
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
fn resolve_rows(
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
fn read_table(
    py: Python<'_>,
    cards: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    rows_arg: Option<&Bound<'_, PyAny>>,
    columns_requested: Option<Vec<String>>,
    scale: bool,
    mask_null: bool,
) -> PyResult<Py<PyAny>> {
    let n_rows = parse_keyword(cards, "NAXIS2").unwrap_or(0).max(0) as usize;
    let row_width =
        parse_keyword(cards, "NAXIS1").unwrap_or(0).max(0) as usize;
    let all_columns = parse_columns(cards)?;
    let columns = match columns_requested {
        None => all_columns,
        Some(names) => resolve_columns(&all_columns, &names)?,
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
        py, &arr, file_handle, &columns, data_offset, cards,
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
fn read_one_column(
    py: Python<'_>,
    cards: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    name: &str,
    rows_arg: Option<&Bound<'_, PyAny>>,
    as_bytes: bool,
    scale: bool,
    mask_null: bool,
) -> PyResult<Py<PyAny>> {
    let n_rows_total =
        parse_keyword(cards, "NAXIS2").unwrap_or(0).max(0) as usize;
    let row_width =
        parse_keyword(cards, "NAXIS1").unwrap_or(0).max(0) as usize;
    let all_columns = parse_columns(cards)?;

    let col = all_columns.iter()
        .find(|c| c.name.eq_ignore_ascii_case(name.trim()))
        .ok_or_else(|| {
            let available: Vec<&str> =
                all_columns.iter().map(|c| c.name.as_str()).collect();
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
            py, &arr, file_handle, columns_slice, data_offset, cards,
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

#[pyclass(extends = HDU)]
pub(crate) struct TableHDU;

impl TableHDU {
    pub(crate) fn new(
        header: Vec<String>,
        index: usize,
        filename: String,
        offsets: Arc<HduOffsets>,
        layout: Arc<FileLayout>,
        file: FileHandle,
        tainted: TaintFlag,
    ) -> (Self, HDU) {
        (
            TableHDU,
            HDU::new(header, index, filename, offsets, layout, file, tainted),
        )
    }
}

// Per-column line in the TableHDU.__repr__ column-info block.  Returns
// the numpy dtype string + an optional shape annotation:
//   - fixed scalar (repeat == 1, no TDIM):         dtype, None
//   - fixed multi (repeat > 1 or TDIM):            dtype, Some("array[a,b,...]")
//   - variable-length (P/Q):                       inner-dtype, Some("array[var]")
// Scaled dtype is shown when possible (e.g. unsigned-int trick → u2),
// falling back to the unscaled mapping if scale-based dtype resolution
// would error (C/M with non-default TSCAL/TZERO).
fn column_repr_info(col: &Column) -> (String, Option<String>) {
    if col.var_kind.is_some() {
        let inner = match col.tform_letter {
            'L' => "?",
            'B' => "u1",
            'I' => "i2",
            'J' => "i4",
            'K' => "i8",
            'E' => "f4",
            'D' => "f8",
            'C' => "c8",
            'M' => "c16",
            'A' => "S",
            _   => return (col.tform_letter.to_string(),
                           Some("array[var]".to_string())),
        };
        return (inner.to_string(), Some("array[var]".to_string()));
    }
    let (dtype_str, shape) = field_dtype_and_shape(col, /* scale = */ true)
        .or_else(|_| field_dtype_and_shape(col, /* scale = */ false))
        .unwrap_or_else(|_| ("?".to_string(), Vec::new()));
    let shape_str = if shape.is_empty() {
        None
    } else {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        Some(format!("array[{}]", dims.join(",")))
    };
    (dtype_str, shape_str)
}

#[pymethods]
impl TableHDU {
    // Multi-line, fitsio-style repr.  Shows file, extension, type,
    // EXTNAME (if present), row count, and per-column dtype + shape
    // annotation.  Column lines are dynamically aligned to the longest
    // column name.
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let columns = parse_columns(&cards)?;
        let nrows = parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0);
        let extname = parse_string_keyword(&cards, "EXTNAME");

        let mut out = String::new();
        out.push_str(&format!("  file: {}\n", super_.filename));
        out.push_str(&format!("  extension: {}\n", super_.index));
        out.push_str("  type: BINARY_TBL\n");
        if let Some(name) = extname {
            out.push_str(&format!("  extname: {}\n", name));
        }
        out.push_str(&format!("  rows: {}\n", nrows));
        out.push_str("  column info:\n");

        let max_name_len = columns.iter().map(|c| c.name.len()).max().unwrap_or(0);
        let name_w = max_name_len + 4;
        for col in &columns {
            let (dtype_str, shape_str) = column_repr_info(col);
            out.push_str(&format!(
                "    {:<w$}{}", col.name, dtype_str, w = name_w,
            ));
            if let Some(shape) = shape_str {
                out.push_str(&format!("  {}", shape));
            }
            if let Some(unit) = &col.tunit {
                out.push_str(&format!("  ({})", unit));
            }
            out.push('\n');
        }
        Ok(out)
    }

    // numpy structured dtype the table would read into.  Useful for
    // inspecting the column layout (names, per-cell shapes, types)
    // without paying for an actual read.  Reflects the default-read
    // (scale=True) dtype — i.e. columns with the TSCAL/TZERO unsigned
    // trick appear as u2/u4/u8/i1, and other scaled columns as f8.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let columns = parse_columns(&cards)?;
        build_numpy_dtype(py, &columns, /* scale = */ true)
    }

    // Column-units dict: maps column name (case preserved) to the
    // TUNITn string, or None when TUNITn is unset for that column.
    // Informational only — nothing in the read path consumes units.
    // Dict preserves the on-disk column order.
    #[getter]
    fn units(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let columns = parse_columns(&cards)?;
        let dict = PyDict::new(py);
        for col in &columns {
            dict.set_item(&col.name, col.tunit.as_deref())?;
        }
        Ok(dict.unbind())
    }

    // Read the table into a numpy structured array of native-endian
    // dtype.  Returned shape is `(n_selected,)`:
    //   - rows=None: every row in file order (n_selected == NAXIS2).
    //   - rows=slice or iterable of int: deduped, in user-requested
    //     order; negative indices supported.
    //   - columns=None: every column in file order.
    //   - columns=list of names: subset + reorder, case-insensitive.
    //   - scale=True (default): apply TSCAL/TZERO; the unsigned-int
    //     trick promotes to the matching unsigned dtype, other scaling
    //     produces f8.  scale=False returns raw stored values.
    //   - mask_null=False (default): return a plain structured ndarray;
    //     integer columns with TNULL set return the raw sentinel.
    //     mask_null=True: return numpy.ma.MaskedArray with per-field
    //     bool masks set True where the stored integer equals TNULLn.
    //     Mask compare is in stored-int space (pre-scaling).  TNULL on
    //     variable-length columns is not yet supported and raises.
    //
    // Both subsets validate fully before any I/O happens.
    #[pyo3(signature = (*, rows=None, columns=None, scale=true, mask_null=false))]
    fn read(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        columns: Option<Vec<String>>,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<Py<PyAny>> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let data_offset = super_.offsets.data_offset();
        read_table(
            py, &cards, data_offset, &super_.file, rows, columns, scale,
            mask_null,
        )
    }

    // Read a single column into a plain (non-structured) ndarray of
    // shape `(n_selected_rows,) + field_shape`.  rows= mirrors read()'s
    // semantics.  `as_bytes=True` is meaningful only for A (character)
    // columns; it returns the on-disk bytes in an S<n> field with no
    // decode, no null-truncation, and no trailing-space strip — useful
    // when a column has non-ASCII bytes that the default U decode would
    // reject.  `scale` and `mask_null` match read(); when mask_null=True
    // and this column carries TNULL, returns a numpy.ma.MaskedArray.
    #[pyo3(signature = (name, *, rows=None, as_bytes=false, scale=true, mask_null=false))]
    fn read_column(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        name: &str,
        rows: Option<&Bound<'_, PyAny>>,
        as_bytes: bool,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<Py<PyAny>> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let data_offset = super_.offsets.data_offset();
        read_one_column(
            py, &cards, data_offset, &super_.file, name, rows, as_bytes,
            scale, mask_null,
        )
    }

    // hdu[key] dispatches based on what `key` looks like:
    //
    //   slice or iterable-of-int → reads rows now, returns ndarray
    //     (equivalent to hdu.read(rows=key))
    //   single str/bytes/np.str_/np.bytes_ → returns a SingleColumnSubset
    //     (no read; user must add [rows] to trigger read_column)
    //   iterable-of-str/bytes → returns a ColumnSubset
    //     (no read; user must add [rows] to trigger read with columns=)
    //
    // Specifying a column or columns alone never invokes I/O — only rows
    // do.  Empty sequences are rejected as ambiguous.
    fn __getitem__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let kind = classify_table_key(key)?;
        match kind {
            TableKey::Rows => {
                let pyref = slf.borrow();
                let super_ = pyref.into_super();
                let cards = super_.header_snapshot()?;
                let data_offset = super_.offsets.data_offset();
                read_table(
                    py, &cards, data_offset, &super_.file, Some(key), None,
                    /* scale = */ true, /* mask_null = */ false,
                )
            }
            TableKey::SingleRow(idx) => {
                let pyref = slf.borrow();
                let super_ = pyref.into_super();
                let cards = super_.header_snapshot()?;
                let data_offset = super_.offsets.data_offset();
                // Wrap idx in a single-element list so resolve_rows
                // handles negative-index normalization and range
                // validation the same way it does for `hdu[[idx]]`.
                let one = PyList::new(py, [idx])?;
                let arr_py = read_table(
                    py, &cards, data_offset, &super_.file,
                    Some(one.as_any()), None,
                    /* scale = */ true, /* mask_null = */ false,
                )?;
                // arr is shape (1,); index [0] yields numpy's 0-d
                // record (np.void), matching `structured_arr[i]`
                // semantics for the user.
                let arr_bound = arr_py.bind(py);
                Ok(arr_bound.get_item(0)?.unbind())
            }
            TableKey::SingleColumn(name) => {
                let hdu_py: Py<TableHDU> = slf.clone().unbind();
                Ok(Py::new(py, SingleColumnSubset { hdu: hdu_py, name })?
                    .into())
            }
            TableKey::MultiColumns(names) => {
                let hdu_py: Py<TableHDU> = slf.clone().unbind();
                Ok(Py::new(py, ColumnSubset { hdu: hdu_py, columns: names })?
                    .into())
            }
        }
    }
}

// What kind of selection the user passed to TableHDU.__getitem__.
// `Rows` covers both slices and integer iterables: in both cases the
// key flows through to read_table unchanged.  `SingleRow` is the
// bare-integer case (`hdu[5]`); read_table still does the I/O but
// the result is unwrapped to a numpy 0-d record (np.void) before
// returning, matching `structured_arr[i]` semantics.
enum TableKey {
    Rows,
    SingleRow(i64),
    SingleColumn(String),
    MultiColumns(Vec<String>),
}

// Try to extract `obj` as a string-like column name: str, bytes,
// numpy.str_, or numpy.bytes_.  Returns Ok(None) for anything else.
//
// Type checks are explicit (PyString / PyBytes instance checks) rather
// than relying on extract::<String>() / extract::<Vec<u8>>() — the
// latter is generic over iterables, so a Python list of small ints
// like [2, 0] would silently succeed as Vec<u8>=[2,0] and be
// mis-routed to a column lookup with control-char "name".
fn try_extract_column_name(obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
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

// Inspect the __getitem__ key and decide which path to take.  Rules:
//   - PySlice                            → Rows  (read flowing path)
//   - bare int (not bool)                → SingleRow (np.void scalar)
//   - single str/bytes/np.str_/np.bytes_ → SingleColumn
//   - non-empty iterable
//       first element string-like        → MultiColumns
//       first element int-like           → Rows
//       mixed or unknown                 → ValueError
//   - empty iterable                     → ValueError (ambiguous)
//   - anything else                      → ValueError
fn classify_table_key(key: &Bound<'_, PyAny>) -> PyResult<TableKey> {
    if key.is_instance_of::<PySlice>() {
        return Ok(TableKey::Rows);
    }
    if let Some(name) = try_extract_column_name(key)? {
        return Ok(TableKey::SingleColumn(name));
    }
    // Bare integer (not bool — Python bool is a subclass of int and
    // would otherwise sneak through).  Float/non-int Python objects
    // are rejected by extract::<i64>.
    if !key.is_instance_of::<PyBool>() {
        if let Ok(idx) = key.extract::<i64>() {
            return Ok(TableKey::SingleRow(idx));
        }
    }
    let iter = key.try_iter().map_err(|_| PyValueError::new_err(
        "TableHDU[key] requires a slice, an int (row index), a \
         str/bytes column name, an iterable of ints (rows), or an \
         iterable of str/bytes (columns)"
    ))?;
    let items: Vec<Bound<'_, PyAny>> = iter.collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Err(PyValueError::new_err(
            "TableHDU[key] received an empty sequence (ambiguous: rows \
             or columns?); pass a non-empty selection or use read() with \
             explicit rows=/columns="
        ));
    }
    let first = &items[0];
    if let Some(_) = try_extract_column_name(first)? {
        let names: Vec<String> = items.iter()
            .map(|i| try_extract_column_name(i)?.ok_or_else(|| {
                PyValueError::new_err(
                    "TableHDU[key] sequence mixes column names and \
                     non-string elements; pass all str/bytes (columns) \
                     or all int (rows)"
                )
            }))
            .collect::<PyResult<_>>()?;
        Ok(TableKey::MultiColumns(names))
    } else if !first.is_instance_of::<PyBool>() && first.extract::<i64>().is_ok() {
        // Defer per-element validation to resolve_rows; we only need to
        // route here.
        Ok(TableKey::Rows)
    } else {
        Err(PyValueError::new_err(
            "TableHDU[key] sequence elements must be all int (rows) or \
             all str/bytes (columns)"
        ))
    }
}

// Returned by hdu[col] for a single str/bytes column name.  Holds onto
// the parent TableHDU via Py<TableHDU> (a refcount bump) so the read
// can re-borrow it; carries the column name verbatim (case preserved,
// matching is case-insensitive at read time).
#[pyclass]
pub(crate) struct SingleColumnSubset {
    hdu: Py<TableHDU>,
    name: String,
}

#[pymethods]
impl SingleColumnSubset {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        Ok(format!(
            "<TableColumn '{}' of HDU #{}>",
            self.name, super_.index(),
        ))
    }

    // [rows] triggers the actual read.  `rows` may be a slice or any
    // iterable of ints (negative supported), with semantics matching
    // TableHDU.read_column(rows=...).
    fn __getitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        let cards = super_.header_snapshot()?;
        let data_offset = super_.offsets.data_offset();
        read_one_column(
            py, &cards, data_offset, &super_.file,
            &self.name, Some(rows), /* as_bytes = */ false,
            /* scale = */ true, /* mask_null = */ false,
        )
    }
}

// Returned by hdu[[col1, col2, ...]] for an iterable of column names.
// Same Py<TableHDU> hold-onto-parent pattern as SingleColumnSubset.
#[pyclass]
pub(crate) struct ColumnSubset {
    hdu: Py<TableHDU>,
    columns: Vec<String>,
}

#[pymethods]
impl ColumnSubset {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        Ok(format!(
            "<TableColumns {:?} of HDU #{}>",
            self.columns, super_.index(),
        ))
    }

    // [rows] triggers the actual read.  `rows` may be a slice or any
    // iterable of ints (negative supported), with semantics matching
    // TableHDU.read(rows=..., columns=...).
    fn __getitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        let cards = super_.header_snapshot()?;
        let data_offset = super_.offsets.data_offset();
        read_table(
            py, &cards, data_offset, &super_.file,
            Some(rows), Some(self.columns.clone()),
            /* scale = */ true, /* mask_null = */ false,
        )
    }
}
