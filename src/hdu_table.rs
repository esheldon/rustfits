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
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::common::{
    check_not_tainted, lock_file, parse_keyword, parse_keyword_float,
    parse_string_keyword, shift_file_tail_and_update_offsets, zero_fill_range,
    FileHandle, FileLayout, HduOffsets, RawBuffer, TaintFlag,
};
use crate::hdu::HDU;
use crate::hdu_image::{round_up_to_block, serialize_header_to_disk_bytes};
use crate::header::{card_int, card_string, card_uint, pad_to_card};

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

// ===========================================================================
// Write side (Phase 1a + 1b): scalar numeric BINTABLE columns
// ===========================================================================
//
// Phase 1a (MVP) added scalar i2/i4/i8/u1/f4/f8 with Identity (byteswap-
// only) transforms.  Phase 1b adds:
//   - u2/u4/u8 → I/J/K + TZERO=2^(n-1) (unsigned-int trick).  Write
//     transform: byteswap, then XOR the top bit of the high byte
//     (equivalent to subtracting 2^(n-1) in two's-complement).
//   - b1 (numpy bool) → L (FITS logical).  Per-byte: 0 → 0x46 ('F'),
//     nonzero → 0x54 ('T').  No byteswap.
//   - c8 / c16 → C / M.  Identity transform; byteswap_unit() returns
//     4 / 8 so the existing per-half byteswap is correct.
//
// Per the Table Write Roadmap in CLAUDE.md, subsequent commits add
// subarrays + TDIM (1c), strings (1d), and dict / list+names input
// forms (1e).
//
// Performance model: when the input numpy structured dtype's field
// offsets exactly match the HDU's on-disk column offsets (1a/1b
// guarantee this via validate — all 1b transforms preserve byte widths),
// the per-strip fill is one bulk memcpy from `src_bytes` into the strip
// buffer, followed by per-column in-place transform.  Slow-path (per-
// column strided copies) returns when 1d lets inputs deviate from the
// on-disk layout.

// Per-column metadata derived from the user's numpy structured dtype at
// create_table_hdu time, used to emit the BINTABLE header cards.
// `tzero` is Some only when the column uses the unsigned-int trick (its
// value is the power-of-two offset added on read to recover the
// unsigned interpretation).  `tdim` is Some only when the field is a
// 2-D-or-higher subarray (1-D shapes are fully described by the TFORM
// repeat count and don't need a TDIM card).  The stored string is
// already in FITS (FORTRAN, fastest-first) order.
// Per-column write spec.  Fixed columns carry tform_letter (the FITS
// type) + repeat + byte_width; VLA columns set var_kind to Some('P')
// or Some('Q'), with `tform_letter` repurposed as the INNER element
// letter (matching the read-side Column convention) and byte_width
// fixed at 8 (P) or 16 (Q) — the size of one descriptor in the main
// row.  repeat is always 1 for VLAs.
struct WriteColumn {
    name: String,
    tform_letter: char,
    repeat: usize,
    byte_width: usize,
    tzero: Option<u64>,
    tdim: Option<String>,
    tunit: Option<String>,
    var_kind: Option<char>,
}

// Classification of a single numpy field base dtype.  For numeric
// kinds, `chars_per_string` is None and `elem_bytes` is the per-
// element byte width on disk (and in memory).  For string kinds (S
// and U), `elem_bytes` is 1 (FITS 'A' is byte-oriented) and
// `chars_per_string` carries the per-string character count, which
// the caller uses to compute total bytes per cell and emit TDIM
// (strings need TDIM even for 1-D shapes — see dtype_to_write_columns).
struct ScalarClass {
    tform_letter: char,
    elem_bytes: usize,
    tzero: Option<u64>,
    chars_per_string: Option<usize>,
}

// Returns the classification for one numpy field's base dtype.
fn classify_scalar_numpy_field(
    field_dtype: &Bound<'_, PyAny>,
    col_name: &str,
) -> PyResult<ScalarClass> {
    let kind: String = field_dtype.getattr("kind")?.extract()?;
    let itemsize: usize = field_dtype.getattr("itemsize")?.extract()?;
    let err = |reason: &str| PyValueError::new_err(format!(
        "column '{}': numpy dtype kind '{}' itemsize {} — {}",
        col_name, kind, itemsize, reason));
    let plain = |letter: char, bytes: usize, tz: Option<u64>| ScalarClass {
        tform_letter: letter, elem_bytes: bytes, tzero: tz,
        chars_per_string: None,
    };
    match (kind.as_str(), itemsize) {
        ("i", 2) => Ok(plain('I', 2, None)),
        ("i", 4) => Ok(plain('J', 4, None)),
        ("i", 8) => Ok(plain('K', 8, None)),
        ("u", 1) => Ok(plain('B', 1, None)),
        ("u", 2) => Ok(plain('I', 2, Some(1u64 << 15))),
        ("u", 4) => Ok(plain('J', 4, Some(1u64 << 31))),
        ("u", 8) => Ok(plain('K', 8, Some(1u64 << 63))),
        ("f", 4) => Ok(plain('E', 4, None)),
        ("f", 8) => Ok(plain('D', 8, None)),
        ("c", 8) => Ok(plain('C', 8, None)),
        ("c", 16) => Ok(plain('M', 16, None)),
        ("b", 1) => Ok(plain('L', 1, None)),
        ("S", n) if n > 0 => Ok(ScalarClass {
            tform_letter: 'A', elem_bytes: 1, tzero: None,
            chars_per_string: Some(n),
        }),
        ("U", n) if n > 0 && n % 4 == 0 => Ok(ScalarClass {
            tform_letter: 'A', elem_bytes: 1, tzero: None,
            chars_per_string: Some(n / 4),
        }),
        ("S", 0) | ("U", 0) => Err(err("zero-length string column")),
        ("i", 1) => Err(err(
            "int8 has no native FITS BINTABLE code (deferred)")),
        ("f", 2) => Err(err("float16 has no FITS BINTABLE code")),
        _ => Err(err(
            "unsupported numpy dtype (supported: \
             i2/i4/i8/u1/u2/u4/u8/f4/f8/c8/c16/b1 scalars and S/U strings)")),
    }
}

// Classification for one VLA column's inner element type.  Maps a
// numpy dtype string (e.g. "f4", "i4") to the FITS inner-element
// letter + per-element byte width on disk.  Mirrors the read-side
// inner-letter → numpy dtype mapping in `field_dtype_and_shape`.
struct VarClass {
    inner_letter: char,
    elem_size: usize,
}

fn classify_var_numpy_field(
    inner_dtype: &str,
    col_name: &str,
) -> PyResult<VarClass> {
    let err = |reason: &str| PyValueError::new_err(format!(
        "var_dtypes['{}'] = '{}': {}", col_name, inner_dtype, reason));
    let s = inner_dtype
        .trim_start_matches(|c| c == '<' || c == '>' || c == '|' || c == '=');
    let normalized = s.to_lowercase();
    let (letter, size) = match normalized.as_str() {
        "u1" | "uint8"  => ('B', 1),
        "i2" | "int16"  => ('I', 2),
        "i4" | "int32"  => ('J', 4),
        "i8" | "int64"  => ('K', 8),
        "f4" | "float32" => ('E', 4),
        "f8" | "float64" => ('D', 8),
        "c8" | "complex64"  => ('C', 8),
        "c16" | "complex128" => ('M', 16),
        "?" | "b1" | "bool" | "bool_" => ('L', 1),
        _ => return Err(err(
            "unsupported inner dtype (supported: \
             u1/i2/i4/i8/f4/f8/c8/c16/? / bool)")),
    };
    Ok(VarClass { inner_letter: letter, elem_size: size })
}

// Pull the base dtype and numpy subarray shape out of a numpy field
// dtype.  For a scalar field, returns (field_dtype, []).  For a
// subarray field like ('f4', (3, 4)), returns (f4_dtype, [3, 4]).
fn extract_field_base_and_shape<'py>(
    field_dtype: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Vec<usize>)> {
    let subdtype = field_dtype.getattr("subdtype")?;
    if subdtype.is_none() {
        Ok((field_dtype.clone(), Vec::new()))
    } else {
        let tup = subdtype.cast::<PyTuple>()?;
        let base = tup.get_item(0)?;
        let shape: Vec<usize> = tup.get_item(1)?.extract()?;
        Ok((base, shape))
    }
}

// Walk a numpy structured dtype, emit per-column write specs in field
// order.  Subarray fields like ('flux', 'f4', (3, 4)) are supported in
// Phase 1c: TFORM repeat is the product of the subarray shape, TDIM is
// emitted in FITS (FORTRAN, fastest-first) order = reversed numpy
// shape.  1-D shapes are fully captured by the repeat count and TDIM
// is omitted (matches astropy convention).
fn dtype_to_write_columns(
    dtype: &Bound<'_, PyAny>,
    units: Option<&Bound<'_, PyDict>>,
    var_dtypes: Option<&Bound<'_, PyDict>>,
    descriptor: char,
) -> PyResult<Vec<WriteColumn>> {
    let names_attr = dtype.getattr("names")?;
    if names_attr.is_none() {
        return Err(PyValueError::new_err(
            "create_table_hdu: dtype must be a numpy structured dtype \
             with named fields (got a plain dtype)"));
    }
    let names: Vec<String> = names_attr.extract()?;
    if names.is_empty() {
        return Err(PyValueError::new_err(
            "create_table_hdu: dtype has no fields"));
    }
    // Build a set of every name the var_dtypes kwarg mentions so we
    // can reject keys that don't match any column.
    let var_dtypes_names: Option<std::collections::HashSet<String>> =
        if let Some(d) = var_dtypes {
            let mut s = std::collections::HashSet::new();
            for k in d.keys() {
                let key: String = k.extract().map_err(|_| {
                    PyValueError::new_err(
                        "var_dtypes keys must be strings")
                })?;
                s.insert(key);
            }
            Some(s)
        } else {
            None
        };
    let fields = dtype.getattr("fields")?;
    let mut out = Vec::with_capacity(names.len());
    for name in &names {
        let entry = fields.get_item(name.as_str())?;
        let entry_tup = entry.cast::<PyTuple>()?;
        let field_dtype = entry_tup.get_item(0)?;
        let (base_dtype, np_shape) =
            extract_field_base_and_shape(&field_dtype)?;
        let base_kind: String = base_dtype.getattr("kind")?.extract()?;

        // Object-dtype field → VLA column.  The user must specify
        // the inner element type via var_dtypes={name: 'f4'} (or
        // similar); without it we can't pick a TFORM letter.
        if base_kind == "O" {
            if !np_shape.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "column '{}': VLA columns must be scalar Object \
                     (numpy dtype 'O' with no subarray shape); got \
                     shape {:?}", name, np_shape)));
            }
            let inner_dtype_str: String = match var_dtypes {
                Some(d) => match d.get_item(name.as_str())? {
                    Some(v) => v.extract().map_err(|_| {
                        PyValueError::new_err(format!(
                            "var_dtypes['{}'] must be a string \
                             (e.g. 'f4', 'i4')", name))
                    })?,
                    None => return Err(PyValueError::new_err(format!(
                        "column '{}': Object dtype requires the inner \
                         element type via var_dtypes['{}'] = ...",
                        name, name))),
                },
                None => return Err(PyValueError::new_err(format!(
                    "column '{}': Object dtype requires the inner \
                     element type via the var_dtypes= kwarg",
                    name))),
            };
            let vc = classify_var_numpy_field(&inner_dtype_str, name)?;
            let descriptor_size = if descriptor == 'P' { 8 } else { 16 };
            let tunit = units.and_then(|d| {
                d.get_item(name.as_str()).ok().flatten()
                    .and_then(|v| v.extract::<String>().ok())
            });
            // vc.elem_size is consumed by the write-path heap writer
            // via bytes_per_element(tform_letter); no per-column copy
            // is stored here.
            let _ = vc.elem_size;
            out.push(WriteColumn {
                name: name.clone(),
                tform_letter: vc.inner_letter,
                repeat: 1,
                byte_width: descriptor_size,
                tzero: None,
                tdim: None,
                tunit,
                var_kind: Some(descriptor),
            });
            continue;
        }

        let cls = classify_scalar_numpy_field(&base_dtype, name)?;
        let array_count: usize =
            np_shape.iter().copied().product::<usize>().max(1);
        // Total byte_width (and TFORM repeat for 'A' columns) depends
        // on whether this is a string column: A counts total bytes
        // per cell (chars × strings), other letters count elements.
        let (repeat, byte_width) = match cls.chars_per_string {
            Some(chars) => {
                let total_bytes = chars * array_count;
                (total_bytes, total_bytes)
            }
            None => (array_count, cls.elem_bytes * array_count),
        };
        // TDIM (FITS = FORTRAN, fastest-first).  Two distinct rules:
        //   - Strings ('A'): TDIM is (chars_per_string, ...reversed
        //     np_shape) and is required whenever np_shape is
        //     non-empty, because TFORM='NA' alone is ambiguous
        //     between "one N-char string" and "N 1-char strings".
        //   - Numeric / bool / complex: TDIM only for rank ≥ 2.  1-D
        //     fields are fully described by the TFORM repeat.
        let tdim = match cls.chars_per_string {
            Some(chars) if !np_shape.is_empty() => {
                let mut dims: Vec<String> =
                    Vec::with_capacity(np_shape.len() + 1);
                dims.push(chars.to_string());
                dims.extend(np_shape.iter().rev().map(|d| d.to_string()));
                Some(format!("({})", dims.join(",")))
            }
            None if np_shape.len() >= 2 => {
                let dims: Vec<String> = np_shape.iter().rev()
                    .map(|d| d.to_string()).collect();
                Some(format!("({})", dims.join(",")))
            }
            _ => None,
        };
        let tunit = units.and_then(|d| {
            d.get_item(name.as_str()).ok().flatten()
                .and_then(|v| v.extract::<String>().ok())
        });
        out.push(WriteColumn {
            name: name.clone(),
            tform_letter: cls.tform_letter,
            repeat,
            byte_width,
            tzero: cls.tzero,
            tdim,
            tunit,
            var_kind: None,
        });
    }
    // Reject var_dtypes keys that don't match any column — usually
    // a typo.  Build the set of matched names from the final out and
    // diff against the user-supplied keys.
    if let Some(provided) = var_dtypes_names {
        let column_names: std::collections::HashSet<String> =
            out.iter().map(|c| c.name.clone()).collect();
        for k in &provided {
            if !column_names.contains(k) {
                return Err(PyValueError::new_err(format!(
                    "var_dtypes contains key '{}' that does not match \
                     any column in the dtype", k)));
            }
        }
    }
    Ok(out)
}

// Build the BINTABLE header card sequence (structural keys + EXTNAME/
// EXTVER if given + per-column TTYPEn/TFORMn/TUNITn + END).  Padding to
// the BLOCK_SIZE boundary happens at write time in create_table_hdu.
fn build_bintable_header_cards(
    write_columns: &[WriteColumn],
    nrows: i64,
    extname: Option<&str>,
    extver: Option<i64>,
) -> Vec<String> {
    let row_width: usize = write_columns.iter().map(|c| c.byte_width).sum();
    let mut cards: Vec<String> = Vec::new();
    cards.push(card_string("XTENSION", "BINTABLE", "binary table extension"));
    cards.push(card_int("BITPIX", 8, "8-bit bytes"));
    cards.push(card_int("NAXIS", 2, "2-dimensional binary table"));
    cards.push(card_int("NAXIS1", row_width as i64, "width of table in bytes"));
    cards.push(card_int("NAXIS2", nrows, "number of rows in table"));
    cards.push(card_int("PCOUNT", 0, "size of special data area"));
    cards.push(card_int("GCOUNT", 1, "one data group (required keyword)"));
    cards.push(card_int(
        "TFIELDS", write_columns.len() as i64, "number of columns"));
    if let Some(name) = extname {
        cards.push(card_string("EXTNAME", name, "name of this HDU"));
    }
    if let Some(ver) = extver {
        cards.push(card_int("EXTVER", ver, "extension version"));
    }
    for (i, col) in write_columns.iter().enumerate() {
        let n = i + 1;
        let tform = match col.var_kind {
            Some(desc) => format!("1{}{}", desc, col.tform_letter),
            None => format!("{}{}", col.repeat, col.tform_letter),
        };
        cards.push(card_string(
            &format!("TTYPE{}", n), &col.name, "label for column"));
        cards.push(card_string(
            &format!("TFORM{}", n), &tform, "data format of column"));
        if let Some(tdim) = &col.tdim {
            cards.push(card_string(
                &format!("TDIM{}", n), tdim,
                "array dimensions (FORTRAN, fastest-first)"));
        }
        if let Some(tz) = col.tzero {
            cards.push(card_uint(
                &format!("TZERO{}", n), tz,
                "offset for unsigned integer (unsigned-int trick)"));
        }
        if let Some(unit) = &col.tunit {
            cards.push(card_string(
                &format!("TUNIT{}", n), unit, "physical unit of column"));
        }
    }
    cards.push(pad_to_card("END"));
    cards
}

// Used by FITS.create_table_hdu: the user-facing entry takes a PyAny
// that may be a numpy.dtype OR a descr list of tuples.  Normalize
// through numpy.dtype(), then emit cards.  Returns (cards, row_width).
pub(crate) fn normalize_and_build_table_header(
    py: Python<'_>,
    dtype_in: &Bound<'_, PyAny>,
    nrows: i64,
    extname: Option<&str>,
    extver: Option<i64>,
    units: Option<&Bound<'_, PyDict>>,
    var_dtypes: Option<&Bound<'_, PyDict>>,
    descriptor: char,
) -> PyResult<(Vec<String>, u64)> {
    let np = py.import("numpy")?;
    let np_dtype = np.getattr("dtype")?.call1((dtype_in,))?;
    let write_columns = dtype_to_write_columns(
        &np_dtype, units, var_dtypes, descriptor)?;
    let row_width: u64 = write_columns.iter()
        .map(|c| c.byte_width as u64).sum();
    let cards = build_bintable_header_cards(
        &write_columns, nrows, extname, extver);
    Ok((cards, row_width))
}

// Per-column transform applied during the strip-by-strip write.
// Identity / UnsignedXor / BoolToLogical / BytesCopy preserve byte
// width (source per-cell width == FITS per-cell width); they can run
// in place on a strip buffer pre-filled by bulk memcpy (the fast
// path).  UnicodeToAscii grows or shrinks width (src is 4×dst, since
// numpy U is UTF-32-LE while FITS A is one byte per char), so it can
// only run on the slow path with explicit per-column strided copies
// from source bytes into the destination buffer.
#[derive(Debug, Clone, Copy)]
enum WriteTransform {
    // Per-element byteswap by `elem_w` bytes.  `elem_w==1` is a no-op
    // copy (covers B/u1).  Complex types use elem_w==4 (C) or
    // elem_w==8 (M) so real/imag halves swap independently.
    Identity { elem_w: usize, num_elems: usize },
    // u2/u4/u8 unsigned-int trick: byteswap, then XOR top bit of the
    // resulting high byte (equivalent to subtracting 2^(n-1) in two's
    // complement so the stored value is the signed-int representation
    // matching the column's TZERO).
    UnsignedXor { elem_w: usize, num_elems: usize },
    // numpy bool (1 byte 0/1) → FITS L (1 byte ASCII 'T'=0x54 or
    // 'F'=0x46).  Per-byte over `num_bytes`.  No byteswap needed.
    BoolToLogical { num_bytes: usize },
    // numpy S<n> → FITS A.  Verbatim byte copy of `num_bytes` per
    // cell.  No swap, no validation — the bytes already represent
    // the on-disk characters.
    BytesCopy { num_bytes: usize },
    // numpy U<n> (UTF-32-LE, 4 bytes per codepoint) → FITS A.  For
    // each of `num_chars` codepoints, validate that it fits in 7-bit
    // ASCII and emit a single byte.  src_size = 4 * num_chars,
    // dst_size = num_chars.  Non-ASCII codepoints raise ValueError
    // up the stack (no silent lossy conversion).
    UnicodeToAscii { num_chars: usize },
}

// Resolve the per-column write transform given the on-disk column
// (tform_letter + tzero) and the input numpy field's (kind, base
// itemsize).  Rejects mismatches with a message naming both sides.
fn column_transform(
    col: &Column,
    input_kind: &str,
    input_size: usize,
) -> PyResult<WriteTransform> {
    // 'A' column: input must be numpy S<n> or U<n> with the per-
    // string char count matching the column's per-string width
    // (derived from TDIM or the bare repeat).
    if col.tform_letter == 'A' {
        let per_string_bytes = match &col.tdim {
            Some(tdim) => tdim[0],
            None => col.repeat,
        };
        if input_kind == "S" && input_size == per_string_bytes {
            return Ok(WriteTransform::BytesCopy { num_bytes: col.byte_width });
        }
        if input_kind == "U" && input_size == 4 * per_string_bytes {
            return Ok(WriteTransform::UnicodeToAscii {
                num_chars: col.byte_width,
            });
        }
        return Err(PyValueError::new_err(format!(
            "column '{}' (A, {} chars/string): expected numpy S{} or U{} \
             input, got kind '{}' itemsize {}",
            col.name, per_string_bytes, per_string_bytes, per_string_bytes,
            input_kind, input_size)));
    }

    // Natural (no-scaling) input dtype per FITS letter; swap_unit is
    // the per-element byteswap width (== bytes_per_element for most,
    // half-element for complex).
    let (nat_kind, nat_size, swap_unit) = match col.tform_letter {
        'L' => ("b", 1, 1),
        'B' => ("u", 1, 1),
        'I' => ("i", 2, 2),
        'J' => ("i", 4, 4),
        'K' => ("i", 8, 8),
        'E' => ("f", 4, 4),
        'D' => ("f", 8, 8),
        'C' => ("c", 8, 4),
        'M' => ("c", 16, 8),
        c => return Err(PyValueError::new_err(format!(
            "column '{}': unsupported TFORM letter '{}' on write",
            col.name, c))),
    };
    let num_elems = col.byte_width / swap_unit;

    // BoolToLogical: input is numpy bool and column is L.
    if col.tform_letter == 'L' {
        if input_kind == "b" && input_size == 1 {
            return Ok(WriteTransform::BoolToLogical { num_bytes: num_elems });
        }
        return Err(PyValueError::new_err(format!(
            "column '{}' (L): expected numpy bool input, got kind '{}' \
             itemsize {}", col.name, input_kind, input_size)));
    }

    // Unsigned-int trick: column letter + TZERO matches one of the
    // power-of-two offsets, and TSCAL is 1.
    let unsigned_trick = col.tscal == 1.0 && matches!(
        (col.tform_letter, col.tzero),
        ('I', t) if t == 32768.0
    ) || col.tscal == 1.0 && matches!(
        (col.tform_letter, col.tzero),
        ('J', t) if t == 2147483648.0
    ) || col.tscal == 1.0 && matches!(
        (col.tform_letter, col.tzero),
        ('K', t) if t == 9223372036854775808.0
    );
    if unsigned_trick {
        if input_kind == "u" && input_size == nat_size {
            return Ok(WriteTransform::UnsignedXor {
                elem_w: swap_unit, num_elems,
            });
        }
        return Err(PyValueError::new_err(format!(
            "column '{}' ({} + TZERO unsigned-int trick): expected u{} \
             input, got kind '{}' itemsize {}",
            col.name, col.tform_letter, nat_size * 8,
            input_kind, input_size)));
    }

    // Natural mapping: input matches the no-scaling expected dtype.
    if input_kind == nat_kind && input_size == nat_size
        && col.tscal == 1.0 && col.tzero == 0.0
    {
        return Ok(WriteTransform::Identity {
            elem_w: swap_unit, num_elems,
        });
    }

    // Other scaling on the column (general TSCAL/TZERO) is read-only
    // for now — write-side support has no implementation yet.
    if col.tscal != 1.0 || col.tzero != 0.0 {
        return Err(PyValueError::new_err(format!(
            "column '{}' has TSCAL/TZERO scaling other than the unsigned-\
             int trick; writing scaled columns is not yet supported",
            col.name)));
    }
    Err(PyValueError::new_err(format!(
        "column '{}' ({}): expected input dtype kind '{}' itemsize {}, \
         got kind '{}' itemsize {}",
        col.name, col.tform_letter, nat_kind, nat_size,
        input_kind, input_size)))
}

// Expected per-cell numpy shape for an on-disk column.  Mirrors the
// read side's dtype-building rule:
//   - 'A' columns: TDIM present → reversed(tdim[1..]) (first TDIM
//     dim is per-string width, NOT a user-visible axis); TDIM absent
//     → () (scalar U<repeat>).
//   - Other letters: TDIM present → reversed(tdim); else (repeat,)
//     for 1-D; else () for scalar.
fn column_expected_shape(col: &Column) -> Vec<usize> {
    if col.tform_letter == 'A' {
        return match &col.tdim {
            Some(tdim) => tdim[1..].iter().rev().copied().collect(),
            None => Vec::new(),
        };
    }
    match &col.tdim {
        Some(tdim) => tdim.iter().rev().copied().collect(),
        None => if col.repeat > 1 { vec![col.repeat] } else { Vec::new() },
    }
}

// Per-column source view for the strip writer.
//
// For structured-array input, all columns share the same `src_bytes`
// (the array's raw buffer) and `src_row_stride` (the array's itemsize),
// with each column's `src_offset` set to its field offset within a row.
//
// For dict / list+names input, each column has its OWN `src_bytes`
// (its own ndarray's raw buffer), `src_offset` is 0, and
// `src_row_stride == src_total_size` (the per-cell byte count).
struct ColumnSource<'a> {
    src_bytes: &'a [u8],
    src_offset: usize,
    src_row_stride: usize,
    src_total_size: usize,
}

// Per-column source metadata that doesn't carry the borrowed bytes,
// so the per-input-form preparation functions can build it before the
// final Vec<ColumnSource> is assembled.  buffer_idx indexes into a
// per-call Vec<RawBuffer> owned by the write pymethod.
struct ColumnSourceMeta {
    buffer_idx: usize,
    src_offset: usize,
    src_row_stride: usize,
    src_total_size: usize,
}

// Result of input preparation, common across all input forms.
struct PreparedInput {
    transforms: Vec<WriteTransform>,
    metas: Vec<ColumnSourceMeta>,
    // True iff the fast-path bulk-memcpy is safe: all sources share
    // the same buffer, each src_offset == col.byte_offset, each
    // src_total_size == col.byte_width, and src_row_stride == row_width.
    layout_matches: bool,
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
fn prepare_structured_input(
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
fn acquire_per_column_array(
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
fn apply_transform_cell(
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
fn write_table_data(
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
fn write_table_strided(
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
fn write_table_one_column(
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
fn coerce_to_len1_record<'py>(
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

// Build a Vec<ColumnSource> by walking PreparedInput.metas and the
// per-call Vec<RawBuffer>.  Same pattern used by the bulk write entry
// point; factored out so the setitem helpers share it.
fn build_sources<'a>(
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

// hdu[i] = record: overwrite a single row.  The value is coerced into
// a length-1 structured ndarray and validated against the HDU columns
// the same way bulk write validates; the write then targets the byte
// range [data_offset + i*row_width, +row_width).
#[allow(clippy::too_many_arguments)]
fn setitem_single_row(
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
fn setitem_row_slice(
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
fn setitem_single_column(
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
// VLA write support (Phase 4)
// ---------------------------------------------------------------------------

// True iff any column is variable-length (P/Q).  Dispatches the write
// path: fixed-only tables take the existing fast/slow strip writer;
// tables with any VLA column take the heap-aware path below.
fn any_var_column(columns: &[Column]) -> bool {
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
fn extract_per_column_inputs<'py>(
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
        'L' => ("b", 1),
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
struct VlaCellPlan {
    nelements: usize,
    bytes_offset_in_heap: usize,
}

// Validate one VLA cell's ndarray + return its element count.  The
// cell must be a 1-D numpy ndarray with C-contiguous layout and the
// dtype matching the column's inner letter.  Empty cells (nelements
// == 0) are accepted (descriptor is just (0, current_heap_offset)).
fn validate_vla_cell(
    cell: &Bound<'_, PyAny>,
    ndarray: &Bound<'_, PyAny>,
    inner_letter: char,
    col_name: &str,
    row_idx: usize,
) -> PyResult<usize> {
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
fn plan_vla_heap_layout(
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
            let elem_size = bytes_per_element(col.tform_letter)
                .unwrap_or(0);
            let cell = per_col[col_idx].get_item(row_idx)?;
            let nelements = validate_vla_cell(
                &cell, ndarray, col.tform_letter, &col.name, row_idx)?;
            let bytes = nelements * elem_size;
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
fn serialize_vla_cell(
    cell: &Bound<'_, PyAny>,
    inner_letter: char,
    nelements: usize,
    dst: &mut [u8],
) -> PyResult<()> {
    if nelements == 0 {
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
fn write_descriptor(
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
struct FixedColInfo {
    buffer: RawBuffer,
    per_cell_bytes: usize,
    transform: WriteTransform,
}

struct VlaColInfo<'py> {
    // Per-row (nelements, heap_offset) plans, indexed by input row.
    plans: Vec<VlaCellPlan>,
    // The 1-D Object ndarray (held so the heap-serialization pass
    // can `arr[i]` to get each row's cell).
    per_col_array: Bound<'py, PyAny>,
}

// Validate a fixed column's per-column input ndarray against the on-
// disk column and acquire its raw buffer.  Mirrors
// acquire_per_column_array but the inputs are already extracted into
// per-column ndarrays by extract_per_column_inputs, so this function
// can borrow the buffer directly into the per-column FixedColInfo
// without going through the shared Vec<RawBuffer> indirection that
// the prepare_*_input functions need.
fn build_fixed_col_info(
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
fn write_vla_data_range(
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
    tainted: &TaintFlag,
) -> PyResult<usize> {
    if input_nrows == 0 {
        return Ok(0);
    }
    // Build the heap buffer in memory.  For very large heaps this
    // could be streamed but for MVP we accumulate; total_heap_bytes
    // is the upper bound (matches the planner output).
    let added_heap_bytes = total_heap_bytes - heap_start_offset_in_heap;
    let mut heap_buf: Vec<u8> = vec![0u8; added_heap_bytes];
    for (col_idx, col) in columns.iter().enumerate() {
        let Some(vci) = &vla[col_idx] else { continue; };
        let elem_size = bytes_per_element(col.tform_letter)
            .unwrap_or(0);
        for input_row in 0..input_nrows {
            let plan = vci.plans[input_row];
            if plan.nelements == 0 { continue; }
            let cell = vci.per_col_array.get_item(input_row)?;
            let local_off =
                plan.bytes_offset_in_heap - heap_start_offset_in_heap;
            let n_bytes = plan.nelements * elem_size;
            let dst = &mut heap_buf[local_off..local_off + n_bytes];
            serialize_vla_cell(&cell, col.tform_letter, plan.nelements, dst)?;
        }
    }

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

    // Now write the heap.  Single seek + write; heap_buf can be
    // large but we already committed to building it in RAM.
    if !heap_buf.is_empty() {
        f.seek(SeekFrom::Start(heap_start_offset_in_file))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&heap_buf) {
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
    Ok(added_heap_bytes)
}

// VLA-aware append path.  Mirrors the fixed append flow but
// additionally:
//   - Plans the heap layout starting at the current PCOUNT (so
//     descriptors for new rows point to offsets after the existing
//     heap).
//   - Relocates the existing heap forward (within the data section)
//     to sit after the appended main rows.
// Bulk write path for tables with no VLA columns.  Validates input
// against the table schema, then dispatches to write_table_data,
// which writes contiguous main-section rows.
fn write_fixed_only(
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

// Bulk write path for tables with at least one VLA (P/Q) column.
// Validates fixed + VLA columns, plans the heap layout from scratch
// (a full overwrite resets the heap to start at offset 0), grows the
// data section if needed, writes main rows + heap, then updates
// PCOUNT in the header.
#[allow(clippy::too_many_arguments)]
fn write_vla_aware(
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

    // PCOUNT update — disk-write-before-commit ordering.
    let mut cards_guard = super_.header.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards_guard.clone();
    set_pcount_in_cards(&mut new_cards, total_heap_bytes as u64);
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

//   - Updates PCOUNT alongside NAXIS2 in the header rewrite.
// Reads the old heap into memory once, before any byte movement,
// and writes it back to its new position after the new main rows
// are in place.  For very large heaps this could chunk; MVP is
// in-RAM for clarity.
#[allow(clippy::too_many_arguments)]
fn append_vla_aware(
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

// Append rows to a table with no VLA columns.  Validates input, grows
// the data section if needed (last-HDU branch uses set_len; non-last
// branch shifts the file tail and zero-fills the gap), rewrites
// NAXIS2 on disk, then writes the appended rows at the end of the
// existing data section.
#[allow(clippy::too_many_arguments)]
fn append_fixed_only(
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
            file.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
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
    let mut cards_guard = super_.header.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards_guard.clone();
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
    *cards_guard = new_cards;
    drop(cards_guard);

    // A write failure here taints — header already advertises the
    // larger NAXIS2 but the new rows are partly or wholly stale.
    let sources = build_sources(&prep.metas, &buffers);
    let append_offset = data_offset
        + (current_nrows * row_width) as u64;
    write_table_data(
        columns, &prep.transforms, &sources, prep.layout_matches,
        &super_.file, append_offset, append_nrows, row_width,
        &super_.tainted,
    ).map_err(|e| {
        super_.tainted.store(true, Ordering::Release);
        e
    })
}

// Rewrite (or insert) the PCOUNT card in `new_cards` to `new_pcount`.
// PCOUNT is mandatory in BINTABLE headers so we expect it to exist;
// fall back to inserting it just before TFIELDS if it's missing,
// which keeps things sane for hand-built headers.
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
fn determine_input_nrows(
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
fn dispatch_write_input(
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

// What kind of selection the user passed to TableHDU.__setitem__.
// Scope is intentionally narrower than __getitem__'s TableKey:
// multi-column subset writes, fancy row-list writes, and (row, col)
// tuple writes are all rejected with a clear message until a use case
// for them shows up.
enum SetItemKey {
    SingleRow(i64),
    RowSlice,
    SingleColumn(String),
}

fn classify_setitem_key(key: &Bound<'_, PyAny>) -> PyResult<SetItemKey> {
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
    Err(PyValueError::new_err(
        "TableHDU[key] = value: key must be an int (single row), a slice \
         (range of rows), or a str/bytes column name; other forms are \
         not yet supported"))
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

    // Number of rows (NAXIS2).  Returns 0 if the keyword is absent
    // (malformed header) or set to a negative value.
    #[getter]
    fn nrows(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize)
    }

    // Number of columns (TFIELDS).
    #[getter]
    fn ncols(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "TFIELDS").unwrap_or(0).max(0) as usize)
    }

    // Column names in file order (case preserved verbatim).  Returns
    // a Python tuple so the value is immutable from the caller side.
    #[getter]
    fn colnames(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let columns = parse_columns(&cards)?;
        let names: Vec<&str> = columns.iter().map(|c| c.name.as_str()).collect();
        Ok(PyTuple::new(py, &names)?.unbind())
    }

    // Pythonic length: `len(table_hdu)` == row count.  Mirrors
    // `len(structured_array)` for the equivalent numpy structured
    // array a full read would return.
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize)
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

    // Bulk-write data into this table's data section.  Three input
    // forms are accepted; all normalize through the same per-column
    // strip-write kernel:
    //
    //   - Structured numpy ndarray.  Field names must match the HDU's
    //     columns (extras, missing, or duplicates rejected); field
    //     order may differ from HDU order (the slow path handles
    //     reordering).  `len(data)` must equal NAXIS2.
    //
    //   - Dict of {name: ndarray}.  One entry per HDU column;
    //     extras / missing rejected.  Each value is a per-column
    //     ndarray with shape (NAXIS2,) + per-cell shape.
    //
    //   - List/tuple of ndarrays + `names=[...]` keyword.  Parallel
    //     sequences; same per-column model as dict.
    //
    // The fast path (one bulk memcpy per strip + per-column in-place
    // transform) is used when the input is a structured ndarray
    // whose fields are in HDU order with no width / offset mismatches
    // (no U columns, no padding).  All other cases run the slow path
    // (per-column strided copy + per-cell transform).
    #[pyo3(signature = (data, *, names=None))]
    fn write(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let super_ = slf.into_super();
        check_not_tainted(&super_.tainted)?;
        let cards = super_.header_snapshot()?;
        let nrows =
            parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize;
        let row_width =
            parse_keyword(&cards, "NAXIS1").unwrap_or(0).max(0) as usize;
        let columns = parse_columns(&cards)?;
        let data_offset = super_.offsets.data_offset();

        if any_var_column(&columns) {
            write_vla_aware(
                py, &super_, &cards, &columns, data, names,
                nrows, row_width, data_offset)
        } else {
            write_fixed_only(
                py, &super_, &columns, data, names,
                nrows, row_width, data_offset)
        }
    }

    // hdu[key] = value dispatches based on what `key` looks like:
    //
    //   bare int (not bool) → single-row write at row index `key`
    //     (negative supported); `value` must be a numpy.void record
    //     or a length-1 structured ndarray.
    //   slice → range-of-rows write; `value` must be a structured
    //     ndarray of length equal to the slicelength.  step=1 uses
    //     the bulk-write fast path; step>1 does per-row writes.
    //     step<=0 is rejected.
    //   single str/bytes/np.str_/np.bytes_ → whole-column write
    //     across all rows; `value` must be an ndarray of shape
    //     (nrows,) + per-cell shape, matching what __getitem__
    //     would return for that column.
    //
    // Multi-column subset writes, (row, col) tuple writes, and fancy
    // row-list writes are rejected; add when a use case shows up.
    fn __setitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let super_ = slf.into_super();
        check_not_tainted(&super_.tainted)?;
        let cards = super_.header_snapshot()?;
        let nrows =
            parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize;
        let row_width =
            parse_keyword(&cards, "NAXIS1").unwrap_or(0).max(0) as usize;
        let columns = parse_columns(&cards)?;
        let data_offset = super_.offsets.data_offset();
        let kind = classify_setitem_key(key)?;
        match kind {
            SetItemKey::SingleRow(i) => setitem_single_row(
                py, &columns, &super_.file, data_offset, nrows, row_width,
                i, value, &super_.tainted),
            SetItemKey::RowSlice => {
                let slice_py = key.cast::<PySlice>()?;
                setitem_row_slice(
                    py, &columns, &super_.file, data_offset, nrows,
                    row_width, slice_py, value, &super_.tainted)
            }
            SetItemKey::SingleColumn(name) => setitem_single_column(
                py, &columns, &super_.file, data_offset, nrows, row_width,
                &name, value, &super_.tainted),
        }
    }

    // Append rows to the table.  Grows NAXIS2 in the header and the
    // data section to fit the new rows; for HDUs that are not the last
    // on disk, the file tail is shifted forward and every later HDU's
    // offsets are bumped in lockstep (shared shift_file_tail primitive
    // — see CLAUDE.md "Image overflow: in-place data-section grow"
    // and "Header overflow: in-place file grow").  Accepts the same
    // three input forms as TableHDU.write: structured ndarray, dict
    // {name: ndarray}, or list/tuple of ndarrays with names=[...].
    //
    // Order of operations is validate-then-mutate: input is fully
    // validated (columns, dtypes, shapes) before any file or header
    // bytes are touched, so a dtype mismatch can't leave the file
    // half-grown.  After validation: grow the file → write the new
    // NAXIS2 card → write the new rows.  Any mid-write I/O failure
    // taints the file (close + reopen to recover).
    #[pyo3(signature = (data, *, names=None))]
    fn append(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let super_: PyRefMut<HDU> = slf.into_super();
        check_not_tainted(&super_.tainted)?;
        let cards = super_.header_snapshot()?;
        let current_nrows = parse_keyword(&cards, "NAXIS2")
            .unwrap_or(0).max(0) as usize;
        let row_width = parse_keyword(&cards, "NAXIS1")
            .unwrap_or(0).max(0) as usize;
        let columns = parse_columns(&cards)?;
        let data_offset = super_.offsets.data_offset();

        // Validate-then-mutate: determine append size and run input
        // validation BEFORE touching the file, so a dtype error
        // leaves the file untouched.
        let append_nrows = determine_input_nrows(data, names)?;
        if append_nrows == 0 {
            return Ok(());
        }
        let new_nrows = current_nrows + append_nrows;

        if any_var_column(&columns) {
            append_vla_aware(
                py, &super_, &cards, &columns, data, names,
                current_nrows, append_nrows, new_nrows, row_width,
                data_offset)
        } else {
            append_fixed_only(
                py, &super_, &columns, data, names,
                current_nrows, append_nrows, new_nrows, row_width,
                data_offset)
        }
    }

    // Alias for append().  Kept for symmetry with ImageHDU.extend so
    // generic code that iterates HDUs and calls .extend(...) on each
    // continues to work.  The primary table-side name is `append`
    // because that's the natural verb for adding rows to a table.
    #[pyo3(signature = (data, *, names=None))]
    fn extend(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Self::append(slf, py, data, names)
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
