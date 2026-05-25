// TableHDU column metadata: the `Column` struct, TFORM/TDIM/TNULL
// parsing, and the TSCAL/TZERO scaling classifier (used by both read
// and write paths).
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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::{parse_keyword, parse_keyword_float, parse_string_keyword};

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
pub(crate) fn bytes_per_element(letter: char) -> Option<usize> {
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
pub(crate) fn byteswap_unit(letter: char) -> usize {
    match letter {
        // X (bit-packed) is byte-flat on disk; no swap needed.
        'L' | 'B' | 'A' | 'X' => 1,
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
pub(crate) enum ScalingKind {
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
pub(crate) fn scaling_kind(col: &Column) -> PyResult<ScalingKind> {
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
pub(crate) fn scaled_output_dtype(letter: char, kind: ScalingKind) -> &'static str {
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
