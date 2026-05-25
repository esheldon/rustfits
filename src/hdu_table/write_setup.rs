// Write-side setup: WriteColumn metadata + scalar/var classification +
// numpy structured dtype → BINTABLE header card builder +
// WriteTransform classifier + column_expected_shape.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::header::{card_int, card_string, card_uint, pad_to_card};

use super::columns::Column;

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
pub(crate) struct WriteColumn {
    pub(crate) name: String,
    pub(crate) tform_letter: char,
    pub(crate) repeat: usize,
    pub(crate) byte_width: usize,
    pub(crate) tzero: Option<u64>,
    pub(crate) tdim: Option<String>,
    pub(crate) tunit: Option<String>,
    pub(crate) var_kind: Option<char>,
}

// Caller's `bit_columns=` kwarg, parsed from the Python form.  `All`
// means "promote every b1 column to X" (matches fitsio's
// `write_bitcols=True`).  `Names(set)` means "promote only the
// listed names" (case-insensitive against the table columns; names
// the user passed are looked up verbatim, the comparison happens
// inside the classifier).  `None` at the call site means "default":
// b1 columns map to L as before.
pub(crate) enum BitColumnsSpec {
    All,
    Names(std::collections::HashSet<String>),
}

impl BitColumnsSpec {
    fn contains(&self, name: &str) -> bool {
        match self {
            BitColumnsSpec::All => true,
            BitColumnsSpec::Names(set) => {
                set.contains(&name.to_uppercase())
            }
        }
    }
}

// Classification of a single numpy field base dtype.  For numeric
// kinds, `chars_per_string` is None and `elem_bytes` is the per-
// element byte width on disk (and in memory).  For string kinds (S
// and U), `elem_bytes` is 1 (FITS 'A' is byte-oriented) and
// `chars_per_string` carries the per-string character count, which
// the caller uses to compute total bytes per cell and emit TDIM
// (strings need TDIM even for 1-D shapes — see dtype_to_write_columns).
// `bit_packed=true` means a numpy bool field opted in to FITS X
// (one bit per element on disk, MSB-packed); the caller computes
// byte_width = ceil(repeat/8) instead of repeat * elem_bytes and
// emits TFORM=NX.
struct ScalarClass {
    tform_letter: char,
    elem_bytes: usize,
    tzero: Option<u64>,
    chars_per_string: Option<usize>,
    bit_packed: bool,
}

// Returns the classification for one numpy field's base dtype.
// `bit_packed=true` is set only when the caller has opted this
// column into FITS X via the bit_columns= kwarg AND the field is
// b1; for any other dtype + bit_packed=true we reject.
fn classify_scalar_numpy_field(
    field_dtype: &Bound<'_, PyAny>,
    col_name: &str,
    bit_packed_opt_in: bool,
) -> PyResult<ScalarClass> {
    let kind: String = field_dtype.getattr("kind")?.extract()?;
    let itemsize: usize = field_dtype.getattr("itemsize")?.extract()?;
    let err = |reason: &str| PyValueError::new_err(format!(
        "column '{}': numpy dtype kind '{}' itemsize {} — {}",
        col_name, kind, itemsize, reason));
    let plain = |letter: char, bytes: usize, tz: Option<u64>| ScalarClass {
        tform_letter: letter, elem_bytes: bytes, tzero: tz,
        chars_per_string: None, bit_packed: false,
    };
    if bit_packed_opt_in {
        if kind != "b" || itemsize != 1 {
            return Err(PyValueError::new_err(format!(
                "column '{}': bit_columns= entry requires numpy bool \
                 (kind 'b1') input, got kind '{}' itemsize {}",
                col_name, kind, itemsize)));
        }
        return Ok(ScalarClass {
            tform_letter: 'X', elem_bytes: 1, tzero: None,
            chars_per_string: None, bit_packed: true,
        });
    }
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
            chars_per_string: Some(n), bit_packed: false,
        }),
        ("U", n) if n > 0 && n % 4 == 0 => Ok(ScalarClass {
            tform_letter: 'A', elem_bytes: 1, tzero: None,
            chars_per_string: Some(n / 4), bit_packed: false,
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
    // String VLA aliases (FITS letter 'A').  Check BEFORE the numeric
    // lowercase match below — numpy's uppercase 'S' / 'U' string-kind
    // characters lowercase to 'u' / 's', which would collide with the
    // numeric uint8 ('u1') entry.  Bare lowercase 'u' / 's' are NOT
    // string aliases (no numpy precedent).  Both 'A' and 'a' accepted
    // because 'A' is a FITS letter rather than a numpy dtype.
    match s {
        "S" | "U" | "S1" | "U1" | "A" | "a" => {
            return Ok(VarClass { inner_letter: 'A', elem_size: 1 });
        }
        _ => {}
    }
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
             u1/i2/i4/i8/f4/f8/c8/c16/? / bool, plus S/U/A for \
             ASCII string VLA)")),
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
pub(crate) fn dtype_to_write_columns(
    dtype: &Bound<'_, PyAny>,
    units: Option<&Bound<'_, PyDict>>,
    var_dtypes: Option<&Bound<'_, PyDict>>,
    bit_columns: Option<&BitColumnsSpec>,
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

        // `bit_columns=True` (All) is a soft global toggle: promote
        // only the b1 columns to X and silently leave other types
        // alone (matches fitsio's `write_bitcols=True` semantics).
        // `bit_columns=["name", ...]` is a hard per-name opt-in:
        // every listed column gets `bit_packed=true`, and the
        // classifier rejects below if any of those columns isn't b1.
        let bit_packed_opt_in = match bit_columns {
            None => false,
            Some(BitColumnsSpec::All) => {
                let kind: String =
                    base_dtype.getattr("kind")?.extract()?;
                let itemsize: usize =
                    base_dtype.getattr("itemsize")?.extract()?;
                kind == "b" && itemsize == 1
            }
            Some(BitColumnsSpec::Names(set)) => {
                set.contains(&name.to_uppercase())
            }
        };
        let cls = classify_scalar_numpy_field(
            &base_dtype, name, bit_packed_opt_in)?;
        let array_count: usize =
            np_shape.iter().copied().product::<usize>().max(1);
        // Total byte_width (and TFORM repeat) depends on the on-disk
        // shape.  Three cases:
        //   - X (bit-packed): TFORM repeat = bit count = array_count;
        //     byte_width = ceil(repeat/8).  Trailing bits in the last
        //     byte are zero per the FITS spec.
        //   - A (strings): TFORM repeat = total bytes per cell
        //     (chars × strings).
        //   - Other letters: TFORM repeat = element count;
        //     byte_width = repeat * elem_bytes.
        let (repeat, byte_width) = if cls.bit_packed {
            (array_count, array_count.div_ceil(8))
        } else {
            match cls.chars_per_string {
                Some(chars) => {
                    let total_bytes = chars * array_count;
                    (total_bytes, total_bytes)
                }
                None => (array_count, cls.elem_bytes * array_count),
            }
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
    // Same diff for bit_columns: an entry that names no column is a
    // user error (typo or stale name).  Skip the check when the
    // user passed `bit_columns=True`/`All` — that's intentionally
    // universal and matches any future column.
    if let Some(BitColumnsSpec::Names(provided)) = bit_columns {
        let column_names_upper: std::collections::HashSet<String> =
            out.iter().map(|c| c.name.to_uppercase()).collect();
        for k in provided {
            if !column_names_upper.contains(k) {
                return Err(PyValueError::new_err(format!(
                    "bit_columns contains entry '{}' that does not \
                     match any column in the dtype", k)));
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
    bit_columns: Option<&BitColumnsSpec>,
    descriptor: char,
) -> PyResult<(Vec<String>, u64)> {
    let np = py.import("numpy")?;
    let np_dtype = np.getattr("dtype")?.call1((dtype_in,))?;
    let write_columns = dtype_to_write_columns(
        &np_dtype, units, var_dtypes, bit_columns, descriptor)?;
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
pub(crate) enum WriteTransform {
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
    // numpy bool (1 byte 0/1 per element) → FITS X (bit-packed,
    // MSB-first per byte).  src_size = `num_bits` bytes; dst_size =
    // ceil(num_bits/8) bytes.  Trailing bits in the last byte (when
    // num_bits % 8 != 0) are zeroed.  Only valid on the slow path —
    // the width difference rules out the bulk-memcpy fast path.
    BitsPackMsb { num_bits: usize },
}

// Resolve the per-column write transform given the on-disk column
// (tform_letter + tzero) and the input numpy field's (kind, base
// itemsize).  Rejects mismatches with a message naming both sides.
pub(crate) fn column_transform(
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

    // BitsPackMsb: input is numpy bool, column letter is X.  src_size
    // (num_bits bytes — one per bool) differs from dst_size
    // (ceil(num_bits/8) bytes), which is why this transform is
    // slow-path only.  col.repeat carries the bit count for X
    // columns; col.byte_width = ceil(repeat/8).
    if col.tform_letter == 'X' {
        if input_kind == "b" && input_size == 1 {
            return Ok(WriteTransform::BitsPackMsb {
                num_bits: col.repeat,
            });
        }
        return Err(PyValueError::new_err(format!(
            "column '{}' (X, {} bits): expected numpy bool input, got \
             kind '{}' itemsize {}",
            col.name, col.repeat, input_kind, input_size)));
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
pub(crate) fn column_expected_shape(col: &Column) -> Vec<usize> {
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
