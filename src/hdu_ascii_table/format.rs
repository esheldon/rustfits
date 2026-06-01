// Per-cell text<->value conversion for ASCII-table fields.  This file
// owns both directions because the format spec is the seam — every
// parser has a matching formatter and they evolve together.
//
// Phase 1 only ships the parse side (text -> value); Phase 3 (create
// + write) will add the formatter side here as well.
//
// Operations work on `&[u8]` slices (the on-disk bytes of one field
// in one row) so the hot path doesn't allocate Strings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Trim ASCII whitespace (spaces, tabs) from both ends of a byte slice.
// FITS fields are ASCII so this is sufficient.
pub(crate) fn trim_ascii(src: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < src.len() && (src[start] == b' ' || src[start] == b'\t') {
        start += 1;
    }
    let mut end = src.len();
    while end > start && (src[end - 1] == b' ' || src[end - 1] == b'\t') {
        end -= 1;
    }
    &src[start..end]
}

// Decide whether the trimmed field text matches a trimmed TNULL string.
// The compare is byte-for-byte on the trimmed slice; trailing/leading
// spaces are not significant.  Per FITS spec, an all-blank field is
// also considered undefined — when TNULL is set we treat empty as
// null; when TNULL is absent the column's read path handles empty
// separately (parse-or-zero).
//
// Used by Phase 2's mask_null=True path; not exercised in Phase 1.
#[allow(dead_code)]
pub(crate) fn matches_tnull(trimmed_field: &[u8], tnull_trimmed: &str) -> bool {
    trimmed_field == tnull_trimmed.as_bytes()
}

// Parse an ASCII-table 'I' field (integer) into i64.  Empty or
// all-whitespace field returns Ok(0) per FITS spec for undefined ints
// (the caller may override with TNULL masking).  Other parse failures
// raise — the file is internally inconsistent and silently zero-ing
// would mask data corruption.
pub(crate) fn parse_int_field(
    src: &[u8], col_name: &str, row_index: usize,
) -> PyResult<i64> {
    let t = trim_ascii(src);
    if t.is_empty() {
        return Ok(0);
    }
    // i64::from_str expects str.  ASCII bytes are always valid UTF-8.
    let s = std::str::from_utf8(t).map_err(|_| PyValueError::new_err(format!(
        "column '{}' row {}: integer field contains non-ASCII bytes",
        col_name, row_index,
    )))?;
    s.parse::<i64>().map_err(|_| PyValueError::new_err(format!(
        "column '{}' row {}: failed to parse integer from '{}'",
        col_name, row_index, s,
    )))
}

// Parse an ASCII-table F/E/D field into f64.  Accepts FORTRAN-style 'D'
// exponent (e.g. "1.5D+03") by converting D/d -> E/e before parsing.
// Empty / all-whitespace returns 0.0 (same parse-or-zero contract as
// integer fields).
pub(crate) fn parse_float_field(
    src: &[u8], col_name: &str, row_index: usize,
) -> PyResult<f64> {
    let t = trim_ascii(src);
    if t.is_empty() {
        return Ok(0.0);
    }
    let s = std::str::from_utf8(t).map_err(|_| PyValueError::new_err(format!(
        "column '{}' row {}: float field contains non-ASCII bytes",
        col_name, row_index,
    )))?;
    // FORTRAN 'D' exponent marker — Rust's f64::from_str only accepts
    // 'E'/'e'.  Replace in place via an owned String only when needed.
    let parsed = if s.bytes().any(|b| b == b'D' || b == b'd') {
        let normalized: String = s.chars().map(|c| match c {
            'D' => 'E', 'd' => 'e', other => other,
        }).collect();
        normalized.parse::<f64>()
    } else {
        s.parse::<f64>()
    };
    parsed.map_err(|_| PyValueError::new_err(format!(
        "column '{}' row {}: failed to parse float from '{}'",
        col_name, row_index, s,
    )))
}

// ---------------------------------------------------------------------------
// Value -> text formatters (write side, Phase 3)
// ---------------------------------------------------------------------------
//
// Each formatter writes `width` bytes into `dst` in the FITS-spec
// layout for its TFORM letter.  Width-overflow raises a clear error
// naming the column and value (no silent truncation).

// Format an integer value into an `Iw` field.  Right-justified,
// padded with leading spaces.  Raises if the formatted digits don't
// fit.
pub(crate) fn format_int_field(
    value: i64, dst: &mut [u8], col_name: &str, row_index: usize,
) -> PyResult<()> {
    let width = dst.len();
    let s = format!("{}", value);
    if s.len() > width {
        return Err(PyValueError::new_err(format!(
            "column '{}' row {}: integer value {} does not fit in width \
             {} (TFORM I{}); widen the format or check the data",
            col_name, row_index, value, width, width,
        )));
    }
    let pad = width - s.len();
    for b in dst.iter_mut().take(pad) {
        *b = b' ';
    }
    dst[pad..].copy_from_slice(s.as_bytes());
    Ok(())
}

// Format a floating-point value into an `Fw.d` field (fixed-point).
pub(crate) fn format_f_field(
    value: f64, decimals: usize, dst: &mut [u8],
    col_name: &str, row_index: usize,
) -> PyResult<()> {
    let width = dst.len();
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "column '{}' row {}: F format cannot represent non-finite \
             value {} (NaN/Inf); use E/D format instead",
            col_name, row_index, value,
        )));
    }
    let s = format!("{:.*}", decimals, value);
    if s.len() > width {
        return Err(PyValueError::new_err(format!(
            "column '{}' row {}: F-formatted value {:?} does not fit in \
             width {} (TFORM F{}.{}); widen the format or check the data",
            col_name, row_index, s, width, width, decimals,
        )));
    }
    let pad = width - s.len();
    for b in dst.iter_mut().take(pad) {
        *b = b' ';
    }
    dst[pad..].copy_from_slice(s.as_bytes());
    Ok(())
}

// Format a floating-point value into an `Ew.d` field (exponential).
// Output shape: "[ +-]m.fff...E[+-]ee" — matches cfitsio's output
// (explicit exponent sign, ≥2-digit exponent).
pub(crate) fn format_e_field(
    value: f64, decimals: usize, dst: &mut [u8],
    col_name: &str, row_index: usize,
) -> PyResult<()> {
    let width = dst.len();
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "column '{}' row {}: E format cannot represent non-finite \
             value {} (NaN/Inf); the FITS spec leaves this undefined",
            col_name, row_index, value,
        )));
    }
    let s = format_scientific(value, decimals, 'E');
    if s.len() > width {
        return Err(PyValueError::new_err(format!(
            "column '{}' row {}: E-formatted value {:?} does not fit in \
             width {} (TFORM E{}.{}); widen the format or check the data",
            col_name, row_index, s, width, width, decimals,
        )));
    }
    let pad = width - s.len();
    for b in dst.iter_mut().take(pad) {
        *b = b' ';
    }
    dst[pad..].copy_from_slice(s.as_bytes());
    Ok(())
}

// `Dw.d` is the double-precision counterpart of E; cfitsio emits the
// same exponent format with letter 'E' for both, and the read side
// accepts 'D' or 'E' interchangeably per FITS spec.  Emit 'E' to
// match cfitsio.
pub(crate) fn format_d_field(
    value: f64, decimals: usize, dst: &mut [u8],
    col_name: &str, row_index: usize,
) -> PyResult<()> {
    format_e_field(value, decimals, dst, col_name, row_index)
}

// Manual scientific formatter producing "M.FFFE+XX".  Rust's `{:E}`
// emits "1.5E3" (no '+' sign, no leading zero on exponent); cfitsio
// emits "1.5000E+03" — match that for portability.  Round-up of the
// mantissa to 10 is detected and re-normalized (e.g. 9.9995 with
// d=3 becomes "1.000E+01" not "10.000E+00").
fn format_scientific(value: f64, decimals: usize, exp_letter: char) -> String {
    if value == 0.0 {
        let mantissa = if decimals == 0 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat(decimals))
        };
        return format!("{}{}+00", mantissa, exp_letter);
    }
    let sign = if value < 0.0 { "-" } else { "" };
    let abs = value.abs();
    let exp = abs.log10().floor() as i32;
    let mantissa = abs / 10f64.powi(exp);
    let formatted = format!("{:.*}", decimals, mantissa);
    let (mantissa_str, final_exp) = if formatted.starts_with("10") {
        let renorm = abs / 10f64.powi(exp + 1);
        (format!("{:.*}", decimals, renorm), exp + 1)
    } else {
        (formatted, exp)
    };
    let exp_sign = if final_exp < 0 { '-' } else { '+' };
    let exp_abs = final_exp.unsigned_abs();
    format!("{}{}{}{}{:02}", sign, mantissa_str, exp_letter, exp_sign, exp_abs)
}

// Format a fixed-width A field from raw bytes.  Copies source bytes
// left-justified; pads right with spaces if the source is shorter
// than the field.  Source bytes that exceed `width` are silently
// truncated (matches cfitsio's column overflow behavior on strings).
// Non-ASCII source bytes raise with row context.
pub(crate) fn format_a_field(
    src: &[u8], dst: &mut [u8], col_name: &str, row_index: usize,
) -> PyResult<()> {
    let width = dst.len();
    let copied = src.len().min(width);
    for (i, &b) in src.iter().take(copied).enumerate() {
        if !b.is_ascii() {
            return Err(PyValueError::new_err(format!(
                "column '{}' row {}: A field contains non-ASCII byte \
                 0x{:02X} at position {}",
                col_name, row_index, b, i,
            )));
        }
        dst[i] = b;
    }
    for b in dst.iter_mut().skip(copied) {
        *b = b' ';
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_ascii_simple() {
        assert_eq!(trim_ascii(b"  hello  "), b"hello");
        assert_eq!(trim_ascii(b"hello"), b"hello");
        assert_eq!(trim_ascii(b"   "), b"");
        assert_eq!(trim_ascii(b""), b"");
        assert_eq!(trim_ascii(b"\t hello \t"), b"hello");
    }

    #[test]
    fn matches_tnull_basic() {
        assert!(matches_tnull(b"NULL", "NULL"));
        assert!(!matches_tnull(b"null", "NULL"));
        assert!(matches_tnull(b"", ""));
    }

    #[test]
    fn format_int_right_justified() {
        let mut dst = [0u8; 5];
        format_int_field(42, &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b"   42");
        format_int_field(-17, &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b"  -17");
        format_int_field(0, &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b"    0");
    }

    #[test]
    fn format_int_overflow_raises() {
        let mut dst = [0u8; 2];
        assert!(format_int_field(100, &mut dst, "x", 0).is_err());
    }

    #[test]
    fn format_e_field_matches_cfitsio_shape() {
        let mut dst = [0u8; 12];
        format_e_field(1500.0, 4, &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b"  1.5000E+03");
        format_e_field(-0.025, 4, &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b" -2.5000E-02");
        format_e_field(0.0, 4, &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b"  0.0000E+00");
    }

    #[test]
    fn format_e_field_renormalizes_round_to_ten() {
        // 9.9999 with 3 decimals would format as "10.000E+00" but
        // must re-normalize to "1.000E+01".  (9.9995 picks 9.999
        // via banker's rounding — not a renorm case.)
        let mut dst = [0u8; 10];
        format_e_field(9.9999, 3, &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b" 1.000E+01");
    }

    #[test]
    fn format_f_basic() {
        let mut dst = [0u8; 8];
        format_f_field(3.14, 3, &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b"   3.140");
        format_f_field(-2.5, 1, &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b"    -2.5");
    }

    #[test]
    fn format_a_pads_and_truncates() {
        let mut dst = [0u8; 5];
        format_a_field(b"hi", &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b"hi   ");
        format_a_field(b"toolong", &mut dst, "x", 0).unwrap();
        assert_eq!(&dst, b"toolo"); // truncated
    }
}
