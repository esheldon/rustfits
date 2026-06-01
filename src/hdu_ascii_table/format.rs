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
}
