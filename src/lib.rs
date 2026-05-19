use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyDict, PyEllipsis, PyList, PySlice, PyTuple};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::Bound;
use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex, MutexGuard};
use std::sync::atomic::{AtomicBool, Ordering};

// Shared, mutable file handle.  FITS owns the master Arc, each HDU clones it.
// `None` after close().
type FileHandle = Arc<Mutex<Option<std::fs::File>>>;

fn lock_file(handle: &FileHandle) -> PyResult<MutexGuard<'_, Option<std::fs::File>>> {
    handle.lock().map_err(|_| PyIOError::new_err("file lock poisoned"))
}

// Per-FITS-file taint flag.  Set when a mid-write disk error inside
// `rewrite_header_to_disk` may have left the on-disk header partially
// overwritten.  Subsequent header / image reads and writes refuse with a
// clear error until the file is closed and reopened.  One flag is shared
// across all HDUs of the same file (and across all FITSHeader views).
type TaintFlag = Arc<AtomicBool>;

fn check_not_tainted(tainted: &TaintFlag) -> PyResult<()> {
    if tainted.load(Ordering::Acquire) {
        return Err(PyIOError::new_err(
            "this FITS file is in an indeterminate state after a mid-write \
             I/O failure; the on-disk header may be partially overwritten — \
             close the FITS object and reopen the file to recover"
        ));
    }
    Ok(())
}

const BLOCK_SIZE: usize = 2880;
const CARD_SIZE: usize = 80;

// ====================== Helper: parse integer keyword (for data size) ======================
// Strict match: the keyword field in cols 1-8 (trimmed) must equal `key`, and
// col 9 must be `=`.  This avoids the trap that `starts_with("NAXIS")` would
// also match `NAXIS1`, `NAXIS2`, etc.
fn parse_keyword(cards: &[String], key: &str) -> Option<i64> {
    for card in cards {
        if card.len() < 9 { continue; }
        if card[..8].trim() != key { continue; }
        if !card[8..].starts_with('=') { continue; }
        let value_part = &card[9..];
        if let Some(num_str) = value_part.split_whitespace().next() {
            let cleaned = num_str.trim_end_matches(&['\'', ' ', '/'][..]);
            return cleaned.parse::<i64>().ok();
        }
    }
    None
}

// ====================== Helper: validate that a header has mandatory keywords ======================
// Per the FITS standard, the primary HDU must begin with `SIMPLE`, extension
// HDUs must begin with `XTENSION`, and every HDU must declare BITPIX, NAXIS,
// and NAXIS1..NAXISn.  `END` is enforced implicitly by the reader (it keeps
// pulling 2880-byte blocks until END is seen or EOF is hit).
fn validate_header(cards: &[String], is_primary: bool) -> PyResult<()> {
    let first = match cards.iter().find(|c| !c.trim().is_empty()) {
        Some(c) => c,
        None => return Err(PyValueError::new_err("empty header")),
    };
    let first_key = if first.len() >= 8 { first[..8].trim() } else { first.trim() };

    if is_primary {
        if first_key != "SIMPLE" {
            return Err(PyValueError::new_err(format!(
                "primary header must start with SIMPLE, found '{}'", first_key
            )));
        }
    } else if first_key != "XTENSION" {
        return Err(PyValueError::new_err(format!(
            "extension header must start with XTENSION, found '{}'", first_key
        )));
    }

    if parse_keyword(cards, "BITPIX").is_none() {
        return Err(PyValueError::new_err("missing required BITPIX keyword"));
    }
    let naxis = match parse_keyword(cards, "NAXIS") {
        Some(n) => n,
        None => return Err(PyValueError::new_err("missing required NAXIS keyword")),
    };
    if !(0..=999).contains(&naxis) {
        return Err(PyValueError::new_err(format!(
            "NAXIS={} out of range (must be 0..999)", naxis
        )));
    }
    for n in 1..=naxis {
        let key = format!("NAXIS{}", n);
        if parse_keyword(cards, &key).is_none() {
            return Err(PyValueError::new_err(format!(
                "missing required {} keyword", key
            )));
        }
    }
    Ok(())
}

// ====================== Calculate exact padded data section size ======================
// Implements the general FITS formula:
//
//     N_bytes = |BITPIX|/8 * GCOUNT * (PCOUNT + Π NAXISn)
//
// For images: GCOUNT defaults to 1, PCOUNT defaults to 0, reducing to
// bytes_per_pixel * Π NAXISn.  For binary tables, PCOUNT carries the size of
// the variable-length-array heap, which must be included so that the next
// HDU is located correctly.  An HDU with NAXIS=0 has no data unit regardless
// of PCOUNT/GCOUNT.
fn calculate_data_size(header_cards: &[String]) -> u64 {
    let bitpix = parse_keyword(header_cards, "BITPIX").unwrap_or(0);
    let naxis = parse_keyword(header_cards, "NAXIS").unwrap_or(0) as usize;

    if bitpix == 0 || naxis == 0 {
        return 0;
    }

    let bytes_per_pixel = (bitpix.abs() / 8) as u64;

    let pcount_raw = parse_keyword(header_cards, "PCOUNT").unwrap_or(0);
    let pcount: u64 = if pcount_raw > 0 { pcount_raw as u64 } else { 0 };

    let gcount_raw = parse_keyword(header_cards, "GCOUNT").unwrap_or(1);
    let gcount: u64 = if gcount_raw > 0 { gcount_raw as u64 } else { 1 };

    let mut product: u64 = 1;
    for i in 1..=naxis {
        if let Some(dim) = parse_keyword(header_cards, &format!("NAXIS{}", i)) {
            product = product.saturating_mul(dim as u64);
        }
    }

    let raw_size = bytes_per_pixel
        .saturating_mul(gcount)
        .saturating_mul(product.saturating_add(pcount));

    if raw_size == 0 {
        0
    } else {
        ((raw_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64) * BLOCK_SIZE as u64
    }
}

// ====================== Helper: split a card's value/comment portion ======================
// `value_part` is the substring after `=` for a regular card, or the substring after
// the `CONTINUE` keyword (cols 9..) for a continuation card.  Returns (trimmed raw
// value with quotes intact, trimmed comment).  A `/` that appears inside a quoted
// FITS string is not treated as a comment delimiter.  The single-quote toggle
// happens to also handle `''` escape correctly, because each escape contributes
// two toggles (net no change).
fn split_value_comment(value_part: &str) -> (String, String) {
    let mut comment_start = value_part.len();
    let mut in_string = false;
    for (i, ch) in value_part.char_indices() {
        if ch == '\'' {
            in_string = !in_string;
        } else if ch == '/' && !in_string {
            comment_start = i;
            break;
        }
    }
    let raw = value_part[..comment_start].trim().to_string();
    let comment = if comment_start < value_part.len() {
        value_part[comment_start + 1..].trim().to_string()
    } else {
        String::new()
    };
    (raw, comment)
}

// ====================== Helper: extract a quoted FITS string ======================
// `raw` is the trimmed value substring including outer single quotes (e.g.
// "'O''Brien   '").  Strips the outer quotes, converts `''` back to `'`, and
// drops trailing spaces (per the FITS standard, trailing spaces in a string
// value are not significant).  FITS values are restricted to printable ASCII,
// so byte-position slicing is safe.
fn extract_fits_string(raw: &str) -> String {
    let inner = &raw[1..raw.len() - 1];
    inner.replace("''", "'").trim_end().to_string()
}

// ====================== Helper: parse a FITS float ======================
// FITS permits Fortran-style `D` as the exponent indicator in addition to `E`.
fn parse_fits_float(s: &str) -> Option<f64> {
    if s.contains('D') || s.contains('d') {
        s.replace('D', "E").replace('d', "e").parse::<f64>().ok()
    } else {
        s.parse::<f64>().ok()
    }
}

// ====================== Helper: parse a FITS complex literal ======================
// Returns Some((real, imag)) if `raw` looks like "(real, imag)".  Whitespace
// is allowed around the comma and inside the parentheses.  Each component may
// use either E or D exponent notation.
fn parse_fits_complex(raw: &str) -> Option<(f64, f64)> {
    if !(raw.starts_with('(') && raw.ends_with(')') && raw.len() >= 2) {
        return None;
    }
    let inner = &raw[1..raw.len() - 1];
    let comma = inner.find(',')?;
    let r = parse_fits_float(inner[..comma].trim())?;
    let i = parse_fits_float(inner[comma + 1..].trim())?;
    Some((r, i))
}

// ====================== Parse FITS header cards into Python dict ======================
fn parse_header_dict(py: Python<'_>, cards: &[String]) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let comments_list = PyList::empty(py);
    let history_list = PyList::empty(py);
    let blank_list = PyList::empty(py);
    let mut has_comments = false;
    let mut has_history = false;
    let mut has_blank = false;

    let mut i = 0;
    while i < cards.len() {
        let card = cards[i].trim_end();
        if card.is_empty() {
            i += 1;
            continue;
        }

        // Inspect the keyword field (cols 1-8).
        let key_field: &str = if card.len() >= 8 { &card[..8] } else { card };
        let keyword_trimmed = key_field.trim();

        // Commentary / structural cards: dispatched by the keyword field.
        match keyword_trimmed {
            "END" => { i += 1; continue; }
            "COMMENT" => {
                let text = if card.len() > 8 { card[8..].to_string() } else { String::new() };
                comments_list.append(text)?;
                has_comments = true;
                i += 1;
                continue;
            }
            "HISTORY" => {
                let text = if card.len() > 8 { card[8..].to_string() } else { String::new() };
                history_list.append(text)?;
                has_history = true;
                i += 1;
                continue;
            }
            "" => {
                // Blank keyword field, content in cols 9-80: commentary card
                // with no associated keyword.
                let text = if card.len() > 8 { card[8..].to_string() } else { String::new() };
                blank_list.append(text)?;
                has_blank = true;
                i += 1;
                continue;
            }
            "CONTINUE" => { i += 1; continue; }  // orphan CONTINUE
            _ => {}
        }

        // Determine keyword and value substring.  For HIERARCH, the keyword is
        // everything between `HIERARCH ` and the first `=`.  For regular cards,
        // the keyword field is cols 1-8.
        let (keyword, value_part) = if keyword_trimmed == "HIERARCH" {
            let eq_pos = match card.find('=') {
                Some(p) => p,
                None => { i += 1; continue; }
            };
            let kw = card[8..eq_pos].trim().to_string();
            if kw.is_empty() { i += 1; continue; }
            (kw, &card[eq_pos + 1..])
        } else {
            let eq_pos = match card.find('=') {
                Some(p) => p,
                None => { i += 1; continue; }
            };
            (keyword_trimmed.to_string(), &card[eq_pos + 1..])
        };

        let (raw_value, mut comment) = split_value_comment(value_part);

        let py_value: Py<PyAny> = if raw_value.is_empty() {
            py.None()
        } else if raw_value.starts_with('\'') && raw_value.ends_with('\'') && raw_value.len() >= 2 {
            // String value, possibly continued via the long-string convention.
            let mut s = extract_fits_string(&raw_value);
            while s.ends_with('&') {
                if i + 1 >= cards.len() { break; }
                let next_card = cards[i + 1].trim_end();
                if !next_card.starts_with("CONTINUE") { break; }
                let rest = if next_card.len() > 8 { &next_card[8..] } else { "" };
                let (cont_raw, cont_comment) = split_value_comment(rest);
                if !(cont_raw.starts_with('\'')
                    && cont_raw.ends_with('\'')
                    && cont_raw.len() >= 2)
                {
                    break;
                }
                let segment = extract_fits_string(&cont_raw);
                s.pop(); // drop trailing `&`
                s.push_str(&segment);
                if !cont_comment.is_empty() {
                    if !comment.is_empty() { comment.push(' '); }
                    comment.push_str(&cont_comment);
                }
                i += 1;
            }
            s.into_py_any(py)?
        } else if raw_value == "T" {
            true.into_py_any(py)?
        } else if raw_value == "F" {
            false.into_py_any(py)?
        } else if let Some((r, im)) = parse_fits_complex(&raw_value) {
            PyComplex::from_doubles(py, r, im).into_any().unbind()
        } else if let Ok(n) = raw_value.parse::<i64>() {
            n.into_py_any(py)?
        } else if let Some(f) = parse_fits_float(&raw_value) {
            f.into_py_any(py)?
        } else {
            raw_value.into_py_any(py)?
        };

        let inner = PyDict::new(py);
        inner.set_item("value", py_value)?;
        inner.set_item("comment", comment)?;
        dict.set_item(keyword, inner)?;

        i += 1;
    }

    if has_comments {
        dict.set_item("COMMENT", comments_list)?;
    }
    if has_history {
        dict.set_item("HISTORY", history_list)?;
    }
    if has_blank {
        dict.set_item("", blank_list)?;
    }

    Ok(dict.unbind())
}

// ====================== FITSHeader: card-list view of a header ======================
//
// Wraps the raw 80-char card list (`Vec<String>`) and exposes it to Python as
// an ordered, dict-like view.  Cards are the single source of truth — every
// value/comment/iteration access re-parses the relevant card(s).  Headers are
// tiny (tens to a few hundred cards) so parse-on-demand cost is invisible,
// and keeping no cache means future mutation methods only have to rewrite
// cards without worrying about cache invalidation.
//
// Iteration order is the order keys first appear in the file.  Commentary
// keys (COMMENT, HISTORY, blank-keyword cards) collapse to a single appearance
// even if their cards repeat; `header[key]` for those returns a list of all
// matching card texts in card order.

// Extract the keyword name for a card.  Returns:
//   - Some("X")   for a normal/HIERARCH keyword card
//   - Some("END" / "COMMENT" / "HISTORY" / "CONTINUE")
//   - Some("")    for a blank-keyword commentary card (cols 1-8 all spaces)
//   - None        for an empty card after trimming
fn keyword_of(card: &str) -> Option<String> {
    let card = card.trim_end();
    if card.is_empty() {
        return None;
    }
    let key_field = if card.len() >= 8 { &card[..8] } else { card };
    let trimmed = key_field.trim();
    if trimmed == "HIERARCH" {
        if let Some(eq_pos) = card.find('=') {
            let kw = card[8..eq_pos].trim();
            if !kw.is_empty() {
                return Some(kw.to_string());
            }
        }
        return Some("HIERARCH".to_string());
    }
    Some(trimmed.to_string())
}

// Walk cards once, return unique keys in order of first appearance.  END and
// orphan CONTINUE cards are skipped; COMMENT/HISTORY/blank appear once each.
fn unique_keys_in_order(cards: &[String]) -> Vec<String> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for card in cards {
        let kw = match keyword_of(card) {
            Some(k) => k,
            None => continue,
        };
        if kw == "END" || kw == "CONTINUE" {
            continue;
        }
        if seen.insert(kw.clone()) {
            out.push(kw);
        }
    }
    out
}

// Find the first card with a matching key, returning (card_index, value_part).
// Skips commentary keys (they require list-of-cards collection by the caller).
// Handles HIERARCH long keys.
fn find_card_for_key(cards: &[String], key: &str) -> Option<(usize, String)> {
    for (i, card) in cards.iter().enumerate() {
        let card = card.trim_end();
        if card.is_empty() {
            continue;
        }
        let key_field = if card.len() >= 8 { &card[..8] } else { card };
        let kw_trimmed = key_field.trim();
        if matches!(kw_trimmed, "END" | "COMMENT" | "HISTORY" | "" | "CONTINUE") {
            continue;
        }
        if kw_trimmed == "HIERARCH" {
            if let Some(eq_pos) = card.find('=') {
                if card[8..eq_pos].trim() == key {
                    return Some((i, card[eq_pos + 1..].to_string()));
                }
            }
        } else if kw_trimmed == key {
            if let Some(eq_pos) = card.find('=') {
                return Some((i, card[eq_pos + 1..].to_string()));
            }
        }
    }
    None
}

// Collect all card texts (cols 9-80) for cards whose keyword field equals
// `keyword` (e.g. "COMMENT", "HISTORY").
fn collect_commentary_texts(cards: &[String], keyword: &str) -> Vec<String> {
    let mut out = Vec::new();
    for card in cards {
        let card = card.trim_end();
        if card.is_empty() {
            continue;
        }
        let key_field = if card.len() >= 8 { &card[..8] } else { card };
        if key_field.trim() == keyword {
            let text = if card.len() > 8 {
                card[8..].to_string()
            } else {
                String::new()
            };
            out.push(text);
        }
    }
    out
}

// Collect text from blank-keyword commentary cards (cols 1-8 all spaces).
fn collect_blank_commentary_texts(cards: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for card in cards {
        let card = card.trim_end();
        if card.is_empty() {
            continue;
        }
        if card.len() >= 8 && card[..8].trim().is_empty() && card.len() > 8 {
            out.push(card[8..].to_string());
        }
    }
    out
}

// Parse a card's value_part into (Python value, comment), following the
// FITS CONTINUE long-string convention when the value is a string ending
// in `&`.  Comments on continuation cards are space-joined to the base
// card's comment (matching `parse_header_dict`).
fn parse_value_with_continue(
    py: Python<'_>,
    cards: &[String],
    start_idx: usize,
    value_part: &str,
) -> PyResult<(Py<PyAny>, String)> {
    let (raw_value, mut comment) = split_value_comment(value_part);
    if raw_value.is_empty() {
        return Ok((py.None(), comment));
    }
    if raw_value.starts_with('\'') && raw_value.ends_with('\'') && raw_value.len() >= 2 {
        let mut s = extract_fits_string(&raw_value);
        let mut i = start_idx;
        while s.ends_with('&') {
            if i + 1 >= cards.len() {
                break;
            }
            let next_card = cards[i + 1].trim_end();
            if !next_card.starts_with("CONTINUE") {
                break;
            }
            let rest = if next_card.len() > 8 { &next_card[8..] } else { "" };
            let (cont_raw, cont_comment) = split_value_comment(rest);
            if !(cont_raw.starts_with('\'')
                && cont_raw.ends_with('\'')
                && cont_raw.len() >= 2)
            {
                break;
            }
            let segment = extract_fits_string(&cont_raw);
            s.pop();
            s.push_str(&segment);
            if !cont_comment.is_empty() {
                if !comment.is_empty() {
                    comment.push(' ');
                }
                comment.push_str(&cont_comment);
            }
            i += 1;
        }
        return Ok((s.into_py_any(py)?, comment));
    }
    let value: Py<PyAny> = if raw_value == "T" {
        true.into_py_any(py)?
    } else if raw_value == "F" {
        false.into_py_any(py)?
    } else if let Some((r, im)) = parse_fits_complex(&raw_value) {
        PyComplex::from_doubles(py, r, im).into_any().unbind()
    } else if let Ok(n) = raw_value.parse::<i64>() {
        n.into_py_any(py)?
    } else if let Some(f) = parse_fits_float(&raw_value) {
        f.into_py_any(py)?
    } else {
        raw_value.into_py_any(py)?
    };
    Ok((value, comment))
}

// ====================== Mutation helpers (phase 2a) ======================

// Normalize a user-supplied keyword to the canonical FITS form (ASCII upper).
// Keywords in conforming FITS files are uppercase; we accept any case from
// callers and upper-case before lookup or write, matching cfitsio/astropy/
// fitsio behavior.  Applied uniformly across __getitem__/__setitem__/etc.
fn normalize_keyword(key: &str) -> String {
    key.trim().to_ascii_uppercase()
}

// Does this (post-normalization) key require the HIERARCH convention?
// Keys longer than 8 chars or containing spaces must use HIERARCH.
fn is_hierarch_key(key: &str) -> bool {
    key.len() > 8 || key.contains(' ')
}

// Validate a (post-normalization) keyword for writing.  Standard keywords:
// 1-8 ASCII chars from A-Z, 0-9, '-', '_'.  HIERARCH long keys: any length
// (subject to fitting in an 80-char card), additionally allowing space,
// '.', '+'.  The literal "HIERARCH" is rejected — it's the convention
// prefix, not a user key.
fn validate_keyword(key: &str) -> PyResult<()> {
    if key.is_empty() {
        return Err(PyValueError::new_err("keyword cannot be empty"));
    }
    if key == "HIERARCH" {
        return Err(PyValueError::new_err(
            "'HIERARCH' is not a valid user keyword; HIERARCH is the convention \
             prefix for long keys — pass a longer key (>8 chars or containing spaces) instead"
        ));
    }
    let hierarch = is_hierarch_key(key);
    for c in key.chars() {
        let ok = c.is_ascii_uppercase()
            || c.is_ascii_digit()
            || c == '-'
            || c == '_'
            || (hierarch && (c == ' ' || c == '.' || c == '+'));
        if !ok {
            return Err(PyValueError::new_err(format!(
                "keyword '{}' contains invalid character '{}' \
                 (standard keys allow A-Z/a-z, 0-9, '-', '_'; \
                 HIERARCH long keys additionally allow ' ', '.', '+')",
                key, c
            )));
        }
    }
    Ok(())
}

// Pull the comment off the first existing card for `key`, if any.  Walks
// the CONTINUE chain (if the key's value is a `&`-terminated string) so a
// long-string value's chained comment is preserved on update.  Returns
// None if the key is not present.
fn extract_existing_comment(py: Python<'_>, cards: &[String], key: &str) -> Option<String> {
    let (idx, value_part) = find_card_for_key(cards, key)?;
    parse_value_with_continue(py, cards, idx, &value_part)
        .ok()
        .map(|(_, c)| c)
}

// Build the card(s) representing (key, value, comment).  Most types emit a
// single card; string values longer than what fits in one card emit a
// CONTINUE-chained sequence.  HIERARCH long keys use a different card
// layout and (in phase 2c) always emit exactly one card — CONTINUE-chained
// HIERARCH long-string values are deferred.  Bool is checked before int
// because Python bools are also ints (`extract::<i64>()` succeeds on `True`).
fn build_card_from_value(
    key: &str,
    value: &Bound<'_, PyAny>,
    comment: &str,
) -> PyResult<Vec<String>> {
    if is_hierarch_key(key) {
        return Ok(vec![build_hierarch_card(key, value, comment)?]);
    }
    if value.is_instance_of::<PyBool>() {
        let b: bool = value.extract()?;
        return Ok(vec![card_logical(key, b, comment)]);
    }
    if let Ok(n) = value.extract::<i64>() {
        return Ok(vec![card_int(key, n, comment)]);
    }
    // PyComplex check before f64: Python complex doesn't extract as f64
    // (no implicit complex->float coercion), but being explicit avoids
    // surprising failure modes if pyo3 ever adds such a coercion.
    if value.is_instance_of::<PyComplex>() {
        let c = value.cast::<PyComplex>()?;
        return Ok(vec![card_complex(key, c.real(), c.imag(), comment)?]);
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(vec![card_float(key, f, comment)]);
    }
    if let Ok(s) = value.extract::<String>() {
        return build_string_value_cards(key, &s, comment);
    }
    Err(PyValueError::new_err(format!(
        "unsupported value type for key '{}'; \
         supported types: bool, int, float, complex, str", key
    )))
}

// Build a HIERARCH-format card for a long key.  Layout:
//   HIERARCH <key> = <value>[ / <comment>]
// String values: quoted, single quotes escaped (`'` -> `''`), padded to a
// minimum of 8 chars inside the quotes (FITS convention).  Errors if the
// resulting card exceeds 80 chars.
fn build_hierarch_card(
    key: &str,
    value: &Bound<'_, PyAny>,
    comment: &str,
) -> PyResult<String> {
    // Format the value side per type.  Bool first (Python bools are ints);
    // PyComplex before f64 (Python complex doesn't coerce to float).
    let value_str = if value.is_instance_of::<PyBool>() {
        let b: bool = value.extract()?;
        (if b { "T" } else { "F" }).to_string()
    } else if let Ok(n) = value.extract::<i64>() {
        format!("{}", n)
    } else if value.is_instance_of::<PyComplex>() {
        let c = value.cast::<PyComplex>()?;
        format!("({}, {})", format_fits_float(c.real()), format_fits_float(c.imag()))
    } else if let Ok(f) = value.extract::<f64>() {
        format_fits_float(f)
    } else if let Ok(s) = value.extract::<String>() {
        let escaped = s.replace('\'', "''");
        let padded = if escaped.len() < 8 {
            format!("{:<8}", escaped)
        } else {
            escaped
        };
        format!("'{}'", padded)
    } else {
        return Err(PyValueError::new_err(format!(
            "unsupported value type for HIERARCH key '{}'; \
             supported: bool, int, float, complex, str", key
        )));
    };

    let prefix = format!("HIERARCH {} = ", key);
    let body = if comment.is_empty() {
        format!("{}{}", prefix, value_str)
    } else {
        format!("{}{} / {}", prefix, value_str, comment)
    };
    if body.len() > CARD_SIZE {
        return Err(PyValueError::new_err(format!(
            "HIERARCH card too long ({} chars) for key '{}'; \
             shorten the key, value, or comment to fit in 80 chars \
             (CONTINUE-on-write for HIERARCH long-string values is deferred)",
            body.len(), key
        )));
    }
    Ok(pad_to_card(&body))
}

// Does this card's value (parsed as a FITS string) end with `&`?  Returns
// false for non-string values and non-card text.  Used to walk CONTINUE
// chains during updates.
fn card_value_ends_with_amp(card: &str) -> bool {
    let trimmed = card.trim_end();
    let value_part: &str = if trimmed.starts_with("CONTINUE") {
        if trimmed.len() > 8 { &trimmed[8..] } else { return false; }
    } else if let Some(eq_pos) = trimmed.find('=') {
        &trimmed[eq_pos + 1..]
    } else {
        return false;
    };
    let (raw, _) = split_value_comment(value_part);
    if raw.starts_with('\'') && raw.ends_with('\'') && raw.len() >= 2 {
        let inner = &raw[1..raw.len() - 1];
        return inner.ends_with('&');
    }
    false
}

// Given that cards[start] is the first card for some key, walk forward as
// long as the previous card has a `&`-terminated string value AND the next
// card has keyword field "CONTINUE".  Returns the index one past the last
// card in the chain.  Always returns at least start + 1.
fn find_chain_end(cards: &[String], start: usize) -> usize {
    let mut end = start + 1;
    while end < cards.len() && card_value_ends_with_amp(&cards[end - 1]) {
        let next = cards[end].trim_end();
        if !next.starts_with("CONTINUE") {
            break;
        }
        end += 1;
    }
    end
}

// Replace the chain of cards for `key` with `new_cards`, preserving the
// chain's original position.  If `key` is not present, insert `new_cards`
// just before END (or appended if there's no END card).  When updating an
// existing key whose value is a CONTINUE-chained string, the entire chain
// is removed before the new cards are inserted.  Matches HIERARCH cards
// by their long key, not by the literal "HIERARCH" prefix.
fn set_card_for_key(cards: &mut Vec<String>, key: &str, new_cards: Vec<String>) {
    let normalized: Vec<String> = new_cards
        .into_iter()
        .map(|c| c.trim_end().to_string())
        .collect();

    let mut existing_start: Option<usize> = None;
    for i in 0..cards.len() {
        let card_key = match keyword_of(&cards[i]) {
            Some(k) => k,
            None => continue,
        };
        // Skip structural / commentary / continuation cards — never match them.
        if matches!(card_key.as_str(), "END" | "COMMENT" | "HISTORY" | "" | "CONTINUE") {
            continue;
        }
        if card_key == key {
            existing_start = Some(i);
            break;
        }
    }

    match existing_start {
        Some(start) => {
            let end = find_chain_end(cards, start);
            cards.splice(start..end, normalized);
        }
        None => {
            let end_pos = cards
                .iter()
                .position(|c| c.trim_end() == "END")
                .unwrap_or(cards.len());
            cards.splice(end_pos..end_pos, normalized);
        }
    }
}

// Remove the first card matching `key`, plus any CONTINUE chain attached
// to it (for long-string values).  Returns true iff at least one card was
// removed.  Matches HIERARCH cards by their long key.
fn delete_card_for_key(cards: &mut Vec<String>, key: &str) -> bool {
    for i in 0..cards.len() {
        let card_key = match keyword_of(&cards[i]) {
            Some(k) => k,
            None => continue,
        };
        if matches!(card_key.as_str(), "END" | "COMMENT" | "HISTORY" | "" | "CONTINUE") {
            continue;
        }
        if card_key == key {
            let end = find_chain_end(cards, i);
            cards.drain(i..end);
            return true;
        }
    }
    false
}

// Parse the value side of `header[key] = value`.  Either a bare scalar or a
// `(value, comment)` 2-tuple is accepted; any other tuple shape is an error.
fn parse_setitem_value<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Option<String>)> {
    if let Ok(tup) = value.cast::<PyTuple>() {
        if tup.len() != 2 {
            return Err(PyValueError::new_err(
                "tuple value must be (value, comment) — exactly 2 elements"
            ));
        }
        let v = tup.get_item(0)?;
        let c: String = tup.get_item(1)?.extract().map_err(|_| {
            PyValueError::new_err(
                "second element of (value, comment) tuple must be a string"
            )
        })?;
        Ok((v, Some(c)))
    } else {
        Ok((value.clone(), None))
    }
}

// Apply a single header set — validate + resolve comment + build card +
// place it in `cards` (in-place or append-before-END).  Shared between the
// auto-flush FITSHeader path and the batched FITSHeaderEdit path; neither
// writes to disk here.  `key` is assumed normalized to uppercase.
fn apply_setitem(
    cards: &mut Vec<String>,
    key: &str,
    value: &Bound<'_, PyAny>,
    explicit_comment: Option<String>,
) -> PyResult<()> {
    validate_keyword(key)?;
    let comment = match explicit_comment {
        Some(c) => c,
        None => extract_existing_comment(value.py(), cards, key).unwrap_or_default(),
    };
    let new_cards = build_card_from_value(key, value, &comment)?;
    set_card_for_key(cards, key, new_cards);
    Ok(())
}

// Collect (key, value, explicit_comment) triples to apply from a source —
// either a FITSHeader (cards are walked to preserve comments) or any object
// with `.items()` returning (key, value-or-(value, comment)).  Commentary
// keys in the source raise (deferred to phase 2c).
fn collect_update_triples(
    py: Python<'_>,
    other: &Bound<'_, PyAny>,
) -> PyResult<Vec<(String, Py<PyAny>, Option<String>)>> {
    let mut triples: Vec<(String, Py<PyAny>, Option<String>)> = Vec::new();

    if let Ok(src) = other.cast::<FITSHeader>() {
        let src_cards = src.borrow().snapshot()?;
        for (i, card) in src_cards.iter().enumerate() {
            let trimmed = card.trim_end();
            if trimmed.is_empty() {
                continue;
            }
            let key_field = if trimmed.len() >= 8 { &trimmed[..8] } else { trimmed };
            let kw_field_trimmed = key_field.trim();
            if matches!(kw_field_trimmed, "END" | "CONTINUE") {
                continue;
            }
            if matches!(kw_field_trimmed, "COMMENT" | "HISTORY" | "") {
                return Err(PyValueError::new_err(format!(
                    "source header contains commentary key '{}' which \
                     update() does not yet support (deferred)",
                    kw_field_trimmed
                )));
            }
            let kw = keyword_of(card).unwrap_or_default();
            if kw.is_empty() {
                continue;
            }
            if let Some(eq_pos) = trimmed.find('=') {
                let (value, comment) = parse_value_with_continue(
                    py, &src_cards, i, &trimmed[eq_pos + 1..],
                )?;
                triples.push((kw, value, Some(comment)));
            }
        }
    } else {
        let items_call = other.call_method0("items").map_err(|_| {
            PyValueError::new_err(
                "update() expects a FITSHeader or a mapping with .items()"
            )
        })?;
        let iter = items_call.try_iter()?;
        for item in iter {
            let item = item?;
            let pair = item.cast::<PyTuple>().map_err(|_| {
                PyValueError::new_err(
                    "update() expects pairs of (key, value); got non-tuple"
                )
            })?;
            if pair.len() != 2 {
                return Err(PyValueError::new_err(
                    "update() expects pairs of (key, value); got tuple of wrong length"
                ));
            }
            let k_raw: String = pair.get_item(0)?.extract()?;
            let k = normalize_keyword(&k_raw);
            if matches!(k.as_str(), "COMMENT" | "HISTORY" | "") {
                return Err(PyValueError::new_err(format!(
                    "update() does not yet support commentary key '{}' (deferred)", k
                )));
            }
            let val_obj = pair.get_item(1)?;
            let (v, c) = parse_setitem_value(&val_obj)?;
            triples.push((k, v.unbind(), c));
        }
    }

    Ok(triples)
}

// Rewrite the header on disk in place.  Cards are serialized to one or more
// 2880-byte blocks; the resulting byte count must equal `header_block_count
// * BLOCK_SIZE` (slack-only).  If `cards` overflows the reserved blocks,
// returns a clear error rather than relocating subsequent HDUs.
//
// Taint semantics: a failure of the slack-overflow check, the file lock,
// the closed-file check, or the initial `seek` leaves the on-disk file
// untouched and is NOT a taint event.  A failure of `write_all` or `flush`
// MAY have partially overwritten the header on disk, so we set the taint
// flag and return a diagnostic error pointing the user at the recovery
// path (close + reopen).
fn rewrite_header_to_disk(
    file_handle: &FileHandle,
    header_offset: u64,
    header_block_count: u64,
    cards: &[String],
    tainted: &TaintFlag,
) -> PyResult<()> {
    let max_cards = (header_block_count as usize) * 36;
    if cards.len() > max_cards {
        return Err(PyValueError::new_err(format!(
            "header overflow: {} cards do not fit in {} block(s) (max {} cards); \
             growing the header beyond its reserved blocks requires rewriting \
             subsequent HDUs, which is not yet supported",
            cards.len(), header_block_count, max_cards
        )));
    }

    let target_bytes = (header_block_count * BLOCK_SIZE as u64) as usize;
    let mut bytes = Vec::with_capacity(target_bytes);
    for card in cards {
        let mut padded = card.clone();
        if padded.len() < CARD_SIZE {
            padded.push_str(&" ".repeat(CARD_SIZE - padded.len()));
        } else if padded.len() > CARD_SIZE {
            padded.truncate(CARD_SIZE);
        }
        bytes.extend_from_slice(padded.as_bytes());
    }
    while bytes.len() < target_bytes {
        bytes.push(b' ');
    }

    let mut guard = lock_file(file_handle)?;
    let f = guard.as_mut().ok_or_else(|| PyIOError::new_err("file is closed"))?;
    // seek before any write: failure here does not modify the file.
    f.seek(SeekFrom::Start(header_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    // write_all: a failure here may have written some bytes already.  Taint.
    f.write_all(&bytes).map_err(|e| {
        tainted.store(true, Ordering::Release);
        PyIOError::new_err(format!(
            "header write failed mid-stream: {}; \
             the on-disk file may now be inconsistent — \
             close this FITS object and reopen the file to recover",
            e
        ))
    })?;
    // flush: an OS-level flush failure means data is in flight.  Taint.
    f.flush().map_err(|e| {
        tainted.store(true, Ordering::Release);
        PyIOError::new_err(format!(
            "header flush failed: {}; \
             the on-disk file may now be inconsistent — \
             close this FITS object and reopen the file to recover",
            e
        ))
    })?;
    Ok(())
}

// ====================== Commentary write helpers (phase 2c) ======================

// Validate text destined for a commentary card.  Per the FITS standard,
// every byte in a header card must be printable ASCII (0x20-0x7E).
fn validate_commentary_text(text: &str) -> PyResult<()> {
    for c in text.chars() {
        let b = c as u32;
        if !(0x20..=0x7E).contains(&b) {
            return Err(PyValueError::new_err(format!(
                "commentary text contains non-printable character (0x{:02X}); \
                 FITS restricts cards to ASCII 0x20-0x7E", b
            )));
        }
    }
    Ok(())
}

// Build one or more commentary cards for the given keyword.  Text longer
// than 72 chars (cols 9-80) is split across multiple cards each carrying
// the same keyword — FITS has no CONTINUE convention for COMMENT/HISTORY.
// `keyword` is "COMMENT", "HISTORY", or "" (blank-keyword commentary).
fn make_commentary_cards(keyword: &str, text: &str) -> Vec<String> {
    let max_text = CARD_SIZE - 8; // 72
    let prefix = if keyword.is_empty() {
        "        ".to_string()
    } else {
        format!("{:<8}", keyword)
    };
    if text.is_empty() {
        return vec![pad_to_card(&prefix)];
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len().div_ceil(max_text));
    for chunk in bytes.chunks(max_text) {
        // Safe: validate_commentary_text confirmed ASCII printable; ASCII
        // chunks are valid UTF-8.
        let chunk_str = std::str::from_utf8(chunk).unwrap_or("");
        out.push(pad_to_card(&format!("{}{}", prefix, chunk_str)));
    }
    out
}

// Does this card match the commentary keyword?  "" means blank-keyword
// commentary (cols 1-8 all spaces with content in cols 9-80).
fn card_matches_commentary_keyword(card: &str, keyword: &str) -> bool {
    let trimmed = card.trim_end();
    if keyword.is_empty() {
        trimmed.len() > 8 && trimmed[..8].trim().is_empty()
    } else {
        if trimmed.is_empty() {
            return false;
        }
        let kf = if trimmed.len() >= 8 { &trimmed[..8] } else { trimmed };
        kf.trim() == keyword
    }
}

// Append commentary cards to `cards`.  Position: immediately after the last
// existing card with the same commentary keyword; if none exists, just
// before END (or appended if no END card is present).
fn append_commentary_to_cards(cards: &mut Vec<String>, keyword: &str, text: &str) {
    let new_cards = make_commentary_cards(keyword, text);
    let mut last_match: Option<usize> = None;
    for (i, card) in cards.iter().enumerate() {
        if card_matches_commentary_keyword(card, keyword) {
            last_match = Some(i);
        }
    }
    let pos = match last_match {
        Some(i) => i + 1,
        None => cards.iter()
            .position(|c| c.trim_end() == "END")
            .unwrap_or(cards.len()),
    };
    for (offset, card) in new_cards.into_iter().enumerate() {
        cards.insert(pos + offset, card.trim_end().to_string());
    }
}

// Remove all cards matching the commentary keyword.  Returns how many were
// removed (0 means the key was absent).
fn delete_commentary_cards(cards: &mut Vec<String>, keyword: &str) -> usize {
    let before = cards.len();
    cards.retain(|card| !card_matches_commentary_keyword(card, keyword));
    before - cards.len()
}

#[pyclass]
struct FITSHeader {
    // Shared with the owning HDU so mutations propagate back.  Reads take the
    // lock briefly and clone (headers are tiny); writes take the lock for the
    // duration of the rewrite.
    cards: Arc<Mutex<Vec<String>>>,
    // File handle + on-disk position needed to rewrite the header in place.
    file: FileHandle,
    header_offset: u64,
    // Number of 2880-byte header blocks reserved on disk.  Mutations may
    // grow or shrink the card list up to this block's worth of cards;
    // overflow returns an error (file-rewrite path is deferred).
    header_block_count: u64,
    // Shared per-file taint flag.  Set by `rewrite_header_to_disk` on a
    // mid-write I/O failure; checked at the entry of every read and write
    // method.  See `check_not_tainted`.
    tainted: TaintFlag,
}

impl FITSHeader {
    fn from_state(
        cards: Arc<Mutex<Vec<String>>>,
        file: FileHandle,
        header_offset: u64,
        header_block_count: u64,
        tainted: TaintFlag,
    ) -> Self {
        FITSHeader { cards, file, header_offset, header_block_count, tainted }
    }

    fn snapshot(&self) -> PyResult<Vec<String>> {
        check_not_tainted(&self.tainted)?;
        let g = self.cards.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        Ok(g.clone())
    }

    // Shared body for add_comment / add_history / add_blank.  Validates the
    // text, builds the new card(s), writes to disk first, then commits the
    // mutation to the in-memory cards.
    fn append_commentary(&self, keyword: &str, text: &str) -> PyResult<()> {
        check_not_tainted(&self.tainted)?;
        validate_commentary_text(text)?;
        let mut guard = self.cards.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        let mut new_cards = guard.clone();
        append_commentary_to_cards(&mut new_cards, keyword, text);
        rewrite_header_to_disk(
            &self.file,
            self.header_offset,
            self.header_block_count,
            &new_cards,
            &self.tainted,
        )?;
        *guard = new_cards;
        Ok(())
    }
}

#[pymethods]
impl FITSHeader {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let key = normalize_keyword(key);
        let cards = self.snapshot()?;

        // Commentary keys are returned as ordered lists of card texts.
        let commentary_list = match key.as_str() {
            "COMMENT" => Some(collect_commentary_texts(&cards, "COMMENT")),
            "HISTORY" => Some(collect_commentary_texts(&cards, "HISTORY")),
            "" => Some(collect_blank_commentary_texts(&cards)),
            _ => None,
        };
        if let Some(items) = commentary_list {
            if items.is_empty() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    format!("'{}' not in header", key)
                ));
            }
            return Ok(PyList::new(py, &items)?.into_any().unbind());
        }

        match find_card_for_key(&cards, &key) {
            Some((idx, value_part)) => {
                let (value, _comment) =
                    parse_value_with_continue(py, &cards, idx, &value_part)?;
                Ok(value)
            }
            None => Err(pyo3::exceptions::PyKeyError::new_err(
                format!("'{}' not in header", key)
            )),
        }
    }

    fn __contains__(&self, key: &str) -> PyResult<bool> {
        let key = normalize_keyword(key);
        let cards = self.snapshot()?;
        Ok(match key.as_str() {
            "COMMENT" => cards.iter().any(|c| keyword_of(c).as_deref() == Some("COMMENT")),
            "HISTORY" => cards.iter().any(|c| keyword_of(c).as_deref() == Some("HISTORY")),
            "" => cards.iter().any(|c| {
                let trimmed = c.trim_end();
                trimmed.len() > 8 && trimmed[..8].trim().is_empty()
            }),
            _ => find_card_for_key(&cards, &key).is_some(),
        })
    }

    fn __len__(&self) -> PyResult<usize> {
        let cards = self.snapshot()?;
        Ok(unique_keys_in_order(&cards).len())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let cards = slf.snapshot()?;
        let keys = unique_keys_in_order(&cards);
        let list = PyList::new(py, &keys)?;
        Ok(list.call_method0("__iter__")?.unbind())
    }

    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let cards = self.snapshot()?;
        let keys = unique_keys_in_order(&cards);
        Ok(PyList::new(py, &keys)?.unbind())
    }

    fn values(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let cards = self.snapshot()?;
        let keys = unique_keys_in_order(&cards);
        let mut values: Vec<Py<PyAny>> = Vec::with_capacity(keys.len());
        for k in &keys {
            values.push(self.__getitem__(py, k)?);
        }
        Ok(PyList::new(py, &values)?.unbind())
    }

    fn items(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let cards = self.snapshot()?;
        let keys = unique_keys_in_order(&cards);
        let mut items: Vec<Py<PyAny>> = Vec::with_capacity(keys.len());
        for k in &keys {
            let v = self.__getitem__(py, k)?;
            let tup = pyo3::types::PyTuple::new(py, &[k.clone().into_py_any(py)?, v])?;
            items.push(tup.into_any().unbind());
        }
        Ok(PyList::new(py, &items)?.unbind())
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match self.__getitem__(py, key) {
            Ok(v) => Ok(v),
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyKeyError>(py) => {
                Ok(default.unwrap_or_else(|| py.None()))
            }
            Err(e) => Err(e),
        }
    }

    fn comment_of(&self, py: Python<'_>, key: &str) -> PyResult<String> {
        let key = normalize_keyword(key);
        if matches!(key.as_str(), "COMMENT" | "HISTORY" | "") {
            return Err(PyValueError::new_err(
                "comment_of() is not defined for commentary keys (COMMENT/HISTORY/blank)"
            ));
        }
        let cards = self.snapshot()?;
        match find_card_for_key(&cards, &key) {
            Some((idx, value_part)) => {
                let (_value, comment) =
                    parse_value_with_continue(py, &cards, idx, &value_part)?;
                Ok(comment)
            }
            None => Err(pyo3::exceptions::PyKeyError::new_err(
                format!("'{}' not in header", key)
            )),
        }
    }

    #[getter]
    fn cards(&self) -> PyResult<Vec<String>> {
        self.snapshot()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let cards = self.snapshot()?;
        parse_header_dict(py, &cards)
    }

    // ---- mutation (phase 2a) ----

    // Set or update a key.  `value` is either a bare scalar (bool/int/float/str)
    // or a `(value, comment)` tuple.  Bare values preserve the existing
    // comment if the key was already present.  Existing keys are updated in
    // place (position preserved); new keys are inserted just before END.
    // Each call auto-flushes — one disk rewrite per mutation.
    fn __setitem__(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        check_not_tainted(&self.tainted)?;
        let key = normalize_keyword(key);
        if matches!(key.as_str(), "COMMENT" | "HISTORY" | "") {
            return Err(PyValueError::new_err(format!(
                "subscript assignment to commentary key '{}' is not supported; \
                 use add_comment(text) / add_history(text) / add_blank(text) to append",
                key
            )));
        }
        let (value_obj, explicit_comment) = parse_setitem_value(value)?;
        // Build the candidate card list on a clone, write to disk, only then
        // commit the new state to the shared in-memory cards.  Order matters:
        // if the disk write fails (e.g. header overflow), in-memory must not
        // diverge from disk.
        let mut guard = self.cards.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        let mut new_cards = guard.clone();
        apply_setitem(&mut new_cards, &key, &value_obj, explicit_comment)?;
        rewrite_header_to_disk(
            &self.file,
            self.header_offset,
            self.header_block_count,
            &new_cards,
            &self.tainted,
        )?;
        *guard = new_cards;
        Ok(())
    }

    fn __delitem__(&self, key: &str) -> PyResult<()> {
        check_not_tainted(&self.tainted)?;
        let key = normalize_keyword(key);
        let mut guard = self.cards.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        let mut new_cards = guard.clone();
        let removed = if matches!(key.as_str(), "COMMENT" | "HISTORY" | "") {
            delete_commentary_cards(&mut new_cards, &key) > 0
        } else {
            delete_card_for_key(&mut new_cards, &key)
        };
        if !removed {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                format!("'{}' not in header", key)
            ));
        }
        rewrite_header_to_disk(
            &self.file,
            self.header_offset,
            self.header_block_count,
            &new_cards,
            &self.tainted,
        )?;
        *guard = new_cards;
        Ok(())
    }

    // Append a COMMENT card.  Long text auto-splits across multiple cards
    // (FITS has no CONTINUE convention for commentary keys).
    fn add_comment(&self, text: &str) -> PyResult<()> {
        self.append_commentary("COMMENT", text)
    }

    // Append a HISTORY card.  Long text auto-splits across multiple cards.
    fn add_history(&self, text: &str) -> PyResult<()> {
        self.append_commentary("HISTORY", text)
    }

    // Append a blank-keyword commentary card (cols 1-8 blank).  Long text
    // auto-splits across multiple cards.
    fn add_blank(&self, text: &str) -> PyResult<()> {
        self.append_commentary("", text)
    }

    // Bulk update from a mapping or another FITSHeader.  Performs all
    // mutations under a single lock and disk rewrite.  Commentary keys
    // (COMMENT/HISTORY/blank) in the source raise — they need explicit
    // append/replace helpers (phase 2b).
    //
    // For a FITSHeader source we copy cards directly to preserve comments;
    // for any other mapping we use `.items()`, where each value is either a
    // bare scalar or a `(value, comment)` tuple.
    fn update(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        check_not_tainted(&self.tainted)?;
        let triples = collect_update_triples(py, other)?;
        if triples.is_empty() {
            return Ok(());
        }
        // Stage all mutations on a working clone, write to disk, then
        // commit — so any failure mid-update leaves the in-memory state
        // consistent with what's on disk.
        let mut guard = self.cards.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        let mut new_cards = guard.clone();
        for (k, v, explicit_comment) in &triples {
            apply_setitem(&mut new_cards, k, v.bind(py), explicit_comment.clone())?;
        }
        rewrite_header_to_disk(
            &self.file,
            self.header_offset,
            self.header_block_count,
            &new_cards,
            &self.tainted,
        )?;
        *guard = new_cards;
        Ok(())
    }

    // Return a transactional edit handle.  Mutations on the handle stage in
    // memory; on successful __exit__ they're committed to the parent header
    // and disk in a single write.  An exception inside the `with` block
    // discards the staged mutations.
    fn edit(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<FITSHeaderEdit>> {
        let snapshot = slf.bind(py).borrow().snapshot()?;
        Py::new(py, FITSHeaderEdit {
            parent: slf,
            cards: snapshot,
            entered: false,
            committed: false,
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let cards = self.snapshot()?;
        let n_unique = unique_keys_in_order(&cards).len();
        Ok(format!(
            "<FITSHeader: {} unique keys, {} cards>",
            n_unique, cards.len()
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let cards = self.snapshot()?;
        Ok(cards.iter().map(|c| c.trim_end()).collect::<Vec<_>>().join("\n"))
    }
}

// ====================== FITSHeaderEdit: transactional header edits ======================
//
// A FITSHeaderEdit is the value bound by `with header.edit() as h:`.  It owns
// a snapshot of the parent FITSHeader's cards taken at edit() time; all
// mutations on the handle stage into that snapshot, leaving the parent (and
// the on-disk file) untouched.  On a successful __exit__ the snapshot is
// committed back to the parent and written to disk in one rewrite; an
// exception that escapes the `with` block discards the staged changes.
//
// Mutations outside a `with` block are rejected with a clear error so a
// stale, never-committed handle cannot silently swallow user changes.
#[pyclass]
struct FITSHeaderEdit {
    parent: Py<FITSHeader>,
    cards: Vec<String>,    // staged state — owned, no locking needed
    entered: bool,
    committed: bool,
}

impl FITSHeaderEdit {
    fn ensure_active(&self) -> PyResult<()> {
        if !self.entered {
            return Err(PyValueError::new_err(
                "FITSHeaderEdit must be used inside a `with header.edit():` block"
            ));
        }
        if self.committed {
            return Err(PyValueError::new_err(
                "FITSHeaderEdit has already been committed"
            ));
        }
        Ok(())
    }

    // Push staged cards into the parent FITSHeader and write to disk.
    // Called from __exit__; never invoked from Python directly.  Disk write
    // happens first; if it fails (e.g. header overflow), the parent's
    // in-memory cards are left untouched and stay consistent with disk.
    fn commit_internal(&self, py: Python<'_>) -> PyResult<()> {
        let parent = self.parent.bind(py).borrow();
        check_not_tainted(&parent.tainted)?;
        let mut g = parent.cards.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        rewrite_header_to_disk(
            &parent.file,
            parent.header_offset,
            parent.header_block_count,
            &self.cards,
            &parent.tainted,
        )?;
        *g = self.cards.clone();
        Ok(())
    }
}

#[pymethods]
impl FITSHeaderEdit {
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.entered = true;
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let was_entered = self.entered;
        self.entered = false;
        // Commit only if we actually entered, no exception escaped, and we
        // haven't already committed.  Don't swallow the user's exception.
        if was_entered && exc_type.is_none() && !self.committed {
            self.commit_internal(py)?;
            self.committed = true;
        }
        Ok(false)
    }

    fn __setitem__(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_active()?;
        let key = normalize_keyword(key);
        if matches!(key.as_str(), "COMMENT" | "HISTORY" | "") {
            return Err(PyValueError::new_err(format!(
                "subscript assignment to commentary key '{}' is not supported; \
                 use add_comment(text) / add_history(text) / add_blank(text) to append",
                key
            )));
        }
        let (value_obj, explicit_comment) = parse_setitem_value(value)?;
        apply_setitem(&mut self.cards, &key, &value_obj, explicit_comment)
    }

    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self.ensure_active()?;
        let key = normalize_keyword(key);
        let removed = if matches!(key.as_str(), "COMMENT" | "HISTORY" | "") {
            delete_commentary_cards(&mut self.cards, &key) > 0
        } else {
            delete_card_for_key(&mut self.cards, &key)
        };
        if !removed {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                format!("'{}' not in header", key)
            ));
        }
        Ok(())
    }

    fn add_comment(&mut self, text: &str) -> PyResult<()> {
        self.ensure_active()?;
        validate_commentary_text(text)?;
        append_commentary_to_cards(&mut self.cards, "COMMENT", text);
        Ok(())
    }

    fn add_history(&mut self, text: &str) -> PyResult<()> {
        self.ensure_active()?;
        validate_commentary_text(text)?;
        append_commentary_to_cards(&mut self.cards, "HISTORY", text);
        Ok(())
    }

    fn add_blank(&mut self, text: &str) -> PyResult<()> {
        self.ensure_active()?;
        validate_commentary_text(text)?;
        append_commentary_to_cards(&mut self.cards, "", text);
        Ok(())
    }

    fn update(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_active()?;
        let triples = collect_update_triples(py, other)?;
        for (k, v, explicit_comment) in &triples {
            apply_setitem(&mut self.cards, k, v.bind(py), explicit_comment.clone())?;
        }
        Ok(())
    }

    // Reads observe the staged state — what you'd see if you committed now.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let key = normalize_keyword(key);
        let commentary_list = match key.as_str() {
            "COMMENT" => Some(collect_commentary_texts(&self.cards, "COMMENT")),
            "HISTORY" => Some(collect_commentary_texts(&self.cards, "HISTORY")),
            "" => Some(collect_blank_commentary_texts(&self.cards)),
            _ => None,
        };
        if let Some(items) = commentary_list {
            if items.is_empty() {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    format!("'{}' not in header", key)
                ));
            }
            return Ok(PyList::new(py, &items)?.into_any().unbind());
        }
        match find_card_for_key(&self.cards, &key) {
            Some((idx, value_part)) => {
                let (value, _comment) =
                    parse_value_with_continue(py, &self.cards, idx, &value_part)?;
                Ok(value)
            }
            None => Err(pyo3::exceptions::PyKeyError::new_err(
                format!("'{}' not in header", key)
            )),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        let key = normalize_keyword(key);
        match key.as_str() {
            "COMMENT" => self.cards.iter().any(|c| keyword_of(c).as_deref() == Some("COMMENT")),
            "HISTORY" => self.cards.iter().any(|c| keyword_of(c).as_deref() == Some("HISTORY")),
            "" => self.cards.iter().any(|c| {
                let trimmed = c.trim_end();
                trimmed.len() > 8 && trimmed[..8].trim().is_empty()
            }),
            _ => find_card_for_key(&self.cards, &key).is_some(),
        }
    }

    fn __repr__(&self) -> String {
        let state = if self.committed {
            "committed"
        } else if self.entered {
            "active"
        } else {
            "pending"
        };
        format!(
            "<FITSHeaderEdit: {} cards, {}>",
            self.cards.len(), state
        )
    }
}

// ====================== Base HDU ======================
// HDUs hold a clone of the FITS file handle plus the byte offset of their
// data section, enabling write-back methods on subclasses (e.g. ImageHDU.write).
// `#[new]` is intentionally omitted from HDU and its subclasses: instances are
// constructed only via FITS internals (which know the file handle and offset);
// direct Python instantiation is not supported.
#[pyclass(subclass)]
struct HDU {
    // Shared with FITSHeader so mutations through the header view propagate
    // back to the HDU's canonical card list (and any other readers).
    header: Arc<Mutex<Vec<String>>>,
    index: usize,
    // On-disk byte offset where this HDU's header begins.  Fixed for the HDU's
    // lifetime; used by FITSHeader to rewrite header blocks in place.
    header_offset: u64,
    data_offset: u64,
    file: FileHandle,
    // Shared per-file taint flag (one clone of the same Arc across all HDUs
    // of this FITS and across every FITSHeader / FITSHeaderEdit produced).
    tainted: TaintFlag,
}

impl HDU {
    fn new(
        header: Vec<String>,
        index: usize,
        header_offset: u64,
        data_offset: u64,
        file: FileHandle,
        tainted: TaintFlag,
    ) -> Self {
        HDU {
            header: Arc::new(Mutex::new(header)),
            index,
            header_offset,
            data_offset,
            file,
            tainted,
        }
    }

    // Snapshot of the current header cards.  Takes the lock briefly, clones
    // out, releases.  Internal readers should call this rather than holding
    // the lock for the duration of (e.g.) a numpy allocation.
    fn header_snapshot(&self) -> PyResult<Vec<String>> {
        check_not_tainted(&self.tainted)?;
        let g = self.header.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        Ok(g.clone())
    }

    // How many 2880-byte blocks the on-disk header occupies (fixed; equal to
    // (data_offset - header_offset) / BLOCK_SIZE).  Used to enforce slack-only
    // header mutations.
    fn header_block_count(&self) -> u64 {
        (self.data_offset - self.header_offset) / BLOCK_SIZE as u64
    }
}

#[pymethods]
impl HDU {
    fn __repr__(&self) -> String {
        format!("<HDU #{}>", self.index)
    }

    // The header view.  Returns a FITSHeader that shares the HDU's card list,
    // file handle, and on-disk position — mutations through the returned
    // object propagate back to the HDU and write through to disk.
    #[getter]
    fn header(&self, py: Python<'_>) -> PyResult<Py<FITSHeader>> {
        Py::new(py, FITSHeader::from_state(
            Arc::clone(&self.header),
            Arc::clone(&self.file),
            self.header_offset,
            self.header_block_count(),
            Arc::clone(&self.tainted),
        ))
    }

    // Test-only hook: flip the taint flag without an actual disk failure.
    // Used by the phase-2d test suite to verify the rejection-behavior
    // contract (subsequent reads/writes raise with the diagnostic message)
    // without needing to produce a real I/O failure on the host filesystem.
    // The underscore prefix is the standard Python signal for "private,
    // not part of the public API."
    fn _force_taint(&self) {
        self.tainted.store(true, Ordering::Release);
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }
}

// ====================== Specialized HDU subclasses ======================
#[pyclass(extends = HDU)]
struct ImageHDU;

impl ImageHDU {
    fn new(
        header: Vec<String>,
        index: usize,
        header_offset: u64,
        data_offset: u64,
        file: FileHandle,
        tainted: TaintFlag,
    ) -> (Self, HDU) {
        (
            ImageHDU,
            HDU::new(header, index, header_offset, data_offset, file, tainted),
        )
    }
}

#[pymethods]
impl ImageHDU {
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let index: usize = super_.index();
        Ok(format!("<ImageHDU #{}>", index))
    }

    // Write a numpy array into this HDU's data section.  `data` may cover the
    // whole HDU or a sub-region; `start` (numpy-order, defaults to origin)
    // names the top-left corner of the region in the HDU.  Delegates to the
    // free function `write_image_data` (also used by `extend`).
    #[pyo3(signature = (data, start=None))]
    fn write(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        start: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        write_image_data(&header_cards, super_.data_offset, &super_.file, data, start)
    }

    // Read this HDU's entire data section into a newly-allocated numpy
    // array.  The result is native-endian; FITS on-disk big-endian bytes
    // are swapped into host byte order in place.
    fn read(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        read_image_data(py, &header_cards, super_.data_offset, &super_.file)
    }

    // Grow the HDU's slow axis (numpy axis 0 = FITS NAXISn) if needed to fit
    // the data being written, then write it.  `start` is in numpy order and
    // defaults to (current_slow_axis_dim, 0, 0, ...) so the new data lands
    // immediately after the existing data.
    //
    // Only the slow axis may grow; inner axes (numpy 1..n-1) must accommodate
    // `start + data_shape` within the existing HDU dimensions.  The HDU must
    // currently be the last on disk (extending non-last HDUs requires moving
    // subsequent HDUs, which is not yet supported).
    //
    // On success: the file is extended (sparse, so zero-filled), the on-disk
    // header's NAXISn card and the in-memory header are updated to the new
    // size, and then the array is written via `write_image_data`.
    #[pyo3(signature = (data, start=None))]
    fn extend(
        slf: PyRefMut<'_, Self>,
        data: &Bound<'_, PyAny>,
        start: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let super_: PyRefMut<HDU> = slf.into_super();

        let header_snapshot = super_.header_snapshot()?;
        let (bitpix, current_hdu_shape) = parse_image_hdu_shape(&header_snapshot)?;
        let naxis = current_hdu_shape.len();

        // Validate dtype matches BITPIX up front so we don't touch the file
        // before we know the write will be acceptable.
        let dtype = data.getattr("dtype")?;
        let kind: String = dtype.getattr("kind")?.extract()?;
        let itemsize_attr: u64 = dtype.getattr("itemsize")?.extract()?;
        let (expected_kind, expected_size) = bitpix_to_numpy_kind(bitpix)?;
        if kind != expected_kind || itemsize_attr != expected_size {
            return Err(PyValueError::new_err(format!(
                "data dtype ({}{}) does not match HDU BITPIX={} (expected {}{})",
                kind, itemsize_attr, bitpix, expected_kind, expected_size,
            )));
        }

        let data_shape: Vec<u64> = data.getattr("shape")?.extract()?;
        if data_shape.len() != naxis {
            return Err(PyValueError::new_err(format!(
                "data has {} axes, HDU has {}", data_shape.len(), naxis
            )));
        }

        // Resolve `start`.  Default puts the new data immediately after the
        // existing data along numpy axis 0 (FITS slow axis).
        let start_vec: Vec<u64> = match start {
            Some(s) => {
                if s.len() != naxis {
                    return Err(PyValueError::new_err(format!(
                        "start has {} components, expected {}", s.len(), naxis
                    )));
                }
                let mut out = Vec::with_capacity(naxis);
                for (i, &v) in s.iter().enumerate() {
                    if v < 0 {
                        return Err(PyValueError::new_err(format!(
                            "start[{}] must be >= 0, got {}", i, v
                        )));
                    }
                    out.push(v as u64);
                }
                out
            }
            None => {
                let mut out = vec![0u64; naxis];
                out[0] = current_hdu_shape[0];
                out
            }
        };

        // Inner axes (numpy 1..n-1) cannot grow — only the slow axis may.
        for i in 1..naxis {
            if start_vec[i] + data_shape[i] > current_hdu_shape[i] {
                return Err(PyValueError::new_err(format!(
                    "axis {}: start ({}) + data shape ({}) exceeds HDU dim ({}); \
                     extend only grows the slow axis (numpy axis 0)",
                    i, start_vec[i], data_shape[i], current_hdu_shape[i]
                )));
            }
        }

        // Compute new shape.  Only axis 0 may grow.
        let mut new_hdu_shape = current_hdu_shape.clone();
        let needed = start_vec[0] + data_shape[0];
        if needed > new_hdu_shape[0] {
            new_hdu_shape[0] = needed;
        }

        let start_for_write: Vec<i64> = start_vec.iter().map(|&v| v as i64).collect();

        // No growth needed: fall through to a plain write.
        if new_hdu_shape == current_hdu_shape {
            return write_image_data(
                &header_snapshot,
                super_.data_offset,
                &super_.file,
                data,
                Some(start_for_write),
            );
        }

        // ----- Growth path -----
        let bpp = expected_size;
        let current_data_size: u64 = current_hdu_shape.iter().product::<u64>() * bpp;
        let new_data_size: u64 = new_hdu_shape.iter().product::<u64>() * bpp;
        let current_padded = round_up_to_block(current_data_size);
        let new_padded = round_up_to_block(new_data_size);

        let data_offset = super_.data_offset;
        let current_hdu_end = data_offset + current_padded;
        let new_hdu_end = data_offset + new_padded;

        // Phase 1: verify this is the last HDU on disk, and extend the file
        // (sparse zero-fill via set_len).  File is now larger than the header
        // advertises — readers tolerate trailing junk after the last HDU.
        {
            let mut guard = lock_file(&super_.file)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;

            let file_len = file.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len();
            if file_len != current_hdu_end {
                return Err(PyValueError::new_err(format!(
                    "cannot extend HDU #{}: file_size ({}) != HDU end ({}); \
                     extending non-last HDUs is not yet supported",
                    super_.index, file_len, current_hdu_end
                )));
            }

            if new_hdu_end > current_hdu_end {
                file.set_len(new_hdu_end)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
        }

        // Phase 2: update the in-memory NAXISn card under the lock.
        let naxisn_key = format!("NAXIS{}", naxis);
        let new_card = card_int(
            &naxisn_key,
            new_hdu_shape[0] as i64,
            &format!("length of data axis {}", naxis),
        );
        let updated_header: Vec<String> = {
            let mut g = super_.header.lock()
                .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
            let card_idx = g.iter()
                .position(|c| c.len() >= 8 && c[..8].trim() == naxisn_key)
                .ok_or_else(|| PyValueError::new_err(
                    format!("header missing {}", naxisn_key)
                ))?;
            g[card_idx] = new_card.trim_end().to_string();
            g.clone()
        };

        // Phase 3: rewrite the on-disk header block(s) so they match the new
        // NAXISn value.  After this, file and header are consistent.
        {
            let mut guard = lock_file(&super_.file)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;

            let header_bytes = serialize_header_to_disk_bytes(&updated_header);
            let header_offset = data_offset - header_bytes.len() as u64;
            file.seek(SeekFrom::Start(header_offset))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            file.write_all(&header_bytes)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            file.flush()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }

        // Phase 4: write the data itself.  write_image_data re-validates
        // against the updated header (cheap) and uses the same byte-order /
        // buffer-protocol fast path as a normal write.
        write_image_data(
            &updated_header,
            data_offset,
            &super_.file,
            data,
            Some(start_for_write),
        )
    }

    // Numpy-style indexing over the HDU data.  Accepts int, slice (with
    // positive step), Ellipsis, or a tuple of those.  Integer-indexed axes
    // are removed from the output shape; slice-indexed axes are preserved
    // with their selected length.  Reads only the requested bytes from disk
    // (one I/O per outer-position; fully-covered fast axes coalesce).
    fn __getitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        let (_bitpix, hdu_shape) = parse_image_hdu_shape(&header_cards)?;
        let slices = normalize_slice_key(key, &hdu_shape)?;
        read_image_slice(py, &header_cards, super_.data_offset, &super_.file, &slices)
    }
}

// Allocate a numpy array sized + typed to this HDU and fill it with the
// HDU's data from disk.  Shared by ImageHDU::read.  Steps:
//   - parse BITPIX + shape from the header
//   - allocate via `numpy.empty(shape, native_dtype)` (no zero-fill cost)
//   - acquire a writable C-contiguous buffer view into the array's memory
//   - read the entire data section directly into that buffer
//   - byte-swap in place if host endian != big-endian (FITS on-disk order)
// The result is a native-endian array, ready for downstream numpy ops.
fn read_image_data(
    py: Python<'_>,
    header: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
) -> PyResult<Py<PyAny>> {
    let (bitpix, hdu_shape) = parse_image_hdu_shape(header)?;
    let bpp = (bitpix.abs() / 8) as u64;
    let total_pixels: u64 = hdu_shape.iter().product();
    let total_bytes = (total_pixels * bpp) as usize;

    let dtype_str = bitpix_to_native_dtype(bitpix)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("empty", (hdu_shape.clone(), dtype_str))?;

    // Fill the array from disk via the buffer protocol.  Scope the buffer
    // so PyBuffer_Release fires before we return the array to Python.
    {
        let mut buffer = RawBuffer::acquire_writable(&arr)?;
        if buffer.len() != total_bytes {
            return Err(PyValueError::new_err(format!(
                "allocated buffer length ({}) does not match expected ({})",
                buffer.len(), total_bytes
            )));
        }

        if total_bytes > 0 {
            let mut guard = lock_file(file_handle)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            file.seek(SeekFrom::Start(data_offset))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            file.read_exact(buffer.as_mut_slice())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }

        // FITS stores big-endian; numpy.empty gave us native.  Swap when
        // the host is little-endian and we have multi-byte elements.
        if bpp > 1 && !cfg!(target_endian = "big") {
            byteswap_in_place(buffer.as_mut_slice(), bpp as usize);
        }
    }

    Ok(arr.unbind())
}

// ====================== Slicing for image reads ======================

// One axis of a normalized index.  `start`/`step`/`count` describe the source
// range in HDU coordinates (numpy order — same convention as `hdu_shape`).
// `is_int` distinguishes integer indexing (which consumes the axis from the
// output shape) from slice indexing (which preserves it).
#[derive(Debug, Clone)]
struct AxisSlice {
    start: u64,
    step: u64,
    count: u64,
    is_int: bool,
}

fn full_axis_slice(dim: u64) -> AxisSlice {
    AxisSlice { start: 0, step: 1, count: dim, is_int: false }
}

// Parse a single axis indexer (int or slice) given the corresponding HDU
// dimension.  Ellipsis is handled by the caller (it expands to multiple
// full-axis slices).  Negative ints are normalized; negative or zero steps
// are rejected.
fn parse_axis_indexer(item: &Bound<'_, PyAny>, dim: u64) -> PyResult<AxisSlice> {
    if let Ok(slice) = item.cast::<PySlice>() {
        let indices = slice.indices(dim as isize)?;
        if indices.step <= 0 {
            return Err(PyValueError::new_err(
                "negative or zero step is not supported"
            ));
        }
        // For positive step, `start` is non-negative per the docs.
        let start = indices.start.max(0) as u64;
        let step = indices.step as u64;
        let count = indices.slicelength as u64;
        Ok(AxisSlice { start, step, count, is_int: false })
    } else if let Ok(i) = item.extract::<i64>() {
        let dim_i = dim as i64;
        let normalized = if i < 0 { dim_i + i } else { i };
        if normalized < 0 || normalized >= dim_i {
            return Err(PyIndexError::new_err(format!(
                "index {} out of bounds for axis of size {}", i, dim
            )));
        }
        Ok(AxisSlice {
            start: normalized as u64,
            step: 1,
            count: 1,
            is_int: true,
        })
    } else {
        Err(PyValueError::new_err(
            "unsupported index type (expected int, slice, or Ellipsis)",
        ))
    }
}

// Normalize a Python __getitem__ key into a per-axis list (length == naxis).
// Accepts a single int/slice/Ellipsis or a tuple of these.  Ellipsis (at most
// one) is expanded to enough full-axis slices to bring the total to naxis.
// Trailing missing axes are filled with full-axis slices, matching numpy
// semantics.
fn normalize_slice_key(
    key: &Bound<'_, PyAny>,
    hdu_shape: &[u64],
) -> PyResult<Vec<AxisSlice>> {
    let naxis = hdu_shape.len();

    let items: Vec<Bound<PyAny>> = if let Ok(tup) = key.cast::<PyTuple>() {
        tup.iter().collect()
    } else {
        vec![key.clone()]
    };

    let mut ellipsis_pos: Option<usize> = None;
    for (i, item) in items.iter().enumerate() {
        if item.is_instance_of::<PyEllipsis>() {
            if ellipsis_pos.is_some() {
                return Err(PyValueError::new_err(
                    "an index can only have a single ellipsis",
                ));
            }
            ellipsis_pos = Some(i);
        }
    }

    let n_explicit = items.len() - usize::from(ellipsis_pos.is_some());
    if n_explicit > naxis {
        return Err(PyValueError::new_err(format!(
            "too many indices for array: HDU has {} axes, got {} explicit",
            naxis, n_explicit
        )));
    }
    let n_ellipsis_fill = naxis - n_explicit;

    let mut out: Vec<AxisSlice> = Vec::with_capacity(naxis);
    let mut axis = 0usize;
    for (i, item) in items.iter().enumerate() {
        if Some(i) == ellipsis_pos {
            for _ in 0..n_ellipsis_fill {
                out.push(full_axis_slice(hdu_shape[axis]));
                axis += 1;
            }
        } else {
            out.push(parse_axis_indexer(item, hdu_shape[axis])?);
            axis += 1;
        }
    }
    while axis < naxis {
        out.push(full_axis_slice(hdu_shape[axis]));
        axis += 1;
    }

    Ok(out)
}

// Strided-read variant of compute_strip_layout.  An axis can be folded into
// the contiguous file strip only if its step is 1; any axis with step != 1
// becomes an outer iteration axis.  The slowest axis that's still in the
// strip may be partial, but partial coverage stops further coalescing.
// Walks from the fastest axis (numpy axis n-1, i.e. FITS NAXIS1) inward,
// which matches the on-disk byte order.
fn compute_read_strip_layout(
    hdu_shape: &[u64],
    slices: &[AxisSlice],
) -> (usize, u64) {
    let n = hdu_shape.len();
    let mut strip_pixels: u64 = 1;
    for axis in (0..n).rev() {
        if slices[axis].step != 1 {
            // Can't include this axis in the strip — it becomes outer.
            return (axis + 1, strip_pixels);
        }
        strip_pixels *= slices[axis].count;
        if slices[axis].start != 0 || slices[axis].count != hdu_shape[axis] {
            // Partial coverage: include this axis but stop coalescing.
            return (axis, strip_pixels);
        }
    }
    (0, strip_pixels)
}

// Read a sub-region of an image HDU and return a freshly-allocated,
// native-endian numpy array.  Axes consumed by integer indexing are dropped
// from the output shape.  Coalesces fully-covered fast axes with step==1
// into one contiguous read per strip; falls back to a seek+read per outer
// position otherwise.  A full read (all axes full, all step==1) collapses
// to a single big read.
fn read_image_slice(
    py: Python<'_>,
    header: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    slices: &[AxisSlice],
) -> PyResult<Py<PyAny>> {
    let (bitpix, hdu_shape) = parse_image_hdu_shape(header)?;
    let naxis = hdu_shape.len();
    if slices.len() != naxis {
        return Err(PyValueError::new_err(format!(
            "internal error: {} slices for {} axes", slices.len(), naxis
        )));
    }

    let output_shape: Vec<u64> = slices.iter()
        .filter(|s| !s.is_int)
        .map(|s| s.count)
        .collect();

    let bpp = (bitpix.abs() / 8) as u64;
    let dtype_str = bitpix_to_native_dtype(bitpix)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("empty", (output_shape.clone(), dtype_str))?;

    let total_pixels: u64 = slices.iter().map(|s| s.count).product();
    if total_pixels == 0 {
        return Ok(arr.unbind());
    }

    let hdu_strides = row_major_strides(&hdu_shape);
    let (outer_axes, strip_pixels) = compute_read_strip_layout(&hdu_shape, slices);
    let strip_bytes = (strip_pixels * bpp) as usize;

    let outer_count: u64 = slices[..outer_axes].iter().map(|s| s.count).product();

    // Fixed-position contribution from axes inside the strip — their `start`
    // values are constant across the outer iteration.
    let inner_start_pixels: u64 = (outer_axes..naxis)
        .map(|k| slices[k].start * hdu_strides[k])
        .sum();

    let mut buffer = RawBuffer::acquire_writable(&arr)?;
    let expected_buffer_len = (output_shape.iter().product::<u64>() * bpp) as usize;
    if buffer.len() != expected_buffer_len {
        return Err(PyValueError::new_err(format!(
            "allocated buffer length ({}) does not match expected ({})",
            buffer.len(), expected_buffer_len
        )));
    }

    {
        let mut guard = lock_file(file_handle)?;
        let file = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;

        let mut output_offset: usize = 0;
        let mut iter_idx = vec![0u64; outer_axes];

        for _ in 0..outer_count {
            let mut src_pixel: u64 = inner_start_pixels;
            for k in 0..outer_axes {
                let src_axis_idx = slices[k].start + iter_idx[k] * slices[k].step;
                src_pixel += src_axis_idx * hdu_strides[k];
            }
            let file_pos = data_offset + src_pixel * bpp;

            file.seek(SeekFrom::Start(file_pos))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            let dest = &mut buffer.as_mut_slice()[output_offset..output_offset + strip_bytes];
            file.read_exact(dest)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            output_offset += strip_bytes;

            // Row-major increment over outer axes (axis 0 slowest, outer_axes-1 fastest).
            for axis in (0..outer_axes).rev() {
                iter_idx[axis] += 1;
                if iter_idx[axis] < slices[axis].count {
                    break;
                }
                iter_idx[axis] = 0;
            }
        }
    }

    if bpp > 1 && !cfg!(target_endian = "big") {
        byteswap_in_place(buffer.as_mut_slice(), bpp as usize);
    }
    drop(buffer);

    Ok(arr.unbind())
}

// Validate and write a numpy array into an image HDU's data section.
// Shared by ImageHDU::write and ImageHDU::extend.  Internally:
//   - validates dtype matches BITPIX, and shape fits inside the HDU
//   - acquires data via the Python buffer protocol (C-contiguous required)
//   - detects byte order from numpy.dtype.str; if already big-endian, writes
//     directly from the numpy buffer (zero copy), otherwise copies one strip
//     at a time into a scratch buffer and byte-swaps there
//   - coalesces fully-aligned axes from the fast end so the strip is as
//     large as possible (a full overwrite collapses to a single write)
fn write_image_data(
    header: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    data: &Bound<'_, PyAny>,
    start: Option<Vec<i64>>,
) -> PyResult<()> {
    let (bitpix, hdu_shape) = parse_image_hdu_shape(header)?;
    let naxis = hdu_shape.len();

    // Validate dtype against BITPIX via numpy attributes.
    let dtype = data.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize_attr: u64 = dtype.getattr("itemsize")?.extract()?;
    let (expected_kind, expected_size) = bitpix_to_numpy_kind(bitpix)?;
    if kind != expected_kind || itemsize_attr != expected_size {
        return Err(PyValueError::new_err(format!(
            "data dtype ({}{}) does not match HDU BITPIX={} (expected {}{})",
            kind, itemsize_attr, bitpix, expected_kind, expected_size,
        )));
    }

    let data_shape: Vec<u64> = data.getattr("shape")?.extract()?;
    if data_shape.len() != naxis {
        return Err(PyValueError::new_err(format!(
            "data has {} axes, HDU has {}", data_shape.len(), naxis
        )));
    }

    // start: default to origin (numpy order).
    let start_vec: Vec<u64> = match start {
        Some(s) => {
            if s.len() != naxis {
                return Err(PyValueError::new_err(format!(
                    "start has {} components, expected {}", s.len(), naxis
                )));
            }
            let mut out = Vec::with_capacity(naxis);
            for (i, &v) in s.iter().enumerate() {
                if v < 0 {
                    return Err(PyValueError::new_err(format!(
                        "start[{}] must be >= 0, got {}", i, v
                    )));
                }
                out.push(v as u64);
            }
            out
        }
        None => vec![0u64; naxis],
    };

    // Bounds: start + data_shape <= hdu_shape, per axis.
    for i in 0..naxis {
        if start_vec[i] + data_shape[i] > hdu_shape[i] {
            return Err(PyValueError::new_err(format!(
                "axis {}: start ({}) + data shape ({}) exceeds HDU dim ({})",
                i, start_vec[i], data_shape[i], hdu_shape[i]
            )));
        }
    }

    let total_pixels: u64 = data_shape.iter().product();
    if total_pixels == 0 {
        return Ok(());
    }

    // Byte-order detection from numpy's canonical typestring (uses one
    // of '<', '>', '|'; '|' is single-byte where order is moot).
    let dtype_str: String = dtype.getattr("str")?.extract()?;
    let needs_swap = if expected_size == 1 {
        false
    } else {
        match dtype_str.chars().next() {
            Some('>') | Some('|') => false,
            Some('<') => true,
            _ => return Err(PyValueError::new_err(format!(
                "unrecognized dtype byteorder in '{}'", dtype_str
            ))),
        }
    };

    // Acquire raw bytes via the buffer protocol.  PyBUF_C_CONTIGUOUS will
    // cause numpy to fail for non-contiguous input; surface that with a
    // hint about np.ascontiguousarray.
    let buffer = RawBuffer::acquire(data).map_err(|e| {
        PyValueError::new_err(format!(
            "data must be a C-contiguous numpy array \
             (try np.ascontiguousarray): {}", e
        ))
    })?;
    if buffer.itemsize() as u64 != expected_size {
        return Err(PyValueError::new_err(
            "buffer itemsize disagrees with dtype.itemsize",
        ));
    }
    let bpp = expected_size;
    if buffer.len() as u64 != total_pixels * bpp {
        return Err(PyValueError::new_err(
            "buffer length disagrees with data shape",
        ));
    }
    let data_bytes = buffer.as_slice();

    let (outer_axes, strip_pixels) =
        compute_strip_layout(&hdu_shape, &data_shape, &start_vec);
    let strip_bytes = (strip_pixels * bpp) as usize;
    let hdu_strides = row_major_strides(&hdu_shape);
    let src_strides = row_major_strides(&data_shape);

    // Base file offset (in pixels) from the sub-region's leading corner.
    // For a full write this is 0; for sub-regions it folds in start[].
    let base_pixel: u64 = (0..naxis)
        .map(|k| start_vec[k] * hdu_strides[k])
        .sum();

    // Scratch buffer for byte-swapping, sized for one strip.  Empty when
    // no swap is needed (zero-copy path).
    let mut scratch: Vec<u8> = if needs_swap {
        vec![0u8; strip_bytes]
    } else {
        Vec::new()
    };

    let outer_count: u64 = data_shape[..outer_axes].iter().product();
    let mut idx = vec![0u64; outer_axes];

    let mut guard = lock_file(file_handle)?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    for _ in 0..outer_count {
        let mut hdu_pixel = base_pixel;
        let mut src_pixel: u64 = 0;
        for axis in 0..outer_axes {
            hdu_pixel += idx[axis] * hdu_strides[axis];
            src_pixel += idx[axis] * src_strides[axis];
        }
        let file_pos = data_offset + hdu_pixel * bpp;
        let src_byte = (src_pixel * bpp) as usize;
        let src = &data_bytes[src_byte..src_byte + strip_bytes];

        file.seek(SeekFrom::Start(file_pos))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if needs_swap {
            scratch.copy_from_slice(src);
            byteswap_in_place(&mut scratch, bpp as usize);
            file.write_all(&scratch)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        } else {
            file.write_all(src)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }

        // Row-major increment over the outer axes (slowest at idx 0,
        // fastest at idx outer_axes-1).
        for axis in (0..outer_axes).rev() {
            idx[axis] += 1;
            if idx[axis] < data_shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }

    file.flush()
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    Ok(())
}

#[pyclass(extends = HDU)]
struct TableHDU; // Binary table (BINTABLE)

impl TableHDU {
    fn new(
        header: Vec<String>,
        index: usize,
        header_offset: u64,
        data_offset: u64,
        file: FileHandle,
        tainted: TaintFlag,
    ) -> (Self, HDU) {
        (
            TableHDU,
            HDU::new(header, index, header_offset, data_offset, file, tainted),
        )
    }
}

#[pymethods]
impl TableHDU {
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let index: usize = super_.index();
        Ok(format!("<TableHDU (binary) #{}>", index))
    }
}

#[pyclass(extends = HDU)]
struct AsciiTableHDU; // ASCII table (TABLE)

impl AsciiTableHDU {
    fn new(
        header: Vec<String>,
        index: usize,
        header_offset: u64,
        data_offset: u64,
        file: FileHandle,
        tainted: TaintFlag,
    ) -> (Self, HDU) {
        (
            AsciiTableHDU,
            HDU::new(header, index, header_offset, data_offset, file, tainted),
        )
    }
}

#[pymethods]
impl AsciiTableHDU {
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let index: usize = super_.index();
        Ok(format!("<AsciiTableHDU #{}>", index))
    }
}

// ====================== Parse all HDUs from an open file ======================
// Walks the file from byte 0, extracting every HDU header and skipping over
// each data section, returning the parsed HDU Python objects.  Each HDU is
// constructed with its data-section byte offset and a clone of the file
// handle so that write-back methods can locate themselves on disk.
fn parse_hdus_from_file(
    py: Python<'_>,
    handle: &FileHandle,
    tainted: &TaintFlag,
) -> PyResult<Vec<Py<PyAny>>> {
    let mut guard = lock_file(handle)?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    file.seek(SeekFrom::Start(0))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    let mut hdus: Vec<Py<PyAny>> = Vec::new();
    let mut offset = 0u64;

    loop {
        let mut header_cards: Vec<String> = Vec::new();
        let mut end_found = false;

        while !end_found {
            let mut block = vec![0u8; BLOCK_SIZE];
            match file.read_exact(&mut block) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    if header_cards.is_empty() {
                        break;
                    } else {
                        return Err(PyIOError::new_err("truncated FITS file"));
                    }
                }
                Err(e) => return Err(PyIOError::new_err(e.to_string())),
            }

            // FITS header bytes are restricted to printable ASCII (0x20-0x7E).
            for (j, &b) in block.iter().enumerate() {
                if !(0x20..=0x7E).contains(&b) {
                    return Err(PyValueError::new_err(format!(
                        "non-printable byte 0x{:02X} in header block at byte offset {}",
                        b, offset + j as u64
                    )));
                }
            }

            for i in (0..BLOCK_SIZE).step_by(CARD_SIZE) {
                // Safe: bytes have just been validated as printable ASCII.
                let card = std::str::from_utf8(&block[i..i + CARD_SIZE])
                    .unwrap()
                    .trim_end()
                    .to_string();
                header_cards.push(card.clone());
                if card == "END" {
                    end_found = true;
                    break;
                }
            }
        }

        if header_cards.is_empty() {
            break;
        }

        validate_header(&header_cards, hdus.is_empty())?;

        let num_header_blocks = ((header_cards.len() + 35) / 36) as u64;
        let header_size = num_header_blocks * BLOCK_SIZE as u64;
        let header_offset = offset;
        let data_offset = offset + header_size;
        let data_size = calculate_data_size(&header_cards);

        let is_image = header_cards.iter().any(|c| {
            c.starts_with("SIMPLE  =") || c.starts_with("XTENSION= 'IMAGE")
        });
        let is_binary_table = header_cards.iter().any(|c| c.starts_with("XTENSION= 'BINTABLE'"));
        let is_ascii_table = header_cards.iter().any(|c| c.starts_with("XTENSION= 'TABLE'"));

        let hdu_file = Arc::clone(handle);
        let hdu_taint = Arc::clone(tainted);
        let hdu_py: Py<PyAny> = if is_image {
            Py::new(py, ImageHDU::new(
                header_cards.clone(), hdus.len(),
                header_offset, data_offset, hdu_file, hdu_taint,
            ))?.into()
        } else if is_binary_table {
            Py::new(py, TableHDU::new(
                header_cards.clone(), hdus.len(),
                header_offset, data_offset, hdu_file, hdu_taint,
            ))?.into()
        } else if is_ascii_table {
            Py::new(py, AsciiTableHDU::new(
                header_cards.clone(), hdus.len(),
                header_offset, data_offset, hdu_file, hdu_taint,
            ))?.into()
        } else {
            let h = HDU::new(
                header_cards.clone(), hdus.len(),
                header_offset, data_offset, hdu_file, hdu_taint,
            );
            Py::new(py, h)?.into()
        };

        hdus.push(hdu_py);

        offset += header_size + data_size;
        let _ = file.seek(SeekFrom::Start(offset));

        if offset >= file.metadata().map(|m| m.len()).unwrap_or(0) {
            break;
        }
    }

    let _ = file.seek(SeekFrom::Start(0));
    Ok(hdus)
}

// ====================== Card-formatting helpers (for write) ======================
fn pad_to_card(s: &str) -> String {
    let mut out = s.to_string();
    if out.len() < CARD_SIZE {
        out.push_str(&" ".repeat(CARD_SIZE - out.len()));
    } else if out.len() > CARD_SIZE {
        out.truncate(CARD_SIZE);
    }
    out
}

fn card_int(key: &str, value: i64, comment: &str) -> String {
    let head = format!("{:<8}= {:>20}", key, value);
    let body = if comment.is_empty() {
        head
    } else {
        format!("{} / {}", head, comment)
    };
    pad_to_card(&body)
}

fn card_logical(key: &str, value: bool, comment: &str) -> String {
    let v = if value { "T" } else { "F" };
    let head = format!("{:<8}= {:>20}", key, v);
    let body = if comment.is_empty() {
        head
    } else {
        format!("{} / {}", head, comment)
    };
    pad_to_card(&body)
}

fn card_string(key: &str, value: &str, comment: &str) -> String {
    // FITS values of string type require a minimum length of 8 characters
    // inside the quotes; embedded single quotes are doubled.
    let escaped = value.replace('\'', "''");
    let padded = if escaped.len() < 8 {
        format!("{:<8}", escaped)
    } else {
        escaped
    };
    let quoted = format!("'{}'", padded);
    let head = format!("{:<8}= {}", key, quoted);
    let body = if comment.is_empty() {
        head
    } else {
        format!("{} / {}", head, comment)
    };
    pad_to_card(&body)
}

// Format a float for the FITS value field, guaranteeing the result is
// distinguishable from an integer (always contains either '.' or 'e'/'E').
// Uses Rust's default Display which is round-trip safe for finite f64s.
fn format_fits_float(value: f64) -> String {
    if !value.is_finite() {
        // NaN and infinity aren't in the FITS fixed-format float spec, but
        // we don't reject them here — caller can validate if it cares.
        return format!("{}", value);
    }
    let s = format!("{}", value);
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{}.0", s)
    }
}

fn card_float(key: &str, value: f64, comment: &str) -> String {
    let formatted = format_fits_float(value);
    let head = format!("{:<8}= {:>20}", key, formatted);
    let body = if comment.is_empty() {
        head
    } else {
        format!("{} / {}", head, comment)
    };
    pad_to_card(&body)
}

// FITS complex literal: `(real, imag)`.  Components are formatted the same
// way as standalone floats (so `D` exponent and special values like nan/inf
// are handled identically).  Errors if the resulting card exceeds 80 chars
// — complex literals can be wide and silent truncation would corrupt the
// imaginary half.
fn card_complex(key: &str, real: f64, imag: f64, comment: &str) -> PyResult<String> {
    let r = format_fits_float(real);
    let i = format_fits_float(imag);
    let value_str = format!("({}, {})", r, i);
    let head = format!("{:<8}= {}", key, value_str);
    let body = if comment.is_empty() {
        head
    } else {
        format!("{} / {}", head, comment)
    };
    if body.len() > CARD_SIZE {
        return Err(PyValueError::new_err(format!(
            "complex card too long ({} chars) for key '{}'; \
             shorten the comment", body.len(), key
        )));
    }
    Ok(pad_to_card(&body))
}

// Build the card sequence for a string value, applying the FITS Long Strings
// CONTINUE convention when necessary.  Most strings produce a single card;
// values longer than what fits in one card produce a chain of cards each
// with payload up to 67 chars (with `&` continuation marker) plus a final
// card without `&`.  Comments are placed on the final card; if the comment
// is too long to fit there, an error is raised (no truncation).
fn build_string_value_cards(key: &str, value: &str, comment: &str) -> PyResult<Vec<String>> {
    let escaped = value.replace('\'', "''");

    // How many payload chars fit on the LAST card (which has no `&`).
    // Layout: prefix(10) + `'` + payload + `'` [+ ` / ` + comment] ≤ 80.
    //   - prefix is 10 chars for both first ("KEYWORD = ") and CONTINUE
    //     ("CONTINUE  ").
    //   - Without comment: payload ≤ 68.
    //   - With comment of length C: payload ≤ 65 − C (i.e. 80 − 12 − 3 − C).
    let max_last_payload: usize = if comment.is_empty() {
        68
    } else {
        if comment.len() >= 65 {
            return Err(PyValueError::new_err(format!(
                "comment is too long ({} chars) to fit in a FITS card alongside a string value; \
                 max comment length is 64 chars for CONTINUE-chained values",
                comment.len()
            )));
        }
        65 - comment.len()
    };

    // Single-card fast path: the whole (escaped) value fits along with the
    // comment.  Delegate to the existing single-card builder.
    if escaped.len() <= max_last_payload {
        return Ok(vec![card_string(key, value, comment)]);
    }

    // Multi-card CONTINUE path.  Iterate the escaped bytes, emitting chunks
    // of up to 67 chars on non-last cards and up to max_last_payload on the
    // last card.  Avoid splitting a `''` quote-escape across cards (the
    // boundary check below).
    let bytes = escaped.as_bytes();
    let total = bytes.len();
    let mut cards: Vec<String> = Vec::new();
    let mut pos = 0;

    while pos < total {
        let remaining = total - pos;
        let mut take;
        let is_last;
        if remaining <= max_last_payload {
            take = remaining;
            is_last = true;
        } else {
            // Leave at least `max_last_payload` bytes for the final chunk so
            // it can fit the comment.
            take = (remaining - max_last_payload).min(67);
            is_last = false;

            // Avoid splitting a `''` escape: if the chunk would end at the
            // first `'` of a pair, back off by one.
            while take > 0 && pos + take < total {
                if bytes[pos + take - 1] == b'\'' && bytes[pos + take] == b'\'' {
                    take -= 1;
                } else {
                    break;
                }
            }
            // Corner: if backoff drove `take` to 0 (escaped value begins
            // with a quote pair), include the whole pair.  Inner-string
            // length stays well below 67 in that case.
            if take == 0 {
                take = 2;
            }
        }

        let inner = std::str::from_utf8(&bytes[pos..pos + take]).unwrap();
        let is_first = cards.is_empty();

        let card = if is_first {
            // First card: KEYWORD = '<inner>&' (non-last branch only, since
            // the single-card path is handled by the early return above).
            format!("{:<8}= '{}&'", key, inner)
        } else if is_last {
            if comment.is_empty() {
                format!("CONTINUE  '{}'", inner)
            } else {
                format!("CONTINUE  '{}' / {}", inner, comment)
            }
        } else {
            format!("CONTINUE  '{}&'", inner)
        };
        cards.push(pad_to_card(&card));
        pos += take;
    }

    Ok(cards)
}

// ====================== Helpers for writing image data ======================

// Inverse of dtype_to_bitpix: the numpy (dtype.kind, dtype.itemsize) tuple
// expected for a given BITPIX.  Used to validate input arrays.
fn bitpix_to_numpy_kind(bitpix: i32) -> PyResult<(&'static str, u64)> {
    match bitpix {
        8   => Ok(("u", 1)),
        16  => Ok(("i", 2)),
        32  => Ok(("i", 4)),
        64  => Ok(("i", 8)),
        -32 => Ok(("f", 4)),
        -64 => Ok(("f", 8)),
        _   => Err(PyValueError::new_err(format!("unsupported BITPIX {}", bitpix))),
    }
}

// Native-endian numpy dtype short-code for a FITS BITPIX.  Used by the
// read path: `numpy.empty(shape, dtype)` produces a native-endian array,
// which is what downstream user code expects.  The on-disk big-endian
// bytes are byte-swapped into native form after read.
fn bitpix_to_native_dtype(bitpix: i32) -> PyResult<&'static str> {
    match bitpix {
        8   => Ok("u1"),
        16  => Ok("i2"),
        32  => Ok("i4"),
        64  => Ok("i8"),
        -32 => Ok("f4"),
        -64 => Ok("f8"),
        _   => Err(PyValueError::new_err(format!("unsupported BITPIX {}", bitpix))),
    }
}

// Row-major (numpy / C-order) strides in pixels.  stride[k] = product of
// shape[k+1..].  For an empty shape this returns an empty vec.
fn row_major_strides(shape: &[u64]) -> Vec<u64> {
    let n = shape.len();
    if n == 0 {
        return Vec::new();
    }
    let mut strides = vec![1u64; n];
    for k in (0..n - 1).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

// Plan a strided write of a sub-region into an HDU.  Returns (outer_axes,
// strip_pixels) where:
//   - strip_pixels: number of contiguous pixels written per strip.  Coalesces
//     adjacent fully-covered axes (start=0 and data dim = HDU dim) from the
//     fast (last) end inward.  The first axis that's partial is included in
//     the strip and stops further coalescing.
//   - outer_axes: number of "slower" axes to iterate over (axes 0..outer_axes).
//     For a full write, outer_axes=0 and strip_pixels = total pixel count, so
//     the iteration runs exactly once with one giant strip.
fn compute_strip_layout(
    hdu_shape: &[u64],
    data_shape: &[u64],
    start: &[u64],
) -> (usize, u64) {
    let n = hdu_shape.len();
    let mut strip_pixels: u64 = 1;
    for axis in (0..n).rev() {
        strip_pixels *= data_shape[axis];
        if start[axis] != 0 || data_shape[axis] != hdu_shape[axis] {
            return (axis, strip_pixels);
        }
    }
    (0, strip_pixels)
}

// Parse BITPIX, NAXIS, and NAXIS1..NAXISn out of an image HDU header.
// Returns (bitpix, hdu_shape_in_numpy_order).  Errors if NAXIS=0 (no data
// section) or any required keyword is missing.
fn parse_image_hdu_shape(header: &[String]) -> PyResult<(i32, Vec<u64>)> {
    let bitpix = parse_keyword(header, "BITPIX")
        .ok_or_else(|| PyValueError::new_err("HDU header missing BITPIX"))?
        as i32;
    let naxis = parse_keyword(header, "NAXIS")
        .ok_or_else(|| PyValueError::new_err("HDU header missing NAXIS"))?
        as usize;
    if naxis == 0 {
        return Err(PyValueError::new_err(
            "HDU has NAXIS=0 (no data section)",
        ));
    }
    let mut fits_dims: Vec<u64> = Vec::with_capacity(naxis);
    for i in 1..=naxis {
        let d = parse_keyword(header, &format!("NAXIS{}", i))
            .ok_or_else(|| PyValueError::new_err(
                format!("HDU header missing NAXIS{}", i)
            ))?;
        if d < 0 {
            return Err(PyValueError::new_err(
                format!("NAXIS{} is negative", i)
            ));
        }
        fits_dims.push(d as u64);
    }
    let hdu_shape: Vec<u64> = fits_dims.iter().rev().copied().collect();
    Ok((bitpix, hdu_shape))
}

// Round `n` up to the next 2880-byte FITS block boundary.
fn round_up_to_block(n: u64) -> u64 {
    let block = BLOCK_SIZE as u64;
    ((n + block - 1) / block) * block
}

// Serialize a Vec<String> header (cards stored with trailing whitespace
// trimmed, as `parse_hdus_from_file` stores them) back to its on-disk byte
// representation: each card padded to 80, the whole sequence padded with
// spaces to a multiple of 2880.
fn serialize_header_to_disk_bytes(header: &[String]) -> Vec<u8> {
    let num_blocks = (header.len() + 35) / 36;
    let total_size = num_blocks * BLOCK_SIZE;
    let mut out = Vec::with_capacity(total_size);
    for card in header {
        let mut padded = card.clone();
        if padded.len() < CARD_SIZE {
            padded.push_str(&" ".repeat(CARD_SIZE - padded.len()));
        } else if padded.len() > CARD_SIZE {
            padded.truncate(CARD_SIZE);
        }
        out.extend_from_slice(padded.as_bytes());
    }
    while out.len() < total_size {
        out.push(b' ');
    }
    out
}

// Reverse the bytes of every `itemsize`-byte element in `buf` (in place).
// itemsize=1 is a no-op.  Used to translate native little-endian to FITS
// big-endian on the write path.
fn byteswap_in_place(buf: &mut [u8], itemsize: usize) {
    if itemsize <= 1 {
        return;
    }
    for chunk in buf.chunks_exact_mut(itemsize) {
        chunk.reverse();
    }
}

// RAII wrapper around the Python buffer protocol.  Asks for a C-contiguous
// view via the C API (bypassing pyo3's PyBuffer<T> typed wrapper, which
// rejects non-native byte orders such as '>f8' for a PyBuffer<f64>).
// PyBuffer_Release is guaranteed to run on drop.  Box the Py_buffer so its
// address is stable across moves.
struct RawBuffer {
    view: Box<pyo3::ffi::Py_buffer>,
}

impl RawBuffer {
    fn acquire_with_flags(obj: &Bound<'_, PyAny>, flags: std::os::raw::c_int) -> PyResult<Self> {
        let mut view: Box<pyo3::ffi::Py_buffer> =
            Box::new(unsafe { std::mem::zeroed() });
        let rc = unsafe {
            pyo3::ffi::PyObject_GetBuffer(
                obj.as_ptr(),
                &mut *view as *mut _,
                flags,
            )
        };
        if rc != 0 {
            // Python set an exception during the failed GetBuffer call.
            return Err(PyErr::take(obj.py()).unwrap_or_else(|| {
                PyValueError::new_err("buffer acquisition failed")
            }));
        }
        Ok(RawBuffer { view })
    }

    // Read-only, C-contiguous.  Used by write paths.
    fn acquire(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::acquire_with_flags(obj, pyo3::ffi::PyBUF_C_CONTIGUOUS)
    }

    // Writable + C-contiguous.  Used by the read path to receive disk bytes
    // straight into a freshly-allocated numpy array's memory.
    fn acquire_writable(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::acquire_with_flags(
            obj,
            pyo3::ffi::PyBUF_C_CONTIGUOUS | pyo3::ffi::PyBUF_WRITABLE,
        )
    }

    fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.view.buf as *const u8,
                self.view.len as usize,
            )
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.view.buf as *mut u8,
                self.view.len as usize,
            )
        }
    }

    fn itemsize(&self) -> usize {
        self.view.itemsize as usize
    }

    fn len(&self) -> usize {
        self.view.len as usize
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        unsafe { pyo3::ffi::PyBuffer_Release(&mut *self.view) };
    }
}

// Map a numpy short-code or long-name dtype string to a FITS BITPIX value.
// Endianness prefixes (`<`, `>`, `|`, `=`) are stripped.  Only the dtypes
// directly representable in FITS without BZERO/BSCALE are supported; the
// unsigned wide ints (uint16/32/64) need scaling and will be added later.
fn dtype_to_bitpix(dtype: &str) -> PyResult<i32> {
    let s = dtype.trim_start_matches(|c| c == '<' || c == '>' || c == '|' || c == '=');
    let normalized = s.to_lowercase();
    match normalized.as_str() {
        "u1" | "uint8" => Ok(8),
        "i2" | "int16" => Ok(16),
        "i4" | "int32" => Ok(32),
        "i8" | "int64" => Ok(64),
        "f4" | "float32" => Ok(-32),
        "f8" | "float64" => Ok(-64),
        _ => Err(PyValueError::new_err(format!(
            "unsupported numpy dtype '{}'. Supported: 'u1','i2','i4','i8','f4','f8'",
            dtype
        ))),
    }
}

// ====================== Main FITS class ======================
#[pyclass]
struct FITS {
    filename: String,
    file: FileHandle,
    hdus: Vec<Py<PyAny>>,
    // Per-file taint flag (see TaintFlag).  Owned here; cloned into every
    // HDU and FITSHeader so a mid-write failure anywhere taints the lot.
    tainted: TaintFlag,
}

#[pymethods]
impl FITS {
    #[new]
    fn new(py: Python<'_>, filename: String, mode: String) -> PyResult<Self> {
        let mut options = OpenOptions::new();

        match mode.as_str() {
            "r"  => options.read(true),
            "r+" => options.read(true).write(true),
            "w+" => options.read(true).write(true).truncate(true).create(true),
            _ => return Err(PyIOError::new_err(format!(
                "Unsupported mode '{}'. Supported modes: 'r', 'r+', 'w+'",
                mode
            ))),
        };

        let file = options.open(&filename)
            .map_err(|e| PyIOError::new_err(format!("Failed to open '{}': {}", filename, e)))?;

        let handle: FileHandle = Arc::new(Mutex::new(Some(file)));
        let tainted: TaintFlag = Arc::new(AtomicBool::new(false));
        let hdus = parse_hdus_from_file(py, &handle, &tainted)?;

        Ok(FITS {
            filename,
            file: handle,
            hdus,
            tainted,
        })
    }

    #[getter]
    fn hdus(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::new(py, &self.hdus)?;
        Ok(list.unbind())
    }

    #[getter]
    fn filename(&self) -> String {
        self.filename.clone()
    }

    fn close(&mut self) -> PyResult<()> {
        let mut guard = lock_file(&self.file)?;
        if let Some(file) = guard.take() {
            let _ = file.sync_all();
        }
        Ok(())
    }

    #[getter]
    fn closed(&self) -> PyResult<bool> {
        let guard = lock_file(&self.file)?;
        Ok(guard.is_none())
    }

    fn __repr__(&self) -> String {
        let status = match self.file.lock() {
            Ok(guard) if guard.is_none() => "closed",
            Ok(_) => "open",
            Err(_) => "poisoned",
        };
        format!("FITS('{}', {} HDUs, {})", self.filename, self.hdus.len(), status)
    }

    fn __len__(&self) -> usize {
        self.hdus.len()
    }

    // Create a new image HDU.  `dtype` follows numpy short-code convention
    // (e.g. 'f8', 'i4').  `dims` is the array shape in numpy (row-major) order
    // and is reversed internally to produce FITS NAXISn (where NAXIS1 is the
    // fastest-varying axis).  The first HDU created becomes the primary HDU
    // (SIMPLE=T, EXTEND=T); subsequent calls produce 'IMAGE' extensions.  The
    // data section is allocated as zeros via sparse file extension.  After
    // writing, the new HDU is appended to `self.hdus` without re-reading.
    #[pyo3(signature = (dtype, dims, extname=None, extver=None))]
    fn create_image_hdu(
        &mut self,
        py: Python<'_>,
        dtype: String,
        dims: Vec<i64>,
        extname: Option<String>,
        extver: Option<i64>,
    ) -> PyResult<()> {
        for (i, &d) in dims.iter().enumerate() {
            if d <= 0 {
                return Err(PyValueError::new_err(format!(
                    "dimension {} must be > 0, got {}", i, d
                )));
            }
        }

        let bitpix = dtype_to_bitpix(&dtype)?;
        let naxis = dims.len() as i64;

        // numpy (row-major) -> FITS (NAXIS1 is fastest-varying): reverse.
        let fits_dims: Vec<i64> = dims.iter().rev().copied().collect();

        let is_primary = self.hdus.is_empty();

        let mut cards: Vec<String> = Vec::new();
        if is_primary {
            cards.push(card_logical("SIMPLE", true, "file conforms to FITS standard"));
        } else {
            cards.push(card_string("XTENSION", "IMAGE", "image extension"));
        }
        cards.push(card_int("BITPIX", bitpix as i64, "number of bits per data pixel"));
        cards.push(card_int("NAXIS", naxis, "number of data axes"));
        for (i, &d) in fits_dims.iter().enumerate() {
            cards.push(card_int(
                &format!("NAXIS{}", i + 1),
                d,
                &format!("length of data axis {}", i + 1),
            ));
        }
        if is_primary {
            cards.push(card_logical("EXTEND", true, "FITS dataset may contain extensions"));
        } else {
            cards.push(card_int("PCOUNT", 0, "required keyword; must = 0"));
            cards.push(card_int("GCOUNT", 1, "required keyword; must = 1"));
        }
        if let Some(name) = extname.as_deref() {
            cards.push(card_string("EXTNAME", name, "name of this HDU"));
        }
        if let Some(ver) = extver {
            cards.push(card_int("EXTVER", ver, "extension version"));
        }
        cards.push(pad_to_card("END"));

        // Pad header to a 2880-byte boundary.
        let header_bytes_len = cards.len() * CARD_SIZE;
        let pad_n = (BLOCK_SIZE - header_bytes_len % BLOCK_SIZE) % BLOCK_SIZE;

        // Data size (NAXIS=0 means no data unit).
        let bytes_per_pixel = (bitpix.abs() / 8) as u64;
        let mut product: u64 = 1;
        for &d in &fits_dims {
            product = product.saturating_mul(d as u64);
        }
        let data_size = if naxis == 0 { 0 } else { bytes_per_pixel * product };
        let data_padded = if data_size == 0 {
            0
        } else {
            ((data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64) * BLOCK_SIZE as u64
        };

        let (header_offset, data_offset) = {
            let mut guard = lock_file(&self.file)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;

            file.seek(SeekFrom::End(0))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            let header_start = file.stream_position()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            for c in &cards {
                file.write_all(c.as_bytes())
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
            if pad_n > 0 {
                let padding = vec![b' '; pad_n];
                file.write_all(&padding)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }

            let header_end = file.stream_position()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            if data_padded > 0 {
                let new_len = header_end + data_padded;
                file.set_len(new_len)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                file.seek(SeekFrom::Start(new_len))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }

            file.flush()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            (header_start, header_end)
        };

        // Construct the matching ImageHDU in memory and append.  The parser
        // stores cards with trailing whitespace trimmed; mirror that here so
        // the in-memory HDU is byte-equivalent to what a re-parse would yield.
        let stored_cards: Vec<String> = cards.iter()
            .map(|c| c.trim_end().to_string())
            .collect();
        let new_hdu: Py<PyAny> = Py::new(
            py,
            ImageHDU::new(
                stored_cards,
                self.hdus.len(),
                header_offset,
                data_offset,
                Arc::clone(&self.file),
                Arc::clone(&self.tainted),
            ),
        )?.into();
        self.hdus.push(new_hdu);

        Ok(())
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let len = self.hdus.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx < 0 || idx >= len {
            return Err(PyValueError::new_err(format!("HDU index {} out of range", index)));
        }
        Ok(self.hdus[idx as usize].clone_ref(py))
    }

    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let _ = self.close();
        Ok(false)
    }
}

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FITS>()?;
    m.add_class::<HDU>()?;
    m.add_class::<ImageHDU>()?;
    m.add_class::<TableHDU>()?;
    m.add_class::<AsciiTableHDU>()?;
    m.add_class::<FITSHeader>()?;
    m.add_class::<FITSHeaderEdit>()?;
    Ok(())
}
