// FITSHeader + FITSHeaderEdit + all card-level helpers (parsing, building,
// CONTINUE chains, HIERARCH, commentary, protected keys, batched update).
//
// Cards are the single source of truth — every value/comment/iteration
// access re-parses the relevant card(s).  Headers are tiny (tens to a few
// hundred cards) so parse-on-demand cost is invisible, and keeping no cache
// means mutation methods only have to rewrite cards without worrying about
// cache invalidation.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyDict, PyList, PyTuple};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::Bound;
use std::io::{Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;

use crate::common::{
    check_not_tainted, lock_file, FileHandle, TaintFlag, BLOCK_SIZE, CARD_SIZE,
};

// ===== card-level parse helpers =====

// `value_part` is the substring after `=` for a regular card, or the
// substring after the `CONTINUE` keyword (cols 9..) for a continuation card.
// Returns (trimmed raw value with quotes intact, trimmed comment).  A `/`
// that appears inside a quoted FITS string is not treated as a comment
// delimiter.  The single-quote toggle happens to also handle `''` escape
// correctly, because each escape contributes two toggles (net no change).
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

// `raw` is the trimmed value substring including outer single quotes (e.g.
// "'O''Brien   '").  Strips the outer quotes, converts `''` back to `'`,
// and drops trailing spaces (per the FITS standard, trailing spaces in a
// string value are not significant).  FITS values are restricted to
// printable ASCII, so byte-position slicing is safe.
fn extract_fits_string(raw: &str) -> String {
    let inner = &raw[1..raw.len() - 1];
    inner.replace("''", "'").trim_end().to_string()
}

// FITS permits Fortran-style `D` as the exponent indicator in addition to `E`.
fn parse_fits_float(s: &str) -> Option<f64> {
    if s.contains('D') || s.contains('d') {
        s.replace('D', "E").replace('d', "e").parse::<f64>().ok()
    } else {
        s.parse::<f64>().ok()
    }
}

// Returns Some((real, imag)) if `raw` looks like "(real, imag)".  Whitespace
// is allowed around the comma and inside the parentheses.  Each component
// may use either E or D exponent notation.
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

fn parse_header_dict(
    py: Python<'_>,
    cards: &[String],
) -> PyResult<Py<PyDict>> {
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

        let key_field: &str = if card.len() >= 8 { &card[..8] } else { card };
        let keyword_trimmed = key_field.trim();

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
                let text = if card.len() > 8 { card[8..].to_string() } else { String::new() };
                blank_list.append(text)?;
                has_blank = true;
                i += 1;
                continue;
            }
            "CONTINUE" => { i += 1; continue; }
            _ => {}
        }

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
                s.pop();
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
// Skips commentary keys.  Handles HIERARCH long keys.
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
// in `&`.
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

// ===== mutation helpers =====

fn normalize_keyword(key: &str) -> String {
    key.trim().to_ascii_uppercase()
}

fn is_hierarch_key(key: &str) -> bool {
    key.len() > 8 || key.contains(' ')
}

// Is this (post-normalization) key one that rustfits manages on the user's
// behalf — i.e., one whose value is determined by the file's structure,
// integrity contract, or compression layout, and which the user must NOT
// mutate directly?
//
// Categories: image-HDU structural, binary/ASCII table structural, random
// groups, tiled image compression, integrity (CHECKSUM/DATASUM).  Not
// protected: user metadata like OBJECT, EXPTIME, EXTNAME, BUNIT, BSCALE,
// BZERO, CTYPEn, CRVALn, etc.
fn is_protected_key(key: &str) -> bool {
    const LITERAL_PROTECTED: &[&str] = &[
        "SIMPLE", "XTENSION", "EXTEND", "BITPIX", "NAXIS",
        "PCOUNT", "GCOUNT", "END",
        "TFIELDS", "THEAP",
        "GROUPS",
        "ZIMAGE", "ZCMPTYPE", "ZBITPIX", "ZNAXIS",
        "ZSIMPLE", "ZEXTEND", "ZBLOCKED", "ZPCOUNT", "ZGCOUNT",
        "ZHECKSUM", "ZDATASUM", "ZTENSION",
        "ZQUANTIZ", "ZDITHER0", "ZMASKCMP", "ZBLANK",
        "CHECKSUM", "DATASUM",
    ];
    if LITERAL_PROTECTED.contains(&key) {
        return true;
    }
    const INDEXED_PREFIXES: &[&str] = &[
        "NAXIS",
        "TFORM", "TDIM", "TTYPE", "TSCAL", "TZERO", "TNULL", "TBCOL",
        "PTYPE", "PSCAL", "PZERO",
        "ZNAXIS", "ZTILE", "ZNAME", "ZVAL",
    ];
    for prefix in INDEXED_PREFIXES {
        if let Some(suffix) = key.strip_prefix(prefix) {
            if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}

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

fn extract_existing_comment(py: Python<'_>, cards: &[String], key: &str) -> Option<String> {
    let (idx, value_part) = find_card_for_key(cards, key)?;
    parse_value_with_continue(py, cards, idx, &value_part)
        .ok()
        .map(|(_, c)| c)
}

// Build the card(s) representing (key, value, comment).  Most types emit a
// single card; string values longer than what fits in one card emit a
// CONTINUE-chained sequence.  HIERARCH long keys use a different first-card
// layout.  Bool checked before int (Python bools are also ints).
fn build_card_from_value(
    key: &str,
    value: &Bound<'_, PyAny>,
    comment: &str,
) -> PyResult<Vec<String>> {
    if is_hierarch_key(key) {
        return build_hierarch_cards(key, value, comment);
    }
    if value.is_instance_of::<PyBool>() {
        let b: bool = value.extract()?;
        return Ok(vec![card_logical(key, b, comment)]);
    }
    if let Ok(n) = value.extract::<i64>() {
        return Ok(vec![card_int(key, n, comment)]);
    }
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

fn build_hierarch_cards(
    key: &str,
    value: &Bound<'_, PyAny>,
    comment: &str,
) -> PyResult<Vec<String>> {
    if value.is_instance_of::<PyBool>() {
        let b: bool = value.extract()?;
        return Ok(vec![assemble_hierarch_single(key, if b { "T" } else { "F" }, comment)?]);
    }
    if let Ok(n) = value.extract::<i64>() {
        let s = format!("{}", n);
        return Ok(vec![assemble_hierarch_single(key, &s, comment)?]);
    }
    if value.is_instance_of::<PyComplex>() {
        let c = value.cast::<PyComplex>()?;
        let s = format!(
            "({}, {})",
            format_fits_float(c.real()),
            format_fits_float(c.imag()),
        );
        return Ok(vec![assemble_hierarch_single(key, &s, comment)?]);
    }
    if let Ok(f) = value.extract::<f64>() {
        let s = format_fits_float(f);
        return Ok(vec![assemble_hierarch_single(key, &s, comment)?]);
    }
    if let Ok(s) = value.extract::<String>() {
        return build_hierarch_string_cards(key, &s, comment);
    }
    Err(PyValueError::new_err(format!(
        "unsupported value type for HIERARCH key '{}'; \
         supported: bool, int, float, complex, str", key
    )))
}

fn assemble_hierarch_single(
    key: &str,
    value_str: &str,
    comment: &str,
) -> PyResult<String> {
    let prefix = format!("HIERARCH {} = ", key);
    let body = if comment.is_empty() {
        format!("{}{}", prefix, value_str)
    } else {
        format!("{}{} / {}", prefix, value_str, comment)
    };
    if body.len() > CARD_SIZE {
        return Err(PyValueError::new_err(format!(
            "HIERARCH card too long ({} chars) for key '{}'; \
             shorten the key, value, or comment to fit in 80 chars",
            body.len(), key
        )));
    }
    Ok(pad_to_card(&body))
}

fn build_hierarch_string_cards(
    key: &str,
    value: &str,
    comment: &str,
) -> PyResult<Vec<String>> {
    let escaped = value.replace('\'', "''");

    let padded = if escaped.len() < 8 {
        format!("{:<8}", escaped)
    } else {
        escaped.clone()
    };
    let single_body = if comment.is_empty() {
        format!("HIERARCH {} = '{}'", key, padded)
    } else {
        format!("HIERARCH {} = '{}' / {}", key, padded, comment)
    };
    if single_body.len() <= CARD_SIZE {
        return Ok(vec![pad_to_card(&single_body)]);
    }

    if escaped.is_empty() {
        return Err(PyValueError::new_err(format!(
            "HIERARCH card too long ({} chars) for key '{}'; \
             shorten the key or comment",
            single_body.len(), key
        )));
    }

    if key.len() >= 65 {
        return Err(PyValueError::new_err(format!(
            "HIERARCH key '{}' is {} chars; max length for a CONTINUE-chained \
             HIERARCH string value is 64 chars (the first card must have at \
             least one byte of payload alongside framing)",
            key, key.len()
        )));
    }
    let first_max_payload: usize = 65 - key.len();

    let last_max_payload: usize = if comment.is_empty() {
        68
    } else {
        if comment.len() >= 65 {
            return Err(PyValueError::new_err(format!(
                "comment is too long ({} chars) to fit in a FITS card alongside \
                 a HIERARCH string value; max comment length is 64 chars for \
                 CONTINUE-chained values",
                comment.len()
            )));
        }
        65 - comment.len()
    };

    let bytes = escaped.as_bytes();
    let total = bytes.len();
    let mut cards: Vec<String> = Vec::new();
    let mut pos = 0;

    while pos < total {
        let is_first = cards.is_empty();
        let remaining = total - pos;
        let mut take;
        let is_last;
        if is_first {
            take = first_max_payload.min(remaining);
            is_last = false;
        } else if remaining <= last_max_payload {
            take = remaining;
            is_last = true;
        } else {
            take = (remaining - last_max_payload).min(67);
            is_last = false;
        }

        while take > 0 && pos + take < total {
            if bytes[pos + take - 1] == b'\'' && bytes[pos + take] == b'\'' {
                take -= 1;
            } else {
                break;
            }
        }
        if take == 0 {
            take = 2.min(remaining);
        }

        let inner = std::str::from_utf8(&bytes[pos..pos + take]).unwrap();
        let card = if is_first {
            format!("HIERARCH {} = '{}&'", key, inner)
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

    if cards.len() == 1 {
        let last_card = if comment.is_empty() {
            "CONTINUE  ''".to_string()
        } else {
            format!("CONTINUE  '' / {}", comment)
        };
        cards.push(pad_to_card(&last_card));
    }

    Ok(cards)
}

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

// Return a card list with protected-keyword cards (and their CONTINUE
// chains) removed.  Used by `to_dict(skip_protected=True)`.
fn filter_protected_cards(cards: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < cards.len() {
        let kw = keyword_of(&cards[i]);
        match kw.as_deref() {
            Some(k) if is_protected_key(k) => {
                i = find_chain_end(cards, i);
            }
            _ => {
                out.push(cards[i].clone());
                i += 1;
            }
        }
    }
    out
}

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

// One staged change produced by `collect_update_actions`.  Used by both the
// regular and edit()-batched update() paths.
enum UpdateAction {
    SetKey {
        key: String,
        value: Py<PyAny>,
        explicit_comment: Option<String>,
    },
    AppendCommentary { keyword: String, text: String },
}

// Collect the actions to apply from a source — either a FITSHeader (cards
// are walked to preserve comments) or any object with `.items()` returning
// (key, value-or-(value, comment)).  Policy split is documented in
// CLAUDE.md under "update() and commentary cards" and "update() policy on
// protected keys".
fn collect_update_actions(
    py: Python<'_>,
    other: &Bound<'_, PyAny>,
    copy_commentary: bool,
) -> PyResult<Vec<UpdateAction>> {
    let mut actions: Vec<UpdateAction> = Vec::new();

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
                if copy_commentary {
                    let text = if card.len() > 8 {
                        card[8..].trim_end().to_string()
                    } else {
                        String::new()
                    };
                    actions.push(UpdateAction::AppendCommentary {
                        keyword: kw_field_trimmed.to_string(),
                        text,
                    });
                }
                continue;
            }
            let kw = keyword_of(card).unwrap_or_default();
            if kw.is_empty() {
                continue;
            }
            if is_protected_key(&normalize_keyword(&kw)) {
                continue;
            }
            if let Some(eq_pos) = trimmed.find('=') {
                let (value, comment) = parse_value_with_continue(
                    py, &src_cards, i, &trimmed[eq_pos + 1..],
                )?;
                actions.push(UpdateAction::SetKey {
                    key: kw,
                    value,
                    explicit_comment: Some(comment),
                });
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
                    "update() does not accept commentary key '{}'; \
                     use add_comment(text) / add_history(text) / \
                     add_blank(text) to append commentary cards",
                    k
                )));
            }
            if is_protected_key(&k) {
                return Err(PyValueError::new_err(format!(
                    "'{}' is a protected keyword managed by rustfits (file \
                     structure, integrity, or compression layout) and cannot \
                     be set via update(); structural changes should go \
                     through the dedicated HDU APIs",
                    k
                )));
            }
            let val_obj = pair.get_item(1)?;
            let (v, c) = parse_setitem_value(&val_obj)?;
            actions.push(UpdateAction::SetKey {
                key: k,
                value: v.unbind(),
                explicit_comment: c,
            });
        }
    }

    Ok(actions)
}

// Rewrite the header on disk in place.  Cards are serialized to one or more
// 2880-byte blocks; the resulting byte count must equal `header_block_count
// * BLOCK_SIZE` (slack-only).  Taint semantics: pre-I/O failures don't
// taint; write_all/flush failures do.  See CLAUDE.md "Tainted-header state".
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
    f.seek(SeekFrom::Start(header_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    f.write_all(&bytes).map_err(|e| {
        tainted.store(true, Ordering::Release);
        PyIOError::new_err(format!(
            "header write failed mid-stream: {}; \
             the on-disk file may now be inconsistent — \
             close this FITS object and reopen the file to recover",
            e
        ))
    })?;
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

// ===== commentary write helpers =====

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

fn make_commentary_cards(keyword: &str, text: &str) -> Vec<String> {
    let max_text = CARD_SIZE - 8;
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
        let chunk_str = std::str::from_utf8(chunk).unwrap_or("");
        out.push(pad_to_card(&format!("{}{}", prefix, chunk_str)));
    }
    out
}

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

fn delete_commentary_cards(cards: &mut Vec<String>, keyword: &str) -> usize {
    let before = cards.len();
    cards.retain(|card| !card_matches_commentary_keyword(card, keyword));
    before - cards.len()
}

// ===== card-formatting helpers (write side) =====

pub(crate) fn pad_to_card(s: &str) -> String {
    let mut out = s.to_string();
    if out.len() < CARD_SIZE {
        out.push_str(&" ".repeat(CARD_SIZE - out.len()));
    } else if out.len() > CARD_SIZE {
        out.truncate(CARD_SIZE);
    }
    out
}

pub(crate) fn card_int(key: &str, value: i64, comment: &str) -> String {
    let head = format!("{:<8}= {:>20}", key, value);
    let body = if comment.is_empty() {
        head
    } else {
        format!("{} / {}", head, comment)
    };
    pad_to_card(&body)
}

pub(crate) fn card_logical(key: &str, value: bool, comment: &str) -> String {
    let v = if value { "T" } else { "F" };
    let head = format!("{:<8}= {:>20}", key, v);
    let body = if comment.is_empty() {
        head
    } else {
        format!("{} / {}", head, comment)
    };
    pad_to_card(&body)
}

pub(crate) fn card_string(key: &str, value: &str, comment: &str) -> String {
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

fn format_fits_float(value: f64) -> String {
    if !value.is_finite() {
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

fn build_string_value_cards(key: &str, value: &str, comment: &str) -> PyResult<Vec<String>> {
    let escaped = value.replace('\'', "''");

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

    if escaped.len() <= max_last_payload {
        return Ok(vec![card_string(key, value, comment)]);
    }

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
            take = (remaining - max_last_payload).min(67);
            is_last = false;

            while take > 0 && pos + take < total {
                if bytes[pos + take - 1] == b'\'' && bytes[pos + take] == b'\'' {
                    take -= 1;
                } else {
                    break;
                }
            }
            if take == 0 {
                take = 2;
            }
        }

        let inner = std::str::from_utf8(&bytes[pos..pos + take]).unwrap();
        let is_first = cards.is_empty();

        let card = if is_first {
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

// ===== FITSHeader pyclass =====

#[pyclass]
pub(crate) struct FITSHeader {
    cards: Arc<Mutex<Vec<String>>>,
    file: FileHandle,
    header_offset: u64,
    header_block_count: u64,
    tainted: TaintFlag,
}

impl FITSHeader {
    pub(crate) fn from_state(
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

    #[pyo3(signature = (skip_protected=false))]
    fn to_dict(&self, py: Python<'_>, skip_protected: bool) -> PyResult<Py<PyDict>> {
        let cards = self.snapshot()?;
        if skip_protected {
            let filtered = filter_protected_cards(&cards);
            parse_header_dict(py, &filtered)
        } else {
            parse_header_dict(py, &cards)
        }
    }

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
        if is_protected_key(&key) {
            return Err(PyValueError::new_err(format!(
                "'{}' is a protected keyword managed by rustfits (file structure, \
                 integrity, or compression layout) and cannot be set directly; \
                 structural changes should go through the dedicated HDU APIs",
                key
            )));
        }
        let (value_obj, explicit_comment) = parse_setitem_value(value)?;
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
        if is_protected_key(&key) {
            return Err(PyValueError::new_err(format!(
                "'{}' is a protected keyword managed by rustfits and cannot be \
                 deleted directly; structural changes should go through the \
                 dedicated HDU APIs",
                key
            )));
        }
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

    fn add_comment(&self, text: &str) -> PyResult<()> {
        self.append_commentary("COMMENT", text)
    }

    fn add_history(&self, text: &str) -> PyResult<()> {
        self.append_commentary("HISTORY", text)
    }

    fn add_blank(&self, text: &str) -> PyResult<()> {
        self.append_commentary("", text)
    }

    #[pyo3(signature = (other, *, copy_commentary=false))]
    fn update(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        copy_commentary: bool,
    ) -> PyResult<()> {
        check_not_tainted(&self.tainted)?;
        let actions = collect_update_actions(py, other, copy_commentary)?;
        if actions.is_empty() {
            return Ok(());
        }
        let mut guard = self.cards.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        let mut new_cards = guard.clone();
        for action in &actions {
            match action {
                UpdateAction::SetKey { key, value, explicit_comment } => {
                    apply_setitem(
                        &mut new_cards, key, value.bind(py),
                        explicit_comment.clone(),
                    )?;
                }
                UpdateAction::AppendCommentary { keyword, text } => {
                    append_commentary_to_cards(&mut new_cards, keyword, text);
                }
            }
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

// ===== FITSHeaderEdit: transactional header edits =====

#[pyclass]
pub(crate) struct FITSHeaderEdit {
    parent: Py<FITSHeader>,
    cards: Vec<String>,
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
        if is_protected_key(&key) {
            return Err(PyValueError::new_err(format!(
                "'{}' is a protected keyword managed by rustfits (file structure, \
                 integrity, or compression layout) and cannot be set directly; \
                 structural changes should go through the dedicated HDU APIs",
                key
            )));
        }
        let (value_obj, explicit_comment) = parse_setitem_value(value)?;
        apply_setitem(&mut self.cards, &key, &value_obj, explicit_comment)
    }

    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self.ensure_active()?;
        let key = normalize_keyword(key);
        if is_protected_key(&key) {
            return Err(PyValueError::new_err(format!(
                "'{}' is a protected keyword managed by rustfits and cannot be \
                 deleted directly; structural changes should go through the \
                 dedicated HDU APIs",
                key
            )));
        }
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

    #[pyo3(signature = (other, *, copy_commentary=false))]
    fn update(
        &mut self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        copy_commentary: bool,
    ) -> PyResult<()> {
        self.ensure_active()?;
        let actions = collect_update_actions(py, other, copy_commentary)?;
        for action in &actions {
            match action {
                UpdateAction::SetKey { key, value, explicit_comment } => {
                    apply_setitem(
                        &mut self.cards, key, value.bind(py),
                        explicit_comment.clone(),
                    )?;
                }
                UpdateAction::AppendCommentary { keyword, text } => {
                    append_commentary_to_cards(&mut self.cards, keyword, text);
                }
            }
        }
        Ok(())
    }

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

// Python-facing thin wrapper for is_protected_key.  Case-insensitive (the
// keyword is normalized to uppercase first).
#[pyfunction]
#[pyo3(name = "is_protected_key")]
pub(crate) fn py_is_protected_key(key: &str) -> bool {
    is_protected_key(&normalize_keyword(key))
}
