// Schema edits for ASCII tables: insert_column + delete_column.
//
// Mirrors `hdu_table/edit.rs` minus the BINTABLE-specific machinery:
//
//   - NO heap.  ASCII tables have PCOUNT=0 always, so there's no
//     heap relocation step.
//   - NO VLA branches.  ASCII has no variable-length columns.
//   - NO `WriteTransform`.  ASCII cells are text-formatted directly
//     via `format_one_cell`, so the new column's input is just a
//     1-D numpy ndarray.
//
// **ASCII-specific mechanic — TBCOL value shifts.**  Every column at
// or after the insert/delete slot has its `TBCOLn` byte position
// shifted by ±new_col_width.  This is the load-bearing difference
// from BINTABLE schema edits and the main reason this file exists
// separately rather than sharing code with `hdu_table/edit.rs`.
//
// Order of operations (same as BINTABLE):
//   1. Build the new card list with renumbered + value-shifted per-
//      column cards, then `rewrite_header_to_disk` (may grow header
//      blocks via the shared shift primitive).
//   2. Grow / shrink the data section to fit the new row width (may
//      shift later HDUs).
//   3. Strip-walk main rows back-to-front (insert) or front-to-back
//      (delete) to reshape each row.  Strip target ~1 MiB.
//
// Validate-then-mutate: input is fully validated before any file or
// header bytes are touched.  Mid-write I/O failures taint the file.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::Ordering;

use crate::common::{
    check_not_tainted, lock_file, parse_keyword,
    shift_file_tail_and_update_offsets,
    shift_file_tail_backward_and_update_offsets,
    zero_fill_range,
};
use crate::hdu::HDU;
use crate::hdu_image::round_up_to_block;
use crate::header::{card_int, card_string, card_uint, rewrite_header_to_disk};

use super::columns::{parse_ascii_columns, AsciiColumn};
use super::write_fixed::format_one_cell;
use super::write_setup::{
    dtype_to_ascii_write_columns, AsciiWriteColumn,
};

// Strip target: ~1 MiB per buffer.  Matches the bulk-write path and
// keeps peak RSS bounded regardless of table size.
const STRIP_TARGET_BYTES: usize = 1 << 20;

// --------------------------------------------------------------------
// Public entry points
// --------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_column_impl(
    py: Python<'_>,
    super_: &HDU,
    name: &str,
    data: &Bound<'_, PyAny>,
    position: Option<i64>,
    after: Option<&Bound<'_, PyAny>>,
    before: Option<&Bound<'_, PyAny>>,
    unit: Option<&str>,
    format: Option<&str>,
) -> PyResult<()> {
    check_not_tainted(&super_.tainted)?;

    let cards = super_.header_snapshot()?;
    let columns = parse_ascii_columns(&cards)?;
    let nrows =
        parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize;
    let row_width =
        parse_keyword(&cards, "NAXIS1").unwrap_or(0).max(0) as usize;

    if name.is_empty() {
        return Err(PyValueError::new_err(
            "insert_column: column name must be non-empty"));
    }
    let name_u = name.to_uppercase();
    if columns.iter().any(|c| c.name.to_uppercase() == name_u) {
        return Err(PyValueError::new_err(format!(
            "insert_column: column '{}' already exists", name)));
    }
    let insert_idx =
        resolve_insert_index(&columns, position, after, before)?;

    // Build the new column's write spec via the existing classifier
    // path (one-field structured dtype + optional formats= override).
    let new_write_col =
        build_single_write_column(py, name, data, format, unit)?;
    let new_col_byte_width = new_write_col.width;

    // Validate input shape: 1-D ndarray of length NAXIS2.
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !data.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(
            "insert_column: data must be a 1-D numpy ndarray"));
    }
    let shape: Vec<usize> = data.getattr("shape")?.extract()?;
    if shape.len() != 1 || shape[0] != nrows {
        return Err(PyValueError::new_err(format!(
            "insert_column: data must be a 1-D ndarray of shape ({},); \
             got {:?}", nrows, shape)));
    }

    let pre_width: usize = columns.iter().take(insert_idx)
        .map(|c| c.byte_width).sum();
    let post_width = row_width - pre_width;
    let new_row_width = row_width + new_col_byte_width;

    // Build the new AsciiColumn the row-shuffler will use to format
    // each new cell.  `byte_offset` matches its position in the new
    // row layout (pre_width), since format_one_cell is called against
    // a per-column buffer whose only column starts at offset 0 — but
    // the byte_width must match `new_col_byte_width`.
    let new_col_for_format = AsciiColumn {
        name: new_write_col.name.clone(),
        tform_letter: new_write_col.tform_letter,
        byte_offset: 0,
        byte_width: new_col_byte_width,
        decimals: new_write_col.decimals,
        tscal: 1.0,
        tzero: new_write_col.tzero.map(|x| x as f64).unwrap_or(0.0),
        tnull: None,
        tunit: new_write_col.tunit.clone(),
    };

    // Build the new card list before any file mutation.
    let mut new_cards = cards.clone();
    renumber_and_shift_per_column_cards(
        &mut new_cards, insert_idx, /* delta = */ 1, new_col_byte_width as i64,
    )?;
    let new_column_cards = build_new_column_cards(
        &new_write_col, insert_idx, pre_width,
    );
    insert_cards_before_end(&mut new_cards, &new_column_cards)?;
    update_int_card(
        &mut new_cards, "TFIELDS", (columns.len() + 1) as i64,
    )?;
    update_int_card(&mut new_cards, "NAXIS1", new_row_width as i64)?;

    // (1) Header rewrite first.  Acquire the cards write guard now so
    //     the eventual commit bumps the version counter under the
    //     same lock that protects the cards Vec.
    let cards_guard = super_.cards_write_lock()?;
    rewrite_header_to_disk(
        &super_.file, &super_.offsets, &super_.layout,
        &new_cards, &super_.tainted,
    )?;
    let data_offset = super_.offsets.data_offset();

    // (2) Extend the data section by the row-layout growth.
    let old_main_bytes = (nrows * row_width) as u64;
    let new_main_bytes = (nrows * new_row_width) as u64;
    grow_or_shrink_data_extent(
        super_, data_offset, old_main_bytes, new_main_bytes,
    )?;

    // (3) Shuffle main rows back-to-front into the new layout,
    //     formatting the new column inline.
    shuffle_main_for_insert(
        py, super_, data_offset, nrows, row_width, new_row_width,
        pre_width, post_width, new_col_byte_width,
        &new_col_for_format, data,
    )?;

    cards_guard.commit(new_cards);
    Ok(())
}

pub(crate) fn delete_column_impl(
    super_: &HDU,
    key: &Bound<'_, PyAny>,
) -> PyResult<()> {
    check_not_tainted(&super_.tainted)?;

    let cards = super_.header_snapshot()?;
    let columns = parse_ascii_columns(&cards)?;
    let nrows =
        parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize;
    let row_width =
        parse_keyword(&cards, "NAXIS1").unwrap_or(0).max(0) as usize;
    if columns.is_empty() {
        return Err(PyValueError::new_err(
            "delete_column: table has no columns to delete"));
    }

    let delete_idx = resolve_column_key(&columns, key)?;
    let deleted_col = &columns[delete_idx];
    let deleted_width = deleted_col.byte_width;
    let pre_width: usize = columns.iter().take(delete_idx)
        .map(|c| c.byte_width).sum();
    let post_offset = pre_width + deleted_width;
    let new_row_width = row_width - deleted_width;

    let mut new_cards = cards.clone();
    drop_per_column_cards(&mut new_cards, delete_idx);
    renumber_and_shift_per_column_cards(
        &mut new_cards, delete_idx, /* delta = */ -1,
        -(deleted_width as i64),
    )?;
    update_int_card(
        &mut new_cards, "TFIELDS", (columns.len() - 1) as i64,
    )?;
    update_int_card(&mut new_cards, "NAXIS1", new_row_width as i64)?;

    // (1) Header rewrite.
    let cards_guard = super_.cards_write_lock()?;
    rewrite_header_to_disk(
        &super_.file, &super_.offsets, &super_.layout,
        &new_cards, &super_.tainted,
    )?;
    let data_offset = super_.offsets.data_offset();

    // (2) Shuffle main rows front-to-back into the smaller layout.
    shuffle_main_for_delete(
        super_, data_offset, nrows, row_width, new_row_width,
        pre_width, post_offset,
    )?;

    // (3) Shrink data section.
    let old_main_bytes = (nrows * row_width) as u64;
    let new_main_bytes = (nrows * new_row_width) as u64;
    grow_or_shrink_data_extent(
        super_, data_offset, old_main_bytes, new_main_bytes,
    )?;

    cards_guard.commit(new_cards);
    Ok(())
}

// --------------------------------------------------------------------
// Strip-based row shuffling
// --------------------------------------------------------------------

// Back-to-front strip walker for insert.  Strips are processed from
// the LAST toward the FIRST because writes go to LATER positions (row
// width is growing); forward processing would clobber unread rows.
//
// Per strip: read R old rows, build R new rows (pre + new-column-text
// + post), write to the new (later) positions.
#[allow(clippy::too_many_arguments)]
fn shuffle_main_for_insert(
    py: Python<'_>,
    super_: &HDU,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    new_row_width: usize,
    pre_width: usize,
    post_width: usize,
    new_col_byte_width: usize,
    new_col: &AsciiColumn,
    input_arr: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if nrows == 0 {
        return Ok(());
    }
    let strip_rows =
        ((STRIP_TARGET_BYTES / new_row_width.max(1)).max(1)).min(nrows);
    let mut old_buf = vec![0u8; strip_rows * row_width];
    let mut new_buf = vec![b' '; strip_rows * new_row_width];

    let mut row_end = nrows;
    while row_end > 0 {
        let chunk = row_end.min(strip_rows);
        let row_start = row_end - chunk;
        old_buf.resize(chunk * row_width, 0);
        new_buf.resize(chunk * new_row_width, b' ');
        for b in new_buf.iter_mut() { *b = b' '; }

        // Read this strip's old bytes.
        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.read_exact(&mut old_buf).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "insert_column: read main strip failed: {}; \
                     close + reopen", e))
            })?;
        }

        // Build new rows: pre + new-column text + post.
        for r in 0..chunk {
            let src_row = &old_buf[r * row_width..(r + 1) * row_width];
            let dst_row =
                &mut new_buf[r * new_row_width..(r + 1) * new_row_width];
            dst_row[..pre_width].copy_from_slice(&src_row[..pre_width]);
            let cell_dst =
                &mut dst_row[pre_width..pre_width + new_col_byte_width];
            let input_row = row_start + r;
            format_one_cell(py, new_col, input_arr, input_row, cell_dst)?;
            dst_row[pre_width + new_col_byte_width..]
                .copy_from_slice(&src_row[pre_width..pre_width + post_width]);
        }

        // Write this strip's new bytes.
        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * new_row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            if let Err(e) = f.write_all(&new_buf) {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "insert_column: write main strip failed: {}; \
                     close + reopen", e)));
            }
            if let Err(e) = f.flush() {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "insert_column: flush failed: {}; close + reopen", e)));
            }
        }

        row_end = row_start;
    }
    Ok(())
}

// Front-to-back strip walker for delete.  Row width is shrinking, so
// writes land BEFORE reads — processing forward keeps un-read bytes
// safe.
#[allow(clippy::too_many_arguments)]
fn shuffle_main_for_delete(
    super_: &HDU,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    new_row_width: usize,
    pre_width: usize,
    post_offset: usize,
) -> PyResult<()> {
    if nrows == 0 || new_row_width == row_width {
        return Ok(());
    }
    let post_width = row_width - post_offset;
    let strip_rows =
        ((STRIP_TARGET_BYTES / row_width.max(1)).max(1)).min(nrows);
    let mut old_buf = vec![0u8; strip_rows * row_width];
    let mut new_buf = vec![0u8; strip_rows * new_row_width];

    let mut row_start = 0usize;
    while row_start < nrows {
        let chunk = (nrows - row_start).min(strip_rows);
        old_buf.resize(chunk * row_width, 0);
        new_buf.resize(chunk * new_row_width, 0);

        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.read_exact(&mut old_buf).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "delete_column: read main strip failed: {}; \
                     close + reopen", e))
            })?;
        }

        for r in 0..chunk {
            let src_row = &old_buf[r * row_width..(r + 1) * row_width];
            let dst_row =
                &mut new_buf[r * new_row_width..(r + 1) * new_row_width];
            dst_row[..pre_width].copy_from_slice(&src_row[..pre_width]);
            dst_row[pre_width..pre_width + post_width]
                .copy_from_slice(&src_row[post_offset..]);
        }

        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * new_row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            if let Err(e) = f.write_all(&new_buf) {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "delete_column: write main strip failed: {}; \
                     close + reopen", e)));
            }
            if let Err(e) = f.flush() {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "delete_column: flush failed: {}; close + reopen", e)));
            }
        }

        row_start += chunk;
    }
    Ok(())
}

// --------------------------------------------------------------------
// Data-section grow / shrink (block-aligned)
// --------------------------------------------------------------------

fn grow_or_shrink_data_extent(
    super_: &HDU,
    data_offset: u64,
    old_data_bytes: u64,
    new_data_bytes: u64,
) -> PyResult<()> {
    let current_padded = round_up_to_block(old_data_bytes);
    let new_padded = round_up_to_block(new_data_bytes);
    if new_padded == current_padded {
        return Ok(());
    }
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;
    let file_len = {
        let g = lock_file(&super_.file)?;
        let f = g.as_ref()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.len().map_err(|e| PyIOError::new_err(e.to_string()))?
    };
    if new_padded > current_padded {
        let delta = new_padded - current_padded;
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
            f.set_len(new_hdu_end).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "schema edit: set_len failed: {}; close + reopen", e))
            })?;
        }
    } else {
        let delta = current_padded - new_padded;
        if file_len > current_hdu_end {
            shift_file_tail_backward_and_update_offsets(
                &super_.file, &super_.layout,
                current_hdu_end, delta, &super_.tainted)?;
        } else {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "schema edit: set_len failed: {}; close + reopen", e))
            })?;
        }
    }
    Ok(())
}

// --------------------------------------------------------------------
// Position / column-key resolution
// --------------------------------------------------------------------

fn resolve_insert_index(
    columns: &[AsciiColumn],
    position: Option<i64>,
    after: Option<&Bound<'_, PyAny>>,
    before: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    let n_provided = [position.is_some(), after.is_some(), before.is_some()]
        .iter().filter(|b| **b).count();
    if n_provided > 1 {
        return Err(PyValueError::new_err(
            "insert_column: at most one of position=, after=, before= \
             may be specified"));
    }
    let ncols = columns.len();
    if let Some(p) = position {
        if p < 0 || p as usize > ncols {
            return Err(PyValueError::new_err(format!(
                "insert_column: position={} out of range (table has {} \
                 columns; valid positions are 0..={})",
                p, ncols, ncols)));
        }
        return Ok(p as usize);
    }
    if let Some(a) = after {
        let idx = resolve_column_key(columns, a)?;
        return Ok(idx + 1);
    }
    if let Some(b) = before {
        let idx = resolve_column_key(columns, b)?;
        return Ok(idx);
    }
    Ok(ncols)
}

fn resolve_column_key(
    columns: &[AsciiColumn], key: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let ncols = columns.len();
    if let Ok(name) = key.extract::<String>() {
        let name_u = name.to_uppercase();
        for (i, c) in columns.iter().enumerate() {
            if c.name.to_uppercase() == name_u {
                return Ok(i);
            }
        }
        return Err(PyValueError::new_err(format!(
            "column '{}' not found in table", name)));
    }
    if let Ok(i) = key.extract::<i64>() {
        let resolved = if i < 0 { i + ncols as i64 } else { i };
        if resolved < 0 || resolved as usize >= ncols {
            return Err(PyValueError::new_err(format!(
                "column index {} out of range (table has {} columns)",
                i, ncols)));
        }
        return Ok(resolved as usize);
    }
    Err(PyValueError::new_err(
        "column key must be a string name or integer index"))
}

// --------------------------------------------------------------------
// Per-column header card mutation
// --------------------------------------------------------------------

// Per-column keyword prefixes that need renumbering on insert/delete.
// For ASCII tables this is a narrower set than BINTABLE (no TDIM, no
// TNULL — though TNULL is allowed as a string sentinel; covered for
// safety).  TBCOL gets the value-shift in a second pass below.
const PER_COLUMN_PREFIXES: &[&str] = &[
    "TTYPE", "TFORM", "TUNIT", "TZERO", "TSCAL", "TNULL", "TDISP", "TBCOL",
];

fn parse_per_column_card(card: &str) -> Option<(&'static str, usize)> {
    if card.len() < 8 {
        return None;
    }
    let keyword = card[..8].trim_end();
    for &prefix in PER_COLUMN_PREFIXES {
        if let Some(rest) = keyword.strip_prefix(prefix) {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(n) = rest.parse::<usize>() {
                    if n >= 1 {
                        return Some((prefix, n));
                    }
                }
            }
        }
    }
    None
}

fn rebuild_per_column_card(
    card: &str, prefix: &'static str, new_n: usize,
) -> String {
    let padded_keyword = format!("{:<8}", format!("{}{}", prefix, new_n));
    if card.len() <= 8 {
        return padded_keyword;
    }
    format!("{}{}", padded_keyword, &card[8..])
}

// Renumber per-column cards' index suffixes AND shift TBCOL VALUES.
// Both rules pivot on the same column position (insert / delete idx).
//
// For an insert at `pivot_idx` (0-based) with `delta = +1`:
//   - Every TTYPE/TFORM/.../TBCOL with index n where (n-1) >= pivot_idx
//     gets its keyword index bumped to n+1.
//   - Every TBCOL in that same range gets its VALUE shifted by
//     +tbcol_value_delta.
//
// For a delete at `pivot_idx` with `delta = -1`:
//   - Cards with index n where (n-1) > pivot_idx renumber to n-1.
//     (The card at n-1 == pivot_idx itself is dropped before this
//     pass; see `drop_per_column_cards`.)
//   - The TBCOL VALUE for those same renumbered cards shifts by
//     `tbcol_value_delta` (= -deleted_width, negative).
//
// `tbcol_value_delta` is a signed i64 for the value shift in bytes.
fn renumber_and_shift_per_column_cards(
    cards: &mut [String],
    pivot_idx: usize,
    delta: i64,
    tbcol_value_delta: i64,
) -> PyResult<()> {
    let threshold = pivot_idx + 1; // 1-based
    for card in cards.iter_mut() {
        let Some((prefix, n)) = parse_per_column_card(card) else {
            continue;
        };
        let should_touch = match delta {
            1 => n >= threshold,
            -1 => n > threshold,
            _ => return Err(PyValueError::new_err(
                "internal: renumber delta must be +/-1")),
        };
        if !should_touch {
            continue;
        }
        let new_n = (n as i64 + delta) as usize;
        if prefix == "TBCOL" && tbcol_value_delta != 0 {
            // Combined: shift the value, then rebuild the card with
            // the new index.  The TBCOL value is a positive integer in
            // a standard FITS keyword card: "TBCOLn  = <value> [/ ...]".
            let new_value = shift_tbcol_value(card, tbcol_value_delta)?;
            *card = build_tbcol_card(new_n, new_value);
        } else {
            *card = rebuild_per_column_card(card, prefix, new_n);
        }
    }
    Ok(())
}

// Parse the integer value out of a TBCOLn card and add `delta_bytes`
// to it (signed).  Returns the new value.
fn shift_tbcol_value(card: &str, delta_bytes: i64) -> PyResult<i64> {
    // Card layout: cols 1-8 keyword, col 9 '=', then leading spaces +
    // an integer value at cols 11-30, then optional " / comment".
    if card.len() < 30 {
        return Err(PyValueError::new_err(format!(
            "header card too short for TBCOL: {:?}", card)));
    }
    let value_field = card[10..30].trim();
    let current: i64 = value_field.parse().map_err(|_| {
        PyValueError::new_err(format!(
            "TBCOL value '{}' is not an integer", value_field))
    })?;
    let new = current + delta_bytes;
    if new < 1 {
        return Err(PyValueError::new_err(format!(
            "TBCOL value would shift below 1 ({} + {} = {})",
            current, delta_bytes, new)));
    }
    Ok(new)
}

// Rebuild a TBCOLn card with the new index suffix and new value.
// Mirrors the shape `card_int` produces in `header.rs`.
fn build_tbcol_card(new_n: usize, new_value: i64) -> String {
    card_int(
        &format!("TBCOL{}", new_n), new_value,
        "starting column of field",
    ).trim_end().to_string()
}

fn drop_per_column_cards(cards: &mut Vec<String>, delete_idx: usize) {
    let target = delete_idx + 1;
    cards.retain(|card| match parse_per_column_card(card) {
        Some((_, n)) => n != target,
        None => true,
    });
}

fn insert_cards_before_end(
    cards: &mut Vec<String>, new_cards: &[String],
) -> PyResult<()> {
    let end_idx = cards.iter().rposition(|c| {
        c.len() >= 3 && c[..3].trim() == "END"
    }).ok_or_else(|| PyValueError::new_err(
        "header missing END card"))?;
    for (i, c) in new_cards.iter().enumerate() {
        cards.insert(end_idx + i, c.clone());
    }
    Ok(())
}

fn update_int_card(
    cards: &mut [String], keyword: &str, new_value: i64,
) -> PyResult<()> {
    let comment = match keyword {
        "NAXIS1" => "width of table in bytes",
        "NAXIS2" => "number of rows in table",
        "TFIELDS" => "number of columns",
        _ => "",
    };
    let new_card = card_int(keyword, new_value, comment)
        .trim_end().to_string();
    for card in cards.iter_mut() {
        if card.len() >= 8 && card[..8].trim_end() == keyword {
            *card = new_card;
            return Ok(());
        }
    }
    Err(PyValueError::new_err(format!(
        "header missing required keyword {}", keyword)))
}

// Build the new column's header cards: TTYPE, TBCOL, TFORM, optional
// TZERO and TUNIT.  `pre_width` is the byte_offset (so TBCOL =
// pre_width + 1 for 1-based).  Order matches what
// build_ascii_table_header_cards emits per column in write_setup.rs.
fn build_new_column_cards(
    col: &AsciiWriteColumn, insert_idx: usize, pre_width: usize,
) -> Vec<String> {
    let n = insert_idx + 1;
    let tform = match (col.tform_letter, col.decimals) {
        ('A' | 'I', _) => format!("{}{}", col.tform_letter, col.width),
        (_, Some(d)) => format!("{}{}.{}", col.tform_letter, col.width, d),
        (_, None) => format!("{}{}", col.tform_letter, col.width),
    };
    let mut out = vec![
        card_string(
            &format!("TTYPE{}", n), &col.name, "label for column",
        ).trim_end().to_string(),
        card_int(
            &format!("TBCOL{}", n),
            (pre_width + 1) as i64,
            "starting column of field",
        ).trim_end().to_string(),
        card_string(
            &format!("TFORM{}", n), &tform, "data format of column",
        ).trim_end().to_string(),
    ];
    if let Some(tz) = col.tzero {
        out.push(card_uint(
            &format!("TZERO{}", n), tz,
            "offset for unsigned integer (unsigned-int trick)",
        ).trim_end().to_string());
    }
    if let Some(unit) = &col.tunit {
        out.push(card_string(
            &format!("TUNIT{}", n), unit, "physical unit of column",
        ).trim_end().to_string());
    }
    out
}

// Build a single AsciiWriteColumn from name + data + optional
// format-override + optional unit.  Reuses dtype_to_ascii_write_columns
// by wrapping the inputs in 1-field structured dtype + 1-entry dicts.
fn build_single_write_column(
    py: Python<'_>,
    name: &str,
    data: &Bound<'_, PyAny>,
    format: Option<&str>,
    unit: Option<&str>,
) -> PyResult<AsciiWriteColumn> {
    let np = py.import("numpy")?;
    let arr_dtype = data.getattr("dtype")?;
    // Object dtype is not supported on ASCII (no VLA / no heap).
    let kind: String = arr_dtype.getattr("kind")?.extract()?;
    if kind == "O" {
        return Err(PyValueError::new_err(
            "insert_column: ASCII tables don't support Object-dtype \
             (VLA) columns; use BINTABLE for VLAs"));
    }
    let field_tuple = PyTuple::new(py, [
        name.into_pyobject(py)?.into_any(),
        arr_dtype.clone().into_any(),
    ])?;
    let descr = pyo3::types::PyList::new(py, [field_tuple])?;
    let np_dtype = np.getattr("dtype")?.call1((descr,))?;
    let units_dict = unit.map(|u| {
        let d = PyDict::new(py);
        d.set_item(name, u).ok();
        d
    });
    let formats_dict = format.map(|f| {
        let d = PyDict::new(py);
        d.set_item(name, f).ok();
        d
    });
    let cols = dtype_to_ascii_write_columns(
        &np_dtype, units_dict.as_ref(), formats_dict.as_ref(),
    )?;
    if cols.len() != 1 {
        return Err(PyValueError::new_err(
            "internal: dtype_to_ascii_write_columns returned wrong count"));
    }
    Ok(cols.into_iter().next().unwrap())
}
