// Compressed-image heap repack: drop orphans + shrink the file.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::Ordering;

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};

use crate::common::{
    check_not_tainted, lock_file, parse_keyword,
    shift_file_tail_backward_and_update_offsets,
};
use crate::hdu::HDU;
use crate::hdu_image::serialize_header_to_disk_bytes;
use crate::hdu_table::set_pcount_in_cards;

use super::hdu::TileCache;
use super::meta::{find_data_columns, ZimageColumnInfo};
use super::write::{read_descriptor_from_buf, write_descriptor};

// ---------------------------------------------------------------------------
// Heap repack — drop orphans accumulated by extend/__setitem__
// ---------------------------------------------------------------------------
//
// ZIMAGE heaps are the same shape as VLA-table heaps (a contiguous
// byte region after the main rows, addressed by P/Q descriptors in the
// main rows).  This function mirrors `repack_table_heap` in hdu_table:
// walk every row × every descriptor column (primary + optional GZIP /
// UNCOMPRESSED fallbacks), copy live cells into a compact new heap,
// rewrite the in-memory descriptors, write everything back, shrink the
// on-disk file if the new padded extent is smaller, then update
// PCOUNT.  Clears the tile cache (its entries no longer match the new
// heap layout).
pub(crate) fn repack_compressed_heap(
    super_: &HDU,
    cache: &TileCache,
) -> PyResult<()> {
    check_not_tainted(&super_.tainted)?;
    let cards = super_.header_snapshot()?;
    let naxis1 = parse_keyword(&cards, "NAXIS1")
        .unwrap_or(0).max(0) as u64;
    let naxis2 = parse_keyword(&cards, "NAXIS2")
        .unwrap_or(0).max(0) as u64;
    let current_pcount = parse_keyword(&cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    let data_offset = super_.offsets.data_offset();
    if current_pcount == 0 || naxis2 == 0 {
        return Ok(());
    }

    // Reject non-default THEAP — repack would write the new heap at
    // the default position and corrupt a non-default layout.  Files
    // rustfits creates never set THEAP, so this only blocks the rare
    // case of repacking a file written by another tool with a custom
    // heap offset.
    let theap_raw = parse_keyword(&cards, "THEAP").unwrap_or(0);
    let main_bytes = naxis1.saturating_mul(naxis2);
    if theap_raw > 0 && (theap_raw as u64) != main_bytes {
        return Err(PyValueError::new_err(format!(
            "repack: file has non-default THEAP={} (main rows end at \
             {}); repack would write the new heap at the default \
             position and corrupt the file",
            theap_raw, main_bytes)));
    }

    let cols = find_data_columns(&cards)?;

    // Read whole main table + old heap under a single file lock.
    let mut main_buf = vec![0u8; main_bytes as usize];
    let mut old_heap = vec![0u8; current_pcount as usize];
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut main_buf)
            .map_err(|e| PyIOError::new_err(format!(
                "repack: read main failed: {}", e)))?;
        f.seek(SeekFrom::Start(data_offset + main_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut old_heap)
            .map_err(|e| PyIOError::new_err(format!(
                "repack: read heap failed: {}", e)))?;
    }

    // Walk every row × every descriptor column; copy live cells.
    let primary_slot = Some(cols.primary);
    let cols_list: [&Option<ZimageColumnInfo>; 3] = [
        &primary_slot,
        &cols.gzip_fallback,
        &cols.uncompressed_fallback,
    ];
    let mut new_heap: Vec<u8> = Vec::new();
    for r in 0..naxis2 {
        let row_off = (r * naxis1) as usize;
        for slot in cols_list.iter() {
            let Some(col) = slot.as_ref() else { continue; };
            let desc_at = row_off + col.byte_offset_in_row as usize;
            let (nel, old_off) =
                read_descriptor_from_buf(&main_buf, desc_at, col.is_q);
            if nel == 0 {
                // Empty descriptor; rewrite as (0, 0) to keep the
                // layout canonical.
                write_descriptor(
                    &mut main_buf, desc_at, col.is_q, 0, 0)?;
                continue;
            }
            let n_bytes = nel.saturating_mul(col.inner_byte_width);
            if old_off + n_bytes > current_pcount {
                return Err(PyValueError::new_err(format!(
                    "repack: tile row {}: descriptor points past \
                     heap end (offset+bytes={} > PCOUNT={})",
                    r, old_off + n_bytes, current_pcount)));
            }
            let new_off = new_heap.len() as u64;
            new_heap.extend_from_slice(
                &old_heap[old_off as usize
                    ..(old_off + n_bytes) as usize]);
            write_descriptor(
                &mut main_buf, desc_at, col.is_q, nel, new_off)?;
        }
    }
    drop(old_heap);
    let new_pcount = new_heap.len() as u64;
    if new_pcount == current_pcount {
        return Ok(());
    }

    let current_data_bytes = main_bytes + current_pcount;
    let new_data_bytes = main_bytes + new_pcount;
    let current_padded =
        crate::hdu_image::round_up_to_block(current_data_bytes);
    let new_padded =
        crate::hdu_image::round_up_to_block(new_data_bytes);
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;

    // Write back main table + new heap.
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&main_buf) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: write main: {}; close + reopen", e)));
        }
        f.seek(SeekFrom::Start(data_offset + main_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&new_heap) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: write heap: {}; close + reopen", e)));
        }
        if let Err(e) = f.flush() {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: flush: {}; close + reopen", e)));
        }
    }

    if new_hdu_end < current_hdu_end {
        let delta = current_hdu_end - new_hdu_end;
        let file_len = {
            let g = lock_file(&super_.file)?;
            let f = g.as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
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
                    "repack: set_len: {}; close + reopen", e))
            })?;
        }
    }

    // PCOUNT update — disk-write-before-commit.
    let cards_guard = super_.cards_write_lock()?;
    let mut new_cards = cards_guard.clone_cards();
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
                "repack: PCOUNT header write: {}; close + reopen", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack: PCOUNT header flush: {}; close + reopen", e))
        })?;
    }
    cards_guard.commit(new_cards);
    cache.clear();
    Ok(())
}
