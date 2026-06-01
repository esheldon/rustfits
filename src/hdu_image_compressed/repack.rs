// Compressed-image heap repack: drop orphans + shrink the file.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::Ordering;

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};

use crate::common::{
    check_not_tainted, lock_file, parse_keyword,
    shift_file_tail_backward_and_update_offsets,
    stream_copy_in_file,
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
// Streams live heap blobs to their new compact positions in 1 MiB
// chunks via `stream_copy_in_file`; never holds the whole heap in
// RAM.  Peak working memory:
//
//     ~1 MiB chunk buffer
//   + the descriptor table (naxis1 * naxis2 bytes, bounded by
//     n_tiles * ncols * 16 -- a few KB to a few MB for any real file)
//   + the move-plan vector (~40 bytes per live blob)
//
// On a 1 GB heap that's ~50 MB peak RSS instead of the ~1.5 GB the
// previous whole-heap-into-RAM implementation paid.  Mirrors the
// streaming ZTABLE repack pattern in
// `src/hdu_table_compressed/repack.rs` -- same fast/slow path
// decision, same move-plan shape, same chunked-copy primitive.
//
// Two move strategies:
//
//   Fast path (in-place streaming).  Live blobs read in old-offset
//   order, written to their new positions in place.  Requires that
//   for every adjacent pair `[i, i+1]` in old-offset order,
//   `new_offset[i] + length[i] <= old_offset[i+1]` (so writes never
//   clobber unread blobs).  Holds for the post-`__setitem__` /
//   post-`extend` orphan pattern that compressed-image mutations
//   produce: orphans + live tail, live tail moves backward, no
//   clobbering.
//
//   Slow path (staging).  Live blobs first copied to a staging
//   region appended past the current heap end (writes never clobber
//   any read), then copied back to their final in-heap positions
//   (back-copy is dst < src in fresh space, also safe).  Used as a
//   fallback when the fast-path safety check fails -- forward-
//   compatible for future mutators that might produce arbitrary
//   orphan patterns.
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
    let heap_start = data_offset + main_bytes;

    // Read just the descriptor table (small, bounded by
    // n_tiles * ncols * 16 bytes).  Walking the descriptors gives
    // us every live blob's (old_off, length) -- no need to ever
    // touch the heap bytes.
    let mut main_buf = vec![0u8; main_bytes as usize];
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut main_buf)
            .map_err(|e| PyIOError::new_err(format!(
                "repack: read main failed: {}", e)))?;
    }

    // Build the per-blob move plan.  Walk in scan order
    // (row × column); rewrite each descriptor in `main_buf` to
    // point at its new compact-heap offset.  Empty descriptors get
    // canonicalized to (0, 0).
    struct MovePlan {
        old_off: u64,
        length: u64,
        new_off: u64,
    }
    let primary_slot = Some(cols.primary);
    let cols_list: [&Option<ZimageColumnInfo>; 3] = [
        &primary_slot,
        &cols.gzip_fallback,
        &cols.uncompressed_fallback,
    ];
    let mut plans: Vec<MovePlan> = Vec::new();
    let mut new_cursor: u64 = 0;
    for r in 0..naxis2 {
        let row_off = (r * naxis1) as usize;
        for slot in cols_list.iter() {
            let Some(col) = slot.as_ref() else { continue; };
            let desc_at = row_off + col.byte_offset_in_row as usize;
            let (nel, old_off) =
                read_descriptor_from_buf(&main_buf, desc_at, col.is_q);
            if nel == 0 {
                // Empty descriptor; canonicalize as (0, 0).
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
            let new_off = new_cursor;
            plans.push(MovePlan { old_off, length: n_bytes, new_off });
            write_descriptor(
                &mut main_buf, desc_at, col.is_q, nel, new_off)?;
            new_cursor += n_bytes;
        }
    }
    let new_pcount = new_cursor;
    if new_pcount == current_pcount {
        // Already compact — every live blob's old_off already equals
        // its new_off, so the file is unchanged.
        return Ok(());
    }

    // Sort plans by old_off so the in-place fast path reads
    // sequentially without backtracking.
    plans.sort_by_key(|p| p.old_off);

    // Decide fast vs slow path.  Fast path needs: for every
    // adjacent (i, i+1) pair, the i-th blob's write region must end
    // at or before the (i+1)-th blob's read region (otherwise the
    // write clobbers an unread blob).  Holds for the post-
    // setitem/extend orphan pattern (orphans before live tail) but
    // not for arbitrary orphan layouts.
    let mut fast_path_safe = true;
    for i in 0..plans.len() {
        let cur = &plans[i];
        let next_read_start = if i + 1 < plans.len() {
            plans[i + 1].old_off
        } else {
            current_pcount
        };
        if cur.new_off + cur.length > next_read_start {
            fast_path_safe = false;
            break;
        }
    }

    const CHUNK: u64 = 1 << 20;
    let mut buf = vec![0u8; CHUNK as usize];

    let current_data_bytes = main_bytes + current_pcount;
    let new_data_bytes = main_bytes + new_pcount;
    let current_padded =
        crate::hdu_image::round_up_to_block(current_data_bytes);
    let new_padded =
        crate::hdu_image::round_up_to_block(new_data_bytes);
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;

    // Track the "effective current HDU end" used by the shrink
    // step below.  In the fast path the file extent is unchanged
    // (current_hdu_end).  In the slow path we grow to accommodate
    // staging — the post-grow extent (block-rounded over the
    // stage area) becomes the new "current end" that the shrink
    // logic operates against.
    let effective_current_hdu_end = if fast_path_safe {
        current_hdu_end
    } else {
        data_offset
            + crate::hdu_image::round_up_to_block(
                main_bytes + current_pcount + new_pcount)
    };

    if fast_path_safe {
        // In-place streaming.  Reading in old-offset order means
        // every subsequent read is past any prior write, so no
        // clobbering.
        for plan in &plans {
            if plan.new_off == plan.old_off {
                continue;
            }
            stream_copy_in_file(
                &super_.file,
                heap_start + plan.old_off,
                heap_start + plan.new_off,
                plan.length,
                &mut buf,
                CHUNK,
                &super_.tainted,
                "repack: in-place move",
            )?;
        }
    } else {
        // Slow path — stage live blobs past the current heap end
        // (writes never clobber any read), then back-copy staging →
        // final in-heap positions (back-copy is dst < src in fresh
        // space, also safe).
        //
        // Use `grow_file_to_at_least` so a non-last HDU's trailing
        // HDUs get shifted forward to make room for the staging
        // area — bare `set_len` would silently overwrite the next
        // HDU's bytes.  Exercised by
        // `test_repack_slow_path_non_last_hdu_no_corruption`.
        crate::common::grow_file_to_at_least(
            &super_.file, &super_.layout, data_offset,
            main_bytes + current_pcount + new_pcount,
            &super_.tainted,
        )?;
        let staging_start = heap_start + current_pcount;
        // Copy each live blob to its staging position.
        for plan in &plans {
            stream_copy_in_file(
                &super_.file,
                heap_start + plan.old_off,
                staging_start + plan.new_off,
                plan.length,
                &mut buf,
                CHUNK,
                &super_.tainted,
                "repack: copy to staging",
            )?;
        }
        // Back-copy staging → final positions.
        for plan in &plans {
            stream_copy_in_file(
                &super_.file,
                staging_start + plan.new_off,
                heap_start + plan.new_off,
                plan.length,
                &mut buf,
                CHUNK,
                &super_.tainted,
                "repack: back-copy from staging",
            )?;
        }
    }

    // Write the updated descriptor table back.  Live blobs are
    // already at their new positions in the heap; only the
    // descriptors changed.
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
        if let Err(e) = f.flush() {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: flush: {}; close + reopen", e)));
        }
    }

    // `effective_current_hdu_end` accounts for the slow-path
    // file grow (post-stage); in the fast path it equals
    // `current_hdu_end` (no change).
    if new_hdu_end < effective_current_hdu_end {
        let delta = effective_current_hdu_end - new_hdu_end;
        let file_len = {
            let g = lock_file(&super_.file)?;
            let f = g.as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.len()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        };
        if file_len > effective_current_hdu_end {
            shift_file_tail_backward_and_update_offsets(
                &super_.file, &super_.layout,
                effective_current_hdu_end, delta, &super_.tainted)?;
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
