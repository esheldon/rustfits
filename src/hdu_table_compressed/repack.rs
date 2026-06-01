// Compressed-table heap repack (fixed + VLA dual-descriptor) plus the
// chunked in-file copy primitive.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::io::{Read, Seek, SeekFrom};

use crate::common::{
    lock_file, parse_keyword, stream_copy_in_file,
};
use crate::hdu::HDU;
use crate::hdu_table::{
    parse_columns,
    read_descriptor, write_descriptor, Column,
};

use super::hdu::{ColumnTileCache, synthesize_uncompressed_cards};
use super::read::{gzip_decompress_bytes};
use super::write::grow_file_to_at_least;

// ---------------------------------------------------------------------------
// Phase 6c-1 — repack() on compressed tables (streaming)
// ---------------------------------------------------------------------------
//
// Walk descriptors in scan order ((tile, col) lex), compute each
// live blob's compact-heap position (= cumulative size of live
// blobs), then move bytes from old → new position with chunked
// I/O.  Two move strategies:
//
//   Fast path (in-place streaming): blobs read in old-offset
//   order, written to their new positions in place.  Requires
//   `new_offset[i] + length[i] <= old_offset[i+1]` for every
//   adjacent pair (so writes never clobber unread blobs).  This
//   holds for the post-merge orphan pattern that Phase 6b's
//   append produces (orphans contiguous before live tail).
//   Cost: ≤ `sum_of_live_blobs_that_move` bytes of I/O —
//   typically just the rewritten last tile (~10 MB).
//
//   Slow path (staging): blobs first copied to a "staging" area
//   appended to the file end (read source → write past current
//   heap end), then staged bytes copied back to their new
//   in-heap positions, then file shrunk.  Always safe for
//   arbitrary orphan patterns (writes go to fresh space; the
//   back-copy is front-to-back since dst < src by `new_pcount`).
//   Cost: ~`2 × new_pcount` bytes of I/O.  Used as a fallback
//   when the fast path's safety check fails — important for
//   future mutators (`__setitem__`) that create arbitrary
//   orphans.
//
// Memory bound: ~1 MiB chunk + the descriptor table (`n_tiles *
// ncols * 16` bytes; a few KB to a few MB) + the move-plan
// vector (~32 bytes per live blob).  No heap-in-RAM allocation.
pub(crate) fn repack_compressed_table_heap(
    super_: &HDU,
    cache: &ColumnTileCache,
) -> PyResult<()> {
    use crate::common::shift_file_tail_backward_and_update_offsets;
    use crate::hdu_image::{round_up_to_block, serialize_header_to_disk_bytes};
    use std::io::Write;
    use std::sync::atomic::Ordering;

    crate::common::check_not_tainted(&super_.tainted)?;

    let cards = super_.header_snapshot()?;
    let virtual_cards = synthesize_uncompressed_cards(&cards);
    let columns = parse_columns(&virtual_cards)?;

    let n_tiles = parse_keyword(&cards, "NAXIS2")
        .unwrap_or(0).max(0) as usize;
    let descriptor_row_width = parse_keyword(&cards, "NAXIS1")
        .unwrap_or(0).max(0) as usize;
    let current_pcount = parse_keyword(&cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    if n_tiles == 0 || columns.is_empty() || current_pcount == 0 {
        return Ok(());
    }
    let data_offset = super_.offsets.data_offset();

    // VLA columns add an indirection layer: dual-descriptor
    // blobs (themselves heap-stored) hold compressed-Q
    // descriptors pointing at per-cell compressed bytes (also
    // heap-stored).  Repack must rewrite both layers, then
    // re-GZIP the blobs.  Mixed tables (fixed + VLA cols) take
    // the VLA-aware path; pure-fixed tables use the streamlined
    // fixed-only path below.
    if columns.iter().any(|c| c.var_kind.is_some()) {
        return repack_compressed_table_heap_vla(
            super_, cache, &cards, &columns, n_tiles,
            descriptor_row_width, current_pcount, data_offset,
        );
    }

    let ncols = columns.len();
    let heap_start = data_offset
        + (n_tiles as u64) * (descriptor_row_width as u64);

    // Read just the descriptor table (small; bounded by n_tiles *
    // ncols * 16 bytes).
    let desc_table_size = n_tiles * descriptor_row_width;
    let mut desc_table = vec![0u8; desc_table_size];
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut desc_table).map_err(|e| {
            PyIOError::new_err(format!(
                "repack: read descriptor table: {}", e))
        })?;
    }

    // Build per-blob move plan in scan order: cumulative sum of
    // lengths gives each live blob its new offset.  Skip empty
    // cells (descriptor stays (0, 0)).
    struct MovePlan {
        old_offset: u64,
        length: u64,
        new_offset: u64,
        tile_idx: usize,
        col_idx: usize,
    }
    let mut plans: Vec<MovePlan> = Vec::new();
    let mut cursor: u64 = 0;
    for tile_idx in 0..n_tiles {
        for col_idx in 0..ncols {
            let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
            let nelems_s = i64::from_be_bytes(
                desc_table[desc_off..desc_off + 8].try_into().unwrap(),
            );
            let old_off_s = i64::from_be_bytes(
                desc_table[desc_off + 8..desc_off + 16].try_into().unwrap(),
            );
            if nelems_s < 0 || old_off_s < 0 {
                return Err(PyValueError::new_err(format!(
                    "repack: tile {} col {} descriptor negative: \
                     nelems={} offset={}",
                    tile_idx, col_idx, nelems_s, old_off_s)));
            }
            let length = nelems_s as u64;
            let old_offset = old_off_s as u64;
            if length == 0 {
                continue;
            }
            if old_offset.checked_add(length)
                .map(|e| e > current_pcount)
                .unwrap_or(true)
            {
                return Err(PyValueError::new_err(format!(
                    "repack: tile {} col {} descriptor points past \
                     heap end (offset+bytes={} > PCOUNT={})",
                    tile_idx, col_idx,
                    old_offset.wrapping_add(length), current_pcount)));
            }
            plans.push(MovePlan {
                old_offset, length, new_offset: cursor,
                tile_idx, col_idx,
            });
            cursor += length;
        }
    }
    let new_pcount = cursor;
    if new_pcount == current_pcount {
        return Ok(());  // Already compact.
    }

    // Sort by old_offset so the in-place fast path reads sequentially.
    plans.sort_by_key(|p| p.old_offset);

    // Decide fast vs slow path.  Fast path needs: for every
    // adjacent (i, i+1) pair in old-offset order, the i-th
    // blob's write region must end at or before the (i+1)-th
    // blob's read region (otherwise the write clobbers an
    // unread blob).  Holds for the post-merge orphan pattern.
    let mut fast_path_safe = true;
    for i in 0..plans.len() {
        let cur = &plans[i];
        let next_read_start = if i + 1 < plans.len() {
            plans[i + 1].old_offset
        } else {
            current_pcount
        };
        if cur.new_offset + cur.length > next_read_start {
            fast_path_safe = false;
            break;
        }
    }

    const CHUNK: u64 = 1 << 20;
    let mut buf = vec![0u8; CHUNK as usize];

    if fast_path_safe {
        // In-place streaming.  Reading in old-offset order means
        // every subsequent read is past any prior write, so no
        // clobbering.
        for plan in &plans {
            if plan.new_offset == plan.old_offset {
                continue;
            }
            stream_copy_in_file(
                &super_.file, heap_start + plan.old_offset,
                heap_start + plan.new_offset, plan.length,
                &mut buf, CHUNK, &super_.tainted,
                "repack: in-place move",
            )?;
        }
    } else {
        // Slow path — copy blobs to a staging area appended past
        // the current heap, then back-copy staging → final heap
        // positions.  Always safe regardless of orphan pattern.
        //
        // Step 1: grow data section so the staging area sits at
        // [heap_start + current_pcount, heap_start + current_pcount + new_pcount).
        // grow_file_to_at_least rounds to block-aligned, so the
        // actual staging area may extend a few hundred bytes
        // beyond — that's fine since we only read/write the
        // first new_pcount bytes.
        let staged_data_bytes = (n_tiles as u64
            * descriptor_row_width as u64)
            + current_pcount + new_pcount;
        grow_file_to_at_least(
            &super_.file, &super_.layout, data_offset,
            staged_data_bytes, &super_.tainted,
        )?;
        let staging_start = heap_start + current_pcount;

        // Step 2: copy each blob from its old position to its
        // staging position.  Staging is past the live heap, so
        // these writes never clobber any read.
        for plan in &plans {
            stream_copy_in_file(
                &super_.file, heap_start + plan.old_offset,
                staging_start + plan.new_offset, plan.length,
                &mut buf, CHUNK, &super_.tainted,
                "repack: copy to staging",
            )?;
        }

        // Step 3: copy staging back to the heap's final positions.
        // For each blob: dst = heap_start + new_offset, src =
        // staging_start + new_offset.  dst < src by current_pcount
        // (= the gap between heap and staging), so a front-to-back
        // chunked copy never clobbers an unread source byte.
        for plan in &plans {
            stream_copy_in_file(
                &super_.file, staging_start + plan.new_offset,
                heap_start + plan.new_offset, plan.length,
                &mut buf, CHUNK, &super_.tainted,
                "repack: copy from staging",
            )?;
        }
        // (Staging contents now stale; the file-shrink below
        // reclaims those bytes.)
    }

    // Rewrite descriptor entries with the new offsets.
    for plan in &plans {
        let desc_off = plan.tile_idx * descriptor_row_width
            + plan.col_idx * 16;
        desc_table[desc_off..desc_off + 8]
            .copy_from_slice(&(plan.length as i64).to_be_bytes());
        desc_table[desc_off + 8..desc_off + 16]
            .copy_from_slice(&(plan.new_offset as i64).to_be_bytes());
    }
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&desc_table) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: descriptor table rewrite: {}; close + reopen", e)));
        }
        if let Err(e) = f.flush() {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: flush: {}; close + reopen", e)));
        }
    }

    // Shrink file.  For the fast path, current_padded → new_padded.
    // For the slow path, the staging temporarily grew the file by
    // up to new_pcount; the shrink reclaims both the orphans AND
    // the staging.  Same computation either way: new HDU end is
    // at `data_offset + round_up_to_block(desc_bytes + new_pcount)`.
    let new_data_bytes = (n_tiles as u64
        * descriptor_row_width as u64) + new_pcount;
    let new_padded = round_up_to_block(new_data_bytes);
    let new_hdu_end = data_offset + new_padded;
    let file_len = {
        let g = lock_file(&super_.file)?;
        let f = g.as_ref()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.len().map_err(|e| PyIOError::new_err(e.to_string()))?
    };
    if new_hdu_end < file_len {
        // Identify the next HDU on disk (if any) to decide
        // last-HDU (set_len) vs non-last (shift_file_tail_backward).
        let next_hdu_off: Option<u64> = {
            let guard = super_.layout.hdus.lock()
                .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
            guard.iter()
                .map(|o| o.header_offset())
                .filter(|&h| h > data_offset)
                .min()
        };
        match next_hdu_off {
            None => {
                // Last HDU — just trim the file.
                let mut g = lock_file(&super_.file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.set_len(new_hdu_end).map_err(|e| {
                    super_.tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "repack: set_len({}) failed: {}; close + reopen",
                        new_hdu_end, e))
                })?;
            }
            Some(next_off) => {
                // Non-last HDU.  After grow_file_to_at_least's
                // staging extension (slow path) or the post-merge
                // append's grow (fast path), the next HDU sits at
                // `next_off`.  Slide it (and everything after)
                // backward by `next_off - new_hdu_end` so the
                // current HDU's data section ends precisely at
                // `new_hdu_end` (block-aligned) and HDU N+1's
                // header lands at `new_hdu_end` itself.
                let delta = next_off - new_hdu_end;
                if delta > 0 {
                    shift_file_tail_backward_and_update_offsets(
                        &super_.file, &super_.layout,
                        next_off, delta, &super_.tainted)?;
                }
            }
        }
    }

    // Update PCOUNT — disk-write-before-commit pattern.
    let cards_guard = super_.cards_write_lock()?;
    let mut new_cards = cards.clone();
    crate::hdu_table::set_pcount_in_cards(&mut new_cards, new_pcount);
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

// VLA-aware repack.  Walks every (tile, col) in scan order:
//
//   - Fixed col: stream-copy the existing column blob to the
//     staging area; record new (offset, length) for the main
//     descriptor.
//   - VLA col: read + decompress the dual-descriptor blob; walk
//     each row, stream-copy live per-cell compressed bytes from
//     their old heap position to staging, rewrite cvlastart in
//     the in-RAM blob; re-GZIP the blob; write the freshly
//     gzipped blob to staging; record new (offset, length) for
//     the main descriptor.
//
// Staging area sits past the current heap (writes never clobber
// reads from the old heap).  After all (tile, col) processed,
// one big front-to-back stream copy moves staging[0..new_pcount]
// to heap_start[0..new_pcount] — safe because dst < src by at
// least `current_pcount` so chunks read past the cursor stay
// untouched.  Then file shrink + descriptor rewrite + PCOUNT
// update + cache clear, mirroring the fixed-only path's tail.
//
// Memory bound: ~1 MiB chunk + one decompressed dual-desc blob
// at a time (`rowspertile * (width_orig + 16)` bytes) + one
// gzipped blob held briefly while staging it + the descriptor
// table + the per-(tile, col) move-plan vector (~32 bytes per
// entry).  No heap-in-RAM allocation.  Staging temporarily
// roughly doubles the file's heap region; reclaimed on shrink.
//
// ZPCOUNT is the ORIGINAL (uncompressed) heap size, invariant
// under repack (we don't change which cells exist or their
// nelements, just where their compressed bytes live).  Don't
// touch it.
#[allow(clippy::too_many_arguments)]
fn repack_compressed_table_heap_vla(
    super_: &HDU,
    cache: &ColumnTileCache,
    cards: &[String],
    columns: &[Column],
    n_tiles: usize,
    descriptor_row_width: usize,
    current_pcount: u64,
    data_offset: u64,
) -> PyResult<()> {
    use crate::common::shift_file_tail_backward_and_update_offsets;
    use crate::hdu_image::{round_up_to_block, serialize_header_to_disk_bytes};
    use crate::zimage::gzip::encode_gzip1;
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let ncols = columns.len();
    let heap_start = data_offset
        + (n_tiles as u64) * (descriptor_row_width as u64);
    let ztilelen = parse_keyword(cards, "ZTILELEN")
        .unwrap_or(0).max(0) as usize;
    let total_nrows = parse_keyword(cards, "ZNAXIS2")
        .unwrap_or(0).max(0) as usize;

    // Read descriptor table — small (n_tiles × ncols × 16 bytes).
    let desc_table_size = n_tiles * descriptor_row_width;
    let mut desc_table = vec![0u8; desc_table_size];
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut desc_table).map_err(|e| {
            PyIOError::new_err(format!(
                "repack-vla: read descriptor table: {}", e))
        })?;
    }

    // Move plan: for each (tile, col), the new (offset, length)
    // of its main-table descriptor pointing into the compacted
    // heap.  (0, 0) for any (tile, col) whose source descriptor
    // was already empty — preserves the encode_vla_column_tile
    // convention.
    let mut new_main_descs: Vec<(usize, usize, u64, u64)> =
        Vec::with_capacity(n_tiles * ncols);

    let staging_start = heap_start + current_pcount;
    let mut staging_cursor: u64 = 0;
    const CHUNK: u64 = 1 << 20;
    let mut copy_buf: Vec<u8> = Vec::new();
    // grow_file_to_at_least wants bytes-after-data_offset.  The
    // staging area starts at staging_start = data_offset +
    // desc_bytes + current_pcount, so each write of `n` bytes
    // through cursor `c` reaches up to (desc_bytes + current_pcount
    // + c + n) bytes past data_offset.  Forgetting desc_bytes here
    // under-shifts the trailing HDU by `desc_bytes` (= 64 bytes for
    // a typical multi-col 1QB descriptor row), enough for staging
    // writes to clobber the start of HDU N+1's header.
    let desc_bytes_u64 = (n_tiles * descriptor_row_width) as u64;

    for tile_idx in 0..n_tiles {
        for (col_idx, col) in columns.iter().enumerate() {
            let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
            let nelems_s = i64::from_be_bytes(
                desc_table[desc_off..desc_off + 8].try_into().unwrap(),
            );
            let old_off_s = i64::from_be_bytes(
                desc_table[desc_off + 8..desc_off + 16].try_into().unwrap(),
            );
            if nelems_s <= 0 {
                new_main_descs.push((tile_idx, col_idx, 0, 0));
                continue;
            }
            let old_length = nelems_s as u64;
            let old_offset = old_off_s.max(0) as u64;
            if old_offset.checked_add(old_length)
                .map(|e| e > current_pcount)
                .unwrap_or(true)
            {
                return Err(PyValueError::new_err(format!(
                    "repack-vla: tile {} col '{}' main descriptor \
                     points past heap end (offset+bytes={} > PCOUNT={})",
                    tile_idx, col.name,
                    old_offset.wrapping_add(old_length), current_pcount)));
            }

            if col.var_kind.is_none() {
                // Fixed col blob: stream-copy old → staging at
                // current cursor.
                let want_total = desc_bytes_u64
                    + current_pcount + staging_cursor + old_length;
                grow_file_to_at_least(
                    &super_.file, &super_.layout, data_offset,
                    want_total, &super_.tainted,
                )?;
                stream_copy_in_file(
                    &super_.file, heap_start + old_offset,
                    staging_start + staging_cursor, old_length,
                    &mut copy_buf, CHUNK, &super_.tainted,
                    "repack-vla: stage fixed col blob",
                )?;
                new_main_descs.push((tile_idx, col_idx,
                    staging_cursor, old_length));
                staging_cursor += old_length;
                continue;
            }

            // VLA column path.  Read + decompress the existing
            // dual-descriptor blob from the heap (NOT staging —
            // staging is past current_pcount, sources live in
            // [0, current_pcount)).
            let width_orig = match col.var_kind {
                Some('P') => 8usize,
                Some('Q') => 16usize,
                _ => return Err(PyValueError::new_err(format!(
                    "column '{}': expected P or Q var_kind",
                    col.name))),
            };
            let rowspertile = if tile_idx + 1 == n_tiles {
                total_nrows.saturating_sub(tile_idx * ztilelen)
            } else {
                ztilelen
            };
            let expected_blob_size =
                rowspertile * width_orig + rowspertile * 16;
            let mut compressed_old = vec![0u8; old_length as usize];
            {
                let mut g = lock_file(&super_.file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(heap_start + old_offset))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.read_exact(&mut compressed_old).map_err(|e| {
                    PyIOError::new_err(format!(
                        "repack-vla: read old dual-descriptor blob for \
                         tile {} col '{}': {}",
                        tile_idx, col.name, e))
                })?;
            }
            let mut blob =
                gzip_decompress_bytes(&compressed_old, expected_blob_size)?;
            let comp_desc_start = rowspertile * width_orig;

            // Per-row: stream-copy live cell bytes to staging,
            // rewrite cvlastart in the in-RAM blob.  Empty cells
            // (cvlalen == 0) keep descriptor (0, 0).
            for r in 0..rowspertile {
                let comp_off = comp_desc_start + r * 16;
                let (cvlalen_s, cvlastart_old_s) = read_descriptor(
                    'Q', &blob[comp_off..comp_off + 16]);
                let cvlalen = cvlalen_s.max(0) as u64;
                let cvlastart_old = cvlastart_old_s.max(0) as u64;
                if cvlalen == 0 {
                    write_descriptor(
                        'Q', 0, 0,
                        &mut blob[comp_off..comp_off + 16],
                    );
                    continue;
                }
                if cvlastart_old.checked_add(cvlalen)
                    .map(|e| e > current_pcount)
                    .unwrap_or(true)
                {
                    return Err(PyValueError::new_err(format!(
                        "repack-vla: tile {} col '{}' row {} cell \
                         descriptor points past heap end \
                         (cvlastart+cvlalen={} > PCOUNT={})",
                        tile_idx, col.name, r,
                        cvlastart_old.wrapping_add(cvlalen),
                        current_pcount)));
                }
                let cvlastart_new = staging_cursor;
                let want_total = desc_bytes_u64
                    + current_pcount + staging_cursor + cvlalen;
                grow_file_to_at_least(
                    &super_.file, &super_.layout, data_offset,
                    want_total, &super_.tainted,
                )?;
                stream_copy_in_file(
                    &super_.file, heap_start + cvlastart_old,
                    staging_start + cvlastart_new, cvlalen,
                    &mut copy_buf, CHUNK, &super_.tainted,
                    "repack-vla: stage VLA cell bytes",
                )?;
                staging_cursor += cvlalen;
                write_descriptor(
                    'Q', cvlalen as usize, cvlastart_new as usize,
                    &mut blob[comp_off..comp_off + 16],
                );
            }

            // Re-GZIP the blob (compressed descriptors now point
            // at the staging-area cvlastart values).  Write to
            // staging at the current cursor.
            let gzipped = encode_gzip1(&blob, None)?;
            let blob_new_offset = staging_cursor;
            let blob_new_length = gzipped.len() as u64;
            let want_total = desc_bytes_u64
                + current_pcount + staging_cursor + blob_new_length;
            grow_file_to_at_least(
                &super_.file, &super_.layout, data_offset,
                want_total, &super_.tainted,
            )?;
            {
                let mut g = lock_file(&super_.file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(staging_start + staging_cursor))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.write_all(&gzipped).map_err(|e| {
                    super_.tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "repack-vla: stage dual-descriptor blob \
                         write for tile {} col '{}': {}; close + reopen",
                        tile_idx, col.name, e))
                })?;
            }
            staging_cursor += blob_new_length;
            new_main_descs.push((tile_idx, col_idx,
                blob_new_offset, blob_new_length));
        }
    }

    let new_pcount = staging_cursor;

    // Back-copy staging[0..new_pcount] → heap[0..new_pcount] in
    // ONE chunked copy.  Front-to-back is safe: dst = heap_start
    // + k, src = heap_start + current_pcount + k, so dst < src
    // by current_pcount > 0 for every k.
    if new_pcount > 0 {
        stream_copy_in_file(
            &super_.file, staging_start, heap_start, new_pcount,
            &mut copy_buf, CHUNK, &super_.tainted,
            "repack-vla: copy staging to heap",
        )?;
    }

    // Rewrite descriptor table with new (offset, length).
    for (tile_idx, col_idx, new_off, length) in &new_main_descs {
        let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
        desc_table[desc_off..desc_off + 8]
            .copy_from_slice(&(*length as i64).to_be_bytes());
        desc_table[desc_off + 8..desc_off + 16]
            .copy_from_slice(&(*new_off as i64).to_be_bytes());
    }
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&desc_table) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack-vla: descriptor table rewrite: {}; \
                 close + reopen", e)));
        }
        if let Err(e) = f.flush() {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack-vla: flush: {}; close + reopen", e)));
        }
    }

    // Shrink file (reclaims staging + orphans together).
    let new_data_bytes = (n_tiles as u64
        * descriptor_row_width as u64) + new_pcount;
    let new_padded = round_up_to_block(new_data_bytes);
    let new_hdu_end = data_offset + new_padded;
    let file_len = {
        let g = lock_file(&super_.file)?;
        let f = g.as_ref()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.len().map_err(|e| PyIOError::new_err(e.to_string()))?
    };
    if new_hdu_end < file_len {
        let next_hdu_off: Option<u64> = {
            let guard = super_.layout.hdus.lock()
                .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
            guard.iter()
                .map(|o| o.header_offset())
                .filter(|&h| h > data_offset)
                .min()
        };
        match next_hdu_off {
            None => {
                let mut g = lock_file(&super_.file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.set_len(new_hdu_end).map_err(|e| {
                    super_.tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "repack-vla: set_len({}) failed: {}; \
                         close + reopen", new_hdu_end, e))
                })?;
            }
            Some(next_off) => {
                let delta = next_off - new_hdu_end;
                if delta > 0 {
                    shift_file_tail_backward_and_update_offsets(
                        &super_.file, &super_.layout,
                        next_off, delta, &super_.tainted)?;
                }
            }
        }
    }

    // Update PCOUNT (ZPCOUNT stays unchanged — original-heap
    // size is invariant under repack).
    let cards_guard = super_.cards_write_lock()?;
    let mut new_cards = cards.to_vec();
    crate::hdu_table::set_pcount_in_cards(&mut new_cards, new_pcount);
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
                "repack-vla: PCOUNT header write: {}; \
                 close + reopen", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack-vla: PCOUNT header flush: {}; \
                 close + reopen", e))
        })?;
    }
    cards_guard.commit(new_cards);
    cache.clear();
    Ok(())
}

// `stream_copy_in_file` moved to `src/common.rs` so the ZIMAGE-side
// repack can share the same primitive.

