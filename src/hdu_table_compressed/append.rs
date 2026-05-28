// Compressed-table append: merge-into-partial-last-tile (fixed + VLA),
// heap relocation, and the existing-tile decode helper.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use crate::common::{
    lock_file, parse_keyword,
    FileHandle, FileLayout, TaintFlag,
};
use crate::zimage::compression_config::CompressionConfigKind;
use crate::hdu::HDU;
use crate::hdu_table::{
    apply_transform_cell,
    bytes_per_element, plan_vla_heap_layout,
    read_descriptor,
    serialize_vla_cell, validate_vla_cell, write_descriptor, Column, VlaCellPlan,
};
use crate::zimage::CompressionAlgorithm;

use super::hdu::ColumnTileCache;
use super::read::{decompress_column_slab, gzip_decompress_bytes};
use super::repack::{stream_copy_in_file};
use super::write::{
    encode_vla_column_tile, grow_file_to_at_least, set_zpcount_in_cards,
};
use super::write_setup::{
    ColPrep, build_and_encode_tile_col, encode_be_slab_to_heap_and_record,
    encode_table_column_slab, gzip_level_of, prepare_fixed_column,
    rice_blocksize_of,
};

// Pre-mutation snapshot of an existing tile's VLA dual-descriptor
// blob.  Decompressed eagerly so the original per-row descriptors
// (vlalen, original-heap offset, cvlalen, cvlastart) are available
// without re-touching the file after the heap relocates.
pub(crate) struct VlaMergeOldBlob {
    pub(crate) decompressed: Vec<u8>,
    pub(crate) width_orig: usize,
    pub(crate) rowspertile: usize,
}

// Read + decompress one (tile, col) dual-descriptor blob from the
// CURRENT (pre-mutation) heap.  Called only when merging rows into
// the existing last partial tile.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_vla_merge_old_blob(
    file: &FileHandle,
    data_offset: u64,
    tile_idx: usize,
    col_idx: usize,
    col: &Column,
    rowspertile: usize,
    existing_n_tiles: usize,
    descriptor_row_width: usize,
) -> PyResult<VlaMergeOldBlob> {
    let width_orig = match col.var_kind {
        Some('P') => 8usize,
        Some('Q') => 16usize,
        _ => return Err(PyValueError::new_err(format!(
            "column '{}': expected P or Q var_kind, got {:?}",
            col.name, col.var_kind))),
    };
    let main_desc_off = data_offset
        + (tile_idx as u64) * (descriptor_row_width as u64)
        + (col_idx as u64) * 16;
    let mut main_desc = [0u8; 16];
    {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(main_desc_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut main_desc).map_err(|e| {
            PyIOError::new_err(format!(
                "append: read main desc for VLA tile {} col '{}': {}",
                tile_idx, col.name, e))
        })?;
    }
    let (blob_nelems_s, blob_off_s) = read_descriptor('Q', &main_desc);
    let blob_nelems = blob_nelems_s.max(0) as usize;
    let blob_heap_off = blob_off_s.max(0) as u64;
    let old_heap_start = data_offset
        + (existing_n_tiles as u64) * (descriptor_row_width as u64);
    let mut compressed = vec![0u8; blob_nelems];
    if blob_nelems > 0 {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(old_heap_start + blob_heap_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut compressed).map_err(|e| {
            PyIOError::new_err(format!(
                "append: read existing VLA dual-descriptor blob for \
                 tile {} col '{}': {}", tile_idx, col.name, e))
        })?;
    }
    let expected_blob_size = rowspertile * width_orig + rowspertile * 16;
    let decompressed = if blob_nelems > 0 {
        gzip_decompress_bytes(&compressed, expected_blob_size)?
    } else {
        Vec::new()
    };
    Ok(VlaMergeOldBlob { decompressed, width_orig, rowspertile })
}

// Re-encode the last existing tile of a VLA column for the merge
// path of append.  Existing rows: keep original descriptors verbatim
// (no decompress / re-compress), copy per-cell compressed bytes from
// their old heap position to the heap end, rewrite compressed
// descriptors with the new offset.  New rows: encode per-cell with
// the uncompressed-fallback contract, original-descriptor offset
// from the planner (extends past current ZPCOUNT).
#[allow(clippy::too_many_arguments)]
fn encode_vla_column_tile_with_merge(
    py: Python<'_>,
    ndarray: &Bound<'_, PyAny>,
    file: &FileHandle,
    layout: &Arc<FileLayout>,
    tainted: &TaintFlag,
    data_offset: u64,
    heap_start_offset: u64,
    mut heap_cursor: u64,
    col: &Column,
    col_input: &Bound<'_, PyAny>,
    col_plans: &[VlaCellPlan],
    tile_idx: usize,
    last_existing_tile_rows: usize,
    merge_rows: usize,
    old_blob: &VlaMergeOldBlob,
    algo: CompressionAlgorithm,
    rice_blocksize: u32,
    gzip_level: Option<u32>,
    desc_table: &mut [u8],
    col_idx: usize,
    descriptor_row_width: usize,
) -> PyResult<u64> {
    use crate::zimage::gzip::encode_gzip1;
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let inner_letter = col.tform_letter;
    let elem_size = bytes_per_element(inner_letter).ok_or_else(|| {
        PyValueError::new_err(format!(
            "VLA column '{}': unsupported inner letter '{}'",
            col.name, inner_letter))
    })?;
    let descriptor_kind = col.var_kind
        .expect("encode_vla_column_tile_with_merge called for non-VLA");
    let width_orig = if descriptor_kind == 'P' { 8 } else { 16 };
    let merged_rows = last_existing_tile_rows + merge_rows;
    let blob_size = merged_rows * width_orig + merged_rows * 16;
    let mut new_blob = vec![0u8; blob_size];
    let comp_desc_start = merged_rows * width_orig;
    let old_comp_desc_start =
        old_blob.rowspertile * old_blob.width_orig;

    // Chunk buffer for the in-file copy of existing per-cell bytes.
    // 1 MiB matches the rest of the file (~tight, peak-RSS-bounded).
    let mut copy_buf: Vec<u8> = Vec::new();
    let chunk_size: u64 = 1 << 20;

    // Existing rows: copy descriptors + per-cell compressed bytes.
    for r in 0..last_existing_tile_rows {
        let old_orig_off = r * width_orig;
        new_blob[r * width_orig..r * width_orig + width_orig]
            .copy_from_slice(
                &old_blob.decompressed
                    [old_orig_off..old_orig_off + width_orig]);

        let old_comp_off = old_comp_desc_start + r * 16;
        let (cvlalen_s, cvlastart_s) = read_descriptor(
            'Q', &old_blob.decompressed[old_comp_off..old_comp_off + 16]);
        let cvlalen = cvlalen_s.max(0) as u64;
        let cvlastart_old = cvlastart_s.max(0) as u64;

        let new_cvlastart = if cvlalen == 0 {
            0u64
        } else {
            let src_abs = heap_start_offset + cvlastart_old;
            let dst_abs = heap_start_offset + heap_cursor;
            // Source range [cvlastart_old, +cvlalen) lives in the
            // old heap, which ends at current_pcount (relative to
            // heap_start_offset).  heap_cursor starts at
            // current_pcount and only grows, so dst is always past
            // src — no overlap to worry about.
            let want_total = (dst_abs + cvlalen) - data_offset;
            grow_file_to_at_least(
                file, layout, data_offset, want_total, tainted)?;
            stream_copy_in_file(
                file, src_abs, dst_abs, cvlalen, &mut copy_buf,
                chunk_size, tainted,
                "compressed VLA append: copy existing cell bytes",
            )?;
            let placed = heap_cursor;
            heap_cursor += cvlalen;
            placed
        };
        let new_comp_off = comp_desc_start + r * 16;
        write_descriptor(
            'Q', cvlalen as usize, new_cvlastart as usize,
            &mut new_blob[new_comp_off..new_comp_off + 16],
        );
    }

    // New rows (first `merge_rows` of the input): encode per-cell,
    // original descriptor from the planner.
    for r in 0..merge_rows {
        let input_row_idx = r;
        let cell = col_input.get_item(input_row_idx)?;
        let nelements = validate_vla_cell(
            &cell, ndarray, inner_letter, &col.name, input_row_idx)?;
        let plan = &col_plans[input_row_idx];
        debug_assert_eq!(plan.nelements, nelements);

        let new_orig_off = (last_existing_tile_rows + r) * width_orig;
        write_descriptor(
            descriptor_kind, nelements, plan.bytes_offset_in_heap,
            &mut new_blob[new_orig_off..new_orig_off + width_orig],
        );

        let mut cell_be = vec![0u8; nelements * elem_size];
        if nelements > 0 {
            serialize_vla_cell(
                &cell, inner_letter, nelements, &mut cell_be)?;
        }
        let (cvlalen, cvlastart) = if nelements == 0 {
            (0u64, 0u64)
        } else {
            let compressed = encode_table_column_slab(
                algo, &cell_be, nelements, elem_size,
                rice_blocksize, gzip_level,
            )?;
            let payload = if compressed.len() >= cell_be.len() {
                &cell_be[..]
            } else {
                &compressed[..]
            };
            let plen = payload.len() as u64;
            let want_total = heap_start_offset + heap_cursor
                + plen - data_offset;
            grow_file_to_at_least(
                file, layout, data_offset, want_total, tainted)?;
            {
                let mut g = lock_file(file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.write_all(payload).map_err(|e| {
                    tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "compressed VLA append: merge-tile new-row cell \
                         write failed at col '{}' new-row {}: {}",
                        col.name, r, e))
                })?;
            }
            let placed = heap_cursor;
            heap_cursor += plen;
            (plen, placed)
        };
        let new_comp_off =
            comp_desc_start + (last_existing_tile_rows + r) * 16;
        write_descriptor(
            'Q', cvlalen as usize, cvlastart as usize,
            &mut new_blob[new_comp_off..new_comp_off + 16],
        );
    }
    let _ = py;

    // GZIP_1 the new dual-descriptor blob, write to heap end,
    // record the main-table descriptor for this (tile, col).
    let gzipped = encode_gzip1(&new_blob, None)?;
    let want_total = heap_start_offset + heap_cursor
        + gzipped.len() as u64 - data_offset;
    grow_file_to_at_least(file, layout, data_offset, want_total, tainted)?;
    let blob_heap_offset = heap_cursor;
    {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&gzipped).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed VLA append: merged dual-descriptor blob \
                 write failed at tile {} col '{}': {}",
                tile_idx, col.name, e))
        })?;
    }
    heap_cursor += gzipped.len() as u64;

    let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
    let nelems_be = (gzipped.len() as i64).to_be_bytes();
    let off_be = (blob_heap_offset as i64).to_be_bytes();
    desc_table[desc_off..desc_off + 8].copy_from_slice(&nelems_be);
    desc_table[desc_off + 8..desc_off + 16].copy_from_slice(&off_be);

    Ok(heap_cursor)
}

// ---------------------------------------------------------------------------
// Phase 6b — append rows to a compressed table
// ---------------------------------------------------------------------------
//
// Mechanics:
//   1. Compute layout: how many rows merge into the existing partial
//      last tile vs become fresh full tiles.
//   2. Grow the descriptor table by `added_n_tiles *
//      descriptor_row_width` bytes via a forward file-tail shift —
//      this also shifts the existing heap forward by the same delta.
//      Descriptor heap-offsets stay valid because they're heap-
//      relative.
//   3. Read the (now-larger) descriptor table into RAM so the merge
//      branch can rewrite the last-tile slot and the new-tile
//      branch can fill the just-freed slots.
//   4. If merging: decode the existing last tile's per-column BE
//      bytes (the Phase 2 read path with the final byteswap-to-
//      native skipped), concatenate the first M new rows (via the
//      shared per-cell transform), re-encode, append blobs to the
//      heap end.  Old last-tile blobs become orphans.
//   5. For remaining rows: encode as fresh tiles, write blobs to
//      heap, fill the new descriptor rows.
//   6. Write back the descriptor table.
//   7. Update header: NAXIS2 (n_tiles), PCOUNT (heap size),
//      ZNAXIS2 (original nrows).
//   8. Clear the tile cache — the last-tile entries are stale and
//      it's cheaper to drop everything than to do per-entry
//      invalidation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_compressed_table_data(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    per_column_inputs: &[Bound<'_, PyAny>],
    columns: &[Column],
    algorithms: &[CompressionAlgorithm],
    per_col_configs: Option<&[CompressionConfigKind]>,
    existing_nrows: usize,
    ztilelen: usize,
    existing_n_tiles: usize,
    descriptor_row_width: usize,
    data_offset: u64,
    current_pcount: u64,
    cache: &ColumnTileCache,
) -> PyResult<()> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    crate::common::check_not_tainted(&super_.tainted)?;

    // Determine append size from the first per-column ndarray's
    // shape[0].  All inputs must have matching first axis.
    if per_column_inputs.is_empty() {
        return Err(PyValueError::new_err(
            "CompressedTableHDU.append: no columns to write"));
    }
    let append_nrows: usize = per_column_inputs[0]
        .getattr("shape")?.extract::<Vec<usize>>()?
        .first().copied()
        .ok_or_else(|| PyValueError::new_err(
            "append: input shape is empty"))?;
    if append_nrows == 0 {
        return Ok(());
    }
    for (col, arr) in columns.iter().zip(per_column_inputs.iter()) {
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        if shape.is_empty() || shape[0] != append_nrows {
            return Err(PyValueError::new_err(format!(
                "CompressedTableHDU.append: column '{}' input shape \
                 {:?} does not have first axis == {}",
                col.name, shape, append_nrows)));
        }
    }

    let new_nrows = existing_nrows + append_nrows;
    let new_n_tiles = if new_nrows == 0 {
        0
    } else {
        new_nrows.div_ceil(ztilelen.max(1))
    };
    let added_n_tiles = new_n_tiles - existing_n_tiles;
    let last_existing_tile_rows = if existing_n_tiles == 0 {
        0
    } else {
        existing_nrows - (existing_n_tiles - 1) * ztilelen
    };
    let room_in_last_tile = if existing_n_tiles > 0
        && last_existing_tile_rows < ztilelen
    {
        ztilelen - last_existing_tile_rows
    } else {
        0
    };
    let merge_rows = append_nrows.min(room_in_last_tile);
    let _rows_in_new_tiles = append_nrows - merge_rows;

    // Per-column prep.  Fixed cols get a ColPrep; VLA cols get
    // a None slot — their per-cell work happens later in
    // encode_vla_column_tile{,_with_merge}.  VLA-input validation
    // (Object dtype, length match) happens here so dtype errors
    // raise before any file mutation.
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let mut preps: Vec<Option<ColPrep<'_>>> = Vec::with_capacity(columns.len());
    for (i, (col, arr)) in columns.iter()
        .zip(per_column_inputs.iter()).enumerate()
    {
        if col.var_kind.is_some() {
            if !arr.is_instance(&ndarray)? {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.append: column '{}' value must \
                     be a numpy ndarray", col.name)));
            }
            let kind: String = arr.getattr("dtype")?
                .getattr("kind")?.extract()?;
            if kind != "O" {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.append: VLA column '{}' input \
                     must be a numpy Object dtype ndarray (kind 'O'), \
                     got kind '{}'", col.name, kind)));
            }
            preps.push(None);
            continue;
        }
        let cfg = per_col_configs.and_then(|cs| cs.get(i));
        preps.push(Some(prepare_fixed_column(
            &np, &ndarray, arr, col, append_nrows, cfg,
        )?));
    }

    let any_vla = columns.iter().any(|c| c.var_kind.is_some());
    let current_zpcount = parse_keyword(cards, "ZPCOUNT")
        .unwrap_or(0).max(0) as u64;

    // Plan VLA original-heap offsets for the full input batch, so
    // the original-descriptor offsets (used by funpack to
    // reconstruct) extend the existing original heap.  Returns
    // per-column per-row plans + the new total original-heap size.
    let (vla_plans, new_orig_pcount) = if any_vla {
        let (plans, cursor) = plan_vla_heap_layout(
            columns, per_column_inputs, append_nrows, &ndarray,
            current_zpcount as usize,
        )?;
        (plans, cursor as u64)
    } else {
        (Vec::new(), current_zpcount)
    };

    // Step 1: decode the existing last tile if we're going to
    // merge into it.  Do this BEFORE any file mutation — the
    // heap-offset math depends on the current layout.  Fixed cols
    // need their BE-bytes (decoded slab); VLA cols need their
    // dual-descriptor blob (decompressed) so we can copy the
    // existing per-row descriptors and per-cell compressed bytes.
    let mut existing_be_per_col: Vec<Vec<u8>> = Vec::new();
    let mut vla_merge_blobs: Vec<Option<VlaMergeOldBlob>> = Vec::new();
    if merge_rows > 0 {
        let last_tile_idx = existing_n_tiles - 1;
        existing_be_per_col.reserve(columns.len());
        vla_merge_blobs.reserve(columns.len());
        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_some() {
                existing_be_per_col.push(Vec::new());
                vla_merge_blobs.push(Some(read_vla_merge_old_blob(
                    &super_.file, data_offset, last_tile_idx, col_idx,
                    col, last_existing_tile_rows, existing_n_tiles,
                    descriptor_row_width,
                )?));
            } else {
                existing_be_per_col.push(decode_existing_tile_to_be_bytes(
                    &super_.file, cards, data_offset, last_tile_idx,
                    col_idx, col, algorithms[col_idx],
                    last_existing_tile_rows, descriptor_row_width,
                )?);
                vla_merge_blobs.push(None);
            }
        }
    }

    // Step 2: grow the data section to make room for the new
    // descriptor rows + existing heap.  grow_file_to_at_least
    // rounds to the next BLOCK_SIZE boundary and either set_len's
    // the file (last HDU) or shifts later HDUs forward (non-last);
    // either way the data section ends at a block boundary so
    // subsequent HDU headers stay aligned.  Further heap growth
    // during the encode loop is handled by additional
    // grow_file_to_at_least calls, also block-aligned.
    let delta_desc_bytes = (added_n_tiles as u64)
        * (descriptor_row_width as u64);
    let new_descs_bytes = (new_n_tiles as u64)
        * (descriptor_row_width as u64);
    let want_data_bytes = new_descs_bytes + current_pcount;
    grow_file_to_at_least(
        &super_.file, &super_.layout, data_offset,
        want_data_bytes, &super_.tainted,
    )?;

    // Step 2b: relocate the existing heap forward by
    // delta_desc_bytes (within the file) so it sits right after
    // the new (larger) descriptor table.  Descriptor heap-offsets
    // are heap-relative so they stay valid through the move.
    if delta_desc_bytes > 0 && current_pcount > 0 {
        let old_heap_start = data_offset
            + (existing_n_tiles as u64) * (descriptor_row_width as u64);
        let new_heap_start_local = data_offset + new_descs_bytes;
        relocate_region_forward_local(
            &super_.file, old_heap_start, new_heap_start_local,
            current_pcount, &super_.tainted,
        )?;
    }
    let new_heap_start = data_offset + new_descs_bytes;

    // Step 3: read the existing descriptor table into RAM so we
    // can modify the last-tile entries (merge case) and write new
    // descriptor rows for the appended tiles.  After the shift
    // above, the new descriptor table region is (existing_rows ||
    // zero-shifted-stale-bytes); we overwrite the zero-shifted
    // region with the new descriptors below.
    let desc_table_size = new_n_tiles * descriptor_row_width;
    let mut desc_table = vec![0u8; desc_table_size];
    if existing_n_tiles > 0 {
        let existing_desc_size =
            existing_n_tiles * descriptor_row_width;
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut desc_table[..existing_desc_size])
            .map_err(|e| PyIOError::new_err(format!(
                "append: read existing descriptor table failed: {}", e)))?;
    }

    // Heap cursor starts at current PCOUNT — we append new blobs
    // to the heap end (orphaning old last-tile blobs on merge).
    let mut heap_cursor = current_pcount;

    // Step 4: merge into last tile if applicable.  Fixed cols
    // concatenate freshly-transformed new rows onto the decoded
    // existing slab, then re-encode.  VLA cols copy each existing
    // row's per-cell compressed bytes verbatim (no decode / re-
    // encode) and append per-cell encoded bytes for new rows;
    // existing rows' original-descriptor offsets are preserved so
    // funpack's reconstructed heap stays consistent.
    if merge_rows > 0 {
        let last_tile_idx = existing_n_tiles - 1;
        let merged_rows = last_existing_tile_rows + merge_rows;
        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_some() {
                let cfg = per_col_configs.and_then(|cs| cs.get(col_idx));
                heap_cursor = encode_vla_column_tile_with_merge(
                    py, &ndarray, &super_.file, &super_.layout,
                    &super_.tainted, data_offset, new_heap_start,
                    heap_cursor, col, &per_column_inputs[col_idx],
                    &vla_plans[col_idx], last_tile_idx,
                    last_existing_tile_rows, merge_rows,
                    vla_merge_blobs[col_idx].as_ref().expect(
                        "VLA col has a merge-blob"),
                    algorithms[col_idx],
                    cfg.map(rice_blocksize_of).unwrap_or(32),
                    cfg.and_then(gzip_level_of),
                    &mut desc_table, col_idx, descriptor_row_width,
                )?;
                continue;
            }
            let prep = preps[col_idx].as_ref()
                .expect("non-VLA col has a ColPrep");
            let mut merged = existing_be_per_col[col_idx].clone();
            merged.reserve(merge_rows * prep.per_row_bytes);
            let src_bytes = prep.buf.as_slice();
            for r in 0..merge_rows {
                let src_off = r * prep.src_total_size;
                let src = &src_bytes
                    [src_off..src_off + prep.src_total_size];
                let mut cell = vec![0u8; prep.per_row_bytes];
                apply_transform_cell(
                    &prep.transform, src, &mut cell, &col.name, r)?;
                merged.extend_from_slice(&cell);
            }
            let n_pixels = merged_rows * prep.per_row_pixels;
            heap_cursor = encode_be_slab_to_heap_and_record(
                &merged, n_pixels, algorithms[col_idx],
                prep.elem_size, prep.rice_blocksize, prep.gzip_level,
                last_tile_idx, col_idx, &col.name, descriptor_row_width,
                new_heap_start, heap_cursor, &mut desc_table,
                &super_.file, &super_.layout, data_offset, &super_.tainted,
            )?;
        }
    }

    // Step 5: encode fresh tiles for any remaining rows.  Fixed
    // cols go through the shared helper; VLA cols reuse the
    // Phase 6a per-tile encoder with the planned original-heap
    // offsets (which extend past current_zpcount).
    let mut new_input_row_cursor = merge_rows;
    for new_tile_offset in 0..added_n_tiles {
        let tile_idx = existing_n_tiles + new_tile_offset;
        let tile_row_start_in_new = new_input_row_cursor;
        let rows_in_tile = if new_tile_offset + 1 == added_n_tiles {
            append_nrows - tile_row_start_in_new
        } else {
            ztilelen
        };
        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_some() {
                let cfg = per_col_configs.and_then(|cs| cs.get(col_idx));
                heap_cursor = encode_vla_column_tile(
                    py, &ndarray, &super_.file, &super_.layout,
                    &super_.tainted, data_offset, new_heap_start,
                    heap_cursor, col, &per_column_inputs[col_idx],
                    &vla_plans[col_idx], tile_row_start_in_new,
                    rows_in_tile, algorithms[col_idx],
                    cfg.map(rice_blocksize_of).unwrap_or(32),
                    cfg.and_then(gzip_level_of),
                    &mut desc_table, tile_idx, col_idx,
                    descriptor_row_width,
                )?;
                continue;
            }
            let prep = preps[col_idx].as_ref()
                .expect("non-VLA col has a ColPrep");
            heap_cursor = build_and_encode_tile_col(
                prep, col, algorithms[col_idx],
                tile_idx, col_idx, rows_in_tile,
                /* source_row_offset = */ tile_row_start_in_new,
                descriptor_row_width, new_heap_start, heap_cursor,
                &mut desc_table, &super_.file, &super_.layout,
                data_offset, &super_.tainted,
            )?;
        }
        new_input_row_cursor += rows_in_tile;
    }

    // grow_file_to_at_least keeps the data section block-aligned;
    // the bytes between heap_cursor and the block boundary are
    // either zero (last HDU, set_len from OS) or HDU 2 header
    // bytes that were shifted into place (non-last HDU).  Either
    // way, don't overwrite them.

    // Step 5: write the updated descriptor table.
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&desc_table).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "append: descriptor-table write failed: {}", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!("append: flush failed: {}", e))
        })?;
    }

    // Step 6: update header.  NAXIS2 = new_n_tiles, PCOUNT =
    // heap_cursor, ZNAXIS2 = new_nrows.  ZPCOUNT only matters
    // when any VLA col is present — it's the original
    // (uncompressed) heap size and funpack copies it onto the
    // output PCOUNT.  For fixed-only tables ZPCOUNT stays 0.
    let cards_guard = super_.cards_write_lock()?;
    let mut new_cards = cards.to_vec();
    update_int_card_in_place(
        &mut new_cards, "NAXIS2", new_n_tiles as i64,
        "number of tiles")?;
    update_int_card_in_place(
        &mut new_cards, "ZNAXIS2", new_nrows as i64,
        "original (uncompressed) row count")?;
    crate::hdu_table::set_pcount_in_cards(&mut new_cards, heap_cursor);
    if any_vla {
        set_zpcount_in_cards(&mut new_cards, new_orig_pcount);
    }
    crate::header::rewrite_header_to_disk(
        &super_.file, &super_.offsets, &super_.layout,
        &new_cards, &super_.tainted,
    )?;
    cards_guard.commit(new_cards);

    // Step 7: invalidate the cache.  The merged tile's entries
    // are stale; cheapest correct option is a full clear (cache
    // re-warms on the next read).  For append without merge the
    // existing entries stay valid, but we clear anyway to keep
    // the logic simple — append is a rare-vs-read operation.
    cache.clear();
    Ok(())
}

// Find and rewrite an int-valued structural card.  Used by append
// for NAXIS2 and ZNAXIS2 (PCOUNT goes through set_pcount_in_cards).
fn update_int_card_in_place(
    cards: &mut [String], keyword: &str, value: i64, comment: &str,
) -> PyResult<()> {
    use crate::header::card_int;
    let new_card = card_int(keyword, value, comment).trim_end().to_string();
    let kw_len = keyword.len();
    for card in cards.iter_mut() {
        if card.len() >= kw_len && card[..kw_len].trim_end() == keyword
            && (card.len() == kw_len
                || !card[kw_len..kw_len + 1].chars().next().unwrap().is_ascii_digit())
        {
            *card = new_card;
            return Ok(());
        }
    }
    Err(PyValueError::new_err(format!(
        "append: header missing required keyword {}", keyword)))
}

// Decode one (tile, col) blob back to FITS big-endian bytes —
// the slab format we'd hand to encode_table_column_slab.  Mirrors
// Phase 2's read path but stops before the byteswap-to-native that
// convert_column_cell does.
// Move `total` bytes WITHIN a file from `src_start` to `dst_start`,
// where `dst_start > src_start` (forward move).  Back-to-front
// chunked copy so the overlapping case is safe (later bytes read
// before they're overwritten by the move of earlier ones).  No
// layout offset updates — purely a within-file relocation, used by
// append to slide the existing heap forward inside the (already
// grown) data section to make room for new descriptor rows.
fn relocate_region_forward_local(
    file: &FileHandle,
    src_start: u64,
    dst_start: u64,
    total: u64,
    tainted: &TaintFlag,
) -> PyResult<()> {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    if total == 0 || src_start == dst_start {
        return Ok(());
    }
    let chunk_size: u64 = 1 << 20;  // 1 MiB
    let mut buf = vec![0u8; chunk_size as usize];
    let mut remaining = total;
    while remaining > 0 {
        let n = remaining.min(chunk_size);
        let src_off = src_start + remaining - n;
        let dst_off = dst_start + remaining - n;
        buf.resize(n as usize, 0);
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(src_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "append: heap relocate read failed: {}; \
                 close + reopen", e))
        })?;
        f.seek(SeekFrom::Start(dst_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "append: heap relocate write failed: {}; \
                 close + reopen", e)));
        }
        remaining -= n;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_existing_tile_to_be_bytes(
    file: &FileHandle,
    cards: &[String],
    data_offset: u64,
    tile_idx: usize,
    col_idx: usize,
    col: &Column,
    algo: CompressionAlgorithm,
    rows_in_tile: usize,
    descriptor_row_width: usize,
) -> PyResult<Vec<u8>> {
    // Heap base relative to data_offset.  Default = NAXIS1*NAXIS2.
    let theap_raw = parse_keyword(cards, "THEAP").unwrap_or(0);
    let heap_base_in_data = if theap_raw > 0 {
        theap_raw as u64
    } else {
        let n_tiles = parse_keyword(cards, "NAXIS2")
            .unwrap_or(0).max(0) as u64;
        let row_width = parse_keyword(cards, "NAXIS1")
            .unwrap_or(0).max(0) as u64;
        n_tiles * row_width
    };
    let heap_start = data_offset + heap_base_in_data;

    // Read descriptor at (tile_idx, col_idx).
    let mut desc_buf = [0u8; 16];
    {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let off = data_offset
            + (tile_idx as u64) * (descriptor_row_width as u64)
            + (col_idx as u64) * 16;
        f.seek(SeekFrom::Start(off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut desc_buf).map_err(|e| {
            PyIOError::new_err(format!(
                "append/decode: read descriptor failed: {}", e))
        })?;
    }
    let (nelems_s, heap_offset_s) = read_descriptor('Q', &desc_buf);
    if nelems_s < 0 || heap_offset_s < 0 {
        return Err(PyValueError::new_err(format!(
            "append/decode: tile {} col '{}': descriptor has negative \
             field (nelements={}, offset={})",
            tile_idx, col.name, nelems_s, heap_offset_s)));
    }
    let n_bytes_compressed = nelems_s as usize;
    let mut compressed = vec![0u8; n_bytes_compressed];
    if n_bytes_compressed > 0 {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(heap_start + heap_offset_s as u64))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut compressed).map_err(|e| {
            PyIOError::new_err(format!(
                "append/decode: read heap failed: {}", e))
        })?;
    }
    decompress_column_slab(algo, &compressed, col, rows_in_tile)
}

