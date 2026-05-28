// Compressed-table bulk write: write_compressed_table_data plus the
// file-grow + ZPCOUNT + VLA-tile encode helpers it relies on.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::io::{Seek, SeekFrom};
use std::sync::Arc;

use crate::common::{
    lock_file,
    FileHandle, FileLayout, TaintFlag,
};
use crate::zimage::compression_config::CompressionConfigKind;
use crate::hdu::HDU;
use crate::hdu_table::{
    bytes_per_element, plan_vla_heap_layout,
    serialize_vla_cell, validate_vla_cell, write_descriptor, Column, VlaCellPlan,
};
use crate::zimage::CompressionAlgorithm;

use super::write_setup::{
    ColPrep, build_and_encode_tile_col,
    encode_table_column_slab, gzip_level_of, prepare_fixed_column,
    rice_blocksize_of,
};

// ---------------------------------------------------------------------------
// Phase 5 — write loop
// ---------------------------------------------------------------------------
//
// Encode each column's per-tile slab, stream blob bytes to the
// heap, fill the descriptor table in RAM, then seek back and
// write the descriptor table + update PCOUNT + grow the file
// extent as needed.  Validate-then-mutate: any dtype/shape error
// surfaces before the file is touched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_compressed_table_data<'py>(
    py: Python<'py>,
    super_: &HDU,
    cards: &[String],
    per_column_inputs: &[Bound<'py, PyAny>],
    columns: &[Column],
    algorithms: &[CompressionAlgorithm],
    per_col_configs: Option<&[CompressionConfigKind]>,
    nrows: usize,
    ztilelen: usize,
    n_tiles: usize,
    descriptor_row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    use crate::hdu_image::round_up_to_block;
    use std::io::Write;
    use std::sync::atomic::Ordering;

    crate::common::check_not_tainted(&super_.tainted)?;

    if per_column_inputs.len() != columns.len() {
        return Err(PyValueError::new_err(format!(
            "internal: per-column inputs len {} != columns len {}",
            per_column_inputs.len(), columns.len())));
    }

    // Pre-plan the original heap layout for VLA columns: cfitsio's
    // funpack (and our Phase 4 read for that matter) puts the
    // *original* descriptors' offsets in the per-tile dual-
    // descriptor blob, and funpack uses them to place each cell at
    // its original-heap position when reconstructing the
    // uncompressed table.  If we leave them at 0 all cells of a
    // column collide at heap offset 0 on funpack output.  Match
    // the layout the uncompressed VLA write would produce
    // (`plan_vla_heap_layout`) so a funpack-decompressed file is
    // byte-equivalent to a fresh `create_table_hdu` + `write`
    // without compress.  The cursor return is the total original
    // heap size — emitted as ZPCOUNT below so funpack's
    // `fits_uncompress_table` (which copies ZPCOUNT → output
    // PCOUNT) gets the right heap extent.
    let np_for_plan = py.import("numpy")?;
    let ndarray_for_plan = np_for_plan.getattr("ndarray")?;
    let (vla_plans, original_pcount) = if columns.iter()
        .any(|c| c.var_kind.is_some())
    {
        let (plans, cursor) = plan_vla_heap_layout(
            columns, per_column_inputs, nrows, &ndarray_for_plan, 0,
        )?;
        (plans, cursor as u64)
    } else {
        (Vec::new(), 0u64)
    };

    // Per-column setup.  Fixed cols go through the shared
    // `prepare_fixed_column` helper; VLA cols are validated to be
    // Object-dtype ndarrays and their per-row cells are handled
    // lazily inside the tile loop via `encode_vla_column_tile`.
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let mut preps: Vec<Option<ColPrep<'_>>> =
        Vec::with_capacity(columns.len());
    for (i, (col, arr)) in columns.iter()
        .zip(per_column_inputs.iter()).enumerate()
    {
        if col.var_kind.is_some() {
            // VLA: validate Object-dtype ndarray with the right
            // length; deeper validation happens per cell.
            if !arr.is_instance(&ndarray)? {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.write: column '{}' value must \
                     be a numpy ndarray", col.name)));
            }
            let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
            if shape.is_empty() || shape[0] != nrows {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.write: column '{}' shape {:?} \
                     does not have first axis == ZNAXIS2 ({})",
                    col.name, shape, nrows)));
            }
            let dtype = arr.getattr("dtype")?;
            let kind: String = dtype.getattr("kind")?.extract()?;
            if kind != "O" {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.write: VLA column '{}' input \
                     must be a numpy Object dtype ndarray (kind 'O'), \
                     got kind '{}'", col.name, kind)));
            }
            preps.push(None);
            continue;
        }
        let cfg = per_col_configs.and_then(|cs| cs.get(i));
        preps.push(Some(prepare_fixed_column(
            &np, &ndarray, arr, col, nrows, cfg,
        )?));
    }

    // Stream-encode tile by tile, writing each blob to the heap as
    // it's produced.  Descriptor table is held in RAM (small:
    // n_tiles * ncols * 16 bytes; typically a few KB) and written
    // at the end with one seek-back.
    let mut desc_table: Vec<u8> = vec![0u8; n_tiles * descriptor_row_width];
    let heap_start_offset = data_offset
        + (n_tiles as u64 * descriptor_row_width as u64);
    let mut heap_cursor: u64 = 0;

    // Grow the file extent so we have room for the descriptor table
    // upfront.  The heap grows it further below.
    let current_padded = round_up_to_block(
        (n_tiles as u64) * (descriptor_row_width as u64));
    {
        // Allocate the initial descriptor space within this HDU.
        let mut guard = lock_file(&super_.file)?;
        let f = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let current_end = data_offset + current_padded;
        let file_len = f.metadata()
            .map_err(|e| PyIOError::new_err(e.to_string()))?.len();
        if file_len < current_end {
            f.set_len(current_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    for tile_idx in 0..n_tiles {
        let tile_row_start = tile_idx * ztilelen;
        let rows_in_tile = if tile_idx + 1 == n_tiles {
            nrows - tile_row_start
        } else {
            ztilelen
        };

        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_some() {
                // VLA dual-descriptor blob path.
                let cfg = per_col_configs.and_then(|cs| cs.get(col_idx));
                heap_cursor = encode_vla_column_tile(
                    py, &ndarray, &super_.file, &super_.layout,
                    &super_.tainted, data_offset, heap_start_offset,
                    heap_cursor, col, &per_column_inputs[col_idx],
                    &vla_plans[col_idx], tile_row_start, rows_in_tile,
                    algorithms[col_idx],
                    cfg.map(rice_blocksize_of).unwrap_or(32),
                    cfg.and_then(gzip_level_of),
                    &mut desc_table, tile_idx, col_idx,
                    descriptor_row_width,
                )?;
                continue;
            }
            // Fixed-column path: per-cell transform → encode → write
            // → record, all in the shared helper.
            let prep = preps[col_idx].as_ref()
                .expect("preps[i] is Some for non-VLA columns");
            heap_cursor = build_and_encode_tile_col(
                prep, col, algorithms[col_idx],
                tile_idx, col_idx, rows_in_tile,
                /* source_row_offset = */ tile_row_start,
                descriptor_row_width, heap_start_offset, heap_cursor,
                &mut desc_table, &super_.file, &super_.layout,
                data_offset, &super_.tainted,
            )?;
        }
    }

    // One round-up to the FITS block boundary so the data section
    // ends cleanly.  The grow helper already extends to multiples
    // of BLOCK_SIZE, but if heap_cursor isn't a multiple of
    // BLOCK_SIZE we need to make sure the tail is zero-filled.
    let total_data_bytes = (n_tiles as u64 * descriptor_row_width as u64)
        + heap_cursor;
    let padded = round_up_to_block(total_data_bytes);
    if padded > total_data_bytes {
        let pad = padded - total_data_bytes;
        let mut guard = lock_file(&super_.file)?;
        let f = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset + total_data_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&vec![0u8; pad as usize]).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: tail-pad write failed: {}", e))
        })?;
    }

    // Write the descriptor table at the start of the data section.
    {
        let mut guard = lock_file(&super_.file)?;
        let f = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&desc_table).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: descriptor-table write \
                 failed: {}", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: flush failed: {}", e))
        })?;
    }

    // Update PCOUNT (compressed heap size) AND ZPCOUNT (the
    // original-table heap size).  cfitsio's `fits_uncompress_table`
    // copies ZPCOUNT verbatim onto the output uncompressed PCOUNT
    // and uses it to set the heap extent; leaving it at 0 makes
    // funpack truncate the heap to zero even though the descriptors
    // point at real data.  For fixed-only tables ZPCOUNT stays 0
    // (no original heap to size).
    let cards_guard = super_.cards_write_lock()?;
    let mut new_cards = cards.to_vec();
    crate::hdu_table::set_pcount_in_cards(&mut new_cards, heap_cursor);
    set_zpcount_in_cards(&mut new_cards, original_pcount);
    crate::header::rewrite_header_to_disk(
        &super_.file, &super_.offsets, &super_.layout,
        &new_cards, &super_.tainted,
    )?;
    cards_guard.commit(new_cards);
    Ok(())
}

// Rewrite (or insert) the ZPCOUNT card to `new_value`.  ZPCOUNT
// records the ORIGINAL (uncompressed) table's PCOUNT — funpack
// copies it to the output PCOUNT during decompression.  Always
// present in headers we emit (created in build_compressed_table_header
// with the placeholder value 0); this update fills it in once the
// VLA write knows the real original-heap size.
pub(crate) fn set_zpcount_in_cards(new_cards: &mut Vec<String>, new_value: u64) {
    use crate::header::card_int;
    let card = card_int(
        "ZPCOUNT", new_value as i64,
        "original heap size (0 for fixed-only tables)",
    );
    let trimmed = card.trim_end().to_string();
    if let Some(idx) = new_cards.iter().position(|c|
        c.len() >= 7 && c[..7].trim() == "ZPCOUNT")
    {
        new_cards[idx] = trimmed;
    } else {
        // Defensive — every ZTABLE header we emit has ZPCOUNT.
        // Insert just before END if somehow missing.
        let end_idx = new_cards.iter().position(|c|
            c.len() >= 3 && c[..3].trim() == "END")
            .unwrap_or(new_cards.len());
        new_cards.insert(end_idx, trimmed);
    }
}

// Grow this HDU's data extent so it covers at least `min_bytes`
// (relative to data_offset).  Block-rounds; pushes later HDUs
// forward via the shared shift primitive when needed.  No-op when
// this HDU's data section already extends past `want_end`.
//
// For non-last HDUs the upper bound is the NEXT HDU's start, not
// the file length — file length includes trailing HDU bytes that
// belong to those HDUs and must not be overwritten.  Bare file-
// length as the cap (the original buggy form) silently passes
// writes that overlap a trailing HDU's region whenever the
// growth fits in the block-alignment padding, and corrupts the
// trailing HDU once growth exceeds the padding.
pub(crate) fn grow_file_to_at_least(
    file: &FileHandle,
    layout: &Arc<FileLayout>,
    data_offset: u64,
    min_bytes: u64,
    tainted: &TaintFlag,
) -> PyResult<()> {
    use crate::common::shift_file_tail_and_update_offsets;
    use crate::hdu_image::round_up_to_block;
    use std::sync::atomic::Ordering;

    let want_end = data_offset + round_up_to_block(min_bytes);
    let next_hdu_start = {
        let guard = layout.hdus.lock()
            .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
        guard.iter()
            .map(|o| o.header_offset())
            .filter(|&off| off > data_offset)
            .min()
    };
    let file_len = {
        let g = lock_file(file)?;
        let f = g.as_ref()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.metadata()
            .map_err(|e| PyIOError::new_err(e.to_string()))?.len()
    };
    // Effective end is bounded by the next HDU's start (non-last)
    // or by the file length (last HDU).
    let effective_end = next_hdu_start.unwrap_or(file_len);
    if want_end <= effective_end {
        return Ok(());
    }
    let delta = want_end - effective_end;
    if next_hdu_start.is_none() {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.set_len(want_end).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: set_len({}) failed: {}",
                want_end, e))
        })?;
    } else {
        shift_file_tail_and_update_offsets(
            file, layout, effective_end, delta, tainted,
        )?;
    }
    Ok(())
}

// Encode one VLA column for one tile.  Per row in the tile: compress
// the cell's BE bytes per ZCTYPn (with cfitsio's uncompressed
// fallback when compression doesn't shrink the data); record the
// (nelements_orig, original_offset_unused) descriptor in the first
// half of the dual-descriptor blob and the (cvlalen, cvlastart)
// compressed descriptor in the second half.  Then GZIP_1 the blob,
// write it to the heap, and fill the main-table 1QB descriptor for
// this (tile, col).
//
// Returns the updated `heap_cursor` (one heap slot ahead of where
// the last compressed cell bytes ended).
//
// Original descriptors use whatever P/Q kind the column's TFORMn
// declares (so reads via Phase 4 see the original layout); the
// compressed descriptors are always Q (16 bytes) per the cfitsio
// reference encoder.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_vla_column_tile(
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
    tile_row_start: usize,
    rows_in_tile: usize,
    algo: CompressionAlgorithm,
    rice_blocksize: u32,
    gzip_level: Option<u32>,
    desc_table: &mut [u8],
    tile_idx: usize,
    col_idx: usize,
    descriptor_row_width: usize,
) -> PyResult<u64> {
    use crate::zimage::gzip::encode_gzip1;
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let inner_letter = col.tform_letter;
    let elem_size = bytes_per_element(inner_letter)
        .ok_or_else(|| PyValueError::new_err(format!(
            "VLA column '{}': unsupported inner letter '{}'",
            col.name, inner_letter)))?;
    let descriptor_kind = col.var_kind
        .expect("encode_vla_column_tile called for non-VLA column");
    let width_orig = if descriptor_kind == 'P' { 8 } else { 16 };

    let blob_size = rows_in_tile * width_orig + rows_in_tile * 16;
    let mut descriptor_blob = vec![0u8; blob_size];
    let comp_desc_start = rows_in_tile * width_orig;

    // Per-row: validate cell, serialize to BE bytes, compress (with
    // uncompressed fallback), append to heap.
    for r in 0..rows_in_tile {
        let disk_row = tile_row_start + r;
        let cell = col_input.get_item(disk_row)?;
        let nelements = validate_vla_cell(
            &cell, ndarray, inner_letter, &col.name, disk_row)?;

        let mut cell_bytes_be = vec![0u8; nelements * elem_size];
        if nelements > 0 {
            serialize_vla_cell(
                &cell, inner_letter, nelements, &mut cell_bytes_be)?;
        }

        let (cvlalen, cvlastart) = if nelements == 0 {
            // Empty cell: no heap write, descriptors both (0, 0).
            (0u64, 0u64)
        } else {
            // Try compressing.  If the result isn't smaller than
            // the raw cell, fall back to writing the raw BE bytes
            // (cfitsio's `compressed_size < uncompressed_size`
            // check; see imcompress.c around line 8508).  Phase 4
            // read handles this fallback by detecting cvlalen ==
            // vlalen * elem_size.
            let compressed = encode_table_column_slab(
                algo, &cell_bytes_be, nelements, elem_size,
                rice_blocksize, gzip_level,
            )?;
            let payload = if compressed.len() >= cell_bytes_be.len() {
                &cell_bytes_be[..]
            } else {
                &compressed[..]
            };
            let plen = payload.len() as u64;
            let want_total =
                heap_start_offset + heap_cursor + plen - data_offset;
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
                        "compressed VLA write: cell heap write failed at \
                         tile {} col '{}' row {}: {}",
                        tile_idx, col.name, disk_row, e))
                })?;
            }
            let cvlastart = heap_cursor;
            heap_cursor += plen;
            (plen, cvlastart)
        };

        // Original descriptor (matches user-visible P or Q layout):
        // nelements_orig + planned_original_offset.  We use the
        // offset `plan_vla_heap_layout` would assign for a fresh
        // uncompressed write of the same data — funpack (cfitsio's
        // decompressor) uses this field to place the cell at the
        // right position in the reconstructed uncompressed heap, so
        // 0-for-all would collide every cell at offset 0.  Our own
        // Phase 4 read ignores this field (it only uses the
        // compressed descriptor's cvlalen/cvlastart) so the value
        // doesn't matter for rustfits round trips — but cross-tool
        // interop demands a consistent layout.
        let plan = &col_plans[tile_row_start + r];
        write_descriptor(
            descriptor_kind, nelements, plan.bytes_offset_in_heap,
            &mut descriptor_blob[r * width_orig..r * width_orig + width_orig],
        );
        // Compressed descriptor (always Q, 16 bytes).
        write_descriptor(
            'Q', cvlalen as usize, cvlastart as usize,
            &mut descriptor_blob[comp_desc_start + r * 16
                ..comp_desc_start + r * 16 + 16],
        );
    }
    let _ = py;  // py-handle no longer needed past validate_vla_cell

    // GZIP_1 the dual-descriptor blob — this is always GZIP_1
    // regardless of ZCTYPn (Phase 4 read decompresses via raw gzip).
    let gzipped = encode_gzip1(&descriptor_blob, None)?;

    let want_total =
        heap_start_offset + heap_cursor + gzipped.len() as u64 - data_offset;
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
                "compressed VLA write: descriptor-blob heap write failed \
                 at tile {} col '{}': {}", tile_idx, col.name, e))
        })?;
    }
    heap_cursor += gzipped.len() as u64;

    // Main-table descriptor for this (tile, col): the blob's size +
    // offset.  Two big-endian i64.
    let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
    let nelems_be = (gzipped.len() as i64).to_be_bytes();
    let off_be = (blob_heap_offset as i64).to_be_bytes();
    desc_table[desc_off..desc_off + 8].copy_from_slice(&nelems_be);
    desc_table[desc_off + 8..desc_off + 16].copy_from_slice(&off_be);

    Ok(heap_cursor)
}

