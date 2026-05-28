// Compressed-table __setitem__ machinery: the shared per-tile column
// writer, the VLA tile writer, SetItemCtx, and the row/value coercion
// + resolution helpers.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use std::io::{Read, Seek, SeekFrom};

use crate::common::{
    lock_file, parse_keyword,
};
use crate::zimage::compression_config::CompressionConfigKind;
use crate::hdu::HDU;
use crate::hdu_table::{
    apply_transform_cell,
    bytes_per_element, column_expected_shape, field_dtype_and_shape,
    serialize_vla_cell, validate_vla_cell, write_descriptor, Column,
};
use crate::zimage::CompressionAlgorithm;

use super::append::decode_existing_tile_to_be_bytes;
use super::hdu::ColumnTileCache;
use super::read::{gzip_decompress_bytes};
use super::write::{grow_file_to_at_least, set_zpcount_in_cards};
use super::write_setup::{
    ColPrep, encode_be_slab_to_heap_and_record, encode_table_column_slab,
    gzip_level_of, prepare_fixed_column, rice_blocksize_of,
};

// ---------------------------------------------------------------------------
// Phase 6c-2b / 6c-2c — __setitem__ primitive on compressed tables
// ---------------------------------------------------------------------------
//
// Modify a selected set of (row, column) cells of a compressed
// fixed-column table by re-encoding the affected tiles.  For each
// tile that contains any modified row, the SELECTED columns' blobs
// are decoded to BE bytes, the rows' bytes are replaced via
// `apply_transform_cell` from the input, and each slab is
// re-encoded + appended to the heap end.  Non-selected columns'
// descriptors stay unchanged.  Old blobs become orphans (reclaimed
// by `repack()`).
//
// The primitive takes:
//   - `disk_rows`: flat list of disk row indices to modify (input
//     row K corresponds to disk row `disk_rows[K]`).
//   - `selected_col_indices`: indices into `columns` and
//     `algorithms` naming the columns to modify.  Length must
//     match `per_column_inputs.len()`.
//   - `per_column_inputs[K]`: a per-column ndarray of shape
//     `(disk_rows.len(),) + per_cell_shape` for the K-th selected
//     column.
//
// Dispatcher use cases (column / row narrowing combinations):
//   - 6c-2b row writes (`hdu[i]=record`, `hdu[a:b]=arr`,
//     `hdu[[i,j,k]]=arr`): `selected_col_indices` = all columns;
//     `disk_rows` = the row selection.
//   - 6c-2c whole-column (`hdu["col"]=arr`): `selected_col_indices`
//     = [col_idx]; `disk_rows` = `0..nrows`.
//   - 6c-2c single-cell (`hdu[r, "col"]=v`): `selected_col_indices`
//     = [col_idx]; `disk_rows` = `[r]`.
//   - 6c-2c multi-column (`hdu[[c1, c2]]=arr`):
//     `selected_col_indices` = [c1_idx, c2_idx];
//     `disk_rows` = `0..nrows`.
//
// VLA selected columns are handled per-(tile, col) by
// `setitem_vla_column_tile`: the existing dual-descriptor blob is
// GZIP-decompressed, each edited cell is re-encoded with the
// uncompressed-fallback rule and appended to the heap (orphaning
// the old cell's compressed bytes), the in-RAM blob's
// compressed-Q descriptor is updated with the new (cvlalen,
// cvlastart), and the original-side descriptor gets a fresh
// `original_offset = current ZPCOUNT` (orphaning the cell's old
// original-heap slot in funpack's reconstructed view).  The
// re-GZIP'd blob is appended to heap end and the main-table
// descriptor is rewritten.  ZPCOUNT bumps by the new cell's
// uncompressed-byte size on every edited cell; PCOUNT bumps by
// the new per-cell payload + the new dual-desc blob.
//
// Memory bound: per affected (tile, col), one BE-bytes slab
// (fixed) or one decompressed dual-desc blob (VLA) plus one
// per-cell BE-bytes buffer.  Per-tile work is encoded, written,
// and dropped before the next column.  Plus the full descriptor
// table held in RAM (n_tiles * ncols * 16 bytes; small).
//
// Validate-then-mutate: ColPrep construction up front guarantees
// dtype/shape errors raise BEFORE any file mutation; failures
// inside the encode/write loop taint the file.
// Stable arguments shared across every __setitem__ dispatch branch
// + the per-tile rewrite primitive.  Bundling them avoids 14-arg
// call sites that obscure the per-branch variation (which is just
// `per_column_inputs`, `selected_col_indices`, and `disk_rows`).
pub(crate) struct SetItemCtx<'a> {
    pub(crate) super_: &'a HDU,
    pub(crate) cards: &'a [String],
    pub(crate) columns: &'a [Column],
    pub(crate) algorithms: &'a [CompressionAlgorithm],
    pub(crate) per_col_configs: Option<&'a [CompressionConfigKind]>,
    pub(crate) nrows: usize,
    pub(crate) ztilelen: usize,
    pub(crate) n_tiles: usize,
    pub(crate) descriptor_row_width: usize,
    pub(crate) data_offset: u64,
    pub(crate) current_pcount: u64,
    pub(crate) cache: &'a ColumnTileCache,
}

pub(crate) fn setitem_compressed_cols(
    py: Python<'_>,
    ctx: &SetItemCtx<'_>,
    per_column_inputs: &[Bound<'_, PyAny>],
    selected_col_indices: &[usize],
    disk_rows: &[usize],
) -> PyResult<()> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    crate::common::check_not_tainted(&ctx.super_.tainted)?;

    if disk_rows.is_empty() || selected_col_indices.is_empty() {
        return Ok(());
    }
    if per_column_inputs.len() != selected_col_indices.len() {
        return Err(PyValueError::new_err(format!(
            "internal: per-column inputs len {} != selected columns len {}",
            per_column_inputs.len(), selected_col_indices.len())));
    }
    for &col_idx in selected_col_indices {
        if col_idx >= ctx.columns.len() {
            return Err(PyValueError::new_err(format!(
                "internal: selected col_idx {} out of range (ncols={})",
                col_idx, ctx.columns.len())));
        }
    }
    let n_input_rows = disk_rows.len();

    // Validate-then-mutate: per selected column, either build a
    // ColPrep (fixed) or validate the VLA Object-dtype + length.
    // dtype/shape errors surface before any file I/O.  preps[i] is
    // None for VLA columns; the VLA cells are validated lazily
    // inside the per-tile loop (one call per edited cell).
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let mut preps: Vec<Option<ColPrep<'_>>> =
        Vec::with_capacity(selected_col_indices.len());
    let mut any_vla = false;
    for (&col_idx, arr) in selected_col_indices.iter()
        .zip(per_column_inputs.iter())
    {
        let col = &ctx.columns[col_idx];
        if col.var_kind.is_some() {
            any_vla = true;
            if !arr.is_instance(&ndarray)? {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.__setitem__: VLA column '{}' \
                     value must be a numpy ndarray", col.name)));
            }
            let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
            if shape.is_empty() || shape[0] != n_input_rows {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.__setitem__: VLA column '{}' \
                     shape {:?} does not have first axis == {}",
                    col.name, shape, n_input_rows)));
            }
            let kind: String = arr.getattr("dtype")?
                .getattr("kind")?.extract()?;
            if kind != "O" {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.__setitem__: VLA column '{}' \
                     input must be a numpy Object dtype ndarray \
                     (kind 'O'), got kind '{}'", col.name, kind)));
            }
            preps.push(None);
            continue;
        }
        let cfg = ctx.per_col_configs.and_then(|cs| cs.get(col_idx));
        preps.push(Some(prepare_fixed_column(
            &np, &ndarray, arr, col, n_input_rows, cfg,
        )?));
    }

    // Bucket affected disk rows by tile.  BTreeMap so we walk tiles
    // in increasing index order (better disk locality for the
    // descriptor + existing-heap reads).  Each entry is a vec of
    // (in_tile_offset, input_row_idx) pairs.
    use std::collections::BTreeMap;
    let zt = ctx.ztilelen.max(1);
    let mut by_tile: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (input_row, &disk_row) in disk_rows.iter().enumerate() {
        let tile_idx = disk_row / zt;
        let in_tile = disk_row % zt;
        by_tile.entry(tile_idx).or_default().push((in_tile, input_row));
    }

    // Read the full descriptor table.  Small (n_tiles * ncols * 16
    // bytes; typically a few KB).  Re-emitted in full at the end.
    let desc_table_size = ctx.n_tiles * ctx.descriptor_row_width;
    let mut desc_table = vec![0u8; desc_table_size];
    if desc_table_size > 0 {
        let mut g = lock_file(&ctx.super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(ctx.data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut desc_table).map_err(|e| {
            PyIOError::new_err(format!(
                "__setitem__: read descriptor table: {}", e))
        })?;
    }

    let heap_start_offset = ctx.data_offset
        + (ctx.n_tiles as u64) * (ctx.descriptor_row_width as u64);
    // Heap cursor starts at current PCOUNT — new blobs append to
    // the heap end, orphaning the old blobs (their heap bytes stay
    // until `repack()` reclaims them).
    let mut heap_cursor = ctx.current_pcount;
    // ZPCOUNT cursor: start from the current value parsed from the
    // header.  Each edited VLA cell appends a fresh original-heap
    // slot at `new_zpcount` (orphaning the cell's old original-heap
    // position).  Only rewritten if any VLA col was actually
    // touched, since fixed-only edits leave the original heap
    // untouched.  funpack copies ZPCOUNT → PCOUNT on reconstruction,
    // so this must reflect the new total.
    let mut new_zpcount = parse_keyword(ctx.cards, "ZPCOUNT")
        .unwrap_or(0).max(0) as u64;

    for (&tile_idx, edits) in by_tile.iter() {
        let tile_row_start = tile_idx * ctx.ztilelen;
        let rows_in_tile = if tile_idx + 1 == ctx.n_tiles {
            ctx.nrows - tile_row_start
        } else {
            ctx.ztilelen
        };
        for (sel_k, &col_idx) in selected_col_indices.iter().enumerate() {
            let col = &ctx.columns[col_idx];
            if col.var_kind.is_some() {
                let cfg = ctx.per_col_configs
                    .and_then(|cs| cs.get(col_idx));
                heap_cursor = setitem_vla_column_tile(
                    py, &ndarray, ctx, heap_start_offset, heap_cursor,
                    tile_idx, col_idx, col, rows_in_tile, edits,
                    &per_column_inputs[sel_k],
                    ctx.algorithms[col_idx],
                    cfg.map(rice_blocksize_of).unwrap_or(32),
                    cfg.and_then(gzip_level_of),
                    &mut desc_table, &mut new_zpcount,
                )?;
                continue;
            }
            // Fixed-column path.
            let prep = preps[sel_k].as_ref()
                .expect("non-VLA col has a ColPrep");
            // Decode the existing tile blob into a BE-bytes slab.
            let mut slab = decode_existing_tile_to_be_bytes(
                &ctx.super_.file, ctx.cards, ctx.data_offset, tile_idx,
                col_idx, col, ctx.algorithms[col_idx], rows_in_tile,
                ctx.descriptor_row_width,
            )?;
            // Overwrite the affected rows.  Per-cell layout matches
            // what the encoder expects (rows_in_tile * per_row_bytes).
            let src_bytes = prep.buf.as_slice();
            for &(in_tile, input_row) in edits.iter() {
                let src_off = input_row * prep.src_total_size;
                let src = &src_bytes
                    [src_off..src_off + prep.src_total_size];
                let dst_off = in_tile * prep.per_row_bytes;
                let dst = &mut slab
                    [dst_off..dst_off + prep.per_row_bytes];
                apply_transform_cell(
                    &prep.transform, src, dst, &col.name, input_row,
                )?;
            }
            // Re-encode + append to heap + record new descriptor.
            let n_pixels = rows_in_tile * prep.per_row_pixels;
            heap_cursor = encode_be_slab_to_heap_and_record(
                &slab, n_pixels, ctx.algorithms[col_idx],
                prep.elem_size, prep.rice_blocksize, prep.gzip_level,
                tile_idx, col_idx, &col.name, ctx.descriptor_row_width,
                heap_start_offset, heap_cursor, &mut desc_table,
                &ctx.super_.file, &ctx.super_.layout, ctx.data_offset,
                &ctx.super_.tainted,
            )?;
        }
    }

    // Write the (modified) descriptor table back at data_offset.
    {
        let mut g = lock_file(&ctx.super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(ctx.data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&desc_table).map_err(|e| {
            ctx.super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "__setitem__: descriptor-table write failed: {}", e))
        })?;
        f.flush().map_err(|e| {
            ctx.super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "__setitem__: flush failed: {}", e))
        })?;
    }

    // Update PCOUNT (and ZPCOUNT if any VLA col was touched).
    // Standard disk-write-before-commit + taint discipline.
    let cards_guard = ctx.super_.cards_write_lock()?;
    let mut new_cards = ctx.cards.to_vec();
    crate::hdu_table::set_pcount_in_cards(&mut new_cards, heap_cursor);
    if any_vla {
        set_zpcount_in_cards(&mut new_cards, new_zpcount);
    }
    crate::header::rewrite_header_to_disk(
        &ctx.super_.file, &ctx.super_.offsets, &ctx.super_.layout,
        &new_cards, &ctx.super_.tainted,
    )?;
    cards_guard.commit(new_cards);

    // Invalidate the cache — every modified tile's column entry is
    // stale (descriptor points at a new heap blob, decoded bytes
    // differ).  Full clear is simplest and correct; per-(tile, col)
    // eviction would only matter for hot-path workloads that
    // interleave setitem with reads of unmodified tiles.
    ctx.cache.clear();
    Ok(())
}

// Modify selected rows of ONE VLA column in ONE tile.  Mirrors
// `encode_vla_column_tile_with_merge` (append's merge path) in
// spirit but only the EDITED rows are re-encoded; non-edited rows
// keep their compressed bytes in place (their cvlastart values in
// the in-RAM blob are unchanged) and their original-side descriptors
// unchanged.  Each edited cell gets a fresh `original_offset =
// new_zpcount` so funpack's reconstructed heap stays consistent
// even when nelements changes per cell — the cell's old original-
// heap slot becomes a phantom orphan that funpack never references.
//
// Returns the updated `heap_cursor` (one past the appended GZIP'd
// dual-descriptor blob); `new_zpcount` is mutated in place.
#[allow(clippy::too_many_arguments)]
fn setitem_vla_column_tile(
    py: Python<'_>,
    ndarray: &Bound<'_, PyAny>,
    ctx: &SetItemCtx<'_>,
    heap_start_offset: u64,
    mut heap_cursor: u64,
    tile_idx: usize,
    col_idx: usize,
    col: &Column,
    rows_in_tile: usize,
    edits: &[(usize, usize)],
    cell_inputs: &Bound<'_, PyAny>,
    algo: CompressionAlgorithm,
    rice_blocksize: u32,
    gzip_level: Option<u32>,
    desc_table: &mut [u8],
    new_zpcount: &mut u64,
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
        .expect("setitem_vla_column_tile called for non-VLA column");
    let width_orig = if descriptor_kind == 'P' { 8 } else { 16 };

    // 1. Read the existing main descriptor entry for this (tile, col).
    let main_desc_off = tile_idx * ctx.descriptor_row_width + col_idx * 16;
    let blob_nelems_s = i64::from_be_bytes(
        desc_table[main_desc_off..main_desc_off + 8].try_into().unwrap());
    let blob_offset_s = i64::from_be_bytes(
        desc_table[main_desc_off + 8..main_desc_off + 16]
            .try_into().unwrap());
    let blob_nelems = blob_nelems_s.max(0) as usize;
    let blob_heap_offset = blob_offset_s.max(0) as u64;

    // 2. Read + GZIP-decompress the existing dual-descriptor blob.
    let expected_blob_size = rows_in_tile * width_orig + rows_in_tile * 16;
    let mut blob = if blob_nelems > 0 {
        let mut compressed = vec![0u8; blob_nelems];
        {
            let mut g = lock_file(&ctx.super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(heap_start_offset + blob_heap_offset))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.read_exact(&mut compressed).map_err(|e| {
                PyIOError::new_err(format!(
                    "__setitem__: read VLA dual-descriptor blob for \
                     tile {} col '{}': {}", tile_idx, col.name, e))
            })?;
        }
        gzip_decompress_bytes(&compressed, expected_blob_size)?
    } else {
        // Empty tile slot — start from zeroed blob (shouldn't
        // normally happen, since 6c-1b's write/append emits a
        // populated blob even when every cell has nelements == 0).
        vec![0u8; expected_blob_size]
    };
    let comp_desc_start = rows_in_tile * width_orig;

    // 3. For each edit: serialize + encode the new cell, append to
    // heap, rewrite both descriptors in the blob.
    for &(in_tile, input_row) in edits {
        let cell = cell_inputs.get_item(input_row)?;
        let nelements = validate_vla_cell(
            &cell, ndarray, inner_letter, &col.name, input_row)?;
        let mut cell_bytes_be = vec![0u8; nelements * elem_size];
        if nelements > 0 {
            serialize_vla_cell(
                &cell, inner_letter, nelements, &mut cell_bytes_be)?;
        }
        let (cvlalen, cvlastart) = if nelements == 0 {
            (0u64, 0u64)
        } else {
            // Try compressing; fall back to the raw bytes when the
            // compressed payload isn't smaller.  cfitsio's table
            // VLA encoder uses the same rule; Phase 4 read handles
            // both forms.
            let compressed = encode_table_column_slab(
                algo, &cell_bytes_be, nelements, elem_size,
                rice_blocksize, gzip_level)?;
            let payload = if compressed.len() >= cell_bytes_be.len() {
                &cell_bytes_be[..]
            } else {
                &compressed[..]
            };
            let plen = payload.len() as u64;
            let want_total = heap_start_offset + heap_cursor + plen
                - ctx.data_offset;
            grow_file_to_at_least(
                &ctx.super_.file, &ctx.super_.layout, ctx.data_offset,
                want_total, &ctx.super_.tainted)?;
            {
                let mut g = lock_file(&ctx.super_.file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err(
                        "file is closed"))?;
                f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.write_all(payload).map_err(|e| {
                    ctx.super_.tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "__setitem__: VLA cell heap write at tile {} \
                         col '{}' input_row {}: {}; close + reopen",
                        tile_idx, col.name, input_row, e))
                })?;
            }
            let placed = heap_cursor;
            heap_cursor += plen;
            (plen, placed)
        };
        // Compressed-side Q descriptor (always 16 bytes).
        let comp_off = comp_desc_start + in_tile * 16;
        write_descriptor(
            'Q', cvlalen as usize, cvlastart as usize,
            &mut blob[comp_off..comp_off + 16],
        );
        // Original-side descriptor: assign a fresh slot at
        // new_zpcount and bump.  Old slot becomes a phantom
        // orphan in funpack's reconstructed heap.
        let orig_off = in_tile * width_orig;
        let new_original_offset = *new_zpcount;
        write_descriptor(
            descriptor_kind, nelements, new_original_offset as usize,
            &mut blob[orig_off..orig_off + width_orig],
        );
        *new_zpcount = new_zpcount.checked_add(
            (nelements * elem_size) as u64,
        ).ok_or_else(|| PyValueError::new_err(format!(
            "__setitem__: ZPCOUNT overflow at tile {} col '{}'",
            tile_idx, col.name)))?;
    }
    let _ = py;

    // 4. Re-GZIP the (modified) blob and append to the heap end.
    let gzipped = encode_gzip1(&blob, None)?;
    let want_total = heap_start_offset + heap_cursor
        + gzipped.len() as u64 - ctx.data_offset;
    grow_file_to_at_least(
        &ctx.super_.file, &ctx.super_.layout, ctx.data_offset,
        want_total, &ctx.super_.tainted)?;
    let blob_new_offset = heap_cursor;
    {
        let mut g = lock_file(&ctx.super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&gzipped).map_err(|e| {
            ctx.super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "__setitem__: VLA dual-descriptor blob write at \
                 tile {} col '{}': {}; close + reopen",
                tile_idx, col.name, e))
        })?;
    }
    heap_cursor += gzipped.len() as u64;

    // 5. Update the main-table descriptor for this (tile, col).
    desc_table[main_desc_off..main_desc_off + 8]
        .copy_from_slice(&(gzipped.len() as i64).to_be_bytes());
    desc_table[main_desc_off + 8..main_desc_off + 16]
        .copy_from_slice(&(blob_new_offset as i64).to_be_bytes());

    Ok(heap_cursor)
}

// Dispatcher helpers — small input-validation primitives shared
// across the __setitem__ branches.

// Reject value if it isn't a numpy ndarray instance.  Error message
// names the user-facing key form via `key_label`.
pub(crate) fn require_ndarray(
    py: Python<'_>, value: &Bound<'_, PyAny>, key_label: &str,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    if !value.is_instance(&ndarray)? {
        return Err(PyValueError::new_err(format!(
            "{} = value: value must be a numpy ndarray", key_label)));
    }
    Ok(())
}

// require_ndarray + an exact length check.  Used by branches whose
// `value.len()` is meaningful (slices, fancy rows, multi-col).
pub(crate) fn require_ndarray_with_length(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    expected_len: usize,
    key_label: &str,
) -> PyResult<()> {
    require_ndarray(py, value, key_label)?;
    let v_len: usize = value.len().unwrap_or(0);
    if v_len != expected_len {
        return Err(PyValueError::new_err(format!(
            "{} = value: expected length {}, got {}",
            key_label, expected_len, v_len)));
    }
    Ok(())
}

// Validate a structured-ndarray multi-column subset value: check
// for named fields, case-insensitive resolve each name against the
// table columns, dedup, and materialize each per-column view as a
// contiguous ndarray.  Returns (selected_col_indices, per_column).
pub(crate) fn resolve_structured_subset_value<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    columns: &[Column],
    names: &[String],
) -> PyResult<(Vec<usize>, Vec<Bound<'py, PyAny>>)> {
    let dtype = value.getattr("dtype")?;
    let value_names_attr = dtype.getattr("names")?;
    if value_names_attr.is_none() {
        return Err(PyValueError::new_err(
            "CompressedTableHDU[[names]] = value: value must be a \
             structured ndarray with named fields"));
    }
    let value_names: Vec<String> = value_names_attr.extract()?;
    let value_names_upper: std::collections::HashSet<String> =
        value_names.iter().map(|n| n.to_uppercase()).collect();
    let np = py.import("numpy")?;
    let mut selected: Vec<usize> = Vec::with_capacity(names.len());
    let mut seen_upper: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut per_column: Vec<Bound<'py, PyAny>> =
        Vec::with_capacity(names.len());
    for name in names {
        let name_u = name.to_uppercase();
        if !seen_upper.insert(name_u.clone()) {
            return Err(PyValueError::new_err(format!(
                "CompressedTableHDU[[names]] = value: duplicate \
                 column name '{}'", name)));
        }
        let idx = find_compressed_column_index(columns, name)?;
        if !value_names_upper.contains(&name_u) {
            return Err(PyValueError::new_err(format!(
                "CompressedTableHDU[[names]] = value: value \
                 structured dtype is missing field '{}'", name)));
        }
        selected.push(idx);
        let field_view = value.get_item(name.as_str())?;
        let per_col = np.call_method1(
            "ascontiguousarray", (field_view,))?;
        per_column.push(per_col);
    }
    Ok((selected, per_column))
}

// Promote a single-cell RHS to a length-1 per-column ndarray
// matching the column's expected dtype + per-cell shape.  Same
// coercion shape as the uncompressed-side `setitem_cell` helper:
// asarray(value, dtype) + broadcast_to((1,) + per_cell_shape) +
// ascontiguousarray.  Numpy's asarray + broadcast_to handle the
// scalar / 0-d / pre-shaped cases uniformly and surface shape
// mismatches as `ValueError`.
pub(crate) fn coerce_cell_value_to_len1<'py>(
    py: Python<'py>,
    col: &Column,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let expected_shape: Vec<usize> = column_expected_shape(col);
    let full_shape: Vec<usize> = std::iter::once(1)
        .chain(expected_shape.iter().copied()).collect();
    let (dtype_str, _) = field_dtype_and_shape(col, false)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", &dtype_str)?;
    let arr = np.call_method("asarray", (value,), Some(&kwargs))?;
    let broadcast = np.getattr("broadcast_to")?
        .call1((arr, full_shape))?;
    np.call_method1("ascontiguousarray", (broadcast,))
}

// Wrap a single VLA cell value as a length-1 Object-dtype ndarray
// for the setitem primitive.  Used by `hdu[r, "vla_col"] = v` and
// by `hdu["vla_col"][int_row] = v` — both paths want to dispatch
// to the same per-row VLA encoder, which expects an Object ndarray
// it can index via `arr.get_item(0)`.  The inner-element type
// validation runs later via `validate_vla_cell`.
pub(crate) fn coerce_vla_cell_value_to_len1<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "O")?;
    let arr = np.call_method("empty", ((1usize,),), Some(&kwargs))?;
    arr.set_item(0, value)?;
    Ok(arr)
}

// Resolve a `rows` argument (int / slice / iterable of ints) for
// the subset `__setitem__` methods into a flat list of disk row
// indices + a flag indicating whether the original key was a bare
// int (which lets the caller treat the value as a scalar / record
// rather than an ndarray of length 1).
//
// Parallel to `setitem::resolve_rows_key` on the uncompressed side
// — duplicated here so the compressed errors come out with the
// "CompressedTableHDU" prefix and the helper composes with
// `normalize_disk_row` directly.
pub(crate) fn resolve_compressed_rows_key(
    rows: &Bound<'_, PyAny>, nrows: usize,
) -> PyResult<(Vec<usize>, bool)> {
    if rows.is_instance_of::<PySlice>() {
        let slice_py = rows.cast::<PySlice>()?;
        let indices = slice_py.indices(nrows as isize)?;
        let count = indices.slicelength as i64;
        if count <= 0 {
            return Ok((Vec::new(), false));
        }
        let step = indices.step as i64;
        if step <= 0 {
            return Err(PyValueError::new_err(
                "CompressedTableHDU subset write: negative or zero \
                 slice step is not supported"));
        }
        let start = indices.start as i64;
        let mut out = Vec::with_capacity(count as usize);
        for k in 0..count {
            let r = start + k * step;
            if r < 0 || r >= nrows as i64 {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    format!("row index {} out of bounds for {} rows",
                            r, nrows)));
            }
            out.push(r as usize);
        }
        return Ok((out, false));
    }
    if !rows.is_instance_of::<pyo3::types::PyBool>() {
        if let Ok(i) = rows.extract::<i64>() {
            let r = normalize_disk_row(i, nrows)?;
            return Ok((vec![r], true));
        }
    }
    let iter = rows.try_iter().map_err(|_| PyValueError::new_err(
        "row key must be an int, slice, or iterable of ints"))?;
    let items: Vec<Bound<'_, PyAny>> = iter.collect::<PyResult<_>>()?;
    let mut out: Vec<usize> = Vec::with_capacity(items.len());
    for item in items.iter() {
        if item.is_instance_of::<pyo3::types::PyBool>() {
            return Err(PyValueError::new_err(
                "row iterable contains a bool"));
        }
        let i: i64 = item.extract().map_err(|_| PyValueError::new_err(
            "row iterable contains a non-int element"))?;
        out.push(normalize_disk_row(i, nrows)?);
    }
    Ok((out, false))
}

// Find a column by name (case-insensitive); shared by the __setitem__
// dispatch branches that take a column name (SingleColumn /
// MultiColumns / Cell).  Error message names the user-supplied
// spelling so the diagnostic is useful regardless of case.
pub(crate) fn find_compressed_column_index(
    columns: &[Column], name: &str,
) -> PyResult<usize> {
    let name_u = name.to_uppercase();
    columns.iter()
        .position(|c| c.name.to_uppercase() == name_u)
        .ok_or_else(|| PyValueError::new_err(format!(
            "CompressedTableHDU[name] = value: no column named '{}'",
            name)))
}

// Normalize a possibly-negative disk row index against ZNAXIS2;
// reject out-of-range.  Mirrors numpy/structured-array semantics —
// same shape as the uncompressed-side helper.
pub(crate) fn normalize_disk_row(i: i64, nrows: usize) -> PyResult<usize> {
    let n = nrows as i64;
    let r = if i < 0 { i + n } else { i };
    if r < 0 || r >= n {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "CompressedTableHDU row index {} out of bounds for {} rows",
            i, nrows)));
    }
    Ok(r as usize)
}

// ---------------------------------------------------------------------------
// ZHECKSUM / ZDATASUM on compressed tables
// ---------------------------------------------------------------------------
//
// Both are computed against the EQUIVALENT UNCOMPRESSED table — the
// BITPIX-native big-endian bytes the original (pre-compression)
// BINTABLE would have stored.  Astropy + cfitsio use the same
// convention, so our values agree bit-exact with what funpack +
// verify_checksum would compute on the decompressed file.
//
// Streaming: we never materialize the full equivalent-uncompressed
// data section in RAM (real survey tables can be many GB after
// decompression).  Per-tile decode happens one tile at a time and
// feeds the running sum via `ChecksumStream`.  Peak memory bounded
// at a few MiB per tile regardless of file size.
//
// Scope: fixed-column tables only.  VLA-bearing compressed tables
// raise NotImplementedError because reconstructing their
// equivalent-uncompressed heap requires per-cell ORIGINAL offsets
// stored in the dual-descriptor blob — surfacing those to the
// checksum path is a deferred follow-up.  Workaround for VLA:
// rebuild via create_table_hdu (without compress) + write, then
// add_checksum the resulting uncompressed TableHDU.

