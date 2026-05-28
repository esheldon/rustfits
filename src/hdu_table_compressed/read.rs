// Compressed-table read path: whole-table + rows= read, the per-tile
// row planner, column-slab + VLA-cell decompression, and the gzip
// blob helper.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use crate::common::{
    byteswap_in_place, lock_file, parse_keyword, parse_string_keyword,
    FileHandle, RawBuffer,
};
use crate::hdu_table::{
    build_numpy_dtype, build_var_cell_value,
    bytes_per_element, byteswap_unit, convert_column_cell,
    numpy_field_layout, parse_columns,
    read_descriptor, resolve_columns, resolve_rows, scaling_kind, Column,
    ScalingKind,
};
use crate::zimage::gzip::{decode_gzip1, decode_gzip2};
use crate::zimage::rice::decode_rice;
use crate::zimage::{parse_algorithm, CompressionAlgorithm};

use super::hdu::{CacheKey, ColumnTileCache, synthesize_uncompressed_cards};

// ---------------------------------------------------------------------------
// Phase 2 — whole-table read
// ---------------------------------------------------------------------------
//
// For each tile T = 0..n_tiles:
//   1. Read the row of N descriptors (one per ORIGINAL column) at
//      data_offset + T * descriptor_row_width.  Each descriptor is
//      16 bytes (Q kind: two big-endian i64 = nelements + heap_offset).
//   2. For each selected column C:
//      - Read `nelements` compressed bytes from the heap.
//      - Decompress per ZCTYPn:
//          - GZIP_1 → gzip decode (native-order bytes).
//          - GZIP_2 → gzip decode + reverse byte-shuffle.
//          - RICE_1 → rice decode (B/I/J only — cfitsio's table
//            compressor doesn't emit RICE for other letters).
//      - The decoder returns NATIVE-order bytes.  Byteswap back to
//        big-endian so the shared per-row cell converter
//        (`convert_column_cell`) can consume them — that function
//        is the one used by the uncompressed read path and expects
//        BE input.
//      - For each row R in the tile, copy + scale + byteswap the
//        cell into the output ndarray at row (tile_row_start + R),
//        field C.
//
// Peak memory bound per call: output ndarray + one tile's worth of
// decompressed bytes per column being processed (a few MB for
// typical fpack tile sizes).  No whole-table intermediate buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_compressed_table(
    py: Python<'_>,
    cards: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    rows_requested: Option<&Bound<'_, PyAny>>,
    columns_requested: Option<Vec<String>>,
    scale: bool,
    cache: &ColumnTileCache,
) -> PyResult<Py<PyAny>> {
    let virtual_cards = synthesize_uncompressed_cards(cards);
    let all_columns = parse_columns(&virtual_cards)?;

    let selected: Vec<Column> = match columns_requested {
        None => all_columns.clone(),
        Some(names) => resolve_columns(&all_columns, &names)?,
    };
    let scaling_kinds: Vec<ScalingKind> = selected.iter()
        .map(|c| if scale { scaling_kind(c) } else { Ok(ScalingKind::None) })
        .collect::<PyResult<Vec<_>>>()?;

    let n_rows = parse_keyword(cards, "ZNAXIS2")
        .unwrap_or(0).max(0) as usize;
    let n_tiles = parse_keyword(cards, "NAXIS2")
        .unwrap_or(0).max(0) as usize;
    let ztilelen = parse_keyword(cards, "ZTILELEN")
        .unwrap_or(0).max(0) as usize;
    let descriptor_row_width = parse_keyword(cards, "NAXIS1")
        .unwrap_or(0).max(0) as usize;

    // Per-column algorithm — parsed once up front.  Reject unsupported
    // algorithms (HCOMPRESS_1 and PLIO_1 are image-only) so we don't
    // start reading just to bomb on a tile-by-tile basis.
    let algorithms: Vec<CompressionAlgorithm> = (0..all_columns.len())
        .map(|i| {
            let key = format!("ZCTYP{}", i + 1);
            let zctyp = parse_string_keyword(cards, &key)
                .ok_or_else(|| PyValueError::new_err(format!(
                    "compressed table missing {} card", key)))?;
            let algo = parse_algorithm(&zctyp)?;
            match algo {
                CompressionAlgorithm::Gzip1
                | CompressionAlgorithm::Gzip2
                | CompressionAlgorithm::Rice1 => Ok(algo),
                CompressionAlgorithm::Hcompress1
                | CompressionAlgorithm::Plio1 => Err(
                    PyValueError::new_err(format!(
                        "{} = '{}' — only GZIP_1, GZIP_2, and RICE_1 \
                         are valid for compressed tables",
                        key, zctyp))),
            }
        })
        .collect::<PyResult<_>>()?;

    // Heap base: respect THEAP if present, otherwise default to the
    // end of the descriptor rows.
    let theap_raw = parse_keyword(cards, "THEAP").unwrap_or(0);
    let heap_base_in_data = if theap_raw > 0 {
        theap_raw as u64
    } else {
        (n_tiles as u64) * (descriptor_row_width as u64)
    };
    let heap_start = data_offset + heap_base_in_data;

    // Resolve row selection.  When rows_requested is None we walk
    // the whole table; otherwise we get a list of disk-row indices
    // in the user's requested order (deduped, range-validated).
    let row_plan = match rows_requested {
        None => RowPlan::all(n_rows),
        Some(arg) => {
            let indices = resolve_rows(arg, n_rows)?;
            RowPlan::from_indices(indices, ztilelen)
        }
    };
    let n_out = row_plan.n_output_rows;

    // Allocate output ndarray.
    let dtype = build_numpy_dtype(py, &selected, scale)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("empty", (n_out, dtype.bind(py)))?;
    if n_out == 0 || selected.is_empty() {
        return Ok(arr.unbind());
    }

    let arr_dtype = arr.getattr("dtype")?;
    let itemsize: usize = arr_dtype.getattr("itemsize")?.extract()?;
    let field_layout = numpy_field_layout(py, &arr_dtype, &selected)?;

    // Map each selected column back to its index in the original
    // column list (case-insensitive name lookup), so we know which
    // descriptor slot to read in each tile's descriptor row.
    let selected_orig_idx: Vec<usize> = selected.iter()
        .map(|sc| all_columns.iter()
            .position(|c| c.name.eq_ignore_ascii_case(&sc.name))
            .expect("resolve_columns guaranteed presence"))
        .collect();

    // Validate descriptor row width: must be ncols * 16 (each
    // descriptor is a 1QB pair = two i64).
    let expected_desc_width = all_columns.len() * 16;
    if descriptor_row_width != expected_desc_width {
        return Err(PyValueError::new_err(format!(
            "compressed table NAXIS1 = {} but expected ncols ({}) * 16 \
             = {} bytes per descriptor row",
            descriptor_row_width, all_columns.len(), expected_desc_width)));
    }

    let mut out_buf = RawBuffer::acquire_writable(&arr)?;
    let out = out_buf.as_mut_slice();

    // Per-tile buffer (descriptors) — reused across tiles.
    let mut desc_buf = vec![0u8; descriptor_row_width];

    // Walk tiles in increasing tile_idx (best disk locality for the
    // descriptor reads + the heap-blob reads).  Output_row indices
    // come from the per-tile requests so the user's row order is
    // preserved in the final array.
    let tile_plan = row_plan.tiles_with_requests(n_tiles, ztilelen);
    for (tile_idx, requests) in tile_plan {
        let tile_row_start = tile_idx * ztilelen;
        let rows_in_tile = if tile_idx + 1 == n_tiles {
            n_rows - tile_row_start
        } else {
            ztilelen
        };

        // Read this tile's descriptor row.
        {
            let mut g = lock_file(file_handle)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            let off = data_offset
                + (tile_idx as u64) * (descriptor_row_width as u64);
            f.seek(SeekFrom::Start(off))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.read_exact(&mut desc_buf).map_err(|e| {
                PyIOError::new_err(format!(
                    "read descriptor row for tile {}: {}", tile_idx, e))
            })?;
        }

        for (out_col_idx, sel_col) in selected.iter().enumerate() {
            let orig_idx = selected_orig_idx[out_col_idx];
            let desc_slice = &desc_buf
                [orig_idx * 16..(orig_idx + 1) * 16];
            let (nelems_s, heap_offset_s) =
                read_descriptor('Q', desc_slice);
            if nelems_s < 0 || heap_offset_s < 0 {
                return Err(PyValueError::new_err(format!(
                    "tile {} column '{}': descriptor has negative field \
                     (nelements={}, offset={})",
                    tile_idx, sel_col.name, nelems_s, heap_offset_s)));
            }

            if sel_col.var_kind.is_some() {
                read_vla_column_tile(
                    py, &arr, file_handle, sel_col,
                    algorithms[orig_idx], cache, tile_idx, orig_idx,
                    nelems_s as usize, heap_start + heap_offset_s as u64,
                    heap_start, rows_in_tile,
                    scaling_kinds[out_col_idx], &requests,
                )?;
                continue;
            }

            let cache_key = CacheKey(tile_idx as u32, orig_idx as u32);
            let slab_arc = match cache.get(&cache_key) {
                Some(arc) => arc,
                None => {
                    let n_bytes_compressed = nelems_s as usize;
                    let mut compressed = vec![0u8; n_bytes_compressed];
                    if n_bytes_compressed > 0 {
                        let mut g = lock_file(file_handle)?;
                        let f = g.as_mut().ok_or_else(|| {
                            PyIOError::new_err("file is closed")
                        })?;
                        f.seek(SeekFrom::Start(
                            heap_start + heap_offset_s as u64
                        )).map_err(|e| {
                            PyIOError::new_err(e.to_string())
                        })?;
                        f.read_exact(&mut compressed).map_err(|e| {
                            PyIOError::new_err(format!(
                                "read heap for tile {} col '{}': {}",
                                tile_idx, sel_col.name, e))
                        })?;
                    }
                    let slab = decompress_column_slab(
                        algorithms[orig_idx], &compressed, sel_col,
                        rows_in_tile,
                    )?;
                    let arc = Arc::new(slab);
                    cache.put(cache_key, Arc::clone(&arc));
                    arc
                }
            };

            let kind = scaling_kinds[out_col_idx];
            let (field_offset, field_itemsize) = field_layout[out_col_idx];
            let src_cell_w = sel_col.byte_width;
            for req in &requests {
                let in_tile = req.in_tile_offset;
                let out_row = req.output_row;
                let src = &slab_arc
                    [in_tile * src_cell_w..(in_tile + 1) * src_cell_w];
                let dst_start = out_row * itemsize + field_offset;
                let dst = &mut out
                    [dst_start..dst_start + field_itemsize];
                convert_column_cell(sel_col, src, dst, out_row, kind)?;
            }
        }
    }

    drop(out_buf);
    Ok(arr.unbind())
}

// ---------------------------------------------------------------------------
// Row planning — group requested rows by tile
// ---------------------------------------------------------------------------

// One row to be filled in the output array: which row inside the tile
// to pull from, and which slot of the output to write into.
struct TileRowRequest {
    in_tile_offset: usize,
    output_row: usize,
}

// Plan describing which tiles are needed and, for each, which rows to
// read from and where to put them in the output.  `all_rows` flag
// distinguishes the full-table case (synthesize sequential requests
// per tile lazily) from the subset case (per-tile bucket built from
// resolve_rows output).
struct RowPlan {
    by_tile: std::collections::HashMap<usize, Vec<TileRowRequest>>,
    n_output_rows: usize,
    all_rows: bool,
}

impl RowPlan {
    fn all(n_rows: usize) -> Self {
        RowPlan {
            by_tile: std::collections::HashMap::new(),
            n_output_rows: n_rows,
            all_rows: true,
        }
    }

    // rows= path: bucket each requested disk row into its tile.
    fn from_indices(indices: Vec<usize>, ztilelen: usize) -> Self {
        let mut by_tile: std::collections::HashMap<usize, Vec<TileRowRequest>>
            = std::collections::HashMap::new();
        let n_out = indices.len();
        for (output_row, disk_row) in indices.into_iter().enumerate() {
            let tile_idx = if ztilelen > 0 { disk_row / ztilelen } else { 0 };
            let in_tile = if ztilelen > 0 { disk_row % ztilelen } else { 0 };
            by_tile.entry(tile_idx).or_default().push(TileRowRequest {
                in_tile_offset: in_tile,
                output_row,
            });
        }
        RowPlan { by_tile, n_output_rows: n_out, all_rows: false }
    }

    // Build the list of (tile_idx, requests) to walk, in increasing
    // tile_idx order (best disk locality for the descriptor + heap
    // reads).  For the all-rows path, synthesizes sequential requests
    // per tile; per-tile Vec is bounded by ztilelen so total
    // allocation is O(n_rows) — same as a row-subset call.
    fn tiles_with_requests(
        self, n_tiles: usize, ztilelen: usize,
    ) -> Vec<(usize, Vec<TileRowRequest>)> {
        if self.all_rows {
            (0..n_tiles).map(|tile_idx| {
                let tile_row_start = tile_idx * ztilelen;
                let rows_in_tile = if tile_idx + 1 == n_tiles {
                    self.n_output_rows - tile_row_start
                } else {
                    ztilelen
                };
                let reqs: Vec<TileRowRequest> = (0..rows_in_tile)
                    .map(|r| TileRowRequest {
                        in_tile_offset: r,
                        output_row: tile_row_start + r,
                    })
                    .collect();
                (tile_idx, reqs)
            }).collect()
        } else {
            let mut out: Vec<(usize, Vec<TileRowRequest>)> =
                self.by_tile.into_iter().collect();
            out.sort_by_key(|(idx, _)| *idx);
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Column-subset pyclasses returned by hdu[col] / hdu[[cols]]
// ---------------------------------------------------------------------------

// Dispatch on the per-column algorithm, decompress the heap blob, and
// byteswap the result back to FITS big-endian (the shared
// `convert_column_cell` expects BE input).  The existing decoders in
// `crate::zimage` byteswap to native as their last step; we undo that
// here.  The double-swap is one redundant pass per (tile, column) —
// trivially cheap relative to decompression itself; refactoring the
// decoders to expose a "leave BE" mode would shave it but isn't
// worth touching the ZIMAGE write paths for.
pub(crate) fn decompress_column_slab(
    algo: CompressionAlgorithm,
    compressed: &[u8],
    col: &Column,
    rowspertile: usize,
) -> PyResult<Vec<u8>> {
    // X (bit-packed) columns are byte-flat on disk (one cell = ceil
    // (repeat/8) bytes); the per-cell unpack into bool happens later
    // in convert_x_cell.  All other letters have a fixed element
    // width; A's elem_bytes is 1 and its repeat is total bytes.
    let (elem_bytes, n_elements) = if col.tform_letter == 'X' {
        (1usize, rowspertile * col.byte_width)
    } else {
        let n = bytes_per_element(col.tform_letter)
            .ok_or_else(|| PyValueError::new_err(format!(
                "column '{}': TFORM letter '{}' has no fixed element \
                 width", col.name, col.tform_letter)))?;
        (n, rowspertile * col.repeat)
    };
    let mut slab = match algo {
        CompressionAlgorithm::Gzip1 => {
            decode_gzip1(compressed, n_elements, elem_bytes as u32)?
        }
        CompressionAlgorithm::Gzip2 => {
            decode_gzip2(compressed, n_elements, elem_bytes as u32)?
        }
        CompressionAlgorithm::Rice1 => {
            // cfitsio's table compressor only emits RICE_1 for
            // bytepix in {1, 2, 4} (B / I / J), corresponding to
            // `fits_rcomp_byte` / `fits_rcomp_short` / `fits_rcomp`.
            // Reject anything else up front rather than letting the
            // generic image-side decoder mishandle it.
            if !matches!(col.tform_letter, 'B' | 'I' | 'J') {
                return Err(PyValueError::new_err(format!(
                    "column '{}' has TFORM letter '{}' with ZCTYP=RICE_1; \
                     cfitsio's table compressor only emits RICE_1 for \
                     B/I/J columns, so this file is malformed (or written \
                     by a non-conforming tool)",
                    col.name, col.tform_letter)));
            }
            let blocksize = 32u32;  // cfitsio table-comp constant
            let zbitpix = (elem_bytes * 8) as i32;
            decode_rice(
                compressed, n_elements, elem_bytes as u32,
                blocksize, zbitpix,
            )?
        }
        _ => unreachable!("non-table algorithm filtered upstream"),
    };
    // Decoder returns native-order bytes; convert_column_cell expects
    // FITS big-endian.  Swap back so the per-cell converter (which
    // handles unsigned-trick, general scaling, A/L, etc.) just works.
    let swap_w = byteswap_unit(col.tform_letter);
    if swap_w > 1 && !cfg!(target_endian = "big") {
        byteswap_in_place(&mut slab, swap_w);
    }
    Ok(slab)
}

// ---------------------------------------------------------------------------
// Phase 4 — VLA column read
// ---------------------------------------------------------------------------
//
// For a VLA column in a single tile, the column's heap blob (pointed
// at by the 1QB main-row descriptor) is GZIP_1-compressed regardless
// of ZCTYPn — the inner data is what ZCTYPn governs.  After GZIP
// decompression the blob is exactly `rowspertile * width_orig +
// rowspertile * 16` bytes, laid out as two concatenated descriptor
// arrays:
//
//   bytes [0 .. rowspertile * width_orig)
//     original P/Q descriptors from the user-visible BINTABLE.
//     `vlalen` here is the number of *inner-type elements* in the
//     original cell — the user-visible count.
//   bytes [rowspertile * width_orig .. rowspertile * width_orig + rowspertile * 16)
//     compressed-side Q descriptors.  `cvlalen` is the number of
//     compressed bytes for the cell, `cvlastart` is the offset of
//     those bytes inside the compressed table's heap.
//
// Per-row decompression then:
//   1. Read cvlalen bytes from heap at heap_start + cvlastart.
//   2. If cvlalen == vlalen * elem_size: the cell was stored raw
//      (cfitsio's "compression didn't help" fallback) — those bytes
//      are the original BE inner-element bytes verbatim.
//   3. Else: decompress per ZCTYPn (RICE_1 / GZIP_1 / GZIP_2).
//   4. Hand the resulting BE bytes to `build_var_cell_value`, which
//      builds the per-cell numpy ndarray (or str / bytes for A) with
//      byteswap + scaling + ASCII validation handled the same way the
//      uncompressed read path handles them.
//
// The descriptor blob is cached per (tile, col) — same as fixed-
// column slabs.  Per-cell decompressed bytes are NOT cached (could
// blow up the budget on VLA-of-images patterns); each cell read
// decompresses fresh.
#[allow(clippy::too_many_arguments)]
fn read_vla_column_tile(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    file_handle: &FileHandle,
    col: &Column,
    algo: CompressionAlgorithm,
    cache: &ColumnTileCache,
    tile_idx: usize,
    orig_idx: usize,
    blob_nelems: usize,
    blob_heap_offset: u64,
    heap_start: u64,
    rowspertile: usize,
    kind: ScalingKind,
    requests: &[TileRowRequest],
) -> PyResult<()> {
    let width_orig = match col.var_kind {
        Some('P') => 8usize,
        Some('Q') => 16usize,
        _ => return Err(PyValueError::new_err(format!(
            "column '{}': expected P or Q var_kind, got {:?}",
            col.name, col.var_kind))),
    };
    let elem_size = bytes_per_element(col.tform_letter)
        .ok_or_else(|| PyValueError::new_err(format!(
            "column '{}': unsupported VLA inner letter '{}'",
            col.name, col.tform_letter)))?;
    let expected_blob_size = rowspertile * width_orig + rowspertile * 16;

    let cache_key = CacheKey(tile_idx as u32, orig_idx as u32);
    let blob_arc = match cache.get(&cache_key) {
        Some(arc) => arc,
        None => {
            let mut compressed = vec![0u8; blob_nelems];
            if blob_nelems > 0 {
                let mut g = lock_file(file_handle)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(blob_heap_offset))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.read_exact(&mut compressed).map_err(|e| {
                    PyIOError::new_err(format!(
                        "read VLA descriptor blob for tile {} col '{}': {}",
                        tile_idx, col.name, e))
                })?;
            }
            // The descriptor blob itself is ALWAYS gzip-framed (cfitsio
            // uses compress2mem_from_mem with deflateInit2 + gzip
            // windowBits) regardless of ZCTYPn — that controls the
            // *inner* per-cell compression only.  Skip the trailing
            // native byteswap that decode_gzip1 applies (we keep BE
            // descriptors for read_descriptor to consume directly).
            let blob = if blob_nelems > 0 {
                gzip_decompress_bytes(&compressed, expected_blob_size)?
            } else {
                Vec::new()
            };
            let arc = Arc::new(blob);
            cache.put(cache_key, Arc::clone(&arc));
            arc
        }
    };
    let blob = blob_arc.as_slice();

    let compressed_desc_start = rowspertile * width_orig;
    let orig_kind = col.var_kind.unwrap();

    for req in requests {
        let in_tile = req.in_tile_offset;
        let out_row = req.output_row;

        let orig_desc = &blob
            [in_tile * width_orig..(in_tile + 1) * width_orig];
        let (vlalen_s, _orig_offset) = read_descriptor(orig_kind, orig_desc);
        if vlalen_s < 0 {
            return Err(PyValueError::new_err(format!(
                "tile {} col '{}' row {}: original VLA descriptor has \
                 negative nelements ({})",
                tile_idx, col.name, in_tile, vlalen_s)));
        }
        let vlalen = vlalen_s as usize;

        let comp_desc_off =
            compressed_desc_start + in_tile * 16;
        let comp_desc = &blob[comp_desc_off..comp_desc_off + 16];
        let (cvlalen_s, cvlastart_s) = read_descriptor('Q', comp_desc);
        if cvlalen_s < 0 || cvlastart_s < 0 {
            return Err(PyValueError::new_err(format!(
                "tile {} col '{}' row {}: compressed-VLA descriptor has \
                 negative field (cvlalen={}, cvlastart={})",
                tile_idx, col.name, in_tile, cvlalen_s, cvlastart_s)));
        }
        let cvlalen = cvlalen_s as usize;

        let value = if vlalen == 0 {
            // Empty cell — no heap read, no decompression.  Defer to
            // build_var_cell_value which materializes a 0-length
            // ndarray (or "" / b"" for A).
            build_var_cell_value(
                py, col, &[], 0, out_row, /* as_bytes = */ false, kind,
            )?
        } else {
            let mut compressed_cell = vec![0u8; cvlalen];
            if cvlalen > 0 {
                let mut g = lock_file(file_handle)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(
                    heap_start + cvlastart_s as u64
                )).map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.read_exact(&mut compressed_cell).map_err(|e| {
                    PyIOError::new_err(format!(
                        "read compressed VLA bytes for tile {} col '{}' \
                         row {}: {}",
                        tile_idx, col.name, in_tile, e))
                })?;
            }

            let raw_bytes_len = vlalen.checked_mul(elem_size)
                .ok_or_else(|| PyValueError::new_err(
                    "VLA cell size overflowed usize"))?;
            let cell_be_bytes: Vec<u8> = if cvlalen == raw_bytes_len {
                // cfitsio's uncompressed fallback: when the compressed
                // form was larger than the raw, the original BE bytes
                // are stored verbatim.  No decoder invocation needed.
                compressed_cell
            } else {
                decompress_vla_cell(
                    algo, &compressed_cell, col, vlalen,
                )?
            };
            build_var_cell_value(
                py, col, &cell_be_bytes, vlalen, out_row,
                /* as_bytes = */ false, kind,
            )?
        };

        arr.get_item(col.name.as_str())?
            .set_item(out_row, value)?;
    }
    Ok(())
}

// Decompress one VLA cell's compressed bytes into BE inner-element
// bytes.  Returns `vlalen * elem_size` bytes ready for
// `build_var_cell_value`.  Same algorithm contract as the column
// decompressor: decoders return native-order bytes; we byteswap back
// to BE.
pub(crate) fn decompress_vla_cell(
    algo: CompressionAlgorithm,
    compressed: &[u8],
    col: &Column,
    vlalen: usize,
) -> PyResult<Vec<u8>> {
    let elem_size = bytes_per_element(col.tform_letter)
        .ok_or_else(|| PyValueError::new_err(format!(
            "column '{}': unsupported VLA inner letter '{}'",
            col.name, col.tform_letter)))?;
    let mut bytes = match algo {
        CompressionAlgorithm::Gzip1 => {
            decode_gzip1(compressed, vlalen, elem_size as u32)?
        }
        CompressionAlgorithm::Gzip2 => {
            decode_gzip2(compressed, vlalen, elem_size as u32)?
        }
        CompressionAlgorithm::Rice1 => {
            if !matches!(col.tform_letter, 'B' | 'I' | 'J') {
                return Err(PyValueError::new_err(format!(
                    "VLA column '{}' with inner letter '{}' + ZCTYP=RICE_1: \
                     cfitsio only emits RICE_1 for B/I/J VLA inner types",
                    col.name, col.tform_letter)));
            }
            decode_rice(
                compressed, vlalen, elem_size as u32, 32,
                (elem_size * 8) as i32,
            )?
        }
        _ => unreachable!("non-table algorithm filtered upstream"),
    };
    let swap_w = byteswap_unit(col.tform_letter);
    if swap_w > 1 && !cfg!(target_endian = "big") {
        byteswap_in_place(&mut bytes, swap_w);
    }
    Ok(bytes)
}

// Raw-gzip decompress to a known output length, no byteswap.  Same
// primitive as crate::zimage::gzip::decode_gzip1 but without the
// trailing native byteswap — used here because the descriptor blob
// is itself a packed array of BE descriptors that we want to feed
// to read_descriptor unchanged.
pub(crate) fn gzip_decompress_bytes(compressed: &[u8], expected_len: usize) -> PyResult<Vec<u8>> {
    use flate2::read::GzDecoder;
    let mut decoder = GzDecoder::new(compressed);
    let mut out: Vec<u8> = Vec::with_capacity(expected_len);
    decoder.read_to_end(&mut out).map_err(|e| {
        PyValueError::new_err(format!(
            "GZIP decompress (VLA descriptor blob): {}", e))
    })?;
    if out.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "GZIP decompress (VLA descriptor blob): expected {} bytes, \
             got {}", expected_len, out.len())));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Phase 5 — write side
// ---------------------------------------------------------------------------
//
// cfitsio's `fits_compress_table` picks per-dtype defaults that
// differ from the image side.  See CLAUDE.md for the full table;
// rules below mirror imcompress.c around line 8261:
//
//   B (u1)  -> GZIP_1   {GZIP_1, RICE_1}
//   I (i2)  -> GZIP_2   {GZIP_1, GZIP_2, RICE_1}
//   J (i4)  -> RICE_1   {GZIP_1, GZIP_2, RICE_1}
//   K (i8)  -> GZIP_2   {GZIP_1, GZIP_2}
//   E (f4)  -> GZIP_2   {GZIP_1, GZIP_2}
//   D (f8)  -> GZIP_2   {GZIP_1, GZIP_2}
//   C (c8)  -> GZIP_2   {GZIP_1, GZIP_2}
//   M (c16) -> GZIP_2   {GZIP_1, GZIP_2}
//   L (b1)  -> GZIP_1   {GZIP_1}
//   A (str) -> GZIP_1   {GZIP_1}
//   X (bit) -> GZIP_1   {GZIP_1}
//
// We're strict about the allowed-algorithm list: an explicit
// algorithm choice that's incompatible with a column dtype
// produces a ValueError naming the allowed algorithms.  Cfitsio
// silently falls back to a default; that "tolerance" silently
// gives the user something they didn't ask for, which is worse
// than asking them to fix the call.

