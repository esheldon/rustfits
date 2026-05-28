// Compressed-table checksum: ZHECKSUM/ZDATASUM over the equivalent
// uncompressed-table bytes (streaming per-tile, incl. VLA synthetic heap).

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::io::{Read, Seek, SeekFrom};

use crate::common::{
    lock_file, parse_keyword, parse_string_keyword,
    FileHandle,
};
use crate::hdu::HDU;
use crate::hdu_table::{
    bytes_per_element, parse_columns,
    read_descriptor, Column,
};
use crate::zimage::{parse_algorithm, CompressionAlgorithm};

use super::append::{decode_existing_tile_to_be_bytes, read_vla_merge_old_blob};
use super::hdu::{synthesize_uncompressed_cards};
use super::read::{decompress_vla_cell};

// Walk every tile of a fixed-column compressed table, decode each
// (tile, col) blob to BE bytes, interleave columns per row into a
// tile-sized main buffer, and feed it to the streaming checksum.
// After all tiles, pad to BLOCK_SIZE with zeros (so the equivalent-
// uncompressed data section ends on a FITS block boundary, matching
// what a fresh uncompressed write would produce).  Peak memory per
// tile: rows_in_tile × row_width main buffer + one per-(tile,col)
// decompressed slab.  No whole-file buffer is ever allocated.
//
// `seed` is the running checksum seed — pass 0 for DATASUM, or the
// already-summed header bytes for the verify_checksum path.
// Per-VLA-cell metadata collected during the tile walk, then sorted
// by original_offset for the synthetic-heap pass.  Holds just enough
// to read + decompress the cell at feed time; the compressed bytes
// themselves stay on disk until needed.  ~40 bytes per cell.
struct VlaCellMeta {
    orig_offset: u64,
    vlalen: usize,
    cvlalen: usize,
    cvlastart: u64,
    col_idx: usize,
}

// For one (tile, vla_col), decode the dual-descriptor blob, copy
// the ORIGINAL P/Q descriptors into `tile_buf` at the column's
// byte_offset slot for each row, and push per-cell metadata
// (orig_offset, vlalen, cvlalen, cvlastart) for non-empty cells to
// `vla_cells` so the heap pass can walk them in offset order.
#[allow(clippy::too_many_arguments)]
fn collect_vla_tile_descriptors_and_meta(
    file: &FileHandle,
    data_offset: u64,
    tile_idx: usize,
    col_idx: usize,
    col: &Column,
    rows_in_tile: usize,
    n_tiles: usize,
    descriptor_row_width: usize,
    row_width: usize,
    tile_buf: &mut [u8],
    vla_cells: &mut Vec<VlaCellMeta>,
) -> PyResult<()> {
    let blob = read_vla_merge_old_blob(
        file, data_offset, tile_idx, col_idx,
        col, rows_in_tile, n_tiles, descriptor_row_width,
    )?;
    let width_orig = blob.width_orig;
    let comp_desc_start = rows_in_tile * width_orig;
    let orig_kind = col.var_kind.unwrap();
    for r in 0..rows_in_tile {
        let orig_desc_off = r * width_orig;
        let orig_desc = &blob.decompressed
            [orig_desc_off..orig_desc_off + width_orig];
        let (vlalen_s, orig_off_s) =
            read_descriptor(orig_kind, orig_desc);
        if vlalen_s < 0 || orig_off_s < 0 {
            return Err(PyValueError::new_err(format!(
                "compressed table checksum: tile {} col '{}' \
                 row {} original descriptor has negative \
                 field (vlalen={}, orig_offset={})",
                tile_idx, col.name, r, vlalen_s, orig_off_s)));
        }
        // Copy the original descriptor into the tile main buffer
        // at this row's col.byte_offset slot — same bytes the
        // equivalent uncompressed table would store in its main
        // rows.
        let dst_off = r * row_width + col.byte_offset;
        tile_buf[dst_off..dst_off + width_orig]
            .copy_from_slice(orig_desc);
        // Collect per-cell metadata for non-empty cells; empty
        // cells contribute zero bytes to the heap.
        let vlalen = vlalen_s as usize;
        if vlalen == 0 {
            continue;
        }
        let comp_off = comp_desc_start + r * 16;
        let (cvlalen_s, cvlastart_s) = read_descriptor(
            'Q', &blob.decompressed[comp_off..comp_off + 16]);
        if cvlalen_s < 0 || cvlastart_s < 0 {
            return Err(PyValueError::new_err(format!(
                "compressed table checksum: tile {} col '{}' \
                 row {} compressed descriptor has negative \
                 field (cvlalen={}, cvlastart={})",
                tile_idx, col.name, r, cvlalen_s, cvlastart_s)));
        }
        vla_cells.push(VlaCellMeta {
            orig_offset: orig_off_s as u64,
            vlalen,
            cvlalen: cvlalen_s as usize,
            cvlastart: cvlastart_s as u64,
            col_idx,
        });
    }
    Ok(())
}

// Walk `vla_cells` in original-offset order and feed the synthetic
// heap bytes to `stream`: gap zeros between cells (sparse layouts
// are legal), each cell's decompressed BE bytes (or its raw bytes
// when cfitsio's uncompressed-fallback applies), and trailing
// zeros to reach ZPCOUNT.  Holds the file lock for the whole pass.
#[allow(clippy::too_many_arguments)]
fn feed_vla_synthetic_heap(
    file: &FileHandle,
    data_offset: u64,
    n_tiles: usize,
    descriptor_row_width: usize,
    columns: &[Column],
    algorithms: &[CompressionAlgorithm],
    vla_cells: &mut [VlaCellMeta],
    zpcount: u64,
    stream: &mut crate::checksum::ChecksumStream,
) -> PyResult<()> {
    vla_cells.sort_by_key(|c| c.orig_offset);
    let heap_start = data_offset
        + (n_tiles as u64) * (descriptor_row_width as u64);
    let mut current_pos: u64 = 0;
    // Reusable zero-pad buffer for gap fills.
    const ZERO_CHUNK: usize = 1 << 16;  // 64 KiB
    let zeros = vec![0u8; ZERO_CHUNK];
    let feed_zeros = |stream: &mut crate::checksum::ChecksumStream,
                      count: u64| {
        let mut remaining = count;
        while remaining > 0 {
            let n = remaining.min(ZERO_CHUNK as u64) as usize;
            stream.feed(&zeros[..n]);
            remaining -= n as u64;
        }
    };
    let mut compressed = Vec::<u8>::new();
    let mut g = lock_file(file)?;
    let f = g.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    for cell in vla_cells.iter() {
        // Gap to this cell's start.
        if cell.orig_offset < current_pos {
            return Err(PyValueError::new_err(format!(
                "compressed table checksum: VLA cell at orig_offset \
                 {} overlaps a previous cell (current_pos={})",
                cell.orig_offset, current_pos)));
        }
        if cell.orig_offset > current_pos {
            feed_zeros(stream, cell.orig_offset - current_pos);
            current_pos = cell.orig_offset;
        }
        // Read the cell's compressed bytes from the heap.
        compressed.resize(cell.cvlalen, 0);
        f.seek(SeekFrom::Start(heap_start + cell.cvlastart))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut compressed).map_err(|e| {
            PyIOError::new_err(format!(
                "compressed table checksum: read VLA cell at \
                 cvlastart={}: {}", cell.cvlastart, e))
        })?;
        // Decompress to BE bytes (with uncompressed-fallback per
        // cfitsio's table-compression convention).
        let col = &columns[cell.col_idx];
        let elem_size = bytes_per_element(col.tform_letter)
            .ok_or_else(|| PyValueError::new_err(format!(
                "compressed table checksum: column '{}' inner \
                 letter '{}' isn't supported in VLA checksum \
                 (X-inner VLA in compressed tables is a deferred \
                 follow-up)", col.name, col.tform_letter)))?;
        let raw_bytes_len = cell.vlalen.checked_mul(elem_size)
            .ok_or_else(|| PyValueError::new_err(
                "compressed table checksum: VLA cell size overflow"))?;
        let cell_be_bytes = if cell.cvlalen == raw_bytes_len {
            // Uncompressed fallback — bytes are already BE.
            compressed.clone()
        } else {
            decompress_vla_cell(
                algorithms[cell.col_idx], &compressed, col, cell.vlalen,
            )?
        };
        stream.feed(&cell_be_bytes);
        current_pos += cell_be_bytes.len() as u64;
    }
    // Trailing zeros to reach ZPCOUNT.
    if current_pos > zpcount {
        return Err(PyValueError::new_err(format!(
            "compressed table checksum: VLA cells exceed ZPCOUNT \
             (sum={}, ZPCOUNT={})", current_pos, zpcount)));
    }
    if current_pos < zpcount {
        feed_zeros(stream, zpcount - current_pos);
    }
    Ok(())
}

fn stream_uncompressed_table_data_checksum(
    super_: &HDU,
    seed: u32,
) -> PyResult<u32> {
    use crate::common::check_not_tainted;
    use crate::hdu_image::round_up_to_block;
    check_not_tainted(&super_.tainted)?;
    let cards = super_.header_snapshot()?;
    let virtual_cards = synthesize_uncompressed_cards(&cards);
    let columns = parse_columns(&virtual_cards)?;
    let any_vla = columns.iter().any(|c| c.var_kind.is_some());
    let nrows_orig = parse_keyword(&cards, "ZNAXIS2")
        .unwrap_or(0).max(0) as usize;
    let row_width = parse_keyword(&cards, "ZNAXIS1")
        .unwrap_or(0).max(0) as usize;
    let zpcount = parse_keyword(&cards, "ZPCOUNT")
        .unwrap_or(0).max(0) as u64;
    let n_tiles = parse_keyword(&cards, "NAXIS2")
        .unwrap_or(0).max(0) as usize;
    let ztilelen = parse_keyword(&cards, "ZTILELEN")
        .unwrap_or(0).max(0) as usize;
    let descriptor_row_width = parse_keyword(&cards, "NAXIS1")
        .unwrap_or(0).max(0) as usize;
    let data_offset = super_.offsets.data_offset();

    // Per-column algorithm from ZCTYPn.
    let mut algorithms: Vec<CompressionAlgorithm> =
        Vec::with_capacity(columns.len());
    for i in 0..columns.len() {
        let key = format!("ZCTYP{}", i + 1);
        let zctyp = parse_string_keyword(&cards, &key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "compressed table missing {} card", key))
        })?;
        algorithms.push(parse_algorithm(&zctyp)?);
    }

    let mut stream = crate::checksum::ChecksumStream::new(seed);
    let mut tile_buf: Vec<u8> = Vec::new();
    // Collected per-VLA-cell metadata across the tile walk; sorted
    // by orig_offset before the heap pass.  Empty when no VLA cols.
    let mut vla_cells: Vec<VlaCellMeta> = Vec::new();
    for tile_idx in 0..n_tiles {
        let tile_row_start = tile_idx * ztilelen;
        let rows_in_tile = if tile_idx + 1 == n_tiles {
            nrows_orig - tile_row_start
        } else {
            ztilelen
        };
        tile_buf.clear();
        tile_buf.resize(rows_in_tile * row_width, 0);
        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_some() {
                collect_vla_tile_descriptors_and_meta(
                    &super_.file, data_offset, tile_idx, col_idx,
                    col, rows_in_tile, n_tiles, descriptor_row_width,
                    row_width, &mut tile_buf, &mut vla_cells,
                )?;
                continue;
            }
            // Fixed column: decode (tile, col) blob to BE bytes,
            // interleave into the tile main buffer.
            let slab = decode_existing_tile_to_be_bytes(
                &super_.file, &cards, data_offset, tile_idx, col_idx,
                col, algorithms[col_idx], rows_in_tile,
                descriptor_row_width,
            )?;
            for r in 0..rows_in_tile {
                let src_off = r * col.byte_width;
                let dst_off = r * row_width + col.byte_offset;
                tile_buf[dst_off..dst_off + col.byte_width]
                    .copy_from_slice(
                        &slab[src_off..src_off + col.byte_width]);
            }
        }
        stream.feed(&tile_buf);
    }

    // Heap pass: walk per-cell metadata in original-offset order,
    // feeding gap zeros + per-cell decompressed BE bytes + trailing
    // pad to ZPCOUNT.  No-op when the table has no VLA columns.
    if any_vla {
        feed_vla_synthetic_heap(
            &super_.file, data_offset, n_tiles, descriptor_row_width,
            &columns, &algorithms, &mut vla_cells, zpcount, &mut stream,
        )?;
    }

    // Feed BLOCK_SIZE zero-pad so the equivalent-uncompressed data
    // section ends on the FITS block boundary it would naturally
    // have if it were stored uncompressed.
    let total_main = (nrows_orig as u64) * (row_width as u64);
    let total_data = total_main + zpcount;
    let padded = round_up_to_block(total_data);
    let pad = (padded - total_data) as usize;
    if pad > 0 {
        // Pad in <= BLOCK_SIZE chunks (only one needed in practice;
        // BLOCK_SIZE = 2880).
        let zeros = vec![0u8; pad];
        stream.feed(&zeros);
    }
    Ok(stream.finish())
}

// Build the synthetic header bytes of the equivalent uncompressed
// table HDU — what the BINTABLE header would look like if the same
// table were stored without compression.  Used to compute ZHECKSUM:
// we sum (synthetic_uncompressed_header + uncompressed data) and
// encode the complement.
//
// Reuses synthesize_uncompressed_cards (which already substitutes
// NAXIS1/NAXIS2/PCOUNT and TFORMn from their Z-prefixed counterparts
// and drops Z-prefixed cards).  Then strips any existing
// DATASUM/CHECKSUM cards (those refer to the on-disk compressed
// BINTABLE, not the equivalent uncompressed) and inserts the
// caller's datasum_value / checksum_value just before END.
fn build_equivalent_uncompressed_table_header(
    cards: &[String],
    datasum_value: &str,
    checksum_value: &str,
) -> PyResult<Vec<String>> {
    use crate::header::card_string;
    let mut synth = synthesize_uncompressed_cards(cards);
    synth.retain(|c| {
        if c.len() < 8 {
            return true;
        }
        let kw = c[..8].trim_end();
        kw != "DATASUM" && kw != "CHECKSUM"
    });
    let datasum_card = card_string(
        "DATASUM", datasum_value, "data unit checksum");
    let checksum_card = card_string(
        "CHECKSUM", checksum_value, "HDU checksum");
    let end_idx = synth.iter().position(|c|
        c.len() >= 3 && c[..3].trim() == "END"
    ).unwrap_or(synth.len());
    synth.insert(end_idx, datasum_card);
    synth.insert(end_idx + 1, checksum_card);
    Ok(synth)
}

pub(crate) fn compressed_table_add_datasum(super_: &HDU) -> PyResult<()> {
    let sum = stream_uncompressed_table_data_checksum(super_, 0)?;
    let cards = super_.header_snapshot()?;
    let new_cards =
        crate::checksum::cards_with_datasum(&cards, sum, "ZDATASUM");
    crate::hdu_image::commit_header_update(super_, new_cards)
}

pub(crate) fn compressed_table_add_checksum(super_: &HDU) -> PyResult<()> {
    let datasum = stream_uncompressed_table_data_checksum(super_, 0)?;
    let datasum_str = crate::checksum::format_datasum(datasum);
    let cards = super_.header_snapshot()?;
    // ZHECKSUM: sum the equivalent-uncompressed header bytes with
    // the CHECKSUM placeholder, add the data checksum, encode the
    // complement.  Same recipe as the image side
    // (compressed_add_checksum in hdu_image_compressed.rs).
    let synth_zero = build_equivalent_uncompressed_table_header(
        &cards, &datasum_str, "0000000000000000")?;
    let synth_bytes =
        crate::hdu_image::serialize_header_to_disk_bytes(&synth_zero);
    let hsum = crate::checksum::compute_checksum_bytes(0, &synth_bytes);
    let total = crate::checksum::ones_complement_add(hsum, datasum);
    let encoded = crate::checksum::encode_checksum_ascii(total, true);
    let encoded_str = std::str::from_utf8(&encoded)
        .expect("encode_checksum_ascii produces printable ASCII");
    let mut new_cards = cards.clone();
    crate::checksum::set_or_insert_string_card(
        &mut new_cards, "ZDATASUM", &datasum_str,
        "checksum of uncompressed data",
    );
    crate::checksum::set_or_insert_string_card(
        &mut new_cards, "ZHECKSUM", encoded_str,
        "checksum of equivalent uncompressed HDU",
    );
    crate::hdu_image::commit_header_update(super_, new_cards)
}

pub(crate) fn compressed_table_verify_datasum(super_: &HDU) -> PyResult<Option<bool>> {
    let cards = super_.header_snapshot()?;
    let Some(expected_str) = parse_string_keyword(&cards, "ZDATASUM")
    else {
        return Ok(None);
    };
    let Some(expected) =
        crate::checksum::parse_datasum(expected_str.trim())
    else {
        return Ok(None);
    };
    let computed = stream_uncompressed_table_data_checksum(super_, 0)?;
    Ok(Some(computed == expected))
}

pub(crate) fn compressed_table_verify_checksum(super_: &HDU) -> PyResult<Option<bool>> {
    let cards = super_.header_snapshot()?;
    let Some(zhecksum_str) = parse_string_keyword(&cards, "ZHECKSUM")
    else {
        return Ok(None);
    };
    let Some(zdatasum_str) = parse_string_keyword(&cards, "ZDATASUM")
    else {
        // The convention requires ZDATASUM for the
        // total == 0xFFFFFFFF invariant to hold.
        return Ok(Some(false));
    };
    let synth = build_equivalent_uncompressed_table_header(
        &cards, zdatasum_str.trim(), zhecksum_str.trim(),
    )?;
    let synth_bytes =
        crate::hdu_image::serialize_header_to_disk_bytes(&synth);
    let hsum = crate::checksum::compute_checksum_bytes(0, &synth_bytes);
    // synth_bytes is BLOCK_SIZE-padded (2880 % 4 == 0), so we can
    // seed the data stream with hsum directly — no leftover bytes
    // straddle the header/data boundary.
    let total = stream_uncompressed_table_data_checksum(super_, hsum)?;
    Ok(Some(total == 0xFFFF_FFFF))
}
