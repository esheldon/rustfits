// Compressed-image checksum: ZHECKSUM/ZDATASUM over the equivalent
// uncompressed-image bytes (streamed tile-stripe by tile-stripe).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::checksum::ChecksumStream;
use crate::common::{byteswap_in_place, parse_keyword, parse_string_keyword};

use super::hdu::CompressedImageHDU;
use super::read::{get_or_decode_tile, tile_origin_and_shape, MAX_NAXIS};

// Stream the equivalent-uncompressed-image bytes through a
// ``ChecksumStream`` and return the finished checksum.  Bounded
// memory: peak working set is one tile-stripe of decoded bytes
// (all G_last tiles along the last image axis for one outer-axis
// tile coordinate) plus a one-image-row scanline buffer.  For a
// (20 000 × 4 000) f4 image with (100, 4000) tiles that's
// ~1.6 MB instead of the ~320 MB the old "decode whole image
// into a numpy ndarray then ``.tobytes()``" path needed.
//
// Why a stripe and not a smaller fixed chunk: the FITS checksum
// is order-sensitive (bytes must arrive in image-row-major); and
// the tile decoders are whole-tile.  Tile bytes within a single
// tile are already image-row-major over the tile's own shape,
// but to emit one image-row we need to interleave row segments
// from every tile in the stripe, so they all have to be live at
// once.  For typical FITS tile choices this lands in the
// 1-10 MB range — comparable to the 1 MiB streaming-chunk
// convention used elsewhere in the codebase.
//
// Byte order: ``get_or_decode_tile`` returns tile bytes in
// native-endian dtype order (matching what gets placed into a
// numpy ndarray on the read path); the checksum is over FITS
// big-endian bytes, so we byteswap each scanline before feeding
// (no-op on bytepix=1 or on a big-endian host).
//
// Scaling: NOT applied.  The checksum is over the BITPIX-native
// stored bytes — same representation a ``BITPIX=ZBITPIX``
// uncompressed HDU would hold.  For quantized-float HDUs the
// result is the lossy dequantized floats (cfitsio convention),
// which is what ``get_or_decode_tile`` produces for the
// float-output case (it runs dequant before returning).
fn stream_uncompressed_image_be_checksum(
    slf: &PyRef<'_, CompressedImageHDU>,
    py: Python<'_>,
    seed: u32,
) -> PyResult<u32> {
    use crate::common::check_not_tainted;
    use crate::hdu_image::round_up_to_block;

    let super_ = slf.as_super().as_super();
    check_not_tainted(&super_.tainted)?;
    let meta = slf.meta(super_)?;

    let zbitpix = meta.zbitpix;
    let bytepix_image: usize = match zbitpix {
        8 => 1,
        16 => 2,
        32 => 4,
        64 => 8,
        -32 => 4,
        -64 => 8,
        other => {
            return Err(PyValueError::new_err(format!(
                "compressed checksum: unsupported ZBITPIX {}",
                other
            )));
        }
    };
    let image_shape = meta.image_shape.as_slice();
    let tile_shape = meta.tile_shape.as_slice();
    let n_dims = image_shape.len();

    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)",
        ));
    }

    let mut stream = ChecksumStream::new(seed);

    // Tile-grid shape (numpy order).  G[i] = ceil(S[i] / T[i]).
    let mut grid_shape: Vec<u64> = Vec::with_capacity(n_dims);
    for axis in 0..n_dims {
        grid_shape.push(image_shape[axis].div_ceil(tile_shape[axis]));
    }
    let last_dim_tiles = grid_shape[n_dims - 1] as usize;

    // Outer tile-grid: every axis except the last.  For 1-D the
    // outer grid is empty (one "stripe" = one tile).
    let outer_len = n_dims - 1;
    let n_stripes: u64 = grid_shape[..outer_len].iter().product::<u64>().max(1);

    let mut outer_grid_coord = vec![0u64; outer_len];
    let mut within_stripe_coord = vec![0u64; outer_len];
    let row_nbytes = (image_shape[n_dims - 1] as usize) * bytepix_image;
    let mut row_buf: Vec<u8> = Vec::with_capacity(row_nbytes);
    // Decoded tiles for the current stripe.  Each entry holds
    // (actual_last_axis_size, tile_bytes).  Cleared and refilled
    // per stripe.
    let mut stripe_tiles: Vec<(usize, std::sync::Arc<Vec<u8>>)> =
        Vec::with_capacity(last_dim_tiles);
    let mut origin_buf = [0u64; MAX_NAXIS];
    let mut shape_buf = [0u64; MAX_NAXIS];

    for stripe_idx in 0..n_stripes {
        // Decode outer_grid_coord from stripe_idx (numpy-row-major
        // over the outer grid).
        let mut idx = stripe_idx;
        for axis in (0..outer_len).rev() {
            outer_grid_coord[axis] = idx % grid_shape[axis];
            idx /= grid_shape[axis];
        }

        // Decode all G_last tiles in this stripe.  Boundary tiles
        // (right/bottom image edge) can have a smaller actual
        // shape; we record each tile's last-axis size for the
        // scanline reads below.
        stripe_tiles.clear();
        for g_last in 0..last_dim_tiles as u64 {
            // tile_idx is numpy-last-fastest in the tile-grid.
            let mut tile_idx_u64: u64 = 0;
            for axis in 0..outer_len {
                tile_idx_u64 =
                    tile_idx_u64 * grid_shape[axis] + outer_grid_coord[axis];
            }
            tile_idx_u64 = tile_idx_u64 * grid_shape[n_dims - 1] + g_last;

            let d = tile_origin_and_shape(
                tile_idx_u64,
                image_shape,
                tile_shape,
                &mut origin_buf,
                &mut shape_buf,
            );
            let actual_shape = &shape_buf[..d];

            let tile_bytes = get_or_decode_tile(
                py,
                &slf.cache,
                &super_.file,
                &super_.tainted,
                tile_idx_u64,
                super_.offsets.data_offset(),
                meta.naxis1,
                meta.theap,
                &meta.cols,
                meta.algorithm,
                actual_shape,
                meta.bytepix,
                meta.blocksize,
                if zbitpix < 0 { 32 } else { zbitpix },
                zbitpix,
                meta.quant.as_ref(),
                meta.smooth,
            )?;
            stripe_tiles.push((actual_shape[n_dims - 1] as usize, tile_bytes));
        }

        // Within-stripe outer-image shape (= actual outer dims of
        // this stripe, which differ from nominal only at the image
        // edges along the outer axes).  Same for every tile in the
        // stripe because they share outer_grid_coord.
        let mut within_stripe_outer_shape = [0u64; MAX_NAXIS];
        for axis in 0..outer_len {
            let origin = outer_grid_coord[axis] * tile_shape[axis];
            let end = (origin + tile_shape[axis]).min(image_shape[axis]);
            within_stripe_outer_shape[axis] = end - origin;
        }
        let within_stripe_outer_size: u64 = within_stripe_outer_shape
            [..outer_len]
            .iter()
            .product::<u64>()
            .max(1);

        for row_idx in 0..within_stripe_outer_size {
            // Decode within_stripe_coord (numpy-row-major over the
            // stripe's outer shape).
            let mut idx = row_idx;
            for axis in (0..outer_len).rev() {
                within_stripe_coord[axis] = idx % within_stripe_outer_shape[axis];
                idx /= within_stripe_outer_shape[axis];
            }

            // Build one image-row by walking the stripe's tiles in
            // last-axis order, extracting the (within_stripe_coord)
            // row from each, concatenating into row_buf.
            row_buf.clear();
            for (tile_last_size, tile_bytes) in stripe_tiles.iter() {
                // Within-tile linear offset to the start of this
                // outer-coord's last-axis row.  The tile's outer
                // dimensions match within_stripe_outer_shape, so the
                // coord IS the within-tile outer offset.
                let mut within_tile_row_offset: u64 = 0;
                for axis in 0..outer_len {
                    within_tile_row_offset = within_tile_row_offset
                        * within_stripe_outer_shape[axis]
                        + within_stripe_coord[axis];
                }
                let row_start = (within_tile_row_offset as usize)
                    * tile_last_size
                    * bytepix_image;
                let row_end = row_start + tile_last_size * bytepix_image;
                row_buf.extend_from_slice(&tile_bytes[row_start..row_end]);
            }

            // Byteswap native -> FITS BE.  No-op on bytepix=1 or on
            // a big-endian target.
            if bytepix_image > 1 && !cfg!(target_endian = "big") {
                byteswap_in_place(&mut row_buf, bytepix_image);
            }
            stream.feed(&row_buf);
        }
    }

    // Pad to BLOCK_SIZE so the checksum covers the same span an
    // equivalent uncompressed HDU's data section would carry on disk.
    let total_pixels: u64 = image_shape.iter().product();
    let total_bytes = total_pixels * (bytepix_image as u64);
    let padded = round_up_to_block(total_bytes);
    let pad = (padded - total_bytes) as usize;
    if pad > 0 {
        let zeros = vec![0u8; pad];
        stream.feed(&zeros);
    }

    Ok(stream.finish())
}

// Build the synthetic header bytes of the *equivalent uncompressed
// image HDU* — i.e., what the header would look like if the same
// image were stored without compression.  Used to compute
// ZHECKSUM: we sum (synthetic_uncompressed_header + uncompressed
// data) and encode the complement.
//
// Cards included (minimum required for a valid IMAGE extension
// header + the cards a reader would care about for round-trip):
//   XTENSION = 'IMAGE'  (compressed HDUs can't be primary)
//   BITPIX   = <ZBITPIX>
//   NAXIS    = <ZNAXIS>
//   NAXISn   = <ZNAXISn>
//   PCOUNT   = 0
//   GCOUNT   = 1
//   BSCALE / BZERO / BLANK   (if present in the BINTABLE header
//                              — unsigned-int trick / BLANK
//                              sentinel propagate to the
//                              uncompressed equivalent)
//   EXTNAME / EXTVER         (if present)
//   DATASUM  = <ZDATASUM placeholder / value>
//   CHECKSUM = <ZHECKSUM placeholder / value>
//   END
fn build_equivalent_uncompressed_header(
    cards: &[String],
    datasum_value: &str,
    checksum_value: &str,
) -> PyResult<Vec<String>> {
    use crate::header::{card_int, card_string, pad_to_card};
    let mut out: Vec<String> = Vec::new();

    let zbitpix: i64 = parse_keyword(cards, "ZBITPIX").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZBITPIX")
    })?;
    let znaxis: i64 = parse_keyword(cards, "ZNAXIS").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZNAXIS")
    })?;

    out.push(card_string("XTENSION", "IMAGE", "image extension"));
    out.push(card_int("BITPIX", zbitpix, "number of bits per data pixel"));
    out.push(card_int("NAXIS", znaxis, "number of data axes"));
    for i in 1..=znaxis {
        let key = format!("ZNAXIS{}", i);
        let v: i64 = parse_keyword(cards, &key).ok_or_else(|| {
            PyValueError::new_err(format!("compressed HDU missing {}", key,))
        })?;
        out.push(card_int(
            &format!("NAXIS{}", i),
            v,
            &format!("length of data axis {}", i),
        ));
    }
    out.push(card_int("PCOUNT", 0, "required keyword; must = 0"));
    out.push(card_int("GCOUNT", 1, "required keyword; must = 1"));

    // Propagate optional integer-scaling cards.
    for key in &["BSCALE", "BZERO", "BLANK"] {
        if let Some(idx) = cards
            .iter()
            .position(|c| c.len() >= key.len() && c[..key.len()].trim() == *key)
        {
            // Take the card verbatim — preserves the value
            // formatting (signed int, unsigned int, etc.).
            out.push(cards[idx].trim_end().to_string());
        }
    }
    for key in &["EXTNAME", "EXTVER"] {
        if let Some(idx) = cards
            .iter()
            .position(|c| c.len() >= key.len() && c[..key.len()].trim() == *key)
        {
            out.push(cards[idx].trim_end().to_string());
        }
    }

    out.push(card_string("DATASUM", datasum_value, "data unit checksum"));
    out.push(card_string("CHECKSUM", checksum_value, "HDU checksum"));
    out.push(pad_to_card("END"));
    Ok(out)
}

pub(crate) fn compressed_add_datasum(
    slf: PyRef<'_, CompressedImageHDU>,
    py: Python<'_>,
) -> PyResult<()> {
    let sum = stream_uncompressed_image_be_checksum(&slf, py, 0)?;
    let super_ = slf.into_super().into_super();
    let cards = super_.header_snapshot()?;
    let new_cards = crate::checksum::cards_with_datasum(&cards, sum, "ZDATASUM");
    crate::hdu_image::commit_header_update(&super_, new_cards)
}

pub(crate) fn compressed_add_checksum(
    slf: PyRef<'_, CompressedImageHDU>,
    py: Python<'_>,
) -> PyResult<()> {
    let datasum = stream_uncompressed_image_be_checksum(&slf, py, 0)?;
    let super_ = slf.into_super().into_super();
    let cards = super_.header_snapshot()?;
    // ZHECKSUM is computed against the *equivalent uncompressed*
    // header bytes (per the FITS Tile Compression Convention),
    // not the BINTABLE header.  Build that synthetic header,
    // sum it + the uncompressed data, encode the complement,
    // then store the encoded value as ZHECKSUM on the BINTABLE.
    let datasum_str = crate::checksum::format_datasum(datasum);
    let synth_zero = build_equivalent_uncompressed_header(
        &cards,
        &datasum_str,
        "0000000000000000",
    )?;
    let synth_bytes = crate::hdu_image::serialize_header_to_disk_bytes(&synth_zero);
    let hsum = crate::checksum::compute_checksum_bytes(0, &synth_bytes);
    let total = crate::checksum::ones_complement_add(hsum, datasum);
    let encoded = crate::checksum::encode_checksum_ascii(total, true);
    let encoded_str = std::str::from_utf8(&encoded)
        .expect("encode_checksum_ascii produces printable ASCII");
    // Update the BINTABLE's ZDATASUM and ZHECKSUM cards.
    let mut new_cards = cards.clone();
    crate::checksum::set_or_insert_string_card(
        &mut new_cards,
        "ZDATASUM",
        &datasum_str,
        "checksum of uncompressed data",
    );
    crate::checksum::set_or_insert_string_card(
        &mut new_cards,
        "ZHECKSUM",
        encoded_str,
        "checksum of equivalent uncompressed HDU",
    );
    crate::hdu_image::commit_header_update(&super_, new_cards)
}

pub(crate) fn compressed_verify_datasum(
    slf: PyRef<'_, CompressedImageHDU>,
    py: Python<'_>,
) -> PyResult<Option<bool>> {
    let super_ref = slf.as_super().as_super();
    let cards = super_ref.header_snapshot()?;
    let Some(expected_str) = parse_string_keyword(&cards, "ZDATASUM") else {
        return Ok(None);
    };
    let Some(expected) = crate::checksum::parse_datasum(expected_str.trim())
    else {
        return Ok(None);
    };
    let computed = stream_uncompressed_image_be_checksum(&slf, py, 0)?;
    Ok(Some(computed == expected))
}

pub(crate) fn compressed_verify_checksum(
    slf: PyRef<'_, CompressedImageHDU>,
    py: Python<'_>,
) -> PyResult<Option<bool>> {
    let super_ref = slf.as_super().as_super();
    let cards = super_ref.header_snapshot()?;
    let Some(_zhecksum) = parse_string_keyword(&cards, "ZHECKSUM") else {
        return Ok(None);
    };
    let Some(zdatasum_str) = parse_string_keyword(&cards, "ZDATASUM") else {
        // Spec requires ZDATASUM for the invariant to hold.
        return Ok(Some(false));
    };
    // Re-run the equivalent-uncompressed-HDU sum and check the
    // invariant total == 0xFFFFFFFF.  Streaming the data section
    // with seed=header_checksum threads the running checksum
    // through both halves without ever materializing the data
    // bytes in RAM.
    let zhecksum_str = parse_string_keyword(&cards, "ZHECKSUM").unwrap();
    let synth = build_equivalent_uncompressed_header(
        &cards,
        zdatasum_str.trim(),
        zhecksum_str.trim(),
    )?;
    let synth_bytes = crate::hdu_image::serialize_header_to_disk_bytes(&synth);
    let hsum = crate::checksum::compute_checksum_bytes(0, &synth_bytes);
    let total = stream_uncompressed_image_be_checksum(&slf, py, hsum)?;
    Ok(Some(total == 0xFFFF_FFFF))
}
