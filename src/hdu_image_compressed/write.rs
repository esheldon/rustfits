// Compressed-image write path: per-tile encode (int/float/quantized),
// bulk write, extend, __setitem__, and descriptor-buffer helpers.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::types::{PyBytes, PySlice, PyTuple};

use crate::common::{
    check_not_tainted, lock_file, parse_keyword, parse_string_keyword,
    shift_file_tail_and_update_offsets,
    FileHandle, FileLayout, HduOffsets, TaintFlag, BLOCK_SIZE,
};
use crate::hdu_image::{
    normalize_slice_key, serialize_header_to_disk_bytes,
};
use crate::hdu_table::set_pcount_in_cards;

use super::hdu::TileCache;
use super::meta::{
    build_quant_context, compute_n_tiles, find_data_columns,
    parse_compressed_image_shape, parse_hcompress_scale, parse_rice_params,
    parse_tile_shape, tform_vla_inner_byte_width, zbitpix_to_native_dtype,
};
use super::read::{
    axis_overlap, get_or_decode_tile, tile_origin_and_shape,
    MAX_NAXIS,
};

// Compressed-write counterpart of hdu_image.rs::normalize_input_dtype.
// Inspects the HDU's BSCALE/BZERO (regular header cards, NOT
// Z-prefixed) and the input array's dtype:
//
//   - Input dtype matches ZBITPIX (e.g. i2 for ZBITPIX=16):
//     pass through unchanged.
//   - HDU has the unsigned-int trick (BSCALE=1 + BZERO=2^(n-1) or
//     -128 for ZBITPIX=8) AND input dtype matches the SCALED form
//     (e.g. u2 for ZBITPIX=16 + BZERO=32768): reverse_unsigned_trick
//     XORs the sign bit to produce the BITPIX-native bytes the
//     encoder needs.
//   - Otherwise (input doesn't match either): error.
//
// Integer ZBITPIX only — caller is responsible for skipping this
// for float HDUs (where BSCALE/BZERO would be a malformed file
// since the spec forbids them on floating-point arrays).
fn normalize_compressed_input_dtype(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    cards: &[String],
    zbitpix: i32,
) -> PyResult<Py<PyAny>> {
    let dtype = arr.getattr("dtype")?;
    let input_kind: String = dtype.getattr("kind")?.extract()?;
    let input_size: u64 = dtype.getattr("itemsize")?.extract()?;
    let expected = zbitpix_to_native_dtype(zbitpix)?;
    let (expected_kind, expected_size) = match expected {
        "u1" => ("u", 1u64),
        "i2" => ("i", 2),
        "i4" => ("i", 4),
        "i8" => ("i", 8),
        _ => unreachable!("integer ZBITPIX expected"),
    };

    // Fast path: input already in ZBITPIX-native dtype.
    if input_kind == expected_kind && input_size == expected_size {
        return Ok(arr.clone().unbind());
    }

    // Scaled-dtype input?  Check BSCALE/BZERO match the
    // unsigned-int trick for this ZBITPIX.
    let (bscale, bzero) = crate::hdu_image::parse_bscale_bzero(cards);
    let kind = crate::hdu_image::image_scaling_kind(zbitpix, bscale, bzero);
    if matches!(kind, crate::hdu_image::ScalingKind::UnsignedTrick) {
        // Expected scaled dtype is the opposite signedness.
        let (scaled_kind, scaled_size): (&str, u64) = match zbitpix {
            8 => ("i", 1),   // u1 stored ← i1 scaled (BZERO=-128)
            16 => ("u", 2),  // i2 stored ← u2 scaled (BZERO=32768)
            32 => ("u", 4),
            64 => ("u", 8),
            _ => unreachable!("unsigned trick on non-int ZBITPIX"),
        };
        if input_kind == scaled_kind && input_size == scaled_size {
            return crate::hdu_image::reverse_unsigned_trick(
                py, arr, zbitpix,
            );
        }
    }

    Err(PyValueError::new_err(format!(
        "compressed image write: input dtype '{}{}' does not match \
         the HDU's ZBITPIX={} (expected '{}{}'{})",
        input_kind, input_size, zbitpix, expected_kind, expected_size,
        if matches!(kind, crate::hdu_image::ScalingKind::UnsignedTrick) {
            match zbitpix {
                8 => " or scaled 'i1' (BZERO=-128)",
                16 => " or scaled 'u2' (BZERO=32768)",
                32 => " or scaled 'u4' (BZERO=2^31)",
                64 => " or scaled 'u8' (BZERO=2^63)",
                _ => "",
            }
        } else {
            ""
        },
    )))
}

// Per-tile row data captured during the encode loop.  For integer
// HDUs only `primary_nelem` / `primary_off` are meaningful (the
// other fields stay at their default values).  For float HDUs
// either the primary fields are non-zero (tile quantized cleanly)
// OR the fallback fields are non-zero (tile went to the GZIP
// fallback column).  `zscale` / `zzero` are the per-tile
// quantization parameters; they're meaningless when the fallback
// path fires but get written anyway since the row width is fixed.
pub(crate) struct TileRow {
    primary_nelem: u64,
    primary_off: u64,
    zscale: f64,
    zzero: f64,
    fallback_nelem: u64,
    fallback_off: u64,
}

// HDU-invariant context for the integer-tile encode helper.
// Everything that doesn't vary tile-to-tile.
pub(crate) struct IntTileCtx {
    algorithm: crate::zimage::CompressionAlgorithm,
    bytepix: u32,
    zbitpix: i32,
    inner_byte_width: u64,
    blocksize: u32,
    hcompress_scale: i32,
    // None → codec default (zlib level 6).  Only consulted by
    // GZIP_1 / GZIP_2 encoders; other algorithms ignore it.
    gzip_level: Option<u32>,
}

// HDU-invariant context for the float-tile encode helper.
// Carries the noise-estimation knobs (qlevel) and dither state
// (method + zdither0) alongside the algorithm parameters.
pub(crate) struct FloatTileCtx {
    algorithm: crate::zimage::CompressionAlgorithm,
    zbitpix: i32, // -32 or -64
    inner_byte_width: u64,
    blocksize: u32,
    hcompress_scale: i32,
    method: crate::zimage::quantize::DitherMethod,
    qlevel: f64,
    zdither0: i64,
    // GZIP level for the GZIP_1 lossless fallback path (and for
    // the primary algo when it's GZIP_1/GZIP_2).  None → codec
    // default (level 6).
    gzip_level: Option<u32>,
}

// Encode one integer tile: run it through the chosen algorithm,
// append the encoded bytes to `primary_heap`, return the row's
// descriptor.  The float fields of TileRow stay at their defaults
// (caller knows to ignore them for integer HDUs).
pub(crate) fn encode_tile_int(
    ctx: &IntTileCtx,
    tile_bytes: &[u8],
    tile_idx: u64,
    actual_shape: &[u64],
    n_pixels: usize,
    primary_heap: &mut Vec<u8>,
) -> PyResult<TileRow> {
    let encode_params = crate::zimage::AlgorithmEncodeParams {
        blocksize: ctx.blocksize,
        tile_shape_numpy: actual_shape,
        scale: ctx.hcompress_scale,
        gzip_level: ctx.gzip_level,
    };
    let encoded = crate::zimage::encode_tile_from_bytes(
        ctx.algorithm, tile_bytes, ctx.bytepix, n_pixels,
        ctx.zbitpix, encode_params,
    )?;
    if encoded.len() as u64 % ctx.inner_byte_width != 0 {
        return Err(PyValueError::new_err(format!(
            "internal: encoded tile {} bytes={} not a multiple \
             of inner_byte_width={}",
            tile_idx, encoded.len(), ctx.inner_byte_width,
        )));
    }
    let off = primary_heap.len() as u64;
    let nelem = encoded.len() as u64 / ctx.inner_byte_width;
    primary_heap.extend(encoded);
    Ok(TileRow {
        primary_nelem: nelem,
        primary_off: off,
        zscale: 0.0,
        zzero: 0.0,
        fallback_nelem: 0,
        fallback_off: 0,
    })
}

// Encode one float tile.  Convert big-endian float bytes to
// native floats, run quantize_float / quantize_double, then either:
//   - encode the quantized i32 stream through the chosen algorithm
//     and append to `primary_heap`, OR
//   - GZIP-compress the raw float bytes (lossless) and append to
//     `fallback_heap` (the "couldn't quantize" path).
// Per-pixel NaN handling: NaN inputs become NULL_VALUE_I32 in the
// quantized stream regardless of dither method (cfitsio's
// convention).  Exact zeros become ZERO_VALUE_I32 under DITHER_2.
pub(crate) fn encode_tile_float(
    ctx: &FloatTileCtx,
    tile_bytes: &[u8],
    tile_idx: u64,
    actual_shape: &[u64],
    n_pixels: usize,
    primary_heap: &mut Vec<u8>,
    fallback_heap: &mut Vec<u8>,
) -> PyResult<TileRow> {
    // Quantize.  nxpix = numpy-last (fast) axis; nypix = the rest.
    // 1-based tile index drives the dither seed.
    let nxpix = actual_shape[actual_shape.len() - 1] as usize;
    let nypix = if nxpix == 0 { 0 } else { n_pixels / nxpix };
    let row_1based = tile_idx + 1;
    let qt_opt = if ctx.zbitpix == -32 {
        let mut tile_f32: Vec<f32> = Vec::with_capacity(n_pixels);
        for chunk in tile_bytes.chunks_exact(4) {
            tile_f32.push(f32::from_be_bytes(chunk.try_into().unwrap()));
        }
        crate::zimage::quantize::quantize_float(
            &tile_f32, nxpix, nypix, Some(f32::NAN),
            ctx.qlevel, ctx.method, row_1based, ctx.zdither0,
        )
    } else {
        let mut tile_f64: Vec<f64> = Vec::with_capacity(n_pixels);
        for chunk in tile_bytes.chunks_exact(8) {
            tile_f64.push(f64::from_be_bytes(chunk.try_into().unwrap()));
        }
        crate::zimage::quantize::quantize_double(
            &tile_f64, nxpix, nypix, Some(f64::NAN),
            ctx.qlevel, ctx.method, row_1based, ctx.zdither0,
        )
    };

    if let Some(qt) = qt_opt {
        // Quantized successfully — encode the i32 stream through
        // the chosen algorithm (acts as if the input were a 32-bit
        // integer image).
        let mut i32_be: Vec<u8> = Vec::with_capacity(n_pixels * 4);
        for &v in &qt.idata {
            i32_be.extend_from_slice(&v.to_be_bytes());
        }
        let encode_params = crate::zimage::AlgorithmEncodeParams {
            blocksize: ctx.blocksize,
            tile_shape_numpy: actual_shape,
            scale: ctx.hcompress_scale,
            gzip_level: ctx.gzip_level,
        };
        let encoded = crate::zimage::encode_tile_from_bytes(
            ctx.algorithm, &i32_be, 4, n_pixels, 32,
            encode_params,
        )?;
        if encoded.len() as u64 % ctx.inner_byte_width != 0 {
            return Err(PyValueError::new_err(format!(
                "internal: encoded tile {} bytes={} not a multiple \
                 of inner_byte_width={}",
                tile_idx, encoded.len(), ctx.inner_byte_width,
            )));
        }
        let off = primary_heap.len() as u64;
        let nelem = encoded.len() as u64 / ctx.inner_byte_width;
        primary_heap.extend(encoded);
        Ok(TileRow {
            primary_nelem: nelem,
            primary_off: off,
            zscale: qt.bscale,
            zzero: qt.bzero,
            fallback_nelem: 0,
            fallback_off: 0,
        })
    } else {
        // Couldn't quantize (constant tile, range too wide, etc.)
        // — GZIP-compress the raw float bytes into the lossless
        // fallback column.  Primary descriptor stays empty so the
        // reader falls through to the GZIP fallback.
        let encoded =
            crate::zimage::gzip::encode_gzip1(tile_bytes, ctx.gzip_level)?;
        let off = fallback_heap.len() as u64;
        let nelem = encoded.len() as u64;
        fallback_heap.extend(encoded);
        Ok(TileRow {
            primary_nelem: 0,
            primary_off: 0,
            zscale: 1.0,
            zzero: 0.0,
            fallback_nelem: nelem,
            fallback_off: off,
        })
    }
}

// Encode a boundary tile for the quantized-float extend / __setitem__
// path.  The tile has already been decoded + combined-with-new-data
// upstream; this helper does the requantize-and-encode step.
//
// Dispatches on whether the existing tile was stored in the primary
// column (had a non-empty primary descriptor before the call) or in
// the GZIP fallback (primary nelem == 0).  Tiles stored in primary
// stay in primary — re-quantized with the EXISTING per-tile bscale/
// bzero so unchanged pixels round-trip exactly.  Tiles stored in
// the fallback stay in the fallback — re-encoded as GZIP_1 raw
// float bytes (the fallback always works, so we don't try to
// promote them back to primary).
//
// `combined_be` is the full re-formed tile in FITS big-endian float
// bytes (f4 or f8 matching `zbitpix`).  Returns a TileRow ready for
// descriptor emission; encoded bytes are appended to the right
// heap buffer (`primary_heap` for primary path, `fallback_heap`
// for fallback path).
#[allow(clippy::too_many_arguments)]
fn encode_quant_boundary_tile(
    algorithm: crate::zimage::CompressionAlgorithm,
    zbitpix: i32,
    inner_byte_width: u64,
    blocksize: u32,
    hcompress_scale: i32,
    gzip_level: Option<u32>,
    old_main_buf: &[u8],
    tile_idx: u64,
    row_width: u64,
    descriptor_size: u64,
    heap_is_q: bool,
    quant_setup: (
        crate::zimage::quantize::DitherMethod,
        i64,
        f64,
    ),
    combined_be: &[u8],
    actual_shape: &[u64],
    n_pixels: usize,
    primary_heap: &mut Vec<u8>,
    fallback_heap: &mut Vec<u8>,
) -> PyResult<TileRow> {
    let row_at = (tile_idx * row_width) as usize;
    let (old_p_nel, _old_p_off, old_zscale, old_zzero, _, _) =
        read_quant_descriptor_row(
            old_main_buf, row_at, heap_is_q, descriptor_size,
        );
    let (method, zdither0, _qlevel) = quant_setup;
    let row_1based = tile_idx + 1;

    if old_p_nel > 0 {
        // Was in primary: requantize with the existing per-tile
        // bscale/bzero (no compounding loss on unchanged pixels)
        // and encode via the algorithm.  Reject if any value
        // doesn't fit the existing scale.
        let i32_stream = if zbitpix == -32 {
            let floats: Vec<f32> = combined_be
                .chunks_exact(4)
                .map(|c| f32::from_be_bytes(c.try_into().unwrap()))
                .collect();
            crate::zimage::quantize::requantize_float_fixed_scale(
                &floats, old_zscale, old_zzero, method, row_1based,
                zdither0,
            )
        } else {
            let floats: Vec<f64> = combined_be
                .chunks_exact(8)
                .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
                .collect();
            crate::zimage::quantize::requantize_double_fixed_scale(
                &floats, old_zscale, old_zzero, method, row_1based,
                zdither0,
            )
        }
        .map_err(PyValueError::new_err)?;
        let mut i32_be: Vec<u8> = Vec::with_capacity(i32_stream.len() * 4);
        for &v in &i32_stream {
            i32_be.extend_from_slice(&v.to_be_bytes());
        }
        let encode_params = crate::zimage::AlgorithmEncodeParams {
            blocksize,
            tile_shape_numpy: actual_shape,
            scale: hcompress_scale,
            gzip_level,
        };
        let encoded = crate::zimage::encode_tile_from_bytes(
            algorithm, &i32_be, 4, n_pixels, 32, encode_params,
        )?;
        let primary_off_start = primary_heap.len() as u64;
        let nelem = encoded.len() as u64 / inner_byte_width;
        primary_heap.extend(encoded);
        Ok(TileRow {
            primary_nelem: nelem,
            primary_off: primary_off_start,
            zscale: old_zscale,
            zzero: old_zzero,
            fallback_nelem: 0,
            fallback_off: 0,
        })
    } else {
        // Was in fallback: stay in fallback (lossless raw float
        // bytes, GZIP_1).  ZSCALE/ZZERO are placeholder.
        let encoded =
            crate::zimage::gzip::encode_gzip1(combined_be, gzip_level)?;
        let fb_off = fallback_heap.len() as u64;
        let nelem = encoded.len() as u64;
        fallback_heap.extend(encoded);
        Ok(TileRow {
            primary_nelem: 0,
            primary_off: 0,
            zscale: 1.0,
            zzero: 0.0,
            fallback_nelem: nelem,
            fallback_off: fb_off,
        })
    }
}

// Bulk-write entry point for CompressedImageHDU.write.  Encodes every
// tile in RAM, then mutates the file (grow if non-last HDU, write
// descriptors + heap, update PCOUNT card).  Phase 7 supports GZIP_1
// and GZIP_2 with integer ZBITPIX (u1/i2/i4/i8).
//
// Order:
//   1. Parse header for shape, tile shape, algorithm, ZBITPIX, heap_format.
//   2. Validate the input ndarray (shape, dtype).
//   3. Encode all tiles into RAM → descriptors + heap.  (Validate-then-
//      mutate: any error from here back leaves the file untouched.)
//   4. Grow the file if (main_size + heap_size, block-padded) exceeds
//      the current padded data extent.  Use shift_file_tail when not
//      the last HDU; set_len when it is.
//   5. Write descriptors into the main data section + heap bytes
//      immediately after.
//   6. Rewrite the header in place with the updated PCOUNT.
//   7. Commit the in-memory cards and clear the tile cache.
//
// Taint semantics match the rest of the write paths: anything before
// the first file mutation can fail without tainting; failures inside
// the shift / write / header rewrite taint and the user has to
// close+reopen.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_compressed_image_data(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    cards: &[String],
    offsets: &Arc<HduOffsets>,
    file_handle: &FileHandle,
    layout: &FileLayout,
    tainted: &TaintFlag,
    cache: &TileCache,
    cards_arc: &Arc<Mutex<Vec<String>>>,
    cards_version: &Arc<AtomicU64>,
    quantize_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::Quantize>>
    >,
    compress_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::CompressionConfigKind>>
    >,
) -> PyResult<()> {
    check_not_tainted(tainted)?;

    // ----- header parse -----
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE")
        .ok_or_else(|| PyValueError::new_err(
            "compressed HDU missing ZCMPTYPE"
        ))?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;
    let (zbitpix, image_shape) = parse_compressed_image_shape(cards)?;

    // GZIP compression level (write-only param not recoverable
    // from the on-disk file).  When the user passed Gzip1/Gzip2
    // with an explicit level, the full config sits in
    // compress_config (see HduKind::CompressedImage); read out
    // just the level here.  For reopened HDUs or non-GZIP
    // algorithms the value is None → encoder uses codec default
    // (zlib level 6).
    let gzip_level: Option<u32> = compress_config
        .lock()
        .map_err(|_| PyIOError::new_err(
            "compress config lock poisoned",
        ))?
        .as_ref()
        .and_then(|c| c.gzip_level());
    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)"
        ));
    }
    let is_float = zbitpix < 0;
    // For integer ZBITPIX, bytepix matches the image pixel width.
    // For float ZBITPIX, we use the i32 width (4) for the encoded
    // primary stream — quantize_float/_double output i32 values.
    let bytepix: u32 = match zbitpix {
        8 => 1, 16 => 2, 32 => 4, 64 => 8,
        -32 | -64 => 4,
        other => return Err(PyValueError::new_err(format!(
            "unsupported ZBITPIX {} for compressed write", other
        ))),
    };
    let float_bytepix: u32 = match zbitpix {
        -32 => 4,
        -64 => 8,
        _ => 0,
    };
    let tile_shape = parse_tile_shape(cards, &image_shape);
    let n_tiles = compute_n_tiles(&image_shape, &tile_shape);

    // Determine heap_format from TFORM1.  Encoder needs to know
    // 'P' (8-byte descriptors, u32 fields) vs 'Q' (16-byte, u64).
    // Inner type matters too: '1PB'/'1QB' (byte) is used by
    // GZIP/RICE/HCOMPRESS, '1PI'/'1QI' (i16 short) by PLIO.  For
    // VLA descriptors the `nelements` field counts ELEMENTS of the
    // inner type, not bytes — so we divide encoded.len() by
    // `inner_byte_width` when filling descriptors.
    let tform1 = parse_string_keyword(cards, "TFORM1")
        .ok_or_else(|| PyValueError::new_err(
            "compressed HDU missing TFORM1"
        ))?;
    let tform1_trim = tform1.trim();
    let heap_is_q = tform1_trim.starts_with("1Q")
        || tform1_trim.starts_with('Q');
    let descriptor_size: u64 = if heap_is_q { 16 } else { 8 };
    let inner_byte_width: u64 = tform_vla_inner_byte_width(tform1_trim)
        .unwrap_or(1);

    // ----- MaskedArray entry -----
    // If `data` is a numpy.ma.MaskedArray, fill masked positions
    // with the appropriate sentinel (NaN for float HDUs; ZBLANK
    // from the header for integer HDUs).  No-op for plain ndarrays.
    let unmasked =
        crate::hdu_image::unwrap_masked_input(py, data, cards, true)?;
    let data = unmasked.bind(py);

    // ----- input ndarray validation -----
    let np = py.import("numpy")?;
    let ascontig0 = np.call_method1("ascontiguousarray", (data,))?;
    let in_shape: Vec<usize> = ascontig0.getattr("shape")?.extract()?;
    let expected_shape: Vec<usize> =
        image_shape.iter().map(|&d| d as usize).collect();
    if in_shape != expected_shape {
        return Err(PyValueError::new_err(format!(
            "compressed image write: input shape {:?} != image shape {:?}",
            in_shape, expected_shape
        )));
    }

    // ----- unsigned-int trick reverse-transform -----
    // If the HDU has BSCALE=1 + BZERO=2^(n-1) (or -128 for i1) AND
    // the input dtype is the scaled form (u2/u4/u8 for BITPIX
    // 16/32/64, i1 for BITPIX 8), reverse-transform via XOR so the
    // encoder receives the on-disk (BITPIX-native) bytes.  Fast
    // path: input already matches BITPIX → pass through unchanged.
    // Float HDUs (is_float) never trigger this path — BSCALE/BZERO
    // are only checked for integer ZBITPIX.
    let ascontig_owned = if !is_float {
        normalize_compressed_input_dtype(
            py, &ascontig0, cards, zbitpix,
        )?
    } else {
        ascontig0.clone().unbind()
    };
    let ascontig = ascontig_owned.bind(py);

    // Convert each tile to the expected on-disk byte order in one
    // pass via numpy.  Integer ZBITPIX → u1/i2/i4/i8; float ZBITPIX
    // → f4/f8 (so we can re-extract the floats for quantization).
    let be_dtype = match zbitpix {
        8  => ">u1",
        16 => ">i2",
        32 => ">i4",
        64 => ">i8",
        -32 => ">f4",
        -64 => ">f8",
        other => return Err(PyValueError::new_err(format!(
            "unsupported ZBITPIX {} for compressed write", other
        ))),
    };

    // ----- algorithm-specific encode params -----
    // RICE_1 needs BLOCKSIZE from the header; HCOMPRESS_1 needs
    // SCALE.  Both keys are emitted by create_compressed_image_hdu_impl
    // so they're present on rustfits-written files; defaults match
    // cfitsio when absent (32 / 0 = lossless).
    let (blocksize_opt, _bytepix_header_opt) = parse_rice_params(cards);
    let hcompress_scale = parse_hcompress_scale(cards);

    // ----- detect unquantized-float mode -----
    // Signal: float HDU + single-column schema (TFIELDS=1).
    // Works for both freshly-created HDUs (we set TFIELDS=1 at
    // create time when quantize=None) and reopened files.
    // Unquantized-float files store raw GZIP-compressed float
    // bytes directly in COMPRESSED_DATA — same single-column
    // layout as integer HDUs.  Astropy's quantize_level=0 output
    // matches this exactly.
    let tfields = parse_keyword(cards, "TFIELDS").unwrap_or(0) as u64;
    let is_unquantized_float = is_float && tfields == 1;
    let is_quantized_float = is_float && !is_unquantized_float;

    // ----- float-only quantize setup -----
    // For quantized float HDUs we need:
    //   - DitherMethod from ZQUANTIZ
    //   - ZDITHER0 seed
    //   - qlevel from the create-time Quantize config (the FITS
    //     spec records method+seed only, not the level)
    // Skipped entirely for unquantized floats: there's no dither
    // stream and no quantization step.
    let (quant_method, zdither0, qlevel) = if is_quantized_float {
        let zq = parse_string_keyword(cards, "ZQUANTIZ");
        let method = crate::zimage::quantize::parse_dither_method(
            zq.as_deref(),
        )?.ok_or_else(|| PyValueError::new_err(
            "quantized-float HDU has ZQUANTIZ='NONE' but the schema \
             has 4 columns (ZSCALE/ZZERO present).  Malformed file."
        ))?;
        let zd = parse_keyword(cards, "ZDITHER0").unwrap_or(1).max(1);
        let level = quantize_config.lock()
            .map_err(|_| PyIOError::new_err(
                "quantize config lock poisoned"
            ))?.as_ref().map(|q| q.level).unwrap_or(4.0);
        (Some(method), zd, level)
    } else {
        (None, 0, 0.0)
    };

    // ----- per-tile encode loop -----
    //
    // Three dispatch modes:
    //   - Integer HDU OR unquantized float: route through
    //     `encode_tile_int` (single-column primary heap, no
    //     ZSCALE/ZZERO, no fallback).  For unquantized floats,
    //     bytepix is the float width (4 or 8); GZIP_1/GZIP_2
    //     treat the bytes opaquely.
    //   - Quantized float: route through `encode_tile_float`
    //     which may quantize → primary_heap OR fall back to
    //     GZIP-compressed raw float bytes → fallback_heap.
    // HDU-invariant params travel in IntTileCtx / FloatTileCtx,
    // leaving this loop to just extract tile bytes and call the
    // right helper.
    let int_ctx = if !is_quantized_float {
        // For unquantized floats, use float_bytepix; for integer
        // HDUs, use the integer bytepix.
        let effective_bytepix = if is_unquantized_float {
            float_bytepix
        } else {
            bytepix
        };
        Some(IntTileCtx {
            algorithm,
            bytepix: effective_bytepix,
            zbitpix,
            inner_byte_width,
            blocksize: blocksize_opt.unwrap_or(32),
            hcompress_scale,
            gzip_level,
        })
    } else {
        None
    };
    let float_ctx = if is_quantized_float {
        Some(FloatTileCtx {
            algorithm,
            zbitpix,
            inner_byte_width,
            blocksize: blocksize_opt.unwrap_or(32),
            hcompress_scale,
            method: quant_method.unwrap(),
            qlevel,
            zdither0,
            gzip_level,
        })
    } else {
        None
    };

    let mut rows: Vec<TileRow> = Vec::with_capacity(n_tiles as usize);
    let mut primary_heap: Vec<u8> = Vec::new();
    let mut fallback_heap: Vec<u8> = Vec::new();

    let mut origin_buf = [0u64; MAX_NAXIS];
    let mut shape_buf = [0u64; MAX_NAXIS];
    for tile_idx in 0..n_tiles {
        let d = tile_origin_and_shape(
            tile_idx, &image_shape, &tile_shape,
            &mut origin_buf, &mut shape_buf,
        );
        let origin = &origin_buf[..d];
        let actual_shape = &shape_buf[..d];
        let slice_objs: Vec<Bound<'_, PySlice>> = origin.iter()
            .zip(actual_shape.iter())
            .map(|(&o, &s)| PySlice::new(
                py, o as isize, (o + s) as isize, 1,
            ))
            .collect();
        let slice_tuple = PyTuple::new(py, &slice_objs)?;
        let tile_view = ascontig.get_item(slice_tuple)?;
        let tile_be = np.call_method1(
            "ascontiguousarray", (tile_view, be_dtype),
        )?;
        let tile_bytes_py = tile_be.call_method0("tobytes")?;
        let tile_bytes: Vec<u8> = tile_bytes_py.extract()?;
        let n_pixels = actual_shape.iter().product::<u64>() as usize;
        let pixel_width_bytes = if is_float {
            float_bytepix as usize
        } else {
            bytepix as usize
        };
        let expected_bytes = n_pixels * pixel_width_bytes;
        if tile_bytes.len() != expected_bytes {
            return Err(PyValueError::new_err(format!(
                "internal: tile {} bytes={} expected {}",
                tile_idx, tile_bytes.len(), expected_bytes
            )));
        }

        let row = if is_quantized_float {
            encode_tile_float(
                float_ctx.as_ref().unwrap(),
                &tile_bytes, tile_idx, actual_shape, n_pixels,
                &mut primary_heap, &mut fallback_heap,
            )?
        } else {
            // Integer HDU OR unquantized float — both route
            // through encode_tile_int.  Unquantized float feeds
            // raw float bytes through GZIP_1/GZIP_2 (the encoder
            // treats them opaquely).
            encode_tile_int(
                int_ctx.as_ref().unwrap(),
                &tile_bytes, tile_idx, actual_shape, n_pixels,
                &mut primary_heap,
            )?
        };
        rows.push(row);
    }

    // Combine the two heaps.  Bump fallback offsets so they land
    // in the right position of the concatenated heap.  Only the
    // quantized-float path produces a non-empty fallback heap;
    // for integer + unquantized-float paths fallback_heap is
    // empty and this loop is a no-op.
    let primary_size = primary_heap.len() as u64;
    let fallback_size = fallback_heap.len() as u64;
    let total_heap_bytes = primary_size + fallback_size;
    for row in rows.iter_mut() {
        if row.fallback_nelem > 0 {
            row.fallback_off += primary_size;
        }
    }

    // ----- compute file extents -----
    let data_offset = offsets.data_offset();
    // Main data section = NAXIS1 (row width) × n_tiles.  Row
    // width depends on the column layout: single descriptor for
    // integer HDUs and unquantized-float HDUs (both TFIELDS=1);
    // primary + ZSCALE + ZZERO + fallback descriptor for
    // quantized-float HDUs (TFIELDS=4).
    let row_width: u64 = if is_quantized_float {
        descriptor_size + 8 + 8 + descriptor_size
    } else {
        descriptor_size
    };
    let main_bytes = row_width.saturating_mul(n_tiles);
    let new_data_size = main_bytes.saturating_add(total_heap_bytes);
    let new_padded = if new_data_size == 0 {
        0
    } else {
        ((new_data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            * BLOCK_SIZE as u64
    };
    // Current data extent = file size between data_offset and the next
    // HDU's header_offset (or EOF if this is the last HDU).  We can
    // approximate via "is the data section padded enough?" — we know
    // the HDU's allocated size from layout/EOF.
    let current_hdu_end = {
        let guard = layout.hdus.lock()
            .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
        // Find this HDU's index by matching offsets.  We don't have
        // an index passed in, but Arc equality works since they share.
        let mut found_next: Option<u64> = None;
        let mut found_self = false;
        for h in guard.iter() {
            if found_self {
                found_next = Some(h.header_offset());
                break;
            }
            if Arc::ptr_eq(h, offsets) {
                found_self = true;
            }
        }
        match found_next {
            Some(end) => end,
            None => {
                // Last HDU — current_hdu_end is the file length.
                let g = lock_file(file_handle)?;
                let f = g.as_ref()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.metadata()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?
                    .len()
            }
        }
    };
    let current_padded_data = current_hdu_end.saturating_sub(data_offset);
    let new_hdu_end = data_offset + new_padded;

    // ----- grow if needed -----
    if new_padded > current_padded_data {
        let delta = new_padded - current_padded_data;
        // file_len lets us tell if there are bytes after this HDU
        // that need to be shifted forward.
        let file_len = {
            let g = lock_file(file_handle)?;
            let f = g.as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            // Non-last HDU — shift the tail forward; later HDU
            // offsets bump via the shared FileLayout.
            shift_file_tail_and_update_offsets(
                file_handle, layout, current_hdu_end, delta, tainted,
            )?;
        } else {
            // Last HDU — extend the file in place.
            let mut g = lock_file(file_handle)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // ----- write descriptors + heap -----
    {
        let mut g = lock_file(file_handle)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed write: seek to data_offset failed: {}", e))
            })?;
        // Build the per-row descriptor table.  Layout:
        //   Integer: primary descriptor only.
        //   Float:   primary descriptor + ZSCALE (8 BE) + ZZERO (8
        //            BE) + fallback descriptor.
        // P-format descriptors are 8 bytes (u32 nelem BE + u32 off BE);
        // Q-format are 16 bytes (u64 + u64 BE).  cfitsio agrees on
        // both layouts.
        let mut desc_buf: Vec<u8> =
            Vec::with_capacity((row_width * n_tiles) as usize);
        let push_desc = |buf: &mut Vec<u8>, nel: u64, off: u64|
            -> PyResult<()>
        {
            if heap_is_q {
                buf.extend_from_slice(&nel.to_be_bytes());
                buf.extend_from_slice(&off.to_be_bytes());
            } else {
                if nel > u32::MAX as u64 || off > u32::MAX as u64 {
                    return Err(PyValueError::new_err(format!(
                        "compressed write: P-descriptor overflow \
                         (nelem={}, offset={}); use heap_format='Q' \
                         for heaps > 4 GB",
                        nel, off,
                    )));
                }
                buf.extend_from_slice(&(nel as u32).to_be_bytes());
                buf.extend_from_slice(&(off as u32).to_be_bytes());
            }
            Ok(())
        };
        for row in &rows {
            if let Err(e) = push_desc(
                &mut desc_buf, row.primary_nelem, row.primary_off,
            ) {
                tainted.store(true, Ordering::Release);
                return Err(e);
            }
            if is_quantized_float {
                desc_buf.extend_from_slice(&row.zscale.to_be_bytes());
                desc_buf.extend_from_slice(&row.zzero.to_be_bytes());
                if let Err(e) = push_desc(
                    &mut desc_buf, row.fallback_nelem, row.fallback_off,
                ) {
                    tainted.store(true, Ordering::Release);
                    return Err(e);
                }
            }
        }
        f.write_all(&desc_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed write: descriptor write failed: {}", e))
        })?;
        // Heap = primary_heap followed by fallback_heap (already
        // composed into one logical stream above).  Write both
        // halves consecutively.
        f.write_all(&primary_heap).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed write: primary heap write failed: {}", e))
        })?;
        f.write_all(&fallback_heap).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed write: fallback heap write failed: {}", e))
        })?;
        // Zero-pad to the block boundary so the data section is FITS-
        // conforming.  Cheap: at most BLOCK_SIZE-1 bytes.
        let written = main_bytes + total_heap_bytes;
        if new_padded > written {
            let pad = vec![0u8; (new_padded - written) as usize];
            f.write_all(&pad).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed write: data pad write failed: {}", e))
            })?;
        }
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed write: flush failed: {}", e))
        })?;
    }

    // ----- header rewrite (PCOUNT update) -----
    let cards_guard = crate::hdu::CardsWriteGuard::from_parts(
        cards_arc.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?,
        cards_version,
    );
    let mut new_cards = cards_guard.clone_cards();
    set_pcount_in_cards(&mut new_cards, total_heap_bytes);
    {
        let mut g = lock_file(file_handle)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        // header_offset stays valid; PCOUNT rewrite doesn't change
        // the header block count.
        let header_offset = offsets.header_offset();
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "PCOUNT header write failed: {}; close + reopen", e))
        })?;
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "PCOUNT header flush failed: {}; close + reopen", e))
        })?;
    }
    cards_guard.commit(new_cards);

    // Any tiles cached from earlier reads are now stale.
    cache.clear();

    Ok(())
}

// Update an existing standard-keyword int card to a new value in
// place, preserving its position and (best-effort) comment text.
// No-op if the card isn't present — caller must guarantee it
// exists, which holds for the cards we update at extend time
// (NAXIS2 / PCOUNT / ZNAXISn are all mandatory).
fn update_int_card(cards: &mut Vec<String>, key: &str, value: i64) {
    let key_uc = key.to_uppercase();
    if let Some(idx) = cards.iter().position(|c| {
        c.len() >= 8 && c[..8].trim().to_uppercase() == key_uc
    }) {
        // Best-effort preserve the comment after the '/'.
        let existing = &cards[idx];
        let comment = existing
            .find('/')
            .map(|p| existing[p + 1..].trim().to_string())
            .unwrap_or_default();
        cards[idx] = crate::header::card_int(key, value, &comment);
    }
}

// Append new rows along the slow axis (numpy axis 0) of a
// tile-compressed image.  Existing tiles outside the last tile row
// are preserved untouched; the partial last tile row (if any) gets
// re-encoded to absorb new data; truly new tile rows are encoded
// fresh.  See CLAUDE.md "Compressed extend" for the full design.
//
// On-disk mechanics:
//   1. Pre-read existing main descriptor table + heap into RAM (we
//      have to move them; reading first means an I/O failure here
//      leaves the file unchanged).
//   2. Update boundary tile descriptors in the in-memory main buf
//      to point at re-encoded heap bytes (which live in the
//      appended portion of the new heap).
//   3. Grow file (shift later HDUs if non-last; set_len if last).
//   4. Write:
//        data_offset                        : updated main table
//        data_offset + new_main_bytes       : old heap (relocated)
//        + old_pcount                       : appended heap (boundary
//                                              re-encoded + new
//                                              tile bytes)
//      + block-padding.
//   5. Rewrite header to update NAXIS2 + PCOUNT + ZNAXIS<last>.
//
// Heap layout note: the old boundary-tile bytes stay in the old
// heap (now orphaned — descriptors no longer point at them).
// Slightly bloats files when boundary re-encoding happens, but
// keeps the file logically valid and is dramatically simpler than
// rewriting the whole heap with the old tiles' bytes removed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extend_compressed_image_data(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    cards: &[String],
    offsets: &Arc<HduOffsets>,
    file_handle: &FileHandle,
    layout: &FileLayout,
    tainted: &TaintFlag,
    cache: &TileCache,
    cards_arc: &Arc<Mutex<Vec<String>>>,
    cards_version: &Arc<AtomicU64>,
    quantize_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::Quantize>>,
    >,
    compress_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::CompressionConfigKind>>,
    >,
) -> PyResult<()> {
    check_not_tainted(tainted)?;

    // ----- header parse -----
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZCMPTYPE")
    })?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;
    let (zbitpix, old_image_shape) = parse_compressed_image_shape(cards)?;
    if old_image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)",
        ));
    }
    let is_float = zbitpix < 0;
    let tfields = parse_keyword(cards, "TFIELDS").unwrap_or(0) as u64;

    // GZIP level (write-only param; None → codec default).
    let gzip_level: Option<u32> = compress_config
        .lock()
        .map_err(|_| PyIOError::new_err(
            "compress config lock poisoned",
        ))?
        .as_ref()
        .and_then(|c| c.gzip_level());
    // Three schema dispatches based on (is_float, tfields):
    //   integer:          single-col (TFIELDS=1)
    //   unquantized float: single-col (TFIELDS=1, quantize=None)
    //   quantized float:  4-col (TFIELDS=4) — re-uses existing
    //                     per-tile ZSCALE/ZZERO via
    //                     requantize_*_fixed_scale to avoid
    //                     compounding loss on unchanged pixels
    let is_quantized_float = is_float && tfields == 4;
    let is_unquantized_float = is_float && !is_quantized_float;

    let bytepix: u32 = match zbitpix {
        8 => 1,
        16 => 2,
        32 => 4,
        64 => 8,
        -32 => 4,
        -64 => 8,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported ZBITPIX {} for compressed extend",
                other
            )))
        }
    };

    let tile_shape = parse_tile_shape(cards, &old_image_shape);
    let old_n_tiles = compute_n_tiles(&old_image_shape, &tile_shape);

    let tform1 = parse_string_keyword(cards, "TFORM1").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing TFORM1")
    })?;
    let tform1_trim = tform1.trim();
    let heap_is_q =
        tform1_trim.starts_with("1Q") || tform1_trim.starts_with('Q');
    let descriptor_size: u64 = if heap_is_q { 16 } else { 8 };
    let inner_byte_width: u64 =
        tform_vla_inner_byte_width(tform1_trim).unwrap_or(1);

    let old_pcount =
        parse_keyword(cards, "PCOUNT").unwrap_or(0).max(0) as u64;

    let (blocksize_opt, _) = parse_rice_params(cards);
    let blocksize = blocksize_opt.unwrap_or(32);
    let hcompress_scale = parse_hcompress_scale(cards);

    // Quantized-float HDUs use a 4-column row layout:
    //   primary descriptor + ZSCALE (8 BE) + ZZERO (8 BE) +
    //   fallback descriptor.
    // Integer + unquantized-float HDUs use a single descriptor.
    let row_width: u64 = if is_quantized_float {
        descriptor_size + 8 + 8 + descriptor_size
    } else {
        descriptor_size
    };
    let old_main_bytes = row_width.saturating_mul(old_n_tiles);
    let data_offset = offsets.data_offset();

    // For the quantized-float path we need dither method + seed + qlevel.
    // These are read from the header (ZQUANTIZ / ZDITHER0) plus the
    // create-time Quantize config (the FITS spec records method +
    // seed only, not the qlevel; reopened HDUs fall back to 4.0).
    // None for integer + unquantized-float HDUs.
    let quant_setup: Option<(crate::zimage::quantize::DitherMethod, i64, f64)> =
        if is_quantized_float {
            let zq = parse_string_keyword(cards, "ZQUANTIZ");
            let method = crate::zimage::quantize::parse_dither_method(
                zq.as_deref(),
            )?
            .ok_or_else(|| {
                PyValueError::new_err(
                    "quantized-float HDU has ZQUANTIZ='NONE' but the \
                     4-column schema is present (malformed file).",
                )
            })?;
            let zd =
                parse_keyword(cards, "ZDITHER0").unwrap_or(1).max(1);
            let level = quantize_config
                .lock()
                .map_err(|_| {
                    PyIOError::new_err("quantize config lock poisoned")
                })?
                .as_ref()
                .map(|q| q.level)
                .unwrap_or(4.0);
            Some((method, zd, level))
        } else {
            None
        };

    // ----- MaskedArray entry -----
    let unmasked =
        crate::hdu_image::unwrap_masked_input(py, data, cards, true)?;
    let data = unmasked.bind(py);

    // ----- input validation -----
    let np = py.import("numpy")?;
    let ascontig0 = np.call_method1("ascontiguousarray", (data,))?;
    let in_shape: Vec<usize> = ascontig0.getattr("shape")?.extract()?;
    let naxis = old_image_shape.len();
    if in_shape.len() != naxis {
        return Err(PyValueError::new_err(format!(
            "compressed extend: input has {} axes, HDU has {}",
            in_shape.len(),
            naxis,
        )));
    }
    if in_shape[0] == 0 {
        return Err(PyValueError::new_err(
            "compressed extend: data.shape[0] must be > 0",
        ));
    }
    // Only axis 0 may grow; remaining axes must match exactly.
    for axis in 1..naxis {
        if in_shape[axis] as u64 != old_image_shape[axis] {
            return Err(PyValueError::new_err(format!(
                "compressed extend: data shape[{}]={} != image \
                 shape[{}]={} (only the slow axis (numpy axis 0) \
                 can grow)",
                axis, in_shape[axis], axis, old_image_shape[axis],
            )));
        }
    }

    // Reverse-transform if the HDU has unsigned-int trick BSCALE/BZERO.
    // For unquantized-float and integer-no-trick HDUs this is a pass-
    // through.  Float HDUs never carry BSCALE/BZERO (forbidden by
    // the spec) so we skip the helper.
    let ascontig_owned = if !is_float {
        normalize_compressed_input_dtype(py, &ascontig0, cards, zbitpix)?
    } else {
        ascontig0.clone().unbind()
    };
    let ascontig = ascontig_owned.bind(py);

    // ----- compute new layout -----
    let old_naxis0 = old_image_shape[0];
    let added = in_shape[0] as u64;
    let new_naxis0 = old_naxis0 + added;
    let mut new_image_shape = old_image_shape.clone();
    new_image_shape[0] = new_naxis0;
    let new_n_tiles = compute_n_tiles(&new_image_shape, &tile_shape);

    let t_r = tile_shape[0];
    let n_old_tile_rows_slow = (old_naxis0 + t_r - 1) / t_r;
    let n_tiles_per_row_slow: u64 = {
        let mut prod = 1u64;
        for ax in 1..naxis {
            let n_along = (old_image_shape[ax] + tile_shape[ax] - 1)
                / tile_shape[ax];
            prod *= n_along;
        }
        prod
    };

    // Boundary tiles: the LAST tile row of the OLD image is a
    // boundary iff old_naxis0 is not a multiple of T_r.  Those
    // tiles need re-encoding because their actual shape grows in
    // the NEW image (old partial → fuller or full).  When
    // old_naxis0 % T_r == 0 there are no boundary tiles.
    let has_boundary = old_naxis0 > 0 && old_naxis0 % t_r != 0;
    let boundary_range: Option<(u64, u64)> = if has_boundary {
        let start = (n_old_tile_rows_slow - 1) * n_tiles_per_row_slow;
        let end = n_old_tile_rows_slow * n_tiles_per_row_slow;
        Some((start, end))
    } else {
        None
    };
    // Tiles in [first_new_tile, new_n_tiles) are entirely new (no
    // overlap with any old tile).  Tiles in [0, first_new_tile) are
    // either unchanged or boundary.
    let first_new_tile = match boundary_range {
        Some((_, end)) => end,
        None => old_n_tiles,
    };

    // ----- encode setup -----
    let int_ctx = IntTileCtx {
        algorithm,
        bytepix,
        zbitpix,
        inner_byte_width,
        blocksize,
        hcompress_scale,
        gzip_level,
    };
    let be_dtype = match zbitpix {
        8 => ">u1",
        16 => ">i2",
        32 => ">i4",
        64 => ">i8",
        -32 => ">f4",
        -64 => ">f8",
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported ZBITPIX {} for compressed extend",
                other
            )))
        }
    };
    let _ = is_unquantized_float; // (signal kept; used in helpers below)

    // ----- read existing main descriptor table early -----
    // For quantized-float HDUs we need each boundary tile's existing
    // ZSCALE/ZZERO (to reuse the same per-tile quantization scale
    // and avoid compounding loss) AND whether it was stored in the
    // primary or GZIP fallback column.  Both are in main_buf.  For
    // integer/unquantized we don't strictly need it before encoding,
    // but reading early keeps the I/O sequencing simple: all
    // pre-write reads happen first, then encoding (which is pure
    // CPU), then writes.
    let mut old_main_buf: Vec<u8> = vec![0; old_main_bytes as usize];
    if old_main_bytes > 0 {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut old_main_buf).map_err(|e| {
            PyIOError::new_err(format!(
                "compressed extend: read old main: {}",
                e
            ))
        })?;
    }

    // Quantization context (only meaningful for quantized-float HDUs).
    // Used by get_or_decode_tile to dispatch the dequantize step on
    // boundary tile reads.
    let cols_lookup = find_data_columns(cards)?;
    let quant_ctx_opt = if is_quantized_float {
        build_quant_context(cards, &cols_lookup)?
    } else {
        None
    };

    // FloatTileCtx for the quantized-float path's new-tile encoding
    // (mirrors what write_compressed_image_data uses).
    let float_ctx = quant_setup.map(|(method, zd, level)| FloatTileCtx {
        algorithm,
        zbitpix,
        inner_byte_width,
        blocksize,
        hcompress_scale,
        method,
        qlevel: level,
        zdither0: zd,
        gzip_level,
    });

    // ----- encode boundary tiles -----
    // For each boundary tile, decode the existing partial tile via
    // the cache machinery, slice the new data portion that falls
    // into the same tile, concatenate, and re-encode.  Integer +
    // unquantized-float HDUs go through encode_tile_int; quantized-
    // float HDUs go through requantize_with_fixed_scale (preserves
    // the existing per-tile bscale/bzero so unchanged pixels suffer
    // no compounding loss) and may use the GZIP fallback for tiles
    // already stored there.
    let mut appended_heap: Vec<u8> = Vec::new();
    let mut appended_fallback_heap: Vec<u8> = Vec::new();
    let mut boundary_updates: Vec<(u64, TileRow)> = Vec::new();
    let mut new_descs: Vec<TileRow> = Vec::new();

    if let Some((b_start, b_end)) = boundary_range {
        let cols = &cols_lookup;
        let theap = parse_keyword(cards, "THEAP")
            .map(|x| x.max(0) as u64)
            .unwrap_or(row_width * old_n_tiles);
        let mut origin_buf = [0u64; MAX_NAXIS];
        let mut new_shape_buf = [0u64; MAX_NAXIS];
        let mut _scratch_origin = [0u64; MAX_NAXIS];
        let mut old_shape_buf = [0u64; MAX_NAXIS];
        for tile_idx in b_start..b_end {
            let d = tile_origin_and_shape(
                tile_idx, &new_image_shape, &tile_shape,
                &mut origin_buf, &mut new_shape_buf,
            );
            tile_origin_and_shape(
                tile_idx, &old_image_shape, &tile_shape,
                &mut _scratch_origin, &mut old_shape_buf,
            );
            let origin = &origin_buf[..d];
            let new_actual_shape = &new_shape_buf[..d];
            let old_actual_shape = &old_shape_buf[..d];
            // For quantized-float HDUs the decoder runs the
            // dequantize step (i32 stored → f4/f8 physical),
            // returning float bytes.  For integer/unquantized-float
            // it returns BITPIX-native bytes.
            let (dec_bytepix, dec_stored_zbitpix) = if is_quantized_float {
                (4u32, 32i32)
            } else {
                (bytepix, zbitpix)
            };
            let old_bytes_arc = get_or_decode_tile(
                py,
                cache,
                file_handle,
                tainted,
                tile_idx,
                data_offset,
                row_width,
                theap,
                cols,
                algorithm,
                old_actual_shape,
                dec_bytepix,
                blocksize,
                dec_stored_zbitpix,
                zbitpix,
                quant_ctx_opt.as_ref(),
                false,
            )?;

            // Slice the corresponding portion of new data:
            //   axis 0: data rows [0, new_actual_shape[0] - old_actual_shape[0])
            //   axes 1..N: the tile's column extent (origin[ax]..origin[ax]+shape[ax])
            let new_rows_in_tile =
                new_actual_shape[0] - old_actual_shape[0];
            let slice_objs: Vec<Bound<'_, PySlice>> = (0..naxis)
                .map(|ax| {
                    if ax == 0 {
                        PySlice::new(py, 0, new_rows_in_tile as isize, 1)
                    } else {
                        let o = origin[ax] as isize;
                        let s = new_actual_shape[ax] as isize;
                        PySlice::new(py, o, o + s, 1)
                    }
                })
                .collect();
            let slice_tuple = PyTuple::new(py, &slice_objs)?;
            let new_part_view = ascontig.get_item(slice_tuple)?;
            // Convert new portion to BE bytes
            let new_part_be =
                np.call_method1("ascontiguousarray", (new_part_view, be_dtype))?;
            let new_part_bytes_py = new_part_be.call_method0("tobytes")?;
            let new_part_be_bytes: Vec<u8> = new_part_bytes_py.extract()?;

            // Convert old bytes (native-endian, from cache) to BE
            let native_dtype = zbitpix_to_native_dtype(zbitpix)?;
            let old_pyb = PyBytes::new(py, &old_bytes_arc);
            let old_arr_flat =
                np.call_method1("frombuffer", (old_pyb, native_dtype))?;
            let old_shape_tuple = PyTuple::new(py, old_actual_shape)?;
            let old_arr =
                old_arr_flat.call_method1("reshape", (old_shape_tuple,))?;
            let old_be =
                np.call_method1("ascontiguousarray", (old_arr, be_dtype))?;
            let old_be_bytes: Vec<u8> =
                old_be.call_method0("tobytes")?.extract()?;

            // Byte-concat (axis 0 concat in C-order = byte concat
            // because axis 0 is the OUTER dim).
            let mut combined_be: Vec<u8> = Vec::with_capacity(
                old_be_bytes.len() + new_part_be_bytes.len(),
            );
            combined_be.extend_from_slice(&old_be_bytes);
            combined_be.extend_from_slice(&new_part_be_bytes);

            let n_pixels: usize =
                new_actual_shape.iter().product::<u64>() as usize;
            let row = if is_quantized_float {
                encode_quant_boundary_tile(
                    algorithm,
                    zbitpix,
                    inner_byte_width,
                    blocksize,
                    hcompress_scale,
                    gzip_level,
                    &old_main_buf,
                    tile_idx,
                    row_width,
                    descriptor_size,
                    heap_is_q,
                    quant_setup.unwrap(),
                    &combined_be,
                    new_actual_shape,
                    n_pixels,
                    &mut appended_heap,
                    &mut appended_fallback_heap,
                )?
            } else {
                encode_tile_int(
                    &int_ctx,
                    &combined_be,
                    tile_idx,
                    new_actual_shape,
                    n_pixels,
                    &mut appended_heap,
                )?
            };
            boundary_updates.push((tile_idx, row));
        }
    }

    // ----- encode truly-new tile rows -----
    let mut new_origin_buf = [0u64; MAX_NAXIS];
    let mut new_shape_buf2 = [0u64; MAX_NAXIS];
    for tile_idx in first_new_tile..new_n_tiles {
        let d = tile_origin_and_shape(
            tile_idx, &new_image_shape, &tile_shape,
            &mut new_origin_buf, &mut new_shape_buf2,
        );
        let origin = &new_origin_buf[..d];
        let new_actual_shape = &new_shape_buf2[..d];
        // The tile's data lives entirely in the input array,
        // offset by old_naxis0 along axis 0.
        let data_axis0_start = origin[0].saturating_sub(old_naxis0);
        let data_axis0_end = data_axis0_start + new_actual_shape[0];
        let slice_objs: Vec<Bound<'_, PySlice>> = (0..naxis)
            .map(|ax| {
                if ax == 0 {
                    PySlice::new(
                        py,
                        data_axis0_start as isize,
                        data_axis0_end as isize,
                        1,
                    )
                } else {
                    let o = origin[ax] as isize;
                    let s = new_actual_shape[ax] as isize;
                    PySlice::new(py, o, o + s, 1)
                }
            })
            .collect();
        let slice_tuple = PyTuple::new(py, &slice_objs)?;
        let tile_view = ascontig.get_item(slice_tuple)?;
        let tile_be =
            np.call_method1("ascontiguousarray", (tile_view, be_dtype))?;
        let tile_bytes_py = tile_be.call_method0("tobytes")?;
        let tile_bytes: Vec<u8> = tile_bytes_py.extract()?;
        let n_pixels: usize =
            new_actual_shape.iter().product::<u64>() as usize;
        let row = if is_quantized_float {
            // Truly-new tile rows: standard quantize_float (may
            // fall to the GZIP fallback if range too wide).  The
            // existing encode_tile_float helper already handles
            // both paths and writes to the right heap.
            encode_tile_float(
                float_ctx.as_ref().unwrap(),
                &tile_bytes,
                tile_idx,
                new_actual_shape,
                n_pixels,
                &mut appended_heap,
                &mut appended_fallback_heap,
            )?
        } else {
            encode_tile_int(
                &int_ctx,
                &tile_bytes,
                tile_idx,
                new_actual_shape,
                n_pixels,
                &mut appended_heap,
            )?
        };
        new_descs.push(row);
    }

    // ----- combine the two appended heaps -----
    // For quantized-float HDUs the appended heap is
    // [appended_primary; appended_fallback].  Bump fallback offsets
    // by primary size so descriptors point at the right slot.  No-op
    // for integer/unquantized-float (fallback heap is empty).
    let appended_primary_size = appended_heap.len() as u64;
    appended_heap.extend(appended_fallback_heap.drain(..));
    for (_, row) in boundary_updates.iter_mut() {
        if row.fallback_nelem > 0 {
            row.fallback_off += appended_primary_size;
        }
    }
    for row in new_descs.iter_mut() {
        if row.fallback_nelem > 0 {
            row.fallback_off += appended_primary_size;
        }
    }

    // ----- shift descriptor offsets into the combined-heap frame -----
    // appended_heap offsets start at 0 (relative to the start of
    // appended).  The combined heap is [old_heap; appended], so
    // absolute offsets are old_pcount + offset_in_appended.
    for (_, row) in boundary_updates.iter_mut() {
        if row.primary_nelem > 0 {
            row.primary_off += old_pcount;
        }
        if row.fallback_nelem > 0 {
            row.fallback_off += old_pcount;
        }
    }
    for row in new_descs.iter_mut() {
        if row.primary_nelem > 0 {
            row.primary_off += old_pcount;
        }
        if row.fallback_nelem > 0 {
            row.fallback_off += old_pcount;
        }
    }

    // ----- compute new file extents -----
    let new_main_bytes = row_width.saturating_mul(new_n_tiles);
    let new_pcount = old_pcount + appended_heap.len() as u64;
    let new_data_size = new_main_bytes + new_pcount;
    let new_padded = if new_data_size == 0 {
        0
    } else {
        ((new_data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            * BLOCK_SIZE as u64
    };
    let current_hdu_end = {
        let guard = layout
            .hdus
            .lock()
            .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
        let mut found_next: Option<u64> = None;
        let mut found_self = false;
        for h in guard.iter() {
            if found_self {
                found_next = Some(h.header_offset());
                break;
            }
            if Arc::ptr_eq(h, offsets) {
                found_self = true;
            }
        }
        match found_next {
            Some(end) => end,
            None => {
                let g = lock_file(file_handle)?;
                let f = g
                    .as_ref()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.metadata()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?
                    .len()
            }
        }
    };
    let current_padded_data = current_hdu_end.saturating_sub(data_offset);
    let new_hdu_end = data_offset + new_padded;

    // ----- read existing heap into RAM -----
    // (main_buf was already read earlier so quantized boundary
    // tiles could inspect existing ZSCALE/ZZERO.  This step reads
    // the heap so we can relocate it past the new descriptor
    // table.  Done BEFORE the grow so an I/O failure leaves the
    // file untouched.)
    let mut old_heap_buf: Vec<u8> = vec![0; old_pcount as usize];
    if old_pcount > 0 {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset + old_main_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut old_heap_buf).map_err(|e| {
            PyIOError::new_err(format!(
                "compressed extend: read old heap: {}",
                e
            ))
        })?;
    }

    // Apply boundary descriptor updates to old_main_buf in place.
    // Descriptor format: P → u32 nelem BE + u32 off BE (8 bytes);
    // Q → u64 nelem BE + u64 off BE (16 bytes).  For quantized-
    // float HDUs each row is the full 4-column layout (primary +
    // ZSCALE + ZZERO + fallback).
    for (tile_idx, row) in &boundary_updates {
        let desc_offset = (tile_idx * row_width) as usize;
        if is_quantized_float {
            write_quant_descriptor_row(
                &mut old_main_buf,
                desc_offset,
                heap_is_q,
                descriptor_size,
                row.primary_nelem,
                row.primary_off,
                row.zscale,
                row.zzero,
                row.fallback_nelem,
                row.fallback_off,
            )?;
        } else {
            write_descriptor(
                &mut old_main_buf,
                desc_offset,
                heap_is_q,
                row.primary_nelem,
                row.primary_off,
            )?;
        }
    }

    // Serialize new tile descriptors into a buffer.
    let mut new_descs_buf: Vec<u8> = Vec::with_capacity(
        (new_descs.len() as u64 * row_width) as usize,
    );
    for row in &new_descs {
        let mut tmp = vec![0u8; row_width as usize];
        if is_quantized_float {
            write_quant_descriptor_row(
                &mut tmp,
                0,
                heap_is_q,
                descriptor_size,
                row.primary_nelem,
                row.primary_off,
                row.zscale,
                row.zzero,
                row.fallback_nelem,
                row.fallback_off,
            )?;
        } else {
            write_descriptor(
                &mut tmp,
                0,
                heap_is_q,
                row.primary_nelem,
                row.primary_off,
            )?;
        }
        new_descs_buf.extend_from_slice(&tmp);
    }

    // ----- grow the file if needed -----
    if new_padded > current_padded_data {
        let delta = new_padded - current_padded_data;
        let file_len = {
            let g = lock_file(file_handle)?;
            let f = g
                .as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            shift_file_tail_and_update_offsets(
                file_handle,
                layout,
                current_hdu_end,
                delta,
                tainted,
            )?;
        } else {
            let mut g = lock_file(file_handle)?;
            let f = g
                .as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // ----- write the new layout to disk -----
    {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset)).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: seek to data_offset failed: {}",
                e
            ))
        })?;
        // Main descriptor table: [existing (with updated boundary
        // descriptors)] + [new tile descriptors].
        f.write_all(&old_main_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: write old main: {}",
                e
            ))
        })?;
        f.write_all(&new_descs_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: write new descs: {}",
                e
            ))
        })?;
        // Heap at the new position: [old heap (relocated)] +
        // [appended (boundary re-encodes + new tiles)].
        f.seek(SeekFrom::Start(data_offset + new_main_bytes)).map_err(
            |e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed extend: seek to new heap: {}",
                    e
                ))
            },
        )?;
        f.write_all(&old_heap_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: write old heap: {}",
                e
            ))
        })?;
        f.write_all(&appended_heap).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: write appended heap: {}",
                e
            ))
        })?;
        // Block-pad to FITS block boundary.
        let written = new_main_bytes + new_pcount;
        if new_padded > written {
            let pad = vec![0u8; (new_padded - written) as usize];
            f.write_all(&pad).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed extend: pad: {}",
                    e
                ))
            })?;
        }
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!("compressed extend: flush: {}", e))
        })?;
    }

    // ----- header rewrite (PCOUNT + NAXIS2 + ZNAXIS<last>) -----
    let cards_guard = crate::hdu::CardsWriteGuard::from_parts(
        cards_arc.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?,
        cards_version,
    );
    let mut new_cards = cards_guard.clone_cards();
    set_pcount_in_cards(&mut new_cards, new_pcount);
    update_int_card(&mut new_cards, "NAXIS2", new_n_tiles as i64);
    // numpy axis 0 = slowest = FITS NAXIS<znaxis> (highest-numbered).
    let znaxis = naxis;
    let zaxis_key = format!("ZNAXIS{}", znaxis);
    update_int_card(&mut new_cards, &zaxis_key, new_naxis0 as i64);
    {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = offsets.header_offset();
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: header write failed: {}; close + reopen",
                e
            ))
        })?;
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: header flush failed: {}; close + reopen",
                e
            ))
        })?;
    }
    cards_guard.commit(new_cards);

    // Boundary tiles' content changed (and any cached new tiles
    // would be inconsistent with the new heap layout anyway).
    cache.clear();

    Ok(())
}

// Write a single VLA descriptor (nelem + offset, P or Q format) into
// `buf` at `at`.  Used by extend's descriptor table updates.
pub(crate) fn write_descriptor(
    buf: &mut [u8],
    at: usize,
    heap_is_q: bool,
    nel: u64,
    off: u64,
) -> PyResult<()> {
    if heap_is_q {
        buf[at..at + 8].copy_from_slice(&nel.to_be_bytes());
        buf[at + 8..at + 16].copy_from_slice(&off.to_be_bytes());
    } else {
        if nel > u32::MAX as u64 || off > u32::MAX as u64 {
            return Err(PyValueError::new_err(format!(
                "compressed extend: P-descriptor overflow \
                 (nelem={}, offset={}); use heap_format='Q' for \
                 heaps > 4 GB",
                nel, off,
            )));
        }
        buf[at..at + 4].copy_from_slice(&(nel as u32).to_be_bytes());
        buf[at + 4..at + 8].copy_from_slice(&(off as u32).to_be_bytes());
    }
    Ok(())
}

// Read a single VLA descriptor (nelem + offset) from a buffer at `at`.
// (Named with the `_buf` suffix to disambiguate from the existing
// file-reading `read_descriptor` upstream.)  Used by extend/
// __setitem__ on quantized-float HDUs to inspect existing per-tile
// descriptors (specifically: is the tile in the primary column or
// the GZIP fallback column?).
pub(crate) fn read_descriptor_from_buf(
    buf: &[u8], at: usize, heap_is_q: bool,
) -> (u64, u64) {
    if heap_is_q {
        let nel = u64::from_be_bytes(buf[at..at + 8].try_into().unwrap());
        let off = u64::from_be_bytes(buf[at + 8..at + 16].try_into().unwrap());
        (nel, off)
    } else {
        let nel = u32::from_be_bytes(buf[at..at + 4].try_into().unwrap())
            as u64;
        let off = u32::from_be_bytes(buf[at + 4..at + 8].try_into().unwrap())
            as u64;
        (nel, off)
    }
}

// Quantized-float HDUs use a 4-column row layout: primary
// descriptor + ZSCALE (1D) + ZZERO (1D) + GZIP fallback descriptor.
// This helper writes one such row into `buf` at `at` (the start of
// the row).  `descriptor_size` is 8 for P-format, 16 for Q-format.
fn write_quant_descriptor_row(
    buf: &mut [u8],
    at: usize,
    heap_is_q: bool,
    descriptor_size: u64,
    primary_nelem: u64,
    primary_off: u64,
    zscale: f64,
    zzero: f64,
    fallback_nelem: u64,
    fallback_off: u64,
) -> PyResult<()> {
    write_descriptor(buf, at, heap_is_q, primary_nelem, primary_off)?;
    let zs_at = at + descriptor_size as usize;
    buf[zs_at..zs_at + 8].copy_from_slice(&zscale.to_be_bytes());
    buf[zs_at + 8..zs_at + 16].copy_from_slice(&zzero.to_be_bytes());
    let fb_at = zs_at + 16;
    write_descriptor(buf, fb_at, heap_is_q, fallback_nelem, fallback_off)?;
    Ok(())
}

// Read a quantized-float row (primary descriptor + ZSCALE + ZZERO
// + fallback descriptor) from `buf` at `at`.  Returns
// (primary_nelem, primary_off, zscale, zzero, fallback_nelem,
// fallback_off).  Inverse of write_quant_descriptor_row.
fn read_quant_descriptor_row(
    buf: &[u8],
    at: usize,
    heap_is_q: bool,
    descriptor_size: u64,
) -> (u64, u64, f64, f64, u64, u64) {
    let (p_nel, p_off) = read_descriptor_from_buf(buf, at, heap_is_q);
    let zs_at = at + descriptor_size as usize;
    let zscale =
        f64::from_be_bytes(buf[zs_at..zs_at + 8].try_into().unwrap());
    let zzero =
        f64::from_be_bytes(buf[zs_at + 8..zs_at + 16].try_into().unwrap());
    let fb_at = zs_at + 16;
    let (f_nel, f_off) = read_descriptor_from_buf(buf, fb_at, heap_is_q);
    (p_nel, p_off, zscale, zzero, f_nel, f_off)
}

// In-place modification of compressed image pixels via numpy-style
// slicing.  For each tile that overlaps the selection, decode the
// existing tile, apply the user's value to the affected portion,
// re-encode, and append the new bytes to the heap.  Old bytes for
// modified tiles become orphans (left in place; descriptors no
// longer reference them).
//
// Simpler than extend: no descriptor table growth (n_tiles is
// fixed), no heap relocation (the existing heap is preserved at its
// position).  Only:
//   - Each modified tile's descriptor is updated to point at its
//     new bytes appended at the end of the heap.
//   - PCOUNT grows by the total new-tile-bytes appended.
//   - If the new total exceeds the padded data section, grow the
//     file (shift later HDUs if non-last; set_len if last).
//
// Supports: contiguous slices, stepped slices, single integer
// indices, ellipsis, mixed combinations (all reuse the same
// AxisSlice + axis_overlap machinery that __getitem__ uses).
// RHS: numpy scalar / 0-d array (broadcast across selection), or
// N-d array whose shape matches the selection.  Unsigned-int
// trick HDUs accept the scaled dtype (e.g. u2 on a BITPIX=16 +
// BZERO=32768 HDU) and reverse-transform via
// normalize_compressed_input_dtype.
//
// Quantized-float HDUs (4-column schema) are deferred — the
// per-tile ZSCALE/ZZERO would need recomputation and the dither
// stream contract for SUBTRACTIVE_DITHER_* must hold across
// modifications.  Same scope cut as extend.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setitem_compressed_image(
    py: Python<'_>,
    cards: &[String],
    offsets: &Arc<HduOffsets>,
    file_handle: &FileHandle,
    layout: &FileLayout,
    tainted: &TaintFlag,
    cache: &TileCache,
    cards_arc: &Arc<Mutex<Vec<String>>>,
    cards_version: &Arc<AtomicU64>,
    quantize_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::Quantize>>,
    >,
    compress_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::CompressionConfigKind>>,
    >,
    key: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    check_not_tainted(tainted)?;

    // ----- header parse (mirrors extend) -----
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZCMPTYPE")
    })?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;
    let (zbitpix, image_shape) = parse_compressed_image_shape(cards)?;
    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)",
        ));
    }
    let is_float = zbitpix < 0;
    let tfields = parse_keyword(cards, "TFIELDS").unwrap_or(0) as u64;
    // Schema dispatch: quantized-float HDUs use 4-col rows;
    // integer + unquantized-float use 1-col rows.
    let is_quantized_float = is_float && tfields == 4;

    // GZIP level (write-only param; None → codec default).
    let gzip_level: Option<u32> = compress_config
        .lock()
        .map_err(|_| PyIOError::new_err(
            "compress config lock poisoned",
        ))?
        .as_ref()
        .and_then(|c| c.gzip_level());

    let bytepix: u32 = match zbitpix {
        8 => 1,
        16 => 2,
        32 => 4,
        64 => 8,
        -32 => 4,
        -64 => 8,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported ZBITPIX {} for compressed __setitem__",
                other
            )))
        }
    };

    let tile_shape = parse_tile_shape(cards, &image_shape);
    let n_tiles = compute_n_tiles(&image_shape, &tile_shape);

    let tform1 = parse_string_keyword(cards, "TFORM1").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing TFORM1")
    })?;
    let tform1_trim = tform1.trim();
    let heap_is_q =
        tform1_trim.starts_with("1Q") || tform1_trim.starts_with('Q');
    let descriptor_size: u64 = if heap_is_q { 16 } else { 8 };
    let inner_byte_width: u64 =
        tform_vla_inner_byte_width(tform1_trim).unwrap_or(1);

    let old_pcount =
        parse_keyword(cards, "PCOUNT").unwrap_or(0).max(0) as u64;

    let (blocksize_opt, _) = parse_rice_params(cards);
    let blocksize = blocksize_opt.unwrap_or(32);
    let hcompress_scale = parse_hcompress_scale(cards);

    // Row width: 4-col for quantized-float, single-col otherwise.
    let row_width: u64 = if is_quantized_float {
        descriptor_size + 8 + 8 + descriptor_size
    } else {
        descriptor_size
    };
    let main_bytes = row_width.saturating_mul(n_tiles);
    let data_offset = offsets.data_offset();

    // Quantization setup (only for quantized-float path).
    let quant_setup: Option<(crate::zimage::quantize::DitherMethod, i64, f64)> =
        if is_quantized_float {
            let zq = parse_string_keyword(cards, "ZQUANTIZ");
            let method = crate::zimage::quantize::parse_dither_method(
                zq.as_deref(),
            )?
            .ok_or_else(|| {
                PyValueError::new_err(
                    "quantized-float HDU has ZQUANTIZ='NONE' but the \
                     4-column schema is present (malformed file).",
                )
            })?;
            let zd =
                parse_keyword(cards, "ZDITHER0").unwrap_or(1).max(1);
            let level = quantize_config
                .lock()
                .map_err(|_| {
                    PyIOError::new_err("quantize config lock poisoned")
                })?
                .as_ref()
                .map(|q| q.level)
                .unwrap_or(4.0);
            Some((method, zd, level))
        } else {
            None
        };

    // ----- parse slice key -----
    let slices = normalize_slice_key(key, &image_shape)?;
    // Output shape: drop is_int axes.  For all-int → empty shape
    // (RHS must be a scalar in that case).
    let output_shape: Vec<u64> = slices
        .iter()
        .filter(|s| !s.is_int)
        .map(|s| s.count)
        .collect();
    // Zero-count anywhere = empty selection = no-op (same as
    // numpy's `arr[5:5] = ...` semantics).
    if slices.iter().any(|s| s.count == 0) {
        return Ok(());
    }

    // ----- MaskedArray entry -----
    // If value is a numpy.ma.MaskedArray, fill masked positions
    // with the appropriate sentinel before the rest of the
    // pipeline.  No-op for plain ndarray / scalar RHS.
    let unmasked_value =
        crate::hdu_image::unwrap_masked_input(py, value, cards, true)?;
    let value = unmasked_value.bind(py);

    // ----- RHS validation + dtype normalization -----
    let np = py.import("numpy")?;
    let value_arr = np.call_method1("asarray", (value,))?;
    let value_shape: Vec<u64> = value_arr.getattr("shape")?.extract()?;
    let is_scalar_rhs = value_shape.is_empty();

    let value_norm: Py<PyAny> = if is_scalar_rhs {
        value_arr.clone().unbind()
    } else {
        // Shape must match the selection exactly (no broadcasting
        // beyond the scalar case — matches numpy's stricter
        // assignment rules).
        if value_shape != output_shape {
            return Err(PyValueError::new_err(format!(
                "compressed __setitem__: value shape {:?} does not \
                 match selection shape {:?}",
                value_shape, output_shape,
            )));
        }
        // Reverse-transform unsigned-int trick if needed.
        if !is_float {
            normalize_compressed_input_dtype(
                py, &value_arr, cards, zbitpix,
            )?
        } else {
            value_arr.clone().unbind()
        }
    };
    let value_bound = value_norm.bind(py);

    // ----- encode setup -----
    let int_ctx = IntTileCtx {
        algorithm,
        bytepix,
        zbitpix,
        inner_byte_width,
        blocksize,
        hcompress_scale,
        gzip_level,
    };
    let be_dtype = match zbitpix {
        8 => ">u1",
        16 => ">i2",
        32 => ">i4",
        64 => ">i8",
        -32 => ">f4",
        -64 => ">f8",
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported ZBITPIX {} for compressed __setitem__",
                other
            )))
        }
    };
    let native_dtype = zbitpix_to_native_dtype(zbitpix)?;

    let cols = find_data_columns(cards)?;
    let theap = parse_keyword(cards, "THEAP")
        .map(|x| x.max(0) as u64)
        .unwrap_or(row_width * n_tiles);

    // Quantization context (only meaningful for quantized-float
    // HDUs; runs the dequantize step inside get_or_decode_tile).
    let quant_ctx_opt = if is_quantized_float {
        build_quant_context(cards, &cols)?
    } else {
        None
    };

    // ----- read main descriptor table early -----
    // For quantized-float HDUs we need each affected tile's
    // existing ZSCALE/ZZERO and primary/fallback column status to
    // pick the right re-encode path.  For integer/unquantized
    // we'd read main_buf later anyway; doing it here unifies the
    // structure.
    let mut main_buf: Vec<u8> = vec![0; main_bytes as usize];
    if main_bytes > 0 {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut main_buf).map_err(|e| {
            PyIOError::new_err(format!(
                "compressed __setitem__: read main table: {}",
                e
            ))
        })?;
    }

    // ----- per-tile re-encode for overlapping tiles -----
    let mut appended_heap: Vec<u8> = Vec::new();
    let mut appended_fallback_heap: Vec<u8> = Vec::new();
    let mut tile_updates: Vec<(u64, TileRow)> = Vec::new();

    let mut setitem_origin_buf = [0u64; MAX_NAXIS];
    let mut setitem_shape_buf = [0u64; MAX_NAXIS];
    for tile_idx in 0..n_tiles {
        let d = tile_origin_and_shape(
            tile_idx, &image_shape, &tile_shape,
            &mut setitem_origin_buf, &mut setitem_shape_buf,
        );
        let origin = &setitem_origin_buf[..d];
        let actual_shape = &setitem_shape_buf[..d];
        // Build per-axis tile_indexer + output_indexer using the
        // same machinery __getitem__ uses.
        let mut tile_indexers: Vec<Bound<PyAny>> =
            Vec::with_capacity(slices.len());
        let mut output_indexers: Vec<Bound<PyAny>> = Vec::new();
        let mut overlapping = true;
        for (axis_idx, axis) in slices.iter().enumerate() {
            match axis_overlap(
                py,
                origin[axis_idx],
                actual_shape[axis_idx],
                axis,
            )? {
                Some(ov) => {
                    tile_indexers.push(ov.tile_indexer);
                    if let Some(out) = ov.output_indexer {
                        output_indexers.push(out);
                    }
                }
                None => {
                    overlapping = false;
                    break;
                }
            }
        }
        if !overlapping {
            continue;
        }

        // Decode the existing tile.  For quantized-float HDUs the
        // decoder runs dequantize, returning f4/f8 native bytes;
        // for integer/unquantized-float it returns BITPIX-native
        // bytes directly.  frombuffer returns a read-only view;
        // .copy() gives us a writable ndarray.
        let (dec_bytepix, dec_stored_zbitpix) = if is_quantized_float {
            (4u32, 32i32)
        } else {
            (bytepix, zbitpix)
        };
        let old_bytes_arc = get_or_decode_tile(
            py,
            cache,
            file_handle,
            tainted,
            tile_idx,
            data_offset,
            row_width,
            theap,
            &cols,
            algorithm,
            actual_shape,
            dec_bytepix,
            blocksize,
            dec_stored_zbitpix,
            zbitpix,
            quant_ctx_opt.as_ref(),
            false,
        )?;
        let pyb = PyBytes::new(py, &old_bytes_arc);
        let arr_flat =
            np.call_method1("frombuffer", (pyb, native_dtype))?;
        let shape_tuple = PyTuple::new(py, actual_shape)?;
        let arr_view = arr_flat.call_method1("reshape", (shape_tuple,))?;
        let tile_arr = arr_view.call_method0("copy")?;

        // Apply RHS to the tile's selected pixels.  For scalar
        // RHS, numpy broadcasts the value.  For array RHS, slice
        // the value with the output indexer to extract the
        // tile-specific portion.
        let tile_idx_tuple = PyTuple::new(py, &tile_indexers)?;
        if is_scalar_rhs {
            tile_arr.set_item(tile_idx_tuple, value_bound.clone())?;
        } else {
            let out_idx_tuple = PyTuple::new(py, &output_indexers)?;
            let rhs_for_tile = value_bound.get_item(out_idx_tuple)?;
            tile_arr.set_item(tile_idx_tuple, rhs_for_tile)?;
        }

        // Cast to BE bytes and re-encode (dispatch on schema).
        let tile_be =
            np.call_method1("ascontiguousarray", (tile_arr, be_dtype))?;
        let tile_bytes: Vec<u8> =
            tile_be.call_method0("tobytes")?.extract()?;
        let n_pixels: usize = actual_shape.iter().product::<u64>() as usize;
        let row = if is_quantized_float {
            encode_quant_boundary_tile(
                algorithm,
                zbitpix,
                inner_byte_width,
                blocksize,
                hcompress_scale,
                gzip_level,
                &main_buf,
                tile_idx,
                row_width,
                descriptor_size,
                heap_is_q,
                quant_setup.unwrap(),
                &tile_bytes,
                actual_shape,
                n_pixels,
                &mut appended_heap,
                &mut appended_fallback_heap,
            )?
        } else {
            encode_tile_int(
                &int_ctx,
                &tile_bytes,
                tile_idx,
                actual_shape,
                n_pixels,
                &mut appended_heap,
            )?
        };
        tile_updates.push((tile_idx, row));
    }

    // No tiles actually overlapped (e.g. out-of-bounds slice).
    // Per numpy convention, this is a silent no-op.
    if tile_updates.is_empty() {
        return Ok(());
    }

    // ----- combine primary + fallback appended heaps -----
    // (No-op for integer/unquantized-float — fallback_heap is
    // empty.  Quantized-float may have both.)
    let appended_primary_size = appended_heap.len() as u64;
    appended_heap.extend(appended_fallback_heap.drain(..));
    for (_, row) in tile_updates.iter_mut() {
        if row.fallback_nelem > 0 {
            row.fallback_off += appended_primary_size;
        }
    }
    // Shift descriptor offsets into the absolute-heap frame.
    // appended_heap sits at old_pcount in the combined heap.
    for (_, row) in tile_updates.iter_mut() {
        if row.primary_nelem > 0 {
            row.primary_off += old_pcount;
        }
        if row.fallback_nelem > 0 {
            row.fallback_off += old_pcount;
        }
    }

    // Apply descriptor updates to the already-read main_buf in place.
    for (tile_idx, row) in &tile_updates {
        let desc_offset = (tile_idx * row_width) as usize;
        if is_quantized_float {
            write_quant_descriptor_row(
                &mut main_buf,
                desc_offset,
                heap_is_q,
                descriptor_size,
                row.primary_nelem,
                row.primary_off,
                row.zscale,
                row.zzero,
                row.fallback_nelem,
                row.fallback_off,
            )?;
        } else {
            write_descriptor(
                &mut main_buf,
                desc_offset,
                heap_is_q,
                row.primary_nelem,
                row.primary_off,
            )?;
        }
    }

    // ----- compute new file extents -----
    let new_pcount = old_pcount + appended_heap.len() as u64;
    let new_data_size = main_bytes + new_pcount;
    let new_padded = if new_data_size == 0 {
        0
    } else {
        ((new_data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            * BLOCK_SIZE as u64
    };
    let current_hdu_end = {
        let guard = layout
            .hdus
            .lock()
            .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
        let mut found_next: Option<u64> = None;
        let mut found_self = false;
        for h in guard.iter() {
            if found_self {
                found_next = Some(h.header_offset());
                break;
            }
            if Arc::ptr_eq(h, offsets) {
                found_self = true;
            }
        }
        match found_next {
            Some(end) => end,
            None => {
                let g = lock_file(file_handle)?;
                let f = g
                    .as_ref()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.metadata()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?
                    .len()
            }
        }
    };
    let current_padded_data = current_hdu_end.saturating_sub(data_offset);
    let new_hdu_end = data_offset + new_padded;

    // ----- grow file if needed -----
    if new_padded > current_padded_data {
        let delta = new_padded - current_padded_data;
        let file_len = {
            let g = lock_file(file_handle)?;
            let f = g
                .as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            shift_file_tail_and_update_offsets(
                file_handle,
                layout,
                current_hdu_end,
                delta,
                tainted,
            )?;
        } else {
            let mut g = lock_file(file_handle)?;
            let f = g
                .as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // ----- write updated main + appended heap to disk -----
    {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        // Main descriptor table (with updated tile descriptors).
        f.seek(SeekFrom::Start(data_offset)).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: seek to data_offset: {}",
                e
            ))
        })?;
        f.write_all(&main_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: write main: {}",
                e
            ))
        })?;
        // Appended heap bytes at end of existing heap.
        if !appended_heap.is_empty() {
            f.seek(SeekFrom::Start(
                data_offset + main_bytes + old_pcount,
            ))
            .map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed __setitem__: seek to heap end: {}",
                    e
                ))
            })?;
            f.write_all(&appended_heap).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed __setitem__: write appended heap: {}",
                    e
                ))
            })?;
        }
        // Block-pad to FITS block boundary.
        let written_end = main_bytes + new_pcount;
        if new_padded > written_end {
            let pad = vec![0u8; (new_padded - written_end) as usize];
            f.write_all(&pad).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed __setitem__: pad: {}",
                    e
                ))
            })?;
        }
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: flush: {}",
                e
            ))
        })?;
    }

    // ----- header rewrite (PCOUNT only — shape unchanged) -----
    let cards_guard = crate::hdu::CardsWriteGuard::from_parts(
        cards_arc.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?,
        cards_version,
    );
    let mut new_cards = cards_guard.clone_cards();
    set_pcount_in_cards(&mut new_cards, new_pcount);
    {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = offsets.header_offset();
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: header write: {}; close + reopen",
                e
            ))
        })?;
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: header flush: {}; close + reopen",
                e
            ))
        })?;
    }
    cards_guard.commit(new_cards);

    // Modified tiles are now stale in the cache.
    cache.clear();

    Ok(())
}

