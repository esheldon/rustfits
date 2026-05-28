// Compressed-image read path: whole-image read, per-tile decode +
// cache, slice walking, and the descriptor/heap byte readers.

use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::types::{PyBytes, PySlice, PyTuple};

use crate::common::{
    check_not_tainted,
    FileHandle, RawBuffer, TaintFlag,
};
use crate::hdu_image::{
    normalize_slice_key, AxisSlice,
};
use crate::zimage::CompressionAlgorithm;

use super::hdu::TileCache;
use super::meta::{
    zbitpix_to_native_dtype, CompressedImageMeta, ZimageDataColumns,
    ZimageQuantContext,
};

// Top-level entry point invoked from CompressedImageHDU::read.
// Walks the BINTABLE one tile at a time, decoding each via the
// algorithm-specific code in src/zimage/, and assembling the
// tiles into the output ndarray.  Applies BSCALE/BZERO via the
// shared image-side scaling machinery so a scaled compressed
// HDU returns the same dtype as an equivalent uncompressed one.
//
// Current limits (will lift later):
//   - RICE_1 / GZIP_1 / GZIP_2 supported; HCOMPRESS_1 / PLIO_1 are
//     Phase 6
//   - Integer ZBITPIX only (Phase 5 adds quantized floats)
//   - mask_blank=True rejected (ZBLANK handling is a follow-up)
pub(crate) fn read_compressed_image_data(
    py: Python<'_>,
    meta: &CompressedImageMeta,
    data_offset: u64,
    file_handle: &FileHandle,
    tainted: &TaintFlag,
    cache: &TileCache,
    scale: bool,
    mask_blank: bool,
) -> PyResult<Py<PyAny>> {
    check_not_tainted(tainted)?;
    let zbitpix = meta.zbitpix;
    let image_shape = meta.image_shape.as_slice();
    let tile_shape = meta.tile_shape.as_slice();
    let algorithm = meta.algorithm;
    let blocksize = meta.blocksize;
    let bytepix = meta.bytepix;
    let smooth = meta.smooth;
    let naxis1 = meta.naxis1;
    let theap = meta.theap;
    let cols = &meta.cols;
    let quant = meta.quant.as_ref();
    let n_tiles = meta.n_tiles;

    // mask_blank parallels the uncompressed ImageHDU behavior, but
    // uses ZBLANK instead of BLANK (per the FITS Tile Compression
    // Convention).  Forbidden on float ZBITPIX — the spec says
    // BLANK is integer-only, NaN serves the same role for floats.
    // (For quantized-float compressed HDUs, the ZBLANK card stores
    // cfitsio's NaN sentinel value -2147483647 in i32 stored
    // space, but that's separate from the integer-pixel mask
    // semantics; NaN is already preserved on dequantize, so float
    // mask_blank stays rejected here too.)
    if mask_blank && zbitpix < 0 {
        return Err(PyValueError::new_err(format!(
            "mask_blank=True is not valid on float ZBITPIX ({}); the \
             FITS standard forbids BLANK on floating-point arrays \
             (NaN serves that role).  Use mask_blank=False, or \
             post-process with numpy.isnan.",
            zbitpix
        )));
    }

    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)"
        ));
    }
    // Decoder parameters.  Two distinct "ZBITPIX" values are in
    // play: `zbitpix` is the *image-side* output dtype (8/16/32/64
    // for integer images, -32/-64 for quantized float); the
    // decoder needs to know the *stored integer* dtype, which is
    // the same as zbitpix for integer images but is always 32
    // (i32) for the quantized-float path.  Same idea for bytepix:
    // 1/2/4/8 for integer images, always 4 for quantized float.
    let stored_zbitpix: i32 = if zbitpix < 0 { 32 } else { zbitpix };

    // Allocate output ndarray of the right shape + native dtype.
    let np = py.import("numpy")?;
    let dtype_str = zbitpix_to_native_dtype(zbitpix)?;
    let shape_tuple = PyTuple::new(py, image_shape)?;
    let out_arr = np.call_method1("empty", (shape_tuple, dtype_str))?;
    // Zero the array so that any test that asserts initial state
    // (or any future code that exits early) sees a clean value.
    out_arr.call_method0("fill")
        .or_else(|_| {
            // fill() requires an argument; pass 0.
            out_arr.call_method1("fill", (0i32,))
        })?;

    // Iterate tiles.  Each BINTABLE row = one tile; rows are
    // ordered FITS-row-major (numpy-last-axis fastest).  Each
    // tile is fetched via the cache (decoded on miss, served from
    // RAM on hit).  File I/O happens per tile under the file lock;
    // decode happens outside the lock so concurrent threads aren't
    // serialized through long decode runs.
    let mut origin_buf = [0u64; MAX_NAXIS];
    let mut shape_buf = [0u64; MAX_NAXIS];
    for tile_idx in 0..n_tiles {
        let d = tile_origin_and_shape(
            tile_idx, &image_shape, &tile_shape,
            &mut origin_buf, &mut shape_buf,
        );
        let origin = &origin_buf[..d];
        let actual_shape = &shape_buf[..d];
        let tile_bytes = get_or_decode_tile(
            py, cache, file_handle, tainted, tile_idx, data_offset,
            naxis1, theap, cols, algorithm, actual_shape,
            bytepix, blocksize, stored_zbitpix,
            zbitpix, quant, smooth,
        )?;
        place_tile_bytes_into_output(
            py, &out_arr, &tile_bytes, dtype_str,
            actual_shape, origin,
        )?;
    }

    // Compute the blank mask BEFORE scaling (stored space, per
    // the FITS spec — ZBLANK names the raw on-disk sentinel).
    // For integer ZBITPIX this is a simple `arr == ZBLANK` mask;
    // the float branch was rejected at the top.
    let mask_opt = if mask_blank {
        crate::hdu_image::compute_blank_mask_from_value(
            meta.zblank, &out_arr,
        )?
    } else {
        None
    };

    // Apply BSCALE/BZERO (same dispatch the uncompressed path uses).
    let unbound = out_arr.unbind();
    let scaled = if scale {
        let kind = crate::hdu_image::image_scaling_kind(
            zbitpix, meta.bscale, meta.bzero,
        );
        crate::hdu_image::apply_image_scaling(
            py, unbound, zbitpix, kind, meta.bscale, meta.bzero,
        )?
    } else {
        unbound
    };

    if mask_blank {
        crate::hdu_image::wrap_in_masked_array(py, scaled, mask_opt)
    } else {
        Ok(scaled)
    }
}

// What we got back from the heap for one tile.  Three cases:
//
//   - PrimaryCompressed: the primary COMPRESSED_DATA column had
//     bytes.  For a quantized-float HDU these are quantized i32
//     values; for an integer HDU they're the matching int dtype.
//     The caller decodes, then (for float HDUs) runs dequant.
//
//   - FallbackCompressed: one of the fallback columns had bytes
//     (primary was empty).  cfitsio's lossless-fallback convention
//     is: bytes are the *original physical* values, not quantized.
//     For a float HDU that means raw f4/f8 bytes — the dequant
//     step must be SKIPPED.  For an integer HDU it's just raw int
//     bytes (same shape as PrimaryCompressed would have been).
//
//   - Uncompressed: UNCOMPRESSED_DATA column had bytes; same
//     "raw physical" convention as FallbackCompressed, just
//     without the compression layer.
//
// The `algorithm` on the Compressed variants tells the decoder
// which decompressor to run; the *primary-vs-fallback* split is
// what determines whether dequant runs afterwards.
enum TilePayload {
    PrimaryCompressed { bytes: Vec<u8>, algorithm: CompressionAlgorithm },
    FallbackCompressed { bytes: Vec<u8>, algorithm: CompressionAlgorithm },
    Uncompressed { bytes: Vec<u8> },
}

// Upper bound on image dimensionality for the hot-loop stack
// arrays in `tile_origin_and_shape`.  The FITS spec allows ZNAXIS
// up to 999, but real images are 1-4 dims (sometimes 5 for
// hyperspectral cubes); 8 is a comfortable margin.  The function
// asserts d <= MAX_NAXIS so a malformed input fails fast rather
// than silently truncating.
pub(crate) const MAX_NAXIS: usize = 8;

// Given a tile index (0..n_tiles, FITS-row-major), the image
// shape (numpy order), and the nominal tile shape (numpy order),
// fill the caller-provided fixed-size buffers with the tile's
// numpy-order origin and its actual shape (which may be smaller
// than nominal at the image edges).  Returns the active length
// `d`; callers slice the buffers with `[..d]` for the live data.
//
// Stack-allocates everything — no Vec, no allocator round-trip
// per call.  This function runs millions of times on chunked
// reads of large compressed images, where the three `vec![0u64; d]`
// of the previous Vec-returning version dominated 12% of slice
// time on small-chunk workloads.
pub(crate) fn tile_origin_and_shape(
    tile_idx: u64,
    image_shape_numpy: &[u64],
    nominal_tile_shape_numpy: &[u64],
    origin_out: &mut [u64; MAX_NAXIS],
    shape_out: &mut [u64; MAX_NAXIS],
) -> usize {
    let d = image_shape_numpy.len();
    assert!(
        d <= MAX_NAXIS,
        "tile_origin_and_shape: image NAXIS={} exceeds MAX_NAXIS={}",
        d, MAX_NAXIS,
    );
    let mut idx = tile_idx;
    let mut tile_coord = [0u64; MAX_NAXIS];
    // Unfold from numpy-last (= FITS-fastest = varies fastest in
    // the BINTABLE row ordering) to numpy-first.
    for axis_numpy in (0..d).rev() {
        let n_along = (image_shape_numpy[axis_numpy]
            + nominal_tile_shape_numpy[axis_numpy] - 1)
            / nominal_tile_shape_numpy[axis_numpy];
        tile_coord[axis_numpy] = idx % n_along;
        idx /= n_along;
    }
    for axis in 0..d {
        origin_out[axis] = tile_coord[axis]
            * nominal_tile_shape_numpy[axis];
        let end = (origin_out[axis] + nominal_tile_shape_numpy[axis])
            .min(image_shape_numpy[axis]);
        shape_out[axis] = end - origin_out[axis];
    }
    d
}

// Look up a tile in the cache or, on miss, read payload + decode
// + insert.  Returns the tile's bytes in target (stored) dtype,
// numpy C-order, ready to be wrapped in a numpy ndarray.  The
// returned Arc is shared with the cache; both points keep the
// allocation alive until the consumer drops the reference.
//
// File I/O (descriptor read + heap read) is done under the file
// lock; the lock is released before decode.  The cache lock is
// taken twice (once for `get`, once for `put`) and held only
// across in-memory ops.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_or_decode_tile(
    _py: Python<'_>,
    cache: &TileCache,
    file_handle: &FileHandle,
    tainted: &TaintFlag,
    tile_idx: u64,
    data_offset: u64,
    naxis1: u64,
    theap: u64,
    cols: &ZimageDataColumns,
    algorithm: CompressionAlgorithm,
    actual_shape: &[u64],
    bytepix: u32,
    blocksize: u32,
    // ZBITPIX as seen by the decoder.  For integer ZBITPIX this is
    // just the image's ZBITPIX; for quantized float (-32/-64) the
    // caller passes 32 so the decoder produces i32 bytes that the
    // dequant step then consumes.
    stored_zbitpix: i32,
    // Output ZBITPIX (image-side dtype).  Used by the dequant step
    // to decide f32 vs f64 output.  Equals stored_zbitpix when
    // there's no quantization in play.
    output_zbitpix: i32,
    quant: Option<&ZimageQuantContext>,
    // HCOMPRESS_1 SMOOTH flag (ignored by every other algorithm).
    smooth: bool,
) -> PyResult<Arc<Vec<u8>>> {
    if let Some(arc) = cache.get(&tile_idx) {
        return Ok(arc);
    }
    check_not_tainted(tainted)?;

    // Read payload (descriptor + heap) under the file lock.  The
    // returned variant tells us how to interpret the bytes.  When
    // quantization is in play, also read this tile's ZSCALE/ZZERO
    // from their fixed-width columns under the same lock acquire.
    let (payload, scale_zero) = fetch_tile_payload_and_quant(
        file_handle, tile_idx, data_offset, naxis1, theap, cols,
        algorithm, quant,
    )?;

    // Decode (no file lock held here).  Two distinct decode paths:
    //
    //   - Primary path: bytes are in the *stored* int representation
    //     (i32 for quantized float, native int for int HDU).  Decode
    //     with stored_zbitpix / bytepix.  For float output, the
    //     dequant step then converts to f4/f8.
    //
    //   - Fallback path (FallbackCompressed / Uncompressed): bytes
    //     are in the *physical* representation already.  For a float
    //     HDU that's raw f4/f8 in FITS big-endian; we need to decode
    //     at the float bytepix (not 4) and skip the dequant step.
    //     For an int HDU it's the same int bytes the primary would
    //     have produced.
    let tile_n_pixels: usize = actual_shape.iter()
        .product::<u64>() as usize;
    let is_float_output = output_zbitpix < 0;
    let float_bytepix: u32 = match output_zbitpix {
        -32 => 4,
        -64 => 8,
        _ => 0, // unused for integer-output HDUs
    };
    // dequant runs iff the primary column produced bytes AND
    // quantization is in play.  Three cases drop into the
    // "physical bytes" handling below (no dequant, decode at the
    // float bytepix for float HDUs):
    //   - primary column + unquantized float HDU (ZQUANTIZ='NONE'
    //     or missing ZSCALE/ZZERO)
    //   - fallback column (lossless GZIP fallback) on any HDU
    //   - uncompressed column on any HDU
    let dequant_applies = matches!(payload, TilePayload::PrimaryCompressed { .. })
        && quant.is_some();

    // Decoder bytepix / stored_zbitpix depending on whether
    // dequant will fire.  When dequant applies we decode to i32
    // (the quantized representation); otherwise we decode to the
    // physical dtype (float bytepix for float HDUs, int bytepix
    // for int HDUs).
    let (decode_bp, decode_zb) = if dequant_applies {
        (bytepix, stored_zbitpix)
    } else if is_float_output {
        (float_bytepix, output_zbitpix.abs())
    } else {
        (bytepix, stored_zbitpix)
    };

    let decoded_bytes = match payload {
        TilePayload::PrimaryCompressed { bytes, algorithm }
        | TilePayload::FallbackCompressed { bytes, algorithm } => {
            let params = crate::zimage::AlgorithmParams {
                tile_shape_numpy: actual_shape,
                smooth,
            };
            crate::zimage::decode_tile_to_bytes(
                algorithm, &bytes, tile_n_pixels, decode_bp, blocksize,
                decode_zb, params,
            )?
        }
        TilePayload::Uncompressed { mut bytes } => {
            let expected = tile_n_pixels.checked_mul(decode_bp as usize)
                .ok_or_else(|| PyValueError::new_err(
                    "ZIMAGE: tile pixel count * bytepix overflowed usize"
                ))?;
            if bytes.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "ZIMAGE tile {}: UNCOMPRESSED_DATA payload is {} bytes \
                     but expected {} ({} pixels * {} bytes/pixel)",
                    tile_idx, bytes.len(), expected, tile_n_pixels, decode_bp
                )));
            }
            if decode_bp > 1 && !cfg!(target_endian = "big") {
                crate::common::byteswap_in_place(&mut bytes, decode_bp as usize);
            }
            bytes
        }
    };

    let target_bytes = if dequant_applies {
        let ctx = quant.expect("dequant_applies implies quant.is_some()");
        let (scale, zero) = scale_zero
            .expect("dequant_applies implies scale_zero was read");
        let stored_i32 = crate::zimage::quantize::i32_bytes_to_values(
            &decoded_bytes,
        )?;
        // FITS tile_row is 1-based per cfitsio's convention.
        let tile_row_1based = tile_idx + 1;
        match ctx.output_zbitpix {
            -32 => crate::zimage::quantize::dequantize_to_f32(
                &stored_i32, scale, zero, ctx.method,
                tile_row_1based, ctx.zdither0,
            ),
            -64 => crate::zimage::quantize::dequantize_to_f64(
                &stored_i32, scale, zero, ctx.method,
                tile_row_1based, ctx.zdither0,
            ),
            other => return Err(PyValueError::new_err(format!(
                "ZimageQuantContext.output_zbitpix={} not in {{-32,-64}}",
                other
            ))),
        }
    } else {
        decoded_bytes
    };

    let arc = Arc::new(target_bytes);
    cache.put(tile_idx, Arc::clone(&arc));
    Ok(arc)
}

// Read one tile's heap payload.  Checks the data columns in
// priority order: primary COMPRESSED_DATA → GZIP_COMPRESSED_DATA
// fallback → UNCOMPRESSED_DATA fallback.  Returns whichever has
// non-zero nelements first, tagged so the caller knows which
// decoder to run.  All present columns' descriptors are read for
// the row; reading is cheap (8 or 16 bytes apiece) and keeping a
// single lock-acquire is simpler than retrying.
// Same as fetch_tile_payload but also reads the per-tile
// ZSCALE/ZZERO doubles under the same file lock acquire.  Returns
// `(payload, Some((scale, zero)))` when `quant` is Some, else
// `(payload, None)`.  Pairing the reads avoids a second lock cycle
// in the hot tile loop.
fn fetch_tile_payload_and_quant(
    file_handle: &FileHandle,
    tile_idx: u64,
    data_offset: u64,
    naxis1: u64,
    theap: u64,
    cols: &ZimageDataColumns,
    algorithm: CompressionAlgorithm,
    quant: Option<&ZimageQuantContext>,
) -> PyResult<(TilePayload, Option<(f64, f64)>)> {
    let mut guard = file_handle.lock()
        .map_err(|_| PyIOError::new_err("file lock poisoned"))?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let row_offset = data_offset + tile_idx * naxis1;

    let payload = fetch_tile_payload_inner(
        file, tile_idx, data_offset, theap, cols, algorithm, row_offset,
    )?;
    let scale_zero = match quant {
        Some(ctx) => Some((
            read_big_endian_f64(
                file, row_offset + ctx.zscale_offset_in_row,
            )?,
            read_big_endian_f64(
                file, row_offset + ctx.zzero_offset_in_row,
            )?,
        )),
        None => None,
    };
    Ok((payload, scale_zero))
}

// Original payload-only entry point.  Still called from
// fetch_tile_payload_and_quant above; kept as a free fn so the
// payload dispatch logic lives in one place.
#[allow(dead_code)]
fn fetch_tile_payload(
    file_handle: &FileHandle,
    tile_idx: u64,
    data_offset: u64,
    naxis1: u64,
    theap: u64,
    cols: &ZimageDataColumns,
    algorithm: CompressionAlgorithm,
) -> PyResult<TilePayload> {
    let mut guard = file_handle.lock()
        .map_err(|_| PyIOError::new_err("file lock poisoned"))?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let row_offset = data_offset + tile_idx * naxis1;
    fetch_tile_payload_inner(
        file, tile_idx, data_offset, theap, cols, algorithm, row_offset,
    )
}

fn fetch_tile_payload_inner(
    file: &mut std::fs::File,
    tile_idx: u64,
    data_offset: u64,
    theap: u64,
    cols: &ZimageDataColumns,
    algorithm: CompressionAlgorithm,
    row_offset: u64,
) -> PyResult<TilePayload> {
    // Primary column first.
    let (prim_nelem, prim_off) = read_descriptor(
        file, row_offset + cols.primary.byte_offset_in_row, cols.primary.is_q,
    )?;
    if prim_nelem > 0 {
        let bytes = read_heap_bytes(
            file, data_offset + theap + prim_off,
            prim_nelem.saturating_mul(cols.primary.inner_byte_width),
        )?;
        return Ok(TilePayload::PrimaryCompressed { bytes, algorithm });
    }

    // GZIP fallback — always GZIP_1 decode regardless of the
    // primary algorithm; this is cfitsio's lossless-fallback
    // convention.
    if let Some(gcol) = &cols.gzip_fallback {
        let (nelem, off) = read_descriptor(
            file, row_offset + gcol.byte_offset_in_row, gcol.is_q,
        )?;
        if nelem > 0 {
            let bytes = read_heap_bytes(
                file, data_offset + theap + off,
                nelem.saturating_mul(gcol.inner_byte_width),
            )?;
            return Ok(TilePayload::FallbackCompressed {
                bytes,
                algorithm: CompressionAlgorithm::Gzip1,
            });
        }
    }

    // Uncompressed fallback.
    if let Some(ucol) = &cols.uncompressed_fallback {
        let (nelem, off) = read_descriptor(
            file, row_offset + ucol.byte_offset_in_row, ucol.is_q,
        )?;
        if nelem > 0 {
            let bytes = read_heap_bytes(
                file, data_offset + theap + off,
                nelem.saturating_mul(ucol.inner_byte_width),
            )?;
            return Ok(TilePayload::Uncompressed { bytes });
        }
    }

    Err(PyValueError::new_err(format!(
        "ZIMAGE tile {} has no data in any of COMPRESSED_DATA / \
         GZIP_COMPRESSED_DATA / UNCOMPRESSED_DATA", tile_idx
    )))
}

// Read one big-endian f64 at the given absolute file offset.
// Used for the per-tile ZSCALE / ZZERO (TFORM=`1D`) reads on the
// quantized-float path.
fn read_big_endian_f64(
    file: &mut std::fs::File,
    offset: u64,
) -> PyResult<f64> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(f64::from_be_bytes(buf))
}

// Read one variable-length descriptor at the given absolute file
// offset.  Returns (nelements, heap_offset).  Both `P` (8 bytes,
// two u32s) and `Q` (16 bytes, two u64s) forms are big-endian.
pub(crate) fn read_descriptor(
    file: &mut std::fs::File,
    desc_offset: u64,
    is_q: bool,
) -> PyResult<(u64, u64)> {
    file.seek(SeekFrom::Start(desc_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    if is_q {
        let mut buf = [0u8; 16];
        file.read_exact(&mut buf)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let nelem = u64::from_be_bytes(buf[..8].try_into().unwrap());
        let off = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        Ok((nelem, off))
    } else {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let nelem = u32::from_be_bytes(buf[..4].try_into().unwrap()) as u64;
        let off = u32::from_be_bytes(buf[4..8].try_into().unwrap()) as u64;
        Ok((nelem, off))
    }
}

// Read `n_bytes` of heap payload starting at the absolute file
// offset.  Convenience wrapper for the read_descriptor + heap-read
// pair done in fetch_tile_payload.
fn read_heap_bytes(
    file: &mut std::fs::File,
    heap_byte_offset: u64,
    n_bytes: u64,
) -> PyResult<Vec<u8>> {
    let mut buf = vec![0u8; n_bytes as usize];
    file.seek(SeekFrom::Start(heap_byte_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    file.read_exact(&mut buf)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(buf)
}

// Write a cached/decoded tile's bytes into a region of the output
// ndarray.  Bytes are already in the target dtype in numpy
// C-order; we wrap them as a 1-D numpy view (via np.frombuffer on
// a PyBytes), reshape to the tile's actual shape, and assign to
// the slice of out_arr covering the tile's image-coords range.
fn place_tile_bytes_into_output(
    py: Python<'_>,
    out_arr: &Bound<'_, PyAny>,
    tile_bytes: &[u8],
    target_dtype: &str,
    actual_shape_numpy: &[u64],
    origin_numpy: &[u64],
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let pybytes = PyBytes::new(py, tile_bytes);
    let arr1d = np.call_method1("frombuffer", (pybytes, target_dtype))?;
    let shape_tuple = PyTuple::new(py, actual_shape_numpy)?;
    let reshaped = arr1d.call_method1("reshape", (shape_tuple,))?;

    let slice_objs: Vec<Bound<'_, PySlice>> = origin_numpy
        .iter()
        .zip(actual_shape_numpy.iter())
        .map(|(&o, &s)| PySlice::new(
            py, o as isize, (o + s) as isize, 1,
        ))
        .collect();
    let slice_tuple = PyTuple::new(py, &slice_objs)?;
    out_arr.set_item(slice_tuple, reshaped)?;
    Ok(())
}

// Per-axis overlap between one tile and one slice descriptor.
// Returned indexers are Python objects (PyInt for is_int axes,
// PySlice otherwise) ready to drop into a tuple for numpy
// indexing.  `output_indexer` is `None` for is_int axes because
// those axes don't exist in the output array (they're dropped
// by numpy's standard int-indexing-collapses-axis rule).
pub(crate) struct AxisOverlap<'py> {
    pub(crate) tile_indexer: Bound<'py, PyAny>,
    pub(crate) output_indexer: Option<Bound<'py, PyAny>>,
}

// Pre-computed per-axis tile-coord range of tiles that overlap a
// slice key.  Used by the slice_compressed_image fast path to
// enumerate ONLY overlapping tiles (instead of walking
// 0..n_tiles and rejecting most of them in the body).  Only
// valid when every axis is a step=1 slice with !is_int — caller
// checks first.
//
// `tc_first[ax]..=tc_first[ax]+tc_extent[ax]-1` is the range of
// overlapping tile coords on axis `ax`; `n_along[ax]` is the
// total tile count along that axis (used by `unfold` to convert
// per-axis coords back into a BINTABLE row-major tile_idx).
struct OverlappingTileRange {
    n_along: [u64; MAX_NAXIS],
    tc_first: [u64; MAX_NAXIS],
    tc_extent: [u64; MAX_NAXIS],
    d_img: usize,
    total: u64,
}

impl OverlappingTileRange {
    fn new(
        image_shape: &[u64],
        tile_shape: &[u64],
        slices: &[AxisSlice],
    ) -> Self {
        let d_img = image_shape.len();
        let mut n_along = [0u64; MAX_NAXIS];
        let mut tc_first = [0u64; MAX_NAXIS];
        let mut tc_extent = [0u64; MAX_NAXIS];
        let mut total = 1u64;
        for ax in 0..d_img {
            n_along[ax] = (image_shape[ax] + tile_shape[ax] - 1)
                / tile_shape[ax];
            let s = &slices[ax];
            debug_assert!(!s.is_int && s.step == 1 && s.count > 0);
            tc_first[ax] = s.start / tile_shape[ax];
            let last_image_idx = s.start + s.count - 1;
            let tc_last = (last_image_idx / tile_shape[ax])
                .min(n_along[ax] - 1);
            tc_extent[ax] = tc_last - tc_first[ax] + 1;
            total *= tc_extent[ax];
        }
        OverlappingTileRange { n_along, tc_first, tc_extent, d_img, total }
    }

    fn total(&self) -> u64 {
        self.total
    }

    // For iter_idx in 0..total(), unfold into per-axis tile coords
    // and return the BINTABLE row-major tile_idx.  Last numpy axis
    // varies fastest inside the (tc_first..=tc_last) box so
    // consecutive iter_idx values land on adjacent BINTABLE rows
    // (best disk locality).
    #[inline]
    fn unfold(
        &self,
        iter_idx: u64,
        tile_coord_out: &mut [u64; MAX_NAXIS],
    ) -> u64 {
        let mut idx = iter_idx;
        for ax in (0..self.d_img).rev() {
            tile_coord_out[ax] = self.tc_first[ax]
                + idx % self.tc_extent[ax];
            idx /= self.tc_extent[ax];
        }
        let mut tile_idx: u64 = 0;
        for ax in 0..self.d_img {
            tile_idx = tile_idx * self.n_along[ax] + tile_coord_out[ax];
        }
        tile_idx
    }
}

// Copy a d-dimensional rectangular subregion from one C-contiguous
// byte buffer to another.  Used by the slice_compressed_image
// fast path to land decoded tile bytes directly into the output
// ndarray's buffer, avoiding the PyBytes::new + frombuffer +
// reshape + set_item round-trip.
//
// Both `dst` and `src` are interpreted as C-contiguous N-D arrays
// of `itemsize` bytes per element.  The innermost axis becomes
// one memcpy per outer-axis coordinate; for d == 1 it's a single
// memcpy.  Stack-only — no heap allocations regardless of d.
fn strided_copy_c_contig_to_c_contig(
    dst: &mut [u8],
    dst_shape: &[u64],
    dst_start: &[u64],
    src: &[u8],
    src_shape: &[u64],
    src_start: &[u64],
    copy_shape: &[u64],
    itemsize: usize,
) {
    let d = dst_shape.len();
    debug_assert_eq!(d, src_shape.len());
    debug_assert_eq!(d, copy_shape.len());
    debug_assert_eq!(d, dst_start.len());
    debug_assert_eq!(d, src_start.len());
    debug_assert!(d >= 1 && d <= MAX_NAXIS);

    let last = d - 1;
    let row_bytes = (copy_shape[last] as usize) * itemsize;

    // C-contiguous byte strides: stride[i] = itemsize * prod(shape[i+1..])
    let mut dst_byte_strides = [0u64; MAX_NAXIS];
    let mut src_byte_strides = [0u64; MAX_NAXIS];
    {
        let mut s_dst = itemsize as u64;
        let mut s_src = itemsize as u64;
        for i in (0..d).rev() {
            dst_byte_strides[i] = s_dst;
            src_byte_strides[i] = s_src;
            s_dst *= dst_shape[i];
            s_src *= src_shape[i];
        }
    }

    // Base offsets at the start corner.
    let mut base_dst_off: usize = 0;
    let mut base_src_off: usize = 0;
    for i in 0..d {
        base_dst_off += (dst_start[i] * dst_byte_strides[i]) as usize;
        base_src_off += (src_start[i] * src_byte_strides[i]) as usize;
    }

    if d == 1 {
        dst[base_dst_off..base_dst_off + row_bytes]
            .copy_from_slice(&src[base_src_off..base_src_off + row_bytes]);
        return;
    }

    // Walk all outer (d-1) axes; innermost is one memcpy of row_bytes.
    let outer_count: u64 = copy_shape[..last].iter().product();
    let mut coord = [0u64; MAX_NAXIS];
    for _ in 0..outer_count {
        let mut dst_off = base_dst_off;
        let mut src_off = base_src_off;
        for ax in 0..last {
            dst_off += (coord[ax] * dst_byte_strides[ax]) as usize;
            src_off += (coord[ax] * src_byte_strides[ax]) as usize;
        }
        dst[dst_off..dst_off + row_bytes]
            .copy_from_slice(&src[src_off..src_off + row_bytes]);
        // Increment coord; innermost outer axis (last-1) varies fastest.
        for ax in (0..last).rev() {
            coord[ax] += 1;
            if coord[ax] < copy_shape[ax] {
                break;
            }
            coord[ax] = 0;
        }
    }
}

// For a tile at image-range [tile_origin, tile_origin+tile_size)
// on one axis, decide which of the slice's image indices fall in
// the tile, and build the numpy indexers needed to copy that
// region from tile to output.  Returns None when the tile and
// the slice don't overlap on this axis (skip the tile).
pub(crate) fn axis_overlap<'py>(
    py: Python<'py>,
    tile_origin: u64,
    tile_size: u64,
    axis: &AxisSlice,
) -> PyResult<Option<AxisOverlap<'py>>> {
    let tile_end = tile_origin + tile_size;
    if axis.is_int {
        // is_int: single index, axis.start is the indexed coord.
        if axis.start >= tile_origin && axis.start < tile_end {
            let local: i64 = (axis.start - tile_origin) as i64;
            let pyint = local.into_pyobject(py)?.into_any();
            Ok(Some(AxisOverlap {
                tile_indexer: pyint,
                output_indexer: None,
            }))
        } else {
            Ok(None)
        }
    } else {
        // Slice — image indices are start, start+step, ...,
        // start+(count-1)*step.  Find first and last that land in
        // [tile_origin, tile_end).
        if axis.start >= tile_end {
            return Ok(None);
        }
        let last_image_index = axis.start + (axis.count - 1) * axis.step;
        if last_image_index < tile_origin {
            return Ok(None);
        }
        let k_first = if axis.start >= tile_origin {
            0u64
        } else {
            let diff = tile_origin - axis.start;
            (diff + axis.step - 1) / axis.step
        };
        if k_first >= axis.count {
            return Ok(None);
        }
        let k_last_unclamped = (tile_end - 1 - axis.start) / axis.step;
        let k_last = std::cmp::min(k_last_unclamped, axis.count - 1);
        if k_last < k_first {
            return Ok(None);
        }
        let image_first = axis.start + k_first * axis.step;
        let image_last = axis.start + k_last * axis.step;
        let tile_local_first = (image_first - tile_origin) as isize;
        let tile_local_stop = (image_last - tile_origin + 1) as isize;
        let step = axis.step as isize;
        let tile_slice = PySlice::new(
            py, tile_local_first, tile_local_stop, step,
        );
        let output_slice = PySlice::new(
            py, k_first as isize, (k_last + 1) as isize, 1,
        );
        Ok(Some(AxisOverlap {
            tile_indexer: tile_slice.into_any(),
            output_indexer: Some(output_slice.into_any()),
        }))
    }
}

// Top-level entry point for CompressedImageHDU::__getitem__.
// Parses the slice key, walks every tile in the BINTABLE, checks
// per-axis overlap with the slice, and for overlapping tiles
// copies the slice region from the (cached or freshly-decoded)
// tile into the assembled output ndarray.  Always applies
// BSCALE/BZERO; for all-int multi-axis indexes, the 0-d output
// is unwrapped to a numpy scalar.
pub(crate) fn slice_compressed_image(
    py: Python<'_>,
    meta: &CompressedImageMeta,
    data_offset: u64,
    file_handle: &FileHandle,
    tainted: &TaintFlag,
    cache: &TileCache,
    key: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    check_not_tainted(tainted)?;
    let zbitpix = meta.zbitpix;
    let image_shape = meta.image_shape.as_slice();
    let tile_shape = meta.tile_shape.as_slice();
    let algorithm = meta.algorithm;
    let blocksize = meta.blocksize;
    let bytepix = meta.bytepix;
    let smooth = meta.smooth;
    let naxis1 = meta.naxis1;
    let theap = meta.theap;
    let cols = &meta.cols;
    let quant = meta.quant.as_ref();
    let n_tiles = meta.n_tiles;
    // Same stored vs output ZBITPIX split as the whole-image read
    // path; see read_compressed_image_data for the rationale.
    let stored_zbitpix: i32 = if zbitpix < 0 { 32 } else { zbitpix };

    // Parse the slice key.  Fancy lists/arrays raise here via
    // parse_axis_indexer's "unsupported index type" error, matching
    // the ImageHDU surface.
    let slices = normalize_slice_key(key, &image_shape)?;
    let all_int = slices.iter().all(|s| s.is_int);

    // Output shape: drop is_int axes.  All-int → empty shape (0-d
    // ndarray); collapsed to a numpy scalar at the end.
    let output_shape: Vec<u64> = slices.iter()
        .filter(|s| !s.is_int)
        .map(|s| s.count)
        .collect();

    let dtype_str = zbitpix_to_native_dtype(zbitpix)?;
    let np = py.import("numpy")?;
    let shape_tuple = PyTuple::new(py, &output_shape)?;
    let out_arr = np.call_method1("empty", (shape_tuple, dtype_str))?;

    // If any axis has zero-width slice, output is empty — skip the
    // tile walk.  (np.empty with a 0 in the shape returns an empty
    // array of the right shape; that's the desired result.)
    let any_empty = slices.iter().any(|s| s.count == 0);
    // Fast path: every axis is a contiguous slice with step=1 and
    // none are int-collapse.  In this regime we can land tile bytes
    // directly into the output ndarray's buffer with a strided
    // memcpy, skipping the per-tile PyBytes::new + frombuffer +
    // reshape + set_item round-trip (which dominated the slice path
    // on small-chunk workloads — see "Performance" in CLAUDE.md).
    // Stepped slicing and int-collapse axes take the slow path
    // below which still uses numpy's set_item for scatter semantics.
    let fast_path = !any_empty
        && slices.iter().all(|s| !s.is_int && s.step == 1);
    if fast_path {
        let out_itemsize = (zbitpix.unsigned_abs() / 8) as usize;
        let mut out_buf = RawBuffer::acquire_writable(&out_arr)?;
        let dst_bytes = out_buf.as_mut_slice();
        let d_img = image_shape.len();
        let range = OverlappingTileRange::new(
            &image_shape, &tile_shape, &slices,
        );
        let mut tile_coord = [0u64; MAX_NAXIS];
        let mut origin_buf = [0u64; MAX_NAXIS];
        let mut shape_buf = [0u64; MAX_NAXIS];
        let mut tile_start_buf = [0u64; MAX_NAXIS];
        let mut out_start_buf = [0u64; MAX_NAXIS];
        let mut copy_shape_buf = [0u64; MAX_NAXIS];
        for iter_idx in 0..range.total() {
            let tile_idx = range.unfold(iter_idx, &mut tile_coord);
            // One pass: origin + actual_shape + per-axis overlap
            // (tile-local + output-local start + copy count).
            // Overlap is guaranteed non-empty for every axis by
            // the tc_first/tc_extent pre-computation, so the
            // saturating arithmetic always produces a positive
            // copy count.
            for ax in 0..d_img {
                origin_buf[ax] = tile_coord[ax] * tile_shape[ax];
                let tile_end = (origin_buf[ax] + tile_shape[ax])
                    .min(image_shape[ax]);
                shape_buf[ax] = tile_end - origin_buf[ax];
                let s = &slices[ax];
                let slice_end = s.start + s.count;
                let overlap_start = s.start.max(origin_buf[ax]);
                let overlap_end = slice_end.min(tile_end);
                tile_start_buf[ax] = overlap_start - origin_buf[ax];
                out_start_buf[ax] = overlap_start - s.start;
                copy_shape_buf[ax] = overlap_end - overlap_start;
            }
            let actual_shape = &shape_buf[..d_img];
            let tile_bytes = get_or_decode_tile(
                py, cache, file_handle, tainted, tile_idx, data_offset,
                naxis1, theap, cols, algorithm, actual_shape,
                bytepix, blocksize, stored_zbitpix,
                zbitpix, quant, smooth,
            )?;
            strided_copy_c_contig_to_c_contig(
                dst_bytes, &output_shape, &out_start_buf[..d_img],
                &tile_bytes, actual_shape, &tile_start_buf[..d_img],
                &copy_shape_buf[..d_img], out_itemsize,
            );
        }
    } else if !any_empty {
        let mut origin_buf = [0u64; MAX_NAXIS];
        let mut shape_buf = [0u64; MAX_NAXIS];
        for tile_idx in 0..n_tiles {
            let d = tile_origin_and_shape(
                tile_idx, &image_shape, &tile_shape,
                &mut origin_buf, &mut shape_buf,
            );
            let origin = &origin_buf[..d];
            let actual_shape = &shape_buf[..d];
            // Per-axis overlap check.  If any axis returns None,
            // the tile doesn't intersect the slice — skip.
            let mut tile_indexers: Vec<Bound<PyAny>> =
                Vec::with_capacity(slices.len());
            let mut output_indexers: Vec<Bound<PyAny>> = Vec::new();
            let mut overlapping = true;
            for (axis_idx, axis) in slices.iter().enumerate() {
                match axis_overlap(
                    py, origin[axis_idx], actual_shape[axis_idx], axis,
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

            // Fetch tile bytes (cache or decode), wrap as ndarray.
            let tile_bytes = get_or_decode_tile(
                py, cache, file_handle, tainted, tile_idx, data_offset,
                naxis1, theap, cols, algorithm, actual_shape,
                bytepix, blocksize, stored_zbitpix,
                zbitpix, quant, smooth,
            )?;
            let pybytes = PyBytes::new(py, &tile_bytes);
            let arr1d = np.call_method1(
                "frombuffer", (pybytes, dtype_str),
            )?;
            let tile_shape_tuple = PyTuple::new(py, actual_shape)?;
            let tile_arr = arr1d.call_method1(
                "reshape", (tile_shape_tuple,),
            )?;

            let tile_idx_tuple = PyTuple::new(py, &tile_indexers)?;
            let sub = tile_arr.get_item(tile_idx_tuple)?;
            if all_int {
                // Output is 0-d — assign with empty-tuple key.
                out_arr.set_item(PyTuple::empty(py), sub)?;
            } else {
                let out_idx_tuple = PyTuple::new(py, &output_indexers)?;
                out_arr.set_item(out_idx_tuple, sub)?;
            }
        }
    }

    // Always scale on __getitem__ (matches the table-side
    // convention; user can read(scale=False) for the whole image).
    let unbound = out_arr.unbind();
    let kind = crate::hdu_image::image_scaling_kind(
        zbitpix, meta.bscale, meta.bzero,
    );
    let scaled = crate::hdu_image::apply_image_scaling(
        py, unbound, zbitpix, kind, meta.bscale, meta.bzero,
    )?;

    if all_int {
        let bound = scaled.bind(py);
        Ok(bound.get_item(PyTuple::empty(py))?.unbind())
    } else {
        Ok(scaled)
    }
}

// ====== Compressed write (Phases 7 + 8) ======
//
// Layout: `write_compressed_image_data` is the top-level
// dispatcher.  It parses the header, loops over tiles calling
// either `encode_tile_int` or `encode_tile_float`, then grows the
// file and writes the descriptor table + heap.  The per-tile
// helpers return a `TileRow` (descriptor + per-tile ZSCALE/ZZERO
// + optional GZIP fallback descriptor) and mutate the heap
// buffers in place.

