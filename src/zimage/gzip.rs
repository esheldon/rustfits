// GZIP_1 / GZIP_2 tile decompression for the FITS Tile Compression
// Convention.  Both algorithms wrap a tile's pixels in a DEFLATE
// stream with a gzip header (cfitsio's encoder uses zlib's
// deflateInit2 with windowBits=15+16, which selects gzip framing —
// magic 0x1F 0x8B + CRC32 footer).  The two algorithms differ only
// by a byte-shuffle preprocessor that GZIP_2 applies before
// compression:
//
//   GZIP_1: decompress → pixel bytes in FITS big-endian order
//           (NAXIS1 fastest).  Byteswap to native order if the
//           target dtype is multi-byte.
//   GZIP_2: decompress → shuffled bytes (all most-significant
//           bytes first, then next-most-significant, ..., then
//           least-significant) → reverse-shuffle → byteswap to
//           native order.
//
// Reference: cfitsio's `imcomp.c` (the gzip / gzip_2 branches in
// `imcomp_decompress_tile`).

use std::io::{Read, Write};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::common::byteswap_in_place;

// Decompress a gzip-framed payload into exactly `expected_len`
// bytes.  Both GZIP_1 and GZIP_2 share this primitive.  cfitsio
// writes a single gzip member per tile (no concat), so the
// single-member `GzDecoder` is sufficient.
fn gzip_decompress(compressed: &[u8], expected_len: usize) -> PyResult<Vec<u8>> {
    let mut decoder = GzDecoder::new(compressed);
    let mut out: Vec<u8> = Vec::with_capacity(expected_len);
    decoder.read_to_end(&mut out).map_err(|e| PyValueError::new_err(
        format!("GZIP decode: gzip decompression failed: {}", e)
    ))?;
    if out.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "GZIP decode: expected {} decompressed bytes, got {}",
            expected_len, out.len()
        )));
    }
    Ok(out)
}

// Reverse the GZIP_2 byte-shuffle preprocessor:
//   shuffled[j * n_pixels + i]  →  out[i * bytepix + j]
// for i in 0..n_pixels, j in 0..bytepix.  The shuffled layout
// groups bytes by significance (all MSBs first, all 2nd MSBs
// next, ...) which makes the decorrelated runs compress better.
//
// We allocate `out` uninitialized (via Vec::with_capacity +
// spare_capacity_mut + MaybeUninit::write + set_len) rather than
// `vec![0u8; n]`, because every byte is written exactly once by
// the loop below and the `alloc_zeroed`-then-overwrite pattern
// was ~17% of total profile time on big-chunk reads of large
// f8 GZIP_2 images.
fn unshuffle(shuffled: &[u8], n_pixels: usize, bytepix: usize) -> Vec<u8> {
    let n = n_pixels * bytepix;
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let uninit = out.spare_capacity_mut();
    for j in 0..bytepix {
        let src_row = &shuffled[j * n_pixels..(j + 1) * n_pixels];
        for i in 0..n_pixels {
            uninit[i * bytepix + j].write(src_row[i]);
        }
    }
    // SAFETY: the nested loop writes every position in [0, n) of
    // the spare capacity exactly once (j ∈ [0, bytepix), i ∈
    // [0, n_pixels) → covers i*bytepix+j ∈ [0, n) bijectively).
    unsafe { out.set_len(n); }
    out
}

// Decode one GZIP_1-compressed tile.  Returns `n_pixels * bytepix`
// bytes in *numpy native* byte order, ready to copy into the
// caller's tile-bytes buffer.
pub(crate) fn decode_gzip1(
    compressed: &[u8],
    n_pixels: usize,
    bytepix: u32,
) -> PyResult<Vec<u8>> {
    if !matches!(bytepix, 1 | 2 | 4 | 8) {
        return Err(PyValueError::new_err(format!(
            "GZIP_1 decode: unsupported bytepix {} (must be 1/2/4/8)",
            bytepix
        )));
    }
    let expected_len = n_pixels.checked_mul(bytepix as usize)
        .ok_or_else(|| PyValueError::new_err(
            "GZIP_1 decode: n_pixels * bytepix overflowed usize"
        ))?;
    let mut out = gzip_decompress(compressed, expected_len)?;
    if bytepix > 1 && !cfg!(target_endian = "big") {
        byteswap_in_place(&mut out, bytepix as usize);
    }
    Ok(out)
}

// Decode one GZIP_2-compressed tile.  Same as GZIP_1 plus a
// reverse byte-shuffle between decompress and byteswap.  For
// bytepix=1 the shuffle is a no-op (each pixel is one byte), so
// the path collapses to GZIP_1.
pub(crate) fn decode_gzip2(
    compressed: &[u8],
    n_pixels: usize,
    bytepix: u32,
) -> PyResult<Vec<u8>> {
    if !matches!(bytepix, 1 | 2 | 4 | 8) {
        return Err(PyValueError::new_err(format!(
            "GZIP_2 decode: unsupported bytepix {} (must be 1/2/4/8)",
            bytepix
        )));
    }
    let expected_len = n_pixels.checked_mul(bytepix as usize)
        .ok_or_else(|| PyValueError::new_err(
            "GZIP_2 decode: n_pixels * bytepix overflowed usize"
        ))?;
    let shuffled = gzip_decompress(compressed, expected_len)?;
    let mut out = if bytepix == 1 {
        shuffled
    } else {
        unshuffle(&shuffled, n_pixels, bytepix as usize)
    };
    if bytepix > 1 && !cfg!(target_endian = "big") {
        byteswap_in_place(&mut out, bytepix as usize);
    }
    Ok(out)
}

// Encode one tile for GZIP_1: feed the pixel bytes (in FITS
// big-endian order — caller's responsibility) to a fresh
// GzEncoder and collect the gzip-framed output.  Matches cfitsio's
// `deflateInit2(... windowBits=15+16 ...)` which produces gzip
// framing rather than raw deflate or zlib (see the Phase 4 notes
// in CLAUDE.md for the framing gotcha).
//
// Compression level: `None` (the default) uses zlib level 6 — the
// same as cfitsio/zlib/astropy default.  Caller can pass
// `Some(0..=9)` to override (0 = no compression, 1 = fastest /
// least, 9 = slowest / best).  Exposed to Python via
// `Gzip1(level=...)` / `Gzip2(level=...)`.
pub(crate) fn encode_gzip1(
    pixel_bytes_be: &[u8], level: Option<u32>,
) -> PyResult<Vec<u8>> {
    gzip_compress(pixel_bytes_be, "GZIP_1", level)
}

// Apply the GZIP_2 byte-shuffle preprocessor (inverse of
// `unshuffle`).  Groups bytes by significance — all MSBs first,
// then 2nd-MSBs, ..., then LSBs — which produces longer runs of
// similar values for deflate to compress.  Mapping:
//   pixels[i * bytepix + j]  →  out[j * n_pixels + i]
// for i in 0..n_pixels, j in 0..bytepix.
fn shuffle(pixels: &[u8], n_pixels: usize, bytepix: usize) -> Vec<u8> {
    // Same alloc_zeroed-elimination as `unshuffle` above — see
    // that function's docstring for the rationale and the
    // bijection-proves-every-byte-written argument.
    let n = n_pixels * bytepix;
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let uninit = out.spare_capacity_mut();
    for j in 0..bytepix {
        for i in 0..n_pixels {
            uninit[j * n_pixels + i].write(pixels[i * bytepix + j]);
        }
    }
    // SAFETY: (i, j) → j*n_pixels+i is a bijection from
    // [0,n_pixels) × [0,bytepix) onto [0, n), so every byte is
    // written exactly once before set_len.
    unsafe { out.set_len(n); }
    out
}

// Encode one tile for GZIP_2: byte-shuffle the input first, then
// run the GZIP_1 encoder over the shuffled bytes.  For `bytepix=1`
// the shuffle is a no-op and this collapses to `encode_gzip1`.
pub(crate) fn encode_gzip2(
    pixel_bytes_be: &[u8],
    bytepix: u32,
    level: Option<u32>,
) -> PyResult<Vec<u8>> {
    if !matches!(bytepix, 1 | 2 | 4 | 8) {
        return Err(PyValueError::new_err(format!(
            "GZIP_2 encode: unsupported bytepix {} (must be 1/2/4/8)",
            bytepix
        )));
    }
    if !pixel_bytes_be.len().is_multiple_of(bytepix as usize) {
        return Err(PyValueError::new_err(format!(
            "GZIP_2 encode: input length {} not a multiple of bytepix {}",
            pixel_bytes_be.len(), bytepix
        )));
    }
    if bytepix == 1 {
        return gzip_compress(pixel_bytes_be, "GZIP_2", level);
    }
    let n_pixels = pixel_bytes_be.len() / bytepix as usize;
    let shuffled = shuffle(pixel_bytes_be, n_pixels, bytepix as usize);
    gzip_compress(&shuffled, "GZIP_2", level)
}

// Shared gzip-compress primitive used by both GZIP_1 and GZIP_2
// encoders.  Caller passes a label (for error messages) and an
// optional compression level.  None → zlib default (level 6).
fn gzip_compress(
    bytes: &[u8], algo_label: &str, level: Option<u32>,
) -> PyResult<Vec<u8>> {
    let compression = match level {
        None => Compression::default(),
        Some(v) => Compression::new(v),
    };
    let mut encoder = GzEncoder::new(Vec::new(), compression);
    encoder.write_all(bytes).map_err(|e| {
        PyValueError::new_err(format!("{} encode: write failed: {}", algo_label, e))
    })?;
    encoder.finish().map_err(|e| {
        PyValueError::new_err(format!("{} encode: finish failed: {}", algo_label, e))
    })
}
