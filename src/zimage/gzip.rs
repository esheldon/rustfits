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
fn unshuffle(shuffled: &[u8], n_pixels: usize, bytepix: usize) -> Vec<u8> {
    let mut out = vec![0u8; n_pixels * bytepix];
    for j in 0..bytepix {
        for i in 0..n_pixels {
            out[i * bytepix + j] = shuffled[j * n_pixels + i];
        }
    }
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
// Compression level is `Compression::default()` (zlib level 6) —
// same as cfitsio/zlib's default.  Not exposed as a knob yet;
// neither fitsio nor astropy expose it at the high-level API
// either.
pub(crate) fn encode_gzip1(pixel_bytes_be: &[u8]) -> PyResult<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(pixel_bytes_be).map_err(|e| {
        PyValueError::new_err(format!("GZIP_1 encode: write failed: {}", e))
    })?;
    encoder.finish().map_err(|e| {
        PyValueError::new_err(format!("GZIP_1 encode: finish failed: {}", e))
    })
}
