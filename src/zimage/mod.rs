// Tile-decoder dispatch.  Owns the CompressionAlgorithm enum
// (parsed from ZCMPTYPE) and routes a tile's compressed bytes to
// the right per-algorithm decoder.  Phase 4 wires RICE_1, GZIP_1,
// and GZIP_2; HCOMPRESS_1 / PLIO_1 are Phase 6+ and still raise.

pub(crate) mod gzip;
pub(crate) mod rice;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompressionAlgorithm {
    Rice1,
    Gzip1,
    Gzip2,
    Hcompress1,
    Plio1,
}

pub(crate) fn parse_algorithm(zcmptype: &str) -> PyResult<CompressionAlgorithm> {
    match zcmptype.trim() {
        "RICE_1" => Ok(CompressionAlgorithm::Rice1),
        "GZIP_1" => Ok(CompressionAlgorithm::Gzip1),
        "GZIP_2" => Ok(CompressionAlgorithm::Gzip2),
        "HCOMPRESS_1" => Ok(CompressionAlgorithm::Hcompress1),
        "PLIO_1" => Ok(CompressionAlgorithm::Plio1),
        other => Err(PyValueError::new_err(format!(
            "unknown ZCMPTYPE '{}'", other
        ))),
    }
}

// Decode one tile's compressed bytes directly to target-dtype bytes
// in numpy native byte order, sized `n_pixels * bytepix`.  The
// caller wraps the buffer as a numpy view and copies it into the
// assembled output array.
//
// Each per-algorithm decoder owns the entire bytes-to-bytes path
// (decompress, byteswap, byte-shuffle reverse, dtype cast).  This
// keeps dispatch dumb and lets each algorithm decide what's
// cheapest — RICE goes via Vec<i64> (its natural intermediate),
// GZIP stays in u8 throughout.
//
// `blocksize` is RICE-specific and ignored by GZIP; same for
// `zbitpix` — RICE needs it to know the target dtype to cast its
// i64 stream down to, GZIP infers everything from `bytepix`.
pub(crate) fn decode_tile_to_bytes(
    algorithm: CompressionAlgorithm,
    compressed: &[u8],
    n_pixels: usize,
    bytepix: u32,
    blocksize: u32,
    zbitpix: i32,
) -> PyResult<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::Rice1 => {
            rice::decode_rice(compressed, n_pixels, bytepix, blocksize, zbitpix)
        }
        CompressionAlgorithm::Gzip1 => {
            gzip::decode_gzip1(compressed, n_pixels, bytepix)
        }
        CompressionAlgorithm::Gzip2 => {
            gzip::decode_gzip2(compressed, n_pixels, bytepix)
        }
        CompressionAlgorithm::Hcompress1 => Err(PyValueError::new_err(
            "HCOMPRESS_1 decompression is not yet implemented \
             (planned: Phase 6 of the ZIMAGE roadmap)"
        )),
        CompressionAlgorithm::Plio1 => Err(PyValueError::new_err(
            "PLIO_1 decompression is not yet implemented \
             (planned: Phase 6 of the ZIMAGE roadmap)"
        )),
    }
}
