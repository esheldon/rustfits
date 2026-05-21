// Tile-decoder dispatch.  Owns the CompressionAlgorithm enum
// (parsed from ZCMPTYPE) and routes a tile's compressed bytes to
// the right per-algorithm decoder.  Phase 2 wires RICE_1 only;
// other algorithms parse correctly (so the accessor on
// CompressedImageHDU works) but raise NotImplementedError at
// decode time.

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

// Decode a single tile's compressed bytes to pixel values widened
// to i64.  The caller is responsible for casting back to the
// target BITPIX dtype and placing the tile in the output ndarray.
//
// Phase 2 implements RICE_1 only.  Other algorithms return a
// NotImplementedError-equivalent ValueError naming the phase that
// will add them — see CLAUDE.md "Tile-compressed images (ZIMAGE)
// roadmap".
pub(crate) fn decode_tile_to_i64(
    algorithm: CompressionAlgorithm,
    compressed: &[u8],
    n_pixels: usize,
    bytepix: u32,
    blocksize: u32,
) -> PyResult<Vec<i64>> {
    match algorithm {
        CompressionAlgorithm::Rice1 => {
            rice::decode(compressed, n_pixels, bytepix, blocksize)
        }
        CompressionAlgorithm::Gzip1 => Err(PyValueError::new_err(
            "GZIP_1 decompression is not yet implemented \
             (planned: Phase 4 of the ZIMAGE roadmap)"
        )),
        CompressionAlgorithm::Gzip2 => Err(PyValueError::new_err(
            "GZIP_2 decompression is not yet implemented \
             (planned: Phase 4 of the ZIMAGE roadmap)"
        )),
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
