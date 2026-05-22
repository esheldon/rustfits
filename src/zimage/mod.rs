// Tile-decoder dispatch.  Owns the CompressionAlgorithm enum
// (parsed from ZCMPTYPE) and routes a tile's compressed bytes to
// the right per-algorithm decoder.  Phase 4 wires RICE_1, GZIP_1,
// and GZIP_2; HCOMPRESS_1 / PLIO_1 are Phase 6+ and still raise.

pub(crate) mod compression_config;
pub(crate) mod gzip;
pub(crate) mod hcompress;
pub(crate) mod plio;
pub(crate) mod quantize;
pub(crate) mod rice;

use pyo3::prelude::*;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompressionAlgorithm {
    Rice1,
    Gzip1,
    Gzip2,
    Hcompress1,
    Plio1,
}

// Map ZCMPTYPE to a CompressionAlgorithm.  Accepts the FITS Tile
// Compression Convention names plus the older cfitsio synonyms
// (RICE_ONE, GZIP, HCOMPRESS) that some encoders still emit —
// fitsio in particular writes RICE_ONE for some quantization
// configurations.
pub(crate) fn parse_algorithm(zcmptype: &str) -> PyResult<CompressionAlgorithm> {
    match zcmptype.trim() {
        "RICE_1" | "RICE_ONE" => Ok(CompressionAlgorithm::Rice1),
        "GZIP_1" | "GZIP" => Ok(CompressionAlgorithm::Gzip1),
        "GZIP_2" => Ok(CompressionAlgorithm::Gzip2),
        "HCOMPRESS_1" | "HCOMPRESS" => Ok(CompressionAlgorithm::Hcompress1),
        "PLIO_1" => Ok(CompressionAlgorithm::Plio1),
        other => Err(PyValueError::new_err(format!(
            "unknown ZCMPTYPE '{}'", other
        ))),
    }
}

// Algorithm-specific parameters carried alongside the generic
// (compressed, n_pixels, bytepix, blocksize, zbitpix) signature.
// Each algorithm consumes only the fields it cares about — RICE
// and GZIP ignore everything in here; HCOMPRESS uses both.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AlgorithmParams<'a> {
    // Tile shape in numpy axis order (slowest first).  HCOMPRESS_1
    // requires 2D tiles; the decoder reads nx = shape[0], ny =
    // shape[1] where ny is the FITS-fastest axis (NAXIS1).  Other
    // algorithms ignore.
    pub tile_shape_numpy: &'a [u64],
    // Whether to apply HCOMPRESS_1's smoothing pass during inverse
    // H-transform.  Ignored by every other algorithm.
    pub smooth: bool,
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
// GZIP stays in u8 throughout, HCOMPRESS via Vec<i32> or Vec<i64>.
//
// `blocksize` is RICE-specific and ignored by GZIP; same for
// `zbitpix` — RICE needs it to know the target dtype to cast its
// i64 stream down to, GZIP infers everything from `bytepix`,
// HCOMPRESS needs zbitpix to pick the i32 vs i64 path.
pub(crate) fn decode_tile_to_bytes(
    algorithm: CompressionAlgorithm,
    compressed: &[u8],
    n_pixels: usize,
    bytepix: u32,
    blocksize: u32,
    zbitpix: i32,
    params: AlgorithmParams<'_>,
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
        CompressionAlgorithm::Hcompress1 => {
            if params.tile_shape_numpy.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "HCOMPRESS_1 only supports 2-D tiles; got tile shape \
                     of {} dimensions", params.tile_shape_numpy.len()
                )));
            }
            let nx = params.tile_shape_numpy[0] as usize;
            let ny = params.tile_shape_numpy[1] as usize;
            hcompress::decode_hcompress(
                compressed, nx, ny, bytepix, zbitpix, params.smooth,
            )
        }
        CompressionAlgorithm::Plio1 => {
            plio::decode_plio(compressed, n_pixels, bytepix, zbitpix)
        }
    }
}

// Encode one tile from its native-endian pixel bytes to the
// algorithm's on-disk compressed bytes.  Mirror of
// `decode_tile_to_bytes` for the write path.  The caller is
// responsible for byteswapping native → FITS big-endian before
// passing in the bytes (matches the decode side, which produces
// native-endian output).
//
// `bytepix` is needed by GZIP_2 to drive its byte-shuffle
// preprocessor; GZIP_1 ignores it (deflate operates byte-wise).
// Algorithms that need additional per-tile parameters (HCOMPRESS,
// RICE) will need their own extensions when wired in follow-up
// sub-phases.
pub(crate) fn encode_tile_from_bytes(
    algorithm: CompressionAlgorithm,
    pixel_bytes_be: &[u8],
    bytepix: u32,
) -> PyResult<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::Gzip1 => gzip::encode_gzip1(pixel_bytes_be),
        CompressionAlgorithm::Gzip2 => {
            gzip::encode_gzip2(pixel_bytes_be, bytepix)
        }
        CompressionAlgorithm::Rice1 => Err(PyNotImplementedError::new_err(
            "RICE_1 encoding is not yet implemented \
             (planned: a follow-up sub-phase under Phase 7)"
        )),
        CompressionAlgorithm::Hcompress1 => Err(PyNotImplementedError::new_err(
            "HCOMPRESS_1 encoding is not yet implemented \
             (planned: a follow-up sub-phase under Phase 7)"
        )),
        CompressionAlgorithm::Plio1 => Err(PyNotImplementedError::new_err(
            "PLIO_1 encoding is not yet implemented \
             (planned: a follow-up sub-phase under Phase 7)"
        )),
    }
}
