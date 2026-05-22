// Quantized-float dequantization for the FITS Tile Compression
// Convention.  When ZBITPIX is negative (-32 / -64) the on-disk
// payload is a stream of i32 *quantized* integers; to recover the
// physical float values we apply per-tile ZSCALE and ZZERO from
// adjacent BINTABLE columns plus, for the SUBTRACTIVE_DITHER_*
// methods, an offset drawn from a deterministic random table
// (Park-Miller LCG, seed 1, multiplier 16807, modulus 2^31 - 1).
//
// Algorithm reference: cfitsio's `quantize.c` (functions
// `fits_init_randoms`, `unquantize_i4r4`, `unquantize_i4r8`).
// The spec is Pence et al. 2010 (Tile Compression Convention,
// section 5) plus the FITS WG agreement on the PRNG.

use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// Length of the random offset table.  Fixed by the FITS spec —
// changing it would break round-trip compatibility with cfitsio-
// produced files.
const N_RANDOM: usize = 10000;

// Value cfitsio reserves to mean "this pixel was NaN" under
// SUBTRACTIVE_DITHER_2.  Picked as INT32_MIN+1 so the absolute
// value still fits in i32.
const NULL_VALUE_I32: i32 = -2147483647;

// ZQUANTIZ values that imply some form of quantization was
// applied.  Absence of the keyword defaults to NoDither when
// ZSCALE/ZZERO columns are present (cfitsio's convention); when
// ZQUANTIZ='NONE' (or the columns are absent) no quantization
// happened at all — the on-disk payload is raw float bytes, not
// quantized integers, and the caller takes a different code path
// that skips this module entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DitherMethod {
    NoDither,
    SubtractiveDither1,
    SubtractiveDither2,
}

// Parse ZQUANTIZ.  Returns `Some(method)` when quantization was
// applied, `None` when it was not.
//
// Per the FITS Tile Compression Convention (Pence et al. 2010
// plus WG revisions), ZQUANTIZ takes exactly three values:
// `NO_DITHER`, `SUBTRACTIVE_DITHER_1`, `SUBTRACTIVE_DITHER_2`.
// When no quantization is applied the Convention says the
// keyword should simply be omitted.
//
// `NONE` is NOT in the spec but is what astropy's CompImageHDU
// writes for unquantized float-compressed HDUs.  cfitsio reads
// it tolerantly.  We accept it for real-world compatibility —
// astropy-written files in the wild use it.
//
// Convention-correct "no quantization" is also signalled by
// missing ZSCALE/ZZERO columns (regardless of ZQUANTIZ); the
// caller checks that separately.  Absent ZQUANTIZ + present
// ZSCALE/ZZERO defaults to NO_DITHER per cfitsio's convention.
pub(crate) fn parse_dither_method(
    zquantiz: Option<&str>,
) -> PyResult<Option<DitherMethod>> {
    match zquantiz.map(|s| s.trim()) {
        Some("NONE") => Ok(None),
        None | Some("NO_DITHER") => Ok(Some(DitherMethod::NoDither)),
        Some("SUBTRACTIVE_DITHER_1") => Ok(Some(DitherMethod::SubtractiveDither1)),
        Some("SUBTRACTIVE_DITHER_2") => Ok(Some(DitherMethod::SubtractiveDither2)),
        Some(other) => Err(PyValueError::new_err(format!(
            "unsupported ZQUANTIZ '{}'", other
        ))),
    }
}

// Build the 10000-element random-offset table used by both
// SUBTRACTIVE_DITHER_* methods.  Park-Miller "minimal standard"
// LCG: `x_{n+1} = (16807 * x_n) mod (2^31 - 1)`, seeded with 1.
// Each entry is `x_n / m` cast to f32, which lands in [0, 1).
// Initialised lazily via OnceLock — every call after the first
// is a single atomic load.
fn random_table() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        const A: f64 = 16807.0;
        const M: f64 = 2147483647.0;
        let mut seed: f64 = 1.0;
        let mut out = Vec::with_capacity(N_RANDOM);
        for _ in 0..N_RANDOM {
            let temp = A * seed;
            // `temp - M * floor(temp / M)` is the standard Lehmer
            // remainder.  Cast through i64 to match cfitsio's
            // `(int) (temp / m)` (which is truncation toward
            // zero — equivalent to floor here because both are
            // positive).
            seed = temp - M * ((temp / M) as i64 as f64);
            out.push((seed / M) as f32);
        }
        out
    })
}

// Iterator over the dither offsets that one tile's pixels see.
// Tracks cfitsio's exact `iseed` / `nextrand` advancement so a
// rustfits-decoded tile matches cfitsio byte-for-byte.
//
// Per cfitsio:
//   iseed    = (tile_row_index_1based - 1 + zdither0 - 1) % N_RANDOM
//   nextrand = floor(table[iseed] * 500)
//   for each pixel:
//     offset = table[nextrand]
//     nextrand += 1
//     if nextrand == N_RANDOM: iseed = (iseed + 1) % N_RANDOM;
//                              nextrand = floor(table[iseed] * 500)
//
// The "* 500" jump on rollover prevents adjacent tiles from sharing
// the same random sequence — important for the dither's whitening
// property to hold across the image.
struct DitherStream {
    table: &'static [f32],
    iseed: usize,
    nextrand: usize,
}

impl DitherStream {
    fn new(tile_row_1based: u64, zdither0: i64) -> Self {
        let table = random_table();
        // The spec is `(tile_row - 1 + zdither0 - 1) % N_RANDOM` in
        // 1-based row indexing.  Promote everything to i64 to avoid
        // a u64 - 1 underflow when `tile_row_1based == 0` (shouldn't
        // happen in practice but the math is cleaner this way).
        let raw: i64 = (tile_row_1based as i64) - 1 + zdither0 - 1;
        let iseed = raw.rem_euclid(N_RANDOM as i64) as usize;
        let nextrand = (table[iseed] * 500.0) as usize;
        DitherStream { table, iseed, nextrand }
    }

    fn next_offset(&mut self) -> f32 {
        let v = self.table[self.nextrand];
        self.nextrand += 1;
        if self.nextrand == N_RANDOM {
            self.iseed = (self.iseed + 1) % N_RANDOM;
            self.nextrand = (self.table[self.iseed] * 500.0) as usize;
        }
        v
    }
}

// Dequantize a tile of i32 quantized values to f32 (BITPIX=-32),
// returning bytes in native byte order.  Per cfitsio, the
// dither-1/2 formula is `(stored - offset + 0.5) * scale + zero`;
// NO_DITHER drops the offset entirely.  For SUBTRACTIVE_DITHER_2,
// stored values equal to NULL_VALUE_I32 are replaced by NaN
// (preserves NaN through round-trip).
pub(crate) fn dequantize_to_f32(
    stored: &[i32],
    scale: f64,
    zero: f64,
    method: DitherMethod,
    tile_row_1based: u64,
    zdither0: i64,
) -> Vec<u8> {
    let n = stored.len();
    let mut out = Vec::with_capacity(n * 4);
    match method {
        DitherMethod::NoDither => {
            for &q in stored {
                let v = (q as f64) * scale + zero;
                out.extend_from_slice(&(v as f32).to_ne_bytes());
            }
        }
        DitherMethod::SubtractiveDither1 => {
            let mut dither = DitherStream::new(tile_row_1based, zdither0);
            for &q in stored {
                let offset = dither.next_offset();
                let v = ((q as f64) - (offset as f64) + 0.5) * scale + zero;
                out.extend_from_slice(&(v as f32).to_ne_bytes());
            }
        }
        DitherMethod::SubtractiveDither2 => {
            let mut dither = DitherStream::new(tile_row_1based, zdither0);
            for &q in stored {
                let offset = dither.next_offset();
                let v: f32 = if q == NULL_VALUE_I32 {
                    f32::NAN
                } else {
                    let phys = ((q as f64) - (offset as f64) + 0.5) * scale + zero;
                    phys as f32
                };
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    out
}

// Same as above but for ZBITPIX=-64.  Output dtype is f64; the
// dequantization arithmetic stays in f64 throughout (no precision
// loss in the cast).
pub(crate) fn dequantize_to_f64(
    stored: &[i32],
    scale: f64,
    zero: f64,
    method: DitherMethod,
    tile_row_1based: u64,
    zdither0: i64,
) -> Vec<u8> {
    let n = stored.len();
    let mut out = Vec::with_capacity(n * 8);
    match method {
        DitherMethod::NoDither => {
            for &q in stored {
                let v = (q as f64) * scale + zero;
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
        DitherMethod::SubtractiveDither1 => {
            let mut dither = DitherStream::new(tile_row_1based, zdither0);
            for &q in stored {
                let offset = dither.next_offset();
                let v = ((q as f64) - (offset as f64) + 0.5) * scale + zero;
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
        DitherMethod::SubtractiveDither2 => {
            let mut dither = DitherStream::new(tile_row_1based, zdither0);
            for &q in stored {
                let offset = dither.next_offset();
                let v: f64 = if q == NULL_VALUE_I32 {
                    f64::NAN
                } else {
                    ((q as f64) - (offset as f64) + 0.5) * scale + zero
                };
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    out
}

// Reinterpret a slice of i32-stored bytes as a Vec<i32> for the
// dequantizer to consume.  Bytes are assumed to be in native byte
// order (the decoder already byteswapped); this just walks them in
// 4-byte chunks.
pub(crate) fn i32_bytes_to_values(bytes: &[u8]) -> PyResult<Vec<i32>> {
    if bytes.len() % 4 != 0 {
        return Err(PyValueError::new_err(format!(
            "quantized-int byte count {} not divisible by 4",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(i32::from_ne_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The first few values of the FITS-spec PRNG are well-known
    // (they're embedded in cfitsio's test suite).  These three
    // anchors are what the C reference produces from seed=1.
    #[test]
    fn random_table_first_entries_match_cfitsio() {
        let tbl = random_table();
        // Park-Miller seed=1 → next seed = 16807; / 2147483647 ≈
        // 7.826e-6.  These are the exact values cfitsio sees too;
        // a mismatch would silently corrupt every dithered tile.
        let expected_0 = (16807.0_f64 / 2147483647.0) as f32;
        let expected_1 = ((16807.0_f64 * 16807.0_f64) % 2147483647.0
            / 2147483647.0) as f32;
        assert_eq!(tbl[0], expected_0);
        assert_eq!(tbl[1], expected_1);
        assert_eq!(tbl.len(), N_RANDOM);
    }

    #[test]
    fn no_dither_is_linear() {
        let stored = vec![100, 200, 300];
        let bytes = dequantize_to_f32(
            &stored, 0.5, 10.0, DitherMethod::NoDither, 1, 0,
        );
        let vals: Vec<f32> = bytes.chunks(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![60.0, 110.0, 160.0]);
    }

    #[test]
    fn parse_dither_method_handles_no_quantization_signals() {
        // 'NONE' (astropy's marker) → None (no quantization).
        assert_eq!(parse_dither_method(Some("NONE")).unwrap(), None);
        // Whitespace tolerant.
        assert_eq!(parse_dither_method(Some(" NONE ")).unwrap(), None);
        // Absent ZQUANTIZ defaults to NO_DITHER (cfitsio's convention
        // when ZSCALE/ZZERO are present; caller separately checks
        // those columns).
        assert_eq!(parse_dither_method(None).unwrap(),
                   Some(DitherMethod::NoDither));
        // Explicit NO_DITHER.
        assert_eq!(parse_dither_method(Some("NO_DITHER")).unwrap(),
                   Some(DitherMethod::NoDither));
        // Unknown value rejects.
        assert!(parse_dither_method(Some("SOMETHING_ELSE")).is_err());
    }

    #[test]
    fn dither2_null_value_becomes_nan() {
        let stored = vec![100, NULL_VALUE_I32, 200];
        let bytes = dequantize_to_f32(
            &stored, 1.0, 0.0, DitherMethod::SubtractiveDither2, 1, 1,
        );
        let vals: Vec<f32> = bytes.chunks(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert!(vals[1].is_nan());
        assert!(!vals[0].is_nan());
        assert!(!vals[2].is_nan());
    }
}
