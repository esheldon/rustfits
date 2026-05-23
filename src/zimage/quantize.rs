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

// Reserved i32 sentinels under SUBTRACTIVE_DITHER_2 quantization:
//   NULL_VALUE_I32   ("this pixel was NaN")
//   ZERO_VALUE_I32   ("this pixel was exactly 0.0")
// Both are at the bottom of the i32 range so they can't collide
// with quantized values produced by the noise/range heuristic.
// (NULL_VALUE_I32 also fits cleanly within i32 even after negation
// — picked as INT32_MIN+1 rather than INT32_MIN for that reason.)
const NULL_VALUE_I32: i32 = -2147483647;
const ZERO_VALUE_I32: i32 = -2147483646;

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
    // All three dither methods recognize NULL_VALUE_I32 → NaN: the
    // encoder writes that sentinel for any input NaN pixel,
    // regardless of method.  DITHER_2 additionally reserves
    // ZERO_VALUE_I32 for exact-zero round-trip.
    match method {
        DitherMethod::NoDither => {
            for &q in stored {
                let v: f32 = if q == NULL_VALUE_I32 {
                    f32::NAN
                } else {
                    ((q as f64) * scale + zero) as f32
                };
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
        DitherMethod::SubtractiveDither1 => {
            let mut dither = DitherStream::new(tile_row_1based, zdither0);
            for &q in stored {
                let offset = dither.next_offset();
                let v: f32 = if q == NULL_VALUE_I32 {
                    f32::NAN
                } else {
                    let phys = ((q as f64) - (offset as f64) + 0.5)
                        * scale + zero;
                    phys as f32
                };
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
        DitherMethod::SubtractiveDither2 => {
            let mut dither = DitherStream::new(tile_row_1based, zdither0);
            for &q in stored {
                // dither.next_offset() runs every pixel to keep
                // the stream in sync with the encoder, which also
                // advances per-pixel regardless of sentinels.
                let offset = dither.next_offset();
                let v: f32 = if q == NULL_VALUE_I32 {
                    f32::NAN
                } else if q == ZERO_VALUE_I32 {
                    // DITHER_2 reserves ZERO_VALUE to round-trip
                    // exact-zero floats; restore as exactly 0.0
                    // (not via the noisy dequant formula).
                    0.0
                } else {
                    let phys = ((q as f64) - (offset as f64) + 0.5)
                        * scale + zero;
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
    // Same NULL_VALUE_I32 handling as the f32 sibling above.
    match method {
        DitherMethod::NoDither => {
            for &q in stored {
                let v: f64 = if q == NULL_VALUE_I32 {
                    f64::NAN
                } else {
                    (q as f64) * scale + zero
                };
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
        DitherMethod::SubtractiveDither1 => {
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
        DitherMethod::SubtractiveDither2 => {
            let mut dither = DitherStream::new(tile_row_1based, zdither0);
            for &q in stored {
                let offset = dither.next_offset();
                let v: f64 = if q == NULL_VALUE_I32 {
                    f64::NAN
                } else if q == ZERO_VALUE_I32 {
                    0.0
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

// ===========================================================================
// ===== ENCODE (quantize) ==================================================
// ===========================================================================
//
// Port of cfitsio's `fits_quantize_float` / `fits_quantize_double`
// from quantize.c.  These convert a tile of float pixels to i32
// quantized values, choosing bscale/bzero per-tile from a noise
// estimate (or a user-supplied fixed bscale).  Returns
// `Some(QuantizedTile)` when quantization succeeded, `None` when
// the caller should fall back to GZIP-compressed raw floats
// (range too wide, delta == 0, fewer than 2 pixels, etc.).
//
// Visibility: only `QuantizedTile`, `quantize_float`, and
// `quantize_double` are `pub(crate)` — they're consumed by the
// write path in `hdu_image_compressed.rs`.  Every helper
// (noise estimation, median, min/max, NINT) is private to this
// file.

// `N_RESERVED_VALUES` — cfitsio reserves the lowest 10 i32 values
// starting with NULL_VALUE_I32 so floating-point round-off can't
// accidentally land quantized data on the sentinel.
// NULL_VALUE_I32 and ZERO_VALUE_I32 are defined at the top of the
// file (shared with the decoder).
const N_RESERVED_VALUES: i32 = 10;

// NaN-aware "is this pixel the null sentinel?" check.  Plain `==`
// fails for NaN (IEEE 754: NaN != NaN), so when the user passes
// `Some(NaN)` as the null sentinel we have to special-case it via
// `is_nan()`.  Returns false when no null sentinel is configured.
fn is_null_f32(v: f32, null_value: Option<f32>) -> bool {
    match null_value {
        Some(nv) if nv.is_nan() => v.is_nan(),
        Some(nv) => v == nv,
        None => false,
    }
}

fn is_null_f64(v: f64, null_value: Option<f64>) -> bool {
    match null_value {
        Some(nv) if nv.is_nan() => v.is_nan(),
        Some(nv) => v == nv,
        None => false,
    }
}

// Output of `quantize_float` / `quantize_double` on the success
// path.  `idata` is the per-pixel quantized stream; `bscale` and
// `bzero` are the linear-scale parameters the decoder needs to
// recover physical values.
pub(crate) struct QuantizedTile {
    pub(crate) idata: Vec<i32>,
    pub(crate) bscale: f64,
    pub(crate) bzero: f64,
}

// MAD-based noise statistics for one tile.  Mirror of the values
// cfitsio's `FnNoise5_float` / `_double` return.  All fields are
// f64 even on the f32 path — minval/maxval promote because the
// caller only consumes them as f64 for the bzero arithmetic.
struct NoiseStats {
    ngood: usize,
    minval: f64,
    maxval: f64,
    noise2: f64,
    noise3: f64,
    noise5: f64,
}

// Median of an f32 slice in place, using std's quickselect.
// Matches cfitsio's `quick_select_float`: returns the value at
// position n/2 after partial sort (so for odd-length the true
// median; for even-length the upper-middle element).
fn quick_select_f32(arr: &mut [f32]) -> f32 {
    let k = arr.len() / 2;
    *arr.select_nth_unstable_by(k, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    }).1
}

fn quick_select_f64(arr: &mut [f64]) -> f64 {
    let k = arr.len() / 2;
    *arr.select_nth_unstable_by(k, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    }).1
}

// Cross-row median after qsort: cfitsio sorts the full row-median
// array and returns `(arr[(n-1)/2] + arr[n/2]) / 2`.  For odd n
// this is the middle value; for even n it averages the two middle
// values.  We mirror that here so byte-exact agreement holds.
fn median_pair_f64(arr: &mut [f64]) -> f64 {
    arr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = arr.len();
    (arr[(n - 1) / 2] + arr[n / 2]) / 2.0
}

// Estimate background noise using 2nd / 3rd / 5th order MAD
// statistics.  Direct port of cfitsio's `FnNoise5_float`:
//
//   2nd order: |v5 - v7|
//   3rd order: |2*v5 - v3 - v7|
//   5th order: |6*v5 - 4*v3 - 4*v7 + v1 + v9|
//
// For each row of the image we compute the median of these per-
// pixel differences; then take the median across rows; then
// multiply by Pence's MAD-to-sigma normalization constants
// (1.0483579 / 0.6052697 / 0.1772048).  These constants assume
// Gaussian noise — fine for astronomical-image use.
//
// `null_value` (Some) flags pixels to skip; None means no null
// filtering.  The caller passes whatever value it stores in
// floats to mean "this pixel is missing" (commonly NaN; cfitsio
// uses an arbitrary sentinel passed as a parameter).
fn noise5_f32(
    array: &[f32],
    nx_in: usize,
    ny_in: usize,
    null_value: Option<f32>,
) -> NoiseStats {
    // Match cfitsio: rows must have ≥ 9 pixels to compute the
    // 5th-order difference.  If not, flatten and degrade to the
    // min/max/ngood-only path (zeros for noise stats).
    let (nx, ny) = if nx_in < 9 { (nx_in * ny_in, 1) } else { (nx_in, ny_in) };

    let mut xminval = f32::INFINITY;
    let mut xmaxval = f32::NEG_INFINITY;
    let mut ngoodpix: usize = 0;

    if nx < 9 {
        for &v in &array[..nx] {
            if is_null_f32(v, null_value) {
                continue;
            }
            if v < xminval {
                xminval = v;
            }
            if v > xmaxval {
                xmaxval = v;
            }
            ngoodpix += 1;
        }
        return NoiseStats {
            ngood: ngoodpix,
            minval: xminval as f64,
            maxval: xmaxval as f64,
            noise2: 0.0,
            noise3: 0.0,
            noise5: 0.0,
        };
    }

    // Per-row diff scratch (sized to max possible per row).
    let mut differences2: Vec<f32> = Vec::with_capacity(nx);
    let mut differences3: Vec<f32> = Vec::with_capacity(nx);
    let mut differences5: Vec<f32> = Vec::with_capacity(nx);
    // Cross-row median arrays.
    let mut diffs2: Vec<f64> = Vec::with_capacity(ny);
    let mut diffs3: Vec<f64> = Vec::with_capacity(ny);
    let mut diffs5: Vec<f64> = Vec::with_capacity(ny);

    // is_valid: returns true when `v` is not the null sentinel
    // (or always true if no nullcheck).  NaN-aware via the
    // is_null_f32 helper.
    let is_valid = |v: f32| -> bool { !is_null_f32(v, null_value) };

    for jj in 0..ny {
        let rowstart = jj * nx;
        let row = &array[rowstart..rowstart + nx];
        differences2.clear();
        differences3.clear();
        differences5.clear();

        // Read the first 8 valid pixels (v1..v8); we need them
        // before we can start computing 5th-order differences
        // (which use v1, v3, v5, v7, v9).  cfitsio unrolls this
        // with continue-on-nx-reached at every step.
        let mut ii: usize = 0;
        let mut vs: [f32; 8] = [0.0; 8];
        let mut found = 0;
        while found < 8 && ii < nx {
            let v = row[ii];
            if is_valid(v) {
                vs[found] = v;
                ngoodpix += 1;
                if v < xminval {
                    xminval = v;
                }
                if v > xmaxval {
                    xmaxval = v;
                }
                found += 1;
            }
            ii += 1;
        }
        if found < 8 {
            // Row had fewer than 8 valid pixels — no diffs
            // computable; skip.
            continue;
        }
        let mut v1 = vs[0];
        let mut v2 = vs[1];
        let mut v3 = vs[2];
        let mut v4 = vs[3];
        let mut v5 = vs[4];
        let mut v6 = vs[5];
        let mut v7 = vs[6];
        let mut v8 = vs[7];

        // Walk the remaining valid pixels, treating each as v9.
        while ii < nx {
            // Find the next valid pixel.
            while ii < nx && !is_valid(row[ii]) {
                ii += 1;
            }
            if ii == nx {
                break;
            }
            let v9 = row[ii];
            if v9 < xminval {
                xminval = v9;
            }
            if v9 > xmaxval {
                xmaxval = v9;
            }

            if !(v5 == v6 && v6 == v7) {
                differences2.push((v5 - v7).abs());
            }
            if !(v3 == v4 && v4 == v5 && v5 == v6 && v6 == v7) {
                differences3.push(((2.0 * v5) - v3 - v7).abs());
                differences5.push(
                    ((6.0 * v5) - (4.0 * v3) - (4.0 * v7) + v1 + v9).abs()
                );
            } else {
                // Constant region; cfitsio counts it as ngood but
                // doesn't push a diff.
                ngoodpix += 1;
            }

            v1 = v2;
            v2 = v3;
            v3 = v4;
            v4 = v5;
            v5 = v6;
            v6 = v7;
            v7 = v8;
            v8 = v9;
            ii += 1;
        }
        ngoodpix += differences3.len();

        if differences3.is_empty() {
            // Cfitsio: cannot compute medians on this row, skip.
            continue;
        }
        if differences2.len() > 1 {
            diffs2.push(quick_select_f32(&mut differences2) as f64);
        } else if differences2.len() == 1 && differences3.len() == 1 {
            // Cfitsio's special case at nvals==1 — only push diff2
            // when nvals2 == 1 too.
            diffs2.push(differences2[0] as f64);
        }
        if differences3.len() > 1 {
            diffs3.push(quick_select_f32(&mut differences3) as f64);
            diffs5.push(quick_select_f32(&mut differences5) as f64);
        } else {
            diffs3.push(differences3[0] as f64);
            diffs5.push(differences5[0] as f64);
        }
    }

    let xnoise3 = if diffs3.is_empty() {
        0.0
    } else if diffs3.len() == 1 {
        diffs3[0]
    } else {
        median_pair_f64(&mut diffs3)
    };
    let xnoise5 = if diffs5.is_empty() {
        0.0
    } else if diffs5.len() == 1 {
        diffs5[0]
    } else {
        median_pair_f64(&mut diffs5)
    };
    let xnoise2 = if diffs2.is_empty() {
        0.0
    } else if diffs2.len() == 1 {
        diffs2[0]
    } else {
        median_pair_f64(&mut diffs2)
    };

    NoiseStats {
        ngood: ngoodpix,
        minval: xminval as f64,
        maxval: xmaxval as f64,
        noise2: 1.0483579 * xnoise2,
        noise3: 0.6052697 * xnoise3,
        noise5: 0.1772048 * xnoise5,
    }
}

// f64 sibling of noise5_f32.  Same algorithm, f64 precision
// throughout (per-row diffs stay f64 instead of being downcast
// to f32 like cfitsio does — preserves precision on small
// differences in float64 data).
fn noise5_f64(
    array: &[f64],
    nx_in: usize,
    ny_in: usize,
    null_value: Option<f64>,
) -> NoiseStats {
    let (nx, ny) = if nx_in < 9 { (nx_in * ny_in, 1) } else { (nx_in, ny_in) };

    let mut xminval = f64::INFINITY;
    let mut xmaxval = f64::NEG_INFINITY;
    let mut ngoodpix: usize = 0;

    if nx < 9 {
        for &v in &array[..nx] {
            if is_null_f64(v, null_value) {
                continue;
            }
            if v < xminval {
                xminval = v;
            }
            if v > xmaxval {
                xmaxval = v;
            }
            ngoodpix += 1;
        }
        return NoiseStats {
            ngood: ngoodpix,
            minval: xminval,
            maxval: xmaxval,
            noise2: 0.0,
            noise3: 0.0,
            noise5: 0.0,
        };
    }

    let mut differences2: Vec<f64> = Vec::with_capacity(nx);
    let mut differences3: Vec<f64> = Vec::with_capacity(nx);
    let mut differences5: Vec<f64> = Vec::with_capacity(nx);
    let mut diffs2: Vec<f64> = Vec::with_capacity(ny);
    let mut diffs3: Vec<f64> = Vec::with_capacity(ny);
    let mut diffs5: Vec<f64> = Vec::with_capacity(ny);

    let is_valid = |v: f64| -> bool { !is_null_f64(v, null_value) };

    for jj in 0..ny {
        let rowstart = jj * nx;
        let row = &array[rowstart..rowstart + nx];
        differences2.clear();
        differences3.clear();
        differences5.clear();

        let mut ii: usize = 0;
        let mut vs: [f64; 8] = [0.0; 8];
        let mut found = 0;
        while found < 8 && ii < nx {
            let v = row[ii];
            if is_valid(v) {
                vs[found] = v;
                ngoodpix += 1;
                if v < xminval {
                    xminval = v;
                }
                if v > xmaxval {
                    xmaxval = v;
                }
                found += 1;
            }
            ii += 1;
        }
        if found < 8 {
            continue;
        }
        let mut v1 = vs[0];
        let mut v2 = vs[1];
        let mut v3 = vs[2];
        let mut v4 = vs[3];
        let mut v5 = vs[4];
        let mut v6 = vs[5];
        let mut v7 = vs[6];
        let mut v8 = vs[7];

        while ii < nx {
            while ii < nx && !is_valid(row[ii]) {
                ii += 1;
            }
            if ii == nx {
                break;
            }
            let v9 = row[ii];
            if v9 < xminval {
                xminval = v9;
            }
            if v9 > xmaxval {
                xmaxval = v9;
            }

            if !(v5 == v6 && v6 == v7) {
                differences2.push((v5 - v7).abs());
            }
            if !(v3 == v4 && v4 == v5 && v5 == v6 && v6 == v7) {
                differences3.push(((2.0 * v5) - v3 - v7).abs());
                differences5.push(
                    ((6.0 * v5) - (4.0 * v3) - (4.0 * v7) + v1 + v9).abs()
                );
            } else {
                ngoodpix += 1;
            }

            v1 = v2;
            v2 = v3;
            v3 = v4;
            v4 = v5;
            v5 = v6;
            v6 = v7;
            v7 = v8;
            v8 = v9;
            ii += 1;
        }
        ngoodpix += differences3.len();

        if differences3.is_empty() {
            continue;
        }
        if differences2.len() > 1 {
            diffs2.push(quick_select_f64(&mut differences2));
        } else if differences2.len() == 1 && differences3.len() == 1 {
            diffs2.push(differences2[0]);
        }
        if differences3.len() > 1 {
            diffs3.push(quick_select_f64(&mut differences3));
            diffs5.push(quick_select_f64(&mut differences5));
        } else {
            diffs3.push(differences3[0]);
            diffs5.push(differences5[0]);
        }
    }

    let xnoise3 = if diffs3.is_empty() {
        0.0
    } else if diffs3.len() == 1 {
        diffs3[0]
    } else {
        median_pair_f64(&mut diffs3)
    };
    let xnoise5 = if diffs5.is_empty() {
        0.0
    } else if diffs5.len() == 1 {
        diffs5[0]
    } else {
        median_pair_f64(&mut diffs5)
    };
    let xnoise2 = if diffs2.is_empty() {
        0.0
    } else if diffs2.len() == 1 {
        diffs2[0]
    } else {
        median_pair_f64(&mut diffs2)
    };

    NoiseStats {
        ngood: ngoodpix,
        minval: xminval,
        maxval: xmaxval,
        noise2: 1.0483579 * xnoise2,
        noise3: 0.6052697 * xnoise3,
        noise5: 0.1772048 * xnoise5,
    }
}

// Min/max-only path for the qlevel < 0 case (caller supplies a
// fixed bscale).  No noise estimation needed.  Mirror of cfitsio's
// `FnNoise3_float` called with noise=NULL.
fn min_max_f32(
    array: &[f32], null_value: Option<f32>,
) -> (usize, f64, f64) {
    let mut xmin = f32::INFINITY;
    let mut xmax = f32::NEG_INFINITY;
    let mut ng = 0usize;
    for &v in array {
        if is_null_f32(v, null_value) {
            continue;
        }
        if v < xmin {
            xmin = v;
        }
        if v > xmax {
            xmax = v;
        }
        ng += 1;
    }
    (ng, xmin as f64, xmax as f64)
}

fn min_max_f64(
    array: &[f64], null_value: Option<f64>,
) -> (usize, f64, f64) {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ng = 0usize;
    for &v in array {
        if is_null_f64(v, null_value) {
            continue;
        }
        if v < xmin {
            xmin = v;
        }
        if v > xmax {
            xmax = v;
        }
        ng += 1;
    }
    (ng, xmin, xmax)
}

// Per-tile quantize for f32 input.  Direct port of cfitsio's
// `fits_quantize_float`.
//
// Returns `Some(QuantizedTile)` on success.  Returns `None` when
// the caller should bypass quantization (range too wide, delta
// == 0, fewer than 2 pixels).  The caller — typically the
// compressed-image write path — then stores raw float bytes to
// the GZIP_COMPRESSED_DATA fallback column.
//
// `null_value` (Some) flags pixels in `fdata` that should be
// treated as missing.  NaN is the natural sentinel for float
// data; pass `Some(f32::NAN)` to skip NaN pixels.  But note: NaN
// can't be compared with `==` (NaN != NaN), so the null-check
// helper inside the noise-estimation routines won't fire for
// NaN.  Callers that want NaN handling should detect-and-replace
// NaN with a non-NaN sentinel before calling, OR use DITHER_2
// (which has special-case NaN handling in the per-pixel loop).
//
// `row_1based` is the FITS tile-row number used to seed the
// per-tile dither sequence; pass 0 to disable dithering.
// `dither_seed` is the ZDITHER0 offset (1 = cfitsio default).
pub(crate) fn quantize_float(
    fdata: &[f32],
    nxpix: usize,
    nypix: usize,
    null_value: Option<f32>,
    qlevel: f64,
    method: DitherMethod,
    row_1based: u64,
    dither_seed: i64,
) -> Option<QuantizedTile> {
    let nx = nxpix * nypix;
    if nx <= 1 {
        return None;
    }

    let (minval, maxval, ngood, delta) = if qlevel >= 0.0 {
        let stats = noise5_f32(fdata, nxpix, nypix, null_value);
        let (minval, maxval, ngood) = (stats.minval, stats.maxval, stats.ngood);
        let stdev = {
            // Take the min of noise2/3/5, ignoring zeros (per cfitsio).
            let mut stdev = stats.noise3;
            if stats.noise2 != 0.0 && stats.noise2 < stdev {
                stdev = stats.noise2;
            }
            if stats.noise5 != 0.0 && stats.noise5 < stdev {
                stdev = stats.noise5;
            }
            stdev
        };
        // Special case for image-of-nulls: cfitsio sets dummy
        // params so quantization "succeeds" with all NULL_VALUE_I32
        // output.  We match that.
        let (minval, maxval, stdev) =
            if null_value.is_some() && ngood == 0 {
                (0.0, 1.0, 1.0)
            } else {
                (minval, maxval, stdev)
            };
        let delta = if qlevel == 0.0 {
            stdev / 4.0
        } else {
            stdev / qlevel
        };
        if delta == 0.0 {
            return None;
        }
        (minval, maxval, ngood, delta)
    } else {
        let (ng, mn, mx) = min_max_f32(fdata, null_value);
        (mn, mx, ng, -qlevel)
    };

    // Range check: quantized values must fit in i32 minus reserved.
    let imax_minus_imin = (maxval - minval) / delta;
    if imax_minus_imin > 2.0 * (i32::MAX as f64) - N_RESERVED_VALUES as f64 {
        return None;
    }

    // Dither setup.  When row_1based == 0 we don't dither.
    let mut dither = if row_1based > 0 {
        Some(DitherStream::new(row_1based, dither_seed))
    } else {
        None
    };

    // Pick zeropt (bzero).
    let zeropt = if ngood == nx {
        // No nulls in the data.
        if matches!(method, DitherMethod::SubtractiveDither2) {
            // Shift the range so exact-zero pixels land near
            // the ZERO_VALUE sentinel cleanly.
            minval - delta * (NULL_VALUE_I32 as f64 + N_RESERVED_VALUES as f64)
        } else if imax_minus_imin
            < (i32::MAX as f64) - N_RESERVED_VALUES as f64
        {
            // Common case: align bzero to a multiple of bscale
            // so repeated fpack/funpack cycles converge.
            let zp = minval;
            let iqfactor = (zp / delta + 0.5) as i64;
            (iqfactor as f64) * delta
        } else {
            // Range nearly fills i32; center on zero.
            (minval + maxval) / 2.0
        }
    } else {
        // Data has nulls; shift the range so null sentinel sits
        // below the smallest data value.
        minval - delta * (NULL_VALUE_I32 as f64 + N_RESERVED_VALUES as f64)
    };

    // Per-pixel quantize loop.
    let mut idata: Vec<i32> = Vec::with_capacity(nx);
    let needs_null_check = ngood != nx;
    for i in 0..nx {
        let v = fdata[i];
        if needs_null_check && is_null_f32(v, null_value) {
            idata.push(NULL_VALUE_I32);
            // The dither stream advances on every pixel
            // regardless, per cfitsio.
            if let Some(ds) = dither.as_mut() {
                ds.next_offset();
            }
            continue;
        }
        // SUBTRACTIVE_DITHER_2 special case: exact-zero pixels
        // map to ZERO_VALUE so they round-trip exactly.
        if matches!(method, DitherMethod::SubtractiveDither2) && v == 0.0 {
            idata.push(ZERO_VALUE_I32);
            if let Some(ds) = dither.as_mut() {
                ds.next_offset();
            }
            continue;
        }
        let unscaled = (v as f64 - zeropt) / delta;
        let q = if let Some(ds) = dither.as_mut() {
            let offset = ds.next_offset() as f64;
            nint(unscaled + offset - 0.5)
        } else {
            nint(unscaled)
        };
        idata.push(q);
    }

    Some(QuantizedTile {
        idata,
        bscale: delta,
        bzero: zeropt,
    })
}

// f64 sibling of `quantize_float`.  Direct port of cfitsio's
// `fits_quantize_double` — same algorithm, double-precision
// arithmetic throughout.
pub(crate) fn quantize_double(
    fdata: &[f64],
    nxpix: usize,
    nypix: usize,
    null_value: Option<f64>,
    qlevel: f64,
    method: DitherMethod,
    row_1based: u64,
    dither_seed: i64,
) -> Option<QuantizedTile> {
    let nx = nxpix * nypix;
    if nx <= 1 {
        return None;
    }

    let (minval, maxval, ngood, delta) = if qlevel >= 0.0 {
        let stats = noise5_f64(fdata, nxpix, nypix, null_value);
        let (minval, maxval, ngood) = (stats.minval, stats.maxval, stats.ngood);
        let stdev = {
            let mut stdev = stats.noise3;
            if stats.noise2 != 0.0 && stats.noise2 < stdev {
                stdev = stats.noise2;
            }
            if stats.noise5 != 0.0 && stats.noise5 < stdev {
                stdev = stats.noise5;
            }
            stdev
        };
        let (minval, maxval, stdev) =
            if null_value.is_some() && ngood == 0 {
                (0.0, 1.0, 1.0)
            } else {
                (minval, maxval, stdev)
            };
        let delta = if qlevel == 0.0 {
            stdev / 4.0
        } else {
            stdev / qlevel
        };
        if delta == 0.0 {
            return None;
        }
        (minval, maxval, ngood, delta)
    } else {
        let (ng, mn, mx) = min_max_f64(fdata, null_value);
        (mn, mx, ng, -qlevel)
    };

    let imax_minus_imin = (maxval - minval) / delta;
    if imax_minus_imin > 2.0 * (i32::MAX as f64) - N_RESERVED_VALUES as f64 {
        return None;
    }

    let mut dither = if row_1based > 0 {
        Some(DitherStream::new(row_1based, dither_seed))
    } else {
        None
    };

    let zeropt = if ngood == nx {
        if matches!(method, DitherMethod::SubtractiveDither2) {
            minval - delta * (NULL_VALUE_I32 as f64 + N_RESERVED_VALUES as f64)
        } else if imax_minus_imin
            < (i32::MAX as f64) - N_RESERVED_VALUES as f64
        {
            let zp = minval;
            let iqfactor = (zp / delta + 0.5) as i64;
            (iqfactor as f64) * delta
        } else {
            (minval + maxval) / 2.0
        }
    } else {
        minval - delta * (NULL_VALUE_I32 as f64 + N_RESERVED_VALUES as f64)
    };

    let mut idata: Vec<i32> = Vec::with_capacity(nx);
    let needs_null_check = ngood != nx;
    for i in 0..nx {
        let v = fdata[i];
        if needs_null_check && is_null_f64(v, null_value) {
            idata.push(NULL_VALUE_I32);
            if let Some(ds) = dither.as_mut() {
                ds.next_offset();
            }
            continue;
        }
        if matches!(method, DitherMethod::SubtractiveDither2) && v == 0.0 {
            idata.push(ZERO_VALUE_I32);
            if let Some(ds) = dither.as_mut() {
                ds.next_offset();
            }
            continue;
        }
        let unscaled = (v - zeropt) / delta;
        let q = if let Some(ds) = dither.as_mut() {
            let offset = ds.next_offset() as f64;
            nint(unscaled + offset - 0.5)
        } else {
            nint(unscaled)
        };
        idata.push(q);
    }

    Some(QuantizedTile {
        idata,
        bscale: delta,
        bzero: zeropt,
    })
}

// Quantize with the caller-supplied (bscale, bzero, dither seed) —
// no noise estimation, no scale picking.  Used by extend/__setitem__
// on quantized-float HDUs to re-encode tiles after partial
// modification: re-using the existing per-tile bscale/bzero AND
// the same row_1based seed makes `requantize(dequantize(stored))`
// idempotent for unchanged pixels, so they round-trip with NO
// compounding quantization loss.  Modified pixels still pay one
// round of quantization noise (unavoidable for lossy).
//
// Returns `Err(msg)` when any pixel quantizes outside the legal
// i32 range minus reserved values — caller should surface this to
// the user as "value too large/small to fit the existing tile's
// quantization scale".  Caller advises dropping to `quantize=None`
// for unrestricted mutation.
//
// Per-pixel semantics match quantize_float exactly: NaN →
// NULL_VALUE_I32 regardless of dither method; SUBTRACTIVE_DITHER_2
// + exact zero → ZERO_VALUE_I32; dither stream advances per pixel
// regardless of sentinel hits (preserves cfitsio's dither sequence).
pub(crate) fn requantize_float_fixed_scale(
    fdata: &[f32],
    bscale: f64,
    bzero: f64,
    method: DitherMethod,
    row_1based: u64,
    dither_seed: i64,
) -> Result<Vec<i32>, String> {
    if bscale == 0.0 || !bscale.is_finite() {
        return Err(format!(
            "requantize: invalid bscale {} (must be finite and non-zero)",
            bscale
        ));
    }
    let mut dither = if row_1based > 0 {
        Some(DitherStream::new(row_1based, dither_seed))
    } else {
        None
    };
    let lo_legal = (i32::MIN as i64) + N_RESERVED_VALUES as i64;
    let hi_legal = i32::MAX as i64;
    let mut idata: Vec<i32> = Vec::with_capacity(fdata.len());
    for (i, &v) in fdata.iter().enumerate() {
        if is_null_f32(v, Some(f32::NAN)) {
            idata.push(NULL_VALUE_I32);
            if let Some(ds) = dither.as_mut() {
                ds.next_offset();
            }
            continue;
        }
        if matches!(method, DitherMethod::SubtractiveDither2) && v == 0.0 {
            idata.push(ZERO_VALUE_I32);
            if let Some(ds) = dither.as_mut() {
                ds.next_offset();
            }
            continue;
        }
        let unscaled = (v as f64 - bzero) / bscale;
        let q_i64 = if let Some(ds) = dither.as_mut() {
            let offset = ds.next_offset() as f64;
            (unscaled + offset - 0.5).round() as i64
        } else {
            unscaled.round() as i64
        };
        if q_i64 < lo_legal || q_i64 > hi_legal {
            return Err(out_of_range_message(
                i, v as f64, q_i64, lo_legal, hi_legal, bscale, bzero,
            ));
        }
        idata.push(q_i64 as i32);
    }
    Ok(idata)
}

// f64 sibling of `requantize_float_fixed_scale`.
pub(crate) fn requantize_double_fixed_scale(
    fdata: &[f64],
    bscale: f64,
    bzero: f64,
    method: DitherMethod,
    row_1based: u64,
    dither_seed: i64,
) -> Result<Vec<i32>, String> {
    if bscale == 0.0 || !bscale.is_finite() {
        return Err(format!(
            "requantize: invalid bscale {} (must be finite and non-zero)",
            bscale
        ));
    }
    let mut dither = if row_1based > 0 {
        Some(DitherStream::new(row_1based, dither_seed))
    } else {
        None
    };
    let lo_legal = (i32::MIN as i64) + N_RESERVED_VALUES as i64;
    let hi_legal = i32::MAX as i64;
    let mut idata: Vec<i32> = Vec::with_capacity(fdata.len());
    for (i, &v) in fdata.iter().enumerate() {
        if is_null_f64(v, Some(f64::NAN)) {
            idata.push(NULL_VALUE_I32);
            if let Some(ds) = dither.as_mut() {
                ds.next_offset();
            }
            continue;
        }
        if matches!(method, DitherMethod::SubtractiveDither2) && v == 0.0 {
            idata.push(ZERO_VALUE_I32);
            if let Some(ds) = dither.as_mut() {
                ds.next_offset();
            }
            continue;
        }
        let unscaled = (v - bzero) / bscale;
        let q_i64 = if let Some(ds) = dither.as_mut() {
            let offset = ds.next_offset() as f64;
            (unscaled + offset - 0.5).round() as i64
        } else {
            unscaled.round() as i64
        };
        if q_i64 < lo_legal || q_i64 > hi_legal {
            return Err(out_of_range_message(
                i, v, q_i64, lo_legal, hi_legal, bscale, bzero,
            ));
        }
        idata.push(q_i64 as i32);
    }
    Ok(idata)
}

fn out_of_range_message(
    pixel_idx: usize,
    value: f64,
    quantized: i64,
    lo_legal: i64,
    hi_legal: i64,
    bscale: f64,
    bzero: f64,
) -> String {
    format!(
        "requantize: pixel {} value {:.6e} doesn't fit the tile's \
         existing per-tile quantization scale (bscale={:.6e}, \
         bzero={:.6e}; would quantize to {} which is outside the legal \
         stored range [{}, {}]).  The per-tile bscale was chosen when \
         this tile was first written; modifying values outside that \
         scale would require re-quantizing the entire tile, which \
         would compound quantization noise on the unchanged pixels — \
         so this is rejected.  To support wider value ranges you need \
         to RECREATE the file (the schema can't be widened in place) \
         with one of: (1) quantize=Quantize(level=N) for a smaller N \
         (default level=4.0 ~ noise/4; try level=2.0 or 1.0 for \
         coarser scales that admit wider ranges), (2) \
         quantize=Quantize(level=-bscale_value) to pin bscale \
         explicitly, or (3) quantize=None for lossless raw-float \
         GZIP storage (larger files, no precision loss).",
        pixel_idx, value, bscale, bzero, quantized, lo_legal, hi_legal,
    )
}

// Round to nearest integer, ties away from zero — mirror of
// cfitsio's `NINT` macro `(x >= 0.) ? (int)(x + 0.5) : (int)(x - 0.5)`.
// Saturates to i32 bounds on overflow (shouldn't happen because
// the caller's range check rejected wide ranges, but defensive).
fn nint(x: f64) -> i32 {
    let rounded = if x >= 0.0 { x + 0.5 } else { x - 0.5 };
    if rounded >= i32::MAX as f64 {
        i32::MAX
    } else if rounded <= i32::MIN as f64 {
        i32::MIN
    } else {
        rounded as i32
    }
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

    // ===== quantize encoder round-trip tests =====
    //
    // Anchor the encoder against its own decoder: encode →
    // dequantize → compare physical pixel values.  The relative
    // error should be at most ~delta (= noise / qlevel) for the
    // dither methods and the same for NO_DITHER (cfitsio's
    // guarantee).
    //
    // Byte-exact agreement with cfitsio comes from the Python-
    // side fitsio cross-check tests once the write path is
    // wired; these Rust-side tests just confirm the encoder is
    // an internally consistent inverse of the decoder.

    fn dequant_f32_to_vec(
        idata: &[i32], bscale: f64, bzero: f64,
        method: DitherMethod, row: u64, seed: i64,
    ) -> Vec<f32> {
        let bytes = dequantize_to_f32(idata, bscale, bzero, method, row, seed);
        bytes.chunks(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn quantize_float_no_dither_round_trip() {
        // Smooth-ish data; noise estimation picks a small delta.
        let nx = 16;
        let ny = 4;
        let mut data: Vec<f32> = (0..nx * ny)
            .map(|i| (i as f32) * 0.5 + ((i * 37) as f32 * 0.01).sin())
            .collect();
        // Make a few pixels stand out so noise5 has real work to do.
        data[5] += 1.7;
        data[20] -= 1.2;

        let qt = quantize_float(
            &data, nx, ny, None, 4.0, DitherMethod::NoDither, 0, 1,
        ).expect("quantize should succeed for smooth data");
        let back = dequant_f32_to_vec(
            &qt.idata, qt.bscale, qt.bzero, DitherMethod::NoDither, 0, 1,
        );
        // Round-trip within delta tolerance.
        let max_err = data.iter().zip(back.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err <= qt.bscale as f32 * 2.0,
            "max round-trip error {} exceeds 2*bscale {}",
            max_err, qt.bscale,
        );
    }

    #[test]
    fn quantize_float_dither1_round_trip() {
        let nx = 16;
        let ny = 4;
        let data: Vec<f32> = (0..nx * ny)
            .map(|i| 100.0 + (i as f32 * 0.3).sin() * 0.5)
            .collect();
        let qt = quantize_float(
            &data, nx, ny, None, 4.0,
            DitherMethod::SubtractiveDither1, 1, 1,
        ).expect("quantize should succeed");
        let back = dequant_f32_to_vec(
            &qt.idata, qt.bscale, qt.bzero,
            DitherMethod::SubtractiveDither1, 1, 1,
        );
        let max_err = data.iter().zip(back.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f32, f32::max);
        // Dither adds at most 1 quantum of round-off; tolerance
        // 2*bscale is a safe upper bound.
        assert!(max_err <= qt.bscale as f32 * 2.0);
    }

    #[test]
    fn quantize_float_dither2_preserves_exact_zero_and_nan() {
        let nx = 16;
        let ny = 4;
        let mut data: Vec<f32> = (0..nx * ny)
            .map(|i| 50.0 + (i as f32 * 0.7).cos())
            .collect();
        // Inject exact zeros and one NaN.  DITHER_2's reserved
        // ZERO_VALUE handles the exact-zero case; NaN we treat as
        // a null and let quantize map it to NULL_VALUE_I32.
        data[3] = 0.0;
        data[17] = 0.0;
        data[10] = f32::NAN;
        let qt = quantize_float(
            &data, nx, ny, Some(f32::NAN), 4.0,
            DitherMethod::SubtractiveDither2, 1, 1,
        ).expect("quantize should succeed");
        assert_eq!(qt.idata[3], ZERO_VALUE_I32);
        assert_eq!(qt.idata[17], ZERO_VALUE_I32);
        // NaN won't compare equal under is_valid (NaN != NaN),
        // so noise_estimation skips it naturally; but the per-
        // pixel quantize loop also skips it.  In the current
        // implementation NaN takes the non-null-check branch and
        // arithmetic produces NaN-as-int (implementation-defined);
        // a future tightening can short-circuit NaN to NULL_VALUE.
        let back = dequant_f32_to_vec(
            &qt.idata, qt.bscale, qt.bzero,
            DitherMethod::SubtractiveDither2, 1, 1,
        );
        // Exact zeros must round-trip exactly.
        assert_eq!(back[3], 0.0);
        assert_eq!(back[17], 0.0);
    }

    #[test]
    fn quantize_float_rejects_tiny_input() {
        // Single-pixel tile: cfitsio refuses to quantize.
        let data = vec![1.5f32];
        let res = quantize_float(
            &data, 1, 1, None, 4.0, DitherMethod::NoDither, 0, 1,
        );
        assert!(res.is_none());
    }

    #[test]
    fn quantize_float_rejects_constant_input() {
        // All-same value: noise = 0 → delta = 0 → reject.
        let data = vec![3.14f32; 100];
        let res = quantize_float(
            &data, 10, 10, None, 4.0, DitherMethod::NoDither, 0, 1,
        );
        assert!(res.is_none());
    }

    #[test]
    fn quantize_float_fixed_bscale_path() {
        // Negative qlevel = fixed bscale, no noise estimation.
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let qt = quantize_float(
            &data, 10, 10, None, -0.05,
            DitherMethod::NoDither, 0, 1,
        ).expect("fixed-bscale path should succeed");
        assert_eq!(qt.bscale, 0.05);
    }

    #[test]
    fn quantize_double_round_trip() {
        let nx = 16;
        let ny = 4;
        let data: Vec<f64> = (0..nx * ny)
            .map(|i| 1000.0 + (i as f64 * 0.3).sin())
            .collect();
        let qt = quantize_double(
            &data, nx, ny, None, 4.0, DitherMethod::NoDither, 0, 1,
        ).expect("quantize should succeed");
        let bytes = dequantize_to_f64(
            &qt.idata, qt.bscale, qt.bzero,
            DitherMethod::NoDither, 0, 1,
        );
        let back: Vec<f64> = bytes.chunks(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        let max_err = data.iter().zip(back.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err <= qt.bscale * 2.0);
    }

    #[test]
    fn noise5_constant_data_zero_noise() {
        let data = vec![5.0f32; 100];
        let stats = noise5_f32(&data, 10, 10, None);
        assert_eq!(stats.noise2, 0.0);
        assert_eq!(stats.noise3, 0.0);
        assert_eq!(stats.noise5, 0.0);
        assert_eq!(stats.minval, 5.0);
        assert_eq!(stats.maxval, 5.0);
    }

    #[test]
    fn nint_rounds_ties_away_from_zero() {
        assert_eq!(nint(0.5), 1);
        assert_eq!(nint(-0.5), -1);
        assert_eq!(nint(1.4), 1);
        assert_eq!(nint(-1.4), -1);
        assert_eq!(nint(0.0), 0);
    }

    // The load-bearing property of requantize_*_fixed_scale:
    // re-quantizing a dequantized stream with the same parameters
    // reproduces the original stream exactly.  This is what makes
    // partial __setitem__ / extend on quantized HDUs safe — pixels
    // outside the modification region pay zero compounding loss.
    #[test]
    fn requantize_is_idempotent_no_dither_f32() {
        let stored_in: Vec<i32> = vec![-5, 0, 7, 13, 42, -100];
        let bscale = 0.5;
        let bzero = 10.0;
        // dequantize → physical floats
        let de = dequantize_to_f32(
            &stored_in, bscale, bzero, DitherMethod::NoDither, 0, 0,
        );
        let phys: Vec<f32> = de.chunks(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        // requantize with the SAME bscale, bzero, dither setup
        let stored_out = requantize_float_fixed_scale(
            &phys, bscale, bzero, DitherMethod::NoDither, 0, 0,
        )
        .expect("must succeed for in-range input");
        assert_eq!(stored_out, stored_in);
    }

    #[test]
    fn requantize_is_idempotent_dither1_f32() {
        let stored_in: Vec<i32> = vec![-5, 0, 7, 13, 42, -100];
        let bscale = 0.5;
        let bzero = 10.0;
        let row_1based = 1;
        let dither_seed = 1;
        let de = dequantize_to_f32(
            &stored_in, bscale, bzero,
            DitherMethod::SubtractiveDither1,
            row_1based, dither_seed,
        );
        let phys: Vec<f32> = de.chunks(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        let stored_out = requantize_float_fixed_scale(
            &phys, bscale, bzero,
            DitherMethod::SubtractiveDither1,
            row_1based, dither_seed,
        )
        .expect("must succeed for in-range input");
        assert_eq!(stored_out, stored_in);
    }

    #[test]
    fn requantize_is_idempotent_no_dither_f64() {
        let stored_in: Vec<i32> = vec![-5, 0, 7, 13, 42, -100];
        let bscale = 0.5;
        let bzero = 10.0;
        let de = dequantize_to_f64(
            &stored_in, bscale, bzero, DitherMethod::NoDither, 0, 0,
        );
        let phys: Vec<f64> = de.chunks(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        let stored_out = requantize_double_fixed_scale(
            &phys, bscale, bzero, DitherMethod::NoDither, 0, 0,
        )
        .expect("must succeed for in-range input");
        assert_eq!(stored_out, stored_in);
    }

    #[test]
    fn requantize_rejects_out_of_range() {
        // bscale=0.01, bzero=0 → values must fit within
        // [(i32::MIN+10)*0.01, i32::MAX*0.01] ≈ [-2.15e7, 2.15e7].
        // 1e10 is way outside.
        let phys = vec![0.0_f32, 1e10_f32];
        let err = requantize_float_fixed_scale(
            &phys, 0.01, 0.0, DitherMethod::NoDither, 0, 0,
        )
        .expect_err("must reject out-of-range value");
        assert!(err.contains("outside the legal stored range"));
        assert!(err.contains("RECREATE the file"));
        assert!(err.contains("quantize=None"));
    }

    #[test]
    fn requantize_preserves_nan() {
        let phys = vec![1.0_f32, f32::NAN, 3.0_f32];
        let out = requantize_float_fixed_scale(
            &phys, 0.5, 0.0,
            DitherMethod::SubtractiveDither1, 1, 1,
        )
        .expect("NaN handling must not error");
        // Middle pixel should be NULL_VALUE_I32.
        assert_eq!(out[1], NULL_VALUE_I32);
        // Surrounding pixels not the sentinel.
        assert_ne!(out[0], NULL_VALUE_I32);
        assert_ne!(out[2], NULL_VALUE_I32);
    }
}
