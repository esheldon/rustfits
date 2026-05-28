// clippy: the index-based loops in this file mirror the cfitsio C
// source line-for-line, keeping the port diffable against upstream;
// don't rewrite them into iterator form.
#![allow(clippy::needless_range_loop)]

// PLIO_1 tile decompression for the FITS Tile Compression
// Convention.  Port of cfitsio's `pl_l2pi` from
// `/home/esheldon/git/fitsio/cfitsio-4.6.4/pliocomp.c` (which is
// itself a translation of IRAF's SPP routine via f2c).  The SPP
// source's goto-heavy control flow has been straightened out here
// — the opcode dispatch is now a `match` on the top 4 bits of
// each encoded word.
//
// PLIO is designed for mask / pixel-list images where most pixels
// are 0 and the non-zero pixels come in runs of constant value.
// The encoded stream is a sequence of 16-bit big-endian shorts:
//
//   Header (7 shorts):
//     [0] = 0                    (unused)
//     [1] = 7                    (offset to first data word)
//     [2] = -100                 (version marker; old format uses
//                                 a positive value here as the
//                                 stream length, in which case
//                                 the data starts at offset 3)
//     [3] = (length-1) & 0x7FFF  (lower 15 bits of total length
//                                 in shorts)
//     [4] = (length-1) >> 15     (upper bits of total length)
//     [5] = 0
//     [6] = 0
//
//   Data words (each is opcode in top 4 bits, data in bottom 12):
//     opcode 0 (ZN):  run of `data` zero pixels
//     opcode 1 (SH):  set high value pv = (next_word << 12) + data
//                     — the next word is consumed and skipped
//     opcode 2 (PH):  pv += data
//     opcode 3 (MH):  pv -= data
//     opcode 4 (PN):  run of `data` pixels each equal to pv
//     opcode 5 (ZN+):  run of `data-1` zeros followed by pv
//     opcode 6 (PS):  pv += data, then write single pixel pv
//     opcode 7 (MS):  pv -= data, then write single pixel pv
//
// The encoder normalizes all input pixels via `max(0, v)`, so the
// output is always non-negative even if some clever caller fed
// negatives to the encoder.  Trailing un-written pixels stay 0
// (our output buffer is zero-initialised).
//
// Pixel values produced by PLIO are integer; cfitsio's
// imcompress.c always uses TINT (i32) as the intermediate type,
// then casts to the target bitpix.  We do the same: decode to
// `Vec<i32>`, then cast.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// ===== Entry point =====

pub(crate) fn decode_plio(
    compressed: &[u8],
    n_pixels: usize,
    bytepix: u32,
    zbitpix: i32,
) -> PyResult<Vec<u8>> {
    if !matches!(zbitpix, 8 | 16 | 32) {
        return Err(PyValueError::new_err(format!(
            "PLIO_1: unsupported ZBITPIX {} (must be 8, 16, or 32)",
            zbitpix
        )));
    }
    if !matches!(bytepix, 1 | 2 | 4) {
        return Err(PyValueError::new_err(format!(
            "PLIO_1: unsupported bytepix {} (must be 1, 2, or 4)",
            bytepix
        )));
    }
    if !compressed.len().is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "PLIO_1: compressed payload is {} bytes (must be even — \
             stream is i16 shorts)",
            compressed.len()
        )));
    }
    let n_words = compressed.len() / 2;
    if n_words < 7 {
        return Err(PyValueError::new_err(format!(
            "PLIO_1: header truncated (need 7 shorts, got {})", n_words
        )));
    }

    // Read one BE i16 at 0-based word index.
    let word_at = |i: usize| -> i16 {
        i16::from_be_bytes([compressed[2 * i], compressed[2 * i + 1]])
    };

    // Parse header.  Two formats coexist: the old one stores length
    // in word [2] as a positive value (and data starts at word [3]);
    // the modern one stores -100 there as a marker, length split
    // across words [3..5], and data starts at word [1]+1 = 7.
    let header_2 = word_at(2);
    let (lllen, llfirt) = if header_2 > 0 {
        // Old format: word [2] is the total length (positive),
        // data starts at word [3] (0-based).
        (header_2 as usize, 3usize)
    } else {
        // Modern format: length is split across words [3] (lower
        // 15 bits) and [4] (upper bits).  Word [1] holds the
        // 1-based index of the last header word, which is the
        // 0-based offset of the first data word — i.e., 7.
        let lower = (word_at(3) as i32) & 0x7FFF;
        let upper = word_at(4) as i32;
        let lllen = (upper << 15) | lower;
        if lllen < 0 {
            return Err(PyValueError::new_err(
                "PLIO_1: negative computed stream length",
            ));
        }
        let llfirt_w = word_at(1);
        if llfirt_w < 0 {
            return Err(PyValueError::new_err(
                "PLIO_1: negative header data-offset",
            ));
        }
        (lllen as usize, llfirt_w as usize)
    };

    if lllen > n_words {
        return Err(PyValueError::new_err(format!(
            "PLIO_1: header claims {} shorts but only {} are present",
            lllen, n_words
        )));
    }
    if llfirt > lllen {
        return Err(PyValueError::new_err(format!(
            "PLIO_1: header offset {} is past stream end {}",
            llfirt, lllen
        )));
    }

    // Output buffer, zero-initialised so trailing un-written pixels
    // are correct without an explicit fill pass.
    let mut output = vec![0i32; n_pixels];

    // Decode loop.  cfitsio uses 1-based indexing throughout (xs
    // = 1, xe = npix, x1 starts at 1, op starts at 1).  We keep
    // the 1-based pixel indices (xs/xe/x1) because comparisons
    // with xs/xe are arithmetic checks rather than buffer accesses,
    // but `op` is converted to a 0-based output index.
    let xs: i64 = 1;
    let xe: i64 = n_pixels as i64;
    let mut x1: i64 = 1;
    let mut pv: i32 = 1;
    let mut op: usize = 0; // 0-based output index
    let mut skipwd = false;

    let mut ip = llfirt;
    while ip < lllen && x1 <= xe {
        if skipwd {
            skipwd = false;
            ip += 1;
            continue;
        }
        let raw = word_at(ip) as u16;
        let opcode = (raw >> 12) & 0x0F;
        let data: i64 = (raw & 0x0FFF) as i64;

        match opcode {
            0 | 4 | 5 => {
                // Run.  opcode 0 = all zeros; 4 = all pv; 5 = zeros
                // with last pixel = pv.
                let x2 = x1 + data - 1;
                let i1 = x1.max(xs);
                let i2 = x2.min(xe);
                let np = i2 - i1 + 1;
                if np > 0 {
                    let np_us = np as usize;
                    let otop = op + np_us - 1;
                    if otop >= n_pixels {
                        return Err(PyValueError::new_err(format!(
                            "PLIO_1: decoded output overflow at opcode {} \
                             (op={}, np={}, n_pixels={})",
                            opcode, op, np_us, n_pixels
                        )));
                    }
                    if opcode == 4 {
                        for v in &mut output[op..=otop] { *v = pv; }
                    } else {
                        // opcode 0 and 5: zeros (already zero-init,
                        // but explicit so re-decoding a tile into
                        // a recycled buffer would still be correct).
                        for v in &mut output[op..=otop] { *v = 0; }
                        if opcode == 5 && i2 == x2 {
                            output[otop] = pv;
                        }
                    }
                    op = otop + 1;
                }
                x1 = x2 + 1;
            }
            1 => {
                // SH: pv = (next_word << 12) + data.  The next
                // word is "data only" — skip its opcode dispatch.
                if ip + 1 >= lllen {
                    return Err(PyValueError::new_err(
                        "PLIO_1: SH opcode at end of stream (no following \
                         data word)",
                    ));
                }
                let next = word_at(ip + 1) as i32;
                pv = (next << 12) + data as i32;
                skipwd = true;
            }
            2 => {
                // PH: increment pv.
                pv = pv.wrapping_add(data as i32);
            }
            3 => {
                // MH: decrement pv.
                pv = pv.wrapping_sub(data as i32);
            }
            6 => {
                // PS: pv += data, then write a single pixel at x1.
                pv = pv.wrapping_add(data as i32);
                if x1 >= xs && x1 <= xe {
                    if op >= n_pixels {
                        return Err(PyValueError::new_err(
                            "PLIO_1: PS overflow",
                        ));
                    }
                    output[op] = pv;
                    op += 1;
                }
                x1 += 1;
            }
            7 => {
                // MS: pv -= data, then write a single pixel.
                pv = pv.wrapping_sub(data as i32);
                if x1 >= xs && x1 <= xe {
                    if op >= n_pixels {
                        return Err(PyValueError::new_err(
                            "PLIO_1: MS overflow",
                        ));
                    }
                    output[op] = pv;
                    op += 1;
                }
                x1 += 1;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "PLIO_1: unknown opcode {} at word index {}",
                    opcode, ip
                )));
            }
        }

        ip += 1;
    }

    // Cast to target dtype in numpy native byte order.
    cast_i32_to_target_bytes(&output, bytepix, zbitpix)
}

// ===== dtype cast =====
//
// PLIO produces non-negative integer pixel values.  Cast to the
// target ZBITPIX dtype in native byte order; values exceeding the
// target type's range are silently truncated (cfitsio does the
// same — it's the encoder's job to choose ZBITPIX wide enough for
// the data).

fn cast_i32_to_target_bytes(
    a: &[i32], bytepix: u32, zbitpix: i32,
) -> PyResult<Vec<u8>> {
    let n = a.len();
    match (bytepix, zbitpix) {
        (1, 8) => {
            let mut out = Vec::with_capacity(n);
            for &v in a { out.push(v as u8); }
            Ok(out)
        }
        (2, 16) => {
            let mut out = Vec::with_capacity(n * 2);
            for &v in a {
                out.extend_from_slice(&(v as i16).to_ne_bytes());
            }
            Ok(out)
        }
        (4, 32) => {
            let mut out = Vec::with_capacity(n * 4);
            for &v in a {
                out.extend_from_slice(&v.to_ne_bytes());
            }
            Ok(out)
        }
        _ => Err(PyValueError::new_err(format!(
            "PLIO_1 cast i32→target: unsupported (bytepix={}, zbitpix={})",
            bytepix, zbitpix
        ))),
    }
}

// ===========================================================================
// ===== ENCODE ==============================================================
// ===========================================================================
//
// Port of cfitsio's `pl_p2li` from
// `<cfitsio>/pliocomp.c::pl_p2li`.  The SPP/f2c-generated goto soup is
// straightened into a single `while` loop with explicit state and the
// emit logic written linearly; the bit-exactness is verified against
// fitsio in the test suite.
//
// The encoder works pixel-by-pixel.  It coalesces runs of equal value:
// for each run of value `pv` we may emit three things (any subset, in
// order):
//   1. A "set pv" word (opcode 2/3) or two-word set-high (opcode 1)
//      when pv differs from `hi` (the last set value).  Single-pixel
//      runs with no preceding zero gap collapse into opcode 6/7
//      (set-and-write).
//   2. Zero-run words (opcode 0) for any preceding zero gap.  A
//      single trailing pv pixel after a zero gap collapses into
//      opcode 5 (zero-run-with-trailing-pv).
//   3. Solid-pv run words (opcode 4) for the run itself.
//
// Wire format: 7-short header (see decoder's doc comment) plus N data
// words.  Total length lives at header[3]/[4].  All values are i16
// big-endian on disk.
//
// Pixel-value range: PLIO encodes pv as a 12-bit low half + a 15-bit
// high half (since the high word is read as a signed i16 then
// composed as `(high << 12) | low12`).  So the max representable pv
// is (2^15 - 1) * 4096 + 4095 = 134_217_727 = 2^27 - 1.  The encoder
// rejects anything beyond that range.

// PLIO max representable pv (2^27 - 1).  Anything larger overflows
// the two-word set-pv encoding (cfitsio truncates silently; we
// reject with a clear error instead).
const PLIO_MAX_PV: i32 = 0x07FF_FFFF;

// Encode `pixels` (already cast to i32) to a PLIO_1 byte stream.
// Returns the heap bytes (big-endian i16 shorts) ready to write
// into the COMPRESSED_DATA column.
fn encode_plio_i32(pixels: &[i32]) -> PyResult<Vec<u8>> {
    // Up-front validation: PLIO inputs must be non-negative integers
    // (the algorithm is built around `pv += data` increments from
    // pv=1; negatives can't be represented).
    for (i, &v) in pixels.iter().enumerate() {
        if v < 0 {
            return Err(PyValueError::new_err(format!(
                "PLIO_1 encode: pixel {} = {} is negative; PLIO is \
                 only defined for non-negative integer masks",
                i, v
            )));
        }
        if v > PLIO_MAX_PV {
            return Err(PyValueError::new_err(format!(
                "PLIO_1 encode: pixel {} = {} exceeds the PLIO \
                 maximum of {} (2^27 - 1)",
                i, v, PLIO_MAX_PV
            )));
        }
    }

    let npix = pixels.len();
    let mut out: Vec<i16> = Vec::with_capacity(7 + npix.max(1));

    // 7-short header; length fields at [3]/[4] filled after the
    // emit loop.
    out.extend_from_slice(&[0i16, 7, -100, 0, 0, 0, 0]);

    if npix == 0 {
        let total: i32 = 7;
        out[3] = (total % 32768) as i16;
        out[4] = (total / 32768) as i16;
        return Ok(shorts_to_be_bytes(&out));
    }

    // State variables — names mirror cfitsio's `pl_p2li`:
    //   pv  : value of the current run
    //   hi  : last "set" pv value (delta-coded against)
    //   x1  : start index of current pv run
    //   iz  : start index of current zero region
    let mut pv: i32 = pixels[0].max(0);
    let mut hi: i32 = 1;
    let mut x1: usize = 0;
    let mut iz: usize = 0;

    let mut ip: usize = 0;
    while ip < npix {
        let mut next_pv = pv;
        if ip < npix - 1 {
            let nv = pixels[ip + 1].max(0);
            if nv == pv {
                // Same value; keep building the run.
                ip += 1;
                continue;
            }
            if pv == 0 {
                // Transition zero → nonzero; slide the run start
                // without emitting anything yet.
                pv = nv;
                x1 = ip + 1;
                ip += 1;
                continue;
            }
            // pv > 0 and nv != pv — emit the pv run.
            next_pv = nv;
        } else {
            // Last pixel; no lookahead.
            if pv == 0 {
                x1 = npix; // np becomes 0 below
            }
        }

        let np = (ip + 1).saturating_sub(x1);
        let nz = x1 - iz;

        // (1) Set-pv emission (opcode 2/3 or two-word opcode 1).
        let mut consumed_by_single_pixel = false;
        if pv > 0 {
            let dv = pv - hi;
            if dv != 0 {
                hi = pv;
                if dv.abs() > 4095 {
                    out.push(((pv & 4095) + 4096) as i16);
                    out.push((pv >> 12) as i16);
                } else if dv < 0 {
                    out.push((-dv + 12288) as i16);
                } else {
                    out.push((dv + 8192) as i16);
                }
                if np == 1 && nz == 0 {
                    // Single-pixel write: OR 16384 onto the set-pv
                    // word, turning opcode 2/3 into 6/7.
                    let last = out.len() - 1;
                    out[last] = ((out[last] as u16) | 16384) as i16;
                    consumed_by_single_pixel = true;
                }
            }
        }

        if !consumed_by_single_pixel {
            // (2) Zero-run emission (opcode 0).
            let mut nz_remain = nz as i32;
            let mut emitted_zero = false;
            while nz_remain > 0 {
                out.push(nz_remain.min(4095) as i16);
                nz_remain -= 4095;
                emitted_zero = true;
            }

            if emitted_zero && np == 1 && pv > 0 {
                // Combined opcode 5: zero run + last pv pixel.
                // Adding 20481 = 5*4096 + 1 changes the opcode of
                // the last zero-run word from 0 to 5 and bumps its
                // length by 1 (the trailing pv pixel).
                let last = out.len() - 1;
                out[last] = (out[last] as i32 + 20481) as i16;
            } else {
                // (3) Solid-pv run emission (opcode 4).
                let mut np_remain = np as i32;
                while np_remain > 0 {
                    out.push((np_remain.min(4095) + 16384) as i16);
                    np_remain -= 4095;
                }
            }
        }

        // Advance to the next pixel; next_pv is the peeked value
        // (or pv when at the last iteration, which is fine since
        // the loop exits).
        ip += 1;
        x1 = ip;
        iz = ip;
        pv = next_pv;
    }

    // Write the total length (in shorts) back into the header.
    let total = out.len() as i32;
    out[3] = (total % 32768) as i16;
    out[4] = (total / 32768) as i16;

    Ok(shorts_to_be_bytes(&out))
}

// Pack a slice of i16 values as a big-endian byte buffer.  Two
// allocations would be wasteful here; we flatten directly into a
// pre-sized Vec<u8>.
fn shorts_to_be_bytes(shorts: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(shorts.len() * 2);
    for &s in shorts {
        bytes.extend_from_slice(&s.to_be_bytes());
    }
    bytes
}

// Public encode entry.  Mirror of `decode_plio`: convert the input
// pixel bytes (FITS big-endian) to i32, then run `encode_plio_i32`.
//
// PLIO is integer-only; float ZBITPIX is rejected upstream in the
// dispatch (encode_tile_from_bytes / create_compressed_image_hdu_impl).
pub(crate) fn encode_plio(
    pixel_bytes_be: &[u8],
    n_pixels: usize,
    bytepix: u32,
    zbitpix: i32,
) -> PyResult<Vec<u8>> {
    if !matches!(zbitpix, 8 | 16 | 32) {
        return Err(PyValueError::new_err(format!(
            "PLIO_1 encode: unsupported ZBITPIX {} (must be 8/16/32; \
             PLIO is integer-only and has no 64-bit variant)",
            zbitpix
        )));
    }
    if !matches!(bytepix, 1 | 2 | 4) {
        return Err(PyValueError::new_err(format!(
            "PLIO_1 encode: unsupported bytepix {} (must be 1/2/4)",
            bytepix
        )));
    }
    let expected = n_pixels.checked_mul(bytepix as usize)
        .ok_or_else(|| PyValueError::new_err(
            "PLIO_1 encode: n_pixels * bytepix overflowed usize"
        ))?;
    if pixel_bytes_be.len() != expected {
        return Err(PyValueError::new_err(format!(
            "PLIO_1 encode: input length {} != n_pixels * bytepix ({})",
            pixel_bytes_be.len(), expected
        )));
    }

    // Convert input bytes to i32, sign-extending from the natural
    // width.  u8 input zero-extends; i16 and i32 sign-extend.
    let mut pixels: Vec<i32> = Vec::with_capacity(n_pixels);
    match (bytepix, zbitpix) {
        (1, 8) => {
            for i in 0..n_pixels {
                pixels.push(pixel_bytes_be[i] as i32);
            }
        }
        (2, 16) => {
            for i in 0..n_pixels {
                let off = i * 2;
                let bytes = [pixel_bytes_be[off], pixel_bytes_be[off + 1]];
                pixels.push(i16::from_be_bytes(bytes) as i32);
            }
        }
        (4, 32) => {
            for i in 0..n_pixels {
                let off = i * 4;
                let bytes = [
                    pixel_bytes_be[off],
                    pixel_bytes_be[off + 1],
                    pixel_bytes_be[off + 2],
                    pixel_bytes_be[off + 3],
                ];
                pixels.push(i32::from_be_bytes(bytes));
            }
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "PLIO_1 encode: bytepix/zbitpix mismatch \
                 (bytepix={}, zbitpix={})",
                bytepix, zbitpix
            )));
        }
    }
    encode_plio_i32(&pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny hand-rolled stream: header + 1 data word (opcode 4 =
    // PN, data=3) encodes "run of 3 pv=1 pixels" (pv defaults to 1).
    #[test]
    fn pn_run_of_three_ones() {
        // 7-short header: [0, 7, -100, lllen_lower, lllen_upper, 0, 0]
        // followed by 1 data word.  Total lllen = 8.
        let mut bytes = Vec::new();
        let put = |b: &mut Vec<u8>, v: i16| b.extend_from_slice(&v.to_be_bytes());
        put(&mut bytes, 0);
        put(&mut bytes, 7);
        put(&mut bytes, -100);
        put(&mut bytes, 8);  // lllen lower
        put(&mut bytes, 0);  // lllen upper
        put(&mut bytes, 0);
        put(&mut bytes, 0);
        // Data word: opcode 4, data 3 → 4*4096 + 3 = 16387
        put(&mut bytes, 16387);

        let out_bytes = decode_plio(&bytes, 5, 4, 32).unwrap();
        let out: Vec<i32> = out_bytes
            .chunks(4)
            .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        // First 3 pixels = pv = 1; rest = 0
        assert_eq!(out, vec![1, 1, 1, 0, 0]);
    }
}
