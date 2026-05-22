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
    if compressed.len() % 2 != 0 {
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
