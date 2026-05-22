// RICE_1 tile decompression for the FITS Tile Compression
// Convention.  Algorithm reference: cfitsio's `ricecomp.c` plus
// Pence et al. 2010 (FITS Tile Compression Convention).
//
// Stream layout for one tile of N pixels (BITS = BYTEPIX*8):
//   - First BITS bits: seed pixel value (literal, big-endian, two's
//     complement).
//   - Repeating blocks of up to BLOCKSIZE pixels:
//       - FSBITS bits: stored fs value; fs = stored - 1.
//       - For each pixel in block:
//           * fs == -1 (stored 0): low-entropy run.  Pixel value
//             equals the previous pixel; no further bits emitted.
//           * fs == FSMAX: raw encoding.  Read BITS bits, ZigZag-
//             decode to a signed diff, add to lastpix.
//           * 0 <= fs < FSMAX: Rice with parameter k=fs.  Read a
//             unary prefix (count of leading zeros up to and
//             including the terminating 1 bit), then fs low-order
//             bits.  Combine into ZigZag code, decode to signed
//             diff, add to lastpix.
//
// FSBITS / FSMAX per BYTEPIX (matches cfitsio):
//
//   BYTEPIX | BITS | FSBITS | FSMAX
//   --------+------+--------+------
//      1    |   8  |   3    |   6
//      2    |  16  |   4    |  14
//      4    |  32  |   5    |  25
//      8    |  64  |   6    |  53

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// (BITS, FSBITS, FSMAX) for the given BYTEPIX.
fn rice_params(bytepix: u32) -> PyResult<(u32, u32, u32)> {
    match bytepix {
        1 => Ok((8, 3, 6)),
        2 => Ok((16, 4, 14)),
        4 => Ok((32, 5, 25)),
        8 => Ok((64, 6, 53)),
        _ => Err(PyValueError::new_err(format!(
            "unsupported RICE BYTEPIX {} (must be 1, 2, 4, or 8)",
            bytepix
        ))),
    }
}

// MSB-first bit reader over a borrowed byte slice.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u32, // 0..7 — next-bit-to-read position within current byte
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, byte_pos: 0, bit_pos: 0 }
    }

    // Read up to 64 bits (MSB-first), return as u64.
    fn read_bits(&mut self, n: u32) -> PyResult<u64> {
        if n == 0 {
            return Ok(0);
        }
        if n > 64 {
            return Err(PyValueError::new_err(
                "RICE BitReader: requested > 64 bits at once"
            ));
        }
        let mut result: u64 = 0;
        let mut remaining = n;
        while remaining > 0 {
            if self.byte_pos >= self.data.len() {
                return Err(PyValueError::new_err(
                    "RICE decode: unexpected end of compressed stream"
                ));
            }
            let avail = 8 - self.bit_pos;
            let take = remaining.min(avail);
            let shift = avail - take;
            let mask = if take == 32 {
                u32::MAX
            } else {
                (1u32 << take) - 1
            };
            let bits = ((self.data[self.byte_pos] as u32 >> shift) & mask) as u64;
            result = (result << take) | bits;
            self.bit_pos += take;
            if self.bit_pos == 8 {
                self.byte_pos += 1;
                self.bit_pos = 0;
            }
            remaining -= take;
        }
        Ok(result)
    }

    // Read a unary code: count of leading 0 bits before the first
    // 1.  Returns the count of zeros; the terminating 1 bit is
    // consumed.  Caps at 1024 to avoid spinning on corrupt input.
    fn read_unary(&mut self) -> PyResult<u32> {
        let mut count: u32 = 0;
        loop {
            let b = self.read_bits(1)?;
            if b == 1 {
                return Ok(count);
            }
            count += 1;
            if count > 1024 {
                return Err(PyValueError::new_err(
                    "RICE decode: unary code exceeded 1024 zeros \
                     (corrupt stream)"
                ));
            }
        }
    }
}

// Reverse the ZigZag encoding used by the FITS Rice variant:
//   z = 0 → diff =  0
//   z = 1 → diff = -1
//   z = 2 → diff =  1
//   z = 3 → diff = -2
// In general: even z → diff = z/2; odd z → diff = -(z+1)/2.
fn unzigzag(z: u64) -> i64 {
    if z & 1 != 0 {
        -(((z >> 1) + 1) as i64)
    } else {
        (z >> 1) as i64
    }
}

// Sign-extend the low `nbits` bits of `val` to a 64-bit signed
// integer.  Used for the seed pixel and for raw-encoded diffs.
fn sign_extend(val: u64, nbits: u32) -> i64 {
    if nbits == 0 {
        return 0;
    }
    if nbits >= 64 {
        return val as i64;
    }
    let shift = 64 - nbits;
    ((val << shift) as i64) >> shift
}

// Decode one RICE_1-compressed tile to target-dtype bytes in
// numpy native byte order.  The raw decode produces `n_pixels`
// values widened to i64 in the order the encoder consumed them
// (row-major FITS order within the tile); we then cast each
// value down to the storage dtype (matching ZBITPIX) and write
// it out in native byte order.
//
// `zbitpix` must be one of 8/16/32/64 — float ZBITPIX is rejected
// upstream because the decompressor needs the quantization
// (ZSCALE/ZZERO) layer that Phase 5 will add.
pub(crate) fn decode_rice(
    compressed: &[u8],
    n_pixels: usize,
    bytepix: u32,
    blocksize: u32,
    zbitpix: i32,
) -> PyResult<Vec<u8>> {
    if n_pixels == 0 {
        return Ok(Vec::new());
    }
    if blocksize == 0 {
        return Err(PyValueError::new_err(
            "RICE decode: BLOCKSIZE must be > 0"
        ));
    }
    let (bbits, fsbits, fsmax) = rice_params(bytepix)?;

    let mut br = BitReader::new(compressed);

    // Seed: first pixel as `bbits` bits, sign-extended.  For
    // BYTEPIX=8 we read 64 bits which fills u64; sign_extend
    // returns it as-is via the early-out.
    let seed_raw = br.read_bits(bbits)?;
    let mut lastpix: i64 = sign_extend(seed_raw, bbits);

    let mut out: Vec<i64> = Vec::with_capacity(n_pixels);
    let mut i: usize = 0;
    while i < n_pixels {
        let stored_fs = br.read_bits(fsbits)? as i32;
        let fs: i32 = stored_fs - 1;
        let block_pixels = (blocksize as usize).min(n_pixels - i);

        if fs < 0 {
            // Low-entropy run — every pixel in this block equals
            // the previous decoded value.  No further bits.
            for _ in 0..block_pixels {
                out.push(lastpix);
            }
        } else if (fs as u32) >= fsmax {
            // Raw branch.  Each diff is `bbits` bits, ZigZag-
            // decoded, added to lastpix.  In practice this rarely
            // fires — only when the encoder gave up on Rice.
            for _ in 0..block_pixels {
                let raw = br.read_bits(bbits)?;
                let diff = unzigzag(raw);
                lastpix = lastpix.wrapping_add(diff);
                out.push(lastpix);
            }
        } else {
            // Standard Rice with parameter k = fs.  Unary high
            // bits + k low bits → ZigZag code → diff.
            let k = fs as u32;
            for _ in 0..block_pixels {
                let top = br.read_unary()? as u64;
                let bottom = if k > 0 { br.read_bits(k)? } else { 0 };
                let zz = (top << k) | bottom;
                let diff = unzigzag(zz);
                lastpix = lastpix.wrapping_add(diff);
                out.push(lastpix);
            }
        }
        i += block_pixels;
    }

    if out.len() != n_pixels {
        return Err(PyValueError::new_err(format!(
            "RICE decode: produced {} pixels but expected {}",
            out.len(), n_pixels
        )));
    }
    Ok(cast_i64_to_target_bytes(&out, zbitpix))
}

// Cast a Vec<i64> of decoded pixel values to bytes in the target
// (stored) dtype, numpy-native byte order.  ZBITPIX must be one
// of the supported integer values (8/16/32/64); float ZBITPIX is
// rejected upstream because the quantization layer (ZSCALE/ZZERO,
// Phase 5) hasn't landed yet.
fn cast_i64_to_target_bytes(values: &[i64], zbitpix: i32) -> Vec<u8> {
    match zbitpix {
        8 => {
            let mut out = Vec::with_capacity(values.len());
            for &v in values {
                out.push(v as u8);
            }
            out
        }
        16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for &v in values {
                out.extend_from_slice(&(v as i16).to_ne_bytes());
            }
            out
        }
        32 => {
            let mut out = Vec::with_capacity(values.len() * 4);
            for &v in values {
                out.extend_from_slice(&(v as i32).to_ne_bytes());
            }
            out
        }
        64 => {
            let mut out = Vec::with_capacity(values.len() * 8);
            for &v in values {
                out.extend_from_slice(&v.to_ne_bytes());
            }
            out
        }
        _ => Vec::new(), // unreachable: zbitpix validated upstream
    }
}
