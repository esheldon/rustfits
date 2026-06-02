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
use pyo3::exceptions::{PyNotImplementedError, PyValueError};

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
// numpy native byte order.  Hot path (BYTEPIX in {1, 2, 4}) uses a
// cfitsio-shaped 32-bit bit buffer + u32::leading_zeros for unary
// counting + direct index-write into a typed scratch.  BYTEPIX=8
// (no canonical writer produces it; we accept it on read for
// completeness) falls through to a slow general-purpose path
// because the 64-bit raw diffs don't fit in a 32-bit bit buffer.
//
// `zbitpix` is the final output bit-depth (8/16/32/64); float
// ZBITPIX is rejected upstream because the decompressor needs the
// quantization (ZSCALE/ZZERO) layer applied separately.
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
    if bytepix == 8 {
        return decode_rice_slow_i64(
            compressed, n_pixels, bbits, fsbits, fsmax, blocksize, zbitpix,
        );
    }
    // Allocate the target byte buffer up-front and decode straight
    // into it.  Dispatching on ZBITPIX once at the outer layer lets
    // the compiler monomorphize the inner loop's per-pixel store to
    // the exact target width — no Vec<i32> intermediate, no cast
    // pass over the data.
    let zbytes: usize = match zbitpix {
        8 | 16 | 32 | 64 => (zbitpix / 8) as usize,
        _ => return Err(PyValueError::new_err(format!(
            "RICE decode: unsupported ZBITPIX={}", zbitpix
        ))),
    };
    let mut out_bytes: Vec<u8> = vec![0u8; n_pixels * zbytes];
    match zbitpix {
        8 => decode_rice_int_u32buf::<I8Out>(
            compressed, n_pixels, bbits, fsbits, fsmax, blocksize, &mut out_bytes,
        )?,
        16 => decode_rice_int_u32buf::<I16Out>(
            compressed, n_pixels, bbits, fsbits, fsmax, blocksize, &mut out_bytes,
        )?,
        32 => decode_rice_int_u32buf::<I32Out>(
            compressed, n_pixels, bbits, fsbits, fsmax, blocksize, &mut out_bytes,
        )?,
        64 => decode_rice_int_u32buf::<I64Out>(
            compressed, n_pixels, bbits, fsbits, fsmax, blocksize, &mut out_bytes,
        )?,
        _ => unreachable!(),
    }
    Ok(out_bytes)
}

// Trait + zero-sized witness types: dispatch on ZBITPIX once at the
// outer layer, then specialize the decoder's inner store width via
// monomorphization.  `write_at` is `#[inline(always)]` so each impl
// reduces to a single store at the call site.
trait PixelWrite {
    const BYTES: usize;
    fn write_at(out: &mut [u8], i: usize, v: i32);
}

struct I8Out;
struct I16Out;
struct I32Out;
struct I64Out;

impl PixelWrite for I8Out {
    const BYTES: usize = 1;
    #[inline(always)]
    fn write_at(out: &mut [u8], i: usize, v: i32) {
        out[i] = v as u8;
    }
}
impl PixelWrite for I16Out {
    const BYTES: usize = 2;
    #[inline(always)]
    fn write_at(out: &mut [u8], i: usize, v: i32) {
        let bytes = (v as i16).to_ne_bytes();
        out[i * 2..i * 2 + 2].copy_from_slice(&bytes);
    }
}
impl PixelWrite for I32Out {
    const BYTES: usize = 4;
    #[inline(always)]
    fn write_at(out: &mut [u8], i: usize, v: i32) {
        let bytes = v.to_ne_bytes();
        out[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
}
impl PixelWrite for I64Out {
    const BYTES: usize = 8;
    #[inline(always)]
    fn write_at(out: &mut [u8], i: usize, v: i32) {
        let bytes = (v as i64).to_ne_bytes();
        out[i * 8..i * 8 + 8].copy_from_slice(&bytes);
    }
}

// Fast core: BYTEPIX in {1, 2, 4}.  Algorithm follows cfitsio's
// `fits_rdecomp`, but with a wider MSB-aligned u64 bit buffer that
// can refill 32 bits at a time (4x fewer refill ops than cfitsio's
// 8-bit refill of a u32 buffer).  Bit-buffer invariant: `buf` has
// `nbits` valid bits in positions 63..(64-nbits); positions below
// 64-nbits are zero.  This makes take_bits a single shift and
// makes unary counting a single `u64::leading_zeros` (LZCNT on
// x86-64-v2) across up to 64 bits in one shot.
fn decode_rice_int_u32buf<T: PixelWrite>(
    c: &[u8],
    nx: usize,
    bbits: u32,
    fsbits: u32,
    fsmax: u32,
    nblock: u32,
    out: &mut [u8],
) -> PyResult<()> {
    debug_assert!(out.len() == nx * T::BYTES);
    debug_assert!(bbits == 8 || bbits == 16 || bbits == 32);

    let seed_bytes = (bbits / 8) as usize;
    if c.len() < seed_bytes + 1 {
        return Err(PyValueError::new_err(
            "RICE decode: input shorter than seed + 1 byte"
        ));
    }

    // Seed: bbits-bit big-endian, sign-extended to i32.
    let mut seed_u: u32 = 0;
    for k in 0..seed_bytes {
        seed_u = (seed_u << 8) | (c[k] as u32);
    }
    let mut lastpix: i32 = if bbits < 32 {
        let shift = 32 - bbits;
        ((seed_u << shift) as i32) >> shift
    } else {
        seed_u as i32
    };

    // Bit buffer state.  We refill before reading whenever fewer
    // than `n` bits are buffered; the refill helper prefers 32-bit
    // reads when there's room (4 bytes available AND nbits <= 32)
    // and falls back to 8-bit reads near the stream tail.
    let mut buf: u64 = 0;
    let mut nbits: u32 = 0;
    let mut pos = seed_bytes;
    refill(&mut buf, &mut nbits, c, &mut pos);

    let mut i: usize = 0;
    while i < nx {
        // Need fsbits.  fsbits in {3, 4, 5} ≤ 32 so refill suffices.
        if nbits < fsbits {
            refill(&mut buf, &mut nbits, c, &mut pos);
            if nbits < fsbits {
                return Err(PyValueError::new_err(
                    "RICE decode: unexpected end of stream reading fs"
                ));
            }
        }
        let stored_fs: u32 = (buf >> (64 - fsbits)) as u32;
        buf <<= fsbits;
        nbits -= fsbits;
        let fs: i32 = stored_fs as i32 - 1;

        let imax = (i + nblock as usize).min(nx);

        if fs < 0 {
            // Low-entropy: every pixel in the block equals lastpix.
            for k in i..imax {
                T::write_at(out, k, lastpix);
            }
        } else if (fs as u32) == fsmax {
            // High-entropy: each diff is bbits raw bits.  bbits ≤
            // 32, so a single ensure-then-take suffices per pixel.
            for k in i..imax {
                if nbits < bbits {
                    refill(&mut buf, &mut nbits, c, &mut pos);
                    if nbits < bbits {
                        return Err(PyValueError::new_err(
                            "RICE decode: unexpected end of stream (raw)"
                        ));
                    }
                }
                let diff: u32 = (buf >> (64 - bbits)) as u32;
                buf <<= bbits;
                nbits -= bbits;
                let zz: i32 = if (diff & 1) == 0 {
                    (diff >> 1) as i32
                } else {
                    !(diff >> 1) as i32
                };
                lastpix = lastpix.wrapping_add(zz);
                T::write_at(out, k, lastpix);
            }
        } else {
            // Normal Rice with parameter k = fs.  Unary count of
            // leading zeros (up to 64 in one shot via leading_zeros
            // on the u64), then fs trailing bits.  This is the hot
            // path.  fs in 0..fsmax (fsmax ≤ 25), and nzero in
            // practice is small, so refill 32 at a time covers
            // every typical pixel with one bounds check.
            let fs_u = fs as u32;
            for k in i..imax {
                // Make sure buf has at least one 1 bit somewhere
                // (and enough bits beyond it for fs).  Worst-case
                // unary run for typical files is small; we cap
                // refills at 1024 zero bits to detect corrupt
                // streams.
                let mut nzero: u32 = 0;
                loop {
                    if buf == 0 {
                        nzero += nbits;
                        nbits = 0;
                        if nzero > 1024 {
                            return Err(PyValueError::new_err(
                                "RICE decode: unary code exceeded 1024 \
                                 zeros (corrupt stream)"
                            ));
                        }
                        refill(&mut buf, &mut nbits, c, &mut pos);
                        if nbits == 0 {
                            return Err(PyValueError::new_err(
                                "RICE decode: unexpected end of stream (unary)"
                            ));
                        }
                        continue;
                    }
                    let lz = buf.leading_zeros();
                    nzero += lz;
                    // Consume the lz zeros + the terminating 1 bit.
                    buf <<= lz + 1;
                    nbits -= lz + 1;
                    break;
                }

                // Read fs more trailing bits.
                if nbits < fs_u {
                    refill(&mut buf, &mut nbits, c, &mut pos);
                    if nbits < fs_u {
                        return Err(PyValueError::new_err(
                            "RICE decode: unexpected end of stream (rice tail)"
                        ));
                    }
                }
                let bottom: u32 = if fs_u == 0 {
                    0
                } else {
                    let v = (buf >> (64 - fs_u)) as u32;
                    buf <<= fs_u;
                    nbits -= fs_u;
                    v
                };
                let diff: u32 = (nzero << fs_u) | bottom;
                let zz: i32 = if (diff & 1) == 0 {
                    (diff >> 1) as i32
                } else {
                    !(diff >> 1) as i32
                };
                lastpix = lastpix.wrapping_add(zz);
                T::write_at(out, k, lastpix);
            }
        }
        i = imax;
    }

    Ok(())
}

// Refill the MSB-aligned bit buffer.  Reads 32 bits at a time when
// nbits <= 32 AND at least 4 bytes are available; falls back to
// 8 bits near the tail.  Stops when nbits > 56 (no room for another
// full byte) or when input is exhausted.
#[inline(always)]
fn refill(buf: &mut u64, nbits: &mut u32, c: &[u8], pos: &mut usize) {
    // Fast path: pull a 32-bit big-endian word when buffer has room
    // and input has enough bytes.  Most refills hit this branch.
    while *nbits <= 32 && *pos + 4 <= c.len() {
        let w = u32::from_be_bytes([
            c[*pos],
            c[*pos + 1],
            c[*pos + 2],
            c[*pos + 3],
        ]) as u64;
        *buf |= w << (32 - *nbits);
        *pos += 4;
        *nbits += 32;
    }
    // Tail: byte-at-a-time refill for the last few bytes.
    while *nbits <= 56 && *pos < c.len() {
        *buf |= (c[*pos] as u64) << (56 - *nbits);
        *pos += 1;
        *nbits += 8;
    }
}

// Slow path retained for BYTEPIX=8: 64-bit diffs don't fit in a
// 32-bit bit buffer.  Uses the generic BitReader.  Rarely exercised
// in practice — neither cfitsio nor fitsio nor astropy produce
// BYTEPIX=8 RICE files (rustfits's encoder also rejects it).
fn decode_rice_slow_i64(
    compressed: &[u8],
    n_pixels: usize,
    bbits: u32,
    fsbits: u32,
    fsmax: u32,
    blocksize: u32,
    zbitpix: i32,
) -> PyResult<Vec<u8>> {
    let mut br = BitReader::new(compressed);
    let seed_raw = br.read_bits(bbits)?;
    let mut lastpix: i64 = sign_extend(seed_raw, bbits);

    let mut out: Vec<i64> = Vec::with_capacity(n_pixels);
    let mut i: usize = 0;
    while i < n_pixels {
        let stored_fs = br.read_bits(fsbits)? as i32;
        let fs: i32 = stored_fs - 1;
        let block_pixels = (blocksize as usize).min(n_pixels - i);

        if fs < 0 {
            for _ in 0..block_pixels {
                out.push(lastpix);
            }
        } else if (fs as u32) >= fsmax {
            for _ in 0..block_pixels {
                let raw = br.read_bits(bbits)?;
                let diff = unzigzag(raw);
                lastpix = lastpix.wrapping_add(diff);
                out.push(lastpix);
            }
        } else {
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

// MSB-first bit writer for the RICE_1 encoder.  Cousin of
// `BitReader`: bits are written into a partial byte from the high
// end down, flushed whenever the byte fills.  Output is identical
// (byte-for-byte) to cfitsio's `output_nbits` + `done_outputing_bits`
// flush — the encoded stream is uniquely determined by the bit
// sequence, regardless of internal buffer layout.
struct BitWriter {
    out: Vec<u8>,
    cur_byte: u8,
    bits_used: u32, // 0..7 — bits written into cur_byte so far, MSB side
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { out: Vec::new(), cur_byte: 0, bits_used: 0 }
    }

    // Write the low `n` bits of `value` MSB-first.  Caller must
    // pass n <= 64; the high bits beyond n are masked off.
    fn write_bits(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 64);
        if n == 0 {
            return;
        }
        let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        let mut value = value & mask;
        let mut remaining = n;
        while remaining > 0 {
            let avail = 8 - self.bits_used;
            let take = remaining.min(avail);
            let shift = remaining - take;
            let mask_u8: u8 = if take == 8 { 0xFF } else { (1u8 << take) - 1 };
            let top_bits = ((value >> shift) as u8) & mask_u8;
            self.cur_byte |= top_bits << (avail - take);
            self.bits_used += take;
            if shift > 0 {
                value &= (1u64 << shift) - 1;
            }
            remaining -= take;
            if self.bits_used == 8 {
                self.out.push(self.cur_byte);
                self.cur_byte = 0;
                self.bits_used = 0;
            }
        }
    }

    // Advance the write position by `n` zero bits (no bit-setting
    // needed; cur_byte is already zero-initialised after each flush).
    fn write_zeros(&mut self, n: u32) {
        if n == 0 {
            return;
        }
        let mut remaining = n;
        while remaining > 0 {
            let avail = 8 - self.bits_used;
            let take = remaining.min(avail);
            self.bits_used += take;
            remaining -= take;
            if self.bits_used == 8 {
                self.out.push(self.cur_byte);
                self.cur_byte = 0;
                self.bits_used = 0;
            }
        }
    }

    // Unary code: `count` zero bits followed by a single `1` bit.
    fn write_unary(&mut self, count: u32) {
        self.write_zeros(count);
        self.write_bits(1, 1);
    }

    // Flush any partial byte and return the accumulated output.
    // Trailing bits within the final byte stay zero — matches
    // cfitsio's `done_outputing_bits` semantic.
    fn finish(mut self) -> Vec<u8> {
        if self.bits_used > 0 {
            self.out.push(self.cur_byte);
        }
        self.out
    }
}

// Encode one tile to RICE_1-compressed bytes.  Caller passes the
// tile's pixel bytes in FITS big-endian order (one packed
// integer per pixel, `bytepix` bytes wide).  Output bytes are
// byte-exact with cfitsio's `fits_rcomp` / `_short` / `_byte`
// encoders given the same input.
//
// Scope: BYTEPIX ∈ {1, 2, 4}.  Matches cfitsio's encoder family —
// there is no `fits_rcomp_longlong`, fitsio refuses i64 RICE
// outright, and astropy silently downcasts i64 to i32 before
// encoding.  Producing BYTEPIX=8 RICE files would make them
// unreadable by every canonical FITS tool; we reject upstream so
// users get a clear error pointing at GZIP_2 instead (which gives
// within ~5% of i64 RICE's hypothetical compression with no
// interop cost).
pub(crate) fn encode_rice(
    pixel_bytes_be: &[u8],
    n_pixels: usize,
    bytepix: u32,
    blocksize: u32,
) -> PyResult<Vec<u8>> {
    if n_pixels == 0 {
        return Ok(Vec::new());
    }
    if blocksize == 0 {
        return Err(PyValueError::new_err(
            "RICE encode: BLOCKSIZE must be > 0"
        ));
    }
    if bytepix == 8 {
        return Err(PyNotImplementedError::new_err(
            "RICE_1 does not support 64-bit pixels (BYTEPIX=8): no \
             canonical FITS writer (cfitsio, fitsio, astropy) produces \
             such files, so they would be unreadable outside rustfits. \
             Use GZIP_2 for i64 imaging data — typically within ~5% of \
             RICE compression and universally readable."
        ));
    }
    let (bbits, fsbits, fsmax) = rice_params(bytepix)?;
    debug_assert!(bbits <= 32);

    let expected_bytes = n_pixels.checked_mul(bytepix as usize)
        .ok_or_else(|| PyValueError::new_err(
            "RICE encode: n_pixels * bytepix overflowed usize"
        ))?;
    if pixel_bytes_be.len() != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "RICE encode: input length {} != n_pixels * bytepix ({})",
            pixel_bytes_be.len(), expected_bytes
        )));
    }

    // Read one pixel from the BE byte stream, sign-extended to i32.
    // cfitsio's encoder variants work in `int` (32-bit) precision
    // regardless of pixel width — the truncation happens at the
    // pdiff assignment back to the natural width.  We mirror that:
    // `lastpix`/`nextpix` stay in i32, and pdiff is truncated below.
    let read_pixel = |i: usize| -> i32 {
        let off = i * bytepix as usize;
        match bytepix {
            1 => pixel_bytes_be[off] as i8 as i32,
            2 => i16::from_be_bytes([
                pixel_bytes_be[off],
                pixel_bytes_be[off + 1],
            ]) as i32,
            4 => i32::from_be_bytes([
                pixel_bytes_be[off],
                pixel_bytes_be[off + 1],
                pixel_bytes_be[off + 2],
                pixel_bytes_be[off + 3],
            ]),
            _ => unreachable!(),
        }
    };

    let mut writer = BitWriter::new();

    // Seed: first pixel as `bbits` bits.  cfitsio's output_nbits
    // masks to n bits, so sign-extension above doesn't matter for
    // the seed's on-disk encoding.
    let seed = read_pixel(0);
    writer.write_bits(seed as u32 as u64, bbits);
    let mut lastpix = seed;

    let mut diff_buf: Vec<u32> = Vec::with_capacity(blocksize as usize);
    let mut i = 0;
    while i < n_pixels {
        let thisblock = (blocksize as usize).min(n_pixels - i);
        diff_buf.clear();
        let mut pixelsum: f64 = 0.0;

        for j in 0..thisblock {
            let nextpix = read_pixel(i + j);
            // pdiff = nextpix - lastpix truncated to the natural
            // bytepix width, then sign-extended back to i32.  For
            // bytepix=4 this is identity (the wrapping subtraction
            // already returns i32); for narrower widths the cast
            // chain truncates modulo 2^BITS and re-extends.
            let pdiff_raw = nextpix.wrapping_sub(lastpix);
            let pdiff: i32 = match bytepix {
                1 => pdiff_raw as i8 as i32,
                2 => pdiff_raw as i16 as i32,
                4 => pdiff_raw,
                _ => unreachable!(),
            };
            // ZigZag: maps signed → unsigned with negative-prefers-odd.
            // Equivalent to cfitsio's
            //   (pdiff<0) ? ~(pdiff<<1) : (pdiff<<1)
            // in int (32-bit) arithmetic — and inverse of our
            // decoder's `unzigzag`.
            let zz: u32 = (pdiff as u32).wrapping_shl(1)
                ^ ((pdiff >> 31) as u32);
            diff_buf.push(zz);
            pixelsum += zz as f64;
            lastpix = nextpix;
        }

        // Compute the Rice parameter `fs` matching cfitsio's
        // per-bytepix heuristic exactly.  The cast type of `psum`
        // (u8 / u16 / u32) caps it at the natural unsigned width
        // for the bytepix; small dpsum values then drive a small
        // fs, large ones top out at FSMAX.
        let dpsum_raw = (pixelsum
            - (thisblock / 2) as f64
            - 1.0)
            / thisblock as f64;
        let dpsum = if dpsum_raw < 0.0 { 0.0 } else { dpsum_raw };
        let mut psum: u32 = match bytepix {
            1 => (dpsum as u8 as u32) >> 1,
            2 => (dpsum as u16 as u32) >> 1,
            4 => (dpsum as u32) >> 1,
            _ => unreachable!(),
        };
        let mut fs: u32 = 0;
        while psum > 0 {
            fs += 1;
            psum >>= 1;
        }

        if fs >= fsmax {
            // High-entropy: write fsmax+1 (the "raw mode" marker
            // for this block), then each diff verbatim in bbits.
            writer.write_bits((fsmax + 1) as u64, fsbits);
            for &d in &diff_buf {
                writer.write_bits(d as u64, bbits);
            }
        } else if fs == 0 && pixelsum == 0.0 {
            // Low-entropy: every pixel in the block equals lastpix.
            // Stored as fs=0; no further bits.
            writer.write_bits(0, fsbits);
        } else {
            // Normal Rice with parameter k=fs.  Each diff splits
            // into top (unary-coded count of high bits) + bottom
            // (fs low bits raw).
            writer.write_bits((fs + 1) as u64, fsbits);
            let fsmask: u32 = if fs > 0 { (1u32 << fs) - 1 } else { 0 };
            for &d in &diff_buf {
                let top = d >> fs;
                writer.write_unary(top);
                if fs > 0 {
                    writer.write_bits((d & fsmask) as u64, fs);
                }
            }
        }

        i += thisblock;
    }

    Ok(writer.finish())
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
