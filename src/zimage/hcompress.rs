// clippy: the index-based loops in this file mirror the cfitsio C
// source line-for-line, keeping the port diffable against upstream;
// don't rewrite them into iterator form.
#![allow(clippy::needless_range_loop)]

// HCOMPRESS_1 tile decompression for the FITS Tile Compression
// Convention.  Port of cfitsio's `fits_hdecompress.c` (and its i64
// sibling `fits_hdecompress64`).  Function names mirror the C source
// (`decode`, `dodecode`, `qtree_decode`, `hinv`, `unshuffle`,
// `hsmooth`, `undigitize`, `qtree_expand`, `qtree_bitins`,
// `qtree_copy`, `input_bit`, `input_nbits`, `input_nybble`,
// `input_nnybble`, `input_huffman`) so diffing against
// /home/esheldon/git/fitsio/cfitsio-4.6.4/fits_hdecompress.c is
// straightforward when debugging.
//
// On-disk stream layout for one tile (per the FITS Tile Compression
// Convention, sec. 4.3 — HCOMPRESS_1):
//   - 2 bytes magic: 0xDD 0x99
//   - i32 BE: nx, ny, scale            (decoded image dims + SCALE)
//   - i64 BE: sumall                   (sum of all pixels)
//   - 3 bytes nbitplanes[3]            (bit planes per quadrant)
//   - quadtree-coded bit planes for the 4 quadrants, separated by
//     terminating-nybble check
//   - sign bits, one per non-zero pixel
//
// The decoder produces an integer coefficient array, multiplies by
// SCALE (undigitize), then runs the inverse H-transform to recover
// the original image.  `smooth` toggles a smoothing pass inside hinv
// that reduces visible block artifacts at high SCALE values; it's
// read from header keyword ZNAMEn='SMOOTH' / ZVALn=0|1 (cfitsio
// indexes this as ZVAL2 by convention).
//
// HCOMPRESS supports ZBITPIX = 8, 16, 32.  The 8/16 path runs the
// whole pipeline at i32 internal precision; the 32 path uses i64
// throughout because the H-transform's intermediate sums can
// overflow i32 for a 32-bit input.  ZBITPIX = 64 is NOT supported
// (cfitsio rejects it on the write side, and the spec doesn't
// define a 64-bit HCOMPRESS variant).

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// ===== Bit-level reader (stateful, mirrors cfitsio's `nextchar` +
// `buffer2`/`bits_to_go` globals).  Replaces them with a local
// struct so each decode call is self-contained — no FFLOCK needed,
// no global mutable state. =====

struct HState<'a> {
    bytes: &'a [u8],
    pos: usize,
    // The bit-input buffer.  cfitsio uses `int buffer2`; we use u32
    // for safe arithmetic without sign extension.
    buffer: u32,
    bits_to_go: u32,
}

impl<'a> HState<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        HState { bytes, pos: 0, buffer: 0, bits_to_go: 0 }
    }

    // qread: copy `n` bytes from the stream into `out`.  Bumps `pos`.
    fn qread(&mut self, out: &mut [u8]) -> PyResult<()> {
        let n = out.len();
        if self.pos + n > self.bytes.len() {
            return Err(PyValueError::new_err(
                "HCOMPRESS: unexpected end of compressed stream",
            ));
        }
        out.copy_from_slice(&self.bytes[self.pos..self.pos + n]);
        self.pos += n;
        Ok(())
    }

    // Read one big-endian i32 (cfitsio's readint).
    fn readint(&mut self) -> PyResult<i32> {
        let mut b = [0u8; 4];
        self.qread(&mut b)?;
        Ok(i32::from_be_bytes(b))
    }

    // Read one big-endian i64 (cfitsio's readlonglong).
    fn readlonglong(&mut self) -> PyResult<i64> {
        let mut b = [0u8; 8];
        self.qread(&mut b)?;
        Ok(i64::from_be_bytes(b))
    }

    // Reset the bit buffer (mirrors `start_inputing_bits`).
    fn start_inputing_bits(&mut self) {
        self.bits_to_go = 0;
        // Note: cfitsio doesn't reset buffer2 either — only
        // bits_to_go.  Leaving the stale bits in `buffer` is fine
        // since they're shifted out before the next read uses them.
    }

    // Read one bit (mirrors `input_bit`).
    fn input_bit(&mut self) -> PyResult<u32> {
        if self.bits_to_go == 0 {
            if self.pos >= self.bytes.len() {
                return Err(PyValueError::new_err(
                    "HCOMPRESS input_bit: stream exhausted",
                ));
            }
            self.buffer = self.bytes[self.pos] as u32;
            self.pos += 1;
            self.bits_to_go = 8;
        }
        self.bits_to_go -= 1;
        Ok((self.buffer >> self.bits_to_go) & 1)
    }

    // Read N bits (N <= 8) (mirrors `input_nbits`).  cfitsio's
    // version allows the buffer to grow past 8 bits temporarily.
    fn input_nbits(&mut self, n: u32) -> PyResult<u32> {
        debug_assert!(n <= 8);
        if self.bits_to_go < n {
            if self.pos >= self.bytes.len() {
                return Err(PyValueError::new_err(
                    "HCOMPRESS input_nbits: stream exhausted",
                ));
            }
            self.buffer = (self.buffer << 8) | (self.bytes[self.pos] as u32);
            self.pos += 1;
            self.bits_to_go += 8;
        }
        self.bits_to_go -= n;
        let mask = if n == 0 { 0 } else { (1u32 << n) - 1 };
        Ok((self.buffer >> self.bits_to_go) & mask)
    }

    // Read 4 bits (mirrors `input_nybble`).
    fn input_nybble(&mut self) -> PyResult<u32> {
        if self.bits_to_go < 4 {
            if self.pos >= self.bytes.len() {
                return Err(PyValueError::new_err(
                    "HCOMPRESS input_nybble: stream exhausted",
                ));
            }
            self.buffer = (self.buffer << 8) | (self.bytes[self.pos] as u32);
            self.pos += 1;
            self.bits_to_go += 8;
        }
        self.bits_to_go -= 4;
        Ok((self.buffer >> self.bits_to_go) & 0x0f)
    }

    // Read N nybbles into `array` (mirrors `input_nnybble`).
    // The cfitsio implementation is highly optimized for two cases
    // (bits_to_go == 0 vs general) — we faithfully reproduce both
    // since the optimizations preserve the exact bit-stream
    // semantics the encoder relies on.
    fn input_nnybble(
        &mut self, n: usize, array: &mut [u8],
    ) -> PyResult<u32> {
        if n == 1 {
            array[0] = self.input_nybble()? as u8;
            return Ok(0);
        }
        if self.bits_to_go == 8 {
            // Back up one byte; reuse it via the byte-aligned path.
            self.pos -= 1;
            self.bits_to_go = 0;
        }
        let shift1 = self.bits_to_go + 4;
        let shift2 = self.bits_to_go;
        let mut kk: usize = 0;
        let half = n / 2;
        if self.bits_to_go == 0 {
            for _ in 0..half {
                if self.pos >= self.bytes.len() {
                    return Err(PyValueError::new_err(
                        "HCOMPRESS input_nnybble: stream exhausted",
                    ));
                }
                self.buffer = (self.buffer << 8)
                    | (self.bytes[self.pos] as u32);
                self.pos += 1;
                array[kk] = ((self.buffer >> 4) & 0x0f) as u8;
                array[kk + 1] = (self.buffer & 0x0f) as u8;
                kk += 2;
            }
        } else {
            for _ in 0..half {
                if self.pos >= self.bytes.len() {
                    return Err(PyValueError::new_err(
                        "HCOMPRESS input_nnybble: stream exhausted",
                    ));
                }
                self.buffer = (self.buffer << 8)
                    | (self.bytes[self.pos] as u32);
                self.pos += 1;
                array[kk] = ((self.buffer >> shift1) & 0x0f) as u8;
                array[kk + 1] = ((self.buffer >> shift2) & 0x0f) as u8;
                kk += 2;
            }
        }
        if half * 2 != n {
            array[n - 1] = self.input_nybble()? as u8;
        }
        Ok((self.buffer >> self.bits_to_go) & 0x0f)
    }

    // Decode one Huffman-coded value in {0..15} (mirrors
    // `input_huffman`).  Code table:
    //   value: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    //   bits:  6  3  3  4  3  4  5  5  3  5  4  5  4  5  6  4
    //   hex:  3e 00 01 08 02 09 1a 1b 03 1c 0a 1d 0b 1e 3f 0c
    fn input_huffman(&mut self) -> PyResult<u32> {
        let mut c = self.input_nbits(3)?;
        if c < 4 {
            // 1, 2, 4, 8 for c = 0, 1, 2, 3
            return Ok(1u32 << c);
        }
        c = self.input_bit()? | (c << 1);
        if c < 13 {
            return Ok(match c {
                8 => 3, 9 => 5, 10 => 10, 11 => 12, 12 => 15,
                _ => unreachable!(),
            });
        }
        c = self.input_bit()? | (c << 1);
        if c < 31 {
            return Ok(match c {
                26 => 6, 27 => 7, 28 => 9, 29 => 11, 30 => 13,
                _ => unreachable!(),
            });
        }
        c = self.input_bit()? | (c << 1);
        if c == 62 { Ok(0) } else { Ok(14) }
    }
}

// ===== Magic =====

const CODE_MAGIC: [u8; 2] = [0xDD, 0x99];

// ===== qtree_copy (4-bit-to-2x2 expansion, shared between i32/i64) =====
//
// Mirrors cfitsio's `qtree_copy`.  Takes a (nx2 x ny2) array of
// 4-bit codes (one per 2x2 block) and expands to a (nx x ny)
// bitmap where each output element is 0 or 1, packed 4-per-nybble
// representing the 2x2 block:
//   bit3 → b[i,j]    bit2 → b[i,j+1]
//   bit1 → b[i+1,j]  bit0 → b[i+1,j+1]
//
// Same array may be passed for input and output — caller works
// back-to-front so the in-place case is safe.
fn qtree_copy(
    a: &mut [u8], nx: usize, ny: usize, n: usize,
) {
    let nx2 = nx.div_ceil(2);
    let ny2 = ny.div_ceil(2);
    // First copy 4-bit values from start-of-a (input) into b
    // positions [2*i, 2*j], walking back-to-front so we don't
    // overwrite source data.
    let mut k: isize = (ny2 as isize) * (nx2 as isize - 1) + ny2 as isize - 1;
    for i in (0..nx2 as isize).rev() {
        let mut s00: isize = 2 * (n as isize * i + ny2 as isize - 1);
        for _ in (0..ny2).rev() {
            a[s00 as usize] = a[k as usize];
            k -= 1;
            s00 -= 2;
        }
    }
    // Now expand each 2x2 block.  Walk forward.
    let nx_isize = nx as isize;
    let ny_isize = ny as isize;
    let n_isize = n as isize;
    let mut i: isize = 0;
    while i < nx_isize - 1 {
        let s00_base = n_isize * i;
        let s10_base = s00_base + n_isize;
        let mut j: isize = 0;
        let mut s00 = s00_base;
        let mut s10 = s10_base;
        while j < ny_isize - 1 {
            let code = a[s00 as usize];
            // bit3 → s00, bit2 → s00+1, bit1 → s10, bit0 → s10+1
            let b3 = (code >> 3) & 1;
            let b2 = (code >> 2) & 1;
            let b1 = (code >> 1) & 1;
            let b0 = code & 1;
            a[(s10 + 1) as usize] = b0;
            a[s10 as usize] = b1;
            a[(s00 + 1) as usize] = b2;
            a[s00 as usize] = b3;
            s00 += 2;
            s10 += 2;
            j += 2;
        }
        if j < ny_isize {
            // Odd row length: do last element, s00+1/s10+1 are off edge.
            let code = a[s00 as usize];
            a[s10 as usize] = (code >> 1) & 1;
            a[s00 as usize] = (code >> 3) & 1;
        }
        i += 2;
    }
    if i < nx_isize {
        // Odd column length: do last row, s10/s10+1 are off edge.
        let s00_base = n_isize * i;
        let mut j: isize = 0;
        let mut s00 = s00_base;
        while j < ny_isize - 1 {
            let code = a[s00 as usize];
            a[(s00 + 1) as usize] = (code >> 2) & 1;
            a[s00 as usize] = (code >> 3) & 1;
            s00 += 2;
            j += 2;
        }
        if j < ny_isize {
            let code = a[s00 as usize];
            a[s00 as usize] = (code >> 3) & 1;
        }
    }
}

// ===== qtree_expand (one quadtree expansion step) =====
//
// Mirror of cfitsio's `qtree_expand`.  Expands a[(nx+1)/2,(ny+1)/2]
// to b[nx,ny] (in-place when a==b), then reads a fresh Huffman code
// for each non-zero element to refine it from 1-bit to 4-bit.
fn qtree_expand(
    state: &mut HState<'_>, a: &mut [u8], nx: usize, ny: usize,
) -> PyResult<()> {
    qtree_copy(a, nx, ny, ny);
    // Now read new 4-bit values into b for each non-zero element,
    // walking back-to-front (cfitsio does the same — order matters
    // because the qtree_copy left the 2x2 blocks in a specific
    // layout that the reverse walk consumes).
    for i in (0..nx * ny).rev() {
        if a[i] != 0 {
            a[i] = state.input_huffman()? as u8;
        }
    }
    Ok(())
}

// ===== Generic helpers for qtree_bitins / dodecode i32/i64 =====
//
// The bit-insertion code is identical for i32 and i64 modulo the
// element type.  We use a tiny trait so the long switch-table can
// be written once.

trait HBit: Copy + std::ops::BitOrAssign + Default {
    fn shl_bit(bit: u32) -> Self;
}

impl HBit for i32 {
    fn shl_bit(bit: u32) -> Self { 1i32 << bit }
}

impl HBit for i64 {
    fn shl_bit(bit: u32) -> Self { 1i64 << bit }
}

// Mirror of cfitsio's `qtree_bitins` / `qtree_bitins64`.
// Copies 4-bit values from a[(nx+1)/2,(ny+1)/2] into bitplane `bit`
// of b[nx,ny], expanding each value to a 2x2 block.  a and b MUST
// NOT alias.  `n` is the row stride of b (declared y dimension).
//
// The cfitsio source uses a 16-arm switch on the 4-bit value; we
// compute the four bits directly which the optimizer should fold
// into the same machine code with much less source duplication.
fn qtree_bitins<T: HBit>(
    a: &[u8], nx: usize, ny: usize, b: &mut [T], n: usize, bit: u32,
) {
    let plane_val = T::shl_bit(bit);
    let mut k: usize = 0;
    let nx_isize = nx as isize;
    let ny_isize = ny as isize;
    let n_isize = n as isize;
    let mut i: isize = 0;
    while i < nx_isize - 1 {
        let s00_base = n_isize * i;
        let mut s00 = s00_base;
        let mut j: isize = 0;
        while j < ny_isize - 1 {
            let code = a[k];
            // Same bit assignment as qtree_copy:
            //   bit3 → s00,     bit2 → s00+1
            //   bit1 → s00+n,   bit0 → s00+n+1
            if code & 0b1000 != 0 { b[s00 as usize] |= plane_val; }
            if code & 0b0100 != 0 { b[(s00 + 1) as usize] |= plane_val; }
            if code & 0b0010 != 0 { b[(s00 + n_isize) as usize] |= plane_val; }
            if code & 0b0001 != 0 {
                b[(s00 + n_isize + 1) as usize] |= plane_val;
            }
            s00 += 2;
            k += 1;
            j += 2;
        }
        if j < ny_isize {
            // Odd row length: skip the +1 columns (bit0, bit2).
            let code = a[k];
            if code & 0b1000 != 0 { b[s00 as usize] |= plane_val; }
            if code & 0b0010 != 0 { b[(s00 + n_isize) as usize] |= plane_val; }
            k += 1;
        }
        i += 2;
    }
    if i < nx_isize {
        // Odd column length: skip the +n rows (bit0, bit1).
        let s00_base = n_isize * i;
        let mut s00 = s00_base;
        let mut j: isize = 0;
        while j < ny_isize - 1 {
            let code = a[k];
            if code & 0b1000 != 0 { b[s00 as usize] |= plane_val; }
            if code & 0b0100 != 0 { b[(s00 + 1) as usize] |= plane_val; }
            s00 += 2;
            k += 1;
            j += 2;
        }
        if j < ny_isize {
            // Corner element: only bit3.
            let code = a[k];
            if code & 0b1000 != 0 { b[s00 as usize] |= plane_val; }
            k += 1;
        }
    }
    let _ = k;
}

// ===== read_bdirect (read bit image packed 4 pixels/nybble) =====
//
// Mirror of cfitsio's `read_bdirect`.  Used when a bit plane was
// written directly (no quadtree encoding).
fn read_bdirect<T: HBit>(
    state: &mut HState<'_>, a: &mut [T], n: usize,
    nqx: usize, nqy: usize, scratch: &mut [u8], bit: u32,
) -> PyResult<()> {
    let count = nqx.div_ceil(2) * nqy.div_ceil(2);
    state.input_nnybble(count, &mut scratch[..count])?;
    qtree_bitins(&scratch[..count], nqx, nqy, a, n, bit);
    Ok(())
}

// ===== qtree_decode (one bit plane of one quadrant) =====
//
// Mirror of cfitsio's `qtree_decode` / `qtree_decode64`.  Decodes
// `nbitplanes` planes from the stream into the quadrant starting
// at `a[0..]`, of partial dims (nqx, nqy) within a parent array of
// row stride `n`.  Caller passes `&mut a[base..]` to position the
// quadrant.
fn qtree_decode<T: HBit>(
    state: &mut HState<'_>, a: &mut [T], n: usize,
    nqx: usize, nqy: usize, nbitplanes: u32,
) -> PyResult<()> {
    let nqmax = nqx.max(nqy);
    // log2n = ceil(log2(nqmax)).  Match cfitsio's `log/log + 0.5`
    // formula exactly so edge cases agree.
    let mut log2n = (((nqmax as f64).ln() / std::f64::consts::LN_2) + 0.5)
        as u32;
    if nqmax > (1usize << log2n) {
        log2n += 1;
    }
    let nqx2 = nqx.div_ceil(2);
    let nqy2 = nqy.div_ceil(2);
    let mut scratch = vec![0u8; nqx2 * nqy2];

    for bit_idx in (0..nbitplanes).rev() {
        let b = state.input_nybble()?;
        if b == 0 {
            // Bit map written directly.
            read_bdirect(
                state, a, n, nqx, nqy, &mut scratch, bit_idx,
            )?;
        } else if b != 0x0f {
            return Err(PyValueError::new_err(
                "HCOMPRESS qtree_decode: bad format code",
            ));
        } else {
            // Quadtree-coded.  Read first code, then log2n-1
            // expansions.
            scratch[0] = state.input_huffman()? as u8;
            let mut nx: usize = 1;
            let mut ny: usize = 1;
            let mut nfx = nqx;
            let mut nfy = nqy;
            let mut c: usize = 1 << log2n;
            for _k in 1..log2n {
                c >>= 1;
                nx <<= 1;
                ny <<= 1;
                if nfx <= c { nx -= 1; } else { nfx -= c; }
                if nfy <= c { ny -= 1; } else { nfy -= c; }
                qtree_expand(state, &mut scratch, nx, ny)?;
            }
            qtree_bitins(&scratch, nqx, nqy, a, n, bit_idx);
        }
    }
    Ok(())
}

// ===== dodecode (decode the 4 quadrants) =====
//
// Mirror of cfitsio's `dodecode` / `dodecode64`.  Reads 4 quadrant
// bit-plane streams, EOF nybble, then sign bits.
fn dodecode<T: HBit + std::ops::Neg<Output = T> + std::cmp::PartialEq>(
    state: &mut HState<'_>, a: &mut [T], nx: usize, ny: usize,
    nbitplanes: [u8; 3],
) -> PyResult<()> {
    let nel = nx * ny;
    let nx2 = nx.div_ceil(2);
    let ny2 = ny.div_ceil(2);
    // Caller must have zero-filled a.
    for v in a.iter_mut() { *v = T::default(); }

    state.start_inputing_bits();

    // Quadrant 0: a[0..], partial dims (nx2, ny2), plane count nbitplanes[0]
    qtree_decode(state, a, ny, nx2, ny2, nbitplanes[0] as u32)?;
    // Quadrant 1: a[ny2..], partial dims (nx2, ny/2), plane count nbitplanes[1]
    qtree_decode(state, &mut a[ny2..], ny, nx2, ny / 2, nbitplanes[1] as u32)?;
    // Quadrant 2: a[ny*nx2..], partial dims (nx/2, ny2), plane count nbitplanes[1]
    qtree_decode(
        state, &mut a[ny * nx2..], ny, nx / 2, ny2, nbitplanes[1] as u32,
    )?;
    // Quadrant 3: a[ny*nx2+ny2..], partial dims (nx/2, ny/2), plane count nbitplanes[2]
    qtree_decode(
        state, &mut a[ny * nx2 + ny2..],
        ny, nx / 2, ny / 2, nbitplanes[2] as u32,
    )?;

    // EOF nybble must be zero.
    if state.input_nybble()? != 0 {
        return Err(PyValueError::new_err(
            "HCOMPRESS dodecode: bad bit plane values (missing EOF nybble)",
        ));
    }

    // Sign bits.
    state.start_inputing_bits();
    for v in a.iter_mut() {
        if *v != T::default()
            && state.input_bit()? != 0 {
                *v = -*v;
            }
    }
    let _ = nel;
    Ok(())
}

// ===== decode i32 / i64 (parse top-of-stream + call dodecode) =====

fn decode_i32(
    state: &mut HState<'_>, a: &mut [i32],
) -> PyResult<(usize, usize, i32)> {
    let mut tmagic = [0u8; 2];
    state.qread(&mut tmagic)?;
    if tmagic != CODE_MAGIC {
        return Err(PyValueError::new_err(
            "HCOMPRESS: bad magic in tile stream",
        ));
    }
    let nx = state.readint()?;
    let ny = state.readint()?;
    let scale = state.readint()?;
    if nx <= 0 || ny <= 0 {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS: invalid dims nx={}, ny={}", nx, ny
        )));
    }
    let sumall = state.readlonglong()?;
    let mut nbitplanes = [0u8; 3];
    state.qread(&mut nbitplanes)?;
    let nx_us = nx as usize;
    let ny_us = ny as usize;
    if a.len() != nx_us * ny_us {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS: tile dims {}x{} disagree with allocated buffer {}",
            nx, ny, a.len()
        )));
    }
    dodecode::<i32>(state, a, nx_us, ny_us, nbitplanes)?;
    a[0] = sumall as i32;
    Ok((nx_us, ny_us, scale))
}

fn decode_i64(
    state: &mut HState<'_>, a: &mut [i64],
) -> PyResult<(usize, usize, i32)> {
    let mut tmagic = [0u8; 2];
    state.qread(&mut tmagic)?;
    if tmagic != CODE_MAGIC {
        return Err(PyValueError::new_err(
            "HCOMPRESS: bad magic in tile stream",
        ));
    }
    let nx = state.readint()?;
    let ny = state.readint()?;
    let scale = state.readint()?;
    if nx <= 0 || ny <= 0 {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS: invalid dims nx={}, ny={}", nx, ny
        )));
    }
    let sumall = state.readlonglong()?;
    let mut nbitplanes = [0u8; 3];
    state.qread(&mut nbitplanes)?;
    let nx_us = nx as usize;
    let ny_us = ny as usize;
    if a.len() != nx_us * ny_us {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS: tile dims {}x{} disagree with allocated buffer {}",
            nx, ny, a.len()
        )));
    }
    dodecode::<i64>(state, a, nx_us, ny_us, nbitplanes)?;
    a[0] = sumall;
    Ok((nx_us, ny_us, scale))
}

// ===== undigitize (multiply by scale) =====

fn undigitize_i32(a: &mut [i32], scale: i32) {
    if scale <= 1 { return; }
    for v in a.iter_mut() { *v = v.wrapping_mul(scale); }
}

fn undigitize_i64(a: &mut [i64], scale: i32) {
    if scale <= 1 { return; }
    let s = scale as i64;
    for v in a.iter_mut() { *v = v.wrapping_mul(s); }
}

// ===== unshuffle (interleave two halves of an array along an axis) =====
//
// Mirror of cfitsio's `unshuffle`.  Stride is `n2` (1 for rows,
// row-stride for columns).  `n` is the length along the unshuffled
// dimension.  `tmp` is scratch, at least ceil(n/2) elements.
fn unshuffle_i32(a: &mut [i32], n: usize, n2: usize, tmp: &mut [i32]) {
    let nhalf = (n + 1) >> 1;
    // Copy 2nd half of array to tmp.
    for i in nhalf..n {
        tmp[i - nhalf] = a[n2 * i];
    }
    // Distribute 1st half to even elements (back-to-front, in-place).
    for i in (0..nhalf).rev() {
        let src = n2 * i;
        let dst = src << 1;
        a[dst] = a[src];
    }
    // Distribute 2nd half (from tmp) to odd elements.
    let n_odd = n / 2;
    for k in 0..n_odd {
        a[n2 + 2 * n2 * k] = tmp[k];
    }
}

fn unshuffle_i64(a: &mut [i64], n: usize, n2: usize, tmp: &mut [i64]) {
    let nhalf = (n + 1) >> 1;
    for i in nhalf..n {
        tmp[i - nhalf] = a[n2 * i];
    }
    for i in (0..nhalf).rev() {
        let src = n2 * i;
        let dst = src << 1;
        a[dst] = a[src];
    }
    let n_odd = n / 2;
    for k in 0..n_odd {
        a[n2 + 2 * n2 * k] = tmp[k];
    }
}

// ===== hsmooth =====
//
// Mirror of cfitsio's `hsmooth` / `hsmooth64`.  Smooth-by-
// interpolation pass that runs inside hinv when SMOOTH=1.  Only
// active when scale > 1; for scale=0/1 the smoothing math collapses
// to a no-op anyway, so we skip the call from hinv.
// Smoothing pass applied inside the inverse H-transform when
// SMOOTH=1.  Direct port of cfitsio's `hsmooth` / `hsmooth64`.
// Walks the (nxtop, nytop) coefficient block in 2x2 chunks and
// adjusts the hx / hy / hc differences toward their interpolated
// values, subject to monotonicity constraints and an overall
// change cap of ±(scale/2).  Edge coefficients are NOT touched —
// the loops start at 2 and stop at nxtop-2 / nytop-2 by design.
// scale <= 1 → no-op (smax = 0).
//
// Division semantics: cfitsio uses
//   s = (s>=0) ? (s>>n) : ((s+(2^n-1))>>n)
// which is signed division truncating toward zero.  Rust's `/`
// operator on signed ints already truncates toward zero, so we
// translate the shift-with-bias dance directly into `s / 8` and
// `s / 64` for readability.
fn hsmooth_i32(
    a: &mut [i32], nxtop: usize, nytop: usize, ny: usize, scale: i32,
) {
    let smax = scale >> 1;
    if smax <= 0 { return; }
    let ny2 = ny << 1;

    // Adjust x difference hx: for i in [2, nxtop-2) step 2, j in [0, nytop) step 2.
    if nxtop >= 4 {
        let mut i = 2usize;
        while i + 2 < nxtop {
            let row = ny * i;
            let mut j = 0usize;
            while j < nytop {
                let s00 = row + j;
                let s10 = s00 + ny;
                let hm = a[s00 - ny2];
                let h0 = a[s00];
                let hp = a[s00 + ny2];
                let mut diff = hp - hm;
                let dmax = (hp - h0).min(h0 - hm).max(0) << 2;
                let dmin = (hp - h0).max(h0 - hm).min(0) << 2;
                if dmin < dmax {
                    diff = diff.min(dmax).max(dmin);
                    let s = (diff - (a[s10] << 3)) / 8;
                    let s = s.min(smax).max(-smax);
                    a[s10] += s;
                }
                j += 2;
            }
            i += 2;
        }
    }

    // Adjust y difference hy: for i in [0, nxtop) step 2, j in [2, nytop-2) step 2.
    if nytop >= 4 {
        let mut i = 0usize;
        while i < nxtop {
            let row = ny * i;
            let mut j = 2usize;
            while j + 2 < nytop {
                let s00 = row + j;
                let hm = a[s00 - 2];
                let h0 = a[s00];
                let hp = a[s00 + 2];
                let mut diff = hp - hm;
                let dmax = (hp - h0).min(h0 - hm).max(0) << 2;
                let dmin = (hp - h0).max(h0 - hm).min(0) << 2;
                if dmin < dmax {
                    diff = diff.min(dmax).max(dmin);
                    let s = (diff - (a[s00 + 1] << 3)) / 8;
                    let s = s.min(smax).max(-smax);
                    a[s00 + 1] += s;
                }
                j += 2;
            }
            i += 2;
        }
    }

    // Adjust curvature difference hc: for i in [2, nxtop-2) step 2, j in [2, nytop-2) step 2.
    if nxtop >= 4 && nytop >= 4 {
        let mut i = 2usize;
        while i + 2 < nxtop {
            let row = ny * i;
            let mut j = 2usize;
            while j + 2 < nytop {
                let s00 = row + j;
                let s10 = s00 + ny;
                let hmm = a[s00 - ny2 - 2];
                let hpm = a[s00 + ny2 - 2];
                let hmp = a[s00 - ny2 + 2];
                let hpp = a[s00 + ny2 + 2];
                let h0  = a[s00];
                let mut diff = hpp + hmm - hmp - hpm;
                let hx2 = a[s10]     << 1;
                let hy2 = a[s00 + 1] << 1;
                let m1 = ((hpp - h0).max(0) - hx2 - hy2)
                    .min((h0 - hpm).max(0) + hx2 - hy2);
                let m2 = ((h0 - hmp).max(0) - hx2 + hy2)
                    .min((hmm - h0).max(0) + hx2 + hy2);
                let dmax = m1.min(m2) << 4;
                let m1 = ((hpp - h0).min(0) - hx2 - hy2)
                    .max((h0 - hpm).min(0) + hx2 - hy2);
                let m2 = ((h0 - hmp).min(0) - hx2 + hy2)
                    .max((hmm - h0).min(0) + hx2 + hy2);
                let dmin = m1.max(m2) << 4;
                if dmin < dmax {
                    diff = diff.min(dmax).max(dmin);
                    let s = (diff - (a[s10 + 1] << 6)) / 64;
                    let s = s.min(smax).max(-smax);
                    a[s10 + 1] += s;
                }
                j += 2;
            }
            i += 2;
        }
    }
}

fn hsmooth_i64(
    a: &mut [i64], nxtop: usize, nytop: usize, ny: usize, scale: i32,
) {
    let smax = (scale >> 1) as i64;
    if smax <= 0 { return; }
    let ny2 = ny << 1;

    if nxtop >= 4 {
        let mut i = 2usize;
        while i + 2 < nxtop {
            let row = ny * i;
            let mut j = 0usize;
            while j < nytop {
                let s00 = row + j;
                let s10 = s00 + ny;
                let hm = a[s00 - ny2];
                let h0 = a[s00];
                let hp = a[s00 + ny2];
                let mut diff = hp - hm;
                let dmax = (hp - h0).min(h0 - hm).max(0) << 2;
                let dmin = (hp - h0).max(h0 - hm).min(0) << 2;
                if dmin < dmax {
                    diff = diff.min(dmax).max(dmin);
                    let s = (diff - (a[s10] << 3)) / 8;
                    let s = s.min(smax).max(-smax);
                    a[s10] += s;
                }
                j += 2;
            }
            i += 2;
        }
    }

    if nytop >= 4 {
        let mut i = 0usize;
        while i < nxtop {
            let row = ny * i;
            let mut j = 2usize;
            while j + 2 < nytop {
                let s00 = row + j;
                let hm = a[s00 - 2];
                let h0 = a[s00];
                let hp = a[s00 + 2];
                let mut diff = hp - hm;
                let dmax = (hp - h0).min(h0 - hm).max(0) << 2;
                let dmin = (hp - h0).max(h0 - hm).min(0) << 2;
                if dmin < dmax {
                    diff = diff.min(dmax).max(dmin);
                    let s = (diff - (a[s00 + 1] << 3)) / 8;
                    let s = s.min(smax).max(-smax);
                    a[s00 + 1] += s;
                }
                j += 2;
            }
            i += 2;
        }
    }

    if nxtop >= 4 && nytop >= 4 {
        let mut i = 2usize;
        while i + 2 < nxtop {
            let row = ny * i;
            let mut j = 2usize;
            while j + 2 < nytop {
                let s00 = row + j;
                let s10 = s00 + ny;
                let hmm = a[s00 - ny2 - 2];
                let hpm = a[s00 + ny2 - 2];
                let hmp = a[s00 - ny2 + 2];
                let hpp = a[s00 + ny2 + 2];
                let h0  = a[s00];
                let mut diff = hpp + hmm - hmp - hpm;
                let hx2 = a[s10]     << 1;
                let hy2 = a[s00 + 1] << 1;
                let m1 = ((hpp - h0).max(0) - hx2 - hy2)
                    .min((h0 - hpm).max(0) + hx2 - hy2);
                let m2 = ((h0 - hmp).max(0) - hx2 + hy2)
                    .min((hmm - h0).max(0) + hx2 + hy2);
                let dmax = m1.min(m2) << 4;
                let m1 = ((hpp - h0).min(0) - hx2 - hy2)
                    .max((h0 - hpm).min(0) + hx2 - hy2);
                let m2 = ((h0 - hmp).min(0) - hx2 + hy2)
                    .max((hmm - h0).min(0) + hx2 + hy2);
                let dmin = m1.max(m2) << 4;
                if dmin < dmax {
                    diff = diff.min(dmax).max(dmin);
                    let s = (diff - (a[s10 + 1] << 6)) / 64;
                    let s = s.min(smax).max(-smax);
                    a[s10 + 1] += s;
                }
                j += 2;
            }
            i += 2;
        }
    }
}

// ===== hinv (inverse H-transform) =====
//
// Mirror of cfitsio's `hinv` / `hinv64`.  Walks log2n expansion
// passes from coarsest to finest, unshuffling along each axis and
// then applying the 4-coefficient combine.
fn hinv_i32(
    a: &mut [i32], nx: usize, ny: usize, smooth: bool, scale: i32,
) -> PyResult<()> {
    let nmax = nx.max(ny);
    let mut log2n = (((nmax as f64).ln() / std::f64::consts::LN_2) + 0.5)
        as u32;
    if nmax > (1usize << log2n) {
        log2n += 1;
    }
    let scratch_len = nmax.div_ceil(2);
    let mut tmp = vec![0i32; scratch_len];

    let mut shift: u32 = 1;
    let mut bit0: i32 = 1 << (log2n - 1);
    let mut bit1: i32 = bit0 << 1;
    let bit2: i32 = bit0 << 2;
    let mut mask0: i32 = -bit0;
    let mut mask1: i32 = mask0 << 1;
    let mask2: i32 = mask0 << 2;
    let mut prnd0: i32 = bit0 >> 1;
    let mut prnd1: i32 = bit1 >> 1;
    let prnd2: i32 = bit2 >> 1;
    let mut nrnd0: i32 = prnd0 - 1;
    let mut nrnd1: i32 = prnd1 - 1;
    let nrnd2: i32 = prnd2 - 1;

    // Round h0 to multiple of bit2.
    let a0 = a[0];
    a[0] = (a0 + if a0 >= 0 { prnd2 } else { nrnd2 }) & mask2;

    let mut nxtop: usize = 1;
    let mut nytop: usize = 1;
    let mut nxf = nx;
    let mut nyf = ny;
    let mut c: usize = 1 << log2n;

    for k in (0..log2n).rev() {
        c >>= 1;
        nxtop <<= 1;
        nytop <<= 1;
        if nxf <= c { nxtop -= 1; } else { nxf -= c; }
        if nyf <= c { nytop -= 1; } else { nyf -= c; }
        if k == 0 {
            nrnd0 = 0;
            shift = 2;
        }
        // Unshuffle each row of the active region.
        for i in 0..nxtop {
            let row_start = ny * i;
            unshuffle_i32(&mut a[row_start..], nytop, 1, &mut tmp);
        }
        // Unshuffle each column of the active region.
        for j in 0..nytop {
            unshuffle_i32(&mut a[j..], nxtop, ny, &mut tmp);
        }

        if smooth { hsmooth_i32(a, nxtop, nytop, ny, scale); }

        let oddx = nxtop & 1;
        let oddy = nytop & 1;

        let mut i: usize = 0;
        while i + 1 < nxtop {
            let s00_row = ny * i;
            let s10_row = s00_row + ny;
            let mut j: usize = 0;
            while j + 1 < nytop {
                let s00 = s00_row + j;
                let s10 = s10_row + j;
                let mut h0 = a[s00];
                let mut hx = a[s10];
                let mut hy = a[s00 + 1];
                let mut hc = a[s10 + 1];
                hx = (hx + if hx >= 0 { prnd1 } else { nrnd1 }) & mask1;
                hy = (hy + if hy >= 0 { prnd1 } else { nrnd1 }) & mask1;
                hc = (hc + if hc >= 0 { prnd0 } else { nrnd0 }) & mask0;
                let lowbit0 = hc & bit0;
                hx = if hx >= 0 { hx - lowbit0 } else { hx + lowbit0 };
                hy = if hy >= 0 { hy - lowbit0 } else { hy + lowbit0 };
                let lowbit1 = (hc ^ hx ^ hy) & bit1;
                h0 = if h0 >= 0 {
                    h0 + lowbit0 - lowbit1
                } else if lowbit0 == 0 {
                    h0 + lowbit1
                } else {
                    h0 + (lowbit0 - lowbit1)
                };
                a[s10 + 1] = (h0 + hx + hy + hc) >> shift;
                a[s10]     = (h0 + hx - hy - hc) >> shift;
                a[s00 + 1] = (h0 - hx + hy - hc) >> shift;
                a[s00]     = (h0 - hx - hy + hc) >> shift;
                j += 2;
            }
            if oddy == 1 {
                // Last element in row; s00+1, s10+1 off edge.
                let s00 = s00_row + j;
                let s10 = s10_row + j;
                let mut h0 = a[s00];
                let mut hx = a[s10];
                hx = (hx + if hx >= 0 { prnd1 } else { nrnd1 }) & mask1;
                let lowbit1 = hx & bit1;
                h0 = if h0 >= 0 { h0 - lowbit1 } else { h0 + lowbit1 };
                a[s10] = (h0 + hx) >> shift;
                a[s00] = (h0 - hx) >> shift;
            }
            i += 2;
        }
        if oddx == 1 {
            // Last row; s10, s10+1 off edge.
            let s00_row = ny * i;
            let mut j: usize = 0;
            while j + 1 < nytop {
                let s00 = s00_row + j;
                let mut h0 = a[s00];
                let mut hy = a[s00 + 1];
                hy = (hy + if hy >= 0 { prnd1 } else { nrnd1 }) & mask1;
                let lowbit1 = hy & bit1;
                h0 = if h0 >= 0 { h0 - lowbit1 } else { h0 + lowbit1 };
                a[s00 + 1] = (h0 + hy) >> shift;
                a[s00]     = (h0 - hy) >> shift;
                j += 2;
            }
            if oddy == 1 {
                // Corner; s00+1, s10, s10+1 off edge.
                let s00 = s00_row + j;
                let h0 = a[s00];
                a[s00] = h0 >> shift;
            }
        }

        // Divide masks/rounding by 2 for next iteration.  bit2 /
        // mask2 / prnd2 / nrnd2 aren't reused after the next pass
        // either, so we don't propagate them; cfitsio does, but it
        // costs a write each iteration and the values are dead.
        bit1 = bit0;
        bit0 >>= 1;
        mask1 = mask0;
        mask0 >>= 1;
        prnd1 = prnd0;
        prnd0 >>= 1;
        nrnd1 = nrnd0;
        nrnd0 = prnd0 - 1;
    }
    let _ = (bit2, mask2, prnd2, nrnd2);
    Ok(())
}

fn hinv_i64(
    a: &mut [i64], nx: usize, ny: usize, smooth: bool, scale: i32,
) -> PyResult<()> {
    let nmax = nx.max(ny);
    let mut log2n = (((nmax as f64).ln() / std::f64::consts::LN_2) + 0.5)
        as u32;
    if nmax > (1usize << log2n) {
        log2n += 1;
    }
    let scratch_len = nmax.div_ceil(2);
    let mut tmp = vec![0i64; scratch_len];

    let mut shift: u32 = 1;
    let mut bit0: i64 = 1i64 << (log2n - 1);
    let mut bit1: i64 = bit0 << 1;
    let bit2: i64 = bit0 << 2;
    let mut mask0: i64 = -bit0;
    let mut mask1: i64 = mask0 << 1;
    let mask2: i64 = mask0 << 2;
    let mut prnd0: i64 = bit0 >> 1;
    let mut prnd1: i64 = bit1 >> 1;
    let prnd2: i64 = bit2 >> 1;
    let mut nrnd0: i64 = prnd0 - 1;
    let mut nrnd1: i64 = prnd1 - 1;
    let nrnd2: i64 = prnd2 - 1;

    let a0 = a[0];
    a[0] = (a0 + if a0 >= 0 { prnd2 } else { nrnd2 }) & mask2;

    let mut nxtop: usize = 1;
    let mut nytop: usize = 1;
    let mut nxf = nx;
    let mut nyf = ny;
    let mut c: usize = 1 << log2n;

    for k in (0..log2n).rev() {
        c >>= 1;
        nxtop <<= 1;
        nytop <<= 1;
        if nxf <= c { nxtop -= 1; } else { nxf -= c; }
        if nyf <= c { nytop -= 1; } else { nyf -= c; }
        if k == 0 {
            nrnd0 = 0;
            shift = 2;
        }
        for i in 0..nxtop {
            let row_start = ny * i;
            unshuffle_i64(&mut a[row_start..], nytop, 1, &mut tmp);
        }
        for j in 0..nytop {
            unshuffle_i64(&mut a[j..], nxtop, ny, &mut tmp);
        }

        if smooth { hsmooth_i64(a, nxtop, nytop, ny, scale); }

        let oddx = nxtop & 1;
        let oddy = nytop & 1;

        let mut i: usize = 0;
        while i + 1 < nxtop {
            let s00_row = ny * i;
            let s10_row = s00_row + ny;
            let mut j: usize = 0;
            while j + 1 < nytop {
                let s00 = s00_row + j;
                let s10 = s10_row + j;
                let mut h0 = a[s00];
                let mut hx = a[s10];
                let mut hy = a[s00 + 1];
                let mut hc = a[s10 + 1];
                hx = (hx + if hx >= 0 { prnd1 } else { nrnd1 }) & mask1;
                hy = (hy + if hy >= 0 { prnd1 } else { nrnd1 }) & mask1;
                hc = (hc + if hc >= 0 { prnd0 } else { nrnd0 }) & mask0;
                let lowbit0 = hc & bit0;
                hx = if hx >= 0 { hx - lowbit0 } else { hx + lowbit0 };
                hy = if hy >= 0 { hy - lowbit0 } else { hy + lowbit0 };
                let lowbit1 = (hc ^ hx ^ hy) & bit1;
                h0 = if h0 >= 0 {
                    h0 + lowbit0 - lowbit1
                } else if lowbit0 == 0 {
                    h0 + lowbit1
                } else {
                    h0 + (lowbit0 - lowbit1)
                };
                a[s10 + 1] = (h0 + hx + hy + hc) >> shift;
                a[s10]     = (h0 + hx - hy - hc) >> shift;
                a[s00 + 1] = (h0 - hx + hy - hc) >> shift;
                a[s00]     = (h0 - hx - hy + hc) >> shift;
                j += 2;
            }
            if oddy == 1 {
                let s00 = s00_row + j;
                let s10 = s10_row + j;
                let mut h0 = a[s00];
                let mut hx = a[s10];
                hx = (hx + if hx >= 0 { prnd1 } else { nrnd1 }) & mask1;
                let lowbit1 = hx & bit1;
                h0 = if h0 >= 0 { h0 - lowbit1 } else { h0 + lowbit1 };
                a[s10] = (h0 + hx) >> shift;
                a[s00] = (h0 - hx) >> shift;
            }
            i += 2;
        }
        if oddx == 1 {
            let s00_row = ny * i;
            let mut j: usize = 0;
            while j + 1 < nytop {
                let s00 = s00_row + j;
                let mut h0 = a[s00];
                let mut hy = a[s00 + 1];
                hy = (hy + if hy >= 0 { prnd1 } else { nrnd1 }) & mask1;
                let lowbit1 = hy & bit1;
                h0 = if h0 >= 0 { h0 - lowbit1 } else { h0 + lowbit1 };
                a[s00 + 1] = (h0 + hy) >> shift;
                a[s00]     = (h0 - hy) >> shift;
                j += 2;
            }
            if oddy == 1 {
                let s00 = s00_row + j;
                let h0 = a[s00];
                a[s00] = h0 >> shift;
            }
        }

        bit1 = bit0;
        bit0 >>= 1;
        mask1 = mask0;
        mask0 >>= 1;
        prnd1 = prnd0;
        prnd0 >>= 1;
        nrnd1 = nrnd0;
        nrnd0 = prnd0 - 1;
    }
    let _ = (bit2, mask2, prnd2, nrnd2);
    Ok(())
}

// ===== Cast the decoded coefficient array to the target dtype bytes
// in numpy native byte order.  Same shape as the rice.rs
// cast_i64_to_target_bytes helper. =====

fn cast_i32_to_target_bytes(
    a: &[i32], bytepix: u32, zbitpix: i32,
) -> PyResult<Vec<u8>> {
    let n = a.len();
    match (bytepix, zbitpix) {
        (1, 8) => {
            // u8 image — values are stored as u8 but the coefficient
            // pipeline is signed.  cfitsio casts via `(unsigned char)`.
            let mut out = Vec::with_capacity(n);
            for &v in a { out.push(v as u8); }
            Ok(out)
        }
        (2, 16) => {
            let mut out = Vec::with_capacity(n * 2);
            for &v in a {
                let bytes = (v as i16).to_ne_bytes();
                out.extend_from_slice(&bytes);
            }
            Ok(out)
        }
        (4, 32) => {
            let mut out = Vec::with_capacity(n * 4);
            for &v in a {
                let bytes = v.to_ne_bytes();
                out.extend_from_slice(&bytes);
            }
            Ok(out)
        }
        _ => Err(PyValueError::new_err(format!(
            "HCOMPRESS cast i32→target: unsupported (bytepix={}, zbitpix={})",
            bytepix, zbitpix
        ))),
    }
}

fn cast_i64_to_target_bytes(
    a: &[i64], bytepix: u32, zbitpix: i32,
) -> PyResult<Vec<u8>> {
    let n = a.len();
    match (bytepix, zbitpix) {
        (4, 32) => {
            // 32-bit image, decoded via i64 internal precision.
            // cfitsio's fits_hdecompress64 packs back to int via
            // `iarray[ii] = (int) a[ii]` (lines 173-174).
            let mut out = Vec::with_capacity(n * 4);
            for &v in a {
                let bytes = (v as i32).to_ne_bytes();
                out.extend_from_slice(&bytes);
            }
            Ok(out)
        }
        _ => Err(PyValueError::new_err(format!(
            "HCOMPRESS cast i64→target: unsupported (bytepix={}, zbitpix={})",
            bytepix, zbitpix
        ))),
    }
}

// ===== Public entry point =====

// Decode one HCOMPRESS_1-compressed tile.
//
// `nx_numpy`, `ny_numpy`: tile dimensions in numpy order (slowest
// first).  These correspond to cfitsio's nx (slow) and ny (fast)
// after the convention swap.
//
// `bytepix`: byte width of the target dtype (1, 2, or 4).
// `zbitpix`: image-side ZBITPIX (8, 16, or 32 — float ZBITPIX uses
// the quantized i32 path with zbitpix=32 passed here).
// `smooth`: whether to apply the smoothing pass during inverse
// H-transform.
//
// Output: `nx*ny*bytepix` bytes in numpy native byte order.
pub(crate) fn decode_hcompress(
    compressed: &[u8],
    nx_numpy: usize,
    ny_numpy: usize,
    bytepix: u32,
    zbitpix: i32,
    smooth: bool,
) -> PyResult<Vec<u8>> {
    if !matches!(zbitpix, 8 | 16 | 32) {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS: unsupported ZBITPIX {} (must be 8, 16, or 32)",
            zbitpix
        )));
    }
    if !matches!(bytepix, 1 | 2 | 4) {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS: unsupported bytepix {} (must be 1, 2, or 4)",
            bytepix
        )));
    }

    let n_pixels = nx_numpy * ny_numpy;
    let mut state = HState::new(compressed);

    if zbitpix == 32 {
        let mut buf = vec![0i64; n_pixels];
        let (decoded_nx, decoded_ny, scale) = decode_i64(&mut state, &mut buf)?;
        if decoded_nx != nx_numpy || decoded_ny != ny_numpy {
            return Err(PyValueError::new_err(format!(
                "HCOMPRESS: stream dims {}x{} disagree with header tile shape {}x{}",
                decoded_nx, decoded_ny, nx_numpy, ny_numpy
            )));
        }
        undigitize_i64(&mut buf, scale);
        hinv_i64(&mut buf, nx_numpy, ny_numpy, smooth, scale)?;
        let out = cast_i64_to_target_bytes(&buf, bytepix, zbitpix)?;
        Ok(out)
    } else {
        let mut buf = vec![0i32; n_pixels];
        let (decoded_nx, decoded_ny, scale) = decode_i32(&mut state, &mut buf)?;
        if decoded_nx != nx_numpy || decoded_ny != ny_numpy {
            return Err(PyValueError::new_err(format!(
                "HCOMPRESS: stream dims {}x{} disagree with header tile shape {}x{}",
                decoded_nx, decoded_ny, nx_numpy, ny_numpy
            )));
        }
        undigitize_i32(&mut buf, scale);
        hinv_i32(&mut buf, nx_numpy, ny_numpy, smooth, scale)?;
        let out = cast_i32_to_target_bytes(&buf, bytepix, zbitpix)?;
        // Output is native-endian (to_ne_bytes); same contract as
        // the i64 branch.
        Ok(out)
    }
}

// ===========================================================================
// ===== ENCODE ==============================================================
// ===========================================================================
//
// Port of cfitsio's `fits_hcompress.c` encoder family (fits_hcompress,
// fits_hcompress64, htrans, htrans64, digitize, digitize64, shuffle,
// shuffle64, encode, encode64, doencode, doencode64, qtree_encode,
// qtree_encode64, qtree_onebit, qtree_onebit64, qtree_reduce, bufcopy,
// write_bdirect, write_bdirect64, output_nbits/nybble/nnybble, etc.).
//
// The wire format is the inverse of the decoder above; bytes produced
// here are byte-exact with cfitsio's encoder given the same input.
// Function structure mirrors the C source for diffability.

// ===== Forward shuffle (inverse of unshuffle): split even/odd elements =====
//
// Takes the first `n` elements of `a` (with stride `n2`) and reorders
// them so the even-indexed elements occupy the first half and the
// odd-indexed elements occupy the second half.  `tmp` is scratch of at
// least ceil(n/2) elements.
fn shuffle_i32(a: &mut [i32], n: usize, n2: usize, tmp: &mut [i32]) {
    // Copy odd elements to tmp.
    let mut k: usize = 0;
    let mut i: usize = 1;
    while i < n {
        tmp[k] = a[n2 * i];
        k += 1;
        i += 2;
    }
    // Compress even elements into first half of A (skip a[0], shift
    // a[2*n2] → a[1*n2], a[4*n2] → a[2*n2], etc.).
    let mut dst = n2;
    let mut src = n2 + n2;
    let mut i: usize = 2;
    while i < n {
        a[dst] = a[src];
        dst += n2;
        src += n2 + n2;
        i += 2;
    }
    // Put odd elements (from tmp) into 2nd half.
    let mut p = dst;
    let mut k: usize = 0;
    let mut i: usize = 1;
    while i < n {
        a[p] = tmp[k];
        p += n2;
        k += 1;
        i += 2;
    }
}

fn shuffle_i64(a: &mut [i64], n: usize, n2: usize, tmp: &mut [i64]) {
    let mut k: usize = 0;
    let mut i: usize = 1;
    while i < n {
        tmp[k] = a[n2 * i];
        k += 1;
        i += 2;
    }
    let mut dst = n2;
    let mut src = n2 + n2;
    let mut i: usize = 2;
    while i < n {
        a[dst] = a[src];
        dst += n2;
        src += n2 + n2;
        i += 2;
    }
    let mut p = dst;
    let mut k: usize = 0;
    let mut i: usize = 1;
    while i < n {
        a[p] = tmp[k];
        p += n2;
        k += 1;
        i += 2;
    }
}

// ===== Forward H-transform =====
//
// Mirror of cfitsio's `htrans` / `htrans64`.  Walks log2n reductions:
// each combines 2x2 coefficient blocks into (h0, hx, hy, hc) and stores
// them shuffled across the array.  The result is a coefficient array
// ready for digitize + encode.
//
// `shift` controls the per-iteration division by 2 (skipped on the first
// pass so the H-transform's first level preserves more precision); the
// `prnd / nrnd2` rounding constants accompany the mask-and-shift, and
// double after each pass.
fn htrans_i32(a: &mut [i32], nx: usize, ny: usize) {
    let nmax = nx.max(ny);
    let mut log2n = (((nmax as f64).ln() / std::f64::consts::LN_2) + 0.5)
        as u32;
    if nmax > (1usize << log2n) {
        log2n += 1;
    }
    let mut tmp = vec![0i32; nmax.div_ceil(2)];

    let mut shift: u32 = 0;
    let mut mask: i32 = -2;
    let mut mask2: i32 = mask.wrapping_shl(1);
    let mut prnd: i32 = 1;
    let mut prnd2: i32 = prnd.wrapping_shl(1);
    let mut nrnd2: i32 = prnd2 - 1;

    let mut nxtop = nx;
    let mut nytop = ny;

    for _k in 0..log2n {
        let oddx = nxtop & 1;
        let oddy = nytop & 1;
        let mut i: usize = 0;
        while i + oddx < nxtop {
            let mut s00 = i * ny;
            let mut s10 = s00 + ny;
            let mut j: usize = 0;
            while j + oddy < nytop {
                let h0 = (a[s10 + 1] + a[s10] + a[s00 + 1] + a[s00])
                    .wrapping_shr(shift);
                let hx = (a[s10 + 1] + a[s10] - a[s00 + 1] - a[s00])
                    .wrapping_shr(shift);
                let hy = (a[s10 + 1] - a[s10] + a[s00 + 1] - a[s00])
                    .wrapping_shr(shift);
                let hc = (a[s10 + 1] - a[s10] - a[s00 + 1] + a[s00])
                    .wrapping_shr(shift);
                a[s10 + 1] = hc;
                a[s10] = (if hx >= 0 { hx + prnd } else { hx }) & mask;
                a[s00 + 1] = (if hy >= 0 { hy + prnd } else { hy }) & mask;
                a[s00] = (if h0 >= 0 { h0 + prnd2 } else { h0 + nrnd2 })
                    & mask2;
                s00 += 2;
                s10 += 2;
                j += 2;
            }
            if oddy == 1 {
                let h0 = (a[s10] + a[s00]).wrapping_shl(1 - shift);
                let hx = (a[s10] - a[s00]).wrapping_shl(1 - shift);
                a[s10] = (if hx >= 0 { hx + prnd } else { hx }) & mask;
                a[s00] = (if h0 >= 0 { h0 + prnd2 } else { h0 + nrnd2 })
                    & mask2;
            }
            i += 2;
        }
        if oddx == 1 {
            let mut s00 = i * ny;
            let mut j: usize = 0;
            while j + oddy < nytop {
                let h0 = (a[s00 + 1] + a[s00]).wrapping_shl(1 - shift);
                let hy = (a[s00 + 1] - a[s00]).wrapping_shl(1 - shift);
                a[s00 + 1] = (if hy >= 0 { hy + prnd } else { hy }) & mask;
                a[s00] = (if h0 >= 0 { h0 + prnd2 } else { h0 + nrnd2 })
                    & mask2;
                s00 += 2;
                j += 2;
            }
            if oddy == 1 {
                let h0 = a[s00].wrapping_shl(2 - shift);
                a[s00] = (if h0 >= 0 { h0 + prnd2 } else { h0 + nrnd2 })
                    & mask2;
            }
        }
        // Shuffle along each dimension to group coefficients by order.
        for i in 0..nxtop {
            shuffle_i32(&mut a[ny * i..], nytop, 1, &mut tmp);
        }
        for j in 0..nytop {
            shuffle_i32(&mut a[j..], nxtop, ny, &mut tmp);
        }
        nxtop = (nxtop + 1) >> 1;
        nytop = (nytop + 1) >> 1;
        shift = 1;
        mask = mask2;
        prnd = prnd2;
        mask2 = mask2.wrapping_shl(1);
        prnd2 = prnd2.wrapping_shl(1);
        nrnd2 = prnd2 - 1;
    }
}

fn htrans_i64(a: &mut [i64], nx: usize, ny: usize) {
    let nmax = nx.max(ny);
    let mut log2n = (((nmax as f64).ln() / std::f64::consts::LN_2) + 0.5)
        as u32;
    if nmax > (1usize << log2n) {
        log2n += 1;
    }
    let mut tmp = vec![0i64; nmax.div_ceil(2)];

    let mut shift: u32 = 0;
    let mut mask: i64 = -2;
    let mut mask2: i64 = mask.wrapping_shl(1);
    let mut prnd: i64 = 1;
    let mut prnd2: i64 = prnd.wrapping_shl(1);
    let mut nrnd2: i64 = prnd2 - 1;

    let mut nxtop = nx;
    let mut nytop = ny;

    for _k in 0..log2n {
        let oddx = nxtop & 1;
        let oddy = nytop & 1;
        let mut i: usize = 0;
        while i + oddx < nxtop {
            let mut s00 = i * ny;
            let mut s10 = s00 + ny;
            let mut j: usize = 0;
            while j + oddy < nytop {
                let h0 = (a[s10 + 1] + a[s10] + a[s00 + 1] + a[s00])
                    .wrapping_shr(shift);
                let hx = (a[s10 + 1] + a[s10] - a[s00 + 1] - a[s00])
                    .wrapping_shr(shift);
                let hy = (a[s10 + 1] - a[s10] + a[s00 + 1] - a[s00])
                    .wrapping_shr(shift);
                let hc = (a[s10 + 1] - a[s10] - a[s00 + 1] + a[s00])
                    .wrapping_shr(shift);
                a[s10 + 1] = hc;
                a[s10] = (if hx >= 0 { hx + prnd } else { hx }) & mask;
                a[s00 + 1] = (if hy >= 0 { hy + prnd } else { hy }) & mask;
                a[s00] = (if h0 >= 0 { h0 + prnd2 } else { h0 + nrnd2 })
                    & mask2;
                s00 += 2;
                s10 += 2;
                j += 2;
            }
            if oddy == 1 {
                let h0 = (a[s10] + a[s00]).wrapping_shl(1 - shift);
                let hx = (a[s10] - a[s00]).wrapping_shl(1 - shift);
                a[s10] = (if hx >= 0 { hx + prnd } else { hx }) & mask;
                a[s00] = (if h0 >= 0 { h0 + prnd2 } else { h0 + nrnd2 })
                    & mask2;
            }
            i += 2;
        }
        if oddx == 1 {
            let mut s00 = i * ny;
            let mut j: usize = 0;
            while j + oddy < nytop {
                let h0 = (a[s00 + 1] + a[s00]).wrapping_shl(1 - shift);
                let hy = (a[s00 + 1] - a[s00]).wrapping_shl(1 - shift);
                a[s00 + 1] = (if hy >= 0 { hy + prnd } else { hy }) & mask;
                a[s00] = (if h0 >= 0 { h0 + prnd2 } else { h0 + nrnd2 })
                    & mask2;
                s00 += 2;
                j += 2;
            }
            if oddy == 1 {
                let h0 = a[s00].wrapping_shl(2 - shift);
                a[s00] = (if h0 >= 0 { h0 + prnd2 } else { h0 + nrnd2 })
                    & mask2;
            }
        }
        for i in 0..nxtop {
            shuffle_i64(&mut a[ny * i..], nytop, 1, &mut tmp);
        }
        for j in 0..nytop {
            shuffle_i64(&mut a[j..], nxtop, ny, &mut tmp);
        }
        nxtop = (nxtop + 1) >> 1;
        nytop = (nytop + 1) >> 1;
        shift = 1;
        mask = mask2;
        prnd = prnd2;
        mask2 = mask2.wrapping_shl(1);
        prnd2 = prnd2.wrapping_shl(1);
        nrnd2 = prnd2 - 1;
    }
}

// ===== Digitize (multiply each coefficient by 1/scale, rounded to
// nearest integer with ties away from zero) =====
fn digitize_i32(a: &mut [i32], scale: i32) {
    if scale <= 1 {
        return;
    }
    let d: i32 = (scale + 1) / 2 - 1;
    for v in a.iter_mut() {
        *v = if *v > 0 { *v + d } else { *v - d } / scale;
    }
}

fn digitize_i64(a: &mut [i64], scale: i32) {
    if scale <= 1 {
        return;
    }
    let s = scale as i64;
    let d: i64 = (s + 1) / 2 - 1;
    for v in a.iter_mut() {
        *v = if *v > 0 { *v + d } else { *v - d } / s;
    }
}

// ===== Bit output state (mirror of cfitsio's start/output/done_outputing_bits) =====
//
// Two distinct bit buffers are in play during encode:
//
//   - The MAIN buffer (buffer2 / bits_to_go2) feeds the on-disk output
//     stream via output_nbits/nybble/nnybble.  Bits are accumulated
//     MSB-first per byte and flushed when the buffer fills.  Lives in
//     HWriter.
//
//   - The HUFFMAN buffer (bitbuffer3 / bits_to_go3) is local to one
//     qtree_encode call: bufcopy LSB-shifts Huffman codes into it and
//     drains complete bytes into a scratch buffer.  The scratch is then
//     written back to the main stream byte-by-byte in reverse order at
//     the end of the bit plane.
//
// cfitsio uses module-level statics (`buffer2`/`bits_to_go2` for main,
// `bitbuffer`/`bits_to_go3` for huffman).  We pass the Huffman state by
// value into bufcopy and back; the main state lives in HWriter.
struct HWriter {
    out: Vec<u8>,
    buffer2: u32,
    // Signed because the cfitsio source uses `bits_to_go2 -= n` which
    // can go negative (the "overflow" signal triggers a byte flush).
    bits_to_go2: i32,
}

impl HWriter {
    fn new() -> Self {
        HWriter { out: Vec::new(), buffer2: 0, bits_to_go2: 8 }
    }

    fn qwrite(&mut self, bytes: &[u8]) {
        self.out.extend_from_slice(bytes);
    }

    fn writeint(&mut self, a: i32) {
        self.out.extend_from_slice(&a.to_be_bytes());
    }

    fn writelonglong(&mut self, a: i64) {
        self.out.extend_from_slice(&a.to_be_bytes());
    }

    fn start_outputing_bits(&mut self) {
        self.buffer2 = 0;
        self.bits_to_go2 = 8;
    }

    // Write N bits (N <= 8).  Matches cfitsio's `output_nbits` exactly:
    // shift the buffer left by N, OR in (bits & mask), and flush the
    // top 8 bits if the buffer overflowed.
    fn output_nbits(&mut self, bits: u32, n: u32) {
        debug_assert!(n <= 8);
        let mask = if n == 0 { 0 } else { (1u32 << n) - 1 };
        self.buffer2 = self.buffer2.wrapping_shl(n) | (bits & mask);
        self.bits_to_go2 -= n as i32;
        if self.bits_to_go2 <= 0 {
            let byte =
                (self.buffer2 >> ((-self.bits_to_go2) as u32)) & 0xff;
            self.out.push(byte as u8);
            self.bits_to_go2 += 8;
        }
    }

    // Write a 4-bit nybble.  Matches cfitsio's `output_nybble`.
    fn output_nybble(&mut self, bits: u32) {
        self.buffer2 = self.buffer2.wrapping_shl(4) | (bits & 0x0f);
        self.bits_to_go2 -= 4;
        if self.bits_to_go2 <= 0 {
            let byte =
                (self.buffer2 >> ((-self.bits_to_go2) as u32)) & 0xff;
            self.out.push(byte as u8);
            self.bits_to_go2 += 8;
        }
    }

    // Write an array of nybbles (the lower 4 bits of each byte).
    // cfitsio's `output_nnybble` has byte-alignment fast paths;
    // matching them byte-for-byte requires reproducing both branches
    // since they handle slightly different bit-buffer states.
    fn output_nnybble(&mut self, array: &[u8]) {
        let n = array.len();
        if n == 0 {
            return;
        }
        if n == 1 {
            self.output_nybble(array[0] as u32);
            return;
        }
        let mut kk: usize = 0;
        if self.bits_to_go2 <= 4 {
            // Just room for one nybble; write it separately.
            self.output_nybble(array[0] as u32);
            kk += 1;
            if n == 2 {
                self.output_nybble(array[1] as u32);
                return;
            }
        }
        // bits_to_go2 is now in {5,6,7,8}.
        let shift: u32 = 8 - self.bits_to_go2 as u32;
        let jj = (n - kk) / 2;
        if self.bits_to_go2 == 8 {
            // Byte-aligned: write packed nybbles directly.
            self.buffer2 = 0;
            for _ in 0..jj {
                let byte = ((array[kk] & 15) << 4) | (array[kk + 1] & 15);
                self.out.push(byte);
                kk += 2;
            }
        } else {
            for _ in 0..jj {
                self.buffer2 = self.buffer2.wrapping_shl(8)
                    | (((array[kk] & 15) as u32) << 4)
                    | ((array[kk + 1] & 15) as u32);
                let byte = (self.buffer2 >> shift) & 0xff;
                self.out.push(byte as u8);
                kk += 2;
            }
        }
        // Trailing odd nybble.
        if kk != n {
            self.output_nybble(array[n - 1] as u32);
        }
    }

    // Flush partial trailing byte (the last output byte may be
    // partial; pad with zeros at the bottom).
    fn done_outputing_bits(&mut self) {
        if self.bits_to_go2 < 8 {
            let byte = self.buffer2.wrapping_shl(self.bits_to_go2 as u32);
            self.out.push((byte & 0xff) as u8);
        }
    }
}

// Huffman code table for values 0..15 (lifted verbatim from cfitsio
// `qtree_encode.c`).  Indexed by the 4-bit value, returns (code, nbits).
const HCODE: [u32; 16] = [
    0x3e, 0x00, 0x01, 0x08, 0x02, 0x09, 0x1a, 0x1b,
    0x03, 0x1c, 0x0a, 0x1d, 0x0b, 0x1e, 0x3f, 0x0c,
];
const HNCODE: [u32; 16] = [
    6, 3, 3, 4, 3, 4, 5, 5,
    3, 5, 4, 5, 4, 5, 6, 4,
];

// ===== qtree_onebit_i32 / qtree_onebit_i64 =====
//
// Extract bit `bit` from each 2x2 block of `a` (dims nx*ny, row
// stride n), packing the 4 bits into the corresponding element of
// `b` (dims (nx+1)/2 * (ny+1)/2) using the same bit positions as the
// decoder's qtree_copy:
//   bit3 → a[s00],   bit2 → a[s00+1]
//   bit1 → a[s10],   bit0 → a[s10+1]
fn qtree_onebit_i32(
    a: &[i32], n: usize, nx: usize, ny: usize, b: &mut [u8], bit: u32,
) {
    let b0: i32 = 1i32 << bit;
    let b1: i32 = b0.wrapping_shl(1);
    let b2: i32 = b0.wrapping_shl(2);
    let b3: i32 = b0.wrapping_shl(3);
    let mut k: usize = 0;
    let mut i: usize = 0;
    while i + 1 < nx {
        let mut s00 = n * i;
        let mut s10 = s00 + n;
        let mut j: usize = 0;
        while j + 1 < ny {
            let v: i32 = (a[s10 + 1] & b0)
                | (a[s10].wrapping_shl(1) & b1)
                | (a[s00 + 1].wrapping_shl(2) & b2)
                | (a[s00].wrapping_shl(3) & b3);
            b[k] = (v.wrapping_shr(bit) & 0x0f) as u8;
            k += 1;
            s00 += 2;
            s10 += 2;
            j += 2;
        }
        if j < ny {
            // Odd row length — only bit1 and bit3 contribute.
            let v: i32 = (a[s10].wrapping_shl(1) & b1)
                | (a[s00].wrapping_shl(3) & b3);
            b[k] = (v.wrapping_shr(bit) & 0x0f) as u8;
            k += 1;
        }
        i += 2;
    }
    if i < nx {
        // Odd column length — only bit2 and bit3 contribute.
        let mut s00 = n * i;
        let mut j: usize = 0;
        while j + 1 < ny {
            let v: i32 = (a[s00 + 1].wrapping_shl(2) & b2)
                | (a[s00].wrapping_shl(3) & b3);
            b[k] = (v.wrapping_shr(bit) & 0x0f) as u8;
            k += 1;
            s00 += 2;
            j += 2;
        }
        if j < ny {
            // Corner element — only bit3.
            let v: i32 = a[s00].wrapping_shl(3) & b3;
            b[k] = (v.wrapping_shr(bit) & 0x0f) as u8;
        }
    }
}

fn qtree_onebit_i64(
    a: &[i64], n: usize, nx: usize, ny: usize, b: &mut [u8], bit: u32,
) {
    let b0: i64 = 1i64 << bit;
    let b1: i64 = b0.wrapping_shl(1);
    let b2: i64 = b0.wrapping_shl(2);
    let b3: i64 = b0.wrapping_shl(3);
    let mut k: usize = 0;
    let mut i: usize = 0;
    while i + 1 < nx {
        let mut s00 = n * i;
        let mut s10 = s00 + n;
        let mut j: usize = 0;
        while j + 1 < ny {
            let v: i64 = (a[s10 + 1] & b0)
                | (a[s10].wrapping_shl(1) & b1)
                | (a[s00 + 1].wrapping_shl(2) & b2)
                | (a[s00].wrapping_shl(3) & b3);
            b[k] = (v.wrapping_shr(bit) & 0x0f) as u8;
            k += 1;
            s00 += 2;
            s10 += 2;
            j += 2;
        }
        if j < ny {
            let v: i64 = (a[s10].wrapping_shl(1) & b1)
                | (a[s00].wrapping_shl(3) & b3);
            b[k] = (v.wrapping_shr(bit) & 0x0f) as u8;
            k += 1;
        }
        i += 2;
    }
    if i < nx {
        let mut s00 = n * i;
        let mut j: usize = 0;
        while j + 1 < ny {
            let v: i64 = (a[s00 + 1].wrapping_shl(2) & b2)
                | (a[s00].wrapping_shl(3) & b3);
            b[k] = (v.wrapping_shr(bit) & 0x0f) as u8;
            k += 1;
            s00 += 2;
            j += 2;
        }
        if j < ny {
            let v: i64 = a[s00].wrapping_shl(3) & b3;
            b[k] = (v.wrapping_shr(bit) & 0x0f) as u8;
        }
    }
}

// ===== qtree_reduce =====
//
// Take a (nx,ny) array of 4-bit values and reduce by 2x2 OR (each
// output element is a 4-bit value of [s10+1 != 0, s10 != 0, s00+1 !=
// 0, s00 != 0]).  Same bit positions as qtree_onebit so the reduction
// composes with the next plane's bufcopy correctly.  Works in place
// (same buffer for a and b).
fn qtree_reduce(a: &mut [u8], n: usize, nx: usize, ny: usize) {
    let mut k: usize = 0;
    let mut i: usize = 0;
    while i + 1 < nx {
        let mut s00 = n * i;
        let mut s10 = s00 + n;
        let mut j: usize = 0;
        while j + 1 < ny {
            let v: u8 = ((a[s10 + 1] != 0) as u8)
                | (((a[s10] != 0) as u8) << 1)
                | (((a[s00 + 1] != 0) as u8) << 2)
                | (((a[s00] != 0) as u8) << 3);
            a[k] = v;
            k += 1;
            s00 += 2;
            s10 += 2;
            j += 2;
        }
        if j < ny {
            let v: u8 = (((a[s10] != 0) as u8) << 1)
                | (((a[s00] != 0) as u8) << 3);
            a[k] = v;
            k += 1;
        }
        i += 2;
    }
    if i < nx {
        let mut s00 = n * i;
        let mut j: usize = 0;
        while j + 1 < ny {
            let v: u8 = (((a[s00 + 1] != 0) as u8) << 2)
                | (((a[s00] != 0) as u8) << 3);
            a[k] = v;
            k += 1;
            s00 += 2;
            j += 2;
        }
        if j < ny {
            let v: u8 = ((a[s00] != 0) as u8) << 3;
            a[k] = v;
        }
    }
}

// ===== bufcopy =====
//
// Pack the Huffman codes of every non-zero element of `a[..n]` into the
// `buffer` byte array, draining complete bytes as we go.  Returns
// `true` if the buffer would overflow (caller falls back to bdirect).
// The local Huffman bit buffer (`bitbuffer`, `bits_to_go3`) is passed
// in and out so the caller can drain the trailing bits after the loop.
fn bufcopy(
    a: &[u8], n: usize, buffer: &mut [u8], b: &mut usize, bmax: usize,
    bitbuffer: &mut u32, bits_to_go3: &mut u32,
) -> bool {
    for i in 0..n {
        let v = a[i] as usize;
        if v != 0 {
            *bitbuffer |= HCODE[v].wrapping_shl(*bits_to_go3);
            *bits_to_go3 += HNCODE[v];
            if *bits_to_go3 >= 8 {
                buffer[*b] = (*bitbuffer & 0xFF) as u8;
                *b += 1;
                if *b >= bmax {
                    return true;
                }
                *bitbuffer >>= 8;
                *bits_to_go3 -= 8;
            }
        }
    }
    false
}

// ===== write_bdirect_i32 / write_bdirect_i64 =====
//
// Fallback when the quadtree representation would have exceeded the
// scratch buffer: output a 0x0 nybble (bdirect marker), then
// qtree_onebit the bit plane and dump it via output_nnybble.
fn write_bdirect_i32(
    writer: &mut HWriter, a: &[i32], n: usize, nqx: usize, nqy: usize,
    scratch: &mut [u8], bit: u32,
) {
    writer.output_nybble(0x0);
    qtree_onebit_i32(a, n, nqx, nqy, scratch, bit);
    let count = nqx.div_ceil(2) * nqy.div_ceil(2);
    writer.output_nnybble(&scratch[..count]);
}

fn write_bdirect_i64(
    writer: &mut HWriter, a: &[i64], n: usize, nqx: usize, nqy: usize,
    scratch: &mut [u8], bit: u32,
) {
    writer.output_nybble(0x0);
    qtree_onebit_i64(a, n, nqx, nqy, scratch, bit);
    let count = nqx.div_ceil(2) * nqy.div_ceil(2);
    writer.output_nnybble(&scratch[..count]);
}

// ===== qtree_encode_i32 / qtree_encode_i64 =====
//
// Encode one quadrant (positioned via a's caller-supplied offset) by
// bit-plane.  For each plane top-down: first qtree_onebit extracts the
// plane into a scratch byte array, then we try to pack the resulting
// codes via repeated qtree_reduce + bufcopy passes.  If at any point
// bufcopy reports overflow, fall back to write_bdirect for that plane.
// Otherwise emit the 0xF marker followed by the buffered bytes
// (reversed) plus any trailing Huffman bits.
fn qtree_encode_i32(
    writer: &mut HWriter, a: &[i32], n: usize,
    nqx: usize, nqy: usize, nbitplanes: u32,
) {
    let nqmax = nqx.max(nqy);
    let mut log2n = (((nqmax as f64).ln() / std::f64::consts::LN_2) + 0.5)
        as u32;
    if nqmax > (1usize << log2n) {
        log2n += 1;
    }
    let nqx2 = nqx.div_ceil(2);
    let nqy2 = nqy.div_ceil(2);
    let bmax = (nqx2 * nqy2).div_ceil(2);
    let mut scratch = vec![0u8; 2 * bmax];
    let mut buffer = vec![0u8; bmax];

    let mut bit_signed: i32 = nbitplanes as i32 - 1;
    while bit_signed >= 0 {
        let bit = bit_signed as u32;
        let mut b: usize = 0;
        let mut bitbuffer: u32 = 0;
        let mut bits_to_go3: u32 = 0;
        // First pass: qtree_onebit into scratch.
        qtree_onebit_i32(a, n, nqx, nqy, &mut scratch, bit);
        let mut nx: usize = (nqx + 1) >> 1;
        let mut ny: usize = (nqy + 1) >> 1;
        let mut overflowed = false;
        if bufcopy(
            &scratch, nx * ny, &mut buffer, &mut b, bmax,
            &mut bitbuffer, &mut bits_to_go3,
        ) {
            overflowed = true;
        } else {
            for _k in 1..log2n {
                qtree_reduce(&mut scratch, ny, nx, ny);
                nx = (nx + 1) >> 1;
                ny = (ny + 1) >> 1;
                if bufcopy(
                    &scratch, nx * ny, &mut buffer, &mut b, bmax,
                    &mut bitbuffer, &mut bits_to_go3,
                ) {
                    overflowed = true;
                    break;
                }
            }
        }
        if overflowed {
            write_bdirect_i32(writer, a, n, nqx, nqy, &mut scratch, bit);
        } else {
            // Quadtree-encoded: marker + trailing bits + buffer reversed.
            writer.output_nybble(0xF);
            if b == 0 {
                if bits_to_go3 > 0 {
                    let mask = if bits_to_go3 == 0 {
                        0
                    } else {
                        (1u32 << bits_to_go3) - 1
                    };
                    writer.output_nbits(bitbuffer & mask, bits_to_go3);
                } else {
                    // No 1s in this plane: emit zero Huffman code.
                    writer.output_nbits(HCODE[0], HNCODE[0]);
                }
            } else {
                if bits_to_go3 > 0 {
                    let mask = (1u32 << bits_to_go3) - 1;
                    writer.output_nbits(bitbuffer & mask, bits_to_go3);
                }
                for i in (0..b).rev() {
                    writer.output_nbits(buffer[i] as u32, 8);
                }
            }
        }
        bit_signed -= 1;
    }
}

fn qtree_encode_i64(
    writer: &mut HWriter, a: &[i64], n: usize,
    nqx: usize, nqy: usize, nbitplanes: u32,
) {
    let nqmax = nqx.max(nqy);
    let mut log2n = (((nqmax as f64).ln() / std::f64::consts::LN_2) + 0.5)
        as u32;
    if nqmax > (1usize << log2n) {
        log2n += 1;
    }
    let nqx2 = nqx.div_ceil(2);
    let nqy2 = nqy.div_ceil(2);
    let bmax = (nqx2 * nqy2).div_ceil(2);
    let mut scratch = vec![0u8; 2 * bmax];
    let mut buffer = vec![0u8; bmax];

    let mut bit_signed: i32 = nbitplanes as i32 - 1;
    while bit_signed >= 0 {
        let bit = bit_signed as u32;
        let mut b: usize = 0;
        let mut bitbuffer: u32 = 0;
        let mut bits_to_go3: u32 = 0;
        qtree_onebit_i64(a, n, nqx, nqy, &mut scratch, bit);
        let mut nx: usize = (nqx + 1) >> 1;
        let mut ny: usize = (nqy + 1) >> 1;
        let mut overflowed = false;
        if bufcopy(
            &scratch, nx * ny, &mut buffer, &mut b, bmax,
            &mut bitbuffer, &mut bits_to_go3,
        ) {
            overflowed = true;
        } else {
            for _k in 1..log2n {
                qtree_reduce(&mut scratch, ny, nx, ny);
                nx = (nx + 1) >> 1;
                ny = (ny + 1) >> 1;
                if bufcopy(
                    &scratch, nx * ny, &mut buffer, &mut b, bmax,
                    &mut bitbuffer, &mut bits_to_go3,
                ) {
                    overflowed = true;
                    break;
                }
            }
        }
        if overflowed {
            write_bdirect_i64(writer, a, n, nqx, nqy, &mut scratch, bit);
        } else {
            writer.output_nybble(0xF);
            if b == 0 {
                if bits_to_go3 > 0 {
                    let mask = (1u32 << bits_to_go3) - 1;
                    writer.output_nbits(bitbuffer & mask, bits_to_go3);
                } else {
                    writer.output_nbits(HCODE[0], HNCODE[0]);
                }
            } else {
                if bits_to_go3 > 0 {
                    let mask = (1u32 << bits_to_go3) - 1;
                    writer.output_nbits(bitbuffer & mask, bits_to_go3);
                }
                for i in (0..b).rev() {
                    writer.output_nbits(buffer[i] as u32, 8);
                }
            }
        }
        bit_signed -= 1;
    }
}

// ===== doencode_i32 / doencode_i64 =====
//
// Encode all 4 quadrants of the assumed-positive array `a`, append the
// EOF nybble (0), and flush the trailing partial byte.
fn doencode_i32(
    writer: &mut HWriter, a: &[i32], nx: usize, ny: usize,
    nbitplanes: [u8; 3],
) {
    let nx2 = nx.div_ceil(2);
    let ny2 = ny.div_ceil(2);
    writer.start_outputing_bits();
    qtree_encode_i32(writer, &a[0..], ny, nx2, ny2, nbitplanes[0] as u32);
    qtree_encode_i32(writer, &a[ny2..], ny, nx2, ny / 2, nbitplanes[1] as u32);
    qtree_encode_i32(writer, &a[ny * nx2..], ny, nx / 2, ny2, nbitplanes[1] as u32);
    qtree_encode_i32(
        writer, &a[ny * nx2 + ny2..], ny, nx / 2, ny / 2,
        nbitplanes[2] as u32,
    );
    writer.output_nybble(0);
    writer.done_outputing_bits();
}

fn doencode_i64(
    writer: &mut HWriter, a: &[i64], nx: usize, ny: usize,
    nbitplanes: [u8; 3],
) {
    let nx2 = nx.div_ceil(2);
    let ny2 = ny.div_ceil(2);
    writer.start_outputing_bits();
    qtree_encode_i64(writer, &a[0..], ny, nx2, ny2, nbitplanes[0] as u32);
    qtree_encode_i64(writer, &a[ny2..], ny, nx2, ny / 2, nbitplanes[1] as u32);
    qtree_encode_i64(writer, &a[ny * nx2..], ny, nx / 2, ny2, nbitplanes[1] as u32);
    qtree_encode_i64(
        writer, &a[ny * nx2 + ny2..], ny, nx / 2, ny / 2,
        nbitplanes[2] as u32,
    );
    writer.output_nybble(0);
    writer.done_outputing_bits();
}

// ===== encode_i32 / encode_i64 =====
//
// Top-level encoder: write magic + dims + scale + sumall (= a[0]),
// zero a[0], extract sign bits, replace a with |a|, compute per-
// quadrant nbitplanes, write nbitplanes, call doencode, append sign
// bits.  Matches cfitsio's encode/encode64 exactly.
fn encode_i32(
    writer: &mut HWriter, a: &mut [i32], nx: usize, ny: usize, scale: i32,
) {
    let nel = nx * ny;
    writer.qwrite(&CODE_MAGIC);
    writer.writeint(nx as i32);
    writer.writeint(ny as i32);
    writer.writeint(scale);
    writer.writelonglong(a[0] as i64);
    a[0] = 0;
    // Pack sign bits, 8 per byte.  Same packing order as cfitsio's
    // encode: left-shift the byte each step, OR in 1 for negative,
    // increment to next byte when 8 bits accumulated.
    let mut signbits = vec![0u8; nel.div_ceil(8)];
    let mut nsign: usize = 0;
    let mut bits_to_go: i32 = 8;
    for i in 0..nel {
        if a[i] > 0 {
            signbits[nsign] <<= 1;
            bits_to_go -= 1;
        } else if a[i] < 0 {
            signbits[nsign] = (signbits[nsign] << 1) | 1;
            bits_to_go -= 1;
            a[i] = -a[i];
        }
        if bits_to_go == 0 {
            bits_to_go = 8;
            nsign += 1;
        }
    }
    if bits_to_go != 8 {
        signbits[nsign] <<= bits_to_go as u32;
        nsign += 1;
    }
    // Compute per-quadrant max absolute value, then nbitplanes per
    // quadrant.
    let nx2 = nx.div_ceil(2);
    let ny2 = ny.div_ceil(2);
    let mut vmax: [i32; 3] = [0; 3];
    let mut j: usize = 0;
    let mut k: usize = 0;
    for i in 0..nel {
        let q = ((j >= ny2) as usize) + ((k >= nx2) as usize);
        if vmax[q] < a[i] {
            vmax[q] = a[i];
        }
        j += 1;
        if j >= ny {
            j = 0;
            k += 1;
        }
    }
    let mut nbitplanes = [0u8; 3];
    for q in 0..3 {
        let mut v = vmax[q];
        while v > 0 {
            v >>= 1;
            nbitplanes[q] += 1;
        }
    }
    writer.qwrite(&nbitplanes);
    doencode_i32(writer, a, nx, ny, nbitplanes);
    if nsign > 0 {
        writer.qwrite(&signbits[..nsign]);
    }
}

fn encode_i64(
    writer: &mut HWriter, a: &mut [i64], nx: usize, ny: usize, scale: i32,
) {
    let nel = nx * ny;
    writer.qwrite(&CODE_MAGIC);
    writer.writeint(nx as i32);
    writer.writeint(ny as i32);
    writer.writeint(scale);
    writer.writelonglong(a[0]);
    a[0] = 0;
    let mut signbits = vec![0u8; nel.div_ceil(8)];
    let mut nsign: usize = 0;
    let mut bits_to_go: i32 = 8;
    for i in 0..nel {
        if a[i] > 0 {
            signbits[nsign] <<= 1;
            bits_to_go -= 1;
        } else if a[i] < 0 {
            signbits[nsign] = (signbits[nsign] << 1) | 1;
            bits_to_go -= 1;
            a[i] = -a[i];
        }
        if bits_to_go == 0 {
            bits_to_go = 8;
            nsign += 1;
        }
    }
    if bits_to_go != 8 {
        signbits[nsign] <<= bits_to_go as u32;
        nsign += 1;
    }
    let nx2 = nx.div_ceil(2);
    let ny2 = ny.div_ceil(2);
    let mut vmax: [i64; 3] = [0; 3];
    let mut j: usize = 0;
    let mut k: usize = 0;
    for i in 0..nel {
        let q = ((j >= ny2) as usize) + ((k >= nx2) as usize);
        if vmax[q] < a[i] {
            vmax[q] = a[i];
        }
        j += 1;
        if j >= ny {
            j = 0;
            k += 1;
        }
    }
    let mut nbitplanes = [0u8; 3];
    for q in 0..3 {
        let mut v = vmax[q];
        while v > 0 {
            v >>= 1;
            nbitplanes[q] += 1;
        }
    }
    writer.qwrite(&nbitplanes);
    doencode_i64(writer, a, nx, ny, nbitplanes);
    if nsign > 0 {
        writer.qwrite(&signbits[..nsign]);
    }
}

// ===== Public encode entry =====
//
// Encode one tile's pixel bytes (FITS big-endian) to HCOMPRESS_1 bytes.
// `nx_numpy` / `ny_numpy` are the tile dims in numpy axis order
// (slowest first); they map directly to cfitsio's nx (slow) and ny
// (fast) since the convention is the same internally.
//
// `bytepix` and `zbitpix` together pick the internal precision: i32 for
// zbitpix=8 or 16, i64 for zbitpix=32.  ZBITPIX=64 is unsupported
// (cfitsio's encoder family stops at i64 internal driven by a 32-bit
// input; there is no 64-bit-input HCOMPRESS variant in the FITS Tile
// Compression Convention).
//
// `scale` controls quantization: 0 or 1 = lossless (no digitization),
// >1 = lossy (each H-transform coefficient is divided by `scale`
// before encoding; reader undigitizes by multiplying back).
pub(crate) fn encode_hcompress(
    pixel_bytes_be: &[u8],
    nx_numpy: usize,
    ny_numpy: usize,
    bytepix: u32,
    zbitpix: i32,
    scale: i32,
) -> PyResult<Vec<u8>> {
    if !matches!(zbitpix, 8 | 16 | 32) {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS encode: unsupported ZBITPIX {} (must be 8/16/32; \
             ZBITPIX=64 is not supported by the encoder)",
            zbitpix
        )));
    }
    if !matches!(bytepix, 1 | 2 | 4) {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS encode: unsupported bytepix {} (must be 1/2/4)",
            bytepix
        )));
    }
    if scale < 0 {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS encode: scale must be >= 0, got {}", scale
        )));
    }
    let n_pixels = nx_numpy * ny_numpy;
    let expected = n_pixels.checked_mul(bytepix as usize)
        .ok_or_else(|| PyValueError::new_err(
            "HCOMPRESS encode: n_pixels * bytepix overflowed usize"
        ))?;
    if pixel_bytes_be.len() != expected {
        return Err(PyValueError::new_err(format!(
            "HCOMPRESS encode: input length {} != n_pixels * bytepix ({})",
            pixel_bytes_be.len(), expected
        )));
    }

    let mut writer = HWriter::new();
    if zbitpix == 32 {
        // 32-bit input → i64 internal (H-transform sums can overflow
        // i32).  Read big-endian i32, sign-extend to i64.
        let mut a: Vec<i64> = Vec::with_capacity(n_pixels);
        for i in 0..n_pixels {
            let off = i * 4;
            let bytes = [
                pixel_bytes_be[off],
                pixel_bytes_be[off + 1],
                pixel_bytes_be[off + 2],
                pixel_bytes_be[off + 3],
            ];
            a.push(i32::from_be_bytes(bytes) as i64);
        }
        htrans_i64(&mut a, nx_numpy, ny_numpy);
        digitize_i64(&mut a, scale);
        encode_i64(&mut writer, &mut a, nx_numpy, ny_numpy, scale);
    } else {
        // 8/16-bit input → i32 internal.  u8 zero-extends; i16
        // sign-extends.  cfitsio's encoder takes int* and the value
        // domain depends on caller's conversion.
        let mut a: Vec<i32> = Vec::with_capacity(n_pixels);
        match (bytepix, zbitpix) {
            (1, 8) => {
                for i in 0..n_pixels {
                    a.push(pixel_bytes_be[i] as i32);
                }
            }
            (2, 16) => {
                for i in 0..n_pixels {
                    let off = i * 2;
                    let bytes = [pixel_bytes_be[off], pixel_bytes_be[off + 1]];
                    a.push(i16::from_be_bytes(bytes) as i32);
                }
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "HCOMPRESS encode: bytepix/zbitpix mismatch \
                     (bytepix={}, zbitpix={})",
                    bytepix, zbitpix
                )));
            }
        }
        htrans_i32(&mut a, nx_numpy, ny_numpy);
        digitize_i32(&mut a, scale);
        encode_i32(&mut writer, &mut a, nx_numpy, ny_numpy, scale);
    }
    Ok(writer.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // qtree_copy on a tiny example: 4-bit code 0b1010 (=10) means
    // bits at s00 and s00+n+1 should be 1, others 0.
    #[test]
    fn qtree_copy_2x2_code_10() {
        // Input has a single 4-bit code at position 0; output is
        // a 2x2 expansion.
        let mut a = vec![0u8; 4];
        a[0] = 10; // 0b1010 → bit3=1, bit2=0, bit1=1, bit0=0
        qtree_copy(&mut a, 2, 2, 2);
        // After expansion: b[0,0] = bit3 = 1, b[0,1] = bit2 = 0,
        //                  b[1,0] = bit1 = 1, b[1,1] = bit0 = 0
        assert_eq!(a, vec![1u8, 0u8, 1u8, 0u8]);
    }

    // Huffman value 1 has code 0b000 (3 bits).  Build a 1-bit
    // stream containing just that code and verify input_huffman.
    #[test]
    fn input_huffman_value_1() {
        // value 1 ↔ 3-bit code 0b000 ↔ "1<<0".  Pack into one
        // byte aligned to the top.
        let bytes = [0b000_00000u8];
        let mut state = HState::new(&bytes);
        assert_eq!(state.input_huffman().unwrap(), 1);
    }

    #[test]
    fn input_huffman_value_2() {
        // value 2 ↔ 3-bit code 0b001 ↔ "1<<1".
        let bytes = [0b001_00000u8];
        let mut state = HState::new(&bytes);
        assert_eq!(state.input_huffman().unwrap(), 2);
    }

    // Magic bytes round trip via qread.
    #[test]
    fn qread_magic() {
        let bytes = [0xDDu8, 0x99, 0xAA];
        let mut state = HState::new(&bytes);
        let mut m = [0u8; 2];
        state.qread(&mut m).unwrap();
        assert_eq!(m, [0xDDu8, 0x99]);
    }
}

