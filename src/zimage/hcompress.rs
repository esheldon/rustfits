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
    let nx2 = (nx + 1) / 2;
    let ny2 = (ny + 1) / 2;
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
    let count = ((nqx + 1) / 2) * ((nqy + 1) / 2);
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
    let nqx2 = (nqx + 1) / 2;
    let nqy2 = (nqy + 1) / 2;
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
    let nx2 = (nx + 1) / 2;
    let ny2 = (ny + 1) / 2;
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
        if *v != T::default() {
            if state.input_bit()? != 0 {
                *v = -*v;
            }
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
// Placeholder for the smoothing pass.  `decode_hcompress` rejects
// SMOOTH=1 up front, so neither i32 nor i64 hinv ever calls
// hsmooth in practice — the function signatures exist so the hinv
// body can stay structurally identical to cfitsio.  When SMOOTH
// support lands, replace these bodies with the full port of
// cfitsio's `hsmooth` / `hsmooth64`.
#[allow(unused_variables)]
fn hsmooth_i32(
    a: &mut [i32], nxtop: usize, nytop: usize, ny: usize, scale: i32,
) {
}

#[allow(unused_variables)]
fn hsmooth_i64(
    a: &mut [i64], nxtop: usize, nytop: usize, ny: usize, scale: i32,
) {
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
    let scratch_len = (nmax + 1) / 2;
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
    let scratch_len = (nmax + 1) / 2;
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
    if smooth {
        return Err(PyValueError::new_err(
            "HCOMPRESS: SMOOTH=1 is not yet implemented — the hsmooth \
             boundary clauses still need porting.  All real-world \
             HCOMPRESS files we've seen use SMOOTH=0; report if you \
             hit this in practice.",
        ));
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

