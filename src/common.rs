// Shared primitives used by every module: file handles, taint flag, block
// constants, and the integer-keyword lookup used by both the header-parsing
// and image-shape paths.

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex, MutexGuard};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// The backing store for an open FITS file.  This is the `FitsStorage`
// seam from the storage-driver roadmap (see CLAUDE.md), realized as an
// enum rather than `Box<dyn ...>` so dispatch is monomorphized and every
// call site keeps working with a concrete `&mut Storage` (no vtable, no
// deref dance).  All random-access I/O flows through the std
// `Read`/`Write`/`Seek` impls below; only the three operations that
// `std::fs::File` exposes outside those traits — `set_len`, byte length,
// and durable sync — need explicit forwarding.
//
// `Disk` is the on-disk `file://` backend; `Mem` is the in-memory
// `mem://` / `memkeep://` backend (the two URL spellings are aliases —
// both land here, see CLAUDE.md).  A future lazy remote-range backend
// that can't be enumerated cheaply would be the moment to reconsider
// `dyn`.
pub(crate) enum Storage {
    Disk(std::fs::File),
    Mem(Cursor<Vec<u8>>),
}

impl Storage {
    // Truncate or zero-extend the backing store to `size` bytes.  Takes
    // `&mut self` (not `&self` like `File::set_len`) because the in-memory
    // backend resizes its `Vec`; every caller already holds `&mut Storage`
    // via `guard.as_mut()`.  The cursor position is left unchanged (matches
    // `File::set_len`, which never moves the file offset) — every read/write
    // site seeks explicitly anyway.
    pub(crate) fn set_len(&mut self, size: u64) -> io::Result<()> {
        match self {
            Storage::Disk(f) => f.set_len(size),
            Storage::Mem(c) => {
                c.get_mut().resize(size as usize, 0);
                Ok(())
            }
        }
    }

    // Current length of the backing store in bytes (replaces the old
    // `f.metadata()?.len()` pattern at every call site).
    pub(crate) fn len(&self) -> io::Result<u64> {
        match self {
            Storage::Disk(f) => Ok(f.metadata()?.len()),
            Storage::Mem(c) => Ok(c.get_ref().len() as u64),
        }
    }

    // Flush durable state: fsync on disk, no-op for an in-memory buffer.
    pub(crate) fn sync(&self) -> io::Result<()> {
        match self {
            Storage::Disk(f) => f.sync_all(),
            Storage::Mem(_) => Ok(()),
        }
    }

    // Copy the entire backing store out as a fresh Vec, from byte 0.
    // Backs `FITS.to_bytes()`.  Seeks (and leaves the cursor at EOF);
    // harmless because every other I/O site seeks before reading/writing.
    pub(crate) fn read_all(&mut self) -> io::Result<Vec<u8>> {
        let len = self.len()?;
        self.seek(SeekFrom::Start(0))?;
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        Ok(buf)
    }
}

impl Read for Storage {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Storage::Disk(f) => f.read(buf),
            Storage::Mem(c) => c.read(buf),
        }
    }
}

impl Write for Storage {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Storage::Disk(f) => f.write(buf),
            Storage::Mem(c) => c.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Storage::Disk(f) => f.flush(),
            Storage::Mem(c) => c.flush(),
        }
    }
}

impl Seek for Storage {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            Storage::Disk(f) => f.seek(pos),
            Storage::Mem(c) => c.seek(pos),
        }
    }
}

// Shared, mutable file handle.  FITS owns the master Arc, each HDU clones it.
// `None` after close().
pub(crate) type FileHandle = Arc<Mutex<Option<Storage>>>;

// Per-HDU byte offsets stored as atomics so they can be mutated in place
// when a header (or, later, an image/table data section) grows and shifts
// subsequent HDUs forward in the file.  Each HDU and each FITSHeader view
// holds an Arc to the same record, so updates here are visible everywhere.
//
// Invariant: `data_offset == header_offset + header_block_count * BLOCK_SIZE`
// at all times.  Mutators must update both fields atomically *together* under
// the file lock (the file lock is the serialization point for grow operations,
// so a single writer is guaranteed; readers tolerate transient inconsistency
// only because no read happens concurrently with a grow — the same file lock
// gates both).
pub(crate) struct HduOffsets {
    pub(crate) header_offset: AtomicU64,
    pub(crate) header_block_count: AtomicU64,
    pub(crate) data_offset: AtomicU64,
}

impl HduOffsets {
    pub(crate) fn new(
        header_offset: u64,
        header_block_count: u64,
        data_offset: u64,
    ) -> Arc<Self> {
        Arc::new(HduOffsets {
            header_offset: AtomicU64::new(header_offset),
            header_block_count: AtomicU64::new(header_block_count),
            data_offset: AtomicU64::new(data_offset),
        })
    }

    pub(crate) fn header_offset(&self) -> u64 {
        self.header_offset.load(Ordering::Acquire)
    }

    pub(crate) fn header_block_count(&self) -> u64 {
        self.header_block_count.load(Ordering::Acquire)
    }

    pub(crate) fn data_offset(&self) -> u64 {
        self.data_offset.load(Ordering::Acquire)
    }
}

// File-wide layout: the per-HDU offset records for every HDU in order.
// Shared between FITS (which appends on create_image_hdu) and every HDU /
// FITSHeader (which walks subsequent HDUs during a grow).  The Mutex
// protects the Vec itself; individual HduOffsets are lock-free atomics.
pub(crate) struct FileLayout {
    pub(crate) hdus: Mutex<Vec<Arc<HduOffsets>>>,
}

impl FileLayout {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(FileLayout {
            hdus: Mutex::new(Vec::new()),
        })
    }
}

pub(crate) fn lock_file(
    handle: &FileHandle,
) -> PyResult<MutexGuard<'_, Option<Storage>>> {
    handle.lock().map_err(|_| PyIOError::new_err("file lock poisoned"))
}

// Per-FITS-file taint flag.  Set when a mid-write disk error inside
// `rewrite_header_to_disk` may have left the on-disk header partially
// overwritten.  Subsequent header / image reads and writes refuse with a
// clear error until the file is closed and reopened.  One flag is shared
// across all HDUs of the same file (and across all FITSHeader views).
pub(crate) type TaintFlag = Arc<AtomicBool>;

pub(crate) fn check_not_tainted(tainted: &TaintFlag) -> PyResult<()> {
    if tainted.load(Ordering::Acquire) {
        return Err(PyIOError::new_err(
            "this FITS file is in an indeterminate state after a mid-write \
             I/O failure; the on-disk header may be partially overwritten — \
             close the FITS object and reopen the file to recover"
        ));
    }
    Ok(())
}

pub(crate) const BLOCK_SIZE: usize = 2880;
pub(crate) const CARD_SIZE: usize = 80;
// 36 cards per 2880-byte header block (BLOCK_SIZE / CARD_SIZE).  This is
// fixed by the FITS standard and shows up everywhere we convert between a
// card count and the number of reserved blocks needed to hold it.
pub(crate) const CARDS_PER_BLOCK: usize = BLOCK_SIZE / CARD_SIZE;

// Shift every byte in the range [after_offset..EOF] forward by `delta` bytes,
// growing the file as needed, and update every HDU offset record in `layout`
// whose `header_offset >= after_offset` to reflect the shift.  The caller's
// own HDU is intentionally NOT touched here — its `header_offset` is strictly
// less than `after_offset` for every current caller (header grow inserts at
// self.data_offset; image/table data grow inserts at self.data_offset +
// data_size) — so updating self's other fields (header_block_count, possibly
// data_offset) is the caller's responsibility.
//
// ----- Why the back-to-front copy is safe -----
//
// Source range:      [after_offset .. original_len)  (length = tail_len)
// Destination range: [after_offset + delta .. original_len + delta)
//
// Because `delta > 0`, source and destination overlap in the middle of the
// file.  We must move bytes in an order that never writes over source bytes
// before they have been read.
//
// First, `set_len(new_len)` extends the file to its final size.  The newly
// allocated bytes at [original_len .. new_len) are sparse / unread and
// contain no data we care about.
//
// We then copy the tail in fixed-size chunks, working back-to-front (last
// chunk of the tail first).  For each chunk we read its source bytes into
// a buffer, then write the buffer to the destination.  This is safe at two
// levels:
//
//   1. WITHIN a chunk: the chunk's destination range may overlap its
//      source range, but since we read the source bytes into an in-memory
//      buffer first and then write from the buffer, the overlap is harmless.
//
//   2. ACROSS chunks: when we move chunk k (source [Sk, Sk+n), destination
//      [Sk+delta, Sk+n+delta)), the next chunk we will read is the chunk
//      immediately before it in the file (source [Sk-n, Sk)).  The earliest
//      byte chunk k writes is at Sk+delta, which is >= Sk (delta >= 0), so
//      chunk k's writes never reach into [Sk-n, Sk) — the source range of
//      any earlier chunk we have not yet read.  The first chunk's writes
//      land partly in the freshly extended space [original_len, new_len),
//      which is virgin.
//
// Working forward would violate property (2): chunk 0's write would land
// at [after_offset+delta, after_offset+n+delta), which overlaps chunk 1's
// source range [after_offset+n, after_offset+2n) whenever delta < n.
//
// Taint semantics: failures before any byte has moved (lock acquisition,
// metadata, initial set_len) MUST NOT taint — the file is untouched.  Any
// failure inside the shift loop, or in the post-shift flush, taints because
// the file may now be inconsistent and the user has to reopen.
pub(crate) fn shift_file_tail_and_update_offsets(
    file_handle: &FileHandle,
    layout: &FileLayout,
    after_offset: u64,
    delta: u64,
    tainted: &TaintFlag,
) -> PyResult<()> {
    if delta == 0 {
        return Ok(());
    }

    let mut guard = lock_file(file_handle)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    let original_len = f.len()
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    // `tail_len` is the number of pre-existing bytes that must be relocated
    // (everything from after_offset to EOF).  When growing the header of
    // the LAST HDU of a file whose data section is empty (NAXIS=0) the
    // condition `after_offset == original_len` holds and tail_len == 0:
    // there is nothing to move and we only need to grow the file.  The
    // saturating_sub also guards the theoretical `after_offset > original_len`
    // case (caller bug), which would otherwise underflow.
    //
    // `new_len` is the post-shift file size.  Almost always
    // `original_len + delta` is correct (we're inserting `delta` bytes in
    // the middle).  The `max(after_offset + delta)` term covers the
    // edge case where `after_offset` itself lies past `original_len` —
    // there we still must allocate at least up through the destination
    // range so the writes below are addressable.
    let tail_len = original_len.saturating_sub(after_offset);
    let new_len = original_len.saturating_add(delta).max(after_offset + delta);

    // set_len before any writes so the destination range is addressable;
    // failure here happens *before* any byte movement so we do not taint.
    f.set_len(new_len)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    if tail_len > 0 {
        const CHUNK: u64 = 1 << 20;
        let mut buf = vec![0u8; CHUNK as usize];
        let mut remaining = tail_len;
        while remaining > 0 {
            let n = std::cmp::min(remaining, CHUNK);
            // `src` is the start of the chunk currently being relocated;
            // `dst` is where it lands.  See the "back-to-front" argument
            // in the function-level comment: writes here can only overlap
            // bytes that have already been read (this chunk's source) or
            // bytes in the freshly extended region — never the source of
            // any chunk we have not yet read.
            let src = after_offset + remaining - n;
            let dst = src + delta;

            f.seek(SeekFrom::Start(src)).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "seek failed during file shift: {}; \
                     the on-disk file is inconsistent — \
                     close this FITS object and reopen the file to recover", e
                ))
            })?;
            let chunk = &mut buf[..n as usize];
            f.read_exact(chunk).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "read failed during file shift: {}; \
                     the on-disk file is inconsistent — \
                     close this FITS object and reopen the file to recover", e
                ))
            })?;

            f.seek(SeekFrom::Start(dst)).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "seek failed during file shift: {}; \
                     the on-disk file is inconsistent — \
                     close this FITS object and reopen the file to recover", e
                ))
            })?;
            f.write_all(chunk).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "write failed during file shift: {}; \
                     the on-disk file is inconsistent — \
                     close this FITS object and reopen the file to recover", e
                ))
            })?;

            remaining -= n;
        }

        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "flush failed after file shift: {}; \
                 the on-disk file may be inconsistent — \
                 close this FITS object and reopen the file to recover", e
            ))
        })?;
    }

    // Update offsets while still holding the file lock so any reader who
    // wakes up next will see both the new file contents and new offsets.
    let layout_guard = layout.hdus.lock()
        .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
    for hdu in layout_guard.iter() {
        if hdu.header_offset() >= after_offset {
            hdu.header_offset.fetch_add(delta, Ordering::Release);
            hdu.data_offset.fetch_add(delta, Ordering::Release);
        }
    }

    Ok(())
}

// Zero-fill `[start..start+len)` in the file, chunked.  The caller is
// responsible for having grown the file to cover this range (e.g. via
// shift_file_tail_and_update_offsets, which `set_len`s up to new_len).
// Mid-write failures taint — by the time we're here, an earlier shift has
// already mutated the file layout and any inconsistency means the file is
// not safely reopenable without close+reopen.
//
// Used by the image-extend grow path: after shift_file_tail relocates the
// next HDU(s) forward, the gap [old_end_of_self_data .. old_end + delta)
// contains the original first delta bytes of the shifted tail (see the
// "back-to-front copy" argument in shift_file_tail's doc-comment); we
// must overwrite those bytes with zeros so reading the now-grown image
// returns FITS-conforming padding rather than stray header bytes.
pub(crate) fn zero_fill_range(
    file_handle: &FileHandle,
    start: u64,
    len: u64,
    tainted: &TaintFlag,
) -> PyResult<()> {
    if len == 0 {
        return Ok(());
    }
    let mut guard = lock_file(file_handle)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    f.seek(SeekFrom::Start(start)).map_err(|e| {
        tainted.store(true, Ordering::Release);
        PyIOError::new_err(format!(
            "seek failed during zero-fill: {}; \
             the on-disk file is inconsistent — \
             close this FITS object and reopen the file to recover", e
        ))
    })?;
    const CHUNK: usize = 1 << 20;
    let buf = vec![0u8; CHUNK];
    let mut remaining = len as usize;
    while remaining > 0 {
        let n = std::cmp::min(remaining, CHUNK);
        f.write_all(&buf[..n]).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "write failed during zero-fill: {}; \
                 the on-disk file is inconsistent — \
                 close this FITS object and reopen the file to recover", e
            ))
        })?;
        remaining -= n;
    }
    f.flush().map_err(|e| {
        tainted.store(true, Ordering::Release);
        PyIOError::new_err(format!(
            "flush failed after zero-fill: {}; \
             the on-disk file may be inconsistent — \
             close this FITS object and reopen the file to recover", e
        ))
    })?;
    Ok(())
}

// Shift every byte in [old_after_offset..EOF] BACKWARD by `delta` bytes,
// truncate the file to its new (smaller) size, and decrement every HDU
// offset in `layout` whose `header_offset >= old_after_offset` by `delta`.
// Mirror of `shift_file_tail_and_update_offsets` for the shrink direction;
// used by the repack/compact path when an HDU's data section gets smaller
// and the file tail must move forward to reclaim the freed space.
//
// Source range:      [old_after_offset .. original_len)
// Destination range: [old_after_offset - delta .. original_len - delta)
//
// Because the destination starts BEFORE the source, a forward-walking
// copy is safe: chunk k's write at [dst_k .. dst_k + n) cannot overlap
// any chunk we haven't yet read (which all sit at offsets > dst_k + n).
// The opposite-direction primitive needs back-to-front; this one needs
// front-to-back.
//
// Taint semantics: pre-loop failures (lock acquisition, metadata) do
// NOT taint — the file is untouched.  Failures inside the loop, the
// post-loop flush, or the final set_len DO taint — the file may be
// inconsistent and the user must close + reopen.
pub(crate) fn shift_file_tail_backward_and_update_offsets(
    file_handle: &FileHandle,
    layout: &FileLayout,
    old_after_offset: u64,
    delta: u64,
    tainted: &TaintFlag,
) -> PyResult<()> {
    if delta == 0 {
        return Ok(());
    }
    if delta > old_after_offset {
        return Err(PyIOError::new_err(
            "shift_file_tail_backward: delta exceeds the source offset"));
    }

    let mut guard = lock_file(file_handle)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let original_len = f.len()
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    let tail_len = original_len.saturating_sub(old_after_offset);

    if tail_len > 0 {
        const CHUNK: u64 = 1 << 20;
        let mut buf = vec![0u8; CHUNK as usize];
        let mut moved: u64 = 0;
        while moved < tail_len {
            let n = std::cmp::min(tail_len - moved, CHUNK);
            let src = old_after_offset + moved;
            let dst = src - delta;
            f.seek(SeekFrom::Start(src)).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "seek failed during backward shift: {}; \
                     close + reopen", e))
            })?;
            let chunk = &mut buf[..n as usize];
            f.read_exact(chunk).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "read failed during backward shift: {}; \
                     close + reopen", e))
            })?;
            f.seek(SeekFrom::Start(dst)).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "seek failed during backward shift: {}; \
                     close + reopen", e))
            })?;
            f.write_all(chunk).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "write failed during backward shift: {}; \
                     close + reopen", e))
            })?;
            moved += n;
        }
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "flush failed after backward shift: {}; close + reopen", e))
        })?;
    }

    let new_len = original_len - delta;
    f.set_len(new_len).map_err(|e| {
        tainted.store(true, Ordering::Release);
        PyIOError::new_err(format!(
            "set_len failed after backward shift: {}; close + reopen", e))
    })?;

    let layout_guard = layout.hdus.lock()
        .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
    for hdu in layout_guard.iter() {
        if hdu.header_offset() >= old_after_offset {
            hdu.header_offset.fetch_sub(delta, Ordering::Release);
            hdu.data_offset.fetch_sub(delta, Ordering::Release);
        }
    }
    Ok(())
}

// Strict match: the keyword field in cols 1-8 (trimmed) must equal `key`, and
// col 9 must be `=`.  This avoids the trap that `starts_with("NAXIS")` would
// also match `NAXIS1`, `NAXIS2`, etc.
pub(crate) fn parse_keyword(cards: &[String], key: &str) -> Option<i64> {
    for card in cards {
        if card.len() < 9 { continue; }
        if card[..8].trim() != key { continue; }
        if !card[8..].starts_with('=') { continue; }
        let value_part = &card[9..];
        if let Some(num_str) = value_part.split_whitespace().next() {
            let cleaned = num_str.trim_end_matches(&['\'', ' ', '/'][..]);
            return cleaned.parse::<i64>().ok();
        }
    }
    None
}

// Same matching rules as parse_keyword, but the value is parsed as
// f64.  TSCAL/TZERO are float-typed keywords in the FITS standard,
// and the unsigned-int trick TZERO values (-128, 2^15, 2^31, 2^63)
// are all exact in f64 since they're powers of 2.
pub(crate) fn parse_keyword_float(cards: &[String], key: &str) -> Option<f64> {
    for card in cards {
        if card.len() < 9 { continue; }
        if card[..8].trim() != key { continue; }
        if !card[8..].starts_with('=') { continue; }
        let value_part = &card[9..];
        if let Some(num_str) = value_part.split_whitespace().next() {
            let cleaned = num_str.trim_end_matches(&['\'', ' ', '/'][..]);
            return cleaned.parse::<f64>().ok();
        }
    }
    None
}

// Extract the string value for the given keyword from a card list.  The
// card's keyword field (cols 1-8 trimmed) must match `key` exactly and col 9
// must be `=`.  Inner `''` is unescaped to `'`, then trailing spaces are
// stripped (per the FITS standard, trailing spaces in a string value are
// not significant).  Returns None if there is no such card or the value
// isn't a quoted string.
pub(crate) fn parse_string_keyword(cards: &[String], key: &str) -> Option<String> {
    for card in cards {
        if card.len() < 9 { continue; }
        if card[..8].trim() != key { continue; }
        if !card[8..].starts_with('=') { continue; }
        let value_part = card[9..].trim_start();
        if !value_part.starts_with('\'') { return None; }
        let after_open = &value_part[1..];
        let bytes = after_open.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\'' {
                // `''` is the FITS escape for a single quote; skip both.
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                let inner = &after_open[..i];
                return Some(inner.replace("''", "'").trim_end().to_string());
            }
            i += 1;
        }
        return None;
    }
    None
}

// Reverse each `itemsize`-byte chunk in place.  Used to convert FITS
// big-endian numeric data to native little-endian after a raw read on
// little-endian hosts.  No-op when itemsize <= 1; callers also guard
// with `cfg!(target_endian = "big")` so this never runs on BE hosts
// (FITS is already native).
//
// Dispatched on itemsize so each branch can use primitive `swap_bytes`
// (a portable intrinsic LLVM lowers to BSWAP on x86-64, REV on ARM64,
// `rev8` on RISC-V Zbb, etc.).  Modern LLVM auto-vectorizes the
// resulting chunks_exact_mut::<uN> loop with the platform's byte-
// shuffle SIMD (PSHUFB on SSSE3/AVX2, NEON REV/TBL on ARM, V on
// RISC-V).  The fallback `chunk.reverse()` for unknown widths is
// ~5x slower because per-byte swap with a dynamic chunk length
// doesn't vectorize.
//
// itemsize=16 covers complex doubles (c16 = 2 × f8 components); the
// byteswap is *per 8-byte component*, NOT a full 16-byte reversal
// (which would swap real/imag).  Same for itemsize=8 vs c8 (2 × f4
// components) — but c8 callers pass itemsize=4 explicitly via
// byteswap_unit, so the 8-byte branch handles plain f8/i8 only.
pub(crate) fn byteswap_in_place(buf: &mut [u8], itemsize: usize) {
    match itemsize {
        0 | 1 => {}
        2 => swap_chunks_2(buf),
        4 => swap_chunks_4(buf),
        8 => swap_chunks_8(buf),
        16 => {
            // c16 = 2 × f8 components.  Swap each component, NOT the
            // whole 16-byte unit (which would flip real <-> imag).
            swap_chunks_8(buf);
        }
        _ => {
            // Should never fire for FITS data (no other itemsize is
            // legal).  Kept as a safety net.
            for chunk in buf.chunks_exact_mut(itemsize) {
                chunk.reverse();
            }
        }
    }
}

// Specialized helpers — `chunks_exact_mut::<u16/u32/u64>` returns
// aligned chunks the compiler vectorizes cleanly.  Each chunk's
// `swap_bytes` is a single BSWAP instruction; LLVM coalesces
// consecutive BSWAPs into PSHUFB on AVX2 / SSSE3 hosts.
#[inline]
fn swap_chunks_2(buf: &mut [u8]) {
    let (head, body, tail) = unsafe { buf.align_to_mut::<u16>() };
    for x in body {
        *x = x.swap_bytes();
    }
    debug_assert!(head.is_empty() && tail.is_empty(),
        "byteswap_in_place(itemsize=2): expected u16-aligned buffer");
    // numpy buffers are always primitive-aligned; head/tail handle
    // the pathological case for safety.
    for chunk in head.chunks_exact_mut(2) { chunk.reverse(); }
    for chunk in tail.chunks_exact_mut(2) { chunk.reverse(); }
}

#[inline]
fn swap_chunks_4(buf: &mut [u8]) {
    let (head, body, tail) = unsafe { buf.align_to_mut::<u32>() };
    for x in body {
        *x = x.swap_bytes();
    }
    debug_assert!(head.is_empty() && tail.is_empty(),
        "byteswap_in_place(itemsize=4): expected u32-aligned buffer");
    for chunk in head.chunks_exact_mut(4) { chunk.reverse(); }
    for chunk in tail.chunks_exact_mut(4) { chunk.reverse(); }
}

#[inline]
fn swap_chunks_8(buf: &mut [u8]) {
    let (head, body, tail) = unsafe { buf.align_to_mut::<u64>() };
    for x in body {
        *x = x.swap_bytes();
    }
    debug_assert!(head.is_empty() && tail.is_empty(),
        "byteswap_in_place(itemsize=8): expected u64-aligned buffer");
    for chunk in head.chunks_exact_mut(8) { chunk.reverse(); }
    for chunk in tail.chunks_exact_mut(8) { chunk.reverse(); }
}

// ===== RawBuffer: raw Py_buffer wrapper =====
//
// Holds a contiguous, mutable byte view into a Python object that supports
// the buffer protocol (numpy ndarrays, bytearrays, ...).  PyBuffer_Release
// runs on drop.  Used by both image read/write and binary-table read to
// move bytes between disk and numpy storage without going through Python
// element-by-element.

pub(crate) struct RawBuffer {
    view: Box<pyo3::ffi::Py_buffer>,
}

impl RawBuffer {
    fn acquire_with_flags(
        obj: &Bound<'_, PyAny>,
        flags: std::os::raw::c_int,
    ) -> PyResult<Self> {
        let mut view: Box<pyo3::ffi::Py_buffer> =
            Box::new(unsafe { std::mem::zeroed() });
        let rc = unsafe {
            pyo3::ffi::PyObject_GetBuffer(
                obj.as_ptr(),
                &mut *view as *mut _,
                flags,
            )
        };
        if rc != 0 {
            return Err(PyErr::take(obj.py()).unwrap_or_else(|| {
                PyValueError::new_err("buffer acquisition failed")
            }));
        }
        Ok(RawBuffer { view })
    }

    pub(crate) fn acquire(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::acquire_with_flags(obj, pyo3::ffi::PyBUF_C_CONTIGUOUS)
    }

    pub(crate) fn acquire_writable(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::acquire_with_flags(
            obj,
            pyo3::ffi::PyBUF_C_CONTIGUOUS | pyo3::ffi::PyBUF_WRITABLE,
        )
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.view.buf as *const u8,
                self.view.len as usize,
            )
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.view.buf as *mut u8,
                self.view.len as usize,
            )
        }
    }

    pub(crate) fn itemsize(&self) -> usize {
        self.view.itemsize as usize
    }

    pub(crate) fn len(&self) -> usize {
        self.view.len as usize
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        unsafe { pyo3::ffi::PyBuffer_Release(&mut *self.view) };
    }
}

// No-op context manager returned by `extending()` / `appending()`
// on the uncompressed `ImageHDU` and `TableHDU`.  Both subclasses
// (CompressedImageHDU + CompressedTableHDU) override those
// pymethods with their real buffering contexts via Python MRO;
// this no-op exists so generic code that iterates HDUs of mixed
// types can use the pattern uniformly without branching::
//
//     for hdu in fits:
//         with hdu.extending():
//             for batch in batches:
//                 hdu.extend(batch)
//
// The uncompressed extend / append paths don't have a partial-
// trailing-tile re-encode tax to amortize, so the context is
// genuinely a no-op: `__enter__` hands the user's HDU back so
// `as` works (`with hdu.extending() as h: ...`), and `__exit__`
// returns False so any in-flight exception propagates.
#[pyclass(module = "rustfits._rust")]
pub(crate) struct NoopExtendContext {
    pub(crate) hdu: Py<PyAny>,
}

#[pymethods]
impl NoopExtendContext {
    fn __enter__(&self, py: Python<'_>) -> Py<PyAny> {
        self.hdu.clone_ref(py)
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        false
    }
}
