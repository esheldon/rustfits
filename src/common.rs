// Shared primitives used by every module: file handles, taint flag, block
// constants, and the integer-keyword lookup used by both the header-parsing
// and image-shape paths.

use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex, MutexGuard};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// Shared, mutable file handle.  FITS owns the master Arc, each HDU clones it.
// `None` after close().
pub(crate) type FileHandle = Arc<Mutex<Option<std::fs::File>>>;

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
) -> PyResult<MutexGuard<'_, Option<std::fs::File>>> {
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

    let original_len = f.metadata()
        .map_err(|e| PyIOError::new_err(e.to_string()))?
        .len();

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
