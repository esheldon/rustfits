// Shared primitives used by every module: file handles, taint flag, block
// constants, and the integer-keyword lookup used by both the header-parsing
// and image-shape paths.

use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
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
