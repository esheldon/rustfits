// Shared primitives used by every module: file handles, taint flag, block
// constants, and the integer-keyword lookup used by both the header-parsing
// and image-shape paths.

use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use std::sync::{Arc, Mutex, MutexGuard};
use std::sync::atomic::{AtomicBool, Ordering};

// Shared, mutable file handle.  FITS owns the master Arc, each HDU clones it.
// `None` after close().
pub(crate) type FileHandle = Arc<Mutex<Option<std::fs::File>>>;

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
