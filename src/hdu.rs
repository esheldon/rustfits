// Base HDU pyclass — shared parent of ImageHDU, TableHDU, AsciiTableHDU.
//
// HDUs hold a clone of the FITS file handle plus shared offset state
// (HduOffsets + FileLayout) so write-back methods on subclasses can locate
// themselves on disk *and* react when an earlier HDU's grow shifts them
// forward.  `#[new]` is intentionally omitted: instances are constructed
// only via FITS internals.

use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use std::sync::{Arc, Mutex, MutexGuard};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::common::{
    check_not_tainted, parse_keyword, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, TaintFlag,
};
use crate::header::FITSHeader;

// Smart-guard for the cards Vec.  Holds the cards mutex AND a
// reference to the per-HDU `cards_version` atomic counter.  The only
// way to mutate the cards is `commit(new_cards)`, which atomically
// replaces the Vec and bumps the version — making it impossible to
// forget a version bump on a successful mutation.  Read-only access
// is via `as_slice()` and `clone_cards()`; there is no `DerefMut`.
//
// If the caller drops the guard without calling `commit`, no version
// bump and no cards replacement happens — correct for the "decided
// not to mutate after all" path (e.g. a validation error before any
// disk write).
//
// Memory ordering: `commit` does the version bump with Release while
// still holding the cards mutex.  Any cache reader using Acquire on
// the version observes the post-mutation Vec because the cards mutex
// would have to be re-acquired (and released by us) for the reader
// to even reach the new cards.
pub(crate) struct CardsWriteGuard<'a> {
    inner: MutexGuard<'a, Vec<String>>,
    version: &'a AtomicU64,
}

impl CardsWriteGuard<'_> {
    /// Read-only view of the current cards.  Used by future cache
    /// readers that need to parse fresh metadata on miss without
    /// dropping the lock.
    #[allow(dead_code)]
    pub(crate) fn as_slice(&self) -> &[String] {
        &self.inner
    }

    /// Clone the current cards out so the caller can stage a modified
    /// replacement.  Typical pattern:
    ///
    ///   let g = super_.cards_write_lock()?;
    ///   let mut new_cards = g.clone_cards();
    ///   // ... mutate new_cards, do disk I/O ...
    ///   g.commit(new_cards);
    pub(crate) fn clone_cards(&self) -> Vec<String> {
        self.inner.clone()
    }

    /// Atomically replace the cards Vec and bump the version counter.
    /// Consumes the guard so a single commit closes the write window.
    pub(crate) fn commit(mut self, new_cards: Vec<String>) {
        *self.inner = new_cards;
        self.version.fetch_add(1, Ordering::Release);
    }
}

/// Base class for every FITS Header-Data Unit (HDU).
///
/// All HDU types — :class:`ImageHDU`, :class:`TableHDU`,
/// :class:`CompressedImageHDU`, :class:`CompressedTableHDU`,
/// :class:`AsciiTableHDU` — inherit from this class and share the
/// header access, identity, and "do you have data?" surface
/// defined here.
///
/// HDU instances are produced by indexing a :class:`FITS` object
/// (``hdu = fits[1]``) or iterating it (``for hdu in fits: ...``).
/// They cannot be constructed directly from Python; the file
/// object owns the on-disk layout.
///
/// The shared inherited surface is small on purpose — most of the
/// useful methods live on the subclasses, where the data layout
/// is known.
// `HDU` is the public Python base-class name; the acronym can't be
// lowercased without breaking the API.
#[allow(clippy::upper_case_acronyms)]
#[pyclass(subclass)]
pub(crate) struct HDU {
    // Shared with FITSHeader so mutations through the header view propagate
    // back to the HDU's canonical card list (and any other readers).
    pub(crate) header: Arc<Mutex<Vec<String>>>,
    // Monotonic counter, bumped (via `bump_cards_version`, called from
    // `rewrite_header_to_disk` on every successful disk-and-commit) every
    // time the cards Vec mutates.  Shared with FITSHeader so mutations
    // through the header view propagate to caches keyed off this counter
    // on the HDU (planned per-HDU-type Meta caches read cur_version,
    // compare against the cached version, and re-parse on miss).
    pub(crate) cards_version: Arc<AtomicU64>,
    pub(crate) index: usize,
    // Source filename, cloned in from FITS at construction.  Used only
    // by __repr__; size is negligible compared to header storage.
    pub(crate) filename: String,
    // Shared with FITSHeader and FITS so grows update everyone in lockstep.
    pub(crate) offsets: Arc<HduOffsets>,
    pub(crate) layout: Arc<FileLayout>,
    pub(crate) file: FileHandle,
    pub(crate) tainted: TaintFlag,
}

impl HDU {
    pub(crate) fn new(
        header: Vec<String>,
        index: usize,
        filename: String,
        offsets: Arc<HduOffsets>,
        layout: Arc<FileLayout>,
        file: FileHandle,
        tainted: TaintFlag,
    ) -> Self {
        HDU {
            header: Arc::new(Mutex::new(header)),
            cards_version: Arc::new(AtomicU64::new(0)),
            index,
            filename,
            offsets,
            layout,
            file,
            tainted,
        }
    }

    // Snapshot of the current header cards.  Takes the lock briefly, clones
    // out, releases.
    pub(crate) fn header_snapshot(&self) -> PyResult<Vec<String>> {
        check_not_tainted(&self.tainted)?;
        let g = self.header.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        Ok(g.clone())
    }

    // Acquire the cards mutex for a mutation.  The returned guard's
    // ONLY mutation path is `commit(new_cards)`, which atomically
    // replaces the cards Vec and bumps the version counter — so any
    // metadata cache keyed off `cards_version` invalidates correctly
    // without per-callsite vigilance.  See `CardsWriteGuard` above
    // for the read-only accessors and the commit semantics.
    pub(crate) fn cards_write_lock(&self) -> PyResult<CardsWriteGuard<'_>> {
        let inner = self.header.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        Ok(CardsWriteGuard { inner, version: &self.cards_version })
    }
}

impl CardsWriteGuard<'_> {
    /// Construct a CardsWriteGuard from a raw MutexGuard plus the
    /// matching version counter.  Used by `FITSHeader::cards_write_lock`
    /// and similar helpers that can't go through `HDU::cards_write_lock`
    /// because they sit on a different pyclass.  Keep this surface
    /// internal: callers MUST use a `_write_lock()` accessor on the
    /// owning type rather than constructing a guard directly, to avoid
    /// version-counter mismatches.
    pub(crate) fn from_parts<'a>(
        inner: MutexGuard<'a, Vec<String>>,
        version: &'a AtomicU64,
    ) -> CardsWriteGuard<'a> {
        CardsWriteGuard { inner, version }
    }
}

#[pymethods]
impl HDU {
    // __repr__ is a pyo3 slot dunder — see the TableHDU.__len__ note
    // for why we don't put a docstring on it.  Default repr is the
    // bare class name + index; subclasses override with multi-line
    // fitsio-style output.
    fn __repr__(&self) -> String {
        format!("<HDU #{}>", self.index)
    }

    /// The HDU's :class:`FITSHeader`.
    ///
    /// Returns a live view of this HDU's header cards.  Mutations
    /// via the header object (``__setitem__``, ``__delitem__``,
    /// ``update``, ``add_comment``, ``add_history``,
    /// ``add_blank``) write through to disk immediately, following
    /// the disk-write-before-commit ordering documented on
    /// :class:`FITSHeader`.
    ///
    /// Reads are cheap; mutations may grow the reserved header
    /// blocks in place if the new card list exceeds the current
    /// allotment.
    #[getter]
    fn header(&self, py: Python<'_>) -> PyResult<Py<FITSHeader>> {
        Py::new(py, FITSHeader::from_state(
            Arc::clone(&self.header),
            Arc::clone(&self.file),
            Arc::clone(&self.offsets),
            Arc::clone(&self.layout),
            Arc::clone(&self.tainted),
            Arc::clone(&self.cards_version),
        ))
    }

    /// Test-only hook: flip the taint flag without an actual I/O
    /// failure.  Used by ``tests/test_header_taint.py`` to verify
    /// rejection semantics on hosts where producing a real mid-write
    /// failure is hard.  Underscored to signal "not a public API"
    /// — do not call from user code.
    fn _force_taint(&self) {
        self.tainted.store(true, Ordering::Release);
    }

    /// The HDU's 0-based position in its file.
    ///
    /// Stable for the lifetime of the :class:`FITS` object — even
    /// when an earlier HDU grows and shifts this HDU's bytes
    /// forward, the index is unchanged because the HDU is still
    /// at the same position in the file's HDU list.
    #[getter]
    pub(crate) fn index(&self) -> usize {
        self.index
    }

    /// ``EXTNAME`` header value, or ``None`` when the keyword is
    /// absent.
    ///
    /// EXTNAME is the user-visible name of the HDU (e.g.
    /// ``'SCI'``, ``'CATALOG'``).  Combined with :attr:`extver`
    /// it's the standard way to identify HDUs without relying on
    /// position-by-index.
    #[getter]
    fn extname(&self) -> PyResult<Option<String>> {
        let cards = self.header_snapshot()?;
        Ok(parse_string_keyword(&cards, "EXTNAME"))
    }

    /// ``EXTVER`` header value, defaulting to ``1`` when absent.
    ///
    /// Per the FITS standard, multiple HDUs may share an
    /// ``EXTNAME`` and are distinguished by ``EXTVER``.  Returns
    /// ``1`` rather than ``None`` for the absent case so callers
    /// can compare/select without handling ``Optional[int]``.
    #[getter]
    fn extver(&self) -> PyResult<i64> {
        let cards = self.header_snapshot()?;
        Ok(parse_keyword(&cards, "EXTVER").unwrap_or(1))
    }

    /// ``True`` iff this HDU has a non-empty data section.
    ///
    /// Works uniformly across image and table HDUs: the test is
    /// ``NAXIS > 0`` AND every ``NAXISn > 0``.  For images that
    /// means "at least one pixel"; for tables it means "at least
    /// one row of at least one column".
    ///
    /// Useful for picking the first HDU worth reading in a file
    /// (primary HDUs are often empty stubs)::
    ///
    ///     hdu = next(h for h in fits if h.has_data)
    ///     arr = hdu.read()
    ///
    /// Edge case: a VLA table with ``NAXIS2=0`` but ``PCOUNT>0``
    /// (heap-only) returns ``False`` — no main rows means there's
    /// nothing to interpret the heap through, which is the right
    /// answer for the "is this HDU worth reading?" question.
    #[getter]
    fn has_data(&self) -> PyResult<bool> {
        let cards = self.header_snapshot()?;
        let naxis = parse_keyword(&cards, "NAXIS").unwrap_or(0);
        if naxis <= 0 {
            return Ok(false);
        }
        for i in 1..=naxis {
            let d = parse_keyword(&cards, &format!("NAXIS{}", i))
                .unwrap_or(0);
            if d <= 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
