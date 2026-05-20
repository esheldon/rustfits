// Base HDU pyclass — shared parent of ImageHDU, TableHDU, AsciiTableHDU.
//
// HDUs hold a clone of the FITS file handle plus shared offset state
// (HduOffsets + FileLayout) so write-back methods on subclasses can locate
// themselves on disk *and* react when an earlier HDU's grow shifts them
// forward.  `#[new]` is intentionally omitted: instances are constructed
// only via FITS internals.

use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;

use crate::common::{
    check_not_tainted, parse_keyword, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, TaintFlag,
};
use crate::header::FITSHeader;

#[pyclass(subclass)]
pub(crate) struct HDU {
    // Shared with FITSHeader so mutations through the header view propagate
    // back to the HDU's canonical card list (and any other readers).
    pub(crate) header: Arc<Mutex<Vec<String>>>,
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
}

#[pymethods]
impl HDU {
    fn __repr__(&self) -> String {
        format!("<HDU #{}>", self.index)
    }

    #[getter]
    fn header(&self, py: Python<'_>) -> PyResult<Py<FITSHeader>> {
        Py::new(py, FITSHeader::from_state(
            Arc::clone(&self.header),
            Arc::clone(&self.file),
            Arc::clone(&self.offsets),
            Arc::clone(&self.layout),
            Arc::clone(&self.tainted),
        ))
    }

    // Test-only hook: flip the taint flag without an actual disk failure.
    // Used by tests/test_header_taint.py to verify rejection semantics
    // without producing a real I/O failure on the host filesystem.
    fn _force_taint(&self) {
        self.tainted.store(true, Ordering::Release);
    }

    #[getter]
    pub(crate) fn index(&self) -> usize {
        self.index
    }

    // EXTNAME header value, or None when the keyword is absent.
    // Inherited by every HDU subclass.
    #[getter]
    fn extname(&self) -> PyResult<Option<String>> {
        let cards = self.header_snapshot()?;
        Ok(parse_string_keyword(&cards, "EXTNAME"))
    }

    // EXTVER header value; default 1 per the FITS standard when the
    // keyword is absent.  Always returns a usable version number so
    // callers can compare/select without handling Optional[int].
    #[getter]
    fn extver(&self) -> PyResult<i64> {
        let cards = self.header_snapshot()?;
        Ok(parse_keyword(&cards, "EXTVER").unwrap_or(1))
    }

    // True iff this HDU has a non-empty data section: NAXIS > 0 and
    // every NAXISn > 0.  Works uniformly for image and table HDUs
    // (the FITS data-section formula is Π NAXISn pixels for images,
    // and row_width × nrows = NAXIS1 × NAXIS2 bytes for tables — both
    // collapse to "no NAXISn is zero").  Intended use: `fits.read()`-
    // style convenience that picks the first HDU worth reading, and
    // user code that wants a quick "is this HDU empty?" check.
    //
    // Heap-only edge case: a VLA table with NAXIS2=0 and PCOUNT>0
    // returns False (no rows to interpret the heap through), which
    // is the right call for read-convenience semantics.
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
