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
    check_not_tainted, FileHandle, FileLayout, HduOffsets, TaintFlag,
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
}
