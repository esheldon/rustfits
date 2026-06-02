// Lazy row / chunk iteration over a TableHDU (and, via inheritance +
// polymorphic read dispatch, CompressedTableHDU).
//
// `TableIter` is a small stateful pyclass: Python drives it through
// `__next__`, and `Ok(None)` is how we raise StopIteration.  There is
// no Rust "generator" — the buffer + cursor live in the pyclass.
//
// One class covers both modes:
//   - row mode (chunksize=None): read `buffersize` rows into `buf`,
//     yield them one at a time as np.void records, refill when spent.
//   - chunk mode (chunksize=Some(n)): each `__next__` reads <= n rows
//     and yields the structured ndarray directly.
//
// Refills go through the HDU's OWN `read()` via `call_method`, so a
// `CompressedTableHDU` iterates correctly with zero compressed-specific
// code here (its `read` override handles per-tile decompression).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};

#[pyclass]
pub(crate) struct TableIter {
    // The parent HDU, kept as the most-derived Python object so that
    // `call_method("read")` / `len()` / `dtype` dispatch
    // polymorphically (the compressed subclass overrides all three).
    hdu: Py<PyAny>,
    // Row count snapshotted at construction — appends mid-iteration
    // are not seen.
    nrows: usize,
    // Global read cursor into the table.
    next_row: usize,
    // None => row mode; Some(n) => chunk mode (yield n-row arrays).
    chunksize: Option<usize>,
    // Rows per physical read in row mode (auto-sized byte budget).
    buffersize: usize,
    // Forwarded to read() on every refill.
    columns: Option<Py<PyAny>>,
    scale: bool,
    // Row-mode buffer state (idle in chunk mode).
    buf: Option<Py<PyAny>>,
    buf_len: usize,
    buf_cursor: usize,
}

impl TableIter {
    // Read rows [lo, hi) by dispatching through the HDU's own read(),
    // so CompressedTableHDU.read handles compressed tables with no
    // special-casing here.  Returns a structured ndarray.
    fn read_slice(
        &self,
        py: Python<'_>,
        lo: usize,
        hi: usize,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("rows", PySlice::new(py, lo as isize, hi as isize, 1))?;
        if let Some(cols) = &self.columns {
            kwargs.set_item("columns", cols.bind(py))?;
        }
        kwargs.set_item("scale", self.scale)?;
        Ok(self
            .hdu
            .bind(py)
            .call_method("read", (), Some(&kwargs))?
            .unbind())
    }
}

#[pymethods]
impl TableIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    // Remaining items still to be yielded.  Exposing __len__ lets
    // tqdm(hdu.iter()) show a real progress bar with total + ETA
    // instead of a bare counter.  Returns rows in row mode and
    // chunks in chunk mode (i.e. the count that matches what
    // subsequent __next__ calls will yield).
    fn __len__(&self) -> usize {
        let on_disk = self.nrows.saturating_sub(self.next_row);
        match self.chunksize {
            None => self.buf_len.saturating_sub(self.buf_cursor) + on_disk,
            Some(n) => on_disk.div_ceil(n),
        }
    }

    fn __next__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Option<Py<PyAny>>> {
        match slf.chunksize {
            // Chunk mode: each call is exactly one read of <= n rows.
            Some(n) => {
                if slf.next_row >= slf.nrows {
                    return Ok(None); // StopIteration
                }
                let lo = slf.next_row;
                let hi = (lo + n).min(slf.nrows);
                let arr = slf.read_slice(py, lo, hi)?;
                slf.next_row = hi;
                Ok(Some(arr))
            }
            // Row mode: refill the buffer when spent, then yield one
            // np.void record out of it.
            None => {
                if slf.buf_cursor >= slf.buf_len {
                    if slf.next_row >= slf.nrows {
                        return Ok(None); // StopIteration
                    }
                    let lo = slf.next_row;
                    let hi = (lo + slf.buffersize).min(slf.nrows);
                    let arr = slf.read_slice(py, lo, hi)?;
                    slf.buf_len = hi - lo;
                    slf.buf_cursor = 0;
                    slf.next_row = hi;
                    slf.buf = Some(arr);
                }
                let k = slf.buf_cursor;
                // Indexing a structured ndarray with a scalar int gives
                // a 0-d record (np.void) — matches hdu[i] semantics.
                let row =
                    slf.buf.as_ref().unwrap().bind(py).get_item(k)?.unbind();
                slf.buf_cursor += 1;
                Ok(Some(row))
            }
        }
    }
}

// Construct a TableIter for any tabular HDU pyclass (TableHDU,
// CompressedTableHDU, AsciiTableHDU).  All HDU access goes through
// `call_method` / `getattr` / `len`, so the signature accepts the
// generic `Bound<'_, PyAny>`.  `nrows` and `itemsize` are read
// POLYMORPHICALLY (`len()` / `dtype`) so subclasses that override
// those (e.g. CompressedTableHDU's uncompressed-view schema) get
// the right values without per-subclass code here.
pub(crate) fn make_table_iter(
    slf: Bound<'_, PyAny>,
    chunksize: Option<usize>,
    columns: Option<Py<PyAny>>,
    scale: bool,
) -> PyResult<TableIter> {
    if chunksize == Some(0) {
        return Err(PyValueError::new_err(
            "chunksize must be a positive integer; omit chunksize \
             (or pass None) to iterate one row at a time",
        ));
    }
    let nrows: usize = slf.len()?;
    let buffersize = match chunksize {
        Some(n) => n,
        None => {
            // 8 MiB byte budget -> rows, derived from the per-row
            // itemsize (the dtype a default read returns).  Bounds
            // resident memory regardless of row width.  VLA columns
            // count only their 8-byte object pointer in itemsize, so
            // the budget underestimates their true heap footprint —
            // an accepted approximation for a buffer heuristic.
            const BUDGET: usize = 8 << 20;
            let itemsize: usize =
                slf.getattr("dtype")?.getattr("itemsize")?.extract()?;
            (BUDGET / itemsize.max(1)).max(1)
        }
    };
    Ok(TableIter {
        hdu: slf.into_any().unbind(),
        nrows,
        next_row: 0,
        chunksize,
        buffersize,
        columns,
        scale,
        buf: None,
        buf_len: 0,
        buf_cursor: 0,
    })
}
