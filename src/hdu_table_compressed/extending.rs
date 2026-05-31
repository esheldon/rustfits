// `hdu.appending()` (alias `extending()`) context manager for
// batched compressed-table appends.  Mirrors the ZIMAGE-side
// `hdu.extending()` design — same pattern, same 32 MB cap, same
// strict Path A semantics — but tailored to BINTABLE row inputs.
// See CLAUDE.md TODO #12 (ZTABLE follow-up) for the motivation
// and `src/hdu_image_compressed/extending.rs` for the parent
// design discussion (memory cap, why-Path-A, etc.).
//
// Pattern:
//
//     with hdu.appending():
//         for batch in batches:
//             hdu.append(batch)
//
// Inside the `with` block every `append()` call validates its
// input through `extract_per_column_inputs` (so dict / structured
// / list+names all normalize to per-column ndarrays) and pushes
// the per-column ndarrays onto an in-memory buffer.  When the
// buffer would exceed `MAX_PENDING_BYTES` (32 MB), the largest
// ZTILELEN-aligned slice drains via the existing
// `append_compressed_table_data` code; any sub-tile residual
// stays buffered.  On `__exit__` (normal or exceptional) the
// residual drains — the only call that pays a partial-trailing-
// tile merge-and-re-encode.
//
// Path A semantics (strict):
//   - Only `append()` / `extend()` is permitted inside the
//     context.  Any read / `write` / `__getitem__` /
//     `__setitem__` / `repack` / `add_checksum` /
//     `verify_checksum` raises `ValueError`.  Discipline is
//     enforced by `check_not_in_context` calls at the entry of
//     those methods (see `hdu.rs`).
//   - `FITS.close()` raises if any HDU is still in a context
//     (see `fits.rs`).

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use crate::common::check_not_tainted;
use crate::hdu_table::extract_per_column_inputs;

use super::append::append_compressed_table_data;
use super::hdu::CompressedTableHDU;

// Soft cap on buffered bytes before a tile-aligned mid-context
// drain triggers.  Same 32 MiB cap as the ZIMAGE side — the
// design rationale (small enough to bound RSS in long streaming
// loops, big enough to fit many ZTILELEN-row tiles per drain
// and amortize per-call append overhead) is unchanged.  See
// `src/hdu_image_compressed/extending.rs::MAX_PENDING_BYTES`.
pub(crate) const MAX_PENDING_BYTES: u64 = 32 * 1024 * 1024;

// In-flight buffer of pending `append()` inputs.  Each chunk is
// stored as a per-column `Vec<Py<PyAny>>` in canonical column
// order (matching `meta.columns`), so all input shapes
// (structured ndarray, dict, list+names) normalize to the same
// shape at append time and the drain step can just
// `np.concatenate` per column.
pub(crate) struct PendingBuffer {
    pub(crate) chunks: Vec<Vec<Py<PyAny>>>,
    pub(crate) total_rows: u64,
    pub(crate) total_bytes: u64,
}

impl PendingBuffer {
    pub(crate) fn new() -> Self {
        PendingBuffer {
            chunks: Vec::new(),
            total_rows: 0,
            total_bytes: 0,
        }
    }
}

// True when an `appending()` context is currently open on this
// HDU.  Used by the data-mutating / data-reading pymethods to
// refuse operations that would race the pending buffer.
pub(crate) fn is_in_context(
    pending: &Arc<Mutex<Option<PendingBuffer>>>,
) -> PyResult<bool> {
    Ok(pending
        .lock()
        .map_err(|_| PyIOError::new_err("pending buffer lock poisoned"))?
        .is_some())
}

// Raise if an `appending()` context is open.  Called from every
// public method that touches HDU data besides `append()` /
// `extend()`.  Cheap: one Mutex lock + Option::is_none in the
// common path.
pub(crate) fn check_not_in_context(
    pending: &Arc<Mutex<Option<PendingBuffer>>>,
) -> PyResult<()> {
    if is_in_context(pending)? {
        return Err(PyValueError::new_err(
            "operation not permitted while inside hdu.appending() \
             context; exit the context first",
        ));
    }
    Ok(())
}

// Sum the `.nbytes` of every ndarray in `per_column`.
fn per_column_bytes(
    py: Python<'_>,
    per_column: &[Bound<'_, PyAny>],
) -> PyResult<u64> {
    let mut total: u64 = 0;
    for c in per_column {
        total += c.getattr("nbytes")?.extract::<u64>()?;
    }
    let _ = py; // suppress unused-py warning if any
    Ok(total)
}

// Append one batch into the pending buffer.  Called by the
// `append()` pymethod when in context.  After appending, may
// trigger a tile-aligned mid-context drain if the buffer is over
// the RAM cap.  Inputs go through `extract_per_column_inputs`
// here so any shape (structured / dict / list+names) is
// validated and normalized to per-column ndarrays before being
// stored.
pub(crate) fn append_to_buffer(
    py: Python<'_>,
    hdu_obj: &Bound<'_, CompressedTableHDU>,
    data: &Bound<'_, PyAny>,
    names: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let hdu = hdu_obj.borrow();
    check_not_tainted(&hdu.as_super().as_super().tainted)?;
    let super_ = hdu.as_super().as_super();
    let meta = hdu.meta(super_)?;

    let per_column = extract_per_column_inputs(
        py, data, names, &meta.columns,
    )?;
    if per_column.is_empty() {
        return Err(PyValueError::new_err(
            "compressed appending(): no columns to write",
        ));
    }
    // All per-column arrays must report the same shape[0]; the
    // existing append code expects this too and would raise
    // later.  Catch here so the error fires at the append call
    // that produced the mismatch, not at __exit__.
    let rows: u64 = {
        let s: Vec<usize> = per_column[0].getattr("shape")?.extract()?;
        s.first().copied().unwrap_or(0) as u64
    };
    if rows == 0 {
        return Err(PyValueError::new_err(
            "compressed appending(): per-column data must have at \
             least one row",
        ));
    }
    for (i, c) in per_column.iter().enumerate().skip(1) {
        let s: Vec<usize> = c.getattr("shape")?.extract()?;
        let r = s.first().copied().unwrap_or(0) as u64;
        if r != rows {
            return Err(PyValueError::new_err(format!(
                "compressed appending(): column {} has {} rows but \
                 column 0 has {}",
                i, r, rows,
            )));
        }
    }
    let nbytes = per_column_bytes(py, &per_column)?;

    let chunk_owned: Vec<Py<PyAny>> =
        per_column.into_iter().map(|b| b.unbind()).collect();

    {
        let mut g = hdu.pending.lock().map_err(|_| {
            PyIOError::new_err("pending buffer lock poisoned")
        })?;
        let buf = g.as_mut().ok_or_else(|| {
            PyValueError::new_err(
                "internal: append_to_buffer called outside appending() \
                 context",
            )
        })?;
        buf.chunks.push(chunk_owned);
        buf.total_rows += rows;
        buf.total_bytes += nbytes;
    }

    // Drain if we crossed the cap.
    drop(hdu);
    drain_aligned_subset(py, hdu_obj)?;
    Ok(())
}

// Compute the largest ZTILELEN-aligned drain size given the
// current ZNAXIS2 (post-previous-drain) and how many rows are
// buffered.  An aligned drain ends on a ZTILELEN boundary so the
// next drain starts cleanly (no merge-with-existing-partial-
// tile re-encode).  Returns 0 if no aligned drain fits.
fn aligned_drain_rows(
    current_nrows: u64,
    ztilelen: u64,
    buffered: u64,
) -> u64 {
    if ztilelen == 0 || buffered == 0 {
        return 0;
    }
    let partial = current_nrows % ztilelen;
    let rows_to_fill = if partial == 0 { 0 } else { ztilelen - partial };
    if buffered < rows_to_fill {
        return 0;
    }
    let remainder = buffered - rows_to_fill;
    let full_tile_rows = remainder / ztilelen;
    rows_to_fill + full_tile_rows * ztilelen
}

// Pop the front `rows_to_take` rows out of the buffer, returning
// them as a per-column Vec of concatenated ndarrays.  Walks
// `chunks` from the front: takes whole chunks until the
// accumulator reaches the target, then slices the last chunk if
// needed (slicing applied uniformly across every column).  The
// buffer's `total_rows` / `total_bytes` are recomputed from the
// residual.
fn pop_front_rows(
    py: Python<'_>,
    buf: &mut PendingBuffer,
    rows_to_take: u64,
) -> PyResult<Vec<Py<PyAny>>> {
    let np = py.import("numpy")?;
    let ncols = buf
        .chunks
        .first()
        .map(|c| c.len())
        .unwrap_or(0);
    // For each column we accumulate the per-chunk slices into a
    // Vec, then concatenate once at the end.
    let mut per_col_taken: Vec<Vec<Py<PyAny>>> =
        (0..ncols).map(|_| Vec::new()).collect();
    let mut accum: u64 = 0;
    let mut consume_idx: usize = 0;

    while consume_idx < buf.chunks.len() && accum < rows_to_take {
        let chunk_rows: u64 = {
            let arr0 = buf.chunks[consume_idx][0].bind(py);
            let s: Vec<usize> = arr0.getattr("shape")?.extract()?;
            s[0] as u64
        };
        if accum + chunk_rows <= rows_to_take {
            // Whole chunk goes into the drain.
            for col_idx in 0..ncols {
                per_col_taken[col_idx]
                    .push(buf.chunks[consume_idx][col_idx].clone_ref(py));
            }
            accum += chunk_rows;
            consume_idx += 1;
        } else {
            // Slice the chunk: first `want` rows for the drain,
            // remainder stays in the chunk slot.
            let want = (rows_to_take - accum) as isize;
            let total = chunk_rows as isize;
            for col_idx in 0..ncols {
                let arr = buf.chunks[consume_idx][col_idx].bind(py);
                let head = arr.get_item(PySlice::new(py, 0, want, 1))?;
                let head_copy =
                    np.call_method1("ascontiguousarray", (head,))?;
                let tail = arr.get_item(PySlice::new(py, want, total, 1))?;
                let tail_copy =
                    np.call_method1("ascontiguousarray", (tail,))?;
                per_col_taken[col_idx].push(head_copy.unbind());
                buf.chunks[consume_idx][col_idx] = tail_copy.unbind();
            }
            // Don't advance consume_idx; chunks[consume_idx] now
            // holds the residual.
            break;
        }
    }

    // Discard the chunks fully consumed at the front.
    buf.chunks.drain(0..consume_idx);

    // Concatenate the per-column drain pieces.
    let mut concatenated: Vec<Py<PyAny>> = Vec::with_capacity(ncols);
    for col_taken in per_col_taken {
        if col_taken.len() == 1 {
            concatenated.push(col_taken.into_iter().next().unwrap());
        } else {
            let tup = PyTuple::new(
                py, col_taken.iter().map(|p| p.bind(py)),
            )?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("axis", 0)?;
            let cc = np.call_method("concatenate", (tup,), Some(&kwargs))?;
            concatenated.push(cc.unbind());
        }
    }

    // Recompute totals from what's left in the buffer.
    let mut new_rows: u64 = 0;
    let mut new_bytes: u64 = 0;
    for chunk in buf.chunks.iter() {
        if let Some(arr0) = chunk.first() {
            let s: Vec<usize> =
                arr0.bind(py).getattr("shape")?.extract()?;
            new_rows += s[0] as u64;
        }
        for c in chunk.iter() {
            new_bytes += c.bind(py).getattr("nbytes")?.extract::<u64>()?;
        }
    }
    buf.total_rows = new_rows;
    buf.total_bytes = new_bytes;

    Ok(concatenated)
}

// Run the existing `append_compressed_table_data` with a
// per-column drain slice.  Re-reads meta to pick up the current
// (post-previous-drain) NAXIS2 / PCOUNT / etc.  The meta cache
// auto-invalidates on each drain (cards bump via
// CardsWriteGuard) so subsequent drains see fresh values.
fn run_append_with_per_column(
    py: Python<'_>,
    hdu_obj: &Bound<'_, CompressedTableHDU>,
    per_column: Vec<Py<PyAny>>,
) -> PyResult<()> {
    let hdu = hdu_obj.borrow();
    let cfgs = hdu
        .compress_configs
        .lock()
        .map_err(|_| {
            PyIOError::new_err("compress_configs lock poisoned")
        })?
        .clone();
    let cache = Arc::clone(&hdu.cache);
    let super_ = hdu.as_super().as_super();
    let meta = hdu.meta(super_)?;
    let data_offset = super_.offsets.data_offset();

    // Rebind per_column as Bound for the call.  We keep the
    // Py<PyAny> alive through `per_column` so the borrowed
    // Bounds are valid for the duration of the call.
    let bound_per_column: Vec<Bound<'_, PyAny>> =
        per_column.iter().map(|p| p.bind(py).clone()).collect();

    append_compressed_table_data(
        py,
        super_,
        &meta.cards,
        &bound_per_column,
        &meta.columns,
        &meta.algorithms,
        cfgs.as_deref(),
        meta.nrows,
        meta.ztilelen,
        meta.n_tiles,
        meta.descriptor_row_width,
        data_offset,
        meta.current_pcount,
        &cache,
    )
}

// Drain whatever tile-aligned bytes fit while we're over the
// cap.  Called from `append_to_buffer` after each append.
// Reads ztilelen + current_nrows via the meta cache, so it sees
// the post-previous-drain state automatically.  No-op if under
// cap or no tile-aligned drain fits.
fn drain_aligned_subset(
    py: Python<'_>,
    hdu_obj: &Bound<'_, CompressedTableHDU>,
) -> PyResult<()> {
    // Cheap pre-check.
    {
        let hdu = hdu_obj.borrow();
        check_not_tainted(&hdu.as_super().as_super().tainted)?;
        let g = hdu.pending.lock().map_err(|_| {
            PyIOError::new_err("pending buffer lock poisoned")
        })?;
        let Some(buf) = g.as_ref() else { return Ok(()) };
        if buf.total_bytes <= MAX_PENDING_BYTES {
            return Ok(());
        }
    }

    // Look up ztilelen + current_nrows.
    let (ztilelen, current_nrows) = {
        let hdu = hdu_obj.borrow();
        let super_ = hdu.as_super().as_super();
        let meta = hdu.meta(super_)?;
        (meta.ztilelen as u64, meta.nrows as u64)
    };
    if ztilelen == 0 {
        return Ok(()); // degenerate; defer to __exit__ drain
    }

    // Extract the drain slice under the pending lock.
    let per_column: Vec<Py<PyAny>> = {
        let hdu = hdu_obj.borrow();
        let mut g = hdu.pending.lock().map_err(|_| {
            PyIOError::new_err("pending buffer lock poisoned")
        })?;
        let buf = g.as_mut().ok_or_else(|| {
            PyValueError::new_err(
                "internal: pending state vanished mid-drain",
            )
        })?;
        let drain_rows =
            aligned_drain_rows(current_nrows, ztilelen, buf.total_rows);
        if drain_rows == 0 {
            return Ok(());
        }
        pop_front_rows(py, buf, drain_rows)?
    };

    run_append_with_per_column(py, hdu_obj, per_column)
}

// Drain whatever residual is left in the buffer.  Called from
// `__exit__` (normal or exceptional).  Always resets `pending`
// to None before the append call so that append's own routing
// (which checks `is_in_context`) sees the post-context state.
// This is the only drain in a typical context that pays a
// merge-with-partial-trailing-tile cost (mid-context drains are
// ZTILELEN-aligned by construction).
fn drain_residual_and_exit(
    py: Python<'_>,
    hdu_obj: &Py<CompressedTableHDU>,
) -> PyResult<()> {
    let bound = hdu_obj.bind(py);

    let per_column: Vec<Py<PyAny>> = {
        let hdu = bound.borrow();
        check_not_tainted(&hdu.as_super().as_super().tainted)?;
        let taken = {
            let mut g = hdu.pending.lock().map_err(|_| {
                PyIOError::new_err("pending buffer lock poisoned")
            })?;
            g.take()
        };
        let Some(mut buf) = taken else { return Ok(()) };
        if buf.total_rows == 0 {
            return Ok(());
        }
        // Take everything that's left.
        let rows = buf.total_rows;
        pop_front_rows(py, &mut buf, rows)?
    };

    run_append_with_per_column(py, bound, per_column)
}

/// Context-manager handle returned by
/// :meth:`CompressedTableHDU.appending` (or its alias
/// :meth:`extending`).  Use via a ``with`` statement; the only
/// legal operations inside the block are
/// :meth:`CompressedTableHDU.append` and its
/// :meth:`extend` alias.
#[pyclass(module = "rustfits._rust")]
pub(crate) struct CompressedTableAppendContext {
    pub(crate) hdu: Py<CompressedTableHDU>,
}

#[pymethods]
impl CompressedTableAppendContext {
    fn __enter__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Py<CompressedTableHDU>> {
        let hdu = slf.hdu.bind(py).borrow();
        check_not_tainted(&hdu.as_super().as_super().tainted)?;
        let mut g = hdu.pending.lock().map_err(|_| {
            PyIOError::new_err("pending buffer lock poisoned")
        })?;
        if g.is_some() {
            return Err(PyValueError::new_err(
                "this HDU is already inside an appending() context",
            ));
        }
        *g = Some(PendingBuffer::new());
        Ok(slf.hdu.clone_ref(py))
    }

    fn __exit__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        // Always drain.  Returning False lets any exception in
        // flight propagate; a drain error itself replaces the
        // in-flight exception (Python prints "another exception
        // occurred during cleanup").
        drain_residual_and_exit(py, &slf.hdu)?;
        Ok(false)
    }
}
