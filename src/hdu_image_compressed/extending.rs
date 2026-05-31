// `hdu.extending()` context manager for batched compressed-image
// extends.  See CLAUDE.md TODO #12 / "Compressed-image extend:
// sub-tile chunk re-encode tax" for the motivation.
//
// Pattern:
//
//     with hdu.extending():
//         for batch in batches:
//             hdu.extend(batch)
//
// Inside the `with` block every `extend()` call appends its input
// to an in-memory buffer.  Whenever the buffer would exceed
// `MAX_PENDING_BYTES` (32 MB), the largest tile-row-aligned
// portion is drained via the existing extend code; any sub-tile
// residual stays in the buffer.  On `__exit__` (normal or
// exceptional) the residual is drained (this is the only call
// that pays a partial-trailing-tile re-encode).
//
// Net effect: N partial-tile re-encodes collapse to 1, while
// peak buffered RAM stays bounded at ~32 MB regardless of how
// long the extend loop runs.  See `MAX_PENDING_BYTES` below for
// the rationale on the chosen cap.
//
// Path A semantics (strict):
//   - Only `extend()` is permitted inside the context.  Any
//     read / `__setitem__` / `repack` / `add_checksum` /
//     `verify_checksum` call while the context is open raises
//     `ValueError`.  Discipline is enforced by `check_not_in_context`
//     calls at the entry of those methods (see `hdu.rs`).
//   - `FITS.close()` raises if any HDU is still in a context
//     (see `fits.rs`).  In the natural nested-with pattern this
//     never fires because Python guarantees inner `__exit__`
//     runs first.  When it does fire it's a sign of forgotten
//     `__exit__()`.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};

use crate::common::check_not_tainted;

use super::hdu::CompressedImageHDU;
use super::write::extend_compressed_image_data;

// Soft cap on buffered bytes before a tile-aligned mid-context
// drain is triggered.  32 MiB is well above a typical tile size
// (a 100-row × 4 kpix f4 tile is 1.6 MB; 2-D survey tiles are
// usually 100×100 = 40 KB to a few MB) so the cap fits many
// tiles per drain — amortizing the per-call extend overhead.
// At the same time it's small enough that even a long-running
// streaming-extend loop stays in a tight RAM budget.
//
// At the cap the buffer holds at most ~32 MB of input ndarrays
// plus one tile-row's worth of residual; peak RSS during the
// drain itself can briefly reach ~2× the cap because the
// concatenate step allocates a fresh ndarray of the drain size.
//
// Documented in `docs/tutorial/performance.rst` under the
// "Compressed image extend (`extending()` context)" section.
pub(crate) const MAX_PENDING_BYTES: u64 = 32 * 1024 * 1024;

// In-flight buffer of pending `extend()` inputs.  Lives in
// `CompressedImageHDU::pending` as `Arc<Mutex<Option<PendingBuffer>>>`:
// `None` outside the context, `Some(empty)` immediately after
// `__enter__`, `Some(populated)` while extends accumulate, taken
// back to `None` by `__exit__`.
//
// `total_bytes` is updated incrementally on append so the cap
// check is O(1) per `extend()` call (no walk through `chunks`).
pub(crate) struct PendingBuffer {
    pub(crate) chunks: Vec<Py<PyAny>>,
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

// True when an `extending()` context is currently open on this HDU.
// Used by the data-mutating / data-reading pymethods to refuse
// operations that would race the pending buffer.
pub(crate) fn is_in_context(
    pending: &Arc<Mutex<Option<PendingBuffer>>>,
) -> PyResult<bool> {
    Ok(pending
        .lock()
        .map_err(|_| PyIOError::new_err("pending buffer lock poisoned"))?
        .is_some())
}

// Raise if an `extending()` context is open.  Called from every
// public method that touches HDU data besides `extend()` itself.
// Cheap: one Mutex lock + Option::is_none in the common path.
pub(crate) fn check_not_in_context(
    pending: &Arc<Mutex<Option<PendingBuffer>>>,
) -> PyResult<()> {
    if is_in_context(pending)? {
        return Err(PyValueError::new_err(
            "operation not permitted while inside hdu.extending() \
             context; exit the context first",
        ));
    }
    Ok(())
}

// Append one extend input to the pending buffer.  Called by the
// `extend()` pymethod when in context.  After appending, may
// trigger a tile-aligned mid-context drain if the buffer is over
// the RAM cap.  No shape validation: the drain step runs the
// full extend validator on the concatenated array.
pub(crate) fn append_to_buffer(
    py: Python<'_>,
    hdu_obj: &Bound<'_, CompressedImageHDU>,
    data: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (data,))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed extending(): input must have at least 1 axis",
        ));
    }
    let rows = shape[0] as u64;
    if rows == 0 {
        return Err(PyValueError::new_err(
            "compressed extending(): data.shape[0] must be > 0",
        ));
    }
    let nbytes: u64 = arr.getattr("nbytes")?.extract()?;

    let hdu = hdu_obj.borrow();
    {
        let mut g = hdu.pending.lock().map_err(|_| {
            PyIOError::new_err("pending buffer lock poisoned")
        })?;
        let buf = g.as_mut().ok_or_else(|| {
            PyValueError::new_err(
                "internal: append_to_buffer called outside extending() context",
            )
        })?;
        buf.chunks.push(arr.unbind());
        buf.total_rows += rows;
        buf.total_bytes += nbytes;
    }

    // Drain if we crossed the cap.  drain_aligned_subset
    // determines the largest tile-aligned slice to send through
    // the existing extend code; any sub-tile residual stays
    // buffered for either the next drain or the __exit__ final
    // drain.
    drop(hdu);
    drain_aligned_subset(py, hdu_obj)?;
    Ok(())
}

// Compute the largest tile-aligned drain size (in rows) given
// the on-disk image's current NAXIS0, the tile's slow-axis size,
// and how many rows are buffered.  An "aligned" drain means the
// post-drain NAXIS0 is a multiple of tile_rows — so each drain
// ends on a tile boundary and the next drain starts cleanly with
// no partial-trailing-tile re-encode cost.
//
// Returns 0 when no tile-aligned drain fits in `buffered` (e.g.
// tile_rows is bigger than the buffer); the caller should then
// leave the buffer alone and wait for more appends.
fn aligned_drain_rows(
    current_naxis0: u64,
    tile_rows: u64,
    buffered: u64,
) -> u64 {
    if tile_rows == 0 || buffered == 0 {
        return 0;
    }
    let partial = current_naxis0 % tile_rows;
    let rows_to_fill = if partial == 0 { 0 } else { tile_rows - partial };
    if buffered < rows_to_fill {
        return 0;
    }
    let remainder_after_fill = buffered - rows_to_fill;
    let full_tile_rows = remainder_after_fill / tile_rows;
    rows_to_fill + full_tile_rows * tile_rows
}

// Pop the front `rows_to_take` rows out of the buffer, returning
// them as a single contiguous ndarray.  Walks `chunks` from the
// front: takes whole chunks until the accumulator reaches the
// target, then slices the last chunk if needed.  The buffer's
// `total_rows` / `total_bytes` are updated to reflect what's
// left behind.
fn pop_front_rows(
    py: Python<'_>,
    buf: &mut PendingBuffer,
    rows_to_take: u64,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let mut taken: Vec<Py<PyAny>> = Vec::new();
    let mut accum: u64 = 0;
    let mut consume_idx: usize = 0;

    while consume_idx < buf.chunks.len() && accum < rows_to_take {
        let arr = buf.chunks[consume_idx].bind(py);
        let chunk_rows: u64 = {
            let s: Vec<usize> = arr.getattr("shape")?.extract()?;
            s[0] as u64
        };
        if accum + chunk_rows <= rows_to_take {
            // Whole chunk goes into the drain.
            taken.push(buf.chunks[consume_idx].clone_ref(py));
            accum += chunk_rows;
            consume_idx += 1;
        } else {
            // Slice the chunk: first `want` rows for the drain,
            // remainder stays in `buf.chunks[consume_idx]`.
            let want = (rows_to_take - accum) as isize;
            let total = chunk_rows as isize;
            let head_slice = PySlice::new(py, 0, want, 1);
            let head = arr.get_item(head_slice)?;
            // numpy slicing returns a view; force an owned copy so
            // the long residual storage isn't pinned by the small
            // drain head (and so the residual doesn't share memory
            // with the about-to-be-encoded drain array).
            let head_copy =
                np.call_method1("ascontiguousarray", (head,))?;
            let tail_slice = PySlice::new(py, want, total, 1);
            let tail = arr.get_item(tail_slice)?;
            let tail_copy =
                np.call_method1("ascontiguousarray", (tail,))?;
            taken.push(head_copy.unbind());
            buf.chunks[consume_idx] = tail_copy.unbind();
            // accum is unused after the break (we'd reset
            // totals below by walking the residual chunks); the
            // chunk at consume_idx now holds the residual, so
            // don't advance consume_idx.
            break;
        }
    }

    // Discard the chunks fully consumed at the front.
    buf.chunks.drain(0..consume_idx);

    // Concatenate the drain pieces into one ndarray.  Single
    // piece is a no-op cast back through the Py<PyAny>.
    let concat: Py<PyAny> = if taken.len() == 1 {
        taken.into_iter().next().unwrap()
    } else {
        let tup = PyTuple::new(py, taken.iter().map(|p| p.bind(py)))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("axis", 0)?;
        np.call_method("concatenate", (tup,), Some(&kwargs))?.unbind()
    };

    // Recompute totals from what's left.  Walking is O(n_chunks)
    // but n_chunks shrinks at every drain so this stays bounded.
    let mut new_rows: u64 = 0;
    let mut new_bytes: u64 = 0;
    for c in buf.chunks.iter() {
        let arr = c.bind(py);
        let s: Vec<usize> = arr.getattr("shape")?.extract()?;
        new_rows += s[0] as u64;
        new_bytes += arr.getattr("nbytes")?.extract::<u64>()?;
    }
    buf.total_rows = new_rows;
    buf.total_bytes = new_bytes;

    Ok(concat)
}

// Drain whatever tile-aligned bytes fit while staying under the
// cap.  Called from `append_to_buffer` after each append.  Reads
// the current image NAXIS0 + tile rows via the meta cache, so it
// sees the post-previous-drain state automatically (each drain
// bumps cards_version which invalidates the cache).  No-op if
// the buffer is under cap or if no tile-aligned drain fits.
fn drain_aligned_subset(
    py: Python<'_>,
    hdu_obj: &Bound<'_, CompressedImageHDU>,
) -> PyResult<()> {
    let hdu = hdu_obj.borrow();
    check_not_tainted(&hdu.as_super().as_super().tainted)?;

    // Cheap pre-check while holding only the pending lock: if
    // we're under cap, skip the meta lookup + drain math entirely.
    {
        let g = hdu.pending.lock().map_err(|_| {
            PyIOError::new_err("pending buffer lock poisoned")
        })?;
        let Some(buf) = g.as_ref() else { return Ok(()) };
        if buf.total_bytes <= MAX_PENDING_BYTES {
            return Ok(());
        }
    }

    // Look up tile_rows + current NAXIS0 via the meta cache.
    let super_ = hdu.as_super().as_super();
    let meta = hdu.meta(super_)?;
    if meta.image_shape.is_empty() || meta.tile_shape.is_empty() {
        return Ok(()); // degenerate HDU, defer to __exit__ drain
    }
    let tile_rows = meta.tile_shape[0];
    let current_naxis0 = meta.image_shape[0];

    // Extract the drain ndarray under the pending lock, then
    // release the lock before calling the (potentially slow)
    // extend code.  Holding the lock through the file I/O would
    // serialize other contexts but there is only one context
    // per HDU by construction.
    let drain_arr: Py<PyAny> = {
        let mut g = hdu.pending.lock().map_err(|_| {
            PyIOError::new_err("pending buffer lock poisoned")
        })?;
        let buf = g.as_mut().ok_or_else(|| {
            PyValueError::new_err(
                "internal: pending state vanished mid-drain",
            )
        })?;
        let drain_rows =
            aligned_drain_rows(current_naxis0, tile_rows, buf.total_rows);
        if drain_rows == 0 {
            // Buffer is over cap but no tile-aligned drain fits
            // (tile bigger than cap, or first partial-tile fill
            // not yet possible).  Leave buffer as-is; next
            // append will retry, and __exit__ will drain the
            // residual no matter what.
            return Ok(());
        }
        pop_front_rows(py, buf, drain_rows)?
    };

    // Call the existing extend code with the tile-aligned slice.
    // Cards are re-snapshotted to pick up any state changed by a
    // prior drain in this same context.
    let cache = Arc::clone(&hdu.cache);
    let quantize_config = Arc::clone(&hdu.quantize_config);
    let compress_config = Arc::clone(&hdu.compress_config);
    let cards = super_.header_snapshot()?;
    extend_compressed_image_data(
        py,
        drain_arr.bind(py),
        &cards,
        &super_.offsets,
        &super_.file,
        &super_.layout,
        &super_.tainted,
        &cache,
        &super_.header,
        &super_.cards_version,
        &quantize_config,
        &compress_config,
    )
}

// Drain whatever residual is left in the buffer.  Called from
// `__exit__` (normal or exceptional).  Always resets `pending`
// to None before the extend call so that extend's own routing
// (which checks `is_in_context`) sees the post-context state.
// This is the only drain in a typical context that pays a
// partial-trailing-tile re-encode (because the cap-triggered
// mid-context drains are tile-aligned by construction).
fn drain_residual_and_exit(
    py: Python<'_>,
    hdu_obj: &Py<CompressedImageHDU>,
) -> PyResult<()> {
    let hdu = hdu_obj.bind(py).borrow();
    check_not_tainted(&hdu.as_super().as_super().tainted)?;

    // Take the buffer first so the subsequent extend call (if
    // any) routes through the normal path, not the context-
    // buffering path.
    let taken: Option<PendingBuffer> = {
        let mut g = hdu.pending.lock().map_err(|_| {
            PyIOError::new_err("pending buffer lock poisoned")
        })?;
        g.take()
    };
    let Some(mut buf) = taken else { return Ok(()) };
    if buf.total_rows == 0 {
        return Ok(());
    }

    // Concatenate the residual chunks into one ndarray.
    let np = py.import("numpy")?;
    let concat_obj: Py<PyAny> = if buf.chunks.len() == 1 {
        buf.chunks.remove(0)
    } else {
        let tup = PyTuple::new(py, buf.chunks.iter().map(|p| p.bind(py)))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("axis", 0)?;
        np.call_method("concatenate", (tup,), Some(&kwargs))?.unbind()
    };
    let concat_bound = concat_obj.bind(py);

    let cache = Arc::clone(&hdu.cache);
    let quantize_config = Arc::clone(&hdu.quantize_config);
    let compress_config = Arc::clone(&hdu.compress_config);
    let super_ = hdu.as_super().as_super();
    let cards = super_.header_snapshot()?;
    extend_compressed_image_data(
        py,
        concat_bound,
        &cards,
        &super_.offsets,
        &super_.file,
        &super_.layout,
        &super_.tainted,
        &cache,
        &super_.header,
        &super_.cards_version,
        &quantize_config,
        &compress_config,
    )
}

/// Context-manager handle returned by
/// :meth:`CompressedImageHDU.extending`.  Use via a
/// ``with`` statement; the only legal operation inside the block
/// is :meth:`CompressedImageHDU.extend`.
#[pyclass(module = "rustfits._rust")]
pub(crate) struct CompressedImageExtendContext {
    pub(crate) hdu: Py<CompressedImageHDU>,
}

#[pymethods]
impl CompressedImageExtendContext {
    fn __enter__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Py<CompressedImageHDU>> {
        let hdu = slf.hdu.bind(py).borrow();
        check_not_tainted(&hdu.as_super().as_super().tainted)?;
        let mut g = hdu.pending.lock().map_err(|_| {
            PyIOError::new_err("pending buffer lock poisoned")
        })?;
        if g.is_some() {
            return Err(PyValueError::new_err(
                "this HDU is already inside an extending() context",
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
