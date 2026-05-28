// Schema edits: insert_column + delete_column.  Both rewrite the header
// (TFIELDS / NAXIS1 + per-column TTYPEn / TFORMn / TDIMn / TUNITn /
// TZEROn / TSCALn / TNULLn / TDISPn / TBCOLn renumbering), reshape the
// main data section row-by-row in 1 MiB strips, relocate the heap (if
// any) to its new position, and grow / shrink the data extent.  Strip-
// based I/O keeps peak memory bounded — the whole table is never read
// into RAM at once.
//
// Scope (first cut): insert_column accepts a regular (non-Object) numpy
// ndarray — VLA column insertion is not supported and rejects with a
// clear error.  delete_column DOES support VLA columns: the descriptor
// bytes are removed from each row but the heap is left as-is; the
// deleted column's heap cells become orphans that hdu.repack() reclaims.
// Existing VLA columns in the table are preserved through both
// operations (the heap is relocated forward/backward to sit after the
// new main rows; descriptor offsets are relative to heap start, so
// they remain valid).
//
// Both reject non-default THEAP (THEAP != NAXIS1*NAXIS2) up front, same
// constraint as repack(): the rewrite always writes the heap at the
// default position and would corrupt a custom THEAP layout.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::IntoPyObjectExt;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::Ordering;

use crate::common::{
    check_not_tainted, lock_file, parse_keyword,
    shift_file_tail_and_update_offsets,
    shift_file_tail_backward_and_update_offsets,
    zero_fill_range, FileHandle, RawBuffer, TaintFlag,
};
use crate::hdu::HDU;
use crate::hdu_image::round_up_to_block;
use crate::header::{card_int, card_string, card_uint, rewrite_header_to_disk};

use super::columns::{bytes_per_element, parse_columns, Column};
use super::write_fixed::{
    acquire_per_column_array, apply_transform_cell,
    set_x_vla_tform_maxlen_in_cards,
};
use super::write_setup::{
    dtype_to_write_columns, WriteColumn, WriteTransform,
};
use super::write_vla::{
    plan_vla_heap_layout, serialize_vla_cell, write_descriptor,
};

// Strip target: ~1 MiB per buffer.  Same constant the bulk-write path
// (write_table_data) uses; keeps peak RSS bounded regardless of table
// size.
const STRIP_TARGET_BYTES: usize = 1 << 20;

// --------------------------------------------------------------------
// Public entry points
// --------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_column_impl(
    py: Python<'_>,
    super_: &HDU,
    name: &str,
    data: &Bound<'_, PyAny>,
    position: Option<i64>,
    after: Option<&Bound<'_, PyAny>>,
    before: Option<&Bound<'_, PyAny>>,
    unit: Option<&str>,
    inner_dtype: Option<&str>,
    heap_format: Option<&str>,
    bit_packed: bool,
) -> PyResult<()> {
    check_not_tainted(&super_.tainted)?;

    let cards = super_.header_snapshot()?;
    let columns = parse_columns(&cards)?;
    let nrows = parse_keyword(&cards, "NAXIS2")
        .unwrap_or(0).max(0) as usize;
    let row_width = parse_keyword(&cards, "NAXIS1")
        .unwrap_or(0).max(0) as usize;
    let pcount = parse_keyword(&cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    reject_non_default_theap(&cards, nrows, row_width)?;

    if name.is_empty() {
        return Err(PyValueError::new_err(
            "insert_column: column name must be non-empty"));
    }
    let name_u = name.to_uppercase();
    if columns.iter().any(|c| c.name.to_uppercase() == name_u) {
        return Err(PyValueError::new_err(format!(
            "insert_column: column '{}' already exists", name)));
    }
    let insert_idx =
        resolve_insert_index(&columns, position, after, before)?;

    // Dispatch on data.dtype.kind == 'O': route Object-dtype input
    // to the VLA-aware path (Phase: VLA insert_column), everything
    // else to the existing fixed-column path.  The two paths share
    // the header-card prep, insert-index resolution, and the
    // row-shuffler shape but diverge in how the new column's bytes
    // are produced (fixed = per-cell transform from input ndarray;
    // VLA = descriptor bytes referencing planned heap offsets).
    let arr_dtype = data.getattr("dtype")?;
    let arr_kind: String = arr_dtype.getattr("kind")?.extract()?;
    if arr_kind == "O" {
        return insert_vla_column_impl(
            py, super_, &cards, &columns, nrows, row_width, pcount,
            name, data, insert_idx, unit, inner_dtype, heap_format,
            bit_packed,
        );
    }
    // Reject VLA-only kwargs on non-Object input — silent ignore
    // would hide typos.
    if inner_dtype.is_some() {
        return Err(PyValueError::new_err(
            "insert_column: inner_dtype= is only valid when data is \
             an Object-dtype ndarray (VLA column)"));
    }
    if heap_format.is_some() {
        return Err(PyValueError::new_err(
            "insert_column: heap_format= is only valid when data is \
             an Object-dtype ndarray (VLA column)"));
    }
    let new_write_col =
        build_single_write_column(py, name, data, unit, bit_packed)?;
    let new_col_byte_width = new_write_col.byte_width;

    // Input validation + RawBuffer acquisition.  The RawBuffer pins the
    // numpy ndarray's memory for the whole strip loop; per-row slices
    // come out of buf_slice as the loop walks rows.
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let mut buffers: Vec<RawBuffer> = Vec::new();
    let encode_col = synth_column_for_encode(&new_write_col)?;
    let (transform, src_total_size, buffer_idx) =
        acquire_per_column_array(data, &ndarray, &encode_col, nrows, &mut buffers)?;

    let pre_width: usize = columns.iter().take(insert_idx)
        .map(|c| c.byte_width).sum();
    let post_width = row_width - pre_width;
    let new_row_width = row_width + new_col_byte_width;

    // Build the new card list before any file mutation.
    let mut new_cards = cards.clone();
    renumber_per_column_cards(&mut new_cards, insert_idx, /* delta = */ 1)?;
    let new_column_cards = build_new_column_cards(&new_write_col, insert_idx);
    insert_cards_before_end(&mut new_cards, &new_column_cards)?;
    update_int_card(&mut new_cards, "TFIELDS", (columns.len() + 1) as i64)?;
    update_int_card(&mut new_cards, "NAXIS1", new_row_width as i64)?;

    // (1) Header rewrite first — may grow header by N blocks via the
    //     shared shift primitive, bumping self.data_offset.  Acquire
    //     the cards write guard now so the eventual commit bumps the
    //     version counter under the same lock that protects the cards
    //     Vec; the lock is held through the data work below.
    let cards_guard = super_.cards_write_lock()?;
    rewrite_header_to_disk(
        &super_.file, &super_.offsets, &super_.layout,
        &new_cards, &super_.tainted,
    )?;
    let data_offset = super_.offsets.data_offset();

    // (2) Extend the data section by the row-layout growth + push later
    //     HDUs forward.  No-op if same block-rounded size.
    let old_main_bytes = (nrows * row_width) as u64;
    let new_main_bytes = (nrows * new_row_width) as u64;
    grow_or_shrink_data_extent(
        super_, data_offset,
        old_main_bytes + pcount,
        new_main_bytes + pcount,
    )?;

    // (3) Relocate heap forward (out of the way before we shuffle main
    //     rows over the old heap region).
    if pcount > 0 {
        relocate_region_forward(
            &super_.file,
            /* src_start = */ data_offset + old_main_bytes,
            /* dst_start = */ data_offset + new_main_bytes,
            /* total     = */ pcount,
            &super_.tainted,
        )?;
    }

    // (4) Shuffle main rows back-to-front into the new layout, encoding
    //     the new column inline.
    shuffle_main_for_insert(
        super_, data_offset, nrows, row_width, new_row_width,
        pre_width, post_width, new_col_byte_width, insert_idx,
        &buffers[buffer_idx], src_total_size,
        &transform, &encode_col.name,
    )?;

    cards_guard.commit(new_cards);
    Ok(())
}

// VLA branch of insert_column.  Same shape as the fixed branch
// (header rewrite → grow → relocate heap → shuffle main rows) with
// two divergences: the new column slot in each row gets a P or Q
// descriptor (not the cell bytes), and the cell bytes themselves
// are written into the heap AFTER the main shuffle.  The existing
// heap is relocated forward by `nrows * descriptor_size` (the row-
// layout growth) plus has the new heap cells appended at its end.
//
// Caller must have already verified:
//   - data.dtype.kind == "O"
//   - name + insert_idx are valid
//   - the HDU isn't tainted and uses the default THEAP
#[allow(clippy::too_many_arguments)]
fn insert_vla_column_impl(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    columns: &[Column],
    nrows: usize,
    row_width: usize,
    current_pcount: u64,
    name: &str,
    data: &Bound<'_, PyAny>,
    insert_idx: usize,
    unit: Option<&str>,
    inner_dtype: Option<&str>,
    heap_format: Option<&str>,
    bit_packed: bool,
) -> PyResult<()> {
    let inner_dtype = inner_dtype.ok_or_else(|| PyValueError::new_err(
        "insert_column: VLA (Object dtype) columns require the \
         inner_dtype= kwarg (e.g. 'f4', 'i4', '?' / 'bool')",
    ))?;
    let desc_char = match heap_format {
        None | Some("P") | Some("p") => 'P',
        Some("Q") | Some("q") => 'Q',
        Some(other) => return Err(PyValueError::new_err(format!(
            "insert_column: heap_format must be 'P' or 'Q', got '{}'",
            other))),
    };
    let descriptor_size = if desc_char == 'P' { 8usize } else { 16 };

    // Build the WriteColumn via the same classifier path
    // create_table_hdu uses: a one-field Object structured dtype +
    // var_dtypes={name: inner_dtype} + optional bit_columns=[name].
    let np = py.import("numpy")?;
    let arr_shape: Vec<usize> = data.getattr("shape")?.extract()?;
    if arr_shape.is_empty() {
        return Err(PyValueError::new_err(
            "insert_column: data must have at least one dimension (rows)"));
    }
    if arr_shape.len() != 1 || arr_shape[0] != nrows {
        return Err(PyValueError::new_err(format!(
            "insert_column: VLA data must be a 1-D Object ndarray \
             of shape ({},); got {:?}", nrows, arr_shape)));
    }
    let field_tuple = PyTuple::new(py, [
        name.into_py_any(py)?, "O".into_py_any(py)?,
    ])?;
    let descr = PyList::new(py, [field_tuple])?;
    let np_dtype = np.getattr("dtype")?.call1((descr,))?;
    let var_dtypes_dict = PyDict::new(py);
    var_dtypes_dict.set_item(name, inner_dtype)?;
    let units_dict = if let Some(u) = unit {
        let d = PyDict::new(py);
        d.set_item(name, u)?;
        Some(d)
    } else {
        None
    };
    let bit_columns_spec = if bit_packed {
        let mut set = std::collections::HashSet::new();
        set.insert(name.to_uppercase());
        Some(super::write_setup::BitColumnsSpec::Names(set))
    } else {
        None
    };
    let cols = dtype_to_write_columns(
        &np_dtype,
        units_dict.as_ref(),
        Some(&var_dtypes_dict),
        bit_columns_spec.as_ref(),
        desc_char,
    )?;
    if cols.len() != 1 {
        return Err(PyValueError::new_err(
            "internal: dtype_to_write_columns returned wrong column count"));
    }
    let new_write_col = cols.into_iter().next().unwrap();
    let inner_letter = new_write_col.tform_letter;
    let is_x = inner_letter == 'X';

    // Validate-then-mutate: plan the heap layout for the new
    // column's cells using the existing planner.  Each cell's
    // bytes_offset_in_heap is relative to the (post-relocation)
    // heap start; we start the cursor at current_pcount so the new
    // cells append to the existing heap.
    let ndarray = np.getattr("ndarray")?;
    let synth_col = synth_column_for_encode(&new_write_col)?;
    let synth_cols = vec![synth_col];
    let per_col_inputs: Vec<Bound<'_, PyAny>> = vec![data.clone()];
    let (vla_plans, new_pcount_usize) = plan_vla_heap_layout(
        &synth_cols, &per_col_inputs, nrows, &ndarray,
        current_pcount as usize,
    )?;
    let new_pcount = new_pcount_usize as u64;
    let plans = &vla_plans[0];

    // Element size used for the heap-cell byte count.  X is
    // bit-counted (not byte-counted); other letters have a fixed
    // element width.
    let elem_size = if is_x {
        0  // sentinel; X path uses div_ceil(8) below
    } else {
        bytes_per_element(inner_letter).ok_or_else(|| {
            PyValueError::new_err(format!(
                "insert_column: VLA inner letter '{}' has no fixed \
                 element width", inner_letter))
        })?
    };

    // Header card rewrite.  Same prep as the fixed path PLUS the
    // PCOUNT bump (the new heap grows by sum of new cell bytes).
    let pre_width: usize = columns.iter().take(insert_idx)
        .map(|c| c.byte_width).sum();
    let new_row_width = row_width + descriptor_size;
    let mut new_cards = cards.to_vec();
    renumber_per_column_cards(&mut new_cards, insert_idx, /* delta = */ 1)?;
    let new_column_cards = build_new_column_cards(&new_write_col, insert_idx);
    insert_cards_before_end(&mut new_cards, &new_column_cards)?;
    update_int_card(&mut new_cards, "TFIELDS", (columns.len() + 1) as i64)?;
    update_int_card(&mut new_cards, "NAXIS1", new_row_width as i64)?;
    update_int_card(&mut new_cards, "PCOUNT", new_pcount as i64)?;
    // X-inner VLA columns get a (maxbits) hint in their TFORM so
    // astropy's strict parser accepts the file (other libraries
    // and the FITS spec itself treat it as informational).
    if is_x {
        let max_bits = plans.iter()
            .map(|p| p.nelements).max().unwrap_or(0);
        set_x_vla_tform_maxlen_in_cards(
            &mut new_cards, insert_idx + 1, desc_char, max_bits,
        );
    }

    // (1) Header rewrite first — may grow header blocks.  Acquire
    //     the cards write guard now; the lock is held through the
    //     data work below and the eventual commit bumps the version
    //     counter under that lock.
    let cards_guard = super_.cards_write_lock()?;
    rewrite_header_to_disk(
        &super_.file, &super_.offsets, &super_.layout,
        &new_cards, &super_.tainted,
    )?;
    let data_offset = super_.offsets.data_offset();

    // (2) Extend data section.
    let old_main_bytes = (nrows * row_width) as u64;
    let new_main_bytes = (nrows * new_row_width) as u64;
    grow_or_shrink_data_extent(
        super_, data_offset,
        old_main_bytes + current_pcount,
        new_main_bytes + new_pcount,
    )?;

    // (3) Relocate existing heap forward to sit after the new
    //     (wider) main-row region.  The relocated heap occupies
    //     [new_main_bytes, new_main_bytes + current_pcount); the
    //     new cells will append at [new_main_bytes + current_pcount,
    //     new_main_bytes + new_pcount).
    if current_pcount > 0 {
        relocate_region_forward(
            &super_.file,
            data_offset + old_main_bytes,
            data_offset + new_main_bytes,
            current_pcount,
            &super_.tainted,
        )?;
    }

    // (4) Shuffle main rows back-to-front into the new layout,
    //     writing a descriptor (nelements, heap_offset) at the
    //     new column's slot per row.
    shuffle_main_for_vla_insert(
        super_, data_offset, nrows, row_width, new_row_width,
        pre_width, descriptor_size, desc_char, plans,
    )?;

    // (5) Write new VLA cells to the heap.  Cell K lands at
    //     data_offset + new_main_bytes + plans[K].bytes_offset_in_heap
    //     (the planner used heap_start_offset=current_pcount, so
    //     new cells append at the relocated heap's tail).
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        for (r, &plan) in plans.iter().enumerate() {
            if plan.nelements == 0 {
                continue;
            }
            let cell = data.get_item(r)?;
            let cell_bytes = if is_x {
                plan.nelements.div_ceil(8)
            } else {
                plan.nelements * elem_size
            };
            let mut buf = vec![0u8; cell_bytes];
            serialize_vla_cell(&cell, inner_letter, plan.nelements, &mut buf)?;
            let abs_offset = data_offset + new_main_bytes
                + plan.bytes_offset_in_heap as u64;
            f.seek(SeekFrom::Start(abs_offset))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.write_all(&buf).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "insert_column: VLA cell heap write row {}: {}; \
                     close + reopen", r, e))
            })?;
        }
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "insert_column: VLA heap flush: {}; close + reopen", e))
        })?;
    }

    cards_guard.commit(new_cards);
    Ok(())
}

// Variant of shuffle_main_for_insert for VLA columns: instead of
// running a per-cell transform from an input buffer, emit a
// descriptor (nelements, heap_offset) per row at the new column's
// slot.  Heap offsets come from plan_vla_heap_layout (relative to
// the heap start that the caller has already established with the
// `forward` relocate).
#[allow(clippy::too_many_arguments)]
fn shuffle_main_for_vla_insert(
    super_: &HDU,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    new_row_width: usize,
    pre_width: usize,
    descriptor_size: usize,
    descriptor_kind: char,
    plans: &[super::write_vla::VlaCellPlan],
) -> PyResult<()> {
    if nrows == 0 {
        return Ok(());
    }
    let post_width = row_width - pre_width;
    let strip_rows = ((STRIP_TARGET_BYTES / new_row_width.max(1)).max(1))
        .min(nrows);
    let mut old_buf = vec![0u8; strip_rows * row_width];
    let mut new_buf = vec![0u8; strip_rows * new_row_width];

    let mut row_end = nrows;
    while row_end > 0 {
        let chunk = row_end.min(strip_rows);
        let row_start = row_end - chunk;
        old_buf.resize(chunk * row_width, 0);
        new_buf.resize(chunk * new_row_width, 0);
        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.read_exact(&mut old_buf).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "insert_column (VLA): read main strip failed: {}; \
                     close + reopen", e))
            })?;
        }
        for r in 0..chunk {
            let src_row = &old_buf[r * row_width..(r + 1) * row_width];
            let dst_row =
                &mut new_buf[r * new_row_width..(r + 1) * new_row_width];
            dst_row[..pre_width].copy_from_slice(&src_row[..pre_width]);
            let input_row = row_start + r;
            let plan = plans[input_row];
            write_descriptor(
                descriptor_kind, plan.nelements, plan.bytes_offset_in_heap,
                &mut dst_row[pre_width..pre_width + descriptor_size],
            );
            dst_row[pre_width + descriptor_size..]
                .copy_from_slice(&src_row[pre_width..pre_width + post_width]);
        }
        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * new_row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            if let Err(e) = f.write_all(&new_buf) {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "insert_column (VLA): write main strip failed: {}; \
                     close + reopen", e)));
            }
            if let Err(e) = f.flush() {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "insert_column (VLA): flush failed: {}; \
                     close + reopen", e)));
            }
        }
        row_end = row_start;
    }
    Ok(())
}

pub(crate) fn delete_column_impl(
    super_: &HDU,
    key: &Bound<'_, PyAny>,
) -> PyResult<()> {
    check_not_tainted(&super_.tainted)?;

    let cards = super_.header_snapshot()?;
    let columns = parse_columns(&cards)?;
    let nrows = parse_keyword(&cards, "NAXIS2")
        .unwrap_or(0).max(0) as usize;
    let row_width = parse_keyword(&cards, "NAXIS1")
        .unwrap_or(0).max(0) as usize;
    let pcount = parse_keyword(&cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    reject_non_default_theap(&cards, nrows, row_width)?;
    if columns.is_empty() {
        return Err(PyValueError::new_err(
            "delete_column: table has no columns to delete"));
    }

    let delete_idx = resolve_column_key(&columns, key)?;
    let deleted_col = &columns[delete_idx];
    let deleted_width = deleted_col.byte_width;
    let pre_width: usize = columns.iter().take(delete_idx)
        .map(|c| c.byte_width).sum();
    let post_offset = pre_width + deleted_width;
    let new_row_width = row_width - deleted_width;

    let mut new_cards = cards.clone();
    drop_per_column_cards(&mut new_cards, delete_idx);
    renumber_per_column_cards(&mut new_cards, delete_idx, /* delta = */ -1)?;
    update_int_card(&mut new_cards, "TFIELDS", (columns.len() - 1) as i64)?;
    update_int_card(&mut new_cards, "NAXIS1", new_row_width as i64)?;

    // (1) Header rewrite.  delete usually doesn't grow the header (drops
    //     cards), but rewrite_header_to_disk handles both cases.  Acquire
    //     the cards write guard now; lock is held through the data work
    //     below and the eventual commit bumps the version counter.
    let cards_guard = super_.cards_write_lock()?;
    rewrite_header_to_disk(
        &super_.file, &super_.offsets, &super_.layout,
        &new_cards, &super_.tainted,
    )?;
    let data_offset = super_.offsets.data_offset();

    // (2) Shuffle main rows front-to-back into the smaller layout.
    shuffle_main_for_delete(
        super_, data_offset, nrows, row_width, new_row_width,
        pre_width, post_offset,
    )?;

    // (3) Relocate heap backward (heap follows the now-shorter main).
    if pcount > 0 {
        let old_heap_start = data_offset + (nrows * row_width) as u64;
        let new_heap_start = data_offset + (nrows * new_row_width) as u64;
        relocate_region_backward(
            &super_.file,
            old_heap_start,
            new_heap_start,
            pcount,
            &super_.tainted,
        )?;
    }

    // (4) Shrink data section (may shift later HDUs backward or set_len
    //     if last HDU).
    let old_data_bytes = (nrows * row_width) as u64 + pcount;
    let new_data_bytes = (nrows * new_row_width) as u64 + pcount;
    grow_or_shrink_data_extent(
        super_, data_offset, old_data_bytes, new_data_bytes,
    )?;

    cards_guard.commit(new_cards);
    Ok(())
}

// --------------------------------------------------------------------
// Strip-based row shuffling
// --------------------------------------------------------------------

// Back-to-front strip walker.  For each strip [a, b) processed from the
// LAST strip toward the FIRST, reads R rows of old main bytes, builds R
// rows of new main bytes (pre + new-column + post), writes them to the
// new (later) position.  Back-to-front because we're growing the row
// width — writing to later positions means we'd clobber the read region
// of earlier strips if we went forward.
#[allow(clippy::too_many_arguments)]
fn shuffle_main_for_insert(
    super_: &HDU,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    new_row_width: usize,
    pre_width: usize,
    post_width: usize,
    new_col_byte_width: usize,
    _insert_idx: usize,
    input_buf: &RawBuffer,
    src_total_size: usize,
    transform: &WriteTransform,
    col_name: &str,
) -> PyResult<()> {
    if nrows == 0 {
        return Ok(());
    }
    let strip_rows = ((STRIP_TARGET_BYTES / new_row_width.max(1)).max(1))
        .min(nrows);
    let mut old_buf = vec![0u8; strip_rows * row_width];
    let mut new_buf = vec![0u8; strip_rows * new_row_width];
    let input_bytes = input_buf.as_slice();

    let mut row_end = nrows;
    while row_end > 0 {
        let chunk = row_end.min(strip_rows);
        let row_start = row_end - chunk;
        old_buf.resize(chunk * row_width, 0);
        new_buf.resize(chunk * new_row_width, 0);

        // Read this strip's old bytes.
        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.read_exact(&mut old_buf).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "insert_column: read main strip failed: {}; \
                     close + reopen", e))
            })?;
        }

        // Build new rows.
        for r in 0..chunk {
            let src_row = &old_buf[r * row_width..(r + 1) * row_width];
            let dst_row =
                &mut new_buf[r * new_row_width..(r + 1) * new_row_width];
            dst_row[..pre_width].copy_from_slice(&src_row[..pre_width]);
            let input_row = row_start + r;
            let cell_src = &input_bytes
                [input_row * src_total_size..(input_row + 1) * src_total_size];
            let cell_dst =
                &mut dst_row[pre_width..pre_width + new_col_byte_width];
            apply_transform_cell(transform, cell_src, cell_dst, col_name, r)?;
            dst_row[pre_width + new_col_byte_width..]
                .copy_from_slice(&src_row[pre_width..pre_width + post_width]);
        }

        // Write this strip's new bytes.
        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * new_row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            if let Err(e) = f.write_all(&new_buf) {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "insert_column: write main strip failed: {}; \
                     close + reopen", e)));
            }
            if let Err(e) = f.flush() {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "insert_column: flush failed: {}; close + reopen", e)));
            }
        }

        row_end = row_start;
    }
    Ok(())
}

// Front-to-back strip walker for delete.  Smaller row width → writes
// land BEFORE reads, so processing forward keeps un-read bytes safe.
#[allow(clippy::too_many_arguments)]
fn shuffle_main_for_delete(
    super_: &HDU,
    data_offset: u64,
    nrows: usize,
    row_width: usize,
    new_row_width: usize,
    pre_width: usize,
    post_offset: usize,
) -> PyResult<()> {
    if nrows == 0 || new_row_width == row_width {
        return Ok(());
    }
    let post_width = row_width - post_offset;
    let strip_rows = ((STRIP_TARGET_BYTES / row_width.max(1)).max(1))
        .min(nrows);
    let mut old_buf = vec![0u8; strip_rows * row_width];
    let mut new_buf = vec![0u8; strip_rows * new_row_width];

    let mut row_start = 0usize;
    while row_start < nrows {
        let chunk = (nrows - row_start).min(strip_rows);
        old_buf.resize(chunk * row_width, 0);
        new_buf.resize(chunk * new_row_width, 0);

        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.read_exact(&mut old_buf).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "delete_column: read main strip failed: {}; \
                     close + reopen", e))
            })?;
        }

        for r in 0..chunk {
            let src_row = &old_buf[r * row_width..(r + 1) * row_width];
            let dst_row =
                &mut new_buf[r * new_row_width..(r + 1) * new_row_width];
            dst_row[..pre_width].copy_from_slice(&src_row[..pre_width]);
            dst_row[pre_width..pre_width + post_width]
                .copy_from_slice(&src_row[post_offset..]);
        }

        {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.seek(SeekFrom::Start(
                data_offset + (row_start * new_row_width) as u64))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            if let Err(e) = f.write_all(&new_buf) {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "delete_column: write main strip failed: {}; \
                     close + reopen", e)));
            }
            if let Err(e) = f.flush() {
                super_.tainted.store(true, Ordering::Release);
                return Err(PyIOError::new_err(format!(
                    "delete_column: flush failed: {}; close + reopen", e)));
            }
        }

        row_start += chunk;
    }
    Ok(())
}

// --------------------------------------------------------------------
// Heap relocation (1 MiB chunked, in-file copy)
// --------------------------------------------------------------------

// Forward relocation: dst > src.  Back-to-front chunked copy keeps the
// overlap safe (we read late bytes, write to even later positions; next
// chunk reads earlier bytes that haven't been written yet).
fn relocate_region_forward(
    file: &FileHandle,
    src_start: u64,
    dst_start: u64,
    total: u64,
    tainted: &TaintFlag,
) -> PyResult<()> {
    if total == 0 || src_start == dst_start {
        return Ok(());
    }
    let chunk_size = STRIP_TARGET_BYTES as u64;
    let mut buf = vec![0u8; chunk_size as usize];
    let mut remaining = total;
    while remaining > 0 {
        let n = remaining.min(chunk_size);
        let src_off = src_start + remaining - n;
        let dst_off = dst_start + remaining - n;
        buf.resize(n as usize, 0);
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(src_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "schema edit: heap read failed: {}; close + reopen", e))
        })?;
        f.seek(SeekFrom::Start(dst_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "schema edit: heap write failed: {}; close + reopen", e)));
        }
        if let Err(e) = f.flush() {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "schema edit: heap flush failed: {}; close + reopen", e)));
        }
        remaining -= n;
    }
    Ok(())
}

// Backward relocation: dst < src.  Front-to-back chunked copy.
fn relocate_region_backward(
    file: &FileHandle,
    src_start: u64,
    dst_start: u64,
    total: u64,
    tainted: &TaintFlag,
) -> PyResult<()> {
    if total == 0 || src_start == dst_start {
        return Ok(());
    }
    let chunk_size = STRIP_TARGET_BYTES as u64;
    let mut buf = vec![0u8; chunk_size as usize];
    let mut processed = 0u64;
    while processed < total {
        let n = (total - processed).min(chunk_size);
        let src_off = src_start + processed;
        let dst_off = dst_start + processed;
        buf.resize(n as usize, 0);
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(src_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "schema edit: heap read failed: {}; close + reopen", e))
        })?;
        f.seek(SeekFrom::Start(dst_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "schema edit: heap write failed: {}; close + reopen", e)));
        }
        if let Err(e) = f.flush() {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "schema edit: heap flush failed: {}; close + reopen", e)));
        }
        processed += n;
    }
    Ok(())
}

// --------------------------------------------------------------------
// Data-section grow / shrink (block-aligned)
// --------------------------------------------------------------------

fn grow_or_shrink_data_extent(
    super_: &HDU,
    data_offset: u64,
    old_data_bytes: u64,
    new_data_bytes: u64,
) -> PyResult<()> {
    let current_padded = round_up_to_block(old_data_bytes);
    let new_padded = round_up_to_block(new_data_bytes);
    if new_padded == current_padded {
        return Ok(());
    }
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;
    let file_len = {
        let g = lock_file(&super_.file)?;
        let f = g.as_ref()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.len()
            .map_err(|e| PyIOError::new_err(e.to_string()))?
    };
    if new_padded > current_padded {
        let delta = new_padded - current_padded;
        if file_len > current_hdu_end {
            shift_file_tail_and_update_offsets(
                &super_.file, &super_.layout,
                current_hdu_end, delta, &super_.tainted)?;
            zero_fill_range(
                &super_.file, current_hdu_end, delta, &super_.tainted)?;
        } else {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "schema edit: set_len failed: {}; close + reopen", e))
            })?;
        }
    } else {
        let delta = current_padded - new_padded;
        if file_len > current_hdu_end {
            shift_file_tail_backward_and_update_offsets(
                &super_.file, &super_.layout,
                current_hdu_end, delta, &super_.tainted)?;
        } else {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "schema edit: set_len failed: {}; close + reopen", e))
            })?;
        }
    }
    Ok(())
}

// --------------------------------------------------------------------
// Position / column-key resolution
// --------------------------------------------------------------------

fn resolve_insert_index(
    columns: &[Column],
    position: Option<i64>,
    after: Option<&Bound<'_, PyAny>>,
    before: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    let n_provided = [position.is_some(), after.is_some(), before.is_some()]
        .iter().filter(|b| **b).count();
    if n_provided > 1 {
        return Err(PyValueError::new_err(
            "insert_column: at most one of position=, after=, before= may \
             be specified"));
    }
    let ncols = columns.len();
    if let Some(p) = position {
        if p < 0 || p as usize > ncols {
            return Err(PyValueError::new_err(format!(
                "insert_column: position={} out of range (table has {} \
                 columns; valid positions are 0..={})", p, ncols, ncols)));
        }
        return Ok(p as usize);
    }
    if let Some(a) = after {
        let idx = resolve_column_key(columns, a)?;
        return Ok(idx + 1);
    }
    if let Some(b) = before {
        let idx = resolve_column_key(columns, b)?;
        return Ok(idx);
    }
    Ok(ncols)
}

fn resolve_column_key(
    columns: &[Column],
    key: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let ncols = columns.len();
    if let Ok(name) = key.extract::<String>() {
        let name_u = name.to_uppercase();
        for (i, c) in columns.iter().enumerate() {
            if c.name.to_uppercase() == name_u {
                return Ok(i);
            }
        }
        return Err(PyValueError::new_err(format!(
            "column '{}' not found in table", name)));
    }
    if let Ok(i) = key.extract::<i64>() {
        let resolved = if i < 0 { i + ncols as i64 } else { i };
        if resolved < 0 || resolved as usize >= ncols {
            return Err(PyValueError::new_err(format!(
                "column index {} out of range (table has {} columns)",
                i, ncols)));
        }
        return Ok(resolved as usize);
    }
    Err(PyValueError::new_err(
        "column key must be a string name or integer index"))
}

// --------------------------------------------------------------------
// Per-column header card mutation
// --------------------------------------------------------------------

// Per-column keyword prefixes that need renumbering on insert/delete.
// Covers both keywords this codebase emits (TTYPE / TFORM / TDIM / TUNIT
// / TZERO / TSCAL / TNULL) and keywords other writers may have stamped
// on a file (TDISP / TBCOL).  TFIELDS and NAXIS1 are NOT per-column
// (they're updated separately via update_int_card).
const PER_COLUMN_PREFIXES: &[&str] = &[
    "TTYPE", "TFORM", "TDIM", "TUNIT", "TZERO", "TSCAL", "TNULL",
    "TDISP", "TBCOL",
];

fn parse_per_column_card(card: &str) -> Option<(&'static str, usize)> {
    if card.len() < 8 {
        return None;
    }
    let keyword = card[..8].trim_end();
    for &prefix in PER_COLUMN_PREFIXES {
        if let Some(rest) = keyword.strip_prefix(prefix) {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(n) = rest.parse::<usize>() {
                    if n >= 1 {
                        return Some((prefix, n));
                    }
                }
            }
        }
    }
    None
}

fn rebuild_per_column_card(card: &str, prefix: &'static str, new_n: usize) -> String {
    let padded_keyword = format!("{:<8}", format!("{}{}", prefix, new_n));
    if card.len() <= 8 {
        return padded_keyword;
    }
    format!("{}{}", padded_keyword, &card[8..])
}

fn renumber_per_column_cards(
    cards: &mut [String], pivot_idx: usize, delta: i64,
) -> PyResult<()> {
    let threshold = pivot_idx + 1;  // 1-based threshold
    for card in cards.iter_mut() {
        if let Some((prefix, n)) = parse_per_column_card(card) {
            let should_renumber = match delta {
                1 => n >= threshold,
                -1 => n > threshold,
                _ => return Err(PyValueError::new_err(
                    "internal: renumber delta must be +/-1")),
            };
            if should_renumber {
                let new_n = (n as i64 + delta) as usize;
                *card = rebuild_per_column_card(card, prefix, new_n);
            }
        }
    }
    Ok(())
}

fn drop_per_column_cards(cards: &mut Vec<String>, delete_idx: usize) {
    let target = delete_idx + 1;
    cards.retain(|card| match parse_per_column_card(card) {
        Some((_, n)) => n != target,
        None => true,
    });
}

fn insert_cards_before_end(
    cards: &mut Vec<String>, new_cards: &[String],
) -> PyResult<()> {
    let end_idx = cards.iter().rposition(|c| {
        c.len() >= 3 && c[..3].trim() == "END"
    }).ok_or_else(|| PyValueError::new_err(
        "header missing END card"))?;
    for (i, c) in new_cards.iter().enumerate() {
        cards.insert(end_idx + i, c.clone());
    }
    Ok(())
}

fn update_int_card(
    cards: &mut [String], keyword: &str, new_value: i64,
) -> PyResult<()> {
    let comment = match keyword {
        "NAXIS1" => "width of table in bytes",
        "NAXIS2" => "number of rows in table",
        "TFIELDS" => "number of columns",
        _ => "",
    };
    let new_card = card_int(keyword, new_value, comment)
        .trim_end().to_string();
    for card in cards.iter_mut() {
        if card.len() >= 8 && card[..8].trim_end() == keyword {
            *card = new_card;
            return Ok(());
        }
    }
    Err(PyValueError::new_err(format!(
        "header missing required keyword {}", keyword)))
}

fn build_new_column_cards(col: &WriteColumn, insert_idx: usize) -> Vec<String> {
    let n = insert_idx + 1;
    let tform = match col.var_kind {
        Some(desc) => format!("1{}{}", desc, col.tform_letter),
        None => format!("{}{}", col.repeat, col.tform_letter),
    };
    let mut out = Vec::new();
    out.push(card_string(
        &format!("TTYPE{}", n), &col.name, "label for column")
        .trim_end().to_string());
    out.push(card_string(
        &format!("TFORM{}", n), &tform, "data format of column")
        .trim_end().to_string());
    if let Some(tdim) = &col.tdim {
        out.push(card_string(
            &format!("TDIM{}", n), tdim,
            "array dimensions (FORTRAN, fastest-first)")
            .trim_end().to_string());
    }
    if let Some(tz) = col.tzero {
        out.push(card_uint(
            &format!("TZERO{}", n), tz,
            "offset for unsigned integer (unsigned-int trick)")
            .trim_end().to_string());
    }
    if let Some(unit) = &col.tunit {
        out.push(card_string(
            &format!("TUNIT{}", n), unit, "physical unit of column")
            .trim_end().to_string());
    }
    out
}

// --------------------------------------------------------------------
// Building a WriteColumn from the user's ndarray input
// --------------------------------------------------------------------

fn build_single_write_column(
    py: Python<'_>,
    name: &str,
    data: &Bound<'_, PyAny>,
    unit: Option<&str>,
    bit_packed: bool,
) -> PyResult<WriteColumn> {
    use super::write_setup::BitColumnsSpec;
    let np = py.import("numpy")?;
    let arr_dtype = data.getattr("dtype")?;
    let arr_kind: String = arr_dtype.getattr("kind")?.extract()?;
    // VLA is dispatched by the caller; this helper is fixed-only.
    if arr_kind == "O" {
        return Err(PyValueError::new_err(
            "internal: build_single_write_column called with Object \
             dtype (caller should route to insert_vla_column_impl)"));
    }
    let arr_shape: Vec<usize> = data.getattr("shape")?.extract()?;
    if arr_shape.is_empty() {
        return Err(PyValueError::new_err(
            "insert_column: data must have at least one dimension (rows)"));
    }
    let per_cell_shape: Vec<usize> = arr_shape[1..].to_vec();

    // Build np.dtype([(name, arr.dtype.str, per_cell_shape)]) — when
    // per_cell_shape is empty, the 2-tuple form avoids numpy's
    // empty-subarray rejection.
    let dtype_str: String = arr_dtype.getattr("str")?.extract()?;
    let field_tuple = if per_cell_shape.is_empty() {
        PyTuple::new(py, [name.into_py_any(py)?, dtype_str.into_py_any(py)?])?
    } else {
        let shape_py = PyTuple::new(py,
            per_cell_shape.iter().map(|d| d.into_py_any(py).unwrap()))?;
        PyTuple::new(py, [
            name.into_py_any(py)?,
            dtype_str.into_py_any(py)?,
            shape_py.into_py_any(py)?,
        ])?
    };
    let descr = PyList::new(py, [field_tuple])?;
    let np_dtype = np.getattr("dtype")?.call1((descr,))?;

    let units_dict = if let Some(u) = unit {
        let d = PyDict::new(py);
        d.set_item(name, u)?;
        Some(d)
    } else {
        None
    };
    // bit_packed=true on a fixed column opts it into FITS X (one
    // bit per element, MSB-packed); the classifier rejects if the
    // column isn't b1.
    let bit_columns_spec = if bit_packed {
        let mut set = std::collections::HashSet::new();
        set.insert(name.to_uppercase());
        Some(BitColumnsSpec::Names(set))
    } else {
        None
    };
    let cols = dtype_to_write_columns(
        &np_dtype,
        units_dict.as_ref(),
        /* var_dtypes = */ None,
        bit_columns_spec.as_ref(),
        /* descriptor (irrelevant for fixed) = */ 'P',
    )?;
    if cols.len() != 1 {
        return Err(PyValueError::new_err(
            "internal: dtype_to_write_columns returned wrong column count"));
    }
    Ok(cols.into_iter().next().unwrap())
}

// Build a synthetic Column matching the new WriteColumn so we can reuse
// acquire_per_column_array / column_transform / apply_transform_cell to
// validate the user's input and encode each row's bytes.
fn synth_column_for_encode(wc: &WriteColumn) -> PyResult<Column> {
    Ok(Column {
        name: wc.name.clone(),
        tform_letter: wc.tform_letter,
        repeat: wc.repeat,
        tdim: wc.tdim.as_ref().map(|s| parse_tdim_for_encode(s)).transpose()?,
        byte_offset: 0,
        byte_width: wc.byte_width,
        var_kind: wc.var_kind,
        tscal: 1.0,
        tzero: wc.tzero.map(|t| t as f64).unwrap_or(0.0),
        tnull: None,
        tunit: wc.tunit.clone(),
    })
}

// Parse a TDIM string like "(4,3)" back into a Vec<usize> for the
// synthetic Column.  Matches the format produced by
// build_bintable_header_cards.
fn parse_tdim_for_encode(s: &str) -> PyResult<Vec<usize>> {
    let inner = s.trim()
        .trim_start_matches('(')
        .trim_end_matches(')');
    let mut out = Vec::new();
    for part in inner.split(',') {
        let v: usize = part.trim().parse().map_err(|_| {
            PyValueError::new_err(format!(
                "internal: synthesized TDIM '{}' failed to parse", s))
        })?;
        out.push(v);
    }
    Ok(out)
}

// --------------------------------------------------------------------
// THEAP guard (shared with repack)
// --------------------------------------------------------------------

fn reject_non_default_theap(
    cards: &[String], nrows: usize, row_width: usize,
) -> PyResult<()> {
    let theap_raw = parse_keyword(cards, "THEAP").unwrap_or(0);
    let default_main_bytes = (nrows as u64) * (row_width as u64);
    if theap_raw > 0 && (theap_raw as u64) != default_main_bytes {
        return Err(PyValueError::new_err(format!(
            "schema edit: file has non-default THEAP={} (main rows end at \
             {}); insert/delete writes the heap at the default position \
             and would corrupt this file.  Workaround: rewrite the file \
             through a fresh create_table_hdu + write",
            theap_raw, default_main_bytes)));
    }
    Ok(())
}
