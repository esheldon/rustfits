// TableHDU pyclass + #[pymethods] impl block, plus the read-side
// __getitem__ classifier (TableKey) and the SingleColumnSubset /
// ColumnSubset pyclasses used for chained subset reads.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PySlice, PyTuple};
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;

use crate::common::{
    check_not_tainted, parse_keyword, parse_string_keyword, FileHandle,
    FileLayout, HduOffsets, TaintFlag,
};
use crate::hdu::HDU;

use super::columns::{parse_columns, parse_table_meta, Column, TableMeta};
use super::edit::{delete_column_impl, insert_column_impl};
use super::read::{
    build_numpy_dtype, field_dtype_and_shape, read_one_column, read_table,
};
use super::setitem::{
    classify_setitem_key, setitem_cell, setitem_fancy_rows,
    setitem_fancy_rows_vla_aware, setitem_multi_columns, setitem_row_slice,
    setitem_row_slice_vla_aware, setitem_single_column,
    setitem_single_column_vla, setitem_single_row,
    setitem_single_row_vla_aware, try_extract_column_name,
    write_column_subset_at_rows, write_one_column_at_rows, SetItemKey,
};
use super::write_fixed::{
    append_fixed_only, determine_input_nrows, write_fixed_only,
};
use super::write_vla::{
    any_var_column, append_vla_aware, repack_table_heap, write_vla_aware,
};

/// A binary-table HDU (``XTENSION='BINTABLE'``).
///
/// Returned by indexing a :class:`FITS` object at a position
/// containing a binary table — e.g. ``hdu = fits[1]``.  Carries
/// the table's schema (column names, dtypes, units, scaling) and
/// the I/O surface for reading and modifying rows.
///
/// All reads return numpy structured arrays whose dtype reflects
/// the on-disk schema after the default ``scale=True`` mapping:
/// the unsigned-int trick (BSCALE=1 + BZERO=2**(n-1)) promotes
/// to the matching unsigned dtype, other scaling produces ``f8``,
/// and TNULL-bearing integer columns can be returned as
/// ``numpy.ma.MaskedArray`` via ``mask_null=True``.
///
/// Indexing is symmetric for reads and writes: anything readable
/// via ``hdu[key]`` (rows, single column, multi-column subset,
/// ``(row, col)`` cell) is writable via ``hdu[key] = value``.
/// See the individual methods for details.
///
/// Examples
/// --------
/// Read a whole table::
///
///     arr = hdu.read()
///
/// Read a row subset::
///
///     rows = np.arange(20, 30)
///     ra_dec = hdu.read(rows=rows)
///
/// Read a column subset::
///
///     columns = ['ra', 'dec']
///     ra_dec = hdu.read(columns=columns)
///     ra_dec = hdu.read(rows=rows, columns=columns)
///
/// Slicing and column subsets::
///
///     arr = hdu[35:100]
///     arr = hdu[rows]
///     single = hdu[35]
///
///     # a column subset.  The read does not occur until
///     # rows are specified or .read() is called
///     ra_dec = hdu[columns].read()
///     ra_dec = hdu[columns][rows]
///
/// Writing is symmetric::
///
///     hdu[35:100] = data
///     hdu[rows] = data
///     hdu[columns][rows] = sub_data
///
/// Overwrite the whole table::
///
///    hdu[:] = new_data
///    hdu.write(new_data)
///
/// Appending::
///
///     hdu.append(new_data)
///
/// Iterate rows lazily via slicing::
///
///     for chunk in (hdu[i:i+1000] for i in range(0, len(hdu), 1000)):
///         process(chunk)
///
/// Notes
/// -----
/// Compressed binary tables (``ZTABLE=T``) return the subclass
/// :class:`CompressedTableHDU` instead, which overrides the read
/// and write methods to handle per-tile decompression.
#[pyclass(extends = HDU, subclass)]
pub(crate) struct TableHDU {
    // Lazily-populated per-HDU parsed-metadata cache.  See `meta()`
    // for the hot-path accessor; entry is `(version, meta)` where
    // `version` is the value of the base-HDU `cards_version` at
    // the time of the parse.  None until the first call; auto-
    // invalidates on any cards mutation because the next `meta()`
    // call observes a higher version (Phase 1 atomic) and re-parses.
    // See `CompressedImageHDU.meta_cache` for the parallel pattern
    // on the image side.
    pub(crate) meta_cache: Arc<Mutex<Option<(u64, Arc<TableMeta>)>>>,
}

impl TableHDU {
    pub(crate) fn new(
        header: Vec<String>,
        index: usize,
        filename: String,
        offsets: Arc<HduOffsets>,
        layout: Arc<FileLayout>,
        file: FileHandle,
        tainted: TaintFlag,
    ) -> (Self, HDU) {
        (
            TableHDU::new_empty_cache(),
            HDU::new(header, index, filename, offsets, layout, file, tainted),
        )
    }

    // Factory for the subclass-construction path: CompressedTableHDU
    // builds its PyClassInitializer with `.add_subclass(TableHDU)`,
    // which requires a fresh value.
    pub(crate) fn new_empty_cache() -> Self {
        TableHDU { meta_cache: Arc::new(Mutex::new(None)) }
    }

    // Return the parsed-once metadata for this HDU.  Hot-path
    // accessor: one Mutex lock + Acquire version load + (on hit)
    // an Arc clone.  On miss (first call, or any cards mutation
    // since the previous call) takes a header snapshot under the
    // cards mutex and re-parses.  See `TableMeta` for what's cached.
    //
    // Callers reach this method while also needing the base HDU
    // (for offsets / file / etc.) by going through
    // `slf.as_super()`, which borrows up the class chain instead
    // of consuming `slf` — keeping both alive for the call.
    pub(crate) fn meta(
        &self, super_: &HDU,
    ) -> PyResult<Arc<TableMeta>> {
        let cur_version = super_.cards_version.load(Ordering::Acquire);
        {
            let cache = self.meta_cache.lock()
                .map_err(|_| PyIOError::new_err("meta cache poisoned"))?;
            if let Some((v, m)) = &*cache {
                if *v == cur_version {
                    return Ok(Arc::clone(m));
                }
            }
        }
        // Miss: re-parse outside the lock so concurrent readers
        // racing the same miss each parse once but only one wins
        // the cache slot — same loss-of-work pattern as the
        // ZIMAGE meta cache.
        let cards = super_.header_snapshot()?;
        let meta = Arc::new(parse_table_meta(&cards)?);
        let mut cache = self.meta_cache.lock()
            .map_err(|_| PyIOError::new_err("meta cache poisoned"))?;
        *cache = Some((cur_version, Arc::clone(&meta)));
        Ok(meta)
    }
}

// Per-column line in the TableHDU.__repr__ column-info block.  Returns
// the numpy dtype string + an optional shape annotation:
//   - fixed scalar (repeat == 1, no TDIM):         dtype, None
//   - fixed multi (repeat > 1 or TDIM):            dtype, Some("array[a,b,...]")
//   - variable-length (P/Q):                       inner-dtype, Some("array[var]")
// Scaled dtype is shown when possible (e.g. unsigned-int trick → u2),
// falling back to the unscaled mapping if scale-based dtype resolution
// would error (C/M with non-default TSCAL/TZERO).
fn column_repr_info(col: &Column) -> (String, Option<String>) {
    if col.var_kind.is_some() {
        let inner = match col.tform_letter {
            'L' => "?",
            'B' => "u1",
            'I' => "i2",
            'J' => "i4",
            'K' => "i8",
            'E' => "f4",
            'D' => "f8",
            'C' => "c8",
            'M' => "c16",
            'A' => "S",
            _   => return (col.tform_letter.to_string(),
                           Some("array[var]".to_string())),
        };
        return (inner.to_string(), Some("array[var]".to_string()));
    }
    let (dtype_str, shape) = field_dtype_and_shape(col, /* scale = */ true)
        .or_else(|_| field_dtype_and_shape(col, /* scale = */ false))
        .unwrap_or_else(|_| ("?".to_string(), Vec::new()));
    let shape_str = if shape.is_empty() {
        None
    } else {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        Some(format!("array[{}]", dims.join(",")))
    };
    (dtype_str, shape_str)
}

#[pymethods]
impl TableHDU {
    /// Compute and store the ``DATASUM`` checksum card.
    ///
    /// ``DATASUM`` is the unsigned-32-bit checksum of the HDU's
    /// data section, per the FITS Checksum Convention.  Call
    /// after any write that changes the data — :meth:`write`,
    /// :meth:`append`, ``__setitem__``, :meth:`insert_column`,
    /// :meth:`delete_column`, :meth:`repack`.  rustfits does NOT
    /// auto-refresh checksums on mutation; the user opts in
    /// explicitly because checksum computation can be expensive
    /// on large data sections.
    ///
    /// See Also
    /// --------
    /// add_checksum : Also compute the full HDU ``CHECKSUM`` card.
    /// verify_datasum : Compare the stored ``DATASUM`` against the
    ///     current data bytes.
    fn add_datasum(slf: PyRef<'_, Self>) -> PyResult<()> {
        let super_ = slf.into_super();
        crate::hdu_image::checksum_hdu_add_datasum(&super_, "DATASUM")
    }

    /// Compute and store both ``DATASUM`` and ``CHECKSUM`` cards.
    ///
    /// ``CHECKSUM`` is the encoded complement of the running
    /// checksum over (header + data) so that the total HDU
    /// checksum lands on ``0xFFFFFFFF``, per the FITS Checksum
    /// Convention.  This is the call most users want — it writes
    /// both cards atomically.
    ///
    /// See :meth:`add_datasum` for the manual-refresh contract.
    fn add_checksum(slf: PyRef<'_, Self>) -> PyResult<()> {
        let super_ = slf.into_super();
        crate::hdu_image::checksum_hdu_add_checksum(
            &super_, "CHECKSUM", "DATASUM",
        )
    }

    /// Verify the stored ``DATASUM`` against the current data.
    ///
    /// Returns
    /// -------
    /// bool or None
    ///     ``True`` if the stored ``DATASUM`` matches the current
    ///     data section; ``False`` if it doesn't; ``None`` if the
    ///     ``DATASUM`` card is absent.
    fn verify_datasum(slf: PyRef<'_, Self>) -> PyResult<Option<bool>> {
        let super_ = slf.into_super();
        crate::hdu_image::checksum_hdu_verify_datasum(&super_, "DATASUM")
    }

    /// Verify the stored ``CHECKSUM`` over the full HDU.
    ///
    /// Returns
    /// -------
    /// bool or None
    ///     ``True`` if the stored ``CHECKSUM`` matches the current
    ///     header + data; ``False`` if it doesn't; ``None`` if the
    ///     ``CHECKSUM`` card is absent.
    fn verify_checksum(slf: PyRef<'_, Self>) -> PyResult<Option<bool>> {
        let super_ = slf.into_super();
        crate::hdu_image::checksum_hdu_verify_checksum(&super_, "CHECKSUM")
    }

    // Multi-line, fitsio-style repr.  Shows file, extension, type,
    // EXTNAME (if present), row count, and per-column dtype + shape
    // annotation.  Column lines are dynamically aligned to the longest
    // column name.
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.as_super();
        // Try the cached meta first; fall back to a fresh cards
        // snapshot + tolerant parse if the file is degenerate enough
        // that meta parsing raises — repr must never crash.
        let cached = slf.meta(super_).ok();
        let cards = super_.header_snapshot()?;
        let (columns, nrows): (Vec<Column>, i64) = match &cached {
            Some(m) => (m.columns.clone(), m.nrows as i64),
            None => (
                parse_columns(&cards)?,
                parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0),
            ),
        };
        let extname = parse_string_keyword(&cards, "EXTNAME");

        let mut out = String::new();
        out.push_str(&format!("  file: {}\n", super_.filename));
        out.push_str(&format!("  extension: {}\n", super_.index));
        out.push_str("  type: BINARY_TBL\n");
        if let Some(name) = extname {
            out.push_str(&format!("  extname: {}\n", name));
        }
        out.push_str(&format!("  rows: {}\n", nrows));
        out.push_str("  column info:\n");

        let max_name_len = columns.iter().map(|c| c.name.len()).max().unwrap_or(0);
        let name_w = max_name_len + 4;
        for col in &columns {
            let (dtype_str, shape_str) = column_repr_info(col);
            out.push_str(&format!(
                "    {:<w$}{}", col.name, dtype_str, w = name_w,
            ));
            if let Some(shape) = shape_str {
                out.push_str(&format!("  {}", shape));
            }
            if let Some(unit) = &col.tunit {
                out.push_str(&format!("  ({})", unit));
            }
            out.push('\n');
        }
        Ok(out)
    }

    /// The numpy structured dtype the table reads into.
    ///
    /// Reflects the default-read (``scale=True``) dtype — i.e.
    /// columns with the TSCAL/TZERO unsigned trick appear as
    /// ``u2`` / ``u4`` / ``u8`` / ``i1``, and other scaled columns
    /// as ``f8``.  Useful for inspecting the column layout (names,
    /// per-cell shapes, types) without paying for an actual read.
    ///
    /// Returns
    /// -------
    /// numpy.dtype
    ///     Structured dtype with one field per column.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        build_numpy_dtype(py, &meta.columns, /* scale = */ true)
    }

    /// Per-column units (``TUNITn``), as a dict.
    ///
    /// Maps each column name (case preserved) to its ``TUNITn``
    /// string, or ``None`` when ``TUNITn`` is unset for that
    /// column.  Dict iteration follows on-disk column order.
    ///
    /// Informational only — nothing in the read path consumes
    /// units.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{column_name: unit_or_None}``.
    #[getter]
    fn units(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        let dict = PyDict::new(py);
        for col in &meta.columns {
            dict.set_item(&col.name, col.tunit.as_deref())?;
        }
        Ok(dict.unbind())
    }

    /// Number of rows in the table.
    ///
    /// Reads the ``NAXIS2`` header keyword.  Equivalent to
    /// ``len(hdu)``; both are provided for symmetry with numpy
    /// (``len(arr)``) and pandas (``df.nrows``) idioms.
    #[getter]
    fn nrows(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.nrows as usize)
    }

    /// Number of columns in the table (``TFIELDS``).
    #[getter]
    fn ncols(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.columns.len())
    }

    /// Column names in on-disk order, as a tuple.
    ///
    /// Names are returned with their on-disk case preserved
    /// verbatim.  Lookup against this list (e.g. by :meth:`read`'s
    /// ``columns=`` argument) is case-insensitive throughout the
    /// API.  Returned as a tuple so the value is immutable from
    /// the caller side.
    ///
    /// Returns
    /// -------
    /// tuple of str
    #[getter]
    fn colnames(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        let names: Vec<&str> =
            meta.columns.iter().map(|c| c.name.as_str()).collect();
        Ok(PyTuple::new(py, &names)?.unbind())
    }

    // Pythonic length: `len(table_hdu)` == row count.  Mirrors
    // `len(structured_array)` for the equivalent numpy structured
    // array a full read would return.
    //
    // No `///` docstring here: pyo3 installs __len__ as the C
    // `tp_len` slot, whose canonical "Return len(self)." overrides
    // any per-method docstring.  The semantics are covered in the
    // class docstring instead.
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.nrows as usize)
    }

    /// Read rows from the table into a numpy structured array.
    ///
    /// Parameters
    /// ----------
    /// rows : slice, list of int, or None, optional
    ///     Rows to read.  ``None`` (default) reads every row in file
    ///     order.  A slice or iterable of ints selects a subset;
    ///     negative indices are supported.  Iterables are deduped
    ///     with first-occurrence-wins ordering.
    /// columns : list of str, or None, optional
    ///     Column names to read.  ``None`` (default) reads every
    ///     column in file order.  A list selects + reorders;
    ///     matching is case-insensitive against the table's column
    ///     names.
    /// scale : bool, optional
    ///     If ``True`` (default), apply ``TSCAL`` / ``TZERO``
    ///     scaling: the unsigned-int trick promotes to the matching
    ///     unsigned dtype with no precision loss, and other scaling
    ///     produces ``f8``.  If ``False``, return raw stored values
    ///     in the on-disk BITPIX dtype.
    /// mask_null : bool, optional
    ///     If ``True``, return a ``numpy.ma.MaskedArray`` with
    ///     per-field bool masks set ``True`` where the stored
    ///     integer equals ``TNULLn``.  The mask compare is in
    ///     stored-int space (pre-scaling), so it composes
    ///     correctly with the ``TSCAL`` / ``TZERO`` paths.  Only
    ///     applies to integer fixed-width columns; variable-length
    ///     columns with ``TNULL`` are rejected.  Default ``False``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray or numpy.ma.MaskedArray
    ///     Structured array of shape ``(n_selected,)`` with one
    ///     field per selected column.  Dtype reflects the
    ///     ``scale`` choice (scaled values for ``True``, raw
    ///     stored dtype for ``False``).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a row index is out of range, a column name is unknown,
    ///     or ``mask_null=True`` is requested on a variable-length
    ///     column carrying ``TNULL``.
    ///
    /// Notes
    /// -----
    /// Both the ``rows=`` and ``columns=`` subsets validate fully
    /// before any file I/O happens, so an invalid selection leaves
    /// the file untouched.
    ///
    /// Examples
    /// --------
    /// Read the whole table::
    ///
    ///     arr = hdu.read()
    ///
    /// Read three columns from rows 100..200::
    ///
    ///     arr = hdu.read(rows=slice(100, 200),
    ///                    columns=["RA", "DEC", "MAG"])
    ///
    /// Read with masking on a column that has ``TNULL=-99``::
    ///
    ///     arr = hdu.read(mask_null=True)
    ///     assert arr["FLAG"].mask.any()
    #[pyo3(signature = (*, rows=None, columns=None, scale=true, mask_null=false))]
    fn read(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        columns: Option<Vec<String>>,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_table(
            py, &meta, data_offset, &super_.file, rows, columns, scale,
            mask_null,
        )
    }

    // Read a single column into a plain (non-structured) ndarray of
    // shape `(n_selected_rows,) + field_shape`.  rows= mirrors read()'s
    // semantics.  `as_bytes=True` is meaningful only for A (character)
    // columns; it returns the on-disk bytes in an S<n> field with no
    // decode, no null-truncation, and no trailing-space strip — useful
    // when a column has non-ASCII bytes that the default U decode would
    // reject.  `scale` and `mask_null` match read(); when mask_null=True
    // and this column carries TNULL, returns a numpy.ma.MaskedArray.
    /// Read a single column into a plain (non-structured) ndarray.
    ///
    /// Equivalent to ``hdu.read(columns=[name])[name]`` but skips the
    /// structured-array packaging — useful when you only want one
    /// column's data.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Column name.  Case-insensitive against the table's
    ///     ``TTYPEn`` values.
    /// rows : slice, list of int, or None, optional
    ///     Same semantics as :meth:`read`'s ``rows=``.
    /// as_bytes : bool, optional
    ///     Only meaningful for ``A`` (character) columns.  If
    ///     ``True``, return the on-disk bytes in an ``S<n>`` field
    ///     with no decode, no NUL-truncation, and no trailing-space
    ///     strip — useful when a column has non-ASCII bytes that
    ///     the default ``U`` decode would reject.  Default ``False``.
    /// scale : bool, optional
    ///     Same as :meth:`read`'s ``scale=``.
    /// mask_null : bool, optional
    ///     If ``True`` and this column carries ``TNULL``, return a
    ///     ``numpy.ma.MaskedArray``.  Default ``False``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray or numpy.ma.MaskedArray
    ///     Array of shape ``(n_selected,) + field_shape`` —
    ///     ``field_shape`` is empty for scalar columns, ``(repeat,)``
    ///     or the ``TDIM`` shape for subarray columns.
    #[pyo3(signature = (name, *, rows=None, as_bytes=false, scale=true, mask_null=false))]
    fn read_column(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        name: &str,
        rows: Option<&Bound<'_, PyAny>>,
        as_bytes: bool,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_one_column(
            py, &meta, data_offset, &super_.file, name, rows, as_bytes,
            scale, mask_null,
        )
    }

    // hdu[key] dispatches based on what `key` looks like:
    //
    //   slice or iterable-of-int → reads rows now, returns ndarray
    //     (equivalent to hdu.read(rows=key))
    //   single str/bytes/np.str_/np.bytes_ → returns a SingleColumnSubset
    //     (no read; user must add [rows] to trigger read_column)
    //   iterable-of-str/bytes → returns a ColumnSubset
    //     (no read; user must add [rows] to trigger read with columns=)
    //
    // Specifying a column or columns alone never invokes I/O — only rows
    // do.  Empty sequences are rejected as ambiguous.
    fn __getitem__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let kind = classify_table_key(key)?;
        match kind {
            TableKey::Rows => {
                let pyref = slf.borrow();
                let super_ = pyref.as_super();
                let meta = pyref.meta(super_)?;
                let data_offset = super_.offsets.data_offset();
                read_table(
                    py, &meta, data_offset, &super_.file, Some(key), None,
                    /* scale = */ true, /* mask_null = */ false,
                )
            }
            TableKey::SingleRow(idx) => {
                let pyref = slf.borrow();
                let super_ = pyref.as_super();
                let meta = pyref.meta(super_)?;
                let data_offset = super_.offsets.data_offset();
                // Wrap idx in a single-element list so resolve_rows
                // handles negative-index normalization and range
                // validation the same way it does for `hdu[[idx]]`.
                let one = PyList::new(py, [idx])?;
                let arr_py = read_table(
                    py, &meta, data_offset, &super_.file,
                    Some(one.as_any()), None,
                    /* scale = */ true, /* mask_null = */ false,
                )?;
                // arr is shape (1,); index [0] yields numpy's 0-d
                // record (np.void), matching `structured_arr[i]`
                // semantics for the user.
                let arr_bound = arr_py.bind(py);
                Ok(arr_bound.get_item(0)?.unbind())
            }
            TableKey::SingleColumn(name) => {
                let hdu_py: Py<TableHDU> = slf.clone().unbind();
                Ok(Py::new(py, SingleColumnSubset { hdu: hdu_py, name })?
                    .into())
            }
            TableKey::MultiColumns(names) => {
                let hdu_py: Py<TableHDU> = slf.clone().unbind();
                Ok(Py::new(py, ColumnSubset { hdu: hdu_py, columns: names })?
                    .into())
            }
        }
    }

    /// Bulk-write data into the table's data section.
    ///
    /// Overwrites all ``NAXIS2`` rows; for appending new rows
    /// instead, use :meth:`append`.  Accepts three input forms,
    /// all normalizing through the same per-column strip-write
    /// kernel:
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray, dict, or list/tuple of ndarrays
    ///     * **Structured ndarray** — field names must match the
    ///       HDU's columns (extras, missing, or duplicates
    ///       rejected); field order may differ from HDU order.
    ///       ``len(data)`` must equal ``NAXIS2``.
    ///     * **Dict** ``{name: ndarray}`` — one entry per HDU
    ///       column; extras / missing rejected.  Each value is a
    ///       per-column ndarray with shape ``(NAXIS2,) +
    ///       per_cell_shape``.
    ///     * **List or tuple of ndarrays** with ``names=[...]`` —
    ///       parallel sequences; same per-column model as dict.
    /// names : list of str, optional
    ///     Required only when ``data`` is a list/tuple of ndarrays.
    ///     Ignored for the structured-ndarray and dict forms.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Field name mismatch, missing/extra columns, length
    ///     mismatch with ``NAXIS2``, or per-cell shape mismatch.
    ///
    /// Notes
    /// -----
    /// Validate-then-mutate: any dtype/shape error is raised
    /// BEFORE the file is touched, so an invalid input leaves the
    /// table unchanged.
    ///
    /// See Also
    /// --------
    /// append : Add new rows to the table.
    /// __setitem__ : Modify a subset of rows / columns / cells.
    #[pyo3(signature = (data, *, names=None))]
    fn write(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let super_ = slf.into_super();
        check_not_tainted(&super_.tainted)?;
        let cards = super_.header_snapshot()?;
        let nrows =
            parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize;
        let row_width =
            parse_keyword(&cards, "NAXIS1").unwrap_or(0).max(0) as usize;
        let columns = parse_columns(&cards)?;
        let data_offset = super_.offsets.data_offset();

        if any_var_column(&columns) {
            write_vla_aware(
                py, &super_, &cards, &columns, data, names,
                nrows, row_width, data_offset)
        } else {
            write_fixed_only(
                py, &super_, &columns, data, names,
                nrows, row_width, data_offset)
        }
    }

    // hdu[key] = value dispatches based on what `key` looks like:
    //
    //   bare int (not bool) → single-row write at row index `key`
    //     (negative supported); `value` must be a numpy.void record
    //     or a length-1 structured ndarray.
    //   slice → range-of-rows write; `value` must be a structured
    //     ndarray of length equal to the slicelength.  step=1 uses
    //     the bulk-write fast path; step>1 does per-row writes.
    //     step<=0 is rejected.
    //   single str/bytes/np.str_/np.bytes_ → whole-column write
    //     across all rows; `value` must be an ndarray of shape
    //     (nrows,) + per-cell shape, matching what __getitem__
    //     would return for that column.
    //
    // Multi-column subset writes, (row, col) tuple writes, and fancy
    // row-list writes are rejected; add when a use case shows up.
    fn __setitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let super_ = slf.into_super();
        check_not_tainted(&super_.tainted)?;
        let cards = super_.header_snapshot()?;
        let nrows =
            parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize;
        let row_width =
            parse_keyword(&cards, "NAXIS1").unwrap_or(0).max(0) as usize;
        let columns = parse_columns(&cards)?;
        let data_offset = super_.offsets.data_offset();
        let kind = classify_setitem_key(key)?;
        let has_vla = any_var_column(&columns);
        match kind {
            SetItemKey::SingleRow(i) => {
                if has_vla {
                    setitem_single_row_vla_aware(
                        py, &super_, &cards, &columns, nrows, row_width,
                        i, value, data_offset)
                } else {
                    setitem_single_row(
                        py, &columns, &super_.file, data_offset, nrows,
                        row_width, i, value, &super_.tainted)
                }
            }
            SetItemKey::RowSlice => {
                let slice_py = key.cast::<PySlice>()?;
                if has_vla {
                    setitem_row_slice_vla_aware(
                        py, &super_, &cards, &columns, nrows, row_width,
                        slice_py, value, data_offset)
                } else {
                    setitem_row_slice(
                        py, &columns, &super_.file, data_offset, nrows,
                        row_width, slice_py, value, &super_.tainted)
                }
            }
            SetItemKey::SingleColumn(name) => {
                let name_u = name.to_uppercase();
                let col_idx = columns.iter()
                    .position(|c| c.name.to_uppercase() == name_u);
                if let Some(idx) = col_idx {
                    if columns[idx].var_kind.is_some() {
                        return setitem_single_column_vla(
                            py, &super_, &cards, &columns, idx, nrows,
                            row_width, value, data_offset);
                    }
                }
                setitem_single_column(
                    py, &columns, &super_.file, data_offset, nrows,
                    row_width, &name, value, &super_.tainted)
            }
            SetItemKey::FancyRows(rows) => {
                if has_vla {
                    setitem_fancy_rows_vla_aware(
                        py, &super_, &cards, &columns, nrows, row_width,
                        &rows, value, data_offset)
                } else {
                    setitem_fancy_rows(
                        py, &columns, &super_.file, data_offset, nrows,
                        row_width, &rows, value, &super_.tainted)
                }
            }
            SetItemKey::MultiColumns(names) => setitem_multi_columns(
                py, &super_, &cards, &columns, nrows, row_width,
                &names, value, data_offset),
            SetItemKey::Cell(i, name) => setitem_cell(
                py, &super_, &cards, &columns, nrows, row_width,
                i, &name, value, data_offset),
        }
    }

    /// Append rows to the end of the table.
    ///
    /// Grows ``NAXIS2`` in the header and the data section to fit
    /// the new rows.  For HDUs that are not the last on disk, the
    /// file tail is shifted forward and every later HDU's offsets
    /// are bumped in lockstep; previously-issued handles remain
    /// valid.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray, dict, or list/tuple of ndarrays
    ///     Same three input forms as :meth:`write`: a structured
    ///     ndarray, a ``{name: ndarray}`` dict, or a list/tuple
    ///     of per-column ndarrays paired with ``names=``.  Length
    ///     defines the number of new rows.
    /// names : list of str, optional
    ///     Required for the list/tuple form; ignored otherwise.
    ///
    /// Notes
    /// -----
    /// Validate-then-mutate: input is fully validated (columns,
    /// dtypes, shapes) before any file or header bytes are
    /// touched, so a dtype mismatch can't leave the file
    /// half-grown.
    ///
    /// Mid-write I/O failures taint the file — subsequent reads
    /// and writes will raise until the user closes and reopens.
    ///
    /// See Also
    /// --------
    /// extend : Alias of ``append``, kept for symmetry with
    ///     :meth:`ImageHDU.extend`.
    /// write : Overwrite all rows in place.
    #[pyo3(signature = (data, *, names=None))]
    fn append(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let super_: PyRefMut<HDU> = slf.into_super();
        check_not_tainted(&super_.tainted)?;
        let cards = super_.header_snapshot()?;
        let current_nrows = parse_keyword(&cards, "NAXIS2")
            .unwrap_or(0).max(0) as usize;
        let row_width = parse_keyword(&cards, "NAXIS1")
            .unwrap_or(0).max(0) as usize;
        let columns = parse_columns(&cards)?;
        let data_offset = super_.offsets.data_offset();

        // Validate-then-mutate: determine append size and run input
        // validation BEFORE touching the file, so a dtype error
        // leaves the file untouched.
        let append_nrows = determine_input_nrows(data, names)?;
        if append_nrows == 0 {
            return Ok(());
        }
        let new_nrows = current_nrows + append_nrows;

        if any_var_column(&columns) {
            append_vla_aware(
                py, &super_, &cards, &columns, data, names,
                current_nrows, append_nrows, new_nrows, row_width,
                data_offset)
        } else {
            append_fixed_only(
                py, &super_, &columns, data, names,
                current_nrows, append_nrows, new_nrows, row_width,
                data_offset)
        }
    }

    /// Alias for :meth:`append`.
    ///
    /// Kept for symmetry with :meth:`ImageHDU.extend` so generic
    /// code that iterates HDUs and calls ``.extend(...)`` on each
    /// continues to work.  The primary table-side name is
    /// :meth:`append` because that's the natural verb for adding
    /// rows to a table.
    #[pyo3(signature = (data, *, names=None))]
    fn extend(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Self::append(slf, py, data, names)
    }

    /// Rebuild the VLA heap, reclaiming orphan cells.
    ///
    /// VLA writes (``__setitem__`` on a variable-length column)
    /// always append new cell bytes to the end of the heap,
    /// leaving the old bytes as orphans referenced by no
    /// descriptor.  ``repack()`` walks every live descriptor,
    /// streams the referenced bytes into a compact new heap, and
    /// rewrites the descriptors to point at it.  If the heap
    /// shrinks, the on-disk file shrinks too: the last HDU uses
    /// ``set_len``, and a non-last HDU shifts the trailing HDUs
    /// backward in lockstep.
    ///
    /// No-op for tables without VLA columns or with an
    /// already-compact heap.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the file uses a non-default ``THEAP`` (heap offset
    ///     other than ``NAXIS1 * NAXIS2``).  Rustfits never emits
    ///     such files itself; the limitation only blocks repacking
    ///     files written by other tools with a custom heap offset.
    ///     Workaround: rewrite through a fresh
    ///     :meth:`FITS.create_table_hdu` + :meth:`write`.
    fn repack(slf: PyRef<'_, Self>) -> PyResult<()> {
        let super_ = slf.into_super();
        repack_table_heap(&super_)
    }

    /// Insert a new column into the table.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Column name (becomes ``TTYPEn``).  Must not duplicate
    ///     an existing column (case-insensitive check).
    /// data : numpy.ndarray
    ///     Column values, shape ``(NAXIS2,) + per_cell_shape``.
    ///     For fixed columns the dtype determines the FITS letter
    ///     (``i2`` / ``i4`` / ``i8`` / ``u1`` / ``u2`` / ``u4`` /
    ///     ``u8`` / ``f4`` / ``f8`` / ``c8`` / ``c16`` / ``b1`` +
    ///     ``S`` / ``U`` strings); the unsigned-int trick on
    ///     ``u2`` / ``u4`` / ``u8`` emits ``TZERO``.  For VLA
    ///     columns, pass Object dtype with one inner ndarray per
    ///     row and set ``inner_dtype=``.
    /// position : int, optional
    ///     0-based column index in the result, ``0..=ncols``.
    ///     ``ncols`` appends at the end (also the default when
    ///     none of position / after / before is set).  Mutually
    ///     exclusive with ``after`` and ``before``.
    /// after : str or int, optional
    ///     Insert after this column.  Accepts a name
    ///     (case-insensitive) or a 0-based integer index
    ///     (negative wraps).  Mutually exclusive with
    ///     ``position`` and ``before``.
    /// before : str or int, optional
    ///     Insert before this column.  Same rules as ``after``.
    ///     Mutually exclusive with ``position`` and ``after``.
    /// unit : str, optional
    ///     ``TUNITn`` string.
    /// inner_dtype : str, optional
    ///     Required when ``data`` is Object dtype (VLA insert).
    ///     Inner element dtype as a string: ``'f4'`` / ``'i4'`` /
    ///     ``'?'`` etc.  Maps to the FITS inner-element letter.
    /// heap_format : {'P', 'Q'}, optional
    ///     For VLA columns only.  ``'P'`` (default) uses 8-byte
    ///     descriptors with a 4 GB heap ceiling; ``'Q'`` uses
    ///     16-byte descriptors with no practical ceiling.
    /// bit_packed : bool, optional
    ///     For boolean columns only.  If ``True``, emit an ``X``
    ///     (or ``PX`` / ``QX`` for VLA) bit-packed column instead
    ///     of the default ``L`` (one byte per bool).  Default
    ///     ``False``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Duplicate name; multiple position kwargs set; unknown
    ///     position; dtype mismatch; row count mismatch with
    ///     ``NAXIS2``; ``inner_dtype`` / ``heap_format`` set on
    ///     non-Object input; or the file uses a non-default
    ///     ``THEAP`` (see :meth:`repack` for the same limitation).
    ///
    /// Notes
    /// -----
    /// Strip-based row shuffler bounds peak memory at ~1 MiB
    /// regardless of table size.  Existing VLA columns are
    /// preserved across the insert; their heap is relocated
    /// forward to sit after the new (wider) main rows.  Mid-write
    /// I/O failures taint the file (close + reopen to recover).
    #[pyo3(signature = (
        name, data, *, position=None, after=None, before=None, unit=None,
        inner_dtype=None, heap_format=None, bit_packed=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn insert_column(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
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
        let super_: PyRefMut<HDU> = slf.into_super();
        insert_column_impl(
            py, &super_, name, data, position, after, before, unit,
            inner_dtype, heap_format, bit_packed,
        )
    }

    /// Remove a column from the table.
    ///
    /// Parameters
    /// ----------
    /// key : str or int
    ///     Column name (case-insensitive) or a 0-based integer
    ///     index.  Negative indices wrap from the end.
    ///
    /// Notes
    /// -----
    /// Works on both fixed and VLA columns.  For a VLA column,
    /// the descriptor bytes are removed from each row but the
    /// heap cells the column referenced are left as-is — they
    /// become orphans that :meth:`repack` reclaims.  Existing
    /// other VLA columns are preserved across the delete; their
    /// heap relocates backward to sit after the new (shorter)
    /// main rows.
    ///
    /// Row shuffle runs in 1 MiB front-to-back strips so peak
    /// memory stays bounded.  Mid-write I/O failures taint the
    /// file (close + reopen to recover).
    fn delete_column(
        slf: PyRefMut<'_, Self>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let super_: PyRefMut<HDU> = slf.into_super();
        delete_column_impl(&super_, key)
    }
}

// What kind of selection the user passed to TableHDU.__getitem__.
// `Rows` covers both slices and integer iterables: in both cases the
// key flows through to read_table unchanged.  `SingleRow` is the
// bare-integer case (`hdu[5]`); read_table still does the I/O but
// the result is unwrapped to a numpy 0-d record (np.void) before
// returning, matching `structured_arr[i]` semantics.
pub(crate) enum TableKey {
    Rows,
    SingleRow(i64),
    SingleColumn(String),
    MultiColumns(Vec<String>),
}

// Inspect the __getitem__ key and decide which path to take.  Rules:
//   - PySlice                            → Rows  (read flowing path)
//   - bare int (not bool)                → SingleRow (np.void scalar)
//   - single str/bytes/np.str_/np.bytes_ → SingleColumn
//   - non-empty iterable
//       first element string-like        → MultiColumns
//       first element int-like           → Rows
//       mixed or unknown                 → ValueError
//   - empty iterable                     → ValueError (ambiguous)
//   - anything else                      → ValueError
pub(crate) fn classify_table_key(key: &Bound<'_, PyAny>) -> PyResult<TableKey> {
    if key.is_instance_of::<PySlice>() {
        return Ok(TableKey::Rows);
    }
    if let Some(name) = try_extract_column_name(key)? {
        return Ok(TableKey::SingleColumn(name));
    }
    // Bare integer (not bool — Python bool is a subclass of int and
    // would otherwise sneak through).  Float/non-int Python objects
    // are rejected by extract::<i64>.
    if !key.is_instance_of::<PyBool>() {
        if let Ok(idx) = key.extract::<i64>() {
            return Ok(TableKey::SingleRow(idx));
        }
    }
    let iter = key.try_iter().map_err(|_| PyValueError::new_err(
        "TableHDU[key] requires a slice, an int (row index), a \
         str/bytes column name, an iterable of ints (rows), or an \
         iterable of str/bytes (columns)"
    ))?;
    let items: Vec<Bound<'_, PyAny>> = iter.collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Err(PyValueError::new_err(
            "TableHDU[key] received an empty sequence (ambiguous: rows \
             or columns?); pass a non-empty selection or use read() with \
             explicit rows=/columns="
        ));
    }
    let first = &items[0];
    if try_extract_column_name(first)?.is_some() {
        let names: Vec<String> = items.iter()
            .map(|i| try_extract_column_name(i)?.ok_or_else(|| {
                PyValueError::new_err(
                    "TableHDU[key] sequence mixes column names and \
                     non-string elements; pass all str/bytes (columns) \
                     or all int (rows)"
                )
            }))
            .collect::<PyResult<_>>()?;
        Ok(TableKey::MultiColumns(names))
    } else if !first.is_instance_of::<PyBool>() && first.extract::<i64>().is_ok() {
        // Defer per-element validation to resolve_rows; we only need to
        // route here.
        Ok(TableKey::Rows)
    } else {
        Err(PyValueError::new_err(
            "TableHDU[key] sequence elements must be all int (rows) or \
             all str/bytes (columns)"
        ))
    }
}

/// A deferred handle for one column of a :class:`TableHDU`.
///
/// Returned by ``hdu["name"]`` (a single str/bytes column name).
/// Carries a reference to the parent table and the column name;
/// no I/O happens at construction.  Add ``[rows]`` to trigger
/// the read::
///
///     col = hdu["RA"]          # no I/O, returns a subset handle
///     all_ra  = col[:]         # read every row
///     subset  = col[100:200]   # read 100 rows
///     fancy   = col[[7, 3, 9]] # read three rows in that order
///
/// Writing to ``[rows]`` mutates only that column::
///
///     hdu["FLAG"][bad_rows] = -99
///
/// Equivalent to :meth:`TableHDU.read_column` for reads and to
/// the cell / slice forms of ``TableHDU.__setitem__`` for writes;
/// the subset object exists so the ``hdu["name"][...]`` idiom
/// composes naturally.
#[pyclass]
pub(crate) struct SingleColumnSubset {
    hdu: Py<TableHDU>,
    name: String,
}

#[pymethods]
impl SingleColumnSubset {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        Ok(format!(
            "<TableColumn '{}' of HDU #{}>",
            self.name, super_.index(),
        ))
    }

    // __getitem__ / __setitem__ are pyo3 slot dunders — their
    // docstrings would be overridden by Python's canonical slot
    // text.  Their semantics are covered in the class docstring's
    // "Add ``[rows]``" examples.
    fn __getitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.as_super();
        let meta = pyref.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_one_column(
            py, &meta, data_offset, &super_.file,
            &self.name, Some(rows), /* as_bytes = */ false,
            /* scale = */ true, /* mask_null = */ false,
        )
    }

    // See the __getitem__ comment above re: slot dunders.
    // Per-cell loop through setitem_cell — simple and correct.
    // Cards are re-snapshotted between cells so VLA writes (which
    // mutate PCOUNT) see fresh state.
    fn __setitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        let cards = super_.header_snapshot()?;
        let columns = parse_columns(&cards)?;
        let nrows = parse_keyword(&cards, "NAXIS2")
            .unwrap_or(0).max(0) as usize;
        let row_width = parse_keyword(&cards, "NAXIS1")
            .unwrap_or(0).max(0) as usize;
        let data_offset = super_.offsets.data_offset();
        write_one_column_at_rows(
            py, &super_, &columns, nrows, row_width, &self.name,
            rows, value, data_offset)
    }
}

/// A deferred handle for a column subset of a :class:`TableHDU`.
///
/// Returned by ``hdu[[name1, name2, ...]]`` (an iterable of
/// str/bytes column names).  Carries a reference to the parent
/// table and the column list; no I/O happens at construction.
/// Add ``[rows]`` to trigger the read::
///
///     pos = hdu[["RA", "DEC"]]
///     all_pos = pos[:]              # structured ndarray w/ 2 fields
///     subset  = pos[100:200]
///
/// Writing to ``[rows]`` mutates only those columns; the value
/// must be a structured ndarray with the matching field names
/// (extras tolerated for forward compatibility)::
///
///     hdu[["RA", "DEC"]][bad_rows] = corrected
///
/// Equivalent to :meth:`TableHDU.read` with ``columns=`` for
/// reads, and to the multi-column form of ``TableHDU.__setitem__``
/// for writes.  The subset object exists so the
/// ``hdu[[...]][...]`` idiom composes naturally.
#[pyclass]
pub(crate) struct ColumnSubset {
    hdu: Py<TableHDU>,
    columns: Vec<String>,
}

#[pymethods]
impl ColumnSubset {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        Ok(format!(
            "<TableColumns {:?} of HDU #{}>",
            self.columns, super_.index(),
        ))
    }

    // __getitem__ / __setitem__ are pyo3 slot dunders — see the
    // SingleColumnSubset note for why we don't put docstrings on
    // them.  Semantics are in the class docstring above.
    fn __getitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.as_super();
        let meta = pyref.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_table(
            py, &meta, data_offset, &super_.file,
            Some(rows), Some(self.columns.clone()),
            /* scale = */ true, /* mask_null = */ false,
        )
    }

    fn __setitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        let cards = super_.header_snapshot()?;
        let columns = parse_columns(&cards)?;
        let nrows = parse_keyword(&cards, "NAXIS2")
            .unwrap_or(0).max(0) as usize;
        let row_width = parse_keyword(&cards, "NAXIS1")
            .unwrap_or(0).max(0) as usize;
        let data_offset = super_.offsets.data_offset();
        write_column_subset_at_rows(
            py, &super_, &columns, nrows, row_width, &self.columns,
            rows, value, data_offset)
    }
}
