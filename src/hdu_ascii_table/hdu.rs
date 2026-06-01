// AsciiTableHDU pyclass: TABLE extension HDU (ASCII tables).
//
// Phase 2: read(rows=, columns=, scale=, mask_null=), __getitem__
// (slice / int / list-of-int / str / list-of-str), __iter__ +
// iter(chunksize=, columns=, scale=), and the read-only subset
// pyclasses AsciiSingleColumnSubset / AsciiColumnSubset.  Writing
// (subset and HDU __setitem__, append, insert / delete column) lands
// in Phases 3-5.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PySlice, PyTuple};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::common::{
    check_not_tainted, parse_string_keyword, FileHandle, FileLayout,
    HduOffsets, TaintFlag,
};
use crate::hdu::HDU;
use crate::hdu_table::try_extract_column_name;
use crate::hdu_table::TableIter;

use super::meta::{parse_ascii_table_meta, AsciiTableMeta};
use super::read::{
    ascii_repr_dtype_str, build_ascii_numpy_dtype, read_ascii_table,
    read_one_column,
};
use super::write_fixed::{
    append_ascii_table, determine_input_nrows, write_ascii_table_full,
};

// Version-stamped parsed-metadata cache.  Same shape as TableHDU's
// `MetaCache`.  See `meta()` for the hot-path accessor.
type AsciiMetaCache = Arc<Mutex<Option<(u64, Arc<AsciiTableMeta>)>>>;

/// An ASCII-table extension HDU (``XTENSION='TABLE'``).
///
/// ASCII tables store row data as fixed-width text, distinct
/// from binary tables (:class:`TableHDU`).  Returned by indexing
/// a :class:`FITS` object at a position containing a TABLE HDU.
///
/// Reads return a numpy structured array whose dtype reflects
/// the per-column ``TFORMn`` mapping:
///
/// * ``Aw`` (text) → ``U<w>`` (numpy unicode string)
/// * ``Iw`` (integer) → ``i8``
/// * ``Fw.d`` / ``Ew.d`` (single-precision float) → ``f4`` when
///   ``d <= 7``; ``f8`` when ``d > 7`` (cfitsio's E26.17 default
///   for f8 input is correctly read as f8)
/// * ``Dw.d`` (double-precision float) → ``f8``
///
/// Default ``scale=True`` applies ``TSCAL`` / ``TZERO``: the
/// unsigned-int trick on ``I`` columns (with ``TZERO=2**63``)
/// promotes to ``u8``, and other non-trivial scaling produces
/// ``f8``.  Pass ``scale=False`` for raw stored values.  Default
/// ``mask_null=False`` returns a plain ndarray; pass ``True`` for
/// a :class:`numpy.ma.MaskedArray` with cells matching ``TNULLn``
/// masked.
///
/// Indexing supports the same forms as :class:`TableHDU`:
///
///     arr = hdu[5]               # one row as a np.void record
///     arr = hdu[10:20]           # 10 rows as a structured ndarray
///     arr = hdu[[1, 3, 5]]       # fancy row select
///     col = hdu["RA"]            # SingleColumnSubset (deferred read)
///     sub = hdu[["RA", "DEC"]]   # ColumnSubset (deferred read)
///
/// ASCII tables are rare in modern FITS files; most data
/// pipelines use binary tables instead.  Both round-trip
/// bit-exactly with astropy and fitsio.
#[pyclass(extends = HDU)]
pub(crate) struct AsciiTableHDU {
    meta_cache: AsciiMetaCache,
}

impl AsciiTableHDU {
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
            AsciiTableHDU {
                meta_cache: Arc::new(Mutex::new(None)),
            },
            HDU::new(header, index, filename, offsets, layout, file, tainted),
        )
    }

    // Return the parsed-once metadata for this HDU.  Hot-path accessor:
    // one Mutex lock + Acquire version load + (on hit) an Arc clone.
    // On miss, re-parses outside the lock.  Same shape as
    // TableHDU::meta and CompressedImageHDU::meta.
    pub(crate) fn meta(
        &self, super_: &HDU,
    ) -> PyResult<Arc<AsciiTableMeta>> {
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
        let cards = super_.header_snapshot()?;
        let meta = Arc::new(parse_ascii_table_meta(&cards)?);
        let mut cache = self.meta_cache.lock()
            .map_err(|_| PyIOError::new_err("meta cache poisoned"))?;
        *cache = Some((cur_version, Arc::clone(&meta)));
        Ok(meta)
    }
}

// What kind of selection the user passed to AsciiTableHDU.__getitem__.
// Mirrors hdu_table::TableKey exactly (subset of forms — VLA-cell
// dispatch doesn't exist for ASCII tables since they have no heap).
enum AsciiTableKey {
    Rows,
    SingleRow(i64),
    SingleColumn(String),
    MultiColumns(Vec<String>),
}

fn classify_ascii_table_key(
    key: &Bound<'_, PyAny>,
) -> PyResult<AsciiTableKey> {
    if key.is_instance_of::<PySlice>() {
        return Ok(AsciiTableKey::Rows);
    }
    if let Some(name) = try_extract_column_name(key)? {
        return Ok(AsciiTableKey::SingleColumn(name));
    }
    if !key.is_instance_of::<PyBool>() {
        if let Ok(idx) = key.extract::<i64>() {
            return Ok(AsciiTableKey::SingleRow(idx));
        }
    }
    let iter = key.try_iter().map_err(|_| PyValueError::new_err(
        "AsciiTableHDU[key] requires a slice, an int (row index), a \
         str/bytes column name, an iterable of ints (rows), or an \
         iterable of str/bytes (columns)"
    ))?;
    let items: Vec<Bound<'_, PyAny>> = iter.collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Err(PyValueError::new_err(
            "AsciiTableHDU[key] received an empty sequence (ambiguous: rows \
             or columns?); pass a non-empty selection or use read() with \
             explicit rows=/columns="
        ));
    }
    let first = &items[0];
    if try_extract_column_name(first)?.is_some() {
        let names: Vec<String> = items.iter()
            .map(|i| try_extract_column_name(i)?.ok_or_else(|| {
                PyValueError::new_err(
                    "AsciiTableHDU[key] sequence mixes column names and \
                     non-string elements; pass all str/bytes (columns) \
                     or all int (rows)"
                )
            }))
            .collect::<PyResult<_>>()?;
        Ok(AsciiTableKey::MultiColumns(names))
    } else if !first.is_instance_of::<PyBool>() && first.extract::<i64>().is_ok() {
        Ok(AsciiTableKey::Rows)
    } else {
        Err(PyValueError::new_err(
            "AsciiTableHDU[key] sequence elements must be all int (rows) or \
             all str/bytes (columns)"
        ))
    }
}

#[pymethods]
impl AsciiTableHDU {
    // Multi-line, fitsio-style repr.  Shows file, extension, type,
    // EXTNAME (if present), row count, and per-column dtype.
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.as_super();
        let cached = slf.meta(super_).ok();
        let cards = super_.header_snapshot()?;
        let nrows = cached.as_ref()
            .map(|m| m.nrows as i64)
            .unwrap_or_else(|| crate::common::parse_keyword(&cards, "NAXIS2")
                .unwrap_or(0).max(0));
        let extname = parse_string_keyword(&cards, "EXTNAME");

        let mut out = String::new();
        out.push_str(&format!("  file: {}\n", super_.filename));
        out.push_str(&format!("  extension: {}\n", super_.index));
        out.push_str("  type: ASCII_TBL\n");
        if let Some(name) = extname {
            out.push_str(&format!("  extname: {}\n", name));
        }
        out.push_str(&format!("  rows: {}\n", nrows));

        if let Some(meta) = &cached {
            if !meta.columns.is_empty() {
                out.push_str("  column info:\n");
                let max_name_len = meta.columns.iter()
                    .map(|c| c.name.len()).max().unwrap_or(0);
                let name_w = max_name_len + 4;
                for col in &meta.columns {
                    let dtype_str = ascii_repr_dtype_str(col);
                    out.push_str(&format!(
                        "    {:<w$}{}", col.name, dtype_str, w = name_w,
                    ));
                    if let Some(unit) = &col.tunit {
                        out.push_str(&format!("  ({})", unit));
                    }
                    out.push('\n');
                }
            }
        }
        Ok(out)
    }

    /// The numpy structured dtype the table reads into.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        build_ascii_numpy_dtype(py, &meta.columns, /* scale = */ true)
    }

    /// Per-column units (``TUNITn``), as a dict.
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

    /// Number of rows in the table (``NAXIS2``).
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
    #[getter]
    fn colnames(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        let names: Vec<&str> =
            meta.columns.iter().map(|c| c.name.as_str()).collect();
        Ok(PyTuple::new(py, &names)?.unbind())
    }

    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.nrows as usize)
    }

    /// Read rows from the table into a numpy structured array.
    ///
    /// Parameters
    /// ----------
    /// rows : slice, int, or iterable of int, optional
    ///     Rows to read.  ``None`` (default) reads every row in
    ///     file order.  A slice or iterable selects a subset;
    ///     negative indices supported.  Iterables are deduped
    ///     with first-occurrence-wins ordering.
    /// columns : list of str, optional
    ///     Column names to read (case-insensitive against the
    ///     table's column names).  ``None`` (default) reads every
    ///     column in file order.  A list selects + reorders.
    /// scale : bool, optional
    ///     If ``True`` (default), apply ``TSCAL`` / ``TZERO``
    ///     scaling.  See the class docstring for per-letter rules.
    /// mask_null : bool, optional
    ///     If ``True``, return a :class:`numpy.ma.MaskedArray`
    ///     with cells masked where the trimmed field text equals
    ///     the trimmed ``TNULLn`` string.  Compare is on stored
    ///     text (pre-scaling).  Columns without ``TNULLn`` stay
    ///     unmasked.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray or numpy.ma.MaskedArray
    ///     Structured array of shape ``(n_selected,)`` with one
    ///     field per selected column.
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
        check_not_tainted(&super_.tainted)?;
        let meta = slf.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_ascii_table(
            py, &meta, data_offset, &super_.file,
            rows, columns, scale, mask_null,
        )
    }

    /// Read one column into a plain (non-structured) ndarray.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Column name (case-insensitive).
    /// rows : slice, int, or iterable of int, optional
    ///     Row subset to read.  ``None`` (default) reads every row.
    /// as_bytes : bool, optional
    ///     For ``A`` columns only.  If ``True``, return raw on-disk
    ///     bytes in an ``S<width>`` ndarray (no trim, no ASCII
    ///     validation) — escape hatch for non-ASCII content the
    ///     default ``U`` decode would reject.
    /// scale : bool, optional
    ///     Apply ``TSCAL`` / ``TZERO`` scaling.  Default ``True``.
    /// mask_null : bool, optional
    ///     Return a :class:`numpy.ma.MaskedArray` with cells
    ///     matching ``TNULLn`` masked.  Default ``False``.
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
        check_not_tainted(&super_.tainted)?;
        let meta = slf.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_one_column(
            py, &meta, data_offset, &super_.file, name, rows, as_bytes,
            scale, mask_null,
        )
    }

    /// Bulk-write data into the table's data section.
    ///
    /// Overwrites all ``NAXIS2`` rows in the table; for appending
    /// new rows instead use :meth:`append`.  Accepts three input
    /// forms (matching :meth:`TableHDU.write`):
    ///
    /// * **Structured ndarray** — field names must match the HDU's
    ///   columns (case-insensitive); extras / missing rejected.
    ///   ``len(data)`` must equal ``NAXIS2``.
    /// * **Dict** ``{name: ndarray}`` — one entry per HDU column;
    ///   extras / missing rejected.
    /// * **List or tuple of ndarrays** with ``names=[...]`` —
    ///   parallel sequences; same per-column model as dict.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray, dict, or list/tuple of ndarrays
    /// names : list of str, optional
    ///     Required only when ``data`` is a list/tuple.
    ///
    /// Notes
    /// -----
    /// Validate-then-mutate: dtype / length errors are raised
    /// before any file bytes are touched.  Mid-write I/O failures
    /// taint the file (close + reopen to recover).
    #[pyo3(signature = (data, *, names=None))]
    fn write(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let super_ = slf.as_super();
        check_not_tainted(&super_.tainted)?;
        let meta = slf.meta(super_)?;
        let nrows = meta.nrows as usize;
        let row_width = meta.row_width as usize;
        let data_offset = super_.offsets.data_offset();
        write_ascii_table_full(
            py, super_, &meta.columns, data, names,
            nrows, row_width, data_offset,
        )
    }

    /// Append rows to the end of the table.
    ///
    /// Grows ``NAXIS2`` and the data section.  For non-last HDUs,
    /// the file tail shifts forward and later-HDU offsets are
    /// bumped in lockstep; previously-issued handles remain valid.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray, dict, or list/tuple of ndarrays
    ///     Same three input forms as :meth:`write`.  Length
    ///     defines the number of new rows.
    /// names : list of str, optional
    ///     Required for the list/tuple form; ignored otherwise.
    ///
    /// Notes
    /// -----
    /// Validate-then-mutate: dtype / shape errors are raised
    /// before any file bytes are touched.  Mid-write I/O failures
    /// taint the file.
    #[pyo3(signature = (data, *, names=None))]
    fn append(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let super_ = slf.as_super();
        check_not_tainted(&super_.tainted)?;
        let meta = slf.meta(super_)?;
        let current_nrows = meta.nrows as usize;
        let row_width = meta.row_width as usize;
        let append_nrows = determine_input_nrows(data, names)?;
        if append_nrows == 0 {
            return Ok(());
        }
        let new_nrows = current_nrows + append_nrows;
        let data_offset = super_.offsets.data_offset();
        append_ascii_table(
            py, super_, &meta.columns, data, names,
            current_nrows, append_nrows, new_nrows, row_width, data_offset,
        )
    }

    /// Alias for :meth:`append`.
    ///
    /// Mirrors :meth:`TableHDU.extend` so generic code that
    /// iterates HDUs and calls ``.extend(...)`` keeps working.
    #[pyo3(signature = (data, *, names=None))]
    fn extend(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Self::append(slf, py, data, names)
    }

    // hdu[key] dispatches based on what `key` looks like:
    //   slice or iterable-of-int        -> reads rows now -> ndarray
    //   bare int                        -> reads + unwraps to np.void
    //   single str/bytes column name    -> AsciiSingleColumnSubset
    //   iterable of str/bytes           -> AsciiColumnSubset
    // Specifying a column alone never invokes I/O — only rows do.
    fn __getitem__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let kind = classify_ascii_table_key(key)?;
        match kind {
            AsciiTableKey::Rows => {
                let pyref = slf.borrow();
                let super_ = pyref.as_super();
                check_not_tainted(&super_.tainted)?;
                let meta = pyref.meta(super_)?;
                let data_offset = super_.offsets.data_offset();
                read_ascii_table(
                    py, &meta, data_offset, &super_.file,
                    Some(key), None,
                    /* scale = */ true, /* mask_null = */ false,
                )
            }
            AsciiTableKey::SingleRow(idx) => {
                let pyref = slf.borrow();
                let super_ = pyref.as_super();
                check_not_tainted(&super_.tainted)?;
                let meta = pyref.meta(super_)?;
                let data_offset = super_.offsets.data_offset();
                let one = PyList::new(py, [idx])?;
                let arr_py = read_ascii_table(
                    py, &meta, data_offset, &super_.file,
                    Some(one.as_any()), None,
                    /* scale = */ true, /* mask_null = */ false,
                )?;
                let arr_bound = arr_py.bind(py);
                Ok(arr_bound.get_item(0)?.unbind())
            }
            AsciiTableKey::SingleColumn(name) => {
                let hdu_py: Py<AsciiTableHDU> = slf.clone().unbind();
                Ok(Py::new(
                    py,
                    AsciiSingleColumnSubset { hdu: hdu_py, name },
                )?.into())
            }
            AsciiTableKey::MultiColumns(names) => {
                let hdu_py: Py<AsciiTableHDU> = slf.clone().unbind();
                Ok(Py::new(
                    py,
                    AsciiColumnSubset { hdu: hdu_py, columns: names },
                )?.into())
            }
        }
    }

    // `for row in hdu:` — yields one row per iteration as a numpy
    // scalar record.  Rows are read in internally-buffered batches;
    // iterating a large table stays memory-bounded.  pyo3 installs
    // this as the `tp_iter` slot (no per-method docstring).
    fn __iter__(slf: Bound<'_, Self>) -> PyResult<TableIter> {
        crate::hdu_table::make_table_iter(slf.into_any(), None, None, true)
    }

    /// Iterate over table rows or row-chunks.
    ///
    /// ``hdu.iter()`` is equivalent to ``for row in hdu`` — one row
    /// per iteration as a numpy scalar record.  Passing ``chunksize``
    /// switches to yielding structured arrays instead.
    ///
    /// Parameters
    /// ----------
    /// chunksize : int, optional
    ///     ``None`` (default) yields one row per iteration as a numpy
    ///     scalar record.  A positive integer yields a structured
    ///     ndarray of up to ``chunksize`` rows per iteration.
    /// columns : list of str, optional
    ///     Restrict iteration to these columns (case-insensitive),
    ///     forwarded to :meth:`read`.
    /// scale : bool, default True
    ///     Apply scaling, forwarded to :meth:`read`.
    #[pyo3(signature = (*, chunksize=None, columns=None, scale=true))]
    fn iter(
        slf: Bound<'_, Self>,
        chunksize: Option<usize>,
        columns: Option<Py<PyAny>>,
        scale: bool,
    ) -> PyResult<TableIter> {
        crate::hdu_table::make_table_iter(
            slf.into_any(), chunksize, columns, scale,
        )
    }
}

/// A deferred handle for one column of a :class:`AsciiTableHDU`.
///
/// Returned by ``hdu["name"]``.  Carries a reference to the
/// parent table and the column name; no I/O happens at
/// construction.  Add ``[rows]`` to trigger the read::
///
///     col = hdu["RA"]
///     all_ra  = col[:]
///     subset  = col[100:200]
///     fancy   = col[[7, 3, 9]]
///
/// Writing is not yet supported on ASCII subsets (Phase 4 of
/// the rustfits ASCII-tables roadmap).
#[pyclass]
pub(crate) struct AsciiSingleColumnSubset {
    hdu: Py<AsciiTableHDU>,
    name: String,
}

#[pymethods]
impl AsciiSingleColumnSubset {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        Ok(format!(
            "<AsciiTableColumn '{}' of HDU #{}>",
            self.name, super_.index(),
        ))
    }

    fn __getitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.as_super();
        check_not_tainted(&super_.tainted)?;
        let meta = pyref.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_one_column(
            py, &meta, data_offset, &super_.file,
            &self.name, Some(rows), /* as_bytes = */ false,
            /* scale = */ true, /* mask_null = */ false,
        )
    }

    /// Read this column.
    #[pyo3(signature = (*, rows=None, as_bytes=false, scale=true, mask_null=false))]
    fn read(
        &self,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        as_bytes: bool,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<Py<PyAny>> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.as_super();
        check_not_tainted(&super_.tainted)?;
        let meta = pyref.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_one_column(
            py, &meta, data_offset, &super_.file,
            &self.name, rows, as_bytes, scale, mask_null,
        )
    }
}

/// A deferred handle for a column subset of a :class:`AsciiTableHDU`.
///
/// Returned by ``hdu[[name1, name2, ...]]``.  Carries a reference
/// to the parent table and the column list; no I/O happens at
/// construction.  Add ``[rows]`` to trigger the read::
///
///     pos = hdu[["RA", "DEC"]]
///     all_pos = pos[:]
///     subset  = pos[100:200]
///
/// Writing is not yet supported on ASCII subsets (Phase 4 of
/// the rustfits ASCII-tables roadmap).
#[pyclass]
pub(crate) struct AsciiColumnSubset {
    hdu: Py<AsciiTableHDU>,
    columns: Vec<String>,
}

#[pymethods]
impl AsciiColumnSubset {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super();
        Ok(format!(
            "<AsciiTableColumns {:?} of HDU #{}>",
            self.columns, super_.index(),
        ))
    }

    fn __getitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.as_super();
        check_not_tainted(&super_.tainted)?;
        let meta = pyref.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_ascii_table(
            py, &meta, data_offset, &super_.file,
            Some(rows), Some(self.columns.clone()),
            /* scale = */ true, /* mask_null = */ false,
        )
    }

    /// Read these columns.
    #[pyo3(signature = (*, rows=None, scale=true, mask_null=false))]
    fn read(
        &self,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<Py<PyAny>> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.as_super();
        check_not_tainted(&super_.tainted)?;
        let meta = pyref.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_ascii_table(
            py, &meta, data_offset, &super_.file,
            rows, Some(self.columns.clone()), scale, mask_null,
        )
    }
}
