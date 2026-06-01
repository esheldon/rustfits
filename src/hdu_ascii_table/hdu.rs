// AsciiTableHDU pyclass: TABLE extension HDU (ASCII tables).
//
// Phase 1: whole-table read + accessors (nrows, ncols, colnames,
// dtype, units, __len__, __repr__).  rows= / columns= subsets,
// __getitem__ slicing, iteration, subset objects, write paths all
// land in later phases (see CLAUDE.md "ASCII tables" roadmap).

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::common::{
    check_not_tainted, parse_string_keyword, FileHandle, FileLayout,
    HduOffsets, TaintFlag,
};
use crate::hdu::HDU;

use super::meta::{parse_ascii_table_meta, AsciiTableMeta};
use super::read::{ascii_repr_dtype_str, build_ascii_numpy_dtype, read_ascii_table};

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
/// * ``Fw.d`` / ``Ew.d`` (single-precision float) → ``f4``
/// * ``Dw.d`` (double-precision float) → ``f8``
///
/// Default ``scale=True`` applies ``TSCAL`` / ``TZERO``: the
/// unsigned-int trick on ``I`` columns (with ``TZERO=2**63``)
/// promotes to ``u8``, and other non-trivial scaling produces
/// ``f8``.  Pass ``scale=False`` for raw stored values.
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

    // Return the parsed-once metadata for this HDU.  Mirrors the
    // hot-path pattern from TableHDU::meta.  One Mutex lock + Acquire
    // version load + (on hit) an Arc clone.  On miss, re-parses outside
    // the lock so concurrent racers each parse once but only one wins
    // the cache slot.
    fn meta(&self, super_: &HDU) -> PyResult<Arc<AsciiTableMeta>> {
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

#[pymethods]
impl AsciiTableHDU {
    // Multi-line, fitsio-style repr.  Shows file, extension, type,
    // EXTNAME (if present), row count, and per-column dtype.  Column
    // lines are dynamically aligned to the longest name.
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.as_super();
        // Try the cached meta first; fall back to a fresh cards parse
        // if meta parsing raises (repr must never crash).
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
    ///
    /// Reflects the default-read (``scale=True``) dtype.  Useful
    /// for inspecting the column layout (names, types) without
    /// paying for an actual read.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        build_ascii_numpy_dtype(py, &meta.columns, /* scale = */ true)
    }

    /// Per-column units (``TUNITn``), as a dict.
    ///
    /// Maps each column name (case preserved) to its ``TUNITn``
    /// string, or ``None`` when ``TUNITn`` is unset for that
    /// column.  Dict iteration follows on-disk column order.
    /// Informational only — nothing in the read path consumes
    /// units.
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
    ///
    /// Names are returned with their on-disk case preserved
    /// verbatim.  Returned as a tuple so the value is immutable
    /// from the caller side.
    #[getter]
    fn colnames(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        let names: Vec<&str> =
            meta.columns.iter().map(|c| c.name.as_str()).collect();
        Ok(PyTuple::new(py, &names)?.unbind())
    }

    // pyo3 installs __len__ as the C tp_len slot; no docstring needed.
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.nrows as usize)
    }

    /// Read the whole table into a numpy structured array.
    ///
    /// Parameters
    /// ----------
    /// scale : bool, optional
    ///     If ``True`` (default), apply ``TSCAL`` / ``TZERO``
    ///     scaling.  See the class docstring for the per-letter
    ///     scaling rules.  If ``False``, return raw text-parsed
    ///     values in the default dtype.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Structured array of shape ``(NAXIS2,)`` with one field
    ///     per column.
    ///
    /// Notes
    /// -----
    /// Row / column subsets, MaskedArray support for ``TNULL``,
    /// per-column reads, and ``__getitem__`` slicing land in
    /// Phase 2 (see the rustfits roadmap in CLAUDE.md).
    #[pyo3(signature = (*, scale = true))]
    fn read(slf: PyRef<'_, Self>, py: Python<'_>, scale: bool) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super();
        check_not_tainted(&super_.tainted)?;
        let meta = slf.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        read_ascii_table(py, &meta, data_offset, &super_.file, scale)
    }
}
