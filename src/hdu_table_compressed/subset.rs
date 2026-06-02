// Column-subset pyclasses returned by CompressedTableHDU indexing:
// CompressedSingleColumnSubset + CompressedColumnSubset (read + write).

use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;

use crate::hdu_table::{coerce_to_len1_record, read_rows_maybe_scalar};

use super::hdu::{CompressedTableHDU};
use super::read::{read_compressed_table};
use super::setitem::{
    SetItemCtx, coerce_cell_value_to_len1, coerce_vla_cell_value_to_len1,
    find_compressed_column_index, require_ndarray_with_length,
    resolve_compressed_rows_key, resolve_structured_subset_value,
    setitem_compressed_cols,
};

/// A deferred handle for one column of a :class:`CompressedTableHDU`.
///
/// The compressed-table counterpart of
/// :class:`SingleColumnSubset`.  Returned by ``hdu["name"]``
/// when ``hdu`` is a :class:`CompressedTableHDU`.  Add
/// ``[rows]`` to trigger the read::
///
///     col = compressed_hdu["RA"]
///     subset = col[100:200]            # decodes only overlapping tiles
///
/// Writing via ``[rows] = value`` is supported and routes to
/// the same per-tile decode → modify → re-encode path as
/// ``CompressedTableHDU.__setitem__``.
#[pyclass]
pub(crate) struct CompressedSingleColumnSubset {
    pub(crate) hdu: Py<CompressedTableHDU>,
    pub(crate) name: String,
}

#[pymethods]
impl CompressedSingleColumnSubset {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super().into_super();
        Ok(format!(
            "<CompressedTableColumn '{}' of HDU #{}>",
            self.name, super_.index(),
        ))
    }

    // [rows] returns the column's values for those rows as a plain
    // (non-structured) ndarray — same convention as
    // SingleColumnSubset on the uncompressed side.
    fn __getitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let cache = Arc::clone(&pyref.cache);
        let super_ = pyref.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let data_offset = super_.offsets.data_offset();
        // Closure does the read AND the field extract so the helper's
        // [0] strip works on the plain (non-structured) column view —
        // mirrors the uncompressed SingleColumnSubset return shape.
        read_rows_maybe_scalar(py, rows, |rk| {
            let arr = read_compressed_table(
                py, &cards, data_offset, &super_.file,
                Some(rk), Some(vec![self.name.clone()]),
                /* scale = */ true, &cache,
            )?;
            Ok(arr.bind(py).get_item(self.name.as_str())?.unbind())
        })
    }

    /// Read this column.
    ///
    /// Returns a plain (non-structured) ndarray of the column's
    /// values — same shape as ``self[rows]`` with the rows key.
    /// All kwargs map to the matching
    /// :meth:`CompressedTableHDU.read` arguments.
    ///
    /// Parameters
    /// ----------
    /// rows : slice, int, or iterable of int, optional
    ///     Row subset to read.  None (default) reads every row.
    /// scale : bool, default True
    ///     Apply ``TSCALn`` / ``TZEROn`` scaling on the way out.
    /// mask_null : bool, default False
    ///     Currently raises ``NotImplementedError`` on compressed
    ///     tables (parity with :meth:`CompressedTableHDU.read`).
    ///
    /// Returns
    /// -------
    /// data : ndarray
    ///     The column's values.
    #[pyo3(signature = (*, rows=None, scale=true, mask_null=false))]
    fn read(
        &self,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<Py<PyAny>> {
        if mask_null {
            return Err(PyNotImplementedError::new_err(
                "CompressedTableHDU[name].read(mask_null=True): \
                 TNULL masking on compressed-table reads is not \
                 yet implemented"));
        }
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let cache = Arc::clone(&pyref.cache);
        let super_ = pyref.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let data_offset = super_.offsets.data_offset();
        let arr = read_compressed_table(
            py, &cards, data_offset, &super_.file,
            rows, Some(vec![self.name.clone()]),
            scale, &cache,
        )?;
        Ok(arr.bind(py).get_item(self.name.as_str())?.unbind())
    }

    /// Write this column.
    ///
    /// With ``rows=None`` (default) writes all ``NAXIS2`` rows
    /// — equivalent to ``hdu[name] = data``.  With
    /// ``rows=<spec>`` writes only the named rows — equivalent
    /// to ``self[rows] = data`` (the ``__setitem__`` form).
    /// Re-encodes every affected tile and appends the new
    /// compressed bytes to the heap; old tile bytes become
    /// orphans (reclaim with
    /// :meth:`CompressedTableHDU.repack`).
    ///
    /// Parameters
    /// ----------
    /// data : ndarray
    ///     For ``rows=None``, a length-``NAXIS2`` ndarray of
    ///     the column's expected dtype and per-cell shape (for
    ///     VLA columns, an Object-dtype ndarray).  For
    ///     ``rows=<spec>``, the shape must match what
    ///     ``self[rows] = data`` would accept.
    /// rows : slice, int, or iterable of int, optional
    ///     Restrict the write to these rows.  None (default)
    ///     writes every row.
    #[pyo3(signature = (data, *, rows=None))]
    fn write(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        rows: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        match rows {
            None => self.hdu.bind(py).set_item(&self.name, data),
            Some(rows_key) => self.__setitem__(py, rows_key, data),
        }
    }

    // [rows] = value writes to this one column at the selected rows.
    // For a bare-int `rows`, `value` is a scalar / 0-d / per-cell
    // ndarray (broadcast over the column's per-cell shape).  For a
    // slice or iterable `rows`, `value` is an ndarray of shape
    // (len(rows),) + per_cell_shape.  Routes through the shared
    // per-tile rewrite primitive with `selected = [col_idx]`.
    fn __setitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let cfgs = pyref.compress_configs.lock()
            .map_err(|_| PyIOError::new_err(
                "compress_configs lock poisoned"))?
            .clone();
        let super_ = pyref.as_super().as_super();
        let meta = pyref.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        let ctx = SetItemCtx {
            super_,
            cards: &meta.cards,
            columns: &meta.columns,
            algorithms: &meta.algorithms,
            per_col_configs: cfgs.as_deref(),
            nrows: meta.nrows,
            ztilelen: meta.ztilelen,
            n_tiles: meta.n_tiles,
            descriptor_row_width: meta.descriptor_row_width,
            data_offset,
            current_pcount: meta.current_pcount,
            cache: &pyref.cache,
        };
        let col_idx = find_compressed_column_index(&meta.columns, &self.name)?;
        let col = &meta.columns[col_idx];
        let (disk_rows, was_single) =
            resolve_compressed_rows_key(rows, meta.nrows)?;
        if disk_rows.is_empty() {
            return Ok(());
        }
        let is_vla = col.var_kind.is_some();
        let per_column = if was_single {
            let promoted = if is_vla {
                coerce_vla_cell_value_to_len1(py, value)?
            } else {
                coerce_cell_value_to_len1(py, col, value)?
            };
            vec![promoted]
        } else {
            let label = format!(
                "CompressedTableHDU['{}'][rows]", self.name);
            // For VLA columns, the value is an Object-dtype ndarray
            // of length len(rows); the primitive validates the
            // Object kind itself.  For fixed columns it's an
            // ndarray of `(len(rows),) + per_cell_shape`.  Either
            // way we just check the outer length here.
            require_ndarray_with_length(
                py, value, disk_rows.len(), &label)?;
            vec![value.clone()]
        };
        setitem_compressed_cols(
            py, &ctx, &per_column, &[col_idx], &disk_rows,
        )
    }
}

/// A deferred handle for a column subset of a
/// :class:`CompressedTableHDU`.
///
/// The compressed-table counterpart of :class:`ColumnSubset`.
/// Returned by ``hdu[[name1, name2, ...]]`` when ``hdu`` is a
/// :class:`CompressedTableHDU`.  Add ``[rows]`` to read or
/// assign to write::
///
///     pos = compressed_hdu[["RA", "DEC"]]
///     subset = pos[100:200]
///     compressed_hdu[["RA", "DEC"]][bad_rows] = corrected
#[pyclass]
pub(crate) struct CompressedColumnSubset {
    pub(crate) hdu: Py<CompressedTableHDU>,
    pub(crate) columns: Vec<String>,
}

#[pymethods]
impl CompressedColumnSubset {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let super_ = pyref.into_super().into_super();
        Ok(format!(
            "<CompressedTableColumns {:?} of HDU #{}>",
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
        let cache = Arc::clone(&pyref.cache);
        let super_ = pyref.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let data_offset = super_.offsets.data_offset();
        read_rows_maybe_scalar(py, rows, |rk| read_compressed_table(
            py, &cards, data_offset, &super_.file,
            Some(rk), Some(self.columns.clone()),
            /* scale = */ true, &cache,
        ))
    }

    /// Read these columns.
    ///
    /// Returns a structured ndarray with the subset's named
    /// fields — same shape as ``self[rows]`` with the rows key.
    /// All kwargs map to the matching
    /// :meth:`CompressedTableHDU.read` arguments.
    ///
    /// Parameters
    /// ----------
    /// rows : slice, int, or iterable of int, optional
    ///     Row subset to read.  None (default) reads every row.
    /// scale : bool, default True
    ///     Apply ``TSCALn`` / ``TZEROn`` scaling on the way out.
    /// mask_null : bool, default False
    ///     Currently raises ``NotImplementedError`` on compressed
    ///     tables (parity with :meth:`CompressedTableHDU.read`).
    ///
    /// Returns
    /// -------
    /// data : structured ndarray
    ///     One row per selected source row; one field per column
    ///     in the subset.
    #[pyo3(signature = (*, rows=None, scale=true, mask_null=false))]
    fn read(
        &self,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<Py<PyAny>> {
        if mask_null {
            return Err(PyNotImplementedError::new_err(
                "CompressedTableHDU[[names]].read(mask_null=True): \
                 TNULL masking on compressed-table reads is not \
                 yet implemented"));
        }
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let cache = Arc::clone(&pyref.cache);
        let super_ = pyref.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let data_offset = super_.offsets.data_offset();
        read_compressed_table(
            py, &cards, data_offset, &super_.file,
            rows, Some(self.columns.clone()),
            scale, &cache,
        )
    }

    /// Write this subset.
    ///
    /// With ``rows=None`` (default) writes all ``NAXIS2`` rows
    /// — equivalent to ``hdu[[names...]] = data``.  With
    /// ``rows=<spec>`` writes only the named rows — equivalent
    /// to ``self[rows] = data`` (the ``__setitem__`` form).
    /// Each named column is re-encoded per affected tile; the
    /// other columns' stored bytes are untouched.  Old tile
    /// bytes for modified columns become orphans (reclaim with
    /// :meth:`CompressedTableHDU.repack`).
    ///
    /// Parameters
    /// ----------
    /// data : structured ndarray
    ///     For ``rows=None``, a length-``NAXIS2`` structured
    ///     ndarray with the subset's named fields.  For
    ///     ``rows=<spec>``, the shape must match what
    ///     ``self[rows] = data`` would accept.  Extra fields
    ///     are tolerated (matched by name); missing fields
    ///     raise.
    /// rows : slice, int, or iterable of int, optional
    ///     Restrict the write to these rows.  None (default)
    ///     writes every row.
    #[pyo3(signature = (data, *, rows=None))]
    fn write(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        rows: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        match rows {
            None => {
                let key = PyList::new(py, &self.columns)?;
                self.hdu.bind(py).set_item(&key, data)
            }
            Some(rows_key) => self.__setitem__(py, rows_key, data),
        }
    }

    // [rows] = value writes a column-subset at the selected rows.
    // For a bare-int `rows`, `value` is a structured record /
    // shape-(1,) ndarray with all the subset's field names (the
    // coerce-to-length-1 helper accepts numpy.void scalars too).
    // For slice / iterable `rows`, `value` is a structured ndarray
    // of length == len(rows).  Each named column is dispatched
    // through the shared per-tile rewrite primitive with
    // `selected = [c1_idx, c2_idx, ...]`.
    fn __setitem__(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if self.columns.is_empty() {
            return Err(PyValueError::new_err(
                "CompressedTableHDU[[names]][rows] = value: empty \
                 column list"));
        }
        let bound = self.hdu.bind(py);
        let pyref = bound.borrow();
        let cfgs = pyref.compress_configs.lock()
            .map_err(|_| PyIOError::new_err(
                "compress_configs lock poisoned"))?
            .clone();
        let super_ = pyref.as_super().as_super();
        let meta = pyref.meta(super_)?;
        let data_offset = super_.offsets.data_offset();
        let ctx = SetItemCtx {
            super_,
            cards: &meta.cards,
            columns: &meta.columns,
            algorithms: &meta.algorithms,
            per_col_configs: cfgs.as_deref(),
            nrows: meta.nrows,
            ztilelen: meta.ztilelen,
            n_tiles: meta.n_tiles,
            descriptor_row_width: meta.descriptor_row_width,
            data_offset,
            current_pcount: meta.current_pcount,
            cache: &pyref.cache,
        };
        let (disk_rows, was_single) =
            resolve_compressed_rows_key(rows, meta.nrows)?;
        if disk_rows.is_empty() {
            return Ok(());
        }
        // For single-int row keys, coerce value to a shape-(1,)
        // structured ndarray so resolve_structured_subset_value can
        // pull per-column views via arr[name].  For slice/fancy,
        // value must already be a structured ndarray of length
        // len(disk_rows).
        let value_arr = if was_single {
            coerce_to_len1_record(py, value)?
        } else {
            require_ndarray_with_length(
                py, value, disk_rows.len(),
                "CompressedTableHDU[[names]][rows]",
            )?;
            value.clone()
        };
        let (selected, per_column) = resolve_structured_subset_value(
            py, &value_arr, &meta.columns, &self.columns,
        )?;
        setitem_compressed_cols(
            py, &ctx, &per_column, &selected, &disk_rows,
        )
    }
}

