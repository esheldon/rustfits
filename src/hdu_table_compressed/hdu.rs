// CompressedTableHDU pyclass: struct + impls + #[pymethods] dispatch,
// ZTABLE detection (header_has_ztable), the CacheKey/ColumnTileCache
// types, original-schema synthesis, and repr helpers.

use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};
use std::sync::{Arc, Mutex};

use crate::cache::BytesBoundLruCache;
use crate::common::{
    parse_keyword, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, TaintFlag,
};
use crate::zimage::compression_config::CompressionConfigKind;
use crate::hdu::HDU;
use crate::hdu_table::{
    build_numpy_dtype, classify_setitem_key,
    classify_table_key, coerce_to_len1_record, field_dtype_and_shape, parse_columns, Column, SetItemKey, TableHDU, TableKey,
};
use crate::zimage::{parse_algorithm, CompressionAlgorithm};

use crate::zimage::tile_io::DEFAULT_TILE_CACHE_BYTES;
use super::append::{append_compressed_table_data};
use super::checksum::{
    compressed_table_add_checksum, compressed_table_add_datasum,
    compressed_table_verify_checksum, compressed_table_verify_datasum,
};
use super::meta::{CompressedTableMeta, parse_compressed_table_meta};
use super::read::{read_compressed_table};
use super::repack::{repack_compressed_table_heap};
use super::setitem::{
    SetItemCtx, find_compressed_column_index, normalize_disk_row,
    require_ndarray, require_ndarray_with_length,
    resolve_structured_subset_value, setitem_compressed_cols,
};
use super::subset::{CompressedColumnSubset, CompressedSingleColumnSubset};
use super::write::{write_compressed_table_data};

// Per-(tile_idx, col_idx) decompressed-bytes cache key.  Packed
// into a tuple so the shared `BytesBoundLruCache` can hash it
// directly.  Finer granularity than ZIMAGE's per-tile key is what
// makes reading `hdu["col"][i:j]` reusable across nearby rows /
// sibling columns.
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub(crate) struct CacheKey(pub(crate) u32, pub(crate) u32);

// Per-(tile, col) decompressed-bytes cache.  See
// `crate::cache::BytesBoundLruCache` for the eviction policy and
// concurrency model.
pub(crate) type ColumnTileCache = BytesBoundLruCache<CacheKey>;

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

// True iff the header contains `ZTABLE = T`.  Mirrors `header_has_zimage`:
// looks for the keyword, parses the logical value tolerantly (any 'T'
// after the '=' is treated as true).
pub(crate) fn header_has_ztable(header: &[String]) -> bool {
    for card in header {
        if card.len() < 9 {
            continue;
        }
        if card[..8].trim() != "ZTABLE" {
            continue;
        }
        if let Some(eq) = card.find('=') {
            let trimmed = card[eq + 1..].trim_start();
            if trimmed.starts_with('T') {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// pyclass
// ---------------------------------------------------------------------------

/// A tile-compressed binary-table HDU (``ZTABLE=T`` on disk).
///
/// The user-facing surface mirrors :class:`TableHDU` exactly:
/// every accessor (:attr:`nrows`, :attr:`ncols`, :attr:`dtype`,
/// :attr:`colnames`, :attr:`units`) reports the **uncompressed**
/// table schema, and every I/O method (:meth:`read`,
/// :meth:`write`, :meth:`append`, ``__setitem__``,
/// ``__getitem__``) operates on the uncompressed rows.  The
/// per-(tile, column) compressed storage is an implementation
/// detail.
///
/// Subclasses :class:`TableHDU` (so ``isinstance(hdu, TableHDU)``
/// holds), but overrides every I/O method to handle per-tile
/// (de)compression.  Returned by indexing a :class:`FITS` object
/// at a position containing a ZTABLE HDU.
///
/// Compression-specific surface beyond the inherited
/// :class:`TableHDU` API:
///
/// * :attr:`compression` — per-column algorithm dict
///   (``{col_name: 'GZIP_1'}`` etc.).
/// * :attr:`n_tiles` — number of tile chunks on disk.
/// * :attr:`ztile_rows` — rows per tile (the ``ZTILELEN``
///   parameter the file was created with).
/// * :attr:`tile_cache_size`, :meth:`set_tile_cache_size`,
///   :attr:`tile_cache_used`, :meth:`clear_tile_cache` — LRU
///   cache controls for decoded ``(tile, column)`` slabs.
/// * :meth:`repack` — drop orphans accumulated by
///   :meth:`append` (merge-into-partial-last-tile) and
///   ``__setitem__`` mutations.
///
/// Examples
/// --------
/// Read just like an uncompressed table::
///
///     arr = hdu.read()
///     ra_dec = hdu.read(columns=["RA", "DEC"])
///     chunk = hdu[1000:2000]
///
/// Inspect compression::
///
///     print(hdu.compression)         # {'RA': 'GZIP_2', 'FLAG': 'RICE_1', ...}
///     print(hdu.n_tiles, hdu.ztile_rows)
///
/// Notes
/// -----
/// Use :meth:`FITS.create_table_hdu` with ``compress=`` to
/// create a tile-compressed table.  Direct construction from
/// Python is not supported.
///
/// :meth:`insert_column` and :meth:`delete_column` are NOT
/// supported on compressed tables; rebuild the table through
/// a fresh :meth:`FITS.create_table_hdu` + :meth:`write` to
/// change the schema.
// Version-stamped parsed-metadata cache (see `meta()`); the u64 is the
// `cards_version` at parse time.
type MetaCache = Arc<Mutex<Option<(u64, Arc<CompressedTableMeta>)>>>;

#[pyclass(extends = TableHDU)]
pub(crate) struct CompressedTableHDU {
    pub(crate) cache: Arc<ColumnTileCache>,
    // Per-column compression configs as the user passed them to
    // create_table_hdu(..., compress=...).  Stored so that
    // write-only kwargs like Gzip1(level=9) round-trip via
    // `.compression` within the same session.  For reopened HDUs
    // this is None and `.compression` falls back to rebuilding
    // dict-of-strings from the ZCTYPn cards.  One entry per column,
    // in file order; None when the HDU wasn't created with compress.
    pub(crate) compress_configs: Arc<
        Mutex<Option<Vec<CompressionConfigKind>>>,
    >,
    // Phase 4 meta cache: parsed CompressedTableMeta keyed by
    // cards_version.  Mirrors the TableHDU + CompressedImageHDU
    // pattern — first hit re-parses the synthesized uncompressed
    // schema plus per-column ZCTYPn algorithms; subsequent hits on
    // the same version return an Arc clone.  Invalidates on every
    // cards mutation via the version bump in CardsWriteGuard.
    meta_cache: MetaCache,
}

impl CompressedTableHDU {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        header: Vec<String>,
        index: usize,
        filename: String,
        offsets: Arc<HduOffsets>,
        layout: Arc<FileLayout>,
        file: FileHandle,
        tainted: TaintFlag,
        compress_configs: Option<Vec<CompressionConfigKind>>,
    ) -> PyClassInitializer<Self> {
        let hdu = HDU::new(
            header, index, filename, offsets, layout, file, tainted,
        );
        PyClassInitializer::from(hdu)
            .add_subclass(TableHDU::new_empty_cache())
            .add_subclass(CompressedTableHDU {
                cache: Arc::new(ColumnTileCache::new(
                    DEFAULT_TILE_CACHE_BYTES,
                )),
                compress_configs: Arc::new(Mutex::new(compress_configs)),
                meta_cache: Arc::new(Mutex::new(None)),
            })
    }

    // Return the parsed-once metadata for this HDU.  Hot-path
    // accessor: one Mutex lock + Acquire version load + (on hit)
    // an Arc clone.  On miss takes a header snapshot and re-parses.
    // Same shape as TableHDU::meta and CompressedImageHDU::meta;
    // callers reach this via `slf.as_super()` to keep both `slf`
    // and the base HDU alive for the call.  Note: data_offset is
    // NOT in the returned meta (offsets can change when earlier
    // HDUs grow); callers fetch it fresh from `super_.offsets`.
    pub(crate) fn meta(
        &self, super_: &HDU,
    ) -> PyResult<Arc<CompressedTableMeta>> {
        crate::common::check_not_tainted(&super_.tainted)?;
        let cur_version = super_.cards_version
            .load(std::sync::atomic::Ordering::Acquire);
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
        let meta = Arc::new(parse_compressed_table_meta(cards)?);
        let mut cache = self.meta_cache.lock()
            .map_err(|_| PyIOError::new_err("meta cache poisoned"))?;
        *cache = Some((cur_version, Arc::clone(&meta)));
        Ok(meta)
    }
}

#[pymethods]
impl CompressedTableHDU {
    // Multi-line repr matching TableHDU's, but reporting the
    // *uncompressed* row count and per-column dtypes (what the user
    // would see after .read()), plus a compression-info line listing
    // the per-column algorithm.
    fn __repr__(slf: PyRef<'_, Self>, _py: Python<'_>) -> PyResult<String> {
        let super_ = slf.as_super().as_super();
        // Try the cached meta first; fall back to a fresh cards
        // snapshot if the file is degenerate — repr must never crash.
        let cached = slf.meta(super_).ok();
        let cards = super_.header_snapshot()?;
        let (columns, nrows, n_tiles, ztilelen): (Vec<Column>, i64, i64, Option<i64>) =
            match &cached {
                Some(m) => (
                    m.columns.clone(),
                    m.nrows as i64,
                    m.n_tiles as i64,
                    Some(m.ztilelen as i64),
                ),
                None => {
                    let virtual_cards =
                        synthesize_uncompressed_cards(&cards);
                    (
                        parse_columns(&virtual_cards).unwrap_or_default(),
                        parse_keyword(&cards, "ZNAXIS2")
                            .unwrap_or(0).max(0),
                        parse_keyword(&cards, "NAXIS2")
                            .unwrap_or(0).max(0),
                        parse_keyword(&cards, "ZTILELEN"),
                    )
                }
            };
        let extname = parse_string_keyword(&cards, "EXTNAME");

        let mut out = String::new();
        out.push_str(&format!("  file: {}\n", super_.filename));
        out.push_str(&format!("  extension: {}\n", super_.index));
        out.push_str("  type: BINARY_TBL (compressed)\n");
        if let Some(name) = extname {
            out.push_str(&format!("  extname: {}\n", name));
        }
        out.push_str(&format!("  rows: {}\n", nrows));
        out.push_str(&format!("  tiles: {}\n", n_tiles));
        if let Some(z) = ztilelen {
            if z > 0 {
                out.push_str(&format!("  rows per tile: {}\n", z));
            }
        }
        // compression_algorithms is a cheap walk (one TFIELDS lookup
        // + per-column TTYPE/ZCTYP lookups) and returns raw strings
        // rather than the enum form the meta cache holds — leave
        // it as a direct card walk.
        let algos = compression_algorithms(&cards);
        if !algos.is_empty() {
            let summary: Vec<String> = algos.iter()
                .map(|(n, a)| format!("{}={}", n, a))
                .collect();
            out.push_str(&format!(
                "  compression: {}\n", summary.join(", ")));
        }
        out.push_str("  column info:\n");
        let max_name = columns.iter().map(|c| c.name.len()).max().unwrap_or(0);
        let width = max_name + 4;
        for col in &columns {
            let (dtype_s, shape_s) = column_repr_info(col);
            out.push_str(&format!(
                "    {:<w$}{}", col.name, dtype_s, w = width));
            if let Some(s) = shape_s {
                out.push_str(&format!("  {}", s));
            }
            if let Some(u) = &col.tunit {
                out.push_str(&format!("  ({})", u));
            }
            out.push('\n');
        }
        Ok(out)
    }

    // -------------------------------------------------------------------
    // Uncompressed-view accessors (override the inherited TableHDU ones)
    // -------------------------------------------------------------------
    //
    // All sourced from the SYNTHESIZED uncompressed schema (see
    // synthesize_uncompressed_cards): ZNAXIS1 → row width, ZNAXIS2
    // → row count, ZFORMn → original TFORMn.  The on-disk BINTABLE
    // has TFORMn='1QB(...)' per column and would mislead the
    // inherited TableHDU accessors.

    /// Number of rows in the ORIGINAL (uncompressed) table.
    ///
    /// Sourced from ``ZNAXIS2``; the on-disk ``NAXIS2`` holds the
    /// number of tile chunks, not the user-visible row count.
    #[getter]
    fn nrows(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super().as_super();
        Ok(slf.meta(super_)?.nrows)
    }

    // __len__ is a pyo3 slot dunder — no per-method docstring.
    // Same value as nrows; matches len(structured_arr) for the
    // array a full read() returns.
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super().as_super();
        Ok(slf.meta(super_)?.nrows)
    }

    /// Number of columns in the table (``TFIELDS``).
    #[getter]
    fn ncols(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super().as_super();
        Ok(slf.meta(super_)?.columns.len())
    }

    /// The numpy structured dtype the original (uncompressed)
    /// table reads into.
    ///
    /// Same scaling rules as :attr:`TableHDU.dtype`; sourced from
    /// the synthesized uncompressed-view cards.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        build_numpy_dtype(py, &meta.columns, /* scale = */ true)
    }

    /// Column names in on-disk order, as a tuple.  Same semantics
    /// as :attr:`TableHDU.colnames`.
    #[getter]
    fn colnames(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        let names: Vec<&str> = meta.columns.iter()
            .map(|c| c.name.as_str()).collect();
        Ok(PyTuple::new(py, &names)?.unbind())
    }

    /// Per-column units (``TUNITn``), as a dict.  Same semantics
    /// as :attr:`TableHDU.units`.
    #[getter]
    fn units(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        let dict = PyDict::new(py);
        for col in &meta.columns {
            dict.set_item(&col.name, col.tunit.as_deref())?;
        }
        Ok(dict.unbind())
    }

    // -------------------------------------------------------------------
    // Compression-specific accessors
    // -------------------------------------------------------------------

    /// Per-column compression algorithm, as a dict.
    ///
    /// ``{column_name: zctyp_value}`` preserving on-disk column
    /// order.  Column names come from ``TTYPEn`` (preserved
    /// verbatim from the original table).  Algorithm strings
    /// are the FITS-spec ``ZCTYPn`` values found on disk
    /// (``'RICE_1'`` / ``'GZIP_1'`` / ``'GZIP_2'``).
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{column_name: algorithm_string}``.
    #[getter]
    fn compression(
        slf: PyRef<'_, Self>, py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let dict = PyDict::new(py);
        for (name, algo) in compression_algorithms(&cards) {
            dict.set_item(name, algo)?;
        }
        Ok(dict.unbind())
    }

    /// Number of tile chunks the original table was split into.
    ///
    /// Equals the compressed table's on-disk ``NAXIS2`` — one
    /// BINTABLE row per tile.
    #[getter]
    fn n_tiles(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super().as_super();
        Ok(slf.meta(super_)?.n_tiles)
    }

    /// Rows per tile used at compression time (``ZTILELEN``).
    ///
    /// The last tile may contain fewer rows if ``ZNAXIS2`` is
    /// not a multiple of ``ZTILELEN``.
    #[getter]
    fn ztile_rows(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super().as_super();
        Ok(slf.meta(super_)?.ztilelen)
    }

    // -------------------------------------------------------------------
    // I/O surface (overrides the inherited TableHDU versions)
    // -------------------------------------------------------------------

    /// Read the (decompressed) table into a numpy structured array.
    ///
    /// Same signature as :meth:`TableHDU.read`: ``rows`` /
    /// ``columns`` / ``scale`` / ``mask_null``.  Per-tile decode
    /// runs lazily — only the tiles overlapping the requested
    /// row range are decompressed.
    ///
    /// Currently unsupported:
    ///
    /// * ``mask_null=True`` raises ``NotImplementedError`` —
    ///   TNULL masking on compressed-table reads is a separate
    ///   follow-up.
    ///
    /// Notes
    /// -----
    /// Decoded (tile, column) byte slabs populate the LRU cache
    /// (subject to :attr:`tile_cache_size`).  Subsequent reads
    /// of the same column range, or of other columns within the
    /// same tile range, hit warm slabs.
    #[pyo3(signature = (*, rows=None, columns=None, scale=true, mask_null=false))]
    fn read(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        columns: Option<Vec<String>>,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<Py<PyAny>> {
        if mask_null {
            return Err(PyNotImplementedError::new_err(
                "CompressedTableHDU.read(mask_null=True): TNULL masking \
                 on compressed-table reads is not yet implemented"));
        }
        let cache = Arc::clone(&slf.cache);
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let data_offset = super_.offsets.data_offset();
        read_compressed_table(
            py, &cards, data_offset, &super_.file, rows, columns, scale,
            &cache,
        )
    }

    // __getitem__ is a pyo3 slot dunder — no per-method docstring.
    // Same dispatch as TableHDU.__getitem__:
    //   hdu[i]            → 0-d structured record at row i
    //   hdu[i:j[:s]]      → structured ndarray covering the slice
    //   hdu[[i, j, k]]    → fancy-row structured ndarray
    //   hdu["col"]        → CompressedSingleColumnSubset
    //   hdu[["a", "b"]]   → CompressedColumnSubset
    fn __getitem__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let kind = classify_table_key(key)?;
        match kind {
            TableKey::Rows => {
                let pyref = slf.borrow();
                let cache = Arc::clone(&pyref.cache);
                let super_ = pyref.into_super().into_super();
                let cards = super_.header_snapshot()?;
                let data_offset = super_.offsets.data_offset();
                read_compressed_table(
                    py, &cards, data_offset, &super_.file,
                    Some(key), None, /* scale = */ true, &cache,
                )
            }
            TableKey::SingleRow(idx) => {
                let pyref = slf.borrow();
                let cache = Arc::clone(&pyref.cache);
                let super_ = pyref.into_super().into_super();
                let cards = super_.header_snapshot()?;
                let data_offset = super_.offsets.data_offset();
                // Same trick as TableHDU.__getitem__: wrap the
                // bare-int request as a single-element list so
                // resolve_rows handles the negative-index +
                // bounds-check semantics, then unwrap [0] for the
                // 0-d record return value.
                let one = PyList::new(py, [idx])?;
                let arr = read_compressed_table(
                    py, &cards, data_offset, &super_.file,
                    Some(one.as_any()), None,
                    /* scale = */ true, &cache,
                )?;
                Ok(arr.bind(py).get_item(0)?.unbind())
            }
            TableKey::SingleColumn(name) => {
                let hdu_py: Py<CompressedTableHDU> = slf.clone().unbind();
                Ok(Py::new(py, CompressedSingleColumnSubset {
                    hdu: hdu_py, name,
                })?.into())
            }
            TableKey::MultiColumns(names) => {
                let hdu_py: Py<CompressedTableHDU> = slf.clone().unbind();
                Ok(Py::new(py, CompressedColumnSubset {
                    hdu: hdu_py, columns: names,
                })?.into())
            }
        }
    }

    // ----- Tile cache controls (mirror CompressedImageHDU) -----
    //
    // Cache keys are (tile_idx, col_idx) — per-(tile, column)
    // granularity, so reading just hdu["col"][i:j] only loads
    // that one column's overlapping tiles.

    /// Current tile-cache capacity in bytes.  Default 32 MiB.
    /// See :meth:`CompressedImageHDU.tile_cache_size` for details.
    #[getter]
    fn tile_cache_size(slf: PyRef<'_, Self>) -> u64 {
        slf.cache.capacity()
    }

    /// Set the tile-cache capacity in bytes.  ``0`` disables.
    fn set_tile_cache_size(&self, bytes: u64) {
        self.cache.set_capacity(bytes);
    }

    /// Bytes currently held in the tile cache.
    #[getter]
    fn tile_cache_used(slf: PyRef<'_, Self>) -> u64 {
        slf.cache.used_bytes()
    }

    /// Drop every cached ``(tile, column)`` decompressed slab.
    /// Keeps :attr:`tile_cache_size` as-is.
    fn clear_tile_cache(&self) {
        self.cache.clear();
    }

    // __setitem__ is a pyo3 slot dunder — no per-method docstring.
    // Same surface as TableHDU.__setitem__ (all 6 forms: row /
    // slice / fancy-row / column / cell / multi-column / subset
    // chained), including VLA columns.  Modified tiles are
    // decoded → row bytes overwritten → re-encoded + appended to
    // the heap end (orphaning the old blobs; repack() reclaims).
    fn __setitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let cfgs = slf.compress_configs.lock()
            .map_err(|_| PyIOError::new_err(
                "compress_configs lock poisoned"))?
            .clone();
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
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
            cache: &slf.cache,
        };
        let all_cols: Vec<usize> = (0..meta.columns.len()).collect();

        match classify_setitem_key(key)? {
            SetItemKey::SingleRow(i) => {
                let r = normalize_disk_row(i, meta.nrows)?;
                let arr = coerce_to_len1_record(py, value)?;
                let per_column =
                    crate::hdu_table::extract_per_column_inputs(
                        py, &arr, None, &meta.columns)?;
                setitem_compressed_cols(
                    py, &ctx, &per_column, &all_cols, &[r],
                )
            }
            SetItemKey::RowSlice => {
                let slice_py = key.cast::<PySlice>()?;
                let indices = slice_py.indices(meta.nrows as isize)?;
                if indices.step <= 0 {
                    return Err(PyValueError::new_err(
                        "CompressedTableHDU[slice] = value: negative or \
                         zero step is not supported"));
                }
                let count = indices.slicelength;
                require_ndarray_with_length(
                    py, value, count, "CompressedTableHDU[slice]",
                )?;
                if count == 0 {
                    return Ok(());
                }
                let start = indices.start as usize;
                let step = indices.step as usize;
                let disk_rows: Vec<usize> =
                    (0..count).map(|r| start + r * step).collect();
                let per_column =
                    crate::hdu_table::extract_per_column_inputs(
                        py, value, None, &meta.columns)?;
                setitem_compressed_cols(
                    py, &ctx, &per_column, &all_cols, &disk_rows,
                )
            }
            SetItemKey::FancyRows(rows) => {
                let count = rows.len();
                require_ndarray_with_length(
                    py, value, count, "CompressedTableHDU[[rows]]",
                )?;
                if count == 0 {
                    return Ok(());
                }
                let disk_rows: Vec<usize> = rows.iter()
                    .map(|&i| normalize_disk_row(i, meta.nrows))
                    .collect::<PyResult<_>>()?;
                let per_column =
                    crate::hdu_table::extract_per_column_inputs(
                        py, value, None, &meta.columns)?;
                setitem_compressed_cols(
                    py, &ctx, &per_column, &all_cols, &disk_rows,
                )
            }
            SetItemKey::SingleColumn(name) => {
                let col_idx = find_compressed_column_index(
                    &meta.columns, &name)?;
                let label = format!("CompressedTableHDU['{}']", name);
                require_ndarray(py, value, &label)?;
                let disk_rows: Vec<usize> = (0..meta.nrows).collect();
                let per_column = vec![value.clone()];
                setitem_compressed_cols(
                    py, &ctx, &per_column, &[col_idx], &disk_rows,
                )
            }
            SetItemKey::MultiColumns(names) => {
                if names.is_empty() {
                    return Err(PyValueError::new_err(
                        "CompressedTableHDU[[names]] = value: empty \
                         column list"));
                }
                require_ndarray_with_length(
                    py, value, meta.nrows, "CompressedTableHDU[[names]]",
                )?;
                let (selected, per_column) =
                    resolve_structured_subset_value(
                        py, value, &meta.columns, &names,
                    )?;
                let disk_rows: Vec<usize> = (0..meta.nrows).collect();
                setitem_compressed_cols(
                    py, &ctx, &per_column, &selected, &disk_rows,
                )
            }
        }
    }

    /// Compress and write data to the table.
    ///
    /// Same signature and input forms as :meth:`TableHDU.write`:
    /// structured ndarray, ``{name: ndarray}`` dict, or list of
    /// per-column ndarrays + ``names=``.  Encodes each
    /// ``(tile, column)`` per the per-column ``ZCTYPn`` algorithm
    /// the file was created with, streams compressed blobs to the
    /// heap, fills the descriptor table, and updates ``PCOUNT``.
    ///
    /// Notes
    /// -----
    /// Mid-write I/O failures taint the file (close + reopen to
    /// recover).  See :meth:`TableHDU.write` for the per-form
    /// validation rules.
    #[pyo3(signature = (data, *, names=None))]
    fn write(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let cfgs = slf.compress_configs.lock()
            .map_err(|_| PyIOError::new_err(
                "compress_configs lock poisoned"))?
            .clone();
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let virtual_cards = synthesize_uncompressed_cards(&cards);
        let columns = parse_columns(&virtual_cards)?;
        let nrows = parse_keyword(&cards, "ZNAXIS2")
            .unwrap_or(0).max(0) as usize;
        let ztilelen = parse_keyword(&cards, "ZTILELEN")
            .unwrap_or(0).max(0) as usize;
        let n_tiles = parse_keyword(&cards, "NAXIS2")
            .unwrap_or(0).max(0) as usize;
        let descriptor_row_width = parse_keyword(&cards, "NAXIS1")
            .unwrap_or(0).max(0) as usize;
        let data_offset = super_.offsets.data_offset();

        // Algorithms come from ZCTYPn cards (single source of truth
        // — for reopened HDUs the stored configs may be None).
        let mut algorithms: Vec<CompressionAlgorithm> =
            Vec::with_capacity(columns.len());
        for i in 0..columns.len() {
            let key = format!("ZCTYP{}", i + 1);
            let zctyp = parse_string_keyword(&cards, &key)
                .ok_or_else(|| PyValueError::new_err(format!(
                    "compressed table missing {} card", key)))?;
            algorithms.push(parse_algorithm(&zctyp)?);
        }

        // Normalize the input to a Vec of per-column ndarrays.  The
        // helper handles structured-ndarray / dict / list+names
        // dispatch + per-form validation (extras / missing / wrong
        // names) — same logic the uncompressed TableHDU.write uses.
        let per_column = crate::hdu_table::extract_per_column_inputs(
            py, data, names, &columns,
        )?;

        write_compressed_table_data(
            py, &super_, &cards, &per_column, &columns, &algorithms,
            cfgs.as_deref(), nrows, ztilelen, n_tiles,
            descriptor_row_width, data_offset,
        )
    }

    /// Append rows to a compressed table.
    ///
    /// Same signature and input forms as :meth:`TableHDU.append`.
    ///
    /// Notes
    /// -----
    /// **Merge-into-last-partial semantics.**  If the existing
    /// last tile has fewer than ``ztile_rows`` rows, ``append``
    /// decodes it, concatenates the first ``M`` new rows (``M =
    /// ztile_rows - last_tile_rows``), re-encodes the now-fuller
    /// tile, and appends the new blobs to the heap end.  Old
    /// last-tile blobs become orphans (``PCOUNT`` grows
    /// monotonically — call :meth:`repack` to reclaim them).
    /// Any rows that didn't fit are encoded as fresh full tiles.
    /// Maintains the FITS Tile Compression Convention's "all
    /// tiles same size except the last" invariant.
    ///
    /// VLA columns are supported: existing per-cell compressed
    /// bytes are copied verbatim (no decode + re-encode);
    /// merge-tile new rows are encoded via the per-cell-then-
    /// fallback path; original-table heap offsets continue from
    /// the current ``ZPCOUNT`` so a funpack-reconstructed file
    /// stays consistent.
    #[pyo3(signature = (data, *, names=None))]
    fn append(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let cfgs = slf.compress_configs.lock()
            .map_err(|_| PyIOError::new_err(
                "compress_configs lock poisoned"))?
            .clone();
        let cache = Arc::clone(&slf.cache);
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let virtual_cards = synthesize_uncompressed_cards(&cards);
        let columns = parse_columns(&virtual_cards)?;
        let existing_nrows = parse_keyword(&cards, "ZNAXIS2")
            .unwrap_or(0).max(0) as usize;
        let ztilelen = parse_keyword(&cards, "ZTILELEN")
            .unwrap_or(0).max(0) as usize;
        let existing_n_tiles = parse_keyword(&cards, "NAXIS2")
            .unwrap_or(0).max(0) as usize;
        let descriptor_row_width = parse_keyword(&cards, "NAXIS1")
            .unwrap_or(0).max(0) as usize;
        let current_pcount = parse_keyword(&cards, "PCOUNT")
            .unwrap_or(0).max(0) as u64;
        let data_offset = super_.offsets.data_offset();

        let mut algorithms: Vec<CompressionAlgorithm> =
            Vec::with_capacity(columns.len());
        for i in 0..columns.len() {
            let key = format!("ZCTYP{}", i + 1);
            let zctyp = parse_string_keyword(&cards, &key)
                .ok_or_else(|| PyValueError::new_err(format!(
                    "compressed table missing {} card", key)))?;
            algorithms.push(parse_algorithm(&zctyp)?);
        }

        let per_column = crate::hdu_table::extract_per_column_inputs(
            py, data, names, &columns,
        )?;
        append_compressed_table_data(
            py, &super_, &cards, &per_column, &columns, &algorithms,
            cfgs.as_deref(), existing_nrows, ztilelen, existing_n_tiles,
            descriptor_row_width, data_offset, current_pcount, &cache,
        )
    }

    /// Alias for :meth:`append`.  Mirrors :meth:`TableHDU.extend`
    /// for parity with :meth:`ImageHDU.extend` — generic code
    /// iterating HDUs and calling ``.extend(...)`` keeps working.
    #[pyo3(signature = (data, *, names=None))]
    fn extend(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Self::append(slf, py, data, names)
    }

    /// Rebuild the heap, reclaiming orphan blobs.
    ///
    /// :meth:`append` (when a merge into the last partial tile
    /// re-encodes the existing blobs) and ``__setitem__`` (every
    /// affected tile re-encoded and appended to the heap end)
    /// leave the old compressed bytes as orphans referenced by
    /// no descriptor.  ``repack`` walks every live descriptor,
    /// streams its referenced bytes into a compact new heap, and
    /// rewrites descriptors to point at it.
    ///
    /// Shrinks the on-disk file when the new heap is smaller
    /// (last HDU: ``set_len``; non-last HDU: trailing HDUs shift
    /// backward in lockstep).  No-op for an already-compact
    /// heap.
    fn repack(slf: PyRef<'_, Self>) -> PyResult<()> {
        let cache = Arc::clone(&slf.cache);
        let super_ = slf.into_super().into_super();
        repack_compressed_table_heap(&super_, &cache)
    }

    /// Not supported on compressed tables.
    ///
    /// Schema edits would require re-encoding every tile.
    /// Workaround: build a fresh :class:`CompressedTableHDU` with
    /// the new schema via :meth:`FITS.create_table_hdu`
    /// (``compress=`` set) + :meth:`write`.
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     Always.
    fn insert_column(
        _slf: PyRef<'_, Self>,
        _name: &str,
        _data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.insert_column() — schema edits on \
             compressed tables are not planned for the current roadmap"))
    }

    /// Not supported on compressed tables.  See
    /// :meth:`insert_column` for the workaround.
    fn delete_column(
        _slf: PyRef<'_, Self>,
        _key: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.delete_column() — schema edits on \
             compressed tables are not planned for the current roadmap"))
    }

    // Compressed tables use ZHECKSUM / ZDATASUM per the FITS Tile
    // Compression Convention: both are computed against the
    // EQUIVALENT UNCOMPRESSED table (the original schema rebuilt
    // from the Z-prefixed cards, with cell data decoded back to
    // its BITPIX-native big-endian layout).  Astropy uses the
    // same convention and reads our values bit-exact.

    /// Compute and store the ``ZDATASUM`` checksum card.
    ///
    /// Computed against the equivalent uncompressed table
    /// (original schema rebuilt from the Z-prefixed cards with
    /// cell data decoded back to its BITPIX-native big-endian
    /// layout), not the on-disk BINTABLE — per the FITS Tile
    /// Compression Convention.  Same manual-refresh contract as
    /// :meth:`TableHDU.add_datasum` — re-run after :meth:`write`
    /// / :meth:`append` / ``__setitem__`` / :meth:`repack`.
    fn add_datasum(slf: PyRef<'_, Self>) -> PyResult<()> {
        let super_ = slf.into_super().into_super();
        compressed_table_add_datasum(&super_)
    }

    /// Compute and store both ``ZDATASUM`` and ``ZHECKSUM`` cards.
    ///
    /// Same equivalent-uncompressed convention as
    /// :meth:`add_datasum`.  This is the call most users want.
    fn add_checksum(slf: PyRef<'_, Self>) -> PyResult<()> {
        let super_ = slf.into_super().into_super();
        compressed_table_add_checksum(&super_)
    }

    /// Verify the stored ``ZDATASUM`` against the equivalent
    /// uncompressed table bytes.
    ///
    /// Returns ``True`` / ``False`` / ``None`` (``None`` means
    /// the card is absent).
    fn verify_datasum(slf: PyRef<'_, Self>) -> PyResult<Option<bool>> {
        let super_ = slf.into_super().into_super();
        compressed_table_verify_datasum(&super_)
    }

    /// Verify the stored ``ZHECKSUM`` over the full HDU.
    ///
    /// Returns ``True`` / ``False`` / ``None`` (``None`` means
    /// the card is absent).
    fn verify_checksum(slf: PyRef<'_, Self>) -> PyResult<Option<bool>> {
        let super_ = slf.into_super().into_super();
        compressed_table_verify_checksum(&super_)
    }
}

// ---------------------------------------------------------------------------
// Header card synthesis: build the virtual "uncompressed" cards Vec
// ---------------------------------------------------------------------------

// Substitute the Z-prefixed cards back to their non-Z counterparts so
// `parse_columns` and friends from the regular TableHDU code path see
// the schema of the original (pre-compression) BINTABLE.  Specifically:
//   - NAXIS1 ← ZNAXIS1 (original row width)
//   - NAXIS2 ← ZNAXIS2 (original row count)
//   - PCOUNT ← ZPCOUNT (original heap size)
//   - TFORMn ← ZFORMn  (original column TFORM, including repeat count)
// Other per-column cards (TTYPEn, TDIMn, TUNITn, TZEROn, TSCALn,
// TNULLn) are preserved on disk by cfitsio's compressor and don't need
// substitution.
pub(crate) fn synthesize_uncompressed_cards(cards: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(cards.len());
    for card in cards {
        if card.len() < 8 {
            out.push(card.clone());
            continue;
        }
        let kw = card[..8].trim_end();
        // Whole-header structural substitutions
        if kw == "NAXIS1" {
            if let Some(v) = parse_keyword(cards, "ZNAXIS1") {
                out.push(format_int_card("NAXIS1", v));
                continue;
            }
        } else if kw == "NAXIS2" {
            if let Some(v) = parse_keyword(cards, "ZNAXIS2") {
                out.push(format_int_card("NAXIS2", v));
                continue;
            }
        } else if kw == "PCOUNT" {
            if let Some(v) = parse_keyword(cards, "ZPCOUNT") {
                out.push(format_int_card("PCOUNT", v));
                continue;
            }
        } else if let Some(suffix) = kw.strip_prefix("TFORM") {
            // Per-column TFORMn → look up ZFORMn for the same n.
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                let zkey = format!("ZFORM{}", suffix);
                if let Some(zform) = parse_string_keyword(cards, &zkey) {
                    out.push(format_string_card(
                        &format!("TFORM{}", suffix),
                        &zform,
                        "data format of column",
                    ));
                    continue;
                }
            }
        }
        // Drop the original Z-prefixed cards so they don't pollute the
        // synthesized header.  (parse_columns ignores them anyway, but
        // leaving them in inflates the cards count for no reason.)
        if kw.starts_with('Z') {
            continue;
        }
        out.push(card.clone());
    }
    out
}

// Helpers to build properly padded structural / string cards for the
// synthesized header.  Could share with header.rs but the call sites
// here are simple enough that local helpers are clearer than a
// re-export.
fn format_int_card(keyword: &str, value: i64) -> String {
    let raw = format!("{:<8}= {:>20}", keyword, value);
    pad_card(&raw)
}

fn format_string_card(keyword: &str, value: &str, comment: &str) -> String {
    let body = format!("{:<8}= '{}'", keyword, value);
    let with_comment = if comment.is_empty() {
        body
    } else {
        format!("{} / {}", body, comment)
    };
    pad_card(&with_comment)
}

fn pad_card(s: &str) -> String {
    let mut out = s.to_string();
    if out.len() < 80 {
        out.push_str(&" ".repeat(80 - out.len()));
    } else if out.len() > 80 {
        out.truncate(80);
    }
    out
}

// Local repr helper — mirrors the one inside hdu_table/hdu.rs but on
// the compressed-table side.  Returns the numpy dtype string +
// optional shape annotation for one column.  VLA columns shouldn't
// appear in Phase 1 (read path isn't there yet), but render
// defensively if they do.
fn column_repr_info(col: &Column) -> (String, Option<String>) {
    if col.var_kind.is_some() {
        let inner = match col.tform_letter {
            'L' => "?", 'B' => "u1", 'I' => "i2", 'J' => "i4",
            'K' => "i8", 'E' => "f4", 'D' => "f8",
            'C' => "c8", 'M' => "c16", 'A' => "S",
            _ => return (col.tform_letter.to_string(),
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

// ---------------------------------------------------------------------------
// Per-column compression-algorithm map
// ---------------------------------------------------------------------------

// Walk per-column ZCTYPn cards and pair them with the TTYPEn names.
// Falls back to "COL<n>" when TTYPEn is missing (consistent with
// parse_columns's naming) and "UNKNOWN" when ZCTYPn is missing.
fn compression_algorithms(cards: &[String]) -> Vec<(String, String)> {
    let tfields = parse_keyword(cards, "TFIELDS")
        .unwrap_or(0).max(0) as usize;
    let mut out = Vec::with_capacity(tfields);
    for i in 1..=tfields {
        let name = parse_string_keyword(cards, &format!("TTYPE{}", i))
            .unwrap_or_else(|| format!("COL{}", i));
        let algo = parse_string_keyword(cards, &format!("ZCTYP{}", i))
            .unwrap_or_else(|| "UNKNOWN".to_string());
        out.push((name, algo));
    }
    out
}

