// CompressedTableHDU — pyclass for tile-compressed BINTABLEs
// (FITS Tile Compression Convention, `ZTABLE=T`).
//
// Phase 1 (this file): detection + pyclass + accessors + stubbed I/O.
// The class subclasses TableHDU so `isinstance(hdu, TableHDU)` holds on
// a compressed-table HDU, matching the CompressedImageHDU / ImageHDU
// shape on the image side.  Accessors return values from the *original*
// (uncompressed) table — `nrows` is `ZNAXIS2` rather than NAXIS2 (which
// is the number of tile chunks); `dtype` is built from the per-column
// `ZFORMn` cards rather than the on-disk `TFORMn` (which are all
// `1QB(maxlen)` heap descriptors).
//
// `read()`, `__getitem__`, `write()`, `append()`, `__setitem__`,
// `repack()`, `insert_column()`, `delete_column()`, and the checksum
// methods all raise `NotImplementedError("ZTABLE Phase N — coming
// later")`.  Phase 2 will land whole-table read across all three
// algorithms (GZIP_1 / GZIP_2 / RICE_1); later phases add slicing, VLA,
// and the write side.

use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

use crate::cache::BytesBoundLruCache;
use crate::common::{
    byteswap_in_place, lock_file, parse_keyword, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, RawBuffer, TaintFlag,
};
use crate::zimage::compression_config::CompressionConfigKind;
use crate::hdu::HDU;
use crate::hdu_table::{
    apply_transform_cell, build_numpy_dtype, build_var_cell_value,
    bytes_per_element, byteswap_unit, classify_table_key,
    column_expected_shape, column_transform, convert_column_cell,
    field_dtype_and_shape, numpy_field_layout, parse_columns,
    plan_vla_heap_layout, read_descriptor, resolve_columns, resolve_rows,
    scaling_kind, serialize_vla_cell, validate_vla_cell, write_descriptor,
    Column, ScalingKind, TableHDU, TableKey, VlaCellPlan, WriteTransform,
};
use crate::zimage::gzip::{decode_gzip1, decode_gzip2};
use crate::zimage::rice::decode_rice;
use crate::zimage::{parse_algorithm, CompressionAlgorithm};

// 32 MiB default — matches CompressedImageHDU's default; large enough
// to cache the per-column slabs for a handful of typical tiles, small
// enough not to surprise desktop users.
const DEFAULT_TILE_CACHE_BYTES: u64 = 32 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Per-(tile, column) decompressed-bytes cache
// ---------------------------------------------------------------------------
//
// Same shape as the ZIMAGE TileCache (bytes-bound LRU, brief mutex
// around get/put, value is `Arc<Vec<u8>>` so callers work without
// holding the lock).  Key is the packed (tile_idx, col_idx) pair so
// the LRU policy is per-(tile, col) — useful when a user reads
// `hdu[i:j]` to pull a single column from many tiles: subsequent
// reads of nearby rows or sibling columns can reuse adjacent cache
// entries instead of re-decompressing.

// Per-(tile_idx, col_idx) decompressed-bytes cache key.  Packed
// into a tuple so the shared `BytesBoundLruCache` can hash it
// directly.  Finer granularity than ZIMAGE's per-tile key is what
// makes reading `hdu["col"][i:j]` reusable across nearby rows /
// sibling columns.
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub(crate) struct CacheKey(u32, u32);

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

#[pyclass(extends = TableHDU)]
pub(crate) struct CompressedTableHDU {
    cache: Arc<ColumnTileCache>,
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
}

impl CompressedTableHDU {
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
            .add_subclass(TableHDU)
            .add_subclass(CompressedTableHDU {
                cache: Arc::new(ColumnTileCache::new(
                    DEFAULT_TILE_CACHE_BYTES,
                )),
                compress_configs: Arc::new(Mutex::new(compress_configs)),
            })
    }
}

#[pymethods]
impl CompressedTableHDU {
    // Multi-line repr matching TableHDU's, but reporting the
    // *uncompressed* row count and per-column dtypes (what the user
    // would see after .read()), plus a compression-info line listing
    // the per-column algorithm.
    fn __repr__(slf: PyRef<'_, Self>, _py: Python<'_>) -> PyResult<String> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let virtual_cards = synthesize_uncompressed_cards(&cards);
        let nrows = parse_keyword(&cards, "ZNAXIS2")
            .unwrap_or(0).max(0);
        let n_tiles = parse_keyword(&cards, "NAXIS2")
            .unwrap_or(0).max(0);
        let extname = parse_string_keyword(&cards, "EXTNAME");
        let columns = parse_columns(&virtual_cards).unwrap_or_default();

        let mut out = String::new();
        out.push_str(&format!("  file: {}\n", super_.filename));
        out.push_str(&format!("  extension: {}\n", super_.index));
        out.push_str("  type: BINARY_TBL (compressed)\n");
        if let Some(name) = extname {
            out.push_str(&format!("  extname: {}\n", name));
        }
        out.push_str(&format!("  rows: {}\n", nrows));
        out.push_str(&format!("  tiles: {}\n", n_tiles));
        if let Some(ztilelen) = parse_keyword(&cards, "ZTILELEN") {
            out.push_str(&format!("  rows per tile: {}\n", ztilelen));
        }
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

    // Number of rows in the ORIGINAL (uncompressed) table.  Reads
    // ZNAXIS2 — the compressed-table NAXIS2 holds the number of tile
    // chunks, not the user-visible row count.
    #[getter]
    fn nrows(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "ZNAXIS2").unwrap_or(0).max(0) as usize)
    }

    // Pythonic length: matches `len(structured_arr)` for the array
    // a future `.read()` will return.
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "ZNAXIS2").unwrap_or(0).max(0) as usize)
    }

    // numpy structured dtype the original (uncompressed) table would
    // read into.  Synthesized from the Z-prefixed cards (ZFORMn carries
    // the original TFORMn, ZNAXIS1 the original row width, ZNAXIS2 the
    // original row count, ZPCOUNT the original heap size).  TDIM /
    // TTYPE / TUNIT are preserved on disk and don't need
    // substitution.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let virtual_cards = synthesize_uncompressed_cards(&cards);
        let columns = parse_columns(&virtual_cards)?;
        build_numpy_dtype(py, &columns, /* scale = */ true)
    }

    // Column names in file order.  Overrides TableHDU.colnames so the
    // parser walks the SYNTHESIZED cards (the on-disk TFORMn are all
    // 1QB descriptors and may carry TDIMn that parse_columns rejects
    // on a variable-length column).
    #[getter]
    fn colnames(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let virtual_cards = synthesize_uncompressed_cards(&cards);
        let columns = parse_columns(&virtual_cards)?;
        let names: Vec<&str> = columns.iter()
            .map(|c| c.name.as_str()).collect();
        Ok(PyTuple::new(py, &names)?.unbind())
    }

    // Per-column units dict (TUNITn).  Same override reason as
    // colnames — parser must see the original schema.
    #[getter]
    fn units(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let virtual_cards = synthesize_uncompressed_cards(&cards);
        let columns = parse_columns(&virtual_cards)?;
        let dict = PyDict::new(py);
        for col in &columns {
            dict.set_item(&col.name, col.tunit.as_deref())?;
        }
        Ok(dict.unbind())
    }

    // -------------------------------------------------------------------
    // Compression-specific accessors
    // -------------------------------------------------------------------

    // Per-column compression algorithm: dict {col_name: zctyp_value}
    // preserving on-disk column order.  Column names are read from
    // TTYPEn (preserved verbatim from the original table).  Algorithm
    // strings are returned as the FITS-spec form found on disk
    // (RICE_1 / GZIP_1 / GZIP_2 / NOCOMPRESS — typically one of the
    // three algorithms cfitsio's fits_compress_table emits).
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

    // Number of tile chunks the original table was split into.
    // Equals the compressed table's NAXIS2 — one row per tile.
    #[getter]
    fn n_tiles(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize)
    }

    // Rows-per-tile setting used at compression time.  Reads
    // ZTILELEN.  The last tile may contain fewer rows if
    // ZNAXIS2 is not a multiple of ZTILELEN.
    #[getter]
    fn ztile_rows(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "ZTILELEN").unwrap_or(0).max(0) as usize)
    }

    // -------------------------------------------------------------------
    // I/O surface — all stubbed; later phases will fill these in.
    // -------------------------------------------------------------------

    // Read the (decompressed) table into a numpy structured ndarray.
    //
    //   rows=None    every row in file order, shape (ZNAXIS2,).
    //   rows=slice   range of rows; step!=1 supported via per-tile
    //                row dispatch.
    //   rows=list    arbitrary indices in user-requested order
    //                (deduped — first occurrence wins).
    //   columns=None all columns in file order.
    //   columns=list subset + reorder, case-insensitive name lookup.
    //   scale=True   apply TSCAL/TZERO (unsigned trick + general).
    //
    // Only fixed columns are supported in Phase 2/3; VLA columns
    // raise NotImplementedError (Phase 4).  mask_null=True is not
    // yet implemented (TNULL masking on compressed-table reads is a
    // separate follow-up).
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

    // Standard table-shaped __getitem__:
    //   hdu[i]            → 0-d structured record (numpy.void) at row i
    //   hdu[i:j[:s]]      → structured ndarray covering the slice
    //   hdu[[i, j, k]]    → fancy-row structured ndarray in input order
    //   hdu["col"]        → CompressedSingleColumnSubset (chained reads)
    //   hdu[["a", "b"]]   → CompressedColumnSubset
    //
    // Dispatch mirrors TableHDU.__getitem__ exactly via the shared
    // classify_table_key/TableKey machinery.
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

    // Current cache capacity in bytes (default 32 MiB).
    #[getter]
    fn tile_cache_size(slf: PyRef<'_, Self>) -> u64 {
        slf.cache.capacity()
    }

    // Set capacity in bytes; 0 disables caching.  Shrinking below
    // current usage evicts LRU entries to fit.
    fn set_tile_cache_size(&self, bytes: u64) {
        self.cache.set_capacity(bytes);
    }

    // Bytes currently held in the cache.
    #[getter]
    fn tile_cache_used(slf: PyRef<'_, Self>) -> u64 {
        slf.cache.used_bytes()
    }

    // Drop every cached (tile, column) decompressed slab.  Keeps the
    // capacity setting in place.
    fn clear_tile_cache(&self) {
        self.cache.clear();
    }

    fn __setitem__(
        _slf: PyRef<'_, Self>,
        _key: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.__setitem__ — ZTABLE Phase 6+ will add \
             this; in-place mutation of compressed tables requires \
             re-encoding affected tiles"))
    }

    // Compress and write `data` to the table.  Accepts the same
    // three input forms as TableHDU.write:
    //   - structured numpy ndarray with the table's field names
    //   - dict {col_name: ndarray}
    //   - list/tuple of ndarrays + names=[...]
    //
    // Encodes each (tile, column) per the per-column ZCTYPn
    // algorithm, streams compressed blobs to the heap, fills the
    // descriptor table, and updates PCOUNT.  Mid-write I/O failures
    // taint the file (close + reopen to recover).
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

    // Append rows to a compressed table.  Accepts the same three
    // input forms as TableHDU.write (structured ndarray / dict /
    // list+names).
    //
    // **Merge-into-last-partial semantics.**  If the existing last
    // tile has fewer than ZTILELEN rows, append() decodes it,
    // concatenates the first M new rows (M = ZTILELEN - last_tile_rows),
    // re-encodes the now-fuller tile, and appends the new blobs to
    // the heap end.  The old last-tile blobs become orphans (PCOUNT
    // grows monotonically — `repack()` would reclaim them, not yet
    // implemented on compressed tables).  Any rows that didn't fit
    // are encoded as fresh full tiles.  Maintains the FITS Tile
    // Compression Convention's "all tiles same size except the
    // last" invariant so funpack and our own reader both work.
    //
    // VLA columns supported: existing per-cell compressed bytes are
    // copied verbatim (no decode/re-encode); merge-tile new rows are
    // encoded via the per-cell-then-fallback path; original-table
    // heap offsets continue from current ZPCOUNT so the funpack-
    // reconstructed file stays consistent.
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

    // Alias for append, mirroring TableHDU's `extend` alias which
    // exists for parity with ImageHDU.extend (generic code that
    // iterates HDUs and calls .extend(...) keeps working).
    #[pyo3(signature = (data, *, names=None))]
    fn extend(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Self::append(slf, py, data, names)
    }

    // Rewrite the heap with only live blobs (those pointed at by
    // current descriptors), dropping orphans left behind by
    // append-with-merge (and any future mutation that always-
    // appends).  Shrinks the on-disk file when the new heap is
    // smaller.  No-op when the heap is already compact (PCOUNT
    // unchanged after the rewrite).
    //
    // VLA columns add a level of indirection (dual-descriptor
    // blobs contain compressed-Q descriptors pointing at per-cell
    // bytes); not yet supported by repack().
    fn repack(slf: PyRef<'_, Self>) -> PyResult<()> {
        let cache = Arc::clone(&slf.cache);
        let super_ = slf.into_super().into_super();
        repack_compressed_table_heap(&super_, &cache)
    }

    fn insert_column(
        _slf: PyRef<'_, Self>,
        _name: &str,
        _data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.insert_column() — schema edits on \
             compressed tables are not planned for the current roadmap"))
    }

    fn delete_column(
        _slf: PyRef<'_, Self>,
        _key: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.delete_column() — schema edits on \
             compressed tables are not planned for the current roadmap"))
    }

    // Compressed tables use ZHECKSUM / ZDATASUM per the FITS Tile
    // Compression Convention (the integrity check is against the
    // equivalent uncompressed table, not the on-disk BINTABLE).
    // Defer until the read path lands; computing ZDATASUM requires
    // the uncompressed bytes which Phase 2 provides.
    fn add_datasum(_slf: PyRef<'_, Self>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.add_datasum — ZTABLE Phase 2+ will add \
             this (ZDATASUM emitted, not DATASUM)"))
    }
    fn add_checksum(_slf: PyRef<'_, Self>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.add_checksum — ZTABLE Phase 2+ will add \
             this (ZHECKSUM emitted, not CHECKSUM)"))
    }
    fn verify_datasum(_slf: PyRef<'_, Self>) -> PyResult<Option<bool>> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.verify_datasum — ZTABLE Phase 2+ will \
             add this"))
    }
    fn verify_checksum(_slf: PyRef<'_, Self>) -> PyResult<Option<bool>> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.verify_checksum — ZTABLE Phase 2+ will \
             add this"))
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
fn synthesize_uncompressed_cards(cards: &[String]) -> Vec<String> {
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

// ---------------------------------------------------------------------------
// Phase 2 — whole-table read
// ---------------------------------------------------------------------------
//
// For each tile T = 0..n_tiles:
//   1. Read the row of N descriptors (one per ORIGINAL column) at
//      data_offset + T * descriptor_row_width.  Each descriptor is
//      16 bytes (Q kind: two big-endian i64 = nelements + heap_offset).
//   2. For each selected column C:
//      - Read `nelements` compressed bytes from the heap.
//      - Decompress per ZCTYPn:
//          - GZIP_1 → gzip decode (native-order bytes).
//          - GZIP_2 → gzip decode + reverse byte-shuffle.
//          - RICE_1 → rice decode (B/I/J only — cfitsio's table
//            compressor doesn't emit RICE for other letters).
//      - The decoder returns NATIVE-order bytes.  Byteswap back to
//        big-endian so the shared per-row cell converter
//        (`convert_column_cell`) can consume them — that function
//        is the one used by the uncompressed read path and expects
//        BE input.
//      - For each row R in the tile, copy + scale + byteswap the
//        cell into the output ndarray at row (tile_row_start + R),
//        field C.
//
// Peak memory bound per call: output ndarray + one tile's worth of
// decompressed bytes per column being processed (a few MB for
// typical fpack tile sizes).  No whole-table intermediate buffer.
#[allow(clippy::too_many_arguments)]
fn read_compressed_table(
    py: Python<'_>,
    cards: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    rows_requested: Option<&Bound<'_, PyAny>>,
    columns_requested: Option<Vec<String>>,
    scale: bool,
    cache: &ColumnTileCache,
) -> PyResult<Py<PyAny>> {
    let virtual_cards = synthesize_uncompressed_cards(cards);
    let all_columns = parse_columns(&virtual_cards)?;

    let selected: Vec<Column> = match columns_requested {
        None => all_columns.clone(),
        Some(names) => resolve_columns(&all_columns, &names)?,
    };
    let scaling_kinds: Vec<ScalingKind> = selected.iter()
        .map(|c| if scale { scaling_kind(c) } else { Ok(ScalingKind::None) })
        .collect::<PyResult<Vec<_>>>()?;

    let n_rows = parse_keyword(cards, "ZNAXIS2")
        .unwrap_or(0).max(0) as usize;
    let n_tiles = parse_keyword(cards, "NAXIS2")
        .unwrap_or(0).max(0) as usize;
    let ztilelen = parse_keyword(cards, "ZTILELEN")
        .unwrap_or(0).max(0) as usize;
    let descriptor_row_width = parse_keyword(cards, "NAXIS1")
        .unwrap_or(0).max(0) as usize;

    // Per-column algorithm — parsed once up front.  Reject unsupported
    // algorithms (HCOMPRESS_1 and PLIO_1 are image-only) so we don't
    // start reading just to bomb on a tile-by-tile basis.
    let algorithms: Vec<CompressionAlgorithm> = (0..all_columns.len())
        .map(|i| {
            let key = format!("ZCTYP{}", i + 1);
            let zctyp = parse_string_keyword(cards, &key)
                .ok_or_else(|| PyValueError::new_err(format!(
                    "compressed table missing {} card", key)))?;
            let algo = parse_algorithm(&zctyp)?;
            match algo {
                CompressionAlgorithm::Gzip1
                | CompressionAlgorithm::Gzip2
                | CompressionAlgorithm::Rice1 => Ok(algo),
                CompressionAlgorithm::Hcompress1
                | CompressionAlgorithm::Plio1 => Err(
                    PyValueError::new_err(format!(
                        "{} = '{}' — only GZIP_1, GZIP_2, and RICE_1 \
                         are valid for compressed tables",
                        key, zctyp))),
            }
        })
        .collect::<PyResult<_>>()?;

    // Heap base: respect THEAP if present, otherwise default to the
    // end of the descriptor rows.
    let theap_raw = parse_keyword(cards, "THEAP").unwrap_or(0);
    let heap_base_in_data = if theap_raw > 0 {
        theap_raw as u64
    } else {
        (n_tiles as u64) * (descriptor_row_width as u64)
    };
    let heap_start = data_offset + heap_base_in_data;

    // Resolve row selection.  When rows_requested is None we walk
    // the whole table; otherwise we get a list of disk-row indices
    // in the user's requested order (deduped, range-validated).
    let row_plan = match rows_requested {
        None => RowPlan::all(n_rows),
        Some(arg) => {
            let indices = resolve_rows(arg, n_rows)?;
            RowPlan::from_indices(indices, ztilelen)
        }
    };
    let n_out = row_plan.n_output_rows;

    // Allocate output ndarray.
    let dtype = build_numpy_dtype(py, &selected, scale)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("empty", (n_out, dtype.bind(py)))?;
    if n_out == 0 || selected.is_empty() {
        return Ok(arr.unbind());
    }

    let arr_dtype = arr.getattr("dtype")?;
    let itemsize: usize = arr_dtype.getattr("itemsize")?.extract()?;
    let field_layout = numpy_field_layout(py, &arr_dtype, &selected)?;

    // Map each selected column back to its index in the original
    // column list (case-insensitive name lookup), so we know which
    // descriptor slot to read in each tile's descriptor row.
    let selected_orig_idx: Vec<usize> = selected.iter()
        .map(|sc| all_columns.iter()
            .position(|c| c.name.eq_ignore_ascii_case(&sc.name))
            .expect("resolve_columns guaranteed presence"))
        .collect();

    // Validate descriptor row width: must be ncols * 16 (each
    // descriptor is a 1QB pair = two i64).
    let expected_desc_width = all_columns.len() * 16;
    if descriptor_row_width != expected_desc_width {
        return Err(PyValueError::new_err(format!(
            "compressed table NAXIS1 = {} but expected ncols ({}) * 16 \
             = {} bytes per descriptor row",
            descriptor_row_width, all_columns.len(), expected_desc_width)));
    }

    let mut out_buf = RawBuffer::acquire_writable(&arr)?;
    let out = out_buf.as_mut_slice();

    // Per-tile buffer (descriptors) — reused across tiles.
    let mut desc_buf = vec![0u8; descriptor_row_width];

    // Walk tiles in increasing tile_idx (best disk locality for the
    // descriptor reads + the heap-blob reads).  Output_row indices
    // come from the per-tile requests so the user's row order is
    // preserved in the final array.
    let tile_plan = row_plan.tiles_with_requests(n_tiles, ztilelen);
    for (tile_idx, requests) in tile_plan {
        let tile_row_start = tile_idx * ztilelen;
        let rows_in_tile = if tile_idx + 1 == n_tiles {
            n_rows - tile_row_start
        } else {
            ztilelen
        };

        // Read this tile's descriptor row.
        {
            let mut g = lock_file(file_handle)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            let off = data_offset
                + (tile_idx as u64) * (descriptor_row_width as u64);
            f.seek(SeekFrom::Start(off))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            f.read_exact(&mut desc_buf).map_err(|e| {
                PyIOError::new_err(format!(
                    "read descriptor row for tile {}: {}", tile_idx, e))
            })?;
        }

        for (out_col_idx, sel_col) in selected.iter().enumerate() {
            let orig_idx = selected_orig_idx[out_col_idx];
            let desc_slice = &desc_buf
                [orig_idx * 16..(orig_idx + 1) * 16];
            let (nelems_s, heap_offset_s) =
                read_descriptor('Q', desc_slice);
            if nelems_s < 0 || heap_offset_s < 0 {
                return Err(PyValueError::new_err(format!(
                    "tile {} column '{}': descriptor has negative field \
                     (nelements={}, offset={})",
                    tile_idx, sel_col.name, nelems_s, heap_offset_s)));
            }

            if sel_col.var_kind.is_some() {
                read_vla_column_tile(
                    py, &arr, file_handle, sel_col,
                    algorithms[orig_idx], cache, tile_idx, orig_idx,
                    nelems_s as usize, heap_start + heap_offset_s as u64,
                    heap_start, rows_in_tile,
                    scaling_kinds[out_col_idx], &requests,
                )?;
                continue;
            }

            let cache_key = CacheKey(tile_idx as u32, orig_idx as u32);
            let slab_arc = match cache.get(&cache_key) {
                Some(arc) => arc,
                None => {
                    let n_bytes_compressed = nelems_s as usize;
                    let mut compressed = vec![0u8; n_bytes_compressed];
                    if n_bytes_compressed > 0 {
                        let mut g = lock_file(file_handle)?;
                        let f = g.as_mut().ok_or_else(|| {
                            PyIOError::new_err("file is closed")
                        })?;
                        f.seek(SeekFrom::Start(
                            heap_start + heap_offset_s as u64
                        )).map_err(|e| {
                            PyIOError::new_err(e.to_string())
                        })?;
                        f.read_exact(&mut compressed).map_err(|e| {
                            PyIOError::new_err(format!(
                                "read heap for tile {} col '{}': {}",
                                tile_idx, sel_col.name, e))
                        })?;
                    }
                    let slab = decompress_column_slab(
                        algorithms[orig_idx], &compressed, sel_col,
                        rows_in_tile,
                    )?;
                    let arc = Arc::new(slab);
                    cache.put(cache_key, Arc::clone(&arc));
                    arc
                }
            };

            let kind = scaling_kinds[out_col_idx];
            let (field_offset, field_itemsize) = field_layout[out_col_idx];
            let src_cell_w = sel_col.byte_width;
            for req in &requests {
                let in_tile = req.in_tile_offset;
                let out_row = req.output_row;
                let src = &slab_arc
                    [in_tile * src_cell_w..(in_tile + 1) * src_cell_w];
                let dst_start = out_row * itemsize + field_offset;
                let dst = &mut out
                    [dst_start..dst_start + field_itemsize];
                convert_column_cell(sel_col, src, dst, out_row, kind)?;
            }
        }
    }

    drop(out_buf);
    Ok(arr.unbind())
}

// ---------------------------------------------------------------------------
// Row planning — group requested rows by tile
// ---------------------------------------------------------------------------

// One row to be filled in the output array: which row inside the tile
// to pull from, and which slot of the output to write into.
struct TileRowRequest {
    in_tile_offset: usize,
    output_row: usize,
}

// Plan describing which tiles are needed and, for each, which rows to
// read from and where to put them in the output.  `all_rows` flag
// distinguishes the full-table case (synthesize sequential requests
// per tile lazily) from the subset case (per-tile bucket built from
// resolve_rows output).
struct RowPlan {
    by_tile: std::collections::HashMap<usize, Vec<TileRowRequest>>,
    n_output_rows: usize,
    all_rows: bool,
}

impl RowPlan {
    fn all(n_rows: usize) -> Self {
        RowPlan {
            by_tile: std::collections::HashMap::new(),
            n_output_rows: n_rows,
            all_rows: true,
        }
    }

    // rows= path: bucket each requested disk row into its tile.
    fn from_indices(indices: Vec<usize>, ztilelen: usize) -> Self {
        let mut by_tile: std::collections::HashMap<usize, Vec<TileRowRequest>>
            = std::collections::HashMap::new();
        let n_out = indices.len();
        for (output_row, disk_row) in indices.into_iter().enumerate() {
            let tile_idx = if ztilelen > 0 { disk_row / ztilelen } else { 0 };
            let in_tile = if ztilelen > 0 { disk_row % ztilelen } else { 0 };
            by_tile.entry(tile_idx).or_default().push(TileRowRequest {
                in_tile_offset: in_tile,
                output_row,
            });
        }
        RowPlan { by_tile, n_output_rows: n_out, all_rows: false }
    }

    // Build the list of (tile_idx, requests) to walk, in increasing
    // tile_idx order (best disk locality for the descriptor + heap
    // reads).  For the all-rows path, synthesizes sequential requests
    // per tile; per-tile Vec is bounded by ztilelen so total
    // allocation is O(n_rows) — same as a row-subset call.
    fn tiles_with_requests(
        self, n_tiles: usize, ztilelen: usize,
    ) -> Vec<(usize, Vec<TileRowRequest>)> {
        if self.all_rows {
            (0..n_tiles).map(|tile_idx| {
                let tile_row_start = tile_idx * ztilelen;
                let rows_in_tile = if tile_idx + 1 == n_tiles {
                    self.n_output_rows - tile_row_start
                } else {
                    ztilelen
                };
                let reqs: Vec<TileRowRequest> = (0..rows_in_tile)
                    .map(|r| TileRowRequest {
                        in_tile_offset: r,
                        output_row: tile_row_start + r,
                    })
                    .collect();
                (tile_idx, reqs)
            }).collect()
        } else {
            let mut out: Vec<(usize, Vec<TileRowRequest>)> =
                self.by_tile.into_iter().collect();
            out.sort_by_key(|(idx, _)| *idx);
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Column-subset pyclasses returned by hdu[col] / hdu[[cols]]
// ---------------------------------------------------------------------------

// Returned by CompressedTableHDU[col] for a single str/bytes column.
// Read-only — write-side mutations on compressed tables aren't in
// the current roadmap.
#[pyclass]
pub(crate) struct CompressedSingleColumnSubset {
    hdu: Py<CompressedTableHDU>,
    name: String,
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
        let arr = read_compressed_table(
            py, &cards, data_offset, &super_.file,
            Some(rows), Some(vec![self.name.clone()]),
            /* scale = */ true, &cache,
        )?;
        // Unwrap to a plain (non-structured) ndarray view of the
        // single column — mirrors the uncompressed SingleColumnSubset.
        Ok(arr.bind(py).get_item(self.name.as_str())?.unbind())
    }
}

// Returned by CompressedTableHDU[[col1, col2, ...]].  Read-only.
#[pyclass]
pub(crate) struct CompressedColumnSubset {
    hdu: Py<CompressedTableHDU>,
    columns: Vec<String>,
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
        read_compressed_table(
            py, &cards, data_offset, &super_.file,
            Some(rows), Some(self.columns.clone()),
            /* scale = */ true, &cache,
        )
    }
}

// Dispatch on the per-column algorithm, decompress the heap blob, and
// byteswap the result back to FITS big-endian (the shared
// `convert_column_cell` expects BE input).  The existing decoders in
// `crate::zimage` byteswap to native as their last step; we undo that
// here.  The double-swap is one redundant pass per (tile, column) —
// trivially cheap relative to decompression itself; refactoring the
// decoders to expose a "leave BE" mode would shave it but isn't
// worth touching the ZIMAGE write paths for.
fn decompress_column_slab(
    algo: CompressionAlgorithm,
    compressed: &[u8],
    col: &Column,
    rowspertile: usize,
) -> PyResult<Vec<u8>> {
    let elem_bytes = bytes_per_element(col.tform_letter)
        .ok_or_else(|| PyValueError::new_err(format!(
            "column '{}': TFORM letter '{}' has no fixed element width",
            col.name, col.tform_letter)))?;
    // `col.repeat` is the number of elements per row for non-A;
    // for A it's the total byte width per row (which == repeat
    // since A's elem_bytes is 1).
    let n_elements = rowspertile * col.repeat;
    let mut slab = match algo {
        CompressionAlgorithm::Gzip1 => {
            decode_gzip1(compressed, n_elements, elem_bytes as u32)?
        }
        CompressionAlgorithm::Gzip2 => {
            decode_gzip2(compressed, n_elements, elem_bytes as u32)?
        }
        CompressionAlgorithm::Rice1 => {
            // cfitsio's table compressor only emits RICE_1 for
            // bytepix in {1, 2, 4} (B / I / J), corresponding to
            // `fits_rcomp_byte` / `fits_rcomp_short` / `fits_rcomp`.
            // Reject anything else up front rather than letting the
            // generic image-side decoder mishandle it.
            if !matches!(col.tform_letter, 'B' | 'I' | 'J') {
                return Err(PyValueError::new_err(format!(
                    "column '{}' has TFORM letter '{}' with ZCTYP=RICE_1; \
                     cfitsio's table compressor only emits RICE_1 for \
                     B/I/J columns, so this file is malformed (or written \
                     by a non-conforming tool)",
                    col.name, col.tform_letter)));
            }
            let blocksize = 32u32;  // cfitsio table-comp constant
            let zbitpix = (elem_bytes * 8) as i32;
            decode_rice(
                compressed, n_elements, elem_bytes as u32,
                blocksize, zbitpix,
            )?
        }
        _ => unreachable!("non-table algorithm filtered upstream"),
    };
    // Decoder returns native-order bytes; convert_column_cell expects
    // FITS big-endian.  Swap back so the per-cell converter (which
    // handles unsigned-trick, general scaling, A/L, etc.) just works.
    let swap_w = byteswap_unit(col.tform_letter);
    if swap_w > 1 && !cfg!(target_endian = "big") {
        byteswap_in_place(&mut slab, swap_w);
    }
    Ok(slab)
}

// ---------------------------------------------------------------------------
// Phase 4 — VLA column read
// ---------------------------------------------------------------------------
//
// For a VLA column in a single tile, the column's heap blob (pointed
// at by the 1QB main-row descriptor) is GZIP_1-compressed regardless
// of ZCTYPn — the inner data is what ZCTYPn governs.  After GZIP
// decompression the blob is exactly `rowspertile * width_orig +
// rowspertile * 16` bytes, laid out as two concatenated descriptor
// arrays:
//
//   bytes [0 .. rowspertile * width_orig)
//     original P/Q descriptors from the user-visible BINTABLE.
//     `vlalen` here is the number of *inner-type elements* in the
//     original cell — the user-visible count.
//   bytes [rowspertile * width_orig .. rowspertile * width_orig + rowspertile * 16)
//     compressed-side Q descriptors.  `cvlalen` is the number of
//     compressed bytes for the cell, `cvlastart` is the offset of
//     those bytes inside the compressed table's heap.
//
// Per-row decompression then:
//   1. Read cvlalen bytes from heap at heap_start + cvlastart.
//   2. If cvlalen == vlalen * elem_size: the cell was stored raw
//      (cfitsio's "compression didn't help" fallback) — those bytes
//      are the original BE inner-element bytes verbatim.
//   3. Else: decompress per ZCTYPn (RICE_1 / GZIP_1 / GZIP_2).
//   4. Hand the resulting BE bytes to `build_var_cell_value`, which
//      builds the per-cell numpy ndarray (or str / bytes for A) with
//      byteswap + scaling + ASCII validation handled the same way the
//      uncompressed read path handles them.
//
// The descriptor blob is cached per (tile, col) — same as fixed-
// column slabs.  Per-cell decompressed bytes are NOT cached (could
// blow up the budget on VLA-of-images patterns); each cell read
// decompresses fresh.
#[allow(clippy::too_many_arguments)]
fn read_vla_column_tile(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    file_handle: &FileHandle,
    col: &Column,
    algo: CompressionAlgorithm,
    cache: &ColumnTileCache,
    tile_idx: usize,
    orig_idx: usize,
    blob_nelems: usize,
    blob_heap_offset: u64,
    heap_start: u64,
    rowspertile: usize,
    kind: ScalingKind,
    requests: &[TileRowRequest],
) -> PyResult<()> {
    let width_orig = match col.var_kind {
        Some('P') => 8usize,
        Some('Q') => 16usize,
        _ => return Err(PyValueError::new_err(format!(
            "column '{}': expected P or Q var_kind, got {:?}",
            col.name, col.var_kind))),
    };
    let elem_size = bytes_per_element(col.tform_letter)
        .ok_or_else(|| PyValueError::new_err(format!(
            "column '{}': unsupported VLA inner letter '{}'",
            col.name, col.tform_letter)))?;
    let expected_blob_size = rowspertile * width_orig + rowspertile * 16;

    let cache_key = CacheKey(tile_idx as u32, orig_idx as u32);
    let blob_arc = match cache.get(&cache_key) {
        Some(arc) => arc,
        None => {
            let mut compressed = vec![0u8; blob_nelems];
            if blob_nelems > 0 {
                let mut g = lock_file(file_handle)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(blob_heap_offset))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.read_exact(&mut compressed).map_err(|e| {
                    PyIOError::new_err(format!(
                        "read VLA descriptor blob for tile {} col '{}': {}",
                        tile_idx, col.name, e))
                })?;
            }
            // The descriptor blob itself is ALWAYS gzip-framed (cfitsio
            // uses compress2mem_from_mem with deflateInit2 + gzip
            // windowBits) regardless of ZCTYPn — that controls the
            // *inner* per-cell compression only.  Skip the trailing
            // native byteswap that decode_gzip1 applies (we keep BE
            // descriptors for read_descriptor to consume directly).
            let blob = if blob_nelems > 0 {
                gzip_decompress_bytes(&compressed, expected_blob_size)?
            } else {
                Vec::new()
            };
            let arc = Arc::new(blob);
            cache.put(cache_key, Arc::clone(&arc));
            arc
        }
    };
    let blob = blob_arc.as_slice();

    let compressed_desc_start = rowspertile * width_orig;
    let orig_kind = col.var_kind.unwrap();

    for req in requests {
        let in_tile = req.in_tile_offset;
        let out_row = req.output_row;

        let orig_desc = &blob
            [in_tile * width_orig..(in_tile + 1) * width_orig];
        let (vlalen_s, _orig_offset) = read_descriptor(orig_kind, orig_desc);
        if vlalen_s < 0 {
            return Err(PyValueError::new_err(format!(
                "tile {} col '{}' row {}: original VLA descriptor has \
                 negative nelements ({})",
                tile_idx, col.name, in_tile, vlalen_s)));
        }
        let vlalen = vlalen_s as usize;

        let comp_desc_off =
            compressed_desc_start + in_tile * 16;
        let comp_desc = &blob[comp_desc_off..comp_desc_off + 16];
        let (cvlalen_s, cvlastart_s) = read_descriptor('Q', comp_desc);
        if cvlalen_s < 0 || cvlastart_s < 0 {
            return Err(PyValueError::new_err(format!(
                "tile {} col '{}' row {}: compressed-VLA descriptor has \
                 negative field (cvlalen={}, cvlastart={})",
                tile_idx, col.name, in_tile, cvlalen_s, cvlastart_s)));
        }
        let cvlalen = cvlalen_s as usize;

        let value = if vlalen == 0 {
            // Empty cell — no heap read, no decompression.  Defer to
            // build_var_cell_value which materializes a 0-length
            // ndarray (or "" / b"" for A).
            build_var_cell_value(
                py, col, &[], 0, out_row, /* as_bytes = */ false, kind,
            )?
        } else {
            let mut compressed_cell = vec![0u8; cvlalen];
            if cvlalen > 0 {
                let mut g = lock_file(file_handle)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(
                    heap_start + cvlastart_s as u64
                )).map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.read_exact(&mut compressed_cell).map_err(|e| {
                    PyIOError::new_err(format!(
                        "read compressed VLA bytes for tile {} col '{}' \
                         row {}: {}",
                        tile_idx, col.name, in_tile, e))
                })?;
            }

            let raw_bytes_len = vlalen.checked_mul(elem_size)
                .ok_or_else(|| PyValueError::new_err(
                    "VLA cell size overflowed usize"))?;
            let cell_be_bytes: Vec<u8> = if cvlalen == raw_bytes_len {
                // cfitsio's uncompressed fallback: when the compressed
                // form was larger than the raw, the original BE bytes
                // are stored verbatim.  No decoder invocation needed.
                compressed_cell
            } else {
                decompress_vla_cell(
                    algo, &compressed_cell, col, vlalen,
                )?
            };
            build_var_cell_value(
                py, col, &cell_be_bytes, vlalen, out_row,
                /* as_bytes = */ false, kind,
            )?
        };

        arr.get_item(col.name.as_str())?
            .set_item(out_row, value)?;
    }
    Ok(())
}

// Decompress one VLA cell's compressed bytes into BE inner-element
// bytes.  Returns `vlalen * elem_size` bytes ready for
// `build_var_cell_value`.  Same algorithm contract as the column
// decompressor: decoders return native-order bytes; we byteswap back
// to BE.
fn decompress_vla_cell(
    algo: CompressionAlgorithm,
    compressed: &[u8],
    col: &Column,
    vlalen: usize,
) -> PyResult<Vec<u8>> {
    let elem_size = bytes_per_element(col.tform_letter)
        .ok_or_else(|| PyValueError::new_err(format!(
            "column '{}': unsupported VLA inner letter '{}'",
            col.name, col.tform_letter)))?;
    let mut bytes = match algo {
        CompressionAlgorithm::Gzip1 => {
            decode_gzip1(compressed, vlalen, elem_size as u32)?
        }
        CompressionAlgorithm::Gzip2 => {
            decode_gzip2(compressed, vlalen, elem_size as u32)?
        }
        CompressionAlgorithm::Rice1 => {
            if !matches!(col.tform_letter, 'B' | 'I' | 'J') {
                return Err(PyValueError::new_err(format!(
                    "VLA column '{}' with inner letter '{}' + ZCTYP=RICE_1: \
                     cfitsio only emits RICE_1 for B/I/J VLA inner types",
                    col.name, col.tform_letter)));
            }
            decode_rice(
                compressed, vlalen, elem_size as u32, 32,
                (elem_size * 8) as i32,
            )?
        }
        _ => unreachable!("non-table algorithm filtered upstream"),
    };
    let swap_w = byteswap_unit(col.tform_letter);
    if swap_w > 1 && !cfg!(target_endian = "big") {
        byteswap_in_place(&mut bytes, swap_w);
    }
    Ok(bytes)
}

// Raw-gzip decompress to a known output length, no byteswap.  Same
// primitive as crate::zimage::gzip::decode_gzip1 but without the
// trailing native byteswap — used here because the descriptor blob
// is itself a packed array of BE descriptors that we want to feed
// to read_descriptor unchanged.
fn gzip_decompress_bytes(compressed: &[u8], expected_len: usize) -> PyResult<Vec<u8>> {
    use flate2::read::GzDecoder;
    let mut decoder = GzDecoder::new(compressed);
    let mut out: Vec<u8> = Vec::with_capacity(expected_len);
    decoder.read_to_end(&mut out).map_err(|e| {
        PyValueError::new_err(format!(
            "GZIP decompress (VLA descriptor blob): {}", e))
    })?;
    if out.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "GZIP decompress (VLA descriptor blob): expected {} bytes, \
             got {}", expected_len, out.len())));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Phase 5 — write side
// ---------------------------------------------------------------------------
//
// cfitsio's `fits_compress_table` picks per-dtype defaults that
// differ from the image side.  See CLAUDE.md for the full table;
// rules below mirror imcompress.c around line 8261:
//
//   B (u1)  -> GZIP_1   {GZIP_1, RICE_1}
//   I (i2)  -> GZIP_2   {GZIP_1, GZIP_2, RICE_1}
//   J (i4)  -> RICE_1   {GZIP_1, GZIP_2, RICE_1}
//   K (i8)  -> GZIP_2   {GZIP_1, GZIP_2}
//   E (f4)  -> GZIP_2   {GZIP_1, GZIP_2}
//   D (f8)  -> GZIP_2   {GZIP_1, GZIP_2}
//   C (c8)  -> GZIP_2   {GZIP_1, GZIP_2}
//   M (c16) -> GZIP_2   {GZIP_1, GZIP_2}
//   L (b1)  -> GZIP_1   {GZIP_1}
//   A (str) -> GZIP_1   {GZIP_1}
//   X (bit) -> GZIP_1   {GZIP_1}
//
// We're strict about the allowed-algorithm list: an explicit
// algorithm choice that's incompatible with a column dtype
// produces a ValueError naming the allowed algorithms.  Cfitsio
// silently falls back to a default; that "tolerance" silently
// gives the user something they didn't ask for, which is worse
// than asking them to fix the call.

pub(crate) fn default_table_algorithm(letter: char) -> CompressionAlgorithm {
    match letter {
        'B' | 'L' | 'A' | 'X' => CompressionAlgorithm::Gzip1,
        'J' => CompressionAlgorithm::Rice1,
        'I' | 'K' | 'E' | 'D' | 'C' | 'M' => CompressionAlgorithm::Gzip2,
        // Unknown letters land at Gzip1 (universally allowed).
        // parse_columns would have rejected anything truly bad
        // upstream; this is a safety net.
        _ => CompressionAlgorithm::Gzip1,
    }
}

fn algorithm_allowed_for_letter(
    letter: char, algo: CompressionAlgorithm,
) -> bool {
    use CompressionAlgorithm::*;
    match algo {
        Gzip1 => true,  // universally allowed
        Gzip2 => !matches!(letter, 'L' | 'A' | 'X'),
        Rice1 => matches!(letter, 'B' | 'I' | 'J'),
        // Hcompress1 and Plio1 are image-only — caller filters
        // them out before reaching this function.
        _ => false,
    }
}

fn allowed_algorithm_names_for_letter(letter: char) -> &'static str {
    match letter {
        'B' => "GZIP_1, RICE_1",
        'I' | 'J' => "GZIP_1, GZIP_2, RICE_1",
        'K' | 'E' | 'D' | 'C' | 'M' => "GZIP_1, GZIP_2",
        'L' | 'A' | 'X' => "GZIP_1",
        _ => "GZIP_1",
    }
}

// Resolve the user's compress= argument into a per-column config
// list.  Returns None when no compression was requested
// (compress=None / False), Some(Vec) otherwise.  Cell types are
// validated against the chosen algorithm before any file mutation.
//
// Accepted shapes:
//   - None / False       -> None (caller falls back to uncompressed)
//   - True               -> defaults per column
//   - str / class        -> same algorithm across all columns
//                          (must be allowed for every column)
//   - dict<str, ...>     -> per-column overrides; unspecified
//                          columns use defaults; values are
//                          strings or config-class instances
pub(crate) fn resolve_compress_arg(
    py: Python<'_>,
    compress: Option<&Bound<'_, PyAny>>,
    columns: &[Column],
) -> PyResult<Option<Vec<CompressionConfigKind>>> {
    let Some(arg) = compress else {
        return Ok(None);
    };
    if arg.is_none() {
        return Ok(None);
    }
    // bool: False -> uncompressed; True -> defaults
    if let Ok(b) = arg.extract::<bool>() {
        if !b {
            return Ok(None);
        }
        return Ok(Some(default_per_column_configs(columns)));
    }

    // dict<col_name, algo>
    if let Ok(dict) = arg.cast::<PyDict>() {
        let mut out: Vec<CompressionConfigKind> = columns.iter()
            .map(|c| build_default_config_for_letter(c.tform_letter))
            .collect();
        // Walk dict items and apply per-column overrides.
        for (key, val) in dict.iter() {
            let name: String = key.extract().map_err(|_| {
                PyValueError::new_err(
                    "compress= dict keys must be strings (column names)")
            })?;
            let pos = columns.iter()
                .position(|c| c.name.eq_ignore_ascii_case(&name))
                .ok_or_else(|| PyValueError::new_err(format!(
                    "compress= dict key '{}' does not match any column \
                     in the table",
                    name)))?;
            let cfg = CompressionConfigKind::from_pyany(&val)?;
            check_table_algorithm_allowed(
                &columns[pos], algorithm_of(&cfg),
            )?;
            out[pos] = cfg;
        }
        return Ok(Some(out));
    }

    // Otherwise treat as a single algorithm (string or class) and
    // apply it everywhere, with per-column validation.
    let cfg = CompressionConfigKind::from_pyany(arg)?;
    let _ = py;  // silence unused in the all-config path
    let algo = algorithm_of(&cfg);
    let mut out = Vec::with_capacity(columns.len());
    for col in columns {
        check_table_algorithm_allowed(col, algo)?;
        out.push(cfg.clone());
    }
    Ok(Some(out))
}

fn default_per_column_configs(columns: &[Column]) -> Vec<CompressionConfigKind> {
    columns.iter()
        .map(|c| build_default_config_for_letter(c.tform_letter))
        .collect()
}

fn build_default_config_for_letter(letter: char) -> CompressionConfigKind {
    let name = match default_table_algorithm(letter) {
        CompressionAlgorithm::Gzip1 => "GZIP_1",
        CompressionAlgorithm::Gzip2 => "GZIP_2",
        CompressionAlgorithm::Rice1 => "RICE_1",
        // Defaults for tables only use the three algorithms above.
        _ => "GZIP_1",
    };
    CompressionConfigKind::from_str(name)
        .expect("default algorithm name is always recognized")
}

fn algorithm_of(cfg: &CompressionConfigKind) -> CompressionAlgorithm {
    match cfg {
        CompressionConfigKind::Gzip1(_) => CompressionAlgorithm::Gzip1,
        CompressionConfigKind::Gzip2(_) => CompressionAlgorithm::Gzip2,
        CompressionConfigKind::Rice1(_) => CompressionAlgorithm::Rice1,
        CompressionConfigKind::Hcompress1(_) => CompressionAlgorithm::Hcompress1,
        CompressionConfigKind::Plio1(_) => CompressionAlgorithm::Plio1,
    }
}

fn check_table_algorithm_allowed(
    col: &Column, algo: CompressionAlgorithm,
) -> PyResult<()> {
    use CompressionAlgorithm::*;
    if matches!(algo, Hcompress1 | Plio1) {
        return Err(PyValueError::new_err(format!(
            "compress= column '{}': {} is an image-only algorithm and \
             cannot be used for tables (the FITS Tile Compression \
             Convention only allows GZIP_1, GZIP_2, and RICE_1 for ZTABLE)",
            col.name,
            match algo {
                Hcompress1 => "HCOMPRESS_1",
                Plio1 => "PLIO_1",
                _ => "?",
            },
        )));
    }
    if !algorithm_allowed_for_letter(col.tform_letter, algo) {
        let algo_name = match algo {
            Gzip1 => "GZIP_1",
            Gzip2 => "GZIP_2",
            Rice1 => "RICE_1",
            _ => "?",
        };
        return Err(PyValueError::new_err(format!(
            "compress= column '{}' (TFORM letter '{}'): {} is not \
             a valid algorithm for this column type.  Allowed for \
             this dtype: {}.  Pass `compress=True` for cfitsio \
             defaults or change this column's algorithm.",
            col.name, col.tform_letter, algo_name,
            allowed_algorithm_names_for_letter(col.tform_letter),
        )));
    }
    Ok(())
}

// Default ZTILELEN, picked the way cfitsio's fits_compress_table
// does (imcompress.c line 8135ish): rowspertile = max(1,
// min(nrows, 10_000_000 / row_width)).
pub(crate) fn default_ztilelen(nrows: usize, row_width: usize) -> usize {
    if nrows == 0 {
        return 1;
    }
    let cap = 10_000_000usize / row_width.max(1);
    cap.max(1).min(nrows)
}

// ---------------------------------------------------------------------------
// Encode one column's per-tile slab
// ---------------------------------------------------------------------------
//
// Input is the column's bytes for this tile, in native order
// (numpy's default).  Output is the compressed blob ready to land
// in the heap.  We don't byteswap to BE first — instead the
// per-algorithm encoder does it (RICE encodes from BE; GZIP_1 and
// GZIP_2 expect BE bytes too because that's what the read side
// reverses).  So caller passes `bytes_be: &[u8]` of length
// `n_pixels * elem_size`.
pub(crate) fn encode_table_column_slab(
    algo: CompressionAlgorithm,
    bytes_be: &[u8],
    n_pixels: usize,
    elem_size: usize,
    rice_blocksize: u32,
    gzip_level: Option<u32>,
) -> PyResult<Vec<u8>> {
    use crate::zimage::gzip::{encode_gzip1, encode_gzip2};
    use crate::zimage::rice::encode_rice;
    match algo {
        CompressionAlgorithm::Gzip1 => encode_gzip1(bytes_be, gzip_level),
        CompressionAlgorithm::Gzip2 => encode_gzip2(
            bytes_be, elem_size as u32, gzip_level,
        ),
        CompressionAlgorithm::Rice1 => encode_rice(
            bytes_be, n_pixels, elem_size as u32, rice_blocksize,
        ),
        _ => Err(PyValueError::new_err(format!(
            "internal: non-table algorithm reached encode_table_column_slab",
        ))),
    }
}

// Pull the gzip level (if set) and rice blocksize from a per-
// column config so the encoder gets the user's chosen params.
pub(crate) fn gzip_level_of(cfg: &CompressionConfigKind) -> Option<u32> {
    match cfg {
        CompressionConfigKind::Gzip1(g) => g.level,
        CompressionConfigKind::Gzip2(g) => g.level,
        _ => None,
    }
}

// Per-column setup shared between the bulk write and append paths.
// Holds everything the per-tile encode loop needs:
//   - the source ndarray's raw byte buffer (`buf`) + per-row stride
//     (`src_total_size`) + the per-cell `WriteTransform` derived from
//     the column's TFORM letter and the input dtype;
//   - encoder-side params (`elem_size` / `per_row_bytes` /
//     `per_row_pixels`) for the slab→blob call;
//   - per-column algorithm params (`rice_blocksize`, `gzip_level`)
//     pulled from the user's compression config.
//
// `contig_arr` pins the ndarray for the buf's lifetime; numpy could
// otherwise free the underlying buffer mid-encode.  Field is held,
// not read.
pub(crate) struct ColPrep<'py> {
    pub(crate) buf: RawBuffer,
    pub(crate) src_total_size: usize,
    pub(crate) transform: WriteTransform,
    #[allow(dead_code)]
    pub(crate) contig_arr: Bound<'py, PyAny>,
    pub(crate) elem_size: usize,
    pub(crate) per_row_bytes: usize,
    pub(crate) per_row_pixels: usize,
    pub(crate) rice_blocksize: u32,
    pub(crate) gzip_level: Option<u32>,
}

// Build a ColPrep from one input ndarray + column metadata + the
// per-column compression config (None when the HDU was reopened,
// in which case algorithm-level defaults apply).  Validates the
// input shape against the column's expected per-cell shape and
// derives the per-cell WriteTransform via the shared classifier;
// failures here surface before any file mutation.
pub(crate) fn prepare_fixed_column<'py>(
    np: &Bound<'py, PyAny>,
    ndarray: &Bound<'py, PyAny>,
    arr: &Bound<'py, PyAny>,
    col: &Column,
    nrows: usize,
    cfg: Option<&CompressionConfigKind>,
) -> PyResult<ColPrep<'py>> {
    if !arr.is_instance(ndarray)? {
        return Err(PyValueError::new_err(format!(
            "compressed table: column '{}' value must be a numpy ndarray",
            col.name)));
    }
    let arr_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if arr_shape.is_empty() || arr_shape[0] != nrows {
        return Err(PyValueError::new_err(format!(
            "compressed table: column '{}' shape {:?} does not have \
             first axis == {}", col.name, arr_shape, nrows)));
    }
    let per_cell_shape: Vec<usize> = arr_shape[1..].to_vec();
    let expected_shape = column_expected_shape(col);
    if per_cell_shape != expected_shape {
        return Err(PyValueError::new_err(format!(
            "compressed table: column '{}' per-cell shape {:?} does \
             not match expected {:?}",
            col.name, per_cell_shape, expected_shape)));
    }
    let dtype = arr.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let input_elem_size: usize = dtype.getattr("itemsize")?.extract()?;
    let transform = column_transform(col, &kind, input_elem_size)?;
    let cell_elements: usize = per_cell_shape.iter()
        .product::<usize>().max(1);
    let src_total_size = input_elem_size * cell_elements;
    let contig = np.call_method1("ascontiguousarray", (arr,))?;
    let buf = RawBuffer::acquire(&contig)?;
    let inner_elem_size = bytes_per_element(col.tform_letter)
        .ok_or_else(|| PyValueError::new_err(format!(
            "column '{}': unsupported TFORM letter '{}' on compressed \
             write", col.name, col.tform_letter)))?;
    Ok(ColPrep {
        buf, src_total_size, transform, contig_arr: contig,
        elem_size: inner_elem_size,
        per_row_bytes: col.byte_width,
        per_row_pixels: col.repeat,
        rice_blocksize: cfg.map(rice_blocksize_of).unwrap_or(32),
        gzip_level: cfg.and_then(gzip_level_of),
    })
}

// Take a pre-built FITS big-endian slab (one column × `n_pixels`
// elements), encode it per algorithm, write the compressed blob to
// the heap, and fill the descriptor table entry for this
// (tile_idx, col_idx).  Used by both the write path (after building
// the slab via per-cell transforms) and the append-merge path
// (slab is already in hand from decoded-old + new-transformed bytes).
// Returns the updated heap_cursor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_be_slab_to_heap_and_record(
    slab: &[u8],
    n_pixels: usize,
    algo: CompressionAlgorithm,
    elem_size: usize,
    rice_blocksize: u32,
    gzip_level: Option<u32>,
    tile_idx: usize,
    col_idx: usize,
    col_name: &str,
    descriptor_row_width: usize,
    heap_start_offset: u64,
    mut heap_cursor: u64,
    desc_table: &mut [u8],
    file: &FileHandle,
    layout: &Arc<FileLayout>,
    data_offset: u64,
    tainted: &TaintFlag,
) -> PyResult<u64> {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    let blob = encode_table_column_slab(
        algo, slab, n_pixels, elem_size, rice_blocksize, gzip_level,
    )?;
    let want_total =
        heap_start_offset + heap_cursor + blob.len() as u64 - data_offset;
    grow_file_to_at_least(file, layout, data_offset, want_total, tainted)?;
    {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&blob).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: heap write failed at \
                 tile {} col '{}': {}", tile_idx, col_name, e))
        })?;
    }
    let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
    let nelems_be = (blob.len() as i64).to_be_bytes();
    let off_be = (heap_cursor as i64).to_be_bytes();
    desc_table[desc_off..desc_off + 8].copy_from_slice(&nelems_be);
    desc_table[desc_off + 8..desc_off + 16].copy_from_slice(&off_be);
    heap_cursor += blob.len() as u64;
    Ok(heap_cursor)
}

// Build the per-tile per-column BE slab from `prep`'s native-order
// source bytes (applying the per-cell WriteTransform — byteswap,
// unsigned-int trick XOR, bool→ASCII, etc.) and hand off to
// `encode_be_slab_to_heap_and_record`.  Used by both write (with
// `source_row_offset = tile_row_start`) and append's new-tile branch
// (with `source_row_offset` past the merged rows).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_and_encode_tile_col(
    prep: &ColPrep,
    col: &Column,
    algo: CompressionAlgorithm,
    tile_idx: usize,
    col_idx: usize,
    rows_in_tile: usize,
    source_row_offset: usize,
    descriptor_row_width: usize,
    heap_start_offset: u64,
    heap_cursor: u64,
    desc_table: &mut [u8],
    file: &FileHandle,
    layout: &Arc<FileLayout>,
    data_offset: u64,
    tainted: &TaintFlag,
) -> PyResult<u64> {
    let src_bytes = prep.buf.as_slice();
    let mut slab = vec![0u8; rows_in_tile * prep.per_row_bytes];
    for r in 0..rows_in_tile {
        let src_row = source_row_offset + r;
        let src_off = src_row * prep.src_total_size;
        let src = &src_bytes
            [src_off..src_off + prep.src_total_size];
        let dst_off = r * prep.per_row_bytes;
        let dst = &mut slab[dst_off..dst_off + prep.per_row_bytes];
        apply_transform_cell(
            &prep.transform, src, dst, &col.name, src_row,
        )?;
    }
    let n_pixels = rows_in_tile * prep.per_row_pixels;
    encode_be_slab_to_heap_and_record(
        &slab, n_pixels, algo, prep.elem_size,
        prep.rice_blocksize, prep.gzip_level,
        tile_idx, col_idx, &col.name, descriptor_row_width,
        heap_start_offset, heap_cursor, desc_table,
        file, layout, data_offset, tainted,
    )
}

pub(crate) fn rice_blocksize_of(cfg: &CompressionConfigKind) -> u32 {
    match cfg {
        CompressionConfigKind::Rice1(r) => r.blocksize,
        _ => 32,
    }
}

// ---------------------------------------------------------------------------
// ZTABLE header construction
// ---------------------------------------------------------------------------
//
// Build the cards for a freshly-created compressed table.  Mirrors
// the cfitsio `fits_compress_table` header layout but produced
// directly from the user's structured dtype (the original
// uncompressed schema) — no copy from an existing BINTABLE.
//
// Result: a Vec<String> of cards ready to serialize, plus the
// computed (n_tiles, descriptor_row_width) the caller needs to
// reserve the data section.
pub(crate) fn build_compressed_table_header(
    cards_in: &[String],            // Pre-built uncompressed header
    row_width: u64,                 // From normalize_and_build_table_header
    nrows: i64,
    ztilelen: usize,
    algorithms: &[CompressionAlgorithm],
    columns: &[Column],
) -> PyResult<(Vec<String>, usize, u64)> {
    use crate::header::{card_int, card_logical, card_string, pad_to_card};

    let ncols = columns.len();
    let n_tiles_u = if nrows <= 0 {
        0usize
    } else {
        let n = nrows as usize;
        n.div_ceil(ztilelen.max(1))
    };
    // Each compressed-table row holds N descriptors; each is 1QB
    // (Q kind, 16 bytes).  Phase 5 only emits 1QB regardless of
    // input — Q-format heap supports arbitrarily large compressed
    // heaps and is what fpack always writes.
    let descriptor_row_width = ncols * 16;

    let mut out: Vec<String> = Vec::with_capacity(cards_in.len() + 16);

    // Structural keys: rewrite NAXIS1/NAXIS2/PCOUNT/TFIELDS into
    // the compressed shape, replace TFORMn with '1QB', drop the
    // input PCOUNT (we set it to 0 for now), and rewrite the
    // commentary lines so the user sees compressed-table semantics.
    for card in cards_in {
        if card.len() < 8 {
            out.push(card.clone());
            continue;
        }
        let kw = card[..8].trim_end();
        if kw == "NAXIS1" {
            out.push(card_int(
                "NAXIS1", descriptor_row_width as i64,
                "width of one compressed-table row in bytes"));
        } else if kw == "NAXIS2" {
            out.push(card_int(
                "NAXIS2", n_tiles_u as i64,
                "number of tiles"));
        } else if kw == "PCOUNT" {
            out.push(card_int(
                "PCOUNT", 0,
                "size of heap in bytes (filled on write)"));
        } else if let Some(suffix) = kw.strip_prefix("TFORM") {
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(n) = suffix.parse::<usize>() {
                    if n >= 1 && n <= ncols {
                        out.push(card_string(
                            &format!("TFORM{}", n), "1QB",
                            "compressed data descriptor"));
                        continue;
                    }
                }
            }
            out.push(card.clone());
        } else if kw == "END" {
            // Skip — we'll add the END after our Z-prefix cards.
        } else {
            out.push(card.clone());
        }
    }

    // ZTABLE / ZTILELEN / Z*-shape cards, ZFORM, ZCTYP.
    out.push(card_logical("ZTABLE", true, "this is a compressed table"));
    out.push(card_int(
        "ZTILELEN", ztilelen as i64, "number of rows in each tile"));
    out.push(card_int(
        "ZNAXIS1", row_width as i64,
        "original (uncompressed) row width in bytes"));
    out.push(card_int(
        "ZNAXIS2", nrows, "original (uncompressed) row count"));
    out.push(card_int(
        "ZPCOUNT", 0,
        "original heap size (0 for fixed-only tables)"));

    // ZFORMn (the original TFORMn).  Build from the Column list —
    // cfitsio copies via fits_read_card from the pre-compress
    // header, but constructing from the columns is equivalent and
    // doesn't require parsing the input cards twice.
    //   - Fixed columns: `<repeat><letter>` (e.g. '6E', '10A').
    //   - VLA columns: `1P<inner>` or `1Q<inner>` (e.g. '1PE',
    //     '1QJ').  parse_columns puts `tform_letter` = inner and
    //     `var_kind` = Some('P' | 'Q').
    for (i, col) in columns.iter().enumerate() {
        let n = i + 1;
        let tform = match col.var_kind {
            Some(desc) => format!("1{}{}", desc, col.tform_letter),
            None => format!("{}{}", col.repeat, col.tform_letter),
        };
        out.push(card_string(
            &format!("ZFORM{}", n), &tform,
            "original column TFORM"));
    }
    for (i, &algo) in algorithms.iter().enumerate() {
        let n = i + 1;
        let name = match algo {
            CompressionAlgorithm::Gzip1 => "GZIP_1",
            CompressionAlgorithm::Gzip2 => "GZIP_2",
            CompressionAlgorithm::Rice1 => "RICE_1",
            _ => return Err(PyValueError::new_err(format!(
                "internal: non-table algorithm in build_compressed_table_header"))),
        };
        out.push(card_string(
            &format!("ZCTYP{}", n), name,
            "compression algorithm for this column"));
    }
    out.push(pad_to_card("END"));

    let data_size = (n_tiles_u as u64).saturating_mul(descriptor_row_width as u64);
    Ok((out, n_tiles_u, data_size))
}

// ---------------------------------------------------------------------------
// Phase 5 — write loop
// ---------------------------------------------------------------------------
//
// Encode each column's per-tile slab, stream blob bytes to the
// heap, fill the descriptor table in RAM, then seek back and
// write the descriptor table + update PCOUNT + grow the file
// extent as needed.  Validate-then-mutate: any dtype/shape error
// surfaces before the file is touched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_compressed_table_data<'py>(
    py: Python<'py>,
    super_: &HDU,
    cards: &[String],
    per_column_inputs: &[Bound<'py, PyAny>],
    columns: &[Column],
    algorithms: &[CompressionAlgorithm],
    per_col_configs: Option<&[CompressionConfigKind]>,
    nrows: usize,
    ztilelen: usize,
    n_tiles: usize,
    descriptor_row_width: usize,
    data_offset: u64,
) -> PyResult<()> {
    use crate::hdu_image::round_up_to_block;
    use std::io::Write;
    use std::sync::atomic::Ordering;

    crate::common::check_not_tainted(&super_.tainted)?;

    if per_column_inputs.len() != columns.len() {
        return Err(PyValueError::new_err(format!(
            "internal: per-column inputs len {} != columns len {}",
            per_column_inputs.len(), columns.len())));
    }

    // Pre-plan the original heap layout for VLA columns: cfitsio's
    // funpack (and our Phase 4 read for that matter) puts the
    // *original* descriptors' offsets in the per-tile dual-
    // descriptor blob, and funpack uses them to place each cell at
    // its original-heap position when reconstructing the
    // uncompressed table.  If we leave them at 0 all cells of a
    // column collide at heap offset 0 on funpack output.  Match
    // the layout the uncompressed VLA write would produce
    // (`plan_vla_heap_layout`) so a funpack-decompressed file is
    // byte-equivalent to a fresh `create_table_hdu` + `write`
    // without compress.  The cursor return is the total original
    // heap size — emitted as ZPCOUNT below so funpack's
    // `fits_uncompress_table` (which copies ZPCOUNT → output
    // PCOUNT) gets the right heap extent.
    let np_for_plan = py.import("numpy")?;
    let ndarray_for_plan = np_for_plan.getattr("ndarray")?;
    let (vla_plans, original_pcount) = if columns.iter()
        .any(|c| c.var_kind.is_some())
    {
        let (plans, cursor) = plan_vla_heap_layout(
            columns, per_column_inputs, nrows, &ndarray_for_plan, 0,
        )?;
        (plans, cursor as u64)
    } else {
        (Vec::new(), 0u64)
    };

    // Per-column setup.  Fixed cols go through the shared
    // `prepare_fixed_column` helper; VLA cols are validated to be
    // Object-dtype ndarrays and their per-row cells are handled
    // lazily inside the tile loop via `encode_vla_column_tile`.
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let mut preps: Vec<Option<ColPrep<'_>>> =
        Vec::with_capacity(columns.len());
    for (i, (col, arr)) in columns.iter()
        .zip(per_column_inputs.iter()).enumerate()
    {
        if col.var_kind.is_some() {
            // VLA: validate Object-dtype ndarray with the right
            // length; deeper validation happens per cell.
            if !arr.is_instance(&ndarray)? {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.write: column '{}' value must \
                     be a numpy ndarray", col.name)));
            }
            let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
            if shape.is_empty() || shape[0] != nrows {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.write: column '{}' shape {:?} \
                     does not have first axis == ZNAXIS2 ({})",
                    col.name, shape, nrows)));
            }
            let dtype = arr.getattr("dtype")?;
            let kind: String = dtype.getattr("kind")?.extract()?;
            if kind != "O" {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.write: VLA column '{}' input \
                     must be a numpy Object dtype ndarray (kind 'O'), \
                     got kind '{}'", col.name, kind)));
            }
            preps.push(None);
            continue;
        }
        let cfg = per_col_configs.and_then(|cs| cs.get(i));
        preps.push(Some(prepare_fixed_column(
            &np, &ndarray, arr, col, nrows, cfg,
        )?));
    }

    // Stream-encode tile by tile, writing each blob to the heap as
    // it's produced.  Descriptor table is held in RAM (small:
    // n_tiles * ncols * 16 bytes; typically a few KB) and written
    // at the end with one seek-back.
    let mut desc_table: Vec<u8> = vec![0u8; n_tiles * descriptor_row_width];
    let heap_start_offset = data_offset
        + (n_tiles as u64 * descriptor_row_width as u64);
    let mut heap_cursor: u64 = 0;

    // Grow the file extent so we have room for the descriptor table
    // upfront.  The heap grows it further below.
    let current_padded = round_up_to_block(
        (n_tiles as u64) * (descriptor_row_width as u64));
    {
        // Allocate the initial descriptor space within this HDU.
        let mut guard = lock_file(&super_.file)?;
        let f = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let current_end = data_offset + current_padded;
        let file_len = f.metadata()
            .map_err(|e| PyIOError::new_err(e.to_string()))?.len();
        if file_len < current_end {
            f.set_len(current_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    for tile_idx in 0..n_tiles {
        let tile_row_start = tile_idx * ztilelen;
        let rows_in_tile = if tile_idx + 1 == n_tiles {
            nrows - tile_row_start
        } else {
            ztilelen
        };

        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_some() {
                // VLA dual-descriptor blob path.
                let cfg = per_col_configs.and_then(|cs| cs.get(col_idx));
                heap_cursor = encode_vla_column_tile(
                    py, &ndarray, &super_.file, &super_.layout,
                    &super_.tainted, data_offset, heap_start_offset,
                    heap_cursor, col, &per_column_inputs[col_idx],
                    &vla_plans[col_idx], tile_row_start, rows_in_tile,
                    algorithms[col_idx],
                    cfg.map(rice_blocksize_of).unwrap_or(32),
                    cfg.and_then(gzip_level_of),
                    &mut desc_table, tile_idx, col_idx,
                    descriptor_row_width,
                )?;
                continue;
            }
            // Fixed-column path: per-cell transform → encode → write
            // → record, all in the shared helper.
            let prep = preps[col_idx].as_ref()
                .expect("preps[i] is Some for non-VLA columns");
            heap_cursor = build_and_encode_tile_col(
                prep, col, algorithms[col_idx],
                tile_idx, col_idx, rows_in_tile,
                /* source_row_offset = */ tile_row_start,
                descriptor_row_width, heap_start_offset, heap_cursor,
                &mut desc_table, &super_.file, &super_.layout,
                data_offset, &super_.tainted,
            )?;
        }
    }

    // One round-up to the FITS block boundary so the data section
    // ends cleanly.  The grow helper already extends to multiples
    // of BLOCK_SIZE, but if heap_cursor isn't a multiple of
    // BLOCK_SIZE we need to make sure the tail is zero-filled.
    let total_data_bytes = (n_tiles as u64 * descriptor_row_width as u64)
        + heap_cursor;
    let padded = round_up_to_block(total_data_bytes);
    if padded > total_data_bytes {
        let pad = padded - total_data_bytes;
        let mut guard = lock_file(&super_.file)?;
        let f = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset + total_data_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&vec![0u8; pad as usize]).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: tail-pad write failed: {}", e))
        })?;
    }

    // Write the descriptor table at the start of the data section.
    {
        let mut guard = lock_file(&super_.file)?;
        let f = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&desc_table).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: descriptor-table write \
                 failed: {}", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: flush failed: {}", e))
        })?;
    }

    // Update PCOUNT (compressed heap size) AND ZPCOUNT (the
    // original-table heap size).  cfitsio's `fits_uncompress_table`
    // copies ZPCOUNT verbatim onto the output uncompressed PCOUNT
    // and uses it to set the heap extent; leaving it at 0 makes
    // funpack truncate the heap to zero even though the descriptors
    // point at real data.  For fixed-only tables ZPCOUNT stays 0
    // (no original heap to size).
    let mut cards_guard = super_.header.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards.to_vec();
    crate::hdu_table::set_pcount_in_cards(&mut new_cards, heap_cursor);
    set_zpcount_in_cards(&mut new_cards, original_pcount);
    crate::header::rewrite_header_to_disk(
        &super_.file, &super_.offsets, &super_.layout,
        &new_cards, &super_.tainted,
    )?;
    *cards_guard = new_cards;
    Ok(())
}

// Rewrite (or insert) the ZPCOUNT card to `new_value`.  ZPCOUNT
// records the ORIGINAL (uncompressed) table's PCOUNT — funpack
// copies it to the output PCOUNT during decompression.  Always
// present in headers we emit (created in build_compressed_table_header
// with the placeholder value 0); this update fills it in once the
// VLA write knows the real original-heap size.
fn set_zpcount_in_cards(new_cards: &mut Vec<String>, new_value: u64) {
    use crate::header::card_int;
    let card = card_int(
        "ZPCOUNT", new_value as i64,
        "original heap size (0 for fixed-only tables)",
    );
    let trimmed = card.trim_end().to_string();
    if let Some(idx) = new_cards.iter().position(|c|
        c.len() >= 7 && c[..7].trim() == "ZPCOUNT")
    {
        new_cards[idx] = trimmed;
    } else {
        // Defensive — every ZTABLE header we emit has ZPCOUNT.
        // Insert just before END if somehow missing.
        let end_idx = new_cards.iter().position(|c|
            c.len() >= 3 && c[..3].trim() == "END")
            .unwrap_or(new_cards.len());
        new_cards.insert(end_idx, trimmed);
    }
}

// Grow this HDU's data extent so it covers at least `min_bytes`
// (relative to data_offset).  Block-rounds; pushes later HDUs
// forward via the shared shift primitive when needed.  No-op when
// this HDU's data section already extends past `want_end`.
//
// For non-last HDUs the upper bound is the NEXT HDU's start, not
// the file length — file length includes trailing HDU bytes that
// belong to those HDUs and must not be overwritten.  Bare file-
// length as the cap (the original buggy form) silently passes
// writes that overlap a trailing HDU's region whenever the
// growth fits in the block-alignment padding, and corrupts the
// trailing HDU once growth exceeds the padding.
fn grow_file_to_at_least(
    file: &FileHandle,
    layout: &Arc<FileLayout>,
    data_offset: u64,
    min_bytes: u64,
    tainted: &TaintFlag,
) -> PyResult<()> {
    use crate::common::shift_file_tail_and_update_offsets;
    use crate::hdu_image::round_up_to_block;
    use std::sync::atomic::Ordering;

    let want_end = data_offset + round_up_to_block(min_bytes);
    let next_hdu_start = {
        let guard = layout.hdus.lock()
            .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
        guard.iter()
            .map(|o| o.header_offset())
            .filter(|&off| off > data_offset)
            .min()
    };
    let file_len = {
        let g = lock_file(file)?;
        let f = g.as_ref()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.metadata()
            .map_err(|e| PyIOError::new_err(e.to_string()))?.len()
    };
    // Effective end is bounded by the next HDU's start (non-last)
    // or by the file length (last HDU).
    let effective_end = next_hdu_start.unwrap_or(file_len);
    if want_end <= effective_end {
        return Ok(());
    }
    let delta = want_end - effective_end;
    if next_hdu_start.is_none() {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.set_len(want_end).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: set_len({}) failed: {}",
                want_end, e))
        })?;
    } else {
        shift_file_tail_and_update_offsets(
            file, layout, effective_end, delta, tainted,
        )?;
    }
    Ok(())
}

// Encode one VLA column for one tile.  Per row in the tile: compress
// the cell's BE bytes per ZCTYPn (with cfitsio's uncompressed
// fallback when compression doesn't shrink the data); record the
// (nelements_orig, original_offset_unused) descriptor in the first
// half of the dual-descriptor blob and the (cvlalen, cvlastart)
// compressed descriptor in the second half.  Then GZIP_1 the blob,
// write it to the heap, and fill the main-table 1QB descriptor for
// this (tile, col).
//
// Returns the updated `heap_cursor` (one heap slot ahead of where
// the last compressed cell bytes ended).
//
// Original descriptors use whatever P/Q kind the column's TFORMn
// declares (so reads via Phase 4 see the original layout); the
// compressed descriptors are always Q (16 bytes) per the cfitsio
// reference encoder.
#[allow(clippy::too_many_arguments)]
fn encode_vla_column_tile(
    py: Python<'_>,
    ndarray: &Bound<'_, PyAny>,
    file: &FileHandle,
    layout: &Arc<FileLayout>,
    tainted: &TaintFlag,
    data_offset: u64,
    heap_start_offset: u64,
    mut heap_cursor: u64,
    col: &Column,
    col_input: &Bound<'_, PyAny>,
    col_plans: &[VlaCellPlan],
    tile_row_start: usize,
    rows_in_tile: usize,
    algo: CompressionAlgorithm,
    rice_blocksize: u32,
    gzip_level: Option<u32>,
    desc_table: &mut [u8],
    tile_idx: usize,
    col_idx: usize,
    descriptor_row_width: usize,
) -> PyResult<u64> {
    use crate::zimage::gzip::encode_gzip1;
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let inner_letter = col.tform_letter;
    let elem_size = bytes_per_element(inner_letter)
        .ok_or_else(|| PyValueError::new_err(format!(
            "VLA column '{}': unsupported inner letter '{}'",
            col.name, inner_letter)))?;
    let descriptor_kind = col.var_kind
        .expect("encode_vla_column_tile called for non-VLA column");
    let width_orig = if descriptor_kind == 'P' { 8 } else { 16 };

    let blob_size = rows_in_tile * width_orig + rows_in_tile * 16;
    let mut descriptor_blob = vec![0u8; blob_size];
    let comp_desc_start = rows_in_tile * width_orig;

    // Per-row: validate cell, serialize to BE bytes, compress (with
    // uncompressed fallback), append to heap.
    for r in 0..rows_in_tile {
        let disk_row = tile_row_start + r;
        let cell = col_input.get_item(disk_row)?;
        let nelements = validate_vla_cell(
            &cell, ndarray, inner_letter, &col.name, disk_row)?;

        let mut cell_bytes_be = vec![0u8; nelements * elem_size];
        if nelements > 0 {
            serialize_vla_cell(
                &cell, inner_letter, nelements, &mut cell_bytes_be)?;
        }

        let (cvlalen, cvlastart) = if nelements == 0 {
            // Empty cell: no heap write, descriptors both (0, 0).
            (0u64, 0u64)
        } else {
            // Try compressing.  If the result isn't smaller than
            // the raw cell, fall back to writing the raw BE bytes
            // (cfitsio's `compressed_size < uncompressed_size`
            // check; see imcompress.c around line 8508).  Phase 4
            // read handles this fallback by detecting cvlalen ==
            // vlalen * elem_size.
            let compressed = encode_table_column_slab(
                algo, &cell_bytes_be, nelements, elem_size,
                rice_blocksize, gzip_level,
            )?;
            let payload = if compressed.len() >= cell_bytes_be.len() {
                &cell_bytes_be[..]
            } else {
                &compressed[..]
            };
            let plen = payload.len() as u64;
            let want_total =
                heap_start_offset + heap_cursor + plen - data_offset;
            grow_file_to_at_least(
                file, layout, data_offset, want_total, tainted)?;
            {
                let mut g = lock_file(file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.write_all(payload).map_err(|e| {
                    tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "compressed VLA write: cell heap write failed at \
                         tile {} col '{}' row {}: {}",
                        tile_idx, col.name, disk_row, e))
                })?;
            }
            let cvlastart = heap_cursor;
            heap_cursor += plen;
            (plen, cvlastart)
        };

        // Original descriptor (matches user-visible P or Q layout):
        // nelements_orig + planned_original_offset.  We use the
        // offset `plan_vla_heap_layout` would assign for a fresh
        // uncompressed write of the same data — funpack (cfitsio's
        // decompressor) uses this field to place the cell at the
        // right position in the reconstructed uncompressed heap, so
        // 0-for-all would collide every cell at offset 0.  Our own
        // Phase 4 read ignores this field (it only uses the
        // compressed descriptor's cvlalen/cvlastart) so the value
        // doesn't matter for rustfits round trips — but cross-tool
        // interop demands a consistent layout.
        let plan = &col_plans[tile_row_start + r];
        write_descriptor(
            descriptor_kind, nelements, plan.bytes_offset_in_heap,
            &mut descriptor_blob[r * width_orig..r * width_orig + width_orig],
        );
        // Compressed descriptor (always Q, 16 bytes).
        write_descriptor(
            'Q', cvlalen as usize, cvlastart as usize,
            &mut descriptor_blob[comp_desc_start + r * 16
                ..comp_desc_start + r * 16 + 16],
        );
    }
    let _ = py;  // py-handle no longer needed past validate_vla_cell

    // GZIP_1 the dual-descriptor blob — this is always GZIP_1
    // regardless of ZCTYPn (Phase 4 read decompresses via raw gzip).
    let gzipped = encode_gzip1(&descriptor_blob, None)?;

    let want_total =
        heap_start_offset + heap_cursor + gzipped.len() as u64 - data_offset;
    grow_file_to_at_least(file, layout, data_offset, want_total, tainted)?;
    let blob_heap_offset = heap_cursor;
    {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&gzipped).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed VLA write: descriptor-blob heap write failed \
                 at tile {} col '{}': {}", tile_idx, col.name, e))
        })?;
    }
    heap_cursor += gzipped.len() as u64;

    // Main-table descriptor for this (tile, col): the blob's size +
    // offset.  Two big-endian i64.
    let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
    let nelems_be = (gzipped.len() as i64).to_be_bytes();
    let off_be = (blob_heap_offset as i64).to_be_bytes();
    desc_table[desc_off..desc_off + 8].copy_from_slice(&nelems_be);
    desc_table[desc_off + 8..desc_off + 16].copy_from_slice(&off_be);

    Ok(heap_cursor)
}

// Pre-mutation snapshot of an existing tile's VLA dual-descriptor
// blob.  Decompressed eagerly so the original per-row descriptors
// (vlalen, original-heap offset, cvlalen, cvlastart) are available
// without re-touching the file after the heap relocates.
struct VlaMergeOldBlob {
    decompressed: Vec<u8>,
    width_orig: usize,
    rowspertile: usize,
}

// Read + decompress one (tile, col) dual-descriptor blob from the
// CURRENT (pre-mutation) heap.  Called only when merging rows into
// the existing last partial tile.
fn read_vla_merge_old_blob(
    file: &FileHandle,
    data_offset: u64,
    tile_idx: usize,
    col_idx: usize,
    col: &Column,
    rowspertile: usize,
    existing_n_tiles: usize,
    descriptor_row_width: usize,
) -> PyResult<VlaMergeOldBlob> {
    let width_orig = match col.var_kind {
        Some('P') => 8usize,
        Some('Q') => 16usize,
        _ => return Err(PyValueError::new_err(format!(
            "column '{}': expected P or Q var_kind, got {:?}",
            col.name, col.var_kind))),
    };
    let main_desc_off = data_offset
        + (tile_idx as u64) * (descriptor_row_width as u64)
        + (col_idx as u64) * 16;
    let mut main_desc = [0u8; 16];
    {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(main_desc_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut main_desc).map_err(|e| {
            PyIOError::new_err(format!(
                "append: read main desc for VLA tile {} col '{}': {}",
                tile_idx, col.name, e))
        })?;
    }
    let (blob_nelems_s, blob_off_s) = read_descriptor('Q', &main_desc);
    let blob_nelems = blob_nelems_s.max(0) as usize;
    let blob_heap_off = blob_off_s.max(0) as u64;
    let old_heap_start = data_offset
        + (existing_n_tiles as u64) * (descriptor_row_width as u64);
    let mut compressed = vec![0u8; blob_nelems];
    if blob_nelems > 0 {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(old_heap_start + blob_heap_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut compressed).map_err(|e| {
            PyIOError::new_err(format!(
                "append: read existing VLA dual-descriptor blob for \
                 tile {} col '{}': {}", tile_idx, col.name, e))
        })?;
    }
    let expected_blob_size = rowspertile * width_orig + rowspertile * 16;
    let decompressed = if blob_nelems > 0 {
        gzip_decompress_bytes(&compressed, expected_blob_size)?
    } else {
        Vec::new()
    };
    Ok(VlaMergeOldBlob { decompressed, width_orig, rowspertile })
}

// Re-encode the last existing tile of a VLA column for the merge
// path of append.  Existing rows: keep original descriptors verbatim
// (no decompress / re-compress), copy per-cell compressed bytes from
// their old heap position to the heap end, rewrite compressed
// descriptors with the new offset.  New rows: encode per-cell with
// the uncompressed-fallback contract, original-descriptor offset
// from the planner (extends past current ZPCOUNT).
#[allow(clippy::too_many_arguments)]
fn encode_vla_column_tile_with_merge(
    py: Python<'_>,
    ndarray: &Bound<'_, PyAny>,
    file: &FileHandle,
    layout: &Arc<FileLayout>,
    tainted: &TaintFlag,
    data_offset: u64,
    heap_start_offset: u64,
    mut heap_cursor: u64,
    col: &Column,
    col_input: &Bound<'_, PyAny>,
    col_plans: &[VlaCellPlan],
    tile_idx: usize,
    last_existing_tile_rows: usize,
    merge_rows: usize,
    old_blob: &VlaMergeOldBlob,
    algo: CompressionAlgorithm,
    rice_blocksize: u32,
    gzip_level: Option<u32>,
    desc_table: &mut [u8],
    col_idx: usize,
    descriptor_row_width: usize,
) -> PyResult<u64> {
    use crate::zimage::gzip::encode_gzip1;
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let inner_letter = col.tform_letter;
    let elem_size = bytes_per_element(inner_letter).ok_or_else(|| {
        PyValueError::new_err(format!(
            "VLA column '{}': unsupported inner letter '{}'",
            col.name, inner_letter))
    })?;
    let descriptor_kind = col.var_kind
        .expect("encode_vla_column_tile_with_merge called for non-VLA");
    let width_orig = if descriptor_kind == 'P' { 8 } else { 16 };
    let merged_rows = last_existing_tile_rows + merge_rows;
    let blob_size = merged_rows * width_orig + merged_rows * 16;
    let mut new_blob = vec![0u8; blob_size];
    let comp_desc_start = merged_rows * width_orig;
    let old_comp_desc_start =
        old_blob.rowspertile * old_blob.width_orig;

    // Chunk buffer for the in-file copy of existing per-cell bytes.
    // 1 MiB matches the rest of the file (~tight, peak-RSS-bounded).
    let mut copy_buf: Vec<u8> = Vec::new();
    let chunk_size: u64 = 1 << 20;

    // Existing rows: copy descriptors + per-cell compressed bytes.
    for r in 0..last_existing_tile_rows {
        let old_orig_off = r * width_orig;
        new_blob[r * width_orig..r * width_orig + width_orig]
            .copy_from_slice(
                &old_blob.decompressed
                    [old_orig_off..old_orig_off + width_orig]);

        let old_comp_off = old_comp_desc_start + r * 16;
        let (cvlalen_s, cvlastart_s) = read_descriptor(
            'Q', &old_blob.decompressed[old_comp_off..old_comp_off + 16]);
        let cvlalen = cvlalen_s.max(0) as u64;
        let cvlastart_old = cvlastart_s.max(0) as u64;

        let new_cvlastart = if cvlalen == 0 {
            0u64
        } else {
            let src_abs = heap_start_offset + cvlastart_old;
            let dst_abs = heap_start_offset + heap_cursor;
            // Source range [cvlastart_old, +cvlalen) lives in the
            // old heap, which ends at current_pcount (relative to
            // heap_start_offset).  heap_cursor starts at
            // current_pcount and only grows, so dst is always past
            // src — no overlap to worry about.
            let want_total = (dst_abs + cvlalen) - data_offset;
            grow_file_to_at_least(
                file, layout, data_offset, want_total, tainted)?;
            stream_copy_in_file(
                file, src_abs, dst_abs, cvlalen, &mut copy_buf,
                chunk_size, tainted,
                "compressed VLA append: copy existing cell bytes",
            )?;
            let placed = heap_cursor;
            heap_cursor += cvlalen;
            placed
        };
        let new_comp_off = comp_desc_start + r * 16;
        write_descriptor(
            'Q', cvlalen as usize, new_cvlastart as usize,
            &mut new_blob[new_comp_off..new_comp_off + 16],
        );
    }

    // New rows (first `merge_rows` of the input): encode per-cell,
    // original descriptor from the planner.
    for r in 0..merge_rows {
        let input_row_idx = r;
        let cell = col_input.get_item(input_row_idx)?;
        let nelements = validate_vla_cell(
            &cell, ndarray, inner_letter, &col.name, input_row_idx)?;
        let plan = &col_plans[input_row_idx];
        debug_assert_eq!(plan.nelements, nelements);

        let new_orig_off = (last_existing_tile_rows + r) * width_orig;
        write_descriptor(
            descriptor_kind, nelements, plan.bytes_offset_in_heap,
            &mut new_blob[new_orig_off..new_orig_off + width_orig],
        );

        let mut cell_be = vec![0u8; nelements * elem_size];
        if nelements > 0 {
            serialize_vla_cell(
                &cell, inner_letter, nelements, &mut cell_be)?;
        }
        let (cvlalen, cvlastart) = if nelements == 0 {
            (0u64, 0u64)
        } else {
            let compressed = encode_table_column_slab(
                algo, &cell_be, nelements, elem_size,
                rice_blocksize, gzip_level,
            )?;
            let payload = if compressed.len() >= cell_be.len() {
                &cell_be[..]
            } else {
                &compressed[..]
            };
            let plen = payload.len() as u64;
            let want_total = heap_start_offset + heap_cursor
                + plen - data_offset;
            grow_file_to_at_least(
                file, layout, data_offset, want_total, tainted)?;
            {
                let mut g = lock_file(file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.write_all(payload).map_err(|e| {
                    tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "compressed VLA append: merge-tile new-row cell \
                         write failed at col '{}' new-row {}: {}",
                        col.name, r, e))
                })?;
            }
            let placed = heap_cursor;
            heap_cursor += plen;
            (plen, placed)
        };
        let new_comp_off =
            comp_desc_start + (last_existing_tile_rows + r) * 16;
        write_descriptor(
            'Q', cvlalen as usize, cvlastart as usize,
            &mut new_blob[new_comp_off..new_comp_off + 16],
        );
    }
    let _ = py;

    // GZIP_1 the new dual-descriptor blob, write to heap end,
    // record the main-table descriptor for this (tile, col).
    let gzipped = encode_gzip1(&new_blob, None)?;
    let want_total = heap_start_offset + heap_cursor
        + gzipped.len() as u64 - data_offset;
    grow_file_to_at_least(file, layout, data_offset, want_total, tainted)?;
    let blob_heap_offset = heap_cursor;
    {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&gzipped).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed VLA append: merged dual-descriptor blob \
                 write failed at tile {} col '{}': {}",
                tile_idx, col.name, e))
        })?;
    }
    heap_cursor += gzipped.len() as u64;

    let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
    let nelems_be = (gzipped.len() as i64).to_be_bytes();
    let off_be = (blob_heap_offset as i64).to_be_bytes();
    desc_table[desc_off..desc_off + 8].copy_from_slice(&nelems_be);
    desc_table[desc_off + 8..desc_off + 16].copy_from_slice(&off_be);

    Ok(heap_cursor)
}

// ---------------------------------------------------------------------------
// Phase 6b — append rows to a compressed table
// ---------------------------------------------------------------------------
//
// Mechanics:
//   1. Compute layout: how many rows merge into the existing partial
//      last tile vs become fresh full tiles.
//   2. Grow the descriptor table by `added_n_tiles *
//      descriptor_row_width` bytes via a forward file-tail shift —
//      this also shifts the existing heap forward by the same delta.
//      Descriptor heap-offsets stay valid because they're heap-
//      relative.
//   3. Read the (now-larger) descriptor table into RAM so the merge
//      branch can rewrite the last-tile slot and the new-tile
//      branch can fill the just-freed slots.
//   4. If merging: decode the existing last tile's per-column BE
//      bytes (the Phase 2 read path with the final byteswap-to-
//      native skipped), concatenate the first M new rows (via the
//      shared per-cell transform), re-encode, append blobs to the
//      heap end.  Old last-tile blobs become orphans.
//   5. For remaining rows: encode as fresh tiles, write blobs to
//      heap, fill the new descriptor rows.
//   6. Write back the descriptor table.
//   7. Update header: NAXIS2 (n_tiles), PCOUNT (heap size),
//      ZNAXIS2 (original nrows).
//   8. Clear the tile cache — the last-tile entries are stale and
//      it's cheaper to drop everything than to do per-entry
//      invalidation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_compressed_table_data(
    py: Python<'_>,
    super_: &HDU,
    cards: &[String],
    per_column_inputs: &[Bound<'_, PyAny>],
    columns: &[Column],
    algorithms: &[CompressionAlgorithm],
    per_col_configs: Option<&[CompressionConfigKind]>,
    existing_nrows: usize,
    ztilelen: usize,
    existing_n_tiles: usize,
    descriptor_row_width: usize,
    data_offset: u64,
    current_pcount: u64,
    cache: &ColumnTileCache,
) -> PyResult<()> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    crate::common::check_not_tainted(&super_.tainted)?;

    // Determine append size from the first per-column ndarray's
    // shape[0].  All inputs must have matching first axis.
    if per_column_inputs.is_empty() {
        return Err(PyValueError::new_err(
            "CompressedTableHDU.append: no columns to write"));
    }
    let append_nrows: usize = per_column_inputs[0]
        .getattr("shape")?.extract::<Vec<usize>>()?
        .first().copied()
        .ok_or_else(|| PyValueError::new_err(
            "append: input shape is empty"))?;
    if append_nrows == 0 {
        return Ok(());
    }
    for (col, arr) in columns.iter().zip(per_column_inputs.iter()) {
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        if shape.is_empty() || shape[0] != append_nrows {
            return Err(PyValueError::new_err(format!(
                "CompressedTableHDU.append: column '{}' input shape \
                 {:?} does not have first axis == {}",
                col.name, shape, append_nrows)));
        }
    }

    let new_nrows = existing_nrows + append_nrows;
    let new_n_tiles = if new_nrows == 0 {
        0
    } else {
        new_nrows.div_ceil(ztilelen.max(1))
    };
    let added_n_tiles = new_n_tiles - existing_n_tiles;
    let last_existing_tile_rows = if existing_n_tiles == 0 {
        0
    } else {
        existing_nrows - (existing_n_tiles - 1) * ztilelen
    };
    let room_in_last_tile = if existing_n_tiles > 0
        && last_existing_tile_rows < ztilelen
    {
        ztilelen - last_existing_tile_rows
    } else {
        0
    };
    let merge_rows = append_nrows.min(room_in_last_tile);
    let _rows_in_new_tiles = append_nrows - merge_rows;

    // Per-column prep.  Fixed cols get a ColPrep; VLA cols get
    // a None slot — their per-cell work happens later in
    // encode_vla_column_tile{,_with_merge}.  VLA-input validation
    // (Object dtype, length match) happens here so dtype errors
    // raise before any file mutation.
    let np = py.import("numpy")?;
    let ndarray = np.getattr("ndarray")?;
    let mut preps: Vec<Option<ColPrep<'_>>> = Vec::with_capacity(columns.len());
    for (i, (col, arr)) in columns.iter()
        .zip(per_column_inputs.iter()).enumerate()
    {
        if col.var_kind.is_some() {
            if !arr.is_instance(&ndarray)? {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.append: column '{}' value must \
                     be a numpy ndarray", col.name)));
            }
            let kind: String = arr.getattr("dtype")?
                .getattr("kind")?.extract()?;
            if kind != "O" {
                return Err(PyValueError::new_err(format!(
                    "CompressedTableHDU.append: VLA column '{}' input \
                     must be a numpy Object dtype ndarray (kind 'O'), \
                     got kind '{}'", col.name, kind)));
            }
            preps.push(None);
            continue;
        }
        let cfg = per_col_configs.and_then(|cs| cs.get(i));
        preps.push(Some(prepare_fixed_column(
            &np, &ndarray, arr, col, append_nrows, cfg,
        )?));
    }

    let any_vla = columns.iter().any(|c| c.var_kind.is_some());
    let current_zpcount = parse_keyword(cards, "ZPCOUNT")
        .unwrap_or(0).max(0) as u64;

    // Plan VLA original-heap offsets for the full input batch, so
    // the original-descriptor offsets (used by funpack to
    // reconstruct) extend the existing original heap.  Returns
    // per-column per-row plans + the new total original-heap size.
    let (vla_plans, new_orig_pcount) = if any_vla {
        let (plans, cursor) = plan_vla_heap_layout(
            columns, per_column_inputs, append_nrows, &ndarray,
            current_zpcount as usize,
        )?;
        (plans, cursor as u64)
    } else {
        (Vec::new(), current_zpcount)
    };

    // Step 1: decode the existing last tile if we're going to
    // merge into it.  Do this BEFORE any file mutation — the
    // heap-offset math depends on the current layout.  Fixed cols
    // need their BE-bytes (decoded slab); VLA cols need their
    // dual-descriptor blob (decompressed) so we can copy the
    // existing per-row descriptors and per-cell compressed bytes.
    let mut existing_be_per_col: Vec<Vec<u8>> = Vec::new();
    let mut vla_merge_blobs: Vec<Option<VlaMergeOldBlob>> = Vec::new();
    if merge_rows > 0 {
        let last_tile_idx = existing_n_tiles - 1;
        existing_be_per_col.reserve(columns.len());
        vla_merge_blobs.reserve(columns.len());
        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_some() {
                existing_be_per_col.push(Vec::new());
                vla_merge_blobs.push(Some(read_vla_merge_old_blob(
                    &super_.file, data_offset, last_tile_idx, col_idx,
                    col, last_existing_tile_rows, existing_n_tiles,
                    descriptor_row_width,
                )?));
            } else {
                existing_be_per_col.push(decode_existing_tile_to_be_bytes(
                    &super_.file, cards, data_offset, last_tile_idx,
                    col_idx, col, algorithms[col_idx],
                    last_existing_tile_rows, descriptor_row_width,
                )?);
                vla_merge_blobs.push(None);
            }
        }
    }

    // Step 2: grow the data section to make room for the new
    // descriptor rows + existing heap.  grow_file_to_at_least
    // rounds to the next BLOCK_SIZE boundary and either set_len's
    // the file (last HDU) or shifts later HDUs forward (non-last);
    // either way the data section ends at a block boundary so
    // subsequent HDU headers stay aligned.  Further heap growth
    // during the encode loop is handled by additional
    // grow_file_to_at_least calls, also block-aligned.
    let delta_desc_bytes = (added_n_tiles as u64)
        * (descriptor_row_width as u64);
    let new_descs_bytes = (new_n_tiles as u64)
        * (descriptor_row_width as u64);
    let want_data_bytes = new_descs_bytes + current_pcount;
    grow_file_to_at_least(
        &super_.file, &super_.layout, data_offset,
        want_data_bytes, &super_.tainted,
    )?;

    // Step 2b: relocate the existing heap forward by
    // delta_desc_bytes (within the file) so it sits right after
    // the new (larger) descriptor table.  Descriptor heap-offsets
    // are heap-relative so they stay valid through the move.
    if delta_desc_bytes > 0 && current_pcount > 0 {
        let old_heap_start = data_offset
            + (existing_n_tiles as u64) * (descriptor_row_width as u64);
        let new_heap_start_local = data_offset + new_descs_bytes;
        relocate_region_forward_local(
            &super_.file, old_heap_start, new_heap_start_local,
            current_pcount, &super_.tainted,
        )?;
    }
    let new_heap_start = data_offset + new_descs_bytes;

    // Step 3: read the existing descriptor table into RAM so we
    // can modify the last-tile entries (merge case) and write new
    // descriptor rows for the appended tiles.  After the shift
    // above, the new descriptor table region is (existing_rows ||
    // zero-shifted-stale-bytes); we overwrite the zero-shifted
    // region with the new descriptors below.
    let desc_table_size = new_n_tiles * descriptor_row_width;
    let mut desc_table = vec![0u8; desc_table_size];
    if existing_n_tiles > 0 {
        let existing_desc_size =
            existing_n_tiles * descriptor_row_width;
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut desc_table[..existing_desc_size])
            .map_err(|e| PyIOError::new_err(format!(
                "append: read existing descriptor table failed: {}", e)))?;
    }

    // Heap cursor starts at current PCOUNT — we append new blobs
    // to the heap end (orphaning old last-tile blobs on merge).
    let mut heap_cursor = current_pcount;

    // Step 4: merge into last tile if applicable.  Fixed cols
    // concatenate freshly-transformed new rows onto the decoded
    // existing slab, then re-encode.  VLA cols copy each existing
    // row's per-cell compressed bytes verbatim (no decode / re-
    // encode) and append per-cell encoded bytes for new rows;
    // existing rows' original-descriptor offsets are preserved so
    // funpack's reconstructed heap stays consistent.
    if merge_rows > 0 {
        let last_tile_idx = existing_n_tiles - 1;
        let merged_rows = last_existing_tile_rows + merge_rows;
        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_some() {
                let cfg = per_col_configs.and_then(|cs| cs.get(col_idx));
                heap_cursor = encode_vla_column_tile_with_merge(
                    py, &ndarray, &super_.file, &super_.layout,
                    &super_.tainted, data_offset, new_heap_start,
                    heap_cursor, col, &per_column_inputs[col_idx],
                    &vla_plans[col_idx], last_tile_idx,
                    last_existing_tile_rows, merge_rows,
                    vla_merge_blobs[col_idx].as_ref().expect(
                        "VLA col has a merge-blob"),
                    algorithms[col_idx],
                    cfg.map(rice_blocksize_of).unwrap_or(32),
                    cfg.and_then(gzip_level_of),
                    &mut desc_table, col_idx, descriptor_row_width,
                )?;
                continue;
            }
            let prep = preps[col_idx].as_ref()
                .expect("non-VLA col has a ColPrep");
            let mut merged = existing_be_per_col[col_idx].clone();
            merged.reserve(merge_rows * prep.per_row_bytes);
            let src_bytes = prep.buf.as_slice();
            for r in 0..merge_rows {
                let src_off = r * prep.src_total_size;
                let src = &src_bytes
                    [src_off..src_off + prep.src_total_size];
                let mut cell = vec![0u8; prep.per_row_bytes];
                apply_transform_cell(
                    &prep.transform, src, &mut cell, &col.name, r)?;
                merged.extend_from_slice(&cell);
            }
            let n_pixels = merged_rows * prep.per_row_pixels;
            heap_cursor = encode_be_slab_to_heap_and_record(
                &merged, n_pixels, algorithms[col_idx],
                prep.elem_size, prep.rice_blocksize, prep.gzip_level,
                last_tile_idx, col_idx, &col.name, descriptor_row_width,
                new_heap_start, heap_cursor, &mut desc_table,
                &super_.file, &super_.layout, data_offset, &super_.tainted,
            )?;
        }
    }

    // Step 5: encode fresh tiles for any remaining rows.  Fixed
    // cols go through the shared helper; VLA cols reuse the
    // Phase 6a per-tile encoder with the planned original-heap
    // offsets (which extend past current_zpcount).
    let mut new_input_row_cursor = merge_rows;
    for new_tile_offset in 0..added_n_tiles {
        let tile_idx = existing_n_tiles + new_tile_offset;
        let tile_row_start_in_new = new_input_row_cursor;
        let rows_in_tile = if new_tile_offset + 1 == added_n_tiles {
            append_nrows - tile_row_start_in_new
        } else {
            ztilelen
        };
        for (col_idx, col) in columns.iter().enumerate() {
            if col.var_kind.is_some() {
                let cfg = per_col_configs.and_then(|cs| cs.get(col_idx));
                heap_cursor = encode_vla_column_tile(
                    py, &ndarray, &super_.file, &super_.layout,
                    &super_.tainted, data_offset, new_heap_start,
                    heap_cursor, col, &per_column_inputs[col_idx],
                    &vla_plans[col_idx], tile_row_start_in_new,
                    rows_in_tile, algorithms[col_idx],
                    cfg.map(rice_blocksize_of).unwrap_or(32),
                    cfg.and_then(gzip_level_of),
                    &mut desc_table, tile_idx, col_idx,
                    descriptor_row_width,
                )?;
                continue;
            }
            let prep = preps[col_idx].as_ref()
                .expect("non-VLA col has a ColPrep");
            heap_cursor = build_and_encode_tile_col(
                prep, col, algorithms[col_idx],
                tile_idx, col_idx, rows_in_tile,
                /* source_row_offset = */ tile_row_start_in_new,
                descriptor_row_width, new_heap_start, heap_cursor,
                &mut desc_table, &super_.file, &super_.layout,
                data_offset, &super_.tainted,
            )?;
        }
        new_input_row_cursor += rows_in_tile;
    }

    // grow_file_to_at_least keeps the data section block-aligned;
    // the bytes between heap_cursor and the block boundary are
    // either zero (last HDU, set_len from OS) or HDU 2 header
    // bytes that were shifted into place (non-last HDU).  Either
    // way, don't overwrite them.

    // Step 5: write the updated descriptor table.
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&desc_table).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "append: descriptor-table write failed: {}", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!("append: flush failed: {}", e))
        })?;
    }

    // Step 6: update header.  NAXIS2 = new_n_tiles, PCOUNT =
    // heap_cursor, ZNAXIS2 = new_nrows.  ZPCOUNT only matters
    // when any VLA col is present — it's the original
    // (uncompressed) heap size and funpack copies it onto the
    // output PCOUNT.  For fixed-only tables ZPCOUNT stays 0.
    let mut cards_guard = super_.header.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards.to_vec();
    update_int_card_in_place(
        &mut new_cards, "NAXIS2", new_n_tiles as i64,
        "number of tiles")?;
    update_int_card_in_place(
        &mut new_cards, "ZNAXIS2", new_nrows as i64,
        "original (uncompressed) row count")?;
    crate::hdu_table::set_pcount_in_cards(&mut new_cards, heap_cursor);
    if any_vla {
        set_zpcount_in_cards(&mut new_cards, new_orig_pcount);
    }
    crate::header::rewrite_header_to_disk(
        &super_.file, &super_.offsets, &super_.layout,
        &new_cards, &super_.tainted,
    )?;
    *cards_guard = new_cards;

    // Step 7: invalidate the cache.  The merged tile's entries
    // are stale; cheapest correct option is a full clear (cache
    // re-warms on the next read).  For append without merge the
    // existing entries stay valid, but we clear anyway to keep
    // the logic simple — append is a rare-vs-read operation.
    cache.clear();
    Ok(())
}

// Find and rewrite an int-valued structural card.  Used by append
// for NAXIS2 and ZNAXIS2 (PCOUNT goes through set_pcount_in_cards).
fn update_int_card_in_place(
    cards: &mut [String], keyword: &str, value: i64, comment: &str,
) -> PyResult<()> {
    use crate::header::card_int;
    let new_card = card_int(keyword, value, comment).trim_end().to_string();
    let kw_len = keyword.len();
    for card in cards.iter_mut() {
        if card.len() >= kw_len && card[..kw_len].trim_end() == keyword
            && (card.len() == kw_len
                || !card[kw_len..kw_len + 1].chars().next().unwrap().is_ascii_digit())
        {
            *card = new_card;
            return Ok(());
        }
    }
    Err(PyValueError::new_err(format!(
        "append: header missing required keyword {}", keyword)))
}

// Decode one (tile, col) blob back to FITS big-endian bytes —
// the slab format we'd hand to encode_table_column_slab.  Mirrors
// Phase 2's read path but stops before the byteswap-to-native that
// convert_column_cell does.
// Move `total` bytes WITHIN a file from `src_start` to `dst_start`,
// where `dst_start > src_start` (forward move).  Back-to-front
// chunked copy so the overlapping case is safe (later bytes read
// before they're overwritten by the move of earlier ones).  No
// layout offset updates — purely a within-file relocation, used by
// append to slide the existing heap forward inside the (already
// grown) data section to make room for new descriptor rows.
fn relocate_region_forward_local(
    file: &FileHandle,
    src_start: u64,
    dst_start: u64,
    total: u64,
    tainted: &TaintFlag,
) -> PyResult<()> {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    if total == 0 || src_start == dst_start {
        return Ok(());
    }
    let chunk_size: u64 = 1 << 20;  // 1 MiB
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
                "append: heap relocate read failed: {}; \
                 close + reopen", e))
        })?;
        f.seek(SeekFrom::Start(dst_off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "append: heap relocate write failed: {}; \
                 close + reopen", e)));
        }
        remaining -= n;
    }
    Ok(())
}

fn decode_existing_tile_to_be_bytes(
    file: &FileHandle,
    cards: &[String],
    data_offset: u64,
    tile_idx: usize,
    col_idx: usize,
    col: &Column,
    algo: CompressionAlgorithm,
    rows_in_tile: usize,
    descriptor_row_width: usize,
) -> PyResult<Vec<u8>> {
    // Heap base relative to data_offset.  Default = NAXIS1*NAXIS2.
    let theap_raw = parse_keyword(cards, "THEAP").unwrap_or(0);
    let heap_base_in_data = if theap_raw > 0 {
        theap_raw as u64
    } else {
        let n_tiles = parse_keyword(cards, "NAXIS2")
            .unwrap_or(0).max(0) as u64;
        let row_width = parse_keyword(cards, "NAXIS1")
            .unwrap_or(0).max(0) as u64;
        n_tiles * row_width
    };
    let heap_start = data_offset + heap_base_in_data;

    // Read descriptor at (tile_idx, col_idx).
    let mut desc_buf = [0u8; 16];
    {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let off = data_offset
            + (tile_idx as u64) * (descriptor_row_width as u64)
            + (col_idx as u64) * 16;
        f.seek(SeekFrom::Start(off))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut desc_buf).map_err(|e| {
            PyIOError::new_err(format!(
                "append/decode: read descriptor failed: {}", e))
        })?;
    }
    let (nelems_s, heap_offset_s) = read_descriptor('Q', &desc_buf);
    if nelems_s < 0 || heap_offset_s < 0 {
        return Err(PyValueError::new_err(format!(
            "append/decode: tile {} col '{}': descriptor has negative \
             field (nelements={}, offset={})",
            tile_idx, col.name, nelems_s, heap_offset_s)));
    }
    let n_bytes_compressed = nelems_s as usize;
    let mut compressed = vec![0u8; n_bytes_compressed];
    if n_bytes_compressed > 0 {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(heap_start + heap_offset_s as u64))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut compressed).map_err(|e| {
            PyIOError::new_err(format!(
                "append/decode: read heap failed: {}", e))
        })?;
    }
    decompress_column_slab(algo, &compressed, col, rows_in_tile)
}

// ---------------------------------------------------------------------------
// Phase 6c-1 — repack() on compressed tables (streaming)
// ---------------------------------------------------------------------------
//
// Walk descriptors in scan order ((tile, col) lex), compute each
// live blob's compact-heap position (= cumulative size of live
// blobs), then move bytes from old → new position with chunked
// I/O.  Two move strategies:
//
//   Fast path (in-place streaming): blobs read in old-offset
//   order, written to their new positions in place.  Requires
//   `new_offset[i] + length[i] <= old_offset[i+1]` for every
//   adjacent pair (so writes never clobber unread blobs).  This
//   holds for the post-merge orphan pattern that Phase 6b's
//   append produces (orphans contiguous before live tail).
//   Cost: ≤ `sum_of_live_blobs_that_move` bytes of I/O —
//   typically just the rewritten last tile (~10 MB).
//
//   Slow path (staging): blobs first copied to a "staging" area
//   appended to the file end (read source → write past current
//   heap end), then staged bytes copied back to their new
//   in-heap positions, then file shrunk.  Always safe for
//   arbitrary orphan patterns (writes go to fresh space; the
//   back-copy is front-to-back since dst < src by `new_pcount`).
//   Cost: ~`2 × new_pcount` bytes of I/O.  Used as a fallback
//   when the fast path's safety check fails — important for
//   future mutators (`__setitem__`) that create arbitrary
//   orphans.
//
// Memory bound: ~1 MiB chunk + the descriptor table (`n_tiles *
// ncols * 16` bytes; a few KB to a few MB) + the move-plan
// vector (~32 bytes per live blob).  No heap-in-RAM allocation.
pub(crate) fn repack_compressed_table_heap(
    super_: &HDU,
    cache: &ColumnTileCache,
) -> PyResult<()> {
    use crate::common::shift_file_tail_backward_and_update_offsets;
    use crate::hdu_image::{round_up_to_block, serialize_header_to_disk_bytes};
    use std::io::Write;
    use std::sync::atomic::Ordering;

    crate::common::check_not_tainted(&super_.tainted)?;

    let cards = super_.header_snapshot()?;
    let virtual_cards = synthesize_uncompressed_cards(&cards);
    let columns = parse_columns(&virtual_cards)?;

    let n_tiles = parse_keyword(&cards, "NAXIS2")
        .unwrap_or(0).max(0) as usize;
    let descriptor_row_width = parse_keyword(&cards, "NAXIS1")
        .unwrap_or(0).max(0) as usize;
    let current_pcount = parse_keyword(&cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    if n_tiles == 0 || columns.is_empty() || current_pcount == 0 {
        return Ok(());
    }
    let data_offset = super_.offsets.data_offset();

    // VLA columns add an indirection layer: dual-descriptor
    // blobs (themselves heap-stored) hold compressed-Q
    // descriptors pointing at per-cell compressed bytes (also
    // heap-stored).  Repack must rewrite both layers, then
    // re-GZIP the blobs.  Mixed tables (fixed + VLA cols) take
    // the VLA-aware path; pure-fixed tables use the streamlined
    // fixed-only path below.
    if columns.iter().any(|c| c.var_kind.is_some()) {
        return repack_compressed_table_heap_vla(
            super_, cache, &cards, &columns, n_tiles,
            descriptor_row_width, current_pcount, data_offset,
        );
    }

    let ncols = columns.len();
    let heap_start = data_offset
        + (n_tiles as u64) * (descriptor_row_width as u64);

    // Read just the descriptor table (small; bounded by n_tiles *
    // ncols * 16 bytes).
    let desc_table_size = n_tiles * descriptor_row_width;
    let mut desc_table = vec![0u8; desc_table_size];
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut desc_table).map_err(|e| {
            PyIOError::new_err(format!(
                "repack: read descriptor table: {}", e))
        })?;
    }

    // Build per-blob move plan in scan order: cumulative sum of
    // lengths gives each live blob its new offset.  Skip empty
    // cells (descriptor stays (0, 0)).
    struct MovePlan {
        old_offset: u64,
        length: u64,
        new_offset: u64,
        tile_idx: usize,
        col_idx: usize,
    }
    let mut plans: Vec<MovePlan> = Vec::new();
    let mut cursor: u64 = 0;
    for tile_idx in 0..n_tiles {
        for col_idx in 0..ncols {
            let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
            let nelems_s = i64::from_be_bytes(
                desc_table[desc_off..desc_off + 8].try_into().unwrap(),
            );
            let old_off_s = i64::from_be_bytes(
                desc_table[desc_off + 8..desc_off + 16].try_into().unwrap(),
            );
            if nelems_s < 0 || old_off_s < 0 {
                return Err(PyValueError::new_err(format!(
                    "repack: tile {} col {} descriptor negative: \
                     nelems={} offset={}",
                    tile_idx, col_idx, nelems_s, old_off_s)));
            }
            let length = nelems_s as u64;
            let old_offset = old_off_s as u64;
            if length == 0 {
                continue;
            }
            if old_offset.checked_add(length)
                .map(|e| e > current_pcount)
                .unwrap_or(true)
            {
                return Err(PyValueError::new_err(format!(
                    "repack: tile {} col {} descriptor points past \
                     heap end (offset+bytes={} > PCOUNT={})",
                    tile_idx, col_idx,
                    old_offset.wrapping_add(length), current_pcount)));
            }
            plans.push(MovePlan {
                old_offset, length, new_offset: cursor,
                tile_idx, col_idx,
            });
            cursor += length;
        }
    }
    let new_pcount = cursor;
    if new_pcount == current_pcount {
        return Ok(());  // Already compact.
    }

    // Sort by old_offset so the in-place fast path reads sequentially.
    plans.sort_by_key(|p| p.old_offset);

    // Decide fast vs slow path.  Fast path needs: for every
    // adjacent (i, i+1) pair in old-offset order, the i-th
    // blob's write region must end at or before the (i+1)-th
    // blob's read region (otherwise the write clobbers an
    // unread blob).  Holds for the post-merge orphan pattern.
    let mut fast_path_safe = true;
    for i in 0..plans.len() {
        let cur = &plans[i];
        let next_read_start = if i + 1 < plans.len() {
            plans[i + 1].old_offset
        } else {
            current_pcount
        };
        if cur.new_offset + cur.length > next_read_start {
            fast_path_safe = false;
            break;
        }
    }

    const CHUNK: u64 = 1 << 20;
    let mut buf = vec![0u8; CHUNK as usize];

    if fast_path_safe {
        // In-place streaming.  Reading in old-offset order means
        // every subsequent read is past any prior write, so no
        // clobbering.
        for plan in &plans {
            if plan.new_offset == plan.old_offset {
                continue;
            }
            stream_copy_in_file(
                &super_.file, heap_start + plan.old_offset,
                heap_start + plan.new_offset, plan.length,
                &mut buf, CHUNK, &super_.tainted,
                "repack: in-place move",
            )?;
        }
    } else {
        // Slow path — copy blobs to a staging area appended past
        // the current heap, then back-copy staging → final heap
        // positions.  Always safe regardless of orphan pattern.
        //
        // Step 1: grow data section so the staging area sits at
        // [heap_start + current_pcount, heap_start + current_pcount + new_pcount).
        // grow_file_to_at_least rounds to block-aligned, so the
        // actual staging area may extend a few hundred bytes
        // beyond — that's fine since we only read/write the
        // first new_pcount bytes.
        let staged_data_bytes = (n_tiles as u64
            * descriptor_row_width as u64)
            + current_pcount + new_pcount;
        grow_file_to_at_least(
            &super_.file, &super_.layout, data_offset,
            staged_data_bytes, &super_.tainted,
        )?;
        let staging_start = heap_start + current_pcount;

        // Step 2: copy each blob from its old position to its
        // staging position.  Staging is past the live heap, so
        // these writes never clobber any read.
        for plan in &plans {
            stream_copy_in_file(
                &super_.file, heap_start + plan.old_offset,
                staging_start + plan.new_offset, plan.length,
                &mut buf, CHUNK, &super_.tainted,
                "repack: copy to staging",
            )?;
        }

        // Step 3: copy staging back to the heap's final positions.
        // For each blob: dst = heap_start + new_offset, src =
        // staging_start + new_offset.  dst < src by current_pcount
        // (= the gap between heap and staging), so a front-to-back
        // chunked copy never clobbers an unread source byte.
        for plan in &plans {
            stream_copy_in_file(
                &super_.file, staging_start + plan.new_offset,
                heap_start + plan.new_offset, plan.length,
                &mut buf, CHUNK, &super_.tainted,
                "repack: copy from staging",
            )?;
        }
        // (Staging contents now stale; the file-shrink below
        // reclaims those bytes.)
    }

    // Rewrite descriptor entries with the new offsets.
    for plan in &plans {
        let desc_off = plan.tile_idx * descriptor_row_width
            + plan.col_idx * 16;
        desc_table[desc_off..desc_off + 8]
            .copy_from_slice(&(plan.length as i64).to_be_bytes());
        desc_table[desc_off + 8..desc_off + 16]
            .copy_from_slice(&(plan.new_offset as i64).to_be_bytes());
    }
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&desc_table) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: descriptor table rewrite: {}; close + reopen", e)));
        }
        if let Err(e) = f.flush() {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: flush: {}; close + reopen", e)));
        }
    }

    // Shrink file.  For the fast path, current_padded → new_padded.
    // For the slow path, the staging temporarily grew the file by
    // up to new_pcount; the shrink reclaims both the orphans AND
    // the staging.  Same computation either way: new HDU end is
    // at `data_offset + round_up_to_block(desc_bytes + new_pcount)`.
    let new_data_bytes = (n_tiles as u64
        * descriptor_row_width as u64) + new_pcount;
    let new_padded = round_up_to_block(new_data_bytes);
    let new_hdu_end = data_offset + new_padded;
    let file_len = {
        let g = lock_file(&super_.file)?;
        let f = g.as_ref()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.metadata().map_err(|e| PyIOError::new_err(e.to_string()))?.len()
    };
    if new_hdu_end < file_len {
        // Identify the next HDU on disk (if any) to decide
        // last-HDU (set_len) vs non-last (shift_file_tail_backward).
        let next_hdu_off: Option<u64> = {
            let guard = super_.layout.hdus.lock()
                .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
            guard.iter()
                .map(|o| o.header_offset())
                .filter(|&h| h > data_offset)
                .min()
        };
        match next_hdu_off {
            None => {
                // Last HDU — just trim the file.
                let mut g = lock_file(&super_.file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.set_len(new_hdu_end).map_err(|e| {
                    super_.tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "repack: set_len({}) failed: {}; close + reopen",
                        new_hdu_end, e))
                })?;
            }
            Some(next_off) => {
                // Non-last HDU.  After grow_file_to_at_least's
                // staging extension (slow path) or the post-merge
                // append's grow (fast path), the next HDU sits at
                // `next_off`.  Slide it (and everything after)
                // backward by `next_off - new_hdu_end` so the
                // current HDU's data section ends precisely at
                // `new_hdu_end` (block-aligned) and HDU N+1's
                // header lands at `new_hdu_end` itself.
                let delta = next_off - new_hdu_end;
                if delta > 0 {
                    shift_file_tail_backward_and_update_offsets(
                        &super_.file, &super_.layout,
                        next_off, delta, &super_.tainted)?;
                }
            }
        }
    }

    // Update PCOUNT — disk-write-before-commit pattern.
    let mut cards_guard = super_.header.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards.clone();
    crate::hdu_table::set_pcount_in_cards(&mut new_cards, new_pcount);
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = data_offset - header_bytes.len() as u64;
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack: PCOUNT header write: {}; close + reopen", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack: PCOUNT header flush: {}; close + reopen", e))
        })?;
    }
    *cards_guard = new_cards;
    cache.clear();
    Ok(())
}

// VLA-aware repack.  Walks every (tile, col) in scan order:
//
//   - Fixed col: stream-copy the existing column blob to the
//     staging area; record new (offset, length) for the main
//     descriptor.
//   - VLA col: read + decompress the dual-descriptor blob; walk
//     each row, stream-copy live per-cell compressed bytes from
//     their old heap position to staging, rewrite cvlastart in
//     the in-RAM blob; re-GZIP the blob; write the freshly
//     gzipped blob to staging; record new (offset, length) for
//     the main descriptor.
//
// Staging area sits past the current heap (writes never clobber
// reads from the old heap).  After all (tile, col) processed,
// one big front-to-back stream copy moves staging[0..new_pcount]
// to heap_start[0..new_pcount] — safe because dst < src by at
// least `current_pcount` so chunks read past the cursor stay
// untouched.  Then file shrink + descriptor rewrite + PCOUNT
// update + cache clear, mirroring the fixed-only path's tail.
//
// Memory bound: ~1 MiB chunk + one decompressed dual-desc blob
// at a time (`rowspertile * (width_orig + 16)` bytes) + one
// gzipped blob held briefly while staging it + the descriptor
// table + the per-(tile, col) move-plan vector (~32 bytes per
// entry).  No heap-in-RAM allocation.  Staging temporarily
// roughly doubles the file's heap region; reclaimed on shrink.
//
// ZPCOUNT is the ORIGINAL (uncompressed) heap size, invariant
// under repack (we don't change which cells exist or their
// nelements, just where their compressed bytes live).  Don't
// touch it.
#[allow(clippy::too_many_arguments)]
fn repack_compressed_table_heap_vla(
    super_: &HDU,
    cache: &ColumnTileCache,
    cards: &[String],
    columns: &[Column],
    n_tiles: usize,
    descriptor_row_width: usize,
    current_pcount: u64,
    data_offset: u64,
) -> PyResult<()> {
    use crate::common::shift_file_tail_backward_and_update_offsets;
    use crate::hdu_image::{round_up_to_block, serialize_header_to_disk_bytes};
    use crate::zimage::gzip::encode_gzip1;
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let ncols = columns.len();
    let heap_start = data_offset
        + (n_tiles as u64) * (descriptor_row_width as u64);
    let ztilelen = parse_keyword(cards, "ZTILELEN")
        .unwrap_or(0).max(0) as usize;
    let total_nrows = parse_keyword(cards, "ZNAXIS2")
        .unwrap_or(0).max(0) as usize;

    // Read descriptor table — small (n_tiles × ncols × 16 bytes).
    let desc_table_size = n_tiles * descriptor_row_width;
    let mut desc_table = vec![0u8; desc_table_size];
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut desc_table).map_err(|e| {
            PyIOError::new_err(format!(
                "repack-vla: read descriptor table: {}", e))
        })?;
    }

    // Move plan: for each (tile, col), the new (offset, length)
    // of its main-table descriptor pointing into the compacted
    // heap.  (0, 0) for any (tile, col) whose source descriptor
    // was already empty — preserves the encode_vla_column_tile
    // convention.
    let mut new_main_descs: Vec<(usize, usize, u64, u64)> =
        Vec::with_capacity(n_tiles * ncols);

    let staging_start = heap_start + current_pcount;
    let mut staging_cursor: u64 = 0;
    const CHUNK: u64 = 1 << 20;
    let mut copy_buf: Vec<u8> = Vec::new();
    // grow_file_to_at_least wants bytes-after-data_offset.  The
    // staging area starts at staging_start = data_offset +
    // desc_bytes + current_pcount, so each write of `n` bytes
    // through cursor `c` reaches up to (desc_bytes + current_pcount
    // + c + n) bytes past data_offset.  Forgetting desc_bytes here
    // under-shifts the trailing HDU by `desc_bytes` (= 64 bytes for
    // a typical multi-col 1QB descriptor row), enough for staging
    // writes to clobber the start of HDU N+1's header.
    let desc_bytes_u64 = (n_tiles * descriptor_row_width) as u64;

    for tile_idx in 0..n_tiles {
        for col_idx in 0..ncols {
            let col = &columns[col_idx];
            let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
            let nelems_s = i64::from_be_bytes(
                desc_table[desc_off..desc_off + 8].try_into().unwrap(),
            );
            let old_off_s = i64::from_be_bytes(
                desc_table[desc_off + 8..desc_off + 16].try_into().unwrap(),
            );
            if nelems_s <= 0 {
                new_main_descs.push((tile_idx, col_idx, 0, 0));
                continue;
            }
            let old_length = nelems_s as u64;
            let old_offset = old_off_s.max(0) as u64;
            if old_offset.checked_add(old_length)
                .map(|e| e > current_pcount)
                .unwrap_or(true)
            {
                return Err(PyValueError::new_err(format!(
                    "repack-vla: tile {} col '{}' main descriptor \
                     points past heap end (offset+bytes={} > PCOUNT={})",
                    tile_idx, col.name,
                    old_offset.wrapping_add(old_length), current_pcount)));
            }

            if col.var_kind.is_none() {
                // Fixed col blob: stream-copy old → staging at
                // current cursor.
                let want_total = desc_bytes_u64
                    + current_pcount + staging_cursor + old_length;
                grow_file_to_at_least(
                    &super_.file, &super_.layout, data_offset,
                    want_total, &super_.tainted,
                )?;
                stream_copy_in_file(
                    &super_.file, heap_start + old_offset,
                    staging_start + staging_cursor, old_length,
                    &mut copy_buf, CHUNK, &super_.tainted,
                    "repack-vla: stage fixed col blob",
                )?;
                new_main_descs.push((tile_idx, col_idx,
                    staging_cursor, old_length));
                staging_cursor += old_length;
                continue;
            }

            // VLA column path.  Read + decompress the existing
            // dual-descriptor blob from the heap (NOT staging —
            // staging is past current_pcount, sources live in
            // [0, current_pcount)).
            let width_orig = match col.var_kind {
                Some('P') => 8usize,
                Some('Q') => 16usize,
                _ => return Err(PyValueError::new_err(format!(
                    "column '{}': expected P or Q var_kind",
                    col.name))),
            };
            let rowspertile = if tile_idx + 1 == n_tiles {
                total_nrows.saturating_sub(tile_idx * ztilelen)
            } else {
                ztilelen
            };
            let expected_blob_size =
                rowspertile * width_orig + rowspertile * 16;
            let mut compressed_old = vec![0u8; old_length as usize];
            {
                let mut g = lock_file(&super_.file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(heap_start + old_offset))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.read_exact(&mut compressed_old).map_err(|e| {
                    PyIOError::new_err(format!(
                        "repack-vla: read old dual-descriptor blob for \
                         tile {} col '{}': {}",
                        tile_idx, col.name, e))
                })?;
            }
            let mut blob =
                gzip_decompress_bytes(&compressed_old, expected_blob_size)?;
            let comp_desc_start = rowspertile * width_orig;

            // Per-row: stream-copy live cell bytes to staging,
            // rewrite cvlastart in the in-RAM blob.  Empty cells
            // (cvlalen == 0) keep descriptor (0, 0).
            for r in 0..rowspertile {
                let comp_off = comp_desc_start + r * 16;
                let (cvlalen_s, cvlastart_old_s) = read_descriptor(
                    'Q', &blob[comp_off..comp_off + 16]);
                let cvlalen = cvlalen_s.max(0) as u64;
                let cvlastart_old = cvlastart_old_s.max(0) as u64;
                if cvlalen == 0 {
                    write_descriptor(
                        'Q', 0, 0,
                        &mut blob[comp_off..comp_off + 16],
                    );
                    continue;
                }
                if cvlastart_old.checked_add(cvlalen)
                    .map(|e| e > current_pcount)
                    .unwrap_or(true)
                {
                    return Err(PyValueError::new_err(format!(
                        "repack-vla: tile {} col '{}' row {} cell \
                         descriptor points past heap end \
                         (cvlastart+cvlalen={} > PCOUNT={})",
                        tile_idx, col.name, r,
                        cvlastart_old.wrapping_add(cvlalen),
                        current_pcount)));
                }
                let cvlastart_new = staging_cursor;
                let want_total = desc_bytes_u64
                    + current_pcount + staging_cursor + cvlalen;
                grow_file_to_at_least(
                    &super_.file, &super_.layout, data_offset,
                    want_total, &super_.tainted,
                )?;
                stream_copy_in_file(
                    &super_.file, heap_start + cvlastart_old,
                    staging_start + cvlastart_new, cvlalen,
                    &mut copy_buf, CHUNK, &super_.tainted,
                    "repack-vla: stage VLA cell bytes",
                )?;
                staging_cursor += cvlalen;
                write_descriptor(
                    'Q', cvlalen as usize, cvlastart_new as usize,
                    &mut blob[comp_off..comp_off + 16],
                );
            }

            // Re-GZIP the blob (compressed descriptors now point
            // at the staging-area cvlastart values).  Write to
            // staging at the current cursor.
            let gzipped = encode_gzip1(&blob, None)?;
            let blob_new_offset = staging_cursor;
            let blob_new_length = gzipped.len() as u64;
            let want_total = desc_bytes_u64
                + current_pcount + staging_cursor + blob_new_length;
            grow_file_to_at_least(
                &super_.file, &super_.layout, data_offset,
                want_total, &super_.tainted,
            )?;
            {
                let mut g = lock_file(&super_.file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.seek(SeekFrom::Start(staging_start + staging_cursor))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                f.write_all(&gzipped).map_err(|e| {
                    super_.tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "repack-vla: stage dual-descriptor blob \
                         write for tile {} col '{}': {}; close + reopen",
                        tile_idx, col.name, e))
                })?;
            }
            staging_cursor += blob_new_length;
            new_main_descs.push((tile_idx, col_idx,
                blob_new_offset, blob_new_length));
        }
    }

    let new_pcount = staging_cursor;

    // Back-copy staging[0..new_pcount] → heap[0..new_pcount] in
    // ONE chunked copy.  Front-to-back is safe: dst = heap_start
    // + k, src = heap_start + current_pcount + k, so dst < src
    // by current_pcount > 0 for every k.
    if new_pcount > 0 {
        stream_copy_in_file(
            &super_.file, staging_start, heap_start, new_pcount,
            &mut copy_buf, CHUNK, &super_.tainted,
            "repack-vla: copy staging to heap",
        )?;
    }

    // Rewrite descriptor table with new (offset, length).
    for (tile_idx, col_idx, new_off, length) in &new_main_descs {
        let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
        desc_table[desc_off..desc_off + 8]
            .copy_from_slice(&(*length as i64).to_be_bytes());
        desc_table[desc_off + 8..desc_off + 16]
            .copy_from_slice(&(*new_off as i64).to_be_bytes());
    }
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&desc_table) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack-vla: descriptor table rewrite: {}; \
                 close + reopen", e)));
        }
        if let Err(e) = f.flush() {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack-vla: flush: {}; close + reopen", e)));
        }
    }

    // Shrink file (reclaims staging + orphans together).
    let new_data_bytes = (n_tiles as u64
        * descriptor_row_width as u64) + new_pcount;
    let new_padded = round_up_to_block(new_data_bytes);
    let new_hdu_end = data_offset + new_padded;
    let file_len = {
        let g = lock_file(&super_.file)?;
        let f = g.as_ref()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.metadata().map_err(|e| PyIOError::new_err(e.to_string()))?.len()
    };
    if new_hdu_end < file_len {
        let next_hdu_off: Option<u64> = {
            let guard = super_.layout.hdus.lock()
                .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
            guard.iter()
                .map(|o| o.header_offset())
                .filter(|&h| h > data_offset)
                .min()
        };
        match next_hdu_off {
            None => {
                let mut g = lock_file(&super_.file)?;
                let f = g.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.set_len(new_hdu_end).map_err(|e| {
                    super_.tainted.store(true, Ordering::Release);
                    PyIOError::new_err(format!(
                        "repack-vla: set_len({}) failed: {}; \
                         close + reopen", new_hdu_end, e))
                })?;
            }
            Some(next_off) => {
                let delta = next_off - new_hdu_end;
                if delta > 0 {
                    shift_file_tail_backward_and_update_offsets(
                        &super_.file, &super_.layout,
                        next_off, delta, &super_.tainted)?;
                }
            }
        }
    }

    // Update PCOUNT (ZPCOUNT stays unchanged — original-heap
    // size is invariant under repack).
    let mut cards_guard = super_.header.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards.to_vec();
    crate::hdu_table::set_pcount_in_cards(&mut new_cards, new_pcount);
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = data_offset - header_bytes.len() as u64;
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack-vla: PCOUNT header write: {}; \
                 close + reopen", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack-vla: PCOUNT header flush: {}; \
                 close + reopen", e))
        })?;
    }
    *cards_guard = new_cards;
    cache.clear();
    Ok(())
}

// Chunked read-then-write copy of `length` bytes from `src_off` to
// `dst_off` in `file`.  Used by repack's in-place fast path AND the
// staging slow path.  Caller is responsible for ensuring the move
// is safe (writes don't clobber unread regions).
#[allow(clippy::too_many_arguments)]
fn stream_copy_in_file(
    file: &FileHandle,
    src_off: u64,
    dst_off: u64,
    length: u64,
    buf: &mut Vec<u8>,
    chunk: u64,
    tainted: &TaintFlag,
    op_label: &str,
) -> PyResult<()> {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    let mut processed: u64 = 0;
    while processed < length {
        let n = (length - processed).min(chunk);
        buf.resize(n as usize, 0);
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(src_off + processed))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "{}: read at offset {}: {}; close + reopen",
                op_label, src_off + processed, e))
        })?;
        f.seek(SeekFrom::Start(dst_off + processed))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(buf) {
            tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "{}: write at offset {}: {}; close + reopen",
                op_label, dst_off + processed, e)));
        }
        processed += n;
    }
    Ok(())
}


