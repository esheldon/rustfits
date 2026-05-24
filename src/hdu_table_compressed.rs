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
use std::sync::Arc;

use crate::cache::BytesBoundLruCache;
use crate::common::{
    byteswap_in_place, lock_file, parse_keyword, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, RawBuffer, TaintFlag,
};
use crate::hdu::HDU;
use crate::hdu_table::{
    build_numpy_dtype, build_var_cell_value, bytes_per_element,
    byteswap_unit, classify_table_key, convert_column_cell,
    field_dtype_and_shape, numpy_field_layout, parse_columns,
    read_descriptor, resolve_columns, resolve_rows, scaling_kind, Column,
    ScalingKind, TableHDU, TableKey,
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
struct CacheKey(u32, u32);

// Per-(tile, col) decompressed-bytes cache.  See
// `crate::cache::BytesBoundLruCache` for the eviction policy and
// concurrency model.
type ColumnTileCache = BytesBoundLruCache<CacheKey>;

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

    fn write(
        _slf: PyRef<'_, Self>,
        _data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.write() — ZTABLE Phase 5 will add this \
             (bulk write with create_table_hdu(..., compress=...))"))
    }

    fn append(
        _slf: PyRef<'_, Self>,
        _data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.append() — ZTABLE Phase 6 will add this"))
    }

    fn extend(
        _slf: PyRef<'_, Self>,
        _data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.extend() — ZTABLE Phase 6 will add this"))
    }

    fn repack(_slf: PyRef<'_, Self>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.repack() — ZTABLE Phase 6 will add this"))
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

