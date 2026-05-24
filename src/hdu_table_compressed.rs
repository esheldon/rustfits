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

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::Arc;

use crate::common::{
    parse_keyword, parse_string_keyword, FileHandle, FileLayout, HduOffsets,
    TaintFlag,
};
use crate::hdu::HDU;
use crate::hdu_table::{
    build_numpy_dtype, field_dtype_and_shape, parse_columns, Column, TableHDU,
};

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
pub(crate) struct CompressedTableHDU;

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
            .add_subclass(CompressedTableHDU)
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

    #[pyo3(signature = (*, rows=None, columns=None, scale=true, mask_null=false))]
    fn read(
        _slf: PyRef<'_, Self>,
        _py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        columns: Option<&Bound<'_, PyAny>>,
        scale: bool,
        mask_null: bool,
    ) -> PyResult<()> {
        let _ = (rows, columns, scale, mask_null);
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.read() — ZTABLE Phase 2 will add this \
             (whole-table read across GZIP_1, GZIP_2, RICE_1)"))
    }

    fn __getitem__(
        _slf: PyRef<'_, Self>,
        _key: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "CompressedTableHDU.__getitem__ — ZTABLE Phase 2/3 will add \
             this (whole-table read in Phase 2, slicing in Phase 3)"))
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
