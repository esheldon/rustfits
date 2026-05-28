// CompressedTableMeta cache struct + parser (parses the original
// uncompressed-schema view used by the meta cache).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::{
    parse_keyword, parse_string_keyword,
};
use crate::hdu_table::{
    parse_columns, Column,
};
use crate::zimage::{parse_algorithm, CompressionAlgorithm};

use super::hdu::synthesize_uncompressed_cards;

// Snapshot of the compressed-table metadata needed by every
// __setitem__ dispatch path (main HDU + the subset pyclasses)
// AND by every accessor + I/O entry point (read, write, append,
// repack, ...).  Cached per-HDU keyed by `cards_version` (see
// `CompressedTableHDU::meta()`), so a hot inner loop of accessor
// calls pays one Mutex lock + integer compare + Arc clone instead
// of re-parsing the synthesized cards every time.
//
// Notably absent: `data_offset`.  It's *not* a header-derived
// value — it can change when an earlier HDU grows (the shared
// Arc<HduOffsets> takes care of the propagation, but caching the
// old value here would defeat that).  Callers fetch it fresh
// from `super_.offsets.data_offset()` alongside the meta.
pub(crate) struct CompressedTableMeta {
    pub(crate) cards: Vec<String>,
    pub(crate) columns: Vec<Column>,
    pub(crate) algorithms: Vec<CompressionAlgorithm>,
    pub(crate) nrows: usize,
    pub(crate) ztilelen: usize,
    pub(crate) n_tiles: usize,
    pub(crate) descriptor_row_width: usize,
    pub(crate) current_pcount: u64,
}

// Parse all of the above from the cards Vec.  Same shape as
// `parse_table_meta` and `parse_compressed_image_meta` — a
// pure function the meta accessor calls on cache miss.
pub(crate) fn parse_compressed_table_meta(
    cards: Vec<String>,
) -> PyResult<CompressedTableMeta> {
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
    let current_pcount = parse_keyword(&cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    let mut algorithms: Vec<CompressionAlgorithm> =
        Vec::with_capacity(columns.len());
    for i in 0..columns.len() {
        let key = format!("ZCTYP{}", i + 1);
        let zctyp = parse_string_keyword(&cards, &key)
            .ok_or_else(|| PyValueError::new_err(format!(
                "compressed table missing {} card", key)))?;
        algorithms.push(parse_algorithm(&zctyp)?);
    }
    Ok(CompressedTableMeta {
        cards, columns, algorithms, nrows, ztilelen, n_tiles,
        descriptor_row_width, current_pcount,
    })
}

