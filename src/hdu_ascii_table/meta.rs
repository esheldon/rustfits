// Per-HDU parsed metadata for an ASCII table, cached on the
// AsciiTableHDU and re-derived from the cards Vec only when the
// `cards_version` has bumped since the last parse.  Same shape as
// `TableMeta` in `hdu_table/columns.rs`.

use pyo3::prelude::*;

use crate::common::parse_keyword;

use super::columns::{parse_ascii_columns, AsciiColumn};

// Parsed-once snapshot of all per-HDU ASCII-table metadata.
// `nrows` = NAXIS2, `row_width` = NAXIS1.  ASCII tables have no
// heap (PCOUNT=0 always), so there is no THEAP equivalent.
pub(crate) struct AsciiTableMeta {
    pub(crate) nrows: u64,
    pub(crate) row_width: u64,
    pub(crate) columns: Vec<AsciiColumn>,
}

pub(crate) fn parse_ascii_table_meta(
    cards: &[String],
) -> PyResult<AsciiTableMeta> {
    let columns = parse_ascii_columns(cards)?;
    let nrows = parse_keyword(cards, "NAXIS2").unwrap_or(0).max(0) as u64;
    let row_width =
        parse_keyword(cards, "NAXIS1").unwrap_or(0).max(0) as u64;
    Ok(AsciiTableMeta { nrows, row_width, columns })
}
