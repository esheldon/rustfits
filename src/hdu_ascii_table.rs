// AsciiTableHDU: TABLE extension HDU (ASCII table).  Read/write API is not
// yet implemented; the pyclass exists so ASCII tables in input files are
// surfaced as their own type rather than as a bare HDU.

use pyo3::prelude::*;
use std::sync::Arc;

use crate::common::{
    parse_keyword, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, TaintFlag,
};
use crate::hdu::HDU;

#[pyclass(extends = HDU)]
pub(crate) struct AsciiTableHDU;

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
            AsciiTableHDU,
            HDU::new(header, index, filename, offsets, layout, file, tainted),
        )
    }
}

#[pymethods]
impl AsciiTableHDU {
    // Multi-line, fitsio-style repr.  Column info is intentionally
    // omitted: ASCII tables aren't yet parsed into a typed Vec<Column>
    // (the pyclass exists so input files with ASCII tables surface as
    // their own type, but read/write isn't implemented yet).  Add
    // column listing when the read API lands.
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let nrows = parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0);
        let extname = parse_string_keyword(&cards, "EXTNAME");

        let mut out = String::new();
        out.push_str(&format!("  file: {}\n", super_.filename));
        out.push_str(&format!("  extension: {}\n", super_.index));
        out.push_str("  type: ASCII_TBL\n");
        if let Some(name) = extname {
            out.push_str(&format!("  extname: {}\n", name));
        }
        out.push_str(&format!("  rows: {}\n", nrows));
        Ok(out)
    }

    // Number of rows (NAXIS2).  Mirrors TableHDU.nrows so a user
    // who doesn't care about ASCII-vs-binary distinction can write
    // `for hdu in fits: ... hdu.nrows ...` generically.
    #[getter]
    fn nrows(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize)
    }

    // Pythonic length: same as `nrows`.  Mirrors TableHDU.__len__.
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize)
    }
}
