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

/// An ASCII-table extension HDU (``XTENSION='TABLE'``).
///
/// ASCII tables store row data as fixed-width text, distinct
/// from binary tables (:class:`TableHDU`).  rustfits parses
/// enough of the header to surface ASCII tables as their own
/// HDU type — :attr:`nrows`, ``__len__``, and ``__repr__`` work
/// — but the read/write surface is not yet implemented.
///
/// ASCII tables are rare in modern FITS files; most data
/// pipelines use binary tables instead.  Support beyond
/// inspection is on the roadmap but unprioritized; if you need
/// it, please file an issue with your use case.
///
/// Notes
/// -----
/// To read ASCII-table data today, fall back to astropy or
/// fitsio for that specific HDU.  rustfits opens the file fine
/// and surfaces the ASCII table as this class; just don't call
/// ``.read()`` on it.
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
    // __repr__ is a pyo3 slot dunder — no per-method docstring.
    // Column info is intentionally omitted (ASCII tables aren't
    // yet parsed into a typed Vec<Column>).  Add column listing
    // when the read API lands.
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

    /// Number of rows in the table (``NAXIS2``).
    ///
    /// Mirrors :attr:`TableHDU.nrows` so a user who doesn't care
    /// about ASCII-vs-binary distinction can write
    /// ``for hdu in fits: ... hdu.nrows ...`` generically.
    #[getter]
    fn nrows(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize)
    }

    // __len__ is a pyo3 slot dunder — no per-method docstring.
    // Same value as nrows; mirrors TableHDU.__len__.
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_keyword(&cards, "NAXIS2").unwrap_or(0).max(0) as usize)
    }
}
