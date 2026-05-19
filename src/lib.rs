// Entry point for the rustfits Rust extension.  Module layout:
//
//   common         — file handle, taint flag, lock helpers, block constants
//   header         — FITSHeader + FITSHeaderEdit + all card-level helpers
//   hdu            — base HDU pyclass
//   hdu_image      — ImageHDU + image read/write/slicing
//   hdu_table      — TableHDU (BINTABLE) stub
//   hdu_ascii_table — AsciiTableHDU (TABLE) stub
//   fits           — FITS pyclass + HDU-list parser

use pyo3::prelude::*;

mod common;
mod header;
mod hdu;
mod hdu_image;
mod hdu_table;
mod hdu_ascii_table;
mod fits;

use crate::header::{py_is_protected_key, FITSHeader, FITSHeaderEdit};
use crate::hdu::HDU;
use crate::hdu_image::ImageHDU;
use crate::hdu_table::TableHDU;
use crate::hdu_ascii_table::AsciiTableHDU;
use crate::fits::FITS;

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FITS>()?;
    m.add_class::<HDU>()?;
    m.add_class::<ImageHDU>()?;
    m.add_class::<TableHDU>()?;
    m.add_class::<AsciiTableHDU>()?;
    m.add_class::<FITSHeader>()?;
    m.add_class::<FITSHeaderEdit>()?;
    m.add_function(wrap_pyfunction!(py_is_protected_key, m)?)?;
    Ok(())
}
