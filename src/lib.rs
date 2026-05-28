// Entry point for the rustfits Rust extension.  Module layout:
//
//   common         — file handle, taint flag, lock helpers, block constants
//   header         — FITSHeader + FITSHeaderEdit + all card-level helpers
//   hdu            — base HDU pyclass
//   hdu_image      — ImageHDU + image read/write/slicing
//   hdu_image_compressed — CompressedImageHDU (ZIMAGE convention)
//   hdu_table      — TableHDU (BINTABLE) stub
//   hdu_table_compressed — CompressedTableHDU (ZTABLE convention)
//   hdu_ascii_table — AsciiTableHDU (TABLE) stub
//   zimage         — per-algorithm tile encoders/decoders + config classes
//   fits           — FITS pyclass + HDU-list parser

use pyo3::prelude::*;

mod cache;
mod common;
mod header;
mod hdu;
mod hdu_image;
mod hdu_image_compressed;
mod hdu_table;
mod hdu_table_compressed;
mod hdu_ascii_table;
mod zimage;
mod fits;
mod checksum;

use crate::header::{py_is_protected_key, FITSHeader, FITSHeaderEdit};
use crate::hdu::HDU;
use crate::hdu_image::ImageHDU;
use crate::hdu_image_compressed::CompressedImageHDU;
use crate::hdu_table::{ColumnSubset, SingleColumnSubset, TableHDU, TableIter};
use crate::hdu_table_compressed::{
    CompressedColumnSubset, CompressedSingleColumnSubset, CompressedTableHDU,
};
use crate::hdu_ascii_table::AsciiTableHDU;
use crate::fits::FITS;
use crate::zimage::compression_config::{
    Gzip1, Gzip2, Hcompress1, Plio1, Quantize, Rice1,
};

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FITS>()?;
    m.add_class::<HDU>()?;
    m.add_class::<ImageHDU>()?;
    m.add_class::<CompressedImageHDU>()?;
    m.add_class::<TableHDU>()?;
    m.add_class::<CompressedTableHDU>()?;
    m.add_class::<CompressedSingleColumnSubset>()?;
    m.add_class::<CompressedColumnSubset>()?;
    m.add_class::<ColumnSubset>()?;
    m.add_class::<SingleColumnSubset>()?;
    m.add_class::<TableIter>()?;
    m.add_class::<AsciiTableHDU>()?;
    m.add_class::<FITSHeader>()?;
    m.add_class::<FITSHeaderEdit>()?;
    m.add_class::<Gzip1>()?;
    m.add_class::<Gzip2>()?;
    m.add_class::<Rice1>()?;
    m.add_class::<Hcompress1>()?;
    m.add_class::<Plio1>()?;
    m.add_class::<Quantize>()?;
    m.add_function(wrap_pyfunction!(py_is_protected_key, m)?)?;
    Ok(())
}
