// TableHDU: BINTABLE extension HDU.  Read/write API is not yet implemented;
// the pyclass exists so binary tables in input files are surfaced as their
// own type rather than as a bare HDU.

use pyo3::prelude::*;

use crate::common::{FileHandle, TaintFlag};
use crate::hdu::HDU;

#[pyclass(extends = HDU)]
pub(crate) struct TableHDU;

impl TableHDU {
    pub(crate) fn new(
        header: Vec<String>,
        index: usize,
        header_offset: u64,
        data_offset: u64,
        file: FileHandle,
        tainted: TaintFlag,
    ) -> (Self, HDU) {
        (
            TableHDU,
            HDU::new(header, index, header_offset, data_offset, file, tainted),
        )
    }
}

#[pymethods]
impl TableHDU {
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let index: usize = super_.index();
        Ok(format!("<TableHDU (binary) #{}>", index))
    }
}
