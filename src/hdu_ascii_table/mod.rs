// AsciiTableHDU module: TABLE extension HDU (ASCII tables).
//
// Split across single-responsibility files; this mod.rs only wires
// them together and re-exports the external surface (the items
// imported by `crate::fits` and `crate::lib`).  See the per-file
// headers for what each chunk owns.

mod columns;
mod edit;
mod format;
mod hdu;
mod meta;
mod read;
mod setitem;
mod write_fixed;
mod write_setup;

pub(crate) use hdu::{
    AsciiColumnSubset, AsciiSingleColumnSubset, AsciiTableHDU,
};
pub(crate) use write_setup::normalize_and_build_ascii_table_header;
