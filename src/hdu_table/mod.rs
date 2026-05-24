// TableHDU module: BINTABLE extension HDU.
//
// Split across single-responsibility files; this mod.rs only wires
// them together and re-exports the external surface (the items
// imported by `crate::fits`, `crate::hdu_image_compressed`, and
// `crate::lib`).
//
// See the per-file headers for what each chunk owns.

mod columns;
mod edit;
mod hdu;
mod read;
mod setitem;
mod write_fixed;
mod write_setup;
mod write_vla;

pub(crate) use columns::{
    bytes_per_element, byteswap_unit, parse_columns, scaling_kind, Column,
    ScalingKind,
};
pub(crate) use hdu::{ColumnSubset, SingleColumnSubset, TableHDU};
pub(crate) use read::{
    build_numpy_dtype, convert_column_cell, field_dtype_and_shape,
    numpy_field_layout, read_descriptor, resolve_columns,
};
pub(crate) use write_fixed::set_pcount_in_cards;
pub(crate) use write_setup::normalize_and_build_table_header;
