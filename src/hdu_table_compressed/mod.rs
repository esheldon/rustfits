// CompressedTableHDU — pyclass for tile-compressed BINTABLEs
// (FITS Tile Compression Convention, `ZTABLE=T`).
//
// Subclasses TableHDU so `isinstance(hdu, TableHDU)` holds on a
// compressed-table HDU, matching the CompressedImageHDU / ImageHDU
// shape on the image side.  Accessors return values from the
// *original* (uncompressed) table — `nrows` is `ZNAXIS2`, `dtype` is
// built from the per-column `ZFORMn` cards rather than the on-disk
// `TFORMn` (all `1QB(maxlen)` heap descriptors).
//
// Split into single-responsibility files (mirrors the `hdu_table/` and
// `hdu_image_compressed/` splits); this mod.rs only wires them and
// re-exports the external surface consumed by `crate::fits` and
// `crate::lib`.  See each file's header for what it owns.

mod append;
mod checksum;
mod hdu;
mod meta;
mod read;
mod repack;
mod setitem;
mod subset;
mod write;
mod write_setup;

pub(crate) use hdu::{header_has_ztable, CompressedTableHDU};
pub(crate) use subset::{CompressedColumnSubset, CompressedSingleColumnSubset};
pub(crate) use write_setup::{
    build_compressed_table_header, default_ztilelen, resolve_compress_arg,
};
