// CompressedImageHDU — tile-compressed image extension (ZIMAGE
// convention).  Subclasses ImageHDU so `isinstance(hdu, ImageHDU)`
// works on tile-compressed HDUs; overrides the data-access methods so
// the uncompressed read/write paths never run on BINTABLE bytes.
//
// On disk the HDU is a BINTABLE with the standard COMPRESSED_DATA /
// ZSCALE / ZZERO column conventions, but the user-facing API mirrors
// ImageHDU: shape / dtype / bitpix / ndim / size / __len__ / unit
// return image semantics, reading the Z*-prefixed keys instead of
// NAXIS/BITPIX.  Raw `hdu.header["BITPIX"]` still returns 8 (the
// on-disk BINTABLE bitpix) — astropy's convention.
//
// Split into single-responsibility files (mirrors the `hdu_table/`
// split); this mod.rs only wires them and re-exports the external
// surface consumed by `crate::fits` and `crate::lib`.  See each
// file's header for what it owns.

mod checksum;
mod extending;
mod hdu;
mod meta;
mod read;
mod repack;
mod write;

pub(crate) use extending::CompressedImageExtendContext;
pub(crate) use hdu::{header_has_zimage, CompressedImageHDU};
pub(crate) use meta::compute_n_tiles;
