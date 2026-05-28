// Shared tile-cache policy for the compressed image + table HDUs.
//
// The codecs themselves already live in the sibling `zimage` modules
// (gzip / rice / hcompress / plio) and are shared by both compressed
// HDU types.  This module holds the one remaining piece of cache
// policy the two sides have in common.
//
// The per-tile encode/decode *dispatch* is intentionally NOT unified
// here: the image path is 2-D-tile + quant-context shaped, the table
// path is per-column-slab shaped, and a single abstraction would fit
// neither cleanly.  Each compressed HDU keeps its own dispatch.

// 32 MiB by default — large enough to cache a few hundred typical
// 256x256 i4 tiles, small enough not to be surprising on a desktop.
pub(crate) const DEFAULT_TILE_CACHE_BYTES: u64 = 32 * 1024 * 1024;
