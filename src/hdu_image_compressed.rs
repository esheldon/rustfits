// CompressedImageHDU — tile-compressed image extension (ZIMAGE
// convention).  Subclasses ImageHDU so `isinstance(hdu, ImageHDU)`
// works on tile-compressed HDUs; overrides the data-access
// methods so the uncompressed read/write paths never run on
// BINTABLE bytes.
//
// Phase 2 (current): RICE_1 whole-image read.  Subsequent phases
// add slicing (Phase 3), GZIP_1/2 (Phase 4), quantized floats
// (Phase 5), and compressed writes (Phase 7+).  Decoder modules
// live in src/zimage/.
//
// On disk the HDU is a BINTABLE with the standard COMPRESSED_DATA
// / ZSCALE / ZZERO column conventions, but the user-facing API
// mirrors ImageHDU: shape / dtype / bitpix / ndim / size / __len__
// / unit return image semantics, reading the Z*-prefixed keys
// instead of NAXIS/BITPIX.  Raw `hdu.header["BITPIX"]` still
// returns 8 (the on-disk BINTABLE bitpix) — that's astropy's
// convention and what the FITS bytes actually say.
//
// `extname`, `extver`, `has_data`, `header`, `index`, and
// `_force_taint` are inherited from HDU (through ImageHDU).
// `has_data` reads the BINTABLE NAXIS/NAXISn (not the Z* keys)
// but the two agree: non-empty image → non-zero n_tiles →
// BINTABLE NAXIS2 > 0; empty image → n_tiles = 0 → NAXIS2 = 0.
// So no override needed.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use lru::LruCache;
use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyValueError};
use pyo3::types::{PyBytes, PySlice, PyTuple};

use crate::common::{
    check_not_tainted, lock_file, parse_keyword, parse_string_keyword,
    shift_file_tail_and_update_offsets,
    FileHandle, FileLayout, HduOffsets, TaintFlag, BLOCK_SIZE,
};
use crate::hdu::HDU;
use crate::hdu_image::{
    normalize_slice_key, serialize_header_to_disk_bytes, AxisSlice, ImageHDU,
};
use crate::hdu_table::set_pcount_in_cards;
use crate::zimage::CompressionAlgorithm;

// 32 MiB by default — large enough to cache a few hundred typical
// 256x256 i4 tiles, small enough not to be surprising on a desktop.
const DEFAULT_TILE_CACHE_BYTES: u64 = 32 * 1024 * 1024;

// Bytes-bound LRU cache of decoded tiles.  Phase 3 storage layer
// for ZIMAGE reads + slicing.  Values are full-tile bytes in the
// target (stored) dtype, in numpy C-order.  Scaling is applied
// once at the end of the read on the assembled output array, so
// the cache contents compose with `scale=False` reads.
//
// Concurrency model: the inner mutex serializes cache access.
// `get` holds the lock briefly to clone out an Arc — the caller
// then works on the clone with the lock released, so multiple
// readers don't serialize on decoding.  A race where two callers
// both miss the same tile and both decode is acceptable: each
// gets a correct Arc, and `put` is idempotent.
struct TileCache {
    inner: Mutex<TileCacheInner>,
    max_bytes: AtomicU64,
}

struct TileCacheInner {
    // Unbounded count; we evict by byte budget on insert.
    lru: LruCache<u64, Arc<Vec<u8>>>,
    cur_bytes: u64,
}

impl TileCache {
    fn new(max_bytes: u64) -> Self {
        TileCache {
            inner: Mutex::new(TileCacheInner {
                lru: LruCache::unbounded(),
                cur_bytes: 0,
            }),
            max_bytes: AtomicU64::new(max_bytes),
        }
    }

    fn capacity(&self) -> u64 {
        self.max_bytes.load(Ordering::Relaxed)
    }

    fn used_bytes(&self) -> u64 {
        self.inner.lock().map(|g| g.cur_bytes).unwrap_or(0)
    }

    // Look up a tile.  On hit, marks the entry MRU and returns a
    // clone of the Arc — caller works without holding the lock.
    fn get(&self, idx: u64) -> Option<Arc<Vec<u8>>> {
        let mut guard = self.inner.lock().ok()?;
        guard.lru.get(&idx).cloned()
    }

    // Insert a tile.  If the value alone is larger than the cap,
    // it's silently dropped (no point evicting everything else
    // for one giant tile that won't fit).  If cap == 0, caching
    // is disabled — also a no-op.
    fn put(&self, idx: u64, bytes: Arc<Vec<u8>>) {
        let cap = self.capacity();
        if cap == 0 {
            return;
        }
        let size = bytes.len() as u64;
        if size > cap {
            return;
        }
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        // Replace existing entry (frees its bytes from the
        // accounting) before evicting more.
        if let Some(old) = guard.lru.pop(&idx) {
            guard.cur_bytes = guard.cur_bytes.saturating_sub(old.len() as u64);
        }
        while guard.cur_bytes + size > cap {
            match guard.lru.pop_lru() {
                Some((_, val)) => {
                    guard.cur_bytes = guard.cur_bytes
                        .saturating_sub(val.len() as u64);
                }
                None => break,
            }
        }
        guard.lru.put(idx, bytes);
        guard.cur_bytes += size;
    }

    // Change the capacity.  Evicts LRU entries if the new cap is
    // smaller than current usage.  Setting cap = 0 effectively
    // empties the cache.
    fn set_capacity(&self, max_bytes: u64) {
        self.max_bytes.store(max_bytes, Ordering::Relaxed);
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        while guard.cur_bytes > max_bytes {
            match guard.lru.pop_lru() {
                Some((_, val)) => {
                    guard.cur_bytes = guard.cur_bytes
                        .saturating_sub(val.len() as u64);
                }
                None => break,
            }
        }
    }

    fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.lru.clear();
            guard.cur_bytes = 0;
        }
    }
}

#[pyclass(extends = ImageHDU)]
pub(crate) struct CompressedImageHDU {
    cache: Arc<TileCache>,
    // Quantization config from `create_image_hdu(..., quantize=...)`.
    // Populated when the HDU was just created in this session for a
    // float ZBITPIX; `None` after reopen (the FITS Tile Compression
    // Convention only records method+seed in ZQUANTIZ/ZDITHER0 on
    // disk, not the qlevel).  The write path consults this for the
    // qlevel value — for reopened HDUs it falls back to defaults
    // (level=4.0).
    quantize_config:
        Arc<Mutex<Option<crate::zimage::compression_config::Quantize>>>,
}

impl CompressedImageHDU {
    pub(crate) fn new(
        header: Vec<String>,
        index: usize,
        filename: String,
        offsets: Arc<HduOffsets>,
        layout: Arc<FileLayout>,
        file: FileHandle,
        tainted: TaintFlag,
        quantize_config:
            Option<crate::zimage::compression_config::Quantize>,
    ) -> PyClassInitializer<Self> {
        let hdu = HDU::new(
            header, index, filename, offsets, layout, file, tainted,
        );
        PyClassInitializer::from(hdu)
            .add_subclass(ImageHDU)
            .add_subclass(CompressedImageHDU {
                cache: Arc::new(TileCache::new(DEFAULT_TILE_CACHE_BYTES)),
                quantize_config: Arc::new(Mutex::new(quantize_config)),
            })
    }
}

#[pymethods]
impl CompressedImageHDU {
    // Multi-line repr matching ImageHDU's, with the compression
    // config surfaced on one line via the algorithm's __repr__.
    // Reports the *uncompressed* image dtype + dims (what the user
    // sees after .read()), not the on-disk BINTABLE.
    //
    // Compression line uses the same single-line repr that
    // `print(hdu.compression)` shows, so the HDU repr stays the
    // single source of truth no matter which algorithm-specific
    // parameters apply (Rice1 has blocksize, Hcompress1 has
    // scale / smooth, GZIPs have none — all formatted consistently).
    //
    // Degraded fallbacks (repr never crashes on a malformed file):
    //   - ZCMPTYPE present but unrecognized → show the raw string
    //     verbatim (useful for debugging an unknown algorithm).
    //   - ZCMPTYPE missing entirely → show `None` (Python idiom).
    fn __repr__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (zbitpix, shape) = parse_compressed_image_shape(&cards)?;
        let dtype = zbitpix_to_native_dtype(zbitpix)?;
        let extname = parse_string_keyword(&cards, "EXTNAME");
        let bunit = parse_string_keyword(&cards, "BUNIT");

        let compression_repr = match build_compression_config(py, &cards) {
            Ok(cfg) => cfg.bind(py).repr()?.extract::<String>()?,
            Err(_) => parse_string_keyword(&cards, "ZCMPTYPE")
                .unwrap_or_else(|| "None".to_string()),
        };

        let mut out = String::new();
        out.push_str(&format!("  file: {}\n", super_.filename));
        out.push_str(&format!("  extension: {}\n", super_.index));
        out.push_str("  type: COMPRESSED_IMAGE_HDU\n");
        if let Some(name) = extname {
            out.push_str(&format!("  extname: {}\n", name));
        }
        out.push_str("  image info:\n");
        out.push_str(&format!("    data type: {}\n", dtype));
        out.push_str(&format!("    dims: {:?}\n", shape));
        if let Some(u) = bunit {
            out.push_str(&format!("    unit: {}\n", u));
        }
        out.push_str(&format!("  compression: {}\n", compression_repr));
        Ok(out)
    }

    // ----- image-side accessors (parallel to ImageHDU) -----

    // Image dimensions in numpy axis order (slowest first).
    #[getter]
    fn shape(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_compressed_image_shape(&cards)?;
        Ok(PyTuple::new(py, &shape)?.unbind())
    }

    // numpy dtype matching ZBITPIX — what .read() will return.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (zbitpix, _) = parse_compressed_image_shape(&cards)?;
        let dtype_str = zbitpix_to_native_dtype(zbitpix)?;
        let np = py.import("numpy")?;
        Ok(np.call_method1("dtype", (dtype_str,))?.unbind())
    }

    // ZNAXIS — number of image axes.
    #[getter]
    fn ndim(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_compressed_image_shape(&cards)?;
        Ok(shape.len())
    }

    // Total pixel count (product of all ZNAXISn).  Returns 0 for
    // an empty shape so the accessor is well-defined on degenerate
    // headers; real compressed HDUs always have NAXIS > 0.
    #[getter]
    fn size(slf: PyRef<'_, Self>) -> PyResult<u64> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_compressed_image_shape(&cards)?;
        Ok(if shape.is_empty() { 0 } else { shape.iter().product() })
    }

    // Raw ZBITPIX — the image-side bitpix the user will read into.
    #[getter]
    fn bitpix(slf: PyRef<'_, Self>) -> PyResult<i32> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (zbitpix, _) = parse_compressed_image_shape(&cards)?;
        Ok(zbitpix)
    }

    // numpy convention: `len(arr)` is shape[0].
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_compressed_image_shape(&cards)?;
        if shape.is_empty() {
            Ok(0)
        } else {
            Ok(shape[0] as usize)
        }
    }

    // BUNIT (informational).
    #[getter]
    fn unit(slf: PyRef<'_, Self>) -> PyResult<Option<String>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_string_keyword(&cards, "BUNIT"))
    }

    // ----- compression-specific accessors -----

    // Structured compression config: returns the same Gzip1 /
    // Gzip2 / Rice1 / Hcompress1 pyclass instance that would have
    // been passed to `create_image_hdu(..., compress=...)` to
    // produce a file with the same on-disk parameters.  This is
    // the single source of truth for "how is this HDU compressed":
    //
    //   if isinstance(hdu.compression, rustfits.Rice1):
    //       print(hdu.compression.blocksize)
    //
    // The FITS-spec ZCMPTYPE string is available as
    // `hdu.compression.zcmptype`; tile shape as
    // `hdu.compression.tile_shape`.  Two HDUs can be compared with
    // `hdu_a.compression == hdu_b.compression` (field-wise __eq__).
    //
    // Construction cost per access is small (parses cards, builds
    // a tiny pyclass).  No caching — keep parity with the other
    // accessors that re-parse the header on each call.
    #[getter]
    fn compression(
        slf: PyRef<'_, Self>, py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        build_compression_config(py, &cards)
    }

    // Total number of tiles in the image: product of
    // ceil(NAXISn / TILEn).  Storage-layout property (kept here
    // rather than on .compression because it's about how the
    // BINTABLE is laid out, not the algorithm).
    #[getter]
    fn n_tiles(slf: PyRef<'_, Self>) -> PyResult<u64> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (_, image_shape) = parse_compressed_image_shape(&cards)?;
        let tile_shape = parse_tile_shape(&cards, &image_shape);
        Ok(compute_n_tiles(&image_shape, &tile_shape))
    }

    // ----- tile cache -----

    // Current cache capacity in bytes.  Default 32 MiB; tune with
    // `set_tile_cache_size`.  Reads and slicing both consult this
    // budget — there's no per-call opt-out.
    #[getter]
    fn tile_cache_size(slf: PyRef<'_, Self>) -> u64 {
        slf.cache.capacity()
    }

    // Configure the cache capacity in bytes.  Setting to 0
    // disables caching entirely (every decode is one-shot —
    // useful for memory-constrained workflows).  Shrinking the
    // cap below current usage evicts LRU tiles to fit.
    fn set_tile_cache_size(&self, bytes: u64) {
        self.cache.set_capacity(bytes);
    }

    // Bytes currently held in the cache.  Useful for monitoring
    // memory usage and for tests that assert eviction worked.
    #[getter]
    fn tile_cache_used(slf: PyRef<'_, Self>) -> u64 {
        slf.cache.used_bytes()
    }

    // Drop every cached tile.  Keeps `tile_cache_size` as-is, so
    // subsequent decodes will repopulate.  Useful right after a
    // one-shot `read()` to release the tile copies while keeping
    // the cache configured for later slicing.
    fn clear_tile_cache(&self) {
        self.cache.clear();
    }

    // ----- overrides for ImageHDU's data-access methods -----
    //
    // These shadow ImageHDU's implementations so the uncompressed
    // read/write paths never run on a compressed BINTABLE.  Only
    // `read` does real work in Phase 2; everything else raises.

    // Whole-image read.  Walks every tile in the BINTABLE, decodes
    // it (or pulls from the tile cache), places it in the output
    // ndarray.  Honors `scale` and `mask_blank` for compatibility
    // with the ImageHDU signature; `mask_blank=True` currently
    // raises because ZBLANK isn't wired in yet.
    //
    // Cache behavior: every tile read/decode populates the LRU
    // cache (subject to `tile_cache_size`).  This makes subsequent
    // slicing into the same region cheap.  To run cache-free, set
    // `tile_cache_size=0` before the read or call
    // `clear_tile_cache()` afterward.
    #[pyo3(signature = (*, scale=true, mask_blank=false))]
    fn read(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        scale: bool,
        mask_blank: bool,
    ) -> PyResult<Py<PyAny>> {
        let cache = Arc::clone(&slf.cache);
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        read_compressed_image_data(
            py, &cards, super_.offsets.data_offset(),
            &super_.file, &super_.tainted, &cache, scale, mask_blank,
        )
    }

    // Slicing on tile-compressed images.  Reuses the same slice-
    // key parser as ImageHDU (slice + int + ellipsis per axis,
    // with step support), and decodes only the tiles overlapping
    // the requested region.  Each accessed tile is cached, so
    // overlapping slices later hit warm tiles.
    //
    // Always applies BSCALE/BZERO (matches the table-side
    // convention; use .read(scale=False) to bypass for the whole
    // image).  All-int multi-axis (`hdu[i, j, k]`) returns a
    // numpy scalar, matching ImageHDU semantics.
    //
    // Phase 3 supports the same slicing surface as ImageHDU
    // (slice/int/ellipsis); fancy list/array indexing falls
    // through `parse_axis_indexer` and raises the existing
    // "unsupported index type" error.
    fn __getitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let cache = Arc::clone(&slf.cache);
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        slice_compressed_image(
            py, &cards, super_.offsets.data_offset(),
            &super_.file, &super_.tainted, &cache, key,
        )
    }

    // Compressed writes are Phase 7+ territory.  Reject every
    // path that would otherwise inherit ImageHDU's uncompressed
    // implementation, since those would happily blast bytes into
    // the BINTABLE data area and produce a corrupt file.
    // Bulk-write a compressed image.  Encodes every tile in RAM
    // first (validate-then-mutate), then grows the file as needed,
    // writes the per-tile descriptors into the main data section
    // and the encoded bytes into the heap, and rewrites the PCOUNT
    // card.  `start=` is not supported on compressed images (every
    // tile is encoded independently; partial writes would mean a
    // partial tile, which doesn't compose with the encode-once
    // model).  Pass the full image array as `data`.
    //
    // Phase 7 supports Gzip1, Gzip2, and Rice1 with integer
    // ZBITPIX (u1/i2/i4/i8 for GZIP; u1/i2/i4 for RICE).
    #[pyo3(signature = (data, start=None))]
    fn write(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        start: Option<Vec<i64>>,
    ) -> PyResult<()> {
        if start.is_some() {
            return Err(PyNotImplementedError::new_err(
                "start= is not supported for compressed-image writes; \
                 the encoder operates per-tile and partial writes would \
                 mean partial tiles, which don't compose with the \
                 single-pass encode model"
            ));
        }
        let cache = Arc::clone(&slf.cache);
        let quantize_config = Arc::clone(&slf.quantize_config);
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        write_compressed_image_data(
            py, data, &cards,
            &super_.offsets,
            &super_.file, &super_.layout, &super_.tainted,
            &cache,
            &super_.header,
            &quantize_config,
        )
    }

    #[pyo3(signature = (data, start=None))]
    #[allow(unused_variables)]
    fn extend(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        start: Option<Vec<i64>>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "extending tile-compressed images is not yet implemented \
             (planned: Phase 7+ of the ZIMAGE roadmap)."
        ))
    }

    #[allow(unused_variables)]
    fn __setitem__(
        slf: PyRef<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "assigning to tile-compressed images is not yet \
             implemented (planned: Phase 7+ of the ZIMAGE roadmap)."
        ))
    }
}

// ----- header-parsing helpers -----

// Parse ZBITPIX, ZNAXIS, ZNAXISn → (zbitpix, numpy-order shape).
// Tolerant of NAXIS=0 in the same way parse_image_hdu_shape_lax is:
// returns an empty shape rather than erroring, so accessors and
// repr work on malformed headers.
fn parse_compressed_image_shape(
    header: &[String],
) -> PyResult<(i32, Vec<u64>)> {
    let zbitpix = parse_keyword(header, "ZBITPIX")
        .ok_or_else(|| PyValueError::new_err(
            "compressed image HDU missing ZBITPIX"
        ))? as i32;
    let znaxis = parse_keyword(header, "ZNAXIS")
        .unwrap_or(0).max(0) as usize;
    let mut shape: Vec<u64> = Vec::with_capacity(znaxis);
    for i in 1..=znaxis {
        let d = parse_keyword(header, &format!("ZNAXIS{}", i))
            .unwrap_or(0).max(0) as u64;
        shape.push(d);
    }
    shape.reverse();
    Ok((zbitpix, shape))
}

// Parse ZTILEn → numpy-order tile shape.  When ZTILEn is absent the
// FITS convention default is: ZTILE1 = ZNAXIS1, all others = 1
// (i.e. "row tiles").  Returns the tile shape in numpy axis order
// (slowest first), so a 2-D image with default tiles gets
// [1, ZNAXIS1].
fn parse_tile_shape(header: &[String], image_shape: &[u64]) -> Vec<u64> {
    let n = image_shape.len();
    // image_shape is in numpy order; ZTILE1 corresponds to the
    // FITS-fastest axis = numpy-last axis = image_shape[n-1].
    let mut fits_order: Vec<u64> = Vec::with_capacity(n);
    for i in 1..=n {
        let key = format!("ZTILE{}", i);
        let default = if i == 1 { image_shape[n - 1] } else { 1 };
        let v = parse_keyword(header, &key)
            .map(|x| x.max(0) as u64)
            .unwrap_or(default);
        fits_order.push(v);
    }
    fits_order.into_iter().rev().collect()
}

// Product of ceil(image / tile) across all axes.  A zero in either
// shape collapses the product to 0 (no tiles).
pub(crate) fn compute_n_tiles(image_shape: &[u64], tile_shape: &[u64]) -> u64 {
    if image_shape.is_empty() {
        return 0;
    }
    let mut total: u64 = 1;
    for (&img, &tile) in image_shape.iter().zip(tile_shape.iter()) {
        if img == 0 || tile == 0 {
            return 0;
        }
        total = total.saturating_mul((img + tile - 1) / tile);
    }
    total
}

// Image dtype string for a given ZBITPIX value.  Same supported set
// as the uncompressed image side (u1/i2/i4/i8/f4/f8); the unsigned-
// int trick still operates via BSCALE/BZERO at read time and isn't
// a property of ZBITPIX itself.
fn zbitpix_to_native_dtype(zbitpix: i32) -> PyResult<&'static str> {
    match zbitpix {
        8 => Ok("u1"),
        16 => Ok("i2"),
        32 => Ok("i4"),
        64 => Ok("i8"),
        -32 => Ok("f4"),
        -64 => Ok("f8"),
        _ => Err(PyValueError::new_err(format!(
            "unsupported ZBITPIX {}", zbitpix
        ))),
    }
}

// Detect ZIMAGE=T in a BINTABLE header.  Called from
// parse_hdus_from_file's dispatch to route to CompressedImageHDU
// instead of TableHDU.  Looks for a logical-T card; tolerant of
// the various whitespace forms FITS allows.
pub(crate) fn header_has_zimage(header: &[String]) -> bool {
    for card in header {
        if card.len() < 9 {
            continue;
        }
        if card[..8].trim() != "ZIMAGE" {
            continue;
        }
        // Logical value: column 30 is the value-position 'T' or 'F'
        // per FITS fixed-format, but be lenient and just look for
        // a 'T' in the value portion (after the '=').
        if let Some(eq) = card.find('=') {
            let value = &card[eq + 1..];
            let trimmed = value.trim_start();
            if trimmed.starts_with('T') {
                return true;
            }
        }
    }
    false
}

// ====== Tile read ======
//
// REFACTOR NOTE (revisit when Phase 5 adds quantized floats):
// `parse_rice_params` is RICE-specific.  When Phase 5 adds
// quantized floats it needs siblings for ZSCALE/ZZERO/ZBLANK
// column positions (each is a per-tile column in the BINTABLE,
// not a ZNAMEn/ZVALn pair).  Likely shape: a
// `CompressionContext` struct passed through the read loop,
// holding per-algorithm params plus optional quantization-
// column offsets.  Outer flow (tile loop → decode → place)
// stays unchanged.

// Top-level entry point invoked from CompressedImageHDU::read.
// Walks the BINTABLE one tile at a time, decoding each via the
// algorithm-specific code in src/zimage/, and assembling the
// tiles into the output ndarray.  Applies BSCALE/BZERO via the
// shared image-side scaling machinery so a scaled compressed
// HDU returns the same dtype as an equivalent uncompressed one.
//
// Current limits (will lift later):
//   - RICE_1 / GZIP_1 / GZIP_2 supported; HCOMPRESS_1 / PLIO_1 are
//     Phase 6
//   - Integer ZBITPIX only (Phase 5 adds quantized floats)
//   - mask_blank=True rejected (ZBLANK handling is a follow-up)
fn read_compressed_image_data(
    py: Python<'_>,
    cards: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    tainted: &TaintFlag,
    cache: &TileCache,
    scale: bool,
    mask_blank: bool,
) -> PyResult<Py<PyAny>> {
    check_not_tainted(tainted)?;

    if mask_blank {
        return Err(PyNotImplementedError::new_err(
            "mask_blank on tile-compressed images requires ZBLANK \
             handling, not yet implemented (planned as a small \
             follow-up after Phase 2 of the ZIMAGE roadmap)."
        ));
    }

    // Algorithm dispatch — RICE_1 / GZIP_1 / GZIP_2 supported.
    // HCOMPRESS_1 / PLIO_1 fall through the per-algorithm decoder
    // dispatch in src/zimage/mod.rs and surface its NotImplemented
    // error.
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE")
        .ok_or_else(|| PyValueError::new_err(
            "compressed HDU missing ZCMPTYPE"
        ))?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;

    // Image shape + bitpix (image-side).
    let (zbitpix, image_shape) = parse_compressed_image_shape(cards)?;
    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)"
        ));
    }
    // Decoder parameters.  Two distinct "ZBITPIX" values are in
    // play: `zbitpix` is the *image-side* output dtype (8/16/32/64
    // for integer images, -32/-64 for quantized float); the
    // decoder needs to know the *stored integer* dtype, which is
    // the same as zbitpix for integer images but is always 32
    // (i32) for the quantized-float path.  Same idea for bytepix:
    // 1/2/4/8 for integer images, always 4 for quantized float.
    let stored_zbitpix: i32 = if zbitpix < 0 { 32 } else { zbitpix };
    let default_bytepix: u32 = match stored_zbitpix {
        8 => 1, 16 => 2, 32 => 4, 64 => 8,
        _ => return Err(PyValueError::new_err(format!(
            "unsupported ZBITPIX {}", zbitpix
        ))),
    };

    // RICE parameters: BLOCKSIZE and BYTEPIX from ZNAMEn/ZVALn.
    // Defaults: BLOCKSIZE=32, BYTEPIX = stored_zbitpix/8.
    let (blocksize, bytepix_from_header) = parse_rice_params(cards);
    let blocksize = blocksize.unwrap_or(32);
    let bytepix = bytepix_from_header.unwrap_or(default_bytepix);

    // HCOMPRESS smoothing flag (ZNAMEn='SMOOTH').  False for any
    // other algorithm (the decoder dispatch ignores it).
    let smooth = parse_hcompress_smooth(cards);

    // BINTABLE layout.
    let naxis1 = parse_keyword(cards, "NAXIS1")
        .ok_or_else(|| PyValueError::new_err("BINTABLE missing NAXIS1"))?
        as u64;
    let naxis2 = parse_keyword(cards, "NAXIS2")
        .ok_or_else(|| PyValueError::new_err("BINTABLE missing NAXIS2"))?
        as u64;
    // THEAP default = NAXIS1 * NAXIS2 (heap immediately after the
    // main data section).
    let theap = parse_keyword(cards, "THEAP")
        .map(|x| x.max(0) as u64)
        .unwrap_or(naxis1 * naxis2);
    let cols = find_data_columns(cards)?;

    // Quantization context (Some when ZBITPIX is float).  Carries
    // the dither method, ZDITHER0 seed, and the per-tile column
    // offsets needed for dequant.
    // For float ZBITPIX this is Some(ctx) when quantization is in
    // play (ZSCALE/ZZERO present + ZQUANTIZ != 'NONE'); None when
    // the file stores raw unquantized floats.  Always None for
    // integer ZBITPIX.
    let quant = if zbitpix < 0 {
        build_quant_context(cards, &cols)?
    } else {
        None
    };

    // Tile shape + sanity-check against NAXIS2.
    let tile_shape = parse_tile_shape(cards, &image_shape);
    let n_tiles = compute_n_tiles(&image_shape, &tile_shape);
    if n_tiles != naxis2 {
        return Err(PyValueError::new_err(format!(
            "ZIMAGE row count NAXIS2={} disagrees with computed \
             tile count {} (image shape {:?}, tile shape {:?})",
            naxis2, n_tiles, image_shape, tile_shape,
        )));
    }

    // Allocate output ndarray of the right shape + native dtype.
    let np = py.import("numpy")?;
    let dtype_str = zbitpix_to_native_dtype(zbitpix)?;
    let shape_tuple = PyTuple::new(py, &image_shape)?;
    let out_arr = np.call_method1("empty", (shape_tuple, dtype_str))?;
    // Zero the array so that any test that asserts initial state
    // (or any future code that exits early) sees a clean value.
    out_arr.call_method0("fill")
        .or_else(|_| {
            // fill() requires an argument; pass 0.
            out_arr.call_method1("fill", (0i32,))
        })?;

    // Iterate tiles.  Each BINTABLE row = one tile; rows are
    // ordered FITS-row-major (numpy-last-axis fastest).  Each
    // tile is fetched via the cache (decoded on miss, served from
    // RAM on hit).  File I/O happens per tile under the file lock;
    // decode happens outside the lock so concurrent threads aren't
    // serialized through long decode runs.
    for tile_idx in 0..n_tiles {
        let (origin, actual_shape) = tile_origin_and_shape(
            tile_idx, &image_shape, &tile_shape,
        );
        let tile_bytes = get_or_decode_tile(
            py, cache, file_handle, tainted, tile_idx, data_offset,
            naxis1, theap, &cols, algorithm, &actual_shape,
            bytepix, blocksize, stored_zbitpix,
            zbitpix, quant.as_ref(), smooth,
        )?;
        place_tile_bytes_into_output(
            py, &out_arr, &tile_bytes, dtype_str,
            &actual_shape, &origin,
        )?;
    }

    // Apply BSCALE/BZERO (same dispatch the uncompressed path uses).
    let unbound = out_arr.unbind();
    let final_arr = if scale {
        let (bscale, bzero) = crate::hdu_image::parse_bscale_bzero(cards);
        let kind = crate::hdu_image::image_scaling_kind(zbitpix, bscale, bzero);
        crate::hdu_image::apply_image_scaling(
            py, unbound, zbitpix, kind, bscale, bzero,
        )?
    } else {
        unbound
    };
    Ok(final_arr)
}

// Per-column descriptor needed to locate and interpret the heap
// bytes for one tile.  All three ZIMAGE data columns (primary,
// GZIP fallback, UNCOMPRESSED fallback) are variable-length, so
// each carries `is_q` (descriptor width) and `inner_byte_width`
// (size of one heap element, used to convert descriptor
// nelements → byte count).  COMPRESSED_DATA and
// GZIP_COMPRESSED_DATA always use byte inner type (B → 1);
// UNCOMPRESSED_DATA uses whichever inner type matches ZBITPIX
// (B/I/J/K), so the byte-count math has to consult inner_byte_width.
struct ColumnInfo {
    byte_offset_in_row: u64,
    is_q: bool,
    inner_byte_width: u64,
}

// All ZIMAGE data columns of interest; the primary heap column is
// required, fallbacks and quantization columns are optional and
// only used in specific code paths (fallbacks when the primary
// tile is empty; ZSCALE/ZZERO when ZBITPIX is float).  Resolved
// in find_data_columns by walking TTYPEn.
struct DataColumns {
    primary: ColumnInfo,
    gzip_fallback: Option<ColumnInfo>,
    uncompressed_fallback: Option<ColumnInfo>,
    // Fixed-width 1D (double) columns used for per-tile
    // dequantization.  Stored as raw row-byte offsets — the
    // dequant path seeks to `data_offset + row*naxis1 + offset`
    // and reads 8 big-endian bytes.
    zscale_offset_in_row: Option<u64>,
    zzero_offset_in_row: Option<u64>,
}

// What we got back from the heap for one tile.  Three cases:
//
//   - PrimaryCompressed: the primary COMPRESSED_DATA column had
//     bytes.  For a quantized-float HDU these are quantized i32
//     values; for an integer HDU they're the matching int dtype.
//     The caller decodes, then (for float HDUs) runs dequant.
//
//   - FallbackCompressed: one of the fallback columns had bytes
//     (primary was empty).  cfitsio's lossless-fallback convention
//     is: bytes are the *original physical* values, not quantized.
//     For a float HDU that means raw f4/f8 bytes — the dequant
//     step must be SKIPPED.  For an integer HDU it's just raw int
//     bytes (same shape as PrimaryCompressed would have been).
//
//   - Uncompressed: UNCOMPRESSED_DATA column had bytes; same
//     "raw physical" convention as FallbackCompressed, just
//     without the compression layer.
//
// The `algorithm` on the Compressed variants tells the decoder
// which decompressor to run; the *primary-vs-fallback* split is
// what determines whether dequant runs afterwards.
enum TilePayload {
    PrimaryCompressed { bytes: Vec<u8>, algorithm: CompressionAlgorithm },
    FallbackCompressed { bytes: Vec<u8>, algorithm: CompressionAlgorithm },
    Uncompressed { bytes: Vec<u8> },
}

// Quantization parameters for the float-ZBITPIX path.  Built once
// per read at the top of `read_compressed_image_data` /
// `slice_compressed_image` when ZBITPIX is negative, then carried
// through the tile loop so `get_or_decode_tile` can read per-tile
// ZSCALE/ZZERO and dequantize after decode.
struct QuantContext {
    method: crate::zimage::quantize::DitherMethod,
    zdither0: i64,
    zscale_offset_in_row: u64,
    zzero_offset_in_row: u64,
    // ZBITPIX of the *output* float dtype (-32 or -64).  Decoder
    // always works in i32; this picks dequantize_to_f32 vs _f64.
    output_zbitpix: i32,
}

// Parse ZQUANTIZ + ZDITHER0 + verify ZSCALE/ZZERO columns are
// present.  ZDITHER0 is the per-file PRNG offset; absent or
// non-positive values fall back to 1 (cfitsio's default).
// Returns `Some(ctx)` when quantization is in play, `None` when
// the HDU is float but stores raw (un-quantized) bytes.  Three
// independent signals can indicate "no quantization":
//   - ZQUANTIZ='NONE' (astropy's explicit marker)
//   - parse_dither_method returns None
//   - ZSCALE / ZZERO columns are missing
// Any of these → return None.  Otherwise we need both columns
// present and a known dither method; missing them is an error
// (would indicate a malformed file).
fn build_quant_context(
    cards: &[String],
    cols: &DataColumns,
) -> PyResult<Option<QuantContext>> {
    let zquantiz = parse_string_keyword(cards, "ZQUANTIZ");
    let method_opt = crate::zimage::quantize::parse_dither_method(
        zquantiz.as_deref(),
    )?;

    // Three ways to land on the no-quantization path: ZQUANTIZ
    // explicitly says NONE, or one of the per-tile scale/zero
    // columns is absent (cfitsio's convention).
    let method = match (method_opt, cols.zscale_offset_in_row,
                        cols.zzero_offset_in_row) {
        (Some(m), Some(_), Some(_)) => m,
        _ => return Ok(None),
    };

    let zdither0 = parse_keyword(cards, "ZDITHER0").unwrap_or(0);
    // cfitsio uses zdither0 of 1 as the default when the keyword
    // is missing for SUBTRACTIVE_DITHER_*.  For NO_DITHER the
    // value is unused but we leave whatever was parsed.
    let zdither0 = if zdither0 <= 0 { 1 } else { zdither0 };

    let zscale_offset_in_row = cols.zscale_offset_in_row.unwrap();
    let zzero_offset_in_row = cols.zzero_offset_in_row.unwrap();

    // ZBITPIX is parsed by the caller; we pull it again here so
    // the context is self-contained.
    let zbitpix = parse_keyword(cards, "ZBITPIX")
        .ok_or_else(|| PyValueError::new_err(
            "compressed HDU missing ZBITPIX"
        ))? as i32;
    if zbitpix != -32 && zbitpix != -64 {
        return Err(PyValueError::new_err(format!(
            "build_quant_context called for non-float ZBITPIX={}",
            zbitpix
        )));
    }
    Ok(Some(QuantContext {
        method,
        zdither0,
        zscale_offset_in_row,
        zzero_offset_in_row,
        output_zbitpix: zbitpix,
    }))
}

// Walk TFORMn / TTYPEn to locate the primary COMPRESSED_DATA
// column (required) plus the optional GZIP_COMPRESSED_DATA and
// UNCOMPRESSED_DATA fallback columns.  All preceding columns
// contribute their byte width to the running offset.
fn find_data_columns(header: &[String]) -> PyResult<DataColumns> {
    let tfields = parse_keyword(header, "TFIELDS").unwrap_or(0).max(0) as u64;
    if tfields == 0 {
        return Err(PyValueError::new_err(
            "ZIMAGE BINTABLE has TFIELDS=0"
        ));
    }
    let mut primary: Option<ColumnInfo> = None;
    let mut gzip_fallback: Option<ColumnInfo> = None;
    let mut uncompressed_fallback: Option<ColumnInfo> = None;
    let mut zscale_offset_in_row: Option<u64> = None;
    let mut zzero_offset_in_row: Option<u64> = None;

    let mut offset: u64 = 0;
    for i in 1..=tfields {
        let ttype = parse_string_keyword(header, &format!("TTYPE{}", i))
            .unwrap_or_default();
        let tform = parse_string_keyword(header, &format!("TFORM{}", i))
            .ok_or_else(|| PyValueError::new_err(format!(
                "ZIMAGE BINTABLE column {} missing TFORM", i
            )))?;
        let width = tform_byte_width(&tform)?;
        let info = ColumnInfo {
            byte_offset_in_row: offset,
            is_q: tform_is_q_descriptor(&tform),
            inner_byte_width: tform_vla_inner_byte_width(&tform).unwrap_or(1),
        };
        match ttype.trim() {
            "COMPRESSED_DATA" => primary = Some(info),
            "GZIP_COMPRESSED_DATA" => gzip_fallback = Some(info),
            "UNCOMPRESSED_DATA" => uncompressed_fallback = Some(info),
            // ZSCALE / ZZERO are fixed-width 1D columns — the
            // ColumnInfo's VLA fields are meaningless here, but
            // the byte_offset is what the dequant path needs.
            "ZSCALE" => zscale_offset_in_row = Some(offset),
            "ZZERO" => zzero_offset_in_row = Some(offset),
            _ => {}
        }
        offset += width;
    }
    let primary = primary.ok_or_else(|| PyValueError::new_err(
        "ZIMAGE BINTABLE missing COMPRESSED_DATA column"
    ))?;
    Ok(DataColumns {
        primary,
        gzip_fallback,
        uncompressed_fallback,
        zscale_offset_in_row,
        zzero_offset_in_row,
    })
}

// Byte width of one row's slot for a given TFORM value.  Handles
// the column types that appear in ZIMAGE BINTABLEs.  Unknown
// types raise — better than silently mis-computing an offset.
fn tform_byte_width(tform: &str) -> PyResult<u64> {
    let trimmed = tform.trim();
    let bytes = trimmed.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    let repeat: u64 = if idx == 0 {
        1
    } else {
        trimmed[..idx].parse().map_err(|_| PyValueError::new_err(
            format!("bad TFORM repeat: {}", tform)
        ))?
    };
    if idx >= bytes.len() {
        return Err(PyValueError::new_err(format!(
            "TFORM missing type letter: {}", tform
        )));
    }
    let t = bytes[idx] as char;
    match t {
        // Fixed-width scalars.
        'L' | 'A' | 'B' => Ok(repeat),
        'I' => Ok(repeat * 2),
        'J' | 'E' => Ok(repeat * 4),
        'K' | 'D' | 'C' => Ok(repeat * 8),
        'M' => Ok(repeat * 16),
        // Bit-array: ceil(repeat / 8) bytes.
        'X' => Ok((repeat + 7) / 8),
        // Variable-length descriptors — fixed bytes per
        // descriptor; `repeat` here is the descriptor count.
        'P' => Ok(repeat * 8),
        'Q' => Ok(repeat * 16),
        other => Err(PyValueError::new_err(format!(
            "unsupported TFORM type '{}' in ZIMAGE BINTABLE", other
        ))),
    }
}

// Inner element byte width for a VLA TFORM (`Pt` / `Qt`).  Used
// to convert a descriptor's `nelements` to a byte count when
// reading heap payload.  Returns None for non-VLA TFORMs (fixed-
// width columns don't have an inner type letter, since their
// repeat count already gives the byte width directly).
fn tform_vla_inner_byte_width(tform: &str) -> Option<u64> {
    let trimmed = tform.trim();
    let bytes = trimmed.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    if idx >= bytes.len() {
        return None;
    }
    let outer = bytes[idx] as char;
    if outer != 'P' && outer != 'Q' {
        return None;
    }
    let inner_idx = idx + 1;
    if inner_idx >= bytes.len() {
        return None;
    }
    match bytes[inner_idx] as char {
        'L' | 'A' | 'B' => Some(1),
        'I' => Some(2),
        'J' | 'E' => Some(4),
        'K' | 'D' | 'C' => Some(8),
        'M' => Some(16),
        _ => None,
    }
}

// Does this TFORM use 'Q' descriptors (16 bytes) rather than 'P'?
fn tform_is_q_descriptor(tform: &str) -> bool {
    let trimmed = tform.trim();
    let bytes = trimmed.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    idx < bytes.len() && bytes[idx] as char == 'Q'
}

// Walk ZNAMEn/ZVALn pairs to extract the BLOCKSIZE and BYTEPIX
// values used by the RICE encoder.  Either may be absent; the
// caller substitutes defaults (32 / ZBITPIX/8).
fn parse_rice_params(header: &[String]) -> (Option<u32>, Option<u32>) {
    let mut blocksize: Option<u32> = None;
    let mut bytepix: Option<u32> = None;
    // ZNAMEn / ZVALn pairs are indexed 1..; in practice there
    // are at most a few.  Iterate until we hit a gap.
    for n in 1.. {
        let name_key = format!("ZNAME{}", n);
        let val_key = format!("ZVAL{}", n);
        let name = parse_string_keyword(header, &name_key);
        if name.is_none() {
            break;
        }
        let v = parse_keyword(header, &val_key);
        match (name.as_deref().map(|s| s.trim()), v) {
            (Some("BLOCKSIZE"), Some(val)) => {
                if val > 0 {
                    blocksize = Some(val as u32);
                }
            }
            (Some("BYTEPIX"), Some(val)) => {
                if val > 0 {
                    bytepix = Some(val as u32);
                }
            }
            _ => {}
        }
    }
    (blocksize, bytepix)
}

// Walk ZNAMEn/ZVALn pairs to extract the HCOMPRESS_1 SCALE value.
// On the read side cfitsio reads SCALE from the compressed stream;
// the header card is informational only.  On the WRITE side this
// reader pulls it back out of the header (which create_compressed_
// image_hdu_impl just emitted from the user's Hcompress1 config)
// to drive the encoder.  Defaults to 0 (lossless) when absent.
fn parse_hcompress_scale(header: &[String]) -> i32 {
    for n in 1.. {
        let name_key = format!("ZNAME{}", n);
        let val_key = format!("ZVAL{}", n);
        let name = parse_string_keyword(header, &name_key);
        if name.is_none() {
            break;
        }
        if let Some(name) = name.as_deref().map(|s| s.trim()) {
            if name.eq_ignore_ascii_case("SCALE") {
                if let Some(v) = parse_keyword(header, &val_key) {
                    return v as i32;
                }
            }
        }
    }
    0
}

// Walk ZNAMEn/ZVALn pairs to extract the SMOOTH flag for HCOMPRESS_1.
// SCALE also lives there (ZNAMEn='SCALE') but the decoder reads
// SCALE directly from the compressed stream (per cfitsio's
// fits_hdecompress.c line 1076), so we don't need it here for read
// — only for the write side (see parse_hcompress_scale).  Returns
// false (no smoothing) when the keyword is absent.
fn parse_hcompress_smooth(header: &[String]) -> bool {
    for n in 1.. {
        let name_key = format!("ZNAME{}", n);
        let val_key = format!("ZVAL{}", n);
        let name = parse_string_keyword(header, &name_key);
        if name.is_none() {
            break;
        }
        if let Some(name) = name.as_deref().map(|s| s.trim()) {
            if name.eq_ignore_ascii_case("SMOOTH") {
                if let Some(v) = parse_keyword(header, &val_key) {
                    return v != 0;
                }
            }
        }
    }
    false
}

// Build the structured compression-config pyclass instance for a
// compressed-image HDU's header.  Backs the `.compression` getter
// on CompressedImageHDU: parses ZCMPTYPE, the tile shape, the heap
// format from TFORM1, plus algorithm-specific ZNAMEn/ZVALn cards
// (BLOCKSIZE for RICE, SCALE+SMOOTH for HCOMPRESS), and returns the
// matching Gzip1 / Gzip2 / Rice1 / Hcompress1 pyclass.  PLIO_1
// support lands when the encoder ships; until then it falls
// through to the unknown-algorithm branch.
//
// Errors: missing ZCMPTYPE → ValueError; unknown algorithm name →
// the error from parse_algorithm.  Caller (the .compression getter)
// surfaces both directly.
fn build_compression_config(
    py: Python<'_>, cards: &[String],
) -> PyResult<Py<PyAny>> {
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE")
        .ok_or_else(|| PyValueError::new_err(
            "compressed HDU missing ZCMPTYPE"
        ))?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;

    let (_, image_shape) = parse_compressed_image_shape(cards)?;
    let tile_shape = parse_tile_shape(cards, &image_shape);

    // Heap format: 'P' (8-byte descriptors) or 'Q' (16-byte) from
    // TFORM1.  Defaults to 'P' if TFORM1 is missing (malformed
    // header, but match the parse-tolerantly convention).
    let heap_format = parse_string_keyword(cards, "TFORM1")
        .map(|t| {
            let trimmed = t.trim();
            if trimmed.starts_with("1Q") || trimmed.starts_with('Q') {
                'Q'
            } else {
                'P'
            }
        })
        .unwrap_or('P');

    use crate::zimage::compression_config::{
        Gzip1, Gzip2, Hcompress1, Plio1, Rice1,
    };
    use crate::zimage::CompressionAlgorithm;
    match algorithm {
        CompressionAlgorithm::Gzip1 => {
            let cfg = Gzip1 {
                tile_shape: Some(tile_shape),
                heap_format,
            };
            Ok(Py::new(py, cfg)?.into_any())
        }
        CompressionAlgorithm::Gzip2 => {
            let cfg = Gzip2 {
                tile_shape: Some(tile_shape),
                heap_format,
            };
            Ok(Py::new(py, cfg)?.into_any())
        }
        CompressionAlgorithm::Rice1 => {
            let (blocksize_opt, _) = parse_rice_params(cards);
            let cfg = Rice1 {
                tile_shape: Some(tile_shape),
                heap_format,
                blocksize: blocksize_opt.unwrap_or(32),
            };
            Ok(Py::new(py, cfg)?.into_any())
        }
        CompressionAlgorithm::Hcompress1 => {
            let scale = parse_hcompress_scale(cards);
            let smooth = parse_hcompress_smooth(cards);
            let cfg = Hcompress1 {
                tile_shape: Some(tile_shape),
                heap_format,
                scale,
                smooth,
            };
            Ok(Py::new(py, cfg)?.into_any())
        }
        CompressionAlgorithm::Plio1 => {
            let cfg = Plio1 {
                tile_shape: Some(tile_shape),
                heap_format,
            };
            Ok(Py::new(py, cfg)?.into_any())
        }
    }
}

// Given a tile index (0..n_tiles, FITS-row-major), the image
// shape (numpy order), and the nominal tile shape (numpy order),
// return the tile's numpy-order origin and its actual shape
// (which may be smaller than nominal at the image edges).
fn tile_origin_and_shape(
    tile_idx: u64,
    image_shape_numpy: &[u64],
    nominal_tile_shape_numpy: &[u64],
) -> (Vec<u64>, Vec<u64>) {
    let d = image_shape_numpy.len();
    let mut idx = tile_idx;
    let mut tile_coord_numpy = vec![0u64; d];
    // Unfold from numpy-last (= FITS-fastest = varies fastest in
    // the BINTABLE row ordering) to numpy-first.
    for axis_numpy in (0..d).rev() {
        let n_along = (image_shape_numpy[axis_numpy]
            + nominal_tile_shape_numpy[axis_numpy] - 1)
            / nominal_tile_shape_numpy[axis_numpy];
        tile_coord_numpy[axis_numpy] = idx % n_along;
        idx /= n_along;
    }
    let mut origin = vec![0u64; d];
    let mut shape = vec![0u64; d];
    for axis in 0..d {
        origin[axis] = tile_coord_numpy[axis]
            * nominal_tile_shape_numpy[axis];
        let end = (origin[axis] + nominal_tile_shape_numpy[axis])
            .min(image_shape_numpy[axis]);
        shape[axis] = end - origin[axis];
    }
    (origin, shape)
}

// Look up a tile in the cache or, on miss, read payload + decode
// + insert.  Returns the tile's bytes in target (stored) dtype,
// numpy C-order, ready to be wrapped in a numpy ndarray.  The
// returned Arc is shared with the cache; both points keep the
// allocation alive until the consumer drops the reference.
//
// File I/O (descriptor read + heap read) is done under the file
// lock; the lock is released before decode.  The cache lock is
// taken twice (once for `get`, once for `put`) and held only
// across in-memory ops.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn get_or_decode_tile(
    _py: Python<'_>,
    cache: &TileCache,
    file_handle: &FileHandle,
    tainted: &TaintFlag,
    tile_idx: u64,
    data_offset: u64,
    naxis1: u64,
    theap: u64,
    cols: &DataColumns,
    algorithm: CompressionAlgorithm,
    actual_shape: &[u64],
    bytepix: u32,
    blocksize: u32,
    // ZBITPIX as seen by the decoder.  For integer ZBITPIX this is
    // just the image's ZBITPIX; for quantized float (-32/-64) the
    // caller passes 32 so the decoder produces i32 bytes that the
    // dequant step then consumes.
    stored_zbitpix: i32,
    // Output ZBITPIX (image-side dtype).  Used by the dequant step
    // to decide f32 vs f64 output.  Equals stored_zbitpix when
    // there's no quantization in play.
    output_zbitpix: i32,
    quant: Option<&QuantContext>,
    // HCOMPRESS_1 SMOOTH flag (ignored by every other algorithm).
    smooth: bool,
) -> PyResult<Arc<Vec<u8>>> {
    if let Some(arc) = cache.get(tile_idx) {
        return Ok(arc);
    }
    check_not_tainted(tainted)?;

    // Read payload (descriptor + heap) under the file lock.  The
    // returned variant tells us how to interpret the bytes.  When
    // quantization is in play, also read this tile's ZSCALE/ZZERO
    // from their fixed-width columns under the same lock acquire.
    let (payload, scale_zero) = fetch_tile_payload_and_quant(
        file_handle, tile_idx, data_offset, naxis1, theap, cols,
        algorithm, quant,
    )?;

    // Decode (no file lock held here).  Two distinct decode paths:
    //
    //   - Primary path: bytes are in the *stored* int representation
    //     (i32 for quantized float, native int for int HDU).  Decode
    //     with stored_zbitpix / bytepix.  For float output, the
    //     dequant step then converts to f4/f8.
    //
    //   - Fallback path (FallbackCompressed / Uncompressed): bytes
    //     are in the *physical* representation already.  For a float
    //     HDU that's raw f4/f8 in FITS big-endian; we need to decode
    //     at the float bytepix (not 4) and skip the dequant step.
    //     For an int HDU it's the same int bytes the primary would
    //     have produced.
    let tile_n_pixels: usize = actual_shape.iter()
        .product::<u64>() as usize;
    let is_float_output = output_zbitpix < 0;
    let float_bytepix: u32 = match output_zbitpix {
        -32 => 4,
        -64 => 8,
        _ => 0, // unused for integer-output HDUs
    };
    // dequant runs iff the primary column produced bytes AND
    // quantization is in play.  Three cases drop into the
    // "physical bytes" handling below (no dequant, decode at the
    // float bytepix for float HDUs):
    //   - primary column + unquantized float HDU (ZQUANTIZ='NONE'
    //     or missing ZSCALE/ZZERO)
    //   - fallback column (lossless GZIP fallback) on any HDU
    //   - uncompressed column on any HDU
    let dequant_applies = matches!(payload, TilePayload::PrimaryCompressed { .. })
        && quant.is_some();

    // Decoder bytepix / stored_zbitpix depending on whether
    // dequant will fire.  When dequant applies we decode to i32
    // (the quantized representation); otherwise we decode to the
    // physical dtype (float bytepix for float HDUs, int bytepix
    // for int HDUs).
    let (decode_bp, decode_zb) = if dequant_applies {
        (bytepix, stored_zbitpix)
    } else if is_float_output {
        (float_bytepix, output_zbitpix.abs())
    } else {
        (bytepix, stored_zbitpix)
    };

    let decoded_bytes = match payload {
        TilePayload::PrimaryCompressed { bytes, algorithm }
        | TilePayload::FallbackCompressed { bytes, algorithm } => {
            let params = crate::zimage::AlgorithmParams {
                tile_shape_numpy: actual_shape,
                smooth,
            };
            crate::zimage::decode_tile_to_bytes(
                algorithm, &bytes, tile_n_pixels, decode_bp, blocksize,
                decode_zb, params,
            )?
        }
        TilePayload::Uncompressed { mut bytes } => {
            let expected = tile_n_pixels.checked_mul(decode_bp as usize)
                .ok_or_else(|| PyValueError::new_err(
                    "ZIMAGE: tile pixel count * bytepix overflowed usize"
                ))?;
            if bytes.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "ZIMAGE tile {}: UNCOMPRESSED_DATA payload is {} bytes \
                     but expected {} ({} pixels * {} bytes/pixel)",
                    tile_idx, bytes.len(), expected, tile_n_pixels, decode_bp
                )));
            }
            if decode_bp > 1 && !cfg!(target_endian = "big") {
                crate::common::byteswap_in_place(&mut bytes, decode_bp as usize);
            }
            bytes
        }
    };

    let target_bytes = if dequant_applies {
        let ctx = quant.expect("dequant_applies implies quant.is_some()");
        let (scale, zero) = scale_zero
            .expect("dequant_applies implies scale_zero was read");
        let stored_i32 = crate::zimage::quantize::i32_bytes_to_values(
            &decoded_bytes,
        )?;
        // FITS tile_row is 1-based per cfitsio's convention.
        let tile_row_1based = tile_idx + 1;
        match ctx.output_zbitpix {
            -32 => crate::zimage::quantize::dequantize_to_f32(
                &stored_i32, scale, zero, ctx.method,
                tile_row_1based, ctx.zdither0,
            ),
            -64 => crate::zimage::quantize::dequantize_to_f64(
                &stored_i32, scale, zero, ctx.method,
                tile_row_1based, ctx.zdither0,
            ),
            other => return Err(PyValueError::new_err(format!(
                "QuantContext.output_zbitpix={} not in {{-32,-64}}",
                other
            ))),
        }
    } else {
        decoded_bytes
    };

    let arc = Arc::new(target_bytes);
    cache.put(tile_idx, Arc::clone(&arc));
    Ok(arc)
}

// Read one tile's heap payload.  Checks the data columns in
// priority order: primary COMPRESSED_DATA → GZIP_COMPRESSED_DATA
// fallback → UNCOMPRESSED_DATA fallback.  Returns whichever has
// non-zero nelements first, tagged so the caller knows which
// decoder to run.  All present columns' descriptors are read for
// the row; reading is cheap (8 or 16 bytes apiece) and keeping a
// single lock-acquire is simpler than retrying.
// Same as fetch_tile_payload but also reads the per-tile
// ZSCALE/ZZERO doubles under the same file lock acquire.  Returns
// `(payload, Some((scale, zero)))` when `quant` is Some, else
// `(payload, None)`.  Pairing the reads avoids a second lock cycle
// in the hot tile loop.
fn fetch_tile_payload_and_quant(
    file_handle: &FileHandle,
    tile_idx: u64,
    data_offset: u64,
    naxis1: u64,
    theap: u64,
    cols: &DataColumns,
    algorithm: CompressionAlgorithm,
    quant: Option<&QuantContext>,
) -> PyResult<(TilePayload, Option<(f64, f64)>)> {
    let mut guard = file_handle.lock()
        .map_err(|_| PyIOError::new_err("file lock poisoned"))?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let row_offset = data_offset + tile_idx * naxis1;

    let payload = fetch_tile_payload_inner(
        file, tile_idx, data_offset, theap, cols, algorithm, row_offset,
    )?;
    let scale_zero = match quant {
        Some(ctx) => Some((
            read_big_endian_f64(
                file, row_offset + ctx.zscale_offset_in_row,
            )?,
            read_big_endian_f64(
                file, row_offset + ctx.zzero_offset_in_row,
            )?,
        )),
        None => None,
    };
    Ok((payload, scale_zero))
}

// Original payload-only entry point.  Still called from
// fetch_tile_payload_and_quant above; kept as a free fn so the
// payload dispatch logic lives in one place.
#[allow(dead_code)]
fn fetch_tile_payload(
    file_handle: &FileHandle,
    tile_idx: u64,
    data_offset: u64,
    naxis1: u64,
    theap: u64,
    cols: &DataColumns,
    algorithm: CompressionAlgorithm,
) -> PyResult<TilePayload> {
    let mut guard = file_handle.lock()
        .map_err(|_| PyIOError::new_err("file lock poisoned"))?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let row_offset = data_offset + tile_idx * naxis1;
    fetch_tile_payload_inner(
        file, tile_idx, data_offset, theap, cols, algorithm, row_offset,
    )
}

fn fetch_tile_payload_inner(
    file: &mut std::fs::File,
    tile_idx: u64,
    data_offset: u64,
    theap: u64,
    cols: &DataColumns,
    algorithm: CompressionAlgorithm,
    row_offset: u64,
) -> PyResult<TilePayload> {
    // Primary column first.
    let (prim_nelem, prim_off) = read_descriptor(
        file, row_offset + cols.primary.byte_offset_in_row, cols.primary.is_q,
    )?;
    if prim_nelem > 0 {
        let bytes = read_heap_bytes(
            file, data_offset + theap + prim_off,
            prim_nelem.saturating_mul(cols.primary.inner_byte_width),
        )?;
        return Ok(TilePayload::PrimaryCompressed { bytes, algorithm });
    }

    // GZIP fallback — always GZIP_1 decode regardless of the
    // primary algorithm; this is cfitsio's lossless-fallback
    // convention.
    if let Some(gcol) = &cols.gzip_fallback {
        let (nelem, off) = read_descriptor(
            file, row_offset + gcol.byte_offset_in_row, gcol.is_q,
        )?;
        if nelem > 0 {
            let bytes = read_heap_bytes(
                file, data_offset + theap + off,
                nelem.saturating_mul(gcol.inner_byte_width),
            )?;
            return Ok(TilePayload::FallbackCompressed {
                bytes,
                algorithm: CompressionAlgorithm::Gzip1,
            });
        }
    }

    // Uncompressed fallback.
    if let Some(ucol) = &cols.uncompressed_fallback {
        let (nelem, off) = read_descriptor(
            file, row_offset + ucol.byte_offset_in_row, ucol.is_q,
        )?;
        if nelem > 0 {
            let bytes = read_heap_bytes(
                file, data_offset + theap + off,
                nelem.saturating_mul(ucol.inner_byte_width),
            )?;
            return Ok(TilePayload::Uncompressed { bytes });
        }
    }

    Err(PyValueError::new_err(format!(
        "ZIMAGE tile {} has no data in any of COMPRESSED_DATA / \
         GZIP_COMPRESSED_DATA / UNCOMPRESSED_DATA", tile_idx
    )))
}

// Read one big-endian f64 at the given absolute file offset.
// Used for the per-tile ZSCALE / ZZERO (TFORM=`1D`) reads on the
// quantized-float path.
fn read_big_endian_f64(
    file: &mut std::fs::File,
    offset: u64,
) -> PyResult<f64> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(f64::from_be_bytes(buf))
}

// Read one variable-length descriptor at the given absolute file
// offset.  Returns (nelements, heap_offset).  Both `P` (8 bytes,
// two u32s) and `Q` (16 bytes, two u64s) forms are big-endian.
fn read_descriptor(
    file: &mut std::fs::File,
    desc_offset: u64,
    is_q: bool,
) -> PyResult<(u64, u64)> {
    file.seek(SeekFrom::Start(desc_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    if is_q {
        let mut buf = [0u8; 16];
        file.read_exact(&mut buf)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let nelem = u64::from_be_bytes(buf[..8].try_into().unwrap());
        let off = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        Ok((nelem, off))
    } else {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let nelem = u32::from_be_bytes(buf[..4].try_into().unwrap()) as u64;
        let off = u32::from_be_bytes(buf[4..8].try_into().unwrap()) as u64;
        Ok((nelem, off))
    }
}

// Read `n_bytes` of heap payload starting at the absolute file
// offset.  Convenience wrapper for the read_descriptor + heap-read
// pair done in fetch_tile_payload.
fn read_heap_bytes(
    file: &mut std::fs::File,
    heap_byte_offset: u64,
    n_bytes: u64,
) -> PyResult<Vec<u8>> {
    let mut buf = vec![0u8; n_bytes as usize];
    file.seek(SeekFrom::Start(heap_byte_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    file.read_exact(&mut buf)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(buf)
}

// Write a cached/decoded tile's bytes into a region of the output
// ndarray.  Bytes are already in the target dtype in numpy
// C-order; we wrap them as a 1-D numpy view (via np.frombuffer on
// a PyBytes), reshape to the tile's actual shape, and assign to
// the slice of out_arr covering the tile's image-coords range.
fn place_tile_bytes_into_output(
    py: Python<'_>,
    out_arr: &Bound<'_, PyAny>,
    tile_bytes: &[u8],
    target_dtype: &str,
    actual_shape_numpy: &[u64],
    origin_numpy: &[u64],
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let pybytes = PyBytes::new(py, tile_bytes);
    let arr1d = np.call_method1("frombuffer", (pybytes, target_dtype))?;
    let shape_tuple = PyTuple::new(py, actual_shape_numpy)?;
    let reshaped = arr1d.call_method1("reshape", (shape_tuple,))?;

    let slice_objs: Vec<Bound<'_, PySlice>> = origin_numpy
        .iter()
        .zip(actual_shape_numpy.iter())
        .map(|(&o, &s)| PySlice::new(
            py, o as isize, (o + s) as isize, 1,
        ))
        .collect();
    let slice_tuple = PyTuple::new(py, &slice_objs)?;
    out_arr.set_item(slice_tuple, reshaped)?;
    Ok(())
}

// Per-axis overlap between one tile and one slice descriptor.
// Returned indexers are Python objects (PyInt for is_int axes,
// PySlice otherwise) ready to drop into a tuple for numpy
// indexing.  `output_indexer` is `None` for is_int axes because
// those axes don't exist in the output array (they're dropped
// by numpy's standard int-indexing-collapses-axis rule).
struct AxisOverlap<'py> {
    tile_indexer: Bound<'py, PyAny>,
    output_indexer: Option<Bound<'py, PyAny>>,
}

// For a tile at image-range [tile_origin, tile_origin+tile_size)
// on one axis, decide which of the slice's image indices fall in
// the tile, and build the numpy indexers needed to copy that
// region from tile to output.  Returns None when the tile and
// the slice don't overlap on this axis (skip the tile).
fn axis_overlap<'py>(
    py: Python<'py>,
    tile_origin: u64,
    tile_size: u64,
    axis: &AxisSlice,
) -> PyResult<Option<AxisOverlap<'py>>> {
    let tile_end = tile_origin + tile_size;
    if axis.is_int {
        // is_int: single index, axis.start is the indexed coord.
        if axis.start >= tile_origin && axis.start < tile_end {
            let local: i64 = (axis.start - tile_origin) as i64;
            let pyint = local.into_pyobject(py)?.into_any();
            Ok(Some(AxisOverlap {
                tile_indexer: pyint,
                output_indexer: None,
            }))
        } else {
            Ok(None)
        }
    } else {
        // Slice — image indices are start, start+step, ...,
        // start+(count-1)*step.  Find first and last that land in
        // [tile_origin, tile_end).
        if axis.start >= tile_end {
            return Ok(None);
        }
        let last_image_index = axis.start + (axis.count - 1) * axis.step;
        if last_image_index < tile_origin {
            return Ok(None);
        }
        let k_first = if axis.start >= tile_origin {
            0u64
        } else {
            let diff = tile_origin - axis.start;
            (diff + axis.step - 1) / axis.step
        };
        if k_first >= axis.count {
            return Ok(None);
        }
        let k_last_unclamped = (tile_end - 1 - axis.start) / axis.step;
        let k_last = std::cmp::min(k_last_unclamped, axis.count - 1);
        if k_last < k_first {
            return Ok(None);
        }
        let image_first = axis.start + k_first * axis.step;
        let image_last = axis.start + k_last * axis.step;
        let tile_local_first = (image_first - tile_origin) as isize;
        let tile_local_stop = (image_last - tile_origin + 1) as isize;
        let step = axis.step as isize;
        let tile_slice = PySlice::new(
            py, tile_local_first, tile_local_stop, step,
        );
        let output_slice = PySlice::new(
            py, k_first as isize, (k_last + 1) as isize, 1,
        );
        Ok(Some(AxisOverlap {
            tile_indexer: tile_slice.into_any(),
            output_indexer: Some(output_slice.into_any()),
        }))
    }
}

// Top-level entry point for CompressedImageHDU::__getitem__.
// Parses the slice key, walks every tile in the BINTABLE, checks
// per-axis overlap with the slice, and for overlapping tiles
// copies the slice region from the (cached or freshly-decoded)
// tile into the assembled output ndarray.  Always applies
// BSCALE/BZERO; for all-int multi-axis indexes, the 0-d output
// is unwrapped to a numpy scalar.
fn slice_compressed_image(
    py: Python<'_>,
    cards: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    tainted: &TaintFlag,
    cache: &TileCache,
    key: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    check_not_tainted(tainted)?;

    // Algorithm + image-side bitpix + image shape — same
    // validation as the whole-image read path.
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE")
        .ok_or_else(|| PyValueError::new_err(
            "compressed HDU missing ZCMPTYPE"
        ))?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;
    let (zbitpix, image_shape) = parse_compressed_image_shape(cards)?;
    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)"
        ));
    }
    // Same stored vs output ZBITPIX split as the whole-image read
    // path; see read_compressed_image_data for the rationale.
    let stored_zbitpix: i32 = if zbitpix < 0 { 32 } else { zbitpix };
    let default_bytepix: u32 = match stored_zbitpix {
        8 => 1, 16 => 2, 32 => 4, 64 => 8,
        _ => return Err(PyValueError::new_err(format!(
            "unsupported ZBITPIX {}", zbitpix
        ))),
    };
    let (blocksize, bytepix_from_header) = parse_rice_params(cards);
    let blocksize = blocksize.unwrap_or(32);
    let bytepix = bytepix_from_header.unwrap_or(default_bytepix);
    let smooth = parse_hcompress_smooth(cards);

    let naxis1 = parse_keyword(cards, "NAXIS1")
        .ok_or_else(|| PyValueError::new_err("BINTABLE missing NAXIS1"))?
        as u64;
    let naxis2 = parse_keyword(cards, "NAXIS2")
        .ok_or_else(|| PyValueError::new_err("BINTABLE missing NAXIS2"))?
        as u64;
    let theap = parse_keyword(cards, "THEAP")
        .map(|x| x.max(0) as u64)
        .unwrap_or(naxis1 * naxis2);
    let cols = find_data_columns(cards)?;
    // For float ZBITPIX this is Some(ctx) when quantization is in
    // play (ZSCALE/ZZERO present + ZQUANTIZ != 'NONE'); None when
    // the file stores raw unquantized floats.  Always None for
    // integer ZBITPIX.
    let quant = if zbitpix < 0 {
        build_quant_context(cards, &cols)?
    } else {
        None
    };

    let tile_shape = parse_tile_shape(cards, &image_shape);
    let n_tiles = compute_n_tiles(&image_shape, &tile_shape);
    if n_tiles != naxis2 {
        return Err(PyValueError::new_err(format!(
            "ZIMAGE row count NAXIS2={} disagrees with tile count {}",
            naxis2, n_tiles
        )));
    }

    // Parse the slice key.  Fancy lists/arrays raise here via
    // parse_axis_indexer's "unsupported index type" error, matching
    // the ImageHDU surface.
    let slices = normalize_slice_key(key, &image_shape)?;
    let all_int = slices.iter().all(|s| s.is_int);

    // Output shape: drop is_int axes.  All-int → empty shape (0-d
    // ndarray); collapsed to a numpy scalar at the end.
    let output_shape: Vec<u64> = slices.iter()
        .filter(|s| !s.is_int)
        .map(|s| s.count)
        .collect();

    let dtype_str = zbitpix_to_native_dtype(zbitpix)?;
    let np = py.import("numpy")?;
    let shape_tuple = PyTuple::new(py, &output_shape)?;
    let out_arr = np.call_method1("empty", (shape_tuple, dtype_str))?;

    // If any axis has zero-width slice, output is empty — skip the
    // tile walk.  (np.empty with a 0 in the shape returns an empty
    // array of the right shape; that's the desired result.)
    let any_empty = slices.iter().any(|s| s.count == 0);
    if !any_empty {
        for tile_idx in 0..n_tiles {
            let (origin, actual_shape) = tile_origin_and_shape(
                tile_idx, &image_shape, &tile_shape,
            );
            // Per-axis overlap check.  If any axis returns None,
            // the tile doesn't intersect the slice — skip.
            let mut tile_indexers: Vec<Bound<PyAny>> =
                Vec::with_capacity(slices.len());
            let mut output_indexers: Vec<Bound<PyAny>> = Vec::new();
            let mut overlapping = true;
            for (axis_idx, axis) in slices.iter().enumerate() {
                match axis_overlap(
                    py, origin[axis_idx], actual_shape[axis_idx], axis,
                )? {
                    Some(ov) => {
                        tile_indexers.push(ov.tile_indexer);
                        if let Some(out) = ov.output_indexer {
                            output_indexers.push(out);
                        }
                    }
                    None => {
                        overlapping = false;
                        break;
                    }
                }
            }
            if !overlapping {
                continue;
            }

            // Fetch tile bytes (cache or decode), wrap as ndarray.
            let tile_bytes = get_or_decode_tile(
                py, cache, file_handle, tainted, tile_idx, data_offset,
                naxis1, theap, &cols, algorithm, &actual_shape,
                bytepix, blocksize, stored_zbitpix,
                zbitpix, quant.as_ref(), smooth,
            )?;
            let pybytes = PyBytes::new(py, &tile_bytes);
            let arr1d = np.call_method1(
                "frombuffer", (pybytes, dtype_str),
            )?;
            let tile_shape_tuple = PyTuple::new(py, &actual_shape)?;
            let tile_arr = arr1d.call_method1(
                "reshape", (tile_shape_tuple,),
            )?;

            let tile_idx_tuple = PyTuple::new(py, &tile_indexers)?;
            let sub = tile_arr.get_item(tile_idx_tuple)?;
            if all_int {
                // Output is 0-d — assign with empty-tuple key.
                out_arr.set_item(PyTuple::empty(py), sub)?;
            } else {
                let out_idx_tuple = PyTuple::new(py, &output_indexers)?;
                out_arr.set_item(out_idx_tuple, sub)?;
            }
        }
    }

    // Always scale on __getitem__ (matches the table-side
    // convention; user can read(scale=False) for the whole image).
    let unbound = out_arr.unbind();
    let (bscale, bzero) = crate::hdu_image::parse_bscale_bzero(cards);
    let kind = crate::hdu_image::image_scaling_kind(zbitpix, bscale, bzero);
    let scaled = crate::hdu_image::apply_image_scaling(
        py, unbound, zbitpix, kind, bscale, bzero,
    )?;

    if all_int {
        let bound = scaled.bind(py);
        Ok(bound.get_item(PyTuple::empty(py))?.unbind())
    } else {
        Ok(scaled)
    }
}

// ====== Compressed write (Phases 7 + 8) ======
//
// REFACTOR TODO (post-Phase-8): `write_compressed_image_data` has
// grown unwieldy — it branches on int-vs-float in several places,
// embeds a per-tile-row struct, and duplicates descriptor-writing
// logic.  Once the quantize matrix tests are in place (commit 3 of
// Phase 8), refactor into:
//   - `TileRow` promoted to a module-level struct (out of the fn).
//   - `encode_tile_int` / `encode_tile_float` helpers, each
//     returning `(TileRow, primary_bytes, fallback_bytes)`.
//   - The main dispatcher just loops, calls the right helper,
//     accumulates the heap, and emits the descriptor table.
// Don't do it before the dither-matrix tests anchor behavior —
// landing the refactor on top of a known-good baseline is safer.

// Bulk-write entry point for CompressedImageHDU.write.  Encodes every
// tile in RAM, then mutates the file (grow if non-last HDU, write
// descriptors + heap, update PCOUNT card).  Phase 7 supports GZIP_1
// and GZIP_2 with integer ZBITPIX (u1/i2/i4/i8).
//
// Order:
//   1. Parse header for shape, tile shape, algorithm, ZBITPIX, heap_format.
//   2. Validate the input ndarray (shape, dtype).
//   3. Encode all tiles into RAM → descriptors + heap.  (Validate-then-
//      mutate: any error from here back leaves the file untouched.)
//   4. Grow the file if (main_size + heap_size, block-padded) exceeds
//      the current padded data extent.  Use shift_file_tail when not
//      the last HDU; set_len when it is.
//   5. Write descriptors into the main data section + heap bytes
//      immediately after.
//   6. Rewrite the header in place with the updated PCOUNT.
//   7. Commit the in-memory cards and clear the tile cache.
//
// Taint semantics match the rest of the write paths: anything before
// the first file mutation can fail without tainting; failures inside
// the shift / write / header rewrite taint and the user has to
// close+reopen.
#[allow(clippy::too_many_arguments)]
fn write_compressed_image_data(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    cards: &[String],
    offsets: &Arc<HduOffsets>,
    file_handle: &FileHandle,
    layout: &FileLayout,
    tainted: &TaintFlag,
    cache: &TileCache,
    cards_arc: &Arc<Mutex<Vec<String>>>,
    quantize_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::Quantize>>
    >,
) -> PyResult<()> {
    check_not_tainted(tainted)?;

    // ----- header parse -----
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE")
        .ok_or_else(|| PyValueError::new_err(
            "compressed HDU missing ZCMPTYPE"
        ))?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;
    let (zbitpix, image_shape) = parse_compressed_image_shape(cards)?;
    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)"
        ));
    }
    let is_float = zbitpix < 0;
    // For integer ZBITPIX, bytepix matches the image pixel width.
    // For float ZBITPIX, we use the i32 width (4) for the encoded
    // primary stream — quantize_float/_double output i32 values.
    let bytepix: u32 = match zbitpix {
        8 => 1, 16 => 2, 32 => 4, 64 => 8,
        -32 | -64 => 4,
        other => return Err(PyValueError::new_err(format!(
            "unsupported ZBITPIX {} for compressed write", other
        ))),
    };
    let float_bytepix: u32 = match zbitpix {
        -32 => 4,
        -64 => 8,
        _ => 0,
    };
    let tile_shape = parse_tile_shape(cards, &image_shape);
    let n_tiles = compute_n_tiles(&image_shape, &tile_shape);

    // Determine heap_format from TFORM1.  Encoder needs to know
    // 'P' (8-byte descriptors, u32 fields) vs 'Q' (16-byte, u64).
    // Inner type matters too: '1PB'/'1QB' (byte) is used by
    // GZIP/RICE/HCOMPRESS, '1PI'/'1QI' (i16 short) by PLIO.  For
    // VLA descriptors the `nelements` field counts ELEMENTS of the
    // inner type, not bytes — so we divide encoded.len() by
    // `inner_byte_width` when filling descriptors.
    let tform1 = parse_string_keyword(cards, "TFORM1")
        .ok_or_else(|| PyValueError::new_err(
            "compressed HDU missing TFORM1"
        ))?;
    let tform1_trim = tform1.trim();
    let heap_is_q = tform1_trim.starts_with("1Q")
        || tform1_trim.starts_with('Q');
    let descriptor_size: u64 = if heap_is_q { 16 } else { 8 };
    let inner_byte_width: u64 = tform_vla_inner_byte_width(tform1_trim)
        .unwrap_or(1);

    // ----- input ndarray validation -----
    let np = py.import("numpy")?;
    let ascontig = np.call_method1("ascontiguousarray", (data,))?;
    let in_shape: Vec<usize> = ascontig.getattr("shape")?.extract()?;
    let expected_shape: Vec<usize> =
        image_shape.iter().map(|&d| d as usize).collect();
    if in_shape != expected_shape {
        return Err(PyValueError::new_err(format!(
            "compressed image write: input shape {:?} != image shape {:?}",
            in_shape, expected_shape
        )));
    }
    // Convert each tile to the expected on-disk byte order in one
    // pass via numpy.  Integer ZBITPIX → u1/i2/i4/i8; float ZBITPIX
    // → f4/f8 (so we can re-extract the floats for quantization).
    let be_dtype = match zbitpix {
        8  => ">u1",
        16 => ">i2",
        32 => ">i4",
        64 => ">i8",
        -32 => ">f4",
        -64 => ">f8",
        other => return Err(PyValueError::new_err(format!(
            "unsupported ZBITPIX {} for compressed write", other
        ))),
    };

    // ----- algorithm-specific encode params -----
    // RICE_1 needs BLOCKSIZE from the header; HCOMPRESS_1 needs
    // SCALE.  Both keys are emitted by create_compressed_image_hdu_impl
    // so they're present on rustfits-written files; defaults match
    // cfitsio when absent (32 / 0 = lossless).
    let (blocksize_opt, _bytepix_header_opt) = parse_rice_params(cards);
    let hcompress_scale = parse_hcompress_scale(cards);

    // ----- float-only quantize setup -----
    // For float HDUs we need:
    //   - DitherMethod from ZQUANTIZ
    //   - ZDITHER0 seed
    //   - qlevel from the create-time Quantize config (the FITS
    //     spec records method+seed only, not the level)
    let (quant_method, zdither0, qlevel) = if is_float {
        let zq = parse_string_keyword(cards, "ZQUANTIZ");
        let method = crate::zimage::quantize::parse_dither_method(
            zq.as_deref(),
        )?.ok_or_else(|| PyNotImplementedError::new_err(
            "compressed-float write with ZQUANTIZ='NONE' \
             (unquantized) is not yet supported; pass quantize= \
             with a method other than 'none'"
        ))?;
        let zd = parse_keyword(cards, "ZDITHER0").unwrap_or(1).max(1);
        let level = quantize_config.lock()
            .map_err(|_| PyIOError::new_err(
                "quantize config lock poisoned"
            ))?.as_ref().map(|q| q.level).unwrap_or(4.0);
        (Some(method), zd, level)
    } else {
        (None, 0, 0.0)
    };

    // ----- per-tile encode loop -----
    //
    // For each tile we accumulate either:
    //   - (primary stream) bytes appended to `primary_heap`, or
    //   - (lossless fallback) raw float bytes GZIP-compressed and
    //     appended to `fallback_heap`,
    // plus per-row metadata (descriptors + ZSCALE/ZZERO for floats)
    // captured in `rows`.  The two heaps are concatenated to form
    // the on-disk heap after the loop; fallback offsets are then
    // bumped by `primary_heap.len()` so their descriptors point at
    // the correct combined-heap location.
    struct TileRow {
        primary_nelem: u64,
        primary_off: u64,
        zscale: f64,
        zzero: f64,
        fallback_nelem: u64,
        fallback_off: u64,
    }
    let mut rows: Vec<TileRow> = Vec::with_capacity(n_tiles as usize);
    let mut primary_heap: Vec<u8> = Vec::new();
    let mut fallback_heap: Vec<u8> = Vec::new();

    for tile_idx in 0..n_tiles {
        let (origin, actual_shape) = tile_origin_and_shape(
            tile_idx, &image_shape, &tile_shape,
        );
        let slice_objs: Vec<Bound<'_, PySlice>> = origin.iter()
            .zip(actual_shape.iter())
            .map(|(&o, &s)| PySlice::new(
                py, o as isize, (o + s) as isize, 1,
            ))
            .collect();
        let slice_tuple = PyTuple::new(py, &slice_objs)?;
        let tile_view = ascontig.get_item(slice_tuple)?;
        let tile_be = np.call_method1(
            "ascontiguousarray", (tile_view, be_dtype),
        )?;
        let tile_bytes_py = tile_be.call_method0("tobytes")?;
        let tile_bytes: Vec<u8> = tile_bytes_py.extract()?;
        let n_pixels = actual_shape.iter().product::<u64>() as usize;
        let pixel_width_bytes = if is_float {
            float_bytepix as usize
        } else {
            bytepix as usize
        };
        let expected_bytes = n_pixels * pixel_width_bytes;
        if tile_bytes.len() != expected_bytes {
            return Err(PyValueError::new_err(format!(
                "internal: tile {} bytes={} expected {}",
                tile_idx, tile_bytes.len(), expected_bytes
            )));
        }

        if is_float {
            // ---- float quantization path ----
            let method = quant_method.unwrap();
            // Tile dims as passed to quantize_float/_double:
            //   nxpix = numpy-last (fast) axis
            //   nypix = remaining (product of the rest)
            // The noise estimator walks rows of length nxpix.
            let nxpix = actual_shape[actual_shape.len() - 1] as usize;
            let nypix = if nxpix == 0 { 0 } else { n_pixels / nxpix };
            // 1-based tile index drives the dither seed.
            let row_1based = tile_idx + 1;
            let qt_opt = if zbitpix == -32 {
                let mut tile_f32: Vec<f32> = Vec::with_capacity(n_pixels);
                for chunk in tile_bytes.chunks_exact(4) {
                    tile_f32.push(f32::from_be_bytes(
                        chunk.try_into().unwrap(),
                    ));
                }
                crate::zimage::quantize::quantize_float(
                    &tile_f32, nxpix, nypix, None,
                    qlevel, method, row_1based, zdither0,
                )
            } else {
                let mut tile_f64: Vec<f64> = Vec::with_capacity(n_pixels);
                for chunk in tile_bytes.chunks_exact(8) {
                    tile_f64.push(f64::from_be_bytes(
                        chunk.try_into().unwrap(),
                    ));
                }
                crate::zimage::quantize::quantize_double(
                    &tile_f64, nxpix, nypix, None,
                    qlevel, method, row_1based, zdither0,
                )
            };

            if let Some(qt) = qt_opt {
                // Quantized successfully — encode the i32 stream
                // through the chosen algorithm (acts as if the
                // input were a 32-bit integer image).
                let mut i32_be: Vec<u8> = Vec::with_capacity(n_pixels * 4);
                for &v in &qt.idata {
                    i32_be.extend_from_slice(&v.to_be_bytes());
                }
                let encode_params = crate::zimage::AlgorithmEncodeParams {
                    blocksize: blocksize_opt.unwrap_or(32),
                    tile_shape_numpy: &actual_shape,
                    scale: hcompress_scale,
                };
                let encoded = crate::zimage::encode_tile_from_bytes(
                    algorithm, &i32_be, 4 /* bytepix=4 for i32 */,
                    n_pixels, 32 /* zbitpix=32 for the integer encode */,
                    encode_params,
                )?;
                if encoded.len() as u64 % inner_byte_width != 0 {
                    return Err(PyValueError::new_err(format!(
                        "internal: encoded tile {} bytes={} not a \
                         multiple of inner_byte_width={}",
                        tile_idx, encoded.len(), inner_byte_width,
                    )));
                }
                let off = primary_heap.len() as u64;
                let nelem = encoded.len() as u64 / inner_byte_width;
                primary_heap.extend(encoded);
                rows.push(TileRow {
                    primary_nelem: nelem,
                    primary_off: off,
                    zscale: qt.bscale,
                    zzero: qt.bzero,
                    fallback_nelem: 0,
                    fallback_off: 0,
                });
            } else {
                // Couldn't quantize (constant tile, range too
                // wide, etc.) — GZIP-compress the raw float bytes
                // into the lossless fallback column.
                let encoded = crate::zimage::gzip::encode_gzip1(
                    &tile_bytes,
                )?;
                let off = fallback_heap.len() as u64;
                let nelem = encoded.len() as u64;
                fallback_heap.extend(encoded);
                rows.push(TileRow {
                    primary_nelem: 0,
                    primary_off: 0,
                    zscale: 1.0,
                    zzero: 0.0,
                    fallback_nelem: nelem,
                    fallback_off: off,
                });
            }
        } else {
            // ---- integer path ----
            let encode_params = crate::zimage::AlgorithmEncodeParams {
                blocksize: blocksize_opt.unwrap_or(32),
                tile_shape_numpy: &actual_shape,
                scale: hcompress_scale,
            };
            let encoded = crate::zimage::encode_tile_from_bytes(
                algorithm, &tile_bytes, bytepix, n_pixels,
                zbitpix, encode_params,
            )?;
            if encoded.len() as u64 % inner_byte_width != 0 {
                return Err(PyValueError::new_err(format!(
                    "internal: encoded tile {} bytes={} not a multiple \
                     of inner_byte_width={}",
                    tile_idx, encoded.len(), inner_byte_width,
                )));
            }
            let off = primary_heap.len() as u64;
            let nelem = encoded.len() as u64 / inner_byte_width;
            primary_heap.extend(encoded);
            rows.push(TileRow {
                primary_nelem: nelem,
                primary_off: off,
                zscale: 0.0,
                zzero: 0.0,
                fallback_nelem: 0,
                fallback_off: 0,
            });
        }
    }

    // Combine the two heaps.  Bump fallback offsets so they land
    // in the right position of the concatenated heap.
    let primary_size = primary_heap.len() as u64;
    let fallback_size = fallback_heap.len() as u64;
    let total_heap_bytes = primary_size + fallback_size;
    for row in rows.iter_mut() {
        if row.fallback_nelem > 0 {
            row.fallback_off += primary_size;
        }
    }

    // ----- compute file extents -----
    let data_offset = offsets.data_offset();
    // Main data section = NAXIS1 (row width) × n_tiles.  Row width
    // depends on the column layout: 1 VLA descriptor for integer
    // HDUs, primary + ZSCALE + ZZERO + fallback descriptor for
    // floats.
    let row_width: u64 = if is_float {
        descriptor_size + 8 + 8 + descriptor_size
    } else {
        descriptor_size
    };
    let main_bytes = row_width.saturating_mul(n_tiles);
    let new_data_size = main_bytes.saturating_add(total_heap_bytes);
    let new_padded = if new_data_size == 0 {
        0
    } else {
        ((new_data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            * BLOCK_SIZE as u64
    };
    // Current data extent = file size between data_offset and the next
    // HDU's header_offset (or EOF if this is the last HDU).  We can
    // approximate via "is the data section padded enough?" — we know
    // the HDU's allocated size from layout/EOF.
    let current_hdu_end = {
        let guard = layout.hdus.lock()
            .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
        // Find this HDU's index by matching offsets.  We don't have
        // an index passed in, but Arc equality works since they share.
        let mut found_next: Option<u64> = None;
        let mut found_self = false;
        for h in guard.iter() {
            if found_self {
                found_next = Some(h.header_offset());
                break;
            }
            if Arc::ptr_eq(h, offsets) {
                found_self = true;
            }
        }
        match found_next {
            Some(end) => end,
            None => {
                // Last HDU — current_hdu_end is the file length.
                let g = lock_file(file_handle)?;
                let f = g.as_ref()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.metadata()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?
                    .len()
            }
        }
    };
    let current_padded_data = current_hdu_end.saturating_sub(data_offset);
    let new_hdu_end = data_offset + new_padded;

    // ----- grow if needed -----
    if new_padded > current_padded_data {
        let delta = new_padded - current_padded_data;
        // file_len lets us tell if there are bytes after this HDU
        // that need to be shifted forward.
        let file_len = {
            let g = lock_file(file_handle)?;
            let f = g.as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            // Non-last HDU — shift the tail forward; later HDU
            // offsets bump via the shared FileLayout.
            shift_file_tail_and_update_offsets(
                file_handle, layout, current_hdu_end, delta, tainted,
            )?;
        } else {
            // Last HDU — extend the file in place.
            let mut g = lock_file(file_handle)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // ----- write descriptors + heap -----
    {
        let mut g = lock_file(file_handle)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed write: seek to data_offset failed: {}", e))
            })?;
        // Build the per-row descriptor table.  Layout:
        //   Integer: primary descriptor only.
        //   Float:   primary descriptor + ZSCALE (8 BE) + ZZERO (8
        //            BE) + fallback descriptor.
        // P-format descriptors are 8 bytes (u32 nelem BE + u32 off BE);
        // Q-format are 16 bytes (u64 + u64 BE).  cfitsio agrees on
        // both layouts.
        let mut desc_buf: Vec<u8> =
            Vec::with_capacity((row_width * n_tiles) as usize);
        let push_desc = |buf: &mut Vec<u8>, nel: u64, off: u64|
            -> PyResult<()>
        {
            if heap_is_q {
                buf.extend_from_slice(&nel.to_be_bytes());
                buf.extend_from_slice(&off.to_be_bytes());
            } else {
                if nel > u32::MAX as u64 || off > u32::MAX as u64 {
                    return Err(PyValueError::new_err(format!(
                        "compressed write: P-descriptor overflow \
                         (nelem={}, offset={}); use heap_format='Q' \
                         for heaps > 4 GB",
                        nel, off,
                    )));
                }
                buf.extend_from_slice(&(nel as u32).to_be_bytes());
                buf.extend_from_slice(&(off as u32).to_be_bytes());
            }
            Ok(())
        };
        for row in &rows {
            if let Err(e) = push_desc(
                &mut desc_buf, row.primary_nelem, row.primary_off,
            ) {
                tainted.store(true, Ordering::Release);
                return Err(e);
            }
            if is_float {
                desc_buf.extend_from_slice(&row.zscale.to_be_bytes());
                desc_buf.extend_from_slice(&row.zzero.to_be_bytes());
                if let Err(e) = push_desc(
                    &mut desc_buf, row.fallback_nelem, row.fallback_off,
                ) {
                    tainted.store(true, Ordering::Release);
                    return Err(e);
                }
            }
        }
        f.write_all(&desc_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed write: descriptor write failed: {}", e))
        })?;
        // Heap = primary_heap followed by fallback_heap (already
        // composed into one logical stream above).  Write both
        // halves consecutively.
        f.write_all(&primary_heap).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed write: primary heap write failed: {}", e))
        })?;
        f.write_all(&fallback_heap).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed write: fallback heap write failed: {}", e))
        })?;
        // Zero-pad to the block boundary so the data section is FITS-
        // conforming.  Cheap: at most BLOCK_SIZE-1 bytes.
        let written = main_bytes + total_heap_bytes;
        if new_padded > written {
            let pad = vec![0u8; (new_padded - written) as usize];
            f.write_all(&pad).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed write: data pad write failed: {}", e))
            })?;
        }
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed write: flush failed: {}", e))
        })?;
    }

    // ----- header rewrite (PCOUNT update) -----
    let mut cards_guard = cards_arc.lock()
        .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
    let mut new_cards = cards_guard.clone();
    set_pcount_in_cards(&mut new_cards, total_heap_bytes);
    {
        let mut g = lock_file(file_handle)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        // header_offset stays valid; PCOUNT rewrite doesn't change
        // the header block count.
        let header_offset = offsets.header_offset();
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "PCOUNT header write failed: {}; close + reopen", e))
        })?;
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "PCOUNT header flush failed: {}; close + reopen", e))
        })?;
    }
    *cards_guard = new_cards;

    // Any tiles cached from earlier reads are now stale.
    cache.clear();

    Ok(())
}
