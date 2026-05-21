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

use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use lru::LruCache;
use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyValueError};
use pyo3::types::{PyBytes, PySlice, PyTuple};

use crate::common::{
    check_not_tainted, parse_keyword, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, TaintFlag,
};
use crate::hdu::HDU;
use crate::hdu_image::{normalize_slice_key, AxisSlice, ImageHDU};
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
    ) -> PyClassInitializer<Self> {
        let hdu = HDU::new(
            header, index, filename, offsets, layout, file, tainted,
        );
        PyClassInitializer::from(hdu)
            .add_subclass(ImageHDU)
            .add_subclass(CompressedImageHDU {
                cache: Arc::new(TileCache::new(DEFAULT_TILE_CACHE_BYTES)),
            })
    }
}

#[pymethods]
impl CompressedImageHDU {
    // Multi-line repr matching ImageHDU's, with the compression
    // type and tile shape surfaced.  Reports the *uncompressed*
    // image dtype + dims (what the user sees after .read()), not
    // the on-disk BINTABLE.
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (zbitpix, shape) = parse_compressed_image_shape(&cards)?;
        let dtype = zbitpix_to_native_dtype(zbitpix)?;
        let extname = parse_string_keyword(&cards, "EXTNAME");
        let bunit = parse_string_keyword(&cards, "BUNIT");
        let zcmptype = parse_string_keyword(&cards, "ZCMPTYPE")
            .unwrap_or_else(|| "(unknown)".to_string());
        let tile_shape = parse_tile_shape(&cards, &shape);

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
        out.push_str("  compression:\n");
        out.push_str(&format!("    type: {}\n", zcmptype));
        out.push_str(&format!("    tile shape: {:?}\n", tile_shape));
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

    // ZCMPTYPE — e.g. "RICE_1", "GZIP_1", "GZIP_2", "HCOMPRESS_1",
    // "PLIO_1".  Returns None when the keyword is absent (which
    // would be a malformed compressed HDU, but the accessor is
    // tolerant so callers can introspect without crashing).
    #[getter]
    fn compression_type(slf: PyRef<'_, Self>) -> PyResult<Option<String>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_string_keyword(&cards, "ZCMPTYPE"))
    }

    // Per-tile shape in numpy axis order (slowest first).  Picks up
    // the FITS convention defaults when ZTILEn is missing (ZTILE1
    // defaults to ZNAXIS1, others default to 1 → "row tiles").
    #[getter]
    fn tile_shape(
        slf: PyRef<'_, Self>, py: Python<'_>,
    ) -> PyResult<Py<PyTuple>> {
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        let (_, image_shape) = parse_compressed_image_shape(&cards)?;
        let tile_shape = parse_tile_shape(&cards, &image_shape);
        Ok(PyTuple::new(py, &tile_shape)?.unbind())
    }

    // Total number of tiles in the image: product of
    // ceil(NAXISn / TILEn).
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
    #[pyo3(signature = (data, start=None))]
    #[allow(unused_variables)]
    fn write(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        start: Option<Vec<i64>>,
    ) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "writing tile-compressed images is not yet implemented \
             (planned: Phase 7+ of the ZIMAGE roadmap)."
        ))
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
fn compute_n_tiles(image_shape: &[u64], tile_shape: &[u64]) -> u64 {
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

// ====== Phase 2: whole-image read ======
//
// REFACTOR NOTE (revisit when Phase 4+ adds more algorithms):
// Phase 2 only knows about COMPRESSED_DATA.  Two places will
// need to widen when fallback / quantization columns enter:
//
//   1. `find_compressed_data_column` returns *one* column.  When
//      Phase 4 wires GZIP_COMPRESSED_DATA + UNCOMPRESSED_DATA
//      fallbacks (used by the encoder when a tile doesn't
//      benefit from RICE), this becomes a "find all data
//      columns" call returning offsets for whichever are
//      present.  The per-tile read then checks each column's
//      nelements in priority order (primary → GZIP → UNCOMPRESSED)
//      and dispatches to the corresponding decoder.  Empty
//      COMPRESSED_DATA on a row currently raises; that error
//      becomes the "try fallback" path.
//
//   2. `parse_rice_params` is RICE-specific.  When Phase 5 adds
//      quantized floats it needs siblings for ZSCALE/ZZERO/
//      ZBLANK column positions (each is a per-tile column in
//      the BINTABLE, not a ZNAMEn/ZVALn pair).  Likely shape:
//      a `CompressionContext` struct passed through the read
//      loop, holding per-algorithm params plus optional
//      quantization-column offsets.
//
// Both refactors keep the outer flow (tile loop → decode →
// place) intact; they just widen the "what to read per row"
// step.

// Top-level entry point invoked from CompressedImageHDU::read.
// Walks the BINTABLE one tile at a time, decoding each via the
// algorithm-specific code in src/zimage/, and assembling the
// tiles into the output ndarray.  Applies BSCALE/BZERO via the
// shared image-side scaling machinery so a scaled compressed
// HDU returns the same dtype as an equivalent uncompressed one.
//
// Phase 2 limits (will lift later):
//   - RICE_1 only (Phase 4 adds GZIP_1/2, Phase 6 HCOMPRESS/PLIO)
//   - Integer ZBITPIX only (Phase 5 adds quantized floats)
//   - mask_blank=True rejected (ZBLANK handling is a follow-up)
//   - Empty COMPRESSED_DATA on a row → error (GZIP_COMPRESSED_DATA
//     / UNCOMPRESSED_DATA fallback comes with Phase 4)
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

    // Algorithm dispatch — Phase 2 supports RICE_1 only.
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE")
        .ok_or_else(|| PyValueError::new_err(
            "compressed HDU missing ZCMPTYPE"
        ))?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;
    if algorithm != CompressionAlgorithm::Rice1 {
        return Err(PyNotImplementedError::new_err(format!(
            "{} decompression is not yet implemented (Phase 2 \
             supports RICE_1 only; see CLAUDE.md for the roadmap)",
            zcmptype
        )));
    }

    // Image shape + bitpix (image-side).
    let (zbitpix, image_shape) = parse_compressed_image_shape(cards)?;
    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)"
        ));
    }
    if zbitpix < 0 {
        return Err(PyNotImplementedError::new_err(format!(
            "compressed image with ZBITPIX={} (floating point) is \
             not yet supported — this requires per-tile \
             quantization (ZSCALE/ZZERO + dithering), planned for \
             Phase 5 of the ZIMAGE roadmap",
            zbitpix
        )));
    }
    let bytepix: u32 = match zbitpix {
        8 => 1,
        16 => 2,
        32 => 4,
        64 => 8,
        _ => return Err(PyValueError::new_err(format!(
            "unsupported ZBITPIX {}", zbitpix
        ))),
    };

    // RICE parameters: BLOCKSIZE and BYTEPIX from ZNAMEn/ZVALn.
    // Defaults: BLOCKSIZE=32, BYTEPIX = zbitpix/8.
    let (blocksize, bytepix_from_header) = parse_rice_params(cards);
    let blocksize = blocksize.unwrap_or(32);
    let bytepix = bytepix_from_header.unwrap_or(bytepix);

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
    let col = find_compressed_data_column(cards)?;

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
            naxis1, theap, &col, algorithm, &actual_shape,
            bytepix, blocksize, zbitpix,
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

// Where the COMPRESSED_DATA column lives in a BINTABLE row.
struct CompressedDataColumn {
    // Byte offset of this column's first byte within a row.
    byte_offset_in_row: u64,
    // True if descriptors are 'Q' (16 bytes); false for 'P' (8).
    is_q: bool,
}

// Walk TFORMn / TTYPEn to find the COMPRESSED_DATA column.  All
// preceding columns contribute their byte width to the offset.
fn find_compressed_data_column(
    header: &[String],
) -> PyResult<CompressedDataColumn> {
    let tfields = parse_keyword(header, "TFIELDS").unwrap_or(0).max(0) as u64;
    if tfields == 0 {
        return Err(PyValueError::new_err(
            "ZIMAGE BINTABLE has TFIELDS=0"
        ));
    }
    let mut offset: u64 = 0;
    for i in 1..=tfields {
        let ttype = parse_string_keyword(header, &format!("TTYPE{}", i))
            .unwrap_or_default();
        let tform = parse_string_keyword(header, &format!("TFORM{}", i))
            .ok_or_else(|| PyValueError::new_err(format!(
                "ZIMAGE BINTABLE column {} missing TFORM", i
            )))?;
        let width = tform_byte_width(&tform)?;
        if ttype.trim() == "COMPRESSED_DATA" {
            return Ok(CompressedDataColumn {
                byte_offset_in_row: offset,
                is_q: tform_is_q_descriptor(&tform),
            });
        }
        offset += width;
    }
    Err(PyValueError::new_err(
        "ZIMAGE BINTABLE missing COMPRESSED_DATA column"
    ))
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

// Look up a tile in the cache or, on miss, read + decode + cast
// + insert.  Returns the tile's bytes in target (stored) dtype,
// numpy C-order, ready to be wrapped in a numpy ndarray.  The
// returned Arc is shared with the cache; both points keep the
// allocation alive until the consumer drops the reference.
//
// File I/O (descriptor read + heap read) is done under the file
// lock; the lock is released before decode + cast.  The cache
// lock is taken twice (once for `get`, once for `put`) and held
// only across in-memory ops.
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
    col: &CompressedDataColumn,
    algorithm: CompressionAlgorithm,
    actual_shape: &[u64],
    bytepix: u32,
    blocksize: u32,
    zbitpix: i32,
) -> PyResult<Arc<Vec<u8>>> {
    if let Some(arc) = cache.get(tile_idx) {
        return Ok(arc);
    }
    check_not_tainted(tainted)?;

    // Read compressed bytes for this tile (descriptor + heap).
    let compressed = read_tile_compressed_bytes(
        file_handle, tile_idx, data_offset, naxis1, theap, col,
    )?;

    // Decode + cast (no file lock held here).
    let tile_n_pixels: usize = actual_shape.iter()
        .product::<u64>() as usize;
    let pixels = crate::zimage::decode_tile_to_i64(
        algorithm, &compressed, tile_n_pixels, bytepix, blocksize,
    )?;
    let target_bytes = cast_i64_to_target_bytes(&pixels, zbitpix);

    let arc = Arc::new(target_bytes);
    cache.put(tile_idx, Arc::clone(&arc));
    Ok(arc)
}

// Read one tile's compressed bytes from disk: the row's
// COMPRESSED_DATA descriptor, then the heap payload.  File lock
// is held only for the duration of these reads.
fn read_tile_compressed_bytes(
    file_handle: &FileHandle,
    tile_idx: u64,
    data_offset: u64,
    naxis1: u64,
    theap: u64,
    col: &CompressedDataColumn,
) -> PyResult<Vec<u8>> {
    let mut guard = file_handle.lock()
        .map_err(|_| PyIOError::new_err("file lock poisoned"))?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    let desc_offset = data_offset
        + tile_idx * naxis1
        + col.byte_offset_in_row;
    file.seek(SeekFrom::Start(desc_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    let (nelements, heap_offset) = if col.is_q {
        let mut buf = [0u8; 16];
        file.read_exact(&mut buf)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let nelem = u64::from_be_bytes(buf[..8].try_into().unwrap());
        let off = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        (nelem, off)
    } else {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let nelem = u32::from_be_bytes(buf[..4].try_into().unwrap()) as u64;
        let off = u32::from_be_bytes(buf[4..8].try_into().unwrap()) as u64;
        (nelem, off)
    };
    if nelements == 0 {
        return Err(PyNotImplementedError::new_err(format!(
            "ZIMAGE tile {} has empty COMPRESSED_DATA; \
             GZIP_COMPRESSED_DATA / UNCOMPRESSED_DATA fallback is \
             not yet implemented (planned: Phase 4 of the ZIMAGE \
             roadmap)", tile_idx
        )));
    }

    let heap_byte_offset = data_offset + theap + heap_offset;
    let mut compressed = vec![0u8; nelements as usize];
    file.seek(SeekFrom::Start(heap_byte_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    file.read_exact(&mut compressed)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(compressed)
}

// Cast a Vec<i64> of decoded pixel values to bytes in the target
// (stored) dtype, numpy-native byte order.  ZBITPIX must be one
// of the supported integer values (8/16/32/64); float ZBITPIX is
// rejected upstream.
fn cast_i64_to_target_bytes(values: &[i64], zbitpix: i32) -> Vec<u8> {
    match zbitpix {
        8 => {
            let mut out = Vec::with_capacity(values.len());
            for &v in values {
                out.push(v as u8);
            }
            out
        }
        16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for &v in values {
                out.extend_from_slice(&(v as i16).to_ne_bytes());
            }
            out
        }
        32 => {
            let mut out = Vec::with_capacity(values.len() * 4);
            for &v in values {
                out.extend_from_slice(&(v as i32).to_ne_bytes());
            }
            out
        }
        64 => {
            let mut out = Vec::with_capacity(values.len() * 8);
            for &v in values {
                out.extend_from_slice(&v.to_ne_bytes());
            }
            out
        }
        _ => Vec::new(), // unreachable: zbitpix validated upstream
    }
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
    if algorithm != CompressionAlgorithm::Rice1 {
        return Err(PyNotImplementedError::new_err(format!(
            "{} decompression is not yet implemented (Phase 2 \
             supports RICE_1 only)", zcmptype
        )));
    }
    let (zbitpix, image_shape) = parse_compressed_image_shape(cards)?;
    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)"
        ));
    }
    if zbitpix < 0 {
        return Err(PyNotImplementedError::new_err(format!(
            "compressed image with ZBITPIX={} (floating point) is \
             not yet supported (Phase 5)",
            zbitpix
        )));
    }
    let bytepix: u32 = match zbitpix {
        8 => 1,
        16 => 2,
        32 => 4,
        64 => 8,
        _ => return Err(PyValueError::new_err(format!(
            "unsupported ZBITPIX {}", zbitpix
        ))),
    };
    let (blocksize, bytepix_from_header) = parse_rice_params(cards);
    let blocksize = blocksize.unwrap_or(32);
    let bytepix = bytepix_from_header.unwrap_or(bytepix);

    let naxis1 = parse_keyword(cards, "NAXIS1")
        .ok_or_else(|| PyValueError::new_err("BINTABLE missing NAXIS1"))?
        as u64;
    let naxis2 = parse_keyword(cards, "NAXIS2")
        .ok_or_else(|| PyValueError::new_err("BINTABLE missing NAXIS2"))?
        as u64;
    let theap = parse_keyword(cards, "THEAP")
        .map(|x| x.max(0) as u64)
        .unwrap_or(naxis1 * naxis2);
    let col = find_compressed_data_column(cards)?;

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
                naxis1, theap, &col, algorithm, &actual_shape,
                bytepix, blocksize, zbitpix,
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
