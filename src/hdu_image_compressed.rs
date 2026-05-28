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

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyValueError};
use pyo3::types::{PyBytes, PySlice, PyTuple};

use crate::cache::BytesBoundLruCache;
use crate::common::{
    check_not_tainted, lock_file, parse_keyword, parse_string_keyword,
    shift_file_tail_and_update_offsets,
    shift_file_tail_backward_and_update_offsets,
    FileHandle, FileLayout, HduOffsets, RawBuffer, TaintFlag, BLOCK_SIZE,
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
// Per-tile decompressed-bytes cache.  Key is the flat tile index
// (lexicographic over the tile-shape grid).  See
// `crate::cache::BytesBoundLruCache` for the eviction policy and
// concurrency model.
type TileCache = BytesBoundLruCache<u64>;

/// A tile-compressed image HDU (``ZIMAGE=T`` on disk).
///
/// The user-facing surface mirrors :class:`ImageHDU` exactly:
/// every accessor (:attr:`shape`, :attr:`dtype`, :attr:`bitpix`,
/// ...) reports the **uncompressed** image properties, and every
/// I/O method (:meth:`read`, :meth:`write`, :meth:`extend`,
/// ``__getitem__``, ``__setitem__``) operates on the
/// uncompressed pixels.  The tile-compressed storage is an
/// implementation detail.
///
/// Subclasses :class:`ImageHDU` (so ``isinstance(hdu,
/// ImageHDU)`` holds), but overrides every I/O method to handle
/// per-tile (de)compression.  Returned by indexing a
/// :class:`FITS` object at a position containing a ZIMAGE HDU.
///
/// Compression-specific surface beyond the inherited
/// :class:`ImageHDU` API:
///
/// * :attr:`compression` — the algorithm config object (e.g.
///   :class:`Rice1` instance) that would reproduce the on-disk
///   layout.
/// * :attr:`n_tiles` — number of tile chunks on disk.
/// * :attr:`tile_cache_size`, :meth:`set_tile_cache_size`,
///   :attr:`tile_cache_used`, :meth:`clear_tile_cache` — LRU
///   cache controls for decoded tiles.
/// * :meth:`repack` — drop orphaned tile bytes accumulated by
///   ``__setitem__`` / :meth:`extend`.
///
/// Examples
/// --------
/// Read tile-compressed data the same way as uncompressed::
///
///     arr = hdu.read()
///     stamp = hdu[100:200, 50:150]   # decodes only overlapping tiles
///
/// Inspect the compression::
///
///     print(hdu.compression)         # Rice1(tile_shape=[100,100], ...)
///     print(hdu.n_tiles)
///
/// Tune the tile cache for a memory-constrained workflow::
///
///     hdu.set_tile_cache_size(0)     # disable caching
///
/// Notes
/// -----
/// Use :meth:`FITS.create_image_hdu` with ``compress=`` to
/// create a tile-compressed image.  Direct construction from
/// Python is not supported.
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
    // Full compression-config object (Gzip1 / Gzip2 / Rice1 /
    // Hcompress1 / Plio1) as the user passed it to
    // `create_image_hdu(compress=...)`.  Stored so that
    // (a) write-only parameters like `Gzip1(level=...)` survive
    // through write / extend / __setitem__ calls (they're not
    // recoverable from the file), and (b) the `.compression`
    // accessor returns the SAME object the user passed in (same-
    // session round-trip).  For reopened HDUs this is None and
    // `.compression` builds a fresh config from the cards via
    // `build_compression_config`.
    pub(crate) compress_config: Arc<
        Mutex<Option<crate::zimage::compression_config::CompressionConfigKind>>,
    >,
    // Lazily-populated cache of the parsed compressed-image
    // metadata.  See `meta()` for the hot-path accessor; entry is
    // `(version, meta)` where `version` is the value of the
    // base-HDU `cards_version` at the time of the parse.  None
    // until the first call; auto-invalidates on any cards mutation
    // because the next `meta()` call observes a higher version and
    // re-parses.  Wrapped in Arc<Mutex<...>> so concurrent readers
    // briefly serialize on the lock (uncontended in the GIL-held
    // case today, correct for future allow_threads use).
    meta_cache: Arc<Mutex<Option<(u64, Arc<CompressedImageMeta>)>>>,
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
        compress_config: Option<
            crate::zimage::compression_config::CompressionConfigKind,
        >,
    ) -> PyClassInitializer<Self> {
        let hdu = HDU::new(
            header, index, filename, offsets, layout, file, tainted,
        );
        PyClassInitializer::from(hdu)
            .add_subclass(ImageHDU::new_empty_cache())
            .add_subclass(CompressedImageHDU {
                cache: Arc::new(TileCache::new(DEFAULT_TILE_CACHE_BYTES)),
                quantize_config: Arc::new(Mutex::new(quantize_config)),
                compress_config: Arc::new(Mutex::new(compress_config)),
                meta_cache: Arc::new(Mutex::new(None)),
            })
    }

    // Return the parsed-once metadata for this HDU.  Hot-path
    // accessor: one Mutex lock + Acquire version load + (on hit)
    // an Arc clone.  On miss (first call, or any cards mutation
    // since the previous call) takes a header snapshot under the
    // cards mutex and re-parses.  See `CompressedImageMeta` for
    // what's cached and what is intentionally not.
    //
    // Callers reach this method while also needing the base HDU
    // (for offsets/file/etc.) by going through `slf.as_super()
    // .as_super()`, which borrows up the class chain instead of
    // consuming `slf` — keeping both alive for the call.
    pub(crate) fn meta(
        &self, super_: &HDU,
    ) -> PyResult<Arc<CompressedImageMeta>> {
        let cur_version = super_.cards_version
            .load(std::sync::atomic::Ordering::Acquire);
        {
            let cache = self.meta_cache.lock()
                .map_err(|_| PyIOError::new_err("meta cache poisoned"))?;
            if let Some((v, m)) = &*cache {
                if *v == cur_version {
                    return Ok(Arc::clone(m));
                }
            }
        }
        // Miss: re-parse outside the lock so concurrent readers
        // racing the same miss each parse once but only one wins
        // the cache slot — same loss-of-work pattern as the
        // TileCache, acceptable because parsing is cheap (10s of
        // microseconds) compared to the lock contention savings.
        let cards = super_.header_snapshot()?;
        let meta = Arc::new(parse_compressed_image_meta(&cards)?);
        let mut cache = self.meta_cache.lock()
            .map_err(|_| PyIOError::new_err("meta cache poisoned"))?;
        *cache = Some((cur_version, Arc::clone(&meta)));
        Ok(meta)
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
        let super_ = slf.as_super().as_super();
        // Try the cached meta first; fall back to a fresh cards
        // snapshot if the file is degenerate enough that parsing
        // raises (repr must never crash — see the
        // "Degraded fallbacks" doc on this method).
        let cached_meta = slf.meta(super_).ok();
        let cards = super_.header_snapshot()?;
        let (zbitpix, shape): (i32, Vec<u64>) = match &cached_meta {
            Some(m) => (m.zbitpix, m.image_shape.clone()),
            None => parse_compressed_image_shape(&cards)?,
        };
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
    //
    // These mirror ImageHDU's accessors exactly but report the
    // ORIGINAL (uncompressed) image properties — `shape` is the
    // shape pixels would have after `.read()`, NOT the on-disk
    // BINTABLE shape.  Sources: ZBITPIX / ZNAXIS / ZNAXISn cards
    // (the Z-prefixed substitutes the FITS Tile Compression
    // Convention defines for the original image).

    /// Original image dimensions in numpy axis order (slowest
    /// first).  Sourced from ``ZNAXISn``; the user sees what
    /// :meth:`read` would return, not the on-disk BINTABLE
    /// shape.
    #[getter]
    fn shape(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        Ok(PyTuple::new(py, &meta.image_shape)?.unbind())
    }

    /// The numpy dtype :meth:`read` would return.  Sourced from
    /// ``ZBITPIX``.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        let dtype_str = zbitpix_to_native_dtype(meta.zbitpix)?;
        let np = py.import("numpy")?;
        Ok(np.call_method1("dtype", (dtype_str,))?.unbind())
    }

    /// Number of image axes.  Sourced from ``ZNAXIS``.
    #[getter]
    fn ndim(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.image_shape.len())
    }

    /// Total pixel count (product of all ``ZNAXISn``).
    #[getter]
    fn size(slf: PyRef<'_, Self>) -> PyResult<u64> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.image_shape.iter().product())
    }

    /// Raw FITS ``ZBITPIX`` value (the image-side bitpix the
    /// user reads into).  Distinct from ``hdu.header["BITPIX"]``
    /// (which is 8, the BINTABLE BITPIX on disk).
    #[getter]
    fn bitpix(slf: PyRef<'_, Self>) -> PyResult<i32> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.zbitpix)
    }

    // __len__ is a pyo3 slot dunder — no per-method docstring.
    // numpy convention: shape[0] of the original image.
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.image_shape[0] as usize)
    }

    /// ``BUNIT`` header value, or ``None`` when unset.
    /// Informational only.
    #[getter]
    fn unit(slf: PyRef<'_, Self>) -> PyResult<Option<String>> {
        let super_ = slf.as_super().as_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_string_keyword(&cards, "BUNIT"))
    }

    // ----- compression-specific accessors -----

    /// Compression configuration of this HDU.
    ///
    /// Returns the same :class:`Gzip1` / :class:`Gzip2` /
    /// :class:`Rice1` / :class:`Hcompress1` / :class:`Plio1`
    /// pyclass instance that would have been passed to
    /// :meth:`FITS.create_image_hdu`'s ``compress=`` argument to
    /// reproduce the on-disk layout.  Single source of truth for
    /// "how is this HDU compressed"::
    ///
    ///     if isinstance(hdu.compression, rustfits.Rice1):
    ///         print(hdu.compression.blocksize)
    ///
    /// The FITS-spec ``ZCMPTYPE`` string is available as
    /// ``hdu.compression.zcmptype``; tile shape as
    /// ``hdu.compression.tile_shape``.  Two HDUs can be compared
    /// with ``hdu_a.compression == hdu_b.compression``
    /// (field-wise ``__eq__``).
    ///
    /// Notes
    /// -----
    /// For HDUs that were just created in this Python session,
    /// the returned object is the exact config passed to
    /// ``create_image_hdu(..., compress=...)`` — write-only
    /// parameters (like ``Gzip1(level=9)``) round-trip.  For
    /// reopened HDUs, the config is rebuilt from header cards;
    /// the ``level`` field comes back as ``None`` because gzip
    /// framing doesn't preserve the level.
    #[getter]
    fn compression(
        slf: PyRef<'_, Self>, py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        // Prefer the stored config (set at create time from the
        // user's `compress=` argument) so write-only kwargs like
        // `Gzip1(level=9)` round-trip via .compression within the
        // same session.  For reopened HDUs the stored field is None
        // and we fall back to rebuilding from header cards (level
        // not recoverable from disk → comes back as None).
        let stored: Option<crate::zimage::compression_config::CompressionConfigKind> =
            slf.compress_config.lock()
                .map_err(|_| PyIOError::new_err(
                    "compress config lock poisoned"))?
                .clone();
        if let Some(cfg) = stored {
            return compression_config_kind_to_py(py, cfg);
        }
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        build_compression_config(py, &cards)
    }

    /// Number of tile chunks the image is split into on disk.
    ///
    /// Product of ``ceil(ZNAXISn / ZTILEn)`` across all axes.
    /// Storage-layout property (kept here rather than on
    /// :attr:`compression` because it's about how the BINTABLE
    /// is laid out, not the algorithm).
    #[getter]
    fn n_tiles(slf: PyRef<'_, Self>) -> PyResult<u64> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        Ok(meta.n_tiles)
    }

    // ----- tile cache -----

    /// Current cache capacity in bytes.
    ///
    /// Default 32 MiB.  Reads and slicing both consult this
    /// budget — there's no per-call opt-out.  Tune with
    /// :meth:`set_tile_cache_size`.
    #[getter]
    fn tile_cache_size(slf: PyRef<'_, Self>) -> u64 {
        slf.cache.capacity()
    }

    /// Set the cache capacity in bytes.
    ///
    /// Setting to ``0`` disables caching entirely — every decode
    /// is one-shot, useful for memory-constrained workflows.
    /// Shrinking the cap below current usage evicts LRU tiles to
    /// fit.
    fn set_tile_cache_size(&self, bytes: u64) {
        self.cache.set_capacity(bytes);
    }

    /// Bytes currently held in the tile cache.
    ///
    /// Useful for monitoring memory usage and for tests that
    /// assert eviction worked.
    #[getter]
    fn tile_cache_used(slf: PyRef<'_, Self>) -> u64 {
        slf.cache.used_bytes()
    }

    /// Drop every cached tile.
    ///
    /// Keeps :attr:`tile_cache_size` as-is, so subsequent decodes
    /// will repopulate.  Useful right after a one-shot
    /// :meth:`read` to release the tile copies while keeping the
    /// cache configured for later slicing.
    fn clear_tile_cache(&self) {
        self.cache.clear();
    }

    // ----- FITS checksum convention (compressed-image variant) -----
    //
    // Compressed HDUs use ZHECKSUM + ZDATASUM (not the BINTABLE-
    // level CHECKSUM/DATASUM) per the FITS Tile Compression
    // Convention; both compute against the equivalent uncompressed
    // image bytes (astropy uses the same convention).

    /// Compute and store the ``ZDATASUM`` checksum card.
    ///
    /// Computed against the equivalent uncompressed image bytes,
    /// not the on-disk BINTABLE, per the FITS Tile Compression
    /// Convention.  Same manual-refresh contract as
    /// :meth:`ImageHDU.add_datasum` — re-run after
    /// :meth:`write` / :meth:`extend` / ``__setitem__``.
    fn add_datasum(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<()> {
        compressed_add_datasum(slf, py)
    }

    /// Compute and store both ``ZDATASUM`` and ``ZHECKSUM`` cards.
    ///
    /// Same convention as :meth:`add_datasum`.  This is the call
    /// most users want.
    fn add_checksum(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<()> {
        compressed_add_checksum(slf, py)
    }

    /// Verify the stored ``ZDATASUM`` against the equivalent
    /// uncompressed image bytes.
    ///
    /// Returns ``True`` / ``False`` / ``None`` (``None`` means
    /// the card is absent).
    fn verify_datasum(
        slf: PyRef<'_, Self>, py: Python<'_>,
    ) -> PyResult<Option<bool>> {
        compressed_verify_datasum(slf, py)
    }

    fn verify_checksum(
        slf: PyRef<'_, Self>, py: Python<'_>,
    ) -> PyResult<Option<bool>> {
        compressed_verify_checksum(slf, py)
    }

    // ----- overrides for ImageHDU's data-access methods -----
    //
    // These shadow ImageHDU's implementations so the uncompressed
    // read/write paths never run on a compressed BINTABLE.  User-
    // facing semantics are identical to the ImageHDU versions;
    // docstrings here cross-reference the parent and add the
    // compression-specific notes (tile-cache behavior etc.).

    /// Read the (decompressed) image into a numpy array.
    ///
    /// Same signature and semantics as :meth:`ImageHDU.read`:
    /// ``scale`` applies ``BSCALE``/``BZERO``; ``mask_blank``
    /// returns a ``MaskedArray`` on ``ZBLANK`` matches.
    ///
    /// Notes
    /// -----
    /// Tile cache: every tile decoded during the read populates
    /// the LRU cache (subject to :attr:`tile_cache_size`).  This
    /// makes subsequent overlapping slicing reads hit warm tiles.
    /// To run cache-free, set ``set_tile_cache_size(0)`` before
    /// the read, or call :meth:`clear_tile_cache` afterward.
    #[pyo3(signature = (*, scale=true, mask_blank=false))]
    fn read(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        scale: bool,
        mask_blank: bool,
    ) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        read_compressed_image_data(
            py, &meta, super_.offsets.data_offset(),
            &super_.file, &super_.tainted, &slf.cache, scale, mask_blank,
        )
    }

    // __getitem__ is a pyo3 slot dunder — no per-method docstring.
    // Same slice surface as ImageHDU (slice / int / ellipsis per
    // axis, stepped slices, mixed combinations), but decodes only
    // the tiles overlapping the requested region.  Always scales
    // (use .read(scale=False) for raw).  Tile cache populated on
    // every access — see the read() docstring for cache control.
    fn __getitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let super_ = slf.as_super().as_super();
        let meta = slf.meta(super_)?;
        slice_compressed_image(
            py, &meta, super_.offsets.data_offset(),
            &super_.file, &super_.tainted, &slf.cache, key,
        )
    }

    /// Write the entire image (encoding tile-by-tile).
    ///
    /// Same input contract as :meth:`ImageHDU.write` — data must
    /// match the HDU shape exactly.  ``start=`` is rejected
    /// because compressed writes are encode-once per tile;
    /// partial writes don't compose with the tile-encoding
    /// model.  Use ``__setitem__`` for region updates instead.
    ///
    /// Validate-then-mutate: every tile is encoded into RAM
    /// first; only after all encodes succeed does the file grow
    /// + descriptor table + heap get written.  A dtype or shape
    /// error leaves the file untouched.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray or numpy.ma.MaskedArray
    ///     See :meth:`ImageHDU.write` for the dtype rules
    ///     (BITPIX-native, scaled, or f8 for general scaling;
    ///     MaskedArray auto-fills via ``ZBLANK`` or NaN).
    /// start : Any, optional
    ///     Not supported.  Passing anything other than ``None``
    ///     raises ``NotImplementedError``.
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
        let compress_config = Arc::clone(&slf.compress_config);
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        write_compressed_image_data(
            py, data, &cards,
            &super_.offsets,
            &super_.file, &super_.layout, &super_.tainted,
            &cache,
            &super_.header,
            &super_.cards_version,
            &quantize_config,
            &compress_config,
        )
    }

    /// Append new pixels along the slow axis.
    ///
    /// Same shape contract as :meth:`ImageHDU.extend`: ``data``
    /// inner-axis shape must match ``hdu.shape[1:]``; the slow
    /// axis can be any size.
    ///
    /// Unlike :meth:`ImageHDU.extend`, there is no ``start=``
    /// argument — writes to existing tile rows are
    /// ``__setitem__``'s job (which re-encodes the affected
    /// tiles in place).  ``extend`` only appends at the end of
    /// the slow axis.  Partial last tiles are handled: the
    /// boundary tile is decoded, combined with the first portion
    /// of new data, and re-encoded.
    ///
    /// A first ``extend`` on an empty HDU created with
    /// ``create_image_hdu(dtype, (0, ...), compress=...)`` fills
    /// a streaming-write compressed image (every algorithm except
    /// HCOMPRESS_1, which requires every axis ``>= 4`` at create
    /// time).
    ///
    /// Notes
    /// -----
    /// Validate-then-mutate; mid-write I/O failures taint the
    /// file.  Old boundary-tile bytes become orphans; call
    /// :meth:`repack` to reclaim them.
    fn extend(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let cache = Arc::clone(&slf.cache);
        let quantize_config = Arc::clone(&slf.quantize_config);
        let compress_config = Arc::clone(&slf.compress_config);
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        extend_compressed_image_data(
            py,
            data,
            &cards,
            &super_.offsets,
            &super_.file,
            &super_.layout,
            &super_.tainted,
            &cache,
            &super_.header,
            &super_.cards_version,
            &quantize_config,
            &compress_config,
        )
    }

    // __setitem__ is a pyo3 slot dunder — no per-method docstring.
    // Same slice surface as ImageHDU.__setitem__: anything
    // hdu[key] reads, hdu[key] = value writes.  Internally
    // re-encodes every overlapping tile + appends to the heap;
    // old tile bytes become orphans that repack() reclaims.
    fn __setitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let cache = Arc::clone(&slf.cache);
        let quantize_config = Arc::clone(&slf.quantize_config);
        let compress_config = Arc::clone(&slf.compress_config);
        let super_ = slf.into_super().into_super();
        let cards = super_.header_snapshot()?;
        setitem_compressed_image(
            py,
            &cards,
            &super_.offsets,
            &super_.file,
            &super_.layout,
            &super_.tainted,
            &cache,
            &super_.header,
            &super_.cards_version,
            &quantize_config,
            &compress_config,
            key,
            value,
        )
    }

    /// Rebuild the tile-data heap, reclaiming orphan tiles.
    ///
    /// :meth:`extend` (when a partial last tile is re-encoded)
    /// and ``__setitem__`` (every affected tile is re-encoded
    /// and appended to the heap) both leave the old compressed
    /// bytes as orphans referenced by no descriptor.  ``repack``
    /// walks every live descriptor, streams its referenced bytes
    /// into a compact new heap, and rewrites the descriptors to
    /// point at it.  If the heap shrinks, the on-disk file
    /// shrinks too: the last HDU uses ``set_len``, a non-last
    /// HDU shifts the trailing HDUs backward in lockstep.
    ///
    /// Also clears the tile cache (its entries were keyed
    /// against the old heap layout).
    ///
    /// No-op for an already-compact heap.
    fn repack(slf: PyRef<'_, Self>) -> PyResult<()> {
        let cache = Arc::clone(&slf.cache);
        let super_ = slf.into_super().into_super();
        repack_compressed_heap(&super_, &cache)
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

// Compressed-write counterpart of hdu_image.rs::normalize_input_dtype.
// Inspects the HDU's BSCALE/BZERO (regular header cards, NOT
// Z-prefixed) and the input array's dtype:
//
//   - Input dtype matches ZBITPIX (e.g. i2 for ZBITPIX=16):
//     pass through unchanged.
//   - HDU has the unsigned-int trick (BSCALE=1 + BZERO=2^(n-1) or
//     -128 for ZBITPIX=8) AND input dtype matches the SCALED form
//     (e.g. u2 for ZBITPIX=16 + BZERO=32768): reverse_unsigned_trick
//     XORs the sign bit to produce the BITPIX-native bytes the
//     encoder needs.
//   - Otherwise (input doesn't match either): error.
//
// Integer ZBITPIX only — caller is responsible for skipping this
// for float HDUs (where BSCALE/BZERO would be a malformed file
// since the spec forbids them on floating-point arrays).
fn normalize_compressed_input_dtype(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    cards: &[String],
    zbitpix: i32,
) -> PyResult<Py<PyAny>> {
    let dtype = arr.getattr("dtype")?;
    let input_kind: String = dtype.getattr("kind")?.extract()?;
    let input_size: u64 = dtype.getattr("itemsize")?.extract()?;
    let expected = zbitpix_to_native_dtype(zbitpix)?;
    let (expected_kind, expected_size) = match expected {
        "u1" => ("u", 1u64),
        "i2" => ("i", 2),
        "i4" => ("i", 4),
        "i8" => ("i", 8),
        _ => unreachable!("integer ZBITPIX expected"),
    };

    // Fast path: input already in ZBITPIX-native dtype.
    if input_kind == expected_kind && input_size == expected_size {
        return Ok(arr.clone().unbind());
    }

    // Scaled-dtype input?  Check BSCALE/BZERO match the
    // unsigned-int trick for this ZBITPIX.
    let (bscale, bzero) = crate::hdu_image::parse_bscale_bzero(cards);
    let kind = crate::hdu_image::image_scaling_kind(zbitpix, bscale, bzero);
    if matches!(kind, crate::hdu_image::ScalingKind::UnsignedTrick) {
        // Expected scaled dtype is the opposite signedness.
        let (scaled_kind, scaled_size): (&str, u64) = match zbitpix {
            8 => ("i", 1),   // u1 stored ← i1 scaled (BZERO=-128)
            16 => ("u", 2),  // i2 stored ← u2 scaled (BZERO=32768)
            32 => ("u", 4),
            64 => ("u", 8),
            _ => unreachable!("unsigned trick on non-int ZBITPIX"),
        };
        if input_kind == scaled_kind && input_size == scaled_size {
            return crate::hdu_image::reverse_unsigned_trick(
                py, arr, zbitpix,
            );
        }
    }

    Err(PyValueError::new_err(format!(
        "compressed image write: input dtype '{}{}' does not match \
         the HDU's ZBITPIX={} (expected '{}{}'{})",
        input_kind, input_size, zbitpix, expected_kind, expected_size,
        if matches!(kind, crate::hdu_image::ScalingKind::UnsignedTrick) {
            match zbitpix {
                8 => " or scaled 'i1' (BZERO=-128)",
                16 => " or scaled 'u2' (BZERO=32768)",
                32 => " or scaled 'u4' (BZERO=2^31)",
                64 => " or scaled 'u8' (BZERO=2^63)",
                _ => "",
            }
        } else {
            ""
        },
    )))
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
    meta: &CompressedImageMeta,
    data_offset: u64,
    file_handle: &FileHandle,
    tainted: &TaintFlag,
    cache: &TileCache,
    scale: bool,
    mask_blank: bool,
) -> PyResult<Py<PyAny>> {
    check_not_tainted(tainted)?;
    let zbitpix = meta.zbitpix;
    let image_shape = meta.image_shape.as_slice();
    let tile_shape = meta.tile_shape.as_slice();
    let algorithm = meta.algorithm;
    let blocksize = meta.blocksize;
    let bytepix = meta.bytepix;
    let smooth = meta.smooth;
    let naxis1 = meta.naxis1;
    let theap = meta.theap;
    let cols = &meta.cols;
    let quant = meta.quant.as_ref();
    let n_tiles = meta.n_tiles;

    // mask_blank parallels the uncompressed ImageHDU behavior, but
    // uses ZBLANK instead of BLANK (per the FITS Tile Compression
    // Convention).  Forbidden on float ZBITPIX — the spec says
    // BLANK is integer-only, NaN serves the same role for floats.
    // (For quantized-float compressed HDUs, the ZBLANK card stores
    // cfitsio's NaN sentinel value -2147483647 in i32 stored
    // space, but that's separate from the integer-pixel mask
    // semantics; NaN is already preserved on dequantize, so float
    // mask_blank stays rejected here too.)
    if mask_blank && zbitpix < 0 {
        return Err(PyValueError::new_err(format!(
            "mask_blank=True is not valid on float ZBITPIX ({}); the \
             FITS standard forbids BLANK on floating-point arrays \
             (NaN serves that role).  Use mask_blank=False, or \
             post-process with numpy.isnan.",
            zbitpix
        )));
    }

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

    // Allocate output ndarray of the right shape + native dtype.
    let np = py.import("numpy")?;
    let dtype_str = zbitpix_to_native_dtype(zbitpix)?;
    let shape_tuple = PyTuple::new(py, image_shape)?;
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
    let mut origin_buf = [0u64; MAX_NAXIS];
    let mut shape_buf = [0u64; MAX_NAXIS];
    for tile_idx in 0..n_tiles {
        let d = tile_origin_and_shape(
            tile_idx, &image_shape, &tile_shape,
            &mut origin_buf, &mut shape_buf,
        );
        let origin = &origin_buf[..d];
        let actual_shape = &shape_buf[..d];
        let tile_bytes = get_or_decode_tile(
            py, cache, file_handle, tainted, tile_idx, data_offset,
            naxis1, theap, cols, algorithm, actual_shape,
            bytepix, blocksize, stored_zbitpix,
            zbitpix, quant, smooth,
        )?;
        place_tile_bytes_into_output(
            py, &out_arr, &tile_bytes, dtype_str,
            actual_shape, origin,
        )?;
    }

    // Compute the blank mask BEFORE scaling (stored space, per
    // the FITS spec — ZBLANK names the raw on-disk sentinel).
    // For integer ZBITPIX this is a simple `arr == ZBLANK` mask;
    // the float branch was rejected at the top.
    let mask_opt = if mask_blank {
        crate::hdu_image::compute_blank_mask_from_value(
            meta.zblank, &out_arr,
        )?
    } else {
        None
    };

    // Apply BSCALE/BZERO (same dispatch the uncompressed path uses).
    let unbound = out_arr.unbind();
    let scaled = if scale {
        let kind = crate::hdu_image::image_scaling_kind(
            zbitpix, meta.bscale, meta.bzero,
        );
        crate::hdu_image::apply_image_scaling(
            py, unbound, zbitpix, kind, meta.bscale, meta.bzero,
        )?
    } else {
        unbound
    };

    if mask_blank {
        crate::hdu_image::wrap_in_masked_array(py, scaled, mask_opt)
    } else {
        Ok(scaled)
    }
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
pub(crate) struct ZimageColumnInfo {
    byte_offset_in_row: u64,
    is_q: bool,
    inner_byte_width: u64,
}

// All ZIMAGE data columns of interest; the primary heap column is
// required, fallbacks and quantization columns are optional and
// only used in specific code paths (fallbacks when the primary
// tile is empty; ZSCALE/ZZERO when ZBITPIX is float).  Resolved
// in find_data_columns by walking TTYPEn.
pub(crate) struct ZimageDataColumns {
    primary: ZimageColumnInfo,
    gzip_fallback: Option<ZimageColumnInfo>,
    uncompressed_fallback: Option<ZimageColumnInfo>,
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
pub(crate) struct ZimageQuantContext {
    method: crate::zimage::quantize::DitherMethod,
    zdither0: i64,
    zscale_offset_in_row: u64,
    zzero_offset_in_row: u64,
    // ZBITPIX of the *output* float dtype (-32 or -64).  Decoder
    // always works in i32; this picks dequantize_to_f32 vs _f64.
    output_zbitpix: i32,
}

// Parsed-once snapshot of all per-HDU compressed-image metadata
// that the hot paths (read / slice / extend / __setitem__ / repack
// + the inspection accessors) re-derived from the cards Vec on
// every call before Phase 2 of header-cache landed.  Cached on the
// HDU keyed by `cards_version` (see `meta()`), so successive calls
// against an unchanged header pay only a single Mutex lock + integer
// compare + Arc clone.
//
// What is NOT cached here: per-call inputs (slice keys, mask_blank
// flag, scale flag) — those vary across calls and don't belong in
// the meta.  BSCALE / BZERO are cached because they're needed every
// read+slice path.  ZBLANK is NOT cached — only the mask_blank=True
// path consults it, and that's a single keyword lookup.
pub(crate) struct CompressedImageMeta {
    pub(crate) zbitpix: i32,
    pub(crate) image_shape: Vec<u64>,
    pub(crate) tile_shape: Vec<u64>,
    pub(crate) n_tiles: u64,
    pub(crate) algorithm: CompressionAlgorithm,
    // Decoder parameters.
    pub(crate) blocksize: u32,
    pub(crate) bytepix: u32,
    pub(crate) smooth: bool,
    // On-disk BINTABLE layout used by tile lookups.  NAXIS2 is
    // not stored separately because it must equal `n_tiles` (the
    // parser sanity-checks the two).
    pub(crate) naxis1: u64,
    pub(crate) theap: u64,
    pub(crate) cols: ZimageDataColumns,
    // None for integer ZBITPIX, None for unquantized-float HDUs
    // (ZQUANTIZ='NONE' or ZSCALE/ZZERO columns missing).  Some(ctx)
    // when per-tile dequantization needs to run after decode.
    pub(crate) quant: Option<ZimageQuantContext>,
    // Image-side BSCALE/BZERO for output scaling (independent of
    // the decoder; applied by the read path on the assembled array
    // when scale=True).  Defaults are (1.0, 0.0).
    pub(crate) bscale: f64,
    pub(crate) bzero: f64,
    // ZBLANK card value (the integer sentinel for masked pixels);
    // None when absent.  Only the `read(mask_blank=True)` path
    // consults it.  Cached so neither hot path needs cards at all.
    pub(crate) zblank: Option<i64>,
}

// Parse all of the above from the cards Vec.  Same validation rules
// the inlined parsing used (in slice_compressed_image et al.) — just
// in one place instead of every hot-path entry point.
fn parse_compressed_image_meta(cards: &[String]) -> PyResult<CompressedImageMeta> {
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
    let (bscale, bzero) = crate::hdu_image::parse_bscale_bzero(cards);
    let zblank = parse_keyword(cards, "ZBLANK");
    Ok(CompressedImageMeta {
        zbitpix, image_shape, tile_shape, n_tiles, algorithm,
        blocksize, bytepix, smooth,
        naxis1, theap,
        cols, quant, bscale, bzero, zblank,
    })
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
    cols: &ZimageDataColumns,
) -> PyResult<Option<ZimageQuantContext>> {
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
    Ok(Some(ZimageQuantContext {
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
fn find_data_columns(header: &[String]) -> PyResult<ZimageDataColumns> {
    let tfields = parse_keyword(header, "TFIELDS").unwrap_or(0).max(0) as u64;
    if tfields == 0 {
        return Err(PyValueError::new_err(
            "ZIMAGE BINTABLE has TFIELDS=0"
        ));
    }
    let mut primary: Option<ZimageColumnInfo> = None;
    let mut gzip_fallback: Option<ZimageColumnInfo> = None;
    let mut uncompressed_fallback: Option<ZimageColumnInfo> = None;
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
        let info = ZimageColumnInfo {
            byte_offset_in_row: offset,
            is_q: tform_is_q_descriptor(&tform),
            inner_byte_width: tform_vla_inner_byte_width(&tform).unwrap_or(1),
        };
        match ttype.trim() {
            "COMPRESSED_DATA" => primary = Some(info),
            "GZIP_COMPRESSED_DATA" => gzip_fallback = Some(info),
            "UNCOMPRESSED_DATA" => uncompressed_fallback = Some(info),
            // ZSCALE / ZZERO are fixed-width 1D columns — the
            // ZimageColumnInfo's VLA fields are meaningless here, but
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
    Ok(ZimageDataColumns {
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
// Wrap a CompressionConfigKind variant in a Py<PyAny> by handing
// the inner per-algorithm pyclass to PyO3.  Used by the
// `.compression` getter when the stored config (set at create
// time) is present so the user gets back exactly what they passed
// in — including write-only fields like `Gzip1(level=9)` that
// aren't recoverable from the file.
fn compression_config_kind_to_py(
    py: Python<'_>,
    cfg: crate::zimage::compression_config::CompressionConfigKind,
) -> PyResult<Py<PyAny>> {
    use crate::zimage::compression_config::CompressionConfigKind as K;
    match cfg {
        K::Gzip1(g) => Ok(Py::new(py, g)?.into_any()),
        K::Gzip2(g) => Ok(Py::new(py, g)?.into_any()),
        K::Rice1(r) => Ok(Py::new(py, r)?.into_any()),
        K::Hcompress1(h) => Ok(Py::new(py, h)?.into_any()),
        K::Plio1(p) => Ok(Py::new(py, p)?.into_any()),
    }
}

// ---------- ZHECKSUM / ZDATASUM machinery ----------

// Read the entire uncompressed image as FITS-big-endian bytes
// (the conceptual data section the equivalent uncompressed HDU
// would carry).  Padded to BLOCK_SIZE.  Scaling is NOT applied
// — the checksum is over the BITPIX-native stored bytes, same
// representation a `BITPIX=ZBITPIX` uncompressed HDU would
// hold.  For quantized-float HDUs the result is the lossy
// dequantized floats (cfitsio convention).
fn read_uncompressed_image_be_bytes(
    slf: &PyRef<'_, CompressedImageHDU>,
    py: Python<'_>,
) -> PyResult<Vec<u8>> {
    // Get the equivalent (BITPIX-native, native-endian) data via
    // the existing read path with scaling off.
    let super_ = slf.as_super().as_super();
    let meta = slf.meta(super_)?;
    let arr_native = read_compressed_image_data(
        py, &meta, super_.offsets.data_offset(),
        &super_.file, &super_.tainted, &slf.cache,
        false, // scale=False — we want stored-space BITPIX-native
        false, // mask_blank=False — checksum is over raw bytes
    )?;
    let arr = arr_native.bind(py);
    let zbitpix = meta.zbitpix;
    let be_dtype = match zbitpix {
        8 => ">u1",
        16 => ">i2",
        32 => ">i4",
        64 => ">i8",
        -32 => ">f4",
        -64 => ">f8",
        other => {
            return Err(PyValueError::new_err(format!(
                "compressed checksum: unsupported ZBITPIX {}",
                other
            )))
        }
    };
    let np = py.import("numpy")?;
    let be = np.call_method1("ascontiguousarray", (arr, be_dtype))?;
    let raw_bytes: Vec<u8> =
        be.call_method0("tobytes")?.extract()?;
    // Pad to FITS block.
    let mut padded = raw_bytes;
    let pad = crate::hdu_image::round_up_to_block(padded.len() as u64)
        - padded.len() as u64;
    padded.extend(std::iter::repeat(0u8).take(pad as usize));
    Ok(padded)
}

// Build the synthetic header bytes of the *equivalent uncompressed
// image HDU* — i.e., what the header would look like if the same
// image were stored without compression.  Used to compute
// ZHECKSUM: we sum (synthetic_uncompressed_header + uncompressed
// data) and encode the complement.
//
// Cards included (minimum required for a valid IMAGE extension
// header + the cards a reader would care about for round-trip):
//   XTENSION = 'IMAGE'  (compressed HDUs can't be primary)
//   BITPIX   = <ZBITPIX>
//   NAXIS    = <ZNAXIS>
//   NAXISn   = <ZNAXISn>
//   PCOUNT   = 0
//   GCOUNT   = 1
//   BSCALE / BZERO / BLANK   (if present in the BINTABLE header
//                              — unsigned-int trick / BLANK
//                              sentinel propagate to the
//                              uncompressed equivalent)
//   EXTNAME / EXTVER         (if present)
//   DATASUM  = <ZDATASUM placeholder / value>
//   CHECKSUM = <ZHECKSUM placeholder / value>
//   END
fn build_equivalent_uncompressed_header(
    cards: &[String],
    datasum_value: &str,
    checksum_value: &str,
) -> PyResult<Vec<String>> {
    use crate::header::{card_int, card_string, pad_to_card};
    let mut out: Vec<String> = Vec::new();

    let zbitpix: i64 = parse_keyword(cards, "ZBITPIX").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZBITPIX")
    })?;
    let znaxis: i64 = parse_keyword(cards, "ZNAXIS").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZNAXIS")
    })?;

    out.push(card_string("XTENSION", "IMAGE", "image extension"));
    out.push(card_int("BITPIX", zbitpix, "number of bits per data pixel"));
    out.push(card_int("NAXIS", znaxis, "number of data axes"));
    for i in 1..=znaxis {
        let key = format!("ZNAXIS{}", i);
        let v: i64 = parse_keyword(cards, &key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "compressed HDU missing {}", key,
            ))
        })?;
        out.push(card_int(
            &format!("NAXIS{}", i), v,
            &format!("length of data axis {}", i),
        ));
    }
    out.push(card_int("PCOUNT", 0, "required keyword; must = 0"));
    out.push(card_int("GCOUNT", 1, "required keyword; must = 1"));

    // Propagate optional integer-scaling cards.
    for key in &["BSCALE", "BZERO", "BLANK"] {
        if let Some(idx) =
            cards.iter().position(|c|
                c.len() >= key.len()
                    && c[..key.len()].trim() == *key)
        {
            // Take the card verbatim — preserves the value
            // formatting (signed int, unsigned int, etc.).
            out.push(cards[idx].trim_end().to_string());
        }
    }
    for key in &["EXTNAME", "EXTVER"] {
        if let Some(idx) =
            cards.iter().position(|c|
                c.len() >= key.len()
                    && c[..key.len()].trim() == *key)
        {
            out.push(cards[idx].trim_end().to_string());
        }
    }

    out.push(card_string(
        "DATASUM", datasum_value, "data unit checksum",
    ));
    out.push(card_string(
        "CHECKSUM", checksum_value, "HDU checksum",
    ));
    out.push(pad_to_card("END"));
    Ok(out)
}

fn compressed_add_datasum(
    slf: PyRef<'_, CompressedImageHDU>, py: Python<'_>,
) -> PyResult<()> {
    let data_bytes = read_uncompressed_image_be_bytes(&slf, py)?;
    let sum = crate::checksum::compute_datasum_of(&data_bytes);
    let super_ = slf.into_super().into_super();
    let cards = super_.header_snapshot()?;
    let new_cards =
        crate::checksum::cards_with_datasum(&cards, sum, "ZDATASUM");
    crate::hdu_image::commit_header_update(&super_, new_cards)
}

fn compressed_add_checksum(
    slf: PyRef<'_, CompressedImageHDU>, py: Python<'_>,
) -> PyResult<()> {
    let data_bytes = read_uncompressed_image_be_bytes(&slf, py)?;
    let datasum = crate::checksum::compute_datasum_of(&data_bytes);
    let super_ = slf.into_super().into_super();
    let cards = super_.header_snapshot()?;
    // ZHECKSUM is computed against the *equivalent uncompressed*
    // header bytes (per the FITS Tile Compression Convention),
    // not the BINTABLE header.  Build that synthetic header,
    // sum it + the uncompressed data, encode the complement,
    // then store the encoded value as ZHECKSUM on the BINTABLE.
    let datasum_str = crate::checksum::format_datasum(datasum);
    let synth_zero = build_equivalent_uncompressed_header(
        &cards, &datasum_str, "0000000000000000",
    )?;
    let synth_bytes =
        crate::hdu_image::serialize_header_to_disk_bytes(&synth_zero);
    let hsum = crate::checksum::compute_checksum_bytes(0, &synth_bytes);
    let total = crate::checksum::ones_complement_add(hsum, datasum);
    let encoded = crate::checksum::encode_checksum_ascii(total, true);
    let encoded_str = std::str::from_utf8(&encoded)
        .expect("encode_checksum_ascii produces printable ASCII");
    // Update the BINTABLE's ZDATASUM and ZHECKSUM cards.
    let mut new_cards = cards.clone();
    crate::checksum::set_or_insert_string_card(
        &mut new_cards, "ZDATASUM", &datasum_str,
        "checksum of uncompressed data",
    );
    crate::checksum::set_or_insert_string_card(
        &mut new_cards, "ZHECKSUM", encoded_str,
        "checksum of equivalent uncompressed HDU",
    );
    crate::hdu_image::commit_header_update(&super_, new_cards)
}

fn compressed_verify_datasum(
    slf: PyRef<'_, CompressedImageHDU>, py: Python<'_>,
) -> PyResult<Option<bool>> {
    let super_ref = slf.as_super().as_super();
    let cards = super_ref.header_snapshot()?;
    let Some(expected_str) = parse_string_keyword(&cards, "ZDATASUM")
    else {
        return Ok(None);
    };
    let Some(expected) =
        crate::checksum::parse_datasum(expected_str.trim())
    else {
        return Ok(None);
    };
    let data_bytes = read_uncompressed_image_be_bytes(&slf, py)?;
    let computed = crate::checksum::compute_datasum_of(&data_bytes);
    Ok(Some(computed == expected))
}

fn compressed_verify_checksum(
    slf: PyRef<'_, CompressedImageHDU>, py: Python<'_>,
) -> PyResult<Option<bool>> {
    let super_ref = slf.as_super().as_super();
    let cards = super_ref.header_snapshot()?;
    let Some(_zhecksum) = parse_string_keyword(&cards, "ZHECKSUM")
    else {
        return Ok(None);
    };
    let Some(zdatasum_str) = parse_string_keyword(&cards, "ZDATASUM")
    else {
        // Spec requires ZDATASUM for the invariant to hold.
        return Ok(Some(false));
    };
    // Re-run the equivalent-uncompressed-HDU sum and check the
    // invariant total == 0xFFFFFFFF.
    let zhecksum_str = parse_string_keyword(&cards, "ZHECKSUM").unwrap();
    let synth = build_equivalent_uncompressed_header(
        &cards, zdatasum_str.trim(), zhecksum_str.trim(),
    )?;
    let synth_bytes =
        crate::hdu_image::serialize_header_to_disk_bytes(&synth);
    let data_bytes = read_uncompressed_image_be_bytes(&slf, py)?;
    let hsum = crate::checksum::compute_checksum_bytes(0, &synth_bytes);
    let total =
        crate::checksum::compute_checksum_bytes(hsum, &data_bytes);
    Ok(Some(total == 0xFFFF_FFFF))
}

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
            // level is not recoverable from the on-disk file —
            // always return None.  Users wanting round-trip equality
            // should compare with `Gzip1(level=None, ...)`.
            let cfg = Gzip1 {
                tile_shape: Some(tile_shape),
                heap_format,
                level: None,
            };
            Ok(Py::new(py, cfg)?.into_any())
        }
        CompressionAlgorithm::Gzip2 => {
            let cfg = Gzip2 {
                tile_shape: Some(tile_shape),
                heap_format,
                level: None,
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

// Upper bound on image dimensionality for the hot-loop stack
// arrays in `tile_origin_and_shape`.  The FITS spec allows ZNAXIS
// up to 999, but real images are 1-4 dims (sometimes 5 for
// hyperspectral cubes); 8 is a comfortable margin.  The function
// asserts d <= MAX_NAXIS so a malformed input fails fast rather
// than silently truncating.
const MAX_NAXIS: usize = 8;

// Given a tile index (0..n_tiles, FITS-row-major), the image
// shape (numpy order), and the nominal tile shape (numpy order),
// fill the caller-provided fixed-size buffers with the tile's
// numpy-order origin and its actual shape (which may be smaller
// than nominal at the image edges).  Returns the active length
// `d`; callers slice the buffers with `[..d]` for the live data.
//
// Stack-allocates everything — no Vec, no allocator round-trip
// per call.  This function runs millions of times on chunked
// reads of large compressed images, where the three `vec![0u64; d]`
// of the previous Vec-returning version dominated 12% of slice
// time on small-chunk workloads.
fn tile_origin_and_shape(
    tile_idx: u64,
    image_shape_numpy: &[u64],
    nominal_tile_shape_numpy: &[u64],
    origin_out: &mut [u64; MAX_NAXIS],
    shape_out: &mut [u64; MAX_NAXIS],
) -> usize {
    let d = image_shape_numpy.len();
    assert!(
        d <= MAX_NAXIS,
        "tile_origin_and_shape: image NAXIS={} exceeds MAX_NAXIS={}",
        d, MAX_NAXIS,
    );
    let mut idx = tile_idx;
    let mut tile_coord = [0u64; MAX_NAXIS];
    // Unfold from numpy-last (= FITS-fastest = varies fastest in
    // the BINTABLE row ordering) to numpy-first.
    for axis_numpy in (0..d).rev() {
        let n_along = (image_shape_numpy[axis_numpy]
            + nominal_tile_shape_numpy[axis_numpy] - 1)
            / nominal_tile_shape_numpy[axis_numpy];
        tile_coord[axis_numpy] = idx % n_along;
        idx /= n_along;
    }
    for axis in 0..d {
        origin_out[axis] = tile_coord[axis]
            * nominal_tile_shape_numpy[axis];
        let end = (origin_out[axis] + nominal_tile_shape_numpy[axis])
            .min(image_shape_numpy[axis]);
        shape_out[axis] = end - origin_out[axis];
    }
    d
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
    cols: &ZimageDataColumns,
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
    quant: Option<&ZimageQuantContext>,
    // HCOMPRESS_1 SMOOTH flag (ignored by every other algorithm).
    smooth: bool,
) -> PyResult<Arc<Vec<u8>>> {
    if let Some(arc) = cache.get(&tile_idx) {
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
                "ZimageQuantContext.output_zbitpix={} not in {{-32,-64}}",
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
    cols: &ZimageDataColumns,
    algorithm: CompressionAlgorithm,
    quant: Option<&ZimageQuantContext>,
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
    cols: &ZimageDataColumns,
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
    cols: &ZimageDataColumns,
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

// Pre-computed per-axis tile-coord range of tiles that overlap a
// slice key.  Used by the slice_compressed_image fast path to
// enumerate ONLY overlapping tiles (instead of walking
// 0..n_tiles and rejecting most of them in the body).  Only
// valid when every axis is a step=1 slice with !is_int — caller
// checks first.
//
// `tc_first[ax]..=tc_first[ax]+tc_extent[ax]-1` is the range of
// overlapping tile coords on axis `ax`; `n_along[ax]` is the
// total tile count along that axis (used by `unfold` to convert
// per-axis coords back into a BINTABLE row-major tile_idx).
struct OverlappingTileRange {
    n_along: [u64; MAX_NAXIS],
    tc_first: [u64; MAX_NAXIS],
    tc_extent: [u64; MAX_NAXIS],
    d_img: usize,
    total: u64,
}

impl OverlappingTileRange {
    fn new(
        image_shape: &[u64],
        tile_shape: &[u64],
        slices: &[AxisSlice],
    ) -> Self {
        let d_img = image_shape.len();
        let mut n_along = [0u64; MAX_NAXIS];
        let mut tc_first = [0u64; MAX_NAXIS];
        let mut tc_extent = [0u64; MAX_NAXIS];
        let mut total = 1u64;
        for ax in 0..d_img {
            n_along[ax] = (image_shape[ax] + tile_shape[ax] - 1)
                / tile_shape[ax];
            let s = &slices[ax];
            debug_assert!(!s.is_int && s.step == 1 && s.count > 0);
            tc_first[ax] = s.start / tile_shape[ax];
            let last_image_idx = s.start + s.count - 1;
            let tc_last = (last_image_idx / tile_shape[ax])
                .min(n_along[ax] - 1);
            tc_extent[ax] = tc_last - tc_first[ax] + 1;
            total *= tc_extent[ax];
        }
        OverlappingTileRange { n_along, tc_first, tc_extent, d_img, total }
    }

    fn total(&self) -> u64 {
        self.total
    }

    // For iter_idx in 0..total(), unfold into per-axis tile coords
    // and return the BINTABLE row-major tile_idx.  Last numpy axis
    // varies fastest inside the (tc_first..=tc_last) box so
    // consecutive iter_idx values land on adjacent BINTABLE rows
    // (best disk locality).
    #[inline]
    fn unfold(
        &self,
        iter_idx: u64,
        tile_coord_out: &mut [u64; MAX_NAXIS],
    ) -> u64 {
        let mut idx = iter_idx;
        for ax in (0..self.d_img).rev() {
            tile_coord_out[ax] = self.tc_first[ax]
                + idx % self.tc_extent[ax];
            idx /= self.tc_extent[ax];
        }
        let mut tile_idx: u64 = 0;
        for ax in 0..self.d_img {
            tile_idx = tile_idx * self.n_along[ax] + tile_coord_out[ax];
        }
        tile_idx
    }
}

// Copy a d-dimensional rectangular subregion from one C-contiguous
// byte buffer to another.  Used by the slice_compressed_image
// fast path to land decoded tile bytes directly into the output
// ndarray's buffer, avoiding the PyBytes::new + frombuffer +
// reshape + set_item round-trip.
//
// Both `dst` and `src` are interpreted as C-contiguous N-D arrays
// of `itemsize` bytes per element.  The innermost axis becomes
// one memcpy per outer-axis coordinate; for d == 1 it's a single
// memcpy.  Stack-only — no heap allocations regardless of d.
fn strided_copy_c_contig_to_c_contig(
    dst: &mut [u8],
    dst_shape: &[u64],
    dst_start: &[u64],
    src: &[u8],
    src_shape: &[u64],
    src_start: &[u64],
    copy_shape: &[u64],
    itemsize: usize,
) {
    let d = dst_shape.len();
    debug_assert_eq!(d, src_shape.len());
    debug_assert_eq!(d, copy_shape.len());
    debug_assert_eq!(d, dst_start.len());
    debug_assert_eq!(d, src_start.len());
    debug_assert!(d >= 1 && d <= MAX_NAXIS);

    let last = d - 1;
    let row_bytes = (copy_shape[last] as usize) * itemsize;

    // C-contiguous byte strides: stride[i] = itemsize * prod(shape[i+1..])
    let mut dst_byte_strides = [0u64; MAX_NAXIS];
    let mut src_byte_strides = [0u64; MAX_NAXIS];
    {
        let mut s_dst = itemsize as u64;
        let mut s_src = itemsize as u64;
        for i in (0..d).rev() {
            dst_byte_strides[i] = s_dst;
            src_byte_strides[i] = s_src;
            s_dst *= dst_shape[i];
            s_src *= src_shape[i];
        }
    }

    // Base offsets at the start corner.
    let mut base_dst_off: usize = 0;
    let mut base_src_off: usize = 0;
    for i in 0..d {
        base_dst_off += (dst_start[i] * dst_byte_strides[i]) as usize;
        base_src_off += (src_start[i] * src_byte_strides[i]) as usize;
    }

    if d == 1 {
        dst[base_dst_off..base_dst_off + row_bytes]
            .copy_from_slice(&src[base_src_off..base_src_off + row_bytes]);
        return;
    }

    // Walk all outer (d-1) axes; innermost is one memcpy of row_bytes.
    let outer_count: u64 = copy_shape[..last].iter().product();
    let mut coord = [0u64; MAX_NAXIS];
    for _ in 0..outer_count {
        let mut dst_off = base_dst_off;
        let mut src_off = base_src_off;
        for ax in 0..last {
            dst_off += (coord[ax] * dst_byte_strides[ax]) as usize;
            src_off += (coord[ax] * src_byte_strides[ax]) as usize;
        }
        dst[dst_off..dst_off + row_bytes]
            .copy_from_slice(&src[src_off..src_off + row_bytes]);
        // Increment coord; innermost outer axis (last-1) varies fastest.
        for ax in (0..last).rev() {
            coord[ax] += 1;
            if coord[ax] < copy_shape[ax] {
                break;
            }
            coord[ax] = 0;
        }
    }
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
    meta: &CompressedImageMeta,
    data_offset: u64,
    file_handle: &FileHandle,
    tainted: &TaintFlag,
    cache: &TileCache,
    key: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    check_not_tainted(tainted)?;
    let zbitpix = meta.zbitpix;
    let image_shape = meta.image_shape.as_slice();
    let tile_shape = meta.tile_shape.as_slice();
    let algorithm = meta.algorithm;
    let blocksize = meta.blocksize;
    let bytepix = meta.bytepix;
    let smooth = meta.smooth;
    let naxis1 = meta.naxis1;
    let theap = meta.theap;
    let cols = &meta.cols;
    let quant = meta.quant.as_ref();
    let n_tiles = meta.n_tiles;
    // Same stored vs output ZBITPIX split as the whole-image read
    // path; see read_compressed_image_data for the rationale.
    let stored_zbitpix: i32 = if zbitpix < 0 { 32 } else { zbitpix };

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
    // Fast path: every axis is a contiguous slice with step=1 and
    // none are int-collapse.  In this regime we can land tile bytes
    // directly into the output ndarray's buffer with a strided
    // memcpy, skipping the per-tile PyBytes::new + frombuffer +
    // reshape + set_item round-trip (which dominated the slice path
    // on small-chunk workloads — see "Performance" in CLAUDE.md).
    // Stepped slicing and int-collapse axes take the slow path
    // below which still uses numpy's set_item for scatter semantics.
    let fast_path = !any_empty
        && slices.iter().all(|s| !s.is_int && s.step == 1);
    if fast_path {
        let out_itemsize = (zbitpix.unsigned_abs() / 8) as usize;
        let mut out_buf = RawBuffer::acquire_writable(&out_arr)?;
        let dst_bytes = out_buf.as_mut_slice();
        let d_img = image_shape.len();
        let range = OverlappingTileRange::new(
            &image_shape, &tile_shape, &slices,
        );
        let mut tile_coord = [0u64; MAX_NAXIS];
        let mut origin_buf = [0u64; MAX_NAXIS];
        let mut shape_buf = [0u64; MAX_NAXIS];
        let mut tile_start_buf = [0u64; MAX_NAXIS];
        let mut out_start_buf = [0u64; MAX_NAXIS];
        let mut copy_shape_buf = [0u64; MAX_NAXIS];
        for iter_idx in 0..range.total() {
            let tile_idx = range.unfold(iter_idx, &mut tile_coord);
            // One pass: origin + actual_shape + per-axis overlap
            // (tile-local + output-local start + copy count).
            // Overlap is guaranteed non-empty for every axis by
            // the tc_first/tc_extent pre-computation, so the
            // saturating arithmetic always produces a positive
            // copy count.
            for ax in 0..d_img {
                origin_buf[ax] = tile_coord[ax] * tile_shape[ax];
                let tile_end = (origin_buf[ax] + tile_shape[ax])
                    .min(image_shape[ax]);
                shape_buf[ax] = tile_end - origin_buf[ax];
                let s = &slices[ax];
                let slice_end = s.start + s.count;
                let overlap_start = s.start.max(origin_buf[ax]);
                let overlap_end = slice_end.min(tile_end);
                tile_start_buf[ax] = overlap_start - origin_buf[ax];
                out_start_buf[ax] = overlap_start - s.start;
                copy_shape_buf[ax] = overlap_end - overlap_start;
            }
            let actual_shape = &shape_buf[..d_img];
            let tile_bytes = get_or_decode_tile(
                py, cache, file_handle, tainted, tile_idx, data_offset,
                naxis1, theap, cols, algorithm, actual_shape,
                bytepix, blocksize, stored_zbitpix,
                zbitpix, quant, smooth,
            )?;
            strided_copy_c_contig_to_c_contig(
                dst_bytes, &output_shape, &out_start_buf[..d_img],
                &tile_bytes, actual_shape, &tile_start_buf[..d_img],
                &copy_shape_buf[..d_img], out_itemsize,
            );
        }
    } else if !any_empty {
        let mut origin_buf = [0u64; MAX_NAXIS];
        let mut shape_buf = [0u64; MAX_NAXIS];
        for tile_idx in 0..n_tiles {
            let d = tile_origin_and_shape(
                tile_idx, &image_shape, &tile_shape,
                &mut origin_buf, &mut shape_buf,
            );
            let origin = &origin_buf[..d];
            let actual_shape = &shape_buf[..d];
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
                naxis1, theap, cols, algorithm, actual_shape,
                bytepix, blocksize, stored_zbitpix,
                zbitpix, quant, smooth,
            )?;
            let pybytes = PyBytes::new(py, &tile_bytes);
            let arr1d = np.call_method1(
                "frombuffer", (pybytes, dtype_str),
            )?;
            let tile_shape_tuple = PyTuple::new(py, actual_shape)?;
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
    let kind = crate::hdu_image::image_scaling_kind(
        zbitpix, meta.bscale, meta.bzero,
    );
    let scaled = crate::hdu_image::apply_image_scaling(
        py, unbound, zbitpix, kind, meta.bscale, meta.bzero,
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
// Layout: `write_compressed_image_data` is the top-level
// dispatcher.  It parses the header, loops over tiles calling
// either `encode_tile_int` or `encode_tile_float`, then grows the
// file and writes the descriptor table + heap.  The per-tile
// helpers return a `TileRow` (descriptor + per-tile ZSCALE/ZZERO
// + optional GZIP fallback descriptor) and mutate the heap
// buffers in place.

// Per-tile row data captured during the encode loop.  For integer
// HDUs only `primary_nelem` / `primary_off` are meaningful (the
// other fields stay at their default values).  For float HDUs
// either the primary fields are non-zero (tile quantized cleanly)
// OR the fallback fields are non-zero (tile went to the GZIP
// fallback column).  `zscale` / `zzero` are the per-tile
// quantization parameters; they're meaningless when the fallback
// path fires but get written anyway since the row width is fixed.
struct TileRow {
    primary_nelem: u64,
    primary_off: u64,
    zscale: f64,
    zzero: f64,
    fallback_nelem: u64,
    fallback_off: u64,
}

// HDU-invariant context for the integer-tile encode helper.
// Everything that doesn't vary tile-to-tile.
struct IntTileCtx {
    algorithm: crate::zimage::CompressionAlgorithm,
    bytepix: u32,
    zbitpix: i32,
    inner_byte_width: u64,
    blocksize: u32,
    hcompress_scale: i32,
    // None → codec default (zlib level 6).  Only consulted by
    // GZIP_1 / GZIP_2 encoders; other algorithms ignore it.
    gzip_level: Option<u32>,
}

// HDU-invariant context for the float-tile encode helper.
// Carries the noise-estimation knobs (qlevel) and dither state
// (method + zdither0) alongside the algorithm parameters.
struct FloatTileCtx {
    algorithm: crate::zimage::CompressionAlgorithm,
    zbitpix: i32, // -32 or -64
    inner_byte_width: u64,
    blocksize: u32,
    hcompress_scale: i32,
    method: crate::zimage::quantize::DitherMethod,
    qlevel: f64,
    zdither0: i64,
    // GZIP level for the GZIP_1 lossless fallback path (and for
    // the primary algo when it's GZIP_1/GZIP_2).  None → codec
    // default (level 6).
    gzip_level: Option<u32>,
}

// Encode one integer tile: run it through the chosen algorithm,
// append the encoded bytes to `primary_heap`, return the row's
// descriptor.  The float fields of TileRow stay at their defaults
// (caller knows to ignore them for integer HDUs).
fn encode_tile_int(
    ctx: &IntTileCtx,
    tile_bytes: &[u8],
    tile_idx: u64,
    actual_shape: &[u64],
    n_pixels: usize,
    primary_heap: &mut Vec<u8>,
) -> PyResult<TileRow> {
    let encode_params = crate::zimage::AlgorithmEncodeParams {
        blocksize: ctx.blocksize,
        tile_shape_numpy: actual_shape,
        scale: ctx.hcompress_scale,
        gzip_level: ctx.gzip_level,
    };
    let encoded = crate::zimage::encode_tile_from_bytes(
        ctx.algorithm, tile_bytes, ctx.bytepix, n_pixels,
        ctx.zbitpix, encode_params,
    )?;
    if encoded.len() as u64 % ctx.inner_byte_width != 0 {
        return Err(PyValueError::new_err(format!(
            "internal: encoded tile {} bytes={} not a multiple \
             of inner_byte_width={}",
            tile_idx, encoded.len(), ctx.inner_byte_width,
        )));
    }
    let off = primary_heap.len() as u64;
    let nelem = encoded.len() as u64 / ctx.inner_byte_width;
    primary_heap.extend(encoded);
    Ok(TileRow {
        primary_nelem: nelem,
        primary_off: off,
        zscale: 0.0,
        zzero: 0.0,
        fallback_nelem: 0,
        fallback_off: 0,
    })
}

// Encode one float tile.  Convert big-endian float bytes to
// native floats, run quantize_float / quantize_double, then either:
//   - encode the quantized i32 stream through the chosen algorithm
//     and append to `primary_heap`, OR
//   - GZIP-compress the raw float bytes (lossless) and append to
//     `fallback_heap` (the "couldn't quantize" path).
// Per-pixel NaN handling: NaN inputs become NULL_VALUE_I32 in the
// quantized stream regardless of dither method (cfitsio's
// convention).  Exact zeros become ZERO_VALUE_I32 under DITHER_2.
fn encode_tile_float(
    ctx: &FloatTileCtx,
    tile_bytes: &[u8],
    tile_idx: u64,
    actual_shape: &[u64],
    n_pixels: usize,
    primary_heap: &mut Vec<u8>,
    fallback_heap: &mut Vec<u8>,
) -> PyResult<TileRow> {
    // Quantize.  nxpix = numpy-last (fast) axis; nypix = the rest.
    // 1-based tile index drives the dither seed.
    let nxpix = actual_shape[actual_shape.len() - 1] as usize;
    let nypix = if nxpix == 0 { 0 } else { n_pixels / nxpix };
    let row_1based = tile_idx + 1;
    let qt_opt = if ctx.zbitpix == -32 {
        let mut tile_f32: Vec<f32> = Vec::with_capacity(n_pixels);
        for chunk in tile_bytes.chunks_exact(4) {
            tile_f32.push(f32::from_be_bytes(chunk.try_into().unwrap()));
        }
        crate::zimage::quantize::quantize_float(
            &tile_f32, nxpix, nypix, Some(f32::NAN),
            ctx.qlevel, ctx.method, row_1based, ctx.zdither0,
        )
    } else {
        let mut tile_f64: Vec<f64> = Vec::with_capacity(n_pixels);
        for chunk in tile_bytes.chunks_exact(8) {
            tile_f64.push(f64::from_be_bytes(chunk.try_into().unwrap()));
        }
        crate::zimage::quantize::quantize_double(
            &tile_f64, nxpix, nypix, Some(f64::NAN),
            ctx.qlevel, ctx.method, row_1based, ctx.zdither0,
        )
    };

    if let Some(qt) = qt_opt {
        // Quantized successfully — encode the i32 stream through
        // the chosen algorithm (acts as if the input were a 32-bit
        // integer image).
        let mut i32_be: Vec<u8> = Vec::with_capacity(n_pixels * 4);
        for &v in &qt.idata {
            i32_be.extend_from_slice(&v.to_be_bytes());
        }
        let encode_params = crate::zimage::AlgorithmEncodeParams {
            blocksize: ctx.blocksize,
            tile_shape_numpy: actual_shape,
            scale: ctx.hcompress_scale,
            gzip_level: ctx.gzip_level,
        };
        let encoded = crate::zimage::encode_tile_from_bytes(
            ctx.algorithm, &i32_be, 4, n_pixels, 32,
            encode_params,
        )?;
        if encoded.len() as u64 % ctx.inner_byte_width != 0 {
            return Err(PyValueError::new_err(format!(
                "internal: encoded tile {} bytes={} not a multiple \
                 of inner_byte_width={}",
                tile_idx, encoded.len(), ctx.inner_byte_width,
            )));
        }
        let off = primary_heap.len() as u64;
        let nelem = encoded.len() as u64 / ctx.inner_byte_width;
        primary_heap.extend(encoded);
        Ok(TileRow {
            primary_nelem: nelem,
            primary_off: off,
            zscale: qt.bscale,
            zzero: qt.bzero,
            fallback_nelem: 0,
            fallback_off: 0,
        })
    } else {
        // Couldn't quantize (constant tile, range too wide, etc.)
        // — GZIP-compress the raw float bytes into the lossless
        // fallback column.  Primary descriptor stays empty so the
        // reader falls through to the GZIP fallback.
        let encoded =
            crate::zimage::gzip::encode_gzip1(tile_bytes, ctx.gzip_level)?;
        let off = fallback_heap.len() as u64;
        let nelem = encoded.len() as u64;
        fallback_heap.extend(encoded);
        Ok(TileRow {
            primary_nelem: 0,
            primary_off: 0,
            zscale: 1.0,
            zzero: 0.0,
            fallback_nelem: nelem,
            fallback_off: off,
        })
    }
}

// Encode a boundary tile for the quantized-float extend / __setitem__
// path.  The tile has already been decoded + combined-with-new-data
// upstream; this helper does the requantize-and-encode step.
//
// Dispatches on whether the existing tile was stored in the primary
// column (had a non-empty primary descriptor before the call) or in
// the GZIP fallback (primary nelem == 0).  Tiles stored in primary
// stay in primary — re-quantized with the EXISTING per-tile bscale/
// bzero so unchanged pixels round-trip exactly.  Tiles stored in
// the fallback stay in the fallback — re-encoded as GZIP_1 raw
// float bytes (the fallback always works, so we don't try to
// promote them back to primary).
//
// `combined_be` is the full re-formed tile in FITS big-endian float
// bytes (f4 or f8 matching `zbitpix`).  Returns a TileRow ready for
// descriptor emission; encoded bytes are appended to the right
// heap buffer (`primary_heap` for primary path, `fallback_heap`
// for fallback path).
#[allow(clippy::too_many_arguments)]
fn encode_quant_boundary_tile(
    algorithm: crate::zimage::CompressionAlgorithm,
    zbitpix: i32,
    inner_byte_width: u64,
    blocksize: u32,
    hcompress_scale: i32,
    gzip_level: Option<u32>,
    old_main_buf: &[u8],
    tile_idx: u64,
    row_width: u64,
    descriptor_size: u64,
    heap_is_q: bool,
    quant_setup: (
        crate::zimage::quantize::DitherMethod,
        i64,
        f64,
    ),
    combined_be: &[u8],
    actual_shape: &[u64],
    n_pixels: usize,
    primary_heap: &mut Vec<u8>,
    fallback_heap: &mut Vec<u8>,
) -> PyResult<TileRow> {
    let row_at = (tile_idx * row_width) as usize;
    let (old_p_nel, _old_p_off, old_zscale, old_zzero, _, _) =
        read_quant_descriptor_row(
            old_main_buf, row_at, heap_is_q, descriptor_size,
        );
    let (method, zdither0, _qlevel) = quant_setup;
    let row_1based = tile_idx + 1;

    if old_p_nel > 0 {
        // Was in primary: requantize with the existing per-tile
        // bscale/bzero (no compounding loss on unchanged pixels)
        // and encode via the algorithm.  Reject if any value
        // doesn't fit the existing scale.
        let i32_stream = if zbitpix == -32 {
            let floats: Vec<f32> = combined_be
                .chunks_exact(4)
                .map(|c| f32::from_be_bytes(c.try_into().unwrap()))
                .collect();
            crate::zimage::quantize::requantize_float_fixed_scale(
                &floats, old_zscale, old_zzero, method, row_1based,
                zdither0,
            )
        } else {
            let floats: Vec<f64> = combined_be
                .chunks_exact(8)
                .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
                .collect();
            crate::zimage::quantize::requantize_double_fixed_scale(
                &floats, old_zscale, old_zzero, method, row_1based,
                zdither0,
            )
        }
        .map_err(PyValueError::new_err)?;
        let mut i32_be: Vec<u8> = Vec::with_capacity(i32_stream.len() * 4);
        for &v in &i32_stream {
            i32_be.extend_from_slice(&v.to_be_bytes());
        }
        let encode_params = crate::zimage::AlgorithmEncodeParams {
            blocksize,
            tile_shape_numpy: actual_shape,
            scale: hcompress_scale,
            gzip_level,
        };
        let encoded = crate::zimage::encode_tile_from_bytes(
            algorithm, &i32_be, 4, n_pixels, 32, encode_params,
        )?;
        let primary_off_start = primary_heap.len() as u64;
        let nelem = encoded.len() as u64 / inner_byte_width;
        primary_heap.extend(encoded);
        Ok(TileRow {
            primary_nelem: nelem,
            primary_off: primary_off_start,
            zscale: old_zscale,
            zzero: old_zzero,
            fallback_nelem: 0,
            fallback_off: 0,
        })
    } else {
        // Was in fallback: stay in fallback (lossless raw float
        // bytes, GZIP_1).  ZSCALE/ZZERO are placeholder.
        let encoded =
            crate::zimage::gzip::encode_gzip1(combined_be, gzip_level)?;
        let fb_off = fallback_heap.len() as u64;
        let nelem = encoded.len() as u64;
        fallback_heap.extend(encoded);
        Ok(TileRow {
            primary_nelem: 0,
            primary_off: 0,
            zscale: 1.0,
            zzero: 0.0,
            fallback_nelem: nelem,
            fallback_off: fb_off,
        })
    }
}

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
    cards_version: &Arc<AtomicU64>,
    quantize_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::Quantize>>
    >,
    compress_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::CompressionConfigKind>>
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

    // GZIP compression level (write-only param not recoverable
    // from the on-disk file).  When the user passed Gzip1/Gzip2
    // with an explicit level, the full config sits in
    // compress_config (see HduKind::CompressedImage); read out
    // just the level here.  For reopened HDUs or non-GZIP
    // algorithms the value is None → encoder uses codec default
    // (zlib level 6).
    let gzip_level: Option<u32> = compress_config
        .lock()
        .map_err(|_| PyIOError::new_err(
            "compress config lock poisoned",
        ))?
        .as_ref()
        .and_then(|c| c.gzip_level());
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

    // ----- MaskedArray entry -----
    // If `data` is a numpy.ma.MaskedArray, fill masked positions
    // with the appropriate sentinel (NaN for float HDUs; ZBLANK
    // from the header for integer HDUs).  No-op for plain ndarrays.
    let unmasked =
        crate::hdu_image::unwrap_masked_input(py, data, cards, true)?;
    let data = unmasked.bind(py);

    // ----- input ndarray validation -----
    let np = py.import("numpy")?;
    let ascontig0 = np.call_method1("ascontiguousarray", (data,))?;
    let in_shape: Vec<usize> = ascontig0.getattr("shape")?.extract()?;
    let expected_shape: Vec<usize> =
        image_shape.iter().map(|&d| d as usize).collect();
    if in_shape != expected_shape {
        return Err(PyValueError::new_err(format!(
            "compressed image write: input shape {:?} != image shape {:?}",
            in_shape, expected_shape
        )));
    }

    // ----- unsigned-int trick reverse-transform -----
    // If the HDU has BSCALE=1 + BZERO=2^(n-1) (or -128 for i1) AND
    // the input dtype is the scaled form (u2/u4/u8 for BITPIX
    // 16/32/64, i1 for BITPIX 8), reverse-transform via XOR so the
    // encoder receives the on-disk (BITPIX-native) bytes.  Fast
    // path: input already matches BITPIX → pass through unchanged.
    // Float HDUs (is_float) never trigger this path — BSCALE/BZERO
    // are only checked for integer ZBITPIX.
    let ascontig_owned = if !is_float {
        normalize_compressed_input_dtype(
            py, &ascontig0, cards, zbitpix,
        )?
    } else {
        ascontig0.clone().unbind()
    };
    let ascontig = ascontig_owned.bind(py);

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

    // ----- detect unquantized-float mode -----
    // Signal: float HDU + single-column schema (TFIELDS=1).
    // Works for both freshly-created HDUs (we set TFIELDS=1 at
    // create time when quantize=None) and reopened files.
    // Unquantized-float files store raw GZIP-compressed float
    // bytes directly in COMPRESSED_DATA — same single-column
    // layout as integer HDUs.  Astropy's quantize_level=0 output
    // matches this exactly.
    let tfields = parse_keyword(cards, "TFIELDS").unwrap_or(0) as u64;
    let is_unquantized_float = is_float && tfields == 1;
    let is_quantized_float = is_float && !is_unquantized_float;

    // ----- float-only quantize setup -----
    // For quantized float HDUs we need:
    //   - DitherMethod from ZQUANTIZ
    //   - ZDITHER0 seed
    //   - qlevel from the create-time Quantize config (the FITS
    //     spec records method+seed only, not the level)
    // Skipped entirely for unquantized floats: there's no dither
    // stream and no quantization step.
    let (quant_method, zdither0, qlevel) = if is_quantized_float {
        let zq = parse_string_keyword(cards, "ZQUANTIZ");
        let method = crate::zimage::quantize::parse_dither_method(
            zq.as_deref(),
        )?.ok_or_else(|| PyValueError::new_err(
            "quantized-float HDU has ZQUANTIZ='NONE' but the schema \
             has 4 columns (ZSCALE/ZZERO present).  Malformed file."
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
    // Three dispatch modes:
    //   - Integer HDU OR unquantized float: route through
    //     `encode_tile_int` (single-column primary heap, no
    //     ZSCALE/ZZERO, no fallback).  For unquantized floats,
    //     bytepix is the float width (4 or 8); GZIP_1/GZIP_2
    //     treat the bytes opaquely.
    //   - Quantized float: route through `encode_tile_float`
    //     which may quantize → primary_heap OR fall back to
    //     GZIP-compressed raw float bytes → fallback_heap.
    // HDU-invariant params travel in IntTileCtx / FloatTileCtx,
    // leaving this loop to just extract tile bytes and call the
    // right helper.
    let int_ctx = if !is_quantized_float {
        // For unquantized floats, use float_bytepix; for integer
        // HDUs, use the integer bytepix.
        let effective_bytepix = if is_unquantized_float {
            float_bytepix
        } else {
            bytepix
        };
        Some(IntTileCtx {
            algorithm,
            bytepix: effective_bytepix,
            zbitpix,
            inner_byte_width,
            blocksize: blocksize_opt.unwrap_or(32),
            hcompress_scale,
            gzip_level,
        })
    } else {
        None
    };
    let float_ctx = if is_quantized_float {
        Some(FloatTileCtx {
            algorithm,
            zbitpix,
            inner_byte_width,
            blocksize: blocksize_opt.unwrap_or(32),
            hcompress_scale,
            method: quant_method.unwrap(),
            qlevel,
            zdither0,
            gzip_level,
        })
    } else {
        None
    };

    let mut rows: Vec<TileRow> = Vec::with_capacity(n_tiles as usize);
    let mut primary_heap: Vec<u8> = Vec::new();
    let mut fallback_heap: Vec<u8> = Vec::new();

    let mut origin_buf = [0u64; MAX_NAXIS];
    let mut shape_buf = [0u64; MAX_NAXIS];
    for tile_idx in 0..n_tiles {
        let d = tile_origin_and_shape(
            tile_idx, &image_shape, &tile_shape,
            &mut origin_buf, &mut shape_buf,
        );
        let origin = &origin_buf[..d];
        let actual_shape = &shape_buf[..d];
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

        let row = if is_quantized_float {
            encode_tile_float(
                float_ctx.as_ref().unwrap(),
                &tile_bytes, tile_idx, actual_shape, n_pixels,
                &mut primary_heap, &mut fallback_heap,
            )?
        } else {
            // Integer HDU OR unquantized float — both route
            // through encode_tile_int.  Unquantized float feeds
            // raw float bytes through GZIP_1/GZIP_2 (the encoder
            // treats them opaquely).
            encode_tile_int(
                int_ctx.as_ref().unwrap(),
                &tile_bytes, tile_idx, actual_shape, n_pixels,
                &mut primary_heap,
            )?
        };
        rows.push(row);
    }

    // Combine the two heaps.  Bump fallback offsets so they land
    // in the right position of the concatenated heap.  Only the
    // quantized-float path produces a non-empty fallback heap;
    // for integer + unquantized-float paths fallback_heap is
    // empty and this loop is a no-op.
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
    // Main data section = NAXIS1 (row width) × n_tiles.  Row
    // width depends on the column layout: single descriptor for
    // integer HDUs and unquantized-float HDUs (both TFIELDS=1);
    // primary + ZSCALE + ZZERO + fallback descriptor for
    // quantized-float HDUs (TFIELDS=4).
    let row_width: u64 = if is_quantized_float {
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
            if is_quantized_float {
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
    let cards_guard = crate::hdu::CardsWriteGuard::from_parts(
        cards_arc.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?,
        cards_version,
    );
    let mut new_cards = cards_guard.clone_cards();
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
    cards_guard.commit(new_cards);

    // Any tiles cached from earlier reads are now stale.
    cache.clear();

    Ok(())
}

// Update an existing standard-keyword int card to a new value in
// place, preserving its position and (best-effort) comment text.
// No-op if the card isn't present — caller must guarantee it
// exists, which holds for the cards we update at extend time
// (NAXIS2 / PCOUNT / ZNAXISn are all mandatory).
fn update_int_card(cards: &mut Vec<String>, key: &str, value: i64) {
    let key_uc = key.to_uppercase();
    if let Some(idx) = cards.iter().position(|c| {
        c.len() >= 8 && c[..8].trim().to_uppercase() == key_uc
    }) {
        // Best-effort preserve the comment after the '/'.
        let existing = &cards[idx];
        let comment = existing
            .find('/')
            .map(|p| existing[p + 1..].trim().to_string())
            .unwrap_or_default();
        cards[idx] = crate::header::card_int(key, value, &comment);
    }
}

// Append new rows along the slow axis (numpy axis 0) of a
// tile-compressed image.  Existing tiles outside the last tile row
// are preserved untouched; the partial last tile row (if any) gets
// re-encoded to absorb new data; truly new tile rows are encoded
// fresh.  See CLAUDE.md "Compressed extend" for the full design.
//
// On-disk mechanics:
//   1. Pre-read existing main descriptor table + heap into RAM (we
//      have to move them; reading first means an I/O failure here
//      leaves the file unchanged).
//   2. Update boundary tile descriptors in the in-memory main buf
//      to point at re-encoded heap bytes (which live in the
//      appended portion of the new heap).
//   3. Grow file (shift later HDUs if non-last; set_len if last).
//   4. Write:
//        data_offset                        : updated main table
//        data_offset + new_main_bytes       : old heap (relocated)
//        + old_pcount                       : appended heap (boundary
//                                              re-encoded + new
//                                              tile bytes)
//      + block-padding.
//   5. Rewrite header to update NAXIS2 + PCOUNT + ZNAXIS<last>.
//
// Heap layout note: the old boundary-tile bytes stay in the old
// heap (now orphaned — descriptors no longer point at them).
// Slightly bloats files when boundary re-encoding happens, but
// keeps the file logically valid and is dramatically simpler than
// rewriting the whole heap with the old tiles' bytes removed.
#[allow(clippy::too_many_arguments)]
fn extend_compressed_image_data(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    cards: &[String],
    offsets: &Arc<HduOffsets>,
    file_handle: &FileHandle,
    layout: &FileLayout,
    tainted: &TaintFlag,
    cache: &TileCache,
    cards_arc: &Arc<Mutex<Vec<String>>>,
    cards_version: &Arc<AtomicU64>,
    quantize_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::Quantize>>,
    >,
    compress_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::CompressionConfigKind>>,
    >,
) -> PyResult<()> {
    check_not_tainted(tainted)?;

    // ----- header parse -----
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZCMPTYPE")
    })?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;
    let (zbitpix, old_image_shape) = parse_compressed_image_shape(cards)?;
    if old_image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)",
        ));
    }
    let is_float = zbitpix < 0;
    let tfields = parse_keyword(cards, "TFIELDS").unwrap_or(0) as u64;

    // GZIP level (write-only param; None → codec default).
    let gzip_level: Option<u32> = compress_config
        .lock()
        .map_err(|_| PyIOError::new_err(
            "compress config lock poisoned",
        ))?
        .as_ref()
        .and_then(|c| c.gzip_level());
    // Three schema dispatches based on (is_float, tfields):
    //   integer:          single-col (TFIELDS=1)
    //   unquantized float: single-col (TFIELDS=1, quantize=None)
    //   quantized float:  4-col (TFIELDS=4) — re-uses existing
    //                     per-tile ZSCALE/ZZERO via
    //                     requantize_*_fixed_scale to avoid
    //                     compounding loss on unchanged pixels
    let is_quantized_float = is_float && tfields == 4;
    let is_unquantized_float = is_float && !is_quantized_float;

    let bytepix: u32 = match zbitpix {
        8 => 1,
        16 => 2,
        32 => 4,
        64 => 8,
        -32 => 4,
        -64 => 8,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported ZBITPIX {} for compressed extend",
                other
            )))
        }
    };

    let tile_shape = parse_tile_shape(cards, &old_image_shape);
    let old_n_tiles = compute_n_tiles(&old_image_shape, &tile_shape);

    let tform1 = parse_string_keyword(cards, "TFORM1").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing TFORM1")
    })?;
    let tform1_trim = tform1.trim();
    let heap_is_q =
        tform1_trim.starts_with("1Q") || tform1_trim.starts_with('Q');
    let descriptor_size: u64 = if heap_is_q { 16 } else { 8 };
    let inner_byte_width: u64 =
        tform_vla_inner_byte_width(tform1_trim).unwrap_or(1);

    let old_pcount =
        parse_keyword(cards, "PCOUNT").unwrap_or(0).max(0) as u64;

    let (blocksize_opt, _) = parse_rice_params(cards);
    let blocksize = blocksize_opt.unwrap_or(32);
    let hcompress_scale = parse_hcompress_scale(cards);

    // Quantized-float HDUs use a 4-column row layout:
    //   primary descriptor + ZSCALE (8 BE) + ZZERO (8 BE) +
    //   fallback descriptor.
    // Integer + unquantized-float HDUs use a single descriptor.
    let row_width: u64 = if is_quantized_float {
        descriptor_size + 8 + 8 + descriptor_size
    } else {
        descriptor_size
    };
    let old_main_bytes = row_width.saturating_mul(old_n_tiles);
    let data_offset = offsets.data_offset();

    // For the quantized-float path we need dither method + seed + qlevel.
    // These are read from the header (ZQUANTIZ / ZDITHER0) plus the
    // create-time Quantize config (the FITS spec records method +
    // seed only, not the qlevel; reopened HDUs fall back to 4.0).
    // None for integer + unquantized-float HDUs.
    let quant_setup: Option<(crate::zimage::quantize::DitherMethod, i64, f64)> =
        if is_quantized_float {
            let zq = parse_string_keyword(cards, "ZQUANTIZ");
            let method = crate::zimage::quantize::parse_dither_method(
                zq.as_deref(),
            )?
            .ok_or_else(|| {
                PyValueError::new_err(
                    "quantized-float HDU has ZQUANTIZ='NONE' but the \
                     4-column schema is present (malformed file).",
                )
            })?;
            let zd =
                parse_keyword(cards, "ZDITHER0").unwrap_or(1).max(1);
            let level = quantize_config
                .lock()
                .map_err(|_| {
                    PyIOError::new_err("quantize config lock poisoned")
                })?
                .as_ref()
                .map(|q| q.level)
                .unwrap_or(4.0);
            Some((method, zd, level))
        } else {
            None
        };

    // ----- MaskedArray entry -----
    let unmasked =
        crate::hdu_image::unwrap_masked_input(py, data, cards, true)?;
    let data = unmasked.bind(py);

    // ----- input validation -----
    let np = py.import("numpy")?;
    let ascontig0 = np.call_method1("ascontiguousarray", (data,))?;
    let in_shape: Vec<usize> = ascontig0.getattr("shape")?.extract()?;
    let naxis = old_image_shape.len();
    if in_shape.len() != naxis {
        return Err(PyValueError::new_err(format!(
            "compressed extend: input has {} axes, HDU has {}",
            in_shape.len(),
            naxis,
        )));
    }
    if in_shape[0] == 0 {
        return Err(PyValueError::new_err(
            "compressed extend: data.shape[0] must be > 0",
        ));
    }
    // Only axis 0 may grow; remaining axes must match exactly.
    for axis in 1..naxis {
        if in_shape[axis] as u64 != old_image_shape[axis] {
            return Err(PyValueError::new_err(format!(
                "compressed extend: data shape[{}]={} != image \
                 shape[{}]={} (only the slow axis (numpy axis 0) \
                 can grow)",
                axis, in_shape[axis], axis, old_image_shape[axis],
            )));
        }
    }

    // Reverse-transform if the HDU has unsigned-int trick BSCALE/BZERO.
    // For unquantized-float and integer-no-trick HDUs this is a pass-
    // through.  Float HDUs never carry BSCALE/BZERO (forbidden by
    // the spec) so we skip the helper.
    let ascontig_owned = if !is_float {
        normalize_compressed_input_dtype(py, &ascontig0, cards, zbitpix)?
    } else {
        ascontig0.clone().unbind()
    };
    let ascontig = ascontig_owned.bind(py);

    // ----- compute new layout -----
    let old_naxis0 = old_image_shape[0];
    let added = in_shape[0] as u64;
    let new_naxis0 = old_naxis0 + added;
    let mut new_image_shape = old_image_shape.clone();
    new_image_shape[0] = new_naxis0;
    let new_n_tiles = compute_n_tiles(&new_image_shape, &tile_shape);

    let t_r = tile_shape[0];
    let n_old_tile_rows_slow = (old_naxis0 + t_r - 1) / t_r;
    let n_tiles_per_row_slow: u64 = {
        let mut prod = 1u64;
        for ax in 1..naxis {
            let n_along = (old_image_shape[ax] + tile_shape[ax] - 1)
                / tile_shape[ax];
            prod *= n_along;
        }
        prod
    };

    // Boundary tiles: the LAST tile row of the OLD image is a
    // boundary iff old_naxis0 is not a multiple of T_r.  Those
    // tiles need re-encoding because their actual shape grows in
    // the NEW image (old partial → fuller or full).  When
    // old_naxis0 % T_r == 0 there are no boundary tiles.
    let has_boundary = old_naxis0 > 0 && old_naxis0 % t_r != 0;
    let boundary_range: Option<(u64, u64)> = if has_boundary {
        let start = (n_old_tile_rows_slow - 1) * n_tiles_per_row_slow;
        let end = n_old_tile_rows_slow * n_tiles_per_row_slow;
        Some((start, end))
    } else {
        None
    };
    // Tiles in [first_new_tile, new_n_tiles) are entirely new (no
    // overlap with any old tile).  Tiles in [0, first_new_tile) are
    // either unchanged or boundary.
    let first_new_tile = match boundary_range {
        Some((_, end)) => end,
        None => old_n_tiles,
    };

    // ----- encode setup -----
    let int_ctx = IntTileCtx {
        algorithm,
        bytepix,
        zbitpix,
        inner_byte_width,
        blocksize,
        hcompress_scale,
        gzip_level,
    };
    let be_dtype = match zbitpix {
        8 => ">u1",
        16 => ">i2",
        32 => ">i4",
        64 => ">i8",
        -32 => ">f4",
        -64 => ">f8",
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported ZBITPIX {} for compressed extend",
                other
            )))
        }
    };
    let _ = is_unquantized_float; // (signal kept; used in helpers below)

    // ----- read existing main descriptor table early -----
    // For quantized-float HDUs we need each boundary tile's existing
    // ZSCALE/ZZERO (to reuse the same per-tile quantization scale
    // and avoid compounding loss) AND whether it was stored in the
    // primary or GZIP fallback column.  Both are in main_buf.  For
    // integer/unquantized we don't strictly need it before encoding,
    // but reading early keeps the I/O sequencing simple: all
    // pre-write reads happen first, then encoding (which is pure
    // CPU), then writes.
    let mut old_main_buf: Vec<u8> = vec![0; old_main_bytes as usize];
    if old_main_bytes > 0 {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut old_main_buf).map_err(|e| {
            PyIOError::new_err(format!(
                "compressed extend: read old main: {}",
                e
            ))
        })?;
    }

    // Quantization context (only meaningful for quantized-float HDUs).
    // Used by get_or_decode_tile to dispatch the dequantize step on
    // boundary tile reads.
    let cols_lookup = find_data_columns(cards)?;
    let quant_ctx_opt = if is_quantized_float {
        build_quant_context(cards, &cols_lookup)?
    } else {
        None
    };

    // FloatTileCtx for the quantized-float path's new-tile encoding
    // (mirrors what write_compressed_image_data uses).
    let float_ctx = quant_setup.map(|(method, zd, level)| FloatTileCtx {
        algorithm,
        zbitpix,
        inner_byte_width,
        blocksize,
        hcompress_scale,
        method,
        qlevel: level,
        zdither0: zd,
        gzip_level,
    });

    // ----- encode boundary tiles -----
    // For each boundary tile, decode the existing partial tile via
    // the cache machinery, slice the new data portion that falls
    // into the same tile, concatenate, and re-encode.  Integer +
    // unquantized-float HDUs go through encode_tile_int; quantized-
    // float HDUs go through requantize_with_fixed_scale (preserves
    // the existing per-tile bscale/bzero so unchanged pixels suffer
    // no compounding loss) and may use the GZIP fallback for tiles
    // already stored there.
    let mut appended_heap: Vec<u8> = Vec::new();
    let mut appended_fallback_heap: Vec<u8> = Vec::new();
    let mut boundary_updates: Vec<(u64, TileRow)> = Vec::new();
    let mut new_descs: Vec<TileRow> = Vec::new();

    if let Some((b_start, b_end)) = boundary_range {
        let cols = &cols_lookup;
        let theap = parse_keyword(cards, "THEAP")
            .map(|x| x.max(0) as u64)
            .unwrap_or(row_width * old_n_tiles);
        let mut origin_buf = [0u64; MAX_NAXIS];
        let mut new_shape_buf = [0u64; MAX_NAXIS];
        let mut _scratch_origin = [0u64; MAX_NAXIS];
        let mut old_shape_buf = [0u64; MAX_NAXIS];
        for tile_idx in b_start..b_end {
            let d = tile_origin_and_shape(
                tile_idx, &new_image_shape, &tile_shape,
                &mut origin_buf, &mut new_shape_buf,
            );
            tile_origin_and_shape(
                tile_idx, &old_image_shape, &tile_shape,
                &mut _scratch_origin, &mut old_shape_buf,
            );
            let origin = &origin_buf[..d];
            let new_actual_shape = &new_shape_buf[..d];
            let old_actual_shape = &old_shape_buf[..d];
            // For quantized-float HDUs the decoder runs the
            // dequantize step (i32 stored → f4/f8 physical),
            // returning float bytes.  For integer/unquantized-float
            // it returns BITPIX-native bytes.
            let (dec_bytepix, dec_stored_zbitpix) = if is_quantized_float {
                (4u32, 32i32)
            } else {
                (bytepix, zbitpix)
            };
            let old_bytes_arc = get_or_decode_tile(
                py,
                cache,
                file_handle,
                tainted,
                tile_idx,
                data_offset,
                row_width,
                theap,
                cols,
                algorithm,
                old_actual_shape,
                dec_bytepix,
                blocksize,
                dec_stored_zbitpix,
                zbitpix,
                quant_ctx_opt.as_ref(),
                false,
            )?;

            // Slice the corresponding portion of new data:
            //   axis 0: data rows [0, new_actual_shape[0] - old_actual_shape[0])
            //   axes 1..N: the tile's column extent (origin[ax]..origin[ax]+shape[ax])
            let new_rows_in_tile =
                new_actual_shape[0] - old_actual_shape[0];
            let slice_objs: Vec<Bound<'_, PySlice>> = (0..naxis)
                .map(|ax| {
                    if ax == 0 {
                        PySlice::new(py, 0, new_rows_in_tile as isize, 1)
                    } else {
                        let o = origin[ax] as isize;
                        let s = new_actual_shape[ax] as isize;
                        PySlice::new(py, o, o + s, 1)
                    }
                })
                .collect();
            let slice_tuple = PyTuple::new(py, &slice_objs)?;
            let new_part_view = ascontig.get_item(slice_tuple)?;
            // Convert new portion to BE bytes
            let new_part_be =
                np.call_method1("ascontiguousarray", (new_part_view, be_dtype))?;
            let new_part_bytes_py = new_part_be.call_method0("tobytes")?;
            let new_part_be_bytes: Vec<u8> = new_part_bytes_py.extract()?;

            // Convert old bytes (native-endian, from cache) to BE
            let native_dtype = zbitpix_to_native_dtype(zbitpix)?;
            let old_pyb = PyBytes::new(py, &old_bytes_arc);
            let old_arr_flat =
                np.call_method1("frombuffer", (old_pyb, native_dtype))?;
            let old_shape_tuple = PyTuple::new(py, old_actual_shape)?;
            let old_arr =
                old_arr_flat.call_method1("reshape", (old_shape_tuple,))?;
            let old_be =
                np.call_method1("ascontiguousarray", (old_arr, be_dtype))?;
            let old_be_bytes: Vec<u8> =
                old_be.call_method0("tobytes")?.extract()?;

            // Byte-concat (axis 0 concat in C-order = byte concat
            // because axis 0 is the OUTER dim).
            let mut combined_be: Vec<u8> = Vec::with_capacity(
                old_be_bytes.len() + new_part_be_bytes.len(),
            );
            combined_be.extend_from_slice(&old_be_bytes);
            combined_be.extend_from_slice(&new_part_be_bytes);

            let n_pixels: usize =
                new_actual_shape.iter().product::<u64>() as usize;
            let row = if is_quantized_float {
                encode_quant_boundary_tile(
                    algorithm,
                    zbitpix,
                    inner_byte_width,
                    blocksize,
                    hcompress_scale,
                    gzip_level,
                    &old_main_buf,
                    tile_idx,
                    row_width,
                    descriptor_size,
                    heap_is_q,
                    quant_setup.unwrap(),
                    &combined_be,
                    new_actual_shape,
                    n_pixels,
                    &mut appended_heap,
                    &mut appended_fallback_heap,
                )?
            } else {
                encode_tile_int(
                    &int_ctx,
                    &combined_be,
                    tile_idx,
                    new_actual_shape,
                    n_pixels,
                    &mut appended_heap,
                )?
            };
            boundary_updates.push((tile_idx, row));
        }
    }

    // ----- encode truly-new tile rows -----
    let mut new_origin_buf = [0u64; MAX_NAXIS];
    let mut new_shape_buf2 = [0u64; MAX_NAXIS];
    for tile_idx in first_new_tile..new_n_tiles {
        let d = tile_origin_and_shape(
            tile_idx, &new_image_shape, &tile_shape,
            &mut new_origin_buf, &mut new_shape_buf2,
        );
        let origin = &new_origin_buf[..d];
        let new_actual_shape = &new_shape_buf2[..d];
        // The tile's data lives entirely in the input array,
        // offset by old_naxis0 along axis 0.
        let data_axis0_start = origin[0].saturating_sub(old_naxis0);
        let data_axis0_end = data_axis0_start + new_actual_shape[0];
        let slice_objs: Vec<Bound<'_, PySlice>> = (0..naxis)
            .map(|ax| {
                if ax == 0 {
                    PySlice::new(
                        py,
                        data_axis0_start as isize,
                        data_axis0_end as isize,
                        1,
                    )
                } else {
                    let o = origin[ax] as isize;
                    let s = new_actual_shape[ax] as isize;
                    PySlice::new(py, o, o + s, 1)
                }
            })
            .collect();
        let slice_tuple = PyTuple::new(py, &slice_objs)?;
        let tile_view = ascontig.get_item(slice_tuple)?;
        let tile_be =
            np.call_method1("ascontiguousarray", (tile_view, be_dtype))?;
        let tile_bytes_py = tile_be.call_method0("tobytes")?;
        let tile_bytes: Vec<u8> = tile_bytes_py.extract()?;
        let n_pixels: usize =
            new_actual_shape.iter().product::<u64>() as usize;
        let row = if is_quantized_float {
            // Truly-new tile rows: standard quantize_float (may
            // fall to the GZIP fallback if range too wide).  The
            // existing encode_tile_float helper already handles
            // both paths and writes to the right heap.
            encode_tile_float(
                float_ctx.as_ref().unwrap(),
                &tile_bytes,
                tile_idx,
                new_actual_shape,
                n_pixels,
                &mut appended_heap,
                &mut appended_fallback_heap,
            )?
        } else {
            encode_tile_int(
                &int_ctx,
                &tile_bytes,
                tile_idx,
                new_actual_shape,
                n_pixels,
                &mut appended_heap,
            )?
        };
        new_descs.push(row);
    }

    // ----- combine the two appended heaps -----
    // For quantized-float HDUs the appended heap is
    // [appended_primary; appended_fallback].  Bump fallback offsets
    // by primary size so descriptors point at the right slot.  No-op
    // for integer/unquantized-float (fallback heap is empty).
    let appended_primary_size = appended_heap.len() as u64;
    appended_heap.extend(appended_fallback_heap.drain(..));
    for (_, row) in boundary_updates.iter_mut() {
        if row.fallback_nelem > 0 {
            row.fallback_off += appended_primary_size;
        }
    }
    for row in new_descs.iter_mut() {
        if row.fallback_nelem > 0 {
            row.fallback_off += appended_primary_size;
        }
    }

    // ----- shift descriptor offsets into the combined-heap frame -----
    // appended_heap offsets start at 0 (relative to the start of
    // appended).  The combined heap is [old_heap; appended], so
    // absolute offsets are old_pcount + offset_in_appended.
    for (_, row) in boundary_updates.iter_mut() {
        if row.primary_nelem > 0 {
            row.primary_off += old_pcount;
        }
        if row.fallback_nelem > 0 {
            row.fallback_off += old_pcount;
        }
    }
    for row in new_descs.iter_mut() {
        if row.primary_nelem > 0 {
            row.primary_off += old_pcount;
        }
        if row.fallback_nelem > 0 {
            row.fallback_off += old_pcount;
        }
    }

    // ----- compute new file extents -----
    let new_main_bytes = row_width.saturating_mul(new_n_tiles);
    let new_pcount = old_pcount + appended_heap.len() as u64;
    let new_data_size = new_main_bytes + new_pcount;
    let new_padded = if new_data_size == 0 {
        0
    } else {
        ((new_data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            * BLOCK_SIZE as u64
    };
    let current_hdu_end = {
        let guard = layout
            .hdus
            .lock()
            .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
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
                let g = lock_file(file_handle)?;
                let f = g
                    .as_ref()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.metadata()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?
                    .len()
            }
        }
    };
    let current_padded_data = current_hdu_end.saturating_sub(data_offset);
    let new_hdu_end = data_offset + new_padded;

    // ----- read existing heap into RAM -----
    // (main_buf was already read earlier so quantized boundary
    // tiles could inspect existing ZSCALE/ZZERO.  This step reads
    // the heap so we can relocate it past the new descriptor
    // table.  Done BEFORE the grow so an I/O failure leaves the
    // file untouched.)
    let mut old_heap_buf: Vec<u8> = vec![0; old_pcount as usize];
    if old_pcount > 0 {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset + old_main_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut old_heap_buf).map_err(|e| {
            PyIOError::new_err(format!(
                "compressed extend: read old heap: {}",
                e
            ))
        })?;
    }

    // Apply boundary descriptor updates to old_main_buf in place.
    // Descriptor format: P → u32 nelem BE + u32 off BE (8 bytes);
    // Q → u64 nelem BE + u64 off BE (16 bytes).  For quantized-
    // float HDUs each row is the full 4-column layout (primary +
    // ZSCALE + ZZERO + fallback).
    for (tile_idx, row) in &boundary_updates {
        let desc_offset = (tile_idx * row_width) as usize;
        if is_quantized_float {
            write_quant_descriptor_row(
                &mut old_main_buf,
                desc_offset,
                heap_is_q,
                descriptor_size,
                row.primary_nelem,
                row.primary_off,
                row.zscale,
                row.zzero,
                row.fallback_nelem,
                row.fallback_off,
            )?;
        } else {
            write_descriptor(
                &mut old_main_buf,
                desc_offset,
                heap_is_q,
                row.primary_nelem,
                row.primary_off,
            )?;
        }
    }

    // Serialize new tile descriptors into a buffer.
    let mut new_descs_buf: Vec<u8> = Vec::with_capacity(
        (new_descs.len() as u64 * row_width) as usize,
    );
    for row in &new_descs {
        let mut tmp = vec![0u8; row_width as usize];
        if is_quantized_float {
            write_quant_descriptor_row(
                &mut tmp,
                0,
                heap_is_q,
                descriptor_size,
                row.primary_nelem,
                row.primary_off,
                row.zscale,
                row.zzero,
                row.fallback_nelem,
                row.fallback_off,
            )?;
        } else {
            write_descriptor(
                &mut tmp,
                0,
                heap_is_q,
                row.primary_nelem,
                row.primary_off,
            )?;
        }
        new_descs_buf.extend_from_slice(&tmp);
    }

    // ----- grow the file if needed -----
    if new_padded > current_padded_data {
        let delta = new_padded - current_padded_data;
        let file_len = {
            let g = lock_file(file_handle)?;
            let f = g
                .as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            shift_file_tail_and_update_offsets(
                file_handle,
                layout,
                current_hdu_end,
                delta,
                tainted,
            )?;
        } else {
            let mut g = lock_file(file_handle)?;
            let f = g
                .as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // ----- write the new layout to disk -----
    {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset)).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: seek to data_offset failed: {}",
                e
            ))
        })?;
        // Main descriptor table: [existing (with updated boundary
        // descriptors)] + [new tile descriptors].
        f.write_all(&old_main_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: write old main: {}",
                e
            ))
        })?;
        f.write_all(&new_descs_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: write new descs: {}",
                e
            ))
        })?;
        // Heap at the new position: [old heap (relocated)] +
        // [appended (boundary re-encodes + new tiles)].
        f.seek(SeekFrom::Start(data_offset + new_main_bytes)).map_err(
            |e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed extend: seek to new heap: {}",
                    e
                ))
            },
        )?;
        f.write_all(&old_heap_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: write old heap: {}",
                e
            ))
        })?;
        f.write_all(&appended_heap).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: write appended heap: {}",
                e
            ))
        })?;
        // Block-pad to FITS block boundary.
        let written = new_main_bytes + new_pcount;
        if new_padded > written {
            let pad = vec![0u8; (new_padded - written) as usize];
            f.write_all(&pad).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed extend: pad: {}",
                    e
                ))
            })?;
        }
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!("compressed extend: flush: {}", e))
        })?;
    }

    // ----- header rewrite (PCOUNT + NAXIS2 + ZNAXIS<last>) -----
    let cards_guard = crate::hdu::CardsWriteGuard::from_parts(
        cards_arc.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?,
        cards_version,
    );
    let mut new_cards = cards_guard.clone_cards();
    set_pcount_in_cards(&mut new_cards, new_pcount);
    update_int_card(&mut new_cards, "NAXIS2", new_n_tiles as i64);
    // numpy axis 0 = slowest = FITS NAXIS<znaxis> (highest-numbered).
    let znaxis = naxis;
    let zaxis_key = format!("ZNAXIS{}", znaxis);
    update_int_card(&mut new_cards, &zaxis_key, new_naxis0 as i64);
    {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = offsets.header_offset();
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: header write failed: {}; close + reopen",
                e
            ))
        })?;
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed extend: header flush failed: {}; close + reopen",
                e
            ))
        })?;
    }
    cards_guard.commit(new_cards);

    // Boundary tiles' content changed (and any cached new tiles
    // would be inconsistent with the new heap layout anyway).
    cache.clear();

    Ok(())
}

// Write a single VLA descriptor (nelem + offset, P or Q format) into
// `buf` at `at`.  Used by extend's descriptor table updates.
fn write_descriptor(
    buf: &mut [u8],
    at: usize,
    heap_is_q: bool,
    nel: u64,
    off: u64,
) -> PyResult<()> {
    if heap_is_q {
        buf[at..at + 8].copy_from_slice(&nel.to_be_bytes());
        buf[at + 8..at + 16].copy_from_slice(&off.to_be_bytes());
    } else {
        if nel > u32::MAX as u64 || off > u32::MAX as u64 {
            return Err(PyValueError::new_err(format!(
                "compressed extend: P-descriptor overflow \
                 (nelem={}, offset={}); use heap_format='Q' for \
                 heaps > 4 GB",
                nel, off,
            )));
        }
        buf[at..at + 4].copy_from_slice(&(nel as u32).to_be_bytes());
        buf[at + 4..at + 8].copy_from_slice(&(off as u32).to_be_bytes());
    }
    Ok(())
}

// Read a single VLA descriptor (nelem + offset) from a buffer at `at`.
// (Named with the `_buf` suffix to disambiguate from the existing
// file-reading `read_descriptor` upstream.)  Used by extend/
// __setitem__ on quantized-float HDUs to inspect existing per-tile
// descriptors (specifically: is the tile in the primary column or
// the GZIP fallback column?).
fn read_descriptor_from_buf(
    buf: &[u8], at: usize, heap_is_q: bool,
) -> (u64, u64) {
    if heap_is_q {
        let nel = u64::from_be_bytes(buf[at..at + 8].try_into().unwrap());
        let off = u64::from_be_bytes(buf[at + 8..at + 16].try_into().unwrap());
        (nel, off)
    } else {
        let nel = u32::from_be_bytes(buf[at..at + 4].try_into().unwrap())
            as u64;
        let off = u32::from_be_bytes(buf[at + 4..at + 8].try_into().unwrap())
            as u64;
        (nel, off)
    }
}

// Quantized-float HDUs use a 4-column row layout: primary
// descriptor + ZSCALE (1D) + ZZERO (1D) + GZIP fallback descriptor.
// This helper writes one such row into `buf` at `at` (the start of
// the row).  `descriptor_size` is 8 for P-format, 16 for Q-format.
fn write_quant_descriptor_row(
    buf: &mut [u8],
    at: usize,
    heap_is_q: bool,
    descriptor_size: u64,
    primary_nelem: u64,
    primary_off: u64,
    zscale: f64,
    zzero: f64,
    fallback_nelem: u64,
    fallback_off: u64,
) -> PyResult<()> {
    write_descriptor(buf, at, heap_is_q, primary_nelem, primary_off)?;
    let zs_at = at + descriptor_size as usize;
    buf[zs_at..zs_at + 8].copy_from_slice(&zscale.to_be_bytes());
    buf[zs_at + 8..zs_at + 16].copy_from_slice(&zzero.to_be_bytes());
    let fb_at = zs_at + 16;
    write_descriptor(buf, fb_at, heap_is_q, fallback_nelem, fallback_off)?;
    Ok(())
}

// Read a quantized-float row (primary descriptor + ZSCALE + ZZERO
// + fallback descriptor) from `buf` at `at`.  Returns
// (primary_nelem, primary_off, zscale, zzero, fallback_nelem,
// fallback_off).  Inverse of write_quant_descriptor_row.
fn read_quant_descriptor_row(
    buf: &[u8],
    at: usize,
    heap_is_q: bool,
    descriptor_size: u64,
) -> (u64, u64, f64, f64, u64, u64) {
    let (p_nel, p_off) = read_descriptor_from_buf(buf, at, heap_is_q);
    let zs_at = at + descriptor_size as usize;
    let zscale =
        f64::from_be_bytes(buf[zs_at..zs_at + 8].try_into().unwrap());
    let zzero =
        f64::from_be_bytes(buf[zs_at + 8..zs_at + 16].try_into().unwrap());
    let fb_at = zs_at + 16;
    let (f_nel, f_off) = read_descriptor_from_buf(buf, fb_at, heap_is_q);
    (p_nel, p_off, zscale, zzero, f_nel, f_off)
}

// In-place modification of compressed image pixels via numpy-style
// slicing.  For each tile that overlaps the selection, decode the
// existing tile, apply the user's value to the affected portion,
// re-encode, and append the new bytes to the heap.  Old bytes for
// modified tiles become orphans (left in place; descriptors no
// longer reference them).
//
// Simpler than extend: no descriptor table growth (n_tiles is
// fixed), no heap relocation (the existing heap is preserved at its
// position).  Only:
//   - Each modified tile's descriptor is updated to point at its
//     new bytes appended at the end of the heap.
//   - PCOUNT grows by the total new-tile-bytes appended.
//   - If the new total exceeds the padded data section, grow the
//     file (shift later HDUs if non-last; set_len if last).
//
// Supports: contiguous slices, stepped slices, single integer
// indices, ellipsis, mixed combinations (all reuse the same
// AxisSlice + axis_overlap machinery that __getitem__ uses).
// RHS: numpy scalar / 0-d array (broadcast across selection), or
// N-d array whose shape matches the selection.  Unsigned-int
// trick HDUs accept the scaled dtype (e.g. u2 on a BITPIX=16 +
// BZERO=32768 HDU) and reverse-transform via
// normalize_compressed_input_dtype.
//
// Quantized-float HDUs (4-column schema) are deferred — the
// per-tile ZSCALE/ZZERO would need recomputation and the dither
// stream contract for SUBTRACTIVE_DITHER_* must hold across
// modifications.  Same scope cut as extend.
#[allow(clippy::too_many_arguments)]
fn setitem_compressed_image(
    py: Python<'_>,
    cards: &[String],
    offsets: &Arc<HduOffsets>,
    file_handle: &FileHandle,
    layout: &FileLayout,
    tainted: &TaintFlag,
    cache: &TileCache,
    cards_arc: &Arc<Mutex<Vec<String>>>,
    cards_version: &Arc<AtomicU64>,
    quantize_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::Quantize>>,
    >,
    compress_config: &Arc<
        Mutex<Option<crate::zimage::compression_config::CompressionConfigKind>>,
    >,
    key: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    check_not_tainted(tainted)?;

    // ----- header parse (mirrors extend) -----
    let zcmptype = parse_string_keyword(cards, "ZCMPTYPE").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZCMPTYPE")
    })?;
    let algorithm = crate::zimage::parse_algorithm(&zcmptype)?;
    let (zbitpix, image_shape) = parse_compressed_image_shape(cards)?;
    if image_shape.is_empty() {
        return Err(PyValueError::new_err(
            "compressed HDU has ZNAXIS=0 (no image data)",
        ));
    }
    let is_float = zbitpix < 0;
    let tfields = parse_keyword(cards, "TFIELDS").unwrap_or(0) as u64;
    // Schema dispatch: quantized-float HDUs use 4-col rows;
    // integer + unquantized-float use 1-col rows.
    let is_quantized_float = is_float && tfields == 4;

    // GZIP level (write-only param; None → codec default).
    let gzip_level: Option<u32> = compress_config
        .lock()
        .map_err(|_| PyIOError::new_err(
            "compress config lock poisoned",
        ))?
        .as_ref()
        .and_then(|c| c.gzip_level());

    let bytepix: u32 = match zbitpix {
        8 => 1,
        16 => 2,
        32 => 4,
        64 => 8,
        -32 => 4,
        -64 => 8,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported ZBITPIX {} for compressed __setitem__",
                other
            )))
        }
    };

    let tile_shape = parse_tile_shape(cards, &image_shape);
    let n_tiles = compute_n_tiles(&image_shape, &tile_shape);

    let tform1 = parse_string_keyword(cards, "TFORM1").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing TFORM1")
    })?;
    let tform1_trim = tform1.trim();
    let heap_is_q =
        tform1_trim.starts_with("1Q") || tform1_trim.starts_with('Q');
    let descriptor_size: u64 = if heap_is_q { 16 } else { 8 };
    let inner_byte_width: u64 =
        tform_vla_inner_byte_width(tform1_trim).unwrap_or(1);

    let old_pcount =
        parse_keyword(cards, "PCOUNT").unwrap_or(0).max(0) as u64;

    let (blocksize_opt, _) = parse_rice_params(cards);
    let blocksize = blocksize_opt.unwrap_or(32);
    let hcompress_scale = parse_hcompress_scale(cards);

    // Row width: 4-col for quantized-float, single-col otherwise.
    let row_width: u64 = if is_quantized_float {
        descriptor_size + 8 + 8 + descriptor_size
    } else {
        descriptor_size
    };
    let main_bytes = row_width.saturating_mul(n_tiles);
    let data_offset = offsets.data_offset();

    // Quantization setup (only for quantized-float path).
    let quant_setup: Option<(crate::zimage::quantize::DitherMethod, i64, f64)> =
        if is_quantized_float {
            let zq = parse_string_keyword(cards, "ZQUANTIZ");
            let method = crate::zimage::quantize::parse_dither_method(
                zq.as_deref(),
            )?
            .ok_or_else(|| {
                PyValueError::new_err(
                    "quantized-float HDU has ZQUANTIZ='NONE' but the \
                     4-column schema is present (malformed file).",
                )
            })?;
            let zd =
                parse_keyword(cards, "ZDITHER0").unwrap_or(1).max(1);
            let level = quantize_config
                .lock()
                .map_err(|_| {
                    PyIOError::new_err("quantize config lock poisoned")
                })?
                .as_ref()
                .map(|q| q.level)
                .unwrap_or(4.0);
            Some((method, zd, level))
        } else {
            None
        };

    // ----- parse slice key -----
    let slices = normalize_slice_key(key, &image_shape)?;
    // Output shape: drop is_int axes.  For all-int → empty shape
    // (RHS must be a scalar in that case).
    let output_shape: Vec<u64> = slices
        .iter()
        .filter(|s| !s.is_int)
        .map(|s| s.count)
        .collect();
    // Zero-count anywhere = empty selection = no-op (same as
    // numpy's `arr[5:5] = ...` semantics).
    if slices.iter().any(|s| s.count == 0) {
        return Ok(());
    }

    // ----- MaskedArray entry -----
    // If value is a numpy.ma.MaskedArray, fill masked positions
    // with the appropriate sentinel before the rest of the
    // pipeline.  No-op for plain ndarray / scalar RHS.
    let unmasked_value =
        crate::hdu_image::unwrap_masked_input(py, value, cards, true)?;
    let value = unmasked_value.bind(py);

    // ----- RHS validation + dtype normalization -----
    let np = py.import("numpy")?;
    let value_arr = np.call_method1("asarray", (value,))?;
    let value_shape: Vec<u64> = value_arr.getattr("shape")?.extract()?;
    let is_scalar_rhs = value_shape.is_empty();

    let value_norm: Py<PyAny> = if is_scalar_rhs {
        value_arr.clone().unbind()
    } else {
        // Shape must match the selection exactly (no broadcasting
        // beyond the scalar case — matches numpy's stricter
        // assignment rules).
        if value_shape != output_shape {
            return Err(PyValueError::new_err(format!(
                "compressed __setitem__: value shape {:?} does not \
                 match selection shape {:?}",
                value_shape, output_shape,
            )));
        }
        // Reverse-transform unsigned-int trick if needed.
        if !is_float {
            normalize_compressed_input_dtype(
                py, &value_arr, cards, zbitpix,
            )?
        } else {
            value_arr.clone().unbind()
        }
    };
    let value_bound = value_norm.bind(py);

    // ----- encode setup -----
    let int_ctx = IntTileCtx {
        algorithm,
        bytepix,
        zbitpix,
        inner_byte_width,
        blocksize,
        hcompress_scale,
        gzip_level,
    };
    let be_dtype = match zbitpix {
        8 => ">u1",
        16 => ">i2",
        32 => ">i4",
        64 => ">i8",
        -32 => ">f4",
        -64 => ">f8",
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported ZBITPIX {} for compressed __setitem__",
                other
            )))
        }
    };
    let native_dtype = zbitpix_to_native_dtype(zbitpix)?;

    let cols = find_data_columns(cards)?;
    let theap = parse_keyword(cards, "THEAP")
        .map(|x| x.max(0) as u64)
        .unwrap_or(row_width * n_tiles);

    // Quantization context (only meaningful for quantized-float
    // HDUs; runs the dequantize step inside get_or_decode_tile).
    let quant_ctx_opt = if is_quantized_float {
        build_quant_context(cards, &cols)?
    } else {
        None
    };

    // ----- read main descriptor table early -----
    // For quantized-float HDUs we need each affected tile's
    // existing ZSCALE/ZZERO and primary/fallback column status to
    // pick the right re-encode path.  For integer/unquantized
    // we'd read main_buf later anyway; doing it here unifies the
    // structure.
    let mut main_buf: Vec<u8> = vec![0; main_bytes as usize];
    if main_bytes > 0 {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut main_buf).map_err(|e| {
            PyIOError::new_err(format!(
                "compressed __setitem__: read main table: {}",
                e
            ))
        })?;
    }

    // ----- per-tile re-encode for overlapping tiles -----
    let mut appended_heap: Vec<u8> = Vec::new();
    let mut appended_fallback_heap: Vec<u8> = Vec::new();
    let mut tile_updates: Vec<(u64, TileRow)> = Vec::new();

    let mut setitem_origin_buf = [0u64; MAX_NAXIS];
    let mut setitem_shape_buf = [0u64; MAX_NAXIS];
    for tile_idx in 0..n_tiles {
        let d = tile_origin_and_shape(
            tile_idx, &image_shape, &tile_shape,
            &mut setitem_origin_buf, &mut setitem_shape_buf,
        );
        let origin = &setitem_origin_buf[..d];
        let actual_shape = &setitem_shape_buf[..d];
        // Build per-axis tile_indexer + output_indexer using the
        // same machinery __getitem__ uses.
        let mut tile_indexers: Vec<Bound<PyAny>> =
            Vec::with_capacity(slices.len());
        let mut output_indexers: Vec<Bound<PyAny>> = Vec::new();
        let mut overlapping = true;
        for (axis_idx, axis) in slices.iter().enumerate() {
            match axis_overlap(
                py,
                origin[axis_idx],
                actual_shape[axis_idx],
                axis,
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

        // Decode the existing tile.  For quantized-float HDUs the
        // decoder runs dequantize, returning f4/f8 native bytes;
        // for integer/unquantized-float it returns BITPIX-native
        // bytes directly.  frombuffer returns a read-only view;
        // .copy() gives us a writable ndarray.
        let (dec_bytepix, dec_stored_zbitpix) = if is_quantized_float {
            (4u32, 32i32)
        } else {
            (bytepix, zbitpix)
        };
        let old_bytes_arc = get_or_decode_tile(
            py,
            cache,
            file_handle,
            tainted,
            tile_idx,
            data_offset,
            row_width,
            theap,
            &cols,
            algorithm,
            actual_shape,
            dec_bytepix,
            blocksize,
            dec_stored_zbitpix,
            zbitpix,
            quant_ctx_opt.as_ref(),
            false,
        )?;
        let pyb = PyBytes::new(py, &old_bytes_arc);
        let arr_flat =
            np.call_method1("frombuffer", (pyb, native_dtype))?;
        let shape_tuple = PyTuple::new(py, actual_shape)?;
        let arr_view = arr_flat.call_method1("reshape", (shape_tuple,))?;
        let tile_arr = arr_view.call_method0("copy")?;

        // Apply RHS to the tile's selected pixels.  For scalar
        // RHS, numpy broadcasts the value.  For array RHS, slice
        // the value with the output indexer to extract the
        // tile-specific portion.
        let tile_idx_tuple = PyTuple::new(py, &tile_indexers)?;
        if is_scalar_rhs {
            tile_arr.set_item(tile_idx_tuple, value_bound.clone())?;
        } else {
            let out_idx_tuple = PyTuple::new(py, &output_indexers)?;
            let rhs_for_tile = value_bound.get_item(out_idx_tuple)?;
            tile_arr.set_item(tile_idx_tuple, rhs_for_tile)?;
        }

        // Cast to BE bytes and re-encode (dispatch on schema).
        let tile_be =
            np.call_method1("ascontiguousarray", (tile_arr, be_dtype))?;
        let tile_bytes: Vec<u8> =
            tile_be.call_method0("tobytes")?.extract()?;
        let n_pixels: usize = actual_shape.iter().product::<u64>() as usize;
        let row = if is_quantized_float {
            encode_quant_boundary_tile(
                algorithm,
                zbitpix,
                inner_byte_width,
                blocksize,
                hcompress_scale,
                gzip_level,
                &main_buf,
                tile_idx,
                row_width,
                descriptor_size,
                heap_is_q,
                quant_setup.unwrap(),
                &tile_bytes,
                actual_shape,
                n_pixels,
                &mut appended_heap,
                &mut appended_fallback_heap,
            )?
        } else {
            encode_tile_int(
                &int_ctx,
                &tile_bytes,
                tile_idx,
                actual_shape,
                n_pixels,
                &mut appended_heap,
            )?
        };
        tile_updates.push((tile_idx, row));
    }

    // No tiles actually overlapped (e.g. out-of-bounds slice).
    // Per numpy convention, this is a silent no-op.
    if tile_updates.is_empty() {
        return Ok(());
    }

    // ----- combine primary + fallback appended heaps -----
    // (No-op for integer/unquantized-float — fallback_heap is
    // empty.  Quantized-float may have both.)
    let appended_primary_size = appended_heap.len() as u64;
    appended_heap.extend(appended_fallback_heap.drain(..));
    for (_, row) in tile_updates.iter_mut() {
        if row.fallback_nelem > 0 {
            row.fallback_off += appended_primary_size;
        }
    }
    // Shift descriptor offsets into the absolute-heap frame.
    // appended_heap sits at old_pcount in the combined heap.
    for (_, row) in tile_updates.iter_mut() {
        if row.primary_nelem > 0 {
            row.primary_off += old_pcount;
        }
        if row.fallback_nelem > 0 {
            row.fallback_off += old_pcount;
        }
    }

    // Apply descriptor updates to the already-read main_buf in place.
    for (tile_idx, row) in &tile_updates {
        let desc_offset = (tile_idx * row_width) as usize;
        if is_quantized_float {
            write_quant_descriptor_row(
                &mut main_buf,
                desc_offset,
                heap_is_q,
                descriptor_size,
                row.primary_nelem,
                row.primary_off,
                row.zscale,
                row.zzero,
                row.fallback_nelem,
                row.fallback_off,
            )?;
        } else {
            write_descriptor(
                &mut main_buf,
                desc_offset,
                heap_is_q,
                row.primary_nelem,
                row.primary_off,
            )?;
        }
    }

    // ----- compute new file extents -----
    let new_pcount = old_pcount + appended_heap.len() as u64;
    let new_data_size = main_bytes + new_pcount;
    let new_padded = if new_data_size == 0 {
        0
    } else {
        ((new_data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            * BLOCK_SIZE as u64
    };
    let current_hdu_end = {
        let guard = layout
            .hdus
            .lock()
            .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
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
                let g = lock_file(file_handle)?;
                let f = g
                    .as_ref()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                f.metadata()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?
                    .len()
            }
        }
    };
    let current_padded_data = current_hdu_end.saturating_sub(data_offset);
    let new_hdu_end = data_offset + new_padded;

    // ----- grow file if needed -----
    if new_padded > current_padded_data {
        let delta = new_padded - current_padded_data;
        let file_len = {
            let g = lock_file(file_handle)?;
            let f = g
                .as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            shift_file_tail_and_update_offsets(
                file_handle,
                layout,
                current_hdu_end,
                delta,
                tainted,
            )?;
        } else {
            let mut g = lock_file(file_handle)?;
            let f = g
                .as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
    }

    // ----- write updated main + appended heap to disk -----
    {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        // Main descriptor table (with updated tile descriptors).
        f.seek(SeekFrom::Start(data_offset)).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: seek to data_offset: {}",
                e
            ))
        })?;
        f.write_all(&main_buf).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: write main: {}",
                e
            ))
        })?;
        // Appended heap bytes at end of existing heap.
        if !appended_heap.is_empty() {
            f.seek(SeekFrom::Start(
                data_offset + main_bytes + old_pcount,
            ))
            .map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed __setitem__: seek to heap end: {}",
                    e
                ))
            })?;
            f.write_all(&appended_heap).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed __setitem__: write appended heap: {}",
                    e
                ))
            })?;
        }
        // Block-pad to FITS block boundary.
        let written_end = main_bytes + new_pcount;
        if new_padded > written_end {
            let pad = vec![0u8; (new_padded - written_end) as usize];
            f.write_all(&pad).map_err(|e| {
                tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "compressed __setitem__: pad: {}",
                    e
                ))
            })?;
        }
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: flush: {}",
                e
            ))
        })?;
    }

    // ----- header rewrite (PCOUNT only — shape unchanged) -----
    let cards_guard = crate::hdu::CardsWriteGuard::from_parts(
        cards_arc.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?,
        cards_version,
    );
    let mut new_cards = cards_guard.clone_cards();
    set_pcount_in_cards(&mut new_cards, new_pcount);
    {
        let mut g = lock_file(file_handle)?;
        let f = g
            .as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = offsets.header_offset();
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: header write: {}; close + reopen",
                e
            ))
        })?;
        f.flush().map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed __setitem__: header flush: {}; close + reopen",
                e
            ))
        })?;
    }
    cards_guard.commit(new_cards);

    // Modified tiles are now stale in the cache.
    cache.clear();

    Ok(())
}

// ---------------------------------------------------------------------------
// Heap repack — drop orphans accumulated by extend/__setitem__
// ---------------------------------------------------------------------------
//
// ZIMAGE heaps are the same shape as VLA-table heaps (a contiguous
// byte region after the main rows, addressed by P/Q descriptors in the
// main rows).  This function mirrors `repack_table_heap` in hdu_table:
// walk every row × every descriptor column (primary + optional GZIP /
// UNCOMPRESSED fallbacks), copy live cells into a compact new heap,
// rewrite the in-memory descriptors, write everything back, shrink the
// on-disk file if the new padded extent is smaller, then update
// PCOUNT.  Clears the tile cache (its entries no longer match the new
// heap layout).
fn repack_compressed_heap(
    super_: &HDU,
    cache: &TileCache,
) -> PyResult<()> {
    check_not_tainted(&super_.tainted)?;
    let cards = super_.header_snapshot()?;
    let naxis1 = parse_keyword(&cards, "NAXIS1")
        .unwrap_or(0).max(0) as u64;
    let naxis2 = parse_keyword(&cards, "NAXIS2")
        .unwrap_or(0).max(0) as u64;
    let current_pcount = parse_keyword(&cards, "PCOUNT")
        .unwrap_or(0).max(0) as u64;
    let data_offset = super_.offsets.data_offset();
    if current_pcount == 0 || naxis2 == 0 {
        return Ok(());
    }

    // Reject non-default THEAP — repack would write the new heap at
    // the default position and corrupt a non-default layout.  Files
    // rustfits creates never set THEAP, so this only blocks the rare
    // case of repacking a file written by another tool with a custom
    // heap offset.
    let theap_raw = parse_keyword(&cards, "THEAP").unwrap_or(0);
    let main_bytes = naxis1.saturating_mul(naxis2);
    if theap_raw > 0 && (theap_raw as u64) != main_bytes {
        return Err(PyValueError::new_err(format!(
            "repack: file has non-default THEAP={} (main rows end at \
             {}); repack would write the new heap at the default \
             position and corrupt the file",
            theap_raw, main_bytes)));
    }

    let cols = find_data_columns(&cards)?;

    // Read whole main table + old heap under a single file lock.
    let mut main_buf = vec![0u8; main_bytes as usize];
    let mut old_heap = vec![0u8; current_pcount as usize];
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut main_buf)
            .map_err(|e| PyIOError::new_err(format!(
                "repack: read main failed: {}", e)))?;
        f.seek(SeekFrom::Start(data_offset + main_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.read_exact(&mut old_heap)
            .map_err(|e| PyIOError::new_err(format!(
                "repack: read heap failed: {}", e)))?;
    }

    // Walk every row × every descriptor column; copy live cells.
    let primary_slot = Some(cols.primary);
    let cols_list: [&Option<ZimageColumnInfo>; 3] = [
        &primary_slot,
        &cols.gzip_fallback,
        &cols.uncompressed_fallback,
    ];
    let mut new_heap: Vec<u8> = Vec::new();
    for r in 0..naxis2 {
        let row_off = (r * naxis1) as usize;
        for slot in cols_list.iter() {
            let Some(col) = slot.as_ref() else { continue; };
            let desc_at = row_off + col.byte_offset_in_row as usize;
            let (nel, old_off) =
                read_descriptor_from_buf(&main_buf, desc_at, col.is_q);
            if nel == 0 {
                // Empty descriptor; rewrite as (0, 0) to keep the
                // layout canonical.
                write_descriptor(
                    &mut main_buf, desc_at, col.is_q, 0, 0)?;
                continue;
            }
            let n_bytes = nel.saturating_mul(col.inner_byte_width);
            if old_off + n_bytes > current_pcount {
                return Err(PyValueError::new_err(format!(
                    "repack: tile row {}: descriptor points past \
                     heap end (offset+bytes={} > PCOUNT={})",
                    r, old_off + n_bytes, current_pcount)));
            }
            let new_off = new_heap.len() as u64;
            new_heap.extend_from_slice(
                &old_heap[old_off as usize
                    ..(old_off + n_bytes) as usize]);
            write_descriptor(
                &mut main_buf, desc_at, col.is_q, nel, new_off)?;
        }
    }
    drop(old_heap);
    let new_pcount = new_heap.len() as u64;
    if new_pcount == current_pcount {
        return Ok(());
    }

    let current_data_bytes = main_bytes + current_pcount;
    let new_data_bytes = main_bytes + new_pcount;
    let current_padded =
        crate::hdu_image::round_up_to_block(current_data_bytes);
    let new_padded =
        crate::hdu_image::round_up_to_block(new_data_bytes);
    let current_hdu_end = data_offset + current_padded;
    let new_hdu_end = data_offset + new_padded;

    // Write back main table + new heap.
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(data_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&main_buf) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: write main: {}; close + reopen", e)));
        }
        f.seek(SeekFrom::Start(data_offset + main_bytes))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if let Err(e) = f.write_all(&new_heap) {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: write heap: {}; close + reopen", e)));
        }
        if let Err(e) = f.flush() {
            super_.tainted.store(true, Ordering::Release);
            return Err(PyIOError::new_err(format!(
                "repack: flush: {}; close + reopen", e)));
        }
    }

    if new_hdu_end < current_hdu_end {
        let delta = current_hdu_end - new_hdu_end;
        let file_len = {
            let g = lock_file(&super_.file)?;
            let f = g.as_ref()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len()
        };
        if file_len > current_hdu_end {
            shift_file_tail_backward_and_update_offsets(
                &super_.file, &super_.layout,
                current_hdu_end, delta, &super_.tainted)?;
        } else {
            let mut g = lock_file(&super_.file)?;
            let f = g.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            f.set_len(new_hdu_end).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "repack: set_len: {}; close + reopen", e))
            })?;
        }
    }

    // PCOUNT update — disk-write-before-commit.
    let cards_guard = super_.cards_write_lock()?;
    let mut new_cards = cards_guard.clone_cards();
    set_pcount_in_cards(&mut new_cards, new_pcount);
    {
        let mut g = lock_file(&super_.file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        let header_bytes = serialize_header_to_disk_bytes(&new_cards);
        let header_offset = data_offset - header_bytes.len() as u64;
        f.seek(SeekFrom::Start(header_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&header_bytes).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack: PCOUNT header write: {}; close + reopen", e))
        })?;
        f.flush().map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "repack: PCOUNT header flush: {}; close + reopen", e))
        })?;
    }
    cards_guard.commit(new_cards);
    cache.clear();
    Ok(())
}
