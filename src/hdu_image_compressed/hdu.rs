// CompressedImageHDU pyclass: struct + impls + #[pymethods] dispatch,
// ZIMAGE detection (header_has_zimage), the TileCache alias, and the
// per-HDU meta-cache accessor.  Free helpers live in sibling modules.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyNotImplementedError};
use pyo3::types::PyTuple;

use crate::cache::BytesBoundLruCache;
use crate::common::{
    parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, TaintFlag,
};
use crate::hdu::HDU;
use crate::hdu_image::ImageHDU;

use crate::zimage::tile_io::DEFAULT_TILE_CACHE_BYTES;
use super::checksum::{
    compressed_add_checksum, compressed_add_datasum,
    compressed_verify_checksum, compressed_verify_datasum,
};
use super::meta::{
    build_compression_config, compression_config_kind_to_py,
    parse_compressed_image_meta, parse_compressed_image_shape, zbitpix_to_native_dtype, CompressedImageMeta,
};
use super::read::{read_compressed_image_data, slice_compressed_image};
use super::repack::repack_compressed_heap;
use super::write::{
    extend_compressed_image_data, setitem_compressed_image,
    write_compressed_image_data,
};

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
pub(crate) type TileCache = BytesBoundLruCache<u64>;

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
// Version-stamped parsed-metadata cache (see `meta()`); the u64 is the
// `cards_version` at parse time.
type MetaCache = Arc<Mutex<Option<(u64, Arc<CompressedImageMeta>)>>>;

#[pyclass(extends = ImageHDU)]
pub(crate) struct CompressedImageHDU {
    pub(crate) cache: Arc<TileCache>,
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
    meta_cache: MetaCache,
}

impl CompressedImageHDU {
    #[allow(clippy::too_many_arguments)]
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

