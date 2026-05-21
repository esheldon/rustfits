// CompressedImageHDU — tile-compressed image extension (ZIMAGE
// convention).  Phase 1 (this file): detection + dispatch wiring,
// image-shape and compression accessors, and a configurable LRU
// tile cache placeholder.  Decoding lives in src/zimage/* and is
// added in later phases (Phase 2: RICE_1 whole-image read; Phase
// 3: slicing + cache fill).
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
// `_force_taint` are inherited from HDU.  `has_data` reads the
// BINTABLE NAXIS/NAXISn (not the Z* keys) but the two agree:
// non-empty image → non-zero n_tiles → BINTABLE NAXIS2 > 0;
// empty image → n_tiles = 0 → NAXIS2 = 0.  So no override needed.
//
// TODO (Phase 2): make CompressedImageHDU extend ImageHDU instead
// of HDU directly, so `isinstance(hdu, ImageHDU)` is True.  This
// requires restructuring `new()` to use PyClassInitializer with
// the three-level chain (HDU + ImageHDU + CompressedImageHDU),
// updating accessor `into_super()` chains to step through both
// parents, and overriding ImageHDU's data-access methods (`read`,
// `write`, `extend`, `__getitem__`, `__setitem__`) so the
// inherited uncompressed implementations don't silently produce
// wrong results on a tile-compressed HDU.  The override of
// `read` becomes the actual decoder in Phase 2; the override of
// `__getitem__` becomes the slice path in Phase 3; the write-
// side overrides stay as NotImplementedError until compressed
// writes (Phase 7+).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyTuple;

use crate::common::{
    parse_keyword, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, TaintFlag,
};
use crate::hdu::HDU;

// 32 MiB by default — large enough to cache a few hundred typical
// 256x256 i4 tiles, small enough not to be surprising on a desktop.
pub(crate) const DEFAULT_TILE_CACHE_BYTES: u64 = 32 * 1024 * 1024;

#[pyclass(extends = HDU)]
pub(crate) struct CompressedImageHDU {
    // Configured cache size in bytes.  Stored in an Arc<AtomicU64>
    // so the value survives across get/set calls and so Phase 3's
    // LRU storage (added alongside slicing) can co-own it.  Phase
    // 1 reads/writes the size but does not yet store decoded tiles
    // anywhere — there's no .read() / __getitem__ yet.
    pub(crate) tile_cache_max_bytes: Arc<AtomicU64>,
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
    ) -> (Self, HDU) {
        (
            CompressedImageHDU {
                tile_cache_max_bytes: Arc::new(
                    AtomicU64::new(DEFAULT_TILE_CACHE_BYTES),
                ),
            },
            HDU::new(header, index, filename, offsets, layout, file, tainted),
        )
    }
}

#[pymethods]
impl CompressedImageHDU {
    // Multi-line repr matching ImageHDU's, with the compression
    // type and tile shape surfaced.  Reports the *uncompressed*
    // image dtype + dims (what the user sees after .read()), not
    // the on-disk BINTABLE.
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
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
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_compressed_image_shape(&cards)?;
        Ok(PyTuple::new(py, &shape)?.unbind())
    }

    // numpy dtype matching ZBITPIX — what .read() will return.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (zbitpix, _) = parse_compressed_image_shape(&cards)?;
        let dtype_str = zbitpix_to_native_dtype(zbitpix)?;
        let np = py.import("numpy")?;
        Ok(np.call_method1("dtype", (dtype_str,))?.unbind())
    }

    // ZNAXIS — number of image axes.
    #[getter]
    fn ndim(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_compressed_image_shape(&cards)?;
        Ok(shape.len())
    }

    // Total pixel count (product of all ZNAXISn).  Returns 0 for
    // an empty shape so the accessor is well-defined on degenerate
    // headers; real compressed HDUs always have NAXIS > 0.
    #[getter]
    fn size(slf: PyRef<'_, Self>) -> PyResult<u64> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_compressed_image_shape(&cards)?;
        Ok(if shape.is_empty() { 0 } else { shape.iter().product() })
    }

    // Raw ZBITPIX — the image-side bitpix the user will read into.
    #[getter]
    fn bitpix(slf: PyRef<'_, Self>) -> PyResult<i32> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (zbitpix, _) = parse_compressed_image_shape(&cards)?;
        Ok(zbitpix)
    }

    // numpy convention: `len(arr)` is shape[0].
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
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
        let super_ = slf.into_super();
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
        let super_ = slf.into_super();
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
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (_, image_shape) = parse_compressed_image_shape(&cards)?;
        let tile_shape = parse_tile_shape(&cards, &image_shape);
        Ok(PyTuple::new(py, &tile_shape)?.unbind())
    }

    // Total number of tiles in the image: product of
    // ceil(NAXISn / TILEn).
    #[getter]
    fn n_tiles(slf: PyRef<'_, Self>) -> PyResult<u64> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (_, image_shape) = parse_compressed_image_shape(&cards)?;
        let tile_shape = parse_tile_shape(&cards, &image_shape);
        Ok(compute_n_tiles(&image_shape, &tile_shape))
    }

    // ----- tile cache (Phase 1: config plumbing only) -----

    // Current cache capacity in bytes.  Stored Phase 1; consulted
    // Phase 3 when slicing fills the cache.
    #[getter]
    fn tile_cache_size(slf: PyRef<'_, Self>) -> u64 {
        slf.tile_cache_max_bytes.load(Ordering::Relaxed)
    }

    // Configure the cache capacity.  0 disables caching (each
    // tile decode is one-shot — useful for memory-constrained
    // one-shot reads).
    fn set_tile_cache_size(&self, bytes: u64) {
        self.tile_cache_max_bytes.store(bytes, Ordering::Relaxed);
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
