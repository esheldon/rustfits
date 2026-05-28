// Header-derived metadata for compressed images: shape/tile parsing,
// the CompressedImageMeta cache struct + parser, data-column discovery,
// quant-context build, TFORM helpers, and compression-config rebuild.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::common::{
    parse_keyword, parse_string_keyword,
};
use crate::zimage::CompressionAlgorithm;

// Parse ZBITPIX, ZNAXIS, ZNAXISn → (zbitpix, numpy-order shape).
// Tolerant of NAXIS=0 in the same way parse_image_hdu_shape_lax is:
// returns an empty shape rather than erroring, so accessors and
// repr work on malformed headers.
pub(crate) fn parse_compressed_image_shape(
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
pub(crate) fn parse_tile_shape(header: &[String], image_shape: &[u64]) -> Vec<u64> {
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
        total = total.saturating_mul(img.div_ceil(tile));
    }
    total
}

// Image dtype string for a given ZBITPIX value.  Same supported set
// as the uncompressed image side (u1/i2/i4/i8/f4/f8); the unsigned-
// int trick still operates via BSCALE/BZERO at read time and isn't
// a property of ZBITPIX itself.
pub(crate) fn zbitpix_to_native_dtype(zbitpix: i32) -> PyResult<&'static str> {
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
    pub(crate) byte_offset_in_row: u64,
    pub(crate) is_q: bool,
    pub(crate) inner_byte_width: u64,
}

// All ZIMAGE data columns of interest; the primary heap column is
// required, fallbacks and quantization columns are optional and
// only used in specific code paths (fallbacks when the primary
// tile is empty; ZSCALE/ZZERO when ZBITPIX is float).  Resolved
// in find_data_columns by walking TTYPEn.
pub(crate) struct ZimageDataColumns {
    pub(crate) primary: ZimageColumnInfo,
    pub(crate) gzip_fallback: Option<ZimageColumnInfo>,
    pub(crate) uncompressed_fallback: Option<ZimageColumnInfo>,
    // Fixed-width 1D (double) columns used for per-tile
    // dequantization.  Stored as raw row-byte offsets — the
    // dequant path seeks to `data_offset + row*naxis1 + offset`
    // and reads 8 big-endian bytes.
    pub(crate) zscale_offset_in_row: Option<u64>,
    pub(crate) zzero_offset_in_row: Option<u64>,
}

// Quantization parameters for the float-ZBITPIX path.  Built once
// per read at the top of `read_compressed_image_data` /
// `slice_compressed_image` when ZBITPIX is negative, then carried
// through the tile loop so `get_or_decode_tile` can read per-tile
// ZSCALE/ZZERO and dequantize after decode.
pub(crate) struct ZimageQuantContext {
    pub(crate) method: crate::zimage::quantize::DitherMethod,
    pub(crate) zdither0: i64,
    pub(crate) zscale_offset_in_row: u64,
    pub(crate) zzero_offset_in_row: u64,
    // ZBITPIX of the *output* float dtype (-32 or -64).  Decoder
    // always works in i32; this picks dequantize_to_f32 vs _f64.
    pub(crate) output_zbitpix: i32,
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
pub(crate) fn parse_compressed_image_meta(cards: &[String]) -> PyResult<CompressedImageMeta> {
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
pub(crate) fn build_quant_context(
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
pub(crate) fn find_data_columns(header: &[String]) -> PyResult<ZimageDataColumns> {
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
        'X' => Ok(repeat.div_ceil(8)),
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
pub(crate) fn tform_vla_inner_byte_width(tform: &str) -> Option<u64> {
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
pub(crate) fn parse_rice_params(header: &[String]) -> (Option<u32>, Option<u32>) {
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
            (Some("BLOCKSIZE"), Some(val))
                if val > 0 => {
                    blocksize = Some(val as u32);
                }
            (Some("BYTEPIX"), Some(val))
                if val > 0 => {
                    bytepix = Some(val as u32);
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
pub(crate) fn parse_hcompress_scale(header: &[String]) -> i32 {
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
pub(crate) fn compression_config_kind_to_py(
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

pub(crate) fn build_compression_config(
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

