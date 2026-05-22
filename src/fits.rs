// FITS pyclass — top-level handle for an open FITS file.  Owns the file
// handle, the HDU list, and the per-file taint flag.  Also home to the
// HDU-list parser (`parse_hdus_from_file`) and the header-shape validators
// it uses.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::Bound;
use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicBool;

use crate::common::{
    lock_file, parse_keyword, parse_string_keyword,
    FileHandle, FileLayout, HduOffsets, TaintFlag,
    BLOCK_SIZE, CARDS_PER_BLOCK, CARD_SIZE,
};
use crate::hdu::HDU;
use crate::hdu_image::{dtype_to_bitpix, ImageHDU};
use crate::hdu_image_compressed::{header_has_zimage, CompressedImageHDU};
use crate::hdu_table::{normalize_and_build_table_header, TableHDU};
use crate::hdu_ascii_table::AsciiTableHDU;
use crate::header::{card_int, card_logical, card_string, card_uint, pad_to_card};

// Per the FITS standard, the primary HDU must begin with `SIMPLE`, extension
// HDUs must begin with `XTENSION`, and every HDU must declare BITPIX, NAXIS,
// and NAXIS1..NAXISn.  `END` is enforced implicitly by the reader (it keeps
// pulling 2880-byte blocks until END is seen or EOF is hit).
fn validate_header(cards: &[String], is_primary: bool) -> PyResult<()> {
    let first = match cards.iter().find(|c| !c.trim().is_empty()) {
        Some(c) => c,
        None => return Err(PyValueError::new_err("empty header")),
    };
    let first_key = if first.len() >= 8 { first[..8].trim() } else { first.trim() };

    if is_primary {
        if first_key != "SIMPLE" {
            return Err(PyValueError::new_err(format!(
                "primary header must start with SIMPLE, found '{}'", first_key
            )));
        }
    } else if first_key != "XTENSION" {
        return Err(PyValueError::new_err(format!(
            "extension header must start with XTENSION, found '{}'", first_key
        )));
    }

    if parse_keyword(cards, "BITPIX").is_none() {
        return Err(PyValueError::new_err("missing required BITPIX keyword"));
    }
    let naxis = match parse_keyword(cards, "NAXIS") {
        Some(n) => n,
        None => return Err(PyValueError::new_err("missing required NAXIS keyword")),
    };
    if !(0..=999).contains(&naxis) {
        return Err(PyValueError::new_err(format!(
            "NAXIS={} out of range (must be 0..999)", naxis
        )));
    }
    for n in 1..=naxis {
        let key = format!("NAXIS{}", n);
        if parse_keyword(cards, &key).is_none() {
            return Err(PyValueError::new_err(format!(
                "missing required {} keyword", key
            )));
        }
    }
    Ok(())
}

// General FITS data-section formula:
//   N_bytes = |BITPIX|/8 * GCOUNT * (PCOUNT + Π NAXISn)
// For images: GCOUNT=1, PCOUNT=0, reducing to bytes_per_pixel * Π NAXISn.
// For binary tables, PCOUNT carries the variable-length-array heap size,
// which must be included so the next HDU is located correctly.  NAXIS=0
// means no data unit regardless of PCOUNT/GCOUNT.
fn calculate_data_size(header_cards: &[String]) -> u64 {
    let bitpix = parse_keyword(header_cards, "BITPIX").unwrap_or(0);
    let naxis = parse_keyword(header_cards, "NAXIS").unwrap_or(0) as usize;

    if bitpix == 0 || naxis == 0 {
        return 0;
    }

    let bytes_per_pixel = (bitpix.abs() / 8) as u64;

    let pcount_raw = parse_keyword(header_cards, "PCOUNT").unwrap_or(0);
    let pcount: u64 = if pcount_raw > 0 { pcount_raw as u64 } else { 0 };

    let gcount_raw = parse_keyword(header_cards, "GCOUNT").unwrap_or(1);
    let gcount: u64 = if gcount_raw > 0 { gcount_raw as u64 } else { 1 };

    let mut product: u64 = 1;
    for i in 1..=naxis {
        if let Some(dim) = parse_keyword(header_cards, &format!("NAXIS{}", i)) {
            product = product.saturating_mul(dim as u64);
        }
    }

    let raw_size = bytes_per_pixel
        .saturating_mul(gcount)
        .saturating_mul(product.saturating_add(pcount));

    if raw_size == 0 {
        0
    } else {
        ((raw_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64) * BLOCK_SIZE as u64
    }
}

// Walks the file from byte 0, extracting every HDU header and skipping over
// each data section, returning the parsed HDU Python objects.  Each HDU is
// constructed with its data-section byte offset and a clone of the file
// handle so that write-back methods can locate themselves on disk.
fn parse_hdus_from_file(
    py: Python<'_>,
    filename: &str,
    handle: &FileHandle,
    layout: &Arc<FileLayout>,
    tainted: &TaintFlag,
) -> PyResult<Vec<Py<PyAny>>> {
    let mut guard = lock_file(handle)?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    file.seek(SeekFrom::Start(0))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    let mut hdus: Vec<Py<PyAny>> = Vec::new();
    let mut offset = 0u64;

    loop {
        let mut header_cards: Vec<String> = Vec::new();
        let mut end_found = false;

        while !end_found {
            let mut block = vec![0u8; BLOCK_SIZE];
            match file.read_exact(&mut block) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    if header_cards.is_empty() {
                        break;
                    } else {
                        return Err(PyIOError::new_err("truncated FITS file"));
                    }
                }
                Err(e) => return Err(PyIOError::new_err(e.to_string())),
            }

            // FITS header bytes are restricted to printable ASCII (0x20-0x7E).
            for (j, &b) in block.iter().enumerate() {
                if !(0x20..=0x7E).contains(&b) {
                    return Err(PyValueError::new_err(format!(
                        "non-printable byte 0x{:02X} in header block at byte offset {}",
                        b, offset + j as u64
                    )));
                }
            }

            for i in (0..BLOCK_SIZE).step_by(CARD_SIZE) {
                // Safe: bytes have just been validated as printable ASCII.
                let card = std::str::from_utf8(&block[i..i + CARD_SIZE])
                    .unwrap()
                    .trim_end()
                    .to_string();
                header_cards.push(card.clone());
                if card == "END" {
                    end_found = true;
                    break;
                }
            }
        }

        if header_cards.is_empty() {
            break;
        }

        validate_header(&header_cards, hdus.is_empty())?;

        let num_header_blocks =
            ((header_cards.len() + CARDS_PER_BLOCK - 1) / CARDS_PER_BLOCK) as u64;
        let header_size = num_header_blocks * BLOCK_SIZE as u64;
        let header_offset = offset;
        let data_offset = offset + header_size;
        let data_size = calculate_data_size(&header_cards);

        let is_image = header_cards.iter().any(|c| {
            c.starts_with("SIMPLE  =") || c.starts_with("XTENSION= 'IMAGE")
        });
        let is_binary_table = header_cards.iter().any(|c| c.starts_with("XTENSION= 'BINTABLE'"));
        // Match both 'TABLE' (unpadded, 5 chars, non-conforming but
        // accepted) and 'TABLE   ' (padded to the FITS 8-char minimum
        // for string values).  Mirrors the IMAGE pattern just above.
        let is_ascii_table = header_cards.iter().any(|c| c.starts_with("XTENSION= 'TABLE"));
        // ZIMAGE convention: BINTABLE with ZIMAGE=T is a
        // tile-compressed image.  Detect here so we route to
        // CompressedImageHDU instead of TableHDU.
        let is_compressed_image = is_binary_table
            && header_has_zimage(&header_cards);

        let hdu_offsets = HduOffsets::new(
            header_offset, num_header_blocks, data_offset,
        );
        {
            let mut layout_guard = layout.hdus.lock()
                .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
            layout_guard.push(Arc::clone(&hdu_offsets));
        }

        let hdu_file = Arc::clone(handle);
        let hdu_layout = Arc::clone(layout);
        let hdu_taint = Arc::clone(tainted);
        let hdu_filename = filename.to_string();
        let hdu_py: Py<PyAny> = if is_image {
            Py::new(py, ImageHDU::new(
                header_cards.clone(), hdus.len(), hdu_filename,
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
            ))?.into()
        } else if is_compressed_image {
            Py::new(py, CompressedImageHDU::new(
                header_cards.clone(), hdus.len(), hdu_filename,
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
            ))?.into()
        } else if is_binary_table {
            Py::new(py, TableHDU::new(
                header_cards.clone(), hdus.len(), hdu_filename,
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
            ))?.into()
        } else if is_ascii_table {
            Py::new(py, AsciiTableHDU::new(
                header_cards.clone(), hdus.len(), hdu_filename,
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
            ))?.into()
        } else {
            let h = HDU::new(
                header_cards.clone(), hdus.len(), hdu_filename,
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
            );
            Py::new(py, h)?.into()
        };

        hdus.push(hdu_py);

        offset += header_size + data_size;
        let _ = file.seek(SeekFrom::Start(offset));

        if offset >= file.metadata().map(|m| m.len()).unwrap_or(0) {
            break;
        }
    }

    let _ = file.seek(SeekFrom::Start(0));
    Ok(hdus)
}

// HDU kind tag used by FITS::finalize_hdu to pick the right pyclass
// constructor when appending a freshly-written HDU.
enum HduKind {
    Image,
    Table,
    CompressedImage,
}

// Round a byte count up to the next BLOCK_SIZE boundary.  Returns 0
// when input is 0 (no data section).
fn data_section_padded(data_size: u64) -> u64 {
    if data_size == 0 {
        0
    } else {
        ((data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
            * BLOCK_SIZE as u64
    }
}

// The five cards that make up an empty primary image HDU
// (SIMPLE=T, BITPIX=8, NAXIS=0, EXTEND=T, END).  Used both as the
// auto-primary when create_table_hdu is the first call on a fresh
// file, and (in the future) anywhere else a placeholder primary is
// needed.
fn empty_primary_cards() -> Vec<String> {
    vec![
        card_logical("SIMPLE", true, "file conforms to FITS standard"),
        card_int("BITPIX", 8, "8-bit bytes"),
        card_int("NAXIS", 0, "number of data axes"),
        card_logical("EXTEND", true,
                     "FITS dataset may contain extensions"),
        pad_to_card("END"),
    ]
}

// Append one HDU (header padded to BLOCK_SIZE + zero-allocated data
// section) to the end of `file`.  Acquires the file lock for the
// duration of the write, flushes once on exit, and returns the
// freshly-constructed HduOffsets describing the appended bytes.
// Caller is responsible for registering the offsets in the file's
// layout and constructing the matching Py<HDU>.
fn append_header_and_data_to_file(
    file: &FileHandle,
    cards: &[String],
    data_padded: u64,
) -> PyResult<Arc<HduOffsets>> {
    let mut guard = lock_file(file)?;
    let f = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    let io_err = |e: std::io::Error| PyIOError::new_err(e.to_string());

    f.seek(SeekFrom::End(0)).map_err(io_err)?;
    let header_start = f.stream_position().map_err(io_err)?;

    for c in cards {
        f.write_all(c.as_bytes()).map_err(io_err)?;
    }
    let header_bytes_len = cards.len() * CARD_SIZE;
    let pad_n = (BLOCK_SIZE - header_bytes_len % BLOCK_SIZE) % BLOCK_SIZE;
    if pad_n > 0 {
        f.write_all(&vec![b' '; pad_n]).map_err(io_err)?;
    }
    let header_end = f.stream_position().map_err(io_err)?;
    if data_padded > 0 {
        let new_len = header_end + data_padded;
        f.set_len(new_len).map_err(io_err)?;
        f.seek(SeekFrom::Start(new_len)).map_err(io_err)?;
    }
    f.flush().map_err(io_err)?;
    let num_blocks = (header_end - header_start) / BLOCK_SIZE as u64;
    Ok(HduOffsets::new(header_start, num_blocks, header_end))
}

#[pyclass]
pub(crate) struct FITS {
    filename: String,
    // Held verbatim for __repr__; the open() flags are derived from
    // this at construction time and not stored separately.
    mode: String,
    file: FileHandle,
    hdus: Vec<Py<PyAny>>,
    // Shared with every HDU and FITSHeader; the upcoming grow path will
    // walk this to update offsets of subsequent HDUs in lockstep.  Owned
    // here, cloned into each HDU at construction.
    layout: Arc<FileLayout>,
    // Per-file taint flag (see TaintFlag).  Owned here; cloned into every
    // HDU and FITSHeader so a mid-write failure anywhere taints the lot.
    tainted: TaintFlag,
}

// Internal wrapper over the compression-config pyclasses.  The
// `compress=` argument to `create_image_hdu` may be any of the
// per-algorithm config classes (`Gzip1`, `Gzip2`, ...).  Extracting
// directly to one specific type would force a separate isinstance
// branch per algorithm at the call site; this enum centralises the
// "try each known class in turn" logic and exposes the small set of
// shared accessors (tile shape, heap format, on-disk ZCMPTYPE name).
enum CompressionConfigKind {
    Gzip1(crate::zimage::compression_config::Gzip1),
    Gzip2(crate::zimage::compression_config::Gzip2),
    Rice1(crate::zimage::compression_config::Rice1),
    Hcompress1(crate::zimage::compression_config::Hcompress1),
}

impl CompressionConfigKind {
    fn from_pyany(bound: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(g) = bound.extract::<
            crate::zimage::compression_config::Gzip1>()
        {
            return Ok(Self::Gzip1(g));
        }
        if let Ok(g) = bound.extract::<
            crate::zimage::compression_config::Gzip2>()
        {
            return Ok(Self::Gzip2(g));
        }
        if let Ok(r) = bound.extract::<
            crate::zimage::compression_config::Rice1>()
        {
            return Ok(Self::Rice1(r));
        }
        if let Ok(h) = bound.extract::<
            crate::zimage::compression_config::Hcompress1>()
        {
            return Ok(Self::Hcompress1(h));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "compress= must be a compression-config object \
             (e.g. rustfits.Gzip1(...), rustfits.Gzip2(...), \
             rustfits.Rice1(...), rustfits.Hcompress1(...))"
        ))
    }

    fn tile_shape(&self) -> &Option<Vec<u64>> {
        match self {
            Self::Gzip1(g) => &g.tile_shape,
            Self::Gzip2(g) => &g.tile_shape,
            Self::Rice1(r) => &r.tile_shape,
            Self::Hcompress1(h) => &h.tile_shape,
        }
    }

    fn heap_format(&self) -> char {
        match self {
            Self::Gzip1(g) => g.heap_format,
            Self::Gzip2(g) => g.heap_format,
            Self::Rice1(r) => r.heap_format,
            Self::Hcompress1(h) => h.heap_format,
        }
    }

    fn zcmptype(&self) -> &'static str {
        match self {
            Self::Gzip1(_) => "GZIP_1",
            Self::Gzip2(_) => "GZIP_2",
            Self::Rice1(_) => "RICE_1",
            Self::Hcompress1(_) => "HCOMPRESS_1",
        }
    }

    // Algorithm-specific (ZNAMEn, ZVALn) pairs to emit alongside
    // the standard ZIMAGE header cards.  RICE_1 carries BLOCKSIZE
    // and BYTEPIX so the decoder can pick the right parameter
    // table; HCOMPRESS_1 carries SCALE and SMOOTH so the decoder
    // can find the smoothing flag (and so the SCALE is documented
    // even though the decoder reads it from the stream); GZIP
    // variants have no extras.  Caller supplies the image BITPIX
    // so we can compute BYTEPIX = bitpix/8.
    fn extra_z_cards(&self, bitpix: i32) -> Vec<(&'static str, i64)> {
        match self {
            Self::Gzip1(_) | Self::Gzip2(_) => Vec::new(),
            Self::Rice1(r) => vec![
                ("BLOCKSIZE", r.blocksize as i64),
                ("BYTEPIX", (bitpix / 8) as i64),
            ],
            Self::Hcompress1(h) => vec![
                ("SCALE", h.scale as i64),
                ("SMOOTH", if h.smooth { 1 } else { 0 }),
            ],
        }
    }
}

// HCOMPRESS_1 default stripe height along the slow axis when the
// user doesn't pass tile_shape.  Direct port of cfitsio's heuristic
// in imcompress.c (the `actual_tilesize[0] <= 0` branch under the
// HCOMPRESS_1 case).  For NAXIS2 ≤ 30 the whole image is one tile
// (no benefit to striping such a small image); otherwise we pick
// the first value from the preferred list that leaves a last-tile
// remainder of 0 or ≥ 4 — i.e., one that doesn't violate the
// 4-pixel-per-dim minimum.  16 is preferred because it's the
// cfitsio default everyone sees in HST/DECam/HSC files; 24..14 are
// nearby alternatives; 17 is the last-resort fallback since it
// rarely lands the remainder in [1, 3] for typical image heights.
fn hcompress_default_slow_tile(naxis2: u64) -> u64 {
    if naxis2 <= 30 {
        return naxis2;
    }
    for &t in &[16u64, 24, 20, 30, 28, 26, 22, 18, 14] {
        let r = naxis2 % t;
        if r == 0 || r >= 4 {
            return t;
        }
    }
    17
}

// Rust-only helpers on FITS — not exposed to Python.  Used by the
// create_image_hdu / create_table_hdu / ensure_primary code paths to
// avoid duplicating the "register HduOffsets + construct Py<HDU> +
// push to self.hdus" pattern.
impl FITS {
    // Register an Arc<HduOffsets> in the file's layout, construct the
    // matching Py<HDU> (image or table), trim cards to the canonical
    // in-memory form, and push the HDU onto self.hdus.  The HDU's
    // index is set to the post-push position automatically.
    fn finalize_hdu(
        &mut self,
        py: Python<'_>,
        cards: &[String],
        offsets: Arc<HduOffsets>,
        kind: HduKind,
    ) -> PyResult<()> {
        {
            let mut lg = self.layout.hdus.lock()
                .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
            lg.push(Arc::clone(&offsets));
        }
        // Match what the on-disk parser would yield on a re-read: the
        // header reader trims trailing whitespace from each 80-char
        // card.  Cloning that here keeps the in-memory representation
        // byte-equivalent to a fresh open.
        let trimmed: Vec<String> = cards.iter()
            .map(|c| c.trim_end().to_string())
            .collect();
        let index = self.hdus.len();
        let hdu_py: Py<PyAny> = match kind {
            HduKind::Image => Py::new(py, ImageHDU::new(
                trimmed, index, self.filename.clone(),
                offsets, Arc::clone(&self.layout),
                Arc::clone(&self.file), Arc::clone(&self.tainted),
            ))?.into(),
            HduKind::Table => Py::new(py, TableHDU::new(
                trimmed, index, self.filename.clone(),
                offsets, Arc::clone(&self.layout),
                Arc::clone(&self.file), Arc::clone(&self.tainted),
            ))?.into(),
            HduKind::CompressedImage => Py::new(py, CompressedImageHDU::new(
                trimmed, index, self.filename.clone(),
                offsets, Arc::clone(&self.layout),
                Arc::clone(&self.file), Arc::clone(&self.tainted),
            ))?.into(),
        };
        self.hdus.push(hdu_py);
        Ok(())
    }

    // If the file has no HDUs yet, write an empty primary image
    // (SIMPLE=T NAXIS=0) and register it.  Used by create_table_hdu
    // (and any future extension-creating method) so that the user
    // doesn't have to manually create a placeholder primary before
    // their first extension.
    fn ensure_primary(&mut self, py: Python<'_>) -> PyResult<()> {
        if !self.hdus.is_empty() {
            return Ok(());
        }
        let cards = empty_primary_cards();
        let offsets = append_header_and_data_to_file(&self.file, &cards, 0)?;
        self.finalize_hdu(py, &cards, offsets, HduKind::Image)
    }

    // Create a tile-compressed image HDU.  Routes from
    // `create_image_hdu(..., compress=Gzip1(...))`.  Builds a
    // BINTABLE-with-ZIMAGE header, allocates the per-tile descriptor
    // table (n_tiles rows × 8 or 16 bytes) zero-filled, and leaves
    // the heap empty (PCOUNT=0) until CompressedImageHDU.write is
    // called.
    //
    // Phase 7 supports Gzip1, Gzip2, and Rice1 with integer ZBITPIX
    // (u1/i2/i4/i8 for GZIP; u1/i2/i4 for RICE).  Other algorithms
    // and float ZBITPIX raise NotImplementedError.
    fn create_compressed_image_hdu_impl(
        &mut self,
        py: Python<'_>,
        dtype: String,
        dims: Vec<i64>,
        extname: Option<String>,
        extver: Option<i64>,
        compress: Py<PyAny>,
    ) -> PyResult<()> {
        for (i, &d) in dims.iter().enumerate() {
            if d <= 0 {
                return Err(PyValueError::new_err(format!(
                    "dimension {} must be > 0, got {}", i, d
                )));
            }
        }
        if dims.is_empty() {
            return Err(PyValueError::new_err(
                "compressed images must have NAXIS >= 1"
            ));
        }

        // Extract the compress config.  Try each supported algorithm
        // class in turn; the resulting wrapper carries the algorithm
        // identity (for ZCMPTYPE) and the shared tile_shape /
        // heap_format params used by both encoders.
        let bound = compress.bind(py);
        let cfg = CompressionConfigKind::from_pyany(bound)?;

        // Dtype validation: Phase 7 supports BITPIX-direct integer
        // types (u1/i2/i4/i8) only.  Floats deferred to Phase 8
        // (quantization); unsigned-int trick types (i1/u2/u4/u8)
        // deferred until we wire the reverse-cast on the write side.
        let (bitpix, bzero) = crate::hdu_image::dtype_to_bitpix(&dtype)?;
        if bitpix < 0 {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "compressed float images are not yet supported \
                 (planned: Phase 8 — float quantization)"
            ));
        }
        if bzero.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "compressed unsigned-int trick dtypes (i1/u2/u4/u8) \
                 are not yet supported on write; pass the matching \
                 signed dtype (i1→u1 mismatch, u2→i2, u4→i4, u8→i8) \
                 for now, or wait for a Phase 7 follow-up"
            ));
        }
        // RICE_1 rejects bitpix=64 (BYTEPIX=8).  cfitsio has no
        // 64-bit RICE encoder; producing such files would make
        // them unreadable outside rustfits.  GZIP_2 typically
        // gets within ~5% on real i64 imagery.
        if matches!(cfg, CompressionConfigKind::Rice1(_)) && bitpix == 64 {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "RICE_1 does not support 64-bit pixels (i8 dtype): no \
                 canonical FITS writer (cfitsio, fitsio, astropy) \
                 produces such files. Use Gzip2 for i64 imaging data \
                 — typically within ~5% of RICE compression and \
                 universally readable."
            ));
        }
        // HCOMPRESS_1 is a 2-D wavelet algorithm; only 2-D images
        // are valid.  Also reject bitpix=64 — the FITS Tile
        // Compression Convention has no 64-bit HCOMPRESS variant
        // and cfitsio's encoder family stops at i32 input (i64
        // internal precision).
        if matches!(cfg, CompressionConfigKind::Hcompress1(_)) {
            if dims.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "Hcompress1 only supports 2-D images; got {}-D",
                    dims.len()
                )));
            }
            if bitpix == 64 {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "HCOMPRESS_1 does not support 64-bit pixels (i8 \
                     dtype): the FITS Tile Compression Convention has \
                     no 64-bit HCOMPRESS variant. Use Gzip2 for i64 \
                     imaging data."
                ));
            }
        }

        // Build the tile shape in numpy axis order.
        //
        // None → algorithm-specific default:
        //   - HCOMPRESS_1: cfitsio's default heuristic.  Full image
        //     along the fast axis; along the slow axis, full image
        //     when NAXIS2 ≤ 30 (single-tile small-image case), else
        //     the first value from {16, 24, 20, 30, 28, 26, 22, 18,
        //     14} that leaves a last-tile remainder of 0 or ≥ 4,
        //     falling back to 17 (which is unlikely to leave a bad
        //     remainder).  Matches what HST / DECam / HSC files in
        //     the wild use (cfitsio is the dominant writer).
        //   - Other algorithms: FITS-convention "row tiles"
        //     (ZTILE1=NAXIS1, others=1), which is `[1, ..., 1,
        //     NAXIS_last]` in numpy order since numpy-last
        //     corresponds to FITS-NAXIS1.
        let numpy_dims: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        let tile_shape_numpy: Vec<u64> = match cfg.tile_shape() {
            Some(ts) => {
                if ts.len() != numpy_dims.len() {
                    return Err(PyValueError::new_err(format!(
                        "tile_shape has {} dimensions but image has {}",
                        ts.len(), numpy_dims.len()
                    )));
                }
                ts.clone()
            }
            None => {
                if matches!(cfg, CompressionConfigKind::Hcompress1(_)) {
                    vec![
                        hcompress_default_slow_tile(numpy_dims[0]),
                        numpy_dims[1],
                    ]
                } else {
                    let n = numpy_dims.len();
                    let mut v = vec![1u64; n];
                    v[n - 1] = numpy_dims[n - 1];
                    v
                }
            }
        };

        // HCOMPRESS_1 tile-shape constraints (FITS Tile Compression
        // Convention): every dimension must have at least 4 pixels,
        // and every tile (including the last along each axis) must
        // have at least 4 pixels.  astropy raises in this case;
        // cfitsio silently rewrites the tile dim upward.  We follow
        // astropy — explicit is safer — and suggest the cfitsio-style
        // adjusted value in the error so the user can just copy it
        // back into their config.
        if matches!(cfg, CompressionConfigKind::Hcompress1(_)) {
            for (axis, (&dim, &tile)) in numpy_dims.iter()
                .zip(tile_shape_numpy.iter()).enumerate()
            {
                if dim < 4 {
                    return Err(PyValueError::new_err(format!(
                        "Hcompress1: image axis {} has size {}, below \
                         the HCOMPRESS_1 minimum of 4 pixels per \
                         dimension",
                        axis, dim,
                    )));
                }
                if tile < 4 {
                    return Err(PyValueError::new_err(format!(
                        "Hcompress1: tile_shape[{}]={} is below the \
                         HCOMPRESS_1 minimum of 4 pixels per dimension",
                        axis, tile,
                    )));
                }
                let remain = dim % tile;
                if remain > 0 && remain < 4 {
                    // cfitsio's adjustment: tile += ceil(remain / ndiv)
                    // where ndiv = dim / tile (integer truncation).
                    // ndiv >= 1 here because dim >= 4 and remain < 4
                    // implies tile <= dim - 4 < dim, so dim / tile >= 1.
                    let ndiv = dim / tile;
                    let add = (remain + ndiv - 1) / ndiv;
                    let suggested = tile + add;
                    return Err(PyValueError::new_err(format!(
                        "Hcompress1: image axis {} (size {}) with \
                         tile_shape[{}]={} leaves a last tile of {} \
                         pixels, below the HCOMPRESS_1 minimum of 4. \
                         Try tile_shape[{}]={} (last tile {} pixels) \
                         to satisfy the constraint.",
                        axis, dim, axis, tile, remain,
                        axis, suggested, dim % suggested,
                    )));
                }
            }
        }

        let n_tiles = crate::hdu_image_compressed::compute_n_tiles(
            &numpy_dims, &tile_shape_numpy,
        );

        // Compressed images can't be the primary HDU (they're stored
        // as BINTABLE).  Auto-write an empty primary first if the
        // file is fresh — same as create_table_hdu does.
        self.ensure_primary(py)?;

        let heap_format = cfg.heap_format();
        let descriptor_size: u64 = if heap_format == 'P' { 8 } else { 16 };
        let tform_val = if heap_format == 'P' { "1PB" } else { "1QB" };

        // FITS-order copies of image + tile shapes for the Z* cards.
        let fits_dims: Vec<u64> = numpy_dims.iter().rev().copied().collect();
        let tile_shape_fits: Vec<u64> =
            tile_shape_numpy.iter().rev().copied().collect();

        let mut cards: Vec<String> = Vec::new();
        cards.push(card_string("XTENSION", "BINTABLE",
                               "binary table extension"));
        cards.push(card_int("BITPIX", 8, "8-bit bytes"));
        cards.push(card_int("NAXIS", 2, "2-dimensional binary table"));
        cards.push(card_int("NAXIS1", descriptor_size as i64,
                            "width of table row in bytes"));
        cards.push(card_int("NAXIS2", n_tiles as i64,
                            "number of rows in table (= n_tiles)"));
        cards.push(card_int("PCOUNT", 0, "size of heap in bytes"));
        cards.push(card_int("GCOUNT", 1, "one data group"));
        cards.push(card_int("TFIELDS", 1, "number of fields per row"));
        cards.push(card_string("TFORM1", tform_val,
                               "VLA byte-array descriptor"));
        cards.push(card_string("TTYPE1", "COMPRESSED_DATA",
                               "label for column 1"));
        cards.push(card_logical("ZIMAGE", true,
                                "tile-compressed image"));
        cards.push(card_string("ZCMPTYPE", cfg.zcmptype(),
                               "compression algorithm"));
        cards.push(card_int("ZBITPIX", bitpix as i64,
                            "image bits per pixel"));
        cards.push(card_int("ZNAXIS", dims.len() as i64,
                            "image dimensions"));
        for (i, &d) in fits_dims.iter().enumerate() {
            cards.push(card_int(&format!("ZNAXIS{}", i + 1), d as i64,
                                &format!("image axis {}", i + 1)));
        }
        for (i, &t) in tile_shape_fits.iter().enumerate() {
            cards.push(card_int(&format!("ZTILE{}", i + 1), t as i64,
                                &format!("tile size on axis {}", i + 1)));
        }
        // Algorithm-specific ZNAMEn/ZVALn pairs (RICE BLOCKSIZE +
        // BYTEPIX; GZIP has none).
        for (n, (name, val)) in cfg.extra_z_cards(bitpix).iter().enumerate() {
            let idx = n + 1;
            cards.push(card_string(
                &format!("ZNAME{}", idx), name,
                &format!("compression parameter {}", idx),
            ));
            cards.push(card_int(
                &format!("ZVAL{}", idx), *val,
                &format!("value of ZNAME{}", idx),
            ));
        }
        if let Some(name) = extname.as_deref() {
            cards.push(card_string("EXTNAME", name, "name of this HDU"));
        }
        if let Some(ver) = extver {
            cards.push(card_int("EXTVER", ver, "extension version"));
        }
        cards.push(pad_to_card("END"));

        // Main data section: one descriptor per tile, all zeroes
        // (nelements=0, offset=0) until CompressedImageHDU.write
        // populates them.  Heap is empty (PCOUNT=0).
        let data_size = descriptor_size.saturating_mul(n_tiles);
        let data_padded = data_section_padded(data_size);

        let offsets =
            append_header_and_data_to_file(&self.file, &cards, data_padded)?;
        self.finalize_hdu(py, &cards, offsets, HduKind::CompressedImage)
    }
}

#[pymethods]
impl FITS {
    // Default mode is 'r' so FITS(filename) reads — matches the
    // built-in open(filename) convention.  'r+' opens for in-place
    // mutation; 'w+' truncates / creates.
    #[new]
    #[pyo3(signature = (filename, mode="r"))]
    fn new(py: Python<'_>, filename: String, mode: &str) -> PyResult<Self> {
        let mut options = OpenOptions::new();

        match mode {
            "r"  => options.read(true),
            "r+" => options.read(true).write(true),
            "w+" => options.read(true).write(true).truncate(true).create(true),
            _ => return Err(PyIOError::new_err(format!(
                "Unsupported mode '{}'. Supported modes: 'r', 'r+', 'w+'",
                mode
            ))),
        };

        let file = options.open(&filename)
            .map_err(|e| PyIOError::new_err(format!("Failed to open '{}': {}", filename, e)))?;

        let handle: FileHandle = Arc::new(Mutex::new(Some(file)));
        let tainted: TaintFlag = Arc::new(AtomicBool::new(false));
        let layout = FileLayout::new();
        let hdus = parse_hdus_from_file(
            py, &filename, &handle, &layout, &tainted,
        )?;

        Ok(FITS {
            filename,
            mode: mode.to_string(),
            file: handle,
            hdus,
            layout,
            tainted,
        })
    }

    #[getter]
    fn hdus(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::new(py, &self.hdus)?;
        Ok(list.unbind())
    }

    #[getter]
    fn filename(&self) -> String {
        self.filename.clone()
    }

    fn close(&mut self) -> PyResult<()> {
        let mut guard = lock_file(&self.file)?;
        if let Some(file) = guard.take() {
            let _ = file.sync_all();
        }
        Ok(())
    }

    #[getter]
    fn closed(&self) -> PyResult<bool> {
        let guard = lock_file(&self.file)?;
        Ok(guard.is_none())
    }

    // Multi-line, fitsio-style repr.  Typing the bound name + Enter in
    // a REPL calls __repr__ (not __str__), so the rich layout has to
    // live here.  For a healthy open file:
    //
    //   file: foo.fits
    //   mode: r+
    //   extnum  hdutype     extname
    //   0       IMAGE_HDU
    //   1       BINARY_TBL  MYTABLE
    //
    // For a closed/poisoned file we skip the per-HDU table (the HDU
    // refs themselves still work for header inspection, but pulling
    // EXTNAME may go through the file lock, and the cleaner thing is
    // just to show the status and return).
    fn __repr__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        let status = match slf.file.lock() {
            Ok(guard) if guard.is_none() => "closed",
            Ok(_) => "open",
            Err(_) => "poisoned",
        };

        let mut out = String::new();
        out.push_str(&format!("  file: {}\n", slf.filename));
        out.push_str(&format!("  mode: {}\n", slf.mode));
        if status != "open" {
            out.push_str(&format!("  status: {}\n", status));
            return Ok(out);
        }

        out.push_str("  extnum  hdutype     extname\n");
        for (i, hdu) in slf.hdus.iter().enumerate() {
            let hdu_bound = hdu.bind(py);
            let kind = if hdu_bound.is_instance_of::<ImageHDU>() {
                "IMAGE_HDU"
            } else if hdu_bound.is_instance_of::<TableHDU>() {
                "BINARY_TBL"
            } else if hdu_bound.is_instance_of::<AsciiTableHDU>() {
                "ASCII_TBL"
            } else {
                "UNKNOWN"
            };
            // Every HDU subclass extends HDU, so this downcast succeeds.
            let base = hdu_bound.cast::<HDU>()?.borrow();
            let cards = base.header_snapshot()?;
            let extname = parse_string_keyword(&cards, "EXTNAME")
                .unwrap_or_default();
            out.push_str(&format!(
                "  {:<7} {:<11} {}\n", i, kind, extname,
            ));
        }
        Ok(out)
    }

    fn __len__(&self) -> usize {
        self.hdus.len()
    }

    // Make FITS iterable: `for hdu in fits` walks the HDUs in file
    // order, same as `for hdu in fits.hdus`.  Matches fitsio's API.
    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = PyList::new(py, &self.hdus)?;
        Ok(list.try_iter()?.into_any().unbind())
    }

    // Create a new image HDU.  `dtype` follows numpy short-code convention
    // (e.g. 'f8', 'i4').  `dims` is the array shape in numpy (row-major)
    // order and is reversed internally to produce FITS NAXISn.  The first
    // HDU created becomes the primary HDU (SIMPLE=T, EXTEND=T); subsequent
    // calls produce 'IMAGE' extensions.  The data section is allocated as
    // zeros via sparse file extension.
    //
    // `compress`, when non-None, is a compression-config object (e.g.
    // `rustfits.Gzip1(tile_shape=..., heap_format='P')`).  In that case
    // the HDU is created as a tile-compressed image (BINTABLE+ZIMAGE on
    // disk, `CompressedImageHDU` in Python) instead of a plain IMAGE
    // extension.  Phase 7 supports `Gzip1`, `Gzip2`, and `Rice1`;
    // the remaining algorithms (HCOMPRESS_1, PLIO_1) will be added
    // in follow-up sub-phases.
    #[pyo3(signature = (dtype, dims, *, extname=None, extver=None, compress=None))]
    fn create_image_hdu(
        &mut self,
        py: Python<'_>,
        dtype: String,
        dims: Vec<i64>,
        extname: Option<String>,
        extver: Option<i64>,
        compress: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if let Some(cfg) = compress {
            return self.create_compressed_image_hdu_impl(
                py, dtype, dims, extname, extver, cfg,
            );
        }
        for (i, &d) in dims.iter().enumerate() {
            if d <= 0 {
                return Err(PyValueError::new_err(format!(
                    "dimension {} must be > 0, got {}", i, d
                )));
            }
        }

        let (bitpix, bzero) = dtype_to_bitpix(&dtype)?;
        let naxis = dims.len() as i64;

        // numpy (row-major) -> FITS (NAXIS1 is fastest-varying): reverse.
        let fits_dims: Vec<i64> = dims.iter().rev().copied().collect();

        let is_primary = self.hdus.is_empty();

        let mut cards: Vec<String> = Vec::new();
        if is_primary {
            cards.push(card_logical("SIMPLE", true, "file conforms to FITS standard"));
        } else {
            cards.push(card_string("XTENSION", "IMAGE", "image extension"));
        }
        cards.push(card_int("BITPIX", bitpix as i64, "number of bits per data pixel"));
        cards.push(card_int("NAXIS", naxis, "number of data axes"));
        for (i, &d) in fits_dims.iter().enumerate() {
            cards.push(card_int(
                &format!("NAXIS{}", i + 1),
                d,
                &format!("length of data axis {}", i + 1),
            ));
        }
        if is_primary {
            cards.push(card_logical("EXTEND", true, "FITS dataset may contain extensions"));
        } else {
            cards.push(card_int("PCOUNT", 0, "required keyword; must = 0"));
            cards.push(card_int("GCOUNT", 1, "required keyword; must = 1"));
        }
        if let Some(name) = extname.as_deref() {
            cards.push(card_string("EXTNAME", name, "name of this HDU"));
        }
        if let Some(ver) = extver {
            cards.push(card_int("EXTVER", ver, "extension version"));
        }
        // Unsigned-int trick: user-facing dtype was u2/u4/u8/i1 but the
        // on-disk BITPIX is the opposite signedness.  Emit BZERO so
        // readers (rustfits + astropy + cfitsio) recover the original
        // dtype on read.  BSCALE=1 is the default but is emitted
        // alongside for clarity.  Use card_uint for the u8 case (BZERO
        // = 2^63 overflows i64); card_int for the others.
        if let Some(bz) = bzero {
            cards.push(card_int(
                "BSCALE", 1, "default linear scaling"));
            let bz_card = if bz > i64::MAX as f64 {
                card_uint(
                    "BZERO", bz as u64,
                    "offset for unsigned-int storage")
            } else {
                card_int(
                    "BZERO", bz as i64,
                    "offset for unsigned-int storage")
            };
            cards.push(bz_card);
        }
        cards.push(pad_to_card("END"));

        let bytes_per_pixel = (bitpix.abs() / 8) as u64;
        let mut product: u64 = 1;
        for &d in &fits_dims {
            product = product.saturating_mul(d as u64);
        }
        let data_size = if naxis == 0 { 0 } else { bytes_per_pixel * product };
        let data_padded = data_section_padded(data_size);

        let offsets =
            append_header_and_data_to_file(&self.file, &cards, data_padded)?;
        self.finalize_hdu(py, &cards, offsets, HduKind::Image)
    }

    // Create a new BINTABLE extension HDU.  `dtype` is either a
    // numpy.dtype OR a "descr" list of tuples (any form numpy.dtype()
    // accepts) — it is normalized internally to a structured dtype.
    // `nrows` (default 0) is the row count to allocate in the data
    // section; subsequent TableHDU.write(arr) currently requires
    // len(arr) == nrows.
    //
    // If the file has no HDUs yet, an empty primary image
    // (SIMPLE=T, NAXIS=0) is automatically written first so that the
    // BINTABLE can land as an extension — the FITS standard forbids
    // BINTABLE as the primary HDU.
    //
    // MVP supports scalar fields with i2/i4/i8/u1/f4/f8 dtypes.
    // Subsequent commits add the unsigned-int trick (u2/u4/u8),
    // bool/complex, subarray fields, strings, and dict/list+names
    // input forms.  See the Table Write Roadmap in CLAUDE.md.
    #[pyo3(signature = (
        dtype, nrows=0, *,
        extname=None, extver=None, units=None,
        var_dtypes=None, heap_format=None
    ))]
    fn create_table_hdu(
        &mut self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        nrows: i64,
        extname: Option<String>,
        extver: Option<i64>,
        units: Option<&Bound<'_, PyDict>>,
        var_dtypes: Option<&Bound<'_, PyDict>>,
        heap_format: Option<String>,
    ) -> PyResult<()> {
        if nrows < 0 {
            return Err(PyValueError::new_err(format!(
                "create_table_hdu: nrows must be >= 0, got {}", nrows)));
        }
        // heap_format is 'P' (default — 8-byte descriptors, 4 GB heap
        // ceiling) or 'Q' (16-byte, no practical ceiling).  Only
        // relevant when any VLA columns are declared; ignored
        // otherwise.  The name refers to how the VLA heap is
        // addressed in the BINTABLE row; values match the FITS
        // TFORM letter (`1PE` vs `1QE`).
        let desc_char = match heap_format.as_deref() {
            None | Some("P") | Some("p") => 'P',
            Some("Q") | Some("q") => 'Q',
            Some(other) => return Err(PyValueError::new_err(format!(
                "create_table_hdu: heap_format must be 'P' or 'Q', got '{}'",
                other))),
        };
        let (table_cards, row_width) = normalize_and_build_table_header(
            py, dtype, nrows, extname.as_deref(), extver, units,
            var_dtypes, desc_char,
        )?;
        let data_size = (nrows as u64).saturating_mul(row_width);
        let data_padded = data_section_padded(data_size);

        // BINTABLE cannot be primary — write an empty primary image
        // first if the file has no HDUs yet.
        self.ensure_primary(py)?;

        let offsets = append_header_and_data_to_file(
            &self.file, &table_cards, data_padded)?;
        self.finalize_hdu(py, &table_cards, offsets, HduKind::Table)
    }

    // Accept either an integer (positional, with Python-style negative
    // indexing) or a string (EXTNAME lookup, case-insensitive).  A bool is
    // rejected explicitly because Python's int/bool subclass relationship
    // would otherwise let `fits[True]` resolve as `fits[1]`.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if key.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(
                "FITS index must be int (HDU position) or str (EXTNAME); got bool",
            ));
        }
        if let Ok(index) = key.extract::<isize>() {
            let len = self.hdus.len() as isize;
            let idx = if index < 0 { len + index } else { index };
            if idx < 0 || idx >= len {
                return Err(PyValueError::new_err(format!(
                    "HDU index {} out of range", index
                )));
            }
            return Ok(self.hdus[idx as usize].clone_ref(py));
        }
        // Accept str (incl. np.str_, which subclasses str) and bytes
        // (incl. np.bytes_, which subclasses bytes).  FITS keyword and
        // string values are restricted to printable ASCII by spec, so a
        // non-ASCII byte sequence can't match anything and is rejected.
        //
        // Type checks are explicit (PyString / PyBytes) rather than
        // relying on extract::<String>() / extract::<Vec<u8>>() — the
        // latter is generic over iterables, so a list of small ints
        // like [5, 0, 2] would silently succeed as Vec<u8>=[5,0,2] and
        // be misinterpreted as a control-character EXTNAME lookup.
        let name: Option<String> = if key.is_instance_of::<PyString>() {
            Some(key.extract::<String>()?)
        } else if key.is_instance_of::<PyBytes>() {
            let b: Vec<u8> = key.extract()?;
            if !b.iter().all(|c| c.is_ascii()) {
                return Err(PyValueError::new_err(
                    "FITS EXTNAME lookup key must be ASCII",
                ));
            }
            Some(String::from_utf8(b).unwrap())
        } else {
            None
        };
        if let Some(name) = name {
            let target = name.trim().to_ascii_uppercase();
            for hdu_obj in &self.hdus {
                let bound = hdu_obj.bind(py);
                let hdu_ref = bound.cast::<HDU>()?.borrow();
                let cards_guard = hdu_ref.header.lock()
                    .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
                let matched = parse_string_keyword(&cards_guard, "EXTNAME")
                    .map(|s| s.trim().to_ascii_uppercase() == target)
                    .unwrap_or(false);
                drop(cards_guard);
                if matched {
                    return Ok(hdu_obj.clone_ref(py));
                }
            }
            return Err(PyValueError::new_err(format!("no HDU named '{}'", name)));
        }
        Err(PyValueError::new_err(
            "FITS index must be int (HDU position) or str/bytes (EXTNAME)",
        ))
    }

    fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let _ = self.close();
        Ok(false)
    }
}
