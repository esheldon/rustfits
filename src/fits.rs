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
use crate::hdu_table_compressed::{
    build_compressed_table_header, default_ztilelen, header_has_ztable,
    resolve_compress_arg, CompressedTableHDU,
};
use crate::hdu_table::{
    normalize_and_build_table_header, parse_columns, BitColumnsSpec,
    TableHDU,
};
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
        raw_size.div_ceil(BLOCK_SIZE as u64) * BLOCK_SIZE as u64
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
            header_cards.len().div_ceil(CARDS_PER_BLOCK) as u64;
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
        // ZTABLE convention: BINTABLE with ZTABLE=T is a
        // tile-compressed table.  Routes to CompressedTableHDU.
        // ZIMAGE wins over ZTABLE if both are somehow set
        // (shouldn't happen on a valid file).
        let is_compressed_table = is_binary_table
            && !is_compressed_image
            && header_has_ztable(&header_cards);

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
            // Reopened from disk: both write-only configs
            // (quantize_config, compress_config) are None.  Write
            // path falls back to defaults (qlevel=4.0, gzip level=
            // codec default 6); method+seed recover from ZQUANTIZ
            // + ZDITHER0 cards.  The .compression accessor builds
            // a fresh CompressionConfigKind from the cards.
            Py::new(py, CompressedImageHDU::new(
                header_cards.clone(), hdus.len(), hdu_filename,
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
                None, None,
            ))?.into()
        } else if is_compressed_table {
            // Reopened from disk: per-column configs unknown
            // (level= etc. aren't stored on disk).  .compression
            // falls back to dict-of-strings from ZCTYPn cards.
            Py::new(py, CompressedTableHDU::new(
                header_cards.clone(), hdus.len(), hdu_filename,
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
                None,
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
// constructor when appending a freshly-written HDU.  The
// `CompressedImage` variant carries the create-time `Quantize`
// config (None for integer compressed HDUs; Some for float ones)
// so the encoder can recover the user's qlevel at write time —
// the FITS Tile Compression Convention only records the method
// and seed on disk, not the level.
enum HduKind {
    Image,
    Table,
    CompressedImage {
        quantize: Option<crate::zimage::compression_config::Quantize>,
        compress_config: Option<CompressionConfigKind>,
    },
    CompressedTable {
        compress_configs: Option<Vec<CompressionConfigKind>>,
    },
}

// Round a byte count up to the next BLOCK_SIZE boundary.  Returns 0
// when input is 0 (no data section).
// Parse the `bit_columns=` kwarg.  Accepted forms:
//   - None / Python None: no opt-in (default; b1 → L).
//   - bool: True promotes ALL b1 columns to X; False is the same as
//     None (explicit "no").  Matches fitsio's write_bitcols=True
//     global toggle.
//   - list/tuple of str: per-column opt-in.  Names are folded to
//     uppercase for case-insensitive matching against the table
//     columns; duplicates are tolerated (HashSet dedup).
// Everything else is rejected with a clear type-error message.
fn parse_bit_columns_arg(
    arg: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<BitColumnsSpec>> {
    use pyo3::types::{PyBool, PyList, PyTuple};
    let Some(v) = arg else { return Ok(None); };
    if v.is_none() {
        return Ok(None);
    }
    // bool BEFORE int extract — Python bool is an int subclass and
    // would otherwise route through the iterable arm.
    if v.is_instance_of::<PyBool>() {
        let b: bool = v.extract()?;
        return Ok(if b { Some(BitColumnsSpec::All) } else { None });
    }
    if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() {
        let mut set = std::collections::HashSet::new();
        for item in v.try_iter()? {
            let name: String = item?.extract().map_err(|_| {
                PyValueError::new_err(
                    "bit_columns: list entries must be strings")
            })?;
            set.insert(name.to_uppercase());
        }
        return Ok(Some(BitColumnsSpec::Names(set)));
    }
    Err(PyValueError::new_err(
        "bit_columns: must be None, a bool, or a list/tuple of str \
         column names",
    ))
}

fn data_section_padded(data_size: u64) -> u64 {
    if data_size == 0 {
        0
    } else {
        data_size.div_ceil(BLOCK_SIZE as u64)
            * BLOCK_SIZE as u64
    }
}

// Convert a user-supplied BLANK/ZBLANK value from PHYSICAL space
// (the user's dtype) to STORED space (the on-disk BITPIX).  For
// plain integer dtypes (no unsigned trick) physical == stored and
// this is the identity.  For unsigned-trick dtypes
// (i1/u2/u4/u8) it subtracts BZERO so the card matches what's on
// disk after reverse_unsigned_trick has XOR'd the sign bit.
//
// Validates the stored value fits the signed BITPIX range
// [BITPIX_min, BITPIX_max].  Rejects float BITPIX (BLANK is
// spec-forbidden on floating-point arrays).
fn physical_blank_to_stored(
    physical: i64, bitpix: i32, bzero: Option<f64>,
) -> PyResult<i64> {
    if bitpix < 0 {
        return Err(PyValueError::new_err(format!(
            "blank= is not valid on float BITPIX ({}); the FITS \
             standard forbids BLANK on floating-point arrays \
             (NaN serves that role).  Omit blank= for float images.",
            bitpix
        )));
    }
    // Transform to stored space.  For unsigned-trick: stored =
    // physical - BZERO (cast through f64 because BZERO=2^63 for u8
    // is exactly representable in f64 but overflows i64).
    let stored: i64 = match bzero {
        None => physical,
        Some(bz) => {
            let s = (physical as f64) - bz;
            if !s.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "blank value {} produces a non-finite stored \
                     value after the unsigned-int trick transform",
                    physical
                )));
            }
            s as i64
        }
    };
    let (lo, hi) = match bitpix {
        8 => (0i64, 255i64),         // u1, no trick → unsigned range
        16 => (i16::MIN as i64, i16::MAX as i64),
        32 => (i32::MIN as i64, i32::MAX as i64),
        64 => (i64::MIN, i64::MAX),
        _ => unreachable!("integer BITPIX expected, got {}", bitpix),
    };
    if stored < lo || stored > hi {
        return Err(PyValueError::new_err(format!(
            "blank value {} (stored as {} after any unsigned-trick \
             transform) is outside the legal BITPIX={} stored range \
             [{}, {}]",
            physical, stored, bitpix, lo, hi,
        )));
    }
    Ok(stored)
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

/// A FITS file open for reading, writing, or both.
///
/// The top-level entry point of rustfits.  Open an existing file
/// or create a new one, then index or iterate to reach individual
/// HDUs::
///
///     # Read an existing file
///     with rustfits.FITS("data.fits", "r") as fits:
///         for hdu in fits:
///             print(hdu.extname, hdu.has_data)
///         sci = fits["SCI"]            # by EXTNAME
///         hdu2 = fits[2]               # by position
///         arr = fits[1].read()
///
///     # Append a new HDU to an existing file
///     with rustfits.FITS("data.fits", "r+") as fits:
///         fits.create_image_hdu("f4", (100, 100), extname="MODEL")
///         fits[-1].write(model)
///
///     # Create a new file (or truncate an existing one)
///     with rustfits.FITS("out.fits", "w+") as fits:
///         fits.create_table_hdu(my_dtype, nrows=1000)
///         fits[1].write(rows)
///
/// Parameters
/// ----------
/// filename : str
///     Path to the FITS file.
/// mode : {'r', 'r+', 'w+'}, optional
///     File open mode.
///
///     * ``'r'`` (default) — read-only; the file must exist.
///     * ``'r+'`` — read+write; the file must exist.  Used to
///       modify or append HDUs to an existing file.
///     * ``'w+'`` — read+write; creates the file if it doesn't
///       exist, **truncates** to zero length if it does.
///       Equivalent to fitsio's ``'rw'`` + ``clobber=True``.
///
/// Notes
/// -----
/// Use as a context manager (``with rustfits.FITS(...) as fits:``)
/// to guarantee the file handle is closed.  ``len(fits)`` returns
/// the HDU count; iteration yields each HDU in file order.
// `FITS` is the public Python class name (rustfits.FITS); the acronym
// can't be lowercased without breaking the API.
#[allow(clippy::upper_case_acronyms)]
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

use crate::zimage::compression_config::CompressionConfigKind;

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

// Derive a structured numpy dtype and row count from one of the three
// `write_table` input forms.  This is the table-side counterpart of
// numpy.asanyarray's dtype-extraction; `create_table_hdu` is kept
// schema-only (first arg = dtype), and this helper handles the
// data-shaped inputs in one place.
//
// Accepted shapes:
//   - structured ndarray (dtype.names is not None) -> use data.dtype,
//     len(data) as nrows
//   - dict {name: ndarray} -> compose np.dtype([(name, arr.dtype) for
//     ...]); nrows is the (validated-equal) length of each column
//   - list/tuple of ndarrays + names= -> same as dict, names supplied
//     as the second arg
fn derive_table_schema_from_data<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    names: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyAny>, i64)> {
    let np = py.import("numpy")?;

    // Dict input: {name: ndarray}.  names= is meaningless (the dict
    // keys ARE the names) and rejected so a user typo doesn't silently
    // get ignored.
    if let Ok(d) = data.cast::<PyDict>() {
        if names.is_some() {
            return Err(PyValueError::new_err(
                "write_table: names= is not valid with dict input \
                 (the dict keys are the column names)"));
        }
        if d.is_empty() {
            return Err(PyValueError::new_err(
                "write_table: dict input is empty (no columns)"));
        }
        let descr = PyList::empty(py);
        let mut nrows: Option<i64> = None;
        for (key, val) in d.iter() {
            let name: String = key.extract().map_err(|_| {
                PyValueError::new_err(
                    "write_table: dict keys must be strings (column names)")
            })?;
            let arr = np.call_method1("asanyarray", (val,))?;
            let n: i64 = arr.len()? as i64;
            match nrows {
                Some(prev) if prev != n => {
                    return Err(PyValueError::new_err(format!(
                        "write_table: column lengths disagree \
                         (column '{}' has {} rows, previous columns had {})",
                        name, n, prev)));
                }
                None => nrows = Some(n),
                _ => {}
            }
            let dtype = arr.getattr("dtype")?;
            descr.append((name, dtype))?;
        }
        let composed = np.call_method1("dtype", (descr,))?;
        return Ok((composed, nrows.unwrap_or(0)));
    }

    // List / tuple of ndarrays — requires names= alongside.
    let is_list = data.is_instance_of::<PyList>();
    let is_tuple = data.is_instance_of::<pyo3::types::PyTuple>();
    if is_list || is_tuple {
        let names = names.ok_or_else(|| PyValueError::new_err(
            "write_table: list/tuple input requires names= \
             (one column name per array)"))?;
        let name_list: Vec<String> = names.extract().map_err(|_| {
            PyValueError::new_err(
                "write_table: names= must be a list/tuple of strings")
        })?;
        let data_seq: Vec<Bound<PyAny>> = data.extract()?;
        if data_seq.len() != name_list.len() {
            return Err(PyValueError::new_err(format!(
                "write_table: names= length {} does not match \
                 data length {}",
                name_list.len(), data_seq.len())));
        }
        if data_seq.is_empty() {
            return Err(PyValueError::new_err(
                "write_table: empty list/tuple input (no columns)"));
        }
        let descr = PyList::empty(py);
        let mut nrows: Option<i64> = None;
        for (name, val) in name_list.iter().zip(data_seq.iter()) {
            let arr = np.call_method1("asanyarray", (val,))?;
            let n: i64 = arr.len()? as i64;
            match nrows {
                Some(prev) if prev != n => {
                    return Err(PyValueError::new_err(format!(
                        "write_table: column lengths disagree \
                         (column '{}' has {} rows, previous columns had {})",
                        name, n, prev)));
                }
                None => nrows = Some(n),
                _ => {}
            }
            let dtype = arr.getattr("dtype")?;
            descr.append((name.as_str(), dtype))?;
        }
        let composed = np.call_method1("dtype", (descr,))?;
        return Ok((composed, nrows.unwrap_or(0)));
    }

    // Structured ndarray — already has a named dtype.
    let dtype = data.getattr("dtype").map_err(|_| {
        PyValueError::new_err(
            "write_table: data must be a structured ndarray, dict, \
             or list/tuple of ndarrays + names=")
    })?;
    let dnames = dtype.getattr("names")?;
    if dnames.is_none() {
        return Err(PyValueError::new_err(
            "write_table: ndarray must have a structured dtype \
             (with named fields); got a plain dtype"));
    }
    if names.is_some() {
        return Err(PyValueError::new_err(
            "write_table: names= is not valid with structured ndarray \
             input (the dtype already names the columns)"));
    }
    let nrows: i64 = data.len()? as i64;
    Ok((dtype, nrows))
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
            HduKind::CompressedImage { quantize, compress_config } => {
                Py::new(py, CompressedImageHDU::new(
                    trimmed, index, self.filename.clone(),
                    offsets, Arc::clone(&self.layout),
                    Arc::clone(&self.file), Arc::clone(&self.tainted),
                    quantize,
                    compress_config,
                ))?.into()
            }
            HduKind::CompressedTable { compress_configs } => {
                Py::new(py, CompressedTableHDU::new(
                    trimmed, index, self.filename.clone(),
                    offsets, Arc::clone(&self.layout),
                    Arc::clone(&self.file), Arc::clone(&self.tainted),
                    compress_configs,
                ))?.into()
            }
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
    // table (n_tiles rows × column_width bytes) zero-filled, and
    // leaves the heap empty (PCOUNT=0) until CompressedImageHDU.write
    // is called.
    //
    // Integer ZBITPIX: BINTABLE has one column (COMPRESSED_DATA).
    // Float ZBITPIX with quantize=Some: BINTABLE has at least three
    // columns (COMPRESSED_DATA + ZSCALE + ZZERO), plus an optional
    // GZIP_COMPRESSED_DATA fallback for tiles that can't be
    // quantized.  ZQUANTIZ + ZDITHER0 header cards are emitted.
    #[allow(clippy::too_many_arguments)]
    fn create_compressed_image_hdu_impl(
        &mut self,
        py: Python<'_>,
        dtype: String,
        dims: Vec<i64>,
        extname: Option<String>,
        extver: Option<i64>,
        compress: Py<PyAny>,
        quantize: Option<Py<PyAny>>,
        blank: Option<i64>,
    ) -> PyResult<()> {
        if dims.is_empty() {
            return Err(PyValueError::new_err(
                "compressed images must have NAXIS >= 1"
            ));
        }
        // Axis 0 (numpy slowest, = FITS NAXIS-last) may be 0 to
        // permit "create empty + extend later" workflows.  Every
        // other axis must be strictly positive — the FITS standard
        // forbids zero-pixel inner axes.  HCOMPRESS_1 imposes its
        // own dim >= 4 check further below; this looser check just
        // controls the create-time empty case.
        for (i, &d) in dims.iter().enumerate() {
            let allow_zero = i == 0;
            if (allow_zero && d < 0) || (!allow_zero && d <= 0) {
                return Err(PyValueError::new_err(format!(
                    "dimension {} must be {} 0, got {}",
                    i,
                    if allow_zero { ">=" } else { ">" },
                    d
                )));
            }
        }

        // Extract the compress config.  Try each supported algorithm
        // class in turn; the resulting wrapper carries the algorithm
        // identity (for ZCMPTYPE) and the shared tile_shape /
        // heap_format params used by both encoders.
        let bound = compress.bind(py);
        let cfg = CompressionConfigKind::from_pyany(bound)?;

        // Dtype validation:
        //   - integer types (u1/i2/i4/i8) go via the BITPIX-direct
        //     path; quantize= is rejected.
        //   - float types (f4/f8) optionally take a Quantize config.
        //     With a Quantize, tiles get lossy quantization to i32
        //     then compressed (4-column BINTABLE schema).  Without
        //     (quantize=None or omitted), tiles are stored as raw
        //     float bytes through GZIP_1/GZIP_2 losslessly
        //     (single-column schema).  Lossy compression is opt-in
        //     to avoid silently throwing away precision on
        //     scientific data.
        //   - unsigned-int trick types (i1/u2/u4/u8) are deferred.
        let (bitpix, bzero) = crate::hdu_image::dtype_to_bitpix(&dtype)?;
        let is_float = bitpix < 0;
        let quantize_cfg: Option<crate::zimage::compression_config::Quantize> =
            if is_float {
                match quantize {
                    Some(qpy) => Some(qpy.extract::<
                        crate::zimage::compression_config::Quantize
                    >(py)?),
                    None => None, // unquantized lossless float path
                }
            } else {
                if quantize.is_some() {
                    return Err(PyValueError::new_err(
                        "quantize= is only valid for floating-point \
                         dtypes (f4/f8); for integer images, omit \
                         quantize=",
                    ));
                }
                None
            };
        let is_unquantized_float = is_float && quantize_cfg.is_none();
        // PLIO_1: integer-only mask data — float quantization
        // produces an i32 stream with negative values (bzero shifts
        // the range), which PLIO's non-negative-only encoder
        // refuses.  Reject upfront so the user gets a clear error
        // instead of a downstream "pixel is negative" failure.
        // Also reject i8 (bitpix=64) — PLIO has no 64-bit variant.
        // Unsigned-int trick dtypes (u2/u4/u8) reverse-transform to
        // negative signed stored values too, so we reject those for
        // PLIO as well; the user wanting PLIO + mask data should
        // pass plain u1/i2/i4 with no BZERO instead.
        if matches!(cfg, CompressionConfigKind::Plio1(_)) {
            if is_float {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "PLIO_1 does not support float dtypes: PLIO is \
                     designed for mask images with non-negative \
                     integer values.  For float data, use Gzip2 or \
                     Rice1 with a quantize= argument."
                ));
            }
            if bitpix == 64 {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "PLIO_1 does not support 64-bit pixels (i8 dtype): \
                     PLIO is designed for mask images with small \
                     non-negative integer values; use i2 or i4 instead."
                ));
            }
            if bzero.is_some() {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "PLIO_1 does not support unsigned-int trick dtypes \
                     (i1/u2/u4/u8): the FITS unsigned-int convention \
                     reverse-transforms unsigned input into signed \
                     stored values, which include negatives that PLIO's \
                     non-negative-only encoder rejects.  Use u1 or \
                     plain signed integer dtypes (i2/i4) instead."
                ));
            }
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
        // Unquantized floats require GZIP_1 or GZIP_2: only the
        // byte-stream codecs round-trip raw float bytes bit-exact.
        // RICE_1 / HCOMPRESS_1 are integer-only algorithms; astropy
        // accepts the combination but the round-trip silently
        // corrupts the data (the algorithms reinterpret float bit
        // patterns as integers).  This check fires AFTER the
        // algorithm-specific rejections above so that PLIO + float
        // (etc.) gets its more-specific error message — those errors
        // hold regardless of quantize.
        if is_unquantized_float
            && !matches!(
                cfg,
                CompressionConfigKind::Gzip1(_)
                    | CompressionConfigKind::Gzip2(_),
            )
        {
            return Err(PyValueError::new_err(format!(
                "unquantized float compression (quantize=None) \
                 requires compress=Gzip1(...) or compress=Gzip2(...). \
                 Got compress={}.  Other algorithms only round-trip \
                 quantized integer streams; using them on raw float \
                 bytes silently corrupts the data.  Either pass \
                 quantize=Quantize(...) to enable lossy quantization, \
                 or switch to compress=Gzip2(...) for lossless \
                 float compression (typically 3-5% better than Gzip1 \
                 on float data thanks to byte-shuffling).",
                cfg.zcmptype(),
            )));
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
                    let add = remain.div_ceil(ndiv);
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
        // Heap inner type: GZIP/RICE/HCOMPRESS write raw bytes
        // (TFORM1='1PB'/'1QB'); PLIO writes i16 BE shorts
        // (TFORM1='1PI'/'1QI').  The read-side dispatch already
        // handles both via tform_vla_inner_byte_width.
        let primary_tform = match (heap_format, &cfg) {
            ('P', CompressionConfigKind::Plio1(_)) => "1PI",
            ('Q', CompressionConfigKind::Plio1(_)) => "1QI",
            ('P', _) => "1PB",
            ('Q', _) => "1QB",
            _ => unreachable!("heap_format validated to P or Q"),
        };

        // Column layout depends on whether quantization is in play:
        //
        //   Integer ZBITPIX or unquantized float (quantize=None):
        //     col 1: COMPRESSED_DATA  (1PB / 1QB / 1PI / 1QI)
        //
        //   Quantized float (quantize=Quantize(...)):
        //     col 1: COMPRESSED_DATA       (primary tform, quantized i32 stream)
        //     col 2: ZSCALE                (1D, per-tile bscale)
        //     col 3: ZZERO                 (1D, per-tile bzero)
        //     col 4: GZIP_COMPRESSED_DATA  (1PB / 1QB, lossless float fallback)
        //
        // For the quantized-float layout the GZIP fallback column is
        // always present at column 4 so any tile that can't be
        // quantized can store raw float bytes losslessly.  Empty
        // descriptor (nelements=0) on tiles that DID quantize
        // cleanly.  Unquantized floats use the same single-column
        // layout as integers — astropy's quantize_level=0 schema.
        let is_quantized = quantize_cfg.is_some();
        let gzip_fallback_tform = if heap_format == 'P' { "1PB" } else { "1QB" };
        let n_columns: u64 = if is_quantized { 4 } else { 1 };
        // Row width in bytes: VLA columns contribute `descriptor_size`
        // bytes each; fixed-width ZSCALE/ZZERO contribute 8 bytes each.
        let naxis1: u64 = if is_quantized {
            descriptor_size + 8 + 8 + descriptor_size
        } else {
            descriptor_size
        };

        // FITS-order copies of image + tile shapes for the Z* cards.
        let fits_dims: Vec<u64> = numpy_dims.iter().rev().copied().collect();
        let tile_shape_fits: Vec<u64> =
            tile_shape_numpy.iter().rev().copied().collect();

        let mut cards: Vec<String> = vec![card_string(
            "XTENSION", "BINTABLE", "binary table extension",
        )];
        cards.push(card_int("BITPIX", 8, "8-bit bytes"));
        cards.push(card_int("NAXIS", 2, "2-dimensional binary table"));
        cards.push(card_int("NAXIS1", naxis1 as i64,
                            "width of table row in bytes"));
        cards.push(card_int("NAXIS2", n_tiles as i64,
                            "number of rows in table (= n_tiles)"));
        cards.push(card_int("PCOUNT", 0, "size of heap in bytes"));
        cards.push(card_int("GCOUNT", 1, "one data group"));
        cards.push(card_int("TFIELDS", n_columns as i64,
                            "number of fields per row"));
        cards.push(card_string("TFORM1", primary_tform,
                               "compressed data descriptor"));
        cards.push(card_string("TTYPE1", "COMPRESSED_DATA",
                               "label for column 1"));
        if is_quantized {
            cards.push(card_string("TFORM2", "1D",
                                   "per-tile linear-scale factor"));
            cards.push(card_string("TTYPE2", "ZSCALE",
                                   "label for column 2"));
            cards.push(card_string("TFORM3", "1D",
                                   "per-tile linear-scale zero point"));
            cards.push(card_string("TTYPE3", "ZZERO",
                                   "label for column 3"));
            cards.push(card_string("TFORM4", gzip_fallback_tform,
                                   "lossless GZIP fallback for unquantizable tiles"));
            cards.push(card_string("TTYPE4", "GZIP_COMPRESSED_DATA",
                                   "label for column 4"));
        }
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
        // Float-image quantization parameters (Phase 8).  ZQUANTIZ
        // names the dither method; ZDITHER0 is the per-file random
        // seed.  When the user passed Quantize(seed=0) we use 1
        // here as the on-disk default (cfitsio also defaults to 1
        // when the user hasn't picked a seed).
        //
        // ZBLANK records the integer sentinel cfitsio's quantized-
        // float decoder treats as "this pixel was NaN".  Always
        // -2147483647 (NULL_VALUE_I32 in our quantize module).
        // Emitting it lets fitsio / astropy readers recognize our
        // NaN-encoded pixels on the way back through their own
        // dequantize path.
        if let Some(q) = &quantize_cfg {
            cards.push(card_string(
                "ZQUANTIZ", q.method.zquantiz(),
                "dithering method",
            ));
            let seed_on_disk = if q.seed > 0 { q.seed } else { 1 };
            cards.push(card_int(
                "ZDITHER0", seed_on_disk,
                "dithering offset/seed",
            ));
            cards.push(card_int(
                "ZBLANK", -2147483647,
                "quantized null-pixel sentinel",
            ));
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
        // Unsigned-int trick: same pattern as the uncompressed image
        // path above.  User-facing dtype was u2/u4/u8/i1 but the
        // on-disk ZBITPIX is the opposite signedness.  Emit BSCALE/
        // BZERO as regular (NOT Z-prefixed) cards so readers
        // (rustfits + astropy + cfitsio) recover the original dtype
        // when they decompress and apply scaling.
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
        // ZBLANK sentinel for "missing" integer pixels.  Same physical
        // → stored transform as the uncompressed path.  Per the FITS
        // Tile Compression Convention, ZBLANK replaces BLANK for
        // compressed integer images; the stored value is in the
        // signed-BITPIX (i.e., post-XOR) integer space.  Reject on
        // quantized float — those HDUs already emit their own ZBLANK
        // (cfitsio NaN sentinel) and the user shouldn't override it.
        if let Some(b) = blank {
            if is_float {
                return Err(PyValueError::new_err(
                    "blank= is not valid on float dtypes (compressed). \
                     For quantized-float HDUs, NaN is automatically \
                     preserved via the i32 sentinel; for unquantized \
                     float HDUs (quantize=None), NaN is preserved \
                     directly in the lossless float bytes.  Omit \
                     blank=.",
                ));
            }
            let stored = physical_blank_to_stored(b, bitpix, bzero)?;
            cards.push(card_int(
                "ZBLANK", stored,
                "integer sentinel for blank pixels"));
        }
        cards.push(pad_to_card("END"));

        // Main data section: one descriptor per tile, all zeroes
        // (nelements=0, offset=0) until CompressedImageHDU.write
        // populates them.  Heap is empty (PCOUNT=0).
        // Main data section = row width × n_tiles, zero-filled until
        // the .write() call populates descriptors + (for floats)
        // ZSCALE / ZZERO columns and the heap.
        let data_size = naxis1.saturating_mul(n_tiles);
        let data_padded = data_section_padded(data_size);

        let offsets =
            append_header_and_data_to_file(&self.file, &cards, data_padded)?;
        // Store the cfg with the RESOLVED tile_shape (the actual
        // on-disk value) so .compression.tile_shape returns the
        // real shape even when the user passed `Gzip1()` etc.
        // without specifying tile_shape.  Same for the other
        // algorithm configs.
        let stored_cfg =
            cfg.with_resolved_tile_shape(tile_shape_numpy.clone());
        self.finalize_hdu(
            py, &cards, offsets,
            HduKind::CompressedImage {
                quantize: quantize_cfg,
                compress_config: Some(stored_cfg),
            },
        )
    }

    // Compressed-table create path.  Called by create_table_hdu when
    // `compress=` is non-None.  Builds the ZTABLE-shaped header
    // (ZTABLE=T, ZTILELEN, ZNAXIS1/2/PCOUNT, ZFORMn, ZCTYPn) from the
    // already-built uncompressed cards, replaces the on-disk
    // structural keys (NAXIS1/2, TFORMn = '1QB') with their
    // compressed-shell values, and reserves space for the descriptor
    // table.  The heap is grown on demand by hdu.write() — at create
    // time PCOUNT=0 and the data section is just the descriptor table.
    //
    // Phase 5 scope: fixed columns only.  VLA + compress is Phase 6
    // and rejected upstream in create_table_hdu.
    #[allow(clippy::too_many_arguments)]
    fn create_compressed_table_hdu_impl(
        &mut self,
        py: Python<'_>,
        table_cards: Vec<String>,
        row_width: u64,
        nrows: i64,
        columns: &[crate::hdu_table::Column],
        per_col_cfgs: Vec<CompressionConfigKind>,
        ztilelen: Option<i64>,
    ) -> PyResult<()> {
        // Validate ztilelen if user-provided; otherwise pick the
        // cfitsio default (~10 MB worth of rows).
        let ztilelen_u: usize = match ztilelen {
            Some(v) if v <= 0 => return Err(PyValueError::new_err(format!(
                "ztilelen must be > 0, got {}", v))),
            Some(v) => (v as usize).min(nrows.max(1) as usize),
            None => default_ztilelen(nrows as usize, row_width as usize),
        };

        // Translate per-column configs to algorithm enums (header
        // builder + write path both want this lighter form).  The
        // full configs are stored on the HDU for in-session round trip
        // of write-only params like Gzip1(level=).
        let algorithms: Vec<crate::zimage::CompressionAlgorithm> =
            per_col_cfgs.iter().map(|cfg| match cfg {
                CompressionConfigKind::Gzip1(_) =>
                    crate::zimage::CompressionAlgorithm::Gzip1,
                CompressionConfigKind::Gzip2(_) =>
                    crate::zimage::CompressionAlgorithm::Gzip2,
                CompressionConfigKind::Rice1(_) =>
                    crate::zimage::CompressionAlgorithm::Rice1,
                CompressionConfigKind::Hcompress1(_) =>
                    crate::zimage::CompressionAlgorithm::Hcompress1,
                CompressionConfigKind::Plio1(_) =>
                    crate::zimage::CompressionAlgorithm::Plio1,
            }).collect();

        let (cards, _n_tiles, data_size) = build_compressed_table_header(
            &table_cards, row_width, nrows, ztilelen_u, &algorithms,
            columns,
        )?;
        let data_padded = data_section_padded(data_size);

        // BINTABLE cannot be primary — write an empty primary image
        // first if the file has no HDUs yet (same as create_table_hdu).
        self.ensure_primary(py)?;

        let offsets = append_header_and_data_to_file(
            &self.file, &cards, data_padded)?;
        self.finalize_hdu(
            py, &cards, offsets,
            HduKind::CompressedTable {
                compress_configs: Some(per_col_cfgs),
            },
        )
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

    /// List of HDU objects in file order.
    ///
    /// Equivalent to iterating the :class:`FITS` instance, but
    /// returns a real Python list (e.g. for slicing / length
    /// queries without consuming the iterator).
    #[getter]
    fn hdus(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::new(py, &self.hdus)?;
        Ok(list.unbind())
    }

    /// Path passed to the constructor.
    #[getter]
    fn filename(&self) -> String {
        self.filename.clone()
    }

    /// Close the file handle and sync pending writes to disk.
    ///
    /// Called automatically when the :class:`FITS` is used as a
    /// context manager (``with rustfits.FITS(...) as fits:``).
    /// Safe to call multiple times.  After closing, attempting
    /// any read or write through the FITS object or its HDUs
    /// raises ``IOError``.
    fn close(&mut self) -> PyResult<()> {
        let mut guard = lock_file(&self.file)?;
        if let Some(file) = guard.take() {
            let _ = file.sync_all();
        }
        Ok(())
    }

    /// ``True`` once :meth:`close` (or context-manager ``__exit__``)
    /// has been called; ``False`` while the file is open.
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

    /// Create a new image HDU and append it to the file.
    ///
    /// Allocates the data section as zeros via sparse-file extension
    /// — call :meth:`ImageHDU.write` (or use the returned HDU as
    /// ``fits[-1]``) to actually write pixel data.
    ///
    /// The first HDU created becomes the primary HDU (``SIMPLE=T``,
    /// ``EXTEND=T``); subsequent calls produce ``XTENSION='IMAGE'``
    /// extensions.
    ///
    /// Parameters
    /// ----------
    /// dtype : dtype-like
    ///     Anything :func:`numpy.dtype` accepts: a short-code
    ///     string (``'f8'``, ``'i4'``, ``'u2'``), a numpy scalar
    ///     type (``numpy.int32``, ``numpy.float64``), a Python
    ///     builtin (``int``, ``float``), or a
    ///     :class:`numpy.dtype` instance.  Normalized internally
    ///     via ``numpy.dtype(...)``.  Both the BITPIX-native
    ///     dtypes (``u1`` / ``i2`` / ``i4`` / ``i8`` / ``f4`` /
    ///     ``f8``) and the unsigned-int trick dtypes (``i1`` /
    ///     ``u2`` / ``u4`` / ``u8``) are accepted; the latter
    ///     emit the corresponding ``BSCALE`` + ``BZERO`` cards.
    /// dims : sequence of int
    ///     Image shape in numpy (row-major) axis order — slowest
    ///     axis first.  Reversed internally to produce FITS
    ///     ``NAXISn``.  Axis 0 (the slowest-varying axis) may be
    ///     ``0`` to create an empty HDU that a later
    ///     :meth:`ImageHDU.extend` fills incrementally (parallel
    ///     to ``create_table_hdu(nrows=0)`` + ``append``).  Every
    ///     other axis must be ``> 0`` — the FITS standard forbids
    ///     zero-pixel inner axes.  (HCOMPRESS_1 additionally
    ///     requires every axis ``>= 4``, so the empty-axis-0 form
    ///     is unavailable under ``compress=Hcompress1(...)``.)
    /// extname : str, optional
    ///     ``EXTNAME`` to assign.  Defaults to no EXTNAME card.
    /// extver : int, optional
    ///     ``EXTVER`` to assign.  Defaults to no EXTVER card.
    /// compress : Gzip1 / Gzip2 / Rice1 / Hcompress1 / Plio1, optional
    ///     If set, create a tile-compressed image
    ///     (``BINTABLE`` + ``ZIMAGE`` on disk, returned in Python
    ///     as a :class:`CompressedImageHDU`) instead of a plain
    ///     ``IMAGE`` extension.  All five algorithms are
    ///     supported for integer dtypes; only GZIP_1 / GZIP_2
    ///     for unquantized floats; all-but-PLIO for quantized
    ///     floats.
    /// quantize : Quantize, optional
    ///     Per-tile quantization config for float-image
    ///     compression: ``rustfits.Quantize(level=...,
    ///     method='dither1', seed=0)``.  Required when
    ///     ``compress=`` is set and the dtype is ``f4``/``f8``
    ///     and you want lossy storage.  Ignored for integer
    ///     dtypes.  Omit on float input to write losslessly
    ///     (raw float bytes through GZIP).
    /// blank : int, optional
    ///     Sentinel value for masked pixels (emits ``BLANK`` for
    ///     uncompressed, ``ZBLANK`` for compressed integer HDUs).
    ///     Only valid for integer dtypes; float HDUs use NaN.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Unsupported dtype, a non-positive inner dimension (axis
    ///     0 may be ``0`` but must not be negative), ``quantize=``
    ///     without ``compress=``, ``blank=`` on a float dtype, or
    ///     other algorithm/dtype incompatibilities (see
    ///     :class:`Rice1` / :class:`Hcompress1` / :class:`Plio1`
    ///     for per-algorithm constraints).
    ///
    /// See Also
    /// --------
    /// create_table_hdu : The table-side counterpart.
    /// Gzip1, Gzip2, Rice1, Hcompress1, Plio1 : Compression
    ///     config classes for the ``compress=`` argument.
    /// Quantize : Per-tile quantization config for float-image
    ///     compression.
    #[pyo3(signature = (
        dtype, dims, *, extname=None, extver=None,
        compress=None, quantize=None, blank=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create_image_hdu(
        &mut self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        dims: Vec<i64>,
        extname: Option<String>,
        extver: Option<i64>,
        compress: Option<Py<PyAny>>,
        quantize: Option<Py<PyAny>>,
        blank: Option<i64>,
    ) -> PyResult<()> {
        // Normalize the dtype-like input through `numpy.dtype(...)`
        // so callers can pass any of: a short-code string ('f8'),
        // numpy scalar type (np.int32), Python builtin (float), or
        // an existing np.dtype object.  Use .str (e.g. '<i4') which
        // dtype_to_bitpix already handles (it trims byte-order
        // prefixes and lowercases).
        let np = py.import("numpy")?;
        let dtype_str: String = np
            .call_method1("dtype", (dtype,))?
            .getattr("str")?
            .extract()?;
        if let Some(cfg) = compress {
            return self.create_compressed_image_hdu_impl(
                py, dtype_str, dims, extname, extver, cfg, quantize, blank,
            );
        }
        if quantize.is_some() {
            return Err(PyValueError::new_err(
                "quantize= is only valid with compress= (it controls \
                 the per-tile quantization for tile-compressed float \
                 images); for uncompressed images, omit quantize=",
            ));
        }
        // Axis 0 (numpy slowest, = FITS NAXIS-last) may be 0 to
        // permit "create empty + extend later" workflows.  Every
        // other axis must be strictly positive — the FITS standard
        // forbids zero-pixel inner axes.
        for (i, &d) in dims.iter().enumerate() {
            let allow_zero = i == 0;
            if (allow_zero && d < 0) || (!allow_zero && d <= 0) {
                return Err(PyValueError::new_err(format!(
                    "dimension {} must be {} 0, got {}",
                    i,
                    if allow_zero { ">=" } else { ">" },
                    d
                )));
            }
        }

        let (bitpix, bzero) = dtype_to_bitpix(&dtype_str)?;
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
        // BLANK sentinel for "missing" integer pixels.  User passes
        // the value in PHYSICAL space (their dtype).  Transform to
        // STORED space (the on-disk BITPIX) before emitting the
        // card — for unsigned-trick dtypes this means subtracting
        // BZERO so the card reflects the on-disk i2/i4 value.
        if let Some(b) = blank {
            let stored = physical_blank_to_stored(b, bitpix, bzero)?;
            cards.push(card_int(
                "BLANK", stored,
                "integer sentinel for blank pixels"));
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

    /// Create a new BINTABLE extension HDU and append it to the file.
    ///
    /// Allocates the data section as zeros — call
    /// :meth:`TableHDU.write` (or use the returned HDU as
    /// ``fits[-1]``) to actually write row data.
    ///
    /// If the file has no HDUs yet, an empty primary image
    /// (``SIMPLE=T``, ``NAXIS=0``) is written first so the
    /// BINTABLE can land as an extension — the FITS standard
    /// forbids BINTABLE as the primary HDU.
    ///
    /// Parameters
    /// ----------
    /// dtype : numpy.dtype or list of tuples
    ///     Structured dtype describing the table schema, or any
    ///     form ``numpy.dtype()`` accepts (e.g. a "descr" list
    ///     like ``[('x', 'f4'), ('y', 'f4'), ('name', 'S10')]``).
    ///     For VLA columns, use Object dtype (``'O'``) for the
    ///     field and declare its inner type via ``var_dtypes=``.
    /// nrows : int, optional
    ///     Initial row count.  Default ``0``; subsequent
    ///     :meth:`TableHDU.write` requires the value to match
    ///     this exactly, while :meth:`TableHDU.append` adds
    ///     rows beyond it.
    /// extname : str, optional
    ///     ``EXTNAME`` to assign.
    /// extver : int, optional
    ///     ``EXTVER`` to assign.
    /// units : dict, optional
    ///     ``{column_name: unit_string}`` to populate ``TUNITn``
    ///     cards.  Unspecified columns get no TUNIT.
    /// var_dtypes : dict, optional
    ///     For VLA columns: ``{column_name: inner_dtype_str}``,
    ///     where ``inner_dtype_str`` is a numpy short-code for
    ///     the per-cell element type (``'f4'`` / ``'i4'`` / etc.)
    ///     or ``'S'`` for ASCII strings.  The column itself must
    ///     be declared as Object dtype (``'O'``) in ``dtype``.
    /// bit_columns : list of str or True, optional
    ///     Opt-in to bit-packed ``X`` storage for bool columns:
    ///     a list of column names (case-insensitive) restricts the
    ///     opt-in to those columns; ``True`` is a soft global
    ///     toggle for ALL ``b1`` columns.  Default is one byte
    ///     per bool (``L``).
    /// heap_format : {'P', 'Q'}, optional
    ///     Descriptor format for VLA columns.  ``'P'`` (default)
    ///     uses 8-byte descriptors with a 4 GB heap ceiling;
    ///     ``'Q'`` uses 16-byte descriptors with no practical
    ///     ceiling.  Ignored when no VLA columns are declared.
    /// compress : str, bool, or per-algorithm config / dict, optional
    ///     Create a tile-compressed table (``ZTABLE`` on disk,
    ///     :class:`CompressedTableHDU` in Python) instead of a
    ///     plain BINTABLE.  Accepts:
    ///
    ///     * ``True`` — compress every column with cfitsio's
    ///       per-dtype defaults.
    ///     * a string alias (``'GZIP_1'`` / ``'GZIP_2'`` /
    ///       ``'RICE_1'``) or config-class instance
    ///       (``Gzip1()`` / ``Gzip2()`` / ``Rice1()``) — apply
    ///       to every column.
    ///     * a dict ``{column_name: algo}`` — per-column
    ///       override; unspecified columns get the default.
    /// ztilelen : int, optional
    ///     Rows per tile for table compression.  Defaults to
    ///     cfitsio's ``max(1, min(nrows, 10_000_000 /
    ///     row_width))``.  Requires ``compress=``; rejected
    ///     otherwise.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Negative ``nrows``; unsupported per-column dtype;
    ///     ``var_dtypes=`` declared for a non-Object field;
    ///     ``ztilelen=`` set without ``compress=``; invalid
    ///     ``compress=`` form; or an algorithm not legal for
    ///     a column's dtype (e.g. ``RICE_1`` on float).
    ///
    /// See Also
    /// --------
    /// create_image_hdu : The image-side counterpart.
    /// TableHDU.write : Write data into the created table.
    /// TableHDU.append : Add rows beyond ``nrows``.
    #[pyo3(signature = (
        dtype, nrows=0, *,
        extname=None, extver=None, units=None,
        var_dtypes=None, bit_columns=None, heap_format=None,
        compress=None, ztilelen=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create_table_hdu(
        &mut self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        nrows: i64,
        extname: Option<String>,
        extver: Option<i64>,
        units: Option<&Bound<'_, PyDict>>,
        var_dtypes: Option<&Bound<'_, PyDict>>,
        bit_columns: Option<&Bound<'_, PyAny>>,
        heap_format: Option<String>,
        compress: Option<&Bound<'_, PyAny>>,
        ztilelen: Option<i64>,
    ) -> PyResult<()> {
        if nrows < 0 {
            return Err(PyValueError::new_err(format!(
                "create_table_hdu: nrows must be >= 0, got {}", nrows)));
        }
        // ztilelen is meaningful only with compress=; reject early
        // if the user set it without compress=.
        if compress.is_none() && ztilelen.is_some() {
            return Err(PyValueError::new_err(
                "create_table_hdu: ztilelen= requires compress="));
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
        let bit_columns_spec = parse_bit_columns_arg(bit_columns)?;
        let (table_cards, row_width) = normalize_and_build_table_header(
            py, dtype, nrows, extname.as_deref(), extver, units,
            var_dtypes, bit_columns_spec.as_ref(), desc_char,
        )?;

        // Dispatch on compress=.  None/False -> uncompressed (the
        // existing path).  Anything else routes to the ZTABLE create
        // impl, which builds a fresh ZTABLE-shaped header from the
        // uncompressed cards and reserves descriptor space for write().
        let columns = parse_columns(&table_cards)?;
        let resolved_compress = resolve_compress_arg(
            py, compress, &columns,
        )?;
        if let Some(per_col_cfgs) = resolved_compress {
            return self.create_compressed_table_hdu_impl(
                py, table_cards, row_width, nrows, &columns,
                per_col_cfgs, ztilelen,
            );
        }

        let data_size = (nrows as u64).saturating_mul(row_width);
        let data_padded = data_section_padded(data_size);

        // BINTABLE cannot be primary — write an empty primary image
        // first if the file has no HDUs yet.
        self.ensure_primary(py)?;

        let offsets = append_header_and_data_to_file(
            &self.file, &table_cards, data_padded)?;
        self.finalize_hdu(py, &table_cards, offsets, HduKind::Table)
    }

    /// Create an image HDU from ``data`` and write the pixels.
    ///
    /// One-call convenience that combines
    /// :meth:`create_image_hdu` and :meth:`ImageHDU.write`.  The
    /// HDU's dtype and shape are taken from ``data``; everything
    /// else is forwarded to :meth:`create_image_hdu`.
    ///
    /// Parameters
    /// ----------
    /// data : array_like
    ///     The pixel data to write.  Anything ``numpy.asanyarray``
    ///     accepts: an ndarray, a MaskedArray (mask is preserved
    ///     through the write — see :class:`ImageHDU.write`), or a
    ///     nested Python sequence.  Must have a supported numpy
    ///     dtype (``u1`` / ``i1`` / ``i2`` / ``u2`` / ``i4`` /
    ///     ``u4`` / ``i8`` / ``u8`` / ``f4`` / ``f8``).
    /// extname, extver, compress, quantize, blank
    ///     Forwarded to :meth:`create_image_hdu`; see that method
    ///     for the full kwarg semantics.
    /// header : FITSHeader or dict, optional
    ///     Cards to copy into the new HDU after the write.  Routed
    ///     through :meth:`FITSHeader.update`, which silently skips
    ///     protected/structural cards when copying from another
    ///     FITSHeader and raises on protected cards in a dict.
    ///
    /// Returns
    /// -------
    /// hdu : ImageHDU or CompressedImageHDU
    ///     The newly created HDU, ready for further reads/writes
    ///     while the FITS handle is open.
    ///
    /// See Also
    /// --------
    /// create_image_hdu : Schema-only create (no data write).
    /// write_table : The table-side counterpart.
    #[pyo3(signature = (
        data, *, extname=None, extver=None,
        compress=None, quantize=None, blank=None, header=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn write_image(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        extname: Option<String>,
        extver: Option<i64>,
        compress: Option<Py<PyAny>>,
        quantize: Option<Py<PyAny>>,
        blank: Option<i64>,
        header: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // asanyarray preserves MaskedArray (so write() can unwrap
        // the mask) but coerces plain lists/tuples to ndarrays.
        let np = py.import("numpy")?;
        let arr = np.call_method1("asanyarray", (data,))?;
        let dtype = arr.getattr("dtype")?;
        let shape: Vec<i64> = arr.getattr("shape")?.extract()?;

        self.create_image_hdu(
            py, &dtype, shape, extname, extver,
            compress, quantize, blank,
        )?;

        let hdu = self.hdus.last()
            .ok_or_else(|| PyIOError::new_err(
                "write_image: create_image_hdu did not append an HDU"))?
            .clone_ref(py);
        let bound = hdu.bind(py);
        bound.call_method1("write", (&arr,))?;
        if let Some(hdr) = header {
            bound.getattr("header")?.call_method1("update", (hdr,))?;
        }
        Ok(hdu)
    }

    /// Create a BINTABLE HDU from ``data`` and write the rows.
    ///
    /// One-call convenience that combines
    /// :meth:`create_table_hdu` and :meth:`TableHDU.write`.  The
    /// table schema (dtype + nrows) is derived from ``data``;
    /// everything else is forwarded to :meth:`create_table_hdu`.
    ///
    /// If the file has no HDUs yet, an empty primary image is
    /// written first (the FITS standard forbids BINTABLE as the
    /// primary HDU).
    ///
    /// Parameters
    /// ----------
    /// data : structured ndarray, dict, or list/tuple of arrays
    ///     The rows to write.  Three accepted shapes:
    ///
    ///     * **structured ndarray** — dtype is taken from
    ///       ``data.dtype``, nrows from ``len(data)``.
    ///     * **dict** ``{name: array}`` — dtype is composed
    ///       field-by-field from each column's ndarray dtype;
    ///       all arrays must have the same length.
    ///     * **list / tuple of arrays + names=** — same as dict,
    ///       with names supplied as a separate argument.
    /// names : sequence of str, optional
    ///     Column names for the list/tuple input form.  Required
    ///     when ``data`` is a list or tuple; rejected for the
    ///     other two forms (the names are already implied).
    /// extname, extver, units, var_dtypes, bit_columns, heap_format, compress, ztilelen
    ///     Forwarded to :meth:`create_table_hdu`; see that method
    ///     for the full kwarg semantics.
    /// header : FITSHeader or dict, optional
    ///     Cards to copy into the new HDU after the write.  Same
    ///     semantics as ``write_image``'s ``header=``.
    ///
    /// Returns
    /// -------
    /// hdu : TableHDU or CompressedTableHDU
    ///     The newly created HDU, ready for further reads/writes
    ///     while the FITS handle is open.
    ///
    /// See Also
    /// --------
    /// create_table_hdu : Schema-only create (no data write).
    /// write_image : The image-side counterpart.
    #[pyo3(signature = (
        data, *, names=None,
        extname=None, extver=None, units=None,
        var_dtypes=None, bit_columns=None, heap_format=None,
        compress=None, ztilelen=None, header=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn write_table(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        names: Option<&Bound<'_, PyAny>>,
        extname: Option<String>,
        extver: Option<i64>,
        units: Option<&Bound<'_, PyDict>>,
        var_dtypes: Option<&Bound<'_, PyDict>>,
        bit_columns: Option<&Bound<'_, PyAny>>,
        heap_format: Option<String>,
        compress: Option<&Bound<'_, PyAny>>,
        ztilelen: Option<i64>,
        header: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let (dtype, nrows) = derive_table_schema_from_data(py, data, names)?;
        self.create_table_hdu(
            py, &dtype, nrows, extname, extver, units,
            var_dtypes, bit_columns, heap_format, compress, ztilelen,
        )?;

        let hdu = self.hdus.last()
            .ok_or_else(|| PyIOError::new_err(
                "write_table: create_table_hdu did not append an HDU"))?
            .clone_ref(py);
        let bound = hdu.bind(py);
        // hdu.write(data, names=names) — forward the original input
        // (TableHDU.write already dispatches on structured/dict/
        // list+names).
        let kwargs = PyDict::new(py);
        if let Some(n) = names {
            kwargs.set_item("names", n)?;
        }
        bound.call_method("write", (data,), Some(&kwargs))?;
        if let Some(hdr) = header {
            bound.getattr("header")?.call_method1("update", (hdr,))?;
        }
        Ok(hdu)
    }

    /// Write ``data`` to a new HDU, auto-detecting image vs table.
    ///
    /// Minimal-tier counterpart to :meth:`write_image` and
    /// :meth:`write_table` — accepts only the universal kwargs
    /// (``extname``, ``header``) and dispatches on the data type:
    ///
    ///   * plain (non-structured) :class:`numpy.ndarray` →
    ///     :meth:`write_image`
    ///   * structured :class:`numpy.ndarray`
    ///     (``dtype.fields is not None``) → :meth:`write_table`
    ///   * ``{name: array}`` dict → :meth:`write_table`
    ///   * list-of-arrays + ``names=`` → rejected (call
    ///     :meth:`write_table` directly)
    ///   * anything else → :class:`ValueError`
    ///
    /// Convenient for copying HDUs between files without caring
    /// about their type::
    ///
    ///     with rustfits.FITS(infile) as src:
    ///         with rustfits.FITS(outfile, "w+") as dst:
    ///             for hdu in src:
    ///                 if hdu.has_data:
    ///                     dst.write(hdu.read())
    ///
    /// For knobs like ``compress=``, ``quantize=``, ``blank=``,
    /// ``var_dtypes=``, ``units=``, etc., use the type-specific
    /// :meth:`write_image` / :meth:`write_table` directly.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray or dict
    ///     Image: a plain ndarray.  Table: a structured ndarray or
    ///     a ``{name: ndarray}`` dict.
    /// extname : str, optional
    ///     EXTNAME to set on the new HDU.
    /// header : FITSHeader or dict, optional
    ///     Cards to copy into the new HDU after the write.
    ///
    /// Returns
    /// -------
    /// hdu : ImageHDU, TableHDU, or compressed variant
    ///     The newly created HDU, ready for further reads/writes
    ///     while the FITS handle is open.
    ///
    /// See Also
    /// --------
    /// write_image : Image-specific create+write with all knobs.
    /// write_table : Table-specific create+write with all knobs.
    #[pyo3(signature = (data, *, extname=None, header=None))]
    fn write(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        extname: Option<String>,
        header: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // Dict → table.  Checked first so a dict subclass with
        // a `dtype` attribute (unlikely but possible) still routes
        // to the table path.
        if data.is_instance_of::<PyDict>() {
            return self.write_table(
                py, data, None, extname, None, None,
                None, None, None, None, None, header,
            );
        }
        // numpy.ndarray → image or table depending on structured dtype.
        // Strict isinstance against numpy.ndarray rather than
        // asanyarray-coercion: a bare list could be either "image
        // pixels" or "column arrays for a table", and the ambiguity
        // is better resolved by sending the user to the explicit
        // write_table(names=...) form than by guessing.
        let np = py.import("numpy")?;
        let ndarray = np.getattr("ndarray")?;
        if data.is_instance(&ndarray)? {
            let dtype = data.getattr("dtype")?;
            let is_structured = !dtype.getattr("fields")?.is_none();
            if is_structured {
                return self.write_table(
                    py, data, None, extname, None, None,
                    None, None, None, None, None, header,
                );
            }
            return self.write_image(
                py, data, extname, None, None, None, None, header,
            );
        }
        Err(PyValueError::new_err(format!(
            "FITS.write() accepts a numpy ndarray (image or \
             structured) or a {{name: array}} dict (table); got {}.  \
             For lists of arrays with names=, or any of the \
             type-specific kwargs (compress=, blank=, var_dtypes=, \
             ...), use FITS.write_image() / FITS.write_table().",
            data.get_type().name()?,
        )))
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
