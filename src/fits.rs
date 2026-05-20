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
use crate::hdu_table::{normalize_and_build_table_header, TableHDU};
use crate::hdu_ascii_table::AsciiTableHDU;
use crate::header::{card_int, card_logical, card_string, pad_to_card};

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
    #[pyo3(signature = (dtype, dims, extname=None, extver=None))]
    fn create_image_hdu(
        &mut self,
        py: Python<'_>,
        dtype: String,
        dims: Vec<i64>,
        extname: Option<String>,
        extver: Option<i64>,
    ) -> PyResult<()> {
        for (i, &d) in dims.iter().enumerate() {
            if d <= 0 {
                return Err(PyValueError::new_err(format!(
                    "dimension {} must be > 0, got {}", i, d
                )));
            }
        }

        let bitpix = dtype_to_bitpix(&dtype)?;
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
        cards.push(pad_to_card("END"));

        let header_bytes_len = cards.len() * CARD_SIZE;
        let pad_n = (BLOCK_SIZE - header_bytes_len % BLOCK_SIZE) % BLOCK_SIZE;

        let bytes_per_pixel = (bitpix.abs() / 8) as u64;
        let mut product: u64 = 1;
        for &d in &fits_dims {
            product = product.saturating_mul(d as u64);
        }
        let data_size = if naxis == 0 { 0 } else { bytes_per_pixel * product };
        let data_padded = if data_size == 0 {
            0
        } else {
            ((data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64) * BLOCK_SIZE as u64
        };

        let (header_offset, data_offset) = {
            let mut guard = lock_file(&self.file)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;

            file.seek(SeekFrom::End(0))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            let header_start = file.stream_position()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            for c in &cards {
                file.write_all(c.as_bytes())
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
            if pad_n > 0 {
                let padding = vec![b' '; pad_n];
                file.write_all(&padding)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }

            let header_end = file.stream_position()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            if data_padded > 0 {
                let new_len = header_end + data_padded;
                file.set_len(new_len)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                file.seek(SeekFrom::Start(new_len))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }

            file.flush()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            (header_start, header_end)
        };

        // Construct the matching ImageHDU in memory and append.  The parser
        // stores cards with trailing whitespace trimmed; mirror that here so
        // the in-memory HDU is byte-equivalent to what a re-parse would yield.
        let stored_cards: Vec<String> = cards.iter()
            .map(|c| c.trim_end().to_string())
            .collect();

        let header_block_count = (data_offset - header_offset) / BLOCK_SIZE as u64;
        let hdu_offsets = HduOffsets::new(
            header_offset, header_block_count, data_offset,
        );
        {
            let mut layout_guard = self.layout.hdus.lock()
                .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
            layout_guard.push(Arc::clone(&hdu_offsets));
        }

        let new_hdu: Py<PyAny> = Py::new(
            py,
            ImageHDU::new(
                stored_cards,
                self.hdus.len(),
                self.filename.clone(),
                hdu_offsets,
                Arc::clone(&self.layout),
                Arc::clone(&self.file),
                Arc::clone(&self.tainted),
            ),
        )?.into();
        self.hdus.push(new_hdu);

        Ok(())
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
        dtype, nrows=0, *, extname=None, extver=None, units=None
    ))]
    fn create_table_hdu(
        &mut self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        nrows: i64,
        extname: Option<String>,
        extver: Option<i64>,
        units: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if nrows < 0 {
            return Err(PyValueError::new_err(format!(
                "create_table_hdu: nrows must be >= 0, got {}", nrows)));
        }
        let (table_cards, row_width) = normalize_and_build_table_header(
            py, dtype, nrows, extname.as_deref(), extver, units,
        )?;
        let table_header_bytes_len = table_cards.len() * CARD_SIZE;
        let table_pad =
            (BLOCK_SIZE - table_header_bytes_len % BLOCK_SIZE) % BLOCK_SIZE;
        let data_size = (nrows as u64).saturating_mul(row_width);
        let data_padded = if data_size == 0 {
            0
        } else {
            ((data_size + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64)
                * BLOCK_SIZE as u64
        };

        // Compose the empty primary header up front so all writes can
        // happen inside one file-lock acquisition.
        let primary_cards: Option<Vec<String>> = if self.hdus.is_empty() {
            Some(vec![
                card_logical("SIMPLE", true, "file conforms to FITS standard"),
                card_int("BITPIX", 8, "8-bit bytes"),
                card_int("NAXIS", 0, "number of data axes"),
                card_logical("EXTEND", true,
                             "FITS dataset may contain extensions"),
                pad_to_card("END"),
            ])
        } else {
            None
        };

        // Writes: primary (if any), then BINTABLE header (padded), then
        // zero-allocate the data section via set_len.  All in one lock.
        let (primary_loc, table_loc) = {
            let mut guard = lock_file(&self.file)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            file.seek(SeekFrom::End(0))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            let primary_loc = if let Some(ref p_cards) = primary_cards {
                let p_start = file.stream_position()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                for c in p_cards {
                    file.write_all(c.as_bytes())
                        .map_err(|e| PyIOError::new_err(e.to_string()))?;
                }
                let p_bytes_len = p_cards.len() * CARD_SIZE;
                let p_pad = (BLOCK_SIZE - p_bytes_len % BLOCK_SIZE) % BLOCK_SIZE;
                if p_pad > 0 {
                    file.write_all(&vec![b' '; p_pad])
                        .map_err(|e| PyIOError::new_err(e.to_string()))?;
                }
                let p_end = file.stream_position()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                let p_blocks = (p_end - p_start) / BLOCK_SIZE as u64;
                Some((p_start, p_blocks, p_end))
            } else {
                None
            };

            let t_start = file.stream_position()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            for c in &table_cards {
                file.write_all(c.as_bytes())
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
            if table_pad > 0 {
                file.write_all(&vec![b' '; table_pad])
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
            let t_header_end = file.stream_position()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            if data_padded > 0 {
                let new_len = t_header_end + data_padded;
                file.set_len(new_len)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
                file.seek(SeekFrom::Start(new_len))
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
            file.flush()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            let t_blocks = (t_header_end - t_start) / BLOCK_SIZE as u64;
            (primary_loc, (t_start, t_blocks, t_header_end))
        };

        // Trim trailing whitespace on stored cards to match what the
        // parser would yield on a re-read (canonical in-memory form).
        let trim = |cards: &[String]| -> Vec<String> {
            cards.iter().map(|c| c.trim_end().to_string()).collect()
        };

        if let (Some(p_cards), Some((p_start, p_blocks, p_end)))
            = (primary_cards.as_ref(), primary_loc)
        {
            let p_arc = HduOffsets::new(p_start, p_blocks, p_end);
            {
                let mut lg = self.layout.hdus.lock()
                    .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
                lg.push(Arc::clone(&p_arc));
            }
            let img: Py<PyAny> = Py::new(
                py,
                ImageHDU::new(
                    trim(p_cards),
                    self.hdus.len(),
                    self.filename.clone(),
                    p_arc,
                    Arc::clone(&self.layout),
                    Arc::clone(&self.file),
                    Arc::clone(&self.tainted),
                ),
            )?.into();
            self.hdus.push(img);
        }

        let (t_start, t_blocks, t_end) = table_loc;
        let t_arc = HduOffsets::new(t_start, t_blocks, t_end);
        {
            let mut lg = self.layout.hdus.lock()
                .map_err(|_| PyIOError::new_err("layout lock poisoned"))?;
            lg.push(Arc::clone(&t_arc));
        }
        let tab: Py<PyAny> = Py::new(
            py,
            TableHDU::new(
                trim(&table_cards),
                self.hdus.len(),
                self.filename.clone(),
                t_arc,
                Arc::clone(&self.layout),
                Arc::clone(&self.file),
                Arc::clone(&self.tainted),
            ),
        )?.into();
        self.hdus.push(tab);
        Ok(())
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
