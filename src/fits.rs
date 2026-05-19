// FITS pyclass — top-level handle for an open FITS file.  Owns the file
// handle, the HDU list, and the per-file taint flag.  Also home to the
// HDU-list parser (`parse_hdus_from_file`) and the header-shape validators
// it uses.

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::Bound;
use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicBool;

use crate::common::{
    lock_file, parse_keyword, FileHandle, FileLayout, HduOffsets, TaintFlag,
    BLOCK_SIZE, CARDS_PER_BLOCK, CARD_SIZE,
};
use crate::hdu::HDU;
use crate::hdu_image::{dtype_to_bitpix, ImageHDU};
use crate::hdu_table::TableHDU;
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

// Extract the EXTNAME string value from a card list, or None if there is no
// EXTNAME card.  Used by FITS::__getitem__ to support string-keyed HDU
// lookup; case-insensitive matching is the caller's responsibility.
fn extract_extname(cards: &[String]) -> Option<String> {
    for card in cards {
        if card.len() < 9 { continue; }
        if card[..8].trim() != "EXTNAME" { continue; }
        if !card[8..].starts_with('=') { continue; }
        let value_part = card[9..].trim_start();
        if !value_part.starts_with('\'') { return None; }
        let after_open = &value_part[1..];
        let bytes = after_open.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\'' {
                // `''` is the FITS escape for a single quote; skip both.
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                    continue;
                }
                let inner = &after_open[..i];
                return Some(inner.replace("''", "'").trim_end().to_string());
            }
            i += 1;
        }
        return None;
    }
    None
}

// Walks the file from byte 0, extracting every HDU header and skipping over
// each data section, returning the parsed HDU Python objects.  Each HDU is
// constructed with its data-section byte offset and a clone of the file
// handle so that write-back methods can locate themselves on disk.
fn parse_hdus_from_file(
    py: Python<'_>,
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
        let is_ascii_table = header_cards.iter().any(|c| c.starts_with("XTENSION= 'TABLE'"));

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
        let hdu_py: Py<PyAny> = if is_image {
            Py::new(py, ImageHDU::new(
                header_cards.clone(), hdus.len(),
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
            ))?.into()
        } else if is_binary_table {
            Py::new(py, TableHDU::new(
                header_cards.clone(), hdus.len(),
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
            ))?.into()
        } else if is_ascii_table {
            Py::new(py, AsciiTableHDU::new(
                header_cards.clone(), hdus.len(),
                hdu_offsets, hdu_layout, hdu_file, hdu_taint,
            ))?.into()
        } else {
            let h = HDU::new(
                header_cards.clone(), hdus.len(),
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
    #[new]
    fn new(py: Python<'_>, filename: String, mode: String) -> PyResult<Self> {
        let mut options = OpenOptions::new();

        match mode.as_str() {
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
        let hdus = parse_hdus_from_file(py, &handle, &layout, &tainted)?;

        Ok(FITS {
            filename,
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

    fn __repr__(&self) -> String {
        let status = match self.file.lock() {
            Ok(guard) if guard.is_none() => "closed",
            Ok(_) => "open",
            Err(_) => "poisoned",
        };
        format!("FITS('{}', {} HDUs, {})", self.filename, self.hdus.len(), status)
    }

    fn __len__(&self) -> usize {
        self.hdus.len()
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
                hdu_offsets,
                Arc::clone(&self.layout),
                Arc::clone(&self.file),
                Arc::clone(&self.tainted),
            ),
        )?.into();
        self.hdus.push(new_hdu);

        Ok(())
    }

    // Accept either an integer (positional, with Python-style negative
    // indexing) or a string (EXTNAME lookup, case-insensitive).  A bool is
    // rejected explicitly because Python's int/bool subclass relationship
    // would otherwise let `fits[True]` resolve as `fits[1]`.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if key.is_instance_of::<pyo3::types::PyBool>() {
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
        let name: Option<String> = if let Ok(s) = key.extract::<String>() {
            Some(s)
        } else if let Ok(b) = key.extract::<Vec<u8>>() {
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
                let matched = extract_extname(&cards_guard)
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
