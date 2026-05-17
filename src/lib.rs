use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyList};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::Bound;
use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom};

const BLOCK_SIZE: usize = 2880;
const CARD_SIZE: usize = 80;

// ====================== Helper: parse integer keyword (for data size) ======================
// Strict match: the keyword field in cols 1-8 (trimmed) must equal `key`, and
// col 9 must be `=`.  This avoids the trap that `starts_with("NAXIS")` would
// also match `NAXIS1`, `NAXIS2`, etc.
fn parse_keyword(cards: &[String], key: &str) -> Option<i64> {
    for card in cards {
        if card.len() < 9 { continue; }
        if card[..8].trim() != key { continue; }
        if !card[8..].starts_with('=') { continue; }
        let value_part = &card[9..];
        if let Some(num_str) = value_part.split_whitespace().next() {
            let cleaned = num_str.trim_end_matches(&['\'', ' ', '/'][..]);
            return cleaned.parse::<i64>().ok();
        }
    }
    None
}

// ====================== Helper: validate that a header has mandatory keywords ======================
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

// ====================== Calculate exact padded data section size ======================
// Implements the general FITS formula:
//
//     N_bytes = |BITPIX|/8 * GCOUNT * (PCOUNT + Π NAXISn)
//
// For images: GCOUNT defaults to 1, PCOUNT defaults to 0, reducing to
// bytes_per_pixel * Π NAXISn.  For binary tables, PCOUNT carries the size of
// the variable-length-array heap, which must be included so that the next
// HDU is located correctly.  An HDU with NAXIS=0 has no data unit regardless
// of PCOUNT/GCOUNT.
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

// ====================== Helper: split a card's value/comment portion ======================
// `value_part` is the substring after `=` for a regular card, or the substring after
// the `CONTINUE` keyword (cols 9..) for a continuation card.  Returns (trimmed raw
// value with quotes intact, trimmed comment).  A `/` that appears inside a quoted
// FITS string is not treated as a comment delimiter.  The single-quote toggle
// happens to also handle `''` escape correctly, because each escape contributes
// two toggles (net no change).
fn split_value_comment(value_part: &str) -> (String, String) {
    let mut comment_start = value_part.len();
    let mut in_string = false;
    for (i, ch) in value_part.char_indices() {
        if ch == '\'' {
            in_string = !in_string;
        } else if ch == '/' && !in_string {
            comment_start = i;
            break;
        }
    }
    let raw = value_part[..comment_start].trim().to_string();
    let comment = if comment_start < value_part.len() {
        value_part[comment_start + 1..].trim().to_string()
    } else {
        String::new()
    };
    (raw, comment)
}

// ====================== Helper: extract a quoted FITS string ======================
// `raw` is the trimmed value substring including outer single quotes (e.g.
// "'O''Brien   '").  Strips the outer quotes, converts `''` back to `'`, and
// drops trailing spaces (per the FITS standard, trailing spaces in a string
// value are not significant).  FITS values are restricted to printable ASCII,
// so byte-position slicing is safe.
fn extract_fits_string(raw: &str) -> String {
    let inner = &raw[1..raw.len() - 1];
    inner.replace("''", "'").trim_end().to_string()
}

// ====================== Helper: parse a FITS float ======================
// FITS permits Fortran-style `D` as the exponent indicator in addition to `E`.
fn parse_fits_float(s: &str) -> Option<f64> {
    if s.contains('D') || s.contains('d') {
        s.replace('D', "E").replace('d', "e").parse::<f64>().ok()
    } else {
        s.parse::<f64>().ok()
    }
}

// ====================== Helper: parse a FITS complex literal ======================
// Returns Some((real, imag)) if `raw` looks like "(real, imag)".  Whitespace
// is allowed around the comma and inside the parentheses.  Each component may
// use either E or D exponent notation.
fn parse_fits_complex(raw: &str) -> Option<(f64, f64)> {
    if !(raw.starts_with('(') && raw.ends_with(')') && raw.len() >= 2) {
        return None;
    }
    let inner = &raw[1..raw.len() - 1];
    let comma = inner.find(',')?;
    let r = parse_fits_float(inner[..comma].trim())?;
    let i = parse_fits_float(inner[comma + 1..].trim())?;
    Some((r, i))
}

// ====================== Parse FITS header cards into Python dict ======================
fn parse_header_dict(py: Python<'_>, cards: &[String]) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let comments_list = PyList::empty(py);
    let history_list = PyList::empty(py);
    let blank_list = PyList::empty(py);
    let mut has_comments = false;
    let mut has_history = false;
    let mut has_blank = false;

    let mut i = 0;
    while i < cards.len() {
        let card = cards[i].trim_end();
        if card.is_empty() {
            i += 1;
            continue;
        }

        // Inspect the keyword field (cols 1-8).
        let key_field: &str = if card.len() >= 8 { &card[..8] } else { card };
        let keyword_trimmed = key_field.trim();

        // Commentary / structural cards: dispatched by the keyword field.
        match keyword_trimmed {
            "END" => { i += 1; continue; }
            "COMMENT" => {
                let text = if card.len() > 8 { card[8..].to_string() } else { String::new() };
                comments_list.append(text)?;
                has_comments = true;
                i += 1;
                continue;
            }
            "HISTORY" => {
                let text = if card.len() > 8 { card[8..].to_string() } else { String::new() };
                history_list.append(text)?;
                has_history = true;
                i += 1;
                continue;
            }
            "" => {
                // Blank keyword field, content in cols 9-80: commentary card
                // with no associated keyword.
                let text = if card.len() > 8 { card[8..].to_string() } else { String::new() };
                blank_list.append(text)?;
                has_blank = true;
                i += 1;
                continue;
            }
            "CONTINUE" => { i += 1; continue; }  // orphan CONTINUE
            _ => {}
        }

        // Determine keyword and value substring.  For HIERARCH, the keyword is
        // everything between `HIERARCH ` and the first `=`.  For regular cards,
        // the keyword field is cols 1-8.
        let (keyword, value_part) = if keyword_trimmed == "HIERARCH" {
            let eq_pos = match card.find('=') {
                Some(p) => p,
                None => { i += 1; continue; }
            };
            let kw = card[8..eq_pos].trim().to_string();
            if kw.is_empty() { i += 1; continue; }
            (kw, &card[eq_pos + 1..])
        } else {
            let eq_pos = match card.find('=') {
                Some(p) => p,
                None => { i += 1; continue; }
            };
            (keyword_trimmed.to_string(), &card[eq_pos + 1..])
        };

        let (raw_value, mut comment) = split_value_comment(value_part);

        let py_value: Py<PyAny> = if raw_value.is_empty() {
            py.None()
        } else if raw_value.starts_with('\'') && raw_value.ends_with('\'') && raw_value.len() >= 2 {
            // String value, possibly continued via the long-string convention.
            let mut s = extract_fits_string(&raw_value);
            while s.ends_with('&') {
                if i + 1 >= cards.len() { break; }
                let next_card = cards[i + 1].trim_end();
                if !next_card.starts_with("CONTINUE") { break; }
                let rest = if next_card.len() > 8 { &next_card[8..] } else { "" };
                let (cont_raw, cont_comment) = split_value_comment(rest);
                if !(cont_raw.starts_with('\'')
                    && cont_raw.ends_with('\'')
                    && cont_raw.len() >= 2)
                {
                    break;
                }
                let segment = extract_fits_string(&cont_raw);
                s.pop(); // drop trailing `&`
                s.push_str(&segment);
                if !cont_comment.is_empty() {
                    if !comment.is_empty() { comment.push(' '); }
                    comment.push_str(&cont_comment);
                }
                i += 1;
            }
            s.into_py_any(py)?
        } else if raw_value == "T" {
            true.into_py_any(py)?
        } else if raw_value == "F" {
            false.into_py_any(py)?
        } else if let Some((r, im)) = parse_fits_complex(&raw_value) {
            PyComplex::from_doubles(py, r, im).into_any().unbind()
        } else if let Ok(n) = raw_value.parse::<i64>() {
            n.into_py_any(py)?
        } else if let Some(f) = parse_fits_float(&raw_value) {
            f.into_py_any(py)?
        } else {
            raw_value.into_py_any(py)?
        };

        let inner = PyDict::new(py);
        inner.set_item("value", py_value)?;
        inner.set_item("comment", comment)?;
        dict.set_item(keyword, inner)?;

        i += 1;
    }

    if has_comments {
        dict.set_item("COMMENT", comments_list)?;
    }
    if has_history {
        dict.set_item("HISTORY", history_list)?;
    }
    if has_blank {
        dict.set_item("", blank_list)?;
    }

    Ok(dict.unbind())
}

// ====================== Base HDU ======================
#[pyclass(subclass)]
struct HDU {
    header: Vec<String>,
    index: usize,
}

#[pymethods]
impl HDU {
    #[new]
    fn new(header: Vec<String>, index: usize) -> Self {
        HDU { header, index }
    }

    fn __repr__(&self) -> String {
        format!("<HDU #{}>", self.index)
    }

    #[getter]
    fn header(&self) -> Vec<String> {
        self.header.clone()
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter]
    fn header_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        parse_header_dict(py, &self.header)
    }
}

// ====================== Specialized HDU subclasses ======================
#[pyclass(extends = HDU)]
struct ImageHDU;

#[pymethods]
impl ImageHDU {
    #[new]
    fn new(header: Vec<String>, index: usize) -> (Self, HDU) {
        (ImageHDU, HDU::new(header, index))
    }

    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let index: usize = super_.index();
        Ok(format!("<ImageHDU #{}>", index))
    }
}

#[pyclass(extends = HDU)]
struct TableHDU; // Binary table (BINTABLE)

#[pymethods]
impl TableHDU {
    #[new]
    fn new(header: Vec<String>, index: usize) -> (Self, HDU) {
        (TableHDU, HDU::new(header, index))
    }

    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let index: usize = super_.index();
        Ok(format!("<TableHDU (binary) #{}>", index))
    }
}

#[pyclass(extends = HDU)]
struct AsciiTableHDU; // ASCII table (TABLE)

#[pymethods]
impl AsciiTableHDU {
    #[new]
    fn new(header: Vec<String>, index: usize) -> (Self, HDU) {
        (AsciiTableHDU, HDU::new(header, index))
    }

    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let index: usize = super_.index();
        Ok(format!("<AsciiTableHDU #{}>", index))
    }
}

// ====================== Main FITS class ======================
#[pyclass]
struct FITS {
    filename: String,
    file: Option<std::fs::File>,
    hdus: Vec<Py<PyAny>>,
}

#[pymethods]
impl FITS {
    #[new]
    fn new(py: Python<'_>, filename: String, mode: String) -> PyResult<Self> {
        let mut options = OpenOptions::new();

        match mode.as_str() {
            "r"  => options.read(true),
            "w"  => options.write(true).truncate(true).create(true),
            "a"  => options.write(true).append(true).create(true),
            "r+" => options.read(true).write(true),
            "w+" => options.read(true).write(true).truncate(true).create(true),
            "a+" => options.read(true).write(true).append(true).create(true),
            _ => return Err(PyIOError::new_err(format!(
                "Unsupported mode '{}'. Supported modes: 'r', 'w', 'a', 'r+', 'w+', 'a+'",
                mode
            ))),
        };

        let mut file = options.open(&filename)
            .map_err(|e| PyIOError::new_err(format!("Failed to open '{}': {}", filename, e)))?;

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
                    // The FITS END card is exactly "END" in cols 1-8 with
                    // blanks in cols 9-80, which trims to the bare string "END".
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

            let is_image = header_cards.iter().any(|c| {
                c.starts_with("SIMPLE  =") || c.starts_with("XTENSION= 'IMAGE")
            });
            let is_binary_table = header_cards.iter().any(|c| c.starts_with("XTENSION= 'BINTABLE'"));
            let is_ascii_table = header_cards.iter().any(|c| c.starts_with("XTENSION= 'TABLE'"));

            let hdu_py: Py<PyAny> = if is_image {
                Py::new(py, ImageHDU::new(header_cards.clone(), hdus.len()))?.into()
            } else if is_binary_table {
                Py::new(py, TableHDU::new(header_cards.clone(), hdus.len()))?.into()
            } else if is_ascii_table {
                Py::new(py, AsciiTableHDU::new(header_cards.clone(), hdus.len()))?.into()
            } else {
                Py::new(py, HDU::new(header_cards.clone(), hdus.len()))?.into()
            };

            hdus.push(hdu_py);

            let data_size = calculate_data_size(&header_cards);
            let num_header_blocks = ((header_cards.len() + 35) / 36) as u64;
            let header_size = num_header_blocks * BLOCK_SIZE as u64;
            let total_hdu_size = header_size + data_size;

            offset += total_hdu_size;
            let _ = file.seek(SeekFrom::Start(offset));

            if offset >= file.metadata().map(|m| m.len()).unwrap_or(0) {
                break;
            }
        }

        let _ = file.seek(SeekFrom::Start(0));

        Ok(FITS {
            filename,
            file: Some(file),
            hdus,
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
        if let Some(file) = self.file.take() {
            let _ = file.sync_all();
        }
        Ok(())
    }

    #[getter]
    fn closed(&self) -> bool {
        self.file.is_none()
    }

    fn __repr__(&self) -> String {
        let status = if self.closed() { "closed" } else { "open" };
        format!("FITS('{}', {} HDUs, {})", self.filename, self.hdus.len(), status)
    }

    fn __len__(&self) -> usize {
        self.hdus.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let len = self.hdus.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        if idx < 0 || idx >= len {
            return Err(PyValueError::new_err(format!("HDU index {} out of range", index)));
        }
        Ok(self.hdus[idx as usize].clone_ref(py))
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

#[pymodule]
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FITS>()?;
    m.add_class::<HDU>()?;
    m.add_class::<ImageHDU>()?;
    m.add_class::<TableHDU>()?;
    m.add_class::<AsciiTableHDU>()?;
    Ok(())
}
