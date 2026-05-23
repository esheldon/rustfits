// ImageHDU pyclass + image read/write/slicing helpers + bitpix conversions
// + image-shape parsing + image-data write/extend helpers.  RawBuffer and
// byteswap_in_place live in common.rs (also used by the binary-table reader).

use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PySlice, PyTuple};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::Bound;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use std::sync::atomic::Ordering;

use crate::common::{
    byteswap_in_place, lock_file, parse_keyword, parse_keyword_float,
    parse_string_keyword,
    shift_file_tail_and_update_offsets, zero_fill_range,
    FileHandle, FileLayout, HduOffsets, RawBuffer, TaintFlag,
    BLOCK_SIZE, CARDS_PER_BLOCK, CARD_SIZE,
};
use crate::hdu::HDU;
use crate::header::card_int;

#[pyclass(extends = HDU, subclass)]
pub(crate) struct ImageHDU;

impl ImageHDU {
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
            ImageHDU,
            HDU::new(header, index, filename, offsets, layout, file, tainted),
        )
    }
}

#[pymethods]
impl ImageHDU {
    // Multi-line, fitsio-style repr.  Shows file, extension index,
    // type, EXTNAME (if present), and the image dtype + dims in numpy
    // axis order (slowest first — same order the user gets back from
    // a read).  Primary HDUs with NAXIS=0 show `dims: []`.
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (bitpix, shape) = parse_image_hdu_shape_lax(&cards)?;
        let dtype = bitpix_to_native_dtype(bitpix)?;
        let extname = parse_string_keyword(&cards, "EXTNAME");
        let bunit = parse_string_keyword(&cards, "BUNIT");

        let mut out = String::new();
        out.push_str(&format!("  file: {}\n", super_.filename));
        out.push_str(&format!("  extension: {}\n", super_.index));
        out.push_str("  type: IMAGE_HDU\n");
        if let Some(name) = extname {
            out.push_str(&format!("  extname: {}\n", name));
        }
        out.push_str("  image info:\n");
        out.push_str(&format!("    data type: {}\n", dtype));
        out.push_str(&format!("    dims: {:?}\n", shape));
        if let Some(u) = bunit {
            out.push_str(&format!("    unit: {}\n", u));
        }
        Ok(out)
    }

    // BUNIT header value (e.g. "Jy", "counts/s"), or None when unset.
    // Purely informational; nothing in the read/write path consumes
    // it.  Mirrors TableHDU.units (per-column) at the image level.
    #[getter]
    fn unit(slf: PyRef<'_, Self>) -> PyResult<Option<String>> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        Ok(parse_string_keyword(&cards, "BUNIT"))
    }

    // Image dimensions in numpy axis order (slowest first), as a
    // tuple.  Primary HDUs with NAXIS=0 return ().
    #[getter]
    fn shape(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_image_hdu_shape_lax(&cards)?;
        Ok(PyTuple::new(py, &shape)?.unbind())
    }

    // numpy dtype matching BITPIX — the type `read()` would return.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (bitpix, _) = parse_image_hdu_shape_lax(&cards)?;
        let dtype_str = bitpix_to_native_dtype(bitpix)?;
        let np = py.import("numpy")?;
        Ok(np.call_method1("dtype", (dtype_str,))?.unbind())
    }

    // NAXIS — number of image axes.  0 for primary HDUs with no data.
    #[getter]
    fn ndim(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_image_hdu_shape_lax(&cards)?;
        Ok(shape.len())
    }

    // Total pixel count (product of all NAXISn).  Returns 0 for
    // NAXIS=0 (empty shape would otherwise give the empty-product
    // identity 1, which is wrong for "no data").
    #[getter]
    fn size(slf: PyRef<'_, Self>) -> PyResult<u64> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_image_hdu_shape_lax(&cards)?;
        Ok(if shape.is_empty() { 0 } else { shape.iter().product() })
    }

    // Raw FITS BITPIX value (e.g. 8, 16, -32, -64).  Useful for
    // round-trip / standards-level inspection; everyday code should
    // prefer `.dtype`.
    #[getter]
    fn bitpix(slf: PyRef<'_, Self>) -> PyResult<i32> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (bitpix, _) = parse_image_hdu_shape_lax(&cards)?;
        Ok(bitpix)
    }

    // numpy convention: `len(arr)` is shape[0].  For a 2-D image
    // that's the row count; for a 1-D image the pixel count.  Returns
    // 0 when NAXIS=0 (no data section).
    fn __len__(slf: PyRef<'_, Self>) -> PyResult<usize> {
        let super_ = slf.into_super();
        let cards = super_.header_snapshot()?;
        let (_, shape) = parse_image_hdu_shape_lax(&cards)?;
        Ok(shape.first().copied().unwrap_or(0) as usize)
    }

    #[pyo3(signature = (data, start=None))]
    fn write(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        start: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        write_image_data(
            py, &header_cards, super_.offsets.data_offset(),
            &super_.file, data, start,
        )
    }

    // `scale=True` (default) applies BSCALE/BZERO on read.  For files
    // with the unsigned-int trick (BITPIX=16/32/64, BZERO=2^(n-1), or
    // BITPIX=8, BZERO=-128), the result is returned in the matching
    // unsigned (or i1) dtype.  For general scaling, the result is
    // promoted to f8.  `scale=False` returns raw stored values in the
    // BITPIX native dtype.
    //
    // `mask_blank=True` (opt-in, default False) returns a
    // numpy.ma.MaskedArray with True at pixels whose stored value
    // matches the header's `BLANK` keyword.  Comparison is in stored
    // (pre-scaling) space per the FITS spec.  Only valid on integer
    // BITPIX (8/16/32/64); float BITPIX rejects up-front because the
    // spec forbids BLANK on floating-point arrays (NaN serves that
    // role).  When BLANK is absent from the header, returns a
    // MaskedArray with an all-False mask for consistent return type.
    #[pyo3(signature = (*, scale=true, mask_blank=false))]
    fn read(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        scale: bool,
        mask_blank: bool,
    ) -> PyResult<Py<PyAny>> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        read_image_data(
            py, &header_cards, super_.offsets.data_offset(),
            &super_.file, scale, mask_blank,
        )
    }

    // Grow the HDU's slow axis (numpy axis 0 = FITS NAXISn) if needed to
    // fit the data being written, then write it.  For HDUs that are not the
    // last on disk, the file tail (every byte from this HDU's data-section
    // end to EOF) is shifted forward to make room and every later HDU's
    // offsets are bumped in lockstep, so any previously-issued handles
    // remain valid (same shared-Arc model as the header-grow path).  See
    // CLAUDE.md "Header overflow: in-place file grow" for the shared
    // shift_file_tail_and_update_offsets primitive.
    #[pyo3(signature = (data, start=None))]
    fn extend(
        slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        start: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let super_: PyRefMut<HDU> = slf.into_super();

        let header_snapshot = super_.header_snapshot()?;
        let (bitpix, current_hdu_shape) = parse_image_hdu_shape(&header_snapshot)?;
        let naxis = current_hdu_shape.len();

        // MaskedArray entry — fill masked positions with sentinel.
        let unmasked = unwrap_masked_input(py, data, &header_snapshot, false)?;
        let data = unmasked.bind(py);

        // Accept BITPIX-native, or (for scaled HDUs) the scaled
        // dtype — reverse-transform to BITPIX-native in flight.
        // Done before any file mutation so a dtype error leaves the
        // file untouched.
        let data_owned = normalize_input_dtype(py, &header_snapshot, data)?;
        let data = data_owned.bind(py);
        let (_expected_kind, expected_size) = bitpix_to_numpy_kind(bitpix)?;

        let data_shape: Vec<u64> = data.getattr("shape")?.extract()?;
        if data_shape.len() != naxis {
            return Err(PyValueError::new_err(format!(
                "data has {} axes, HDU has {}", data_shape.len(), naxis
            )));
        }

        let start_vec: Vec<u64> = match start {
            Some(s) => {
                if s.len() != naxis {
                    return Err(PyValueError::new_err(format!(
                        "start has {} components, expected {}", s.len(), naxis
                    )));
                }
                let mut out = Vec::with_capacity(naxis);
                for (i, &v) in s.iter().enumerate() {
                    if v < 0 {
                        return Err(PyValueError::new_err(format!(
                            "start[{}] must be >= 0, got {}", i, v
                        )));
                    }
                    out.push(v as u64);
                }
                out
            }
            None => {
                let mut out = vec![0u64; naxis];
                out[0] = current_hdu_shape[0];
                out
            }
        };

        for i in 1..naxis {
            if start_vec[i] + data_shape[i] > current_hdu_shape[i] {
                return Err(PyValueError::new_err(format!(
                    "axis {}: start ({}) + data shape ({}) exceeds HDU dim ({}); \
                     extend only grows the slow axis (numpy axis 0)",
                    i, start_vec[i], data_shape[i], current_hdu_shape[i]
                )));
            }
        }

        let mut new_hdu_shape = current_hdu_shape.clone();
        let needed = start_vec[0] + data_shape[0];
        if needed > new_hdu_shape[0] {
            new_hdu_shape[0] = needed;
        }

        let start_for_write: Vec<i64> = start_vec.iter().map(|&v| v as i64).collect();

        if new_hdu_shape == current_hdu_shape {
            return write_image_data(
                py,
                &header_snapshot,
                super_.offsets.data_offset(),
                &super_.file,
                data,
                Some(start_for_write),
            );
        }

        let bpp = expected_size;
        let current_data_size: u64 = current_hdu_shape.iter().product::<u64>() * bpp;
        let new_data_size: u64 = new_hdu_shape.iter().product::<u64>() * bpp;
        let current_padded = round_up_to_block(current_data_size);
        let new_padded = round_up_to_block(new_data_size);

        let data_offset = super_.offsets.data_offset();
        let current_hdu_end = data_offset + current_padded;
        let new_hdu_end = data_offset + new_padded;

        // Dispatch by whether this HDU is the last thing on disk.  Last HDU:
        // just set_len (zero-extends).  Non-last: shift the tail forward to
        // make room, then zero-fill the gap (which contains the original
        // first delta bytes of the shifted tail — see shift_file_tail's
        // doc-comment).  Both branches end with the file sized to hold the
        // new HDU; the subsequent header+data writes treat the cases the same.
        if new_hdu_end > current_hdu_end {
            let delta = new_hdu_end - current_hdu_end;
            let file_len = {
                let guard = lock_file(&super_.file)?;
                let file = guard.as_ref()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                file.metadata()
                    .map_err(|e| PyIOError::new_err(e.to_string()))?
                    .len()
            };
            if file_len > current_hdu_end {
                shift_file_tail_and_update_offsets(
                    &super_.file, &super_.layout,
                    current_hdu_end, delta, &super_.tainted,
                )?;
                zero_fill_range(
                    &super_.file, current_hdu_end, delta, &super_.tainted,
                )?;
            } else {
                let mut guard = lock_file(&super_.file)?;
                let file = guard.as_mut()
                    .ok_or_else(|| PyIOError::new_err("file is closed"))?;
                file.set_len(new_hdu_end)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
        }

        // Disk-write-before-commit: build the candidate cards on a clone,
        // write to disk under the file lock (taint on mid-write failure),
        // and only commit the in-memory cards on success.  Holding the
        // cards lock across the header write keeps any concurrent reader
        // from seeing the new-on-disk / old-in-memory mismatch.
        let naxisn_key = format!("NAXIS{}", naxis);
        let new_card = card_int(
            &naxisn_key,
            new_hdu_shape[0] as i64,
            &format!("length of data axis {}", naxis),
        );
        let mut cards_guard = super_.header.lock()
            .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
        let mut new_cards = cards_guard.clone();
        let card_idx = new_cards.iter()
            .position(|c| c.len() >= 8 && c[..8].trim() == naxisn_key)
            .ok_or_else(|| PyValueError::new_err(
                format!("header missing {}", naxisn_key)
            ))?;
        new_cards[card_idx] = new_card.trim_end().to_string();

        {
            let mut guard = lock_file(&super_.file)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;

            let header_bytes = serialize_header_to_disk_bytes(&new_cards);
            let header_offset = data_offset - header_bytes.len() as u64;
            file.seek(SeekFrom::Start(header_offset))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            file.write_all(&header_bytes).map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "header write failed mid-stream during extend: {}; \
                     the on-disk file may be inconsistent — \
                     close this FITS object and reopen the file to recover", e
                ))
            })?;
            file.flush().map_err(|e| {
                super_.tainted.store(true, Ordering::Release);
                PyIOError::new_err(format!(
                    "header flush failed during extend: {}; \
                     the on-disk file may be inconsistent — \
                     close this FITS object and reopen the file to recover", e
                ))
            })?;
        }

        *cards_guard = new_cards.clone();
        drop(cards_guard);

        // Image-data write completes the extend.  A failure here leaves
        // the on-disk header advertising the new shape but the file's
        // data section partly stale or unwritten — taint so the user is
        // forced to reopen rather than reading inconsistent bytes.
        write_image_data(
            py,
            &new_cards,
            data_offset,
            &super_.file,
            data,
            Some(start_for_write),
        ).map_err(|e| {
            super_.tainted.store(true, Ordering::Release);
            e
        })
    }

    // Image indexing follows numpy semantics: each integer index
    // reduces a dimension, each slice keeps one.  When EVERY axis is
    // indexed by an integer the result has zero dimensions left — we
    // unwrap the 0-d array to a numpy scalar (e.g. np.float64,
    // np.int32) so `hdu[5, 6]` matches `numpy_arr[5, 6]`.  Mixed
    // slice + int (e.g. `hdu[5, :]`) still returns an ndarray.
    fn __getitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        let (_bitpix, hdu_shape) = parse_image_hdu_shape(&header_cards)?;
        let slices = normalize_slice_key(key, &hdu_shape)?;
        let all_int = slices.iter().all(|s| s.is_int);
        // Always scale on __getitem__ — matches the table-side
        // convention.  Use ImageHDU.read(scale=False) to bypass.
        let arr_py = read_image_slice(
            py, &header_cards, super_.offsets.data_offset(),
            &super_.file, &slices, true,
        )?;
        if all_int {
            // arr[()] indexes a 0-d ndarray to extract its single
            // value as a numpy scalar.  Empty PyTuple is the (,)
            // empty-tuple key Python sees.
            let arr_bound = arr_py.bind(py);
            Ok(arr_bound.get_item(PyTuple::empty(py))?.unbind())
        } else {
            Ok(arr_py)
        }
    }

    // Symmetric write surface for __getitem__.  Same slice parser, so
    // anything `img[key]` reads, `img[key] = value` writes.
    //
    // RHS forms:
    //   - Python int / float, numpy scalar, or 0-d ndarray
    //     (`np.ndim(value) == 0`) → broadcast: every pixel in the
    //     selection gets this value.
    //   - numpy ndarray with `shape == img[key].shape`, dtype
    //     matching BITPIX → write elementwise.
    //
    // No general numpy broadcasting (scalar-only).  Dtype is strict;
    // convert with `.astype(...)` if you need to.  Mid-write I/O
    // failures taint the file (close + reopen to recover).
    fn __setitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        let (_bitpix, hdu_shape) = parse_image_hdu_shape(&header_cards)?;
        let slices = normalize_slice_key(key, &hdu_shape)?;
        write_image_slice(
            py, &header_cards, super_.offsets.data_offset(),
            &super_.file, &slices, value, &super_.tainted,
        )
    }

    // ----- FITS checksum convention -----
    //
    // add_datasum: compute DATASUM from the data section, update
    // the card on disk + in memory.  add_checksum: add_datasum
    // first, then compute the full HDU checksum and encode it
    // into the CHECKSUM card.  verify_datasum / verify_checksum
    // return True/False/None (None = card absent).  Manual
    // semantics — checksums become stale after write/__setitem__/
    // extend; users must re-run add_checksum after mutations.

    fn add_datasum(slf: PyRef<'_, Self>) -> PyResult<()> {
        let super_: PyRef<HDU> = slf.into_super();
        checksum_hdu_add_datasum(&super_, "DATASUM")
    }

    fn add_checksum(slf: PyRef<'_, Self>) -> PyResult<()> {
        let super_: PyRef<HDU> = slf.into_super();
        checksum_hdu_add_checksum(&super_, "CHECKSUM", "DATASUM")
    }

    fn verify_datasum(slf: PyRef<'_, Self>) -> PyResult<Option<bool>> {
        let super_: PyRef<HDU> = slf.into_super();
        checksum_hdu_verify_datasum(&super_, "DATASUM")
    }

    fn verify_checksum(slf: PyRef<'_, Self>) -> PyResult<Option<bool>> {
        let super_: PyRef<HDU> = slf.into_super();
        checksum_hdu_verify_checksum(&super_, "CHECKSUM")
    }
}

// ----- shared HDU-level checksum helpers -----
//
// Used by ImageHDU + TableHDU directly (BLANK→DATASUM /
// CHECKSUM keys) and by CompressedImageHDU under different
// keys (ZDATASUM / ZHECKSUM), but the latter computes against
// the conceptual UNCOMPRESSED bytes, not the on-disk BINTABLE
// — so it implements its own dispatch and only shares the
// card-rewrite + disk-IO scaffolding indirectly.

// Read the entire on-disk data section (padded to BLOCK_SIZE)
// for this HDU.  Used by every checksum operation that needs
// raw bytes.  Returns 0 bytes for HDUs with no data section.
pub(crate) fn read_padded_data_section(
    super_: &HDU,
) -> PyResult<Vec<u8>> {
    let data_offset = super_.offsets.data_offset();
    let bytes_count = data_section_padded_size_for(super_)?;
    if bytes_count == 0 {
        return Ok(Vec::new());
    }
    let mut buf = vec![0u8; bytes_count as usize];
    let mut g = lock_file(&super_.file)?;
    let f = g
        .as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;
    f.seek(SeekFrom::Start(data_offset))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    f.read_exact(&mut buf)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(buf)
}

// Compute the padded byte-size of the data section for any HDU
// type the checksum routines care about (image, table,
// compressed BINTABLE).  Uses NAXIS / NAXISn / PCOUNT / GCOUNT
// — generic to FITS HDU layout.
fn data_section_padded_size_for(super_: &HDU) -> PyResult<u64> {
    let cards = super_.header_snapshot()?;
    let naxis: i64 = parse_keyword(&cards, "NAXIS").unwrap_or(0);
    if naxis == 0 {
        return Ok(0);
    }
    let bitpix: i64 = parse_keyword(&cards, "BITPIX").ok_or_else(|| {
        PyValueError::new_err("HDU header missing BITPIX")
    })?;
    let bytes_per_pixel = (bitpix.unsigned_abs() / 8) as u64;
    let mut nelements: u64 = 1;
    for i in 1..=naxis {
        let n: i64 = parse_keyword(&cards, &format!("NAXIS{}", i))
            .unwrap_or(0)
            .max(0);
        nelements = nelements.saturating_mul(n as u64);
    }
    // BINTABLE: PCOUNT may be > 0 (heap follows main data).
    let pcount: u64 =
        parse_keyword(&cards, "PCOUNT").unwrap_or(0).max(0) as u64;
    let gcount: u64 =
        parse_keyword(&cards, "GCOUNT").unwrap_or(1).max(1) as u64;
    let data_size = bytes_per_pixel
        .saturating_mul(nelements)
        .saturating_mul(gcount)
        .saturating_add(pcount);
    Ok(round_up_to_block(data_size))
}

pub(crate) fn checksum_hdu_add_datasum(
    super_: &HDU, datasum_key: &str,
) -> PyResult<()> {
    let data_bytes = read_padded_data_section(super_)?;
    let sum = crate::checksum::compute_datasum_of(&data_bytes);
    let cards = super_.header_snapshot()?;
    let new_cards =
        crate::checksum::cards_with_datasum(&cards, sum, datasum_key);
    commit_header_update(super_, new_cards)
}

pub(crate) fn checksum_hdu_add_checksum(
    super_: &HDU, checksum_key: &str, datasum_key: &str,
) -> PyResult<()> {
    let data_bytes = read_padded_data_section(super_)?;
    let datasum = crate::checksum::compute_datasum_of(&data_bytes);
    let cards = super_.header_snapshot()?;
    // First put DATASUM in (so the checksum step sees its bytes
    // in the header buffer).
    let cards = crate::checksum::cards_with_datasum(
        &cards, datasum, datasum_key,
    );
    // Then compute CHECKSUM against (header with placeholder +
    // data) and encode the complement.
    let new_cards = crate::checksum::cards_with_checksum(
        &cards, datasum, checksum_key,
    );
    commit_header_update(super_, new_cards)
}

pub(crate) fn checksum_hdu_verify_datasum(
    super_: &HDU, datasum_key: &str,
) -> PyResult<Option<bool>> {
    let cards = super_.header_snapshot()?;
    let Some(expected_str) = parse_string_keyword(&cards, datasum_key)
    else {
        return Ok(None);
    };
    let Some(expected) =
        crate::checksum::parse_datasum(expected_str.trim())
    else {
        return Ok(None);
    };
    let data_bytes = read_padded_data_section(super_)?;
    let computed = crate::checksum::compute_datasum_of(&data_bytes);
    Ok(Some(computed == expected))
}

pub(crate) fn checksum_hdu_verify_checksum(
    super_: &HDU, checksum_key: &str,
) -> PyResult<Option<bool>> {
    let cards = super_.header_snapshot()?;
    if parse_string_keyword(&cards, checksum_key).is_none() {
        return Ok(None);
    }
    let data_bytes = read_padded_data_section(super_)?;
    let total =
        crate::checksum::compute_hdu_checksum(&cards, &data_bytes);
    Ok(Some(total == 0xFFFF_FFFF))
}

// Rewrite the header on disk with `new_cards` and commit to
// memory.  Uses rewrite_header_to_disk so a header that grows
// past its reserved blocks gets in-place expansion (same machinery
// the FITSHeader edit path uses).
pub(crate) fn commit_header_update(
    super_: &HDU, new_cards: Vec<String>,
) -> PyResult<()> {
    let mut header_guard = super_.header.lock().map_err(|_| {
        PyIOError::new_err("header lock poisoned")
    })?;
    crate::header::rewrite_header_to_disk(
        &super_.file,
        &super_.offsets,
        &super_.layout,
        &new_cards,
        &super_.tainted,
    )?;
    *header_guard = new_cards;
    Ok(())
}

// Allocate a numpy array sized + typed to this HDU and fill it with the
// HDU's data from disk.  Result is native-endian; FITS big-endian bytes
// are swapped in place after read.
fn read_image_data(
    py: Python<'_>,
    header: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    scale: bool,
    mask_blank: bool,
) -> PyResult<Py<PyAny>> {
    let (bitpix, hdu_shape) = parse_image_hdu_shape(header)?;
    if mask_blank && bitpix < 0 {
        return Err(PyValueError::new_err(format!(
            "mask_blank=True is not valid on float BITPIX ({}); the \
             FITS standard forbids BLANK on floating-point arrays \
             (NaN serves that role).  Use mask_blank=False, or \
             post-process with numpy.isnan.",
            bitpix
        )));
    }
    let bpp = (bitpix.abs() / 8) as u64;
    let total_pixels: u64 = hdu_shape.iter().product();
    let total_bytes = (total_pixels * bpp) as usize;

    let dtype_str = bitpix_to_native_dtype(bitpix)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("empty", (hdu_shape.clone(), dtype_str))?;

    {
        let mut buffer = RawBuffer::acquire_writable(&arr)?;
        if buffer.len() != total_bytes {
            return Err(PyValueError::new_err(format!(
                "allocated buffer length ({}) does not match expected ({})",
                buffer.len(), total_bytes
            )));
        }

        if total_bytes > 0 {
            let mut guard = lock_file(file_handle)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;
            file.seek(SeekFrom::Start(data_offset))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            file.read_exact(buffer.as_mut_slice())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }

        if bpp > 1 && !cfg!(target_endian = "big") {
            byteswap_in_place(buffer.as_mut_slice(), bpp as usize);
        }
    }

    // Compute mask in stored (pre-scaling) space before applying any
    // scaling.  Per the FITS spec, BLANK is the raw on-disk sentinel.
    let mask_opt = if mask_blank {
        compute_blank_mask(header, &arr)?
    } else {
        None
    };

    let arr_unbound = if scale {
        let (bscale, bzero) = parse_bscale_bzero(header);
        let kind = image_scaling_kind(bitpix, bscale, bzero);
        apply_image_scaling(py, arr.unbind(), bitpix, kind, bscale, bzero)?
    } else {
        arr.unbind()
    };

    if mask_blank {
        wrap_in_masked_array(py, arr_unbound, mask_opt)
    } else {
        Ok(arr_unbound)
    }
}

// Compute a per-pixel bool mask of `arr == <header[key]>`.  Returns
// None when the keyword is absent (caller wraps the data with nomask
// for consistent return type without an unused mask allocation).
// Used for both `BLANK` (uncompressed images) and `ZBLANK` (tile-
// compressed images) — the spec mandates the integer sentinel
// comparison happens in stored space (pre-scaling).
pub(crate) fn compute_blank_mask_for_key(
    header: &[String],
    arr: &Bound<'_, PyAny>,
    key: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let Some(blank) = parse_keyword(header, key) else {
        return Ok(None);
    };
    // arr == blank: numpy broadcasts the Python int against arr's
    // dtype.  If `blank` is out of range for the dtype, no element
    // matches and the result is all-False — harmless.
    let mask = arr.call_method1("__eq__", (blank,))?;
    Ok(Some(mask.unbind()))
}

fn compute_blank_mask(
    header: &[String],
    arr: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    compute_blank_mask_for_key(header, arr, "BLANK")
}

// Wrap a plain ndarray in numpy.ma.MaskedArray.  None mask → nomask
// (no allocation overhead, but the return type stays MaskedArray for
// consistency).  Mirrors `wrap_masked` in hdu_table.rs.
// MaskedArray input handling for write/__setitem__/extend.
//
// User-facing API: pass a numpy.ma.MaskedArray to any write entry
// point; masked positions are auto-filled with the appropriate
// sentinel before encoding.  Sentinel source:
//   - Float dtype: NaN (always available, regardless of header).
//   - Integer dtype: header keyword (BLANK for uncompressed,
//     ZBLANK for compressed), in STORED space — transformed to
//     PHYSICAL space for the fill since the user's MaskedArray
//     is in their dtype (which may be u2/u4/etc. after the
//     unsigned-int trick).
//   - Integer dtype with no BLANK/ZBLANK in header: clear error
//     pointing user at create_image_hdu(..., blank=...).
//
// Returns the underlying ndarray when input is not a MaskedArray
// (zero overhead for the common case) or when the mask is
// `nomask` (no masked positions to fill).
//
// Single entry point shared by all 6 write paths (uncompressed +
// compressed × write/extend/__setitem__).  Each call site is a
// one-line `let data = unwrap_masked_input(py, data, header,
// is_compressed)?.bind(py)` — the helper internally parses
// BITPIX/ZBITPIX and BSCALE/BZERO from the header.
pub(crate) fn unwrap_masked_input(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    header: &[String],
    is_compressed: bool,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let ma_mod = np.getattr("ma")?;
    let is_ma: bool = ma_mod
        .call_method1("isMaskedArray", (data,))?
        .extract()?;
    if !is_ma {
        return Ok(data.clone().unbind());
    }
    // Get the mask; if it's the nomask singleton, no fill is needed
    // and we can return the underlying data directly.
    let mask = data.getattr("mask")?;
    let nomask = ma_mod.getattr("nomask")?;
    if mask.is(&nomask) {
        return Ok(data.getattr("data")?.unbind());
    }
    // Also handle the case where mask is a bool array but all-False —
    // `.any()` is cheap and lets us skip the fill.
    let any_masked: bool = mask.call_method0("any")?.extract()?;
    if !any_masked {
        return Ok(data.getattr("data")?.unbind());
    }

    // Determine is_float from the image-side BITPIX (BITPIX for
    // uncompressed, ZBITPIX for compressed).
    let bitpix_key = if is_compressed { "ZBITPIX" } else { "BITPIX" };
    let bitpix: i32 = parse_keyword(header, bitpix_key).ok_or_else(|| {
        PyValueError::new_err(format!("header missing {}", bitpix_key))
    })? as i32;
    let is_float = bitpix < 0;

    // Determine the fill value.
    let fill_value: Py<PyAny> = if is_float {
        // NaN works for both f4 and f8 (numpy.ma.filled casts the
        // scalar to the array's dtype).
        np.getattr("nan")?.unbind()
    } else {
        let key = if is_compressed { "ZBLANK" } else { "BLANK" };
        let Some(stored) = parse_keyword(header, key) else {
            return Err(PyValueError::new_err(format!(
                "MaskedArray input requires {} to be set in the \
                 header so masked positions can be filled with the \
                 sentinel value.  Set it at create time with \
                 `create_image_hdu(..., blank=<sentinel>)`, or fill \
                 the masked positions yourself before write.",
                key,
            )));
        };
        // Transform stored space → physical space (the user's
        // dtype).  Identity for plain integer dtypes; addition of
        // BZERO for unsigned-trick dtypes.
        let (_bs, bz) = parse_bscale_bzero(header);
        let physical = if bz == 0.0 {
            stored
        } else {
            ((stored as f64) + bz) as i64
        };
        physical.into_pyobject(py)?.unbind().into_any()
    };

    let filled = data.call_method1("filled", (fill_value,))?;
    Ok(filled.unbind())
}

pub(crate) fn wrap_in_masked_array(
    py: Python<'_>,
    data: Py<PyAny>,
    mask: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let ma = py.import("numpy")?.getattr("ma")?;
    let mask_obj = match mask {
        Some(m) => m.into_bound(py),
        None => ma.getattr("nomask")?,
    };
    Ok(ma.call_method1(
        "MaskedArray", (data.into_bound(py), mask_obj))?.unbind())
}

// ===== Slicing for image reads =====

#[derive(Debug, Clone)]
pub(crate) struct AxisSlice {
    pub(crate) start: u64,
    pub(crate) step: u64,
    pub(crate) count: u64,
    pub(crate) is_int: bool,
}

fn full_axis_slice(dim: u64) -> AxisSlice {
    AxisSlice { start: 0, step: 1, count: dim, is_int: false }
}

fn parse_axis_indexer(item: &Bound<'_, PyAny>, dim: u64) -> PyResult<AxisSlice> {
    if let Ok(slice) = item.cast::<PySlice>() {
        let indices = slice.indices(dim as isize)?;
        if indices.step <= 0 {
            return Err(PyValueError::new_err(
                "negative or zero step is not supported"
            ));
        }
        let start = indices.start.max(0) as u64;
        let step = indices.step as u64;
        let count = indices.slicelength as u64;
        Ok(AxisSlice { start, step, count, is_int: false })
    } else if let Ok(i) = item.extract::<i64>() {
        let dim_i = dim as i64;
        let normalized = if i < 0 { dim_i + i } else { i };
        if normalized < 0 || normalized >= dim_i {
            return Err(PyIndexError::new_err(format!(
                "index {} out of bounds for axis of size {}", i, dim
            )));
        }
        Ok(AxisSlice {
            start: normalized as u64,
            step: 1,
            count: 1,
            is_int: true,
        })
    } else {
        Err(PyValueError::new_err(
            "unsupported index type (expected int, slice, or Ellipsis)",
        ))
    }
}

pub(crate) fn normalize_slice_key(
    key: &Bound<'_, PyAny>,
    hdu_shape: &[u64],
) -> PyResult<Vec<AxisSlice>> {
    let naxis = hdu_shape.len();

    let items: Vec<Bound<PyAny>> = if let Ok(tup) = key.cast::<PyTuple>() {
        tup.iter().collect()
    } else {
        vec![key.clone()]
    };

    let mut ellipsis_pos: Option<usize> = None;
    for (i, item) in items.iter().enumerate() {
        if item.is_instance_of::<PyEllipsis>() {
            if ellipsis_pos.is_some() {
                return Err(PyValueError::new_err(
                    "an index can only have a single ellipsis",
                ));
            }
            ellipsis_pos = Some(i);
        }
    }

    let n_explicit = items.len() - usize::from(ellipsis_pos.is_some());
    if n_explicit > naxis {
        return Err(PyValueError::new_err(format!(
            "too many indices for array: HDU has {} axes, got {} explicit",
            naxis, n_explicit
        )));
    }
    let n_ellipsis_fill = naxis - n_explicit;

    let mut out: Vec<AxisSlice> = Vec::with_capacity(naxis);
    let mut axis = 0usize;
    for (i, item) in items.iter().enumerate() {
        if Some(i) == ellipsis_pos {
            for _ in 0..n_ellipsis_fill {
                out.push(full_axis_slice(hdu_shape[axis]));
                axis += 1;
            }
        } else {
            out.push(parse_axis_indexer(item, hdu_shape[axis])?);
            axis += 1;
        }
    }
    while axis < naxis {
        out.push(full_axis_slice(hdu_shape[axis]));
        axis += 1;
    }

    Ok(out)
}

fn compute_read_strip_layout(
    hdu_shape: &[u64],
    slices: &[AxisSlice],
) -> (usize, u64) {
    let n = hdu_shape.len();
    let mut strip_pixels: u64 = 1;
    for axis in (0..n).rev() {
        if slices[axis].step != 1 {
            return (axis + 1, strip_pixels);
        }
        strip_pixels *= slices[axis].count;
        if slices[axis].start != 0 || slices[axis].count != hdu_shape[axis] {
            return (axis, strip_pixels);
        }
    }
    (0, strip_pixels)
}

fn read_image_slice(
    py: Python<'_>,
    header: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    slices: &[AxisSlice],
    scale: bool,
) -> PyResult<Py<PyAny>> {
    let (bitpix, hdu_shape) = parse_image_hdu_shape(header)?;
    let naxis = hdu_shape.len();
    if slices.len() != naxis {
        return Err(PyValueError::new_err(format!(
            "internal error: {} slices for {} axes", slices.len(), naxis
        )));
    }

    let output_shape: Vec<u64> = slices.iter()
        .filter(|s| !s.is_int)
        .map(|s| s.count)
        .collect();

    let bpp = (bitpix.abs() / 8) as u64;
    let dtype_str = bitpix_to_native_dtype(bitpix)?;
    let np = py.import("numpy")?;
    let arr = np.call_method1("empty", (output_shape.clone(), dtype_str))?;

    let total_pixels: u64 = slices.iter().map(|s| s.count).product();
    if total_pixels == 0 {
        return Ok(arr.unbind());
    }

    let hdu_strides = row_major_strides(&hdu_shape);
    let (outer_axes, strip_pixels) = compute_read_strip_layout(&hdu_shape, slices);
    let strip_bytes = (strip_pixels * bpp) as usize;

    let outer_count: u64 = slices[..outer_axes].iter().map(|s| s.count).product();

    let inner_start_pixels: u64 = (outer_axes..naxis)
        .map(|k| slices[k].start * hdu_strides[k])
        .sum();

    let mut buffer = RawBuffer::acquire_writable(&arr)?;
    let expected_buffer_len = (output_shape.iter().product::<u64>() * bpp) as usize;
    if buffer.len() != expected_buffer_len {
        return Err(PyValueError::new_err(format!(
            "allocated buffer length ({}) does not match expected ({})",
            buffer.len(), expected_buffer_len
        )));
    }

    {
        let mut guard = lock_file(file_handle)?;
        let file = guard.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;

        let mut output_offset: usize = 0;
        let mut iter_idx = vec![0u64; outer_axes];

        for _ in 0..outer_count {
            let mut src_pixel: u64 = inner_start_pixels;
            for k in 0..outer_axes {
                let src_axis_idx = slices[k].start + iter_idx[k] * slices[k].step;
                src_pixel += src_axis_idx * hdu_strides[k];
            }
            let file_pos = data_offset + src_pixel * bpp;

            file.seek(SeekFrom::Start(file_pos))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            let dest = &mut buffer.as_mut_slice()[output_offset..output_offset + strip_bytes];
            file.read_exact(dest)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;

            output_offset += strip_bytes;

            for axis in (0..outer_axes).rev() {
                iter_idx[axis] += 1;
                if iter_idx[axis] < slices[axis].count {
                    break;
                }
                iter_idx[axis] = 0;
            }
        }
    }

    if bpp > 1 && !cfg!(target_endian = "big") {
        byteswap_in_place(buffer.as_mut_slice(), bpp as usize);
    }
    drop(buffer);

    let arr_unbound = arr.unbind();
    if !scale {
        return Ok(arr_unbound);
    }
    let (bscale, bzero) = parse_bscale_bzero(header);
    let kind = image_scaling_kind(bitpix, bscale, bzero);
    apply_image_scaling(py, arr_unbound, bitpix, kind, bscale, bzero)
}

fn write_image_data(
    py: Python<'_>,
    header: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    data: &Bound<'_, PyAny>,
    start: Option<Vec<i64>>,
) -> PyResult<()> {
    let (bitpix, hdu_shape) = parse_image_hdu_shape(header)?;
    let naxis = hdu_shape.len();

    // MaskedArray-aware entry: if `data` is a numpy.ma.MaskedArray,
    // fill masked positions with the appropriate sentinel (NaN for
    // floats; BLANK from the header for integers).  No-op for plain
    // ndarrays.
    let unmasked = unwrap_masked_input(py, data, header, false)?;
    let unmasked_bound = unmasked.bind(py);

    // Single source of truth for input-dtype rules: accepts BITPIX
    // native dtype directly, OR the scaled dtype for HDUs with the
    // unsigned-int trick, OR f8 (physical) for HDUs with general
    // BSCALE/BZERO scaling — last two cases reverse-transform in flight.
    let data_owned = normalize_input_dtype(py, header, unmasked_bound)?;
    let data = data_owned.bind(py);

    let dtype = data.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize_attr: u64 = dtype.getattr("itemsize")?.extract()?;
    let (expected_kind, expected_size) = bitpix_to_numpy_kind(bitpix)?;
    // Defensive: normalize_input_dtype must produce BITPIX-native
    // dtype.  Mismatch here is an internal logic bug, not user input.
    debug_assert!(
        kind == expected_kind && itemsize_attr == expected_size,
        "normalize_input_dtype returned wrong dtype: {}{} != {}{}",
        kind, itemsize_attr, expected_kind, expected_size,
    );

    let data_shape: Vec<u64> = data.getattr("shape")?.extract()?;
    if data_shape.len() != naxis {
        return Err(PyValueError::new_err(format!(
            "data has {} axes, HDU has {}", data_shape.len(), naxis
        )));
    }

    let start_vec: Vec<u64> = match start {
        Some(s) => {
            if s.len() != naxis {
                return Err(PyValueError::new_err(format!(
                    "start has {} components, expected {}", s.len(), naxis
                )));
            }
            let mut out = Vec::with_capacity(naxis);
            for (i, &v) in s.iter().enumerate() {
                if v < 0 {
                    return Err(PyValueError::new_err(format!(
                        "start[{}] must be >= 0, got {}", i, v
                    )));
                }
                out.push(v as u64);
            }
            out
        }
        None => vec![0u64; naxis],
    };

    for i in 0..naxis {
        if start_vec[i] + data_shape[i] > hdu_shape[i] {
            return Err(PyValueError::new_err(format!(
                "axis {}: start ({}) + data shape ({}) exceeds HDU dim ({})",
                i, start_vec[i], data_shape[i], hdu_shape[i]
            )));
        }
    }

    let total_pixels: u64 = data_shape.iter().product();
    if total_pixels == 0 {
        return Ok(());
    }

    let dtype_str: String = dtype.getattr("str")?.extract()?;
    let needs_swap = if expected_size == 1 {
        false
    } else {
        match dtype_str.chars().next() {
            Some('>') | Some('|') => false,
            Some('<') => true,
            _ => return Err(PyValueError::new_err(format!(
                "unrecognized dtype byteorder in '{}'", dtype_str
            ))),
        }
    };

    let buffer = RawBuffer::acquire(data).map_err(|e| {
        PyValueError::new_err(format!(
            "data must be a C-contiguous numpy array \
             (try np.ascontiguousarray): {}", e
        ))
    })?;
    if buffer.itemsize() as u64 != expected_size {
        return Err(PyValueError::new_err(
            "buffer itemsize disagrees with dtype.itemsize",
        ));
    }
    let bpp = expected_size;
    if buffer.len() as u64 != total_pixels * bpp {
        return Err(PyValueError::new_err(
            "buffer length disagrees with data shape",
        ));
    }
    let data_bytes = buffer.as_slice();

    let (outer_axes, strip_pixels) =
        compute_strip_layout(&hdu_shape, &data_shape, &start_vec);
    let strip_bytes = (strip_pixels * bpp) as usize;
    let hdu_strides = row_major_strides(&hdu_shape);
    let src_strides = row_major_strides(&data_shape);

    let base_pixel: u64 = (0..naxis)
        .map(|k| start_vec[k] * hdu_strides[k])
        .sum();

    let mut scratch: Vec<u8> = if needs_swap {
        vec![0u8; strip_bytes]
    } else {
        Vec::new()
    };

    let outer_count: u64 = data_shape[..outer_axes].iter().product();
    let mut idx = vec![0u64; outer_axes];

    let mut guard = lock_file(file_handle)?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    for _ in 0..outer_count {
        let mut hdu_pixel = base_pixel;
        let mut src_pixel: u64 = 0;
        for axis in 0..outer_axes {
            hdu_pixel += idx[axis] * hdu_strides[axis];
            src_pixel += idx[axis] * src_strides[axis];
        }
        let file_pos = data_offset + hdu_pixel * bpp;
        let src_byte = (src_pixel * bpp) as usize;
        let src = &data_bytes[src_byte..src_byte + strip_bytes];

        file.seek(SeekFrom::Start(file_pos))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        if needs_swap {
            scratch.copy_from_slice(src);
            byteswap_in_place(&mut scratch, bpp as usize);
            file.write_all(&scratch)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        } else {
            file.write_all(src)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }

        for axis in (0..outer_axes).rev() {
            idx[axis] += 1;
            if idx[axis] < data_shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }

    file.flush()
        .map_err(|e| PyIOError::new_err(e.to_string()))?;

    Ok(())
}

// Encode a Python scalar (int/float, numpy scalar, or 0-d ndarray)
// into FITS on-disk bytes (big-endian) for the given BITPIX.  Out-of-
// range values fail the relevant `extract::<...>()` with OverflowError;
// type mismatches (e.g. float -> int dtype) fail with TypeError.
// Both surface unmodified to the caller, which is what we want.
fn scalar_to_be_bytes(value: &Bound<'_, PyAny>, bitpix: i32) -> PyResult<Vec<u8>> {
    Ok(match bitpix {
        8 => {
            let v: u8 = value.extract()?;
            v.to_be_bytes().to_vec()
        }
        16 => {
            let v: i16 = value.extract()?;
            v.to_be_bytes().to_vec()
        }
        32 => {
            let v: i32 = value.extract()?;
            v.to_be_bytes().to_vec()
        }
        64 => {
            let v: i64 = value.extract()?;
            v.to_be_bytes().to_vec()
        }
        -32 => {
            let v: f32 = value.extract()?;
            v.to_be_bytes().to_vec()
        }
        -64 => {
            let v: f64 = value.extract()?;
            v.to_be_bytes().to_vec()
        }
        _ => return Err(PyValueError::new_err(format!(
            "unsupported BITPIX {}", bitpix
        ))),
    })
}

// Slice-based image write: the __setitem__ companion to
// `read_image_slice`.  Walks the file using the same strip layout as
// the read path (compute_read_strip_layout handles stepped slices by
// moving them into the outer iteration), writing one strip per outer
// step.  RHS is either a scalar (Python int/float, numpy scalar, or
// 0-d ndarray — broadcast to every selected pixel) or a numpy ndarray
// whose shape exactly matches the slice's output shape and whose
// dtype matches BITPIX.  Failures inside the seek/write/flush loop
// set the per-file taint flag because the data section may now be
// inconsistent — recovery is via close+reopen.
fn write_image_slice(
    py: Python<'_>,
    header: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    slices: &[AxisSlice],
    rhs: &Bound<'_, PyAny>,
    tainted: &TaintFlag,
) -> PyResult<()> {
    let (bitpix, hdu_shape) = parse_image_hdu_shape(header)?;
    let bpp = (bitpix.abs() / 8) as u64;
    let bpp_usize = bpp as usize;
    let naxis = hdu_shape.len();

    // MaskedArray entry — fill masked positions with sentinel before
    // the per-strip byte-pump below.  No-op for plain ndarray or
    // scalar (the scalar broadcast branch never sees a MaskedArray
    // because np.ndim(scalar) == 0).
    let unmasked = unwrap_masked_input(py, rhs, header, false)?;
    let rhs = unmasked.bind(py);

    // The slice's output shape (in numpy axis order) — what an
    // equivalent read would return.  Used for RHS shape validation.
    let output_shape: Vec<u64> = slices.iter()
        .filter(|s| !s.is_int)
        .map(|s| s.count)
        .collect();

    let total_pixels: u64 = slices.iter().map(|s| s.count).product();
    if total_pixels == 0 {
        return Ok(());
    }

    let (outer_axes, strip_pixels) =
        compute_read_strip_layout(&hdu_shape, slices);
    let strip_bytes = (strip_pixels as usize) * bpp_usize;

    // Discriminate scalar from ndarray via `numpy.ndim`.  This returns
    // 0 for Python int/float, numpy scalars, AND 0-d ndarrays — exactly
    // the broadcast cases.  Higher ndim is treated as an ndarray RHS
    // whose shape must match output_shape exactly.
    let np = py.import("numpy")?;
    let rhs_ndim: usize = np.call_method1("ndim", (rhs,))?.extract()?;
    let is_scalar = rhs_ndim == 0;

    // Source bytes in big-endian (disk) layout.  For scalar broadcast
    // we build a one-strip buffer and reuse it; for ndarray we copy +
    // byteswap the whole thing once and slice strip-by-strip below.
    let (source_bytes, per_strip): (Vec<u8>, bool) = if is_scalar {
        let pixel_bytes = scalar_to_be_bytes(rhs, bitpix)?;
        let mut strip = Vec::with_capacity(strip_bytes);
        for _ in 0..strip_pixels {
            strip.extend_from_slice(&pixel_bytes);
        }
        (strip, false)
    } else {
        let rhs_shape: Vec<u64> = rhs.getattr("shape")?.extract()?;
        if rhs_shape != output_shape {
            return Err(PyValueError::new_err(format!(
                "RHS shape {:?} does not match indexed output shape {:?}",
                rhs_shape, output_shape,
            )));
        }
        // Accept BITPIX-native, or scaled dtype for scaled HDUs —
        // reverse-transform in flight so the byte-pump below always
        // sees BITPIX-native bytes.
        let rhs_owned = normalize_input_dtype(py, header, rhs)?;
        let rhs = rhs_owned.bind(py);
        let dtype = rhs.getattr("dtype")?;
        let dtype_str: String = dtype.getattr("str")?.extract()?;
        let needs_swap = if bpp_usize == 1 {
            false
        } else {
            match dtype_str.chars().next() {
                Some('>') | Some('|') => false,
                Some('<') => true,
                Some('=') => !cfg!(target_endian = "big"),
                _ => return Err(PyValueError::new_err(format!(
                    "unrecognized dtype byteorder in '{}'", dtype_str
                ))),
            }
        };
        let buffer = RawBuffer::acquire(&rhs).map_err(|e| {
            PyValueError::new_err(format!(
                "RHS must be a C-contiguous numpy array \
                 (try np.ascontiguousarray): {}", e
            ))
        })?;
        let mut bytes = buffer.as_slice().to_vec();
        if needs_swap {
            byteswap_in_place(&mut bytes, bpp_usize);
        }
        (bytes, true)
    };

    // Outer iteration mirrors the read path exactly: walk strided
    // outer axes, seek to the file position for each strip, write the
    // strip.  Inner axes (always step=1 by the strip layout's
    // construction) contribute a fixed `inner_start_pixels` offset
    // that doesn't change across iterations.
    let hdu_strides = row_major_strides(&hdu_shape);
    let outer_count: u64 = slices[..outer_axes].iter()
        .map(|s| s.count)
        .product();
    let inner_start_pixels: u64 = (outer_axes..naxis)
        .map(|k| slices[k].start * hdu_strides[k])
        .sum();

    let mut output_offset: usize = 0;
    let mut iter_idx = vec![0u64; outer_axes];

    let mut guard = lock_file(file_handle)?;
    let file = guard.as_mut()
        .ok_or_else(|| PyIOError::new_err("file is closed"))?;

    let taint_io = |e: std::io::Error| {
        tainted.store(true, Ordering::Release);
        PyIOError::new_err(e.to_string())
    };

    for _ in 0..outer_count {
        let mut dst_pixel: u64 = inner_start_pixels;
        for k in 0..outer_axes {
            let dst_axis_idx =
                slices[k].start + iter_idx[k] * slices[k].step;
            dst_pixel += dst_axis_idx * hdu_strides[k];
        }
        let file_pos = data_offset + dst_pixel * bpp;

        file.seek(SeekFrom::Start(file_pos)).map_err(taint_io)?;
        let src = if per_strip {
            &source_bytes[output_offset..output_offset + strip_bytes]
        } else {
            &source_bytes[..]
        };
        file.write_all(src).map_err(taint_io)?;
        if per_strip {
            output_offset += strip_bytes;
        }

        for axis in (0..outer_axes).rev() {
            iter_idx[axis] += 1;
            if iter_idx[axis] < slices[axis].count {
                break;
            }
            iter_idx[axis] = 0;
        }
    }

    file.flush().map_err(taint_io)?;
    Ok(())
}

// ===== bitpix conversions =====

fn bitpix_to_numpy_kind(bitpix: i32) -> PyResult<(&'static str, u64)> {
    match bitpix {
        8   => Ok(("u", 1)),
        16  => Ok(("i", 2)),
        32  => Ok(("i", 4)),
        64  => Ok(("i", 8)),
        -32 => Ok(("f", 4)),
        -64 => Ok(("f", 8)),
        _   => Err(PyValueError::new_err(format!("unsupported BITPIX {}", bitpix))),
    }
}

pub(crate) fn bitpix_to_native_dtype(bitpix: i32) -> PyResult<&'static str> {
    match bitpix {
        8   => Ok("u1"),
        16  => Ok("i2"),
        32  => Ok("i4"),
        64  => Ok("i8"),
        -32 => Ok("f4"),
        -64 => Ok("f8"),
        _   => Err(PyValueError::new_err(format!("unsupported BITPIX {}", bitpix))),
    }
}

// ===== BSCALE / BZERO scaling =====

// Classification of how to apply BSCALE/BZERO on read.  Pre-computed
// once at read entry; mirrors the table-side ScalingKind enum in
// hdu_table.rs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalingKind {
    // BSCALE == 1 AND BZERO == 0 (or scale=False) — return stored
    // values in the BITPIX native dtype.
    None,
    // "Unsigned-int trick": BSCALE=1 plus BZERO equal to the type's
    // sign-bias (2^15 for BITPIX=16, 2^31 for BITPIX=32, 2^63 for
    // BITPIX=64, or -128 for BITPIX=8 → signed bytes).  Output
    // preserves integer semantics with no precision loss.
    UnsignedTrick,
    // Anything else: physical = BSCALE * stored + BZERO computed in
    // f64, output as f8.  i64 inputs may lose precision (53-bit
    // mantissa) — unavoidable.
    General,
}

pub(crate) fn parse_bscale_bzero(header: &[String]) -> (f64, f64) {
    let bscale = parse_keyword_float(header, "BSCALE").unwrap_or(1.0);
    let bzero = parse_keyword_float(header, "BZERO").unwrap_or(0.0);
    (bscale, bzero)
}

pub(crate) fn image_scaling_kind(bitpix: i32, bscale: f64, bzero: f64) -> ScalingKind {
    if bscale == 1.0 && bzero == 0.0 {
        return ScalingKind::None;
    }
    if bscale == 1.0 {
        let trick = matches!(
            (bitpix, bzero),
            (8, b)  if b == -128.0
        ) || matches!(
            (bitpix, bzero),
            (16, b) if b == 32768.0
        ) || matches!(
            (bitpix, bzero),
            (32, b) if b == 2147483648.0
        ) || matches!(
            (bitpix, bzero),
            (64, b) if b == 9223372036854775808.0
        );
        if trick {
            return ScalingKind::UnsignedTrick;
        }
    }
    ScalingKind::General
}

// numpy dtype string the array reads into after applying scaling.
// Only valid when kind != None.
fn scaled_image_dtype(bitpix: i32, kind: ScalingKind) -> &'static str {
    match kind {
        ScalingKind::UnsignedTrick => match bitpix {
            8  => "i1",
            16 => "u2",
            32 => "u4",
            64 => "u8",
            _ => unreachable!(
                "unsigned-trick scaling on float BITPIX {}", bitpix
            ),
        },
        ScalingKind::General => "f8",
        ScalingKind::None => unreachable!("scaled_image_dtype called with None"),
    }
}

// (numpy_kind, itemsize) of the scaled (user-facing) dtype for an
// HDU configured with the unsigned-int trick.  Mirrors
// `scaled_image_dtype` but returns the parts in the same shape as
// `bitpix_to_numpy_kind`, so input-dtype matching can use the same
// shape on both sides.  Only valid when the HDU's BSCALE/BZERO
// classify as UnsignedTrick.
fn scaled_dtype_kind_size(bitpix: i32) -> (&'static str, u64) {
    match bitpix {
        8  => ("i", 1),
        16 => ("u", 2),
        32 => ("u", 4),
        64 => ("u", 8),
        _ => unreachable!(
            "scaled_dtype_kind_size on unexpected BITPIX {}", bitpix
        ),
    }
}

// Validate input dtype against the HDU's BITPIX (and, for HDUs
// configured with the unsigned-int trick, the scaled dtype too).
// Returns the data to write — either the input array unchanged (in
// BITPIX dtype) or a freshly reverse-transformed array (BITPIX dtype,
// when the input was the scaled dtype).  Single source of truth for
// the write-side dtype rules; called from write_image_data,
// write_image_slice, and ImageHDU.extend.
fn normalize_input_dtype(
    py: Python<'_>,
    header: &[String],
    data: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let (bitpix, _) = parse_image_hdu_shape(header)?;
    let dtype = data.getattr("dtype")?;
    let input_kind: String = dtype.getattr("kind")?.extract()?;
    let input_size: u64 = dtype.getattr("itemsize")?.extract()?;
    let (expected_kind, expected_size) = bitpix_to_numpy_kind(bitpix)?;

    // Fast path: input already in BITPIX-native dtype.
    if input_kind == expected_kind && input_size == expected_size {
        return Ok(data.clone().unbind());
    }

    // Allow scaled-dtype input when the HDU has scaling configured;
    // reverse-transform to BITPIX dtype.
    let (bscale, bzero) = parse_bscale_bzero(header);
    let kind = image_scaling_kind(bitpix, bscale, bzero);
    match kind {
        ScalingKind::UnsignedTrick => {
            let (scaled_kind, scaled_size) = scaled_dtype_kind_size(bitpix);
            if input_kind == scaled_kind && input_size == scaled_size {
                return reverse_unsigned_trick(py, data, bitpix);
            }
        }
        ScalingKind::General => {
            // f8 (physical) input → reverse-transform to BITPIX dtype.
            if input_kind == "f" && input_size == 8 {
                return reverse_general_scaling(
                    py, data, bitpix, bscale, bzero,
                );
            }
        }
        ScalingKind::None => {}
    }

    let extra = match kind {
        ScalingKind::UnsignedTrick => {
            format!(" or scaled '{}'", scaled_image_dtype(bitpix, kind))
        }
        ScalingKind::General => " or scaled 'f8'".to_string(),
        ScalingKind::None => String::new(),
    };
    Err(PyValueError::new_err(format!(
        "data dtype ({}{}) does not match HDU BITPIX={} \
         (expected '{}{}'{})",
        input_kind, input_size, bitpix,
        expected_kind, expected_size, extra,
    )))
}

// Inverse of `apply_image_scaling`'s UnsignedTrick branch: scaled-
// dtype input (u2/u4/u8/i1) → BITPIX-native dtype output (i2/i4/i8/u1).
// Same primitives (XOR with sign bit + view-cast for bit reinterpret)
// applied in reverse.  Caller must guarantee input dtype matches the
// scaled dtype for this BITPIX.
//
// `pub(crate)` because the compressed-write path
// (`hdu_image_compressed.rs::write_compressed_image_data`) calls it
// directly when the HDU has BSCALE=1 + BZERO=2^(n-1) configured —
// the uncompressed `normalize_input_dtype` doesn't help because it
// parses from regular NAXIS/BITPIX rather than the Z-prefixed cards.
pub(crate) fn reverse_unsigned_trick(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    bitpix: i32,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    match bitpix {
        8 => {
            // i1 input → u1 stored.  View as u1 (bit reinterpret),
            // then XOR with 0x80 in u1 space.
            let view = arr.call_method1("view", ("u1",))?;
            let mask = np.call_method1("uint8", (0x80u8,))?;
            Ok(view.call_method1("__xor__", (mask,))?.unbind())
        }
        16 => {
            // u2 input → i2 stored.  XOR 0x8000 in u2 space, then
            // view as i2.
            let mask = np.call_method1("uint16", (0x8000u16,))?;
            let xored = arr.call_method1("__xor__", (mask,))?;
            Ok(xored.call_method1("view", ("i2",))?.unbind())
        }
        32 => {
            let mask = np.call_method1("uint32", (0x80000000u32,))?;
            let xored = arr.call_method1("__xor__", (mask,))?;
            Ok(xored.call_method1("view", ("i4",))?.unbind())
        }
        64 => {
            let mask = np.call_method1(
                "uint64", (0x8000000000000000u64,))?;
            let xored = arr.call_method1("__xor__", (mask,))?;
            Ok(xored.call_method1("view", ("i8",))?.unbind())
        }
        _ => unreachable!(
            "reverse_unsigned_trick on unexpected BITPIX {}", bitpix
        ),
    }
}

// Inverse of `apply_image_scaling`'s General branch: f8 (physical)
// input → BITPIX-native dtype output via
// `stored = (physical - bzero) / bscale`.  For integer BITPIX, non-
// finite values are rejected, rounding is half-to-even via `np.rint`,
// and post-rounding bounds violations are rejected too (so e.g.
// 32767.5 → rint → 32768 against BITPIX=16 raises rather than
// wrapping).  For float BITPIX (-32/-64), no rounding or bounds
// check — the cast is exact within the target dtype's precision.
// Caller must guarantee input dtype is f8 and kind == General.
fn reverse_general_scaling(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    bitpix: i32,
    bscale: f64,
    bzero: f64,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let shifted = arr.call_method1("__sub__", (bzero,))?;
    let stored_f8 = shifted.call_method1("__truediv__", (bscale,))?;
    let native_dtype = bitpix_to_native_dtype(bitpix)?;

    if bitpix < 0 {
        return Ok(
            stored_f8.call_method1("astype", (native_dtype,))?.unbind()
        );
    }

    let finite = np.call_method1("isfinite", (&stored_f8,))?;
    let all_finite: bool = finite.call_method0("all")?.extract()?;
    if !all_finite {
        return Err(PyValueError::new_err(format!(
            "cannot write non-finite values (NaN/Inf) to integer \
             BITPIX={} HDU with BSCALE/BZERO scaling: reverse \
             transform produced non-finite stored values",
            bitpix
        )));
    }

    let rounded = np.call_method1("rint", (&stored_f8,))?;
    let (min_f, max_f, min_str, max_str) = bitpix_int_bounds(bitpix);
    let lt_min = rounded.call_method1("__lt__", (min_f,))?;
    let gt_max = rounded.call_method1("__gt__", (max_f,))?;
    let any_lt: bool = lt_min.call_method0("any")?.extract()?;
    let any_gt: bool = gt_max.call_method0("any")?.extract()?;
    if any_lt || any_gt {
        return Err(PyValueError::new_err(format!(
            "values overflow BITPIX={} stored range [{}, {}] after \
             reverse BSCALE/BZERO transform (rounded half-to-even)",
            bitpix, min_str, max_str,
        )));
    }

    Ok(rounded.call_method1("astype", (native_dtype,))?.unbind())
}

// f64 bounds for the BITPIX integer dtypes, plus their literal-int
// string forms for error messages.  For BITPIX=64 the upper bound is
// 2^63 - 1024 (largest f64 below 2^63) because i64::MAX (2^63 - 1) is
// not exactly representable in f64.  Any physical input that would
// reverse-transform to a value beyond this can't be expressed in f64
// anyway, so the conservative bound doesn't lose anything reachable.
fn bitpix_int_bounds(bitpix: i32) -> (f64, f64, &'static str, &'static str) {
    match bitpix {
        8 => (0.0, 255.0, "0", "255"),
        16 => (-32768.0, 32767.0, "-32768", "32767"),
        32 => (
            -2147483648.0, 2147483647.0,
            "-2147483648", "2147483647",
        ),
        64 => (
            -9.223372036854776e18, 9.223372036854775e18,
            "-9223372036854775808", "9223372036854775807",
        ),
        _ => unreachable!(
            "bitpix_int_bounds on non-integer BITPIX {}", bitpix
        ),
    }
}

// Apply BSCALE/BZERO scaling to an as-read numpy array.  Returns a new
// array of the scaled dtype.  For UnsignedTrick: zero-copy view-cast
// followed by a vectorized XOR with the sign bit (equivalent to adding
// 2^(n-1) modulo 2^n in two's complement).  For General: promote to f8,
// multiply by BSCALE, add BZERO — all in numpy's vectorized loops.
pub(crate) fn apply_image_scaling(
    py: Python<'_>,
    arr: Py<PyAny>,
    bitpix: i32,
    kind: ScalingKind,
    bscale: f64,
    bzero: f64,
) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let arr_b = arr.bind(py);
    match kind {
        ScalingKind::None => Ok(arr),
        ScalingKind::UnsignedTrick => {
            let scaled_dtype = scaled_image_dtype(bitpix, kind);
            match bitpix {
                8 => {
                    // u1 → i1: XOR in u1 space, view as i1.
                    let mask = np.call_method1("uint8", (0x80u8,))?;
                    let xored = arr_b.call_method1("__xor__", (mask,))?;
                    Ok(xored.call_method1("view", ("i1",))?.unbind())
                }
                16 => {
                    let view = arr_b.call_method1("view", (scaled_dtype,))?;
                    let mask = np.call_method1("uint16", (0x8000u16,))?;
                    Ok(view.call_method1("__xor__", (mask,))?.unbind())
                }
                32 => {
                    let view = arr_b.call_method1("view", (scaled_dtype,))?;
                    let mask = np.call_method1("uint32", (0x80000000u32,))?;
                    Ok(view.call_method1("__xor__", (mask,))?.unbind())
                }
                64 => {
                    let view = arr_b.call_method1("view", (scaled_dtype,))?;
                    let mask = np.call_method1(
                        "uint64", (0x8000000000000000u64,))?;
                    Ok(view.call_method1("__xor__", (mask,))?.unbind())
                }
                _ => unreachable!(
                    "unsigned-trick scaling on unexpected BITPIX {}",
                    bitpix
                ),
            }
        }
        ScalingKind::General => {
            // physical = stored * bscale + bzero, in f8.
            let promoted = arr_b.call_method1("astype", ("f8",))?;
            let scaled = promoted.call_method1("__mul__", (bscale,))?;
            Ok(scaled.call_method1("__add__", (bzero,))?.unbind())
        }
    }
}

fn row_major_strides(shape: &[u64]) -> Vec<u64> {
    let n = shape.len();
    if n == 0 {
        return Vec::new();
    }
    let mut strides = vec![1u64; n];
    for k in (0..n - 1).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    strides
}

fn compute_strip_layout(
    hdu_shape: &[u64],
    data_shape: &[u64],
    start: &[u64],
) -> (usize, u64) {
    let n = hdu_shape.len();
    let mut strip_pixels: u64 = 1;
    for axis in (0..n).rev() {
        strip_pixels *= data_shape[axis];
        if start[axis] != 0 || data_shape[axis] != hdu_shape[axis] {
            return (axis, strip_pixels);
        }
    }
    (0, strip_pixels)
}

// Tolerant variant of `parse_image_hdu_shape`: returns `[]` for the
// NAXIS=0 case instead of erroring.  Used by __repr__ and the
// .shape / .dtype / .ndim / .size / .bitpix / __len__ accessors,
// which all need to work on primary HDUs with no data section.  The
// strict parse_image_hdu_shape (errors on NAXIS=0) stays in place for
// the read/write code paths that genuinely need a data section.
fn parse_image_hdu_shape_lax(header: &[String]) -> PyResult<(i32, Vec<u64>)> {
    let bitpix = parse_keyword(header, "BITPIX")
        .ok_or_else(|| PyValueError::new_err("HDU header missing BITPIX"))?
        as i32;
    let naxis = parse_keyword(header, "NAXIS").unwrap_or(0).max(0) as usize;
    let mut shape: Vec<u64> = Vec::with_capacity(naxis);
    for i in 1..=naxis {
        let d = parse_keyword(header, &format!("NAXIS{}", i))
            .unwrap_or(0).max(0) as u64;
        shape.push(d);
    }
    shape.reverse();  // numpy axis order: slowest first.
    Ok((bitpix, shape))
}

// Parse BITPIX, NAXIS, and NAXIS1..NAXISn out of an image HDU header.
// Returns (bitpix, hdu_shape_in_numpy_order).
fn parse_image_hdu_shape(header: &[String]) -> PyResult<(i32, Vec<u64>)> {
    let bitpix = parse_keyword(header, "BITPIX")
        .ok_or_else(|| PyValueError::new_err("HDU header missing BITPIX"))?
        as i32;
    let naxis = parse_keyword(header, "NAXIS")
        .ok_or_else(|| PyValueError::new_err("HDU header missing NAXIS"))?
        as usize;
    if naxis == 0 {
        return Err(PyValueError::new_err(
            "HDU has NAXIS=0 (no data section)",
        ));
    }
    let mut fits_dims: Vec<u64> = Vec::with_capacity(naxis);
    for i in 1..=naxis {
        let d = parse_keyword(header, &format!("NAXIS{}", i))
            .ok_or_else(|| PyValueError::new_err(
                format!("HDU header missing NAXIS{}", i)
            ))?;
        if d < 0 {
            return Err(PyValueError::new_err(
                format!("NAXIS{} is negative", i)
            ));
        }
        fits_dims.push(d as u64);
    }
    let hdu_shape: Vec<u64> = fits_dims.iter().rev().copied().collect();
    Ok((bitpix, hdu_shape))
}

pub(crate) fn round_up_to_block(n: u64) -> u64 {
    let block = BLOCK_SIZE as u64;
    ((n + block - 1) / block) * block
}

pub(crate) fn serialize_header_to_disk_bytes(header: &[String]) -> Vec<u8> {
    let num_blocks = (header.len() + CARDS_PER_BLOCK - 1) / CARDS_PER_BLOCK;
    let total_size = num_blocks * BLOCK_SIZE;
    let mut out = Vec::with_capacity(total_size);
    for card in header {
        let mut padded = card.clone();
        if padded.len() < CARD_SIZE {
            padded.push_str(&" ".repeat(CARD_SIZE - padded.len()));
        } else if padded.len() > CARD_SIZE {
            padded.truncate(CARD_SIZE);
        }
        out.extend_from_slice(padded.as_bytes());
    }
    while out.len() < total_size {
        out.push(b' ');
    }
    out
}

// Map a numpy short-code or long-name dtype string to a FITS BITPIX
// value plus an optional BZERO offset for the unsigned-int trick.
// Direct-mapped dtypes (u1/i2/i4/i8/f4/f8) return None for bzero;
// "scaled" dtypes (i1/u2/u4/u8) return the matching sign-bias so
// `create_image_hdu` can emit the BZERO card.
//
// The trick lets users round-trip unsigned (or signed-byte) arrays
// through FITS even though the on-disk BITPIX representation is the
// opposite signedness: write u2 input → stored as i2 + BZERO=32768;
// read back recovers the u2 dtype via apply_image_scaling.
//
// Lives next to its inverses (bitpix_to_numpy_kind /
// bitpix_to_native_dtype) so the supported-dtype set stays in one
// place.
pub(crate) fn dtype_to_bitpix(dtype: &str) -> PyResult<(i32, Option<f64>)> {
    let s = dtype.trim_start_matches(
        |c| c == '<' || c == '>' || c == '|' || c == '=');
    let normalized = s.to_lowercase();
    match normalized.as_str() {
        "u1" | "uint8"   => Ok((8,   None)),
        "i1" | "int8"    => Ok((8,   Some(-128.0))),
        "i2" | "int16"   => Ok((16,  None)),
        "u2" | "uint16"  => Ok((16,  Some(32768.0))),
        "i4" | "int32"   => Ok((32,  None)),
        "u4" | "uint32"  => Ok((32,  Some(2147483648.0))),
        "i8" | "int64"   => Ok((64,  None)),
        "u8" | "uint64"  => Ok((64,  Some(9223372036854775808.0))),
        "f4" | "float32" => Ok((-32, None)),
        "f8" | "float64" => Ok((-64, None)),
        _ => Err(PyValueError::new_err(format!(
            "unsupported numpy dtype '{}'. Supported: \
             'u1','i1','i2','u2','i4','u4','i8','u8','f4','f8'",
            dtype
        ))),
    }
}
