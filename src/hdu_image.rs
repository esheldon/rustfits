// ImageHDU pyclass + image read/write/slicing helpers + RawBuffer + bitpix
// conversions + image-shape parsing + image-data write/extend helpers.

use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PySlice, PyTuple};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::Bound;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use crate::common::{
    lock_file, parse_keyword, FileHandle, FileLayout, HduOffsets, TaintFlag,
    BLOCK_SIZE, CARDS_PER_BLOCK, CARD_SIZE,
};
use crate::hdu::HDU;
use crate::header::card_int;

#[pyclass(extends = HDU)]
pub(crate) struct ImageHDU;

impl ImageHDU {
    pub(crate) fn new(
        header: Vec<String>,
        index: usize,
        offsets: Arc<HduOffsets>,
        layout: Arc<FileLayout>,
        file: FileHandle,
        tainted: TaintFlag,
    ) -> (Self, HDU) {
        (
            ImageHDU,
            HDU::new(header, index, offsets, layout, file, tainted),
        )
    }
}

#[pymethods]
impl ImageHDU {
    fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let super_ = slf.into_super();
        let index: usize = super_.index();
        Ok(format!("<ImageHDU #{}>", index))
    }

    #[pyo3(signature = (data, start=None))]
    fn write(
        slf: PyRef<'_, Self>,
        data: &Bound<'_, PyAny>,
        start: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        write_image_data(&header_cards, super_.offsets.data_offset(), &super_.file, data, start)
    }

    fn read(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        read_image_data(py, &header_cards, super_.offsets.data_offset(), &super_.file)
    }

    // Grow the HDU's slow axis (numpy axis 0 = FITS NAXISn) if needed to
    // fit the data being written, then write it.  See CLAUDE.md for the
    // last-HDU constraint and the multi-phase write order.
    #[pyo3(signature = (data, start=None))]
    fn extend(
        slf: PyRefMut<'_, Self>,
        data: &Bound<'_, PyAny>,
        start: Option<Vec<i64>>,
    ) -> PyResult<()> {
        let super_: PyRefMut<HDU> = slf.into_super();

        let header_snapshot = super_.header_snapshot()?;
        let (bitpix, current_hdu_shape) = parse_image_hdu_shape(&header_snapshot)?;
        let naxis = current_hdu_shape.len();

        let dtype = data.getattr("dtype")?;
        let kind: String = dtype.getattr("kind")?.extract()?;
        let itemsize_attr: u64 = dtype.getattr("itemsize")?.extract()?;
        let (expected_kind, expected_size) = bitpix_to_numpy_kind(bitpix)?;
        if kind != expected_kind || itemsize_attr != expected_size {
            return Err(PyValueError::new_err(format!(
                "data dtype ({}{}) does not match HDU BITPIX={} (expected {}{})",
                kind, itemsize_attr, bitpix, expected_kind, expected_size,
            )));
        }

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

        {
            let mut guard = lock_file(&super_.file)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;

            let file_len = file.metadata()
                .map_err(|e| PyIOError::new_err(e.to_string()))?
                .len();
            if file_len != current_hdu_end {
                return Err(PyValueError::new_err(format!(
                    "cannot extend HDU #{}: file_size ({}) != HDU end ({}); \
                     extending non-last HDUs is not yet supported",
                    super_.index, file_len, current_hdu_end
                )));
            }

            if new_hdu_end > current_hdu_end {
                file.set_len(new_hdu_end)
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
        }

        let naxisn_key = format!("NAXIS{}", naxis);
        let new_card = card_int(
            &naxisn_key,
            new_hdu_shape[0] as i64,
            &format!("length of data axis {}", naxis),
        );
        let updated_header: Vec<String> = {
            let mut g = super_.header.lock()
                .map_err(|_| PyIOError::new_err("header lock poisoned"))?;
            let card_idx = g.iter()
                .position(|c| c.len() >= 8 && c[..8].trim() == naxisn_key)
                .ok_or_else(|| PyValueError::new_err(
                    format!("header missing {}", naxisn_key)
                ))?;
            g[card_idx] = new_card.trim_end().to_string();
            g.clone()
        };

        {
            let mut guard = lock_file(&super_.file)?;
            let file = guard.as_mut()
                .ok_or_else(|| PyIOError::new_err("file is closed"))?;

            let header_bytes = serialize_header_to_disk_bytes(&updated_header);
            let header_offset = data_offset - header_bytes.len() as u64;
            file.seek(SeekFrom::Start(header_offset))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            file.write_all(&header_bytes)
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            file.flush()
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }

        write_image_data(
            &updated_header,
            data_offset,
            &super_.file,
            data,
            Some(start_for_write),
        )
    }

    fn __getitem__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let super_: PyRef<HDU> = slf.into_super();
        let header_cards = super_.header_snapshot()?;
        let (_bitpix, hdu_shape) = parse_image_hdu_shape(&header_cards)?;
        let slices = normalize_slice_key(key, &hdu_shape)?;
        read_image_slice(py, &header_cards, super_.offsets.data_offset(), &super_.file, &slices)
    }
}

// Allocate a numpy array sized + typed to this HDU and fill it with the
// HDU's data from disk.  Result is native-endian; FITS big-endian bytes
// are swapped in place after read.
fn read_image_data(
    py: Python<'_>,
    header: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
) -> PyResult<Py<PyAny>> {
    let (bitpix, hdu_shape) = parse_image_hdu_shape(header)?;
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

    Ok(arr.unbind())
}

// ===== Slicing for image reads =====

#[derive(Debug, Clone)]
struct AxisSlice {
    start: u64,
    step: u64,
    count: u64,
    is_int: bool,
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

fn normalize_slice_key(
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

    Ok(arr.unbind())
}

fn write_image_data(
    header: &[String],
    data_offset: u64,
    file_handle: &FileHandle,
    data: &Bound<'_, PyAny>,
    start: Option<Vec<i64>>,
) -> PyResult<()> {
    let (bitpix, hdu_shape) = parse_image_hdu_shape(header)?;
    let naxis = hdu_shape.len();

    let dtype = data.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize_attr: u64 = dtype.getattr("itemsize")?.extract()?;
    let (expected_kind, expected_size) = bitpix_to_numpy_kind(bitpix)?;
    if kind != expected_kind || itemsize_attr != expected_size {
        return Err(PyValueError::new_err(format!(
            "data dtype ({}{}) does not match HDU BITPIX={} (expected {}{})",
            kind, itemsize_attr, bitpix, expected_kind, expected_size,
        )));
    }

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

fn bitpix_to_native_dtype(bitpix: i32) -> PyResult<&'static str> {
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

fn round_up_to_block(n: u64) -> u64 {
    let block = BLOCK_SIZE as u64;
    ((n + block - 1) / block) * block
}

fn serialize_header_to_disk_bytes(header: &[String]) -> Vec<u8> {
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

fn byteswap_in_place(buf: &mut [u8], itemsize: usize) {
    if itemsize <= 1 {
        return;
    }
    for chunk in buf.chunks_exact_mut(itemsize) {
        chunk.reverse();
    }
}

// ===== RawBuffer: raw Py_buffer wrapper =====

struct RawBuffer {
    view: Box<pyo3::ffi::Py_buffer>,
}

impl RawBuffer {
    fn acquire_with_flags(obj: &Bound<'_, PyAny>, flags: std::os::raw::c_int) -> PyResult<Self> {
        let mut view: Box<pyo3::ffi::Py_buffer> =
            Box::new(unsafe { std::mem::zeroed() });
        let rc = unsafe {
            pyo3::ffi::PyObject_GetBuffer(
                obj.as_ptr(),
                &mut *view as *mut _,
                flags,
            )
        };
        if rc != 0 {
            return Err(PyErr::take(obj.py()).unwrap_or_else(|| {
                PyValueError::new_err("buffer acquisition failed")
            }));
        }
        Ok(RawBuffer { view })
    }

    fn acquire(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::acquire_with_flags(obj, pyo3::ffi::PyBUF_C_CONTIGUOUS)
    }

    fn acquire_writable(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::acquire_with_flags(
            obj,
            pyo3::ffi::PyBUF_C_CONTIGUOUS | pyo3::ffi::PyBUF_WRITABLE,
        )
    }

    fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.view.buf as *const u8,
                self.view.len as usize,
            )
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.view.buf as *mut u8,
                self.view.len as usize,
            )
        }
    }

    fn itemsize(&self) -> usize {
        self.view.itemsize as usize
    }

    fn len(&self) -> usize {
        self.view.len as usize
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        unsafe { pyo3::ffi::PyBuffer_Release(&mut *self.view) };
    }
}

// Map a numpy short-code or long-name dtype string to a FITS BITPIX value.
// Lives next to its inverses (bitpix_to_numpy_kind / bitpix_to_native_dtype)
// so the supported-dtype set is maintained in one place.
pub(crate) fn dtype_to_bitpix(dtype: &str) -> PyResult<i32> {
    let s = dtype.trim_start_matches(|c| c == '<' || c == '>' || c == '|' || c == '=');
    let normalized = s.to_lowercase();
    match normalized.as_str() {
        "u1" | "uint8" => Ok(8),
        "i2" | "int16" => Ok(16),
        "i4" | "int32" => Ok(32),
        "i8" | "int64" => Ok(64),
        "f4" | "float32" => Ok(-32),
        "f8" | "float64" => Ok(-64),
        _ => Err(PyValueError::new_err(format!(
            "unsupported numpy dtype '{}'. Supported: 'u1','i2','i4','i8','f4','f8'",
            dtype
        ))),
    }
}
