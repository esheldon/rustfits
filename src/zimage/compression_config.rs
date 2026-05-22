// Compression-config pyclasses passed to `create_image_hdu(...,
// compress=...)`.  Each algorithm has its own pyclass so the
// constructor's kwargs are exactly the parameters the algorithm
// accepts — no kwarg bloat, no conditionally-meaningful params,
// and the Python IDE / `help(Gzip1)` shows the right surface.
//
// Common kwargs (`tile_shape`, `heap_format`) are duplicated
// across each constructor rather than inherited.  Five classes
// × two shared kwargs = trivial duplication; inheritance would
// require a PyClassInitializer chain that's harder to read.
//
// Validation runs eagerly in `#[new]` — config objects are
// immutable (no setters) so once constructed, the object is
// known-good.  `tile_shape` content is range-checked here; the
// "tile shape dimensionality matches the image" check is
// deferred to `create_image_hdu` since we don't know the image
// shape at construct time.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// ---------- helpers shared by every config constructor ----------

// Validate a tile_shape tuple/list: each entry must be a positive
// int.  Returns the parsed shape as Vec<u64>.  None means "use the
// FITS-convention default (row tiles)" — that decision happens at
// HDU creation time when the image shape is known.
fn validate_tile_shape(
    tile_shape: Option<Vec<i64>>,
) -> PyResult<Option<Vec<u64>>> {
    match tile_shape {
        None => Ok(None),
        Some(v) => {
            if v.is_empty() {
                return Err(PyValueError::new_err(
                    "tile_shape must be a non-empty sequence of positive \
                     integers (got an empty tuple)",
                ));
            }
            let mut out = Vec::with_capacity(v.len());
            for (i, x) in v.iter().enumerate() {
                if *x <= 0 {
                    return Err(PyValueError::new_err(format!(
                        "tile_shape[{}] = {} (must be positive)", i, x
                    )));
                }
                out.push(*x as u64);
            }
            Ok(Some(out))
        }
    }
}

// Validate heap_format: must be 'P' or 'Q' (case-insensitive).
// Returns the canonical uppercase form.
fn validate_heap_format(heap_format: &str) -> PyResult<char> {
    match heap_format {
        "P" | "p" => Ok('P'),
        "Q" | "q" => Ok('Q'),
        other => Err(PyValueError::new_err(format!(
            "heap_format must be 'P' or 'Q', got '{}'", other
        ))),
    }
}

// ---------- Gzip1 ----------

/// Configuration for GZIP_1 tile compression.
///
/// Stores each tile as a single gzip-framed stream over the
/// pixel bytes in FITS big-endian order.  No quantization, no
/// preprocessing — the simplest of the FITS Tile Compression
/// algorithms.  Pairs well with mostly-uniform integer data;
/// for noisy data RICE_1 typically compresses tighter.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gzip1 {
    pub(crate) tile_shape: Option<Vec<u64>>,
    pub(crate) heap_format: char,
}

#[pymethods]
impl Gzip1 {
    /// Build a Gzip1 compression config.
    ///
    /// Parameters
    /// ----------
    /// tile_shape : tuple of positive ints, optional
    ///     Tile dimensions in numpy axis order (slowest first).
    ///     When omitted, defaults to the FITS-convention "row
    ///     tiles" layout (ZTILE1 = NAXIS1, others = 1) once the
    ///     image shape is known.
    /// heap_format : {'P', 'Q'}, default 'P'
    ///     Heap addressing format.  'P' uses 8-byte descriptors
    ///     with 32-bit nelements/offset (4 GB heap ceiling).
    ///     'Q' uses 16-byte descriptors (no practical ceiling)
    ///     for files whose compressed heap exceeds 4 GB.
    #[new]
    #[pyo3(signature = (*, tile_shape=None, heap_format=String::from("P")))]
    fn new(
        tile_shape: Option<Vec<i64>>,
        heap_format: String,
    ) -> PyResult<Self> {
        let tile_shape = validate_tile_shape(tile_shape)?;
        let heap_format = validate_heap_format(&heap_format)?;
        Ok(Gzip1 { tile_shape, heap_format })
    }

    #[getter]
    fn tile_shape(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.tile_shape {
            None => py.None(),
            Some(v) => pyo3::types::PyTuple::new(py, v)
                .expect("PyTuple::new of u64 always succeeds")
                .unbind()
                .into_any(),
        }
    }

    #[getter]
    fn heap_format(&self) -> String {
        self.heap_format.to_string()
    }

    /// FITS-spec ZCMPTYPE string for this algorithm.
    #[getter]
    fn zcmptype(&self) -> &'static str {
        "GZIP_1"
    }

    fn __repr__(&self) -> String {
        let ts = match &self.tile_shape {
            None => "None".to_string(),
            Some(v) => format!("{:?}", v),
        };
        format!(
            "Gzip1(tile_shape={}, heap_format='{}')",
            ts, self.heap_format
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

// ---------- Gzip2 ----------

/// Configuration for GZIP_2 tile compression.
///
/// Like GZIP_1 but with a byte-shuffle preprocessor applied
/// before compression: bytes are reordered so that all
/// most-significant bytes of every pixel come first, then all
/// second-most-significant bytes, and so on down to the
/// least-significant.  The shuffle decorrelates the byte
/// streams, producing longer runs of similar values that
/// deflate compresses tighter than the interleaved layout of
/// GZIP_1.  For 1-byte data the shuffle is a no-op, so GZIP_2
/// and GZIP_1 produce identical output on `u1` images.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gzip2 {
    pub(crate) tile_shape: Option<Vec<u64>>,
    pub(crate) heap_format: char,
}

#[pymethods]
impl Gzip2 {
    /// Build a Gzip2 compression config.
    ///
    /// Parameters
    /// ----------
    /// tile_shape : tuple of positive ints, optional
    ///     Tile dimensions in numpy axis order (slowest first).
    ///     When omitted, defaults to the FITS-convention "row
    ///     tiles" layout (ZTILE1 = NAXIS1, others = 1) once the
    ///     image shape is known.
    /// heap_format : {'P', 'Q'}, default 'P'
    ///     Heap addressing format.  'P' uses 8-byte descriptors
    ///     with 32-bit nelements/offset (4 GB heap ceiling).
    ///     'Q' uses 16-byte descriptors (no practical ceiling)
    ///     for files whose compressed heap exceeds 4 GB.
    #[new]
    #[pyo3(signature = (*, tile_shape=None, heap_format=String::from("P")))]
    fn new(
        tile_shape: Option<Vec<i64>>,
        heap_format: String,
    ) -> PyResult<Self> {
        let tile_shape = validate_tile_shape(tile_shape)?;
        let heap_format = validate_heap_format(&heap_format)?;
        Ok(Gzip2 { tile_shape, heap_format })
    }

    #[getter]
    fn tile_shape(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.tile_shape {
            None => py.None(),
            Some(v) => pyo3::types::PyTuple::new(py, v)
                .expect("PyTuple::new of u64 always succeeds")
                .unbind()
                .into_any(),
        }
    }

    #[getter]
    fn heap_format(&self) -> String {
        self.heap_format.to_string()
    }

    /// FITS-spec ZCMPTYPE string for this algorithm.
    #[getter]
    fn zcmptype(&self) -> &'static str {
        "GZIP_2"
    }

    fn __repr__(&self) -> String {
        let ts = match &self.tile_shape {
            None => "None".to_string(),
            Some(v) => format!("{:?}", v),
        };
        format!(
            "Gzip2(tile_shape={}, heap_format='{}')",
            ts, self.heap_format
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

// ---------- Plio1 ----------

/// Configuration for PLIO_1 tile compression (IRAF Pixel List
/// encoding).  RLE for non-negative integer masks built via
/// increments; output is i16 BE shorts (TFORM1=`1PI`).  Supports
/// integer ZBITPIX = 8 / 16 / 32; not meaningful for floats.
///
/// Read side: full support — `hdu.compression` returns this
/// object when ZCMPTYPE=PLIO_1.  Write side: not yet implemented
/// (the encoder is a Phase 7 follow-up).  Until then, passing
/// `compress=Plio1(...)` to `create_image_hdu` raises
/// `NotImplementedError`.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Plio1 {
    pub(crate) tile_shape: Option<Vec<u64>>,
    pub(crate) heap_format: char,
}

#[pymethods]
impl Plio1 {
    /// Build a Plio1 compression config.
    ///
    /// Parameters
    /// ----------
    /// tile_shape : tuple of positive ints, optional
    ///     Tile dimensions in numpy axis order (slowest first).
    ///     When omitted, defaults to the FITS-convention "row
    ///     tiles" layout (ZTILE1 = NAXIS1, others = 1) once the
    ///     image shape is known.
    /// heap_format : {'P', 'Q'}, default 'P'
    ///     Heap addressing format.
    #[new]
    #[pyo3(signature = (*, tile_shape=None, heap_format=String::from("P")))]
    fn new(
        tile_shape: Option<Vec<i64>>,
        heap_format: String,
    ) -> PyResult<Self> {
        let tile_shape = validate_tile_shape(tile_shape)?;
        let heap_format = validate_heap_format(&heap_format)?;
        Ok(Plio1 { tile_shape, heap_format })
    }

    #[getter]
    fn tile_shape(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.tile_shape {
            None => py.None(),
            Some(v) => pyo3::types::PyTuple::new(py, v)
                .expect("PyTuple::new of u64 always succeeds")
                .unbind()
                .into_any(),
        }
    }

    #[getter]
    fn heap_format(&self) -> String {
        self.heap_format.to_string()
    }

    /// FITS-spec ZCMPTYPE string for this algorithm.
    #[getter]
    fn zcmptype(&self) -> &'static str {
        "PLIO_1"
    }

    fn __repr__(&self) -> String {
        let ts = match &self.tile_shape {
            None => "None".to_string(),
            Some(v) => format!("{:?}", v),
        };
        format!(
            "Plio1(tile_shape={}, heap_format='{}')",
            ts, self.heap_format
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

// ---------- Hcompress1 ----------

/// Configuration for HCOMPRESS_1 tile compression.
///
/// 2-D wavelet-like H-transform with quadtree bit-plane encoding.
/// Designed for astronomical images where the smoothing pass can
/// substantially reduce visible blockiness at higher compression
/// ratios.  Supports BYTEPIX ∈ {1, 2, 4} (u1, i2, i4 in numpy
/// terms); BYTEPIX=8 (i8) is rejected — the FITS Tile Compression
/// Convention has no 64-bit HCOMPRESS variant and cfitsio's encoder
/// family doesn't produce one.
///
/// Two algorithm parameters: `scale` (quantization level — 0 or 1
/// for lossless, larger for more compression at the cost of
/// precision) and `smooth` (post-decompression smoothing pass that
/// reduces block artifacts — has no effect at `scale <= 1` since
/// nothing is quantized).  Tiles must be 2-D; 1-D and 3-D images
/// are rejected at create time.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Hcompress1 {
    pub(crate) tile_shape: Option<Vec<u64>>,
    pub(crate) heap_format: char,
    pub(crate) scale: i32,
    pub(crate) smooth: bool,
}

#[pymethods]
impl Hcompress1 {
    /// Build an Hcompress1 compression config.
    ///
    /// Parameters
    /// ----------
    /// tile_shape : tuple of positive ints, optional
    ///     Tile dimensions in numpy axis order (slowest first).
    ///     Must be 2-D when supplied — HCOMPRESS_1 is a 2-D
    ///     algorithm.  When omitted, defaults to the FITS-
    ///     convention "row tiles" layout (ZTILE1 = NAXIS1,
    ///     others = 1) once the image shape is known.
    /// heap_format : {'P', 'Q'}, default 'P'
    ///     Heap addressing format.  'P' uses 8-byte descriptors
    ///     with 32-bit nelements/offset (4 GB heap ceiling).
    ///     'Q' uses 16-byte descriptors (no practical ceiling)
    ///     for files whose compressed heap exceeds 4 GB.
    /// scale : int, default 0
    ///     Quantization scale.  0 or 1 = lossless.  Larger values
    ///     divide each H-transform coefficient by `scale`, giving
    ///     more compression but lower precision.  Must be >= 0.
    /// smooth : bool, default False
    ///     Enable the smoothing pass during inverse H-transform on
    ///     read; reduces visible blockiness at higher scales but
    ///     adds CPU work on each tile decode.  No effect at
    ///     scale <= 1.
    #[new]
    #[pyo3(signature = (
        *, tile_shape=None, heap_format=String::from("P"),
        scale=0, smooth=false,
    ))]
    fn new(
        tile_shape: Option<Vec<i64>>,
        heap_format: String,
        scale: i64,
        smooth: bool,
    ) -> PyResult<Self> {
        let tile_shape = validate_tile_shape(tile_shape)?;
        if let Some(ref ts) = tile_shape {
            if ts.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "Hcompress1 only supports 2-D tiles; got tile_shape \
                     with {} dimensions", ts.len()
                )));
            }
        }
        let heap_format = validate_heap_format(&heap_format)?;
        if scale < 0 {
            return Err(PyValueError::new_err(format!(
                "scale must be >= 0, got {}", scale
            )));
        }
        if scale > i32::MAX as i64 {
            return Err(PyValueError::new_err(format!(
                "scale {} exceeds i32::MAX", scale
            )));
        }
        Ok(Hcompress1 {
            tile_shape,
            heap_format,
            scale: scale as i32,
            smooth,
        })
    }

    #[getter]
    fn tile_shape(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.tile_shape {
            None => py.None(),
            Some(v) => pyo3::types::PyTuple::new(py, v)
                .expect("PyTuple::new of u64 always succeeds")
                .unbind()
                .into_any(),
        }
    }

    #[getter]
    fn heap_format(&self) -> String {
        self.heap_format.to_string()
    }

    #[getter]
    fn scale(&self) -> i32 {
        self.scale
    }

    #[getter]
    fn smooth(&self) -> bool {
        self.smooth
    }

    /// FITS-spec ZCMPTYPE string for this algorithm.
    #[getter]
    fn zcmptype(&self) -> &'static str {
        "HCOMPRESS_1"
    }

    fn __repr__(&self) -> String {
        let ts = match &self.tile_shape {
            None => "None".to_string(),
            Some(v) => format!("{:?}", v),
        };
        format!(
            "Hcompress1(tile_shape={}, heap_format='{}', scale={}, smooth={})",
            ts, self.heap_format, self.scale,
            if self.smooth { "True" } else { "False" },
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

// ---------- Rice1 ----------

/// Configuration for RICE_1 tile compression.
///
/// Rice coding over per-pixel deltas: for each block of `blocksize`
/// pixels the encoder picks a split parameter k from a per-block
/// entropy heuristic, encodes the high bits unary and the low k
/// bits raw.  Strong on smooth-but-noisy integer imagery; cfitsio
/// uses this as the default compression for many surveys.
///
/// Supports BYTEPIX ∈ {1, 2, 4} (u1, i2, i4 in numpy terms).
/// BYTEPIX=8 (i64) is rejected: no canonical FITS writer produces
/// such files (cfitsio refuses, astropy silently downcasts to
/// i32), so they would be unreadable outside rustfits.  Use
/// Gzip2 for i64 data — within ~5% of RICE compression on real
/// imagery and universally readable.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Rice1 {
    pub(crate) tile_shape: Option<Vec<u64>>,
    pub(crate) heap_format: char,
    pub(crate) blocksize: u32,
}

#[pymethods]
impl Rice1 {
    /// Build a Rice1 compression config.
    ///
    /// Parameters
    /// ----------
    /// tile_shape : tuple of positive ints, optional
    ///     Tile dimensions in numpy axis order (slowest first).
    ///     When omitted, defaults to the FITS-convention "row
    ///     tiles" layout (ZTILE1 = NAXIS1, others = 1) once the
    ///     image shape is known.
    /// heap_format : {'P', 'Q'}, default 'P'
    ///     Heap addressing format.  'P' uses 8-byte descriptors
    ///     with 32-bit nelements/offset (4 GB heap ceiling).
    ///     'Q' uses 16-byte descriptors (no practical ceiling).
    /// blocksize : int, default 32
    ///     Number of pixels per Rice coding block.  Larger blocks
    ///     amortise the per-block FS overhead but adapt more
    ///     slowly to changes in local entropy.  cfitsio uses 32
    ///     as its default and exposes no knob; some encoders use
    ///     16 or 64.  Must be > 0.
    #[new]
    #[pyo3(signature = (
        *, tile_shape=None, heap_format=String::from("P"), blocksize=32
    ))]
    fn new(
        tile_shape: Option<Vec<i64>>,
        heap_format: String,
        blocksize: i64,
    ) -> PyResult<Self> {
        let tile_shape = validate_tile_shape(tile_shape)?;
        let heap_format = validate_heap_format(&heap_format)?;
        if blocksize <= 0 {
            return Err(PyValueError::new_err(format!(
                "blocksize must be > 0, got {}", blocksize
            )));
        }
        if blocksize > u32::MAX as i64 {
            return Err(PyValueError::new_err(format!(
                "blocksize {} exceeds u32::MAX", blocksize
            )));
        }
        Ok(Rice1 {
            tile_shape,
            heap_format,
            blocksize: blocksize as u32,
        })
    }

    #[getter]
    fn tile_shape(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.tile_shape {
            None => py.None(),
            Some(v) => pyo3::types::PyTuple::new(py, v)
                .expect("PyTuple::new of u64 always succeeds")
                .unbind()
                .into_any(),
        }
    }

    #[getter]
    fn heap_format(&self) -> String {
        self.heap_format.to_string()
    }

    #[getter]
    fn blocksize(&self) -> u32 {
        self.blocksize
    }

    /// FITS-spec ZCMPTYPE string for this algorithm.
    #[getter]
    fn zcmptype(&self) -> &'static str {
        "RICE_1"
    }

    fn __repr__(&self) -> String {
        let ts = match &self.tile_shape {
            None => "None".to_string(),
            Some(v) => format!("{:?}", v),
        };
        format!(
            "Rice1(tile_shape={}, heap_format='{}', blocksize={})",
            ts, self.heap_format, self.blocksize,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}
