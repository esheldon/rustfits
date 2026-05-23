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

// Validate GZIP compression level: must be None or 0..=9.
fn validate_gzip_level(level: Option<u32>) -> PyResult<Option<u32>> {
    match level {
        None => Ok(None),
        Some(v) if v <= 9 => Ok(Some(v)),
        Some(v) => Err(PyValueError::new_err(format!(
            "level must be in 0..=9 (0=none, 1=fastest, 9=best), \
             or None for the codec default (6).  Got {}.",
            v
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
#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gzip1 {
    pub(crate) tile_shape: Option<Vec<u64>>,
    pub(crate) heap_format: char,
    pub(crate) level: Option<u32>,
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
    /// level : int in 0..=9, optional
    ///     zlib compression level for the GZIP_1 stream.  0 means
    ///     no compression (just gzip framing), 1 is fastest with
    ///     least compression, 9 is slowest with most.  None (the
    ///     default) uses the codec default of 6 — the same as
    ///     cfitsio/zlib/astropy.  The level is a write-only
    ///     parameter; it isn't recorded in the file, so
    ///     `.compression` on a reopened HDU returns `level=None`
    ///     regardless of what was used to write the file.
    #[new]
    #[pyo3(signature = (
        *, tile_shape=None, heap_format=String::from("P"), level=None,
    ))]
    fn new(
        tile_shape: Option<Vec<i64>>,
        heap_format: String,
        level: Option<u32>,
    ) -> PyResult<Self> {
        let tile_shape = validate_tile_shape(tile_shape)?;
        let heap_format = validate_heap_format(&heap_format)?;
        let level = validate_gzip_level(level)?;
        Ok(Gzip1 { tile_shape, heap_format, level })
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
    fn level(&self, py: Python<'_>) -> Py<PyAny> {
        match self.level {
            None => py.None(),
            Some(v) => v.into_pyobject(py).unwrap().unbind().into_any(),
        }
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
        let lvl = match self.level {
            None => "None".to_string(),
            Some(v) => v.to_string(),
        };
        format!(
            "Gzip1(tile_shape={}, heap_format='{}', level={})",
            ts, self.heap_format, lvl,
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
#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Gzip2 {
    pub(crate) tile_shape: Option<Vec<u64>>,
    pub(crate) heap_format: char,
    pub(crate) level: Option<u32>,
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
    /// level : int in 0..=9, optional
    ///     zlib compression level for the GZIP_2 stream.  Same
    ///     semantics as `Gzip1(level=...)` — None uses the
    ///     codec default (6).  Write-only; not recoverable from
    ///     the file.
    #[new]
    #[pyo3(signature = (
        *, tile_shape=None, heap_format=String::from("P"), level=None,
    ))]
    fn new(
        tile_shape: Option<Vec<i64>>,
        heap_format: String,
        level: Option<u32>,
    ) -> PyResult<Self> {
        let tile_shape = validate_tile_shape(tile_shape)?;
        let heap_format = validate_heap_format(&heap_format)?;
        let level = validate_gzip_level(level)?;
        Ok(Gzip2 { tile_shape, heap_format, level })
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
    fn level(&self, py: Python<'_>) -> Py<PyAny> {
        match self.level {
            None => py.None(),
            Some(v) => v.into_pyobject(py).unwrap().unbind().into_any(),
        }
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
        let lvl = match self.level {
            None => "None".to_string(),
            Some(v) => v.to_string(),
        };
        format!(
            "Gzip2(tile_shape={}, heap_format='{}', level={})",
            ts, self.heap_format, lvl,
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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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

// ---------- Quantize ----------

/// Quantization parameters for float-image compression.
///
/// FITS Tile Compression of floating-point images works by
/// quantizing each tile to i32, then compressing the quantized
/// values with one of the integer compression algorithms
/// (`Rice1`, `Gzip1`, `Hcompress1`, etc.).  This pyclass carries
/// the quantization parameters separately from the algorithm
/// config:
///
///     fits.create_image_hdu(
///         "f4", shape,
///         compress=Rice1(tile_shape=(100, 100)),
///         quantize=Quantize(level=4.0, method="dither1"),
///     )
///
/// The orthogonal split mirrors cfitsio's separate
/// `fits_set_quantize_level` / `fits_set_quantize_method` calls.
/// Integer HDUs never see `quantize=`; float HDUs use a default
/// `Quantize` when the argument is omitted.
///
/// Parameters
/// ----------
/// level : float, default 4.0
///     Quantization level.  Positive values mean "N sigma per
///     quanta": the per-tile bscale is set to `tile_noise / level`
///     so each quantization step covers `1/level` of the noise.
///     Larger `level` = finer steps = higher fidelity + larger
///     compressed output.  cfitsio's default is 4.0 (with
///     dithering) or 16.0 (without).  Negative values mean
///     "fixed bscale = -level" (skip noise estimation).
/// method : str, default 'dither1'
///     Dithering scheme: 'no_dither' (NO_DITHER: simple round-to-
///     nearest, no dither), 'dither1' (SUBTRACTIVE_DITHER_1: add
///     a pseudorandom offset before rounding so quantization noise
///     is white), 'dither2' (SUBTRACTIVE_DITHER_2: like dither1
///     but reserves a sentinel for NaN so float-NaN survives the
///     round-trip).  cfitsio's default is SUBTRACTIVE_DITHER_1.
///     To skip quantization entirely (lossless raw float GZIP),
///     pass `quantize=None` (or omit the kwarg) to
///     `create_image_hdu` — do NOT construct a `Quantize` at all.
/// seed : int, default 0
///     ZDITHER0 random seed, recorded as a header card.  Zero
///     means "pick one automatically" at create time (cfitsio
///     uses a checksum of the data).  Positive values are passed
///     through verbatim — useful when reproducible output across
///     runs matters.
#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Quantize {
    pub(crate) level: f64,
    pub(crate) method: QuantizeMethod,
    pub(crate) seed: i64,
}

// Method enum mirrors the FITS Tile Compression Convention
// ZQUANTIZ values.  "No quantization" is signalled at the
// `create_image_hdu` call site by passing `quantize=None`
// (or omitting the kwarg) — there is no Quantize variant for
// that case, since constructing a Quantize implies the user
// wants quantization with some dither method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantizeMethod {
    NoDither,           // ZQUANTIZ='NO_DITHER'
    SubtractiveDither1, // ZQUANTIZ='SUBTRACTIVE_DITHER_1'
    SubtractiveDither2, // ZQUANTIZ='SUBTRACTIVE_DITHER_2'
}

impl QuantizeMethod {
    pub(crate) fn parse(s: &str) -> PyResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "no_dither" | "nodither" => Ok(Self::NoDither),
            "dither1" | "dither_1" | "subtractive_dither_1" => {
                Ok(Self::SubtractiveDither1)
            }
            "dither2" | "dither_2" | "subtractive_dither_2" => {
                Ok(Self::SubtractiveDither2)
            }
            other => Err(PyValueError::new_err(format!(
                "Quantize: unknown method '{}'; expected one of \
                 'no_dither', 'dither1', 'dither2'.  For \
                 unquantized lossless storage, pass quantize=None \
                 (or omit the kwarg) instead of constructing a \
                 Quantize.",
                other
            ))),
        }
    }

    /// Canonical FITS-spec ZQUANTIZ string for this method.
    pub(crate) fn zquantiz(&self) -> &'static str {
        match self {
            Self::NoDither => "NO_DITHER",
            Self::SubtractiveDither1 => "SUBTRACTIVE_DITHER_1",
            Self::SubtractiveDither2 => "SUBTRACTIVE_DITHER_2",
        }
    }

    /// Pythonic short name (matches the kwarg value).
    pub(crate) fn short_name(&self) -> &'static str {
        match self {
            Self::NoDither => "no_dither",
            Self::SubtractiveDither1 => "dither1",
            Self::SubtractiveDither2 => "dither2",
        }
    }
}

#[pymethods]
impl Quantize {
    #[new]
    #[pyo3(signature = (
        *, level=4.0, method=String::from("dither1"), seed=0,
    ))]
    fn new(level: f64, method: String, seed: i64) -> PyResult<Self> {
        let method = QuantizeMethod::parse(&method)?;
        if !level.is_finite() {
            return Err(PyValueError::new_err(format!(
                "Quantize: level must be finite, got {}", level
            )));
        }
        Ok(Quantize { level, method, seed })
    }

    #[getter]
    fn level(&self) -> f64 {
        self.level
    }

    #[getter]
    fn method(&self) -> &'static str {
        self.method.short_name()
    }

    #[getter]
    fn seed(&self) -> i64 {
        self.seed
    }

    /// FITS-spec ZQUANTIZ string this Quantize would emit on
    /// write — exposed for symmetry with the algorithm-config
    /// classes' `zcmptype`.
    #[getter]
    fn zquantiz(&self) -> &'static str {
        self.method.zquantiz()
    }

    fn __repr__(&self) -> String {
        format!(
            "Quantize(level={}, method='{}', seed={})",
            self.level, self.method.short_name(), self.seed,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

// ---------- CompressionConfigKind ----------

// Internal wrapper over the per-algorithm config pyclasses.  The
// `compress=` argument to `create_image_hdu` may be any of
// `Gzip1` / `Gzip2` / `Rice1` / `Hcompress1` / `Plio1`; extracting
// directly to one specific type would force a separate isinstance
// branch per algorithm at the call site, so this enum centralises
// the "try each known class in turn" logic and exposes the small
// set of shared accessors.
//
// Stored on `CompressedImageHDU::compress_config` after create so
// that write-only parameters (`Gzip1(level=...)`,
// `Gzip2(level=...)`, future additions) survive across
// write/extend/__setitem__ calls AND so the `.compression`
// accessor returns the SAME object the user passed in for
// freshly-created HDUs.  For reopened HDUs the field is None and
// callers fall back to reconstructing from header cards via
// `build_compression_config`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CompressionConfigKind {
    Gzip1(Gzip1),
    Gzip2(Gzip2),
    Rice1(Rice1),
    Hcompress1(Hcompress1),
    Plio1(Plio1),
}

impl CompressionConfigKind {
    pub(crate) fn from_pyany(bound: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(g) = bound.extract::<Gzip1>() {
            return Ok(Self::Gzip1(g));
        }
        if let Ok(g) = bound.extract::<Gzip2>() {
            return Ok(Self::Gzip2(g));
        }
        if let Ok(r) = bound.extract::<Rice1>() {
            return Ok(Self::Rice1(r));
        }
        if let Ok(h) = bound.extract::<Hcompress1>() {
            return Ok(Self::Hcompress1(h));
        }
        if let Ok(p) = bound.extract::<Plio1>() {
            return Ok(Self::Plio1(p));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "compress= must be a compression-config object (e.g. \
             rustfits.Gzip1(...), rustfits.Gzip2(...), \
             rustfits.Rice1(...), rustfits.Hcompress1(...), \
             rustfits.Plio1(...))",
        ))
    }

    pub(crate) fn tile_shape(&self) -> &Option<Vec<u64>> {
        match self {
            Self::Gzip1(g) => &g.tile_shape,
            Self::Gzip2(g) => &g.tile_shape,
            Self::Rice1(r) => &r.tile_shape,
            Self::Hcompress1(h) => &h.tile_shape,
            Self::Plio1(p) => &p.tile_shape,
        }
    }

    pub(crate) fn heap_format(&self) -> char {
        match self {
            Self::Gzip1(g) => g.heap_format,
            Self::Gzip2(g) => g.heap_format,
            Self::Rice1(r) => r.heap_format,
            Self::Hcompress1(h) => h.heap_format,
            Self::Plio1(p) => p.heap_format,
        }
    }

    pub(crate) fn zcmptype(&self) -> &'static str {
        match self {
            Self::Gzip1(_) => "GZIP_1",
            Self::Gzip2(_) => "GZIP_2",
            Self::Rice1(_) => "RICE_1",
            Self::Hcompress1(_) => "HCOMPRESS_1",
            Self::Plio1(_) => "PLIO_1",
        }
    }

    // Algorithm-specific (ZNAMEn, ZVALn) pairs to emit alongside
    // the standard ZIMAGE header cards.  RICE_1 carries BLOCKSIZE
    // + BYTEPIX; HCOMPRESS_1 carries SCALE + SMOOTH; GZIP variants
    // have no extras (level is write-only and not emitted).
    pub(crate) fn extra_z_cards(
        &self, bitpix: i32,
    ) -> Vec<(&'static str, i64)> {
        match self {
            Self::Gzip1(_) | Self::Gzip2(_) | Self::Plio1(_) => Vec::new(),
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

    // GZIP compression level for Gzip1 / Gzip2; None for other
    // algorithms (they don't use a zlib level).  Used by the write
    // path to thread the user's level through to the encoder.
    pub(crate) fn gzip_level(&self) -> Option<u32> {
        match self {
            Self::Gzip1(g) => g.level,
            Self::Gzip2(g) => g.level,
            _ => None,
        }
    }

    // Replace the wrapped algorithm config's tile_shape with the
    // resolved value (the actual on-disk tile shape, computed by
    // `create_compressed_image_hdu_impl` from the user's input or
    // the algorithm-specific default).  Stored on the HDU so that
    // `.compression.tile_shape` returns the real tile shape even
    // when the user passed `Gzip1()` etc. without specifying it.
    pub(crate) fn with_resolved_tile_shape(self, ts: Vec<u64>) -> Self {
        match self {
            Self::Gzip1(mut g) => { g.tile_shape = Some(ts); Self::Gzip1(g) }
            Self::Gzip2(mut g) => { g.tile_shape = Some(ts); Self::Gzip2(g) }
            Self::Rice1(mut r) => { r.tile_shape = Some(ts); Self::Rice1(r) }
            Self::Hcompress1(mut h) => {
                h.tile_shape = Some(ts); Self::Hcompress1(h)
            }
            Self::Plio1(mut p) => { p.tile_shape = Some(ts); Self::Plio1(p) }
        }
    }
}
