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
#[derive(Clone, Debug)]
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
}
