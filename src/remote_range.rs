// Ranged (partial) remote reads over HTTP(S) — the
// `Storage::HttpRange` backend — plus the `Remote` transport-config
// pyclass accepted by `FITS(..., remote=...)`.
//
// Instead of downloading the whole file at open (flavor #1 in the
// remote roadmap, `download_remote` in fits.rs), a ranged open probes
// the server once to learn the file length, then serves every
// subsequent `read` from a bytes-bounded LRU of fixed-size blocks
// fetched on demand with HTTP `Range` requests.  The block cache is
// the load-bearing piece: the parse and read paths do many tiny reads
// (2880-byte header blocks, 8/16-byte heap descriptors), and a naive
// request-per-read would pay one WAN round trip each.  Missing blocks
// are fetched in contiguous runs, so a large strip read costs ONE
// request regardless of how many blocks it spans.
//
// Read-only by construction: `FITS::new` rejects `mode != "r"` for
// every remote URL before any network I/O, and the `Storage` write
// arms error defensively.  Nothing here can taint a file — there is
// no local state to diverge.

use crate::cache::BytesBoundLruCache;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Arc;
use std::time::Duration;

// Fetch granularity.  1 MiB matches the codebase's streaming-chunk
// convention; at typical WAN round-trip times anything in the
// 256 KiB - 1 MiB range is RTT-dominated, so the convention value is
// fine.  User-tunable via Remote(block_bytes=...) — smaller for
// scattered tile reads (less over-fetch), larger for long strips.
pub(crate) const DEFAULT_BLOCK_BYTES: u64 = 1 << 20;

// Block-LRU budget, matching the compressed-HDU tile-cache default.
// The tile cache sits ABOVE this cache (decoded tiles vs raw file
// bytes); both are bounded.
pub(crate) const DEFAULT_CACHE_BYTES: u64 = 32 << 20;

/// Transport configuration for opening a remote (URL) FITS file.
///
/// Pass to ``FITS(url, "r", remote=...)``.  With ``ranged=False``
/// (the default) the whole file is downloaded at open — today's
/// behavior — but ``headers=`` / ``timeout=`` still apply, so
/// token-authenticated archives work.  With ``ranged=True`` the file
/// is opened WITHOUT downloading it: reads fetch only the byte
/// ranges they touch via HTTP ``Range`` requests, cached in
/// fixed-size blocks.  The string shorthand ``remote="ranged"`` is
/// equivalent to ``Remote(ranged=True)``.
///
/// Parameters
/// ----------
/// ranged : bool, default False
///     Use HTTP Range requests instead of downloading the whole
///     file.  http/https only; requires the server to honor Range
///     (a server that ignores it raises at open — no silent
///     fallback).  Ranged files are read-only and ``to_bytes()`` is
///     rejected on them.
/// headers : dict of str -> str, optional
///     Extra HTTP headers sent with every request (both modes),
///     e.g. ``{"Authorization": "Bearer <token>"}``.
/// timeout : float, optional
///     Per-request timeout in seconds (both modes).  None (the
///     default) means no timeout.
/// block_bytes : int, optional
///     Ranged mode only: fetch granularity in bytes (default
///     1 MiB).  Smaller reduces over-fetch for scattered small
///     reads; larger amortizes round trips for long sequential
///     reads.
/// cache_bytes : int, optional
///     Ranged mode only: byte budget for the LRU of fetched blocks
///     (default 32 MiB).  0 disables caching (every read re-fetches
///     its blocks — correct but slow).
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct Remote {
    pub(crate) ranged: bool,
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) timeout: Option<f64>,
    pub(crate) block_bytes: u64,
    pub(crate) cache_bytes: u64,
}

#[pymethods]
impl Remote {
    #[new]
    #[pyo3(signature = (
        *, ranged=false, headers=None, timeout=None, block_bytes=None,
        cache_bytes=None,
    ))]
    fn new(
        ranged: bool,
        headers: Option<&Bound<'_, PyDict>>,
        timeout: Option<f64>,
        block_bytes: Option<u64>,
        cache_bytes: Option<u64>,
    ) -> PyResult<Self> {
        if !ranged && (block_bytes.is_some() || cache_bytes.is_some()) {
            return Err(PyValueError::new_err(
                "block_bytes= and cache_bytes= apply to ranged reads \
                 only; pass ranged=True to use them",
            ));
        }
        if let Some(t) = timeout {
            if !t.is_finite() || t <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "timeout must be a positive number of seconds, \
                     got {}", t
                )));
            }
        }
        if block_bytes == Some(0) {
            return Err(PyValueError::new_err(
                "block_bytes must be positive",
            ));
        }
        let headers = match headers {
            None => Vec::new(),
            Some(d) => {
                let mut out = Vec::with_capacity(d.len());
                for (k, v) in d.iter() {
                    let (Ok(k), Ok(v)) =
                        (k.extract::<String>(), v.extract::<String>())
                    else {
                        return Err(PyValueError::new_err(
                            "headers must be a dict of str -> str",
                        ));
                    };
                    out.push((k, v));
                }
                out
            }
        };
        Ok(Remote {
            ranged,
            headers,
            timeout,
            block_bytes: block_bytes.unwrap_or(DEFAULT_BLOCK_BYTES),
            cache_bytes: cache_bytes.unwrap_or(DEFAULT_CACHE_BYTES),
        })
    }

    /// Whether HTTP Range partial reads are enabled.
    #[getter]
    fn ranged(&self) -> bool {
        self.ranged
    }

    /// Extra HTTP headers as a dict, or ``None`` when none were
    /// given.
    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.headers.is_empty() {
            return Ok(py.None());
        }
        let d = PyDict::new(py);
        for (k, v) in &self.headers {
            d.set_item(k, v)?;
        }
        Ok(d.unbind().into_any())
    }

    /// Per-request timeout in seconds, or ``None`` for no timeout.
    #[getter]
    fn timeout(&self) -> Option<f64> {
        self.timeout
    }

    /// Ranged-mode fetch granularity in bytes.
    #[getter]
    fn block_bytes(&self) -> u64 {
        self.block_bytes
    }

    /// Ranged-mode block-LRU byte budget.
    #[getter]
    fn cache_bytes(&self) -> u64 {
        self.cache_bytes
    }

    fn __repr__(&self) -> String {
        let headers = if self.headers.is_empty() {
            "None".to_string()
        } else {
            format!("<{} entries>", self.headers.len())
        };
        let timeout = match self.timeout {
            None => "None".to_string(),
            Some(t) => format!("{}", t),
        };
        format!(
            "Remote(ranged={}, headers={}, timeout={}, block_bytes={}, \
             cache_bytes={})",
            if self.ranged { "True" } else { "False" },
            headers, timeout, self.block_bytes, self.cache_bytes,
        )
    }
}

// Parse the `remote=` argument to FITS(): None, the string "ranged"
// (shorthand for Remote(ranged=True)), or a Remote instance.  A bare
// bool gets a tailored rejection — "remote: yes" is already implied
// by the URL, so `remote=True` has no meaning.
pub(crate) fn parse_remote_arg(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Remote>> {
    let Some(obj) = obj else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    if obj.extract::<bool>().is_ok() {
        return Err(PyValueError::new_err(
            "remote=True/False has no meaning; pass remote='ranged' \
             for ranged reads, or a rustfits.Remote(...) instance for \
             full control (default download mode needs no remote= at \
             all)",
        ));
    }
    if let Ok(s) = obj.extract::<String>() {
        if s == "ranged" {
            return Ok(Some(Remote {
                ranged: true,
                headers: Vec::new(),
                timeout: None,
                block_bytes: DEFAULT_BLOCK_BYTES,
                cache_bytes: DEFAULT_CACHE_BYTES,
            }));
        }
        return Err(PyValueError::new_err(format!(
            "remote= string must be 'ranged'; got '{}'.  For other \
             options pass a rustfits.Remote(...) instance", s
        )));
    }
    if let Ok(r) = obj.extract::<Remote>() {
        return Ok(Some(r));
    }
    Err(PyTypeError::new_err(
        "remote= must be None, the string 'ranged', or a \
         rustfits.Remote instance",
    ))
}

// One agent per open file (shared with the download path): connection
// keep-alive means one TCP + TLS handshake amortized over every
// request.
pub(crate) fn build_agent(timeout: Option<f64>) -> ureq::Agent {
    let mut cfg = ureq::Agent::config_builder();
    if let Some(secs) = timeout {
        cfg = cfg.timeout_global(Some(Duration::from_secs_f64(secs)));
    }
    ureq::Agent::new_with_config(cfg.build())
}

// The `Storage::HttpRange` backend: a lazy, read-only view of a
// remote file.  `seek` is pure arithmetic; `read` is served from the
// block cache, fetching misses in coalesced runs.
pub(crate) struct HttpRangeStore {
    url: String,
    agent: ureq::Agent,
    headers: Vec<(String, String)>,
    len: u64,
    pos: u64,
    block_bytes: u64,
    cache: BytesBoundLruCache<u64>, // block index -> block bytes
    // Stats, updated per fetch.  Not yet exposed to Python (a
    // deferred follow-up); kept for debugging and future use.
    bytes_fetched: u64,
    requests: u64,
}

impl HttpRangeStore {
    // Probe the server and build the store.  GET with
    // `Range: bytes=0-0` rather than HEAD (some servers mishandle
    // HEAD; the probe doubles as a reachability + auth check).  A 206
    // with Content-Range gives the file length; a 200 means the
    // server ignored Range — hard error per the roadmap decision (a
    // user who asked for ranged mode must not silently fall back to
    // downloading a possibly-huge file).
    //
    // Pure Rust (no GIL): FITS::new wraps this call in `py.detach`.
    pub(crate) fn open(
        url: &str,
        headers: &[(String, String)],
        timeout: Option<f64>,
        block_bytes: u64,
        cache_bytes: u64,
    ) -> io::Result<HttpRangeStore> {
        let agent = build_agent(timeout);
        let mut req = agent.get(url).header("Range", "bytes=0-0");
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let response = req.call().map_err(|e| {
            io::Error::other(format!("Failed to probe '{}': {}", url, e))
        })?;
        let status = response.status().as_u16();
        if status == 200 {
            // Do NOT read the body — it is the entire file.
            return Err(io::Error::other(format!(
                "the server for '{}' does not support HTTP Range \
                 requests (the probe returned 200 instead of 206); \
                 open with ranged=False (the default download mode) \
                 to fetch the whole file instead",
                url
            )));
        }
        if status != 206 {
            return Err(io::Error::other(format!(
                "range probe for '{}' returned status {} (expected \
                 206)", url, status
            )));
        }
        let len = response
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_content_range)
            .ok_or_else(|| {
                io::Error::other(format!(
                    "the server for '{}' returned 206 without a \
                     parseable Content-Range total; cannot determine \
                     the file length", url
                ))
            })?;
        // Drain the 1-byte probe body so the pooled connection is
        // reusable for the block fetches.
        let mut sink = Vec::new();
        let _ = response.into_body().into_reader().read_to_end(&mut sink);
        Ok(HttpRangeStore {
            url: url.to_string(),
            agent,
            headers: headers.to_vec(),
            len,
            pos: 0,
            block_bytes,
            cache: BytesBoundLruCache::new(cache_bytes),
            bytes_fetched: 0,
            requests: 1,
        })
    }

    // File length learned from the probe.  Infallible, unlike
    // `File::metadata()` — `Storage::len` wraps it in Ok.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    // Fetch `n_blocks` consecutive blocks starting at `start_block`
    // with ONE Range request, returning each block as its own Arc
    // (the file's last block may be short).  Blocks are read straight
    // off the body reader into per-block Vecs — no whole-run
    // intermediate buffer.
    fn fetch_blocks(
        &mut self,
        start_block: u64,
        n_blocks: u64,
    ) -> io::Result<Vec<Arc<Vec<u8>>>> {
        let first_byte = start_block * self.block_bytes;
        let last_byte = ((start_block + n_blocks) * self.block_bytes)
            .min(self.len)
            - 1;
        let range = format!("bytes={}-{}", first_byte, last_byte);
        let mut req = self.agent.get(&self.url).header("Range", &range);
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let response = req.call().map_err(|e| {
            io::Error::other(format!(
                "Range request '{}' for '{}' failed: {}",
                range, self.url, e
            ))
        })?;
        let status = response.status().as_u16();
        if status != 206 {
            return Err(io::Error::other(format!(
                "Range request '{}' for '{}' returned status {} \
                 (expected 206)",
                range, self.url, status
            )));
        }
        let mut reader = response.into_body().into_reader();
        let mut out = Vec::with_capacity(n_blocks as usize);
        for i in 0..n_blocks {
            let block_start = (start_block + i) * self.block_bytes;
            let expect = self.block_bytes.min(self.len - block_start);
            let mut block = vec![0u8; expect as usize];
            reader.read_exact(&mut block).map_err(|e| {
                io::Error::other(format!(
                    "short body for Range request '{}' on '{}': {}",
                    range, self.url, e
                ))
            })?;
            self.bytes_fetched += expect;
            out.push(Arc::new(block));
        }
        self.requests += 1;
        Ok(out)
    }
}

impl Read for HttpRangeStore {
    // Serve from the block cache, fetching missing blocks in
    // contiguous runs.  Never returns a short read except at EOF,
    // matching what the read_exact-heavy call sites expect from a
    // File.
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() || self.pos >= self.len {
            return Ok(0);
        }
        let want = (buf.len() as u64).min(self.len - self.pos);
        let span_start = self.pos;
        let span_end = span_start + want; // exclusive
        let first_block = span_start / self.block_bytes;
        let last_block = (span_end - 1) / self.block_bytes;
        // Pull what we have, HOLDING the Arcs — a tiny cache (or
        // cache_bytes=0, where put() no-ops) can't evict a block
        // between fetch and copy because the copy pass below never
        // re-consults the cache.
        let mut blocks: Vec<Option<Arc<Vec<u8>>>> = (first_block
            ..=last_block)
            .map(|b| self.cache.get(&b))
            .collect();
        let have: Vec<bool> =
            blocks.iter().map(|b| b.is_some()).collect();
        for (run_start, run_len) in plan_block_fetches(first_block, &have)
        {
            let fetched = self.fetch_blocks(run_start, run_len)?;
            for (i, blk) in fetched.into_iter().enumerate() {
                let idx = run_start + i as u64;
                self.cache.put(idx, blk.clone());
                blocks[(idx - first_block) as usize] = Some(blk);
            }
        }
        for (i, blk) in blocks.iter().enumerate() {
            let blk =
                blk.as_ref().expect("all missing runs were fetched");
            let block_start =
                (first_block + i as u64) * self.block_bytes;
            let copy_from = span_start.max(block_start);
            let copy_to =
                span_end.min(block_start + blk.len() as u64);
            let s = (copy_from - block_start) as usize;
            let e = (copy_to - block_start) as usize;
            let d = (copy_from - span_start) as usize;
            buf[d..d + (e - s)].copy_from_slice(&blk[s..e]);
        }
        self.pos = span_end;
        Ok(want as usize)
    }
}

impl Seek for HttpRangeStore {
    // Pure position arithmetic — no network.  Seeking past EOF is
    // allowed (like File); the next read returns 0.
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new = match pos {
            SeekFrom::Start(p) => Some(p),
            SeekFrom::End(d) => add_offset(self.len, d),
            SeekFrom::Current(d) => add_offset(self.pos, d),
        };
        match new {
            Some(p) => {
                self.pos = p;
                Ok(p)
            }
            None => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek to a negative position",
            )),
        }
    }
}

fn add_offset(base: u64, delta: i64) -> Option<u64> {
    if delta >= 0 {
        base.checked_add(delta as u64)
    } else {
        base.checked_sub(delta.unsigned_abs())
    }
}

// Given the first block index of a read span and, for each block in
// the span, whether it is already cached, return the contiguous runs
// of MISSING blocks as (start_block, n_blocks) pairs.  Coalescing is
// what turns a large strip read into a single Range request.
fn plan_block_fetches(first_block: u64, have: &[bool]) -> Vec<(u64, u64)> {
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < have.len() {
        if have[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < have.len() && !have[i] {
            i += 1;
        }
        runs.push((first_block + start as u64, (i - start) as u64));
    }
    runs
}

// Parse the total-length field of a `Content-Range: bytes a-b/total`
// header.  Returns None for an unknown total (`*`) or anything
// malformed.
fn parse_content_range(value: &str) -> Option<u64> {
    let rest = value.trim().strip_prefix("bytes")?.trim_start();
    let total = rest.split('/').nth(1)?.trim();
    total.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_block_fetches_coalesces_runs() {
        // Nothing missing -> no fetches.
        assert!(plan_block_fetches(3, &[true, true]).is_empty());
        assert!(plan_block_fetches(9, &[]).is_empty());
        // All missing -> one coalesced run.
        assert_eq!(plan_block_fetches(5, &[false; 4]), vec![(5, 4)]);
        // Interior hits split the runs.
        assert_eq!(
            plan_block_fetches(
                0,
                &[false, true, false, false, true, false]
            ),
            vec![(0, 1), (2, 2), (5, 1)]
        );
        // Leading/trailing hits trim the runs.
        assert_eq!(
            plan_block_fetches(7, &[true, false, false, true]),
            vec![(8, 2)]
        );
    }

    #[test]
    fn parse_content_range_totals() {
        assert_eq!(parse_content_range("bytes 0-0/12345"), Some(12345));
        assert_eq!(
            parse_content_range(" bytes 100-199/200 "),
            Some(200)
        );
        assert_eq!(parse_content_range("bytes */2000"), Some(2000));
        assert_eq!(parse_content_range("bytes 0-0/*"), None);
        assert_eq!(parse_content_range("meters 0-0/5"), None);
        assert_eq!(parse_content_range("bytes 0-0"), None);
        assert_eq!(parse_content_range(""), None);
    }

    #[test]
    fn seek_arithmetic() {
        assert_eq!(add_offset(10, 5), Some(15));
        assert_eq!(add_offset(10, -5), Some(5));
        assert_eq!(add_offset(10, -10), Some(0));
        assert_eq!(add_offset(10, -11), None);
        assert_eq!(add_offset(u64::MAX, 1), None);
    }
}
