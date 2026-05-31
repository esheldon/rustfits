# Tile-compressed images (ZIMAGE) — full implementation roadmap

*Extracted from `CLAUDE.md` to keep the always-loaded file lean.
Read this when working on `src/hdu_image_compressed/` or
`src/zimage/`.  The CLAUDE.md entry "Tile-compressed images
(ZIMAGE)" carries the one-paragraph summary + a pointer here.*


ZIMAGE-format tile-compressed images are stored on disk as a
BINTABLE with `ZIMAGE=T` plus Z-prefixed image-shape and tile-
shape cards.  The user-facing API mirrors `ImageHDU`; internally
the reader walks tiles and decodes them.

**Status: feature-complete for typical workloads.**

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Detection + accessors + tile cache | ✅ Shipped |
| 2-3 | RICE_1 read + slicing + LRU cache | ✅ Shipped |
| 4 | GZIP_1 / GZIP_2 reads + fallback columns | ✅ Shipped |
| 5 | Quantized + unquantized float reads (all 3 dither methods) | ✅ Shipped |
| 6 | HCOMPRESS_1 + PLIO_1 reads | ✅ Shipped |
| 7 | All 5 algorithms' integer-ZBITPIX writes | ✅ Shipped |
| 7b | Unsigned-int trick (i1/u2/u4/u8) on writes | ✅ Shipped |
| 8 | Quantized + unquantized float writes (`Quantize` config; `quantize=None`) | ✅ Shipped |
| 9 | `extend(data)` + `__setitem__` (integer + unsigned-trick + unquantized-float) | ✅ Shipped |
| 9q | `extend` + `__setitem__` for quantized-float (no compounding loss) | ✅ Shipped |
| 10a | `blank=` + `mask_blank=True` + MaskedArray input | ✅ Shipped |
| 10b | `Gzip1(level=)` / `Gzip2(level=)` (custom zlib level) | ✅ Shipped |
| 10c | `add_checksum` / `verify_checksum` (CHECKSUM/DATASUM + ZHECKSUM/ZDATASUM) | ✅ Shipped |

**Open follow-ups (low priority):**

- **Performance** — release-mode chunked-read of large GZIP_2
  float files: **3.2× FASTER** than fitsio on big chunks (50k
  rows; at the decoder physics floor), **40× FASTER** on small
  chunks (1k rows; up from 24× before the header-meta cache
  landed — see Phase 2 in "Header-derived metadata caching"
  below).  All numbers are vs fitsio (the Python wrapper around
  cfitsio); fitsio carries its own per-call wrapping overhead
  that likely inflates these ratios relative to a direct-
  cfitsio comparison (worth measuring some day, but the
  realistic comparison for users is fitsio anyway).
  See "Performance — ZIMAGE chunked-read profiling history"
  below for how the gap closed.
- **Byte-exact heap agreement with cfitsio on quantized floats**
  — decoded values are bit-exact; raw heap bytes differ by qsort
  tie-breaking quirks.  Not worth fixing absent a specific need.

**Deferred (not in the punch list):**

- **Per-tile ZBLANK column** — header-level ZBLANK is supported;
  the convention also allows a per-tile column form.  Rare in
  practice and nobody's asked.  Defer.
- **`mask_blank=True` on quantized-float compressed reads** —
  rejected by design (matches the uncompressed-float rule; FITS
  spec forbids BLANK on floats — NaN serves that role).  Not a
  gap.

Test fixtures use fitsio for normal-path round-trips and hand-
crafted bytes for synthetic fallback-column cases (astropy is
also in the env for richer cases).

Implementation lives in `src/hdu_image_compressed/` (the
pyclass + dispatch) and `src/zimage/` (algorithm-specific
decoders).  Detection happens in `parse_hdus_from_file` in
`fits.rs`: a BINTABLE with `ZIMAGE=T` routes to
`CompressedImageHDU` instead of `TableHDU`.

**Phase 1 — Detection + accessors + cache plumbing.**  Done.
- `CompressedImageHDU` pyclass subclassing `HDU`.  Inherits
  `header` / `index` / `extname` / `extver` / `has_data` from
  the base; defines image-side accessors (`shape`, `dtype`,
  `bitpix`, `ndim`, `size`, `__len__`, `unit`) that read the
  Z-prefixed cards instead of NAXIS/BITPIX; and compression-
  specific accessors `compression` (returns the structured
  config object — see "Compression config" below) and
  `n_tiles` (storage-layout property).
- Tile-cache config: `tile_cache_size` getter + `set_tile_cache_size(bytes)`
  setter, default 32 MiB.  Storage of decoded tiles in the LRU
  itself landed in Phase 3 — Phase 1 only held the configured
  size.
- Detection is BINTABLE-with-ZIMAGE-card; the helper
  `header_has_zimage` is in the `hdu_image_compressed` module
  and called from `fits.rs::parse_hdus_from_file`.
- `header_has_zimage` is lenient about whitespace; the value
  parse just looks for a `T` in the value portion of the
  card after `=`.
- `extname`/`extver`/`has_data` are inherited from the HDU
  base.  `has_data` reads BINTABLE NAXIS2 (n_tiles), which
  happens to agree with "image has data" because an empty
  image has n_tiles=0 → NAXIS2=0.
- Raw `hdu.header["BITPIX"]` returns 8 (the BINTABLE bitpix
  on disk).  Use `hdu.bitpix` for the image-side value
  (`ZBITPIX`).  Astropy follows the same convention.

**Phase 2 — RICE_1 whole-image read.**  Done.  Decoder in
`src/zimage/rice.rs` (`BitReader` + RICE_1 spec per cfitsio's
ricecomp.c).  Dispatch in `src/zimage/mod.rs` via the
`CompressionAlgorithm` enum (parsed from ZCMPTYPE).  Integer
ZBITPIX only (8/16/32/64); `-32`/`-64` raises a "Phase 5"
NotImplementedError.  Non-RICE_1 ZCMPTYPE raises naming the
phase that will add it.  BSCALE/BZERO applied via the shared
`apply_image_scaling` machinery on the assembled array — the
read-side scaling code is unchanged.

Also landed: **inheritance restructure** —
`CompressedImageHDU` now extends `ImageHDU` via a
`PyClassInitializer` chain through HDU + ImageHDU +
CompressedImageHDU, so `isinstance(hdu, ImageHDU)` is True
on a compressed HDU.  Accessor `into_super()` calls step
through both parents (`slf.into_super().into_super()`).
ImageHDU's data-access methods are overridden by compressed-
specific implementations:
- `read` → the cache-aware tile decoder.
- `__getitem__` → tile-by-tile slice path.
- `write` → the bulk-write encoder (Phase 7 / 8).
- `extend` → append-along-axis-0 with partial-last-tile
  re-encoding (Phase 9).
- `__setitem__` → in-place pixel modification (Phase 9).

Phase 2 follow-ups (not blocking):
- **Add CompressedImageHDU cases to tests/test_repr.py** —
  the Phase 1 smoke test in test_image_compressed_accessors.py
  exists, but the full _show-instrumented visual-inspection
  suite in test_repr.py doesn't include compressed images
  yet.  Add alongside the ImageHDU/TableHDU cases.

**Phase 3 — Slicing + LRU cache.**  Done.  `__getitem__` walks
only the tiles overlapping the slice (per-axis `axis_overlap`
helper computes which slice indices land in each tile).  Same
slice surface as `ImageHDU`: slice / int / ellipsis per axis,
stepped slices, mixed int+slice, all-int → numpy scalar.

Bytes-bound LRU cache via the `lru` crate, keyed by flat tile
index.  Values are full-tile bytes already cast to the target
(stored) dtype in numpy C-order, so cache hits skip both
decode and dtype conversion.  Single tuning knob:
`set_tile_cache_size(bytes)` — 0 disables, default 32 MiB.
`tile_cache_used` reports current bytes; `clear_tile_cache()`
drops all entries (keeps the size setting).

`read()` and `__getitem__` both go through the same cache —
no per-call `cache=` kwarg.  Cache-on-by-default for `read()`
warms tiles for any follow-up slicing on the same handle; the
overhead is <1% wall time and ≤ tile_cache_size in memory.

Concurrency: the inner mutex is held only briefly for `get`
and `put` (Arc clone-out, then drop the lock).  File I/O for
a missed tile happens under the file lock per tile, but decode
runs without the file lock — multi-threaded callers don't
serialize through long decode runs.

**Phase 4 — GZIP_1 / GZIP_2 + fallback columns.**  Done.
Decoders in `src/zimage/gzip.rs`; trait widened to return
target-dtype native-order bytes; data-column lookup widened
to find primary + optional GZIP + optional UNCOMPRESSED
fallbacks; per-tile dispatch picks the first non-empty
column.

*Trait shape.*  `decode_tile_to_i64` was renamed to
`decode_tile_to_bytes(algorithm, compressed, n_pixels,
bytepix, blocksize, zbitpix) -> PyResult<Vec<u8>>`.  Each
decoder owns the entire bytes-to-bytes path including the
target-dtype cast and host-endian byteswap; the caller just
caches and places the bytes:
- RICE: bitstream decode → `cast_i64_to_target_bytes` (now
  living in rice.rs).
- GZIP_1: gzip decompress → byteswap (`!cfg!(target_endian
  = "big")`).
- GZIP_2: gzip decompress → reverse byte-shuffle → byteswap.

*Framing — the spec-vs-reality gotcha.*  The original Phase
4 plan in this file claimed cfitsio wrote a **zlib** stream
(magic `0x78 0x9C`) and recommended `ZlibDecoder`.  That was
wrong: cfitsio calls `deflateInit2` with `windowBits = 15 +
16`, which selects **gzip** framing (magic `0x1F 0x8B` plus
the CRC32/ISIZE footer).  We use `flate2::read::GzDecoder`
in `src/zimage/gzip.rs`; the framing check at the top of
the test suite confirms this is what fitsio actually writes.
Future code reading or producing tile-compressed gzip payloads
should use gzip framing, not raw zlib.

*Fallback columns.*  `find_compressed_data_column` became
`find_data_columns` returning `DataColumns { primary,
gzip_fallback, uncompressed_fallback }`.  Per-tile dispatch
lives in `fetch_tile_payload`: it reads the primary
descriptor first and, if nelements is 0, falls through to
the GZIP fallback and then the UNCOMPRESSED fallback.  The
return type is a `TilePayload` enum (`Compressed { bytes,
algorithm }` vs `Uncompressed { bytes }`) so `get_or_decode_tile`
knows whether to call the decoder dispatch or just byteswap.

`ColumnInfo` carries `inner_byte_width` so the descriptor's
`nelements` field converts to a byte count correctly even
when the column's inner type isn't byte (`UNCOMPRESSED_DATA`
can be `1PI`/`1PJ` matching the pixel type, in which case
nelements counts pixels not bytes).  COMPRESSED_DATA and
GZIP_COMPRESSED_DATA are always `1PB` so inner_byte_width=1.

*Test fixtures.*  Normal-path GZIP_1 / GZIP_2 round-trips
use fitsio.  Fitsio's high-level write API doesn't expose
fallback-column knobs, so the three fallback tests in
`tests/test_image_compressed_read_gzip.py` build minimal
FITS files by hand (`_build_fallback_fixture` and friends):
empty primary descriptor + populated fallback column + a
hand-built BINTABLE header.  This pattern is reusable for
any future fallback-related test.

*Scope boundaries unchanged.*  Phase 4 still supports
**integer ZBITPIX only** (`8/16/32/64`).  Float ZBITPIX
with GZIP still raises until Phase 5 (quantization).  The
GZIP decoders in principle handle any byte width, but with
no quantization layer the FP storage isn't meaningful yet.

*Known follow-up — i8 (TLONGLONG) with GZIP.*  fitsio
refuses to *write* i8 compressed images, raising "writing
TLONGLONG to compressed image is not supported" (cfitsio
`imcomp.c`).  This appears to be a cfitsio implementation
limitation rather than a FITS Tile Compression Convention
restriction — GZIP is a byte-stream codec with no inherent
BITPIX dependency, and the rustfits reader path already
handles `bytepix=8`.  Worth investigating later whether the
spec actually permits i8 GZIP (highly likely yes) and what
the cfitsio constraint is exactly; if the spec allows it,
our reader is already fine and the only missing piece is
test fixtures (hand-crafted, or maybe astropy can do it).

**Phase 5 — Quantized floats + unquantized float HDUs.**  Done.
Reader handles ZBITPIX=-32/-64 across all common shapes:
- **Quantized**: `NO_DITHER`, `SUBTRACTIVE_DITHER_1`, and
  `SUBTRACTIVE_DITHER_2` with bit-for-bit cfitsio agreement on
  the full {RICE_1, GZIP_1, GZIP_2} × {f4, f8} × {NO_DITHER,
  DITHER_1, DITHER_2} matrix (including NaN preservation
  through DITHER_2's reserved sentinel).
- **Unquantized**: ZQUANTIZ='NONE' (astropy's convention for
  "no quantization happened") and the astropy-quantize_level=0
  variant where ZQUANTIZ='NO_DITHER' is set but ZSCALE/ZZERO
  columns are absent.  On-disk bytes are raw GZIP-compressed
  floats; dequant is skipped.

*Code layout.*  All quantization logic lives in
`src/zimage/quantize.rs`:
- `DitherMethod` enum + `parse_dither_method` returning
  `Option<DitherMethod>` — `None` for ZQUANTIZ='NONE' (the
  "no quantization happened" signal).  Absent ZQUANTIZ
  defaults to `Some(NoDither)` per cfitsio's convention when
  ZSCALE/ZZERO are present.
- `random_table()` initialises the 10000-element Park-Miller
  table on first use (multiplier 16807, modulus 2^31-1,
  seed 1).  Lazy via `OnceLock`.
- `DitherStream` reproduces cfitsio's exact iseed / nextrand
  advancement (initial `nextrand = floor(table[iseed] * 500)`
  jump on roll-over) — required for byte-exact agreement with
  cfitsio's output.
- `dequantize_to_f32` / `_f64` apply the per-tile formula
  `(stored - dither + 0.5) * scale + zero`.  For DITHER_2 the
  reserved value `-2147483647` becomes NaN.
- 4 unit tests at the bottom of the module (Park-Miller
  anchor check, NO_DITHER linearity, DITHER_2 NaN handling,
  parse_dither_method covering 'NONE' + defaults + unknown).

*Integration in `hdu_image_compressed.rs`.*
- `DataColumns` widened to also hold `zscale_offset_in_row`
  and `zzero_offset_in_row` (fixed-width `1D` columns located
  by walking TTYPEn).
- `QuantContext { method, zdither0, zscale_offset_in_row,
  zzero_offset_in_row, output_zbitpix }`.  `build_quant_context`
  returns `Option<QuantContext>`: `None` when ZQUANTIZ='NONE'
  OR when ZSCALE/ZZERO columns are missing (the two signals
  for "no quantization happened" on a float HDU).  For integer
  HDUs the quant context is always None.
- A new `fetch_tile_payload_and_quant` reads the heap payload
  AND the per-tile ZSCALE/ZZERO doubles under the same file
  lock acquire (when quant is Some).  Falls back to the old
  `fetch_tile_payload` helper for the integer / unquantized
  paths (still present, kept simple).
- `get_or_decode_tile` decides bytepix and effective ZBITPIX
  for the decoder, then runs dequantization when the payload
  came from the primary column AND quant is Some.  A single
  unified rule: `dequant_applies = primary_payload &&
  quant.is_some()`.
- Both `read_compressed_image_data` and
  `slice_compressed_image` route through the new path; the
  cache stores final-output bytes (f4/f8 for quantized HDUs),
  same as the integer path.

*Decoder-vs-output ZBITPIX split.*  Two distinct values are
in play, and conflating them is a tempting bug:
- `stored_zbitpix`: what the *decoder* casts to — 32 (i32)
  for quantized float, same as image ZBITPIX otherwise.
- `output_zbitpix`: the image-side dtype — -32 / -64 for
  quantized float, integer ZBITPIX otherwise.
- `bytepix`: 4 for quantized float (matches stored i32),
  matches stored_zbitpix/8 otherwise.

When dequant doesn't apply (unquantized float HDU, or
fallback column on a float HDU), the decoder is called with
`bytepix = float_bytepix` (4 or 8 matching ZBITPIX) instead
of 4, since the bytes are already physical floats.

*Lossless-fallback convention.*  cfitsio's
GZIP_COMPRESSED_DATA fallback (and UNCOMPRESSED_DATA) for a
*float* HDU stores **raw original floats**, not quantized
i32.  This is the "lossless backup when quantization would
lose too much" path.  `TilePayload` was split into
`PrimaryCompressed` / `FallbackCompressed` / `Uncompressed`
variants so `get_or_decode_tile` can decide: dequant only
runs when the *primary* column produced bytes.  For the
fallback paths on float HDUs, bytes are already physical
floats — the decoder is called with the float bytepix
(4 or 8 matching ZBITPIX) and dequant is skipped.  This
unifies cleanly with the unquantized-float case: same code
path, same `dequant_applies = false` outcome.

*ZQUANTIZ='NONE' is NOT in the FITS spec.*  The FITS Tile
Compression Convention (Pence et al. 2010 + WG revisions)
defines exactly three ZQUANTIZ values (`NO_DITHER`,
`SUBTRACTIVE_DITHER_1`, `SUBTRACTIVE_DITHER_2`).  Per spec,
"no quantization" should be signalled by *omitting* the
keyword.  But astropy's `CompImageHDU` emits `'NONE'`
explicitly when writing unquantized float-compressed HDUs,
and cfitsio reads it tolerantly — so we accept it for
real-world compatibility.  Documented as a comment in
`parse_dither_method`.

*ZCMPTYPE alias accepted.*  fitsio writes `ZCMPTYPE='RICE_ONE'`
for some quantized-RICE configurations (an older cfitsio
synonym for `RICE_1`).  `parse_algorithm` in
`src/zimage/mod.rs` now accepts both names plus other older
cfitsio synonyms (`GZIP` for `GZIP_1`, `HCOMPRESS` for
`HCOMPRESS_1`).

*Known follow-ups not blocking the phase:*
- Header-level ZBLANK on integer compressed images (analog
  of `mask_blank` on uncompressed) still raises with the
  "Phase 2" message.  Lift when there's a use case.
- Per-tile ZBLANK column (column-form rather than header-
  level) — read code locates it (`find_data_columns` could be
  widened) but `QuantContext` doesn't carry it yet.  None of
  the matrix tests exercise this configuration; cfitsio
  typically emits a header-level ZBLANK for DITHER_2 rather
  than a per-tile column.

**Phase 6 — HCOMPRESS_1 read.**  Done.

Decoder in `src/zimage/hcompress.rs` — port of cfitsio's
`fits_hdecompress.c` (see "Reference sources for byte-exact
ports" under "Build / dev workflow" for how to obtain the
cfitsio source; the function names mirror cfitsio exactly so
side-by-side diffing during debug stays easy).  Both internal-precision paths shipped: i32 for
ZBITPIX = 8/16, i64 for ZBITPIX = 32 (cfitsio uses i64 internal
for 32-bit because the H-transform's intermediate sums can
overflow i32).  ZBITPIX = 64 is unsupported — cfitsio's encoder
doesn't write it and the spec has no 64-bit HCOMPRESS variant.

*Wire format gotchas.*  Stream starts with magic `0xDD 0x99`,
then i32 BE `nx, ny, scale`, then i64 BE `sumall`, then 3 bytes
of `nbitplanes`, then quadtree-coded bit planes per quadrant.
**SCALE comes from the stream, not the header** — the
ZNAMEn='SCALE' / ZVALn header value is informational only and
cfitsio ignores it in the decode path.  Only **SMOOTH** needs
header parsing (`parse_hcompress_smooth` in
hdu_image_compressed.rs, scans ZNAMEn pairs for 'SMOOTH').

*Axis convention.*  In cfitsio's HCOMPRESS code, `ny` is the
*FITS-fastest* axis (= NAXIS1 = numpy-last) and `nx` is the
slower one — opposite to "x = horizontal" usage but matching
the C row-major layout `a[i*ny + j]`.  In numpy order
(slowest first) that maps cleanly: `nx = tile_shape[0],
ny = tile_shape[1]`.  No reverse anywhere downstream.

*Dispatch.*  HCOMPRESS needs two extra pieces of info that
RICE/GZIP don't: the 2-D tile shape and the SMOOTH flag.  Added
`AlgorithmParams { tile_shape_numpy, smooth }` in
`src/zimage/mod.rs`; every call site (currently just
`get_or_decode_tile`) constructs one and passes it through
`decode_tile_to_bytes`.  RICE/GZIP ignore both fields.

*SMOOTH=1.*  Ported.  Three adjustment loops (hx / hy / hc) on
the inverse-H-transform coefficient block; edge coefficients
left untouched by the loop bounds (start at 2, end at nxtop-2 /
nytop-2), not by explicit boundary clauses.  Translation note:
cfitsio's
   `s = (s>=0) ? (s>>n) : ((s+(2^n-1))>>n)`
is "divide by 2^n truncating toward zero", which Rust's signed
`/` does directly — translated as `s / 8` / `s / 64` for
readability.

*Tests.*  `tests/test_image_compressed_read_hcompress.py`
covers accessors, lossless round-trip on u8/i16/i32, non-square
edge tiles, default tiles, slicing parity, bit-exact agreement
with cfitsio on a lossy hcomp_scale=4 fixture, and SMOOTH=1
round-trip at hcomp_scale={4, 16} for i16 plus hcomp_scale=8
for the i32 (i64-internal) path.

**Phase 6 — PLIO_1 read.**  Done.

Decoder in `src/zimage/plio.rs` — port of cfitsio's `pl_l2pi`
(`pliocomp.c`).  The cfitsio source is f2c-generated SPP with
heavy goto control flow; the Rust port replaces it with a
straightforward `match` on the opcode (top 4 bits of each
encoded word).

*Wire format.*  Stream is big-endian i16 shorts: 7-short header
(magic word `-100` at index [2] marks the modern format; an
older format puts a positive length there) followed by RLE data
words.  Each data word's top 4 bits select one of 8 opcodes:
zero runs (0, 5), set-high-pv (1 — consumes the next word for
the upper 16 bits of pv), increment/decrement pv (2, 3),
solid-pv run (4), single-pixel write with pv += data (6) or
pv -= data (7).

*Output type.*  Decoder produces `Vec<i32>` (PLIO is designed
for non-negative integer masks; values are built up through
increments from a starting pv=1).  Cast to the target dtype
follows the same shape as hcompress's: supports ZBITPIX = 8,
16, 32; rejects float ZBITPIX (PLIO doesn't make sense for
floats).

*Column TFORM.*  PLIO stores the compressed-data column as
`1PI` (VLA of i16 shorts) instead of the `1PB` used by RICE /
GZIP / HCOMPRESS.  No special-casing needed at the dispatch
layer — `tform_vla_inner_byte_width` returns 2 for 'I' so
`fetch_tile_payload_inner` correctly reads `nelements * 2`
bytes from the heap.

*Tests.*  `tests/test_image_compressed_read_plio.py` covers
the compression_type accessor, bit-exact agreement with cfitsio
on u8/i16/i32 mask-style fixtures (mostly zero with random
runs), all-zero tiles (header-only encoded stream), solid-value
tiles, and slicing parity.

All four ZIMAGE compression algorithms (RICE_1, GZIP_1/2,
HCOMPRESS_1, PLIO_1) now have full read support.

**Phase 7 — Compressed image writes (Gzip1, Gzip2, Rice1).**  Done.

*API.*  Compression is opted into via a structured config object
passed to `create_image_hdu`:

```python
f.create_image_hdu(
    "i4", (1000, 1000), extname="SCI",
    compress=rustfits.Gzip1(tile_shape=(100, 100), heap_format="P"),
)
f[1].write(data)
```

`compress=None` (default) → uncompressed `ImageHDU` (current
behavior, unchanged).  `compress=Gzip1(...)` / `Gzip2(...)` /
`Rice1(...)` / `Hcompress1(...)` / `Plio1(...)` →
`CompressedImageHDU`.  Algorithm objects (`Gzip1`, `Gzip2`,
`Rice1`, `Hcompress1`, `Plio1` — all shipped) live in
`src/zimage/compression_config.rs`.  Each is a Rust pyclass exposed
at `rustfits.<Name>`, validated at construction
(`Gzip1(blocksize=0)` raises immediately), immutable (no setters).
No inheritance — common kwargs (`tile_shape`, `heap_format`) are
duplicated across each constructor; for 5 small config classes the
duplication is trivial and avoids a PyClassInitializer chain.

Dispatch in `fits.rs::create_compressed_image_hdu_impl` goes
through an internal `CompressionConfigKind` enum that wraps the
per-algorithm pyclasses.  `CompressionConfigKind::from_pyany`
tries each variant's `extract::<Gzip1>()` / `extract::<Gzip2>()` /
`extract::<Rice1>()` / `extract::<Hcompress1>()` / `extract::<Plio1>()`
in turn — pyo3 0.28 doesn't grow a clean "first-match" extractor,
so the manual loop is the simplest shape.  The enum exposes the
shared surface (`tile_shape()`, `heap_format()`, `zcmptype()`)
plus an `extra_z_cards(bitpix)` accessor that returns `(ZNAMEn,
ZVALn)` pairs to emit alongside the standard ZIMAGE cards (RICE_1
emits BLOCKSIZE + BYTEPIX; HCOMPRESS_1 emits SCALE + SMOOTH;
GZIP and PLIO variants emit nothing).  PLIO uses TFORM1='1PI'
(i16 inner type) rather than '1PB' (byte) used by the others —
the encoder produces big-endian i16 shorts, and descriptor
nelements counts shorts not bytes.

**Structured `.compression` API.**  The read-side mirror image of
`compress=`: every `CompressedImageHDU` exposes a single
`.compression` getter that returns the same `Gzip1` / `Gzip2` /
`Rice1` / `Hcompress1` / `Plio1` pyclass instance that would have
been passed to `create_image_hdu(..., compress=...)` to produce a
file with the same on-disk parameters.  Round-trip pattern:

```python
cfg = Rice1(tile_shape=(16, 16), blocksize=64)
fits.create_image_hdu("i4", shape, compress=cfg)
assert fits[1].compression == cfg   # __eq__ is field-wise
```

Backing implementation: `hdu_image_compressed.rs::build_compression_config`
parses ZCMPTYPE, ZTILEn, TFORM1 (for heap_format), plus the
algorithm-specific ZNAMEn/ZVALn cards (BLOCKSIZE for RICE,
SCALE + SMOOTH for HCOMPRESS), and constructs the matching pyclass.
The flat HDU accessors `compression_type` and `tile_shape` were
removed in this refactor — use `hdu.compression.zcmptype` and
`hdu.compression.tile_shape` instead.  `n_tiles` stays on the HDU
(it's a storage-layout property derived from image_shape +
tile_shape, not an algorithm parameter).  Tile-cache runtime
knobs (`tile_cache_size`, `set_tile_cache_size`,
`clear_tile_cache`) also stay on the HDU.

The HDU repr uses the config object's own `__repr__`:

```
  compression: Rice1(tile_shape=[16, 16], heap_format='P', blocksize=32)
```

So the HDU repr stays the single source of truth no matter which
algorithm-specific parameters apply.  Fallbacks (repr never
crashes on a malformed file): unrecognized ZCMPTYPE → show the
raw string verbatim; missing ZCMPTYPE → show `None`.

Config classes also expose:
- `__eq__` (field-wise; cheap, useful for round-trip patterns).
- `zcmptype` getter returning the FITS-spec string (`"GZIP_1"`,
  `"RICE_1"`, ...).  Class name is the Pythonic form (`Gzip1`,
  `Rice1`); `.zcmptype` returns the FITS string when needed.

*Design discussion captured.*  See git log around this commit
for the full thread: the chosen shape is one unified
`create_image_hdu` (Option B) rather than a separate
`create_compressed_image_hdu` (Option A), with structured config
in the `compress=` slot rather than flat kwargs.  Decisions:
(1) `tile_shape` / `heap_format` live on the algorithm object
(not hoisted onto `create_image_hdu`); (2) Pythonic class names
(`Gzip1`, not `GZIP_1`); (3) eager validation, immutable objects;
(4) `heap_format` rather than `descriptor` (the kwarg name in
`create_table_hdu` was renamed at the same time for consistency).

*Layered architecture.*  rustfits's surface is designed for the
best ergonomics we can build.  A future `rustfits.compat.fitsio`
shim will present a fitsio-shaped surface for migrators (or as
the backend for a fitsio 2.0).  Decisions like the
`compress=Gzip1(...)` shape don't have to match fitsio's
`compress='GZIP_1', tile_dims=...` flat-kwarg style — the shim
absorbs the translation.

*Scope (current).*  `Gzip1` + `Gzip2` + `Rice1`; integer ZBITPIX
(8/16/32/64 for GZIP; 8/16/32 for RICE — i8/BYTEPIX=8 rejected,
see below) only; create + bulk `write` only.  Float ZBITPIX
(-32/-64) and unsigned-int trick dtypes (i1/u2/u4/u8) raise
`NotImplementedError`.  `extend` and `__setitem__` still raise
the original Phase 2 stub message.

*Mechanics.*  `CompressedImageHDU.write` encodes every tile into
RAM first (validate-then-mutate), then mutates the file: grows
via `shift_file_tail_and_update_offsets` when not the last HDU
(later HDU offsets bump in lockstep through the shared
`Arc<FileLayout>`), or `set_len` when it is; writes per-tile
descriptors into the main data section, then heap bytes; finally
rewrites the PCOUNT card via the same disk-write-before-commit
+ taint pattern every other write path uses.  Tile cache is
cleared on each write (cached entries from any prior read are
stale).

*GZIP_1 encoder.*  `encode_gzip1` in `src/zimage/gzip.rs` uses
`flate2::write::GzEncoder` with `Compression::default()` (zlib
level 6 — same as cfitsio/zlib defaults).  Input bytes are in
FITS big-endian order; caller (`write_compressed_image_data` in
`hdu_image_compressed.rs`) handles the byteswap via numpy
`ascontiguousarray(view, ">i4")` etc. — one pass that handles
dtype + endian together.

*GZIP_2 encoder.*  `encode_gzip2(pixel_bytes_be, bytepix)` in
`src/zimage/gzip.rs` runs the GZIP_1 encoder's underlying gzip
primitive over byte-shuffled input.  The `shuffle()` helper is
the inverse of `unshuffle()` from the decoder side — for
`bytepix=1` the shuffle is a no-op and the path collapses to
GZIP_1 (we even round-trip the same bytes through the file in
that case; tests anchor this invariant).  `encode_tile_from_bytes`
in `src/zimage/mod.rs` grew a `bytepix` parameter to thread the
pixel width through to the encoder (GZIP_1 ignores it).

*RICE_1 encoder.*  `encode_rice(pixel_bytes_be, n_pixels, bytepix,
blocksize)` in `src/zimage/rice.rs` is a byte-exact port of
cfitsio's `fits_rcomp` / `_short` / `_byte` family.  The
algorithm: read pixels as sign-extended integers of the natural
bytepix width into i32 working precision; for each block of up to
`blocksize` pixels compute ZigZag-mapped deltas, sum them in f64,
derive the Rice split parameter `fs` from cfitsio's exact dpsum/psum
heuristic (with per-bytepix psum cast type — u8/u16/u32 caps psum
at the natural unsigned width); pick one of three branches per
block (low-entropy: emit `fs=0` marker only; high-entropy: emit
`fsmax+1` marker + raw bbits-wide diffs; normal Rice: emit `fs+1`
marker + unary top + raw fs-wide bottom for each pixel).  A small
MSB-first `BitWriter` (cousin of the existing `BitReader`) handles
the bit packing.  `encode_tile_from_bytes` grew an
`AlgorithmEncodeParams { blocksize }` struct (mirror of the
decode-side `AlgorithmParams`) plus an `n_pixels` arg so RICE knows
how many integers to pull out of the input bytes.  The output is
**byte-exact identical to cfitsio's encoder output** on the same
input — verified across u1/i2/i4 dtypes, multi-tile layouts,
low-entropy / high-entropy / mixed blocks.

*RICE_1 i8 (BYTEPIX=8) rejected.*  cfitsio's encoder family stops
at BYTEPIX=4: there is no `fits_rcomp_longlong`, fitsio raises
"writing TLONGLONG to compressed image is not supported", and
astropy silently downcasts i64 → i32 before encoding (lossy when
values exceed the i32 range, no warning).  No production reader
verifies BYTEPIX=8 RICE streams, so writing them would produce
files unreadable outside rustfits.  `create_compressed_image_hdu`
rejects `compress=Rice1()` + i8 dtype upfront with a clear
`NotImplementedError` pointing at `Gzip2` instead.  Empirical check
(see git log around this commit): on real i64 imaging patterns
(smooth + Poisson, smooth + 1e12 bias, wide-spread, sparse),
Gzip2 lands within ~5% of where i64 RICE would compress, and is
universally readable.  The encoder itself also rejects bytepix=8
as a defensive check.

*Tests.*  `tests/test_image_compressed_write_gzip.py` (GZIP_1),
`tests/test_image_compressed_write_gzip2.py` (GZIP_2), and
`tests/test_image_compressed_write_rice.py` (RICE_1) —
each covers accessors after create, dtype matrix (u1/i2/i4 +
i8 for GZIP only), shape matrix (1-D / 2-D square / 2-D non-square
with edge tiles / 3-D / whole-image single tile), default-tile-shape
round trip, bidirectional cross-check with fitsio (rustfits-written
read by fitsio AND fitsio-written read by rustfits, both bit-exact),
non-last HDU growth (heap shifts later HDU forward), and all
rejection paths (float, unsigned trick, shape mismatch, start kwarg).
The GZIP_2 file adds the bytepix=1 shuffle-collapses-to-GZIP_1
invariant (file sizes equal on `u1` input) and the mixed-algorithm
case (one GZIP_1 + one GZIP_2 HDU in the same file).  The RICE_1
file adds **byte-exact heap-comparison tests against fitsio**
(low-entropy, high-entropy, multi-tile, dtype matrix), a custom
blocksize test, and the i8-rejected test.

*HCOMPRESS_1 encoder.*  `encode_hcompress` in
`src/zimage/hcompress.rs` is a byte-exact port of cfitsio's
`fits_hcompress.c` family (`htrans`, `digitize`, `shuffle`,
`encode`, `doencode`, `qtree_encode`, `qtree_onebit`,
`qtree_reduce`, `bufcopy`, `write_bdirect`, plus the
`output_nbits`/`nybble`/`nnybble` bit-output state).  Internal
precision tracks the read side: i32 for ZBITPIX = 8/16, i64 for
ZBITPIX = 32 (the H-transform's intermediate sums can overflow
i32 for 32-bit input).  ZBITPIX = 64 (i8 dtype) is rejected
upstream — the FITS Tile Compression Convention has no 64-bit
HCOMPRESS variant.  SCALE controls quantization (0 or 1 =
lossless; larger = lossy); SMOOTH is encoder-irrelevant (the
smoothing pass runs on read only) but its ZNAME card must be
emitted so the reader knows whether to smooth.  Output is
byte-exact with cfitsio's encoder across all tested cases.

*HCOMPRESS_1 tile-shape policy.*  Two FITS Tile Compression
Convention constraints apply: (1) every image dimension must
have ≥ 4 pixels (HCOMPRESS is a 2-D wavelet); (2) every tile
along each axis (including the edge tile) must have ≥ 4 pixels.
The three major writers disagree on how to enforce constraint
(2): **astropy** raises `ValueError`; **cfitsio** silently
rewrites the user's tile dim upward (e.g. `tile=16` becomes
`tile=17` for `naxis=50` to make `50 % 17 = 16` instead of
`50 % 16 = 2`); **rustfits follows astropy** — explicit is
safer — but the error message includes the cfitsio-style
adjusted suggestion so the user can copy it back into their
config.  Default `tile_shape` (when `Hcompress1(tile_shape=
None)`) is **not** the FITS row-tiles convention (which would
always violate constraint 2); instead it ports cfitsio's
default heuristic in `fits.rs::hcompress_default_slow_tile`:
whole image when `NAXIS2 ≤ 30`, otherwise the first value from
`{16, 24, 20, 30, 28, 26, 22, 18, 14}` that leaves a valid
edge tile, with `17` as the last-resort fallback.  16-row
stripes are what HST/DECam/HSC files in the wild use (cfitsio
is the dominant writer of these), so the default matches.

*HCOMPRESS_1 tests.*
`tests/test_image_compressed_write_hcompress.py` covers
accessors after create, dtype matrix (u1/i2/i4), shape matrix
(divisible 2-D cases), default-tile-shape (small-image,
large-image stripe, and fallback-when-16-doesn't-work cases),
lossless + lossy round trips (scale ∈ {4, 8, 16}), SMOOTH=True
round trip, **byte-exact heap agreement with fitsio** on
single-tile + multi-tile-with-divisible-dims + valid-edge-tile
cases for both lossless and lossy, bidirectional fitsio
cross-check, non-last HDU growth, mixed-algorithm file, and all
rejection paths (float, i8, unsigned trick, 1-D / 3-D images,
shape mismatch, start kwarg, dim < 4, tile < 4, and thin edge
tile with cfitsio-style suggested fix in the error message).

*HCOMPRESS_1 lossy testing gotcha — `hcomp_scale` sign.*  When
cross-checking against fitsio's writer with a fixed scale (e.g.
to compare byte-exact heap output at `scale=4`), pass
`hcomp_scale=-4` (negative).  cfitsio's `hcomp_scale > 0`
branch multiplies the user value by a per-tile noise estimate
to derive the actual scale, giving a *different* scale per
tile; the `hcomp_scale < 0` branch uses the absolute value
directly as a fixed scale.  Tests in `phase7_hcompress_write.py`
use the negative form.

*PLIO_1 encoder.*  `encode_plio` in `src/zimage/plio.rs` is a
byte-exact port of cfitsio's `pl_p2li` from
`<cfitsio>/pliocomp.c`.  The SPP/f2c source's goto soup is
replaced by a single `while` loop with explicit state and
linear emit logic — much easier to read than cfitsio's labelled
control flow.  The opcode dispatch matches the read side
(0=zero-run, 1=set-high-pv, 2/3=±pv, 4=solid-pv-run,
5=zero-run-with-trailing-pv, 6/7=set-and-write-single-pixel);
single-pixel run combinations (opcodes 5/6/7) are emitted via
the same `+ 20481` / `| 16384` modifications cfitsio uses on
the previous word.  PLIO is integer-only (ZBITPIX 8/16/32; no
64-bit variant in the FITS Tile Compression Convention);
inputs must be non-negative (encoder rejects negatives with a
clear error), and pv must fit in 2^27 - 1 (encoder rejects
larger values rather than silently truncating as cfitsio does).

*PLIO_1 TFORM and descriptor mechanics.*  PLIO writes its
heap as i16 big-endian shorts (TFORM1='1PI' or '1QI'), unlike
the other algorithms which use byte-inner TFORM1='1PB'/'1QB'.
The descriptor's `nelements` field counts ELEMENTS of the
inner type, not bytes — so on the write side
`write_compressed_image_data` divides `encoded.len()` by
`inner_byte_width` (= 2 for PLIO, 1 for the others) when
filling descriptors.  The read side already worked correctly
because `tform_vla_inner_byte_width` returns the right value.

*PLIO_1 tests.*  `tests/test_image_compressed_write_plio.py`
covers accessors after create (including TFORM1='1PI'
verification), dtype matrix (u1/i2/i4), shape matrix, default
tile shape, degenerate cases (all-zeros, all-solid, large-pv,
sparse single-pixel), **byte-exact heap agreement with fitsio**
across mask-style and degenerate inputs, bidirectional fitsio
cross-check, non-last HDU growth, mixed-algorithm file
(Plio1 + Gzip2), and rejection paths (float, i8, unsigned
trick, negative pixels, values > 2^27, shape mismatch, start
kwarg).

**Unsigned-int trick on write (i1/u2/u4/u8).**  Shipped
2026-05-23.  Works for Gzip1/Gzip2/Rice1/Hcompress1 (PLIO_1 is
rejected because the reverse XOR produces signed stored values
that include negatives, which PLIO's non-negative encoder can't
represent).  Implementation: reuses
`hdu_image.rs::reverse_unsigned_trick`; the compressed-write
path's `normalize_compressed_input_dtype` helper dispatches
fast-path (BITPIX-native input pass-through) vs reverse-XOR
based on the input dtype and the HDU's BSCALE/BZERO.  BSCALE/
BZERO cards are emitted at create time in
`create_compressed_image_hdu_impl` using the same pattern as
the uncompressed path.  See
`tests/test_image_compressed_unsigned_trick.py` (33 cases).
*Known limitation (not new):* astropy returns f8 (with
precision loss) on u8 + BZERO=2^63 — also affects uncompressed
u8.  rustfits's own round-trip is bit-exact.

**Phase 8 — Quantized float compressed writes.**  Shipped (read +
write).

*API.*  Quantization parameters live in a separate `Quantize`
config object passed alongside the algorithm:

```python
fits.create_image_hdu(
    "f4", shape,
    compress=Rice1(tile_shape=(100, 100)),
    quantize=Quantize(level=4.0, method="dither1", seed=0),
)
```

Kwargs: `level` (default 4.0 = "N sigma per quanta"; negative =
fixed bscale = -level), `method` (`'no_dither'` / `'dither1'` /
`'dither2'`; default `'dither1'` matches cfitsio), `seed`
(ZDITHER0; 0 → on-disk default of 1).  Integer HDUs reject
`quantize=` with a clear error.  Float HDUs WITHOUT `quantize=`
(or with `quantize=None`) write **unquantized** — lossless raw
float bytes through GZIP_1 / GZIP_2 — see the "Unquantized
float compression" subsection below.  Float HDUs WITH a
`Quantize` object emit the 4-column quantized schema described
in the *Schema* section.  Pythonic kebab-style strings for
method names; FITS-spec ZQUANTIZ values are available via
`quantize.zquantiz`.

*Schema (quantized floats).*  Float HDUs WITH a Quantize emit
a 4-column BINTABLE:

  | column | TFORM | role |
  |--------|-------|------|
  | COMPRESSED_DATA       | 1PB / 1QB / 1PI / 1QI | quantized i32 → algorithm output |
  | ZSCALE                | 1D                    | per-tile bscale |
  | ZZERO                 | 1D                    | per-tile bzero  |
  | GZIP_COMPRESSED_DATA  | 1PB / 1QB             | lossless raw float fallback |

Plus ZQUANTIZ (NO_DITHER / SUBTRACTIVE_DITHER_1 /
SUBTRACTIVE_DITHER_2), ZDITHER0, and ZBLANK = -2147483647
(NULL_VALUE_I32 sentinel).  The FITS Tile Compression Convention
records method + seed only — not the qlevel — so the
`CompressedImageHDU` carries `quantize_config: Arc<Mutex<Option<
Quantize>>>` populated at create time and consulted by
`.write()`.  Reopened HDUs get `None` and fall back to qlevel=4.0
(cfitsio's default).

*Quantize / dequantize algorithm.*  Direct ports of cfitsio's
`fits_quantize_float` / `_double` and the `FnNoise5_float` /
`_double` MAD noise estimator.  Per-tile noise = min of 2nd / 3rd
/ 5th-order Median Absolute Differences (Pence MAD-to-sigma
normalization constants 1.0483579 / 0.6052697 / 0.1772048);
bscale = noise / qlevel; bzero chosen to keep the quantized
range inside i32 minus 10 reserved sentinels.  Per-pixel:
DITHER_2 reserves NULL_VALUE_I32 for NaN and ZERO_VALUE_I32 for
exact-zero floats; the other methods just reserve NULL_VALUE_I32
for NaN.  The dither stream (Park-Miller PRNG, same one already
shared with the decoder) advances per-pixel regardless of
sentinel hits to stay synced with the encoder.

*GZIP fallback.*  When `quantize_float` returns `None` (constant
tile, range too wide for i32, fewer than 2 pixels), the tile's
raw float bytes are GZIP-1 compressed and stored in the
GZIP_COMPRESSED_DATA column; the primary descriptor stays empty
(nelements=0).  The read side already routed empty primary to
the fallback column in Phase 4, so no read changes needed.

*PLIO + float rejected.*  PLIO's encoder requires non-negative
inputs.  Quantization produces an i32 stream with negative
values (bzero shifts the range), which PLIO can't represent.
The `create_compressed_image_hdu_impl` dispatch rejects the
combination up-front with a clear error pointing the user at
Gzip2 or Rice1.

*Lossless decoder fix.*  Phase 5 had a latent bug — only
SUBTRACTIVE_DITHER_2 in `dequantize_to_f32/_f64` checked
NULL_VALUE_I32 → NaN; NoDither and SUBTRACTIVE_DITHER_1 silently
turned the sentinel into a giant negative number.  Phase 8
commit 3 added the missing checks to both branches.  No
existing tests caught the gap because no Phase 5 fixture had
NULL_VALUE_I32 in the stored stream under NoDither or DITHER_1.

*Tests (`test_image_compressed_write_quantize.py`).*  29
cases covering: schema (TFORM / TTYPE / ZQUANTIZ / ZDITHER0 /
ZBLANK cards), round-trip across f4/f8 + (NO_DITHER, DITHER_1,
DITHER_2), default Quantize defaults, fitsio cross-read agreement
across the (algorithm, method) matrix, DITHER_2 exact-zero
preservation, NaN round-trip for all three methods, GZIP
fallback on constant input, ZBLANK absent for integer HDUs,
seed=0 → on-disk default of 1, rejection paths (integer dtype,
no compress).  Plus 18 Rust-side unit tests anchoring the noise
estimator + per-pixel quantize against round-trip math.

*Known limitation: byte-exact heap agreement with cfitsio.*  The
fitsio cross-read tests assert **physical-pixel-value** agreement
(bit-exact decoded output), not raw-heap-byte equality.  The
two writers can pick slightly different bscale on the same input
because cfitsio's per-row diff arrays are f32 while ours are
f32 (matched on f32 path) but the cross-row averaging order
differs subtly with `qsort` stability quirks.  The decoded
result is still identical because ZSCALE/ZZERO are recorded
per-tile.  Heap-byte equality would require a more painstaking
port of cfitsio's qsort tie-breaking; not worth doing absent a
specific need.

*Unquantized float compression (`quantize=None`).*  Shipped
2026-05-23.  Float HDUs without an explicit `Quantize` config
get lossless raw-byte storage through GZIP_1 or GZIP_2 —
matching astropy's `quantize_level=0` layout exactly.

**API.**
- `f.create_image_hdu("f4", shape, compress=Gzip1(...))` —
  unquantized, lossless raw float bytes through GZIP_1.
- `f.create_image_hdu("f4", shape, compress=Gzip2(...),
  quantize=None)` — same thing, explicit; GZIP_2's byte-shuffle
  typically gives 3-5% better compression than GZIP_1 on float
  bit patterns.
- `f.create_image_hdu("f4", shape, compress=Gzip1(...),
  quantize=Quantize(...))` — opt in to lossy quantization
  (4-10× better compression at the cost of precision).
- Integer HDUs reject `quantize=` regardless of value.

The default for float HDUs is **unquantized**.  Lossy
quantization is opt-in to avoid silently throwing away
precision on scientific data.  (Before 2026-05-23 the default
was `Quantize()` = dither1; that was changed at the same time
as removing `Quantize(method='none')` since no users existed
yet.  Decision logged with the empirical algorithm-vs-
algorithm comparison that motivated it.)

**Compress requirement.**  Empirically verified (astropy
7.2.0 + each algorithm at `quantize_level=0`):

| compress= | unquantized float behavior |
|-----------|---------------------------|
| Gzip1     | bit-exact lossless ✓ |
| Gzip2     | bit-exact lossless ✓ (~3-5% better than Gzip1 on float data) |
| Rice1     | astropy writes it but the round-trip is NOT bit-exact — Rice coding on float bit patterns produces garbage |
| Hcompress1| astropy writes it but the round-trip is NOT bit-exact — H-transform is an integer wavelet |
| Plio1     | astropy hard-rejects (mask-only encoder) |

So `quantize=None` requires `compress=Gzip1(...)` OR
`compress=Gzip2(...)`.  Rice1/Hcompress1 + unquantized float
returns a `ValueError` pointing at Gzip1/Gzip2.  Plio1 +
float returns its own algorithm-specific
`NotImplementedError` (PLIO + float is rejected upstream
since PLIO never works with floats regardless of
quantization).

**On-disk format.**  Matches astropy's `quantize_level=0`
schema exactly:
- ZCMPTYPE='GZIP_1' or 'GZIP_2'.
- ZQUANTIZ + ZDITHER0 + ZBLANK all omitted (the FITS Tile
  Compression Convention says ZQUANTIZ is optional and
  defaults to NO_DITHER when absent; no quantization happened,
  so there's no dither stream and no NaN sentinel to record).
- Single-column BINTABLE: TFIELDS=1, TTYPE1='COMPRESSED_DATA',
  TFORM1='1PB' or '1QB'.  No ZSCALE, no ZZERO, no
  GZIP_COMPRESSED_DATA fallback (the primary column IS
  lossless for GZIP).
- Each tile's COMPRESSED_DATA descriptor points at a
  gzip-framed stream of the tile's raw float bytes in FITS
  big-endian order.

**Standards landscape** (corrected from prior CLAUDE.md notes
— astropy 7.2.0's actual behavior was empirically tested):

|  | FITS spec | cfitsio | astropy (7.2.0) |
|--|--|--|--|
| ZQUANTIZ for unquantized | optional; absent → defaults to `NO_DITHER` | absent OR `'NO_DITHER'` | `'NO_DITHER'` |
| ZCMPTYPE for unquantized | mandatory; must name algorithm | reflects user's choice | reflects user's choice (any algorithm — though only GZIP gives bit-exact) |
| Schema for unquantized GZIP | spec doesn't mandate | n/a (typically quantized) | single COMPRESSED_DATA column with raw GZIP'd float bytes |
| `'NONE'` as ZQUANTIZ value | NOT in spec | tolerated on read | not emitted |

**Implementation.**  The mode is detected at write time via
`is_float && TFIELDS==1` (works for both freshly-created and
reopened HDUs, since the create path always emits TFIELDS=1
for unquantized floats).  In that mode,
`write_compressed_image_data` routes through `encode_tile_int`
with the GZIP_1/GZIP_2 algorithm + float bytepix — the
encoder treats the bytes opaquely.  Single-column descriptor
emission; no ZSCALE/ZZERO/fallback.  Read-side handling is
unchanged from Phase 5: `build_quant_context` returns None
(missing ZSCALE/ZZERO columns), the decoder is called with
float bytepix, no dequantization applied.

Tests in
`tests/test_image_compressed_write_unquantized.py`:
schema validation, round-trip f4/f8 × Gzip1/Gzip2, astropy +
fitsio cross-read agreement (both directions), non-last HDU
growth, omit-kwarg semantics, rejections (Rice1/Hcompress1
generic, Plio1 algorithm-specific, removed
`Quantize(method='none')` spelling).

**Phase 9 — `CompressedImageHDU.extend(data)`.**  Shipped
2026-05-23 for integer, unsigned-int trick, unquantized-float,
and quantized-float HDUs (all 5 algorithms except PLIO_1
paired with unsigned trick).  The quantized-float path reuses
each tile's existing per-tile bscale/bzero/dither seed so
unchanged pixels round-trip with NO compounding loss — see
"Quantized-float mutation" section below.

API: `extend(data)` — no `start=` kwarg.  In-place writes to
existing tile rows are `__setitem__`'s job (re-encode affected
tiles); extend only appends along numpy axis 0.  Partial-last-tile
case is supported: the boundary tile is decoded, combined with the
first portion of new data, and re-encoded.  Old boundary-tile
heap bytes become orphans (left in place; descriptors no longer
point at them).

Implementation in `hdu_image_compressed.rs::extend_compressed_image_data`
(file-private, mirrors `write_compressed_image_data`).  Mechanics:
pre-read existing main + heap into RAM; encode boundary tiles
(decode via cache → byte-concat with new portion → encode); encode
new tile rows; grow file (shift later HDUs or set_len); write
updated main table + relocated heap + appended heap; rewrite
header (NAXIS2 + PCOUNT + ZNAXIS<last>).  Same taint contract as
write: pre-shift failures don't taint; failures inside the write
loop do.

Tests in `tests/test_image_compressed_extend.py` (46 cases):
1-D + 2-D + 3-D extends, tile-aligned and partial-last-tile,
dtype matrix (u1/i2/i4 + unsigned trick i1/u2/u4), algorithm
matrix (Gzip1/Gzip2/Rice1/Hcompress1), multiple sequential
extends, astropy cross-read, non-last HDU growth, unquantized
float, all rejection paths.

**Phase 9 — `CompressedImageHDU.__setitem__(key, value)`.**
Shipped 2026-05-23 for integer, unsigned-int trick,
unquantized-float, and quantized-float HDUs.

API: same slice surface as `__getitem__` (slice / int / ellipsis
per axis, stepped slices, mixed combinations).  RHS is a scalar
(numpy broadcasts across the selection) or an ndarray whose
shape exactly matches the selection's output shape.  Unsigned-
int trick HDUs accept the scaled dtype (e.g. u2 on a u2-trick
HDU) and reverse-transform via `normalize_compressed_input_dtype`.

Implementation in `hdu_image_compressed.rs::setitem_compressed_image`
(file-private, mirrors `extend_compressed_image_data`'s shape
but simpler — no descriptor table growth, no heap relocation).
Mechanics: for each tile that overlaps the selection (via the
same `axis_overlap` helper `__getitem__` uses), decode existing
tile → wrap in numpy (with `.copy()` since `frombuffer` is
read-only) → numpy `set_item` with the appropriate RHS portion
→ re-encode → append to heap.  Modified-tile heap bytes become
orphans (left in place; descriptors no longer reference them).
PCOUNT grows; file may grow if past the padded extent.  Same
taint discipline as extend.

Tests in `tests/test_image_compressed_setitem.py` (29 cases):
single-pixel writes (1-D and 2-D), row/col writes, multi-tile
contiguous slices, scalar broadcast, stepped slices, 3-D,
dtype matrix including unsigned trick, algorithm matrix,
unquantized float, multiple sequential modifications, astropy
cross-read, non-last HDU growth, empty-slice no-op, all
rejection paths.

**Heap orphaning trade-off.**  After many `__setitem__` calls
modifying the same tiles, the heap grows monotonically with
orphaned bytes.  Call `hdu.repack()` to rebuild the heap with
only live tiles (see "Heap repack" below for the shared
mechanism with `TableHDU.repack()`).

**GZIP compression level (`Gzip1(level=...)` / `Gzip2(level=...)`).**
Shipped 2026-05-23.  Accepts 0..=9 (zlib levels: 0 = none, 1 =
fastest, 9 = best); `None` (default) uses flate2's default of 6.

**Caveat (corrected 2026-05-29):** that default of 6 is the
zlib/flate2 library default, but it is NOT what cfitsio uses for tile
compression — cfitsio hardcodes `Z_BEST_SPEED` (level 1) in
`<cfitsio>/zcompress.c` (`deflateInit2(..., Z_BEST_SPEED, ...)`).  So
rustfits at its default writes ~18% smaller `.fz` files than cfitsio
but does more encode work (a deliberate size/speed tradeoff, not a
deficiency).  At MATCHED level 1, rustfits's GZIP_2 encode is ~2.3×
FASTER than fitsio (see the perf section).  Whether rustfits should
adopt cfitsio's level-1 default is an open question.

The level is a **write-only** parameter: the gzip stream format
itself doesn't preserve the level that produced it (the decoder
handles any level identically), and the FITS Tile Compression
Convention defines no ZNAMEn/ZVALn slot for it.  Within the same
Python session `hdu.compression.level` returns the user's value
(stored on the HDU); after close + reopen the field comes back
as `None`.

**`compress_config` field on `CompressedImageHDU`.**  Added
alongside `quantize_config` to hold the user's full
`CompressionConfigKind` (i.e., the `Gzip1(...) / Gzip2(...) /
Rice1(...) / Hcompress1(...) / Plio1(...)` they passed to
`compress=`).  Same-session `hdu.compression` returns the stored
config so write-only params (`level`, and any future additions)
round-trip via `.compression`.  At create time the stored cfg's
`tile_shape` is replaced with the resolved value (the actual
on-disk tile shape, after applying the algorithm's default-tile
heuristic) so `.compression.tile_shape` returns the real value
even when the user passed `Gzip1()` without specifying it.  For
reopened HDUs the field is `None`; `.compression` falls back to
rebuilding from header cards (level → None).

`CompressionConfigKind` was moved from `fits.rs` (file-private)
to `compression_config.rs` (`pub(crate)`) so the HDU can hold
it.  The enum gained a `gzip_level()` accessor (Some for Gzip1/
Gzip2, None for others) and a `with_resolved_tile_shape(ts)`
method used at create time.  The write/extend/__setitem__
dispatchers take `compress_config: &Arc<Mutex<Option<
CompressionConfigKind>>>` and extract `gzip_level` from it,
threading it through `IntTileCtx` / `FloatTileCtx` /
`encode_quant_boundary_tile` / `AlgorithmEncodeParams` and into
`encode_gzip1` / `encode_gzip2`.

Tests in `tests/test_image_compressed_gzip_level.py` (26 cases): config-class API
(default, repr, equality, validation), same-session round-trip,
reopen loses level, file-size proxy (level=9 ≥ 10% smaller than
level=1 on compressible data), bit-exact round-trip across all
levels × Gzip1/Gzip2, level applies to extend + __setitem__,
non-GZIP algorithms reject level kwarg.

**BLANK / ZBLANK + MaskedArray support.**  Shipped 2026-05-23
for both compressed and uncompressed image HDUs.

API:
- `create_image_hdu(..., blank=<sentinel>)` — emits `BLANK`
  (uncompressed) or `ZBLANK` (compressed integer) so reads
  with `mask_blank=True` find the sentinel.  Value is in
  PHYSICAL space (user's dtype); transformed to STORED space
  for the card when the unsigned-int trick is in play (e.g.,
  user passes `blank=65535` for `u2`, card lands as
  `BLANK=32767` because the on-disk i2 storage XORs the sign
  bit).  Rejected on float dtypes (FITS spec forbids BLANK
  on floats; NaN serves that role).
- `hdu.read(mask_blank=True)` — returns
  `numpy.ma.MaskedArray` with True at pixels whose stored value
  equals BLANK/ZBLANK.  Comparison happens in stored space
  (pre-scaling) per FITS spec.  Rejected on float ZBITPIX.
  Already worked for uncompressed; this PR enables it for
  compressed integer HDUs.  When the keyword is absent the
  return is still a MaskedArray (consistent return type) but
  with `nomask`.
- `write` / `__setitem__` / `extend` accept
  `numpy.ma.MaskedArray` input on both compressed and
  uncompressed HDUs.  Masked positions are auto-filled with
  the sentinel from the header (BLANK or ZBLANK in physical
  space) — error if the header has no sentinel and the input
  is integer.  For float HDUs, fill value is NaN (no header
  dependency).  All-False masks short-circuit (no header
  lookup needed; just unwrap to underlying data).

Implementation: single shared helper
`hdu_image.rs::unwrap_masked_input(py, data, header, is_compressed)`
runs at the top of all 6 write entry points (uncompressed +
compressed × write/extend/__setitem__).  The helper internally
parses BITPIX/ZBITPIX (image-side) and BSCALE/BZERO to
determine the fill semantics — caller passes only a one-line
preamble.  Read-side reuses
`compute_blank_mask_for_key(header, arr, key)` generalized
from the existing `compute_blank_mask`.

Tests in `tests/test_image_blank_mask.py` (27 cases): round-trip
across u1/i2/i4 × {compressed Gzip1/Gzip2/Rice1, uncompressed};
unsigned-int trick interaction; MaskedArray input on all 6
write paths; float-NaN fill; all rejection paths; astropy +
fitsio cross-read of rustfits-written ZBLANK files.

Interop notes (verified empirically): astropy 7.2.0 promotes
integer + BLANK/ZBLANK reads to f64 + NaN (rather than
returning a MaskedArray); fitsio reads the file fine but
doesn't auto-mask.  Neither tool has a clean MaskedArray write
API for compressed images — rustfits's `blank=` kwarg + auto-
fill behavior is the friendlier shape.

**Quantized-float mutation (extend + __setitem__).**  Shipped
2026-05-23.  When extending or mutating a tile that was
originally stored in the **primary** (quantized i32) column,
rustfits reuses the EXISTING per-tile bscale/bzero (read from
the tile's ZSCALE/ZZERO row entries) AND the existing dither
seed (deterministic from `row_1based` + `ZDITHER0`).  This
makes `requantize(dequantize(stored))` idempotent: unchanged
pixels in the modified tile round-trip BIT-EXACT — no
compounding quantization noise.  Verified by tests in
`tests/test_image_compressed_quant_mutation.py` (24 cases)
that compare a reference no-mutation file against a post-
mutation file across f4/f8 × {no_dither, dither1, dither2}.

Tiles originally in the **GZIP fallback** column (lossless raw
float bytes, used when the original quantize couldn't fit the
range) stay in the fallback after modification — re-encoded as
GZIP_1 of the modified raw float bytes.  No new precision loss.

**Out-of-range rejection.**  If a user writes a value that
doesn't fit the tile's existing per-tile bscale/bzero (after
the unscale arithmetic the i32 stored value would land outside
the legal range), the mutation is rejected with a clear error
message that names the three recovery options: (1) recreate
the file with `Quantize(level=N)` for a smaller N (coarser
scales admit wider ranges), (2) recreate with
`Quantize(level=-bscale)` to pin bscale explicitly, or
(3) recreate with `quantize=None` for lossless raw-float GZIP.
The error message also reports the failing pixel index, value,
and the tile's bscale/bzero so the user can size the new
scale.

Implementation in
`src/zimage/quantize.rs::requantize_float_fixed_scale` /
`requantize_double_fixed_scale` (idempotency anchored by
Rust unit tests).  The mutation dispatchers
(`extend_compressed_image_data` and `setitem_compressed_image`)
read each affected tile's primary descriptor to detect
primary-vs-fallback storage, then route through
`encode_quant_boundary_tile` which handles both cases.

**Checksum convention (CHECKSUM/DATASUM + ZHECKSUM/ZDATASUM).**
Shipped 2026-05-23.

API (matches astropy's 4-method shape; same on every HDU type):
- `hdu.add_datasum()` — compute the data-section checksum,
  update the DATASUM card (or ZDATASUM for compressed).
- `hdu.add_checksum()` — also computes CHECKSUM (or ZHECKSUM):
  the encoded complement of (header + data) so the total HDU
  checksum lands on 0xFFFFFFFF, per the FITS Checksum
  Convention.
- `hdu.verify_datasum()` — returns `True` / `False` / `None`
  (None = DATASUM card absent).
- `hdu.verify_checksum()` — same return shape.

Per HDU type:
- **ImageHDU + TableHDU**: emit CHECKSUM + DATASUM (standard
  uncompressed).
- **CompressedImageHDU**: emit ZHECKSUM + ZDATASUM (per the
  FITS Tile Compression Convention, the integrity check is
  against the *equivalent uncompressed image*, not the
  on-disk BINTABLE).  Astropy uses the same convention.
- **CompressedTableHDU** (fixed + VLA columns): same
  ZHECKSUM + ZDATASUM convention; the equivalent
  uncompressed bytes are reconstructed tile-by-tile (decode
  each (tile, col) blob to BE bytes, interleave per row)
  and fed to a streaming checksum.  For VLA columns: the
  per-tile dual-descriptor blob is GZIP-decompressed, the
  ORIGINAL P/Q descriptors are copied into the tile's main
  buffer at the column's offset, and per-cell metadata
  `(orig_offset, vlalen, cvlalen, cvlastart, col_idx)` is
  collected.  After the tile walk feeds all main buffers,
  cell metadata is sorted by `orig_offset` and walked in
  order: gap bytes between cells (zero padding from sparse
  layouts) are fed as zeros in <=64 KiB chunks, then each
  cell's compressed bytes are read from the heap and
  decompressed to BE (with cfitsio's uncompressed-fallback
  branch when `cvlalen == vlalen * elem_size`) and fed to
  the stream.  Trailing zeros are fed to reach ZPCOUNT.
  Peak memory bounded at one tile's main buffer + one
  decompressed dual-desc blob + one per-cell decoded buffer
  + the cell-metadata vector (~40 bytes per non-empty VLA
  cell), regardless of file size.  Compressed-VLA-X cells
  are rejected (matches the read-path scope: X inner letter
  isn't supported in compressed VLA columns).

Manual semantics — re-run `add_checksum()` after any mutation
(write / __setitem__ / extend / append).  Auto-update on
every write would be expensive and surprising; cfitsio and
astropy both defer to the user.

Cross-tool agreement:
- astropy + fitsio both verify uncompressed rustfits-written
  CHECKSUM/DATASUM as valid (cfitsio-byte-exact encoding).
- astropy's `verify_checksum` for **compressed** HDUs has its
  own internal bug (TypeError on
  `_compute_checksum(None)`) that triggers on its own writes
  too; we don't cross-verify compressed against astropy.
  Our self-verify catches corruption correctly.

Implementation in `src/checksum.rs` — direct port of cfitsio's
`checksum.c` (`ffcsum` / `ffesum` / `ffdsum`).  Byte-exact
agreement with cfitsio anchored by Rust unit tests
(`encode_matches_cfitsio` reference values were generated by
compiling cfitsio's `ffesum` standalone — see commit log).
Shared HDU-level scaffold in `hdu_image.rs`
(`checksum_hdu_add_*` / `checksum_hdu_verify_*`).  Both the
uncompressed scaffold and the compressed-table
implementation **stream** the data section (1 MiB chunks for
uncompressed; per-tile decode for compressed-table) through
the shared `ChecksumStream` accumulator in `src/checksum.rs`,
so peak memory is bounded at a per-chunk constant regardless
of file size.  The compressed-image checksum
(`read_uncompressed_image_be_bytes` in
`hdu_image_compressed.rs`) still materializes the full
ndarray — predates the streaming rule; flagged as a
follow-up perf item.  All three variants build a synthetic
equivalent-uncompressed-HDU header for the ZHECKSUM
computation.  Reuses the `rewrite_header_to_disk` helper
from `header.rs` (now `pub(crate)`) so a header that grows
past its reserved blocks gets in-place expansion (same
machinery the FITSHeader edit path uses).

`ChecksumStream` (in `src/checksum.rs`) is the streaming
accumulator that makes per-chunk feeding correct:
`compute_checksum_bytes` zero-pads partial trailing 4-byte
groups, so calling it on consecutive sub-multiples-of-4
chunks produces a different result than one call on the
concatenation.  The accumulator buffers up to 3 leftover
bytes between feeds so callers (the per-tile compressed-
table walk; the 1 MiB chunked uncompressed-data reader) can
stream arbitrary chunk sizes safely.  `finish()` applies the
tail-padding rule one last time on the final remainder.

The `card_string` helper was tightened in the same commit to
match the cfitsio/astropy convention of putting `/` at column
32 (1-indexed) for fixed-format string-valued cards.
Previously we placed `/` immediately after the closing quote,
which is spec-legal but produced different byte sequences
than cfitsio — and astropy's `verify_checksum` does a
byte-exact comparison of the encoded sum, so the offset
mattered.  Fix is purely formatting; values unchanged.

Tests in `tests/test_hdu_checksum.py` (19 cases) — round-trip on
all three uncompressed/compressed-image HDU types, add_datasum
independence, None when absent, corruption detection on
ZDATASUM card + heap byte, astropy/fitsio cross-verify
(uncompressed), algorithm matrix (compressed image), ZDATASUM
= uncompressed-equivalent DATASUM.
Plus `tests/test_table_compressed_checksum.py` (22 cases) for
the table side: round-trip with various nrows/ztilelen
combinations (single tile, partial last tile, parametrized
matrix), add_datasum-only, verify-None-when-absent, ZDATASUM
card corruption detection, heap-byte corruption raising
ValueError (mirrors the image-side test), ZDATASUM equals
the DATASUM of the equivalent uncompressed table, funpack
cross-tool (cfitsio decompresses our file → the reconstructed
DATASUM matches our ZDATASUM), same-handle vs reopen parity,
refresh-after-setitem (re-running add_checksum on a mutated
table picks up the new value), and VLA-bearing tables across
six cases: VLA-only round trip, mixed fixed+VLA round trip,
empty cells, ZDATASUM equals the uncompressed-equivalent
DATASUM (anchors the synthetic-heap walk), ZDATASUM card
corruption detected, and funpack-decompresses VLA file →
DATASUM matches ZDATASUM (strongest interop check).

