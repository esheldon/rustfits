# Performance — ZIMAGE chunked-read profiling history + perf benchmark suite

*Extracted from `CLAUDE.md` to keep the always-loaded file lean.
Read this when working on `perf/`, debugging a perf regression,
or extending the benchmark suite.  The CLAUDE.md "Performance"
entry carries the one-paragraph summary + a pointer here.*


Reading a 1.49-billion-pixel f8 GZIP_2 ZIMAGE file (~12 GB;
per-user fixture, not committed — see "Performance / large
fixture files" under "Build / dev workflow" for how to obtain or
synthesize an equivalent) in chunks via `f[1][lo:hi]` slicing.

**The `perf/` benchmark suite.**  Standalone scripts under `perf/`
(run directly, NOT pytest tests — the `perf-*.py` names stay out of
pytest collection) compare rustfits vs fitsio.  Shared methodology
(see `perf/_harness.py` and the per-script docstrings): a release
build is required (a debug build reads ~7× slower and reports rustfits
as the loser); a fresh handle is opened inside every timed iteration
(so fitsio's forever-cache can't masquerade as decode speed against
rustfits's bounded LRU); and the harness warmup pass primes the OS
page (FS) cache so timing measures decode, not cold disk I/O.
**Write benches use a fresh FILE per iter** (via `h.fresh_path`,
not just a fresh handle to the same path) — overwriting the same
large file in a tight loop triggers a kernel page-cache penalty that
masks rustfits's actual speed (one-time finding 2026-05-29; flipped
the 1-D image write from 0.64× SLOW to 1.07× FASTER).
Synthetic data is tuned to reproduce a real file's *timing ratios*,
not its compression ratio.  Scripts:
- `perf-compressed-image-read-healsparse.py` — 1-D GZIP_2 lossless
  (the win documented below; validated against a real 706 MB map).
- `perf-compressed-image-read-random.py` — 1-D GZIP_2 on incompressible
  data (the gap narrows without runs to exploit, but rustfits still wins).
- `perf-compressed-image-read-des.py` — 2-D RICE_1 + dither2 lossy
  (DES-like), random 32×32 postage-stamp reads + whole-file tile-order
  band walk.  `_read2d.py` holds the shared 2-D read regimes;
  `_compread.py` the shared 1-D runner.
- `perf-compressed-2d-isolation.py` — the codec isolation sweep below.

**2-D RICE decode rewrite (2026-05-29).**  The DES-like 2-D lossy
workload originally showed rustfits RICE decode at 0.38× (~2.6×
slower than cfitsio); the isolation sweep
(`perf-compressed-2d-isolation.py`) pinned the cost to the decoder
itself, not quantization or tile assembly.  Three iterative fixes
landed (each its own commit), arriving at **rustfits faster than
cfitsio across the board**:

| isolation case | original | cfitsio-port | direct-bytes | **u64 buffer (current)** |
|---|---|---|---|---|
| rice i4 lossless [whole]  | 0.39× SLOW | 0.91× | 0.95× | **1.30× FASTER** |
| rice f4 quant [whole]     | 0.39× SLOW | 0.83× | 0.87× | **1.11× FASTER** |
| rice i4 lossless [stamps] | 0.62× SLOW | 1.42× | 1.46× | **1.94× FASTER** |
| rice f4 quant [stamps]    | 0.62× SLOW | 1.22× | 1.29× | **1.58× FASTER** |
| rice f4 quant t=1000 [whole] | 0.42× SLOW | 0.77× | 0.81× | **1.01× (~par)** |

RICE i4 decode throughput: ~236 MB/s → **~791 MB/s** (3.3× faster
than the original; ~1.3× cfitsio's ~615 MB/s).  The DES bench (the
original workload that surfaced the slowdown) now reports **1.12×
FASTER** on whole reads and **1.58× FASTER** on stamps.

The three fixes:

1. **cfitsio-style 32-bit bit buffer + LZCNT unary** (commit
   `97cfbc1`).  Replaced our generic BitReader (per-bit
   `read_bits(1)` for unary, byte_pos+bit_pos+byte-spanning
   `read_bits` loop) with cfitsio's `fits_rdecomp` shape: single
   u32 bit buffer + `nbits` counter refilled 8 bits at a time,
   `u32::leading_zeros` (LZCNT on x86-64-v2) for unary counting.
   Output written directly to a typed `i32` scratch by index
   instead of `Vec<i64>::push` + cast.

2. **Direct-to-bytes output** (commit `6122d2d`).  Made the
   decoder generic over a `PixelWrite` trait with zero-sized
   witness types (`I8Out`/`I16Out`/`I32Out`/`I64Out`); dispatch
   on ZBITPIX at the outer layer.  Compiler monomorphizes the
   inner loop's per-pixel store to the exact target width — no
   intermediate `Vec<i32>` alloc, no cast pass.

3. **u64 bit buffer + 32-bit refills** (current).  Replaced the
   u32 buffer with an MSB-aligned u64 buffer that refills 32 bits
   at a time via `u32::from_be_bytes` when nbits ≤ 32 AND ≥ 4
   bytes remain.  4× fewer refill ops, and unary counting becomes
   a single `u64::leading_zeros` across up to 64 bits in one shot
   (vs cfitsio's per-byte refill + 256-entry LUT).  Bit-buffer
   invariant: `buf` has `nbits` valid bits in positions
   `63..(64-nbits)`; positions below are zero.  This makes
   take_bits a single shift.

Implementation lives in `decode_rice_int_u32buf` (the fast core
for BYTEPIX in {1, 2, 4} — 99% of real RICE files) and a `refill`
helper.  BYTEPIX=8 (no canonical writer produces it; we accept on
read for completeness) falls through to `decode_rice_slow_i64`
because 64-bit raw diffs don't fit in a 64-bit bit buffer with any
useful headroom for refills.  Byte-exact tested by the existing
RICE round-trip suite (all algorithms × all dtypes × encoder
cross-check with fitsio).

**Write/encode benchmarks (2026-05-29).**  The write side
(`perf-compressed-image-write-healsparse.py` for 1-D GZIP_2,
`perf-compressed-image-write-des.py` for 2-D RICE+dither2) completes
the encode/decode × GZIP/RICE matrix:

| compression path | rustfits vs fitsio |
|---|---|
| GZIP_2 decode (1-D healsparse) | 1.8–45× faster |
| GZIP_2 encode (matched level 1) | **2.32× faster** |
| RICE decode (2-D)  | **1.11–1.30× FASTER** whole / **1.58–1.94× FASTER** stamps (rewritten 2026-05-29) |
| RICE encode        | 0.99× (≈ par) |

- **GZIP_2 encode is rustfits-faster once the level is matched.**  A
  first run looked 0.48× (slower), but that was a level mismatch:
  rustfits defaults to level 6 while cfitsio uses `Z_BEST_SPEED` (1).
  At matched level 1 rustfits encodes ~2.3× faster (833 vs 359 MB/s).
  The write benchmark defaults to `--level 1` for the fair comparison.
  Lesson for write comparisons: match the gzip level (cfitsio = 1),
  and watch that fitsio's float GZIP **lossy-quantizes by default** —
  pass `qlevel=0` for a lossless comparison.
- **RICE encode is ~par** (0.99×) even though RICE *decode* is the slow
  path — the two RICE directions need separate attention; only decode
  is a problem.

**Extend / bounded-memory build (2026-05-29).**
`perf-compressed-image-extend-healsparse.py` measures building a 1-D
GZIP_2 map incrementally via `CompressedImageHDU.extend` — a capability
healsparse doesn't use today (it holds the whole map in RAM and writes
once via fitsio).  fitsio can't append to a compressed image, so this is
a rustfits self-characterization (per-build wall time + peak `ru_maxrss`,
each build in its own subprocess).  Building a **1 GB** map:

| regime | build | peak RSS |
|---|---|---|
| fitsio write-once | 3.05 s | 1,225 MB |
| rustfits write-once | 2.03 s | 1,146 MB |
| rustfits extend, C=1 tile (K=128) | 2.96 s | **129 MB** (8.9× less) |
| rustfits extend, C=8 tiles (K=16) | 1.41 s | **185 MB** (6.2× less) |

- **The win is peak memory (~9× less)**: extend builds maps that don't
  fit in RAM; write-once needs the whole array resident.
- **Time is competitive — even faster with larger chunks** (C=8 tiles is
  0.70× write-once, the bounded chunks avoiding full-array memory
  pressure).  `extend` relocates the growing compressed heap each call,
  so many tiny chunks cost more (C=1 tile = 1.46×, 128 relocations) — a
  clean chunk-size = speed/RAM tradeoff the caller tunes by memory
  budget.  (Heap-relocation-per-extend is O(current compressed size), so
  build-via-K-extends is ~quadratic in K; large chunks keep K small.)

**Uncompressed image reads (2026-05-29).**  `perf-image-read-1d.py`
(1-D f8) and `perf-image-read-2d.py` (2-D f4) benchmark plain
(uncompressed) reads — data content is irrelevant (raw bytes), so no
healsparse/random distinction.  After the byteswap fix (see below),
all cases ≥ par:

| regime | rustfits vs fitsio |
|---|---|
| 1-D chunk 1000 (partial) | **1.90× FASTER** |
| 1-D chunk 50000 (whole)  | **1.17× FASTER** (9,133 MB/s) |
| 1-D whole `.read()`      | **1.01× ≈par** (2,379 MB/s) |
| 2-D stamps 32×32 (x1000) | **2.43× FASTER** |
| 2-D whole `.read()`      | **1.02× ≈par** |

**The fix (2026-05-29):** `byteswap_in_place` (in `src/common.rs`)
used `chunks_exact_mut(itemsize)` + `chunk.reverse()`, which has a
dynamic chunk length and doesn't auto-vectorize — the per-byte
reverse loop emits scalar code.  Specialized it to dispatch on
itemsize and use `chunks_exact_mut::<uN>` + primitive `swap_bytes`
(a portable intrinsic LLVM lowers to BSWAP on x86-64, REV on ARM64,
`rev8` on RISC-V Zbb).  Modern LLVM auto-vectorizes the resulting
fixed-stride loop to the platform's byte-shuffle SIMD (PSHUFB on
AVX2, NEON REV/TBL, etc.).

Cost dropped from ~75 ms to ~16 ms for a 512 MB f8 byteswap (~5×).
Every call site benefits — image read/write, table read/write,
compressed read/write all picked up a measurable gain from this one
function.  itemsize=16 (c16 = 2× f8 complex) routes through the
8-byte path because the swap is *per-component*, NOT a full 16-byte
reversal (which would swap real and imag).

Pre-fix history (committed `3e21d03`): bulk reads were 0.56–0.83×,
documented as "a third optimization lead alongside RICE decode."
That lead is now closed; all uncompressed image read regimes are
≥ par.

**Uncompressed image writes + extend (2026-05-29).**
`perf-image-write-1d.py` / `-2d.py` and `perf-image-extend-1d.py`.

| regime | rustfits vs fitsio |
|---|---|
| 1-D f8 write (512 MB) | **1.40× FASTER** (3,029 vs 2,166 MB/s) |
| 2-D f4 write (64 MB)  | **1.40× FASTER** (2,709 vs 1,937 MB/s) |

- **Three fixes landed to get here.**  (1) The byteswap-copy in
  `write_image_data` was capped at 1 MiB scratch and chunked within
  each strip — peak RSS for a 1 GB build dropped from 2,101 MB to
  1,078 MB, and the bench-time effect was also material.  (2) The
  bench methodology was flipped to **fresh file per iter** (via
  `h.fresh_path`) instead of overwriting the same path in a tight
  loop.  Overwriting the same large file repeatedly turned out to
  trigger a kernel page-cache penalty that was masking rustfits's
  actual speed.  (3) `byteswap_in_place` was specialized on itemsize
  (commit `3e21d03`; ~5× faster), flipping the 2-D write from 0.85×
  to 1.40× and bumping the 1-D from 1.07× to 1.40×.
- **Uncompressed extend is a bounded-memory win.**  Building a 1 GB
  map: extend uses **16× less RAM** (67 MB vs 1,078 MB) at C=1 MiB
  chunks, and is **0.54× the wall time of write-once** (it sidesteps
  the whole-array byteswap copy entirely).  Time is ~flat across
  chunk size (no compressed heap to relocate → O(N) linear), unlike
  the compressed extend's ~quadratic chunk penalty.

So uncompressed write is no longer a tall pole.  Every uncompressed
image read+write regime now favors rustfits.  GZIP_2 read+encode,
RICE read (post-rewrite) + encode, uncompressed table read+write,
uncompressed extend, and uncompressed image read+write all favor
rustfits.  No remaining sub-par bench in the matrix.

**Uncompressed BINTABLE reads (2026-05-29).**  `perf-table-read.py`
reads a deliberately type-exhaustive 34-column catalog (every scalar
type, f4/f8 fixed sub-arrays 1-D & 2-D, both S and U fixed + VLA
strings, an f4 VLA; see `_data.catalog_arrays`).  After the VLA
heap-batching fix, all four regimes win:

| regime | rustfits vs fitsio |
|---|---|
| whole table          | **1.23× FASTER** |
| column subset (3/34) | **2.04× FASTER** |
| row slice            | **1.32× FASTER** |
| scattered rows       | **2.56× FASTER** |

**The fix (2026-05-29).**  Pre-fix, whole-table and row-slice both
trailed fitsio (~0.83-0.88×).  Isolating with VLA columns excluded
showed the VLA path was the bottleneck — fixed-column-only reads
were already 1.23× FASTER than fitsio.  `heap_pass` in
`src/hdu_table/read.rs` was doing one `seek + read_exact` per VLA
cell (1.5M cells in the test = 1.5M syscalls).  Replaced with a
chunked heap reader: cells are sorted by heap_offset, the
contiguous extent containing each cell is loaded in bounded chunks
of 1 MiB (the same convention every other large-data path uses —
see `common.rs::CHUNK`), and each refill greedy-extends to cover
as many following cells as fit in the budget.  Sparse cells
(scattered-row reads with cells far apart) naturally collapse to
~per-cell reads of just what's needed; dense reads (whole-table /
contiguous slice) collapse to ~1 syscall per 1 MiB of heap.  Peak
per-call memory bounded at the chunk size regardless of heap size.

- **Selective access still favors rustfits strongly** — column
  projection (2.04×) and scattered object lookups (2.56×), the
  dominant catalog patterns.
- A 290 MB (500k-row) file gives the same ratios as a 1.1 GB
  (2M-row) one, so the smaller file is representative for iteration.
- Cross-tool VLA caveat: fitsio reads VLA columns padded-to-max by
  default (`vstorage="fixed"`); the benchmark reads fitsio with
  `vstorage="object"` to match rustfits's object cells.  The timing
  barely moves (VLA cells are tiny vs the fixed-column bulk), but it
  makes the comparison apples-to-apples.

**Uncompressed BINTABLE write (2026-05-29).**  `perf-table-write.py`
writes the same catalog (fitsio writes the VLA object columns as true
1PA/1PE, like rustfits).  rustfits is **2.60× FASTER** (0.94 s vs
2.44 s; 326 vs 125 MB/s).  fitsio's table write (mixed types + VLA
heap) is comparatively slow at 132 MB/s vs its ~1870 MB/s image
write rate — the per-column / per-VLA-cell overhead in cfitsio's
table writer is the bottleneck on that side.

**Compressed BINTABLE (ZTABLE) read (2026-05-29).**
`perf-table-compressed-read.py` is a rustfits SELF-comparison —
**fitsio's Python API does not decompress ZTABLE** (it returns the raw
compressed structure / wrong values) and astropy can't read it either,
so there's no cross-tool Python comparison.  rustfits is uniquely able
to read ZTABLE transparently from Python (cfitsio's `funpack` CLI is the
only other decompressor).  So this measures rustfits ZTABLE read vs its
own uncompressed read of the same 34-column catalog (complex now
round-trips — issue #8 fixed):

| regime | ZTABLE / uncompressed |
|---|---|
| whole table        | 1.35× (decompression tax) |
| column subset (3)  | 1.07× (≈ free) |
| row slice          | 1.39× |
| scattered rows     | **~74× slower** |

- **Column projection is nearly free** — decompresses only the selected
  columns' tiles.  Bulk reads pay a modest ~1.3× decompression tax.
- **Scattered random-row reads are catastrophic (~74×)**: each random
  row forces decompressing its whole tile (all columns).  This is
  *inherent*, NOT cache thrashing: the benchmark sizes the tile cache to
  hold the whole table (cache-neutral, like the image stamp test) and
  the number is unchanged, because the read planner already decompresses
  each touched tile once per call.  The default `ZTILELEN` is large
  (~17k rows → ~30 tiles for 500k rows), so 2000 random rows touch ~all
  tiles → ~whole-table decompress.  (Contrast the image stamp test:
  ~10k tiles, stamps touch only a fraction.)  ZTABLE is for
  bulk/columnar access, not random row lookups.
- **Smaller tiles don't help — they hurt on every axis.**  Sweeping
  `--ztilelen` (200k rows, 2000 scattered): 128 → scattered 59×, whole
  2.0×, compression 1.1×; 2048 → 32× / 1.45× / 1.2×; default(~17k) → 28×
  / 1.31× / 1.2×.  Bigger tiles win monotonically on scattered, bulk,
  AND compression.  A ZTABLE stores each (tile, column) as its own gzip
  blob, so with 34 columns `ztilelen=128` over 200k rows = ~53k tiny
  blobs; scattered reads then do ~36k tiny gzip decompresses and the
  **per-blob fixed overhead dominates** (slower despite decompressing
  *less* data).  So shrinking tiles is not a fix for scattered access —
  it could be a future optimization lead (lower per-blob overhead), but
  as implemented, larger tiles are better everywhere.
- Synthetic random data compresses only ~1.2× here; real catalogs
  compress far more (the actual reason to use ZTABLE).

**Compressed BINTABLE (ZTABLE) write (2026-05-29).**
`perf-table-compressed-write.py`, same self-comparison shape (no
high-level fitsio/astropy ZTABLE writer): rustfits ZTABLE write vs its
own uncompressed write of the same catalog.  ZTABLE write is **~26×
slower** (28.0 s vs 1.1 s; 11 MB/s) — the per-(tile, column) transpose +
compress over 34 columns, plus the VLA dual-descriptor encode, is
expensive.  A clear encode-bound optimization lead if ZTABLE write
throughput ever matters.

**Current state (release builds; 2026-05-26) — GZIP_2 1-D chunked
read:**

- **Big chunks (50k rows)**: rustfits is **3.2× FASTER** than
  fitsio.  `decode_gzip2`'s remaining ~48% of slice time is now
  ~100% the flate2/miniz_oxide inflate itself — the physics floor.
- **Small chunks (1k rows)**: rustfits is **40× FASTER** than
  fitsio (up from 24× after the Phase 2 header-meta cache
  landed).  fitsio caches all decoded tiles forever (memory
  bloat); rustfits has an LRU AND a per-HDU parsed-meta cache
  that eliminates the 8+ linear card scans per slice call.

**Note on the comparison baseline.**  All ratios are vs **fitsio**
(the Python wrapper around cfitsio), not direct cfitsio.  fitsio
carries its own per-call wrapping overhead — Python-level argument
marshalling, ndarray construction, etc. — that likely inflates
these numbers relative to a direct-cfitsio comparison.  The
realistic comparison for users is fitsio, so that's what the
benchmark uses; a direct-cfitsio comparison would be worth doing
some day to attribute the gap more precisely.

**How the gap closed.**  Four sequential profile-and-fix passes
plus the header-meta cache:

1. **`tile_origin_and_shape` stack-allocation** (commit a85583f).
   The function did three `vec![0u64; d]` per call for what is
   pure arithmetic on 2-element arrays.  Replaced with caller-
   provided `[u64; MAX_NAXIS]` stack buffers (MAX_NAXIS=8).
   Closed small-chunks 1.4× slower → 1.05× of cfitsio.
2. **Direct-write to output buffer** (commit a85583f).  Old per-
   tile path was `PyBytes::new + frombuffer + reshape + set_item`
   — four Python round-trips and a full Vec→PyBytes copy per
   tile.  New `strided_copy_c_contig_to_c_contig` helper writes
   tile bytes directly into the output ndarray's buffer via
   `RawBuffer::acquire_writable`.  Stepped slices and int-
   collapse axes still take the old PyBytes path.  No new dep.
3. **`OverlappingTileRange` iterator** (commit a85583f).  Old
   loop walked `0..n_tiles` per chunk and skipped non-overlapping
   tiles inside the body.  On row-stripe layouts (fpack default)
   this is most of the work.  Pre-computes per-axis
   `(tc_first, tc_extent)` and yields a tight iteration over
   only overlapping tiles in BINTABLE row-major order.
4. **Eliminate alloc_zeroed in unshuffle/shuffle** (commit
   d7edc54).  `vec![0u8; n_pixels * bytepix]` → `Vec::with_capacity
   + spare_capacity_mut + MaybeUninit::write + unsafe { set_len }`.
   The zero-init pass was 17.67% of total profile on big chunks.
   Took big-chunks 2.8× → 3.2× faster.
5. **Header-meta cache (Phase 2 of header-derived metadata
   caching).**  Per-call re-parsing of ~8 header fields
   (ZCMPTYPE, ZNAXISn, ZTILEn, ZNAMEn/ZVALn, TFORMn, TTYPEn,
   ZBLANK, BSCALE/BZERO) collapsed to one Mutex lock + Acquire
   load + Arc clone via the cached `CompressedImageMeta`.  Took
   small-chunks 24× → **40× faster than fitsio** — the cache hit
   eliminates ~40% of the per-call overhead.  See "Header-derived
   metadata caching" section below for the design (CardsWriteGuard
   foundation in Phase 1, parsed-meta cache in Phase 2).

**Profiling gotchas worth remembering.**

- numpy on import spawns OpenBLAS worker threads that
  busy-spin even when idle.  They'll dominate any flamegraph
  100% unless you set `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` in the env.
- Even with `OPENBLAS_NUM_THREADS=1`, Python interpreter
  startup dominates short workloads.  Scale the benchmark up
  until the read loop takes ≥30s of wall time so it dominates
  the sampling window.
- Debug-build profiles are misleading — the original "3×
  slower" observation in this file came from a debug build.
  Always profile with `maturin develop --release` (debug
  symbols stay via `[profile.release] debug = "line-tables-only"`
  in Cargo.toml).
- The per-user 12 GB GZIP_2 f8 fixture isn't committed.  The
  bench script at `bench_chunked_read.py` (outside the repo)
  cycles `lo = (i * CHUNK_ROWS) % (n_rows - CHUNK_ROWS)` over
  100k chunks of 1k rows for small-chunks, 1k chunks of 50k
  rows for big-chunks.  Cycling pattern is what amortizes the
  LRU tile cache — sequential reads wouldn't get the same
  cache amplification.

