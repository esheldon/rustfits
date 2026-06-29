# Upstream bugs we've found (cfitsio / fitsio / astropy)

This file is the single inventory of bugs, crashes, and
misbehaviors rustfits has found in the **upstream** tools —
cfitsio (the C library), its `funpack` / `fpack` CLIs, the
`fitsio` Python wrapper, and `astropy.io.fits`.  It does NOT
list bugs that were in rustfits and got fixed (those live in
git history + the roadmaps in `CLAUDE.md`).

Each entry records: the tool, the trigger, the symptom, where
it's documented/pinned in this repo, how rustfits works around
it, and whether it's been filed upstream.  When you fix or file
one of these, update its **Upstream status** line.

Filed upstream so far:

- cfitsio **#134** — PA-VLA `funpack` crash (entry 2; also rustfits
  issue #9).  Pure-C repro in `tools/cfitsio-repro/`.
- cfitsio **#135** — GZIP_2 complex-column `funpack` rejection
  (entry 4).  Pure-C repro in `tools/cfitsio-repro/`.
- cfitsio **#136** — PLIO_1 small-image compression-buffer
  under-allocation (entry 1; the C-level root cause of the macOS
  crash also reported as fitsio #496).  Pure-C repro in
  `tools/cfitsio-repro/`.
- fitsio **#496** — macOS compressed-image-write crash (entry 1;
  root-caused to cfitsio #136).

Everything else is documented inline at the cited locations.

---

## cfitsio (C library) — crashes / memory corruption

### 1. macOS compressed-image-write crash (`ffbinit` bad-free / PLIO_1 abort)

- **Tool:** cfitsio on macOS, surfaced via the fitsio Python
  wrapper.
- **Trigger:** writing a compressed image on macOS, at
  `tests/test_image_compressed_accessors.py::test_other_compression_types_dispatched`.
  Two manifestations, almost certainly **one underlying bug** seen
  under two cfitsio builds:
  - **conda-forge binary:** the *first* compressed-image write
    (`PyFITSObject_create_image_hdu` path), both py3.12 and py3.14.
  - **pip source build:** specifically the `compress="PLIO_1"` +
    default-qlevel *create* path, py3.12 only, ~50% hit rate.  The
    read-side path (`compress="PLIO"` + explicit `qlevel=None`) does
    NOT trigger it.
- **Symptom:** conda binary → libmalloc bad-free inside cfitsio's
  `ffbinit`; source build → bare `Fatal Python error: Aborted` with
  the stack entirely inside `_fitsio_wrap.create_image_hdu`.  The
  two signatures are plausibly the same heap overrun caught at
  different points by two allocators.
- **Documented:** `CLAUDE.md` §§ "CI: macOS fitsio workaround" +
  "macOS CI flakes — resolved 2026-06-01";
  `tests/test_image_compressed_accessors.py`
  (`_SKIP_FITSIO_PLIO_ON_MACOS`); pure-C repro
  `tools/cfitsio-repro/plio_small_image_overflow.c` (ASAN-confirmed).
- **Workaround:** CI replaces the conda-forge fitsio with a pip
  source build on the macOS legs only (dodges the `ffbinit`
  variant); a skipif mark on just the PLIO_1 parametrize case
  dodges the source-build variant.  PLIO read coverage on macOS is
  preserved via `test_image_compressed_read_plio.py` (non-crashing
  path).
- **Root cause (cfitsio #136):** PLIO_1's compression work buffer
  is under-allocated for very small tiles.  The crashing test
  writes a 4×4 image with 2×2 tiles, so each PLIO tile has
  `nx = 4` and cfitsio sizes the buffer at
  `nx * sizeof(int) = 16` bytes — too small for PLIO's framing
  overhead.  The encoder then writes a few bytes past the 16-byte
  region: macOS's nano allocator catches the overrun (~50%,
  guard-layout dependent), glibc silently tolerates it (hence
  Linux-clean).  This subsumes BOTH manifestations above — the
  conda-forge `ffbinit` bad-free and the source-build
  `Fatal Python error: Aborted` are the same overrun caught by
  different allocators at different points.  Proposed upstream
  fix: floor the allocation at 32 bytes
  (`if (nx * sizeof(int) < 32) return 32;`) in `imcompress.c`.
  The under-allocation itself is platform-independent — provable
  on Linux under ASAN even though only macOS aborts on it.
- **Upstream status:** root-caused and filed as cfitsio **#136**
  (https://github.com/heasarc/cfitsio/issues/136) with a proposed
  patch; previously reported as fitsio **#496**
  (https://github.com/esheldon/fitsio/issues/496).  The
  `_SKIP_FITSIO_PLIO_ON_MACOS` mark + the macOS source-build of
  fitsio stay in place until a cfitsio release carrying the #136
  fix reaches the macOS CI legs; the skip can then be dropped and
  the PLIO_1 case re-enabled.

### 2. String-VLA (PA/QA) ZTABLE columns crash `funpack`

- **Tool:** cfitsio table (de)compression, surfaced via `funpack`.
- **Trigger:** a variable-length string column (`1PA` / `1QA`) in a
  tile-compressed table (ZTABLE).
- **Symptom:** `funpack` crashes with heap corruption
  (`realloc(): invalid next size` / `mremap_chunk(): invalid
  pointer`, SIGABRT).
- **Key fact:** cfitsio crashes on its **own** output — write an
  uncompressed table with a PA column, `fpack -table` it (cfitsio
  compresses the PA column), then `funpack` → crash.  So the defect
  is reproducible without rustfits or fitsio's high-level API.
- **Root cause (ASAN, 2026-06):** built cfitsio with
  `-fsanitize=address` and ran the repro through the instrumented
  `funpack`.  ASAN reports a **heap-buffer-overflow READ of 32768
  bytes, 0 bytes past a 24000-byte region** allocated in
  `fits_uncompress_table`; the over-read happens in zlib
  `inflate` → `updatewindow` called from `uncompress2mem_from_mem`
  ← `fits_uncompress_table` (call path
  `funpack.c:23` → `fpackutil.c` → `fits_uncompress_table`).  i.e.
  cfitsio **under-allocates the decompression output buffer** for
  the variable-length-string column and zlib's window copy runs off
  the end.  Build recipe + full trace in
  `tools/cfitsio-repro/README.md`.
- **Documented:** rustfits **issue #9**
  (https://github.com/esheldon/rustfits/issues/9);
  pure-C repro `tools/cfitsio-repro/pa_vla_funpack_crash.c`;
  `tests/test_table_compressed_write_vla.py`
  (`test_pa_vla_ztable_rustfits_roundtrip` proves rustfits is
  correct; `test_funpack_pa_vla_ztable_cfitsio_crash_documented`
  is the skip-documented repro).
- **Workaround:** none needed in rustfits (our PA-VLA ZTABLE
  read/write is correct).  Interop limitation: PA-VLA ZTABLE files
  are not funpack-readable with current cfitsio.
- **Upstream status:** filed as **cfitsio #134**
  (https://github.com/heasarc/cfitsio/issues/134); rustfits issue
  #9 open.

### 3. Arch-dependent dequantization (arm64 vs x86_64)

- **Tool:** cfitsio (compiler codegen — FMA contraction of the
  `value*scale + zero` dequant), via fitsio.
- **Trigger:** reading quantized-float compressed images on **arm64**
  (both macOS arm64 AND Linux aarch64) vs x86_64.
- **Symptom:** slightly different decoded float values on arm64 (max
  relative diff ~9.5e-6 on normal-magnitude values; near-zero values
  inflate the ratio but their ~1e-16 absolute diff is atol-bounded).
  rustfits's Rust dequant bit-matches x86_64 cfitsio.
- **It's arch, not OS.**  Originally read as "macOS vs Linux" because
  macos-latest is arm64 and Linux CI was x86_64 — the two confounded
  OS with arch.  Adding a Linux **aarch64** test leg (`ubuntu-24.04-
  arm`) reproduced the exact same divergence on Linux, isolating the
  cause to the arm64 arch (cfitsio's C fuses the multiply-add into a
  single-rounding FMA; rustfits's strict-IEEE Rust rounds twice).
- **Isolation (which side drifts):** empirically fitsio's, not
  rustfits's.  `test_rustfits_quantized_bytes_stable_across_os`
  (in `tests/test_image_compressed_write_quantize.py`) pins
  rustfits's written `.fz` bytes AND decoded f4 bytes to x86_64-
  captured SHA-256 goldens across the (algorithm, dither) matrix, and
  **passes on aarch64** — proving rustfits is byte-identical across
  arches, so the divergence the cross-check tolerates is entirely on
  fitsio/cfitsio's side.  (All supported targets are little-endian,
  so the byte compare is direct.  The encode pin relies on the
  committed `Cargo.lock` pinning miniz_oxide for the gzip payloads.)
- **Documented:** `tests/test_image_compressed_read_quantize.py` and
  `tests/test_image_compressed_write_quantize.py` — the fitsio
  cross-check tolerance is gated on `_FITSIO_FP_LOOSE`
  (`platform.machine()` arm64/aarch64, or any macOS), with
  `_FITSIO_FP_RTOL` / `_FITSIO_FP_ATOL`, plus the
  `test_rustfits_quantized_bytes_stable_across_os` byte pin.
- **Workaround:** loosen the fitsio cross-check on arm64 (and macOS
  generally) so the fitsio-side divergence is tolerated but a
  regression beyond the bound surfaces; x86_64 stays strictly
  bit-exact; the byte test pins rustfits's side exactly everywhere.
- **Upstream status:** not a clean bug to file (numerical
  reproducibility in cfitsio, not a defect); documented for our own
  sake, with the rustfits side empirically ruled out.

---

## cfitsio (C library) — wrong output / self-incompatibility

### 4. GZIP_2 table decompressor can't read complex (C/M) columns it writes

- **Tool:** cfitsio (and therefore `funpack`), affecting fitsio.
- **Trigger:** a GZIP_2-compressed complex (`C` / `M`) column in a
  ZTABLE.  cfitsio's GZIP_2 *encoder* silently skips complex
  (it shuffles only 2/4/8-byte I/J/E/D/K); its GZIP_2 *decompressor*
  unshuffle `switch(colcode)` has no C/M case and falls into
  `default: "...unsuitable data type"` → `DATA_DECOMPRESSION_ERR`
  (`imcompress.c`).
- **Symptom:** cfitsio cannot `funpack` even its own GZIP_2-complex
  output (`fpack -g2 -table` writes `ZCTYP='GZIP_2'` on the complex
  column; `funpack` then fails with status 414).
- **Documented:** `docs/internal/ztable.md` § "Complex columns
  (C / M)" (referenced there as rustfits-side issue #8);
  pure-C repro `tools/cfitsio-repro/gzip2_complex_funpack.c`.
- **Workaround:** rustfits **defaults complex columns to GZIP_1**
  (round-trips in both tools).  Explicit
  `compress={"col": "GZIP_2"}` on complex round-trips in rustfits
  but won't funpack.
- **Upstream status:** filed as **cfitsio #135**
  (https://github.com/heasarc/cfitsio/issues/135).

### 5. i8 (TLONGLONG) RICE / GZIP compressed images refused

- **Tool:** cfitsio encoder family, via fitsio.
- **Trigger:** RICE_1 or GZIP compression of an int64 image.
- **Symptom:** fitsio raises "writing TLONGLONG to compressed
  image is not supported."  RICE has no `fits_rcomp_longlong`
  (encoder stops at BYTEPIX=4).  For GZIP this is purely a cfitsio
  implementation limit — GZIP is a byte stream, BITPIX-independent,
  so there's no format reason to refuse it.
- **Documented:** `docs/internal/zimage.md` (RICE i8 + GZIP i8
  sections); `docs/tutorial/limitations.rst`.
- **Workaround:** rustfits rejects `Rice1()` + i8 upfront
  (`NotImplementedError` pointing at `Gzip2`); rustfits's own
  GZIP i8 reader/writer work, so a file written by rustfits with
  GZIP i8 round-trips in rustfits (but not in cfitsio).
- **Upstream status:** RICE i8 is arguably a missing-feature, not a
  bug.  GZIP i8 refusal is the more file-able "cfitsio could
  support this" item.

---

## fitsio (Python wrapper)

### 6. compressed-image `__setitem__` sweep `free(): invalid next size` — NOT a real bug (non-local corruption)

- **Originally observed:** `free(): invalid next size` heap
  corruption aborting the Python process while patching a
  compressed image via fitsio's `write(start=)` repeatedly across
  an algorithm sweep in one process.  Each call was fine standalone;
  the cumulative sweep "tickled" it — which is itself the tell.
- **Investigation (2026-06) — could not reproduce; root cause is
  elsewhere:**
  - **6 fitsio patterns on the current stack** (fitsio 1.3.0 /
    cfitsio 4.060, Linux) all run clean: write+patch one handle per
    algo; one algo × 50 repeats; 8 configs incl. f4 quantized/
    unquantized × 4 selections; reopen-then-patch; 2000-patch churn;
    cross-tool (rustfits writes the fixture, fitsio patches it).
  - **Pure-C probe** (`tools/cfitsio-repro/setitem_sweep_corruption.c`,
    same-handle + reopen variants, all five algorithms) runs clean
    under the normal allocator.
  - **Pure-C probe under ASAN-instrumented cfitsio** also runs
    **clean** — ASAN's red zones find no overflow anywhere in
    cfitsio's compressed-image patch path.
- **Conclusion:** this was **non-local heap corruption** — an
  out-of-bounds write *somewhere else* in that one process clobbered
  a malloc chunk header, and the abort surfaced at an innocent
  `free()` inside the patch sweep.  The original bench ran rustfits
  AND fitsio on the same heap, so the culprit could have been
  rustfits `unsafe` code at the time (much of it since rewritten —
  macos-heap-hunt, issue #19, the streaming-repack rewrite) or any
  other C extension.  It is **not** a bug in cfitsio's patch path
  and **not** a fitsio bug at the crash site.  Distinct from entry 1.
- **Documented:** `CLAUDE.md` § "Performance TODO" item 6;
  `perf/perf-compressed-image-setitem.py`;
  `tools/cfitsio-repro/setitem_sweep_corruption.c` +
  `README.md` § "Negative result".
- **Upstream status:** nothing to file — no reproducer, ASAN-clean.
  If the abort ever recurs, run the full pytest suite under an ASAN
  build of rustfits + cfitsio; ASAN names the real overflow site
  regardless of where `free()` trips.

### 7. Unbounded tile cache OOMs on large compressed images

- **Tool:** fitsio Python wrapper (cfitsio cache layer).
- **Trigger:** scattered/random reads on a multi-GB compressed
  image — fitsio caches every decoded tile forever, no bound, no
  knob.
- **Symptom:** out-of-memory (not reproduced in benches — it's
  RAM-dependent; degradation is by design).
- **Documented:** `docs/tutorial/performance.rst` § "Scattered
  reads"; `CLAUDE.md` § "Performance TODO" item 8;
  `perf/perf-compressed-image-read-1d-scattered.py`.
- **Workaround:** rustfits uses a bytes-bounded LRU and exposes
  `hdu.set_tile_cache_size(bytes)`; fitsio has no such bound.
- **Upstream status:** known fitsio design limitation; not filed.

### 8. VLA-append per-call HDU close/reopen (200× slowdown)

- **Tool:** fitsio Python wrapper (`fitsio_pywrap.c`) on top of
  cfitsio's flush (`ffflus` / `fits_flush_file`).
- **Trigger:** appending to a VLA table column — the wrapper calls
  `fits_flush_file` after every per-column write, and cfitsio's
  flush is close-current-HDU + flush + re-open-current-HDU
  (re-walking the header, re-parsing column descriptors each time).
- **Symptom:** catastrophic slowness — ~300 close/reopen cycles
  (~40 s overhead) for 3 VLA columns × 100 appends; rustfits is
  200–210× faster.
- **Documented:** `docs/tutorial/performance.rst` § "Incremental
  table builds" (cites `fitsio_pywrap.c` ~line 2710 and cfitsio
  `buffers.c::ffflus`).
- **Workaround:** none needed (rustfits doesn't flush per write).
- **Upstream status:** fixable in fitsio (the underlying
  `fits_write_col` doesn't need the flush); not filed.

### 9. fitsio can't extend compressed images or write ZTABLE

- **Tool:** fitsio Python wrapper (capability gap; cfitsio status
  107 on the extend path).
- **Trigger:** appending rows/strips to an existing tile-compressed
  image, or writing a compressed BINTABLE (ZTABLE).
- **Symptom:** cfitsio returns `status = 107: tried to move past
  end of file` for compressed-image extend; fitsio's high-level
  API has no ZTABLE writer at all.
- **Documented:** `perf/perf-compressed-image-extend-2d.py`;
  `docs/tutorial/performance.rst`; `CLAUDE.md`.
- **Workaround:** rustfits supports both; cross-tool comparison
  drops the extend rows for fitsio.
- **Upstream status:** capability gap, not a crash; not filed.

---

## astropy (`astropy.io.fits`)

### 10. `1PX(N)` / `1QX` bit-packed VLA column parser rejection

- **Trigger:** reading a spec-conforming bit-packed VLA column
  (`1PX(maxbits)` / `1QX(maxbits)`).  astropy's `FITS2NUMPY` map
  has no `'X'`, so `_FormatP.from_tform()` raises even with the
  `(maxlen)` hint present.
- **Symptom:** `VerifyError: Invalid column format` during column
  setup.
- **Documented:** `CLAUDE.md` § "Bit-packed X columns";
  `docs/tutorial/limitations.rst`;
  `tests/test_table_vla_x_bit.py::test_astropy_pxqx_documented_limitation`.
- **Note:** fitsio reads PX/QX fine (one-time maxlen warning);
  cfitsio supports it natively — astropy-only.

### 11. `CompImageHDU.verify_checksum` internal TypeError

- **Trigger:** calling `verify_checksum` on a compressed-image HDU
  (even on astropy's own writes).
- **Symptom:** `TypeError` on internal `_compute_checksum(None)`.
- **Documented:** `docs/tutorial/limitations.rst`;
  `docs/internal/zimage.md`; `tests/test_hdu_checksum.py`.
- **Workaround:** rustfits doesn't cross-verify compressed
  ZHECKSUM against astropy; its own self-verify is correct.

### 12. Silent i64 → i32 downcast before RICE encoding

- **Trigger:** RICE-compressing an int64 image with astropy.
- **Symptom:** astropy silently truncates i64 → i32 before
  encoding — lossy when values exceed the i32 range, no warning.
- **Documented:** `docs/internal/zimage.md` (RICE i8 section).
- **Workaround:** rustfits rejects the combination upfront.

### 13. PLIO_1 + float hard-rejected; unquantized-float RICE/HCOMPRESS silently lossy

- **Trigger:** compressing float images: PLIO_1 (a mask-only
  encoder) is hard-rejected; RICE_1 / HCOMPRESS_1 with
  *unquantized* floats are written but do NOT round-trip
  bit-exactly (RICE codes float bit patterns → garbage; HCOMPRESS
  H-transform is an integer wavelet).
- **Symptom:** PLIO → rejection; RICE/HCOMPRESS unquantized-float
  → silent non-bit-exact round-trip.
- **Documented:** `docs/internal/zimage.md` (algorithm/dtype
  table).
- **Workaround:** rustfits requires Gzip1/Gzip2 for unquantized
  floats and rejects the lossy combos with a pointed error.

---

## Filing cfitsio issues

Three clean pure-C reproducers (no rustfits, no fitsio) are filed
with cfitsio; their source + draft issue text live in
`tools/cfitsio-repro/`:

1. **Entry 2 — PA-VLA funpack crash** → cfitsio **#134**.
   `pa_vla_funpack_crash.c`: build an uncompressed table with a
   `1PA` column, `fpack -table`, `funpack` → SIGABRT.
2. **Entry 4 — GZIP_2 complex self-incompatibility** → cfitsio
   **#135**.  `gzip2_complex_funpack.c`: `fpack -g2 -table` a `1C`
   column, `funpack` → status 414.
3. **Entry 1 — PLIO_1 small-image write overflow** → cfitsio
   **#136** (also reported as fitsio #496).
   `plio_small_image_overflow.c`: write a 4×4 PLIO_1 image with 2×2
   tiles; cfitsio overflows the `nx*sizeof(int)`-byte tile buffer
   inside `pl_p2li` at compress time (no fpack/funpack step).  ASAN
   (instrumented cfitsio) pins it to a 2-byte WRITE 0 bytes past the
   16-byte buffer — `pliocomp.c` / `imcompress.c:1924`+`:1970`.
   macOS aborts on it; glibc tolerates the overrun, so the repro is
   built against an ASAN cfitsio to make it fail on Linux.

Investigated, **not** filed on cfitsio:

4. **Entry 6 — setitem `free()` corruption** — the pure-C probe
   (`setitem_sweep_corruption.c`) runs clean, so this is not a
   cfitsio bug; it belongs to the fitsio wrapper.  File there if
   pursued.

Reference C source for building repros lives in the bundled
cfitsio tarball — see `CLAUDE.md` § "Reference sources for
byte-exact ports" for the untar-and-locate recipe (e.g.
`<cfitsio>/imcompress.c` for the table compress/uncompress
paths).  Build/run the existing reproducers with
`bash tools/cfitsio-repro/run.sh`.
