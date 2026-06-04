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

Only **issue #9** (PA-VLA funpack crash) has a rustfits issue so
far; everything else is documented inline at the cited locations.

The candidates worth turning into standalone C reproducers + real
cfitsio issues are flagged **[C-repro candidate]** — see the
"Filing cfitsio issues" section at the end.

---

## cfitsio (C library) — crashes / memory corruption

### 1. `ffbinit` libmalloc bad-free on macOS compressed-image create

- **Tool:** cfitsio (conda-forge binary), surfaced via the fitsio
  Python wrapper.
- **Trigger:** the first compressed-image *write* on macOS (the
  `PyFITSObject_create_image_hdu` path), both py3.12 and py3.14 —
  so it's the conda-forge build, not the Python version.
- **Symptom:** libmalloc bad-free crash inside cfitsio's `ffbinit`.
- **Documented:** `CLAUDE.md` § "CI: macOS fitsio workaround";
  first aborting test was
  `tests/test_image_compressed_accessors.py::test_other_compression_types_dispatched`.
- **Workaround:** CI replaces the conda-forge fitsio with a pip
  source build on the macOS legs only
  (`pip install --force-reinstall --no-deps --no-binary=fitsio fitsio`);
  Linux keeps the fast conda install.
- **Upstream status:** not filed (suspected conda-forge build /
  toolchain interaction; needs isolating before it's a clean
  cfitsio bug).

### 2. `free(): invalid next size` on compressed-image `__setitem__` sweep

- **Tool:** cfitsio, via fitsio's `write(start=)`.
- **Trigger:** running fitsio's `write(start=)` compressed-image
  patch across an *algorithm sweep in one process* (each call is
  fine standalone; the sweep tickles it).
- **Symptom:** cfitsio `free(): invalid next size` heap corruption
  (same shape as #1).
- **Documented:** `CLAUDE.md` § "Performance TODO" item 6;
  `perf/perf-compressed-image-setitem.py`;
  `perf/perf-all.py`.
- **Workaround:** the cross-tool comparison was dropped from the
  `__setitem__` perf bench (rustfits-self per-tile cost is reported
  instead); subprocess isolation would dodge it but wasn't worth it.
- **Upstream status:** not filed (needs a pure-C minimal sweep to
  confirm it's cfitsio, not the wrapper).

### 3. String-VLA (PA/QA) ZTABLE columns crash `funpack`  **[C-repro candidate]**

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
- **Documented:** rustfits **issue #9**
  (https://github.com/esheldon/rustfits/issues/9);
  `tests/test_table_compressed_write_vla.py`
  (`test_pa_vla_ztable_rustfits_roundtrip` proves rustfits is
  correct; `test_funpack_pa_vla_ztable_cfitsio_crash_documented`
  is the skip-documented repro).
- **Workaround:** none needed in rustfits (our PA-VLA ZTABLE
  read/write is correct).  Interop limitation: PA-VLA ZTABLE files
  are not funpack-readable with current cfitsio.
- **Upstream status:** rustfits issue #9 open; **not yet filed on
  cfitsio**.

### 4. Platform-dependent dequantization (macOS vs Linux)

- **Tool:** cfitsio (compiler codegen / libm — FMA fusion, Apple
  libm vs glibc), via fitsio.
- **Trigger:** reading quantized-float compressed images on macOS
  arm64 vs Linux.
- **Symptom:** slightly different decoded float values on macOS
  (rtol up to ~1.6e-5, atol up to ~2.6e-9 near zero).  rustfits's
  Rust dequant bit-matches Linux cfitsio.
- **Documented:** `tests/test_image_compressed_read_quantize.py`
  (`_MACOS_FP_RTOL` / `_MACOS_FP_ATOL`).
- **Workaround:** loosened macOS tolerances pin the divergence so a
  regression beyond the bound surfaces.
- **Upstream status:** not a clean bug to file (numerical
  reproducibility, not a defect); documented for our own sake.

---

## cfitsio (C library) — wrong output / self-incompatibility

### 5. GZIP_2 table decompressor can't read complex (C/M) columns it writes  **[C-repro candidate]**

- **Tool:** cfitsio (and therefore `funpack`), affecting fitsio.
- **Trigger:** a GZIP_2-compressed complex (`C` / `M`) column in a
  ZTABLE.  cfitsio's GZIP_2 *encoder* silently skips complex
  (it shuffles only 2/4/8-byte I/J/E/D/K); its GZIP_2 *decompressor*
  unshuffle `switch(colcode)` has no C/M case and falls into
  `default: "...unsuitable data type"` → `DATA_DECOMPRESSION_ERR`
  (`imcompress.c`).
- **Symptom:** cfitsio cannot `funpack` even its own GZIP_2-complex
  output.
- **Documented:** `docs/internal/ztable.md` § "Complex columns
  (C / M)" (referenced there as issue #8, a rustfits-side fix).
- **Workaround:** rustfits **defaults complex columns to GZIP_1**
  (round-trips in both tools).  Explicit
  `compress={"col": "GZIP_2"}` on complex round-trips in rustfits
  but won't funpack.
- **Upstream status:** not filed; clean pure-C repro is feasible
  (fpack a complex column with GZIP_2, funpack it).

### 6. i8 (TLONGLONG) RICE / GZIP compressed images refused

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

### 7. PLIO_1 write abort on macOS

- **Tool:** fitsio Python wrapper + its source-built cfitsio.
- **Trigger:** `create_image_hdu` with `compress="PLIO_1"` and
  default qlevel, on macOS py3.12 (~50% hit rate).  The read-side
  path (`compress="PLIO"` + explicit `qlevel=None`) does NOT
  trigger it, so it's fitsio's specific PLIO_1 + default-qlevel
  *write* path.
- **Symptom:** bare `Fatal Python error: Aborted`, crash stack
  entirely inside fitsio's `_fitsio_wrap.create_image_hdu` (no
  rustfits frames).
- **Documented:** `CLAUDE.md` § "macOS CI flakes — resolved
  2026-06-01"; `tests/test_image_compressed_accessors.py`
  (`_SKIP_FITSIO_PLIO_ON_MACOS`).
- **Workaround:** skipif mark on just the PLIO_1 parametrize case;
  PLIO read coverage on macOS preserved via
  `test_image_compressed_read_plio.py` (non-crashing path).
- **Upstream status:** not filed (intermittent, macOS-specific).

### 8. Unbounded tile cache OOMs on large compressed images

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

### 9. VLA-append per-call HDU close/reopen (200× slowdown)

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

### 10. fitsio can't extend compressed images or write ZTABLE

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

### 11. `1PX(N)` / `1QX` bit-packed VLA column parser rejection

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

### 12. `CompImageHDU.verify_checksum` internal TypeError

- **Trigger:** calling `verify_checksum` on a compressed-image HDU
  (even on astropy's own writes).
- **Symptom:** `TypeError` on internal `_compute_checksum(None)`.
- **Documented:** `docs/tutorial/limitations.rst`;
  `docs/internal/zimage.md`; `tests/test_hdu_checksum.py`.
- **Workaround:** rustfits doesn't cross-verify compressed
  ZHECKSUM against astropy; its own self-verify is correct.

### 13. Silent i64 → i32 downcast before RICE encoding

- **Trigger:** RICE-compressing an int64 image with astropy.
- **Symptom:** astropy silently truncates i64 → i32 before
  encoding — lossy when values exceed the i32 range, no warning.
- **Documented:** `docs/internal/zimage.md` (RICE i8 section).
- **Workaround:** rustfits rejects the combination upfront.

### 14. PLIO_1 + float hard-rejected; unquantized-float RICE/HCOMPRESS silently lossy

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

The cleanest candidates for standalone C reproducers + real
cfitsio issues are the ones that reproduce **without rustfits or
fitsio's high-level API** — i.e. pure libcfitsio + the
`fpack` / `funpack` CLIs:

1. **#3 PA-VLA funpack crash** — the strongest case (cfitsio
   crashes on its own `fpack -table` output; already rustfits
   issue #9).  A pure-C repro: build an uncompressed table with a
   `1PA` column via `fits_create_tbl` + `fits_write_col`, then
   either shell out to `fpack`/`funpack` or call the internal
   compress/uncompress directly.
2. **#5 GZIP_2 complex self-incompatibility** — write a `C`/`M`
   column, `fpack -table` with GZIP_2, `funpack` → decompression
   error.  Pure-C and deterministic.

Secondary (need more isolation before they're clean cfitsio
issues, since they currently only manifest through a wrapper or a
specific platform/build):

3. **#2 setitem free-corruption** — needs a pure-C minimal
   algorithm-sweep to confirm it's cfitsio and not the wrapper.
4. **#1 ffbinit macOS bad-free** — platform/build-specific;
   isolate the conda-forge toolchain factor first.

Reference C source for building repros lives in the bundled
cfitsio tarball — see `CLAUDE.md` § "Reference sources for
byte-exact ports" for the untar-and-locate recipe (e.g.
`<cfitsio>/imcompress.c` for the table compress/uncompress
paths).  Standalone repro programs, when written, should live in
`tools/cfitsio-repro/` with a short README per bug and a
build/run line that links against the system or bundled cfitsio.
