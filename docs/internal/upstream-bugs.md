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
- fitsio **#496** — macOS compressed-image-write crash (entry 1).

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
  (`_SKIP_FITSIO_PLIO_ON_MACOS`).
- **Workaround:** CI replaces the conda-forge fitsio with a pip
  source build on the macOS legs only (dodges the `ffbinit`
  variant); a skipif mark on just the PLIO_1 parametrize case
  dodges the source-build variant.  PLIO read coverage on macOS is
  preserved via `test_image_compressed_read_plio.py` (non-crashing
  path).
- **Upstream status:** filed as **fitsio #496**
  (https://github.com/esheldon/fitsio/issues/496); a maintainer
  with a Mac is investigating.  macOS-only — not reproducible on
  the Linux dev box.

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

### 3. Platform-dependent dequantization (macOS vs Linux)

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

### 6. compressed-image `__setitem__` sweep `free(): invalid next size`

- **Tool:** fitsio Python wrapper (Linux).  **Reclassified from
  "cfitsio" — see finding below.**
- **Trigger:** running fitsio's `write(start=)` compressed-image
  patch across an *algorithm sweep in one process* (each call is
  fine standalone; the cumulative sweep tickles it).
- **Symptom:** `free(): invalid next size` heap corruption that
  aborts the Python process.
- **Finding (2026-06):** a pure-C reproducer
  (`tools/cfitsio-repro/setitem_sweep_corruption.c`) exercises the
  same shape with NO Python — create a (256,256) int image with
  (32,32) tiles, then patch sub-regions via `fits_write_subset`
  repeatedly, for all five algorithms in one process, in both
  same-handle and reopen-then-patch variants.  **It runs clean** —
  no corruption, no abort.  So the bug is **not** in cfitsio's
  compressed-image patch path; it's most likely in the fitsio
  wrapper's buffer/call handling (or a wrapper-specific call
  sequence).  Distinct from entry 1 (different platform — Linux vs
  macOS; different path — patch vs create).
- **Documented:** `CLAUDE.md` § "Performance TODO" item 6;
  `perf/perf-compressed-image-setitem.py`; the C probe above.
- **Workaround:** the cross-tool comparison was dropped from the
  `__setitem__` perf bench (rustfits-self per-tile cost is reported
  instead).
- **Upstream status:** not a cfitsio bug (per the C probe).  If
  pursued, file on fitsio after isolating the wrapper call that
  triggers it.

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

Two clean pure-C reproducers (no rustfits, no fitsio) are filed
with cfitsio; their source + draft issue text live in
`tools/cfitsio-repro/`:

1. **Entry 2 — PA-VLA funpack crash** → cfitsio **#134**.
   `pa_vla_funpack_crash.c`: build an uncompressed table with a
   `1PA` column, `fpack -table`, `funpack` → SIGABRT.
2. **Entry 4 — GZIP_2 complex self-incompatibility** → cfitsio
   **#135**.  `gzip2_complex_funpack.c`: `fpack -g2 -table` a `1C`
   column, `funpack` → status 414.

Investigated, **not** filed on cfitsio:

3. **Entry 6 — setitem `free()` corruption** — the pure-C probe
   (`setitem_sweep_corruption.c`) runs clean, so this is not a
   cfitsio bug; it belongs to the fitsio wrapper.  File there if
   pursued.
4. **Entry 1 — macOS compressed-image-write crash** — filed on
   fitsio (**#496**); macOS-only, not reproducible on the Linux
   dev box, so no pure-C reproducer here.

Reference C source for building repros lives in the bundled
cfitsio tarball — see `CLAUDE.md` § "Reference sources for
byte-exact ports" for the untar-and-locate recipe (e.g.
`<cfitsio>/imcompress.c` for the table compress/uncompress
paths).  Build/run the existing reproducers with
`bash tools/cfitsio-repro/run.sh`.
