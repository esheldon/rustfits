# cfitsio reproducers

Standalone C programs that reproduce cfitsio bugs **without rustfits
or the fitsio Python wrapper** — pure libcfitsio plus the
`fpack` / `funpack` CLIs.  Built to back real issues on the
[cfitsio repo](https://github.com/HEASARC/cfitsio).

See [`docs/internal/upstream-bugs.md`](../../docs/internal/upstream-bugs.md)
for the full inventory of upstream bugs rustfits has found; this
directory holds the three with clean pure-C reproducers.

## Building / running

Needs cfitsio (header + lib) and `fpack`/`funpack` on PATH.  In a
conda env that provides cfitsio it works out of the box
(`$CONDA_PREFIX` supplies the include/lib paths):

```bash
bash run.sh          # all three reproducers
bash run.sh pa       # just #3 (PA-VLA funpack crash)
bash run.sh cplx     # just #5 (GZIP_2 complex)
bash run.sh plio     # just #1 (PLIO_1 small-image write overflow)
```

Override the cfitsio location with `CFITSIO_PREFIX=/path bash run.sh`.

Confirmed against cfitsio 4.6.0 (fpack/funpack 1.7.0) on Linux
x86_64.

Two of the three (`pa`, `cplx`) crash inside `funpack`; the third
(`plio`) overflows a buffer inside **libcfitsio at write time**, so
it has no `fpack`/`funpack` step — the program is the reproducer.  On
glibc/Linux that few-byte overrun is usually tolerated (a plain run
exits 0); to make it abort with an exact site, link it against an
ASAN-instrumented cfitsio (next section).

### Running under AddressSanitizer

To get an exact overflow site (file/line) instead of a vague
`realloc(): invalid next size`, build cfitsio with ASAN and run the
instrumented `fpack`/`funpack` against the repro fixtures.  Using
the bundled cfitsio source (see `CLAUDE.md` § "Reference sources for
byte-exact ports" for where it untars):

```bash
SRC=~/git/fitsio/cfitsio-4.6.4
B=$(mktemp -d); cp -r "$SRC"/. "$B"/
( cd "$B" \
  && CFLAGS="-fsanitize=address -g -O1 -fno-omit-frame-pointer" \
       ./configure --disable-curl \
  && make -j )
# instrumented binaries land in $B/.libs/{fpack,funpack}
./pa_vla_funpack_crash pa.fits          # write fixture (normal libcfitsio)
"$B/.libs/fpack" -table pa.fits
ASAN_OPTIONS=detect_leaks=0 "$B/.libs/funpack" -O out.fits pa.fits.fz
```

This is how the root-cause traces below were captured.

For the write-time overflow (`#1`, PLIO) there is no funpack step —
compile the reproducer itself against the instrumented cfitsio so
the overrun in `pl_p2li` is caught the moment cfitsio compresses the
tile:

```bash
# $B is the ASAN cfitsio build dir from above (header in $B, lib in
# $B/.libs).  -lm because the instrumented lib pulls in libm.
cc -O1 -g -fsanitize=address -fno-omit-frame-pointer -I"$B" \
   plio_small_image_overflow.c -o plio_asan \
   -L"$B/.libs" -lcfitsio -lm -Wl,-rpath,"$B/.libs"
ASAN_OPTIONS=detect_leaks=0 ./plio_asan
# (run.sh plio does the same when ASAN=1 and CFITSIO_PREFIX point at
#  an instrumented build laid out as prefix/include + prefix/lib)
```

---

## #1 — PLIO_1 small-image compression buffer overflow (write time)

**filed:** cfitsio [#136](https://github.com/heasarc/cfitsio/issues/136)
· **also:** fitsio [#496](https://github.com/esheldon/fitsio/issues/496)
· **file:** `plio_small_image_overflow.c`

Writing a PLIO_1-compressed image with small tiles overflows
cfitsio's per-tile compression buffer.  `imcomp_calc_max_elem()`
sizes the PLIO buffer (the catch-all `else` branch) at
`nx * sizeof(int)` **bytes** with no minimum — unlike HCOMPRESS_1
just above it, which adds a `+ 26` overhead "only significant for
very small tiles".  cfitsio then `calloc`s `cbuf` to that many bytes
and hands it to `pl_p2li()`, which writes the IRAF line-list as
`short` (2-byte) elements with a fixed **7-short (14-byte) header**
before any pixel data (`lldst[1..7]`; the data cursor `op` starts at
8).  For a 2×2 tile, `nx = 4`, so the buffer is `4*sizeof(int) = 16`
bytes = only 8 shorts: the header nearly fills it and the first
encoded data short writes past the allocation.

This is macOS-surfaced (the nano allocator's guard bytes abort the
process ~50% of the time — how it first showed up as a flaky
compressed-image-write crash on CI) but the under-allocation is
platform-independent: glibc usually tolerates the few-byte overrun,
and an ASAN build flags it exactly on Linux.

Observed (reproducer linked against an ASAN-instrumented cfitsio):

```
==ERROR: AddressSanitizer: heap-buffer-overflow
WRITE of size 2 at 0x... thread T0
    #0 pl_p2li                       pliocomp.c:136
    #1 imcomp_compress_tile          imcompress.c:1970   <- pl_p2li(idata,1,cbuf,tilelen)
    #2 fits_write_compressed_img     imcompress.c:3739
    ...
    #7 main  plio_small_image_overflow.c:97              <- fits_write_img

0x... is located 0 bytes after a 16-byte region
allocated by thread T0 here:
    #0 calloc
    #1 imcomp_compress_tile          imcompress.c:1924   <- cbuf = calloc(clen,1); clen = nx*sizeof(int) = 16
    ...
SUMMARY: AddressSanitizer: heap-buffer-overflow pliocomp.c:136 in pl_p2li
```

The overflow is a 2-byte (`short`) WRITE 0 bytes past the 16-byte
`nx*sizeof(int)` buffer — exactly the predicted off-by-header-size.

### Draft cfitsio issue

Everything from here to the next `---` is self-contained — copy it
straight into a new cfitsio issue.

**Title:** PLIO_1 compression buffer is under-allocated for small
tiles (heap-buffer-overflow in `pl_p2li`)

**Summary:** Writing a PLIO_1-compressed image with small tiles
overflows the per-tile compression buffer.
`imcomp_calc_max_elem()` sizes the PLIO buffer (its catch-all `else`
branch) at `nx * sizeof(int)` bytes with no floor, but `pl_p2li()`
writes the line-list as `short`s preceded by a fixed 7-short
(14-byte) header. For a 2×2 tile (`nx = 4`) the buffer is 16 bytes =
8 shorts, so the header alone nearly fills it and the first data
short writes out of bounds. glibc usually tolerates the overrun;
macOS's allocator aborts; ASAN reports it precisely.

**Environment:** cfitsio 4.6.x, Linux x86_64 (ASAN) / macOS arm64.

**Reproduce.** Compile and run this program (no fpack/funpack — the
overflow is at write time inside libcfitsio):

```c
/* plio_small.c — build: cc plio_small.c -o plio_small -lcfitsio */
#include <stdio.h>
#include "fitsio.h"

int main(void) {
    fitsfile *fptr = NULL;
    int status = 0;
    remove("plio_small.fits.fz");
    fits_create_file(&fptr, "plio_small.fits.fz", &status);
    fits_set_compression_type(fptr, PLIO_1, &status);
    long tile[2] = {2, 2};
    fits_set_tile_dim(fptr, 2, tile, &status);   /* nx = 4 */
    long naxes[2] = {4, 4};
    fits_create_img(fptr, LONG_IMG, 2, naxes, &status);
    int pix[16];
    for (int i = 0; i < 16; i++) pix[i] = i % 4;
    fits_write_img(fptr, TINT, 1, 16, pix, &status);  /* overflow here */
    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);
    return status;
}
```

Build cfitsio with `CFLAGS="-fsanitize=address -g" ./configure
--disable-curl`, compile the program with `-fsanitize=address`, and
run:

```
==ERROR: AddressSanitizer: heap-buffer-overflow
WRITE of size 2 ... 0 bytes after a 16-byte region
  #0 pl_p2li               pliocomp.c
  #1 imcomp_compress_tile  imcompress.c   (pl_p2li(idata,1,cbuf,tilelen))
  ...
allocated by:
  #1 imcomp_compress_tile  imcompress.c   (cbuf = calloc(nx*sizeof(int), 1))
```

**Root cause** — `imcomp_calc_max_elem()` (`imcompress.c`):

```c
    else
        return(nx * sizeof(int));   /* PLIO_1: BYTES, no minimum */
```

`cbuf` is then `calloc`'d to that many bytes, but `pl_p2li()`
(`pliocomp.c`) writes `short`s after a 14-byte header, so small
`nx` overflows.

**Proposed fix** — floor the PLIO branch at 32 bytes:

```c
    else {
        if (nx * sizeof(int) < 32)
            return 32;
        return(nx * sizeof(int));
    }
```

**Expected:** PLIO_1 compresses small tiles without overrunning.
**Actual:** heap-buffer-overflow write in `pl_p2li` (aborts on
macOS / under ASAN; silently corrupts the heap on glibc).

---

## #3 — String-VLA (`1PA`) ZTABLE columns crash `funpack`

**filed:** cfitsio [#134](https://github.com/heasarc/cfitsio/issues/134)
· **rustfits issue:** [#9](https://github.com/esheldon/rustfits/issues/9)
· **file:** `pa_vla_funpack_crash.c`

A variable-length string column (`1PA`/`1QA`) in a tile-compressed
table makes `funpack` abort with heap corruption.  cfitsio crashes
on its **own** `fpack -table` output, so the defect is entirely in
cfitsio's table (de)compression of PA columns.

Observed:

```
--- fpack -table ---
fpack rc=0
--- funpack ---
realloc(): invalid next size
Aborted (core dumped)
funpack exit code: 134
```

Root cause (AddressSanitizer build of cfitsio):

```
==ERROR: AddressSanitizer: heap-buffer-overflow
READ of size 32768 at 0x... thread T0
    #0 memcpy
    #1 updatewindow (libz)
    #2 inflate (libz)
    #3 uncompress2mem_from_mem (libcfitsio)
    #4 fits_uncompress_table (libcfitsio)
    #5 fp_unpack_hdu utilities/fpackutil.c:1649
    #6 fp_unpack / fp_loop / main (funpack.c:23)

0x... is located 0 bytes after 24000-byte region
allocated by thread T0 here:
    #0 malloc
    #1 fits_uncompress_table (libcfitsio)
```

C-level cause — `fits_uncompress_table` reserves `cm_buffer` space
for VLA *descriptors*, then asks `uncompress2mem_from_mem` to
`realloc`-grow a slice of that shared buffer it doesn't own:

- `imcompress.c:8857` / `:8898` — per VLA column, `addspace += 16`,
  and `cm_size = naxis1*rowspertile + addspace*rowspertile`.  So a
  VLA column's reservation is sized for its descriptors
  (`descriptor_width + 16` bytes/row), **not** the decompressed
  string payload.
- `imcompress.c:9008` (the `default:` VLA branch) gunzips the
  payload into `cptr = cm_buffer + cmajor_colstart[ii]` with
  `fullsize` = that descriptor-sized reservation, passing `realloc`
  as the grow callback.
- `zcompress.c:227` — when the decompressed strings exceed
  `fullsize`, `uncompress2mem_from_mem` does
  `realloc(cm_buffer + offset, …)` on a pointer **interior** to the
  shared `cm_buffer` (undefined behavior → heap corruption).  For
  the first column (`cptr == cm_buffer`) the realloc instead
  silently moves the block, leaving `cm_buffer` dangling for the
  transpose loop at `:9018`.

The contract mismatch: `uncompress2mem_from_mem` is built to own and
`realloc`-grow its output buffer, but it's handed a slice of a
shared one.  Fix is a maintainer design call — decompress each VLA
column's payload into a separate owned buffer then copy into
`cm_buffer`, or size the reservation to the true uncompressed
extent.

### Draft cfitsio issue

Everything from here to the next `---` is self-contained — copy it
straight into a new cfitsio issue.

**Title:** funpack crashes (heap corruption) on tile-compressed
tables with variable-length string (`1PA`) columns

**Summary:** A binary table with a variable-length character column
(TFORM `1PA`) compressed with `fpack -table` cannot be decompressed
by `funpack` — it aborts with `realloc(): invalid next size`
(SIGABRT). cfitsio crashes on its own `fpack` output, so no
third-party tooling is involved. Numeric VLA inner types (`1PE`,
`1PJ`, …) compress and funpack fine; only the string (`A`) inner
type triggers it.

**Environment:** cfitsio 4.6.0, fpack/funpack 1.7.0, Linux x86_64
(glibc).

**Reproduce.** Compile and run this program (writes an ordinary
uncompressed BINTABLE with one `1PA` column), then `fpack`/`funpack`
it:

```c
/* pa_vla.c — build: cc pa_vla.c -o pa_vla -lcfitsio */
#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"

int main(void) {
    const char *fname = "pa_vla.fits";
    fitsfile *fptr = NULL;
    int status = 0;

    remove(fname);
    fits_create_file(&fptr, fname, &status);

    const long nrows = 3000;
    char *ttype[] = {"name"};
    char *tform[] = {"1PA"};  /* variable-length string column */
    char *tunit[] = {""};
    fits_create_tbl(fptr, BINARY_TBL, 0, 1,
                    ttype, tform, tunit, "DATA", &status);

    /* One variable-length string per row: "obj_" + (i % 15) 'x's. */
    for (long i = 0; i < nrows; i++) {
        char buf[64];
        int xn = (int)(i % 15);
        int p = snprintf(buf, sizeof buf, "obj_");
        for (int k = 0; k < xn; k++) buf[p++] = 'x';
        buf[p] = '\0';
        char *cell[1] = {buf};
        fits_write_col(fptr, TSTRING, 1, i + 1, 1, 1, cell, &status);
    }

    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);
    return status;
}
```

```console
$ cc pa_vla.c -o pa_vla -lcfitsio && ./pa_vla
$ fpack -table pa_vla.fits          # rc 0, writes pa_vla.fits.fz
$ funpack -O out.fits pa_vla.fits.fz
realloc(): invalid next size
Aborted (core dumped)               # exit 134 = 128 + SIGABRT
```

**Root cause** (AddressSanitizer build of cfitsio — `CFLAGS="-fsanitize=address -g" ./configure --disable-curl`):

```
ERROR: AddressSanitizer: heap-buffer-overflow
READ of size 32768 ... 0 bytes after a 24000-byte region
  #1 updatewindow (libz)
  #2 inflate (libz)
  #3 uncompress2mem_from_mem   (cfitsio)
  #4 fits_uncompress_table     (cfitsio)   <- buffer allocated here
  #5 fp_unpack_hdu  utilities/fpackutil.c:1649
  #6 main           utilities/funpack.c:23
```

C-level cause: `fits_uncompress_table` reserves `cm_buffer` space
for VLA *descriptors* only (`imcompress.c:8857`/`:8898`), then the
`default:` VLA branch (`:9008`) gunzips the larger string *payload*
into an interior pointer of that shared buffer with a `realloc` grow
callback.  When the payload exceeds the reservation,
`uncompress2mem_from_mem` (`zcompress.c:227`) reallocs an interior
pointer of `cm_buffer` — undefined behavior, which corrupts the
heap.  `uncompress2mem_from_mem` is built to own and grow its output
buffer; here it's handed a slice of a shared one.

**Expected:** funpack reconstructs the original table.
**Actual:** heap corruption / abort (ASAN: out-of-bounds read off
`fits_uncompress_table`'s under-sized output buffer).

---

## #5 — `GZIP_2` complex (`1C`/`1M`) columns: cfitsio can't funpack its own output

**filed:** cfitsio [#135](https://github.com/heasarc/cfitsio/issues/135)
· **rustfits doc:** `docs/internal/ztable.md` (referenced as issue #8
on the rustfits side) · **file:** `gzip2_complex_funpack.c`

`fpack -g2 -table` writes a complex column with `ZCTYP='GZIP_2'`,
but `funpack` then refuses it: the GZIP_2 table decompressor's
unshuffle dispatch has no complex (`C`/`M`) case.  The encoder
silently skips shuffling complex (it shuffles only 2/4/8-byte
I/J/E/D/K) yet still labels the column GZIP_2, so the produced file
is unreadable by the same library.

Observed (`ZCTYP1 = 'GZIP_2'` confirmed in the `.fz` header):

```
--- fpack -g2 -table ---
fpack rc=0
--- funpack ---
FITSIO status = 414: error uncompressing image
Error: unexpected attempt to use GZIP_2 to compress a column
       unsuitable data type
funpack exit code: 158
```

### Draft cfitsio issue

Everything from here to the end is self-contained — copy it
straight into a new cfitsio issue.

**Title:** `fpack -g2` produces GZIP_2 complex-column tables that
`funpack` rejects ("unsuitable data type")

**Summary:** When a binary table with a complex column (TFORM `1C`
or `1M`) is compressed with `fpack -g2 -table`, cfitsio writes the
column with `ZCTYP='GZIP_2'`. `funpack` then fails with
`status 414 — unexpected attempt to use GZIP_2 to compress a column
unsuitable data type`. The GZIP_2 *encoder* skips byte-shuffling
complex columns (it shuffles only 2/4/8-byte I/J/E/D/K) but still
tags the column GZIP_2, while the GZIP_2 *decompressor*'s unshuffle
`switch` has no `C`/`M` case — so cfitsio writes a file it cannot
read back.

**Environment:** cfitsio 4.6.0, fpack/funpack 1.7.0, Linux x86_64.

**Reproduce.** Compile and run this program (writes an ordinary
uncompressed BINTABLE with one `1C` column), then `fpack`/`funpack`
it:

```c
/* cplx.c — build: cc cplx.c -o cplx -lcfitsio */
#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"

int main(void) {
    const char *fname = "cplx.fits";
    fitsfile *fptr = NULL;
    int status = 0;

    remove(fname);
    fits_create_file(&fptr, fname, &status);

    const long nrows = 2000;
    char *ttype[] = {"z"};
    char *tform[] = {"1C"};  /* single-precision complex */
    char *tunit[] = {""};
    fits_create_tbl(fptr, BINARY_TBL, 0, 1,
                    ttype, tform, tunit, "DATA", &status);

    /* fits_write_col with TCOMPLEX takes float pairs [re, im, ...]. */
    float *cell = malloc((size_t)nrows * 2 * sizeof(float));
    for (long i = 0; i < nrows; i++) {
        cell[2 * i] = (float)i * 0.5f;
        cell[2 * i + 1] = (float)(-i) * 0.25f;
    }
    fits_write_col(fptr, TCOMPLEX, 1, 1, 1, nrows, cell, &status);
    free(cell);

    fits_close_file(fptr, &status);
    fits_report_error(stderr, status);
    return status;
}
```

```console
$ cc cplx.c -o cplx -lcfitsio && ./cplx
$ fpack -g2 -table cplx.fits        # rc 0; writes ZCTYP1='GZIP_2'
$ funpack -O out.fits cplx.fits.fz
FITSIO status = 414: error uncompressing image
Error: unexpected attempt to use GZIP_2 to compress a column
       unsuitable data type                # exit 158
```

**Expected:** either funpack decompresses it, or fpack declines to
use GZIP_2 for complex columns (falling back to GZIP_1, which
round-trips).
**Actual:** cfitsio writes a file it cannot read back.

---

## Negative result — compressed-image `__setitem__` sweep

**file:** `setitem_sweep_corruption.c` · `bash run.sh sweep`

`upstream-bugs.md` entry 6 records a `free(): invalid next size`
heap corruption seen on Linux through fitsio's `write(start=)` when
patching a compressed image repeatedly across an algorithm sweep in
one process.  This program is the pure-C check of whether cfitsio is
at fault: for each of the five algorithms it creates a (256,256)
int image with (32,32) tiles, writes it, then patches sub-regions
via `fits_write_subset` repeatedly — in both same-handle and
reopen-then-patch variants, all in one process.

**Result: it runs clean** — no corruption, no abort, exit 0 — under
both the normal allocator **and an AddressSanitizer build of
cfitsio** (ASAN's red zones find no overflow in the patch path).
Together with six clean fitsio patterns on the current stack
(fitsio 1.3.0 / cfitsio 4.060), this means the original
`free(): invalid next size` was **non-local heap corruption**: an
out-of-bounds write *elsewhere* in that one process clobbered a
malloc chunk header, and the abort surfaced at an innocent `free()`
here.  The original bench ran rustfits + fitsio on the same heap, so
the culprit was plausibly since-fixed rustfits `unsafe` code, not
cfitsio's patch path.  Kept as the documented baseline; nothing to
file (see `docs/internal/upstream-bugs.md` entry 6).
