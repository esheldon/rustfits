# cfitsio reproducers

Standalone C programs that reproduce cfitsio bugs **without rustfits
or the fitsio Python wrapper** — pure libcfitsio plus the
`fpack` / `funpack` CLIs.  Built to back real issues on the
[cfitsio repo](https://github.com/HEASARC/cfitsio).

See [`docs/internal/upstream-bugs.md`](../../docs/internal/upstream-bugs.md)
for the full inventory of upstream bugs rustfits has found; this
directory holds the two with clean pure-C reproducers.

## Building / running

Needs cfitsio (header + lib) and `fpack`/`funpack` on PATH.  In a
conda env that provides cfitsio it works out of the box
(`$CONDA_PREFIX` supplies the include/lib paths):

```bash
bash run.sh          # both reproducers
bash run.sh pa       # just #3 (PA-VLA funpack crash)
bash run.sh cplx     # just #5 (GZIP_2 complex)
```

Override the cfitsio location with `CFITSIO_PREFIX=/path bash run.sh`.

Confirmed against cfitsio 4.6.0 (fpack/funpack 1.7.0) on Linux
x86_64.

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

So `fits_uncompress_table` **under-allocates the decompression
output buffer** for the variable-length-string column (24000 bytes
here), and zlib's `inflate`/`updatewindow` window copy reads 32 KiB
off the end.

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

`fits_uncompress_table` under-allocates the decompression output
buffer for the variable-length-string column (24000 bytes in this
repro); zlib's `inflate`/`updatewindow` window copy then reads
32 KiB past the end of it.

**Expected:** funpack reconstructs the original table.
**Actual:** heap corruption / abort (ASAN: out-of-bounds read in
`fits_uncompress_table`'s output buffer).

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
