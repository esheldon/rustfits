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

---

## #3 — String-VLA (`1PA`) ZTABLE columns crash `funpack`

**rustfits issue:** [#9](https://github.com/esheldon/rustfits/issues/9)
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

### Draft cfitsio issue

> **Title:** funpack crashes (heap corruption) on tile-compressed
> tables with variable-length string (`1PA`) columns
>
> **Summary:** A binary table with a variable-length character
> column (TFORM `1PA`) compressed with `fpack -table` cannot be
> decompressed by `funpack` — it aborts with
> `realloc(): invalid next size` (SIGABRT). cfitsio crashes on its
> own output, so no third-party tooling is involved.
>
> **Reproduce** (cfitsio + CLIs only):
> ```c
> /* see pa_vla_funpack_crash.c: create a BINTABLE with one 1PA
>    column, write ~3000 variable-length strings, close. */
> ```
> ```
> fpack -table pa_vla.fits     # rc 0, writes pa_vla.fits.fz
> funpack pa_vla.fits.fz       # realloc(): invalid next size -> SIGABRT
> ```
>
> **Expected:** funpack reconstructs the original table.
> **Actual:** heap corruption / abort.
>
> **Environment:** cfitsio 4.6.0, Linux x86_64. Numeric VLA inner
> types (`1PE`, `1PJ`, …) compress and funpack fine — only the
> string (`A`) inner type triggers it.

---

## #5 — `GZIP_2` complex (`1C`/`1M`) columns: cfitsio can't funpack its own output

**rustfits doc:** `docs/internal/ztable.md` (referenced as issue #8
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

> **Title:** `fpack -g2` produces GZIP_2 complex-column tables that
> `funpack` rejects ("unsuitable data type")
>
> **Summary:** When a binary table with a complex column (TFORM
> `1C` or `1M`) is compressed with `fpack -g2 -table`, cfitsio
> writes the column with `ZCTYP='GZIP_2'`. `funpack` then fails
> with `status 414 — unexpected attempt to use GZIP_2 to compress a
> column unsuitable data type`. The GZIP_2 *encoder* skips
> byte-shuffling complex but still tags the column GZIP_2, while
> the GZIP_2 *decompressor*'s unshuffle switch has no `C`/`M` case.
>
> **Reproduce** (cfitsio + CLIs only):
> ```c
> /* see gzip2_complex_funpack.c: create a BINTABLE with one 1C
>    column, write ~2000 complex values, close. */
> ```
> ```
> fpack -g2 -table cplx.fits   # rc 0; ZCTYP1='GZIP_2'
> funpack cplx.fits.fz         # status 414, unsuitable data type
> ```
>
> **Expected:** either funpack decompresses it, or fpack declines
> to use GZIP_2 for complex columns (falling back to GZIP_1, which
> round-trips).
> **Actual:** cfitsio writes a file it cannot read back.
>
> **Environment:** cfitsio 4.6.0, Linux x86_64.
