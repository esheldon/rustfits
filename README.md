# rustfits

[![Documentation Status](https://readthedocs.org/projects/rustfits/badge/?version=latest)](https://rustfits.readthedocs.io/en/latest/)

A Python FITS library with the heavy lifting written in Rust.

## What it does

`rustfits` reads and writes [FITS](https://fits.gsfc.nasa.gov/) files — the
astronomical data format — from Python. The Python surface mirrors `fitsio`
conventions: open a file with `FITS(path)`, index into HDUs with `fits[i]` or
`fits["sci"]`, call `hdu.read()` for the data, and slice with `hdu[a:b, c:d]`.

What's there today:

- **Images** — read, slice, write, in-place edits, append rows
  (`extend`), BSCALE/BZERO and BLANK/MaskedArray support,
  unsigned-int trick (`u2`/`u4`/`u8`/`i1`).
- **Tables** — read/write structured arrays, dict, and list+names
  inputs; row / slice / fancy / column / cell / multi-column writes;
  `append` and schema edits (`insert_column` / `delete_column`);
  variable-length columns including string `PA`; bit-packed `X` /
  `PX` / `QX`.
- **Tile-compressed images** (ZIMAGE) — all five algorithms
  (GZIP_1, GZIP_2, RICE_1, HCOMPRESS_1, PLIO_1), quantized and
  unquantized floats, mutation (`__setitem__` + `extend`) without
  compounding quantization loss, `repack()` to reclaim orphans.
- **Tile-compressed tables** (ZTABLE) — full surface including VLA
  columns and the dual-descriptor heap.
- **Headers** — case-insensitive lookup, CONTINUE chains, HIERARCH
  long keys, batched updates via `FITSHeaderEdit`, in-place header
  grow, `CHECKSUM` / `DATASUM` / `ZHECKSUM` / `ZDATASUM`.
- **Cross-tool interop** — files written by `rustfits` round-trip
  bit-exactly through `astropy.io.fits` and `fitsio`. Files written
  by those tools (or by `cfitsio` / `fpack`) read back in `rustfits`
  unchanged.

## Quick examples

```python
import rustfits

# Read an image — auto-picks the first HDU with data.
img = rustfits.read("image.fits")

# Slice an existing image without loading the full array.
with rustfits.FITS("image.fits") as fits:
    image = fits["sci"].read()
    image = fits["sci"][:, :]

    stamp = fits["sci"][100:200, 50:150]

# Read a table; subset columns and rows.
with rustfits.FITS("catalog.fits") as fits:
    hdu = fits[1]

    # using numpy-style slicing
    tab = hdu[:]
    tabsub = hdu[0:100]
    ra = hdu['ra'][20:30]
    radec = hdu[['ra', 'dec']][[3, 5, 25]]

    # using the read() function
    tab = hdu.read()  # same as hdu[:]
    tabsub = hdu.read(columns=["ra", "dec"], rows=[3, 5, 25])

# Write an image or table to a new file (auto-detects).
import numpy as np
rustfits.write("out.fits", np.zeros((1024, 1024), dtype="f4"))

cat = np.zeros(100, dtype=[("ra", "f8"), ("dec", "f8")])
rustfits.write("cat.fits", cat)

# Update the data with numpy-style setitem
with rustfits.FITS(fname, 'r+') as fits:
    hdu = fits["table"]
    hdu[10:20] = updated_rows
    hdu['ra'] = new_values
    hdu[['ra', 'dec']][10:50] = new_radec
```

See the [tutorial](https://rustfits.readthedocs.io/en/latest/tutorial/index.html)
for a guided tour covering images, tables, compression, headers,
the error model, and known limitations.

## Documentation

Full documentation is hosted at
[rustfits.readthedocs.io](https://rustfits.readthedocs.io/).
The `latest` build tracks `main`; `stable` tracks the most recent
release tag.

To build the docs locally instead:

```bash
sphinx-build docs docs/_build/html
xdg-open docs/_build/html/index.html
```

Sphinx and the [Furo theme](https://pradyunsg.me/furo/) are listed
in `docs/requirements.txt` if you need them in your env.

## Building from source

There's no PyPI or conda-forge release yet; install from source.

The build needs:

- A Python environment with `numpy` and `maturin>=1.0,<2.0`.
- A Rust toolchain installed via [`rustup`](https://rustup.rs).
  Don't use conda's `rust` package — it interacts badly with
  PyO3's libpython linking.

The simplest setup uses conda for Python and rustup for Rust:

```bash
# 1. Clone.
git clone https://github.com/esheldon/rustfits
cd rustfits

# 2. Create / activate a Python env, then install the runtime
#    + build deps.
conda install --file conda-requirements.txt

# 3. Install the Rust toolchain (skip if you already have rustup).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 4. Build and install the editable wheel into the active env.
maturin develop          # debug build, for iteration
# OR
maturin develop --release  # optimized build, for actual use
```

After the build finishes, `import rustfits` works inside the
active env. Re-run the same command after any change to the
Rust code.

**Debug vs release.** `maturin develop` (no flag) produces an
unoptimized debug build: it compiles fast and is what you want
during a tight edit-build-test loop. `maturin develop --release`
turns on the full optimizer (LLVM `-O3`-equivalent, inlining,
SIMD, etc.) — slower to compile but the resulting `.so` runs
**5–50× faster** depending on workload. **Use `--release` for
anything other than development** — benchmarks, real data
reductions, scripts that touch lots of bytes. The debug build
is roughly 100× slower at decompressing tiles, for example,
which makes ZIMAGE reads feel broken if you forget the flag.

The release profile keeps line-table debug info
(`debug = "line-tables-only"` in `Cargo.toml`), so backtraces
still resolve to source lines without paying the dead-code-
elimination penalty of a true debug build.

## Running tests

To run the test suite or contribute, install the dev requirements
on top of the build env:

```bash
conda install --file conda-test-requirements.txt
```

Then:

```bash
pytest                 # Python tests
tools/cargo-test.sh    # Rust unit tests (wrapped to find libpython)
```

The wrapper `tools/cargo-test.sh` prepends the conda env's lib
directory to `LD_LIBRARY_PATH` so the PyO3-linked test binary can
find `libpython.X.so` — bare `cargo test` fails without it.

## Contributing

Contributions and bug reports are welcome.  Please feel free to open pull
requests or issues at https://github.com/esheldon/rustfits

`rustfits` is mostly written in rust.  If you don't know rust, you can
still make contributions.  The extensive CLAUDE.md file can be used
by claude code or other agents to help you fix bugs or add features.
Just start the agent, ask it to load from CLAUDE.md, and start working.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
