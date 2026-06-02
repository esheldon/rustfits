# Dict-of-arrays (DoA) table layout — implementation plan

Working document — not user-facing.  Drafted 2026-06-01 from a design
discussion.  CLAUDE.md unchanged until the work actually lands; this
file is the staging area.

## Goal

Add a file-level `table_layout=` setting to `rustfits.FITS()` that
governs the return type of all table reads.  Default
`"structured"` preserves today's behavior (numpy structured ndarray);
`"dict"` returns `{column_name: ndarray}` (dict of contiguous
ndarrays).

```python
# Today: only structured returns
with rustfits.FITS(fname) as f:
    cat = f[1].read()           # structured ndarray
    sub = f[1][["RA", "DEC"]][:]  # structured ndarray

# After: choose at open time
with rustfits.FITS(fname, table_layout="dict") as f:
    cat = f[1].read()           # {"RA": ndarray, "DEC": ndarray, ...}
    sub = f[1][["RA", "DEC"]][:]  # {"RA": ndarray, "DEC": ndarray}
```

## Why a file-level setting

Three forms were considered:

1. Per-call kwarg (`hdu.read(layout="dict")`) — can't reach
   `hdu[cols][rows]` or `hdu[rows]`.  Rejected.
2. Sibling method (`hdu.read_dict()`) — same gap.  Rejected.
3. **File-level setting** (`FITS(fname, table_layout=...)`) — one
   decision applied uniformly to every read path.  Chosen.

The "spooky action at a distance" concern about a return-type-affecting
setting is real but mitigated by the setting living on `FITS()` itself,
right next to the filename — anyone reading the code sees it.

## Decisions banked

These were settled in the design discussion; capturing them so the
implementation doesn't re-litigate.

- **Default is `"structured"`.**  fitsio/astropy migration target —
  zero break on existing user code.  `table_layout="dict"` is opt-in.
- **Single-column reads return plain ndarray either way.**  A
  one-entry dict for `hdu["x"]` would be silly; today's behavior stays.
- **Row-level reads always return `np.void`, regardless of layout.**
  A row is intrinsically structured; named-field access matters at
  the row level; layout choice doesn't.  Affects:
    - `hdu[i]` (bare-int single row)
    - `for row in hdu:` and `hdu.iter()` row mode
  Allocating a dict-per-row would be wasteful for big tables.  If a
  user wants per-row dicts, add `hdu.iter(layout="dict")` later as
  the explicit form — not in scope for v1.
- **Multi-row reads always honor the layout setting.**  Includes:
    - `hdu.read()` / `hdu.read(rows=, columns=)`
    - `hdu[i:j]` (slice — even without column projection)
    - `hdu[[i, j, k]]` (fancy rows)
    - `hdu[[cols]][rows]` via `ColumnSubset`
    - `hdu.iter(chunksize=N)` chunk mode (chunks of multiple rows)
  Rule: "more than one row out → layout-controlled; single row →
  np.void."
- **MaskedArray under `"dict"`:** dict has plain ndarrays for
  unmasked columns and `numpy.ma.MaskedArray` only for columns
  where TNULL fired.  The numpy structured-mask wart disappears —
  this is the cleanest semantic win of the change.
- **Writes unchanged.**  `hdu.write(data)` already accepts dict,
  structured ndarray, or list+names.  `table_layout=` governs reads
  only.
- **Naming:** `table_layout="structured"` / `table_layout="dict"`.
  Symmetric and grep-able; leaves room for `"arrow"` / `"polars"`
  later without an API rename.
- **fitsio compat shim** (`rustfits.compat.fitsio`) hardcodes
  `table_layout="structured"` when opening, regardless of any user
  arg — wrapped fitsio code keeps working.

## File-by-file work

### Core types + plumbing

- **`src/common.rs`** (or `src/fits.rs`): add
  `pub(crate) enum TableLayout { Structured, Dict }` with
  `Default = Structured`.  Parse the kwarg string here (reject
  unknown values with a clear error listing the accepted forms).
  ~20 lines.
- **`src/hdu.rs`**: `HDU` base struct gains
  `pub(crate) table_layout: TableLayout` (Copy + Clone, no Arc
  needed — it's a 1-byte enum).  ~5 lines.
- **`src/fits.rs`**: `FITS::new` accepts `table_layout: Option<&str>`
  kwarg (default `"structured"`), parses to `TableLayout`, stores
  on `FITS`, and stamps each `HDU` at parse time
  (`parse_hdus_from_file`).  Subset objects already hold a parent
  ref, so they inherit transitively.  ~15 lines.

### BINTABLE read path

- **`src/hdu_table/read.rs`** `read_table`: today allocates one
  structured ndarray and fills via `arr[col_name][row] = value`.
  Add a parallel path that allocates a dict of N contiguous
  ndarrays (one per output column) and fills via
  `cols[col_name][row] = value`.  The per-cell loops are identical
  modulo the destination object.  Cleanest implementation: build the
  dict of per-column ndarrays first, then write into them — no
  structured-ndarray intermediate, no extra memcpy.  ~50 lines.
- **`src/hdu_table/hdu.rs`**: `ColumnSubset.read` / `__getitem__`
  check parent's layout, swap return shape.  `SingleColumnSubset`
  unchanged (always plain ndarray).  Bare-int `hdu[i]` unchanged
  (always np.void).  Slice / fancy `hdu[rows]` route through
  `read_table` so the layout switch flows through automatically.
  ~20 lines.

### ASCII table read path

- **`src/hdu_ascii_table/read.rs`** `read_ascii_table`: mirror the
  BINTABLE change.  ~50 lines.
- **`src/hdu_ascii_table/hdu.rs`**: `AsciiColumnSubset` parallel to
  `ColumnSubset`.  ~20 lines.

### ZTABLE (compressed) read path

- **`src/hdu_table_compressed/read.rs`** `read_compressed_table`:
  same pattern.  Tile reads currently build a structured ndarray;
  alternative builds dict-of-ndarrays.  ~50 lines.
- **`src/hdu_table_compressed/subset.rs`**:
  `CompressedColumnSubset` parallel.  ~20 lines.

### Iteration

- **`src/hdu_table/iter.rs`**: NO CHANGE.  Row mode always yields
  np.void; chunk mode yields the table-layout-controlled type
  (today's behavior is already correct for the dict case once
  `read_table` learns layout-awareness, because `TableIter` just
  calls `self.hdu.read(rows=slice, ...)` polymorphically).
- Document the row-vs-chunk asymmetry in the iter docstring.

### Writes

- Unchanged.  No code touched.

### fitsio compat shim

- **`rustfits/compat/fitsio.py`** (or wherever the shim lives):
  the `open` wrapper hardcodes `table_layout="structured"`.  ~3 lines
  + a comment explaining why.

### Helper considerations

- `build_numpy_dtype(py, columns, scale)` (used by accessors like
  `dtype`) stays unchanged — `TableHDU.dtype` still reports the
  structured dtype even under `"dict"` mode.  The dtype IS the
  schema, regardless of how data comes out.  Cheap symmetry.
- The shared `parse_columns` / `TableMeta` machinery is unaffected
  — meta is layout-agnostic.

## Test plan

### New file: `tests/test_table_layout_dict.py`

Parametrize over both layouts; cases:

- Whole-table read returns the expected shape (structured vs dict).
- `rows=` / `columns=` selection honors layout.
- `hdu[[cols]][rows]` via subset, with various row keys (int / slice
  / fancy).
- Bare-int `hdu[i]` returns np.void regardless of layout.
- `for row in hdu:` returns np.void regardless of layout.
- `hdu.iter(chunksize=N)` returns layout-controlled type.
- `mask_null=True` returns:
    - structured: `MaskedArray` wrapping structured ndarray
    - dict: dict with `MaskedArray` only for the column(s) that
      actually had TNULL fire; other columns are plain ndarray
- Single-column `hdu["x"]` returns plain ndarray either way.
- VLA columns (Object dtype) — dict form has the Object ndarray
  directly under the column name; structured form same as today.
- Invalid `table_layout="unknown"` rejected at `FITS()` with a
  clear message listing accepted values.

### Parametrize the existing suite

The `table_layout` choice is orthogonal to almost every existing
test.  Two options:

- A) Parametrize all existing `read()` / `__getitem__` tests across
  both layouts via a session-level fixture.  Doubles test count but
  catches every regression.
- B) Only parametrize a curated subset (each major read code path
  hit once).  Smaller test footprint.

Recommend (A) for the BINTABLE + ASCII + ZTABLE read test files
specifically; (B) elsewhere.

### Cross-tool

- Round-trip test: read a fitsio-written file under `"dict"`,
  convert back to structured ndarray, assert bit-equal to fitsio's
  own decode.  Catches any layout-swap bug that would silently drop
  bytes.

## Documentation

- **`docs/tutorial/tables.rst`** — new "Output layout" section near
  the top of the tables tutorial.  Explains the two choices, when
  to pick each, MaskedArray cleanness under `"dict"`, the
  row-vs-table rule.  ~30 lines.
- **`docs/tutorial/migration.rst`** — one paragraph in the fitsio
  porting section noting that `table_layout="dict"` makes rustfits
  more pandas/polars-shaped, link to the new tables.rst section.
- **CLAUDE.md** — add a short paragraph under "Table read
  Supported" once the work lands.  Roughly:
    > **Output layout.**  `FITS(fname, table_layout="dict")`
    > switches the multi-row return type from a numpy structured
    > ndarray to `{col_name: ndarray}`.  Single-row reads
    > (`hdu[i]`, iteration) still return `np.void` regardless.
    > Writes unchanged (already accept dict).

## Phasing

Each step is shippable on its own; after each, `pytest` is green
and any user code keeps working (the default never changes).
Recommended commit cadence:

1. **Foundation.**  `TableLayout` enum + `FITS` kwarg + plumb
   through to HDU base struct.  No behavior change yet (layout is
   stored but unread).  Tests: layout round-trips through
   `FITS(...)` → `hdu` reads back.  Bisectable.
2. **BINTABLE read.**  `read_table` + `ColumnSubset` honor the
   setting.  Tests for BINTABLE only.  Now `table_layout="dict"`
   genuinely works for BINTABLE files.
3. **ASCII table read.**  Mirror change.
4. **ZTABLE read.**  Mirror change.
5. **fitsio compat shim hardcode + final tutorial docs.**

Total: ~5 commits, ~2 days of focused work.

## Open questions, deferred

- **Per-HDU layout override** (e.g.
  `hdu.with_layout("dict")` returning a flavored handle).  Not in
  scope.  Add later if a user wants per-HDU control within one
  file — the file-level setting covers the common case.
- **`hdu.iter(layout="dict")` for per-row dicts.**  Skipped per
  the "single-row always np.void" decision.  Add later if a real
  workload asks; the discussion log in CLAUDE.md doesn't need to
  cover it preemptively.
- **`"arrow"` / `"polars"` / `"pandas"` layouts.**  Out of scope
  for this work.  The enum can grow new variants later without
  any user-facing rename.
- **Layout-aware `__setitem__`.**  Writes already accept dict.
  No work needed.

## What this work does NOT do

- Does not change the on-disk format or the read I/O path
  (same bytes, same syscalls).
- Does not change `hdu.dtype` (still reports the structured dtype
  — it's the schema).
- Does not break any existing fitsio-style user code.
- Does not improve read throughput.  Allocation cost is the same
  total (one structured ndarray of size `N*W` bytes ≡ N column
  ndarrays totaling `N*W` bytes); the per-cell fill cost is
  identical.
