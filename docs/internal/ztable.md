# Tile-compressed tables (ZTABLE) — full implementation roadmap

*Extracted from `CLAUDE.md` to keep the always-loaded file lean.
Read this when working on `src/hdu_table_compressed/`.  The
CLAUDE.md entry "Tile-compressed tables (ZTABLE)" carries the
one-paragraph summary + a pointer here.*


ZTABLE is the BINTABLE counterpart of ZIMAGE: a normal BINTABLE
shell carrying tile-compressed column data, with the original
table's schema preserved via Z-prefixed cards.  Detection is
`ZTABLE=T`; on-disk layout (per `<cfitsio>/imcompress.c::fits_compress_table`):

- The compressed table has `NAXIS2 = num_tiles` and each
  user-visible column becomes a `1QB(maxlen)` heap-descriptor
  column.
- Per tile per column: the original bytes for that column over
  `ZTILELEN` rows are transposed to column-major, optionally
  byte-shuffled (GZIP_2), and compressed (RICE_1 / GZIP_1 /
  GZIP_2).  Each (tile, column) pair lands as its own
  variable-length heap blob.
- Header preserves the original schema via `ZNAXIS1`, `ZNAXIS2`,
  `ZPCOUNT`, `ZFORMn` (original TFORMn including repeat),
  `ZCTYPn` (per-column algorithm), `ZTILELEN`.  `TTYPEn`,
  `TDIMn`, `TUNITn`, `TZEROn`, `TSCALn`, `TNULLn` are preserved
  on disk unchanged.
- VLA columns get a dual-descriptor scheme: each cell is
  individually compressed AND the original/compressed
  descriptor pairs are themselves gzipped — relevant for
  Phase 4.

**Status:**

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Detection + `CompressedTableHDU` subclass + accessors + I/O stubs | ✅ Shipped |
| 2 | Whole-table read (fixed columns) across GZIP_1 + GZIP_2 + RICE_1 | ✅ Shipped |
| 3 | `read(rows=)` / `__getitem__` / column-subset objects / tile cache | ✅ Shipped |
| 4 | VLA-column read (dual-descriptor heap) | ✅ Shipped |
| 5 | Bulk write — `create_table_hdu(..., compress=...)` for fixed cols | ✅ Shipped |
| 6a | VLA write (dual-descriptor heap, ZPCOUNT, funpack interop) | ✅ Shipped |
| 6b | `append()` for fixed-column tables (merge into partial last tile) | ✅ Shipped |
| 6c-1a | `repack()` for fixed-column tables (streaming + staging fallback) | ✅ Shipped |
| 6c-1b | VLA `append()` (existing-cell copy + per-cell re-encode for new rows) | ✅ Shipped |
| 6c-2a | VLA `repack()` (streaming staging + dual-descriptor blob re-gzip) | ✅ Shipped |
| 6c-2b | Fixed-col row writes: `hdu[i]=record`, `hdu[a:b]=arr`, `hdu[[i,j,k]]=arr` | ✅ Shipped |
| 6c-2c | Fixed-col col/multi writes: `hdu["col"]=arr`, `hdu[[c1,c2]]=arr` | ✅ Shipped |
| 6c-2d | Stepped slices + subset-object writes (`hdu[a:b:s]=arr`, `hdu["name"][rows]=v`, `hdu[[a,b]][rows]=v`) | ✅ Shipped |
| 6c-2e | VLA `__setitem__` (all forms, decode → modify → re-encode dual-descriptor blob) | ✅ Shipped |

**Phase 1 — detection + accessors + stubs.**  Shipped.

- `header_has_ztable` in `src/hdu_table_compressed.rs` finds
  `ZTABLE=T`; `parse_hdus_from_file` in `fits.rs` checks ZTABLE
  AFTER ZIMAGE (defensive — they shouldn't both be set, but
  ZIMAGE wins if they are).
- `CompressedTableHDU` subclasses `TableHDU`, so
  `isinstance(hdu, TableHDU)` is True on a compressed-table
  HDU.  Construction chain: `HDU` → `TableHDU` → `CompressedTableHDU`.
  `TableHDU` was promoted to `#[pyclass(extends = HDU, subclass)]`
  to enable the chain (parallel to `ImageHDU` for ZIMAGE).
- **Accessors return the original-schema view.**  `nrows`,
  `__len__`, `dtype`, `colnames`, `units` all override the
  TableHDU getters and route through `synthesize_uncompressed_cards`,
  which builds a virtual cards list with `NAXIS1`←`ZNAXIS1`,
  `NAXIS2`←`ZNAXIS2`, `PCOUNT`←`ZPCOUNT`, `TFORMn`←`ZFORMn`
  and drops all Z-prefixed cards.  Without this substitution,
  `parse_columns` would trip on the on-disk `1QB` descriptors
  + preserved `TDIMn` cards (which it correctly rejects as
  TDIM-on-VLA).  Compression-specific accessors: `compression`
  returns `{col_name: ZCTYPn_value}`, `n_tiles` is the on-disk
  `NAXIS2`, `ztile_rows` is `ZTILELEN`.
- **I/O surface stubbed.**  `read`, `__getitem__`, `__setitem__`,
  `write`, `append`, `extend`, `repack`, `insert_column`,
  `delete_column`, `add_datasum`, `add_checksum`,
  `verify_datasum`, `verify_checksum` all raise
  `NotImplementedError` naming the phase that will land them.
  Schema-edit methods are documented as "not planned for the
  current roadmap" — rebuilding through fresh `create_table_hdu`
  is the workaround.
- **Test fixtures via `fpack -table`.**  Astropy doesn't expose
  a `CompTableHDU` writer; fitsio doesn't have a high-level
  wrapper for `fits_compress_table`.  `fpack -table` (CLI tool
  shipped with cfitsio) is the only widely-available writer,
  so the Phase 1 test module skips itself if `fpack` is not
  on PATH.  Fixtures cover scalar + multi-D subarray + S10
  string + per-column units.
- **Stub-only surface area is the source of truth for what each
  phase needs to fill in.**  Phase 2 implements `read` (and
  removes the stub raise); Phase 3 implements `__getitem__`;
  etc.  The NotImplementedError messages name the phase by
  number so contributors can grep `Phase N` to find what's
  pending.

Tests: `tests/test_table_compressed_accessors.py` (21 cases) —
detection, isinstance chain, all accessors, remaining stubs,
plain table unaffected, raw header cards still visible.

**Phase 2 — whole-table read.**  Shipped.

`CompressedTableHDU.read(*, columns=None, scale=True)` returns
the original (uncompressed) table as a numpy structured ndarray.
`rows=` and `mask_null=True` are rejected with NotImplementedError
pointing at Phases 3 and a follow-up respectively.  VLA columns
are rejected at read time (Phase 4).

Per-tile / per-column read loop in `read_compressed_table`
(`hdu_table_compressed.rs`):

1. Read this tile's descriptor row (`NAXIS1 = ncols * 16` bytes,
   each descriptor a 1QB pair = two big-endian i64).
2. For each selected column, read `nelements` compressed bytes
   from the heap, dispatch on `ZCTYPn` to `decode_gzip1` /
   `decode_gzip2` / `decode_rice`.
3. The decoders return NATIVE-order bytes (their image-side
   contract).  Byteswap back to FITS big-endian so the shared
   `convert_column_cell` from `hdu_table::read` can consume them.
   The double-swap (decoder swaps to native, we swap back to BE,
   convert_column_cell swaps to native) costs one trivial extra
   pass per (tile, column) — far cheaper than decompression
   itself, and avoids touching the ZIMAGE decoders.  Refactoring
   to a "leave BE" decoder mode would shave it but isn't worth
   the surface-area churn.
4. For each row in the tile, call `convert_column_cell` with the
   cell's BE bytes → native bytes in the output ndarray's field
   slice.  This is the same per-cell converter the uncompressed
   path uses, so all scaling (unsigned-trick, general TSCAL/
   TZERO, A/L/X) "just works" without duplication.

**RICE_1 scope.**  cfitsio's `fits_compress_table` only emits
RICE_1 for columns with TFORM letter B/I/J (calling
`fits_rcomp_byte` / `_short` / nothing for K).  The Phase 2
reader rejects RICE_1 on any other letter up front with a
"malformed file" message.  Float / K / complex columns appearing
with ZCTYP=RICE_1 indicate a non-conforming writer.

**Reused helpers (promoted to `pub(crate)` for this phase).**
`hdu_table::columns::{bytes_per_element, byteswap_unit, scaling_kind,
ScalingKind}`; `hdu_table::read::{convert_column_cell, numpy_field_layout,
read_descriptor, resolve_columns}`.  These are the per-cell
conversion + column-selection primitives the uncompressed read
path uses; the compressed path reuses them verbatim so the two
paths stay in sync (e.g. when TSCAL semantics evolve, both pick
up the change).

**Algorithm-default reminder for fixtures.**  fpack (cfitsio's
`fits_compress_table`) picks per-dtype defaults that are
non-obvious:

| Letter | Default ZCTYPn |
|--------|----------------|
| B (u1) | GZIP_1 |
| I (i2) | GZIP_2 |
| J (i4) | RICE_1 |
| K (i8) | RICE_1 |
| E, D, C, M | GZIP_2 |
| A (string), L (bool) | GZIP_1 |

To override, write a `FZALG<n>` keyword to the source file
before fpack runs.  To set `ZTILELEN`, write `FZTILELN` on the
source HDU (fpack has no CLI flag for it — surprising and
worth noting in the test helper).

**Complex columns (C / M) — byte-flat, GZIP_1 default (fixed
2026-05-29, issue #8).**  rustfits compresses/decompresses complex
columns as **byte-flat** (no GZIP_2 byte-shuffle, bytepix 1), letting
`convert_column_cell` do the component-wise byteswap.  This matches
cfitsio's *write* side, whose GZIP_2 shuffle dispatch silently skips
complex (it shuffles only 2/4/8-byte I/J/E/D/K).  Two bugs were fixed:
the old code shuffled/byteswapped complex as one 8/16-byte unit, which
(a) swapped real/imag on `c8` read and (b) hit "unsupported bytepix 16"
on `c16`.  **rustfits defaults complex to GZIP_1, not GZIP_2**, because
cfitsio's GZIP_2 table *decompressor* errors on complex — its
unshuffle `switch(colcode)` has no C/M case and falls into a
`default: "...unsuitable data type"` → `DATA_DECOMPRESSION_ERR`
(`imcompress.c`), so even cfitsio can't `funpack` its own
GZIP_2-complex output.  GZIP_1-complex round-trips in both rustfits and
cfitsio.  Explicit `compress={"col": "GZIP_2"}` on complex still
round-trips in rustfits but won't `funpack` (the cfitsio read bug).

Tests: `tests/test_table_compressed_read.py` (17 cases) —
mixed-dtype round trip; per-algorithm coverage (RICE_1 on i4 +
the GZIP defaults for u1/i2/f4/f8/A); multi-tile + partial last
tile; unsigned-int trick round trip; `scale=False` returns raw
stored dtype; `columns=` subset (decompresses only selected,
reorders, case-insensitive); multi-D subarray TDIM round trip;
compressed table not at index 1 (programmatic lookup).

**Phase 3 — slicing, `__getitem__`, column-subset, tile cache.**
Shipped.

`read(rows=...)`: accepts a Python slice (with arbitrary step,
including negative — handled by `resolve_rows`) or an iterable
of ints (negative wrap-around, deduped — first occurrence wins,
output order preserved).  Combined with `columns=` for
two-dimensional subsets.

`__getitem__` dispatch mirrors `TableHDU.__getitem__` exactly
(reuses `classify_table_key` + `TableKey` from `hdu_table::hdu`,
promoted to `pub(crate)` for this purpose):

  - `hdu[i]` → 0-d structured record (numpy.void).
  - `hdu[i:j[:s]]` → structured ndarray.
  - `hdu[[i, j, k]]` → fancy-row ndarray (user order preserved).
  - `hdu["col"]` → `CompressedSingleColumnSubset`.
  - `hdu[["a", "b"]]` → `CompressedColumnSubset`.

Subset objects are READ-ONLY (no `__setitem__`).
`hdu["col"][rows]` returns a plain (non-structured) ndarray of
just that column's values — matches `SingleColumnSubset` on the
uncompressed side.

**Row planner (`RowPlan`).**  Buckets the requested disk-row
indices into per-tile lists of `TileRowRequest { in_tile_offset,
output_row }`.  The walk iterates tiles in increasing order
(best disk locality for the descriptor + heap reads), and
within each tile walks each selected column.  For each
`(tile, col)` the cache is consulted; on miss, decompress +
byteswap-to-BE + insert into cache.  Then for each request in
that tile, copy + scale the cell into `out[output_row].field(C)`.

The "all rows" path uses a flag rather than materializing N
requests up front; `tiles_with_requests` synthesizes sequential
requests per tile lazily, so the planning cost is O(n_rows)
either way.

**Tile cache (`ColumnTileCache`).**  Bytes-bound LRU keyed by
`(tile_idx, col_idx)` packed into a `CacheKey` struct.  Value is
`Arc<Vec<u8>>` of decompressed BE bytes for that (tile, column)
slab — so callers work without holding the inner mutex and
concurrent readers don't serialize through long decode runs.

Per-(tile, column) granularity is the right knob for tables:
reading just `hdu["col"][i:j]` only decompresses that column's
tiles; subsequent reads of sibling columns or nearby rows reuse
adjacent cache entries.  This contrasts with image tiles (one
cache entry per tile) — same primitive shape, finer key.

Default capacity 32 MiB; accessors match
`CompressedImageHDU`:

  - `hdu.tile_cache_size` / `set_tile_cache_size(bytes)` —
    capacity in bytes; 0 disables caching entirely.
  - `hdu.tile_cache_used` — current bytes held.
  - `hdu.clear_tile_cache()` — drop all entries, keep capacity.

`set_tile_cache_size` to a value below current usage triggers
LRU eviction until it fits.

**Reused helpers (additionally promoted `pub(crate)`).**  Phase 3
also promotes `hdu_table::read::resolve_rows`,
`hdu_table::hdu::classify_table_key`, and the `TableKey` enum to
`pub(crate)` so the compressed-table dispatch reuses the same
key-classification logic as the uncompressed path.

Tests: `tests/test_table_compressed_read_slice.py` (27 cases) —
`read(rows=...)` for slice / stepped slice / iterable / negative
indices / duplicates / dedup / combined with `columns=`;
`__getitem__` for int / negative int / slice / stepped /
fancy / col-name / col-list (with subset-class type checks);
chained subset reads (single + multi) including reorder;
cache default / warming / repeat-from-cache / clear /
set-to-zero / set-smaller-evicts; cross-call correctness when a
partial read is followed by a full read; out-of-range int +
empty-iterable rejection.

**Fixture guard worth noting.**  cfitsio's `fits_compress_table`
silently *copies the HDU verbatim* (no compression) when the
table's data extent is under 5760 bytes (2 BLOCK_SIZE blocks).
The Phase 3 test helper asserts the produced HDU IS
`CompressedTableHDU` so future tests don't silently exercise
the uncompressed path while pretending to test ZTABLE.

**Phase 4 — VLA-column read (dual-descriptor heap).**  Shipped.

For each VLA column per tile, the column's heap blob (referenced
by the 1QB main-row descriptor) is GZIP_1-compressed regardless
of ZCTYPn — ZCTYPn governs only the *inner per-cell* compression.
After GZIP decompression the blob is exactly
`rowspertile * width_orig + rowspertile * 16` bytes, laid out as
two concatenated descriptor arrays:

  - First `rowspertile * width_orig` bytes: ORIGINAL P/Q
    descriptors from the user-visible BINTABLE.  `vlalen` here is
    the number of inner-type elements (the user-visible count).
    Original heap-offset field is irrelevant on read (the original
    heap doesn't exist in the compressed file).
  - Next `rowspertile * 16` bytes: COMPRESSED-side Q descriptors.
    `cvlalen` is the number of compressed bytes for that cell;
    `cvlastart` is the offset of those bytes inside the
    compressed table's heap.

Per-row decompression:

  1. Read `cvlalen` bytes from heap at `heap_start + cvlastart`.
  2. **Uncompressed fallback**: if `cvlalen == vlalen * elem_size`,
     the cell was stored raw (cfitsio's "compression didn't help"
     branch) — those bytes are the original BE inner-element bytes
     verbatim, no decoder invocation.
  3. Otherwise decompress per ZCTYPn (RICE_1 for B/I/J inner;
     GZIP_1 / GZIP_2 for everything).
  4. Hand the resulting BE bytes to the shared
     `build_var_cell_value` (promoted to `pub(crate)` for this
     phase) which builds the per-cell numpy ndarray (or str / bytes
     for A) with byteswap + scaling + ASCII validation handled the
     same way the uncompressed read path handles them.

**Caching.**  Per-(tile, col) descriptor blob goes into the same
`ColumnTileCache` that fixed columns use — cache key is identical
(`CacheKey(tile_idx, orig_idx)`).  Per-cell decompressed bytes
are NOT cached: cells can be tiny but there are many, and
VLA-of-images patterns would blow the budget.  Each cell read
decompresses fresh; the heap-blob cache is what amortizes the
GZIP-decompress + descriptor parse across multiple row requests
in the same tile.

**`gzip_decompress_bytes`.**  Phase 4 needed a "raw gzip decompress
to a known length, no byteswap" primitive distinct from
`decode_gzip1`/`decode_gzip2` (which both byteswap to native at
the end).  The descriptor blob is a packed array of BE
descriptors that the per-cell loop feeds straight to
`read_descriptor` — byteswapping would corrupt it.  Implemented
inline in `hdu_table_compressed.rs` using `flate2::read::GzDecoder`
directly.

Tests: `tests/test_table_compressed_read_vla.py` (16 cases) —
whole-table VLA round trip across u1 / i2 / i4 / i8 / f4 / f8
inner dtypes (covers all three algorithms fpack picks
per-dtype); empty cells; multiple VLA columns in one table;
VLA mixed with fixed columns; multi-tile read; slice across
tiles; fancy-row read preserving user order; single row via
`__getitem__`; `columns=` subset; single-column subset chained.

**Phase 5 — bulk write (fixed columns).**  Shipped.

`create_table_hdu(dtype, nrows, *, compress=..., ztilelen=None)`
routes to the ZTABLE path when `compress` is non-None.  Accepted
`compress` shapes:

  - `False` / `None` (default) — uncompressed (existing path,
    unchanged).
  - `True` — compressed with cfitsio defaults per dtype.
  - `"GZIP_2"` / `Gzip2()` (string alias or class instance) —
    same algorithm everywhere, validated per column.
  - `{"col": "RICE_1", "other": Gzip2(level=9)}` — per-column
    overrides; unspecified columns get cfitsio defaults.  Dict
    values can be string aliases or config-class instances.

Strict validation: an algorithm that isn't legal for a column's
dtype raises with the allowed-list ("`b` is f8 so RICE_1 isn't
allowed; permitted: GZIP_1, GZIP_2").  No silent fallback —
cfitsio's tolerant behavior silently gives you a different
algorithm than asked, which is worse than asking the user to fix
the call.  `compress=True` is the escape hatch when you don't
want to think about it.

Per-column defaults match cfitsio (imcompress.c around line 8261):

  | TFORM | numpy | Default | Allowed |
  |---|---|---|---|
  | B | u1 | GZIP_1 | GZIP_1, RICE_1 |
  | I | i2 | GZIP_2 | GZIP_1, GZIP_2, RICE_1 |
  | J | i4 | RICE_1 | GZIP_1, GZIP_2, RICE_1 |
  | K | i8 | GZIP_2 | GZIP_1, GZIP_2 |
  | E | f4 | GZIP_2 | GZIP_1, GZIP_2 |
  | D | f8 | GZIP_2 | GZIP_1, GZIP_2 |
  | C | c8 | GZIP_2 | GZIP_1, GZIP_2 |
  | M | c16 | GZIP_2 | GZIP_1, GZIP_2 |
  | L | b1 | GZIP_1 | GZIP_1 only |
  | A | str | GZIP_1 | GZIP_1 only |
  | X | bit | GZIP_1 | GZIP_1 only |

`ztilelen=None` defaults to cfitsio's `max(1, min(nrows,
10_000_000 / row_width))` — ~10 MB worth of rows per tile.

**String aliases on the image side too.**  This phase also rolled
out string-alias support on `create_image_hdu(compress="GZIP_1")`
(case-insensitive, accepts cfitsio synonyms like `GZIP`,
`RICE_ONE`, `HCOMPRESS`).  Shared normalizer is
`CompressionConfigKind::from_str` (next to `from_pyany` in
`zimage/compression_config.rs`).  Strings are useful both as
standalone shortcuts on images and as values in the table per-
column dict.

**`write()` accepts the same three input forms as the uncompressed
`TableHDU.write`**: structured ndarray, dict `{name: ndarray}`, or
list/tuple of ndarrays + `names=[...]`.  All three normalize via
`extract_per_column_inputs` (the same helper the uncompressed VLA
write uses for per-column dispatch — already `pub(crate)`).

**Per-cell transform reuse.**  Per-tile per-column slab encoding
uses the shared `apply_transform_cell` from `hdu_table/write_fixed`
(the slow path of the uncompressed write).  That single dispatch
covers byteswap (Identity), unsigned-int trick XOR (UnsignedXor),
bool 0/1 → 'F'/'T' ASCII (BoolToLogical), S<n> verbatim copy
(BytesCopy), and U<n> UTF-32 → 7-bit ASCII (UnicodeToAscii).
Replicating the conversion logic in the compressed path would
have invited drift; reusing the existing one means TSCAL/TZERO
behavior evolves in one place for both write paths.

**Encode loop.**  `write_compressed_table_data` is per-tile,
per-column.  For each `(tile, col)`:

  1. Apply the per-cell transform from the column's native-order
     source bytes into a BE slab of size `rows_in_tile *
     byte_width`.
  2. Encode the slab per the column's ZCTYPn (using the existing
     `encode_gzip1` / `encode_gzip2` / `encode_rice` primitives
     from `zimage/`).
  3. Append the compressed blob to the heap at the running
     `heap_cursor`; record `(blob_len, heap_cursor)` in the
     descriptor table.
  4. `grow_file_to_at_least` extends the data section in
     block-aligned chunks as the heap grows; for non-last HDUs
     this shifts later HDUs forward via the shared primitive.

After all tiles are encoded, the in-RAM descriptor table is
written at `data_offset`, the trailing block is zero-padded, and
the header's `PCOUNT` card is rewritten to the final heap size
via the standard disk-write-before-commit + taint pattern.

**Stored compress configs.**  `CompressedTableHDU::new` gained a
new `compress_configs: Option<Vec<CompressionConfigKind>>` field
holding the user's per-column configs from create time.  Lets
write-only params like `Gzip1(level=9)` survive within a session
through `.compression`.  Reopened HDUs have `None` here and
`.compression` falls back to the dict-of-strings from ZCTYPn.

Tests: `tests/test_table_compressed_write.py` (27 cases) —
every `compress=` shape (True / False / string / class / dict),
default per-dtype matrix, invalid-algorithm rejection,
image-only-algorithm rejection, three input forms (structured
ndarray / dict / list+names), dtype round trip for every fixed
scalar type, unsigned-int trick (u2 / u4), subarray TDIM,
string columns, default vs explicit ztilelen, **byte-exact
funpack interop** (cfitsio decompresses our files and we get
the original BINTABLE bit-exactly).

Plus `tests/test_image_compressed_write_gzip.py`
(11 new cases) covering image-side string aliases including
cfitsio synonyms and the case-insensitive match.

**Phase 6a — VLA-column write (dual-descriptor heap).**  Shipped.

Lifts the Phase 5 "fixed-only" restriction on the table-side
write path.  `create_table_hdu(dtype, nrows, *, compress=...,
var_dtypes={...})` now accepts both forms, producing a
ZTABLE-shaped HDU with the VLA columns ready to encode.
`hdu.write(data)` per-tile per-VLA-column:

  1. For each row in the tile, serialize the cell's BE bytes via
     the shared `serialize_vla_cell` (same helper the uncompressed
     VLA write uses).
  2. Encode per ZCTYPn (RICE_1 / GZIP_1 / GZIP_2 — the same
     algorithms allowed for VLA inner types as for fixed cols).
  3. **Uncompressed fallback** (cfitsio's
     `dlen < vlamemlen` branch in imcompress.c line 8508):
     when the compressed output is NOT smaller than the raw cell,
     store the raw BE bytes instead.  Phase 4 read already
     handles this by detecting `cvlalen == vlalen * elem_size`.
  4. Build the dual-descriptor blob (rowspertile original P/Q
     descriptors + rowspertile compressed Q descriptors), GZIP_1
     it, and write to the heap.  The main-table 1QB descriptor
     for this (tile, col) points at the gzipped blob.

**Original-descriptor offsets matter.**  cfitsio's
`fits_uncompress_table` uses the *original* P/Q descriptors'
offsets to position cells in the reconstructed uncompressed
heap (`ffpbyt(outfptr, vlamemlen, uncompressed_vla, status)` at
`heapstart + vlastart`).  Setting them all to 0 collides every
cell at offset 0.  We pre-compute the original-heap layout via
`plan_vla_heap_layout` (the same helper the uncompressed VLA
write uses) and emit the matching offsets — so a
funpack-decompressed file is byte-equivalent to a fresh
`create_table_hdu` + `write` without compress.

**ZPCOUNT must be the original heap size.**  funpack's
decompressor copies the source's ZPCOUNT verbatim onto the
output's PCOUNT.  We previously emitted ZPCOUNT=0 at create
time (no data yet); fixed by patching ZPCOUNT alongside PCOUNT
in the post-write header rewrite to the total original-heap
bytes (returned by `plan_vla_heap_layout`'s cursor).  Fixed-only
tables keep ZPCOUNT=0.

**Promoted to `pub(crate)`.**  `validate_vla_cell`,
`serialize_vla_cell`, `plan_vla_heap_layout`, `write_descriptor`,
and `VlaCellPlan` from `hdu_table::write_vla` — shared with the
uncompressed VLA write path so per-cell semantics evolve in one
place.

Tests: `tests/test_table_compressed_write_vla.py` (14 cases) —
inner dtype matrix (u1/i2/i4/i8/f4/f8) covering all three
per-dtype default algorithms; empty cells; multiple VLA columns;
mixed VLA + fixed columns; multi-tile VLA writes; PA (string
VLA) columns; per-column compression overrides on VLA cols;
**byte-exact funpack interop** (cfitsio decompresses our
VLA-bearing files and round-trips the data through fitsio).

**Phase 6b — append for fixed-column tables.**  Shipped.

`CompressedTableHDU.append(data, *, names=None)` extends the
table with new rows.  Accepts the same three input forms as
`TableHDU.append` (structured ndarray / dict / list+names).
`extend()` is the symmetric alias.

**Merge-into-last-partial semantics.**  If the existing last
tile has fewer than ZTILELEN rows:

  1. Decode the existing last tile per column (BE bytes, via a
     stripped-down read path that stops before the byteswap-to-
     native that `convert_column_cell` does).
  2. Concatenate the first M new rows (M = min(append_nrows,
     ZTILELEN - last_tile_rows)) via the shared
     `apply_transform_cell`.
  3. Re-encode the merged tile per ZCTYPn, append the new blobs
     to the heap end.  Old last-tile blobs become orphans —
     PCOUNT grows monotonically until a `repack()` reclaims them
     (shipped as Phase 6c-1a for fixed and 6c-2a for VLA).

Remaining rows after the merge become fresh tiles encoded
exactly like the bulk write path.  This maintains the FITS Tile
Compression Convention's "all tiles same size except the last"
invariant so funpack reads back correctly.

**Block-alignment gotcha (debugged + fixed).**  The data section
must end at a `BLOCK_SIZE` boundary so subsequent HDU headers
stay aligned.  Initial naive implementation shifted the file
tail by `delta_desc_bytes` (typically a few tens of bytes, not
block-aligned), which:

  - Left HDU N+1's header at a non-aligned offset.
  - The append's tail-pad write then overwrote the start of
    HDU N+1 with zeros.

Fixed by using `grow_file_to_at_least` (rounds to block,
handles last vs non-last HDU via set_len vs shift_file_tail)
plus a within-file `relocate_region_forward_local` to slide the
existing heap forward to its new position after the larger
descriptor table.  No tail-pad write — the bytes between
`heap_cursor` and the next block boundary are either zeros
(last HDU, OS set_len) or HDU N+1's shifted content (non-last);
either way don't overwrite them.

**VLA append**: shipped as Phase 6c-1b — see that section for
the dual-descriptor merge mechanics.  Fixed-only append kept
its own dispatcher; the VLA-aware branch lives alongside it
and is selected by `columns.iter().any(|c| c.var_kind.is_some())`.

**Cache invalidation.**  Cleared after each append (cheaper
than per-tile invalidation; append is rare-vs-read in any
realistic workflow).

Tests: `tests/test_table_compressed_append.py` (14 cases) —
merge-only, exact-fill, merge + new tile, no-merge (full last
tile), multiple appends accumulating, three input forms,
`extend()` alias, zero-row no-op, **non-last-HDU preserves
trailing HDU**, **funpack byte-exact interop**, VLA rejection,
PCOUNT grows after merge, ZNAXIS2 + NAXIS2 update correctly.

**Shared encode primitives (refactor done post-6b).**  Three
helpers in `hdu_table_compressed.rs` carry the per-column +
per-tile work for both write and append (and any future
mutator):

  - `ColPrep` struct — unified per-column metadata (RawBuffer
    pin + src_total_size + WriteTransform + encoder params).
  - `prepare_fixed_column(...)` — validates one input ndarray
    against the column's expected per-cell shape, builds the
    `ColPrep`.  Used by both write and append.
  - `encode_be_slab_to_heap_and_record(...)` — takes a
    pre-built BE slab, encodes per algorithm, writes blob to
    heap, fills the descriptor entry, returns updated
    `heap_cursor`.  Used by write's per-tile loop, append's
    new-tiles loop, AND append's merge branch (where the slab
    is decoded-old + new-transformed bytes).
  - `build_and_encode_tile_col(...)` — wrapper that builds the
    BE slab via `apply_transform_cell` from a `ColPrep`'s
    source bytes, then defers to
    `encode_be_slab_to_heap_and_record`.  Used by both write's
    per-(tile, col) loop and append's new-tiles loop.

Net result: 33 lines removed from the file, 200 lines of
duplication eliminated.  The merge branch in append keeps its
own per-cell-transform loop (it builds the slab differently —
existing BE bytes ++ new transformed bytes) and uses
`encode_be_slab_to_heap_and_record` to finish.

**Phase 6c-1a — `repack()` for fixed-column compressed tables.**
Shipped.

`CompressedTableHDU.repack()` reclaims heap orphans (from
append-with-merge and any future mutation that always-appends).
Walks descriptors in scan order, computes each live blob's
compact-heap position, moves bytes from old → new with chunked
I/O.

**Two move strategies; same plan-build, descriptor-rewrite, file-
shrink + PCOUNT-commit code; only the heap I/O differs.**

  - **Fast path (in-place streaming)**: blobs read in old-offset
    order, written to their new positions in place.  Requires
    `new_offset[i] + length[i] <= old_offset[i+1]` for every
    adjacent pair (so writes never clobber unread bytes).
    Holds for the post-merge orphan pattern that Phase 6b
    produces.  Cost: `~sum_of_live_blob_bytes_that_move` of I/O
    — typically just the rewritten last tile (~10 MB).
  - **Slow path (staging at end-of-file)**: copy blobs to a
    staging area appended past the current heap (writes go to
    fresh space — always safe), then copy staging back to the
    final in-heap positions (front-to-back; dst < src by
    `current_pcount`).  Cost: `~2 × new_pcount` of I/O.  Used
    when the fast path's safety check fails — handles any
    orphan pattern, including the arbitrary ones that
    `__setitem__` will produce in Phase 6c-2.

The slow path is the contract: works for any future mutator.
The fast path is an optimization for the merge-orphan common
case.  A single runtime check picks between them; same code
path for everything else.

**Memory bound**: ~1 MiB chunk + the descriptor table (`n_tiles
× ncols × 16` bytes; KB to a few MB) + the per-blob move-plan
vector (~32 bytes per live blob).  **No heap-in-RAM** — even a
10 GB compressed heap repacks within this budget.

**Cache invalidation**: descriptors all rewrite to new heap
offsets, so the per-(tile, col) tile cache is cleared at the
end of repack.

**Block-aligned shrink**: `new_hdu_end = data_offset +
round_up_to_block(desc_bytes + new_pcount)`.  For the last
HDU, `set_len(new_hdu_end)` trims.  For non-last,
`shift_file_tail_backward_and_update_offsets` at the next
HDU's current offset with `delta = next_off - new_hdu_end`
slides everything after back into place.  (Earlier bug: I
initially used `delta = file_len - new_hdu_end`, which
double-counted the trailing HDU's own size.  Fixed.)

**Promoted to `pub(crate)`**: `repack_compressed_table_heap`,
`stream_copy_in_file` (the chunked read-then-write primitive
used by both fast and slow paths).

**Scope limitation — VLA tables rejected.**  The dual-descriptor
heap layout used by VLA columns requires indirection-aware
rewriting (per-cell compressed bytes referenced from inside
each tile's GZIP_1'd descriptor blob).  Deferred until VLA
repack is genuinely needed.

Tests: `tests/test_table_compressed_repack.py` (9 cases) —
PCOUNT shrinks after merge orphan, no-op when compact,
multiple appends accumulating then one repack, last-HDU file
shrinks via set_len, **non-last-HDU preserves trailing HDU**
(catches the delta-computation bug), cache cleared, funpack
interop on repacked file, VLA rejection, same-handle + reopen
parity.

**Phase 6c-1b — VLA `append()` on compressed tables.**  Shipped.

`CompressedTableHDU.append()` now handles tables with VLA
columns.  Existing per-cell compressed bytes on the merge tile
are copied verbatim (no decompress / re-compress, no
precision loss for lossy inner algorithms); merge-tile new
rows go through the per-cell encode + uncompressed-fallback
path; spillover into fresh tiles reuses the Phase 6a per-tile
encoder.

**Pre-mutation snapshot.**  For each VLA col, the existing
last tile's dual-descriptor blob is fetched + GZIP_1-decompressed
into a `VlaMergeOldBlob { decompressed, width_orig, rowspertile }`
BEFORE any file mutation.  This snapshot is what later supplies
the existing rows' original descriptors (preserved verbatim)
AND their (cvlalen, cvlastart_old) pairs (used to drive the
per-cell stream copy into the new heap).  Done pre-mutation so
the read happens against the still-valid old heap position; the
upcoming shift relocates the per-cell bytes themselves but their
relative offsets in `cvlastart_old` still resolve correctly via
the new `heap_start_offset`.

**Original-heap planner.**  `plan_vla_heap_layout` is called
with `heap_start_offset = current_zpcount` so the new rows'
original-descriptor offsets extend the existing
original heap rather than overlapping it.  funpack copies
ZPCOUNT to the reconstructed output's PCOUNT and uses these
offsets to place each cell; collisions corrupt the funpack
output (this is the same gotcha that motivated the original
6a planner).  ZPCOUNT is rewritten at the end to the new
cursor value.

**Per-cell stream copy.**  `stream_copy_in_file` (the same
~1 MiB-chunked primitive added by repack) copies each existing
cell's compressed bytes from its old heap position to the new
heap end.  Source range always ends before destination starts
(source < `current_pcount` ≤ `heap_cursor` ≤ destination), so
no overlap-safety concerns.  Old positions become orphans (in
the same heap region the merge-tile's old dual-descriptor blob
is also orphaning); `repack()` would reclaim them — once VLA
repack lands (6c-2 follow-up).

**`grow_file_to_at_least` bug fixed in the same change.**  The
function used to compare `want_end <= file_len` to decide
whether a grow was needed.  That's wrong for non-last HDUs:
`file_len` includes the trailing HDU's bytes, so a write that
overlaps a trailing HDU's region passed silently as long as
the overlap fit inside the block-alignment padding, and
corrupted the trailing HDU once it exceeded the padding.  Fixed
fixed-only append tests passed because their per-tile growth
fit in padding; VLA append's larger per-cell growth tripped it.
Fix: compare against `next_hdu_start` (the layout query already
needed for the shift branch) when one exists, falling back to
`file_len` for the genuine last-HDU case.

**Validate-then-mutate.**  VLA input is dtype-checked (Object
kind, length match) at the top of the prep loop, before any
file shift or heap-relocate, so dtype errors leave the file
untouched.  Per-cell `validate_vla_cell` is called twice (once
during `plan_vla_heap_layout`, once during encode); both
acceptable since the call is cheap and the second pass keeps
the encode loop simple.

Tests: `tests/test_table_compressed_vla_append.py` (12 cases) —
append into fresh tile, merge-only, merge-and-spill, mixed
fixed + VLA, empty cells, multiple VLA cols, ZPCOUNT
accounting (bytes match the uncompressed-cell-bytes sum),
multiple sequential appends, **non-last HDU preserves trailing**
(catches the grow_file_to_at_least bug above), string VLA
('PA') append, funpack interop.

**Phase 6c-2a — VLA `repack()` on compressed tables.**  Shipped.

`CompressedTableHDU.repack()` now handles tables with VLA
columns (including mixed fixed + VLA).  Reclaims orphans from
append-with-merge (and the upcoming __setitem__ once 6c-2b
lands).  Streaming, staging-only path — no in-place fast path
for VLA because the per-cell+per-blob interleaving makes the
safety check substantially more complex and the staging path's
2× new_pcount of I/O is already acceptable.

**Mechanics.**  Walk each (tile, col) in scan order:

  - **Fixed col**: stream-copy the existing blob (heap_start +
    old_offset, length=descriptor.nelements) to the staging area
    at staging_cursor; record (staging_cursor, length) for the
    new main descriptor.
  - **VLA col**: read + GZIP-decompress the dual-descriptor blob
    from the OLD heap into RAM.  For each row, stream-copy the
    cell's compressed bytes from `heap_start + cvlastart_old` to
    `staging_start + staging_cursor`, then rewrite the row's
    compressed-descriptor in the in-RAM blob with the new
    cvlastart.  After all rows: re-GZIP the blob (now with new
    cvlastart values inside) and stage-write it; record the new
    main descriptor at the blob's staging position.

Then one big front-to-back stream copy moves
`staging[0..new_pcount]` to `heap_start[0..new_pcount]` (safe
because dst < src by exactly `current_pcount`).  Then descriptor
table rewrite, file shrink (set_len for last HDU,
shift_file_tail_backward for non-last), PCOUNT update, cache
clear — mirroring the fixed repack tail.

**ZPCOUNT invariant.**  Repack reorganizes COMPRESSED heap
bytes; cell `nelements` values are unchanged, so the
ORIGINAL-heap size (the sum of `nelements * elem_size` over
live cells, recorded in ZPCOUNT) doesn't move.  Don't touch
ZPCOUNT.

**Bug fixed in the same change: descriptor-bytes off-by-N in
`grow_file_to_at_least` `want_total`.**  The staging area
starts at `data_offset + desc_bytes + current_pcount`, but the
initial `want_total` I computed forgot `desc_bytes`.  For a
typical multi-col 1QB descriptor row that's 64 bytes — small,
but enough that the trail-HDU shift under-counted by one block,
letting staging writes clobber the start of HDU N+1's header.
Same-handle reads passed (in-memory layout was right); reopen
failed because the on-disk XTENSION card was overwritten by
heap floats.  The non-last HDU test caught it.

**Memory bound.**  ~1 MiB chunk + one decompressed dual-desc
blob at a time (`rowspertile * (width_orig + 16)` bytes) + one
gzipped blob held briefly while staging + descriptor table
(few KB to few MB) + per-(tile, col) move-plan vector
(~32 bytes per entry).  No heap-in-RAM allocation.  Staging
temporarily roughly doubles the file's heap region; reclaimed
on shrink.

**Mixed-table handling.**  Pure-fixed tables take the
streamlined fixed-only path (with fast path + slow path);
any-VLA tables take the VLA-aware path which handles both kinds
of columns uniformly per (tile, col).

Tests: `tests/test_table_compressed_vla_repack.py` (14 cases) —
PCOUNT shrinks after merge orphan, no-op when compact, multi-
append accumulation then one repack, last-HDU file shrinks,
**non-last HDU preserves trailing HDU** (catches the want_total
bug above), mixed fixed + VLA cols, multiple VLA cols, ZPCOUNT
preserved, cache cleared, empty cells survive, **funpack
byte-exact interop on repacked file** (strongest correctness
check), repack→append→repack composition, string VLA ('PA')
column, same-handle + reopen parity.

**Reference source.**  cfitsio's `fits_compress_table` and
`fits_uncompress_table` in `<cfitsio>/imcompress.c` (around
line 8003 and 8695 respectively).  Read/write loops there are
the byte-exact spec.

**Phase 6c-2b / 6c-2c / 6c-2d / 6c-2e — `__setitem__` on
compressed tables.**  Complete (full surface, both fixed and
VLA columns).

Surface — nine dispatch forms across the main HDU and the two
subset pyclasses, dispatched by `classify_setitem_key` (shared
with uncompressed-side `TableHDU.__setitem__`):

**Row-targeted (6c-2b — touch ALL columns of the selected rows):**
  - `hdu[i] = record` — single-row write.  RHS is a `numpy.void`
    scalar or a shape-`(1,)` structured ndarray with the
    table's field names.  Negative `i` wraps; out-of-range
    raises `IndexError`.
  - `hdu[a:b] = arr` — step=1 slice write.  RHS is a
    structured ndarray of length equal to the slicelength.
    Empty slice + empty RHS is a no-op (PCOUNT unchanged).
  - `hdu[[i, j, k]] = arr` — fancy-row write.  RHS is a
    structured ndarray of length = `len(row_list)`.  Duplicates
    in the row list follow numpy fancy-assignment semantics
    (last write wins in the input list).  Negative indices
    wrap.

**Column-targeted (6c-2c — narrow the column selection):**
  - `hdu["col"] = arr` — whole-column write.  For a fixed
    column: RHS is an ndarray of shape `(nrows,) + per_cell_shape`
    matching the column's dtype.  For a VLA column: RHS is an
    Object-dtype ndarray of length `nrows` with per-row inner
    ndarrays.  Touches all tiles; other columns' descriptors
    stay unchanged.
  - `hdu[[c1, c2]] = arr` — multi-column subset write.  RHS is
    a structured ndarray of length = nrows with the named
    fields (extras tolerated; missing rejected; duplicates in
    the name list rejected).  Mixed fixed + VLA subsets are
    supported — each named column dispatches to its own per-
    column encoder per tile.  Unnamed columns untouched.  Names
    are case-insensitive against the table columns.

**Stepped / subset-object (6c-2d — generalize row + col selection):**
  - `hdu[a:b:s] = arr` — stepped slice (positive step only).
    Negative or zero step rejected for parity with the
    uncompressed-side `TableHDU[slice] = value`.  Disk rows
    enumerate `(start, start+step, ..., start+(N-1)*step)`;
    bucketed by tile by the shared primitive.
  - `hdu["name"][rows] = value` — single-column subset write.
    `rows` accepts int (bare-int shortcut, value is scalar /
    0-d / per-cell ndarray or — for VLA — the cell value
    directly), slice (any positive step), or iterable of ints
    (negative indices wrap).  Value is an ndarray of shape
    `(len(rows),) + per_cell_shape` for non-int row keys on
    fixed columns; for VLA columns it's an Object ndarray of
    length `len(rows)`.
  - `hdu[[c1, c2]][rows] = value` — multi-column subset write.
    `rows` shapes as above; value is a structured ndarray of
    length `len(rows)` with the subset's field names (extras
    tolerated).  For bare-int `rows`, value is a structured
    record / shape-`(1,)` ndarray.  Mixed fixed + VLA subsets
    supported.

**VLA columns (6c-2e — fold VLA into every form above):**

For a (tile, vla_col) edit, the per-tile helper
`setitem_vla_column_tile`:

  1. Reads + GZIP-decompresses the existing dual-descriptor
     blob (size = `rows_in_tile × (width_orig + 16)`).
  2. For each edited row in the tile: validates the input cell
     against the column's inner type, serializes to BE bytes,
     encodes per ZCTYPn with cfitsio's uncompressed-fallback
     (raw bytes when compressed isn't smaller), appends the
     payload to the heap end, and updates the in-RAM blob's
     compressed-Q descriptor with `(new_cvlalen, new_cvlastart)`.
     The original-side descriptor gets a fresh
     `original_offset = current ZPCOUNT`; ZPCOUNT bumps by the
     new cell's uncompressed-byte size.  Old per-cell bytes
     and old original-heap slots both become orphans (compressed
     orphans live in the heap until `repack()`; original-heap
     orphans are conceptual — funpack's reconstructed view never
     references them).
  3. Re-GZIPs the (modified) blob and appends to heap end.
  4. Updates the main-table descriptor entry in the in-RAM
     descriptor table.

**Note on nelements changes.**  Because each edited cell gets a
fresh `original_offset = current ZPCOUNT`, the cell can change
length arbitrarily without overlapping other cells in funpack's
reconstructed heap.  ZPCOUNT grows monotonically per edit.

**Shared primitive: `setitem_compressed_cols`.**  (Renamed from
`setitem_compressed_fixed_rows` when 6c-2e generalized it.)
Takes `disk_rows: &[usize]` (input row K → `disk_rows[K]`),
`selected_col_indices: &[usize]`, and `per_column_inputs:
&[Bound<'_, PyAny>]` (one per selected column — either a
shape-`(N,) + per_cell_shape` ndarray for fixed cols or a
shape-`(N,)` Object-dtype ndarray for VLA cols).  The
dispatcher dispatch table:

| Form | `selected_col_indices` | `disk_rows` |
|------|------------------------|-------------|
| `hdu[i]=record` (6c-2b) | all columns | `[i]` |
| `hdu[a:b]=arr` (6c-2b) | all columns | `a..b` |
| `hdu[[i,j,k]]=arr` (6c-2b) | all columns | row list |
| `hdu[a:b:s]=arr` (6c-2d) | all columns | stepped range |
| `hdu["col"]=arr` (6c-2c) | `[col_idx]` | `0..nrows` |
| `hdu[[c1,c2]]=arr` (6c-2c) | `[c1_idx, c2_idx]` | `0..nrows` |
| `hdu["col"][rows]=v` (6c-2d) | `[col_idx]` | rows |
| `hdu[[c1,c2]][rows]=v` (6c-2d) | `[c1_idx, c2_idx]` | rows |

Algorithm.  Per affected tile (sorted by tile index for disk
locality), per selected column, dispatched on `var_kind`:

  - **Fixed column path**: decode the existing tile blob to a
    BE bytes slab via `decode_existing_tile_to_be_bytes`,
    overwrite the affected rows' per-cell bytes via
    `apply_transform_cell` (byteswap / unsigned-int trick / bool
    / string), and re-encode + append + record the new blob via
    `encode_be_slab_to_heap_and_record`.
  - **VLA column path**: `setitem_vla_column_tile` (see above).

After all tiles: rewrite the descriptor table region at
`data_offset`, update PCOUNT through the standard
disk-write-before-commit + taint discipline, and update ZPCOUNT
(only when any VLA column was touched).  Cache is cleared at
the end (modified tiles' entries are stale).

**Reused primitives** (already `pub(crate)` from earlier
phases):
- Key dispatch + per-row coercion: `classify_setitem_key`,
  `SetItemKey`, `coerce_to_len1_record` (all promoted to
  `pub(crate)` in setitem.rs for 6c-2b).
- Per-column input unpack: `extract_per_column_inputs`.
- Fixed-column validation: `prepare_fixed_column` + `ColPrep`.
  Validate-then-mutate: dtype/shape errors raise BEFORE any
  file I/O.
- Fixed-column decode/encode: `decode_existing_tile_to_be_bytes`,
  `apply_transform_cell`, `encode_be_slab_to_heap_and_record`.
- VLA cell validation + serialization: `validate_vla_cell`,
  `serialize_vla_cell`, `extract_string_vla_cell_bytes`,
  `write_descriptor`.
- VLA encoder: `encode_table_column_slab`, `encode_gzip1`,
  `gzip_decompress_bytes` for the dual-descriptor blob.
- Single-cell coercion helpers: `coerce_cell_value_to_len1`
  (fixed; uses `column_expected_shape` + `field_dtype_and_shape`),
  `coerce_vla_cell_value_to_len1` (VLA; wraps the cell value
  in a length-1 Object ndarray).

**Dispatcher refactor.**  The dispatcher branches share 12
stable arguments (super_, cards, columns, algorithms, configs,
nrows, ztilelen, n_tiles, descriptor_row_width, data_offset,
current_pcount, cache).  Bundled into a `SetItemCtx` struct so
each branch's call to the primitive is `(py, &ctx, per_column,
selected_cols, disk_rows)` — four args, isolating the
per-branch variation.  Three small input-validation helpers:
`require_ndarray`, `require_ndarray_with_length`,
`resolve_structured_subset_value` (multi-col name resolution +
per-column ascontiguousarray materialization).

**Prelude helper.**  `read_compressed_table_meta(super_) →
CompressedTableMeta` packages the cards / columns / algorithms
/ nrows / ztilelen / n_tiles / descriptor_row_width /
current_pcount / data_offset parse done at the top of
`__setitem__`.  The subset pyclasses
(`CompressedSingleColumnSubset`, `CompressedColumnSubset`)
call the same helper so the dispatch prelude lives in one
place.  Each subset method then builds its own `SetItemCtx`
from the meta + Arc-cloned cache + cfgs.

**Rows-key resolver.**  `resolve_compressed_rows_key` mirrors
uncompressed-side `resolve_rows_key` (int / slice / iterable →
`(Vec<usize>, was_single)`) but with "CompressedTableHDU"
error messages and direct dependency on `normalize_disk_row`.
Returns `was_single = true` only for a bare int key, so the
subset dispatch can route through `coerce_cell_value_to_len1`
(fixed) or `coerce_vla_cell_value_to_len1` (VLA) or
`coerce_to_len1_record` (multi-col) instead of requiring an
ndarray RHS.

**Memory bound.**  Per affected (tile, col): for fixed columns,
one BE-bytes slab (`rows_in_tile × per_row_bytes`).  For VLA
columns, one decompressed dual-descriptor blob (`rows_in_tile
× (width_orig + 16)` bytes) plus one per-edited-cell BE-bytes
buffer.  Encoded and dropped before the next column.  Plus the
full descriptor table in RAM (`n_tiles × ncols × 16` bytes;
small).  Heap writes go to the file as they're produced.

**Implementation.**  `setitem_compressed_cols` +
`setitem_vla_column_tile` + `SetItemCtx` + dispatcher helpers
(`read_compressed_table_meta`, `find_compressed_column_index`,
`normalize_disk_row`, `resolve_compressed_rows_key`,
`require_ndarray*`, `resolve_structured_subset_value`,
`coerce_cell_value_to_len1`, `coerce_vla_cell_value_to_len1`)
at the bottom of `hdu_table_compressed.rs`; dispatcher in the
`__setitem__` pymethod plus the two subset pyclasses'
`__setitem__` methods all use the same shape.

Tests across four files:
- `tests/test_table_compressed_setitem_rows.py` (27 cases,
  6c-2b).
- `tests/test_table_compressed_setitem_cols.py` (34 cases,
  6c-2c).
- `tests/test_table_compressed_setitem_stepped_subsets.py`
  (25 cases, 6c-2d).
- `tests/test_table_compressed_setitem_vla.py` (25 cases,
  6c-2e).

Combined coverage: row-form (single-row / slice-step=1 /
fancy-row / stepped slice × within-tile / across-tiles /
negative indices / duplicates last-wins / boundary),
column-form (whole-col / cell / multi-col × case-insensitive
lookup / extra fields tolerated / missing fields rejected /
duplicate names rejected / subarray TDIM column / scalar
broadcast over subarray), subset-object form (single-col +
multi-col subset × int-row / slice / stepped / fancy /
full-slice / negative int / out-of-range), VLA cells
(same-length / grows / shrinks-to-empty / ZPCOUNT accounting
/ wrong-inner-dtype rejected / out-of-range), VLA whole-column
+ row-form + subset-form + mixed fixed+VLA multi-col, PA
(string VLA) cell write, algorithm matrix (GZIP_1 / GZIP_2 /
RICE_1), cache invalidation, `repack()` reclaims orphans (both
fixed and VLA), non-last HDU preserves trailing, **funpack
byte-exact interop** on mutated files combining all forms, and
same-handle vs reopen parity.

