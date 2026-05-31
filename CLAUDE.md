# rustfits — design notes for contributors

A Rust+PyO3 implementation of a FITS reader/writer for Python.  This file
captures architectural decisions and conventions that aren't obvious from
reading the code.  Claude Code loads this automatically into every session
in this directory; humans should read it before making structural changes.

For the project-level distribution plan (how rustfits relates to fitsio,
the freeze-and-shim handoff strategy, the migration doc location), see
[STRATEGY.md](STRATEGY.md) at the repo root.  This file (`CLAUDE.md`) is
for code architecture; `STRATEGY.md` is for project management.

## Project structure

The Rust extension is split into single-responsibility modules.  Each only
exposes (`pub(crate)`) what neighboring modules actually import; everything
else stays private to its file.

- `src/lib.rs` — `#[pymodule]` init + `mod` declarations.  Nothing else.
- `src/cache.rs` — `BytesBoundLruCache<K>`, a generic bytes-budgeted
  LRU used by `CompressedImageHDU` (key = `u64` flat tile index) and
  `CompressedTableHDU` (key = `(u32, u32)` tile + column).  Same
  Mutex<Inner> + AtomicU64 capacity, same `get` / `put` /
  `set_capacity` / `clear` / `used_bytes` / `capacity` surface; values
  are `Arc<Vec<u8>>` so callers clone the Arc out under the lock and
  decode work runs lock-free.
- `src/common.rs` — `FileHandle`, `TaintFlag`, `HduOffsets`, `FileLayout`,
  `lock_file`, `check_not_tainted`, `shift_file_tail_and_update_offsets`,
  `zero_fill_range`, `BLOCK_SIZE`/`CARD_SIZE`/`CARDS_PER_BLOCK`,
  `parse_keyword`, `parse_string_keyword`, `byteswap_in_place`,
  `RawBuffer`.  File primitives plus the shared byte-shift helper that
  the header-grow path (and the future image/table data-grow paths) all
  call into, plus the Python-buffer-protocol wrapper used by both image
  and table read/write to move bytes between disk and numpy storage.
- `src/header.rs` — `FITSHeader` and `FITSHeaderEdit` pyclasses plus every
  card-level helper (parsing, building, CONTINUE chains, HIERARCH,
  commentary, protected keys, batched update, `rewrite_header_to_disk`
  including the in-place file-grow branch).  Exports just the pyclasses,
  `FITSHeader::from_state`, `py_is_protected_key`, and the card-builder
  formatters (`pad_to_card`, `card_int`, `card_logical`, `card_string`)
  used by image-HDU creation.
- `src/hdu.rs` — base `HDU` pyclass.  Holds `Arc<HduOffsets>` (its own
  shared-mutable offsets) and `Arc<FileLayout>` (for cross-HDU offset
  updates during a grow); fields are `pub(crate)` because subclass
  `#[pymethods]` access them via `into_super()`.
- `src/hdu_image.rs` — `ImageHDU` pyclass + image read/write/slicing,
  bitpix conversions, shape parsing.  Only exports `ImageHDU` (+ `new`)
  and `dtype_to_bitpix` (used by `FITS::create_image_hdu`).
- `src/hdu_image_compressed/` — `CompressedImageHDU` (`ZIMAGE`
  convention) pyclass, split (2026-05) into single-responsibility files
  behind a `mod.rs` that re-exports only the external surface
  (`CompressedImageHDU`, `header_has_zimage`, `compute_n_tiles`).
  Subclasses `ImageHDU` (so `isinstance(hdu, ImageHDU)` holds on a
  tile-compressed HDU).  Submodules:
  - `hdu.rs` — the pyclass + `#[pymethods]` dispatch, ZIMAGE detection
    (`header_has_zimage`), the `TileCache` alias, and the per-HDU
    meta-cache accessor.
  - `meta.rs` — `CompressedImageMeta` cache struct + parser, shape /
    tile parsing, data-column discovery, quant-context build, TFORM
    helpers, compression-config rebuild.
  - `read.rs` — whole-image read, per-tile decode + cache, slice
    walking, descriptor/heap byte readers.
  - `write.rs` — per-tile encode (int / float / quantized), bulk
    write, `extend`, `__setitem__`, descriptor-buffer helpers.
  - `repack.rs` — heap repack (drop orphans + shrink file).
  - `checksum.rs` — ZHECKSUM/ZDATASUM over the equivalent
    uncompressed-image bytes.
  Imports the shared `crate::zimage::tile_io::DEFAULT_TILE_CACHE_BYTES`.
  Full feature status + phase history is in the "Tile-compressed images
  (ZIMAGE)" roadmap below.
- `src/hdu_table/` — `TableHDU` (BINTABLE) pyclass split across eight
  single-responsibility files (was a 6400-line `hdu_table.rs` until
  the 2026-05 refactor).  `mod.rs` just wires the submodules and
  re-exports the external surface (`TableHDU`, `SingleColumnSubset`,
  `ColumnSubset`, plus the two helpers `normalize_and_build_table_header`
  and `set_pcount_in_cards` that other modules import directly).
  Submodules:
  - `columns.rs` — `Column` struct, TFORM/TDIM/THEAP/PCOUNT parsing,
    scaling-kind classifier, element-width helpers (`bytes_per_element`,
    `byteswap_unit`).  The on-disk-schema reader; everything downstream
    operates on a typed `Vec<Column>`.
  - `read.rs` — `read_table` + `read_one_column` + the run planner
    (`resolve_columns`, `resolve_rows`, `plan_runs`, `process_runs`) +
    heap-pass for VLA reads + all read-side conversion helpers
    (unsigned-trick, general scaling, X bit unpack, A string decode,
    TNULL mask construction).
  - `write_setup.rs` — `WriteColumn`, dtype-to-FITS classifiers
    (`classify_scalar_numpy_field`, `classify_var_numpy_field`),
    `dtype_to_write_columns`, header-card emission
    (`build_bintable_header_cards`, `normalize_and_build_table_header`),
    `WriteTransform` + `column_transform`, `column_expected_shape`,
    `determine_input_nrows`, `dispatch_write_input`.  Everything
    write-side that's shared between fixed and VLA paths.
  - `write_fixed.rs` — `ColumnSource`, `prepare_structured_input` /
    `prepare_dict_input` / `prepare_list_names_input`,
    `acquire_per_column_array`, transform appliers
    (`apply_transform_cell`, `apply_in_place_transform`), the three
    bulk writers (`write_table_data` / `_strided` / `_one_column`),
    the three fixed setitem helpers (`setitem_single_row` / `_slice`
    / `_column`), `write_fixed_only` + `append_fixed_only`
    dispatchers, plus utilities (`normalize_row_index`,
    `find_column_by_name`, `coerce_to_len1_record`, `build_sources`,
    `set_pcount_in_cards`).
  - `write_vla.rs` — `any_var_column`, `extract_per_column_inputs`,
    cell validators (`validate_vla_cell`, `extract_string_vla_cell_bytes`,
    `vla_cell_expected_dtype`), heap planning (`plan_vla_heap_layout`,
    `VlaCellPlan`), descriptor I/O (`write_descriptor`,
    `serialize_vla_cell`), `FixedColInfo` / `VlaColInfo` strip
    builders (`build_fixed_col_info`, `fill_main_row`,
    `write_vla_data_range`), `write_vla_aware` + `append_vla_aware`
    dispatchers, and `repack_table_heap`.
  - `setitem.rs` — top-level `__setitem__` machinery: `SetItemKey`
    enum + `classify_setitem_key`, the cross-cutting cases
    (`setitem_fancy_rows`, `setitem_multi_columns`, `setitem_cell` +
    `setitem_cell_vla`), the subset-object helpers
    (`resolve_rows_key`, `write_one_column_at_rows`,
    `write_column_subset_at_rows`), and the VLA-aware row/slice/
    column dispatchers (`setitem_rows_vla_aware_inner`,
    `setitem_single_row_vla_aware`, `setitem_row_slice_vla_aware`,
    `setitem_single_column_vla`).
  - `edit.rs` — `insert_column` + `delete_column` schema-edit machinery:
    header card mutation (renumbering per-column TTYPEn/TFORMn/TDIMn/
    TUNITn/TZEROn/TSCALn/TNULLn/TDISPn/TBCOLn for shifted columns,
    inserting/dropping the target column's cards, updating TFIELDS +
    NAXIS1), strip-based row shuffler (`shuffle_main_for_insert`
    back-to-front for growing rows, `shuffle_main_for_delete`
    front-to-back for shrinking rows; both bounded at ~1 MiB per
    buffer), in-file heap relocation (`relocate_region_forward` /
    `_backward` for grow / shrink), data-extent grow/shrink dispatcher.
  - `hdu.rs` — `TableHDU` pyclass + `#[pymethods]` impl blocks (read,
    write, append, repack, insert_column, delete_column, checksum,
    `__getitem__`, `__setitem__`, accessors, repr), `TableKey` +
    `classify_table_key` + `try_extract_column_name` for `__getitem__`
    dispatch, and the `SingleColumnSubset` / `ColumnSubset` pyclasses
    returned by the one-column / multi-column read shortcuts.
- `src/hdu_table_compressed/` — `CompressedTableHDU` (`ZTABLE`
  convention) pyclass, split (2026-05) into single-responsibility
  files behind a `mod.rs` that re-exports only the external surface
  (`CompressedTableHDU`, `CompressedSingleColumnSubset`,
  `CompressedColumnSubset`, `header_has_ztable`,
  `build_compressed_table_header`, `default_ztilelen`,
  `resolve_compress_arg`).  Subclasses `TableHDU` (so
  `isinstance(hdu, TableHDU)` holds on a compressed-table HDU).
  Submodules:
  - `hdu.rs` — the pyclass + `#[pymethods]` dispatch (read /
    `__getitem__` / write / append / repack / `__setitem__` /
    checksum / accessors / repr), ZTABLE detection
    (`header_has_ztable`), the `CacheKey` + `ColumnTileCache`
    types, original-schema synthesis (`synthesize_uncompressed_cards`),
    and the repr helpers.
  - `meta.rs` — `CompressedTableMeta` cache struct + parser.
  - `read.rs` — whole-table + `rows=` read, the per-tile `RowPlan`
    planner, column-slab + VLA-cell decompression, the gzip-blob
    helper.
  - `subset.rs` — the `CompressedSingleColumnSubset` /
    `CompressedColumnSubset` pyclasses.
  - `write_setup.rs` — algorithm/config resolution + per-dtype
    defaults, per-column prep (`ColPrep`), tile-slab encode helpers,
    `build_compressed_table_header`.
  - `write.rs` — `write_compressed_table_data` + the file-grow /
    ZPCOUNT / VLA-tile encode helpers.
  - `append.rs` — merge-into-partial-last-tile (fixed + VLA), heap
    relocation, existing-tile decode.
  - `repack.rs` — fixed + VLA dual-descriptor heap repack + the
    chunked in-file copy primitive.
  - `setitem.rs` — the shared per-tile column writer + VLA tile
    writer + `SetItemCtx` + the row/value coercion + resolution
    helpers.
  - `checksum.rs` — ZHECKSUM/ZDATASUM over the equivalent
    uncompressed-table bytes (streaming per-tile, incl. the VLA
    synthetic heap).
  Accessors `nrows`, `dtype`, `colnames`, `units`, `__len__`
  override TableHDU's so they parse the ORIGINAL-schema view via
  `synthesize_uncompressed_cards` (NAXIS1←ZNAXIS1, NAXIS2←ZNAXIS2,
  PCOUNT←ZPCOUNT, TFORMn←ZFORMn, dropping the Z-prefixed cards).
  Compression-specific accessors: `compression` returns
  `{col_name: ZCTYPn_value}`, `n_tiles` is the on-disk NAXIS2,
  `ztile_rows` is ZTILELEN.  Routing in `fits.rs::parse_hdus_from_file`
  checks ZTABLE before ZIMAGE (defensive ordering).  Imports shared
  helpers from `crate::hdu_table` (`parse_columns`,
  `build_numpy_dtype`, `field_dtype_and_shape`, the descriptor
  codecs, ...) and `crate::zimage::tile_io::DEFAULT_TILE_CACHE_BYTES`.
  Full feature status + phase history is in the "Tile-compressed
  tables (ZTABLE)" roadmap below.
- `src/hdu_ascii_table.rs` — `AsciiTableHDU` (TABLE) pyclass stub
  (read returns header only; no column read/write yet).
- `src/fits.rs` — `FITS` pyclass + `parse_hdus_from_file` +
  `validate_header` + `calculate_data_size`.  Owns the per-file
  `Arc<FileLayout>` and pushes a new `Arc<HduOffsets>` into it for every
  HDU that gets parsed or created.  All free functions are private to
  this file (only `FITS` is exported).
- `rustfits/` — the Python package; `_rust.so` is built into it by maturin.
- `tests/` — pytest suite; tests pair same-handle and post-reopen
  assertions for mutations.
- Build: `maturin develop` (compiles the cdylib AND installs the editable
  wheel in one step — don't use bare `cargo build`).

Visibility discipline: keep every helper `fn foo()` (private to its file).
Promote to `pub(crate)` only when the compiler complains — the smaller
module surface area is the whole point of the split.  Inside the
`hdu_table/` directory module, sibling files reach each other via
`use super::write_fixed::*` (etc.) — those cross-file calls do need
`pub(crate)`, but external modules (`fits.rs`, `lib.rs`,
`hdu_image_compressed/`) only see what `hdu_table/mod.rs` explicitly
re-exports.  The two compressed-HDU directory modules
(`hdu_image_compressed/`, `hdu_table_compressed/`) follow the same
rule internally.

## Axis order: numpy throughout, FITS only at the boundary

FITS stores arrays with the fastest-varying axis (NAXIS1) first; numpy uses
row-major order with the fastest-varying axis last.  `parse_image_hdu_shape`
(in `hdu_image.rs`) reverses once at the boundary; everything downstream
(`row_major_strides`, `compute_strip_layout`, `read_image_slice`,
`write_image_data`, `__getitem__` slicing) operates in numpy order.  Don't
reverse again anywhere else.

## FITSHeader: cards are the single source of truth

The `FITSHeader` pyclass owns one piece of state: `cards: Arc<Mutex<Vec<String>>>`.
All accessors (`__getitem__`, `__contains__`, `keys`, `to_dict`, ...)
re-parse the cards from scratch on every call.  **Do not add a cache** —
no parsed-dict alongside cards, no keyword index, no value table.

Why: FITS headers are tiny (tens to a few hundred cards), so parse-on-demand
cost is invisible in profiling.  Mutation (`__setitem__`, `__delitem__`,
`update`, `add_comment`, etc., plus `FITSHeaderEdit` batching) only has to
rewrite cards — no cache to keep in sync.  The vast majority of bugs in
classes shaped like this come from sync drift.

If iteration ever shows up as hot in profiling, the transparent optimization
is a cached `Vec<String>` of unique keywords (not values), invalidated on
any card mutation.  Don't do this preemptively.

Related conventions:
- `header[key]` returns the value directly (astropy-style), not a
  `{value, comment}` dict.  Per-key comments come from `header.comment_of(key)`.
  The legacy `header_dict` shape is still available via `header.to_dict()`
  for serialization or tests.
- Keyword normalization: `normalize_keyword()` returns the lookup form
  (trimmed + uppercased; HIERARCH long keys additionally have internal
  whitespace collapsed to single spaces).  All in-module matching —
  `find_card_for_key`, `set_card_for_key`, `delete_card_for_key`,
  `unique_keys_in_order` — goes through this form, so user-facing lookup
  is case-insensitive for both standard and HIERARCH keys.  Storage
  spelling is decided separately by `storage_keyword()` — see the
  HIERARCH section below.

## Header mutation order: disk write before in-memory commit

Every method that mutates the header (`FITSHeader.__setitem__`,
`__delitem__`, `update`, `add_comment`, `add_history`, `add_blank`, and
`FITSHeaderEdit.commit_internal`) must follow this order:

1. Lock the cards mutex.
2. Build a candidate `new_cards: Vec<String>` (clone the locked Vec, apply
   changes to the clone).
3. Call `rewrite_header_to_disk(..., &new_cards)` — this is the failure-
   prone step (slack-overflow check, I/O errors).
4. **Only on success**, commit by `*guard = new_cards`.
5. Release the lock.

The shared helpers (`apply_setitem`, `set_card_for_key`, `delete_card_for_key`,
`append_commentary_to_cards`) operate on `&mut Vec<String>` so they compose
naturally — pass the working clone to them, write to disk, then commit.

Why: without this order, a pre-I/O failure (most commonly the slack-overflow
check) would leave the in-memory `cards` Vec already mutated but the on-disk
header unchanged.  The next `header[k]` read would see the staged value, the
file wouldn't, and retrying would re-fire the same error.  Reversing the
order makes any pre-I/O failure leave both states untouched and consistent.

**Caveat (not yet fixed):** mid-write I/O failures inside
`rewrite_header_to_disk` or `shift_file_tail_and_update_offsets`
(e.g. ENOSPC partway through `write_all`/`flush`, or during the chunked
back-to-front copy of the file tail) can still leave the on-disk file
partially overwritten.  The user gets an IOError naming the failure mode,
the per-file taint flag is set, and the user is expected to close + reopen
to recover.  An atomic temp-file-and-rename rewrite would be safer; it's
not implemented because the in-place path matches the model of every
other write in the codebase, and we have not yet seen mid-write failures
in practice.

## Protected keywords

Some keywords represent state rustfits manages on the user's behalf —
file structure, integrity contracts, or compression layout — and must
not be mutated through `header[k] = v` or `del header[k]`.  The predicate
`is_protected_key()` (header.rs) covers:

- **Image HDU structural:** `SIMPLE`, `XTENSION`, `EXTEND`, `BITPIX`,
  `NAXIS`, `NAXIS1..NAXIS999`, `PCOUNT`, `GCOUNT`, `END`.
- **Binary table structural:** `TFIELDS`, `TFORMn`, `TDIMn`, `TTYPEn`,
  `TSCALn`, `TZEROn`, `TNULLn`, `THEAP`.
- **ASCII table structural:** `TBCOLn` (plus the shared bintable keys).
- **Random groups:** `GROUPS`, `PTYPEn`, `PSCALn`, `PZEROn` (plus PCOUNT/GCOUNT).
- **Tiled image compression:** `ZIMAGE`, `ZCMPTYPE`, `ZBITPIX`, `ZNAXIS`,
  `ZNAXISn`, `ZTILEn`, `ZNAMEn`, `ZVALn`, `ZSIMPLE`, `ZEXTEND`, `ZBLOCKED`,
  `ZPCOUNT`, `ZGCOUNT`, `ZHECKSUM`, `ZDATASUM`, `ZTENSION`, `ZQUANTIZ`,
  `ZDITHER0`, `ZMASKCMP`, `ZBLANK`.
- **Integrity:** `CHECKSUM`, `DATASUM`.

`__setitem__` and `__delitem__` (both on `FITSHeader` and `FITSHeaderEdit`)
raise `ValueError` for any protected key.  `to_dict(skip_protected=True)`
returns a filtered copy suitable for serialization or as a base for
hand-copied updates.  `is_protected_key()` is also exposed at the Python
module level for callers writing their own filter loops.

Internal paths that legitimately update structural keys (e.g.
`extend_image_hdu` updating `NAXISn`) operate on the cards `Vec` directly,
not through `__setitem__`, so the guard doesn't break them.

`update()` policy on protected keys is split by source type, both
enforced in `collect_update_actions`:

- **FITSHeader source:** silently skips protected keys.  The use case is
  "copy the metadata from this other HDU's header" — the destination
  already has its own correct structural/integrity/compression keys and
  they must not be clobbered.  The filter is in the FITSHeader branch,
  applied right after `keyword_of` extracts the keyword from each card.
- **Dict (mapping) source:** raises `ValueError` on any protected key.
  An explicit hand-written `{"BITPIX": 32}` in user code is almost
  certainly a mistake, not an intent to be silently dropped.  The whole
  update is rejected wholesale (no partial commit).

## update() and commentary cards

Commentary keys (COMMENT, HISTORY, blank) are handled in parallel to
protected keys but with an opt-in carry-over via the keyword-only
`copy_commentary: bool = False` arg on `FITSHeader.update()` and
`FITSHeaderEdit.update()`:

- **FITSHeader source, `copy_commentary=False` (default):** silently skip
  commentary cards.  Matches the protected-key skip rule — the common
  case is "copy structured metadata from this other header" and forcing
  the caller to write try/except just to make that work would be hostile.
- **FITSHeader source, `copy_commentary=True`:** append each commentary
  card verbatim, one append per source card (a long commentary that the
  source split across N cards stays split as N cards in the destination).
  No deduplication — see "design notes" below.
- **Dict source:** always raises on `COMMENT`/`HISTORY`/`""` (blank),
  regardless of `copy_commentary`.  The flag is meaningful only for
  header-to-header copy; an explicit `"COMMENT"` in a hand-written dict
  is ambiguous and almost certainly a mistake.

Design notes:

- **No dedup against the destination.** Tempting but ultimately broken:
  multi-card commentary would be partially matched (skipping card 1 while
  appending cards 2+3 leaves a corrupted entry), and HISTORY collisions
  often refer to different things (e.g. two "Bias subtracted" lines about
  different bias frames).  HISTORY is also an append-only audit trail per
  the FITS standard, so duplicate-on-rerun is arguably the right behavior.
  The user opts in once per source for the metadata-copy workflow.
- **Mechanically:** `collect_update_actions` returns a `Vec<UpdateAction>`
  where each entry is `SetKey { key, value, explicit_comment }` or
  `AppendCommentary { keyword, text }`.  Both `FITSHeader::update` and
  `FITSHeaderEdit::update` walk the list and dispatch via match — set
  actions go through `apply_setitem`, commentary actions through
  `append_commentary_to_cards`.  Same disk-write-before-commit ordering
  as every other mutation path.

**Forward-looking — cross-HDU-type copy.**  The image-only metadata
keys `BUNIT`, `BSCALE`, `BZERO` are *not* protected — they're fine to
set on an image HDU — but they're meaningless on a table HDU.  When
`update()` learns to copy a header across HDU types, the destination's
HDU type needs to be consulted: if the target is a table, these three
keys should be stripped from the source.  Same direction in reverse
(table → image) would strip TUNIT/TDISP/etc.

## CONTINUE-chained string values

Long string values (escaped length > 68 chars) auto-emit a CONTINUE chain:
`KEYWORD = 'first&'` + N `CONTINUE  'chunk&'` cards + final
`CONTINUE  'last' [/ comment]`.  Chunker (in `build_string_value_cards`)
never splits a `''` quote-escape pair across cards.  Comments live on the
last card only; if a comment is too long to fit there, the write raises
(no silent truncation).

When mutating a key whose existing value is a CONTINUE chain, the *entire*
chain is removed before the new cards are inserted.  This is what
`find_chain_end` + `card_value_ends_with_amp` are for; both
`set_card_for_key` and `delete_card_for_key` use them.

## HIERARCH long keys

Keys >8 chars or containing spaces auto-route through `build_hierarch_cards`,
which emits `HIERARCH <key> = <value> [/ comment]`.  Validation is relaxed
to allow space, `.`, `+` in addition to A-Z/a-z/0-9/`-`/`_`.  The literal
"HIERARCH" is rejected as a user key (case-insensitively — it's the
convention prefix).

**Case preservation and case-insensitive lookup (ESO convention).** Standard
8-char keys are uppercased on disk (FITS standard requires it).  HIERARCH
long keys preserve the caller's case on disk; lookup, comparison, dedup,
and `__contains__`/`__delitem__` are case-insensitive across the whole
header.  Two helpers split the concern:

- `normalize_keyword(key)` → lookup form: trimmed, uppercased, and (for
  HIERARCH) with internal whitespace collapsed to single spaces.  Used
  everywhere comparisons happen.
- `storage_keyword(key)` → on-disk form: trimmed + whitespace-collapsed
  for HIERARCH (case preserved); uppercased for standard keys.  Used by
  `apply_setitem` when inserting a new card.

When `apply_setitem` is called for an existing key, `existing_storage_keyword`
returns the on-disk card's current spelling, which then wins over the
caller's spelling for the rebuilt card.  Consequence: writing the same
HIERARCH key with different case (`h["Eso Ins Det1"]` then
`h["ESO INS DET1"]`) updates the value but leaves the keyword text as
first written.  This matches astropy and extends the existing "in-place
update preserves position" rule to also preserve spelling.

Whitespace canonicalization runs at write time: `"Eso  Ins   Det1"` (extra
spaces) lands on disk as `HIERARCH Eso Ins Det1`, conforming to the ESO
single-space-separator convention.  Non-space whitespace (tab, etc.) is
rejected by `validate_keyword` rather than silently rewritten — HIERARCH
keywords are space-separated by spec.

Long HIERARCH string values auto-chain via `build_hierarch_string_cards`:
the first card carries the HIERARCH prefix and a `&'`-terminated chunk
(payload budget `65 - len(key)` bytes — measured against the storage form
including case-preserved characters), followed by N standard CONTINUE
cards with the comment on the last one.  HIERARCH keys ≥ 65 chars cannot
chain (no room for any first-card payload) and are rejected.  The
read-side already follows CONTINUE chains regardless of the first card's
key shape, so no reader changes are needed.

## Tainted-header state on mid-write I/O failure

A per-file `Arc<AtomicBool>` is created in `FITS::new`, cloned into every
HDU (and from there into every `FITSHeader` view).  When `rewrite_header_to_disk`'s
`write_all` or `flush` fails, the file is partially overwritten and the
in-memory cards are still pre-mutation — they have diverged.  Setting the
taint flag (`tainted.store(true, Ordering::Release)`) makes every
subsequent read or write on any view of the file refuse with a diagnostic
`IOError` that names the inconsistency and tells the user to close +
reopen.

Where the check fires (`check_not_tainted` at the top of each):
- `HDU::header_snapshot` — picks up image reads/writes that go through it.
- `FITSHeader::snapshot` — picks up most header reads (`__getitem__`,
  `__contains__`, `__iter__`, `keys`, `to_dict`, etc.).
- `FITSHeader::__setitem__`, `__delitem__`, `update`, `append_commentary`,
  and `FITSHeaderEdit::commit_internal` — the mutation entry points.

Pre-I/O failures (slack-overflow check, file-lock, closed-file, initial
`seek`) leave the file untouched and MUST NOT taint.  The split is enforced
inside `rewrite_header_to_disk`: only `write_all` and `flush` errors set
the flag.  Recovery is via close+reopen; the flag is per-handle, not
per-file-on-disk.

Testing hook: `HDU._force_taint()` flips the flag directly.  Underscored
to signal "test plumbing, not API."  Used by `tests/test_header_taint.py`
to verify rejection semantics without needing to produce a real I/O
failure on the host filesystem.

## Durability contract: close does NOT fsync

`FITS.close()` drops the file handle without calling `fsync(2)`.  Data
is left in the OS page cache, which persists across normal program
exit (process death by `SIGSEGV`/`SIGKILL`/uncaught exception/etc.
all keep the cache intact — only power loss or kernel panic loses
unflushed data).  This matches fitsio and astropy.

For callers who need power-loss durability, `FITS.sync()` is the
explicit opt-in: it calls `fsync` on the underlying file.  Cheap to
call repeatedly when there are no new dirty pages; expensive when
there are (it blocks until the storage device confirms).

Prior to 2026-05-29 `close()` unconditionally fsynced.  Removed
because: (1) ecosystem parity (no other Python FITS library does it);
(2) on a single-write-then-exit pattern the user paid ~200 ms for a
512 MB fsync they didn't ask for; (3) doesn't actually improve
steady-state throughput in a write loop — the kernel ends up
flushing either way, fsync just changes when.

## Header overflow: in-place file grow

When a header mutation would push past the currently reserved header
blocks, `rewrite_header_to_disk` falls into the grow branch instead of
raising.  The mechanics:

1. Compute `delta_blocks = ceil(cards.len() / CARDS_PER_BLOCK) − current`
   and `delta_bytes = delta_blocks * BLOCK_SIZE`.
2. Call `shift_file_tail_and_update_offsets(file, layout, after_offset =
   self.data_offset, delta_bytes, taint)` — this extends the file with
   `set_len(original_len + delta)`, copies every byte from `after_offset
   .. original_len` forward by `delta` (back-to-front, in 1 MiB chunks),
   flushes, and then atomically bumps every `HduOffsets` whose
   `header_offset >= after_offset` by `delta`.  Self's `header_offset` is
   strictly less than `after_offset`, so self is not touched here.
3. Bump self's `header_block_count += delta_blocks` and
   `data_offset += delta_bytes` via `fetch_add`.
4. Serialize the now-larger card list into the (now-larger) reserved
   region with the normal write path.

Because the shared `Arc<HduOffsets>` is co-owned by every HDU view and
every FITSHeader view of that HDU, previously-issued handles transparently
see the post-grow offsets — there is no stale-view problem and no need
for the user to re-fetch via `fits[i]`.

The back-to-front copy order is what makes the in-place shift safe with
overlapping source and destination.  The argument is documented in detail
in the doc-comment on `shift_file_tail_and_update_offsets` in `common.rs`.

## Image overflow: in-place data-section grow

`ImageHDU.extend` uses the same `shift_file_tail_and_update_offsets`
primitive when growing an HDU that is not the last on disk.  The
mechanics differ from header grow only in *where* bytes are inserted
and *what* the caller updates on self:

1. Compute `delta = new_padded - current_padded` (block-rounded new
   data size minus block-rounded old data size).
2. If `file_len > current_hdu_end` (non-last HDU): call
   `shift_file_tail_and_update_offsets(file, layout,
   after_offset = self.data_offset + current_padded, delta, taint)`.
   Then `zero_fill_range(file, after_offset, delta, taint)` to overwrite
   the gap left behind (the first `delta` bytes of the shifted tail,
   which contain stray bytes from the old next-HDU's header).
3. Else (last HDU): just `set_len(new_hdu_end)` — the OS zero-extends,
   no shift needed.
4. Update self's `NAXISn` card in memory (clone-then-replace, disk-write-
   before-commit ordering).  Self's `HduOffsets` are **unchanged**: only
   the data section grew, header_offset/data_offset/header_block_count
   are all the same.
5. Write the updated header to self.header_offset.
6. `write_image_data` writes the new rows.

Because data-section grow doesn't touch self's offsets, the helper's
"update everyone with header_offset >= after_offset" rule still skips
self (self.header_offset < self.data_offset <= after_offset) — only
later HDUs shift.  The shared `Arc<HduOffsets>` model gives the same
transparent post-grow handles as the header-grow path.

**Taint semantics in image grow:** same as header grow — failures inside
the shift loop, zero-fill, header write, or image-data write all taint
because the file layout has been mutated by then.  Pre-shift failures
(file lock, metadata) do not taint.

**Table data grow uses the same primitive.**  `TableHDU.append`
(Phase 3 of the table-write roadmap) calls
`shift_file_tail_and_update_offsets` with `after_offset =
self.data_offset + current_padded` to make room for the appended
rows.  For VLA tables (Phase 4) the padded extent also includes the
heap (PCOUNT) and the existing heap is relocated forward to sit
after the new main rows.  Self's offsets stay unchanged just like
image grow.

**Taint semantics in the grow path:** pre-shift failures (file lock,
metadata, `set_len`) do NOT taint — nothing on disk has moved yet.  Any
failure inside the shift loop, the post-shift `flush`, or the subsequent
header `write_all`/`flush` DOES taint, because the file may now be
inconsistent.  See the `check_not_tainted` block above.

## Heap repack: drop orphans + shrink the file

Both `TableHDU.repack()` (VLA tables) and
`CompressedImageHDU.repack()` rebuild the heap with only the bytes
that live descriptors point at, dropping orphans accumulated by
`__setitem__` (and `extend` on the compressed side).  When the new
padded extent is smaller than the old, the on-disk file shrinks too:
the last HDU uses `set_len`; non-last HDUs go through a backward
file-tail shift via the shared primitive
`shift_file_tail_backward_and_update_offsets` (common.rs), the
mirror image of the forward grow primitive — destination offsets
land BEFORE source so the copy walks front-to-back without overlap
worries.  Later HDUs' offsets bump DOWN by `delta` via the shared
`Arc<HduOffsets>` model, so previously-issued handles transparently
see the post-shrink layout.

Algorithm (same shape on both sides):
1. Read the whole main table + the whole old heap into RAM under a
   single file lock.
2. Walk every row × every descriptor column.  For each live cell
   (`nelements > 0`) copy its bytes from the old-heap snapshot into
   a fresh `Vec<u8>` (in row-major × descriptor-order, which matches
   what the bulk write path emits), record the new offset, and
   rewrite the in-memory descriptor to point at the new location.
3. Drop the old heap; if `new_pcount == current_pcount` already
   (compact), bail out.
4. Compute new padded extent.  Write the rebuilt main table + new
   heap back at the same `data_offset`.
5. If the new padded extent is smaller: shift the file tail
   backward (non-last) or `set_len` (last).
6. PCOUNT update via the standard disk-write-before-commit header
   rewrite + taint discipline.
7. Compressed side also `cache.clear()`s the tile cache (entries
   pointed at the old heap layout).

**Scope limitations.**  Both sides reject the call up front when
the file has a non-default `THEAP` (where `THEAP != NAXIS1*NAXIS2`)
because repack writes the new heap at the default position and
would corrupt a file with a custom layout.  rustfits never emits
THEAP itself, so this only blocks repack on files written by other
tools with a non-standard heap offset — workaround is to clone the
file through a fresh `create_*` + write.

**Implementation.**  `repack_table_heap` in
`src/hdu_table/write_vla.rs` and `repack_compressed_heap` in
`hdu_image_compressed.rs`.  Same shape,
different descriptor-column enumeration: tables iterate every VLA
column per row; compressed images iterate the primary
`COMPRESSED_DATA` column plus the optional `GZIP_COMPRESSED_DATA`
and `UNCOMPRESSED_DATA` fallbacks (each with its own
`inner_byte_width`).  Validate-then-mutate (descriptor sanity
checked before any write); mid-write failures taint the file.
Pre-write failures (THEAP rejection, locks) don't taint.

**Tests.**  `tests/test_table_vla_repack.py` (9 cases: drop orphans,
shrink last HDU, no-op compact, no-op non-VLA, multiple VLA
columns, mixed fixed+VLA, non-last shift, all-empty cells,
repack→setitem→repack).  `tests/test_image_compressed_repack.py`
(11 cases: drop orphans, shrink last HDU, no-op compact, non-last
shift, algorithm matrix Gzip1/Gzip2/Rice1/Hcompress1, quantized
float, unquantized float, cache invalidation).

## Feature status: supported and missing

Snapshot of what's implemented across (image | table) × (read | write).
Items in **missing** are roughly ordered by likely value; cross them
off as they land, revisit ordering when new use cases show up.

### Inspection accessors (shared, no I/O)

Every HDU has `extname` (Optional[str]), `extver` (int; default 1 per
FITS standard), and `has_data` (True iff NAXIS > 0 AND every NAXISn > 0
— suitable for picking the first HDU worth reading).  `ImageHDU` adds
`shape` (tuple, numpy axis order), `dtype` (numpy.dtype), `ndim`,
`size` (total pixels; 0 for NAXIS=0), `bitpix` (raw FITS value),
`unit` (BUNIT, informational), and `__len__` (== shape[0]).
`TableHDU` adds `nrows`, `ncols`, `colnames` (tuple, case preserved),
`units` (dict, informational), and `__len__` (== nrows).
`AsciiTableHDU` has `nrows` and `__len__` so generic code can iterate
any HDU type uniformly.

### Image read

**Supported.**  Whole-array read via `ImageHDU.read()`.  Slicing via
`__getitem__` over arbitrary mixes of slice / int / list, including
fancy row-list selection.  When every axis is indexed with an int the
result is a numpy scalar of the *scaled* BITPIX dtype, matching the
numpy rule for `arr[i, j, ...]`; mixed slice + int returns an ndarray.
Internal strip-layout walk keeps peak RSS at ~1 MiB above the output
array.  BSCALE/BZERO applied on by default (`ImageHDU.read(scale=
False)` opt-out): unsigned-int trick on BITPIX 16/32/64 with
BZERO=2^(n-1) returns the matching `u2`/`u4`/`u8` dtype; BITPIX=8 with
BZERO=-128 returns `i1`; anything else promotes to `f8` via
`physical = stored * BSCALE + BZERO`.  `__getitem__` always scales
(no opt-out, matches table convention).  BLANK masking via
`ImageHDU.read(mask_blank=True)` (opt-in, integer BITPIX only): returns
a `numpy.ma.MaskedArray` with True at pixels whose stored value matches
the header's `BLANK`; comparison happens in stored (pre-scaling) space
per the FITS spec, so it composes correctly with the unsigned-int
trick and general scaling.  When `BLANK` is absent the return is a
MaskedArray with `nomask` for consistent return type.  Float BITPIX +
`mask_blank=True` is rejected up-front because the spec forbids BLANK
on floating-point arrays.

**Tile-compressed image read** is supported end-to-end for all five
algorithms (RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1 with SMOOTH, PLIO_1)
including quantized + unquantized floats, slicing via `__getitem__`,
a bytes-bound LRU tile cache, and GZIP and uncompressed fallback
columns.  Details in the "Tile-compressed images (ZIMAGE)" section
below.

**Missing.**
- (None tracked on the image-read side.)

### Image write

**Supported.**  `FITS.create_image_hdu(dtype, shape, *, extname,
extver)` writes header + zero-filled data.  Accepted dtypes:
`u1/i2/i4/i8/f4/f8` (BITPIX-direct) plus `u2/u4/u8/i1` (unsigned-int
trick — see below).  `ImageHDU.write(data, start=...)` does an
explicit-start bulk write.  `__setitem__` is symmetric with
`__getitem__`: anything readable is writable.  RHS can be a scalar
(Python int/float, numpy scalar, or 0-d ndarray — broadcast across
the selection) or a shape-matching ndarray.  Stepped slices fall
into per-pixel writes via the same strip-layout walk as the read
path.  `ImageHDU.extend(new_shape)` grows the data section in place,
shifting the file tail and bumping later-HDU offsets when the image
is not the last HDU on disk.  Mid-write I/O failures taint the file
(close + reopen to recover).

Unsigned-int trick on write (symmetric with the read side):
`create_image_hdu(dtype='u2'/'u4'/'u8'/'i1', ...)` emits
`BITPIX=signed-int + BSCALE=1 + BZERO=2^(n-1)` (or `BZERO=-128` for
`i1`).  `write` / `__setitem__` / `extend` accept either the
BITPIX-native (signed) dtype OR the scaled (unsigned/i1) dtype as
input; scaled-dtype input is reverse-transformed on the fly via XOR
+ view-cast (the inverse of `apply_image_scaling`).  Dispatch lives
in the shared `normalize_input_dtype` helper which is the single
source of truth for the write-side dtype rules — used by
`write_image_data`, `write_image_slice`, and `ImageHDU.extend`.
**No-copy fast path**: when input dtype already matches BITPIX,
`normalize_input_dtype` returns the input via a Python refcount
bump only; `RawBuffer::acquire` pins the numpy buffer; native-endian
writes flow straight from the user's array to disk.  Reverse-
transform allocates one new array (unavoidable).

General reverse-scaling on write (symmetric with the read side's
General branch): when an HDU has non-trivial `BSCALE`/`BZERO` that
isn't the unsigned-int trick (e.g. `BSCALE=0.5, BZERO=10`),
`write` / `__setitem__` / `extend` accept f8 (physical) input
alongside BITPIX-native input.  f8 input goes through
`reverse_general_scaling`: `stored = (physical - BZERO) / BSCALE`.
For integer BITPIX, non-finite values (NaN/Inf) raise, rounding is
half-to-even via `np.rint`, and post-rounding values outside the
BITPIX range raise (no silent saturation, no wrap).  For float
BITPIX (-32), no rounding/bounds checks — the cast is exact within
target precision.  BITPIX=64 with non-trivial scaling carries the
usual f64-precision caveat (i64 values beyond 2^53 lose precision
through the f8 intermediate); the upper bound check uses 2^63 -
1024 (largest f64 below 2^63).

**BLANK / MaskedArray support** (uncompressed; symmetric with
compressed — see
[docs/internal/zimage.md](docs/internal/zimage.md)).
`create_image_hdu(..., blank=<sentinel>)` emits the `BLANK`
card (in stored space after the unsigned-int trick transform);
rejected for float dtypes.  `hdu.read(mask_blank=True)` returns
a `numpy.ma.MaskedArray` masking pixels matching `BLANK`
(comparison in stored space, per spec).  `write` / `__setitem__`
/ `extend` accept `numpy.ma.MaskedArray` input; masked positions
auto-fill with the sentinel from the header (NaN for float HDUs).
The shared `unwrap_masked_input` helper is documented in the
extracted ZIMAGE doc.

**Tile-compressed image writes (`ZIMAGE`)** — feature-complete.
See [docs/internal/zimage.md](docs/internal/zimage.md) for the
full surface: all 5 algorithms, all integer + unsigned-trick +
unquantized-float + quantized-float dtypes, `extend(data)`,
`__setitem__`, `blank=` / `mask_blank=True` / MaskedArray input,
`Gzip1(level=)` / `Gzip2(level=)`.

**Scalar broadcast with scaling.**  `__setitem__` scalar RHS
(`img[k] = 42`) on a scaled HDU accepts the value in user-facing
space — same rule as the ndarray RHS path.  For unsigned-trick
HDUs (e.g. u2 stored as i2 + BZERO=32768) the user passes the
unsigned value (50000 works on a u2 HDU); for generally-scaled
HDUs the user passes the f8 physical value, and the same
`reverse_general_scaling` used by the ndarray path applies (with
its rint + bounds check on integer BITPIX, finite-check
rejection for NaN/Inf on integer BITPIX).  No-scaling HDUs take
the original BITPIX-direct fast path (extract scalar at native
dtype, no asarray round-trip).  Implementation: `scalar_to_be_bytes`
in `hdu_image.rs` dispatches on `image_scaling_kind`; the scaled
branch promotes the scalar to a 0-d ndarray of the scaled dtype
and routes through `normalize_input_dtype`.

**Empty-shape create + extend-later.**  Shipped.  Parallel to
`create_table_hdu(nrows=0)` + `append()`, `create_image_hdu`
accepts a zero in the slowest-varying axis
(e.g. `create_image_hdu("f4", (0, 1024))`) and a subsequent
`ImageHDU.extend(data)` fills it incrementally.  Inner axes
stay strictly positive (the FITS standard forbids zero pixels
on inner axes — `create_image_hdu`'s dim loop in `fits.rs`
allows `dims[0] >= 0` but keeps `d <= 0` rejection on every
other axis; same split in `create_compressed_image_hdu_impl`).
The zero-data-section case needed no special handling:
`data_section_padded(0) == 0`, `append_header_and_data_to_file`
skips its `set_len` when `data_padded == 0`, and the extend
path's "0-rows → N-rows" grow exercises the same
`shift_file_tail_and_update_offsets` / `set_len` code the
partial-last-tile case already does.  Works for both
uncompressed and compressed HDUs across all algorithms except
HCOMPRESS_1 (whose own dim >= 4 check rejects axis 0 == 0 —
workaround: use Gzip1/Gzip2/Rice1).  Useful for streaming
writers and the fitsio-style copy loop documented under
`FITS.write`: a destination with an empty image HDU + an
`extend(src.read())` per source HDU.  Tests in
`tests/test_image_write.py` (`test_empty_*`, 13 cases:
1-D/2-D/3-D uncompressed, Gzip1/Rice1 compressed, copy-loop
pattern, non-last HDU shift, inner-axis rejection, HCOMPRESS
rejection, negative-axis-0 rejection).

**Missing.**
- (None tracked on the image-write side.)

### Table read

**Supported.**  Fixed L/B/I/J/K/A/E/D/C/M/X (with TDIM reshape on
numeric and X).  Variable-length P/Q descriptors with inner
L/B/I/J/K/A/E/D/C/M (Object dtype, one ndarray per row; A as str or
`as_bytes` bytes).  THEAP respected.  TSCAL/TZERO scaling on by default
(unsigned-int trick → matching unsigned dtype, general → f8;
`scale=False` opt-out).  TNULLn integer-sentinel masking on fixed
B/I/J/K columns via `mask_null=True` (opt-in): returns
`numpy.ma.MaskedArray` with per-field bool mask; compare is in
stored-int space (pre-scaling) so it composes correctly with all
TSCAL/TZERO paths.  TUNITn surfaced via `TableHDU.units` (informational).
`rows=` / `columns=` subsets + `__getitem__` column-subset objects.
Bare-int indexing `hdu[i]` returns a 0-d numpy record (np.void),
matching `structured_arr[i]` semantics — distinct from `hdu[[i]]`
and `hdu[i:i+1]` which still return shape-(1,) structured arrays.

Quirk worth knowing for the MaskedArray return: numpy.ma materializes
an all-False structured bool mask on construction with structured
input regardless of `nomask` being passed.  So `MaskedArray.mask is
np.ma.nomask` holds for single-column reads (plain ndarray) but NOT
for full-table reads (structured) — even when no row was actually
masked.  Tests assert "no element is masked" rather than identity
against `nomask`.

**Row / chunk iteration.**  `for row in hdu:` (i.e.
`TableHDU.__iter__`) yields one `np.void` record per row — the same
0-d scalar `hdu[i]` returns.  `hdu.iter(*, chunksize=None,
columns=None, scale=True)` is the explicit form: `chunksize=None`
(default) yields rows, `chunksize=N` (>0; 0/negative rejected) yields
structured ndarrays of ≤N rows (last chunk short).  `columns=` and
`scale=` are forwarded to `read`, so `hdu.iter(columns=["x"])` is the
supported way to iterate a column subset (a single-column iter still
yields 1-field records — use `row["x"]`; there is no iterable
column-subset object by design).

Implementation in `src/hdu_table/iter.rs`: a `TableIter` pyclass
(no Rust generator — Python drives `__next__`, `Ok(None)` ⇒
StopIteration) holding the current buffer + cursor.  Row mode reads
`buffersize` rows into a buffer and yields `buf[k]` out of it,
refilling when spent; chunk mode reads exactly `chunksize` rows per
`__next__`.  **The refill calls the HDU's own polymorphic
`read(rows=slice, columns=, scale=)`** via `call_method`, so
`CompressedTableHDU` (which overrides `read` + `__len__` + `dtype`)
iterates correctly with zero compressed-specific code — `__iter__` /
`iter` are defined once on `TableHDU` and inherited.  `nrows` +
`itemsize` are snapshotted POLYMORPHICALLY (`slf.len()` /
`slf.getattr("dtype")`) at construction so the compressed subclass
reports its uncompressed-view schema, not the on-disk `1QB` layout.
Row-mode buffer is auto-sized to an ~8 MiB byte budget
(`rows = 8 MiB / itemsize`); not user-configurable yet (documented in
the `iter` docstring — for a huge-row table, drive a manual
`hdu[lo:hi]` loop).  Contract: `nrows` is frozen at iterator creation
(appends mid-iteration aren't seen); closing the file mid-iteration
makes the next batch read raise the usual closed-file error.  Tests
in `tests/test_table_iter.py` (33 cases, parametrized over plain +
compressed): row/chunk content + sizes, `iter()`==`__iter__`,
independent cursors, `chunksize=1` → shape-(1,) arrays, chunk >
table, `columns=`/`scale=` forwarding, single-column 1-field quirk,
empty table, `chunksize=0`/negative rejection, nrows snapshot,
multi-refill via wide rows, VLA columns.

**Missing (ordered by likely value).**
1. **Variable-length P/Q with `repeat > 1`** — currently rejected.
   Rare (most VLA columns are `1Pt`) but legal.  Multi-descriptor
   means N descriptors per row, each pointing at its own heap cell.
   Field dtype would need to be an Object array of shape `(repeat,)`
   per row, or some other reshape — decide before coding.  See
   `docs/vla-shapes.md` for the per-shape mental-model diagram.
2. **Variable-length P/Q with TDIMn** — currently rejected.  TDIMn on
   a P/Q column would mean "reshape each heap cell to these dims",
   useful for VLA-of-images.  Each cell still uses the inner element
   type; the reshape is just on the ndarray after the heap read.
   FITS only allows ONE variable axis per cell (TDIM has exactly
   one zero) — see `docs/vla-shapes.md` for the limitation note
   and standard workarounds for fully-variable `(n, m)` shapes.
3. **VLA TNULL masking** — fixed-col TNULL is implemented; VLA
   columns with TNULL in the header are rejected when `mask_null=
   True`.  Adding support means a per-row bool ndarray for each
   masked VLA cell (parallel Object dtype mask field, or
   MaskedArrays for each cell — decide representation before coding).
4. **`max_size`-style read for variable columns** — fitsio offers a
   mode where each variable cell becomes a fixed-size N-D array
   padded to the largest cell.  Explicitly deferred (user request);
   noted here so we don't forget.
5. **`TDISPn`** — display format hint.  Informational, similar
   shape to TUNIT but rarely used.

### Table write

**Supported.**  `FITS.create_table_hdu(dtype, nrows=0, *, extname,
extver, units, var_dtypes, heap_format)` maps a numpy structured
dtype to TFORMn / TDIMn / TUNITn cards.  `var_dtypes={col:
inner_dtype}` sidecar declares VLA columns (numpy `'O'` field +
sidecar entry); `heap_format='P'` (default; 8-byte descriptors,
32-bit nelements/offset, 4 GB heap ceiling) or `'Q'` (16-byte,
no practical ceiling).  The parameter was originally called
`descriptor` — renamed to `heap_format` (Phase 7 prep, 2026-05) so
the kwarg conveys its purpose (the heap's addressing format)
rather than the FITS-spec term for `{nelements, offset}` pairs.  `TableHDU.write(data)` bulk-writes the table (fixed +
VLA columns supported); accepts structured ndarray, dict `{name:
ndarray}`, or list/tuple of arrays with `names=[...]`.
`TableHDU.__setitem__` covers single-row `hdu[i] = record`, slice
`hdu[a:b[:s]] = arr` (step=1 fast path, step>1 strided), and whole-
column `hdu["col"] = arr` (per-row direct cell writes; no read-
modify-write since the other columns' bytes are preserved by not
being touched).  `TableHDU.append(rows)` (alias `extend`) grows
NAXIS2 and the data section to append new rows, shifting the file
tail and bumping later-HDU offsets when the table is not the last
HDU on disk.  For VLA tables, append relocates the existing heap
forward to sit after the new main rows.  Validate-then-mutate so
dtype errors leave the file untouched.  Mid-write I/O failures
taint the file the same way as image writes.

**VLA `__setitem__`.**  Shipped.  Full row-selection surface
supported on tables with at least one variable-length column:
- `hdu[i] = record` — single row write.
- `hdu[a:b] = arr` — step=1 slice write (contiguous strip writer).
- `hdu[a:b:s] = arr` — stepped slice (positive step; negative or
  zero step rejected for parity with fixed).
- `hdu[[i, j, k]] = arr` — fancy-row write.  Duplicates in the
  list follow numpy fancy-assignment semantics (last write wins).
- `hdu["vla_col"] = arr` — whole-column write (Object dtype, one
  inner ndarray per row).

Heap model: new cells are appended at the end of the existing heap
(heap_start_offset = current PCOUNT) and the old cells become
orphans.  PCOUNT grows monotonically with every mutation.  Matches
the compressed-image `__setitem__` pattern; call `hdu.repack()` to
rebuild the heap with only live cells (see "Heap repack" below).

Implementation lives in `src/hdu_table/setitem.rs` (with the VLA
primitives it calls — `validate_vla_cell`, `plan_vla_heap_layout`,
`write_vla_data_range`, `write_vla_data_strided` — in `write_vla.rs`).
Four helpers:
- `setitem_single_row_vla_aware`, `setitem_row_slice_vla_aware`,
  and `setitem_fancy_rows_vla_aware` share a
  `setitem_rows_vla_aware_inner` core that takes a `VlaRowSpec`
  enum (`Contiguous { first_row, count }` or `Strided { disk_rows }`)
  and routes to either `write_vla_data_range` (strip-walk) or
  `write_vla_data_strided` (per-row seek+write) accordingly.
- `setitem_single_column_vla` writes ONLY the column's descriptor
  bytes at each row (other columns' bytes untouched) plus the new
  heap.

The two writers in `write_vla.rs` share the heap-buffer build
(`build_vla_heap_buf`) and the heap-write + flush tail
(`write_heap_and_flush`); they differ only in their main-row write
loop (1 MiB strip walk vs per-row seek+write).

Validate-then-mutate (input fully validated before any file/header
bytes are touched).  Same taint discipline as every other write —
mid-write failures taint the file (close + reopen to recover).
Fixed-column writes on a VLA-bearing table still take the existing
fixed-only path (PCOUNT and heap untouched).

**String VLAs (`PA`) on write.**  Shipped.  Declare with
`var_dtypes={col: 'S'}` (or `'U'` / `'A'` / `'S1'` / `'U1'` — bare
lowercase `'s'`/`'u'` rejected to avoid collision with `'u1'` =
uint8).  Per-cell input is a Python `str` (ASCII; non-ASCII raises
with the same shape as the read-side check) or `bytes`/`numpy.bytes_`
(verbatim, including embedded NULs / non-ASCII).  Empty cells
accepted (descriptor is `(0, current_heap_offset)`).  Works through
the full table-write surface: `write`, `append`, `__setitem__`
(single-row / slice / whole-column), `repack`.  All paths share the
existing `validate_vla_cell` / `serialize_vla_cell` machinery via a
small `extract_string_vla_cell_bytes` helper that handles the str/
bytes type-check and ASCII validation in one place.  Cross-tool:
astropy reads our PA columns as per-cell chararrays of single chars;
we read astropy's PA columns as Python str (the natural mapping).
Tests in `tests/test_table_vla_string_write.py` (24 cases).

**Multi-column / fancy-row `__setitem__`.**  Shipped.  Two
additional forms complete the table `__setitem__` surface:

- `hdu[[c1, c2]] = arr` — multi-column subset write.  Value is a
  structured ndarray with the named fields (extras tolerated for
  forward compatibility), length = NAXIS2.  Each column routes
  through the existing single-column writer; if any subset column
  is VLA, that column goes through `setitem_single_column_vla`
  (other subset columns can be fixed).  Case-insensitive name
  lookup; duplicate name in the list raises.
- `hdu[[1, 3, 5]] = arr` — fancy-row write.  Value is a structured
  ndarray of length = len(row list).  Reuses `write_table_strided`
  (the existing stepped-slice writer).  VLA tables are rejected
  with a clear error pointing at the per-row / whole-column
  workarounds (strided VLA writes would need per-row heap layouts).

Single-cell writes go through the symmetric subset form
`hdu["col"][row] = v` — see "Subset `__setitem__`" below.  The
tuple form `hdu[row, "col"] = v` is NOT supported: the read side's
`classify_table_key` has no `Cell` variant (a `(int, str)` tuple
falls through to the iterator branch and raises "sequence must be
all int or all str"), and `__setitem__` matches that exact
behavior.  Anything readable via `hdu[key]` is writable via
`hdu[key] = value`, nothing more.

`SetItemKey` has five variants (mirrors the read-side `TableKey`);
`classify_setitem_key` mirrors `classify_table_key`'s iterable
inspection.  Tests in `tests/test_table_setitem_multi.py`.

**Subset `__setitem__`.**  Shipped.  Both subset objects returned by
`hdu["name"]` and `hdu[["a","b"]]` are now writable, so anything the
subset can READ via `[rows]` can also be WRITTEN via `[rows] = v`:

- `hdu["name"][i] = v` / `[i:j] = arr` / `[[i,j,k]] = arr` —
  one-column rows-restricted writes (cell / slice / fancy).
- `hdu[["a","b"]][i] = record` / `[i:j] = arr` / `[[i,j,k]] = arr` —
  column-subset rows-restricted writes (record / slice / fancy).

Both forms route through a shared `resolve_rows_key` (int / slice /
iterable) and loop per-cell through `setitem_cell` — simple and
correct.  Cards are re-snapshotted between cells so VLA cell writes
(which mutate PCOUNT in the header) see fresh state from the
previous iteration.  Performance: O(rows × cols) seek+write
syscalls; fine for typical "patch a few cells" workloads.  If a
hot-path workload ever needs bulk per-column slice writes, the
existing `write_table_one_column` / `write_table_strided` can be
specialized for the rows-restricted case.

Tests in `tests/test_table_setitem_subset.py` (20 cases) — single-column
+ multi-column subset across cell / slice / fancy / full-slice /
negative-index / VLA-numeric / VLA-string / round-trip read+write.

**Subset `read()` / `write()` methods.**  Shipped.  Both subset
objects (`SingleColumnSubset` / `ColumnSubset` on uncompressed,
`CompressedSingleColumnSubset` / `CompressedColumnSubset` on
compressed) gained named `read()` and `write()` methods so the
surface is symmetric with `TableHDU.read()` / `TableHDU.write()`
and self-documenting (rather than requiring users to discover
`subset[:]` / `subset[:] = data` from the slicing surface).

Read: `subset.read(*, rows=None, scale=True, mask_null=False)`.
SingleColumn returns a plain ndarray (matches `__getitem__`'s
contract); ColumnSubset returns a structured ndarray with the
named fields.  Forwards `rows=` / `scale=` / `mask_null=` to the
underlying `read_one_column` / `read_table` call (uncompressed)
or `read_compressed_table` (compressed; `mask_null=True` raises
`NotImplementedError` matching `CompressedTableHDU.read`).

Write: `subset.write(data, *, rows=None)`.  With `rows=None`
(default) dispatches to the parent HDU's `set_item(key, data)` —
the efficient strip-based whole-column writer on uncompressed,
the per-tile writer on compressed.  With `rows=<spec>` dispatches
to the subset's own `__setitem__(rows, data)`, equivalent to
`subset[rows] = data`.  Same value-shape contract as the matching
`__setitem__` form.

Tests in `tests/test_table_subset_read_write.py` (29 cases) — read
returns expected shape, matches `subset[:]`, kwargs forward
correctly; write round-trips both wholesale (`rows=None`) and
row-restricted (`rows=` slice / int / fancy) across all four
subset classes, plus unsigned-int trick and the
`mask_null=True` rejection on compressed.

**Add / remove columns (`insert_column` + `delete_column`).**
Shipped.  Schema-edit methods on `TableHDU`:

- `hdu.insert_column(name, data, *, position=None, after=None,
  before=None, unit=None)` — at most one of position / after /
  before may be set; default (all None) appends at the end.
  `after` and `before` accept either a column name (str,
  case-insensitive) or a 0-based integer index (negative wraps).
  `data` is a regular numpy ndarray of shape `(NAXIS2,) + per-cell
  shape`; dtype maps to FITS letter via the same `dtype_to_write_columns`
  rules as `create_table_hdu` (i2/i4/i8/u1/u2/u4/u8/f4/f8/c8/c16/b1
  + S/U strings; unsigned-int trick emits TZERO; subarray shape
  emits TDIM).  VLA columns are also supported via the
  `inner_dtype=` kwarg (Object-dtype input + `inner_dtype='f4'` /
  `'i4'` / `'?'` / etc., paralleling `create_table_hdu`'s
  `var_dtypes={name: ...}`).  Optional `heap_format='P'` (default)
  or `'Q'`.  X-packed bits (fixed or VLA) opt in via
  `bit_packed=True` — single-column equivalent of
  `create_table_hdu`'s `bit_columns=` toggle.
- `hdu.delete_column(name_or_index)` — name (str, case-insensitive)
  or 0-based integer index (negative wraps).  Works on BOTH fixed
  and VLA columns: deleting a VLA column drops the descriptor bytes
  but leaves the heap as-is (those cells become orphans that
  `hdu.repack()` reclaims).  Other VLA columns are preserved (heap
  relocates after the new shorter main rows; descriptor offsets are
  relative to heap start and remain valid).

**Implementation.**  `src/hdu_table/edit.rs`.  Strip-based I/O —
peak memory bounded at ~1 MiB regardless of table size, NOT the
whole-table-into-RAM approach repack() uses.  Insert grows the
row width, so the row shuffler walks back-to-front (writing later
strips first so reads don't clobber unwritten rows); delete
shrinks, so the shuffler walks front-to-back.  The heap (if any)
is relocated forward (insert) or backward (delete) via chunked
in-file copies of the same ~1 MiB size.  Standard ordering:
header rewrite first (may grow header blocks via the shared
`rewrite_header_to_disk` primitive), then data-extent grow/shrink
(may shift later HDUs), then heap relocate, then row shuffle.

**Header card mutation.**  Renumbers every per-column keyword
(TTYPEn / TFORMn / TDIMn / TUNITn / TZEROn / TSCALn / TNULLn /
TDISPn / TBCOLn) for columns at indices shifted by the
insert/delete; inserts the new column's TTYPE / TFORM / +
TDIM/TUNIT/TZERO cards just before END; updates TFIELDS + NAXIS1.
Same disk-write-before-commit + taint discipline as every other
mutation path.

**Scope limitation.**  Rejects non-default THEAP (THEAP !=
NAXIS1*NAXIS2) up front — same constraint as `repack()`.  Files
rustfits creates never set THEAP; this only blocks the operation
on files written by other tools with a custom heap offset.
Workaround: rewrite through a fresh `create_table_hdu` + `write`.

**VLA insert.**  `insert_column` accepts Object-dtype input when
the caller passes `inner_dtype='f4'` / `'i4'` / `'?'` (etc., same
inner-letter dispatch as `create_table_hdu`'s `var_dtypes=`),
optionally with `heap_format='P'` (default) or `'Q'`, and
`bit_packed=True` to emit a `PX`/`QX` bit column instead of
`PL`/`QL`.  Mechanics: re-uses the planner / serializer from
`write_vla.rs` (`plan_vla_heap_layout` with
`heap_start_offset=current_pcount`, `serialize_vla_cell`,
`write_descriptor`).  Order of operations matches the fixed
insert: header rewrite (adds PCOUNT bump + optional
`(maxbits)` hint for X) → grow data extent → relocate existing
heap forward by `nrows * descriptor_size` → strip-walk main rows
back-to-front writing descriptor bytes at the new column slot →
write new cell bytes to the appended-heap region.  Reject
conditions: missing `inner_dtype=` on Object input;
`inner_dtype=` / `heap_format=` passed on non-Object input;
unknown inner-dtype string; per-cell dtype mismatch.

Tests in `tests/test_table_edit_columns.py` (54 cases): default
append + position / after / before with name and index forms,
case-insensitive lookup, unsigned-int trick, multi-D / TDIM,
S-string columns, units, into VLA-bearing tables (heap relocate),
delete by name / positive+negative index, delete VLA column +
repack reclaims orphans, non-last HDU shifts, insert-then-delete
restores layout, astropy cross-read, all rejection paths, a
50k-row strip-loop test to anchor the bounded-memory invariant,
and (Phase: VLA insert) 14 VLA-insert cases covering insert into
fixed-only + VLA-bearing tables, all position forms, P + Q
descriptors, `bit_packed=True` (PX), empty cells, non-last HDU
preservation, insert-then-delete round-trip, astropy cross-read,
and rejection paths (wrong row count, unknown inner dtype, cell
dtype mismatch, name collision, `inner_dtype=`/`heap_format=` on
non-Object input).

**Missing (low priority / niche):**
- **ASCII tables (creating, writing)** — rare in modern files.
- **`TDISPn` on write** — informational, low priority.

### Bit-packed `X` columns

FITS `X` columns store booleans 8-per-byte (MSB-first within
each byte; trailing bits in the last byte are zero).  Default
for `numpy.bool_` stays `L` (one byte per bool) for ecosystem
parity with astropy / fitsio / cfitsio — those all default to
`L` as well, and rustfits files would look unusual in diff/repr
if we flipped the default.  Opt-in is via the `bit_columns=`
kwarg on `create_table_hdu`:

  - `bit_columns=["flags", "mask"]` — per-column opt-in (most
    explicit).  Listed name must resolve to a bool column on
    disk; non-bool / unknown names rejected with a clear
    message.  Name matching is case-insensitive against the
    table columns.
  - `bit_columns=True` — soft global toggle.  Promotes ALL b1
    columns to `X`; leaves non-bool columns at their natural
    letter.  Matches fitsio's `write_bitcols=True` semantics.
  - `bit_columns=False` / `None` / absent — default: b1 stays
    as `L`.

Works for both fixed and VLA columns:

  - **Fixed `X` / `NX`**: scalar `b1` → `1X`, subarray
    `("mask", "b1", (4, 8))` → `32X` + `TDIM='(8,4)'`.
    Non-multiple-of-8 repeats (`13X` → 2 bytes/row, top 13
    bits used, bottom 3 zero) round-trip correctly.
  - **VLA `PX`/`QX`**: combine `var_dtypes={col: "?"}` (or
    `"bool"` / `"b1"`) with `bit_columns=[col]`.  Without
    `bit_columns`, a bool VLA stays as `PL`/`QL` (one byte per
    bool on the heap).  With it, the heap holds
    `ceil(nelements/8)` MSB-packed bytes per cell.  Per the
    FITS spec, the descriptor's `nelements` is the BIT count
    (not byte count).

  The compressed-table path (`compress=...`) also handles `X`
  columns — they go through `GZIP_1` (the only algorithm
  cfitsio's table compressor accepts for X).

**Implementation.**  Read path is straightforward — `convert_x_cell`
(fixed) and `build_var_cell_value`'s X branch (VLA) MSB-unpack
into a numpy bool ndarray.  Write path adds a
`WriteTransform::BitsPackMsb { num_bits }` variant (slow-path
only — source per-cell width = `num_bits` bytes, destination
width = `ceil(num_bits/8)` bytes, so the bulk-memcpy fast path
can't apply; `prepare_structured_input` forces
`layout_matches = false` whenever any column is `X`).  The
classifier (`classify_scalar_numpy_field` / `dtype_to_write_columns`'s
VLA branch) consults `bit_columns` to override `L` → `X` per
the rules above.  `bytes_per_element('X')` returns `None`
(X is bit-counted, not byte-counted); every byte-size
computation in the write/append/setitem/repack paths
explicitly branches on `tform_letter == 'X'` to use
`ceil(nelements/8)`.

**Astropy `(maxlen)` hint on PX/QX.**  The FITS spec treats
the `(maxlen)` field on `1PX(N)` as informational (the real
per-cell length is the descriptor's `nelements`), but
astropy's `from_tform()` rejects `1PX` without it (regex
parses fine, but `FITS2NUMPY` doesn't include `X` — see the
documented limitation note below).  rustfits emits
`1{P,Q}X(maxbits)` after each VLA-X write via
`set_x_vla_tform_maxlen_in_cards`; the update is monotonic
(an `append` or `__setitem__` that lengthens a cell bumps the
hint, but a shorter write never decreases it).  Other VLA
letters (`PE`, `PJ`, etc.) still ship without the hint
because no library needs it for those.

**Astropy limitation (documented, not a bug here).**
`astropy.io.fits.column.FITS2NUMPY` doesn't include `'X'`, so
even with `(maxlen)` present, `_FormatP.from_tform('1PX(N)')`
raises `VerifyError: Invalid column format`.  rustfits' tests
pin this limitation (`test_astropy_pxqx_documented_limitation`)
so an astropy upgrade that adds X support surfaces here.
fitsio reads PX/QX correctly (with a one-time warning about
the maxlen) and cfitsio supports it natively, so cross-tool
interop is solid in practice — just not with astropy.

Tests: `tests/test_table_bit_columns.py` (18 cases, fixed X)
and `tests/test_table_vla_x_bit.py` (16 cases, VLA PX/QX).

### Cross-cutting (read + write)

- **Random groups (`GROUPS=T`, `PTYPEn`)** — legacy format,
  vanishingly rare in new files.
- **Memory-mapped reads** — chunked sequential I/O already keeps peak
  RSS at ~1 MiB above the output array, so motivation is weak.
- **Streaming / row-iterator API** — for tables that don't fit in
  RAM.  No user has asked yet; add when one does.
- **Remote file reads (`http`/`https`/`ftp`/`ftps`)** — open a FITS
  file from a URL.  **Shipped** (download-then-open, read-only; see the
  "Remote file reads" roadmap below).  Range-based partial reads, and
  `root`/`gsiftp`, are still deferred.
- **In-memory files (`mem://` / `memkeep://`) + gzip read** —
  create / read / extract a FITS file with no disk access, via
  `FITS("mem://", "w+")` + `to_bytes()` / `FITS.from_bytes(b)`; and
  read a gzipped file via a `.gz` path.  **Shipped** (the `Storage`
  seam + `Disk`/`Mem` backends + gunzip-on-open).  The rest of the
  cfitsio driver set (`.gz` write-back, stdin/stdout, shared memory,
  remote range reads) is still sketched — see the "In-memory files +
  the storage-driver abstraction" roadmap below, which plugs each
  remaining backend into the shipped `enum Storage`.

## Top-level convenience functions

Three minimal one-call wrappers in `rustfits/convenience.py`,
re-exported at the package top level so users write
`rustfits.read(...)` / `rustfits.read_header(...)` /
`rustfits.write(...)`:

- `rustfits.read(filename, ext=None, *, header=False)` — opens
  in `'r'`, picks the first HDU with data (`ext=None`) or the
  requested ext, dispatches on HDU type, returns the array
  (and optionally the `FITSHeader`).
- `rustfits.read_header(filename, ext=0)` — opens in `'r'`,
  returns the chosen HDU's `FITSHeader`.  Default `ext=0` reads
  the primary HDU (where file-level metadata typically lives).
- `rustfits.write(filename, data, *, mode='w+', extname=None,
  header=None)` — auto-detects image vs table:
    - plain (non-structured) `numpy.ndarray` → image
    - structured `numpy.ndarray` (`dtype.fields is not None`)
      or `{name: ndarray}` dict → table
    - list-of-arrays + names= → rejected (use the explicit
      `FITS.write_table` form for that)
    - anything else → `ValueError`
  Default `mode='w+'` truncates-or-creates (equivalent to
  fitsio's `'rw'` + `clobber=True`).  Pass `mode='r+'` to
  append HDUs without truncating.  Supported modes: `'r'`,
  `'r+'`, `'w+'`.

**Intentionally minimal.**  These accept only the universal
kwargs (`ext` / `mode` / `extname` / `header`).  No
type-specific knobs (`compress=`, `quantize=`, `blank=`,
`var_dtypes=`, `units=`, `bit_columns=`, `scale=`, `rows=`,
`columns=`, ...).  The boundary keeps `convenience.py`
genuinely convenient (no kwarg-sync burden against the
underlying create/write surface) and pushes callers who need
knobs into the explicit `with FITS(...) as f: f.write_image(
...)` shape, which is two lines and reads more clearly.

The `write` dispatch logic lives on the Rust side as
`FITS.write(data, *, extname=None, header=None)` — the
top-level wrapper is a 2-line `with FITS(filename, mode) as f:
f.write(data, ...)` around it.  This is the form fitsio users
reach for when copying HDUs between files without caring about
type:

```python
with rustfits.FITS(infile) as src:
    with rustfits.FITS(outfile, "w+") as dst:
        for hdu in src:
            if hdu.has_data:
                dst.write(hdu.read())
```

### Rich tier — `FITS.write_image` / `FITS.write_table`

`FITS.write_image(data, *, extname, extver, compress, quantize,
blank, header)` and `FITS.write_table(data, *, names, extname,
extver, units, var_dtypes, bit_columns, heap_format, compress,
ztilelen, header)` are pymethods on `FITS` in `src/fits.rs`
that combine `create_*_hdu` + `write` into one call.  Both
return the new HDU (`ImageHDU` / `CompressedImageHDU` /
`TableHDU` / `CompressedTableHDU`) so callers can continue
operating on it within the same `FITS` handle.

These are NOT exposed as top-level filename-taking wrappers —
that boundary was deliberately collapsed (2026-05) to keep
`convenience.py` minimal and avoid keyword-sync drift between
the wrappers and the underlying methods.  For one-call write
+ close from a filename, the explicit shape
`with rustfits.FITS(path, "w+") as f: f.write_image(...)` is
two lines and gets all knobs.

Schema derivation: `create_table_hdu` stays schema-only (first
arg = dtype).  A small free helper
`derive_table_schema_from_data` in `src/fits.rs` handles the
three data-shaped inputs the user-facing `write_table` accepts:

  - **structured ndarray** — `(data.dtype, len(data))`.
  - **dict `{name: array}`** — composes
    `np.dtype([(name, arr.dtype) for name, arr in data.items()])`,
    validates equal lengths.  Empty dict + length mismatch
    rejected; `names=` rejected (the dict keys ARE the names).
  - **list/tuple of arrays + names=`** — same as dict, names
    supplied separately.  Missing `names=` rejected; length
    mismatch between `names=` and the data sequence rejected.

`header=` accepts a `FITSHeader` or a `dict` and routes through
`new_hdu.header.update(header)`, inheriting that method's
protected-key and commentary policies (FITSHeader source silently
skips protected; dict source raises on protected).

`write_table` on an empty file auto-creates the primary HDU via
the existing `ensure_primary` path inside `create_table_hdu` —
no special-casing needed in `write_table` itself.

Tests: `tests/test_convenience_write.py` covers
`FITS.write_image` / `FITS.write_table` across the dtype matrix,
unsigned-int trick, `compress=`, `blank=`/`mask_blank=True`,
MaskedArray input, `header=` (dict + FITSHeader source), all
four reject paths, and auto-primary on table writes.  The
`FITS.write` dispatcher (and the top-level `rustfits.write`
that delegates to it) is covered for the image / structured /
dict matrix plus list-of-arrays rejection, the
copy-from-source-file loop pattern, multi-HDU writes in one
handle, `extname=` / `header=` forwarding, and bit-exact
equivalence between top-level and method form.
`tests/test_convenience_read.py` (15 cases) covers the minimal
`read` + `read_header` surface: default-picks-first-with-data,
ext by int / by extname, removed-kwargs (rows / columns /
scale / mask_null) rejected with TypeError, header=True returns
tuple, read_header default-is-primary / int / extname /
outlives-close / bad-ext rejection.

## Table write roadmap

Plan for getting table creation + writing on par with the image side.
Reading is mature; Phases 1 (create + bulk write), 2 (`__setitem__`),
3 (append/extend), 4 (VLA columns on write/append), and 5 (VLA
`__setitem__`) have all shipped.  Plus the post-roadmap additions
(string VLA `PA` on write, multi-column / fancy / cell forms of
`__setitem__`, subset-object `__setitem__`, `repack()`).  The image
side has the parallel surface and the patterns translated directly.

Add / remove columns (`insert_column` / `delete_column`) shipped
post-roadmap — see the "Add / remove columns" section under "Table
write Supported" above for the API + implementation.

**Coming next.**  Compressed-table (ZTABLE) is feature-complete
through Phase 6c-2e (full `__setitem__` surface, including VLA
columns).  All phases of the original ZTABLE roadmap have
shipped; the next compressed-table work item is open-ended
(performance, additional dtypes, or whatever a real user request
prompts).  See the dedicated "Tile-compressed tables (ZTABLE)"
roadmap below for the full status table.

**Phase 6 — out of scope for now.**  ASCII tables (rare in modern
files; create / write missing, read returns header only).  `X`
(bit-packed) BINTABLE columns — both fixed and VLA PX/QX — are
fully supported on both read and write (opt-in via `bit_columns=`;
see the "Bit-packed `X` columns" section).

**Phase 1 — `create_table_hdu` + bulk `TableHDU.write()`.**  Foundation
for everything else.
- `FITS.create_table_hdu(dtype, nrows=0, *, extname=None, extver=None,
  units=None)` — maps a numpy structured dtype to TFORMn / TDIMn /
  TUNITn cards, writes the header block, allocates `nrows * row_width`
  zero-filled bytes.
- `TableHDU.write(data)` — dtype-checked, byteswapped bulk overwrite.
- Scope: fixed-width columns only (B/I/J/K/A/E/D/L/C/M, plus subarray
  fields via numpy `(T, shape)`).  No VLA, no ASCII tables.  (`X`
  bit columns were added later; see the "Bit-packed `X` columns"
  section.)

**Phase 2 — `TableHDU.__setitem__`.**  Done.  Symmetric with
`__getitem__`, reusing the Phase 1 `prepare_structured_input` +
`acquire_per_column_array` validators and the same WriteTransform
dispatch.  Forms supported:
- `hdu[i] = record` — single-row write.  Value is a `numpy.void`
  scalar or a shape-`(1,)` structured ndarray; negative `i` allowed.
  Routes through `write_table_data` with `start_offset = data_offset
  + i*row_width` and `nrows=1`.
- `hdu[a:b[:s]] = arr` — slice write.  Value must be a structured
  ndarray of length equal to the slicelength.  `step==1` falls
  through to `write_table_data` (fast/slow path as in bulk write);
  `step>1` goes through `write_table_strided` (per-row seek + write
  with the same per-row strip machinery).  `step<=0` is rejected.
- `hdu["col"] = arr` — whole-column write.  Value is an ndarray of
  shape `(nrows,) + per-cell shape`.  Routes through
  `write_table_one_column`, which does per-row direct writes of just
  `byte_width` bytes per row (no read-modify-write — the other
  columns' bytes are preserved by not being touched).  Strip RMW
  was considered and rejected because it would read+write `~2 ×
  full-table` to modify a thin column slice (pathological when
  `byte_width << row_width`, which is the common case).

Multi-column subset writes (`hdu[[c1,c2]] = ...`), fancy row-list
writes (`hdu[[1,3,5]] = ...`), and the subset-then-rows forms
(`hdu["name"][rows] = ...`, `hdu[[c1,c2]][rows] = ...`) have since
shipped — see the "Multi-column / fancy-row `__setitem__`" and
"Subset `__setitem__`" sections under "Table write Supported"
above.  Single-cell writes go through the symmetric subset form
`hdu["col"][row] = v`; the tuple form `hdu[row, "col"]` was
removed for symmetry with the read side (which never accepted it).

**Phase 3 — `TableHDU.append()` (with `extend` alias).**  Done.
Primary method name is `append` because that's the natural verb
for adding rows to a table (matches list/pandas usage); `extend`
is a thin alias that calls through to `append`, kept for symmetry
with `ImageHDU.extend` so generic code iterating HDUs and calling
`.extend(...)` keeps working.  Accepts the same three input forms
as `write`: structured ndarray, dict `{name: ndarray}`, or
list/tuple of ndarrays with `names=[...]`.

Order is **validate-then-mutate** to keep dtype/shape errors from
leaving the file half-grown: `determine_input_nrows` + the shared
`dispatch_write_input` validator run first (acquiring buffers),
then the file/header are mutated.  Mechanics after validation
mirror `ImageHDU.extend`:

1. Compute `current_padded` / `new_padded` (block-rounded data
   bytes) and `delta = new_padded - current_padded`.
2. If `delta > 0`: last-HDU branch uses `set_len`; non-last branch
   uses `shift_file_tail_and_update_offsets` (which bumps every
   later HDU's offsets) followed by `zero_fill_range` on the gap.
3. Rewrite the NAXIS2 card to disk (disk-write-before-commit
   ordering with taint on failure), then commit the in-memory
   cards.
4. Call `write_table_data` with `start_offset = data_offset +
   current_nrows * row_width` and `nrows = append_nrows`, reusing
   the Phase 1/2 fast and slow paths.

Shared `Arc<HduOffsets>` means previously-issued handles to later
HDUs see the post-shift offsets without re-fetching — same
transparency as the header-grow path.

The dispatch helpers (`dispatch_write_input`, `build_sources`,
`determine_input_nrows`) are shared between `write` and `append`
so the input-form handling stays in one place.

**Phase 4 — VLA columns on write.**  Done for the bulk-write +
append surfaces.  `__setitem__` for VLA columns is the subject of
Phase 5 below.

**API.**  `create_table_hdu(dtype, ..., var_dtypes=None,
heap_format=None)`.  The user puts `'O'` for VLA fields in the
numpy structured dtype and passes a sidecar `var_dtypes={col_name:
inner_dtype_str}` so we can pick a FITS inner letter (we considered
extending the dtype-list 3-tuple to carry inner type, but the
sidecar mirrors the existing `units=` pattern and keeps numpy
dtypes free of FITS-specific DSL).  `heap_format=` is `'P'`
(default; 8-byte descriptors, 32-bit nelements/offset, 4 GB heap
ceiling) or `'Q'` (16-byte, 64-bit, no practical ceiling).  No
maxlen hint is emitted in TFORM for numeric inner letters (read
side already accepts both `1PE` and `1PE(100)` shapes); for `PX`/
`QX` we DO emit `(maxbits)` because astropy's TFORM parser
strictly requires it — see the "Bit-packed `X` columns" section.

**Inner types supported.**  Numeric: `B / I / J / K / E / D / C /
M`; plus `L` (bool), `A` (ASCII string — see "String VLAs (`PA`)
on write"), and `X` (bit-packed; opt-in via `bit_columns=`; see
"Bit-packed `X` columns").

**Write path.**  Dispatches on `any_var_column(&columns)`.  No-VLA
tables take the existing fast/slow strip writer untouched.  With
any VLA column, the path is:

1. `extract_per_column_inputs` pulls a per-column ndarray out of
   the input (structured / dict / list+names).  For structured-
   array input this calls `np.ascontiguousarray` on each per-
   column field view, because `arr[col_name]` returns a strided
   view (stride == record itemsize, not field itemsize) that
   would fail `RawBuffer.acquire`'s C-contiguous check.  Cost is
   one memcpy per fixed column per write; `ascontiguousarray` is
   a no-op when the view is already contiguous (dict / list+names
   inputs pay nothing).  FUTURE: a stride-aware `FixedColInfo`
   carrying `src_stride` and indexing rows by stride would avoid
   this copy — worth doing if it ever shows up in a profile.
2. `build_fixed_col_info` validates each fixed column (same
   contract as `acquire_per_column_array`).
3. `plan_vla_heap_layout` walks every VLA cell in row-major
   per-row order and assigns each a `(nelements, heap_offset)`.
   The cursor starts at `heap_start_offset` (0 for full write,
   `current_PCOUNT` for append), and the returned total is the
   absolute heap end (not just added bytes).
4. Compute new padded data section size and grow the file if it
   exceeds the current padded extent.  Same set_len / shift /
   zero_fill primitives as image grow and fixed-table append.
5. `write_vla_data_range` does the actual I/O: builds the heap
   buffer in RAM (size = added bytes; MVP accumulates rather
   than streaming), then writes main rows + embedded descriptors
   strip by strip via `fill_main_row`, then seeks to the heap
   start and writes the heap buffer.
6. PCOUNT card update via `set_pcount_in_cards` + the standard
   disk-write-before-commit header rewrite.

**Append path.**  Like fixed append, but the existing heap must
relocate forward to sit after the appended main rows.  Order:

1. Plan + validate exactly as in write, but with `heap_start_offset
   = current_PCOUNT`.
2. Read the old heap into memory BEFORE any write — the upcoming
   new-main write may overwrite the first `M * row_width` bytes
   of the old heap region in place.
3. Grow the file by `delta_padded` (block-aligned) if needed.
4. `write_vla_data_range` writes new main rows + new heap bytes
   (at their final position, AFTER the relocated old heap).
5. Seek + write the captured old heap to its new position
   `data_offset + new_main_bytes`.
6. Update NAXIS2 + PCOUNT together via the same header rewrite.

The MVP holds the entire old heap in RAM during step 2-5; a
chunked back-to-front copy would bound peak memory at strip size,
trade for code complexity.  Add when a real workload pushes
heap-in-RAM into pain.

**Refactor.**  Done in `b0e37df`.  The inline VLA branches in
`write()` and `append()` were extracted into the free functions
`write_fixed_only` / `write_vla_aware` / `append_fixed_only` /
`append_vla_aware`; both pymethods are now ~25-line dispatchers
that branch on `any_var_column(&columns)`.  Same pattern as the
post-Phase-1 `dbc7bfc` refactor.

**Phase 5 — VLA `__setitem__`.**  Shipped.  See "VLA `__setitem__`"
under the Table write Supported section above for the API + heap
model (always-append-and-orphan).  19 tests in
`tests/test_table_vla_setitem.py`.

**Phase 6 — out of scope.**  ASCII tables (rare in modern files);
adding / removing columns from existing tables (header rewriting
+ byte shuffling).

## Tile-compressed images (ZIMAGE)

ZIMAGE-format tile-compressed images are stored on disk as a
BINTABLE with `ZIMAGE=T` plus Z-prefixed shape cards.  The user-
facing API mirrors `ImageHDU` (the class subclasses `ImageHDU`
so `isinstance(hdu, ImageHDU)` holds); internally the reader
walks tiles and decodes them.

**Status: feature-complete for typical workloads.**  All five
algorithms (RICE_1, GZIP_1/2, HCOMPRESS_1, PLIO_1) supported on
read AND write, including integer + unsigned-int-trick +
unquantized-float + quantized-float dtypes.  `extend(data)` +
`__setitem__` shipped for every dtype/algorithm combination
(PLIO_1 + unsigned-trick is the one rejected combo because the
reverse XOR produces negatives PLIO can't represent).  Custom
zlib levels via `Gzip1(level=)` / `Gzip2(level=)`.  Quantized-
float mutation reuses each tile's existing per-tile
bscale/bzero/dither seed so unchanged pixels round-trip
bit-exactly (no compounding loss).  BLANK / ZBLANK + MaskedArray
input on both compressed and uncompressed paths.
`add_checksum` / `verify_checksum` (ZHECKSUM / ZDATASUM over the
equivalent uncompressed-image bytes).

Code: `src/hdu_image_compressed/` (pyclass + per-responsibility
files: `hdu.rs`, `meta.rs`, `read.rs`, `write.rs`, `repack.rs`,
`checksum.rs`) and `src/zimage/` (algorithm-specific
encoders/decoders: `gzip.rs`, `rice.rs`, `hcompress.rs`,
`plio.rs`, `quantize.rs`, `tile_io.rs`).  Detection in
`fits.rs::parse_hdus_from_file`: a BINTABLE with `ZIMAGE=T`
routes to `CompressedImageHDU`.  Header detection helper:
`header_has_zimage`.

**For the full phase-by-phase implementation history (Phases
1–10c), design decisions, algorithm-specific notes (cfitsio
port details, byte-exact verification strategy, lossless-
fallback convention, HCOMPRESS axis convention, PLIO TFORM
mechanics, quantized-float dither stream, etc.), open
follow-ups, and the Python-side API + structured `compress=` /
`quantize=` config object design, see
[docs/internal/zimage.md](docs/internal/zimage.md).**

## Tile-compressed tables (ZTABLE)

ZTABLE is the BINTABLE counterpart of ZIMAGE: a normal BINTABLE
shell carrying tile-compressed column data, with the original
table's schema preserved via Z-prefixed cards.  Detection is
`ZTABLE=T`.  Per-column: the original bytes for `ZTILELEN` rows
are transposed to column-major, optionally byte-shuffled
(GZIP_2), compressed, and stored as one `1QB`-descriptor heap
blob.  The pyclass subclasses `TableHDU` so
`isinstance(hdu, TableHDU)` holds.

**Status: feature-complete.**  Read (whole-table, `rows=`,
`__getitem__`, fancy-row, column-subset objects, VLA columns
with the dual-descriptor heap layout, per-(tile, col) tile
cache) and write (bulk via
`create_table_hdu(..., compress=...)` accepting
True/None/string/class/dict-of-overrides, VLA columns including
PA strings, `append()` with merge-into-partial-last-tile,
`repack()` with streaming staging, full `__setitem__` surface
including all row forms, all column forms, all
subset-then-rows forms, and VLA cells) all shipped.
`add_checksum`/`verify_checksum` (ZHECKSUM/ZDATASUM over the
equivalent uncompressed bytes, streamed per-tile incl. the
VLA synthetic heap) covers fixed + VLA columns.  funpack
(cfitsio's CLI decompressor) round-trips rustfits-written
files byte-exactly across the entire test suite.

Code: `src/hdu_table_compressed/` (per-responsibility files:
`hdu.rs`, `meta.rs`, `read.rs`, `write_setup.rs`, `write.rs`,
`append.rs`, `repack.rs`, `setitem.rs`, `checksum.rs`,
`subset.rs`).  Reuses shared helpers from `src/hdu_table/`
(`parse_columns`, `build_numpy_dtype`,
`field_dtype_and_shape`, descriptor codecs) and `src/zimage/`
(`encode_gzip1`, `encode_gzip2`, `encode_rice`,
`gzip_decompress_bytes`).  Detection in
`fits.rs::parse_hdus_from_file`: checks ZTABLE BEFORE ZIMAGE
(defensive — they shouldn't both be set, ZIMAGE wins if they
are).

Accessors override TableHDU's so `nrows`, `__len__`, `dtype`,
`colnames`, `units` parse the ORIGINAL-schema view via
`synthesize_uncompressed_cards`.  Compression-specific
accessors: `compression` returns `{col_name: ZCTYPn_value}`,
`n_tiles` is the on-disk NAXIS2, `ztile_rows` is ZTILELEN.

**Notable gotchas captured in the deep doc:** per-dtype default
algorithm table (matches cfitsio's `fits_compress_table`
defaults); complex columns (C/M) compressed as byte-flat
GZIP_1 (not GZIP_2 — cfitsio's GZIP_2 table *decompressor*
errors on complex even though its encoder writes it);
original-descriptor offsets matter for funpack reconstruction
(can't be all-zero); ZPCOUNT = original-heap size, not
compressed heap; the merge-tile append's per-cell stream copy
+ dual-descriptor blob re-gzip dance; the streaming repack's
fast-path/slow-path split.

**For the full phase-by-phase implementation history (Phases
1 → 6c-2e), the dual-descriptor heap mechanics, the per-tile
read planner, the bench-fixtures-via-`fpack -table` setup
because no Python lib writes ZTABLE, and the funpack-cross-
verification strategy, see
[docs/internal/ztable.md](docs/internal/ztable.md).**

## Remote file reads (download-then-open) — roadmap

cfitsio can open files over the network (HTTP/FTP/root drivers).
This is the rustfits plan for the equivalent.  **Status: flavor #1
download-then-open shipped for `http`/`https` (2026-05-28) and
`ftp`/`ftps` (2026-05-28); flavor #2 (range reads) deferred;
`root`/`gsiftp` deferred (need separate protocol crates).**

Two distinct flavors, very different cost:

1. **Download-then-open (shipped: http/https/ftp/ftps).**  Fetch the
   whole remote file into a `Storage::Mem` buffer, then parse it like
   any in-memory file.  Read-only against the remote (no write-back);
   pays the full download even for a one-tile read.  This is
   essentially what cfitsio does for most of its network drivers.
   See "Design (flavor #1, as built)" below.
2. **Range-based partial reads (deferred, NOT this roadmap).**
   HTTP `Range` requests so `fits["sci"][100:200]` or a single
   compressed tile pulls only the bytes it needs — the real
   payoff for a multi-GB `.fz` on a server.  The storage seam this
   needs is **already done**: `common.rs`'s `FileHandle` is now
   `Arc<Mutex<Option<Storage>>>` (see the "In-memory files +
   storage-driver abstraction" roadmap below).  A range-read source
   is a lazy, read-only backend; because it carries its own state
   (URL, http client, position) and isn't cheaply enumerable, it's
   the specific case that would justify switching `enum Storage` to
   `Box<dyn FitsStorage>` rather than adding a variant.  Read-only
   (the in-place grow / `shift_file_tail` / taint machinery is
   fundamentally local), and depends on the server honoring Range.

### Design (flavor #1, as built)

Implemented by downloading the whole file into a `Storage::Mem`
buffer and parsing it like any in-memory file — simpler than the
original temp-file sketch, and it reuses the `Mem` backend the
storage seam already provides (no `TempPath` guard, no new `FITS`
field, no cleanup).

**Where it lands.**  One branch at the top of `FITS::new`
(`src/fits.rs`), checked before the local-`.gz` and disk branches:

```rust
} else if is_remote_url(&filename) {       // http:// or https://
    if mode != "r" { /* reject: remote is read-only */ }
    let mut bytes = download_remote(py, &filename)?;   // GIL released
    if url_path_is_gz(&filename) { /* gunzip bytes */ }
    Storage::Mem(Cursor::new(bytes))
}
```

`FITS.filename` keeps the URL (for repr).  `to_bytes()` returns the
downloaded (and, for a `.gz` URL, decompressed) bytes.

**Decisions (as built).**
- *HTTP crate: `ureq` 3.x* (blocking-native, rustls TLS, no tokio) —
  the right fit for the synchronous PyO3 path.  `download_remote`
  does `ureq::get(url).call()?.into_body().into_reader()` →
  `read_to_end`.  **Kept default-on, NOT behind a Cargo feature.**
  Footprint is moderate: ~30 transitive crates (≈29 net new;
  `flate2` was already ours) on `ring` + `rustls` — the leanest real
  HTTPS stack (`reqwest` would add tokio + hyper + far more).
  One-time `ring` `cc` build; minor `.so` size.  Not worth the
  cfg-gating complexity; revisit only if the build weight becomes a
  real complaint.
- *Whole file in RAM*, not streamed to a temp file.  The original
  sketch favored a temp file for bounded memory; the `Mem` backend
  makes the in-RAM path trivial and consistent with `mem://` /
  `.gz`.  For huge remote files the real answer is flavor #2 (range
  reads), which avoids downloading the whole thing at all — so
  bounded-memory *whole-file* download isn't worth the temp-file
  complexity.  (A streamed-to-temp-file variant stays a possible
  follow-up if a workload needs it.)
- *GIL released during the fetch* via `py.detach(...)` (pyo3 0.28's
  rename of `allow_threads`).
- *Read-only, enforced early.*  `r+`/`w+` raise before any network
  request.  `rustfits.read` / `read_header` get remote support for
  free (they open via `FITS`).
- *FTP crate: `suppaftp` 6.x* (`rustls` feature → **shares** ureq's
  rustls 0.23, no second TLS stack).  `download_ftp` parses
  `ftp://[user[:pass]@]host[:port]/path` (anonymous default, port 21),
  forces `FileType::Binary` (FTP defaults to ASCII, which would mangle
  FITS), and `RETR`s into a Vec.  `ftps` = explicit `AUTH TLS` via
  `RustlsFtpStream::into_secure` with a connector built from
  `webpki-roots` (`ftps_connector`); we declare `rustls` +
  `webpki-roots` as direct deps (already in-tree, `default-features =
  false` + `ring` to avoid the aws-lc-rs default).  Plain vs FTPS are
  distinct suppaftp types, so `download_ftp` has two arms.  `root` /
  `gsiftp` still deferred (separate protocol crates).
- *`.gz` composes:* a URL whose path (sans `?query` / `#frag`) ends
  in `.gz` is gunzipped after download (`maybe_gunzip_url`, shared by
  the http and ftp branches), same as a local `.gz` path.
- *No caching.*  Each `FITS(url)` downloads fresh.  An on-disk cache
  keyed by URL + ETag/Last-Modified is a possible follow-up; deferred
  (invalidation complexity) unless repeat-open is a real pattern.

Tests: `tests/test_fits_remote.py` (11 cases, local `http.server`) and
`tests/test_fits_ftp.py` (10 cases, local `aioftp` server in a
background asyncio thread — a test-only dep in
`conda-test-requirements.txt`; we use `aioftp` not `pyftpdlib` because
the latter still imports the `asynchat` module removed in Python 3.12).
Both are deterministic, no external network.  A full `https` / `ftps` handshake
isn't unit-tested locally (would need a CA-valid server); `https` is
all inside `ureq`, and the `ftps` path is exercised by a negative test
(`ftps://` to the plain server → the `AUTH TLS` upgrade fails, proving
the connector build + `into_secure` run).

## In-memory files + the storage-driver abstraction — roadmap

cfitsio's `mem://` driver opens a FITS file entirely in RAM (create,
manipulate, extract bytes — no disk).  Supporting it in rustfits is
the same architectural work as cfitsio's *whole* driver set, so this
roadmap is scoped as "the storage abstraction" rather than just the
mem case.

**Status: the seam + Mem/gzip read backends shipped (2026-05-28).**

| Piece | Status |
|---|---|
| Storage seam (`FileHandle` over `enum Storage`) | ✅ Shipped |
| `Disk` backend (`file://`) | ✅ Shipped |
| `Mem` backend + `mem://` / `memkeep://` + `to_bytes`/`from_bytes` | ✅ Shipped |
| Whole-file `.gz` **read** (gunzip-on-open → `Mem`) | ✅ Shipped |
| `.gz` write-back (recompress-on-close), `.Z`/`.zip` | ⬜ Sketched below |
| `stdin://` / `stdout://`, `shmem://` | ⬜ Sketched below |
| `http`/`https`/`ftp`/`ftps` **download** read (→ `Mem`) | ✅ Shipped |
| `http`/`https` range reads, `root`/`gsiftp` | ⬜ See "Remote file reads" roadmap |

**Gzip read (shipped).**  A path with a `.gz` extension
(case-insensitive, detected by `is_gz_path` in `fits.rs`) is gunzipped
whole into a `Storage::Mem` buffer at open via `flate2::read::GzDecoder`,
then parsed like any in-memory file.  Read-only: `r+`/`w+` on a `.gz`
raise (write-back not implemented).  Decompressed file lives in RAM
(gzip isn't seekable; FITS needs random access) — same caveat as
`mem://`.  `to_bytes()` returns the decompressed bytes.  Only gzip
(`.gz`); `.Z` (LZW) and `.zip` are out of scope (different codecs).
Tests: `tests/test_fits_gz.py` (13 cases).

The seam is realized as an **`enum Storage`** (in `common.rs`), NOT
`Box<dyn FitsStorage>` — see "The shape (as built)" below for the
decision.  The remaining rows below stay as design notes; they all
plug into the same `enum Storage` (a new variant each) or, for the
non-enumerable lazy backends, would be the moment to revisit `dyn`.
Tests for the shipped part: `tests/test_fits_mem.py`.

### The obstacle (shared with remote range-reads) — RESOLVED

`common.rs`'s `FileHandle` used to be `Arc<Mutex<Option<std::fs::File>>>`,
hardcoding the on-disk file.  In-memory needed that source abstracted,
and — unlike the read-only remote-download case — mem must support
**read AND write** (creating a FITS in RAM and extracting the bytes
is the primary use case).  So this was the *more complete*
abstraction, and now that it exists it also unlocks the remote
range-read flavor (#2 in the "Remote file reads" roadmap above) — one
refactor, multiple features.  `FileHandle` is now
`Arc<Mutex<Option<Storage>>>`; the rest of this section is the design
rationale, kept for the remaining (unshipped) backends.

### Why it's tractable

A survey of `src/` (2026-05) found ~350 direct I/O calls across 20
files, but the overwhelming majority are `std::io` *trait* methods —
`seek` (108), `write_all` (74), `flush` (45), `read_exact` (42),
`read` (31) — which `std::io::Cursor<Vec<u8>>` already implements.
Only `set_len` (23), `metadata().len()` (23), and `sync_all` (1) are
`std::fs::File`-specific.  So most call sites compile unchanged
against the abstracted handle; only the File-specific sites got
mechanically rewritten.

**Played out as predicted:** the ~30 `guard.as_mut()` sites needed
zero changes (they just yield `&mut Storage` now); the rewrite was 24
`metadata().len()`→`len()`, 2 `sync_all`→`sync`, and 5 helper-fn
signatures (`&mut std::fs::File`→`&mut Storage`).  Behavior-preserving
— the full suite was unchanged after the seam landed.

### The shape (as built)

The original sketch was a `trait FitsStorage: Read + Write + Seek +
Send` with `Box<dyn FitsStorage>`.  **What shipped is the enum
alternative** the sketch named — chosen because it monomorphizes (no
vtable), every call site keeps a concrete `&mut Storage` (the ~30
`guard.as_mut()` sites needed no change, and the 5 helper fns just
swapped `&mut std::fs::File` → `&mut Storage` with no deref dance),
and it matches the codebase's no-premature-abstraction style:

```rust
// common.rs
pub(crate) enum Storage {
    Disk(std::fs::File),
    Mem(std::io::Cursor<Vec<u8>>),
}

// std Read/Write/Seek impls forward by matching the variant.
// Three inherent methods cover the File-specific operations:
impl Storage {
    fn set_len(&mut self, size: u64) -> io::Result<()>;  // File::set_len
                                                          // / Vec::resize
    fn len(&self) -> io::Result<u64>;       // metadata().len() / vec.len()
    fn sync(&self) -> io::Result<()>;       // fsync / no-op
    fn read_all(&mut self) -> io::Result<Vec<u8>>;  // backs to_bytes()
}
```

`FileHandle` is now `Arc<Mutex<Option<Storage>>>`; `lock_file` returns
`MutexGuard<'_, Option<Storage>>`.  `set_len` takes `&mut self` (the
`Mem` variant resizes its `Vec`; every caller already holds
`&mut Storage`).  The two backends:
- **`Disk`** — trivial forwarding to `std::fs::File`.
- **`Mem`** — `Cursor<Vec<u8>>` for `Read/Write/Seek`; `set_len`
  truncates / zero-extends the `Vec`; `len` is `vec.len()`; `sync`
  is a no-op.

A future lazy remote-range backend (its own state, not cheaply
enumerable) would be the point to reconsider `dyn` — until then the
enum is the right call.

### This is the basis for cfitsio's full driver set

cfitsio's drivers are all either **(a) a seekable random-access
store** or **(b) something materialized into a memory buffer**.  The
`Disk` variant *is* (a); the `Mem` variant *is* (b).  Coverage map
(✅ = shipped, ⬜ = sketched):

| cfitsio driver | how it rides on `Storage` | status |
|---|---|---|
| `file://` | the `Disk` variant | ✅ |
| `mem://` / `memkeep://` | the `Mem` (`Cursor<Vec<u8>>`) variant; aliases (same thing in rustfits — see "Python surface"); `from_bytes`/`to_bytes` are the byte I/O pair | ✅ |
| whole-file `.gz` (read) | `Mem` filled by gunzip-on-open | ✅ |
| `.gz` write-back / `.Z` / `.zip` | `Mem` + recompress-on-close (gz); LZW / zip codecs (others) | ⬜ |
| `stdin://` / `stdout://` / `"-"` | `Mem`: slurp the non-seekable reader at open / flush the sink at close | ⬜ |
| `http`/`https` (download) | fill the `Mem` buffer from the network at open (`ureq`); read-only; gunzips a `.gz` URL | ✅ |
| `ftp`/`ftps` (download) | same, via `suppaftp` (shares ureq's rustls); anonymous default, binary mode | ✅ |
| `root` / `gsiftp` (download) | needs a separate protocol crate (XRootD / GridFTP) | ⬜ |
| http **range** reads (remote roadmap #2) | a *lazy* read-only backend: seek+read → Range requests, no full buffer (the case that would justify `dyn`) | ⬜ |
| `shmem://` | `Mem` backed by an OS shared-memory mapping | ⬜ |
| `root://` / `gsiftp://` | same as http — materialize-to-mem, or a lazy protocol backend | ⬜ |

**The one genuine caveat:** non-seekable sources (pipes, streaming
network) cannot be a `Storage` variant directly — FITS needs random
access (HDU offsets, `shift_file_tail`, etc.).  They are buffered
into the `Mem` variant at the boundary (slurp stdin at open, flush
stdout at close).  That is exactly what cfitsio does too, so it's the
standard pattern, not a limitation of the abstraction.

So `Storage` is the **single seam** every non-`file://` driver plugs
into — the argument for having done it once, properly, rather than
bolting each backend on ad hoc.

### Cheap shortcut (superseded for read-from-bytes)

Before the seam landed, the "I already have FITS bytes (DB blob,
socket, astropy) and want to parse them" case had a zero-refactor
shortcut: spill the bytes to a temp file and open that.  **No longer
needed** — `FITS.from_bytes(b)` now parses directly from a
`Cursor<Vec<u8>>`, zero-disk.  The temp-file trick remains the
fallback for the *remote* roadmap (download whole file → open),
where it's the deliberate "download-then-open" design, not a
shortcut.

### Python surface (shipped)

Two layers, chosen so cfitsio/fitsio migrators see familiar names and
everyone else gets a Pythonic path:

- **cfitsio driver names (primary surface).**  `FITS("mem://", "w+")`
  and `FITS("memkeep://", "w+")` both open an empty in-memory file.
  These are the names sophisticated users already know from
  cfitsio/fitsio, and — being `scheme://`-shaped — they slot into the
  *same* constructor prefix-dispatch branch (`is_mem_url` in
  `fits.rs`) the remote roadmap will use for `http://` / `https://`,
  rather than needing a special-cased magic filename.  (An earlier
  sketch proposed a SQLite-style `":memory:"` sentinel; dropped in
  favor of the cfitsio spelling for familiarity and dispatch
  uniformity.)
- **`to_bytes()` / `from_bytes()` (Pythonic pair).**
  `fits.to_bytes()` returns the file's current bytes as a Python
  `bytes` (works on `Mem` *and* `Disk` — for `Disk` it flushes then
  reads the whole file, loading it into RAM; call before `close()`,
  which drops the buffer).  `rustfits.FITS.from_bytes(b, mode='r')`
  parses bytes you already hold (DB blob, socket, astropy) directly
  from a `Cursor<Vec<u8>>` — no disk.  This layer exists because
  cfitsio's *native* extraction is clunky (hand it a buffer pointer +
  size); the byte methods are the discoverable, idiomatic way in and
  out.

**Shipped semantics:**
- `from_bytes` **copies** the input into a private `Vec`, so the
  returned `FITS` is fully independent of the source object; accepts
  `mode='r'` (default) or `'r+'`; rejects `'w+'` (it would discard the
  bytes you just passed — use `FITS("mem://", "w+")` to start empty).
- An empty `mem://` file has zero HDUs (same as a fresh `w+` disk
  file); `to_bytes()` on it returns `b""`.
- **Read-only mode is advisory for mem files.**  A `Cursor<Vec<u8>>`
  has no OS permission layer, so (unlike a `Disk` file opened `"r"`)
  writes to a `mem://`/`from_bytes` file aren't rejected by the OS.
  Harmless — writes only touch the private in-memory copy — but worth
  knowing.  A cross-cutting Rust-level writable gate was deemed out of
  scope (disk relies entirely on OS enforcement today).

**`mem://` and `memkeep://` are aliases — both do the same thing.**  In
C the distinction is load-bearing (the caller manages the buffer
pointer; `memkeep://` means "don't free it on close so I can read it
back").  In rustfits the buffer is a `Vec` owned by the `FITS`
pyclass, alive exactly as long as the Python object, and `to_bytes()`
copies it out regardless of which name opened the file — so the
keep/free semantic doesn't map onto anything the Python user controls.
Both names back the same `Cursor<Vec<u8>>` impl; `to_bytes()` is the
supported extraction path either way.  We accept both purely so a
migrator's existing spelling keeps working.

### Main tradeoff

The whole file lives in RAM for a `mem://` file — which negates
rustfits's "~1 MiB above the output array" RSS property, but that's
inherent to mem files, not a flaw.  (The `Disk` path is unchanged and
keeps the streaming property.)  The enum change touched the
locked-handle type, so the whole `src/` tree recompiled once even
though most sites didn't change semantically — a one-time cost.

**Where the remaining work lives:** the unshipped rows in the coverage
map above (gz / stdin / shmem) are each a new `enum Storage` variant
filled at the open boundary.  The remote `http`/`https` *download*
case is the same (fill `Mem` from the network); the remote **range**
case (remote roadmap #2) is the lazy read-only backend that would
justify switching `Storage` to `dyn` — build it together with the
remote roadmap when a real workload needs partial reads of a remote
multi-GB `.fz`.

## cfitsio extended filename syntax (EFS) — deferred

cfitsio parses a rich mini-language embedded in the filename string
(`fits_open_file` does it in C, so fitsio gets it for free).  rustfits
has **only** implemented the *driver-prefix* subset of EFS — `mem://`,
`http(s)://`, `ftp(s)://`, and the `.gz` suffix (see the storage-driver
and remote roadmaps above).  **Everything else is deferred until a user
asks**, and when it lands it should go in the planned
`rustfits.compat.fitsio` shim (translating the string into core API
calls), NOT in the core `FITS()` constructor — the core deliberately
favors explicit Python (`fits["EVENTS"]`, `hdu[a:b:c]`, `columns=`,
`rows=`) over an embedded string DSL (same reason `compress=Gzip1(...)`
won over cfitsio's `[compress R]` flat-string form).

The deferred surface, by cost (key realization: most of it rustfits
already exposes more cleanly — the genuinely-missing capability is the
Tier 3 row-filter calculator):

- **Tier 1 — trivial, high-value (small parser).**  HDU selection
  `file.fits[3]` / `[EVENTS]` / `[EVENTS,2]` (extname+extver) — by far
  the most-used part; maps to `fits[i]` / pick-by-extname.  `!file.fits`
  clobber prefix on write → `w+`.
- **Tier 2 — moderate (reuse existing machinery).**  Image subsections
  with strides `file.fits[1:512:2, *]` → `__getitem__` slicing; column
  selection `[col TIME,X,Y]` → `read(columns=...)`.
- **Tier 3 — large, each its own subproject.**  Row-filter expressions
  `[EVENTS][TIME>5000 && X<100]` — cfitsio's *calculator*: a full
  lexer/parser/evaluator with arithmetic, booleans, functions (trig,
  regexp, `gtifilter()`, `regfilter()`), column refs.  Plus
  binning/histogramming `[bin X,Y]`, GTI filtering, spatial-region
  filtering, and `[compress R 100,100]` write specifiers.

**Decision:** defer all of it; the value is migration smoothness for
existing fitsio scripts, not new capability.  Before building Tier 3,
scope how many real migrant scripts use the calculator/binning vs. just
`[EXTNAME]` — Tier 1 is a weekend in the shim; Tier 3 is a parser
project worth its own design pass.

## Build / dev workflow

Three sets of dependencies, three install commands, one
build tool:

- **Runtime + build deps (Python side)**:
  `conda install --file conda-requirements.txt` — python,
  maturin, numpy.  No Rust here.
- **Dev + test deps**:
  `conda install --file conda-test-requirements.txt` —
  pytest, fitsio, astropy, ruff.  Add `pytest-cov` if running
  coverage locally.
- **Rust toolchain**: rustup (NOT conda).  Local + CI both
  use rustup.  Don't add `rust` to `conda-requirements.txt` —
  conda's rust packaging interacts badly with PyO3's PyFFI
  linking and is harder to keep in sync with CI.  Install
  rustup via the standard installer:
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.
- **Build**: `maturin develop` builds the cdylib AND installs
  the editable wheel in one step.  Never use bare `cargo build`
  (won't install into the Python env so Python can't import).

### Local iteration loop

Chain compile + tests in one command:

    tools/cargo-test.sh && maturin develop && pytest tests/<focused>.py

- `tools/cargo-test.sh` wraps `cargo test` so the test binary
  (which links libpython via PyO3) can find libpython.X.so at
  runtime.  Bare `cargo test` fails with "libpython3.X.so.1.0:
  cannot open shared object file" because conda's env
  activation doesn't put `$CONDA_PREFIX/lib` on
  `LD_LIBRARY_PATH`.  The wrapper resolves the lib dir via
  `sysconfig.LIBDIR` and prepends it.  See
  `tools/cargo-test.sh` for the one-liner.
- `maturin develop` builds + installs into the conda env.
- `pytest tests/<focused>.py` for the focused suite during
  iteration; chase with full `pytest` before commit.

### Reference sources for byte-exact ports

Several decoders/encoders in `src/zimage/` are direct ports of
cfitsio C source (HCOMPRESS, PLIO, RICE, GZIP framing details,
the Park-Miller dither sequence).  When debugging or extending
these, side-by-side comparison with cfitsio is the fastest path
to correctness.

The reference C source lives in the fitsio repo's bundled
tarball:

1. Clone fitsio (any recent release works; it pins the cfitsio
   version it ships against):
   `git clone https://github.com/esheldon/fitsio  ~/git/fitsio`
2. Untar the bundled `cfitsio-<X>.<Y>.<Z>.tar.gz` next to it:
   `cd ~/git/fitsio && tar xzf cfitsio-*.tar.gz`
3. Reference paths in the rest of this document are written as
   `<cfitsio>/fits_hdecompress.c`, etc.  `<cfitsio>` resolves to
   the untarred directory (e.g. `~/git/fitsio/cfitsio-4.6.4`).

Notable files for ZIMAGE work:

- `<cfitsio>/fits_hcompress.c` / `fits_hdecompress.c` — HCOMPRESS_1
  encode/decode + hsmooth.
- `<cfitsio>/pliocomp.c` — PLIO_1 encode + decode (`pl_p2li` /
  `pl_l2pi`).
- `<cfitsio>/ricecomp.c` — RICE_1 encode/decode.
- `<cfitsio>/imcompress.c` — top-level tile compression dispatch +
  GZIP / fallback-column wiring.
- `<cfitsio>/quantize.c` — float quantization, Park-Miller table,
  dither method selection.
- `<cfitsio>/fitsio2.h` — internal types and constants
  (e.g. `BYTE_IMG = 8`, `SHORT_IMG = 16`, `LONG_IMG = 32`).

Function names in `src/zimage/hcompress.rs` mirror cfitsio's
`fits_hdecompress.c` exactly, so `diff <cfitsio>/fits_hdecompress.c
src/zimage/hcompress.rs` is a useful debugging tool.  The RICE
encoder in `src/zimage/rice.rs::encode_rice` is a structural port
of cfitsio's `fits_rcomp` / `_short` / `_byte` family — the
per-bytepix arithmetic (pdiff truncation, ZigZag, dpsum/psum cast)
matches byte-for-byte, but the encoder is a single function
(`encode_rice`) rather than three separate functions and uses a
clean `BitWriter` rather than cfitsio's `Buffer` struct.  Byte-
exactness is verified by the heap-comparison tests in
`tests/test_image_compressed_write_rice.py`.

### Performance / large fixture files

The Performance TODO section below mentions a 12 GB GZIP_2
benchmark file.  That fixture is per-user (lives on the repo
owner's disk; not committed, not downloadable from CI).  When
profiling on a fresh machine, generate an equivalent: any large
real-world float-image survey file (DECam, HSC, etc.) compressed
with fitsio + `compress='GZIP_2', tile_dims=(<rows>, <cols>)` works.
Alternatively, synthesize:

```python
import numpy as np, fitsio
data = np.random.default_rng(0).standard_normal((30000, 50000),
                                                dtype=np.float64)
fitsio.write('big.fits.fz', data,
             compress='GZIP_2', tile_dims=(100, 100))
```

The point of the benchmark is throughput on long chunked reads;
the file's *content* doesn't matter, just its size and tile count.

### CI

`.github/workflows/ci.yml` runs five jobs:
1. `lint` — `ruff format --check` + `ruff check`.
2. `rust-test` — installs Python via `setup-python` (needed
   so the wrapper can resolve libpython) and runs
   `tools/cargo-test.sh`.
3. `test` matrix — ubuntu × {py3.12, py3.14}, full conda env
   from the requirements files + `maturin develop` + `pytest`.
4. `coverage` — single ubuntu/py3.12 leg, uploads Python +
   Rust coverage to codecov via `cargo-llvm-cov`.

macOS is intentionally absent from the matrix — see "Known
CI limitations" below.

## Testing conventions

For any mutation test, verify the outcome through **both**:

1. **Same-handle read** — through the same `FITS` object the mutation went
   through, without closing.  Exercises the in-memory cards Vec / data view.
2. **Post-reopen read** — close, reopen, read.  Exercises on-disk
   persistence.

Inline two-block pattern is common:

```python
with rustfits.FITS(fname, "r+") as fits:
    fits[0].header.add_comment("hello")
    assert fits[0].header["COMMENT"] == ["hello"]    # same-handle
with rustfits.FITS(fname, "r") as fits:
    assert fits[0].header["COMMENT"] == ["hello"]    # post-reopen
```

Or a `_check_both(fname, fits, predicate)` helper for tests with multiple
assertions.  Either is fine.

### Test file naming (topic-prefix scheme)

`tests/` is a flat directory; files are named `test_<area>_<feature>.py`
so a plain `ls` groups them by area.  `<area>` is one of:

- `image` / `table` — uncompressed image / BINTABLE features.  Sub-clusters
  follow as further words: `compressed`, `vla`, `create`, `read`, `write`,
  `setitem`, `append`, `repack`, etc. — e.g.
  `test_image_compressed_read_rice.py`,
  `test_table_compressed_setitem_vla.py`, `test_table_create_dtypes.py`,
  `test_table_vla_write.py`.  Compressed-HDU tests live under the HDU type
  they belong to (`test_image_compressed_*`, `test_table_compressed_*`),
  and VLA is a table concept (`test_table_vla_*`).
- `header` — `FITSHeader` / `FITSHeaderEdit` features.
- `hdu` — cross-HDU / base-HDU concerns that aren't type-specific
  (`test_hdu_accessors`, `test_hdu_checksum`, `test_hdu_data_size`,
  `test_hdu_units`).
- `fits` — `FITS` container / file level (`test_fits_open`,
  `test_fits_open_multi`, `test_fits_getitem`).
- `convenience` — the top-level `rustfits.read` / `rustfits.write` wrappers
  (`test_convenience_read`, `test_convenience_write`).

Two files are intentionally left unprefixed because they don't belong to a
single area: `test_repr.py` (spans `FITS` + every HDU type) and
`test_healsparse_bitpack_roundtrip.py` (an end-to-end integration test).
When adding a test, name it for the area + feature it exercises, not the
work item that introduced it (no "phaseN" / dev-stage names).

## Known CI limitations

**macOS dropped from the test matrix (May 2026).**  The CI
workflow at `.github/workflows/ci.yml` originally ran
`{ubuntu-latest, macos-latest} x {python 3.12, 3.14}`.  Every
macOS leg crashed during the first test that asks fitsio to
write a compressed-image fixture: libmalloc detected a bad
free inside cfitsio's `ffbinit`, called from
`PyFITSObject_create_image_hdu`.  Both py3.12 and py3.14
hit it, so it's macOS-specific (conda-forge build of fitsio
or cfitsio), not Python-version specific.  Linux is
unaffected.

The repo owner is upstream on fitsio and plans to fix the
macOS build there.  Once a fixed fitsio is released on
conda-forge, re-add `macos-latest` to the `test.matrix.os`
list in `.github/workflows/ci.yml` and delete this note.
Worth keeping an eye on the failing test
(`tests/test_image_compressed_accessors.py`'s
`test_other_compression_types_dispatched` was the first to
abort) when the time comes.

## Documentation TODO — tutorial gap audit

The Sphinx tutorial under `docs/tutorial/` covers the main
surface but a 2026-05 audit identified gaps users would expect.
Cross items off as they land; revisit ordering when a real user
request flags something.

1. ✅ **File modes table** — `"r"` / `"r+"` / `"w+"` in
   ``quickstart.rst`` covering read-only vs read-write,
   truncates vs preserves, creates vs requires-exists.
2. ✅ **Walking HDUs / picking the right one** — section in
   ``quickstart.rst`` showing the realistic ``hdu.has_data +
   isinstance(hdu, ImageHDU)`` pattern.
3. ✅ **`AsciiTableHDU` note** — section in ``tables.rst``
   documenting the read-stub state and pointing at astropy as
   the fallback.
4. ✅ **Known limitations** — new ``limitations.rst`` page
   listing gaps tagged ``(not yet)`` vs ``(by design)`` with
   workarounds, plus cross-tool interop caveats.
5. ✅ **Cross-tool interop** — one-sentence callout at the top
   of each topic page asserting bit-exact round-trip with
   astropy and fitsio.
6. ✅ **Subset object semantics** — "How subsets relate to
   the parent table" subsection in ``tables.rst`` covering
   lazy selectors, fresh-read semantics, and parent-handle
   lifetime.
7. ✅ **Error/recovery model** — new ``errors.rst`` page
   covering the standard-Python-exceptions choice, what raises
   what, the taint flag, recovery via close+reopen, and the
   in-process-mutex / no-OS-lock multi-writer story.
8. **Performance / chunked reads** — the big-chunk and small-
   chunk wins vs fitsio (see
   [docs/internal/perf-history.md](docs/internal/perf-history.md))
   are unmatched and currently undocumented user-facing.  Worth a
   short page if we want users to know; deferred because benchmark
   numbers age fast and invite arguments about methodology.
9. ✅ **Migration guide** — ``docs/tutorial/migration.rst``,
   wired into the toctree between ``headers.rst`` and
   ``errors.rst``.  Frames rustfits as the modern successor to
   fitsio (matches the [STRATEGY.md](STRATEGY.md) freeze-and-
   shim plan), walks through the behavior differences (headers,
   compression kwargs), shows side-by-side porting recipes for
   the common patterns (open+read, table subset, write image /
   table to fresh file, compressed image, lossy float
   compression, read header only), lists what rustfits doesn't
   have (``vstorage=fixed``, ``case_sensitive=True``, ASCII
   table writes), and lists what fitsio doesn't have that
   rustfits does (full ``__getitem__`` / ``__setitem__`` on
   every HDU, ZTABLE writes, faster compressed reads).
   Includes a short "from astropy.io.fits" section for users
   coming from there instead.

Out of scope (don't write these without a specific ask):
migration guide from astropy / fitsio; WCS handling (rustfits
doesn't parse WCS — users hand the header to `astropy.wcs`).

## Coverage TODO — sweep once feature-complete

First codecov run (commit `cf619a2`, May 2026) reported
`/rustfits` (Python) at 100% and `/src` (Rust) at 89.61%.
The Python number is mostly informational — `rustfits/*.py`
is just re-exports plus a thin `convenience.read`, so all
the real logic lives in Rust and the 89.61% is the one
worth moving.

Plan: **don't chase coverage incrementally** during
feature work.  Wait until we're past the feature-complete
threshold for the current phase set (probably some time
after ZIMAGE Phase 5+6 and image/table write parity is
done), then do one focused coverage-driven sweep.  Two
buckets to look at when that happens:

- **Cheap wins (the "dishonest" miss bucket)** — branches
  that one extra test would exercise (a specific TFORM
  case, an edge tile in a 1-D image, a `__setitem__`
  permutation we never hit).  A 30-min pass usually moves
  the library 90% → ~95%.
- **Honest gaps (the "fault injection" bucket)** — mid-
  write `flush()` failures, OS-level I/O errors, taint-
  flag re-rejections, malformed-file branches.  These need
  either real fault injection (LD_PRELOAD-style write
  intercept, or a faulty-file-system test harness) or
  hand-crafted broken FITS fixtures.  Some of these
  branches *should* stay uncovered — they're the panic-on-
  impossible variants that exist for safety — so the
  judgment is which ones are worth the test cost.

Don't aim for 100% on `/src`.  90-95% covered + the
remaining gaps explicitly catalogued (in code comments or
this section) is the actual target.

## Cleanup / refactoring TODO

Code-organization backlog (logged 2026-05-28).  These are
janitorial, not feature work — none changes behavior.  **Suggested
ordering: do the structural splits (1 + 2) first, then the cleanup
passes (3 + 4) on the settled structure so you don't clean code
you're about to move; 5 is independent and can happen anytime.**
Items 1–5 all shipped 2026-05-28; this backlog is complete.

1. ✅ **Split the two giant `.rs` files into directory modules.**
   Done 2026-05-28.  `src/hdu_table_compressed.rs` (5970 lines) →
   `src/hdu_table_compressed/` (10 files) and
   `src/hdu_image_compressed.rs` (5364 lines) →
   `src/hdu_image_compressed/` (7 files), mirroring the
   `src/hdu_table/` pattern: single-responsibility files behind a
   `mod.rs` that re-exports only the external surface, visibility
   tightened to `pub(crate)` only where a sibling imports.  Pure code
   move (full pytest + Rust unit tests green).  See the two
   directory-module entries under "Project structure".
   `src/zimage/hcompress.rs` (~2440) was left intact — a faithful
   line-by-line port of cfitsio's `fits_hdecompress.c`/`fits_hcompress.c`
   whose function-name parity with cfitsio is a debugging asset.

2. ✅ **Extract compression code shared by the image + table compressed
   HDUs.**  Done 2026-05-28 alongside #1.  Outcome was deliberately
   small: the codecs were already shared via `src/zimage/`
   (gzip / rice / hcompress / plio) and `src/cache.rs`'s
   `BytesBoundLruCache`, so the only genuine cross-file duplication
   left was the tile-cache default, now in `src/zimage/tile_io.rs`
   (`DEFAULT_TILE_CACHE_BYTES`).  The per-tile encode/decode
   **dispatch** was intentionally NOT unified: the image path is
   2-D-tile + quant-context shaped, the table path is
   per-column-slab shaped, and one abstraction would fit neither
   cleanly — documented in `tile_io.rs`.  (P/Q descriptor I/O also
   stayed split: the image side uses buffer-based codecs, the table
   side reuses the uncompressed `hdu_table` descriptor helpers; a
   forced merge wasn't worth the surface churn.)

3. ✅ **Audit for gratuitous `pub(crate)`.**  Done 2026-05-28.
   Demoted every top-level item in the two compressed directory
   modules, then let the compiler re-bump only what a sibling or
   external module actually imports — 8 file-internal helpers went
   back to private (`default_table_algorithm`, `encode_tile_int`/
   `_float`, `TileRow`, `IntTileCtx`, `FloatTileCtx`, `read_descriptor`,
   `encode_vla_column_tile_with_merge`).  Struct fields on the
   cross-module data-carrier structs were left `pub(crate)` (rustc
   has no over-visible-field lint and they genuinely cross modules).

4. ✅ **Clippy: fix or consciously `#[allow]`.**  Done 2026-05-28.
   ~141 lib warnings → zero.  Auto-fixed the bulk via `cargo clippy
   --fix --lib`; hand-fixed the non-machine-applicable remainder
   (`checked_div`, `strip_prefix`, `nonminimal_bool` factoring,
   `&mut Vec` → `&mut [_]`, `vec![]` init, `enumerate`/`zip` loop
   rewrites only where provably equivalent, type aliases for the
   meta-cache fields).  `#[allow]` with rationale for the intentional
   cases: per-function `too_many_arguments` on the wide dispatchers
   (matching the codebase's existing convention — 44 such allows
   predated this pass), module-level `needless_range_loop` on the
   cfitsio-port files (`hcompress`/`plio`/`quantize`/`checksum` —
   index loops mirror the C source), `upper_case_acronyms` on the
   public `FITS`/`HDU` pyclass names, and `dead_code` on the
   test-only `decode_checksum_ascii`.  CI does not yet gate on
   clippy; re-running `cargo clippy --lib` should stay at zero.

5. ✅ **Reorganize tests by feature, not dev phase.**  Done
   2026-05-28 in two steps: the phase-numbered suites were first renamed
   to feature names, then the whole suite was renamed to a **topic-prefix
   scheme** so a flat `ls tests/` groups by area.  See the test-naming
   convention under "Testing conventions" for the final scheme.  All
   `git mv` (history preserved); module-docstring "Phase N" summary tokens
   dropped; every filename reference in the roadmaps below was swept to
   the new names (the "Phase N" *narrative* there is kept as dev history).
   Subdirectory layout was considered and declined — flat + descriptive
   names was preferred for `ls`-browsability.

## Performance summary

rustfits is **as fast or faster than fitsio on every benchmark
in `perf/`** (release-mode; vs fitsio, which is the Python
wrapper around cfitsio).  Headline wins:

| Path | Speedup vs fitsio |
|---|---|
| GZIP_2 1-D compressed read (big chunks) | 3.2× faster |
| GZIP_2 1-D compressed read (small chunks) | 40× faster |
| GZIP_2 encode (matched cfitsio level 1) | 2.32× faster |
| RICE decode (2-D, post-rewrite) | 1.11–1.94× faster |
| Uncompressed image read (post byteswap fix) | 1.0–2.4× faster |
| Uncompressed image write (post byteswap fix) | 1.40× faster |
| Uncompressed table read (post chunked VLA reader) | 1.23–2.56× faster |
| Uncompressed table write | 2.60× faster |
| Uncompressed table append (fixed, vs fitsio append) | 2.7–3.1× faster |
| Uncompressed table append (VLA, vs fitsio append) | **200–210× faster** (fitsio per-call HDU close/reopen) |
| Image extend (peak RSS) | up to 16× less RAM than write-once |
| Table append (VLA, peak RSS) | 1.3× less RAM than write-once |
| 2-D image extend, uncompressed (vs fitsio extend) | ~2× faster, ~5× less RAM than write-once |
| 2-D image extend, GZIP_2 multi-tile chunks | 1.5× write-once, 1.8× less RAM |

ZTABLE (compressed BINTABLE) is a rustfits self-comparison —
fitsio's Python API can't decompress it.

**Bench suite.**  Scripts under `perf/` (run directly, NOT
pytest tests — the `perf-` filename prefix keeps them out of
collection).  Shared methodology in `perf/_harness.py` (see
[[feedback_perf_test_methodology]]): release build required;
fresh open per timed iter; warmup primes FS cache; fresh FILE
per write iter; median of 5.  `perf-all.py` is the runner;
results land in `docs/tutorial/performance.rst` via the
auto-generated `_perf_tables.rst`.

**Key reusable lessons** (each documented in detail in the
deep doc):
- `OPENBLAS_NUM_THREADS=1` (+ MKL/OMP/NUMEXPR) — without it,
  numpy's idle worker threads dominate every flamegraph.
- Debug builds read ~7× slower than release — always
  `maturin develop --release` before profiling.
- Fresh FILE per write iter (not just fresh handle) — kernel
  page-cache penalty on same-file overwrite was masking real
  perf.
- 1 MiB chunks for any streaming I/O ([[feedback_chunk_size_convention]]).

**Open perf work** lives in the "Performance TODO" section
below — coverage gaps, the one fix item (compressed-image
checksum still materializes full ndarray), etc.

**For the full optimization narrative — every fix
commit-by-commit, every debugging gotcha, the RICE decode
rewrite's three-fix sequence, the header-derived metadata
cache phase history, the methodology pitfalls (warm-vs-cold
file cache, OpenBLAS threads, debug-vs-release), the per-
algorithm encode/decode wins, and the 12 GB GPFS fixture
recipe — see
[docs/internal/perf-history.md](docs/internal/perf-history.md).**

## Performance TODO

Open perf investigations not yet measured.  Add to this list (with
a one-line "why we suspect" and a "how to check") when something
worth checking comes up; cross items off as they ship.

1. ✅ **Opening a file with many HDUs.**  Done 2026-05-30.
   `perf/perf-fits-open-many-hdus.py` sweeps N ∈ {100, 1000, 10000},
   builds a fixture with N minimal image HDUs (1-byte u1), and
   times fresh open per iter in two regimes.

   **Result: both rustfits and fitsio scale linearly** (per-HDU
   time flat across the 100× range in N — rustfits 3.6–4.1 µs/HDU,
   fitsio 5.7–6.1 µs/HDU on the apples-to-apples walk).  No
   quadratic-on-open bug.  On the apples-to-apples comparison
   (rustfits eager parse vs fitsio open + `update_hdu_list()`),
   **rustfits is ~1.6× FASTER** at every N.

   Two regimes per N because the constructors have different
   contracts: rustfits is EAGER (`parse_hdus_from_file` runs at
   open), fitsio is LAZY at construction (no HDU list built until
   first access — at which point fitsio also walks every HDU, not
   just the requested one).  The "bare open" row therefore shows
   rustfits "slower" — it's honestly doing real work fitsio is
   deferring; the "open + walk all HDUs" row is the truthful
   comparison and the one that would show a quadratic bug if
   there was one.

2. **Discuss: should rustfits be lazy on open like fitsio?**  The
   bench above (item 1) showed rustfits's eager `parse_hdus_from_file`
   is fast on a local SSD — ~4 µs/HDU, so even a 10 k-HDU file opens
   in ~40 ms.  But the cost scales linearly with N, and on slow /
   high-latency filesystems (GPFS / Lustre / network-mounted
   archives) where every seek is a network round trip, that
   ~4 µs/HDU could explode to milliseconds/HDU — a 10 k-HDU file
   would take seconds to open even before the user has touched any
   HDU.

   The benefit of fitsio's model is NOT "skip the HDUs you don't
   read" — once you touch ANY HDU it walks them all (`fits[0]`,
   `__repr__`, `len(fits)` after touch all trigger
   `update_hdu_list()`).  The benefit is purely REPL ergonomics:
   `f = fitsio.FITS(fname)` returns immediately so the prompt comes
   back fast, even on a slow filesystem.  Total parse cost is the
   same — just shifted from the constructor to the first access.

   Reasons to stay eager (what the codebase assumes today):
   simpler invariants (`fits[i]` is O(1), never raises a parse
   error after construction); `len(fits)` works immediately;
   the shared `Arc<FileLayout>` model in `common.rs` relies on
   knowing every HDU's offset up front so cross-HDU growth (header
   grow, image extend, table append) can atomically bump later
   offsets — every mutation site would need a "walk before shift"
   guard, which is a wide cross-cutting surface easy to miss.

   **Scope cut: lazy is read-only only.**  Reject `lazy=True` with
   `mode != "r"` in the constructor and the entire mutation guard
   surface disappears (mutation pymethods are unreachable when
   lazy=True).  This collapses the change from "audit every
   shift_file_tail_and_update_offsets caller" to "wire one new
   cursor into the read-side getters."  Read-only is also the
   workload where slow-filesystem latency actually matters — write
   workflows tend to be local-scratch.

   **Three options — NOT mutually exclusive.**  These can ship
   independently and a future state could combine multiple
   (e.g. C provides fitsio-feel for the REPL while A provides the
   top-level batch-script form; both address different use cases
   and don't conflict).

   **Option A — top-level `rustfits.read(fname, ext=N, lazy=True)`.**
   Smallest viable form for the batch case.  Bypasses the `FITS`
   pyclass entirely: open the file, walk HDUs 0..N-1 parsing just
   enough header to compute each one's data extent + advance the
   cursor, then call the existing per-HDU read code on the matched
   offset.  Returns the array (and optionally header, like the
   existing `rustfits.read`).  No warm handle, no `FITS` object
   lifecycle, no offset cache — two calls = two walks.  Pros: tiny
   code change (~150 lines), zero invariant changes, ships
   immediate value for batch scripts on archive filesystems (the
   dominant slow-FS case).  Cons: doesn't help the REPL / multi-HDU
   workflow.

   **Option B — `FITS(fname, "r", lazy=True)` per-HDU cursor.**
   Warm-handle form with genuine pay-per-HDU.  Constructor parses
   only HDU 0; subsequent `fits[i]` triggers
   `ensure_parsed_through(i)` which walks the cursor forward,
   pushing new `HduOffsets` as it goes.  `len(fits)` and
   `__iter__` trigger a full walk (forced eager).  `__repr__`
   shows only the parsed HDUs plus a "(lazy; N additional HDUs
   not yet inspected)" banner.  Pros: matches fitsio's REPL
   ergonomics PLUS genuine pay-per-HDU for partial-walk patterns
   (`fits[0]`, `fits[1]` = 2 HDUs parsed, not all N).  Cons:
   `FileLayout` grows a parse cursor + `complete: AtomicBool`;
   ~100-200 lines including repr tweak; lazy mode adds one new
   user-visible kwarg with its own test surface.

   **Option C — `FITS(fname, "r", lazy=True)` defer-the-full-walk
   (fitsio's model).**  Smallest of the three, ~50-100 lines.
   Constructor skips `parse_hdus_from_file` entirely; the FIRST
   access to anything that needs offsets (`fits[i]`, `__len__`,
   `__iter__`, `__repr__`) triggers the walk all at once,
   identical to today's eager parse just deferred.  After first
   touch the FileLayout is identical to today's at-construction
   state — every subsequent access is O(1).  Mechanics:
   `FileLayout` gets a `walked: AtomicBool`; a small
   `ensure_walked()` helper runs the walk under a mutex + Acquire
   load, idempotent on subsequent calls.  A handful of `FITS`
   pymethods that touch offsets prepend `self.ensure_walked()?;`.
   HDU pymethods don't change (once you have an HDU object via
   `fits[i]`, the walk already happened).  Pros: smallest LOC of
   the three; no shared refactor needed; total parse cost
   unchanged (just deferred) so steady-state behavior is
   identical.  Cons: only buys REPL feel — first access still
   walks every HDU; strictly weaker than B for partial-walk
   patterns.

   **Default vs opt-in for B and C.**  fitsio's lazy IS the
   default — there's no opt-in kwarg.  Two routes for rustfits:
   - **`lazy=True` opt-in kwarg (default eager).**  Backwards
     compatible: existing user code keeps today's "construct =
     parse" semantics including the error timing (corrupt-HDU
     errors surface at the `FITS(fname)` call).  Explicit signal
     that the user is opting into the deferred model.  This is
     the safer default for a library that already shipped with
     eager semantics.
   - **Lazy by default (matching fitsio).**  Migration-friendly;
     no kwarg discovery burden.  But shifts error timing —
     corrupt HDU 17 fails at `fits[i]` or `len()` instead of
     `FITS(fname)`.  Existing test code that does `try:
     FITS(fname) except IOError:` would need to move the except
     to the first access.

   Recommendation: start with `lazy=True` opt-in to avoid
   silently changing error semantics for existing users.  Flip
   the default later if real-world usage shows the opt-in form
   is awkward.

   **Shared enabling refactor (only needed for A + B; C doesn't
   need it; ~50 lines).**  Pull the per-HDU walk step (parse
   header at offset, compute data extent, advance cursor) out of
   `parse_hdus_from_file` into a stand-alone iterator/primitive.
   Then today's `parse_hdus_from_file` consumes it to completion,
   Option A drives it until match, Option B holds it and consumes
   on demand.  No code duplication between A and B.  Option C
   reuses `parse_hdus_from_file` verbatim under the
   `ensure_walked()` guard — no refactor needed.

   **Composition.**  The realistic shipping shape is probably
   **C + A**: C gives fitsio-feel for interactive / REPL use of
   the `FITS` handle (cheapest to build, matches the model
   migrating users already understand); A gives the genuine
   one-shot read-this-HDU shortcut that bypasses the handle
   entirely.  B is the most powerful but also the most code;
   defer until C + A are in the wild and there's evidence the
   warm-handle-with-partial-walk pattern matters.

   **How to validate.**  Bench `perf-fits-open-many-hdus.py` on
   a GPFS mount (the user's HPC has access) to quantify the
   per-seek penalty.  If a 1000-HDU file's eager open is e.g.
   1 s on GPFS vs 4 ms on SSD, the case for shipping at least
   C (and maybe A) is concrete.  The companion
   `perf-fits-open-one.py` script times a single open against a
   pre-existing path (build a fixture today with `PERF_KEEP=1`,
   come back days later when the FS metadata cache has actually
   evicted, time the cold open) — see its docstring for the
   archive-cold workflow.

   **Bench findings (2026-05-30) — shelved pending user
   complaint.**  Ran `perf-fits-open-many-hdus.py` on local SSD,
   a local HDD `/tmp`, and a GPFS mount; also ran with `--cold`
   (no warmup, worst-sample) and against `vmtouch -e`-evicted
   fixtures on GPFS.

   - **rustfits scales linearly on every filesystem tested**
     (per-HDU time flat across 100×–10000× range in N): 3.6–4.1
     µs/HDU on SSD, 10–11 µs/HDU on HDD /tmp, 11–14 µs/HDU on
     GPFS.  No quadratic-on-open bug.
   - **rustfits is 1.2×–1.6× FASTER than fitsio** on the
     apples-to-apples walk across every regime.
   - **Eager-open absolute cost on GPFS** (warm + OS-evicted
     are equivalent here): 1.3 ms at N=100, 11 ms at N=1000,
     111–148 ms at N=10000.  Tolerable for any interactive use.
   - **OS page cache eviction (`vmtouch -e`) had no effect** —
     the GPFS pagepool is the dominant cache layer and isn't
     reachable from userspace.  We could not measure a
     *truly* archive-cold open (file untouched for days +
     pagepool evicted under memory pressure); that's the regime
     that would make lazy a clear win, but we lack a test path
     to it without waiting days or finding a real archive
     fixture.
   - **Verdict**: no smoking gun.  Within everything we can
     measure, eager open is fast on GPFS.  Shelved pending
     either (a) a real user complaint about slow open on a
     cold-archive file, or (b) future opportunity to time a
     genuinely cold archive fixture (multi-week-old multi-HDU
     survey file) — the unmeasured regime that could still
     surprise us.  When either of those lands, the design space
     above (A / B / C, plus the shared iterator refactor) is
     ready to pick from.

3. ✅ **Table append (uncompressed + compressed).**  Done.
   `perf/perf-table-append.py` covers all four variants
   {uncompressed, ZTABLE} × {fixed-only, with VLA}, building
   N rows by N/K calls to `append(K rows)` vs one `write_table`
   of all N.  Each build runs in a subprocess for a clean
   `ru_maxrss`; fitsio write-once appears as a reference on the
   uncompressed variants only (no fitsio ZTABLE writer).

   **Findings (N=100 k, chunks 1000/10000; default ZTILELEN
   ≈ 16 k rows):**
   - Uncompressed fixed-only: 1.03×–1.27× slower than write-once
     (essentially flat at chunk=10 k).  Linear scaling, RSS
     identical to write-once.
   - Uncompressed VLA: 0.93×–1.19× — chunk=10 k is actually
     *faster* than write-once, AND peak RSS drops from 216 MB
     (write-once) to ~166 MB (append, either chunk size) =
     **1.3× less peak RSS**.  The write-once path plans the
     whole heap layout in RAM up front; the append loop pays it
     incrementally.  The real bounded-memory win on the table
     side.
   - ZTABLE fixed-only: **14.3×–15.4× SLOWER** — the
     merge-into-partial-last-tile re-encode dominates when
     chunks << ZTILELEN.  Both chunk sizes (1 k and 10 k) are
     below ZTILELEN so every append touches the trailing tile.
     RSS comparable to write-once.  Surfaced as a follow-up
     item (#10 below).
   - ZTABLE VLA: 6.3×–6.7× slower — same merge-tile cost in
     absolute terms, but the write-once baseline is 3× the
     fixed-only case so the relative ratio is softer.

   Documented in `docs/tutorial/performance.rst` under
   "Incremental table builds".

4. ✅ **2-D image extend (uncompressed + compressed).**  Done.
   `perf/perf-image-extend-2d.py` and
   `perf/perf-compressed-image-extend-2d.py` cover the mosaic /
   strip-build pattern on a (20,000 × 4,000) f4 image.

   **Findings (uncompressed):** rustfits extend matches rustfits
   write-once on time (~117 ms either way) and achieves a ~5×
   peak-RSS win at chunk=100 rows (70 MB chunk vs 363 MB
   whole-array).  fitsio CAN extend uncompressed images via
   `f[0].write(strip, start=(row, 0))`, and fitsio extend gets
   the same bounded-memory benefit; rustfits extend is ~2×
   faster than fitsio extend (same ratio as write-once, so it's
   general per-call overhead in fitsio).  fitsio also holds
   ~80 MB more RAM during write-once (440 vs 363 MB) — likely
   an extra byteswapped copy rustfits avoids.

   **Findings (compressed, GZIP_2, tile=(100, cols)):**
   - multi-tile chunks (10 tiles per call): 1.5× write-once, 1.8× less RAM
   - exact-tile chunks (1 tile per call): 7.6× write-once
   - sub-tile chunks (½ tile per call): **21.6× write-once** —
     same partial-last-tile re-encode pattern as the ZTABLE
     small-chunk append finding.  For mosaic builds align chunks
     to a multiple of tile-rows to avoid the re-encode tax.
     Tracked in TODO #10.

   **Bench methodology note:** `_data.des_array` was rewritten to
   generate in row-strips because the naive
   `rng.standard_normal((R, C))` + `rng.random((R, C)) < zero_frac`
   pattern allocates two full-image temporaries before freeing
   them, inflating peak RSS by ~2× for the write-once row and
   making the bounded-memory win look better than it is.  After
   the fix, peak RSS reflects "the user's array + FITS handle
   overhead", not "the user's array + numpy data-gen churn".

   fitsio cannot extend compressed images
   (`fits status = 107: tried to move past end of file`), so the
   compressed bench is rustfits-only on the extend rows.

   Documented in `docs/tutorial/performance.rst` under
   "2-D image extend".

5. **`repack()` timing on a large heap.**  Both
   `TableHDU.repack()` (VLA orphan reclaim) and
   `CompressedImageHDU.repack()` / `CompressedTableHDU.repack()`
   ship as streaming + staging implementations that should be
   bounded-memory, but neither has been timed against a large
   heap.  How to check: build a fixture with K cycles of
   `(write_vla / __setitem__) → repack` to grow heap orphans,
   then time the repack pass at increasing heap sizes (10 MB,
   100 MB, 1 GB).  Verify peak RSS stays bounded.  Bench
   targets: `perf/perf-table-repack.py`,
   `perf/perf-compressed-image-repack.py`,
   `perf/perf-table-compressed-repack.py`.

6. ✅ **Compressed-image `__setitem__` cost.**  Done 2026-05-30.
   `perf/perf-compressed-image-setitem.py` sweeps 4 selections
   {single pixel, 1-tile-aligned, 4-tile-spanning, 16-tile-
   aligned} × 8 algorithms {uncompressed, GZIP_1, GZIP_2,
   RICE_1, HCOMPRESS_1, PLIO_1, GZIP_1 unquantized-f4, GZIP_1
   quantized-f4} on a (256, 256) image with (32, 32) tiles.

   **Findings**:
   - **Per-call cost is dominated by per-tile re-encode**, as
     expected.  Single-pixel writes cost 19–197 µs depending on
     algorithm vs ~3 µs for uncompressed (the memcpy floor).
   - **PLIO_1 is fastest** at ~19 µs single-pixel
     (run-length encoding makes decode + re-encode trivial);
     **quantized f4 is slowest** at ~197 µs (re-quantization
     against the existing per-tile bscale/bzero/seed dominates).
     RICE_1 and HCOMPRESS_1 are 40–146 µs; the GZIP variants
     are 80–104 µs.
   - **Full-tile-aligned writes are CHEAPER than single-pixel
     writes** (20–50 µs vs 40–200 µs across all algos): the
     aligned full-tile write skips the decode step because
     every pixel is being replaced — read-modify-write
     collapses to just write.
   - **Per-pixel rate amortizes with selection size.**  16-tile
     batched writes drop to <0.1 µs/pixel — comparable to
     uncompressed.

   **Practical guidance**: budget 50–200 µs per single-pixel
   touch on a compressed image (algorithm-dependent), so up to
   a few thousand single-pixel patches per second.  For bulk
   masking, align patches to whole tiles where possible.

   **Cross-tool dropped** (fitsio's `write(start=)` API can
   patch compressed images, and works fine standalone, but
   running it across an algorithm sweep in the same Python
   process triggers a cfitsio `free(): invalid next size`
   memory corruption — same shape as the macOS ffbinit issue
   noted under "Known CI limitations".  Subprocess isolation
   would work around it but adds complexity for limited
   value — the rustfits-self per-tile cost is the headline
   number users need).

   Documented in `docs/tutorial/performance.rst` under
   "Compressed-image `__setitem__` — per-tile re-encode tax".

7. ✅ **Fix: compressed-image checksum still materializes the
   full ndarray.**  Done.  Replaced `read_uncompressed_image_be_bytes`
   with `stream_uncompressed_image_be_checksum` in
   `hdu_image_compressed/checksum.rs`: walks the tile grid in
   tile-stripes (outer N-1 axes), decodes the G_last tiles per
   stripe, then emits image-rows in numpy-row-major by
   interleaving row segments from each tile in the stripe
   (FITS checksum is order-sensitive so tiles in the same
   tile-row have to be live simultaneously).  Peak working set
   = one tile-stripe of decoded bytes + one image-row scanline
   buffer; for typical FITS tile choices (per-row strips or
   sub-MB tiles) lands in the 1–10 MB range, comparable to the
   codebase's 1 MiB streaming-chunk convention.

   **Measured wins** on a (10 000 × 4 000) f4 GZIP_2 image
   (160 MB raw, 135 MB compressed): `add_checksum` peak rose
   only +38 MB (52 MB total) above baseline.  Old version
   would have allocated +320 MB (decoded ndarray + BE-byte
   buffer + numpy intermediates).  ~8× less RAM here, and the
   win scales with image size — a (100 k × 4 k) image would
   have gone from ~13 GB old to ~50 MB now.

8. **Random/scattered access on compressed 1-D.**  Covered for
   2-D RICE (stamps in the DES bench); no equivalent for 1-D
   GZIP_2.  Tile cache behavior may differ when stamps land on
   1-D strip tiles.  How to check: read 1000 random 1k-row
   chunks from a 50M-row GZIP_2 file; compare rustfits vs
   fitsio; vary the tile cache size to expose the cache-hit-
   rate dependency.  Bench target:
   `perf/perf-compressed-image-read-1d-scattered.py`.

9. **Remote read perf (http/https/ftp/ftps).**  Shipped but not
   benched.  Network IO dominates total time so the comparison
   is probably uninteresting until / unless a user asks.  How
   to check: serve a fixture via local `http.server`, time
   `rustfits.FITS("http://localhost/...")` end-to-end; compare
   to local-file open of the same file to isolate the
   download-then-open overhead from the parse cost.  Lowest
   priority of this batch; only worth doing if remote-read
   usage comes up.

10. **ZTABLE small-chunk append: partial-last-tile re-encode
    cost.**  Surfaced by perf-table-append.py (#3 above): with
    `chunk << ZTILELEN` (default ZTILELEN ≈ 10 MB / row_width,
    which is ~16 k rows for the test catalog) every append
    decompresses + merges into the same partial trailing tile
    and re-encodes it.  Measured 14× slower than write-once on
    the fixed-only smoke run (N=20 k, chunks 500/5000 — every
    chunk touched the trailing tile).  Real-world streaming
    pipelines (per-frame source extraction, per-file harvest)
    tend to deliver small batches, so this is the typical use
    case, not a corner.  Possible optimizations: (a) buffer
    in-RAM until the partial tile fills, then encode once;
    (b) skip the re-decode by keeping the trailing tile's raw
    bytes in a per-HDU "pending tile" buffer and growing it in
    place.  (a) is simpler but adds a user-visible "what if I
    close without flushing" question; (b) is invisible to the
    user but more invasive.  Decide approach after measuring on
    a realistic streaming workload (e.g. simulate
    per-detector-frame catalog writes).  Could pair with #4
    (2-D image extend) since the same pattern applies to
    compressed-image extend.

**Out of scope of this list but mentioned elsewhere in CLAUDE.md:**
- Write-side header-meta cache extension (the read-side cache
  Phases 1–5 are shipped; write paths still re-parse — deferred
  until measured to be hot).  See "Header-derived metadata
  caching" below.
- Byte-exact heap agreement with cfitsio on quantized floats —
  correctness/parity item, not perf.  Decoded values are
  bit-exact; raw heap bytes differ by qsort tie-breaking.  Not
  worth fixing absent a specific need.

**Status: all five phases shipped.  Some write-side paths still
re-parse; deferred until measured to be hot.**

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation: `cards_version` + `CardsWriteGuard` | ✅ Shipped |
| 2 | `CompressedImageHDU` cache (read / slice / accessors) | ✅ Shipped |
| 3 | `TableHDU` cache (read paths + accessors) | ✅ Shipped |
| 4 | `CompressedTableHDU` cache (setitem + accessors) | ✅ Shipped |
| 5 | `ImageHDU` cache (accessors + read paths) | ✅ Shipped |

**Measured outcome (Phase 2):** **40× faster than fitsio on
small-chunk compressed reads** (up from 24× before the cache),
on the 100k chunks × 1k rows benchmark.  The cache eliminates
~40% of per-call overhead by collapsing 8+ linear card scans per
slice call to one Mutex lock + Acquire load + Arc clone.  See
[docs/internal/perf-history.md](docs/internal/perf-history.md)
for the full fix-by-fix breakdown.

Other phases (3-5) were not benchmarked but follow the identical
shape — the win scales with how much parsing happened per call.
`TableHDU` likely benefits substantially on per-row workflows
because `parse_columns` does ~7 linear scans × N_columns.

**Why.**  After the four perf fixes above, the remaining ~15-20%
of slice-path time on small-chunks compressed reads is per-
`__getitem__` re-parsing of the same header fields:
`parse_compressed_image_shape`, `parse_string_keyword("ZCMPTYPE")`,
`parse_rice_params`, `parse_tile_shape`, `parse_hcompress_smooth`,
`find_data_columns`, `build_quant_context`, `parse_bscale_bzero`.
That's 8+ O(N_cards) linear scans per call for what is fixed
per-HDU metadata.  Profile shows `parse_string_keyword` at
17.81% of total on big-chunks, similar fraction on small-chunks.

The pattern recurs in EVERY HDU type, not just compressed images
— see the "Pattern recurrence" subsection below.

**Why we built it without caching originally.**  The "FITSHeader:
cards are the single source of truth" rule (see that section
above) deliberately rejects parsed-value caches on the public
`FITSHeader` API to avoid sync drift.  That rule is correct for
single-key lookups (`header["KEY"]`) where parse cost is
invisible.  It DOESN'T account for derived structures (table
schema, compressed-image tile layout, etc.) used in tight
public-method loops, where parse-derive cost is orders of
magnitude bigger than a single key lookup.

The proposed cache is for **derived structures**, NOT individual
keys.  The cards Vec stays the source of truth; the cache is a
memoization keyed by a version counter that bumps on any
mutation.  This is structurally simpler than the per-key cache
the FITSHeader rule warns about — invalidation is "have the
cards mutated since the last fill?" not per-field.

**Design.**

```rust
// On the base HDU: a version counter that bumps every time
// cards are mutated.  Lives in `Arc<AtomicU64>` so all HDU views
// of the file share the same counter.
struct HDU {
    cards_version: Arc<AtomicU64>,
    // ... existing fields
}

// Per-HDU-type meta struct holding everything currently
// re-parsed every __getitem__.  Wrapped in Arc so cache hits
// return a cheap clone, not a deep copy.
struct CompressedImageMeta {
    zbitpix: i32,
    image_shape: Vec<u64>,
    tile_shape: Vec<u64>,
    algorithm: CompressionAlgorithm,
    cols: DataColumns,
    quant: Option<QuantContext>,
    blocksize: u32,
    bytepix: u32,
    smooth: bool,
    bscale_bzero: (Option<f64>, Option<f64>),
    naxis1: u64,
    naxis2: u64,
    theap: u64,
}

// On each HDU subclass: a Mutex-protected cached
// (version, Arc<Meta>) pair.  None on first access; populated
// lazily.
struct CompressedImageHDU {
    meta_cache: Arc<Mutex<Option<(u64, Arc<CompressedImageMeta>)>>>,
    // ... existing fields
}

impl CompressedImageHDU {
    fn meta(&self, py: Python) -> PyResult<Arc<CompressedImageMeta>> {
        let cur_version = self.cards_version.load(Ordering::Acquire);
        {
            let cache = self.meta_cache.lock();
            if let Some((v, m)) = &*cache {
                if *v == cur_version { return Ok(m.clone()); }
            }
        }
        // Cold path: re-parse + cache.
        let cards = self.header_snapshot()?;
        let m = Arc::new(parse_compressed_image_meta(&cards)?);
        let mut cache = self.meta_cache.lock();
        *cache = Some((cur_version, m.clone()));
        Ok(m)
    }
}
```

Hot path: one Mutex lock + integer compare + Arc clone.  Cold
path (first call OR after mutation): the existing parse code.

**Mutation discipline.**  Every code path that mutates the cards
Vec must call `bump_cards_version()` on the HDU's
`cards_version`.  Mutation surface (today):

- `header.rs::rewrite_header_to_disk` — every FITSHeader
  mutation goes through this.
- `hdu_table::edit::insert_column` / `delete_column` — direct
  card-Vec mutations + header rewrite.
- Internal structural updates in `hdu_image.rs::ImageHDU::extend`
  (NAXISn card update), `hdu_table::write_*` (NAXIS2/PCOUNT
  card updates), `hdu_image_compressed.rs` (NAXIS2/PCOUNT/
  ZNAXIS<last> card updates), `hdu_table_compressed.rs`
  (similar).
- `checksum.rs::compressed_table_add_*` etc.

Suggested helper:

```rust
// On HDU; called by every code path that mutates cards.
fn bump_cards_version(&self) {
    self.cards_version.fetch_add(1, Ordering::AcqRel);
}
```

Audit pass: grep for `set_card_for_key`, `delete_card_for_key`,
`apply_setitem`, `append_commentary_to_cards`, direct
`cards.clone()`/`cards.push()` patterns; ensure each followed
by a `bump_cards_version()` call.  Debug-only safety net: a
`debug_assert!` in the cache fill path that computes a hash
of the cards Vec and compares to the last-cached hash; if
they differ but version matches, panic with a clear "missing
bump_cards_version() call" message.

**Pattern recurrence — what to cache, per HDU type.**

| HDU type | Per-call re-parse cost | Cached struct |
|---|---|---|
| `CompressedImageHDU` | ~15-20% of slice path (measured) | `CompressedImageMeta` (above) |
| `ImageHDU` | similar pattern, single-shot reads make it less hot | `ImageMeta` (BITPIX, shape, BSCALE, BZERO, BLANK) |
| `TableHDU` | `parse_columns` does ~6 scans × N_columns per call — likely the biggest unmeasured beneficiary | `TableMeta` (columns Vec, row_width, heap layout) |
| `CompressedTableHDU` | both `TableHDU` overhead PLUS Z-prefix parse PLUS `synthesize_uncompressed_cards` per call | `CompressedTableMeta` (compression algos + everything `TableMeta` has) |
| `AsciiTableHDU` | same shape as `TableHDU` | `AsciiTableMeta` |

For tables especially, `parse_columns`' per-column 6 scans means
a 100-column table does 600 O(N_cards) scans per `read()` /
`__getitem__` / `__setitem__` / etc.  Nobody's profiled this,
but it's almost certainly hot for any per-row workload.

**Phase outcomes.**

1. ✅ **Foundation** (`b36a864`).  `cards_version: Arc<AtomicU64>`
   on `HDU`, shared with every `FITSHeader` view.
   `CardsWriteGuard` with consuming `commit(new_cards)` and no
   `DerefMut` → impossible to mutate without bumping the version.
   No `bump_cards_version()` free helper was needed — the guard's
   `commit` does it; the compile-time guarantee replaced the
   "audit every mutation path" plan in the original design.
2. ✅ **`CompressedImageHDU` cache** (`c021a35`, `935b9a6`).
   `CompressedImageMeta` cached as
   `Arc<Mutex<Option<(u64, Arc<CompressedImageMeta>)>>>` on the
   HDU; `meta(&self, super_)` accessor.  Wired into
   `slice_compressed_image` + `read_compressed_image_data` (the
   `.read()` + checksum path) + every accessor (`shape`, `dtype`,
   `bitpix`, `ndim`, `size`, `__len__`, `n_tiles`, `__repr__`).
   Three private structs renamed for cross-module clarity now
   that they appear in the `pub(crate)` Meta API: `ColumnInfo` →
   `ZimageColumnInfo`, `DataColumns` → `ZimageDataColumns`,
   `QuantContext` → `ZimageQuantContext`.  Helper
   `compute_blank_mask_from_value` (in `hdu_image.rs`) replaces
   the card-parsing variants so the cached `meta.zblank` flows
   straight through.  **Measured: 40× faster than fitsio on the
   small-chunks slice benchmark, up from 24×.**  Write side
   (`write_compressed_image_data`, `extend_compressed_image_data`,
   `setitem_compressed_image`, `repack_compressed_heap`) still
   re-parses — deferred (writes are rare; refactoring is invasive
   because the helpers also call `unwrap_masked_input` /
   `normalize_input_dtype` which carry their own header parsing).
3. ✅ **`TableHDU` cache** (`ba70eb2`).  `TableMeta` (nrows /
   row_width / theap / columns) cached on `TableHDU`.  `read_table`
   and `read_one_column` refactored to take `&TableMeta` instead
   of `&[String]`.  Accessors (`nrows`, `__len__`, `ncols`,
   `dtype`, `colnames`, `units`, `__repr__`) routed through meta.
   `TableHDU::new_empty_cache()` factory exposed for the
   `CompressedTableHDU` PyClassInitializer chain.  Write paths
   (`write`, `append`, `__setitem__`, `insert_column`,
   `delete_column`, `repack`) still re-parse via `parse_columns`
   — deferred.
4. ✅ **`CompressedTableHDU` cache** (`0bc6c5b`).  Existing
   `CompressedTableMeta` (created for `__setitem__` paths)
   promoted to a cached field with the same shape as the other
   three.  `parse_compressed_table_meta` runs
   `synthesize_uncompressed_cards` first so `parse_columns` sees
   the original schema instead of the on-disk
   `TFORMn='1QB(...)'` + preserved `TDIMn` cards (which would
   trip on TDIM-on-VLA).  `data_offset` dropped from the meta
   (an earlier-HDU grow can shift it — caching would silently
   regress); callers fetch it fresh.  The Phase-3-era `ncols`
   workaround (direct TFIELDS parse, needed because
   `TableHDU.meta()` can't handle compressed cards) gone — Phase
   4's `meta()?.columns.len()` works naturally.
5. ✅ **`ImageHDU` cache** (`7ce3a83`, `8bfff7d`).  `ImageMeta`
   (bitpix / shape / bscale / bzero / blank) cached on
   `ImageHDU` (was a unit struct).  All 7 accessors routed
   through meta; `read_image_data` and `read_image_slice` take
   `&ImageMeta` instead of `&[String]`.  Lax parsing (NAXIS=0
   yields empty shape) keeps the accessor "never crash on
   primary HDU" contract intact — the strict NAXIS=0 check
   that `parse_image_hdu_shape` used to do is now done by the
   read helpers explicitly.  Write paths
   (`write_image_data`, `write_image_slice`, `extend`) still
   re-parse — deferred for the same reason as the table side
   (`unwrap_masked_input` + `normalize_input_dtype` carry their
   own parsing).

Five phases were independently landable and testable.  Foundation
(phase 1) was the only structurally interesting one; the others
applied the same pattern.

**Risks.**

- ~~**Forgotten bump in a mutation path**~~ — eliminated by
  Phase 1's `CardsWriteGuard` design: the type system makes
  `*guard = new_cards` impossible (no `DerefMut`), and the only
  mutation path (`commit`) always bumps.  New mutation sites
  pick up the bump automatically by going through
  `cards_write_lock()`.
- **Mutex contention** — only the lock is per-call; the parse
  is amortized.  Should be invisible vs the slice work.  If it
  ever shows up, switch to `parking_lot::Mutex` or `RwLock`.
- **Mutation of one HDU's header invalidating another HDU's
  cache** — shouldn't happen (each HDU has its own
  `cards_version`); FITS headers belong to one HDU.  Note: the
  `Arc<AtomicU64>` is shared between an HDU and any
  `FITSHeader`/`FITSHeaderEdit` views of that HDU's header
  (those views mutate THIS HDU's cards, so they must bump THIS
  counter).  Cross-HDU header copies via `update()` go through
  the destination HDU's own mutation path, so they bump the
  destination's counter naturally.

**Out of scope (don't get sidetracked).**

- Generic key→value parsed-cache on FITSHeader's public API
  (the "single source of truth" rule above explicitly rejects
  this; it's a different design and a different risk profile).
- SIMD-vectorizing the unshuffle/shuffle loops.  Maybe later,
  but the remaining `unshuffle` cost at 10.79% on small chunks
  is the algorithmic loop itself, not allocator overhead.
  Different kind of work.
- Adding the `numpy` Rust crate dep.  We chose the direct-write
  approach to avoid it; the header cache doesn't change that
  decision.
