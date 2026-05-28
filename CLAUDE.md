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
compressed — see the ZIMAGE section).  `create_image_hdu(...,
blank=<sentinel>)` emits the `BLANK` card (in stored space after
the unsigned-int trick transform); rejected for float dtypes.
`hdu.read(mask_blank=True)` returns a `numpy.ma.MaskedArray`
masking pixels matching `BLANK` (comparison in stored space, per
spec).  `write` / `__setitem__` / `extend` accept
`numpy.ma.MaskedArray` input; masked positions auto-fill with the
sentinel from the header (NaN for float HDUs).  See "BLANK /
ZBLANK + MaskedArray support" under the ZIMAGE section for the
shared `unwrap_masked_input` helper.

**Tile-compressed image writes (`ZIMAGE`)** — feature-complete.
See the ZIMAGE section for the full surface: all 5 algorithms,
all integer + unsigned-trick + unquantized-float + quantized-
float dtypes, `extend(data)`, `__setitem__`, `blank=` /
`mask_blank=True` / MaskedArray input, `Gzip1(level=)` /
`Gzip2(level=)`.

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

- **Compressed BINTABLE (tile-compressed tables, `ZTABLE`)** —
  separate FITS spec.  Phase 1 (detection + accessors + I/O stubs)
  shipped in `src/hdu_table_compressed.rs`.  Later phases will add
  read / slicing / VLA / write — see the "Tile-compressed tables
  (ZTABLE)" roadmap below.
- **Random groups (`GROUPS=T`, `PTYPEn`)** — legacy format,
  vanishingly rare in new files.
- **Memory-mapped reads** — chunked sequential I/O already keeps peak
  RSS at ~1 MiB above the output array, so motivation is weak.
- **Streaming / row-iterator API** — for tables that don't fit in
  RAM.  No user has asked yet; add when one does.
- **Remote file reads (`http`/`https`)** — open a FITS file from a
  URL.  **Shipped** (download-then-open, read-only; see the "Remote
  file reads" roadmap below).  Range-based partial reads, and
  `ftp`/`root`, are still deferred.
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

## Tile-compressed images (ZIMAGE) roadmap

ZIMAGE-format tile-compressed images are stored on disk as a
BINTABLE with `ZIMAGE=T` plus Z-prefixed image-shape and tile-
shape cards.  The user-facing API mirrors `ImageHDU`; internally
the reader walks tiles and decodes them.

**Status: feature-complete for typical workloads.**

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Detection + accessors + tile cache | ✅ Shipped |
| 2-3 | RICE_1 read + slicing + LRU cache | ✅ Shipped |
| 4 | GZIP_1 / GZIP_2 reads + fallback columns | ✅ Shipped |
| 5 | Quantized + unquantized float reads (all 3 dither methods) | ✅ Shipped |
| 6 | HCOMPRESS_1 + PLIO_1 reads | ✅ Shipped |
| 7 | All 5 algorithms' integer-ZBITPIX writes | ✅ Shipped |
| 7b | Unsigned-int trick (i1/u2/u4/u8) on writes | ✅ Shipped |
| 8 | Quantized + unquantized float writes (`Quantize` config; `quantize=None`) | ✅ Shipped |
| 9 | `extend(data)` + `__setitem__` (integer + unsigned-trick + unquantized-float) | ✅ Shipped |
| 9q | `extend` + `__setitem__` for quantized-float (no compounding loss) | ✅ Shipped |
| 10a | `blank=` + `mask_blank=True` + MaskedArray input | ✅ Shipped |
| 10b | `Gzip1(level=)` / `Gzip2(level=)` (custom zlib level) | ✅ Shipped |
| 10c | `add_checksum` / `verify_checksum` (CHECKSUM/DATASUM + ZHECKSUM/ZDATASUM) | ✅ Shipped |

**Open follow-ups (low priority):**

- **Performance** — release-mode chunked-read of large GZIP_2
  float files: **3.2× FASTER** than fitsio on big chunks (50k
  rows; at the decoder physics floor), **40× FASTER** on small
  chunks (1k rows; up from 24× before the header-meta cache
  landed — see Phase 2 in "Header-derived metadata caching"
  below).  All numbers are vs fitsio (the Python wrapper around
  cfitsio); fitsio carries its own per-call wrapping overhead
  that likely inflates these ratios relative to a direct-
  cfitsio comparison (worth measuring some day, but the
  realistic comparison for users is fitsio anyway).
  See "Performance — ZIMAGE chunked-read profiling history"
  below for how the gap closed.
- **Byte-exact heap agreement with cfitsio on quantized floats**
  — decoded values are bit-exact; raw heap bytes differ by qsort
  tie-breaking quirks.  Not worth fixing absent a specific need.

**Deferred (not in the punch list):**

- **Per-tile ZBLANK column** — header-level ZBLANK is supported;
  the convention also allows a per-tile column form.  Rare in
  practice and nobody's asked.  Defer.
- **`mask_blank=True` on quantized-float compressed reads** —
  rejected by design (matches the uncompressed-float rule; FITS
  spec forbids BLANK on floats — NaN serves that role).  Not a
  gap.

Test fixtures use fitsio for normal-path round-trips and hand-
crafted bytes for synthetic fallback-column cases (astropy is
also in the env for richer cases).

Implementation lives in `src/hdu_image_compressed/` (the
pyclass + dispatch) and `src/zimage/` (algorithm-specific
decoders).  Detection happens in `parse_hdus_from_file` in
`fits.rs`: a BINTABLE with `ZIMAGE=T` routes to
`CompressedImageHDU` instead of `TableHDU`.

**Phase 1 — Detection + accessors + cache plumbing.**  Done.
- `CompressedImageHDU` pyclass subclassing `HDU`.  Inherits
  `header` / `index` / `extname` / `extver` / `has_data` from
  the base; defines image-side accessors (`shape`, `dtype`,
  `bitpix`, `ndim`, `size`, `__len__`, `unit`) that read the
  Z-prefixed cards instead of NAXIS/BITPIX; and compression-
  specific accessors `compression` (returns the structured
  config object — see "Compression config" below) and
  `n_tiles` (storage-layout property).
- Tile-cache config: `tile_cache_size` getter + `set_tile_cache_size(bytes)`
  setter, default 32 MiB.  Storage of decoded tiles in the LRU
  itself landed in Phase 3 — Phase 1 only held the configured
  size.
- Detection is BINTABLE-with-ZIMAGE-card; the helper
  `header_has_zimage` is in the `hdu_image_compressed` module
  and called from `fits.rs::parse_hdus_from_file`.
- `header_has_zimage` is lenient about whitespace; the value
  parse just looks for a `T` in the value portion of the
  card after `=`.
- `extname`/`extver`/`has_data` are inherited from the HDU
  base.  `has_data` reads BINTABLE NAXIS2 (n_tiles), which
  happens to agree with "image has data" because an empty
  image has n_tiles=0 → NAXIS2=0.
- Raw `hdu.header["BITPIX"]` returns 8 (the BINTABLE bitpix
  on disk).  Use `hdu.bitpix` for the image-side value
  (`ZBITPIX`).  Astropy follows the same convention.

**Phase 2 — RICE_1 whole-image read.**  Done.  Decoder in
`src/zimage/rice.rs` (`BitReader` + RICE_1 spec per cfitsio's
ricecomp.c).  Dispatch in `src/zimage/mod.rs` via the
`CompressionAlgorithm` enum (parsed from ZCMPTYPE).  Integer
ZBITPIX only (8/16/32/64); `-32`/`-64` raises a "Phase 5"
NotImplementedError.  Non-RICE_1 ZCMPTYPE raises naming the
phase that will add it.  BSCALE/BZERO applied via the shared
`apply_image_scaling` machinery on the assembled array — the
read-side scaling code is unchanged.

Also landed: **inheritance restructure** —
`CompressedImageHDU` now extends `ImageHDU` via a
`PyClassInitializer` chain through HDU + ImageHDU +
CompressedImageHDU, so `isinstance(hdu, ImageHDU)` is True
on a compressed HDU.  Accessor `into_super()` calls step
through both parents (`slf.into_super().into_super()`).
ImageHDU's data-access methods are overridden by compressed-
specific implementations:
- `read` → the cache-aware tile decoder.
- `__getitem__` → tile-by-tile slice path.
- `write` → the bulk-write encoder (Phase 7 / 8).
- `extend` → append-along-axis-0 with partial-last-tile
  re-encoding (Phase 9).
- `__setitem__` → in-place pixel modification (Phase 9).

Phase 2 follow-ups (not blocking):
- **Add CompressedImageHDU cases to tests/test_repr.py** —
  the Phase 1 smoke test in test_image_compressed_accessors.py
  exists, but the full _show-instrumented visual-inspection
  suite in test_repr.py doesn't include compressed images
  yet.  Add alongside the ImageHDU/TableHDU cases.

**Phase 3 — Slicing + LRU cache.**  Done.  `__getitem__` walks
only the tiles overlapping the slice (per-axis `axis_overlap`
helper computes which slice indices land in each tile).  Same
slice surface as `ImageHDU`: slice / int / ellipsis per axis,
stepped slices, mixed int+slice, all-int → numpy scalar.

Bytes-bound LRU cache via the `lru` crate, keyed by flat tile
index.  Values are full-tile bytes already cast to the target
(stored) dtype in numpy C-order, so cache hits skip both
decode and dtype conversion.  Single tuning knob:
`set_tile_cache_size(bytes)` — 0 disables, default 32 MiB.
`tile_cache_used` reports current bytes; `clear_tile_cache()`
drops all entries (keeps the size setting).

`read()` and `__getitem__` both go through the same cache —
no per-call `cache=` kwarg.  Cache-on-by-default for `read()`
warms tiles for any follow-up slicing on the same handle; the
overhead is <1% wall time and ≤ tile_cache_size in memory.

Concurrency: the inner mutex is held only briefly for `get`
and `put` (Arc clone-out, then drop the lock).  File I/O for
a missed tile happens under the file lock per tile, but decode
runs without the file lock — multi-threaded callers don't
serialize through long decode runs.

**Phase 4 — GZIP_1 / GZIP_2 + fallback columns.**  Done.
Decoders in `src/zimage/gzip.rs`; trait widened to return
target-dtype native-order bytes; data-column lookup widened
to find primary + optional GZIP + optional UNCOMPRESSED
fallbacks; per-tile dispatch picks the first non-empty
column.

*Trait shape.*  `decode_tile_to_i64` was renamed to
`decode_tile_to_bytes(algorithm, compressed, n_pixels,
bytepix, blocksize, zbitpix) -> PyResult<Vec<u8>>`.  Each
decoder owns the entire bytes-to-bytes path including the
target-dtype cast and host-endian byteswap; the caller just
caches and places the bytes:
- RICE: bitstream decode → `cast_i64_to_target_bytes` (now
  living in rice.rs).
- GZIP_1: gzip decompress → byteswap (`!cfg!(target_endian
  = "big")`).
- GZIP_2: gzip decompress → reverse byte-shuffle → byteswap.

*Framing — the spec-vs-reality gotcha.*  The original Phase
4 plan in this file claimed cfitsio wrote a **zlib** stream
(magic `0x78 0x9C`) and recommended `ZlibDecoder`.  That was
wrong: cfitsio calls `deflateInit2` with `windowBits = 15 +
16`, which selects **gzip** framing (magic `0x1F 0x8B` plus
the CRC32/ISIZE footer).  We use `flate2::read::GzDecoder`
in `src/zimage/gzip.rs`; the framing check at the top of
the test suite confirms this is what fitsio actually writes.
Future code reading or producing tile-compressed gzip payloads
should use gzip framing, not raw zlib.

*Fallback columns.*  `find_compressed_data_column` became
`find_data_columns` returning `DataColumns { primary,
gzip_fallback, uncompressed_fallback }`.  Per-tile dispatch
lives in `fetch_tile_payload`: it reads the primary
descriptor first and, if nelements is 0, falls through to
the GZIP fallback and then the UNCOMPRESSED fallback.  The
return type is a `TilePayload` enum (`Compressed { bytes,
algorithm }` vs `Uncompressed { bytes }`) so `get_or_decode_tile`
knows whether to call the decoder dispatch or just byteswap.

`ColumnInfo` carries `inner_byte_width` so the descriptor's
`nelements` field converts to a byte count correctly even
when the column's inner type isn't byte (`UNCOMPRESSED_DATA`
can be `1PI`/`1PJ` matching the pixel type, in which case
nelements counts pixels not bytes).  COMPRESSED_DATA and
GZIP_COMPRESSED_DATA are always `1PB` so inner_byte_width=1.

*Test fixtures.*  Normal-path GZIP_1 / GZIP_2 round-trips
use fitsio.  Fitsio's high-level write API doesn't expose
fallback-column knobs, so the three fallback tests in
`tests/test_image_compressed_read_gzip.py` build minimal
FITS files by hand (`_build_fallback_fixture` and friends):
empty primary descriptor + populated fallback column + a
hand-built BINTABLE header.  This pattern is reusable for
any future fallback-related test.

*Scope boundaries unchanged.*  Phase 4 still supports
**integer ZBITPIX only** (`8/16/32/64`).  Float ZBITPIX
with GZIP still raises until Phase 5 (quantization).  The
GZIP decoders in principle handle any byte width, but with
no quantization layer the FP storage isn't meaningful yet.

*Known follow-up — i8 (TLONGLONG) with GZIP.*  fitsio
refuses to *write* i8 compressed images, raising "writing
TLONGLONG to compressed image is not supported" (cfitsio
`imcomp.c`).  This appears to be a cfitsio implementation
limitation rather than a FITS Tile Compression Convention
restriction — GZIP is a byte-stream codec with no inherent
BITPIX dependency, and the rustfits reader path already
handles `bytepix=8`.  Worth investigating later whether the
spec actually permits i8 GZIP (highly likely yes) and what
the cfitsio constraint is exactly; if the spec allows it,
our reader is already fine and the only missing piece is
test fixtures (hand-crafted, or maybe astropy can do it).

**Phase 5 — Quantized floats + unquantized float HDUs.**  Done.
Reader handles ZBITPIX=-32/-64 across all common shapes:
- **Quantized**: `NO_DITHER`, `SUBTRACTIVE_DITHER_1`, and
  `SUBTRACTIVE_DITHER_2` with bit-for-bit cfitsio agreement on
  the full {RICE_1, GZIP_1, GZIP_2} × {f4, f8} × {NO_DITHER,
  DITHER_1, DITHER_2} matrix (including NaN preservation
  through DITHER_2's reserved sentinel).
- **Unquantized**: ZQUANTIZ='NONE' (astropy's convention for
  "no quantization happened") and the astropy-quantize_level=0
  variant where ZQUANTIZ='NO_DITHER' is set but ZSCALE/ZZERO
  columns are absent.  On-disk bytes are raw GZIP-compressed
  floats; dequant is skipped.

*Code layout.*  All quantization logic lives in
`src/zimage/quantize.rs`:
- `DitherMethod` enum + `parse_dither_method` returning
  `Option<DitherMethod>` — `None` for ZQUANTIZ='NONE' (the
  "no quantization happened" signal).  Absent ZQUANTIZ
  defaults to `Some(NoDither)` per cfitsio's convention when
  ZSCALE/ZZERO are present.
- `random_table()` initialises the 10000-element Park-Miller
  table on first use (multiplier 16807, modulus 2^31-1,
  seed 1).  Lazy via `OnceLock`.
- `DitherStream` reproduces cfitsio's exact iseed / nextrand
  advancement (initial `nextrand = floor(table[iseed] * 500)`
  jump on roll-over) — required for byte-exact agreement with
  cfitsio's output.
- `dequantize_to_f32` / `_f64` apply the per-tile formula
  `(stored - dither + 0.5) * scale + zero`.  For DITHER_2 the
  reserved value `-2147483647` becomes NaN.
- 4 unit tests at the bottom of the module (Park-Miller
  anchor check, NO_DITHER linearity, DITHER_2 NaN handling,
  parse_dither_method covering 'NONE' + defaults + unknown).

*Integration in `hdu_image_compressed.rs`.*
- `DataColumns` widened to also hold `zscale_offset_in_row`
  and `zzero_offset_in_row` (fixed-width `1D` columns located
  by walking TTYPEn).
- `QuantContext { method, zdither0, zscale_offset_in_row,
  zzero_offset_in_row, output_zbitpix }`.  `build_quant_context`
  returns `Option<QuantContext>`: `None` when ZQUANTIZ='NONE'
  OR when ZSCALE/ZZERO columns are missing (the two signals
  for "no quantization happened" on a float HDU).  For integer
  HDUs the quant context is always None.
- A new `fetch_tile_payload_and_quant` reads the heap payload
  AND the per-tile ZSCALE/ZZERO doubles under the same file
  lock acquire (when quant is Some).  Falls back to the old
  `fetch_tile_payload` helper for the integer / unquantized
  paths (still present, kept simple).
- `get_or_decode_tile` decides bytepix and effective ZBITPIX
  for the decoder, then runs dequantization when the payload
  came from the primary column AND quant is Some.  A single
  unified rule: `dequant_applies = primary_payload &&
  quant.is_some()`.
- Both `read_compressed_image_data` and
  `slice_compressed_image` route through the new path; the
  cache stores final-output bytes (f4/f8 for quantized HDUs),
  same as the integer path.

*Decoder-vs-output ZBITPIX split.*  Two distinct values are
in play, and conflating them is a tempting bug:
- `stored_zbitpix`: what the *decoder* casts to — 32 (i32)
  for quantized float, same as image ZBITPIX otherwise.
- `output_zbitpix`: the image-side dtype — -32 / -64 for
  quantized float, integer ZBITPIX otherwise.
- `bytepix`: 4 for quantized float (matches stored i32),
  matches stored_zbitpix/8 otherwise.

When dequant doesn't apply (unquantized float HDU, or
fallback column on a float HDU), the decoder is called with
`bytepix = float_bytepix` (4 or 8 matching ZBITPIX) instead
of 4, since the bytes are already physical floats.

*Lossless-fallback convention.*  cfitsio's
GZIP_COMPRESSED_DATA fallback (and UNCOMPRESSED_DATA) for a
*float* HDU stores **raw original floats**, not quantized
i32.  This is the "lossless backup when quantization would
lose too much" path.  `TilePayload` was split into
`PrimaryCompressed` / `FallbackCompressed` / `Uncompressed`
variants so `get_or_decode_tile` can decide: dequant only
runs when the *primary* column produced bytes.  For the
fallback paths on float HDUs, bytes are already physical
floats — the decoder is called with the float bytepix
(4 or 8 matching ZBITPIX) and dequant is skipped.  This
unifies cleanly with the unquantized-float case: same code
path, same `dequant_applies = false` outcome.

*ZQUANTIZ='NONE' is NOT in the FITS spec.*  The FITS Tile
Compression Convention (Pence et al. 2010 + WG revisions)
defines exactly three ZQUANTIZ values (`NO_DITHER`,
`SUBTRACTIVE_DITHER_1`, `SUBTRACTIVE_DITHER_2`).  Per spec,
"no quantization" should be signalled by *omitting* the
keyword.  But astropy's `CompImageHDU` emits `'NONE'`
explicitly when writing unquantized float-compressed HDUs,
and cfitsio reads it tolerantly — so we accept it for
real-world compatibility.  Documented as a comment in
`parse_dither_method`.

*ZCMPTYPE alias accepted.*  fitsio writes `ZCMPTYPE='RICE_ONE'`
for some quantized-RICE configurations (an older cfitsio
synonym for `RICE_1`).  `parse_algorithm` in
`src/zimage/mod.rs` now accepts both names plus other older
cfitsio synonyms (`GZIP` for `GZIP_1`, `HCOMPRESS` for
`HCOMPRESS_1`).

*Known follow-ups not blocking the phase:*
- Header-level ZBLANK on integer compressed images (analog
  of `mask_blank` on uncompressed) still raises with the
  "Phase 2" message.  Lift when there's a use case.
- Per-tile ZBLANK column (column-form rather than header-
  level) — read code locates it (`find_data_columns` could be
  widened) but `QuantContext` doesn't carry it yet.  None of
  the matrix tests exercise this configuration; cfitsio
  typically emits a header-level ZBLANK for DITHER_2 rather
  than a per-tile column.

**Phase 6 — HCOMPRESS_1 read.**  Done.

Decoder in `src/zimage/hcompress.rs` — port of cfitsio's
`fits_hdecompress.c` (see "Reference sources for byte-exact
ports" under "Build / dev workflow" for how to obtain the
cfitsio source; the function names mirror cfitsio exactly so
side-by-side diffing during debug stays easy).  Both internal-precision paths shipped: i32 for
ZBITPIX = 8/16, i64 for ZBITPIX = 32 (cfitsio uses i64 internal
for 32-bit because the H-transform's intermediate sums can
overflow i32).  ZBITPIX = 64 is unsupported — cfitsio's encoder
doesn't write it and the spec has no 64-bit HCOMPRESS variant.

*Wire format gotchas.*  Stream starts with magic `0xDD 0x99`,
then i32 BE `nx, ny, scale`, then i64 BE `sumall`, then 3 bytes
of `nbitplanes`, then quadtree-coded bit planes per quadrant.
**SCALE comes from the stream, not the header** — the
ZNAMEn='SCALE' / ZVALn header value is informational only and
cfitsio ignores it in the decode path.  Only **SMOOTH** needs
header parsing (`parse_hcompress_smooth` in
hdu_image_compressed.rs, scans ZNAMEn pairs for 'SMOOTH').

*Axis convention.*  In cfitsio's HCOMPRESS code, `ny` is the
*FITS-fastest* axis (= NAXIS1 = numpy-last) and `nx` is the
slower one — opposite to "x = horizontal" usage but matching
the C row-major layout `a[i*ny + j]`.  In numpy order
(slowest first) that maps cleanly: `nx = tile_shape[0],
ny = tile_shape[1]`.  No reverse anywhere downstream.

*Dispatch.*  HCOMPRESS needs two extra pieces of info that
RICE/GZIP don't: the 2-D tile shape and the SMOOTH flag.  Added
`AlgorithmParams { tile_shape_numpy, smooth }` in
`src/zimage/mod.rs`; every call site (currently just
`get_or_decode_tile`) constructs one and passes it through
`decode_tile_to_bytes`.  RICE/GZIP ignore both fields.

*SMOOTH=1.*  Ported.  Three adjustment loops (hx / hy / hc) on
the inverse-H-transform coefficient block; edge coefficients
left untouched by the loop bounds (start at 2, end at nxtop-2 /
nytop-2), not by explicit boundary clauses.  Translation note:
cfitsio's
   `s = (s>=0) ? (s>>n) : ((s+(2^n-1))>>n)`
is "divide by 2^n truncating toward zero", which Rust's signed
`/` does directly — translated as `s / 8` / `s / 64` for
readability.

*Tests.*  `tests/test_image_compressed_read_hcompress.py`
covers accessors, lossless round-trip on u8/i16/i32, non-square
edge tiles, default tiles, slicing parity, bit-exact agreement
with cfitsio on a lossy hcomp_scale=4 fixture, and SMOOTH=1
round-trip at hcomp_scale={4, 16} for i16 plus hcomp_scale=8
for the i32 (i64-internal) path.

**Phase 6 — PLIO_1 read.**  Done.

Decoder in `src/zimage/plio.rs` — port of cfitsio's `pl_l2pi`
(`pliocomp.c`).  The cfitsio source is f2c-generated SPP with
heavy goto control flow; the Rust port replaces it with a
straightforward `match` on the opcode (top 4 bits of each
encoded word).

*Wire format.*  Stream is big-endian i16 shorts: 7-short header
(magic word `-100` at index [2] marks the modern format; an
older format puts a positive length there) followed by RLE data
words.  Each data word's top 4 bits select one of 8 opcodes:
zero runs (0, 5), set-high-pv (1 — consumes the next word for
the upper 16 bits of pv), increment/decrement pv (2, 3),
solid-pv run (4), single-pixel write with pv += data (6) or
pv -= data (7).

*Output type.*  Decoder produces `Vec<i32>` (PLIO is designed
for non-negative integer masks; values are built up through
increments from a starting pv=1).  Cast to the target dtype
follows the same shape as hcompress's: supports ZBITPIX = 8,
16, 32; rejects float ZBITPIX (PLIO doesn't make sense for
floats).

*Column TFORM.*  PLIO stores the compressed-data column as
`1PI` (VLA of i16 shorts) instead of the `1PB` used by RICE /
GZIP / HCOMPRESS.  No special-casing needed at the dispatch
layer — `tform_vla_inner_byte_width` returns 2 for 'I' so
`fetch_tile_payload_inner` correctly reads `nelements * 2`
bytes from the heap.

*Tests.*  `tests/test_image_compressed_read_plio.py` covers
the compression_type accessor, bit-exact agreement with cfitsio
on u8/i16/i32 mask-style fixtures (mostly zero with random
runs), all-zero tiles (header-only encoded stream), solid-value
tiles, and slicing parity.

All four ZIMAGE compression algorithms (RICE_1, GZIP_1/2,
HCOMPRESS_1, PLIO_1) now have full read support.

**Phase 7 — Compressed image writes (Gzip1, Gzip2, Rice1).**  Done.

*API.*  Compression is opted into via a structured config object
passed to `create_image_hdu`:

```python
f.create_image_hdu(
    "i4", (1000, 1000), extname="SCI",
    compress=rustfits.Gzip1(tile_shape=(100, 100), heap_format="P"),
)
f[1].write(data)
```

`compress=None` (default) → uncompressed `ImageHDU` (current
behavior, unchanged).  `compress=Gzip1(...)` / `Gzip2(...)` /
`Rice1(...)` / `Hcompress1(...)` / `Plio1(...)` →
`CompressedImageHDU`.  Algorithm objects (`Gzip1`, `Gzip2`,
`Rice1`, `Hcompress1`, `Plio1` — all shipped) live in
`src/zimage/compression_config.rs`.  Each is a Rust pyclass exposed
at `rustfits.<Name>`, validated at construction
(`Gzip1(blocksize=0)` raises immediately), immutable (no setters).
No inheritance — common kwargs (`tile_shape`, `heap_format`) are
duplicated across each constructor; for 5 small config classes the
duplication is trivial and avoids a PyClassInitializer chain.

Dispatch in `fits.rs::create_compressed_image_hdu_impl` goes
through an internal `CompressionConfigKind` enum that wraps the
per-algorithm pyclasses.  `CompressionConfigKind::from_pyany`
tries each variant's `extract::<Gzip1>()` / `extract::<Gzip2>()` /
`extract::<Rice1>()` / `extract::<Hcompress1>()` / `extract::<Plio1>()`
in turn — pyo3 0.28 doesn't grow a clean "first-match" extractor,
so the manual loop is the simplest shape.  The enum exposes the
shared surface (`tile_shape()`, `heap_format()`, `zcmptype()`)
plus an `extra_z_cards(bitpix)` accessor that returns `(ZNAMEn,
ZVALn)` pairs to emit alongside the standard ZIMAGE cards (RICE_1
emits BLOCKSIZE + BYTEPIX; HCOMPRESS_1 emits SCALE + SMOOTH;
GZIP and PLIO variants emit nothing).  PLIO uses TFORM1='1PI'
(i16 inner type) rather than '1PB' (byte) used by the others —
the encoder produces big-endian i16 shorts, and descriptor
nelements counts shorts not bytes.

**Structured `.compression` API.**  The read-side mirror image of
`compress=`: every `CompressedImageHDU` exposes a single
`.compression` getter that returns the same `Gzip1` / `Gzip2` /
`Rice1` / `Hcompress1` / `Plio1` pyclass instance that would have
been passed to `create_image_hdu(..., compress=...)` to produce a
file with the same on-disk parameters.  Round-trip pattern:

```python
cfg = Rice1(tile_shape=(16, 16), blocksize=64)
fits.create_image_hdu("i4", shape, compress=cfg)
assert fits[1].compression == cfg   # __eq__ is field-wise
```

Backing implementation: `hdu_image_compressed.rs::build_compression_config`
parses ZCMPTYPE, ZTILEn, TFORM1 (for heap_format), plus the
algorithm-specific ZNAMEn/ZVALn cards (BLOCKSIZE for RICE,
SCALE + SMOOTH for HCOMPRESS), and constructs the matching pyclass.
The flat HDU accessors `compression_type` and `tile_shape` were
removed in this refactor — use `hdu.compression.zcmptype` and
`hdu.compression.tile_shape` instead.  `n_tiles` stays on the HDU
(it's a storage-layout property derived from image_shape +
tile_shape, not an algorithm parameter).  Tile-cache runtime
knobs (`tile_cache_size`, `set_tile_cache_size`,
`clear_tile_cache`) also stay on the HDU.

The HDU repr uses the config object's own `__repr__`:

```
  compression: Rice1(tile_shape=[16, 16], heap_format='P', blocksize=32)
```

So the HDU repr stays the single source of truth no matter which
algorithm-specific parameters apply.  Fallbacks (repr never
crashes on a malformed file): unrecognized ZCMPTYPE → show the
raw string verbatim; missing ZCMPTYPE → show `None`.

Config classes also expose:
- `__eq__` (field-wise; cheap, useful for round-trip patterns).
- `zcmptype` getter returning the FITS-spec string (`"GZIP_1"`,
  `"RICE_1"`, ...).  Class name is the Pythonic form (`Gzip1`,
  `Rice1`); `.zcmptype` returns the FITS string when needed.

*Design discussion captured.*  See git log around this commit
for the full thread: the chosen shape is one unified
`create_image_hdu` (Option B) rather than a separate
`create_compressed_image_hdu` (Option A), with structured config
in the `compress=` slot rather than flat kwargs.  Decisions:
(1) `tile_shape` / `heap_format` live on the algorithm object
(not hoisted onto `create_image_hdu`); (2) Pythonic class names
(`Gzip1`, not `GZIP_1`); (3) eager validation, immutable objects;
(4) `heap_format` rather than `descriptor` (the kwarg name in
`create_table_hdu` was renamed at the same time for consistency).

*Layered architecture.*  rustfits's surface is designed for the
best ergonomics we can build.  A future `rustfits.compat.fitsio`
shim will present a fitsio-shaped surface for migrators (or as
the backend for a fitsio 2.0).  Decisions like the
`compress=Gzip1(...)` shape don't have to match fitsio's
`compress='GZIP_1', tile_dims=...` flat-kwarg style — the shim
absorbs the translation.

*Scope (current).*  `Gzip1` + `Gzip2` + `Rice1`; integer ZBITPIX
(8/16/32/64 for GZIP; 8/16/32 for RICE — i8/BYTEPIX=8 rejected,
see below) only; create + bulk `write` only.  Float ZBITPIX
(-32/-64) and unsigned-int trick dtypes (i1/u2/u4/u8) raise
`NotImplementedError`.  `extend` and `__setitem__` still raise
the original Phase 2 stub message.

*Mechanics.*  `CompressedImageHDU.write` encodes every tile into
RAM first (validate-then-mutate), then mutates the file: grows
via `shift_file_tail_and_update_offsets` when not the last HDU
(later HDU offsets bump in lockstep through the shared
`Arc<FileLayout>`), or `set_len` when it is; writes per-tile
descriptors into the main data section, then heap bytes; finally
rewrites the PCOUNT card via the same disk-write-before-commit
+ taint pattern every other write path uses.  Tile cache is
cleared on each write (cached entries from any prior read are
stale).

*GZIP_1 encoder.*  `encode_gzip1` in `src/zimage/gzip.rs` uses
`flate2::write::GzEncoder` with `Compression::default()` (zlib
level 6 — same as cfitsio/zlib defaults).  Input bytes are in
FITS big-endian order; caller (`write_compressed_image_data` in
`hdu_image_compressed.rs`) handles the byteswap via numpy
`ascontiguousarray(view, ">i4")` etc. — one pass that handles
dtype + endian together.

*GZIP_2 encoder.*  `encode_gzip2(pixel_bytes_be, bytepix)` in
`src/zimage/gzip.rs` runs the GZIP_1 encoder's underlying gzip
primitive over byte-shuffled input.  The `shuffle()` helper is
the inverse of `unshuffle()` from the decoder side — for
`bytepix=1` the shuffle is a no-op and the path collapses to
GZIP_1 (we even round-trip the same bytes through the file in
that case; tests anchor this invariant).  `encode_tile_from_bytes`
in `src/zimage/mod.rs` grew a `bytepix` parameter to thread the
pixel width through to the encoder (GZIP_1 ignores it).

*RICE_1 encoder.*  `encode_rice(pixel_bytes_be, n_pixels, bytepix,
blocksize)` in `src/zimage/rice.rs` is a byte-exact port of
cfitsio's `fits_rcomp` / `_short` / `_byte` family.  The
algorithm: read pixels as sign-extended integers of the natural
bytepix width into i32 working precision; for each block of up to
`blocksize` pixels compute ZigZag-mapped deltas, sum them in f64,
derive the Rice split parameter `fs` from cfitsio's exact dpsum/psum
heuristic (with per-bytepix psum cast type — u8/u16/u32 caps psum
at the natural unsigned width); pick one of three branches per
block (low-entropy: emit `fs=0` marker only; high-entropy: emit
`fsmax+1` marker + raw bbits-wide diffs; normal Rice: emit `fs+1`
marker + unary top + raw fs-wide bottom for each pixel).  A small
MSB-first `BitWriter` (cousin of the existing `BitReader`) handles
the bit packing.  `encode_tile_from_bytes` grew an
`AlgorithmEncodeParams { blocksize }` struct (mirror of the
decode-side `AlgorithmParams`) plus an `n_pixels` arg so RICE knows
how many integers to pull out of the input bytes.  The output is
**byte-exact identical to cfitsio's encoder output** on the same
input — verified across u1/i2/i4 dtypes, multi-tile layouts,
low-entropy / high-entropy / mixed blocks.

*RICE_1 i8 (BYTEPIX=8) rejected.*  cfitsio's encoder family stops
at BYTEPIX=4: there is no `fits_rcomp_longlong`, fitsio raises
"writing TLONGLONG to compressed image is not supported", and
astropy silently downcasts i64 → i32 before encoding (lossy when
values exceed the i32 range, no warning).  No production reader
verifies BYTEPIX=8 RICE streams, so writing them would produce
files unreadable outside rustfits.  `create_compressed_image_hdu`
rejects `compress=Rice1()` + i8 dtype upfront with a clear
`NotImplementedError` pointing at `Gzip2` instead.  Empirical check
(see git log around this commit): on real i64 imaging patterns
(smooth + Poisson, smooth + 1e12 bias, wide-spread, sparse),
Gzip2 lands within ~5% of where i64 RICE would compress, and is
universally readable.  The encoder itself also rejects bytepix=8
as a defensive check.

*Tests.*  `tests/test_image_compressed_write_gzip.py` (GZIP_1),
`tests/test_image_compressed_write_gzip2.py` (GZIP_2), and
`tests/test_image_compressed_write_rice.py` (RICE_1) —
each covers accessors after create, dtype matrix (u1/i2/i4 +
i8 for GZIP only), shape matrix (1-D / 2-D square / 2-D non-square
with edge tiles / 3-D / whole-image single tile), default-tile-shape
round trip, bidirectional cross-check with fitsio (rustfits-written
read by fitsio AND fitsio-written read by rustfits, both bit-exact),
non-last HDU growth (heap shifts later HDU forward), and all
rejection paths (float, unsigned trick, shape mismatch, start kwarg).
The GZIP_2 file adds the bytepix=1 shuffle-collapses-to-GZIP_1
invariant (file sizes equal on `u1` input) and the mixed-algorithm
case (one GZIP_1 + one GZIP_2 HDU in the same file).  The RICE_1
file adds **byte-exact heap-comparison tests against fitsio**
(low-entropy, high-entropy, multi-tile, dtype matrix), a custom
blocksize test, and the i8-rejected test.

*HCOMPRESS_1 encoder.*  `encode_hcompress` in
`src/zimage/hcompress.rs` is a byte-exact port of cfitsio's
`fits_hcompress.c` family (`htrans`, `digitize`, `shuffle`,
`encode`, `doencode`, `qtree_encode`, `qtree_onebit`,
`qtree_reduce`, `bufcopy`, `write_bdirect`, plus the
`output_nbits`/`nybble`/`nnybble` bit-output state).  Internal
precision tracks the read side: i32 for ZBITPIX = 8/16, i64 for
ZBITPIX = 32 (the H-transform's intermediate sums can overflow
i32 for 32-bit input).  ZBITPIX = 64 (i8 dtype) is rejected
upstream — the FITS Tile Compression Convention has no 64-bit
HCOMPRESS variant.  SCALE controls quantization (0 or 1 =
lossless; larger = lossy); SMOOTH is encoder-irrelevant (the
smoothing pass runs on read only) but its ZNAME card must be
emitted so the reader knows whether to smooth.  Output is
byte-exact with cfitsio's encoder across all tested cases.

*HCOMPRESS_1 tile-shape policy.*  Two FITS Tile Compression
Convention constraints apply: (1) every image dimension must
have ≥ 4 pixels (HCOMPRESS is a 2-D wavelet); (2) every tile
along each axis (including the edge tile) must have ≥ 4 pixels.
The three major writers disagree on how to enforce constraint
(2): **astropy** raises `ValueError`; **cfitsio** silently
rewrites the user's tile dim upward (e.g. `tile=16` becomes
`tile=17` for `naxis=50` to make `50 % 17 = 16` instead of
`50 % 16 = 2`); **rustfits follows astropy** — explicit is
safer — but the error message includes the cfitsio-style
adjusted suggestion so the user can copy it back into their
config.  Default `tile_shape` (when `Hcompress1(tile_shape=
None)`) is **not** the FITS row-tiles convention (which would
always violate constraint 2); instead it ports cfitsio's
default heuristic in `fits.rs::hcompress_default_slow_tile`:
whole image when `NAXIS2 ≤ 30`, otherwise the first value from
`{16, 24, 20, 30, 28, 26, 22, 18, 14}` that leaves a valid
edge tile, with `17` as the last-resort fallback.  16-row
stripes are what HST/DECam/HSC files in the wild use (cfitsio
is the dominant writer of these), so the default matches.

*HCOMPRESS_1 tests.*
`tests/test_image_compressed_write_hcompress.py` covers
accessors after create, dtype matrix (u1/i2/i4), shape matrix
(divisible 2-D cases), default-tile-shape (small-image,
large-image stripe, and fallback-when-16-doesn't-work cases),
lossless + lossy round trips (scale ∈ {4, 8, 16}), SMOOTH=True
round trip, **byte-exact heap agreement with fitsio** on
single-tile + multi-tile-with-divisible-dims + valid-edge-tile
cases for both lossless and lossy, bidirectional fitsio
cross-check, non-last HDU growth, mixed-algorithm file, and all
rejection paths (float, i8, unsigned trick, 1-D / 3-D images,
shape mismatch, start kwarg, dim < 4, tile < 4, and thin edge
tile with cfitsio-style suggested fix in the error message).

*HCOMPRESS_1 lossy testing gotcha — `hcomp_scale` sign.*  When
cross-checking against fitsio's writer with a fixed scale (e.g.
to compare byte-exact heap output at `scale=4`), pass
`hcomp_scale=-4` (negative).  cfitsio's `hcomp_scale > 0`
branch multiplies the user value by a per-tile noise estimate
to derive the actual scale, giving a *different* scale per
tile; the `hcomp_scale < 0` branch uses the absolute value
directly as a fixed scale.  Tests in `phase7_hcompress_write.py`
use the negative form.

*PLIO_1 encoder.*  `encode_plio` in `src/zimage/plio.rs` is a
byte-exact port of cfitsio's `pl_p2li` from
`<cfitsio>/pliocomp.c`.  The SPP/f2c source's goto soup is
replaced by a single `while` loop with explicit state and
linear emit logic — much easier to read than cfitsio's labelled
control flow.  The opcode dispatch matches the read side
(0=zero-run, 1=set-high-pv, 2/3=±pv, 4=solid-pv-run,
5=zero-run-with-trailing-pv, 6/7=set-and-write-single-pixel);
single-pixel run combinations (opcodes 5/6/7) are emitted via
the same `+ 20481` / `| 16384` modifications cfitsio uses on
the previous word.  PLIO is integer-only (ZBITPIX 8/16/32; no
64-bit variant in the FITS Tile Compression Convention);
inputs must be non-negative (encoder rejects negatives with a
clear error), and pv must fit in 2^27 - 1 (encoder rejects
larger values rather than silently truncating as cfitsio does).

*PLIO_1 TFORM and descriptor mechanics.*  PLIO writes its
heap as i16 big-endian shorts (TFORM1='1PI' or '1QI'), unlike
the other algorithms which use byte-inner TFORM1='1PB'/'1QB'.
The descriptor's `nelements` field counts ELEMENTS of the
inner type, not bytes — so on the write side
`write_compressed_image_data` divides `encoded.len()` by
`inner_byte_width` (= 2 for PLIO, 1 for the others) when
filling descriptors.  The read side already worked correctly
because `tform_vla_inner_byte_width` returns the right value.

*PLIO_1 tests.*  `tests/test_image_compressed_write_plio.py`
covers accessors after create (including TFORM1='1PI'
verification), dtype matrix (u1/i2/i4), shape matrix, default
tile shape, degenerate cases (all-zeros, all-solid, large-pv,
sparse single-pixel), **byte-exact heap agreement with fitsio**
across mask-style and degenerate inputs, bidirectional fitsio
cross-check, non-last HDU growth, mixed-algorithm file
(Plio1 + Gzip2), and rejection paths (float, i8, unsigned
trick, negative pixels, values > 2^27, shape mismatch, start
kwarg).

**Unsigned-int trick on write (i1/u2/u4/u8).**  Shipped
2026-05-23.  Works for Gzip1/Gzip2/Rice1/Hcompress1 (PLIO_1 is
rejected because the reverse XOR produces signed stored values
that include negatives, which PLIO's non-negative encoder can't
represent).  Implementation: reuses
`hdu_image.rs::reverse_unsigned_trick`; the compressed-write
path's `normalize_compressed_input_dtype` helper dispatches
fast-path (BITPIX-native input pass-through) vs reverse-XOR
based on the input dtype and the HDU's BSCALE/BZERO.  BSCALE/
BZERO cards are emitted at create time in
`create_compressed_image_hdu_impl` using the same pattern as
the uncompressed path.  See
`tests/test_image_compressed_unsigned_trick.py` (33 cases).
*Known limitation (not new):* astropy returns f8 (with
precision loss) on u8 + BZERO=2^63 — also affects uncompressed
u8.  rustfits's own round-trip is bit-exact.

**Phase 8 — Quantized float compressed writes.**  Shipped (read +
write).

*API.*  Quantization parameters live in a separate `Quantize`
config object passed alongside the algorithm:

```python
fits.create_image_hdu(
    "f4", shape,
    compress=Rice1(tile_shape=(100, 100)),
    quantize=Quantize(level=4.0, method="dither1", seed=0),
)
```

Kwargs: `level` (default 4.0 = "N sigma per quanta"; negative =
fixed bscale = -level), `method` (`'no_dither'` / `'dither1'` /
`'dither2'`; default `'dither1'` matches cfitsio), `seed`
(ZDITHER0; 0 → on-disk default of 1).  Integer HDUs reject
`quantize=` with a clear error.  Float HDUs WITHOUT `quantize=`
(or with `quantize=None`) write **unquantized** — lossless raw
float bytes through GZIP_1 / GZIP_2 — see the "Unquantized
float compression" subsection below.  Float HDUs WITH a
`Quantize` object emit the 4-column quantized schema described
in the *Schema* section.  Pythonic kebab-style strings for
method names; FITS-spec ZQUANTIZ values are available via
`quantize.zquantiz`.

*Schema (quantized floats).*  Float HDUs WITH a Quantize emit
a 4-column BINTABLE:

  | column | TFORM | role |
  |--------|-------|------|
  | COMPRESSED_DATA       | 1PB / 1QB / 1PI / 1QI | quantized i32 → algorithm output |
  | ZSCALE                | 1D                    | per-tile bscale |
  | ZZERO                 | 1D                    | per-tile bzero  |
  | GZIP_COMPRESSED_DATA  | 1PB / 1QB             | lossless raw float fallback |

Plus ZQUANTIZ (NO_DITHER / SUBTRACTIVE_DITHER_1 /
SUBTRACTIVE_DITHER_2), ZDITHER0, and ZBLANK = -2147483647
(NULL_VALUE_I32 sentinel).  The FITS Tile Compression Convention
records method + seed only — not the qlevel — so the
`CompressedImageHDU` carries `quantize_config: Arc<Mutex<Option<
Quantize>>>` populated at create time and consulted by
`.write()`.  Reopened HDUs get `None` and fall back to qlevel=4.0
(cfitsio's default).

*Quantize / dequantize algorithm.*  Direct ports of cfitsio's
`fits_quantize_float` / `_double` and the `FnNoise5_float` /
`_double` MAD noise estimator.  Per-tile noise = min of 2nd / 3rd
/ 5th-order Median Absolute Differences (Pence MAD-to-sigma
normalization constants 1.0483579 / 0.6052697 / 0.1772048);
bscale = noise / qlevel; bzero chosen to keep the quantized
range inside i32 minus 10 reserved sentinels.  Per-pixel:
DITHER_2 reserves NULL_VALUE_I32 for NaN and ZERO_VALUE_I32 for
exact-zero floats; the other methods just reserve NULL_VALUE_I32
for NaN.  The dither stream (Park-Miller PRNG, same one already
shared with the decoder) advances per-pixel regardless of
sentinel hits to stay synced with the encoder.

*GZIP fallback.*  When `quantize_float` returns `None` (constant
tile, range too wide for i32, fewer than 2 pixels), the tile's
raw float bytes are GZIP-1 compressed and stored in the
GZIP_COMPRESSED_DATA column; the primary descriptor stays empty
(nelements=0).  The read side already routed empty primary to
the fallback column in Phase 4, so no read changes needed.

*PLIO + float rejected.*  PLIO's encoder requires non-negative
inputs.  Quantization produces an i32 stream with negative
values (bzero shifts the range), which PLIO can't represent.
The `create_compressed_image_hdu_impl` dispatch rejects the
combination up-front with a clear error pointing the user at
Gzip2 or Rice1.

*Lossless decoder fix.*  Phase 5 had a latent bug — only
SUBTRACTIVE_DITHER_2 in `dequantize_to_f32/_f64` checked
NULL_VALUE_I32 → NaN; NoDither and SUBTRACTIVE_DITHER_1 silently
turned the sentinel into a giant negative number.  Phase 8
commit 3 added the missing checks to both branches.  No
existing tests caught the gap because no Phase 5 fixture had
NULL_VALUE_I32 in the stored stream under NoDither or DITHER_1.

*Tests (`test_image_compressed_write_quantize.py`).*  29
cases covering: schema (TFORM / TTYPE / ZQUANTIZ / ZDITHER0 /
ZBLANK cards), round-trip across f4/f8 + (NO_DITHER, DITHER_1,
DITHER_2), default Quantize defaults, fitsio cross-read agreement
across the (algorithm, method) matrix, DITHER_2 exact-zero
preservation, NaN round-trip for all three methods, GZIP
fallback on constant input, ZBLANK absent for integer HDUs,
seed=0 → on-disk default of 1, rejection paths (integer dtype,
no compress).  Plus 18 Rust-side unit tests anchoring the noise
estimator + per-pixel quantize against round-trip math.

*Known limitation: byte-exact heap agreement with cfitsio.*  The
fitsio cross-read tests assert **physical-pixel-value** agreement
(bit-exact decoded output), not raw-heap-byte equality.  The
two writers can pick slightly different bscale on the same input
because cfitsio's per-row diff arrays are f32 while ours are
f32 (matched on f32 path) but the cross-row averaging order
differs subtly with `qsort` stability quirks.  The decoded
result is still identical because ZSCALE/ZZERO are recorded
per-tile.  Heap-byte equality would require a more painstaking
port of cfitsio's qsort tie-breaking; not worth doing absent a
specific need.

*Unquantized float compression (`quantize=None`).*  Shipped
2026-05-23.  Float HDUs without an explicit `Quantize` config
get lossless raw-byte storage through GZIP_1 or GZIP_2 —
matching astropy's `quantize_level=0` layout exactly.

**API.**
- `f.create_image_hdu("f4", shape, compress=Gzip1(...))` —
  unquantized, lossless raw float bytes through GZIP_1.
- `f.create_image_hdu("f4", shape, compress=Gzip2(...),
  quantize=None)` — same thing, explicit; GZIP_2's byte-shuffle
  typically gives 3-5% better compression than GZIP_1 on float
  bit patterns.
- `f.create_image_hdu("f4", shape, compress=Gzip1(...),
  quantize=Quantize(...))` — opt in to lossy quantization
  (4-10× better compression at the cost of precision).
- Integer HDUs reject `quantize=` regardless of value.

The default for float HDUs is **unquantized**.  Lossy
quantization is opt-in to avoid silently throwing away
precision on scientific data.  (Before 2026-05-23 the default
was `Quantize()` = dither1; that was changed at the same time
as removing `Quantize(method='none')` since no users existed
yet.  Decision logged with the empirical algorithm-vs-
algorithm comparison that motivated it.)

**Compress requirement.**  Empirically verified (astropy
7.2.0 + each algorithm at `quantize_level=0`):

| compress= | unquantized float behavior |
|-----------|---------------------------|
| Gzip1     | bit-exact lossless ✓ |
| Gzip2     | bit-exact lossless ✓ (~3-5% better than Gzip1 on float data) |
| Rice1     | astropy writes it but the round-trip is NOT bit-exact — Rice coding on float bit patterns produces garbage |
| Hcompress1| astropy writes it but the round-trip is NOT bit-exact — H-transform is an integer wavelet |
| Plio1     | astropy hard-rejects (mask-only encoder) |

So `quantize=None` requires `compress=Gzip1(...)` OR
`compress=Gzip2(...)`.  Rice1/Hcompress1 + unquantized float
returns a `ValueError` pointing at Gzip1/Gzip2.  Plio1 +
float returns its own algorithm-specific
`NotImplementedError` (PLIO + float is rejected upstream
since PLIO never works with floats regardless of
quantization).

**On-disk format.**  Matches astropy's `quantize_level=0`
schema exactly:
- ZCMPTYPE='GZIP_1' or 'GZIP_2'.
- ZQUANTIZ + ZDITHER0 + ZBLANK all omitted (the FITS Tile
  Compression Convention says ZQUANTIZ is optional and
  defaults to NO_DITHER when absent; no quantization happened,
  so there's no dither stream and no NaN sentinel to record).
- Single-column BINTABLE: TFIELDS=1, TTYPE1='COMPRESSED_DATA',
  TFORM1='1PB' or '1QB'.  No ZSCALE, no ZZERO, no
  GZIP_COMPRESSED_DATA fallback (the primary column IS
  lossless for GZIP).
- Each tile's COMPRESSED_DATA descriptor points at a
  gzip-framed stream of the tile's raw float bytes in FITS
  big-endian order.

**Standards landscape** (corrected from prior CLAUDE.md notes
— astropy 7.2.0's actual behavior was empirically tested):

|  | FITS spec | cfitsio | astropy (7.2.0) |
|--|--|--|--|
| ZQUANTIZ for unquantized | optional; absent → defaults to `NO_DITHER` | absent OR `'NO_DITHER'` | `'NO_DITHER'` |
| ZCMPTYPE for unquantized | mandatory; must name algorithm | reflects user's choice | reflects user's choice (any algorithm — though only GZIP gives bit-exact) |
| Schema for unquantized GZIP | spec doesn't mandate | n/a (typically quantized) | single COMPRESSED_DATA column with raw GZIP'd float bytes |
| `'NONE'` as ZQUANTIZ value | NOT in spec | tolerated on read | not emitted |

**Implementation.**  The mode is detected at write time via
`is_float && TFIELDS==1` (works for both freshly-created and
reopened HDUs, since the create path always emits TFIELDS=1
for unquantized floats).  In that mode,
`write_compressed_image_data` routes through `encode_tile_int`
with the GZIP_1/GZIP_2 algorithm + float bytepix — the
encoder treats the bytes opaquely.  Single-column descriptor
emission; no ZSCALE/ZZERO/fallback.  Read-side handling is
unchanged from Phase 5: `build_quant_context` returns None
(missing ZSCALE/ZZERO columns), the decoder is called with
float bytepix, no dequantization applied.

Tests in
`tests/test_image_compressed_write_unquantized.py`:
schema validation, round-trip f4/f8 × Gzip1/Gzip2, astropy +
fitsio cross-read agreement (both directions), non-last HDU
growth, omit-kwarg semantics, rejections (Rice1/Hcompress1
generic, Plio1 algorithm-specific, removed
`Quantize(method='none')` spelling).

**Phase 9 — `CompressedImageHDU.extend(data)`.**  Shipped
2026-05-23 for integer, unsigned-int trick, unquantized-float,
and quantized-float HDUs (all 5 algorithms except PLIO_1
paired with unsigned trick).  The quantized-float path reuses
each tile's existing per-tile bscale/bzero/dither seed so
unchanged pixels round-trip with NO compounding loss — see
"Quantized-float mutation" section below.

API: `extend(data)` — no `start=` kwarg.  In-place writes to
existing tile rows are `__setitem__`'s job (re-encode affected
tiles); extend only appends along numpy axis 0.  Partial-last-tile
case is supported: the boundary tile is decoded, combined with the
first portion of new data, and re-encoded.  Old boundary-tile
heap bytes become orphans (left in place; descriptors no longer
point at them).

Implementation in `hdu_image_compressed.rs::extend_compressed_image_data`
(file-private, mirrors `write_compressed_image_data`).  Mechanics:
pre-read existing main + heap into RAM; encode boundary tiles
(decode via cache → byte-concat with new portion → encode); encode
new tile rows; grow file (shift later HDUs or set_len); write
updated main table + relocated heap + appended heap; rewrite
header (NAXIS2 + PCOUNT + ZNAXIS<last>).  Same taint contract as
write: pre-shift failures don't taint; failures inside the write
loop do.

Tests in `tests/test_image_compressed_extend.py` (46 cases):
1-D + 2-D + 3-D extends, tile-aligned and partial-last-tile,
dtype matrix (u1/i2/i4 + unsigned trick i1/u2/u4), algorithm
matrix (Gzip1/Gzip2/Rice1/Hcompress1), multiple sequential
extends, astropy cross-read, non-last HDU growth, unquantized
float, all rejection paths.

**Phase 9 — `CompressedImageHDU.__setitem__(key, value)`.**
Shipped 2026-05-23 for integer, unsigned-int trick,
unquantized-float, and quantized-float HDUs.

API: same slice surface as `__getitem__` (slice / int / ellipsis
per axis, stepped slices, mixed combinations).  RHS is a scalar
(numpy broadcasts across the selection) or an ndarray whose
shape exactly matches the selection's output shape.  Unsigned-
int trick HDUs accept the scaled dtype (e.g. u2 on a u2-trick
HDU) and reverse-transform via `normalize_compressed_input_dtype`.

Implementation in `hdu_image_compressed.rs::setitem_compressed_image`
(file-private, mirrors `extend_compressed_image_data`'s shape
but simpler — no descriptor table growth, no heap relocation).
Mechanics: for each tile that overlaps the selection (via the
same `axis_overlap` helper `__getitem__` uses), decode existing
tile → wrap in numpy (with `.copy()` since `frombuffer` is
read-only) → numpy `set_item` with the appropriate RHS portion
→ re-encode → append to heap.  Modified-tile heap bytes become
orphans (left in place; descriptors no longer reference them).
PCOUNT grows; file may grow if past the padded extent.  Same
taint discipline as extend.

Tests in `tests/test_image_compressed_setitem.py` (29 cases):
single-pixel writes (1-D and 2-D), row/col writes, multi-tile
contiguous slices, scalar broadcast, stepped slices, 3-D,
dtype matrix including unsigned trick, algorithm matrix,
unquantized float, multiple sequential modifications, astropy
cross-read, non-last HDU growth, empty-slice no-op, all
rejection paths.

**Heap orphaning trade-off.**  After many `__setitem__` calls
modifying the same tiles, the heap grows monotonically with
orphaned bytes.  Call `hdu.repack()` to rebuild the heap with
only live tiles (see "Heap repack" below for the shared
mechanism with `TableHDU.repack()`).

**GZIP compression level (`Gzip1(level=...)` / `Gzip2(level=...)`).**
Shipped 2026-05-23.  Accepts 0..=9 (zlib levels: 0 = none, 1 =
fastest, 9 = best); `None` (default) uses the codec default of 6
— same as cfitsio/zlib/astropy.

The level is a **write-only** parameter: the gzip stream format
itself doesn't preserve the level that produced it (the decoder
handles any level identically), and the FITS Tile Compression
Convention defines no ZNAMEn/ZVALn slot for it.  Within the same
Python session `hdu.compression.level` returns the user's value
(stored on the HDU); after close + reopen the field comes back
as `None`.

**`compress_config` field on `CompressedImageHDU`.**  Added
alongside `quantize_config` to hold the user's full
`CompressionConfigKind` (i.e., the `Gzip1(...) / Gzip2(...) /
Rice1(...) / Hcompress1(...) / Plio1(...)` they passed to
`compress=`).  Same-session `hdu.compression` returns the stored
config so write-only params (`level`, and any future additions)
round-trip via `.compression`.  At create time the stored cfg's
`tile_shape` is replaced with the resolved value (the actual
on-disk tile shape, after applying the algorithm's default-tile
heuristic) so `.compression.tile_shape` returns the real value
even when the user passed `Gzip1()` without specifying it.  For
reopened HDUs the field is `None`; `.compression` falls back to
rebuilding from header cards (level → None).

`CompressionConfigKind` was moved from `fits.rs` (file-private)
to `compression_config.rs` (`pub(crate)`) so the HDU can hold
it.  The enum gained a `gzip_level()` accessor (Some for Gzip1/
Gzip2, None for others) and a `with_resolved_tile_shape(ts)`
method used at create time.  The write/extend/__setitem__
dispatchers take `compress_config: &Arc<Mutex<Option<
CompressionConfigKind>>>` and extract `gzip_level` from it,
threading it through `IntTileCtx` / `FloatTileCtx` /
`encode_quant_boundary_tile` / `AlgorithmEncodeParams` and into
`encode_gzip1` / `encode_gzip2`.

Tests in `tests/test_image_compressed_gzip_level.py` (26 cases): config-class API
(default, repr, equality, validation), same-session round-trip,
reopen loses level, file-size proxy (level=9 ≥ 10% smaller than
level=1 on compressible data), bit-exact round-trip across all
levels × Gzip1/Gzip2, level applies to extend + __setitem__,
non-GZIP algorithms reject level kwarg.

**BLANK / ZBLANK + MaskedArray support.**  Shipped 2026-05-23
for both compressed and uncompressed image HDUs.

API:
- `create_image_hdu(..., blank=<sentinel>)` — emits `BLANK`
  (uncompressed) or `ZBLANK` (compressed integer) so reads
  with `mask_blank=True` find the sentinel.  Value is in
  PHYSICAL space (user's dtype); transformed to STORED space
  for the card when the unsigned-int trick is in play (e.g.,
  user passes `blank=65535` for `u2`, card lands as
  `BLANK=32767` because the on-disk i2 storage XORs the sign
  bit).  Rejected on float dtypes (FITS spec forbids BLANK
  on floats; NaN serves that role).
- `hdu.read(mask_blank=True)` — returns
  `numpy.ma.MaskedArray` with True at pixels whose stored value
  equals BLANK/ZBLANK.  Comparison happens in stored space
  (pre-scaling) per FITS spec.  Rejected on float ZBITPIX.
  Already worked for uncompressed; this PR enables it for
  compressed integer HDUs.  When the keyword is absent the
  return is still a MaskedArray (consistent return type) but
  with `nomask`.
- `write` / `__setitem__` / `extend` accept
  `numpy.ma.MaskedArray` input on both compressed and
  uncompressed HDUs.  Masked positions are auto-filled with
  the sentinel from the header (BLANK or ZBLANK in physical
  space) — error if the header has no sentinel and the input
  is integer.  For float HDUs, fill value is NaN (no header
  dependency).  All-False masks short-circuit (no header
  lookup needed; just unwrap to underlying data).

Implementation: single shared helper
`hdu_image.rs::unwrap_masked_input(py, data, header, is_compressed)`
runs at the top of all 6 write entry points (uncompressed +
compressed × write/extend/__setitem__).  The helper internally
parses BITPIX/ZBITPIX (image-side) and BSCALE/BZERO to
determine the fill semantics — caller passes only a one-line
preamble.  Read-side reuses
`compute_blank_mask_for_key(header, arr, key)` generalized
from the existing `compute_blank_mask`.

Tests in `tests/test_image_blank_mask.py` (27 cases): round-trip
across u1/i2/i4 × {compressed Gzip1/Gzip2/Rice1, uncompressed};
unsigned-int trick interaction; MaskedArray input on all 6
write paths; float-NaN fill; all rejection paths; astropy +
fitsio cross-read of rustfits-written ZBLANK files.

Interop notes (verified empirically): astropy 7.2.0 promotes
integer + BLANK/ZBLANK reads to f64 + NaN (rather than
returning a MaskedArray); fitsio reads the file fine but
doesn't auto-mask.  Neither tool has a clean MaskedArray write
API for compressed images — rustfits's `blank=` kwarg + auto-
fill behavior is the friendlier shape.

**Quantized-float mutation (extend + __setitem__).**  Shipped
2026-05-23.  When extending or mutating a tile that was
originally stored in the **primary** (quantized i32) column,
rustfits reuses the EXISTING per-tile bscale/bzero (read from
the tile's ZSCALE/ZZERO row entries) AND the existing dither
seed (deterministic from `row_1based` + `ZDITHER0`).  This
makes `requantize(dequantize(stored))` idempotent: unchanged
pixels in the modified tile round-trip BIT-EXACT — no
compounding quantization noise.  Verified by tests in
`tests/test_image_compressed_quant_mutation.py` (24 cases)
that compare a reference no-mutation file against a post-
mutation file across f4/f8 × {no_dither, dither1, dither2}.

Tiles originally in the **GZIP fallback** column (lossless raw
float bytes, used when the original quantize couldn't fit the
range) stay in the fallback after modification — re-encoded as
GZIP_1 of the modified raw float bytes.  No new precision loss.

**Out-of-range rejection.**  If a user writes a value that
doesn't fit the tile's existing per-tile bscale/bzero (after
the unscale arithmetic the i32 stored value would land outside
the legal range), the mutation is rejected with a clear error
message that names the three recovery options: (1) recreate
the file with `Quantize(level=N)` for a smaller N (coarser
scales admit wider ranges), (2) recreate with
`Quantize(level=-bscale)` to pin bscale explicitly, or
(3) recreate with `quantize=None` for lossless raw-float GZIP.
The error message also reports the failing pixel index, value,
and the tile's bscale/bzero so the user can size the new
scale.

Implementation in
`src/zimage/quantize.rs::requantize_float_fixed_scale` /
`requantize_double_fixed_scale` (idempotency anchored by
Rust unit tests).  The mutation dispatchers
(`extend_compressed_image_data` and `setitem_compressed_image`)
read each affected tile's primary descriptor to detect
primary-vs-fallback storage, then route through
`encode_quant_boundary_tile` which handles both cases.

**Checksum convention (CHECKSUM/DATASUM + ZHECKSUM/ZDATASUM).**
Shipped 2026-05-23.

API (matches astropy's 4-method shape; same on every HDU type):
- `hdu.add_datasum()` — compute the data-section checksum,
  update the DATASUM card (or ZDATASUM for compressed).
- `hdu.add_checksum()` — also computes CHECKSUM (or ZHECKSUM):
  the encoded complement of (header + data) so the total HDU
  checksum lands on 0xFFFFFFFF, per the FITS Checksum
  Convention.
- `hdu.verify_datasum()` — returns `True` / `False` / `None`
  (None = DATASUM card absent).
- `hdu.verify_checksum()` — same return shape.

Per HDU type:
- **ImageHDU + TableHDU**: emit CHECKSUM + DATASUM (standard
  uncompressed).
- **CompressedImageHDU**: emit ZHECKSUM + ZDATASUM (per the
  FITS Tile Compression Convention, the integrity check is
  against the *equivalent uncompressed image*, not the
  on-disk BINTABLE).  Astropy uses the same convention.
- **CompressedTableHDU** (fixed + VLA columns): same
  ZHECKSUM + ZDATASUM convention; the equivalent
  uncompressed bytes are reconstructed tile-by-tile (decode
  each (tile, col) blob to BE bytes, interleave per row)
  and fed to a streaming checksum.  For VLA columns: the
  per-tile dual-descriptor blob is GZIP-decompressed, the
  ORIGINAL P/Q descriptors are copied into the tile's main
  buffer at the column's offset, and per-cell metadata
  `(orig_offset, vlalen, cvlalen, cvlastart, col_idx)` is
  collected.  After the tile walk feeds all main buffers,
  cell metadata is sorted by `orig_offset` and walked in
  order: gap bytes between cells (zero padding from sparse
  layouts) are fed as zeros in <=64 KiB chunks, then each
  cell's compressed bytes are read from the heap and
  decompressed to BE (with cfitsio's uncompressed-fallback
  branch when `cvlalen == vlalen * elem_size`) and fed to
  the stream.  Trailing zeros are fed to reach ZPCOUNT.
  Peak memory bounded at one tile's main buffer + one
  decompressed dual-desc blob + one per-cell decoded buffer
  + the cell-metadata vector (~40 bytes per non-empty VLA
  cell), regardless of file size.  Compressed-VLA-X cells
  are rejected (matches the read-path scope: X inner letter
  isn't supported in compressed VLA columns).

Manual semantics — re-run `add_checksum()` after any mutation
(write / __setitem__ / extend / append).  Auto-update on
every write would be expensive and surprising; cfitsio and
astropy both defer to the user.

Cross-tool agreement:
- astropy + fitsio both verify uncompressed rustfits-written
  CHECKSUM/DATASUM as valid (cfitsio-byte-exact encoding).
- astropy's `verify_checksum` for **compressed** HDUs has its
  own internal bug (TypeError on
  `_compute_checksum(None)`) that triggers on its own writes
  too; we don't cross-verify compressed against astropy.
  Our self-verify catches corruption correctly.

Implementation in `src/checksum.rs` — direct port of cfitsio's
`checksum.c` (`ffcsum` / `ffesum` / `ffdsum`).  Byte-exact
agreement with cfitsio anchored by Rust unit tests
(`encode_matches_cfitsio` reference values were generated by
compiling cfitsio's `ffesum` standalone — see commit log).
Shared HDU-level scaffold in `hdu_image.rs`
(`checksum_hdu_add_*` / `checksum_hdu_verify_*`).  Both the
uncompressed scaffold and the compressed-table
implementation **stream** the data section (1 MiB chunks for
uncompressed; per-tile decode for compressed-table) through
the shared `ChecksumStream` accumulator in `src/checksum.rs`,
so peak memory is bounded at a per-chunk constant regardless
of file size.  The compressed-image checksum
(`read_uncompressed_image_be_bytes` in
`hdu_image_compressed.rs`) still materializes the full
ndarray — predates the streaming rule; flagged as a
follow-up perf item.  All three variants build a synthetic
equivalent-uncompressed-HDU header for the ZHECKSUM
computation.  Reuses the `rewrite_header_to_disk` helper
from `header.rs` (now `pub(crate)`) so a header that grows
past its reserved blocks gets in-place expansion (same
machinery the FITSHeader edit path uses).

`ChecksumStream` (in `src/checksum.rs`) is the streaming
accumulator that makes per-chunk feeding correct:
`compute_checksum_bytes` zero-pads partial trailing 4-byte
groups, so calling it on consecutive sub-multiples-of-4
chunks produces a different result than one call on the
concatenation.  The accumulator buffers up to 3 leftover
bytes between feeds so callers (the per-tile compressed-
table walk; the 1 MiB chunked uncompressed-data reader) can
stream arbitrary chunk sizes safely.  `finish()` applies the
tail-padding rule one last time on the final remainder.

The `card_string` helper was tightened in the same commit to
match the cfitsio/astropy convention of putting `/` at column
32 (1-indexed) for fixed-format string-valued cards.
Previously we placed `/` immediately after the closing quote,
which is spec-legal but produced different byte sequences
than cfitsio — and astropy's `verify_checksum` does a
byte-exact comparison of the encoded sum, so the offset
mattered.  Fix is purely formatting; values unchanged.

Tests in `tests/test_hdu_checksum.py` (19 cases) — round-trip on
all three uncompressed/compressed-image HDU types, add_datasum
independence, None when absent, corruption detection on
ZDATASUM card + heap byte, astropy/fitsio cross-verify
(uncompressed), algorithm matrix (compressed image), ZDATASUM
= uncompressed-equivalent DATASUM.
Plus `tests/test_table_compressed_checksum.py` (22 cases) for
the table side: round-trip with various nrows/ztilelen
combinations (single tile, partial last tile, parametrized
matrix), add_datasum-only, verify-None-when-absent, ZDATASUM
card corruption detection, heap-byte corruption raising
ValueError (mirrors the image-side test), ZDATASUM equals
the DATASUM of the equivalent uncompressed table, funpack
cross-tool (cfitsio decompresses our file → the reconstructed
DATASUM matches our ZDATASUM), same-handle vs reopen parity,
refresh-after-setitem (re-running add_checksum on a mutated
table picks up the new value), and VLA-bearing tables across
six cases: VLA-only round trip, mixed fixed+VLA round trip,
empty cells, ZDATASUM equals the uncompressed-equivalent
DATASUM (anchors the synthetic-heap walk), ZDATASUM card
corruption detected, and funpack-decompresses VLA file →
DATASUM matches ZDATASUM (strongest interop check).

## Tile-compressed tables (ZTABLE) roadmap

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

## Remote file reads (download-then-open) — roadmap

cfitsio can open files over the network (HTTP/FTP/root drivers).
This is the rustfits plan for the equivalent.  **Status: flavor #1
(http/https download) shipped 2026-05-28; flavor #2 (range reads)
deferred.**

Two distinct flavors, very different cost:

1. **Download-then-open (shipped).**  Fetch the whole remote file
   into a `Storage::Mem` buffer, then parse it like any in-memory
   file.  Read-only against the remote (no write-back); pays the
   full download even for a one-tile read.  This is essentially what
   cfitsio does for most of its network drivers.  See "Design
   (flavor #1, as built)" below.
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
- *Schemes: `http` / `https` only.*  `ftp` / `root` need separate
  protocol crates and are out of scope — the user's `ftp://` example
  is not handled (download it with another tool first).
- *`.gz` composes:* a URL whose path (sans `?query` / `#frag`) ends
  in `.gz` is gunzipped after download, same as a local `.gz` path.
- *No caching.*  Each `FITS(url)` downloads fresh.  An on-disk cache
  keyed by URL + ETag/Last-Modified is a possible follow-up; deferred
  (invalidation complexity) unless repeat-open is a real pattern.

Tests: `tests/test_fits_remote.py` (11 cases) run against a local
`http.server` — deterministic, no external network.  `https` uses the
same code path (only the scheme differs) and isn't unit-tested (would
need a self-signed cert).

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
| `http`/`https` **download** read (→ `Mem`) | ✅ Shipped |
| `http`/`https` range reads, `ftp`/`root` | ⬜ See "Remote file reads" roadmap |

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
| `ftp` / `root` (download) | needs a separate protocol crate | ⬜ |
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
   chunk wins vs fitsio (see "Performance — ZIMAGE chunked-read
   profiling history" elsewhere in this file) are unmatched and
   currently undocumented user-facing.  Worth a short page if
   we want users to know; deferred because benchmark numbers age
   fast and invite arguments about methodology.
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

## Performance — ZIMAGE chunked-read profiling history

Reading a 1.49-billion-pixel f8 GZIP_2 ZIMAGE file (~12 GB;
per-user fixture, not committed — see "Performance / large
fixture files" under "Build / dev workflow" for how to obtain or
synthesize an equivalent) in chunks via `f[1][lo:hi]` slicing.

**Current state (release builds; 2026-05-26):**

- **Big chunks (50k rows)**: rustfits is **3.2× FASTER** than
  fitsio.  `decode_gzip2`'s remaining ~48% of slice time is now
  ~100% the flate2/miniz_oxide inflate itself — the physics floor.
- **Small chunks (1k rows)**: rustfits is **40× FASTER** than
  fitsio (up from 24× after the Phase 2 header-meta cache
  landed).  fitsio caches all decoded tiles forever (memory
  bloat); rustfits has an LRU AND a per-HDU parsed-meta cache
  that eliminates the 8+ linear card scans per slice call.

**Note on the comparison baseline.**  All ratios are vs **fitsio**
(the Python wrapper around cfitsio), not direct cfitsio.  fitsio
carries its own per-call wrapping overhead — Python-level argument
marshalling, ndarray construction, etc. — that likely inflates
these numbers relative to a direct-cfitsio comparison.  The
realistic comparison for users is fitsio, so that's what the
benchmark uses; a direct-cfitsio comparison would be worth doing
some day to attribute the gap more precisely.

**How the gap closed.**  Four sequential profile-and-fix passes
plus the header-meta cache:

1. **`tile_origin_and_shape` stack-allocation** (commit a85583f).
   The function did three `vec![0u64; d]` per call for what is
   pure arithmetic on 2-element arrays.  Replaced with caller-
   provided `[u64; MAX_NAXIS]` stack buffers (MAX_NAXIS=8).
   Closed small-chunks 1.4× slower → 1.05× of cfitsio.
2. **Direct-write to output buffer** (commit a85583f).  Old per-
   tile path was `PyBytes::new + frombuffer + reshape + set_item`
   — four Python round-trips and a full Vec→PyBytes copy per
   tile.  New `strided_copy_c_contig_to_c_contig` helper writes
   tile bytes directly into the output ndarray's buffer via
   `RawBuffer::acquire_writable`.  Stepped slices and int-
   collapse axes still take the old PyBytes path.  No new dep.
3. **`OverlappingTileRange` iterator** (commit a85583f).  Old
   loop walked `0..n_tiles` per chunk and skipped non-overlapping
   tiles inside the body.  On row-stripe layouts (fpack default)
   this is most of the work.  Pre-computes per-axis
   `(tc_first, tc_extent)` and yields a tight iteration over
   only overlapping tiles in BINTABLE row-major order.
4. **Eliminate alloc_zeroed in unshuffle/shuffle** (commit
   d7edc54).  `vec![0u8; n_pixels * bytepix]` → `Vec::with_capacity
   + spare_capacity_mut + MaybeUninit::write + unsafe { set_len }`.
   The zero-init pass was 17.67% of total profile on big chunks.
   Took big-chunks 2.8× → 3.2× faster.
5. **Header-meta cache (Phase 2 of header-derived metadata
   caching).**  Per-call re-parsing of ~8 header fields
   (ZCMPTYPE, ZNAXISn, ZTILEn, ZNAMEn/ZVALn, TFORMn, TTYPEn,
   ZBLANK, BSCALE/BZERO) collapsed to one Mutex lock + Acquire
   load + Arc clone via the cached `CompressedImageMeta`.  Took
   small-chunks 24× → **40× faster than fitsio** — the cache hit
   eliminates ~40% of the per-call overhead.  See "Header-derived
   metadata caching" section below for the design (CardsWriteGuard
   foundation in Phase 1, parsed-meta cache in Phase 2).

**Profiling gotchas worth remembering.**

- numpy on import spawns OpenBLAS worker threads that
  busy-spin even when idle.  They'll dominate any flamegraph
  100% unless you set `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` in the env.
- Even with `OPENBLAS_NUM_THREADS=1`, Python interpreter
  startup dominates short workloads.  Scale the benchmark up
  until the read loop takes ≥30s of wall time so it dominates
  the sampling window.
- Debug-build profiles are misleading — the original "3×
  slower" observation in this file came from a debug build.
  Always profile with `maturin develop --release` (debug
  symbols stay via `[profile.release] debug = "line-tables-only"`
  in Cargo.toml).
- The per-user 12 GB GZIP_2 f8 fixture isn't committed.  The
  bench script at `bench_chunked_read.py` (outside the repo)
  cycles `lo = (i * CHUNK_ROWS) % (n_rows - CHUNK_ROWS)` over
  100k chunks of 1k rows for small-chunks, 1k chunks of 50k
  rows for big-chunks.  Cycling pattern is what amortizes the
  LRU tile cache — sequential reads wouldn't get the same
  cache amplification.

## Header-derived metadata caching (shipped, 2026-05-26)

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
"Performance — ZIMAGE chunked-read profiling history" above for
the full fix-by-fix breakdown.

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
