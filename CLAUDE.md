# rustfits — design notes for contributors

A Rust+PyO3 implementation of a FITS reader/writer for Python.  This file
captures architectural decisions and conventions that aren't obvious from
reading the code.  Claude Code loads this automatically into every session
in this directory; humans should read it before making structural changes.

## Project structure

The Rust extension is split into single-responsibility modules.  Each only
exposes (`pub(crate)`) what neighboring modules actually import; everything
else stays private to its file.

- `src/lib.rs` — `#[pymodule]` init + `mod` declarations.  Nothing else.
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
- `src/hdu_table.rs` — `TableHDU` (BINTABLE) pyclass: parses TFORM /
  TDIM / TSCAL / TZERO / TNULL / TUNIT / THEAP / PCOUNT, reads fixed
  and variable-length columns, writes via `create_table_hdu` +
  `TableHDU.write` / `__setitem__` / `append`.  Free helpers
  `write_fixed_only` / `write_vla_aware` / `append_fixed_only` /
  `append_vla_aware` carry the per-path I/O so the pymethods stay
  thin dispatchers.
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
module surface area is the whole point of the split.

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

**Missing.**
- **Tile-compressed images (`ZIMAGE`)** — large, separate spec; the
  most commonly-needed extension.  Do before any ZTABLE work.

### Image write

**Supported.**  `FITS.create_image_hdu(dtype, shape, *, extname,
extver)` writes header + zero-filled data.  `ImageHDU.write(data,
start=...)` does an explicit-start bulk write.  `__setitem__` is
symmetric with `__getitem__`: anything readable is writable.  RHS can
be a scalar (Python int/float, numpy scalar, or 0-d ndarray —
broadcast across the selection) or a shape-matching ndarray with
dtype matching BITPIX.  Stepped slices fall into per-pixel writes via
the same strip-layout walk as the read path.  `ImageHDU.extend(new_shape)`
grows the data section in place, shifting the file tail and bumping
later-HDU offsets when the image is not the last HDU on disk.  Mid-
write I/O failures taint the file (close + reopen to recover).

**Missing.**
- **BSCALE/BZERO scaling on write** — the read side does scaling, but
  the write side stores values as-given.  Symmetric round-trip would
  need to emit `BITPIX=signed-int + BZERO=2^(n-1)` when the user gives
  an unsigned-int dtype, and reverse-transform float input for HDUs
  with non-trivial BSCALE/BZERO already set.
- **Tile-compressed image writes (`ZIMAGE`)** — pair with the read
  side; do both at once when ZIMAGE work lands.

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

**Missing (ordered by likely value).**
1. **Variable-length P/Q with `repeat > 1`** — currently rejected.
   Rare (most VLA columns are `1Pt`) but legal.  Multi-descriptor
   means N descriptors per row, each pointing at its own heap cell.
   Field dtype would need to be an Object array of shape `(repeat,)`
   per row, or some other reshape — decide before coding.
2. **Variable-length P/Q with TDIMn** — currently rejected.  TDIMn on
   a P/Q column would mean "reshape each heap cell to these dims",
   useful for VLA-of-images.  Each cell still uses the inner element
   type; the reshape is just on the ndarray after the heap read.
3. **Variable-length P/QX (bit array in heap)** — rejected (inner X).
   Niche.  Heap bytes are the same MSB-packed format as fixed X;
   the heap-side unpacker would mirror `convert_x_cell`.
4. **VLA TNULL masking** — fixed-col TNULL is implemented; VLA
   columns with TNULL in the header are rejected when `mask_null=
   True`.  Adding support means a per-row bool ndarray for each
   masked VLA cell (parallel Object dtype mask field, or
   MaskedArrays for each cell — decide representation before coding).
5. **`max_size`-style read for variable columns** — fitsio offers a
   mode where each variable cell becomes a fixed-size N-D array
   padded to the largest cell.  Explicitly deferred (user request);
   noted here so we don't forget.
6. **`TDISPn`** — display format hint.  Informational, similar
   shape to TUNIT but rarely used.

### Table write

**Supported.**  `FITS.create_table_hdu(dtype, nrows=0, *, extname,
extver, units, var_dtypes, descriptor)` maps a numpy structured
dtype to TFORMn / TDIMn / TUNITn cards.  `var_dtypes={col:
inner_dtype}` sidecar declares VLA columns (numpy `'O'` field +
sidecar entry); `descriptor='P'` (default; 8-byte descriptors, 32-bit
nelements/offset, 4 GB heap ceiling) or `'Q'` (16-byte, no practical
ceiling).  `TableHDU.write(data)` bulk-writes the table (fixed +
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

**Missing.**
- **VLA `__setitem__`** — fixed `__setitem__` is implemented; VLA
  is deferred because cell-resize forces heap re-layout (either
  rewrite the heap, or always-append-and-orphan which bloats).
  Revisit when an actual workload demands it.
- **String VLAs (`PA`) on write** — read side has niche support;
  not on the write side.  Defer until requested.
- **Bit VLAs (`PX`) on write** — paired with the read-side gap.
- **`X` (bit) columns on write** — numpy `bool` currently maps to
  `L` (one byte per bool).  True `X` would need an explicit opt-in.
- **Multi-column / fancy / `(row, col)` `__setitem__`** —
  `hdu[[c1, c2]] = ...`, `hdu[[1, 3, 5]] = ...`, and tuple writes
  are rejected with a clear `ValueError`.  Add when a use case
  shows up.
- **ASCII tables (creating, writing)** — rare in modern files.
- **Add / remove columns from existing tables** — header rewriting
  + byte shuffling; non-trivial.
- **`TDISPn` on write** — informational, low priority.

### Cross-cutting (read + write)

- **Compressed BINTABLE (tile-compressed tables, `ZTABLE`)** — large,
  separate spec.  `ZIMAGE` is the more commonly-needed sibling and
  isn't done either; do that first.
- **Random groups (`GROUPS=T`, `PTYPEn`)** — legacy format,
  vanishingly rare in new files.
- **Memory-mapped reads** — chunked sequential I/O already keeps peak
  RSS at ~1 MiB above the output array, so motivation is weak.
- **Streaming / row-iterator API** — for tables that don't fit in
  RAM.  No user has asked yet; add when one does.

## Table write roadmap

Plan for getting table creation + writing on par with the image side.
Reading is mature; Phases 1 (create + bulk write), 2 (__setitem__),
3 (append/extend), and 4 (VLA columns on write/append, except
`__setitem__`) have all shipped.  Only Phase 5 (ASCII tables,
add/remove columns, VLA `__setitem__`) is deferred.  The image
side has all four and the patterns translated directly.

**Phase 1 — `create_table_hdu` + bulk `TableHDU.write()`.**  Foundation
for everything else.
- `FITS.create_table_hdu(dtype, nrows=0, *, extname=None, extver=None,
  units=None)` — maps a numpy structured dtype to TFORMn / TDIMn /
  TUNITn cards, writes the header block, allocates `nrows * row_width`
  zero-filled bytes.
- `TableHDU.write(data)` — dtype-checked, byteswapped bulk overwrite.
- Scope: fixed-width columns only (B/I/J/K/A/E/D/L/C/M, plus subarray
  fields via numpy `(T, shape)`).  No VLA, no `X` bit columns, no
  ASCII tables.

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

Deferred (not implemented; clear `ValueError` on attempt): multi-
column subset writes `hdu[[c1,c2]] = ...`, fancy row-list writes
`hdu[[1,3,5]] = ...`, tuple `(row, col)` writes.  Add when a use
case shows up.

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
append surfaces.  `__setitem__` for VLA columns is deferred.

**API.**  `create_table_hdu(dtype, ..., var_dtypes=None,
descriptor=None)`.  The user puts `'O'` for VLA fields in the numpy
structured dtype and passes a sidecar `var_dtypes={col_name:
inner_dtype_str}` so we can pick a FITS inner letter (we considered
extending the dtype-list 3-tuple to carry inner type, but the
sidecar mirrors the existing `units=` pattern and keeps numpy
dtypes free of FITS-specific DSL).  `descriptor=` is `'P'`
(default; 8-byte descriptors, 32-bit nelements/offset, 4 GB heap
ceiling) or `'Q'` (16-byte, 64-bit, no practical ceiling).  No
maxlen hint is emitted in TFORM for now (read side already accepts
both `1PE` and `1PE(100)` shapes).

**Inner types supported.**  Numeric: `B / I / J / K / E / D / C /
M`; plus `L` (bool).  String VLAs (`PA`) and bit VLAs (`PX`) are
NOT implemented on write — these are the two cases the read side
also rejects/has-niche-support-for; defer until a user asks.

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

**Phase 5 — out of scope.**  ASCII tables (rare in modern files);
adding / removing columns from existing tables (header rewriting
+ byte shuffling); VLA `__setitem__` (resizing a cell forces heap
re-layout — either rewrite the heap, or always-append-and-orphan
which bloats; revisit when an actual workload demands it).

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
