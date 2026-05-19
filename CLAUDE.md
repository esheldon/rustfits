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
  `BLOCK_SIZE`/`CARD_SIZE`/`CARDS_PER_BLOCK`, `parse_keyword`.  File
  primitives plus the shared byte-shift helper that the header-grow path
  (and the future image/table data-grow paths) all call into.
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
  `RawBuffer`, bitpix conversions, shape parsing.  Only exports `ImageHDU`
  (+ `new`) and `dtype_to_bitpix` (used by `FITS::create_image_hdu`).
- `src/hdu_table.rs` — `TableHDU` (BINTABLE) pyclass stub.
- `src/hdu_ascii_table.rs` — `AsciiTableHDU` (TABLE) pyclass stub.
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
- Keyword normalization: `normalize_keyword()` trims whitespace and ASCII-
  uppercases.  Case-insensitive lookup and writes.  HIERARCH long keys also
  go through this normalization (case preservation deferred).

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

**Forward-looking (Tier 2, deferred until table writing lands):** the
image-only metadata keys `BUNIT`, `BSCALE`, `BZERO` are *not* protected —
they're fine to set on an image HDU — but they're meaningless on a table
HDU.  When `update()` is extended to copy from one HDU to another, the
destination's HDU type needs to be consulted: if the target is a table,
these three keys should be stripped from the source.

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
to allow space, `.`, `+` in addition to the standard A-Z/0-9/`-`/`_`.  The
literal "HIERARCH" is rejected as a user key (it's the convention prefix).
Find/set/delete match HIERARCH cards via `keyword_of`, which returns the
long-key for HIERARCH cards.

Long HIERARCH string values auto-chain via `build_hierarch_string_cards`:
the first card carries the HIERARCH prefix and a `&'`-terminated chunk
(payload budget `65 - len(key)` bytes), followed by N standard CONTINUE
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

**Reuse for image/table data grow (not yet implemented).** When image
extend (`ImageHDU.extend`) eventually grows non-last HDUs, and when
table append/heap-grow lands, both will call the same
`shift_file_tail_and_update_offsets` helper — image/table grow inserts
bytes at `self.data_offset + self.data_size` instead of `self.data_offset`,
and self's own offset record stays unchanged (only `header_block_count`
is header-specific; data-section size is computed from the header by
`calculate_data_size`).  This is the reason the shared-offsets refactor
went in before the table writing work.

**Taint semantics in the grow path:** pre-shift failures (file lock,
metadata, `set_len`) do NOT taint — nothing on disk has moved yet.  Any
failure inside the shift loop, the post-shift `flush`, or the subsequent
header `write_all`/`flush` DOES taint, because the file may now be
inconsistent.  See the `check_not_tainted` block above.

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
