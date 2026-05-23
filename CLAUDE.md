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

**Missing.**
- **Scalar broadcast with scaling** — `__setitem__` scalar RHS
  (`img[k] = 42`) currently goes through `scalar_to_be_bytes` which
  reads the value at the BITPIX dtype; for unsigned-trick HDUs this
  means the user must pass the BITPIX (signed) value, and for
  generally-scaled HDUs the user must pass the stored (pre-scaling)
  value.  Promoting to a 0-d ndarray (which routes through
  `normalize_input_dtype` and gets the reverse-transform) is the
  workaround.  Add when there's a use case.
- **Tile-compressed image writes (`ZIMAGE`)** — Phase 7 shipped
  all five integer-ZBITPIX encoders (`Gzip1`, `Gzip2`, `Rice1`,
  `Hcompress1`, `Plio1`); Phase 8 added float ZBITPIX writes via
  the new `Quantize` config object (`compress=Rice1(...)` +
  `quantize=Quantize(level=..., method=..., seed=...)`).
  Compressed `extend`/`__setitem__` are Phase 9+ (mutation).
  Remaining follow-ups: unsigned-int trick dtypes
  (i1/u2/u4/u8) on the compressed-write side; PLIO_1 + float
  (rejected — non-negative-only encoder).

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
heap_format=None)`.  The user puts `'O'` for VLA fields in the
numpy structured dtype and passes a sidecar `var_dtypes={col_name:
inner_dtype_str}` so we can pick a FITS inner letter (we considered
extending the dtype-list 3-tuple to carry inner type, but the
sidecar mirrors the existing `units=` pattern and keeps numpy
dtypes free of FITS-specific DSL).  `heap_format=` is `'P'`
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

## Tile-compressed images (ZIMAGE) roadmap

ZIMAGE-format tile-compressed images are stored on disk as a
BINTABLE with `ZIMAGE=T` plus Z-prefixed image-shape and tile-
shape cards.  The user-facing API mirrors `ImageHDU`; internally
the reader walks tiles and decodes them.

**Status:** Phases 1-6 shipped (full read support for all five
algorithms — RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1 with SMOOTH,
PLIO_1 — plus quantized + unquantized floats).  Phase 7
compressed-write shipped for **all five** integer-ZBITPIX
encoders: GZIP_1, GZIP_2, RICE_1, HCOMPRESS_1, PLIO_1.  Phase 8
shipped float writes (quantized via Quantize config; lossless
GZIP fallback for unquantizable tiles).  Phase 9+ (compressed
mutation) is the remaining subsystem.  Test fixtures use fitsio for
normal-path round-trips and hand-crafted bytes for synthetic
fallback-column cases (astropy is also in the env for richer
cases).

Implementation lives in `src/hdu_image_compressed.rs` (the
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
ImageHDU's data-access methods are overridden:
- `read` → the Phase 2 decoder (now Phase 3, cache-aware).
- `__getitem__` → Phase 3's slice path.
- `write` / `extend` / `__setitem__` → `NotImplementedError`
  until compressed writes (Phase 7+).  These overrides have
  `#[allow(unused_variables)]` because the bodies don't use
  their parameters yet — remove when the methods get real
  implementations.

Phase 2 follow-ups (not blocking):
- **Add CompressedImageHDU cases to tests/test_repr.py** —
  the Phase 1 smoke test in test_compressed_image_phase1.py
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
`tests/test_compressed_image_phase4_gzip.py` build minimal
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

*Tests.*  `tests/test_compressed_image_phase6_hcompress.py`
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

*Tests.*  `tests/test_compressed_image_phase6_plio.py` covers
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

*Tests.*  `tests/test_compressed_image_phase7_gzip_write.py` (GZIP_1),
`tests/test_compressed_image_phase7_gzip2_write.py` (GZIP_2), and
`tests/test_compressed_image_phase7_rice_write.py` (RICE_1) —
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
`tests/test_compressed_image_phase7_hcompress_write.py` covers
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

*PLIO_1 tests.*  `tests/test_compressed_image_phase7_plio_write.py`
covers accessors after create (including TFORM1='1PI'
verification), dtype matrix (u1/i2/i4), shape matrix, default
tile shape, degenerate cases (all-zeros, all-solid, large-pv,
sparse single-pixel), **byte-exact heap agreement with fitsio**
across mask-style and degenerate inputs, bidirectional fitsio
cross-check, non-last HDU growth, mixed-algorithm file
(Plio1 + Gzip2), and rejection paths (float, i8, unsigned
trick, negative pixels, values > 2^27, shape mismatch, start
kwarg).

**Phase 7 follow-ups (deferred).**  Cold-pickup notes for each
remaining encoder.  The shape of the work is well established
by the three shipped algorithms: each adds an `encode_*`
function in the algorithm module, an algorithm class in
`compression_config.rs`, a variant in the `CompressionConfigKind`
enum in `fits.rs` (touching `tile_shape()` / `heap_format()` /
`zcmptype()` / `extra_z_cards(bitpix)`), and an arm in the
encode dispatch in `src/zimage/mod.rs::encode_tile_from_bytes`.
Before starting any of them, work the same two judgment calls
we worked for RICE:
1.  **i8 (BYTEPIX=8) interop**: does the corresponding cfitsio
    encoder exist for 64-bit pixels?  If no, reject (matches
    Rice1's call).  If yes, implement.  See `<cfitsio>/...` paths
    below.
2.  **Bit-exact vs algorithmic agreement**: port cfitsio's
    arithmetic exactly so heap bytes match byte-for-byte
    (matches Rice1 + GZIP/PLIO/HCOMPRESS reads).  Makes testing
    trivial via fitsio cross-write + heap-byte diff.

- **Unsigned-int trick on write (i1/u2/u4/u8)** — reverse the
  XOR view-cast before encoding; emit `BSCALE=1, BZERO=2^(n-1)`
  cards.  Symmetric with `create_image_hdu`'s existing handling
  for uncompressed HDUs.  All shipped encoders (GZIP_1/2, RICE_1)
  currently reject these dtypes upfront in
  `create_compressed_image_hdu_impl` (now also PLIO_1 and
  HCOMPRESS_1 — all five algorithms share the rejection); the
  branch tests `bzero.is_some()`, so the fix lives there.

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
fixed bscale = -level; 0 = no quantization, all tiles route to
GZIP fallback), `method` (`'none'` / `'no_dither'` /
`'dither1'` / `'dither2'`; default `'dither1'` matches cfitsio),
`seed` (ZDITHER0; 0 → on-disk default of 1).  Integer HDUs
reject `quantize=` with a clear error; float HDUs without
`quantize=` use the default `Quantize()`.  Pythonic kebab-style
strings for method names; FITS-spec ZQUANTIZ values are
available via `quantize.zquantiz`.

*Schema.*  Float HDUs emit a 4-column BINTABLE:

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

*Tests (`test_compressed_image_phase8_quantize_write.py`).*  29
cases covering: schema (TFORM / TTYPE / ZQUANTIZ / ZDITHER0 /
ZBLANK cards), round-trip across f4/f8 + (NO_DITHER, DITHER_1,
DITHER_2), default Quantize defaults, fitsio cross-read agreement
across the (algorithm, method) matrix, DITHER_2 exact-zero
preservation, NaN round-trip for all three methods, GZIP
fallback on constant input, ZBLANK absent for integer HDUs,
seed=0 → on-disk default of 1, rejection paths (integer dtype,
no compress).  Plus 18 Rust-side unit tests anchoring the noise
estimator + per-pixel quantize against round-trip math.

*Known follow-up: refactor `write_compressed_image_data`.*  The
function grew unwieldy in Phase 8 commit 2 (int and float
branches inline, embedded TileRow struct).  Captured in the
function's REFACTOR TODO comment and in the project's task list:
hoist TileRow to a module-level struct, extract
`encode_tile_int` / `encode_tile_float` helpers each returning
`(TileRow, primary_bytes, fallback_bytes)`, reduce the main
dispatcher to "loop + call helper + accumulate + write
descriptors".  Land on top of the commit 3 dither-matrix tests
so behavior is anchored before the refactor.

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

*Pending follow-up: implement `Quantize(method='none')` write
path.*  The kwarg is currently accepted at create time and
writes `ZQUANTIZ='NONE'` to the header, but `.write(data)`
raises `NotImplementedError` (see `hdu_image_compressed.rs`
near the `compressed-float write with ZQUANTIZ='NONE'`
message).  The intent: every tile stored losslessly as GZIP_1
compressed raw float bytes, no quantization at all.

**Design (settled 2026-05-23 after empirical astropy
investigation).**

Standards landscape (corrected from prior CLAUDE.md notes —
astropy 7.2.0's actual behavior was different):

|  | FITS spec | cfitsio | astropy (7.2.0) |
|--|--|--|--|
| ZQUANTIZ for unquantized | optional; absent → defaults to `NO_DITHER` | absent OR `'NO_DITHER'` | `'NO_DITHER'` |
| ZCMPTYPE | mandatory; must name algorithm | reflects user's choice | reflects user's choice (with quantize_level=0 → GZIP_1 only) |
| Schema for unquantized GZIP | spec doesn't mandate | n/a (typically quantized) | single COMPRESSED_DATA column with raw GZIP'd float bytes |
| `'NONE'` as ZQUANTIZ value | NOT in spec | tolerated on read | not emitted |

Empirical findings from astropy 7.2.0 with `quantize_level=0,
compression_type='GZIP_1'`:
- ZQUANTIZ='NO_DITHER' on disk (NOT 'NONE' as older notes
  suggested; NOT absent either).
- Single COMPRESSED_DATA column (1PB) with GZIP-compressed
  raw float bytes per tile.
- No ZSCALE, no ZZERO, no GZIP_COMPRESSED_DATA fallback.
- Astropy reads bit-exact with ZQUANTIZ absent OR present.

**Decisions:**

1.  **ZCMPTYPE.**  Require `compress=Gzip1(...)`; reject
    other algorithms with a clear error pointing at Gzip1.
    ZCMPTYPE='GZIP_1' on disk.  Honest header (every tile
    really is GZIP_1-compressed); user is forced to think
    about the relationship.

2.  **ZQUANTIZ.**  Omit entirely.  FITS Tile Compression
    Convention says ZQUANTIZ is optional and defaults to
    NO_DITHER when absent; both astropy and cfitsio read
    omitted ZQUANTIZ correctly.  No reason to introduce the
    non-spec 'NONE' value.

3.  **Schema.**  Single COMPRESSED_DATA column (1PB for
    heap_format='P', 1QB for 'Q') containing GZIP-compressed
    raw float bytes per tile.  No ZSCALE, no ZZERO, no
    GZIP_COMPRESSED_DATA fallback.  Matches astropy's layout
    exactly; smallest files; simplest dispatch (the existing
    reader code already handles this case — `find_data_columns`
    returns just `primary`, `quant` evaluates to `None` because
    ZSCALE/ZZERO columns are absent, decoder is called with
    float bytepix, no dequant applied).

4.  **Skip ZDITHER0 + ZBLANK.**  No dither stream is in use;
    no quantized-NaN sentinel to mark.  Both keywords absent.

**Implementation sketch:**

  - `fits.rs::create_compressed_image_hdu_impl`: when
    `quantize_cfg.method == None_`, validate
    `matches!(cfg, CompressionConfigKind::Gzip1(_))` (clear
    error otherwise); emit single-column schema
    (TFIELDS=1, TFORM1=1PB/1QB, TTYPE1=COMPRESSED_DATA);
    skip ZQUANTIZ + ZDITHER0 + ZBLANK emission.
  - `hdu_image_compressed.rs::encode_tile_float`: when
    `method == None_`, skip `quantize_double/float` entirely;
    GZIP_1-encode the raw float bytes and place directly in
    the primary COMPRESSED_DATA descriptor (NOT routed through
    the GZIP fallback column).  Different code path from the
    quantized-with-fallback case.
  - Replace the current `NotImplementedError` in
    `write_compressed_image_data` with the dispatch above.
  - Tests: round-trip across f4/f8, astropy cross-read
    agreement (rustfits writes → astropy reads bit-exact and
    vice versa), fitsio cross-read agreement, confirm
    ZQUANTIZ + ZDITHER0 + ZBLANK absent, confirm TFIELDS=1
    + COMPRESSED_DATA is the only column.

**Phase 9+ — Mutation.**  `CompressedImageHDU.extend` and
`__setitem__`.  Changing a single pixel requires re-encoding the
affected tile and possibly re-laying out the heap (same problem
as VLA-table `__setitem__`, which we explicitly deferred for
the same reason).  Wait for a real use case.

When write phases beyond 7 land, **remove the
`#[allow(unused_variables)]` on `extend` and `__setitem__` in
`hdu_image_compressed.rs`** — Phase 2 set them because the
bodies just raise NotImplementedError; real implementations
will consume those parameters.

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
`tests/test_compressed_image_phase7_rice_write.py`.

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
(`tests/test_compressed_image_phase1.py`'s
`test_other_compression_types_dispatched` was the first to
abort) when the time comes.

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

## Performance TODO — ZIMAGE chunked-read profiling

**Observation (May 2026, post-Phase-5):** reading a 1.49-billion-
pixel f8 GZIP_2 ZIMAGE file (~12 GB; per-user fixture, not
committed — see "Performance / large fixture files" under
"Build / dev workflow" for how to obtain or synthesize an
equivalent) in chunks via `f[1][lo:hi]` slicing is **roughly 3×
slower than cfitsio's equivalent read**.  The output is bit-
exact; this is purely a throughput gap.

No profiling done yet — this is just the baseline observation
from the read-correctness work.  When we revisit, candidate
suspects to rule in / out:

- **Per-tile file lock overhead** — `fetch_tile_payload_and_quant`
  acquires the file mutex for each tile descriptor read; under
  chunked reads that's one acquire per tile in the slice range.
  cfitsio uses a single FILE* and no lock.
- **Per-tile cache mutex** — `TileCache::get` / `put` lock the
  inner Mutex twice per tile.  Briefly held, but the count
  adds up on long slices.
- **`GzDecoder` allocation per tile** — `flate2`'s decoder
  is constructed fresh per call.  cfitsio reuses a single
  `z_stream` across tiles in the inner loop.
- **`place_tile_bytes_into_output` Python round-trips** — for
  each tile we go Rust → `PyBytes::new` → numpy `frombuffer` →
  `reshape` → `set_item` on the output ndarray.  That's several
  PyO3 calls per tile; cfitsio writes directly into the output
  buffer with a memcpy.
- **`PyBytes::new` copy** — `frombuffer` views the PyBytes, but
  `PyBytes::new` itself copies the Rust `Vec<u8>` into a new
  Python-owned buffer.  A `PyArray` constructor that takes
  ownership of the Vec would skip this copy.
- **No I/O pipelining** — tiles are decoded strictly in order
  with sync reads.  cfitsio also does sync reads but with a
  warm OS page cache the kernel prefetch helps; we may be
  paying for cold-cache lookups on first read.

When the time comes: pick the 12 GB GZIP_2 file as the
benchmark target, use `cargo flamegraph` or `perf` for a
profile of the slice path, compare against cfitsio's
`imcomp_decompress_tile` hot loop.  The user-facing target is
"competitive with cfitsio for typical chunked reads" — not
necessarily faster, just not dramatically slower.

Do **not** chase this incrementally during ZIMAGE Phase 6+
feature work — fold it in with the post-feature coverage
sweep, since both want a quiet codebase to profile.
