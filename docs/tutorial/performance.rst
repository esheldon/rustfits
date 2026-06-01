Performance
===========

.. role:: perf-fast
.. role:: perf-par
.. role:: perf-slow

rustfits aims to be **as fast or faster than fitsio** on every
benchmark it shares with that library, plus offer capabilities
(bounded-memory extend builds, transparent ZTABLE read/write) that
fitsio doesn't have.  This page is a snapshot of where things stand.

Ratios in the tables below are ``fitsio_time / rustfits_time`` —
greater than 1.0 means rustfits is faster.  Cells are colorized so
the overall pattern is visible at a glance:
:perf-fast:`green` for ≥ 1.10× faster, :perf-par:`yellow` for
~par (0.95–1.10×), :perf-slow:`red` for slower (< 0.95×).

Numbers below are point-in-time measurements from one Linux x86_64
machine.  They illustrate the rough shape of the comparison
(rustfits typically 1.0×–2.5× cfitsio on common workloads, with a
few cases at 30×+ where structural choices give a large win).  Your
mileage will vary with CPU, disk, file size, and data content —
re-run the benchmarks yourself for numbers that match your hardware:

.. code-block:: shell

   # release build required for representative timings
   maturin develop --release

   # full sweep (~5 minutes; --skip extend to skip the RSS benches)
   python perf/perf-all.py --skip extend

The benchmark scripts live under ``perf/`` in the source tree;
each is a standalone script with a docstring explaining its
methodology.  The runner ``perf-all.py`` collects every script's
results into the two summary tables below.  Refresh both tables
with::

   python perf/perf-all.py \
       --rst-out-xtool docs/tutorial/_perf_tables_xtool.rst \
       --rst-out-self  docs/tutorial/_perf_tables_self.rst

How comparisons are timed
-------------------------

The conventions every ``perf-*.py`` script follows so the numbers
mean what they claim:

* **Release build.**  Debug builds read ~7× slower than release and
  produce misleading "rustfits is slower" results.  ``maturin
  develop --release`` is required.
* **Fresh open per timed iteration.**  fitsio caches decoded
  compressed tiles forever; timing repeated reads of one open
  handle would measure fitsio's cache hits against rustfits's
  bounded LRU re-decoding (backward from the real workload, which
  reads each tile once).  Both tools get a fresh open per iter so
  caches start empty.
* **Warmup primes the FS cache.**  The first read off disk is
  I/O-bound and washes out the comparison; a warmup pass loads the
  compressed bytes into the OS page cache so timed passes measure
  decoder + Python boundary speed, not disk.
* **Fresh FILE per timed iter for write benches.**  Overwriting a
  large file in a tight loop triggers a kernel page-cache penalty
  that masks the realistic single-file-per-program-run pattern.
  Write benches generate a unique filename per iter via
  ``h.fresh_path``.
* **Median of 5.**  GC is paused around each timed call; samples
  are sorted and the median is reported.

What's measured
---------------

* **Cross-tool comparisons** — rustfits vs fitsio for every
  benchmark fitsio's Python API supports.  Each row is one
  (script, operation) pair; the ``vs fitsio`` column is
  ``fitsio_time / rustfits_time`` (so > 1.0 means rustfits is
  faster).
* **Self-comparisons** — rustfits compressed vs rustfits
  uncompressed for ZTABLE read/write (fitsio's Python API does
  not decompress ZTABLE, so cross-tool isn't possible).  The
  ratio is the compression cost in time, *not* a tool comparison.
* **Build wall + peak RSS** — extend benchmarks measure
  bounded-memory incremental builds vs whole-array write-once.
  Each build runs in its own subprocess so ``ru_maxrss`` is a
  clean per-build high-water mark.

Cross-tool comparisons (rustfits vs fitsio)
-------------------------------------------

Every benchmark whose operation fitsio also implements.  Each
row's ``vs fitsio`` cell is ``fitsio_time / rustfits_time``,
colorized green / yellow / red as described above.  Tables are
grouped by script; the per-script subtitle supplies the data
shape so the operation labels make sense in isolation.

.. include:: _perf_tables_xtool.rst

Self-comparisons (rustfits self)
--------------------------------

Benchmarks that have no cross-tool equivalent — either because
fitsio's Python API can't do the operation (ZTABLE read/write)
or because the comparison is structural (incremental
``append`` / ``extend`` vs whole-array ``write``).  Numbers
here are rustfits vs rustfits, so the headline is the
trade-off (compression cost, append overhead) rather than a
tool ranking.

Incremental table builds (``append`` and the ``appending()`` context)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building a catalog by ``N/K`` calls to ``hdu.append(K rows)``
is the natural pattern for streaming pipelines (per-frame
source extraction, per-file harvest).

**TL;DR for ZTABLE streaming pipelines: use the
:meth:`CompressedTableHDU.appending` context manager.**  It
makes the append-chunk size irrelevant — every chunk size
lands within ~1-7% of the one-shot ``write_table`` cost::

    with hdu.appending():
        for batch in batches:
            hdu.append(batch)

``perf/perf-table-compressed-append-chunks.py`` (4×f4 schema,
N=100,000 rows, ZTILELEN=10,000) shows the win directly:

====================  ========  =========================  =================================
regime                  chunk    unbuffered ``append``      ``appending()`` context
====================  ========  =========================  =================================
1% of tile                100   :perf-slow:`48.91× time`   :perf-fast:`1.07× time` (46× win)
10% of tile             1,000   :perf-slow:`5.40× time`    :perf-fast:`1.00× time` (5.4× win)
50% of tile             5,000   :perf-slow:`1.53× time`    :perf-fast:`0.99× time`
exact tile             10,000    1.02× time                :perf-fast:`1.00× time`
2 tiles                20,000    1.01× time                :perf-fast:`0.99× time`
====================  ========  =========================  =================================

Without the context, ZTABLE ``append`` pays a partial-trailing-
tile decode + merge + re-encode on every call — fine when the
chunk is at least one ZTILELEN, expensive when it's smaller.
The context buffers row batches in RAM and drains them in
ZTILELEN-aligned bursts (cap at 32 MB,
``MAX_PENDING_BYTES`` in
``src/hdu_table_compressed/extending.rs``), collapsing N
partial-tile re-encodes into 1.  For a 16-byte row schema the
cap allows 2 M rows per drain — a typical streaming pipeline
does 0-1 mid-context drains plus one residual drain at
``__exit__``.

``extending()`` is a symmetric alias for ``appending()``;
``extend()`` is the same alias on the append call itself.
Strict context semantics (the same as
:meth:`CompressedImageHDU.extending`): only :meth:`append` /
:meth:`extend` is permitted inside the ``with`` block; any
:meth:`read` / ``__getitem__`` / :meth:`write` / ``__setitem__``
/ :meth:`repack` / :meth:`add_checksum` /
:meth:`verify_checksum` raises ``ValueError`` (exit the
context first).  Uncompressed ``TableHDU.append`` doesn't need
the context — no merge tax — but a no-op ``appending()`` /
``extending()`` on the uncompressed types is on the roadmap
so generic code that doesn't know the HDU type works
uniformly.

The rest of this section is the underlying behavior: how
``append`` itself performs across the four variants (the
context wraps it, so understanding the per-call cost is still
useful when ZTILELEN is small or the streaming-loop budget is
tight).

``perf/perf-table-append.py`` measures wall time + peak RSS of
the per-call ``append`` loop against the equivalent one-shot
``write_table(N rows)`` across four variants:
``{uncompressed, ZTABLE} × {fixed-only, with VLA}``.  Each
build runs in its own subprocess for a clean per-build
``ru_maxrss``.

Sample numbers from the reference machine
(N=100,000 rows, 34-column type-exhaustive catalog at
~600 B/row, ZTILELEN ≈ 16 k rows).  The first table is the
rustfits self-comparison (append vs write-once across all
four variants); the second is the rustfits-vs-fitsio cross-
tool comparison on the uncompressed variants, where pairing
is possible (fitsio's Python API cannot write ZTABLE).

In the self-comparison table, ``vs rf write-once`` reports
each row's wall time / RSS divided by the rustfits write-once
measurement for that variant; values < 1 mean the row was
faster / lighter than the one-shot rustfits write.

.. list-table:: Table append (N=100,000) — rustfits self
   :widths: 24 26 14 12 22
   :header-rows: 1

   * - variant
     - regime
     - build
     - peak RSS
     - vs rf write-once
   * - uncompressed, fixed-only
     - rustfits write-once
     - 49.6 ms
     - 163 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 63.1 ms
     - 163 MB
     - 1.27× time, ≈ RAM
   * -
     - rustfits append C=10k (K=10)
     - 50.5 ms
     - 163 MB
     - 1.02× time, ≈ RAM
   * - uncompressed, with VLA
     - rustfits write-once
     - 174.8 ms
     - 216 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 199.9 ms
     - 165 MB
     - 1.14× time, 1.3× less RAM
   * -
     - rustfits append C=10k (K=10)
     - 156.4 ms
     - 165 MB
     - 0.89× time, 1.3× less RAM
   * - ZTABLE, fixed-only
     - rustfits write-once
     - 2.03 s
     - 174 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 20.45 s
     - 163 MB
     - 10.1× time, ≈ RAM
   * -
     - rustfits append C=10k (K=10)
     - 3.73 s
     - 163 MB
     - 1.8× time, ≈ RAM
   * - ZTABLE, with VLA
     - rustfits write-once
     - 5.63 s
     - 213 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 29.70 s
     - 170 MB
     - 5.3× time, 1.3× less RAM
   * -
     - rustfits append C=10k (K=10)
     - 7.94 s
     - 173 MB
     - 1.4× time, 1.2× less RAM

Cross-tool comparison on the uncompressed variants, paired
row-by-row (``vs fitsio`` = ``fitsio_time / rustfits_time``;
> 1.0 means rustfits is faster):

.. list-table:: Table append (N=100,000) — rustfits vs fitsio (uncompressed)
   :widths: 24 26 14 14 22
   :header-rows: 1

   * - variant
     - operation
     - rustfits
     - fitsio
     - vs fitsio
   * - fixed-only
     - write-once
     - 49.6 ms
     - 166.7 ms
     - :perf-fast:`3.36×`
   * -
     - append C=1k (K=100)
     - 63.1 ms
     - 171.0 ms
     - :perf-fast:`2.71×`
   * -
     - append C=10k (K=10)
     - 50.5 ms
     - 153.9 ms
     - :perf-fast:`3.05×`
   * - with VLA
     - write-once
     - 174.8 ms
     - 495.3 ms
     - :perf-fast:`2.83×`
   * -
     - append C=1k (K=100)
     - 199.9 ms
     - **40.08 s**
     - :perf-fast:`200×` (see note)
   * -
     - append C=10k (K=10)
     - 156.4 ms
     - **32.79 s**
     - :perf-fast:`210×` (see note)

Five things to take away from the per-call ``append`` numbers
above (the ``appending()`` context wraps these, so it inherits
the wins and erases the losses):

1. **Uncompressed write-once: rustfits is ~3× fitsio.**  The
   one-shot ``write_table`` call beats fitsio's equivalent by
   3.36× on the fixed-only variant and 2.83× on the VLA
   variant.  (fitsio doesn't write ZTABLE.)

2. **Uncompressed append is essentially free.**  At chunk=10 k
   the fixed-only rustfits append loop runs within 3 % of the
   one-shot rustfits write, and the VLA append loop is
   *faster* than the one-shot write (the write-once path plans
   the whole VLA heap layout in RAM up front; the append loop
   pays it incrementally).

3. **rustfits append crushes fitsio append on VLA — ~200×.**
   On the fixed-only variant rustfits append is ~2.7–3.0×
   faster than fitsio append (in line with the write-once
   ratio).  On VLA the gap blows out to **200× faster** at
   chunk=1k and **210× faster** at chunk=10k — fitsio takes
   ~33–40 s where rustfits takes ~160–200 ms.  See the note
   below for the root cause.

4. **VLA append wins on peak RSS.**  The rustfits whole-table
   write holds ~216 MB resident; the append loop holds
   ~165 MB.  That's the bounded-memory story repeating from
   the image side — incremental write keeps live memory near
   the chunk size instead of the full output.

5. **ZTABLE ``append`` (without the context) has a chunk-size
   cliff at ZTILELEN.**  At chunk=10 k (≈ default ZTILELEN of
   ~17 k for this 588 B row width), append is only 1.8× the
   rustfits ZTABLE write-once for fixed-only and 1.4× for VLA.
   At chunk=1 k (well below ZTILELEN), the partial-last-tile
   re-encode tax kicks in: every append decompresses + merges
   into the trailing tile and re-encodes it, costing 10× for
   fixed-only and 5× for VLA.  The fix is the ``appending()``
   context (lead of this section); the alternative is to pick
   chunks that are an integer multiple of ZTILELEN.

   .. note::

      The 1.8× / 1.4× / 10× / 5× numbers above reflect a
      2026-05-31 fix in ``create_table_hdu``: prior to that
      commit, the ``nrows=0`` streaming-create pattern
      (``create_table_hdu(nrows=0, compress=True[, ztilelen=K])``
      + repeated ``append(chunk)``) silently collapsed
      ZTILELEN to 1 (regardless of what the user passed),
      forcing every appended row into its own
      single-row-per-tile, independently-gzip-compressed tile.
      The pre-fix numbers were 14–15× for fixed-only and 6–7×
      for VLA at *any* chunk size, since the bug masked the
      true per-call cost.  Resolved in CLAUDE.md TODO #10
      (commit 957f94a).

.. note::

   **Known fitsio issue behind the 200× VLA-append gap.**
   fitsio's ``write_var_column`` Python wrapper calls
   ``fits_flush_file`` after every per-column write
   (`fitsio_pywrap.c
   <https://github.com/esheldon/fitsio/blob/master/fitsio/fitsio_pywrap.c>`_,
   line 2710), and cfitsio's ``fits_flush_file`` (``ffflus``
   in ``buffers.c``) is *close-current-HDU + flush-buffers +
   re-open-current-HDU*.  Each reopen re-walks the header
   and re-parses column descriptors.  With 3 VLA columns and
   100 appends at chunk=1k that's 300 close-and-reopen
   cycles — about 40 s of overhead unrelated to the actual
   data write.  The fix is in fitsio (the underlying cfitsio
   ``fits_write_col`` doesn't need the flush); the gap will
   close once that wrapper is patched.

2-D image extend — uncompressed mosaic build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mosaic / strip-build pattern: append per-detector frames
(or per-night strips) to a growing image via
``hdu.extend(strip)`` (rustfits) or
``hdu.write(strip, start=(row, 0))`` (fitsio).  Both grow the
slowest-varying axis — same primitive, two APIs.  Unlike
ZTABLE/ZIMAGE, fitsio CAN extend uncompressed images, so this
is a true cross-tool comparison.
``perf/perf-image-extend-2d.py`` measures wall + peak RSS for
write-once vs extend at two chunk sizes (100 rows and 1000
rows) on a (20,000 × 4,000) ``f4`` image (~320 MB).

.. list-table:: Uncompressed 2-D image extend (20 k × 4 k f4) — N=320 MB
   :widths: 38 12 12 22
   :header-rows: 1

   * - regime
     - build
     - peak RSS
     - vs rf write-once
   * - fitsio write-once
     - 174.7 ms
     - 440 MB
     - :perf-fast:`1.48× time, 1.2× more RAM` (fitsio)
   * - rustfits write-once
     - 117.9 ms
     - 363 MB
     - (ref)
   * - rustfits extend C=100 rows (K=200)
     - 119.0 ms
     - 70 MB
     - 1.01× time, 5.2× less RAM
   * - fitsio extend C=100 rows (K=200)
     - 231.4 ms
     - 70 MB
     - :perf-fast:`1.96× time` (fitsio), 5.2× less RAM
   * - rustfits extend C=1000 rows (K=20)
     - 117.1 ms
     - 74 MB
     - 0.99× time, 4.9× less RAM
   * - fitsio extend C=1000 rows (K=20)
     - 239.7 ms
     - 74 MB
     - :perf-fast:`2.03× time` (fitsio), 4.9× less RAM

Three takeaways:

1. **Bounded memory works in both tools.**  Either ``extend``
   path keeps peak RSS at ~70–75 MB regardless of the final
   image size (dominated by Python + numpy baseline; the
   actual chunk's data is only 1.6–16 MB).  write-once needs
   the whole image resident, plus per-tool overhead.  Both
   tools deliver the same ~5× RAM win.

2. **rustfits extend ≈ rustfits write-once on time.**  Both
   chunk sizes match the write-once baseline within 2 %.  No
   per-call overhead worth worrying about — for uncompressed
   incremental builds the only cost vs write-once is the
   bookkeeping for the NAXIS2 header update.

3. **fitsio extend is ~2× slower than rustfits extend.**  At
   both chunk sizes fitsio's per-extend cost is roughly double
   rustfits' — same ratio as the write-once gap (1.48×), so
   this is a general fitsio per-call overhead rather than
   something specific to ``start=`` writes.  fitsio also holds
   ~80 MB more RAM during write-once (440 vs 363 MB) —
   suggesting it keeps an extra byteswapped copy that rustfits
   avoids.

2-D compressed image extend — mosaic build with GZIP_2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same shape as the uncompressed bench (20,000 × 4,000 f4,
~320 MB raw) but tile-compressed with GZIP_2 and tile shape
``(100, cols)`` so the chunk-row axis maps directly to
"tile-rows per append".  fitsio cannot extend a compressed
image (cfitsio returns ``status = 107: tried to move past end
of file`` on the second write), so the extend rows are
rustfits-only; fitsio appears only as a write-once reference.
``perf/perf-compressed-image-extend-2d.py``.

.. list-table:: Compressed 2-D image extend (20 k × 4 k f4, GZIP_2 tile=(100, cols))
   :widths: 52 12 12 22
   :header-rows: 1

   * - regime
     - build
     - peak RSS
     - vs rf write-once
   * - fitsio write-once
     - 6.44 s
     - 440 MB
     - :perf-fast:`2.98× time` (fitsio)
   * - rustfits write-once
     - 2.17 s
     - 626 MB
     - (ref)
   * - rustfits extend C=50 rows (K=400, sub-tile)
     - 48.30 s
     - 454 MB
     - :perf-slow:`22.3× time`, 1.4× less RAM
   * - rustfits extending() C=50 rows (K=400, sub-tile)
     - 2.73 s
     - 364 MB
     - :perf-fast:`1.26× time, 1.7× less RAM`
   * - rustfits extend C=100 rows (K=200, exact tile)
     - 16.68 s
     - 321 MB
     - :perf-slow:`7.7× time`, 1.9× less RAM
   * - rustfits extending() C=100 rows (K=200, exact tile)
     - 2.74 s
     - 365 MB
     - :perf-fast:`1.26× time, 1.7× less RAM`
   * - rustfits extend C=1000 rows (K=20, 10 tiles)
     - 3.34 s
     - 338 MB
     - :perf-slow:`1.54× time`, 1.8× less RAM
   * - rustfits extending() C=1000 rows (K=20, 10 tiles)
     - 2.49 s
     - 394 MB
     - :perf-fast:`1.15× time, 1.6× less RAM`

Four takeaways:

1. **Multi-tile chunks are nearly free.**  At chunk=1000 rows
   (10 tiles per call) extend is 1.54× write-once — the
   per-extend overhead is one heap-relocate-forward and a
   PCOUNT bump, amortized across the 10 tiles' encode work.
   For mosaic builds that can buffer multi-tile strips, this
   is the regime to aim for.

2. **Exact-tile chunks are moderate.**  At chunk=100 rows (1
   tile per call) extend costs 7.7× write-once — the per-call
   overhead dominates because there's only one tile's worth
   of "real" encode work per call but the same bookkeeping.

3. **Sub-tile chunks pay heavily.**  At chunk=50 rows (½ tile)
   every append decompresses + merges into the partial last
   tile then re-encodes it: 22.3× write-once — the same
   mirror-pattern as the ZTABLE small-chunk re-encode finding
   in the table-append section above.  For compressed 2-D
   mosaic builds, **align chunks to a multiple of tile-rows**
   (or buffer to that size in user code) to skip the
   re-encode tax — or use the ``extending()`` context (takeaway
   4) to do the buffering automatically.

4. **The ``extending()`` context manager wins at every
   chunk size.**  Wrapping the extend loop in ``with
   hdu.extending():`` buffers chunks in RAM and drains in
   tile-aligned bursts (triggered by a 32 MB soft cap),
   collapsing N partial-tile re-encodes into 1::

       with hdu.extending():
           for batch in batches:
               hdu.extend(batch)

   At sub-tile chunks the speedup is dramatic: 22.3× → 1.26×
   write-once (18× faster than the unbuffered extend) AND
   peak RSS drops to 364 MB — 1.7× *less* RAM than write-once
   because the cap bounds the buffer below what
   write-once's whole-array path holds.  Even at exact-tile
   chunks the context helps (7.7× → 1.26×).  At multi-tile
   chunks it's a modest win (1.54× → 1.15×).

   **Memory cap.**  The mid-context drain is triggered at
   ``MAX_PENDING_BYTES = 32 MB`` of accumulated input,
   chosen as a soft cap that fits many tiles per drain
   while staying small enough to not balloon RSS during long
   streaming loops.  Sized as a code constant; not currently
   user-configurable — if a workload needs a different cap
   please open an issue.

   **Strict context semantics.**  Inside ``with
   hdu.extending():`` only :meth:`extend` is permitted; any
   :meth:`read`, ``__getitem__``, :meth:`write`, ``__setitem__``,
   :meth:`repack`, :meth:`add_checksum`, or :meth:`verify_checksum`
   call raises ``ValueError``.  Exit the context first.  The
   natural nested-``with`` pattern with ``FITS`` composes
   correctly (Python guarantees the inner ``__exit__`` runs
   before the outer ``close``), so this restriction only
   surfaces in mixed-operation loops that should be
   restructured anyway.

Scattered reads on compressed images — tune the tile cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compressed-read benches above all walk the file in chunks
of increasing size — sequential access, where the tile cache
holds whatever was most recently decoded.  Scattered /
random-access workloads behave differently: every read may
revisit a tile decoded much earlier, so cache hit rate
dominates.  ``perf/perf-compressed-image-read-1d-scattered.py``
measures 1000 random 1000-row windows against a 64-tile 1-D
``f8`` GZIP_2 image (537 MB raw, 15 MB compressed,
healsparse-like).  With random sampling across 64 tiles, every
tile is touched ~16× per iteration, so the cache size you pick
matters a lot.

The two regimes:

.. list-table:: Scattered compressed-1D read (1000 × 1k-row windows, 64-tile f8 GZIP_2)
   :widths: 40 14 14 14 18
   :header-rows: 1

   * - cache
     - rustfits
     - fitsio
     - rf vs fi
     - note
   * - default (rf=32 MB, fi=unbounded)
     - 5.34 s
     - 1.73 s
     - :perf-slow:`0.32×`
     - rf holds 4/64 tiles → thrashes
   * - large (rf=528 MB, fi=unbounded)
     - 483 ms
     - 1.74 s
     - :perf-fast:`3.60×`
     - both backends fully cache

What's going on:

* **rustfits's default cache is 32 MB (4 tiles for an 8 MB
  tile).**  On scattered access across 64 unique tiles, every
  read evicts something we still need — the workload collapses
  to "decode every read" and the headline ratio looks bad.
* **fitsio's cache is unbounded** — it caches every decoded
  tile forever, so on this small file it implicitly covers the
  whole working set with no user tuning.  The unbounded cache
  is what makes fitsio look ~3× faster than rustfits in the
  default-cache regime.
* **At equal coverage, rustfits is 3.6× faster** because the
  per-read lookup-and-slice path is leaner (same shape as the
  small-chunk wins in the sequential-read benches above).

If your access pattern is scattered, the fix is one line —
size the tile cache to the working set::

    hdu = fits[1]
    hdu.set_tile_cache_size(500 * 1024 * 1024)   # 500 MB
    for row in random_rows:
        chunk = hdu[row:row + 1000]
        ...

.. important::

   **rustfits's cache is bounded for a reason.**  fitsio's
   "remember every tile forever" works great when the
   compressed image fits in RAM (the case in this bench), but
   **on a multi-GB compressed image fitsio's unbounded cache
   will OOM** — there's no knob to bound it.  rustfits's
   bounded LRU degrades gracefully: above the cap, the per-
   tile decode cost replaces the cache hit, but the process
   keeps running.  The cache size is a memory-vs-speed knob
   you tune for your workload; there's no universally-good
   value.

Practical tuning rules of thumb (until a real workload
disagrees):

* **Sequential / chunked reads**: default 32 MB is fine —
  LRU naturally holds the few most-recent tiles, and the
  chunked-read benches already show large wins.
* **Scattered reads, file fits in RAM**: bump
  ``set_tile_cache_size`` to roughly the size of the working
  set's unique tiles (or the whole compressed-image data
  section if you can spare the RSS).
* **Scattered reads, file doesn't fit in RAM**: pick a cache
  size you can afford; the bench above shows that even at
  256 MB / 32 tiles, partial coverage halves the cost
  vs. default.

Compressed-image ``__setitem__`` — per-tile re-encode tax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``hdu[selection] = value`` on a tile-compressed image decodes
every tile the selection touches, modifies it in numpy, and
re-encodes / appends to the heap.  A "patch a few pixels"
workflow (interactive masking, bad-pixel fix-up, hot-pixel
flagging) therefore pays a full tile re-encode for every tile
the selection covers — even a single-pixel write.
``perf/perf-compressed-image-setitem.py`` measures that per-call
cost across all five algorithms + the unquantized / quantized
float paths, with an uncompressed memcpy floor for reference.

Setup: (256 × 256) image with (32, 32) tiles (an 8×8 tile grid).
Four selection shapes:

* **single pixel** — one ``int`` per axis (touches 1 tile)
* **1 tile aligned** — a 32×32 slice on a tile boundary (1 tile)
* **4 tiles spanning** — an 8×8 slice straddling a tile corner
  (touches 4 tiles)
* **16 tiles aligned** — a 4×4 block of tiles (16 tiles)

Per-call costs span 3 µs to 200 µs, which at the low end is
below scheduler / cache / IRQ jitter on a typical Linux box
(early prototypes of this bench wobbled 55% run-to-run on the
3 µs uncompressed-single-pixel row).  The bench mitigates by
auto-calibrating the inner loop size per (algo, sel): a 20-call
probe estimates per-call cost, then the timed loop is sized so
each window takes at least 100 ms.  Result: 30 000 inner calls
for the cheap rows down to 200 for the expensive ones, every
row stable to within 1% run-to-run.  GC is disabled during
each timed window; reported value is the median across 5
windows.  PCOUNT grows monotonically across iters but per-call
cost is constant (``__setitem__`` just appends new tile bytes
to the heap; it doesn't repack).

.. list-table:: Compressed-image ``__setitem__`` cost (256×256, tile 32×32)
   :widths: 26 18 7 12 12 13
   :header-rows: 1

   * - algorithm
     - selection
     - tiles
     - per call
     - per tile
     - per pixel
   * - uncompressed i4
     - single pixel
     - 1
     - 3.0 µs
     - 3.0 µs
     - 3.0 µs
   * - uncompressed i4
     - 1 tile aligned
     - 1
     - 17.5 µs
     - 17.5 µs
     - 0.0 µs
   * - uncompressed i4
     - 4 tiles spanning
     - 4
     - 6.7 µs
     - 1.7 µs
     - 0.1 µs
   * - uncompressed i4
     - 16 tiles aligned
     - 16
     - 64.8 µs
     - 4.0 µs
     - 0.0 µs
   * - GZIP_1 i4
     - single pixel
     - 1
     - 99.5 µs
     - 99.5 µs
     - 99.5 µs
   * - GZIP_1 i4
     - 1 tile aligned
     - 1
     - 37.6 µs
     - 37.6 µs
     - 0.0 µs
   * - GZIP_1 i4
     - 4 tiles spanning
     - 4
     - 361.6 µs
     - 90.4 µs
     - 5.7 µs
   * - GZIP_1 i4
     - 16 tiles aligned
     - 16
     - 401.4 µs
     - 25.1 µs
     - 0.0 µs
   * - GZIP_2 i4
     - single pixel
     - 1
     - 81.5 µs
     - 81.5 µs
     - 81.5 µs
   * - GZIP_2 i4
     - 1 tile aligned
     - 1
     - 40.8 µs
     - 40.8 µs
     - 0.0 µs
   * - GZIP_2 i4
     - 4 tiles spanning
     - 4
     - 263.0 µs
     - 65.8 µs
     - 4.1 µs
   * - GZIP_2 i4
     - 16 tiles aligned
     - 16
     - 452.4 µs
     - 28.3 µs
     - 0.0 µs
   * - RICE_1 i4
     - single pixel
     - 1
     - 39.3 µs
     - 39.3 µs
     - 39.3 µs
   * - RICE_1 i4
     - 1 tile aligned
     - 1
     - 22.8 µs
     - 22.8 µs
     - 0.0 µs
   * - RICE_1 i4
     - 4 tiles spanning
     - 4
     - 103.2 µs
     - 25.8 µs
     - 1.6 µs
   * - RICE_1 i4
     - 16 tiles aligned
     - 16
     - 139.4 µs
     - 8.7 µs
     - 0.0 µs
   * - HCOMPRESS_1 i4
     - single pixel
     - 1
     - 139.8 µs
     - 139.8 µs
     - 139.8 µs
   * - HCOMPRESS_1 i4
     - 1 tile aligned
     - 1
     - 31.0 µs
     - 31.0 µs
     - 0.0 µs
   * - HCOMPRESS_1 i4
     - 4 tiles spanning
     - 4
     - 461.4 µs
     - 115.4 µs
     - 7.2 µs
   * - HCOMPRESS_1 i4
     - 16 tiles aligned
     - 16
     - 275.3 µs
     - 17.2 µs
     - 0.0 µs
   * - PLIO_1 i4
     - single pixel
     - 1
     - 18.8 µs
     - 18.8 µs
     - 18.8 µs
   * - PLIO_1 i4
     - 1 tile aligned
     - 1
     - 20.1 µs
     - 20.1 µs
     - 0.0 µs
   * - PLIO_1 i4
     - 4 tiles spanning
     - 4
     - 40.5 µs
     - 10.1 µs
     - 0.6 µs
   * - PLIO_1 i4
     - 16 tiles aligned
     - 16
     - 124.2 µs
     - 7.8 µs
     - 0.0 µs
   * - GZIP_1 f4 unquantized
     - single pixel
     - 1
     - 93.2 µs
     - 93.2 µs
     - 93.2 µs
   * - GZIP_1 f4 unquantized
     - 1 tile aligned
     - 1
     - 37.4 µs
     - 37.4 µs
     - 0.0 µs
   * - GZIP_1 f4 unquantized
     - 4 tiles spanning
     - 4
     - 281.3 µs
     - 70.4 µs
     - 4.4 µs
   * - GZIP_1 f4 unquantized
     - 16 tiles aligned
     - 16
     - 402.6 µs
     - 25.2 µs
     - 0.0 µs
   * - GZIP_1 f4 quantized
     - single pixel
     - 1
     - 195.4 µs
     - 195.4 µs
     - 195.4 µs
   * - GZIP_1 f4 quantized
     - 1 tile aligned
     - 1
     - 51.2 µs
     - 51.2 µs
     - 0.0 µs
   * - GZIP_1 f4 quantized
     - 4 tiles spanning
     - 4
     - 680.7 µs
     - 170.2 µs
     - 10.6 µs
   * - GZIP_1 f4 quantized
     - 16 tiles aligned
     - 16
     - 560.9 µs
     - 35.1 µs
     - 0.0 µs

Four takeaways:

1. **Per-call cost is dominated by per-tile re-encode.**  A
   single-pixel write costs 19–200 µs depending on algorithm —
   30–60× the uncompressed memcpy (3 µs).  The user's budget
   for "patch one pixel" is the per-tile cost of their chosen
   algorithm; touching N tiles costs roughly N × the per-tile
   rate plus a small constant.

2. **PLIO_1 is fastest, quantized float is slowest.**  At one
   tile touched, PLIO costs ~20 µs / tile (it's encoded as
   ``(value, run-length)`` pairs so decode + re-encode are
   trivial); quantized f4 costs ~200 µs / tile because every
   touched tile has to re-quantize against its existing
   per-tile bscale/bzero/seed.  RICE_1 and HCOMPRESS_1 fall
   between (40–145 µs single-pixel).

3. **Full-tile-aligned writes are CHEAPER than single-pixel
   writes.**  Counterintuitive but real: a "1 tile aligned"
   selection that covers the whole tile (32×32) is 20–50 µs
   across all algorithms, where a single-pixel write on the
   same tile is 40–200 µs.  The reason is that an aligned
   full-tile write doesn't need to decode the existing tile
   first — every pixel is being replaced, so the
   read-modify-write collapses to just write.  Single-pixel
   writes pay the full decode + modify + encode cycle.

4. **Per-pixel rate amortizes with selection size.**  For
   batched writes (the 16-tile column) the per-pixel rate
   drops to <0.1 µs / pixel — comparable to uncompressed.  If
   you can buffer your patches into rectangular tile-aligned
   blocks, the per-tile re-encode tax amortizes away.  For
   true scattered single-pixel workflows there's no shortcut
   (use PLIO_1 if your data fits its non-negative integer
   constraint).

Practical guidance: for interactive masking on a compressed
image, budget ~50–200 µs per single-pixel touch (algorithm-
dependent), so up to a few thousand single-pixel patches per
second.  For bulk masking, group patches into rectangular
selections that cover whole tiles where possible.

``repack()`` — peak RSS scaling at large heap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three repack flavors clean up heap orphans accumulated by
``__setitem__`` / ``extend`` calls:

* ``TableHDU.repack()`` for BINTABLE VLA columns,
* ``CompressedImageHDU.repack()`` for ZIMAGE compressed images,
* ``CompressedTableHDU.repack()`` for ZTABLE compressed tables.

They share a goal (rebuild the heap with only live bytes, shrink
the file) but the implementations differ.  This bench measures the
peak RSS at 10 / 100 / 1000 MB heap sizes for each.  Each fixture
is built in the parent process; the repack runs in a fresh
subprocess so the reported peak RSS is the repack alone, read via
``/proc/self/status:VmHWM`` (the kernel's per-task high-water).
Using ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` here would
silently report the parent's peak, because on Linux ``ru_maxrss``
is accumulated in ``signal_struct`` and inherited across
``fork+exec``; the shared ``h.vm_hwm_kb`` helper is the immune
path.

.. list-table:: ``repack()`` peak RSS vs PCOUNT
   :widths: 36 12 12 14 16
   :header-rows: 1

   * - flavor
     - PCOUNT
     - repack t
     - peak RSS
     - notes
   * - BINTABLE VLA (uncompressed)
     - 10 MB
     - 6.8 ms
     - 59 MB
     - whole-heap-into-RAM impl
   * - BINTABLE VLA
     - 100 MB
     - 51.2 ms
     - 149 MB
     - linear: baseline + heap
   * - BINTABLE VLA
     - 1,000 MB
     - 451.3 ms
     - 1,049 MB
     - 1.05× PCOUNT
   * - ZIMAGE compressed image
     - 11 MB
     - 8.3 ms
     - 64 MB
     - whole-heap-into-RAM impl
   * - ZIMAGE compressed image
     - 119 MB
     - 83.3 ms
     - 219 MB
     - 1.5× PCOUNT (live+orphan coexist)
   * - ZIMAGE compressed image
     - 1,215 MB
     - 818.7 ms
     - 1,787 MB
     - 1.54× PCOUNT
   * - ZTABLE compressed table
     - 7 MB
     - 1.8 ms
     - 50 MB
     - streaming + staging impl
   * - ZTABLE compressed table
     - 102 MB
     - 18.3 ms
     - **50 MB**
     - **flat** — no PCOUNT scaling
   * - ZTABLE compressed table
     - 1,049 MB
     - 160.4 ms
     - **50 MB**
     - **flat** at 1 GB heap

Three takeaways:

1. **ZTABLE repack is the standout.**  RSS is constant at
   ~50 MB (Python + rustfits baseline) regardless of heap size,
   from 10 MB to 1 GB.  The streaming + staging implementation
   (a ~1 MiB chunk plus the descriptor table; documented in
   CLAUDE.md under "Heap repack") pays off — even on a 1 GB
   compressed heap the working set is small enough to run on
   memory-constrained machines.  Time scales linearly at
   ~6 GB/s effective heap throughput.

2. **The uncompressed BINTABLE VLA path is whole-heap-into-RAM.**
   Repack reads the entire old heap into a ``Vec<u8>``, walks
   live descriptors copying small live cells to a new ``Vec``,
   then writes back.  Peak RSS = baseline + old_heap, so RSS
   scales 1:1 with PCOUNT.  For a 1 GB heap, plan on 1 GB of
   working memory plus baseline.  In this bench live ≈ 40 KB
   so new_heap is negligible; a workload with substantial
   live data would push RSS up by that much more.

3. **The ZIMAGE compressed-image path is also whole-heap-into-
   RAM, with a 1.5× peak.**  Same shape as BINTABLE VLA, but
   when ``__setitem__`` overwrites entire tiles (the natural
   orphan pattern), the live half and orphan half coexist in
   RAM during the copy loop — old_heap + new_heap together
   peak around 1.5× PCOUNT.  For a 1 GB heap, plan on ~1.5 GB.

The bounded-memory ZTABLE implementation predates the others;
the uncompressed and ZIMAGE paths could be rewritten with the
same streaming + staging approach but haven't been (no real
user has hit the wall yet).  When they are, this bench will
show the win directly.

Other self-comparisons + RSS benches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZTABLE read/write (rustfits compressed vs rustfits
uncompressed) and the image-extend RSS benches.

.. include:: _perf_tables_self.rst
