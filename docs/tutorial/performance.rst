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

Incremental table builds (``append`` vs ``write_table``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building a catalog by ``N/K`` calls to ``hdu.append(K rows)``
is the natural pattern for streaming pipelines (per-frame
source extraction, per-file harvest).
``perf/perf-table-append.py`` measures wall time + peak RSS of
that pattern against the equivalent one-shot
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
     - :perf-par:`1.27× time, ≈ RAM`
   * -
     - rustfits append C=10k (K=10)
     - 50.5 ms
     - 163 MB
     - :perf-par:`1.02× time, ≈ RAM`
   * - uncompressed, with VLA
     - rustfits write-once
     - 174.8 ms
     - 216 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 199.9 ms
     - 165 MB
     - :perf-fast:`1.14× time, 1.3× less RAM`
   * -
     - rustfits append C=10k (K=10)
     - 156.4 ms
     - 165 MB
     - :perf-fast:`0.89× time, 1.3× less RAM`
   * - ZTABLE, fixed-only
     - rustfits write-once
     - 2.03 s
     - 174 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 31.07 s
     - 187 MB
     - :perf-slow:`15.3× time, ≈ RAM`
   * -
     - rustfits append C=10k (K=10)
     - 28.89 s
     - 188 MB
     - :perf-slow:`14.3× time, ≈ RAM`
   * - ZTABLE, with VLA
     - rustfits write-once
     - 5.62 s
     - 213 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 37.73 s
     - 230 MB
     - :perf-slow:`6.7× time, ≈ RAM`
   * -
     - rustfits append C=10k (K=10)
     - 35.25 s
     - 228 MB
     - :perf-slow:`6.3× time, ≈ RAM`

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

Five things to take away:

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

5. **ZTABLE small-chunk append is expensive.**  With chunk
   sizes (1 k, 10 k rows) well below the default ZTILELEN
   (~16 k rows here, set by cfitsio's
   ``max(1, min(nrows, 10 MB / row_width))`` rule), every
   append decompresses + merges into the partial last tile
   then re-encodes it — a ~14× hit for fixed-only, ~6× for VLA
   (whose write-once baseline is already higher).  For ZTABLE
   streaming pipelines, **prefer chunks ≥ ZTILELEN** so each
   append finishes a tile cleanly; for throughput-focused jobs
   that can hold the data, ``write_table`` (one shot) is the
   faster path.  Improving the small-chunk path is tracked in
   ``CLAUDE.md`` under Performance TODO #10.

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

Other self-comparisons + RSS benches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZTABLE read/write (rustfits compressed vs rustfits
uncompressed) and the image-extend RSS benches.

.. include:: _perf_tables_self.rst
