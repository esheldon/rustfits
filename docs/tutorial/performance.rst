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

fitsio appears as a write-once reference on the uncompressed
variants only (its Python API cannot write ZTABLE).  ``vs rf
write-once`` reports the append loop's measurement divided by
the rustfits write-once measurement (so values < 1 mean the
append loop was faster than the one-shot write); fitsio rows
show the fitsio/rustfits ratio in the same column.

Sample numbers from the reference machine
(N=100,000 rows, 34-column type-exhaustive catalog at
~600 B/row, ZTILELEN ≈ 16 k rows):

.. list-table:: Table append (N=100,000) — wall + peak RSS
   :widths: 24 24 12 12 18
   :header-rows: 1

   * - variant
     - regime
     - build
     - peak RSS
     - vs rf write-once
   * - uncompressed, fixed-only
     - fitsio write-once
     - 164.3 ms
     - 163 MB
     - :perf-slow:`3.35× time` (fitsio)
   * -
     - rustfits write-once
     - 49.1 ms
     - 163 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 62.6 ms
     - 163 MB
     - :perf-par:`1.27× time, 1.0× RAM`
   * -
     - rustfits append C=10k (K=10)
     - 50.5 ms
     - 163 MB
     - :perf-par:`1.03× time, 1.0× RAM`
   * - uncompressed, with VLA
     - fitsio write-once
     - 488.5 ms
     - 186 MB
     - :perf-slow:`2.87× time` (fitsio)
   * -
     - rustfits write-once
     - 170.5 ms
     - 216 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 203.1 ms
     - 166 MB
     - :perf-fast:`1.19× time, 1.3× less RAM`
   * -
     - rustfits append C=10k (K=10)
     - 157.8 ms
     - 165 MB
     - :perf-fast:`0.93× time, 1.3× less RAM`
   * - ZTABLE, fixed-only
     - rustfits write-once
     - 2.02 s
     - 174 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 31.08 s
     - 187 MB
     - :perf-slow:`15.4× time, 0.9× RAM`
   * -
     - rustfits append C=10k (K=10)
     - 28.85 s
     - 207 MB
     - :perf-slow:`14.3× time, 0.8× RAM`
   * - ZTABLE, with VLA
     - rustfits write-once
     - 5.61 s
     - 213 MB
     - (ref)
   * -
     - rustfits append C=1k (K=100)
     - 37.71 s
     - 229 MB
     - :perf-slow:`6.7× time, 0.9× RAM`
   * -
     - rustfits append C=10k (K=10)
     - 35.25 s
     - 228 MB
     - :perf-slow:`6.3× time, 0.9× RAM`

Four things to take away:

1. **Uncompressed write-once: rustfits is ~3× fitsio.**  The
   one-shot ``write_table`` call beats fitsio's equivalent by
   3.35× on the fixed-only variant and 2.87× on the VLA
   variant.  (The fitsio rows above are write-once only; fitsio
   doesn't write ZTABLE.)

2. **Uncompressed append is essentially free.**  At chunk=10 k
   the fixed-only append loop runs within 3 % of the one-shot
   write, and the VLA append loop is *faster* than the one-shot
   write (the write-once path plans the whole VLA heap layout
   in RAM up front; the append loop pays it incrementally).

3. **VLA append wins on peak RSS.**  The whole-table write
   holds ~216 MB resident; the append loop holds ~166 MB.
   That's the bounded-memory story repeating from the image
   side — incremental write keeps live memory near the chunk
   size instead of the full output.

4. **ZTABLE small-chunk append is expensive.**  With chunk
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

Other self-comparisons + RSS benches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZTABLE read/write (rustfits compressed vs rustfits
uncompressed) and the image-extend RSS benches.

.. include:: _perf_tables_self.rst
