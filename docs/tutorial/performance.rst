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
results into the two summary tables below.

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

.. include:: _perf_tables.rst
