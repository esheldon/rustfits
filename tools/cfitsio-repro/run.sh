#!/usr/bin/env bash
#
# Build and run the standalone cfitsio reproducers in this directory.
#
# Requires cfitsio (header + lib) and the fpack/funpack CLIs on PATH.
# In a conda env that provides cfitsio, $CONDA_PREFIX is used for the
# include/lib paths automatically.  Override with CFITSIO_PREFIX=...
#
# Usage:  bash run.sh          # run all reproducers
#         bash run.sh pa       # just the PA-VLA funpack crash (entry 2)
#         bash run.sh cplx     # just the GZIP_2 complex case (entry 4)
#         bash run.sh plio     # PLIO_1 small-image write overflow (entry 1)
#         bash run.sh sweep    # negative-result probe (entry 6): pure-C
#                              # compressed-image patch sweep runs CLEAN
#
# The `plio` reproducer overflows a buffer INSIDE libcfitsio at write
# time (no fpack/funpack step).  On glibc/Linux a plain run usually
# exits 0 (the few-byte overrun is tolerated); to make it abort, point
# it at an ASAN-instrumented cfitsio and set ASAN=1, e.g.
#
#     ASAN=1 CFITSIO_PREFIX=/path/to/asan-cfitsio bash run.sh plio
#
# where CFITSIO_PREFIX has include/ + lib/.  See README.md for the
# cfitsio-with-ASAN build recipe.  On macOS it aborts directly against
# a plain cfitsio (~50%).
#
# Each reproducer is pure cfitsio + fpack/funpack -- no Python, no
# rustfits.  See README.md for the per-bug write-up and draft cfitsio
# issue text.

set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${CFITSIO_PREFIX:-${CONDA_PREFIX:-/usr}}"
CC="${CC:-cc}"

compile() {
    "$CC" -O2 -I"$PREFIX/include" "$DIR/$1.c" -o "$DIR/$1" \
        -L"$PREFIX/lib" -lcfitsio -Wl,-rpath,"$PREFIX/lib"
}

run_pa() {
    echo "==================================================================="
    echo " #3  String-VLA (1PA) ZTABLE funpack crash  (rustfits issue #9)"
    echo "==================================================================="
    compile pa_vla_funpack_crash || return 1
    rm -f "$DIR"/pa_vla.fits "$DIR"/pa_vla.fits.fz "$DIR"/pa_vla_un.fits
    "$DIR/pa_vla_funpack_crash" "$DIR/pa_vla.fits" || return 1
    echo "--- fpack -table (cfitsio compresses the 1PA column) ---"
    fpack -table "$DIR/pa_vla.fits"
    echo "--- funpack (EXPECTED: heap corruption / SIGABRT) ---"
    funpack -O "$DIR/pa_vla_un.fits" "$DIR/pa_vla.fits.fz"
    echo "funpack exit code: $?   (134 = 128 + SIGABRT)"
}

run_cplx() {
    echo "==================================================================="
    echo " #5  GZIP_2 complex (1C/1M) column: cfitsio can't funpack its own"
    echo "==================================================================="
    compile gzip2_complex_funpack || return 1
    rm -f "$DIR"/cplx.fits "$DIR"/cplx.fits.fz "$DIR"/cplx_un.fits
    "$DIR/gzip2_complex_funpack" "$DIR/cplx.fits" || return 1
    echo "--- fpack -g2 -table (writes ZCTYP='GZIP_2' on the complex col) ---"
    fpack -g2 -table "$DIR/cplx.fits"
    echo "--- funpack (EXPECTED: status 414, unsuitable data type) ---"
    funpack -O "$DIR/cplx_un.fits" "$DIR/cplx.fits.fz"
    echo "funpack exit code: $?"
}

run_plio() {
    echo "==================================================================="
    echo " #1  PLIO_1 small-image write overflow  (cfitsio #136 / fitsio #496)"
    echo "==================================================================="
    # Optionally build with ASAN so the overflow is reported on Linux;
    # pair with CFITSIO_PREFIX pointing at an ASAN-instrumented cfitsio.
    local extra=""
    if [ -n "${ASAN:-}" ]; then
        extra="-fsanitize=address -g -fno-omit-frame-pointer"
        echo "(ASAN build: linking $PREFIX -- ensure it is ASAN-instrumented)"
    fi
    # shellcheck disable=SC2086
    "$CC" -O1 $extra -I"$PREFIX/include" \
        "$DIR/plio_small_image_overflow.c" -o "$DIR/plio_small_image_overflow" \
        -L"$PREFIX/lib" -lcfitsio -Wl,-rpath,"$PREFIX/lib" || return 1
    rm -f "$DIR"/plio_small.fits.fz
    echo "--- write 4x4 PLIO_1 image, 2x2 tiles (overflow at compress time) ---"
    "$DIR/plio_small_image_overflow" "$DIR/plio_small.fits.fz"
    echo "exit code: $?   (ASAN/Linux: abort on heap-buffer-overflow;" \
         "macOS plain: ~134 SIGABRT; glibc plain: 0, overrun tolerated)"
}

run_sweep() {
    echo "==================================================================="
    echo " entry 6  compressed-image patch sweep -- NEGATIVE RESULT probe"
    echo "          (pure cfitsio runs clean; bug is in the fitsio wrapper)"
    echo "==================================================================="
    compile setitem_sweep_corruption || return 1
    local t
    t="$(mktemp -d)"
    ( cd "$t" && "$DIR/setitem_sweep_corruption" )
    echo "exit code: $?   (0 = clean, as expected -- NOT a cfitsio bug)"
    rm -rf "$t"
}

case "${1:-all}" in
    pa)    run_pa ;;
    cplx)  run_cplx ;;
    plio)  run_plio ;;
    sweep) run_sweep ;;
    all)   run_pa; echo; run_cplx; echo; run_plio ;;
    *)     echo "usage: bash run.sh [pa|cplx|plio|sweep|all]" >&2; exit 2 ;;
esac
