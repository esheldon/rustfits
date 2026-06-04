/*
 * Standalone cfitsio reproducer: PLIO_1 image compression overflows
 * its output buffer on very small tiles, corrupting the heap.
 *
 * No rustfits, no fitsio Python wrapper -- this writes a tiny
 * PLIO_1-compressed image with libcfitsio's own image compressor.
 * The overrun happens at WRITE time, entirely inside cfitsio
 * (imcomp_compress_tile -> pl_p2li), so there is no fpack/funpack
 * step here: the program itself is the reproducer.
 *
 * The bug (imcompress.c + pliocomp.c):
 *
 *   imcomp_calc_max_elem() sizes the per-tile compression buffer for
 *   PLIO_1 (the catch-all `else` branch) at:
 *
 *       return (nx * sizeof(int));            // BYTES
 *
 *   with no minimum -- unlike HCOMPRESS_1 just above it, which adds a
 *   "+ 26 ... only significant for very small tiles" overhead term.
 *   cfitsio then calloc's cbuf at that many BYTES and hands it to
 *   pl_p2li(), which writes the IRAF line-list as `short` (2-byte)
 *   elements with a FIXED 7-short (14-byte) header before any pixel
 *   data (lldst[1..7]; the data cursor `op` starts at 8).
 *
 *   For a 2x2 tile, nx = 4, so the buffer is 4*sizeof(int) = 16 bytes
 *   = only 8 shorts.  The 7-short header consumes shorts 1..7; the very
 *   first encoded data short (lldst[8], bytes 14..15) is the last that
 *   fits, and any real run writes lldst[9] (bytes 16..17) and beyond --
 *   past the 16-byte allocation.  Heap-buffer-overflow WRITE.
 *
 * Why it's only "sometimes" a visible crash:
 *
 *   - glibc/Linux: the few-byte overrun usually lands in slack within
 *     the malloc chunk, so a plain run exits 0 and looks clean.  Build
 *     cfitsio with -fsanitize=address and the overflow is reported
 *     exactly (heap-buffer-overflow WRITE in pl_p2li) on Linux too.
 *   - macOS: the nano allocator's guard bytes catch the overrun and
 *     abort the process (~50%, depending on guard layout) -- which is
 *     how this first surfaced, as a flaky compressed-image-write crash
 *     on the macOS CI legs.
 *
 * Proposed fix (cfitsio): floor the PLIO branch of
 * imcomp_calc_max_elem() at 32 bytes, e.g.
 *
 *       if (nx * sizeof(int) < 32)  return 32;
 *       else                        return (nx * sizeof(int));
 *
 * Tracked as cfitsio #136:
 *   https://github.com/heasarc/cfitsio/issues/136
 * and downstream as fitsio #496:
 *   https://github.com/esheldon/fitsio/issues/496
 *
 * Build + run via run.sh in this directory (see README.md for the
 * ASAN-instrumented-cfitsio recipe needed to make it abort on Linux).
 */
#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"

int main(int argc, char **argv) {
    const char *fname = (argc > 1) ? argv[1] : "plio_small.fits.fz";
    fitsfile *fptr = NULL;
    int status = 0;

    remove(fname);
    if (fits_create_file(&fptr, fname, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }

    /* PLIO_1, 2x2 tiles -> nx = 4 -> 16-byte (8-short) tile buffer. */
    if (fits_set_compression_type(fptr, PLIO_1, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }
    long tile[2] = {2, 2};
    if (fits_set_tile_dim(fptr, 2, tile, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }

    /* A 4x4 32-bit-int (mask-like) image: values in PLIO's 0..2**24
       range, with runs that force pl_p2li to emit data shorts past
       the header. */
    long naxes[2] = {4, 4};
    if (fits_create_img(fptr, LONG_IMG, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }

    int pix[16];
    for (int i = 0; i < 16; i++)
        pix[i] = i % 4;  /* 0,1,2,3,0,1,2,3,... -> nontrivial line list */

    /* The overflow happens here, when cfitsio compresses each 2x2
       tile via pl_p2li into the undersized cbuf. */
    if (fits_write_img(fptr, TINT, 1, 16, pix, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }

    if (fits_close_file(fptr, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }

    printf("wrote %s (4x4 LONG image, PLIO_1, 2x2 tiles)\n", fname);
    printf("if you reach here without an ASAN report or SIGABRT, the "
           "overrun was silently tolerated by this allocator\n");
    return 0;
}
