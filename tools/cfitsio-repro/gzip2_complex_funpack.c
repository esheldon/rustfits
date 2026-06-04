/*
 * Standalone cfitsio reproducer attempt: GZIP_2-compressed complex
 * (TFORM 'C' / 'M') table columns.
 *
 * cfitsio's GZIP_2 table *encoder* silently skips byte-shuffling for
 * complex columns (it shuffles only 2/4/8-byte I/J/E/D/K), but its
 * GZIP_2 *decompressor* unshuffle switch has no C/M case and falls
 * into a "...unsuitable data type" -> DATA_DECOMPRESSION_ERR branch
 * (imcompress.c).  The hypothesis is that cfitsio cannot funpack a
 * GZIP_2-compressed complex column it wrote itself.
 *
 * This program writes an ordinary uncompressed BINTABLE with one
 * single-precision complex column; run.sh then `fpack -g2 -table`s it
 * and `funpack`s it to observe whether the decompress errors.
 *
 * Build + run via run.sh in this directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"

int main(int argc, char **argv) {
    const char *fname = (argc > 1) ? argv[1] : "cplx.fits";
    fitsfile *fptr = NULL;
    int status = 0;

    remove(fname);
    if (fits_create_file(&fptr, fname, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }

    const long nrows = 2000;
    char *ttype[] = {"z"};
    char *tform[] = {"1C"};  /* single-precision complex */
    char *tunit[] = {""};
    if (fits_create_tbl(fptr, BINARY_TBL, 0, 1,
                        ttype, tform, tunit, "DATA", &status)) {
        fits_report_error(stderr, status);
        return 1;
    }

    /* fits_write_col with TCOMPLEX takes float pairs [re, im, ...]. */
    float *cell = malloc((size_t)nrows * 2 * sizeof(float));
    for (long i = 0; i < nrows; i++) {
        cell[2 * i] = (float)i * 0.5f;
        cell[2 * i + 1] = (float)(-i) * 0.25f;
    }
    if (fits_write_col(fptr, TCOMPLEX, 1, 1, 1, nrows, cell, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }
    free(cell);

    if (fits_close_file(fptr, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }
    printf("wrote %s (%ld rows, one 1C complex column)\n", fname, nrows);
    return 0;
}
