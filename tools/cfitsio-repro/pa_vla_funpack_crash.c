/*
 * Standalone cfitsio reproducer: funpack crashes on a string-VLA
 * (variable-length character, TFORM '1PA') compressed table column.
 *
 * No rustfits, no fitsio Python wrapper -- this writes an ordinary
 * uncompressed BINTABLE with one variable-length string column using
 * libcfitsio, then the companion run.sh shells out to cfitsio's own
 * `fpack -table` (which compresses the PA column) and `funpack`
 * (which crashes with heap corruption).  So the defect is entirely
 * inside cfitsio's table (de)compression of PA columns.
 *
 * Tracked downstream as rustfits issue #9:
 *   https://github.com/esheldon/rustfits/issues/9
 *
 * Build + run via run.sh in this directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fitsio.h"

int main(int argc, char **argv) {
    const char *fname = (argc > 1) ? argv[1] : "pa_vla.fits";
    fitsfile *fptr = NULL;
    int status = 0;

    remove(fname);
    if (fits_create_file(&fptr, fname, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }

    const long nrows = 3000;
    char *ttype[] = {"name"};
    char *tform[] = {"1PA"};  /* variable-length string column */
    char *tunit[] = {""};
    if (fits_create_tbl(fptr, BINARY_TBL, 0, 1,
                        ttype, tform, tunit, "DATA", &status)) {
        fits_report_error(stderr, status);
        return 1;
    }

    /* One variable-length string per row: "obj_" + (i % 15) 'x's. */
    for (long i = 0; i < nrows; i++) {
        char buf[64];
        int xn = (int)(i % 15);
        int p = snprintf(buf, sizeof buf, "obj_");
        for (int k = 0; k < xn; k++) buf[p++] = 'x';
        buf[p] = '\0';
        char *cell[1] = {buf};
        if (fits_write_col(fptr, TSTRING, 1, i + 1, 1, 1, cell, &status)) {
            fits_report_error(stderr, status);
            return 1;
        }
    }

    if (fits_close_file(fptr, &status)) {
        fits_report_error(stderr, status);
        return 1;
    }
    printf("wrote %s (%ld rows, one 1PA variable-length string column)\n",
           fname, nrows);
    return 0;
}
