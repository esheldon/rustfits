/*
 * Reproduction ATTEMPT: does cfitsio corrupt the heap when a
 * compressed image is patched (write-with-start / fits_write_subset)
 * repeatedly across an algorithm sweep in one process?
 *
 * Observed through the fitsio Python wrapper on Linux as
 * `free(): invalid next size` (rustfits upstream-bugs.md #2), but it
 * was never isolated to pure cfitsio.  This program exercises the
 * same shape with NO Python: for each algorithm in turn, create a
 * (256,256) int image with (32,32) tiles, write it, then patch
 * several sub-regions repeatedly.  If cfitsio is the culprit this
 * should abort partway through; if it runs clean, the corruption is
 * likely in the fitsio wrapper, not cfitsio.
 *
 * Build + run via run.sh (target: sweep).
 */
#include <stdio.h>
#include <stdlib.h>
#include "fitsio.h"

static void patch(fitsfile *fptr, int seed, int *status) {
    long fpix[2], lpix[2];

    /* single pixel */
    int one = 42 + seed;
    fpix[0] = 65; fpix[1] = 65; lpix[0] = 65; lpix[1] = 65;
    fits_write_subset(fptr, TINT, fpix, lpix, &one, status);

    /* one full tile, aligned 32x32 */
    int *tile = malloc(32 * 32 * sizeof(int));
    for (int i = 0; i < 32 * 32; i++) tile[i] = i + seed;
    fpix[0] = 65; fpix[1] = 65; lpix[0] = 96; lpix[1] = 96;
    fits_write_subset(fptr, TINT, fpix, lpix, tile, status);
    free(tile);

    /* 8x8 straddling a tile corner -> touches 4 tiles */
    int small[64];
    for (int i = 0; i < 64; i++) small[i] = i + seed;
    fpix[0] = 61; fpix[1] = 61; lpix[0] = 68; lpix[1] = 68;
    fits_write_subset(fptr, TINT, fpix, lpix, small, status);
}

int main(void) {
    int algos[] = {GZIP_1, GZIP_2, RICE_1, HCOMPRESS_1, PLIO_1};
    const char *names[] = {"GZIP_1", "GZIP_2", "RICE_1",
                           "HCOMPRESS_1", "PLIO_1"};
    long naxes[2] = {256, 256};
    long tiledim[2] = {32, 32};
    long npix = naxes[0] * naxes[1];

    for (int a = 0; a < 5; a++) {
        int status = 0;
        char fname[80];
        snprintf(fname, sizeof fname, "!sweep_%s.fits", names[a]);

        fitsfile *fptr = NULL;
        fits_create_file(&fptr, fname, &status);
        fits_set_compression_type(fptr, algos[a], &status);
        fits_set_tile_dim(fptr, 2, tiledim, &status);
        fits_create_img(fptr, LONG_IMG, 2, naxes, &status);

        int *data = malloc(npix * sizeof(int));
        for (long i = 0; i < npix; i++) data[i] = (int)(i % 1000);
        long fpix[2] = {1, 1};
        fits_write_pix(fptr, TINT, fpix, npix, data, &status);
        free(data);

        /* the cumulative-patch shape that tickled the wrapper crash */
        for (int r = 0; r < 5 && !status; r++) patch(fptr, r, &status);

        fits_close_file(fptr, &status);
        printf("%-12s status=%d\n", names[a], status);
        if (status) fits_report_error(stderr, status);
    }
    printf("completed all algorithms without abort\n");
    return 0;
}
