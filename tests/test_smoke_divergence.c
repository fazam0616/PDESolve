#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/grid.h"
#include "../include/literal.h"
#include "../include/smoke_advection.h"

int main(void) {
    uint32_t dims[3] = {32,32,1};
    double spacing[3] = {1.0,1.0,1.0};
    double origin[3] = {0.0,0.0,0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    if (!grid) return EXIT_FAILURE;

    GridField *vx = grid_field_create(grid);
    GridField *vy = grid_field_create(grid);
    GridField *div = grid_field_create(grid);

    uint32_t idx[3];
    for (uint32_t linear=0; linear<grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        double x = idx[0] * spacing[0];
        double y = idx[1] * spacing[1];
        // vx = x, vy = y => divergence = 1 + 1 = 2
        Literal *lvx = literal_create_scalar(x);
        Literal *lvy = literal_create_scalar(y);
        grid_field_set(vx, idx, lvx); literal_free(lvx);
        grid_field_set(vy, idx, lvy); literal_free(lvy);
    }

    smoke_divergence(vx, vy, div);

    double sum_sq = 0.0;
    for (uint32_t linear=0; linear<grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        Literal *l = grid_field_get(div, idx);
        double v = literal_get(l, (uint32_t[]){0,0,0});
        literal_free(l);
        double err = v - 2.0;
        sum_sq += err*err;
    }
    double rms = sqrt(sum_sq / (double)grid->total_points);

    grid_field_free(vx);
    grid_field_free(vy);
    grid_field_free(div);

    if (rms < 1e-12) {
        printf("TEST OK: test_smoke_divergence (rms=%g)\n", rms);
        return EXIT_SUCCESS;
    } else {
        fprintf(stderr, "TEST FAIL: divergence RMS=%g\n", rms);
        return EXIT_FAILURE;
    }
}
