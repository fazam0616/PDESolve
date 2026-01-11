#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/grid.h"
#include "../include/literal.h"
#include "../include/smoke_pressure.h"

int main(void) {
    uint32_t dims[3] = {128,128,1};
    double spacing[3] = {1.0/(dims[0]-1), 1.0/(dims[1]-1), 1.0};
    double origin[3] = {0.0,0.0,0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    if (!grid) return EXIT_FAILURE;

    GridField *rhs = grid_field_create(grid);
    GridField *p = grid_field_create(grid);
    if (!rhs || !p) return EXIT_FAILURE;

    // Simulate many paint operations around (92,64)
    int cx = 92, cy = 64;
    uint32_t idx[3];
    for (int pass = 0; pass < 50; pass++) {
        for (int y = cy-8; y <= cy+8; y++) {
            for (int x = cx-8; x <= cx+8; x++) {
                if (x < 0 || x >= (int)dims[0] || y < 0 || y >= (int)dims[1]) continue;
                idx[0] = x; idx[1] = y; idx[2] = 0;
                double cur = literal_get(&rhs->data, idx);
                cur += 0.02; // small increment per paint
                Literal *l = literal_create_scalar(cur);
                grid_field_set(rhs, idx, l); literal_free(l);
            }
        }
    }

    double tol = 1e-6;
    int maxit = 2000;

    int rc = smoke_solve_poisson(rhs, p, tol, maxit, NULL);

    if (rc != 0) {
        fprintf(stderr, "TEST FAIL: Poisson solver returned rc=%d\n", rc);
        grid_field_free(rhs); grid_field_free(p); grid_metadata_free(grid);
        return EXIT_FAILURE;
    }

    // verify p is finite
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        double v = literal_get(&p->data, idx);
        if (!isfinite(v)) {
            fprintf(stderr, "TEST FAIL: Non-finite solution value at linear=%u\n", linear);
            grid_field_free(rhs); grid_field_free(p); grid_metadata_free(grid);
            return EXIT_FAILURE;
        }
    }

    printf("TEST OK: test_smoke_poisson_paint_sim\n");
    grid_field_free(rhs); grid_field_free(p); grid_metadata_free(grid);
    return EXIT_SUCCESS;
}
