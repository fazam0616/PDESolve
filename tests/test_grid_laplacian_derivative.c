#include <stdio.h>
#include <math.h>
#include "../include/grid.h"

int main() {
    uint32_t dims[2] = {32, 32};
    double spacing[2] = {1.0/(dims[0]-1), 1.0/(dims[1]-1)};
    double origin[2] = {0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    GridField *p = grid_field_create(grid);

    uint32_t indices[N_DIM]; double coords[N_DIM];
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, indices);
        grid_index_to_coord(grid, indices, coords);
        double x = coords[0]; double y = coords[1];
        double val = sin(M_PI * x) * sin(M_PI * y);
        literal_set(&p->data, indices, val);
    }

    GridField *lap = grid_field_laplacian(p);

    // analytic laplacian = -2*pi^2 * p
    double max_err = 0.0;
    double max_rel = 0.0;
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, indices);
        // skip boundary points where BC handling may reduce accuracy
        if (grid_is_boundary(grid, indices)) continue;
        double ana = -2.0 * M_PI * M_PI * literal_get(&p->data, indices);
        double num = literal_get(&lap->data, indices);
        double err = fabs(num - ana);
        double rel = (fabs(ana) > 1e-15) ? err / fabs(ana) : err;
        if (err > max_err) max_err = err;
        if (rel > max_rel) max_rel = rel;
    }

    fprintf(stderr, "[test_grid_laplacian] max_err=%g max_rel=%g\n", max_err, max_rel);

    // Require reasonable accuracy
    if (max_rel > 1e-2) {
        fprintf(stderr, "TEST FAIL: Laplacian relative error too large\n");
        return 1;
    }

    grid_field_free(lap);
    grid_field_free(p);
    grid_metadata_free(grid);

    fprintf(stderr, "TEST PASS: Laplacian within tolerance\n");
    return 0;
}
