#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/grid.h"
#include "../include/literal.h"
#include "../include/smoke_advection.h"

int main(void) {
    uint32_t dims[3] = {64,64,1};
    double spacing[3] = {1.0,1.0,1.0};
    double origin[3] = {0.0,0.0,0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    if (!grid) return EXIT_FAILURE;

    GridField *density = grid_field_create(grid);
    GridField *vx = grid_field_create(grid);
    GridField *vy = grid_field_create(grid);
    GridField *vels[2] = {vx, vy};

    // initialize density: single cell in center
    uint32_t idx[3];
    for (uint32_t linear=0; linear<grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        double val = 0.0;
        if (idx[0] == dims[0]/2 && idx[1] == dims[1]/2) val = 1.0;
        Literal *lv = literal_create_scalar(val);
        grid_field_set(density, idx, lv); literal_free(lv);
        // zero velocity
        Literal *lz = literal_create_scalar(0.0);
        grid_field_set(vx, idx, lz); grid_field_set(vy, idx, lz); literal_free(lz);
    }

    // compute total mass before
    double mass_before = 0.0;
    for (uint32_t linear=0; linear<grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        Literal *l = grid_field_get(density, idx);
        mass_before += literal_get(l, (uint32_t[]){0,0,0});
        literal_free(l);
    }

    GridField *advected = smoke_advect_scalar(density, vels, 2, 0.1);
    if (!advected) {
        fprintf(stderr, "TEST FAIL: advected returned NULL\n");
        return EXIT_FAILURE;
    }

    double mass_after = 0.0;
    for (uint32_t linear=0; linear<grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        Literal *l = grid_field_get(advected, idx);
        mass_after += literal_get(l, (uint32_t[]){0,0,0});
        literal_free(l);
    }

    grid_field_free(density);
    grid_field_free(vx);
    grid_field_free(vy);
    grid_field_free(advected);

    double diff = fabs(mass_after - mass_before);
    if (diff < 1e-12) {
        printf("TEST OK: test_smoke_advection_mass (mass diff=%g)\n", diff);
        return EXIT_SUCCESS;
    } else {
        fprintf(stderr, "TEST FAIL: mass changed by %g\n", diff);
        return EXIT_FAILURE;
    }
}
