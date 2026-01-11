#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/grid.h"
#include "../include/literal.h"

int main(void) {
    uint32_t dims[3] = {11,11,1};
    double spacing[3] = {1.0/(dims[0]-1), 1.0/(dims[1]-1), 1.0};
    double origin[3] = {0.0,0.0,0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    if (!grid) return EXIT_FAILURE;

    // Set Neumann BC (zero normal derivative) on left and right boundaries
    grid_set_boundary(grid, 0, 0, BC_NEUMANN, 0.0);
    grid_set_boundary(grid, 0, 1, BC_NEUMANN, 0.0);

    GridField *f = grid_field_create(grid);
    uint32_t idx[3];
    // u(x,y) = x -> du/dx = 1 everywhere; but boundary Neumann=0 should override derivative at boundary
    for (uint32_t j=0;j<dims[1];j++){
        for (uint32_t i=0;i<dims[0];i++){
            idx[0]=i; idx[1]=j; idx[2]=0;
            double x = origin[0] + i * spacing[0];
            double val = x;
            Literal *lv = literal_create_scalar(val);
            grid_field_set(f, idx, lv); literal_free(lv);
        }
    }

    GridField *dfdx = grid_field_derivative(f, 0, 1);
    if (!dfdx) {
        fprintf(stderr, "TEST FAIL: derivative failed\n");
        grid_field_free(f); grid_metadata_free(grid);
        return EXIT_FAILURE;
    }

    // Check boundary derivatives are equal to Neumann value (0.0)
    double tol = 1e-12;
    for (uint32_t j=0;j<dims[1];j++){
        idx[0]=0; idx[1]=j; idx[2]=0;
        double left = literal_get(&dfdx->data, idx);
        if (fabs(left - 0.0) > 1e-8) {
            fprintf(stderr, "TEST FAIL: left boundary derivative != 0 (got %g)\n", left);
            grid_field_free(f); grid_field_free(dfdx); grid_metadata_free(grid);
            return EXIT_FAILURE;
        }
        idx[0]=dims[0]-1; idx[1]=j; idx[2]=0;
        double right = literal_get(&dfdx->data, idx);
        if (fabs(right - 0.0) > 1e-8) {
            fprintf(stderr, "TEST FAIL: right boundary derivative != 0 (got %g)\n", right);
            grid_field_free(f); grid_field_free(dfdx); grid_metadata_free(grid);
            return EXIT_FAILURE;
        }
    }

    // Check interior derivative approximates 1.0
    for (uint32_t j=0;j<dims[1];j++){
        for (uint32_t i=1;i<dims[0]-1;i++){
            idx[0]=i; idx[1]=j; idx[2]=0;
            double v = literal_get(&dfdx->data, idx);
            if (fabs(v - 1.0) > 1e-6) {
                fprintf(stderr, "TEST FAIL: interior derivative not ~1 (got %g at i=%u)\n", v, i);
                grid_field_free(f); grid_field_free(dfdx); grid_metadata_free(grid);
                return EXIT_FAILURE;
            }
        }
    }

    grid_field_free(f); grid_field_free(dfdx); grid_metadata_free(grid);
    printf("TEST OK: test_neumann_bc\n");
    return EXIT_SUCCESS;
}
