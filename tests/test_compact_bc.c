#include "../include/grid.h"
#include "../include/literal.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

// 1D test function: f(x) = x^2
Literal* square_func(const double *coords, int n_dims) {
    double x = (n_dims > 0) ? coords[0] : 0.0;
    return literal_create_scalar(x * x);
}

int main() {
    printf("Test: Compact derivative BC handling\n");

    uint32_t dims[3] = {11, 1, 1};
    double spacing[3] = {0.1, 1.0, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};

    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 1);
    assert(grid != NULL);

    // Set left boundary Dirichlet: f(0) = 0
    grid_set_boundary(grid, 0, 0, BC_DIRICHLET, 0.0);
    // Set right boundary Neumann: f'(L) = 2*L where L = (n-1)*dx
    double L = (dims[0] - 1) * spacing[0];
    double neumann_val = 2.0 * L;
    grid_set_boundary(grid, 0, 1, BC_NEUMANN, neumann_val);

    GridField *field = grid_field_create(grid);
    assert(field != NULL);

    grid_field_init_from_function(field, square_func);

    GridField *d_std = grid_field_derivative(field, 0, 1);
    GridField *d_compact = grid_field_derivative_compact(field, 0, 1);

    if (!d_std || !d_compact) {
        fprintf(stderr, "Derivative computation returned NULL\n");
        return 2;
    }

    // Compare only boundary points (index 0 and n-1) where BC handling matters
    uint32_t idx0[3] = {0, 0, 0};
    uint32_t idxn[3] = {dims[0] - 1, 0, 0};

    Literal *a0 = grid_field_get(d_std, idx0);
    Literal *b0 = grid_field_get(d_compact, idx0);
    Literal *an = grid_field_get(d_std, idxn);
    Literal *bn = grid_field_get(d_compact, idxn);

    double v_a0 = a0 ? literal_get(a0, (uint32_t[]){0,0,0}) : 0.0;
    double v_b0 = b0 ? literal_get(b0, (uint32_t[]){0,0,0}) : 0.0;
    double v_an = an ? literal_get(an, (uint32_t[]){0,0,0}) : 0.0;
    double v_bn = bn ? literal_get(bn, (uint32_t[]){0,0,0}) : 0.0;

    if (a0) literal_free(a0); if (b0) literal_free(b0);
    if (an) literal_free(an); if (bn) literal_free(bn);

    double diff0 = fabs(v_a0 - v_b0);
    double diffn = fabs(v_an - v_bn);

    printf("  |d_std(0) - d_compact(0)| = %g\n", diff0);
    printf("  |d_std(n-1) - d_compact(n-1)| = %g\n", diffn);

    double tol = 1e-12;
    if (diff0 > tol || diffn > tol) {
        fprintf(stderr, "Compact derivative BC mismatch (diff0=%g, diffn=%g)\n", diff0, diffn);
        grid_field_free(d_std);
        grid_field_free(d_compact);
        grid_field_free(field);
        grid_metadata_free(grid);
        return 1;
    }

    printf("  [OK] compact derivative matches standard at domain boundaries\n");

    grid_field_free(d_std);
    grid_field_free(d_compact);
    grid_field_free(field);
    grid_metadata_free(grid);

    return 0;
}
