#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/grid.h"
#include "../include/smoke_sim.h"
#include "../include/smoke_pressure.h"

// Small utility: compute residual norm ||L p - rhs||_2
static double compute_poisson_residual_norm(GridField *p, GridField *rhs) {
    GridField *lap = grid_field_laplacian(p);
    if (!lap) return INFINITY;
    GridField *diff = grid_field_subtract(lap, rhs);
    double n = grid_field_norm(diff);
    grid_field_free(lap);
    grid_field_free(diff);
    return n;
}

// Create sinusoidal solution and set rhs = Laplacian(u)
static GridField* make_smooth_rhs(GridMetadata *grid) {
    GridField *u = grid_field_create(grid);
    uint32_t idx[3]; double coords[3];
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        grid_index_to_coord(grid, idx, coords);
        double x = coords[0]; double y = coords[1];
        double val = sin(M_PI * x) * sin(M_PI * y);
        literal_set(&u->data, idx, val);
    }
    GridField *rhs = grid_field_laplacian(u);
    grid_field_free(u);
    return rhs;
}

static GridField* make_delta_rhs(GridMetadata *grid) {
    GridField *rhs = grid_field_create(grid);
    uint32_t idx[3];
    idx[0] = grid->dims[0]/2; idx[1] = grid->dims[1]/2; idx[2] = 0;
    literal_set(&rhs->data, idx, 1.0);
    return rhs;
}

static GridField* make_checkerboard_rhs(GridMetadata *grid) {
    GridField *rhs = grid_field_create(grid);
    uint32_t idx[3];
    for (uint32_t j = 0; j < grid->dims[1]; ++j) {
        for (uint32_t i = 0; i < grid->dims[0]; ++i) {
            idx[0]=i; idx[1]=j; idx[2]=0;
            double v = ((i + j) & 1) ? 1.0 : -1.0;
            literal_set(&rhs->data, idx, v);
        }
    }
    return rhs;
}

static GridField* make_noise_rhs(GridMetadata *grid) {
    GridField *rhs = grid_field_create(grid);
    uint32_t idx[3];
    srand(12345);
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        double v = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
        literal_set(&rhs->data, idx, v);
    }
    return rhs;
}

static int run_case(GridMetadata *grid, GridField *rhs, const char *case_name) {
    GridField *p_baseline = grid_field_create(grid);
    GridField *p_rb = grid_field_create(grid);
    double tol = 1e-6; int maxit = 2000;

    // baseline: no RB
    setenv("POISSON_USE_RB", "0", 1);
    setenv("POISSON_PRECOND", "jacobi", 1);
    unsetenv("POISSON_PRE_SMOOTH");
    int iters_base = 0;
    int rc1 = smoke_solve_poisson(rhs, p_baseline, tol, maxit, &iters_base);

    // RB pre-smooth: enable RB and do a few pre-smooth sweeps
    setenv("POISSON_USE_RB", "1", 1);
    setenv("POISSON_PRE_SMOOTH", "10", 1);
    int iters_rb = 0;
    int rc2 = smoke_solve_poisson(rhs, p_rb, tol, maxit, &iters_rb);

    double res_base = compute_poisson_residual_norm(p_baseline, rhs);
    double res_rb = compute_poisson_residual_norm(p_rb, rhs);

    printf("Case %s: base rc=%d iters=%d res=%g | rb rc=%d iters=%d res=%g\n",
           case_name, rc1, iters_base, res_base, rc2, iters_rb, res_rb);

    int passed = 1;
    // Expect RB to not break solves, and to improve difficult cases.
    if (rc1 != 0) passed = 0;
    if (rc2 != 0) passed = 0;
    if (passed) {
        // If baseline already converged immediately or residual is tiny,
        // require RB not to make things catastrophically worse.
        if (iters_base == 0 || res_base < 1e-12) {
            if (!(res_rb < res_base * 1e6)) passed = 0;
        } else {
            // Otherwise expect RB to reduce iterations or noticeably reduce residual
            if (!(iters_rb <= iters_base || res_rb < res_base * 0.9)) passed = 0;
        }
    }

    grid_field_free(p_baseline);
    grid_field_free(p_rb);
    return passed;
}

int main(void) {
    uint32_t dims[3] = {64, 64, 1};
    double spacing[3] = {1.0/(dims[0]-1), 1.0/(dims[1]-1), 1.0};
    double origin[3] = {0,0,0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    if (!grid) return 2;

    int total_pass = 1;

    GridField *rhs;

    rhs = make_smooth_rhs(grid);
    total_pass &= run_case(grid, rhs, "smooth");
    grid_field_free(rhs);

    rhs = make_delta_rhs(grid);
    total_pass &= run_case(grid, rhs, "delta");
    grid_field_free(rhs);

    rhs = make_checkerboard_rhs(grid);
    total_pass &= run_case(grid, rhs, "checkerboard");
    grid_field_free(rhs);

    rhs = make_noise_rhs(grid);
    total_pass &= run_case(grid, rhs, "noise");
    grid_field_free(rhs);

    grid_metadata_free(grid);

    if (total_pass) {
        printf("[PASS] Poisson RBGS smoother tests\n");
        return 0;
    } else {
        printf("[FAIL] Poisson RBGS smoother tests\n");
        return 1;
    }
}
