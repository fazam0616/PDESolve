#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/grid.h"
#include "../include/literal.h"
#include "../include/smoke_diffusion.h"

int main(void) {
    uint32_t dims[3] = {32,32,1};
    double spacing[3] = {1.0/(dims[0]-1), 1.0/(dims[1]-1), 1.0};
    double origin[3] = {0.0,0.0,0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    if (!grid) return EXIT_FAILURE;

    // Enforce zero Dirichlet boundaries for this manufactured-solution test
    for (int ax = 0; ax < 2; ax++) {
        grid->boundaries[ax*2 + 0].type = BC_DIRICHLET;
        grid->boundaries[ax*2 + 0].value = 0.0;
        grid->boundaries[ax*2 + 1].type = BC_DIRICHLET;
        grid->boundaries[ax*2 + 1].value = 0.0;
    }
    GridField *b = grid_field_create(grid);
    GridField *u = grid_field_create(grid);

    // Manufactured solution u_true = sin(pi x) sin(pi y) (zero on boundaries)
    double nu = 0.1;
    double dt = 0.01;
    double alpha = nu * dt;

    uint32_t idx[3];
    for (uint32_t j=0;j<dims[1];j++){
        for (uint32_t i=0;i<dims[0];i++){
            idx[0]=i; idx[1]=j; idx[2]=0;
            double x = origin[0] + i * spacing[0];
            double y = origin[1] + j * spacing[1];
            double u_true = sin(M_PI * x) * sin(M_PI * y);
            // Laplacian(u_true) = -2*pi^2 * u_true
            double factor = 1.0 + 2.0 * alpha * M_PI * M_PI;
            double bval = factor * u_true;
            Literal *lv = literal_create_scalar(bval);
            grid_field_set(b, idx, lv); literal_free(lv);
        }
    }

    double tol = 1e-6;
    int maxit = 10000;
    int rc = smoke_solve_helmholtz(b, u, alpha, tol, maxit);
    if (rc != 0) {
        fprintf(stderr, "TEST FAIL: Helmholtz solver did not converge\n");
        grid_field_free(b); grid_field_free(u); grid_metadata_free(grid);
        return EXIT_FAILURE;
    }

    double sum_sq = 0.0;
    double sum_sq_true = 0.0;
    for (uint32_t j=0;j<dims[1];j++){
        for (uint32_t i=0;i<dims[0];i++){
            idx[0]=i; idx[1]=j; idx[2]=0;
            double x = origin[0] + i * spacing[0];
            double y = origin[1] + j * spacing[1];
            double u_true = sin(M_PI * x) * sin(M_PI * y);
            double u_num = literal_get(&u->data, idx);
            double err = u_num - u_true;
            sum_sq += err*err;
            sum_sq_true += u_true*u_true;
        }
    }

    double rms = sqrt(sum_sq / (double)(grid->total_points));
    double rms_true = sqrt(sum_sq_true / (double)(grid->total_points));

    grid_field_free(b); grid_field_free(u); grid_metadata_free(grid);

    if (rms < 1e-3 * (1.0 + rms_true)) {
        printf("TEST OK: test_smoke_diffusion (rms_err=%g)\n", rms);
        return EXIT_SUCCESS;
    } else {
        fprintf(stderr, "TEST FAIL: helmholtz error RMS=%g\n", rms);
        return EXIT_FAILURE;
    }
}
