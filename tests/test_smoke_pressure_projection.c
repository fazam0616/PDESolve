#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/grid.h"
#include "../include/literal.h"
#include "../include/smoke_pressure.h"

int main(void) {
    uint32_t dims[3] = {32,32,1};
    double spacing[3] = {1.0/(dims[0]-1), 1.0/(dims[1]-1), 1.0};
    double origin[3] = {0.0,0.0,0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    if (!grid) return EXIT_FAILURE;

    GridField *rhs = grid_field_create(grid);
    GridField *p = grid_field_create(grid);

    // Manufactured solution p_true = sin(pi x) sin(pi y) (zero on boundaries)
    uint32_t idx[3];
    for (uint32_t j=0;j<dims[1];j++){
        for (uint32_t i=0;i<dims[0];i++){
            idx[0]=i; idx[1]=j; idx[2]=0;
            double x = origin[0] + i * spacing[0];
            double y = origin[1] + j * spacing[1];
            double p_true = sin(M_PI * x) * sin(M_PI * y);
            double lap = -2.0 * M_PI * M_PI * p_true; // Laplacian of p_true
            Literal *l = literal_create_scalar(lap);
            grid_field_set(rhs, idx, l); literal_free(l);
        }
    }

    double tol = 1e-6;
    int maxit = 10000;
    int rc = smoke_solve_poisson(rhs, p, tol, maxit, NULL);
    if (rc != 0) {
        fprintf(stderr, "TEST FAIL: Poisson solver did not converge\n");
        grid_field_free(rhs); grid_field_free(p); grid_metadata_free(grid);
        return EXIT_FAILURE;
    }

    // Compare p to p_true (up to numerical error)
    double sum_sq = 0.0;
    double sum_sq_true = 0.0;
    for (uint32_t j=0;j<dims[1];j++){
        for (uint32_t i=0;i<dims[0];i++){
            idx[0]=i; idx[1]=j; idx[2]=0;
            double x = origin[0] + i * spacing[0];
            double y = origin[1] + j * spacing[1];
            double p_true = sin(M_PI * x) * sin(M_PI * y);
            double p_num = literal_get(&p->data, idx);
            double err = p_num - p_true;
            sum_sq += err*err;
            sum_sq_true += p_true*p_true;
        }
    }
    double rms = sqrt(sum_sq / (double)(grid->total_points));
    double rms_true = sqrt(sum_sq_true / (double)(grid->total_points));

    grid_field_free(rhs); grid_field_free(p); grid_metadata_free(grid);

    if (rms < 1e-3 * (1.0 + rms_true)) {
        printf("TEST OK: test_smoke_pressure_projection (rms_err=%g)\n", rms);
        return EXIT_SUCCESS;
    } else {
        fprintf(stderr, "TEST FAIL: pressure error RMS=%g\n", rms);
        return EXIT_FAILURE;
    }
}
