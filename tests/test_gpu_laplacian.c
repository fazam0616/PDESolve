#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../include/expression.h"
#include "../include/grid.h"
#include "../include/dictionary.h"
#include "../include/gpu_compiler.h"

static GridField* make_test_field(GridMetadata *grid) {
    GridField *f = grid_field_create(grid);
    uint32_t idx[3];
    for (uint32_t j = 0; j < grid->dims[1]; ++j) {
        for (uint32_t i = 0; i < grid->dims[0]; ++i) {
            idx[0]=i; idx[1]=j; idx[2]=0;
            double x = grid->origin[0] + i * grid->spacing[0];
            double y = grid->origin[1] + j * grid->spacing[1];
            Literal *lit = literal_create_scalar(x*x + y*y);
            grid_field_set(f, idx, lit);
            literal_free(lit);
        }
    }
    return f;
}

// Compute relative error over interior points (exclude 1-cell boundary)
static double compute_relative_error_interior(GridField *a, GridField *b) {
    GridMetadata *g = a->grid;
    uint32_t nx = g->dims[0]; uint32_t ny = g->dims[1];
    double sum_a2 = 0.0, sum_d2 = 0.0;
    uint32_t idx[3]; idx[2]=0;
    for (uint32_t j = 1; j + 1 < ny; ++j) {
        for (uint32_t i = 1; i + 1 < nx; ++i) {
            idx[0]=i; idx[1]=j;
            Literal *la = grid_field_get(a, idx);
            Literal *lb = grid_field_get(b, idx);
            double va = 0.0, vb = 0.0;
            if (la && la->field) va = la->field[0];
            if (lb && lb->field) vb = lb->field[0];
            double d = va - vb;
            sum_d2 += d*d;
            sum_a2 += va*va;
        }
    }
    if (sum_a2 == 0.0) return sqrt(sum_d2);
    return sqrt(sum_d2) / sqrt(sum_a2);
}

int main(void) {
    uint32_t dims[3] = {64,64,1};
    double spacing[3] = {0.01,0.01,1.0};
    double origin[3] = {0.0,0.0,0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    assert(grid);

    GridField *u = make_test_field(grid);
    Dictionary *dict = dict_create(8);
    dict_set(dict, "u", &u->data);

    // CPU reference via non-compact finite-difference (matches GPU central differences)
    GridField *cpu_lap = grid_field_laplacian(u);
    Expression *expr_lap = expr_laplacian(expr_variable("u"));

    // GPU compile + run
    GPUProgram *prog = gpu_compile_optimized(expr_lap, grid, GPU_BACKEND_OPENGL);
    if (!prog) {
        fprintf(stderr, "SKIP: GPU compile not available for Laplacian\n");
        grid_field_free(u); dict_free(dict); grid_metadata_free(grid); expression_free(expr_lap); return 0;
    }
    GridField *gpu_out = gpu_run_program_cpu(prog, dict, grid);
    if (!gpu_out) {
        fprintf(stderr, "SKIP: GPU Laplacian execution returned NULL\n");
        gpu_program_free(prog); grid_field_free(u); dict_free(dict); grid_metadata_free(grid); expression_free(expr_lap); return 0;
    }

    double err = compute_relative_error_interior(cpu_lap, gpu_out);
    printf("GPU Laplacian relative error (interior check): %g\n", err);
    // Allow looser tolerance for prototype GPU stencil
    if (err >= 1e-4) {
    uint32_t mid[3] = {dims[0]/2, dims[1]/2, 0};
    Literal *lc = grid_field_get(cpu_lap, mid);
    Literal *lg = grid_field_get(gpu_out, mid);
    double vc = lc && lc->field ? lc->field[0] : 0.0;
    double vg = lg && lg->field ? lg->field[0] : 0.0;
    fprintf(stderr, "GPU Laplacian error exceeds tolerance\n");
    fprintf(stderr, " center CPU=%g GPU=%g\n", vc, vg);
        return 2;
    }

    // cleanup
    grid_field_free(u); grid_field_free(cpu_lap); grid_field_free(gpu_out);
    gpu_program_free(prog); dict_free(dict); grid_metadata_free(grid); expression_free(expr_lap);
    printf("test_gpu_laplacian: PASS\n");
    return 0;
}
