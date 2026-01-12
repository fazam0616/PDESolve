#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../include/gpu_compiler.h"
#include "../include/expression.h"
#include "../include/grid.h"
#include "../include/dictionary.h"

// Helper: create simple grid and fill with known values: u = x, v = y
static GridField* make_linear_fields(GridMetadata *grid, const char *name, int which) {
    GridField *f = grid_field_create(grid);
    uint32_t idx[3];
    for (uint32_t j = 0; j < grid->dims[1]; ++j) {
        for (uint32_t i = 0; i < grid->dims[0]; ++i) {
            idx[0] = i; idx[1] = j; idx[2] = 0;
            double x = grid->origin[0] + i * grid->spacing[0];
            double y = grid->origin[1] + j * grid->spacing[1];
            double v = which == 0 ? x : y;
            Literal *lit = literal_create_scalar(v);
            grid_field_set(f, idx, lit);
            literal_free(lit);
        }
    }
    return f;
}

static double rel_err(GridField *a, GridField *b) {
    GridField *d = grid_field_subtract(a, b);
    double dn = grid_field_norm(d);
    double an = grid_field_norm(a);
    grid_field_free(d);
    if (an == 0.0) return dn;
    return dn / an;
}

int main() {
    printf("Test GPU pointwise codegen (CPU-backed runner)\n");

    uint32_t dims[3] = {32, 32, 1};
    double spacing[3] = {0.1, 0.1, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);

    GridField *ua = make_linear_fields(grid, "a", 0);
    GridField *ub = make_linear_fields(grid, "b", 1);

    Dictionary *dict = dict_create(8);
    dict_set(dict, "a", &ua->data);
    dict_set(dict, "b", &ub->data);

    // Expression: out = a + 2.0 * b
    Expression *expr = expr_add(expr_variable("a"), expr_multiply(expr_literal(literal_create_scalar(2.0)), expr_variable("b")));

    GPUProgram *prog = gpu_compile_optimized(expr, grid, GPU_BACKEND_OPENGL);
    assert(prog != NULL);

    GridField *gpu_out = gpu_run_program_cpu(prog, dict, grid);
    assert(gpu_out != NULL);

    // CPU reference: a + 2*b
    GridField *b2 = grid_field_copy(ub);
    grid_field_scale_inplace(b2, 2.0);
    GridField *cpu_ref = grid_field_add(ua, b2);

    double err = rel_err(cpu_ref, gpu_out);
    printf("  relative error: %g\n", err);
    // per-sample debug prints removed
    // Allow small floating-point differences between GPU and CPU runs.
    assert(err < 1e-6);

    // Cleanup
    grid_field_free(ua);
    grid_field_free(ub);
    grid_field_free(b2);
    grid_field_free(cpu_ref);
    grid_field_free(gpu_out);
    dict_free(dict);
    expression_free(expr);
    gpu_program_free(prog);
    grid_metadata_free(grid);

    printf("Pointwise GPU (CPU-run) test passed.\n");
    return 0;
}
