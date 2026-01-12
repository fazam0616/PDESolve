#include <stdio.h>
#include <assert.h>
#include "../include/gpu_compiler.h"
#include "../include/expression.h"
#include "../include/grid.h"

int main() {
    printf("Test GPU codegen: start\n");

    uint32_t dims[3] = {32, 32, 1};
    double spacing[3] = {0.1, 0.1, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    assert(grid != NULL);

    // Build expression: u + dt * laplacian(u)
    Expression *u = expr_variable("u");
    Expression *lap = expr_laplacian(expr_variable("u"));
    Expression *dt = expr_literal(literal_create_scalar(0.01));
    Expression *dtlap = expr_multiply(dt, lap);
    Expression *expr = expr_add(u, dtlap);

    GPUProgram *prog = gpu_compile_optimized(expr, grid, GPU_BACKEND_OPENGL);
    if (!prog) { fprintf(stderr, "gpu compile failed\n"); return 1; }

    printf("Compiled program: memory estimate %zu bytes\n", gpu_estimate_memory(prog));
    gpu_print_program(prog);

    gpu_program_free(prog);
    expression_free(expr);
    grid_metadata_free(grid);

    printf("Test GPU codegen: done\n");
    return 0;
}
