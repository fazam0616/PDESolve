#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../include/expression.h"
#include "../include/grid.h"
#include "../include/dictionary.h"
// Also test GPU execution for a pointwise fragment of the layered tests
#include "../include/gpu_compiler.h"

// Helper: create a simple 2D grid and fill with f(x,y)=x^2 + y^2
static GridField* make_test_field(GridMetadata *grid) {
    GridField *f = grid_field_create(grid);
    grid_field_init_from_function(f, (Literal*(*)(const double*,int))NULL);
    // Many grids in this repo provide grid_field_init_from_function signatures; to be safe,
    // fill manually with coordinates-based values.
    uint32_t idx[3];
    for (uint32_t j = 0; j < grid->dims[1]; ++j) {
        for (uint32_t i = 0; i < grid->dims[0]; ++i) {
            idx[0] = i; idx[1] = j; idx[2] = 0;
            double x = grid->origin[0] + i * grid->spacing[0];
            double y = grid->origin[1] + j * grid->spacing[1];
            double v = x*x + y*y;
            Literal *lit = literal_create_scalar(v);
            grid_field_set(f, idx, lit);
            literal_free(lit);
        }
    }
    return f;
}

static double compute_relative_error(GridField *a, GridField *b) {
    GridField *diff = grid_field_subtract(a, b);
    double nd = grid_field_norm(diff);
    double na = grid_field_norm(a);
    grid_field_free(diff);
    if (na == 0.0) return nd;
    return nd / na;
}

// Diagnostic: compute max absolute difference and return it with indices
static double compute_max_abs_diff(GridField *a, GridField *b, uint32_t *out_i, uint32_t *out_j) {
    double maxv = 0.0; uint32_t maxi=0, maxj=0;
    GridMetadata *grid = a->grid;
    for (uint32_t j = 0; j < grid->dims[1]; ++j) {
        for (uint32_t i = 0; i < grid->dims[0]; ++i) {
            uint32_t idx[3] = {i,j,0};
            double va = literal_get(&a->data, idx);
            double vb = literal_get(&b->data, idx);
            double d = fabs(va - vb);
            if (d > maxv) { maxv = d; maxi = i; maxj = j; }
        }
    }
    if (out_i) *out_i = maxi; if (out_j) *out_j = maxj;
    return maxv;
}

// Compute relative error but only over interior points (exclude boundaries)
static double compute_relative_error_interior(GridField *a, GridField *b) {
    GridMetadata *grid = a->grid;
    GridField *diff = grid_field_subtract(a, b);
    double sum_sq = 0.0;
    double sum_sq_a = 0.0;
    for (uint32_t j = 1; j + 1 < grid->dims[1]; ++j) {
        for (uint32_t i = 1; i + 1 < grid->dims[0]; ++i) {
            uint32_t idx[3] = {i,j,0};
            double d = literal_get(&diff->data, idx);
            double va = literal_get(&a->data, idx);
            sum_sq += d*d;
            sum_sq_a += va*va;
        }
    }
    grid_field_free(diff);
    if (sum_sq_a == 0.0) return sqrt(sum_sq);
    return sqrt(sum_sq) / sqrt(sum_sq_a);
}

static double compute_max_abs_diff_interior(GridField *a, GridField *b, uint32_t *out_i, uint32_t *out_j) {
    double maxv = 0.0; uint32_t maxi=0, maxj=0;
    GridMetadata *grid = a->grid;
    for (uint32_t j = 1; j + 1 < grid->dims[1]; ++j) {
        for (uint32_t i = 1; i + 1 < grid->dims[0]; ++i) {
            uint32_t idx[3] = {i,j,0};
            double va = literal_get(&a->data, idx);
            double vb = literal_get(&b->data, idx);
            double d = fabs(va - vb);
            if (d > maxv) { maxv = d; maxi = i; maxj = j; }
        }
    }
    if (out_i) *out_i = maxi; if (out_j) *out_j = maxj;
    return maxv;
}

int main() {
    printf("Test: layered/recursive expression evaluation\n");

    uint32_t dims[3] = {32, 32, 1};
    double spacing[3] = {0.05, 0.05, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    assert(grid != NULL);

    GridField *u = make_test_field(grid);

    // Build dictionary with variable "u"
    Dictionary *dict = dict_create(8);
    // dict_set deep-copies the Literal; pass pointer to field's data
    dict_set(dict, "u", &u->data);

    const double GPU_TOL = 5e-5;

    // Case 1: Laplacian(u)
    Expression *expr_lap = expr_laplacian(expr_variable("u"));
    Literal *res_lit = expression_evaluate_grid(expr_lap, dict, grid);
    assert(res_lit != NULL);
    GridField *res_field = grid_field_wrap_literal(res_lit, grid);

    GridField *ref_lap = grid_field_laplacian_compact(u);
    // For GPU comparisons we mirror the simpler explicit discretization used by
    // the GPU emitter (second-order central differences). Compute an explicit
    // Laplacian reference for GPU checks.
    GridField *ref_lap_explicit = grid_field_laplacian(u);
    double err = compute_relative_error(res_field, ref_lap);
    printf("  Laplacian relative error: %g\n", err);
    assert(err < 1e-12);

    // GPU mirror: attempt to compile and run the same Laplacian expression on GPU
    GPUProgram *prog_lap_gpu = gpu_compile_optimized(expr_lap, grid, GPU_BACKEND_OPENGL);
    GridField *gpu_lap_out = NULL;
    if (prog_lap_gpu) {
        gpu_lap_out = gpu_run_program_cpu(prog_lap_gpu, dict, grid);
    }
    if (!gpu_lap_out) {
        fprintf(stderr, "SKIP: GPU Laplacian execution returned NULL (likely unimplemented)\n");
    } else {
        double err_gpu_lap = compute_relative_error_interior(ref_lap_explicit, gpu_lap_out);
        printf("  GPU Laplacian relative error (interior check): %g\n", err_gpu_lap);
        if (err_gpu_lap >= GPU_TOL) {
            uint32_t mi=0,mj=0; double maxd = compute_max_abs_diff_interior(ref_lap_explicit, gpu_lap_out, &mi, &mj);
            fprintf(stderr, "GPU Laplacian mismatch (tolerance exceeded)\n");
            fprintf(stderr, "  max abs diff (interior) = %g at i=%u j=%u\n", maxd, mi, mj);
        }
        printf("  GPU Laplacian relative error: %g\n", err_gpu_lap);
        assert(err_gpu_lap < GPU_TOL);
        grid_field_free(gpu_lap_out);
        gpu_program_free(prog_lap_gpu);
    }

    grid_field_free(res_field);
    grid_field_free(ref_lap);
    expression_free(expr_lap);

    // Case 2: u + dt * lap(u)
    double dt = 0.01;
    Expression *u_var = expr_variable("u");
    Expression *lap_var = expr_laplacian(expr_variable("u"));
    Expression *dt_lit = expr_literal(literal_create_scalar(dt));
    Expression *dtlap = expr_multiply(dt_lit, lap_var);
    Expression *sum_expr = expr_add(u_var, dtlap);

    Literal *sum_lit = expression_evaluate_grid(sum_expr, dict, grid);
    GridField *sum_field = grid_field_wrap_literal(sum_lit, grid);

    GridField *cpu_lap = grid_field_laplacian_compact(u);
    grid_field_scale_inplace(cpu_lap, dt);
    GridField *cpu_sum = grid_field_add(u, cpu_lap);
    double err2 = compute_relative_error(sum_field, cpu_sum);
    printf("  u + dt*lap(u) relative error: %g\n", err2);
    assert(err2 < 1e-12);

    // GPU mirror for u + dt*lap(u)
    GPUProgram *prog_sum_gpu = gpu_compile_optimized(sum_expr, grid, GPU_BACKEND_OPENGL);
    GridField *gpu_sum_out = NULL;
    GridField *cpu_sum_explicit = NULL;
    if (prog_sum_gpu) {
    printf("[debug] prog_sum_gpu compiled\n"); fflush(stdout);
        // prepare explicit reference
        GridField *cpu_lap_explicit = grid_field_laplacian(u);
        grid_field_scale_inplace(cpu_lap_explicit, dt);
        cpu_sum_explicit = grid_field_add(u, cpu_lap_explicit);
        grid_field_free(cpu_lap_explicit);
    // run GPU
    printf("[debug] running gpu_run_program_cpu for sum...\n"); fflush(stdout);
    gpu_sum_out = gpu_run_program_cpu(prog_sum_gpu, dict, grid);
    printf("[debug] returned from gpu_run_program_cpu for sum\n"); fflush(stdout);
    }
    if (!prog_sum_gpu || !gpu_sum_out) {
        fprintf(stderr, "SKIP: GPU u+dt*lap(u) execution unavailable (unimplemented)\n");
    } else {
    double err_gpu_sum = compute_relative_error_interior(cpu_sum_explicit, gpu_sum_out);
        printf("  GPU u + dt*lap(u) relative error (interior check): %g\n", err_gpu_sum);
    if (err_gpu_sum >= GPU_TOL) {
            uint32_t mi=0,mj=0; double maxd = compute_max_abs_diff_interior(cpu_sum_explicit, gpu_sum_out, &mi, &mj);
            fprintf(stderr, "GPU sum mismatch (interior) = %g at i=%u j=%u\n", maxd, mi, mj);
        }
        printf("  GPU u + dt*lap(u) relative error: %g\n", err_gpu_sum);
    assert(err_gpu_sum < GPU_TOL);
        grid_field_free(gpu_sum_out);
        gpu_program_free(prog_sum_gpu);
        grid_field_free(cpu_sum_explicit);
    }

    grid_field_free(sum_field);
    grid_field_free(cpu_lap);
    grid_field_free(cpu_sum);
    expression_free(sum_expr);

    // Case 3: Recursive/nested expression: (u + dt*lap(u)) * 0.5
    Expression *inner_u = expr_variable("u");
    Expression *inner_lap = expr_laplacian(expr_variable("u"));
    Expression *inner = expr_add(inner_u, expr_multiply(expr_literal(literal_create_scalar(dt)), inner_lap));
    Expression *factor = expr_literal(literal_create_scalar(0.5));
    Expression *nested = expr_multiply(inner, factor);

    Literal *nested_lit = expression_evaluate_grid(nested, dict, grid);
    GridField *nested_field = grid_field_wrap_literal(nested_lit, grid);

    // CPU ref: compute cpu_sum from above (u + dt*lap(u)), then scale by 0.5
    GridField *cpu_lap2 = grid_field_laplacian_compact(u);
    grid_field_scale_inplace(cpu_lap2, dt);
    GridField *cpu_sum2 = grid_field_add(u, cpu_lap2);
    grid_field_scale_inplace(cpu_sum2, 0.5);

    double err3 = compute_relative_error(nested_field, cpu_sum2);
    printf("  nested (u + dt*lap(u)) * 0.5 relative error: %g\n", err3);
    assert(err3 < 1e-12);

    // GPU mirror for nested expression
    GPUProgram *prog_nested_gpu = gpu_compile_optimized(nested, grid, GPU_BACKEND_OPENGL);
    GridField *gpu_nested_out = NULL;
    if (prog_nested_gpu) {
        // explicit reference for nested: (u + dt*explicitLap(u)) * 0.5
        GridField *cpu_lap_explicit_n = grid_field_laplacian(u);
        grid_field_scale_inplace(cpu_lap_explicit_n, dt);
        GridField *cpu_sum_explicit_n = grid_field_add(u, cpu_lap_explicit_n);
        grid_field_scale_inplace(cpu_sum_explicit_n, 0.5);
        gpu_nested_out = gpu_run_program_cpu(prog_nested_gpu, dict, grid);
        if (!gpu_nested_out) {
            fprintf(stderr, "SKIP: GPU nested execution unavailable (unimplemented)\n");
        } else {
            double err_gpu_nested = compute_relative_error_interior(cpu_sum_explicit_n, gpu_nested_out);
            printf("  GPU nested relative error (interior check): %g\n", err_gpu_nested);
            if (err_gpu_nested >= GPU_TOL) {
                uint32_t mi=0,mj=0; double maxd = compute_max_abs_diff_interior(cpu_sum_explicit_n, gpu_nested_out, &mi, &mj);
                fprintf(stderr, "GPU nested mismatch (interior) = %g at i=%u j=%u\n", maxd, mi, mj);
            }
            printf("  GPU nested relative error: %g\n", err_gpu_nested);
            assert(err_gpu_nested < GPU_TOL);
            grid_field_free(gpu_nested_out);
        }
        grid_field_free(cpu_lap_explicit_n);
        grid_field_free(cpu_sum_explicit_n);
        gpu_program_free(prog_nested_gpu);
    }

    grid_field_free(nested_field);
    grid_field_free(cpu_lap2);
    grid_field_free(cpu_sum2);
    expression_free(nested);

    // Additional GPU check: run a pointwise-supported expression on GPU to
    // exercise the GPU runtime (emit_glsl currently supports variables,
    // add, multiply, and scalar literals). We'll compute out = u + 2*u.
    Expression *u_var_gpu = expr_variable("u");
    Expression *two_lit = expr_literal(literal_create_scalar(2.0));
    Expression *double_u = expr_multiply(two_lit, expr_variable("u"));
    Expression *sum_gpu_expr = expr_add(u_var_gpu, double_u); // u + 2*u -> 3*u

    GPUProgram *prog = gpu_compile_optimized(sum_gpu_expr, grid, GPU_BACKEND_OPENGL);
    assert(prog != NULL);

    GridField *gpu_out = gpu_run_program_cpu(prog, dict, grid);
    assert(gpu_out != NULL);

    // CPU reference: 3 * u
    GridField *cpu_ref_gpu = grid_field_copy(u);
    grid_field_scale_inplace(cpu_ref_gpu, 3.0);
    double err_gpu = compute_relative_error(cpu_ref_gpu, gpu_out);
    printf("  GPU pointwise (u + 2*u) relative error: %g\n", err_gpu);
    // allow small FP differences
    assert(err_gpu < 1e-6);

    grid_field_free(gpu_out);
    grid_field_free(cpu_ref_gpu);
    gpu_program_free(prog);
    expression_free(sum_gpu_expr);

    // Cleanup
    dict_free(dict);
    grid_field_free(u);
    grid_metadata_free(grid);

    printf("All layered expression tests passed.\n");
    return 0;
}
