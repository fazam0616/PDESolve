#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../include/boundary_gpu.h"
#include "../include/grid.h"

int main(void) {
    uint32_t dims[3] = {16,16,1};
    double spacing[3] = {0.1,0.1,1.0};
    double origin[3] = {0,0,0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    assert(grid);

    // set x-min Dirichlet = 1.23, y-max Dirichlet = 4.56
    grid_set_boundary(grid, 0, 0, BC_DIRICHLET, 1.23);
    grid_set_boundary(grid, 1, 1, BC_DIRICHLET, 4.56);

    BoundaryMask *bm = boundary_mask_create(grid);
    assert(bm);

    // CPU apply to a zero-initialized field (reference)
    GridField *f = grid_field_create(grid);
    boundary_mask_apply_cpu(bm, f);

    // Now run GPU path: compile trivial expression 'u'. The GPU runtime will
    // auto-create and upload a BoundaryMask for the program based on grid
    // metadata, so tests should not attach or upload the mask explicitly.
    Expression *expr = expr_variable("u");
    GPUProgram *prog = gpu_compile_expression(expr, grid, GPU_BACKEND_OPENGL);
    if (!prog) { fprintf(stderr, "failed to compile gpu program\n"); return 2; }
    // Do NOT attach bm here; runtime will auto-generate/upload its own mask.

    // print emitted GLSL for debugging (must be global-scope uniforms)
    gpu_print_program(prog);

    // prepare input dictionary with a zero field named 'u'
    Dictionary *dict = dict_create(16);
    Literal *zero = literal_create_scalar(0.0);
    GridField *zero_field = grid_field_create(grid);
    grid_field_fill(zero_field, zero);
    dict_set(dict, "u", &zero_field->data);
    GridField *gpu_out = gpu_run_program_cpu(prog, dict, grid);
    if (!gpu_out) { fprintf(stderr, "gpu_run_program_cpu returned NULL\n"); gpu_program_free(prog); dict_free(dict); grid_field_free(zero_field); literal_free(zero); return 2; }

    // cleanup dict/literal wrappers
    grid_field_free(zero_field);
    literal_free(zero);
    dict_free(dict);

    // Compare GPU output to CPU-applied reference field 'f'
    uint32_t idx[3]; idx[2]=0;
    for (uint32_t j=0;j<dims[1];++j) {
        idx[0]=0; idx[1]=j;
        Literal *lr = grid_field_get(f, idx);
        Literal *lg = grid_field_get(gpu_out, idx);
        double vr = (lr && lr->field) ? lr->field[0] : 0.0;
        double vg = (lg && lg->field) ? lg->field[0] : 0.0;
    if (fabs(vr - vg) > 1e-6) { fprintf(stderr, "x-min GPU mismatch at j=%u: cpu=%.17g gpu=%.17g\n", j, vr, vg); return 2; }
    }
    for (uint32_t i=1;i<dims[0];++i) {
        idx[0]=i; idx[1]=dims[1]-1;
        Literal *lr = grid_field_get(f, idx);
        Literal *lg = grid_field_get(gpu_out, idx);
        double vr = (lr && lr->field) ? lr->field[0] : 0.0;
        double vg = (lg && lg->field) ? lg->field[0] : 0.0;
    if (fabs(vr - vg) > 1e-6) { fprintf(stderr, "y-max GPU mismatch at i=%u: cpu=%.17g gpu=%.17g\n", i, vr, vg); return 2; }
    }

    // cleanup
    boundary_mask_free(bm);
    grid_field_free(f);
    grid_field_free(gpu_out);
    gpu_program_free(prog);
    grid_metadata_free(grid);
    printf("test_boundary_mask: PASS\n");
    return 0;
}
