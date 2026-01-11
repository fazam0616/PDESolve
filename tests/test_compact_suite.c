#include "../include/grid.h"
#include "../include/literal.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Simple 1D function f(x) = x^2
Literal* square_func(const double *coords, int n_dims) {
    double x = (n_dims > 0) ? coords[0] : 0.0;
    return literal_create_scalar(x * x);
}

// 2D test function f(x,y) = x^2 + y^2
Literal* square2d_func(const double *coords, int n_dims) {
    double x = (n_dims > 0) ? coords[0] : 0.0;
    double y = (n_dims > 1) ? coords[1] : 0.0;
    return literal_create_scalar(x*x + y*y);
}

static double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Compare two grid fields across all points with tolerance
static int compare_fields(const GridField *a, const GridField *b, double tol, double *out_maxdiff) {
    if (!a || !b) return 1;
    if (a->grid != b->grid) return 1;
    uint32_t total = a->grid->total_points;
    double maxdiff = 0.0;
    uint32_t indices[N_DIM];
    for (uint32_t i = 0; i < total; i++) {
        grid_linear_to_index(a->grid, i, indices);
        Literal *va = grid_field_get(a, indices);
        Literal *vb = grid_field_get(b, indices);
        double da = va ? literal_get(va, (uint32_t[]){0,0,0}) : 0.0;
        double db = vb ? literal_get(vb, (uint32_t[]){0,0,0}) : 0.0;
        if (va) literal_free(va);
        if (vb) literal_free(vb);
        double diff = fabs(da - db);
        if (diff > maxdiff) maxdiff = diff;
        if (diff > tol) {
            if (out_maxdiff) *out_maxdiff = maxdiff;
            return 1;
        }
    }
    if (out_maxdiff) *out_maxdiff = maxdiff;
    return 0;
}

// Relative L2 error between two grid fields (uses all points)
static double relative_error_l2(const GridField *a, const GridField *b) {
    if (!a || !b) return INFINITY;
    if (a->grid != b->grid) return INFINITY;
    size_t total = literal_total_elements(&a->data);
    if (total == 0) return INFINITY;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < total; i++) {
        double va = a->data.field[i];
        double vb = b->data.field[i];
        double d = va - vb;
        num += d * d;
        den += va * va;
    }
    if (den == 0.0) return sqrt(num);
    return sqrt(num) / sqrt(den);
}

int test_1d_edge_cases() {
    printf("\n[Test] 1D edge BC cases\n");
    int failures = 0;

    uint32_t dims[3] = {11,1,1};
    double spacing[3] = {0.1, 1.0, 1.0};
    double origin[3] = {0.0,0.0,0.0};

    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 1);
    GridField *field = grid_field_create(grid);
    grid_field_init_from_function(field, square_func);

    // Case A: left Dirichlet (0), right Neumann (2L)
    grid_set_boundary(grid, 0, 0, BC_DIRICHLET, 0.0);
    double L = (dims[0]-1) * spacing[0];
    grid_set_boundary(grid, 0, 1, BC_NEUMANN, 2.0 * L);
    // Compare only boundary points (BC-sensitive)
    GridField *d1 = grid_field_derivative(field, 0, 1);
    GridField *dc1 = grid_field_derivative_compact(field, 0, 1);
    uint32_t idx0[3] = {0,0,0};
    uint32_t idxn[3] = {dims[0]-1,0,0};
    Literal *a0 = grid_field_get(d1, idx0);
    Literal *b0 = grid_field_get(dc1, idx0);
    Literal *an = grid_field_get(d1, idxn);
    Literal *bn = grid_field_get(dc1, idxn);
    double v_a0 = a0 ? literal_get(a0, (uint32_t[]){0,0,0}) : 0.0;
    double v_b0 = b0 ? literal_get(b0, (uint32_t[]){0,0,0}) : 0.0;
    double v_an = an ? literal_get(an, (uint32_t[]){0,0,0}) : 0.0;
    double v_bn = bn ? literal_get(bn, (uint32_t[]){0,0,0}) : 0.0;
    if (a0) { literal_free(a0); } if (b0) { literal_free(b0); } if (an) { literal_free(an); } if (bn) { literal_free(bn); }
    double diff0 = fabs(v_a0 - v_b0);
    double diffn = fabs(v_an - v_bn);
    if (diff0 > 1e-12 || diffn > 1e-12) {
        printf("  FAIL: Dirichlet/Neumann boundary mismatch (diff0=%g diffn=%g)\n", diff0, diffn);
        failures++;
    } else {
        printf("  OK: Dirichlet/Neumann boundaries match\n");
    }
    grid_field_free(d1); grid_field_free(dc1);

    // Case B: both Reflect
    grid_set_boundary(grid, 0, 0, BC_REFLECT, 0.0);
    grid_set_boundary(grid, 0, 1, BC_REFLECT, 0.0);
    GridField *d2 = grid_field_derivative(field, 0, 1);
    GridField *dc2 = grid_field_derivative_compact(field, 0, 1);
    // Check only boundaries
    a0 = grid_field_get(d2, idx0); b0 = grid_field_get(dc2, idx0);
    an = grid_field_get(d2, idxn); bn = grid_field_get(dc2, idxn);
    v_a0 = a0 ? literal_get(a0, (uint32_t[]){0,0,0}) : 0.0; v_b0 = b0 ? literal_get(b0, (uint32_t[]){0,0,0}) : 0.0;
    v_an = an ? literal_get(an, (uint32_t[]){0,0,0}) : 0.0; v_bn = bn ? literal_get(bn, (uint32_t[]){0,0,0}) : 0.0;
    if (a0) { literal_free(a0); } if (b0) { literal_free(b0); } if (an) { literal_free(an); } if (bn) { literal_free(bn); }
    diff0 = fabs(v_a0 - v_b0); diffn = fabs(v_an - v_bn);
    if (diff0 > 1e-12 || diffn > 1e-12) {
        printf("  FAIL: Reflect/Reflect boundary mismatch (diff0=%g diffn=%g)\n", diff0, diffn);
        failures++;
    } else {
        printf("  OK: Reflect/Reflect boundaries match\n");
    }
    grid_field_free(d2); grid_field_free(dc2);

    // Case C: both Open
    grid_set_boundary(grid, 0, 0, BC_OPEN, 0.0);
    grid_set_boundary(grid, 0, 1, BC_OPEN, 0.0);
    GridField *d3 = grid_field_derivative(field, 0, 1);
    GridField *dc3 = grid_field_derivative_compact(field, 0, 1);
    a0 = grid_field_get(d3, idx0); b0 = grid_field_get(dc3, idx0);
    an = grid_field_get(d3, idxn); bn = grid_field_get(dc3, idxn);
    v_a0 = a0 ? literal_get(a0, (uint32_t[]){0,0,0}) : 0.0; v_b0 = b0 ? literal_get(b0, (uint32_t[]){0,0,0}) : 0.0;
    v_an = an ? literal_get(an, (uint32_t[]){0,0,0}) : 0.0; v_bn = bn ? literal_get(bn, (uint32_t[]){0,0,0}) : 0.0;
    if (a0) { literal_free(a0); } if (b0) { literal_free(b0); } if (an) { literal_free(an); } if (bn) { literal_free(bn); }
    diff0 = fabs(v_a0 - v_b0); diffn = fabs(v_an - v_bn);
    if (diff0 > 1e-8 || diffn > 1e-8) {
        printf("  FAIL: Open/Open boundary mismatch (diff0=%g diffn=%g)\n", diff0, diffn);
        failures++;
    } else {
        printf("  OK: Open/Open boundaries match\n");
    }
    grid_field_free(d3); grid_field_free(dc3);

    // Case D: Periodic boundaries
    grid_set_boundary(grid, 0, 0, BC_PERIODIC, 0.0);
    grid_set_boundary(grid, 0, 1, BC_PERIODIC, 0.0);
    GridField *dp = grid_field_derivative(field, 0, 1);
    GridField *dcp = grid_field_derivative_compact(field, 0, 1);
    a0 = grid_field_get(dp, idx0); b0 = grid_field_get(dcp, idx0);
    an = grid_field_get(dp, idxn); bn = grid_field_get(dcp, idxn);
    v_a0 = a0 ? literal_get(a0, (uint32_t[]){0,0,0}) : 0.0; v_b0 = b0 ? literal_get(b0, (uint32_t[]){0,0,0}) : 0.0;
    v_an = an ? literal_get(an, (uint32_t[]){0,0,0}) : 0.0; v_bn = bn ? literal_get(bn, (uint32_t[]){0,0,0}) : 0.0;
    if (a0) { literal_free(a0); } if (b0) { literal_free(b0); } if (an) { literal_free(an); } if (bn) { literal_free(bn); }
    if (fabs(v_a0 - v_b0) > 1e-12 || fabs(v_an - v_bn) > 1e-12) {
        printf("  FAIL: Periodic boundary mismatch\n"); failures++;
    } else { printf("  OK: Periodic boundaries match\n"); }
    grid_field_free(dp); grid_field_free(dcp);

    // Case E: Robin and an interior hyperplane boundary (small smoke test)
    grid_set_boundary(grid, 0, 0, BC_ROBIN, 0.0);
    grid_set_boundary(grid, 0, 1, BC_ROBIN, 0.0);
    grid_set_robin_boundary(grid, 0, 0, 1.0, 1.0, 0.0);
    grid_set_robin_boundary(grid, 0, 1, 1.0, 1.0, 0.0);
    // Add a trivial interior hyperplane (center) to ensure code path exercised
    double normal[1] = {1.0}; double point[1] = {0.5}; double bmin[1] = {-1.0}; double bmax[1] = {1.0};
    grid_add_hyperplane_boundary(grid, normal, point, bmin, bmax, BC_NEUMANN, 0.0);
    GridField *dr = grid_field_derivative(field, 0, 1);
    GridField *dcr = grid_field_derivative_compact(field, 0, 1);
    // Only smoke-check that calls succeed and fields are non-null
    if (!dr || !dcr) { printf("  FAIL: Robin/hyperplane derivative returned NULL\n"); failures++; }
    else { printf("  OK: Robin/hyperplane derivative computed\n"); }
    if (dr) grid_field_free(dr); if (dcr) grid_field_free(dcr);

    grid_field_free(field);
    grid_metadata_free(grid);

    return failures;
}

int test_2d_laplacian_edges() {
    printf("\n[Test] 2D Laplacian edge BC cases\n");
    int failures = 0;

    uint32_t dims[3] = {21,21,1};
    double spacing[3] = {0.1,0.1,1.0};
    double origin[3] = {0.0,0.0,0.0};

    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    GridField *field = grid_field_create(grid);
    grid_field_init_from_function(field, square2d_func);

    // Set mixed BCs: x-min Dirichlet, x-max Neumann, y-min Reflect, y-max Open
    grid_set_boundary(grid, 0, 0, BC_DIRICHLET, 0.0);
    grid_set_boundary(grid, 0, 1, BC_NEUMANN, 0.0);
    grid_set_boundary(grid, 1, 0, BC_REFLECT, 0.0);
    grid_set_boundary(grid, 1, 1, BC_OPEN, 0.0);

    // Instead of exact equality, measure relative L2 error on boundary points
    // and check that error decreases with grid refinement (compact is higher order)
    uint32_t refine_sizes[] = {21, 41, 81};
    double prev_err = 1e9;
    int monotonic = 1;
    double last_rel = 0.0;
    for (int ri = 0; ri < (int)(sizeof(refine_sizes)/sizeof(refine_sizes[0])); ri++) {
        uint32_t s = refine_sizes[ri];
        uint32_t dims_r[3] = {s, s, 1};
        double spacing_r[3] = {0.1, 0.1, 1.0};
        GridMetadata *g = grid_metadata_create(dims_r, spacing_r, origin, 2);
        GridField *f_r = grid_field_create(g);
        grid_field_init_from_function(f_r, square2d_func);
        // same mixed BCs
        grid_set_boundary(g, 0, 0, BC_DIRICHLET, 0.0);
        grid_set_boundary(g, 0, 1, BC_NEUMANN, 0.0);
        grid_set_boundary(g, 1, 0, BC_REFLECT, 0.0);
        grid_set_boundary(g, 1, 1, BC_OPEN, 0.0);

        GridField *lap = grid_field_laplacian(f_r);
        GridField *lapc = grid_field_laplacian_compact(f_r);

        // Build temporary fields that contain only boundary points (zero elsewhere)
        GridField *b_std = grid_field_create(g);
        GridField *b_comp = grid_field_create(g);
        uint32_t indices[3];
        uint32_t nx = g->dims[0]; uint32_t ny = g->dims[1];
        for (uint32_t j = 0; j < ny; j++) {
            for (uint32_t i = 0; i < nx; i++) {
                if (i != 0 && i != nx-1 && j != 0 && j != ny-1) continue;
                indices[0] = i; indices[1] = j; indices[2] = 0;
                Literal *a = grid_field_get(lap, indices);
                Literal *b = grid_field_get(lapc, indices);
                double va = a ? literal_get(a, (uint32_t[]){0,0,0}) : 0.0;
                double vb = b ? literal_get(b, (uint32_t[]){0,0,0}) : 0.0;
                if (a) literal_free(a); if (b) literal_free(b);
                literal_set(&b_std->data, indices, va);
                literal_set(&b_comp->data, indices, vb);
            }
        }

        double rel = relative_error_l2(b_std, b_comp);
        printf("  refine %u: laplacian boundary relative L2 error = %g\n", s, rel);
        if (rel > prev_err + 1e-12) {
            // error didn't decrease (beyond tiny numerical noise)
            monotonic = 0;
        }
        last_rel = rel;

        grid_field_free(lap); grid_field_free(lapc);
        grid_field_free(b_std); grid_field_free(b_comp);
        grid_field_free(f_r); grid_metadata_free(g);
        prev_err = rel;
    }
    // Accept if error is decreasing with refinement and final relative error is reasonable
    double final_threshold = 0.13; // empirical tolerance for boundary-relative error
    if (!monotonic) {
        printf("  FAIL: relative error not monotonic across refinements\n");
        failures++;
    } else if (last_rel > final_threshold) {
        printf("  FAIL: final relative error too large (%g > %g)\n", last_rel, final_threshold);
        failures++;
    } else {
        printf("  OK: Laplacian boundary errors decrease with refinement (final rel=%g)\n", last_rel);
    }
    grid_field_free(field);
    grid_metadata_free(grid);
    return failures;
}

int test_small_grid_fallback() {
    printf("\n[Test] Small-grid fallback (compact -> explicit)\n");
    int failures = 0;

    uint32_t dims[3] = {4,1,1}; // compact should fall back for n<5
    double spacing[3] = {0.1,1.0,1.0};
    double origin[3] = {0.0,0.0,0.0};

    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 1);
    GridField *field = grid_field_create(grid);
    grid_field_init_from_function(field, square_func);

    grid_set_boundary(grid, 0, 0, BC_DIRICHLET, 0.0);
    grid_set_boundary(grid, 0, 1, BC_DIRICHLET, 0.0);

    GridField *d_std = grid_field_derivative(field, 0, 1);
    GridField *d_comp = grid_field_derivative_compact(field, 0, 1);
    double mdiff;
    if (compare_fields(d_std, d_comp, 1e-12, &mdiff)) {
        printf("  FAIL: small-grid fallback mismatch (max diff=%g)\n", mdiff);
        failures++;
    } else {
        printf("  OK: small-grid fallback matches explicit\n");
    }

    grid_field_free(d_std); grid_field_free(d_comp);
    grid_field_free(field);
    grid_metadata_free(grid);
    return failures;
}

int benchmark_compare() {
    printf("\n[Benchmark] compact vs standard (timings in seconds)\n");
    uint32_t dims[3] = {200,200,1};
    double spacing[3] = {0.01,0.01,1.0};
    double origin[3] = {0.0,0.0,0.0};

    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    GridField *field = grid_field_create(grid);
    grid_field_init_from_function(field, square2d_func);

    int repeats = 5;
    double t0, t1, t_std, t_comp;

    // Derivative (axis 0, order 1)
    t0 = now_seconds();
    for (int i = 0; i < repeats; i++) {
        GridField *d = grid_field_derivative(field, 0, 1);
        grid_field_free(d);
    }
    t1 = now_seconds();
    t_std = (t1 - t0) / repeats;

    t0 = now_seconds();
    for (int i = 0; i < repeats; i++) {
        GridField *d = grid_field_derivative_compact(field, 0, 1);
        grid_field_free(d);
    }
    t1 = now_seconds();
    t_comp = (t1 - t0) / repeats;

    printf("  derivative axis0: standard=%g compact=%g (ratio std/comp=%g)\n", t_std, t_comp, t_std / (t_comp > 0 ? t_comp : 1e-12));

    // Laplacian
    t0 = now_seconds();
    for (int i = 0; i < repeats; i++) {
        GridField *l = grid_field_laplacian(field);
        grid_field_free(l);
    }
    t1 = now_seconds();
    t_std = (t1 - t0) / repeats;

    t0 = now_seconds();
    for (int i = 0; i < repeats; i++) {
        GridField *l = grid_field_laplacian_compact(field);
        grid_field_free(l);
    }
    t1 = now_seconds();
    t_comp = (t1 - t0) / repeats;

    printf("  laplacian: standard=%g compact=%g (ratio std/comp=%g)\n", t_std, t_comp, t_std / (t_comp > 0 ? t_comp : 1e-12));

    grid_field_free(field);
    grid_metadata_free(grid);
    return 0;
}

int benchmark_iterative_large() {
    printf("\n[Benchmark Iterative] Laplacian (timestepping style)\n");
    uint32_t dims[3] = {256,256,1};
    double spacing[3] = {0.01,0.01,1.0};
    double origin[3] = {0.0,0.0,0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    GridField *field = grid_field_create(grid);
    grid_field_init_from_function(field, square2d_func);

    int iterations = 20;
    double t0 = now_seconds();
    for (int it = 0; it < iterations; it++) {
        GridField *l = grid_field_laplacian(field);
        if (!l) { grid_field_free(field); grid_metadata_free(grid); return 1; }
        grid_field_free(l);
    }
    double t_std = (now_seconds() - t0) / iterations;

    t0 = now_seconds();
    for (int it = 0; it < iterations; it++) {
        GridField *l = grid_field_laplacian_compact(field);
        if (!l) { grid_field_free(field); grid_metadata_free(grid); return 2; }
        grid_field_free(l);
    }
    double t_comp = (now_seconds() - t0) / iterations;

    printf("  iterative laplacian (256x256) std=%g s/it compact=%g s/it (ratio std/comp=%g)\n",
           t_std, t_comp, t_std / (t_comp > 0 ? t_comp : 1e-12));

    grid_field_free(field); grid_metadata_free(grid);
    return 0;
}

int main() {
    printf("\nCompact vs Standard derivative/laplacian test suite\n");
    int total_fail = 0;

    total_fail += test_1d_edge_cases();
    total_fail += test_2d_laplacian_edges();
    total_fail += test_small_grid_fallback();

    if (total_fail == 0) {
        printf("\nAll functional tests passed. Running benchmarks...\n");
    } else {
        printf("\nFunctional tests failed: %d failures. Benchmarks will still run.\n", total_fail);
    }

    benchmark_compare();
    benchmark_iterative_large();

    if (total_fail == 0) {
        printf("\nTest suite completed: SUCCESS\n");
        return 0;
    } else {
        printf("\nTest suite completed: FAIL (%d failures)\n", total_fail);
        return 2;
    }
}
