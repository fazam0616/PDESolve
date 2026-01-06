#include "../include/grid.h"
#include "../include/literal.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Test function: f(x,y) = x^2 + y^2
double test_func_2d(double x, double y, double z) {
    (void)z;  // Unused
    return x*x + y*y;
}

// Wrapper for grid_field_init_from_function: f(x,y) = x^2 + y^2
Literal* test_func_2d_literal(const double *coords, int n_dims) {
    double x = (n_dims > 0) ? coords[0] : 0.0;
    double y = (n_dims > 1) ? coords[1] : 0.0;
    double val = test_func_2d(x, y, 0.0);
    return literal_create_scalar(val);
}

// Test function: f(x,y,z) = sin(x) * cos(y) * exp(-z)
double test_func_3d(double x, double y, double z) {
    return sin(x) * cos(y) * exp(-z);
}

// Wrapper for grid_field_init_from_function: f(x,y,z) = sin(x) * cos(y) * exp(-z)
Literal* test_func_3d_literal(const double *coords, int n_dims) {
    double x = (n_dims > 0) ? coords[0] : 0.0;
    double y = (n_dims > 1) ? coords[1] : 0.0;
    double z = (n_dims > 2) ? coords[2] : 0.0;
    double val = test_func_3d(x, y, z);
    return literal_create_scalar(val);
}

void test_grid_creation() {
    printf("Test 1: Grid Creation\n");
    printf("=====================\n");
    
    uint32_t dims[3] = {10, 20, 1};
    double spacing[3] = {0.1, 0.2, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 3);
    assert(grid != NULL);
    
    printf("  Grid dimensions: %u x %u x %u\n", grid->dims[0], grid->dims[1], grid->dims[2]);
    printf("  Grid spacing: %.2f x %.2f x %.2f\n", grid->spacing[0], grid->spacing[1], grid->spacing[2]);
    printf("  Total points: %u\n", grid->total_points);
    printf("  Physical extent: %.2f x %.2f x %.2f\n", grid->extent[0], grid->extent[1], grid->extent[2]);
    
    assert(grid->total_points == 200);
    assert(fabs(grid->extent[0] - 1.0) < 1e-10);
    assert(fabs(grid->extent[1] - 4) < 1e-10);
    
    grid_metadata_free(grid);
    printf("  [OK] Grid creation successful\n\n");
}

void test_grid_indexing() {
    printf("Test 2: Grid Indexing\n");
    printf("=====================\n");
    
    uint32_t dims[3] = {5, 4, 3};
    double spacing[3] = {1.0, 1.0, 1.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, NULL, 3);
    
    // Test index <-> linear conversions
    uint32_t indices[3] = {2, 1, 1};
    uint32_t linear = grid_index_to_linear(grid, indices);
    
    printf("  Index [%u, %u, %u] -> Linear %u\n", indices[0], indices[1], indices[2], linear);
    
    uint32_t recovered[3];
    grid_linear_to_index(grid, linear, recovered);
    
    printf("  Linear %u -> Index [%u, %u, %u]\n", linear, recovered[0], recovered[1], recovered[2]);
    
    assert(recovered[0] == indices[0]);
    assert(recovered[1] == indices[1]);
    assert(recovered[2] == indices[2]);
    
    printf("  [OK] Index conversions correct\n\n");
    
    grid_metadata_free(grid);
}

void test_coordinate_conversion() {
    printf("Test 3: Coordinate Conversion\n");
    printf("==============================\n");
    
    uint32_t dims[3] = {11, 11, 1};
    double spacing[3] = {0.1, 0.1, 1.0};
    double origin[3] = {-0.5, -0.5, 0.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 3);
    
    // Test index -> coord
    uint32_t indices[3] = {5, 5, 0};  // Center point
    double coords[3];
    grid_index_to_coord(grid, indices, coords);
    
    printf("  Index [%u, %u, %u] -> Coord (%.2f, %.2f, %.2f)\n", 
           indices[0], indices[1], indices[2], coords[0], coords[1], coords[2]);
    
    assert(fabs(coords[0] - 0.0) < 1e-10);
    assert(fabs(coords[1] - 0.0) < 1e-10);
    
    // Test coord -> index
    double test_coords[3] = {0.15, -0.20, 0.0};
    uint32_t result_indices[3];
    bool success = grid_coord_to_index(grid, test_coords, result_indices);
    
    printf("  Coord (%.2f, %.2f, %.2f) -> Index [%u, %u, %u]\n",
           test_coords[0], test_coords[1], test_coords[2],
           result_indices[0], result_indices[1], result_indices[2]);
    
    assert(success);
    assert(result_indices[0] == 7);  // (0.15 - (-0.5)) / 0.1 = 6.5 → 7
    assert(result_indices[1] == 3);  // (-0.20 - (-0.5)) / 0.1 = 3.0 → 3
    
    printf("  [OK] Coordinate conversions correct\n\n");
    
    grid_metadata_free(grid);
}

void test_boundary_detection() {
    printf("Test 4: Boundary Detection\n");
    printf("==========================\n");
    
    uint32_t dims[3] = {5, 5, 1};
    double spacing[3] = {1.0, 1.0, 1.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, NULL, 3);
    
    // Test corner (boundary)
    uint32_t corner[3] = {0, 0, 0};
    assert(grid_is_boundary(grid, corner));
    printf("  Point [0, 0, 0]: %s\n", grid_is_boundary(grid, corner) ? "Boundary" : "Interior");
    
    // Test edge (boundary)
    uint32_t edge[3] = {4, 2, 0};
    assert(grid_is_boundary(grid, edge));
    printf("  Point [4, 2, 0]: %s\n", grid_is_boundary(grid, edge) ? "Boundary" : "Interior");
    
    // Test interior
    uint32_t interior[3] = {2, 2, 0};
    assert(!grid_is_boundary(grid, interior));
    printf("  Point [2, 2, 0]: %s\n", grid_is_boundary(grid, interior) ? "Boundary" : "Interior");
    
    printf("  [OK] Boundary detection correct\n\n");
    
    grid_metadata_free(grid);
}

void test_field_operations() {
    printf("Test 5: Field Operations\n");
    printf("========================\n");
    
    uint32_t dims[3] = {5, 5, 1};
    double spacing[3] = {0.5, 0.5, 1.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, NULL, 3);
    
    char *vars[] = {"x", "y"};
    GridField *field = grid_field_create(grid);
    assert(field != NULL);

    printf("  Created field on grid with %u points\n", grid->total_points);
    
    // Test set/get
    uint32_t indices[3] = {2, 3, 0};
    Literal *set_val = literal_create_scalar(42.0);
    grid_field_set(field, indices, set_val);
    literal_free(set_val);
    Literal *get_val = grid_field_get(field, indices);
    double value = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  Set value 42.0 at [2, 3, 0], retrieved: %.1f\n", value);
    assert(fabs(value - 42.0) < 1e-10);
    if (get_val) literal_free(get_val);

    // Test fill
    Literal *fill_val = literal_create_scalar(3.14);
    grid_field_fill(field, fill_val);
    literal_free(fill_val);

    printf("GridField filled with 3.14\n");
    get_val = grid_field_get(field, indices);
    value = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  After fill(3.14), value at [2, 3, 0]: %.2f\n", value);
    assert(fabs(value - 3.14) < 1e-10);
    if (get_val) literal_free(get_val);

    printf("  [OK] Field operations correct\n\n");

    grid_field_free(field);
    grid_metadata_free(grid);
}

void test_field_initialization() {
    printf("Test 6: Field Initialization from Function\n");
    printf("==========================================\n");
    
    uint32_t dims[3] = {5, 5, 1};
    double spacing[3] = {1.0, 1.0, 1.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, NULL, 3);
    
    char *vars[] = {"x", "y"};
    GridField *field = grid_field_create(grid);
    
    // Initialize with f(x,y) = x^2 + y^2
    grid_field_init_from_function(field, test_func_2d_literal);

    printf("  Initialized field with f(x,y) = x^2 + y^2\n");
    
    // Check center point: (2, 2) -> should be 2^2 + 2^2 = 8
    uint32_t indices[3] = {2, 2, 0};
    Literal *get_val = grid_field_get(field, indices);
    double value = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  Retrieved value at (2,2): %.1f\n", value);
    printf("  f(2, 2) = %.1f (expected 8.0)\n", value);
    assert(fabs(value - 8.0) < 1e-10);
    if (get_val) literal_free(get_val);

    // Check corner: (0, 0) -> 0
    uint32_t corner[3] = {0, 0, 0};
    get_val = grid_field_get(field, corner);
    value = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  f(0, 0) = %.1f (expected 0.0)\n", value);
    assert(fabs(value - 0.0) < 1e-10);
    if (get_val) literal_free(get_val);

    // Check edge: (4, 0) -> 16
    uint32_t edge[3] = {4, 0, 0};
    get_val = grid_field_get(field, edge);
    value = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  f(4, 0) = %.1f (expected 16.0)\n", value);
    assert(fabs(value - 16.0) < 1e-10);
    if (get_val) literal_free(get_val);

    printf("  [OK] Field initialization correct\n\n");

    grid_field_free(field);
    grid_metadata_free(grid);
}

void test_derivative() {
    printf("Test 7: Derivative Computation\n");
    printf("==============================\n");
    
    uint32_t dims[3] = {11, 11, 1};
    double spacing[3] = {0.1, 0.1, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 3);
    
    char *vars[] = {"x", "y"};
    GridField *field = grid_field_create(grid);
    
    // f(x,y) = x^2 + y^2
    // df/dx = 2x, df/dy = 2y
    grid_field_init_from_function(field, test_func_2d_literal);
    
    // Compute df/dx
    GridField *dfdx = grid_field_derivative(field, 0, 1);
    assert(dfdx != NULL);
    
    // Check at point (0.5, 0.3): df/dx should be ~ 2*0.5 = 1.0
    uint32_t indices[3] = {5, 3, 0};  // x=0.5, y=0.3
    Literal *get_val = grid_field_get(dfdx, indices);
    double deriv = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  df/dx at (0.5, 0.3) = %.3f (expected ~ 1.0)\n", deriv);
    assert(fabs(deriv - 1.0) < 0.1);  // Finite difference error
    if (get_val) literal_free(get_val);

    // Compute df/dy
    GridField *dfdy = grid_field_derivative(field, 1, 1);
    get_val = grid_field_get(dfdy, indices);
    deriv = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  df/dy at (0.5, 0.3) = %.3f (expected ~ 0.6)\n", deriv);
    if (get_val) literal_free(get_val);
    assert(fabs(deriv - 0.6) < 0.1);
    
    printf("  [OK] Derivative computation correct\n\n");
    
    grid_field_free(dfdy);
    grid_field_free(dfdx);
    grid_field_free(field);
    grid_metadata_free(grid);
}

void test_laplacian() {
    printf("Test 8: Laplacian Computation\n");
    printf("=============================\n");
    
    uint32_t dims[3] = {21, 21, 1};
    double spacing[3] = {0.1, 0.1, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 3);
    
    char *vars[] = {"x", "y"};
    GridField *field = grid_field_create(grid);
    
    // f(x,y) = x^2 + y^2
    // ∇²f = d²f/dx² + d²f/dy² = 2 + 2 = 4
    grid_field_init_from_function(field, test_func_2d_literal);

    printf("  Initialized field with f(x,y) = x^2 + y^2\n");
    
    GridField *lap = grid_field_laplacian(field);
    assert(lap != NULL);

    printf("  Computed Laplacian of the field\n");
    
    // Check at interior point
    uint32_t indices[3] = {10, 10, 0};  // Center
    Literal *get_val = grid_field_get(lap, indices);
    double laplacian = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  Laplacian(x^2 + y^2) at center = %.3f (expected ~= 4.0)\n", laplacian);
    assert(fabs(laplacian - 4.0) < 0.1);
    if (get_val) literal_free(get_val);

    // Check at another interior point
    indices[0] = 5;
    indices[1] = 15;
    get_val = grid_field_get(lap, indices);
    laplacian = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  Laplacian(x^2 + y^2) at (0.5, 1.5) = %.3f (expected ~= 4.0)\n", laplacian);
    assert(fabs(laplacian - 4.0) < 0.1);
    if (get_val) literal_free(get_val);
    
    printf("  [OK] Laplacian computation correct\n\n");
    
    grid_field_free(lap);
    grid_field_free(field);
    grid_metadata_free(grid);
}

void test_field_arithmetic() {
    printf("Test 9: Field Arithmetic\n");
    printf("========================\n");
    
    uint32_t dims[3] = {5, 5, 1};
    double spacing[3] = {1.0, 1.0, 1.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, NULL, 3);
    
    char *vars[] = {"x", "y"};
    GridField *a = grid_field_create(grid);
    GridField *b = grid_field_create(grid);
    
    Literal *fill_val_a = literal_create_scalar(3.0);
    Literal *fill_val_b = literal_create_scalar(2.0);
    grid_field_fill(a, fill_val_a);
    grid_field_fill(b, fill_val_b);
    literal_free(fill_val_a);
    literal_free(fill_val_b);

    // Test addition
    GridField *sum = grid_field_add(a, b);
    uint32_t indices[3] = {2, 2, 0};
    Literal *get_val = grid_field_get(sum, indices);
    double value = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  3.0 + 2.0 = %.1f\n", value);
    assert(fabs(value - 5.0) < 1e-10);
    if (get_val) literal_free(get_val);

    // Test multiplication
    GridField *prod = grid_field_multiply(a, b);
    get_val = grid_field_get(prod, indices);
    value = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  3.0 * 2.0 = %.1f\n", value);
    assert(fabs(value - 6.0) < 1e-10);
    if (get_val) literal_free(get_val);

    // Test scaling
    GridField *scaled = grid_field_scale(a, 1.5);
    get_val = grid_field_get(scaled, indices);
    value = get_val ? literal_get(get_val, (uint32_t[]){0,0,0}) : 0.0;
    printf("  1.5 * 3.0 = %.1f\n", value);
    if (get_val) literal_free(get_val);
    assert(fabs(value - 4.5) < 1e-10);

    printf("  [OK] Field arithmetic correct\n\n");

    grid_field_free(scaled);
    grid_field_free(prod);
    grid_field_free(sum);
    grid_field_free(b);
    grid_field_free(a);
    grid_metadata_free(grid);
}

void test_field_norm() {
    printf("Test 10: Field Norm\n");
    printf("===================\n");
    
    uint32_t dims[3] = {3, 3, 1};
    double spacing[3] = {1.0, 1.0, 1.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, NULL, 3);
    
    char *vars[] = {"x", "y"};
    GridField *field = grid_field_create(grid);
    
    // Fill with constant value 2.0
    // Norm = sqrt(9 * 4.0) = 6.0
    Literal *fill_val = literal_create_scalar(2.0);
    grid_field_fill(field, fill_val);
    literal_free(fill_val);

    double norm = grid_field_norm(field);

    printf("  Norm of field (9 points, each = 2.0) = %.1f (expected 6.0)\n", norm);
    assert(fabs(norm - 6.0) < 1e-10);

    printf("  [OK] Field norm correct\n\n");

    grid_field_free(field);
    grid_metadata_free(grid);
}

int main() {
    printf("===========================================\n");
    printf("Grid Infrastructure Test Suite\n");
    printf("===========================================\n\n");
    
    test_grid_creation();
    test_grid_indexing();
    test_coordinate_conversion();
    test_boundary_detection();
    test_field_operations();
    test_field_initialization();
    test_derivative();
    test_laplacian();
    test_field_arithmetic();
    test_field_norm();
    
    printf("===========================================\n");
    printf("All 10 tests passed! [OK]\n");
    printf("===========================================\n");
    
    return 0;
}
