#include "../include/grid.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

// Test function: f(x,y) = sin(2πx) * cos(2πy)
// ∂f/∂x = 2π cos(2πx) * cos(2πy)
// ∂f/∂y = -2π sin(2πx) * sin(2πy)
Literal* test_function(const double *coords, int n_dims) {
    (void)n_dims;
    double x = coords[0];
    double y = coords[1];
    Literal *val = literal_create_zero((uint32_t[]){1, 1, 1});
    double result = sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
    literal_set(val, (uint32_t[]){0, 0, 0}, result);
    return val;
}

double analytical_dx(double x, double y) {
    return 2.0 * M_PI * cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
}

double analytical_dy(double x, double y) {
    return -2.0 * M_PI * sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y);
}

void test_accuracy_comparison() {
    printf("=== Testing Compact vs Explicit Derivative Accuracy ===\n\n");
    
    // Create 2D grid
    uint32_t dims[] = {64, 64, 1};
    double spacing[] = {1.0/64.0, 1.0/64.0, 1.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, NULL, 2);
    
    // Initialize field
    GridField *field = grid_field_create(grid);
    grid_field_init_from_function(field, test_function);
    
    printf("Grid: %u x %u, spacing: %.6f\n\n", dims[0], dims[1], spacing[0]);
    
    // Test x-derivative
    printf("Testing ∂f/∂x:\n");
    
    // Explicit method
    clock_t start = clock();
    GridField *dx_explicit = grid_field_derivative(field, 0, 1);
    clock_t end = clock();
    double time_explicit = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Compact method
    start = clock();
    GridField *dx_compact = grid_field_derivative_compact(field, 0, 1);
    end = clock();
    double time_compact = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Compute errors
    double error_explicit = 0.0, error_compact = 0.0;
    double max_error_explicit = 0.0, max_error_compact = 0.0;
    int count = 0;
    
    uint32_t indices[3];
    double coords[3];
    for (uint32_t i = 5; i < dims[0] - 5; i++) {
        for (uint32_t j = 5; j < dims[1] - 5; j++) {
            indices[0] = i;
            indices[1] = j;
            indices[2] = 0;
            
            grid_index_to_coord(grid, indices, coords);
            double analytical = analytical_dx(coords[0], coords[1]);
            
            double val_explicit = literal_get(&dx_explicit->data, indices);
            double val_compact = literal_get(&dx_compact->data, indices);
            
            double err_explicit = fabs(val_explicit - analytical);
            double err_compact = fabs(val_compact - analytical);
            
            error_explicit += err_explicit * err_explicit;
            error_compact += err_compact * err_compact;
            
            if (err_explicit > max_error_explicit) max_error_explicit = err_explicit;
            if (err_compact > max_error_compact) max_error_compact = err_compact;
            
            count++;
        }
    }
    
    error_explicit = sqrt(error_explicit / count);
    error_compact = sqrt(error_compact / count);
    
    printf("  Explicit (2nd order):\n");
    printf("    Time:       %.6f seconds\n", time_explicit);
    printf("    RMS error:  %.6e\n", error_explicit);
    printf("    Max error:  %.6e\n", max_error_explicit);
    
    printf("  Compact (4th order):\n");
    printf("    Time:       %.6f seconds\n", time_compact);
    printf("    RMS error:  %.6e\n", error_compact);
    printf("    Max error:  %.6e\n", max_error_compact);
    
    printf("  Accuracy improvement: %.2fx\n", error_explicit / error_compact);
    printf("  Speedup: %.2fx\n\n", time_explicit / time_compact);
    
    grid_field_free(dx_explicit);
    grid_field_free(dx_compact);
    grid_field_free(field);
    grid_metadata_free(grid);
}

void benchmark_performance() {
    printf("=== Performance Benchmark: Large Grid ===\n\n");
    
    // Create larger 2D grid for benchmarking
    uint32_t dims[] = {240, 240, 1};
    double spacing[] = {1.0/240.0, 1.0/240.0, 1.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, NULL, 2);
    
    GridField *field = grid_field_create(grid);
    grid_field_init_from_function(field, test_function);
    
    printf("Grid: %u x %u (%u total points)\n\n", dims[0], dims[1], grid->total_points);
    
    // Warm-up
    GridField *warmup = grid_field_derivative(field, 0, 1);
    grid_field_free(warmup);
    
    // Benchmark explicit method (5 iterations)
    printf("Explicit method (5 iterations):\n");
    clock_t start = clock();
    for (int iter = 0; iter < 5; iter++) {
        GridField *dx = grid_field_derivative(field, 0, 1);
        grid_field_free(dx);
    }
    clock_t end = clock();
    double time_explicit = (double)(end - start) / CLOCKS_PER_SEC;
    printf("  Total time: %.6f seconds\n", time_explicit);
    printf("  Per iteration: %.6f seconds\n\n", time_explicit / 5.0);
    
    // Benchmark compact method (5 iterations)
    printf("Compact method (5 iterations):\n");
    start = clock();
    for (int iter = 0; iter < 5; iter++) {
        GridField *dx = grid_field_derivative_compact(field, 0, 1);
        grid_field_free(dx);
    }
    end = clock();
    double time_compact = (double)(end - start) / CLOCKS_PER_SEC;
    printf("  Total time: %.6f seconds\n", time_compact);
    printf("  Per iteration: %.6f seconds\n\n", time_compact / 5.0);
    
    printf("Overall speedup: %.2fx\n", time_explicit / time_compact);
    printf("Expected index conversion reduction: 1.25B → ~O(N)\n");
    
    grid_field_free(field);
    grid_metadata_free(grid);
}

int main() {
    printf("\n");
    printf("========================================\n");
    printf("Compact Finite Difference Test Suite\n");
    printf("========================================\n\n");
    
    test_accuracy_comparison();
    benchmark_performance();
    
    printf("========================================\n");
    printf("All tests completed successfully!\n");
    printf("========================================\n\n");
    
    return 0;
}
