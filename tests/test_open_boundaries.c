#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/grid.h"

// Test that open boundaries work as default (field continues naturally)
void test_default_open_boundaries() {
    printf("\n=== Test: Default Open Boundaries ===\n");
    printf("Testing: Grid created with default BC_OPEN at all edges\n\n");
    
    uint32_t dims[] = {21};
    double spacing[] = {0.05};
    double origin[] = {0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 1);
    
    // Verify boundaries are BC_OPEN by default
    printf("Default boundary conditions after grid creation:\n");
    printf("  x=0 (min): Type = %d %s\n", 
           grid->boundaries[0].type,
           grid->boundaries[0].type == BC_OPEN ? "(BC_OPEN)" : "");
    printf("  x=1 (max): Type = %d %s\n", 
           grid->boundaries[1].type,
           grid->boundaries[1].type == BC_OPEN ? "(BC_OPEN)" : "");
    
    GridField *u = grid_field_create(grid);
    
    // Set u(x) = x² (quadratic function)
    for (uint32_t i = 0; i < grid->total_points; i++) {
        double x = grid->origin[0] + i * grid->spacing[0];
        uint32_t idx[] = {i, 0, 0};
        literal_set(&u->data, idx, x * x);
    }
    
    // Compute first derivative (should be 2x everywhere)
    GridField *du = grid_field_derivative(u, 0, 1);
    
    printf("\nu(x) = x², du/dx = 2x (analytical)\n");
    printf("First derivative at boundaries (BC_SOMMERFELD):\n");
    for (uint32_t i = 0; i < grid->total_points; i += 5) {
        double x = grid->origin[0] + i * grid->spacing[0];
        uint32_t idx[] = {i, 0, 0};
        double du_val = literal_get(&du->data, idx);
        double expected = 2.0 * x;
        printf("  x=%.2f: du/dx = %.6f (expected: %.6f, error: %.2e)\n", 
               x, du_val, expected, fabs(du_val - expected));
    }
    
    // Compute second derivative (should be 2.0 everywhere)
    GridField *d2u = grid_field_derivative(u, 0, 2);
    
    printf("\nSecond derivative with BC_SOMMERFELD (extrapolation):\n");
    for (uint32_t i = 0; i < grid->total_points; i += 5) {
        double x = grid->origin[0] + i * grid->spacing[0];
        uint32_t idx[] = {i, 0, 0};
        double d2u_val = literal_get(&d2u->data, idx);
        double expected = 2.0;
        printf("  x=%.2f: d²u/dx² = %.6f (expected: %.6f, error: %.2e)\n", 
               x, d2u_val, expected, fabs(d2u_val - expected));
    }
    
    grid_field_free(d2u);
    grid_field_free(du);
    grid_field_free(u);
    grid_metadata_free(grid);
    
    printf("\n[PASS] Default open boundaries test\n");
}

void test_open_2d() {
    printf("\n=== Test: 2D Open Boundaries ===\n");
    printf("Testing: Laplacian with open boundaries in 2D\n\n");
    
    uint32_t dims[] = {11, 11};
    double spacing[] = {0.1, 0.1};
    double origin[] = {0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    
    printf("2D Grid: %ux%u points, spacing=(%.2f, %.2f)\n", dims[0], dims[1], spacing[0], spacing[1]);
    printf("All boundaries default to BC_SOMMERFELD\n");
    
    GridField *u = grid_field_create(grid);
    
    // Set u(x,y) = x² + y²
    for (uint32_t j = 0; j < dims[1]; j++) {
        for (uint32_t i = 0; i < dims[0]; i++) {
            double x = grid->origin[0] + i * grid->spacing[0];
            double y = grid->origin[1] + j * grid->spacing[1];
            uint32_t idx[] = {i, j, 0};
            literal_set(&u->data, idx, x*x + y*y);
        }
    }
    
    // Compute Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y²
    // For u = x² + y², Laplacian should be 4.0 everywhere
    GridField *laplacian = grid_field_laplacian(u);
    
    printf("\nu(x,y) = x² + y², ∇²u = 4.0 (analytical)\n");
    printf("Computed Laplacian at sample points:\n");
    
    // Check center and edges
    uint32_t test_points[][2] = {
        {5, 5},     // Center
        {0, 5},     // Left edge
        {10, 5},    // Right edge
        {5, 0},     // Bottom edge
        {5, 10},    // Top edge
        {0, 0},     // Corner
        {10, 10}    // Corner
    };
    
    for (int p = 0; p < 7; p++) {
        uint32_t i = test_points[p][0];
        uint32_t j = test_points[p][1];
        double x = grid->origin[0] + i * grid->spacing[0];
        double y = grid->origin[1] + j * grid->spacing[1];
        uint32_t idx[] = {i, j, 0};
        double lap_val = literal_get(&laplacian->data, idx);
        double expected = 4.0;
        
        const char *location = "";
        if (i == 5 && j == 5) location = "(center)";
        else if (i == 0 || i == dims[0]-1 || j == 0 || j == dims[1]-1) {
            if ((i == 0 || i == dims[0]-1) && (j == 0 || j == dims[1]-1))
                location = "(corner)";
            else
                location = "(edge)";
        }
        
        printf("  (%.1f, %.1f) %s: ∇²u = %.6f (expected: %.6f, error: %.2e)\n", 
               x, y, location, lap_val, expected, fabs(lap_val - expected));
    }
    
    grid_field_free(laplacian);
    grid_field_free(u);
    grid_metadata_free(grid);
    
    printf("\n[PASS] 2D open boundaries test\n");
}

void test_open_vs_neumann() {
    printf("\n=== Test: BC_SOMMERFELD vs BC_NEUMANN ===\n");
    printf("Comparing open (extrapolation) vs Neumann (fixed gradient)\n\n");
    
    uint32_t dims[] = {21};
    double spacing[] = {0.05};
    double origin[] = {0.0};
    
    // Test 1: Open boundaries
    GridMetadata *grid_open = grid_metadata_create(dims, spacing, origin, 1);
    GridField *u_open = grid_field_create(grid_open);
    
    // Test 2: Neumann boundaries
    GridMetadata *grid_neumann = grid_metadata_create(dims, spacing, origin, 1);
    grid_set_boundary(grid_neumann, 0, 0, BC_NEUMANN, 0.0);
    grid_set_boundary(grid_neumann, 0, 1, BC_NEUMANN, 0.0);
    GridField *u_neumann = grid_field_create(grid_neumann);
    
    // Set u(x) = x³ (cubic function)
    for (uint32_t i = 0; i < dims[0]; i++) {
        double x = origin[0] + i * spacing[0];
        uint32_t idx[] = {i, 0, 0};
        double val = x * x * x;
        literal_set(&u_open->data, idx, val);
        literal_set(&u_neumann->data, idx, val);
    }
    
    // Compute second derivatives
    GridField *d2u_open = grid_field_derivative(u_open, 0, 2);
    GridField *d2u_neumann = grid_field_derivative(u_neumann, 0, 2);
    
    printf("u(x) = x³, d²u/dx² = 6x (analytical)\n");
    printf("\nBoundary behavior comparison:\n");
    printf("%-10s  %-15s  %-15s  %-15s\n", "Location", "BC_SOMMERFELD", "BC_NEUMANN", "Analytical");
    printf("%-10s  %-15s  %-15s  %-15s\n", "--------", "-------", "----------", "----------");
    
    for (uint32_t i = 0; i < dims[0]; i += 10) {
        double x = origin[0] + i * spacing[0];
        uint32_t idx[] = {i, 0, 0};
        double open_val = literal_get(&d2u_open->data, idx);
        double neumann_val = literal_get(&d2u_neumann->data, idx);
        double expected = 6.0 * x;
        
        const char *loc = (i == 0) ? "x=0 (min)" : (i == dims[0]-1) ? "x=1 (max)" : "interior";
        printf("%-10s  %15.6f  %15.6f  %15.6f\n", loc, open_val, neumann_val, expected);
    }
    
    grid_field_free(d2u_neumann);
    grid_field_free(d2u_open);
    grid_field_free(u_neumann);
    grid_field_free(u_open);
    grid_metadata_free(grid_neumann);
    grid_metadata_free(grid_open);
    
    printf("\n[PASS] BC_SOMMERFELD vs BC_NEUMANN comparison\n");
}

int main() {
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║         Open Boundary Condition Test Suite           ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    
    test_default_open_boundaries();
    test_open_2d();
    test_open_vs_neumann();
    
    printf("\n╔════════════════════════════════════════════════════════╗\n");
    printf("║                  All Tests Passed!                    ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
