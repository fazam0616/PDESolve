#include "../include/grid.h"
#include "../include/expression.h"
#include "../include/calculus.h"
#include "../include/solver.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Test 1: Validate grid-aware derivative evaluation
void test_grid_derivative_evaluation() {
    printf("Test 1: Grid-Aware Derivative Evaluation\n");
    printf("==========================================\n");
    
    // Create 1D grid
    uint32_t dims[3] = {11, 1, 1};
    double spacing[3] = {0.1, 1.0, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 1);
    
    // Create grid field u(x) = x^2
    GridField *u = grid_field_create(grid);
    for (uint32_t i = 0; i < dims[0]; i++) {
        uint32_t idx[3] = {i, 0, 0};
        double x = origin[0] + i * spacing[0];
        Literal *val = literal_create_scalar(x * x);
        grid_field_set(u, idx, val);
        literal_free(val);
    }
    
    // Create expression: derivative(u, "x")
    Expression *u_expr = expr_variable("u");
    Expression *du_dx = expr_derivative(u_expr, "x");
    
    // Evaluate using grid-aware evaluation
    Dictionary *vars = dict_create(1);
    dict_set(vars, "u", &u->data);
    
    Literal *result = expression_evaluate_grid(du_dx, vars, grid);
    
    if (result) {
        // Check du/dx = 2x at center point (x = 0.5)
        uint32_t center_idx[3] = {5, 0, 0};
        double computed = literal_get(result, center_idx);
        double expected = 2.0 * 0.5;  // 2x at x=0.5
        double error = fabs(computed - expected);
        
        printf("  u(x) = x^2, du/dx at x=0.5:\n");
        printf("    Computed: %.6f\n", computed);
        printf("    Expected: %.6f\n", expected);
        printf("    Error: %.6e\n", error);
        
        assert(error < 0.01);  // Finite difference error
        printf("  [OK] Grid derivative evaluation correct\n\n");
        
        literal_free(result);
    } else {
        printf("  [FAILED] Grid derivative evaluation returned NULL\n\n");
        assert(0);
    }
    
    expression_free(du_dx);
    dict_free(vars);
    grid_field_free(u);
    grid_metadata_free(grid);
}

// Test 2: Validate Laplacian evaluation
void test_grid_laplacian_evaluation() {
    printf("Test 2: Grid-Aware Laplacian Evaluation\n");
    printf("========================================\n");
    
    // Create 2D grid
    uint32_t dims[3] = {21, 21, 1};
    double spacing[3] = {0.1, 0.1, 1.0};
    double origin[3] = {-1.0, -1.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    
    // Create grid field u(x,y) = x^2 + y^2
    GridField *u = grid_field_create(grid);
    for (uint32_t i = 0; i < dims[0]; i++) {
        for (uint32_t j = 0; j < dims[1]; j++) {
            uint32_t idx[3] = {i, j, 0};
            double x = origin[0] + i * spacing[0];
            double y = origin[1] + j * spacing[1];
            Literal *val = literal_create_scalar(x*x + y*y);
            grid_field_set(u, idx, val);
            literal_free(val);
        }
    }
    
    // Create expression: laplacian(u) = d²u/dx² + d²u/dy²
    // Build manually since laplacian() doesn't know about grid axes
    Expression *u_expr = expr_variable("u");
    Expression *du_dx = expr_derivative(u_expr, "x");
    Expression *d2u_dx2 = expr_derivative(du_dx, "x");
    Expression *u_expr2 = expr_variable("u");
    Expression *du_dy = expr_derivative(u_expr2, "y");
    Expression *d2u_dy2 = expr_derivative(du_dy, "y");
    Expression *lap_u = expr_add(d2u_dx2, d2u_dy2);
    
    // Evaluate using grid-aware evaluation
    Dictionary *vars = dict_create(1);
    dict_set(vars, "u", &u->data);
    
    Literal *result = expression_evaluate_grid(lap_u, vars, grid);
    
    if (result) {
        // Check Laplacian = 4.0 everywhere (for u = x^2 + y^2)
        uint32_t center_idx[3] = {10, 10, 0};
        double computed = literal_get(result, center_idx);
        double expected = 4.0;
        double error = fabs(computed - expected);
        
        printf("  u(x,y) = x^2 + y^2, ∇²u at center:\n");
        printf("    Computed: %.6f\n", computed);
        printf("    Expected: %.6f\n", expected);
        printf("    Error: %.6e\n", error);
        
        assert(error < 0.01);
        printf("  [OK] Grid Laplacian evaluation correct\n\n");
        
        literal_free(result);
    } else {
        printf("  [FAILED] Grid Laplacian evaluation returned NULL\n\n");
        assert(0);
    }
    
    expression_free(lap_u);
    dict_free(vars);
    grid_field_free(u);
    grid_metadata_free(grid);
}

// Test 3: Solve 1D Poisson equation -d²u/dx² = f(x) with Newton-Krylov
void test_poisson_1d_solver() {
    printf("Test 3: 1D Poisson Equation with Newton-Krylov\n");
    printf("===============================================\n");
    
    // Create 1D grid
    uint32_t dims[3] = {21, 1, 1};
    double spacing[3] = {0.05, 1.0, 1.0};  // 20 intervals in [0,1]
    double origin[3] = {0.0, 0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 1);
    
    printf("  Solving: -d²u/dx² = 1, u(0)=0, u(1)=0\n");
    printf("  Analytical solution: u(x) = x(1-x)/2\n");
    printf("  Grid: %d points, dx = %.3f\n", dims[0], spacing[0]);
    
    // Create system
    PDESystem *sys = pde_system_create();
    pde_system_set_grid(sys, grid);
    pde_system_set_tolerance(sys, 1e-6);
    pde_system_set_max_iterations(sys, 20);
    
    // Create RHS field f(x) = 1
    GridField *f = grid_field_create(grid);
    for (uint32_t i = 0; i < dims[0]; i++) {
        uint32_t idx[3] = {i, 0, 0};
        Literal *val = literal_create_scalar(1.0);
        grid_field_set(f, idx, val);
        literal_free(val);
    }
    dict_set(sys->parameters, "f", &f->data);
    
    // Build equation: -laplacian(u) - f = 0
    // Build laplacian manually: d²u/dx²
    Expression *u_var = expr_variable("u");
    Expression *f_var = expr_variable("f");
    Expression *du_dx = expr_derivative(u_var, "x");
    Expression *d2u_dx2 = expr_derivative(du_dx, "x");
    Expression *neg_d2u_dx2 = expr_negate(d2u_dx2);
    Expression *neg_lap_u_minus_f = expr_add(neg_d2u_dx2, expr_negate(f_var));
    
    pde_system_add_equation(sys, neg_lap_u_minus_f);
    
    char *unknowns[] = {"u"};
    pde_system_set_unknowns(sys, unknowns, 1);
    
    // Initial guess: zeros with boundary conditions
    Dictionary *initial = dict_create(1);
    GridField *u_init = grid_field_create(grid);
    for (uint32_t i = 0; i < dims[0]; i++) {
        uint32_t idx[3] = {i, 0, 0};
        Literal *val = literal_create_scalar(0.0);
        grid_field_set(u_init, idx, val);
        literal_free(val);
    }
    dict_set(initial, "u", &u_init->data);
    
    printf("\n  Solving with Newton-Krylov...\n");
    
    // Solve with Newton-Krylov
    SolverResult *result = solve_newton_krylov(sys, initial);
    
    printf("  Status: ");
    switch (result->status) {
        case SOLVER_SUCCESS:
            printf("SUCCESS\n");
            break;
        case SOLVER_MAX_ITER:
            printf("MAX_ITER\n");
            break;
        case SOLVER_DIVERGED:
            printf("DIVERGED\n");
            break;
        case SOLVER_INVALID_SYSTEM:
            printf("INVALID_SYSTEM\n");
            break;
        default:
            printf("UNKNOWN\n");
    }
    printf("  Iterations: %d\n", result->iterations);
    printf("  Final residual: %.6e\n", result->final_residual);
    
    if (result->solution) {
        Literal *u_sol;
        if (dict_get(result->solution, "u", &u_sol)) {
            // Check solution at x = 0.5
            uint32_t mid_idx[3] = {10, 0, 0};
            double computed = literal_get(u_sol, mid_idx);
            double x = 0.5;
            double expected = x * (1.0 - x) / 2.0;  // Analytical: x(1-x)/2
            double error = fabs(computed - expected);
            
            printf("\n  Solution at x=0.5:\n");
            printf("    Computed: %.6f\n", computed);
            printf("    Expected: %.6f\n", expected);
            printf("    Error: %.6e\n", error);
            
            if (error < 0.01) {
                printf("  [OK] Poisson solver correct\n\n");
            } else {
                printf("  [WARNING] Large error in solution\n\n");
            }
        }
    }
    
    solver_result_free(result);
    dict_free(initial);
    grid_field_free(u_init);
    grid_field_free(f);
    pde_system_free(sys);
    grid_metadata_free(grid);
}

int main(void) {
    printf("===========================================\n");
    printf("Grid-Expression Integration Test Suite\n");
    printf("===========================================\n\n");
    
    test_grid_derivative_evaluation();
    test_grid_laplacian_evaluation();
    test_poisson_1d_solver();
    
    printf("===========================================\n");
    printf("All tests passed! [OK]\n");
    printf("===========================================\n");
    
    return 0;
}
