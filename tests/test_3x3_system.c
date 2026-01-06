#include <stdio.h>
#include <math.h>
#include "../include/solver.h"
#include "../include/expression.h"
#include "../include/literal.h"

// Test a larger 3x3 nonlinear system
int main(void) {
    printf("=====================================\n");
    printf("   3x3 NONLINEAR SYSTEM TEST\n");
    printf("=====================================\n\n");
    
    // System:
    // x^2 + y - z = 1
    // x + y^2 - z = 2  
    // x + y + z^2 = 3
    //
    // This is a challenging nonlinear system
    
    PDESystem *sys = pde_system_create();
    
    // Equation 1: x^2 + y - z - 1 = 0
    Expression *eq1 = expr_add(
        expr_add(
            expr_add(
                expr_multiply(expr_variable("x"), expr_variable("x")),
                expr_variable("y")
            ),
            expr_unary(OP_NEGATE, expr_variable("z"))
        ),
        expr_unary(OP_NEGATE, make_scalar(1.0))
    );
    
    // Equation 2: x + y^2 - z - 2 = 0
    Expression *eq2 = expr_add(
        expr_add(
            expr_add(
                expr_variable("x"),
                expr_multiply(expr_variable("y"), expr_variable("y"))
            ),
            expr_unary(OP_NEGATE, expr_variable("z"))
        ),
        expr_unary(OP_NEGATE, make_scalar(2.0))
    );
    
    // Equation 3: x + y + z^2 - 3 = 0
    Expression *eq3 = expr_add(
        expr_add(
            expr_add(
                expr_variable("x"),
                expr_variable("y")
            ),
            expr_multiply(expr_variable("z"), expr_variable("z"))
        ),
        expr_unary(OP_NEGATE, make_scalar(3.0))
    );
    
    pde_system_add_equation(sys, eq1);
    pde_system_add_equation(sys, eq2);
    pde_system_add_equation(sys, eq3);
    
    char *unknowns[] = {"x", "y", "z"};
    pde_system_set_unknowns(sys, unknowns, 3);
    pde_system_set_tolerance(sys, 1e-10);
    pde_system_set_max_iterations(sys, 50);
    
    printf("System:\n");
    printf("  x^2 + y - z = 1\n");
    printf("  x + y^2 - z = 2\n");
    printf("  x + y + z^2 = 3\n\n");
    
    // Try different initial guesses
    printf("=== Test 1: Initial guess (1, 1, 1) ===\n");
    Dictionary *guess1 = dict_create(3);
    Literal *lit_x1 = literal_create_scalar(1.0);
    Literal *lit_y1 = literal_create_scalar(1.0);
    Literal *lit_z1 = literal_create_scalar(1.0);
    dict_set(guess1, "x", lit_x1);
    dict_set(guess1, "y", lit_y1);
    dict_set(guess1, "z", lit_z1);
    literal_free(lit_x1);
    literal_free(lit_y1);
    literal_free(lit_z1);
    
    printf("[DEBUG] About to call solve_newton_raphson for guess1\n");
    SolverResult *res1 = solve_newton_raphson(sys, guess1);
    printf("[DEBUG] Returned from solve_newton_raphson for guess1\n");
    printf("Status: %d (%s)\n", res1->status, 
           res1->status == SOLVER_SUCCESS ? "SUCCESS" : "FAILED");
    printf("Iterations: %d\n", res1->iterations);
    printf("Final residual: %.6e\n", res1->final_residual);
    if (res1->status != SOLVER_SUCCESS) {
        printf("[DEBUG] Newton-Raphson failed for guess1. Status: %d, Iterations: %d, Residual: %.6e\n", res1->status, res1->iterations, res1->final_residual);
    } else {
        Literal *x_val, *y_val, *z_val;
        dict_get(res1->solution, "x", &x_val);
        dict_get(res1->solution, "y", &y_val);
        dict_get(res1->solution, "z", &z_val);
        printf("Solution: x=%.6f, y=%.6f, z=%.6f\n", 
               x_val->field[0], y_val->field[0], z_val->field[0]);
        
        // Verify solution
        double x = x_val->field[0];
        double y = y_val->field[0];
        double z = z_val->field[0];
        double r1 = x*x + y - z - 1.0;
        double r2 = x + y*y - z - 2.0;
        double r3 = x + y + z*z - 3.0;
        printf("Residuals: r1=%.2e, r2=%.2e, r3=%.2e\n", r1, r2, r3);
    }
    
    printf("\n=== Test 2: Initial guess (0.6, 0.6, 0.6) ===\n");
    Dictionary *guess2 = dict_create(3);
    Literal *lit_x2 = literal_create_scalar(0.6);
    Literal *lit_y2 = literal_create_scalar(0.6);
    Literal *lit_z2 = literal_create_scalar(0.6);
    dict_set(guess2, "x", lit_x2);
    dict_set(guess2, "y", lit_y2);
    dict_set(guess2, "z", lit_z2);
    literal_free(lit_x2);
    literal_free(lit_y2);
    literal_free(lit_z2);
    
    printf("[DEBUG] About to call solve_newton_raphson for guess2\n");
    SolverResult *res2 = solve_newton_raphson(sys, guess2);
    printf("[DEBUG] Returned from solve_newton_raphson for guess2\n");
    printf("Status: %d (%s)\n", res2->status,
           res2->status == SOLVER_SUCCESS ? "SUCCESS" : "FAILED");
    printf("Iterations: %d\n", res2->iterations);
    printf("Final residual: %.6e\n", res2->final_residual);
    if (res2->status != SOLVER_SUCCESS) {
        printf("[DEBUG] Newton-Raphson failed for guess2. Status: %d, Iterations: %d, Residual: %.6e\n", res2->status, res2->iterations, res2->final_residual);
    } else {
        Literal *x_val, *y_val, *z_val;
        dict_get(res2->solution, "x", &x_val);
        dict_get(res2->solution, "y", &y_val);
        dict_get(res2->solution, "z", &z_val);
        printf("Solution: x=%.6f, y=%.6f, z=%.6f\n",
               x_val->field[0], y_val->field[0], z_val->field[0]);
        
        // Verify solution
        double x = x_val->field[0];
        double y = y_val->field[0];
        double z = z_val->field[0];
        double r1 = x*x + y - z - 1.0;
        double r2 = x + y*y - z - 2.0;
        double r3 = x + y + z*z - 3.0;
        printf("Residuals: r1=%.2e, r2=%.2e, r3=%.2e\n", r1, r2, r3);
    }
    
    printf("\n=== Test 3: Initial guess (2, 2, 0) ===\n");
    Dictionary *guess3 = dict_create(3);
    Literal *lit_x3 = literal_create_scalar(2.0);
    Literal *lit_y3 = literal_create_scalar(2.0);
    Literal *lit_z3 = literal_create_scalar(0.0);
    dict_set(guess3, "x", lit_x3);
    dict_set(guess3, "y", lit_y3);
    dict_set(guess3, "z", lit_z3);
    literal_free(lit_x3);
    literal_free(lit_y3);
    literal_free(lit_z3);
    
    printf("[DEBUG] About to call solve_newton_raphson for guess3\n");
    SolverResult *res3 = solve_newton_raphson(sys, guess3);
    printf("[DEBUG] Returned from solve_newton_raphson for guess3\n");
    printf("Status: %d (%s)\n", res3->status,
           res3->status == SOLVER_SUCCESS ? "SUCCESS" : "FAILED");
    printf("Iterations: %d\n", res3->iterations);
    printf("Final residual: %.6e\n", res3->final_residual);
    if (res3->status != SOLVER_SUCCESS) {
        printf("[DEBUG] Newton-Raphson failed for guess3. Status: %d, Iterations: %d, Residual: %.6e\n", res3->status, res3->iterations, res3->final_residual);
    } else {
        Literal *x_val, *y_val, *z_val;
        dict_get(res3->solution, "x", &x_val);
        dict_get(res3->solution, "y", &y_val);
        dict_get(res3->solution, "z", &z_val);
        printf("Solution: x=%.6f, y=%.6f, z=%.6f\n",
               x_val->field[0], y_val->field[0], z_val->field[0]);
        
        // Verify solution
        double x = x_val->field[0];
        double y = y_val->field[0];
        double z = z_val->field[0];
        double r1 = x*x + y - z - 1.0;
        double r2 = x + y*y - z - 2.0;
        double r3 = x + y + z*z - 3.0;
        printf("Residuals: r1=%.2e, r2=%.2e, r3=%.2e\n", r1, r2, r3);
    }
    
    // Cleanup
    dict_free(guess1);
    dict_free(guess2);
    dict_free(guess3);
    solver_result_free(res1);
    solver_result_free(res2);
    solver_result_free(res3);
    pde_system_free(sys);
    
    printf("\n=====================================\n");
    printf("   TEST COMPLETE\n");
    printf("=====================================\n");
    
    return 0;
}
