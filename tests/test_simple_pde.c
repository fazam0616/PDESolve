#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/solver.h"
#include "../include/expression.h"
#include "../include/literal.h"
#include "../include/calculus.h"

int main(void) {
    printf("Testing PDE coefficient solving\n\n");
    
    // Simple test: 2*a - 2 = 0, solve for a
    // Expected: a = 1
    
    PDESystem *sys = pde_system_create();
    
    printf("Creating system...\n");
    
    // Equation: 2*a - 2 = 0
    Expression *term1 = expr_multiply(make_scalar(2.0), expr_variable("a"));
    printf("Created term1: 2*a\n");
    
    Expression *term2 = expr_unary(OP_NEGATE, make_scalar(2.0));
    printf("Created term2: -2\n");
    
    Expression *eq = expr_add(term1, term2);
    printf("Created equation: 2*a + (-2)\n");
    
    // Try to evaluate the equation at a=1 (should give 0)
    Dictionary *test_point = dict_create(1);
    Literal *lit_test = literal_create_scalar(1.0);
    dict_set(test_point, "a", lit_test);
    literal_free(lit_test);
    
    Literal *eq_value = expression_evaluate(eq, test_point);
    if (eq_value) {
        printf("Equation value at a=1: %.6f (should be 0)\n", eq_value->field[0]);
        literal_free(eq_value);
    } else {
        printf("ERROR: Equation evaluation returned NULL!\n");
    }
    dict_free(test_point);
    
    printf("Adding equation to system...\n");
    pde_system_add_equation(sys, eq);
    
    printf("Setting unknowns...\n");
    char *unknowns[] = {"a"};
    pde_system_set_unknowns(sys, unknowns, 1);
    
    printf("Setting tolerance and max iterations...\n");
    pde_system_set_tolerance(sys, 1e-10);
    pde_system_set_max_iterations(sys, 50);
    
    printf("System created successfully\n");
    printf("Equation: 2*a - 2 = 0\n");
    
    // Print system details
    printf("\nSystem details:\n");
    printf("  n_equations: %d\n", sys->n_equations);
    printf("  n_unknowns: %d\n", sys->n_unknowns);
    printf("  Well-posed: %s\n", pde_system_is_well_posed(sys) ? "YES" : "NO");
    
    // Manually compute Jacobian to debug
    printf("\nManually computing Jacobian for debugging...\n");
    Expression *jac = derivative(sys->equations[0], "a");
    printf("Jacobian expression: d(eq)/da computed\n");
    
    // Evaluate at a=0
    Dictionary *eval_point = dict_create(1);
    Literal *lit_a = literal_create_scalar(0.0);
    dict_set(eval_point, "a", lit_a);
    literal_free(lit_a);
    
    Literal *jac_value = expression_evaluate(jac, eval_point);
    if (jac_value) {
        printf("Jacobian value at a=0: %.6f (should be 2.0)\n", jac_value->field[0]);
        literal_free(jac_value);
    } else {
        printf("Jacobian evaluation returned NULL!\n");
    }
    
    dict_free(eval_point);
    expression_free(jac);
    
    printf("\nCalling solver...\n");
    fflush(stdout);
    
    SolverResult *result = solve_newton_raphson(sys, NULL);
    
    printf("Solver returned with status: %d\n", result->status);
    
    if (result->status == SOLVER_SUCCESS) {
        printf("SUCCESS!\n");
        Literal *a_val;
        if (dict_get(result->solution, "a", &a_val)) {
            printf("Solution: a = %.6f\n", a_val->field[0]);
            
            if (fabs(a_val->field[0] - 1.0) < 1e-6) {
                printf("✓ Correct: a = 1\n");
            } else {
                printf("✗ Wrong value\n");
            }
        } else {
            printf("✗ Failed to get solution\n");
        }
    } else {
        printf("✗ Solver failed with status %d\n", result->status);
        printf("Message: %s\n", result->message);
        printf("Iterations: %d\n", result->iterations);
        printf("Final residual: %.6e\n", result->final_residual);
    }
    
    solver_result_free(result);
    pde_system_free(sys);
    
    printf("\nTest complete\n");
    return 0;
}
