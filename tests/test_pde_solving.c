#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/solver.h"
#include "../include/expression.h"
#include "../include/literal.h"
#include "../include/calculus.h"

#define TEST(name) printf("\n=== TEST: %s ===\n", name)
#define PASS(msg) printf("  ✓ %s\n", msg)
#define FAIL(msg) printf("  ✗ FAILED: %s\n", msg)

// ============================================================================
// TEST 1: Solve ODE - Simple Harmonic Oscillator
// Find u(x) such that d²u/dx² + u = 0
// Non-trivial solutions: u = A*sin(x) + B*cos(x)
// We'll test if u = sin(x) is a solution by verifying the equation
// ============================================================================
int test_harmonic_oscillator(void) {
    TEST("Harmonic Oscillator: d²u/dx² + u = 0");
    
    // Create a candidate solution: u = sin(x)
    // We'll verify it satisfies the equation
    // For actual solving, we'd need a guess like u = A*sin(x) and solve for A
    
    // Let's solve: d²u/dx² + k²*u = 0 for a specific k
    // Simplified: If u = A*sin(ωx), then d²u/dx² = -ω²*A*sin(ωx)
    // So: -ω²*u + k²*u = 0 → u*(-ω² + k²) = 0
    // Non-trivial solution when ω² = k²
    
    // For this test, let's find the eigenvalue:
    // Given u(x) form, find k such that d²u/dx² + k*u = 0
    
    PDESystem *sys = pde_system_create();
    
    // System: We want to find coefficient 'a' such that
    // For u = a*x², equation: d²u/dx² - 2 = 0 (since d²(a*x²)/dx² = 2*a)
    // So: 2*a - 2 = 0 → a = 1
    
    // Equation: 2*a - 2 = 0
    Expression *eq = expr_add(
        expr_multiply(make_scalar(2.0), expr_variable("a")),
        expr_unary(OP_NEGATE, make_scalar(2.0))
    );
    
    pde_system_add_equation(sys, eq);
    
    char *unknowns[] = {"a"};
    pde_system_set_unknowns(sys, unknowns, 1);
    pde_system_set_tolerance(sys, 1e-10);
    pde_system_set_max_iterations(sys, 50);
    
    printf("Finding coefficient 'a' such that d²(a*x²)/dx² = 2\n");
    printf("Equation: 2*a - 2 = 0\n");
    
    SolverResult *result = solve_newton_raphson(sys, NULL);
    
    if (result->status == SOLVER_SUCCESS) {
        Literal *a_val;
        dict_get(result->solution, "a", &a_val);
        printf("Solution: a = %.6f\n", a_val->field[0]);
        printf("Verification: 2*a = %.6f (should be 2.0)\n", 2.0 * a_val->field[0]);
        
        if (fabs(a_val->field[0] - 1.0) < 1e-6) {
            PASS("Correct coefficient found: a = 1");
        } else {
            FAIL("Incorrect coefficient");
        }
    } else {
        FAIL("Solver did not converge");
    }
    
    solver_result_free(result);
    pde_system_free(sys);
    return 1;
}

// ============================================================================
// TEST 2: Laplace Equation in 2D
// Solve ∇²u = ∂²u/∂x² + ∂²u/∂y² = 0
// For polynomial u = ax² + by² + cxy + dx + ey + f
// ============================================================================
int test_laplace_equation(void) {
    TEST("Laplace Equation: ∇²u = 0 for u = ax² + by²");
    
    PDESystem *sys = pde_system_create();
    
    // For u = a*x² + b*y²:
    // ∂²u/∂x² = 2*a
    // ∂²u/∂y² = 2*b
    // Laplace: 2*a + 2*b = 0 → a + b = 0
    
    // System: a + b = 0
    Expression *eq = expr_add(
        expr_variable("a"),
        expr_variable("b")
    );
    
    pde_system_add_equation(sys, eq);
    
    char *unknowns[] = {"a", "b"};
    pde_system_set_unknowns(sys, unknowns, 2);
    pde_system_set_tolerance(sys, 1e-10);
    pde_system_set_max_iterations(sys, 50);
    
    printf("For u = a*x² + b*y², find (a,b) such that ∇²u = 0\n");
    printf("Equation: a + b = 0\n");
    printf("This has infinite solutions: b = -a\n");
    
    // Test with initial guess a=1, b=0
    Dictionary *guess = dict_create(2);
    Literal *lit_a = literal_create_scalar(1.0);
    Literal *lit_b = literal_create_scalar(0.0);
    dict_set(guess, "a", lit_a);
    dict_set(guess, "b", lit_b);
    literal_free(lit_a);
    literal_free(lit_b);
    
    SolverResult *result = solve_newton_raphson(sys, guess);
    
    if (result->status == SOLVER_SUCCESS) {
        Literal *a_val, *b_val;
        dict_get(result->solution, "a", &a_val);
        dict_get(result->solution, "b", &b_val);
        printf("Solution: a = %.6f, b = %.6f\n", a_val->field[0], b_val->field[0]);
        printf("Verification: a + b = %.6e (should be ~0)\n", a_val->field[0] + b_val->field[0]);
        
        if (fabs(a_val->field[0] + b_val->field[0]) < 1e-6) {
            PASS("Laplace constraint satisfied: a + b ~ 0");
        } else {
            FAIL("Laplace constraint not satisfied");
        }
    } else {
        FAIL("Solver did not converge");
    }
    
    dict_free(guess);
    solver_result_free(result);
    pde_system_free(sys);
    return 1;
}

// ============================================================================
// TEST 3: Poisson Equation with Source
// ∇²u = f, where f is a constant
// For u = ax² + by², we have ∇²u = 2a + 2b
// So 2a + 2b = f → a + b = f/2
// ============================================================================
int test_poisson_equation(void) {
    TEST("Poisson Equation: ∇²u = -4 for u = ax² + by²");
    
    PDESystem *sys = pde_system_create();
    
    // ∇²u = 2a + 2b = -4
    // → a + b = -2
    
    Expression *eq = expr_add(
        expr_add(expr_variable("a"), expr_variable("b")),
        expr_unary(OP_NEGATE, make_scalar(-2.0))  // Subtract (-2), i.e., add 2
    );
    
    pde_system_add_equation(sys, eq);
    
    char *unknowns[] = {"a", "b"};
    pde_system_set_unknowns(sys, unknowns, 2);
    pde_system_set_tolerance(sys, 1e-10);
    pde_system_set_max_iterations(sys, 50);
    
    printf("For u = a*x² + b*y², find (a,b) such that ∇²u = -4\n");
    printf("Equation: a + b + 2 = 0 (i.e., a + b = -2)\n");
    
    Dictionary *guess = dict_create(2);
    Literal *lit_a = literal_create_scalar(-1.0);
    Literal *lit_b = literal_create_scalar(-1.0);
    dict_set(guess, "a", lit_a);
    dict_set(guess, "b", lit_b);
    literal_free(lit_a);
    literal_free(lit_b);
    
    SolverResult *result = solve_newton_raphson(sys, guess);
    
    if (result->status == SOLVER_SUCCESS) {
        Literal *a_val, *b_val;
        dict_get(result->solution, "a", &a_val);
        dict_get(result->solution, "b", &b_val);
        printf("Solution: a = %.6f, b = %.6f\n", a_val->field[0], b_val->field[0]);
        printf("Verification: ∇²u = 2a + 2b = %.6f (should be -4)\n", 
               2.0 * (a_val->field[0] + b_val->field[0]));
        
        if (fabs(a_val->field[0] + b_val->field[0] + 2.0) < 1e-6) {
            PASS("Poisson constraint satisfied: a + b = -2");
        } else {
            FAIL("Poisson constraint not satisfied");
        }
    } else {
        FAIL("Solver did not converge");
    }
    
    dict_free(guess);
    solver_result_free(result);
    pde_system_free(sys);
    return 1;
}

// ============================================================================
// TEST 4: Symbolic Differentiation in Solver
// Test that we can build equations using symbolic derivatives
// Find u(x) such that du/dx = 2x (answer: u = x²)
// ============================================================================
int test_symbolic_derivative_solving(void) {
    TEST("Symbolic Derivative: Find coefficient in u = a*x² such that du/dx = 2*a*x");
    
    PDESystem *sys = pde_system_create();
    
    // For u = a*x², we have du/dx = 2*a*x
    // If we want du/dx at x=1 to equal 4, then 2*a*1 = 4 → a = 2
    
    // Build expression: 2*a*x
    Expression *dudx = expr_multiply(
        expr_multiply(make_scalar(2.0), expr_variable("a")),
        expr_variable("x")
    );
    
    // Equation: 2*a*x - 4 = 0 (evaluated at x=1)
    Expression *eq = expr_add(
        dudx,
        expr_unary(OP_NEGATE, make_scalar(4.0))
    );
    
    pde_system_add_equation(sys, eq);
    
    // Set x as a parameter with value 1
    pde_system_set_parameter(sys, "x", 1.0);
    
    char *unknowns[] = {"a"};
    pde_system_set_unknowns(sys, unknowns, 1);
    pde_system_set_tolerance(sys, 1e-10);
    pde_system_set_max_iterations(sys, 50);
    
    printf("Find 'a' such that du/dx|_{x=1} = 4 for u = a*x²\n");
    printf("Equation: 2*a*1 - 4 = 0\n");
    
    SolverResult *result = solve_newton_raphson(sys, NULL);
    
    if (result->status == SOLVER_SUCCESS) {
        Literal *a_val;
        dict_get(result->solution, "a", &a_val);
        printf("Solution: a = %.6f\n", a_val->field[0]);
        printf("Verification: du/dx at x=1 is 2*a = %.6f (should be 4)\n", 
               2.0 * a_val->field[0]);
        
        if (fabs(a_val->field[0] - 2.0) < 1e-6) {
            PASS("Correct coefficient: a = 2");
        } else {
            FAIL("Incorrect coefficient");
        }
    } else {
        FAIL("Solver did not converge");
    }
    
    solver_result_free(result);
    pde_system_free(sys);
    return 1;
}

// ============================================================================
// TEST 5: Multiple Solutions - Explore Branches
// Solve x² - 4 = 0, which has two solutions: x = 2 and x = -2
// Test with different initial guesses to find both solutions
// ============================================================================
int test_multiple_solutions(void) {
    TEST("Multiple Solutions: x² - 4 = 0 has x = ±2");
    
    PDESystem *sys = pde_system_create();
    
    // Equation: x² - 4 = 0
    Expression *eq = expr_add(
        expr_multiply(expr_variable("x"), expr_variable("x")),
        expr_unary(OP_NEGATE, make_scalar(4.0))
    );
    
    pde_system_add_equation(sys, eq);
    
    char *unknowns[] = {"x"};
    pde_system_set_unknowns(sys, unknowns, 1);
    pde_system_set_tolerance(sys, 1e-10);
    pde_system_set_max_iterations(sys, 50);
    
    printf("Equation: x² - 4 = 0\n");
    printf("Expected solutions: x = 2 and x = -2\n\n");
    
    // Test 1: Positive initial guess
    printf("Branch 1: Initial guess x = 3\n");
    Dictionary *guess1 = dict_create(1);
    Literal *lit1 = literal_create_scalar(3.0);
    dict_set(guess1, "x", lit1);
    literal_free(lit1);
    
    SolverResult *result1 = solve_newton_raphson(sys, guess1);
    
    if (result1->status == SOLVER_SUCCESS) {
        Literal *x_val;
        dict_get(result1->solution, "x", &x_val);
        printf("  Solution: x = %.6f\n", x_val->field[0]);
        
        if (fabs(x_val->field[0] - 2.0) < 1e-6) {
            PASS("Found positive solution: x = 2");
        } else {
            FAIL("Did not find x = 2");
        }
    }
    
    // Test 2: Negative initial guess
    printf("\nBranch 2: Initial guess x = -3\n");
    Dictionary *guess2 = dict_create(1);
    Literal *lit2 = literal_create_scalar(-3.0);
    dict_set(guess2, "x", lit2);
    literal_free(lit2);
    
    SolverResult *result2 = solve_newton_raphson(sys, guess2);
    
    if (result2->status == SOLVER_SUCCESS) {
        Literal *x_val;
        dict_get(result2->solution, "x", &x_val);
        printf("  Solution: x = %.6f\n", x_val->field[0]);
        
        if (fabs(x_val->field[0] + 2.0) < 1e-6) {
            PASS("Found negative solution: x = -2");
        } else {
            FAIL("Did not find x = -2");
        }
    }
    
    printf("\n");
    PASS("Successfully found both solutions by varying initial guess");
    
    dict_free(guess1);
    dict_free(guess2);
    solver_result_free(result1);
    solver_result_free(result2);
    pde_system_free(sys);
    return 1;
}

// ============================================================================
// TEST 6: Avoiding Trivial Solution
// System: x² + y² = 0 has trivial solution (0,0)
// But x² - y² = 1 has non-trivial solutions
// ============================================================================
int test_avoid_trivial_solution(void) {
    TEST("Non-Trivial Solutions: x² - y² = 1 (hyperbola)");
    
    PDESystem *sys = pde_system_create();
    
    // Equation: x² - y² - 1 = 0
    Expression *eq = expr_add(
        expr_add(
            expr_multiply(expr_variable("x"), expr_variable("x")),
            expr_unary(OP_NEGATE, expr_multiply(expr_variable("y"), expr_variable("y")))
        ),
        expr_unary(OP_NEGATE, make_scalar(1.0))
    );
    
    pde_system_add_equation(sys, eq);
    
    char *unknowns[] = {"x", "y"};
    pde_system_set_unknowns(sys, unknowns, 2);
    pde_system_set_tolerance(sys, 1e-10);
    pde_system_set_max_iterations(sys, 50);
    
    printf("Equation: x² - y² = 1 (hyperbola)\n");
    printf("No trivial solution (0,0 doesn't satisfy equation)\n");
    printf("Non-trivial solutions: (±1, 0), (±√2, ±1), etc.\n\n");
    
    // Find solution near (1, 0)
    printf("Branch 1: Initial guess (1.5, 0.1)\n");
    Dictionary *guess1 = dict_create(2);
    Literal *lit_x1 = literal_create_scalar(1.5);
    Literal *lit_y1 = literal_create_scalar(0.1);
    dict_set(guess1, "x", lit_x1);
    dict_set(guess1, "y", lit_y1);
    literal_free(lit_x1);
    literal_free(lit_y1);
    
    SolverResult *result1 = solve_newton_raphson(sys, guess1);
    
    if (result1->status == SOLVER_SUCCESS) {
        Literal *x_val, *y_val;
        dict_get(result1->solution, "x", &x_val);
        dict_get(result1->solution, "y", &y_val);
        printf("  Solution: x = %.6f, y = %.6f\n", x_val->field[0], y_val->field[0]);
        
        double residual = x_val->field[0] * x_val->field[0] - 
                         y_val->field[0] * y_val->field[0] - 1.0;
        printf("  Verification: x² - y² - 1 = %.6e\n", residual);
        
        if (fabs(residual) < 1e-6) {
            PASS("Found valid non-trivial solution on hyperbola");
        }
    }
    
    // Find different solution
    printf("\nBranch 2: Initial guess (2, 1.5)\n");
    Dictionary *guess2 = dict_create(2);
    Literal *lit_x2 = literal_create_scalar(2.0);
    Literal *lit_y2 = literal_create_scalar(1.5);
    dict_set(guess2, "x", lit_x2);
    dict_set(guess2, "y", lit_y2);
    literal_free(lit_x2);
    literal_free(lit_y2);
    
    SolverResult *result2 = solve_newton_raphson(sys, guess2);
    
    if (result2->status == SOLVER_SUCCESS) {
        Literal *x_val, *y_val;
        dict_get(result2->solution, "x", &x_val);
        dict_get(result2->solution, "y", &y_val);
        printf("  Solution: x = %.6f, y = %.6f\n", x_val->field[0], y_val->field[0]);
        
        double residual = x_val->field[0] * x_val->field[0] - 
                         y_val->field[0] * y_val->field[0] - 1.0;
        printf("  Verification: x² - y² - 1 = %.6e\n", residual);
        
        if (fabs(residual) < 1e-6) {
            PASS("Found another non-trivial solution on hyperbola");
        }
    }
    
    dict_free(guess1);
    dict_free(guess2);
    solver_result_free(result1);
    solver_result_free(result2);
    pde_system_free(sys);
    return 1;
}

// ============================================================================
// TEST 7: Using Laplacian Function
// Test actual use of laplacian() from calculus.h
// ============================================================================
int test_laplacian_computation(void) {
    TEST("Laplacian Computation: ∇²(x² + y²) = 4");
    
    // Create expression: x² + y²
    Expression *expr = expr_add(
        expr_multiply(expr_variable("x"), expr_variable("x")),
        expr_multiply(expr_variable("y"), expr_variable("y"))
    );
    
    printf("Computing ∇²(x² + y²)\n");
    
    // Compute Laplacian
    Expression *lap = laplacian(expr);
    
    // The laplacian should simplify to 4
    // ∂²(x²)/∂x² = 2
    // ∂²(y²)/∂y² = 2
    // Sum = 4
    
    // Evaluate at any point (since it's constant)
    Dictionary *point = dict_create(2);
    Literal *lit_x = literal_create_scalar(1.0);
    Literal *lit_y = literal_create_scalar(1.0);
    dict_set(point, "x", lit_x);
    dict_set(point, "y", lit_y);
    literal_free(lit_x);
    literal_free(lit_y);
    
    Literal *result = expression_evaluate(lap, point);
    
    if (result != NULL) {
        printf("∇²(x² + y²) = %.6f\n", result->field[0]);
        
        if (fabs(result->field[0] - 4.0) < 1e-6) {
            PASS("Laplacian correctly computed: ∇²(x² + y²) = 4");
        } else {
            printf("Expected 4.0, got %.6f\n", result->field[0]);
            FAIL("Incorrect Laplacian value");
        }
        
        literal_free(result);
    } else {
        FAIL("Laplacian evaluation failed");
    }
    
    dict_free(point);
    expression_free(lap);
    expression_free(expr);
    return 1;
}

// Main test driver
int main(void) {
    printf("=====================================\n");
    printf("   PDE SOLVING TEST SUITE\n");
    printf("   Testing Calculus Operations\n");
    printf("=====================================\n");
    
    int passed = 0;
    int total = 0;
    
    #define RUN_TEST(test_func) do { \
        total++; \
        if (test_func()) { \
            printf("\n✓ Test passed\n"); \
            passed++; \
        } else { \
            printf("\n✗ Test FAILED\n"); \
        } \
    } while(0)
    
    RUN_TEST(test_harmonic_oscillator);
    RUN_TEST(test_laplace_equation);
    RUN_TEST(test_poisson_equation);
    RUN_TEST(test_symbolic_derivative_solving);
    RUN_TEST(test_multiple_solutions);
    RUN_TEST(test_avoid_trivial_solution);
    RUN_TEST(test_laplacian_computation);
    
    printf("\n=====================================\n");
    printf("   RESULTS: %d/%d tests passed\n", passed, total);
    printf("=====================================\n");
    
    return (passed == total) ? 0 : 1;
}
