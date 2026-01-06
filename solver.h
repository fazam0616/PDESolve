#ifndef SOLVER_H
#define SOLVER_H

#include "expression.h"
#include "calculus.h"
#include "dictionary.h"
#include "graph.h"
#include <stdbool.h>

// ============================================================================
// PDE System Definition
// ============================================================================

typedef struct {
    Expression **equations;      // Array of equations (each = 0 at solution)
    int n_equations;
    
    char **unknowns;            // Variables to solve for
    int n_unknowns;
    
    Dictionary *parameters;     // Known constants (alpha, c, etc.)
    
    double tolerance;           // Convergence threshold
    int max_iterations;         // Maximum solver iterations
} PDESystem;

// ============================================================================
// Solver Status
// ============================================================================

typedef enum {
    SOLVER_SUCCESS,             // Solution found within tolerance
    SOLVER_MAX_ITER,            // Hit iteration limit
    SOLVER_DIVERGED,            // Solution diverging
    SOLVER_INVALID_SYSTEM,      // Malformed system
    SOLVER_NO_SOLUTION          // No solution exists
} SolverStatus;

// ============================================================================
// System Creation and Management
// ============================================================================

// Create a new PDE system
// Returns: Newly allocated PDESystem (caller owns, must call pde_system_free)
// Memory: Allocates PDESystem and empty Dictionary for parameters
PDESystem* pde_system_create(void);

// Add an equation to the system (equation should equal 0 at solution)
// sys: System to add to (must not be NULL)
// equation: Expression to add (ownership transferred to system)
// Memory: System takes ownership, will free equation in pde_system_free()
void pde_system_add_equation(PDESystem *sys, Expression *equation);

// Set unknowns (variables to solve for)
// sys: System to configure (must not be NULL)
// unknowns: Array of variable name strings (copied internally)
// n: Number of unknowns
// Memory: Copies strings, caller retains ownership of unknowns array
void pde_system_set_unknowns(PDESystem *sys, char **unknowns, int n);

// Set a parameter (known constant)
// sys: System to configure (must not be NULL)
// name: Parameter name (copied internally)
// value: Scalar value for parameter
// Memory: Creates internal Literal copy, caller retains name ownership
void pde_system_set_parameter(PDESystem *sys, const char *name, double value);

// Set solver tolerance for convergence (default: 1e-6)
void pde_system_set_tolerance(PDESystem *sys, double tol);

// Set maximum iterations (default: 1000)
void pde_system_set_max_iterations(PDESystem *sys, int max_iter);

// Free system and all owned resources
// sys: System to free (can be NULL)
// Memory: Frees equations, unknowns, parameters dictionary
void pde_system_free(PDESystem *sys);

// ============================================================================
// Residual Computation
// ============================================================================

// Compute residual vector for current guess
// sys: System with equations to evaluate (must not be NULL)
// current_guess: Dictionary with variable values (must not be NULL)
// Returns: Newly allocated array of residuals (caller must free with free())
//          Returns NULL on allocation failure
// Memory: Caller owns returned array, must call free()
double* compute_residuals(PDESystem *sys, Dictionary *current_guess);

// Compute L2 norm of residual vector: sqrt(sum(r_i^2))
// residuals: Array of residual values (must not be NULL)
// n: Number of residuals
// Returns: L2 norm, or INFINITY if residuals is NULL
double residual_norm(double *residuals, int n);

// ============================================================================
// Solution Result
// ============================================================================

typedef struct {
    SolverStatus status;        // Convergence status
    Dictionary *solution;       // Final solution values (owned by result)
    int iterations;             // Iterations taken
    double final_residual;      // Final residual norm
    char *message;              // Status message (owned by result, can be NULL)
} SolverResult;

// Free solver result and all owned resources
// result: Result to free (can be NULL)
// Memory: Frees solution dictionary, message string, and result struct
void solver_result_free(SolverResult *result);

// ============================================================================
// Solver Algorithms
// ============================================================================

// Fixed-point iteration with damping (simple iterative refinement)
// sys: System to solve (must be well-posed)
// initial_guess: Starting values (can be NULL for zeros)
// Returns: SolverResult with solution (caller owns, must call solver_result_free)
// Algorithm: x_new = x_old - damping * residual
// Note: May diverge for stiff systems; use Newton-Raphson for better convergence
// Memory: Returns newly allocated SolverResult, caller must free
SolverResult* solve_fixed_point(PDESystem *sys, Dictionary *initial_guess);

// Newton-Raphson iteration (requires Jacobian computation)
// sys: System to solve (must be well-posed)
// initial_guess: Starting values (can be NULL for zeros)
// Returns: SolverResult with solution (caller owns, must call solver_result_free)
// Algorithm: x_new = x_old - J^(-1) * F(x_old)
// Status: NOT YET IMPLEMENTED - returns SOLVER_INVALID_SYSTEM
// Memory: Returns newly allocated SolverResult, caller must free
SolverResult* solve_newton_raphson(PDESystem *sys, Dictionary *initial_guess);

// Successive over-relaxation (SOR) - for large sparse systems
// sys: System to solve (must be well-posed)
// initial_guess: Starting values (can be NULL for zeros)
// omega: Relaxation parameter (typically 1.0-2.0)
// Returns: SolverResult with solution (caller owns, must call solver_result_free)
// Status: NOT YET IMPLEMENTED - returns SOLVER_INVALID_SYSTEM
// Memory: Returns newly allocated SolverResult, caller must free
SolverResult* solve_sor(PDESystem *sys, Dictionary *initial_guess, double omega);

// ============================================================================
// System Analysis
// ============================================================================

// Check if system is well-posed (n_equations == n_unknowns and n > 0)
// sys: System to check (can be NULL, returns false)
// Returns: true if system is square and non-empty
bool pde_system_is_well_posed(PDESystem *sys);

// Analyze dependency structure between unknowns
// sys: System to analyze (must not be NULL)
// Returns: DependencyGraph (caller owns, must call graph_free)
// Memory: Returns newly allocated graph, caller must free
DependencyGraph* pde_system_analyze_dependencies(PDESystem *sys);

// Get suggested solve order via topological sort
// sys: System to analyze (must not be NULL)
// n_vars: Output parameter for number of variables (must not be NULL)
// Returns: Array of variable name strings (caller owns, must free array and strings)
//          Returns NULL if graph has cycles or analysis fails
// Memory: Caller must free returned array and each string with free()
char** pde_system_solve_order(PDESystem *sys, int *n_vars);

// ============================================================================
// Debugging and Visualization
// ============================================================================

// Print system information to stdout
// sys: System to print (can be NULL)
void pde_system_print(PDESystem *sys);

// Print solver result to stdout
// result: Result to print (can be NULL)
void solver_result_print(SolverResult *result);

#endif // SOLVER_H
