#ifndef CALCULUS_H
#define CALCULUS_H

#include "expression.h"

// ============================================================================
// Symbolic Differentiation
// ============================================================================

// Compute derivative of expression with respect to a variable
// expr: Expression to differentiate (must not be NULL)
// var: Variable name to differentiate with respect to (must not be NULL)
// Returns: New Expression representing d(expr)/d(var) (caller owns, must free)
// Memory: Returns newly allocated Expression, caller must call expression_free()
// Rules: Product rule, chain rule, sum rule automatically applied
Expression* derivative(Expression *expr, const char *var);

// Compute partial derivative (same as derivative, explicit name for clarity)
// expr: Expression to differentiate (must not be NULL)
// var: Variable name to differentiate with respect to (must not be NULL)
// Returns: New Expression representing ∂(expr)/∂(var) (caller owns, must free)
// Memory: Returns newly allocated Expression, caller must call expression_free()
Expression* partial_derivative(Expression *expr, const char *var);

// Compute gradient: vector of all partial derivatives
// expr: Expression to take gradient of (must not be NULL)
// out_n_partials: Output for number of partial derivatives (must not be NULL)
// Returns: Array of Expressions [∂f/∂x1, ∂f/∂x2, ..., ∂f/∂xN] (caller owns)
// Memory: Caller must free array with free() and each Expression with expression_free()
// Note: Extracts all unique variables from expr automatically

// Deep copy an expression
// expr: Expression to copy (can be NULL)
// Returns: New Expression that is a deep copy (caller owns, must call expression_free())
// Memory: Returns newly allocated Expression, caller must free
Expression* expr_copy(Expression *expr);
Expression** gradient(Expression *expr, int *out_n_partials);

// Compute Laplacian: sum of second partial derivatives ∇²f = ∂²f/∂x² + ∂²f/∂y² + ...
// expr: Expression to compute Laplacian of (must not be NULL)
// Returns: New Expression representing ∇²(expr) (caller owns, must free)
// Memory: Returns newly allocated Expression, caller must call expression_free()
// Note: Automatically finds all variables and sums their second partials
Expression* laplacian(Expression *expr);

// Higher-order derivatives
// expr: Expression to differentiate (must not be NULL)
// var: Variable name (must not be NULL)
// n: Order of derivative (must be >= 0)
// Returns: New Expression for d^n(expr)/d(var)^n (caller owns, must free)
// Memory: Caller must call expression_free() on result
Expression* nth_derivative(Expression *expr, const char *var, int n);

// Mixed partial: ∂²f/(∂x ∂y) - order matters for non-commutative operators
// expr: Expression to differentiate (must not be NULL)
// var1: First variable (must not be NULL)
// var2: Second variable (must not be NULL)
// Returns: New Expression for ∂²(expr)/(∂var1 ∂var2) (caller owns, must free)
// Memory: Caller must call expression_free() on result
Expression* mixed_partial(Expression *expr, const char *var1, const char *var2);

// ============================================================================
// Finite Difference Approximations (NOT YET IMPLEMENTED)
// ============================================================================

// Finite difference operators for numerical derivatives
typedef enum {
    FD_FORWARD,   // f'(x) ~ (f(x+h) - f(x))/h
    FD_BACKWARD,  // f'(x) ~ (f(x) - f(x-h))/h
    FD_CENTRAL    // f'(x) ~ (f(x+h) - f(x-h))/(2h)
} FiniteDifferenceType;

// Create finite difference approximation expression
// expr: Expression to approximate derivative of (must not be NULL)
// var: Variable to differentiate with respect to (must not be NULL)
// h: Step size (must be > 0)
// type: Finite difference scheme (FORWARD, BACKWARD, or CENTRAL)
// Returns: Expression tree representing the FD formula (caller owns, must free)
// Memory: Caller must call expression_free() on result
// Status: NOT YET IMPLEMENTED
Expression* finite_difference(Expression *expr, const char *var, 
                              double h, FiniteDifferenceType type);

// Second derivative using finite difference: f''(x) ~ (f(x+h) - 2f(x) + f(x-h))/h²
// expr: Expression to approximate second derivative of (must not be NULL)
// var: Variable to differentiate with respect to (must not be NULL)
// step_size: Step size h (must be > 0)
// Returns: Expression tree for 2nd order FD (caller owns, must free)
// Memory: Caller must call expression_free() on result
// Status: NOT YET IMPLEMENTED
Expression* finite_difference_2nd(Expression *expr, const char *var, double step_size);

// ============================================================================
// Expression Simplification (for cleaner derivative results)
// ============================================================================

// Simplify expression tree by applying algebraic rules recursively
// expr: Expression to simplify (must not be NULL)
// Returns: New simplified Expression (caller owns, must free with expression_free())
// Rules applied: zero elimination, identity elimination, constant folding, like-term combining
// Note: Does NOT modify input expression, returns new tree
// Memory: Caller must call expression_free() on result
Expression* simplify(Expression *expr);

// Specific simplification rules (advanced users, prefer simplify() for general use)
Expression* simplify_zero_terms(Expression *expr);      // 0*x -> 0, x+0 -> x
Expression* simplify_identity_terms(Expression *expr);  // 1*x -> x, x^1 -> x
Expression* simplify_constants(Expression *expr);       // Fold constant operations

// ============================================================================
// Utility Functions
// ============================================================================

// Check if expression contains a specific variable
bool expr_contains_var(Expression *expr, const char *var);

// Count occurrences of a variable in expression
int expr_count_var(Expression *expr, const char *var);

// Extract all unique variables from expression
char** expr_extract_all_vars(Expression *expr, int *out_n_vars);

#endif // CALCULUS_H
