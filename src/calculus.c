#include "../include/calculus.h"
#include "../include/ast_hash.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// ============================================================================
// Forward Declarations
// ============================================================================

// expr_copy is now public (declared in calculus.h)

// ============================================================================
// Helper: Create Literal Expression
// ============================================================================

static Expression* make_literal(double value) {
    Literal *lit = literal_create((uint32_t[]){1,1,1});
    lit->field[0] = value;
    return expr_literal(lit);
}

// ============================================================================
// Helper: Expression Copying (now public)
// ============================================================================

Expression* expr_copy(Expression *expr) {
    if (!expr) return make_literal(0.0);
    
    switch (expr->type) {
        case EXPR_VARIABLE:
            return expr_variable(expr->data.variable);
        case EXPR_LITERAL:
            return expr_literal(literal_copy(expr->data.literal));
        case EXPR_UNARY:
            if (expr->data.unary.op == OP_DERIVATIVE) {
                return expr_derivative(expr_copy(expr->data.unary.operand),
                                      expr->data.unary.with_respect_to);
            } else {
                return expr_unary(expr->data.unary.op, 
                                expr_copy(expr->data.unary.operand));
            }
        case EXPR_BINARY:
            // Copy with index specification if present
            if (expr->data.binary.index_spec != NULL) {
                return expr_einsum(
                    expr_copy(expr->data.binary.left),
                    expr->data.binary.index_spec->left_indices,
                    expr_copy(expr->data.binary.right),
                    expr->data.binary.index_spec->right_indices,
                    expr->data.binary.index_spec->out_indices);
            } else {
                return expr_binary(expr->data.binary.op,
                                 expr_copy(expr->data.binary.left),
                                 expr_copy(expr->data.binary.right));
            }
    }
    
    return make_literal(0.0);
}

// ============================================================================
// Utility Functions: Variable Detection
// ============================================================================

bool expr_contains_var(Expression *expr, const char *var) {
    if (!expr || !var) return false;
    
    switch (expr->type) {
        case EXPR_VARIABLE:
            return strcmp(expr->data.variable, var) == 0;
        case EXPR_LITERAL:
            return false;
        case EXPR_UNARY:
            return expr_contains_var(expr->data.unary.operand, var);
        case EXPR_BINARY:
            return expr_contains_var(expr->data.binary.left, var) ||
                   expr_contains_var(expr->data.binary.right, var);
    }
    return false;
}

int expr_count_var(Expression *expr, const char *var) {
    if (!expr || !var) return 0;
    
    switch (expr->type) {
        case EXPR_VARIABLE:
            return strcmp(expr->data.variable, var) == 0 ? 1 : 0;
        case EXPR_LITERAL:
            return 0;
        case EXPR_UNARY:
            return expr_count_var(expr->data.unary.operand, var);
        case EXPR_BINARY:
            return expr_count_var(expr->data.binary.left, var) +
                   expr_count_var(expr->data.binary.right, var);
    }
    return 0;
}

// ============================================================================
// Utility Functions: Variable Extraction
// ============================================================================

static void add_unique_var(char ***list, int *count, int *capacity, const char *var) {
    for (int i = 0; i < *count; i++) {
        if (strcmp((*list)[i], var) == 0) return;
    }
    
    if (*count >= *capacity) {
        *capacity = (*capacity == 0) ? 4 : (*capacity * 2);
        *list = realloc(*list, sizeof(char*) * (*capacity));
    }
    
    (*list)[*count] = strdup(var);
    (*count)++;
}

static void extract_vars_from_expr(Expression *expr, char ***list, int *count, int *capacity) {
    if (!expr) return;
    
    switch (expr->type) {
        case EXPR_VARIABLE:
            add_unique_var(list, count, capacity, expr->data.variable);
            break;
        case EXPR_LITERAL:
            break;
        case EXPR_UNARY:
            extract_vars_from_expr(expr->data.unary.operand, list, count, capacity);
            break;
        case EXPR_BINARY:
            extract_vars_from_expr(expr->data.binary.left, list, count, capacity);
            extract_vars_from_expr(expr->data.binary.right, list, count, capacity);
            break;
    }
}

char** expr_extract_all_vars(Expression *expr, int *n_vars) {
    if (!expr || !n_vars) {
        if (n_vars) *n_vars = 0;
        return NULL;
    }
    
    char **list = NULL;
    int count = 0;
    int capacity = 0;
    
    extract_vars_from_expr(expr, &list, &count, &capacity);
    
    *n_vars = count;
    return list;
}

// ============================================================================
// Core Differentiation Engine
// ============================================================================

// Main differentiation function for expressions
Expression* derivative(Expression *expr, const char *var) {
    if (!expr || !var) {
        return make_literal(0.0);
    }
    
    switch (expr->type) {
        case EXPR_LITERAL:
            // d/dx(constant) = 0
            return make_literal(0.0);
            
        case EXPR_VARIABLE:
            // d/dx(x) = 1, d/dx(y) = 0
            if (strcmp(expr->data.variable, var) == 0) {
                return make_literal(1.0);
            } else {
                return make_literal(0.0);
            }
            
        case EXPR_UNARY: {
            Expression *u = expr->data.unary.operand;
            
            switch (expr->data.unary.op) {
                case OP_NEGATE:
                    // d/dx(-u) = -du/dx
                    return expr_negate(derivative(u, var));
                    
                case OP_TRANSPOSE:
                    // d/dx(u^T) = (du/dx)^T
                    return expr_transpose(derivative(u, var));
                    
                case OP_DERIVATIVE: {
                    // d/dx(∂u/∂y) = ∂²u/∂x∂y (mixed partial)
                    // Chain rule: compute ∂/∂y first (already stored), then ∂/∂x
                    const char *inner_var = expr->data.unary.with_respect_to;
                    
                    // Check if we're taking the same derivative
                    if (strcmp(inner_var, var) == 0) {
                        // d/dx(∂u/∂x) = ∂²u/∂x² (second derivative)
                        Expression *du_dx = derivative(u, var);
                        return expr_derivative(du_dx, var);
                    } else {
                        // d/dx(∂u/∂y) = ∂²u/∂x∂y (mixed partial)
                        Expression *du_dy = derivative(u, inner_var);
                        return expr_derivative(du_dy, var);
                    }
                }
                    
                default:
                    return make_literal(0.0);
            }
        }
        
        case EXPR_BINARY: {
            Expression *u = expr->data.binary.left;
            Expression *v = expr->data.binary.right;
            
            switch (expr->data.binary.op) {
                case OP_ADD:
                    // d/dx(u + v) = du/dx + dv/dx (Sum rule)
                    return expr_add(derivative(u, var), derivative(v, var));
                    
                case OP_MULTIPLY:
                    // d/dx(u * v) = u * dv/dx + du/dx * v (Product rule for element-wise)
                    // This is the KEY for polynomial differentiation!
                    {
                        Expression *du = derivative(u, var);
                        Expression *dv = derivative(v, var);
                        Expression *u_copy = expr_copy(u);
                        Expression *v_copy = expr_copy(v);
                        
                        // u * dv/dx
                        Expression *left_term = expr_multiply(u_copy, dv);
                        
                        // du/dx * v
                        Expression *right_term = expr_multiply(du, v_copy);
                        
                        return expr_add(left_term, right_term);
                    }
                
                case OP_MATMUL:
                    // d/dx(A @ B) = (dA/dx) @ B + A @ (dB/dx) (Product rule for matrix mult)
                    // Note: Order matters! Matrix multiplication is non-commutative
                    {
                        Expression *du = derivative(u, var);
                        Expression *dv = derivative(v, var);
                        Expression *u_copy = expr_copy(u);
                        Expression *v_copy = expr_copy(v);
                        
                        // (dA/dx) @ B
                        Expression *left_term = expr_matmul(du, v_copy);
                        
                        // A @ (dB/dx)
                        Expression *right_term = expr_matmul(u_copy, dv);
                        
                        return expr_add(left_term, right_term);
                    }
                    
                case OP_DOT:
                    // d/dx(u · v) = u · dv/dx + du/dx · v (Product rule for dot product)
                    {
                        Expression *du = derivative(u, var);
                        Expression *dv = derivative(v, var);
                        Expression *u_copy = expr_copy(u);
                        Expression *v_copy = expr_copy(v);
                        
                        Expression *left_term = expr_dot(u_copy, dv);
                        Expression *right_term = expr_dot(du, v_copy);
                        
                        return expr_add(left_term, right_term);
                    }
                    
                case OP_EINSUM:
                    // d/dx(einsum[...](A, B)) = einsum[...](dA/dx, B) + einsum[...](A, dB/dx)
                    // Product rule with preserved index structure
                    if (expr->data.binary.index_spec != NULL) {
                        Expression *du = derivative(u, var);
                        Expression *dv = derivative(v, var);
                        Expression *u_copy = expr_copy(u);
                        Expression *v_copy = expr_copy(v);
                        
                        IndexSpec *spec = expr->data.binary.index_spec;
                        
                        // (dA/dx) op B with same index spec
                        Expression *left_term = expr_einsum(du, spec->left_indices,
                                                           v_copy, spec->right_indices,
                                                           spec->out_indices);
                        
                        // A op (dB/dx) with same index spec
                        Expression *right_term = expr_einsum(u_copy, spec->left_indices,
                                                            dv, spec->right_indices,
                                                            spec->out_indices);
                        
                        return expr_add(left_term, right_term);
                    }
                    return make_literal(0.0);
                    
                default:
                    return make_literal(0.0);
            }
        }
    }
    
    return make_literal(0.0);
}

// ============================================================================
// Multivariable Calculus
// ============================================================================

Expression* partial_derivative(Expression *expr, const char *var) {
    return derivative(expr, var);
}

Expression** gradient(Expression *expr, int *n_partials) {
    if (!expr || !n_partials) {
        if (n_partials) *n_partials = 0;
        return NULL;
    }
    
    int n_vars;
    char **vars = expr_extract_all_vars(expr, &n_vars);
    
    if (n_vars == 0) {
        *n_partials = 0;
        return NULL;
    }
    
    Expression **grad = malloc(sizeof(Expression*) * n_vars);
    for (int i = 0; i < n_vars; i++) {
        grad[i] = partial_derivative(expr, vars[i]);
        free(vars[i]);
    }
    free(vars);
    
    *n_partials = n_vars;
    return grad;
}

Expression* laplacian(Expression *expr) {
    if (!expr) return NULL;
    
    int n_vars;
    char **vars = expr_extract_all_vars(expr, &n_vars);
    
    if (n_vars == 0) return NULL;
    
    // ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² + ...
    Expression *result = NULL;
    
    for (int i = 0; i < n_vars; i++) {
        Expression *first = partial_derivative(expr, vars[i]);
        Expression *second = partial_derivative(first, vars[i]);
        expression_free(first);
        
        if (result == NULL) {
            result = second;
        } else {
            Expression *sum = expr_add(result, second);
            result = sum;
        }
        
        free(vars[i]);
    }
    free(vars);
    
    return result;
}

Expression* mixed_partial(Expression *expr, const char *var1, const char *var2) {
    if (!expr || !var1 || !var2) return NULL;
    
    Expression *first = partial_derivative(expr, var1);
    Expression *second = partial_derivative(first, var2);
    expression_free(first);
    
    return second;
}

Expression* nth_derivative(Expression *expr, const char *var, int n) {
    if (n < 0 || !expr || !var) return NULL;
    if (n == 0) {
        // 0th derivative is just the expression itself
        return expr_copy(expr);
    }
    
    Expression *result = derivative(expr, var);
    for (int i = 1; i < n; i++) {
        Expression *next = derivative(result, var);
        expression_free(result);
        result = next;
    }
    
    return result;
}

// ============================================================================
// Finite Difference Operators (Stubs - require expression substitution)
// ============================================================================

Expression* finite_difference(Expression *expr, const char *var, double h, FiniteDifferenceType type) {
    // TODO: Requires expression substitution mechanism
    // For numerical PDEs when symbolic differentiation isn't desired
    (void)expr; (void)var; (void)h; (void)type;
    return NULL;
}

Expression* finite_difference_2nd(Expression *expr, const char *var, double h) {
    // TODO: Requires expression substitution mechanism
    (void)expr; (void)var; (void)h;
    return NULL;
}

// ============================================================================
// Expression Simplification Helpers
// ============================================================================

// Check if expression is a literal with specific value
static bool is_literal_value(Expression *expr, double value) {
    if (!expr || expr->type != EXPR_LITERAL) return false;
    Literal *lit = expr->data.literal;
    if (lit->shape[0] != 1 || lit->shape[1] != 1 || lit->shape[2] != 1) return false;
    return lit->field[0] == value;
}

// Check if expression is zero
static bool is_zero(Expression *expr) {
    return is_literal_value(expr, 0.0);
}

// Check if expression is one
static bool is_one(Expression *expr) {
    return is_literal_value(expr, 1.0);
}

// Get coefficient from expression (returns 1.0 for bare variables, actual coefficient for c*x)
static double get_coefficient(Expression *expr, Expression **base_out) {
    if (!expr) {
        *base_out = NULL;
        return 0.0;
    }
    
    // Check if it's c * x pattern
    if (expr->type == EXPR_BINARY && expr->data.binary.op == OP_MULTIPLY) {
        Expression *left = expr->data.binary.left;
        Expression *right = expr->data.binary.right;
        
        // c * x pattern (coefficient on left)
        if (left->type == EXPR_LITERAL) {
            Literal *lit = left->data.literal;
            if (lit->shape[0] == 1 && lit->shape[1] == 1 && lit->shape[2] == 1) {
                *base_out = right;
                return lit->field[0];
            }
        }
        
        // x * c pattern (coefficient on right)
        if (right->type == EXPR_LITERAL) {
            Literal *lit = right->data.literal;
            if (lit->shape[0] == 1 && lit->shape[1] == 1 && lit->shape[2] == 1) {
                *base_out = left;
                return lit->field[0];
            }
        }
    }
    
    // Not a coefficient pattern - treat as 1*expr
    *base_out = expr;
    return 1.0;
}

// Forward declaration for expression simplification
static Expression* simplify_expr(Expression *expr);

// Main expression simplification (TAIL-RECURSIVE with NO wrappers!)
static Expression* simplify_expr(Expression *expr) {
    if (!expr) return NULL;
    
    switch (expr->type) {
        case EXPR_LITERAL:
        case EXPR_VARIABLE:
            // Base cases - copy and return
            return expr_copy(expr);
            
        case EXPR_UNARY: {
            Expression *operand = expr->data.unary.operand;
            Expression *simplified_op = simplify_expr(operand);
            
            switch (expr->data.unary.op) {
                case OP_NEGATE:
                    // -0 → 0
                    if (is_zero(simplified_op)) {
                        expression_free(simplified_op);
                        return make_literal(0.0);
                    }
                    return expr_negate(simplified_op);
                    
                case OP_TRANSPOSE:
                    return expr_transpose(simplified_op);
                    
                default:
                    return expr_unary(expr->data.unary.op, simplified_op);
            }
        }
        
        case EXPR_BINARY: {
            Expression *left = expr->data.binary.left;
            Expression *right = expr->data.binary.right;
            Expression *simplified_left = simplify_expr(left);
            Expression *simplified_right = simplify_expr(right);
            
            switch (expr->data.binary.op) {
                case OP_ADD:
                    // 0 + x → x (NO WRAPPER! Direct return!)
                    if (is_zero(simplified_left)) {
                        expression_free(simplified_left);
                        return simplified_right;
                    }
                    // x + 0 → x (NO WRAPPER! Direct return!)
                    if (is_zero(simplified_right)) {
                        expression_free(simplified_right);
                        return simplified_left;
                    }
                    // Constant folding: c1 + c2 → result (as literal, NO WRAPPER!)
                    if (simplified_left->type == EXPR_LITERAL && simplified_right->type == EXPR_LITERAL) {
                        Literal *l_lit = simplified_left->data.literal;
                        Literal *r_lit = simplified_right->data.literal;
                        if (l_lit->shape[0] == 1 && l_lit->shape[1] == 1 && l_lit->shape[2] == 1 &&
                            r_lit->shape[0] == 1 && r_lit->shape[1] == 1 && r_lit->shape[2] == 1) {
                            double result = l_lit->field[0] + r_lit->field[0];
                            expression_free(simplified_left);
                            expression_free(simplified_right);
                            return make_literal(result);
                        }
                    }
                    
                    // Like-term folding: c1*x + c2*x → (c1+c2)*x
                    // Also handles: x + x → 2*x (since coefficient defaults to 1.0)
                    {
                        Expression *base_left = NULL;
                        Expression *base_right = NULL;
                        double coef_left = get_coefficient(simplified_left, &base_left);
                        double coef_right = get_coefficient(simplified_right, &base_right);
                        
                        // Check if bases are structurally equal
                        if (base_left && base_right && expr_structural_equals(base_left, base_right)) {
                            double combined_coef = coef_left + coef_right;
                            
                            // Special case: coefficient is 0 → return 0
                            if (combined_coef == 0.0) {
                                expression_free(simplified_left);
                                expression_free(simplified_right);
                                return make_literal(0.0);
                            }
                            
                            // Special case: coefficient is 1 → return base
                            if (combined_coef == 1.0) {
                                Expression *result = expr_copy(base_left);
                                expression_free(simplified_left);
                                expression_free(simplified_right);
                                return result;
                            }
                            
                            // General case: return (c1+c2)*base
                            Expression *result = expr_multiply(make_literal(combined_coef), expr_copy(base_left));
                            expression_free(simplified_left);
                            expression_free(simplified_right);
                            return result;
                        }
                    }
                    
                    return expr_add(simplified_left, simplified_right);
                    
                case OP_MULTIPLY:
                    // 0 * x → 0 (NO WRAPPER!)
                    if (is_zero(simplified_left) || is_zero(simplified_right)) {
                        expression_free(simplified_left);
                        expression_free(simplified_right);
                        return make_literal(0.0);
                    }
                    // 1 * x → x (NO WRAPPER! Direct return!)
                    if (is_one(simplified_left)) {
                        expression_free(simplified_left);
                        return simplified_right;
                    }
                    // x * 1 → x (NO WRAPPER! Direct return!)
                    if (is_one(simplified_right)) {
                        expression_free(simplified_right);
                        return simplified_left;
                    }
                    // Constant folding: c1 * c2 → result (NO WRAPPER!)
                    if (simplified_left->type == EXPR_LITERAL && simplified_right->type == EXPR_LITERAL) {
                        Literal *l_lit = simplified_left->data.literal;
                        Literal *r_lit = simplified_right->data.literal;
                        if (l_lit->shape[0] == 1 && l_lit->shape[1] == 1 && l_lit->shape[2] == 1 &&
                            r_lit->shape[0] == 1 && r_lit->shape[1] == 1 && r_lit->shape[2] == 1) {
                            double result = l_lit->field[0] * r_lit->field[0];
                            expression_free(simplified_left);
                            expression_free(simplified_right);
                            return make_literal(result);
                        }
                    }
                    return expr_multiply(simplified_left, simplified_right);
                
                case OP_MATMUL:
                    // 0 @ x → 0 (NO WRAPPER!)
                    if (is_zero(simplified_left) || is_zero(simplified_right)) {
                        expression_free(simplified_left);
                        expression_free(simplified_right);
                        return make_literal(0.0);
                    }
                    // Note: Cannot simplify I @ A → A without identity matrix detection
                    return expr_matmul(simplified_left, simplified_right);
                    
                case OP_DOT:
                    // 0 · x → 0 (NO WRAPPER!)
                    if (is_zero(simplified_left) || is_zero(simplified_right)) {
                        expression_free(simplified_left);
                        expression_free(simplified_right);
                        return make_literal(0.0);
                    }
                    return expr_dot(simplified_left, simplified_right);
                    
                case OP_EINSUM:
                    // Zero propagation: 0 ⊗ B → 0 or A ⊗ 0 → 0
                    if (is_zero(simplified_left) || is_zero(simplified_right)) {
                        expression_free(simplified_left);
                        expression_free(simplified_right);
                        return make_literal(0.0);
                    }
                    // Keep index specification
                    if (expr->data.binary.index_spec != NULL) {
                        return expr_einsum(simplified_left,
                                         expr->data.binary.index_spec->left_indices,
                                         simplified_right,
                                         expr->data.binary.index_spec->right_indices,
                                         expr->data.binary.index_spec->out_indices);
                    }
                    return expr_binary(expr->data.binary.op, simplified_left, simplified_right);
                    
                default:
                    // For unknown operations, preserve index_spec if present
                    if (expr->data.binary.index_spec != NULL) {
                        return expr_einsum(simplified_left,
                                         expr->data.binary.index_spec->left_indices,
                                         simplified_right,
                                         expr->data.binary.index_spec->right_indices,
                                         expr->data.binary.index_spec->out_indices);
                    }
                    return expr_binary(expr->data.binary.op, simplified_left, simplified_right);
            }
        }
    }
    
    return NULL;
}

// ============================================================================
// Expression Simplification (Public API)
// ============================================================================

Expression* simplify(Expression *expr) {
    if (!expr) return NULL;
    
    // TRUE TAIL RECURSION - SINGLE PASS! NO ITERATION NEEDED!
    // With flattened architecture, x*1 → x directly (not x+0)
    return simplify_expr(expr);
}

Expression* simplify_zero_terms(Expression *expr) {
    // With flattened architecture, simplify() already does this in one pass
    return simplify(expr);
}

Expression* simplify_identity_terms(Expression *expr) {
    // With flattened architecture, simplify() already does this in one pass
    return simplify(expr);
}

Expression* simplify_constants(Expression *expr) {
    // With flattened architecture, simplify() already does this in one pass
    return simplify(expr);
}
