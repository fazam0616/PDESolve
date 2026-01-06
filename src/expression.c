#include "../include/expression.h"
#include "../include/grid.h"
#include <string.h>
#include "../include/literal.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdlib.h>

// Forward declaration for grid operations
GridField* grid_field_laplacian(const GridField *field);

// ============================================================================
// Forward Declarations for Helper Functions
// ============================================================================

static Expression* expression_copy(Expression *expr);
// Removed literal helper function declarations

// ============================================================================
// Expression Creation Functions
// ============================================================================

Expression* expr_literal(Literal *lit) {
    Expression *expr = malloc(sizeof(Expression));
    expr->type = EXPR_LITERAL;
    expr->data.literal = lit;
    expr->_hash_cache = 0;
    expr->ref_count = 1; // Initialize reference count
    return expr;
}

Expression* expr_variable(const char *name) {
    Expression *expr = malloc(sizeof(Expression));
    expr->type = EXPR_VARIABLE;
    expr->data.variable = malloc(strlen(name) + 1);
    strcpy(expr->data.variable, name);
    expr->_hash_cache = 0;
    expr->ref_count = 1; // Initialize reference count
    return expr;
}

Expression* expr_unary(Operation op, Expression *operand) {
    Expression *expr = malloc(sizeof(Expression));
    expr->type = EXPR_UNARY;
    expr->data.unary.op = op;
    expr->data.unary.operand = operand;
    expr->data.unary.with_respect_to = NULL;  // Only set for OP_DERIVATIVE
    expr->_hash_cache = 0;
    expr->ref_count = 1; // Initialize reference count
    expression_retain(operand); // Retain operand
    return expr;
}

Expression* expr_binary(Operation op, Expression *left, Expression *right) {
    Expression *expr = malloc(sizeof(Expression));
    expr->type = EXPR_BINARY;
    expr->data.binary.op = op;
    expr->data.binary.left = left;
    expr->data.binary.right = right;
    expr->data.binary.index_spec = NULL;  // NULL for non-Einstein operations
    expr->_hash_cache = 0;
    expr->ref_count = 1; // Initialize reference count
    expression_retain(left); // Retain left operand
    expression_retain(right); // Retain right operand
    return expr;
}

// ============================================================================
// Helper Functions for Common Operations
// ============================================================================

Expression* expr_add(Expression *left, Expression *right) {
    return expr_binary(OP_ADD, left, right);
}

Expression* expr_negate(Expression *operand) {
    return expr_unary(OP_NEGATE, operand);
}

Expression* expr_multiply(Expression *left, Expression *right) {
    return expr_binary(OP_MULTIPLY, left, right);
}

Expression* expr_matmul(Expression *left, Expression *right) {
    return expr_binary(OP_MATMUL, left, right);
}

Expression* expr_dot(Expression *left, Expression *right) {
    return expr_binary(OP_DOT, left, right);
}

Expression* expr_transpose(Expression *operand) {
    return expr_unary(OP_TRANSPOSE, operand);
}

Expression* expr_derivative(Expression *operand, const char *var) {
    Expression *expr = malloc(sizeof(Expression));
    expr->type = EXPR_UNARY;
    expr->data.unary.op = OP_DERIVATIVE;
    expr->data.unary.operand = operand;
    expr->data.unary.with_respect_to = strdup(var);
    expr->_hash_cache = 0;
    return expr;
}

Expression* expr_laplacian(Expression *operand) {
    Expression *expr = malloc(sizeof(Expression));
    expr->type = EXPR_UNARY;
    expr->data.unary.op = OP_LAPLACIAN;
    expr->data.unary.operand = operand;
    expr->data.unary.with_respect_to = NULL;  // Not used for Laplacian
    expr->_hash_cache = 0;
    expr->ref_count = 1;
    expression_retain(operand);
    return expr;
}

Expression* expr_einsum(Expression *left, const char *left_indices,
                       Expression *right, const char *right_indices,
                       const char *out_indices) {
    Expression *expr = malloc(sizeof(Expression));
    expr->type = EXPR_BINARY;
    expr->data.binary.op = OP_EINSUM;
    expr->data.binary.left = left;
    expr->data.binary.right = right;
    
    // Allocate and copy index specification
    expr->data.binary.index_spec = malloc(sizeof(IndexSpec));
    expr->data.binary.index_spec->left_indices = strdup(left_indices ? left_indices : "");
    expr->data.binary.index_spec->right_indices = strdup(right_indices ? right_indices : "");
    expr->data.binary.index_spec->out_indices = strdup(out_indices ? out_indices : "");
    
    expr->_hash_cache = 0;
    return expr;
}

// Create scalar literal expression
Expression *make_scalar(double val) {
    Literal *lit = literal_create_scalar(val);
    Expression *expr = expr_literal(lit);
    return expr;
}

// ============================================================================
// Memory Management
// ============================================================================

void expression_retain(Expression *expr) {
    if (expr) {
        expr->ref_count++;
    }
}

void expression_release(Expression *expr) {
    if (expr && --expr->ref_count == 0) {
        expression_free(expr);
    }
}

void expression_free(Expression *expr) {
    if (expr == NULL) return;

    switch (expr->type) {
        case EXPR_LITERAL:
            literal_free(expr->data.literal);
            break;
        case EXPR_VARIABLE:
            free(expr->data.variable);
            break;
        case EXPR_UNARY:
            expression_release(expr->data.unary.operand);
            if (expr->data.unary.op == OP_DERIVATIVE) {
                free(expr->data.unary.with_respect_to);
            }
            break;
        case EXPR_BINARY:
            expression_release(expr->data.binary.left);
            expression_release(expr->data.binary.right);
            if (expr->data.binary.index_spec) {
                free(expr->data.binary.index_spec->left_indices);
                free(expr->data.binary.index_spec->right_indices);
                free(expr->data.binary.index_spec->out_indices);
                free(expr->data.binary.index_spec);
            }
            break;
    }

    free(expr);
}

// ============================================================================
// Operation Queries
// ============================================================================

bool operation_is_unary(Operation op) {
    return op == OP_NEGATE || op == OP_TRANSPOSE || op == OP_DERIVATIVE || op == OP_LAPLACIAN;
}

bool operation_is_binary(Operation op) {
    return op == OP_ADD || op == OP_MULTIPLY || op == OP_MATMUL || op == OP_DOT;
}

// ============================================================================
// Debug Printing
// ============================================================================

static const char* operation_to_string(Operation op) {
    switch (op) {
        case OP_ADD: return "+";
        case OP_NEGATE: return "-";
        case OP_MULTIPLY: return "*";
        case OP_MATMUL: return "@";
        case OP_DOT: return "dot";
        case OP_TRANSPOSE: return "T";
        case OP_DERIVATIVE: return "d/d";
        case OP_LAPLACIAN: return "∇²";
        case OP_EINSUM: return "einsum";
        default: return "?";
    }
}

void print_expression(Expression *expr) {
    if (expr == NULL) {
        printf("NULL");
        return;
    }
    
    switch (expr->type) {
        case EXPR_LITERAL:
            literal_print(expr->data.literal);
            break;
        case EXPR_VARIABLE:
            printf("%s", expr->data.variable);
            break;
        case EXPR_UNARY:
            if (expr->data.unary.op == OP_DERIVATIVE) {
                printf("d(");
                print_expression(expr->data.unary.operand);
                printf(")/d%s", expr->data.unary.with_respect_to);
            } else if (expr->data.unary.op == OP_LAPLACIAN) {
                printf("∇²(");
                print_expression(expr->data.unary.operand);
                printf(")");
            } else {
                printf("%s", operation_to_string(expr->data.unary.op));
                printf("(");
                print_expression(expr->data.unary.operand);
                printf(")");
            }
            break;
        case EXPR_BINARY:
            if (expr->data.binary.index_spec != NULL) {
                // Einstein notation
                printf("einsum[%s,%s->%s](",
                       expr->data.binary.index_spec->left_indices,
                       expr->data.binary.index_spec->right_indices,
                       expr->data.binary.index_spec->out_indices);
                print_expression(expr->data.binary.left);
                if (expr->data.binary.right != NULL) {
                    printf(", ");
                    print_expression(expr->data.binary.right);
                }
                printf(")");
            } else {
                // Legacy operators: use infix notation
                printf("(");
                print_expression(expr->data.binary.left);
                printf(" %s ", operation_to_string(expr->data.binary.op));
                print_expression(expr->data.binary.right);
                printf(")");
            }
            break;
    }
}


// ============================================================================
// Evaluation
// ============================================================================

Literal* expression_evaluate(Expression *expr, Dictionary *vars) {
    if (expr == NULL) return NULL;
    
    switch (expr->type) {
        case EXPR_LITERAL:
            // Return a copy of the literal
            return literal_copy(expr->data.literal);
            
        case EXPR_VARIABLE: {
            // Look up variable in dictionary
            Literal *temp;
            if (vars != NULL && dict_get(vars, expr->data.variable, &temp)) {
                // Variable found - copy and return
                return literal_copy(temp);
            } else {
                // Variable not found
                fprintf(stderr, "Error: Undefined variable '%s'\n", expr->data.variable);
                return NULL;
            }
        }
            
        case EXPR_UNARY: {
            Literal *operand = expression_evaluate(expr->data.unary.operand, vars);
            if (operand == NULL) return NULL;
            
            Literal *result = NULL;
            bool success = false;
            
            switch (expr->data.unary.op) {
                case OP_NEGATE:
                    result = literal_negate(operand);
                    success = (result != NULL);
                    break;
                case OP_TRANSPOSE:
                    result = literal_transpose(operand, &success);
                    break;
                case OP_DERIVATIVE:
                    fprintf(stderr, "Error: Cannot numerically evaluate symbolic derivative. Use derivative() to compute symbolically first.\n");
                    break;
                case OP_LAPLACIAN:
                    fprintf(stderr, "Error: Cannot numerically evaluate Laplacian without grid context. Use expression_evaluate_grid().\n");
                    break;
                default:
                    fprintf(stderr, "Error: Unknown unary operation\n");
                    break;
            }
            
            literal_free(operand);
            return result;
        }
            
        case EXPR_BINARY: {
            Literal *left = expression_evaluate(expr->data.binary.left, vars);
            if (left == NULL) return NULL;

            Literal *right = NULL;
            bool is_einsum = (expr->data.binary.op == OP_EINSUM && expr->data.binary.index_spec != NULL);
            const char *einsum_right_indices = is_einsum ? expr->data.binary.index_spec->right_indices : NULL;
            bool einsum_right_empty = (einsum_right_indices == NULL || einsum_right_indices[0] == '\0');

            if (!is_einsum || !einsum_right_empty) {
                right = expression_evaluate(expr->data.binary.right, vars);
                if (right == NULL) {
                    literal_free(left);
                    return NULL;
                }
            }

            Literal *result = NULL;
            bool success = false;

            switch (expr->data.binary.op) {
                case OP_ADD:
                    result = literal_add(left, right);
                    success = (result != NULL);
                    break;
                case OP_MULTIPLY:
                    result = literal_multiply(left, right);
                    success = (result != NULL);
                    break;
                case OP_MATMUL:
                    result = literal_matmul(left, right);
                    success = (result != NULL);
                    break;
                case OP_DOT:
                    result = literal_dot(left, right);
                    success = (result != NULL);
                    break;
                case OP_EINSUM:
                    if (expr->data.binary.index_spec != NULL) {
                        result = literal_einsum(left,
                                              expr->data.binary.index_spec->left_indices,
                                              einsum_right_empty ? NULL : right,
                                              expr->data.binary.index_spec->right_indices,
                                              expr->data.binary.index_spec->out_indices,
                                              &success);
                    } else {
                        fprintf(stderr, "Error: OP_EINSUM requires index specification\n");
                    }
                    break;
                default:
                    fprintf(stderr, "Error: Unknown binary operation\n");
                    break;
            }

            literal_free(left);
            if (right) literal_free(right);
            return result;
        }
    }
    
    return NULL;
}

// Grid-aware expression evaluation - applies finite differences for derivatives on grid fields
Literal* expression_evaluate_grid(Expression *expr, Dictionary *vars, GridMetadata *grid) {
    if (expr == NULL || grid == NULL) return NULL;
    
    switch (expr->type) {
        case EXPR_LITERAL:
            // Return a copy of the literal
            return literal_copy(expr->data.literal);
            
        case EXPR_VARIABLE: {
            // Look up variable in dictionary
            Literal *temp;
            if (vars != NULL && dict_get(vars, expr->data.variable, &temp)) {
                // Variable found - copy and return
                return literal_copy(temp);
            } else {
                // Variable not found
                fprintf(stderr, "Error: Undefined variable '%s'\n", expr->data.variable);
                return NULL;
            }
        }
            
        case EXPR_UNARY: {
            Literal *operand = expression_evaluate_grid(expr->data.unary.operand, vars, grid);
            if (operand == NULL) return NULL;
            
            Literal *result = NULL;
            bool success = false;
            
            switch (expr->data.unary.op) {
                case OP_NEGATE:
                    result = literal_negate(operand);
                    success = (result != NULL);
                    break;
                case OP_TRANSPOSE:
                    result = literal_transpose(operand, &success);
                    break;
                case OP_DERIVATIVE: {
                    // Check if operand is a grid field
                    if (grid_literal_matches(operand, grid)) {
                        // Apply finite difference on grid
                        int axis = grid_axis_from_name(expr->data.unary.with_respect_to);
                        if (axis < 0 || axis >= grid->n_dims) {
                            fprintf(stderr, "Error: Unknown axis '%s' for derivative\n", 
                                    expr->data.unary.with_respect_to);
                            literal_free(operand);
                            return NULL;
                        }
                        
                        // Wrap literal as GridField, apply derivative, unwrap result
                        GridField *field = grid_field_wrap_literal(operand, grid);
                        GridField *deriv = grid_field_derivative(field, axis, 1);
                        
                        if (deriv) {
                            result = literal_copy(&deriv->data);
                            // Don't free field->data (shallow copy), just the wrapper
                            field->data.field = NULL;
                            grid_field_free(field);
                            grid_field_free(deriv);
                        } else {
                            field->data.field = NULL;
                            grid_field_free(field);
                        }
                    } else {
                        // Scalar derivative - cannot evaluate
                        fprintf(stderr, "Error: Cannot numerically evaluate symbolic derivative. Use derivative() to compute symbolically first.\n");
                    }
                    break;
                }
                case OP_LAPLACIAN: {
                    // Check if operand is a grid field
                    if (grid_literal_matches(operand, grid)) {
                        // Apply Laplacian on grid (sum of second derivatives)
                        GridField *field = grid_field_wrap_literal(operand, grid);
                        GridField *laplacian = grid_field_laplacian(field);
                        
                        if (laplacian) {
                            result = literal_copy(&laplacian->data);
                            // Don't free field->data (shallow copy), just the wrapper
                            field->data.field = NULL;
                            grid_field_free(field);
                            grid_field_free(laplacian);
                        } else {
                            field->data.field = NULL;
                            grid_field_free(field);
                        }
                    } else {
                        // Scalar Laplacian - cannot evaluate
                        fprintf(stderr, "Error: Cannot evaluate Laplacian on non-grid field\n");
                    }
                    break;
                }
                default:
                    fprintf(stderr, "Error: Unknown unary operation\n");
                    break;
            }
            
            literal_free(operand);
            return result;
        }
            
        case EXPR_BINARY: {
            Literal *left = expression_evaluate_grid(expr->data.binary.left, vars, grid);
            if (left == NULL) return NULL;

            Literal *right = NULL;
            bool is_einsum = (expr->data.binary.op == OP_EINSUM && expr->data.binary.index_spec != NULL);
            const char *einsum_right_indices = is_einsum ? expr->data.binary.index_spec->right_indices : NULL;
            bool einsum_right_empty = (einsum_right_indices == NULL || einsum_right_indices[0] == '\0');

            if (!is_einsum || !einsum_right_empty) {
                right = expression_evaluate_grid(expr->data.binary.right, vars, grid);
                if (right == NULL) {
                    literal_free(left);
                    return NULL;
                }
            }

            Literal *result = NULL;
            bool success = false;

            switch (expr->data.binary.op) {
                case OP_ADD:
                    result = literal_add(left, right);
                    success = (result != NULL);
                    break;
                case OP_MULTIPLY:
                    result = literal_multiply(left, right);
                    success = (result != NULL);
                    break;
                case OP_MATMUL:
                    result = literal_matmul(left, right);
                    success = (result != NULL);
                    break;
                case OP_DOT:
                    result = literal_dot(left, right);
                    success = (result != NULL);
                    break;
                case OP_EINSUM:
                    if (expr->data.binary.index_spec != NULL) {
                        result = literal_einsum(left,
                                              expr->data.binary.index_spec->left_indices,
                                              einsum_right_empty ? NULL : right,
                                              expr->data.binary.index_spec->right_indices,
                                              expr->data.binary.index_spec->out_indices,
                                              &success);
                    } else {
                        fprintf(stderr, "Error: OP_EINSUM requires index specification\n");
                    }
                    break;
                default:
                    fprintf(stderr, "Error: Unknown binary operation\n");
                    break;
            }

            literal_free(left);
            if (right) literal_free(right);
            return result;
        }
    }
    
    return NULL;
}

// ============================================================================
// Helper Functions
// ============================================================================

static Expression* expression_copy(Expression *expr) {
    if (expr == NULL) return NULL;
    
    switch (expr->type) {
        case EXPR_LITERAL:
            return expr_literal(literal_copy(expr->data.literal));
        case EXPR_VARIABLE:
            return expr_variable(expr->data.variable);
        case EXPR_UNARY:
            return expr_unary(expr->data.unary.op, 
                            expression_copy(expr->data.unary.operand));
        case EXPR_BINARY: {
            // Copy with index specification if present
            if (expr->data.binary.index_spec != NULL) {
                return expr_einsum(
                    expression_copy(expr->data.binary.left),
                    expr->data.binary.index_spec->left_indices,
                    expression_copy(expr->data.binary.right),
                    expr->data.binary.index_spec->right_indices,
                    expr->data.binary.index_spec->out_indices);
            } else {
                return expr_binary(expr->data.binary.op,
                                 expression_copy(expr->data.binary.left),
                                 expression_copy(expr->data.binary.right));
            }
        }
    }
    return NULL;
}
