#ifndef EXPRESSION_H
#define EXPRESSION_H

#include "literal.h"
#include "dictionary.h"
#include <stdint.h>
#include <stdbool.h>

// Forward declarations
typedef struct Expression Expression;
typedef struct GridMetadata GridMetadata;

// Index specification for Einstein summation
typedef struct {
    char *left_indices;   // Index string for left operand (e.g., "ij")
    char *right_indices;  // Index string for right operand (e.g., "jk")
    char *out_indices;    // Index string for result (e.g., "ik")
} IndexSpec;

// Expression type discriminator - now includes literals and variables
typedef enum {
    EXPR_LITERAL,    // Scalar or tensor literal value
    EXPR_VARIABLE,   // Variable name
    EXPR_UNARY,      // Unary operation (negate, transpose)
    EXPR_BINARY      // Binary operation (add, multiply, dot)
} ExpressionType;

// Operation types (only for EXPR_UNARY and EXPR_BINARY)
typedef enum {
    OP_ADD,          // Binary: a + b
    OP_NEGATE,       // Unary: -a
    OP_MULTIPLY,     // Binary: a * b (element-wise/scalar, commutative)
    OP_MATMUL,       // Binary: A @ B (matrix multiplication, NON-commutative)
    OP_DOT,          // Binary: dot product
    OP_TRANSPOSE,    // Unary: transpose
    OP_DERIVATIVE,   // Unary: ∂f/∂x (symbolic partial derivative)
    OP_LAPLACIAN,    // Unary: ∇²f (Laplacian - sum of second derivatives)
    OP_EINSUM        // Binary: Einstein summation with explicit indices
} Operation;

// Unified expression structure
struct Expression {
    ExpressionType type;
    union {
        // For EXPR_LITERAL
        Literal *literal;
        
        // For EXPR_VARIABLE
        char *variable;
        
        // For EXPR_UNARY
        struct {
            Operation op;
            Expression *operand;
            char *with_respect_to;  // For OP_DERIVATIVE: variable name (NULL for other ops)
        } unary;
        
        // For EXPR_BINARY
        struct {
            Operation op;
            Expression *left;
            Expression *right;
            IndexSpec *index_spec;  // For OP_EINSUM: index specification (NULL for other ops)
        } binary;
    } data;
    int ref_count;          // Reference count for memory management
    uint64_t _hash_cache;   // Memoized canonical hash (0 = not computed)
};

// Expression creation functions

// Create literal expression from Literal value
// lit: Literal to wrap (ownership transferred to expression)
// Returns: New Expression (caller owns, must call expression_free)
// Memory: Takes ownership of lit, will free it in expression_free
Expression* expr_literal(Literal *lit);

// Create variable expression
// name: Variable name (copied internally, caller retains ownership)
// Returns: New Expression (caller owns, must call expression_free)
// Memory: Copies name string, caller retains name ownership
Expression* expr_variable(const char *name);

// Create unary operation expression
// op: Operation (must be OP_NEGATE or OP_TRANSPOSE)
// operand: Operand expression (ownership transferred)
// Returns: New Expression (caller owns, must call expression_free)
// Memory: Takes ownership of operand, will free it in expression_free
Expression* expr_unary(Operation op, Expression *operand);

// Create binary operation expression
// op: Operation (OP_ADD, OP_MULTIPLY, or OP_DOT)
// left: Left operand (ownership transferred)
// right: Right operand (ownership transferred)
// Returns: New Expression (caller owns, must call expression_free)
// Memory: Takes ownership of both operands, will free them in expression_free
Expression* expr_binary(Operation op, Expression *left, Expression *right);

// Create scalar literal expression
// value: Scalar value to wrap in literal
// Returns: New Expression wrapping scalar literal (caller owns, must call expression_free)
Expression* make_scalar(double value);

// Helper functions for common operations (convenience wrappers)

// Create addition: left + right
Expression* expr_add(Expression *left, Expression *right);

// Create negation: -operand
Expression* expr_negate(Expression *operand);

// Create multiplication: left * right (element-wise/scalar, commutative)
Expression* expr_multiply(Expression *left, Expression *right);

// Create matrix multiplication: A @ B (non-commutative)
Expression* expr_matmul(Expression *left, Expression *right);

// Create dot product: left · right
Expression* expr_dot(Expression *left, Expression *right);

// Create transpose: operand^T
Expression* expr_transpose(Expression *operand);

// Create symbolic derivative: ∂operand/∂var
// operand: Expression to differentiate (ownership transferred)
// var: Variable name (copied internally, caller retains ownership)
// Returns: New Expression representing symbolic derivative (caller owns)
// Memory: Takes ownership of operand, copies var
Expression* expr_derivative(Expression *operand, const char *var);

// Create Laplacian: ∇²operand (sum of second derivatives)
// operand: Expression to apply Laplacian to (ownership transferred)
// Returns: New Expression representing Laplacian (caller owns)
// Memory: Takes ownership of operand
// Note: For grid fields, evaluates as sum of grid_field_derivative(field, axis, 2) over all axes
Expression* expr_laplacian(Expression *operand);

// Create Einstein summation: generalized tensor contraction
// left: Left operand expression (ownership transferred)
// left_indices: Index string for left operand (e.g., "ij") (copied internally)
// right: Right operand expression (ownership transferred, can be NULL for unary)
// right_indices: Index string for right operand (e.g., "jk") (copied internally)
// out_indices: Index string for result (e.g., "ik") (copied internally)
// Returns: New Expression with OP_EINSUM (caller owns, must call expression_free)
// Memory: Takes ownership of operands, copies all index strings
// Examples:
//   expr_einsum(A, "ij", B, "jk", "ik")  // Matrix multiply: C_ik = A_ij B_jk
//   expr_einsum(A, "ii", NULL, "", "")   // Trace: scalar = sum_i A_ii
//   expr_einsum(A, "ij", NULL, "", "ji") // Transpose: B_ji = A_ij
//   expr_einsum(a, "i", b, "j", "ij")    // Outer product: C_ij = a_i b_j
Expression* expr_einsum(Expression *left, const char *left_indices,
                       Expression *right, const char *right_indices,
                       const char *out_indices);

// Memory management

// Free expression and all owned resources recursively
// expr: Expression to free (can be NULL)
// Memory: Recursively frees all child expressions, literals, and strings
void expression_free(Expression *expr);

// Add reference counting to Expression
// Increment reference count
void expression_retain(Expression *expr);

// Decrement reference count and free if no references remain
void expression_release(Expression *expr);

// Operation queries

// Check if operation is unary (OP_NEGATE or OP_TRANSPOSE)
bool operation_is_unary(Operation op);

// Check if operation is binary (OP_ADD, OP_MULTIPLY, OP_MATMUL, or OP_DOT)
bool operation_is_binary(Operation op);

// Print for debugging

// Print expression to stdout in infix notation
// expr: Expression to print (can be NULL)
void print_expression(Expression *expr);

// Evaluation

// Evaluate expression numerically given variable values
// expr: Expression to evaluate (must not be NULL)
// vars: Dictionary of variable values (can be NULL if no variables)
// Returns: Evaluated Literal result (caller owns, must call literal_free)
// Memory: Returns newly allocated Literal, caller must free
// Note: Returns NULL if undefined variable encountered
Literal* expression_evaluate(Expression *expr, Dictionary *vars);

// Evaluate expression with grid-aware operations (finite differences for derivatives)
// expr: Expression to evaluate (must not be NULL)
// vars: Dictionary of variable values (can be NULL if no variables)
// grid: Grid metadata for finite difference operations (must not be NULL)
// Returns: Evaluated Literal result (caller owns, must call literal_free)
// Memory: Returns newly allocated Literal, caller must free
// Note: If variable is grid field (shape matches grid), applies finite differences for OP_DERIVATIVE
Literal* expression_evaluate_grid(Expression *expr, Dictionary *vars, GridMetadata *grid);

// Partially evaluate expression (substitute known variables)
// expr: Expression to evaluate (must not be NULL)
// vars: Dictionary of variable values (can be NULL)
// Returns: New Expression with substitutions (caller owns, must call expression_free)
// Memory: Returns newly allocated Expression, caller must free
// Note: Returns copy of expr if no substitutions made
Expression* expr_evaluate(Expression *expr, Dictionary *vars);

#endif // EXPRESSION_H
