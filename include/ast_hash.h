#ifndef AST_HASH_H
#define AST_HASH_H

#include "expression.h"
#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// AST Canonical Hashing for Expression Deduplication
// ============================================================================
// Implements FNV-1a 64-bit hashing with commutative normalization
// No algebraic simplification - preserves structural forms like 0*x
// Collision handling: expressions with same hash stored in linked lists

// ============================================================================
// Hash Computation
// ============================================================================

// Compute canonical hash for an expression
// Normalizes commutative operations (OP_ADD, OP_MULTIPLY) by sorting child hashes
// Uses memoization via _hash_cache field in Expression struct
uint64_t expr_canonical_hash(Expression *expr);

// Clear cached hashes (call after modifying expression tree)
void expr_clear_hash_cache(Expression *expr);

// ============================================================================
// Expression Registry (Deduplication)
// ============================================================================

// Registry entry: hash -> list of expressions with that hash
typedef struct ExprNode {
    Expression *expr;
    struct ExprNode *next;
} ExprNode;

typedef struct RegistryEntry RegistryEntry;

struct RegistryEntry {
    uint64_t hash;
    ExprNode *expressions;  // Linked list of expressions with this hash
    RegistryEntry *next;
};

typedef struct {
    RegistryEntry **buckets;
    int n_buckets;
    int n_expressions;
} ExpressionRegistry;

// Create a new expression registry
ExpressionRegistry* registry_create(int initial_capacity);

// Register an expression (returns true if new, false if duplicate hash exists)
// Note: false positives possible - same hash doesn't guarantee structural equality
bool registry_register(ExpressionRegistry *reg, Expression *expr);

// Check if expression hash exists in registry
bool registry_contains(ExpressionRegistry *reg, Expression *expr);

// Get all expressions with the same hash as the given expression
// Used for collision handling and template matching
ExprNode* registry_get_matching(ExpressionRegistry *reg, Expression *expr);

// Get all expressions with a specific hash value
ExprNode* registry_get_by_hash(ExpressionRegistry *reg, uint64_t hash);

// Free registry (does NOT free expressions themselves)
void registry_free(ExpressionRegistry *reg);

// ============================================================================
// Structural Equality (for collision detection)
// ============================================================================

// Compare two expressions for structural equality
// Useful for verifying hash collisions
bool expr_structural_equals(Expression *expr1, Expression *expr2);

#endif // AST_HASH_H
