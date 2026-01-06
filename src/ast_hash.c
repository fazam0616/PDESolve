#include "../include/ast_hash.h"
#include "../include/literal.h"
#include <stdlib.h>
#include <string.h>

// ============================================================================
// FNV-1a Hash Constants
// ============================================================================

#define FNV_OFFSET_BASIS 14695981039346656037ULL
#define FNV_PRIME 1099511628211ULL

// ============================================================================
// Hash Utilities
// ============================================================================

static uint64_t fnv1a_hash_bytes(const void *data, size_t len, uint64_t hash) {
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static uint64_t fnv1a_hash_string(const char *str) {
    uint64_t hash = FNV_OFFSET_BASIS;
    return fnv1a_hash_bytes(str, strlen(str), hash);
}

static uint64_t fnv1a_hash_uint64(uint64_t value) {
    uint64_t hash = FNV_OFFSET_BASIS;
    return fnv1a_hash_bytes(&value, sizeof(value), hash);
}

static uint64_t fnv1a_combine(uint64_t hash1, uint64_t hash2) {
    return fnv1a_hash_uint64(hash1) ^ fnv1a_hash_uint64(hash2);
}

// ============================================================================
// Comparison for sorting child hashes (for commutative normalization)
// ============================================================================

static int compare_uint64(const void *a, const void *b) {
    uint64_t val_a = *(const uint64_t *)a;
    uint64_t val_b = *(const uint64_t *)b;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    return 0;
}

// ============================================================================
// Expression Hashing
// ============================================================================

uint64_t expr_canonical_hash(Expression *expr) {
    if (expr == NULL) {
        return FNV_OFFSET_BASIS;
    }
    
    // Check cache
    if (expr->_hash_cache != 0) {
        return expr->_hash_cache;
    }
    
    uint64_t hash = FNV_OFFSET_BASIS;
    
    // Hash expression type
    hash = fnv1a_hash_bytes(&expr->type, sizeof(expr->type), hash);
    
    switch (expr->type) {
        case EXPR_LITERAL: {
            // Hash literal shape
            Literal *lit = expr->data.literal;
            hash = fnv1a_hash_bytes(lit->shape, sizeof(lit->shape), hash);
            
            // Hash literal field values (only if field is allocated)
            if (lit->field) {
                size_t n_elements = literal_total_elements(lit);
                hash = fnv1a_hash_bytes(lit->field, 
                                       n_elements * sizeof(double), hash);
            }
            break;
        }
        
        case EXPR_VARIABLE:
            // Hash variable name
            hash = fnv1a_hash_bytes(expr->data.variable, 
                                   strlen(expr->data.variable), hash);
            break;
            
        case EXPR_UNARY: {
            // Hash operation
            hash = fnv1a_hash_bytes(&expr->data.unary.op, sizeof(expr->data.unary.op), hash);
            // Hash operand recursively
            uint64_t operand_hash = expr_canonical_hash(expr->data.unary.operand);
            hash = fnv1a_combine(hash, operand_hash);
            // For OP_DERIVATIVE, also hash the variable name
            if (expr->data.unary.op == OP_DERIVATIVE && expr->data.unary.with_respect_to) {
                hash = fnv1a_hash_bytes(expr->data.unary.with_respect_to,
                                       strlen(expr->data.unary.with_respect_to), hash);
            }
            break;
        }
        
        case EXPR_BINARY: {
            // Hash operation
            hash = fnv1a_hash_bytes(&expr->data.binary.op, sizeof(expr->data.binary.op), hash);
            
            // Compute child hashes
            uint64_t left_hash = expr_canonical_hash(expr->data.binary.left);
            uint64_t right_hash = expr_canonical_hash(expr->data.binary.right);
            
            // If index specification exists, include it in hash
            if (expr->data.binary.index_spec != NULL) {
                // Hash index strings
                hash = fnv1a_hash_bytes(expr->data.binary.index_spec->left_indices,
                                       strlen(expr->data.binary.index_spec->left_indices), hash);
                hash = fnv1a_hash_bytes(expr->data.binary.index_spec->right_indices,
                                       strlen(expr->data.binary.index_spec->right_indices), hash);
                hash = fnv1a_hash_bytes(expr->data.binary.index_spec->out_indices,
                                       strlen(expr->data.binary.index_spec->out_indices), hash);
                
                // For Einstein operations, order ALWAYS matters
                hash = fnv1a_combine(hash, left_hash);
                hash = fnv1a_combine(hash, right_hash);
            } else {
                // Check if operation is commutative
                // NOTE: OP_MULTIPLY is element-wise/scalar (2*x = x*2), which IS commutative
                // OP_MATMUL is matrix multiplication (A@B != B@A), which is NON-commutative
                // OP_DOT is dot product (order preserved for clarity, though mathematically commutative)
                // OP_EINSUM without index_spec should not happen, but treat as non-commutative
                bool is_commutative = (expr->data.binary.op == OP_ADD || 
                                      expr->data.binary.op == OP_MULTIPLY);
                
                if (is_commutative) {
                    // Sort child hashes for canonical ordering (2*x and x*2 get same hash)
                    uint64_t hashes[2] = {left_hash, right_hash};
                    qsort(hashes, 2, sizeof(uint64_t), compare_uint64);
                    hash = fnv1a_combine(hash, hashes[0]);
                    hash = fnv1a_combine(hash, hashes[1]);
                } else {
                    // Preserve order for non-commutative operations (OP_DOT, OP_MATMUL, OP_EINSUM)
                    hash = fnv1a_combine(hash, left_hash);
                    hash = fnv1a_combine(hash, right_hash);
                }
            }
            break;
        }
    }
    
    // Cache result
    expr->_hash_cache = hash;
    
    return hash;
}

void expr_clear_hash_cache(Expression *expr) {
    if (expr == NULL) return;
    
    expr->_hash_cache = 0;
    
    switch (expr->type) {
        case EXPR_LITERAL:
        case EXPR_VARIABLE:
            // Base cases - nothing to recurse
            break;
            
        case EXPR_UNARY:
            expr_clear_hash_cache(expr->data.unary.operand);
            break;
        
        case EXPR_BINARY:
            expr_clear_hash_cache(expr->data.binary.left);
            expr_clear_hash_cache(expr->data.binary.right);
            break;
    }
}

// ============================================================================
// Expression Registry
// ============================================================================

ExpressionRegistry* registry_create(int initial_capacity) {
    ExpressionRegistry *reg = malloc(sizeof(ExpressionRegistry));
    if (reg == NULL) return NULL;
    
    reg->n_buckets = initial_capacity;
    reg->n_expressions = 0;
    reg->buckets = calloc(initial_capacity, sizeof(RegistryEntry*));
    
    if (reg->buckets == NULL) {
        free(reg);
        return NULL;
    }
    
    return reg;
}

static int registry_bucket_index(ExpressionRegistry *reg, uint64_t hash) {
    return (int)(hash % reg->n_buckets);
}

bool registry_register(ExpressionRegistry *reg, Expression *expr) {
    if (reg == NULL || expr == NULL) return false;
    
    uint64_t hash = expr_canonical_hash(expr);
    int bucket = registry_bucket_index(reg, hash);
    
    // Search for existing entry with this hash
    RegistryEntry *entry = reg->buckets[bucket];
    while (entry != NULL) {
        if (entry->hash == hash) {
            // Hash exists - add to collision list
            ExprNode *node = malloc(sizeof(ExprNode));
            if (node == NULL) return false;
            
            node->expr = expr;
            node->next = entry->expressions;
            entry->expressions = node;
            
            reg->n_expressions++;
            return false;  // Not new hash
        }
        entry = entry->next;
    }
    
    // New hash - create entry
    entry = malloc(sizeof(RegistryEntry));
    if (entry == NULL) return false;
    
    entry->hash = hash;
    entry->expressions = malloc(sizeof(ExprNode));
    if (entry->expressions == NULL) {
        free(entry);
        return false;
    }
    
    entry->expressions->expr = expr;
    entry->expressions->next = NULL;
    
    entry->next = reg->buckets[bucket];
    reg->buckets[bucket] = entry;
    
    reg->n_expressions++;
    return true;  // New hash
}

bool registry_contains(ExpressionRegistry *reg, Expression *expr) {
    if (reg == NULL || expr == NULL) return false;
    
    uint64_t hash = expr_canonical_hash(expr);
    int bucket = registry_bucket_index(reg, hash);
    
    RegistryEntry *entry = reg->buckets[bucket];
    while (entry != NULL) {
        if (entry->hash == hash) {
            return true;
        }
        entry = entry->next;
    }
    
    return false;
}

ExprNode* registry_get_matching(ExpressionRegistry *reg, Expression *expr) {
    if (reg == NULL || expr == NULL) return NULL;
    
    uint64_t hash = expr_canonical_hash(expr);
    return registry_get_by_hash(reg, hash);
}

ExprNode* registry_get_by_hash(ExpressionRegistry *reg, uint64_t hash) {
    if (reg == NULL) return NULL;
    
    int bucket = registry_bucket_index(reg, hash);
    
    RegistryEntry *entry = reg->buckets[bucket];
    while (entry != NULL) {
        if (entry->hash == hash) {
            return entry->expressions;
        }
        entry = entry->next;
    }
    
    return NULL;
}

void registry_free(ExpressionRegistry *reg) {
    if (reg == NULL) return;
    
    for (int i = 0; i < reg->n_buckets; i++) {
        RegistryEntry *entry = reg->buckets[i];
        while (entry != NULL) {
            RegistryEntry *next_entry = entry->next;
            
            // Free expression list
            ExprNode *node = entry->expressions;
            while (node != NULL) {
                ExprNode *next_node = node->next;
                free(node);
                node = next_node;
            }
            
            free(entry);
            entry = next_entry;
        }
    }
    
    free(reg->buckets);
    free(reg);
}

// ============================================================================
// Structural Equality
// ============================================================================

bool expr_structural_equals(Expression *expr1, Expression *expr2) {
    if (expr1 == NULL || expr2 == NULL) {
        return expr1 == expr2;
    }
    
    if (expr1->type != expr2->type) {
        return false;
    }
    
    switch (expr1->type) {
        case EXPR_VARIABLE:
            return strcmp(expr1->data.variable, expr2->data.variable) == 0;
        
        case EXPR_LITERAL: {
            Literal *lit1 = expr1->data.literal;
            Literal *lit2 = expr2->data.literal;
            
            // Compare shapes
            for (int i = 0; i < N_DIM; i++) {
                if (lit1->shape[i] != lit2->shape[i]) {
                    return false;
                }
            }
            
            // Compare field values
            uint64_t n_elements = literal_field_size();
            for (uint64_t i = 0; i < n_elements; i++) {
                if (lit1->field[i] != lit2->field[i]) {
                    return false;
                }
            }
            
            return true;
        }
            
        case EXPR_UNARY:
            if (expr1->data.unary.op != expr2->data.unary.op) {
                return false;
            }
            // For OP_DERIVATIVE, also compare variable names
            if (expr1->data.unary.op == OP_DERIVATIVE) {
                if (expr1->data.unary.with_respect_to == NULL || 
                    expr2->data.unary.with_respect_to == NULL) {
                    return expr1->data.unary.with_respect_to == expr2->data.unary.with_respect_to;
                }
                if (strcmp(expr1->data.unary.with_respect_to, 
                          expr2->data.unary.with_respect_to) != 0) {
                    return false;
                }
            }
            return expr_structural_equals(expr1->data.unary.operand,
                                         expr2->data.unary.operand);
        
        case EXPR_BINARY:
            if (expr1->data.binary.op != expr2->data.binary.op) {
                return false;
            }
            
            // Compare index specifications if present
            if (expr1->data.binary.index_spec != NULL || expr2->data.binary.index_spec != NULL) {
                // Both must have index specs, or both must not have them
                if (expr1->data.binary.index_spec == NULL || expr2->data.binary.index_spec == NULL) {
                    return false;
                }
                // Compare index strings
                if (strcmp(expr1->data.binary.index_spec->left_indices,
                          expr2->data.binary.index_spec->left_indices) != 0 ||
                    strcmp(expr1->data.binary.index_spec->right_indices,
                          expr2->data.binary.index_spec->right_indices) != 0 ||
                    strcmp(expr1->data.binary.index_spec->out_indices,
                          expr2->data.binary.index_spec->out_indices) != 0) {
                    return false;
                }
            }
            
            // NOTE: Structural equality is STRICT - does not consider commutativity
            // 2*x and x*2 have the SAME hash (for deduplication) but are NOT
            // structurally equal (different tree structure). This is intentional:
            // - Hash equality → potential collision, check structural equality
            // - Structural equality → guaranteed identical tree structure
            return expr_structural_equals(expr1->data.binary.left,
                                         expr2->data.binary.left) &&
                   expr_structural_equals(expr1->data.binary.right,
                                         expr2->data.binary.right);
    }
    
    return false;
}
