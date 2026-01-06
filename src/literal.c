#include "../include/literal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

size_t literal_total_elements(const Literal *lit) {
    size_t size = 1;
    for (int i = 0; i < N_DIM; i++) {
        size *= lit->shape[i];
    }
    return size;
}

// Create a zero literal (no data allocated)
Literal* literal_create_zero(const uint32_t *shape) {
    Literal *lit = malloc(sizeof(Literal));
    if (!lit) return NULL;
    for (int i = 0; i < N_DIM; i++) lit->shape[i] = shape[i];
    lit->field = NULL;
    return lit;
}

// Create a literal (allocates data, initialized to zero)
Literal* literal_create(const uint32_t *shape) {
    Literal *lit = malloc(sizeof(Literal));
    if (!lit) return NULL;
    for (int i = 0; i < N_DIM; i++) lit->shape[i] = shape[i];
    size_t n = literal_total_elements(lit);
    lit->field = malloc(sizeof(double) * n);
    if (!lit->field) { free(lit); return NULL; }
    memset(lit->field, 0, sizeof(double) * n);
    return lit;
}

// Get value at index (returns 0.0 if field is NULL)
double literal_get(const Literal *lit, const uint32_t *indices) {
    if (!lit->field) return 0.0;
    size_t offset = 0, stride = 1;
    for (int i = N_DIM - 1; i >= 0; i--) {
        offset += indices[i] * stride;
        stride *= lit->shape[i];
    }
    return lit->field[offset];
}

// Deep copy of a Literal
Literal* literal_copy(Literal *src) {
    if (!src) return NULL;
    Literal *copy = malloc(sizeof(Literal));
    if (!copy) return NULL;
    for (int i = 0; i < N_DIM; i++) copy->shape[i] = src->shape[i];
    size_t n = literal_total_elements(src);
    if (src->field) {
        copy->field = malloc(sizeof(double) * n);
        if (!copy->field) { free(copy); return NULL; }
        memcpy(copy->field, src->field, sizeof(double) * n);
    } else {
        copy->field = NULL;
    }
    return copy;
}

// Free a Literal and its field
void literal_free(Literal *lit) {
    if (!lit) return;
    if (lit->field) free(lit->field);
    free(lit);
}

// Create a scalar Literal (shape {1,1,1})
Literal* literal_create_scalar(double val) {
    uint32_t shape[N_DIM];
    for (int i = 0; i < N_DIM; i++) shape[i] = 1;
    Literal *lit = literal_create(shape);
    if (!lit) return NULL;
    if (lit->field) lit->field[0] = val;
    return lit;
}

// Initialize literal with given shape (assumes field already allocated or allocates if needed)
void literal_init(Literal *lit, uint32_t *shape) {
    if (!lit) return;
    for (int i = 0; i < N_DIM; i++) lit->shape[i] = shape[i];
    size_t n = literal_total_elements(lit);
    if (!lit->field) {
        lit->field = malloc(sizeof(double) * n);
    }
    if (lit->field) {
        memset(lit->field, 0, sizeof(double) * n);
    }
}

// Zero out the entire field array
void literal_zero(Literal *lit) {
    if (!lit || !lit->field) return;
    size_t n = literal_total_elements(lit);
    memset(lit->field, 0, sizeof(double) * n);
}

// Fill an existing literal with given shape
Literal* literal_fill(Literal *lit, uint32_t* shape) {
    if (!lit) return NULL;
    literal_init(lit, shape);
    return lit;
}

// Set value at index (allocates field if needed)
void literal_set(Literal *lit, const uint32_t *indices, double value) {
    if (!lit->field && value != 0.0) {
        size_t n = literal_total_elements(lit);
        lit->field = malloc(sizeof(double) * n);
        memset(lit->field, 0, sizeof(double) * n);
    }
    if (lit->field) {
        size_t offset = 0, stride = 1;
        for (int i = N_DIM - 1; i >= 0; i--) {
            offset += indices[i] * stride;
            stride *= lit->shape[i];
        }
        lit->field[offset] = value;
    }
}

// Check if literal is all zero (field == NULL)
bool literal_is_zero(const Literal *lit) {
    return lit->field == NULL;
}


// Negate all elements in a literal
Literal* literal_negate(Literal *lit) {
    Literal *result = literal_copy(lit);
    if (result == NULL) return NULL;
    uint64_t size = literal_total_elements(lit);
    uint32_t idx[N_DIM];
    for (uint64_t i = 0; i < size; i++) {
        uint64_t rem = i;
        for (int d = N_DIM - 1; d >= 0; d--) {
            idx[d] = rem % lit->shape[d];
            rem /= lit->shape[d];
        }
        double v = literal_get(lit, idx);
        literal_set(result, idx, -v);
    }
    return result;
}

// Broadcast a literal to a target shape (returns new literal, caller must free)
// Returns NULL if broadcasting is not possible
Literal* literal_broadcast_to_shape(const Literal *src, const uint32_t *target_shape) {
    if (!src || !target_shape) return NULL;
    // Check broadcast compatibility
    for (int i = 0; i < N_DIM; i++) {
        if (src->shape[i] != target_shape[i] && src->shape[i] != 1) {
            return NULL; // Incompatible for broadcasting
        }
    }
    Literal *result = literal_create((uint32_t *)target_shape);
    if (!result) return NULL;
    // Fill result by broadcasting src
    uint64_t total = literal_total_elements(result);
    uint32_t idx[N_DIM];
    for (uint64_t flat = 0; flat < total; flat++) {
        uint64_t rem = flat;
        for (int d = N_DIM - 1; d >= 0; d--) {
            idx[d] = rem % target_shape[d];
            rem /= target_shape[d];
        }
        uint32_t src_idx[N_DIM];
        for (int d = 0; d < N_DIM; d++) {
            src_idx[d] = (src->shape[d] == 1) ? 0 : idx[d];
        }
        double v = literal_get(src, src_idx);
        literal_set(result, idx, v);
    }
    return result;
}

// Matrix/tensor multiplication (non-commutative)
// For 2D: standard matrix multiplication  
// For higher dimensions: treat as batch of matrices on last 2 dimensions
Literal* literal_matmul(Literal *left, Literal *right) {
    bool success = false;
    // For N_DIM >= 2, multiply on last two dimensions
    // left: [..., M, K], right: [..., K, N] -> result: [..., M, N]
    if (N_DIM < 2) return NULL;
    if (left->shape[N_DIM - 1] != right->shape[N_DIM - 2]) return NULL;
    uint32_t result_shape[N_DIM];
    for (int i = 0; i < N_DIM - 2; i++) {
        result_shape[i] = left->shape[i];
    }
    result_shape[N_DIM - 2] = left->shape[N_DIM - 2];  // M
    result_shape[N_DIM - 1] = right->shape[N_DIM - 1];  // N
    Literal *result = literal_create(result_shape);
    if (result == NULL) return NULL;
    uint64_t batch_size = 1;
    for (int i = 0; i < N_DIM - 2; i++) {
        batch_size *= left->shape[i];
    }
    uint32_t M = left->shape[N_DIM - 2];
    uint32_t K = left->shape[N_DIM - 1];
    uint32_t N = right->shape[N_DIM - 1];
    uint32_t idx[N_DIM], lidx[N_DIM], ridx[N_DIM];
    for (uint64_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                double sum = 0.0;
                for (uint32_t k = 0; k < K; k++) {
                    // Compute indices for left, right
                    for (int d = 0; d < N_DIM - 2; d++) lidx[d] = ridx[d] = idx[d] = (b / (batch_size / (left->shape[d] ? left->shape[d] : 1))) % (left->shape[d] ? left->shape[d] : 1);
                    lidx[N_DIM-2] = i; lidx[N_DIM-1] = k;
                    ridx[N_DIM-2] = k; ridx[N_DIM-1] = j;
                    sum += literal_get(left, lidx) * literal_get(right, ridx);
                }
                for (int d = 0; d < N_DIM - 2; d++) idx[d] = lidx[d];
                idx[N_DIM-2] = i; idx[N_DIM-1] = j;
                literal_set(result, idx, sum);
            }
        }
    }
    return result;
}

// Dot product (inner product)
// Flattens both literals and computes sum of element-wise products
Literal* literal_dot(Literal *left, Literal *right) {
    if (!left || !right) return NULL;
    for (int i = 0; i < N_DIM; i++) {
        if (left->shape[i] != right->shape[i]) return NULL;
    }
    uint32_t scalar_shape[N_DIM];
    for (int i = 0; i < N_DIM; i++) scalar_shape[i] = 1;
    Literal *result = literal_create(scalar_shape);
    if (result == NULL) return NULL;
    uint64_t size = literal_total_elements(left);
    double sum = 0.0;
    uint32_t idx[N_DIM];
    for (uint64_t i = 0; i < size; i++) {
        uint64_t rem = i;
        for (int d = N_DIM - 1; d >= 0; d--) {
            idx[d] = rem % left->shape[d];
            rem /= left->shape[d];
        }
        sum += literal_get(left, idx) * literal_get(right, idx);
    }
    literal_set(result, (uint32_t[]){0,0,0}, sum);
    return result;
}

// Transpose (swaps last two dimensions)
Literal* literal_transpose(Literal *lit, bool *success) {
    *success = false;
    if (N_DIM < 2) return NULL;
    uint32_t result_shape[N_DIM];
    for (int i = 0; i < N_DIM - 2; i++) {
        result_shape[i] = lit->shape[i];
    }
    result_shape[N_DIM - 2] = lit->shape[N_DIM - 1];
    result_shape[N_DIM - 1] = lit->shape[N_DIM - 2];
    Literal *result = literal_create(result_shape);
    if (result == NULL) return NULL;
    uint64_t batch_size = 1;
    for (int i = 0; i < N_DIM - 2; i++) {
        batch_size *= lit->shape[i];
    }
    uint32_t M = lit->shape[N_DIM - 2];
    uint32_t N = lit->shape[N_DIM - 1];
    uint32_t idx[N_DIM], oidx[N_DIM];
    for (uint64_t b = 0; b < batch_size; b++) {
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                for (int d = 0; d < N_DIM - 2; d++) idx[d] = oidx[d] = (b / (batch_size / (lit->shape[d] ? lit->shape[d] : 1))) % (lit->shape[d] ? lit->shape[d] : 1);
                idx[N_DIM-2] = i; idx[N_DIM-1] = j;
                oidx[N_DIM-2] = j; oidx[N_DIM-1] = i;
                double v = literal_get(lit, idx);
                literal_set(result, oidx, v);
            }
        }
    }
    *success = true;
    return result;
}

// ============================================================================
// Einstein Summation Implementation
// ============================================================================

// Index dimension information for einsum
typedef struct {
    char index;         // Index character ('a'-'z')
    uint32_t size;      // Dimension size
    int left_pos;       // Position in left operand (-1 if not present)
    int right_pos;      // Position in right operand (-1 if not present)
    int out_pos;        // Position in output (-1 if summed)
} IndexDim;

// Helper: Get dimension size from literal at given position
static uint32_t get_dim_size(Literal *lit, int pos, int total_indices) {
    // Map index position to actual dimension in N_DIM array
    // For N_DIM=3, 2-index operation uses dims [1, 2], 1-index uses [1]
    int start_dim = N_DIM - total_indices;
    if (start_dim < 0) start_dim = 0;
    return lit->shape[start_dim + pos];
}

// Validate shapes and build index dimension map
static bool validate_and_map_indices(
    Literal *left, const char *left_indices,
    Literal *right, const char *right_indices,
    const char *out_indices,
    IndexDim dims[26], int *n_unique_indices)
{
    int left_len = strlen(left_indices);
    int right_len = right_indices ? strlen(right_indices) : 0;
    int unique_count = 0;
    
    // Initialize
    for (int i = 0; i < 26; i++) {
        dims[i].index = 0;
        dims[i].size = 0;
        dims[i].left_pos = -1;
        dims[i].right_pos = -1;
        dims[i].out_pos = -1;
    }
    
    // Process left indices
    for (int i = 0; i < left_len; i++) {
        char idx = left_indices[i];
        if (idx < 'a' || idx > 'z') continue;
        
        int idx_num = idx - 'a';
        uint32_t dim_size = get_dim_size(left, i, left_len);
        
        if (dims[idx_num].index == 0) {
            // First occurrence
            dims[idx_num].index = idx;
            dims[idx_num].size = dim_size;
            dims[idx_num].left_pos = i;
            unique_count++;
        } else {
            // Repeated index in same operand (diagonal)
            if (dims[idx_num].size != dim_size) {
                return false;  // Dimension mismatch
            }
        }
    }
    
    // Process right indices
    if (right != NULL && right_indices != NULL) {
        for (int i = 0; i < right_len; i++) {
            char idx = right_indices[i];
            if (idx < 'a' || idx > 'z') continue;
            
            int idx_num = idx - 'a';
            uint32_t dim_size = get_dim_size(right, i, right_len);
            
            if (dims[idx_num].index == 0) {
                // New index from right operand
                dims[idx_num].index = idx;
                dims[idx_num].size = dim_size;
                dims[idx_num].right_pos = i;
                unique_count++;
            } else {
                // Contraction index - must match size
                dims[idx_num].right_pos = i;
                if (dims[idx_num].size != dim_size) {
                    return false;  // Incompatible contraction dimensions
                }
            }
        }
    }
    
    // Process output indices
    int out_len = strlen(out_indices);
    for (int i = 0; i < out_len; i++) {
        char idx = out_indices[i];
        if (idx < 'a' || idx > 'z') continue;
        
        int idx_num = idx - 'a';
        if (dims[idx_num].index == 0) {
            return false;  // Output index not in inputs
        }
        dims[idx_num].out_pos = i;
    }
    
    *n_unique_indices = unique_count;
    return true;
}

Literal* literal_einsum(Literal *left, const char *left_indices,
                       Literal *right, const char *right_indices,
                       const char *out_indices, bool *success) {
    *success = false;
    
    if (left == NULL || left_indices == NULL || out_indices == NULL) {
        return NULL;
    }
    
    // Validate shapes and build index dimension map
    IndexDim dims[26];
    int n_unique_indices = 0;
    
    if (!validate_and_map_indices(left, left_indices, right, right_indices, 
                                  out_indices, dims, &n_unique_indices)) {
        return NULL;  // Shape validation failed
    }
    
    // Determine output shape
    uint32_t out_shape[N_DIM];
    for (int i = 0; i < N_DIM; i++) {
        out_shape[i] = 1;
    }
    
    int out_len = strlen(out_indices);
    int start_dim = (out_len < N_DIM) ? (N_DIM - out_len) : 0;
    
    for (int i = 0; i < out_len && (start_dim + i) < N_DIM; i++) {
        char idx = out_indices[i];
        if (idx >= 'a' && idx <= 'z') {
            int idx_num = idx - 'a';
            out_shape[start_dim + i] = dims[idx_num].size;
        }
    }
    
    // Create output literal (zeroed)
    Literal *result = literal_create(out_shape);
    if (result == NULL) return NULL;
    
    // ========================================================================
    // FAST PATHS: Pattern matching for common operations
    // ========================================================================
    
    // If no right operand (unary operation like trace or transpose)
    if (right == NULL) {
        // Transpose: ij->ji
        if (strlen(left_indices) == 2 && strlen(out_indices) == 2 &&
            left_indices[0] == out_indices[1] && left_indices[1] == out_indices[0]) {
            // Get matrix dimensions from last two dims (assuming [batch, rows, cols])
            uint32_t m = left->shape[N_DIM - 2];
            uint32_t n = left->shape[N_DIM - 1];
            
            for (uint32_t i = 0; i < m; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    result->field[j * m + i] = left->field[i * n + j];
                }
            }
            *success = true;
            return result;
        }
        
        // Trace: ii->  (empty output)
        if (strlen(left_indices) == 2 && left_indices[0] == left_indices[1] &&
            strlen(out_indices) == 0) {
            double sum = 0.0;
            uint32_t n = left->shape[N_DIM - 2];  // Assume square matrix
            uint32_t stride = left->shape[N_DIM - 1];
            for (uint32_t i = 0; i < n; i++) {
                sum += left->field[i * stride + i];
            }
            result->field[0] = sum;
            *success = true;
            return result;
        }
    } else {
        // Binary operations
        
        // Matrix multiply: ij,jk->ik
        if (strlen(left_indices) == 2 && strlen(right_indices) == 2 &&
            strlen(out_indices) == 2 &&
            left_indices[1] == right_indices[0] &&
            left_indices[0] == out_indices[0] &&
            right_indices[1] == out_indices[1]) {
            // Use last two dimensions
            uint32_t m = left->shape[N_DIM - 2];
            uint32_t k = left->shape[N_DIM - 1];
            uint32_t k2 = right->shape[N_DIM - 2];
            uint32_t n = right->shape[N_DIM - 1];
            
            // Check contraction dimension matches
            if (k != k2) {
                literal_free(result);
                return NULL;
            }
            
            for (uint32_t i = 0; i < m; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (uint32_t kk = 0; kk < k; kk++) {
                        sum += left->field[i * k + kk] * right->field[kk * n + j];
                    }
                    result->field[i * n + j] = sum;
                }
            }
            *success = true;
            return result;
        }
        
        // Outer product: i,j->ij
        if (strlen(left_indices) == 1 && strlen(right_indices) == 1 &&
            strlen(out_indices) == 2 &&
            left_indices[0] == out_indices[0] &&
            right_indices[0] == out_indices[1]) {
            // Use last dimension for vector length (N_DIM-1)
            uint32_t m = left->shape[N_DIM - 1];
            uint32_t n = right->shape[N_DIM - 1];
            
            for (uint32_t i = 0; i < m; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    result->field[i * n + j] = left->field[i] * right->field[j];
                }
            }
            *success = true;
            return result;
        }
        
        // Dot product: i,i->  (empty output)
        if (strlen(left_indices) == 1 && strlen(right_indices) == 1 &&
            left_indices[0] == right_indices[0] && strlen(out_indices) == 0) {
            double sum = 0.0;
            uint32_t n = left->shape[N_DIM - 1];
            for (uint32_t i = 0; i < n; i++) {
                sum += left->field[i] * right->field[i];
            }
            result->field[0] = sum;
            *success = true;
            return result;
        }
        
        // Element-wise multiply: ij,ij->ij
        if (strlen(left_indices) == 2 && strlen(right_indices) == 2 &&
            strlen(out_indices) == 2 &&
            strcmp(left_indices, right_indices) == 0 &&
            strcmp(left_indices, out_indices) == 0) {
            uint32_t m = left->shape[N_DIM - 2];
            uint32_t n = left->shape[N_DIM - 1];
            
            for (uint32_t i = 0; i < m; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    result->field[i * n + j] = left->field[i * n + j] * right->field[i * n + j];
                }
            }
            *success = true;
            return result;
        }
    }
    
    // ========================================================================
    // GENERAL CASE: Recursive implementation for arbitrary contractions
    // ========================================================================
    
    // If we reach here, use general recursive approach
    // This handles any valid einsum that doesn't match fast paths above
    
    literal_free(result);
    return NULL;  // TODO: Implement general recursive case
}

// Element-wise addition
Literal* literal_add(Literal *left, Literal *right) {
    if (!left || !right) return NULL;
    // Determine broadcasted shape
    uint32_t target_shape[N_DIM];
    for (int i = 0; i < N_DIM; i++) {
        if (left->shape[i] == right->shape[i]) {
            target_shape[i] = left->shape[i];
        } else if (left->shape[i] == 1) {
            target_shape[i] = right->shape[i];
        } else if (right->shape[i] == 1) {
            target_shape[i] = left->shape[i];
        } else {
            return NULL; // Incompatible shapes
        }
    }
    Literal *left_b = literal_broadcast_to_shape(left, target_shape);
    Literal *right_b = literal_broadcast_to_shape(right, target_shape);
    if (!left_b || !right_b) {
        if (left_b) literal_free(left_b);
        if (right_b) literal_free(right_b);
        return NULL;
    }
    Literal *result = literal_copy(left_b);
    if (!result) {
        literal_free(left_b);
        literal_free(right_b);
        return NULL;
    }
    uint64_t size = literal_total_elements(result);
    uint32_t idx[N_DIM];
    for (uint64_t i = 0; i < size; i++) {
        uint64_t rem = i;
        for (int d = N_DIM - 1; d >= 0; d--) {
            idx[d] = rem % result->shape[d];
            rem /= result->shape[d];
        }
        double v = literal_get(left_b, idx) + literal_get(right_b, idx);
        literal_set(result, idx, v);
    }
    literal_free(left_b);
    literal_free(right_b);
    return result;
}

// Element-wise subtraction
Literal* literal_subtract(Literal *left, Literal *right) {
    if (!left || !right) return NULL;
    // Determine broadcasted shape
    uint32_t target_shape[N_DIM];
    for (int i = 0; i < N_DIM; i++) {
        if (left->shape[i] == right->shape[i]) {
            target_shape[i] = left->shape[i];
        } else if (left->shape[i] == 1) {
            target_shape[i] = right->shape[i];
        } else if (right->shape[i] == 1) {
            target_shape[i] = left->shape[i];
        } else {
            return NULL; // Incompatible shapes
        }
    }
    Literal *left_b = literal_broadcast_to_shape(left, target_shape);
    Literal *right_b = literal_broadcast_to_shape(right, target_shape);
    if (!left_b || !right_b) {
        if (left_b) literal_free(left_b);
        if (right_b) literal_free(right_b);
        return NULL;
    }
    Literal *result = literal_copy(left_b);
    if (!result) {
        literal_free(left_b);
        literal_free(right_b);
        return NULL;
    }
    uint64_t size = literal_total_elements(result);
    uint32_t idx[N_DIM];
    for (uint64_t i = 0; i < size; i++) {
        uint64_t rem = i;
        for (int d = N_DIM - 1; d >= 0; d--) {
            idx[d] = rem % result->shape[d];
            rem /= result->shape[d];
        }
        double v = literal_get(left_b, idx) - literal_get(right_b, idx);
        literal_set(result, idx, v);
    }
    literal_free(left_b);
    literal_free(right_b);
    return result;
}

// Element-wise multiplication
Literal* literal_multiply(Literal *left, Literal *right) {
    if (!left || !right) return NULL;
    // Determine broadcasted shape
    uint32_t target_shape[N_DIM];
    for (int i = 0; i < N_DIM; i++) {
        if (left->shape[i] == right->shape[i]) {
            target_shape[i] = left->shape[i];
        } else if (left->shape[i] == 1) {
            target_shape[i] = right->shape[i];
        } else if (right->shape[i] == 1) {
            target_shape[i] = left->shape[i];
        } else {
            return NULL; // Incompatible shapes
        }
    }
    Literal *left_b = literal_broadcast_to_shape(left, target_shape);
    Literal *right_b = literal_broadcast_to_shape(right, target_shape);
    if (!left_b || !right_b) {
        if (left_b) literal_free(left_b);
        if (right_b) literal_free(right_b);
        return NULL;
    }
    Literal *result = literal_copy(left_b);
    if (!result) {
        literal_free(left_b);
        literal_free(right_b);
        return NULL;
    }
    uint64_t size = literal_total_elements(result);
    uint32_t idx[N_DIM];
    for (uint64_t i = 0; i < size; i++) {
        uint64_t rem = i;
        for (int d = N_DIM - 1; d >= 0; d--) {
            idx[d] = rem % result->shape[d];
            rem /= result->shape[d];
        }
        double v = literal_get(left_b, idx) * literal_get(right_b, idx);
        literal_set(result, idx, v);
    }
    literal_free(left_b);
    literal_free(right_b);
    return result;
}

// Scale all elements
Literal* literal_scale(Literal *lit, double scalar) {
    if (!lit) return NULL;
    Literal *result = literal_copy(lit);
    if (!result) return NULL;
    uint64_t size = literal_total_elements(result);
    uint32_t idx[N_DIM];
    for (uint64_t i = 0; i < size; i++) {
        uint64_t rem = i;
        for (int d = N_DIM - 1; d >= 0; d--) {
            idx[d] = rem % result->shape[d];
            rem /= result->shape[d];
        }
        double v = literal_get(result, idx);
        literal_set(result, idx, v * scalar);
    }
    return result;
}

// L2 norm
double literal_norm(Literal *lit) {
    if (!lit) return 0.0;
    uint64_t size = literal_total_elements(lit);
    double sum = 0.0;
    uint32_t idx[N_DIM];
    for (uint64_t i = 0; i < size; i++) {
        uint64_t rem = i;
        for (int d = N_DIM - 1; d >= 0; d--) {
            idx[d] = rem % lit->shape[d];
            rem /= lit->shape[d];
        }
        double v = literal_get(lit, idx);
        sum += v * v;
    }
    return sqrt(sum);
}
// Helper to print a slice of a tensor recursively
static void _literal_print_recursive(const Literal *lit, int dim, uint32_t *indices) {
    if (dim == N_DIM - 2) {
        // Print a matrix
        uint32_t rows = lit->shape[N_DIM - 2];
        uint32_t cols = lit->shape[N_DIM - 1];
        printf("[");
        for (uint32_t i = 0; i < rows; i++) {
            if (i > 0) printf("\n ");
            printf("[");
            for (uint32_t j = 0; j < cols; j++) {
                indices[N_DIM - 2] = i;
                indices[N_DIM - 1] = j;
                printf("%8.4g", *literal_at((Literal *)lit, indices));
                if (j < cols - 1) printf(", ");
            }
            printf("]");
        }
        printf("]");
    } else if (dim == N_DIM - 1) {
        // Print a vector
        uint32_t len = lit->shape[N_DIM - 1];
        printf("[");
        for (uint32_t i = 0; i < len; i++) {
            indices[N_DIM - 1] = i;
            printf("%8.4g", *literal_at((Literal *)lit, indices));
            if (i < len - 1) printf(", ");
        }
        printf("]");
    } else {
        // Print higher-dimensional tensor as stacked matrices
        for (uint32_t i = 0; i < lit->shape[dim]; i++) {
            indices[dim] = i;
            printf("\n-- Slice %d/%d along axis %d --\n", i, lit->shape[dim], dim);
            _literal_print_recursive(lit, dim + 1, indices);
        }
    }
}

void literal_print(const Literal *lit) {
    if (!lit) {
        printf("(null)\n");
        return;
    }
    // Scalar case
    bool is_scalar = true;
    for (int i = 0; i < N_DIM; i++) {
        if (lit->shape[i] != 1) {
            is_scalar = false;
            break;
        }
    }
    if (is_scalar) {
        printf("%g\n", lit->field[0]);
        return;
    }
    // Otherwise, print recursively
    uint32_t indices[N_DIM];
    memset(indices, 0, sizeof(indices));
    _literal_print_recursive(lit, 0, indices);
    printf("\n");
}
