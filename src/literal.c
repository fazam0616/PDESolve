#include "../include/literal.h"
#include "../include/tensor_ops.h"
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

// Compute row-major strides for a shape: strides[d] = product(shape[d+1..N_DIM-1])
static void compute_strides(const uint32_t *shape, size_t *strides) {
    size_t s = 1;
    for (int d = N_DIM - 1; d >= 0; d--) {
        strides[d] = s;
        s *= shape[d];
    }
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
    if (!lit) return;
    if (!lit->field && value != 0.0) {
        size_t n = literal_total_elements(lit);
        lit->field = malloc(sizeof(double) * n);
        if (!lit->field) return;
        memset(lit->field, 0, sizeof(double) * n);
    }
    if (!lit->field) return; // nothing to do when writing zero into unallocated field

    size_t strides[N_DIM];
    compute_strides(lit->shape, strides);
    size_t offset = 0;
    for (int d = 0; d < N_DIM; d++) {
        offset += (size_t)indices[d] * strides[d];
    }
    lit->field[offset] = value;
}

// Check if literal is all zero (field == NULL)
bool literal_is_zero(const Literal *lit) {
    return lit->field == NULL;
}


// Negate all elements in a literal
Literal* literal_negate(Literal *lit) {
    if (!lit) return NULL;
    Literal *result = literal_create_zero(lit->shape);
    if (!result) return NULL;
    size_t n = literal_total_elements(lit);
    if (!lit->field) return result; // remains zero (no allocation)
    result->field = malloc(sizeof(double) * n);
    if (!result->field) { free(result); return NULL; }
    const double *src = lit->field;
    double *dst = result->field;
    for (size_t i = 0; i < n; i++) dst[i] = -src[i];
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
    // Use strides to avoid repeated division/modulo in main loop.
    size_t src_strides[N_DIM];
    size_t tgt_strides[N_DIM];
    compute_strides(src->shape, src_strides);
    compute_strides(target_shape, tgt_strides);

    uint32_t counters[N_DIM];
    for (int d = 0; d < N_DIM; d++) counters[d] = 0;
    size_t total = literal_total_elements(result);
    size_t src_offset = 0;
    for (size_t flat = 0; flat < total; flat++) {
        // read from src at offset computed from counters but with broadcasting
        src_offset = 0;
        for (int d = 0; d < N_DIM; d++) {
            uint32_t idx = counters[d];
            uint32_t sdim = src->shape[d];
            if (sdim != 1) src_offset += (size_t)idx * src_strides[d];
        }
        double v = src->field ? src->field[src_offset] : 0.0;
        // direct write to result buffer (literal_set would reallocate and be slower)
        if (result->field) result->field[flat] = v;

        // increment counters without division: ripple increment
        for (int d = N_DIM - 1; d >= 0; d--) {
            counters[d]++;
            if (counters[d] < target_shape[d]) break;
            counters[d] = 0;
        }
    }
    return result;
}

// Matrix/tensor multiplication (non-commutative)
// For 2D: standard matrix multiplication  
// For higher dimensions: treat as batch of matrices on last 2 dimensions
Literal* literal_matmul(Literal *left, Literal *right) {
    if (N_DIM < 2) return NULL;
    size_t M = left->shape[N_DIM - 2];
    size_t K = left->shape[N_DIM - 1];
    size_t K2 = right->shape[N_DIM - 2];
    size_t N = right->shape[N_DIM - 1];
    if (K != K2) return NULL;

    uint32_t result_shape[N_DIM];
    for (int d = 0; d < N_DIM - 2; d++) result_shape[d] = left->shape[d];
    result_shape[N_DIM - 2] = (uint32_t)M;
    result_shape[N_DIM - 1] = (uint32_t)N;

    Literal *result = literal_create(result_shape);
    if (!result) return NULL;

    size_t batch = 1;
    for (int d = 0; d < N_DIM - 2; d++) batch *= left->shape[d];

    tops_matmul(result->field, left->field, right->field, M, K, N, batch);
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
    if (!result) return NULL;
    size_t size = literal_total_elements(left);
    if (result->field) result->field[0] = tops_dot(left->field, right->field, size);
    return result;
}

// Transpose (swaps last two dimensions)
Literal* literal_transpose(Literal *lit, bool *success) {
    *success = false;
    if (N_DIM < 2) return NULL;
    uint32_t result_shape[N_DIM];
    for (int i = 0; i < N_DIM - 2; i++) result_shape[i] = lit->shape[i];
    result_shape[N_DIM - 2] = lit->shape[N_DIM - 1];
    result_shape[N_DIM - 1] = lit->shape[N_DIM - 2];
    Literal *result = literal_create(result_shape);
    if (!result) return NULL;

    size_t batch = 1;
    for (int d = 0; d < N_DIM - 2; d++) batch *= lit->shape[d];
    size_t M = lit->shape[N_DIM - 2];
    size_t N = lit->shape[N_DIM - 1];

    tops_transpose(result->field, lit->field, M, N, batch);
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
            size_t M = left->shape[N_DIM - 2];
            size_t N = left->shape[N_DIM - 1];
            size_t batch = 1;
            for (int d = 0; d < N_DIM - 2; d++) batch *= left->shape[d];
            tops_transpose(result->field, left->field, M, N, batch);
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
            size_t m = left->shape[N_DIM - 2];
            size_t k = left->shape[N_DIM - 1];
            size_t k2 = right->shape[N_DIM - 2];
            size_t n = right->shape[N_DIM - 1];
            if (k != k2) { literal_free(result); return NULL; }
            size_t batch = 1;
            for (int d = 0; d < N_DIM - 2; d++) batch *= left->shape[d];
            tops_matmul(result->field, left->field, right->field, m, k, n, batch);
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
            size_t n = literal_total_elements(left);
            if (result->field) result->field[0] = tops_dot(left->field, right->field, n);
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
    // Fast path: identical shapes -> direct pointer loop
    bool identical = true;
    for (int i = 0; i < N_DIM; i++) if (left->shape[i] != right->shape[i]) { identical = false; break; }
    size_t total = 1;
    for (int i = 0; i < N_DIM; i++) total *= target_shape[i];

    Literal *result = literal_create(target_shape);
    if (!result) return NULL;

    if (identical) {
        tops_add_into(result->field, left->field, right->field, total);
        return result;
    }

    // Broadcasting case
    tops_add_bcast(result->field, left->field, right->field,
                   left->shape, right->shape, target_shape, N_DIM);
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
    // Fast path: identical shapes
    bool identical = true;
    for (int i = 0; i < N_DIM; i++) if (left->shape[i] != right->shape[i]) { identical = false; break; }
    size_t total = 1;
    for (int i = 0; i < N_DIM; i++) total *= target_shape[i];

    Literal *result = literal_create(target_shape);
    if (!result) return NULL;

    if (identical) {
        tops_subtract_into(result->field, left->field, right->field, total);
        return result;
    }

    // Broadcasting case
    tops_subtract_bcast(result->field, left->field, right->field,
                        left->shape, right->shape, target_shape, N_DIM);
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
    // Fast path: identical shapes
    bool identical = true;
    for (int i = 0; i < N_DIM; i++) if (left->shape[i] != right->shape[i]) { identical = false; break; }
    size_t total = 1;
    for (int i = 0; i < N_DIM; i++) total *= target_shape[i];

    Literal *result = literal_create(target_shape);
    if (!result) return NULL;

    if (identical) {
        tops_multiply_into(result->field, left->field, right->field, total);
        return result;
    }

    // Broadcasting case
    tops_multiply_bcast(result->field, left->field, right->field,
                        left->shape, right->shape, target_shape, N_DIM);
    return result;
}

// Element-wise power with broadcasting: base ^ exponent
Literal* literal_pow(Literal *base, Literal *exponent) {
    if (!base || !exponent) return NULL;
    // Determine broadcasted shape
    uint32_t target_shape[N_DIM];
    for (int i = 0; i < N_DIM; i++) {
        if (base->shape[i] == exponent->shape[i]) target_shape[i] = base->shape[i];
        else if (base->shape[i] == 1) target_shape[i] = exponent->shape[i];
        else if (exponent->shape[i] == 1) target_shape[i] = base->shape[i];
        else return NULL;
    }
    size_t total = 1; for (int i=0;i<N_DIM;i++) total *= target_shape[i];
    Literal *result = literal_create(target_shape);
    if (!result) return NULL;
    // Fast path identical
    bool identical = true; for (int i=0;i<N_DIM;i++) if (base->shape[i] != exponent->shape[i]) { identical = false; break; }
    if (identical && base->field && exponent->field) {
        for (size_t i=0;i<total;i++) result->field[i] = pow(base->field[i], exponent->field[i]);
        return result;
    }
    // General broadcasted loop
    size_t bstr[N_DIM], estr[N_DIM]; compute_strides(base->shape, bstr); compute_strides(exponent->shape, estr);
    uint32_t counters[N_DIM]; for (int d=0; d<N_DIM; d++) counters[d]=0;
    for (size_t flat=0; flat<total; flat++) {
        size_t boff=0, eoff=0;
        for (int d=0; d<N_DIM; d++) {
            if (base->shape[d] != 1) boff += (size_t)counters[d] * bstr[d];
            if (exponent->shape[d] != 1) eoff += (size_t)counters[d] * estr[d];
        }
        double a = base->field ? base->field[boff] : 0.0;
        double b = exponent->field ? exponent->field[eoff] : 0.0;
        result->field[flat] = pow(a, b);
        for (int d=N_DIM-1; d>=0; d--) { counters[d]++; if (counters[d] < target_shape[d]) break; counters[d]=0; }
    }
    return result;
}

Literal* literal_min(Literal *a, Literal *b) {
    if (!a || !b) return NULL;
    uint32_t target_shape[N_DIM];
    for (int i = 0; i < N_DIM; i++) {
        if (a->shape[i] == b->shape[i]) target_shape[i] = a->shape[i];
        else if (a->shape[i] == 1) target_shape[i] = b->shape[i];
        else if (b->shape[i] == 1) target_shape[i] = a->shape[i];
        else return NULL;
    }
    size_t total = 1; for (int i=0;i<N_DIM;i++) total *= target_shape[i];
    Literal *result = literal_create(target_shape); if (!result) return NULL;
    size_t astr[N_DIM], bstr[N_DIM]; compute_strides(a->shape, astr); compute_strides(b->shape, bstr);
    uint32_t counters[N_DIM]; for (int d=0; d<N_DIM; d++) counters[d]=0;
    for (size_t flat=0; flat<total; flat++) {
        size_t aoff=0, boff=0;
        for (int d=0; d<N_DIM; d++) { if (a->shape[d] != 1) aoff += (size_t)counters[d] * astr[d]; if (b->shape[d] != 1) boff += (size_t)counters[d] * bstr[d]; }
        double av = a->field ? a->field[aoff] : 0.0; double bv = b->field ? b->field[boff] : 0.0;
        result->field[flat] = av < bv ? av : bv;
        for (int d=N_DIM-1; d>=0; d--) { counters[d]++; if (counters[d] < target_shape[d]) break; counters[d]=0; }
    }
    return result;
}

Literal* literal_max(Literal *a, Literal *b) {
    if (!a || !b) return NULL;
    uint32_t target_shape[N_DIM];
    for (int i = 0; i < N_DIM; i++) {
        if (a->shape[i] == b->shape[i]) target_shape[i] = a->shape[i];
        else if (a->shape[i] == 1) target_shape[i] = b->shape[i];
        else if (b->shape[i] == 1) target_shape[i] = a->shape[i];
        else return NULL;
    }
    size_t total = 1; for (int i=0;i<N_DIM;i++) total *= target_shape[i];
    Literal *result = literal_create(target_shape); if (!result) return NULL;
    size_t astr[N_DIM], bstr[N_DIM]; compute_strides(a->shape, astr); compute_strides(b->shape, bstr);
    uint32_t counters[N_DIM]; for (int d=0; d<N_DIM; d++) counters[d]=0;
    for (size_t flat=0; flat<total; flat++) {
        size_t aoff=0, boff=0;
        for (int d=0; d<N_DIM; d++) { if (a->shape[d] != 1) aoff += (size_t)counters[d] * astr[d]; if (b->shape[d] != 1) boff += (size_t)counters[d] * bstr[d]; }
        double av = a->field ? a->field[aoff] : 0.0; double bv = b->field ? b->field[boff] : 0.0;
        result->field[flat] = av > bv ? av : bv;
        for (int d=N_DIM-1; d>=0; d--) { counters[d]++; if (counters[d] < target_shape[d]) break; counters[d]=0; }
    }
    return result;
}

// Scale all elements
Literal* literal_scale(Literal *lit, double scalar) {
    if (!lit) return NULL;
    Literal *result = literal_copy(lit);
    if (!result) return NULL;
    size_t size = literal_total_elements(result);
    tops_scale_inplace(result->field, scalar, size);
    return result;
}

// L2 norm
double literal_norm(Literal *lit) {
    if (!lit) return 0.0;
    size_t size = literal_total_elements(lit);
    return tops_norm(lit->field, size);
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
