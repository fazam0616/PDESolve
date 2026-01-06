#ifndef LITERAL_H
#define LITERAL_H
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define N_DIM 3

typedef struct {
    uint32_t shape[N_DIM];
    double *field; // NULL means all zeros
} Literal;

// Create a zero literal (all elements zero, no data allocated)
Literal* literal_create_zero(const uint32_t *shape);

// Get value at index (returns 0.0 if field is NULL)
double literal_get(const Literal *lit, const uint32_t *indices);

// Set value at index (allocates field if needed)
void literal_set(Literal *lit, const uint32_t *indices, double value);

// Check if literal is all zero (field == NULL)
bool literal_is_zero(const Literal *lit);

// Helper: get total number of elements
size_t literal_total_elements(const Literal *lit);

// Configurable dimensions - can be overridden before including
#define N_DIM 3

#define MAX_DIM_SIZE 100  // Can be large now with dynamic allocation

// Helper to calculate field size at runtime
static inline uint64_t literal_field_size() {
    uint64_t size = 1;
    for (int i = 0; i < N_DIM; i++) {
        size *= MAX_DIM_SIZE;
    }
    return size;
}


// Access function for multi-dimensional indexing
// Example: for 3D with indices (i, j, k)
static inline double* literal_at(Literal *lit, uint32_t *indices) {
    uint64_t offset = 0;
    uint64_t multiplier = 1;
    
    // Row-major ordering
    for (int d = N_DIM - 1; d >= 0; d--) {
        offset += indices[d] * multiplier;
        multiplier *= lit->shape[d];
    }
    
    return &lit->field[offset];
}

// Create and destroy literals

// Create a new literal with given shape
// shape: Array of N_DIM dimensions (must not be NULL)
// Returns: Newly allocated Literal (caller owns, must call literal_free)
// Memory: Allocates Literal struct and field array (MAX_DIM_SIZE^N_DIM doubles)
//         Field is zeroed on creation
Literal* literal_create(const uint32_t *shape);

// Fill an existing literal with given shape
Literal* literal_fill(Literal *lit, uint32_t* shape);

// Free literal and its field array
// lit: Literal to free (can be NULL)
// Memory: Frees field array and Literal struct
void literal_free(Literal *lit);

// Initialize literal with given shape (assumes field already allocated)
// lit: Literal to initialize (must not be NULL, field can be NULL)
// shape: Array of N_DIM dimensions (must not be NULL)
// Note: If field is NULL, allocates it. Always zeros the field.
void literal_init(Literal *lit, uint32_t *shape);

// Zero out the entire field array
// lit: Literal to zero (can be NULL)
// Note: Zeros all MAX_DIM_SIZE^N_DIM elements (not just used elements)
void literal_zero(Literal *lit);

// Create a deep copy of a literal
// src: Source literal to copy (must not be NULL)
// Returns: New Literal with copied data (caller owns, must call literal_free)
// Memory: Allocates new Literal and field array, copies all data
Literal* literal_copy(Literal *src);

// Create a scalar literal (1x1x1 with single value)
// val: Scalar value to store
// Returns: New Literal with shape={1,1,1} (caller owns, must call literal_free)
// Memory: Allocates Literal and field array, sets field[0] = val
Literal* literal_create_scalar(double val);

// Einstein summation: generalized tensor contraction
// left: Left operand literal (must not be NULL)
// left_indices: Index string for left operand (e.g., "ij")
// right: Right operand literal (can be NULL for unary operations)
// right_indices: Index string for right operand (e.g., "jk")
// out_indices: Index string for result (e.g., "ik")
// success: Set to true if operation succeeds, false otherwise
// Returns: New Literal with result (caller owns, must call literal_free)
// Memory: Allocates new Literal, caller must free
// Examples:
//   literal_einsum(A, "ij", B, "jk", "ik", &s)  // Matrix multiply
//   literal_einsum(A, "ii", NULL, "", "", &s)   // Trace
//   literal_einsum(A, "ij", NULL, "", "ji", &s) // Transpose
Literal* literal_einsum(Literal *left, const char *left_indices,
                       Literal *right, const char *right_indices,
                       const char *out_indices, bool *success);

// Arithmetic operations for literals
Literal* literal_add(Literal *left, Literal *right);
Literal* literal_subtract(Literal *left, Literal *right);
Literal* literal_multiply(Literal *left, Literal *right);
Literal* literal_scale(Literal *lit, double scalar);
double literal_norm(Literal *lit);

// Print a literal in human-readable form (vector, matrix, tensor)
void literal_print(const Literal *lit);

// Additional helpers for use in expression.c and elsewhere
Literal* literal_negate(Literal *lit);
Literal* literal_matmul(Literal *left, Literal *right);
Literal* literal_dot(Literal *left, Literal *right);
Literal* literal_transpose(Literal *lit, bool *success);

// Broadcast a literal to a target shape (returns new literal, caller must free)
// Returns NULL if broadcasting is not possible
Literal* literal_broadcast_to_shape(const Literal *src, const uint32_t *target_shape);

#endif // LITERAL_H
