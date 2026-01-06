#include "../include/expression.h"
#include "../include/literal.h"
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

// Copy from literal.c for testing
typedef struct {
    char index;
    uint32_t size;
    int left_pos;
    int right_pos;
    int out_pos;
} IndexDim;

static uint32_t get_dim_size(Literal *lit, int pos, int total_indices) {
    int start_dim = N_DIM - total_indices;
    if (start_dim < 0) start_dim = 0;
    return lit->shape[start_dim + pos];
}

static bool validate_and_map_indices(
    Literal *left, const char *left_indices,
    Literal *right, const char *right_indices,
    const char *out_indices,
    IndexDim dims[26], int *n_unique_indices)
{
    int left_len = strlen(left_indices);
    int right_len = right_indices ? strlen(right_indices) : 0;
    int unique_count = 0;
    
    printf("  left_len=%d, right_len=%d\n", left_len, right_len);
    
    for (int i = 0; i < 26; i++) {
        dims[i].index = 0;
        dims[i].size = 0;
        dims[i].left_pos = -1;
        dims[i].right_pos = -1;
        dims[i].out_pos = -1;
    }
    
    for (int i = 0; i < left_len; i++) {
        char idx = left_indices[i];
        if (idx < 'a' || idx > 'z') continue;
        
        int idx_num = idx - 'a';
        uint32_t dim_size = get_dim_size(left, i, left_len);
        
        printf("  Left index %d: '%c' -> size %u\n", i, idx, dim_size);
        
        if (dims[idx_num].index == 0) {
            dims[idx_num].index = idx;
            dims[idx_num].size = dim_size;
            dims[idx_num].left_pos = i;
            unique_count++;
        } else {
            if (dims[idx_num].size != dim_size) {
                printf("  ERROR: Repeated index mismatch\n");
                return false;
            }
        }
    }
    
    if (right != NULL && right_indices != NULL) {
        for (int i = 0; i < right_len; i++) {
            char idx = right_indices[i];
            if (idx < 'a' || idx > 'z') continue;
            
            int idx_num = idx - 'a';
            uint32_t dim_size = get_dim_size(right, i, right_len);
            
            printf("  Right index %d: '%c' -> size %u\n", i, idx, dim_size);
            
            if (dims[idx_num].index == 0) {
                dims[idx_num].index = idx;
                dims[idx_num].size = dim_size;
                dims[idx_num].right_pos = i;
                unique_count++;
            } else {
                dims[idx_num].right_pos = i;
                if (dims[idx_num].size != dim_size) {
                    printf("  ERROR: Contraction dimension mismatch: expected %u, got %u\n",
                           dims[idx_num].size, dim_size);
                    return false;
                }
            }
        }
    }
    
    int out_len = strlen(out_indices);
    for (int i = 0; i < out_len; i++) {
        char idx = out_indices[i];
        if (idx < 'a' || idx > 'z') continue;
        
        int idx_num = idx - 'a';
        if (dims[idx_num].index == 0) {
            printf("  ERROR: Output index '%c' not in inputs\n", idx);
            return false;
        }
        dims[idx_num].out_pos = i;
    }
    
    *n_unique_indices = unique_count;
    printf("  Validation passed! unique_count=%d\n", unique_count);
    return true;
}

int main() {
    printf("Testing validation with 2x2 matrix multiply\n\n");
    
    Literal *A = literal_create((uint32_t[]){1, 2, 2});
    Literal *B = literal_create((uint32_t[]){1, 2, 2});
    
    printf("A shape: [%u, %u, %u]\n", A->shape[0], A->shape[1], A->shape[2]);
    printf("B shape: [%u, %u, %u]\n\n", B->shape[0], B->shape[1], B->shape[2]);
    
    IndexDim dims[26];
    int n_unique = 0;
    
    printf("Testing 'ij','jk'->'ik':\n");
    bool result = validate_and_map_indices(A, "ij", B, "jk", "ik", dims, &n_unique);
    
    if (result) {
        printf("\n✓ Validation succeeded\n");
        printf("Index mapping:\n");
        for (int i = 0; i < 26; i++) {
            if (dims[i].index != 0) {
                printf("  '%c': size=%u, left=%d, right=%d, out=%d\n",
                       dims[i].index, dims[i].size, 
                       dims[i].left_pos, dims[i].right_pos, dims[i].out_pos);
            }
        }
    } else {
        printf("\n✗ Validation FAILED\n");
    }
    
    literal_free(A);
    literal_free(B);
    return 0;
}
