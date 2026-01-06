
#include "../include/expression.h"
#include "../include/literal.h"
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

int main() {
    printf("Testing literal_einsum matrix multiply fast path\n\n");
    
    // Create 2x2 matrices [1, 2, 2] shape
    Literal *A = literal_create((uint32_t[]){1, 2, 2});
    Literal *B = literal_create((uint32_t[]){1, 2, 2});
    
    A->field[0] = 1; A->field[1] = 2;
    A->field[2] = 3; A->field[3] = 4;
    
    B->field[0] = 5; B->field[1] = 6;
    B->field[2] = 7; B->field[3] = 8;
    
    printf("A = [1 2; 3 4]\n");
    printf("B = [5 6; 7 8]\n");
    printf("Expected A@B = [19 22; 43 50]\n\n");
    
    // Check fast path conditions
    const char *left_idx = "ij";
    const char *right_idx = "jk";
    const char *out_idx = "ik";
    
    printf("Checking fast path conditions for matrix multiply:\n");
    printf("  strlen(left_indices)==2: %s\n", strlen(left_idx)==2 ? "YES" : "NO");
    printf("  strlen(right_indices)==2: %s\n", strlen(right_idx)==2 ? "YES" : "NO");
    printf("  strlen(out_indices)==2: %s\n", strlen(out_idx)==2 ? "YES" : "NO");
    printf("  left_indices[1]=='%c' == right_indices[0]=='%c': %s\n", 
           left_idx[1], right_idx[0], left_idx[1]==right_idx[0] ? "YES" : "NO");
    printf("  left_indices[0]=='%c' == out_indices[0]=='%c': %s\n",
           left_idx[0], out_idx[0], left_idx[0]==out_idx[0] ? "YES" : "NO");
    printf("  right_indices[1]=='%c' == out_indices[1]=='%c': %s\n",
           right_idx[1], out_idx[1], right_idx[1]==out_idx[1] ? "YES" : "NO");
    
    bool should_hit_fast_path = 
        (strlen(left_idx)==2 && strlen(right_idx)==2 && strlen(out_idx)==2 &&
         left_idx[1]==right_idx[0] && left_idx[0]==out_idx[0] && right_idx[1]==out_idx[1]);
    
    printf("\nShould hit fast path: %s\n\n", should_hit_fast_path ? "YES" : "NO");
    
    bool success = false;
    Literal *result = literal_einsum(A, left_idx, B, right_idx, out_idx, &success);
    
    if (result == NULL) {
        printf("ERROR: literal_einsum returned NULL\n");
        printf("success = %d\n", success);
    } else {
        printf("Result shape: [%u, %u, %u]\n", result->shape[0], result->shape[1], result->shape[2]);
        printf("Result values:\n");
        printf("[%.0f %.0f]\n", result->field[0], result->field[1]);
        printf("[%.0f %.0f]\n", result->field[2], result->field[3]);
        printf("success = %d\n", success);
        literal_free(result);
    }
    
    literal_free(A);
    literal_free(B);
    return 0;
}
