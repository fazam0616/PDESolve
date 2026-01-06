#include "../include/expression.h"
#include "../include/literal.h"
#include <stdio.h>
#include <stdbool.h>

int main() {
    printf("Debug: Testing dimension mapping\n\n");
    
    // Create simple 2x2 matrix with shape [1, 2, 2]
    Literal *A = literal_create((uint32_t[]){1, 2, 2});
    A->field[0] = 1; A->field[1] = 2;
    A->field[2] = 3; A->field[3] = 4;
    
    printf("Matrix A shape: [%u, %u, %u]\n", A->shape[0], A->shape[1], A->shape[2]);
    printf("A values: [%.0f, %.0f, %.0f, %.0f]\n\n", 
           A->field[0], A->field[1], A->field[2], A->field[3]);
    
    // For indices "ij" (2 chars), get_dim_size should map to dims [1] and [2]
    // start_dim = N_DIM - 2 = 3 - 2 = 1
    // pos=0 -> dim=1 -> shape[1] = 2
    // pos=1 -> dim=2 -> shape[2] = 2
    
    int total_indices = 2;
    int start_dim = N_DIM - total_indices;
    
    printf("For 2 indices in N_DIM=3:\n");
    printf("  start_dim = %d\n", start_dim);
    printf("  Index 0 maps to dim %d -> size %u\n", start_dim + 0, A->shape[start_dim + 0]);
    printf("  Index 1 maps to dim %d -> size %u\n", start_dim + 1, A->shape[start_dim + 1]);
    
    printf("\nThis should give us 2x2 for matrix multiply\n");
    
    literal_free(A);
    return 0;
}
