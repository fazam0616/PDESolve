#include "../include/expression.h"
#include "../include/literal.h"
#include <stdio.h>
#include <stdbool.h>

int main() {
    printf("Testing literal_einsum directly\n");
    
    // Create simple 2x2 matrices
    Literal *A = literal_create((uint32_t[]){1, 2, 2});
    A->field[0] = 1; A->field[1] = 2;
    A->field[2] = 3; A->field[3] = 4;
    
    Literal *B = literal_create((uint32_t[]){1, 2, 2});
    B->field[0] = 5; B->field[1] = 6;
    B->field[2] = 7; B->field[3] = 8;
    
    printf("A = [1 2; 3 4]\n");
    printf("B = [5 6; 7 8]\n");
    printf("Expected A@B = [19 22; 43 50]\n\n");
    
    bool success = false;
    Literal *result = literal_einsum(A, "ij", B, "jk", "ik", &success);
    
    if (result == NULL) {
        printf("ERROR: literal_einsum returned NULL\n");
        printf("success = %d\n", success);
    } else {
        printf("SUCCESS!\n");
        printf("Result shape: [%u, %u, %u]\n", result->shape[0], result->shape[1], result->shape[2]);
        printf("Result values: [%.0f %.0f; %.0f %.0f]\n",
               result->field[0], result->field[1],
               result->field[2], result->field[3]);
        literal_free(result);
    }
    
    literal_free(A);
    literal_free(B);
    
    return 0;
}
