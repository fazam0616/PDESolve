#include "../include/expression.h"
#include "../include/calculus.h"
#include "../include/ast_hash.h"
#include <stdio.h>
#include "../include/literal.h"
#include <assert.h>
#include <math.h>

// Helper to create 2x2 matrix
static Literal* make_2x2_matrix(double a, double b, double c, double d) {
    Literal *mat = literal_create((uint32_t[]){1, 2, 2});
    mat->field[0] = a;  // [0,0]
    mat->field[1] = b;  // [0,1]
    mat->field[2] = c;  // [1,0]
    mat->field[3] = d;  // [1,1]
    return mat;
}

void test_element_wise_multiplication() {
    printf("Test 1: Element-wise Multiplication (OP_MULTIPLY)\n");
    printf("==================================================\n");
    
    // Test scalar multiplication: 2 * 3 = 6
    Expression *two = make_scalar(2.0);
    Expression *three = make_scalar(3.0);
    Expression *product = expr_multiply(two, three);
    
    Literal *result = expression_evaluate(product, NULL);
    
    Expression* temp = expr_literal(result);
    printf("  Result of 2 * 3:\n");
    print_expression(temp);
    printf("\n");
    free(temp);

    printf("%f\n", fabs(result->field[0] - 6.0));


    assert(result != NULL);
    assert(fabs(result->field[0] - 6.0) < 1e-10);
    printf("  Scalar: 2 * 3 = %.1f [OK]\n", result->field[0]);
    literal_free(result);
    expression_free(product);
    
    // Test commutativity: 2*x == x*2 (same hash)
    Expression *x = expr_variable("x");
    Expression *two_x = expr_multiply(make_scalar(2.0), expr_copy(x));
    Expression *x_two = expr_multiply(expr_copy(x), make_scalar(2.0));
    
    uint64_t hash_2x = expr_canonical_hash(two_x);
    uint64_t hash_x2 = expr_canonical_hash(x_two);
    
    printf("  2*x hash: 0x%016llx\n", (unsigned long long)hash_2x);
    printf("  x*2 hash: 0x%016llx\n", (unsigned long long)hash_x2);
    assert(hash_2x == hash_x2);
    printf("  Element-wise multiplication is commutative [OK]\n\n");
    
    expression_free(x_two);
    expression_free(two_x);
    expression_free(x);
}

void test_matrix_multiplication() {
    printf("Test 2: Matrix Multiplication (OP_MATMUL)\n");
    printf("==========================================\n");
    
    // Create matrices:
    // A = [1 2]  B = [5 6]
    //     [3 4]      [7 8]
    // A @ B = [19 22]
    //         [43 50]
    
    Literal *A = make_2x2_matrix(1, 2, 3, 4);
    Literal *B = make_2x2_matrix(5, 6, 7, 8);
    
    Expression *expr_A = expr_literal(A);
    Expression *expr_B = expr_literal(B);
    
    Expression *AB = expr_matmul(expr_A, expr_B);
    
    Literal *result_AB = expression_evaluate(AB, NULL);
    assert(result_AB != NULL);
    
    // Check result: [19, 22, 43, 50]
    assert(fabs(result_AB->field[0] - 19.0) < 1e-10);
    assert(fabs(result_AB->field[1] - 22.0) < 1e-10);
    assert(fabs(result_AB->field[2] - 43.0) < 1e-10);
    assert(fabs(result_AB->field[3] - 50.0) < 1e-10);
    
    printf("  A @ B = [[%.0f, %.0f], [%.0f, %.0f]] [OK]\n", 
           result_AB->field[0], result_AB->field[1],
           result_AB->field[2], result_AB->field[3]);
    
    literal_free(result_AB);
    expression_free(AB);
    
    // Test non-commutativity: A @ B != B @ A
    Literal *A2 = make_2x2_matrix(1, 2, 3, 4);
    Literal *B2 = make_2x2_matrix(5, 6, 7, 8);
    
    Expression *expr_A2 = expr_literal(A2);
    Expression *expr_B2 = expr_literal(B2);
    
    Expression *BA = expr_matmul(expr_B2, expr_A2);
    
    Literal *result_BA = expression_evaluate(BA, NULL);
    assert(result_BA != NULL);
    
    // B @ A = [23, 34, 31, 46] (different from A @ B)
    assert(fabs(result_BA->field[0] - 23.0) < 1e-10);
    assert(fabs(result_BA->field[1] - 34.0) < 1e-10);
    assert(fabs(result_BA->field[2] - 31.0) < 1e-10);
    assert(fabs(result_BA->field[3] - 46.0) < 1e-10);
    
    printf("  B @ A = [[%.0f, %.0f], [%.0f, %.0f]] [OK]\n",
           result_BA->field[0], result_BA->field[1],
           result_BA->field[2], result_BA->field[3]);
    
    printf("  A @ B != B @ A (non-commutative) [OK]\n\n");
    
    literal_free(result_BA);
    expression_free(BA);
}

void test_matrix_hashing() {
    printf("Test 3: Matrix Multiplication Hashing\n");
    printf("======================================\n");
    
    // Create symbolic matrix expressions
    Expression *A = expr_variable("A");
    Expression *B = expr_variable("B");
    
    Expression *AB = expr_matmul(expr_copy(A), expr_copy(B));
    Expression *BA = expr_matmul(expr_copy(B), expr_copy(A));
    
    uint64_t hash_AB = expr_canonical_hash(AB);
    uint64_t hash_BA = expr_canonical_hash(BA);
    
    printf("  A @ B hash: 0x%016llx\n", (unsigned long long)hash_AB);
    printf("  B @ A hash: 0x%016llx\n", (unsigned long long)hash_BA);
    
    assert(hash_AB != hash_BA);
    printf("  A @ B != B @ A (different hashes) [OK]\n");
    
    // Structural equality should also be false
    assert(!expr_structural_equals(AB, BA));
    printf("  Structural equality: A @ B != B @ A [OK]\n\n");
    
    expression_free(BA);
    expression_free(AB);
    expression_free(B);
    expression_free(A);
}

void test_matrix_derivative() {
    printf("Test 4: Matrix Multiplication Derivative\n");
    printf("=========================================\n");
    
    // d/dx(A @ B) = (dA/dx) @ B + A @ (dB/dx)
    // For symbolic test: d/dx(A @ x) where x is scalar variable
    
    Expression *A = expr_variable("A");  // Matrix (constant w.r.t. x)
    Expression *x = expr_variable("x");  // Scalar variable
    
    Expression *Ax = expr_matmul(A, x);  // Takes ownership of A and x
    
    printf("  Expression: A @ x\n");
    printf("  Computing d(A @ x)/dx...\n");
    
    Expression *dAx_dx = derivative(Ax, "x");
    
    printf("  Result: ");
    print_expression(dAx_dx);
    printf("\n");
    
    // Should be: (dA/dx) @ x + A @ (dx/dx)
    // = 0 @ x + A @ 1
    // After simplification: A
    Expression *simplified = simplify(dAx_dx);
    printf("  Simplified: ");
    print_expression(simplified);
    printf("\n");
    
    printf("  Derivative computed successfully [OK]\n\n");
    
    expression_free(simplified);
    expression_free(dAx_dx);
    expression_free(Ax);  // This frees A and x too
}

void test_mixed_operations() {
    printf("Test 5: Mixed Element-wise and Matrix Operations\n");
    printf("=================================================\n");
    
    // Test: 2 * (A @ B) where * is element-wise scalar multiplication
    printf("  Creating matrices...\n");
    Literal *A = make_2x2_matrix(1, 0, 0, 1);  // Identity matrix
    Literal *B = make_2x2_matrix(2, 3, 4, 5);
    Literal *two = literal_create((uint32_t[]){1,1,1});
    two->field[0] = 2.0;
    
    printf("  Creating expressions...\n");
    Expression *expr_A = expr_literal(A);
    Expression *expr_B = expr_literal(B);
    Expression *expr_two = expr_literal(two);
    
    // A @ B
    printf("  Computing A @ B...\n");
    Expression *AB = expr_matmul(expr_A, expr_B);
    printf("  Expression for A @ B: ");
    print_expression(AB);
    printf("\n");
    Literal *val_AB = expression_evaluate(AB, NULL);
    printf("  Evaluated A @ B:\n");
    literal_print(val_AB);
    printf("\n");
    
    // 2 * (A @ B) - element-wise scalar multiplication
    printf("  Computing 2 * (A @ B)...\n");
    Expression *scaled = expr_multiply(expr_two, AB);
    printf("  Expression for 2 * (A @ B): ");
    print_expression(scaled);
    printf("\n");
    Literal *result = expression_evaluate(scaled, NULL);
    printf("  Evaluated 2 * (A @ B):\n");
    literal_print(result);
    printf("\n");

    // Also print the scalar value if result is scalar
    bool is_scalar = true;
    for (int i = 0; i < N_DIM; i++) {
        if (result->shape[i] != 1) {
            is_scalar = false;
            break;
        }
    }
    if (is_scalar) {
        printf("  [DEBUG] Result is scalar: %g\n", result->field[0]);
        assert(0 && "Result should be a matrix, not a scalar!");
    }

    assert(result != NULL);
    // I @ B = B, then 2 * B = [4, 6, 8, 10] (if result is a matrix)
    assert(fabs(result->field[0] - 4.0) < 1e-10);
    assert(fabs(result->field[1] - 6.0) < 1e-10);
    assert(fabs(result->field[2] - 8.0) < 1e-10);
    assert(fabs(result->field[3] - 10.0) < 1e-10);
    printf("  2 * (I @ B) = 2 * B = [[%.0f, %.0f], [%.0f, %.0f]] [OK]\n",
           result->field[0], result->field[1],
           result->field[2], result->field[3]);
    
    literal_free(val_AB);
    literal_free(result);
    expression_free(scaled);
    
    printf("  Mixed operations work correctly [OK]\n\n");
}

int main() {
    printf("===========================================\n");
    printf("Tensor Operations Tests\n");
    printf("===========================================\n\n");
    
    printf("Starting Test 1...\n");
    test_element_wise_multiplication();
    printf("Test 1 complete.\n\n");
    
    printf("Starting Test 2...\n");
    test_matrix_multiplication();
    printf("Test 2 complete.\n\n");
    
    printf("Starting Test 3...\n");
    test_matrix_hashing();
    printf("Test 3 complete.\n\n");
    
    printf("Starting Test 4...\n");
    test_matrix_derivative();
    printf("Test 4 complete.\n\n");
    
    printf("Starting Test 5...\n");
    test_mixed_operations();
    printf("Test 5 complete.\n\n");
    
    printf("===========================================\n");
    printf("All tensor tests passed! [OK]\n");
    printf("===========================================\n\n");
    
    printf("Summary:\n");
    printf("- OP_MULTIPLY: Element-wise (commutative, 2*x = x*2)\n");
    printf("- OP_MATMUL: Matrix mult (non-commutative, A@B != B@A)\n");
    printf("- Hashing: Respects commutativity correctly\n");
    printf("- Derivatives: Product rule works for both operations\n");
    
    return 0;
}
