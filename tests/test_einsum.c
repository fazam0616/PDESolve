#include "../include/expression.h"
#include "../include/calculus.h"
#include "../include/ast_hash.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

// ============================================================================
// Test Utilities
// ============================================================================

// Helper to create 2x2 matrix
static Literal* make_2x2_matrix(double a, double b, double c, double d) {
    Literal *mat = literal_create((uint32_t[]){1, 2, 2});
    literal_set(mat, (uint32_t[]){0, 0, 0}, a);
    literal_set(mat, (uint32_t[]){0, 0, 1}, b);
    literal_set(mat, (uint32_t[]){0, 1, 0}, c);
    literal_set(mat, (uint32_t[]){0, 1, 1}, d);
    return mat;
}

// Helper to create 3x3 matrix
static Literal* make_3x3_matrix(double a00, double a01, double a02,
                                double a10, double a11, double a12,
                                double a20, double a21, double a22) {
    Literal *mat = literal_create((uint32_t[]){1, 3, 3});
    literal_set(mat, (uint32_t[]){0, 0, 0}, a00);
    literal_set(mat, (uint32_t[]){0, 0, 1}, a01);
    literal_set(mat, (uint32_t[]){0, 0, 2}, a02);
    literal_set(mat, (uint32_t[]){0, 1, 0}, a10);
    literal_set(mat, (uint32_t[]){0, 1, 1}, a11);
    literal_set(mat, (uint32_t[]){0, 1, 2}, a12);
    literal_set(mat, (uint32_t[]){0, 2, 0}, a20);
    literal_set(mat, (uint32_t[]){0, 2, 1}, a21);
    literal_set(mat, (uint32_t[]){0, 2, 2}, a22);
    return mat;
}

// Helper to create vector
// For einsum "i", the data should be in the last dimension (N_DIM-1)
static Literal* make_vector(int size, double *values) {
    Literal *vec = literal_create((uint32_t[]){1, 1, size});
    for (int i = 0; i < size; i++) {
        literal_set(vec, (uint32_t[]){0, 0, i}, values[i]);
    }
    return vec;
}

// ============================================================================
// Test 1: Basic Einstein Summation - Matrix Multiply
// ============================================================================

void test_einsum_matmul() {
    printf("Test 1: Einstein Summation - Matrix Multiply (ij,jk->ik)\n");
    printf("===========================================================\n");
    
    // Create matrices:
    // A = [1 2]  B = [5 6]
    //     [3 4]      [7 8]
    // A @ B = [19 22]
    //         [43 50]
    
    Literal *A = make_2x2_matrix(1, 2, 3, 4);
    Literal *B = make_2x2_matrix(5, 6, 7, 8);
    
    Expression *expr_A = expr_literal(A);
    Expression *expr_B = expr_literal(B);
    
    // Use Einstein notation: C_ik = A_ij B_jk
    Expression *AB = expr_einsum(expr_A, "ij", expr_B, "jk", "ik");
    
    printf("  Expression: einsum[ij,jk->ik](A, B)\n");
    printf("  Expected: [19, 22, 43, 50]\n");
    
    Literal *result = expression_evaluate(AB, NULL);
    assert(result != NULL);
    
    // Check result
    double r00 = literal_get(result, (uint32_t[]){0, 0, 0});
    double r01 = literal_get(result, (uint32_t[]){0, 0, 1});
    double r10 = literal_get(result, (uint32_t[]){0, 1, 0});
    double r11 = literal_get(result, (uint32_t[]){0, 1, 1});
    assert(fabs(r00 - 19.0) < 1e-10);
    assert(fabs(r01 - 22.0) < 1e-10);
    assert(fabs(r10 - 43.0) < 1e-10);
    assert(fabs(r11 - 50.0) < 1e-10);
    
    printf("  Result: [%.0f, %.0f, %.0f, %.0f] [OK]\n\n",
           r00, r01, r10, r11);
    
    literal_free(result);
    expression_free(AB);
}

// ============================================================================
// Test 2: Trace - ii->
// ============================================================================

void test_einsum_trace() {
    printf("Test 2: Einstein Summation - Trace (ii->)\n");
    printf("==========================================\n");
    
    // Create matrix:
    // A = [1 2]  Trace = 1 + 4 = 5
    //     [3 4]
    
    Literal *A = make_2x2_matrix(1, 2, 3, 4);
    Expression *expr_A = expr_literal(A);
    
    // Trace: scalar = sum_i A_ii
    Expression *trace = expr_einsum(expr_A, "ii", NULL, "", "");
    
    printf("  Expression: einsum[ii->](A)\n");
    printf("  Expected: 5.0\n");
    
    Literal *result = expression_evaluate(trace, NULL);
    assert(result != NULL);
    double r = literal_get(result, (uint32_t[]){0, 0, 0});
    assert(fabs(r - 5.0) < 1e-10);
    
    printf("  Result: %.0f [OK]\n\n", r);
    
    literal_free(result);
    expression_free(trace);
}

// ============================================================================
// Test 3: Transpose - ij->ji
// ============================================================================

void test_einsum_transpose() {
    printf("Test 3: Einstein Summation - Transpose (ij->ji)\n");
    printf("================================================\n");
    
    // Create matrix:
    // A = [1 2]  A^T = [1 3]
    //     [3 4]        [2 4]
    
    Literal *A = make_2x2_matrix(1, 2, 3, 4);
    Expression *expr_A = expr_literal(A);
    
    // Transpose: B_ji = A_ij
    Expression *transpose = expr_einsum(expr_A, "ij", NULL, "", "ji");
    
    printf("  Expression: einsum[ij->ji](A)\n");
    printf("  Expected: [1, 3, 2, 4]\n");
    
    Literal *result = expression_evaluate(transpose, NULL);
    assert(result != NULL);
    
    double r00 = literal_get(result, (uint32_t[]){0, 0, 0});
    double r01 = literal_get(result, (uint32_t[]){0, 0, 1});
    double r10 = literal_get(result, (uint32_t[]){0, 1, 0});
    double r11 = literal_get(result, (uint32_t[]){0, 1, 1});
    assert(fabs(r00 - 1.0) < 1e-10);
    assert(fabs(r01 - 3.0) < 1e-10);
    assert(fabs(r10 - 2.0) < 1e-10);
    assert(fabs(r11 - 4.0) < 1e-10);
    
    printf("  Result: [%.0f, %.0f, %.0f, %.0f] [OK]\n\n",
           r00, r01, r10, r11);
    
    literal_free(result);
    expression_free(transpose);
}

// ============================================================================
// Test 4: Outer Product - i,j->ij
// ============================================================================

void test_einsum_outer_product() {
    printf("Test 4: Einstein Summation - Outer Product (i,j->ij)\n");
    printf("======================================================\n");
    
    // Create vectors:
    // a = [1, 2, 3]  b = [4, 5]
    // Outer product: C_ij = a_i * b_j
    // Result = [[4, 5], [8, 10], [12, 15]]
    
    double a_vals[] = {1, 2, 3};
    double b_vals[] = {4, 5};
    
    Literal *a = make_vector(3, a_vals);
    Literal *b = make_vector(2, b_vals);
    
    Expression *expr_a = expr_literal(a);
    Expression *expr_b = expr_literal(b);
    
    // Outer product: C_ij = a_i b_j
    Expression *outer = expr_einsum(expr_a, "i", expr_b, "j", "ij");
    
    printf("  Expression: einsum[i,j->ij](a, b)\n");
    printf("  Expected: [[4, 5], [8, 10], [12, 15]]\n");
    
    Literal *result = expression_evaluate(outer, NULL);
    assert(result != NULL);
    
    // Check 3x2 result
    double r00 = literal_get(result, (uint32_t[]){0, 0, 0});
    double r01 = literal_get(result, (uint32_t[]){0, 0, 1});
    double r10 = literal_get(result, (uint32_t[]){0, 1, 0});
    double r11 = literal_get(result, (uint32_t[]){0, 1, 1});
    double r20 = literal_get(result, (uint32_t[]){0, 2, 0});
    double r21 = literal_get(result, (uint32_t[]){0, 2, 1});
    assert(fabs(r00 - 4.0) < 1e-10);
    assert(fabs(r01 - 5.0) < 1e-10);
    assert(fabs(r10 - 8.0) < 1e-10);
    assert(fabs(r11 - 10.0) < 1e-10);
    assert(fabs(r20 - 12.0) < 1e-10);
    assert(fabs(r21 - 15.0) < 1e-10);
    
    printf("  Result: [[%.0f, %.0f], [%.0f, %.0f], [%.0f, %.0f]] [OK]\n\n",
           r00, r01, r10, r11, r20, r21);
    
    literal_free(result);
    expression_free(outer);
}

// ============================================================================
// Test 5: Dot Product - i,i->
// ============================================================================

void test_einsum_dot_product() {
    printf("Test 5: Einstein Summation - Dot Product (i,i->)\n");
    printf("=================================================\n");
    
    // Create vectors:
    // a = [1, 2, 3]  b = [4, 5, 6]
    // Dot product = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    
    double a_vals[] = {1, 2, 3};
    double b_vals[] = {4, 5, 6};
    
    Literal *a = make_vector(3, a_vals);
    Literal *b = make_vector(3, b_vals);
    
    Expression *expr_a = expr_literal(a);
    Expression *expr_b = expr_literal(b);
    
    // Dot product: scalar = sum_i a_i * b_i
    Expression *dot = expr_einsum(expr_a, "i", expr_b, "i", "");
    
    printf("  Expression: einsum[i,i->](a, b)\n");
    printf("  Expected: 32.0\n");
    
    Literal *result = expression_evaluate(dot, NULL);
    assert(result != NULL);
    double r = literal_get(result, (uint32_t[]){0, 0, 0});
    assert(fabs(r - 32.0) < 1e-10);
    
    printf("  Result: %.0f [OK]\n\n", r);
    
    literal_free(result);
    expression_free(dot);
}

// ============================================================================
// Test 6: Element-wise Multiply - ij,ij->ij
// ============================================================================

void test_einsum_elementwise() {
    printf("Test 6: Einstein Summation - Element-wise (ij,ij->ij)\n");
    printf("======================================================\n");
    
    // Create matrices:
    // A = [1 2]  B = [5 6]  A*B = [5  12]
    //     [3 4]      [7 8]        [21 32]
    
    Literal *A = make_2x2_matrix(1, 2, 3, 4);
    Literal *B = make_2x2_matrix(5, 6, 7, 8);
    
    Expression *expr_A = expr_literal(A);
    Expression *expr_B = expr_literal(B);
    
    // Element-wise multiply: C_ij = A_ij * B_ij
    Expression *elem = expr_einsum(expr_A, "ij", expr_B, "ij", "ij");
    
    printf("  Expression: einsum[ij,ij->ij](A, B)\n");
    printf("  Expected: [5, 12, 21, 32]\n");
    
    Literal *result = expression_evaluate(elem, NULL);
    assert(result != NULL);
    
    double r00 = literal_get(result, (uint32_t[]){0, 0, 0});
    double r01 = literal_get(result, (uint32_t[]){0, 0, 1});
    double r10 = literal_get(result, (uint32_t[]){0, 1, 0});
    double r11 = literal_get(result, (uint32_t[]){0, 1, 1});
    assert(fabs(r00 - 5.0) < 1e-10);
    assert(fabs(r01 - 12.0) < 1e-10);
    assert(fabs(r10 - 21.0) < 1e-10);
    assert(fabs(r11 - 32.0) < 1e-10);
    
    printf("  Result: [%.0f, %.0f, %.0f, %.0f] [OK]\n\n",
           r00, r01, r10, r11);
    
    literal_free(result);
    expression_free(elem);
}

// ============================================================================
// Test 7: Hash Uniqueness - Different Index Patterns
// ============================================================================

void test_einsum_hash_uniqueness() {
    printf("Test 7: Hash Uniqueness - Different Index Patterns\n");
    printf("====================================================\n");
    
    Literal *A = make_2x2_matrix(1, 2, 3, 4);
    Literal *B = make_2x2_matrix(5, 6, 7, 8);
    
    // Create different Einstein operations
    Expression *matmul_ij_jk = expr_einsum(expr_literal(literal_copy(A)), "ij",
                                          expr_literal(literal_copy(B)), "jk", "ik");
    Expression *matmul_ik_kj = expr_einsum(expr_literal(literal_copy(A)), "ik",
                                          expr_literal(literal_copy(B)), "kj", "ij");
    Expression *outer_i_j = expr_einsum(expr_literal(literal_copy(A)), "i",
                                       expr_literal(literal_copy(B)), "j", "ij");
    Expression *trace = expr_einsum(expr_literal(literal_copy(A)), "ii",
                                   NULL, "", "");
    
    uint64_t hash1 = expr_canonical_hash(matmul_ij_jk);
    uint64_t hash2 = expr_canonical_hash(matmul_ik_kj);
    uint64_t hash3 = expr_canonical_hash(outer_i_j);
    uint64_t hash4 = expr_canonical_hash(trace);
    
    printf("  einsum[ij,jk->ik] hash: 0x%016llx\n", (unsigned long long)hash1);
    printf("  einsum[ik,kj->ij] hash: 0x%016llx\n", (unsigned long long)hash2);
    printf("  einsum[i,j->ij]   hash: 0x%016llx\n", (unsigned long long)hash3);
    printf("  einsum[ii->]      hash: 0x%016llx\n", (unsigned long long)hash4);
    
    // All hashes should be different
    assert(hash1 != hash2);
    assert(hash1 != hash3);
    assert(hash1 != hash4);
    assert(hash2 != hash3);
    assert(hash2 != hash4);
    assert(hash3 != hash4);
    
    printf("  All hashes are unique [OK]\n\n");
    
    literal_free(A);
    literal_free(B);
    expression_free(matmul_ij_jk);
    expression_free(matmul_ik_kj);
    expression_free(outer_i_j);
    expression_free(trace);
}

// ============================================================================
// Test 8: Differentiation of Einstein Operations
// ============================================================================

void test_einsum_derivative() {
    printf("Test 8: Differentiation of Einstein Operations\n");
    printf("================================================\n");
    
    // Test: d/dx(A_ij B_jk) where A depends on x
    // Result: (dA_ij/dx) B_jk (since B is constant)
    
    Expression *A = expr_variable("A");
    Literal *B_lit = make_2x2_matrix(1, 2, 3, 4);
    Expression *B = expr_literal(B_lit);
    
    // C = einsum[ij,jk->ik](A, B)
    Expression *C = expr_einsum(A, "ij", B, "jk", "ik");
    
    // dC/dA
    Expression *dC_dA = derivative(C, "A");
    
    printf("  Expression: d/dA(einsum[ij,jk->ik](A, B))\n");
    printf("  Expected: einsum[ij,jk->ik](d/dA(A), B) + einsum[ij,jk->ik](A, d/dA(B))\n");
    printf("  Since B is constant, second term is zero\n");
    
    // Check that derivative is not NULL
    assert(dC_dA != NULL);
    assert(dC_dA->type == EXPR_BINARY);
    assert(dC_dA->data.binary.op == OP_ADD);
    
    printf("  Derivative computed successfully [OK]\n\n");
    
    expression_free(dC_dA);
    expression_free(C);
}

// ============================================================================
// Test 9: Simplification of Einstein Operations
// ============================================================================

void test_einsum_simplification() {
    printf("Test 9: Simplification of Einstein Operations\n");
    printf("==============================================\n");
    
    // Test: 0 * A in Einstein notation should simplify to 0
    Expression *zero = make_scalar(0.0);
    Literal *A_lit = make_2x2_matrix(1, 2, 3, 4);
    Expression *A = expr_literal(A_lit);
    
    // einsum[ij,jk->ik](0, A) should simplify to 0
    Expression *result = expr_einsum(zero, "ij", A, "jk", "ik");
    Expression *simplified = simplify(result);
    
    printf("  Expression: einsum[ij,jk->ik](0, A)\n");
    printf("  Expected: 0 (after simplification)\n");
    
    // Check that simplified expression is zero literal
    assert(simplified != NULL);
    assert(simplified->type == EXPR_LITERAL);
    double r = literal_get(simplified->data.literal, (uint32_t[]){0, 0, 0});
    assert(fabs(r - 0.0) < 1e-10);
    
    printf("  Simplified to 0 [OK]\n\n");
    
    expression_free(simplified);
    expression_free(result);
}

// ============================================================================
// Test 10: Expression Printing
// ============================================================================

void test_einsum_printing() {
    printf("Test 10: Expression Printing\n");
    printf("=============================\n");
    
    Literal *A = make_2x2_matrix(1, 2, 3, 4);
    Literal *B = make_2x2_matrix(5, 6, 7, 8);
    
    Expression *expr_A = expr_literal(A);
    Expression *expr_B = expr_literal(B);
    
    Expression *matmul = expr_einsum(expr_A, "ij", expr_B, "jk", "ik");
    
    printf("  Expression: ");
    print_expression(matmul);
    printf("\n");
    printf("  Expected: einsum[ij,jk->ik](..., ...)\n");
    printf("  [OK]\n\n");
    
    expression_free(matmul);
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    printf("\n");
    printf("================================================================================\n");
    printf("                    EINSTEIN SUMMATION COMPREHENSIVE TESTS\n");
    printf("================================================================================\n");
    printf("\n");
    
    test_einsum_matmul();
    test_einsum_trace();
    test_einsum_transpose();
    test_einsum_outer_product();
    test_einsum_dot_product();
    test_einsum_elementwise();
    test_einsum_hash_uniqueness();
    test_einsum_derivative();
    test_einsum_simplification();
    test_einsum_printing();
    
    printf("================================================================================\n");
    printf("                         ALL TESTS PASSED! ✓\n");
    printf("================================================================================\n");
    printf("\n");
    
    return 0;
}
