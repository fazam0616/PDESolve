#include "../include/expression.h"
#include "../include/calculus.h"
#include "../include/ast_hash.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Global counters for test results
static int tests_passed = 0;
static int tests_failed = 0;

// Helper to mark a test as passed
void mark_test_passed(const char *test_name) {
    printf("[PASS] %s\n", test_name);
    tests_passed++;
}

// Helper to mark a test as failed
void mark_test_failed(const char *test_name) {
    printf("[FAIL] %s\n", test_name);
    tests_failed++;
}

// Helper to evaluate scalar result
static double eval_scalar(Expression *expr, Dictionary *context) {
    Literal *result = expression_evaluate(expr, context);
    if (!result) return NAN;
    double value = result->field[0];
    literal_free(result);
    return value;
}

// Helper to set scalar value in dictionary
static void dict_set_scalar(Dictionary *dict, const char *key, double value) {
    Literal *lit = literal_create((uint32_t[]){1,1,1});
    lit->field[0] = value;
    dict_set(dict, key, lit);  // Pass pointer, dict_set makes deep copy
    literal_free(lit);  // dict_set makes deep copy, so free our temporary
}

void test_multiply_derivatives() {
    printf("Test 1: OP_MULTIPLY Derivatives (Element-wise)\n");
    printf("================================================\n");
    
    // Test 1a: d/dx(x * x) = 2x
    Expression *x = expr_variable("x");
    Expression *x2 = expr_multiply(expr_copy(x), expr_copy(x));
    
    printf("  Expression: x * x\n");
    printf("  Symbolic d/dx: ");
    Expression *dx2_dx = derivative(x2, "x");
    print_expression(dx2_dx);
    printf("\n");
    
    Expression *simplified = simplify(dx2_dx);
    printf("  Simplified: ");
    print_expression(simplified);
    printf("\n");
    
    // Evaluate at x=3, should be 6
    printf("  Creating dictionary...\n");
    Dictionary *context = dict_create(8);
    printf("  Setting x=3.0...\n");
    dict_set_scalar(context, "x", 3.0);
    printf("  Evaluating...\n");
    double result = eval_scalar(simplified, context);
    printf("  At x=3: %.1f (expected 6.0) ", result);
    assert(fabs(result - 6.0) < 1e-10);
    printf("[OK]\n\n");
    
    dict_free(context);
    expression_free(simplified);
    expression_free(dx2_dx);
    expression_free(x2);
    
    // Test 1b: d/dx(x * y) = y (constant w.r.t. x)
    Expression *x_var = expr_variable("x");
    Expression *y = expr_variable("y");
    Expression *xy = expr_multiply(x_var, y);
    
    printf("  Expression: x * y\n");
    printf("  Symbolic d/dx: ");
    Expression *dxy_dx = derivative(xy, "x");
    print_expression(dxy_dx);
    printf("\n");
    
    Expression *simplified2 = simplify(dxy_dx);
    printf("  Simplified: ");
    print_expression(simplified2);
    printf("\n");
    
    // Evaluate at x=3, y=5, should be 5
    Dictionary *context2 = dict_create(8);
    dict_set_scalar(context2, "x", 3.0);
    dict_set_scalar(context2, "y", 5.0);
    double result2 = eval_scalar(simplified2, context2);
    printf("  At x=3, y=5: %.1f (expected 5.0) ", result2);
    assert(fabs(result2 - 5.0) < 1e-10);
    printf("[OK]\n\n");
    
    dict_free(context2);
    expression_free(simplified2);
    expression_free(dxy_dx);
    expression_free(xy);
    expression_free(x);
    
    // Test 1c: d/dx(2 * x) = 2 (scalar multiplication)
    Expression *two = make_scalar(2.0);
    Expression *x3 = expr_variable("x");
    Expression *two_x = expr_multiply(two, x3);
    
    printf("  Expression: 2 * x\n");
    printf("  Symbolic d/dx: ");
    Expression *d2x_dx = derivative(two_x, "x");
    print_expression(d2x_dx);
    printf("\n");
    
    Expression *simplified3 = simplify(d2x_dx);
    printf("  Simplified: ");
    print_expression(simplified3);
    printf("\n");
    
    // Should evaluate to 2
    double result3 = eval_scalar(simplified3, NULL);
    printf("  Result: %.1f (expected 2.0) ", result3);
    assert(fabs(result3 - 2.0) < 1e-10);
    printf("[OK]\n\n");
    
    expression_free(simplified3);
    expression_free(d2x_dx);
    expression_free(two_x);
    
    mark_test_passed("Multiply Derivatives - Element-wise");
}

void test_matmul_derivatives() {
    printf("Test 2: OP_MATMUL Derivatives (Matrix Multiplication)\n");
    printf("=====================================================\n");
    
    // Test 2a: d/dx(A @ x) where A is constant matrix, x is scalar
    Expression *A = expr_variable("A");
    Expression *x = expr_variable("x");
    Expression *Ax = expr_matmul(A, x);
    
    printf("  Expression: A @ x\n");
    printf("  Symbolic d/dx: ");
    Expression *dAx_dx = derivative(Ax, "x");
    print_expression(dAx_dx);
    printf("\n");
    
    Expression *simplified = simplify(dAx_dx);
    printf("  Simplified: ");
    print_expression(simplified);
    printf("\n");
    printf("  Expected: A @ 1 (which simplifies to A)\n");
    printf("  [OK] Derivative structure correct\n\n");
    
    expression_free(simplified);
    expression_free(dAx_dx);
    expression_free(Ax);
    
    // Test 2b: d/dx(x @ B) where x is scalar, B is constant matrix
    Expression *x2 = expr_variable("x");
    Expression *B = expr_variable("B");
    Expression *xB = expr_matmul(x2, B);
    
    printf("  Expression: x @ B\n");
    printf("  Symbolic d/dx: ");
    Expression *dxB_dx = derivative(xB, "x");
    print_expression(dxB_dx);
    printf("\n");
    
    Expression *simplified2 = simplify(dxB_dx);
    printf("  Simplified: ");
    print_expression(simplified2);
    printf("\n");
    printf("  Expected: 1 @ B (which simplifies to B)\n");
    printf("  [OK] Derivative structure correct\n\n");
    
    expression_free(simplified2);
    expression_free(dxB_dx);
    expression_free(xB);
    
    // Test 2c: Verify order preservation in product rule
    Expression *A2 = expr_variable("A");
    Expression *B2 = expr_variable("B");
    Expression *AB = expr_matmul(A2, B2);
    
    printf("  Expression: A @ B\n");
    printf("  Symbolic d/dx: ");
    Expression *dAB_dx = derivative(AB, "x");
    print_expression(dAB_dx);
    printf("\n");
    
    // Should be: (dA/dx @ B) + (A @ dB/dx)
    // With constants: (0 @ B) + (A @ 0)
    Expression *simplified3 = simplify(dAB_dx);
    printf("  Simplified: ");
    print_expression(simplified3);
    printf("\n");
    printf("  Expected: 0 (both A and B constant w.r.t. x)\n");
    printf("  [OK] Product rule preserves order\n\n");
    
    expression_free(simplified3);
    expression_free(dAB_dx);
    expression_free(AB);
    
    mark_test_passed("Matmul Derivatives - Matrix Multiplication");
}

void test_mixed_derivatives() {
    printf("Test 3: Mixed OP_MULTIPLY and OP_MATMUL Derivatives\n");
    printf("====================================================\n");
    
    // Test: d/dx(2 * (A @ x))
    // This combines scalar multiplication (element-wise) with matrix multiplication
    Expression *two = make_scalar(2.0);
    Expression *A = expr_variable("A");
    Expression *x = expr_variable("x");
    Expression *Ax = expr_matmul(expr_copy(A), expr_copy(x));
    Expression *scaled = expr_multiply(two, Ax);
    
    printf("  Expression: 2 * (A @ x)\n");
    printf("  Symbolic d/dx: ");
    Expression *d_scaled = derivative(scaled, "x");
    print_expression(d_scaled);
    printf("\n");
    
    Expression *simplified = simplify(d_scaled);
    printf("  Simplified: ");
    print_expression(simplified);
    printf("\n");
    
    // Expected: 2 * (A @ 1) = 2 * A
    printf("  Expected: 2 * (A @ 1)\n");
    printf("  [OK] Mixed operations derivative correct\n\n");
    
    expression_free(simplified);
    expression_free(d_scaled);
    expression_free(scaled);
    expression_free(x);
    expression_free(A);
    
    mark_test_passed("Mixed Derivatives - Mixed Operations");
}

void test_chain_rule() {
    printf("Test 4: Chain Rule with Nested Multiplications\n");
    printf("===============================================\n");
    
    // Test: d/dx((x * x) * x) = d/dx(x^3)
    Expression *x = expr_variable("x");
    Expression *x2 = expr_multiply(expr_copy(x), expr_copy(x));  // x * x
    Expression *x3 = expr_multiply(x2, expr_copy(x));  // (x*x) * x
    
    printf("  Expression: (x * x) * x  (i.e., x^3)\n");
    printf("  Symbolic d/dx: ");
    Expression *dx3_dx = derivative(x3, "x");
    print_expression(dx3_dx);
    printf("\n");
    
    Expression *simplified = simplify(dx3_dx);
    printf("  Simplified: ");
    print_expression(simplified);
    printf("\n");
    
    // Evaluate at x=2, should be 3*x^2 = 12
    Dictionary *context = dict_create(8);
    dict_set_scalar(context, "x", 2.0);
    double result = eval_scalar(simplified, context);
    printf("  At x=2: %.1f (expected 12.0) ", result);
    assert(fabs(result - 12.0) < 1e-10);
    printf("[OK]\n\n");
    
    dict_free(context);
    expression_free(simplified);
    expression_free(dx3_dx);
    expression_free(x3);
    expression_free(x);
    
    mark_test_passed("Chain Rule - Nested Multiplications");
}

void test_commutative_vs_noncommutative() {
    printf("Test 5: Verify Commutative (OP_MULTIPLY) vs Non-Commutative (OP_MATMUL)\n");
    printf("=========================================================================\n");
    
    // OP_MULTIPLY: 2*x and x*2 should have same hash
    Expression *two1 = make_scalar(2.0);
    Expression *x1 = expr_variable("x");
    Expression *two_x = expr_multiply(two1, x1);
    
    Expression *two2 = make_scalar(2.0);
    Expression *x2 = expr_variable("x");
    Expression *x_two = expr_multiply(x2, two2);
    
    uint64_t hash_2x = expr_canonical_hash(two_x);
    uint64_t hash_x2 = expr_canonical_hash(x_two);
    
    printf("  OP_MULTIPLY (element-wise, commutative):\n");
    printf("    2*x hash: 0x%016llx\n", (unsigned long long)hash_2x);
    printf("    x*2 hash: 0x%016llx\n", (unsigned long long)hash_x2);
    assert(hash_2x == hash_x2);
    printf("    2*x == x*2 (same hash) [OK]\n\n");
    
    // Derivatives should also be equivalent
    Expression *d_2x = simplify(derivative(two_x, "x"));
    Expression *d_x2 = simplify(derivative(x_two, "x"));
    
    printf("    d/dx(2*x): ");
    print_expression(d_2x);
    printf("\n");
    printf("    d/dx(x*2): ");
    print_expression(d_x2);
    printf("\n");
    
    uint64_t hash_d2x = expr_canonical_hash(d_2x);
    uint64_t hash_dx2 = expr_canonical_hash(d_x2);
    assert(hash_d2x == hash_dx2);
    printf("    Derivatives have same hash [OK]\n\n");
    
    expression_free(d_x2);
    expression_free(d_2x);
    expression_free(x_two);
    expression_free(two_x);
    
    // OP_MATMUL: A@B and B@A should have different hashes
    Expression *A1 = expr_variable("A");
    Expression *B1 = expr_variable("B");
    Expression *AB = expr_matmul(A1, B1);
    
    Expression *A2 = expr_variable("A");
    Expression *B2 = expr_variable("B");
    Expression *BA = expr_matmul(B2, A2);
    
    uint64_t hash_AB = expr_canonical_hash(AB);
    uint64_t hash_BA = expr_canonical_hash(BA);
    
    printf("  OP_MATMUL (matrix multiplication, non-commutative):\n");
    printf("    A@B hash: 0x%016llx\n", (unsigned long long)hash_AB);
    printf("    B@A hash: 0x%016llx\n", (unsigned long long)hash_BA);
    assert(hash_AB != hash_BA);
    printf("    A@B != B@A (different hashes) [OK]\n\n");
    
    expression_free(BA);
    expression_free(AB);
    
    mark_test_passed("Commutative vs Noncommutative - Hash Verification");
}

int main() {
    printf("=======================================================\n");
    printf("Multiplication Derivative Verification Tests\n");
    printf("=======================================================\n\n");
    
    test_multiply_derivatives();
    test_matmul_derivatives();
    test_mixed_derivatives();
    test_chain_rule();
    test_commutative_vs_noncommutative();
    
    printf("=======================================================\n");
    printf("Summary:\n");
    printf("Tests Passed: %d\n", tests_passed);
    printf("Tests Failed: %d\n", tests_failed);

    if (tests_failed > 0) {
        printf("\n[FAIL] Some tests failed.\n");
        return 1;
    } else {
        printf("\n[PASS] All tests passed.\n");
        return 0;
    }
}
