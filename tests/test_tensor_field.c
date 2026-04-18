/*
 * test_tensor_field.c
 *
 * Tests for the TensorField N-rank CPU API (grid.h / grid.c) and for the
 * compile-time GLSL generation path of gpu_compile_tensor_expr (gpu_tensor.c).
 *
 * Structure:
 *   Part 1 — CPU TensorField API: create, get, set, linear_index
 *   Part 2 — Parity: same matrix ops produce identical results on both
 *             Literal (the existing engine) and TensorField
 *   Part 3 — GLSL generation: gpu_compile_tensor_expr emits valid source
 *             for every supported operation without requiring a GL context
 *
 * No OpenGL context is needed: GL calls are only reached inside
 * gpu_run_tensor_program, which is NOT exercised here.
 */

#include "../include/grid.h"
#include "../include/expression.h"
#include "../include/gpu_compiler.h"
#include "../include/literal.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

/* =========================================================================
 * Shared helpers
 * ========================================================================= */

/* Build a 2×2 Literal matrix (shape {batch=1, rows=2, cols=2}).
 * Row-major flat layout: field[0]=a (0,0), [1]=b (0,1), [2]=c (1,0), [3]=d (1,1).
 * Ownership is with the caller; pass into expr_literal to transfer. */
static Literal *make_lit_2x2(double a, double b, double c, double d) {
    Literal *L = literal_create((uint32_t[]){1, 2, 2});
    L->field[0] = a;
    L->field[1] = b;
    L->field[2] = c;
    L->field[3] = d;
    return L;
}

/* Build a rank-2 TensorField (shape {2, 2}).
 * Row-major: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d. */
static TensorField *make_tf_2x2(double a, double b, double c, double d) {
    int shape[2] = {2, 2};
    TensorField *tf = tensor_field_create(2, shape);
    assert(tf != NULL);
    int idx[2];
    idx[0] = 0; idx[1] = 0; tensor_field_set(tf, idx, a);
    idx[0] = 0; idx[1] = 1; tensor_field_set(tf, idx, b);
    idx[0] = 1; idx[1] = 0; tensor_field_set(tf, idx, c);
    idx[0] = 1; idx[1] = 1; tensor_field_set(tf, idx, d);
    return tf;
}

/* =========================================================================
 * Part 1 — CPU TensorField API
 * ========================================================================= */

static void test_tf_create_rank1(void) {
    printf("  test_tf_create_rank1 ...\n");
    int shape[1] = {5};
    TensorField *tf = tensor_field_create(1, shape);
    assert(tf != NULL);
    assert(tf->rank == 1);
    assert(tf->shape[0] == 5);
    assert(tf->total == 5);
    assert(tf->strides[0] == 1);
    assert(tf->data != NULL);
    assert(tf->ssbo == 0);
    assert(tf->gpu_dirty == true);
    tensor_field_free(tf);
    printf("    [OK]\n");
}

static void test_tf_create_rank2(void) {
    printf("  test_tf_create_rank2 ...\n");
    /* Row-major strides for shape {3, 4}: strides[0]=4, strides[1]=1 */
    int shape[2] = {3, 4};
    TensorField *tf = tensor_field_create(2, shape);
    assert(tf != NULL);
    assert(tf->rank == 2);
    assert(tf->shape[0] == 3);
    assert(tf->shape[1] == 4);
    assert(tf->total == 12);
    assert(tf->strides[0] == 4);
    assert(tf->strides[1] == 1);
    tensor_field_free(tf);
    printf("    [OK]\n");
}

static void test_tf_create_rank3(void) {
    printf("  test_tf_create_rank3 ...\n");
    /* Row-major strides for shape {2, 3, 5}: strides[0]=15, [1]=5, [2]=1 */
    int shape[3] = {2, 3, 5};
    TensorField *tf = tensor_field_create(3, shape);
    assert(tf != NULL);
    assert(tf->rank == 3);
    assert(tf->total == 30);
    assert(tf->strides[0] == 15);
    assert(tf->strides[1] == 5);
    assert(tf->strides[2] == 1);
    tensor_field_free(tf);
    printf("    [OK]\n");
}

static void test_tf_linear_index(void) {
    printf("  test_tf_linear_index ...\n");
    /* Row-major: index (r, c) → r * ncols + c */
    int shape[2] = {3, 4};
    TensorField *tf = tensor_field_create(2, shape);
    int idx[2];
    idx[0] = 0; idx[1] = 0; assert(tensor_field_linear_index(tf, idx) == 0);
    idx[0] = 0; idx[1] = 3; assert(tensor_field_linear_index(tf, idx) == 3);
    idx[0] = 1; idx[1] = 0; assert(tensor_field_linear_index(tf, idx) == 4);
    idx[0] = 2; idx[1] = 3; assert(tensor_field_linear_index(tf, idx) == 11);
    tensor_field_free(tf);
    printf("    [OK]\n");
}

static void test_tf_get_set(void) {
    printf("  test_tf_get_set ...\n");
    int shape[2] = {2, 3};
    TensorField *tf = tensor_field_create(2, shape);

    /* Write every element with a unique value */
    double val = 0.0;
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            int idx[2] = {r, c};
            tensor_field_set(tf, idx, val);
            val += 1.0;
        }
    }

    /* Read back and verify */
    val = 0.0;
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            int idx[2] = {r, c};
            assert(fabs(tensor_field_get(tf, idx) - val) < 1e-15);
            val += 1.0;
        }
    }

    /* Overwrite a single element */
    int idx[2] = {1, 2};
    tensor_field_set(tf, idx, 99.5);
    assert(fabs(tensor_field_get(tf, idx) - 99.5) < 1e-15);

    tensor_field_free(tf);
    printf("    [OK]\n");
}

static void test_tf_zero_initialized(void) {
    printf("  test_tf_zero_initialized ...\n");
    int shape[2] = {4, 4};
    TensorField *tf = tensor_field_create(2, shape);
    for (size_t i = 0; i < tf->total; i++)
        assert(tf->data[i] == 0.0);
    tensor_field_free(tf);
    printf("    [OK]\n");
}

static void test_tf_rank3_ops(void) {
    printf("  test_tf_rank3_ops — rank-3 get/set ...\n");
    int shape[3] = {2, 3, 4};
    TensorField *tf = tensor_field_create(3, shape);
    assert(tf->total == 24);
    assert(tf->strides[0] == 12);
    assert(tf->strides[1] == 4);
    assert(tf->strides[2] == 1);

    /* Fill with values that encode the 3D index */
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                int idx[3] = {i, j, k};
                tensor_field_set(tf, idx, (double)(i * 100 + j * 10 + k));
            }
        }
    }

    /* Read back */
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 4; k++) {
                int idx[3] = {i, j, k};
                assert(fabs(tensor_field_get(tf, idx)
                            - (double)(i * 100 + j * 10 + k)) < 1e-15);
            }
        }
    }
    tensor_field_free(tf);
    printf("    [OK]\n");
}

/* =========================================================================
 * Part 2 — Parity: Rank-2 TensorField vs Literal (CPU evaluation)
 *
 * All matrix operations available on Literal work identically when the same
 * data is stored in a rank-2 TensorField.  We verify this by running each
 * operation on both representations and comparing element-by-element.
 * ========================================================================= */

static void test_parity_storage(void) {
    printf("  test_parity_storage ...\n");
    /*
     * A = [1 2]
     *     [3 4]
     * Both APIs must store and return the same values for every index.
     */
    Literal    *L  = make_lit_2x2(1.0, 2.0, 3.0, 4.0);
    TensorField *tf = make_tf_2x2(1.0, 2.0, 3.0, 4.0);

    int idx[2];
    idx[0] = 0; idx[1] = 0; assert(fabs(L->field[0] - tensor_field_get(tf, idx)) < 1e-15);
    idx[0] = 0; idx[1] = 1; assert(fabs(L->field[1] - tensor_field_get(tf, idx)) < 1e-15);
    idx[0] = 1; idx[1] = 0; assert(fabs(L->field[2] - tensor_field_get(tf, idx)) < 1e-15);
    idx[0] = 1; idx[1] = 1; assert(fabs(L->field[3] - tensor_field_get(tf, idx)) < 1e-15);

    literal_free(L);
    tensor_field_free(tf);
    printf("    [OK]\n");
}

static void test_parity_matmul_cpu(void) {
    printf("  test_parity_matmul_cpu — A@B matches for Literal and TensorField ...\n");
    /*
     * A = [1 2]   B = [5 6]   A@B = [19 22]
     *     [3 4]       [7 8]         [43 50]
     */
    double expected[4] = {19.0, 22.0, 43.0, 50.0};

    /* ── Literal path via expression_evaluate ── */
    Literal *LA = make_lit_2x2(1, 2, 3, 4);
    Literal *LB = make_lit_2x2(5, 6, 7, 8);
    Expression *eLA = expr_literal(LA);
    Expression *eLB = expr_literal(LB);
    Expression *ematmul = expr_matmul(eLA, eLB);
    Literal *Lresult = expression_evaluate(ematmul, NULL);
    assert(Lresult != NULL);
    for (int i = 0; i < 4; i++)
        assert(fabs(Lresult->field[i] - expected[i]) < 1e-10);
    printf("    Literal: [[%.0f,%.0f],[%.0f,%.0f]]\n",
           Lresult->field[0], Lresult->field[1],
           Lresult->field[2], Lresult->field[3]);
    literal_free(Lresult);
    expression_free(ematmul);

    /* ── TensorField path — manual 2×2 CPU matmul ── */
    TensorField *tfA = make_tf_2x2(1, 2, 3, 4);
    TensorField *tfB = make_tf_2x2(5, 6, 7, 8);
    double tf_vals[4];
    for (int i = 0; i < 2; i++) {
        for (int k = 0; k < 2; k++) {
            tf_vals[i * 2 + k] = 0.0;
            for (int j = 0; j < 2; j++) {
                int ia[2] = {i, j};
                int ib[2] = {j, k};
                tf_vals[i * 2 + k] +=
                    tensor_field_get(tfA, ia) * tensor_field_get(tfB, ib);
            }
        }
    }
    for (int i = 0; i < 4; i++)
        assert(fabs(tf_vals[i] - expected[i]) < 1e-10);
    printf("    TensorField: [[%.0f,%.0f],[%.0f,%.0f]]\n",
           tf_vals[0], tf_vals[1], tf_vals[2], tf_vals[3]);
    tensor_field_free(tfA);
    tensor_field_free(tfB);
    printf("    [OK]\n");
}

static void test_parity_einsum_matmul_cpu(void) {
    printf("  test_parity_einsum_matmul_cpu — einsum ij,jk->ik == matmul ...\n");
    /*
     * Verify that expr_einsum("ij","jk","ik") and expr_matmul produce the
     * same result for the same [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50].
     */
    double expected[4] = {19.0, 22.0, 43.0, 50.0};

    Literal *LA1 = make_lit_2x2(1, 2, 3, 4);
    Literal *LB1 = make_lit_2x2(5, 6, 7, 8);
    Expression *einsum_expr =
        expr_einsum(expr_literal(LA1), "ij", expr_literal(LB1), "jk", "ik");
    Literal *einsum_res = expression_evaluate(einsum_expr, NULL);
    assert(einsum_res != NULL);

    Literal *LA2 = make_lit_2x2(1, 2, 3, 4);
    Literal *LB2 = make_lit_2x2(5, 6, 7, 8);
    Expression *matmul_expr = expr_matmul(expr_literal(LA2), expr_literal(LB2));
    Literal *matmul_res = expression_evaluate(matmul_expr, NULL);
    assert(matmul_res != NULL);

    for (int i = 0; i < 4; i++) {
        assert(fabs(einsum_res->field[i] - expected[i]) < 1e-10);
        assert(fabs(matmul_res->field[i] - expected[i]) < 1e-10);
    }

    literal_free(einsum_res);
    literal_free(matmul_res);
    expression_free(einsum_expr);
    expression_free(matmul_expr);
    printf("    [OK]\n");
}

static void test_parity_dot_cpu(void) {
    printf("  test_parity_dot_cpu — dot product for Literal and TensorField ...\n");
    /*
     * v = [3, 4]   v·v = 9 + 16 = 25
     * Literal shape: {1, 1, 2}  (1D vector in a 3D Literal)
     * TensorField:   rank=1, shape={2}
     */
    uint32_t vshape[3] = {1, 1, 2};
    Literal *Lv = literal_create(vshape);
    Lv->field[0] = 3.0;
    Lv->field[1] = 4.0;
    Expression *eLv1 = expr_literal(Lv);
    /* literal_copy while Lv is still alive (eLv1 owns it but hasn't freed it) */
    Expression *eLv2 = expr_literal(literal_copy(Lv));
    Expression *dot_expr = expr_dot(eLv1, eLv2);
    Literal *dot_res = expression_evaluate(dot_expr, NULL);
    assert(dot_res != NULL);
    assert(fabs(dot_res->field[0] - 25.0) < 1e-10);
    printf("    dot([3,4],[3,4]) = %.0f (Literal) [OK]\n", dot_res->field[0]);
    literal_free(dot_res);
    expression_free(dot_expr);

    /* TensorField rank-1 manual dot */
    int tvshape[1] = {2};
    TensorField *tfv = tensor_field_create(1, tvshape);
    int i0[1] = {0}; tensor_field_set(tfv, i0, 3.0);
    int i1[1] = {1}; tensor_field_set(tfv, i1, 4.0);
    double tf_dot = 0.0;
    for (int i = 0; i < 2; i++) {
        int idx[1] = {i};
        double vi = tensor_field_get(tfv, idx);
        tf_dot += vi * vi;
    }
    assert(fabs(tf_dot - 25.0) < 1e-10);
    printf("    dot([3,4],[3,4]) = %.0f (TensorField) [OK]\n", tf_dot);
    tensor_field_free(tfv);
}

static void test_parity_transpose_cpu(void) {
    printf("  test_parity_transpose_cpu — transpose Literal and TensorField ...\n");
    /*
     * M = [1 2]   M^T = [1 3]
     *     [3 4]          [2 4]
     */

    /* ── Literal path ── */
    Literal *LM = make_lit_2x2(1, 2, 3, 4);
    Expression *trans_expr = expr_transpose(expr_literal(LM));
    Literal *trans_res = expression_evaluate(trans_expr, NULL);
    assert(trans_res != NULL);
    /* Row-major layout of [1 3; 2 4]: field = {1, 3, 2, 4} */
    assert(fabs(trans_res->field[0] - 1.0) < 1e-10);
    assert(fabs(trans_res->field[1] - 3.0) < 1e-10);
    assert(fabs(trans_res->field[2] - 2.0) < 1e-10);
    assert(fabs(trans_res->field[3] - 4.0) < 1e-10);
    printf("    M^T = [[%.0f,%.0f],[%.0f,%.0f]] (Literal) [OK]\n",
           trans_res->field[0], trans_res->field[1],
           trans_res->field[2], trans_res->field[3]);
    literal_free(trans_res);
    expression_free(trans_expr);

    /* ── TensorField path — manual rank-2 transpose ── */
    TensorField *tfM = make_tf_2x2(1, 2, 3, 4);
    int shape_t[2] = {2, 2};
    TensorField *tfT = tensor_field_create(2, shape_t);
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            int src[2] = {r, c};
            int dst[2] = {c, r};
            tensor_field_set(tfT, dst, tensor_field_get(tfM, src));
        }
    }
    /* Verify: T[i][j] == M[j][i] */
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            int a[2] = {r, c};
            int b[2] = {c, r};
            assert(fabs(tensor_field_get(tfT, a) -
                        tensor_field_get(tfM, b)) < 1e-15);
        }
    }
    printf("    M^T validated (TensorField) [OK]\n");
    tensor_field_free(tfM);
    tensor_field_free(tfT);
}

static void test_parity_elementwise_add_cpu(void) {
    printf("  test_parity_elementwise_add_cpu — A+B for Literal and TensorField ...\n");
    /*
     * A = [1 2]   B = [10 20]   A+B = [11 22]
     *     [3 4]       [30 40]          [33 44]
     */
    double expected[4] = {11.0, 22.0, 33.0, 44.0};

    /* Literal path */
    Literal *LA = make_lit_2x2(1, 2, 3, 4);
    Literal *LB = make_lit_2x2(10, 20, 30, 40);
    Expression *add_expr = expr_add(expr_literal(LA), expr_literal(LB));
    Literal *add_res = expression_evaluate(add_expr, NULL);
    assert(add_res != NULL);
    for (int i = 0; i < 4; i++)
        assert(fabs(add_res->field[i] - expected[i]) < 1e-10);
    literal_free(add_res);
    expression_free(add_expr);

    /* TensorField path — element-wise addition */
    TensorField *tfA = make_tf_2x2(1, 2, 3, 4);
    TensorField *tfB = make_tf_2x2(10, 20, 30, 40);
    int shape[2] = {2, 2};
    TensorField *tfC = tensor_field_create(2, shape);
    for (size_t i = 0; i < tfA->total; i++)
        tfC->data[i] = tfA->data[i] + tfB->data[i];
    for (int i = 0; i < 4; i++)
        assert(fabs(tfC->data[i] - expected[i]) < 1e-10);
    tensor_field_free(tfA);
    tensor_field_free(tfB);
    tensor_field_free(tfC);
    printf("    [OK]\n");
}

/* =========================================================================
 * Part 3 — GLSL generation (no GL context required)
 *
 * gpu_compile_tensor_expr only builds string buffers and allocates structs.
 * No GL functions are invoked until gpu_run_tensor_program is called.
 * ========================================================================= */

static void test_glsl_matmul(void) {
    printf("  test_glsl_matmul — compile OP_MATMUL ...\n");
    Expression *e = expr_matmul(expr_variable("A"), expr_variable("B"));
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    assert(prog->n_kernels == 1);
    assert(prog->kernels[0] != NULL);
    assert(prog->kernels[0]->glsl_src != NULL);
    assert(strstr(prog->kernels[0]->glsl_src, "#version 430") != NULL);
    /* Two input SSBOs named after the variables */
    assert(prog->kernels[0]->n_inputs == 2);
    assert(strcmp(prog->kernels[0]->input_names[0], "A") == 0);
    assert(strcmp(prog->kernels[0]->input_names[1], "B") == 0);
    /* Metadata must reflect the synthesised "ij,jk->ik" subscripts */
    assert(prog->op == OP_MATMUL);
    assert(prog->left_indices  && strcmp(prog->left_indices,  "ij") == 0);
    assert(prog->right_indices && strcmp(prog->right_indices, "jk") == 0);
    assert(prog->out_indices   && strcmp(prog->out_indices,   "ik") == 0);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

static void test_glsl_einsum_matmul(void) {
    printf("  test_glsl_einsum_matmul — compile OP_EINSUM \"ij,jk->ik\" ...\n");
    Expression *e =
        expr_einsum(expr_variable("M"), "ij", expr_variable("N"), "jk", "ik");
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    assert(prog->kernels[0]->glsl_src != NULL);
    assert(strstr(prog->kernels[0]->glsl_src, "#version 430") != NULL);
    assert(prog->op == OP_EINSUM);
    assert(strcmp(prog->left_indices,  "ij") == 0);
    assert(strcmp(prog->right_indices, "jk") == 0);
    assert(strcmp(prog->out_indices,   "ik") == 0);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

static void test_glsl_transpose(void) {
    printf("  test_glsl_transpose — compile OP_TRANSPOSE ...\n");
    Expression *e = expr_transpose(expr_variable("A"));
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    /* Unary: one input SSBO */
    assert(prog->kernels[0]->n_inputs == 1);
    assert(prog->kernels[0]->glsl_src != NULL);
    assert(prog->op == OP_TRANSPOSE);
    /* Synthesised from "ij->ji" */
    assert(prog->left_indices && strcmp(prog->left_indices, "ij") == 0);
    assert(prog->out_indices  && strcmp(prog->out_indices,  "ji") == 0);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

static void test_glsl_dot(void) {
    printf("  test_glsl_dot — compile OP_DOT \"i,i->\" ...\n");
    Expression *e = expr_dot(expr_variable("u"), expr_variable("v"));
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    assert(prog->op == OP_DOT);
    /* Scalar output: out_indices is empty string */
    assert(prog->out_indices && strlen(prog->out_indices) == 0);
    /* Scalar output shader guards on flat == 0 to avoid duplicate work */
    assert(strstr(prog->kernels[0]->glsl_src, "flat != 0") != NULL);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

static void test_glsl_elementwise_add(void) {
    printf("  test_glsl_elementwise_add — compile OP_ADD ...\n");
    Expression *e = expr_add(expr_variable("X"), expr_variable("Y"));
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    assert(prog->kernels[0]->n_inputs == 2);
    assert(prog->op == OP_ADD);
    /* Elementwise shader uses a `total` uniform for bounds check */
    assert(strstr(prog->kernels[0]->glsl_src, "total") != NULL);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

static void test_glsl_elementwise_multiply(void) {
    printf("  test_glsl_elementwise_multiply — compile OP_MULTIPLY ...\n");
    Expression *e = expr_multiply(expr_variable("P"), expr_variable("Q"));
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    assert(prog->kernels[0]->n_inputs == 2);
    assert(prog->op == OP_MULTIPLY);
    assert(strstr(prog->kernels[0]->glsl_src, "#version 430") != NULL);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

static void test_glsl_elementwise_negate(void) {
    printf("  test_glsl_elementwise_negate — compile OP_NEGATE ...\n");
    Expression *e = expr_negate(expr_variable("V"));
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    /* Unary: one input SSBO */
    assert(prog->kernels[0]->n_inputs == 1);
    assert(prog->op == OP_NEGATE);
    assert(strstr(prog->kernels[0]->glsl_src, "#version 430") != NULL);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

static void test_glsl_einsum_outer_product(void) {
    printf("  test_glsl_einsum_outer_product — compile \"i,j->ij\" ...\n");
    Expression *e =
        expr_einsum(expr_variable("a"), "i", expr_variable("b"), "j", "ij");
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    assert(prog->op == OP_EINSUM);
    assert(strcmp(prog->left_indices,  "i")  == 0);
    assert(strcmp(prog->right_indices, "j")  == 0);
    assert(strcmp(prog->out_indices,   "ij") == 0);
    assert(strstr(prog->kernels[0]->glsl_src, "#version 430") != NULL);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

static void test_glsl_einsum_trace(void) {
    printf("  test_glsl_einsum_trace — compile \"ii->\" (trace) ...\n");
    /* Trace: scalar = Σ_i A[i,i]  — uses only left operand */
    Expression *e = expr_einsum(expr_variable("A"), "ii", NULL, "", "");
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    assert(prog->op == OP_EINSUM);
    /* Scalar output → out_indices is empty */
    assert(prog->out_indices && strlen(prog->out_indices) == 0);
    /* Scalar shader guards against duplicate threads */
    assert(strstr(prog->kernels[0]->glsl_src, "flat != 0") != NULL);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

static void test_glsl_kernel_has_ssbo_bindings(void) {
    printf("  test_glsl_kernel_has_ssbo_bindings — GLSL declares buffer blocks ...\n");
    Expression *e = expr_matmul(expr_variable("A"), expr_variable("B"));
    TensorProgram *prog = gpu_compile_tensor_expr(e, GPU_BACKEND_OPENGL);
    assert(prog != NULL);
    const char *src = prog->kernels[0]->glsl_src;
    /* Must declare input SSBOs at binding 0 and 1, output at binding 2 */
    assert(strstr(src, "binding = 0") != NULL);
    assert(strstr(src, "binding = 1") != NULL);
    assert(strstr(src, "binding = 2") != NULL);
    /* Must declare uniform ints for each index character used in "ij,jk->ik" */
    assert(strstr(src, "uniform int size_i") != NULL);
    assert(strstr(src, "uniform int size_j") != NULL);
    assert(strstr(src, "uniform int size_k") != NULL);
    tensor_program_free(prog);
    expression_free(e);
    printf("    [OK]\n");
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("TensorField Tests\n");
    printf("============================================================\n\n");

    printf("Part 1: CPU TensorField API\n");
    printf("---------------------------\n");
    test_tf_create_rank1();
    test_tf_create_rank2();
    test_tf_create_rank3();
    test_tf_linear_index();
    test_tf_get_set();
    test_tf_zero_initialized();
    test_tf_rank3_ops();
    printf("\n");

    printf("Part 2: Parity — TensorField vs Literal (CPU)\n");
    printf("-----------------------------------------------\n");
    test_parity_storage();
    test_parity_matmul_cpu();
    test_parity_einsum_matmul_cpu();
    test_parity_dot_cpu();
    test_parity_transpose_cpu();
    test_parity_elementwise_add_cpu();
    printf("\n");

    printf("Part 3: GLSL Generation (no GL context required)\n");
    printf("-------------------------------------------------\n");
    test_glsl_matmul();
    test_glsl_einsum_matmul();
    test_glsl_transpose();
    test_glsl_dot();
    test_glsl_elementwise_add();
    test_glsl_elementwise_multiply();
    test_glsl_elementwise_negate();
    test_glsl_einsum_outer_product();
    test_glsl_einsum_trace();
    test_glsl_kernel_has_ssbo_bindings();
    printf("\n");

    printf("============================================================\n");
    printf("All TensorField tests passed! [OK]\n");
    printf("============================================================\n");
    return 0;
}
