#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

/*
 * tensor_ops.h — canonical tensor arithmetic operations.
 *
 * These operate directly on double* buffers with shape/stride metadata
 * and are the single implementation used by both GridField and Literal
 * arithmetic, eliminating duplicated algorithm code.
 *
 * NULL data pointers are treated as all-zeros buffers.
 */

#include <stddef.h>
#include <stdint.h>

/* ── Stride helper ─────────────────────────────────────────────────────────
 * Row-major stride computation: strides[rank-1] = 1,
 * strides[d] = product(shape[d+1..rank-1]).
 * Output strides[] must hold at least `rank` elements.
 */
void tops_strides(const uint32_t *shape, size_t *strides, int rank);

/* ── Same-size elementwise ops (no broadcasting) ───────────────────────────
 * All three take `n` elements; NULL input treated as all zeros.
 * `out` must be pre-allocated to `n` doubles.
 */
void tops_add_into     (double *out, const double *a, const double *b, size_t n);
void tops_subtract_into(double *out, const double *a, const double *b, size_t n);
void tops_multiply_into(double *out, const double *a, const double *b, size_t n);

/* ── Broadcasting elementwise ops ──────────────────────────────────────────
 * Supports numpy-style broadcasting over `rank` dimensions.
 * out_shape[d] = max(a_shape[d], b_shape[d]); caller must allocate `out`
 * to product(out_shape) doubles.
 */
void tops_add_bcast(
    double *out,
    const double *a, const double *b,
    const uint32_t *a_shape, const uint32_t *b_shape,
    const uint32_t *out_shape, int rank);

void tops_subtract_bcast(
    double *out,
    const double *a, const double *b,
    const uint32_t *a_shape, const uint32_t *b_shape,
    const uint32_t *out_shape, int rank);

void tops_multiply_bcast(
    double *out,
    const double *a, const double *b,
    const uint32_t *a_shape, const uint32_t *b_shape,
    const uint32_t *out_shape, int rank);

/* ── Scalar and utility ops ────────────────────────────────────────────────
 * tops_scale_into  : out[i] = in[i] * scalar  (separate input/output)
 * tops_scale_inplace: data[i] *= scalar        (in-place)
 * tops_negate_into : out[i] = -in[i]
 * tops_axpy        : y[i] += a * x[i]         (in-place)
 * tops_copy_into   : memcpy semantics
 */
void tops_scale_into   (double *out, const double *in, double scalar, size_t n);
void tops_scale_inplace(double *data, double scalar, size_t n);
void tops_negate_into  (double *out, const double *in, size_t n);
void tops_axpy         (double *y, double a, const double *x, size_t n);
void tops_copy_into    (double *dst, const double *src, size_t n);

/* ── Reductions ────────────────────────────────────────────────────────────
 * tops_norm : sqrt(sum(data[i]^2))
 * tops_dot  : sum(a[i] * b[i]) over n elements
 */
double tops_norm(const double *data, size_t n);
double tops_dot (const double *a, const double *b, size_t n);

/* ── Matrix / tensor ops ───────────────────────────────────────────────────
 * tops_matmul  : batched GEMM  out[b,i,j] = sum_k a[b,i,k] * b_mat[b,k,j]
 *   out must hold batch*M*N doubles (zeroed by caller or all results written).
 *
 * tops_transpose: batched transpose  out[b,j,i] = in[b,i,j]
 *   out must hold batch*M*N doubles.
 *
 * For non-batched usage pass batch=1.
 */
void tops_matmul(
    double *out,
    const double *a, const double *b_mat,
    size_t M, size_t K, size_t N, size_t batch);

void tops_transpose(
    double *out, const double *in,
    size_t M, size_t N, size_t batch);

#endif /* TENSOR_OPS_H */
