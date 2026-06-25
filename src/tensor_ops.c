/*
 * tensor_ops.c — canonical tensor arithmetic operations.
 *
 * Single implementation of all element-wise arithmetic, reductions, and
 * matrix operations.  Both GridField (grid.c) and Literal (literal.c)
 * arithmetic delegate here so that every algorithm exists in exactly one
 * place.
 */

#include "../include/tensor_ops.h"

#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* ── Stride helper ─────────────────────────────────────────────────────── */

void tops_strides(const uint32_t *shape, size_t *strides, int rank)
{
    if (rank <= 0) return;
    strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * (size_t)shape[d + 1];
}

/* ── Internal helpers ──────────────────────────────────────────────────── */

/* Broadcast loop kernel: applies `op(a_val, b_val)` to every output element.
 * op_add=1 → add; op_add=0, op_sub=0 → multiply; op_sub=1 → subtract.
 * (Using an enum-style int avoids a function-pointer call overhead.)
 */
typedef enum { BCAST_ADD, BCAST_SUB, BCAST_MUL } BcastOp;

static void bcast_loop(
    double *out,
    const double *a, const double *b,
    const uint32_t *a_shape, const uint32_t *b_shape,
    const uint32_t *out_shape, int rank,
    BcastOp op)
{
    size_t a_strides[32], b_strides[32];
    tops_strides(a_shape, a_strides, rank);
    tops_strides(b_shape, b_strides, rank);

    /* total output elements */
    size_t total = 1;
    for (int d = 0; d < rank; d++) total *= (size_t)out_shape[d];

    uint32_t counters[32];
    for (int d = 0; d < rank; d++) counters[d] = 0;

    for (size_t flat = 0; flat < total; flat++) {
        size_t aoff = 0, boff = 0;
        for (int d = 0; d < rank; d++) {
            if (a_shape[d] != 1) aoff += (size_t)counters[d] * a_strides[d];
            if (b_shape[d] != 1) boff += (size_t)counters[d] * b_strides[d];
        }
        double av = a ? a[aoff] : 0.0;
        double bv = b ? b[boff] : 0.0;

        switch (op) {
            case BCAST_ADD: out[flat] = av + bv; break;
            case BCAST_SUB: out[flat] = av - bv; break;
            default:        out[flat] = av * bv; break;
        }

        /* ripple-increment multi-index */
        for (int d = rank - 1; d >= 0; d--) {
            counters[d]++;
            if (counters[d] < out_shape[d]) break;
            counters[d] = 0;
        }
    }
}

/* ── Same-size elementwise ─────────────────────────────────────────────── */

void tops_add_into(double *out, const double *a, const double *b, size_t n)
{
    if (!out) return;
    for (size_t i = 0; i < n; i++) {
        double av = a ? a[i] : 0.0;
        double bv = b ? b[i] : 0.0;
        out[i] = av + bv;
    }
}

void tops_subtract_into(double *out, const double *a, const double *b, size_t n)
{
    if (!out) return;
    for (size_t i = 0; i < n; i++) {
        double av = a ? a[i] : 0.0;
        double bv = b ? b[i] : 0.0;
        out[i] = av - bv;
    }
}

void tops_multiply_into(double *out, const double *a, const double *b, size_t n)
{
    if (!out) return;
    for (size_t i = 0; i < n; i++) {
        double av = a ? a[i] : 0.0;
        double bv = b ? b[i] : 0.0;
        out[i] = av * bv;
    }
}

/* ── Broadcasting elementwise ──────────────────────────────────────────── */

void tops_add_bcast(
    double *out,
    const double *a, const double *b,
    const uint32_t *a_shape, const uint32_t *b_shape,
    const uint32_t *out_shape, int rank)
{
    bcast_loop(out, a, b, a_shape, b_shape, out_shape, rank, BCAST_ADD);
}

void tops_subtract_bcast(
    double *out,
    const double *a, const double *b,
    const uint32_t *a_shape, const uint32_t *b_shape,
    const uint32_t *out_shape, int rank)
{
    bcast_loop(out, a, b, a_shape, b_shape, out_shape, rank, BCAST_SUB);
}

void tops_multiply_bcast(
    double *out,
    const double *a, const double *b,
    const uint32_t *a_shape, const uint32_t *b_shape,
    const uint32_t *out_shape, int rank)
{
    bcast_loop(out, a, b, a_shape, b_shape, out_shape, rank, BCAST_MUL);
}

/* ── Scalar and utility ops ────────────────────────────────────────────── */

void tops_scale_into(double *out, const double *in, double scalar, size_t n)
{
    if (!out) return;
    for (size_t i = 0; i < n; i++)
        out[i] = (in ? in[i] : 0.0) * scalar;
}

void tops_scale_inplace(double *data, double scalar, size_t n)
{
    if (!data) return;
    for (size_t i = 0; i < n; i++) data[i] *= scalar;
}

void tops_negate_into(double *out, const double *in, size_t n)
{
    if (!out) return;
    for (size_t i = 0; i < n; i++)
        out[i] = in ? -in[i] : 0.0;
}

void tops_axpy(double *y, double a, const double *x, size_t n)
{
    if (!y || !x) return;
    for (size_t i = 0; i < n; i++) y[i] += a * x[i];
}

void tops_copy_into(double *dst, const double *src, size_t n)
{
    if (!dst || !src) return;
    memcpy(dst, src, n * sizeof(double));
}

/* ── Reductions ────────────────────────────────────────────────────────── */

double tops_norm(const double *data, size_t n)
{
    if (!data) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) { double v = data[i]; sum += v * v; }
    return sqrt(sum);
}

double tops_dot(const double *a, const double *b, size_t n)
{
    if (!a || !b) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* ── Matrix / tensor ops ───────────────────────────────────────────────── */

/*
 * Batched GEMM: out[b,i,j] = sum_k a[b,i,k] * b_mat[b,k,j]
 *
 * Memory layout (row-major, batch outer):
 *   a      [batch, M, K]   stride: [M*K, K, 1]
 *   b_mat  [batch, K, N]   stride: [K*N, N, 1]
 *   out    [batch, M, N]   stride: [M*N, N, 1]
 */
void tops_matmul(
    double *out,
    const double *a, const double *b_mat,
    size_t M, size_t K, size_t N, size_t batch)
{
    if (!out) return;
    size_t a_batch_stride   = M * K;
    size_t b_batch_stride   = K * N;
    size_t out_batch_stride = M * N;

    for (size_t b = 0; b < batch; b++) {
        const double *ab  = a     ? a     + b * a_batch_stride   : NULL;
        const double *bb  = b_mat ? b_mat + b * b_batch_stride   : NULL;
        double       *ob  = out   +         b * out_batch_stride;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < K; k++) {
                    double av = ab ? ab[i * K + k] : 0.0;
                    double bv = bb ? bb[k * N + j] : 0.0;
                    sum += av * bv;
                }
                ob[i * N + j] = sum;
            }
        }
    }
}

/*
 * Batched transpose: out[b,j,i] = in[b,i,j]
 *
 * In layout:   [batch, M, N]  stride: [M*N, N, 1]
 * Out layout:  [batch, N, M]  stride: [N*M, M, 1]
 */
void tops_transpose(
    double *out, const double *in,
    size_t M, size_t N, size_t batch)
{
    if (!out) return;
    size_t mn = M * N;
    for (size_t b = 0; b < batch; b++) {
        const double *src = in  ? in  + b * mn : NULL;
        double       *dst = out +       b * mn;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                dst[j * M + i] = src ? src[i * N + j] : 0.0;
            }
        }
    }
}
