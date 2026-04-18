/*
 * gpu_tensor.c — N-rank tensor compute shader infrastructure (OpenGL 4.3+)
 *
 * Implements:
 *   • tensor_field_upload / tensor_field_download / gpu_tensor_field_free
 *   • Compute shader GLSL emission for EINSUM (arbitrary subscripts),
 *     MATMUL, DOT, TRANSPOSE, and elementwise ops (ADD, NEGATE, MULTIPLY …)
 *   • gpu_compile_tensor_expr — compile an Expression to a TensorProgram
 *   • gpu_run_tensor_program  — execute and return a new output TensorField
 *
 * Design notes:
 *   All tensor data lives in GL_SHADER_STORAGE_BUFFERs (std430 layout).
 *   The GLSL type `double` is used throughout; this requires
 *   GL_ARB_gpu_shader_fp64 (core since OpenGL 4.0 / GLSL 4.00).
 *   The compute workgroup size is fixed at 256×1×1; outputs are indexed via
 *   a flat global invocation ID that is unpacked to a multi-dimensional index
 *   inside the shader.  For scalar outputs (e.g. DOT "i,i->") only one thread
 *   executes, performing a sequential CPU-style reduction on the GPU — suitable
 *   for correctness but not optimal for very large vectors.
 *
 * This file has no dependency on GridMetadata or fragment-shader machinery.
 */

#include "../include/gpu_compiler.h"
#include "../include/grid.h"
#include "../include/expression.h"
#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* =========================================================================
 * Internal: dynamic string buffer helpers
 * ========================================================================= */

typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
} StrBuf;

static void sb_init(StrBuf *sb) {
    sb->cap = 4096;
    sb->buf = calloc(1, sb->cap);
    sb->len = 0;
}

static void sb_append(StrBuf *sb, const char *s) {
    size_t slen = strlen(s);
    size_t need  = sb->len + slen + 1;
    if (need > sb->cap) {
        sb->cap = need * 2;
        sb->buf = realloc(sb->buf, sb->cap);
    }
    memcpy(sb->buf + sb->len, s, slen + 1);
    sb->len += slen;
}

static void sb_appendf(StrBuf *sb, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    sb_append(sb, tmp);
}

/* =========================================================================
 * Internal: index-character helpers
 * ========================================================================= */

/* Add c to chars[0..nchars-1] if not already present. */
static void add_char(char c, char *chars, int *nchars) {
    for (int i = 0; i < *nchars; i++)
        if (chars[i] == c) return;
    chars[(*nchars)++] = c;
}

/* Collect all unique characters from string s into chars / nchars. */
static void collect_indices(const char *s, char *chars, int *nchars) {
    if (!s) return;
    for (const char *p = s; *p; p++)
        add_char(*p, chars, nchars);
}

/* Return 1 if c appears in string s. */
static int char_in(char c, const char *s) {
    return s && (strchr(s, c) != NULL);
}

/* =========================================================================
 * Internal: compute shader compilation
 * ========================================================================= */

static GLuint compile_compute_shader(const char *src) {
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "[gpu_tensor] compute shader compile error:\n%s\n"
                        "--- GLSL source ---\n%s\n", log, src);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint link_compute_program(GLuint cs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, cs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), NULL, log);
        fprintf(stderr, "[gpu_tensor] compute program link error:\n%s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

static int compute_kernel_ensure_program(ComputeKernel *k) {
    if (k->gl_program) return 0;
    GLuint cs = compile_compute_shader(k->glsl_src);
    if (!cs) return -1;
    GLuint prog = link_compute_program(cs);
    glDeleteShader(cs);
    if (!prog) return -1;
    k->gl_program = prog;
    return 0;
}

/* =========================================================================
 * TensorField GPU operations
 * ========================================================================= */

void tensor_field_upload(TensorField *tf) {
    if (!tf || !tf->data) return;
    if (!tf->ssbo)
        glGenBuffers(1, &tf->ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tf->ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(tf->total * sizeof(double)),
                 tf->data,
                 GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    tf->gpu_dirty = false;
}

void tensor_field_download(TensorField *tf) {
    if (!tf || !tf->ssbo) return;
    if (!tf->data)
        tf->data = calloc(tf->total, sizeof(double));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tf->ssbo);
    void *ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (ptr) {
        memcpy(tf->data, ptr, tf->total * sizeof(double));
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void gpu_tensor_field_free(TensorField *tf) {
    if (!tf) return;
    if (tf->ssbo) {
        glDeleteBuffers(1, &tf->ssbo);
        tf->ssbo = 0;
    }
    tensor_field_free(tf);
}

/* =========================================================================
 * GLSL EINSUM emit
 *
 * Generates a #version 430 core compute shader that implements:
 *   out[out_indices] = Σ_{contracted} A[left_indices] * B[right_indices]
 *
 * For unary contractions (e.g. "ij->ji"), right_idx may be NULL or "".
 * For pure copies/transposes, the inner "loop" degenerates to a single
 * assignment (acc = 0 + A[...]), which is correct.
 *
 * For scalar outputs (e.g. "i,i->"), a single invocation runs the full
 * reduction sequentially.  This is correct but not optimal for large vectors.
 *
 * The caller must set uniform int size_X for each unique index character X
 * before dispatch.
 * ========================================================================= */

static char* emit_compute_einsum(const char *left_idx,
                                  const char *right_idx,
                                  const char *out_idx) {
    assert(left_idx && out_idx);
    bool is_unary = (!right_idx || right_idx[0] == '\0');

    /* Collect all unique index characters (in encounter order). */
    char all_chars[64]; int n_all = 0;
    collect_indices(left_idx,  all_chars, &n_all);
    if (!is_unary) collect_indices(right_idx, all_chars, &n_all);
    collect_indices(out_idx,   all_chars, &n_all);

    /* Partition into free (in out_idx) and contracted (not in out_idx). */
    char free_chars[64]; int n_free = 0;
    char cont_chars[64]; int n_cont = 0;

    /* Preserve the order imposed by out_idx for free indices so that the
       flat→multi-index unpacking matches the row-major output layout. */
    for (int i = 0; out_idx[i]; i++)
        add_char(out_idx[i], free_chars, &n_free);

    for (int i = 0; i < n_all; i++)
        if (!char_in(all_chars[i], out_idx))
            cont_chars[n_cont++] = all_chars[i];

    StrBuf sb; sb_init(&sb);

    /* ── Shader header & SSBO declarations ── */
    sb_append(&sb, "#version 430 core\n");
    sb_append(&sb, "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n");
    sb_append(&sb, "layout(std430, binding = 0) readonly  buffer BufA   { double dataA[]; };\n");
    if (!is_unary)
        sb_append(&sb, "layout(std430, binding = 1) readonly  buffer BufB   { double dataB[]; };\n");

    int out_binding = is_unary ? 1 : 2;
    sb_appendf(&sb, "layout(std430, binding = %d) writeonly buffer BufOut { double dataOut[]; };\n\n",
               out_binding);

    /* One uniform int per unique index character */
    for (int i = 0; i < n_all; i++)
        sb_appendf(&sb, "uniform int size_%c;\n", all_chars[i]);
    sb_append(&sb, "\n");

    /* ── main() ── */
    sb_append(&sb, "void main() {\n");
    sb_append(&sb, "    int flat = int(gl_GlobalInvocationID.x);\n");

    if (n_free == 0) {
        /* Scalar output: only invocation 0 does work. */
        sb_append(&sb, "    if (flat != 0) return;\n\n");
    } else {
        /* Compute total_out = Π size_{free_chars[i]} */
        sb_append(&sb, "    int total_out = 1;\n");
        for (int i = 0; i < n_free; i++)
            sb_appendf(&sb, "    total_out *= size_%c;\n", free_chars[i]);
        sb_append(&sb, "    if (flat >= total_out) return;\n\n");

        /* Unpack flat → free multi-index (innermost = last char in out_idx). */
        sb_append(&sb, "    int tmp = flat;\n");
        for (int i = n_free - 1; i >= 0; i--)
            sb_appendf(&sb,
                "    int idx_%c = tmp %% size_%c; tmp /= size_%c;\n",
                free_chars[i], free_chars[i], free_chars[i]);
        sb_append(&sb, "\n");
    }

    /* ── Accumulation over contracted indices ── */
    sb_append(&sb, "    double acc = 0.0;\n");
    for (int i = 0; i < n_cont; i++)
        sb_appendf(&sb,
            "    for (int idx_%c = 0; idx_%c < size_%c; idx_%c++) {\n",
            cont_chars[i], cont_chars[i], cont_chars[i], cont_chars[i]);

    /* Flat index for left input A */
    sb_append(&sb, "        int idxA = 0;\n");
    for (int k = 0; left_idx[k]; k++) {
        char c = left_idx[k];
        if (k == 0)
            sb_appendf(&sb, "        idxA = idx_%c;\n", c);
        else
            sb_appendf(&sb, "        idxA = idxA * size_%c + idx_%c;\n", c, c);
    }

    if (!is_unary) {
        /* Flat index for right input B */
        sb_append(&sb, "        int idxB = 0;\n");
        for (int k = 0; right_idx[k]; k++) {
            char c = right_idx[k];
            if (k == 0)
                sb_appendf(&sb, "        idxB = idx_%c;\n", c);
            else
                sb_appendf(&sb, "        idxB = idxB * size_%c + idx_%c;\n", c, c);
        }
        sb_append(&sb, "        acc += dataA[idxA] * dataB[idxB];\n");
    } else {
        sb_append(&sb, "        acc += dataA[idxA];\n");
    }

    /* Close contracted-index loops */
    for (int i = 0; i < n_cont; i++)
        sb_append(&sb, "    }\n");
    sb_append(&sb, "\n");

    /* Write to output */
    if (n_free == 0) {
        sb_append(&sb, "    dataOut[0] = acc;\n");
    } else {
        sb_append(&sb, "    int idxOut = 0;\n");
        for (int k = 0; out_idx[k]; k++) {
            char c = out_idx[k];
            if (k == 0)
                sb_appendf(&sb, "    idxOut = idx_%c;\n", c);
            else
                sb_appendf(&sb, "    idxOut = idxOut * size_%c + idx_%c;\n", c, c);
        }
        sb_append(&sb, "    dataOut[idxOut] = acc;\n");
    }
    sb_append(&sb, "}\n");

    return sb.buf; /* caller owns */
}

/* =========================================================================
 * GLSL elementwise emit
 *
 * Generates a compute shader for pointwise binary/unary operations on flat
 * tensors of equal shape.  The single uniform `int total` gives the flat
 * element count; each invocation processes one element.
 * ========================================================================= */

/* Returns the GLSL expression string for operation `op` on operands a, b.
   `b` is ignored for unary ops (OP_NEGATE). */
static const char* elementwise_op_expr(Operation op) {
    switch (op) {
        case OP_ADD:      return "a + b";
        case OP_MULTIPLY: return "a * b";
        case OP_NEGATE:   return "-a";
        case OP_POW:      return "pow(a, b)";
        case OP_MIN:      return "min(a, b)";
        case OP_MAX:      return "max(a, b)";
        default:          return NULL;
    }
}

static int op_is_unary_elementwise(Operation op) {
    return op == OP_NEGATE;
}

static char* emit_compute_elementwise(Operation op) {
    const char *expr = elementwise_op_expr(op);
    if (!expr) return NULL;

    int is_unary = op_is_unary_elementwise(op);

    StrBuf sb; sb_init(&sb);
    sb_append(&sb, "#version 430 core\n");
    sb_append(&sb, "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n");
    sb_append(&sb, "layout(std430, binding = 0) readonly  buffer BufA   { double dataA[]; };\n");
    if (!is_unary)
        sb_append(&sb, "layout(std430, binding = 1) readonly  buffer BufB   { double dataB[]; };\n");

    int out_binding = is_unary ? 1 : 2;
    sb_appendf(&sb, "layout(std430, binding = %d) writeonly buffer BufOut { double dataOut[]; };\n\n",
               out_binding);
    sb_append(&sb, "uniform int total;\n\n");
    sb_append(&sb, "void main() {\n");
    sb_append(&sb, "    int i = int(gl_GlobalInvocationID.x);\n");
    sb_append(&sb, "    if (i >= total) return;\n");
    sb_appendf(&sb, "    double a = dataA[i];\n");
    if (!is_unary)
        sb_append(&sb, "    double b = dataB[i];\n");
    sb_appendf(&sb, "    dataOut[i] = %s;\n", expr);
    sb_append(&sb, "}\n");
    return sb.buf;
}

/* =========================================================================
 * gpu_compile_tensor_expr
 * ========================================================================= */

/* Build a ComputeKernel from already-emitted GLSL source and input names. */
static ComputeKernel* make_compute_kernel(char *glsl_src,
                                          const char **input_names, int n_inputs) {
    ComputeKernel *k = calloc(1, sizeof(ComputeKernel));
    k->glsl_src      = glsl_src; /* ownership transferred */
    k->n_inputs      = n_inputs;
    k->output_binding = n_inputs;
    k->workgroup_x   = 256;
    k->workgroup_y   = 1;
    k->workgroup_z   = 1;
    if (n_inputs > 0) {
        k->input_names = calloc(n_inputs, sizeof(char *));
        for (int i = 0; i < n_inputs; i++)
            k->input_names[i] = input_names[i] ? strdup(input_names[i]) : NULL;
    }
    return k;
}

static TensorProgram* make_tensor_program(ComputeKernel *k, GPUBackend backend,
                                           Operation op,
                                           const char *left_idx,
                                           const char *right_idx,
                                           const char *out_idx) {
    TensorProgram *p = calloc(1, sizeof(TensorProgram));
    p->kernels    = calloc(1, sizeof(ComputeKernel *));
    p->kernels[0] = k;
    p->n_kernels  = 1;
    p->backend    = backend;
    p->op         = op;
    p->left_indices  = left_idx  ? strdup(left_idx)  : NULL;
    p->right_indices = right_idx ? strdup(right_idx) : NULL;
    p->out_indices   = out_idx   ? strdup(out_idx)   : NULL;
    return p;
}

TensorProgram* gpu_compile_tensor_expr(Expression *expr, GPUBackend backend) {
    if (!expr) return NULL;

    /* ── Binary tensor ops ── */
    if (expr->type == EXPR_BINARY) {
        Operation op = expr->data.binary.op;

        /* Extract variable names from left and right if they are EXPR_VARIABLE;
           for the initial implementation we only support variable leaves.    */
        if (expr->data.binary.left  && expr->data.binary.left->type  != EXPR_VARIABLE) {
            fprintf(stderr, "[gpu_tensor] gpu_compile_tensor_expr: "
                    "only variable leaves supported (left is not EXPR_VARIABLE)\n");
            return NULL;
        }
        if (expr->data.binary.right && expr->data.binary.right->type != EXPR_VARIABLE) {
            fprintf(stderr, "[gpu_tensor] gpu_compile_tensor_expr: "
                    "only variable leaves supported (right is not EXPR_VARIABLE)\n");
            return NULL;
        }

        const char *lname = expr->data.binary.left  ? expr->data.binary.left->data.variable  : "A";
        const char *rname = expr->data.binary.right ? expr->data.binary.right->data.variable : "B";
        const char *names[2] = { lname, rname };

        if (op == OP_EINSUM) {
            IndexSpec *is = expr->data.binary.index_spec;
            if (!is) {
                fprintf(stderr, "[gpu_tensor] OP_EINSUM requires an IndexSpec\n");
                return NULL;
            }
            char *src = emit_compute_einsum(is->left_indices,
                                             is->right_indices,
                                             is->out_indices);
            if (!src) return NULL;
            ComputeKernel *k = make_compute_kernel(src, names, 2);
            return make_tensor_program(k, backend, op,
                                       is->left_indices,
                                       is->right_indices,
                                       is->out_indices);
        }

        if (op == OP_MATMUL) {
            /* Synthesise "ij,jk->ik" — standard 2D matrix multiply.
               For batched / higher-rank matmul, use OP_EINSUM directly.  */
            char *src = emit_compute_einsum("ij", "jk", "ik");
            if (!src) return NULL;
            ComputeKernel *k = make_compute_kernel(src, names, 2);
            return make_tensor_program(k, backend, op, "ij", "jk", "ik");
        }

        if (op == OP_DOT) {
            char *src = emit_compute_einsum("i", "i", "");
            if (!src) return NULL;
            ComputeKernel *k = make_compute_kernel(src, names, 2);
            return make_tensor_program(k, backend, op, "i", "i", "");
        }

        /* Elementwise binary ops */
        if (op == OP_ADD || op == OP_MULTIPLY ||
            op == OP_POW || op == OP_MIN || op == OP_MAX) {
            char *src = emit_compute_elementwise(op);
            if (!src) return NULL;
            ComputeKernel *k = make_compute_kernel(src, names, 2);
            return make_tensor_program(k, backend, op, NULL, NULL, NULL);
        }

        fprintf(stderr, "[gpu_tensor] gpu_compile_tensor_expr: "
                "unsupported binary op %d\n", (int)op);
        return NULL;
    }

    /* ── Unary tensor ops ── */
    if (expr->type == EXPR_UNARY) {
        Operation op = expr->data.unary.op;

        if (expr->data.unary.operand &&
            expr->data.unary.operand->type != EXPR_VARIABLE) {
            fprintf(stderr, "[gpu_tensor] gpu_compile_tensor_expr: "
                    "only variable leaves supported (operand is not EXPR_VARIABLE)\n");
            return NULL;
        }

        const char *oname = expr->data.unary.operand
                                ? expr->data.unary.operand->data.variable : "A";
        const char *names[1] = { oname };

        if (op == OP_TRANSPOSE) {
            /* Rank-2 transpose: "ij->ji".  For higher-rank permutations the
               caller should use OP_EINSUM with explicit subscripts.           */
            char *src = emit_compute_einsum("ij", NULL, "ji");
            if (!src) return NULL;
            ComputeKernel *k = make_compute_kernel(src, names, 1);
            return make_tensor_program(k, backend, op, "ij", NULL, "ji");
        }

        if (op == OP_NEGATE) {
            char *src = emit_compute_elementwise(op);
            if (!src) return NULL;
            ComputeKernel *k = make_compute_kernel(src, names, 1);
            return make_tensor_program(k, backend, op, NULL, NULL, NULL);
        }

        fprintf(stderr, "[gpu_tensor] gpu_compile_tensor_expr: "
                "unsupported unary op %d\n", (int)op);
        return NULL;
    }

    fprintf(stderr, "[gpu_tensor] gpu_compile_tensor_expr: "
            "expression must be EXPR_UNARY or EXPR_BINARY at the root\n");
    return NULL;
}

void tensor_program_free(TensorProgram *prog) {
    if (!prog) return;
    for (int i = 0; i < prog->n_kernels; i++) {
        ComputeKernel *k = prog->kernels[i];
        if (!k) continue;
        if (k->gl_program) { glDeleteProgram(k->gl_program); k->gl_program = 0; }
        if (k->glsl_src)   { free(k->glsl_src); k->glsl_src = NULL; }
        if (k->input_names) {
            for (int j = 0; j < k->n_inputs; j++)
                free(k->input_names[j]);
            free(k->input_names);
        }
        free(k);
    }
    free(prog->kernels);
    free(prog->left_indices);
    free(prog->right_indices);
    free(prog->out_indices);
    free(prog);
}

/* =========================================================================
 * Output shape inference
 * ========================================================================= */

/* Build a map (char → int size) from the subscripts + actual TensorField shapes.
   Populates out_chars[]/out_sizes[] and sets *out_n.
   Returns 0 on success, -1 if an index character is missing.               */
static int build_index_sizes(const char *left_idx,  const TensorField *A,
                              const char *right_idx, const TensorField *B,
                              char *out_chars, int *out_sizes, int *out_n) {
    *out_n = 0;

    /* From left input */
    if (left_idx && A) {
        for (int i = 0; left_idx[i] && i < A->rank; i++) {
            char c = left_idx[i];
            bool found = false;
            for (int j = 0; j < *out_n; j++)
                if (out_chars[j] == c) { found = true; break; }
            if (!found) {
                out_chars[*out_n] = c;
                out_sizes[*out_n] = A->shape[i];
                (*out_n)++;
            }
        }
    }

    /* From right input */
    if (right_idx && B) {
        for (int i = 0; right_idx[i] && i < B->rank; i++) {
            char c = right_idx[i];
            bool found = false;
            for (int j = 0; j < *out_n; j++)
                if (out_chars[j] == c) { found = true; break; }
            if (!found) {
                out_chars[*out_n] = c;
                out_sizes[*out_n] = B->shape[i];
                (*out_n)++;
            }
        }
    }

    return 0;
}

/* Infer EINSUM output shape from subscripts + input shapes.
   Fills out_shape[0..TENSOR_MAX_RANK-1] and *out_rank.
   Returns 0 on success, -1 on error.                                        */
static int infer_einsum_output_shape(const char *left_idx, const char *right_idx,
                                     const char *out_idx,
                                     const TensorField *A, const TensorField *B,
                                     int *out_shape, int *out_rank) {
    char chars[64]; int sizes[64]; int n = 0;
    if (build_index_sizes(left_idx, A, right_idx, B, chars, sizes, &n) != 0)
        return -1;

    int rank = (int)strlen(out_idx);
    if (rank > TENSOR_MAX_RANK) return -1;
    *out_rank = rank;

    for (int i = 0; i < rank; i++) {
        char c = out_idx[i];
        bool found = false;
        for (int j = 0; j < n; j++) {
            if (chars[j] == c) { out_shape[i] = sizes[j]; found = true; break; }
        }
        if (!found) {
            fprintf(stderr, "[gpu_tensor] infer_einsum_output_shape: "
                    "index '%c' not found in inputs\n", c);
            return -1;
        }
    }
    return 0;
}

/* =========================================================================
 * gpu_run_tensor_program
 * ========================================================================= */

TensorField* gpu_run_tensor_program(TensorProgram  *prog,
                                    const char    **input_names,
                                    TensorField   **input_tensors,
                                    int             n_inputs,
                                    GPUContext     *ctx) {
    if (!prog || prog->n_kernels == 0 || !ctx) return NULL;
    ComputeKernel *k = prog->kernels[0];
    if (!k || !k->glsl_src) return NULL;

    /* Ensure the GL context is ready */
    if (gpu_context_ensure(ctx) != 0) {
        fprintf(stderr, "[gpu_tensor] gpu_run_tensor_program: "
                "could not initialise GL context\n");
        return NULL;
    }

    /* Compile compute shader once */
    if (compute_kernel_ensure_program(k) != 0) return NULL;

    /* ── Look up the inputs named by the kernel ── */
    /* Map kernel input slots to caller-supplied TensorFields */
    TensorField *slot[COMPUTE_MAX_INPUTS] = {0};
    for (int ki = 0; ki < k->n_inputs && ki < COMPUTE_MAX_INPUTS; ki++) {
        for (int ci = 0; ci < n_inputs; ci++) {
            if (input_names[ci] &&
                k->input_names[ki] &&
                strcmp(input_names[ci], k->input_names[ki]) == 0) {
                slot[ki] = input_tensors[ci];
                break;
            }
        }
        if (!slot[ki]) {
            fprintf(stderr, "[gpu_tensor] gpu_run_tensor_program: "
                    "input '%s' not found in caller-supplied tensors\n",
                    k->input_names[ki]);
            return NULL;
        }
    }

    TensorField *A = (k->n_inputs >= 1) ? slot[0] : NULL;
    TensorField *B = (k->n_inputs >= 2) ? slot[1] : NULL;

    /* ── Infer output shape ── */
    int out_shape[TENSOR_MAX_RANK] = {0};
    int out_rank  = 0;

    if (prog->op == OP_EINSUM || prog->op == OP_MATMUL ||
        prog->op == OP_DOT    || prog->op == OP_TRANSPOSE) {
        const char *out_idx = prog->out_indices ? prog->out_indices : "";
        if (infer_einsum_output_shape(prog->left_indices, prog->right_indices,
                                      out_idx, A, B,
                                      out_shape, &out_rank) != 0) {
            fprintf(stderr, "[gpu_tensor] gpu_run_tensor_program: "
                    "could not infer output shape\n");
            return NULL;
        }
        /* Scalar output (e.g. DOT "i,i->") → rank-1 shape [1] so the
           TensorField always has at least one axis.                          */
        if (out_rank == 0) { out_rank = 1; out_shape[0] = 1; }
    } else {
        /* Elementwise: output has same shape as first input */
        if (!A) return NULL;
        out_rank = A->rank;
        for (int i = 0; i < A->rank; i++) out_shape[i] = A->shape[i];
    }

    /* ── Allocate output TensorField and its SSBO ── */
    TensorField *output = tensor_field_create(out_rank, out_shape);
    if (!output) return NULL;

    glGenBuffers(1, &output->ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output->ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(output->total * sizeof(double)),
                 NULL,        /* GPU-only initially; will be overwritten */
                 GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    output->gpu_dirty = false; /* SSBO holds authoritative value after dispatch */

    /* ── Upload dirty inputs ── */
    for (int i = 0; i < k->n_inputs && i < COMPUTE_MAX_INPUTS; i++)
        if (slot[i] && slot[i]->gpu_dirty) tensor_field_upload(slot[i]);

    /* ── Use the program and bind SSBOs ── */
    glUseProgram(k->gl_program);

    for (int i = 0; i < k->n_inputs && i < COMPUTE_MAX_INPUTS; i++) {
        if (slot[i] && slot[i]->ssbo)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, slot[i]->ssbo);
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, k->output_binding, output->ssbo);

    /* ── Set uniforms ── */
    if (prog->op == OP_EINSUM || prog->op == OP_MATMUL ||
        prog->op == OP_DOT    || prog->op == OP_TRANSPOSE) {
        /* Build char→size map and set size_X uniforms */
        char   chars[64]; int sizes[64]; int nc = 0;
        build_index_sizes(prog->left_indices, A, prog->right_indices, B,
                          chars, sizes, &nc);
        for (int i = 0; i < nc; i++) {
            char uname[16];
            snprintf(uname, sizeof(uname), "size_%c", chars[i]);
            GLint loc = glGetUniformLocation(k->gl_program, uname);
            if (loc >= 0) glUniform1i(loc, sizes[i]);
        }
    } else {
        /* Elementwise: set `total` uniform */
        GLint loc = glGetUniformLocation(k->gl_program, "total");
        if (loc >= 0) glUniform1i(loc, (GLint)output->total);
    }

    /* ── Dispatch ── */
    GLuint n_groups = ((GLuint)output->total + 255u) / 256u;
    if (n_groups == 0) n_groups = 1;
    glDispatchCompute(n_groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* ── Download result to CPU ── */
    tensor_field_download(output);

    glUseProgram(0);
    return output;
}
