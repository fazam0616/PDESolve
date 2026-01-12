#ifndef GPU_COMPILER_H
#define GPU_COMPILER_H

#include "expression.h"
#include "grid.h"
#include "dictionary.h"
#include <stdbool.h>
#include <stddef.h>
// forward declare BoundaryMask to avoid circular include
struct BoundaryMask;

typedef enum { GPU_BACKEND_OPENGL, GPU_BACKEND_WEBGL, GPU_BACKEND_VULKAN, GPU_BACKEND_METAL } GPUBackend;

typedef struct {
    char *source;
    char **inputs;
    int n_inputs;
    char *output;
    bool can_fuse;
    int compute_intensity;
    /* optional CPU-side expression root for test execution */
    Expression *root;
    char **var_names;
    int n_var_names;
    /* GL runtime handles (set when compiled/executed on GL context) */
    unsigned int gl_program;       /* GL program object id */
    unsigned int gl_output_tex;    /* texture id for output */
    unsigned int *gl_input_tex;    /* array of texture ids for inputs */
    int n_gl_inputs;
    int tex_width, tex_height;     /* texture size matching grid dims (2D) */
    /* optional boundary mask to be applied by emitted shaders */
    struct BoundaryMask *boundary_mask;
} ShaderKernel;

typedef struct {
    ShaderKernel **kernels;
    int n_kernels;
    char **temp_buffers;
    int n_temp_buffers;
    GridMetadata *grid;
    GPUBackend backend;
    bool fused;
    bool use_shared_memory;
    /* optional boundary mask to be applied by emitted shaders */
    struct BoundaryMask *boundary_mask;
} GPUProgram;

typedef struct GPUContext GPUContext;

GPUProgram* gpu_compile_expression(Expression *expr, GridMetadata *grid, GPUBackend backend);
GPUProgram* gpu_compile_optimized(Expression *expr, GridMetadata *grid, GPUBackend backend);
void gpu_program_free(GPUProgram *prog);

GPUContext* gpu_context_create(GPUBackend backend);
int gpu_upload_field(GPUContext *ctx, GridField *field, int slot);
GridField* gpu_download_field(GPUContext *ctx, int slot, GridMetadata *grid);
int gpu_execute_program(GPUContext *ctx, GPUProgram *prog, Dictionary *inputs, int output_slot);
int gpu_execute_kernel(GPUContext *ctx, ShaderKernel *kernel, int *input_slots, int output_slot);
void gpu_context_free(GPUContext *ctx);

void gpu_print_program(GPUProgram *prog);
size_t gpu_estimate_memory(GPUProgram *prog);
double gpu_benchmark_kernel(GPUContext *ctx, ShaderKernel *kernel);

/* Run the compiled program on CPU (helper for testing correctness).
    Returns a newly allocated GridField with results (caller owns). */
GridField* gpu_run_program_cpu(GPUProgram *prog, Dictionary *inputs, GridMetadata *grid);

/* attach a BoundaryMask to a program so its textures are bound at runtime */
void gpu_program_set_boundary_mask(GPUProgram *prog, struct BoundaryMask *bm);

#endif // GPU_COMPILER_H
