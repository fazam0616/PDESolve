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
    /* Expressions owned/retained by this kernel (compiler retains refs)
       The GPUProgram is responsible for releasing these when freed. */
    Expression **owned_exprs;
    int n_owned_exprs;
    char **var_names;
    int n_var_names;
    /* GL runtime handles (set when compiled/executed on GL context) */
    unsigned int gl_program;       /* GL program object id (0 = not compiled yet) */
    unsigned int gl_output_tex;    /* texture id for output (0 = not allocated) */
    unsigned int *gl_input_tex;    /* array of texture ids for inputs (NULL = not allocated) */
    int n_gl_inputs;
    int tex_width, tex_height;     /* texture size matching grid dims (2D) */
    /* Cached uniform locations (populated once after link_program) */
    int uloc_dims;        /* location of 'dims' uniform (-1 = not present) */
    int uloc_spacing;     /* location of 'spacing' uniform */
    int uloc_use_mask;    /* location of 'use_mask' uniform */
    int uloc_mask_tex;    /* location of 'mask_tex' sampler uniform */
    int uloc_val_tex;     /* location of 'val_tex' sampler uniform */
    int *uloc_inputs;     /* per-input sampler uniform locations (k->n_inputs entries) */
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

/* GPUContext owns the persistent SDL window, GL context and shared FBO so
   resources can be reused across multiple gpu_run_program_cpu calls. */
typedef struct GPUContext GPUContext;

/* Definition of a render-mode: for each UI render index this struct contains
    up to three expressions (R,G,B) that should be evaluated by the emitted
    fragment shader when that render mode is active. The compiler will fuse
    all modes into a single monolithic shader that branches on a
    `render_mode` uniform at runtime. */
typedef struct {
     char *name; /* optional name for the mode (debug/labels) */
     Expression *chan_expr[3]; /* expressions for R, G, B channels (may be NULL)
                                          each evaluated in the shader when this mode is selected */
      /* Per-channel compile-time scale applied in addition to runtime `value_scale`.
          Use values < 1.0 to avoid saturating color ranges by default. */
      double chan_scale[3];
      /* Per-channel flag: after computing the channel expression, apply sqrt()
          to the channel value (useful for magnitude-of-square expressions). */
      int chan_apply_sqrt[3];
     /* Optional list of grid names that these expressions reference. This
         is informational for callers; the compiler discovers variable names
         automatically from expressions but callers can provide explicit
         grid bindings here if desired. Strings are owned by the caller. */
     char **grid_names;
     int n_grid_names;
} RenderModeDef;

/* Compile a set of render mode definitions into a single GPUProgram. The
    generated fragment shader will declare a uniform `int render_mode` and
    evaluate the corresponding channel expressions to fill the RGB channels.
    The returned GPUProgram follows the same runtime expectations as
    gpu_compile_expression (sampler uniforms named <var>_tex for variables
    referenced in the expressions). */
GPUProgram* gpu_compile_render_modes(RenderModeDef *modes, int n_modes, GridMetadata *grid, GPUBackend backend);

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

/* Run the compiled program using the provided GPUContext (which owns the
   persistent SDL/GL window and FBO).  Pass NULL to create and destroy a
   temporary context for this call only (legacy / test path).
   Returns a newly allocated GridField with results (caller owns). */
GridField* gpu_run_program_cpu(GPUProgram *prog, Dictionary *inputs, GridMetadata *grid);
GridField* gpu_run_program(GPUProgram *prog, Dictionary *inputs, GPUContext *ctx);

/* attach a BoundaryMask to a program so its textures are bound at runtime */
void gpu_program_set_boundary_mask(GPUProgram *prog, struct BoundaryMask *bm);

/* ── Internal helpers exposed for gpu_sim.c ─────────────────────────── */

/* Fill an already-created GPUContext with caller-owned SDL resources.
   Sets adopted=1 so gpu_context_free skips SDL teardown.
   Creates the shared FBO in the current GL context.                     */
void gpu_context_adopt_sdl(GPUContext *ctx,
                            void       *sdl_window,
                            void       *sdl_gl_context,
                            int         width,
                            int         height);

/* Return the persistent FBO stored in ctx (0 if not initialised).      */
unsigned int gpu_context_get_fbo(GPUContext *ctx);

/* Compile-and-link a kernel's GL program if not already done.
   Caches all uniform locations.  Returns 0 on success.                  */
int gpu_kernel_ensure_program(ShaderKernel *k);

#endif // GPU_COMPILER_H
