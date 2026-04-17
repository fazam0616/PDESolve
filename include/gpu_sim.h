/*
 * gpu_sim.h — High-level GPU simulation binding layer
 *
 * Provides three coordinated abstractions:
 *
 *   1. GPUTexDict   — maps expression variable names to persistent GL texture
 *                     IDs owned by the caller.  Used to bind ping-pong field
 *                     textures into gpu_run_program_to_tex without any CPU
 *                     readback.
 *
 *   2. GPUUniformBinding — links a CPU-side scalar (a slider, a counter, …)
 *                     to a named GLSL uniform in a particular GL program.
 *                     Call gpu_uniform_bindings_push() each frame to flush
 *                     all bindings.
 *
 *   3. GPURenderConfig — bundles a compiled GPUProgram (for display/render
 *                     modes), its linked GL program ID, a set of
 *                     GPUUniformBindings, and the sampler→texture-unit
 *                     mapping so the caller needs one call per frame to
 *                     drive the whole render pass.
 *
 * Additionally declares the GPU context adoption API so interactive examples
 * can share their existing SDL/GL context with the compiler's kernel cache.
 */

#ifndef GPU_SIM_H
#define GPU_SIM_H

#include <stdint.h>
#include <stddef.h>

/* Include the compiler header for GPUContext, GPUProgram, ShaderKernel etc. */
#include "gpu_compiler.h"

/* GL type aliases — the translation unit must include glew.h / gl.h before
   calling any of these functions, but the header itself stays clean.        */
typedef unsigned int GLuintAlias;   /* same as GLuint  */
typedef int          GLintAlias;    /* same as GLint   */

/* ══════════════════════════════════════════════════════════════════════
   1. GPUTexDict — variable-name → GL texture handle dictionary
   ══════════════════════════════════════════════════════════════════════ */

typedef struct GPUTexEntry {
    char           *name;
    GLuintAlias     tex_id;
    struct GPUTexEntry *next;
} GPUTexEntry;

typedef struct {
    GPUTexEntry  **buckets;
    uint32_t       capacity;
    uint32_t       size;
} GPUTexDict;

GPUTexDict  *gpu_tex_dict_create(uint32_t capacity);
void         gpu_tex_dict_set(GPUTexDict *d, const char *name, GLuintAlias tex_id);
GLuintAlias  gpu_tex_dict_get(const GPUTexDict *d, const char *name);
int          gpu_tex_dict_remove(GPUTexDict *d, const char *name);
void         gpu_tex_dict_free(GPUTexDict *d);


/* ══════════════════════════════════════════════════════════════════════
   2. GPUUniformBinding — CPU scalar → GLSL uniform linkage
   ══════════════════════════════════════════════════════════════════════ */

typedef enum {
    GPU_UNIFORM_INT,      /* int *                         */
    GPU_UNIFORM_FLOAT,    /* float *                       */
    GPU_UNIFORM_DOUBLE,   /* double * (cast to float)      */
    GPU_UNIFORM_IVEC2,    /* int[2]                        */
    GPU_UNIFORM_VEC2,     /* float[2]                      */
    GPU_UNIFORM_BOOL,     /* int *; pushed as 0 or 1       */
} GPUUniformType;

typedef struct {
    GLuintAlias    gl_program;
    GLintAlias     location;
    GPUUniformType type;
    const void    *cpu_ptr;
    char          *name;
} GPUUniformBinding;

typedef struct {
    GPUUniformBinding *bindings;
    int                n_bindings;
    int                cap_bindings;
} GPUUniformBindingList;

GPUUniformBindingList *gpu_uniform_bindings_create(void);

/* Add a binding; `name` is copied; location is queried from gl_program immediately. */
int  gpu_uniform_binding_add(GPUUniformBindingList *list,
                              GLuintAlias            gl_program,
                              const char            *uniform_name,
                              GPUUniformType         type,
                              const void            *cpu_ptr);

/* Push all CPU values into their GL uniforms.  Call once per frame.   */
void gpu_uniform_bindings_push(const GPUUniformBindingList *list);

void gpu_uniform_bindings_free(GPUUniformBindingList *list);


/* ══════════════════════════════════════════════════════════════════════
   3. GPURenderConfig — fully-described display / render pass
   ══════════════════════════════════════════════════════════════════════ */

typedef struct {
    GPUProgram            *gpu_prog;
    GLuintAlias            gl_program;
    GPUTexDict            *tex_dict;
    GPUUniformBindingList *uniforms;
    /* Optional visible-subregion for vis_offset / vis_size uniforms.
       Set vis_size_x and vis_size_y to 0 to skip those uniforms.      */
    int vis_offset_x, vis_offset_y;
    int vis_size_x,   vis_size_y;
    /* Internal location cache — populated on first draw, reused after.
       Parallel to tex_dict iteration order (max 64 samplers).          */
    int  _locs_cached;             /* 0 = not yet cached                */
    int  _sampler_locs[64];        /* per-texture-dict-entry locations   */
    int  _sampler_units[64];       /* texture unit assigned to each entry*/
    GLuintAlias _sampler_texids[64]; /* tex id at cache time             */
    int  _n_samplers;              /* how many entries in cache          */
    int  _loc_vis_off;             /* vis_offset uniform location        */
    int  _loc_vis_size;            /* vis_size   uniform location        */
    int  _loc_dims;                /* dims       uniform location        */
    int  _loc_spacing;             /* spacing    uniform location        */
} GPURenderConfig;

GPURenderConfig *gpu_render_config_create(GPUProgram *display_prog,
                                          GLuintAlias gl_program);

/* Map sampler variable `var_name` → texture id `tex_id`.              */
void gpu_render_config_set_tex(GPURenderConfig *rc,
                               const char      *var_name,
                               GLuintAlias      tex_id);

/* Add a CPU-controlled uniform.                                         */
int  gpu_render_config_bind_uniform(GPURenderConfig *rc,
                                    const char      *uniform_name,
                                    GPUUniformType   type,
                                    const void      *cpu_ptr);

/* Execute the display pass: bind program, textures, push uniforms, draw quad. */
void gpu_render_config_draw(const GPURenderConfig *rc);

void gpu_render_config_free(GPURenderConfig *rc);


/* ══════════════════════════════════════════════════════════════════════
   4. Context adoption — share existing SDL/GL context
   ══════════════════════════════════════════════════════════════════════ */

/* Wrap an existing SDL_Window + SDL_GLContext in a GPUContext.
   The returned context is flagged `adopted`; gpu_context_free will not
   destroy the SDL objects.  Call this immediately after your own
   SDL_GL_CreateContext so kernel caching shares your context.          */
GPUContext *gpu_context_adopt(void *sdl_window,
                              void *sdl_gl_context,
                              int   width,
                              int   height);


/* ══════════════════════════════════════════════════════════════════════
   5. GPU-to-GPU execution — no CPU readback
   ══════════════════════════════════════════════════════════════════════ */

/* Run a compiled GPUProgram using caller-owned GL textures as inputs.
   `tex_inputs` maps expression variable names → live GL texture IDs.
   `output_tex` is an existing GL_RGBA32F texture that receives the result.
   No pixels are read back to the CPU.
   Returns 0 on success.                                                  */
int gpu_run_program_to_tex(GPUProgram  *prog,
                            GPUTexDict  *tex_inputs,
                            GLuintAlias  output_tex,
                            GPUContext  *ctx);

#endif /* GPU_SIM_H */
