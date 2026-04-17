/*
 * gpu_sim.c — implementation of the high-level GPU simulation binding layer
 * declared in gpu_sim.h.
 *
 * Depends on the internal helpers in gpu_compiler.c only via the public API
 * (kernel_ensure_program, the GPUContext struct, etc. are exposed through
 * gpu_compiler.h and the forward-declarations in gpu_sim.h).
 */

#include "../include/gpu_sim.h"
#include "../include/gpu_compiler.h"
#include "../include/boundary_gpu.h"

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── internal helpers ────────────────────────────────────────────────── */

static void draw_fullscreen_quad_sim(void) {
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();
}

/* ══════════════════════════════════════════════════════════════════════
   1. GPUTexDict
   ══════════════════════════════════════════════════════════════════════ */

static uint32_t tex_dict_hash(const char *key, uint32_t cap) {
    uint32_t h = 5381;
    for (const unsigned char *p = (const unsigned char *)key; *p; ++p)
        h = ((h << 5) + h) ^ *p;
    return h % cap;
}

GPUTexDict *gpu_tex_dict_create(uint32_t capacity) {
    if (capacity == 0) capacity = 16;
    GPUTexDict *d = calloc(1, sizeof(GPUTexDict));
    if (!d) return NULL;
    d->buckets  = calloc(capacity, sizeof(GPUTexEntry *));
    d->capacity = capacity;
    d->size     = 0;
    return d;
}

void gpu_tex_dict_set(GPUTexDict *d, const char *name, GLuintAlias tex_id) {
    if (!d || !name) return;
    uint32_t idx = tex_dict_hash(name, d->capacity);
    GPUTexEntry *e = d->buckets[idx];
    while (e) {
        if (strcmp(e->name, name) == 0) { e->tex_id = tex_id; return; }
        e = e->next;
    }
    /* new entry */
    GPUTexEntry *ne = calloc(1, sizeof(GPUTexEntry));
    ne->name   = strdup(name);
    ne->tex_id = tex_id;
    ne->next   = d->buckets[idx];
    d->buckets[idx] = ne;
    d->size++;
}

GLuintAlias gpu_tex_dict_get(const GPUTexDict *d, const char *name) {
    if (!d || !name) return 0;
    uint32_t idx = tex_dict_hash(name, d->capacity);
    for (GPUTexEntry *e = d->buckets[idx]; e; e = e->next)
        if (strcmp(e->name, name) == 0) return e->tex_id;
    return 0;
}

int gpu_tex_dict_remove(GPUTexDict *d, const char *name) {
    if (!d || !name) return 0;
    uint32_t idx = tex_dict_hash(name, d->capacity);
    GPUTexEntry **pp = &d->buckets[idx];
    while (*pp) {
        if (strcmp((*pp)->name, name) == 0) {
            GPUTexEntry *dead = *pp;
            *pp = dead->next;
            free(dead->name);
            free(dead);
            d->size--;
            return 1;
        }
        pp = &(*pp)->next;
    }
    return 0;
}

void gpu_tex_dict_free(GPUTexDict *d) {
    if (!d) return;
    for (uint32_t i = 0; i < d->capacity; ++i) {
        GPUTexEntry *e = d->buckets[i];
        while (e) {
            GPUTexEntry *next = e->next;
            free(e->name);
            free(e);
            e = next;
        }
    }
    free(d->buckets);
    free(d);
}


/* ══════════════════════════════════════════════════════════════════════
   2. GPUUniformBindingList
   ══════════════════════════════════════════════════════════════════════ */

GPUUniformBindingList *gpu_uniform_bindings_create(void) {
    GPUUniformBindingList *l = calloc(1, sizeof(GPUUniformBindingList));
    return l;
}

int gpu_uniform_binding_add(GPUUniformBindingList *list,
                            GLuintAlias             gl_program,
                            const char             *uniform_name,
                            GPUUniformType          type,
                            const void             *cpu_ptr) {
    if (!list || !uniform_name || !cpu_ptr) return -1;
    /* look up location immediately */
    GLint loc = glGetUniformLocation((GLuint)gl_program, uniform_name);
    /* loc == -1 means the uniform is not active — still store it so
       the binding list is the authoritative record; push will skip it. */

    if (list->n_bindings >= list->cap_bindings) {
        int ncap = list->cap_bindings ? list->cap_bindings * 2 : 8;
        GPUUniformBinding *nb = realloc(list->bindings,
                                         (size_t)ncap * sizeof(GPUUniformBinding));
        if (!nb) return -1;
        list->bindings     = nb;
        list->cap_bindings = ncap;
    }

    GPUUniformBinding *b = &list->bindings[list->n_bindings++];
    b->gl_program = gl_program;
    b->location   = (GLintAlias)loc;
    b->type       = type;
    b->cpu_ptr    = cpu_ptr;
    b->name       = strdup(uniform_name);
    return 0;
}

void gpu_uniform_bindings_push(const GPUUniformBindingList *list) {
    if (!list) return;
    GLuint current_prog = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint *)&current_prog);

    for (int i = 0; i < list->n_bindings; ++i) {
        const GPUUniformBinding *b = &list->bindings[i];
        if (b->location < 0 || !b->cpu_ptr) continue;

        /* switch program only when needed */
        if (current_prog != (GLuint)b->gl_program) {
            glUseProgram((GLuint)b->gl_program);
            current_prog = (GLuint)b->gl_program;
        }

        switch (b->type) {
            case GPU_UNIFORM_INT:
                glUniform1i(b->location, *(const int *)b->cpu_ptr);
                break;
            case GPU_UNIFORM_BOOL:
                glUniform1i(b->location, *(const int *)b->cpu_ptr ? 1 : 0);
                break;
            case GPU_UNIFORM_FLOAT:
                glUniform1f(b->location, *(const float *)b->cpu_ptr);
                break;
            case GPU_UNIFORM_DOUBLE:
                glUniform1f(b->location, (float)(*(const double *)b->cpu_ptr));
                break;
            case GPU_UNIFORM_IVEC2:
                glUniform2i(b->location,
                             ((const int *)b->cpu_ptr)[0],
                             ((const int *)b->cpu_ptr)[1]);
                break;
            case GPU_UNIFORM_VEC2:
                glUniform2f(b->location,
                             ((const float *)b->cpu_ptr)[0],
                             ((const float *)b->cpu_ptr)[1]);
                break;
        }
    }
}

void gpu_uniform_bindings_free(GPUUniformBindingList *list) {
    if (!list) return;
    for (int i = 0; i < list->n_bindings; ++i)
        if (list->bindings[i].name) free(list->bindings[i].name);
    free(list->bindings);
    free(list);
}


/* ══════════════════════════════════════════════════════════════════════
   3. GPURenderConfig
   ══════════════════════════════════════════════════════════════════════ */

GPURenderConfig *gpu_render_config_create(GPUProgram  *display_prog,
                                          GLuintAlias  gl_program) {
    GPURenderConfig *rc = calloc(1, sizeof(GPURenderConfig));
    if (!rc) return NULL;
    rc->gpu_prog   = display_prog;
    rc->gl_program = gl_program;
    rc->tex_dict   = gpu_tex_dict_create(16);
    rc->uniforms   = gpu_uniform_bindings_create();
    return rc;
}

void gpu_render_config_set_tex(GPURenderConfig *rc,
                               const char      *var_name,
                               GLuintAlias      tex_id) {
    if (!rc || !var_name) return;
    gpu_tex_dict_set(rc->tex_dict, var_name, tex_id);
}

int gpu_render_config_bind_uniform(GPURenderConfig *rc,
                                   const char      *uniform_name,
                                   GPUUniformType   type,
                                   const void      *cpu_ptr) {
    if (!rc) return -1;
    return gpu_uniform_binding_add(rc->uniforms, rc->gl_program,
                                   uniform_name, type, cpu_ptr);
}

/* Draw a fullscreen quad using the render config:
   - sets gl_program
   - binds each texture in tex_dict to consecutive units, sets sampler uniform
   - pushes all CPU→uniform bindings
   - draws quad
   Uniform locations are queried on the first call and cached inside the
   struct so subsequent draws have zero glGetUniformLocation overhead.   */
void gpu_render_config_draw(const GPURenderConfig *rc_const) {
    if (!rc_const || !rc_const->gl_program) return;
    /* Cast away const for internal cache mutation */
    GPURenderConfig *rc = (GPURenderConfig *)rc_const;

    glUseProgram((GLuint)rc->gl_program);

    /* ── One-time location cache population ──────────────────────────── */
    if (!rc->_locs_cached) {
        rc->_n_samplers   = 0;
        rc->_loc_vis_off  = glGetUniformLocation((GLuint)rc->gl_program, "vis_offset");
        rc->_loc_vis_size = glGetUniformLocation((GLuint)rc->gl_program, "vis_size");
        rc->_loc_dims     = glGetUniformLocation((GLuint)rc->gl_program, "dims");
        rc->_loc_spacing  = glGetUniformLocation((GLuint)rc->gl_program, "spacing");
        int unit = 0;
        if (rc->tex_dict) {
            for (uint32_t b = 0; b < rc->tex_dict->capacity && unit < 64; ++b) {
                for (GPUTexEntry *e = rc->tex_dict->buckets[b]; e && unit < 64; e = e->next) {
                    if (!e->tex_id) continue;
                    char uname[128];
                    size_t nlen = strlen(e->name);
                    if (nlen >= 4 && strcmp(e->name + nlen - 4, "_tex") == 0)
                        strncpy(uname, e->name, sizeof(uname) - 1);
                    else
                        snprintf(uname, sizeof(uname), "%s_tex", e->name);
                    uname[sizeof(uname)-1] = '\0';
                    GLint loc = glGetUniformLocation((GLuint)rc->gl_program, uname);
                    if (loc < 0) loc = glGetUniformLocation((GLuint)rc->gl_program, e->name);
                    rc->_sampler_locs[unit]   = (int)loc;
                    rc->_sampler_units[unit]  = unit;
                    rc->_sampler_texids[unit] = e->tex_id;
                    /* wire sampler uniform to its fixed unit once */
                    if (loc >= 0) glUniform1i(loc, unit);
                    unit++;
                }
            }
        }
        rc->_n_samplers  = unit;
        rc->_locs_cached = 1;
    }

    /* ── Per-frame: bind current textures (dict values may change each frame) */
    {
        int unit = 0;
        if (rc->tex_dict) {
            for (uint32_t b = 0; b < rc->tex_dict->capacity && unit < rc->_n_samplers; ++b) {
                for (GPUTexEntry *e = rc->tex_dict->buckets[b]; e && unit < rc->_n_samplers; e = e->next) {
                    if (!e->tex_id) { unit++; continue; }
                    glActiveTexture(GL_TEXTURE0 + unit);
                    glBindTexture(GL_TEXTURE_2D, (GLuint)e->tex_id);
                    unit++;
                }
            }
        }
    }

    /* ── Static uniforms ─────────────────────────────────────────────── */
    if (rc->vis_size_x > 0 || rc->vis_size_y > 0) {
        if (rc->_loc_vis_off  >= 0) glUniform2i(rc->_loc_vis_off,  rc->vis_offset_x, rc->vis_offset_y);
        if (rc->_loc_vis_size >= 0) glUniform2i(rc->_loc_vis_size, rc->vis_size_x,   rc->vis_size_y);
    }
    /* dims and spacing from GPUProgram's grid */
    if (rc->gpu_prog && rc->gpu_prog->grid) {
        if (rc->_loc_dims >= 0)
            glUniform2i(rc->_loc_dims,
                        (GLint)rc->gpu_prog->grid->dims[0],
                        (GLint)rc->gpu_prog->grid->dims[1]);
        if (rc->_loc_spacing >= 0)
            glUniform2f(rc->_loc_spacing,
                        (float)rc->gpu_prog->grid->spacing[0],
                        (float)rc->gpu_prog->grid->spacing[1]);
    }

    /* ── CPU→uniform bindings (sliders etc.) ────────────────────────── */
    if (rc->uniforms) gpu_uniform_bindings_push(rc->uniforms);

    draw_fullscreen_quad_sim();
}

void gpu_render_config_free(GPURenderConfig *rc) {
    if (!rc) return;
    gpu_tex_dict_free(rc->tex_dict);
    gpu_uniform_bindings_free(rc->uniforms);
    free(rc);
}


/* ══════════════════════════════════════════════════════════════════════
   4. Context adoption
   ══════════════════════════════════════════════════════════════════════ */

GPUContext *gpu_context_adopt(void *sdl_window,
                              void *sdl_gl_context,
                              int   width,
                              int   height) {
    GPUContext *ctx = gpu_context_create(GPU_BACKEND_OPENGL);
    if (!ctx) return NULL;

    /* Fill in the adopted SDL resources — requires access to GPUContext
       internals.  We use the dedicated setter declared in gpu_compiler.h. */
    gpu_context_adopt_sdl(ctx, sdl_window, sdl_gl_context, width, height);
    return ctx;
}


/* ══════════════════════════════════════════════════════════════════════
   5. GPU-to-GPU execution
   ══════════════════════════════════════════════════════════════════════ */

/* Internal: ensure k->gl_input_tex is allocated for n_inputs entries.  */
static void ensure_input_tex_array(ShaderKernel *k) {
    if (k->gl_input_tex) return;
    if (k->n_inputs <= 0) return;
    k->gl_input_tex = calloc((size_t)k->n_inputs, sizeof(unsigned int));
    k->n_gl_inputs  = k->n_inputs;
}

int gpu_run_program_to_tex(GPUProgram  *prog,
                           GPUTexDict  *tex_inputs,
                           GLuintAlias  output_tex,
                           GPUContext  *ctx) {
    if (!prog || prog->n_kernels == 0 || !ctx || !output_tex) return -1;
    ShaderKernel *k = prog->kernels[0];
    if (!k || !k->source) return -1;

    int w = (int)prog->grid->dims[0];
    int h = (int)prog->grid->dims[1];

    /* Compile the kernel shader once; subsequent calls reuse it. */
    if (gpu_kernel_ensure_program(k) != 0) return -1;

    /* Ensure input texture slot array exists */
    ensure_input_tex_array(k);

    /* Auto-generate and upload boundary mask if needed */
    if (!prog->boundary_mask && prog->grid && prog->grid->boundaries) {
        bool has_bc = false;
        for (int i = 0; i < prog->grid->n_dims * 2; ++i)
            if (prog->grid->boundaries[i].type != BC_OPEN) { has_bc = true; break; }
        if (has_bc) prog->boundary_mask = boundary_mask_create(prog->grid);
    }
    if (prog->boundary_mask &&
        (prog->boundary_mask->dirty || !prog->boundary_mask->uploaded)) {
        boundary_mask_upload(prog->boundary_mask, ctx);
    }

    /* Get the persistent FBO from ctx */
    GLuint fbo = gpu_context_get_fbo(ctx);
    if (!fbo) return -1;

    /* Attach the caller's output texture to the FBO */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, (GLuint)output_tex, 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "gpu_run_program_to_tex: FBO incomplete 0x%x\n", status);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return -1;
    }

    glViewport(0, 0, w, h);
    glUseProgram((GLuint)k->gl_program);

    /* Bind input textures: look up each variable name in tex_inputs.
       If the caller supplied a GPUTexDict, use it.
       Otherwise fall back to kernel's own cached textures (if any).    */
    for (int i = 0; i < k->n_inputs; ++i) {
        GLuint tid = 0;
        if (tex_inputs) tid = (GLuint)gpu_tex_dict_get(tex_inputs, k->inputs[i]);
        /* fall back to kernel-owned texture if available */
        if (!tid && k->gl_input_tex) tid = k->gl_input_tex[i];
        if (tid) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, tid);
        }
        if (k->uloc_inputs && k->uloc_inputs[i] >= 0)
            glUniform1i(k->uloc_inputs[i], i);
    }

    /* Bind boundary mask textures */
    int bm_unit = k->n_inputs;
    if (prog->boundary_mask) {
        if (prog->boundary_mask->mask_tex) {
            glActiveTexture(GL_TEXTURE0 + bm_unit);
            glBindTexture(GL_TEXTURE_2D, prog->boundary_mask->mask_tex);
            if (k->uloc_mask_tex >= 0) glUniform1i(k->uloc_mask_tex, bm_unit);
        }
        if (prog->boundary_mask->values_tex) {
            glActiveTexture(GL_TEXTURE0 + bm_unit + 1);
            glBindTexture(GL_TEXTURE_2D, prog->boundary_mask->values_tex);
            if (k->uloc_val_tex >= 0) glUniform1i(k->uloc_val_tex, bm_unit + 1);
        }
    }

    /* Grid uniforms */
    if (k->uloc_dims    >= 0) glUniform2i(k->uloc_dims, w, h);
    if (k->uloc_spacing >= 0)
        glUniform2f(k->uloc_spacing,
                    (float)prog->grid->spacing[0],
                    (float)prog->grid->spacing[1]);
    if (k->uloc_use_mask >= 0)
        glUniform1i(k->uloc_use_mask, prog->boundary_mask ? 1 : 0);

    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);   glLoadIdentity();
    glMatrixMode(GL_PROJECTION);  glLoadIdentity();

    draw_fullscreen_quad_sim();
    glFlush();

    /* Restore default FBO so subsequent display rendering goes to the window */
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}
