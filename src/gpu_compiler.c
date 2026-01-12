#include "../include/gpu_compiler.h"
#include "../include/gpu_compiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>
#include "../include/expression.h"
#include "../include/grid.h"
#include "../include/dictionary.h"
#include "../include/boundary_gpu.h"

// GPU-only runtime: emit minimal GLSL for pointwise expressions and execute
// using an offscreen SDL/OpenGL context. This removes the CPU-backed fallback
// so tests exercise the real GPU path.

// Small helpers for GL resources
static int gl_make_context_hidden(SDL_Window **out_win, SDL_GLContext *out_ctx, int w, int h) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return -1;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    *out_win = SDL_CreateWindow("offscreen", 0, 0, w, h, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    if (!*out_win) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }
    *out_ctx = SDL_GL_CreateContext(*out_win);
    if (!*out_ctx) {
        fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(*out_win);
        SDL_Quit();
        return -1;
    }
    // Initialize GLEW to load modern GL functions
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "glewInit failed: %s\n", glewGetErrorString(err));
        SDL_GL_DeleteContext(*out_ctx);
        SDL_DestroyWindow(*out_win);
        SDL_Quit();
        return -1;
    }
    return 0;
}

static void gl_destroy_context(SDL_Window *win, SDL_GLContext ctx) {
    if (ctx) SDL_GL_DeleteContext(ctx);
    if (win) SDL_DestroyWindow(win);
    SDL_Quit();
}

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "Shader compile error:\n%s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint link_program(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetProgramInfoLog(p, sizeof(log), NULL, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

// Create a simple fullscreen quad (legacy immediate mode used for simplicity)
static void draw_fullscreen_quad(void) {
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();
}

// Emit a fragment shader for a + 2*b like expressions. For now support:
// - variables: single-letter names mapped to sampler2D uniforms (a,b,c,...)
// - binary ops: add, multiply
// - scalar literals (double -> float constant)
// This is intentionally small to match current tests.
static char* emit_glsl_for_expr(Expression *expr, GridMetadata *grid, char ***out_var_list, int *out_nvars) {
    // Collect variable names by simple traversal (support variables only)
    char **vars = NULL; int nvars = 0;
    // helper
    void collect(Expression *e) {
        if (!e) return;
        if (e->type == EXPR_VARIABLE) {
            const char *name = e->data.variable;
            // add if missing
            int found = 0;
            for (int i=0;i<nvars;i++) if (strcmp(vars[i], name)==0) { found=1; break; }
            if (!found) {
                vars = realloc(vars, sizeof(char*)*(nvars+1));
                vars[nvars++] = strdup(name);
            }
        } else if (e->type == EXPR_UNARY) {
            collect(e->data.unary.operand);
        } else if (e->type == EXPR_BINARY) {
            collect(e->data.binary.left);
            collect(e->data.binary.right);
        } else if (e->type == EXPR_LITERAL) {
            // nothing
        }
    }
    collect(expr);

    // Build GLSL
    const char *vs_src = "void main() { gl_Position = gl_Vertex; gl_TexCoord[0] = gl_MultiTexCoord0; }";
    // fragment: sample variables and compute result
    size_t buf = 8192;
    char *fs = calloc(1, buf);
    strncat(fs, "#version 120\n", buf- strlen(fs)-1);
    strncat(fs, "uniform sampler2D tex0;\n", buf- strlen(fs)-1);
    // map each var to texN
    for (int i=0;i<nvars;i++) {
        char line[128];
        snprintf(line, sizeof(line), "uniform sampler2D %s_tex;\n", vars[i]);
        strncat(fs, line, buf- strlen(fs)-1);
    }
    // use standard texcoord ordering (s,t)
    // provide grid metadata to shader for stencil offsets
    strncat(fs, "uniform ivec2 dims;\n", buf- strlen(fs)-1);
    strncat(fs, "uniform vec2 spacing;\n", buf- strlen(fs)-1);
    // snap uv to texel centers: floor(uv * dims) + 0.5 -> center coord
    // allow optional boundary mask/value samplers (declare at global scope)
    strncat(fs, "uniform sampler2D mask_tex;\n", buf- strlen(fs)-1);
    strncat(fs, "uniform sampler2D val_tex;\n", buf- strlen(fs)-1);
     /* only sample mask/val when enabled by runtime to avoid accidental
         sampling of unbound samplers which may default to texture unit 0 */
     strncat(fs, "uniform int use_mask;\n", buf- strlen(fs)-1);
    strncat(fs, "void main() { vec2 uv = gl_TexCoord[0].st; uv = (floor(uv * vec2(dims)) + vec2(0.5)) / vec2(dims); float result = 0.0;\n", buf- strlen(fs)-1);

    // function to emit expression recursively
    char expr_buf[4096];
    expr_buf[0] = '\0';
    // return a string fragment representing expression
    char* emit_expr(Expression *e) {
        if (!e) return strdup("0.0");
        if (e->type == EXPR_LITERAL) {
            double v = e->data.literal->field ? e->data.literal->field[0] : 0.0;
            char *s = malloc(64);
            snprintf(s,64, "%g", v);
            return s;
        } else if (e->type == EXPR_VARIABLE) {
            const char *name = e->data.variable;
            char *s = malloc(64);
            snprintf(s,64, "texture2D(%s_tex, uv).r", name);
            return s;
        } else if (e->type == EXPR_UNARY) {
            if (e->data.unary.op == OP_DERIVATIVE) {
                Expression *op = e->data.unary.operand;
                const char *axis = e->data.unary.with_respect_to ? e->data.unary.with_respect_to : "x";
                if (op && op->type == EXPR_VARIABLE) {
                    const char *name = op->data.variable;
                    char *out = malloc(512);
                    if (strcmp(axis, "x") == 0 || strcmp(axis, "i") == 0) {
                        snprintf(out,512,"(texture2D(%s_tex, uv + vec2(1.0/float(dims.x),0)).r - texture2D(%s_tex, uv - vec2(1.0/float(dims.x),0)).r) / (2.0 * spacing.x)", name, name);
                    } else {
                        snprintf(out,512,"(texture2D(%s_tex, uv + vec2(0,1.0/float(dims.y))).r - texture2D(%s_tex, uv - vec2(0,1.0/float(dims.y))).r) / (2.0 * spacing.y)", name, name);
                    }
                    return out;
                }
            } else if (e->data.unary.op == OP_LAPLACIAN) {
                Expression *op = e->data.unary.operand;
                if (op && op->type == EXPR_VARIABLE) {
                    const char *name = op->data.variable;
                    char *out = malloc(1024);
                    snprintf(out,1024,
                        "( (texture2D(%s_tex, uv + vec2(1.0/float(dims.x),0)).r - 2.0*texture2D(%s_tex, uv).r + texture2D(%s_tex, uv - vec2(1.0/float(dims.x),0)).r) / (spacing.x*spacing.x) ) + "
                        "( (texture2D(%s_tex, uv + vec2(0,1.0/float(dims.y))).r - 2.0*texture2D(%s_tex, uv).r + texture2D(%s_tex, uv - vec2(0,1.0/float(dims.y))).r) / (spacing.y*spacing.y) )",
                        name, name, name, name, name, name);
                    return out;
                }
            } else if (e->data.unary.op == OP_NEGATE) {
                char *sub = emit_expr(e->data.unary.operand);
                size_t need = strlen(sub) + 8; /* allow for (-() ) and NUL */
                char *out = malloc(need);
                if (out) snprintf(out, need, "(-(%s))", sub);
                free(sub);
                return out;
            }
        } else if (e->type == EXPR_BINARY) {
            char *L = emit_expr(e->data.binary.left);
            char *R = emit_expr(e->data.binary.right);
            size_t need = strlen(L) + strlen(R) + 64;
            char *out = malloc(need);
            if (out) {
                if (e->data.binary.op == OP_ADD) snprintf(out, need, "((%s) + (%s))", L, R);
                else if (e->data.binary.op == OP_MULTIPLY) snprintf(out, need, "((%s) * (%s))", L, R);
                else snprintf(out, need, "(0.0)");
            }
            free(L); free(R);
            return out;
        }
        return strdup("0.0");
    }

    char *body = emit_expr(expr);
    strncat(fs, " result = ", buf- strlen(fs)-1);
    strncat(fs, body, buf- strlen(fs)-1);
    free(body);
    // if mask indicates a boundary point, override with val_tex; only sample
    // mask/val when runtime sets use_mask to non-zero to avoid sampling
    // uninitialized samplers.
    strncat(fs, "; if (use_mask != 0) { float m = texture2D(mask_tex, uv).r; if (m > 0.5) result = texture2D(val_tex, uv).r; } gl_FragColor = vec4(result, 0.0, 0.0, 0.0); }\n", buf- strlen(fs)-1);

    *out_var_list = vars;
    *out_nvars = nvars;
    // Note: caller will compile shaders with vs_src and fs
    char *full = malloc(strlen(vs_src)+strlen(fs)+1);
    strcpy(full, fs); // return fragment source only (vertex source kept local)
    free(fs);
    return full;
}

GPUProgram* gpu_compile_expression(Expression *expr, GridMetadata *grid, GPUBackend backend) {
    if (!expr || !grid) return NULL;
    GPUProgram *p = calloc(1, sizeof(GPUProgram));
    p->grid = grid;
    p->backend = backend;
    p->n_kernels = 1;
    p->kernels = calloc(1, sizeof(ShaderKernel*));

    ShaderKernel *k = calloc(1, sizeof(ShaderKernel));
    k->root = expr; // keep pointer for codegen

    // Emit GLSL fragment for this expression
    char **var_list = NULL; int nvars = 0;
    char *fs_src = emit_glsl_for_expr(expr, grid, &var_list, &nvars);
    k->source = fs_src;
    k->n_inputs = nvars;
    if (nvars > 0) {
        k->inputs = calloc(nvars, sizeof(char*));
        for (int i=0;i<nvars;i++) k->inputs[i] = strdup(var_list[i]);
    } else {
        k->inputs = NULL;
    }
    k->n_var_names = nvars;
    k->var_names = k->inputs ? calloc(nvars, sizeof(char*)) : NULL;
    for (int i=0;i<nvars;i++) k->var_names[i] = strdup(k->inputs[i]);

    p->kernels[0] = k;
    p->fused = true;
    p->boundary_mask = NULL;
    return p;
}

GPUProgram* gpu_compile_optimized(Expression *expr, GridMetadata *grid, GPUBackend backend) {
    return gpu_compile_expression(expr, grid, backend);
}

void gpu_program_free(GPUProgram *prog) {
    if (!prog) return;
    for (int i = 0; i < prog->n_kernels; ++i) {
        ShaderKernel *k = prog->kernels[i];
        if (!k) continue;
        if (k->source) free(k->source);
        if (k->inputs) {
            for (int j = 0; j < k->n_inputs; ++j) if (k->inputs[j]) free(k->inputs[j]);
            free(k->inputs);
        }
        if (k->var_names) {
            for (int j = 0; j < k->n_var_names; ++j) if (k->var_names[j]) free(k->var_names[j]);
            free(k->var_names);
        }
        free(k);
    }
    free(prog->kernels);
    free(prog);
}

struct GPUContext { int backend; };

GPUContext* gpu_context_create(GPUBackend backend) {
    GPUContext *ctx = calloc(1, sizeof(GPUContext));
    ctx->backend = (int)backend;
    return ctx;
}

// Upload a GridField into an RGBA float texture; returns texture id (caller owns)
static GLuint upload_field_as_texture(const GridField *field) {
    uint32_t w = field->grid->dims[0];
    uint32_t h = field->grid->dims[1];
    float *buf = calloc(w * h * 4, sizeof(float));
    // OpenGL expects image data rows starting from the bottom. The grid
    // metadata uses j=0 as the top row. Literal storage stores elements with
    // offset = i * h + j (i is the slowest index), so read using that
    // convention and write into the GL buffer as row-major (j * w + i).
    const Literal *lit = &field->data;
    for (uint32_t j_gl = 0; j_gl < h; ++j_gl) {
        uint32_t src_j = h - 1 - j_gl; // grid row corresponding to this GL row
        for (uint32_t i = 0; i < w; ++i) {
            double v = 0.0;
            if (lit && lit->field) {
                size_t off = (size_t)i * h + src_j; // literal offset (i major)
                v = lit->field[off];
            }
            size_t base = ((size_t)j_gl * w + i) * 4; // GL expects row-major
            buf[base+0] = (float)v;
            buf[base+1] = 0.0f; buf[base+2] = 0.0f; buf[base+3] = 0.0f;
        }
    }
    GLuint tex; glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Clamp to edge to avoid sampling wrap-around at boundaries when doing stencils
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, buf);
    free(buf);
    return tex;
}

// Read back the current GL color buffer into a new GridField (R channel)
static GridField* readback_to_field(GridMetadata *grid) {
    uint32_t w = grid->dims[0];
    uint32_t h = grid->dims[1];
    float *buf = calloc(w * h * 4, sizeof(float));
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, buf);
    GridField *f = grid_field_create(grid);
    Literal *lit = &f->data;
    // ensure lit->field allocated and shape set
    // assume grid_field_create zeroes the literal if needed
    // glReadPixels returns rows with the bottom row first. Map that into the
    // grid which expects j=0 as the top row by flipping vertically when
    // writing into the grid field.
    for (uint32_t j = 0; j < h; ++j) {
        uint32_t dst_j = h - 1 - j; // destination row in grid
        for (uint32_t i = 0; i < w; ++i) {
            size_t base = ((size_t)j * w + i) * 4;
            double v = buf[base+0];
            uint32_t idx[3] = {i,dst_j,0};
            literal_set(&f->data, idx, v);
        }
    }
    free(buf);
    return f;
}

// Main GPU-run: compile shader, upload inputs, render to FBO, readback result
GridField* gpu_run_program_cpu(GPUProgram *prog, Dictionary *inputs, GridMetadata *grid) {
    (void)grid; // use prog->grid
    if (!prog || prog->n_kernels == 0) return NULL;
    ShaderKernel *k = prog->kernels[0];
    if (!k || !k->source) return NULL;

    int w = prog->grid->dims[0];
    int h = prog->grid->dims[1];
    SDL_Window *win = NULL; SDL_GLContext ctx;
    if (gl_make_context_hidden(&win, &ctx, w, h) != 0) return NULL;
    // Auto-generate boundary mask if any input GridField's grid has boundaries
    if (!prog->boundary_mask) {
    // look through provided inputs dictionary for GridField literals
    // if any grid has non-default boundaries or interior boundaries, create mask
    // Note: this is a conservative heuristic to attach masks automatically
    // Instead, if program has a grid with boundaries defined, auto-generate
        GridMetadata *g = prog->grid;
        if (g && g->boundaries) {
            // check if any edge boundary is Dirichlet or any interior boundary exists
            bool has_bc = false;
            for (int i = 0; i < g->n_dims * 2; ++i) {
                if (g->boundaries[i].type != BC_OPEN) { has_bc = true; break; }
            }
            if (!has_bc && g->n_interior_boundaries > 0) has_bc = true;
            if (has_bc) {
                BoundaryMask *bm = boundary_mask_create(g);
                prog->boundary_mask = bm;
            }
        }
    }

    // If program has an attached boundary mask, ensure its GL textures are created
    if (prog->boundary_mask) {
        // boundary_mask_upload will create GL textures using the current context
        boundary_mask_upload(prog->boundary_mask, NULL);
    }

    // generated fragment shader is available in k->source (debug prints removed)

    // compile shaders
    const char *vs_src = "void main() { gl_Position = gl_Vertex; gl_TexCoord[0] = gl_MultiTexCoord0; }";
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, k->source);
    if (!vs || !fs) { gl_destroy_context(win, ctx); return NULL; }
    GLuint prog_gl = link_program(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    if (!prog_gl) { gl_destroy_context(win, ctx); return NULL; }

    // create textures for inputs from dictionary
    GLuint *texs = calloc(k->n_inputs, sizeof(GLuint));
    for (int i=0;i<k->n_inputs;i++) {
        Literal *lit = NULL;
        dict_get(inputs, k->inputs[i], &lit);
        if (!lit) { texs[i]=0; continue; }
    /* debug prints removed: upload happens below */
        // wrap literal as GridField for upload helper
        GridField tmp = { .name = NULL, .grid = prog->grid, .data = *lit };
        texs[i] = upload_field_as_texture(&tmp);
    }

    /* Readback-of-upload debug removed to avoid noisy output in normal runs */

    // create FBO and output texture
    GLuint out_tex; glGenTextures(1, &out_tex);
    glBindTexture(GL_TEXTURE_2D, out_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Ensure output texture also clamps at edges
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
    GLuint fbo; glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_tex, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "FBO incomplete: 0x%x\n", status);
        // cleanup
        for (int i=0;i<k->n_inputs;i++) if (texs[i]) glDeleteTextures(1, &texs[i]);
        free(texs);
        glDeleteTextures(1, &out_tex);
        glDeleteFramebuffers(1, &fbo);
        glDeleteProgram(prog_gl);
        gl_destroy_context(win, ctx);
        return NULL;
    }

    glViewport(0, 0, w, h);
    glUseProgram(prog_gl);

    // bind input textures to texture units and set uniform samplers
    for (int i=0;i<k->n_inputs;i++) {
        GLenum unit = GL_TEXTURE0 + i;
        glActiveTexture(unit);
        glBindTexture(GL_TEXTURE_2D, texs[i]);
        // try to set uniform by name: <var>_tex
        char uname[128]; snprintf(uname, sizeof(uname), "%s_tex", k->inputs[i]);
        GLint loc = glGetUniformLocation(prog_gl, uname);
        if (loc >= 0) {
            glUniform1i(loc, i);
        }
    }

    // If caller attached a BoundaryMask to the program, bind its textures to following units
    int bm_unit = k->n_inputs; // next free texture unit
    if (prog->boundary_mask) {
        // mask_tex -> unit bm_unit, val_tex -> unit bm_unit+1
        if (prog->boundary_mask->mask_tex) {
            glActiveTexture(GL_TEXTURE0 + bm_unit);
            glBindTexture(GL_TEXTURE_2D, prog->boundary_mask->mask_tex);
            GLint locm = glGetUniformLocation(prog_gl, "mask_tex"); if (locm >= 0) glUniform1i(locm, bm_unit);
        }
        if (prog->boundary_mask->values_tex) {
            glActiveTexture(GL_TEXTURE0 + bm_unit + 1);
            glBindTexture(GL_TEXTURE_2D, prog->boundary_mask->values_tex);
            GLint locv = glGetUniformLocation(prog_gl, "val_tex"); if (locv >= 0) glUniform1i(locv, bm_unit+1);
        }
    }

    // set grid uniforms if present
    GLint loc_dims = glGetUniformLocation(prog_gl, "dims");
    if (loc_dims >= 0) {
        glUniform2i(loc_dims, prog->grid->dims[0], prog->grid->dims[1]);
    }
    GLint loc_spacing = glGetUniformLocation(prog_gl, "spacing");
    if (loc_spacing >= 0) {
        glUniform2f(loc_spacing, (float)prog->grid->spacing[0], (float)prog->grid->spacing[1]);
    }
    // indicate whether boundary mask sampling should be active
    GLint loc_use_mask = glGetUniformLocation(prog_gl, "use_mask");
    if (loc_use_mask >= 0) {
        if (prog->boundary_mask) glUniform1i(loc_use_mask, 1);
        else glUniform1i(loc_use_mask, 0);
    }

    // draw into FBO
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity(); glMatrixMode(GL_PROJECTION); glLoadIdentity();
    draw_fullscreen_quad();
    glFlush();

    // read back into GridField
    GridField *out = readback_to_field(prog->grid);

    // cleanup
    for (int i=0;i<k->n_inputs;i++) if (texs[i]) { glDeleteTextures(1, &texs[i]); }
    free(texs);
    glDeleteTextures(1, &out_tex);
    glDeleteFramebuffers(1, &fbo);
    glDeleteProgram(prog_gl);
    gl_destroy_context(win, ctx);

    return out;
}

int gpu_execute_program(GPUContext *ctx, GPUProgram *prog, Dictionary *inputs, int output_slot) {
    (void)ctx; (void)output_slot;
    GridField *out = gpu_run_program_cpu(prog, inputs, prog->grid);
    if (!out) return 1;
    grid_field_free(out);
    return 0;
}

int gpu_execute_kernel(GPUContext *ctx, ShaderKernel *kernel, int *input_slots, int output_slot) {
    (void)ctx; (void)kernel; (void)input_slots; (void)output_slot; return 0;
}

void gpu_context_free(GPUContext *ctx) { free(ctx); }

void gpu_program_set_boundary_mask(GPUProgram *prog, struct BoundaryMask *bm) {
    if (!prog) return;
    if (!prog->kernels || prog->n_kernels == 0) return;
    // attach to program; runtime will bind textures when present
    prog->boundary_mask = bm;
}

void gpu_print_program(GPUProgram *prog) {
    if (!prog) { printf("GPUProgram: NULL\n"); return; }
    printf("GPUProgram: %d kernels\n", prog->n_kernels);
    for (int i = 0; i < prog->n_kernels; ++i) {
        ShaderKernel *k = prog->kernels[i];
        printf("---- kernel %d source ----\n%s\n", i, k->source ? k->source : "(null)");
    }
}

size_t gpu_estimate_memory(GPUProgram *prog) {
    size_t mem = 0;
    if (!prog) return 0;
    for (int i = 0; i < prog->n_kernels; ++i) if (prog->kernels[i] && prog->kernels[i]->source) mem += strlen(prog->kernels[i]->source);
    return mem;
}

double gpu_benchmark_kernel(GPUContext *ctx, ShaderKernel *kernel) {
    (void)ctx; (void)kernel; return 0.0;
}
