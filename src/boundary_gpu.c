#include "../include/boundary_gpu.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include "../include/grid.h"
#include "../include/gpu_compiler.h"

// Forward declaration
static void populate_axis_aligned(BoundaryMask *bm);

// Note: This file implements a minimal mask-based BC apply path.
// It assumes an active GL context (created by gpu_run_program_cpu). For
// simplicity we provide functions that expect the caller to have created
// a GPUContext and to call upload/apply while the context is current.

BoundaryMask* boundary_mask_create(GridMetadata *grid) {
    BoundaryMask *bm = calloc(1, sizeof(BoundaryMask));
    bm->grid = grid;
    uint32_t total = grid_get_total_points(grid);
    bm->mask = calloc(total, sizeof(uint8_t));
    bm->values = calloc(total, sizeof(double));
    bm->types = calloc(total, sizeof(int));
    bm->priority = calloc(total, sizeof(int));
    bm->mask_tex = 0; bm->values_tex = 0;
    // Populate axis-aligned mask values immediately so CPU-side code that
    // creates a BoundaryMask can use it without requiring a GL context.
    populate_axis_aligned(bm);
    return bm;
}

// Fill mask and values for axis-aligned faces using GridMetadata->boundaries
static void populate_axis_aligned(BoundaryMask *bm) {
    GridMetadata *g = bm->grid;
    if (!g || !g->boundaries) return;
    uint32_t nx = g->dims[0], ny = g->dims[1];
    for (int axis = 0; axis < g->n_dims && axis < 2; ++axis) {
        for (int side = 0; side < 2; ++side) {
            BoundarySpec *spec = &g->boundaries[axis*2 + side];
            if (!spec) continue;
            // Only populate mask for Dirichlet boundaries (prototype behavior)
            if (spec->type != BC_DIRICHLET) continue;
            // iterate over grid points and mark boundary face
            // axis-priority: lower axis index wins (0 has higher priority than 1)
            int incoming_priority = axis; // lower is higher priority
            for (uint32_t j = 0; j < ny; ++j) {
                for (uint32_t i = 0; i < nx; ++i) {
                    if ((axis == 0 && ((side==0 && i==0) || (side==1 && i==nx-1))) ||
                        (axis == 1 && ((side==0 && j==0) || (side==1 && j==ny-1)))) {
                        size_t off = (size_t)i * ny + j;
                        // only accept dirichlet
                        if (spec->type != BC_DIRICHLET) continue;
                        // if mask not set yet, or incoming has higher priority (lower value)
                        if (!bm->mask[off] || incoming_priority < bm->priority[off]) {
                            bm->mask[off] = 1;
                            bm->types[off] = spec->type;
                            bm->values[off] = spec->value;
                            bm->priority[off] = incoming_priority;
                        }
                    }
                }
            }
        }
    }
}

static GLuint create_texture_from_mask(BoundaryMask *bm) {
    uint32_t nx = bm->grid->dims[0]; uint32_t ny = bm->grid->dims[1];
    unsigned char *buf = calloc(nx * ny * 4, 1);
    for (uint32_t j = 0; j < ny; ++j) {
        for (uint32_t i = 0; i < nx; ++i) {
            /* Use direct (i,j) ordering so texture upload matches shader UV coordinates (no Y-flip) */
            size_t off = (size_t)i * ny + j;
            uint8_t m = bm->mask[off];
            size_t idx = ((size_t)j * nx + i) * 4;
            buf[idx+0] = m ? 255 : 0;
            buf[idx+1] = buf[idx+2] = buf[idx+3] = 0;
        }
    }
    GLuint tex; glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nx, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, buf);
    free(buf);
    return tex;
}

static GLuint create_texture_from_values(BoundaryMask *bm) {
    uint32_t nx = bm->grid->dims[0]; uint32_t ny = bm->grid->dims[1];
    float *buf = calloc(nx * ny * 4, sizeof(float));
    for (uint32_t j = 0; j < ny; ++j) {
        for (uint32_t i = 0; i < nx; ++i) {
            /* Use direct (i,j) ordering so texture upload matches shader UV coordinates (no Y-flip) */
            size_t off = (size_t)i * ny + j;
            double v = bm->values[off];
            size_t idx = ((size_t)j * nx + i) * 4;
            buf[idx+0] = (float)v;
        }
    }
    GLuint tex; glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, nx, ny, 0, GL_RGBA, GL_FLOAT, buf);
    free(buf);
    return tex;
}

int boundary_mask_upload(BoundaryMask *bm, GPUContext *ctx) {
    (void)ctx;
    if (!bm) return 1;
    // populate axis-aligned boundaries if not already set
    populate_axis_aligned(bm);
    if (bm->mask_tex) glDeleteTextures(1, &bm->mask_tex);
    if (bm->values_tex) glDeleteTextures(1, &bm->values_tex);
    bm->mask_tex = create_texture_from_mask(bm);
    bm->values_tex = create_texture_from_values(bm);
    return 0;
}

int boundary_mask_apply_cpu(BoundaryMask *bm, GridField *field) {
    if (!bm || !field) return 1;
    GridMetadata *g = bm->grid;
    uint32_t nx = g->dims[0], ny = g->dims[1];
    uint32_t idx[3]; idx[2]=0;
    for (uint32_t j = 0; j < ny; ++j) {
        for (uint32_t i = 0; i < nx; ++i) {
            size_t off = (size_t)i * ny + j;
            if (bm->mask[off]) {
                idx[0]=i; idx[1]=j;
                Literal *lit = literal_create_scalar(bm->values[off]);
                grid_field_set(field, idx, lit);
                literal_free(lit);
            }
        }
    }
    // debug: print x-min column and y-max row values
    fprintf(stderr, "boundary_mask_apply_cpu: nx=%u ny=%u\n", nx, ny);
    for (uint32_t j = 0; j < ny; ++j) {
        idx[0]=0; idx[1]=j; Literal *l = grid_field_get(field, idx);
        double v = (l && l->field) ? l->field[0] : 0.0;
        fprintf(stderr, "applied x-min at j=%u -> %g\n", j, v);
    }
    for (uint32_t i = 0; i < nx; ++i) {
        idx[0]=i; idx[1]=ny-1; Literal *l = grid_field_get(field, idx);
        double v = (l && l->field) ? l->field[0] : 0.0;
        fprintf(stderr, "applied y-max at i=%u -> %g\n", i, v);
    }
    return 0;
}

// Apply the mask by running a simple fragment shader: if mask!=0 use value, else use input
int boundary_mask_apply(BoundaryMask *bm, GPUContext *ctx, unsigned int input_tex) {
    (void)ctx; (void)input_tex;
    if (!bm) return 1;
    const char *vs = "void main() { gl_Position = gl_Vertex; gl_TexCoord[0] = gl_MultiTexCoord0; }";
    const char *fs_fmt = "#version 120\nuniform sampler2D input_tex; uniform sampler2D mask_tex; uniform sampler2D val_tex; void main() { vec2 uv = gl_TexCoord[0].st; float m = texture2D(mask_tex, uv).r; vec4 inv = texture2D(input_tex, uv); if (m > 0.5) gl_FragColor = texture2D(val_tex, uv); else gl_FragColor = inv; }";

    GLuint vs_s = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs_s, 1, &vs, NULL); glCompileShader(vs_s);
    GLuint fs_s = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs_s, 1, &fs_fmt, NULL); glCompileShader(fs_s);
    GLuint prog = glCreateProgram(); glAttachShader(prog, vs_s); glAttachShader(prog, fs_s); glLinkProgram(prog);
    glUseProgram(prog);

    // bind textures: mask -> unit 0, values -> unit 1, input -> unit 2
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, bm->mask_tex);
    GLint locm = glGetUniformLocation(prog, "mask_tex"); if (locm>=0) glUniform1i(locm, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, bm->values_tex);
    GLint locv = glGetUniformLocation(prog, "val_tex"); if (locv>=0) glUniform1i(locv, 1);
    // bind provided input texture to unit 2
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, input_tex);
    GLint loci = glGetUniformLocation(prog, "input_tex"); if (loci>=0) glUniform1i(loci, 2);

    // draw fullscreen quad into currently bound framebuffer
    glMatrixMode(GL_MODELVIEW); glLoadIdentity(); glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2f(-1,-1);
    glTexCoord2f(1,0); glVertex2f(1,-1);
    glTexCoord2f(1,1); glVertex2f(1,1);
    glTexCoord2f(0,1); glVertex2f(-1,1);
    glEnd(); glFlush();

    glDeleteProgram(prog); glDeleteShader(vs_s); glDeleteShader(fs_s);
    return 0;
}

void boundary_mask_update_time(BoundaryMask *bm, double t) { (void)bm; (void)t; }

void boundary_mask_free(BoundaryMask *bm) {
    if (!bm) return;
    if (bm->mask_tex) glDeleteTextures(1, &bm->mask_tex);
    if (bm->values_tex) glDeleteTextures(1, &bm->values_tex);
    free(bm->mask); free(bm->values); free(bm->types); free(bm->priority); free(bm);
}
