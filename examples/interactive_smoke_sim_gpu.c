/*
 * GPU-resident Interactive Wave Simulator (simplified)
 * - Keeps field data and update evaluation on the GPU (ping-pong textures)
 * - Renders current height field to the SDL/OpenGL window
 * - No mouse interaction yet; initial condition includes a barrier and a source
 * - Uses the project's GLSL emitter (via gpu_compile_expression) to build
 *   the fragment shader performing the wave update on the GPU.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

#include "../include/gpu_compiler.h"
#include "../include/grid.h"
#include "../include/expression.h"
#include "../include/calculus.h"
#include "../include/boundary_gpu.h"
#include "../include/Menu.h"

// Simple helpers: compile/link shaders
static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048]; glGetShaderInfoLog(s, sizeof(log), NULL, log);
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
        char log[2048]; glGetProgramInfoLog(p, sizeof(log), NULL, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

// Create an RGBA32F texture and upload provided single-channel float data
static GLuint create_texture_from_field(const double *field, uint32_t nx, uint32_t ny) {
    float *buf = calloc((size_t)nx * ny * 4, sizeof(float));
        for (uint32_t j = 0; j < ny; ++j) {
            uint32_t src_j = ny - 1 - j; // flip vertically for GL
            for (uint32_t i = 0; i < nx; ++i) {
                    size_t off = (size_t)src_j * nx + i; // Corrected indexing
                double v = field ? field[off] : 0.0;
                size_t idx = ((size_t)j * nx + i) * 4;
                    buf[idx+0] = (float)v; // Apply value_scale to the texture
                buf[idx+1] = 0.0f; buf[idx+2] = 0.0f; buf[idx+3] = 0.0f;
        }
    }
    GLuint tex; glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, nx, ny, 0, GL_RGBA, GL_FLOAT, buf);
    free(buf);
    return tex;
}

// Create an empty RGBA32F texture with uninitialized contents (for render target)
static GLuint create_empty_texture(uint32_t nx, uint32_t ny) {
    GLuint tex; glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, nx, ny, 0, GL_RGBA, GL_FLOAT, NULL);
    return tex;
}

// Fullscreen quad (legacy immediate mode)
static void draw_fullscreen_quad(void) {
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f); // Bottom left
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);  // Bottom right
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);   // Top right
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);  // Top left
    glEnd();
}

// Paint helper: add a Gaussian into a CPU-side RGBA float buffer matching the texture layout
static void paint_gaussian_to_rgba(float *buf, uint32_t nx, uint32_t ny,
                                   uint32_t gi, uint32_t gj,
                                   double amp, double sigma,
                                   double spacing_x, double spacing_y) {
    if (!buf) return;
    // compute spatial coordinates of center
    double cx = gi * spacing_x;
    double cy = gj * spacing_y;
    // radius in grid units ~ 3 sigma
    double rmax = 3.0 * sigma;
    int di = (int)ceil(rmax / spacing_x);
    int dj = (int)ceil(rmax / spacing_y);
    for (int djr = -dj; djr <= dj; ++djr) {
        int y = (int)gj + djr;
        if (y < 0 || y >= (int)ny) continue;
        for (int dir = -di; dir <= di; ++dir) {
            int x = (int)gi + dir;
            if (x < 0 || x >= (int)nx) continue;
            double xx = x * spacing_x;
            double yy = y * spacing_y;
            double dx = xx - cx; double dy = yy - cy;
            double r2 = dx*dx + dy*dy;
            double val = amp * exp(-r2 / (2.0 * sigma * sigma));
            // destination texture buffer uses row-major with j as top->bottom (same as create_texture_from_field dest j loop)
            // But create_texture_from_field used dest j as 0..ny-1 and set src_j = ny-1-j from data source. Here we write into dest layout directly.
            int jdest = (int)y; // destination row index
            size_t idx = ((size_t)jdest * nx + (size_t)x) * 4;
            buf[idx + 0] += (float)val; // Accumulate values for painting
        }
    }
}

// Paint helper that targets a specific channel (0=R/smoke,1=G/pressure,2=B,3=A)
static void paint_gaussian_to_rgba_channel(float *buf, uint32_t nx, uint32_t ny,
                                           uint32_t gi, uint32_t gj,
                                           double amp, double sigma,
                                           double spacing_x, double spacing_y,
                                           int channel) {
    if (!buf) return; if (channel < 0 || channel > 3) return;
    double cx = gi * spacing_x;
    double cy = gj * spacing_y;
    double rmax = 3.0 * sigma;
    int di = (int)ceil(rmax / spacing_x);
    int dj = (int)ceil(rmax / spacing_y);
    for (int djr = -dj; djr <= dj; ++djr) {
        int y = (int)gj + djr;
        if (y < 0 || y >= (int)ny) continue;
        for (int dir = -di; dir <= di; ++dir) {
            int x = (int)gi + dir;
            if (x < 0 || x >= (int)nx) continue;
            double xx = x * spacing_x;
            double yy = y * spacing_y;
            double dx = xx - cx; double dy = yy - cy;
            double r2 = dx*dx + dy*dy;
            double val = amp * exp(-r2 / (2.0 * sigma * sigma));
            int jdest = (int)y;
            size_t idx = ((size_t)jdest * nx + (size_t)x) * 4;
                                            buf[idx + channel] += (float)val * 10.0; // Scale the value for the specific channel
        }
    }
}

/* Toggle to enable per-stage debug readbacks. Set to 1 to print min/max/NaN counts after each stage. */
static int debug_per_stage = 1;

// Simple render / app state used by the menu callbacks
typedef enum { RENDER_VELOCITY = 1, RENDER_PRESSURE = 2, RENDER_SMOKE = 3, RENDER_VORTICITY = 4, RENDER_DIVERGENCE = 5, RENDER_TEMPERATURE = 6 } AppRenderMode;
typedef struct {
    AppRenderMode mode;
    double value_scale;
    int show_boundaries;
    int show_stats;
    int mode_velocity;
    int mode_pressure;
    int mode_smoke;
    int mode_vorticity;
    int mode_divergence;
    int mode_temperature;
    double mode_slider; /* UI slider to select render mode (1..n), kept as double for slider storage */
} AppRenderState;

// Minimal app state for menu callbacks (CPU-side only; simulation actions are no-ops)
typedef struct {
    int paused;
    double wave_speed;
    double dt;
    double max_sim_speed;
    double steps_per_frame;
    double wave_amplitude;
    double wave_spread;
    double default_source_frequency;
    double default_source_phase;
    // GUI toggles
    int mouse_none;
    int mouse_add_wave;
    int mouse_add_barrier;
    int mouse_add_barrier_brush;
    int mouse_source;
    // Menu visibility
    int show_base_menu;
    int show_mouse_controls;
    int show_sim_controls;
    int limit_fps;
    /* UI-only dummies */
    int dummy_reset;
    int dummy_clear_sources;
    int dummy_clear_barriers;
    /* Pressure solver / smoke controls */
    double sor_alpha;    /* SOR relaxation parameter (omega) */
    double sor_iters;       /* number of red-black SOR iterations per step */
    double pressure_width; /* width multiplier used in pressure update formula */
    /* Physical params */
    double viscosity_nu;   /* kinematic viscosity */
    double gravity_x;      /* gravity vector x */
    double gravity_y;      /* gravity vector y */
    double temp_kappa;     /* thermal diffusivity */
    double visc_iters;     /* implicit viscosity solver iterations (per step) */
    /* Buoyancy coefficient (beta) controlling strength of buoyant forcing */
    double buoyancy_beta;
} AppState;

// Source representation (grid coords)
typedef struct {
    int gx, gy;           // grid indices
    double amp;
    double freq;
    double phase;
    double radius;        // in grid units
    int selected;
    int target; /* 0=smoke, 1=pressure */
} Source;
    // Scaling factors to map mouse paint controls -> source defaults (make sources smaller)
#define SOURCE_RADIUS_SCALE 0.5
#define SOURCE_AMP_SCALE 0.5
// Additional scale to map mouse amplitude slider -> actual paint amplitude (make mouse additions smaller)
#define MOUSE_AMPLITUDE_SCALE 1.0
    /* trailing-tab label handled inside create_app_menus where menus are constructed */

// Helper macros to map window coords (mx,my) to grid indices (gix,gjy) using visible region and sponge
#define WIN_TO_GX(mx, win_w, nx_vis, sponge) ((int)floor((double)(mx) / (double)(win_w) * (double)(nx_vis)) + (sponge))
/* Map window Y to grid Y with vertical flip (match GL texture upload orientation) */
#define WIN_TO_GY(my, win_h, ny_vis, sponge) ((int)floor((1.0 - (double)(my) / (double)(win_h)) * (double)(ny_vis)) + (sponge))
// Grid -> window pixel mapping for overlay drawing (match WIN_TO_GY flip)
#define GX_TO_WINX(gx, nx_vis, sponge, win_w) (((double)(gx) - (double)(sponge)) / (double)(nx_vis) * (double)(win_w))
#define GY_TO_WINY(gy, ny_vis, sponge, win_h) ((1.0 - (((double)(gy) - (double)(sponge)) / (double)(ny_vis))) * (double)(win_h))

// Callback prototypes for source menu
static void cb_delete_selected_source(VariableInteraction *vi, void *user_data);
static void cb_deselect_selected_source(VariableInteraction *vi, void *user_data);

// Global source storage (main will keep these updated)
static Source *g_sources = NULL;
static int g_n_sources = 0;
static int g_selected_source = -1;
/* Global boundary mask pointer and grid dims so menu callbacks can access them */
static BoundaryMask *g_bm = NULL;
static uint32_t g_grid_nx = 0, g_grid_ny = 0;
/* GPU-side source descriptor texture and dirty flag (file-scope so callbacks can set it) */
static GLuint tex_src_desc = 0;
static int sources_dirty = 1; /* mark true to upload initial data */
/* Static upload buffer for source descriptors (64 cols x 2 rows x RGBA) */
static float src_desc_buf[64 * 2 * 4];
/* Paint buffers/textures moved to file-scope so callbacks (reset) can clear them */
static float *paint_buf = NULL;
static GLuint tex_paint = 0;
static GLuint tex_paint_cpu = 0;
static int paint_pending = 0;
static int paint_from_gpu = 0;
static int paint_buf_dirty = 0;
/* True when tex_paint_cpu currently holds non-zero CPU paint (so we must clear it on composite) */
static int paint_cpu_uploaded = 0;
/* GPU compute timing (measure occasionally with glFinish to avoid constant stalls) */
static int gpu_time_sample_period = 30; /* measure once every N compute steps */
static int gpu_time_step_counter = 0;    /* counts compute steps since last sample */
static double gpu_compute_ms_avg = 0.0;  /* exponential moving average of ms per compute step */
static int gpu_time_samples = 0;

/* GPU solver/field textures and programs (file-scope for lifetime management) */
static GLuint tex_pressure = 0;      /* pressure texture (RGBA32F.r stores p) */
static GLuint tex_pressure_tmp = 0;  /* ping-pong for pressure iterations */
static GLuint tex_divergence = 0;    /* divergence (rhs) texture */
static GLuint tex_velocity = 0;      /* velocity packed in RG channels */
static GLuint tex_velocity_tmp = 0;  /* temp for advected velocity */
static GLuint tex_velocity_rhs = 0;  /* RHS copy for implicit viscosity solver */
static GLuint tex_smoke = 0;         /* advected smoke scalar in R */
static GLuint tex_smoke_tmp = 0;     /* temp for smoke advection */
static GLuint tex_temp = 0;          /* temperature scalar (R) */
static GLuint tex_temp_tmp = 0;      /* temp for temperature advection */

/* helper single-channel velocity textures (vx, vy stored in R channel) */
static GLuint tex_velocity_x = 0, tex_velocity_y = 0;
static GLuint tex_velocity_tmp_x = 0, tex_velocity_tmp_y = 0;

static GLuint prog_divergence = 0;   /* computes divergence from velocity */
static GLuint prog_project = 0;      /* subtract pressure gradient from velocity */
static GLuint prog_jacobi = 0;       /* Jacobi / red-black SOR update for pressure */
static GLuint prog_advect_velocity = 0; /* advect velocity field */
static GLuint prog_advect_smoke = 0; /* advect smoke scalar */
static GLuint prog_mask_velocity = 0; /* mask velocity with barrier (zero orthogonal components) */
static GLuint prog_mask_smoke = 0;    /* mask smoke scalar at/near barriers */
static GLuint prog_advect_temp = 0; /* advect temperature scalar */
static GLuint prog_buoyancy = 0;   /* add buoyancy force to velocity */
static GLuint prog_viscosity = 0;  /* kept for compatibility but unused (old explicit) */
static GLuint prog_visc_jacobi = 0; /* implicit viscosity solver (red-black Jacobi) */
static GLuint prog_diffuse_temp = 0; /* diffusion for temperature */
/* small helper programs: extract components and pack components into RG */
static GLuint prog_extract_vx = 0, prog_extract_vy = 0, prog_pack_velocity = 0;
/* projection programs produced from expressions: compute vx_new (R) and vy_new (R) */
static GLuint prog_proj_vx = 0, prog_proj_vy = 0;
static GLuint prog_add_paint_smoke = 0; /* composite paint -> smoke */
static GLuint prog_add_paint_pressure = 0; /* composite paint -> pressure */
static GLuint prog_add_paint_temp = 0; /* composite paint -> temperature */

/* Uniform locations cache */
static GLint loc_proj_pressure = -1, loc_proj_vel = -1, loc_proj_dims = -1, loc_proj_spacing = -1;
static GLint loc_div_vel = -1, loc_div_dims = -1, loc_div_spacing = -1;
/* when using compiler-emitted divergence that samples components separately */
static GLint loc_div_vx = -1, loc_div_vy = -1;
static GLint loc_jacobi_p = -1, loc_jacobi_b = -1, loc_jacobi_color = -1, loc_jacobi_dims = -1, loc_jacobi_spacing = -1, loc_jacobi_alpha = -1, loc_jacobi_width = -1;
static GLint loc_advect_vel = -1, loc_advect_smoke = -1;
static GLint loc_advect_smoke_vel = -1; /* sampler location for vel_tex in smoke advect */
static GLint loc_advect_smoke_mask = -1; /* sampler location for mask_tex in smoke advect */
static GLint loc_advect_vel_dims = -1, loc_advect_vel_dt = -1, loc_advect_vel_spacing = -1;
static GLint loc_advect_smoke_dims = -1, loc_advect_smoke_dt = -1, loc_advect_smoke_spacing = -1;
static GLint loc_advect_temp = -1, loc_advect_temp_vel = -1, loc_advect_temp_mask = -1;
static GLint loc_advect_temp_dims = -1, loc_advect_temp_dt = -1, loc_advect_temp_spacing = -1;
static GLint loc_buoyancy_temp = -1, loc_buoyancy_vel = -1, loc_buoyancy_beta = -1, loc_buoyancy_T0 = -1, loc_buoyancy_dt = -1, loc_buoyancy_gravity = -1;
static GLint loc_viscosity_nu = -1, loc_viscosity_dt = -1, loc_viscosity_dims = -1, loc_viscosity_spacing = -1;
static GLint loc_visc_jacobi_vel = -1, loc_visc_jacobi_rhs = -1, loc_visc_jacobi_dims = -1, loc_visc_jacobi_spacing = -1, loc_visc_jacobi_nu = -1, loc_visc_jacobi_dt = -1, loc_visc_jacobi_color = -1, loc_visc_jacobi_use_mask = -1;
static GLint loc_diffuse_temp_k = -1, loc_diffuse_temp_dt = -1, loc_diffuse_temp_dims = -1, loc_diffuse_temp_spacing = -1;
static GLint loc_mask_prevvel = -1, loc_mask_mask = -1, loc_mask_dims = -1;
static GLint loc_mask_smoke_prev = -1, loc_mask_smoke_mask = -1, loc_mask_smoke_dims = -1;
static GLint loc_add_paint_target = -1, loc_add_paint_tex = -1, loc_add_paint_cpu = -1;


// Controls mirrored into the source menu for the currently-selected source
    static double sel_src_amp = 0.0;
static double sel_src_freq = 1.0;
static double sel_src_phase = 0.0;
static double sel_src_radius = 6.0; // in grid units
    static double sel_src_target = 0.0; /* slider-backed: 0=smoke,1=pressure,2=temperature */
// menu action booleans for source menu buttons
static int sel_src_deselect = 0;
static int sel_src_delete = 0;

// Callback to write controls back into the selected source
static void cb_source_control_changed(VariableInteraction *vi, void *user_data) {
    (void)vi; (void)user_data;
    if (g_selected_source >= 0 && g_sources && g_selected_source < g_n_sources) {
        g_sources[g_selected_source].amp = sel_src_amp;
        g_sources[g_selected_source].freq = sel_src_freq;
        g_sources[g_selected_source].phase = sel_src_phase;
        g_sources[g_selected_source].radius = sel_src_radius;
            g_sources[g_selected_source].target = (int)lround(sel_src_target); /* allow 0/1/2 via rounded slider */
    sources_dirty = 1;
    }
}

static void on_src_target_slider_change(VariableInteraction *vi, void *user_data) {
    if (!vi) return;
    double val = *(double*)vi->variable;
    int mode = (int)lround(val);
    if (mode < 0) mode = 0; if (mode > 2) mode = 2;
    const char *names[] = { "Smoke", "Pressure", "Temperature" };
    if (vi->name) free(vi->name);
    size_t nb = strlen("Target: ") + strlen(names[mode]) + 1;
    vi->name = (char*)malloc(nb);
    snprintf(vi->name, nb, "Target: %s", names[mode]);
    fprintf(stderr, "[DBG] on_src_target_slider_change called: val=%f mode=%d name=%s\n", val, mode, vi->name); fflush(stderr);
    /* also update mirror variable (already reflected in vi->variable) */
}

static void cb_delete_selected_source(VariableInteraction *vi, void *user_data) {
    (void)vi; (void)user_data;
    if (g_selected_source >= 0 && g_sources) {
        int idx = g_selected_source;
        for (int k = idx; k + 1 < g_n_sources; ++k) g_sources[k] = g_sources[k+1];
        g_n_sources--;
        if (g_n_sources > 0) g_sources = realloc(g_sources, sizeof(Source) * g_n_sources);
        else { free(g_sources); g_sources = NULL; }
        g_selected_source = -1;
    sources_dirty = 1;
    }
    // clear action button state so menu shows unpressed
    sel_src_delete = 0;
}

static void cb_deselect_selected_source(VariableInteraction *vi, void *user_data) { (void)vi; (void)user_data; g_selected_source = -1; sel_src_deselect = 0; }

// Pointer to menus for rendering/input handling
typedef struct { Menu *base_menu; Menu *mouse_menu; Menu *sim_menu; Menu *source_menu; } AppMenus;

// Callback prototypes
static void cb_reset(VariableInteraction *vi, void *user_data);
static void cb_clear_sources(VariableInteraction *vi, void *user_data);
static void cb_clear_barriers(VariableInteraction *vi, void *user_data);
// forward-declare wave speed change callback so create_app_menus can reference it
static void cb_wave_speed_changed(VariableInteraction *vi, void *user_data);
static void cb_dt_changed(VariableInteraction *vi, void *user_data);

// Data passed to reset callback: hold pointers to the texture variables so
// the callback re-uploads into the currently-used textures (after ping-pong swaps).
typedef struct reset_cb_data {
    GLuint *tex_curr;    // pointer to the tex_u_curr variable
    GLuint *tex_prev;    // pointer to the tex_u_prev variable
    GLuint *tex_out;     // pointer to the tex_out variable
    const double *data_curr; // pointer to initial u_curr data (grid-order)
    const double *data_prev; // pointer to initial u_prev data (usually zeros)
    uint32_t nx, ny;
    uint32_t last_ms;
    /* pointers so callbacks can recompile/update compute program */
    GPUProgram **gpu_prog_ptr;
    GLuint *compute_prog_ptr;
    Expression **wave_expr_ptr;
    GridMetadata *grid_ptr;
    double dt_val;
    uint64_t *sim_counter_ptr; /* pointer to simulation step counter so reset can zero it */
    double *app_wave_speed_ptr; /* pointer to app.wave_speed so dt changes can recompile with current c */
} reset_cb_data_t;
static void on_mouse_mode_change(VariableInteraction *vi, void *user_data) {
    if (!vi || !user_data) return;
    AppState *app = (AppState*)user_data;
    /* radio behavior: clear all then set selected */
    app->mouse_none = app->mouse_add_wave = app->mouse_add_barrier = app->mouse_source = 0;
    if (vi->variable == &app->mouse_none) app->mouse_none = 1;
    else if (vi->variable == &app->mouse_add_wave) app->mouse_add_wave = 1;
    else if (vi->variable == &app->mouse_add_barrier) app->mouse_add_barrier = 1;
    else if (vi->variable == &app->mouse_source) app->mouse_source = 1;
}

static void on_render_mode_change(VariableInteraction *vi, void *user_data) {
    if (!vi || !user_data) return;
    AppRenderState *r = (AppRenderState*)user_data;
    r->mode_velocity = r->mode_pressure = r->mode_smoke = r->mode_vorticity = r->mode_divergence = 0;
    if (vi->variable == &r->mode_velocity) { r->mode_velocity = 1; r->mode = RENDER_VELOCITY; }
    else if (vi->variable == &r->mode_pressure) { r->mode_pressure = 1; r->mode = RENDER_PRESSURE; }
    else if (vi->variable == &r->mode_smoke) { r->mode_smoke = 1; r->mode = RENDER_SMOKE; }
    else if (vi->variable == &r->mode_vorticity) { r->mode_vorticity = 1; r->mode = RENDER_VORTICITY; }
    else if (vi->variable == &r->mode_divergence) { r->mode_divergence = 1; r->mode = RENDER_DIVERGENCE; }
    else if (vi->variable == &r->mode_temperature) { r->mode_temperature = 1; r->mode = RENDER_TEMPERATURE; }
    // mask-debug mode removed
}

/* Slider callback: update render.mode from integer slider and update the slider label to show selected mode */
static void on_render_mode_slider_change(VariableInteraction *vi, void *user_data) {
    if (!vi || !user_data) return;
    AppRenderState *r = (AppRenderState*)user_data;
    double val = *(double*)vi->variable;
    int mode = (int)round(val);
    if (mode < RENDER_VELOCITY) mode = RENDER_VELOCITY;
    if (mode > RENDER_TEMPERATURE) mode = RENDER_TEMPERATURE;
    r->mode = (AppRenderMode)mode;
    /* update per-mode flags for compatibility with any code checking them */
    r->mode_velocity = (r->mode == RENDER_VELOCITY);
    r->mode_pressure = (r->mode == RENDER_PRESSURE);
    r->mode_smoke = (r->mode == RENDER_SMOKE);
    r->mode_vorticity = (r->mode == RENDER_VORTICITY);
    r->mode_divergence = (r->mode == RENDER_DIVERGENCE);
    r->mode_temperature = (r->mode == RENDER_TEMPERATURE);
    /* programmatically update the interaction's name to include the selected mode */
    const char *names[] = { "", "Velocity", "Pressure", "Smoke", "Vorticity", "Divergence", "Temperature" };
    if (vi->name) free(vi->name);
    size_t nb = strlen("Mode: ") + strlen(names[mode]) + 1;
    vi->name = (char*)malloc(nb);
    snprintf(vi->name, nb, "Mode: %s", names[mode]);
}

// Ensure mouse and sim control menus are mutually exclusive when toggled
static void cb_toggle_mouse_menu(VariableInteraction *vi, void *user_data) {
    if (!vi || !user_data) return;
    AppState *app = (AppState*)user_data;
    /* If enabling mouse menu, disable sim menu */
    if (vi->variable == &app->show_mouse_controls) {
        if (*(int*)vi->variable) app->show_sim_controls = 0;
    }
}

static void cb_toggle_sim_menu(VariableInteraction *vi, void *user_data) {
    if (!vi || !user_data) return;
    AppState *app = (AppState*)user_data;
    /* If enabling sim menu, disable mouse menu */
    if (vi->variable == &app->show_sim_controls) {
        if (*(int*)vi->variable) app->show_mouse_controls = 0;
    }
}

static void cycle_mouse_mode(AppState *app, int dir) {
    /* Add an extra mode for a circular barrier brush */
    int modes[5] = { app->mouse_none, app->mouse_add_wave, app->mouse_add_barrier, app->mouse_add_barrier_brush, app->mouse_source };
    int idx = 0;
    for (int i = 0; i < 5; ++i) if (modes[i]) { idx = i; break; }
    idx += dir; if (idx < 0) idx = 4; if (idx > 4) idx = 0;
    app->mouse_none = app->mouse_add_wave = app->mouse_add_barrier = app->mouse_add_barrier_brush = app->mouse_source = 0;
    if (idx == 0) app->mouse_none = 1;
    else if (idx == 1) app->mouse_add_wave = 1;
    else if (idx == 2) app->mouse_add_barrier = 1;
    else if (idx == 3) app->mouse_add_barrier_brush = 1;
    else if (idx == 4) app->mouse_source = 1;
}

// Create menus (mirrors interactive_wave_sim_menu.inc layout but lightweight)
static AppMenus *create_app_menus(AppState *app, AppRenderState *render, void *reset_cb_data) {
    AppMenus *m = calloc(1, sizeof(AppMenus));
    Color textColor = {255,255,255,255};
    Color bgColor = {30,30,40,220};

    Color baseBg = {50,50,60,220};
    m->base_menu = menu_create(10,10,250,150,1,"Controls", textColor, baseBg);
    MenuRow *r1 = menurow_create();
    menurow_add_interaction(r1, variableinteraction_create(&app->show_mouse_controls, "Mouse Controls", 0, 1, VAR_BOOL, cb_toggle_mouse_menu, app));
    menu_add_row(m->base_menu, r1);
    MenuRow *r2 = menurow_create();
    menurow_add_interaction(r2, variableinteraction_create(&app->show_sim_controls, "Sim Controls", 0, 1, VAR_BOOL, cb_toggle_sim_menu, app));
    menu_add_row(m->base_menu, r2);

    // Mouse menu
    // mouse controls: radio-like behavior handled by callbacks
    m->mouse_menu = menu_create(270,10,250,230,1,"Mouse Controls", textColor, bgColor);
    MenuRow *mrow1 = menurow_create();
    menurow_add_interaction(mrow1, variableinteraction_create(&app->mouse_none, "None", 0, 1, VAR_BOOL, on_mouse_mode_change, app));
    menurow_add_interaction(mrow1, variableinteraction_create(&app->mouse_add_wave, "Add Wave", 0, 1, VAR_BOOL, on_mouse_mode_change, app));
    menu_add_row(m->mouse_menu, mrow1);
    MenuRow *mrow2 = menurow_create();
    menurow_add_interaction(mrow2, variableinteraction_create(&app->mouse_add_barrier, "Barrier", 0, 1, VAR_BOOL, on_mouse_mode_change, app));
    /* Add brush mode toggle */
    menurow_add_interaction(mrow2, variableinteraction_create(&app->mouse_add_barrier_brush, "Barrier Brush", 0, 1, VAR_BOOL, on_mouse_mode_change, app));
    menurow_add_interaction(mrow2, variableinteraction_create(&app->mouse_source, "Source", 0, 1, VAR_BOOL, on_mouse_mode_change, app));
    menu_add_row(m->mouse_menu, mrow2);
    MenuRow *mrow3 = menurow_create();
    // increase mouse amplitude range by one order of magnitude
    menurow_add_interaction(mrow3, variableinteraction_create(&app->wave_amplitude, "Amplitude", 0.0, 0.08, VAR_SLIDER, NULL, NULL));
    menu_add_row(m->mouse_menu, mrow3);
    MenuRow *mrow4 = menurow_create();
    menurow_add_interaction(mrow4, variableinteraction_create(&app->wave_spread, "Spread", 0.01, 0.2, VAR_SLIDER, NULL, NULL));
    menu_add_row(m->mouse_menu, mrow4);
    MenuRow *mrow5 = menurow_create();
    menurow_add_interaction(mrow5, variableinteraction_create(&app->default_source_frequency, "Source Freq (Hz)\t", 0.5, 300.0, VAR_SLIDER, NULL, NULL));
    menu_add_row(m->mouse_menu, mrow5);
    MenuRow *mrow6 = menurow_create();
    menurow_add_interaction(mrow6, variableinteraction_create(&app->default_source_phase, "Source Phase (rad)", 0.0, 6.28, VAR_SLIDER, NULL, NULL));
    menu_add_row(m->mouse_menu, mrow6);

    // Sim menu (wider for longer slider labels)
    m->sim_menu = menu_create(270,10,320,360,1,"Simulation Controls", textColor, bgColor);
    MenuRow *s1 = menurow_create(); menurow_add_interaction(s1, variableinteraction_create(&app->paused, "Paused", 0, 1, VAR_BOOL, NULL, app)); menu_add_row(m->sim_menu, s1);
    MenuRow *s2 = menurow_create(); menurow_add_interaction(s2, variableinteraction_create(&app->dummy_reset, "Reset", 0, 1, VAR_BOOL, cb_reset, reset_cb_data)); menu_add_row(m->sim_menu, s2);
    // Wave speed slider: allow values from 0.01 .. 2.0
    // Wave speed not used in this simulation variant; remove slider
    // Timestep dt slider: very small .. double initial value
    MenuRow *s4 = menurow_create(); menurow_add_interaction(s4, variableinteraction_create(&app->dt, "dt (s)", 1e-6, fmax(1e-6, 2.0 * ((reset_cb_data_t*)reset_cb_data)->dt_val), VAR_SLIDER, cb_dt_changed, reset_cb_data)); menu_add_row(m->sim_menu, s4);
    // Number of compute iterations per render (1..10 integer steps)
    MenuRow *s4b = menurow_create(); menurow_add_interaction(s4b, variableinteraction_create(&app->steps_per_frame, "Steps/frame", 1.0, 10.0, VAR_SLIDER, NULL, NULL)); menu_add_row(m->sim_menu, s4b);
    MenuRow *s5 = menurow_create(); menurow_add_interaction(s5, variableinteraction_create(&render->value_scale, "Scale", 0.001, 10.0, VAR_SLIDER, NULL, NULL)); menu_add_row(m->sim_menu, s5);
    // Physical parameter sliders: viscosity (nu), gravity (x/y), temperature diffusivity (kappa)
    MenuRow *s_visc = menurow_create(); menurow_add_interaction(s_visc, variableinteraction_create(&app->viscosity_nu, "Viscosity", 1e-8, 5e-2, VAR_SLIDER, NULL, app)); menu_add_row(m->sim_menu, s_visc);
    MenuRow *s_visc_it = menurow_create(); menurow_add_interaction(s_visc_it, variableinteraction_create(&app->visc_iters, "Visc Jacobi iters", 0.0, 200.0, VAR_SLIDER, NULL, app)); menu_add_row(m->sim_menu, s_visc_it);
    MenuRow *s_gx = menurow_create(); menurow_add_interaction(s_gx, variableinteraction_create(&app->gravity_x, "Gravity X", -50.0, 50.0, VAR_SLIDER, NULL, app)); menu_add_row(m->sim_menu, s_gx);
    MenuRow *s_gy = menurow_create(); menurow_add_interaction(s_gy, variableinteraction_create(&app->gravity_y, "Gravity Y", -50.0, 50.0, VAR_SLIDER, NULL, app)); menu_add_row(m->sim_menu, s_gy);
    MenuRow *s_k = menurow_create(); menurow_add_interaction(s_k, variableinteraction_create(&app->temp_kappa, "Temp Diff", 1e-8, 1e-2, VAR_SLIDER, NULL, app)); menu_add_row(m->sim_menu, s_k);
    /* Buoyancy strength slider */
    MenuRow *s_b = menurow_create(); menurow_add_interaction(s_b, variableinteraction_create(&app->buoyancy_beta, "Buoyancy (beta)", 0.0, 0.1, VAR_SLIDER, NULL, app)); menu_add_row(m->sim_menu, s_b);
    /* Put Clear Barriers and Clear Sources on the same row and pass pointers as user_data */
    MenuRow *s6 = menurow_create();
    /* Use NULL user_data; callbacks reference global g_bm, paint_buf, tex_paint_cpu, and g_grid_nx/g_grid_ny */
    menurow_add_interaction(s6, variableinteraction_create(&app->dummy_clear_barriers, "Clear Barriers", 0, 1, VAR_BOOL, cb_clear_barriers, NULL));
    menurow_add_interaction(s6, variableinteraction_create(&app->dummy_clear_sources, "Clear Sources", 0, 1, VAR_BOOL, cb_clear_sources, NULL));
    menu_add_row(m->sim_menu, s6);
    // Pressure solver controls
    MenuRow *s12b = menurow_create(); menurow_add_interaction(s12b, variableinteraction_create(&app->sor_alpha, "SOR alpha (omega)", 1.0, 1.95, VAR_SLIDER, NULL, NULL)); menu_add_row(m->sim_menu, s12b);
    MenuRow *s12c = menurow_create(); menurow_add_interaction(s12c, variableinteraction_create(&app->sor_iters, "SOR iters/frame", 0, 200, VAR_SLIDER, NULL, NULL)); menu_add_row(m->sim_menu, s12c);
    MenuRow *s12d = menurow_create(); menurow_add_interaction(s12d, variableinteraction_create(&app->pressure_width, "Pressure width", 0.0, 4.0, VAR_SLIDER, NULL, NULL)); menu_add_row(m->sim_menu, s12d);
    // Single slider to select render mode (integer steps). Replace radio buttons to save vertical space.
    render->mode_slider = (double)render->mode; /* initialize from current mode */
    MenuRow *s_mode = menurow_create();
    VariableInteraction *mode_vi = variableinteraction_create(&render->mode_slider, "Mode: Velocity", (double)RENDER_VELOCITY, (double)RENDER_TEMPERATURE, VAR_SLIDER, on_render_mode_slider_change, render);
    variableinteraction_set_step(mode_vi, 1.0); /* snap to integer modes */
    menurow_add_interaction(s_mode, mode_vi); menu_add_row(m->sim_menu, s_mode);

    // source_menu left empty for now
    m->source_menu = menu_create(530,10,250,220,1,"Source Controls", textColor, bgColor);
    // increase source amplitude control range: allow negative amplitudes as well
    MenuRow *src1 = menurow_create(); menurow_add_interaction(src1, variableinteraction_create(&sel_src_amp, "Amp", -0.1, 0.1, VAR_SLIDER, cb_source_control_changed, NULL)); menu_add_row(m->source_menu, src1);
    MenuRow *src2 = menurow_create(); menurow_add_interaction(src2, variableinteraction_create(&sel_src_freq, "Freq (Hz)", 0.0, 300.0, VAR_SLIDER, cb_source_control_changed, NULL)); menu_add_row(m->source_menu, src2);
    MenuRow *src3 = menurow_create(); menurow_add_interaction(src3, variableinteraction_create(&sel_src_phase, "Phase (rad)", 0.0, 6.283, VAR_SLIDER, cb_source_control_changed, NULL)); menu_add_row(m->source_menu, src3);
    MenuRow *src4 = menurow_create(); menurow_add_interaction(src4, variableinteraction_create(&sel_src_radius, "Radius (grid)", 1.0, 50.0, VAR_SLIDER, cb_source_control_changed, NULL)); menu_add_row(m->source_menu, src4);
    /* source target: 0=smoke,1=pressure,2=temperature implemented as integer slider */
    MenuRow *src_target = menurow_create();
    VariableInteraction *tvi = variableinteraction_create(&sel_src_target, "Target: Smoke", 0.0, 2.0, VAR_SLIDER, on_src_target_slider_change, NULL);
    variableinteraction_set_step(tvi, 1.0);
    /* update label dynamically when slider changes isn't supported here; keep default name */
    menurow_add_interaction(src_target, tvi);
    menu_add_row(m->source_menu, src_target);
    MenuRow *src5 = menurow_create(); menurow_add_interaction(src5, variableinteraction_create(&sel_src_deselect, "Deselect", 0, 1, VAR_BOOL, cb_deselect_selected_source, NULL)); menu_add_row(m->source_menu, src5);
    MenuRow *src6 = menurow_create(); menurow_add_interaction(src6, variableinteraction_create(&sel_src_delete, "Delete", 0, 1, VAR_BOOL, cb_delete_selected_source, NULL)); menu_add_row(m->source_menu, src6);

    return m;
}

// Simple callback implementations
static void cb_reset(VariableInteraction *vi, void *user_data) {
    (void)vi;
    if (!user_data) { fprintf(stderr, "Reset pressed (no data)\n"); return; }
    reset_cb_data_t *d = (reset_cb_data_t*)user_data;
    uint32_t now = SDL_GetTicks();
    if (d->last_ms && now - d->last_ms < 200) {
        if (vi && vi->variable) *(int*)vi->variable = 0;
        return;
    }
    d->last_ms = now;
    fprintf(stderr, "Reset pressed (will re-upload initial field) data_curr=%p data_prev=%p nx=%u ny=%u\n", (void*)d->data_curr, (void*)d->data_prev, (unsigned)d->nx, (unsigned)d->ny);
    fflush(stderr);

    size_t pixels = (size_t)d->nx * d->ny;
    size_t bytes = pixels * 4 * sizeof(float);
    float *buf = calloc(pixels * 4, sizeof(float));
    if (!buf) return;

    // Upload zeros into the current texture pointer (clear scene on reset)
    if (d->tex_curr && *(d->tex_curr)) {
        /* buf was allocated and zeroed above; just upload it to clear the texture */
        glBindTexture(GL_TEXTURE_2D, *(d->tex_curr));
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
    }

    // Upload zeros into the previous texture pointer as well (clear history)
    if (d->tex_prev && *(d->tex_prev)) {
        glBindTexture(GL_TEXTURE_2D, *(d->tex_prev));
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
    }

    // Clear the out texture to zeros to avoid leftover large values
    if (d->tex_out && *(d->tex_out)) {
        // zero buffer
        memset(buf, 0, bytes);
        glBindTexture(GL_TEXTURE_2D, *(d->tex_out));
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
    }

    /* Also clear paint buffers/textures so wave additions are reset */
    {
        extern float *paint_buf; extern GLuint tex_paint; extern GLuint tex_paint_cpu; extern int paint_buf_dirty; extern int paint_pending; extern int paint_from_gpu;
        size_t psize = (size_t)d->nx * d->ny * 4 * sizeof(float);
        if (paint_buf) {
            memset(paint_buf, 0, psize);
            paint_buf_dirty = 0;
            paint_pending = 0; paint_from_gpu = 0;
            /* upload zeros into paint textures */
            glBindTexture(GL_TEXTURE_2D, tex_paint);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, paint_buf);
            glBindTexture(GL_TEXTURE_2D, tex_paint_cpu);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, paint_buf);
        }
    }

    /* Also clear solver/field textures (pressure, velocity, divergence, smoke, helpers)
       so pressing reset truly zeroes the full simulation state. These are file-scope
       variables so reference them as extern here. */
    {
        extern GLuint tex_pressure; extern GLuint tex_pressure_tmp; extern GLuint tex_divergence;
        extern GLuint tex_velocity; extern GLuint tex_velocity_tmp; extern GLuint tex_smoke; extern GLuint tex_smoke_tmp;
        extern GLuint tex_velocity_x; extern GLuint tex_velocity_y; extern GLuint tex_velocity_tmp_x; extern GLuint tex_velocity_tmp_y;
        /* buf is currently allocated and zeroed above */
        if (tex_pressure) {
            glBindTexture(GL_TEXTURE_2D, tex_pressure);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_pressure_tmp) {
            glBindTexture(GL_TEXTURE_2D, tex_pressure_tmp);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_divergence) {
            glBindTexture(GL_TEXTURE_2D, tex_divergence);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_velocity) {
            glBindTexture(GL_TEXTURE_2D, tex_velocity);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_velocity_tmp) {
            glBindTexture(GL_TEXTURE_2D, tex_velocity_tmp);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_smoke) {
            glBindTexture(GL_TEXTURE_2D, tex_smoke);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_smoke_tmp) {
            glBindTexture(GL_TEXTURE_2D, tex_smoke_tmp);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        /* Clear temperature textures as well so Reset resets temperature field */
        extern GLuint tex_temp; extern GLuint tex_temp_tmp;
        if (tex_temp) {
            glBindTexture(GL_TEXTURE_2D, tex_temp);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_temp_tmp) {
            glBindTexture(GL_TEXTURE_2D, tex_temp_tmp);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_velocity_x) {
            glBindTexture(GL_TEXTURE_2D, tex_velocity_x);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_velocity_y) {
            glBindTexture(GL_TEXTURE_2D, tex_velocity_y);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_velocity_tmp_x) {
            glBindTexture(GL_TEXTURE_2D, tex_velocity_tmp_x);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        if (tex_velocity_tmp_y) {
            glBindTexture(GL_TEXTURE_2D, tex_velocity_tmp_y);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d->nx, d->ny, GL_RGBA, GL_FLOAT, buf);
        }
        /* reset any CPU-side smoke/pressure flags if present */
        extern int paint_cpu_uploaded; (void)paint_cpu_uploaded; /* keep symbol reference if present */
    }

    free(buf);
    if (vi && vi->variable) *(int*)vi->variable = 0;
    // reset simulation step counter if pointer is provided
    if (d->sim_counter_ptr) *(d->sim_counter_ptr) = 0;
}

// Callback to rebuild the compute shader when wave_speed changes
static void cb_wave_speed_changed(VariableInteraction *vi, void *user_data) {
    if (!user_data || !vi) return;
    reset_cb_data_t *d = (reset_cb_data_t*)user_data;
    // vi->variable points to app.wave_speed (double)
    double new_c = *(double*)vi->variable;
    // Rebuild expression with new c and recompile GPU program
    if (!d->wave_expr_ptr || !d->grid_ptr) return;
    // free old program if present
    if (d->gpu_prog_ptr && *(d->gpu_prog_ptr)) { gpu_program_free(*(d->gpu_prog_ptr)); *(d->gpu_prog_ptr) = NULL; }
    if (d->compute_prog_ptr && *(d->compute_prog_ptr)) { glDeleteProgram(*(d->compute_prog_ptr)); *(d->compute_prog_ptr) = 0; }
    // rebuild expression: same as earlier but with new c
    Expression *u_curr = expr_variable("u_curr");
    Expression *u_prev = expr_variable("u_prev");
    Expression *lap = expr_laplacian(expr_variable("u_curr"));
    double dt_local = d->dt_val;
    double c2dt2 = new_c * new_c * dt_local * dt_local;
    Expression *c2dt2_lit = expr_literal(literal_create_scalar(c2dt2));
    Expression *accel = expr_multiply(c2dt2_lit, lap);
    Expression *two = expr_literal(literal_create_scalar(2.0));
    Expression *two_u = expr_multiply(two, u_curr);
    Expression *neg_prev = expr_negate(u_prev);
    Expression *diff = expr_add(two_u, neg_prev);
    Expression *wave_expr_new = expr_add(diff, accel);
    // compile
    GPUProgram *prog_new = gpu_compile_optimized(wave_expr_new, d->grid_ptr, GPU_BACKEND_OPENGL);
    if (prog_new && prog_new->kernels && prog_new->kernels[0] && prog_new->kernels[0]->source) {
        const char *src = prog_new->kernels[0]->source;
        GLuint vs = compile_shader(GL_VERTEX_SHADER, "void main() { gl_Position = gl_Vertex; gl_TexCoord[0] = gl_MultiTexCoord0; }");
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, src);
        if (vs && fs) {
            GLuint new_prog = link_program(vs, fs);
            glDeleteShader(vs); glDeleteShader(fs);
            if (new_prog) {
                if (d->compute_prog_ptr) *(d->compute_prog_ptr) = new_prog;
            }
        }
    }
    // store new GPUProgram pointer
    if (d->gpu_prog_ptr) *(d->gpu_prog_ptr) = prog_new;
    // free the temporary expression (release a ref; GPUProgram retained one)
    expression_release(wave_expr_new);
}

/* Callback to handle dt changes from UI: update reset data and recompile compute program. */
static void cb_dt_changed(VariableInteraction *vi, void *user_data) {
    if (!user_data || !vi) return;
    reset_cb_data_t *d = (reset_cb_data_t*)user_data;
    double new_dt = *(double*)vi->variable;
    d->dt_val = new_dt;
    /* Rebuild using current wave speed (call cb_wave_speed_changed with dummy VariableInteraction pointing at wave_speed) */
    if (d->app_wave_speed_ptr) {
        VariableInteraction tmp = {0};
        tmp.variable = d->app_wave_speed_ptr;
        cb_wave_speed_changed(&tmp, d);
    }
}

static void cb_clear_sources(VariableInteraction *vi, void *user_data) {
    (void)user_data;
    if (vi && vi->variable) *(int*)vi->variable = 0;
    /* Clear g_sources array and mark descriptors dirty so GPU sees zero sources */
    if (g_sources) { free(g_sources); g_sources = NULL; }
    g_n_sources = 0; g_selected_source = -1; sources_dirty = 1;
    /* Clear CPU paint buffer and upload zeros to tex_paint_cpu if pointer available */
    if (paint_buf && g_grid_nx && g_grid_ny) {
        size_t psize = (size_t)g_grid_nx * g_grid_ny * 4 * sizeof(float);
        memset(paint_buf, 0, psize);
        paint_buf_dirty = 0; paint_pending = 0; paint_from_gpu = 0; paint_cpu_uploaded = 0;
        if (tex_paint_cpu) {
            glBindTexture(GL_TEXTURE_2D, tex_paint_cpu);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_grid_nx, g_grid_ny, GL_RGBA, GL_FLOAT, paint_buf);
        }
    }
    fprintf(stderr, "Cleared sources and reset paint buffers.\n"); fflush(stderr);
}

static void cb_clear_barriers(VariableInteraction *vi, void *user_data) {
    (void)user_data;
    if (vi && vi->variable) *(int*)vi->variable = 0;
    BoundaryMask *bm = g_bm;
    if (bm) {
        GridMetadata *g = bm->grid; if (g) {
            uint32_t nx = g->dims[0], ny = g->dims[1];
            /* zero mask and values arrays */
            if (bm->mask) memset(bm->mask, 0, (size_t)nx * ny);
            if (bm->values) memset(bm->values, 0, (size_t)nx * ny * sizeof(double));
            if (bm->types) { for (size_t i=0;i<(size_t)nx*ny;++i) bm->types[i] = BC_DIRICHLET; }
        }
        /* upload cleared mask to GPU */
        boundary_mask_upload(bm, NULL);
    }
    fprintf(stderr, "Cleared barriers and uploaded empty mask.\n"); fflush(stderr);
}

// Simple barrier segment stored in grid indices
typedef struct { int x0, y0, x1, y1; double thickness; } BarrierSeg;

// Rebuild bm->mask from an array of barrier segments (overwrites existing mask)
static void rebuild_barrier_mask(BoundaryMask *bm, BarrierSeg *segs, int nseg) {
    if (!bm) return;
    GridMetadata *g = bm->grid; if (!g) return;
    uint32_t nx = g->dims[0], ny = g->dims[1];
    // clear existing non-axis mask entries while preserving axis-aligned ones populated earlier
    // we'll zero everything and then re-populate axis-aligned to keep consistent with populate_axis_aligned
    memset(bm->mask, 0, (size_t)nx * ny);
    for (int s = 0; s < nseg; ++s) {
        BarrierSeg *B = &segs[s];
        double x0 = (double)B->x0, y0 = (double)B->y0, x1 = (double)B->x1, y1 = (double)B->y1;
        double dx = x1 - x0, dy = y1 - y0;
        double len = sqrt(dx*dx + dy*dy);
        if (len < 1e-6) continue;
        double ux = dx / len, uy = dy / len;
        double half = len * 0.5;
        double cx = 0.5*(x0 + x1), cy = 0.5*(y0 + y1);
        double thick = B->thickness > 0.0 ? B->thickness : 1.0;
        for (uint32_t j = 0; j < ny; ++j) {
            for (uint32_t i = 0; i < nx; ++i) {
                double rx = (double)i - cx;
                double ry = (double)j - cy;
                double t = rx * ux + ry * uy;
                double perp = fabs(-rx * uy + ry * ux);
                if (fabs(t) <= half && perp <= thick) {
                    size_t off = (size_t)i * ny + j;
                    bm->mask[off] = 1;
                    bm->values[off] = 0.0;
                    bm->types[off] = BC_DIRICHLET;
                    bm->priority[off] = 0;
                }
            }
        }
    }
}

// Paint or clear a circular region in the boundary mask (grid coords).
// If 'set' is non-zero the mask is set to 1 (barrier); otherwise cleared to 0.
static void barrier_brush_paint(BoundaryMask *bm, int cx, int cy, int radius, int set) {
    if (!bm) return;
    GridMetadata *g = bm->grid; if (!g) return;
    uint32_t nx = g->dims[0], ny = g->dims[1];
    int r2 = radius * radius;
    int x0 = cx - radius; if (x0 < 0) x0 = 0;
    int x1 = cx + radius; if (x1 >= (int)nx) x1 = (int)nx - 1;
    int y0 = cy - radius; if (y0 < 0) y0 = 0;
    int y1 = cy + radius; if (y1 >= (int)ny) y1 = (int)ny - 1;
    for (int j = y0; j <= y1; ++j) {
        for (int i = x0; i <= x1; ++i) {
            int dx = i - cx; int dy = j - cy; if (dx*dx + dy*dy > r2) continue;
            size_t off = (size_t)i * ny + (size_t)j;
            if (set) {
                bm->mask[off] = 1;
                bm->values[off] = 0.0;
                bm->types[off] = BC_DIRICHLET;
                bm->priority[off] = 0;
            } else {
                bm->mask[off] = 0;
                bm->values[off] = 0.0;
                bm->types[off] = BC_DIRICHLET;
                bm->priority[off] = 0;
            }
        }
    }
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    // Visible domain parameters (match interactive_wave_sim defaults roughly)
    double Lx_vis = 2.0, Ly_vis = 2.0;
    /* scale the visible resolution by 1.5 while keeping physical size */
    int nx_vis = 384, ny_vis = 384; /* 256 * 1.5 = 384 */

    /* No outer rim: the simulated domain equals the visible region. */
    int sponge = 0; /* kept for compatibility with coordinate macros but set to zero */
    uint32_t nx = (uint32_t)(nx_vis);
    uint32_t ny = (uint32_t)(ny_vis);
    uint32_t dims[3] = { nx, ny, 1 };
    double spacing[3] = { Lx_vis / (nx_vis - 1), Ly_vis / (ny_vis - 1), 1.0 };
    double origin[3] = { 0.0, 0.0, 0.0 };
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);

    // Set edge boundaries open (no special Dirichlet masking at edges)
    grid_set_boundary(grid, 0, 0, BC_DIRICHLET, 0.0);
    grid_set_boundary(grid, 0, 1, BC_DIRICHLET, 0.0);
    grid_set_boundary(grid, 1, 0, BC_DIRICHLET, 0.0);
    grid_set_boundary(grid, 1, 1, BC_DIRICHLET, 0.0);

    // Build wave update expression: u_next = 2*u_curr - u_prev + c^2 dt^2 Lap(u_curr)
    Expression *u_curr = expr_variable("u_curr");
    Expression *u_prev = expr_variable("u_prev");
    Expression *lap = expr_laplacian(expr_variable("u_curr"));
    double c = 1.0; double dt = 0.002; double c2dt2 = c*c*dt*dt;
    Expression *c2dt2_lit = expr_literal(literal_create_scalar(c2dt2));
    Expression *accel = expr_multiply(c2dt2_lit, lap);
    Expression *two = expr_literal(literal_create_scalar(2.0));
    Expression *two_u = expr_multiply(two, u_curr);
    Expression *neg_prev = expr_negate(u_prev);
    Expression *diff = expr_add(two_u, neg_prev);
    Expression *wave_expr = expr_add(diff, accel);

    // Compile expression into fragment GLSL using existing emitter
    GPUProgram *prog = gpu_compile_optimized(wave_expr, grid, GPU_BACKEND_OPENGL);
    if (!prog) { fprintf(stderr, "gpu compile failed\n"); return 1; }

    // Initialize SDL2 + OpenGL context and window
    if (SDL_Init(SDL_INIT_VIDEO) != 0) { fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError()); return 1; }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_Window *win = SDL_CreateWindow("interactive_wave_sim_gpu", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                      800, 800, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!win) { fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError()); return 1; }
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) { fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError()); return 1; }
    glewExperimental = GL_TRUE; if (glewInit() != GLEW_OK) { fprintf(stderr, "glewInit failed\n"); }

    // Compile compute shader (fragment shader returned by emitter)
    const char *vs_src = "void main() { gl_Position = gl_Vertex; gl_TexCoord[0] = gl_MultiTexCoord0; }";
    // Use the emitted compute shader so masking and the wave update are applied
    int use_test_compute = 0;
    const char *test_fs = "";
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = 0;
    fs = compile_shader(GL_FRAGMENT_SHADER, prog->kernels[0]->source);
    if (!vs || !fs) { fprintf(stderr, "shader compile failed\n"); return 1; }
    GLuint compute_prog = link_program(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    if (!compute_prog) { fprintf(stderr, "link failed\n"); return 1; }

    // Paint composite shader: add GPU-generated paint and CPU paint into src_tex and write to out
    const char *paint_fs = "#version 120\nuniform sampler2D src_tex; uniform sampler2D paint_gpu; uniform sampler2D paint_cpu; void main() { vec2 uv = gl_TexCoord[0].st; float s = texture2D(src_tex, uv).r; float pg = texture2D(paint_gpu, uv).r; float pc = texture2D(paint_cpu, uv).r; gl_FragColor = vec4(s + pg + pc, 0.0, 0.0, 0.0); }";
    GLuint p_fs = compile_shader(GL_FRAGMENT_SHADER, paint_fs);
    GLuint p_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint p_prog = link_program(p_vs, p_fs);
    glDeleteShader(p_vs); glDeleteShader(p_fs);
    if (!p_prog) { fprintf(stderr, "paint program link failed\n"); }

    /* Small composite shaders to add paint channels into smoke and pressure textures.
       Each program samples the existing target texture and both paint textures (GPU & CPU)
       and writes the target channel plus any paint additions. */
    const char *add_smoke_fs = "#version 120\nuniform sampler2D tgt_tex; uniform sampler2D paint_gpu; uniform sampler2D paint_cpu; void main() { vec2 uv = gl_TexCoord[0].st; float t = texture2D(tgt_tex, uv).r; vec4 pg = texture2D(paint_gpu, uv); vec4 pc = texture2D(paint_cpu, uv); float add = pg.r + pc.r; gl_FragColor = vec4(t + add, 0.0, 0.0, 0.0); }";
    const char *add_press_fs = "#version 120\nuniform sampler2D tgt_tex; uniform sampler2D paint_gpu; uniform sampler2D paint_cpu; void main() { vec2 uv = gl_TexCoord[0].st; float t = texture2D(tgt_tex, uv).r; vec4 pg = texture2D(paint_gpu, uv); vec4 pc = texture2D(paint_cpu, uv); float add = pg.g + pc.g; gl_FragColor = vec4(t + add, 0.0, 0.0, 0.0); }";
    GLuint add_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint adds_fs = compile_shader(GL_FRAGMENT_SHADER, add_smoke_fs);
    GLuint addp_fs = compile_shader(GL_FRAGMENT_SHADER, add_press_fs);
    if (add_vs && adds_fs) prog_add_paint_smoke = link_program(add_vs, adds_fs);
    if (add_vs && addp_fs) prog_add_paint_pressure = link_program(add_vs, addp_fs);
    if (add_vs) glDeleteShader(add_vs);
    if (adds_fs) glDeleteShader(adds_fs);
    if (addp_fs) glDeleteShader(addp_fs);
    /* temperature composite: B channel -> temp texture */
    const char *add_temp_fs = "#version 120\nuniform sampler2D tgt_tex; uniform sampler2D paint_gpu; uniform sampler2D paint_cpu; void main() { vec2 uv = gl_TexCoord[0].st; float t = texture2D(tgt_tex, uv).r; vec4 pg = texture2D(paint_gpu, uv); vec4 pc = texture2D(paint_cpu, uv); float add = pg.b + pc.b; gl_FragColor = vec4(t + add, 0.0, 0.0, 0.0); }";
    GLuint addt_fs = compile_shader(GL_FRAGMENT_SHADER, add_temp_fs);
    if (add_vs && addt_fs) prog_add_paint_temp = link_program(add_vs, addt_fs);
    if (addt_fs) glDeleteShader(addt_fs);
    if (prog_add_paint_smoke) {
        glUseProgram(prog_add_paint_smoke);
        loc_add_paint_target = glGetUniformLocation(prog_add_paint_smoke, "tgt_tex"); if (loc_add_paint_target >= 0) glUniform1i(loc_add_paint_target, 0);
        loc_add_paint_tex = glGetUniformLocation(prog_add_paint_smoke, "paint_gpu"); if (loc_add_paint_tex >= 0) glUniform1i(loc_add_paint_tex, 1);
        loc_add_paint_cpu = glGetUniformLocation(prog_add_paint_smoke, "paint_cpu"); if (loc_add_paint_cpu >= 0) glUniform1i(loc_add_paint_cpu, 2);
        glUseProgram(0);
    }
    if (prog_add_paint_pressure) {
        glUseProgram(prog_add_paint_pressure);
        /* same uniform bindings */
        (void)glGetUniformLocation(prog_add_paint_pressure, "tgt_tex"); glUniform1i(glGetUniformLocation(prog_add_paint_pressure, "tgt_tex"), 0);
        (void)glGetUniformLocation(prog_add_paint_pressure, "paint_gpu"); glUniform1i(glGetUniformLocation(prog_add_paint_pressure, "paint_gpu"), 1);
        (void)glGetUniformLocation(prog_add_paint_pressure, "paint_cpu"); glUniform1i(glGetUniformLocation(prog_add_paint_pressure, "paint_cpu"), 2);
        glUseProgram(0);
    }
    if (prog_add_paint_temp) {
        glUseProgram(prog_add_paint_temp);
        (void)glGetUniformLocation(prog_add_paint_temp, "tgt_tex"); glUniform1i(glGetUniformLocation(prog_add_paint_temp, "tgt_tex"), 0);
        (void)glGetUniformLocation(prog_add_paint_temp, "paint_gpu"); glUniform1i(glGetUniformLocation(prog_add_paint_temp, "paint_gpu"), 1);
        (void)glGetUniformLocation(prog_add_paint_temp, "paint_cpu"); glUniform1i(glGetUniformLocation(prog_add_paint_temp, "paint_cpu"), 2);
        glUseProgram(0);
    }

    // Prepare initial CPU fields (u_curr with a gaussian source; u_prev zeros)
    double *u_curr_data = calloc((size_t)nx * ny, sizeof(double));
    double *u_prev_data = calloc((size_t)nx * ny, sizeof(double));

    // Create an empty BoundaryMask; barrier segments will be user-managed
    BoundaryMask *bm = boundary_mask_create(grid);
    /* expose global pointer for callbacks */
    g_bm = bm; g_grid_nx = nx; g_grid_ny = ny;
    // barrier segments array (dynamic, starts empty)
    BarrierSeg *barrier_segs = NULL; int n_barrier_segs = 0;
    // initially no barrier segments; user adds via mouse in barrier mode
    // Drag state for barrier point editing
    int barrier_drag_active = 0; /* 0/1 */
    int barrier_drag_seg = -1;  /* segment index being edited */
    int barrier_drag_pt = -1;   /* 0 => x0,y0, 1 => x1,y1 */
    /* Source list (use globals g_sources/g_n_sources for callbacks) */
    g_sources = NULL; g_n_sources = 0; g_selected_source = -1;
    /* Source drag locals (candidate + active) */
    int source_drag_candidate = 0; int source_drag_cidx = -1; int source_drag_press_mx = 0, source_drag_press_my = 0;
    int source_drag_active = 0; int source_drag_idx = -1;
    /* Candidate press (pressed near a point but not yet moved enough to count as drag) */
    int barrier_drag_candidate = 0;
    int barrier_drag_cseg = -1;
    int barrier_drag_cpt = -1;
    int barrier_drag_press_mx = 0, barrier_drag_press_my = 0;
    /* Pending start point when building a new barrier (visible before second click) */
    int barrier_pending = 0;
    int barrier_pending_x = -1, barrier_pending_y = -1; // -1 means pending start not yet set; capture on release

    // Create GPU textures from fields (total grid size includes sponge)
    GLuint tex_u_curr = create_texture_from_field(u_curr_data, nx, ny);
    GLuint tex_u_prev = create_texture_from_field(u_prev_data, nx, ny);
    GLuint tex_out = create_empty_texture(nx, ny);
    // damping removed (no outer rim)
    GLuint damping_prog = 0; /* unused */
    GLuint damping_tex = 0;  /* unused */
    (void)damping_prog; (void)damping_tex;
    GLint loc_damp_src = -1, loc_damp_tex = -1, loc_damp_prev = -1, loc_damp_sigma = -1, loc_damp_dt = -1, loc_damp_dims = -1;
    // Paint textures and CPU paint buffer (RGBA32F)
    paint_buf = calloc((size_t)nx * ny * 4, sizeof(float));
    if (paint_buf) {
        /* no initial debug paint seeded; paint_buf starts zero */
        paint_cpu_uploaded = 0;
    }
    tex_paint = create_empty_texture(nx, ny);     /* GPU-generated paint */
    tex_paint_cpu = create_empty_texture(nx, ny); /* CPU-uploaded paint buffer */
    paint_pending = 0;
    paint_from_gpu = 0; /* true when tex_paint was filled by GPU shader this frame */
    paint_buf_dirty = 0; /* true when paint_buf contains CPU paint that must be uploaded */
    int painting_active = 0;
    uint64_t sim_step_counter = 0; /* counts physics steps (used for source phase) */
    uint64_t render_frame_counter = 0; /* counts rendered frames */

    // Prepare FBO used for compute (render to textures)
    GLuint fbo; glGenFramebuffers(1, &fbo);

    int win_w = 800, win_h = 800;

    // Upload the interior barrier mask textures (needs GL context)
    boundary_mask_upload(bm, NULL);

    // FBO already created above

    // Overlay shader to draw white lines where the mask is active (final pass)
    /* Overlay: draw mask as translucent yellow to make it more obvious on top
       of the rendered field. Use simple sampling (texel-centered) and emit
       alpha < 1 so underlying field remains visible. */
    const char *overlay_fs =
        "#version 120\n"
        "uniform sampler2D mask_tex; uniform ivec2 dims; uniform ivec2 vis_offset; uniform ivec2 vis_size;"
        "void main() { vec2 uv = gl_TexCoord[0].st; vec2 tex_idx = vec2(vis_offset) + uv * vec2(vis_size); vec2 c = (floor(tex_idx) + vec2(0.5)) / vec2(dims); float m = texture2D(mask_tex, c).r; if (m > 0.5) gl_FragColor = vec4(1.0,1.0,0.0,1.0); else gl_FragColor = vec4(0.0,0.0,0.0,0.0); }";
    GLuint ov_fs = compile_shader(GL_FRAGMENT_SHADER, overlay_fs);
    GLuint ov_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint overlay_prog = 0;
    if (ov_vs && ov_fs) overlay_prog = link_program(ov_vs, ov_fs);
    if (ov_vs) glDeleteShader(ov_vs);
    if (ov_fs) glDeleteShader(ov_fs);


    /* Build a custom display fragment shader that can read multiple textures and
       show modes: 0=height,1=velocity magnitude,2=pressure,3=smoke,4=vorticity(curl) */
    const char *disp_vs = "void main() { gl_Position = gl_Vertex; gl_TexCoord[0] = gl_MultiTexCoord0; }";
    const char *disp_fs =
        "#version 120\n"
    "uniform sampler2D src_tex; uniform sampler2D vel_tex; uniform sampler2D mask_tex; uniform sampler2D pressure_tex; uniform sampler2D smoke_tex; uniform sampler2D debug_tex; uniform sampler2D temp_tex;\n"
        "uniform ivec2 dims; uniform ivec2 vis_offset; uniform ivec2 vis_size; uniform vec2 spacing; uniform int render_mode; uniform float value_scale; uniform int show_boundaries;\n"
        "void main() { vec2 uv = gl_TexCoord[0].st; vec2 tex_idx = vec2(vis_offset) + uv * vec2(vis_size); vec2 c = (floor(tex_idx) + vec2(0.5)) / vec2(dims); vec3 col = vec3(0.0);\n"
    " if (render_mode == 1) { vec2 vel = texture2D(vel_tex, c).rg; float scale_vel = value_scale * 0.1; float ux = vel.x * scale_vel; float uy = vel.y * scale_vel; float sx = ux / (1.0 + abs(ux)); float sy = uy / (1.0 + abs(uy)); vec3 negx = vec3(0.0, 1.0, 0.0); vec3 posx = vec3(1.0, 0.0, 0.0); vec3 colx = (sx > 0.0) ? posx * sx : negx * (-sx); vec3 negy = vec3(0.6, 0.0, 0.6); vec3 posy = vec3(1.0, 1.0, 0.0); vec3 coly = (sy > 0.0) ? posy * sy : negy * (-sy); col = clamp(colx + coly, 0.0, 1.0); }\n"
    " else if (render_mode == 2) { float p = texture2D(pressure_tex, c).r; float pn = p * value_scale * 50.0; float pcomp = pn / (1.0 + abs(pn)); if (pcomp > 0.0) col = vec3(pcomp, 0.0, 0.0); else col = vec3(0.0, 0.0, -pcomp); }\n"
    " else if (render_mode == 3) { float s = texture2D(smoke_tex, c).r; float sn = s * value_scale; sn = sn / (1.0 + abs(sn)); col = vec3(sn); }\n"
    " else if (render_mode == 4) { vec2 px = 1.0/vec2(dims); float e_y = texture2D(vel_tex, c + vec2(px.x,0)).y; float w_y = texture2D(vel_tex, c - vec2(px.x,0)).y; float n_x = texture2D(vel_tex, c + vec2(0,px.y)).x; float s_x = texture2D(vel_tex, c - vec2(0,px.y)).x; float dvdx = (e_y - w_y) / (2.0 * spacing.x); float du_dy = (n_x - s_x) / (2.0 * spacing.y); float curl = dvdx - du_dy; float cn = curl * (value_scale * 0.1); float ccomp = cn / (1.0 + abs(cn)); if (ccomp > 0.0) col = vec3(abs(ccomp), 0.0, 0.0); else col = vec3(0.0, 0.0, abs(ccomp)); }\n"
    " else if (render_mode == 5) { vec2 px = 1.0/vec2(dims); float ux_e = texture2D(vel_tex, c + vec2(px.x,0)).x; float ux_w = texture2D(vel_tex, c - vec2(px.x,0)).x; float uy_n = texture2D(vel_tex, c + vec2(0,px.y)).y; float uy_s = texture2D(vel_tex, c - vec2(0,px.y)).y; float dudx = (ux_e - ux_w) / (2.0 * spacing.x); float dvdy = (uy_n - uy_s) / (2.0 * spacing.y); float divv = (dudx + dvdy) * (value_scale * 0.1); float dcomp = divv / (1.0 + abs(divv)); if (dcomp > 0.0) col = vec3(dcomp, 0.0, 0.0); else col = vec3(0.0, 0.0, -dcomp); }\n"
    " else if (render_mode == 6) { float T = texture2D(temp_tex, c).r; float tn = T * value_scale * 10.0; tn = tn / (1.0 + abs(tn)); col = vec3(tn, 0.0, 0.0); }\n"
    " if (show_boundaries == 1) { float m = texture2D(mask_tex, c).r; if (m > 0.5) col = vec3(1.0, 0.0, 0.0); }\n"
    " gl_FragColor = vec4(col, 1.0); }";
    GLuint d_vs = compile_shader(GL_VERTEX_SHADER, disp_vs);
    GLuint d_fs = compile_shader(GL_FRAGMENT_SHADER, disp_fs);
    GLuint disp_prog = 0;
    if (d_vs && d_fs) disp_prog = link_program(d_vs, d_fs);
    if (d_vs) glDeleteShader(d_vs); if (d_fs) glDeleteShader(d_fs);
    if (!disp_prog) { fprintf(stderr, "Display program compile failed\n"); return 1; }
    /* bind sampler units for display program so texture units are fixed */
    glUseProgram(disp_prog);
    GLint loc_disp_src = glGetUniformLocation(disp_prog, "src_tex"); if (loc_disp_src >= 0) glUniform1i(loc_disp_src, 0);
    GLint loc_disp_vel = glGetUniformLocation(disp_prog, "vel_tex"); if (loc_disp_vel >= 0) glUniform1i(loc_disp_vel, 1);
    GLint loc_disp_mask = glGetUniformLocation(disp_prog, "mask_tex"); if (loc_disp_mask >= 0) glUniform1i(loc_disp_mask, 2);
    GLint loc_disp_pressure = glGetUniformLocation(disp_prog, "pressure_tex"); if (loc_disp_pressure >= 0) glUniform1i(loc_disp_pressure, 3);
    GLint loc_disp_smoke = glGetUniformLocation(disp_prog, "smoke_tex"); if (loc_disp_smoke >= 0) glUniform1i(loc_disp_smoke, 4);
    GLint loc_disp_debug = glGetUniformLocation(disp_prog, "debug_tex"); if (loc_disp_debug >= 0) glUniform1i(loc_disp_debug, 5);
    GLint loc_disp_temp = glGetUniformLocation(disp_prog, "temp_tex"); if (loc_disp_temp >= 0) glUniform1i(loc_disp_temp, 6);
    glUseProgram(0);

    // Create app/menu state + menus
    AppState app = {0};
    app.paused = 0; app.wave_speed = 1.0; app.max_sim_speed = 1.0; app.dt = 0.002; app.steps_per_frame = 1.0; app.wave_amplitude = 0.002; /* lower default addition amplitude (scaled) */ app.wave_spread = 0.05; app.default_source_frequency = 5.0; app.default_source_phase = 0.0; app.mouse_none = 1; app.show_base_menu = 1; app.show_mouse_controls = 0; app.show_sim_controls = 0; app.max_sim_speed = 1.0; app.limit_fps = 0;
    /* Physical defaults */
    app.viscosity_nu = 1e-5; app.gravity_x = 0.0; app.gravity_y = -9.8; app.temp_kappa = 1e-5;
    app.buoyancy_beta = 0.002; /* default buoyancy coefficient */
    app.visc_iters = 20.0; /* reasonable default Jacobi iterations per step */
    /* Pressure solver defaults */
    app.sor_alpha = 1.3; app.sor_iters = 40.0; app.pressure_width = 1.0;
    AppRenderState render = {0};
    render.mode = RENDER_VELOCITY; render.value_scale = 1.0; render.show_boundaries = 1; render.show_stats = 1; render.mode_velocity = 1; render.mode_pressure = 0; render.mode_smoke = 0; render.mode_vorticity = 0; render.mode_divergence = 0; render.mode_slider = (double)RENDER_VELOCITY;

    // Prepare reset callback data (heap alloc so pointer stays valid). It holds
    // pointers to the texture variables used for ping-pong so the callback
    // re-uploads into the currently-active textures even after swaps.
    reset_cb_data_t *reset_data = malloc(sizeof(*reset_data));
    reset_data->tex_curr = &tex_u_curr;
    reset_data->tex_prev = &tex_u_prev;
    reset_data->tex_out = &tex_out;
    reset_data->data_curr = u_curr_data;
    reset_data->data_prev = u_prev_data;
    reset_data->nx = nx; reset_data->ny = ny; reset_data->last_ms = 0;
    reset_data->gpu_prog_ptr = &prog;
    reset_data->compute_prog_ptr = &compute_prog;
    reset_data->wave_expr_ptr = &wave_expr;
    reset_data->grid_ptr = grid;
    reset_data->dt_val = dt;
    /* wave_speed not used here; do not hook callback */
    reset_data->app_wave_speed_ptr = NULL;
    reset_data->sim_counter_ptr = &sim_step_counter;

     /* Initialize source-menu radius default to match mouse paint spread:
         use 3*sigma (same radius used by paint_gaussian_to_rgba) converted to grid units */
    sel_src_radius = fmax(1.0, (app.wave_spread * 3.0 / spacing[0]) * SOURCE_RADIUS_SCALE);

     // Try to set font for menus (best-effort)
    if (menu_set_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14) != 0) {
        fprintf(stderr, "menu_set_font failed; menu text may be invisible\n");
    }
    AppMenus *menus = create_app_menus(&app, &render, reset_data);

    // Query uniform locations for compute program and bind static sampler indices
    GLint loc_dims = glGetUniformLocation(compute_prog, "dims");
    GLint loc_spacing = glGetUniformLocation(compute_prog, "spacing");
    GLint loc_use_mask = glGetUniformLocation(compute_prog, "use_mask");
    // Set sampler indices for variables (u_curr_tex -> unit 0, u_prev_tex -> unit 1)
    GLint loc_u_curr = glGetUniformLocation(compute_prog, "u_curr_tex"); if (loc_u_curr>=0) glUseProgram(compute_prog), glUniform1i(loc_u_curr, 0);
    GLint loc_u_prev = glGetUniformLocation(compute_prog, "u_prev_tex"); if (loc_u_prev>=0) glUseProgram(compute_prog), glUniform1i(loc_u_prev, 1);
    // Mask/value samplers will be bound to units 2 and 3
    glUseProgram(compute_prog);
    GLint loc_mask = glGetUniformLocation(compute_prog, "mask_tex"); if (loc_mask>=0) glUniform1i(loc_mask, 2);
    GLint loc_val = glGetUniformLocation(compute_prog, "val_tex"); if (loc_val>=0) glUniform1i(loc_val, 3);
    // Restore program 0
    glUseProgram(0);

    // Prepare paint program uniform locations (if created)
    GLint loc_p_src = -1, loc_p_paint_gpu = -1, loc_p_paint_cpu = -1;
    if (p_prog) {
        glUseProgram(p_prog);
        loc_p_src = glGetUniformLocation(p_prog, "src_tex"); if (loc_p_src >= 0) glUniform1i(loc_p_src, 0);
        loc_p_paint_gpu = glGetUniformLocation(p_prog, "paint_gpu"); if (loc_p_paint_gpu >= 0) glUniform1i(loc_p_paint_gpu, 1);
        loc_p_paint_cpu = glGetUniformLocation(p_prog, "paint_cpu"); if (loc_p_paint_cpu >= 0) glUniform1i(loc_p_paint_cpu, 2);
        glUseProgram(0);
    }

    /* GPU-side source generator program: renders a per-texel paint texture
       from the active sources (last-source-wins semantics to match CPU). */
    GLuint srcgen_prog = 0;
    GLint loc_src_n = -1, loc_src_time = -1, loc_src_dims = -1, loc_src_spacing = -1;
    /* cached uniform locations for descriptor sampler and max width */
    GLint loc_src_desc = -1, loc_max_src = -1;
    GLint loc_src_gx = -1, loc_src_gy = -1, loc_src_amp = -1, loc_src_freq = -1, loc_src_phase = -1, loc_src_radius = -1;
    /* Shader reads source descriptors from a 2-row texture: row 0 = (gx, gy, amp, freq), row 1 = (phase, radius, target, unused)
        target: 0=smoke, 1=pressure, 2=temperature */
    const char *srcgen_fs =
        "#version 120\n"
        "uniform sampler2D src_desc_tex;\n"
        "uniform int n_sources;\n"
        "uniform int max_src;\n"
        "uniform float sim_time; uniform vec2 spacing; uniform ivec2 dims;\n"
        "void main() { vec2 uv = gl_TexCoord[0].st; vec2 idx = floor(uv * vec2(dims)); float smoke = 0.0; float press = 0.0; float tempv = 0.0;\n"
        " for (int i = 0; i < n_sources; ++i) { float fu = (0.5 + float(i)) / float(max_src); vec4 a = texture2D(src_desc_tex, vec2(fu, 0.25)); vec4 b = texture2D(src_desc_tex, vec2(fu, 0.75)); float gx = a.r; float gy = a.g; float amp = a.b; float freq = a.a; float phase = b.r; float radius = b.g; float target = b.b; float dx = (idx.x - gx) * spacing.x; float dy = (idx.y - gy) * spacing.y; float r2 = dx*dx + dy*dy; float rr = radius * radius * spacing.x * spacing.x; if (r2 <= rr) { float raw = sin(freq * sim_time + phase); float v = amp * (1.0 + raw); if (target < 0.5) smoke = v; else if (target < 1.5) press = v; else tempv = v * 10.0; } }\n"
        " gl_FragColor = vec4(smoke, press, tempv, 0.0); }";
    GLuint s_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint s_fs = compile_shader(GL_FRAGMENT_SHADER, srcgen_fs);
    if (s_vs && s_fs) srcgen_prog = link_program(s_vs, s_fs);
    if (s_vs) glDeleteShader(s_vs); if (s_fs) glDeleteShader(s_fs);
        if (srcgen_prog) {
        glUseProgram(srcgen_prog);
        loc_src_n = glGetUniformLocation(srcgen_prog, "n_sources");
        loc_src_time = glGetUniformLocation(srcgen_prog, "sim_time"); loc_src_dims = glGetUniformLocation(srcgen_prog, "dims"); loc_src_spacing = glGetUniformLocation(srcgen_prog, "spacing");
        /* cache descriptor sampler and max_src uniform locations to avoid per-frame queries */
        loc_src_desc = glGetUniformLocation(srcgen_prog, "src_desc_tex"); if (loc_src_desc >= 0) glUniform1i(loc_src_desc, 4);
        loc_max_src = glGetUniformLocation(srcgen_prog, "max_src"); if (loc_max_src >= 0) glUniform1i(loc_max_src, 64);
        /* cache descriptor sampler and max_src uniform locations to avoid per-frame queries */
        GLint loc_desc = glGetUniformLocation(srcgen_prog, "src_desc_tex"); if (loc_desc >= 0) { glUniform1i(loc_desc, 4); }
        GLint loc_mx = glGetUniformLocation(srcgen_prog, "max_src"); if (loc_mx >= 0) { glUniform1i(loc_mx, 64); }
        /* create source-descriptor texture (max 64 sources, 2 rows) */
        glGenTextures(1, &tex_src_desc);
        glBindTexture(GL_TEXTURE_2D, tex_src_desc);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        /* allocate 64x2 RGBA32F texture, initialize to zero */
        int max_src = 64;
        float *zero_buf = calloc((size_t)max_src * 2 * 4, sizeof(float));
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, max_src, 2, 0, GL_RGBA, GL_FLOAT, zero_buf);
        free(zero_buf);
        glUseProgram(0);
    }

     /* Create pressure/velocity/divergence/smoke textures (RGBA32F) */
    tex_pressure = create_empty_texture(nx, ny);
    tex_pressure_tmp = create_empty_texture(nx, ny);
    tex_divergence = create_empty_texture(nx, ny);
    tex_velocity = create_empty_texture(nx, ny);
    tex_velocity_rhs = create_empty_texture(nx, ny);
    tex_velocity_tmp = create_empty_texture(nx, ny);
    tex_smoke = create_empty_texture(nx, ny);
    tex_smoke_tmp = create_empty_texture(nx, ny);
    tex_temp = create_empty_texture(nx, ny);
    tex_temp_tmp = create_empty_texture(nx, ny);

    /* Also create single-channel R textures for vx and vy so compiler-emitted kernels
        that expect separate samplers can be used without changing the emitter. */
    tex_velocity_x = create_empty_texture(nx, ny);
    tex_velocity_y = create_empty_texture(nx, ny);
    tex_velocity_tmp_x = create_empty_texture(nx, ny);
    tex_velocity_tmp_y = create_empty_texture(nx, ny);

    /* Use bilinear filtering for velocity and smoke textures so semi-Lagrangian
        backtraces sample smoothly (avoids pixelation and white/flash artifacts). */
    glBindTexture(GL_TEXTURE_2D, tex_velocity);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, tex_velocity_tmp);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, tex_smoke);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, tex_temp);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, tex_smoke_tmp);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // tex_debug removed
    /* pressure may be sampled at non-integer coords for display / projection; linear helps visuals */
    glBindTexture(GL_TEXTURE_2D, tex_pressure);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    /* Zero-initialize pressure/velocity/smoke textures to avoid garbage on first frame */
    size_t pixels = (size_t)nx * ny;
    float *zero = calloc(pixels * 4, sizeof(float));
    if (zero) {
        glBindTexture(GL_TEXTURE_2D, tex_pressure); glPixelStorei(GL_UNPACK_ALIGNMENT, 1); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, zero);
        glBindTexture(GL_TEXTURE_2D, tex_pressure_tmp); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, zero);
        glBindTexture(GL_TEXTURE_2D, tex_divergence); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, zero);
        glBindTexture(GL_TEXTURE_2D, tex_velocity); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, zero);
        glBindTexture(GL_TEXTURE_2D, tex_velocity_tmp); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, zero);
        glBindTexture(GL_TEXTURE_2D, tex_smoke); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, zero);
        glBindTexture(GL_TEXTURE_2D, tex_smoke_tmp); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, zero);
        free(zero);
    }

    /* Insert a small initial pressure pulse at the center for diagnostics (one-shot). We'll write into tex_pressure directly. */
    int diag_pulse_done = 0;
    int cx = (int)(nx/2), cy = (int)(ny/2);
    int pradius = 6;
    float *tmp = calloc(pixels * 4, sizeof(float));
    if (tmp) {
        for (int j = 0; j < (int)ny; ++j) {
            for (int i = 0; i < (int)nx; ++i) {
                int dx = i - cx; int dy = j - cy; if (dx*dx + dy*dy <= pradius*pradius) {
                    size_t off = ((size_t)j * nx + (size_t)i) * 4;
                    tmp[off + 0] = 5.0f; /* R channel holds pressure */
                }
            }
        }
        glBindTexture(GL_TEXTURE_2D, tex_pressure);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, tmp);
        free(tmp);
        diag_pulse_done = 1;
        fprintf(stderr, "Injected initial pressure pulse at center.\n"); fflush(stderr);
    }


    /* Build divergence expression: div = d(vx)/dx + d(vy)/dy where velocity is stored as variable "vel" with two channels.
       We'll create two derivative expressions for vel.x and vel.y by first extracting components as separate variables
       then using expr_derivative. The compiler expects vector fields as separate variables, so create expr_variable("vx") and expr_variable("vy"). */
    Expression *vx = expr_variable("vx");
    Expression *vy = expr_variable("vy");
    Expression *dvx_dx = expr_derivative(expr_copy(vx), "x");
    Expression *dvy_dy = expr_derivative(expr_copy(vy), "y");
    Expression *div_expr = expr_add(dvx_dx, dvy_dy);
    GPUProgram *div_prog = gpu_compile_optimized(div_expr, grid, GPU_BACKEND_OPENGL);
    if (div_prog && div_prog->kernels && div_prog->kernels[0] && div_prog->kernels[0]->source) {
        const char *src = div_prog->kernels[0]->source;
        /* Dump the compiled divergence shader source to stderr for inspection */
        fprintf(stderr, "[GPU DIVERGENCE SHADER SOURCE START]\n%s\n[GPU DIVERGENCE SHADER SOURCE END]\n", src);
        /* Compile and use the compiler-emitted divergence shader. It expects separate vx_tex/vy_tex samplers. */
        GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, src);
        if (vs && fs) {
            GLuint newp = link_program(vs, fs);
            if (newp) prog_divergence = newp;
            glDeleteShader(vs); glDeleteShader(fs);
        }
        if (prog_divergence) {
            glUseProgram(prog_divergence);
            /* bind component samplers to units 4 and 5 */
            loc_div_vx = glGetUniformLocation(prog_divergence, "vx_tex"); if (loc_div_vx >= 0) glUniform1i(loc_div_vx, 4);
            loc_div_vy = glGetUniformLocation(prog_divergence, "vy_tex"); if (loc_div_vy >= 0) glUniform1i(loc_div_vy, 5);
            /* mask/value samplers follow the project's convention (mask -> unit 2, val -> unit 3) */
            GLint loc_div_mask = glGetUniformLocation(prog_divergence, "mask_tex"); if (loc_div_mask >= 0) glUniform1i(loc_div_mask, 2);
            GLint loc_div_val = glGetUniformLocation(prog_divergence, "val_tex"); if (loc_div_val >= 0) glUniform1i(loc_div_val, 3);
            GLint loc_use_mask = glGetUniformLocation(prog_divergence, "use_mask"); if (loc_use_mask >= 0) glUniform1i(loc_use_mask, 1);
            loc_div_dims = glGetUniformLocation(prog_divergence, "dims"); if (loc_div_dims >= 0) glUniform2i(loc_div_dims, (GLint)nx, (GLint)ny);
            loc_div_spacing = glGetUniformLocation(prog_divergence, "spacing"); if (loc_div_spacing >= 0) glUniform2f(loc_div_spacing, (float)spacing[0], (float)spacing[1]);
            glUseProgram(0);
        }
    }
    expression_release(div_expr);

    /* Compile small extract shaders to fill tex_velocity_x/tex_velocity_y from packed tex_velocity when needed.
       extract_vx: sample vel_tex.r -> output.r
       extract_vy: sample vel_tex.g -> output.r
    */
    const char *extract_vs = "void main() { gl_Position = gl_Vertex; gl_TexCoord[0] = gl_MultiTexCoord0; }";
    const char *extract_vx_fs = "#version 120\nuniform sampler2D vel_tex; void main() { vec2 uv = gl_TexCoord[0].st; float v = texture2D(vel_tex, uv).r; gl_FragColor = vec4(v,0.0,0.0,0.0); }";
    const char *extract_vy_fs = "#version 120\nuniform sampler2D vel_tex; void main() { vec2 uv = gl_TexCoord[0].st; float v = texture2D(vel_tex, uv).g; gl_FragColor = vec4(v,0.0,0.0,0.0); }";
    GLuint evs = compile_shader(GL_VERTEX_SHADER, extract_vs);
    GLuint evx = compile_shader(GL_FRAGMENT_SHADER, extract_vx_fs);
    GLuint evy = compile_shader(GL_FRAGMENT_SHADER, extract_vy_fs);
    if (evs && evx) { prog_extract_vx = link_program(evs, evx); }
    if (evs && evy) { prog_extract_vy = link_program(evs, evy); }
    if (evs) glDeleteShader(evs);
    if (evx) glDeleteShader(evx);
    if (evy) glDeleteShader(evy);
    /* if extraction programs exist, cache their vel_tex uniform units to 0 (we bind tex_velocity at unit 0 before extraction) */
    if (prog_extract_vx) { glUseProgram(prog_extract_vx); GLint loc = glGetUniformLocation(prog_extract_vx, "vel_tex"); if (loc >= 0) glUniform1i(loc, 0); glUseProgram(0); }
    if (prog_extract_vy) { glUseProgram(prog_extract_vy); GLint loc = glGetUniformLocation(prog_extract_vy, "vel_tex"); if (loc >= 0) glUniform1i(loc, 0); glUseProgram(0); }

    /* Red-Black SOR pressure update shader (u_color = 0 red, 1 black) */
    /* Keep the red-black SOR implementation as a hand-written shader because it requires parity-based updates.
       We still build divergence and projection via the expression compiler above. */
    if (!prog_jacobi) {
        const char *jacobi_fs = "#version 120\n"
            "uniform sampler2D p_tex; uniform sampler2D b_tex;\n"
            "uniform sampler2D mask_tex; uniform sampler2D val_tex;\n"
            "uniform ivec2 dims; uniform vec2 spacing;\n"
            "uniform int color; uniform float alpha; uniform float width; uniform int use_mask;\n"
            "void main(){\n"
            "  ivec2 d = dims; vec2 px = 1.0/vec2(d);\n"
            "  vec2 uv = gl_TexCoord[0].st; vec2 idxf = floor(uv*vec2(d));\n"
            "  int ix = int(idxf.x); int iy = int(idxf.y);\n"
            "  int parity = int(mod(float(ix + iy), 2.0));\n"
            "  float b = texture2D(b_tex, uv).r;\n"
            "  float pe = texture2D(p_tex, uv + vec2(px.x,0)).r;\n"
            "  float pw = texture2D(p_tex, uv - vec2(px.x,0)).r;\n"
            "  float pn = texture2D(p_tex, uv + vec2(0,px.y)).r;\n"
            "  float ps = texture2D(p_tex, uv - vec2(0,px.y)).r;\n"
            "  float p_old = texture2D(p_tex, uv).r;\n"
            "  if (use_mask != 0) { float m = texture2D(mask_tex, uv).r; if (m > 0.5) { float val = texture2D(val_tex, uv).r; gl_FragColor = vec4(val, 0.0, 0.0, 0.0); return; } }\n"
            "  float lap = (pe + pw + pn + ps);\n"
            "  /* Use correct h^2 scaling for Poisson: h^2 = spacing.x^2 (assumes uniform spacing)\n"
            "     'width' acts as a dimensionless multiplier from the UI, so effective h2 = spacing.x^2 * width */\n"
            "  float h2 = spacing.x * spacing.x * width;\n"
            "  float new_p = (lap - h2 * b) * 0.25;\n"
            "  if (parity != color) { gl_FragColor = vec4(p_old,0,0,0); }\n"
            "  else { float p_upd = p_old + alpha * (new_p - p_old); gl_FragColor = vec4(p_upd,0,0,0); }\n"
            "}\n";
        GLuint jac_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint jac_fsh = compile_shader(GL_FRAGMENT_SHADER, jacobi_fs);
        if (jac_vs && jac_fsh) prog_jacobi = link_program(jac_vs, jac_fsh);
        if (jac_vs) glDeleteShader(jac_vs); if (jac_fsh) glDeleteShader(jac_fsh);
    }
    if (prog_jacobi) {
        glUseProgram(prog_jacobi);
        loc_jacobi_p = glGetUniformLocation(prog_jacobi, "p_tex"); if (loc_jacobi_p >= 0) glUniform1i(loc_jacobi_p, 0);
        loc_jacobi_b = glGetUniformLocation(prog_jacobi, "b_tex"); if (loc_jacobi_b >= 0) glUniform1i(loc_jacobi_b, 1);
        /* bind mask/value so jacobi honors barriers */
        GLint loc_jacobi_mask = glGetUniformLocation(prog_jacobi, "mask_tex"); if (loc_jacobi_mask >= 0) glUniform1i(loc_jacobi_mask, 2);
        GLint loc_jacobi_val = glGetUniformLocation(prog_jacobi, "val_tex"); if (loc_jacobi_val >= 0) glUniform1i(loc_jacobi_val, 3);
        /* use_mask uniform to turn on/off masking at runtime if needed */
        GLint loc_jacobi_use_mask = glGetUniformLocation(prog_jacobi, "use_mask"); if (loc_jacobi_use_mask >= 0) glUniform1i(loc_jacobi_use_mask, 1);
        loc_jacobi_dims = glGetUniformLocation(prog_jacobi, "dims"); if (loc_jacobi_dims >= 0) glUniform2i(loc_jacobi_dims, (GLint)nx, (GLint)ny);
        loc_jacobi_spacing = glGetUniformLocation(prog_jacobi, "spacing"); if (loc_jacobi_spacing >= 0) glUniform2f(loc_jacobi_spacing, (float)spacing[0], (float)spacing[1]);
        loc_jacobi_color = glGetUniformLocation(prog_jacobi, "color"); loc_jacobi_alpha = glGetUniformLocation(prog_jacobi, "alpha"); loc_jacobi_width = glGetUniformLocation(prog_jacobi, "width");
        glUseProgram(0);
    }

    //Lazy building of projection shader
    if (!prog_project) {
        const char *proj_fs = "#version 120\n"
            "uniform sampler2D vel_tex; uniform sampler2D p_tex; uniform sampler2D mask_tex; uniform sampler2D val_tex;\n"
            "uniform ivec2 dims; uniform vec2 spacing; uniform int use_mask;\n"
            "void main(){\n"
            "  vec2 uv = gl_TexCoord[0].st; vec2 px = 1.0/vec2(dims);\n"
            "  float p_e = texture2D(p_tex, uv + vec2(px.x,0)).r; float p_w = texture2D(p_tex, uv - vec2(px.x,0)).r;\n"
            "  float p_n = texture2D(p_tex, uv + vec2(0,px.y)).r; float p_s = texture2D(p_tex, uv - vec2(0,px.y)).r;\n"
            "  vec2 vel = texture2D(vel_tex, uv).rg;\n"
            "  float dpdx = (p_e - p_w) / (2.0 * spacing.x);\n"
            "  float dpdy = (p_n - p_s) / (2.0 * spacing.y);\n"
            "  vec2 vnew = vel - vec2(dpdx, dpdy);\n"
            "  if (use_mask != 0) { float m = texture2D(mask_tex, uv).r; if (m > 0.5) { vnew = vec2(0.0, 0.0); } }\n"
            "  gl_FragColor = vec4(vnew, 0.0, 0.0);\n"
            "}\n";
        GLuint proj_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint proj_fsh = compile_shader(GL_FRAGMENT_SHADER, proj_fs);
        if (proj_vs && proj_fsh) prog_project = link_program(proj_vs, proj_fsh);
        if (proj_vs) glDeleteShader(proj_vs); if (proj_fsh) glDeleteShader(proj_fsh);
    }

    if (prog_project) {
        glUseProgram(prog_project);
        loc_proj_vel = glGetUniformLocation(prog_project, "vel_tex"); if (loc_proj_vel >= 0) glUniform1i(loc_proj_vel, 0);
        loc_proj_pressure = glGetUniformLocation(prog_project, "p_tex"); if (loc_proj_pressure >= 0) glUniform1i(loc_proj_pressure, 1);
        /* mask uniforms */
        GLint loc_proj_mask = glGetUniformLocation(prog_project, "mask_tex"); if (loc_proj_mask >= 0) glUniform1i(loc_proj_mask, 2);
        GLint loc_proj_val = glGetUniformLocation(prog_project, "val_tex"); if (loc_proj_val >= 0) glUniform1i(loc_proj_val, 3);
        GLint loc_proj_use_mask = glGetUniformLocation(prog_project, "use_mask"); if (loc_proj_use_mask >= 0) glUniform1i(loc_proj_use_mask, 1);
        loc_proj_dims = glGetUniformLocation(prog_project, "dims"); if (loc_proj_dims >= 0) glUniform2i(loc_proj_dims, (GLint)nx, (GLint)ny);
        loc_proj_spacing = glGetUniformLocation(prog_project, "spacing"); if (loc_proj_spacing >= 0) glUniform2f(loc_proj_spacing, (float)spacing[0], (float)spacing[1]);
        glUseProgram(0);
    }

    /* Advection kernels: we can express semi-Lagrangian sampling as an expression
       only if the compiler supports texture lookup at an arbitrary coordinate.
       The current emitter supports texture2D lookups inside emitted code via the
       emit_expr_common helper when variables are read. Build advection via a small
       expression: advected(field) = sample(field, uv - dt * vel / spacing)
       However the expression API doesn't have an explicit sample-at-arbitrary-coord
       primitive. For simplicity, keep the hand-written advect shaders (stable and explicit). */
    if (!prog_advect_velocity) {
        const char *advect_vel_fs = "#version 120\n"
            "uniform sampler2D vel_tex; uniform sampler2D prev_vel_tex; uniform ivec2 dims; uniform vec2 spacing; uniform float dt;\n"
            "void main(){\n"
            "  vec2 uv = gl_TexCoord[0].st;\n"
            "  vec2 domain = spacing * vec2(dims); /* physical domain length in x,y */\n"
            "  vec2 vel = texture2D(prev_vel_tex, uv).rg;\n"
            "  /* dt*vel is physical displacement; divide by domain length to convert to UV space */\n"
            "  vec2 prev_pos = uv - dt * vel / domain;\n"
            "  vec2 sampled = texture2D(prev_vel_tex, prev_pos).rg;\n"
            "  gl_FragColor = vec4(sampled,0,0);\n"
            "}\n";
        GLuint adv_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint adv_fsh = compile_shader(GL_FRAGMENT_SHADER, advect_vel_fs);
        if (adv_vs && adv_fsh) prog_advect_velocity = link_program(adv_vs, adv_fsh);
        if (adv_vs) glDeleteShader(adv_vs); if (adv_fsh) glDeleteShader(adv_fsh);
    }
    if (prog_advect_velocity) {
        glUseProgram(prog_advect_velocity);
        /* prev_vel_tex is the sampler the shader reads; bind it to unit 0 where we upload tex_velocity */
        loc_advect_vel = glGetUniformLocation(prog_advect_velocity, "prev_vel_tex"); if (loc_advect_vel >= 0) glUniform1i(loc_advect_vel, 0);
        loc_advect_vel_dims = glGetUniformLocation(prog_advect_velocity, "dims"); if (loc_advect_vel_dims >= 0) glUniform2i(loc_advect_vel_dims, (GLint)nx, (GLint)ny);
        loc_advect_vel_dt = glGetUniformLocation(prog_advect_velocity, "dt"); loc_advect_vel_spacing = glGetUniformLocation(prog_advect_velocity, "spacing");
        glUseProgram(0);
    }
    /* Mask velocity: zero horizontal where current/left/right is barrier, zero vertical where current/up/down is barrier */
    if (!prog_mask_velocity) {
        const char *mask_vel_fs = "#version 120\n"
            "uniform sampler2D prev_vel_tex; uniform sampler2D mask_tex; uniform ivec2 dims;\n"
            "void main() { vec2 uv = gl_TexCoord[0].st; vec2 px = 1.0/vec2(dims); vec4 vel = texture2D(prev_vel_tex, uv);\n"
            "  float m_c = texture2D(mask_tex, uv).r; float m_l = texture2D(mask_tex, uv - vec2(px.x,0)).r; float m_r = texture2D(mask_tex, uv + vec2(px.x,0)).r;\n"
            "  float m_n = texture2D(mask_tex, uv + vec2(0,px.y)).r; float m_s = texture2D(mask_tex, uv - vec2(0,px.y)).r;\n"
            "  float vx = vel.r; float vy = vel.g;\n"
            "  if (m_c > 0.5 || m_l > 0.5 || m_r > 0.5) vx = 0.0;\n"
            "  if (m_c > 0.5 || m_n > 0.5 || m_s > 0.5) vy = 0.0;\n"
            "  gl_FragColor = vec4(vx, vy, 0.0, 0.0); }\n";
        GLuint mask_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint mask_fsh = compile_shader(GL_FRAGMENT_SHADER, mask_vel_fs);
        if (mask_vs && mask_fsh) prog_mask_velocity = link_program(mask_vs, mask_fsh);
        if (mask_vs) glDeleteShader(mask_vs); if (mask_fsh) glDeleteShader(mask_fsh);
    }
    if (prog_mask_velocity) {
        glUseProgram(prog_mask_velocity);
        loc_mask_prevvel = glGetUniformLocation(prog_mask_velocity, "prev_vel_tex"); if (loc_mask_prevvel >= 0) glUniform1i(loc_mask_prevvel, 0);
        loc_mask_mask = glGetUniformLocation(prog_mask_velocity, "mask_tex"); if (loc_mask_mask >= 0) glUniform1i(loc_mask_mask, 1);
        loc_mask_dims = glGetUniformLocation(prog_mask_velocity, "dims"); if (loc_mask_dims >= 0) glUniform2i(loc_mask_dims, (GLint)nx, (GLint)ny);
        glUseProgram(0);
    }

    /* Mask smoke: enforce smoke=0 at barrier cells to prevent scalar leakage into obstacles */
    if (!prog_mask_smoke) {
        const char *mask_smoke_fs = "#version 120\n"
            "uniform sampler2D prev_smoke_tex; uniform sampler2D mask_tex; uniform ivec2 dims;\n"
            "void main() { vec2 uv = gl_TexCoord[0].st; float s = texture2D(prev_smoke_tex, uv).r; float m_c = texture2D(mask_tex, uv).r; if (m_c > 0.5) s = 0.0; gl_FragColor = vec4(s, 0.0, 0.0, 0.0); }\n";
        GLuint mask_vs2 = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint mask_smoke_fsh = compile_shader(GL_FRAGMENT_SHADER, mask_smoke_fs);
        if (mask_vs2 && mask_smoke_fsh) prog_mask_smoke = link_program(mask_vs2, mask_smoke_fsh);
        if (mask_vs2) glDeleteShader(mask_vs2); if (mask_smoke_fsh) glDeleteShader(mask_smoke_fsh);
    }
    if (prog_mask_smoke) {
        glUseProgram(prog_mask_smoke);
        loc_mask_smoke_prev = glGetUniformLocation(prog_mask_smoke, "prev_smoke_tex"); if (loc_mask_smoke_prev >= 0) glUniform1i(loc_mask_smoke_prev, 0);
        loc_mask_smoke_mask = glGetUniformLocation(prog_mask_smoke, "mask_tex"); if (loc_mask_smoke_mask >= 0) glUniform1i(loc_mask_smoke_mask, 1);
        loc_mask_smoke_dims = glGetUniformLocation(prog_mask_smoke, "dims"); if (loc_mask_smoke_dims >= 0) glUniform2i(loc_mask_smoke_dims, (GLint)nx, (GLint)ny);
        glUseProgram(0);
    }
    /* Keep the smoke advect shader handwritten for now to maintain sampling control and stability. */
    if (!prog_advect_smoke) {
        /* Boundary-aware semi-Lagrangian: if backward sample lands inside mask, step toward current position to find first fluid sample along backtrace */
        const char *advect_smoke_fs = "#version 120\n"
            "uniform sampler2D smoke_tex; uniform sampler2D vel_tex; uniform sampler2D mask_tex; uniform ivec2 dims; uniform vec2 spacing; uniform float dt; uniform int debug_show_mask;\n"
            "void main(){\n"
            "  vec2 uv = gl_TexCoord[0].st;\n"
            "  vec2 domain = spacing * vec2(dims);\n"
            "  vec2 vel = texture2D(vel_tex, uv).rg;\n"
            "  vec2 prev_pos = uv - dt * vel / domain;\n"
            "  float s = 0.0;\n"
            "  prev_pos = clamp(prev_pos, vec2(0.0), vec2(1.0));\n"
            "  float m = texture2D(mask_tex, prev_pos).r;\n"
            "  if (m < 0.5) { s = texture2D(smoke_tex, prev_pos).r; }\n"
            "  else {\n"
            "    const int STEPS = 8;\n"
            "    for (int i = 1; i <= STEPS; ++i) {\n"
            "      float t = float(i) / float(STEPS);\n"
            "      vec2 sample_pos = mix(prev_pos, uv, t);\n"
            "      sample_pos = clamp(sample_pos, vec2(0.0), vec2(1.0));\n"
            "      float mm = texture2D(mask_tex, sample_pos).r;\n"
            "      if (mm < 0.5) { s = texture2D(smoke_tex, sample_pos).r; break; }\n"
            "    }\n"
            "    if (s == 0.0) s = texture2D(smoke_tex, uv).r;\n"
            "  }\n"
            "  if (debug_show_mask != 0) {\n"
            "    /* visual debug: red = mask at prev_pos, green = mask at current uv, blue = smoke sample */\n"
            "    float m_prev = texture2D(mask_tex, prev_pos).r; float m_uv = texture2D(mask_tex, uv).r; float s_vis = texture2D(smoke_tex, prev_pos).r;\n"
            "    gl_FragColor = vec4(m_prev, m_uv, s_vis, 1.0); return; }\n"
            "  gl_FragColor = vec4(s,0,0,0);\n"
            "}\n";
        GLuint advs_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint advs_fsh = compile_shader(GL_FRAGMENT_SHADER, advect_smoke_fs);
        if (advs_vs && advs_fsh) prog_advect_smoke = link_program(advs_vs, advs_fsh);
        if (advs_vs) glDeleteShader(advs_vs); if (advs_fsh) glDeleteShader(advs_fsh);
    }
    if (prog_advect_smoke) {
        glUseProgram(prog_advect_smoke);
        loc_advect_smoke = glGetUniformLocation(prog_advect_smoke, "smoke_tex"); if (loc_advect_smoke >= 0) glUniform1i(loc_advect_smoke, 0);
        loc_advect_smoke_dims = glGetUniformLocation(prog_advect_smoke, "dims"); if (loc_advect_smoke_dims >= 0) glUniform2i(loc_advect_smoke_dims, (GLint)nx, (GLint)ny);
        loc_advect_smoke_dt = glGetUniformLocation(prog_advect_smoke, "dt"); loc_advect_smoke_spacing = glGetUniformLocation(prog_advect_smoke, "spacing");
        loc_advect_smoke_vel = glGetUniformLocation(prog_advect_smoke, "vel_tex"); if (loc_advect_smoke_vel >= 0) glUniform1i(loc_advect_smoke_vel, 1);
        loc_advect_smoke_mask = glGetUniformLocation(prog_advect_smoke, "mask_tex"); if (loc_advect_smoke_mask >= 0) glUniform1i(loc_advect_smoke_mask, 2);
        /* debug uniform to visualize mask sampling at prev_pos/uv; default disabled */
        // debug_show_mask uniform removed
        glUseProgram(0);
    }

    /* Temperature advection (semi-Lagrangian, boundary-aware like smoke) */
    if (!prog_advect_temp) {
        const char *advect_temp_fs = "#version 120\n"
            "uniform sampler2D temp_tex; uniform sampler2D vel_tex; uniform sampler2D mask_tex; uniform ivec2 dims; uniform vec2 spacing; uniform float dt;\n"
            "void main(){ vec2 uv = gl_TexCoord[0].st; vec2 domain = spacing * vec2(dims); vec2 vel = texture2D(vel_tex, uv).rg; vec2 prev_pos = uv - dt * vel / domain; prev_pos = clamp(prev_pos, vec2(0.0), vec2(1.0)); float m = texture2D(mask_tex, prev_pos).r; float tval = 0.0; if (m < 0.5) { tval = texture2D(temp_tex, prev_pos).r; } else { const int STEPS = 8; for (int i=1;i<=STEPS;++i) { float s = float(i)/float(STEPS); vec2 p = mix(prev_pos, uv, s); p = clamp(p, vec2(0.0), vec2(1.0)); float mm = texture2D(mask_tex, p).r; if (mm < 0.5) { tval = texture2D(temp_tex, p).r; break; } } if (tval == 0.0) tval = texture2D(temp_tex, uv).r; } gl_FragColor = vec4(tval,0,0,0); }\n";
        GLuint advt_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint advt_fsh = compile_shader(GL_FRAGMENT_SHADER, advect_temp_fs);
        if (advt_vs && advt_fsh) prog_advect_temp = link_program(advt_vs, advt_fsh);
        if (advt_vs) glDeleteShader(advt_vs); if (advt_fsh) glDeleteShader(advt_fsh);
    }
    if (prog_advect_temp) {
        glUseProgram(prog_advect_temp);
        loc_advect_temp = glGetUniformLocation(prog_advect_temp, "temp_tex"); if (loc_advect_temp >= 0) glUniform1i(loc_advect_temp, 0);
        loc_advect_temp_dims = glGetUniformLocation(prog_advect_temp, "dims"); if (loc_advect_temp_dims >= 0) glUniform2i(loc_advect_temp_dims, (GLint)nx, (GLint)ny);
        loc_advect_temp_dt = glGetUniformLocation(prog_advect_temp, "dt"); loc_advect_temp_spacing = glGetUniformLocation(prog_advect_temp, "spacing");
        loc_advect_temp_vel = glGetUniformLocation(prog_advect_temp, "vel_tex"); if (loc_advect_temp_vel >= 0) glUniform1i(loc_advect_temp_vel, 1);
        loc_advect_temp_mask = glGetUniformLocation(prog_advect_temp, "mask_tex"); if (loc_advect_temp_mask >= 0) glUniform1i(loc_advect_temp_mask, 2);
        glUseProgram(0);
    }

    /* Buoyancy: compute per-cell force from temperature and add to velocity (simple additive pass) */
    if (!prog_buoyancy) {
        const char *buoy_fs = "#version 120\n"
            "uniform sampler2D vel_tex; uniform sampler2D temp_tex; uniform float beta; uniform float T0; uniform float dt; uniform vec2 gravity; uniform ivec2 dims;\n"
            "void main(){ vec2 uv = gl_TexCoord[0].st; vec2 v = texture2D(vel_tex, uv).rg; float T = texture2D(temp_tex, uv).r; float b = beta * (T - T0); vec2 force = -b * gravity * dt; v += force; gl_FragColor = vec4(v,0,0); }\n";
        GLuint b_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint b_fsh = compile_shader(GL_FRAGMENT_SHADER, buoy_fs);
        if (b_vs && b_fsh) prog_buoyancy = link_program(b_vs, b_fsh);
        if (b_vs) glDeleteShader(b_vs); if (b_fsh) glDeleteShader(b_fsh);
    }
    if (prog_buoyancy) {
        glUseProgram(prog_buoyancy);
        loc_buoyancy_temp = glGetUniformLocation(prog_buoyancy, "temp_tex"); if (loc_buoyancy_temp>=0) glUniform1i(loc_buoyancy_temp, 0);
        loc_buoyancy_vel = glGetUniformLocation(prog_buoyancy, "vel_tex"); if (loc_buoyancy_vel>=0) glUniform1i(loc_buoyancy_vel, 1);
        loc_buoyancy_beta = glGetUniformLocation(prog_buoyancy, "beta"); loc_buoyancy_T0 = glGetUniformLocation(prog_buoyancy, "T0"); loc_buoyancy_dt = glGetUniformLocation(prog_buoyancy, "dt"); loc_buoyancy_gravity = glGetUniformLocation(prog_buoyancy, "gravity");
        glUseProgram(0);
    }

    /* Simple explicit viscosity/diffusion kernels (note: implicit would be better but expensive to add here) */
    if (!prog_viscosity) {
        const char *visc_fs = "#version 120\n"
            "uniform sampler2D vel_tex; uniform ivec2 dims; uniform float nu; uniform float dt; uniform vec2 spacing;\n"
            "void main(){ vec2 uv = gl_TexCoord[0].st; ivec2 d = dims; vec2 invd = vec2(1.0/d.x,1.0/d.y); float hx = spacing.x, hy = spacing.y; vec2 v = texture2D(vel_tex, uv).rg; vec2 vL = texture2D(vel_tex, uv - vec2(invd.x,0)).rg; vec2 vR = texture2D(vel_tex, uv + vec2(invd.x,0)).rg; vec2 vB = texture2D(vel_tex, uv - vec2(0,invd.y)).rg; vec2 vT = texture2D(vel_tex, uv + vec2(0,invd.y)).rg; vec2 lap = (vL + vR - 2.0*v)/(hx*hx) + (vB + vT - 2.0*v)/(hy*hy); v += nu * dt * lap; gl_FragColor = vec4(v,0,0); }\n";
        GLuint visc_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint visc_fsh = compile_shader(GL_FRAGMENT_SHADER, visc_fs);
        if (visc_vs && visc_fsh) prog_viscosity = link_program(visc_vs, visc_fsh);
        if (visc_vs) glDeleteShader(visc_vs); if (visc_fsh) glDeleteShader(visc_fsh);
    }
    if (prog_viscosity) {
        glUseProgram(prog_viscosity);
        loc_viscosity_nu = glGetUniformLocation(prog_viscosity, "nu"); loc_viscosity_dt = glGetUniformLocation(prog_viscosity, "dt"); loc_viscosity_dims = glGetUniformLocation(prog_viscosity, "dims"); loc_viscosity_spacing = glGetUniformLocation(prog_viscosity, "spacing");
        glUseProgram(0);
    }

    /* Implicit viscosity: red-black Jacobi update for (I - nu*dt*Lap) v_new = v_old
       The shader updates one parity per invocation: uses uniform 'color' to select red/black.
       It reads vel_tex (current iterate) and rhs_vel (original advected velocity) and writes updated vel. */
    if (!prog_visc_jacobi) {
        const char *visc_jacobi_fs = "#version 120\n"
            "uniform sampler2D vel_tex; uniform sampler2D rhs_tex; uniform ivec2 dims; uniform vec2 spacing;\n"
            "uniform float nu; uniform float dt; uniform int color; uniform int use_mask;\n"
            "void main(){ vec2 uv = gl_TexCoord[0].st; ivec2 d = dims; vec2 px = 1.0/vec2(d); vec2 v = texture2D(vel_tex, uv).rg;\n"
            "  float vLx = texture2D(vel_tex, uv - vec2(px.x,0)).r; float vRx = texture2D(vel_tex, uv + vec2(px.x,0)).r; float vBx = texture2D(vel_tex, uv - vec2(0,px.y)).r; float vTx = texture2D(vel_tex, uv + vec2(0,px.y)).r;\n"
            "  float vLy = texture2D(vel_tex, uv - vec2(px.x,0)).g; float vRy = texture2D(vel_tex, uv + vec2(px.x,0)).g; float vBy = texture2D(vel_tex, uv - vec2(0,px.y)).g; float vTy = texture2D(vel_tex, uv + vec2(0,px.y)).g;\n"
            "  float alpha_x = nu * dt / (spacing.x * spacing.x); float alpha_y = nu * dt / (spacing.y * spacing.y); float denom = 1.0 + 2.0*(alpha_x + alpha_y);\n"
            "  vec2 idx = floor(uv * vec2(d)); int ix = int(idx.x); int iy = int(idx.y); int parity = int(mod(float(ix+iy), 2.0));\n"
            "  if (parity != color) { gl_FragColor = vec4(v, 0.0, 0.0); return; }\n"
            "  vec2 rhs = texture2D(rhs_tex, uv).rg;\n"
            "  float new_x = (rhs.x + alpha_x*(vLx + vRx) + alpha_y*(vBx + vTx)) / denom;\n"
            "  float new_y = (rhs.y + alpha_x*(vLy + vRy) + alpha_y*(vBy + vTy)) / denom;\n"
            "  if (use_mask != 0) { float m = texture2D(rhs_tex, uv).a; if (m > 0.5) { new_x = rhs.x; new_y = rhs.y; } }\n"
            "  gl_FragColor = vec4(new_x, new_y, 0.0, 0.0); }\n";
        GLuint vj_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint vj_fsh = compile_shader(GL_FRAGMENT_SHADER, visc_jacobi_fs);
        if (vj_vs && vj_fsh) prog_visc_jacobi = link_program(vj_vs, vj_fsh);
        if (vj_vs) glDeleteShader(vj_vs); if (vj_fsh) glDeleteShader(vj_fsh);
    }
    if (prog_visc_jacobi) {
        glUseProgram(prog_visc_jacobi);
        loc_visc_jacobi_vel = glGetUniformLocation(prog_visc_jacobi, "vel_tex"); if (loc_visc_jacobi_vel >= 0) glUniform1i(loc_visc_jacobi_vel, 0);
        loc_visc_jacobi_rhs = glGetUniformLocation(prog_visc_jacobi, "rhs_tex"); if (loc_visc_jacobi_rhs >= 0) glUniform1i(loc_visc_jacobi_rhs, 1);
        loc_visc_jacobi_dims = glGetUniformLocation(prog_visc_jacobi, "dims"); if (loc_visc_jacobi_dims >= 0) glUniform2i(loc_visc_jacobi_dims, (GLint)nx, (GLint)ny);
        loc_visc_jacobi_spacing = glGetUniformLocation(prog_visc_jacobi, "spacing"); if (loc_visc_jacobi_spacing >= 0) glUniform2f(loc_visc_jacobi_spacing, (float)spacing[0], (float)spacing[1]);
        loc_visc_jacobi_nu = glGetUniformLocation(prog_visc_jacobi, "nu"); loc_visc_jacobi_dt = glGetUniformLocation(prog_visc_jacobi, "dt"); loc_visc_jacobi_color = glGetUniformLocation(prog_visc_jacobi, "color"); loc_visc_jacobi_use_mask = glGetUniformLocation(prog_visc_jacobi, "use_mask");
        glUseProgram(0);
    }

    if (!prog_diffuse_temp) {
        const char *diff_t_fs = "#version 120\n"
            "uniform sampler2D temp_tex; uniform ivec2 dims; uniform float kappa; uniform float dt; uniform vec2 spacing;\n"
            "void main(){ vec2 uv = gl_TexCoord[0].st; ivec2 d = dims; vec2 invd = vec2(1.0/d.x,1.0/d.y); float hx = spacing.x, hy = spacing.y; float t = texture2D(temp_tex, uv).r; float tL = texture2D(temp_tex, uv - vec2(invd.x,0)).r; float tR = texture2D(temp_tex, uv + vec2(invd.x,0)).r; float tB = texture2D(temp_tex, uv - vec2(0,invd.y)).r; float tT = texture2D(temp_tex, uv + vec2(0,invd.y)).r; float lap = (tL + tR - 2.0*t)/(hx*hx) + (tB + tT - 2.0*t)/(hy*hy); t += kappa * dt * lap; gl_FragColor = vec4(t,0,0,0); }\n";
        GLuint dt_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint dt_fsh = compile_shader(GL_FRAGMENT_SHADER, diff_t_fs);
        if (dt_vs && dt_fsh) prog_diffuse_temp = link_program(dt_vs, dt_fsh);
        if (dt_vs) glDeleteShader(dt_vs); if (dt_fsh) glDeleteShader(dt_fsh);
    }
    if (prog_diffuse_temp) {
        glUseProgram(prog_diffuse_temp);
        loc_diffuse_temp_k = glGetUniformLocation(prog_diffuse_temp, "kappa"); loc_diffuse_temp_dt = glGetUniformLocation(prog_diffuse_temp, "dt"); loc_diffuse_temp_dims = glGetUniformLocation(prog_diffuse_temp, "dims"); loc_diffuse_temp_spacing = glGetUniformLocation(prog_diffuse_temp, "spacing");
        glUseProgram(0);
    }

    /* Simple copy shader to duplicate a texture into another (used to set RHS for Jacobi) */
    GLuint copy_prog = 0;
    GLint loc_copy_src = -1;
    {
        const char *copy_fs = "#version 120\nuniform sampler2D src_tex; void main(){ vec2 uv = gl_TexCoord[0].st; gl_FragColor = texture2D(src_tex, uv); }";
        GLuint cp_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
        GLuint cp_fs = compile_shader(GL_FRAGMENT_SHADER, copy_fs);
        if (cp_vs && cp_fs) copy_prog = link_program(cp_vs, cp_fs);
        if (cp_vs) glDeleteShader(cp_vs); if (cp_fs) glDeleteShader(cp_fs);
        if (copy_prog) { glUseProgram(copy_prog); loc_copy_src = glGetUniformLocation(copy_prog, "src_tex"); if (loc_copy_src >= 0) glUniform1i(loc_copy_src, 0); glUseProgram(0); }
    }

    // Set GL state (window size)
    glViewport(0,0,win_w,win_h);
    int quit = 0; SDL_Event ev;
    uint32_t last_time = SDL_GetTicks();
    int debug_readback_done = 0;

    // One-time initial display of the uploaded field so user sees the source/barrier
    glUseProgram(disp_prog);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_u_curr);
    glActiveTexture(GL_TEXTURE2); if (bm && bm->mask_tex) glBindTexture(GL_TEXTURE_2D, bm->mask_tex);
    GLint loc_src_init = glGetUniformLocation(disp_prog, "src_tex"); if (loc_src_init>=0) glUniform1i(loc_src_init, 0);
    GLint loc_mask_disp = glGetUniformLocation(disp_prog, "mask_tex"); if (loc_mask_disp>=0) glUniform1i(loc_mask_disp, 2);
    GLint loc_show_bound_init = glGetUniformLocation(disp_prog, "show_boundaries"); if (loc_show_bound_init>=0) glUniform1i(loc_show_bound_init, render.show_boundaries ? 1 : 0);
    glClearColor(0.1f,0.1f,0.12f,1.0f); glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen_quad();


    // save states
    GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
    GLboolean blendEnabled = glIsEnabled(GL_BLEND);
    if (depthEnabled) glDisable(GL_DEPTH_TEST);
    // draw overlay opaque (disable blending so mask is clearly visible)
    if (blendEnabled) glDisable(GL_BLEND);

    // draw overlay lines for mask every frame (use alpha-blend so we can update fragcolor without discard)
    if (overlay_prog && bm && bm->mask_tex) {
        glUseProgram(overlay_prog);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, bm->mask_tex);
        GLint loc_mask_ov = glGetUniformLocation(overlay_prog, "mask_tex"); if (loc_mask_ov>=0) glUniform1i(loc_mask_ov, 2);
        GLint loc_dims_ov = glGetUniformLocation(overlay_prog, "dims"); if (loc_dims_ov>=0) glUniform2i(loc_dims_ov, (GLint)nx, (GLint)ny);
        draw_fullscreen_quad();
    }
    // Restore states
    glUseProgram(0);
    if (blendEnabled) glEnable(GL_BLEND);
    if (depthEnabled) glEnable(GL_DEPTH_TEST);
    SDL_GL_SwapWindow(win);

     /* Debug helper: print one sample from the mask texture at center after
         the first frame so we can confirm the mask content the overlay will
         sample from. This is a one-shot readback to stderr. */
     int overlay_debug_print = 1;

    // Simulation loop: run until window closed
    while (!quit) {
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) quit = 1;
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) quit = 1;
            if (ev.type == SDL_KEYDOWN && (ev.key.keysym.sym == SDLK_SPACE || ev.key.keysym.sym == SDLK_TAB)) {
                // toggle pause on Space or Tab like CPU sim
                app.paused = !app.paused;
            }
            if (ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_r) {
                // Reset shortcut: trigger the same reset callback used by the menu
                cb_reset(NULL, reset_data);
                // Immediately display the uploaded initial field so the user sees the reset even when paused
                glUseProgram(disp_prog);
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_u_curr);
                glActiveTexture(GL_TEXTURE2); if (bm && bm->mask_tex) glBindTexture(GL_TEXTURE_2D, bm->mask_tex);
                GLint loc_src_tmp = glGetUniformLocation(disp_prog, "src_tex"); if (loc_src_tmp>=0) glUniform1i(loc_src_tmp, 0);
                GLint loc_mask_tmp = glGetUniformLocation(disp_prog, "mask_tex"); if (loc_mask_tmp>=0) glUniform1i(loc_mask_tmp, 2);
                glClearColor(0.1f,0.1f,0.12f,1.0f); glClear(GL_COLOR_BUFFER_BIT);
                draw_fullscreen_quad();
                SDL_GL_SwapWindow(win);
            }
            // Forward mouse events to menus (best-effort)
            if (ev.type == SDL_MOUSEBUTTONDOWN || ev.type == SDL_MOUSEBUTTONUP) {
                int mx = ev.button.x, my = ev.button.y;
                int state = (ev.type == SDL_MOUSEBUTTONDOWN) ? SDL_PRESSED : SDL_RELEASED;
                // Try base menu first, then submenus; stop when consumed
                int consumed = 0;
                if (menus && menus->base_menu && app.show_base_menu) { consumed = menu_handle_mouse_button(menus->base_menu, ev.button.button, state, mx, my); }
                if (!consumed && menus && menus->mouse_menu && app.show_mouse_controls) { consumed = menu_handle_mouse_button(menus->mouse_menu, ev.button.button, state, mx, my); }
                if (!consumed && menus && menus->sim_menu && app.show_sim_controls) { consumed = menu_handle_mouse_button(menus->sim_menu, ev.button.button, state, mx, my); }
                // only forward to source_menu when it is visible (Source mode and a source is selected)
                if (!consumed && menus && menus->source_menu && app.mouse_source && g_selected_source >= 0) { consumed = menu_handle_mouse_button(menus->source_menu, ev.button.button, state, mx, my); }
                // If a menu consumed the event, don't let the simulation also handle it
                if (consumed) continue;
                // painting: if left button down and mouse mode is Add Wave, begin painting
                if (ev.button.button == SDL_BUTTON_LEFT || ev.button.button == SDL_BUTTON_RIGHT) {
                    int is_left = (ev.button.button == SDL_BUTTON_LEFT);
                    if (state == SDL_PRESSED && app.mouse_add_wave) {
                        painting_active = 1;
                        // also paint immediately at current pos
                                /* Map window coordinates to VISIBLE subregion coordinates so clicks
                                    correspond to what the user sees (visible region is scaled to the
                                    full window). */
                                double fx = (double)mx / (double)win_w; double fy = 1.0 - (double)my / (double)win_h;
                                int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                                int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);
                        uint32_t gi = (uint32_t)gix; uint32_t gj = (uint32_t)gjy;
                        /* Left-click adds smoke (R), right-click adds pressure (G) */
                        if (is_left) paint_gaussian_to_rgba_channel(paint_buf, nx, ny, gi, gj, app.wave_amplitude * MOUSE_AMPLITUDE_SCALE, app.wave_spread, spacing[0], spacing[1], 0);
                        else paint_gaussian_to_rgba_channel(paint_buf, nx, ny, gi, gj, app.wave_amplitude * MOUSE_AMPLITUDE_SCALE, app.wave_spread, spacing[0], spacing[1], 1);
                        paint_pending = 1;
                    } else if (state == SDL_PRESSED && app.mouse_add_barrier) {
                        /* Barrier point editing: left-click/drag moves points, clicking empty space
                           adds points (pair start/end); right-click deletes the nearest point's pair.
                           Prefer most-recently added closest point when resolving ties. No Y-flip here. */
                        int mx = ev.button.x, my = ev.button.y;
                        double fx = (double)mx / (double)win_w; double fy = 1.0 - (double)my / (double)win_h; // flip for barrier mode to match GL texture orientation
                        int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                        int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);

                        // Search for closest endpoint among all segments (consider both endpoints).
                        int best_seg = -1, best_pt = -1; double best_d2 = 1e300;
                        const double pick_radius = 12.0; // pixels in window space threshold (increased)
                        for (int s = n_barrier_segs - 1; s >= 0; --s) { // iterate newest-first to prefer recent
                            // endpoint 0
                            double px0 = GX_TO_WINX(barrier_segs[s].x0, nx_vis, sponge, win_w);
                            double py0 = GY_TO_WINY(barrier_segs[s].y0, ny_vis, sponge, win_h);
                            double dx0 = (double)mx - px0; double dy0 = (double)my - py0; double d20 = dx0*dx0 + dy0*dy0;
                            if (d20 < best_d2) { best_d2 = d20; best_seg = s; best_pt = 0; }
                            // endpoint 1
                            double px1 = GX_TO_WINX(barrier_segs[s].x1, nx_vis, sponge, win_w);
                            double py1 = GY_TO_WINY(barrier_segs[s].y1, ny_vis, sponge, win_h);
                            double dx1 = (double)mx - px1; double dy1 = (double)my - py1; double d21 = dx1*dx1 + dy1*dy1;
                            if (d21 < best_d2) { best_d2 = d21; best_seg = s; best_pt = 1; }
                        }
                    } else if (state == SDL_PRESSED && app.mouse_add_barrier_brush) {
                        // Brush paint/clear: immediate effect at click position
                        int mx = ev.button.x, my = ev.button.y;
                        double fx = (double)mx / (double)win_w; double fy = 1.0 - (double)my / (double)win_h; // flip for barrier mode
                        int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                        int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);
                        int set = (ev.button.button == SDL_BUTTON_LEFT) ? 1 : 0; // left->set barrier, right->clear
                        int brush_radius = 6; // in grid cells
                        barrier_brush_paint(bm, gix, gjy, brush_radius, set);
                        boundary_mask_upload(bm, NULL);
                    } else if (state == SDL_PRESSED && app.mouse_source) {
                        // Source placement / selection / candidate for drag
                        int mx = ev.button.x, my = ev.button.y;
                        double fx = (double)mx / (double)win_w; double fy = 1.0 - (double)my / (double)win_h;
                        int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                        int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);
                        // find closest source newest-first using overlay window mapping
                        int best = -1; double bestd2 = 1e300; const double pick_radius = 24.0;
                        for (int s = g_n_sources - 1; s >= 0; --s) {
                            double px = GX_TO_WINX(g_sources[s].gx, nx_vis, sponge, win_w);
                            double py = GY_TO_WINY(g_sources[s].gy, ny_vis, sponge, win_h);
                            double dx = (double)mx - px, dy = (double)my - py; double d2 = dx*dx + dy*dy;
                            if (d2 < bestd2) { bestd2 = d2; best = s; }
                        }
                        if (best >= 0 && bestd2 <= pick_radius * pick_radius) {
                            // press near a source: record candidate for drag/select
                            source_drag_candidate = 1;
                            source_drag_cidx = best;
                            source_drag_press_mx = mx; source_drag_press_my = my;
                            source_drag_active = 0; source_drag_idx = -1;
                        } else {
                            // add new source at click position and select it; deselect others
                            Source *ns = realloc(g_sources, (size_t)(g_n_sources + 1) * sizeof(Source));
                            if (!ns) { fprintf(stderr, "Failed to alloc source\n"); }
                            else {
                                g_sources = ns; int idx = g_n_sources;
                                g_sources[idx].gx = gix; g_sources[idx].gy = gjy;
                                g_sources[idx].amp = app.wave_amplitude * SOURCE_AMP_SCALE;
                                g_sources[idx].freq = app.default_source_frequency;
                                g_sources[idx].phase = app.default_source_phase;
                                g_sources[idx].radius = sel_src_radius; /* default radius in grid units (from mouse spread) */
                                g_sources[idx].target = sel_src_target;
                                // deselect others
                                for (int k = 0; k < g_n_sources; ++k) g_sources[k].selected = 0;
                                g_sources[idx].selected = 1;
                                g_n_sources++;
                                g_selected_source = idx;
                                sel_src_amp = g_sources[idx].amp; sel_src_freq = g_sources[idx].freq; sel_src_phase = g_sources[idx].phase; sel_src_radius = g_sources[idx].radius;
                                sel_src_target = (double)g_sources[idx].target;
                                // Mark descriptors dirty so GPU gets the new source
                                sources_dirty = 1;
                                // Do NOT upload an initial gaussian when creating a source - sources influence the field each frame via paint_buf
                            }
                        }
                    } else if (state == SDL_RELEASED) {
                        painting_active = 0;
                        // If releasing mouse while dragging a barrier point, stop drag and upload
                            if (barrier_drag_active) {
                                barrier_drag_active = 0; barrier_drag_seg = -1; barrier_drag_pt = -1;
                                rebuild_barrier_mask(bm, barrier_segs, n_barrier_segs);
                                boundary_mask_upload(bm, NULL);
                            } else if (barrier_drag_candidate) {
                                // Candidate click without movement: treat as click -> delete that segment
                                if (barrier_drag_cseg >= 0 && barrier_drag_cseg < n_barrier_segs) {
                                    int seg_to_remove = barrier_drag_cseg;
                                    for (int k = seg_to_remove; k+1 < n_barrier_segs; ++k) barrier_segs[k] = barrier_segs[k+1];
                                    n_barrier_segs--;
                                    barrier_segs = n_barrier_segs ? realloc(barrier_segs, (size_t)n_barrier_segs * sizeof(BarrierSeg)) : NULL;
                                    rebuild_barrier_mask(bm, barrier_segs, n_barrier_segs);
                                    boundary_mask_upload(bm, NULL);
                                }
                                barrier_drag_candidate = 0; barrier_drag_cseg = -1; barrier_drag_cpt = -1;
                            }
                            // Handle barrier pending flow: capture release coordinates as start/end
                            if (barrier_pending) {
                                // compute release grid coords from current mouse event coordinates
                                int mx_rel = ev.button.x, my_rel = ev.button.y;
                                double fx_rel = (double)mx_rel / (double)win_w; double fy_rel = 1.0 - (double)my_rel / (double)win_h; // flip to match GL orientation
                                int gix_rel = (int)floor(fx_rel * (double)nx_vis) + sponge; if (gix_rel < (int)sponge) gix_rel = sponge; if (gix_rel >= (int)(sponge + nx_vis)) gix_rel = (int)(sponge + nx_vis - 1);
                                int gjy_rel = (int)floor(fy_rel * (double)ny_vis) + sponge; if (gjy_rel < (int)sponge) gjy_rel = sponge; if (gjy_rel >= (int)(sponge + ny_vis)) gjy_rel = (int)(sponge + ny_vis - 1);
                                if (barrier_pending_x < 0) {
                                    // first release: set pending start
                                    barrier_pending_x = gix_rel; barrier_pending_y = gjy_rel;
                                } else {
                                    // second release: create segment from pending start -> release coords
                                    barrier_segs = realloc(barrier_segs, (size_t)(n_barrier_segs+1) * sizeof(BarrierSeg));
                                    barrier_segs[n_barrier_segs].x0 = barrier_pending_x; barrier_segs[n_barrier_segs].y0 = barrier_pending_y;
                                    barrier_segs[n_barrier_segs].x1 = gix_rel; barrier_segs[n_barrier_segs].y1 = gjy_rel;
                                    barrier_segs[n_barrier_segs].thickness = 1.0; n_barrier_segs++;
                                    barrier_pending = 0; barrier_pending_x = -1; barrier_pending_y = -1;
                                    rebuild_barrier_mask(bm, barrier_segs, n_barrier_segs);
                                    boundary_mask_upload(bm, NULL);
                                }
                            }
                            // Source release handling: finalize drag or toggle selection on click
                            if (source_drag_active) {
                                // finish active drag
                                source_drag_active = 0; source_drag_idx = -1;
                                // reflect mirror controls for selected source
                                if (g_selected_source >= 0 && g_selected_source < g_n_sources) {
                                    sel_src_amp = g_sources[g_selected_source].amp;
                                        sel_src_freq = g_sources[g_selected_source].freq;
                                        sel_src_phase = g_sources[g_selected_source].phase;
                                        sel_src_radius = g_sources[g_selected_source].radius;
                                        sel_src_target = (double)g_sources[g_selected_source].target;
                                }
                            } else if (source_drag_candidate) {
                                // Candidate click without moving: toggle selection of that source
                                if (source_drag_cidx >= 0 && source_drag_cidx < g_n_sources) {
                                    int idx = source_drag_cidx;
                                    // toggle selection: select this and deselect others
                                    for (int k = 0; k < g_n_sources; ++k) g_sources[k].selected = 0;
                                    g_sources[idx].selected = 1;
                                    g_selected_source = idx;
                                    sel_src_amp = g_sources[idx].amp; sel_src_freq = g_sources[idx].freq; sel_src_phase = g_sources[idx].phase; sel_src_radius = g_sources[idx].radius;
                                }
                                source_drag_candidate = 0; source_drag_cidx = -1;
                            }
                    }
                }
            }
            if (ev.type == SDL_MOUSEWHEEL) {
                // scroll cycles through mouse modes
                int dir = ev.wheel.y > 0 ? 1 : -1;
                cycle_mouse_mode(&app, dir);
            }
            if (ev.type == SDL_MOUSEMOTION) {
                int mx = ev.motion.x, my = ev.motion.y;
                if (menus && menus->base_menu && app.show_base_menu) menu_handle_mouse_motion(menus->base_menu, mx, my);
                if (menus && menus->mouse_menu && app.show_mouse_controls) menu_handle_mouse_motion(menus->mouse_menu, mx, my);
                if (menus && menus->sim_menu && app.show_sim_controls) menu_handle_mouse_motion(menus->sim_menu, mx, my);
                if (menus && menus->source_menu && app.mouse_source && g_selected_source >= 0) menu_handle_mouse_motion(menus->source_menu, mx, my);
                if (painting_active && app.mouse_add_wave) {
                    /* Map mouse motion to the visible subregion like mouse clicks. */
                    double fx = (double)mx / (double)win_w; double fy = 1.0 - (double)my / (double)win_h;
                    int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                    int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);
                    uint32_t gi = (uint32_t)gix; uint32_t gj = (uint32_t)gjy;
                    /* Continue painting: if left button active paint smoke (R), else pressure (G) */
                    int btnstate = SDL_GetMouseState(NULL, NULL);
                    int left_down = btnstate & SDL_BUTTON(SDL_BUTTON_LEFT);
                    int right_down = btnstate & SDL_BUTTON(SDL_BUTTON_RIGHT);
                    if (left_down) paint_gaussian_to_rgba_channel(paint_buf, nx, ny, gi, gj, app.wave_amplitude * MOUSE_AMPLITUDE_SCALE, app.wave_spread, spacing[0], spacing[1], 0);
                    if (right_down) paint_gaussian_to_rgba_channel(paint_buf, nx, ny, gi, gj, app.wave_amplitude * MOUSE_AMPLITUDE_SCALE, app.wave_spread, spacing[0], spacing[1], 1);
                    paint_pending = 1;
                    paint_buf_dirty = 1;
                    paint_buf_dirty = 1;
                } else if (app.mouse_add_barrier) {
                    // If there's a candidate press (near endpoint) but not yet active, check motion threshold
                    if (barrier_drag_candidate && !barrier_drag_active) {
                        int dx = mx - barrier_drag_press_mx; int dy = my - barrier_drag_press_my;
                        if ((dx*dx + dy*dy) > (6*6)) { // motion threshold in pixels -> start drag
                            barrier_drag_active = 1;
                            barrier_drag_seg = barrier_drag_cseg;
                            barrier_drag_pt = barrier_drag_cpt;
                            barrier_drag_candidate = 0;
                        }
                    }
                    if (barrier_drag_active) {
                        // Update dragged endpoint position based on mouse motion
                        double fx = (double)mx / (double)win_w; double fy = 1.0 - (double)my / (double)win_h; // flip
                        int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                        int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);
                        if (barrier_drag_seg >= 0 && barrier_drag_seg < n_barrier_segs) {
                            if (barrier_drag_pt == 0) { barrier_segs[barrier_drag_seg].x0 = gix; barrier_segs[barrier_drag_seg].y0 = gjy; }
                            else { barrier_segs[barrier_drag_seg].x1 = gix; barrier_segs[barrier_drag_seg].y1 = gjy; }
                        }
                    }
                } else if (app.mouse_add_barrier_brush) {
                    // While dragging with mouse buttons, paint/clear under cursor
                    int btnstate = SDL_GetMouseState(NULL, NULL);
                    int left_down = btnstate & SDL_BUTTON(SDL_BUTTON_LEFT);
                    int right_down = btnstate & SDL_BUTTON(SDL_BUTTON_RIGHT);
                    if (left_down || right_down) {
                        double fx = (double)mx / (double)win_w; double fy = 1.0 - (double)my / (double)win_h; // flip
                        int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                        int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);
                        int set = left_down ? 1 : 0;
                        int brush_radius = 6;
                        barrier_brush_paint(bm, gix, gjy, brush_radius, set);
                        boundary_mask_upload(bm, NULL);
                    }
                } else if (app.mouse_source) {
                    // Source candidate/drag handling: candidate on press, become active after motion threshold
                    if (source_drag_candidate && !source_drag_active) {
                        int dx = mx - source_drag_press_mx; int dy = my - source_drag_press_my;
                        if ((dx*dx + dy*dy) > (6*6)) {
                            source_drag_active = 1;
                            source_drag_idx = source_drag_cidx;
                            source_drag_candidate = 0;
                            // select this source when starting a drag
                            if (source_drag_idx >= 0 && g_sources) {
                                g_selected_source = source_drag_idx;
                                for (int k = 0; k < g_n_sources; ++k) g_sources[k].selected = (k == g_selected_source) ? 1 : 0;
                                if (g_selected_source >= 0) {
                                    sel_src_amp = g_sources[g_selected_source].amp;
                                    sel_src_freq = g_sources[g_selected_source].freq;
                                    sel_src_phase = g_sources[g_selected_source].phase;
                                    sel_src_radius = g_sources[g_selected_source].radius;
                                }
                            }
                        }
                    }
                    if (source_drag_active && source_drag_idx >= 0 && source_drag_idx < g_n_sources) {
                        // update dragged source grid coords from mouse
                        double fx = (double)mx / (double)win_w; double fy = 1.0 - (double)my / (double)win_h;
                        int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                        int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);
                        g_sources[source_drag_idx].gx = gix;
                        g_sources[source_drag_idx].gy = gjy;
                        sources_dirty = 1;
                    }
                }
            }
        }

    // Perform up to N simulation steps per render on GPU (skip when paused)
        if (!app.paused) {
            /* pick dt and steps_per_frame from UI-controlled app state */
            dt = app.dt; /* runtime dt for sim time and damping uniforms */
            int steps_per_frame = (int)lround(app.steps_per_frame);
            if (steps_per_frame < 1) steps_per_frame = 1;
            if (steps_per_frame > 10) steps_per_frame = 10;
            for (int step = 0; step < steps_per_frame; ++step) {
                // Apply per-step sources by running the GPU source generator into tex_paint
                if (g_n_sources > 0 && g_sources && srcgen_prog) {
                    int nsrc = g_n_sources > 64 ? 64 : g_n_sources;
                    /* If descriptors changed, upload src descriptor texture once */
                    if (sources_dirty) {
                        int max_src = 64;
                        /* Fill static descriptor buffer */
                        for (int i = 0; i < nsrc; ++i) {
                            src_desc_buf[(0 * max_src + i) * 4 + 0] = (float)g_sources[i].gx;
                            src_desc_buf[(0 * max_src + i) * 4 + 1] = (float)g_sources[i].gy;
                            src_desc_buf[(0 * max_src + i) * 4 + 2] = (float)g_sources[i].amp;
                            src_desc_buf[(0 * max_src + i) * 4 + 3] = (float)g_sources[i].freq;
                            src_desc_buf[(1 * max_src + i) * 4 + 0] = (float)g_sources[i].phase;
                            src_desc_buf[(1 * max_src + i) * 4 + 1] = (float)g_sources[i].radius;
                            src_desc_buf[(1 * max_src + i) * 4 + 2] = (float)g_sources[i].target; /* 0=smoke,1=pressure */
                            src_desc_buf[(1 * max_src + i) * 4 + 3] = 0.0f;
                        }
                        glBindTexture(GL_TEXTURE_2D, tex_src_desc);
                        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, max_src, 2, GL_RGBA, GL_FLOAT, src_desc_buf);
                        
                        sources_dirty = 0;
                    }
                    /* Bind FBO to render into tex_paint */
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_paint, 0);
                    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
                    if (st == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0, 0, (GLsizei)nx, (GLsizei)ny);
                        glUseProgram(srcgen_prog);
                        if (loc_src_n >= 0) glUniform1i(loc_src_n, nsrc);
                        if (loc_src_time >= 0) glUniform1f(loc_src_time, (float)((double)sim_step_counter * dt));
                        /* max_src matches descriptor texture width (allocated as 64) */
                        if (loc_max_src >= 0) glUniform1i(loc_max_src, 64);
                        if (loc_src_dims >= 0) glUniform2i(loc_src_dims, (GLint)nx, (GLint)ny);
                        if (loc_src_spacing >= 0) glUniform2f(loc_src_spacing, (float)spacing[0], (float)spacing[1]);
                        /* bind descriptor texture to texture unit 4 */
                        glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, tex_src_desc);
                        if (loc_src_desc >= 0) glUniform1i(loc_src_desc, 4);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        /* Clear to black for source generation render target. */
                        glClearColor(0, 0, 0, 0);
                        glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                        paint_pending = 1;
                        paint_from_gpu = 1;
                    } else {
                        fprintf(stderr, "FBO incomplete for source generation: 0x%x\n", st);
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                }

                // If we have pending paint, upload paint buffer and composite into texture
                if (paint_pending) {
                    /* If CPU paint exists, upload it into tex_paint_cpu once */
                    if (paint_buf_dirty) {
                        glBindTexture(GL_TEXTURE_2D, tex_paint_cpu);
                        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, paint_buf);
                        paint_buf_dirty = 0;
                        paint_cpu_uploaded = 1; /* remember we uploaded non-zero CPU paint */
                    }
                    // composite: render src=tex_u_curr + paint -> tex_out
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_out, 0);
                    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
                    if (st == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0, 0, (GLsizei)nx, (GLsizei)ny);
                        glUseProgram(p_prog);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_u_curr);
                        /* bind GPU paint to unit 1 */
                        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_paint);
                        /* bind CPU paint to unit 2 */
                        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_paint_cpu);
                        if (loc_p_src >= 0) glUniform1i(loc_p_src, 0);
                        if (loc_p_paint_gpu >= 0) glUniform1i(loc_p_paint_gpu, 1);
                        if (loc_p_paint_cpu >= 0) glUniform1i(loc_p_paint_cpu, 2);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                        GLuint tmp = tex_u_curr; tex_u_curr = tex_out; tex_out = tmp;
                    } else {
                        fprintf(stderr, "FBO incomplete for paint composite: 0x%x\n", st);
                        fflush(stderr);
                    }
                          /* Additionally composite paint into smoke and pressure textures (GPU and CPU paint combined).
                             Render-add R channel into tex_smoke and G channel into tex_pressure. */
                          if (prog_add_paint_smoke) {
                              /* render into tmp smoke texture, sampling current smoke + paints */
                              glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                              glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_smoke_tmp, 0);
                              if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                                  glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                                  glUseProgram(prog_add_paint_smoke);
                                  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_smoke); /* input */
                                  glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_paint);
                                  glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_paint_cpu);
                                  draw_fullscreen_quad(); glFlush();
                              }
                              glBindFramebuffer(GL_FRAMEBUFFER, 0);
                              glViewport(0,0,win_w,win_h);
                              { GLuint ttmp = tex_smoke; tex_smoke = tex_smoke_tmp; tex_smoke_tmp = ttmp; }
                          }
                          if (prog_add_paint_pressure) {
                              /* render into tmp pressure texture, sampling current pressure + paints */
                              glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                              glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_pressure_tmp, 0);
                              if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                                  glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                                  glUseProgram(prog_add_paint_pressure);
                                  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_pressure); /* input */
                                  glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_paint);
                                  glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_paint_cpu);
                                  draw_fullscreen_quad(); glFlush();
                              }
                              glBindFramebuffer(GL_FRAMEBUFFER, 0);
                              glViewport(0,0,win_w,win_h);
                              { GLuint ttmp = tex_pressure; tex_pressure = tex_pressure_tmp; tex_pressure_tmp = ttmp; }
                              /* Debug: report min/max/NaN on pressure to help diagnose spikes */
                          }
                         if (prog_add_paint_temp) {
                              /* render into tmp temperature texture, sampling current temp + paints (B channel) */
                              glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                              glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_temp_tmp, 0);
                              if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                                  glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                                  glUseProgram(prog_add_paint_temp);
                                  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_temp); /* input */
                                  glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_paint);
                                  glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tex_paint_cpu);
                                  draw_fullscreen_quad(); glFlush();
                              }
                              glBindFramebuffer(GL_FRAMEBUFFER, 0);
                              glViewport(0,0,win_w,win_h);
                              { GLuint ttmp = tex_temp; tex_temp = tex_temp_tmp; tex_temp_tmp = ttmp; }
                          }
                          /* clear CPU paint buffer and upload zeros to cpu paint texture so
                              CPU additions are applied only once (until user paints again) */
                          size_t psize = (size_t)nx * ny * 4 * sizeof(float);
                          /* If we uploaded CPU paint earlier, clear both host buffer and GPU texture.
                             Otherwise skip the expensive zero upload to keep frames fast when only GPU sources run. */
                          if (paint_cpu_uploaded) {
                              if (paint_buf) { memset(paint_buf, 0, psize); }
                              paint_buf_dirty = 0;
                              glBindTexture(GL_TEXTURE_2D, tex_paint_cpu);
                              glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                              glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGBA, GL_FLOAT, paint_buf);
                              paint_cpu_uploaded = 0;
                          } else {
                              /* ensure host paint buffer isn't considered dirty */
                              if (paint_buf) { /* keep it as-is (likely zero) */ }
                              paint_buf_dirty = 0;
                          }
                          paint_pending = 0;
                          paint_from_gpu = 0;
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                }
                glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_out, 0);
                GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
                if (status != GL_FRAMEBUFFER_COMPLETE) { fprintf(stderr, "FBO incomplete: 0x%x\n", status); break; }
                /* ensure we render at texture resolution so fragments map 1:1 to texels */
                glViewport(0, 0, (GLsizei)nx, (GLsizei)ny);

                glUseProgram(compute_prog);
                // Bind inputs: u_curr -> unit 0, u_prev -> unit 1
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_u_curr);
                glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_u_prev);
                // Bind mask/value textures if present -> units 2,3
                if (bm && bm->mask_tex) { glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, bm->mask_tex); }
                if (bm && bm->values_tex) { glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, bm->values_tex); }

                // Set uniforms while program is bound (ensure sampler units are wired)
                if (loc_u_curr >= 0) glUniform1i(loc_u_curr, 0);
                if (loc_u_prev >= 0) glUniform1i(loc_u_prev, 1);
                if (loc_mask >= 0) glUniform1i(loc_mask, 2);
                if (loc_val >= 0) glUniform1i(loc_val, 3);
                if (loc_dims >= 0) glUniform2i(loc_dims, (GLint)nx, (GLint)ny);
                if (loc_spacing >= 0) glUniform2f(loc_spacing, (float)spacing[0], (float)spacing[1]);
                if (loc_use_mask >= 0) glUniform1i(loc_use_mask, bm ? 1 : 0);

                // Draw quad to compute u_next
                glDrawBuffer(GL_COLOR_ATTACHMENT0);
                glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                /* GPU timing: sample occasionally to avoid blocking every frame. */
                uint32_t _gpu_t_before = 0;
                if (gpu_time_sample_period > 0) _gpu_t_before = SDL_GetTicks();
                draw_fullscreen_quad();
                glFlush();
                gpu_time_step_counter++;
                if (gpu_time_sample_period > 0 && gpu_time_step_counter >= gpu_time_sample_period) {
                    glFinish();
                    uint32_t _gpu_t_after = SDL_GetTicks();
                    uint32_t _elapsed = (_gpu_t_after > _gpu_t_before) ? (_gpu_t_after - _gpu_t_before) : 0;
                    double ms_per_step = (double)_elapsed / (double)gpu_time_sample_period;
                    const double alpha = 0.2;
                    if (gpu_time_samples == 0) gpu_compute_ms_avg = ms_per_step;
                    else gpu_compute_ms_avg = alpha * ms_per_step + (1.0 - alpha) * gpu_compute_ms_avg;
                    gpu_time_samples++;
                    gpu_time_step_counter = 0;
                }

                // Ping-pong: rotate textures so the newly computed tex_out becomes current
                GLuint tex_prev = tex_u_prev;
                tex_u_prev = tex_u_curr;
                tex_u_curr = tex_out;
                tex_out = tex_prev;

                // increment sim counter for each physics step performed
                sim_step_counter++;
                /* --- pressure/divergence/velocity/smoke pipeline reordered: divergence/solve/project moved after velocity advection --- */

                // 3.5) Mask velocity against barriers: zero orthogonal components around barrier cells
                if (prog_mask_velocity) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_velocity_tmp, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        glUseProgram(prog_mask_velocity);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_velocity); /* prev vel */
                        glActiveTexture(GL_TEXTURE1); if (bm && bm->mask_tex) glBindTexture(GL_TEXTURE_2D, bm->mask_tex); else glBindTexture(GL_TEXTURE_2D, 0);
                        if (loc_mask_prevvel >= 0) glUniform1i(loc_mask_prevvel, 0);
                        if (loc_mask_mask >= 0) glUniform1i(loc_mask_mask, 1);
                        if (loc_mask_dims >= 0) glUniform2i(loc_mask_dims, (GLint)nx, (GLint)ny);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                    { GLuint ttmp = tex_velocity; tex_velocity = tex_velocity_tmp; tex_velocity_tmp = ttmp; }
                }

                // 4) Advect velocity semi-Lagrangian
                if (prog_advect_velocity) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_velocity_tmp, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        glUseProgram(prog_advect_velocity);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_velocity);
                        if (loc_advect_vel >= 0) glUniform1i(loc_advect_vel, 0);
                        if (loc_advect_vel_dims >= 0) glUniform2i(loc_advect_vel_dims, (GLint)nx, (GLint)ny);
                        if (loc_advect_vel_dt >= 0) glUniform1f(loc_advect_vel_dt, (float)dt);
                        if (loc_advect_vel_spacing >= 0) glUniform2f(loc_advect_vel_spacing, (float)spacing[0], (float)spacing[1]);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                    { GLuint ttmp = tex_velocity; tex_velocity = tex_velocity_tmp; tex_velocity_tmp = ttmp; }
                    /* velocity advect debug logging removed to focus on smoke advection tests */
                    (void)0;
                }

                /* --- Now compute divergence from the (masked & advected) velocity and solve/project --- */
                // 1) Compute divergence from current velocity into tex_divergence
                if (prog_divergence) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_divergence, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        /* Extract components into single-channel textures and run compiled divergence (vx_tex/vy_tex) */
                        if (prog_divergence && prog_extract_vx && prog_extract_vy) {
                            /* extract vx into tex_velocity_x */
                            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_velocity_x, 0);
                            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                                glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                                glUseProgram(prog_extract_vx);
                                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_velocity);
                                draw_fullscreen_quad(); glFlush();
                            }
                            /* extract vy into tex_velocity_y */
                            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_velocity_y, 0);
                            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                                glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                                glUseProgram(prog_extract_vy);
                                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_velocity);
                                draw_fullscreen_quad(); glFlush();
                            }
                            // ensure the divergence program writes into tex_divergence
                            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_divergence, 0);
                            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                                glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                                /* Now run compiled divergence sampling vx_tex (unit 4) and vy_tex (unit 5) */
                                glUseProgram(prog_divergence);
                            glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, tex_velocity_x);
                            glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, tex_velocity_y);
                            if (loc_div_vx >= 0) glUniform1i(loc_div_vx, 4);
                            if (loc_div_vy >= 0) glUniform1i(loc_div_vy, 5);
                            /* bind mask/value textures to units 2/3 for the compiled shader if present */
                            glActiveTexture(GL_TEXTURE2); if (bm && bm->mask_tex) glBindTexture(GL_TEXTURE_2D, bm->mask_tex); else glBindTexture(GL_TEXTURE_2D, 0);
                            glActiveTexture(GL_TEXTURE3); if (bm && bm->values_tex) glBindTexture(GL_TEXTURE_2D, bm->values_tex); else glBindTexture(GL_TEXTURE_2D, 0);
                            if (loc_div_dims >= 0) glUniform2i(loc_div_dims, (GLint)nx, (GLint)ny);
                            if (loc_div_spacing >= 0) glUniform2f(loc_div_spacing, (float)spacing[0], (float)spacing[1]);
                                glDrawBuffer(GL_COLOR_ATTACHMENT0);
                                glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                                draw_fullscreen_quad(); glFlush();
                            }
                        }
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                }

                // 2) Solve pressure using red-black SOR on GPU with prog_jacobi (ping-pong tex_pressure/tex_pressure_tmp)
                if (prog_jacobi) {
                    int iters = app.sor_iters > 0 ? app.sor_iters : 40;
                    float alpha = (float)(app.sor_alpha);
                    float width = (float)(app.pressure_width);
                    for (int it = 0; it < iters; ++it) {
                        int color = it & 1; // alternate red/black each iteration
                        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_pressure_tmp, 0);
                        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                            glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                            glUseProgram(prog_jacobi);
                            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_pressure);
                            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_divergence);
                            if (loc_jacobi_p >= 0) glUniform1i(loc_jacobi_p, 0);
                            if (loc_jacobi_b >= 0) glUniform1i(loc_jacobi_b, 1);
                            if (loc_jacobi_color >= 0) glUniform1i(loc_jacobi_color, color);
                            if (loc_jacobi_alpha >= 0) glUniform1f(loc_jacobi_alpha, alpha);
                            if (loc_jacobi_width >= 0) glUniform1f(loc_jacobi_width, width);
                            if (loc_jacobi_dims >= 0) glUniform2i(loc_jacobi_dims, (GLint)nx, (GLint)ny);
                            if (loc_jacobi_spacing >= 0) glUniform2f(loc_jacobi_spacing, (float)spacing[0], (float)spacing[1]);
                            glDrawBuffer(GL_COLOR_ATTACHMENT0);
                            glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                            draw_fullscreen_quad(); glFlush();
                        }
                        glBindFramebuffer(GL_FRAMEBUFFER, 0);
                        glViewport(0,0,win_w,win_h);
                        // swap pressure textures
                        GLuint ttmp = tex_pressure; tex_pressure = tex_pressure_tmp; tex_pressure_tmp = ttmp;
                    }
                }

                // 3) Project: subtract pressure gradient from velocity
                if (prog_project) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_velocity_tmp, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        glUseProgram(prog_project);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_velocity);
                        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_pressure);
                        if (loc_proj_vel >= 0) glUniform1i(loc_proj_vel, 0);
                        if (loc_proj_pressure >= 0) glUniform1i(loc_proj_pressure, 1);
                        if (loc_proj_dims >= 0) glUniform2i(loc_proj_dims, (GLint)nx, (GLint)ny);
                        if (loc_proj_spacing >= 0) glUniform2f(loc_proj_spacing, (float)spacing[0], (float)spacing[1]);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                    // swap velocity textures
                    { GLuint ttmp = tex_velocity; tex_velocity = tex_velocity_tmp; tex_velocity_tmp = ttmp; }
                }

                /* Re-apply velocity mask after projection so the final velocity field
                 * used for smoke advection has orthogonal components zeroed near barriers. */
                if (prog_mask_velocity) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_velocity_tmp, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        glUseProgram(prog_mask_velocity);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_velocity); /* prev vel */
                        glActiveTexture(GL_TEXTURE1); if (bm && bm->mask_tex) glBindTexture(GL_TEXTURE_2D, bm->mask_tex); else glBindTexture(GL_TEXTURE_2D, 0);
                        if (loc_mask_prevvel >= 0) glUniform1i(loc_mask_prevvel, 0);
                        if (loc_mask_mask >= 0) glUniform1i(loc_mask_mask, 1);
                        if (loc_mask_dims >= 0) glUniform2i(loc_mask_dims, (GLint)nx, (GLint)ny);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                    { GLuint ttmp = tex_velocity; tex_velocity = tex_velocity_tmp; tex_velocity_tmp = ttmp; }
                }

                /* Periodic debug readback: compute average kinetic energy and max velocity magnitude
                   every ke_readback_interval simulation steps. Controlled by debug_per_stage. */
                const int ke_readback_interval = 60; /* steps between readbacks */
                if (debug_per_stage && (ke_readback_interval > 0) && ((int)(sim_step_counter % ke_readback_interval) == 0)) {
                    size_t pixels = (size_t)nx * (size_t)ny;
                    float *vbuf = (float*)malloc(pixels * 4 * sizeof(float));
                    if (vbuf) {
                        glBindTexture(GL_TEXTURE_2D, tex_velocity);
                        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, vbuf);
                        double sum_ke = 0.0; double maxm = 0.0; size_t nan_count = 0;
                        for (size_t pi = 0; pi < pixels; ++pi) {
                            double vx = vbuf[pi*4 + 0]; double vy = vbuf[pi*4 + 1];
                            double mag2 = vx*vx + vy*vy;
                            if (!(mag2 == mag2)) { nan_count++; continue; }
                            sum_ke += 0.5 * mag2; /* per-cell kinetic energy (unit mass) */
                            if (mag2 > maxm*maxm) maxm = sqrt(mag2);
                        }
                        double mean_ke = sum_ke / (double)pixels;
                        fprintf(stderr, "DEBUG KE: step=%llu mean_ke=%g max_vel=%g nan_count=%zu\n", (unsigned long long)sim_step_counter, mean_ke, maxm, nan_count);
                        fflush(stderr);
                        free(vbuf);
                    } else {
                        fprintf(stderr, "DEBUG: failed to allocate velocity readback buffer (%zu bytes)\n", pixels * 4 * sizeof(float)); fflush(stderr);
                    }
                }

                // 5) Advect smoke (if enabled)
                /* Advect temperature (same boundary-aware SL as smoke) */
                if (prog_advect_temp) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_temp_tmp, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        glUseProgram(prog_advect_temp);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_temp);
                        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_velocity);
                        glActiveTexture(GL_TEXTURE2); if (bm && bm->mask_tex) glBindTexture(GL_TEXTURE_2D, bm->mask_tex); else glBindTexture(GL_TEXTURE_2D, 0);
                        if (loc_advect_temp >= 0) glUniform1i(loc_advect_temp, 0);
                        if (loc_advect_temp_vel >= 0) glUniform1i(loc_advect_temp_vel, 1);
                        if (loc_advect_temp_mask >= 0) glUniform1i(loc_advect_temp_mask, 2);
                        if (loc_advect_temp_dims >= 0) glUniform2i(loc_advect_temp_dims, (GLint)nx, (GLint)ny);
                        if (loc_advect_temp_dt >= 0) glUniform1f(loc_advect_temp_dt, (float)dt);
                        if (loc_advect_temp_spacing >= 0) glUniform2f(loc_advect_temp_spacing, (float)spacing[0], (float)spacing[1]);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                    { GLuint ttmp = tex_temp; tex_temp = tex_temp_tmp; tex_temp_tmp = ttmp; }
                }

                if (prog_advect_smoke) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_smoke_tmp, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        glUseProgram(prog_advect_smoke);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_smoke);
                        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_velocity);
                        glActiveTexture(GL_TEXTURE2); if (bm && bm->mask_tex) glBindTexture(GL_TEXTURE_2D, bm->mask_tex); else glBindTexture(GL_TEXTURE_2D, 0);
                        if (loc_advect_smoke >= 0) glUniform1i(loc_advect_smoke, 0);
                        if (loc_advect_smoke_vel >= 0) glUniform1i(loc_advect_smoke_vel, 1);
                        if (loc_advect_smoke_mask >= 0) glUniform1i(loc_advect_smoke_mask, 2);
                        if (loc_advect_smoke_dims >= 0) glUniform2i(loc_advect_smoke_dims, (GLint)nx, (GLint)ny);
                        if (loc_advect_smoke_dt >= 0) glUniform1f(loc_advect_smoke_dt, (float)dt);
                        if (loc_advect_smoke_spacing >= 0) glUniform2f(loc_advect_smoke_spacing, (float)spacing[0], (float)spacing[1]);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                    /* Mask smoke so barrier cells and their immediate neighbors are forced to zero */
                    if (prog_mask_smoke) {
                        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_smoke, 0);
                        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                            glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                            glUseProgram(prog_mask_smoke);
                            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_smoke_tmp);
                            glActiveTexture(GL_TEXTURE1); if (bm && bm->mask_tex) glBindTexture(GL_TEXTURE_2D, bm->mask_tex); else glBindTexture(GL_TEXTURE_2D, 0);
                            if (loc_mask_smoke_prev >= 0) glUniform1i(loc_mask_smoke_prev, 0);
                            if (loc_mask_smoke_mask >= 0) glUniform1i(loc_mask_smoke_mask, 1);
                            if (loc_mask_smoke_dims >= 0) glUniform2i(loc_mask_smoke_dims, (GLint)nx, (GLint)ny);
                            glDrawBuffer(GL_COLOR_ATTACHMENT0);
                            glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                            draw_fullscreen_quad(); glFlush();
                        }
                        glBindFramebuffer(GL_FRAMEBUFFER, 0);
                        glViewport(0,0,win_w,win_h);
                    }
                }

                /* Apply buoyancy force to velocity: reads temperature and velocity, writes vel_tmp */
                if (prog_buoyancy) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_velocity_tmp, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        glUseProgram(prog_buoyancy);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_temp);
                        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_velocity);
                        if (loc_buoyancy_temp >= 0) glUniform1i(loc_buoyancy_temp, 0);
                        if (loc_buoyancy_vel >= 0) glUniform1i(loc_buoyancy_vel, 1);
                        if (loc_buoyancy_beta >= 0) glUniform1f(loc_buoyancy_beta, (float)app.buoyancy_beta);
                        if (loc_buoyancy_T0 >= 0) glUniform1f(loc_buoyancy_T0, 0.0f);
                        if (loc_buoyancy_dt >= 0) glUniform1f(loc_buoyancy_dt, (float)dt);
                        if (loc_buoyancy_gravity >= 0) glUniform2f(loc_buoyancy_gravity, (float)app.gravity_x, (float)app.gravity_y);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                    { GLuint ttmp = tex_velocity; tex_velocity = tex_velocity_tmp; tex_velocity_tmp = ttmp; }
                }

                /* Implicit viscosity: solve (I - nu*dt*Lap) v_new = v_old using red-black Jacobi iterations.
                We copy the current velocity into tex_velocity_rhs (rhs) once, then perform N iterations
                ping-ponging tex_velocity <-> tex_velocity_tmp using prog_visc_jacobi. The RHS is bound as unit 1. */
                if (prog_visc_jacobi && app.viscosity_nu > 0.0 && app.visc_iters >= 1.0) {
                    int iters = (int)lround(app.visc_iters);
                    /* copy current velocity into rhs texture */
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_velocity_rhs, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        glUseProgram(copy_prog);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_velocity);
                        if (loc_copy_src >= 0) glUniform1i(loc_copy_src, 0);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                    /* perform red-black jacobi iterations */
                    for (int it = 0; it < iters; ++it) {
                        int color = it & 1;
                        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_velocity_tmp, 0);
                        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                            glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                            glUseProgram(prog_visc_jacobi);
                            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_velocity); /* current iterate */
                            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_velocity_rhs); /* rhs */
                            if (loc_visc_jacobi_vel >= 0) glUniform1i(loc_visc_jacobi_vel, 0);
                            if (loc_visc_jacobi_dims >= 0) glUniform2i(loc_visc_jacobi_dims, (GLint)nx, (GLint)ny);
                            if (loc_visc_jacobi_spacing >= 0) glUniform2f(loc_visc_jacobi_spacing, (float)spacing[0], (float)spacing[1]);
                            if (loc_visc_jacobi_nu >= 0) glUniform1f(loc_visc_jacobi_nu, (float)app.viscosity_nu);
                            if (loc_visc_jacobi_dt >= 0) glUniform1f(loc_visc_jacobi_dt, (float)dt);
                            if (loc_visc_jacobi_color >= 0) glUniform1i(loc_visc_jacobi_color, color);
                            if (loc_visc_jacobi_use_mask >= 0) glUniform1i(loc_visc_jacobi_use_mask, bm ? 1 : 0);
                            glDrawBuffer(GL_COLOR_ATTACHMENT0);
                            glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                            draw_fullscreen_quad(); glFlush();
                        }
                        glBindFramebuffer(GL_FRAMEBUFFER, 0);
                        glViewport(0,0,win_w,win_h);
                        { GLuint ttmp = tex_velocity; tex_velocity = tex_velocity_tmp; tex_velocity_tmp = ttmp; }
                    }
                }

        /* Diffuse temperature (explicit) */
        if (prog_diffuse_temp && app.temp_kappa > 0.0) {
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_temp_tmp, 0);
                    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
                        glViewport(0,0,(GLsizei)nx,(GLsizei)ny);
                        glUseProgram(prog_diffuse_temp);
                        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_temp);
            if (loc_diffuse_temp_k >= 0) glUniform1f(loc_diffuse_temp_k, (float)app.temp_kappa);
            if (loc_diffuse_temp_dt >= 0) glUniform1f(loc_diffuse_temp_dt, (float)dt);
            if (loc_diffuse_temp_dims >= 0) glUniform2i(loc_diffuse_temp_dims, (GLint)nx, (GLint)ny);
            if (loc_diffuse_temp_spacing >= 0) glUniform2f(loc_diffuse_temp_spacing, (float)spacing[0], (float)spacing[1]);
                        glDrawBuffer(GL_COLOR_ATTACHMENT0);
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
                        draw_fullscreen_quad(); glFlush();
                    }
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0,0,win_w,win_h);
                    { GLuint ttmp = tex_temp; tex_temp = tex_temp_tmp; tex_temp_tmp = ttmp; }
                }
                /* --- End pressure/divergence/velocity/smoke pipeline --- */
            } // end steps_per_frame loop
        } // end not paused

        // Unbind FBO and restore window viewport
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0,0,win_w,win_h);

        // Render current field to screen using unified display shader
        glUseProgram(disp_prog);
        /* bind textures to fixed units expected by shader */
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex_u_curr); /* src */
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex_velocity); /* vel */
        glActiveTexture(GL_TEXTURE2); if (bm && bm->mask_tex) glBindTexture(GL_TEXTURE_2D, bm->mask_tex); else glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, tex_pressure);
        glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, tex_smoke);
    glActiveTexture(GL_TEXTURE6); glBindTexture(GL_TEXTURE_2D, tex_temp);
        GLint loc_src = glGetUniformLocation(disp_prog, "src_tex"); if (loc_src>=0) glUniform1i(loc_src, 0);
        GLint loc_vel = glGetUniformLocation(disp_prog, "vel_tex"); if (loc_vel>=0) glUniform1i(loc_vel, 1);
        GLint loc_maskd = glGetUniformLocation(disp_prog, "mask_tex"); if (loc_maskd>=0) glUniform1i(loc_maskd, 2);
        GLint loc_ptex = glGetUniformLocation(disp_prog, "pressure_tex"); if (loc_ptex>=0) glUniform1i(loc_ptex, 3);
        GLint loc_stex = glGetUniformLocation(disp_prog, "smoke_tex"); if (loc_stex>=0) glUniform1i(loc_stex, 4);
    GLint loc_ttex = glGetUniformLocation(disp_prog, "temp_tex"); if (loc_ttex>=0) glUniform1i(loc_ttex, 6);
        GLint loc_mode = glGetUniformLocation(disp_prog, "render_mode"); if (loc_mode>=0) glUniform1i(loc_mode, render.mode);
        GLint loc_vscl = glGetUniformLocation(disp_prog, "value_scale"); if (loc_vscl>=0) glUniform1f(loc_vscl, (float)render.value_scale);
        GLint loc_dims_disp = glGetUniformLocation(disp_prog, "dims"); if (loc_dims_disp>=0) glUniform2i(loc_dims_disp, (GLint)nx, (GLint)ny);
        GLint loc_vis_off_disp = glGetUniformLocation(disp_prog, "vis_offset"); if (loc_vis_off_disp>=0) glUniform2i(loc_vis_off_disp, 0, 0);
        GLint loc_vis_size_disp = glGetUniformLocation(disp_prog, "vis_size"); if (loc_vis_size_disp>=0) glUniform2i(loc_vis_size_disp, (GLint)nx_vis, (GLint)ny_vis);
        GLint loc_spacing_disp = glGetUniformLocation(disp_prog, "spacing"); if (loc_spacing_disp>=0) glUniform2f(loc_spacing_disp, (float)spacing[0], (float)spacing[1]);
        GLint loc_view_px = glGetUniformLocation(disp_prog, "view_px"); if (loc_view_px>=0) glUniform2f(loc_view_px, (float)win_w, (float)win_h);
        GLint loc_show_bound = glGetUniformLocation(disp_prog, "show_boundaries"); if (loc_show_bound>=0) glUniform1i(loc_show_bound, render.show_boundaries ? 1 : 0);
        glClearColor(0.1f,0.1f,0.12f,1.0f); glClear(GL_COLOR_BUFFER_BIT);
        draw_fullscreen_quad();
        // Unbind any GL program so menu uses fixed-function pipeline rendering
        glUseProgram(0);
        // Draw overlay lines for mask on top of the display but beneath menus
        if (overlay_prog && bm && bm->mask_tex) {
            GLboolean depthEnabled_ov = glIsEnabled(GL_DEPTH_TEST);
            GLboolean blendEnabled_ov = glIsEnabled(GL_BLEND);
            if (depthEnabled_ov) glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram(overlay_prog);
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, bm->mask_tex);
            GLint loc_mask_ov = glGetUniformLocation(overlay_prog, "mask_tex"); if (loc_mask_ov>=0) glUniform1i(loc_mask_ov, 2);
            GLint loc_dims_ov = glGetUniformLocation(overlay_prog, "dims"); if (loc_dims_ov>=0) glUniform2i(loc_dims_ov, (GLint)nx, (GLint)ny);
            GLint loc_vis_off_ov = glGetUniformLocation(overlay_prog, "vis_offset"); if (loc_vis_off_ov>=0) glUniform2i(loc_vis_off_ov, 0, 0);
            GLint loc_vis_size_ov = glGetUniformLocation(overlay_prog, "vis_size"); if (loc_vis_size_ov>=0) glUniform2i(loc_vis_size_ov, (GLint)nx_vis, (GLint)ny_vis);
            draw_fullscreen_quad();
            // restore states
            glUseProgram(0);
            if (!blendEnabled_ov) glDisable(GL_BLEND);
            if (depthEnabled_ov) glEnable(GL_DEPTH_TEST);
        }

        // Ensure menus draw on top: disable depth test and enable alpha blending
        GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
        if (depthEnabled) glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        // Render menus on top (ensure texture unit 0 is active for the fixed-function text renderer)
        glActiveTexture(GL_TEXTURE0);
        // Draw barrier endpoints as small white squares so points are visible to the user
        if (n_barrier_segs > 0) {
            // Prepare orthographic projection for pixel-aligned quads
            glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
            glDisable(GL_TEXTURE_2D);
            glDisable(GL_LIGHTING);
            glColor3f(1.0f, 1.0f, 1.0f);
            glMatrixMode(GL_PROJECTION);
            glPushMatrix(); glLoadIdentity(); glOrtho(0, win_w, win_h, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix(); glLoadIdentity();
            const float ps = 6.0f; // point square size in pixels
            for (int s = 0; s < n_barrier_segs; ++s) {
                // endpoint 0
                double px0 = (double)(barrier_segs[s].x0 - sponge) / (double)nx_vis * win_w;
                double py0 = (double)(barrier_segs[s].y0 - sponge) / (double)ny_vis * win_h;
                glBegin(GL_QUADS);
                glVertex2f((float)(px0 - ps*0.5), (float)(py0 - ps*0.5));
                glVertex2f((float)(px0 + ps*0.5), (float)(py0 - ps*0.5));
                glVertex2f((float)(px0 + ps*0.5), (float)(py0 + ps*0.5));
                glVertex2f((float)(px0 - ps*0.5), (float)(py0 + ps*0.5));
                glEnd();
                // endpoint 1
                double px1 = (double)(barrier_segs[s].x1 - sponge) / (double)nx_vis * win_w;
                double py1 = (double)(barrier_segs[s].y1 - sponge) / (double)ny_vis * win_h;
                glBegin(GL_QUADS);
                glVertex2f((float)(px1 - ps*0.5), (float)(py1 - ps*0.5));
                glVertex2f((float)(px1 + ps*0.5), (float)(py1 - ps*0.5));
                glVertex2f((float)(px1 + ps*0.5), (float)(py1 + ps*0.5));
                glVertex2f((float)(px1 - ps*0.5), (float)(py1 + ps*0.5));
                glEnd();
            }
            glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
            glPopAttrib();
        }
        // Draw sources as hollow yellow circles (selected one highlighted)
        if (g_n_sources > 0 && g_sources) {
            glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
            glDisable(GL_TEXTURE_2D);
            glDisable(GL_LIGHTING);
            glColor3f(1.0f, 1.0f, 0.2f);
            glMatrixMode(GL_PROJECTION);
            glPushMatrix(); glLoadIdentity(); glOrtho(0, win_w, win_h, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix(); glLoadIdentity();
            for (int s = 0; s < g_n_sources; ++s) {
                double px = GX_TO_WINX(g_sources[s].gx, nx_vis, sponge, win_w);
                double py = GY_TO_WINY(g_sources[s].gy, ny_vis, sponge, win_h);
                double rad_px = fmax(4.0, g_sources[s].radius * ((double)win_w / (double)nx_vis));
                if (g_sources[s].selected) glLineWidth(3.0f); else glLineWidth(1.5f);
                glBegin(GL_LINE_LOOP);
                int segs = 24;
                for (int k = 0; k < segs; ++k) {
                    double a = 2.0 * M_PI * (double)k / (double)segs;
                    double vx = px + cos(a) * rad_px;
                    double vy = py + sin(a) * rad_px;
                    glVertex2f((float)vx, (float)vy);
                }
                glEnd();
                // Draw a small filled center dot for selected source
                if (g_sources[s].selected) {
                    const float ds = 4.0f;
                    glBegin(GL_QUADS);
                        glVertex2f((float)(px - ds*0.5), (float)(py - ds*0.5));
                        glVertex2f((float)(px + ds*0.5), (float)(py - ds*0.5));
                        glVertex2f((float)(px + ds*0.5), (float)(py + ds*0.5));
                        glVertex2f((float)(px - ds*0.5), (float)(py + ds*0.5));
                    glEnd();
                }
            }
            glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
            glPopAttrib();
        }
        // If a drag is active, draw the segment being dragged (thin line) using pixel coords
        if (barrier_drag_active && barrier_drag_seg >= 0 && barrier_drag_seg < n_barrier_segs) {
            glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
            glDisable(GL_TEXTURE_2D);
            glColor3f(1.0f, 1.0f, 1.0f);
            glMatrixMode(GL_PROJECTION);
            glPushMatrix(); glLoadIdentity(); glOrtho(0, win_w, win_h, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix(); glLoadIdentity();
            double x0 = (double)(barrier_segs[barrier_drag_seg].x0 - sponge) / (double)nx_vis * win_w;
            double y0 = (double)(barrier_segs[barrier_drag_seg].y0 - sponge) / (double)ny_vis * win_h;
            double x1 = (double)(barrier_segs[barrier_drag_seg].x1 - sponge) / (double)nx_vis * win_w;
            double y1 = (double)(barrier_segs[barrier_drag_seg].y1 - sponge) / (double)ny_vis * win_h;
            glLineWidth(2.0f);
            glBegin(GL_LINES);
            glVertex2f((float)x0, (float)y0); glVertex2f((float)x1, (float)y1);
            glEnd();
            glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
            glPopAttrib();
        }
        // If a pending start point exists (before second click), draw it as small white square
        if (barrier_pending) {
            glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
            glDisable(GL_TEXTURE_2D);
            glColor3f(1.0f,1.0f,1.0f);
            glMatrixMode(GL_PROJECTION);
            glPushMatrix(); glLoadIdentity(); glOrtho(0, win_w, win_h, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix(); glLoadIdentity();
            double px = (double)(barrier_pending_x - sponge) / (double)nx_vis * win_w;
            double py = (double)(barrier_pending_y - sponge) / (double)ny_vis * win_h;
            const float ps = 6.0f;
            glBegin(GL_QUADS);
            glVertex2f((float)(px - ps*0.5), (float)(py - ps*0.5));
            glVertex2f((float)(px + ps*0.5), (float)(py - ps*0.5));
            glVertex2f((float)(px + ps*0.5), (float)(py + ps*0.5));
            glVertex2f((float)(px - ps*0.5), (float)(py + ps*0.5));
            glEnd();
            glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
            glPopAttrib();
        }
        if (menus && menus->base_menu && app.show_base_menu) menu_render(menus->base_menu, win_w, win_h);
        if (menus && menus->mouse_menu && app.show_mouse_controls) menu_render(menus->mouse_menu, win_w, win_h);
        if (menus && menus->sim_menu && app.show_sim_controls) menu_render(menus->sim_menu, win_w, win_h);
        // Render source menu only when in Source mouse mode and a source is selected
        if (menus && menus->source_menu && app.mouse_source && g_selected_source >= 0) menu_render(menus->source_menu, win_w, win_h);
        // Render current mouse mode as text in the corner like CPU sim
        const char *mmode = "None";
        if (app.mouse_add_wave) mmode = "Add Wave";
        else if (app.mouse_add_barrier) mmode = "Barrier";
        else if (app.mouse_source) mmode = "Source";
        // draw at top-right corner with slight padding
        char modebuf[64]; snprintf(modebuf, sizeof(modebuf), "Mouse Mode: %s", mmode);
        int tw = 0, th = 0;
        int xpos = win_w - 10 - (int)strlen(modebuf)*8;
        if (menu_measure_text(modebuf, &tw, &th) == 0) xpos = win_w - 10 - tw;
        /* menu_draw_text_at expects an orthographic pixel projection; set it briefly here */
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, win_w, win_h, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        menu_draw_text_at(modebuf, xpos, 10, (Color){255,255,255,255});
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        // restore GL state
        glDisable(GL_BLEND);
        if (depthEnabled) glEnable(GL_DEPTH_TEST);
            // Draw stats in bottom-left if enabled
        if (render.show_stats) {
            char statsbuf[256];
            // compute render FPS from last_time difference
            uint32_t now_stats = SDL_GetTicks();
            uint32_t dt_ms = now_stats - last_time;
            double render_fps = dt_ms > 0 ? 1000.0 / (double)dt_ms : 0.0;
            double sim_fps = 1.0 / dt; /* fallback */
            if (gpu_time_samples > 0 && gpu_compute_ms_avg > 0.0) sim_fps = 1000.0 / gpu_compute_ms_avg;
            snprintf(statsbuf, sizeof(statsbuf), "render_fps: %.1f\nsim_fps: %.1f\nframe: %llu", render_fps, sim_fps, (unsigned long long)render_frame_counter);
            // ensure orthographic projection as expected by menu_draw_text_at
            glMatrixMode(GL_PROJECTION);
            glPushMatrix(); glLoadIdentity(); glOrtho(0, win_w, win_h, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix(); glLoadIdentity();
            menu_draw_text_at(statsbuf, 8, win_h - 48, (Color){255,255,255,255});
            glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
        }
        SDL_GL_SwapWindow(win);

        // Simple frame timing throttle (cap ~60 FPS)
        uint32_t now = SDL_GetTicks(); uint32_t elapsed = now - last_time;
        if (app.limit_fps) {
            if (elapsed < 16) SDL_Delay(16 - elapsed);
            last_time = SDL_GetTicks();
        } else {
            last_time = now;
        }
        render_frame_counter++;
    }

    // Cleanup
    glDeleteProgram(compute_prog);
    if (p_prog) glDeleteProgram(p_prog);
    if (disp_prog) glDeleteProgram(disp_prog);
    if (prog_divergence) glDeleteProgram(prog_divergence);
    if (prog_project) glDeleteProgram(prog_project);
    if (prog_jacobi) glDeleteProgram(prog_jacobi);
    if (prog_advect_velocity) glDeleteProgram(prog_advect_velocity);
    if (prog_advect_smoke) glDeleteProgram(prog_advect_smoke);
    glDeleteTextures(1, &tex_u_curr); glDeleteTextures(1, &tex_u_prev); glDeleteTextures(1, &tex_out);
    if (tex_paint) glDeleteTextures(1, &tex_paint);
    if (tex_pressure) glDeleteTextures(1, &tex_pressure);
    if (tex_pressure_tmp) glDeleteTextures(1, &tex_pressure_tmp);
    if (tex_divergence) glDeleteTextures(1, &tex_divergence);
    if (tex_velocity) glDeleteTextures(1, &tex_velocity);
    if (tex_velocity_tmp) glDeleteTextures(1, &tex_velocity_tmp);
    if (tex_smoke) glDeleteTextures(1, &tex_smoke);
    if (tex_smoke_tmp) glDeleteTextures(1, &tex_smoke_tmp);
    if (bm) boundary_mask_free(bm);
    free(u_curr_data); free(u_prev_data);
    if (paint_buf) free(paint_buf);
    gpu_program_free(prog);
    expression_release(wave_expr);
    grid_metadata_free(grid);
    SDL_GL_DeleteContext(ctx); SDL_DestroyWindow(win); SDL_Quit();
    return 0;
}
