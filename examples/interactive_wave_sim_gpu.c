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
#include "../include/gpu_sim.h"
#include "../include/grid.h"
#include "../include/expression.h"
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
            size_t off = (size_t)i * ny + src_j;
            double v = field ? field[off] : 0.0;
            size_t idx = ((size_t)j * nx + i) * 4;
            buf[idx+0] = (float)v;
            buf[idx+1] = 0.0f; buf[idx+2] = 0.0f; buf[idx+3] = 0.0f;
        }
    }
    GLuint tex; glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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
            buf[idx + 0] += (float)val;
        }
    }
}

// Simple render / app state used by the menu callbacks
typedef enum { RENDER_HEIGHT, RENDER_VELOCITY, RENDER_RGB } AppRenderMode;
typedef struct {
    AppRenderMode mode;
    double value_scale;
    int show_boundaries;
    int show_stats;
    int mode_height;
    int mode_velocity;
    int mode_rgb;
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
} AppState;

// Source representation (grid coords)
typedef struct {
    int gx, gy;           // grid indices
    double amp;
    double freq;
    double phase;
    double radius;        // in grid units
    int selected;
} Source;
    // Scaling factors to map mouse paint controls -> source defaults (make sources smaller)
#define SOURCE_RADIUS_SCALE 0.5
#define SOURCE_AMP_SCALE 0.5
// Additional scale to map mouse amplitude slider -> actual paint amplitude (make mouse additions smaller)
#define MOUSE_AMPLITUDE_SCALE 1.0
    /* trailing-tab label handled inside create_app_menus where menus are constructed */

// Helper macros to map window coords (mx,my) to grid indices (gix,gjy) using visible region and sponge
#define WIN_TO_GX(mx, win_w, nx_vis, sponge) ((int)floor((double)(mx) / (double)(win_w) * (double)(nx_vis)) + (sponge))
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

// Controls mirrored into the source menu for the currently-selected source
static double sel_src_amp = 0.0;
static double sel_src_freq = 1.0;
static double sel_src_phase = 0.0;
static double sel_src_radius = 6.0; // in grid units
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
    sources_dirty = 1;
    }
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
    r->mode_height = r->mode_velocity = r->mode_rgb = 0;
    if (vi->variable == &r->mode_height) { r->mode_height = 1; r->mode = RENDER_HEIGHT; }
    else if (vi->variable == &r->mode_velocity) { r->mode_velocity = 1; r->mode = RENDER_VELOCITY; }
    else if (vi->variable == &r->mode_rgb) { r->mode_rgb = 1; r->mode = RENDER_RGB; }
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
    int modes[4] = { app->mouse_none, app->mouse_add_wave, app->mouse_add_barrier, app->mouse_source };
    int idx = 0;
    for (int i = 0; i < 4; ++i) if (modes[i]) { idx = i; break; }
    idx += dir; if (idx < 0) idx = 3; if (idx > 3) idx = 0;
    app->mouse_none = app->mouse_add_wave = app->mouse_add_barrier = app->mouse_source = 0;
    if (idx == 0) app->mouse_none = 1;
    else if (idx == 1) app->mouse_add_wave = 1;
    else if (idx == 2) app->mouse_add_barrier = 1;
    else if (idx == 3) app->mouse_source = 1;
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
    menurow_add_interaction(mrow2, variableinteraction_create(&app->mouse_source, "Source", 0, 1, VAR_BOOL, on_mouse_mode_change, app));
    menu_add_row(m->mouse_menu, mrow2);
    MenuRow *mrow3 = menurow_create();
    // increase mouse amplitude range by one order of magnitude
    menurow_add_interaction(mrow3, variableinteraction_create(&app->wave_amplitude, "Amplitude", -0.08, 0.08, VAR_SLIDER, NULL, NULL));
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

    // Sim menu
    m->sim_menu = menu_create(270,10,250,340,1,"Simulation Controls", textColor, bgColor);
    MenuRow *s1 = menurow_create(); menurow_add_interaction(s1, variableinteraction_create(&app->paused, "Paused", 0, 1, VAR_BOOL, NULL, app)); menu_add_row(m->sim_menu, s1);
    MenuRow *s2 = menurow_create(); menurow_add_interaction(s2, variableinteraction_create(&app->dummy_reset, "Reset", 0, 1, VAR_BOOL, cb_reset, reset_cb_data)); menu_add_row(m->sim_menu, s2);
    // Wave speed slider: allow values from 0.01 .. 2.0
    MenuRow *s3 = menurow_create(); menurow_add_interaction(s3, variableinteraction_create(&app->wave_speed, "Wave Speed", 0.01, 4.0, VAR_SLIDER, cb_wave_speed_changed, reset_cb_data)); menu_add_row(m->sim_menu, s3);
    // Timestep dt slider: very small .. double initial value
    MenuRow *s4 = menurow_create(); menurow_add_interaction(s4, variableinteraction_create(&app->dt, "dt (s)", 1e-6, fmax(1e-6, 2.0 * ((reset_cb_data_t*)reset_cb_data)->dt_val), VAR_SLIDER, cb_dt_changed, reset_cb_data)); menu_add_row(m->sim_menu, s4);
    // Number of compute iterations per render (1..10 integer steps)
    MenuRow *s4b = menurow_create(); menurow_add_interaction(s4b, variableinteraction_create(&app->steps_per_frame, "Steps/frame", 1.0, 10.0, VAR_SLIDER, NULL, NULL)); menu_add_row(m->sim_menu, s4b);
    MenuRow *s5 = menurow_create(); menurow_add_interaction(s5, variableinteraction_create(&render->value_scale, "Scale", 0.001, 10.0, VAR_SLIDER, NULL, NULL)); menu_add_row(m->sim_menu, s5);
    MenuRow *s6 = menurow_create(); menurow_add_interaction(s6, variableinteraction_create(&app->dummy_clear_barriers, "Clear Barriers", 0, 1, VAR_BOOL, cb_clear_barriers, NULL)); menu_add_row(m->sim_menu, s6);
    MenuRow *s7 = menurow_create(); menurow_add_interaction(s7, variableinteraction_create(&app->dummy_clear_sources, "Clear Sources", 0, 1, VAR_BOOL, cb_clear_sources, NULL)); menu_add_row(m->sim_menu, s7);
    MenuRow *s8 = menurow_create(); menurow_add_interaction(s8, variableinteraction_create(&render->show_boundaries, "Show Boundaries", 0, 1, VAR_BOOL, NULL, NULL)); menu_add_row(m->sim_menu, s8);
    MenuRow *s9 = menurow_create(); menurow_add_interaction(s9, variableinteraction_create(&render->show_stats, "Show Stats", 0, 1, VAR_BOOL, NULL, NULL)); menu_add_row(m->sim_menu, s9);
    MenuRow *s9b = menurow_create(); menurow_add_interaction(s9b, variableinteraction_create(&app->limit_fps, "Limit FPS to ~60", 0, 1, VAR_BOOL, NULL, NULL)); menu_add_row(m->sim_menu, s9b);
    // render mode radio buttons
    MenuRow *s10 = menurow_create(); menurow_add_interaction(s10, variableinteraction_create(&render->mode_height, "Mode: Height", 0, 1, VAR_BOOL, on_render_mode_change, render)); menu_add_row(m->sim_menu, s10);
    MenuRow *s11 = menurow_create(); menurow_add_interaction(s11, variableinteraction_create(&render->mode_velocity, "Mode: Velocity", 0, 1, VAR_BOOL, on_render_mode_change, render)); menu_add_row(m->sim_menu, s11);
    MenuRow *s12 = menurow_create(); menurow_add_interaction(s12, variableinteraction_create(&render->mode_rgb, "Mode: RGB", 0, 1, VAR_BOOL, on_render_mode_change, render)); menu_add_row(m->sim_menu, s12);

    // source_menu left empty for now
    m->source_menu = menu_create(530,10,250,220,1,"Source Controls", textColor, bgColor);
    // increase source amplitude control range by one order of magnitude
    MenuRow *src1 = menurow_create(); menurow_add_interaction(src1, variableinteraction_create(&sel_src_amp, "Amp", -0.1, 0.1, VAR_SLIDER, cb_source_control_changed, NULL)); menu_add_row(m->source_menu, src1);
    MenuRow *src2 = menurow_create(); menurow_add_interaction(src2, variableinteraction_create(&sel_src_freq, "Freq (Hz)", 0.0, 300.0, VAR_SLIDER, cb_source_control_changed, NULL)); menu_add_row(m->source_menu, src2);
    MenuRow *src3 = menurow_create(); menurow_add_interaction(src3, variableinteraction_create(&sel_src_phase, "Phase (rad)", 0.0, 6.283, VAR_SLIDER, cb_source_control_changed, NULL)); menu_add_row(m->source_menu, src3);
    MenuRow *src4 = menurow_create(); menurow_add_interaction(src4, variableinteraction_create(&sel_src_radius, "Radius (grid)", 1.0, 50.0, VAR_SLIDER, cb_source_control_changed, NULL)); menu_add_row(m->source_menu, src4);
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
    /* The new prog_new's kernel will be compiled lazily on the next
       gpu_run_program_to_tex call.  Clear the old compute_prog handle
       if one was set (it's NULL after the refactor).                    */
    if (d->compute_prog_ptr && *(d->compute_prog_ptr)) { glDeleteProgram(*(d->compute_prog_ptr)); *(d->compute_prog_ptr) = 0; }
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
    (void)user_data; if (vi && vi->variable) *(int*)vi->variable = 0; fprintf(stderr, "Clear sources pressed (no-op)\n");
}
static void cb_clear_barriers(VariableInteraction *vi, void *user_data) {
    (void)user_data; if (vi && vi->variable) *(int*)vi->variable = 0; fprintf(stderr, "Clear barriers pressed (no-op)\n");
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

/* -----------------------------------------------------------------------
 * WaveSimGPU — holds all GPU objects created during initialisation so
 * they can be passed around as a unit and freed together.
 * ----------------------------------------------------------------------- */
typedef struct {
    GPUContext      *gpu_ctx;
    GPUTexDict      *compute_tex_dict;
    GPUProgram      *paint_gpu_prog;
    GPUTexDict      *paint_tex_dict;
    GPUProgram      *damping_gpu_prog;
    GPUTexDict      *damping_tex_dict;
    GLuint           damping_tex;
    GLuint           overlay_prog;
    GLint            loc_ov_mask, loc_ov_dims, loc_ov_vis_off, loc_ov_vis_size;
    GPUProgram      *disp_prog_gpu;
    GLuint           disp_prog;
    GPURenderConfig *disp_rc;
    GLuint           srcgen_prog;
    GLint            loc_src_n, loc_src_time, loc_src_dims, loc_src_spacing;
    GLint            loc_src_desc, loc_max_src;
    GPUProgram      *compute_prog;   /* wave update program */
    Expression      *wave_expr;      /* current wave expression (kept for cb_wave_speed_changed) */
} WaveSimGPU;

/* Initialise all GPU programs/texdicts/rendercfg that live for the duration
 * of the simulation.  Must be called after the OpenGL context is current.
 * Returns 0 on success, -1 on failure (partial state may have been created;
 * caller should call wave_sim_gpu_free even on failure).
 *
 * render must already be initialised before this call because disp_rc
 * stores pointers into its fields for per-frame uniform updates.
 */
static int wave_sim_gpu_init(WaveSimGPU *g,
                              SDL_Window *win, SDL_GLContext ctx,
                              uint32_t nx, uint32_t ny,
                              int nx_vis, int ny_vis,
                              int sponge, int damp_gap, int damp_width,
                              double dt, double *spacing,
                              GridMetadata *grid,
                              AppRenderState *render) {
    memset(g, 0, sizeof(*g));
    (void)spacing; /* reserved for future use (e.g. source gen uniforms) */

    /* Shared vertex shader used by overlay and source-generator programs */
    static const char *vs_src =
        "void main() { gl_Position = gl_Vertex; gl_TexCoord[0] = gl_MultiTexCoord0; }";

    /* --- Adopt the SDL/GL context into gpu_ctx ----------------------------- */
    g->gpu_ctx = gpu_context_adopt(win, ctx, (int)nx, (int)ny);
    if (!g->gpu_ctx) { fprintf(stderr, "gpu_context_adopt failed\n"); return -1; }

    /* --- Compute tex-dict -------------------------------------------------- */
    g->compute_tex_dict = gpu_tex_dict_create(8);

    /* --- Paint composite (expression-compiled) ----------------------------- */
    Expression *p_src_expr = expr_add(
        expr_add(expr_variable("src"), expr_variable("paint_gpu")),
        expr_variable("paint_cpu"));
    g->paint_gpu_prog = gpu_compile_optimized(p_src_expr, grid, GPU_BACKEND_OPENGL);
    expression_release(p_src_expr);
    g->paint_tex_dict = gpu_tex_dict_create(8);
    if (g->paint_gpu_prog &&
        gpu_kernel_ensure_program(g->paint_gpu_prog->kernels[0]) == 0) {
        /* GL program owned by kernel; paint_gpu_prog kept alive */
    } else {
        fprintf(stderr, "paint composite program compile failed\n");
        if (g->paint_gpu_prog) { gpu_program_free(g->paint_gpu_prog); g->paint_gpu_prog = NULL; }
    }

    /* --- Damping texture + expression-compiled damping pass ---------------- */
    g->damping_tex_dict = gpu_tex_dict_create(8);
    float *damp_buf = calloc((size_t)nx * ny * 4, sizeof(float));
    if (!damp_buf) { fprintf(stderr, "Failed to allocate damping buffer\n"); return -1; }

    int v_x0 = sponge, v_x1 = sponge + nx_vis - 1;
    int v_y0 = sponge, v_y1 = sponge + ny_vis - 1;
    for (int j = 0; j < (int)ny; ++j) {
        for (int i = 0; i < (int)nx; ++i) {
            int cx = i, cy = j;
            int dx = 0, dy = 0;
            if (cx < v_x0) dx = v_x0 - cx; else if (cx > v_x1) dx = cx - v_x1;
            if (cy < v_y0) dy = v_y0 - cy; else if (cy > v_y1) dy = cy - v_y1;
            int dist = dx > dy ? dx : dy;
            float sigma = 0.0f;
            if (dist > damp_gap) {
                float effective_dist  = (float)(dist - damp_gap);
                float effective_width = (float)damp_width;
                if (effective_width < 1.0f) effective_width = 1.0f;
                float t = effective_dist / effective_width;
                if (t < 0.0f) t = 0.0f;
                if (t > 1.0f) t = 1.0f;
                const float DAMPING_SIGMA_MAX = 30.0f;
                sigma = (float)(DAMPING_SIGMA_MAX * (t * t));
            }
            size_t idx = ((size_t)j * nx + i) * 4;
            damp_buf[idx+0] = sigma;
            damp_buf[idx+1] = damp_buf[idx+2] = 0.0f;
            damp_buf[idx+3] = 1.0f;
        }
    }

    GLuint tex_damping;
    glGenTextures(1, &tex_damping);
    glBindTexture(GL_TEXTURE_2D, tex_damping);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)nx, (GLsizei)ny,
                    0, GL_RGBA, GL_FLOAT, damp_buf);
    free(damp_buf);
    g->damping_tex = tex_damping;

    /* Build damping expression:
        u_new = (2 - sigma*dt)*u_curr - (1 - sigma*dt)*u_prev
                + (u_next - 2*u_curr + u_prev)                    */
    Expression *dn_next  = expr_variable("u_next");
    Expression *dn_curr  = expr_variable("u_curr");
    Expression *dn_prev  = expr_variable("u_prev");
    Expression *dn_sigma = expr_variable("sigma");
    Expression *dn_dt    = expr_literal(literal_create_scalar(dt));
    Expression *dn_two   = expr_literal(literal_create_scalar(2.0));
    Expression *dn_one   = expr_literal(literal_create_scalar(1.0));
    Expression *sdt      = expr_multiply(dn_sigma, dn_dt);
    Expression *two_curr = expr_multiply(dn_two, dn_curr);
    Expression *accel    = expr_add(expr_add(dn_next, expr_negate(two_curr)), dn_prev);
    Expression *coeff_curr = expr_add(dn_two, expr_negate(sdt));
    Expression *coeff_prev = expr_add(sdt, expr_negate(dn_one));
    Expression *damp_expr  = expr_add(
        expr_add(expr_multiply(coeff_curr, dn_curr),
                    expr_multiply(coeff_prev, dn_prev)),
        accel);
    GPUProgram *damp_gpu = gpu_compile_optimized(damp_expr, grid, GPU_BACKEND_OPENGL);
    expression_release(damp_expr);
    if (damp_gpu && gpu_kernel_ensure_program(damp_gpu->kernels[0]) == 0) {
        g->damping_gpu_prog = damp_gpu;
    } else {
        fprintf(stderr, "Damping expression compile failed\n");
        if (damp_gpu) gpu_program_free(damp_gpu);
    }

    /* --- Overlay shader (raw GLSL — visualisation only) -------------------- */
    const char *overlay_fs =
        "#version 120\n"
        "uniform sampler2D mask_tex; uniform ivec2 dims; "
        "uniform ivec2 vis_offset; uniform ivec2 vis_size;"
        "void main() { vec2 uv = gl_TexCoord[0].st; "
        "vec2 tex_idx = vec2(vis_offset) + uv * vec2(vis_size); "
        "vec2 c = (floor(tex_idx) + vec2(0.5)) / vec2(dims); "
        "float m = texture2D(mask_tex, c).r; "
        "if (m > 0.5) gl_FragColor = vec4(1.0,1.0,0.0,1.0); "
        "else gl_FragColor = vec4(0.0,0.0,0.0,0.0); }";
    GLuint ov_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint ov_fs = compile_shader(GL_FRAGMENT_SHADER, overlay_fs);
    if (ov_vs && ov_fs) g->overlay_prog = link_program(ov_vs, ov_fs);
    if (ov_vs) glDeleteShader(ov_vs);
    if (ov_fs) glDeleteShader(ov_fs);

    if (g->overlay_prog) {
        g->loc_ov_mask     = glGetUniformLocation(g->overlay_prog, "mask_tex");
        g->loc_ov_dims     = glGetUniformLocation(g->overlay_prog, "dims");
        g->loc_ov_vis_off  = glGetUniformLocation(g->overlay_prog, "vis_offset");
        g->loc_ov_vis_size = glGetUniformLocation(g->overlay_prog, "vis_size");
        glUseProgram(g->overlay_prog);
        if (g->loc_ov_mask >= 0) glUniform1i(g->loc_ov_mask, 2);
        glUseProgram(0);
    }

    /* --- Display render modes (expression-compiled) ------------------------ */
    RenderModeDef modes[3];
    memset(modes, 0, sizeof(modes));

    /* Mode 0: Height — pos/neg channels with gamma compression */
    Expression *src_var  = expr_variable("src");
    Expression *zero_lit = expr_literal(literal_create_scalar(0.0));
    Expression *gamma_lit = expr_literal(literal_create_scalar(0.5));
    Expression *pos = expr_binary(OP_MAX, src_var, zero_lit);
    Expression *neg = expr_binary(OP_MAX, expr_unary(OP_NEGATE, expr_variable("src")), zero_lit);
    modes[0].chan_expr[0] = expr_power(pos, gamma_lit);
    modes[0].chan_expr[2] = expr_power(neg, gamma_lit);
    modes[0].chan_scale[0] = 0.5; modes[0].chan_scale[1] = 1.0; modes[0].chan_scale[2] = 0.5;

    /* Mode 1: Velocity magnitude — sqrt(dx^2 + dy^2) */
    Expression *dx = expr_derivative(expr_variable("src"), "x");
    Expression *dy = expr_derivative(expr_variable("src"), "y");
    Expression *sumsq = expr_binary(OP_ADD,
        expr_binary(OP_MULTIPLY, dx, dx),
        expr_binary(OP_MULTIPLY, dy, dy));
    Expression *half = expr_literal(literal_create_scalar(0.5));
    Expression *sumsq_sqrt = expr_power(sumsq, half);
    modes[1].chan_expr[0] = sumsq_sqrt;
    modes[1].chan_expr[1] = sumsq_sqrt;
    modes[1].chan_expr[2] = sumsq_sqrt;
    modes[1].chan_scale[0] = 0.5; modes[1].chan_scale[1] = 0.5; modes[1].chan_scale[2] = 0.5;

    /* Mode 2: RGB — dx, dy, src */
    modes[2].chan_expr[0] = expr_derivative(expr_variable("src"), "x");
    modes[2].chan_expr[1] = expr_derivative(expr_variable("src"), "y");
    modes[2].chan_expr[2] = expr_variable("src");
    modes[2].chan_scale[0] = 0.5; modes[2].chan_scale[1] = 0.5; modes[2].chan_scale[2] = 0.5;

    g->disp_prog_gpu = gpu_compile_render_modes(modes, 3, grid, GPU_BACKEND_OPENGL);

    /* Release temporary mode expressions */
    expression_release(modes[0].chan_expr[0]); 
    expression_release(modes[0].chan_expr[2]);
    expression_release(modes[1].chan_expr[0]); /* shared across channels — single release */
    expression_release(modes[2].chan_expr[0]); 
    expression_release(modes[2].chan_expr[1]);
    expression_release(modes[2].chan_expr[2]);

    if (g->disp_prog_gpu && g->disp_prog_gpu->n_kernels > 0 &&
        g->disp_prog_gpu->kernels[0] &&
        gpu_kernel_ensure_program(g->disp_prog_gpu->kernels[0]) == 0) {
        g->disp_prog = (GLuint)g->disp_prog_gpu->kernels[0]->gl_program;
    }
    if (!g->disp_prog) {
        fprintf(stderr, "Display program compile failed\n");
        return -1;
    }

    /* --- GPURenderConfig for display pass ---------------------------------- */
    g->disp_rc = gpu_render_config_create(g->disp_prog_gpu, g->disp_prog);
    gpu_render_config_bind_uniform(g->disp_rc, "render_mode",     GPU_UNIFORM_INT,    &render->mode);
    gpu_render_config_bind_uniform(g->disp_rc, "value_scale",     GPU_UNIFORM_DOUBLE, &render->value_scale);
    gpu_render_config_bind_uniform(g->disp_rc, "show_boundaries", GPU_UNIFORM_BOOL,   &render->show_boundaries);
    g->disp_rc->vis_offset_x = sponge;  g->disp_rc->vis_offset_y = sponge;
    g->disp_rc->vis_size_x   = nx_vis;  g->disp_rc->vis_size_y   = ny_vis;

    /* --- Source generator shader (raw GLSL — dynamic loop) ----------------- */
    const char *srcgen_fs =
        "#version 120\n"
        "uniform sampler2D src_desc_tex;\n"
        "uniform int n_sources;\n"
        "uniform int max_src;\n"
        "uniform float sim_time; uniform vec2 spacing; uniform ivec2 dims;\n"
        "void main() { vec2 uv = gl_TexCoord[0].st; vec2 idx = floor(uv * vec2(dims)); float val = 0.0;\n"
        " for (int i = 0; i < n_sources; ++i) { float fu = (0.5 + float(i)) / float(max_src);"
        " vec4 a = texture2D(src_desc_tex, vec2(fu, 0.25)); vec4 b = texture2D(src_desc_tex, vec2(fu, 0.75));"
        " float gx = a.r; float gy = a.g; float amp = a.b; float freq = a.a;"
        " float phase = b.r; float radius = b.g;"
        " float dx = (idx.x - gx) * spacing.x; float dy = (idx.y - gy) * spacing.y;"
        " float r2 = dx*dx + dy*dy; float rr = radius * radius * spacing.x * spacing.x;"
        " if (r2 <= rr) { val = amp * sin(freq * sim_time + phase); } }\n"
        " gl_FragColor = vec4(val, 0.0, 0.0, 0.0); }";
    GLuint s_vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint s_fs = compile_shader(GL_FRAGMENT_SHADER, srcgen_fs);
    if (s_vs && s_fs) g->srcgen_prog = link_program(s_vs, s_fs);
    if (s_vs) glDeleteShader(s_vs);
    if (s_fs) glDeleteShader(s_fs);

    if (g->srcgen_prog) {
        glUseProgram(g->srcgen_prog);
        g->loc_src_n       = glGetUniformLocation(g->srcgen_prog, "n_sources");
        g->loc_src_time    = glGetUniformLocation(g->srcgen_prog, "sim_time");
        g->loc_src_dims    = glGetUniformLocation(g->srcgen_prog, "dims");
        g->loc_src_spacing = glGetUniformLocation(g->srcgen_prog, "spacing");
        g->loc_src_desc    = glGetUniformLocation(g->srcgen_prog, "src_desc_tex");
        g->loc_max_src     = glGetUniformLocation(g->srcgen_prog, "max_src");
        if (g->loc_src_desc >= 0) glUniform1i(g->loc_src_desc, 4);
        if (g->loc_max_src  >= 0) glUniform1i(g->loc_max_src, 64);
        glUseProgram(0);
    }

    /* --- Source-descriptor texture (file-scope global tex_src_desc) --------- */
    if (g->srcgen_prog) {
        glGenTextures(1, &tex_src_desc);
        glBindTexture(GL_TEXTURE_2D, tex_src_desc);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        int max_src_init = 64;
        float *zero_buf = calloc((size_t)max_src_init * 2 * 4, sizeof(float));
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, max_src_init, 2, 0, GL_RGBA, GL_FLOAT, zero_buf);
        free(zero_buf);
    }

    /* --- Wave compute expression + program --------------------------------- */
    Expression *u_curr = expr_variable("u_curr");
    Expression *u_prev = expr_variable("u_prev");
    Expression *lap    = expr_laplacian(expr_variable("u_curr"));
    double c = 1.0;
    double c2dt2 = c * c * dt * dt;
    Expression *c2dt2_lit = expr_literal(literal_create_scalar(c2dt2));
    Expression *wave_accel = expr_multiply(c2dt2_lit, lap);
    Expression *two       = expr_literal(literal_create_scalar(2.0));
    Expression *two_u     = expr_multiply(two, u_curr);
    Expression *neg_prev  = expr_negate(u_prev);
    Expression *diff      = expr_add(two_u, neg_prev);
    g->wave_expr    = expr_add(diff, wave_accel);
    g->compute_prog = gpu_compile_optimized(g->wave_expr, grid, GPU_BACKEND_OPENGL);
    if (!g->compute_prog) {
        fprintf(stderr, "wave compute gpu_compile_optimized failed\n");
        return -1;
    }

    return 0;
}

/* Free all GPU objects owned by a WaveSimGPU.  Safe to call even if init
 * failed partway through (checks each pointer before freeing).           */
static void wave_sim_gpu_free(WaveSimGPU *g) {
    if (!g) return;
    if (g->disp_rc)          gpu_render_config_free(g->disp_rc);
    if (g->compute_tex_dict) gpu_tex_dict_free(g->compute_tex_dict);
    if (g->damping_tex_dict) gpu_tex_dict_free(g->damping_tex_dict);
    if (g->paint_tex_dict)   gpu_tex_dict_free(g->paint_tex_dict);
    if (g->damping_gpu_prog) gpu_program_free(g->damping_gpu_prog);
    if (g->paint_gpu_prog)   gpu_program_free(g->paint_gpu_prog);
    if (g->disp_prog_gpu)    gpu_program_free(g->disp_prog_gpu); /* owns disp_prog GL object */
    if (g->overlay_prog)     glDeleteProgram(g->overlay_prog);
    if (g->srcgen_prog)      glDeleteProgram(g->srcgen_prog);
    if (g->damping_tex)      glDeleteTextures(1, &g->damping_tex);
    if (g->compute_prog) gpu_program_free(g->compute_prog);
    if (g->wave_expr)    expression_release(g->wave_expr);
    if (g->gpu_ctx)          gpu_context_free(g->gpu_ctx);
    memset(g, 0, sizeof(*g));
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    // Visible domain parameters (match interactive_wave_sim defaults roughly)
    double Lx_vis = 2.0, Ly_vis = 2.0;
    /* scale the visible resolution by 1.5 while keeping physical size */
    int nx_vis = 384, ny_vis = 384; /* 256 * 1.5 = 384 */

    // Sponge/damping configuration:
    // - damp_width: number of off-screen cells that contain the damping ramp
    // - damp_gap: buffer of off-screen cells immediately adjacent to the visible
    //   region where damping is intentionally ZERO (provides an inner open buffer)
    // - sponge: total offset (damp_gap + damp_width) reserved on each side
    int avg_vis = (nx_vis + ny_vis) / 2;
    int damp_width = avg_vis;               /* keep damping region large */
    int damp_gap = avg_vis / 4;            /* buffer between visible edge and damping start */
    int sponge = damp_gap + damp_width; if (sponge < 10) sponge = 10;

    // Total grid includes sponge rim so simulation domain is larger than visible
    uint32_t nx = (uint32_t)(nx_vis + 2 * sponge);
    uint32_t ny = (uint32_t)(ny_vis + 2 * sponge);
    uint32_t dims[3] = { nx, ny, 1 };
    double spacing[3] = { Lx_vis / (nx_vis - 1), Ly_vis / (ny_vis - 1), 1.0 };
    double origin[3] = { 0.0, 0.0, 0.0 };
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);

    // Set edge boundaries open (no special Dirichlet masking at edges)
    grid_set_boundary(grid, 0, 0, BC_OPEN, 0.0);
    grid_set_boundary(grid, 0, 1, BC_OPEN, 0.0);
    grid_set_boundary(grid, 1, 0, BC_OPEN, 0.0);
    grid_set_boundary(grid, 1, 1, BC_OPEN, 0.0);

    double dt = 0.002; /* initial timestep; updated from app.dt each frame */

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

    // Prepare initial CPU fields (u_curr zeroed; u_prev zeros)
    double *u_curr_data = calloc((size_t)nx * ny, sizeof(double));
    double *u_prev_data = calloc((size_t)nx * ny, sizeof(double));

    // Create an empty BoundaryMask; barrier segments will be user-managed
    BoundaryMask *bm = boundary_mask_create(grid);
    // barrier segments array (dynamic, starts empty)
    BarrierSeg *barrier_segs = NULL; int n_barrier_segs = 0;
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
    int barrier_pending_x = -1, barrier_pending_y = -1;

    // Create GPU textures from fields (total grid size includes sponge)
    GLuint tex_u_curr = create_texture_from_field(u_curr_data, nx, ny);
    GLuint tex_u_prev = create_texture_from_field(u_prev_data, nx, ny);
    GLuint tex_out = create_empty_texture(nx, ny);
    // temporary texture used by damping pass
    GLuint tex_tmp = create_empty_texture(nx, ny);
    // Paint textures and CPU paint buffer (RGBA32F)
    paint_buf = calloc((size_t)nx * ny * 4, sizeof(float));
    tex_paint = create_empty_texture(nx, ny);     /* GPU-generated paint */
    tex_paint_cpu = create_empty_texture(nx, ny); /* CPU-uploaded paint buffer */
    paint_pending = 0;
    paint_from_gpu = 0;
    paint_buf_dirty = 0;
    int painting_active = 0;
    uint64_t sim_step_counter = 0; /* counts physics steps (used for source phase) */
    uint64_t render_frame_counter = 0; /* counts rendered frames */

    int win_w = 800, win_h = 800;

    // Upload the interior barrier mask textures (needs GL context)
    boundary_mask_upload(bm, NULL);

    // Initialise app/render state before GPU init (disp_rc binds pointers into render)
    AppState app = {0};
    app.paused = 0; app.wave_speed = 1.0; app.max_sim_speed = 1.0; app.dt = 0.002;
    app.steps_per_frame = 1.0; app.wave_amplitude = 0.002; app.wave_spread = 0.05;
    app.default_source_frequency = 5.0; app.default_source_phase = 0.0;
    app.mouse_none = 1; app.show_base_menu = 1; app.limit_fps = 1;
    AppRenderState render = {0};
    render.mode = RENDER_HEIGHT; render.value_scale = 1.0; render.show_boundaries = 1;
    render.show_stats = 1; render.mode_height = 1;

    /* Initialise all GPU programs, tex-dicts and render config. */
    WaveSimGPU g;
    if (wave_sim_gpu_init(&g, win, ctx, nx, ny, nx_vis, ny_vis,
                          sponge, damp_gap, damp_width, dt,
                          spacing, grid, &render) != 0) {
        fprintf(stderr, "wave_sim_gpu_init failed\n");
        return 1;
    }

    /* Convenience aliases so the rest of main can keep its existing names. */
    GPUProgram     **prog_ref         = &g.compute_prog; /* indirection so cb updates are visible */
    GPUContext      *gpu_ctx          = g.gpu_ctx;
    GPUTexDict      *compute_tex_dict = g.compute_tex_dict;
    GPUProgram      *paint_gpu_prog   = g.paint_gpu_prog;
    GPUTexDict      *paint_tex_dict   = g.paint_tex_dict;
    GPUProgram      *damping_gpu_prog = g.damping_gpu_prog;
    GPUTexDict      *damping_tex_dict = g.damping_tex_dict;
    GLuint           damping_tex      = g.damping_tex;
    GLuint           overlay_prog     = g.overlay_prog;
    GLint            loc_ov_mask      = g.loc_ov_mask;
    GLint            loc_ov_dims      = g.loc_ov_dims;
    GLint            loc_ov_vis_off   = g.loc_ov_vis_off;
    GLint            loc_ov_vis_size  = g.loc_ov_vis_size;
    GPURenderConfig *disp_rc          = g.disp_rc;
    GLuint           srcgen_prog      = g.srcgen_prog;
    GLint            loc_src_n        = g.loc_src_n;
    GLint            loc_src_time     = g.loc_src_time;
    GLint            loc_src_dims     = g.loc_src_dims;
    GLint            loc_src_spacing  = g.loc_src_spacing;
    GLint            loc_src_desc     = g.loc_src_desc;
    GLint            loc_max_src      = g.loc_max_src;

    /* Retrieve the shared FBO from gpu_ctx. */
    GLuint fbo = gpu_context_get_fbo(gpu_ctx);
    if (!fbo) { fprintf(stderr, "gpu_context_get_fbo returned 0\n"); return 1; }

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
    reset_data->gpu_prog_ptr = &g.compute_prog;
    reset_data->compute_prog_ptr = NULL;  /* compute shader managed by gpu_ctx kernel cache */
    reset_data->wave_expr_ptr = &g.wave_expr;
    reset_data->grid_ptr = grid;
    reset_data->dt_val = dt;
    reset_data->app_wave_speed_ptr = &app.wave_speed;
    reset_data->sim_counter_ptr = &sim_step_counter;

     /* Initialize source-menu radius default to match mouse paint spread:
         use 3*sigma (same radius used by paint_gaussian_to_rgba) converted to grid units */
    sel_src_radius = fmax(1.0, (app.wave_spread * 3.0 / spacing[0]) * SOURCE_RADIUS_SCALE);

     // Try to set font for menus (best-effort)
    if (menu_set_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14) != 0) {
        fprintf(stderr, "menu_set_font failed; menu text may be invisible\n");
    }
    AppMenus *menus = create_app_menus(&app, &render, reset_data);

    // Set GL state (window size)
    glViewport(0,0,win_w,win_h);
    int quit = 0; SDL_Event ev;
    uint32_t last_time = SDL_GetTicks();

    // One-time initial display — use disp_rc (caches locations on first call)
    gpu_render_config_set_tex(disp_rc, "src",  tex_u_curr);
    gpu_render_config_set_tex(disp_rc, "mask", bm && bm->mask_tex ? bm->mask_tex : 0);
    glClearColor(0.1f,0.1f,0.12f,1.0f); glClear(GL_COLOR_BUFFER_BIT);
    gpu_render_config_draw(disp_rc);
    glUseProgram(0);


    // save states
    GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
    GLboolean blendEnabled = glIsEnabled(GL_BLEND);
    if (depthEnabled) glDisable(GL_DEPTH_TEST);
    // draw overlay opaque (disable blending so mask is clearly visible)
    if (blendEnabled) glDisable(GL_BLEND);

    // draw overlay lines for mask (one-time initial frame; uses cached locations)
    if (overlay_prog && bm && bm->mask_tex) {
        glUseProgram(overlay_prog);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, bm->mask_tex);
        if (loc_ov_mask >= 0) glUniform1i(loc_ov_mask, 2);
        if (loc_ov_dims >= 0) glUniform2i(loc_ov_dims, (GLint)nx, (GLint)ny);
        draw_fullscreen_quad();
    }
    glUseProgram(0);
    if (blendEnabled) glEnable(GL_BLEND);
    if (depthEnabled) glEnable(GL_DEPTH_TEST);
    SDL_GL_SwapWindow(win);

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
                gpu_render_config_set_tex(disp_rc, "src",  tex_u_curr);
                gpu_render_config_set_tex(disp_rc, "mask", bm && bm->mask_tex ? bm->mask_tex : 0);
                glClearColor(0.1f,0.1f,0.12f,1.0f); glClear(GL_COLOR_BUFFER_BIT);
                gpu_render_config_draw(disp_rc);
                glUseProgram(0);
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
                if (ev.button.button == SDL_BUTTON_LEFT) {
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
                        paint_gaussian_to_rgba(paint_buf, nx, ny, gi, gj, app.wave_amplitude * MOUSE_AMPLITUDE_SCALE, app.wave_spread, spacing[0], spacing[1]);
                        paint_pending = 1;
                    } else if (state == SDL_PRESSED && app.mouse_add_barrier) {
                        /* Barrier point editing: left-click/drag moves points, clicking empty space
                           adds points (pair start/end); right-click deletes the nearest point's pair.
                           Prefer most-recently added closest point when resolving ties. No Y-flip here. */
                        int mx = ev.button.x, my = ev.button.y;
                        double fx = (double)mx / (double)win_w; double fy = (double)my / (double)win_h; // no flip for barrier mode
                        int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                        int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);

                        // Search for closest endpoint among all segments (consider both endpoints).
                        int best_seg = -1, best_pt = -1; double best_d2 = 1e300;
                        const double pick_radius = 12.0; // pixels in window space threshold (increased)
                        for (int s = n_barrier_segs - 1; s >= 0; --s) { // iterate newest-first to prefer recent
                            // endpoint 0
                            double px0 = (double)(barrier_segs[s].x0 - sponge) / (double)nx_vis * win_w;
                            double py0 = (double)(barrier_segs[s].y0 - sponge) / (double)ny_vis * win_h;
                            double dx0 = (double)mx - px0; double dy0 = (double)my - py0; double d20 = dx0*dx0 + dy0*dy0;
                            if (d20 < best_d2) { best_d2 = d20; best_seg = s; best_pt = 0; }
                            // endpoint 1
                            double px1 = (double)(barrier_segs[s].x1 - sponge) / (double)nx_vis * win_w;
                            double py1 = (double)(barrier_segs[s].y1 - sponge) / (double)ny_vis * win_h;
                            double dx1 = (double)mx - px1; double dy1 = (double)my - py1; double d21 = dx1*dx1 + dy1*dy1;
                            if (d21 < best_d2) { best_d2 = d21; best_seg = s; best_pt = 1; }
                        }

                        if (ev.button.button == SDL_BUTTON_LEFT) {
                            // If closest endpoint within pick radius, start drag of that endpoint
                            if (best_seg >= 0 && best_d2 <= pick_radius * pick_radius) {
                                // Press near an endpoint: record candidate. Actual drag starts when mouse moves enough.
                                barrier_drag_candidate = 1;
                                barrier_drag_cseg = best_seg;
                                barrier_drag_cpt = best_pt;
                                barrier_drag_press_mx = mx; barrier_drag_press_my = my;
                                barrier_drag_active = 0; barrier_drag_seg = -1; barrier_drag_pt = -1;
                            } else {
                                // Not clicking an existing endpoint: enter pending state.
                                // We capture the actual coordinates on mouse RELEASE so placement follows where the mouse is released.
                                if (!barrier_pending) { barrier_pending = 1; barrier_pending_x = -1; barrier_pending_y = -1; }
                                // else: second click release will create the segment (handled on SDL_RELEASED)
                            }
                        } else if (ev.button.button == SDL_BUTTON_RIGHT) {
                            // Right-click deletes the nearest endpoint's pair (prefer recent)
                            if (best_seg >= 0 && best_d2 <= pick_radius * pick_radius) {
                                int seg_to_remove = best_seg;
                                // remove segment
                                for (int k = seg_to_remove; k+1 < n_barrier_segs; ++k) barrier_segs[k] = barrier_segs[k+1];
                                n_barrier_segs--;
                                barrier_segs = n_barrier_segs ? realloc(barrier_segs, (size_t)n_barrier_segs * sizeof(BarrierSeg)) : NULL;
                                rebuild_barrier_mask(bm, barrier_segs, n_barrier_segs);
                                boundary_mask_upload(bm, NULL);
                            }
                        }
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
                                // deselect others
                                for (int k = 0; k < g_n_sources; ++k) g_sources[k].selected = 0;
                                g_sources[idx].selected = 1;
                                g_n_sources++;
                                g_selected_source = idx;
                                sel_src_amp = g_sources[idx].amp; sel_src_freq = g_sources[idx].freq; sel_src_phase = g_sources[idx].phase; sel_src_radius = g_sources[idx].radius;
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
                                double fx_rel = (double)mx_rel / (double)win_w; double fy_rel = (double)my_rel / (double)win_h; // no flip
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
                    paint_gaussian_to_rgba(paint_buf, nx, ny, gi, gj, app.wave_amplitude * MOUSE_AMPLITUDE_SCALE, app.wave_spread, spacing[0], spacing[1]);
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
                        double fx = (double)mx / (double)win_w; double fy = (double)my / (double)win_h; // no flip
                        int gix = (int)floor(fx * (double)nx_vis) + sponge; if (gix < (int)sponge) gix = sponge; if (gix >= (int)(sponge + nx_vis)) gix = (int)(sponge + nx_vis - 1);
                        int gjy = (int)floor(fy * (double)ny_vis) + sponge; if (gjy < (int)sponge) gjy = sponge; if (gjy >= (int)(sponge + ny_vis)) gjy = (int)(sponge + ny_vis - 1);
                        if (barrier_drag_seg >= 0 && barrier_drag_seg < n_barrier_segs) {
                            if (barrier_drag_pt == 0) { barrier_segs[barrier_drag_seg].x0 = gix; barrier_segs[barrier_drag_seg].y0 = gjy; }
                            else { barrier_segs[barrier_drag_seg].x1 = gix; barrier_segs[barrier_drag_seg].y1 = gjy; }
                        }
                    }
                }
                else if (app.mouse_source) {
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
                            src_desc_buf[(1 * max_src + i) * 4 + 2] = 0.0f;
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
                        glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
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
                    // composite: src + paint_gpu + paint_cpu → tex_out via expression program
                    if (paint_gpu_prog) {
                        gpu_tex_dict_set(paint_tex_dict, "src",       tex_u_curr);
                        gpu_tex_dict_set(paint_tex_dict, "paint_gpu", tex_paint);
                        gpu_tex_dict_set(paint_tex_dict, "paint_cpu", tex_paint_cpu);
                        if (gpu_run_program_to_tex(paint_gpu_prog, paint_tex_dict, tex_out, gpu_ctx) == 0) {
                            GLuint tmp = tex_u_curr; tex_u_curr = tex_out; tex_out = tmp;
                        } else {
                            fprintf(stderr, "paint composite gpu_run_program_to_tex failed\n"); fflush(stderr);
                        }
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

                /* Wave compute pass: use gpu_run_program_to_tex which internally
                   compiles the shader once, binds named input textures from the dict,
                   and renders into tex_out with no CPU readback.                     */
                gpu_tex_dict_set(compute_tex_dict, "u_curr", tex_u_curr);
                gpu_tex_dict_set(compute_tex_dict, "u_prev", tex_u_prev);
                gpu_program_set_boundary_mask(*prog_ref, bm);
                /* GPU timing: sample occasionally to avoid blocking every frame. */
                uint32_t _gpu_t_before = 0;
                if (gpu_time_sample_period > 0) _gpu_t_before = SDL_GetTicks();
                gpu_run_program_to_tex(*prog_ref, compute_tex_dict, tex_out, gpu_ctx);
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

                /* Apply damping (sponge) pass via expression-compiled program. */
                if (damping_gpu_prog && damping_tex) {
                    /* u_next=tex_u_curr (post-wave), u_curr=tex_u_prev, u_prev=tex_prev (pre-ping-pong tex_out),
                       sigma=damping_tex */
                    gpu_tex_dict_set(damping_tex_dict, "u_next",  tex_u_curr);
                    gpu_tex_dict_set(damping_tex_dict, "u_curr",  tex_u_prev);
                    gpu_tex_dict_set(damping_tex_dict, "u_prev",  tex_prev);
                    gpu_tex_dict_set(damping_tex_dict, "sigma",   damping_tex);
                    if (gpu_run_program_to_tex(damping_gpu_prog, damping_tex_dict, tex_tmp, gpu_ctx) == 0) {
                        GLuint ttmp = tex_u_curr; tex_u_curr = tex_tmp; tex_tmp = ttmp;
                    } else {
                        fprintf(stderr, "damping gpu_run_program_to_tex failed\n");
                    }
                }
                // increment sim counter for each physics step performed
                sim_step_counter++;
            } // end steps_per_frame loop
        } // end not paused

        // Unbind FBO and restore window viewport
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0,0,win_w,win_h);

        // Render current field to screen — one call via GPURenderConfig
        gpu_render_config_set_tex(disp_rc, "src",  tex_u_curr);
        gpu_render_config_set_tex(disp_rc, "mask", bm && bm->mask_tex ? bm->mask_tex : 0);
        glClearColor(0.1f,0.1f,0.12f,1.0f); glClear(GL_COLOR_BUFFER_BIT);
        gpu_render_config_draw(disp_rc);
        glUseProgram(0);  /* back to fixed-function for menus */
        // Draw overlay lines for mask on top of the display but beneath menus
        if (overlay_prog && bm && bm->mask_tex) {
            GLboolean depthEnabled_ov = glIsEnabled(GL_DEPTH_TEST);
            GLboolean blendEnabled_ov = glIsEnabled(GL_BLEND);
            if (depthEnabled_ov) glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram(overlay_prog);
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, bm->mask_tex);
            /* Use cached locations — no glGetUniformLocation per frame */
            if (loc_ov_mask     >= 0) glUniform1i(loc_ov_mask, 2);
            if (loc_ov_dims     >= 0) glUniform2i(loc_ov_dims, (GLint)nx, (GLint)ny);
            if (loc_ov_vis_off  >= 0) glUniform2i(loc_ov_vis_off,  (GLint)sponge, (GLint)sponge);
            if (loc_ov_vis_size >= 0) glUniform2i(loc_ov_vis_size, (GLint)nx_vis,  (GLint)ny_vis);
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
    wave_sim_gpu_free(&g);
    glDeleteTextures(1, &tex_u_curr); glDeleteTextures(1, &tex_u_prev); glDeleteTextures(1, &tex_out);
    glDeleteTextures(1, &tex_tmp);
    if (tex_src_desc) glDeleteTextures(1, &tex_src_desc);
    if (tex_paint) glDeleteTextures(1, &tex_paint);
    if (bm) boundary_mask_free(bm);
    free(u_curr_data); free(u_prev_data);
    if (paint_buf) free(paint_buf);
    grid_metadata_free(grid);
    SDL_GL_DeleteContext(ctx); SDL_DestroyWindow(win); SDL_Quit();
    return 0;
}
