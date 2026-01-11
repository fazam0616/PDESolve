/* Interactive smoke simulator using SDL/OpenGL and the Menu system.
 * Controls:
 * - Left-click to add density (amount controlled by "Emission")
 * - Use the menu to pause, change viscosity, diffusion, and emission amount
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <GL/gl.h>
#include "../include/grid.h"
#include "../include/smoke_sim.h"
#include "../include/smoke_sim_internal.h"
#include "../include/Menu.h"
#include "../include/literal.h"

// Helper: add density and velocity distributed with a Gaussian kernel
static void add_smoothed_density_velocity(SmokeSimState *s, int gx, int gy,
                                         double amount, double dvx, double dvy,
                                         double radius, uint32_t nx_u, uint32_t ny_u) {
    if (amount == 0.0 && dvx == 0.0 && dvy == 0.0) return;
    int r = (int)ceil(radius);
    double sigma = (radius > 0.0) ? (radius * 0.5) : 0.5;
    double two_sigma2 = 2.0 * sigma * sigma;
    double total_w = 0.0;
    // precompute weights
    int xmin = gx - r; if (xmin < 0) xmin = 0;
    int xmax = gx + r; if (xmax >= (int)nx_u) xmax = (int)nx_u - 1;
    int ymin = gy - r; if (ymin < 0) ymin = 0;
    int ymax = gy + r; if (ymax >= (int)ny_u) ymax = (int)ny_u - 1;
    int wcols = xmax - xmin + 1;
    int wrows = ymax - ymin + 1;
    double *weights = (double*)malloc(sizeof(double) * wcols * wrows);
    if (!weights) {
        // fallback: single-point add
        uint32_t idx[3] = { (uint32_t)gx, (uint32_t)gy, 0 };
        if (amount != 0.0) smoke_sim_add_density(s, idx, amount);
        if (dvx != 0.0 || dvy != 0.0) smoke_sim_add_velocity(s, idx, dvx, dvy);
        return;
    }
    int wi = 0;
    for (int j = ymin; j <= ymax; ++j) {
        for (int i = xmin; i <= xmax; ++i) {
            double dx = (double)(i - gx);
            double dy = (double)(j - gy);
            double dist2 = dx*dx + dy*dy;
            double w = exp(-dist2 / two_sigma2);
            weights[wi++] = w;
            total_w += w;
        }
    }
    if (total_w <= 0.0) total_w = 1.0;
    wi = 0;
    for (int j = ymin; j <= ymax; ++j) {
        for (int i = xmin; i <= xmax; ++i) {
            double w = weights[wi++] / total_w;
            uint32_t idx[3] = { (uint32_t)i, (uint32_t)j, 0 };
            if (amount != 0.0) smoke_sim_add_density(s, idx, amount * w);
            if (dvx != 0.0 || dvy != 0.0) smoke_sim_add_velocity(s, idx, dvx * w, dvy * w);
        }
    }
    free(weights);
}

// Render mode radio callbacks
typedef struct {
    double *mode;   // pointer to ui_render_mode
    int *density;
    int *velocity;
    int *rgb;
} RenderModeData;

static void on_render_density_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    RenderModeData *d = (RenderModeData*)user_data;
    if (!d) return;
    *(d->density) = 1; *(d->velocity) = 0; *(d->rgb) = 0; *(d->mode) = 0.0;
}

static void on_render_velocity_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    RenderModeData *d = (RenderModeData*)user_data;
    if (!d) return;
    *(d->density) = 0; *(d->velocity) = 1; *(d->rgb) = 0; *(d->mode) = 1.0;
}

static void on_render_rgb_selected(VariableInteraction *vi, void *user_data) {
    (void)vi;
    RenderModeData *d = (RenderModeData*)user_data;
    if (!d) return;
    *(d->density) = 0; *(d->velocity) = 0; *(d->rgb) = 1; *(d->mode) = 2.0;
}

int main(int argc, char **argv) {
    uint32_t nx = 128, ny = 128;
    double dt = 0.02;

    if (argc > 1) nx = (uint32_t)atoi(argv[1]);
    if (argc > 2) ny = (uint32_t)atoi(argv[2]);
    if (argc > 3) dt = atof(argv[3]);

    uint32_t dims[3] = { nx, ny, 1 };
    double spacing[3] = { 1.0/(nx-1), 1.0/(ny-1), 1.0 };
    double origin[3] = { 0.0, 0.0, 0.0 };

    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    if (!grid) {
        fprintf(stderr, "Failed to create grid\n");
        return EXIT_FAILURE;
    }

    SmokeSimState *s = smoke_sim_create(grid, dt);
    if (!s) {
        fprintf(stderr, "Failed to create smoke sim state\n");
        grid_metadata_free(grid);
        return EXIT_FAILURE;
    }

    // initial seed in center
    uint32_t idx[3];
    for (uint32_t j = ny/3; j < 2*ny/3; j++) {
        for (uint32_t i = nx/3; i < 2*nx/3; i++) {
            idx[0] = i; idx[1] = j; idx[2] = 0;
            smoke_sim_add_density(s, idx, 0.6);
        }
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        smoke_sim_free(s); grid_metadata_free(grid);
        return EXIT_FAILURE;
    }

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_Window *window = SDL_CreateWindow("Interactive Smoke Sim", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          800, 800, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit(); smoke_sim_free(s); grid_metadata_free(grid);
        return EXIT_FAILURE;
    }
    SDL_GLContext glctx = SDL_GL_CreateContext(window);
    if (!glctx) {
        fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window); SDL_Quit(); smoke_sim_free(s); grid_metadata_free(grid);
        return EXIT_FAILURE;
    }

    // Menu setup: base menu + mouse menu + sim menu (mutually exclusive)
    menu_set_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14);

    // Base menu toggles which submenu is shown
    int ui_show_mouse_controls = 0;
    int ui_show_sim_controls = 0;
    Menu *base_menu = menu_create(8, 8, 220, 80, 1, "Controls", (Color){255,255,255,255}, (Color){32,32,32,220});
    MenuRow *base_r1 = menurow_create();
    menurow_add_interaction(base_r1, variableinteraction_create(&ui_show_mouse_controls, "Mouse Controls", 0, 1, VAR_BOOL, NULL, NULL));
    menu_add_row(base_menu, base_r1);
    MenuRow *base_r2 = menurow_create();
    menurow_add_interaction(base_r2, variableinteraction_create(&ui_show_sim_controls, "Sim Controls", 0, 1, VAR_BOOL, NULL, NULL));
    menu_add_row(base_menu, base_r2);

    // Mouse menu: default values for adding waves / sources
    double ui_emission = 0.5;
    double ui_default_source_frequency = 1.0;
    double ui_default_source_phase = 0.0;
    double ui_default_source_amplitude = 0.5;
    double ui_default_source_radius = 4.0;
    Menu *mouse_menu = menu_create(240, 8, 260, 160, 1, "Mouse Controls", (Color){255,255,255,255}, (Color){32,32,32,220});
    MenuRow *m_r1 = menurow_create();
    menurow_add_interaction(m_r1, variableinteraction_create(&ui_emission, "Emission", 0.0, 2.0, VAR_SLIDER, NULL, NULL));
    menu_add_row(mouse_menu, m_r1);
    MenuRow *m_r2 = menurow_create();
    menurow_add_interaction(m_r2, variableinteraction_create(&ui_default_source_frequency, "Src Freq", 0.1, 20.0, VAR_SLIDER, NULL, NULL));
    menurow_add_interaction(m_r2, variableinteraction_create(&ui_default_source_phase, "Src Phase", 0.0, 6.28, VAR_SLIDER, NULL, NULL));
    menu_add_row(mouse_menu, m_r2);
    MenuRow *m_r3 = menurow_create();
    menurow_add_interaction(m_r3, variableinteraction_create(&ui_default_source_amplitude, "Src Amp", 0.0, 2.0, VAR_SLIDER, NULL, NULL));
    menurow_add_interaction(m_r3, variableinteraction_create(&ui_default_source_radius, "Src Radius", 1.0, 32.0, VAR_SLIDER, NULL, NULL));
    menu_add_row(mouse_menu, m_r3);

    // Sim menu: simulation controls and render options
    int ui_paused = 0;
    double ui_viscosity = s->viscosity;
    double ui_diffusion = s->diffusion;
    double ui_dt = s->dt;
    double ui_render_mode = 0.0; // 0=DENSITY,1=VELMAG,2=RGB
    double ui_scale = 1.0;
    int ui_show_overlays = 1;
    Menu *sim_menu = menu_create(520, 8, 260, 220, 1, "Simulation Controls", (Color){255,255,255,255}, (Color){32,32,32,220});
    MenuRow *s_r1 = menurow_create();
    menurow_add_interaction(s_r1, variableinteraction_create(&ui_paused, "Paused", 0, 1, VAR_BOOL, NULL, NULL));
    menu_add_row(sim_menu, s_r1);
    MenuRow *s_r2 = menurow_create();
    menurow_add_interaction(s_r2, variableinteraction_create(&ui_viscosity, "Viscosity", 0.0, 0.1, VAR_SLIDER, NULL, NULL));
    menurow_add_interaction(s_r2, variableinteraction_create(&ui_diffusion, "Diffusion", 0.0, 0.01, VAR_SLIDER, NULL, NULL));
    menu_add_row(sim_menu, s_r2);
    MenuRow *s_r3 = menurow_create();
    menurow_add_interaction(s_r3, variableinteraction_create(&ui_dt, "dt", 0.001, 0.05, VAR_SLIDER, NULL, NULL));
    menu_add_row(sim_menu, s_r3);
    // Render mode radio buttons (mutually exclusive)
    int ui_render_density = 1;
    int ui_render_velocity = 0;
    int ui_render_rgb = 0;
    RenderModeData render_mode_data = { &ui_render_mode, &ui_render_density, &ui_render_velocity, &ui_render_rgb };
    MenuRow *s_r4 = menurow_create();
    menurow_add_interaction(s_r4, variableinteraction_create(&ui_render_density, "Density", 0, 1, VAR_BOOL, on_render_density_selected, &render_mode_data));
    menurow_add_interaction(s_r4, variableinteraction_create(&ui_render_velocity, "Velocity", 0, 1, VAR_BOOL, on_render_velocity_selected, &render_mode_data));
    menurow_add_interaction(s_r4, variableinteraction_create(&ui_render_rgb, "RGB", 0, 1, VAR_BOOL, on_render_rgb_selected, &render_mode_data));
    // Scale slider on its own row
    menu_add_row(sim_menu, s_r4);
    MenuRow *s_r_scale = menurow_create();
    menurow_add_interaction(s_r_scale, variableinteraction_create(&ui_scale, "Scale", 0.01, 10.0, VAR_SLIDER, NULL, NULL));
    menu_add_row(sim_menu, s_r_scale);
    MenuRow *s_r5 = menurow_create();
    menurow_add_interaction(s_r5, variableinteraction_create(&ui_show_overlays, "Overlays", 0, 1, VAR_BOOL, NULL, NULL));
    menu_add_row(sim_menu, s_r5);

    int window_w = 800, window_h = 800;

    // grid size for rendering
    uint32_t nx_u = nx; uint32_t ny_u = ny;

    // Mouse modes (mode switching via scroll wheel)
    typedef enum {
        MOUSE_NONE = 0,
        MOUSE_ADD_WAVE = 1,
        MOUSE_ADD_BARRIER = 2,
        MOUSE_SOURCE = 3,
        MOUSE_PROBE = 4,
        MOUSE_NMODES = 5
    } MouseMode;

    const char *mouse_mode_names[] = { "NONE", "ADD_WAVE", "ADD_BARRIER", "SOURCE", "PROBE" };

    int current_mouse_mode = MOUSE_ADD_WAVE;
    int painting = 0; // for continuous wave painting

    // Simple barrier storage (pixel/grid coords)
    typedef struct { int x, y; } BarrierPoint;
    BarrierPoint *barriers = (BarrierPoint*)malloc(sizeof(BarrierPoint) * 256);
    int n_barriers = 0;
    int dragging_barrier = -1;

    // Simple source storage
    typedef struct { int x, y; double frequency, phase, amplitude, radius; int active; } SourcePoint;
    SourcePoint *sources = (SourcePoint*)malloc(sizeof(SourcePoint) * 64);
    int n_sources = 0;
    int dragging_source = -1;

    int running = 1;
    SDL_Event ev;
    const double tol = 1e-6; const int max_iter = 4000;

    while (running) {
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) { running = 0; break; }
            else if (ev.type == SDL_WINDOWEVENT) {
                if (ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    window_w = ev.window.data1; window_h = ev.window.data2;
                    glViewport(0, 0, window_w, window_h);
                }
            } else if (ev.type == SDL_MOUSEWHEEL) {
                // cycle mouse modes
                if (ev.wheel.y > 0) current_mouse_mode = (current_mouse_mode + 1) % MOUSE_NMODES;
                else if (ev.wheel.y < 0) current_mouse_mode = (current_mouse_mode - 1 + MOUSE_NMODES) % MOUSE_NMODES;
                char titlebuf[128];
                snprintf(titlebuf, sizeof(titlebuf), "Interactive Smoke Sim - Mode: %s", mouse_mode_names[current_mouse_mode]);
                SDL_SetWindowTitle(window, titlebuf);
                printf("Mouse mode changed to %s\n", mouse_mode_names[current_mouse_mode]);
            } else if (ev.type == SDL_MOUSEBUTTONDOWN || ev.type == SDL_MOUSEBUTTONUP) {
                int mx = ev.button.x, my = ev.button.y;
                // route to base menu first, then to the visible submenu
                if (menu_handle_mouse_button(base_menu, ev.button.button, ev.button.state, mx, my)) {
                    if (ui_show_mouse_controls) ui_show_sim_controls = 0;
                    if (ui_show_sim_controls) ui_show_mouse_controls = 0;
                    continue;
                }
                if (ui_show_mouse_controls) {
                    if (menu_handle_mouse_button(mouse_menu, ev.button.button, ev.button.state, mx, my)) {
                        if (ui_show_mouse_controls) ui_show_sim_controls = 0;
                        continue;
                    }
                }
                if (ui_show_sim_controls) {
                    if (menu_handle_mouse_button(sim_menu, ev.button.button, ev.button.state, mx, my)) {
                        if (ui_show_sim_controls) ui_show_mouse_controls = 0;
                        continue;
                    }
                }
                // map mouse to grid coordinates (invert Y to match rendered buffer)
                int gx = (int)((double)mx / (double)window_w * (double)nx_u);
                int gy = (int)((double)(window_h - 1 - my) / (double)window_h * (double)ny_u);
                if (gx < 0) gx = 0; if (gx >= (int)nx_u) gx = nx_u-1;
                if (gy < 0) gy = 0; if (gy >= (int)ny_u) gy = ny_u-1;

                if (ev.button.button == SDL_BUTTON_LEFT) {
                    if (ev.button.state == SDL_PRESSED) {
                        if (current_mouse_mode == MOUSE_ADD_WAVE) {
                            painting = 1;
                            uint32_t gidx[3] = { (uint32_t)gx, (uint32_t)gy, 0 };
                            printf("Add wave at grid (%d,%d) amount=%.3f\n", gx, gy, ui_emission);
                            add_smoothed_density_velocity(s, gx, gy, ui_emission, 0.0, ui_emission * 0.1,
                                                          ui_default_source_radius, nx_u, ny_u);
                        } else if (current_mouse_mode == MOUSE_ADD_BARRIER) {
                            // find nearby barrier to drag
                            int found = -1;
                            for (int i = 0; i < n_barriers; ++i) {
                                int dx = barriers[i].x - mx; int dy = barriers[i].y - my;
                                if (dx*dx + dy*dy < 64) { found = i; break; }
                            }
                            if (found >= 0) {
                                dragging_barrier = found;
                                printf("Start dragging barrier %d\n", found);
                            } else if (n_barriers < 256) {
                                barriers[n_barriers].x = mx; barriers[n_barriers].y = my; n_barriers++;
                                dragging_barrier = n_barriers - 1;
                                printf("Added barrier %d at pixel (%d,%d)\n", dragging_barrier, mx, my);
                            }
                        } else if (current_mouse_mode == MOUSE_SOURCE) {
                            int found = -1;
                            for (int i = 0; i < n_sources; ++i) {
                                int dx = sources[i].x - mx; int dy = sources[i].y - my;
                                if (dx*dx + dy*dy < 64) { found = i; break; }
                            }
                            if (found >= 0) {
                                dragging_source = found;
                                printf("Start dragging source %d\n", found);
                            } else if (n_sources < 64) {
                                sources[n_sources].x = mx; sources[n_sources].y = my;
                                sources[n_sources].frequency = ui_default_source_frequency; sources[n_sources].phase = ui_default_source_phase;
                                sources[n_sources].amplitude = ui_default_source_amplitude; sources[n_sources].radius = ui_default_source_radius;
                                sources[n_sources].active = 1;
                                n_sources++; dragging_source = n_sources - 1;
                                printf("Added source %d at pixel (%d,%d) amp=%.3f\n", dragging_source, mx, my, sources[n_sources-1].amplitude);
                            }
                        } else if (current_mouse_mode == MOUSE_PROBE) {
                            // print sample values at probe
                            uint32_t gidx[3] = { (uint32_t)gx, (uint32_t)gy, 0 };
                            double dv = literal_get(&s->density->data, gidx);
                            printf("Probe at (%d,%d): density=%.6f\n", gx, gy, dv);
                        }
                    } else { // released
                        painting = 0; dragging_barrier = -1; dragging_source = -1;
                        printf("Mouse button released\n");
                    }
                }
            } else if (ev.type == SDL_MOUSEMOTION) {
                // route motion to base menu first, then visible submenu
                if (menu_handle_mouse_motion(base_menu, ev.motion.x, ev.motion.y)) {
                    if (ui_show_mouse_controls) ui_show_sim_controls = 0;
                    if (ui_show_sim_controls) ui_show_mouse_controls = 0;
                    continue;
                }
                if (ui_show_mouse_controls && menu_handle_mouse_motion(mouse_menu, ev.motion.x, ev.motion.y)) {
                    if (ui_show_mouse_controls) ui_show_sim_controls = 0;
                    continue;
                }
                if (ui_show_sim_controls && menu_handle_mouse_motion(sim_menu, ev.motion.x, ev.motion.y)) {
                    if (ui_show_sim_controls) ui_show_mouse_controls = 0;
                    continue;
                }
                int mx = ev.motion.x, my = ev.motion.y;
                int gx = (int)((double)mx / (double)window_w * (double)nx_u);
                int gy = (int)((double)(window_h - 1 - my) / (double)window_h * (double)ny_u);
                if (gx < 0) gx = 0; if (gx >= (int)nx_u) gx = nx_u-1;
                if (gy < 0) gy = 0; if (gy >= (int)ny_u) gy = ny_u-1;
                if (painting && current_mouse_mode == MOUSE_ADD_WAVE) {
                    uint32_t gidx[3] = { (uint32_t)gx, (uint32_t)gy, 0 };
                    double addamt = ui_emission * 0.2;
                    printf("Paint at (%d,%d) amount=%.3f\n", gx, gy, addamt);
                    add_smoothed_density_velocity(s, gx, gy, addamt, 0.0, addamt * 0.1,
                                                  ui_default_source_radius, nx_u, ny_u);
                }
                if (dragging_barrier >= 0 && dragging_barrier < n_barriers) {
                    barriers[dragging_barrier].x = mx; barriers[dragging_barrier].y = my;
                    printf("Dragging barrier %d -> (%d,%d)\n", dragging_barrier, mx, my);
                }
                if (dragging_source >= 0 && dragging_source < n_sources) {
                    sources[dragging_source].x = mx; sources[dragging_source].y = my;
                    printf("Dragging source %d -> (%d,%d)\n", dragging_source, mx, my);
                }
            }
        }

        // Apply UI parameters back to sim
        s->viscosity = ui_viscosity;
        s->diffusion = ui_diffusion;
        s->dt = ui_dt;

        if (!ui_paused) {
            smoke_sim_step(s->density, s->vx, s->vy, s->dt, s->viscosity, tol, max_iter);
            s->sim_time += s->dt;
            s->step_count++;
        }

        // Emit from sources: map source pixel coords to grid and add density
        for (int si = 0; si < n_sources; ++si) {
            if (!sources[si].active) continue;
            // convert pixel to grid coords
            int mx = sources[si].x; int my = sources[si].y;
            int gx = (int)((double)mx / (double)window_w * (double)nx_u);
            int gy = (int)((double)(window_h - 1 - my) / (double)window_h * (double)ny_u);
            if (gx < 0) gx = 0; if (gx >= (int)nx_u) gx = nx_u-1;
            if (gy < 0) gy = 0; if (gy >= (int)ny_u) gy = ny_u-1;
            // simple periodic/sinusoidal emission
            double phase = sources[si].phase + s->sim_time * sources[si].frequency * 2.0 * 3.141592653589793;
            double factor = 0.5 * (1.0 + sin(phase));
            double amount = sources[si].amplitude * factor * s->dt;
            uint32_t gidx[3] = { (uint32_t)gx, (uint32_t)gy, 0 };
            if (amount > 0.0) {
                add_smoothed_density_velocity(s, gx, gy, amount, 0.0, amount * 0.2,
                                              sources[si].radius, nx_u, ny_u);
            }
        }

        // Render grid as quads (modeled after interactive_wave_sim)
        int render_mode = (int)floor(ui_render_mode + 0.5);
        glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, window_w, window_h, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();

        float cell_width = (float)window_w / (float)nx_u;
        float cell_height = (float)window_h / (float)ny_u;

        glBegin(GL_QUADS);
        for (uint32_t iy = 0; iy < ny_u; iy++) {
            for (uint32_t ix = 0; ix < nx_u; ix++) {
                uint32_t sx = ix;
                uint32_t sy = (ny_u - 1) - iy; // invert sampling to match top-left origin
                uint32_t id[3] = { sx, sy, 0 };
                double r = 0.0, g = 0.0, b = 0.0;
                if (render_mode == 0) {
                    double v = literal_get(&s->density->data, id);
                    double vv = v * ui_scale;
                    if (vv < 0.0) vv = 0.0; if (vv > 1.0) vv = 1.0;
                    r = g = b = vv;
                } else if (render_mode == 1) {
                    double vxv = literal_get(&s->vx->data, id);
                    double vyv = literal_get(&s->vy->data, id);
                    double mag = sqrt(vxv*vxv + vyv*vyv) * ui_scale;
                    if (mag < 0.0) mag = 0.0; if (mag > 1.0) mag = 1.0;
                    r = g = b = mag;
                } else {
                    double vxv = literal_get(&s->vx->data, id);
                    double vyv = literal_get(&s->vy->data, id);
                    double dv = literal_get(&s->density->data, id);
                    double sr = (vxv * ui_scale + 1.0) * 0.5; if (sr < 0.0) sr = 0.0; if (sr > 1.0) sr = 1.0;
                    double sg = (vyv * ui_scale + 1.0) * 0.5; if (sg < 0.0) sg = 0.0; if (sg > 1.0) sg = 1.0;
                    double sb = dv * ui_scale; if (sb < 0.0) sb = 0.0; if (sb > 1.0) sb = 1.0;
                    r = sr; g = sg; b = sb;
                }

                GLubyte rc = (GLubyte)(fmin(fmax(r,0.0),1.0) * 255.0);
                GLubyte gc = (GLubyte)(fmin(fmax(g,0.0),1.0) * 255.0);
                GLubyte bc = (GLubyte)(fmin(fmax(b,0.0),1.0) * 255.0);
                glColor3ub(rc, gc, bc);

                float x0 = ix * cell_width;
                float y0 = iy * cell_height;
                float x1 = x0 + cell_width;
                float y1 = y0 + cell_height;
                glVertex2f(x0, y0);
                glVertex2f(x1, y0);
                glVertex2f(x1, y1);
                glVertex2f(x0, y1);
            }
        }
        glEnd();

        // render overlays (barriers/sources) if enabled
        if (ui_show_overlays) {
            // draw small red squares for barriers and green for sources
            glColor3ub(255,0,0);
            for (int i = 0; i < n_barriers; ++i) {
                int bx = barriers[i].x; int by = barriers[i].y;
                int size = 6;
                glBegin(GL_QUADS);
                glVertex2i(bx - size, by - size);
                glVertex2i(bx + size, by - size);
                glVertex2i(bx + size, by + size);
                glVertex2i(bx - size, by + size);
                glEnd();
            }
            glColor3ub(0,255,0);
            for (int i = 0; i < n_sources; ++i) {
                int sx = sources[i].x; int sy = sources[i].y; int size = 8;
                glBegin(GL_QUADS);
                glVertex2i(sx - size, sy - size);
                glVertex2i(sx + size, sy - size);
                glVertex2i(sx + size, sy + size);
                glVertex2i(sx - size, sy + size);
                glEnd();
            }
        }

        // render base menu and active submenu (mutually exclusive)
        menu_render(base_menu, window_w, window_h);
        if (ui_show_mouse_controls) menu_render(mouse_menu, window_w, window_h);
        if (ui_show_sim_controls) menu_render(sim_menu, window_w, window_h);

        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
        SDL_GL_SwapWindow(window);

        SDL_Delay(16); // ~60 FPS
    }

    menu_free(base_menu);
    menu_free(mouse_menu);
    menu_free(sim_menu);
    menu_clear_font();
    SDL_GL_DeleteContext(glctx);
    SDL_DestroyWindow(window);
    SDL_Quit();

    smoke_sim_free(s);
    grid_metadata_free(grid);
    return EXIT_SUCCESS;
}
