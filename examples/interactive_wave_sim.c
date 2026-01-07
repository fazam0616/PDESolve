/*
 * Interactive 2D Wave Simulator with OpenGL Rendering
 * 
 * Features:
 * - Real-time OpenGL rendering at fixed ~60 FPS
 * - Decoupled physics simulation (runs as fast as possible)
 * - Interactive menu system for runtime parameter adjustment
 * - Multiple render modes (height, velocity, RGB)
 * - Mouse interaction modes (add waves, barriers)
 * - Simulation controls (play/pause, reset, step size)
 * - Sponge layer boundary absorption
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <GL/gl.h>
#include "../include/grid.h"
#include "../include/literal.h"
#include "../include/Menu.h"
#include "../include/expression.h"
#include "../include/dictionary.h"

// =============================================================================
// Constants and Configuration
// =============================================================================

#define SPONGE_WIDTH_PERCENT 0.5  // Sponge width as fraction of visible region
#define SIGMA_MAX 10.0             // Maximum damping coefficient in sponge layer
#define TARGET_FPS 60              // Target rendering framerate
#define FRAME_TIME_MS (1000.0 / TARGET_FPS)

// Render modes
typedef enum {
    RENDER_HEIGHT,      // Blue for negative, red for positive
    RENDER_VELOCITY,    // Grayscale velocity magnitude
    RENDER_RGB          // R=vx, G=vy, B=height
} RenderMode;

// Mouse interaction modes
typedef enum {
    MOUSE_NONE,         // No interaction
    MOUSE_ADD_WAVE,     // Click or drag to add wave pulses
    MOUSE_ADD_BARRIER,  // Click to add reflecting barriers (line segments)
    MOUSE_SOURCE        // Click to add/select/move oscillating wave sources
} MouseMode;

// Barrier point storage
typedef struct {
    double x, y;        // Physical coordinates
} BarrierPoint;

#define MAX_BARRIER_POINTS 1024
static BarrierPoint barrier_points[MAX_BARRIER_POINTS];
static int n_barrier_points = 0;

// Barrier dragging state
static int dragging_barrier_point = -1;  // Index of point being dragged, -1 if none
static int dragging_barrier_idx = -1;    // Interior boundary index being modified
static int barrier_click_candidate = -1; // Point clicked on mouse down (for delete on release)
static double barrier_click_start_x = 0.0; // Mouse position on mouse down
static double barrier_click_start_y = 0.0;

// Wave source definition
typedef struct {
    double x, y;            // Physical coordinates (visible region)
    double frequency;       // Oscillation frequency (Hz)
    double phase;           // Phase offset (radians)
    double amplitude;       // Wave amplitude
    double radius;          // Application radius (physical units)
    int active;             // Whether this source is active
} WaveSource;

// Source click state (for drag vs deselect detection)
static WaveSource *source_click_candidate = NULL; // Source clicked on mouse down
static double source_click_start_x = 0.0; // Mouse position on mouse down
static double source_click_start_y = 0.0;

#define MAX_WAVE_SOURCES 32
static WaveSource wave_sources[MAX_WAVE_SOURCES];
static int n_wave_sources = 0;
static WaveSource *selected_source = NULL;  // Currently selected source (for editing)
static WaveSource *last_menu_source = NULL; // Track which source the menu was last built for

// =============================================================================
// Global Simulation State
// =============================================================================

typedef struct {
    // Grid and fields
    GridMetadata *grid;
    GridField *u_curr;      // Current wave height
    GridField *u_prev;      // Previous wave height
    GridField *u_next;      // Next wave height (workspace)
    GridField *u_source;    // Source contributions (sinusoidal drivers)
    
    // Expression system for wave equation evaluation
    Expression *wave_expr;  // Compiled expression: 2*u_curr - u_prev + c²*dt²*∇²u_curr
    Dictionary *vars;       // Variable dictionary for evaluation
    
    // Physical parameters (visible region)
    double Lx_visible;      // Visible domain width
    double Ly_visible;      // Visible domain height
    int nx_visible;         // Visible grid points in x
    int ny_visible;         // Visible grid points in y
    
    // Sponge layer parameters
    int sponge_width;       // Number of cells for sponge layer
    
    // Total grid size (includes sponge)
    uint32_t nx_total;
    uint32_t ny_total;
    double dx, dy;
    
    // Simulation parameters
    double dt;              // Time step
    double wave_speed;      // Wave propagation speed
    double max_sim_speed;   // Maximum simulation steps per second (0 = unlimited)
    int paused;             // Simulation paused flag
    
    // Mouse interaction parameters
    double wave_amplitude;  // Amplitude for added waves
    double wave_spread;     // Spread (sigma) for Gaussian waves
    
    // Statistics
    double sim_time;        // Total simulated time
    uint64_t step_count;    // Total simulation steps
    uint64_t frame_count;   // Total rendered frames
    
    // Performance tracking
    double steps_per_second;
    double last_step_time;
    uint64_t steps_since_last_update;
    
    // Energy tracking
    double total_energy;        // Current total energy in the system
    double baseline_energy;     // Expected/initial energy (tracked from waves added)
    double energy_error;        // Relative error: (current - baseline) / baseline
} SimulationState;

// =============================================================================
// Rendering State
// =============================================================================

typedef struct {
    RenderMode mode;
    double value_scale;     // Color intensity scaling
    int show_boundaries;    // Render boundary overlays
    int show_stats;         // Render statistics overlay
    
    // Radio button states for render mode
    int mode_height;
    int mode_velocity;
    int mode_rgb;
} RenderState;

// =============================================================================
// Global state pointers (for menu callbacks)
// =============================================================================

static SimulationState *g_sim = NULL;
static RenderState *g_render = NULL;

// Mouse mode state (radio button group)
static MouseMode g_mouse_mode = MOUSE_NONE;
static int g_mouse_none = 1;
static int g_mouse_add_wave = 0;
static int g_mouse_add_barrier = 0;
static int g_mouse_source = 0;

// Default source frequency for adding new sources
static double g_default_source_frequency = 5.0;

// Default source phase for adding new sources
static double g_default_source_phase = 0.0;

// Wave painting state (for drag support)
static int wave_painting = 0;           // Is user currently painting waves?
static double last_wave_x = -1.0;       // Last position where wave was added
static double last_wave_y = -1.0;
static double wave_paint_spacing = 0.1; // Minimum distance between waves (physical units)

// Centralized menu slider drag state
typedef struct {
    VariableInteraction *dragging_interaction;  // NULL if no drag active
    int drag_start_x;                           // Screen x where drag started
    double drag_start_value;                    // Value before drag started
    Menu *drag_source_menu;                     // Which menu owns the slider
} MenuDragState;

static MenuDragState g_menu_drag_state = {
    .dragging_interaction = NULL,
    .drag_start_x = 0,
    .drag_start_value = 0.0,
    .drag_source_menu = NULL
};

// Mouse mode names for display
static const char* g_mouse_mode_names[] = {
    "NONE",
    "WAVE",
    "BARRIER",
    "SOURCE"
};

// Menu visibility flags
static int g_show_base_menu = 1;
static int g_show_mouse_controls = 0;
static int g_show_sim_controls = 0;

// =============================================================================
// Initialization Functions
// =============================================================================

// Initialize wave with Gaussian bump at center
Literal* init_gaussian_centered(const double *coords, int n_dims) {
    (void)n_dims;  // Unused parameter
    double x = coords[0];
    double y = coords[1];
    
    // Convert to visible coordinates
    double offset_x = g_sim->sponge_width * g_sim->dx;
    double offset_y = g_sim->sponge_width * g_sim->dy;
    double x_vis = x - offset_x;
    double y_vis = y - offset_y;
    
    // Center at middle of visible region
    double center_x = g_sim->Lx_visible / 2.0;
    double center_y = g_sim->Ly_visible / 2.0;
    
    double dx_c = x_vis - center_x;
    double dy_c = y_vis - center_y;
    double r2 = dx_c * dx_c + dy_c * dy_c;
    
    double amplitude = 1.0;
    double sigma = 0.1;
    double value = amplitude * exp(-r2 / (2.0 * sigma * sigma));
    
    return literal_create_scalar(value);
}

// Forward declarations for energy functions
double calculate_total_energy(SimulationState *sim);
double calculate_interaction_energy(SimulationState *sim, double phys_x, double phys_y, double amplitude, double sigma);
double estimate_wave_energy(SimulationState *sim, double amplitude, double sigma);

// Reset simulation to initial state
void reset_simulation(SimulationState *sim) {
    // grid_field_init_from_function(sim->u_curr, init_gaussian_centered);
    // grid_field_init_from_function(sim->u_prev, init_gaussian_centered);
    
    // Fill u_next with zeros
    Literal *zero = literal_create_scalar(0.0);
    grid_field_fill(sim->u_next, zero);
    grid_field_fill(sim->u_prev, zero);
    grid_field_fill(sim->u_curr, zero);
    literal_free(zero);
    
    sim->sim_time = 0.0;
    sim->step_count = 0;
    sim->steps_since_last_update = 0;
    sim->last_step_time = SDL_GetTicks() / 1000.0;
    
    // Calculate baseline energy from initial conditions
    sim->baseline_energy = calculate_total_energy(sim);
    sim->total_energy = sim->baseline_energy;
    sim->energy_error = 0.0;
    
    printf("Initial energy: %.6f\n", sim->baseline_energy);
}

// Create simulation state with given parameters
SimulationState* simulation_create(double Lx_vis, double Ly_vis, int nx_vis, int ny_vis) {
    SimulationState *sim = (SimulationState*)calloc(1, sizeof(SimulationState));
    
    sim->Lx_visible = Lx_vis;
    sim->Ly_visible = Ly_vis;
    sim->nx_visible = nx_vis;
    sim->ny_visible = ny_vis;
    
    // Calculate sponge width as 25% of average visible dimension
    int avg_visible = (nx_vis + ny_vis) / 2;
    sim->sponge_width = (int)(avg_visible * SPONGE_WIDTH_PERCENT);
    if (sim->sponge_width < 10) sim->sponge_width = 10;  // Minimum 10 cells
    
    // Total grid includes sponge
    sim->nx_total = nx_vis + 2 * sim->sponge_width;
    sim->ny_total = ny_vis + 2 * sim->sponge_width;
    
    // Grid spacing based on visible region
    sim->dx = Lx_vis / (nx_vis - 1);
    sim->dy = Ly_vis / (ny_vis - 1);
    
    // Create grid
    uint32_t dims[3] = {sim->nx_total, sim->ny_total, 1};
    double spacing[3] = {sim->dx, sim->dy, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    sim->grid = grid_metadata_create(dims, spacing, origin, 2);
    
    // Set open boundaries (sponge layer absorption)
    grid_set_boundary(sim->grid, 0, 0, BC_OPEN, 0.0);  // Left
    grid_set_boundary(sim->grid, 0, 1, BC_OPEN, 0.0);  // Right
    grid_set_boundary(sim->grid, 1, 0, BC_OPEN, 0.0);  // Bottom
    grid_set_boundary(sim->grid, 1, 1, BC_OPEN, 0.0);  // Top
    
    // Create fields
    sim->u_curr = grid_field_create(sim->grid);
    sim->u_prev = grid_field_create(sim->grid);
    sim->u_next = grid_field_create(sim->grid);
    sim->u_source = grid_field_create(sim->grid);  // Source contributions
    
    // Simulation parameters
    sim->dt = 0.004;
    sim->wave_speed = 1.0;
    sim->max_sim_speed = 2000.0;  // Default max 2000 steps/sec
    sim->paused = 0;
    
    // Mouse interaction parameters
    sim->wave_amplitude = 0.3;  // Default amplitude
    sim->wave_spread = 0.05;  // Default spread in physical units
    
    // Build wave equation expression: u_next = 2*u_curr - u_prev + c²*dt²*∇²u_curr
    // Use expr_laplacian for proper second-order derivative handling
    Expression *u_curr_var = expr_variable("u_curr");
    Expression *u_prev_var = expr_variable("u_prev");
    
    // Laplacian: ∇²u_curr (uses grid_field_laplacian internally - order=2 derivatives)
    Expression *laplacian = expr_laplacian(expr_variable("u_curr"));
    
    // c²*dt²*∇²u_curr
    double c2_dt2 = sim->wave_speed * sim->wave_speed * sim->dt * sim->dt;
    Expression *c2dt2_lit = expr_literal(literal_create_scalar(c2_dt2));
    Expression *accel_term = expr_multiply(c2dt2_lit, laplacian);
    
    // 2*u_curr
    Expression *two_lit = expr_literal(literal_create_scalar(2.0));
    Expression *two_u_curr = expr_multiply(two_lit, u_curr_var);
    
    // 2*u_curr - u_prev
    Expression *neg_u_prev = expr_negate(u_prev_var);
    Expression *diff = expr_add(two_u_curr, neg_u_prev);
    
    // Final: 2*u_curr - u_prev + c²*dt²*∇²u_curr
    sim->wave_expr = expr_add(diff, accel_term);
    
    // Create variable dictionary (capacity for u_curr and u_prev)
    sim->vars = dict_create(8);
    
    // Initialize fields
    g_sim = sim;  // Set global for init function
    reset_simulation(sim);
    
    printf("Simulation initialized:\n");
    printf("  Visible: %dx%d grid, %.2f x %.2f domain\n", 
           nx_vis, ny_vis, Lx_vis, Ly_vis);
    printf("  Sponge: %d cells (%.1f%% of avg dimension)\n",
           sim->sponge_width, SPONGE_WIDTH_PERCENT * 100.0);
    printf("  Total: %dx%d (with sponge)\n", sim->nx_total, sim->ny_total);
    printf("  Spacing: dx=%.4f, dy=%.4f\n", sim->dx, sim->dy);
    printf("  CFL = %.4f (should be < 0.707)\n", 
           sim->wave_speed * sim->dt / fmin(sim->dx, sim->dy));
    
    return sim;
}

void simulation_free(SimulationState *sim) {
    if (!sim) return;
    grid_field_free(sim->u_curr);
    grid_field_free(sim->u_prev);
    grid_field_free(sim->u_next);
    grid_field_free(sim->u_source);
    grid_metadata_free(sim->grid);
    if (sim->wave_expr) expression_free(sim->wave_expr);
    if (sim->vars) dict_free(sim->vars);
    free(sim);
}

// =============================================================================
// Energy Calculation
// =============================================================================

// Calculate total energy in the wave field
// E = ∫(1/2 * v² + 1/2 * c² * |∇u|²) dV
// where v = (u_curr - u_prev) / dt is velocity
// and |∇u|² is the gradient magnitude squared
double calculate_total_energy(SimulationState *sim) {
    double total = 0.0;
    double c_squared = sim->wave_speed * sim->wave_speed;
    double dt = sim->dt;
    double dV = sim->dx * sim->dy;  // Volume element (area in 2D)
    
    // Only calculate over visible region to avoid counting sponge damping
    int start_x = sim->sponge_width;
    int end_x = sim->nx_total - sim->sponge_width;
    int start_y = sim->sponge_width;
    int end_y = sim->ny_total - sim->sponge_width;
    
    #pragma omp parallel for collapse(2) reduction(+:total)
    for (int iy = start_y; iy < end_y; iy++) {
        for (int ix = start_x; ix < end_x; ix++) {
            uint32_t idx[3] = {(uint32_t)ix, (uint32_t)iy, 0};
            
            // Get current and previous values
            Literal *lit_curr = grid_field_get(sim->u_curr, idx);
            Literal *lit_prev = grid_field_get(sim->u_prev, idx);
            double u_curr = lit_curr->field[0];
            double u_prev = lit_prev->field[0];
            literal_free(lit_curr);
            literal_free(lit_prev);
            
            // Kinetic energy: 1/2 * v²
            double velocity = (u_curr - u_prev) / dt;
            double kinetic = 0.5 * velocity * velocity;
            
            // Potential energy: 1/2 * c² * |∇u|²
            // Compute gradient using central/one-sided differences
            double grad_x = 0.0;
            double grad_y = 0.0;
            
            if (ix > start_x && ix < end_x - 1) {
                // Central difference (most accurate)
                uint32_t idx_xp[3] = {(uint32_t)(ix + 1), (uint32_t)iy, 0};
                uint32_t idx_xm[3] = {(uint32_t)(ix - 1), (uint32_t)iy, 0};
                Literal *lit_xp = grid_field_get(sim->u_curr, idx_xp);
                Literal *lit_xm = grid_field_get(sim->u_curr, idx_xm);
                grad_x = (lit_xp->field[0] - lit_xm->field[0]) / (2.0 * sim->dx);
                literal_free(lit_xp);
                literal_free(lit_xm);
            } else if (ix == start_x) {
                // One-sided forward difference at left boundary
                uint32_t idx_xp[3] = {(uint32_t)(ix + 1), (uint32_t)iy, 0};
                Literal *lit_curr = grid_field_get(sim->u_curr, idx);
                Literal *lit_xp = grid_field_get(sim->u_curr, idx_xp);
                grad_x = (lit_xp->field[0] - lit_curr->field[0]) / sim->dx;
                literal_free(lit_curr);
                literal_free(lit_xp);
            } else if (ix == end_x - 1) {
                // One-sided backward difference at right boundary
                uint32_t idx_xm[3] = {(uint32_t)(ix - 1), (uint32_t)iy, 0};
                Literal *lit_curr = grid_field_get(sim->u_curr, idx);
                Literal *lit_xm = grid_field_get(sim->u_curr, idx_xm);
                grad_x = (lit_curr->field[0] - lit_xm->field[0]) / sim->dx;
                literal_free(lit_curr);
                literal_free(lit_xm);
            }
            
            if (iy > start_y && iy < end_y - 1) {
                // Central difference (most accurate)
                uint32_t idx_yp[3] = {(uint32_t)ix, (uint32_t)(iy + 1), 0};
                uint32_t idx_ym[3] = {(uint32_t)ix, (uint32_t)(iy - 1), 0};
                Literal *lit_yp = grid_field_get(sim->u_curr, idx_yp);
                Literal *lit_ym = grid_field_get(sim->u_curr, idx_ym);
                grad_y = (lit_yp->field[0] - lit_ym->field[0]) / (2.0 * sim->dy);
                literal_free(lit_yp);
                literal_free(lit_ym);
            } else if (iy == start_y) {
                // One-sided forward difference at bottom boundary
                uint32_t idx_yp[3] = {(uint32_t)ix, (uint32_t)(iy + 1), 0};
                Literal *lit_curr = grid_field_get(sim->u_curr, idx);
                Literal *lit_yp = grid_field_get(sim->u_curr, idx_yp);
                grad_y = (lit_yp->field[0] - lit_curr->field[0]) / sim->dy;
                literal_free(lit_curr);
                literal_free(lit_yp);
            } else if (iy == end_y - 1) {
                // One-sided backward difference at top boundary
                uint32_t idx_ym[3] = {(uint32_t)ix, (uint32_t)(iy - 1), 0};
                Literal *lit_curr = grid_field_get(sim->u_curr, idx);
                Literal *lit_ym = grid_field_get(sim->u_curr, idx_ym);
                grad_y = (lit_curr->field[0] - lit_ym->field[0]) / sim->dy;
                literal_free(lit_curr);
                literal_free(lit_ym);
            }
            
            double grad_sq = grad_x * grad_x + grad_y * grad_y;
            double potential = 0.5 * c_squared * grad_sq;
            
            // Add to total energy
            total += (kinetic + potential) * dV;
        }
    }
    
    return total;
}

// Calculate interaction energy between new Gaussian wave and existing field
// E_interaction = c² ∫∫ ∇u_old · ∇u_new dV
double calculate_interaction_energy(SimulationState *sim, double phys_x, double phys_y, 
                                   double amplitude, double sigma) {
    double c_squared = sim->wave_speed * sim->wave_speed;
    double dV = sim->dx * sim->dy;
    double sigma_squared = sigma * sigma;
    
    double offset_x = sim->sponge_width * sim->dx;
    double offset_y = sim->sponge_width * sim->dy;
    
    // Center of new Gaussian in full grid coordinates
    double center_x = phys_x + offset_x;
    double center_y = phys_y + offset_y;
    
    double interaction = 0.0;
    
    // Compute over visible region only
    int start_x = sim->sponge_width;
    int end_x = sim->nx_total - sim->sponge_width;
    int start_y = sim->sponge_width;
    int end_y = sim->ny_total - sim->sponge_width;
    
    #pragma omp parallel for collapse(2) reduction(+:interaction)
    for (int iy = start_y; iy < end_y; iy++) {
        for (int ix = start_x; ix < end_x; ix++) {
            uint32_t idx[3] = {(uint32_t)ix, (uint32_t)iy, 0};
            
            // Grid position
            double x = ix * sim->dx;
            double y = iy * sim->dy;
            
            // Vector from Gaussian center to this point
            double dx_from_center = x - center_x;
            double dy_from_center = y - center_y;
            double r_squared = dx_from_center * dx_from_center + dy_from_center * dy_from_center;
            
            // Gradient of old field (numerical - central differences)
            double grad_u_old_x = 0.0;
            double grad_u_old_y = 0.0;
            
            if (ix > start_x && ix < end_x - 1) {
                uint32_t idx_xp[3] = {(uint32_t)(ix + 1), (uint32_t)iy, 0};
                uint32_t idx_xm[3] = {(uint32_t)(ix - 1), (uint32_t)iy, 0};
                Literal *lit_xp = grid_field_get(sim->u_curr, idx_xp);
                Literal *lit_xm = grid_field_get(sim->u_curr, idx_xm);
                grad_u_old_x = (lit_xp->field[0] - lit_xm->field[0]) / (2.0 * sim->dx);
                literal_free(lit_xp);
                literal_free(lit_xm);
            } else if (ix == start_x) {
                uint32_t idx_xp[3] = {(uint32_t)(ix + 1), (uint32_t)iy, 0};
                Literal *lit_curr = grid_field_get(sim->u_curr, idx);
                Literal *lit_xp = grid_field_get(sim->u_curr, idx_xp);
                grad_u_old_x = (lit_xp->field[0] - lit_curr->field[0]) / sim->dx;
                literal_free(lit_curr);
                literal_free(lit_xp);
            } else if (ix == end_x - 1) {
                uint32_t idx_xm[3] = {(uint32_t)(ix - 1), (uint32_t)iy, 0};
                Literal *lit_curr = grid_field_get(sim->u_curr, idx);
                Literal *lit_xm = grid_field_get(sim->u_curr, idx_xm);
                grad_u_old_x = (lit_curr->field[0] - lit_xm->field[0]) / sim->dx;
                literal_free(lit_curr);
                literal_free(lit_xm);
            }
            
            if (iy > start_y && iy < end_y - 1) {
                uint32_t idx_yp[3] = {(uint32_t)ix, (uint32_t)(iy + 1), 0};
                uint32_t idx_ym[3] = {(uint32_t)ix, (uint32_t)(iy - 1), 0};
                Literal *lit_yp = grid_field_get(sim->u_curr, idx_yp);
                Literal *lit_ym = grid_field_get(sim->u_curr, idx_ym);
                grad_u_old_y = (lit_yp->field[0] - lit_ym->field[0]) / (2.0 * sim->dy);
                literal_free(lit_yp);
                literal_free(lit_ym);
            } else if (iy == start_y) {
                uint32_t idx_yp[3] = {(uint32_t)ix, (uint32_t)(iy + 1), 0};
                Literal *lit_curr = grid_field_get(sim->u_curr, idx);
                Literal *lit_yp = grid_field_get(sim->u_curr, idx_yp);
                grad_u_old_y = (lit_yp->field[0] - lit_curr->field[0]) / sim->dy;
                literal_free(lit_curr);
                literal_free(lit_yp);
            } else if (iy == end_y - 1) {
                uint32_t idx_ym[3] = {(uint32_t)ix, (uint32_t)(iy - 1), 0};
                Literal *lit_curr = grid_field_get(sim->u_curr, idx);
                Literal *lit_ym = grid_field_get(sim->u_curr, idx_ym);
                grad_u_old_y = (lit_curr->field[0] - lit_ym->field[0]) / sim->dy;
                literal_free(lit_curr);
                literal_free(lit_ym);
            }
            
            // Gradient of new Gaussian (analytical)
            // ∇u_new = -(A/σ²) * r_vec * exp(-r²/(2σ²))
            double exp_term = exp(-r_squared / (2.0 * sigma_squared));
            double grad_u_new_x = -(amplitude / sigma_squared) * dx_from_center * exp_term;
            double grad_u_new_y = -(amplitude / sigma_squared) * dy_from_center * exp_term;
            
            // Dot product: ∇u_old · ∇u_new
            double dot_product = grad_u_old_x * grad_u_new_x + grad_u_old_y * grad_u_new_y;
            
            // Accumulate interaction energy
            interaction += dot_product * dV;
        }
    }
    
    // Multiply by c²
    return c_squared * interaction;
}

// Estimate self-energy of a Gaussian wave pulse
double estimate_wave_energy(SimulationState *sim, double amplitude, double sigma) {
    // For a Gaussian pulse: u(x,y) = A * exp(-(x²+y²)/(2σ²))
    // Self-energy (analytical): E_self = (π/2) * c² * A²
    // This is independent of σ for a fixed amplitude A
    double c_squared = sim->wave_speed * sim->wave_speed;
    double E_self = (M_PI / 2.0) * c_squared * amplitude * amplitude;
    
    return E_self;
}

// =============================================================================
// Simulation Step
// =============================================================================

void simulation_step(SimulationState *sim) {
    if (sim->paused) return;
    
    // Apply wave sources - compute source contributions and mask onto u_curr and u_prev
    // First, zero out the source field
    Literal *zero_lit = literal_create_scalar(0.0);
    grid_field_fill(sim->u_source, zero_lit);
    literal_free(zero_lit);
    
    // Compute source contributions
    double offset_x = sim->sponge_width * sim->dx;
    double offset_y = sim->sponge_width * sim->dy;
    
    for (int src_idx = 0; src_idx < n_wave_sources; src_idx++) {
        if (!wave_sources[src_idx].active) continue;
        
        WaveSource *src = &wave_sources[src_idx];
        
        // Convert source position to full grid coordinates
        double src_x_full = src->x + offset_x;
        double src_y_full = src->y + offset_y;
        
        // Source value: amplitude * sin(2π * frequency * time + phase)
        double omega = 2.0 * M_PI * src->frequency;
        double source_value = src->amplitude * sin(omega * sim->sim_time + src->phase);
        
        // Find grid cell range to iterate over (square around source)
        int radius_cells = (int)ceil(src->radius / fmin(sim->dx, sim->dy)) + 1;
        int src_ix = (int)round(src_x_full / sim->dx);
        int src_iy = (int)round(src_y_full / sim->dy);
        
        int start_ix = fmax(0, src_ix - radius_cells);
        int end_ix = fmin(sim->nx_total - 1, src_ix + radius_cells);
        int start_iy = fmax(0, src_iy - radius_cells);
        int end_iy = fmin(sim->ny_total - 1, src_iy + radius_cells);
        
        // Apply source within radius
        for (int iy = start_iy; iy <= end_iy; iy++) {
            for (int ix = start_ix; ix <= end_ix; ix++) {
                // Check distance from source center
                double x = ix * sim->dx;
                double y = iy * sim->dy;
                double dx = x - src_x_full;
                double dy = y - src_y_full;
                double dist = (dx * dx + dy * dy);

                if (dist <= src->radius * src->radius) {
                    // Within source radius - set to source value
                    uint32_t idx[3] = {(uint32_t)ix, (uint32_t)iy, 0};
                    Literal *src_val = literal_create_scalar(source_value);
                    grid_field_set(sim->u_source, idx, src_val);
                    literal_free(src_val);
                }
            }
        }
    }
    
    // Mask source contributions onto u_curr and u_prev (overwrite values within source radii)
    for (int src_idx = 0; src_idx < n_wave_sources; src_idx++) {
        if (!wave_sources[src_idx].active) continue;
        
        WaveSource *src = &wave_sources[src_idx];
        
        double src_x_full = src->x + offset_x;
        double src_y_full = src->y + offset_y;
        
        int radius_cells = (int)ceil(src->radius / fmin(sim->dx, sim->dy)) + 1;
        int src_ix = (int)round(src_x_full / sim->dx);
        int src_iy = (int)round(src_y_full / sim->dy);
        
        int start_ix = fmax(0, src_ix - radius_cells);
        int end_ix = fmin(sim->nx_total - 1, src_ix + radius_cells);
        int start_iy = fmax(0, src_iy - radius_cells);
        int end_iy = fmin(sim->ny_total - 1, src_iy + radius_cells);
        
        for (int iy = start_iy; iy <= end_iy; iy++) {
            for (int ix = start_ix; ix <= end_ix; ix++) {
                double x = ix * sim->dx;
                double y = iy * sim->dy;
                double dx = x - src_x_full;
                double dy = y - src_y_full;
                double dist = (dx * dx + dy * dy);

                if (dist <= src->radius*src->radius) {
                    uint32_t idx[3] = {(uint32_t)ix, (uint32_t)iy, 0};
                    
                    // Get source value
                    Literal *src_lit = grid_field_get(sim->u_source, idx);
                    double src_val = src_lit ? src_lit->field[0] : 0.0;
                    if (src_lit) literal_free(src_lit);
                    
                    // Set u_next to source value (drives the oscillation)
                    // Don't touch u_curr or u_prev - let them evolve naturally
                    Literal *set_lit = literal_create_scalar(src_val);
                    grid_field_set(sim->u_curr, idx, set_lit);
                    literal_free(set_lit);
                }
            }
        }
    }
    

    // Update variable dictionary with current field values
    dict_set(sim->vars, "u_curr", &sim->u_curr->data);
    dict_set(sim->vars, "u_prev", &sim->u_prev->data);
    
    // Evaluate wave equation expression using grid-aware evaluation
    Literal *result_lit = expression_evaluate_grid(sim->wave_expr, sim->vars, sim->grid);
    if (!result_lit) return;
    
    // Wrap result as GridField
    GridField *result = grid_field_wrap_literal(result_lit, sim->grid);
    
    // Replace u_next with result
    grid_field_free(sim->u_next);
    sim->u_next = result;
        
    // Apply sponge layer damping
    #pragma omp parallel for collapse(2)
    for (uint32_t iy = 0; iy < sim->ny_total; iy++) {
        for (uint32_t ix = 0; ix < sim->nx_total; ix++) {
            uint32_t idx[3] = {ix, iy, 0};
            
            // Distance from sponge boundary
            int dist_x_min = ix;
            int dist_x_max = sim->nx_total - 1 - ix;
            int dist_y_min = iy;
            int dist_y_max = sim->ny_total - 1 - iy;
            int min_dist = dist_x_min;
            if (dist_x_max < min_dist) min_dist = dist_x_max;
            if (dist_y_min < min_dist) min_dist = dist_y_min;
            if (dist_y_max < min_dist) min_dist = dist_y_max;
            
            if (min_dist < sim->sponge_width) {
                // Quadratic ramp: σ = σ_max * (1 - d/w)²
                double d_normalized = (double)min_dist / (double)sim->sponge_width;
                double sigma = SIGMA_MAX * (1.0 - d_normalized) * (1.0 - d_normalized);
                
                // Get current and previous values
                Literal *lit_c = grid_field_get(sim->u_curr, idx);
                Literal *lit_p = grid_field_get(sim->u_prev, idx);
                Literal *lit_n_old = grid_field_get(sim->u_next, idx);
                
                double u_curr_val = lit_c ? lit_c->field[0] : 0.0;
                double u_prev_val = lit_p ? lit_p->field[0] : 0.0;
                double u_next_old = lit_n_old ? lit_n_old->field[0] : 0.0;
                
                // Free the Literals returned by grid_field_get
                if (lit_c) literal_free(lit_c);
                if (lit_p) literal_free(lit_p);
                if (lit_n_old) literal_free(lit_n_old);
                
                // Compute acceleration from wave equation
                double accel = u_next_old - 2.0 * u_curr_val + u_prev_val;
                
                // Sponge layer formula: includes damping on velocity
                // u_next = (2 - σdt)*u_curr - (1 - σdt)*u_prev + accel
                double u_next_val = (2.0 - sigma * sim->dt) * u_curr_val 
                                  - (1.0 - sigma * sim->dt) * u_prev_val 
                                  + accel;
                
                Literal *new_val = literal_create_scalar(u_next_val);
                grid_field_set(sim->u_next, idx, new_val);
                literal_free(new_val);
            }
        }
    }
    
    
    // Cycle fields
    GridField *tmp = sim->u_prev;
    sim->u_prev = sim->u_curr;
    sim->u_curr = sim->u_next;
    sim->u_next = tmp;
    
    sim->sim_time += sim->dt;
    sim->step_count++;
    sim->steps_since_last_update++;
    
    // Calculate current energy and error
    sim->total_energy = calculate_total_energy(sim);
    if (sim->baseline_energy > 0.0) {
        sim->energy_error = (sim->total_energy - sim->baseline_energy) / sim->baseline_energy;
    } else {
        sim->energy_error = 0.0;
    }
}

// =============================================================================
// Rendering Functions
// =============================================================================

// Map value to RGB color based on render mode
void value_to_color(double height, double vx, double vy, 
                   RenderMode mode, double scale,
                   uint8_t *r, uint8_t *g, uint8_t *b) {
    switch (mode) {
        case RENDER_HEIGHT: {
            // Red for positive, blue for negative
            double val = height * scale;
            if (val > 0) {
                *r = (uint8_t)fmin(255, val * 255);
                *g = 0;
                *b = 0;
            } else {
                *r = 0;
                *g = 0;
                *b = (uint8_t)fmin(255, -val * 255);
            }
            break;
        }
        case RENDER_VELOCITY: {
            double vmag = sqrt(vx*vx + vy*vy) * scale;
            uint8_t gray = (uint8_t)fmin(255, vmag * 255);
            *r = *g = *b = gray;
            break;
        }
        case RENDER_RGB: {
            *r = (uint8_t)fmin(255, fmax(0, 127.5 + vx * scale * 127.5));
            *g = (uint8_t)fmin(255, fmax(0, 127.5 + vy * scale * 127.5));
            *b = (uint8_t)fmin(255, fmax(0, 127.5 + height * scale * 127.5));
            break;
        }
    }
}

// Render visible region of wave field
void render_wave_field(SimulationState *sim, RenderState *render, 
                      int win_width, int win_height) {
    // Render only visible region (skip sponge layer)
    int start_x = sim->sponge_width;
    int start_y = sim->sponge_width;
    int end_x = start_x + sim->nx_visible;
    int end_y = start_y + sim->ny_visible;
    
    // Calculate pixel size for each grid cell
    float cell_width = (float)win_width / sim->nx_visible;
    float cell_height = (float)win_height / sim->ny_visible;
    
    uint32_t idx[3] = {0, 0, 0};
    
    glBegin(GL_QUADS);
    for (int iy = start_y; iy < end_y; iy++) {
        for (int ix = start_x; ix < end_x; ix++) {
            idx[0] = ix;
            idx[1] = iy;
            
            Literal *lit_height = grid_field_get(sim->u_curr, idx);
            double height = lit_height ? lit_height->field[0] : 0.0;
            if (lit_height) literal_free(lit_height);
            
            // Calculate spatial velocity using centered finite differences
            // vx = ∂u/∂x ≈ (u[i+1,j] - u[i-1,j]) / (2*dx)
            // vy = ∂u/∂y ≈ (u[i,j+1] - u[i,j-1]) / (2*dy)
            double vx = 0.0, vy = 0.0;
            
            // Compute x-derivative (if not on x-boundary)
            if (ix > 0 && ix < sim->nx_total - 1) {
                uint32_t idx_left[3] = {ix - 1, iy, 0};
                uint32_t idx_right[3] = {ix + 1, iy, 0};
                Literal *lit_left = grid_field_get(sim->u_curr, idx_left);
                Literal *lit_right = grid_field_get(sim->u_curr, idx_right);
                double u_left = lit_left ? lit_left->field[0] : 0.0;
                double u_right = lit_right ? lit_right->field[0] : 0.0;
                vx = (u_right - u_left) / (2.0 * sim->dx);
                if (lit_left) literal_free(lit_left);
                if (lit_right) literal_free(lit_right);
            }
            
            // Compute y-derivative (if not on y-boundary)
            if (iy > 0 && iy < sim->ny_total - 1) {
                uint32_t idx_down[3] = {ix, iy - 1, 0};
                uint32_t idx_up[3] = {ix, iy + 1, 0};
                Literal *lit_down = grid_field_get(sim->u_curr, idx_down);
                Literal *lit_up = grid_field_get(sim->u_curr, idx_up);
                double u_down = lit_down ? lit_down->field[0] : 0.0;
                double u_up = lit_up ? lit_up->field[0] : 0.0;
                vy = (u_up - u_down) / (2.0 * sim->dy);
                if (lit_down) literal_free(lit_down);
                if (lit_up) literal_free(lit_up);
            }
            
            uint8_t r = 0, g = 0, b = 0;
            value_to_color(height, vx, vy, render->mode, render->value_scale, &r, &g, &b);
            
            // Grid coordinates (visible region)
            int gx = ix - start_x;
            int gy = iy - start_y;
            
            // Screen coordinates (flip y for OpenGL)
            float x0 = gx * cell_width;
            float y0 = (sim->ny_visible - 1 - gy) * cell_height;
            float x1 = x0 + cell_width;
            float y1 = y0 + cell_height;
            
            glColor3ub(r, g, b);
            glVertex2f(x0, y0);
            glVertex2f(x1, y0);
            glVertex2f(x1, y1);
            glVertex2f(x0, y1);
        }
    }
    glEnd();
    
    // Draw interior boundaries if enabled
    if (render->show_boundaries && sim->grid->n_interior_boundaries > 0) {
        glLineWidth(3.0f);
        
        for (int i = 0; i < sim->grid->n_interior_boundaries; i++) {
            HyperplaneBoundary *hb = &sim->grid->interior_boundaries[i];
            if (!hb->active) continue;
            
            // Reconstruct line segment from parametric bounds
            // Reference point is at the midpoint (where t=0)
            // Tangent direction is perpendicular to normal: tangent = (normal[1], -normal[0])
            // Start point is at hb->point + bounds_min[0] * tangent
            // End point is at hb->point + bounds_max[0] * tangent
            
            double tangent[2] = {hb->normal[1], -hb->normal[0]};
            
            // Start point (t=bounds_min, typically -length/2)
            double x1 = hb->point[0] + hb->bounds_min[0] * tangent[0];
            double y1 = hb->point[1] + hb->bounds_min[0] * tangent[1];
            
            // End point (t=bounds_max, typically +length/2)
            double x2 = hb->point[0] + hb->bounds_max[0] * tangent[0];
            double y2 = hb->point[1] + hb->bounds_max[0] * tangent[1];
            
            // Convert to visible coordinates (subtract sponge offset)
            double offset_x = sim->sponge_width * sim->dx;
            double offset_y = sim->sponge_width * sim->dy;
            double vis_x1 = x1 - offset_x;
            double vis_y1 = y1 - offset_y;
            double vis_x2 = x2 - offset_x;
            double vis_y2 = y2 - offset_y;
            
            // Convert to screen coordinates
            float sx1 = (vis_x1 / sim->Lx_visible) * win_width;
            float sy1 = (1.0 - vis_y1 / sim->Ly_visible) * win_height;
            float sx2 = (vis_x2 / sim->Lx_visible) * win_width;
            float sy2 = (1.0 - vis_y2 / sim->Ly_visible) * win_height;
            
            // Draw the line segment in white
            glBegin(GL_LINES);
            glColor3f(1.0f, 1.0f, 1.0f); // White
            glVertex2f(sx1, sy1);
            glVertex2f(sx2, sy2);
            glEnd();
            
            // Draw endpoints as small circles (also white)
            glPointSize(6.0f);
            glBegin(GL_POINTS);
            glColor3f(1.0f, 1.0f, 1.0f); // White
            glVertex2f(sx1, sy1);
            glVertex2f(sx2, sy2);
            glEnd();
        }
    }
}

// Render wave sources as yellow circles
void render_wave_sources(SimulationState *sim, int win_width, int win_height) {
    glLineWidth(2.0f);
    
    for (int i = 0; i < n_wave_sources; i++) {
        if (!wave_sources[i].active) continue;
        
        // Convert physical coordinates to screen coordinates
        float sx = (wave_sources[i].x / sim->Lx_visible) * win_width;
        float sy = (1.0 - wave_sources[i].y / sim->Ly_visible) * win_height;
        
        // Convert radius to screen units (average of x and y scaling)
        float radius_screen = wave_sources[i].radius * ((win_width / sim->Lx_visible) + 
                                                         (win_height / sim->Ly_visible)) / 2.0;
        
        // Draw circle as line loop
        int segments = 32;
        glBegin(GL_LINE_LOOP);
        
        // Selected source: brighter yellow with thicker line
        if (&wave_sources[i] == selected_source) {
            glColor3f(1.0f, 1.0f, 0.0f);  // Bright yellow
            glLineWidth(3.0f);
        } else {
            glColor3f(0.8f, 0.8f, 0.0f);  // Dim yellow
            glLineWidth(2.0f);
        }
        
        for (int j = 0; j < segments; j++) {
            float angle = 2.0f * M_PI * j / segments;
            float x = sx + radius_screen * cosf(angle);
            float y = sy + radius_screen * sinf(angle);
            glVertex2f(x, y);
        }
        glEnd();
        
        // Draw center point
        glPointSize(selected_source == &wave_sources[i] ? 8.0f : 5.0f);
        glBegin(GL_POINTS);
        glVertex2f(sx, sy);
        glEnd();
    }
    
    glLineWidth(1.0f);  // Reset
}

// =============================================================================
// Mouse Interaction Functions
// =============================================================================

// Snap physical coordinates to cell center (visible region)
void snap_to_cell_center(SimulationState *sim, double *phys_x, double *phys_y) {
    // Find nearest cell center
    int ix = (int)round(*phys_x / sim->dx);
    int iy = (int)round(*phys_y / sim->dy);
    
    // Clamp to visible region (in cell indices)
    if (ix < 0) ix = 0;
    if (ix >= (int)sim->nx_visible) ix = sim->nx_visible - 1;
    if (iy < 0) iy = 0;
    if (iy >= (int)sim->ny_visible) iy = sim->ny_visible - 1;
    
    // Convert back to physical coordinates (cell centers)
    *phys_x = ix * sim->dx;
    *phys_y = iy * sim->dy;
}

// Find barrier point near the given coordinates (returns index or -1)
// Returns the LAST (highest index) point within tolerance when multiple points overlap
// This ensures we work on the most recently added barrier
int find_barrier_point_at(double phys_x, double phys_y, double tolerance) {
    int closest_idx = -1;
    double min_dist = tolerance;
    
    // Iterate through all points and find the closest
    // When multiple points are equally close (overlapping), prefer the last one
    for (int i = 0; i < n_barrier_points; i++) {
        double dx = phys_x - barrier_points[i].x;
        double dy = phys_y - barrier_points[i].y;
        double dist = sqrt(dx * dx + dy * dy);
        if (dist <= min_dist) {  // Use <= instead of < to prefer later indices
            min_dist = dist;
            closest_idx = i;
        }
    }
    return closest_idx;
}

// Find wave source at given coordinates (returns pointer or NULL)
WaveSource* find_source_at(double phys_x, double phys_y) {
    WaveSource *found_source = NULL;
    double min_dist = INFINITY;
    
    for (int i = 0; i < n_wave_sources; i++) {
        if (!wave_sources[i].active) continue;
        
        double dx = phys_x - wave_sources[i].x;
        double dy = phys_y - wave_sources[i].y;
        double dist = sqrt(dx * dx + dy * dy);
        
        // Check if within source radius and closest
        if (dist <= wave_sources[i].radius && dist < min_dist) {
            found_source = &wave_sources[i];
            min_dist = dist;
        }
    }
    
    return found_source;
}

// Find interior boundary index that uses the given barrier point
// This properly handles the case where barriers have been deleted and indices no longer match
int find_boundary_using_point(SimulationState *sim, int point_idx) {
    if (point_idx < 0 || point_idx >= n_barrier_points) return -1;
    
    // Points are stored in pairs: (0,1) -> boundary 0, (2,3) -> boundary 1, etc.
    // Calculate which pair this point belongs to
    int pair_idx = point_idx / 2;
    
    // Verify that both points in the pair exist and that a boundary exists for this pair
    int pt2_idx = pair_idx * 2 + 1;
    
    if (pt2_idx < n_barrier_points && pair_idx < sim->grid->n_interior_boundaries) {
        return pair_idx;
    }
    
    return -1;
}

// Add Gaussian wave centered at physical coordinates
void add_wave_at_position(SimulationState *sim, double phys_x, double phys_y) {
    double offset_x = sim->sponge_width * sim->dx;
    double offset_y = sim->sponge_width * sim->dy;

    
    
    // Update baseline energy with self-energy + interaction energy
    double E_self = estimate_wave_energy(sim, sim->wave_amplitude, sim->wave_spread);
    double E_interaction = calculate_interaction_energy(sim, phys_x, phys_y, 
                                                        sim->wave_amplitude, sim->wave_spread);
    
    for (uint32_t iy = 0; iy < sim->ny_total; iy++) {
        for (uint32_t ix = 0; ix < sim->nx_total; ix++) {
            uint32_t idx[3] = {ix, iy, 0};
            
            // Get physical coordinates of this grid point
            double x = ix * sim->dx;
            double y = iy * sim->dy;
            
            // Distance from click point
            double dx = x - (phys_x + offset_x);
            double dy = y - (phys_y + offset_y);
            double r2 = dx * dx + dy * dy;
            
            // Gaussian wave
            double wave_val = sim->wave_amplitude * exp(-r2 / (2.0 * sim->wave_spread * sim->wave_spread));
            
            // Add to current field
            Literal *lit_curr = grid_field_get(sim->u_curr, idx);
            double curr_val = lit_curr ? lit_curr->field[0] : 0.0;
            if (lit_curr) literal_free(lit_curr);

            // Add to prev field
            Literal *lit_prev = grid_field_get(sim->u_prev, idx);
            double prev_val = lit_prev ? lit_prev->field[0] : 0.0;
            if (lit_prev) literal_free(lit_prev);
            
            Literal *new_val_curr = literal_create_scalar(curr_val + wave_val);
            grid_field_set(sim->u_curr, idx, new_val_curr);
            Literal *new_val_prev = literal_create_scalar(prev_val + wave_val);
            // grid_field_set(sim->u_prev, idx, new_val_prev);
            literal_free(new_val_curr);
            literal_free(new_val_prev);
        }
    }
    double wave_energy = E_self + E_interaction;
    sim->baseline_energy += wave_energy;
    
    printf("Added wave at (%.3f, %.3f) with amplitude %.2f, spread %.3f\n", 
           phys_x, phys_y, sim->wave_amplitude, sim->wave_spread);
    printf("  Self-energy: %.6f | Interaction: %.6f | Total added: %.6f\n",
           E_self, E_interaction, wave_energy);
    printf("  New baseline: %.6f\n", sim->baseline_energy);
}

// Add barrier segment between two points
void add_barrier_segment(SimulationState *sim, double x1, double y1, double x2, double y2) {
    // Convert visible coordinates to full grid physical coordinates (add sponge offset)
    double offset_x = sim->sponge_width * sim->dx;
    double offset_y = sim->sponge_width * sim->dy;
    
    double phys_x1 = x1 + offset_x;
    double phys_y1 = y1 + offset_y;
    double phys_x2 = x2 + offset_x;
    double phys_y2 = y2 + offset_y;
    
    // Calculate direction vector
    double dx = phys_x2 - phys_x1;
    double dy = phys_y2 - phys_y1;
    double length = sqrt(dx * dx + dy * dy);
    if (length < 1e-6) return;  // Degenerate segment
    
    // Normal vector (perpendicular to line, pointing "right" of direction)
    double normal[2] = {-dy / length, dx / length};
    
    // Use MIDPOINT of segment as reference point
    // This makes the parametric bounds symmetric and independent of basis orientation
    double point[2] = {(phys_x1 + phys_x2) / 2.0, (phys_y1 + phys_y2) / 2.0};
    
    // Use SYMMETRIC parametric coordinates: -length/2 to +length/2
    // This ensures the barrier works correctly regardless of which direction
    // the basis vectors point (they're constructed arbitrarily by Gram-Schmidt)
    double bounds_min[1] = {-length / 2.0};
    double bounds_max[1] = {length / 2.0};
    
    // Add reflecting hyperplane boundary
    int idx = grid_add_hyperplane_boundary(sim->grid, normal, point, 
                                           bounds_min, bounds_max, 
                                           BC_REFLECT, 1.0);
    if (idx >= 0) {
        HyperplaneBoundary *hb = &sim->grid->interior_boundaries[idx];
        printf("Added barrier #%d: (%.3f, %.3f) to (%.3f, %.3f) [physical coords]\n", 
               idx, phys_x1, phys_y1, phys_x2, phys_y2);
        printf("  Normal: (%.3f, %.3f), Length: %.3f\n", normal[0], normal[1], length);
        printf("  BBox: x[%.3f, %.3f], y[%.3f, %.3f]\n",
               hb->bbox_min[0], hb->bbox_max[0], hb->bbox_min[1], hb->bbox_max[1]);
    } else {
        printf("Failed to add barrier segment\n");
    }
}

// =============================================================================
// Menu System 
// =============================================================================
// NOTE: Menu system included from separate file
// It contains: Menus struct, all callback functions, initialize_menus(), free_menus()
#include "interactive_wave_sim_menu.inc"

// =============================================================================
// Main Program
// =============================================================================

int main(int argc, char **argv) {
    printf("========================================\n");
    printf("Interactive 2D Wave Simulator\n");
    printf("========================================\n\n");
    
    // Parse arguments: Lx Ly nx ny
    double Lx_vis = 2.0;
    double Ly_vis = 2.0;
    int nx_vis = 120;
    int ny_vis = 120;
    
    if (argc >= 3) {
        Lx_vis = atof(argv[1]);
        Ly_vis = atof(argv[2]);
    }
    if (argc >= 5) {
        nx_vis = atoi(argv[3]);
        ny_vis = atoi(argv[4]);
    }
    
    if (nx_vis < 10 || ny_vis < 10) {
        printf("Error: Grid size must be at least 10x10\n");
        return 1;
    }
    
    // Create simulation
    SimulationState *sim = simulation_create(Lx_vis, Ly_vis, nx_vis, ny_vis);
    g_sim = sim;
    
    // Create render state
    RenderState render;
    render.mode = RENDER_HEIGHT;
    render.value_scale = 1.0;
    render.show_boundaries = 1;
    render.show_stats = 1;
    // Initialize radio button states (HEIGHT is default)
    render.mode_height = 1;
    render.mode_velocity = 0;
    render.mode_rgb = 0;
    g_render = &render;
    
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    
    // Create window
    int window_width = 800;
    int window_height = 800;
    SDL_Window *window = SDL_CreateWindow(
        "Interactive Wave Simulator",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        window_width, window_height,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );
    
    if (!window) {
        printf("SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    
    // Create OpenGL context
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        printf("SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    
    SDL_GL_SetSwapInterval(1); // Enable vsync
    
    // Initialize TTF for menu
    if (menu_set_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14) != 0) {
        printf("Warning: Could not load font, menus may not render properly\n");
    }
    
    // Create menus
    Menus *menus = initialize_menus(sim, &render);
    
    printf("\nStarting main loop...\n");
    printf("Controls:\n");
    printf("  - ESC to quit\n");
    printf("  - Space to pause/resume\n");
    printf("  - Tab to toggle main menu\n");
    printf("  - R to reset simulation\n");
    printf("  - Mouse wheel to cycle mouse mode\n\n");
    
    // Main loop
    int running = 1;
    SDL_Event event;
    Uint32 last_frame_time = SDL_GetTicks();
    Uint32 last_stats_time = SDL_GetTicks();
    
    while (running) {
        // Event handling
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = 0;
            } else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = 0;
                } else if (event.key.keysym.sym == SDLK_SPACE) {
                    sim->paused = !sim->paused;
                    printf("Simulation %s\n", sim->paused ? "paused" : "resumed");
                } else if (event.key.keysym.sym == SDLK_TAB) {
                    g_show_base_menu = !g_show_base_menu;
                    printf("Menu %s\n", g_show_base_menu ? "opened" : "closed");
                } else if (event.key.keysym.sym == SDLK_r) {
                    // Reset simulation
                    on_reset_clicked(NULL, sim);
                }
            } else if (event.type == SDL_MOUSEWHEEL) {
                // Cycle mouse mode with scroll wheel
                // Clear unpaired barrier points only
                if (n_barrier_points % 2 == 1) {
                    n_barrier_points--;
                }
                
                // Reset dragging state
                dragging_barrier_point = -1;
                dragging_barrier_idx = -1;
                barrier_click_candidate = -1;
                
                int direction = event.wheel.y;  // Positive = scroll up, Negative = scroll down
                if (direction > 0) {
                    // Scroll up: next mode
                    g_mouse_mode = (MouseMode)((g_mouse_mode + 1) % 4);
                } else if (direction < 0) {
                    // Scroll down: previous mode
                    g_mouse_mode = (MouseMode)((g_mouse_mode + 3) % 4);  // +3 mod 4 = -1 mod 4
                }
                
                // Update radio button states
                g_mouse_none = (g_mouse_mode == MOUSE_NONE) ? 1 : 0;
                g_mouse_add_wave = (g_mouse_mode == MOUSE_ADD_WAVE) ? 1 : 0;
                g_mouse_add_barrier = (g_mouse_mode == MOUSE_ADD_BARRIER) ? 1 : 0;
                g_mouse_source = (g_mouse_mode == MOUSE_SOURCE) ? 1 : 0;
                
                printf("Mouse mode: %s\n", g_mouse_mode_names[g_mouse_mode]);
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                int mx = event.button.x;
                int my = event.button.y;
                
                if (event.button.button == SDL_BUTTON_LEFT) {
                    // Centralized menu and interaction handling
                    int handled = 0;
                    
                    // Try each menu in order of visibility
                    if (g_show_base_menu && !handled) {
                        VariableInteraction *clicked = menu_get_interaction_at(menus->base_menu, mx, my);
                        if (clicked) {
                            if (is_slider_interaction(clicked)) {
                                menu_begin_slider_drag(menus->base_menu, clicked, mx);
                            } else {
                                menu_handle_mouse_button(menus->base_menu, SDL_BUTTON_LEFT, SDL_PRESSED, mx, my);
                            }
                            handled = 1;
                        } else if (mx >= menus->base_menu->x && mx <= menus->base_menu->x + menus->base_menu->width &&
                                   my >= menus->base_menu->y && my <= menus->base_menu->y + menus->base_menu->height) {
                            // Click is on menu but not on interaction - still consume it
                            menu_handle_mouse_button(menus->base_menu, SDL_BUTTON_LEFT, SDL_PRESSED, mx, my);
                            handled = 1;
                        }
                    }
                    
                    if (g_show_mouse_controls && !handled) {
                        VariableInteraction *clicked = menu_get_interaction_at(menus->mouse_menu, mx, my);
                        if (clicked) {
                            if (is_slider_interaction(clicked)) {
                                menu_begin_slider_drag(menus->mouse_menu, clicked, mx);
                            } else {
                                menu_handle_mouse_button(menus->mouse_menu, SDL_BUTTON_LEFT, SDL_PRESSED, mx, my);
                            }
                            handled = 1;
                        } else if (mx >= menus->mouse_menu->x && mx <= menus->mouse_menu->x + menus->mouse_menu->width &&
                                   my >= menus->mouse_menu->y && my <= menus->mouse_menu->y + menus->mouse_menu->height) {
                            menu_handle_mouse_button(menus->mouse_menu, SDL_BUTTON_LEFT, SDL_PRESSED, mx, my);
                            handled = 1;
                        }
                    }
                    
                    if (g_show_sim_controls && !handled) {
                        VariableInteraction *clicked = menu_get_interaction_at(menus->sim_menu, mx, my);
                        if (clicked) {
                            if (is_slider_interaction(clicked)) {
                                menu_begin_slider_drag(menus->sim_menu, clicked, mx);
                            } else {
                                menu_handle_mouse_button(menus->sim_menu, SDL_BUTTON_LEFT, SDL_PRESSED, mx, my);
                            }
                            handled = 1;
                        } else if (mx >= menus->sim_menu->x && mx <= menus->sim_menu->x + menus->sim_menu->width &&
                                   my >= menus->sim_menu->y && my <= menus->sim_menu->y + menus->sim_menu->height) {
                            menu_handle_mouse_button(menus->sim_menu, SDL_BUTTON_LEFT, SDL_PRESSED, mx, my);
                            handled = 1;
                        }
                    }
                    
                    if (selected_source && selected_source->active && menus->source_menu && !handled) {
                        VariableInteraction *clicked = menu_get_interaction_at(menus->source_menu, mx, my);
                        if (clicked) {
                            if (is_slider_interaction(clicked)) {
                                menu_begin_slider_drag(menus->source_menu, clicked, mx);
                            } else {
                                menu_handle_mouse_button(menus->source_menu, SDL_BUTTON_LEFT, SDL_PRESSED, mx, my);
                            }
                            handled = 1;
                        } else if (mx >= menus->source_menu->x && mx <= menus->source_menu->x + menus->source_menu->width &&
                                   my >= menus->source_menu->y && my <= menus->source_menu->y + menus->source_menu->height) {
                            menu_handle_mouse_button(menus->source_menu, SDL_BUTTON_LEFT, SDL_PRESSED, mx, my);
                            handled = 1;
                        }
                    }
                    
                    // Only handle simulation clicks if no menu consumed click
                    if (!handled) {
                        // Handle mouse modes for simulation interaction
                        // Convert screen to physical coordinates (visible region only)
                        double phys_x = ((double)mx / window_width) * sim->Lx_visible;
                        double phys_y = (1.0 - (double)my / window_height) * sim->Ly_visible;
                        
                        if (g_mouse_mode == MOUSE_ADD_WAVE) {
                            add_wave_at_position(sim, phys_x, phys_y);
                            // Start wave painting mode
                            wave_painting = 1;
                            last_wave_x = phys_x;
                            last_wave_y = phys_y;
                        } else if (g_mouse_mode == MOUSE_ADD_BARRIER) {
                            // Snap to cell center
                            snap_to_cell_center(sim, &phys_x, &phys_y);
                            
                            // Check if clicking on an existing point (for potential delete or drag)
                            double click_tolerance = 0.02;  // Physical units
                            int clicked_point = find_barrier_point_at(phys_x, phys_y, click_tolerance);
                            
                            if (clicked_point >= 0) {
                                // Store this as a potential deletion candidate
                                // We'll only delete on mouse up if it wasn't dragged
                                barrier_click_candidate = clicked_point;
                                barrier_click_start_x = phys_x;
                                barrier_click_start_y = phys_y;
                                
                                // Also set as the dragging point immediately to lock it in
                                // This prevents finding a different point during motion
                                dragging_barrier_point = clicked_point;
                                dragging_barrier_idx = find_boundary_using_point(sim, clicked_point);
                                
                                // Validate that dragging_barrier_idx is in range
                                if (dragging_barrier_idx >= sim->grid->n_interior_boundaries) {
                                    // Invalid boundary index - this point might be unpaired
                                    dragging_barrier_idx = -1;
                                    printf("Warning: Clicked point is not part of a complete barrier\n");
                                }
                            } else {
                                // Store this point
                                if (n_barrier_points < MAX_BARRIER_POINTS) {
                                    barrier_points[n_barrier_points].x = phys_x;
                                    barrier_points[n_barrier_points].y = phys_y;
                                    n_barrier_points++;
                                    
                                    // Create individual line segments: every 2 clicks makes one segment
                                    if (n_barrier_points % 2 == 0) {
                                        // We have a pair - create the segment
                                        int i = n_barrier_points - 1;
                                        add_barrier_segment(sim, 
                                                           barrier_points[i-1].x, barrier_points[i-1].y,
                                                           barrier_points[i].x, barrier_points[i].y);
                                    }
                                }
                            }
                        } else if (g_mouse_mode == MOUSE_SOURCE) {
                            // Snap to cell center for consistent placement
                            snap_to_cell_center(sim, &phys_x, &phys_y);
                            
                            // Check if clicking on existing source at snapped position
                            WaveSource *clicked_source = find_source_at(phys_x, phys_y);
                            
                            if (clicked_source) {
                                // Clicking on existing source
                                if (clicked_source == selected_source) {
                                    // Clicking selected source - store as candidate for deselect/drag
                                    source_click_candidate = clicked_source;
                                    source_click_start_x = phys_x;
                                    source_click_start_y = phys_y;
                                } else {
                                    // Select this source
                                    selected_source = clicked_source;
                                    source_click_candidate = NULL;
                                    int idx = clicked_source - wave_sources;
                                    printf("Selected source #%d at (%.3f, %.3f)\n", 
                                           idx, clicked_source->x, clicked_source->y);
                                }
                            } else {
                                // Clicking empty space - add new source
                                if (n_wave_sources < MAX_WAVE_SOURCES) {
                                    wave_sources[n_wave_sources].x = phys_x;
                                    wave_sources[n_wave_sources].y = phys_y;
                                    wave_sources[n_wave_sources].frequency = g_default_source_frequency;
                                    wave_sources[n_wave_sources].phase = g_default_source_phase;  // From mouse controls
                                    wave_sources[n_wave_sources].amplitude = sim->wave_amplitude;  // From mouse controls
                                    wave_sources[n_wave_sources].radius = sim->wave_spread;        // From mouse controls
                                    wave_sources[n_wave_sources].active = 1;
                                    
                                    // Select this new source
                                    selected_source = &wave_sources[n_wave_sources];
                                    n_wave_sources++;
                                    
                                    printf("Added source #%d at (%.3f, %.3f)\n", 
                                           n_wave_sources - 1, phys_x, phys_y);
                                } else {
                                    printf("Maximum number of sources (%d) reached\n", MAX_WAVE_SOURCES);
                                }
                            }
                        }
                    }
                }
            } else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    // End any active slider drag
                    menu_end_slider_drag();
                    
                    // End wave painting mode
                    if (wave_painting) {
                        wave_painting = 0;
                        last_wave_x = -1.0;
                        last_wave_y = -1.0;
                    }
                    
                    // Handle barrier point deletion/dragging end
                    if (g_mouse_mode == MOUSE_ADD_BARRIER && barrier_click_candidate >= 0) {
                        // Validate that the candidate index is still valid
                        if (barrier_click_candidate >= n_barrier_points) {
                            // Index is out of bounds - reset state
                            barrier_click_candidate = -1;
                            dragging_barrier_point = -1;
                            dragging_barrier_idx = -1;
                        } else {
                            // Check if this was a click (no significant drag) or a drag
                            int mx = event.button.x;
                            int my = event.button.y;
                            double phys_x = ((double)mx / window_width) * sim->Lx_visible;
                            double phys_y = (1.0 - (double)my / window_height) * sim->Ly_visible;
                            snap_to_cell_center(sim, &phys_x, &phys_y);
                            
                            double dx = phys_x - barrier_click_start_x;
                            double dy = phys_y - barrier_click_start_y;
                            double drag_distance = sqrt(dx * dx + dy * dy);
                            
                            double drag_threshold = 0.01;  // Physical units - if moved less than this, it's a click
                            
                            if (drag_distance < drag_threshold) {
                                // This was a click, not a drag - delete the barrier
                                int boundary_idx = find_boundary_using_point(sim, barrier_click_candidate);
                                
                                if (boundary_idx >= 0) {
                                    // Valid complete barrier - delete it
                                    // Calculate which points belong to this boundary
                                    int pt1_idx = boundary_idx * 2;
                                    int pt2_idx = boundary_idx * 2 + 1;
                                    
                                    // Double-check bounds
                                    if (pt2_idx < n_barrier_points) {
                                        // Remove this boundary by shifting all subsequent ones
                                        for (int i = boundary_idx; i < sim->grid->n_interior_boundaries - 1; i++) {
                                            sim->grid->interior_boundaries[i] = sim->grid->interior_boundaries[i + 1];
                                        }
                                        sim->grid->n_interior_boundaries--;
                                        
                                        // Remove the two points that made this barrier
                                        // Shift all subsequent points forward by 2
                                        for (int i = pt1_idx; i < n_barrier_points - 2; i++) {
                                            barrier_points[i] = barrier_points[i + 2];
                                        }
                                        n_barrier_points -= 2;
                                        
                                        printf("Deleted barrier #%d (points %d,%d)\n", boundary_idx, pt1_idx, pt2_idx);
                                    }
                                } else if (barrier_click_candidate == n_barrier_points - 1 && 
                                          n_barrier_points % 2 == 1) {
                                    // This is an unpaired point (odd number of points, and it's the last one)
                                    // Just remove this single point
                                    n_barrier_points--;
                                    printf("Removed unpaired barrier point\n");
                                } else {
                                    printf("Warning: Could not determine barrier to delete for point %d\n", barrier_click_candidate);
                                }
                            }
                            // If drag_distance >= threshold, it was a drag (already handled in motion)
                        }
                    }
                    
                    // Handle source click candidate (deselect if not dragged)
                    if (g_mouse_mode == MOUSE_SOURCE && source_click_candidate) {
                        // Check if this was a click (no significant drag) or a drag
                        int mx = event.button.x;
                        int my = event.button.y;
                        double phys_x = ((double)mx / window_width) * sim->Lx_visible;
                        double phys_y = (1.0 - (double)my / window_height) * sim->Ly_visible;
                        
                        double dx = phys_x - source_click_start_x;
                        double dy = phys_y - source_click_start_y;
                        double drag_distance = sqrt(dx * dx + dy * dy);
                        
                        // If drag distance is small, treat it as a click to deselect
                        if (drag_distance < 0.01) {  // Same threshold as barriers
                            selected_source = NULL;
                            printf("Deselected source\n");
                        }
                        // If drag_distance >= threshold, it was a drag (source already moved)
                    }
                    source_click_candidate = NULL;
                    
                    // End barrier point dragging
                    dragging_barrier_point = -1;
                    dragging_barrier_idx = -1;
                    barrier_click_candidate = -1;
                }
            } else if (event.type == SDL_MOUSEMOTION) {
                int mx = event.motion.x;
                int my = event.motion.y;
                
                // If dragging a slider, update it
                if (g_menu_drag_state.dragging_interaction) {
                    menu_update_slider_drag(mx, window_width);
                } else {
                    // Pass motion to menus for hover effects
                    if (g_show_base_menu) menu_handle_mouse_motion(menus->base_menu, mx, my);
                    if (g_show_mouse_controls) menu_handle_mouse_motion(menus->mouse_menu, mx, my);
                    if (g_show_sim_controls) menu_handle_mouse_motion(menus->sim_menu, mx, my);
                    if (selected_source && selected_source->active && menus->source_menu) {
                        menu_handle_mouse_motion(menus->source_menu, mx, my);
                    }
                    
                    // Wave painting (if in ADD_WAVE mode and mouse button held)
                    if (g_mouse_mode == MOUSE_ADD_WAVE && wave_painting && 
                        (event.motion.state & SDL_BUTTON_LMASK)) {
                        // Convert to physical coordinates
                        double phys_x = ((double)mx / window_width) * sim->Lx_visible;
                        double phys_y = (1.0 - (double)my / window_height) * sim->Ly_visible;
                        
                        // Only add wave if we've moved far enough from last position
                        double dx = phys_x - last_wave_x;
                        double dy = phys_y - last_wave_y;
                        double dist = sqrt(dx * dx + dy * dy);
                        
                        if (dist >= wave_paint_spacing) {
                            add_wave_at_position(sim, phys_x, phys_y);
                            last_wave_x = phys_x;
                            last_wave_y = phys_y;
                        }
                    }
                    
                    // Source dragging (if in SOURCE mode and not over menus)
                    if (g_mouse_mode == MOUSE_SOURCE && selected_source && 
                        (event.motion.state & SDL_BUTTON_LMASK)) {
                        // Check if mouse is over any menu
                        int over_menu = 0;
                        if (g_show_base_menu && mx >= menus->base_menu->x && mx <= menus->base_menu->x + menus->base_menu->width &&
                            my >= menus->base_menu->y && my <= menus->base_menu->y + menus->base_menu->height) {
                            over_menu = 1;
                        }
                        if (g_show_mouse_controls && mx >= menus->mouse_menu->x && mx <= menus->mouse_menu->x + menus->mouse_menu->width &&
                            my >= menus->mouse_menu->y && my <= menus->mouse_menu->y + menus->mouse_menu->height) {
                            over_menu = 1;
                        }
                        if (g_show_sim_controls && mx >= menus->sim_menu->x && mx <= menus->sim_menu->x + menus->sim_menu->width &&
                            my >= menus->sim_menu->y && my <= menus->sim_menu->y + menus->sim_menu->height) {
                            over_menu = 1;
                        }
                        if (selected_source->active && menus->source_menu && mx >= menus->source_menu->x && mx <= menus->source_menu->x + menus->source_menu->width &&
                            my >= menus->source_menu->y && my <= menus->source_menu->y + menus->source_menu->height) {
                            over_menu = 1;
                        }
                        
                        if (!over_menu) {
                            // Drag the source
                            double phys_x = ((double)mx / window_width) * sim->Lx_visible;
                            double phys_y = (1.0 - (double)my / window_height) * sim->Ly_visible;
                            
                            // Snap to grid
                            snap_to_cell_center(sim, &phys_x, &phys_y);
                            
                            // Clamp to visible region
                            if (phys_x >= 0.0 && phys_x <= sim->Lx_visible &&
                                phys_y >= 0.0 && phys_y <= sim->Ly_visible) {
                                selected_source->x = phys_x;
                                selected_source->y = phys_y;
                            }
                        }
                    }
                    
                    // Barrier point dragging (if in ADD_BARRIER mode)
                    if (g_mouse_mode == MOUSE_ADD_BARRIER && (event.motion.state & SDL_BUTTON_LMASK)) {
                        double phys_x = ((double)mx / window_width) * sim->Lx_visible;
                        double phys_y = (1.0 - (double)my / window_height) * sim->Ly_visible;
                        snap_to_cell_center(sim, &phys_x, &phys_y);
                        
                        // Update dragged point position (already set from mousedown)
                        if (dragging_barrier_point >= 0) {
                            barrier_points[dragging_barrier_point].x = phys_x;
                            barrier_points[dragging_barrier_point].y = phys_y;
                            
                            // Update the boundary in-place if we have a complete pair
                            if (dragging_barrier_idx >= 0 && dragging_barrier_idx < sim->grid->n_interior_boundaries) {
                                int pt1_idx = dragging_barrier_idx * 2;
                                int pt2_idx = dragging_barrier_idx * 2 + 1;
                                if (pt2_idx < n_barrier_points) {
                                    // Get the existing boundary
                                    HyperplaneBoundary *hb = &sim->grid->interior_boundaries[dragging_barrier_idx];
                                    
                                    // Calculate new line segment parameters
                                    double offset_x = sim->sponge_width * sim->dx;
                                    double offset_y = sim->sponge_width * sim->dy;
                                    
                                    double phys_x1 = barrier_points[pt1_idx].x + offset_x;
                                    double phys_y1 = barrier_points[pt1_idx].y + offset_y;
                                    double phys_x2 = barrier_points[pt2_idx].x + offset_x;
                                    double phys_y2 = barrier_points[pt2_idx].y + offset_y;
                                    
                                    double dx_seg = phys_x2 - phys_x1;
                                    double dy_seg = phys_y2 - phys_y1;
                                    double length = sqrt(dx_seg * dx_seg + dy_seg * dy_seg);
                                    
                                    if (length > 1e-6) {
                                        // Update normal vector (perpendicular to line)
                                        hb->normal[0] = -dy_seg / length;
                                        hb->normal[1] = dx_seg / length;
                                        
                                        // Update midpoint
                                        hb->point[0] = (phys_x1 + phys_x2) / 2.0;
                                        hb->point[1] = (phys_y1 + phys_y2) / 2.0;
                                        
                                        // Update symmetric parametric bounds
                                        hb->bounds_min[0] = -length / 2.0;
                                        hb->bounds_max[0] = length / 2.0;
                                        
                                        // Update bounding box
                                        hb->bbox_min[0] = fmin(phys_x1, phys_x2);
                                        hb->bbox_max[0] = fmax(phys_x1, phys_x2);
                                        hb->bbox_min[1] = fmin(phys_y1, phys_y2);
                                        hb->bbox_max[1] = fmax(phys_y1, phys_y2);
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (event.type == SDL_WINDOWEVENT) {
                if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
                    window_width = event.window.data1;
                    window_height = event.window.data2;
                    glViewport(0, 0, window_width, window_height);
                }
            }
        }
        
        // Run simulation steps (respecting max speed limit)
        Uint32 current_time = SDL_GetTicks();
        Uint32 step_start_time = current_time;
        int steps_this_frame = 0;
        
        while (!sim->paused && (current_time - last_frame_time) < FRAME_TIME_MS) {
            simulation_step(sim);
            steps_this_frame++;
            current_time = SDL_GetTicks();
            
            // Enforce max simulation speed if set
            if (sim->max_sim_speed > 0.0) {
                double elapsed_ms = current_time - step_start_time;
                double required_time_per_step = 1000.0 / sim->max_sim_speed;
                double expected_time = steps_this_frame * required_time_per_step;
                
                if (elapsed_ms < expected_time) {
                    // We're running too fast, sleep a bit
                    Uint32 sleep_time = (Uint32)(expected_time - elapsed_ms);
                    if (sleep_time > 0 && sleep_time < 50) {
                        SDL_Delay(sleep_time);
                        current_time = SDL_GetTicks();
                    }
                }
            }
        }
        
        // Update performance stats
        if (current_time - last_stats_time >= 500) {  // Update every 500ms
            double elapsed = (current_time - last_stats_time) / 1000.0;
            sim->steps_per_second = sim->steps_since_last_update / elapsed;
            sim->steps_since_last_update = 0;
            last_stats_time = current_time;
        }
        
        // Render frame at fixed rate
        if (current_time - last_frame_time >= FRAME_TIME_MS) {
            // Setup OpenGL for rendering
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            
            // Enable alpha blending for transparent previews
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, window_width, window_height, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            
            // Render wave field
            render_wave_field(sim, &render, window_width, window_height);
            
            // Render wave sources
            render_wave_sources(sim, window_width, window_height);
            
            // Render barrier points and preview
            if (g_mouse_mode == MOUSE_ADD_BARRIER) {
                // Render all placed barrier points (white circles)
                if (n_barrier_points > 0) {
                    glPointSize(8.0f);
                    glBegin(GL_POINTS);
                    glColor3f(1.0f, 1.0f, 1.0f);  // White
                    for (int i = 0; i < n_barrier_points; i++) {
                        float sx = (barrier_points[i].x / sim->Lx_visible) * window_width;
                        float sy = (1.0 - barrier_points[i].y / sim->Ly_visible) * window_height;
                        glVertex2f(sx, sy);
                    }
                    glEnd();
                }
                
                // Render preview circle under mouse (grey)
                int mouse_x, mouse_y;
                SDL_GetMouseState(&mouse_x, &mouse_y);
                double preview_phys_x = ((double)mouse_x / window_width) * sim->Lx_visible;
                double preview_phys_y = (1.0 - (double)mouse_y / window_height) * sim->Ly_visible;
                snap_to_cell_center(sim, &preview_phys_x, &preview_phys_y);
                
                float preview_sx = (preview_phys_x / sim->Lx_visible) * window_width;
                float preview_sy = (1.0 - preview_phys_y / sim->Ly_visible) * window_height;
                
                glPointSize(10.0f);
                glBegin(GL_POINTS);
                glColor3f(0.5f, 0.5f, 0.5f);  // Grey
                glVertex2f(preview_sx, preview_sy);
                glEnd();
            }
            
            // Render source placement preview (yellow highlight)
            if (g_mouse_mode == MOUSE_SOURCE) {
                int mouse_x, mouse_y;
                SDL_GetMouseState(&mouse_x, &mouse_y);
                double preview_phys_x = ((double)mouse_x / window_width) * sim->Lx_visible;
                double preview_phys_y = (1.0 - (double)mouse_y / window_height) * sim->Ly_visible;
                snap_to_cell_center(sim, &preview_phys_x, &preview_phys_y);
                
                // Only show preview if clicking would place a new source (not select existing)
                WaveSource *existing = find_source_at(preview_phys_x, preview_phys_y);
                if (!existing) {
                    // Convert to screen coordinates
                    float preview_sx = (preview_phys_x / sim->Lx_visible) * window_width;
                    float preview_sy = (1.0 - preview_phys_y / sim->Ly_visible) * window_height;
                    
                    // Draw preview circle with radius based on current spread setting
                    float radius_screen = sim->wave_spread * ((window_width / sim->Lx_visible) + 
                                                               (window_height / sim->Ly_visible)) / 2.0;
                    
                    // Draw filled circle
                    int segments = 32;
                    glBegin(GL_TRIANGLE_FAN);
                    glColor4f(1.0f, 1.0f, 0.0f, 0.3f);  // Yellow with transparency
                    glVertex2f(preview_sx, preview_sy);  // Center
                    for (int j = 0; j <= segments; j++) {
                        float angle = 2.0f * M_PI * j / segments;
                        float x = preview_sx + radius_screen * cosf(angle);
                        float y = preview_sy + radius_screen * sinf(angle);
                        glVertex2f(x, y);
                    }
                    glEnd();
                    
                    // Draw outline
                    glLineWidth(2.0f);
                    glBegin(GL_LINE_LOOP);
                    glColor3f(1.0f, 1.0f, 0.0f);  // Bright yellow outline
                    for (int j = 0; j < segments; j++) {
                        float angle = 2.0f * M_PI * j / segments;
                        float x = preview_sx + radius_screen * cosf(angle);
                        float y = preview_sy + radius_screen * sinf(angle);
                        glVertex2f(x, y);
                    }
                    glEnd();
                    
                    // Draw center point
                    glPointSize(6.0f);
                    glBegin(GL_POINTS);
                    glColor3f(1.0f, 1.0f, 0.0f);  // Yellow
                    glVertex2f(preview_sx, preview_sy);
                    glEnd();
                }
            }
            
            // Render stats overlay
            if (render.show_stats) {
                char stats_buf[512];
                snprintf(stats_buf, sizeof(stats_buf), 
                        "Steps/s: %.1f | Time: %.3fs | Step: %llu | FPS: %.1f\n"
                        "Energy: %.3e | Baseline: %.3e | Error: %+.2f%%",
                        sim->steps_per_second, sim->sim_time, 
                        (unsigned long long)sim->step_count,
                        1000.0 / FRAME_TIME_MS,
                        sim->total_energy, sim->baseline_energy,
                        sim->energy_error * 100.0);
                
                Color white = {255, 255, 255, 255};
                menu_draw_text_at(stats_buf, 10, window_height - 40, white);
            }
                                   
            // Update and render source menu if source is selected and in SOURCE mode
            if (selected_source && selected_source->active && g_mouse_mode == MOUSE_SOURCE) {
                // Only rebuild menu if source changed
                if (last_menu_source != selected_source) {
                    update_source_menu(&menus->source_menu, selected_source);
                    last_menu_source = selected_source;
                }
                if (menus->source_menu) {
                    menu_render(menus->source_menu, window_width, window_height);
                }
            } else {
                // No source selected - clear menu and tracking
                if (last_menu_source != NULL) {
                    if (menus->source_menu) {
                        if (g_menu_drag_state.drag_source_menu == menus->source_menu) {
                            menu_end_slider_drag();
                        }
                        menu_free(menus->source_menu);
                        menus->source_menu = NULL;
                    }
                    last_menu_source = NULL;
                }
            }
            
            // Render menus conditionally
            if (g_show_base_menu) {
                menu_render(menus->base_menu, window_width, window_height);
            }
            if (g_show_mouse_controls) {
                menu_render(menus->mouse_menu, window_width, window_height);
            }
            if (g_show_sim_controls) {
                menu_render(menus->sim_menu, window_width, window_height);
            }

            // Render mouse mode indicator in top right corner
            {
                char mode_buf[64];
                snprintf(mode_buf, sizeof(mode_buf), "%s", g_mouse_mode_names[g_mouse_mode]);
                Color cyan = {100, 200, 255, 255};
                
                // Measure text width (approximate - 9 pixels per char for better spacing)
                int text_width = strlen(mode_buf) * 9;
                menu_draw_text_at(mode_buf, window_width - text_width - 20, 10, cyan);
            }
            
            SDL_GL_SwapWindow(window);
            sim->frame_count++;
            last_frame_time = current_time;
        }
        
        // Small sleep to prevent busy-waiting
        SDL_Delay(1);
    }
    
    // Cleanup
    printf("\nShutting down...\n");
    printf("Final stats:\n");
    printf("  Total frames: %llu\n", (unsigned long long)sim->frame_count);
    printf("  Total steps: %llu\n", (unsigned long long)sim->step_count);
    printf("  Simulated time: %.3f seconds\n", sim->sim_time);
    
    free_menus(menus);
    menu_clear_font();
    simulation_free(sim);
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}
