#ifndef SMOKE_SIM_H
#define SMOKE_SIM_H

#include "grid.h"

typedef struct {
    GridMetadata *grid;
    GridField *density;
    GridField *pressure;
    GridField *vx;
    GridField *vy;
    double viscosity;
    double diffusion;
    double dt;
    double sim_time;
    unsigned long step_count;
} SmokeSimState;

// Create smoke sim state using existing grid
SmokeSimState* smoke_sim_create(GridMetadata *grid, double dt);
void smoke_sim_free(SmokeSimState *s);

// Render density field to a simple BMP file (grayscale)
// Returns 0 on success, nonzero on failure
int smoke_sim_render_frame(const SmokeSimState *s, const char *filename);

// Minimal helper to add density at a grid index
void smoke_sim_add_density(SmokeSimState *s, const uint32_t *indices, double amount);

// Minimal helper to add velocity perturbation at a grid index
void smoke_sim_add_velocity(SmokeSimState *s, const uint32_t *indices, double dvx, double dvy);

#endif // SMOKE_SIM_H
