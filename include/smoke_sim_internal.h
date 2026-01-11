#ifndef SMOKE_SIM_INTERNAL_H
#define SMOKE_SIM_INTERNAL_H

#include "grid.h"

// Minimal smoke sim skeleton: timestep function signature
// state will be defined in smoke_sim.c; these helpers let other modules call into sim
int smoke_sim_step(GridField *density, GridField *vx, GridField *vy, double dt, double nu, double tol, int max_iter);

#endif // SMOKE_SIM_INTERNAL_H
