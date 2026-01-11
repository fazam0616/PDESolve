#ifndef SMOKE_ADVECTION_H
#define SMOKE_ADVECTION_H

#include "grid.h"

// Compute divergence: out = d(vx)/dx + d(vy)/dy (+ d(vz)/dz if present)
// vx, vy: velocity component fields (must match grid)
// out: pre-allocated GridField to receive divergence (must match grid)
void smoke_divergence(const GridField *vx, const GridField *vy, GridField *out);

// Semi-Lagrangian advection for scalar field
// Returns a newly allocated GridField (caller must free)
GridField* smoke_advect_scalar(const GridField *field, GridField **velocity, int n_dims, double dt);

// Semi-Lagrangian advection for velocity components (advects in-place by returning new fields)
// velocity: array of component fields [vx, vy, (vz)]
// Returns an array of newly allocated GridField* (caller must free each)
GridField** smoke_advect_velocity(GridField **velocity, int n_dims, double dt);

#endif // SMOKE_ADVECTION_H
