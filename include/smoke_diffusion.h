#ifndef SMOKE_DIFFUSION_H
#define SMOKE_DIFFUSION_H

#include "grid.h"

// Solve Helmholtz system: (I - alpha * Lap) u = b
// b: right-hand side GridField
// u: solution GridField (must have same grid)
// alpha: diffusion factor (nu * dt)
// tol, max_iter: convergence controls
// Returns 0 on success, non-zero on failure
int smoke_solve_helmholtz(const GridField *b, GridField *u, double alpha, double tol, int max_iter);

// Apply implicit diffusion to a scalar grid field in-place using backward Euler
// field: field to diffuse (modified in place)
// nu: diffusion coefficient
// dt: time step
// tol, max_iter: solver controls
// Returns 0 on success
int smoke_diffuse_field(GridField *field, double nu, double dt, double tol, int max_iter);

// Apply implicit diffusion to velocity components in-place
int smoke_diffuse_velocity(GridField *vx, GridField *vy, double nu, double dt, double tol, int max_iter);

#endif // SMOKE_DIFFUSION_H
