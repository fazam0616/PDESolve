#ifndef SMOKE_PRESSURE_H
#define SMOKE_PRESSURE_H

#include "grid.h"

// Solve Poisson equation ∇² p = rhs on the grid with Dirichlet zero boundaries
// rhs: GridField containing right-hand side (same grid)
// p: GridField to store solution (must be created with same grid)
// tol: residual tolerance (L2) for convergence
// max_iter: maximum Jacobi iterations
// Returns 0 on success, nonzero on failure or no convergence
// If out_iters != NULL, writes the number of iterations performed into *out_iters
int smoke_solve_poisson(const GridField *rhs, GridField *p, double tol, int max_iter, int *out_iters);

// Project velocity to be divergence-free: computes divergence, solves for pressure,
// and subtracts grad(p) from velocity fields vx, vy (2D)
// Returns 0 on success
int smoke_pressure_project(GridField *vx, GridField *vy, GridField *pressure, double dt, double tol, int max_iter);

#endif // SMOKE_PRESSURE_H
