#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../include/smoke_diffusion.h"
#include "../include/literal.h"

// Solve Helmholtz: (I - alpha * Lap) u = b using Jacobi iterations
int smoke_solve_helmholtz(const GridField *b, GridField *u, double alpha, double tol, int max_iter) {
    if (!b || !u) return 1;
    GridMetadata *grid = b->grid;
    if (grid != u->grid) return 1;

    uint32_t nx = grid->dims[0];
    uint32_t ny = (grid->n_dims > 1) ? grid->dims[1] : 1;
    double dx = grid->spacing[0];
    double dy = (grid->n_dims > 1) ? grid->spacing[1] : 1.0;
    double idx2 = 1.0 / (dx * dx);
    double idy2 = 1.0 / (dy * dy);

    double diag = 1.0 + 2.0 * alpha * (idx2 + idy2);

    // initialize u to zero
    Literal zero = { {nx, ny, 1}, NULL };
    grid_field_fill(u, &zero);

    double *u_new = malloc(sizeof(double) * nx * ny);
    if (!u_new) return 1;

    uint32_t idx[3];
    for (int iter = 0; iter < max_iter; iter++) {
        // Jacobi update
        for (uint32_t j = 0; j < ny; j++) {
            for (uint32_t i = 0; i < nx; i++) {
                idx[0] = i; idx[1] = j; idx[2] = 0;
                // Handle boundaries according to grid BCs
                uint32_t ip[3] = { (i+1 < nx) ? i+1 : i, j, 0 };
                uint32_t im[3] = { (i>0) ? i-1 : i, j, 0 };
                uint32_t jp[3] = { i, (j+1 < ny) ? j+1 : j, 0 };
                uint32_t jm[3] = { i, (j>0) ? j-1 : j, 0 };

                double u_ip = literal_get(&u->data, ip);
                double u_im = literal_get(&u->data, im);
                double u_jp = literal_get(&u->data, jp);
                double u_jm = literal_get(&u->data, jm);

                double bval = literal_get(&b->data, idx);

                // If at left boundary, enforce Dirichlet or approximate ghost for Neumann
                if (i == 0) {
                    BoundarySpec *bc = &grid->boundaries[0*2 + 0];
                    if (bc->type == BC_DIRICHLET) {
                        double bc_val = bc->func ? bc->func((double[]){grid->origin[0], grid->origin[1]}, bc->time) : bc->value;
                        u_new[j*nx + i] = bc_val;
                        continue;
                    } else if (bc->type == BC_NEUMANN || bc->type == BC_REFLECT) {
                        double g = bc->func ? bc->func((double[]){grid->origin[0], grid->origin[1]}, bc->time) : bc->value;
                        // approximate ghost value u_im using u_ip and derivative: (u_ip - u_im)/(2h) = g -> u_im = u_ip - 2h*g
                        u_im = u_ip - 2.0 * grid->spacing[0] * g;
                    } else {
                        // OPEN: mirror/clamp
                        u_im = u_ip;
                    }
                }

                if (i == nx-1) {
                    BoundarySpec *bc = &grid->boundaries[0*2 + 1];
                    if (bc->type == BC_DIRICHLET) {
                        double bc_val = bc->func ? bc->func((double[]){grid->origin[0]+grid->extent[0], grid->origin[1]}, bc->time) : bc->value;
                        u_new[j*nx + i] = bc_val;
                        continue;
                    } else if (bc->type == BC_NEUMANN || bc->type == BC_REFLECT) {
                        double g = bc->func ? bc->func((double[]){grid->origin[0]+grid->extent[0], grid->origin[1]}, bc->time) : bc->value;
                        u_ip = u_im + 2.0 * grid->spacing[0] * g; // u_ip ghost
                    } else {
                        u_ip = u_im;
                    }
                }

                if (j == 0) {
                    BoundarySpec *bc = &grid->boundaries[1*2 + 0];
                    if (bc->type == BC_DIRICHLET) {
                        double bc_val = bc->func ? bc->func((double[]){grid->origin[0], grid->origin[1]}, bc->time) : bc->value;
                        u_new[j*nx + i] = bc_val;
                        continue;
                    } else if (bc->type == BC_NEUMANN || bc->type == BC_REFLECT) {
                        double g = bc->func ? bc->func((double[]){grid->origin[0], grid->origin[1]}, bc->time) : bc->value;
                        u_jm = u_jp - 2.0 * grid->spacing[1] * g;
                    } else {
                        u_jm = u_jp;
                    }
                }

                if (j == ny-1) {
                    BoundarySpec *bc = &grid->boundaries[1*2 + 1];
                    if (bc->type == BC_DIRICHLET) {
                        double bc_val = bc->func ? bc->func((double[]){grid->origin[0], grid->origin[1]+grid->extent[1]}, bc->time) : bc->value;
                        u_new[j*nx + i] = bc_val;
                        continue;
                    } else if (bc->type == BC_NEUMANN || bc->type == BC_REFLECT) {
                        double g = bc->func ? bc->func((double[]){grid->origin[0], grid->origin[1]+grid->extent[1]}, bc->time) : bc->value;
                        u_jp = u_jm + 2.0 * grid->spacing[1] * g;
                    } else {
                        u_jp = u_jm;
                    }
                }

                double numer = bval + alpha * (u_ip + u_im) * idx2 + alpha * (u_jp + u_jm) * idy2;
                double uval = numer / diag;
                u_new[j*nx + i] = uval;
            }
        }

        // copy back and compute residual
        double res_sq = 0.0;
        for (uint32_t j = 0; j < ny; j++) {
            for (uint32_t i = 0; i < nx; i++) {
                idx[0] = i; idx[1] = j; idx[2] = 0;
                double val = u_new[j*nx + i];
                Literal *lv = literal_create_scalar(val);
                grid_field_set(u, idx, lv);
                literal_free(lv);

                if (i == 0 || j == 0 || i == nx-1 || j == ny-1) continue;

                // compute residual r = u - alpha * Lap(u) - b
                uint32_t ip[3] = {i+1,j,0};
                uint32_t im[3] = {i-1,j,0};
                uint32_t jp[3] = {i,j+1,0};
                uint32_t jm[3] = {i,j-1,0};
                double u_c = val;
                double lap = (literal_get(&u->data, ip) - 2.0 * u_c + literal_get(&u->data, im)) * idx2
                           + (literal_get(&u->data, jp) - 2.0 * u_c + literal_get(&u->data, jm)) * idy2;
                double r = u_c - alpha * lap - literal_get(&b->data, idx);
                res_sq += r * r;
            }
        }

        double res = sqrt(res_sq / ((nx-2)*(ny-2)));
        if (res < tol) {
            free(u_new);
            return 0;
        }
    }

    free(u_new);
    return 1; // did not converge
}

// In-place diffusion: u_new solves (I - alpha Lap) u_new = u_old
int smoke_diffuse_field(GridField *field, double nu, double dt, double tol, int max_iter) {
    if (!field) return 1;
    double alpha = nu * dt;
    GridMetadata *grid = field->grid;

    GridField *b = grid_field_copy(field);
    GridField *u = grid_field_create(grid);
    int rc = smoke_solve_helmholtz(b, u, alpha, tol, max_iter);
    if (rc != 0) {
        grid_field_free(b); grid_field_free(u);
        return rc;
    }

    // copy solution back into field
    uint32_t idx[3];
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        double val = literal_get(&u->data, idx);
        Literal *lv = literal_create_scalar(val);
        grid_field_set(field, idx, lv);
        literal_free(lv);
    }

    grid_field_free(b); grid_field_free(u);
    return 0;
}

int smoke_diffuse_velocity(GridField *vx, GridField *vy, double nu, double dt, double tol, int max_iter) {
    if (!vx || !vy) return 1;
    if (vx->grid != vy->grid) return 1;
    int rc = smoke_diffuse_field(vx, nu, dt, tol, max_iter);
    if (rc != 0) return rc;
    rc = smoke_diffuse_field(vy, nu, dt, tol, max_iter);
    return rc;
}
