#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../include/grid.h"
#include "../include/literal.h"
#include "../include/smoke_advection.h"

// Sample a scalar field at physical coordinates with simple BC handling.
// For BC_DIRICHLET: return boundary value.
// For BC_NEUMANN or BC_REFLECT: mirror coordinates across boundary (reflect).
// For BC_OPEN: clamp to nearest valid coordinate.
static double sample_field_with_bc(const GridField *field, const double *coords_in) {
    GridMetadata *grid = field->grid;
    int n_dims = grid->n_dims;
    double coords[3];
    for (int d=0; d<3; d++) coords[d] = coords_in[d];

    // check each axis for out-of-bounds and apply BC handling
    for (int a=0; a<n_dims; a++) {
        double min = grid->origin[a];
        double max = grid->origin[a] + grid->extent[a];
        if (coords[a] < min) {
            BoundarySpec *bc = &grid->boundaries[a*2 + 0];
            if (bc->type == BC_DIRICHLET) return bc->func ? bc->func(coords, bc->time) : bc->value;
            else if (bc->type == BC_NEUMANN || bc->type == BC_REFLECT) {
                // reflect
                coords[a] = min + (min - coords[a]);
            } else { // OPEN or default
                coords[a] = min;
            }
        } else if (coords[a] > max) {
            BoundarySpec *bc = &grid->boundaries[a*2 + 1];
            if (bc->type == BC_DIRICHLET) return bc->func ? bc->func(coords, bc->time) : bc->value;
            else if (bc->type == BC_NEUMANN || bc->type == BC_REFLECT) {
                // reflect
                coords[a] = max - (coords[a] - max);
            } else {
                coords[a] = max;
            }
        }
    }

    // clamp into domain
    for (int a=0; a<n_dims; a++) {
        if (coords[a] < grid->origin[a]) coords[a] = grid->origin[a];
        if (coords[a] > grid->origin[a] + grid->extent[a]) coords[a] = grid->origin[a] + grid->extent[a];
    }

    uint32_t idx[3] = {0,0,0};
    if (!grid_coord_to_index(grid, coords, idx)) {
        for (int a=0;a<n_dims;a++) {
            double rel = (coords[a] - grid->origin[a]) / grid->spacing[a];
            int ii = (int)floor(rel + 0.5);
            if (ii < 0) ii = 0; if (ii >= (int)grid->dims[a]) ii = grid->dims[a]-1;
            idx[a] = ii;
        }
    }

    Literal *lit = grid_field_get(field, idx);
    double v = literal_get(lit, (uint32_t[]){0,0,0});
    literal_free(lit);
    return v;
}
static double lit_to_double_and_free(Literal *lit) {
    if (!lit) return 0.0;
    uint32_t idx[3] = {0,0,0};
    double v = literal_get(lit, idx);
    literal_free(lit);
    return v;
}

void smoke_divergence(const GridField *vx, const GridField *vy, GridField *out) {
    if (!vx || !vy || !out) return;
    GridMetadata *grid = vx->grid;
    uint32_t indices[3];
    double dx = grid->spacing[0];
    double dy = (grid->n_dims > 1) ? grid->spacing[1] : 1.0;

    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, indices);

        // x derivative (central interior, forward/backward at boundaries)
        double dvx_dx = 0.0;
        if (indices[0] == 0) {
            uint32_t idx0[3] = {indices[0], indices[1], indices[2]};
            uint32_t idx1[3] = {indices[0]+1, indices[1], indices[2]};
            double v0 = lit_to_double_and_free(grid_field_get(vx, idx0));
            double v1 = lit_to_double_and_free(grid_field_get(vx, idx1));
            dvx_dx = (v1 - v0) / dx;
        } else if (indices[0] + 1 >= grid->dims[0]) {
            uint32_t idx0[3] = {indices[0]-1, indices[1], indices[2]};
            uint32_t idx1[3] = {indices[0], indices[1], indices[2]};
            double v0 = lit_to_double_and_free(grid_field_get(vx, idx0));
            double v1 = lit_to_double_and_free(grid_field_get(vx, idx1));
            dvx_dx = (v1 - v0) / dx;
        } else {
            uint32_t idxm[3] = {indices[0]-1, indices[1], indices[2]};
            uint32_t idxp[3] = {indices[0]+1, indices[1], indices[2]};
            double vm = lit_to_double_and_free(grid_field_get(vx, idxm));
            double vp = lit_to_double_and_free(grid_field_get(vx, idxp));
            dvx_dx = (vp - vm) / (2.0*dx);
        }

        // y derivative
        double dvy_dy = 0.0;
        if (grid->n_dims > 1) {
            if (indices[1] == 0) {
                uint32_t idx0[3] = {indices[0], indices[1], indices[2]};
                uint32_t idx1[3] = {indices[0], indices[1]+1, indices[2]};
                double v0 = lit_to_double_and_free(grid_field_get(vy, idx0));
                double v1 = lit_to_double_and_free(grid_field_get(vy, idx1));
                dvy_dy = (v1 - v0) / dy;
            } else if (indices[1] + 1 >= grid->dims[1]) {
                uint32_t idx0[3] = {indices[0], indices[1]-1, indices[2]};
                uint32_t idx1[3] = {indices[0], indices[1], indices[2]};
                double v0 = lit_to_double_and_free(grid_field_get(vy, idx0));
                double v1 = lit_to_double_and_free(grid_field_get(vy, idx1));
                dvy_dy = (v1 - v0) / dy;
            } else {
                uint32_t idxm[3] = {indices[0], indices[1]-1, indices[2]};
                uint32_t idxp[3] = {indices[0], indices[1]+1, indices[2]};
                double vm = lit_to_double_and_free(grid_field_get(vy, idxm));
                double vp = lit_to_double_and_free(grid_field_get(vy, idxp));
                dvy_dy = (vp - vm) / (2.0*dy);
            }
        }

        double div = dvx_dx + dvy_dy;
        Literal *out_lit = literal_create_scalar(div);
        grid_field_set(out, indices, out_lit);
        literal_free(out_lit);
    }
}

GridField* smoke_advect_scalar(const GridField *field, GridField **velocity, int n_dims, double dt) {
    if (!field || !velocity) return NULL;
    GridMetadata *grid = field->grid;
    GridField *result = grid_field_create(grid);
    uint32_t indices[3];
    double coords[3];

    for (uint32_t linear=0; linear<grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, indices);
        grid_index_to_coord(grid, indices, coords);

        // sample velocity at current location (nearest grid sample)
        double vx = sample_field_with_bc(velocity[0], coords);
        double vy = sample_field_with_bc(velocity[1], coords);

        double x0 = coords[0] - vx * dt;
        double y0 = coords[1] - vy * dt;
        double back[3] = {x0, y0, coords[2]};


        double sampled_val = sample_field_with_bc(field, back);
        Literal *samp_ptr = literal_create_scalar(sampled_val);
        grid_field_set(result, indices, samp_ptr);
        literal_free(samp_ptr);
    }

    return result;
}

GridField** smoke_advect_velocity(GridField **velocity, int n_dims, double dt) {
    if (!velocity) return NULL;
    GridField **out = malloc(sizeof(GridField*) * n_dims);
    for (int d=0; d<n_dims; d++) {
        out[d] = smoke_advect_scalar(velocity[d], velocity, n_dims, dt);
    }
    return out;
}
