#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../include/smoke_sim_internal.h"
#include "../include/smoke_diffusion.h"
#include "../include/smoke_advection.h"
#include "../include/smoke_pressure.h"
#include <string.h>
#include "../include/smoke_sim.h"
#include "../include/literal.h"

// Minimal timestep: advect -> diffuse -> project
static double timespec_diff_ms(struct timespec *a, struct timespec *b) {
    double sec = (double)(b->tv_sec - a->tv_sec);
    double nsec = (double)(b->tv_nsec - a->tv_nsec);
    return sec * 1000.0 + nsec / 1e6;
}

int smoke_sim_step(GridField *density, GridField *vx, GridField *vy, double dt, double nu, double tol, int max_iter) {
    if (!density || !vx || !vy) return 1;

    struct timespec t0, t1, t2, t3, t4;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Advect density and velocity (semi-Lagrangian)
    GridField *vel_arr[2]; vel_arr[0] = vx; vel_arr[1] = vy;
    GridField *new_density = smoke_advect_scalar(density, vel_arr, 2, dt);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double advect_density_ms = timespec_diff_ms(&t0, &t1);

    if (new_density) {
        // copy back into density
        uint32_t idx[3];
        for (uint32_t linear = 0; linear < density->grid->total_points; linear++) {
            grid_linear_to_index(density->grid, linear, idx);
            double v = literal_get(&new_density->data, idx);
            Literal *lv = literal_create_scalar(v);
            grid_field_set(density, idx, lv); literal_free(lv);
        }
        grid_field_free(new_density);
    }

    GridField **new_vel = smoke_advect_velocity(vel_arr, 2, dt);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double advect_vel_ms = timespec_diff_ms(&t1, &t2);

    if (new_vel) {
        // copy back into vx, vy
        uint32_t idx[3];
        for (uint32_t linear = 0; linear < vx->grid->total_points; linear++) {
            grid_linear_to_index(vx->grid, linear, idx);
            double v0 = literal_get(&new_vel[0]->data, idx);
            double v1 = literal_get(&new_vel[1]->data, idx);
            Literal *l0 = literal_create_scalar(v0);
            Literal *l1 = literal_create_scalar(v1);
            grid_field_set(vx, idx, l0); literal_free(l0);
            grid_field_set(vy, idx, l1); literal_free(l1);
        }
        grid_field_free(new_vel[0]); grid_field_free(new_vel[1]); free(new_vel);
    }

    // Diffuse velocity
    smoke_diffuse_velocity(vx, vy, nu, dt, tol, max_iter);
    clock_gettime(CLOCK_MONOTONIC, &t3);
    double diffuse_ms = timespec_diff_ms(&t2, &t3);

    // Project
    GridField *pressure = grid_field_create(vx->grid);
    smoke_pressure_project(vx, vy, pressure, dt, tol, max_iter);
    clock_gettime(CLOCK_MONOTONIC, &t4);
    double project_ms = timespec_diff_ms(&t3, &t4);
    grid_field_free(pressure);

    double total_ms = timespec_diff_ms(&t0, &t4);
    printf("smoke_sim_step: advect_density=%.3fms advect_vel=%.3fms diffuse=%.3fms project=%.3fms total=%.3fms\n",
           advect_density_ms, advect_vel_ms, diffuse_ms, project_ms, total_ms);

    return 0;
}

SmokeSimState* smoke_sim_create(GridMetadata *grid, double dt) {
    if (!grid) return NULL;
    SmokeSimState *s = calloc(1, sizeof(SmokeSimState));
    s->grid = grid;
    s->dt = dt;
    s->viscosity = 0.001;
    s->diffusion = 0.0001;
    s->density = grid_field_create(grid);
    s->pressure = grid_field_create(grid);
    s->vx = grid_field_create(grid);
    s->vy = grid_field_create(grid);
    s->sim_time = 0.0;
    s->step_count = 0;
    return s;
}

void smoke_sim_free(SmokeSimState *s) {
    if (!s) return;
    grid_field_free(s->density);
    grid_field_free(s->pressure);
    grid_field_free(s->vx);
    grid_field_free(s->vy);
    free(s);
}

void smoke_sim_add_density(SmokeSimState *s, const uint32_t *indices, double amount) {
    if (!s || !indices) return;
    // validate indices against grid dims (protect against negative->uint cast)
    GridMetadata *g = s->grid;
    for (int ax = 0; ax < g->n_dims; ++ax) {
        if (indices[ax] >= g->dims[ax]) return;
    }
    Literal *cur = grid_field_get(s->density, indices);
    if (!cur) return; // out-of-bounds or invalid access; ignore the add
    double v = literal_get(cur, (uint32_t[]){0,0,0});
    literal_free(cur);
    Literal *sum = literal_create_scalar(v + amount);
    grid_field_set(s->density, indices, sum);
    literal_free(sum);
}

void smoke_sim_add_velocity(SmokeSimState *s, const uint32_t *indices, double dvx, double dvy) {
    if (!s || !indices) return;
    // validate indices against grid dims (protect against negative->uint cast)
    GridMetadata *g = s->grid;
    for (int ax = 0; ax < g->n_dims; ++ax) {
        if (indices[ax] >= g->dims[ax]) return;
    }
    // vx
    Literal *curx = grid_field_get(s->vx, indices);
    if (!curx) return;
    double vxv = literal_get(curx, (uint32_t[]){0,0,0});
    literal_free(curx);
    Literal *sumx = literal_create_scalar(vxv + dvx);
    grid_field_set(s->vx, indices, sumx);
    literal_free(sumx);

    // vy
    Literal *cury = grid_field_get(s->vy, indices);
    if (!cury) return;
    double vyv = literal_get(cury, (uint32_t[]){0,0,0});
    literal_free(cury);
    Literal *sumy = literal_create_scalar(vyv + dvy);
    grid_field_set(s->vy, indices, sumy);
    literal_free(sumy);
}

// Minimal grayscale BMP writer for density; returns 0 on success
int smoke_sim_render_frame(const SmokeSimState *s, const char *filename) {
    if (!s || !filename) return 1;
    GridField *density = s->density;
    GridMetadata *grid = density->grid;
    uint32_t nx = grid->dims[0];
    uint32_t ny = (grid->n_dims>1)? grid->dims[1] : 1;

    uint32_t width = nx;
    uint32_t height = ny;
    uint32_t row_size = ((width + 3) / 4) * 4; // padded to 4 bytes per row for 8-bit gray
    uint32_t img_size = row_size * height;

    FILE *f = fopen(filename, "wb");
    if (!f) return 1;

    // Simple 8-bit grayscale BMP header (minimal)
    unsigned char header[54] = {0};
    // BMP header
    header[0] = 'B'; header[1] = 'M';
    uint32_t file_size = 54 + img_size*3;
    memcpy(&header[2], &file_size, 4);
    uint32_t offset = 54;
    memcpy(&header[10], &offset, 4);
    uint32_t dib_size = 40;
    memcpy(&header[14], &dib_size, 4);
    int32_t w = (int32_t)width; int32_t h = (int32_t)height;
    memcpy(&header[18], &w, 4); memcpy(&header[22], &h, 4);
    uint16_t planes = 1; memcpy(&header[26], &planes, 2);
    uint16_t bpp = 24; memcpy(&header[28], &bpp, 2);

    fwrite(header, 1, 54, f);

    // Write pixels bottom-up
    uint32_t indices[3];
    for (int32_t y = height - 1; y >= 0; y--) {
        for (uint32_t x = 0; x < width; x++) {
            indices[0] = x; indices[1] = y; indices[2] = 0;
            Literal *lit = grid_field_get(density, indices);
            double val = literal_get(lit, (uint32_t[]){0,0,0});
            literal_free(lit);
            // map value to byte (clamp)
            int b = (int)(val * 255.0);
            if (b < 0) b = 0; if (b > 255) b = 255;
            unsigned char pix[3] = { (unsigned char)b, (unsigned char)b, (unsigned char)b };
            fwrite(pix, 1, 3, f);
        }
        // padding
        for (uint32_t p=0; p< (row_size - width); p++) fputc(0, f);
    }

    fclose(f);
    return 0;
}
