#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "../include/grid.h"
#include "../include/literal.h"
#include "../include/solver.h"

// Get wall clock time in seconds
double get_wall_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

// Initialize wave equation fields
typedef struct {
    GridField *u;      // displacement
    GridField *v;      // velocity
    GridField *u_next; // next displacement
    GridMetadata *grid;
    double dt;
    double c;  // wave speed
} WaveState;

WaveState* wave_state_create(uint32_t nx, uint32_t ny, double dx, double dt, double c) {
    WaveState *state = malloc(sizeof(WaveState));
    if (!state) return NULL;
    
    uint32_t dims[] = {1, nx, ny};
    double spacing[] = {1.0, dx, dx};
    double origin[] = {0.0, 0.0, 0.0};
    
    state->grid = grid_metadata_create(dims, spacing, origin, 3);
    state->u = grid_field_create(state->grid);
    state->v = grid_field_create(state->grid);
    state->u_next = grid_field_create(state->grid);
    state->dt = dt;
    state->c = c;
    
    // Initialize to zero (fields are already zeroed by grid_field_create)
    
    return state;
}

void wave_state_free(WaveState *state) {
    if (!state) return;
    grid_field_free(state->u);
    grid_field_free(state->v);
    grid_field_free(state->u_next);
    grid_metadata_free(state->grid);
    free(state);
}

// Add a source at position
void add_source(WaveState *state, uint32_t ix, uint32_t iy, double amplitude) {
    uint32_t indices[] = {0, ix, iy};
    double current = literal_get(&state->u->data, indices);
    literal_set(&state->u->data, indices, current + amplitude);
}

// Add a barrier (set u and v to zero)
void add_barrier(WaveState *state, uint32_t ix_start, uint32_t ix_end, 
                 uint32_t iy_start, uint32_t iy_end) {
    for (uint32_t iy = iy_start; iy <= iy_end && iy < state->grid->dims[2]; iy++) {
        for (uint32_t ix = ix_start; ix <= ix_end && ix < state->grid->dims[1]; ix++) {
            uint32_t indices[] = {0, ix, iy};
            literal_set(&state->u->data, indices, 0.0);
            literal_set(&state->v->data, indices, 0.0);
        }
    }
}

// Time step using specified derivative function
typedef GridField* (*DerivativeFunc)(const GridField*, int, int);

void wave_step(WaveState *state, DerivativeFunc deriv_func) {
    double dt = state->dt;
    double c = state->c;
    double c_sq = c * c;
    
    // Compute Laplacian: d²u/dx² + d²u/dy²
    GridField *d2u_dx2 = deriv_func(state->u, 1, 2);  // axis 1 (x), order 2
    GridField *d2u_dy2 = deriv_func(state->u, 2, 2);  // axis 2 (y), order 2
    
    if (!d2u_dx2 || !d2u_dy2) {
        if (d2u_dx2) grid_field_free(d2u_dx2);
        if (d2u_dy2) grid_field_free(d2u_dy2);
        return;
    }
    
    // Wave equation: u_tt = c² * (u_xx + u_yy)
    // Using Verlet integration: u_next = 2u - u_prev + dt² * c² * Laplacian
    // But we store velocity v, so: v_new = v + dt * c² * Laplacian, u_new = u + dt * v_new
    
    uint32_t indices[3];
    indices[0] = 0;
    for (uint32_t iy = 0; iy < state->grid->dims[2]; iy++) {
        for (uint32_t ix = 0; ix < state->grid->dims[1]; ix++) {
            indices[1] = ix;
            indices[2] = iy;
            
            double u_val = literal_get(&state->u->data, indices);
            double v_val = literal_get(&state->v->data, indices);
            double laplacian = literal_get(&d2u_dx2->data, indices) + literal_get(&d2u_dy2->data, indices);
            
            // Update velocity
            double v_new = v_val + dt * c_sq * laplacian;
            literal_set(&state->v->data, indices, v_new);
            
            // Update displacement
            double u_new = u_val + dt * v_new;
            literal_set(&state->u->data, indices, u_new);
        }
    }
    
    grid_field_free(d2u_dx2);
    grid_field_free(d2u_dy2);
}

// Run benchmark for specified duration
uint64_t run_benchmark(const char *method_name, DerivativeFunc deriv_func, double duration_sec) {
    printf("\n=== Testing %s ===\n", method_name);
    
    // Create wave simulation (100x100 grid)
    uint32_t nx = 100, ny = 100;
    double dx = 0.1;
    double dt = 0.01;
    double c = 1.0;
    
    WaveState *state = wave_state_create(nx, ny, dx, dt, c);
    if (!state) {
        printf("Failed to create wave state\n");
        return 0;
    }
    
    // Add barrier in middle (vertical wall with gap)
    uint32_t barrier_x = nx / 2;
    add_barrier(state, barrier_x, barrier_x, 0, ny / 3);
    add_barrier(state, barrier_x, barrier_x, 2 * ny / 3, ny - 1);
    
    printf("Grid: %ux%u, dt=%.4f, c=%.2f\n", nx, ny, dt, c);
    printf("Running for %.1f seconds...\n", duration_sec);
    
    double start_time = get_wall_time();
    double end_time = start_time + duration_sec;
    uint64_t frame_count = 0;
    uint64_t source_interval = 10; // Add source every 10 frames
    
    while (get_wall_time() < end_time) {
        // Add periodic source
        if (frame_count % source_interval == 0) {
            add_source(state, nx / 4, ny / 2, 0.1);
        }
        
        // Simulate one step
        wave_step(state, deriv_func);
        
        // Reset barrier (in case wave affected it)
        if (frame_count % 5 == 0) {
            add_barrier(state, barrier_x, barrier_x, 0, ny / 3);
            add_barrier(state, barrier_x, barrier_x, 2 * ny / 3, ny - 1);
        }
        
        frame_count++;
        
        // Progress update every 1000 frames
        if (frame_count % 1000 == 0) {
            double elapsed = get_wall_time() - start_time;
            double fps = frame_count / elapsed;
            printf("  Frame %llu (%.1fs elapsed, %.1f FPS)\n", 
                   (unsigned long long)frame_count, elapsed, fps);
        }
    }
    
    double total_time = get_wall_time() - start_time;
    double fps = frame_count / total_time;
    
    printf("\nResults for %s:\n", method_name);
    printf("  Total frames: %llu\n", (unsigned long long)frame_count);
    printf("  Total time: %.3f seconds\n", total_time);
    printf("  Average FPS: %.2f\n", fps);
    printf("  Time per frame: %.4f ms\n", (total_time / frame_count) * 1000.0);
    
    wave_state_free(state);
    return frame_count;
}

int main(int argc, char **argv) {
    printf("========================================\n");
    printf("Compact vs Explicit Derivative Benchmark\n");
    printf("========================================\n");
    
    double test_duration = 15.0; // seconds
    
    if (argc > 1) {
        test_duration = atof(argv[1]);
        if (test_duration <= 0) test_duration = 15.0;
    }
    
    printf("\nBenchmark duration: %.1f seconds per method\n", test_duration);
    
    // Test explicit method first
    uint64_t explicit_frames = run_benchmark(
        "Explicit Finite Difference (Original)",
        grid_field_derivative,
        test_duration
    );
    
    printf("\n----------------------------------------\n");
    
    // Test compact method
    uint64_t compact_frames = run_benchmark(
        "Compact Finite Difference (4th Order)",
        grid_field_derivative_compact,
        test_duration
    );
    
    // Summary comparison
    printf("\n========================================\n");
    printf("COMPARISON SUMMARY\n");
    printf("========================================\n");
    printf("Explicit method: %llu frames\n", (unsigned long long)explicit_frames);
    printf("Compact method:  %llu frames\n", (unsigned long long)compact_frames);
    
    if (explicit_frames > 0) {
        double speedup = (double)compact_frames / (double)explicit_frames;
        printf("\nSpeedup: %.2fx ", speedup);
        if (speedup > 1.0) {
            printf("(Compact is %.0f%% faster)\n", (speedup - 1.0) * 100.0);
        } else {
            printf("(Compact is %.0f%% slower)\n", (1.0 - speedup) * 100.0);
        }
    }
    
    printf("\n========================================\n");
    printf("\nTo analyze profiling data:\n");
    printf("  1. Run: ./build/test_compact_benchmark\n");
    printf("  2. Check gmon.out: gprof ./build/test_compact_benchmark gmon.out | less\n");
    printf("  3. Look for time spent in:\n");
    printf("     - grid_field_derivative (explicit)\n");
    printf("     - grid_field_derivative_compact (compact)\n");
    printf("     - grid_linear_to_index (should be much lower with compact)\n");
    printf("     - thomas_solve (only in compact)\n");
    printf("\n");
    
    return 0;
}
