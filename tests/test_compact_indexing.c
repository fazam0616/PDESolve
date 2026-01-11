#include <stdio.h>
#include <stdlib.h>
#include "../include/grid.h"
#include "../include/literal.h"

void test_slice_indexing() {
    // Create a simple 3D grid like the simulator uses: [240, 240, 1]
    uint32_t dims[] = {240, 240, 1};
    double spacing[] = {0.01, 0.01, 1.0};
    double origin[] = {0.0, 0.0, 0.0};
    
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 3);
    
    printf("Grid dimensions: [%u, %u, %u]\n", dims[0], dims[1], dims[2]);
    printf("Total points: %u\n", grid->total_points);
    printf("Strides: [%u, %u, %u]\n", grid->strides[0], grid->strides[1], grid->strides[2]);
    
    // Test slice iteration for axis=0
    int axis = 0;
    uint32_t n = grid->dims[axis];
    uint32_t slice_count = grid->total_points / n;
    
    printf("\nTesting axis=%d (n=%u, slice_count=%u)\n", axis, n, slice_count);
    
    // Build other_dims array (current implementation)
    uint32_t *other_dims = malloc(grid->n_dims * sizeof(uint32_t));
    int n_other_dims = 0;
    for (int d = 0; d < grid->n_dims; d++) {
        if (d != axis) {
            other_dims[n_other_dims++] = grid->dims[d];
        }
    }
    
    printf("Other dims: [");
    for (int i = 0; i < n_other_dims; i++) {
        printf("%u%s", other_dims[i], i < n_other_dims-1 ? ", " : "");
    }
    printf("]\n\n");
    
    // Test first few slice IDs
    uint32_t *idx = malloc(grid->n_dims * sizeof(uint32_t));
    for (uint32_t slice_id = 0; slice_id < 5 && slice_id < slice_count; slice_id++) {
        uint32_t temp = slice_id;
        int other_idx = 0;
        for (int d = 0; d < grid->n_dims; d++) {
            if (d == axis) {
                idx[d] = 0;
            } else {
                idx[d] = temp % other_dims[other_idx];
                temp /= other_dims[other_idx];
                other_idx++;
            }
        }
        
        // Compute linear index
        uint32_t linear = 0;
        for (int d = 0; d < grid->n_dims; d++) {
            linear += idx[d] * grid->strides[d];
        }
        
        printf("slice_id=%u -> idx=[%u, %u, %u] -> linear=%u\n",
               slice_id, idx[0], idx[1], idx[2], linear);
        
        // Check if linear index is in bounds
        if (linear >= grid->total_points) {
            printf("  ERROR: linear index %u out of bounds (total=%u)\n", 
                   linear, grid->total_points);
        }
    }
    
    // Test last few slice IDs
    printf("\nLast few slices:\n");
    for (uint32_t slice_id = slice_count > 5 ? slice_count - 5 : 0; 
         slice_id < slice_count; slice_id++) {
        uint32_t temp = slice_id;
        int other_idx = 0;
        for (int d = 0; d < grid->n_dims; d++) {
            if (d == axis) {
                idx[d] = 0;
            } else {
                idx[d] = temp % other_dims[other_idx];
                temp /= other_dims[other_idx];
                other_idx++;
            }
        }
        
        uint32_t linear = 0;
        for (int d = 0; d < grid->n_dims; d++) {
            linear += idx[d] * grid->strides[d];
        }
        
        printf("slice_id=%u -> idx=[%u, %u, %u] -> linear=%u\n",
               slice_id, idx[0], idx[1], idx[2], linear);
        
        if (linear >= grid->total_points) {
            printf("  ERROR: linear index %u out of bounds (total=%u)\n", 
                   linear, grid->total_points);
        }
    }
    
    // Now test varying along the axis
    printf("\nTesting points along axis at slice_id=0:\n");
    for (int d = 0; d < grid->n_dims; d++) {
        idx[d] = 0;
    }
    
    for (uint32_t i = 0; i < 5 && i < n; i++) {
        idx[axis] = i;
        uint32_t linear = 0;
        for (int d = 0; d < grid->n_dims; d++) {
            linear += idx[d] * grid->strides[d];
        }
        printf("  i=%u -> idx=[%u, %u, %u] -> linear=%u\n",
               i, idx[0], idx[1], idx[2], linear);
        
        if (linear >= grid->total_points) {
            printf("    ERROR: linear index out of bounds\n");
        }
    }
    
    free(idx);
    free(other_dims);
    grid_metadata_free(grid);
}

int main() {
    printf("Testing Compact Derivative Slice Indexing\n");
    printf("==========================================\n\n");
    
    test_slice_indexing();
    
    printf("\nTest complete.\n");
    return 0;
}
