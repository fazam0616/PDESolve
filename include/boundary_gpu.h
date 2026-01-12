#ifndef BOUNDARY_GPU_H
#define BOUNDARY_GPU_H

#include "grid.h"
#include "gpu_compiler.h"

typedef enum {
    BC_STRATEGY_MASK,
    BC_STRATEGY_INLINE,
    BC_STRATEGY_GHOST_CELLS,
    BC_STRATEGY_HYBRID
} BCStrategy;

typedef struct BoundaryMask {
    uint8_t *mask;              // Boundary mask (1 = boundary, 0 = interior)
    double *values;             // Dirichlet values
    int *types;                 // BC type per point
    int *priority;              // per-point priority (lower wins)
    GridMetadata *grid;
    unsigned int mask_tex;      // GL texture id (if uploaded)
    unsigned int values_tex;    // values texture id
} BoundaryMask;

BoundaryMask* boundary_mask_create(GridMetadata *grid);
int boundary_mask_upload(BoundaryMask *bm, GPUContext *ctx);
// Apply mask on GPU by rendering input_tex through mask/values into currently bound FBO.
// input_tex: GL texture id of the uploaded input field.
int boundary_mask_apply(BoundaryMask *bm, GPUContext *ctx, unsigned int input_tex);
// CPU-side apply: writes Dirichlet values into the provided GridField where mask is set
int boundary_mask_apply_cpu(BoundaryMask *bm, GridField *field);
void boundary_mask_update_time(BoundaryMask *bm, double t);
void boundary_mask_free(BoundaryMask *bm);

#endif // BOUNDARY_GPU_H
