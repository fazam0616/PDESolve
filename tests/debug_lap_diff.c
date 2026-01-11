#include "../include/grid.h"
#include "../include/literal.h"
#include <stdio.h>

int main() {
    uint32_t dims[3] = {21,21,1};
    double spacing[3] = {0.1,0.1,1.0};
    double origin[3] = {0,0,0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    GridField *f = grid_field_create(grid);
    grid_field_init_from_function(f, (Literal*(*)(const double*,int))0);
    // Initialize f(x,y)=x^2 + y^2 manually
    uint32_t indices[3];
    double coords[3];
    for (uint32_t j=0;j<dims[1];j++){
        for (uint32_t i=0;i<dims[0];i++){
            indices[0]=i; indices[1]=j; indices[2]=0;
            grid_index_to_coord(grid, indices, coords);
            double v = coords[0]*coords[0] + coords[1]*coords[1];
            Literal *s = literal_create_scalar(v);
            grid_field_set(f, indices, s);
            literal_free(s);
        }
    }
    grid_set_boundary(grid, 0, 0, BC_DIRICHLET, 0.0);
    grid_set_boundary(grid, 0, 1, BC_NEUMANN, 0.0);
    grid_set_boundary(grid, 1, 0, BC_REFLECT, 0.0);
    grid_set_boundary(grid, 1, 1, BC_OPEN, 0.0);

    GridField *lap = grid_field_laplacian(f);
    GridField *lapc = grid_field_laplacian_compact(f);

    printf("i j  lap  lapc  diff\n");
    for (uint32_t j=0;j<dims[1];j++){
        for (uint32_t i=0;i<dims[0];i++){
            if (i!=0 && i!=dims[0]-1 && j!=0 && j!=dims[1]-1) continue;
            indices[0]=i; indices[1]=j; indices[2]=0;
            Literal *a = grid_field_get(lap, indices);
            Literal *b = grid_field_get(lapc, indices);
            double va = a?literal_get(a,(uint32_t[]){0,0,0}):0.0;
            double vb = b?literal_get(b,(uint32_t[]){0,0,0}):0.0;
            if (a) literal_free(a); if (b) literal_free(b);
            printf("%u %u  %g  %g  %g\n", i,j,va,vb,fabs(va-vb));
        }
    }
    grid_field_free(lap); grid_field_free(lapc); grid_field_free(f); grid_metadata_free(grid);
    return 0;
}
