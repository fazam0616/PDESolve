#include "../include/grid.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// Grid Metadata Implementation
// ============================================================================

GridMetadata* grid_metadata_create(const uint32_t *dims, 
                                   const double *spacing,
                                   const double *origin, 
                                   int n_dims) {
    GridMetadata *grid = malloc(sizeof(GridMetadata));
    grid->n_dims = n_dims;
    grid->dims = malloc(sizeof(uint32_t) * n_dims);
    grid->spacing = malloc(sizeof(double) * n_dims);
    grid->origin = malloc(sizeof(double) * n_dims);
    grid->extent = malloc(sizeof(double) * n_dims);
    grid->boundaries = malloc(sizeof(BoundarySpec) * n_dims * 2);
    grid->interior_boundaries = NULL;
    grid->n_interior_boundaries = 0;
    grid->refcount = 1; // Initialize reference counter

    // Allocate and initialize strides array for fast indexing
    grid->strides = malloc(sizeof(uint32_t) * n_dims);

    grid->total_points = 1;
    for (int i = 0; i < n_dims; i++) {
        grid->dims[i] = dims[i];
        grid->spacing[i] = spacing[i];
        grid->origin[i] = origin ? origin[i] : 0.0;
        grid->extent[i] = dims[i] * spacing[i];
        grid->total_points *= dims[i];
        
        // Initialize edge boundaries to open (natural continuation)
        grid->boundaries[i*2].type = BC_OPEN;
        grid->boundaries[i*2].value = 0.0;
        grid->boundaries[i*2].func = NULL;
        grid->boundaries[i*2].extrapolation_order = 5;  // Default to cubic
        grid->boundaries[i*2+1].type = BC_OPEN;
        grid->boundaries[i*2+1].value = 0.0;
        grid->boundaries[i*2+1].func = NULL;
        grid->boundaries[i*2+1].extrapolation_order = 5;  // Default to cubic
    }

    // Pre-compute strides: strides[i] = dims[i+1] * dims[i+2] * ... * dims[n-1]
    // For row-major order: linear = sum(indices[i] * strides[i])
    grid->strides[n_dims - 1] = 1;
    for (int i = n_dims - 2; i >= 0; i--) {
        grid->strides[i] = grid->strides[i + 1] * dims[i + 1];
    }

    return grid;
}

void grid_metadata_free(GridMetadata *grid) {
    if (!grid) return;
    if (--grid->refcount > 0) {
        return;
    }
    free(grid->dims);
    free(grid->spacing);
    free(grid->origin);
    free(grid->extent);
    free(grid->boundaries);
    free(grid->strides);  // Free pre-computed strides
    
    // Free interior hyperplane boundaries
    if (grid->interior_boundaries) {
        for (int i = 0; i < grid->n_interior_boundaries; i++) {
            free(grid->interior_boundaries[i].normal);
            free(grid->interior_boundaries[i].point);
            free(grid->interior_boundaries[i].bounds_min);
            free(grid->interior_boundaries[i].bounds_max);
            free(grid->interior_boundaries[i].bbox_min);
            free(grid->interior_boundaries[i].bbox_max);
        }
        free(grid->interior_boundaries);
    }
    
    free(grid);
}

// Increment reference count for GridMetadata
void grid_metadata_retain(GridMetadata *grid) {
    if (grid) {
        grid->refcount++;
    }
}

uint32_t grid_get_total_points(const GridMetadata *grid) {
    if (!grid) return 0;
    return grid->total_points;
}

uint32_t grid_index_to_linear(const GridMetadata *grid, const uint32_t *indices) {
    uint32_t linear = 0;
    uint32_t stride = 1;
    for (int i = grid->n_dims - 1; i >= 0; i--) {
        linear += indices[i] * stride;
        stride *= grid->dims[i];
    }
    return linear;
}

void grid_linear_to_index(const GridMetadata *grid, uint32_t linear, uint32_t *indices) {   
    if (!grid || !indices) return;
    // Zero all N_DIM entries to ensure callers that expect full
    // N_DIM-indexed tensors see well-defined zeroes in higher dims.
    for (int k = 0; k < N_DIM; k++) indices[k] = 0;

    for (int i = grid->n_dims - 1; i >= 0; i--) {
        indices[i] = linear % grid->dims[i];
        linear /= grid->dims[i];
    }
}

void grid_index_to_coord(const GridMetadata *grid, const uint32_t *indices, double *coords) {
    if (!grid || !indices || !coords) return;
    for (int i = 0; i < grid->n_dims; i++) {
        coords[i] = grid->origin[i] + indices[i] * grid->spacing[i];
    }
}

bool grid_coord_to_index(const GridMetadata *grid, const double *coords, uint32_t *indices) {
    if (!grid || !coords || !indices) return false;
    for (int axis = 0; axis < grid->n_dims; axis++) {
        double rel = coords[axis] - grid->origin[axis];
        if (rel < 0.0 || rel > grid->extent[axis]) {
            return false;
        }
        indices[axis] = (uint32_t)round(rel / grid->spacing[axis]);
        if (indices[axis] >= grid->dims[axis]) {
            indices[axis] = grid->dims[axis] - 1;
        }
    }
    return true;
}

bool grid_is_boundary(const GridMetadata *grid, const uint32_t *indices) {
    if (!grid || !indices) return false;
    for (int axis = 0; axis < grid->n_dims; axis++) {
        if (grid->dims[axis] > 1) {
            if (indices[axis] == 0 || indices[axis] == grid->dims[axis] - 1) {
                return true;
            }
        }
    }
    return false;
}

// Helper function: Construct orthonormal basis for hyperplane
// Given a normal vector, creates (n_dims-1) orthonormal vectors that span the hyperplane
// Uses Gram-Schmidt process
static void construct_hyperplane_basis(const double *normal, int n_dims, double **basis_out) {
    // basis_out should be pre-allocated array of (n_dims-1) pointers to double[n_dims]
    
    int n_tangents = n_dims - 1;
    
    // Start with standard basis vectors and orthogonalize against normal
    for (int i = 0; i < n_tangents; i++) {
        // Choose initial vector (standard basis vector i, or i+1 if i aligns with normal)
        int basis_idx = i;
        
        // Check if normal is strongly aligned with this basis vector
        if (fabs(normal[basis_idx]) > 0.9) {
            // Use next basis vector instead to avoid numerical issues
            basis_idx = (basis_idx + 1) % n_dims;
        }
        
        // Start with standard basis vector
        for (int d = 0; d < n_dims; d++) {
            basis_out[i][d] = (d == basis_idx) ? 1.0 : 0.0;
        }
        
        // Orthogonalize against normal
        double dot_with_normal = 0.0;
        for (int d = 0; d < n_dims; d++) {
            dot_with_normal += basis_out[i][d] * normal[d];
        }
        for (int d = 0; d < n_dims; d++) {
            basis_out[i][d] -= dot_with_normal * normal[d];
        }
        
        // Orthogonalize against previous basis vectors (Gram-Schmidt)
        for (int j = 0; j < i; j++) {
            double dot_with_prev = 0.0;
            for (int d = 0; d < n_dims; d++) {
                dot_with_prev += basis_out[i][d] * basis_out[j][d];
            }
            for (int d = 0; d < n_dims; d++) {
                basis_out[i][d] -= dot_with_prev * basis_out[j][d];
            }
        }
        
        // Normalize
        double magnitude = 0.0;
        for (int d = 0; d < n_dims; d++) {
            magnitude += basis_out[i][d] * basis_out[i][d];
        }
        magnitude = sqrt(magnitude);
        
        if (magnitude > 1e-10) {
            for (int d = 0; d < n_dims; d++) {
                basis_out[i][d] /= magnitude;
            }
        }
    }
}

// Check if a point is within the bounded region of a hyperplane
// Returns true if point is both on/near the hyperplane AND within the parametric bounds
// Generalized to work with n-dimensional hyperplanes (n-1 dimensional surfaces in n-space)
static bool point_in_bounded_hyperplane(const HyperplaneBoundary *hb, const double *coords, int n_dims, double tolerance) {
    // Compute signed distance to hyperplane
    double dist = 0.0;
    for (int d = 0; d < n_dims; d++) {
        dist += hb->normal[d] * (coords[d] - hb->point[d]);
    }
    
    // If not near hyperplane, not in bounded region
    if (fabs(dist) > tolerance) {
        return false;
    }
    
    // If unbounded (no parametric bounds), any point on/near hyperplane is valid
    if (!hb->bounds_min || !hb->bounds_max) {
        return true;
    }
    
    // Early rejection using bounding box (if available)
    if (hb->bbox_min && hb->bbox_max) {
        for (int d = 0; d < n_dims; d++) {
            if (coords[d] < hb->bbox_min[d] - tolerance || 
                coords[d] > hb->bbox_max[d] + tolerance) {
                return false;
            }
        }
    }
    
    // General n-dimensional case:
    // Construct orthonormal basis for the (n-1)-dimensional hyperplane
    // Project point onto this basis to get parametric coordinates
    // Check if all parametric coordinates are within bounds
    //

    
    int n_params = n_dims - 1;  // A hyperplane in n-space has (n-1) dimensions
    
    // Allocate basis vectors
    double **basis = malloc(sizeof(double*) * n_params);
    for (int i = 0; i < n_params; i++) {
        basis[i] = malloc(sizeof(double) * n_dims);
    }
    
    // Construct orthonormal basis
    construct_hyperplane_basis(hb->normal, n_dims, basis);
    
    // Compute vector from reference point to query point
    double *vec_to_point = malloc(sizeof(double) * n_dims);
    for (int d = 0; d < n_dims; d++) {
        vec_to_point[d] = coords[d] - hb->point[d];
    }
    
    // Project onto each basis vector to get parametric coordinates
    bool in_bounds = true;
    for (int i = 0; i < n_params; i++) {
        double param = 0.0;
        for (int d = 0; d < n_dims; d++) {
            param += vec_to_point[d] * basis[i][d];
        }
        
        // Check if this parametric coordinate is within bounds
        // The bounds are defined on the hyperplane itself and should work
        // symmetrically from both sides
        if (param < hb->bounds_min[i] || param > hb->bounds_max[i]) {
            in_bounds = false;
            break;
        }
    }
    
    // Clean up
    free(vec_to_point);
    for (int i = 0; i < n_params; i++) {
        free(basis[i]);
    }
    free(basis);
    
    return in_bounds;
}

// ============================================================================
// Grid Field Implementation
// ============================================================================

GridField* grid_field_create(GridMetadata *grid) {
    GridField *field = malloc(sizeof(GridField));
    field->grid = grid;
    grid_metadata_retain(grid); // Increment reference count
    
    // Initialize data as a single tensor with shape matching grid dimensions
    memset(&field->data, 0, sizeof(Literal));
    for (int i = 0; i < grid->n_dims && i < N_DIM; i++) {
        field->data.shape[i] = grid->dims[i];
    }
    // Fill remaining dimensions with 1
    for (int i = grid->n_dims; i < N_DIM; i++) {
        field->data.shape[i] = 1;
    }
    
    // Allocate field data (all zeros initially)
    size_t total = literal_total_elements(&field->data);
    if (total > 0) {
        field->data.field = calloc(total, sizeof(double));
    } else {
        field->data.field = NULL;
    }
    
    if (getenv("GRID_DEBUG")) {
        fprintf(stderr, "[grid] grid_field_create: grid=%p n_dims=%d total_elems=%zu grid_total=%u shape=[",
                (void*)grid, grid->n_dims, total, grid->total_points);
        for (int i = 0; i < N_DIM; i++) {
            fprintf(stderr, "%u%s", field->data.shape[i], (i == N_DIM - 1) ? "]" : ", ");
        }
        fprintf(stderr, "\n");
    }
    return field;
}

void grid_field_free(GridField *field) {
    if (!field) return;
    free(field->data.field);
    grid_metadata_free(field->grid);
    free(field);
}

Literal* grid_field_get(const GridField *field, const uint32_t *indices) {
    if (!field || !indices) {
        printf("Invalid arguments to grid_field_get\n");
        return NULL;
    }
    
    // Validate indices
    for (int i = 0; i < field->grid->n_dims; i++) {
        if (indices[i] >= field->grid->dims[i]) {
            printf("Index out of bounds in grid_field_get\n");
            return NULL;
        }
    }
    
    // Grid indices map directly to tensor indices in data
    // Return scalar value at this grid point
    uint32_t tensor_indices[N_DIM];
    for (int i = 0; i < N_DIM; i++) {
        tensor_indices[i] = (i < field->grid->n_dims) ? indices[i] : 0;
    }
    
    double value = literal_get(&field->data, tensor_indices);
    return literal_create_scalar(value);
}

void grid_field_set(GridField *field, const uint32_t *indices, const Literal *value) {
    if (!field || !indices || !value) {
        printf("Invalid arguments to grid_field_set\n");
        return;
    }
    
    // Validate indices
    for (int i = 0; i < field->grid->n_dims; i++) {
        if (indices[i] >= field->grid->dims[i]) {
            printf("Index out of bounds in grid_field_set\n");
            return;
        }
    }
    
    // Grid indices map directly to tensor indices in data
    // For scalar values, just set the single element
    uint32_t tensor_indices[N_DIM];
    for (int i = 0; i < N_DIM; i++) {
        tensor_indices[i] = (i < field->grid->n_dims) ? indices[i] : 0;
    }
    
    // Get scalar value from literal (handle various shapes)
    double scalar_value = 0.0;
    if (value->field && literal_total_elements(value) > 0) {
        uint32_t zero_idx[N_DIM] = {0};
        scalar_value = literal_get(value, zero_idx);
    }
    
    literal_set(&field->data, tensor_indices, scalar_value);
}

Literal grid_field_evaluate(const GridField *field, const double *coords) {
    if (!field || !coords) {
        Literal empty;
        memset(&empty, 0, sizeof(Literal));
        return empty;
    }
    uint32_t indices[N_DIM];
    for (int i = 0; i < N_DIM; i++) indices[i] = 0;
    if (!grid_coord_to_index(field->grid, coords, indices)) {
        Literal empty;
        memset(&empty, 0, sizeof(Literal));
        return empty;
    }
    Literal *result = grid_field_get(field, indices);
    if (!result) {
        Literal empty;
        memset(&empty, 0, sizeof(Literal));
        return empty;
    }
    Literal value = *result;
    literal_free(result);
    return value;
}

void grid_field_fill(GridField *field, const Literal *value) {
    uint32_t indices[N_DIM];
    for (int i = 0; i < N_DIM; i++) indices[i] = 0;
    if (!field || !value) return;
    for (uint32_t i = 0; i < field->grid->total_points; i++) {
        grid_linear_to_index(field->grid, i, indices);
        grid_field_set(field, indices, value);
    }
}

void grid_field_init_from_function(GridField *field, Literal* (*func)(const double *coords, int n_dims)) {
    if (!field || !func) return;
    GridMetadata *grid = field->grid;
    int n_dims = grid->n_dims;
    uint32_t indices[N_DIM];
    double coords[N_DIM];
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, indices);
        grid_index_to_coord(grid, indices, coords);
        Literal *value_ptr = func(coords, n_dims);
        if (value_ptr) {
            grid_field_set(field, indices, value_ptr);
            literal_free(value_ptr);
        }
    }
    (void)0; // indices/coords are stack-allocated
}

// ============================================================================
// Finite Difference Operators
// ============================================================================

// Thomas algorithm solver for tridiagonal systems
// Solves Ax = d where A is tridiagonal with diagonals (a, b, c)
static void thomas_solve(const double *a, const double *b, const double *c,
                         const double *d, double *x, int n) {
    if (n <= 0) return;
    
    double *c_prime = malloc(n * sizeof(double));
    double *d_prime = malloc(n * sizeof(double));
    
    if (!c_prime || !d_prime) {
        free(c_prime);
        free(d_prime);
        return;
    }
    
    // Forward elimination
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];
    
    for (int i = 1; i < n; i++) {
        double denom = b[i] - a[i] * c_prime[i-1];
        if (fabs(denom) < 1e-14) {
            free(c_prime);
            free(d_prime);
            return;
        }
        c_prime[i] = (i < n-1) ? c[i] / denom : 0.0;
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom;
    }
    
    // Back substitution
    x[n-1] = d_prime[n-1];
    for (int i = n-2; i >= 0; i--) {
        x[i] = d_prime[i] - c_prime[i] * x[i+1];
    }
    
    free(c_prime);
    free(d_prime);
}

// Helper: Set a point in the tridiagonal system to a fixed value
static inline void set_tridiag_fixed(double *a, double *b, double *c, 
                                     double *rhs, uint32_t i, double value) {
    a[i] = 0.0;
    b[i] = 1.0;
    c[i] = 0.0;
    rhs[i] = value;
}

// Helper: Compute boundary derivative value using one-sided or BC-aware stencils
static double compute_boundary_derivative(const GridField *field, 
                                         const uint32_t *idx,
                                         int axis, int order, double h,
                                         bool is_left_edge) {
    GridMetadata *grid = field->grid;
    uint32_t n = grid->dims[axis];
    uint32_t i = idx[axis];
    
    double coords[N_DIM];
    grid_index_to_coord(grid, idx, coords);
    
    BoundarySpec *bc = is_left_edge ? &grid->boundaries[axis * 2] 
                                     : &grid->boundaries[axis * 2 + 1];
    
    uint32_t neighbor_idx[N_DIM];
    memcpy(neighbor_idx, idx, sizeof(uint32_t) * N_DIM);
    
    if (order == 1) {
        // First derivative boundary conditions
        switch (bc->type) {
            case BC_DIRICHLET: {
                double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                neighbor_idx[axis] = is_left_edge ? 1 : n - 2;
                double f_neighbor = literal_get(&field->data, neighbor_idx);
                return is_left_edge ? (f_neighbor - bc_val) / (2.0 * h)
                                    : (bc_val - f_neighbor) / (2.0 * h);
            }
            case BC_NEUMANN: {
                return bc->func ? bc->func(coords, bc->time) : bc->value;
            }
            case BC_REFLECT: {
                return 0.0;
            }
            case BC_OPEN:
            default: {
                double f0 = literal_get(&field->data, idx);
                neighbor_idx[axis] = is_left_edge ? 1 : n - 2;
                double f_neighbor = literal_get(&field->data, neighbor_idx);
                return is_left_edge ? (f_neighbor - f0) / h
                                    : (f0 - f_neighbor) / h;
            }
        }
    } else { // order == 2
        // Second derivative boundary conditions
        switch (bc->type) {
            case BC_DIRICHLET: {
                double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                double f0 = literal_get(&field->data, idx);
                neighbor_idx[axis] = is_left_edge ? 1 : n - 2;
                double f_neighbor = literal_get(&field->data, neighbor_idx);
                return is_left_edge ? (bc_val - 2.0 * f0 + f_neighbor) / (h * h)
                                    : (f_neighbor - 2.0 * f0 + bc_val) / (h * h);
            }
            case BC_NEUMANN: {
                double g = bc->func ? bc->func(coords, bc->time) : bc->value;
                double f0 = literal_get(&field->data, idx);
                neighbor_idx[axis] = is_left_edge ? 1 : n - 2;
                double f_neighbor = literal_get(&field->data, neighbor_idx);
                double sign = is_left_edge ? -1.0 : 1.0;
                return (2.0 * (f_neighbor - f0) + sign * 2.0 * h * g) / (h * h);
            }
            case BC_OPEN: {
                return 0.0;
            }
            case BC_REFLECT:
            default: {
                // One-sided second derivative using 3-point stencil
                if ((is_left_edge && i + 2 < n) || (!is_left_edge && i >= 2)) {
                    uint32_t idx0[N_DIM], idx1[N_DIM], idx2[N_DIM];
                    memcpy(idx0, idx, sizeof(uint32_t) * N_DIM);
                    memcpy(idx1, idx, sizeof(uint32_t) * N_DIM);
                    memcpy(idx2, idx, sizeof(uint32_t) * N_DIM);
                    
                    if (is_left_edge) {
                        idx0[axis] = 0;
                        idx1[axis] = 1;
                        idx2[axis] = 2;
                    } else {
                        idx0[axis] = n - 1;
                        idx1[axis] = n - 2;
                        idx2[axis] = n - 3;
                    }
                    
                    double f0 = literal_get(&field->data, idx0);
                    double f1 = literal_get(&field->data, idx1);
                    double f2 = literal_get(&field->data, idx2);
                    
                    return is_left_edge ? (f2 - 2.0 * f1 + f0) / (h * h)
                                        : (f0 - 2.0 * f1 + f2) / (h * h);
                }
                return 0.0;
            }
        }
    }
}

// Helper: Check if stencil crosses interior boundary and compute value if so
static bool handle_interior_boundary(const GridField *field, const uint32_t *idx,
                                     int axis, double h, int order,
                                     double f0, double f_minus, double f_plus,
                                     double *out_value) {
    if (!field->grid->n_interior_boundaries) return false;
    
    GridMetadata *grid = field->grid;
    uint32_t n = grid->dims[axis];
    uint32_t i = idx[axis];
    
    double coords[N_DIM];
    grid_index_to_coord(grid, idx, coords);
    
    for (int ib = 0; ib < grid->n_interior_boundaries; ib++) {
        HyperplaneBoundary *hb = &grid->interior_boundaries[ib];
        if (!hb->active) continue;
        
        // Quick bbox rejection
        double tolerance = 1.0 * h;
        bool in_bbox = true;
        for (int d = 0; d < grid->n_dims; d++) {
            if (coords[d] < hb->bbox_min[d] - tolerance || 
                coords[d] > hb->bbox_max[d] + tolerance) {
                in_bbox = false;
                break;
            }
        }
        if (!in_bbox) continue;
        
        // Compute signed distances
        double dist_current = 0.0, dist_minus = 0.0, dist_plus = 0.0;
        for (int d = 0; d < grid->n_dims; d++) {
            dist_current += hb->normal[d] * (coords[d] - hb->point[d]);
        }
        
        uint32_t idx_minus[N_DIM], idx_plus[N_DIM];
        memcpy(idx_minus, idx, sizeof(uint32_t) * N_DIM);
        memcpy(idx_plus, idx, sizeof(uint32_t) * N_DIM);
        idx_minus[axis] = (i > 0) ? i - 1 : 0;
        idx_plus[axis] = (i + 1 < n) ? i + 1 : n - 1;
        
        double coords_minus[N_DIM], coords_plus[N_DIM];
        grid_index_to_coord(grid, idx_minus, coords_minus);
        grid_index_to_coord(grid, idx_plus, coords_plus);
        
        for (int d = 0; d < grid->n_dims; d++) {
            dist_minus += hb->normal[d] * (coords_minus[d] - hb->point[d]);
            dist_plus += hb->normal[d] * (coords_plus[d] - hb->point[d]);
        }
        
        // Check if stencil crosses boundary
        if ((dist_minus * dist_plus < 0) || 
            (dist_current * dist_minus < 0) || 
            (dist_current * dist_plus < 0)) {
            
            bool current_in_bounds = point_in_bounded_hyperplane(hb, coords, grid->n_dims, h);
            bool minus_in_bounds = point_in_bounded_hyperplane(hb, coords_minus, grid->n_dims, h);
            bool plus_in_bounds = point_in_bounded_hyperplane(hb, coords_plus, grid->n_dims, h);
            
            if (current_in_bounds || minus_in_bounds || plus_in_bounds) {
                // Handle based on boundary type
                if (hb->bc_spec.type == BC_REFLECT || hb->bc_spec.type == BC_NEUMANN) {
                    double val = 0.0;
                    
                    if (order == 1) {
                        // For first derivative, reflect boundary gives zero
                        val = 0.0;
                    } else {
                        // For second derivative, use one-sided stencil if possible
                        if (fabs(dist_current) < h) {
                            val = 0.0;
                        } else if (dist_current > 0) {
                            if (dist_minus < 0) {
                                // Use forward difference
                                if (i + 2 < n) {
                                    uint32_t idx2[N_DIM];
                                    memcpy(idx2, idx, sizeof(uint32_t) * N_DIM);
                                    idx2[axis] = i + 2;
                                    double f2 = literal_get(&field->data, idx2);
                                    val = (f2 - 2.0 * f_plus + f0) / (h * h);
                                }
                            } else {
                                val = (f_plus - 2.0 * f0 + f_minus) / (h * h);
                            }
                        } else {
                            if (dist_plus > 0) {
                                // Use backward difference
                                if (i >= 2) {
                                    uint32_t idx2[N_DIM];
                                    memcpy(idx2, idx, sizeof(uint32_t) * N_DIM);
                                    idx2[axis] = i - 2;
                                    double f2 = literal_get(&field->data, idx2);
                                    val = (f0 - 2.0 * f_minus + f2) / (h * h);
                                }
                            } else {
                                val = (f_plus - 2.0 * f0 + f_minus) / (h * h);
                            }
                        }
                    }
                    
                    *out_value = val;
                    return true;
                    
                } else if (hb->bc_spec.type == BC_DIRICHLET) {
                    // For Dirichlet interior boundaries, set derivative to zero
                    *out_value = 0.0;
                    return true;
                }
            }
        }
    }
    
    return false;
}

// Compact (Padé) finite difference method for derivatives
// Uses 4th-order implicit scheme with tridiagonal solve
GridField* grid_field_derivative_compact(const GridField *field, int axis, int order) {
    if (!field || axis < 0 || axis >= field->grid->n_dims) return NULL;
    if (order < 1 || order > 2) return NULL;
    
    GridMetadata *grid = field->grid;
    GridField *result = grid_field_create(grid);
    if (!result) return NULL;
    
    double h = grid->spacing[axis];
    uint32_t n = grid->dims[axis];
    
    // Fall back to explicit for very small dimensions
    if (n < 5) {
        grid_field_free(result);
        return grid_field_derivative(field, axis, order);
    }
    
    // Ensure result field is allocated
    if (!result->data.field) {
        size_t total = literal_total_elements(&result->data);
        result->data.field = (double*)calloc(total, sizeof(double));
        if (!result->data.field) {
            grid_field_free(result);
            return NULL;
        }
    }
    
    // Allocate work arrays for tridiagonal system
    double *rhs = malloc(n * sizeof(double));
    double *sol = malloc(n * sizeof(double));
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    
    if (!rhs || !sol || !a || !b || !c) {
        free(rhs); free(sol); free(a); free(b); free(c);
        grid_field_free(result);
        return NULL;
    }
    
    // Set up tridiagonal coefficients for 4th-order Padé scheme
    // For 1st derivative: (1/4)f'_{i-1} + f'_i + (1/4)f'_{i+1} = (3/4h)(f_{i+1} - f_{i-1})
    // For 2nd derivative using explicit RHS (can be upgraded to compact if needed)
    for (uint32_t i = 0; i < n; i++) {
        a[i] = (i > 0) ? 0.25 : 0.0;          // subdiagonal
        b[i] = 1.0;                            // diagonal
        c[i] = (i < n-1) ? 0.25 : 0.0;        // superdiagonal
    }
    
    // Precompute shifted neighbor fields for efficient access
    GridField *gf_plus = grid_field_shift(field, axis, +1);
    GridField *gf_minus = grid_field_shift(field, axis, -1);
    
    // Calculate number of 1D slices perpendicular to axis
    uint32_t slice_count = grid->total_points / n;
    
    // Build array of dimensions excluding the derivative axis
    uint32_t *other_dims = malloc(grid->n_dims * sizeof(uint32_t));
    int n_other_dims = 0;
    for (int d = 0; d < grid->n_dims; d++) {
        if (d != axis) {
            other_dims[n_other_dims++] = grid->dims[d];
        }
    }
    
    uint32_t *idx = malloc(N_DIM * sizeof(uint32_t));
    
    // Process each 1D slice along the specified axis
    for (uint32_t slice_id = 0; slice_id < slice_count; slice_id++) {
        // Initialize all indices to zero
        memset(idx, 0, N_DIM * sizeof(uint32_t));
        
        // Decode slice_id into grid indices (all dimensions except axis)
        uint32_t temp = slice_id;
        int other_idx = 0;
        for (int d = 0; d < grid->n_dims; d++) {
            if (d == axis) {
                idx[d] = 0;  // Will be set in inner loop
            } else {
                idx[d] = temp % other_dims[other_idx];
                temp /= other_dims[other_idx];
                other_idx++;
            }
        }
        
        // Precompute base offset for contiguous memory access
        uint32_t base_offset = 0;
        for (int d = 0; d < grid->n_dims; d++) {
            if (d != axis) {
                base_offset += idx[d] * grid->strides[d];
            }
        }
        uint32_t axis_stride = grid->strides[axis];
        
        // Build RHS for this slice
        for (uint32_t i = 0; i < n; i++) {
            idx[axis] = i;
            
            // Fetch field values efficiently
            double f0 = 0.0, f_minus = 0.0, f_plus = 0.0;
            if (field->data.field) {
                size_t off = base_offset + (size_t)i * axis_stride;
                f0 = field->data.field[off];
                f_minus = (gf_minus && gf_minus->data.field) ? gf_minus->data.field[off] : f0;
                f_plus = (gf_plus && gf_plus->data.field) ? gf_plus->data.field[off] : f0;
            }
            
            // Check for interior boundary handling first
            double ib_value;
            if (handle_interior_boundary(field, idx, axis, h, order, 
                                        f0, f_minus, f_plus, &ib_value)) {
                set_tridiag_fixed(a, b, c, rhs, i, ib_value);
                continue;
            }
            
            // Handle domain edge boundaries
            if (i == 0 || i == n - 1) {
                double val = compute_boundary_derivative(field, idx, axis, order, h, i == 0);
                set_tridiag_fixed(a, b, c, rhs, i, val);
            } else {
                // Interior point - use compact scheme
                if (order == 1) {
                    // 4th-order compact first derivative
                    rhs[i] = (3.0 / (4.0 * h)) * (f_plus - f_minus);
                } else {
                    // Standard explicit second derivative
                    // (Can be upgraded to compact: rhs[i] = (12.0/(10.0*h*h))*(f_plus - 2*f0 + f_minus))
                    rhs[i] = (f_plus - 2.0 * f0 + f_minus) / (h * h);
                }
            }
        }
        
        // Solve tridiagonal system for this slice
        thomas_solve(a, b, c, rhs, sol, n);
        
        // Store solution back into result grid
        for (uint32_t i = 0; i < n; i++) {
            idx[axis] = i;
            literal_set(&result->data, idx, sol[i]);
        }
    }
    
    // Cleanup
    free(other_dims);
    free(idx);
    free(a);
    free(b);
    free(c);
    free(rhs);
    free(sol);
    if (gf_plus) grid_field_free(gf_plus);
    if (gf_minus) grid_field_free(gf_minus);
    
    return result;
}


GridField* grid_field_derivative(const GridField *field, int axis, int order) {
    if (!field || axis < 0 || axis >= field->grid->n_dims) return NULL;
    if (order < 1 || order > 2) return NULL;
    
    GridMetadata *grid = field->grid;
    GridField *result = grid_field_create(grid);
    if (!result) return NULL;
    
    double h = grid->spacing[axis];
    uint32_t n = grid->dims[axis];

    if (n < 3) {
        grid_field_free(result);
        return NULL;
    }

    // Vectorized neighbor fields using grid shifts
    GridField *f_plus = grid_field_shift(field, axis, +1);
    GridField *f_minus = grid_field_shift(field, axis, -1);

    // Compute vectorized result depending on requested order
    GridField *scaled = NULL;
    if (order == 1) {
        // central differences: (f_plus - f_minus) / (2h)
        GridField *diff = grid_field_subtract(f_plus, f_minus);
        if (diff) {
            double factor = 1.0 / (2.0 * h);
            scaled = grid_field_scale(diff, factor);
            grid_field_free(diff);
        }
    } else { // order == 2
        // second derivative: (f_plus - 2*f0 + f_minus) / (h*h)
        if (f_plus && f_minus) {
            GridField *sum = grid_field_add(f_plus, f_minus); // f+ + f-
            GridField *center2 = grid_field_scale(field, -2.0); // -2*f0
            GridField *sum2 = NULL;
            if (sum && center2) {
                sum2 = grid_field_add(sum, center2); // f+ - 2f0 + f-
            }
            if (sum) grid_field_free(sum);
            if (center2) grid_field_free(center2);
            if (sum2) {
                double factor = 1.0 / (h * h);
                scaled = grid_field_scale(sum2, factor);
                grid_field_free(sum2);
            }
        }
    }

    // If we couldn't compute vectorized result, fall back
    if (!scaled) {
        if (f_plus) grid_field_free(f_plus);
        if (f_minus) grid_field_free(f_minus);
        grid_field_free(result);
        return NULL;
    }

    // Replace result data with scaled central difference
    free(result->data.field);
    result->data = scaled->data;
    free(scaled);

    // Now fix boundary points (edges and BCs) using the original per-point logic
    int n_dims = grid->n_dims;
    uint32_t indices[N_DIM];
    uint32_t tensor_idx[N_DIM];
    uint32_t tensor_idx_minus[N_DIM];
    uint32_t tensor_idx_plus[N_DIM];
    double coords[N_DIM];

    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, indices);
        for (int i = 0; i < N_DIM; i++) tensor_idx[i] = (i < n_dims) ? indices[i] : 0;
        uint32_t idx = indices[axis];

        // Boundary handling for first/last index
        if (idx == 0 || idx == n - 1) {
            // Recompute using one-sided formula + BCs similar to original implementation
            grid_index_to_coord(field->grid, indices, coords);
            tensor_idx_minus[0] = tensor_idx[0]; tensor_idx_plus[0] = tensor_idx[0];
            for (int i = 0; i < N_DIM; i++) { tensor_idx_minus[i] = tensor_idx[i]; tensor_idx_plus[i] = tensor_idx[i]; }

            BoundarySpec *bc = (idx == 0) ? &grid->boundaries[axis * 2] : &grid->boundaries[axis * 2 + 1];

            if (order == 1) {
                if (idx == 0) {
                    switch (bc->type) {
                        case BC_DIRICHLET: {
                            double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                            tensor_idx_plus[axis] = idx + 1;
                            double f1 = literal_get(&field->data, tensor_idx_plus);
                            double val = (f1 - bc_val) / (2.0 * h);
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                        case BC_NEUMANN: {
                            double g = bc->func ? bc->func(coords, bc->time) : bc->value;
                            literal_set(&result->data, tensor_idx, g);
                            break;
                        }
                        case BC_REFLECT: {
                            literal_set(&result->data, tensor_idx, 0.0);
                            break;
                        }
                        case BC_OPEN:
                        default: {
                            tensor_idx_plus[axis] = idx + 1;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f1 = literal_get(&field->data, tensor_idx_plus);
                            double val = (f1 - f0) / h;
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                    }
                } else { // idx == n-1
                    switch (bc->type) {
                        case BC_DIRICHLET: {
                            double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                            tensor_idx_minus[axis] = idx - 1;
                            double f_minus = literal_get(&field->data, tensor_idx_minus);
                            double val = (bc_val - f_minus) / (2.0 * h);
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                        case BC_REFLECT: {
                            literal_set(&result->data, tensor_idx, 0.0);
                            break;
                        }
                        case BC_NEUMANN: {
                            double g = bc->func ? bc->func(coords, bc->time) : bc->value;
                            literal_set(&result->data, tensor_idx, g);
                            break;
                        }
                        case BC_OPEN:
                        default: {
                            tensor_idx_minus[axis] = idx - 1;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f_minus = literal_get(&field->data, tensor_idx_minus);
                            double val = (f0 - f_minus) / h;
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                    }
                }
            } else { // order == 2
                if (idx == 0) {
                    switch (bc->type) {
                        case BC_DIRICHLET: {
                            double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                            tensor_idx_plus[axis] = idx + 1;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f1 = literal_get(&field->data, tensor_idx_plus);
                            double val = (bc_val - 2.0 * f0 + f1) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                        case BC_NEUMANN: {
                            tensor_idx_plus[axis] = idx + 1;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f1 = literal_get(&field->data, tensor_idx_plus);
                            double g = bc->func ? bc->func(coords, bc->time) : bc->value;
                            double val = (2.0 * (f1 - f0) - 2.0 * h * g) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                        case BC_OPEN: {
                            literal_set(&result->data, tensor_idx, 0.0);
                            break;
                        }
                        case BC_REFLECT:
                        default: {
                            tensor_idx_plus[axis] = idx + 1;
                            uint32_t tensor_idx_plus2[N_DIM];
                            for (int i = 0; i < N_DIM; i++) tensor_idx_plus2[i] = tensor_idx[i];
                            tensor_idx_plus2[axis] = idx + 2;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f1 = literal_get(&field->data, tensor_idx_plus);
                            double f2 = literal_get(&field->data, tensor_idx_plus2);
                            double val = (f2 - 2.0 * f1 + f0) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                    }
                } else { // idx == n-1
                    switch (bc->type) {
                        case BC_DIRICHLET: {
                            double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                            tensor_idx_minus[axis] = idx - 1;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f_minus = literal_get(&field->data, tensor_idx_minus);
                            double val = (f_minus - 2.0 * f0 + bc_val) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                        case BC_NEUMANN: {
                            tensor_idx_minus[axis] = idx - 1;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f_minus = literal_get(&field->data, tensor_idx_minus);
                            double g = bc->func ? bc->func(coords, bc->time) : bc->value;
                            double val = (2.0 * (f_minus - f0) + 2.0 * h * g) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                        case BC_OPEN: {
                            literal_set(&result->data, tensor_idx, 0.0);
                            break;
                        }
                        case BC_REFLECT:
                        default: {
                            tensor_idx_minus[axis] = idx - 1;
                            uint32_t tensor_idx_minus2[N_DIM];
                            for (int i = 0; i < N_DIM; i++) tensor_idx_minus2[i] = tensor_idx[i];
                            tensor_idx_minus2[axis] = idx - 2;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f_minus = literal_get(&field->data, tensor_idx_minus);
                            double f_minus2 = literal_get(&field->data, tensor_idx_minus2);
                            double val = (f0 - 2.0 * f_minus + f_minus2) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                            break;
                        }
                    }
                }
            }
        }
    }

    // indices/coords are stack-allocated; nothing to free

    grid_field_free(f_plus);
    grid_field_free(f_minus);
    return result;
}

// Update grid_field_laplacian to support N-dimensions and Literal arithmetic
GridField* grid_field_laplacian(const GridField *field) {
    if (!field) return NULL;
    
    GridMetadata *grid = field->grid;
    GridField *result = grid_field_create(grid);
    if (!result) return NULL;
    
    // Initialize result to zero
    size_t total = literal_total_elements(&result->data);
    if (total > 0 && result->data.field) {
        memset(result->data.field, 0, sizeof(double) * total);
    }
    
    // Add second derivatives for each dimension
    // Skip dimensions with size 1 (no variation in that direction)
    for (int axis = 0; axis < grid->n_dims; axis++) {
        // Skip dimensions with only 1 point (no derivative possible)
        if (grid->dims[axis] <= 1) continue;
        
        GridField *d2 = grid_field_derivative(field, axis, 2);
        if (d2) {
            // Add d2 to result using literal_add
            Literal *sum = literal_add(&result->data, &d2->data);
            if (sum) {
                free(result->data.field);
                result->data = *sum;
                free(sum); // Free wrapper
            }
            grid_field_free(d2);
        }
    }
    
    return result;
}

// In-place variant: write Laplacian of `field` into preallocated `out`.
int grid_field_laplacian_into(const GridField *field, GridField *out) {
    if (!field || !out) return 1;
    if (field->grid != out->grid) return 1;

    GridMetadata *grid = field->grid;
    size_t total = literal_total_elements(&out->data);
    if (total == 0 || !out->data.field) return 1;

    // zero out output
    for (size_t i = 0; i < total; i++) out->data.field[i] = 0.0;

    // accumulate second derivatives
    for (int axis = 0; axis < grid->n_dims; axis++) {
        if (grid->dims[axis] <= 1) continue;
        GridField *d2 = grid_field_derivative_compact(field, axis, 2);
        if (!d2) continue;
        size_t t2 = literal_total_elements(&d2->data);
        if (t2 == total && d2->data.field) {
            for (size_t i = 0; i < total; i++) out->data.field[i] += d2->data.field[i];
        }
        grid_field_free(d2);
    }
    return 0;
}

// Compute Laplacian using compact finite differences (4th order)
GridField* grid_field_laplacian_compact(const GridField *field) {
    if (!field) return NULL;
    
    GridMetadata *grid = field->grid;
    GridField *result = grid_field_create(grid);
    if (!result) return NULL;
    
    // Initialize result to zero
    size_t total = literal_total_elements(&result->data);
    if (total > 0 && result->data.field) {
        memset(result->data.field, 0, sizeof(double) * total);
    }
    
    // Add second derivatives for each dimension using compact method
    // Skip dimensions with size 1 (no variation in that direction)
    for (int axis = 0; axis < grid->n_dims; axis++) {
        // Skip dimensions with only 1 point (no derivative possible)
        if (grid->dims[axis] <= 1) continue;
        
        GridField *d2 = grid_field_derivative_compact(field, axis, 2);
        if (d2) {
            // Add d2 to result using literal_add
            Literal *sum = literal_add(&result->data, &d2->data);
            if (sum) {
                free(result->data.field);
                result->data = *sum;
                free(sum); // Free wrapper
            }
            grid_field_free(d2);
        }
    }
    
    return result;
}

// Note: No post-processing with standard Laplacian is performed here; compact
// implementations must be self-consistent and will be compared via tests/benchmarks.

// Update grid_field_gradient to support N-dimensions and Literal arithmetic
GridField** grid_field_gradient(const GridField *field) {
    if (!field) return NULL;
    int n_dims = field->grid->n_dims;
    GridField **gradients = malloc(n_dims * sizeof(GridField*));
    if (!gradients) return NULL;
    for (int axis = 0; axis < n_dims; axis++) {
        gradients[axis] = grid_field_derivative(field, axis, 1);
        if (!gradients[axis]) {
            for (int i = 0; i < axis; i++) {
                grid_field_free(gradients[i]);
            }
            free(gradients);
            return NULL;
        }
    }
    return gradients;
}

// ============================================================================
// Grid Field Operations
// ============================================================================

// Update grid_field_add to use literal_add on data tensors
GridField* grid_field_add(const GridField *a, const GridField *b) {
    if (!a || !b) return NULL;
    if (a->grid != b->grid) return NULL;
    
    GridField *result = grid_field_create(a->grid);
    if (!result) return NULL;
    
    Literal *sum = literal_add(&a->data, &b->data);
    if (sum) {
        free(result->data.field);
        result->data = *sum;
        free(sum); // Free the wrapper, not the field
    }
    
    return result;
}

// Update grid_field_multiply to use literal_multiply on data tensors
GridField* grid_field_multiply(const GridField *a, const GridField *b) {
    if (!a || !b) return NULL;
    if (a->grid != b->grid) return NULL;
    
    GridField *result = grid_field_create(a->grid);
    if (!result) return NULL;
    
    Literal *prod = literal_multiply(&a->data, &b->data);
    if (prod) {
        free(result->data.field);
        result->data = *prod;
        free(prod); // Free the wrapper, not the field
    }
    
    return result;
}

// Update grid_field_scale to use literal_scale on data tensor
GridField* grid_field_scale(const GridField *field, double scalar) {
    if (!field) return NULL;
    
    GridField *result = grid_field_create(field->grid);
    if (!result) return NULL;
    
    Literal *scaled = literal_scale(&field->data, scalar);
    if (scaled) {
        free(result->data.field);
        result->data = *scaled;
        free(scaled); // Free the wrapper, not the field
    }
    
    return result;
}

void grid_field_scale_inplace(GridField *field, double scalar) {
    if (!field || !field->data.field) return;
    uint64_t size = literal_total_elements(&field->data);
    for (uint64_t i = 0; i < size; i++) field->data.field[i] *= scalar;
}

int grid_field_axpy(GridField *y, double a, const GridField *x) {
    if (!y || !x) return 1;
    if (y->grid != x->grid) return 1;
    if (!y->data.field || !x->data.field) return 1;
    uint64_t size = literal_total_elements(&y->data);
    for (uint64_t i = 0; i < size; i++) y->data.field[i] += a * x->data.field[i];
    return 0;
}

int grid_field_pointwise_multiply_into(const GridField *a, const GridField *b, GridField *out) {
    if (!a || !b || !out) return 1;
    if (a->grid != b->grid || a->grid != out->grid) return 1;
    if (!a->data.field || !b->data.field || !out->data.field) return 1;
    uint64_t size = literal_total_elements(&out->data);
    for (uint64_t i = 0; i < size; i++) out->data.field[i] = a->data.field[i] * b->data.field[i];
    return 0;
}

int grid_field_copy_into(const GridField *src, GridField *dst) {
    if (!src || !dst) return 1;
    if (src->grid != dst->grid) return 1;
    uint64_t size = literal_total_elements(&src->data);
    if (!src->data.field || !dst->data.field) return 1;
    memcpy(dst->data.field, src->data.field, sizeof(double) * size);
    return 0;
}

// Update grid_field_norm to use literal_norm on data tensor
double grid_field_norm(const GridField *field) {
    if (!field) return 0.0;
    return literal_norm(&field->data);
}

// Update grid_field_copy to deep copy data tensor
GridField* grid_field_copy(const GridField *field) {
    if (!field) return NULL;
    
    GridField *copy = grid_field_create(field->grid);
    if (!copy) return NULL;
    
    // Deep copy the data tensor
    free(copy->data.field);
    copy->data = field->data;
    
    size_t total = literal_total_elements(&field->data);
    if (total > 0 && field->data.field) {
        copy->data.field = malloc(sizeof(double) * total);
        memcpy(copy->data.field, field->data.field, sizeof(double) * total);
    } else {
        copy->data.field = NULL;
    }
    
    return copy;
}

// Element-wise subtraction of two grid fields
GridField* grid_field_subtract(const GridField *a, const GridField *b) {
    if (!a || !b) return NULL;
    if (a->grid != b->grid) return NULL;

    GridField *result = grid_field_create(a->grid);
    if (!result) return NULL;

    Literal *diff = literal_subtract((Literal*)&a->data, (Literal*)&b->data);
    if (diff) {
        free(result->data.field);
        result->data = *diff;
        free(diff);
    }
    return result;
}

// Shift a grid field along a given axis by integer steps (positive shifts toward +axis).
// This uses contiguous block copies to avoid per-element loops when possible.
GridField* grid_field_shift(const GridField *field, int axis, int shift) {
    if (!field) return NULL;
    if (axis < 0 || axis >= field->grid->n_dims) return NULL;
    GridMetadata *grid = field->grid;
    uint32_t n = grid->dims[axis];

    GridField *out = grid_field_create(grid);
    if (!out) return NULL;

    // If source is all-zero, return zeroed out field
    if (!field->data.field) return out;

    double *src = field->data.field;
    double *dst = out->data.field;
    uint32_t total = grid->total_points;

    uint32_t slice_count = total / n;

    // Build list of other dimensions
    uint32_t *other_dims = malloc(grid->n_dims * sizeof(uint32_t));
    int n_other = 0;
    for (int d = 0; d < grid->n_dims; d++) {
        if (d == axis) continue;
        other_dims[n_other++] = grid->dims[d];
    }

    uint32_t *indices = malloc(N_DIM * sizeof(uint32_t));
    uint32_t *src_indices = malloc(N_DIM * sizeof(uint32_t));
    for (uint32_t linear = 0; linear < total; linear++) {
        // decode linear index into multi-index
        uint32_t tmp = linear;
        for (int d = 0; d < grid->n_dims; d++) {
            uint32_t s = grid->strides[d];
            indices[d] = tmp / s;
            tmp = tmp % s;
        }

        // build source indices (apply shift: positive shift reads from higher index)
        for (int d = 0; d < grid->n_dims; d++) src_indices[d] = indices[d];
        int src_k = (int)indices[axis] + shift;
        if (src_k < 0 || src_k >= (int)n) {
            dst[linear] = 0.0;
            continue;
        }
        src_indices[axis] = (uint32_t)src_k;

        // compute source linear index
        uint32_t src_lin = 0;
        for (int d = 0; d < grid->n_dims; d++) src_lin += src_indices[d] * grid->strides[d];
        dst[linear] = src[src_lin];
    }

    free(other_dims);
    free(indices);
    free(src_indices);
    return out;
}

// ============================================================================
// Axis Naming and Grid-Literal Helpers
// ============================================================================

// Standard axis names: x, y, z, w, r, s, t, u, v, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q
static const char* AXIS_NAMES[] = {
    "x", "y", "z", "w", "r", "s", "t", "u", "v",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q"
};
static const int NUM_AXIS_NAMES = sizeof(AXIS_NAMES) / sizeof(AXIS_NAMES[0]);

const char* grid_axis_name(int axis) {
    if (axis < 0 || axis >= NUM_AXIS_NAMES) {
        return "?";
    }
    return AXIS_NAMES[axis];
}

int grid_axis_from_name(const char *name) {
    if (!name) return -1;
    for (int i = 0; i < NUM_AXIS_NAMES; i++) {
        if (strcmp(name, AXIS_NAMES[i]) == 0) {
            return i;
        }
    }
    return -1;
}

bool grid_literal_matches(const Literal *lit, const GridMetadata *grid) {
    if (!lit || !grid) return false;
    
    // Check if literal shape matches grid dimensions
    for (int i = 0; i < grid->n_dims && i < N_DIM; i++) {
        if (lit->shape[i] != grid->dims[i]) {
            return false;
        }
    }
    
    // Remaining dimensions should be 1
    for (int i = grid->n_dims; i < N_DIM; i++) {
        if (lit->shape[i] != 1) {
            return false;
        }
    }
    
    return true;
}

GridField* grid_field_wrap_literal(Literal *lit, GridMetadata *grid) {
    if (!lit || !grid) return NULL;
    
    if (!grid_literal_matches(lit, grid)) {
        fprintf(stderr, "Error: Literal shape does not match grid dimensions\n");
        return NULL;
    }
    
    GridField *field = malloc(sizeof(GridField));
    if (!field) return NULL;
    
    field->name = NULL;
    field->grid = grid;
    grid_metadata_retain(grid);
    
    // Shallow copy of literal data (don't allocate new field array)
    field->data = *lit;
    
    return field;
}

int grid_field_exponent_inplace(GridField *field, const Literal *exponent_literal) {
    if (!field || !exponent_literal) return -1;
    // Copy base and exponent to pass to literal_pow (which expects mutable pointers)
    Literal *base = literal_copy(&field->data);
    if (!base) return -1;
    Literal *exp_copy = literal_copy((Literal*)exponent_literal);
    if (!exp_copy) { literal_free(base); return -1; }
    Literal *res = literal_pow(base, exp_copy);
    literal_free(base); literal_free(exp_copy);
    if (!res) return -1;
    // Replace field->data with res (steal data pointer)
    if (field->data.field) free(field->data.field);
    field->data.field = res->field;
    for (int i = 0; i < N_DIM; i++) field->data.shape[i] = res->shape[i];
    free(res);
    return 0;
}

// ============================================================================
// Boundary Condition Configuration
// ============================================================================

void grid_set_boundary(GridMetadata *grid, int axis, int side, 
                       BoundaryType type, double value) {
    if (!grid || axis < 0 || axis >= grid->n_dims || side < 0 || side > 1) {
        fprintf(stderr, "Error: Invalid boundary specification\n");
        return;
    }
    
    int idx = axis * 2 + side;
    grid->boundaries[idx].type = type;
    grid->boundaries[idx].value = value;
    grid->boundaries[idx].func = NULL;
    grid->boundaries[idx].time = 0.0;
    
    if (type == BC_REFLECT) {
        grid->boundaries[idx].reflection_coeff = value;
    }
    
    // Set default extrapolation order for BC_OPEN
    if (type == BC_OPEN) {
        grid->boundaries[idx].extrapolation_order = 5;  // Default to cubic
    }
}

void grid_set_open_boundary(GridMetadata *grid, int axis, int side, int order) {
    if (!grid || axis < 0 || axis >= grid->n_dims || side < 0 || side > 1) {
        fprintf(stderr, "Error: Invalid boundary specification\n");
        return;
    }
    
    if (order < 1) {
        fprintf(stderr, "Error: Extrapolation order must be at least 1\n");
        return;
    }
    
    int idx = axis * 2 + side;
    grid->boundaries[idx].type = BC_OPEN;
    grid->boundaries[idx].value = 0.0;
    grid->boundaries[idx].func = NULL;
    grid->boundaries[idx].time = 0.0;
    grid->boundaries[idx].extrapolation_order = order;
}

void grid_set_boundary_func(GridMetadata *grid, int axis, int side,
                            BoundaryType type, BCFunction func) {
    if (!grid || axis < 0 || axis >= grid->n_dims || side < 0 || side > 1) {
        fprintf(stderr, "Error: Invalid boundary specification\n");
        return;
    }
    
    if (type != BC_DIRICHLET && type != BC_NEUMANN) {
        fprintf(stderr, "Error: Function-based BCs only supported for Dirichlet/Neumann\n");
        return;
    }
    
    int idx = axis * 2 + side;
    grid->boundaries[idx].type = type;
    grid->boundaries[idx].func = func;
    grid->boundaries[idx].value = 0.0;
    grid->boundaries[idx].time = 0.0;
}

void grid_set_robin_boundary(GridMetadata *grid, int axis, int side,
                             double alpha, double beta, double gamma) {
    if (!grid || axis < 0 || axis >= grid->n_dims || side < 0 || side > 1) {
        fprintf(stderr, "Error: Invalid boundary specification\n");
        return;
    }
    
    int idx = axis * 2 + side;
    grid->boundaries[idx].type = BC_ROBIN;
    grid->boundaries[idx].alpha = alpha;
    grid->boundaries[idx].beta = beta;
    grid->boundaries[idx].gamma = gamma;
    grid->boundaries[idx].func = NULL;
    grid->boundaries[idx].value = 0.0;
}

int grid_add_hyperplane_boundary(GridMetadata *grid,
                                 const double *normal,
                                 const double *point,
                                 const double *bounds_min,
                                 const double *bounds_max,
                                 BoundaryType type,
                                 double value) {
    if (!grid || !normal || !point) return -1;
    
    // Expand interior boundaries array
    int new_idx = grid->n_interior_boundaries;
    grid->n_interior_boundaries++;
    grid->interior_boundaries = realloc(grid->interior_boundaries,
                                       sizeof(HyperplaneBoundary) * grid->n_interior_boundaries);
    
    HyperplaneBoundary *hb = &grid->interior_boundaries[new_idx];
    
    // Allocate and copy normal vector
    hb->normal = malloc(sizeof(double) * grid->n_dims);
    double norm_magnitude = 0.0;
    for (int i = 0; i < grid->n_dims; i++) {
        hb->normal[i] = normal[i];
        norm_magnitude += normal[i] * normal[i];
    }
    norm_magnitude = sqrt(norm_magnitude);
    // Normalize
    for (int i = 0; i < grid->n_dims; i++) {
        hb->normal[i] /= norm_magnitude;
    }
    
    // Copy point
    hb->point = malloc(sizeof(double) * grid->n_dims);
    for (int i = 0; i < grid->n_dims; i++) {
        hb->point[i] = point[i];
    }
    
    // Copy bounds (can be NULL for unbounded)
    if (bounds_min && bounds_max) {
        hb->bounds_min = malloc(sizeof(double) * (grid->n_dims - 1));
        hb->bounds_max = malloc(sizeof(double) * (grid->n_dims - 1));
        for (int i = 0; i < grid->n_dims - 1; i++) {
            hb->bounds_min[i] = bounds_min[i];
            hb->bounds_max[i] = bounds_max[i];
        }
        
        // Compute world-space bounding box
        // Construct orthonormal basis and evaluate corner points
        hb->bbox_min = malloc(sizeof(double) * grid->n_dims);
        hb->bbox_max = malloc(sizeof(double) * grid->n_dims);
        
        int n_params = grid->n_dims - 1;
        double **basis = malloc(sizeof(double*) * n_params);
        for (int i = 0; i < n_params; i++) {
            basis[i] = malloc(sizeof(double) * grid->n_dims);
        }
        construct_hyperplane_basis(hb->normal, grid->n_dims, basis);
        
        // Initialize bbox with reference point
        for (int i = 0; i < grid->n_dims; i++) {
            hb->bbox_min[i] = hb->point[i];
            hb->bbox_max[i] = hb->point[i];
        }
        
        // Enumerate corners of parametric domain and expand bbox
        int n_corners = 1 << n_params;  // 2^n_params corners
        for (int corner = 0; corner < n_corners; corner++) {
            double corner_world[grid->n_dims];
            for (int d = 0; d < grid->n_dims; d++) {
                corner_world[d] = hb->point[d];
            }
            
            // Add parametric displacement for this corner
            for (int param = 0; param < n_params; param++) {
                double param_val = (corner & (1 << param)) ? hb->bounds_max[param] : hb->bounds_min[param];
                for (int d = 0; d < grid->n_dims; d++) {
                    corner_world[d] += param_val * basis[param][d];
                }
            }
            
            // Expand bounding box
            for (int d = 0; d < grid->n_dims; d++) {
                if (corner_world[d] < hb->bbox_min[d]) hb->bbox_min[d] = corner_world[d];
                if (corner_world[d] > hb->bbox_max[d]) hb->bbox_max[d] = corner_world[d];
            }
        }
        
        // Clean up basis
        for (int i = 0; i < n_params; i++) {
            free(basis[i]);
        }
        free(basis);
    } else {
        hb->bounds_min = NULL;
        hb->bounds_max = NULL;
        // Unbounded: use grid extents as bbox
        hb->bbox_min = malloc(sizeof(double) * grid->n_dims);
        hb->bbox_max = malloc(sizeof(double) * grid->n_dims);
        for (int i = 0; i < grid->n_dims; i++) {
            hb->bbox_min[i] = grid->origin[i];
            hb->bbox_max[i] = grid->origin[i] + grid->extent[i];
        }
    }
    
    // Set boundary condition
    hb->bc_spec.type = type;
    hb->bc_spec.value = value;
    hb->bc_spec.func = NULL;
    hb->bc_spec.time = 0.0;
    hb->active = true;
    
    if (type == BC_REFLECT) {
        hb->bc_spec.reflection_coeff = value;
    }
    
    return new_idx;
}

int grid_add_hyperplane_boundary_func(GridMetadata *grid,
                                      const double *normal,
                                      const double *point,
                                      const double *bounds_min,
                                      const double *bounds_max,
                                      BoundaryType type,
                                      BCFunction func) {
    int idx = grid_add_hyperplane_boundary(grid, normal, point, bounds_min, bounds_max, type, 0.0);
    if (idx >= 0) {
        grid->interior_boundaries[idx].bc_spec.func = func;
    }
    return idx;
}

int grid_point_near_boundary(const GridMetadata *grid, const double *coords) {
    if (!grid || !coords) return 0;
    
    const double tolerance = 1e-10;
    
    for (int i = 0; i < grid->n_interior_boundaries; i++) {
        if (!grid->interior_boundaries[i].active) continue;
        
        HyperplaneBoundary *hb = &grid->interior_boundaries[i];
        
        // Use the proper bounded hyperplane checking function
        if (point_in_bounded_hyperplane(hb, coords, grid->n_dims, tolerance)) {
            return i + 1;  // Return 1-indexed boundary ID
        }
    }
    
    return 0;  // Interior point
}

void grid_update_bc_time(GridMetadata *grid, double t) {
    if (!grid) return;
    
    // Update edge boundaries
    for (int i = 0; i < grid->n_dims * 2; i++) {
        grid->boundaries[i].time = t;
    }
    
    // Update interior boundaries
    for (int i = 0; i < grid->n_interior_boundaries; i++) {
        grid->interior_boundaries[i].bc_spec.time = t;
    }
}

// ============================================================================
// Tensor Field — CPU-side implementation
// GL-side operations (upload, download, SSBO deletion) are in gpu_tensor.c
// so that grid.c stays free of GL header dependencies.
// ============================================================================

TensorField* tensor_field_create(int rank, const int *shape) {
    assert(rank >= 1 && rank <= TENSOR_MAX_RANK);
    TensorField *tf = calloc(1, sizeof(TensorField));
    if (!tf) return NULL;
    tf->rank = rank;

    size_t total = 1;
    for (int i = 0; i < rank; i++) {
        assert(shape[i] > 0);
        tf->shape[i] = shape[i];
        total *= (size_t)shape[i];
    }

    /* Compute row-major strides: strides[rank-1] = 1,
       strides[i] = shape[i+1] * strides[i+1]               */
    tf->strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--)
        tf->strides[i] = tf->strides[i + 1] * tf->shape[i + 1];

    tf->total      = total;
    tf->data       = calloc(total, sizeof(double));
    tf->ssbo       = 0;
    tf->gpu_dirty  = true;
    return tf;
}

void tensor_field_free(TensorField *tf) {
    if (!tf) return;
    free(tf->name);
    free(tf->data);
    /* Note: ssbo handle, if non-zero, must have been deleted while a GL
       context is current (call tensor_field_delete_ssbo or
       gpu_tensor_field_free beforehand).                                */
    free(tf);
}

void tensor_field_delete_ssbo(TensorField *tf) {
    /* This function body is intentionally left as a no-op stub: the actual
       glDeleteBuffers call lives in gpu_tensor.c where GL headers are
       available.  Callers that hold a GL context should use
       gpu_tensor_field_free() instead of tensor_field_free().           */
    if (!tf) return;
    /* ssbo deletion deferred to gpu_tensor_field_free */
}

size_t tensor_field_linear_index(const TensorField *tf, const int *indices) {
    size_t off = 0;
    for (int i = 0; i < tf->rank; i++)
        off += (size_t)indices[i] * (size_t)tf->strides[i];
    return off;
}

double tensor_field_get(const TensorField *tf, const int *indices) {
    if (!tf || !tf->data) return 0.0;
    return tf->data[tensor_field_linear_index(tf, indices)];
}

void tensor_field_set(TensorField *tf, const int *indices, double value) {
    if (!tf) return;
    if (!tf->data) tf->data = calloc(tf->total, sizeof(double));
    if (!tf->data) return;
    tf->data[tensor_field_linear_index(tf, indices)] = value;
    tf->gpu_dirty = true;
}
