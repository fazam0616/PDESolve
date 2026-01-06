#include "../include/grid.h"
#include <stdlib.h>
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
    
    // Free interior hyperplane boundaries
    if (grid->interior_boundaries) {
        for (int i = 0; i < grid->n_interior_boundaries; i++) {
            free(grid->interior_boundaries[i].normal);
            free(grid->interior_boundaries[i].point);
            free(grid->interior_boundaries[i].bounds_min);
            free(grid->interior_boundaries[i].bounds_max);            free(grid->interior_boundaries[i].bbox_min);
            free(grid->interior_boundaries[i].bbox_max);        }
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
    uint32_t indices[field->grid->n_dims];
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
    uint32_t indices[field->grid->n_dims];
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
    uint32_t *indices = malloc(sizeof(uint32_t) * n_dims);
    double *coords = malloc(sizeof(double) * n_dims);
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, indices);
        grid_index_to_coord(grid, indices, coords);
        Literal *value_ptr = func(coords, n_dims);
        if (value_ptr) {
            grid_field_set(field, indices, value_ptr);
            literal_free(value_ptr);
        }
    }
    free(indices);
    free(coords);
}

// ============================================================================
// Finite Difference Operators
// ============================================================================

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
    
    int n_dims = grid->n_dims;
    uint32_t *indices = malloc(sizeof(uint32_t) * n_dims);
    uint32_t *tensor_idx = malloc(sizeof(uint32_t) * N_DIM);
    uint32_t *tensor_idx_minus = malloc(sizeof(uint32_t) * N_DIM);
    uint32_t *tensor_idx_plus = malloc(sizeof(uint32_t) * N_DIM);
    double *coords = malloc(sizeof(double) * n_dims);
    
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, indices);
        
        // Convert to tensor indices
        for (int i = 0; i < N_DIM; i++) {
            tensor_idx[i] = (i < n_dims) ? indices[i] : 0;
            tensor_idx_minus[i] = tensor_idx[i];
            tensor_idx_plus[i] = tensor_idx[i];
        }
        
        uint32_t idx = indices[axis];
        
        // First check interior boundaries
        bool boundary_handled = false;
        double boundary_val = 0.0;
        
        // Get physical coordinates for boundary checking
        grid_index_to_coord(field->grid, indices, coords);
        
        // Check interior hyperplane boundaries with bounding box optimization
        for (int ib = 0; ib < field->grid->n_interior_boundaries && !boundary_handled; ib++) {
            HyperplaneBoundary *hb = &field->grid->interior_boundaries[ib];
            if (!hb->active) continue;
            
            // Bounding box rejection test (with tolerance for stencil)
            double tolerance = 1.0 * h;  // Stencil width
            bool in_bbox = true;
            for (int d = 0; d < field->grid->n_dims; d++) {
                if (coords[d] < hb->bbox_min[d] - tolerance || 
                    coords[d] > hb->bbox_max[d] + tolerance) {
                    in_bbox = false;
                    break;
                }
            }
            if (!in_bbox) continue;
            
            // Compute signed distance to hyperplane: d = n · (p - p0)
            double dist = 0.0;
            for (int d = 0; d < field->grid->n_dims; d++) {
                dist += hb->normal[d] * (coords[d] - hb->point[d]);
            }
            
            // Check if near boundary (within stencil width)
            if (fabs(dist) > tolerance) continue;
            
            // Check if point is within the bounded segment
            if (!point_in_bounded_hyperplane(hb, coords, field->grid->n_dims, tolerance)) {
                continue;
            }
            
            // Apply boundary condition based on type
            if (order == 1) {
                // First derivative
                if (hb->bc_spec.type == BC_REFLECT || hb->bc_spec.type == BC_NEUMANN) {
                    // Zero normal derivative: ∂u/∂n = 0
                    // For reflecting barriers, apply boundary condition from both sides
                    // by checking if the derivative direction has ANY component along the normal
                    double normal_component = fabs(hb->normal[axis]);
                    if (normal_component > 0.01) {  // Any non-zero alignment
                        boundary_val = 0.0;
                        boundary_handled = true;
                        // Debug: First occurrence only
                        static int first_call = 1;
                        if (first_call) {
                            printf("[DEBUG] Interior boundary handling: order=1, BC_REFLECT at boundary %d\n", ib);
                            printf("        coords=(%.3f, %.3f), normal=(%.3f, %.3f), axis=%d\n",
                                   coords[0], coords[1], hb->normal[0], hb->normal[1], axis);
                            first_call = 0;
                        }
                    }
                } else if (hb->bc_spec.type == BC_DIRICHLET) {
                    // For Dirichlet, enforce derivative using ghost point
                    // Similar to edge handling below
                    double bc_val = hb->bc_spec.func ? hb->bc_spec.func(coords, hb->bc_spec.time) : hb->bc_spec.value;
                    double normal_component = hb->normal[axis];
                    if (fabs(normal_component) > 0.1) {
                        // Approximate derivative at boundary
                        // For now, set to zero (will improve with proper ghost point)
                        boundary_val = 0.0;
                        boundary_handled = true;
                    }
                }
            } else if (order == 2) {
                // Second derivative
                if (hb->bc_spec.type == BC_REFLECT || hb->bc_spec.type == BC_NEUMANN) {
                    // For reflecting boundary, second derivative along normal is zero
                    // Apply from both sides by using absolute value
                    double normal_component = fabs(hb->normal[axis]);
                    if (normal_component > 0.01) {  // Any non-zero alignment
                        boundary_val = 0.0;
                        boundary_handled = true;
                        // Debug: First occurrence only
                        static int first_call_order2 = 1;
                        if (first_call_order2) {
                            printf("[DEBUG] Interior boundary handling: order=2, BC_REFLECT at boundary %d\n", ib);
                            printf("        coords=(%.3f, %.3f), normal=(%.3f, %.3f), axis=%d\n",
                                   coords[0], coords[1], hb->normal[0], hb->normal[1], axis);
                            first_call_order2 = 0;
                        }
                    }
                } else if (hb->bc_spec.type == BC_DIRICHLET) {
                    // For Dirichlet at boundary, second derivative needs ghost points
                    // Set to zero for now (will improve with proper implementation)
                    double normal_component = hb->normal[axis];
                    if (fabs(normal_component) > 0.1) {
                        boundary_val = 0.0;
                        boundary_handled = true;
                    }
                }
            }
        }
        
        // If interior boundary handled it, set value and continue
        if (boundary_handled) {
            literal_set(&result->data, tensor_idx, boundary_val);
            continue;
        }
        
        if (order == 1) {
            if (idx == 0) {
                // Minimum boundary
                BoundarySpec *bc = &grid->boundaries[axis * 2];
                
                switch (bc->type) {
                    case BC_DIRICHLET:
                        // Dirichlet: use ghost point
                        grid_index_to_coord(grid, indices, coords);
                        {
                            double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                            tensor_idx_plus[axis] = idx + 1;
                            double f1 = literal_get(&field->data, tensor_idx_plus);
                            double val = (f1 - bc_val) / (2.0 * h);
                            literal_set(&result->data, tensor_idx, val);
                        }
                        break;
                        
                    case BC_REFLECT:
                        // Reflection: du/dn = 0 at boundary (like Neumann)
                        literal_set(&result->data, tensor_idx, 0.0);
                        break;
                        
                    case BC_OPEN:
                    case BC_NEUMANN:
                    default:
                        // Open/Neumann: use one-sided difference
                        tensor_idx_plus[axis] = idx + 1;
                        {
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f1 = literal_get(&field->data, tensor_idx_plus);
                            double val = (f1 - f0) / h;
                            literal_set(&result->data, tensor_idx, val);
                        }
                        break;
                }
            } else if (idx == n - 1) {
                // Maximum boundary
                BoundarySpec *bc = &grid->boundaries[axis * 2 + 1];
                
                switch (bc->type) {
                    case BC_DIRICHLET:
                        // Dirichlet: use ghost point
                        grid_index_to_coord(grid, indices, coords);
                        {
                            double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                            tensor_idx_minus[axis] = idx - 1;
                            double f_minus = literal_get(&field->data, tensor_idx_minus);
                            double val = (bc_val - f_minus) / (2.0 * h);
                            literal_set(&result->data, tensor_idx, val);
                        }
                        break;
                        
                    case BC_REFLECT:
                        // Reflection: du/dn = 0 at boundary
                        literal_set(&result->data, tensor_idx, 0.0);
                        break;
                        
                    case BC_OPEN:
                    case BC_NEUMANN:
                    default:
                        // Open/Neumann: use one-sided difference
                        tensor_idx_minus[axis] = idx - 1;
                        {
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f_minus = literal_get(&field->data, tensor_idx_minus);
                            double val = (f0 - f_minus) / h;
                            literal_set(&result->data, tensor_idx, val);
                        }
                        break;
                }
            } else {
                // Interior: central difference
                tensor_idx_minus[axis] = idx - 1;
                tensor_idx_plus[axis] = idx + 1;
                double f_minus = literal_get(&field->data, tensor_idx_minus);
                double f_plus = literal_get(&field->data, tensor_idx_plus);
                double val = (f_plus - f_minus) / (2.0 * h);
                literal_set(&result->data, tensor_idx, val);
            }
        } else { // order == 2
            if (idx == 0) {
                // Minimum boundary - second derivative
                BoundarySpec *bc = &grid->boundaries[axis * 2];
                
                switch (bc->type) {
                    case BC_DIRICHLET:
                        // Use ghost point for Dirichlet
                        grid_index_to_coord(grid, indices, coords);
                        {
                            double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                            tensor_idx_plus[axis] = idx + 1;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f1 = literal_get(&field->data, tensor_idx_plus);
                            double val = (bc_val - 2.0 * f0 + f1) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                        }
                        break;
                        
                    case BC_OPEN: {
                        // Open boundary for sponge layer absorption:
                        // Set second derivative to zero to avoid imposing smoothness
                        // This prevents artificial reflections that occur when using
                        // one-sided stencils that enforce derivative continuity.
                        // The sponge layer damping (applied in wave equation) handles absorption.
                        literal_set(&result->data, tensor_idx, 0.0);
                        break;
                    }
                        
                    case BC_NEUMANN:
                    case BC_REFLECT:
                    default:
                        // Neumann/Reflect: use one-sided second derivative
                        tensor_idx_plus[axis] = idx + 1;
                        {
                            uint32_t tensor_idx_plus2[N_DIM];
                            for (int i = 0; i < N_DIM; i++) tensor_idx_plus2[i] = tensor_idx[i];
                            tensor_idx_plus2[axis] = idx + 2;
                            
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f1 = literal_get(&field->data, tensor_idx_plus);
                            double f2 = literal_get(&field->data, tensor_idx_plus2);
                            double val = (f2 - 2.0 * f1 + f0) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                        }
                        break;
                }
            } else if (idx == n - 1) {
                // Maximum boundary - second derivative
                BoundarySpec *bc = &grid->boundaries[axis * 2 + 1];
                
                switch (bc->type) {
                    case BC_DIRICHLET:
                        // Use ghost point for Dirichlet
                        grid_index_to_coord(grid, indices, coords);
                        {
                            double bc_val = bc->func ? bc->func(coords, bc->time) : bc->value;
                            tensor_idx_minus[axis] = idx - 1;
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f_minus = literal_get(&field->data, tensor_idx_minus);
                            double val = (f_minus - 2.0 * f0 + bc_val) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                        }
                        break;
                        
                    case BC_OPEN: {
                        // Open boundary for sponge layer absorption:
                        // Set second derivative to zero to avoid imposing smoothness
                        // This prevents artificial reflections that occur when using
                        // one-sided stencils that enforce derivative continuity.
                        // The sponge layer damping (applied in wave equation) handles absorption.
                        literal_set(&result->data, tensor_idx, 0.0);
                        break;
                    }
                        
                    case BC_NEUMANN:
                    case BC_REFLECT:
                    default:
                        // Neumann/Reflect: use one-sided second derivative
                        tensor_idx_minus[axis] = idx - 1;
                        {
                            uint32_t tensor_idx_minus2[N_DIM];
                            for (int i = 0; i < N_DIM; i++) tensor_idx_minus2[i] = tensor_idx[i];
                            tensor_idx_minus2[axis] = idx - 2;
                            
                            double f0 = literal_get(&field->data, tensor_idx);
                            double f_minus = literal_get(&field->data, tensor_idx_minus);
                            double f_minus2 = literal_get(&field->data, tensor_idx_minus2);
                            double val = (f0 - 2.0 * f_minus + f_minus2) / (h * h);
                            literal_set(&result->data, tensor_idx, val);
                        }
                        break;
                }
            } else {
                // Interior: central second difference, but check for interior boundaries
                tensor_idx_minus[axis] = idx - 1;
                tensor_idx_plus[axis] = idx + 1;
                
                // Check if stencil crosses any interior boundary
                bool stencil_crosses_boundary = false;
                int crossed_boundary_id = -1;
                
                if (field->grid->n_interior_boundaries > 0) {
                    // Get coordinates for stencil points
                    uint32_t idx_minus[N_DIM], idx_plus[N_DIM];
                    for (int i = 0; i < n_dims; i++) {
                        idx_minus[i] = indices[i];
                        idx_plus[i] = indices[i];
                    }
                    idx_minus[axis] = (idx > 0) ? idx - 1 : 0;
                    idx_plus[axis] = (idx < n - 1) ? idx + 1 : n - 1;
                    
                    double coords_minus[N_DIM], coords_plus[N_DIM];
                    grid_index_to_coord(field->grid, idx_minus, coords_minus);
                    grid_index_to_coord(field->grid, idx_plus, coords_plus);
                    
                    // Check each interior boundary
                    for (int ib = 0; ib < field->grid->n_interior_boundaries && !stencil_crosses_boundary; ib++) {
                        HyperplaneBoundary *hb = &field->grid->interior_boundaries[ib];
                        if (!hb->active) continue;
                        
                        // Compute signed distances
                        double dist_current = 0.0, dist_minus = 0.0, dist_plus = 0.0;
                        for (int d = 0; d < field->grid->n_dims; d++) {
                            dist_current += hb->normal[d] * (coords[d] - hb->point[d]);
                            dist_minus += hb->normal[d] * (coords_minus[d] - hb->point[d]);
                            dist_plus += hb->normal[d] * (coords_plus[d] - hb->point[d]);
                        }
                        
                        // Check if stencil crosses boundary (points on opposite sides)
                        if ((dist_minus * dist_plus < 0) || (dist_current * dist_minus < 0) || (dist_current * dist_plus < 0)) {
                            // Check if any of the points are actually within the bounded segment
                            bool current_in_bounds = point_in_bounded_hyperplane(hb, coords, field->grid->n_dims, h);
                            bool minus_in_bounds = point_in_bounded_hyperplane(hb, coords_minus, field->grid->n_dims, h);
                            bool plus_in_bounds = point_in_bounded_hyperplane(hb, coords_plus, field->grid->n_dims, h);
                            
                            if (current_in_bounds || minus_in_bounds || plus_in_bounds) {
                                // Stencil crosses this boundary within its bounded region
                                stencil_crosses_boundary = true;
                                crossed_boundary_id = ib;
                            }
                        }
                    }
                }
                
                double f_minus = literal_get(&field->data, tensor_idx_minus);
                double f0 = literal_get(&field->data, tensor_idx);
                double f_plus = literal_get(&field->data, tensor_idx_plus);
                double val;
                
                if (stencil_crosses_boundary && crossed_boundary_id >= 0) {
                    HyperplaneBoundary *hb = &field->grid->interior_boundaries[crossed_boundary_id];
                    
                    if (hb->bc_spec.type == BC_REFLECT) {
                        // For reflecting boundary, use ghost points that mirror across boundary
                        // Determine which side of boundary we're on
                        double dist_current = 0.0;
                        for (int d = 0; d < field->grid->n_dims; d++) {
                            dist_current += hb->normal[d] * (coords[d] - hb->point[d]);
                        }
                        
                        // Use one-sided stencil away from boundary
                        if (fabs(dist_current) < h) {
                            // Very close to boundary - set second derivative to zero
                            val = 0.0;
                        } else if (dist_current > 0) {
                            // On positive side - use forward stencil if minus crosses
                            double dist_minus = 0.0;
                            for (int d = 0; d < field->grid->n_dims; d++) {
                                uint32_t idx_m[N_DIM];
                                for (int j = 0; j < n_dims; j++) idx_m[j] = indices[j];
                                idx_m[axis] = idx - 1;
                                double coords_m[N_DIM];
                                grid_index_to_coord(field->grid, idx_m, coords_m);
                                dist_minus += hb->normal[d] * (coords_m[d] - hb->point[d]);
                            }
                            
                            if (dist_minus < 0) {
                                // Minus point is on other side - use one-sided forward
                                uint32_t tensor_idx_plus2[N_DIM];
                                for (int i = 0; i < N_DIM; i++) tensor_idx_plus2[i] = tensor_idx[i];
                                tensor_idx_plus2[axis] = idx + 2;
                                if (idx + 2 < n) {
                                    double f2 = literal_get(&field->data, tensor_idx_plus2);
                                    val = (f2 - 2.0 * f_plus + f0) / (h * h);
                                } else {
                                    val = 0.0;
                                }
                            } else {
                                val = (f_plus - 2.0 * f0 + f_minus) / (h * h);
                            }
                        } else {
                            // On negative side - use backward stencil if plus crosses
                            double dist_plus = 0.0;
                            for (int d = 0; d < field->grid->n_dims; d++) {
                                uint32_t idx_p[N_DIM];
                                for (int j = 0; j < n_dims; j++) idx_p[j] = indices[j];
                                idx_p[axis] = idx + 1;
                                double coords_p[N_DIM];
                                grid_index_to_coord(field->grid, idx_p, coords_p);
                                dist_plus += hb->normal[d] * (coords_p[d] - hb->point[d]);
                            }
                            
                            if (dist_plus > 0) {
                                // Plus point is on other side - use one-sided backward
                                uint32_t tensor_idx_minus2[N_DIM];
                                for (int i = 0; i < N_DIM; i++) tensor_idx_minus2[i] = tensor_idx[i];
                                tensor_idx_minus2[axis] = idx - 2;
                                if (idx >= 2) {
                                    double f_minus2 = literal_get(&field->data, tensor_idx_minus2);
                                    val = (f0 - 2.0 * f_minus + f_minus2) / (h * h);
                                } else {
                                    val = 0.0;
                                }
                            } else {
                                val = (f_plus - 2.0 * f0 + f_minus) / (h * h);
                            }
                        }
                    } else {
                        // For other boundary types, use standard central difference
                        val = (f_plus - 2.0 * f0 + f_minus) / (h * h);
                    }
                } else {
                    // No boundary crossing - standard central difference
                    val = (f_plus - 2.0 * f0 + f_minus) / (h * h);
                }
                
                literal_set(&result->data, tensor_idx, val);
            }
        }
    }
    
    free(coords);
    free(indices);
    free(tensor_idx);
    free(tensor_idx_minus);
    free(tensor_idx_plus);
    
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
    for (int axis = 0; axis < grid->n_dims; axis++) {
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
