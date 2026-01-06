#ifndef GRID_H
#define GRID_H

#include <stdint.h>
#include <stdbool.h>
#include "literal.h"

// ============================================================================
// Boundary Condition Types
// ============================================================================

typedef enum {
    BC_DIRICHLET,    // Fixed value: u(boundary) = g(x,y,z,t)
    BC_NEUMANN,      // Fixed derivative: ∂u/∂n = h(x,y,z,t)
    BC_ROBIN,        // Mixed: α*u + β*∂u/∂n = f(x,y,z,t)
    BC_PERIODIC,     // Periodic: u(x_min) = u(x_max)
    BC_OPEN,         // Open/outflow: ∂u/∂n = 0 (natural continuation)
    BC_REFLECT       // Reflected: u(boundary) = coeff * u(interior)
} BoundaryType;

// Function type for boundary condition evaluation
// coords: Physical coordinates [x, y, z]
// t: Time parameter
// Returns: Boundary value at this point
typedef double (*BCFunction)(const double *coords, double t);

// Boundary condition specification
typedef struct {
    BoundaryType type;
    
    // For constant value BCs
    double value;
    
    // For function-based BCs (Dirichlet, Neumann, Robin)
    BCFunction func;
    double time;  // Current time for function evaluation
    
    // For Robin: α*u + β*∂u/∂n = γ
    double alpha, beta, gamma;
    
    // For reflection: u_boundary = coeff * u_interior
    double reflection_coeff;
    
    // For BC_OPEN: maximum order of Taylor series extrapolation
    // Uses iterative Taylor series: f_ghost = f0 + \u03a3(h^k/k! * f^(k)(0))
    // Default is 3 for cubic accuracy, can be set arbitrarily high (limited by available grid points)
    int extrapolation_order;
} BoundarySpec;

// Arbitrary bounded hyperplane boundary (line segment in 2D, plane in 3D)
typedef struct {
    double *normal;         // Normal vector [nx, ny, nz] (unit vector)
    double *point;          // Point on the hyperplane [x0, y0, z0]
    double *bounds_min;     // Minimum bounds in plane coordinates
    double *bounds_max;     // Maximum bounds in plane coordinates
    double *bbox_min;       // Minimum world-space bounding box [x_min, y_min, z_min]
    double *bbox_max;       // Maximum world-space bounding box [x_max, y_max, z_max]
    BoundarySpec bc_spec;   // Boundary condition to apply
    bool active;            // Enable/disable this boundary
} HyperplaneBoundary;

// ============================================================================
// Grid Metadata
// ============================================================================

typedef struct GridMetadata {
    uint32_t *dims;         // Array of dimensions (e.g., [nx, ny, nz])
    double *spacing;        // Array of spacings (e.g., [Δx, Δy, Δz])
    double *origin;         // Array of origins (e.g., [x0, y0, z0])
    uint32_t total_points;  // Total number of grid points
    double *extent;         // Array of extents (e.g., [Lx, Ly, Lz])
    
    // Edge boundary conditions (hypercube faces)
    BoundarySpec *boundaries; // Array[n_dims * 2]: axis*2 + side (side: 0=min, 1=max)
    
    // Interior arbitrary hyperplane boundaries
    HyperplaneBoundary *interior_boundaries;
    int n_interior_boundaries;
    
    int n_dims;             // Number of dimensions
    int refcount;           // Reference count for memory management
} GridMetadata;

// Create uniform grid
// dims: Number of points in each dimension [nx, ny, nz]
// spacing: Grid spacing [Δx, Δy, Δz]
// origin: Starting coordinates [x0, y0, z0] (optional, NULL = [0,0,0])
// Returns: New GridMetadata (caller owns, must call grid_metadata_free)
GridMetadata* grid_metadata_create(const uint32_t *dims, 
                                   const double *spacing,
                                   const double *origin,
                                   int n_dims);

// Free grid metadata
void grid_metadata_free(GridMetadata *grid);

// Get total number of grid points
uint32_t grid_get_total_points(const GridMetadata *grid);

// Convert multi-dimensional index to linear index
// grid: Grid metadata (must not be NULL)
// indices: Multi-dim index [i, j, k] (must not be NULL)
// Returns: Linear index into flat array
uint32_t grid_index_to_linear(const GridMetadata *grid, const uint32_t *indices);

// Convert linear index to multi-dimensional index
// grid: Grid metadata (must not be NULL)
// linear: Linear index
// indices: Output array [i, j, k] (must not be NULL, caller provides)
void grid_linear_to_index(const GridMetadata *grid, uint32_t linear, uint32_t *indices);

// Convert grid index to physical coordinates
// grid: Grid metadata (must not be NULL)
// indices: Grid index [i, j, k] (must not be NULL)
// coords: Output physical coordinates [x, y, z] (must not be NULL, caller provides)
void grid_index_to_coord(const GridMetadata *grid, const uint32_t* indices, double* coords);

// Convert physical coordinates to grid index (nearest grid point)
// grid: Grid metadata (must not be NULL)
// coords: Physical coordinates [x, y, z] (must not be NULL)
// indices: Output grid index [i, j, k] (must not be NULL, caller provides)
// Returns: true if coords are within grid bounds, false otherwise
bool grid_coord_to_index(const GridMetadata *grid, const double* coords, uint32_t* indices);

// Check if grid index is on boundary
// grid: Grid metadata (must not be NULL)
// indices: Grid index [i, j, k] (must not be NULL)
// Returns: true if on any boundary face
bool grid_is_boundary(const GridMetadata *grid, const uint32_t* indices);

// ============================================================================
// Grid Field (Function Values on Grid)
// ============================================================================

typedef struct {
    char *name;             // Field name
    GridMetadata *grid;     // Associated grid metadata
    Literal data;          // Single tensor storing all grid point values
} GridField;

// Create grid field
// name: Field variable name (copied internally)
// vars: Independent variable names (copied internally)
// n_vars: Number of independent variables
// grid: Grid metadata (ownership NOT transferred, field keeps pointer)
// Returns: New GridField (caller owns, must call grid_field_free)
GridField* grid_field_create(GridMetadata *grid);

// Free grid field
// field: Field to free (can be NULL)
// Note: Does NOT free the grid metadata (only keeps pointer)
void grid_field_free(GridField *field);

// Get field value at grid point
// field: Grid field (must not be NULL)
// indices: Grid index [i, j, k] (must not be NULL)
// Returns: Field value at that point
Literal* grid_field_get(const GridField *field, const uint32_t *indices);

// Set field value at grid point
// field: Grid field (must not be NULL)
// indices: Grid index [i, j, k] (must not be NULL)
// value: Value to set
void grid_field_set(GridField *field, const uint32_t *indices, const Literal* value);

// Get field value at physical coordinates (interpolated)
// field: Grid field (must not be NULL)
// coords: Physical coordinates [x, y, z] (must not be NULL)
// Returns: Interpolated value (nearest neighbor for now)
Literal grid_field_evaluate(const GridField *field, const double *coords);

// Initialize field with constant value
// field: Grid field (must not be NULL)
// value: Constant value to fill
void grid_field_fill(GridField *field, const Literal* value);

// Initialize field from function
// field: Grid field (must not be NULL)
// func: Function pointer double f(x, y, z)
void grid_field_init_from_function(GridField *field, Literal* (*func)(const double *coords, int n_dims));

// ============================================================================
// Finite Difference Operators
// ============================================================================

// Compute partial derivative using finite differences
// field: Grid field (must not be NULL)
// axis: Derivative direction (0=x, 1=y, 2=z)
// order: Derivative order (1 or 2)
// Returns: New GridField with derivative values (caller owns)
GridField* grid_field_derivative(const GridField *field, int axis, int order);

// Compute Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
// field: Grid field (must not be NULL)
// Returns: New GridField with Laplacian values (caller owns)
GridField* grid_field_laplacian(const GridField *field);

// Compute gradient ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]
// field: Grid field (must not be NULL)
// Returns: Array of n_vars GridField pointers (caller owns array and fields)
GridField** grid_field_gradient(const GridField *field);

// ============================================================================
// Grid Field Operations
// ============================================================================

// Add two grid fields: result = a + b
GridField* grid_field_add(const GridField *a, const GridField *b);

// Multiply two grid fields: result = a * b (pointwise)
GridField* grid_field_multiply(const GridField *a, const GridField *b);

// Scale grid field: result = scalar * field
GridField* grid_field_scale(const GridField *field, double scalar);

// Compute L2 norm: ||field||₂ = sqrt(Σ field[i]²)
double grid_field_norm(const GridField *field);

// Copy grid field
GridField* grid_field_copy(const GridField *field);

// ============================================================================
// Axis Naming and Grid-Literal Helpers
// ============================================================================

// Get standard axis name for integer index (0='x', 1='y', 2='z', 3='w', 4='r', ...)
// Returns pointer to static string, do not free
const char* grid_axis_name(int axis);

// Get axis index from name ('x'->0, 'y'->1, 'z'->2, 'w'->3, 'r'->4, ...)
// Returns -1 if name not recognized
int grid_axis_from_name(const char *name);

// Check if a Literal's shape matches a grid's dimensions
bool grid_literal_matches(const Literal *lit, const GridMetadata *grid);

// Wrap a Literal as a temporary GridField (shallow copy of data)
// Caller must free the returned GridField with grid_field_free()
GridField* grid_field_wrap_literal(Literal *lit, GridMetadata *grid);

// ============================================================================
// Boundary Condition Configuration
// ============================================================================

// Set edge boundary condition with constant value
// grid: Grid metadata (must not be NULL)
// axis: Dimension index (0=x, 1=y, 2=z)
// side: 0 for minimum face, 1 for maximum face
// type: Boundary condition type
// value: Constant value (for Dirichlet/Neumann) or reflection coefficient
void grid_set_boundary(GridMetadata *grid, int axis, int side, 
                       BoundaryType type, double value);

// Set edge boundary condition with function
// grid: Grid metadata (must not be NULL)
// axis: Dimension index (0=x, 1=y, 2=z)
// side: 0 for minimum face, 1 for maximum face
// type: Boundary condition type (BC_DIRICHLET or BC_NEUMANN)
// func: Function to evaluate boundary value
void grid_set_boundary_func(GridMetadata *grid, int axis, int side,
                            BoundaryType type, BCFunction func);

// Set Robin boundary condition: α*u + β*∂u/∂n = γ
void grid_set_robin_boundary(GridMetadata *grid, int axis, int side,
                             double alpha, double beta, double gamma);

// Set open boundary with specific Taylor series extrapolation order
// grid: Grid metadata (must not be NULL)
// axis: Dimension index (0=x, 1=y, 2=z)
// side: 0 for minimum face, 1 for maximum face
// order: Maximum order of Taylor series (1 or higher, limited by available grid points)
void grid_set_open_boundary(GridMetadata *grid, int axis, int side, int order);

// Add arbitrary bounded hyperplane boundary
// Returns: Index of added boundary, or -1 on error
int grid_add_hyperplane_boundary(GridMetadata *grid,
                                 const double *normal,
                                 const double *point,
                                 const double *bounds_min,
                                 const double *bounds_max,
                                 BoundaryType type,
                                 double value);

// Add arbitrary bounded hyperplane with function
int grid_add_hyperplane_boundary_func(GridMetadata *grid,
                                      const double *normal,
                                      const double *point,
                                      const double *bounds_min,
                                      const double *bounds_max,
                                      BoundaryType type,
                                      BCFunction func);

// Check if a point is near a hyperplane boundary
// Returns: boundary index + 1 if near a boundary, 0 if interior
int grid_point_near_boundary(const GridMetadata *grid, const double *coords);

// Update time parameter for time-dependent BCs
void grid_update_bc_time(GridMetadata *grid, double t);

#endif // GRID_H
