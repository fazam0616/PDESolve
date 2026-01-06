#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include "../include/grid.h"
#include "../include/literal.h"

// Sponge layer parameters
#define SPONGE_WIDTH 48      // Number of cells for absorption layer
#define SIGMA_MAX 10.0        // Maximum damping coefficient in sponge layer

// Color mode enumeration
typedef enum {
    COLOR_MODE_RGB,      // R=vx, G=vy, B=height
    COLOR_MODE_HEIGHT,   // Red for negative, Blue for positive
    COLOR_MODE_VELOCITY  // Grayscale for velocity magnitude
} ColorMode;

// BMP file header structures
#pragma pack(push, 1)
typedef struct {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
} BMPHeader;

typedef struct {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bits_per_pixel;
    uint32_t compression;
    uint32_t image_size;
    int32_t x_pixels_per_meter;
    int32_t y_pixels_per_meter;
    uint32_t colors_used;
    uint32_t colors_important;
} BMPInfoHeader;
#pragma pack(pop)

// Clamp value to [0, 255]
uint8_t clamp_to_byte(double val, double scale) {
    int v = (int)(val * scale + 127.5);
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

// Clamp value to [0, 255] without centering
uint8_t clamp_to_byte_positive(double val, double scale) {
    int v = (int)(val * scale);
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

// Write grid to BMP file with specified color mode
void write_grid_to_bmp(const char *filename, 
                       GridField *height, 
                       GridField *vel_x, 
                       GridField *vel_y,
                       double value_scale,
                       int pixel_scale,
                       ColorMode mode) {
    uint32_t width = height->grid->dims[0];
    uint32_t height_dim = height->grid->dims[1];
    GridMetadata *grid = height->grid;
    
    // Scale output dimensions
    uint32_t output_width = width * pixel_scale;
    uint32_t output_height = height_dim * pixel_scale;
    
    // BMP rows must be padded to multiple of 4 bytes
    uint32_t row_size = ((output_width * 3 + 3) / 4) * 4;
    uint32_t image_size = row_size * output_height;
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }
    
    // Write BMP header
    BMPHeader header = {
        .type = 0x4D42,  // "BM"
        .size = 54 + image_size,
        .reserved1 = 0,
        .reserved2 = 0,
        .offset = 54
    };
    fwrite(&header, sizeof(BMPHeader), 1, f);
    
    // Write info header
    BMPInfoHeader info = {
        .size = 40,
        .width = output_width,
        .height = output_height,
        .planes = 1,
        .bits_per_pixel = 24,
        .compression = 0,
        .image_size = image_size,
        .x_pixels_per_meter = 2835,
        .y_pixels_per_meter = 2835,
        .colors_used = 0,
        .colors_important = 0
    };
    fwrite(&info, sizeof(BMPInfoHeader), 1, f);
    
    // Write pixel data (bottom to top, with scaling)
    uint8_t *row = calloc(row_size, 1);
    uint32_t indices[3];
    double coords[3];
    
    for (int32_t py = output_height - 1; py >= 0; py--) {
        uint32_t y = py / pixel_scale;
        
        for (uint32_t px = 0; px < output_width; px++) {
            uint32_t x = px / pixel_scale;
            
            indices[0] = x;
            indices[1] = y;
            indices[2] = 0;
            
            // Get physical coordinates
            grid_index_to_coord(grid, indices, coords);
            
            // Check if this is a boundary point
            bool is_edge_boundary = (x == 0 || x == width - 1 || y == 0 || y == height_dim - 1);
            bool is_interior_boundary = false;
            BoundaryType boundary_type = BC_OPEN;
            
            if (is_edge_boundary) {
                // Determine which edge and get boundary type
                if (x == 0) boundary_type = grid->boundaries[0 * 2 + 0].type;  // Left
                else if (x == width - 1) boundary_type = grid->boundaries[0 * 2 + 1].type;  // Right
                else if (y == 0) boundary_type = grid->boundaries[1 * 2 + 0].type;  // Bottom
                else if (y == height_dim - 1) boundary_type = grid->boundaries[1 * 2 + 1].type;  // Top
            }
            
            // Check interior boundaries
            if (grid->n_interior_boundaries > 0) {
                double line_offset = 1.5;
                double tolerance = grid->spacing[0] * 0.75;
                double dist_to_line = fabs(coords[0] + coords[1] - line_offset) / sqrt(2.0);
                bool in_segment = (coords[0] >= 0.5 && coords[0] <= 1.2);
                
                if (dist_to_line < tolerance && in_segment) {
                    is_interior_boundary = true;
                    boundary_type = BC_REFLECT;  // Interior line is reflection
                }
            }
            
            uint8_t r, g, b;
            
            // Render boundaries with specific colors
            if (is_interior_boundary || (is_edge_boundary && boundary_type != BC_OPEN)) {
                switch (boundary_type) {
                    case BC_DIRICHLET:
                        // White for rigid walls
                        r = g = b = 255;
                        break;
                    case BC_REFLECT:
                        // White for reflection surfaces
                        r = g = b = 255;
                        break;
                    case BC_NEUMANN:
                        // Magenta for Neumann
                        r = 255; g = 0; b = 255;
                        break;
                    case BC_ROBIN:
                        // Cyan for Robin
                        r = 0; g = 255; b = 255;
                        break;
                    default:
                        // Shouldn't reach here, but use yellow as warning
                        r = 255; g = 255; b = 0;
                        break;
                }
            } else {
                // Normal rendering for non-boundary points or open boundaries
                // Get field values
                Literal *h = grid_field_get(height, indices);
                Literal *vx = grid_field_get(vel_x, indices);
                Literal *vy = grid_field_get(vel_y, indices);
                
                double h_val = h ? literal_get(h, (uint32_t[]){0,0,0}) : 0.0;
                double vx_val = vx ? literal_get(vx, (uint32_t[]){0,0,0}) : 0.0;
                double vy_val = vy ? literal_get(vy, (uint32_t[]){0,0,0}) : 0.0;
                
                // Free the allocated literals
                if (h) literal_free(h);
                if (vx) literal_free(vx);
                if (vy) literal_free(vy);
                
                switch (mode) {
                    case COLOR_MODE_RGB:
                        // BMP is BGR format
                        // R = x-velocity, G = y-velocity, B = wave height
                        b = clamp_to_byte(h_val, value_scale);
                        g = clamp_to_byte(vy_val, value_scale * 2);
                        r = clamp_to_byte(vx_val, value_scale * 2);
                        break;
                        
                    case COLOR_MODE_HEIGHT:
                        // Red for negative height, Blue for positive height
                        if (h_val < 0) {
                            r = clamp_to_byte_positive(-h_val, value_scale);
                            g = 0;
                            b = 0;
                        } else {
                            r = 0;
                            g = 0;
                            b = clamp_to_byte_positive(h_val, value_scale);
                        }
                        break;
                        
                    case COLOR_MODE_VELOCITY:
                        // Grayscale for velocity magnitude
                        {
                            double vel_mag = sqrt(vx_val * vx_val + vy_val * vy_val);
                            uint8_t gray = clamp_to_byte_positive(vel_mag, value_scale * 2);
                            r = gray;
                            g = gray;
                            b = gray;
                        }
                        break;
                        
                    default:
                        r = g = b = 0;
                        break;
                }
            }
            
            row[px * 3 + 0] = b;
            row[px * 3 + 1] = g;
            row[px * 3 + 2] = r;
        }
        fwrite(row, row_size, 1, f);
    }
    
    free(row);
    fclose(f);
}

// Write only a subregion of the grid to BMP (for rendering visible area only)
void write_grid_to_bmp_region(const char *filename, 
                              GridField *height, 
                              GridField *vel_x, 
                              GridField *vel_y,
                              double value_scale,
                              int pixel_scale,
                              ColorMode mode,
                              int start_x, int start_y,
                              int width, int height_dim) {
    GridMetadata *grid = height->grid;
    
    // Scale output dimensions
    uint32_t output_width = width * pixel_scale;
    uint32_t output_height = height_dim * pixel_scale;
    
    // BMP rows must be padded to multiple of 4 bytes
    uint32_t row_size = ((output_width * 3 + 3) / 4) * 4;
    uint32_t image_size = row_size * output_height;
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }
    
    // Write BMP header
    BMPHeader header = {
        .type = 0x4D42,  // "BM"
        .size = 54 + image_size,
        .reserved1 = 0,
        .reserved2 = 0,
        .offset = 54
    };
    fwrite(&header, sizeof(BMPHeader), 1, f);
    
    // Write info header
    BMPInfoHeader info = {
        .size = 40,
        .width = output_width,
        .height = output_height,
        .planes = 1,
        .bits_per_pixel = 24,
        .compression = 0,
        .image_size = image_size,
        .x_pixels_per_meter = 2835,
        .y_pixels_per_meter = 2835,
        .colors_used = 0,
        .colors_important = 0
    };
    fwrite(&info, sizeof(BMPInfoHeader), 1, f);
    
    // Allocate row buffer
    uint8_t *row = calloc(row_size, 1);
    
    // Write pixel data (bottom to top for BMP)
    uint32_t indices[3];
    for (int py = 0; py < (int)output_height; py++) {
        int y = py / pixel_scale;
        int grid_y = start_y + (height_dim - 1 - y);  // BMP is bottom-to-top
        
        for (int px = 0; px < (int)output_width; px++) {
            int x = px / pixel_scale;
            int grid_x = start_x + x;
            
            indices[0] = grid_x;
            indices[1] = grid_y;
            indices[2] = 0;
            
            uint8_t r, g, b;
            
            // Get field values
            Literal *h_lit = grid_field_get(height, indices);
            Literal *vx_lit = grid_field_get(vel_x, indices);
            Literal *vy_lit = grid_field_get(vel_y, indices);
            
            double h_val = h_lit ? literal_get(h_lit, (uint32_t[]){0,0,0}) : 0.0;
            double vx_val = vx_lit ? literal_get(vx_lit, (uint32_t[]){0,0,0}) : 0.0;
            double vy_val = vy_lit ? literal_get(vy_lit, (uint32_t[]){0,0,0}) : 0.0;
            
            if (h_lit) literal_free(h_lit);
            if (vx_lit) literal_free(vx_lit);
            if (vy_lit) literal_free(vy_lit);
            
            // Check if this point is on an interior boundary (draw white line)
            bool on_boundary = false;
            if (grid->n_interior_boundaries > 0) {
                double coords[3];
                coords[0] = grid->origin[0] + grid_x * grid->spacing[0];
                coords[1] = grid->origin[1] + grid_y * grid->spacing[1];
                coords[2] = 0.0;
                
                for (int ib = 0; ib < grid->n_interior_boundaries; ib++) {
                    HyperplaneBoundary *hb = &grid->interior_boundaries[ib];
                    if (!hb->active) continue;
                    
                    // Calculate distance to hyperplane: d = n · (p - p0)
                    double diff[3] = {
                        coords[0] - hb->point[0],
                        coords[1] - hb->point[1],
                        coords[2] - (grid->n_dims > 2 ? hb->point[2] : 0.0)
                    };
                    double dist = hb->normal[0] * diff[0] + hb->normal[1] * diff[1];
                    if (grid->n_dims > 2) dist += hb->normal[2] * diff[2];
                    
                    // Check if within bounds and close to line (within 1 cell width)
                    double cell_width = sqrt(grid->spacing[0]*grid->spacing[0] + grid->spacing[1]*grid->spacing[1]);
                    if (fabs(dist) < cell_width) {
                        // Check bounds along the line
                        double proj_x = coords[0] - dist * hb->normal[0];
                        if (proj_x >= hb->bounds_min[0] && proj_x <= hb->bounds_max[0]) {
                            on_boundary = true;
                            break;
                        }
                    }
                }
            }
            
            // Override with white if on boundary
            if (on_boundary) {
                r = 255;
                g = 255;
                b = 255;
            } else {
                // Color based on mode
                switch (mode) {
                    case COLOR_MODE_RGB:
                        b = clamp_to_byte(h_val, value_scale);
                        g = clamp_to_byte(vy_val, value_scale * 2);
                        r = clamp_to_byte(vx_val, value_scale * 2);
                        break;
                        
                    case COLOR_MODE_HEIGHT:
                        if (h_val < 0) {
                            r = clamp_to_byte_positive(-h_val, value_scale);
                            g = 0;
                            b = 0;
                        } else {
                            r = 0;
                            g = 0;
                            b = clamp_to_byte_positive(h_val, value_scale);
                        }
                        break;
                        
                    case COLOR_MODE_VELOCITY:
                        {
                            double vel_mag = sqrt(vx_val * vx_val + vy_val * vy_val);
                            uint8_t gray = clamp_to_byte_positive(vel_mag, value_scale * 2);
                            r = gray;
                            g = gray;
                            b = gray;
                        }
                        break;
                        
                    default:
                        r = g = b = 0;
                        break;
                }
            }
            
            row[px * 3 + 0] = b;
            row[px * 3 + 1] = g;
            row[px * 3 + 2] = r;
        }
        fwrite(row, row_size, 1, f);
    }
    
    free(row);
    fclose(f);
}

// Helper function to add interior line segment boundary using visible region coordinates
// Points (x1, y1) and (x2, y2) are in visible region coordinates [0, Lx_visible] × [0, Ly_visible]
// Returns boundary ID or -1 on failure
int add_interior_line_visible(GridMetadata *grid, 
                              double x1_vis, double y1_vis,
                              double x2_vis, double y2_vis,
                              BoundaryType bc_type, double coeff) {
    // Convert visible coordinates to absolute grid coordinates
    // Visible region starts at (sponge_width * dx, sponge_width * dy)
    double dx = grid->spacing[0];
    double dy = grid->spacing[1];
    double offset_x = SPONGE_WIDTH * dx;
    double offset_y = SPONGE_WIDTH * dy;
    
    double x1 = x1_vis + offset_x;
    double y1 = y1_vis + offset_y;
    double x2 = x2_vis + offset_x;
    double y2 = y2_vis + offset_y;
    
    // Calculate direction vector and normalize to get normal
    double dir_x = x2 - x1;
    double dir_y = y2 - y1;
    double len = sqrt(dir_x * dir_x + dir_y * dir_y);
    
    if (len < 1e-10) {
        fprintf(stderr, "Error: Line segment has zero length\n");
        return -1;
    }
    
    // Normal is perpendicular to direction: (dy, -dx) normalized
    double normal[2] = {dir_y / len, -dir_x / len};
    
    // Use midpoint as reference point for symmetric bounds
    double point_on_line[2] = {(x1 + x2) / 2.0, (y1 + y2) / 2.0};
    
    // Use symmetric parametric coordinates: -len/2 to +len/2
    double bounds_min[1] = {-len / 2.0};
    double bounds_max[1] = {len / 2.0};
    
    return grid_add_hyperplane_boundary(grid, normal, point_on_line, 
                                       bounds_min, bounds_max,
                                       bc_type, coeff);
}

// Initialize wave with Gaussian bump
// Global variables for visible region size (set in main)
static double g_Lx_visible = 2.0;
static double g_Ly_visible = 2.0;

Literal* init_gaussian(const double *coords, int n_dims) {
    double x = coords[0];
    double y = coords[1];
    
    // Convert absolute coordinates to visible region coordinates
    double dx = g_Lx_visible / 119.0;  // Spacing for 120-point visible grid
    double dy = g_Ly_visible / 119.0;
    double offset_x = SPONGE_WIDTH * dx;
    double offset_y = SPONGE_WIDTH * dy;
    
    double x_vis = x - offset_x;
    double y_vis = y - offset_y;
    
    // Center at (0.5, 0.5) in visible region
    double center_x_vis = 0.5;
    double center_y_vis = 0.5;
    
    double dx_c = x_vis - center_x_vis;
    double dy_c = y_vis - center_y_vis;
    double r2 = dx_c*dx_c + dy_c*dy_c;
    
    // Gaussian with width sigma = 0.1
    double amplitude = 1.0;
    double sigma = 0.1;
    double value = amplitude * exp(-r2 / (2.0 * sigma * sigma));
    
    return literal_create_scalar(value);
}

// Initialize wave with plane wave
Literal* init_plane_wave(const double *coords, int n_dims) {
    double x = coords[0];
    double y = coords[1];
    
    // Plane wave traveling diagonally
    double k = 10.0;  // Wave number
    double value = 0.5 * sin(k * (x + y));
    
    return literal_create_scalar(value);
}

int main(int argc, char **argv) {
    printf("===========================================\n");
    printf("2D Wave Simulation Test\n");
    printf("===========================================\n\n");
    
    // Default simulation parameters
    uint32_t nx = 100;           // Grid size in x
    uint32_t ny = 100;           // Grid size in y
    double dt = 0.002;           // Time step
    int num_steps = 200;         // Total time steps
    int num_frames = 41;         // Number of output frames
    int pixel_scale = 1;         // Pixel scaling factor
    ColorMode color_mode = COLOR_MODE_RGB;  // Default color mode
    const char *mode_name = "RGB";
    
    // Parse command line arguments
    // Usage: ./test_wave_sim [color_mode] [nx] [ny] [dt] [num_steps] [num_frames]
    if (argc > 1) {
        // First argument: color mode
        if (strcmp(argv[1], "rgb") == 0) {
            color_mode = COLOR_MODE_RGB;
            mode_name = "RGB";
        } else if (strcmp(argv[1], "height") == 0) {
            color_mode = COLOR_MODE_HEIGHT;
            mode_name = "HEIGHT";
        } else if (strcmp(argv[1], "velocity") == 0) {
            color_mode = COLOR_MODE_VELOCITY;
            mode_name = "VELOCITY";
        } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            printf("Usage: %s [color_mode] [nx] [ny] [dt] [num_steps] [num_frames] [pixel_scale]\n", argv[0]);
            printf("\nArguments:\n");
            printf("  color_mode  - Visualization mode: rgb|height|velocity (default: rgb)\n");
            printf("  nx          - Grid size in x (default: 100)\n");
            printf("  ny          - Grid size in y (default: 100)\n");
            printf("  dt          - Time step in seconds (default: 0.002)\n");
            printf("  num_steps   - Number of simulation steps (default: 200)\n");
            printf("  num_frames  - Number of output frames (default: 41)\n");
            printf("  pixel_scale - Pixels per grid point (default: 1)\n");
            printf("\nColor modes:\n");
            printf("  rgb      - R=vx, G=vy, B=height\n");
            printf("  height   - Red for negative, Blue for positive\n");
            printf("  velocity - Grayscale for velocity magnitude\n");
            printf("\nBoundary colors:\n");
            printf("  White    - Solid surface (Dirichlet/Reflect)\n");
            printf("  Magenta  - Neumann boundary\n");
            printf("  Cyan     - Robin boundary\n");
            printf("  (none)   - Open boundary (transparent)\n");
            printf("\nExamples:\n");
            printf("  %s rgb 200 200 0.001 500 100 2\n", argv[0]);
            printf("  %s height 50 50 0.005 100 20 3\n", argv[0]);
            return 0;
        } else {
            printf("Unknown color mode: %s\n", argv[1]);
            printf("Use --help for usage information\n");
            return 1;
        }
    }
    
    // Parse optional parameters
    if (argc > 2) nx = atoi(argv[2]);
    if (argc > 3) ny = atoi(argv[3]);
    if (argc > 4) dt = atof(argv[4]);
    if (argc > 5) num_steps = atoi(argv[5]);
    if (argc > 6) num_frames = atoi(argv[6]);
    if (argc > 7) pixel_scale = atoi(argv[7]);
    
    // Validate parameters
    if (nx < 3 || ny < 3) {
        printf("Error: Grid size must be at least 3x3\n");
        return 1;
    }
    if (dt <= 0.0) {
        printf("Error: Time step must be positive\n");
        return 1;
    }
    if (num_steps < 1) {
        printf("Error: Number of steps must be at least 1\n");
        return 1;
    }
    if (num_frames < 1 || num_frames > num_steps + 1) {
        printf("Error: Number of frames must be between 1 and num_steps+1\n");
        return 1;
    }
    if (pixel_scale < 1 || pixel_scale > 10) {
        printf("Error: Pixel scale must be between 1 and 10\n");
        return 1;
    }
    
    // Calculate output interval
    int output_interval = (num_frames == 1) ? num_steps + 1 : num_steps / (num_frames - 1);
    if (output_interval < 1) output_interval = 1;
    
    printf("Color Mode: %s\n", mode_name);
    printf("Pixel Scale: %dx%d per grid point\n", pixel_scale, pixel_scale);
    
    // Domain parameters - visible region only
    double Lx_visible = 2.0;     // Visible domain size
    double Ly_visible = 2.0;
    
    // Set global visible region size for Gaussian initialization
    g_Lx_visible = Lx_visible;
    g_Ly_visible = Ly_visible;
    
    // Define visible region size (independent of input nx, ny)
    int nx_visible = 120;        // Visible region grid points
    int ny_visible = 120;
    
    // Total grid includes sponge margins on all sides
    nx = nx_visible + 2 * SPONGE_WIDTH;  // 120 + 96 = 216
    ny = ny_visible + 2 * SPONGE_WIDTH;  // 120 + 96 = 216
    
    // Grid spacing based on visible region
    double dx = Lx_visible / (nx_visible - 1);
    double dy = Ly_visible / (ny_visible - 1);
    
    // Total physical domain includes sponge layer
    double Lx = Lx_visible * nx / (double)nx_visible;
    double Ly = Ly_visible * ny / (double)ny_visible;
    double wave_speed = 1.0;     // Wave propagation speed
    
    printf("Grid (total): %d x %d\n", nx, ny);
    printf("Grid (visible): %d x %d\n", nx_visible, ny_visible);
    printf("Sponge layer: %d cells (σ_max=%.2f)\n", SPONGE_WIDTH, SIGMA_MAX);
    printf("Domain (total): %.2f x %.2f\n", Lx, Ly);
    printf("Domain (visible): %.2f x %.2f\n", Lx_visible, Ly_visible);
    printf("Spacing: dx=%.4f, dy=%.4f\n", dx, dy);
    printf("Time step: dt=%.4f\n", dt);
    printf("Total steps: %d\n", num_steps);
    printf("Output frames: %d (every %d steps)\n", num_frames, output_interval);
    printf("Total simulation time: %.3f seconds\n", num_steps * dt);
    printf("Wave speed: c=%.2f\n", wave_speed);
    printf("CFL number: %.4f (should be < 1/√2 ≈ 0.707)\n", 
           wave_speed * dt / fmin(dx, dy));
    printf("\n");
    
    // Check CFL condition
    double cfl = wave_speed * dt / fmin(dx, dy);
    if (cfl > 0.7) {
        printf("WARNING: CFL condition may be violated! Reduce dt or increase grid spacing.\n\n");
    }
    
    // Create grid metadata
    uint32_t dims[3] = {nx, ny, 1};
    double spacing[3] = {dx, dy, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    
    // Set different boundary conditions on different edges
    printf("Setting boundary conditions:\n");
    printf("  Left edge (x=0):   BC_OPEN - Sponge layer absorption\n");
    printf("  Right edge (x=%.1f): BC_OPEN - Sponge layer absorption\n", Lx);
    printf("  Bottom edge (y=0): BC_OPEN - Sponge layer absorption\n");
    printf("  Top edge (y=%.1f):  BC_OPEN - Sponge layer absorption\n", Ly);
    
    grid_set_boundary(grid, 0, 0, BC_OPEN, 0.0);  // Left: x_min
    grid_set_boundary(grid, 0, 1, BC_OPEN, 0.0);  // Right: x_max
    grid_set_boundary(grid, 1, 0, BC_OPEN, 0.0);  // Bottom: y_min
    grid_set_boundary(grid, 1, 1, BC_OPEN, 0.0);  // Top: y_max
    
    // Add interior line segment boundary perpendicular to original
    // Using visible region coordinates [0, 2] × [0, 2]
    // Line from (0.6, 0.8) to (0.9, 0.5) - smaller segment with negative slope
    double x1_vis = 0.6;
    double y1_vis = 0.8;
    double x2_vis = 0.9;
    double y2_vis = 0.5;
    
    int boundary_id = add_interior_line_visible(
        grid, x1_vis, y1_vis, x2_vis, y2_vis,
        BC_REFLECT, 1.0
    );
    
    if (boundary_id >= 0) {
        printf("  Interior line segment: From (%.1f, %.1f) to (%.1f, %.1f) in visible region\n",
               x1_vis, y1_vis, x2_vis, y2_vis);
        printf("    Line type: BC_REFLECT (rigid reflection)\n");
    } else {
        printf("  WARNING: Failed to add interior boundary\n");
    }
    printf("\n");
    
    // Create fields for wave simulation
    // u = current height, u_prev = previous height, u_next = next height
    GridField *u_curr = grid_field_create(grid);
    GridField *u_prev = grid_field_create(grid);
    GridField *u_next = grid_field_create(grid);
    
    // Fields for visualization
    GridField *vel_x = grid_field_create(grid);
    GridField *vel_y = grid_field_create(grid);
    
    printf("Initializing wave field with Gaussian bump at center...\n");
    grid_field_init_from_function(u_curr, init_gaussian);
    grid_field_init_from_function(u_prev, init_gaussian);
    
    // Create output directory
    system("mkdir -p wave_output");
    
    printf("Starting simulation (%d steps)...\n", num_steps);
    printf("Outputting BMP files every %d steps to wave_output/\n\n", output_interval);
    
    // Time integration using leapfrog scheme
    // Wave equation: ∂²u/∂t² = c²∇²u
    // Discretization: u_next = 2*u_curr - u_prev + c²*dt²*∇²u_curr
    
    double c2_dt2 = wave_speed * wave_speed * dt * dt;
    
    // Timing variables
    struct timespec t_start, t_end;
    double time_bmp = 0.0;        // Time spent writing BMP files
    double time_calc = 0.0;       // Time spent on calculations
    double time_total_measured = 0.0;
    
    // Track child processes for BMP writing
    int *child_pids = malloc(sizeof(int) * num_frames);
    int num_children = 0;
    int max_concurrent_children = 10;  // Limit concurrent child processes
    
    for (int step = 0; step <= num_steps; step++) {
        // Wait for children if we have too many running
        while (num_children >= max_concurrent_children) {
            int status;
            pid_t finished = waitpid(-1, &status, 0);
            if (finished > 0) {
                // Remove finished child from tracking
                for (int i = 0; i < num_children; i++) {
                    if (child_pids[i] == finished) {
                        child_pids[i] = child_pids[num_children - 1];
                        num_children--;
                        break;
                    }
                }
            }
        }
        
        // Output current state
        if (step % output_interval == 0) {
            printf("Step %4d / %d (t = %.3f)\n", step, num_steps, step * dt);
            
            clock_gettime(CLOCK_MONOTONIC, &t_start);
            
            // Compute spatial derivatives for visualization
            GridField *du_dx = grid_field_derivative(u_curr, 0, 1);
            GridField *du_dy = grid_field_derivative(u_curr, 1, 1);
            
            if (du_dx && du_dy) {
                // Copy to vel_x, vel_y for visualization (only visible region needed for BMP)
                #pragma omp parallel
                {
                    uint32_t *idx = malloc(sizeof(uint32_t) * grid->n_dims);
                    #pragma omp for
                    for (uint32_t i = SPONGE_WIDTH; i < nx - SPONGE_WIDTH; i++) {
                        for (uint32_t j = SPONGE_WIDTH; j < ny - SPONGE_WIDTH; j++) {
                            idx[0] = i;
                            idx[1] = j;
                            idx[2] = 0;
                            
                            Literal *vx = grid_field_get(du_dx, idx);
                            if (vx) {
                                grid_field_set(vel_x, idx, vx);
                                literal_free(vx);
                            }
                            
                            Literal *vy = grid_field_get(du_dy, idx);
                            if (vy) {
                                grid_field_set(vel_y, idx, vy);
                                literal_free(vy);
                            }
                        }
                    }
                    free(idx);
                }
                grid_field_free(du_dx);
                grid_field_free(du_dy);
            }
            
            // Fork to write BMP in parallel with calculations
            // Only render the visible region (excluding sponge layer margins)
            pid_t pid = fork();
            
            if (pid == 0) {
                // Child process: write BMP and exit immediately
                char filename[256];
                snprintf(filename, sizeof(filename), "wave_output/wave_%04d.bmp", step / output_interval);
                write_grid_to_bmp_region(filename, u_curr, vel_x, vel_y, 200.0, pixel_scale, color_mode,
                                        SPONGE_WIDTH, SPONGE_WIDTH, nx_visible, ny_visible);
                
                // Exit immediately without cleanup (OS will handle it)
                _exit(0);
            } else if (pid > 0) {
                // Parent process: track child and continue
                child_pids[num_children++] = pid;
                
                // Prioritize parent process
                setpriority(PRIO_PROCESS, 0, -5);  // Parent gets higher priority
                setpriority(PRIO_PROCESS, pid, 5); // Child gets lower priority
                
                clock_gettime(CLOCK_MONOTONIC, &t_end);
                double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
                time_bmp += elapsed;
            } else {
                // Fork failed, write in parent as fallback
                fprintf(stderr, "Warning: fork failed, writing BMP in main process\n");
                char filename[256];
                snprintf(filename, sizeof(filename), "wave_output/wave_%04d.bmp", step / output_interval);
                write_grid_to_bmp_region(filename, u_curr, vel_x, vel_y, 200.0, pixel_scale, color_mode,
                                        SPONGE_WIDTH, SPONGE_WIDTH, nx_visible, ny_visible);
                
                clock_gettime(CLOCK_MONOTONIC, &t_end);
                double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
                time_bmp += elapsed;
            }
        }
        
        if (step == num_steps) break;
        
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        
        // Compute Laplacian of current field
        GridField *laplacian = grid_field_laplacian(u_curr);
        if (!laplacian) {
            fprintf(stderr, "Failed to compute Laplacian at step %d\n", step);
            break;
        }
        
        // Scale Laplacian by c²*dt²
        GridField *laplacian_scaled = grid_field_scale(laplacian, c2_dt2);
        grid_field_free(laplacian);
        
        // Compute: u_next = 2*u_curr - u_prev + c²*dt²*∇²u_curr
        GridField *two_u_curr = grid_field_scale(u_curr, 2.0);
        GridField *temp = grid_field_add(two_u_curr, laplacian_scaled);
        grid_field_free(two_u_curr);
        grid_field_free(laplacian_scaled);
        
        // Subtract u_prev
        GridField *neg_u_prev = grid_field_scale(u_prev, -1.0);
        GridField *result = grid_field_add(temp, neg_u_prev);
        grid_field_free(temp);
        grid_field_free(neg_u_prev);
        
        // Replace u_next with computed result (pointer assignment instead of copying)
        grid_field_free(u_next);
        u_next = result;
        
        // Apply sponge layer damping in BC_OPEN absorption regions
        // Similar to RippleGL: u_next = (2-σdt)*u_curr - (1-σdt)*u_prev + accel
        // Damping coefficient σ increases quadratically toward boundaries
        #pragma omp parallel
        {
            uint32_t *indices = malloc(sizeof(uint32_t) * grid->n_dims);
            
            #pragma omp for
            for (uint32_t i = 0; i < nx; i++) {
                for (uint32_t j = 0; j < ny; j++) {
                    indices[0] = i;
                    indices[1] = j;
                    indices[2] = 0;
                    
                    // Compute distance to nearest BC_OPEN boundary
                    double dist_to_boundary = (double)nx;  // Large initial value
                    
                    // Distance from each edge
                    if (grid->boundaries[0].type == BC_OPEN) {  // x_min
                        dist_to_boundary = fmin(dist_to_boundary, (double)i);
                    }
                    if (grid->boundaries[1].type == BC_OPEN) {  // x_max
                        dist_to_boundary = fmin(dist_to_boundary, (double)(nx - 1 - i));
                    }
                    if (grid->boundaries[2].type == BC_OPEN) {  // y_min
                        dist_to_boundary = fmin(dist_to_boundary, (double)j);
                    }
                    if (grid->boundaries[3].type == BC_OPEN) {  // y_max
                        dist_to_boundary = fmin(dist_to_boundary, (double)(ny - 1 - j));
                    }
                    
                    // Apply damping in sponge layer region
                    if (dist_to_boundary < SPONGE_WIDTH) {
                        // Quadratic ramp: σ = σ_max * (1 - d/w)²
                        double d_normalized = dist_to_boundary / (double)SPONGE_WIDTH;
                        double sigma = SIGMA_MAX * (1.0 - d_normalized) * (1.0 - d_normalized);
                        
                        // Get current and previous values
                        Literal *u_c = grid_field_get(u_curr, indices);
                        Literal *u_p = grid_field_get(u_prev, indices);
                        Literal *u_n_old = grid_field_get(u_next, indices);
                        
                        if (u_c && u_p && u_n_old) {
                            double u_curr_val = literal_get(u_c, (uint32_t[]){0,0,0});
                            double u_prev_val = literal_get(u_p, (uint32_t[]){0,0,0});
                            double accel = literal_get(u_n_old, (uint32_t[]){0,0,0}) - 2.0*u_curr_val + u_prev_val;
                            
                            // Sponge layer formula: includes damping on velocity
                            double u_next_val = (2.0 - sigma*dt)*u_curr_val - (1.0 - sigma*dt)*u_prev_val + accel;
                            
                            Literal *u_n = literal_create_scalar(u_next_val);
                            grid_field_set(u_next, indices, u_n);
                            literal_free(u_n);
                        }
                        
                        if (u_c) literal_free(u_c);
                        if (u_p) literal_free(u_p);
                        if (u_n_old) literal_free(u_n_old);
                    }
                }
            }
            free(indices);
        }
        
        // Cycle fields: prev <- curr <- next
        GridField *tmp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = tmp;
        
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
        time_calc += elapsed;
    }
    
    printf("\nSimulation complete!\n");
    
    // Wait for all remaining child processes to finish writing
    printf("Waiting for %d BMP writing processes to complete...\n", num_children);
    while (num_children > 0) {
        int status;
        pid_t finished = waitpid(-1, &status, 0);
        if (finished > 0) {
            num_children--;
        }
    }
    free(child_pids);
    printf("All BMP files written.\n");
    
    // Calculate actual number of frames written
    int actual_frames = 0;
    for (int step = 0; step <= num_steps; step++) {
        if (step % output_interval == 0) {
            actual_frames++;
        }
    }
    
    printf("Output files written to wave_output/wave_0000.bmp through wave_%04d.bmp\n", 
           actual_frames - 1);
    printf("Total frames: %d\n", actual_frames);
    
    // Print timing statistics
    time_total_measured = time_bmp + time_calc;
    double pct_bmp = 100.0 * time_bmp / time_total_measured;
    double pct_calc = 100.0 * time_calc / time_total_measured;
    
    printf("\n=== Performance Statistics ===\n");
    printf("Time spent on BMP writing:  %.3f seconds (%.1f%%)\n", time_bmp, pct_bmp);
    printf("Time spent on calculations: %.3f seconds (%.1f%%)\n", time_calc, pct_calc);
    printf("Total measured time:        %.3f seconds\n", time_total_measured);
    printf("Calculation throughput:     %.2f steps/sec\n", num_steps / time_calc);
    
    printf("\nVisualization color scheme (%s mode):\n", mode_name);
    
    switch (color_mode) {
        case COLOR_MODE_RGB:
            printf("  Red   (R) = x-velocity (∂u/∂x)\n");
            printf("  Green (G) = y-velocity (∂u/∂y)\n");
            printf("  Blue  (B) = wave height (u)\n");
            break;
        case COLOR_MODE_HEIGHT:
            printf("  Red       = Negative height (wave troughs)\n");
            printf("  Blue      = Positive height (wave crests)\n");
            printf("  Black     = Zero height\n");
            break;
        case COLOR_MODE_VELOCITY:
            printf("  Grayscale = Velocity magnitude sqrt(vx² + vy²)\n");
            printf("  Black     = No motion\n");
            printf("  White     = High velocity\n");
            break;
    }
    
    // Calculate real-time framerate for ffmpeg
    double total_sim_time = num_steps * dt;
    double realtime_fps = actual_frames / total_sim_time;
    
    printf("\nCreate real-time animation with:\n");
    printf("  ffmpeg -framerate %.2f -i wave_output/wave_%%04d.bmp -c:v libx264 -pix_fmt yuv420p wave.mp4\n", 
           realtime_fps);
    printf("\n  (%.2f fps = %d frames / %.3f seconds of simulated time)\n", 
           realtime_fps, actual_frames, total_sim_time);
    
    printf("\nTo run with custom parameters:\n");
    printf("  %s [color_mode] [nx] [ny] [dt] [num_steps] [num_frames] [pixel_scale]\n", argv[0]);
    printf("  %s --help  (for detailed usage)\n", argv[0]);
    
    // Cleanup
    grid_field_free(u_curr);
    grid_field_free(u_prev);
    grid_field_free(u_next);
    grid_field_free(vel_x);
    grid_field_free(vel_y);
    grid_metadata_free(grid);
    
    return 0;
}
