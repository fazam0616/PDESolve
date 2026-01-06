#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/grid.h"

// Test utilities
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        return 0; \
    } \
} while(0)

#define TEST_PASS(msg) do { \
    printf("PASS: %s\n", msg); \
    return 1; \
} while(0)

// Helper to create a grid and add a hyperplane boundary
static int add_test_boundary(GridMetadata *grid, 
                             const double *normal, 
                             const double *point,
                             const double *bounds_min,
                             const double *bounds_max) {
    return grid_add_hyperplane_boundary(grid, normal, point, bounds_min, bounds_max, 
                                        BC_REFLECT, 1.0);
}

// Test 1: 2D Horizontal Line Segment
int test_2d_horizontal_line() {
    uint32_t dims[] = {100, 100};
    double spacing[] = {0.1, 0.1};
    double origin[] = {0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    
    // Horizontal line at y=5, from x=2 to x=7
    // Normal pointing up: (0, 1)
    double normal[] = {0.0, 1.0};
    double point[] = {4.5, 5.0};  // Midpoint of segment
    double bounds_min[] = {-2.5};  // -half_length
    double bounds_max[] = {2.5};   // +half_length
    
    int bid = add_test_boundary(grid, normal, point, bounds_min, bounds_max);
    TEST_ASSERT(bid >= 0, "Failed to add boundary");
    
    // Test points on the line within bounds
    double test_point1[] = {3.5, 5.0};  // Middle of segment
    int result1 = grid_point_near_boundary(grid, test_point1);
    TEST_ASSERT(result1 > 0, "Point in middle of segment should be on boundary");
    
    double test_point2[] = {2.0, 5.0};  // Left endpoint
    int result2 = grid_point_near_boundary(grid, test_point2);
    TEST_ASSERT(result2 > 0, "Left endpoint should be on boundary");
    
    double test_point3[] = {7.0, 5.0};  // Right endpoint
    int result3 = grid_point_near_boundary(grid, test_point3);
    TEST_ASSERT(result3 > 0, "Right endpoint should be on boundary");
    
    // Test points on the line but outside bounds
    double test_point4[] = {1.5, 5.0};  // Before left endpoint
    int result4 = grid_point_near_boundary(grid, test_point4);
    TEST_ASSERT(result4 == 0, "Point before segment should NOT be on boundary");
    
    double test_point5[] = {7.5, 5.0};  // After right endpoint
    int result5 = grid_point_near_boundary(grid, test_point5);
    TEST_ASSERT(result5 == 0, "Point after segment should NOT be on boundary");
    
    // Test points off the line
    double test_point6[] = {3.5, 5.5};  // Above line
    int result6 = grid_point_near_boundary(grid, test_point6);
    TEST_ASSERT(result6 == 0, "Point above line should NOT be on boundary");
    
    grid_metadata_free(grid);
    TEST_PASS("2D horizontal line segment");
}

// Test 2: 2D Vertical Line Segment
int test_2d_vertical_line() {
    uint32_t dims[] = {100, 100};
    double spacing[] = {0.1, 0.1};
    double origin[] = {0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    
    // Vertical line at x=4, from y=1 to y=6
    // Normal pointing right: (1, 0)
    double normal[] = {1.0, 0.0};
    double point[] = {4.0, 3.5};  // Midpoint of segment
    double bounds_min[] = {-2.5};  // -half_length
    double bounds_max[] = {2.5};   // +half_length
    
    int bid = add_test_boundary(grid, normal, point, bounds_min, bounds_max);
    TEST_ASSERT(bid >= 0, "Failed to add boundary");
    
    // Test points on the line within bounds
    double test_point1[] = {4.0, 3.5};  // Middle of segment
    int result1 = grid_point_near_boundary(grid, test_point1);
    TEST_ASSERT(result1 > 0, "Point in middle of vertical segment should be on boundary");
    
    double test_point2[] = {4.0, 1.0};  // Bottom endpoint
    int result2 = grid_point_near_boundary(grid, test_point2);
    TEST_ASSERT(result2 > 0, "Bottom endpoint should be on boundary");
    
    double test_point3[] = {4.0, 6.0};  // Top endpoint
    int result3 = grid_point_near_boundary(grid, test_point3);
    TEST_ASSERT(result3 > 0, "Top endpoint should be on boundary");
    
    // Test points outside bounds
    double test_point4[] = {4.0, 0.5};  // Below bottom
    int result4 = grid_point_near_boundary(grid, test_point4);
    TEST_ASSERT(result4 == 0, "Point below segment should NOT be on boundary");
    
    double test_point5[] = {4.0, 6.5};  // Above top
    int result5 = grid_point_near_boundary(grid, test_point5);
    TEST_ASSERT(result5 == 0, "Point above segment should NOT be on boundary");
    
    grid_metadata_free(grid);
    TEST_PASS("2D vertical line segment");
}

// Test 3: 2D Diagonal Line Segment
int test_2d_diagonal_line() {
    uint32_t dims[] = {100, 100};
    double spacing[] = {0.1, 0.1};
    double origin[] = {0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    
    // Diagonal line from (2,2) to (7,7)
    // Normal perpendicular to (1,1): normalized (-1,1)/sqrt(2)
    double normal[] = {-1.0/sqrt(2.0), 1.0/sqrt(2.0)};
    double point[] = {4.5, 4.5};  // Midpoint
    double half_length = 2.5 * sqrt(2.0);
    double bounds_min[] = {-half_length};
    double bounds_max[] = {half_length};
    
    int bid = add_test_boundary(grid, normal, point, bounds_min, bounds_max);
    TEST_ASSERT(bid >= 0, "Failed to add boundary");
    
    // Test points on the diagonal within bounds
    double test_point1[] = {4.5, 4.5};  // Middle
    int result1 = grid_point_near_boundary(grid, test_point1);
    TEST_ASSERT(result1 > 0, "Point in middle of diagonal should be on boundary");
    
    double test_point2[] = {2.0, 2.0};  // Start
    int result2 = grid_point_near_boundary(grid, test_point2);
    TEST_ASSERT(result2 > 0, "Start point should be on boundary");
    
    double test_point3[] = {7.0, 7.0};  // End
    int result3 = grid_point_near_boundary(grid, test_point3);
    TEST_ASSERT(result3 > 0, "End point should be on boundary");
    
    // Test point on line but outside bounds
    double test_point4[] = {8.0, 8.0};  // Beyond end
    int result4 = grid_point_near_boundary(grid, test_point4);
    TEST_ASSERT(result4 == 0, "Point beyond diagonal should NOT be on boundary");
    
    grid_metadata_free(grid);
    TEST_PASS("2D diagonal line segment");
}

// Test 4: 3D Rectangular Surface Patch
int test_3d_surface_patch() {
    uint32_t dims[] = {50, 50, 50};
    double spacing[] = {0.2, 0.2, 0.2};
    double origin[] = {0.0, 0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 3);
    
    // Rectangular patch in xy-plane at z=5
    // From (2,3) to (7,8) in xy
    // Normal pointing up in z: (0, 0, 1)
    double normal[] = {0.0, 0.0, 1.0};
    double point[] = {4.5, 5.5, 5.0};  // Center of patch
    double bounds_min[] = {-2.5, -2.5};  // Half-extents
    double bounds_max[] = {2.5, 2.5};
    
    int bid = add_test_boundary(grid, normal, point, bounds_min, bounds_max);
    TEST_ASSERT(bid >= 0, "Failed to add boundary");
    
    // Test points on the surface within bounds
    double test_point1[] = {4.5, 5.5, 5.0};  // Middle of patch
    int result1 = grid_point_near_boundary(grid, test_point1);
    TEST_ASSERT(result1 > 0, "Point in middle of patch should be on boundary");
    
    double test_point2[] = {2.0, 3.0, 5.0};  // Corner
    int result2 = grid_point_near_boundary(grid, test_point2);
    TEST_ASSERT(result2 > 0, "Corner should be on boundary");
    
    double test_point3[] = {7.0, 8.0, 5.0};  // Opposite corner
    int result3 = grid_point_near_boundary(grid, test_point3);
    TEST_ASSERT(result3 > 0, "Opposite corner should be on boundary");
    
    // Test point on plane but outside patch bounds
    double test_point4[] = {1.0, 4.0, 5.0};  // Before x bounds
    int result4 = grid_point_near_boundary(grid, test_point4);
    TEST_ASSERT(result4 == 0, "Point outside patch x-bounds should NOT be on boundary");
    
    double test_point5[] = {4.5, 9.0, 5.0};  // Beyond y bounds
    int result5 = grid_point_near_boundary(grid, test_point5);
    TEST_ASSERT(result5 == 0, "Point outside patch y-bounds should NOT be on boundary");
    
    // Test point off the plane
    double test_point6[] = {4.5, 5.5, 6.0};  // Above plane
    int result6 = grid_point_near_boundary(grid, test_point6);
    TEST_ASSERT(result6 == 0, "Point off plane should NOT be on boundary");
    
    grid_metadata_free(grid);
    TEST_PASS("3D rectangular surface patch");
}

// Test 5: 3D Tilted Surface Patch
int test_3d_tilted_surface() {
    uint32_t dims[] = {50, 50, 50};
    double spacing[] = {0.2, 0.2, 0.2};
    double origin[] = {0.0, 0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 3);
    
    // Surface tilted at 45 degrees in xz-plane
    // Normal: (1, 0, 1)/sqrt(2)
    double normal[] = {1.0/sqrt(2.0), 0.0, 1.0/sqrt(2.0)};
    double point[] = {5.0, 3.0, 5.0};  // Reference point
    double bounds_min[] = {-2.0, -2.0};  // 2D parametric bounds
    double bounds_max[] = {2.0, 2.0};
    
    int bid = add_test_boundary(grid, normal, point, bounds_min, bounds_max);
    TEST_ASSERT(bid >= 0, "Failed to add boundary");
    
    // Test point near center of tilted surface
    double test_point1[] = {5.0, 3.0, 5.0};  // Center (reference point)
    int result1 = grid_point_near_boundary(grid, test_point1);
    TEST_ASSERT(result1 > 0, "Center of tilted surface should be on boundary");
    
    // Test point on plane within bounds (move along y direction)
    double test_point2[] = {5.0, 4.5, 5.0};
    int result2 = grid_point_near_boundary(grid, test_point2);
    TEST_ASSERT(result2 > 0, "Point within bounds should be on boundary");
    
    // Test point far from surface
    double test_point3[] = {10.0, 3.0, 10.0};
    int result3 = grid_point_near_boundary(grid, test_point3);
    TEST_ASSERT(result3 == 0, "Point far from surface should NOT be on boundary");
    
    grid_metadata_free(grid);
    TEST_PASS("3D tilted surface patch");
}

// Test 6: Unbounded Hyperplane
int test_unbounded_hyperplane() {
    uint32_t dims[] = {100, 100};
    double spacing[] = {0.1, 0.1};
    double origin[] = {0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    
    // Unbounded horizontal line at y=5
    double normal[] = {0.0, 1.0};
    double point[] = {0.0, 5.0};
    
    int bid = grid_add_hyperplane_boundary(grid, normal, point, NULL, NULL, 
                                           BC_REFLECT, 1.0);
    TEST_ASSERT(bid >= 0, "Failed to add unbounded boundary");
    
    // Any point on the line should be detected
    double test_point1[] = {2.0, 5.0};
    int result1 = grid_point_near_boundary(grid, test_point1);
    TEST_ASSERT(result1 > 0, "Point on unbounded line should be detected");
    
    double test_point2[] = {50.0, 5.0};  // Far away but still on line
    int result2 = grid_point_near_boundary(grid, test_point2);
    TEST_ASSERT(result2 > 0, "Far point on unbounded line should be detected");
    
    // Point off the line should not be detected (1.0 unit away from y=5)
    double test_point3[] = {2.0, 6.0};
    int result3 = grid_point_near_boundary(grid, test_point3);
    if (result3 != 0) {
        printf("  Debug: point (2.0, 6.0) is %.10f units from line at y=5\n", 6.0 - 5.0);
        printf("  This is expected - tolerance is very small (1e-10)\n");
    }
    TEST_ASSERT(result3 == 0, "Point off line should NOT be detected");
    
    grid_metadata_free(grid);
    TEST_PASS("Unbounded hyperplane");
}

// Test 7: Multiple Boundaries
int test_multiple_boundaries() {
    uint32_t dims[] = {100, 100};
    double spacing[] = {0.1, 0.1};
    double origin[] = {0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    
    // Add two perpendicular line segments
    double normal1[] = {0.0, 1.0};
    double point1[] = {2.0, 5.0};
    double bounds_min1[] = {0.0};
    double bounds_max1[] = {3.0};
    int bid1 = add_test_boundary(grid, normal1, point1, bounds_min1, bounds_max1);
    
    double normal2[] = {1.0, 0.0};
    double point2[] = {5.0, 2.0};
    double bounds_min2[] = {0.0};
    double bounds_max2[] = {3.0};
    int bid2 = add_test_boundary(grid, normal2, point2, bounds_min2, bounds_max2);
    
    TEST_ASSERT(bid1 >= 0 && bid2 >= 0, "Failed to add boundaries");
    
    // Test point on first boundary
    double test_point1[] = {3.5, 5.0};
    int result1 = grid_point_near_boundary(grid, test_point1);
    TEST_ASSERT(result1 > 0, "Point on first boundary should be detected");
    
    // Test point on second boundary
    double test_point2[] = {5.0, 3.5};
    int result2 = grid_point_near_boundary(grid, test_point2);
    TEST_ASSERT(result2 > 0, "Point on second boundary should be detected");
    
    // Test point not on either boundary
    double test_point3[] = {3.0, 3.0};
    int result3 = grid_point_near_boundary(grid, test_point3);
    TEST_ASSERT(result3 == 0, "Point not on either boundary should NOT be detected");
    
    grid_metadata_free(grid);
    TEST_PASS("Multiple boundaries");
}

// Test 8: Edge Cases - Tolerance
int test_tolerance() {
    uint32_t dims[] = {100, 100};
    double spacing[] = {0.01, 0.01};  // Fine grid
    double origin[] = {0.0, 0.0};
    GridMetadata *grid = grid_metadata_create(dims, spacing, origin, 2);
    
    // Line at y=0.5
    double normal[] = {0.0, 1.0};
    double point[] = {0.5, 0.5};  // Midpoint
    double bounds_min[] = {-0.5};
    double bounds_max[] = {0.5};
    int bid = add_test_boundary(grid, normal, point, bounds_min, bounds_max);
    (void)bid;  // Suppress unused warning
    
    // Point exactly on line
    double test_point1[] = {0.5, 0.5};
    int result1 = grid_point_near_boundary(grid, test_point1);
    TEST_ASSERT(result1 > 0, "Point exactly on line should be detected");
    
    // Point very slightly off line (within floating point precision)
    // Note: tolerance in grid_point_near_boundary is 1e-10
    double test_point2[] = {0.5, 0.5 + 1e-15};
    int result2 = grid_point_near_boundary(grid, test_point2);
    TEST_ASSERT(result2 > 0, "Point within floating point precision should be detected");
    
    // Point clearly off the line (much larger than tolerance)
    double test_point3[] = {0.5, 0.51};
    int result3 = grid_point_near_boundary(grid, test_point3);
    TEST_ASSERT(result3 == 0, "Point far from line should NOT be detected");
    
    grid_metadata_free(grid);
    TEST_PASS("Tolerance edge cases");
}

int main(int argc, char **argv) {
    int passed = 0;
    int total = 0;
    
    printf("=== Testing Bounded Hyperplane Functionality ===\n\n");
    
    total++; passed += test_2d_horizontal_line();
    total++; passed += test_2d_vertical_line();
    total++; passed += test_2d_diagonal_line();
    total++; passed += test_3d_surface_patch();
    total++; passed += test_3d_tilted_surface();
    total++; passed += test_unbounded_hyperplane();
    total++; passed += test_multiple_boundaries();
    total++; passed += test_tolerance();
    
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d/%d\n", passed, total);
    
    if (passed == total) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("Some tests failed.\n");
        return 1;
    }
}
