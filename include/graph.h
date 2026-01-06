#ifndef GRAPH_H
#define GRAPH_H

#include "expression.h"
#include "dictionary.h"
#include <stdbool.h>

// ============================================================================
// Dependency Graph for Equation Solving
// ============================================================================
// Tracks dependencies between variables and their defining expressions
// Supports cycle detection, SCC analysis, and topological ordering

// ============================================================================
// Graph Node
// ============================================================================

// DFS colors for cycle detection
typedef enum {
    COLOR_WHITE,  // Unvisited
    COLOR_GRAY,   // Being processed (on stack)
    COLOR_BLACK   // Fully processed
} NodeColor;

typedef struct GraphNode {
    char *variable;                    // Variable name
    Expression *definition;            // Expression defining this variable
    
    char **dependencies;               // Array of variable names this depends on
    int n_dependencies;
    int dependency_capacity;
    
    // For graph algorithms
    NodeColor color;                   // DFS color
    int discovery_time;
    int finish_time;
    int component_id;                  // Strongly connected component ID
    int topological_order;             // Order in topological sort
} GraphNode;

// ============================================================================
// Dependency Graph
// ============================================================================

typedef struct {
    GraphNode **nodes;                 // Array of graph nodes
    int n_nodes;
    int capacity;
    
    Dictionary *var_to_index;          // Variable name -> node index mapping
} DependencyGraph;

// Create empty dependency graph
// Returns: Newly allocated DependencyGraph (caller owns, must call graph_free)
// Memory: Allocates graph structure and internal dictionaries
DependencyGraph* graph_create(void);

// Add variable definition to graph: var = expr
// graph: Graph to add to (must not be NULL)
// var: Variable name being defined (copied internally)
// expr: Expression defining the variable (ownership transferred to graph)
// Memory: Graph takes ownership of expr, will free in graph_free
// Note: Automatically extracts dependencies from expr
void graph_add_definition(DependencyGraph *graph, const char *var, Expression *expr);

// Get node by variable name
// graph: Graph to query (can be NULL, returns NULL)
// var: Variable name to find (must not be NULL)
// Returns: Pointer to GraphNode (owned by graph, do not free)
//          Returns NULL if variable not found
GraphNode* graph_get_node(DependencyGraph *graph, const char *var);

// Free graph and all owned resources
// graph: Graph to free (can be NULL)
// Memory: Frees all nodes, expressions, dependency arrays, and internal dictionaries
// Note: Does free expressions (ownership transferred in graph_add_definition)
void graph_free(DependencyGraph *graph);

// ============================================================================
// Variable Extraction from Expressions
// ============================================================================

// Extract all variables referenced in an expression
// expr: Expression to extract from (must not be NULL)
// count: Output parameter for array size (must not be NULL)
// Returns: Array of variable name strings (caller owns, must free array and strings)
// Memory: Caller must free returned array with free() and each string with free()
char** extract_variables_from_expr(Expression *expr, int *count);

// ============================================================================
// Cycle Detection
// ============================================================================

typedef enum {
    GRAPH_ACYCLIC,           // No cycles - can solve by substitution
    GRAPH_HAS_CYCLES,        // Cycles present - need iterative solver
    GRAPH_SELF_REFERENTIAL   // Variable directly references itself
} GraphCycleStatus;

// Detect if graph has any cycles
// graph: Graph to analyze (must not be NULL)
// cycle_path: Output for cycle path if found (must not be NULL)
// cycle_length: Output for cycle length (must not be NULL)
// Returns: GRAPH_ACYCLIC, GRAPH_HAS_CYCLES, or GRAPH_SELF_REFERENTIAL
// Memory: If cycle found, caller must free cycle_path array and each string
GraphCycleStatus graph_detect_cycles(DependencyGraph *graph, 
                                     char ***cycle_path, 
                                     int *cycle_length);

// ============================================================================
// Strongly Connected Components (Tarjan's Algorithm)
// ============================================================================

// Find all strongly connected components
// graph: Graph to analyze (must not be NULL)
// component_assignment: Output array mapping node index to component ID (must not be NULL)
// Returns: Number of components found
// Memory: Caller must free component_assignment array with free()
int graph_find_sccs(DependencyGraph *graph, int **component_assignment);

// ============================================================================
// Topological Sort (Kahn's Algorithm)
// ============================================================================

// Compute topological ordering of variables for evaluation
// graph: Graph to sort (must not be NULL)
// ordered_vars: Output array of variable names in evaluation order (must not be NULL)
// n_vars: Output for number of variables (must not be NULL)
// Returns: true if successful (graph is acyclic), false if cycles exist
// Memory: Caller must free ordered_vars array with free() and each string with free()
bool graph_topological_sort(DependencyGraph *graph, 
                            char ***ordered_vars, 
                            int *n_vars);

// ============================================================================
// Dependency Queries
// ============================================================================

// Check if var1 depends on var2 (directly or transitively)
bool graph_depends_on(DependencyGraph *graph, const char *var1, const char *var2);

// Find all variables that depend on a given variable
// Caller must free result array and strings
char** graph_find_dependents(DependencyGraph *graph, const char *var, int *count);

// Compute dependency depth (longest path to a leaf node)
int graph_dependency_depth(DependencyGraph *graph, const char *var);

// ============================================================================
// System Analysis
// ============================================================================

typedef struct {
    bool is_acyclic;                   // Can be solved by substitution
    int n_components;                  // Number of strongly connected components
    int max_component_size;            // Size of largest component
    int *component_sizes;              // Size of each component
    int n_isolated_vars;               // Variables with no dependencies
} SystemStructure;

// Analyze the structure of the equation system
SystemStructure graph_analyze_structure(DependencyGraph *graph);

// Free system structure
void system_structure_free(SystemStructure *structure);

#endif // GRAPH_H
