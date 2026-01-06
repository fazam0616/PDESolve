#include "../include/graph.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Graph Creation and Management
// ============================================================================

DependencyGraph* graph_create(void) {
    DependencyGraph *graph = malloc(sizeof(DependencyGraph));
    if (graph == NULL) return NULL;
    
    graph->capacity = 16;
    graph->n_nodes = 0;
    graph->nodes = malloc(sizeof(GraphNode*) * graph->capacity);
    graph->var_to_index = dict_create(32);
    
    if (graph->nodes == NULL || graph->var_to_index == NULL) {
        if (graph->nodes) free(graph->nodes);
        if (graph->var_to_index) dict_free(graph->var_to_index);
        free(graph);
        return NULL;
    }
    
    return graph;
}

static GraphNode* graph_node_create(const char *var, Expression *definition) {
    GraphNode *node = malloc(sizeof(GraphNode));
    if (node == NULL) return NULL;
    
    node->variable = malloc(strlen(var) + 1);
    if (node->variable == NULL) {
        free(node);
        return NULL;
    }
    strcpy(node->variable, var);
    
    node->definition = definition;
    node->dependency_capacity = 8;
    node->n_dependencies = 0;
    node->dependencies = malloc(sizeof(char*) * node->dependency_capacity);
    
    if (node->dependencies == NULL) {
        free(node->variable);
        free(node);
        return NULL;
    }
    
    // Initialize algorithm fields
    node->color = COLOR_WHITE;
    node->discovery_time = -1;
    node->finish_time = -1;
    node->component_id = -1;
    node->topological_order = -1;
    
    return node;
}

static void graph_node_add_dependency(GraphNode *node, const char *dep_var) {
    // Check if dependency already exists
    for (int i = 0; i < node->n_dependencies; i++) {
        if (strcmp(node->dependencies[i], dep_var) == 0) {
            return;  // Already present
        }
    }
    
    // Expand array if needed
    if (node->n_dependencies >= node->dependency_capacity) {
        node->dependency_capacity *= 2;
        char **new_deps = realloc(node->dependencies, 
                                  sizeof(char*) * node->dependency_capacity);
        if (new_deps == NULL) return;
        node->dependencies = new_deps;
    }
    
    // Add dependency
    node->dependencies[node->n_dependencies] = malloc(strlen(dep_var) + 1);
    if (node->dependencies[node->n_dependencies] != NULL) {
        strcpy(node->dependencies[node->n_dependencies], dep_var);
        node->n_dependencies++;
    }
}

void graph_add_definition(DependencyGraph *graph, const char *var, Expression *expr) {
    if (graph == NULL || var == NULL || expr == NULL) return;
    
    // Check if variable already exists
    Literal *temp;
    int index = -1;
    if (dict_get(graph->var_to_index, var, &temp)) {
        // Variable exists - update definition
        index = (int)temp->field[0];
        if (index >= 0 && index < graph->n_nodes) {
            graph->nodes[index]->definition = expr;
        
            // Clear old dependencies
            for (int i = 0; i < graph->nodes[index]->n_dependencies; i++) {
                free(graph->nodes[index]->dependencies[i]);
            }
            graph->nodes[index]->n_dependencies = 0;
        }
    } else {
        // New variable - create node
        GraphNode *node = graph_node_create(var, expr);
        if (node == NULL) return;
        
        // Expand array if needed
        if (graph->n_nodes >= graph->capacity) {
            graph->capacity *= 2;
            GraphNode **new_nodes = realloc(graph->nodes, 
                                           sizeof(GraphNode*) * graph->capacity);
            if (new_nodes == NULL) {
                free(node->variable);
                free(node->dependencies);
                free(node);
                return;
            }
            graph->nodes = new_nodes;
        }
        
        index = graph->n_nodes;
        graph->nodes[index] = node;
        graph->n_nodes++;
        
        // Add to lookup dictionary
        // Create Literal to store index
        Literal *index_lit = literal_create_scalar((double)index);
        dict_set(graph->var_to_index, var, index_lit);
        literal_free(index_lit);
    }
    
    // Extract and add dependencies
    int n_deps;
    char **deps = extract_variables_from_expr(expr, &n_deps);
    for (int i = 0; i < n_deps; i++) {
        graph_node_add_dependency(graph->nodes[index], deps[i]);
        free(deps[i]);
    }
    free(deps);
}

GraphNode* graph_get_node(DependencyGraph *graph, const char *var) {
    if (graph == NULL || var == NULL) return NULL;
    
    Literal *temp;
    if (dict_get(graph->var_to_index, var, &temp)) {
        int index = (int)temp->field[0];
        if (index >= 0 && index < graph->n_nodes) {
            return graph->nodes[index];
        }
    }
    
    return NULL;
}

void graph_free(DependencyGraph *graph) {
    if (graph == NULL) return;
    
    for (int i = 0; i < graph->n_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        free(node->variable);
        for (int j = 0; j < node->n_dependencies; j++) {
            free(node->dependencies[j]);
        }
        free(node->dependencies);
        free(node);
    }
    
    // Free field arrays stored in var_to_index dictionary
    DictIterator iter = dict_iterator(graph->var_to_index);
    char *key;
    Literal *value;
    while (dict_next(&iter, &key, &value)) {
        free(value->field);
    }
    
    free(graph->nodes);
    dict_free(graph->var_to_index);
    free(graph);
}

// ============================================================================
// Variable Extraction
// ============================================================================

static void add_variable_to_list(char ***list, int *count, int *capacity, const char *var) {
    // Check if already in list
    for (int i = 0; i < *count; i++) {
        if (strcmp((*list)[i], var) == 0) {
            return;
        }
    }
    
    // Expand if needed
    if (*count >= *capacity) {
        *capacity *= 2;
        char **new_list = realloc(*list, sizeof(char*) * (*capacity));
        if (new_list == NULL) return;
        *list = new_list;
    }
    
    // Add variable
    (*list)[*count] = malloc(strlen(var) + 1);
    if ((*list)[*count] != NULL) {
        strcpy((*list)[*count], var);
        (*count)++;
    }
}

static void extract_variables_recursive(Expression *expr, char ***vars, int *count, int *capacity) {
    if (expr == NULL) return;
    
    switch (expr->type) {
        case EXPR_VARIABLE:
            add_variable_to_list(vars, count, capacity, expr->data.variable);
            break;
        
        case EXPR_UNARY:
            extract_variables_recursive(expr->data.unary.operand, vars, count, capacity);
            break;
            
        case EXPR_BINARY:
            extract_variables_recursive(expr->data.binary.left, vars, count, capacity);
            extract_variables_recursive(expr->data.binary.right, vars, count, capacity);
            break;
        
        case EXPR_LITERAL:
            // No variables in literals
            break;
    }
}

char** extract_variables_from_expr(Expression *expr, int *count) {
    *count = 0;
    int capacity = 8;
    char **vars = malloc(sizeof(char*) * capacity);
    if (vars == NULL) return NULL;
    
    extract_variables_recursive(expr, &vars, count, &capacity);
    
    return vars;
}

// ============================================================================
// Cycle Detection (DFS-based)
// ============================================================================

static bool dfs_detect_cycle_recursive(DependencyGraph *graph, GraphNode *node, 
                                       char ***cycle_path, int *cycle_length) {
    if (graph == NULL || node == NULL) return false;
    
    node->color = COLOR_GRAY;
    
    for (int i = 0; i < node->n_dependencies; i++) {
        if (node->dependencies[i] == NULL) continue;
        GraphNode *dep = graph_get_node(graph, node->dependencies[i]);
        if (dep == NULL) continue;  // Undefined variable
        
        if (dep->color == COLOR_GRAY) {
            // Back edge - cycle detected!
            *cycle_length = 2;
            *cycle_path = malloc(sizeof(char*) * 2);
            if (*cycle_path != NULL) {
                (*cycle_path)[0] = malloc(strlen(dep->variable) + 1);
                (*cycle_path)[1] = malloc(strlen(node->variable) + 1);
                if ((*cycle_path)[0] && (*cycle_path)[1]) {
                    strcpy((*cycle_path)[0], dep->variable);
                    strcpy((*cycle_path)[1], node->variable);
                }
            }
            return true;
        }
        
        if (dep->color == COLOR_WHITE) {
            if (dfs_detect_cycle_recursive(graph, dep, cycle_path, cycle_length)) {
                // Propagate cycle path
                (*cycle_length)++;
                char **new_path = realloc(*cycle_path, sizeof(char*) * (*cycle_length));
                if (new_path != NULL) {
                    *cycle_path = new_path;
                    (*cycle_path)[*cycle_length - 1] = malloc(strlen(node->variable) + 1);
                    if ((*cycle_path)[*cycle_length - 1] != NULL) {
                        strcpy((*cycle_path)[*cycle_length - 1], node->variable);
                    }
                }
                return true;
            }
        }
    }
    
    node->color = COLOR_BLACK;
    return false;
}

GraphCycleStatus graph_detect_cycles(DependencyGraph *graph, 
                                     char ***cycle_path, 
                                     int *cycle_length) {
    if (graph == NULL) return GRAPH_ACYCLIC;
    
    *cycle_path = NULL;
    *cycle_length = 0;
    
    // Initialize all nodes to WHITE
    for (int i = 0; i < graph->n_nodes; i++) {
        graph->nodes[i]->color = COLOR_WHITE;
    }
    
    // Check for self-references
    for (int i = 0; i < graph->n_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        for (int j = 0; j < node->n_dependencies; j++) {
            if (strcmp(node->variable, node->dependencies[j]) == 0) {
                *cycle_length = 1;
                *cycle_path = malloc(sizeof(char*));
                if (*cycle_path != NULL) {
                    (*cycle_path)[0] = malloc(strlen(node->variable) + 1);
                    if ((*cycle_path)[0] != NULL) {
                        strcpy((*cycle_path)[0], node->variable);
                    }
                }
                return GRAPH_SELF_REFERENTIAL;
            }
        }
    }
    
    // DFS from each unvisited node
    for (int i = 0; i < graph->n_nodes; i++) {
        if (graph->nodes[i]->color == COLOR_WHITE) {
            if (dfs_detect_cycle_recursive(graph, graph->nodes[i], 
                                          cycle_path, cycle_length)) {
                return GRAPH_HAS_CYCLES;
            }
        }
    }
    
    return GRAPH_ACYCLIC;
}

// ============================================================================
// Strongly Connected Components (Tarjan's Algorithm)
// ============================================================================

typedef struct {
    GraphNode **stack;
    int stack_size;
    bool *on_stack;
    int *low_link;
    int *indices;
    int index_counter;
    int component_counter;
} TarjanState;

static void tarjan_scc_recursive(DependencyGraph *graph, GraphNode *node, 
                                 int node_idx, TarjanState *state) {
    state->indices[node_idx] = state->index_counter;
    state->low_link[node_idx] = state->index_counter;
    state->index_counter++;
    
    state->stack[state->stack_size++] = node;
    state->on_stack[node_idx] = true;
    
    // Explore dependencies
    for (int i = 0; i < node->n_dependencies; i++) {
        GraphNode *dep = graph_get_node(graph, node->dependencies[i]);
        if (dep == NULL) continue;
        
        // Find dependency index
        int dep_idx = -1;
        for (int j = 0; j < graph->n_nodes; j++) {
            if (graph->nodes[j] == dep) {
                dep_idx = j;
                break;
            }
        }
        if (dep_idx == -1) continue;
        
        if (state->indices[dep_idx] == -1) {
            tarjan_scc_recursive(graph, dep, dep_idx, state);
            state->low_link[node_idx] = (state->low_link[node_idx] < state->low_link[dep_idx]) ?
                                        state->low_link[node_idx] : state->low_link[dep_idx];
        } else if (state->on_stack[dep_idx]) {
            state->low_link[node_idx] = (state->low_link[node_idx] < state->indices[dep_idx]) ?
                                        state->low_link[node_idx] : state->indices[dep_idx];
        }
    }
    
    // Root of SCC?
    if (state->low_link[node_idx] == state->indices[node_idx]) {
        GraphNode *w;
        do {
            w = state->stack[--state->stack_size];
            int w_idx = -1;
            for (int j = 0; j < graph->n_nodes; j++) {
                if (graph->nodes[j] == w) {
                    w_idx = j;
                    break;
                }
            }
            if (w_idx != -1) {
                state->on_stack[w_idx] = false;
                w->component_id = state->component_counter;
            }
        } while (w != node);
        
        state->component_counter++;
    }
}

int graph_find_sccs(DependencyGraph *graph, int **component_assignment) {
    if (graph == NULL || graph->n_nodes == 0) return 0;
    
    TarjanState state;
    state.stack = malloc(sizeof(GraphNode*) * graph->n_nodes);
    state.stack_size = 0;
    state.on_stack = calloc(graph->n_nodes, sizeof(bool));
    state.low_link = malloc(sizeof(int) * graph->n_nodes);
    state.indices = malloc(sizeof(int) * graph->n_nodes);
    state.index_counter = 0;
    state.component_counter = 0;
    
    // Initialize indices
    for (int i = 0; i < graph->n_nodes; i++) {
        state.indices[i] = -1;
    }
    
    // Run Tarjan's algorithm
    for (int i = 0; i < graph->n_nodes; i++) {
        if (state.indices[i] == -1) {
            tarjan_scc_recursive(graph, graph->nodes[i], i, &state);
        }
    }
    
    // Extract component assignments
    *component_assignment = malloc(sizeof(int) * graph->n_nodes);
    if (*component_assignment != NULL) {
        for (int i = 0; i < graph->n_nodes; i++) {
            (*component_assignment)[i] = graph->nodes[i]->component_id;
        }
    }
    
    int n_components = state.component_counter;
    
    // Cleanup
    free(state.stack);
    free(state.on_stack);
    free(state.low_link);
    free(state.indices);
    
    return n_components;
}

// ============================================================================
// Topological Sort (Kahn's Algorithm)
// ============================================================================

bool graph_topological_sort(DependencyGraph *graph, 
                            char ***ordered_vars, 
                            int *n_vars) {
    if (graph == NULL || graph->n_nodes == 0) {
        *n_vars = 0;
        *ordered_vars = NULL;
        return true;
    }
    
    // Calculate in-degrees
    int *in_degree = calloc(graph->n_nodes, sizeof(int));
    if (in_degree == NULL) return false;
    
    for (int i = 0; i < graph->n_nodes; i++) {
        for (int j = 0; j < graph->nodes[i]->n_dependencies; j++) {
            GraphNode *dep = graph_get_node(graph, graph->nodes[i]->dependencies[j]);
            if (dep != NULL) {
                for (int k = 0; k < graph->n_nodes; k++) {
                    if (graph->nodes[k] == dep) {
                        in_degree[k]++;
                        break;
                    }
                }
            }
        }
    }
    
    // Queue for nodes with in-degree 0
    GraphNode **queue = malloc(sizeof(GraphNode*) * graph->n_nodes);
    if (queue == NULL) {
        free(in_degree);
        return false;
    }
    
    int queue_front = 0, queue_back = 0;
    
    for (int i = 0; i < graph->n_nodes; i++) {
        if (in_degree[i] == 0) {
            queue[queue_back++] = graph->nodes[i];
        }
    }
    
    // Process queue
    *ordered_vars = malloc(sizeof(char*) * graph->n_nodes);
    if (*ordered_vars == NULL) {
        free(in_degree);
        free(queue);
        return false;
    }
    
    int output_idx = 0;
    
    while (queue_front < queue_back) {
        GraphNode *node = queue[queue_front++];
        
        (*ordered_vars)[output_idx] = malloc(strlen(node->variable) + 1);
        if ((*ordered_vars)[output_idx] != NULL) {
            strcpy((*ordered_vars)[output_idx], node->variable);
            output_idx++;
        }
        
        // Reduce in-degree of dependencies
        for (int i = 0; i < node->n_dependencies; i++) {
            GraphNode *dep = graph_get_node(graph, node->dependencies[i]);
            if (dep == NULL) continue;
            
            for (int j = 0; j < graph->n_nodes; j++) {
                if (graph->nodes[j] == dep) {
                    in_degree[j]--;
                    if (in_degree[j] == 0) {
                        queue[queue_back++] = dep;
                    }
                    break;
                }
            }
        }
    }
    
    *n_vars = output_idx;
    
    free(in_degree);
    free(queue);
    
    return (output_idx == graph->n_nodes);
}

// ============================================================================
// Dependency Queries
// ============================================================================

bool graph_depends_on(DependencyGraph *graph, const char *var1, const char *var2) {
    GraphNode *node = graph_get_node(graph, var1);
    if (node == NULL) return false;
    
    // Direct dependency?
    for (int i = 0; i < node->n_dependencies; i++) {
        if (strcmp(node->dependencies[i], var2) == 0) {
            return true;
        }
    }
    
    // Transitive dependency?
    for (int i = 0; i < node->n_dependencies; i++) {
        if (graph_depends_on(graph, node->dependencies[i], var2)) {
            return true;
        }
    }
    
    return false;
}

char** graph_find_dependents(DependencyGraph *graph, const char *var, int *count) {
    *count = 0;
    if (graph == NULL || var == NULL) return NULL;
    
    int capacity = 8;
    char **dependents = malloc(sizeof(char*) * capacity);
    if (dependents == NULL) return NULL;
    
    for (int i = 0; i < graph->n_nodes; i++) {
        if (graph_depends_on(graph, graph->nodes[i]->variable, var)) {
            if (*count >= capacity) {
                capacity *= 2;
                char **new_deps = realloc(dependents, sizeof(char*) * capacity);
                if (new_deps == NULL) break;
                dependents = new_deps;
            }
            
            dependents[*count] = malloc(strlen(graph->nodes[i]->variable) + 1);
            if (dependents[*count] != NULL) {
                strcpy(dependents[*count], graph->nodes[i]->variable);
                (*count)++;
            }
        }
    }
    
    return dependents;
}

int graph_dependency_depth(DependencyGraph *graph, const char *var) {
    GraphNode *node = graph_get_node(graph, var);
    if (node == NULL || node->n_dependencies == 0) {
        return 0;
    }
    
    int max_depth = 0;
    for (int i = 0; i < node->n_dependencies; i++) {
        int depth = graph_dependency_depth(graph, node->dependencies[i]);
        if (depth > max_depth) {
            max_depth = depth;
        }
    }
    
    return max_depth + 1;
}

// ============================================================================
// System Analysis
// ============================================================================

SystemStructure graph_analyze_structure(DependencyGraph *graph) {
    SystemStructure structure;
    structure.is_acyclic = false;
    structure.n_components = 0;
    structure.max_component_size = 0;
    structure.component_sizes = NULL;
    structure.n_isolated_vars = 0;
    
    if (graph == NULL || graph->n_nodes == 0) {
        return structure;
    }
    
    // Check if acyclic
    char **cycle_path;
    int cycle_length;
    GraphCycleStatus status = graph_detect_cycles(graph, &cycle_path, &cycle_length);
    structure.is_acyclic = (status == GRAPH_ACYCLIC);
    
    if (cycle_path != NULL) {
        for (int i = 0; i < cycle_length; i++) {
            free(cycle_path[i]);
        }
        free(cycle_path);
    }
    
    // Find SCCs
    int *component_assignment;
    structure.n_components = graph_find_sccs(graph, &component_assignment);
    
    if (structure.n_components > 0 && component_assignment != NULL) {
        structure.component_sizes = calloc(structure.n_components, sizeof(int));
        
        if (structure.component_sizes != NULL) {
            for (int i = 0; i < graph->n_nodes; i++) {
                structure.component_sizes[component_assignment[i]]++;
                
                int size = structure.component_sizes[component_assignment[i]];
                if (size > structure.max_component_size) {
                    structure.max_component_size = size;
                }
            }
        }
        
        free(component_assignment);
    }
    
    // Count isolated variables
    for (int i = 0; i < graph->n_nodes; i++) {
        if (graph->nodes[i]->n_dependencies == 0) {
            structure.n_isolated_vars++;
        }
    }
    
    return structure;
}

void system_structure_free(SystemStructure *structure) {
    if (structure != NULL && structure->component_sizes != NULL) {
        free(structure->component_sizes);
        structure->component_sizes = NULL;
    }
}
