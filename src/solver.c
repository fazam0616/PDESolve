#include "../include/solver.h"
#include "../include/grid.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// System Creation and Management
// ============================================================================

PDESystem* pde_system_create(void) {
    PDESystem *sys = malloc(sizeof(PDESystem));
    if (!sys) return NULL;
    
    sys->equations = NULL;
    sys->n_equations = 0;
    sys->unknowns = NULL;
    sys->n_unknowns = 0;
    sys->parameters = dict_create(16);
    sys->grid_context = NULL;
    sys->tolerance = 1e-6;
    sys->max_iterations = 1000;
    
    return sys;
}

void pde_system_add_equation(PDESystem *sys, Expression *equation) {
    if (!sys || !equation) return;
    
    sys->n_equations++;
    sys->equations = realloc(sys->equations, sizeof(Expression*) * sys->n_equations);
    sys->equations[sys->n_equations - 1] = equation;
}

void pde_system_set_unknowns(PDESystem *sys, char **unknowns, int n) {
    if (!sys || !unknowns) return;
    
    // Free existing unknowns
    if (sys->unknowns) {
        for (int i = 0; i < sys->n_unknowns; i++) {
            free(sys->unknowns[i]);
        }
        free(sys->unknowns);
    }
    
    sys->n_unknowns = n;
    sys->unknowns = malloc(sizeof(char*) * n);
    for (int i = 0; i < n; i++) {
        sys->unknowns[i] = strdup(unknowns[i]);
    }
}

void pde_system_set_parameter(PDESystem *sys, const char *name, double value) {
    if (!sys || !name) return;
    
    Literal *lit = literal_create_scalar(value);
    if (lit) {
        dict_set(sys->parameters, name, lit);
        literal_free(lit);
    }
}

void pde_system_set_grid(PDESystem *sys, GridMetadata *grid) {
    if (sys) {
        sys->grid_context = grid;
    }
}

void pde_system_set_tolerance(PDESystem *sys, double tol) {
    if (sys) sys->tolerance = tol;
}

void pde_system_set_max_iterations(PDESystem *sys, int max_iter) {
    if (sys) sys->max_iterations = max_iter;
}

void pde_system_free(PDESystem *sys) {
    if (!sys) return;
    
    // Free equations
    if (sys->equations) {
        for (int i = 0; i < sys->n_equations; i++) {
            expression_free(sys->equations[i]);
        }
        free(sys->equations);
    }
    
    // Free unknowns
    if (sys->unknowns) {
        for (int i = 0; i < sys->n_unknowns; i++) {
            free(sys->unknowns[i]);
        }
        free(sys->unknowns);
    }
    
    // Free parameters
    dict_free(sys->parameters);
    
    free(sys);
}

// ============================================================================
// Residual Computation
// ============================================================================

double* compute_residuals(PDESystem *sys, Dictionary *current_guess) {
    if (!sys || !current_guess) return NULL;
    
    double *residuals = malloc(sizeof(double) * sys->n_equations);
    
    // Merge parameters and current guess
    Dictionary *combined = dict_create(32);
    
    // Copy parameters
    DictIterator param_iter = dict_iterator(sys->parameters);
    char *key;
    Literal *val;
    while (dict_next(&param_iter, &key, &val)) {
        dict_set(combined, key, val);
    }
    
    // Copy current guess (overrides parameters if name conflicts)
    DictIterator guess_iter = dict_iterator(current_guess);
    while (dict_next(&guess_iter, &key, &val)) {
        dict_set(combined, key, val);
    }
    
    // Evaluate each equation
    for (int i = 0; i < sys->n_equations; i++) {
        Literal *result = expression_evaluate(sys->equations[i], combined);
        
        if (result && result->shape[0] == 1 && 
            result->shape[1] == 1 && result->shape[2] == 1) {
            residuals[i] = result->field[0];
        } else {
            residuals[i] = INFINITY;  // Invalid evaluation
        }
        
        if (result) literal_free(result);
    }
    
    dict_free(combined);
    return residuals;
}

double residual_norm(double *residuals, int n) {
    if (!residuals) return INFINITY;
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += residuals[i] * residuals[i];
    }
    return sqrt(sum);
}

// ============================================================================
// Solver Result Management
// ============================================================================

void solver_result_free(SolverResult *result) {
    if (!result) return;
    
    if (result->solution) dict_free(result->solution);
    if (result->message) free(result->message);
    free(result);
}

// ============================================================================
// Fixed-Point Solver
// ============================================================================

SolverResult* solve_fixed_point(PDESystem *sys, Dictionary *initial_guess) {
    if (!sys) return NULL;
    
    SolverResult *result = malloc(sizeof(SolverResult));
    result->solution = dict_create(sys->n_unknowns);
    result->iterations = 0;
    result->final_residual = INFINITY;
    result->message = NULL;
    
    // Check system validity
    if (!pde_system_is_well_posed(sys)) {
        result->status = SOLVER_INVALID_SYSTEM;
        result->message = strdup("System is not well-posed (n_equations != n_unknowns)");
        return result;
    }
    
    // Initialize current guess
    Dictionary *current = dict_create(sys->n_unknowns);
    
    if (initial_guess) {
        // Copy initial guess
        DictIterator iter = dict_iterator(initial_guess);
        char *key;
        Literal *val;
        while (dict_next(&iter, &key, &val)) {
            dict_set(current, key, val);
        }
    } else {
        // Default: start at origin
        for (int i = 0; i < sys->n_unknowns; i++) {
            Literal *lit = literal_create_scalar(0.0);
            if (lit) {
                dict_set(current, sys->unknowns[i], lit);
                literal_free(lit);
            }
        }
    }
    
    // Fixed-point iteration
    double damping = 0.5;  // Damping factor for stability
    
    for (int iter = 0; iter < sys->max_iterations; iter++) {
        result->iterations = iter + 1;
        
        // Compute residuals
        double *residuals = compute_residuals(sys, current);
        double norm = residual_norm(residuals, sys->n_equations);
        result->final_residual = norm;
        
        // Check convergence
        if (norm < sys->tolerance) {
            free(residuals);
            
            // Copy final solution
            DictIterator iter = dict_iterator(current);
            char *key;
            Literal *val;
            while (dict_next(&iter, &key, &val)) {
                dict_set(result->solution, key, val);
            }
            
            dict_free(current);
            result->status = SOLVER_SUCCESS;
            result->message = strdup("Converged successfully");
            return result;
        }
        
        // Update: x_new = x_old - damping * residual
        for (int i = 0; i < sys->n_unknowns; i++) {
            Literal *temp;
            if (dict_get(current, sys->unknowns[i], &temp)) {
                double old_val = temp->field[0];
                double new_val = old_val - damping * residuals[i];
                Literal *lit = literal_create_scalar(new_val);
                if (lit) {
                    dict_set(current, sys->unknowns[i], lit);
                    literal_free(lit);
                }
            }
        }
        
        free(residuals);
        
        // Check for divergence
        if (norm > 1e10 || isnan(norm)) {
            dict_free(current);
            result->status = SOLVER_DIVERGED;
            result->message = strdup("Solution diverged");
            return result;
        }
    }
    
    // Max iterations reached
    DictIterator iter = dict_iterator(current);
    char *key;
    Literal *val;
    while (dict_next(&iter, &key, &val)) {
        dict_set(result->solution, key, val);
    }
    
    dict_free(current);
    result->status = SOLVER_MAX_ITER;
    result->message = strdup("Maximum iterations reached");
    return result;
}

// ============================================================================
// Linear System Solver (Gaussian Elimination)
// ============================================================================

// Solve Ax = b using Gaussian elimination with partial pivoting
// A: n x n matrix (row-major: A[i*n + j])
// b: n x 1 vector
// x: n x 1 solution vector (output)
// Returns: true on success, false if singular
static bool gaussian_elimination(double *A, double *b, double *x, int n) {
    // Create augmented matrix [A|b]
    double *aug = malloc(sizeof(double) * n * (n + 1));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (n + 1) + j] = A[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }
    
    // Forward elimination with partial pivoting
    for (int k = 0; k < n; k++) {
        // Find pivot
        int pivot_row = k;
        double max_val = fabs(aug[k * (n + 1) + k]);
        for (int i = k + 1; i < n; i++) {
            double val = fabs(aug[i * (n + 1) + k]);
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }
        
        // Check for singular matrix
        if (max_val < 1e-14) {
            free(aug);
            return false;
        }
        
        // Swap rows
        if (pivot_row != k) {
            for (int j = 0; j < n + 1; j++) {
                double temp = aug[k * (n + 1) + j];
                aug[k * (n + 1) + j] = aug[pivot_row * (n + 1) + j];
                aug[pivot_row * (n + 1) + j] = temp;
            }
        }
        
        // Eliminate below
        for (int i = k + 1; i < n; i++) {
            double factor = aug[i * (n + 1) + k] / aug[k * (n + 1) + k];
            for (int j = k; j < n + 1; j++) {
                aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
            }
        }
    }
    
    // Back substitution
    for (int i = n - 1; i >= 0; i--) {
        x[i] = aug[i * (n + 1) + n];
        for (int j = i + 1; j < n; j++) {
            x[i] -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] /= aug[i * (n + 1) + i];
    }
    
    free(aug);
    return true;
}

// ============================================================================
// Newton-Raphson Solver
// ============================================================================

SolverResult* solve_newton_raphson(PDESystem *sys, Dictionary *initial_guess) {
    if (!sys) return NULL;
    
    SolverResult *result = malloc(sizeof(SolverResult));
    result->solution = dict_create(sys->n_unknowns);
    result->iterations = 0;
    result->final_residual = INFINITY;
    result->message = NULL;
    
    // Check system validity
    if (!pde_system_is_well_posed(sys)) {
        result->status = SOLVER_INVALID_SYSTEM;
        result->message = strdup("System is not well-posed (n_equations != n_unknowns)");
        return result;
    }
    
    int n = sys->n_unknowns;
    
    // Initialize current guess
    Dictionary *current = dict_create(n);
    
    if (initial_guess) {
        // Copy initial guess
        DictIterator iter = dict_iterator(initial_guess);
        char *key;
        Literal *val;
        while (dict_next(&iter, &key, &val)) {
            dict_set(current, key, val);
        }
    } else {
        // Default: start at origin
        for (int i = 0; i < n; i++) {
            Literal *lit = literal_create_scalar(0.0);
            if (lit) {
                dict_set(current, sys->unknowns[i], lit);
                literal_free(lit);
            }
        }
    }
    
    // Compute symbolic Jacobian: J[i][j] = ∂f_i/∂x_j
    Expression ***jacobian = malloc(sizeof(Expression**) * n);
    for (int i = 0; i < n; i++) {
        jacobian[i] = malloc(sizeof(Expression*) * n);
        for (int j = 0; j < n; j++) {
            jacobian[i][j] = derivative(sys->equations[i], sys->unknowns[j]);
        }
    }
    
    // Allocate memory and initialize variables outside the loop
    Dictionary *eval_dict = dict_create(32);
    double *J = malloc(sizeof(double) * n * n);
    double *neg_F = malloc(sizeof(double) * n);
    double *delta = malloc(sizeof(double) * n);

    // Newton-Raphson iteration
    for (int iter = 0; iter < sys->max_iterations; iter++) {
        result->iterations = iter + 1;
        
        // Compute residuals F(x)
        double *F = compute_residuals(sys, current);
        double norm = residual_norm(F, n);
        result->final_residual = norm;
        
        // Check convergence
        if (norm < sys->tolerance) {
            free(F);
            
            // Copy final solution
            DictIterator iter = dict_iterator(current);
            char *key;
            Literal *val;
            while (dict_next(&iter, &key, &val)) {
                dict_set(result->solution, key, val);
            }
            
            // Free Jacobian
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    expression_free(jacobian[i][j]);
                }
                free(jacobian[i]);
            }
            free(jacobian);
            
            dict_free(current);
            result->status = SOLVER_SUCCESS;
            result->message = strdup("Converged successfully");
            return result;
        }
        
        // Reset eval_dict for reuse
        dict_clear(eval_dict);
        DictIterator param_iter = dict_iterator(sys->parameters);
        char *key;
        Literal *val;
        while (dict_next(&param_iter, &key, &val)) {
            dict_set(eval_dict, key, val);
        }
        DictIterator guess_iter = dict_iterator(current);
        while (dict_next(&guess_iter, &key, &val)) {
            dict_set(eval_dict, key, val);
        }
        
        // Evaluate Jacobian matrix J at current point
        memset(J, 0, sizeof(double) * n * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Literal *jac_val = expression_evaluate(jacobian[i][j], eval_dict);
                if (jac_val && jac_val->shape[0] == 1) {
                    J[i * n + j] = jac_val->field[0];
                }
                if (jac_val) literal_free(jac_val);
            }
        }
        
        // Solve J * delta = -F for delta
        for (int i = 0; i < n; i++) {
            neg_F[i] = -F[i];
        }
        
        bool solved = gaussian_elimination(J, neg_F, delta, n);
        free(F);

        if (!solved) {
            // Free Jacobian
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    expression_free(jacobian[i][j]);
                }
                free(jacobian[i]);
            }
            free(jacobian);
            
            dict_free(current);
            result->status = SOLVER_INVALID_SYSTEM;
            result->message = strdup("Singular Jacobian matrix");
            printf("Singular Jacobian matrix encountered during Newton-Raphson iteration.\n");
            return result;
        }
        

        // Update: x_new = x_old + delta
        for (int i = 0; i < n; i++) {
            Literal *temp;
            if (dict_get(current, sys->unknowns[i], &temp)) {
                double old_val = temp->field[0];
                double new_val = old_val + delta[i];
                Literal *lit = literal_create_scalar(new_val);
                if (lit) {
                    dict_set(current, sys->unknowns[i], lit);
                    literal_free(lit);
                }
            }
        }
        
        // Check for divergence
        if (norm > 1e10 || isnan(norm)) {
            // Free Jacobian
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    expression_free(jacobian[i][j]);
                }
                free(jacobian[i]);
            }
            free(jacobian);
            
            dict_free(current);
            result->status = SOLVER_DIVERGED;
            result->message = strdup("Solution diverged");
            return result;
        }
    }
    
    // Max iterations reached
    DictIterator iter = dict_iterator(current);
    char *key;
    Literal *val;
    while (dict_next(&iter, &key, &val)) {
        dict_set(result->solution, key, val);
    }
    
    // Free Jacobian
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            expression_free(jacobian[i][j]);
        }
        free(jacobian[i]);
    }
    free(jacobian);
    
    dict_free(current);
    result->status = SOLVER_MAX_ITER;
    result->message = strdup("Maximum iterations reached");
    return result;
}

// ============================================================================
// SOR Solver (Stub)
// ============================================================================

SolverResult* solve_sor(PDESystem *sys, Dictionary *initial_guess, double omega) {
    // TODO: Implement SOR for large sparse systems
    (void)sys;
    (void)initial_guess;
    (void)omega;
    
    SolverResult *result = malloc(sizeof(SolverResult));
    result->solution = NULL;
    result->iterations = 0;
    result->final_residual = INFINITY;
    result->status = SOLVER_INVALID_SYSTEM;
    result->message = strdup("SOR not yet implemented");
    return result;
}

// ============================================================================
// System Analysis
// ============================================================================

bool pde_system_is_well_posed(PDESystem *sys) {
    if (!sys) return false;
    return sys->n_equations == sys->n_unknowns && sys->n_equations > 0;
}

DependencyGraph* pde_system_analyze_dependencies(PDESystem *sys) {
    if (!sys) return NULL;
    
    DependencyGraph *graph = graph_create();
    
    // Add all unknowns as definitions (each unknown depends on the equation it appears in)
    for (int i = 0; i < sys->n_equations; i++) {
        // For simplicity, just add each equation as a definition for now
        // In a full implementation, we would solve for each unknown explicitly
        if (i < sys->n_unknowns) {
            graph_add_definition(graph, sys->unknowns[i], sys->equations[i]);
        }
    }
    
    return graph;
}

char** pde_system_solve_order(PDESystem *sys, int *n_vars) {
    if (!sys || !n_vars) {
        if (n_vars) *n_vars = 0;
        return NULL;
    }
    
    DependencyGraph *graph = pde_system_analyze_dependencies(sys);
    if (!graph) {
        *n_vars = 0;
        return NULL;
    }
    
    char **order = NULL;
    bool success = graph_topological_sort(graph, &order, n_vars);
    graph_free(graph);
    
    if (!success) {
        *n_vars = 0;
        return NULL;
    }
    
    return order;
}

// ============================================================================
// Debugging and Visualization
// ============================================================================

void pde_system_print(PDESystem *sys) {
    if (!sys) {
        printf("NULL system\n");
        return;
    }
    
    printf("PDE System:\n");
    printf("  Equations: %d\n", sys->n_equations);
    printf("  Unknowns: %d\n", sys->n_unknowns);
    
    if (sys->n_unknowns > 0) {
        printf("  Variables to solve: ");
        for (int i = 0; i < sys->n_unknowns; i++) {
            printf("%s", sys->unknowns[i]);
            if (i < sys->n_unknowns - 1) printf(", ");
        }
        printf("\n");
    }
    
    printf("  Tolerance: %.2e\n", sys->tolerance);
    printf("  Max iterations: %d\n", sys->max_iterations);
    
    printf("\n  Equations (each = 0):\n");
    for (int i = 0; i < sys->n_equations; i++) {
        printf("    [%d] ", i);
        print_expression(sys->equations[i]);
        printf(" = 0\n");
    }
    
    printf("\n  Parameters:\n");
    DictIterator iter = dict_iterator(sys->parameters);
    char *key;
    Literal *val;
    bool has_params = false;
    while (dict_next(&iter, &key, &val)) {
        printf("    %s = %.6g\n", key, val->field[0]);
        has_params = true;
    }
    if (!has_params) {
        printf("    (none)\n");
    }
}

void solver_result_print(SolverResult *result) {
    if (!result) {
        printf("NULL result\n");
        return;
    }
    
    printf("Solver Result:\n");
    
    switch (result->status) {
        case SOLVER_SUCCESS:
            printf("  Status: SUCCESS\n");
            break;
        case SOLVER_MAX_ITER:
            printf("  Status: MAX_ITER\n");
            break;
        case SOLVER_DIVERGED:
            printf("  Status: DIVERGED\n");
            break;
        case SOLVER_INVALID_SYSTEM:
            printf("  Status: INVALID_SYSTEM\n");
            break;
        case SOLVER_NO_SOLUTION:
            printf("  Status: NO_SOLUTION\n");
            break;
    }
    
    printf("  Iterations: %d\n", result->iterations);
    printf("  Final residual: %.6e\n", result->final_residual);
    
    if (result->message) {
        printf("  Message: %s\n", result->message);
    }
    
    if (result->solution) {
        printf("\n  Solution:\n");
        DictIterator iter = dict_iterator(result->solution);
        char *key;
        Literal *val;
        while (dict_next(&iter, &key, &val)) {
            printf("    %s = %.10g\n", key, val->field[0]);
        }
    }
}

// ============================================================================
// Newton-Krylov Solver Implementation
// ============================================================================

// Helper: Compute vector norm
static double vector_norm(double *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

// Helper: Vector operations
static void vector_scale(double *v, double scale, int n) {
    for (int i = 0; i < n; i++) {
        v[i] *= scale;
    }
}

static void vector_add(double *result, double *a, double *b, double scale_b, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + scale_b * b[i];
    }
}

static double vector_dot(double *a, double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Helper: Extract flat vector from Dictionary
static void dict_to_vector(Dictionary *dict, char **var_names, int n_vars, double *vec) {
    for (int i = 0; i < n_vars; i++) {
        Literal *lit;
        if (dict_get(dict, var_names[i], &lit)) {
            if (lit->field && literal_total_elements(lit) > 0) {
                // For grid fields, copy all elements
                size_t n = literal_total_elements(lit);
                for (size_t j = 0; j < n; j++) {
                    vec[j] = lit->field[j];
                }
            } else {
                vec[i] = 0.0;
            }
        } else {
            vec[i] = 0.0;
        }
    }
}

// Helper: Pack flat vector into Dictionary
static void vector_to_dict(double *vec, char **var_names, int n_vars, Dictionary *dict, GridMetadata *grid) {
    if (grid) {
        // Grid field case: create Literal with grid shape
        for (int i = 0; i < n_vars; i++) {
            Literal lit;
            memset(&lit, 0, sizeof(Literal));
            for (int d = 0; d < grid->n_dims && d < N_DIM; d++) {
                lit.shape[d] = grid->dims[d];
            }
            for (int d = grid->n_dims; d < N_DIM; d++) {
                lit.shape[d] = 1;
            }
            size_t n = literal_total_elements(&lit);
            lit.field = malloc(sizeof(double) * n);
            memcpy(lit.field, vec, sizeof(double) * n);
            dict_set(dict, var_names[i], &lit);
            free(lit.field);
        }
    } else {
        // Scalar case
        for (int i = 0; i < n_vars; i++) {
            Literal *lit = literal_create_scalar(vec[i]);
            dict_set(dict, var_names[i], lit);
            literal_free(lit);
        }
    }
}

// GMRES solver for J*delta = -residual
// Implements restarted GMRES(m) with Arnoldi iteration
static bool gmres_solve(PDESystem *sys, Dictionary *x_current, double *residual,
                       double *delta, int n, int restart, int max_iter, double tol) {
    
    // Allocate GMRES workspace
    int m = (restart < n) ? restart : n;
    double **V = malloc((m + 1) * sizeof(double*));
    for (int i = 0; i <= m; i++) {
        V[i] = calloc(n, sizeof(double));
    }
    double **H = malloc((m + 1) * sizeof(double*));
    for (int i = 0; i <= m; i++) {
        H[i] = calloc(m, sizeof(double));
    }
    double *s = calloc(m + 1, sizeof(double));
    double *cs = calloc(m, sizeof(double));
    double *sn = calloc(m, sizeof(double));
    double *w = calloc(n, sizeof(double));
    
    // Initial residual: r0 = -residual (we want to solve J*delta = -F)
    for (int i = 0; i < n; i++) {
        V[0][i] = -residual[i];
    }
    double beta = vector_norm(V[0], n);
    
    if (beta < tol) {
        // Already converged
        for (int i = 0; i <= m; i++) {
            free(V[i]);
            free(H[i]);
        }
        free(V);
        free(H);
        free(s);
        free(cs);
        free(sn);
        free(w);
        return true;
    }
    
    vector_scale(V[0], 1.0 / beta, n);
    s[0] = beta;
    
    // Arnoldi iteration
    for (int j = 0; j < m && j < max_iter; j++) {
        // Compute w = J * V[j] using finite differences
        // J*v \u2248 (F(x + \u03b5*v) - F(x)) / \u03b5
        double epsilon = 1e-7;
        
        // Create perturbed dictionary: x_pert = x_current + epsilon * V[j]
        Dictionary *x_pert = dict_create(sys->n_unknowns);
        for (int i = 0; i < sys->n_unknowns; i++) {
            Literal *lit;
            dict_get(x_current, sys->unknowns[i], &lit);
            
            if (sys->grid_context && grid_literal_matches(lit, sys->grid_context)) {
                // Grid field: perturb all elements
                Literal pert = *lit;
                size_t n_elem = literal_total_elements(lit);
                pert.field = malloc(sizeof(double) * n_elem);
                for (size_t k = 0; k < n_elem; k++) {
                    pert.field[k] = lit->field[k] + epsilon * V[j][k];
                }
                dict_set(x_pert, sys->unknowns[i], &pert);
                free(pert.field);
            } else {
                // Scalar: perturb single value
                double val = lit->field ? lit->field[0] : 0.0;
                Literal *pert = literal_create_scalar(val + epsilon * V[j][i]);
                dict_set(x_pert, sys->unknowns[i], pert);
                literal_free(pert);
            }
        }
        
        // Evaluate F at perturbed point
        Dictionary *combined_pert = dict_create(sys->n_unknowns + dict_size(sys->parameters));
        DictIterator iter_params = dict_iterator(sys->parameters);
        char *key_p;
        Literal *val_p;
        while (dict_next(&iter_params, &key_p, &val_p)) {
            dict_set(combined_pert, key_p, val_p);
        }
        iter_params = dict_iterator(x_pert);
        while (dict_next(&iter_params, &key_p, &val_p)) {
            dict_set(combined_pert, key_p, val_p);
        }
        
        // Compute residual at perturbed point
        for (int eq = 0; eq < sys->n_equations; eq++) {
            Literal *eval_result;
            if (sys->grid_context) {
                eval_result = expression_evaluate_grid(sys->equations[eq], combined_pert, sys->grid_context);
            } else {
                eval_result = expression_evaluate(sys->equations[eq], combined_pert);
            }
            
            if (eval_result) {
                if (sys->grid_context) {
                    // Grid: copy all elements
                    size_t n_elem = literal_total_elements(eval_result);
                    for (size_t k = 0; k < n_elem; k++) {
                        w[k] = (eval_result->field[k] - residual[k]) / epsilon;
                    }
                } else {
                    // Scalar
                    w[eq] = (eval_result->field[0] - residual[eq]) / epsilon;
                }
                literal_free(eval_result);
            }
        }
        
        dict_free(x_pert);
        dict_free(combined_pert);
        
        // Modified Gram-Schmidt orthogonalization
        for (int i = 0; i <= j; i++) {
            H[i][j] = vector_dot(w, V[i], n);
            vector_add(w, w, V[i], -H[i][j], n);
        }
        H[j+1][j] = vector_norm(w, n);
        
        if (fabs(H[j+1][j]) < 1e-14) {
            // Lucky breakdown
            break;
        }
        
        for (int i = 0; i < n; i++) {
            V[j+1][i] = w[i] / H[j+1][j];
        }
        
        // Apply previous Givens rotations to new column of H
        for (int i = 0; i < j; i++) {
            double temp = cs[i] * H[i][j] + sn[i] * H[i+1][j];
            H[i+1][j] = -sn[i] * H[i][j] + cs[i] * H[i+1][j];
            H[i][j] = temp;
        }
        
        // Compute new Givens rotation
        double r = sqrt(H[j][j] * H[j][j] + H[j+1][j] * H[j+1][j]);
        cs[j] = H[j][j] / r;
        sn[j] = H[j+1][j] / r;
        H[j][j] = r;
        H[j+1][j] = 0.0;
        
        // Update residual norm
        s[j+1] = -sn[j] * s[j];
        s[j] = cs[j] * s[j];
        
        if (fabs(s[j+1]) < tol) {
            // Converged - solve upper triangular system
            for (int i = j; i >= 0; i--) {
                s[i] /= H[i][i];
                for (int k = i - 1; k >= 0; k--) {
                    s[k] -= H[k][i] * s[i];
                }
            }
            
            // Form solution delta = V * s
            memset(delta, 0, n * sizeof(double));
            for (int i = 0; i <= j; i++) {
                vector_add(delta, delta, V[i], s[i], n);
            }
            
            // Cleanup
            for (int i = 0; i <= m; i++) {
                free(V[i]);
                free(H[i]);
            }
            free(V);
            free(H);
            free(s);
            free(cs);
            free(sn);
            free(w);
            return true;
        }
    }
    
    // Did not converge or hit restart limit
    for (int i = 0; i <= m; i++) {
        free(V[i]);
        free(H[i]);
    }
    free(V);
    free(H);
    free(s);
    free(cs);
    free(sn);
    free(w);
    return false;
}

// Newton-Krylov solver
SolverResult* solve_newton_krylov(PDESystem *sys, Dictionary *initial_guess) {
    SolverResult *result = malloc(sizeof(SolverResult));
    result->solution = NULL;
    result->iterations = 0;
    result->final_residual = 0.0;
    result->message = NULL;
    
    if (!pde_system_is_well_posed(sys)) {
        result->status = SOLVER_INVALID_SYSTEM;
        result->message = strdup("System is not well-posed");
        return result;
    }
    
    // Determine problem size
    int n;
    if (sys->grid_context) {
        n = sys->grid_context->total_points * sys->n_unknowns;
    } else {
        n = sys->n_unknowns;
    }
    
    // Initialize current guess
    Dictionary *x_current = dict_create(sys->n_unknowns);
    if (initial_guess) {
        DictIterator iter = dict_iterator(initial_guess);
        char *key;
        Literal *val;
        while (dict_next(&iter, &key, &val)) {
            dict_set(x_current, key, val);
        }
    } else {
        // Zero initial guess
        for (int i = 0; i < sys->n_unknowns; i++) {
            if (sys->grid_context) {
                Literal lit;
                memset(&lit, 0, sizeof(Literal));
                for (int d = 0; d < sys->grid_context->n_dims && d < N_DIM; d++) {
                    lit.shape[d] = sys->grid_context->dims[d];
                }
                for (int d = sys->grid_context->n_dims; d < N_DIM; d++) {
                    lit.shape[d] = 1;
                }
                lit.field = calloc(sys->grid_context->total_points, sizeof(double));
                dict_set(x_current, sys->unknowns[i], &lit);
                free(lit.field);
            } else {
                Literal *lit = literal_create_scalar(0.0);
                dict_set(x_current, sys->unknowns[i], lit);
                literal_free(lit);
            }
        }
    }
    
    double *residual = malloc(n * sizeof(double));
    double *delta = malloc(n * sizeof(double));
    
    // Newton iteration
    for (int iter = 0; iter < sys->max_iterations; iter++) {
        // Compute residual F(x_current)
        Dictionary *combined = dict_create(sys->n_unknowns + dict_size(sys->parameters));
        DictIterator iter_params = dict_iterator(sys->parameters);
        char *key;
        Literal *val;
        while (dict_next(&iter_params, &key, &val)) {
            dict_set(combined, key, val);
        }
        iter_params = dict_iterator(x_current);
        while (dict_next(&iter_params, &key, &val)) {
            dict_set(combined, key, val);
        }
        
        for (int eq = 0; eq < sys->n_equations; eq++) {
            Literal *eval_result;
            if (sys->grid_context) {
                eval_result = expression_evaluate_grid(sys->equations[eq], combined, sys->grid_context);
            } else {
                eval_result = expression_evaluate(sys->equations[eq], combined);
            }
            
            if (!eval_result) {
                dict_free(combined);
                dict_free(x_current);
                free(residual);
                free(delta);
                result->status = SOLVER_INVALID_SYSTEM;
                result->message = strdup("Failed to evaluate equation");
                return result;
            }
            
            if (sys->grid_context) {
                // Copy grid field elements
                size_t n_elem = literal_total_elements(eval_result);
                for (size_t i = 0; i < n_elem; i++) {
                    residual[i] = eval_result->field[i];
                }
            } else {
                residual[eq] = eval_result->field[0];
            }
            literal_free(eval_result);
        }
        
        dict_free(combined);
        
        // Check convergence
        double res_norm = vector_norm(residual, n);
        result->final_residual = res_norm;
        result->iterations = iter + 1;
        
        if (res_norm < sys->tolerance) {
            result->status = SOLVER_SUCCESS;
            result->solution = x_current;
            free(residual);
            free(delta);
            return result;
        }
        
        // Solve J*delta = -F using GMRES
        bool gmres_success = gmres_solve(sys, x_current, residual, delta, n, 30, 30, sys->tolerance * 0.1);
        
        if (!gmres_success) {
            // GMRES failed - try smaller step
            vector_scale(delta, 0.5, n);
        }
        
        // Update: x_new = x_current + delta
        double *x_vec = malloc(n * sizeof(double));
        dict_to_vector(x_current, sys->unknowns, sys->n_unknowns, x_vec);
        vector_add(x_vec, x_vec, delta, 1.0, n);
        dict_free(x_current);
        x_current = dict_create(sys->n_unknowns);
        vector_to_dict(x_vec, sys->unknowns, sys->n_unknowns, x_current, sys->grid_context);
        free(x_vec);
    }
    
    // Max iterations reached
    result->status = SOLVER_MAX_ITER;
    result->solution = x_current;
    free(residual);
    free(delta);
    
    return result;
}

