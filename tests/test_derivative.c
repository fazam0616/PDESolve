#include <stdio.h>
#include <stdlib.h>
#include "../include/expression.h"
#include "../include/literal.h"
#include "../include/calculus.h"
#include "../include/dictionary.h"

// Helper to print expression (simplified version)
void print_expr_value(Expression *expr, Dictionary *vars) {
    Literal *result = expression_evaluate(expr, vars);
    if (result) {
        printf("%.6f", result->field[0]);
        literal_free(result);
    } else {
        printf("(eval failed)");
    }
}

int main(void) {
    printf("Testing derivative computation\n\n");
    
    // Test 1: d(2*a - 2)/da should be 2
    printf("Test 1: d(2*a - 2)/da\n");
    Expression *expr1 = expr_add(
        expr_multiply(make_scalar(2.0), expr_variable("a")),
        expr_unary(OP_NEGATE, make_scalar(2.0))
    );
    
    printf("  Expression: 2*a - 2\n");
    
    Expression *deriv1 = derivative(expr1, "a");
    printf("  Derivative: ");
    
    // Evaluate at a=0
    Dictionary *vars = dict_create(1);
    Literal *lit_a = literal_create_scalar(0.0);
    dict_set(vars, "a", lit_a);
    literal_free(lit_a);
    
    print_expr_value(deriv1, vars);
    printf(" (should be 2.0)\n");
    
    expression_free(expr1);
    expression_free(deriv1);
    dict_free(vars);
    
    // Test 2: d(a²)/da should be 2*a
    printf("\nTest 2: d(a²)/da\n");
    Expression *expr2 = expr_multiply(expr_variable("a"), expr_variable("a"));
    printf("  Expression: a²\n");
    
    Expression *deriv2 = derivative(expr2, "a");
    printf("  Derivative at a=3: ");
    
    Dictionary *vars2 = dict_create(1);
    Literal *lit_a2 = literal_create_scalar(3.0);
    dict_set(vars2, "a", lit_a2);
    literal_free(lit_a2);
    
    print_expr_value(deriv2, vars2);
    printf(" (should be 6.0)\n");
    
    expression_free(expr2);
    expression_free(deriv2);
    dict_free(vars2);
    
    // Test 3: d(a)/da should be 1
    printf("\nTest 3: d(a)/da\n");
    Expression *expr3 = expr_variable("a");
    printf("  Expression: a\n");
    
    Expression *deriv3 = derivative(expr3, "a");
    printf("  Derivative: ");
    
    Dictionary *vars3 = dict_create(1);
    Literal *lit_a3 = literal_create_scalar(5.0);
    dict_set(vars3, "a", lit_a3);
    literal_free(lit_a3);
    
    print_expr_value(deriv3, vars3);
    printf(" (should be 1.0)\n");
    
    expression_free(expr3);
    expression_free(deriv3);
    dict_free(vars3);
    
    printf("\nAll tests complete\n");
    return 0;
}
