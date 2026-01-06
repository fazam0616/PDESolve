#include <stdio.h>
#include "../include/expression.h"
#include "../include/literal.h"
#include "../include/dictionary.h"

int main(void) {
    printf("Testing basic expression evaluation\n\n");
    
    // Test 1: Evaluate a literal
    printf("Test 1: Evaluate literal 2.0\n");
    Literal *lit2 = literal_create_scalar(2.0);
    if (!lit2) {
        printf("ERROR: Could not create literal!\n");
        return 1;
    }
    printf("  Created literal: %.2f\n", lit2->field[0]);
    
    Expression *expr2 = expr_literal(lit2);
    if (!expr2) {
        printf("ERROR: Could not create expression!\n");
        return 1;
    }
    printf("  Created expression\n");
    
    Dictionary *vars = dict_create(1);
    printf("  Created dictionary\n");
    
    printf("  Calling expression_evaluate...\n");
    fflush(stdout);
    Literal *result = expression_evaluate(expr2, vars);
    
    if (result) {
        printf("  Result: %.2f\n", result->field[0]);
        literal_free(result);
    } else {
        printf("  ERROR: expression_evaluate returned NULL!\n");
    }
    
    dict_free(vars);
    expression_free(expr2);
    
    // Test 2: Multiply 2 * a
    printf("\nTest 2: Evaluate 2*a at a=3\n");
    Literal *lit_two = literal_create_scalar(2.0);
    Expression *expr_two = expr_literal(lit_two);
    Expression *expr_a = expr_variable("a");
    Expression *expr_mult = expr_multiply(expr_two, expr_a);
    
    printf("  Created expression 2*a\n");
    
    Dictionary *vars2 = dict_create(1);
    Literal *lit_a = literal_create_scalar(3.0);
    dict_set(vars2, "a", lit_a);
    literal_free(lit_a);
    
    printf("  Calling expression_evaluate...\n");
    fflush(stdout);
    Literal *result2 = expression_evaluate(expr_mult, vars2);
    
    if (result2) {
        printf("  Result: %.2f (should be 6.0)\n", result2->field[0]);
        literal_free(result2);
    } else {
        printf("  ERROR: expression_evaluate returned NULL!\n");
    }
    
    dict_free(vars2);
    expression_free(expr_mult);
    
    // Test 3: 2*a + (-2) at a=1 (should be 0)
    printf("\nTest 3: Evaluate 2*a + (-2) at a=1\n");
    Literal *lit_2a = literal_create_scalar(2.0);
    Expression *expr_2 = expr_literal(lit_2a);
    Expression *expr_a2 = expr_variable("a");
    Expression *term1 = expr_multiply(expr_2, expr_a2);
    
    Literal *lit_neg2 = literal_create_scalar(2.0);
    Expression *expr_pos2 = expr_literal(lit_neg2);
    Expression *term2 = expr_unary(OP_NEGATE, expr_pos2);
    
    Expression *full_expr = expr_add(term1, term2);
    
    printf("  Created expression 2*a + (-2)\n");
    
    Dictionary *vars3 = dict_create(1);
    Literal *lit_a3 = literal_create_scalar(1.0);
    dict_set(vars3, "a", lit_a3);
    literal_free(lit_a3);
    
    printf("  Calling expression_evaluate...\n");
    fflush(stdout);
    Literal *result3 = expression_evaluate(full_expr, vars3);
    
    if (result3) {
        printf("  Result: %.2f (should be 0.0)\n", result3->field[0]);
        literal_free(result3);
    } else {
        printf("  ERROR: expression_evaluate returned NULL!\n");
    }
    
    dict_free(vars3);
    expression_free(full_expr);
    
    printf("\nAll tests complete\n");
    return 0;
}
