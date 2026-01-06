#include <stdio.h>
#include <stdlib.h>
#include "../include/expression.h"
#include "../include/literal.h"
#include "../include/calculus.h"
#include "../include/dictionary.h"

// Recursive expression printer
void print_expr_tree(Expression *expr, int depth) {
    if (!expr) {
        printf("%*s(null)\n", depth*2, "");
        return;
    }
    
    switch (expr->type) {
        case EXPR_LITERAL:
            printf("%*sLITERAL(%.2f)\n", depth*2, "", expr->data.literal->field[0]);
            break;
        case EXPR_VARIABLE:
            printf("%*sVARIABLE(\"%s\")\n", depth*2, "", expr->data.variable);
            break;
        case EXPR_UNARY:
            printf("%*sUNARY(%d)\n", depth*2, "", expr->data.unary.op);
            print_expr_tree(expr->data.unary.operand, depth + 1);
            break;
        case EXPR_BINARY:
            printf("%*sBINARY(%d)\n", depth*2, "", expr->data.binary.op);
            print_expr_tree(expr->data.binary.left, depth + 1);
            print_expr_tree(expr->data.binary.right, depth + 1);
            break;
    }
}

int main(void) {
    printf("Testing derivative: d(2*a)/da\n\n");
    
    // Create expression: 2*a
    Expression *two = make_scalar(2.0);
    Expression *a = expr_variable("a");
    Expression *expr = expr_multiply(two, a);
    
    printf("Original expression:\n");
    print_expr_tree(expr, 0);
    
    printf("\nTaking derivative with respect to 'a'...\n");
    Expression *deriv = derivative(expr, "a");
    
    printf("\nDerivative expression:\n");
    print_expr_tree(deriv, 0);
    
    printf("\nEvaluating at a=0:\n");
    Dictionary *vars = dict_create(1);
    Literal *lit_a = literal_create_scalar(0.0);
    dict_set(vars, "a", lit_a);
    literal_free(lit_a);
    
    Literal *result = expression_evaluate(deriv, vars);
    if (result) {
        printf("Result: %.6f (should be 2.0)\n", result->field[0]);
        literal_free(result);
    } else {
        printf("Evaluation failed!\n");
    }
    
    // Try simplifying
    printf("\nSimplifying derivative...\n");
    Expression *simplified = simplify(deriv);
    printf("Simplified expression:\n");
    print_expr_tree(simplified, 0);
    
    printf("\nEvaluating simplified at a=0:\n");
    Literal *result2 = expression_evaluate(simplified, vars);
    if (result2) {
        printf("Result: %.6f (should be 2.0)\n", result2->field[0]);
        literal_free(result2);
    } else {
        printf("Evaluation failed!\n");
    }
    
    dict_free(vars);
    expression_free(expr);
    expression_free(deriv);
    expression_free(simplified);
    
    return 0;
}
