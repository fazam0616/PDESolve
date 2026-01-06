#include "../include/literal.h"
#include <stdio.h>

int main() {
    printf("Starting test...\n");
    fflush(stdout);

    printf("literal_field_size() = %llu\n", (unsigned long long)literal_field_size());
    printf("N_DIM = %d\n", N_DIM);
    printf("MAX_DIM_SIZE = %d\n", MAX_DIM_SIZE);
    fflush(stdout);

    uint32_t shape[] = {10, 20, 30};
    Literal lit;
    memset(&lit, 0, sizeof(Literal));

    printf("About to init literal...\n");
    fflush(stdout);

    literal_init(&lit, shape);

    printf("Literal initialized!\n");
    fflush(stdout);

    return 0;
}
