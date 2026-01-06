#include "../include/datastructures.h"
#include <stdlib.h>
#include <string.h>

DynArray* dynarray_create(size_t initial_capacity) {
    DynArray *arr = (DynArray*)malloc(sizeof(DynArray));
    if (!arr) return NULL;
    
    arr->capacity = initial_capacity > 0 ? initial_capacity : 4;
    arr->size = 0;
    arr->items = (void**)malloc(sizeof(void*) * arr->capacity);
    
    if (!arr->items) {
        free(arr);
        return NULL;
    }
    
    return arr;
}

void dynarray_append(DynArray *arr, void *item) {
    if (!arr) return;
    
    // Grow if needed
    if (arr->size >= arr->capacity) {
        size_t new_capacity = arr->capacity * 2;
        void **new_items = (void**)realloc(arr->items, sizeof(void*) * new_capacity);
        if (!new_items) return; // Failed to grow, skip append
        arr->items = new_items;
        arr->capacity = new_capacity;
    }
    
    arr->items[arr->size++] = item;
}

void* dynarray_get(DynArray *arr, size_t index) {
    if (!arr || index >= arr->size) return NULL;
    return arr->items[index];
}

void dynarray_free(DynArray *arr, void (*free_item)(void*)) {
    if (!arr) return;
    
    if (free_item) {
        for (size_t i = 0; i < arr->size; i++) {
            if (arr->items[i]) {
                free_item(arr->items[i]);
            }
        }
    }
    
    free(arr->items);
    free(arr);
}
