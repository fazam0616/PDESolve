#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include <stddef.h>

// Dynamic array structure for generic pointer storage
typedef struct {
    void **items;       // Array of pointers
    size_t size;        // Current number of items
    size_t capacity;    // Allocated capacity
} DynArray;

// Create a new dynamic array with initial capacity
DynArray* dynarray_create(size_t initial_capacity);

// Append an item to the dynamic array (grows if needed)
void dynarray_append(DynArray *arr, void *item);

// Get item at index (returns NULL if out of bounds)
void* dynarray_get(DynArray *arr, size_t index);

// Free dynamic array (optionally free items with callback)
// free_item: callback to free each item, or NULL to not free items
void dynarray_free(DynArray *arr, void (*free_item)(void*));

#endif // DATASTRUCTURES_H
