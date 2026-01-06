#include "../include/dictionary.h"
#include <string.h>
#include <stdio.h>

// Simple hash function (djb2)
static uint32_t hash_string(const char *str, uint32_t capacity) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }
    return hash % capacity;
}

// ============================================================================
// Dictionary Creation and Destruction
// ============================================================================

Dictionary* dict_create(uint32_t capacity) {
    Dictionary *dict = malloc(sizeof(Dictionary));
    dict->capacity = capacity > 0 ? capacity : 16;
    dict->size = 0;
    dict->buckets = calloc(dict->capacity, sizeof(DictEntry*));
    return dict;
}

void dict_free(Dictionary *dict) {
    if (dict == NULL) return;
    
    dict_clear(dict);
    free(dict->buckets);
    free(dict);
}

// ============================================================================
// Dictionary Operations
// ============================================================================

bool dict_set(Dictionary *dict, const char *key, Literal *value) {
    if (dict == NULL || key == NULL || value == NULL) return false;
    
    uint32_t index = hash_string(key, dict->capacity);
    DictEntry *entry = dict->buckets[index];
    
    // Deep copy the literal
    Literal *value_copy = literal_copy(value);
    if (value_copy == NULL) return false;
    
    // Check if key already exists
    while (entry != NULL) {
        if (strcmp(entry->key, key) == 0) {
            // Free old literal
            literal_free(entry->value);
            // Update existing entry with deep copy
            entry->value = value_copy;
            return true;
        }
        entry = entry->next;
    }
    
    // Create new entry
    DictEntry *new_entry = malloc(sizeof(DictEntry));
    new_entry->key = malloc(strlen(key) + 1);
    strcpy(new_entry->key, key);
    new_entry->value = value_copy;
    new_entry->next = dict->buckets[index];
    dict->buckets[index] = new_entry;
    dict->size++;
    
    return true;
}

bool dict_get(Dictionary *dict, const char *key, Literal **out_value) {
    if (dict == NULL || key == NULL) return false;
    
    uint32_t index = hash_string(key, dict->capacity);
    DictEntry *entry = dict->buckets[index];
    
    while (entry != NULL) {
        if (strcmp(entry->key, key) == 0) {
            if (out_value != NULL) {
                *out_value = entry->value;
            }
            return true;
        }
        entry = entry->next;
    }
    
    return false;
}

bool dict_has(Dictionary *dict, const char *key) {
    return dict_get(dict, key, NULL);
}

bool dict_remove(Dictionary *dict, const char *key) {
    if (dict == NULL || key == NULL) return false;
    
    uint32_t index = hash_string(key, dict->capacity);
    DictEntry *entry = dict->buckets[index];
    DictEntry *prev = NULL;
    
    while (entry != NULL) {
        if (strcmp(entry->key, key) == 0) {
            // Remove entry
            if (prev == NULL) {
                dict->buckets[index] = entry->next;
            } else {
                prev->next = entry->next;
            }
            
            // Free the literal
            literal_free(entry->value);
            free(entry->key);
            free(entry);
            dict->size--;
            return true;
        }
        prev = entry;
        entry = entry->next;
    }
    
    return false;
}

// ============================================================================
// Utility Functions
// ============================================================================

void dict_clear(Dictionary *dict) {
    if (dict == NULL) return;
    
    for (uint32_t i = 0; i < dict->capacity; i++) {
        DictEntry *entry = dict->buckets[i];
        while (entry != NULL) {
            DictEntry *next = entry->next;
            // Free the literal
            literal_free(entry->value);
            free(entry->key);
            free(entry);
            entry = next;
        }
        dict->buckets[i] = NULL;
    }
    dict->size = 0;
}

uint32_t dict_size(Dictionary *dict) {
    return dict ? dict->size : 0;
}

void dict_print(Dictionary *dict) {
    if (dict == NULL) {
        printf("Dictionary: NULL\n");
        return;
    }
    
    printf("Dictionary (size=%u, capacity=%u):\n", dict->size, dict->capacity);
    for (uint32_t i = 0; i < dict->capacity; i++) {
        DictEntry *entry = dict->buckets[i];
        while (entry != NULL) {
            printf("  %s -> Literal[", entry->key);
            if (entry->value && entry->value->shape) {
                for (int d = 0; d < N_DIM; d++) {
                    printf("%u", entry->value->shape[d]);
                    if (d < N_DIM - 1) printf("x");
                }
            }
            printf("]\n");
            entry = entry->next;
        }
    }
}

// ============================================================================
// Iterator
// ============================================================================

DictIterator dict_iterator(Dictionary *dict) {
    DictIterator iter;
    iter.dict = dict;
    iter.bucket_index = 0;
    iter.current = NULL;
    
    if (dict != NULL && dict->capacity > 0) {
        // Find first non-empty bucket
        for (uint32_t i = 0; i < dict->capacity; i++) {
            if (dict->buckets[i] != NULL) {
                iter.bucket_index = i;
                iter.current = dict->buckets[i];
                break;
            }
        }
    }
    
    return iter;
}

bool dict_next(DictIterator *iter, char **out_key, Literal **out_value) {
    if (iter == NULL || iter->dict == NULL || iter->current == NULL) {
        return false;
    }
    
    // Return current entry
    if (out_key != NULL) {
        *out_key = iter->current->key;
    }
    if (out_value != NULL) {
        *out_value = iter->current->value;
    }
    
    // Move to next entry
    if (iter->current->next != NULL) {
        iter->current = iter->current->next;
    } else {
        // Find next non-empty bucket
        iter->current = NULL;
        for (uint32_t i = iter->bucket_index + 1; i < iter->dict->capacity; i++) {
            if (iter->dict->buckets[i] != NULL) {
                iter->bucket_index = i;
                iter->current = iter->dict->buckets[i];
                break;
            }
        }
    }
    
    return true;
}
