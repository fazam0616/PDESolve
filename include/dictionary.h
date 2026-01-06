#ifndef DICTIONARY_H
#define DICTIONARY_H

#include "literal.h"
#include <stdbool.h>

// Dictionary entry
typedef struct DictEntry {
    char *key;              // Variable name
    Literal *value;         // Associated literal value (owned)
    struct DictEntry *next; // For hash collision chaining
} DictEntry;

// Dictionary structure (hash table)
typedef struct {
    DictEntry **buckets;
    uint32_t capacity;
    uint32_t size;
} Dictionary;

// Dictionary creation and destruction
// capacity: Initial hash table size (must be > 0)
// Returns: Newly allocated Dictionary (caller owns, must call dict_free)
// Memory: Allocates Dictionary and bucket array, caller must free
Dictionary* dict_create(uint32_t capacity);

// Free dictionary and all owned resources
// dict: Dictionary to free (can be NULL)
// Memory: Frees all entries (keys, Literal field arrays), buckets, and dict struct
// Note: Deep-frees Literal field arrays that were stored via dict_set
void dict_free(Dictionary *dict);

// Dictionary operations

// Set key-value pair (creates new entry or updates existing)
// dict: Dictionary to modify (must not be NULL)
// key: Key string (copied internally, caller retains ownership)
// value: Literal pointer (deep-copied including field array)
// Returns: true on success, false on allocation failure
// Memory: Creates deep copy of Literal including field array
// Note: Previous value's Literal is freed if key exists
bool dict_set(Dictionary *dict, const char *key, Literal *value);

// Get value for key
// dict: Dictionary to query (can be NULL, returns false)
// key: Key to look up (must not be NULL)
// out_value: Output parameter for Literal pointer (can be NULL to just check existence)
// Returns: true if key found, false otherwise
// Memory: Returns pointer to dictionary's internal Literal, do not free
// Warning: Returned pointer is valid until dict_set/dict_remove/dict_free for this key
bool dict_get(Dictionary *dict, const char *key, Literal **out_value);

// Check if key exists in dictionary
// Returns: true if key exists, false otherwise
bool dict_has(Dictionary *dict, const char *key);

// Remove key-value pair from dictionary
// dict: Dictionary to modify (can be NULL, returns false)
// key: Key to remove (must not be NULL)
// Returns: true if key was removed, false if not found
// Memory: Frees entry's key, field array, and entry struct
bool dict_remove(Dictionary *dict, const char *key);

// Utility functions

// Remove all entries from dictionary (but keep dictionary structure)
// dict: Dictionary to clear (can be NULL)
// Memory: Frees all entries and their field arrays, resets size to 0
void dict_clear(Dictionary *dict);

// Get number of entries in dictionary
// Returns: Number of key-value pairs, or 0 if dict is NULL
uint32_t dict_size(Dictionary *dict);

// Print dictionary contents to stdout (for debugging)
// dict: Dictionary to print (can be NULL)
void dict_print(Dictionary *dict);

// Iterator for traversing all entries

typedef struct {
    Dictionary *dict;
    uint32_t bucket_index;
    DictEntry *current;
} DictIterator;

// Create iterator for dictionary traversal
// dict: Dictionary to iterate over (must not be NULL)
// Returns: Iterator positioned before first entry
// Usage: while (dict_next(&iter, &key, &val)) { use key and val }
DictIterator dict_iterator(Dictionary *dict);

// Advance iterator to next entry
// iter: Iterator to advance (must not be NULL)
// out_key: Output for key string (can be NULL)
// out_value: Output for Literal pointer (can be NULL)
// Returns: true if entry retrieved, false if no more entries
// Warning: out_value points to dictionary's internal storage, do not free
bool dict_next(DictIterator *iter, char **out_key, Literal **out_value);

#endif // DICTIONARY_H
