# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -O3 -march=native -ffast-math -fopenmp -flto
LDFLAGS = -fopenmp -flto
SDL_CFLAGS = $(shell sdl2-config --cflags)
SDL_LDFLAGS = $(shell sdl2-config --libs) -lGL -lSDL2_ttf

# Directories
SRC_DIR = src
TEST_DIR = tests
EXAMPLES_DIR = examples
BUILD_DIR = build

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
TEST_FILES = $(wildcard $(TEST_DIR)/*.c)
EXAMPLE_FILES = $(wildcard $(EXAMPLES_DIR)/*.c)

# Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRC_FILES))
TEST_OBJ_FILES = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_FILES))
EXAMPLE_OBJ_FILES = $(patsubst $(EXAMPLES_DIR)/%.c,$(BUILD_DIR)/%.o,$(EXAMPLE_FILES))

# Executables
TEST_EXECUTABLES = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%,$(TEST_FILES))
EXAMPLE_EXECUTABLES = $(patsubst $(EXAMPLES_DIR)/%.c,$(BUILD_DIR)/%,$(EXAMPLE_FILES))

# Default target
all: $(TEST_EXECUTABLES) $(EXAMPLE_EXECUTABLES)

# Build each test executable
$(BUILD_DIR)/test_%: $(OBJ_FILES) $(BUILD_DIR)/test_%.o
	$(CC) $(CFLAGS) $^ -o $@ -lm $(LDFLAGS)

# Build example executables (with SDL2 support)
$(BUILD_DIR)/interactive_wave_sim: $(OBJ_FILES) $(BUILD_DIR)/interactive_wave_sim.o
	$(CC) $(CFLAGS) $(SDL_CFLAGS) $^ -o $@ -lm $(LDFLAGS) $(SDL_LDFLAGS)

# Special dependency: interactive_wave_sim.o depends on its .inc file
$(BUILD_DIR)/interactive_wave_sim.o: $(EXAMPLES_DIR)/interactive_wave_sim_menu.inc

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile example files (with SDL2 flags)
$(BUILD_DIR)/%.o: $(EXAMPLES_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Run all tests and summarize results
.PHONY: test
test: $(TEST_EXECUTABLES)
	@echo "Running all tests..."
	@total=0; passed=0; failed=0; \
	for exec in $(TEST_EXECUTABLES); do \
		total=$$((total + 1)); \
		echo "Running $$exec..."; \
		if $$exec; then \
			echo "[PASS] $$exec"; \
			passed=$$((passed + 1)); \
		else \
			echo "[FAIL] $$exec"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo "\nSummary:"; \
	echo "Total Tests: $$total"; \
	echo "Passed: $$passed"; \
	echo "Failed: $$failed"; \
	if [ $$failed -gt 0 ]; then \
		exit 1; \
	else \
		exit 0; \
	fi

# Clean build files
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

# Run interactive wave simulator
.PHONY: run_interactive
run_interactive: $(BUILD_DIR)/interactive_wave_sim
	$(BUILD_DIR)/interactive_wave_sim