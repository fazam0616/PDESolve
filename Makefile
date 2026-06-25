CC     = gcc
CFLAGS = -Wall -Wextra -Iinclude -O3 -march=native -ffast-math -fopenmp -DGLEW_STATIC

# Platform detection
UNAME := $(shell uname -s 2>/dev/null)
ifneq (,$(findstring MINGW,$(UNAME)))
	SDL_CFLAGS  := $(shell sdl2-config --cflags)
	SDL_LDFLAGS := $(shell sdl2-config --libs) -lopengl32 -lglew32 -lSDL2_ttf -lcomdlg32 -mconsole
else
    SDL_CFLAGS  := $(shell sdl2-config --cflags)
    SDL_LDFLAGS := $(shell sdl2-config --libs) -lGL -lGLEW -lSDL2_ttf
endif

LDFLAGS = -fopenmp $(SDL_LDFLAGS) -lm

SRC_DIR      = src
TEST_DIR     = tests
EXAMPLES_DIR = examples
BUILD_DIR    = build

SRC_FILES     = $(wildcard $(SRC_DIR)/*.c)
TEST_FILES    = $(wildcard $(TEST_DIR)/*.c)
EXAMPLE_FILES = $(wildcard $(EXAMPLES_DIR)/*.c)

OBJ_FILES           = $(patsubst $(SRC_DIR)/%.c,    $(BUILD_DIR)/%.o, $(SRC_FILES))
TEST_EXECUTABLES    = $(patsubst $(TEST_DIR)/%.c,    $(BUILD_DIR)/%,   $(TEST_FILES))
EXAMPLE_EXECUTABLES = $(patsubst $(EXAMPLES_DIR)/%.c,$(BUILD_DIR)/%,  $(EXAMPLE_FILES))

.PRECIOUS: $(BUILD_DIR)/%.o
.PHONY: all clean test run_interactive run_interactive_gpu run_smoke_gpu run_steel

all: $(TEST_EXECUTABLES) $(EXAMPLE_EXECUTABLES)

# Link test executables
$(BUILD_DIR)/test_%: $(OBJ_FILES) $(BUILD_DIR)/test_%.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Link example executables
$(BUILD_DIR)/%: $(OBJ_FILES) $(BUILD_DIR)/%.o | $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Special dependency for the embedded menu
$(BUILD_DIR)/interactive_wave_sim.o: $(EXAMPLES_DIR)/interactive_wave_sim_menu.inc

# Compile source files (need SDL headers because Menu.c uses SDL)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -c $< -o $@

# Compile test files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -c $< -o $@

# Compile example files
$(BUILD_DIR)/%.o: $(EXAMPLES_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

test: $(TEST_EXECUTABLES)
	@echo "Running all tests..."; \
	total=0; passed=0; failed=0; \
	for exec in $(TEST_EXECUTABLES); do \
		total=$$((total + 1)); \
		if $$exec; then echo "[PASS] $$exec"; passed=$$((passed + 1)); \
		else echo "[FAIL] $$exec"; failed=$$((failed + 1)); fi; \
	done; \
	echo "Summary: $$passed/$$total passed"; \
	[ $$failed -eq 0 ]

run_interactive: $(BUILD_DIR)/interactive_wave_sim
	$(BUILD_DIR)/interactive_wave_sim

run_interactive_gpu: $(BUILD_DIR)/interactive_wave_sim_gpu
	$(BUILD_DIR)/interactive_wave_sim_gpu

run_smoke_gpu: $(BUILD_DIR)/interactive_smoke_sim_gpu
	$(BUILD_DIR)/interactive_smoke_sim_gpu

run_steel: $(BUILD_DIR)/interactive_steel_sim
	$(BUILD_DIR)/interactive_steel_sim
