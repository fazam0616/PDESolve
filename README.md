PDESolve
=======

Overview
--------
PDESolve is a compact C-based engine for expressing and solving partial differential equations (PDEs). The project provides a small core engine (math and grid data structures, symbolic expressions) and a set of example programs and tests that demonstrate use of the core engine, including interactive visualizations and GPU-accelerated examples.

Key components
--------------
- `src/` : Core implementation files (grid handling, calculus primitives, expression evaluation, solvers, GPU support hooks).
- `include/` : Public headers used by examples and tests.
- `tests/` : Unit/integration test programs exercising engine components.
- `examples/` : Small standalone programs built on the engine (interactive sims and GPU variants).
- `build/` : Default output directory for compiled objects and executables.
- `Makefile` : Build rules for compiling core objects, tests, and examples.

Examples
--------
The repository contains several example programs under [examples](examples):

- `interactive_wave_sim.c` — an SDL2-based interactive wave simulation viewer that uses the engine to compute updates and renders them in real time.
- `interactive_wave_sim_gpu.c` — GPU-accelerated variant of the wave simulator that uses the project's GPU helper code to run compute on supported hardware.
- `interactive_smoke_sim_gpu.c` — a GPU-accelerated smoke advection/diffusion example demonstrating more complex advection/diffusion usage.
- `interactive_wave_sim_menu.inc` — helper include used by the interactive wave example to provide UI/menu options at compile time.

Build and run
-------------
Dependencies (Debian/Ubuntu example):

```bash
sudo apt-get update
sudo apt-get install build-essential libsdl2-dev libglew-dev libgl1-mesa-dev libsdl2-ttf-dev
```

Basic build steps (from repository root):

```bash
make            # builds core objects, tests, and example executables into build/
```

Run an example (from repo root):

```bash
./build/interactive_wave_sim
./build/interactive_wave_sim_gpu
./build/interactive_smoke_sim_gpu
```

Run the test suite:

```bash
make test
```

Clean build artifacts:

```bash
make clean
```

Notes
-----
- The Makefile compiles core objects from `src/` into `build/` and then links example/test executables there. SDL2 and OpenGL-related flags are used for example builds.
- GPU examples require a supported GPU and drivers. The repository contains GPU helper code (e.g. `gpu_compiler.c`, `boundary_gpu.c`) but exact runtime requirements may vary by platform.
- Compiler flags in the Makefile include `-O3 -march=native -ffast-math -fopenmp -flto`; adjust `CFLAGS` in the `Makefile` if you need different optimization/debug settings.