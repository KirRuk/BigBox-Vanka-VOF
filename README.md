# Big-Box Vanka–VOF Solver

A finite-volume solver for incompressible, immiscible two-phase flows based on global Big-Box Vanka pressure–velocity coupling and algebraic Volume-of-Fluid (VOF) interface transport.

The pressure–velocity correction is reduced through a Schur-complement formulation and solved using a conjugate-gradient (CG) method. Interface transport uses cubic upwind interpolation (CUI) with explicit second-order Runge–Kutta time integration. The implementation supports variable density and viscosity, surface tension, and gravity, and uses OpenMP for shared-memory parallelism.

## Supported discretization

**The present version supports only uniform, regular Cartesian grids** with a staggered (MAC) finite-volume discretization in two or three dimensions. Nonuniform, adaptive, and unstructured grids are not supported by this release.

## Requirements

Install and configure the following dependencies before building:

- **Intel oneAPI**, including the Intel C/C++ compilers (`icx` and `icpx`), the OpenMP runtime, and **Intel oneAPI Math Kernel Library (MKL)**.
- **CMake 3.21 or newer**.
- **Ninja** build system.
- **TecIO** (optional), for Tecplot SZL output. The application can be built without TecIO by disabling that option during CMake configuration.

The solver is written in **C++20**. Build from a terminal in which the Intel oneAPI environment has been initialized so that the compiler, MKL, and OpenMP are available to CMake.

## Compilation

From the repository root, select the supplied release preset for your platform.

**Linux:**

```bash
cmake --preset linux-release
cd out/build/linux-release
ninja
```

**Windows (Intel oneAPI command prompt):**

```powershell
cmake --preset x64-release
cd out/build/x64-release
ninja
```

The Ninja build is run in `out/build/<preset>/`, where CMake generates `build.ninja`. The compiled executable is placed in the `Vanka_Feldman_patch_solver` subdirectory of that build directory.

### Building without TecIO

TecIO output is enabled by default in the supplied CMake configuration. If TecIO is not installed, disable it when configuring:

```bash
cmake --preset linux-release -DENABLE_TECIO_OUTPUT=OFF
```

On Windows, replace `linux-release` with `x64-release`. Then enter the corresponding `out/build/<preset>/` directory and run `ninja` as above.

To build with TecIO, make its headers and libraries discoverable to CMake; if needed, provide the installation location using `-DTECIO_ROOT=/path/to/TecIO`.

### Running

Configure the input files in `resources/` before starting a simulation. For example, on Linux, from the repository root:

```bash
./out/build/linux-release/Vanka_Feldman_patch_solver/Vanka_Feldman_patch_solver
```

On Windows, run the corresponding `.exe` in `out/build/x64-release/Vanka_Feldman_patch_solver/`.

## Simulation input (`resources/`)

The solver reads its case configuration from the `resources/` directory.

### 1. Domain, resolution, and physical parameters

Edit `resources/user.choice` to specify the computational domain, grid spacing, time stepping, material-property ratios, solver tolerances, and parallel settings. For example, the principal entries are:

```text
X_direction 1.0
Y_direction 4.0
Z_direction 1.0
Grid_step 0.025
Two_dimensional false
Threads 8
time_step 0.0005
Termination_Time 0.5
output_frequency 1
Reynold's_number 341
RhoRelation 3.0
MuRelation 1.0
Fr 1.0
We 1e8
CG 1e-12
ConvgCrit 5e-10
Max_Relax_Number 200
read_data false
```

These are **selected entries**, not a complete replacement for `user.choice`. Retain and configure the other entries in the supplied file, which the current input parser also expects.

`X_direction`, `Y_direction`, and `Z_direction` define the domain dimensions; `Grid_step` is the uniform Cartesian cell size. The expected numbers of control volumes are determined from the domain lengths divided by `Grid_step`. Set `Two_dimensional` appropriately for a 2D or 3D case.

### 2. Initial phase-indicator field

Provide the initial VOF field as a **Tecplot-style ASCII, point-ordered file** in `resources/`:

- **2D:** `PhiInit.dat` (columns `x y phi`).
- **3D:** `PhiInit3D.dat` (columns `x y z phi`).

The first line is a header; the second line must contain the grid dimensions as `I=` and `J=` (and `K=` for 3D). The remaining records contain cell-centre coordinates and the initial phase fraction `phi`, with the x-index varying fastest, followed by y and z. The phase indicator takes values between 0 and 1, with a sharp interface represented by the contrast between the two phases.

The file dimensions must match `X_direction`, `Y_direction`, `Z_direction`, and `Grid_step` in `user.choice`. **The appropriate initial VOF file must be supplied before running.**

### 3. Fluid boundary conditions

Specify the velocity and pressure boundary conditions in:

- `boundariesU.inp` — x-velocity component.
- `boundariesV.inp` — y-velocity component.
- `boundariesW.inp` — z-velocity component.
- `boundariesP.inp` — pressure boundary conditions and reference-pressure specification.

Each file has a comment/header line followed by six boundary entries in the form:

```text
FACE  CONDITION_TYPE  VALUE
```

The face identifiers are `E`, `W`, `N`, `S`, `F`, and `B` (east, west, north, south, front, and back). For example:

```text
E fixed_value 0.0
W fixed_value 0.0
N fixed_value 0.0
S fixed_value 0.0
F fixed_value 0.0
B fixed_value 0.0
```

Supported input condition keywords are:

| Keyword | Meaning |
| --- | --- |
| `fixed_value` | Prescribed value (Dirichlet condition). |
| `slip` | Slip-type velocity boundary. |
| `convective_boundary` | Convective/outflow-type condition. |
| `symetry` | Legacy spelling used for the implemented east–west periodic boundary treatment. |
| `dirichlet_point` | Pointwise pressure reference constraint. |
| `dirichlet_row` | Row of pressure reference constraints. |

For **east–west periodic boundaries**, set both `E` and `W` to `symetry` for `U` and `V`, and also for `W` in 3D. The implementation checks that these periodic faces are paired consistently. The pressure file's `dirichlet_point` and `dirichlet_row` options use special entries to encode reference locations; consult the comments in `resources/boundariesP.inp` rather than treating all six values as independent physical face pressures.

### 4. Optional restart fields

If `read_data` is set to `true`, the solver also reads initial velocity and pressure fields from `Vx.dat`, `Vy.dat`, `Vz.dat`, and `P.dat` in `resources/`. For a new simulation initialized by the solver, use `read_data false`.

## License

See the repository's `LICENSE` file for the applicable license and copyright notice.
