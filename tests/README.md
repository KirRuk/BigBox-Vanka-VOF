# Memory regression checks

These tests compile the solver without its application entry point, MUMPS or
TecIO. They do not modify `resources/` or run the user's large simulation.

Configure in an Intel oneAPI environment.

On Linux, use Intel's C++ driver `icpx`:

```sh
cmake -S tests -B out/build/memory-regression -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/memory-regression --parallel 8
ctest --test-dir out/build/memory-regression --output-on-failure
```

On Windows, Intel recommends the `icx` driver; use the same commands with
`-DCMAKE_CXX_COMPILER=icx` from an initialized oneAPI command prompt.

Checks cover all 27 point boundary codes, six condition names read from a legacy
text input file, coordinate range rejection, the default ID sentinel and linear
IDs above 65535. Small fixed/slip 2D, periodic 2D and 3D cases exercise material
fields, pressure assembly, PCG corrections and VOF, using one and four threads.
The periodic cases also check paired fluid faces and zero net boundary flux.
Separate 2D and 3D VOF cases check flux cancellation and phase volume over five
steps in both flow directions, with smooth and sharp interfaces.

For a numerical before/after comparison, run `memory_regression 1` and
`memory_regression 4` from the build directory and save stdout before and after
the change. Compare the `RESULT` lines (labels and values); the `Point bytes`
line is expected to change. Outputs are checked for finite values, not for full
physical convergence. Generated input fixtures are removed after reading.

Point boundary locations and physical boundary conditions now use separate
one-byte enums in `BoundaryCodes.h`; file names/labels are unchanged. Coordinates
are unsigned 16-bit integers: maximum index 65535 including ghost cells. Point
constructors reject overflow, and grid initialization checks the bounds before
entering OpenMP regions. Linear point IDs remain signed 32-bit and MKL indices
retain the configured MKL ABI; all physical values remain double precision.
