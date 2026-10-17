# 3D VOF CPU performance analysis

Date: 2026-09-29  
Branch: `3D_VOF_CG`  
Numerics: unchanged (`CG=1e-12`, convergence criteria, relaxation, VOF scheme, and physical inputs retained)

## Executive finding

The nonlinear large-grid time rise is concentrated in PCG. On the final 2.048-million-cell profile, PCG takes 71.05 s of an 83.98 s timestep (84.6%). The pressure system requires 4,604 total CG iterations versus 2,147 at 256k cells, but iteration growth is only one part of the scaling loss. PCG throughput falls from about 955 to 133 million unknown-iterations/s as the active matrix/vector data grows beyond the 32 MB L3 cache of the measured Ryzen 7 5800X.

For the handwritten diagonal SpMV specifically, throughput falls from about 1.44 billion pressure points/s at 256k cells to 245 million points/s at 2.048M cells. The 256k SpMV working set (about 12.3 MB of diagonals plus input/output vectors) fits in L3; the 2.048M set (about 98.3 MB of diagonals plus vectors) does not. This is a measured cache-to-memory-bandwidth transition, not evidence that MKL sparse SpMV should replace the handwritten kernel.

RHSu scales much closer to cell count: roughly 0.96 s at 256k and 8.12 s at 2.048M. Nested `Point` storage remains expensive and is the largest non-PCG target, but it is not the main source of the nonlinear scaling.

## Measurement setup

- Local CPU: AMD Ryzen 7 5800X, 8 cores / 16 logical CPUs, 4 MB L2, 32 MB L3.
- Compiler used locally: Intel LLVM 2024.2, Release, `/O3 /fp:precise`.
- Threads: 8 OpenMP and 8 MKL threads, unchanged.
- Representative case: 256,000 VOF cells (`40 x 160 x 40`).
- Large case: 2,048,000 VOF cells (`80 x 320 x 80`).
- Per-step output was disabled for compute profiles; final I/O was measured separately when numerical outputs were needed.
- Reported 256k stage totals are medians of three complete one-step runs.

Profiling is opt-in. Set `VANKA_PROFILE=1` to emit one `PERF_INIT` JSON line and one `PERF_STEP` JSON line per timestep. Set `VANKA_SKIP_FINAL_OUTPUT=1` only for compute benchmarks that do not require output files. `VANKA_LEGACY_BDBT_REBUILD=1` retains the MKL sparse-SYRK rebuild path for controlled comparison.

## Existing-code profile: 256k cells

The median baseline run took 2.584 s and used exactly 18 outer Vanka iterations, 18 PCG solves, and 2,147 total CG iterations in all repetitions.

| Exclusive phase | Time (s) | Share of timestep |
|---|---:|---:|
| RHSu construction | 0.971 | 37.6% |
| PCG total | 0.962 | 37.2% |
| `B.multiplyT` | 0.165 | 6.4% |
| Relaxation updates | 0.128 | 4.9% |
| RHSr construction | 0.096 | 3.7% |
| VOF total | 0.083 | 3.2% |
| BDBT reconstruction | 0.060 | 2.3% |
| `B.multiply` | 0.027 | 1.1% |
| Correction assembly/norms | 0.020 | 0.8% |
| Timestep finalization | 0.018 | 0.7% |
| First `D.multiply_with_vector` | 0.017 | 0.7% |
| Second `D.multiply_with_vector` | 0.014 | 0.5% |
| `updateRho` | 0.007 | 0.3% |
| `DiagonalMatrix::update` | 0.006 | 0.2% |

Within baseline PCG, SpMV is 65.9%, BLAS/vector operations 27.7%, and the diagonal preconditioner loop 6.1%. This validates the handwritten SpMV and the multiple pressure-vector passes as the first implementation targets.

## Incremental optimizations

| Stage | Median timestep (s) | Change vs prior | Cumulative speedup | Measured numerical result |
|---|---:|---:|---:|---|
| Existing code + instrumentation | 2.584 | — | 1.000x | Baseline |
| Branch-free SpMV interior | 2.421 | 6.7% faster | 1.067x | Five output files byte-identical |
| Fused PCG vector passes | 2.106 | 13.0% faster | 1.227x | Same 2,147 CG iterations; FP reduction-order differences only |
| Direct BDBT coefficient update | 2.130 | total hidden by run variance | 1.214x | Two-step pressure/VOF identical at printed precision |
| Explicit SIMD on fused PCG loops | 2.109 | 0.9% faster | 1.225x | Same 2,147 CG iterations; FP reduction-order differences only |

The cumulative median timestep improvement is 18.4% lower time, or 1.225x throughput. The direct BDBT change is best judged by its isolated phase: 53.6 ms to about 6.3 ms on the 256k case (8.5x faster). Whole-step noise is larger than that 47 ms saving.

The final one-step output differences relative to the existing-code baseline are:

| Field | Maximum absolute difference | Relative RMS difference |
|---|---:|---:|
| `Vx` | `1.0e-13` | `2.40e-10` |
| `Vy` | `1.0e-13` | `1.48e-10` |
| `Vz` | `1.0e-13` | `2.52e-10` |
| `P` | `1.0e-9` | `1.11e-12` |
| `Phi` | `1.0e-15` | `2.83e-18` |

## Final large-grid profile: 2.048M cells

The final code took 83.978 s for one complete compute timestep, with 18 outer iterations and 4,604 CG iterations.

| Phase | Time (s) | Share of timestep |
|---|---:|---:|
| PCG total | 71.048 | 84.6% |
| RHSu | 8.119 | 9.7% |
| `B.multiplyT` | 1.492 | 1.8% |
| Relaxation updates | 0.935 | 1.1% |
| RHSr | 0.749 | 0.9% |
| VOF total | 0.532 | 0.6% |
| Remaining measured phases | 1.103 | 1.3% |

PCG subcomponents are 38.607 s SpMV, 21.692 s fused vector/reduction operations, and 10.723 s diagonal preconditioning. Average PCG time is 15.43 ms per CG iteration; SpMV averages 8.35 ms.

The PCG time increase from 256k to 2.048M is multiplicative:

1. Matrix size: 8.0x.
2. Total CG iterations: 4,604 / 2,147 = 2.14x.
3. Normalized PCG throughput: about 955 / 133 = 7.2x lower.

Together these account for approximately the observed PCG increase. The third factor is the dominant implementation/hardware scaling problem.

## Data layout and working set

`sizeof(Point)` is 96 bytes in this build. The profiler counted:

| Storage | 256k cells | 2.048M cells |
|---|---:|---:|
| Point objects only | 130.0 MB | 1.011 GB |
| Material-field doubles | 20.2 MB | 154.5 MB |
| BDBT diagonals | 12.3 MB | 98.3 MB |
| Seven PCG workspace vectors | 14.3 MB | 114.7 MB |

These figures exclude nested-vector allocation metadata, CSR storage, RHS temporaries, VOF work arrays, and allocator overhead. Flattening the field grids would remove row-by-row allocations and pointer chasing and could reduce RHSu/relax constants, but the measured large-grid priority remains PCG memory traffic.

## Compiler findings

The Intel LLVM optimization record confirms that the new branch-free SpMV interior loop is vectorized. The originally measured Windows build used the compiler's generic ISA target and generated a two-double vector width. The release presets now use host-native architecture selection (`-march=native`) and release-only IPO/LTO; on the measured Ryzen 7 5800X, IntelLLVM resolves the target to `znver3`. The Linux release preset uses `icpx`, while Windows retains Intel's recommended `icx` driver and MSVC-compatible flags. The fused PCG loops require explicit OpenMP SIMD reductions under precise floating-point mode; no unsafe global floating-point option is enabled.

A five-run Windows smoke check with the host-native/IPO build had a 2.086 s median timestep, versus 2.12 s in the immediately preceding three-run generic-ISA sample. The difference is only about 1.6% and is within the observed host timing variability, so it is not claimed as a stable solver speedup. The compiler change does remove the generic-ISA restriction. One-timestep comparison with the prior binary gave RMS differences of `1.97e-17` in velocity, `1.37e-14` in pressure, and `1.54e-17` in Phi, with the same 18 outer and 2,147 CG iterations.

## BGU/Windows ratio

The supplied BGU points fit a simple three-point model of approximately 127 ms fixed cost plus 4.99 s per million cells. This describes the small-case shape but does not identify its cause. The supplied PC points do not fit the same linear model; their normalized cost rises at the third case, consistent with the cache transition measured locally.

The BGU-specific source of the roughly 127 ms fixed component (cgroup scheduling, virtualization, affinity, NUMA, frequency, or synchronization) cannot be determined from the Windows host. Run the new profiler on BGU and collect platform counters before attributing it:

```bash
export VANKA_PROFILE=1
export VANKA_SKIP_FINAL_OUTPUT=1
export OMP_DISPLAY_ENV=VERBOSE
export KMP_SETTINGS=1
./Vanka_Feldman_patch_solver

lscpu
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/cpuset.cpus.effective
numactl --show
perf stat -e task-clock,cycles,instructions,cache-references,cache-misses,context-switches,cpu-migrations,page-faults ./Vanka_Feldman_patch_solver
```

Compare `cg_iterations`, `pcg_avg_iteration_s`, `spmv_mpoints_per_s`, and phase shares between hosts. If iterations match but BGU SpMV throughput is low, investigate memory bandwidth/NUMA/CPU quota. If only small phases show nearly constant delays, investigate scheduling and OpenMP team-launch overhead.

## Verification

- Release executable builds successfully.
- `memory_regression` passes for its 2D, periodic 2D, and 3D cases.
- 256k baseline and each optimization used the same numerical configuration.
- Direct BDBT update was compared with the retained legacy MKL sparse-SYRK path over two timesteps.
- User configuration was restored after benchmarks; the pre-existing `Grid_step 0.025` working-tree change is preserved.
