# Coupled outer/PCG tolerance study

Date: 2026-09-30  
Reference: `outer=1e-10`, `PCG=1e-12`  
Target finest-grid spatial-error scale: `dx^2 = 3.90625e-5` for `dx=0.00625`

## Recommendation

Use:

```ini
ConvgCrit 1e-5
CG        1e-10
```

`outer=1e-5` is the loosest tested outer tolerance whose accumulated momentum difference remains comfortably below the expected second-order spatial-error scale. With `PCG=1e-10`, the 50-step absolute RMS differences from the strict reference are `2.43e-6` in velocity and `1.82e-6` in pressure: respectively 6.23% and 4.65% of `dx^2`.

`PCG=1e-10` is the loosest compatible inner tolerance at this outer setting. It gives essentially the same outer iteration count as tighter PCG solves (5.08 versus 5.10 iterations/step), while reducing total CG work from 1,053–1,774 to 477 iterations/step. Tightening PCG further does not materially change the velocity solution.

`PCG=1e-9` is faster, but it is not recommended for the accuracy target: its velocity and pressure differences consume about 45% and 54% of `dx^2`, and its PCG-only contribution is visible. `outer=1e-4` fails clearly; its velocity and pressure differences are several times the spatial-error scale even with tight PCG.

## Method

- The full coupled set of 35 tolerance pairs was run for 10 timesteps.
- The decision boundary was then rerun for 50 timesteps using 15 pairs plus a repeated strict reference.
- Timing uses mean solver time per timestep. The repeated reference differed by 1.26%, which bounds ordinary timing drift in the validation run.
- Accuracy uses absolute cell-centered RMS differences from the strict 50-step reference. This is the appropriate comparison with the absolute nondimensional `dx^2` scale; relative L2 values remain available in the CSV but were not used for acceptance.
- Phi and divergence were retained as guard diagnostics only. They were not used to choose the momentum tolerances.
- Runs used the representative `40 x 160 x 40` (`dx=0.025`) local performance case. The acceptance threshold is the user's intended finest-grid `dx^2`; a short production-grid confirmation remains advisable before final paper timings because iteration counts and wall time are grid-dependent.

## 50-step validation results

| Outer | PCG | Outer it./step | CG it./outer | CG it./step | Time/step (s) | Speedup | RMS dVelocity | RMS dPressure | max difference / `dx^2` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-10 | 1e-12 | 15.08 | 136.9 | 2064 | 1.687 | 1.00x | 0 | 0 | 0% |
| 1e-7 | 1e-10 | 9.04 | 52.6 | 476 | 0.903 | 1.87x | 2.18e-8 | 3.61e-6 | 9.2% |
| 1e-7 | 1e-11 | 9.10 | 119.7 | 1090 | 1.049 | 1.61x | 1.95e-8 | 1.46e-6 | 3.7% |
| 1e-7 | 1e-12 | 9.10 | 226.5 | 2061 | 1.320 | 1.28x | 1.95e-8 | 5.36e-8 | 0.14% |
| 1e-6 | 1e-10 | 7.06 | 67.9 | 479 | 0.766 | 2.20x | 2.28e-7 | 3.75e-6 | 9.6% |
| 1e-6 | 1e-11 | 7.10 | 153.3 | 1088 | 0.923 | 1.83x | 2.10e-7 | 1.31e-6 | 3.4% |
| 1e-6 | 1e-12 | 7.10 | 285.1 | 2024 | 1.129 | 1.49x | 2.10e-7 | 3.26e-7 | 0.84% |
| 1e-5 | 1e-8 | 2.98 | 6.0 | 18 | 0.343 | 4.91x | 4.07e-5 | 3.11e-4 | 797% |
| 1e-5 | 1e-9 | 3.68 | 26.1 | 96 | 0.419 | 4.02x | 1.76e-5 | 2.12e-5 | 54% |
| **1e-5** | **1e-10** | **5.08** | **94.0** | **477** | **0.658** | **2.56x** | **2.43e-6** | **1.82e-6** | **6.2%** |
| 1e-5 | 1e-11 | 5.10 | 206.5 | 1053 | 0.748 | 2.26x | 2.27e-6 | 1.32e-6 | 5.8% |
| 1e-5 | 1e-12 | 5.10 | 347.8 | 1774 | 0.943 | 1.79x | 2.27e-6 | 1.37e-6 | 5.8% |
| 1e-4 | 1e-10 | 1.62 | 122.2 | 198 | 0.290 | 5.82x | 1.94e-4 | 4.39e-4 | 1,125% |
| 1e-4 | 1e-11 | 1.40 | 276.6 | 387 | 0.320 | 5.27x | 1.95e-4 | 1.37e-4 | 500% |
| 1e-4 | 1e-12 | 1.14 | 319.9 | 365 | 0.300 | 5.62x | 2.34e-4 | 1.62e-4 | 599% |

## Separating the PCG contribution at `outer=1e-5`

Differences below are against `outer=1e-5, PCG=1e-12`, so they isolate the effect of the inner tolerance from the effect of relaxing the outer loop.

| PCG | PCG-only RMS dVelocity | PCG-only RMS dPressure | Outer it./step |
|---:|---:|---:|---:|
| 1e-8 | 3.84e-5 | 3.12e-4 | 2.98 |
| 1e-9 | 1.53e-5 | 2.03e-5 | 3.68 |
| **1e-10** | **1.67e-7** | **2.38e-6** | **5.08** |
| 1e-11 | 4.63e-10 | 1.59e-6 | 5.10 |
| 1e-12 | 0 | 0 | 5.10 |

The sharp plateau in both outer iterations and velocity accuracy begins at `PCG=1e-10`. That is the useful inner tolerance; tighter solves mainly add pressure-system work that the outer stopping target cannot use.

## Reproduction

The opt-in harness does not alter the default input file:

```powershell
./benchmarks/run_tolerance_study.ps1 -Timesteps 10
./benchmarks/run_tolerance_study.ps1 -Timesteps 50 -ValidationSet
node ./benchmarks/analyze_tolerance_study.mjs ./out/tolerance-study/validation-50-steps
```

Raw analyzed outputs are in:

- `out/tolerance-study/sweep-10-steps/results.csv`
- `out/tolerance-study/validation-50-steps/results.csv`

