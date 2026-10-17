param(
    [int]$Timesteps = 10,
    [string]$OutputRoot = "",
    [string]$Executable = "",
    [switch]$ValidationSet
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $studyName = if ($ValidationSet) { "validation" } else { "sweep" }
    $OutputRoot = Join-Path $repoRoot "out\tolerance-study\${studyName}-${Timesteps}-steps"
}
if ([string]::IsNullOrWhiteSpace($Executable)) {
    $Executable = Join-Path $repoRoot "out\build\x64-release\Vanka_Feldman_patch_solver\Vanka_Feldman_patch_solver.exe"
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$Executable = [IO.Path]::GetFullPath($Executable)

if (-not (Test-Path -LiteralPath $Executable)) {
    throw "Solver executable not found: $Executable"
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$pairs = [Collections.Generic.List[object]]::new()
if ($ValidationSet) {
    foreach ($specification in @(
        @(-10, -12),
        @(-7, -10),
        @(-7, -11),
        @(-7, -12),
        @(-6, -10),
        @(-6, -11),
        @(-6, -12),
        @(-5, -8),
        @(-5, -9),
        @(-5, -10),
        @(-5, -11),
        @(-5, -12),
        @(-4, -10),
        @(-4, -11),
        @(-4, -12)
    )) {
        $outerExponent = $specification[0]
        $cgExponent = $specification[1]
        $pairs.Add([pscustomobject]@{
            outer = "1e$outerExponent"
            cg = "1e$cgExponent"
            label = "outer_1e$outerExponent`__cg_1e$cgExponent"
            repeat = $false
        })
    }
}
else {
    foreach ($outerExponent in -10..-4) {
        for ($cgExponent = $outerExponent - 1; $cgExponent -ge -12; --$cgExponent) {
            $pairs.Add([pscustomobject]@{
                outer = "1e$outerExponent"
                cg = "1e$cgExponent"
                label = "outer_1e$outerExponent`__cg_1e$cgExponent"
                repeat = $false
            })
        }
    }
}
$pairs.Add([pscustomobject]@{
    outer = "1e-10"
    cg = "1e-12"
    label = "reference_repeat"
    repeat = $true
})

$completed = 0
foreach ($pair in $pairs) {
    ++$completed
    $runDirectory = Join-Path $OutputRoot $pair.label
    $summaryPath = Join-Path $runDirectory "summary.json"
    $snapshotPath = Join-Path $runDirectory "tolerance_snapshot.bin"
    if ((Test-Path -LiteralPath $summaryPath) -and
        (Test-Path -LiteralPath $snapshotPath)) {
        Write-Output "[$completed/$($pairs.Count)] Reusing $($pair.label)"
        continue
    }

    New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
    Write-Output "[$completed/$($pairs.Count)] Running outer=$($pair.outer), CG=$($pair.cg)"

    $env:VANKA_PROFILE = "1"
    $env:VANKA_SKIP_FINAL_OUTPUT = "1"
    $env:VANKA_TOLERANCE_STUDY = "1"
    $env:VANKA_MAX_TIMESTEPS = $Timesteps.ToString([Globalization.CultureInfo]::InvariantCulture)
    $env:VANKA_OUTER_TOL = $pair.outer
    $env:VANKA_CG_TOL = $pair.cg

    Push-Location $runDirectory
    try {
        $wallClock = [Diagnostics.Stopwatch]::StartNew()
        $outputLines = @(& $Executable 2>&1 | ForEach-Object { $_.ToString() })
        $exitCode = $LASTEXITCODE
        $wallClock.Stop()
    }
    finally {
        Pop-Location
    }
    [IO.File]::WriteAllLines((Join-Path $runDirectory "solver.log"), $outputLines)
    if ($exitCode -ne 0) {
        throw "Solver failed for $($pair.label) with exit code $exitCode"
    }
    if (-not (Test-Path -LiteralPath $snapshotPath)) {
        throw "Snapshot missing for $($pair.label)"
    }

    $steps = @(
        $outputLines |
            Where-Object { $_.StartsWith("PERF_STEP ") } |
            ForEach-Object { $_.Substring(10) | ConvertFrom-Json }
    )
    if ($steps.Count -ne $Timesteps) {
        throw "Expected $Timesteps PERF_STEP records for $($pair.label), found $($steps.Count)"
    }

    $totalOuter = ($steps | Measure-Object -Property outer_iterations -Sum).Sum
    $totalCg = ($steps | Measure-Object -Property cg_iterations -Sum).Sum
    $last = $steps[-1]
    $summary = [ordered]@{
        outer_tol = [double]$pair.outer
        cg_tol = [double]$pair.cg
        label = $pair.label
        repeat = [bool]$pair.repeat
        timesteps = $Timesteps
        wall_s = $wallClock.Elapsed.TotalSeconds
        mean_time_s = ($steps | Measure-Object -Property compute_s -Average).Average
        median_time_s = ($steps.compute_s | Sort-Object)[[int][Math]::Floor($steps.Count / 2)]
        mean_outer_iterations = $totalOuter / [double]$steps.Count
        cg_iterations_per_outer = $totalCg / [double]$totalOuter
        total_outer_iterations = $totalOuter
        total_cg_iterations = $totalCg
        final_outer_residual_u = $last.outer_residual_u
        final_outer_residual_v = $last.outer_residual_v
        final_outer_residual_w = $last.outer_residual_w
        final_outer_residual_p = $last.outer_residual_p
        final_time_residual_u = $last.time_residual_u
        final_time_residual_v = $last.time_residual_v
        final_time_residual_w = $last.time_residual_w
        final_time_residual_p = $last.time_residual_p
        final_divergence_rms = $last.divergence_rms
        final_divergence_max = $last.divergence_max
        final_vof_mass_error = $last.vof_mass_error
    }
    [IO.File]::WriteAllText(
        $summaryPath,
        ($summary | ConvertTo-Json -Depth 4))
}

$allSummaries = @(
    Get-ChildItem -LiteralPath $OutputRoot -Directory |
        ForEach-Object {
            $path = Join-Path $_.FullName "summary.json"
            if (Test-Path -LiteralPath $path) {
                Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
            }
        }
)
[IO.File]::WriteAllText(
    (Join-Path $OutputRoot "sweep_summaries.json"),
    ($allSummaries | ConvertTo-Json -Depth 4))
Write-Output "Completed $($allSummaries.Count) runs in $OutputRoot"
