param(
    [string]$JuliaCommand = "julia"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..\..\..")
Push-Location $projectRoot

try {
    if (-not (Get-Command $JuliaCommand -ErrorAction SilentlyContinue)) {
        throw "Julia executable '$JuliaCommand' was not found in PATH. Pass -JuliaCommand with a full path if needed."
    }

    Write-Host "Starting full LMO tolerance sweep from $projectRoot"
    & $JuliaCommand (Join-Path $scriptDir "benchmark_lmo_tolerance_sweep_full.jl")

    if ($LASTEXITCODE -ne 0) {
        throw "Full sweep failed with exit code $LASTEXITCODE"
    }

    Write-Host "Full LMO tolerance sweep finished."
    Write-Host "Results: $projectRoot\Result\benchmarks_lmo_tolerance\full"
}
finally {
    Pop-Location
}
