#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_lmo_tolerance_sweep_full.sh
#   ./run_lmo_tolerance_sweep_full.sh /path/to/julia

JULIA_CMD="${1:-julia}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v "$JULIA_CMD" >/dev/null 2>&1; then
    echo "Error: Julia executable '$JULIA_CMD' was not found in PATH." >&2
    echo "Pass an explicit Julia path as the first argument." >&2
    exit 1
fi

echo "Starting full LMO tolerance sweep from $PROJECT_ROOT"
"$JULIA_CMD" --project="$PROJECT_ROOT" -e 'using Pkg; Pkg.instantiate()'
"$JULIA_CMD" --project="$PROJECT_ROOT" "$SCRIPT_DIR/benchmark_lmo_tolerance_sweep_full.jl"

echo "Full LMO tolerance sweep finished."
echo "Results: $PROJECT_ROOT/Result/benchmarks_lmo_tolerance/full"
