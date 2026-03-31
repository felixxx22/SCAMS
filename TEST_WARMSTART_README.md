# Scaled Eigenvector Warm-Start Testing

## Quick Start

Run the benchmarking script:
```bash
julia test_scaled_warmstart.jl
```

## What It Tests

The script compares **3 warm-start strategies** on 5 small GSET graphs (g1-g5):

| Configuration | Description |
|---|---|
| **baseline** | No warm-start (control baseline) |
| **eigvec_only** | Reuses previous eigenvector as-is (existing method) |
| **scaled** | Transforms eigenvector via gradient scaling (NEW) |

## Output

All results written to `Result/warmstart_comparison/`:

```
Result/warmstart_comparison/
├── g1_YYYYMMDD_HHMMSS_baseline_summary.csv              # Summary stats
├── g1_YYYYMMDD_HHMMSS_baseline_summary.csv.iters.csv    # Per-iteration details
├── g1_YYYYMMDD_HHMMSS_eigvec_only_summary.csv
├── g1_YYYYMMDD_HHMMSS_eigvec_only_summary.csv.iters.csv
├── g1_YYYYMMDD_HHMMSS_scaled_summary.csv
├── g1_YYYYMMDD_HHMMSS_scaled_summary.csv.iters.csv
├── ... (repeated for g2, g3, g4, g5)
```

## CSV Columns

### Summary CSV (`*_summary.csv`)
Key columns for comparison:
- `fw_iterations`: Number of Frank-Wolfe iterations
- `lmo_calls`: Total LMO (Arnoldi) calls
- `solve_total_time_sec`: Total wall-clock time
- `avg_lmo_time_sec`: Average time per LMO call
- `final_gap`: Final Frank-Wolfe gap at convergence
- `converged`: Whether algorithm converged to tolerance
- `median_warmstart_D_change`: Relative change in gradient scaling (||E-D||∞ / ||D||∞)
- `median_warmstart_x0_norm`: Typical norm of warm-start initial vector
- `median_warmstart_alpha`: Typical normalization factor

### Iteration CSV (`*_summary.csv.iters.csv`)
Per-FW-iteration details:
- `iteration`: Frank-Wolfe iteration number
- `gap`: Current Frank-Wolfe gap
- `lmo_init_source`: How Arnoldi was initialized ("random", "previous_eigvec", "scaled_eigvec")
- `lmo_time_sec`: Time for this LMO call
- `D_change_relative_linf`: Gradient scaling change this iteration
- `warmstart_x0_norm`: Norm of initial vector this iteration
- `warmstart_alpha`: Scaling factor this iteration

## Analysis

After the script finishes, you'll see a comparison table like:

```
================================================================================
WARMSTART COMPARISON: g1
================================================================================
   Configuration |  Iterations |   LMO Calls |    Time (s) | Avg LMO (s) |        Gap | Converged
----------------+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        baseline |         127 |         127 |      8.1234 |    0.063892 | 9.99e-03 |        YES
     eigvec_only |          89 |          89 |      5.4321 |    0.061046 | 9.85e-03 |        YES
         scaled  |          76 |          76 |      4.8765 |    0.064155 | 9.91e-03 |        YES

SPEEDUP vs BASELINE (config 1):
  eigvec_only: 1.49x time | 30.0% fewer iters | median(D_change)=1.23e-02 | median(x0_norm)=0.8934
  scaled: 1.67x time | 40.2% fewer iters | median(D_change)=3.45e-02 | median(x0_norm)=1.0012
```

## Key Metrics to Watch

1. **Iterations reduction**: % fewer FW iterations → convergence acceleration
2. **Time speedup**: Wall-clock time ratio (>1 is faster)
3. **LMO time**: Whether warm-start improves Arnoldi convergence speed
4. **x0_norm**: Should be ~1.0 for well-scaled initialization
5. **D_change**: Magnitude of gradient scaling variation across iterations

## Interpretation

- **Good scaled warm-start**: Fewer iterations, similar/faster time, x0_norm ≈ 1
- **Bad scaled warm-start**: More iterations, slower time, or x0_norm >> 1 (indicates scaling issues)
- **Fallback behavior**: If scaled warm-start fails, it gracefully reverts to eigvec_only

## Customization

Edit `test_scaled_warmstart.jl` to modify:
- `SAMPLE_GRAPHS`: Which GSET graphs to test (e.g., `["g1", "g5", "g10"]`)
- `FW_TOLERANCE`: Frank-Wolfe stopping criterion (default: 1e-2)
- `ARNOLDI_TOL`: Arnoldi accuracy (default: -3.0 → 10^-3)
- `NUM_SAMPLES`: Gaussian samples for cut extraction (default: 3)

## Troubleshooting

**Error: "Graph file not found"**
- Ensure `Gset/g1.txt` etc. exist
- From workspace root, run `julia test_scaled_warmstart.jl`

**Error: "lowerBound / D[i]" in ArnoldiGrad**
- Check if `D` parameter is passed correctly in Solve
- Current implementation uses fixed `D=ones((1,n))` throughout

**CSV not written**
- Check that `Result/warmstart_comparison/` directory is writable
- Verify no permission issues on file system
