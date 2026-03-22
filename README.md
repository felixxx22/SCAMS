# SCAMS — SCalable Algorithm for Max-cut SDP

A Julia implementation of a scalable Frank-Wolfe (conditional gradient) algorithm for approximately solving the **Semidefinite Programming (SDP) relaxation** of the **Maximum Cut (Max-Cut)** problem on large graphs. The solver avoids forming full dense matrices by using a modified Arnoldi iteration as the Linear Minimisation Oracle (LMO), making it suitable for graphs with hundreds of thousands to millions of vertices.

## Overview

The **Max-Cut** problem seeks a partition of a graph's vertices into two sets such that the number (or total weight) of edges crossing the partition is maximised. It is NP-hard, but a well-known SDP relaxation (Goemans–Williamson) provides strong approximate bounds. Standard interior-point SDP solvers cannot scale to large instances because they require O(n²) memory.

This project reformulates the SDP relaxation as a **minimum eigenvalue SDP (MESDP)** and solves it with the **Frank-Wolfe algorithm**, where each iteration only needs the leading eigenvector of a gradient matrix. That eigenvector computation is performed with a **modified Arnoldi / Lanczos method** that works directly with sparse edge-incidence matrices, keeping memory and per-iteration cost low.

### Key Techniques

| Technique | Purpose |
|-----------|---------|
| Frank-Wolfe (Conditional Gradient) | First-order method for constrained optimisation over the spectrahedron |
| Arnoldi / Implicit Restart | Computes the leading eigenvector (LMO) without forming the full gradient matrix |
| Line search (ternary search) | Adaptive step-size selection after initial iterations |
| Adaptive Arnoldi tolerance | Tolerance tightens as the duality gap shrinks, trading speed for accuracy |
| Sparse matrix arithmetic | All graph data stored as sparse edge-incidence or Laplacian matrices |

## Folder Structure

```

## Script Layout

Runnable helper scripts are organized under `scripts/`:

- `scripts/benchmarks/benchmark.jl` — quick benchmark suite entry point.
- `scripts/benchmarks/scaling/benchmark_a_scaling.jl` — A-mode scaling benchmark.
- `scripts/benchmarks/scaling/benchmark_c_scaling.jl` — C-mode scaling benchmark.
- `scripts/benchmarks/arnoldi/benchmark_arnoldi_accuracy.jl` — Arnoldi accuracy benchmark.
- `scripts/benchmarks/lmo_tolerance/benchmark_lmo_tolerance_sweep.jl` — subset/full LMO tolerance sweep runner.
- `scripts/benchmarks/lmo_tolerance/benchmark_lmo_tolerance_sweep_full.jl` — full LMO tolerance sweep preset.
- `scripts/benchmarks/lmo_tolerance/run_lmo_tolerance_sweep_full.sh` — Debian/Linux launcher for full sweep.
- `scripts/benchmarks/lmo_tolerance/run_lmo_tolerance_sweep_full.ps1` — PowerShell launcher for full sweep.
- `scripts/benchmarks/lmo_tolerance/plot_lmo_tolerance_asymptotics.py` — gap-vs-iteration plotting for tolerance sweeps.
- `scripts/experiments/experiment.jl` and `scripts/experiments/experiment_C.jl` — small manual run scripts.
- `scripts/plots/plot_benchmarks.py` — aggregate benchmark plotting.
- `scripts/plots/plot_c_warmstart.py` — C-mode warm-start diagnostics plotting.

Core solver files remain at project root: `MESDP.jl`, `ReadGSet.jl`, `testSampling.jl`.
SCAMS/
│
├── README.md                  # This file
│
├── MESDP.jl                   # Core solver: Frank-Wolfe loop, LMO (ArnoldiGrad),
│                              #   linear map B(A,·), gradient ∇g, line search,
│                              #   cut-value rounding, and the main Solve() function
│
├── ReadGSet.jl                # Utility to read edge-list files and build the sparse
│                              #   edge-incidence matrix A and Laplacian matrix C
│
├── GenRegularGraph.jl         # Random regular-graph generator — writes edge lists
│                              #   to text files for benchmarking
│
├── testSampling.jl            # Experiment driver: reads a graph, sets parameters,
│                              #   calls Solve(), and logs convergence results
│
├── Arnoldi/                   # Modified Arnoldi eigensolver module
│   ├── ArnoldiMethodMod.jl    # Main module file (ArnoldiMethodMod): defines
│   │                          #   Arnoldi, RitzValues, PartialSchur structs;
│   │                          #   includes all sub-files below
│   ├── ArnoldiMethod.jl       # Original (unmodified) ArnoldiMethod module for
│   │                          #   reference / comparison
│   ├── ARNOLDIMOD.jl          # Empty placeholder file
│   ├── run.jl                 # partialschur() entry point and the implicit-restart
│   │                          #   loop (_partialschur); modified to accept a
│   │                          #   diagonal scaling vector λ and a mode flag
│   ├── expansion.jl           # Arnoldi basis expansion (iterate_arnoldi!),
│   │                          #   reinitialize!, orthogonalize!; contains the
│   │                          #   key modification that applies the gradient
│   │                          #   operator via sparse mat-vecs instead of a
│   │                          #   dense matrix–vector product
│   ├── schurfact.jl           # In-place QR-iteration (implicit double/single
│   │                          #   shift) to compute the Schur form of the
│   │                          #   Hessenberg matrix; Givens rotations
│   ├── schursort.jl           # Reordering of the Schur form so that wanted
│   │                          #   eigenvalues appear first
│   ├── restore_hessenberg.jl  # Restores upper-Hessenberg structure after an
│   │                          #   implicit restart
│   ├── eigvals.jl             # Extracts eigenvalues from a quasi-upper
│   │                          #   triangular matrix
│   ├── eigenvector_uppertriangular.jl
│   │                          # Backward substitution to recover eigenvectors
│   │                          #   from the Schur form
│   ├── partition.jl           # Partitioning helpers for the Schur decomposition
│   ├── targets.jl             # Eigenvalue selection targets: LM, LR, SR, LI, SI
│   │                          #   and associated ordering functors
│   ├── show.jl                # Pretty-printing for PartialSchur and History
│   ├── sampling.jl            # Earlier/standalone sampling-based solver with
│   │                          #   dense eigenvector computation (uses Arpack);
│   │                          #   includes graph construction helpers and
│   │                          #   sample-generation routines
│   └── test.jl                # Benchmarking script comparing ArnoldiGrad vs.
│                              #   the dense LMOtrueGrad on a single graph
│
├── Gset/                      # Standard Gset benchmark instances (edge lists)
│   ├── g1.txt … g54.txt       # Format: "vertex_i  vertex_j  [weight]"
│   │                          #   per line (1-indexed); sourced from the
│   │                          #   Stanford Gset collection
│   └── (54 files total)
│
└── BigExample/                # Larger randomly generated regular-graph instances
    ├── 1e4n3d.txt             # 10 000 vertices, degree ≈ 3
    ├── 1e4n5d.txt             # 10 000 vertices, degree ≈ 5
    ├── 1e4n10d.txt            # 10 000 vertices, degree ≈ 10
    ├── 100kn3d.txt            # 100 000 vertices, degree ≈ 3
    ├── 100kn10d.txt           # 100 000 vertices, degree ≈ 10
    ├── 1e5nd5.txt             # 100 000 vertices, degree ≈ 5
    ├── 300kn3d.txt            # 300 000 vertices, degree ≈ 3
    ├── 300kn5d.txt            # 300 000 vertices, degree ≈ 5
    ├── 1mn3d.txt              # 1 000 000 vertices, degree ≈ 3
    ├── 1mn4d.txt              # 1 000 000 vertices, degree ≈ 4
    └── *2.txt variants        # Second random instance with the same parameters
```

## File Details

### `MESDP.jl` — Core Solver

The main solver file. Key components:

- **`B(A; v=..., P=..., d=...)`** — Evaluates the linear map $B(X) = (\langle A_i, X A_i \rangle)_{i=1}^n$ in different modes (via eigenvector `v`, matrix `P`, or uniform diagonal `d`).
- **`∇g(v)`** — Computes the gradient of the dual objective $g(v) = -\sum_i \sqrt{v_i}$, with optional clamping.
- **`ArnoldiGrad(A, v)`** — The LMO: computes the leading eigenvector of the gradient matrix using the modified Arnoldi method, supporting two modes:
  - Mode `"A"` — operates on the edge-incidence matrix via sparse column access.
  - Mode `"C"` — operates on the Laplacian matrix via diagonal scaling.
- **`gammaLineSearch(v, q)`** — Ternary search for the optimal step size.
- **`CutValue(A, z)`** — Rounds a fractional solution to an integer cut via sign rounding.
- **`Solve(A, v0; ...)`** — Main Frank-Wolfe loop with adaptive Arnoldi tolerance, optional line search, logging, and cut rounding.

### `ReadGSet.jl` — Graph Reader

Reads an edge-list text file and returns:
- `A` — sparse edge-incidence matrix (m × n)
- `C` — sparse Laplacian matrix (n × n)

### `GenRegularGraph.jl` — Graph Generator

Generates random regular graphs by pairing half-edges uniformly at random, rejecting self-loops and multi-edges. Writes results as edge-list text files into `BigExample/`.

### `testSampling.jl` — Experiment Script

Orchestrates experiments: reads graph files, constructs the incidence/Laplacian matrices, sets solver parameters (tolerance, line search, mode), calls `Solve()`, and writes convergence logs.

## Data Format

All graph files (both `Gset/` and `BigExample/`) use a simple **edge-list** text format:

```
vertex_i  vertex_j  [weight]
```

- Vertices are **1-indexed** integers.
- Gset files include a third column for edge weight.
- BigExample files contain only the two vertex columns (unit weight).

## Dependencies

| Package | Purpose |
|---------|---------|
| `LinearAlgebra` | Standard linear algebra routines |
| `SparseArrays` | Sparse matrix storage and operations |
| `Distributions` | Random sampling (Normal, Uniform) |
| `Statistics` | Mean, standard deviation |
| `StaticArrays` | Small fixed-size arrays (Arnoldi internals) |
| `Plots` / `LaTeXStrings` | Visualisation of convergence (in test scripts) |
| `BenchmarkTools` | Performance benchmarking |
| `Arpack` | Dense eigensolver baseline (in `sampling.jl`) |

## Usage

```julia
# 1. Read a graph
include("ReadGSet.jl")
A, C = readfile("Gset/g1.txt")

# 2. Set globals expected by MESDP.jl
m = size(A, 1)  # number of edges
n = size(A, 2)  # number of vertices

# 3. Include and run the solver
include("MESDP.jl")
v0 = B(A, d=1/m)                          # uniform initialisation
result = Solve(C, v0, ε=1e-2, linesearch=true, mode="C")

# result.val  — SDP objective value
# result.v    — optimal dual variable
# result.z    — rounded cut indicator vector
```

## Benchmark CSV Structure

When `benchmark=true` is passed into `Solve(...)`, the code writes two CSV files per benchmark target:

- `<logfilename>`: run-level summary (one row per run)
- `<logfilename>.iters.csv`: per-iteration trace (one row per Frank-Wolfe iteration)

The scaling drivers (`scripts/benchmarks/scaling/benchmark_a_scaling.jl`, `scripts/benchmarks/scaling/benchmark_c_scaling.jl`) also write additional aggregate CSVs.

### 1) Solver Run Summary CSV (`<logfilename>`)

Generated in `MESDP.jl` by `Solve(...)` when benchmarking is enabled.

| Column | Meaning |
|---|---|
| `run_tag` | User-provided benchmark tag (or timestamp fallback). |
| `mode` | Solver mode: `"A"` (incidence operator) or `"C"` (Laplacian operator). |
| `linesearch` | Whether line search step-size was enabled. |
| `epsilon` | Target stopping tolerance for FW gap. |
| `start_epsilon_d0` | Initial exponent controlling Arnoldi tolerance (`10^(epsilon_d0)`). |
| `fw_iterations` | Number of Frank-Wolfe iterations completed. |
| `solve_total_time_sec` | Total time spent inside `Solve(...)` for this run. |
| `avg_fw_iter_time_sec` | Average FW iteration wall time. |
| `lmo_calls` | Number of LMO (ArnoldiGrad) calls. |
| `total_lmo_time_sec` | Total time spent in LMO calls. |
| `avg_lmo_time_sec` | Average time per LMO call. |
| `avg_partialschur_time_sec` | Average time in `partialschur` per LMO call. |
| `avg_partialeigen_time_sec` | Average time in `partialeigen` per LMO call. |
| `total_mvproducts` | Total Arnoldi matrix-vector products across run. |
| `lmo_time_share` | Fraction of solve time spent in LMO (`total_lmo_time_sec / solve_total_time_sec`). |
| `final_gap` | Final normalized Frank-Wolfe gap at termination. |
| `converged` | Boolean flag, true if `final_gap <= epsilon`. |
| `median_delta_v_l2` | Median FW update size $||v_t - v_{t-1}||_2$ over iterations. |
| `p90_delta_v_l2` | 90th percentile FW update size $||v_t - v_{t-1}||_2$. |
| `corr_delta_v_to_next_lmo_time` | Correlation between `delta_v_l2` at iteration $t$ and next-LMO time at $t+1$. |
| `corr_delta_v_to_next_mvproducts` | Correlation between `delta_v_l2` at iteration $t$ and next-LMO mvproducts at $t+1$. |
| `corr_delta_v_to_delta_lambda` | Correlation between `delta_v_l2` at iteration $t$ and eigenvalue drift $|\lambda_{t+1}-\lambda_t|$. |

### 2) Solver Iteration Trace CSV (`<logfilename>.iters.csv`)

Generated in `MESDP.jl` by `Solve(...)` when benchmarking is enabled.

| Column | Meaning |
|---|---|
| `run_tag` | Run identifier matching summary CSV. |
| `iteration` | FW iteration index (starting at 1). |
| `t` | Internal FW schedule counter after update. |
| `mode` | `"A"` or `"C"`. |
| `linesearch` | Whether linesearch mode was enabled. |
| `gamma_source` | Step-size source: schedule or linesearch. |
| `gamma` | Step size used in this iteration. |
| `gap` | Normalized Frank-Wolfe gap after iteration update. |
| `epsilon` | Target stopping tolerance. |
| `epsilon_d0` | Current Arnoldi tolerance exponent for this iteration. |
| `iter_time_sec` | Total wall time for this FW iteration. |
| `lmo_time_sec` | Time spent in LMO during this iteration. |
| `partialschur_time_sec` | Time spent in Arnoldi `partialschur`. |
| `partialeigen_time_sec` | Time spent in `partialeigen`. |
| `lmo_mvproducts` | Arnoldi matrix-vector product count for this iteration. |
| `lmo_converged` | Whether Arnoldi converged for this LMO call. |
| `lmo_nconverged` | Number of converged Ritz values. |
| `lmo_nev` | Target number of eigenvalues requested. |
| `delta_v_l2` | FW update size $||v_t - v_{t-1}||_2$. |
| `delta_v_linf` | FW update size $||v_t - v_{t-1}||_\infty$. |
| `delta_v_rel_l2` | Relative update size $||v_t - v_{t-1}||_2 / ||v_{t-1}||_2$. |
| `lambda_t` | Leading eigenvalue from the previous LMO call (iteration $t$ baseline). |
| `lambda_next` | Leading eigenvalue from the current LMO call (next state). |
| `delta_lambda_next` | Eigenvalue movement $|\lambda_{t+1} - \lambda_t|$. |
| `cosine_similarity_next` | Sign-invariant cosine similarity between consecutive leading eigenvectors. |
| `delta_w_next` | Sign-invariant eigenvector movement `min(||w_{t+1}-w_t||, ||w_{t+1}+w_t||)`. |
| `next_lmo_time_sec` | Alias of next-call LMO time for direct plotting. |
| `next_partialschur_time_sec` | Alias of next-call `partialschur` time for direct plotting. |
| `next_partialeigen_time_sec` | Alias of next-call `partialeigen` time for direct plotting. |
| `next_lmo_mvproducts` | Alias of next-call Arnoldi matrix-vector products. |
| `next_lmo_converged` | Alias of next-call convergence flag. |

### 3) Scaling Driver Raw Runs CSV (`Result/benchmarks_a_scaling/runs.csv`, `Result/benchmarks_c_scaling/runs.csv`)

Generated by `scripts/benchmarks/scaling/benchmark_a_scaling.jl` and `scripts/benchmarks/scaling/benchmark_c_scaling.jl`.

| Column | Meaning |
|---|---|
| `graph_name` | Input graph filename. |
| `graph_path` | Path to the input graph file. |
| `repeat` | Repeat index for the same graph/setting. |
| `n` | Number of vertices. |
| `m` | Number of edges. |
| `avg_degree` | Approximate average degree (`2m/n`). |
| `linesearch` | Whether line search was enabled. |
| `epsilon` | Solve tolerance used. |
| `start_epsilon_d0` | Initial Arnoldi tolerance exponent. |
| `total_wall_time_sec` | End-to-end runtime including file I/O + solve. |
| `io_time_sec` | Graph loading and setup time. |
| `solve_time_sec` | Time reported by `Solve(...)`. |
| `fw_iterations` | FW iterations for this run. |
| `lmo_calls` | Number of LMO calls. |
| `total_lmo_time_sec` | Total LMO time. |
| `avg_lmo_time_sec` | Mean LMO call time. |
| `avg_partialschur_time_sec` | Mean `partialschur` time. |
| `avg_partialeigen_time_sec` | Mean `partialeigen` time. |
| `total_mvproducts` | Total Arnoldi matrix-vector products. |
| `lmo_share_of_solve` | LMO fraction of solve time. |
| `io_share_of_total` | I/O fraction of total wall time. |
| `non_lmo_solver_share_of_total` | Non-LMO solver fraction of total wall time. |
| `lmo_share_of_total` | LMO fraction of total wall time. |
| `final_gap` | Final FW gap. |
| `converged` | Whether run reached target tolerance. |
| `median_delta_v_l2` | Median FW update size for this run. |
| `p90_delta_v_l2` | 90th percentile FW update size for this run. |
| `corr_delta_v_to_next_lmo_time` | Run-level correlation of update size vs next-LMO time. |
| `corr_delta_v_to_next_mvproducts` | Run-level correlation of update size vs next-LMO mvproducts. |
| `corr_delta_v_to_delta_lambda` | Run-level correlation of update size vs eigenvalue drift. |

## C-Mode Warm-Start Plot Script

Use `scripts/plots/plot_c_warmstart.py` to generate warm-start diagnostics directly from the C-mode benchmark CSV files.

Expected inputs:
- Iteration files: `Result/benchmarks_c_scaling/per_graph/*_summary.csv.iters.csv`
- Run summary file: `Result/benchmarks_c_scaling/runs.csv`

Example:

```bash
python scripts/plots/plot_c_warmstart.py
```

Outputs (under `Result/plots_c_warmstart` by default):
- `delta_v_vs_next_lmo_time_scatter.png`
- `delta_v_vs_next_lmo_mvproducts_scatter.png`
- `delta_v_vs_delta_lambda_scatter.png`
- `delta_v_vs_delta_w_scatter.png`
- `delta_v_vs_next_lmo_time_binned_median.png`
- `delta_v_vs_next_lmo_mvproducts_binned_median.png`
- `delta_v_vs_delta_lambda_binned_median.png`
- `corr_vs_size_c_mode.png` (if `runs.csv` exists)
- `warmstart_correlation_summary.csv`

### 4) Scaling Driver By-Size Aggregate CSV (`Result/benchmarks_a_scaling/by_size.csv`, `Result/benchmarks_c_scaling/by_size.csv`)

Grouped averages over repeated runs at the same size profile.

| Column | Meaning |
|---|---|
| `n` | Mean vertex count for group. |
| `m` | Mean edge count for group. |
| `avg_degree` | Mean average degree for group. |
| `runs` | Number of repeats in group. |
| `mean_total_wall_time_sec` | Mean end-to-end wall time. |
| `std_total_wall_time_sec` | Standard deviation of wall time over repeats. |
| `mean_io_time_sec` | Mean I/O time. |
| `mean_solve_time_sec` | Mean solve time. |
| `mean_fw_iterations` | Mean FW iteration count. |
| `mean_total_lmo_time_sec` | Mean total LMO time. |
| `mean_lmo_share_of_total` | Mean LMO fraction of total wall time. |

### 5) Plot Helper Aggregate CSV (`Result/plots/plot_summary_by_size.csv`)

Generated by `scripts/plots/plot_benchmarks.py` for plotting and cross-mode comparison.

| Column | Meaning |
|---|---|
| `label` | Data source label (for example `A` or `C`). |
| `m` | Number of edges group key (current plotting default). |
| `avg_degree` | Average degree group key. |
| `runs` | Number of source rows in group. |
| `mean_total_wall_time_sec` | Mean total wall time. |
| `mean_io_time_sec` | Mean I/O time. |
| `mean_solve_time_sec` | Mean solve time. |
| `mean_total_lmo_time_sec` | Mean total LMO time. |
| `mean_lmo_share_of_total` | Mean LMO fraction of total. |
| `mean_fw_iterations` | Mean FW iterations. |

## References

- Goemans, M. X., & Williamson, D. P. (1995). *Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming.* JACM 42(6).
- The Arnoldi module is a modified version of [ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl), adapted to apply the gradient operator implicitly via sparse matrix–vector products.
