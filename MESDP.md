# MESDP Component Documentation

This repository does not have a `MESDP/` directory. The MESDP implementation is centered on `MESDP.jl` and related files in the project root and `Arnoldi/`.

## Purpose

`MESDP.jl` implements a scalable Frank-Wolfe style solver for a Max-Cut SDP relaxation using sparse operators and a modified Arnoldi eigensolver.

Core idea:
- Keep graph operators sparse.
- Avoid building dense gradient matrices at each iteration.
- Use a Linear Minimization Oracle (LMO) from `Arnoldi/ArnoldiMethodMod.jl`.

## Files Involved

- `MESDP.jl`: main optimization logic.
- `ReadGSet.jl`: loads graph edge-list files and builds sparse matrices `(A, C)`.
- `Arnoldi/ArnoldiMethodMod.jl` and included files: partial Schur and eigen extraction (`partialschur`, `partialeigen`, `LM`).
- `testSampling.jl`: batch experiment patterns and plotting-oriented run wrapper.
- `experiment.jl`: simple `mode="A"` run script.
- `experiment_C.jl`: simple `mode="C"` run script.

## Data Structures

From `ReadGSet.jl`:
- `A` (`m x n` sparse): edge-incidence-like matrix.
- `C` (`n x n` sparse): graph Laplacian-like matrix.

Global variables expected by `MESDP.jl`:
- `m`: number of edges (`size(A, 1)`).
- `n`: number of vertices (`size(A, 2)`).

These globals are used by many functions in `MESDP.jl`.

## Function Map (`MESDP.jl`)

- `disp(quan; name="")`
  - Small display helper.

- `grad(A; P=nothing, v=nothing)`
  - Builds gradient-like matrix using either `P` or `v`.

- `B(A; P=nothing, v=nothing, d=nothing)`
  - Linear map to compute the current dual coordinates.
  - Modes:
    - `d`: diagonal/constant initialization map.
    - `P`: matrix-based map.
    - `v`: vector-based map used in the LMO update path.

- `∇g(v; lowerBound=0, upperBound=1e16, D=ones(1, n))`
  - Coordinate gradient with clamping by bounds and scaling `D`.

- `Badj(A, w)`
  - Adjoint linear map.

- `f(v)`
  - Objective proxy `abs(-sum(sqrt(v_i)))`.

- `ArnoldiGrad(A, v; lowerBound, upperBound, tol, D, mode)`
  - LMO implementation.
  - Builds `lambda` from clamped gradient weights.
  - Calls:
    - `ArnoldiMethodMod.partialschur(...)`
    - `ArnoldiMethodMod.partialeigen(...)`
  - `mode="A"`: incidence-operator route.
  - `mode="C"`: Laplacian-related route.

- `gammaLineSearch(v, q; ε=1e-8)`
  - Ternary search in `[0, 1]` for FW step size.

- `CutValue(A, z)`
  - Computes a cut-like value from rounded signs of projected samples.

- `Solve(A, v0; D, t0, ε, lowerBound, upperBound, plot, linesearch, numSample, mode, logfilename, startεd0)`
  - Main iterative solver loop.
  - Tracks primal/duality-style gap.
  - Optionally logs gap per iteration.
  - Tightens Arnoldi tolerance as gap shrinks.

## End-to-End Flow

1. Load graph with `readfile(...)` -> `(A, C)`.
2. Set globals:
   - `global m = size(A, 1)`
   - `global n = size(A, 2)`
3. Optionally scale:
   - often `A = A / 2`, `C = C / 4`.
4. Initialize:
   - `v0 = B(A, d=1/m)`.
5. Build `D` and `upperBound`:
   - typical for `mode="C"`: `D[i] = 2 * C[i, i]`.
6. Call `Solve(...)` with chosen mode.

## Running Recipes

### Mode A (incidence route)

```julia
include("ReadGSet.jl")
include("MESDP.jl")

A, C = readfile("Gset/g1.txt")
global m = size(A, 1)
global n = size(A, 2)

A = A / 2
v0 = B(A, d=1/m)

result = Solve(A, v0, ε=1e-2, linesearch=true, mode="A")
println(result.val, " ", result.t)
```

### Mode C (Laplacian route)

```julia
include("ReadGSet.jl")
include("MESDP.jl")

A, C = readfile("Gset/g1.txt")
global m = size(A, 1)
global n = size(A, 2)

A = A / 2
C = C / 4
v0 = B(A, d=1/m)

D = zeros(n)
for i in 1:n
    D[i] = 2 * C[i, i]
end
upper = sqrt(sum(D))

result = Solve(C, v0, D=D, ε=1e-2, lowerBound=0, upperBound=upper, linesearch=true, mode="C")
println(result.val, " ", result.t)
```

## Practical Notes

- `MESDP.jl` currently relies on global `m`, `n`; set them before calling solver functions.
- Keep `D` shape as a vector (`zeros(n)` / `ones(n)`) to avoid dimension-mismatch broadcasting in mode C.
- Use lowercase graph filename path as present in repo, for example `Gset/g1.txt`.
- `include("MESDP.jl")` loads functions into `Main` directly; there is no `MESDP` module namespace.

## Known Improvement Opportunities

- Refactor `MESDP.jl` into a proper `module MESDP` to avoid global-state coupling.
- Replace global `m`, `n` with explicit arguments or a problem struct.
- Standardize `D` type in defaults and call sites.
- Add tests for both mode A and mode C execution paths.
