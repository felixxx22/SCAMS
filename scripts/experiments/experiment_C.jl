project_root = normpath(joinpath(@__DIR__, "..", ".."))
cd(project_root)
using Pkg
Pkg.activate(project_root)

include("ReadGSet.jl")
include("MESDP.jl")

# Load graph and set global sizes expected by MESDP.jl
A, C = readfile("Gset/g1.txt")
global m = size(A, 1)
global n = size(A, 2)

# Keep the same scaling convention used in testSampling.jl
A = A / 2
C = C / 4

# Initial point is built from incidence matrix A
v0 = B(A, d=1 / m)

# For mode="C", use a vector D (not a 1xn matrix) to avoid broadcast shape issues
D = zeros(n)
for i in 1:n
    D[i] = 2 * C[i, i]
end

upper = sqrt(sum(D))

result = Solve(C, v0, D=D, ε=1e-2, lowerBound=0, upperBound=upper, linesearch=true, mode="C")

println("Done (C mode)! Objective: ", result.val)
println("Iterations: ", result.t)
