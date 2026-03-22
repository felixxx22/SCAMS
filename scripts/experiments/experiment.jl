project_root = normpath(joinpath(@__DIR__, "..", ".."))
cd(project_root)
using Pkg
Pkg.activate(project_root)

# Load Arnoldi module first
include("Arnoldi/ArnoldiMethodMod.jl")
using .ArnoldiMethodMod

# Then load utilities
include("ReadGSet.jl")

# Then load the solver
include("MESDP.jl")

# Now run your code
A, C = readfile("Gset/g1.txt")

global m = size(A, 1)
global n = size(A, 2)

v0 = B(A, d=1/m)
result = Solve(A, v0, ε=1e-2, linesearch=true, mode="A")

println("Done! Objective: ", result.val)
println("Iterations: ", result.t)