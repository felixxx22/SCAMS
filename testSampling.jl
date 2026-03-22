include("MESDP.jl")
include("ReadGSet.jl")
using LaTeXStrings
using Plots
using Plots.PlotMeasures
using BenchmarkTools
using Printf
using Random

#Single graph with individual bound on the gradient
function exp1(inputFile, outputfile; linesearch=false, ε=1e-2, v0=nothing, t0=0, bound=true, mode="A", startεd0=0.0, benchmark=false, benchmarkTag=nothing)
    file = inputFile
    print("Readfile ")
    io_start_ns = time_ns()
    @time A, C = readfile(file)
    io_time_sec = (time_ns() - io_start_ns) / 1e9
    #disp(size(A))
    A = A / 2
    C = C / 4
    global m = size(A, 1)
    global n = size(A, 2)
    A_s = sparse(A)

    D = spzeros(n)
    sumai = 0
    for i in 1:n
        D[i] = 2 * C[i, i]
        sumai = sumai + D[i]
    end
    disp(sumai)
    sumai = sqrt(sumai)
    v = B(A, d=1 / m)

    t = 5
    if v0 !== nothing
        v = v0
        t = t0
    end

    if bound
        upper = sumai
    else
        D = ones(n)
        upper = 1e16
    end

    if mode == "A"
        result1 = Solve(A_s, v, t0=t, D=D, lowerBound=0, upperBound=upper, plot=true, linesearch=linesearch, ε=ε, numSample=1, mode=mode, logfilename=outputfile, startεd0=startεd0, benchmark=benchmark, benchmarkTag=benchmarkTag)
    elseif mode == "C"
        result1 = Solve(C, v, t0=t, D=D, lowerBound=0, upperBound=upper, plot=true, linesearch=linesearch, ε=ε, numSample=1, mode=mode, logfilename=outputfile, startεd0=startεd0, benchmark=benchmark, benchmarkTag=benchmarkTag)
    end

    if benchmark
        solve_sec = result1.bench.solve_total_sec
        total_sec = io_time_sec + solve_sec
        lmo_pct = 100 * result1.bench.lmo_share
        println(@sprintf("[bench] tag=%s mode=%s linesearch=%s total=%.3fs io=%.3fs solve=%.3fs fw_iters=%d lmo_calls=%d avg_lmo=%.4fs lmo_share=%.1f%%",
            result1.bench.run_tag,
            mode,
            string(linesearch),
            total_sec,
            io_time_sec,
            solve_sec,
            result1.bench.fw_iterations,
            result1.bench.lmo_calls,
            result1.bench.avg_lmo_sec,
            lmo_pct,
        ))
    end

    return (v=result1.v, t=result1.t, z=result1.z, bench=result1.bench, io_time_sec=io_time_sec)
end

function run_full_benchmark_suite(; repeats=3, warmup=true, seed=42)
    Random.seed!(seed)
    scenarios = [
        (name="small_A_no_ls", input="BigExample/1e4n3d.txt", mode="A", linesearch=false, ε=1e-2, startεd0=0.0),
        (name="small_C_no_ls", input="BigExample/1e4n3d.txt", mode="C", linesearch=false, ε=1e-2, startεd0=0.0),
        (name="small_A_ls", input="BigExample/1e4n3d.txt", mode="A", linesearch=true, ε=1e-2, startεd0=0.0),
        (name="medium_A_no_ls", input="BigExample/1e4n10d.txt", mode="A", linesearch=false, ε=1e-2, startεd0=0.0),
        (name="medium_C_no_ls", input="BigExample/1e4n10d.txt", mode="C", linesearch=false, ε=1e-2, startεd0=0.0),
    ]

    for scenario in scenarios
        println("\n=== Scenario: ", scenario.name, " ===")
        logfile = "Result/benchmarks/" * scenario.name * ".csv"

        if warmup
            println("[bench] warmup run")
            exp1(
                scenario.input,
                logfile,
                linesearch=scenario.linesearch,
                ε=scenario.ε,
                bound=true,
                mode=scenario.mode,
                startεd0=scenario.startεd0,
                benchmark=true,
                benchmarkTag=scenario.name * "_warmup",
            )
        end

        for run_id in 1:repeats
            tag = scenario.name * "_run" * string(run_id)
            exp1(
                scenario.input,
                logfile,
                linesearch=scenario.linesearch,
                ε=scenario.ε,
                bound=true,
                mode=scenario.mode,
                startεd0=scenario.startεd0,
                benchmark=true,
                benchmarkTag=tag,
            )
        end
    end

    println("\nBenchmark suite finished. CSV logs are in Result/benchmarks/")
end

function benchmark_lmo_only(inputFile; mode="A", ε=1e-2, lowerBound=0.0, upperBound=1e16, seed=42)
    Random.seed!(seed)
    A, C = readfile(inputFile)
    A = A / 2
    C = C / 4
    global m = size(A, 1)
    global n = size(A, 2)
    A_s = sparse(A)

    D = spzeros(n)
    for i in 1:n
        D[i] = 2 * C[i, i]
    end

    v = B(A, d=1 / m)
    workA = mode == "A" ? A_s : C

    # Warmup to reduce JIT impact on benchmark numbers.
    ArnoldiGrad(workA, v, lowerBound=lowerBound, upperBound=upperBound, D=D, mode=mode, tol=ε, returnMetrics=true)

    stats = @benchmark ArnoldiGrad($workA, $v, lowerBound=$lowerBound, upperBound=$upperBound, D=$D, mode=$mode, tol=$ε, returnMetrics=true)
    println("[lmo] mode=", mode, " median_time_s=", median(stats).time / 1e9, " mean_time_s=", mean(stats).time / 1e9, " median_alloc_bytes=", median(stats).memory)
    return stats
end

#=
opt = 12083.2
ε1 = 10^(-1.5)
inputfile = "C:/Users/pchib/Desktop/MASTER/MESDP/Gset/g1.txt"
#disp(@benchmark exp1(inputfile, nothing, opt, nothing, ε=ε1, linesearch=true, bound=true, color=:black, mode="A"))
u = @benchmark exp1(inputfile, nothing, opt, nothing, ε=ε1, linesearch=true, bound=true, color=:black, mode="C")
println(mean(u).memory)
println(mean(u).time)
=#

if abspath(PROGRAM_FILE) == @__FILE__
    ε1 = 10^(-2)

    file = ["1e4n3d.txt", "1e4n3d2.txt", "1e4n5d.txt", "1e4n5d2.txt", "1e4n10d.txt", "1e4n10d2.txt"]

    for i = 1:length(file)
        inputfile = "BigExample/" * file[i]
        outputfile = "Result/exp5/" * chop(file[i], tail=4) * "log.txt"
        println(inputfile)
        println(outputfile)
        print("Solve ")
        @time exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=0.0, benchmark=true)
    end
end

#=========================================================================================

outputfile = "MATLABplot/G48log-1.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=-1.0)


outputfile = "MATLABplot/G48log-2.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=-2.0)
=#
#=
outputfile = "MATLABplot/G49log-3.txt"
exp1(inputfile, outputfile, ε=ε1, linesearch=true, bound=true, mode="C", startεd0=-3.0)
=#

