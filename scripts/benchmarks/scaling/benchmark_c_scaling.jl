include("../../../MESDP.jl")
include("../../../ReadGSet.jl")

using Printf
using Random
using Statistics

function _ensure_parent_dir(path::AbstractString)
    parent = dirname(path)
    if !isempty(parent)
        mkpath(parent)
    end
end

function _write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector})
    _ensure_parent_dir(path)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(string.(row), ","))
        end
    end
end

function _scan_graph_stats(path::AbstractString)
    m_local = 0
    n_local = 0
    open(path, "r") do io
        for ln in eachline(io)
            m_local += 1
            parts = split(ln)
            i = parse(Int, parts[1])
            j = parse(Int, parts[2])
            n_local = max(n_local, i, j)
        end
    end
    return (n=n_local, m=m_local, avg_degree=(2.0 * m_local) / n_local)
end

function _list_bigexample_files(; include_variants=true)
    files = filter(f -> endswith(f, ".txt"), readdir("BigExample"))
    if !include_variants
        files = filter(f -> !endswith(f, "2.txt"), files)
    end

    paths = [joinpath("BigExample", f) for f in files]
    stats = [merge((path=p, name=basename(p)), _scan_graph_stats(p)) for p in paths]
    sort!(stats, by=s -> (s.n, s.avg_degree, s.name))
    return stats
end

function _run_one_c_mode(input_file::AbstractString, log_file::AbstractString;
    epsilon=1e-2,
    start_epsilon_d0=0.0,
    tighten_arnoldi_tol=true,
    linesearch=false,
    benchmark_tag=nothing,
    arnoldi_mindim=12,
    arnoldi_maxdim=20,
    arnoldi_restarts=500,
    adaptive_arnoldi_budget=false,
    update_low_threshold=1e-3,
    update_high_threshold=1e-2,
    low_budget_scale=0.7,
    high_budget_scale=1.6,
)
    io_start_ns = time_ns()
    A, C = readfile(input_file)
    io_time_sec = (time_ns() - io_start_ns) / 1e9

    A = A / 2
    C = C / 4

    global m = size(A, 1)
    global n = size(A, 2)

    D = spzeros(n)
    sumai = 0.0
    for i in 1:n
        D[i] = 2 * C[i, i]
        sumai += D[i]
    end
    upper = sqrt(sumai)

    v0 = B(A, d=1 / m)
    solve_result = Solve(
        C,
        v0,
        t0=5,
        D=D,
        lowerBound=0,
        upperBound=upper,
        plot=false,
        linesearch=linesearch,
        ε=epsilon,
        numSample=1,
        mode="C",
        logfilename=log_file,
        startεd0=start_epsilon_d0,
        tighten_arnoldi_tol=tighten_arnoldi_tol,
        benchmark=true,
        benchmarkTag=benchmark_tag,
        arnoldi_mindim=arnoldi_mindim,
        arnoldi_maxdim=arnoldi_maxdim,
        arnoldi_restarts=arnoldi_restarts,
        adaptive_arnoldi_budget=adaptive_arnoldi_budget,
        update_low_threshold=update_low_threshold,
        update_high_threshold=update_high_threshold,
        low_budget_scale=low_budget_scale,
        high_budget_scale=high_budget_scale,
    )

    bench = solve_result.bench
    total_wall_sec = io_time_sec + bench.solve_total_sec
    return (
        n=n,
        m=m,
        avg_degree=(2.0 * m) / n,
        io_time_sec=io_time_sec,
        total_wall_sec=total_wall_sec,
        bench=bench,
    )
end

function run_c_mode_scaling_benchmark(; repeats=3, warmup=true, seed=42, epsilon=1e-2, start_epsilon_d0=0.0, tighten_arnoldi_tol=true, linesearch=false, include_variants=true, arnoldi_mindim=12, arnoldi_maxdim=20, arnoldi_restarts=500, adaptive_arnoldi_budget=false, update_low_threshold=1e-3, update_high_threshold=1e-2, low_budget_scale=0.7, high_budget_scale=1.6)
    Random.seed!(seed)

    outdir = "Result/benchmarks_c_scaling"
    per_graph_dir = joinpath(outdir, "per_graph")
    mkpath(per_graph_dir)

    graph_specs = _list_bigexample_files(include_variants=include_variants)
    if isempty(graph_specs)
        error("No BigExample .txt files found.")
    end

    run_rows = Vector{Vector}()

    for g in graph_specs
        summary_csv = joinpath(per_graph_dir, string(chop(g.name, tail=4), "_summary.csv"))
        iter_csv = string(summary_csv, ".iters.csv")

        # Start fresh for this graph so CSV headers are regenerated cleanly.
        if isfile(summary_csv)
            rm(summary_csv)
        end
        if isfile(iter_csv)
            rm(iter_csv)
        end

        println("\n=== Graph: ", g.name, " (n=", g.n, ", m=", g.m, ", avg_degree=", round(g.avg_degree, digits=3), ") ===")

        if warmup
            warmup_tag = string(chop(g.name, tail=4), "_warmup")
            _run_one_c_mode(
                g.path,
                summary_csv,
                epsilon=epsilon,
                start_epsilon_d0=start_epsilon_d0,
                tighten_arnoldi_tol=tighten_arnoldi_tol,
                linesearch=linesearch,
                benchmark_tag=warmup_tag,
                arnoldi_mindim=arnoldi_mindim,
                arnoldi_maxdim=arnoldi_maxdim,
                arnoldi_restarts=arnoldi_restarts,
                adaptive_arnoldi_budget=adaptive_arnoldi_budget,
                update_low_threshold=update_low_threshold,
                update_high_threshold=update_high_threshold,
                low_budget_scale=low_budget_scale,
                high_budget_scale=high_budget_scale,
            )
        end

        for rep in 1:repeats
            run_tag = string(chop(g.name, tail=4), "_run", rep)
            result = _run_one_c_mode(
                g.path,
                summary_csv,
                epsilon=epsilon,
                start_epsilon_d0=start_epsilon_d0,
                tighten_arnoldi_tol=tighten_arnoldi_tol,
                linesearch=linesearch,
                benchmark_tag=run_tag,
                arnoldi_mindim=arnoldi_mindim,
                arnoldi_maxdim=arnoldi_maxdim,
                arnoldi_restarts=arnoldi_restarts,
                adaptive_arnoldi_budget=adaptive_arnoldi_budget,
                update_low_threshold=update_low_threshold,
                update_high_threshold=update_high_threshold,
                low_budget_scale=low_budget_scale,
                high_budget_scale=high_budget_scale,
            )

            push!(run_rows, [
                g.name,
                g.path,
                rep,
                result.n,
                result.m,
                result.avg_degree,
                linesearch,
                epsilon,
                start_epsilon_d0,
                tighten_arnoldi_tol,
                result.total_wall_sec,
                result.io_time_sec,
                result.bench.solve_total_sec,
                result.bench.fw_iterations,
                result.bench.lmo_calls,
                result.bench.total_lmo_sec,
                result.bench.avg_lmo_sec,
                result.bench.avg_partialschur_sec,
                result.bench.avg_partialeigen_sec,
                result.bench.total_mvproducts,
                result.bench.lmo_share,
                result.io_time_sec / result.total_wall_sec,
                (result.bench.solve_total_sec - result.bench.total_lmo_sec) / result.total_wall_sec,
                result.bench.total_lmo_sec / result.total_wall_sec,
                result.bench.final_gap,
                result.bench.converged,
                result.bench.median_delta_v_l2,
                result.bench.p90_delta_v_l2,
                result.bench.corr_delta_v_to_next_lmo_time,
                result.bench.corr_delta_v_to_next_mvproducts,
                result.bench.corr_delta_v_to_delta_lambda,
                adaptive_arnoldi_budget,
                arnoldi_mindim,
                arnoldi_maxdim,
                arnoldi_restarts,
                update_low_threshold,
                update_high_threshold,
                low_budget_scale,
                high_budget_scale,
            ])

            println(@sprintf(
                "[C-scaling] %s rep=%d total=%.3fs io=%.3fs solve=%.3fs lmo=%.3fs lmo(total)=%.1f%%",
                g.name,
                rep,
                result.total_wall_sec,
                result.io_time_sec,
                result.bench.solve_total_sec,
                result.bench.total_lmo_sec,
                100 * result.bench.total_lmo_sec / result.total_wall_sec,
            ))
        end
    end

    runs_csv = joinpath(outdir, "runs.csv")
    _write_csv(
        runs_csv,
        [
            "graph_name", "graph_path", "repeat", "n", "m", "avg_degree",
            "linesearch", "epsilon", "start_epsilon_d0", "tighten_arnoldi_tol",
            "total_wall_time_sec", "io_time_sec", "solve_time_sec",
            "fw_iterations", "lmo_calls", "total_lmo_time_sec", "avg_lmo_time_sec",
            "avg_partialschur_time_sec", "avg_partialeigen_time_sec", "total_mvproducts",
            "lmo_share_of_solve", "io_share_of_total", "non_lmo_solver_share_of_total", "lmo_share_of_total",
            "final_gap", "converged",
            "median_delta_v_l2", "p90_delta_v_l2",
            "corr_delta_v_to_next_lmo_time", "corr_delta_v_to_next_mvproducts", "corr_delta_v_to_delta_lambda",
            "adaptive_arnoldi_budget", "arnoldi_mindim", "arnoldi_maxdim", "arnoldi_restarts",
            "update_low_threshold", "update_high_threshold", "low_budget_scale", "high_budget_scale",
        ],
        run_rows,
    )

    grouped = Dict{String, Vector{Vector}}()
    for row in run_rows
        key = string(row[4], "|", row[5], "|", row[6])
        if !haskey(grouped, key)
            grouped[key] = Vector{Vector}()
        end
        push!(grouped[key], row)
    end

    by_size_rows = Vector{Vector}()
    for (key, rows) in sort(collect(grouped), by=x -> x[1])
        ns = [Float64(r[4]) for r in rows]
        ms = [Float64(r[5]) for r in rows]
        ds = [Float64(r[6]) for r in rows]
        totals = [Float64(r[10]) for r in rows]
        ios = [Float64(r[11]) for r in rows]
        solves = [Float64(r[12]) for r in rows]
        fw_iters = [Float64(r[13]) for r in rows]
        lmos = [Float64(r[15]) for r in rows]
        lmo_total_shares = [Float64(r[23]) for r in rows]

        push!(by_size_rows, [
            Int(round(mean(ns))),
            Int(round(mean(ms))),
            mean(ds),
            length(rows),
            mean(totals),
            std(totals),
            mean(ios),
            mean(solves),
            mean(fw_iters),
            mean(lmos),
            mean(lmo_total_shares),
        ])
    end

    by_size_csv = joinpath(outdir, "by_size.csv")
    _write_csv(
        by_size_csv,
        [
            "n", "m", "avg_degree", "runs",
            "mean_total_wall_time_sec", "std_total_wall_time_sec",
            "mean_io_time_sec", "mean_solve_time_sec", "mean_fw_iterations",
            "mean_total_lmo_time_sec", "mean_lmo_share_of_total",
        ],
        by_size_rows,
    )

    println("\nC-mode scaling benchmark complete.")
    println("Raw runs CSV: ", runs_csv)
    println("Aggregated by-size CSV: ", by_size_csv)
    println("Per-graph solver CSVs: ", per_graph_dir)
    println("  - *_summary.csv: one row per run (from Solve)")
    println("  - *_summary.csv.iters.csv: one row per FW iteratrunion")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_c_mode_scaling_benchmark(
        repeats=3,
        warmup=true,
        seed=42,
        epsilon=1e-2,
        start_epsilon_d0=0.0,
        linesearch=false,
        include_variants=true,
    )
end
