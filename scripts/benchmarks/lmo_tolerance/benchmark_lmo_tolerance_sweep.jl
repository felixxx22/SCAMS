using Pkg

project_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(project_root)
try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate failed for current Julia version; attempting Pkg.resolve() + Pkg.instantiate()" err
    Pkg.resolve()
    Pkg.instantiate()
end

include("../../../MESDP.jl")
include("../../../ReadGSet.jl")

using Printf
using Random
using Statistics

function _read_existing_run_tags(summary_csv::AbstractString)
    tags = Set{String}()
    if !isfile(summary_csv)
        return tags
    end

    open(summary_csv, "r") do io
        header_line = readline(io)
        header = split(chomp(header_line), ",")
        tag_idx = findfirst(==("run_tag"), header)
        if tag_idx === nothing
            return tags
        end

        for line in eachline(io)
            row = split(chomp(line), ",")
            if length(row) >= tag_idx
                tag = strip(row[tag_idx])
                if !isempty(tag)
                    push!(tags, tag)
                end
            end
        end
    end

    return tags
end

function _read_runs_csv_rows(path::AbstractString)
    rows = Vector{Vector}()
    if !isfile(path)
        return rows
    end

    open(path, "r") do io
        if eof(io)
            return rows
        end
        # Skip header
        readline(io)
        for line in eachline(io)
            line = chomp(line)
            if isempty(line)
                continue
            end
            push!(rows, split(line, ","))
        end
    end

    return rows
end

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
    stats = [merge((path=p, name=basename(p), dataset="BigExample"), _scan_graph_stats(p)) for p in paths]
    sort!(stats, by=s -> (s.n, s.avg_degree, s.name))
    return stats
end

function _gset_numeric_id(name::AbstractString)
    # g1.txt -> 1, g14.txt -> 14
    stem = chop(name, tail=4)
    return parse(Int, stem[2:end])
end

function _list_gset_files()
    files = filter(f -> endswith(f, ".txt"), readdir("Gset"))
    paths = [joinpath("Gset", f) for f in files]
    stats = [merge((path=p, name=basename(p), dataset="Gset"), _scan_graph_stats(p)) for p in paths]
    sort!(stats, by=s -> (_gset_numeric_id(s.name), s.n, s.avg_degree))
    return stats
end

function _representative_indices(len::Int, k::Int)
    if len <= 0
        return Int[]
    end
    kk = clamp(k, 1, len)
    if kk == 1
        return [Int(cld(len, 2))]
    end

    raw = round.(Int, range(1, len, length=kk))
    idx = sort(unique(raw))
    if length(idx) == kk
        return idx
    end

    for i in 1:len
        if !(i in idx)
            push!(idx, i)
            if length(idx) == kk
                break
            end
        end
    end
    sort!(idx)
    return idx
end

function _pick_representative(specs::Vector, k::Int)
    if isempty(specs)
        return specs
    end
    idx = _representative_indices(length(specs), k)
    return specs[idx]
end

function _run_one_c_mode(input_file::AbstractString, log_file::AbstractString;
    epsilon=1e-2,
    start_epsilon_d0=-3.0,
    tighten_arnoldi_tol=false,
    linesearch=false,
    benchmark_tag=nothing,
    arnoldi_mindim=12,
    arnoldi_maxdim=20,
    arnoldi_restarts=500,
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
        adaptive_arnoldi_budget=false,
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

function _graph_stem(name::AbstractString)
    return chop(name, tail=4)
end

function _format_tol_tag(start_epsilon_d0::Real)
    return replace(string(start_epsilon_d0), "." => "p", "-" => "m")
end

function _collect_group_stats(values::Vector{Float64})
    if isempty(values)
        return (mean=0.0, std=0.0, min=0.0, max=0.0)
    end
    return (
        mean=mean(values),
        std=length(values) > 1 ? std(values) : 0.0,
        min=minimum(values),
        max=maximum(values),
    )
end

function _aggregate_by_tolerance(run_rows::Vector{Vector})
    grouped = Dict{Tuple{String, Float64}, Vector{Vector}}()
    for row in run_rows
        key = (String(row[1]), Float64(row[8]))
        if !haskey(grouped, key)
            grouped[key] = Vector{Vector}()
        end
        push!(grouped[key], row)
    end

    out_rows = Vector{Vector}()
    for ((dataset, start_epsilon_d0), rows) in sort(collect(grouped), by=x -> (x[1][1], x[1][2]))
        fw_values = [Float64(r[15]) for r in rows]
        lmo_values = [Float64(r[17]) for r in rows]
        solve_values = [Float64(r[14]) for r in rows]
        gap_values = [Float64(r[18]) for r in rows]

        fw_stats = _collect_group_stats(fw_values)
        lmo_stats = _collect_group_stats(lmo_values)
        solve_stats = _collect_group_stats(solve_values)
        gap_stats = _collect_group_stats(gap_values)

        push!(out_rows, [
            dataset,
            start_epsilon_d0,
            10.0^(start_epsilon_d0),
            length(rows),
            fw_stats.mean,
            fw_stats.std,
            lmo_stats.mean,
            lmo_stats.std,
            solve_stats.mean,
            solve_stats.std,
            gap_stats.mean,
            gap_stats.std,
            gap_stats.min,
            gap_stats.max,
        ])
    end
    return out_rows
end

function _aggregate_by_graph_tolerance(run_rows::Vector{Vector})
    grouped = Dict{Tuple{String, String, Float64}, Vector{Vector}}()
    for row in run_rows
        key = (String(row[1]), String(row[2]), Float64(row[8]))
        if !haskey(grouped, key)
            grouped[key] = Vector{Vector}()
        end
        push!(grouped[key], row)
    end

    out_rows = Vector{Vector}()
    for ((dataset, graph_name, start_epsilon_d0), rows) in sort(collect(grouped), by=x -> (x[1][1], x[1][2], x[1][3]))
        fw_values = [Float64(r[15]) for r in rows]
        lmo_values = [Float64(r[17]) for r in rows]
        solve_values = [Float64(r[14]) for r in rows]
        gap_values = [Float64(r[18]) for r in rows]

        fw_stats = _collect_group_stats(fw_values)
        lmo_stats = _collect_group_stats(lmo_values)
        solve_stats = _collect_group_stats(solve_values)
        gap_stats = _collect_group_stats(gap_values)

        push!(out_rows, [
            dataset,
            graph_name,
            start_epsilon_d0,
            10.0^(start_epsilon_d0),
            length(rows),
            fw_stats.mean,
            fw_stats.std,
            lmo_stats.mean,
            lmo_stats.std,
            solve_stats.mean,
            solve_stats.std,
            gap_stats.mean,
            gap_stats.std,
            gap_stats.min,
            gap_stats.max,
        ])
    end
    return out_rows
end

function run_lmo_tolerance_asymptotic_sweep(; 
    phase="subset",
    repeats=3,
    warmup=true,
    resume_if_possible=false,
    seed=42,
    epsilon=1e-2,
    linesearch=false,
    start_epsilon_d0_values=[-1.0, -2.0, -3.0, -4.0, -5.0],
    subset_k_per_dataset=3,
    include_bigexample_variants=true,
    tighten_arnoldi_tol=false,
    arnoldi_mindim=12,
    arnoldi_maxdim=20,
    arnoldi_restarts=500,
)
    if phase != "subset" && phase != "full"
        error("phase must be \"subset\" or \"full\"")
    end

    Random.seed!(seed)

    outdir = joinpath("Result", "benchmarks_lmo_tolerance", phase)
    mkpath(outdir)

    gset_specs_all = _list_gset_files()
    big_specs_all = _list_bigexample_files(include_variants=include_bigexample_variants)

    if isempty(gset_specs_all)
        error("No Gset .txt files found.")
    end
    if isempty(big_specs_all)
        error("No BigExample .txt files found.")
    end

    gset_specs = phase == "subset" ? _pick_representative(gset_specs_all, subset_k_per_dataset) : gset_specs_all
    big_specs = phase == "subset" ? _pick_representative(big_specs_all, subset_k_per_dataset) : big_specs_all

    selected_rows = Vector{Vector}()
    for g in gset_specs
        push!(selected_rows, ["Gset", g.name, g.path, g.n, g.m, g.avg_degree])
    end
    for g in big_specs
        push!(selected_rows, ["BigExample", g.name, g.path, g.n, g.m, g.avg_degree])
    end

    _write_csv(
        joinpath(outdir, "selected_graphs.csv"),
        ["dataset", "graph_name", "graph_path", "n", "m", "avg_degree"],
        selected_rows,
    )

    dataset_specs = [
        (dataset="Gset", specs=gset_specs),
        (dataset="BigExample", specs=big_specs),
    ]

    runs_csv = joinpath(outdir, "runs.csv")
    run_rows = resume_if_possible ? _read_runs_csv_rows(runs_csv) : Vector{Vector}()
    if resume_if_possible && !isempty(run_rows)
        println("Resume mode: loaded ", length(run_rows), " existing run row(s) from ", runs_csv)
    end

    for ds in dataset_specs
        per_graph_dir = joinpath(outdir, ds.dataset, "per_graph")
        mkpath(per_graph_dir)

        for g in ds.specs
            summary_csv = joinpath(per_graph_dir, string(_graph_stem(g.name), "_summary.csv"))
            iter_csv = string(summary_csv, ".iters.csv")

            if !resume_if_possible
                if isfile(summary_csv)
                    rm(summary_csv)
                end
                if isfile(iter_csv)
                    rm(iter_csv)
                end
            end

            existing_tags = resume_if_possible ? _read_existing_run_tags(summary_csv) : Set{String}()

            println("\n=== Dataset: ", ds.dataset, " Graph: ", g.name,
                " (n=", g.n, ", m=", g.m, ", avg_degree=", round(g.avg_degree, digits=3), ") ===")

            if warmup
                warmup_tag = string(ds.dataset, "_", _graph_stem(g.name), "_warmup")
                if resume_if_possible && (warmup_tag in existing_tags)
                    println("[LMO-sweep] skip existing warmup tag=", warmup_tag)
                else
                    _run_one_c_mode(
                        g.path,
                        summary_csv,
                        epsilon=epsilon,
                        start_epsilon_d0=start_epsilon_d0_values[1],
                        tighten_arnoldi_tol=tighten_arnoldi_tol,
                        linesearch=linesearch,
                        benchmark_tag=warmup_tag,
                        arnoldi_mindim=arnoldi_mindim,
                        arnoldi_maxdim=arnoldi_maxdim,
                        arnoldi_restarts=arnoldi_restarts,
                    )
                    push!(existing_tags, warmup_tag)
                end
            end

            for start_epsilon_d0 in start_epsilon_d0_values
                for rep in 1:repeats
                    tol_tag = _format_tol_tag(start_epsilon_d0)
                    run_tag = string(ds.dataset, "_", _graph_stem(g.name), "_tol", tol_tag, "_run", rep)

                    if resume_if_possible && (run_tag in existing_tags)
                        println("[LMO-sweep] skip existing run tag=", run_tag)
                        continue
                    end

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
                    )

                    push!(run_rows, [
                        ds.dataset,
                        g.name,
                        g.path,
                        rep,
                        g.n,
                        g.m,
                        g.avg_degree,
                        start_epsilon_d0,
                        10.0^(start_epsilon_d0),
                        linesearch,
                        epsilon,
                        tighten_arnoldi_tol,
                        run_tag,
                        result.bench.solve_total_sec,
                        result.bench.fw_iterations,
                        result.bench.lmo_calls,
                        result.bench.total_lmo_sec,
                        result.bench.final_gap,
                        result.bench.converged,
                        result.total_wall_sec,
                        result.io_time_sec,
                        result.bench.total_mvproducts,
                    ])
                    push!(existing_tags, run_tag)

                    println(@sprintf(
                        "[LMO-sweep] dataset=%s graph=%s tol=1e%.0f rep=%d solve=%.3fs fw=%d lmo=%.3fs final_gap=%.3e",
                        ds.dataset,
                        g.name,
                        start_epsilon_d0,
                        rep,
                        result.bench.solve_total_sec,
                        result.bench.fw_iterations,
                        result.bench.total_lmo_sec,
                        result.bench.final_gap,
                    ))
                end
            end
        end
    end

    runs_csv = joinpath(outdir, "runs.csv")
    _write_csv(
        runs_csv,
        [
            "dataset", "graph_name", "graph_path", "repeat", "n", "m", "avg_degree",
            "start_epsilon_d0", "lmo_tolerance", "linesearch", "epsilon", "tighten_arnoldi_tol",
            "run_tag", "solve_time_sec", "fw_iterations", "lmo_calls", "total_lmo_time_sec",
            "final_gap", "converged", "total_wall_time_sec", "io_time_sec", "total_mvproducts",
        ],
        run_rows,
    )

    by_tol_rows = _aggregate_by_tolerance(run_rows)
    _write_csv(
        joinpath(outdir, "by_tolerance.csv"),
        [
            "dataset", "start_epsilon_d0", "lmo_tolerance", "runs",
            "mean_fw_iterations", "std_fw_iterations",
            "mean_total_lmo_time_sec", "std_total_lmo_time_sec",
            "mean_solve_time_sec", "std_solve_time_sec",
            "mean_final_gap", "std_final_gap", "min_final_gap", "max_final_gap",
        ],
        by_tol_rows,
    )

    by_graph_tol_rows = _aggregate_by_graph_tolerance(run_rows)
    _write_csv(
        joinpath(outdir, "by_graph_tolerance.csv"),
        [
            "dataset", "graph_name", "start_epsilon_d0", "lmo_tolerance", "runs",
            "mean_fw_iterations", "std_fw_iterations",
            "mean_total_lmo_time_sec", "std_total_lmo_time_sec",
            "mean_solve_time_sec", "std_solve_time_sec",
            "mean_final_gap", "std_final_gap", "min_final_gap", "max_final_gap",
        ],
        by_graph_tol_rows,
    )

    expected_runs = length(gset_specs) * length(start_epsilon_d0_values) * repeats +
                    length(big_specs) * length(start_epsilon_d0_values) * repeats

    println("\nLMO tolerance sweep complete.")
    println("Phase: ", phase)
    println("Runs CSV: ", runs_csv)
    println("By-tolerance CSV: ", joinpath(outdir, "by_tolerance.csv"))
    println("By-graph/tolerance CSV: ", joinpath(outdir, "by_graph_tolerance.csv"))
    println("Selected graphs CSV: ", joinpath(outdir, "selected_graphs.csv"))
    println("Expected run count: ", expected_runs, " | Actual run count: ", length(run_rows))

    return (
        outdir=outdir,
        expected_runs=expected_runs,
        actual_runs=length(run_rows),
        gset_graphs=length(gset_specs),
        bigexample_graphs=length(big_specs),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_lmo_tolerance_asymptotic_sweep(
        phase="subset",
        repeats=3,
        warmup=true,
        resume_if_possible=true,
        seed=42,
        epsilon=1e-2,
        linesearch=false,
        start_epsilon_d0_values=[-1.0, -2.0, -3.0, -4.0, -5.0],
        subset_k_per_dataset=3,
        include_bigexample_variants=true,
        tighten_arnoldi_tol=false,
        arnoldi_mindim=12,
        arnoldi_maxdim=20,
        arnoldi_restarts=500,
    )
end
