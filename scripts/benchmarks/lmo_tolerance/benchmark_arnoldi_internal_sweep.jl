include("../../../MESDP.jl")
include("../../../ReadGSet.jl")

using Printf
using Random
using SparseArrays
using Statistics

function _read_existing_run_tags(summary_csv::AbstractString)
    tags = Set{String}()
    if !isfile(summary_csv)
        return tags
    end

    open(summary_csv, "r") do io
        if eof(io)
            return tags
        end
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

function _prepare_c_mode_data(input_file::AbstractString)
    A, C = readfile(input_file)
    A = A / 2
    C = C / 4

    local_m = size(A, 1)
    local_n = size(A, 2)

    D = spzeros(local_n)
    sumai = 0.0
    for i in 1:local_n
        D[i] = 2 * C[i, i]
        sumai += D[i]
    end
    upper = sqrt(sumai)

    # Compute v0 locally to avoid dependence on MESDP.jl global n/m state.
    vals = nonzeros(A)
    v0 = zeros(Float64, local_n)
    d0 = 1 / local_m
    for j in 1:local_n
        s = 0.0
        for k in nzrange(A, j)
            s += vals[k]^2
        end
        v0[j] = d0 * s
    end

    return (A=A, C=C, D=D, v0=v0, upper=upper, n=local_n, m=local_m)
end

function _run_one_solve_mode(input_file::AbstractString, log_file::AbstractString;
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
    data = _prepare_c_mode_data(input_file)
    io_time_sec = (time_ns() - io_start_ns) / 1e9

    # MESDP currently uses global n/m in B, CutValue, and related helpers.
    global m = data.m
    global n = data.n

    solve_result = Solve(
        data.C,
        data.v0,
        t0=5,
        D=data.D,
        lowerBound=0,
        upperBound=data.upper,
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
    return (
        n=data.n,
        m=data.m,
        avg_degree=(2.0 * data.m) / data.n,
        io_time_sec=io_time_sec,
        total_wall_sec=io_time_sec + bench.solve_total_sec,
        solve_time_sec=bench.solve_total_sec,
        fw_iterations=bench.fw_iterations,
        lmo_calls=bench.lmo_calls,
        total_lmo_time_sec=bench.total_lmo_sec,
        avg_restart_iters=bench.avg_restart_iters,
        total_arnoldi_expand_time_sec=bench.total_arnoldi_expand_sec,
        total_arnoldi_schur_time_sec=bench.total_arnoldi_schur_sec,
        total_arnoldi_partition_time_sec=bench.total_arnoldi_partition_sec,
        total_arnoldi_restore_time_sec=bench.total_arnoldi_restore_sec,
        total_arnoldi_basis_time_sec=bench.total_arnoldi_basis_sec,
        total_arnoldi_finalize_time_sec=bench.total_arnoldi_finalize_sec,
        total_arnoldi_restart_only_time_sec=bench.total_arnoldi_restart_only_sec,
        total_arnoldi_phase_sum_time_sec=bench.total_arnoldi_phase_sum_sec,
        arnoldi_phase_coverage=bench.arnoldi_phase_coverage,
        final_gap=bench.final_gap,
        converged=bench.converged,
        total_mvproducts=bench.total_mvproducts,
    )
end

function _run_one_direct_mode(input_file::AbstractString;
    start_epsilon_d0=-3.0,
    arnoldi_mindim=12,
    arnoldi_maxdim=20,
    arnoldi_restarts=500,
)
    io_start_ns = time_ns()
    data = _prepare_c_mode_data(input_file)
    io_time_sec = (time_ns() - io_start_ns) / 1e9

    # Direct ArnoldiGrad also calls B(A, v=w), which relies on global n.
    global m = data.m
    global n = data.n

    t0_ns = time_ns()
    _, _, _, metrics = ArnoldiGrad(
        data.C,
        data.v0,
        lowerBound=0,
        upperBound=data.upper,
        D=data.D,
        mode="C",
        tol=10.0^(start_epsilon_d0),
        returnMetrics=true,
        arnoldi_mindim=arnoldi_mindim,
        arnoldi_maxdim=arnoldi_maxdim,
        arnoldi_restarts=arnoldi_restarts,
    )
    direct_wall_sec = (time_ns() - t0_ns) / 1e9

    return (
        n=data.n,
        m=data.m,
        avg_degree=(2.0 * data.m) / data.n,
        io_time_sec=io_time_sec,
        total_wall_sec=io_time_sec + direct_wall_sec,
        solve_time_sec=metrics.total_ns / 1e9,
        fw_iterations=0,
        lmo_calls=1,
        total_lmo_time_sec=metrics.total_ns / 1e9,
        avg_restart_iters=Float64(metrics.restart_iters),
        total_arnoldi_expand_time_sec=metrics.partialschur_expand_ns / 1e9,
        total_arnoldi_schur_time_sec=metrics.partialschur_schur_ns / 1e9,
        total_arnoldi_partition_time_sec=metrics.partialschur_partition_ns / 1e9,
        total_arnoldi_restore_time_sec=metrics.partialschur_restore_ns / 1e9,
        total_arnoldi_basis_time_sec=metrics.partialschur_basis_ns / 1e9,
        total_arnoldi_finalize_time_sec=metrics.partialschur_finalize_ns / 1e9,
        total_arnoldi_restart_only_time_sec=metrics.partialschur_restart_total_ns / 1e9,
        total_arnoldi_phase_sum_time_sec=metrics.partialschur_phase_sum_ns / 1e9,
        arnoldi_phase_coverage=metrics.partialschur_ns == 0 ? 0.0 : metrics.partialschur_phase_sum_ns / metrics.partialschur_ns,
        final_gap=0.0,
        converged=metrics.converged,
        total_mvproducts=metrics.mvproducts,
        restart_expand_ns=metrics.restart_expand_ns,
        restart_schur_ns=metrics.restart_schur_ns,
        restart_partition_ns=metrics.restart_partition_ns,
        restart_restore_ns=metrics.restart_restore_ns,
        restart_basis_ns=metrics.restart_basis_ns,
        restart_total_ns_by_iter=metrics.restart_total_ns_by_iter,
    )
end

function _aggregate_by_tolerance(run_rows::Vector{Vector})
    grouped = Dict{Tuple{String, String, Float64}, Vector{Vector}}()
    for row in run_rows
        key = (String(row[1]), String(row[2]), Float64(row[9]))
        if !haskey(grouped, key)
            grouped[key] = Vector{Vector}()
        end
        push!(grouped[key], row)
    end

    out_rows = Vector{Vector}()
    for ((dataset, driver_mode, start_epsilon_d0), rows) in sort(collect(grouped), by=x -> (x[1][1], x[1][2], x[1][3]))
        solve_values = [Float64(r[15]) for r in rows]
        fw_values = [Float64(r[16]) for r in rows]
        lmo_values = [Float64(r[18]) for r in rows]
        restart_values = [Float64(r[19]) for r in rows]
        expand_values = [Float64(r[20]) for r in rows]
        restore_values = [Float64(r[23]) for r in rows]
        basis_values = [Float64(r[24]) for r in rows]
        coverage_values = [Float64(r[28]) for r in rows]
        gap_values = [Float64(r[29]) for r in rows]

        solve_stats = _collect_group_stats(solve_values)
        fw_stats = _collect_group_stats(fw_values)
        lmo_stats = _collect_group_stats(lmo_values)
        restart_stats = _collect_group_stats(restart_values)
        expand_stats = _collect_group_stats(expand_values)
        restore_stats = _collect_group_stats(restore_values)
        basis_stats = _collect_group_stats(basis_values)
        coverage_stats = _collect_group_stats(coverage_values)
        gap_stats = _collect_group_stats(gap_values)

        push!(out_rows, [
            dataset,
            driver_mode,
            start_epsilon_d0,
            10.0^(start_epsilon_d0),
            length(rows),
            solve_stats.mean,
            solve_stats.std,
            fw_stats.mean,
            fw_stats.std,
            lmo_stats.mean,
            lmo_stats.std,
            restart_stats.mean,
            restart_stats.std,
            expand_stats.mean,
            expand_stats.std,
            restore_stats.mean,
            restore_stats.std,
            basis_stats.mean,
            basis_stats.std,
            coverage_stats.mean,
            coverage_stats.std,
            gap_stats.mean,
            gap_stats.std,
            gap_stats.min,
            gap_stats.max,
        ])
    end
    return out_rows
end

function _aggregate_by_graph_tolerance(run_rows::Vector{Vector})
    grouped = Dict{Tuple{String, String, String, Float64}, Vector{Vector}}()
    for row in run_rows
        key = (String(row[1]), String(row[2]), String(row[3]), Float64(row[9]))
        if !haskey(grouped, key)
            grouped[key] = Vector{Vector}()
        end
        push!(grouped[key], row)
    end

    out_rows = Vector{Vector}()
    for ((dataset, driver_mode, graph_name, start_epsilon_d0), rows) in sort(collect(grouped), by=x -> (x[1][1], x[1][2], x[1][3], x[1][4]))
        solve_values = [Float64(r[15]) for r in rows]
        restart_values = [Float64(r[19]) for r in rows]
        expand_values = [Float64(r[20]) for r in rows]
        restore_values = [Float64(r[23]) for r in rows]
        basis_values = [Float64(r[24]) for r in rows]
        coverage_values = [Float64(r[28]) for r in rows]

        solve_stats = _collect_group_stats(solve_values)
        restart_stats = _collect_group_stats(restart_values)
        expand_stats = _collect_group_stats(expand_values)
        restore_stats = _collect_group_stats(restore_values)
        basis_stats = _collect_group_stats(basis_values)
        coverage_stats = _collect_group_stats(coverage_values)

        push!(out_rows, [
            dataset,
            driver_mode,
            graph_name,
            start_epsilon_d0,
            10.0^(start_epsilon_d0),
            length(rows),
            solve_stats.mean,
            solve_stats.std,
            restart_stats.mean,
            restart_stats.std,
            expand_stats.mean,
            expand_stats.std,
            restore_stats.mean,
            restore_stats.std,
            basis_stats.mean,
            basis_stats.std,
            coverage_stats.mean,
            coverage_stats.std,
        ])
    end
    return out_rows
end

function run_arnoldi_internal_timing_sweep(; 
    phase="subset",
    driver_mode="solve",
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
    if driver_mode != "solve" && driver_mode != "direct"
        error("driver_mode must be \"solve\" or \"direct\"")
    end

    Random.seed!(seed)

    outdir = joinpath("Result", "benchmarks_arnoldi_internal_timing", string(phase, "_", driver_mode))
    mkpath(outdir)

    gset_specs_all = _list_gset_files()
    big_specs_all = _list_bigexample_files(include_variants=include_bigexample_variants)

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

    restart_rows = Vector{Vector}()
    restart_csv = joinpath(outdir, "restarts.csv")

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
                warmup_tag = string(ds.dataset, "_", _graph_stem(g.name), "_", driver_mode, "_warmup")
                if resume_if_possible && (warmup_tag in existing_tags)
                    println("[Arnoldi-internal] skip existing warmup tag=", warmup_tag)
                else
                    if driver_mode == "solve"
                        _run_one_solve_mode(
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
                    else
                        _run_one_direct_mode(
                            g.path,
                            start_epsilon_d0=start_epsilon_d0_values[1],
                            arnoldi_mindim=arnoldi_mindim,
                            arnoldi_maxdim=arnoldi_maxdim,
                            arnoldi_restarts=arnoldi_restarts,
                        )
                    end
                    push!(existing_tags, warmup_tag)
                end
            end

            for start_epsilon_d0 in start_epsilon_d0_values
                for rep in 1:repeats
                    tol_tag = _format_tol_tag(start_epsilon_d0)
                    run_tag = string(ds.dataset, "_", _graph_stem(g.name), "_", driver_mode, "_tol", tol_tag, "_run", rep)

                    if resume_if_possible && (run_tag in existing_tags)
                        println("[Arnoldi-internal] skip existing run tag=", run_tag)
                        continue
                    end

                    result = if driver_mode == "solve"
                        _run_one_solve_mode(
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
                    else
                        _run_one_direct_mode(
                            g.path,
                            start_epsilon_d0=start_epsilon_d0,
                            arnoldi_mindim=arnoldi_mindim,
                            arnoldi_maxdim=arnoldi_maxdim,
                            arnoldi_restarts=arnoldi_restarts,
                        )
                    end

                    push!(run_rows, [
                        ds.dataset,
                        driver_mode,
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
                        result.solve_time_sec,
                        result.fw_iterations,
                        result.lmo_calls,
                        result.total_lmo_time_sec,
                        result.avg_restart_iters,
                        result.total_arnoldi_expand_time_sec,
                        result.total_arnoldi_schur_time_sec,
                        result.total_arnoldi_partition_time_sec,
                        result.total_arnoldi_restore_time_sec,
                        result.total_arnoldi_basis_time_sec,
                        result.total_arnoldi_finalize_time_sec,
                        result.total_arnoldi_restart_only_time_sec,
                        result.total_arnoldi_phase_sum_time_sec,
                        result.arnoldi_phase_coverage,
                        result.final_gap,
                        result.converged,
                        result.total_wall_sec,
                        result.io_time_sec,
                        result.total_mvproducts,
                    ])

                    if driver_mode == "direct" && hasproperty(result, :restart_expand_ns)
                        restart_count = length(result.restart_total_ns_by_iter)
                        for i in 1:restart_count
                            push!(restart_rows, [
                                ds.dataset,
                                g.name,
                                g.path,
                                rep,
                                start_epsilon_d0,
                                10.0^(start_epsilon_d0),
                                run_tag,
                                i,
                                result.restart_expand_ns[i] / 1e9,
                                result.restart_schur_ns[i] / 1e9,
                                result.restart_partition_ns[i] / 1e9,
                                result.restart_restore_ns[i] / 1e9,
                                result.restart_basis_ns[i] / 1e9,
                                result.restart_total_ns_by_iter[i] / 1e9,
                            ])
                        end
                    end

                    push!(existing_tags, run_tag)

                    println(@sprintf(
                        "[Arnoldi-internal] mode=%s dataset=%s graph=%s tol=1e%.0f rep=%d solve=%.3fs expand=%.3fs restore=%.3fs basis=%.3fs",
                        driver_mode,
                        ds.dataset,
                        g.name,
                        start_epsilon_d0,
                        rep,
                        result.solve_time_sec,
                        result.total_arnoldi_expand_time_sec,
                        result.total_arnoldi_restore_time_sec,
                        result.total_arnoldi_basis_time_sec,
                    ))
                end
            end
        end
    end

    _write_csv(
        runs_csv,
        [
            "dataset", "driver_mode", "graph_name", "graph_path", "repeat", "n", "m", "avg_degree",
            "start_epsilon_d0", "lmo_tolerance", "linesearch", "epsilon", "tighten_arnoldi_tol",
            "run_tag", "solve_time_sec", "fw_iterations", "lmo_calls", "total_lmo_time_sec", "avg_restart_iters",
            "total_arnoldi_expand_time_sec", "total_arnoldi_schur_time_sec", "total_arnoldi_partition_time_sec",
            "total_arnoldi_restore_time_sec", "total_arnoldi_basis_time_sec", "total_arnoldi_finalize_time_sec",
            "total_arnoldi_restart_only_time_sec", "total_arnoldi_phase_sum_time_sec", "arnoldi_phase_coverage",
            "final_gap", "converged", "total_wall_time_sec", "io_time_sec", "total_mvproducts",
        ],
        run_rows,
    )

    by_tol_rows = _aggregate_by_tolerance(run_rows)
    _write_csv(
        joinpath(outdir, "by_tolerance.csv"),
        [
            "dataset", "driver_mode", "start_epsilon_d0", "lmo_tolerance", "runs",
            "mean_solve_time_sec", "std_solve_time_sec",
            "mean_fw_iterations", "std_fw_iterations",
            "mean_total_lmo_time_sec", "std_total_lmo_time_sec",
            "mean_avg_restart_iters", "std_avg_restart_iters",
            "mean_total_arnoldi_expand_time_sec", "std_total_arnoldi_expand_time_sec",
            "mean_total_arnoldi_restore_time_sec", "std_total_arnoldi_restore_time_sec",
            "mean_total_arnoldi_basis_time_sec", "std_total_arnoldi_basis_time_sec",
            "mean_arnoldi_phase_coverage", "std_arnoldi_phase_coverage",
            "mean_final_gap", "std_final_gap", "min_final_gap", "max_final_gap",
        ],
        by_tol_rows,
    )

    by_graph_tol_rows = _aggregate_by_graph_tolerance(run_rows)
    _write_csv(
        joinpath(outdir, "by_graph_tolerance.csv"),
        [
            "dataset", "driver_mode", "graph_name", "start_epsilon_d0", "lmo_tolerance", "runs",
            "mean_solve_time_sec", "std_solve_time_sec",
            "mean_avg_restart_iters", "std_avg_restart_iters",
            "mean_total_arnoldi_expand_time_sec", "std_total_arnoldi_expand_time_sec",
            "mean_total_arnoldi_restore_time_sec", "std_total_arnoldi_restore_time_sec",
            "mean_total_arnoldi_basis_time_sec", "std_total_arnoldi_basis_time_sec",
            "mean_arnoldi_phase_coverage", "std_arnoldi_phase_coverage",
        ],
        by_graph_tol_rows,
    )

    if driver_mode == "direct" && !isempty(restart_rows)
        _write_csv(
            restart_csv,
            [
                "dataset", "graph_name", "graph_path", "repeat", "start_epsilon_d0", "lmo_tolerance", "run_tag",
                "restart_index", "expand_time_sec", "schur_time_sec", "partition_time_sec",
                "restore_time_sec", "basis_time_sec", "restart_total_time_sec",
            ],
            restart_rows,
        )
    end

    println("\nArnoldi internal timing sweep complete.")
    println("Phase: ", phase, " | Mode: ", driver_mode)
    println("Runs CSV: ", runs_csv)
    println("By-tolerance CSV: ", joinpath(outdir, "by_tolerance.csv"))
    println("By-graph/tolerance CSV: ", joinpath(outdir, "by_graph_tolerance.csv"))
    println("Selected graphs CSV: ", joinpath(outdir, "selected_graphs.csv"))
    if driver_mode == "direct"
        println("Restart-level CSV: ", restart_csv)
    end

    return (
        outdir=outdir,
        runs=length(run_rows),
        gset_graphs=length(gset_specs),
        bigexample_graphs=length(big_specs),
        driver_mode=driver_mode,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_arnoldi_internal_timing_sweep(
        phase="subset",
        driver_mode="solve",
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
