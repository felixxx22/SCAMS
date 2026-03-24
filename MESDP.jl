
include("Arnoldi/ArnoldiMethodMod.jl")
using .ArnoldiMethodMod
using LinearAlgebra
using Distributions
using Statistics
using SparseArrays
using Dates

#==============================MISC=============================#
#Julia Display with endline
function disp(quan; name="")
    if name != ""
        print(name, ":\n")
    end
    display(quan)
    print('\n')
end

"""
    B(A; P=nothing, v=nothing, d=nothing)

    Paper linear map `B`, evaluated by sparse column operations on `A`.

    Mapping to paper variables:
    - `A in R^{m x n}` with columns `a_i = A[:, i]`.
    - Output `r in R^n`, where `r[i] = a_i' X a_i` for a chosen matrix/atom `X`.

    Supported modes in this file:
    - `d` mode: `X = d * I`, so `r[i] = d * ||a_i||^2`.
    - `v` mode: `X = w w'` (here `w` is passed as keyword `v`), so `r[i] = (a_i' w)^2`.
"""
function B(A; P=nothing, v=nothing, d=nothing)
    rows = rowvals(A)
    vals = nonzeros(A)
    r = spzeros(n)

    # Prioritize vector atom mode when both are provided.
    if P !== nothing && v !== nothing
        P = nothing
    end

    for i in 1:n
        if d !== nothing
            # Initialization map: r[i] = d * ||a_i||^2.
            for k in nzrange(A, i)
                r[i] += vals[k]^2
            end
            r[i] = r[i] * d

        elseif v !== nothing
            # LMO map: r[i] = (a_i' * w)^2 where w is passed as `v`. A^T w. 
            for k in nzrange(A, i)
                r[i] += (vals[k] * v[rows[k]])
            end
            r[i] = r[i]^2
        else
            println("B error: No valid input")
        end

    end
    return r
end

"""
    ∇g(v; lowerBound=0, upperBound=1e16, D=ones(1, n))

    Computes the clamped coordinate-wise quantity used by the FW gap and LMO scaling:
    `∇g_i(v) = clamp(1 / (2 * sqrt(v_i)), lowerBound / D_i, upperBound / D_i)`.

    This corresponds to the positive magnitude of `-∂f/∂v_i` for
    `f(v) = -sum_i sqrt(v_i)`.
"""
function ∇g(v; lowerBound=0, upperBound=1e16, D=ones(1, n))
    res = zeros(n)
    for i in 1:n
        res[i] = clamp(1 / (2 * ((v[i])^(1 / 2))), lowerBound / D[i], upperBound / D[i])
    end
    return res
end

"""
    f(v)

    Objective proxy used in the paper mapping:
    `f(v) = | -sum_i sqrt(v_i) |`.

    The absolute value keeps logging and normalized gap values positive.
"""
function f(v)
    r = 0
    for i in 1:n
        r = r - sqrt(v[i])
    end
    return abs(r)
end

"""
    ArnoldiGrad(A, v; lowerBound=0, upperBound=1e16, tol=1e-2, D=ones(1, n), mode="A")

    Algorithm 3 (LMO) wrapper:
    1. Build diagonal scale `λ_i = sqrt(clamp(1/(2*sqrt(v_i)), ...))`.
    2. Run modified Arnoldi (`partialschur`) to get dominant eigenpair.
    3. Map leading eigenvector `w` to FW atom `q = B(A, v=w)`.

    Returns `(w, q, leading_eigenvalue)`.
"""
function ArnoldiGrad(A, v; lowerBound=0, upperBound=1e16, tol=1e-2, D=ones(1, n), mode="A", returnMetrics=false, arnoldi_mindim=12, arnoldi_maxdim=20, arnoldi_restarts=500, arnoldi_initvec=nothing)
    Mn = lowerBound ./ D
    Mx = upperBound ./ D
    λ = sqrt.(clamp.(1 ./ (2 .* sqrt.(v)), Mn, Mx))

    # Clamp user-provided Arnoldi budget to problem size and validity constraints.
    nA = size(A, 1)
    local_mindim = clamp(arnoldi_mindim, 1, nA)
    local_maxdim = clamp(arnoldi_maxdim, local_mindim, nA)
    local_restarts = max(1, arnoldi_restarts)

    t0_ns = time_ns()
    decomp, history = ArnoldiMethodMod.partialschur(
        A,
        λ,
        tol=tol,
        which=ArnoldiMethodMod.LM(),
        mindim=local_mindim,
        maxdim=local_maxdim,
        restarts=local_restarts,
        mode=mode,
        initvec=arnoldi_initvec,
    )
    t1_ns = time_ns()
    eig, eigv = ArnoldiMethodMod.partialeigen(decomp)
    t2_ns = time_ns()
    # Leading eigenvector defines the rank-1 FW atom.
    w = eigv[:, 1]
    q = B(A, v=w)
    if returnMetrics
        metrics = (
            total_ns=t2_ns - t0_ns,
            partialschur_ns=t1_ns - t0_ns,
            partialeigen_ns=t2_ns - t1_ns,
            mvproducts=history.mvproducts,
            nconverged=history.nconverged,
            converged=history.converged,
            nev=history.nev,
            mindim=local_mindim,
            maxdim=local_maxdim,
            restarts=local_restarts,
        )
        return w, q, eig[1], metrics
    end
    return w, q, eig[1]

end


"""
    gammaLineSearch(v, q; ε=1e-8)

    Ternary search on `γ in [0, 1]` for line-search Frank-Wolfe step size:
    `argmin_γ f((1-γ) * v + γ * q)`.
"""
function gammaLineSearch(v, q; ε=1e-8)
    b = 0
    e = 1
    while e - b > ε
        # Two trisection points in the current search interval.
        mid1 = b + (e - b) / 3
        mid2 = e - (e - b) / 3
        vmid1 = (1 - mid1) * v + mid1 * q
        vmid2 = (1 - mid2) * v + mid2 * q
        # Keep the side with lower objective value.
        if f(vmid1) < f(vmid2)
            b = mid1
        else
            e = mid2
        end
    end
    return (e + b) / 2
end

"""
    CutValue(A, z)

    Computes rounded cut value from one Gaussian sample `z`:
    1. Project each vertex-column: `s_i = a_i' z`.
    2. Round signs: `r_i = sign(s_i)`.
    3. Evaluate `0.5 * r' * A' * A * r`.
"""
function CutValue(A, z)
    rows = rowvals(A)
    vals = nonzeros(A)
    r = spzeros(n)

    for i in 1:n
        for k in nzrange(A, i)
            r[i] += (vals[k] * z[rows[k]])
        end
        r[i] = sign(r[i])
    end
    return r' * A' * A * r / 2

end

"""
    Solve(A, v0; D=ones((1, n)), t0=2, ε=1e-3, lowerBound=0, upperBound=1e16,
          plot=false, linesearch=false, numSample=1, mode="A",
          logfilename=nothing, startεd0=-3.0)

    Algorithm 2 (Frank-Wolfe for MESDP):
    1. Start from `v0`.
    2. Call `ArnoldiGrad` (Algorithm 3) to get FW atom `q`.
    3. Compute normalized FW gap `dot(q - v, ∇g(v)) / abs(f(v))`.
    4. Update `v <- (1-γ)v + γq` until gap <= `ε`.
    5. Optionally tighten Arnoldi tolerance as the gap decreases.

    Returns final iterate and best rounded sample/cut surrogate.
"""
function Solve(A, v0; D=ones((1, n)), t0=2, ε=1e-3, lowerBound=0, upperBound=1e16, plot=false, linesearch=false, numSample=1, mode="A", logfilename=nothing, startεd0=-3.0, benchmark=false, benchmarkTag=nothing, arnoldi_mindim=12, arnoldi_maxdim=20, arnoldi_restarts=500, adaptive_arnoldi_budget=false, update_low_threshold=1e-3, update_high_threshold=1e-2, low_budget_scale=0.7, high_budget_scale=1.6, tighten_arnoldi_tol=true, arnoldi_initial_vector=nothing, warmstart_reuse_leading_eigvec=false)
    v = v0
    t = t0
    # Keeps compatibility with the 2/(t+start) schedule used elsewhere.
    start = 0
    solve_start_ns = time_ns()
    run_tag = benchmarkTag === nothing ? Dates.format(now(), dateformat"yyyymmdd_HHMMSS") : string(benchmarkTag)
    iter_log_path = nothing
    summary_log_path = nothing
    if benchmark && logfilename !== nothing
        summary_log_path = string(logfilename)
        iter_log_path = string(logfilename, ".iters.csv")
    end

    lmo_calls = 0
    total_lmo_ns = 0
    total_partialschur_ns = 0
    total_partialeigen_ns = 0
    total_fw_iter_ns = 0
    total_mvproducts = 0

    # Iteration-level warm-start telemetry samples for summary CSV fields.
    delta_v_l2_samples = Float64[]
    next_lmo_time_samples = Float64[]
    next_lmo_mvproducts_samples = Float64[]
    delta_lambda_samples = Float64[]

    # Pre-sample Gaussian vectors for final cut extraction.
    z = rand(Normal(0, 1 / m), (numSample, m))

    # Preserve legacy gap-log behavior only for non-benchmark mode.
    # In benchmark mode, summary/iteration CSVs are append-only and header-managed.
    if logfilename !== nothing && !benchmark
        _ensure_parent_dir(string(logfilename))
        open(string(logfilename), "w") do io end
    end

    requested_initvec = arnoldi_initial_vector
    if requested_initvec !== nothing
        if !(requested_initvec isa AbstractVector)
            @warn "Ignoring arnoldi_initial_vector: expected a vector input."
            requested_initvec = nothing
        elseif length(requested_initvec) != size(A, 1)
            @warn "Ignoring arnoldi_initial_vector: expected length $(size(A, 1)), got $(length(requested_initvec))."
            requested_initvec = nothing
        elseif !all(isfinite, requested_initvec)
            @warn "Ignoring arnoldi_initial_vector: expected finite values."
            requested_initvec = nothing
        elseif norm(requested_initvec) <= eps(Float64)
            @warn "Ignoring arnoldi_initial_vector: expected non-zero norm."
            requested_initvec = nothing
        end
    end
    first_lmo_init_source = requested_initvec === nothing ? "random" : "user"
    current_arnoldi_initvec = requested_initvec

    # Start from configured Arnoldi tolerance; optional tightening can refine it later.
    εd0 = startεd0
    init_mindim, init_maxdim, init_restarts, init_budget_tier = _resolve_arnoldi_budget(
        arnoldi_mindim,
        arnoldi_maxdim,
        arnoldi_restarts,
        0.0,
        adaptive_arnoldi_budget,
        update_low_threshold,
        update_high_threshold,
        low_budget_scale,
        high_budget_scale,
        is_initial=true,
    )
    w, q, λ, lmo_metrics = ArnoldiGrad(
        A,
        v,
        lowerBound=lowerBound,
        upperBound=upperBound,
        D=D,
        mode=mode,
        tol=10^(εd0),
        returnMetrics=true,
        arnoldi_mindim=init_mindim,
        arnoldi_maxdim=init_maxdim,
        arnoldi_restarts=init_restarts,
        arnoldi_initvec=current_arnoldi_initvec,
    )
    lmo_calls += 1
    total_lmo_ns += lmo_metrics.total_ns
    total_partialschur_ns += lmo_metrics.partialschur_ns
    total_partialeigen_ns += lmo_metrics.partialeigen_ns
    total_mvproducts += lmo_metrics.mvproducts
    gap = dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound, D=D)) / abs(f(v))

    # Previous LMO eigenpair acts as baseline for next-call movement tracking.
    prev_w = copy(w)
    prev_λ = λ
    if warmstart_reuse_leading_eigvec
        current_arnoldi_initvec = copy(w)
    else
        current_arnoldi_initvec = nothing
    end

    fw_iter_idx = 0
    while gap > ε
        iter_start_ns = time_ns()
        fw_iter_idx += 1
        v_before = copy(v)

        # FW step size: optional line search after a short burn-in period.
        gamma_source = "schedule"
        if linesearch && t > 10
            gamma = gammaLineSearch(v, q)
            gamma_source = "linesearch"
        else
            gamma = 2 / (t + start)
        end

        t = t + 1
        # Core FW convex combination update.
        v = (1 - gamma) * v + gamma * q

        delta_v = v - v_before
        delta_v_l2 = norm(delta_v)
        delta_v_linf = norm(delta_v, Inf)
        denom_v = max(norm(v_before), eps(Float64))
        delta_v_rel_l2 = delta_v_l2 / denom_v

        next_mindim, next_maxdim, next_restarts, budget_tier = _resolve_arnoldi_budget(
            arnoldi_mindim,
            arnoldi_maxdim,
            arnoldi_restarts,
            delta_v_rel_l2,
            adaptive_arnoldi_budget,
            update_low_threshold,
            update_high_threshold,
            low_budget_scale,
            high_budget_scale,
        )

        lmo_init_source = current_arnoldi_initvec === nothing ? "random" : "previous_eigvec"
        w, q, λ, lmo_metrics = ArnoldiGrad(
            A,
            v,
            lowerBound=lowerBound,
            upperBound=upperBound,
            D=D,
            mode=mode,
            tol=10^(εd0),
            returnMetrics=true,
            arnoldi_mindim=next_mindim,
            arnoldi_maxdim=next_maxdim,
            arnoldi_restarts=next_restarts,
            arnoldi_initvec=current_arnoldi_initvec,
        )
        lmo_calls += 1
        total_lmo_ns += lmo_metrics.total_ns
        total_partialschur_ns += lmo_metrics.partialschur_ns
        total_partialeigen_ns += lmo_metrics.partialeigen_ns
        total_mvproducts += lmo_metrics.mvproducts
        gap = dot(q - v, ∇g(v, lowerBound=lowerBound, upperBound=upperBound, D=D)) / abs(f(v))

        delta_lambda_next = abs(λ - prev_λ)
        norm_prev_w = norm(prev_w)
        norm_w = norm(w)
        if norm_prev_w == 0 || norm_w == 0
            cosine_similarity_next = 0.0
        else
            cosine_similarity_next = abs(dot(w, prev_w) / (norm_w * norm_prev_w))
        end
        delta_w_next = min(norm(w - prev_w), norm(w + prev_w))

        push!(delta_v_l2_samples, delta_v_l2)
        push!(next_lmo_time_samples, lmo_metrics.total_ns / 1e9)
        push!(next_lmo_mvproducts_samples, lmo_metrics.mvproducts)
        push!(delta_lambda_samples, delta_lambda_next)

        iter_ns = time_ns() - iter_start_ns
        total_fw_iter_ns += iter_ns

        # Keep gap-only logging for non-benchmark runs.
        if logfilename !== nothing && !benchmark
            open(string(logfilename), "a") do io
                println(io, gap)
            end
        end

        if benchmark && iter_log_path !== nothing
            _append_csv_row(
                iter_log_path,
                [
                    "run_tag", "iteration", "t", "mode", "linesearch", "gamma_source", "gamma",
                    "gap", "epsilon", "epsilon_d0", "iter_time_sec", "lmo_time_sec",
                    "partialschur_time_sec", "partialeigen_time_sec", "lmo_mvproducts",
                    "lmo_converged", "lmo_nconverged", "lmo_nev",
                    "delta_v_l2", "delta_v_linf", "delta_v_rel_l2",
                    "lambda_t", "lambda_next", "delta_lambda_next",
                    "cosine_similarity_next", "delta_w_next",
                    "lmo_init_source",
                    "arnoldi_budget_tier", "arnoldi_mindim", "arnoldi_maxdim", "arnoldi_restarts",
                    # Duplicated aliases for Python plotting without additional joins/renaming.
                    "next_lmo_time_sec", "next_partialschur_time_sec", "next_partialeigen_time_sec",
                    "next_lmo_mvproducts", "next_lmo_converged",
                ],
                [
                    run_tag, fw_iter_idx, t, mode, linesearch, gamma_source, gamma,
                    gap, ε, εd0, iter_ns / 1e9, lmo_metrics.total_ns / 1e9,
                    lmo_metrics.partialschur_ns / 1e9, lmo_metrics.partialeigen_ns / 1e9, lmo_metrics.mvproducts,
                    lmo_metrics.converged, lmo_metrics.nconverged, lmo_metrics.nev,
                    delta_v_l2, delta_v_linf, delta_v_rel_l2,
                    prev_λ, λ, delta_lambda_next,
                    cosine_similarity_next, delta_w_next,
                    lmo_init_source,
                    budget_tier, lmo_metrics.mindim, lmo_metrics.maxdim, lmo_metrics.restarts,
                    lmo_metrics.total_ns / 1e9, lmo_metrics.partialschur_ns / 1e9, lmo_metrics.partialeigen_ns / 1e9,
                    lmo_metrics.mvproducts, lmo_metrics.converged,
                ],
            )
        end

        prev_w = copy(w)
        prev_λ = λ
        if warmstart_reuse_leading_eigvec
            current_arnoldi_initvec = copy(w)
        else
            current_arnoldi_initvec = nothing
        end

        # Optionally tighten Arnoldi tolerance to match optimization progress.
        if tighten_arnoldi_tol && gap < 10^(εd0)
            εd0 -= 1
            println("Change accuracy to ", 10^(εd0))
        end
    end

    # Pick the best rounded cut among sampled Gaussian vectors.
    bestRes = 0
    bestIdx = 1
    for i in 1:numSample
        cut = CutValue(A, z[i, :])
        if cut > bestRes
            bestRes = cut
            bestIdx = i
        end
    end

    solve_total_sec = (time_ns() - solve_start_ns) / 1e9
    avg_fw_iter_sec = fw_iter_idx == 0 ? 0.0 : (total_fw_iter_ns / fw_iter_idx) / 1e9
    avg_lmo_sec = lmo_calls == 0 ? 0.0 : (total_lmo_ns / lmo_calls) / 1e9
    avg_partialschur_sec = lmo_calls == 0 ? 0.0 : (total_partialschur_ns / lmo_calls) / 1e9
    avg_partialeigen_sec = lmo_calls == 0 ? 0.0 : (total_partialeigen_ns / lmo_calls) / 1e9
    lmo_share = solve_total_sec == 0 ? 0.0 : (total_lmo_ns / 1e9) / solve_total_sec
    median_delta_v_l2 = _safe_median(delta_v_l2_samples)
    p90_delta_v_l2 = _safe_quantile(delta_v_l2_samples, 0.9)
    corr_delta_v_to_next_lmo_time = _safe_correlation(delta_v_l2_samples, next_lmo_time_samples)
    corr_delta_v_to_next_mvproducts = _safe_correlation(delta_v_l2_samples, next_lmo_mvproducts_samples)
    corr_delta_v_to_delta_lambda = _safe_correlation(delta_v_l2_samples, delta_lambda_samples)

    bench = (
        run_tag=run_tag,
        solve_total_sec=solve_total_sec,
        fw_iterations=fw_iter_idx,
        avg_fw_iter_sec=avg_fw_iter_sec,
        lmo_calls=lmo_calls,
        total_lmo_sec=total_lmo_ns / 1e9,
        avg_lmo_sec=avg_lmo_sec,
        avg_partialschur_sec=avg_partialschur_sec,
        avg_partialeigen_sec=avg_partialeigen_sec,
        total_mvproducts=total_mvproducts,
        final_gap=gap,
        lmo_share=lmo_share,
        converged=gap <= ε,
        median_delta_v_l2=median_delta_v_l2,
        p90_delta_v_l2=p90_delta_v_l2,
        corr_delta_v_to_next_lmo_time=corr_delta_v_to_next_lmo_time,
        corr_delta_v_to_next_mvproducts=corr_delta_v_to_next_mvproducts,
        corr_delta_v_to_delta_lambda=corr_delta_v_to_delta_lambda,
        adaptive_arnoldi_budget=adaptive_arnoldi_budget,
        arnoldi_mindim=arnoldi_mindim,
        arnoldi_maxdim=arnoldi_maxdim,
        arnoldi_restarts=arnoldi_restarts,
        update_low_threshold=update_low_threshold,
        update_high_threshold=update_high_threshold,
        low_budget_scale=low_budget_scale,
        high_budget_scale=high_budget_scale,
        tighten_arnoldi_tol=tighten_arnoldi_tol,
        warmstart_reuse_leading_eigvec=warmstart_reuse_leading_eigvec,
        warmstart_custom_init_provided=arnoldi_initial_vector !== nothing,
        warmstart_custom_init_accepted=requested_initvec !== nothing,
        first_lmo_init_source=first_lmo_init_source,
    )

    if benchmark && summary_log_path !== nothing
        _append_csv_row(
            summary_log_path,
            [
                "run_tag", "mode", "linesearch", "epsilon", "start_epsilon_d0", "fw_iterations",
                "solve_total_time_sec", "avg_fw_iter_time_sec", "lmo_calls", "total_lmo_time_sec",
                "avg_lmo_time_sec", "avg_partialschur_time_sec", "avg_partialeigen_time_sec",
                "total_mvproducts", "lmo_time_share", "final_gap", "converged",
                "median_delta_v_l2", "p90_delta_v_l2",
                "corr_delta_v_to_next_lmo_time", "corr_delta_v_to_next_mvproducts", "corr_delta_v_to_delta_lambda",
                "adaptive_arnoldi_budget", "arnoldi_mindim", "arnoldi_maxdim", "arnoldi_restarts",
                "update_low_threshold", "update_high_threshold", "low_budget_scale", "high_budget_scale",
                "tighten_arnoldi_tol", "warmstart_reuse_leading_eigvec", "warmstart_custom_init_provided",
                "warmstart_custom_init_accepted", "first_lmo_init_source",
            ],
            [
                run_tag, mode, linesearch, ε, startεd0, fw_iter_idx,
                solve_total_sec, avg_fw_iter_sec, lmo_calls, total_lmo_ns / 1e9,
                avg_lmo_sec, avg_partialschur_sec, avg_partialeigen_sec,
                total_mvproducts, lmo_share, gap, gap <= ε,
                median_delta_v_l2, p90_delta_v_l2,
                corr_delta_v_to_next_lmo_time, corr_delta_v_to_next_mvproducts, corr_delta_v_to_delta_lambda,
                adaptive_arnoldi_budget, arnoldi_mindim, arnoldi_maxdim, arnoldi_restarts,
                update_low_threshold, update_high_threshold, low_budget_scale, high_budget_scale,
                tighten_arnoldi_tol, warmstart_reuse_leading_eigvec, arnoldi_initial_vector !== nothing,
                requested_initvec !== nothing, first_lmo_init_source,
            ],
        )
    end

    return (val=f(v), v=v, t=t, z=z[bestIdx, :], bench=bench)
end

function _ensure_parent_dir(path::AbstractString)
    parent = dirname(path)
    if !isempty(parent)
        mkpath(parent)
    end
end

function _append_csv_row(path::AbstractString, header::Vector{String}, row::Vector)
    _ensure_parent_dir(path)
    write_header = !(isfile(path) && filesize(path) > 0)
    open(path, "a") do io
        if write_header
            println(io, join(header, ","))
        end
        println(io, join(string.(row), ","))
    end
end

function _safe_correlation(x::Vector{Float64}, y::Vector{Float64})
    if length(x) < 2 || length(y) < 2 || length(x) != length(y)
        return 0.0
    end
    if std(x) == 0 || std(y) == 0
        return 0.0
    end
    return cor(x, y)
end

function _safe_median(x::Vector{Float64})
    if isempty(x)
        return 0.0
    end
    return median(x)
end

function _safe_quantile(x::Vector{Float64}, q::Float64)
    if isempty(x)
        return 0.0
    end
    return quantile(x, q)
end

function _resolve_arnoldi_budget(base_mindim::Int, base_maxdim::Int, base_restarts::Int, delta_v_rel_l2::Float64, adaptive::Bool, low_thr::Float64, high_thr::Float64, low_scale::Float64, high_scale::Float64; is_initial=false)
    if is_initial
        return base_mindim, base_maxdim, base_restarts, "init"
    end
    if !adaptive
        return base_mindim, base_maxdim, base_restarts, "fixed"
    end

    if delta_v_rel_l2 <= low_thr
        return _scaled_arnoldi_budget(base_mindim, base_maxdim, base_restarts, low_scale)..., "low"
    elseif delta_v_rel_l2 >= high_thr
        return _scaled_arnoldi_budget(base_mindim, base_maxdim, base_restarts, high_scale)..., "high"
    end
    return base_mindim, base_maxdim, base_restarts, "mid"
end

function _scaled_arnoldi_budget(base_mindim::Int, base_maxdim::Int, base_restarts::Int, scale::Float64)
    s = max(scale, 0.1)
    mindim = max(1, round(Int, base_mindim * s))
    maxdim = max(mindim, round(Int, base_maxdim * s))
    restarts = max(1, round(Int, base_restarts * s))
    return mindim, maxdim, restarts
end