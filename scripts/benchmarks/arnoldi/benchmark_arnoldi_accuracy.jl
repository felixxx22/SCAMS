include("../../../ReadGSet.jl")
include("../../../Arnoldi/ArnoldiMethodMod.jl")

using .ArnoldiMethodMod
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Arpack

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

function _initial_v0_from_incidence(A::SparseMatrixCSC)
    m = size(A, 1)
    n = size(A, 2)
    d = 1 / m
    v0 = zeros(Float64, n)
    rows = rowvals(A)
    vals = nonzeros(A)

    @inbounds for j = 1:n
        s = 0.0
        for k in nzrange(A, j)
            s += vals[k]^2
        end
        v0[j] = d * s
    end

    return v0
end

function _lambda_from_v0_and_Cdiag(v0::AbstractVector, C::SparseMatrixCSC)
    n = length(v0)
    D = zeros(Float64, n)
    sumai = 0.0
    @inbounds for i = 1:n
        D[i] = 2 * C[i, i]
        sumai += D[i]
    end

    lower_bound = 0.0
    upper_bound = sqrt(sumai)
    Mn = lower_bound ./ D
    Mx = upper_bound ./ D

    λ = sqrt.(clamp.(1 ./ (2 .* sqrt.(v0)), Mn, Mx))
    return λ
end

function _build_explicit_operator(Aop::SparseMatrixCSC, λ::AbstractVector; mode::String)
    if mode == "A"
        c = -(λ .^ 2)
        return Aop * spdiagm(0 => c) * Aop'
    elseif mode == "C"
        Dλ = spdiagm(0 => λ)
        return -(Dλ * Aop * Dλ)
    else
        throw(ArgumentError("Unsupported mode '$mode'. Use \"A\" or \"C\"."))
    end
end

function _largest_magnitude_eigenvalue_reference(M::SparseMatrixCSC)
    vals, _ = eigs(M; nev=1, which=:LM, tol=1e-12, maxiter=10_000)
    return vals[1]
end

function run_arnoldi_largest_eig_accuracy_benchmark(; 
    input_file::AbstractString="Gset/g1.txt",
    mode::String="A",
    repeats::Int=5,
    seed::Int=42,
    tols::Vector{Float64}=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    nev::Int=1,
    mindim::Union{Nothing,Int}=nothing,
    maxdim::Union{Nothing,Int}=nothing,
    restarts::Int=500,
    output_prefix::AbstractString="Result/benchmarks/arnoldi_largest_eig"
)
    Random.seed!(seed)

    A_raw, C_raw = readfile(input_file)
    A_scaled = sparse(A_raw / 2)
    C_scaled = sparse(C_raw / 4)

    Aop = mode == "A" ? A_scaled : C_scaled

    v0 = _initial_v0_from_incidence(A_scaled)
    λ = _lambda_from_v0_and_Cdiag(v0, C_scaled)

    dim = size(Aop, 1)
    local_mindim = mindim === nothing ? min(max(12, nev), dim) : min(mindim, dim)
    local_maxdim = maxdim === nothing ? min(max(20, 2 * nev), dim) : min(maxdim, dim)
    if local_maxdim < local_mindim
        local_maxdim = local_mindim
    end

    Mref = _build_explicit_operator(Aop, λ; mode=mode)
    λ_ref = _largest_magnitude_eigenvalue_reference(Mref)

    rows = Vector{Vector}()

    for tol in tols
        for rep in 1:repeats
            t0 = time_ns()
            decomp, history = ArnoldiMethodMod.partialschur(
                Aop,
                λ;
                nev=nev,
                which=ArnoldiMethodMod.LM(),
                tol=tol,
                mindim=local_mindim,
                maxdim=local_maxdim,
                restarts=restarts,
                mode=mode,
            )
            t1 = time_ns()

            eigvals, _ = ArnoldiMethodMod.partialeigen(decomp)
            λ_hat = eigvals[1]

            denom = max(abs(λ_ref), eps(real(float(abs(λ_ref)))))
            rel_err = abs(λ_hat - λ_ref) / denom

            push!(rows, [
                basename(input_file),
                mode,
                tol,
                rep,
                dim,
                history.mvproducts,
                history.restart_iters,
                history.nconverged,
                history.converged,
                (t1 - t0) / 1e9,
                real(λ_ref),
                imag(λ_ref),
                real(λ_hat),
                imag(λ_hat),
                rel_err,
            ])
        end
    end

    raw_csv = string(output_prefix, "_raw.csv")
    _write_csv(
        raw_csv,
        [
            "graph_name",
            "mode",
            "tol",
            "repeat",
            "dimension",
            "mvproducts",
            "restart_iters",
            "nconverged",
            "converged",
            "partialschur_time_sec",
            "lambda_ref_re",
            "lambda_ref_im",
            "lambda_hat_re",
            "lambda_hat_im",
            "rel_error",
        ],
        rows,
    )

    grouped = Dict{Float64, Vector{Vector}}()
    for row in rows
        t = Float64(row[3])
        if !haskey(grouped, t)
            grouped[t] = Vector{Vector}()
        end
        push!(grouped[t], row)
    end

    summary_rows = Vector{Vector}()
    for tol in sort(collect(keys(grouped)))
        rs = grouped[tol]
        mv = [Float64(r[6]) for r in rs]
        iters = [Float64(r[7]) for r in rs]
        tm = [Float64(r[10]) for r in rs]
        err = [Float64(r[15]) for r in rs]
        conv = [r[9] == true ? 1.0 : 0.0 for r in rs]

        push!(summary_rows, [
            basename(input_file),
            mode,
            tol,
            length(rs),
            mean(mv),
            std(mv),
            mean(iters),
            std(iters),
            mean(tm),
            std(tm),
            mean(err),
            std(err),
            mean(conv),
        ])
    end

    summary_csv = string(output_prefix, "_summary.csv")
    _write_csv(
        summary_csv,
        [
            "graph_name",
            "mode",
            "tol",
            "runs",
            "mean_mvproducts",
            "std_mvproducts",
            "mean_restart_iters",
            "std_restart_iters",
            "mean_partialschur_time_sec",
            "std_partialschur_time_sec",
            "mean_rel_error",
            "std_rel_error",
            "converged_rate",
        ],
        summary_rows,
    )

    println("Arnoldi largest-eigenvalue benchmark complete.")
    println("Raw CSV: ", raw_csv)
    println("Summary CSV: ", summary_csv)

    return (raw_csv=raw_csv, summary_csv=summary_csv)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_arnoldi_largest_eig_accuracy_benchmark(
        input_file="Gset/g1.txt",
        mode="A",
        repeats=3,
        seed=42,
        tols=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        nev=1,
        restarts=500,
        output_prefix="Result/benchmarks/arnoldi_largest_eig_g1_A",
    )
end
