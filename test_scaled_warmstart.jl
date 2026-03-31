"""
Benchmarking script to test scaled eigenvector warm-start strategy.

Compares three warm-start configurations:
1. No warm-start (baseline)
2. Eigenvector-only reuse (warmstart_reuse_leading_eigvec=true)
3. Scaled eigenvector transform (scaled_warmstart=true)

Runs on small GSET sample graphs and outputs detailed CSV reports.
"""

include("MESDP.jl")
include("ReadGSet.jl")

using Printf
using Statistics
using Dates

# ============================================================================
# Configuration
# ============================================================================
const SAMPLE_GRAPHS = ["g1", "g2", "g3", "g4", "g5"]  # First 5 GSET graphs
const RESULT_DIR = "Result/warmstart_comparison"
const FW_TOLERANCE = 1e-2
const ARNOLDI_TOL = -3.0  # 10^(-3.0) precision
const MODE = "A"
const LINESEARCH = false
const NUM_SAMPLES = 3

# ============================================================================
# Warmstart Configurations
# ============================================================================
const WARMSTART_CONFIGS = [
    (name="baseline", reuse=false, scaled=false),
    (name="eigvec_only", reuse=true, scaled=false),
    (name="scaled", reuse=false, scaled=true),
]

# ============================================================================
# Helper Functions
# ============================================================================

"""Load graph and create initial vector."""
function load_graph_and_init(graph_name::String)
    graph_path = joinpath("Gset", "$(graph_name).txt")
    if !isfile(graph_path)
        error("Graph file not found: $graph_path")
    end
    
    A, C = readfile(graph_path)
    m, n = size(A)
    
    # Initialize v0: uniform scaling of diagonal
    v0 = ones(n)
    
    @printf("Loaded %s: m=%d vertices (edges), n=%d vertices (nodes)\n", graph_name, m, n)
    return A, v0, (m, n)
end

"""Run Solve with specified warm-start configuration."""
function run_solve_config(A, v0, config; benchmarkTag)
    tag = string(benchmarkTag) * "_" * config.name
    logpath = joinpath(RESULT_DIR, "$(tag)_summary.csv")
    
    result = Solve(
        A,
        v0;
        ε=FW_TOLERANCE,
        startεd0=ARNOLDI_TOL,
        mode=MODE,
        linesearch=LINESEARCH,
        numSample=NUM_SAMPLES,
        benchmark=true,
        benchmarkTag=tag,
        logfilename=logpath,
        warmstart_reuse_leading_eigvec=config.reuse,
        scaled_warmstart=config.scaled,
    )
    
    return result
end

"""Format configuration summary."""
function format_config_summary(config, result)
    b = result.bench
    return (
        config=config.name,
        iterations=b.fw_iterations,
        lmo_calls=b.lmo_calls,
        total_time_sec=b.solve_total_sec,
        avg_lmo_sec=b.avg_lmo_sec,
        final_gap=b.final_gap,
        converged=b.converged,
        median_D_change=b.median_warmstart_D_change,
        median_x0_norm=b.median_warmstart_x0_norm,
    )
end

"""Print comparison table."""
function print_comparison_table(graph_name, config_results)
    println("\n" * "="^80)
    println("WARMSTART COMPARISON: $graph_name")
    println("="^80)
    
    # Header
    @printf("%15s | %10s | %10s | %10s | %12s | %12s | %12s\n", 
        "Configuration", "Iterations", "LMO Calls", "Time (s)", "Avg LMO (s)", "Gap", "Converged")
    println("-"^80)
    
    # Rows
    for summary in config_results
        @printf("%15s | %10d | %10d | %10.4f | %12.6f | %12.2e | %12s\n",
            summary.config,
            summary.iterations,
            summary.lmo_calls,
            summary.total_time_sec,
            summary.avg_lmo_sec,
            summary.final_gap,
            (summary.converged ? "YES" : "NO"))
    end
    
    # Speedup analysis
    if length(config_results) >= 2
        baseline_time = config_results[1].total_time_sec
        baseline_iters = config_results[1].iterations
        
        println("\n" * "SPEEDUP vs BASELINE (config 1):")
        for i in 2:length(config_results)
            summary = config_results[i]
            speedup = baseline_time / max(summary.total_time_sec, 1e-6)
            iter_reduction = 100 * (1 - summary.iterations / max(baseline_iters, 1))
            @printf("  %s: %.2fx time | %.1f%% fewer iters | median(D_change)=%.2e | median(x0_norm)=%.4f\n",
                summary.config,
                speedup,
                iter_reduction,
                summary.median_D_change,
                summary.median_x0_norm)
        end
    end
end

"""Ensure output directory exists."""
function ensure_output_dir()
    mkpath(RESULT_DIR)
    @printf("Output directory: %s\n", realpath(RESULT_DIR))
end

# ============================================================================
# Main Benchmark Suite
# ============================================================================

function run_warmstart_benchmark_suite()
    ensure_output_dir()
    
    overall_results = Dict()
    
    for graph_name in SAMPLE_GRAPHS
        @printf("\n┌─ Testing graph: %s\n", graph_name)
        
        # Load graph
        A, v0, dims = load_graph_and_init(graph_name)
        
        # Create timestamp for this graph's runs
        run_timestamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
        
        # Run all configurations
        config_results = []
        for config in WARMSTART_CONFIGS
            @printf("  ├─ Running config: %s...\n", config.name)
            try
                result = run_solve_config(A, v0, config; benchmarkTag="$(graph_name)_$(run_timestamp)")
                summary = format_config_summary(config, result)
                push!(config_results, summary)
                @printf("  │  ✓ Completed: %d iterations, %.4f s\n", summary.iterations, summary.total_time_sec)
            catch e
                @printf("  │  ✗ Failed: %s\n", string(e))
                push!(config_results, (
                    config=config.name, iterations=-1, lmo_calls=-1, total_time_sec=-1,
                    avg_lmo_sec=-1, final_gap=-1, converged=false,
                    median_D_change=0.0, median_x0_norm=0.0
                ))
            end
        end
        
        # Display results for this graph
        print_comparison_table(graph_name, config_results)
        overall_results[graph_name] = config_results
    end
    
    println("\n" * "="^80)
    println("BENCHMARK COMPLETE")
    println("="^80)
    @printf("Results saved to: %s\n", realpath(RESULT_DIR))
    println("\nPer-graph CSVs:")
    for graph_name in SAMPLE_GRAPHS
        for config in WARMSTART_CONFIGS
            pattern = joinpath(RESULT_DIR, "$(graph_name)_*_$(config.name)_summary.csv*")
            println("  - $pattern")
        end
    end
    
    return overall_results
end

# ============================================================================
# Entry Point
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    @printf("Julia Scaled Warmstart Benchmarking Suite\n")
    @printf("========================================\n")
    @printf("Sample graphs: %s\n", join(SAMPLE_GRAPHS, ", "))
    @printf("FW tolerance: %.2e\n", FW_TOLERANCE)
    @printf("Mode: %s\n", MODE)
    @printf("Warmstart configs: %s\n", join([c.name for c in WARMSTART_CONFIGS], ", "))
    
    results = run_warmstart_benchmark_suite()
end
