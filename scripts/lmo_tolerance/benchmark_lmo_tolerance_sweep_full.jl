include("../../benchmark_lmo_tolerance_sweep.jl")

# Full (big) run across all Gset and BigExample graphs.
# Resume mode is enabled so interrupted runs can be restarted safely.
run_lmo_tolerance_asymptotic_sweep(
    phase="full",
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
