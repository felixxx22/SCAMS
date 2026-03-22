#!/usr/bin/env python3
"""Create C-mode warm-start diagnostic plots from benchmark CSV outputs.

Primary inputs:
- Iteration telemetry CSVs: Result/benchmarks_c_scaling/per_graph/*_summary.csv.iters.csv
- Run summary CSV: Result/benchmarks_c_scaling/runs.csv

Usage examples:
  python plot_c_warmstart.py
  python plot_c_warmstart.py --iters-glob "Result/benchmarks_c_scaling/per_graph/*_summary.csv.iters.csv"
  python plot_c_warmstart.py --runs-csv "Result/benchmarks_c_scaling/runs.csv" --output "Result/plots_c_warmstart"
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class IterRow:
    run_tag: str
    iteration: int
    delta_v_l2: float
    delta_v_rel_l2: float
    next_lmo_time_sec: float
    next_lmo_mvproducts: float
    delta_lambda_next: float
    cosine_similarity_next: float
    delta_w_next: float


@dataclass
class RunRow:
    graph_name: str
    n: int
    m: int
    avg_degree: float
    median_delta_v_l2: float
    p90_delta_v_l2: float
    corr_delta_v_to_next_lmo_time: float
    corr_delta_v_to_next_mvproducts: float
    corr_delta_v_to_delta_lambda: float


def _to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_for_log(values: Iterable[float], floor: float = 1e-18) -> List[float]:
    out = []
    for v in values:
        if math.isnan(v):
            out.append(float("nan"))
        else:
            out.append(max(v, floor))
    return out


def _load_iteration_rows(iters_glob: str) -> List[IterRow]:
    rows: List[IterRow] = []
    paths = sorted(glob.glob(iters_glob))
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(
                    IterRow(
                        run_tag=r.get("run_tag", ""),
                        iteration=_to_int(r.get("iteration", "0")),
                        delta_v_l2=_to_float(r.get("delta_v_l2", "nan")),
                        delta_v_rel_l2=_to_float(r.get("delta_v_rel_l2", "nan")),
                        next_lmo_time_sec=_to_float(r.get("next_lmo_time_sec", "nan")),
                        next_lmo_mvproducts=_to_float(r.get("next_lmo_mvproducts", "nan")),
                        delta_lambda_next=_to_float(r.get("delta_lambda_next", "nan")),
                        cosine_similarity_next=_to_float(r.get("cosine_similarity_next", "nan")),
                        delta_w_next=_to_float(r.get("delta_w_next", "nan")),
                    )
                )
    return rows


def _load_runs_rows(path: str) -> List[RunRow]:
    rows: List[RunRow] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                RunRow(
                    graph_name=r.get("graph_name", ""),
                    n=_to_int(r.get("n", "0")),
                    m=_to_int(r.get("m", "0")),
                    avg_degree=_to_float(r.get("avg_degree", "nan")),
                    median_delta_v_l2=_to_float(r.get("median_delta_v_l2", "nan")),
                    p90_delta_v_l2=_to_float(r.get("p90_delta_v_l2", "nan")),
                    corr_delta_v_to_next_lmo_time=_to_float(r.get("corr_delta_v_to_next_lmo_time", "nan")),
                    corr_delta_v_to_next_mvproducts=_to_float(r.get("corr_delta_v_to_next_mvproducts", "nan")),
                    corr_delta_v_to_delta_lambda=_to_float(r.get("corr_delta_v_to_delta_lambda", "nan")),
                )
            )
    return rows


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    vx = sum(d * d for d in dx)
    vy = sum(d * d for d in dy)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum(a * b for a, b in zip(dx, dy))
    return cov / math.sqrt(vx * vy)


def _scatter_with_fit(
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    x_log: bool = True,
    y_log: bool = True,
) -> None:
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if not pairs:
        return

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, s=12, alpha=0.35, edgecolors="none")

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    r = _pearson(xs, ys)
    ax.set_title(f"{title} (pearson={r:.3f}, n={len(xs)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_binned_median(
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    bins: int = 20,
) -> None:
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b)) and a > 0 and b > 0]
    if len(pairs) < 10:
        return

    pairs.sort(key=lambda p: p[0])
    chunk_size = max(1, len(pairs) // bins)

    x_m = []
    y_m = []
    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i : i + chunk_size]
        if not chunk:
            continue
        x_m.append(median([c[0] for c in chunk]))
        y_m.append(median([c[1] for c in chunk]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_m, y_m, marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_correlation_vs_size(run_rows: Sequence[RunRow], output_dir: str) -> None:
    rows = [r for r in run_rows if not math.isnan(r.corr_delta_v_to_next_lmo_time)]
    if not rows:
        return

    rows.sort(key=lambda r: r.m)
    xs = [r.m for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, [r.corr_delta_v_to_next_lmo_time for r in rows], marker="o", label="corr(delta_v_l2, next_lmo_time)")
    ax.plot(xs, [r.corr_delta_v_to_next_mvproducts for r in rows], marker="s", label="corr(delta_v_l2, next_mvproducts)")
    ax.plot(xs, [r.corr_delta_v_to_delta_lambda for r in rows], marker="^", label="corr(delta_v_l2, delta_lambda)")
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Warm-start Correlations vs Graph Size (C mode)")
    ax.set_xlabel("m (number of edges)")
    ax.set_ylabel("correlation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "corr_vs_size_c_mode.png"), dpi=180)
    plt.close(fig)


def _write_warmstart_summary(iter_rows: Sequence[IterRow], output_dir: str) -> None:
    if not iter_rows:
        return

    x = [r.delta_v_l2 for r in iter_rows]
    y_t = [r.next_lmo_time_sec for r in iter_rows]
    y_m = [r.next_lmo_mvproducts for r in iter_rows]
    y_l = [r.delta_lambda_next for r in iter_rows]
    y_w = [r.delta_w_next for r in iter_rows]

    clean = lambda a, b: [(xv, yv) for xv, yv in zip(a, b) if not (math.isnan(xv) or math.isnan(yv))]

    xt = clean(x, y_t)
    xm = clean(x, y_m)
    xl = clean(x, y_l)
    xw = clean(x, y_w)

    out_path = os.path.join(output_dir, "warmstart_correlation_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "pearson", "samples"])
        writer.writerow([
            "delta_v_l2_vs_next_lmo_time_sec",
            _pearson([p[0] for p in xt], [p[1] for p in xt]) if xt else 0.0,
            len(xt),
        ])
        writer.writerow([
            "delta_v_l2_vs_next_lmo_mvproducts",
            _pearson([p[0] for p in xm], [p[1] for p in xm]) if xm else 0.0,
            len(xm),
        ])
        writer.writerow([
            "delta_v_l2_vs_delta_lambda_next",
            _pearson([p[0] for p in xl], [p[1] for p in xl]) if xl else 0.0,
            len(xl),
        ])
        writer.writerow([
            "delta_v_l2_vs_delta_w_next",
            _pearson([p[0] for p in xw], [p[1] for p in xw]) if xw else 0.0,
            len(xw),
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot C-mode warm-start diagnostics from benchmark CSV outputs")
    parser.add_argument(
        "--iters-glob",
        default=os.path.join(PROJECT_ROOT, "Result", "benchmarks_c_scaling", "per_graph", "*_summary.csv.iters.csv"),
        help="Glob for per-iteration CSV files.",
    )
    parser.add_argument(
        "--runs-csv",
        default=os.path.join(PROJECT_ROOT, "Result", "benchmarks_c_scaling", "runs.csv"),
        help="Path to C-mode runs.csv file.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "Result", "plots_c_warmstart"),
        help="Directory for generated plots and summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    iter_rows = _load_iteration_rows(args.iters_glob)
    run_rows = _load_runs_rows(args.runs_csv)

    print(f"Loaded {len(iter_rows)} iteration rows from {args.iters_glob}")
    print(f"Loaded {len(run_rows)} run rows from {args.runs_csv}")

    if iter_rows:
        x_raw = [r.delta_v_l2 for r in iter_rows]
        x = _safe_for_log(x_raw)

        y_time = _safe_for_log([r.next_lmo_time_sec for r in iter_rows])
        y_mv = _safe_for_log([r.next_lmo_mvproducts for r in iter_rows])
        y_lambda = _safe_for_log([r.delta_lambda_next for r in iter_rows])
        y_dw = _safe_for_log([r.delta_w_next for r in iter_rows])

        _scatter_with_fit(
            x,
            y_time,
            "Delta-v vs Next LMO Time",
            "delta_v_l2",
            "next_lmo_time_sec",
            os.path.join(args.output, "delta_v_vs_next_lmo_time_scatter.png"),
            x_log=True,
            y_log=True,
        )
        _scatter_with_fit(
            x,
            y_mv,
            "Delta-v vs Next LMO MV Products",
            "delta_v_l2",
            "next_lmo_mvproducts",
            os.path.join(args.output, "delta_v_vs_next_lmo_mvproducts_scatter.png"),
            x_log=True,
            y_log=True,
        )
        _scatter_with_fit(
            x,
            y_lambda,
            "Delta-v vs Next Eigenvalue Drift",
            "delta_v_l2",
            "delta_lambda_next",
            os.path.join(args.output, "delta_v_vs_delta_lambda_scatter.png"),
            x_log=True,
            y_log=True,
        )
        _scatter_with_fit(
            x,
            y_dw,
            "Delta-v vs Next Eigenvector Drift",
            "delta_v_l2",
            "delta_w_next",
            os.path.join(args.output, "delta_v_vs_delta_w_scatter.png"),
            x_log=True,
            y_log=True,
        )

        _plot_binned_median(
            x,
            y_time,
            "Binned Median: Delta-v -> Next LMO Time",
            "delta_v_l2 (bin median)",
            "next_lmo_time_sec (bin median)",
            os.path.join(args.output, "delta_v_vs_next_lmo_time_binned_median.png"),
        )
        _plot_binned_median(
            x,
            y_mv,
            "Binned Median: Delta-v -> Next LMO MV Products",
            "delta_v_l2 (bin median)",
            "next_lmo_mvproducts (bin median)",
            os.path.join(args.output, "delta_v_vs_next_lmo_mvproducts_binned_median.png"),
        )
        _plot_binned_median(
            x,
            y_lambda,
            "Binned Median: Delta-v -> Next Eigenvalue Drift",
            "delta_v_l2 (bin median)",
            "delta_lambda_next (bin median)",
            os.path.join(args.output, "delta_v_vs_delta_lambda_binned_median.png"),
        )

        _write_warmstart_summary(iter_rows, args.output)

    if run_rows:
        _plot_correlation_vs_size(run_rows, args.output)

    if not iter_rows and not run_rows:
        raise SystemExit(
            "No input rows loaded. Run the C benchmark first to produce CSV files."
        )

    print("Wrote warm-start plots and summary to", args.output)
    print(" - delta_v_vs_next_lmo_time_scatter.png")
    print(" - delta_v_vs_next_lmo_mvproducts_scatter.png")
    print(" - delta_v_vs_delta_lambda_scatter.png")
    print(" - delta_v_vs_delta_w_scatter.png")
    print(" - delta_v_vs_next_lmo_time_binned_median.png")
    print(" - delta_v_vs_next_lmo_mvproducts_binned_median.png")
    print(" - delta_v_vs_delta_lambda_binned_median.png")
    print(" - corr_vs_size_c_mode.png (if runs.csv has warm-start summary columns)")
    print(" - warmstart_correlation_summary.csv")


if __name__ == "__main__":
    main()
