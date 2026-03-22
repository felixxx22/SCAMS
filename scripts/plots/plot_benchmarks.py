#!/usr/bin/env python3
"""Create matplotlib plots from benchmark CSV outputs.

Usage examples:
  python plot_benchmarks.py
  python plot_benchmarks.py --run A:Result/benchmarks_a_scaling/runs.csv --run C:Result/benchmarks_c_scaling/runs.csv
  python plot_benchmarks.py --run Result/benchmarks_a_scaling/runs.csv --output Result/plots

The script expects the runs CSV schema produced by benchmark_a_scaling.jl / benchmark_c_scaling.jl.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class RunRow:
    label: str
    graph_name: str
    n: int
    m: int
    avg_degree: float
    total_wall_time_sec: float
    io_time_sec: float
    solve_time_sec: float
    fw_iterations: float
    total_lmo_time_sec: float
    lmo_share_of_total: float


def _parse_run_spec(spec: str) -> Tuple[str, str]:
    if ":" in spec:
        label, path = spec.split(":", 1)
        label = label.strip()
        path = path.strip()
        if not label:
            label = "run"
        return label, path

    path = spec.strip()
    base = os.path.basename(os.path.normpath(path)).lower()
    parent = os.path.basename(os.path.dirname(path)).lower()
    if "a_scaling" in parent:
        label = "A"
    elif "c_scaling" in parent:
        label = "C"
    elif "a_scaling" in base:
        label = "A"
    elif "c_scaling" in base:
        label = "C"
    else:
        label = os.path.splitext(os.path.basename(path))[0] or "run"
    return label, path


def _to_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_runs_csv(label: str, path: str) -> List[RunRow]:
    rows: List[RunRow] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(_to_float(row.get("n", "nan"), default=float("nan")))
            m = int(_to_float(row.get("m", "nan"), default=float("nan")))
            rows.append(
                RunRow(
                    label=label,
                    graph_name=row.get("graph_name", ""),
                    n=n,
                    m=m,
                    avg_degree=_to_float(row.get("avg_degree", "nan")),
                    total_wall_time_sec=_to_float(row.get("total_wall_time_sec", "nan")),
                    io_time_sec=_to_float(row.get("io_time_sec", "nan")),
                    solve_time_sec=_to_float(row.get("solve_time_sec", "nan")),
                    fw_iterations=_to_float(row.get("fw_iterations", "nan")),
                    total_lmo_time_sec=_to_float(row.get("total_lmo_time_sec", "nan")),
                    lmo_share_of_total=_to_float(row.get("lmo_share_of_total", "nan")),
                )
            )
    return rows


def _group_mean(rows: Sequence[RunRow], key_fn, val_fn) -> Dict[Tuple, float]:
    grouped: Dict[Tuple, List[float]] = defaultdict(list)
    for r in rows:
        value = val_fn(r)
        if math.isnan(value):
            continue
        grouped[key_fn(r)].append(value)
    return {k: mean(vs) for k, vs in grouped.items() if vs}


def _sort_keys_by_m(keys: Iterable[Tuple]) -> List[Tuple]:
    return sorted(keys, key=lambda k: (k[1], k[2] if len(k) > 2 else 0.0, k[0]))


def _plot_total_time_vs_size(rows: Sequence[RunRow], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = _group_mean(
        rows,
        key_fn=lambda r: (r.label, r.m, r.avg_degree),
        val_fn=lambda r: r.total_wall_time_sec,
    )

    by_label: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for (label, m, degree), t in grouped.items():
        by_label[label].append((m, degree, t))

    for label, pts in sorted(by_label.items()):
        for degree in sorted({p[1] for p in pts}):
            line = sorted([(m, t) for m, d, t in pts if d == degree], key=lambda x: x[0])
            if not line:
                continue
            xs = [p[0] for p in line]
            ys = [p[1] for p in line]
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"{label}, d={degree:g}")

    ax.set_title("Total Wall Time vs Number of Edges")
    ax.set_xlabel("m (number of edges)")
    ax.set_ylabel("mean total wall time (sec)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "total_time_vs_size.png"), dpi=180)
    plt.close(fig)


def _plot_lmo_share_vs_size(rows: Sequence[RunRow], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = _group_mean(
        rows,
        key_fn=lambda r: (r.label, r.m, r.avg_degree),
        val_fn=lambda r: r.lmo_share_of_total,
    )

    by_label: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for (label, m, degree), s in grouped.items():
        by_label[label].append((m, degree, s))

    for label, pts in sorted(by_label.items()):
        for degree in sorted({p[1] for p in pts}):
            line = sorted([(m, s) for m, d, s in pts if d == degree], key=lambda x: x[0])
            if not line:
                continue
            xs = [p[0] for p in line]
            ys = [100.0 * p[1] for p in line]
            ax.plot(xs, ys, marker="s", linewidth=2.0, label=f"{label}, d={degree:g}")

    ax.set_title("LMO Share of Total Time vs Number of Edges")
    ax.set_xlabel("m (number of edges)")
    ax.set_ylabel("mean LMO share of total time (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lmo_share_vs_size.png"), dpi=180)
    plt.close(fig)


def _plot_fw_iterations_vs_size(rows: Sequence[RunRow], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = _group_mean(
        rows,
        key_fn=lambda r: (r.label, r.m, r.avg_degree),
        val_fn=lambda r: r.fw_iterations,
    )

    by_label: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for (label, m, degree), fw in grouped.items():
        by_label[label].append((m, degree, fw))

    for label, pts in sorted(by_label.items()):
        for degree in sorted({p[1] for p in pts}):
            line = sorted([(m, fw) for m, d, fw in pts if d == degree], key=lambda x: x[0])
            if not line:
                continue
            xs = [p[0] for p in line]
            ys = [p[1] for p in line]
            ax.plot(xs, ys, marker="^", linewidth=2.0, label=f"{label}, d={degree:g}")

    ax.set_title("FW Iterations vs Number of Edges")
    ax.set_xlabel("m (number of edges)")
    ax.set_ylabel("mean FW iterations")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fw_iterations_vs_size.png"), dpi=180)
    plt.close(fig)


def _plot_time_breakdown(rows: Sequence[RunRow], output_dir: str) -> None:
    labels = sorted({r.label for r in rows})
    if not labels:
        return

    ncols = 1
    nrows = len(labels)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 4 * nrows), squeeze=False)

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r.label, r.m)].append(r)

    for idx, label in enumerate(labels):
        ax = axes[idx][0]
        keys = sorted([k for k in grouped.keys() if k[0] == label], key=lambda x: x[1])
        if not keys:
            ax.set_visible(False)
            continue

        xs = [k[1] for k in keys]
        io_vals = []
        non_lmo_vals = []
        lmo_vals = []

        for k in keys:
            group_rows = grouped[k]
            io_vals.append(mean([r.io_time_sec for r in group_rows]))
            solve_vals = [r.solve_time_sec for r in group_rows]
            lmo_group_vals = [r.total_lmo_time_sec for r in group_rows]
            solve_mean = mean(solve_vals)
            lmo_mean = mean(lmo_group_vals)
            non_lmo_vals.append(max(0.0, solve_mean - lmo_mean))
            lmo_vals.append(lmo_mean)

        ax.bar(xs, io_vals, label="I/O", width=0.18 * max(1, min(xs)) / max(1, len(xs)))
        ax.bar(xs, non_lmo_vals, bottom=io_vals, label="Solver (non-LMO)", width=0.18 * max(1, min(xs)) / max(1, len(xs)))
        bottoms = [a + b for a, b in zip(io_vals, non_lmo_vals)]
        ax.bar(xs, lmo_vals, bottom=bottoms, label="LMO", width=0.18 * max(1, min(xs)) / max(1, len(xs)))

        ax.set_title(f"Time Breakdown vs Number of Edges ({label})")
        ax.set_xlabel("m (number of edges)")
        ax.set_ylabel("mean time (sec)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "time_breakdown_vs_size.png"), dpi=180)
    plt.close(fig)


def _write_summary_csv(rows: Sequence[RunRow], output_dir: str) -> None:
    out_path = os.path.join(output_dir, "plot_summary_by_size.csv")
    grouped: Dict[Tuple[str, int, float], List[RunRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.label, r.m, r.avg_degree)].append(r)

    out_rows: List[List[object]] = []
    for key in _sort_keys_by_m(grouped.keys()):
        label, m, avg_degree = key
        rs = grouped[key]
        out_rows.append(
            [
                label,
                m,
                avg_degree,
                len(rs),
                mean([r.total_wall_time_sec for r in rs]),
                mean([r.io_time_sec for r in rs]),
                mean([r.solve_time_sec for r in rs]),
                mean([r.total_lmo_time_sec for r in rs]),
                mean([r.lmo_share_of_total for r in rs]),
                mean([r.fw_iterations for r in rs]),
            ]
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "m",
                "avg_degree",
                "runs",
                "mean_total_wall_time_sec",
                "mean_io_time_sec",
                "mean_solve_time_sec",
                "mean_total_lmo_time_sec",
                "mean_lmo_share_of_total",
                "mean_fw_iterations",
            ]
        )
        writer.writerows(out_rows)


def _default_run_specs() -> List[str]:
    specs = []
    candidate_a = os.path.join(PROJECT_ROOT, "Result", "benchmarks_a_scaling", "runs.csv")
    candidate_c = os.path.join(PROJECT_ROOT, "Result", "benchmarks_c_scaling", "runs.csv")
    if os.path.exists(candidate_a):
        specs.append(f"A:{candidate_a}")
    if os.path.exists(candidate_c):
        specs.append(f"C:{candidate_c}")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create matplotlib plots from benchmark runs.csv files")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run source in format LABEL:path/to/runs.csv. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "Result", "plots"),
        help="Output directory for plot images and summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_specs = args.run if args.run else _default_run_specs()
    if not run_specs:
        raise SystemExit(
            "No runs CSV provided and no default runs found. "
            "Use --run A:Result/benchmarks_a_scaling/runs.csv (and/or C:...)."
        )

    os.makedirs(args.output, exist_ok=True)

    all_rows: List[RunRow] = []
    for spec in run_specs:
        label, path = _parse_run_spec(spec)
        if not os.path.exists(path):
            raise SystemExit(f"Runs CSV not found: {path}")
        loaded = _load_runs_csv(label, path)
        all_rows.extend(loaded)
        print(f"Loaded {len(loaded)} rows from {path} as label {label}")

    if not all_rows:
        raise SystemExit("No benchmark rows loaded.")

    _plot_total_time_vs_size(all_rows, args.output)
    _plot_lmo_share_vs_size(all_rows, args.output)
    _plot_fw_iterations_vs_size(all_rows, args.output)
    _plot_time_breakdown(all_rows, args.output)
    _write_summary_csv(all_rows, args.output)

    print("Wrote plots and summary to", args.output)
    print(" - total_time_vs_size.png")
    print(" - lmo_share_vs_size.png")
    print(" - fw_iterations_vs_size.png")
    print(" - time_breakdown_vs_size.png")
    print(" - plot_summary_by_size.csv")


if __name__ == "__main__":
    main()
