#!/usr/bin/env python3
"""Plot gap-vs-time asymptotic curves for the LMO tolerance sweep.

Expected input layout from benchmark_lmo_tolerance_sweep.jl:
  Result/benchmarks_lmo_tolerance/<phase>/
    runs.csv
    Gset/per_graph/*_summary.csv.iters.csv
    BigExample/per_graph/*_summary.csv.iters.csv

Usage examples:
  python plot_lmo_tolerance_asymptotics_time.py
  python plot_lmo_tolerance_asymptotics_time.py --phase full
  python plot_lmo_tolerance_asymptotics_time.py --input-root local_results --output local_results/plots_time
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass
class RunInfo:
    dataset: str
    graph_name: str
    start_epsilon_d0: float


@dataclass
class TimeRow:
    dataset: str
    graph_name: str
    start_epsilon_d0: float
    elapsed_time_sec: float
    gap: float


@dataclass
class LmoTimeRow:
    dataset: str
    graph_name: str
    start_epsilon_d0: float
    lmo_time_sec: float
    gap: float


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


def _graph_stem(name: str) -> str:
    if name.endswith(".txt"):
        return name[:-4]
    return name


def _parse_runs(path: str) -> Dict[str, RunInfo]:
    run_map: Dict[str, RunInfo] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("run_tag", "")
            if not tag:
                continue
            run_map[tag] = RunInfo(
                dataset=row.get("dataset", ""),
                graph_name=row.get("graph_name", ""),
                start_epsilon_d0=_to_float(row.get("start_epsilon_d0", "nan")),
            )
    return run_map


def _infer_graph_name_from_iter_path(path: str) -> str:
    base = os.path.basename(path)
    suffix = "_summary.csv.iters.csv"
    if base.endswith(suffix):
        return base[: -len(suffix)] + ".txt"
    return base


def _infer_dataset_from_iter_path(input_root: str, path: str) -> str:
    rel = os.path.relpath(path, input_root)
    parts = rel.split(os.sep)
    if len(parts) >= 3:
        return parts[0]
    return "unknown"


def _iter_duration_sec(row: Dict[str, str]) -> float:
    # Prefer full FW iteration time, fallback to LMO-only time if needed.
    t = _to_float(row.get("iter_time_sec", "nan"))
    if math.isnan(t) or t < 0:
        t = _to_float(row.get("lmo_time_sec", "nan"))
    if math.isnan(t) or t < 0:
        return 0.0
    return t


def _lmo_time_sec(row: Dict[str, str]) -> float:
    """Extract LMO-only time from iteration row."""
    t = _to_float(row.get("lmo_time_sec", "nan"))
    if math.isnan(t) or t < 0:
        return 0.0
    return t


def _load_time_rows(input_root: str, run_map: Dict[str, RunInfo] | None = None) -> List[TimeRow]:
    rows: List[TimeRow] = []
    iter_paths = sorted(glob.glob(os.path.join(input_root, "*", "per_graph", "*_summary.csv.iters.csv")))

    for path in iter_paths:
        per_run: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("run_tag", "")
                if not tag:
                    continue
                per_run[tag].append(row)

        for tag, run_rows in per_run.items():
            info = run_map.get(tag) if run_map else None
            run_rows.sort(key=lambda r: _to_int(r.get("iteration", "0")))
            elapsed = 0.0

            for row in run_rows:
                gap = _to_float(row.get("gap", "nan"))
                if math.isnan(gap):
                    continue

                elapsed += _iter_duration_sec(row)

                if info is not None:
                    dataset = info.dataset
                    graph_name = info.graph_name
                    start_epsilon_d0 = info.start_epsilon_d0
                else:
                    dataset = _infer_dataset_from_iter_path(input_root, path)
                    graph_name = _infer_graph_name_from_iter_path(path)
                    start_epsilon_d0 = _to_float(row.get("epsilon_d0", "nan"))
                    if math.isnan(start_epsilon_d0):
                        continue

                rows.append(
                    TimeRow(
                        dataset=dataset,
                        graph_name=graph_name,
                        start_epsilon_d0=start_epsilon_d0,
                        elapsed_time_sec=elapsed,
                        gap=max(gap, 1e-18),
                    )
                )

    return rows


def _load_lmo_time_rows(input_root: str, run_map: Dict[str, RunInfo] | None = None) -> List[LmoTimeRow]:
    """Load per-iteration LMO times and gaps for plotting LMO time vs gap."""
    rows: List[LmoTimeRow] = []
    iter_paths = sorted(glob.glob(os.path.join(input_root, "*", "per_graph", "*_summary.csv.iters.csv")))

    for path in iter_paths:
        per_run: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("run_tag", "")
                if not tag:
                    continue
                per_run[tag].append(row)

        for tag, run_rows in per_run.items():
            info = run_map.get(tag) if run_map else None
            run_rows.sort(key=lambda r: _to_int(r.get("iteration", "0")))

            for row in run_rows:
                gap = _to_float(row.get("gap", "nan"))
                lmo_time = _lmo_time_sec(row)
                if math.isnan(gap) or lmo_time <= 0:
                    continue

                if info is not None:
                    dataset = info.dataset
                    graph_name = info.graph_name
                    start_epsilon_d0 = info.start_epsilon_d0
                else:
                    dataset = _infer_dataset_from_iter_path(input_root, path)
                    graph_name = _infer_graph_name_from_iter_path(path)
                    start_epsilon_d0 = _to_float(row.get("epsilon_d0", "nan"))
                    if math.isnan(start_epsilon_d0):
                        continue

                rows.append(
                    LmoTimeRow(
                        dataset=dataset,
                        graph_name=graph_name,
                        start_epsilon_d0=start_epsilon_d0,
                        lmo_time_sec=lmo_time,
                        gap=max(gap, 1e-18),
                    )
                )

    return rows


def _group_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = mean(values)
    if len(values) <= 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(max(0.0, var))


def _auto_bin_width(rows: Sequence[TimeRow]) -> float:
    if not rows:
        return 1.0
    max_t = max(r.elapsed_time_sec for r in rows)
    if max_t <= 0:
        return 1.0
    # Aim for ~150 points while keeping bins practical.
    return max(0.1, min(30.0, max_t / 150.0))


def _time_bin(t: float, width: float) -> float:
    idx = int(math.floor(t / width))
    return (idx + 1) * width


def _gap_log_bin(gap: float, num_bins: int = 50) -> float:
    """Bin gaps logarithmically. Returns the upper bound of the bin."""
    if gap <= 0:
        return 1e-18
    log_gap = math.log10(gap)
    # Find min and max log scale across a reasonable range
    log_min = -18.0
    log_max = 2.0
    bin_width = (log_max - log_min) / num_bins
    bin_idx = int(math.floor((log_gap - log_min) / bin_width))
    bin_idx = max(0, min(num_bins - 1, bin_idx))
    return 10.0 ** (log_min + (bin_idx + 1) * bin_width)


def _aggregate_graph_lmo_gap(rows: Sequence[LmoTimeRow], num_gap_bins: int = 50) -> Dict[Tuple[str, str, float, float], Tuple[float, float, int]]:
    grouped: Dict[Tuple[str, str, float, float], List[float]] = defaultdict(list)
    for r in rows:
        gap_bin = _gap_log_bin(r.gap, num_gap_bins)
        grouped[(r.dataset, r.graph_name, r.start_epsilon_d0, gap_bin)].append(r.lmo_time_sec)

    out: Dict[Tuple[str, str, float, float], Tuple[float, float, int]] = {}
    for key, vals in grouped.items():
        tmean, tstd = _group_mean_std(vals)
        out[key] = (tmean, tstd, len(vals))
    return out


def _aggregate_dataset_lmo_gap(rows: Sequence[LmoTimeRow], num_gap_bins: int = 50) -> Dict[Tuple[str, float, float], Tuple[float, float, int]]:
    grouped: Dict[Tuple[str, float, float], List[float]] = defaultdict(list)
    for r in rows:
        gap_bin = _gap_log_bin(r.gap, num_gap_bins)
        grouped[(r.dataset, r.start_epsilon_d0, gap_bin)].append(r.lmo_time_sec)

    out: Dict[Tuple[str, float, float], Tuple[float, float, int]] = {}
    for key, vals in grouped.items():
        tmean, tstd = _group_mean_std(vals)
        out[key] = (tmean, tstd, len(vals))
    return out


def _aggregate_graph(rows: Sequence[TimeRow], bin_width_sec: float) -> Dict[Tuple[str, str, float, float], Tuple[float, float, int]]:
    grouped: Dict[Tuple[str, str, float, float], List[float]] = defaultdict(list)
    for r in rows:
        t_bin = _time_bin(r.elapsed_time_sec, bin_width_sec)
        grouped[(r.dataset, r.graph_name, r.start_epsilon_d0, t_bin)].append(r.gap)

    out: Dict[Tuple[str, str, float, float], Tuple[float, float, int]] = {}
    for key, vals in grouped.items():
        gmean, gstd = _group_mean_std(vals)
        out[key] = (gmean, gstd, len(vals))
    return out


def _aggregate_dataset(rows: Sequence[TimeRow], bin_width_sec: float) -> Dict[Tuple[str, float, float], Tuple[float, float, int]]:
    grouped: Dict[Tuple[str, float, float], List[float]] = defaultdict(list)
    for r in rows:
        t_bin = _time_bin(r.elapsed_time_sec, bin_width_sec)
        grouped[(r.dataset, r.start_epsilon_d0, t_bin)].append(r.gap)

    out: Dict[Tuple[str, float, float], Tuple[float, float, int]] = {}
    for key, vals in grouped.items():
        gmean, gstd = _group_mean_std(vals)
        out[key] = (gmean, gstd, len(vals))
    return out


def _write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _plot_per_graph(rows: Sequence[TimeRow], output_dir: str, bin_width_sec: float) -> None:
    grouped: Dict[Tuple[str, str], List[TimeRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.graph_name)].append(r)

    graph_dir = os.path.join(output_dir, "graphs_time")
    os.makedirs(graph_dir, exist_ok=True)

    for (dataset, graph_name), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        by_tol_time: Dict[Tuple[float, float], List[float]] = defaultdict(list)
        for r in items:
            t_bin = _time_bin(r.elapsed_time_sec, bin_width_sec)
            by_tol_time[(r.start_epsilon_d0, t_bin)].append(r.gap)

        tols = sorted({k[0] for k in by_tol_time.keys()})

        fig, ax = plt.subplots(figsize=(10, 6))
        for d0 in tols:
            pts = []
            for (tol, t_bin), vals in by_tol_time.items():
                if tol != d0:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((t_bin, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.0, label=f"startεd0={d0:g} (tol=1e{d0:g})")

        ax.set_yscale("log")
        ax.set_title(f"Gap vs Elapsed Time | {dataset} | {graph_name}")
        ax.set_xlabel("elapsed FW time (sec)")
        ax.set_ylabel("mean gap")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        stem = _graph_stem(graph_name)
        out_path = os.path.join(graph_dir, f"{dataset}_{stem}_gap_vs_time.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def _plot_dataset_summary(rows: Sequence[TimeRow], output_dir: str, bin_width_sec: float) -> None:
    grouped: Dict[str, List[TimeRow]] = defaultdict(list)
    for r in rows:
        grouped[r.dataset].append(r)

    for dataset, items in sorted(grouped.items()):
        by_tol_time: Dict[Tuple[float, float], List[float]] = defaultdict(list)
        for r in items:
            t_bin = _time_bin(r.elapsed_time_sec, bin_width_sec)
            by_tol_time[(r.start_epsilon_d0, t_bin)].append(r.gap)

        tols = sorted({k[0] for k in by_tol_time.keys()})
        fig, ax = plt.subplots(figsize=(10, 6))

        for d0 in tols:
            pts = []
            for (tol, t_bin), vals in by_tol_time.items():
                if tol != d0:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((t_bin, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.0, label=f"startεd0={d0:g} (tol=1e{d0:g})")

        ax.set_yscale("log")
        ax.set_title(f"Dataset Summary Gap vs Elapsed Time | {dataset}")
        ax.set_xlabel("elapsed FW time (sec)")
        ax.set_ylabel("mean gap across all selected graphs")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{dataset}_dataset_gap_vs_time.png"), dpi=180)
        plt.close(fig)


def _plot_per_graph_lmo_gap(rows: Sequence[LmoTimeRow], output_dir: str, num_gap_bins: int = 50) -> None:
    grouped: Dict[Tuple[str, str], List[LmoTimeRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.graph_name)].append(r)

    graph_dir = os.path.join(output_dir, "graphs_lmo_gap")
    os.makedirs(graph_dir, exist_ok=True)

    for (dataset, graph_name), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        by_tol_gap: Dict[Tuple[float, float], List[float]] = defaultdict(list)
        for r in items:
            gap_bin = _gap_log_bin(r.gap, num_gap_bins)
            by_tol_gap[(r.start_epsilon_d0, gap_bin)].append(r.lmo_time_sec)

        tols = sorted({k[0] for k in by_tol_gap.keys()})
        stem = _graph_stem(graph_name)

        fig, ax = plt.subplots(figsize=(10, 6))
        for d0 in tols:
            pts = []
            for (tol, gap_bin), vals in by_tol_gap.items():
                if tol != d0:
                    continue
                tmean, _ = _group_mean_std(vals)
                pts.append((gap_bin, tmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.0, marker='o', label=f"tol=1e{d0:g}")

        ax.set_xscale("log")
        ax.set_title(f"LMO Time vs Gap Distance | {dataset} | {graph_name}")
        ax.set_xlabel("gap (log scale)")
        ax.set_ylabel("mean LMO time (sec)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(graph_dir, f"{dataset}_{stem}_lmo_time_vs_gap.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def _plot_dataset_summary_lmo_gap(rows: Sequence[LmoTimeRow], output_dir: str, num_gap_bins: int = 50) -> None:
    grouped: Dict[str, List[LmoTimeRow]] = defaultdict(list)
    for r in rows:
        grouped[r.dataset].append(r)

    for dataset, items in sorted(grouped.items()):
        by_tol_gap: Dict[Tuple[float, float], List[float]] = defaultdict(list)
        for r in items:
            gap_bin = _gap_log_bin(r.gap, num_gap_bins)
            by_tol_gap[(r.start_epsilon_d0, gap_bin)].append(r.lmo_time_sec)

        tols = sorted({k[0] for k in by_tol_gap.keys()})
        fig, ax = plt.subplots(figsize=(10, 6))

        for d0 in tols:
            pts = []
            for (tol, gap_bin), vals in by_tol_gap.items():
                if tol != d0:
                    continue
                tmean, _ = _group_mean_std(vals)
                pts.append((gap_bin, tmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.0, marker='o', label=f"tol=1e{d0:g}")

        ax.set_xscale("log")
        ax.set_title(f"Dataset Summary LMO Time vs Gap Distance | {dataset}")
        ax.set_xlabel("gap (log scale)")
        ax.set_ylabel("mean LMO time (sec) across all graphs")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{dataset}_dataset_lmo_time_vs_gap.png"), dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gap-vs-time asymptotic curves for LMO tolerance sweeps.")
    parser.add_argument("--phase", default="subset", choices=["subset", "full"], help="Sweep phase directory under Result/benchmarks_lmo_tolerance")
    parser.add_argument("--input-root", default=None, help="Override full input root directory (if unset: Result/benchmarks_lmo_tolerance/<phase>)")
    parser.add_argument("--output", default=None, help="Output directory for plots and aggregated CSVs")
    parser.add_argument("--time-bin-sec", type=float, default=0.0, help="Time bin width in seconds (<=0 means auto)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    input_root = args.input_root or os.path.join(project_root, "Result", "benchmarks_lmo_tolerance", args.phase)
    output_dir = args.output or os.path.join(input_root, "plots_time")
    os.makedirs(output_dir, exist_ok=True)

    runs_csv = os.path.join(input_root, "runs.csv")
    run_map: Dict[str, RunInfo] | None = None
    if os.path.exists(runs_csv):
        run_map = _parse_runs(runs_csv)
        print("Loaded run map from:", runs_csv)
    else:
        print("runs.csv not found, falling back to per-iteration metadata from .iters.csv files")

    time_rows = _load_time_rows(input_root, run_map)
    if not time_rows:
        raise RuntimeError(f"No iteration rows found under: {input_root}")

    bin_width_sec = args.time_bin_sec if args.time_bin_sec > 0 else _auto_bin_width(time_rows)
    print(f"Using time bin width: {bin_width_sec:.3f} sec")

    graph_agg = _aggregate_graph(time_rows, bin_width_sec)
    graph_rows = []
    for (dataset, graph_name, d0, t_bin), (gmean, gstd, count) in sorted(graph_agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        graph_rows.append([dataset, graph_name, d0, 10.0**d0, t_bin, gmean, gstd, count])

    dataset_agg = _aggregate_dataset(time_rows, bin_width_sec)
    dataset_rows = []
    for (dataset, d0, t_bin), (gmean, gstd, count) in sorted(dataset_agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        dataset_rows.append([dataset, d0, 10.0**d0, t_bin, gmean, gstd, count])

    _write_csv(
        os.path.join(output_dir, "gap_by_graph_tolerance_time.csv"),
        ["dataset", "graph_name", "start_epsilon_d0", "lmo_tolerance", "elapsed_time_sec_bin", "mean_gap", "std_gap", "samples"],
        graph_rows,
    )

    _write_csv(
        os.path.join(output_dir, "gap_by_dataset_tolerance_time.csv"),
        ["dataset", "start_epsilon_d0", "lmo_tolerance", "elapsed_time_sec_bin", "mean_gap", "std_gap", "samples"],
        dataset_rows,
    )

    _plot_per_graph(time_rows, output_dir, bin_width_sec)
    _plot_dataset_summary(time_rows, output_dir, bin_width_sec)

    # Load and process LMO time vs gap data
    lmo_rows = _load_lmo_time_rows(input_root, run_map)
    if lmo_rows:
        print(f"Loaded {len(lmo_rows)} LMO time measurements")

        lmo_graph_agg = _aggregate_graph_lmo_gap(lmo_rows, num_gap_bins=50)
        lmo_graph_rows = []
        for (dataset, graph_name, d0, gap_bin), (tmean, tstd, count) in sorted(lmo_graph_agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
            lmo_graph_rows.append([dataset, graph_name, d0, 10.0**d0, gap_bin, tmean, tstd, count])

        lmo_dataset_agg = _aggregate_dataset_lmo_gap(lmo_rows, num_gap_bins=50)
        lmo_dataset_rows = []
        for (dataset, d0, gap_bin), (tmean, tstd, count) in sorted(lmo_dataset_agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            lmo_dataset_rows.append([dataset, d0, 10.0**d0, gap_bin, tmean, tstd, count])

        _write_csv(
            os.path.join(output_dir, "lmo_time_by_graph_tolerance_gap.csv"),
            ["dataset", "graph_name", "start_epsilon_d0", "lmo_tolerance", "gap_bin", "mean_lmo_time_sec", "std_lmo_time_sec", "samples"],
            lmo_graph_rows,
        )

        _write_csv(
            os.path.join(output_dir, "lmo_time_by_dataset_tolerance_gap.csv"),
            ["dataset", "start_epsilon_d0", "lmo_tolerance", "gap_bin", "mean_lmo_time_sec", "std_lmo_time_sec", "samples"],
            lmo_dataset_rows,
        )

        _plot_per_graph_lmo_gap(lmo_rows, output_dir, num_gap_bins=50)
        _plot_dataset_summary_lmo_gap(lmo_rows, output_dir, num_gap_bins=50)

    print("Plots and aggregate CSVs written to:", output_dir)


if __name__ == "__main__":
    main()
