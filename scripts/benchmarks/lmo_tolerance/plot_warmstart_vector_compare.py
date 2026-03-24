#!/usr/bin/env python3
"""Plot with/without-vector warm-start comparison curves.

Expected input layout from benchmark_warmstart_vector_compare.jl:
  Result/benchmarks_warmstart_vector_compare/<phase>/
    runs.csv
    Gset/per_graph/*_summary.csv.iters.csv
    BigExample/per_graph/*_summary.csv.iters.csv

Outputs:
  plots/
    graphs/*_gap_vs_iter_strategy.png
    graphs_time/*_gap_vs_time_strategy.png
    *_dataset_gap_vs_iter_strategy.png
    *_dataset_gap_vs_time_strategy.png
    gap_by_graph_strategy_iteration.csv
    gap_by_graph_strategy_time.csv
    gap_by_dataset_strategy_iteration.csv
    gap_by_dataset_strategy_time.csv
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
    strategy: str


@dataclass
class IterPoint:
    dataset: str
    graph_name: str
    strategy: str
    iteration: int
    gap: float


@dataclass
class TimePoint:
    dataset: str
    graph_name: str
    strategy: str
    elapsed_time_sec: float
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
    return name[:-4] if name.endswith(".txt") else name


def _pretty_strategy(strategy: str) -> str:
    if strategy == "with_vector":
        return "with hand vector"
    if strategy == "without_vector":
        return "without hand vector"
    return strategy


def _strategy_color(strategy: str) -> str:
    if strategy == "with_vector":
        return "#1f77b4"
    if strategy == "without_vector":
        return "#d62728"
    return "#2ca02c"


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
                strategy=row.get("strategy", ""),
            )
    return run_map


def _iter_duration_sec(row: Dict[str, str]) -> float:
    t = _to_float(row.get("iter_time_sec", "nan"))
    if math.isnan(t) or t < 0:
        t = _to_float(row.get("lmo_time_sec", "nan"))
    if math.isnan(t) or t < 0:
        return 0.0
    return t


def _load_points(input_root: str, run_map: Dict[str, RunInfo]) -> Tuple[List[IterPoint], List[TimePoint]]:
    iter_points: List[IterPoint] = []
    time_points: List[TimePoint] = []

    iter_paths = sorted(glob.glob(os.path.join(input_root, "*", "per_graph", "*_summary.csv.iters.csv")))
    for path in iter_paths:
        per_run_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("run_tag", "")
                if not tag or tag not in run_map:
                    continue
                per_run_rows[tag].append(row)

        for tag, rows in per_run_rows.items():
            info = run_map[tag]
            rows.sort(key=lambda r: _to_int(r.get("iteration", "0")))
            elapsed = 0.0

            for row in rows:
                iteration = _to_int(row.get("iteration", "0"))
                gap = _to_float(row.get("gap", "nan"))
                if iteration <= 0 or math.isnan(gap):
                    continue

                elapsed += _iter_duration_sec(row)
                g = max(gap, 1e-18)

                iter_points.append(
                    IterPoint(
                        dataset=info.dataset,
                        graph_name=info.graph_name,
                        strategy=info.strategy,
                        iteration=iteration,
                        gap=g,
                    )
                )
                time_points.append(
                    TimePoint(
                        dataset=info.dataset,
                        graph_name=info.graph_name,
                        strategy=info.strategy,
                        elapsed_time_sec=elapsed,
                        gap=g,
                    )
                )

    return iter_points, time_points


def _group_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = mean(values)
    if len(values) <= 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(max(0.0, var))


def _write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _auto_bin_width_time(rows: Sequence[TimePoint]) -> float:
    if not rows:
        return 1.0
    max_t = max(r.elapsed_time_sec for r in rows)
    if max_t <= 0:
        return 1.0
    return max(0.1, min(30.0, max_t / 120.0))


def _time_bin(t: float, width: float) -> float:
    idx = int(math.floor(t / width))
    return (idx + 1) * width


def _aggregate_iter_graph(rows: Sequence[IterPoint]) -> List[List[object]]:
    grouped: Dict[Tuple[str, str, str, int], List[float]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.graph_name, r.strategy, r.iteration)].append(r.gap)

    out: List[List[object]] = []
    for (dataset, graph, strategy, it), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        gmean, gstd = _group_mean_std(vals)
        out.append([dataset, graph, strategy, it, gmean, gstd, len(vals)])
    return out


def _aggregate_iter_dataset(rows: Sequence[IterPoint]) -> List[List[object]]:
    grouped: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.strategy, r.iteration)].append(r.gap)

    out: List[List[object]] = []
    for (dataset, strategy, it), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        gmean, gstd = _group_mean_std(vals)
        out.append([dataset, strategy, it, gmean, gstd, len(vals)])
    return out


def _aggregate_time_graph(rows: Sequence[TimePoint], bin_width_sec: float) -> List[List[object]]:
    grouped: Dict[Tuple[str, str, str, float], List[float]] = defaultdict(list)
    for r in rows:
        tb = _time_bin(r.elapsed_time_sec, bin_width_sec)
        grouped[(r.dataset, r.graph_name, r.strategy, tb)].append(r.gap)

    out: List[List[object]] = []
    for (dataset, graph, strategy, tb), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        gmean, gstd = _group_mean_std(vals)
        out.append([dataset, graph, strategy, tb, gmean, gstd, len(vals)])
    return out


def _aggregate_time_dataset(rows: Sequence[TimePoint], bin_width_sec: float) -> List[List[object]]:
    grouped: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    for r in rows:
        tb = _time_bin(r.elapsed_time_sec, bin_width_sec)
        grouped[(r.dataset, r.strategy, tb)].append(r.gap)

    out: List[List[object]] = []
    for (dataset, strategy, tb), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        gmean, gstd = _group_mean_std(vals)
        out.append([dataset, strategy, tb, gmean, gstd, len(vals)])
    return out


def _plot_per_graph_iteration(rows: Sequence[IterPoint], output_dir: str) -> None:
    grouped: Dict[Tuple[str, str], List[IterPoint]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.graph_name)].append(r)

    graph_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    for (dataset, graph), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        by_key: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for r in items:
            by_key[(r.strategy, r.iteration)].append(r.gap)

        strategies = sorted({k[0] for k in by_key.keys()})
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy in strategies:
            pts = []
            for (s, it), vals in by_key.items():
                if s != strategy:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((it, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.0, label=_pretty_strategy(strategy), color=_strategy_color(strategy))

        ax.set_yscale("log")
        ax.set_title(f"Gap vs Iteration | {dataset} | {graph}")
        ax.set_xlabel("FW iteration")
        ax.set_ylabel("mean gap")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(graph_dir, f"{dataset}_{_graph_stem(graph)}_gap_vs_iter_strategy.png"), dpi=180)
        plt.close(fig)


def _plot_per_graph_time(rows: Sequence[TimePoint], output_dir: str, bin_width_sec: float) -> None:
    grouped: Dict[Tuple[str, str], List[TimePoint]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.graph_name)].append(r)

    graph_dir = os.path.join(output_dir, "graphs_time")
    os.makedirs(graph_dir, exist_ok=True)

    for (dataset, graph), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        by_key: Dict[Tuple[str, float], List[float]] = defaultdict(list)
        for r in items:
            tb = _time_bin(r.elapsed_time_sec, bin_width_sec)
            by_key[(r.strategy, tb)].append(r.gap)

        strategies = sorted({k[0] for k in by_key.keys()})
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy in strategies:
            pts = []
            for (s, tb), vals in by_key.items():
                if s != strategy:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((tb, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.0, label=_pretty_strategy(strategy), color=_strategy_color(strategy))

        ax.set_yscale("log")
        ax.set_title(f"Gap vs Elapsed Time | {dataset} | {graph}")
        ax.set_xlabel("elapsed FW time (sec)")
        ax.set_ylabel("mean gap")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(graph_dir, f"{dataset}_{_graph_stem(graph)}_gap_vs_time_strategy.png"), dpi=180)
        plt.close(fig)


def _plot_dataset_iteration(rows: Sequence[IterPoint], output_dir: str) -> None:
    grouped: Dict[str, List[IterPoint]] = defaultdict(list)
    for r in rows:
        grouped[r.dataset].append(r)

    for dataset, items in sorted(grouped.items()):
        by_key: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for r in items:
            by_key[(r.strategy, r.iteration)].append(r.gap)

        strategies = sorted({k[0] for k in by_key.keys()})
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in strategies:
            pts = []
            for (s, it), vals in by_key.items():
                if s != strategy:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((it, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=2.0, label=_pretty_strategy(strategy), color=_strategy_color(strategy))

        ax.set_yscale("log")
        ax.set_title(f"Dataset Summary Gap vs Iteration | {dataset}")
        ax.set_xlabel("FW iteration")
        ax.set_ylabel("mean gap across selected graphs")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{dataset}_dataset_gap_vs_iter_strategy.png"), dpi=180)
        plt.close(fig)


def _plot_dataset_time(rows: Sequence[TimePoint], output_dir: str, bin_width_sec: float) -> None:
    grouped: Dict[str, List[TimePoint]] = defaultdict(list)
    for r in rows:
        grouped[r.dataset].append(r)

    for dataset, items in sorted(grouped.items()):
        by_key: Dict[Tuple[str, float], List[float]] = defaultdict(list)
        for r in items:
            tb = _time_bin(r.elapsed_time_sec, bin_width_sec)
            by_key[(r.strategy, tb)].append(r.gap)

        strategies = sorted({k[0] for k in by_key.keys()})
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in strategies:
            pts = []
            for (s, tb), vals in by_key.items():
                if s != strategy:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((tb, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=2.0, label=_pretty_strategy(strategy), color=_strategy_color(strategy))

        ax.set_yscale("log")
        ax.set_title(f"Dataset Summary Gap vs Time | {dataset}")
        ax.set_xlabel("elapsed FW time (sec)")
        ax.set_ylabel("mean gap across selected graphs")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{dataset}_dataset_gap_vs_time_strategy.png"), dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot warm-start vector comparison curves (with vs without vector).")
    parser.add_argument("--phase", default="subset", choices=["subset", "full"], help="Benchmark phase directory")
    parser.add_argument("--input-root", default=None, help="Override full input root")
    parser.add_argument("--output", default=None, help="Output directory for plots and aggregate CSVs")
    parser.add_argument("--time-bin-sec", type=float, default=0.0, help="Time bin width in seconds (<=0 uses auto)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    input_root = args.input_root or os.path.join(project_root, "Result", "benchmarks_warmstart_vector_compare", args.phase)
    output_dir = args.output or os.path.join(input_root, "plots")
    os.makedirs(output_dir, exist_ok=True)

    runs_csv = os.path.join(input_root, "runs.csv")
    if not os.path.exists(runs_csv):
        raise RuntimeError(f"Missing runs.csv at: {runs_csv}")

    run_map = _parse_runs(runs_csv)
    iter_points, time_points = _load_points(input_root, run_map)
    if not iter_points:
        raise RuntimeError(f"No iteration rows found under: {input_root}")

    bin_width_sec = args.time_bin_sec if args.time_bin_sec > 0 else _auto_bin_width_time(time_points)

    iter_graph_rows = _aggregate_iter_graph(iter_points)
    iter_dataset_rows = _aggregate_iter_dataset(iter_points)
    time_graph_rows = _aggregate_time_graph(time_points, bin_width_sec)
    time_dataset_rows = _aggregate_time_dataset(time_points, bin_width_sec)

    _write_csv(
        os.path.join(output_dir, "gap_by_graph_strategy_iteration.csv"),
        ["dataset", "graph_name", "strategy", "iteration", "mean_gap", "std_gap", "samples"],
        iter_graph_rows,
    )
    _write_csv(
        os.path.join(output_dir, "gap_by_dataset_strategy_iteration.csv"),
        ["dataset", "strategy", "iteration", "mean_gap", "std_gap", "samples"],
        iter_dataset_rows,
    )
    _write_csv(
        os.path.join(output_dir, "gap_by_graph_strategy_time.csv"),
        ["dataset", "graph_name", "strategy", "elapsed_time_sec_bin", "mean_gap", "std_gap", "samples"],
        time_graph_rows,
    )
    _write_csv(
        os.path.join(output_dir, "gap_by_dataset_strategy_time.csv"),
        ["dataset", "strategy", "elapsed_time_sec_bin", "mean_gap", "std_gap", "samples"],
        time_dataset_rows,
    )

    _plot_per_graph_iteration(iter_points, output_dir)
    _plot_per_graph_time(time_points, output_dir, bin_width_sec)
    _plot_dataset_iteration(iter_points, output_dir)
    _plot_dataset_time(time_points, output_dir, bin_width_sec)

    print("Warm-start comparison plots and aggregates written to:", output_dir)


if __name__ == "__main__":
    main()
