#!/usr/bin/env python3
"""Plot warm-start strategy comparison curves.

Expected input layout from benchmark_warmstart_strategy_compare.jl:
  Result/benchmarks_warmstart_strategy_compare/<phase>/
    runs.csv
    Gset/per_graph/*_summary.csv.iters.csv
    BigExample/per_graph/*_summary.csv.iters.csv

Outputs:
  plots/
    graphs_time/*_gap_vs_time_strategy.png
    graphs_mvproducts/*_gap_vs_mvproducts_strategy.png
    *_dataset_gap_vs_time_strategy.png
    *_dataset_gap_vs_mvproducts_strategy.png
    gap_by_graph_strategy_time.csv
    gap_by_graph_strategy_mvproducts.csv
    gap_by_dataset_strategy_time.csv
    gap_by_dataset_strategy_mvproducts.csv
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
class TimePoint:
    dataset: str
    graph_name: str
    strategy: str
    elapsed_time_sec: float
    gap: float


@dataclass
class MVPoint:
    dataset: str
    graph_name: str
    strategy: str
    cumulative_mvproducts: float
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
    if strategy == "random":
        return "random init"
    if strategy == "warmstart_naive":
        return "naive warm-start"
    if strategy == "warmstart_scaled":
        return "scaled warm-start"
    return strategy


def _strategy_color(strategy: str) -> str:
    if strategy == "random":
        return "#d62728"
    if strategy == "warmstart_naive":
        return "#1f77b4"
    if strategy == "warmstart_scaled":
        return "#2ca02c"
    return "#7f7f7f"


def _parse_runs(path: str) -> Dict[str, RunInfo]:
    run_map: Dict[str, RunInfo] = {}
    if not os.path.exists(path):
        return run_map
    try:
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
    except Exception as e:
        print(f"Warning: Could not read runs.csv: {e}")
    return run_map


def _infer_metadata_from_path(file_path: str) -> Tuple[str, str]:
    parts = file_path.replace("\\", "/").split("/")
    dataset = ""
    graph_name = ""
    for part in parts:
        if part in ("Gset", "BigExample"):
            dataset = part
            break
    if parts:
        filename = parts[-1]
        if filename.endswith("_summary.csv.iters.csv"):
            graph_name = filename.replace("_summary.csv.iters.csv", ".txt")
    return dataset, graph_name


def _infer_strategy_from_tag(tag: str) -> str:
    if "warmstart_scaled" in tag:
        return "warmstart_scaled"
    if "warmstart_naive" in tag:
        return "warmstart_naive"
    if "_random_" in tag or tag.endswith("_random"):
        return "random"
    return "unknown"


def _iter_duration_sec(row: Dict[str, str]) -> float:
    t = _to_float(row.get("iter_time_sec", "nan"))
    if math.isnan(t) or t < 0:
        t = _to_float(row.get("lmo_time_sec", "nan"))
    if math.isnan(t) or t < 0:
        return 0.0
    return t


def _iter_mvproducts(row: Dict[str, str]) -> float:
    mvp = _to_float(row.get("lmo_mvproducts", "nan"))
    if math.isnan(mvp) or mvp < 0:
        mvp = _to_float(row.get("next_lmo_mvproducts", "nan"))
    if math.isnan(mvp) or mvp < 0:
        return 0.0
    return mvp


def _load_points(input_root: str, run_map: Dict[str, RunInfo]) -> Tuple[List[TimePoint], List[MVPoint]]:
    time_points: List[TimePoint] = []
    mv_points: List[MVPoint] = []

    iter_paths = sorted(glob.glob(os.path.join(input_root, "*", "per_graph", "*_summary.csv.iters.csv")))
    for path in iter_paths:
        per_run_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("run_tag", "")
                if not tag:
                    continue
                per_run_rows[tag].append(row)

        for tag, rows in per_run_rows.items():
            if tag in run_map:
                info = run_map[tag]
            else:
                dataset, graph_name = _infer_metadata_from_path(path)
                strategy = _infer_strategy_from_tag(tag)
                info = RunInfo(dataset=dataset, graph_name=graph_name, strategy=strategy)
                if not dataset or strategy == "unknown":
                    continue

            rows.sort(key=lambda r: _to_int(r.get("iteration", "0")))
            elapsed = 0.0
            cum_mvproducts = 0.0

            for row in rows:
                iteration = _to_int(row.get("iteration", "0"))
                gap = _to_float(row.get("gap", "nan"))
                if iteration <= 0 or math.isnan(gap):
                    continue

                elapsed += _iter_duration_sec(row)
                cum_mvproducts += _iter_mvproducts(row)
                g = max(gap, 1e-18)

                time_points.append(
                    TimePoint(
                        dataset=info.dataset,
                        graph_name=info.graph_name,
                        strategy=info.strategy,
                        elapsed_time_sec=elapsed,
                        gap=g,
                    )
                )
                mv_points.append(
                    MVPoint(
                        dataset=info.dataset,
                        graph_name=info.graph_name,
                        strategy=info.strategy,
                        cumulative_mvproducts=cum_mvproducts,
                        gap=g,
                    )
                )

    return time_points, mv_points


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


def _auto_bin_width_mvproducts(rows: Sequence[MVPoint]) -> float:
    if not rows:
        return 100.0
    max_mv = max(r.cumulative_mvproducts for r in rows)
    if max_mv <= 0:
        return 100.0
    return max(10.0, min(50000.0, max_mv / 120.0))


def _time_bin(t: float, width: float) -> float:
    idx = int(math.floor(t / width))
    return (idx + 1) * width


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


def _aggregate_mv_graph(rows: Sequence[MVPoint], bin_width_mv: float) -> List[List[object]]:
    grouped: Dict[Tuple[str, str, str, float], List[float]] = defaultdict(list)
    for r in rows:
        mvb = _time_bin(r.cumulative_mvproducts, bin_width_mv)
        grouped[(r.dataset, r.graph_name, r.strategy, mvb)].append(r.gap)

    out: List[List[object]] = []
    for (dataset, graph, strategy, mvb), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        gmean, gstd = _group_mean_std(vals)
        out.append([dataset, graph, strategy, mvb, gmean, gstd, len(vals)])
    return out


def _aggregate_mv_dataset(rows: Sequence[MVPoint], bin_width_mv: float) -> List[List[object]]:
    grouped: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    for r in rows:
        mvb = _time_bin(r.cumulative_mvproducts, bin_width_mv)
        grouped[(r.dataset, r.strategy, mvb)].append(r.gap)

    out: List[List[object]] = []
    for (dataset, strategy, mvb), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        gmean, gstd = _group_mean_std(vals)
        out.append([dataset, strategy, mvb, gmean, gstd, len(vals)])
    return out


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


def _plot_per_graph_mvproducts(rows: Sequence[MVPoint], output_dir: str, bin_width_mv: float) -> None:
    grouped: Dict[Tuple[str, str], List[MVPoint]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.graph_name)].append(r)

    graph_dir = os.path.join(output_dir, "graphs_mvproducts")
    os.makedirs(graph_dir, exist_ok=True)

    for (dataset, graph), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        by_key: Dict[Tuple[str, float], List[float]] = defaultdict(list)
        for r in items:
            mvb = _time_bin(r.cumulative_mvproducts, bin_width_mv)
            by_key[(r.strategy, mvb)].append(r.gap)

        strategies = sorted({k[0] for k in by_key.keys()})
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy in strategies:
            pts = []
            for (s, mvb), vals in by_key.items():
                if s != strategy:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((mvb, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.0, label=_pretty_strategy(strategy), color=_strategy_color(strategy))

        ax.set_yscale("log")
        ax.set_title(f"Gap vs Cumulative MV Products | {dataset} | {graph}")
        ax.set_xlabel("cumulative matrix-vector products")
        ax.set_ylabel("mean gap")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(graph_dir, f"{dataset}_{_graph_stem(graph)}_gap_vs_mvproducts_strategy.png"), dpi=180)
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


def _plot_dataset_mvproducts(rows: Sequence[MVPoint], output_dir: str, bin_width_mv: float) -> None:
    grouped: Dict[str, List[MVPoint]] = defaultdict(list)
    for r in rows:
        grouped[r.dataset].append(r)

    for dataset, items in sorted(grouped.items()):
        by_key: Dict[Tuple[str, float], List[float]] = defaultdict(list)
        for r in items:
            mvb = _time_bin(r.cumulative_mvproducts, bin_width_mv)
            by_key[(r.strategy, mvb)].append(r.gap)

        strategies = sorted({k[0] for k in by_key.keys()})
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in strategies:
            pts = []
            for (s, mvb), vals in by_key.items():
                if s != strategy:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((mvb, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=2.0, label=_pretty_strategy(strategy), color=_strategy_color(strategy))

        ax.set_yscale("log")
        ax.set_title(f"Dataset Summary Gap vs Cumulative MV Products | {dataset}")
        ax.set_xlabel("cumulative matrix-vector products")
        ax.set_ylabel("mean gap across selected graphs")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{dataset}_dataset_gap_vs_mvproducts_strategy.png"), dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot warm-start strategy comparison curves (random vs naive vs scaled).")
    parser.add_argument("--phase", default="subset", choices=["subset", "full"], help="Benchmark phase directory")
    parser.add_argument("--input-root", default=None, help="Override full input root")
    parser.add_argument("--output", default=None, help="Output directory for plots and aggregate CSVs")
    parser.add_argument("--time-bin-sec", type=float, default=0.0, help="Time bin width in seconds (<=0 uses auto)")
    parser.add_argument("--mv-bin", type=float, default=0.0, help="MV-product bin width (<=0 uses auto)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    input_root = args.input_root or os.path.join(project_root, "Result", "benchmarks_warmstart_strategy_compare", args.phase)
    output_dir = args.output or os.path.join(input_root, "plots")
    os.makedirs(output_dir, exist_ok=True)

    runs_csv = os.path.join(input_root, "runs.csv")
    if not os.path.exists(runs_csv):
        print(f"Warning: runs.csv not found at {runs_csv}; will infer metadata from file paths and run tags.")

    run_map = _parse_runs(runs_csv)
    time_points, mv_points = _load_points(input_root, run_map)
    if not time_points:
        raise RuntimeError(f"No iteration rows found under: {input_root}")

    time_bin_sec = args.time_bin_sec if args.time_bin_sec > 0 else _auto_bin_width_time(time_points)
    mv_bin = args.mv_bin if args.mv_bin > 0 else _auto_bin_width_mvproducts(mv_points)

    time_graph_rows = _aggregate_time_graph(time_points, time_bin_sec)
    time_dataset_rows = _aggregate_time_dataset(time_points, time_bin_sec)
    mv_graph_rows = _aggregate_mv_graph(mv_points, mv_bin)
    mv_dataset_rows = _aggregate_mv_dataset(mv_points, mv_bin)

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
    _write_csv(
        os.path.join(output_dir, "gap_by_graph_strategy_mvproducts.csv"),
        ["dataset", "graph_name", "strategy", "cumulative_mvproducts_bin", "mean_gap", "std_gap", "samples"],
        mv_graph_rows,
    )
    _write_csv(
        os.path.join(output_dir, "gap_by_dataset_strategy_mvproducts.csv"),
        ["dataset", "strategy", "cumulative_mvproducts_bin", "mean_gap", "std_gap", "samples"],
        mv_dataset_rows,
    )

    _plot_per_graph_time(time_points, output_dir, time_bin_sec)
    _plot_per_graph_mvproducts(mv_points, output_dir, mv_bin)
    _plot_dataset_time(time_points, output_dir, time_bin_sec)
    _plot_dataset_mvproducts(mv_points, output_dir, mv_bin)

    print("Warm-start strategy comparison plots and aggregates written to:", output_dir)


if __name__ == "__main__":
    main()
