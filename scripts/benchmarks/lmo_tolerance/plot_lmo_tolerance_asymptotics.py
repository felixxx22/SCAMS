#!/usr/bin/env python3
"""Plot gap-vs-iteration asymptotic curves for the LMO tolerance sweep.

Expected input layout from benchmark_lmo_tolerance_sweep.jl:
  Result/benchmarks_lmo_tolerance/<phase>/
    runs.csv
    Gset/per_graph/*_summary.csv.iters.csv
    BigExample/per_graph/*_summary.csv.iters.csv

Usage examples:
  python plot_lmo_tolerance_asymptotics.py
  python plot_lmo_tolerance_asymptotics.py --phase full
  python plot_lmo_tolerance_asymptotics.py --input-root Result/benchmarks_lmo_tolerance/subset
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
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass
class RunInfo:
    dataset: str
    graph_name: str
    start_epsilon_d0: float


@dataclass
class IterRow:
    dataset: str
    graph_name: str
    start_epsilon_d0: float
    iteration: int
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


def _load_iteration_rows(input_root: str, run_map: Dict[str, RunInfo] | None = None) -> List[IterRow]:
    rows: List[IterRow] = []
    iter_paths = sorted(glob.glob(os.path.join(input_root, "*", "per_graph", "*_summary.csv.iters.csv")))

    for path in iter_paths:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("run_tag", "")
                info = run_map.get(tag) if run_map else None
                iteration = _to_int(row.get("iteration", "0"))
                gap = _to_float(row.get("gap", "nan"))
                if iteration <= 0 or math.isnan(gap):
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
                    IterRow(
                        dataset=dataset,
                        graph_name=graph_name,
                        start_epsilon_d0=start_epsilon_d0,
                        iteration=iteration,
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


def _aggregate_graph(rows: Sequence[IterRow]) -> Dict[Tuple[str, str, float, int], Tuple[float, float, int]]:
    grouped: Dict[Tuple[str, str, float, int], List[float]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.graph_name, r.start_epsilon_d0, r.iteration)].append(r.gap)

    out: Dict[Tuple[str, str, float, int], Tuple[float, float, int]] = {}
    for key, vals in grouped.items():
        gmean, gstd = _group_mean_std(vals)
        out[key] = (gmean, gstd, len(vals))
    return out


def _aggregate_dataset(rows: Sequence[IterRow]) -> Dict[Tuple[str, float, int], Tuple[float, float, int]]:
    grouped: Dict[Tuple[str, float, int], List[float]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.start_epsilon_d0, r.iteration)].append(r.gap)

    out: Dict[Tuple[str, float, int], Tuple[float, float, int]] = {}
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


def _plot_per_graph(rows: Sequence[IterRow], output_dir: str) -> None:
    grouped: Dict[Tuple[str, str], List[IterRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.dataset, r.graph_name)].append(r)

    graph_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    for (dataset, graph_name), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        by_tol_iter: Dict[Tuple[float, int], List[float]] = defaultdict(list)
        for r in items:
            by_tol_iter[(r.start_epsilon_d0, r.iteration)].append(r.gap)

        tols = sorted({k[0] for k in by_tol_iter.keys()})

        fig, ax = plt.subplots(figsize=(10, 6))
        for d0 in tols:
            pts = []
            for (tol, it), vals in by_tol_iter.items():
                if tol != d0:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((it, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.0, label=f"startεd0={d0:g} (tol=1e{d0:g})")

        ax.set_yscale("log")
        ax.set_title(f"Gap vs Iteration | {dataset} | {graph_name}")
        ax.set_xlabel("FW iteration")
        ax.set_ylabel("mean gap")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        stem = _graph_stem(graph_name)
        out_path = os.path.join(graph_dir, f"{dataset}_{stem}_gap_vs_iter.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def _plot_dataset_summary(rows: Sequence[IterRow], output_dir: str) -> None:
    grouped: Dict[str, List[IterRow]] = defaultdict(list)
    for r in rows:
        grouped[r.dataset].append(r)

    for dataset, items in sorted(grouped.items()):
        by_tol_iter: Dict[Tuple[float, int], List[float]] = defaultdict(list)
        for r in items:
            by_tol_iter[(r.start_epsilon_d0, r.iteration)].append(r.gap)

        tols = sorted({k[0] for k in by_tol_iter.keys()})
        fig, ax = plt.subplots(figsize=(10, 6))

        for d0 in tols:
            pts = []
            for (tol, it), vals in by_tol_iter.items():
                if tol != d0:
                    continue
                gmean, _ = _group_mean_std(vals)
                pts.append((it, gmean))
            pts.sort(key=lambda p: p[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.0, label=f"startεd0={d0:g} (tol=1e{d0:g})")

        ax.set_yscale("log")
        ax.set_title(f"Dataset Summary Gap vs Iteration | {dataset}")
        ax.set_xlabel("FW iteration")
        ax.set_ylabel("mean gap across all selected graphs")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{dataset}_dataset_gap_vs_iter.png"), dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gap-vs-iteration asymptotic curves for LMO tolerance sweeps.")
    parser.add_argument("--phase", default="subset", choices=["subset", "full"], help="Sweep phase directory under Result/benchmarks_lmo_tolerance")
    parser.add_argument("--input-root", default=None, help="Override full input root directory (if unset: Result/benchmarks_lmo_tolerance/<phase>)")
    parser.add_argument("--output", default=None, help="Output directory for plots and aggregated CSVs")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    input_root = args.input_root or os.path.join(project_root, "Result", "benchmarks_lmo_tolerance", args.phase)
    output_dir = args.output or os.path.join(input_root, "plots")
    os.makedirs(output_dir, exist_ok=True)

    runs_csv = os.path.join(input_root, "runs.csv")
    run_map: Dict[str, RunInfo] | None = None
    if os.path.exists(runs_csv):
        run_map = _parse_runs(runs_csv)
        print("Loaded run map from:", runs_csv)
    else:
        print("runs.csv not found, falling back to per-iteration metadata from .iters.csv files")

    iter_rows = _load_iteration_rows(input_root, run_map)
    if not iter_rows:
        raise RuntimeError(f"No iteration rows found under: {input_root}")

    graph_agg = _aggregate_graph(iter_rows)
    graph_rows = []
    for (dataset, graph_name, d0, it), (gmean, gstd, count) in sorted(graph_agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        graph_rows.append([dataset, graph_name, d0, 10.0**d0, it, gmean, gstd, count])

    dataset_agg = _aggregate_dataset(iter_rows)
    dataset_rows = []
    for (dataset, d0, it), (gmean, gstd, count) in sorted(dataset_agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        dataset_rows.append([dataset, d0, 10.0**d0, it, gmean, gstd, count])

    _write_csv(
        os.path.join(output_dir, "gap_by_graph_tolerance_iteration.csv"),
        ["dataset", "graph_name", "start_epsilon_d0", "lmo_tolerance", "iteration", "mean_gap", "std_gap", "samples"],
        graph_rows,
    )

    _write_csv(
        os.path.join(output_dir, "gap_by_dataset_tolerance_iteration.csv"),
        ["dataset", "start_epsilon_d0", "lmo_tolerance", "iteration", "mean_gap", "std_gap", "samples"],
        dataset_rows,
    )

    _plot_per_graph(iter_rows, output_dir)
    _plot_dataset_summary(iter_rows, output_dir)

    print("Plots and aggregate CSVs written to:", output_dir)


if __name__ == "__main__":
    main()
