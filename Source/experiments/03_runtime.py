"""Experiment 03: Runtime scaling (brute vs KD-tree DBSCAN vs CLARANS).

Owner: M3.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
from dbscan_repro.clarans import CLARANS
from dbscan_repro.config import load_config, set_seed, write_run_metadata
from dbscan_repro.datasets import make_spatial_scaling_dataset
from dbscan_repro.dbscan import DBSCANScratch
from dbscan_repro.metrics import measure_runtime_seconds
from dbscan_repro.plotting import apply_paper_style, save_figure


class RuntimeResult(TypedDict):
    n_samples: int
    method: str
    mean_seconds: float
    std_seconds: float
    repeats: int
    warmups: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Source/experiments/configs/default.yaml"),
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    settings = cfg.params["runtime"]
    sizes = settings["sample_sizes"]
    warmups = int(settings.get("warmup_runs", 1))
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    rows: list[RuntimeResult] = []
    write_run_metadata(cfg, "runtime")
    for n_samples in sizes:
        X = make_spatial_scaling_dataset(n_samples, cfg.seed)
        methods = {
            "DBSCAN brute": lambda data=X: DBSCANScratch(
                eps=settings["eps"], min_pts=settings["min_pts"], algorithm="brute"
            ).fit_predict(data),
            "DBSCAN KD-tree": lambda data=X: DBSCANScratch(
                eps=settings["eps"], min_pts=settings["min_pts"], algorithm="kdtree"
            ).fit_predict(data),
            "CLARANS": lambda data=X: CLARANS(
                n_clusters=settings.get("clarans_clusters", 8),
                num_local=settings.get("clarans_num_local", 1),
                max_neighbor=settings.get("clarans_max_neighbor", 40),
                random_state=cfg.seed,
            ).fit_predict(data),
        }
        for method, run in methods.items():
            for _ in range(warmups):
                run()
            _, mean, std = measure_runtime_seconds(run, repeats=settings["repeats"])
            rows.append(
                {
                    "n_samples": n_samples,
                    "method": method,
                    "mean_seconds": mean,
                    "std_seconds": std,
                    "repeats": settings["repeats"],
                    "warmups": warmups,
                    "seed": cfg.seed,
                }
            )

    with (cfg.results_dir / "runtime_scaling.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for method in ("DBSCAN brute", "DBSCAN KD-tree", "CLARANS"):
        selected = [row for row in rows if row["method"] == method]
        ax.errorbar(
            [row["n_samples"] for row in selected],
            [row["mean_seconds"] for row in selected],
            yerr=[row["std_seconds"] for row in selected],
            marker="o",
            capsize=3,
            label=method,
        )
    ax.set(
        xlabel="Number of points",
        ylabel="Runtime (seconds)",
        yscale="log",
        title="Runtime scaling",
    )
    ax.legend()
    save_figure(fig, cfg.results_dir / "runtime_scaling.png")


if __name__ == "__main__":
    main()
