"""Experiment 01: DBSCAN vs CLARANS cluster effectiveness on synthetic DB1–3.

Owner: M3.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from dbscan_repro.clarans import CLARANS
from dbscan_repro.config import load_config, set_seed, write_run_metadata
from dbscan_repro.datasets import make_synthetic_db
from dbscan_repro.dbscan import DBSCANScratch
from dbscan_repro.metrics import clustering_summary
from dbscan_repro.plotting import apply_paper_style, save_figure


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
    settings = cfg.params["effectiveness"]
    results = cfg.results_dir
    results.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    rows: list[dict[str, object]] = []
    write_run_metadata(cfg, "effectiveness")

    for name in ("db1", "db2", "db3"):
        X, truth = make_synthetic_db(name, settings["n_samples"], cfg.seed)
        models = {
            "DBSCAN": DBSCANScratch(
                eps=settings["eps_by_db"][name],
                min_pts=settings["min_pts"],
                algorithm="kdtree",
            ),
            "CLARANS": CLARANS(
                n_clusters=settings.get("clarans_clusters_by_db", {}).get(
                    name, settings["clarans_clusters"]
                ),
                num_local=settings.get("clarans_num_local", 1),
                max_neighbor=settings.get("clarans_max_neighbor", 50),
                random_state=cfg.seed,
            ),
        }
        predictions: dict[str, object] = {}
        for method, model in models.items():
            labels = model.fit_predict(X)
            predictions[method] = labels
            rows.append(
                {
                    "dataset": name,
                    "method": method,
                    "seed": cfg.seed,
                    "n_samples": len(X),
                    **clustering_summary(truth, labels),
                }
            )

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharex=True, sharey=True)
        for ax, (title, labels) in zip(
            axes, [("Ground truth", truth), *predictions.items()], strict=True
        ):
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=9, cmap="tab10", vmin=-1)
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        fig.suptitle(f"{name.upper()}: DBSCAN and CLARANS comparison")
        save_figure(fig, results / f"effectiveness_{name}.png")

    with (results / "effectiveness.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
