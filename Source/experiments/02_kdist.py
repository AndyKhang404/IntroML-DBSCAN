"""Experiment 02: Sorted k-dist graph and Eps valley selection.

Owner: M3 (using kdist from M2).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from dbscan_repro.config import load_config, set_seed, write_run_metadata
from dbscan_repro.datasets import make_synthetic_db
from dbscan_repro.kdist import compute_sorted_k_distances, detect_kdist_valley_index
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
    k = int(cfg.params["kdist"]["k"])
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    rows: list[dict[str, object]] = []
    write_run_metadata(cfg, "kdist")

    for name in ("db1", "db2", "db3"):
        X, _ = make_synthetic_db(name, cfg.params["effectiveness"]["n_samples"], cfg.seed)
        curve = compute_sorted_k_distances(X, k=k)
        index, eps = detect_kdist_valley_index(curve)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(range(1, len(curve) + 1), curve, color="navy")
        ax.axvline(index + 1, color="crimson", linestyle="--", label=f"valley eps={eps:.3f}")
        ax.axhline(eps, color="crimson", alpha=0.45, linestyle=":")
        ax.set(
            title=f"{name.upper()} sorted {k}-dist graph",
            xlabel="Points sorted by decreasing k-dist",
            ylabel=f"Distance to {k}-th nearest neighbor",
        )
        ax.legend()
        save_figure(fig, cfg.results_dir / f"kdist_{name}.png")
        rows.append(
            {
                "dataset": name,
                "seed": cfg.seed,
                "k": k,
                "valley_index": index,
                "suggested_eps": eps,
            }
        )

    with (cfg.results_dir / "kdist_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
