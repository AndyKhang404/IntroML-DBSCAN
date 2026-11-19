"""Experiment 03: Runtime scaling (brute vs KD-tree DBSCAN vs CLARANS).

Owner: M3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dbscan_repro.config import load_config, set_seed


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
    # TODO(M3): benchmark brute vs KD-tree DBSCAN vs CLARANS across sample_sizes.
    raise NotImplementedError("M3: benchmark runtime scaling at paper sample sizes.")


if __name__ == "__main__":
    main()
