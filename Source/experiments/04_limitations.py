"""Experiment 04: Limitation experiments (varying density, high-dim, sensitivity) & backup dataset.

Owner: M4.
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
    # TODO(M4): run varying-density, high-dim, sensitivity, and backup dataset experiments.
    raise NotImplementedError("M4: run limitation experiments and modern dataset evaluation.")


if __name__ == "__main__":
    main()
