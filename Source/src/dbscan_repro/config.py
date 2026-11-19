"""Configuration and deterministic seeding helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_SEED = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Loaded experiment configuration with defaults for reproducibility."""

    seed: int = DEFAULT_SEED
    results_dir: Path = Path("Source/results")
    params: dict[str, Any] = field(default_factory=dict)


def set_seed(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Fix Python and NumPy RNG states and return a fresh NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML config file."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    seed = int(raw.get("seed", DEFAULT_SEED))
    results_dir = Path(raw.get("results_dir", "Source/results"))
    params = {k: v for k, v in raw.items() if k not in {"seed", "results_dir"}}
    return ExperimentConfig(seed=seed, results_dir=results_dir, params=params)
