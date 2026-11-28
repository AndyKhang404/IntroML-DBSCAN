"""Configuration and deterministic seeding helpers."""

from __future__ import annotations

import json
import platform
import random
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
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


def write_run_metadata(cfg: ExperimentConfig, experiment: str) -> Path:
    """Write seed, full config, and runtime environment beside experiment outputs."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    packages = ("numpy", "scipy", "scikit-learn", "matplotlib", "pandas", "PyYAML")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    destination = cfg.results_dir / f"{experiment}_metadata.json"
    destination.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "seed": cfg.seed,
                "config": cfg.params,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor() or "unknown",
                "packages": versions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
