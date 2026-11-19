"""Dataset generators and loaders for DBSCAN reproduction and bonus studies.

Owners: M3 (DB1-3 + SEQUOIA substitute), M4 (bonus/limitation datasets).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dbscan_repro.config import DEFAULT_SEED

SyntheticDBName = Literal["db1", "db2", "db3"]


def make_synthetic_db(
    name: SyntheticDBName,
    n_samples: int = 1000,
    seed: int = DEFAULT_SEED,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Generate recreations of synthetic Databases 1–3 from Ester et al. (1996)."""
    _ = (name, n_samples, seed)
    # TODO(M3): recreate synthetic DB1-3 point distributions and ground-truth labels.
    raise NotImplementedError("M3: implement synthetic DB1-3 generators.")


def make_spatial_scaling_dataset(
    n_samples: int,
    seed: int = DEFAULT_SEED,
) -> NDArray[np.float64]:
    """Generate a 2D spatial point dataset substituting SEQUOIA 2000 for scaling tests."""
    _ = (n_samples, seed)
    # TODO(M3): generate 2D spatial points substituting SEQUOIA 2000 (and document in report).
    raise NotImplementedError("M3: implement spatial scaling dataset.")
