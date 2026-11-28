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
    """Generate deterministic, figure-inspired recreations of paper DB1–DB3.

    The original point coordinates are unavailable, so these are controlled
    synthetic analogues: DB1 has compact, uneven-density groups; DB2 has
    interlocking non-convex arcs; DB3 has nested rings plus uniform noise.
    Noise is labeled ``-1`` in the ground truth.
    """
    if name not in ("db1", "db2", "db3"):
        raise ValueError(f"unknown synthetic database {name!r}; expected db1, db2, or db3")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    rng = np.random.default_rng(seed)
    labels = np.full(n_samples, -1, dtype=np.intp)
    points = np.empty((n_samples, 2), dtype=np.float64)

    if name == "db1":
        # Four separated groups with deliberately different local densities.
        fractions = np.array([0.28, 0.25, 0.22, 0.15, 0.10])
        counts = np.floor(fractions * n_samples).astype(int)
        counts[-1] += n_samples - int(counts.sum())
        centers = np.array([[0.2, 0.25], [0.72, 0.24], [0.25, 0.75], [0.74, 0.74]])
        scales = (0.025, 0.045, 0.065, 0.035)
        cursor = 0
        for cluster_id, (count, center, scale) in enumerate(
            zip(counts[:4], centers, scales, strict=True)
        ):
            points[cursor : cursor + count] = rng.normal(center, scale, (count, 2))
            labels[cursor : cursor + count] = cluster_id
            cursor += count
        points[cursor:] = rng.uniform(0.0, 1.0, size=(n_samples - cursor, 2))
    elif name == "db2":
        noise_count = max(1, int(round(0.04 * n_samples))) if n_samples > 5 else 0
        cluster_count = n_samples - noise_count
        counts = [cluster_count // 2, cluster_count - cluster_count // 2]
        for cluster_id, count in enumerate(counts):
            theta = rng.uniform(0.0, np.pi, count)
            x = np.cos(theta)
            y = np.sin(theta)
            if cluster_id == 1:
                x, y = 1.0 - x, 0.5 - y
            start = sum(counts[:cluster_id])
            points[start : start + count] = np.column_stack((x, y)) * 0.35 + 0.5
            points[start : start + count] += rng.normal(0.0, 0.012, (count, 2))
            labels[start : start + count] = cluster_id
        points[cluster_count:] = rng.uniform(0.0, 1.0, (noise_count, 2))
    else:
        noise_count = max(1, int(round(0.08 * n_samples))) if n_samples > 5 else 0
        clustered = n_samples - noise_count
        counts = [clustered // 2, clustered - clustered // 2]
        cursor = 0
        for cluster_id, (count, radius) in enumerate(zip(counts, (0.16, 0.34), strict=True)):
            theta = rng.uniform(0.0, 2.0 * np.pi, count)
            radii = radius + rng.normal(0.0, 0.012, count)
            points[cursor : cursor + count] = np.column_stack(
                (0.5 + radii * np.cos(theta), 0.5 + radii * np.sin(theta))
            )
            labels[cursor : cursor + count] = cluster_id
            cursor += count
        points[cursor:] = rng.uniform(0.0, 1.0, (noise_count, 2))

    order = rng.permutation(n_samples)
    return points[order], labels[order]


def make_spatial_scaling_dataset(
    n_samples: int,
    seed: int = DEFAULT_SEED,
) -> NDArray[np.float64]:
    """Generate a 2D spatial point dataset substituting SEQUOIA 2000 for scaling tests."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    rng = np.random.default_rng(seed)
    # A bounded map-like point cloud: population centers plus dispersed points.
    center_count = min(12, max(3, n_samples // 500))
    centers = rng.uniform(0.08, 0.92, size=(center_count, 2))
    cluster_sizes = rng.multinomial(int(n_samples * 0.82), np.full(center_count, 1 / center_count))
    clusters = [
        rng.normal(center, (0.025, 0.04), size=(count, 2))
        for center, count in zip(centers, cluster_sizes, strict=True)
    ]
    dispersed = rng.uniform(0.0, 1.0, size=(n_samples - int(cluster_sizes.sum()), 2))
    points = np.vstack((*clusters, dispersed))
    return np.clip(points[rng.permutation(n_samples)], 0.0, 1.0)
