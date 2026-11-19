"""From-scratch DBSCAN implementations (brute-force and KD-tree spatial index).

Owner: M2 (Core algorithms).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

NOISE_LABEL = -1
AlgorithmType = Literal["brute", "kdtree"]


class DBSCANScratch:
    """Density-Based Spatial Clustering of Applications with Noise (Ester et al., 1996).

    Parameters
    ----------
    eps:
        Maximum distance (Eps) between two points for one to be considered in the
        neighborhood of the other.
    min_pts:
        Minimum number of points in the Eps-neighborhood (including the point itself)
        required to form a core point (MinPts).
    algorithm:
        Neighborhood query strategy: ``"brute"`` (O(n^2) pairwise distances) or
        ``"kdtree"`` (spatial index acceleration).
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_pts: int = 4,
        algorithm: AlgorithmType = "kdtree",
    ) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if min_pts < 1:
            raise ValueError(f"min_pts must be >= 1, got {min_pts}")
        if algorithm not in ("brute", "kdtree"):
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.eps = float(eps)
        self.min_pts = int(min_pts)
        self.algorithm: AlgorithmType = algorithm
        self.labels_: NDArray[np.intp] | None = None
        self.core_sample_indices_: NDArray[np.intp] | None = None

    def fit(self, X: ArrayLike) -> DBSCANScratch:
        """Compute DBSCAN clustering on 2D array ``X`` of shape ``(n_samples, n_features)``."""
        _ = np.asarray(X, dtype=np.float64)
        # TODO(M2): implement brute-force and KD-tree DBSCAN clustering.
        raise NotImplementedError("M2: implement brute-force and KD-tree DBSCAN.")

    def fit_predict(self, X: ArrayLike) -> NDArray[np.intp]:
        """Fit DBSCAN on ``X`` and return cluster labels (``-1`` denotes noise)."""
        # TODO(M2): return cluster labels after fit() is implemented.
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_
