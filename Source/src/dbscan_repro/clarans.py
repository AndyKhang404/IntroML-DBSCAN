"""From-scratch CLARANS baseline (Ng & Han, 1994/2002) for paper comparison.

Owner: M3 (Paper reproduction).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dbscan_repro.config import DEFAULT_SEED


class CLARANS:
    """Clustering Large Applications based upon Randomized Search.

    Parameters
    ----------
    n_clusters:
        Number of medoids ``k``.
    num_local:
        Number of local minima searches.
    max_neighbor:
        Maximum number of neighbors examined per search step.
    random_state:
        Seed for reproducible randomized medoid selection.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        num_local: int = 2,
        max_neighbor: int | None = None,
        random_state: int = DEFAULT_SEED,
    ) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if num_local < 1:
            raise ValueError(f"num_local must be >= 1, got {num_local}")
        if max_neighbor is not None and max_neighbor < 1:
            raise ValueError(f"max_neighbor must be >= 1, got {max_neighbor}")
        self.n_clusters = int(n_clusters)
        self.num_local = int(num_local)
        self.max_neighbor = max_neighbor
        self.random_state = int(random_state)
        self.labels_: NDArray[np.intp] | None = None
        self.medoid_indices_: NDArray[np.intp] | None = None

    def fit(self, X: ArrayLike) -> CLARANS:
        """Fit CLARANS medoids on ``X`` using randomized one-medoid swaps."""
        points = np.asarray(X, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {points.ndim} dimension(s)")
        n_samples = points.shape[0]
        if n_samples == 0 or points.shape[1] == 0:
            raise ValueError("X must contain samples with at least one feature")
        if not np.all(np.isfinite(points)):
            raise ValueError("X must contain only finite values")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples")

        rng = np.random.default_rng(self.random_state)
        medoids = rng.choice(n_samples, size=self.n_clusters, replace=False)
        best_cost = np.inf
        best_medoids = medoids.copy()
        max_neighbor = self.max_neighbor or max(
            250, self.n_clusters * (n_samples - self.n_clusters)
        )

        for _ in range(self.num_local):
            medoids = rng.choice(n_samples, size=self.n_clusters, replace=False)
            distances = self._distances(points, medoids)
            cost = float(distances.min(axis=1).sum())
            neighbor = 0

            while neighbor < max_neighbor:
                medoid_slot = int(rng.integers(self.n_clusters))
                is_medoid = np.zeros(n_samples, dtype=bool)
                is_medoid[medoids] = True
                candidates = np.flatnonzero(~is_medoid)
                replacement = int(rng.choice(candidates))
                candidate_dist = np.linalg.norm(points - points[replacement], axis=1)

                without_slot = distances.copy()
                without_slot[:, medoid_slot] = np.inf
                alternative = without_slot.min(axis=1)
                proposed = np.minimum(alternative, candidate_dist)
                proposed_cost = float(proposed.sum())
                if proposed_cost < cost - 1e-12:
                    medoids[medoid_slot] = replacement
                    distances[:, medoid_slot] = candidate_dist
                    cost = proposed_cost
                    neighbor = 0
                else:
                    neighbor += 1

            if cost < best_cost:
                best_cost = cost
                best_medoids = medoids.copy()

        final_distances = self._distances(points, best_medoids)
        self.medoid_indices_ = np.asarray(best_medoids, dtype=np.intp)
        self.labels_ = final_distances.argmin(axis=1).astype(np.intp)
        return self

    @staticmethod
    def _distances(points: NDArray[np.float64], medoids: NDArray[np.intp]) -> NDArray[np.float64]:
        """Euclidean distances from every sample to each selected medoid."""
        return np.linalg.norm(points[:, None, :] - points[medoids][None, :, :], axis=2)

    def fit_predict(self, X: ArrayLike) -> NDArray[np.intp]:
        """Fit CLARANS on ``X`` and return cluster assignments."""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_
