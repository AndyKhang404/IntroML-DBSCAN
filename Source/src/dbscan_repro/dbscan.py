"""From-scratch DBSCAN implementations (brute-force and KD-tree spatial index).

Owner: M2 (Core algorithms).

The clustering logic follows the pseudocode of Ester et al. (KDD-96), Section
4.1: a sequential scan starts a new cluster at every unclassified core point and
expands it breadth-first via ``ExpandCluster``.  Eps-neighborhoods are served
either by an O(n^2) brute-force scan or by a KD-tree spatial index, which lowers
the average query cost to O(log n) and the overall runtime to O(n log n) for
low-dimensional data.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree

NOISE_LABEL = -1
UNCLASSIFIED = -2
AlgorithmType = Literal["brute", "kdtree"]

RegionQuery = Callable[[int], NDArray[np.intp]]


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
        """Compute DBSCAN clustering on 2D array ``X`` of shape ``(n_samples, n_features)``.

        Every sample receives a cluster label in ``0, 1, ...`` or ``-1`` (noise).
        Samples with at least ``min_pts`` neighbors within ``eps`` (including
        themselves) are reported in ``core_sample_indices_``.
        """
        points = self._validate_input(X)
        region_query = self._make_region_query(points)

        n_samples = points.shape[0]
        labels = np.full(n_samples, UNCLASSIFIED, dtype=np.intp)
        is_core = np.zeros(n_samples, dtype=bool)

        cluster_id = 0
        for point in range(n_samples):
            if labels[point] != UNCLASSIFIED:
                continue
            if self._expand_cluster(point, region_query, labels, is_core, cluster_id):
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(is_core).astype(np.intp)
        return self

    def fit_predict(self, X: ArrayLike) -> NDArray[np.intp]:
        """Fit DBSCAN on ``X`` and return cluster labels (``-1`` denotes noise)."""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_

    def _expand_cluster(
        self,
        point: int,
        region_query: RegionQuery,
        labels: NDArray[np.intp],
        is_core: NDArray[np.bool_],
        cluster_id: int,
    ) -> bool:
        """Expand a cluster from ``point`` following ``ExpandCluster`` in KDD-96.

        Returns ``True`` if ``point`` is a core point and a cluster was formed,
        ``False`` if ``point`` was marked as noise.
        """
        seeds = region_query(point)
        if seeds.size < self.min_pts:
            # No core point: the point is a border point of a cluster that has
            # not been discovered yet, or genuine noise (Definition 6).
            labels[point] = NOISE_LABEL
            return False

        is_core[point] = True
        labels[point] = cluster_id
        queue: deque[int] = deque()
        for neighbor in seeds:
            candidate = int(neighbor)
            if labels[candidate] == UNCLASSIFIED:
                labels[candidate] = cluster_id
                if candidate != point:
                    queue.append(candidate)
            elif labels[candidate] == NOISE_LABEL:
                # Previously visited points may turn out to be border points of
                # this cluster; they are relabeled but never re-queued because a
                # NOISE point is known not to be a core point (KDD-96, p. 229).
                labels[candidate] = cluster_id

        while queue:
            current = queue.popleft()
            neighbors = region_query(current)
            if neighbors.size < self.min_pts:
                # ``current`` is a border point: it joins the cluster but its
                # neighborhood is not expanded.
                continue
            is_core[current] = True
            for neighbor in neighbors:
                candidate = int(neighbor)
                if labels[candidate] == UNCLASSIFIED:
                    labels[candidate] = cluster_id
                    queue.append(candidate)
                elif labels[candidate] == NOISE_LABEL:
                    labels[candidate] = cluster_id
        return True

    def _make_region_query(self, points: NDArray[np.float64]) -> RegionQuery:
        """Build the Eps-neighborhood query used by ``ExpandCluster``."""
        if self.algorithm == "kdtree":
            tree = KDTree(points)

            def query_kdtree(point: int) -> NDArray[np.intp]:
                neighbors = tree.query_ball_point(points[point], r=self.eps)
                return np.sort(np.asarray(neighbors, dtype=np.intp))

            return query_kdtree

        eps_sq = self.eps * self.eps

        def query_brute(point: int) -> NDArray[np.intp]:
            diff = points - points[point]
            dist_sq = np.einsum("ij,ij->i", diff, diff)
            return np.flatnonzero(dist_sq <= eps_sq).astype(np.intp)

        return query_brute

    @staticmethod
    def _validate_input(X: ArrayLike) -> NDArray[np.float64]:
        points = np.asarray(X, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {points.ndim} dimension(s)")
        if points.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(points)):
            raise ValueError("X must contain only finite values")
        return points
