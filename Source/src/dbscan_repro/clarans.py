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
        self.n_clusters = int(n_clusters)
        self.num_local = int(num_local)
        self.max_neighbor = max_neighbor
        self.random_state = int(random_state)
        self.labels_: NDArray[np.intp] | None = None
        self.medoid_indices_: NDArray[np.intp] | None = None

    def fit(self, X: ArrayLike) -> CLARANS:
        """Fit CLARANS medoids on ``X``."""
        _ = np.asarray(X, dtype=np.float64)
        # TODO(M3): implement randomized medoid swap search for CLARANS.
        raise NotImplementedError("M3: implement CLARANS baseline.")

    def fit_predict(self, X: ArrayLike) -> NDArray[np.intp]:
        """Fit CLARANS on ``X`` and return cluster assignments."""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_
