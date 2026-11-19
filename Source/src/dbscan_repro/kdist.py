"""Sorted k-dist calculation and Eps valley heuristic (Section 4.2 of Ester et al., 1996).

Owner: M2 (Core algorithms).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compute_sorted_k_distances(X: ArrayLike, k: int = 4) -> NDArray[np.float64]:
    """Return distances to the k-th nearest neighbor sorted in descending order."""
    _ = (np.asarray(X, dtype=np.float64), k)
    # TODO(M2): compute k-th nearest neighbor distances and sort descending.
    raise NotImplementedError("M2: implement sorted k-dist helper.")
