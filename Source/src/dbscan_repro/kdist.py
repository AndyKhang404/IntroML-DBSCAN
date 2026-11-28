"""Sorted k-dist calculation and Eps heuristics (Section 4.2 of Ester et al., 1996).

Owner: M2 (Core algorithms).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree


def compute_sorted_k_distances(X: ArrayLike, k: int = 4) -> NDArray[np.float64]:
    """Return distances to the k-th nearest neighbor sorted in descending order.

    The distance of a point to its k-th nearest neighbor *excluding the point
    itself* is used (KDD-96, Section 4.2), so setting ``Eps`` to such a value
    yields a neighborhood of ``k + 1`` points for almost all points.

    Parameters
    ----------
    X:
        Point cloud of shape ``(n_samples, n_features)``.
    k:
        Neighbor order, with ``1 <= k < n_samples``.

    Returns
    -------
    NDArray[np.float64]
        The ``n_samples`` k-dist values in descending order.
    """
    points = np.asarray(X, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {points.ndim} dimension(s)")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= points.shape[0]:
        raise ValueError(
            f"k must be smaller than the number of samples ({points.shape[0]}), got {k}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("X must contain only finite values")

    # Column 0 is the point itself (distance 0), column k is its k-th nearest
    # neighbor; dropping the self column would shift the indexing by one.
    distances, _ = KDTree(points).query(points, k=k + 1)
    kth_distances = np.asarray(distances[:, k], dtype=np.float64)
    return np.sort(kth_distances)[::-1]


def suggest_eps_from_noise_ratio(
    sorted_k_dist: ArrayLike,
    noise_ratio: float,
) -> tuple[int, float]:
    """Select the ``Eps`` threshold from a descending k-dist curve.

    Ester et al. (1996, Section 4.2) propose that, when the percentage of noise
    is known, the threshold point is taken at the position of the sorted k-dist
    graph that corresponds to that percentage: all points with a larger k-dist
    (to the left of the threshold) are considered noise.  The value of the
    k-dist at the threshold is used as ``Eps``.

    Parameters
    ----------
    sorted_k_dist:
        Output of :func:`compute_sorted_k_distances` (descending order).
    noise_ratio:
        Expected fraction of noise in ``[0, 1)``.

    Returns
    -------
    tuple[int, float]
        The threshold index and the corresponding ``Eps`` value.
    """
    curve = np.asarray(sorted_k_dist, dtype=np.float64).ravel()
    if curve.size == 0:
        raise ValueError("sorted_k_dist must not be empty")
    if not 0.0 <= noise_ratio < 1.0:
        raise ValueError(f"noise_ratio must be in [0, 1), got {noise_ratio}")

    index = min(int(round(noise_ratio * curve.size)), curve.size - 1)
    return index, float(curve[index])


def detect_kdist_valley_index(sorted_k_dist: ArrayLike) -> tuple[int, float]:
    """Locate the knee (first valley) of a descending sorted k-dist curve.

    The knee is estimated with a kneedle-style heuristic: the point with the
    maximum perpendicular distance to the chord joining the two endpoints of
    the normalized curve.  This automates the threshold choice that the paper
    leaves to visual inspection (KDD-96, Section 4.2).

    Parameters
    ----------
    sorted_k_dist:
        Output of :func:`compute_sorted_k_distances` (descending order).

    Returns
    -------
    tuple[int, float]
        The detected threshold index and the corresponding ``Eps`` value.
    """
    curve = np.asarray(sorted_k_dist, dtype=np.float64).ravel()
    if curve.size < 3:
        raise ValueError("sorted_k_dist must contain at least three values")
    if not np.all(np.isfinite(curve)):
        raise ValueError("sorted_k_dist must contain only finite values")

    span = float(curve.max() - curve.min())
    if span == 0.0:
        # A flat curve has no valley; fall back to the first (largest) value.
        return 0, float(curve[0])

    x = np.linspace(0.0, 1.0, curve.size)
    y = (curve - curve.min()) / span
    deviation = np.abs(x + y - 1.0)
    index = int(np.argmax(deviation))
    return index, float(curve[index])
