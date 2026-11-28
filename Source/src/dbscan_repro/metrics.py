"""Evaluation metrics and timing utilities.

Owners: M2 / M3 / M4.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_summary(
    y_true: NDArray[np.intp],
    y_pred: NDArray[np.intp],
) -> dict[str, float]:
    """Compute ARI, NMI, predicted cluster count (excluding noise), and noise ratio."""
    truth = np.asarray(y_true).ravel()
    prediction = np.asarray(y_pred).ravel()
    if truth.size == 0:
        raise ValueError("label arrays must not be empty")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same number of labels")
    clusters = np.unique(prediction[prediction != -1])
    return {
        "ari": float(adjusted_rand_score(truth, prediction)),
        "nmi": float(normalized_mutual_info_score(truth, prediction)),
        "n_clusters": float(clusters.size),
        "noise_ratio": float(np.mean(prediction == -1)),
    }


def measure_runtime_seconds[T](fn: Callable[[], T], repeats: int = 3) -> tuple[T, float, float]:
    """Run ``fn`` ``repeats`` times and return ``(last_result, mean_sec, std_sec)``."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    durations = np.empty(repeats, dtype=np.float64)
    result = None
    for index in range(repeats):
        start = perf_counter()
        result = fn()
        durations[index] = perf_counter() - start
    return result, float(durations.mean()), float(durations.std(ddof=0))  # type: ignore[return-value]
