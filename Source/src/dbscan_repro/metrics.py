"""Evaluation metrics and timing utilities.

Owners: M2 / M3 / M4.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def clustering_summary(
    y_true: NDArray[np.intp],
    y_pred: NDArray[np.intp],
) -> dict[str, float]:
    """Compute ARI, NMI, predicted cluster count (excluding noise), and noise ratio."""
    _ = (y_true, y_pred)
    # TODO(M3): compute ARI, NMI, non-noise cluster count, and noise ratio.
    raise NotImplementedError("M3: implement clustering summary metrics.")


def measure_runtime_seconds[T](fn: Callable[[], T], repeats: int = 3) -> tuple[T, float, float]:
    """Run ``fn`` ``repeats`` times and return ``(last_result, mean_sec, std_sec)``."""
    _ = (fn, repeats)
    # TODO(M3): benchmark callable runtime across repeated runs.
    raise NotImplementedError("M3: implement runtime measurement helper.")
