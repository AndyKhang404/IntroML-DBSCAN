"""Shared plotting style helpers for figures embedded in the Typst report.

Owners: M3 / M4.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def apply_paper_style() -> None:
    """Configure consistent publication-quality Matplotlib defaults."""
    # TODO(M3): configure shared Matplotlib rcParams for report figures.
    raise NotImplementedError("M3: implement shared plotting style.")


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save figure ensuring parent directories exist."""
    _ = (fig, path)
    # TODO(M3): save Matplotlib figure to disk and close it cleanly.
    raise NotImplementedError("M3: implement figure export helper.")
