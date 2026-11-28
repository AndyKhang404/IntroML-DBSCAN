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
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.autolayout": True,
        }
    )


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Save figure ensuring parent directories exist."""
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output
