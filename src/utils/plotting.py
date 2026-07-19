"""Shared matplotlib styling so every figure in the report looks consistent.

Palette follows a single validated scheme: blue for the primary series,
green for a second category, neutral greys for chrome. Figures are saved
as 200 dpi PNGs into results/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# Series colours (colour-blind-safe adjacent pair)
BLUE = "#2a78d6"
GREEN = "#008300"

# Chrome / ink
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

DPI = 200


def apply_style() -> None:
    """Apply the project-wide matplotlib style (call once per script)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str, figures_dir: str | Path) -> Path:
    """Save a figure as <figures_dir>/<name>.png at report resolution."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"[fig] saved {out}")
    return out
