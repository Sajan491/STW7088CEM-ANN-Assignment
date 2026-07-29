"""Shared matplotlib styling so every figure in the report looks consistent and
publication-quality.

A single, cohesive theme: a colour-blind-safe palette (blue primary, green
secondary, warm accents), recessive grid and axes, generous whitespace, bold
titles with muted subtitles, and selective direct value labels. Every figure is
saved as a 200 dpi PNG into results/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# --- Palette (colour-blind-safe, validated) ---
BLUE = "#2f6fd0"        # primary series
BLUE_SOFT = "#a9c9f3"   # light fill / secondary tint
GREEN = "#0f8a4d"       # secondary series
GREEN_SOFT = "#a8dcc0"
ORANGE = "#e8833a"      # accent / highlight
RED = "#d64545"         # negative / warning highlight
VIOLET = "#6a5acd"

# --- Chrome / ink ---
INK = "#141414"          # primary text
INK_SECONDARY = "#4a4a4a"
INK_MUTED = "#8a8a8a"    # axis labels, footnotes
GRID = "#e9e9e6"         # hairline gridlines
SPINE = "#cfcfca"        # baseline
SURFACE = "#ffffff"      # chart surface (clean white reads best in a report)
PANEL = "#f6f7f9"        # subtle panel fill for callouts

DPI = 200


def apply_style() -> None:
    """Apply the project-wide matplotlib style (call once per script)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.dpi": 110,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 1.0,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 10.5,
            "axes.labelpad": 6,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "legend.handlelength": 1.4,
            "figure.constrained_layout.use": False,
        }
    )


def titles(ax, title: str, subtitle: str | None = None) -> None:
    """Bold left-aligned title with an optional muted subtitle above the axes.

    Both title and subtitle are offset from the axes top in *points* so their
    spacing stays constant regardless of figure size.
    """
    if subtitle:
        ax.set_title(title, loc="left", pad=30, fontsize=13, fontweight="bold", color=INK)
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=9.5, color=INK_MUTED)
    else:
        ax.set_title(title, loc="left", pad=10, fontsize=13, fontweight="bold", color=INK)


def bar_labels(ax, bars, fmt="{:.0f}", fontsize=9, color=INK_SECONDARY, pad=3) -> None:
    """Print a value label centred above each bar."""
    for rect in bars:
        h = rect.get_height()
        ax.annotate(fmt.format(h), (rect.get_x() + rect.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, pad), ha="center",
                    va="bottom", fontsize=fontsize, color=color)


def footnote(fig, text: str) -> None:
    """Small muted source/caption note at the bottom-left of the figure."""
    fig.text(0.008, 0.005, text, ha="left", va="bottom", fontsize=8, color=INK_MUTED)


def save_figure(fig: plt.Figure, name: str, figures_dir: str | Path) -> Path:
    """Save a figure as <figures_dir>/<name>.png at report resolution."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    print(f"[fig] saved {out}")
    return out
