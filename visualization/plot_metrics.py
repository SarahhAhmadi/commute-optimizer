"""
visualization/plot_metrics.py
Plots optimization cost convergence profiles over search steps or generations.

The solvers run for very different lengths (SA ~ thousands of steps, Beam ~ tens),
so a single linear x-axis squeezes the short runs into a thin strip at the left.
This module therefore produces:
  1. <save_path>                      -> log-scale steps: full range + zoom near the optimum
  2. <save_path stem>_per_algo.png    -> one panel per solver, each with its own x-range
"""

import os
from typing import Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Fixed colour per algorithm (identity follows the algorithm, not its rank)
PALETTE = {
    "Hill Climbing": "#2a78d6",
    "Simulated Annealing": "#eb6834",
    "Local Beam Search": "#1baf7a",
    "Genetic Algorithm": "#eda100",
}
STYLES = {
    "Hill Climbing": "-",
    "Simulated Annealing": "--",
    "Local Beam Search": "-.",
    "Genetic Algorithm": ":",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"


def _style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def _first_best_index(hist: List[float]) -> int:
    best = hist[-1]
    for i, v in enumerate(hist):
        if v <= best + 1e-9:
            return i
    return len(hist) - 1


def _y_limits(histories: Dict[str, List[float]]):
    vals = [v for h in histories.values() for v in h if v != float("inf")]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.08, 0.5)
    return lo - pad, hi + pad


def plot_convergence(
    histories: Dict[str, List[float]],
    title: str = "Search Convergence Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plots best-so-far objective f(x) for every solver.

    Args:
        histories: algorithm name -> list of best-so-far costs (one per step).
        title: figure title.
        save_path: if given, PNGs are written and no window is opened.
        show: open an interactive window (ignored when save_path is given).
    """
    histories = {k: v for k, v in histories.items() if v}
    if not histories:
        return
    if save_path:
        show = False
        matplotlib.use("Agg", force=True)

    ylo, yhi = _y_limits(histories)

    # ---------- Figure 1: log-x + normalized progress ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    best_all = min(h[-1] for h in histories.values())
    worst_final = max(h[-1] for h in histories.values())
    zoom_hi = worst_final + max(4.0, 0.12 * best_all)

    for name, hist in histories.items():
        color, ls = PALETTE.get(name), STYLES.get(name, "-")
        steps = list(range(len(hist)))
        k = _first_best_index(hist)
        for ax, lab in ((ax1, f"{name}  (f*={hist[-1]:.2f}, best @ step {k}, total {len(hist) - 1})"),
                        (ax2, None)):
            ax.step(steps, hist, where="post", color=color, linestyle=ls, linewidth=2, label=lab)
            ax.plot(k, hist[k], "o", color=color, markersize=8,
                    markeredgecolor="white", markeredgewidth=2, zorder=5)

    for ax in (ax1, ax2):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, max(len(h) for h in histories.values()))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlabel("Step / Generation (log scale)", color=INK, fontsize=10)
        _style_axes(ax)

    ax1.set_ylim(ylo, yhi)
    ax1.set_ylabel("Best-so-far commute time f(x) [min]", color=INK, fontsize=10)
    ax1.set_title("Full range", color=INK, fontsize=11, loc="left")

    ax2.set_ylim(best_all - 0.6, zoom_hi)
    ax2.axhline(best_all, color=INK_2, linewidth=1, linestyle=(0, (2, 3)))
    ax2.annotate(f"best found = {best_all:.2f} min", xy=(0.01, best_all), xycoords=("axes fraction", "data"),
                 xytext=(4, -14), textcoords="offset points", ha="left", fontsize=9, color=INK_2)
    ax2.set_title("Zoom on the final approach", color=INK, fontsize=11, loc="left")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title + "   (dot = step where the final best was first reached)",
                 color=INK, fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)

    # ---------- Figure 2: one panel per solver ----------
    names = list(histories.keys())
    cols = 2 if len(names) > 1 else 1
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        hist = histories[name]
        color, ls = PALETTE.get(name), STYLES.get(name, "-")
        steps = list(range(len(hist)))
        k = _first_best_index(hist)
        ax.step(steps, hist, where="post", color=color, linestyle=ls, linewidth=2)
        ax.axvline(k, color=INK_2, linewidth=1, linestyle=(0, (2, 3)))
        ax.annotate(f"best {hist[-1]:.2f} min\nreached at step {k}",
                    xy=(k, hist[k]), xytext=(8, 14), textcoords="offset points",
                    fontsize=9, color=INK)
        ax.set_xlim(0, max(len(hist) - 1, 1))
        ax.set_ylim(ylo, yhi)
        ax.set_title(f"{name}  ({len(hist) - 1} steps)", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Step / Generation", color=INK_2, fontsize=9)
        ax.set_ylabel("f(x) [min]", color=INK_2, fontsize=9)
        _style_axes(ax)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Per-algorithm convergence (each panel uses its own step range)",
                 color=INK, fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        stem, ext = os.path.splitext(save_path)
        fig.savefig(f"{stem}_per_algo{ext or '.png'}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
