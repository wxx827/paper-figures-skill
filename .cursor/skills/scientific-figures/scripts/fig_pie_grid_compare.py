"""Eight donuts on one shared scale: composition across capture cities.

Small multiples only work if nothing but the data changes, so every donut here
uses the same five categories, the same colour assignment, the same 12 o'clock
start angle and the same clockwise order. Grey ticks outside each ring mark the
pooled-average slice boundaries, which turns "these look similar" into a
readable deviation: where a coloured boundary sits away from its tick, that
city departs from the corpus average.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig

CATS = ["Vehicle", "Person", "Building", "Vegetation", "Other"]
COLORS = ["#3b5b92", "#b23a48", "#c9a227", "#2e7d5b", "#b3b3b3"]
CITIES = ["Berlin", "Tokyo", "Boston", "Nairobi",
          "Lima", "Delhi", "Sydney", "Oslo"]
BASE = np.array([0.32, 0.24, 0.19, 0.17, 0.08])
INK = "#1a1a1a"
SUB = "#555555"


def make_data(rng):
    comp = rng.dirichlet(BASE * 55.0, size=len(CITIES))
    totals = rng.integers(84, 312, size=len(CITIES)) * 100
    return comp, totals


def draw_donut(ax, frac, total, name, pooled_bounds, flag=""):
    ax.pie(frac, radius=1.0, colors=COLORS, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.40, edgecolor="white", linewidth=1.9))
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(1.13 * np.cos(th), 1.13 * np.sin(th), lw=0.5, color="#e0e0e0",
            zorder=1)
    for b in pooled_bounds:
        a = np.deg2rad(90.0 - b * 360.0)
        ax.plot([1.05 * np.cos(a), 1.21 * np.cos(a)],
                [1.05 * np.sin(a), 1.21 * np.sin(a)],
                lw=1.0, color="#8a8a8a", zorder=2, solid_capstyle="butt")
    ax.text(0, 0.09, f"{total / 1000:.1f}k", ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=INK)
    ax.text(0, -0.11, "instances", ha="center", va="center", fontsize=6.0,
            color="#8a8a8a")
    ax.text(0, -1.40, name, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=INK)
    ax.text(0, -1.62, f"vehicle share {frac[0] * 100:.0f}%{flag}",
            ha="center", va="center", fontsize=6.4,
            color=INK if flag else SUB,
            fontweight="bold" if flag else "normal")
    ax.set_xlim(-1.26, 1.26)
    ax.set_ylim(-1.76, 1.26)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    set_pub_style()
    rng = np.random.default_rng(326)
    comp, totals = make_data(rng)
    pooled = comp.mean(axis=0)
    pooled_bounds = np.cumsum(pooled)[:-1]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 4, left=0.02, right=0.98, top=0.915, bottom=0.13,
                          wspace=0.04, hspace=0.10)
    hi, lo = int(np.argmax(comp[:, 0])), int(np.argmin(comp[:, 0]))
    for i, name in enumerate(CITIES):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        flag = "  highest" if i == hi else ("  lowest" if i == lo else "")
        draw_donut(ax, comp[i], totals[i], name, pooled_bounds, flag)

    handles = [Patch(facecolor=c, edgecolor="white", lw=0.8,
                     label=f"{n}  {p * 100:.0f}%")
               for n, c, p in zip(CATS, COLORS, pooled)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.055), columnspacing=1.8,
               handlelength=1.3, handletextpad=0.55)
    savefig(fig, "326_pie_grid_compare.png")


if __name__ == "__main__":
    main()
