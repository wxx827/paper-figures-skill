"""Circular (polar) grouped barplot grid.

A 3x3 grid of radial barplots, one per climate zone.  Each circle carries the
twelve months clockwise from the top, and three concentric grouped bars for
the three land-cover classes.  A light radial grid gives the plot a readable
scale, and the annual mean sits in the open hub of each circle.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig

COVERS = [("Forest", "#2e7d5b"), ("Grassland", "#d98c5f"),
          ("Cropland", "#3d6b8e")]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ZONES = ["Boreal", "Cold temperate", "Warm temperate", "Subtropical humid",
         "Subtropical dry", "Tropical humid", "Semi-arid", "Arid",
         "Alpine"]
INNER = 1.0
RMAX = 4.6
RTICKS = [1.0, 2.0, 3.0, 4.0]


def circular_panel(ax, rng, title, amp, phase, show_scale):
    n = len(MONTHS)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n / 3.6
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    means = []
    for k, (name, col) in enumerate(COVERS):
        season = np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False) - phase)
        vals = np.clip(amp[k] * (0.55 + 0.45 * season)
                       + rng.normal(0, 0.09, n), 0.05, None)
        means.append(vals.mean())
        ax.bar(theta + (k - 1) * width, vals, width=width * 0.92,
               bottom=INNER, color=col, edgecolor="white", linewidth=0.35,
               alpha=0.95, zorder=3)

    ax.set_ylim(0, RMAX)
    ax.set_xticks(theta)
    ax.set_xticklabels(MONTHS, fontsize=6.0, color="#444444")
    ax.tick_params(axis="x", pad=-1.0)
    ax.set_yticks([INNER + t - 1.0 for t in RTICKS])
    if show_scale:
        ax.set_yticklabels([f"{t - 1.0:.0f}" for t in RTICKS], fontsize=6.0,
                           color="#7a7a7a")
        ax.set_rlabel_position(103)
        for lab in ax.get_yticklabels():
            lab.set_bbox(dict(boxstyle="square,pad=0.1", fc="white",
                              ec="none", alpha=0.75))
    else:
        ax.set_yticklabels([])
    ax.grid(axis="y", color="#dedede", lw=0.45, zorder=0)
    ax.grid(axis="x", color="#f0f0f0", lw=0.4, zorder=0)
    ax.spines["polar"].set_visible(False)

    ax.text(0.5, 0.5, f"{np.mean(means):.1f}", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#3a3a3a",
            transform=ax.transAxes)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=9)


def main():
    set_pub_style()
    rng = np.random.default_rng(4)
    fig, axes = plt.subplots(3, 3, figsize=(9.4, 9.8),
                             subplot_kw=dict(projection="polar"))
    amps = [(2.9, 1.6, 1.1), (2.6, 1.9, 1.4), (2.3, 2.0, 2.2),
            (3.0, 1.5, 2.4), (1.9, 2.2, 2.0), (3.2, 1.3, 1.8),
            (1.2, 2.4, 1.5), (0.7, 1.4, 0.9), (1.5, 2.6, 0.6)]
    phases = np.linspace(0.0, 0.9, 9)
    for i, (ax, zone) in enumerate(zip(axes.ravel(), ZONES)):
        circular_panel(ax, rng, zone, amps[i], phases[i], show_scale=(i == 0))

    handles = [Patch(facecolor=c, edgecolor="white", label=n)
               for n, c in COVERS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.018), columnspacing=2.0,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0.055, 1, 0.955), h_pad=1.6, w_pad=1.0)
    savefig(fig, "09_circular_bars.png")


if __name__ == "__main__":
    main()
