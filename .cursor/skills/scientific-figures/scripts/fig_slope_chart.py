"""Slope chart: start-to-end change with de-collided end labels.

Two time points, ten cities, coloured by the direction of change. Labels on
both sides are spread with a minimum vertical gap so that near-ties stay
readable, the city name carries its 2010 value on the left and its 2020 value
on the right, and the cohort mean is drawn as a grey reference slope.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, PALETTES, savefig

CITIES = ["Beijing", "Shanghai", "Guangzhou", "Chengdu", "Wuhan",
          "Hangzhou", "Nanjing", "Xi'an", "Chongqing", "Shenzhen"]
T0, T1 = "2010", "2020"
UP, DOWN = "#2a7f7a", "#b23a48"
REF = "#9a9a9a"


def spread(values, gap):
    """Push labels apart until every neighbour is at least `gap` apart."""
    order = np.argsort(values)
    y = values[order].astype(float).copy()
    for _ in range(200):
        moved = False
        for i in range(len(y) - 1):
            d = y[i + 1] - y[i]
            if d < gap:
                shift = (gap - d) / 2
                y[i] -= shift
                y[i + 1] += shift
                moved = True
        if not moved:
            break
    out = np.empty_like(y)
    out[order] = y
    return out


def main():
    set_pub_style()
    rng = np.random.default_rng(491)
    v0 = rng.uniform(28, 78, len(CITIES))
    v1 = np.clip(v0 + rng.normal(6, 16, len(CITIES)), 5, 98)

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    span = max(v0.max(), v1.max()) - min(v0.min(), v1.min())
    gap = span * 0.042
    ly = spread(v0, gap)
    ry = spread(v1, gap)

    ax.plot([0, 1], [v0.mean(), v1.mean()], color=REF, lw=1.6, ls="--",
            zorder=1)
    for i, city in enumerate(CITIES):
        col = UP if v1[i] >= v0[i] else DOWN
        ax.plot([0, 1], [v0[i], v1[i]], color=col, lw=1.8, alpha=0.8, zorder=2)
        ax.scatter([0, 1], [v0[i], v1[i]], color=col, s=38, zorder=3,
                   edgecolors="white", linewidths=0.9)
        ax.plot([-0.035, -0.012], [ly[i], v0[i]], color="#dddddd", lw=0.6,
                zorder=1)
        ax.plot([1.012, 1.035], [v1[i], ry[i]], color="#dddddd", lw=0.6,
                zorder=1)
        ax.text(-0.045, ly[i], f"{city}   {v0[i]:.0f}", ha="right",
                va="center", fontsize=8, color="#444444")
        ax.text(1.045, ry[i], f"{v1[i]:.0f}  ({v1[i] - v0[i]:+.1f})",
                ha="left", va="center", fontsize=8, color=col,
                fontweight="bold")

    ax.text(1.045, np.mean([v0.mean(), v1.mean()]) + span * 0.0, "",
            fontsize=7)
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(min(ly.min(), ry.min()) - span * 0.08,
                max(ly.max(), ry.max()) + span * 0.10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([T0, T1], fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Urban greenness index (0-100)", fontsize=9)
    # x=-0.22 put the title 0.98 in to the left of the canvas, which made the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).
    # Anchored on the axes spine it still reads as a left-aligned title.
    ax.set_title("Change in urban greenness, 2010-2020", fontsize=10.5,
                 fontweight="bold", loc="left", x=0.0, pad=10)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", lw=0.5, alpha=0.35)

    handles = [Line2D([], [], color=UP, lw=1.8, label="Increase"),
               Line2D([], [], color=DOWN, lw=1.8, label="Decrease"),
               Line2D([], [], color=REF, lw=1.6, ls="--", label="Cohort mean")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    savefig(fig, "49_slope_chart.png")


if __name__ == "__main__":
    main()
