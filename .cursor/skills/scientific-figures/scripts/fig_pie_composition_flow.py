"""Composition flow: two donuts joined by tapering Sankey ribbons.

A pair of pies side by side forces the reader to hold five angles in memory.
Here each category leaves the 2018 donut as a ribbon whose width is its share,
glides across the gap on a smoothstep curve and lands at its 2024 width, so
growth and decline are read as the ribbon narrowing or widening rather than as
an angle difference. The strip below states the same deltas numerically.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, style_axes

CATS = ["Car", "Person", "Building", "Vegetation", "Others"]
COLORS = ["#3b5b92", "#b23a48", "#c9a227", "#2e7d5b", "#b3b3b3"]
Y2018 = np.array([34.0, 21.0, 18.0, 15.0, 12.0])
Y2024 = np.array([27.0, 25.0, 16.0, 19.0, 13.0])
N2018, N2024 = 48_300, 121_406

CL, CR = 0.0, 5.30            # donut centres
XL0, XL1 = 1.16, 1.32         # left stub column
XR0, XR1 = 3.98, 4.14         # right stub column
HALF = 1.06                   # half height of the flow column
INK = "#1a1a1a"
SUB = "#555555"


def panel_label(fig, ax, text, dx=-0.58, dy=0.04):
    bb = ax.get_position()
    fig.text(bb.x0 + dx / fig.get_figwidth(),
             bb.y1 + dy / fig.get_figheight(),
             text, fontsize=11, fontweight="bold", ha="left", va="bottom")


def stack_edges(shares):
    """Top-down segment edges for a 100 % column of half-height HALF."""
    cum = np.concatenate([[0.0], np.cumsum(shares)]) / shares.sum()
    return HALF - cum * 2.0 * HALF


def draw_donut(ax, centre, shares, year, total):
    wedges, _ = ax.pie(
        shares, radius=1.0, colors=COLORS, startangle=90, counterclock=False,
        center=centre, frame=True,
        wedgeprops=dict(width=0.40, edgecolor="white", linewidth=2.2))
    for w, s, c in zip(wedges, shares, COLORS):
        a = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        ax.text(centre[0] + 0.80 * np.cos(a), centre[1] + 0.80 * np.sin(a),
                f"{s:.0f}%", ha="center", va="center", fontsize=7.0,
                fontweight="bold", color="white", zorder=5)
    ax.text(centre[0], centre[1] + 0.10, year, ha="center", va="center",
            fontsize=12.0, fontweight="bold", color=INK, zorder=5)
    ax.text(centre[0], centre[1] - 0.11, f"{total:,}", ha="center",
            va="center", fontsize=6.6, color=SUB, zorder=5)
    ax.text(centre[0], centre[1] - 0.24, "instances", ha="center",
            va="center", fontsize=6.0, color="#8a8a8a", zorder=5)


def draw_flow(ax):
    draw_donut(ax, (CL, 0.0), Y2018, "2018", N2018)
    draw_donut(ax, (CR, 0.0), Y2024, "2024", N2024)

    eL, eR = stack_edges(Y2018), stack_edges(Y2024)
    t = np.linspace(0, 1, 160)
    s = 3 * t ** 2 - 2 * t ** 3
    x = XL1 + (XR0 - XL1) * t

    for i, col in enumerate(COLORS):
        ax.add_patch(plt.Rectangle((XL0, eL[i + 1]), XL1 - XL0,
                                   eL[i] - eL[i + 1], facecolor=col,
                                   edgecolor="white", lw=1.2, zorder=4))
        ax.add_patch(plt.Rectangle((XR0, eR[i + 1]), XR1 - XR0,
                                   eR[i] - eR[i + 1], facecolor=col,
                                   edgecolor="white", lw=1.2, zorder=4))
        top = eL[i] + (eR[i] - eL[i]) * s
        bot = eL[i + 1] + (eR[i + 1] - eL[i + 1]) * s
        ax.fill_between(x, bot, top, facecolor=col, alpha=0.42,
                        edgecolor="white", linewidth=0.9, zorder=3)

        ym = 0.5 * (top[80] + bot[80])
        d = Y2024[i] - Y2018[i]
        ax.text(x[80], ym + 0.072, CATS[i], ha="center", va="center",
                fontsize=7.2, fontweight="bold", color=INK, zorder=6)
        ax.text(x[80], ym - 0.078,
                f"{Y2018[i]:.0f}%  \u2192  {Y2024[i]:.0f}%   ({d:+.0f} pp)",
                ha="center", va="center", fontsize=6.6, color="#333333",
                zorder=6)

    ax.set_xlim(-1.30, 6.60)
    ax.set_ylim(-1.42, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_delta(ax):
    d = Y2024 - Y2018
    y = np.arange(len(CATS))[::-1]
    ax.barh(y, d, height=0.60, color=COLORS, edgecolor="white", lw=1.2,
            zorder=3)
    for yi, v in zip(y, d):
        off = 0.35 if v >= 0 else -0.35
        ax.text(v + off, yi, f"{v:+.0f} pp", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.6, color=INK)
    ax.axvline(0, lw=0.9, color="#333333", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(CATS, fontsize=7.0)
    ax.set_ylim(-0.65, len(CATS) - 0.35)
    ax.set_xlim(-8.6, 5.6)
    ax.set_xlabel("change in share, 2018 to 2024 (percentage points)",
                  fontsize=6.8)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Share change by category", fontsize=8.5, fontweight="bold",
                 color=INK, pad=5, loc="left")


def main():
    set_pub_style()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.55],
                          left=0.075, right=0.955, top=0.905, bottom=0.115,
                          hspace=0.32)
    ax_flow = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[1, 0])

    bb = ax_delta.get_position()
    fig.patches.append(plt.Rectangle(
        (bb.x0 - 0.062, bb.y0 - 0.070), bb.width + 0.085, bb.height + 0.118,
        transform=fig.transFigure, facecolor="#f5f5f5", edgecolor="none",
        zorder=-1))
    ax_delta.set_facecolor("none")

    draw_flow(ax_flow)
    draw_delta(ax_delta)

    panel_label(fig, ax_flow, "(a)")
    panel_label(fig, ax_delta, "(b)")
    savefig(fig, "332_pie_composition_flow.png")


if __name__ == "__main__":
    main()
