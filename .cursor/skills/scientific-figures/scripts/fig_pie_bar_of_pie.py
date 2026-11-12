"""Bar of pie: the "Others" slice unpacked into a stacked bar beside it.

The one slice that a pie always hides is the aggregate. Here the 18 % Others
wedge is exploded, its two cut edges are carried to the top and bottom of a
stacked bar by dashed leaders, and the pale funnel between them makes the
zoom relationship unmistakable. Everything in the bar is a share of the whole
corpus, so the two views share one scale rather than two.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pubstyle import set_pub_style, savefig, lighten

MAIN = [("Others", 18.0, "#8e6bbf"), ("Car", 34.0, "#3b5b92"),
        ("Person", 24.0, "#b23a48"), ("Building", 15.0, "#c9a227"),
        ("Vegetation", 9.0, "#2e7d5b")]
DETAIL = [("Traffic light", 5.2), ("Bench", 4.1), ("Trash bin", 3.4),
          ("Fire hydrant", 3.0), ("Misc.", 2.3)]
TOTAL = 121406
EXPLODE = 0.07
BAR_X0, BAR_X1 = 2.20, 2.92
BAR_Y0, BAR_Y1 = -1.06, 1.06
INK = "#1a1a1a"
SUB = "#555555"


def draw(ax):
    vals = [v for _, v, _ in MAIN]
    cols = [c for _, _, c in MAIN]
    explode = [EXPLODE] + [0.0] * (len(MAIN) - 1)
    wedges, _ = ax.pie(vals, radius=1.0, colors=cols, startangle=90,
                       counterclock=False, explode=explode, center=(0, 0),
                       wedgeprops=dict(edgecolor="white", linewidth=2.4))
    wedges[0].set_linewidth(2.8)

    for w, (name, v, col) in zip(wedges, MAIN):
        a = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        if v < 12.0:
            # too narrow to hold two lines of text: step outside instead
            ax.plot([1.02 * np.cos(a), 1.10 * np.cos(a)],
                    [1.02 * np.sin(a), 1.10 * np.sin(a)], lw=0.7,
                    color="#9a9a9a", zorder=2)
            ax.text(1.15 * np.cos(a), 1.15 * np.sin(a) + 0.055, name,
                    ha="center", va="center", fontsize=7.2, color=INK)
            ax.text(1.15 * np.cos(a), 1.15 * np.sin(a) - 0.075, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8.4, color=INK,
                    fontweight="bold")
            continue
        off = EXPLODE if name == "Others" else 0.0
        r = 0.62 + off
        ax.text(r * np.cos(a), r * np.sin(a) + 0.055, name, ha="center",
                va="center", fontsize=7.2, color="white")
        ax.text(r * np.cos(a), r * np.sin(a) - 0.075, f"{v:.0f}%",
                ha="center", va="center", fontsize=8.4, color="white",
                fontweight="bold")

    # ---- connector funnel from the two cut edges of the Others wedge -------
    w0 = wedges[0]
    bis = np.deg2rad((w0.theta1 + w0.theta2) / 2.0)
    ox, oy = EXPLODE * np.cos(bis), EXPLODE * np.sin(bis)
    a_hi, a_lo = np.deg2rad(w0.theta2), np.deg2rad(w0.theta1)
    p_hi = (np.cos(a_hi) + ox, np.sin(a_hi) + oy)
    p_lo = (np.cos(a_lo) + ox, np.sin(a_lo) + oy)
    ax.add_patch(Polygon([p_hi, (BAR_X0, BAR_Y1), (BAR_X0, BAR_Y0), p_lo],
                         closed=True, facecolor="#f2f2f2", edgecolor="none",
                         zorder=1))
    for p, ybar in ((p_hi, BAR_Y1), (p_lo, BAR_Y0)):
        ax.plot([p[0], BAR_X0], [p[1], ybar], lw=0.9, ls=(0, (3.5, 2.5)),
                color="#8a8a8a", zorder=2)

    # ---- the exploded stack ------------------------------------------------
    span = BAR_Y1 - BAR_Y0
    group = sum(v for _, v in DETAIL)
    top = BAR_Y1
    for i, (name, v) in enumerate(DETAIL):
        h = v / group * span
        col = lighten("#8e6bbf", 0.06 + 0.60 * i / (len(DETAIL) - 1))
        ax.add_patch(plt.Rectangle((BAR_X0, top - h), BAR_X1 - BAR_X0, h,
                                   facecolor=col, edgecolor="white",
                                   linewidth=1.8, zorder=3))
        yc = top - h / 2.0
        ax.text((BAR_X0 + BAR_X1) / 2.0, yc, f"{v / group * 100:.0f}%",
                ha="center", va="center", fontsize=6.6, fontweight="bold",
                color="white" if i < 3 else INK, zorder=4)
        ax.plot([BAR_X1, BAR_X1 + 0.09], [yc, yc], lw=0.7, color="#c4c4c4",
                zorder=2)
        ax.text(BAR_X1 + 0.14, yc + 0.055, name, ha="left", va="center",
                fontsize=7.0, color=INK)
        ax.text(BAR_X1 + 0.14, yc - 0.075,
                f"{v:.1f}% of corpus  ({int(v / 100 * TOTAL):,})", ha="left",
                va="center", fontsize=6.3, color=SUB)
        top -= h

    ax.text((BAR_X0 + BAR_X1) / 2.0, BAR_Y1 + 0.20, "Inside \"Others\"",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=INK)
    ax.text((BAR_X0 + BAR_X1) / 2.0, BAR_Y1 + 0.09,
            "18.0% of the corpus, 5 tail classes", ha="center", va="center",
            fontsize=6.4, color=SUB)
    ax.text(0, -1.30, f"{TOTAL:,} annotated instances", ha="center",
            va="center", fontsize=7.0, color=SUB)
    ax.set_xlim(-1.30, 4.55)
    ax.set_ylim(-1.46, 1.46)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    set_pub_style()
    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_axes([0.015, 0.085, 0.97, 0.79])
    draw(ax)
    savefig(fig, "330_pie_bar_of_pie.png")


if __name__ == "__main__":
    main()
