"""Sankey flow diagram: multi-stage node-link flows with bezier links.

Pure-matplotlib Sankey (no plotly).  Nodes are vertical bars arranged in
ordered stages, flows are filled bezier bands whose width encodes magnitude
and whose colour follows the source node.  The flow table is explicit, so the
node heights, the labels and the totals all agree.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pubstyle import set_pub_style, savefig

STAGES = [
    ("Source", ["Precipitation", "Runoff", "Fertiliser"]),
    ("Receiving sector", ["Cropland", "Forest", "Urban"]),
    ("Fate", ["Retained", "Lost to water"]),
]
COLORS = {
    "Precipitation": "#3d6b8e", "Runoff": "#2e7d5b", "Fertiliser": "#8c2f2f",
    "Cropland": "#c9a227", "Forest": "#1f6b4d", "Urban": "#a4503f",
    "Retained": "#2a8f86", "Lost to water": "#8e6bbf",
}
# flows in Gg N per year
FLOWS = [
    np.array([[14, 22, 6],       # precipitation -> sectors
              [8, 4, 6],         # runoff        -> sectors
              [22, 2, 6]],       # fertiliser    -> sectors
             dtype=float),
    np.array([[20, 24],          # cropland -> retained / lost
              [22, 6],           # forest
              [5, 13]],          # urban
             dtype=float),
]
UNIT = "Gg N yr$^{-1}$"
HEIGHT = 10.0
BARW = 0.22
XGAP = 3.2


def _cubic(p0, p1, p2, p3, t):
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def band(ax, x0, y0a, y0b, x1, y1a, y1b, color, n=60):
    t = np.linspace(0, 1, n)[:, None]
    xm = (x0 + x1) / 2
    top = np.column_stack([_cubic(x0, xm, xm, x1, t).ravel(),
                           _cubic(y0a, y0a, y1a, y1a, t).ravel()])
    bot = np.column_stack([_cubic(x1, xm, xm, x0, t).ravel(),
                           _cubic(y1b, y1b, y0b, y0b, t).ravel()])
    ax.add_patch(patches.Polygon(np.vstack([top, bot]), closed=True,
                                 facecolor=color, alpha=0.42, lw=0.4,
                                 edgecolor="white", zorder=2))


def main():
    set_pub_style()
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.axis("off")

    totals = [FLOWS[0].sum(axis=1), FLOWS[0].sum(axis=0), FLOWS[1].sum(axis=0)]
    grand = totals[0].sum()
    scale = HEIGHT / grand

    pos = {}
    for s, (_hdr, names) in enumerate(STAGES):
        sizes = totals[s] * scale
        gap = (HEIGHT + 1.2 - sizes.sum()) / (len(names) + 1)
        y = gap
        for k, nm in enumerate(names):
            x = s * XGAP
            pos[(s, k)] = [x, y, sizes[k]]
            ax.add_patch(patches.Rectangle((x, y), BARW, sizes[k],
                                           facecolor=COLORS[nm], ec="white",
                                           lw=0.6, zorder=4))
            if s == 0:
                tx, ha = x - 0.12, "right"
            elif s == len(STAGES) - 1:
                tx, ha = x + BARW + 0.12, "left"
            else:
                tx, ha = x + BARW / 2, "center"
            va = "center"
            if s == 1:
                ty, va = y + sizes[k] + 0.16, "bottom"
            else:
                ty = y + sizes[k] / 2
            box = (dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                        alpha=0.8) if s == 1 else None)
            ax.text(tx, ty, f"{nm}\n{totals[s][k]:.0f}", ha=ha, va=va,
                    fontsize=8, color=COLORS[nm], fontweight="bold",
                    linespacing=1.3, zorder=6, bbox=box)
            y += sizes[k] + gap

    for s, F in enumerate(FLOWS):
        src_off = {i: pos[(s, i)][1] for i in range(F.shape[0])}
        dst_off = {j: pos[(s + 1, j)][1] for j in range(F.shape[1])}
        for i in range(F.shape[0]):
            for j in range(F.shape[1]):
                v = F[i, j]
                if v <= 0:
                    continue
                w = v * scale
                x0 = pos[(s, i)][0] + BARW
                x1 = pos[(s + 1, j)][0]
                band(ax, x0, src_off[i], src_off[i] + w,
                     x1, dst_off[j], dst_off[j] + w,
                     COLORS[STAGES[s][1][i]])
                src_off[i] += w
                dst_off[j] += w

    for s, (hdr, _names) in enumerate(STAGES):
        ax.text(s * XGAP + BARW / 2, HEIGHT + 1.75, hdr, ha="center",
                va="bottom", fontsize=9, fontweight="bold", color="#333333")

    ax.set_xlim(-1.55, (len(STAGES) - 1) * XGAP + 1.75)
    ax.set_ylim(-0.5, HEIGHT + 2.4)
    ax.set_title(f"Nitrogen budget: source to sector to fate "
                 f"({grand:.0f} {UNIT})", fontsize=10.5, fontweight="bold",
                 y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, "22_sankey.png")


if __name__ == "__main__":
    main()
