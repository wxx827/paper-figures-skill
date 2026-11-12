"""Tensor-shape flow through a residual classifier.

Every stage is a cuboid whose height tracks the spatial resolution and whose
width tracks the channel count, so the classic "tall and thin becomes short
and fat" silhouette is readable at a glance.  Operations are named above the
block they produce, shapes and cost sit underneath, and two reduced-size
panels below carry the cumulative budget and the share of parameters versus
the share of compute.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from pubstyle import set_pub_style, PALETTES, savefig

ENC = "#3b5b92"
DEC = "#2e7d5b"
ATT = "#c9a227"
LOSS = "#b23a48"
DATA = "#8c8c8c"
AUX = "#8e6bbf"
INK = "#24303c"
MUTED = "#6f7783"
SOFT = "#f2f4f7"
LINE = "#dde2e8"

R = 0.055
DASH = (0, (4, 2.6))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def shade(color, amount):
    r, g, b = mpl.colors.to_rgb(color)
    if amount >= 0:
        return (r + (1 - r) * amount, g + (1 - g) * amount,
                b + (1 - b) * amount)
    f = 1.0 + amount
    return (r * f, g * f, b * f)


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.2, ls="-", ms=12, zorder=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def panel(ax, x, y, w, h, fc=SOFT, ec=LINE, lw=0.8, ls="-", zorder=1,
          rounding=0.08):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))


def feature_map(ax, x, y, w, h, color=ENC, d=0.22, ec="white", lw=0.7,
                zorder=3):
    """Cuboid: width = channels, height = spatial extent, d = depth skew."""
    dx, dy = 0.52 * d, 0.36 * d
    ax.add_patch(Polygon([(x, y - h / 2), (x + w, y - h / 2),
                          (x + w, y + h / 2), (x, y + h / 2)], closed=True,
                         facecolor=color, edgecolor=ec, lw=lw, zorder=zorder))
    ax.add_patch(Polygon([(x, y + h / 2), (x + w, y + h / 2),
                          (x + w + dx, y + h / 2 + dy), (x + dx,
                                                         y + h / 2 + dy)],
                         closed=True, facecolor=shade(color, 0.28),
                         edgecolor=ec, lw=lw, zorder=zorder))
    ax.add_patch(Polygon([(x + w, y - h / 2), (x + w + dx, y - h / 2 + dy),
                          (x + w + dx, y + h / 2 + dy), (x + w, y + h / 2)],
                         closed=True, facecolor=shade(color, -0.18),
                         edgecolor=ec, lw=lw, zorder=zorder))
    return dict(l=(x, y), r=(x + w + dx, y), top=y + h / 2 + dy,
                bot=y - h / 2, cx=x + (w + dx) / 2)


def titles(ax, x, y, main, sub=None):
    ax.text(x, y, main, fontsize=13.5, fontweight="bold", color=INK,
            ha="left", va="center")
    if sub:
        ax.text(x, y - 0.32, sub, fontsize=8.6, color=MUTED, ha="left",
                va="center")


# ---- figure ----------------------------------------------------------------
W, H = 13.4, 7.25
YROW = 4.72
PITCH = 1.38
X0 = 0.85

STAGES = [
    # name, op above, h, w, shape, params, flops, colour
    ("input", "RGB frame", 1.30, 0.10, "224×224×3", "—", "—", DATA),
    ("conv1", "conv 7×7, /2", 1.06, 0.30, "112×112×64", "9.4 k", "118 M",
     ENC),
    ("pool1", "max-pool 3×3, /2", 0.86, 0.30, "56×56×64", "0", "0", ENC),
    ("res2", "bottleneck ×3", 0.86, 0.52, "56×56×256", "0.22 M", "680 M",
     ATT),
    ("res3", "bottleneck ×4, /2", 0.68, 0.66, "28×28×512", "1.22 M", "950 M",
     ATT),
    ("res4", "bottleneck ×6, /2", 0.52, 0.80, "14×14×1024", "7.10 M",
     "1.44 G", ATT),
    ("res5", "bottleneck ×3, /2", 0.38, 0.94, "7×7×2048", "14.96 M", "810 M",
     ATT),
    ("pool5", "global avg-pool", 0.16, 0.94, "1×1×2048", "0", "2 M", AUX),
    ("fc", "linear 2048→1000", 0.16, 0.72, "1×1×1000", "2.05 M", "2 M", DEC),
]

FAM = [("stem", ENC, 0.0004, 0.030), ("res2–3", shade(ATT, 0.30), 0.056,
                                      0.400),
       ("res4–5", ATT, 0.860, 0.550), ("head", DEC, 0.084, 0.020)]


def main_row(ax):
    boxes = []
    for k, (name, op, h, w, shape, par, fl, col) in enumerate(STAGES):
        x = X0 + PITCH * k
        b = feature_map(ax, x, YROW, w, h, col)
        boxes.append(b)
        ax.text(b["cx"], 5.78, op, ha="center", va="center", fontsize=6.9,
                color=col if col != DATA else MUTED)
        ax.text(b["cx"], 3.92, name, ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=INK)
        ax.text(b["cx"], 3.72, shape, ha="center", va="center", fontsize=6.6,
                color=MUTED)
        ax.text(b["cx"], 3.54, f"{par} par", ha="center", va="center",
                fontsize=6.2, color=MUTED)
        ax.text(b["cx"], 3.38, f"{fl} FLOPs", ha="center", va="center",
                fontsize=6.2, color=MUTED)
    for a, b in zip(boxes[:-1], boxes[1:]):
        arrow(ax, (a["r"][0] + 0.05, YROW), (b["l"][0] - 0.05, YROW),
              color="#8f9aa6", lw=1.2, ms=12)

    ax.annotate("", xy=(12.60, 6.30), xytext=(0.85, 6.30),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#c3ccd5",
                                shrinkA=0, shrinkB=0))
    ax.text(6.70, 6.44, "resolution ÷ 32   ·   channels × 683   ·   "
            "one forward pass", ha="center", va="center", fontsize=7.4,
            color=MUTED)


def budget_panel(fig, ax):
    panel(ax, 4.02, 1.72, 6.34, 2.12)
    ax.text(0.95, 2.60, "Cumulative budget along the network", fontsize=8.2,
            fontweight="bold", color=INK, ha="left", va="center", zorder=4)
    axb = fig.add_axes([1.30 / W, 1.02 / H, 5.30 / W, 1.24 / H])
    par = np.array([0, 0.0094, 0, 0.22, 1.22, 7.10, 14.96, 0, 2.05])
    flo = np.array([0, 0.118, 0, 0.680, 0.950, 1.440, 0.810, 0.002, 0.002])
    x = np.arange(len(STAGES))
    axb.plot(x, np.cumsum(par), color=ENC, lw=1.6, marker="o", ms=3.2,
             label="parameters (M)")
    axb.set_ylabel("params (M)", fontsize=6.4, color=ENC, labelpad=1.5)
    axb.tick_params(labelsize=6.0, length=2, width=0.6, pad=1.5)
    axb.set_xticks(x)
    axb.set_xticklabels([s[0] for s in STAGES], fontsize=6.0, rotation=30,
                        ha="right")
    axb.set_ylim(0, 28)
    ax2 = axb.twinx()
    ax2.plot(x, np.cumsum(flo), color=ATT, lw=1.6, marker="s", ms=3.0,
             label="FLOPs (G)")
    ax2.set_ylabel("FLOPs (G)", fontsize=6.4, color=ATT, labelpad=1.5)
    ax2.tick_params(labelsize=6.0, length=2, width=0.6, pad=1.5)
    ax2.set_ylim(0, 4.6)
    ax2.spines["top"].set_visible(False)
    for sp in list(axb.spines.values()) + list(ax2.spines.values()):
        sp.set_linewidth(0.6)
    axb.set_facecolor("none")
    h1, l1 = axb.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axb.legend(h1 + h2, l1 + l2, fontsize=6.2, loc="upper left",
               handlelength=1.3, borderaxespad=0.3)


def share_panel(fig, ax):
    panel(ax, 10.24, 1.72, 5.62, 2.12)
    ax.text(7.60, 2.60, "Where the cost lives", fontsize=8.2,
            fontweight="bold", color=INK, ha="left", va="center", zorder=4)
    axs = fig.add_axes([8.05 / W, 1.10 / H, 4.55 / W, 1.02 / H])
    rows = ["parameters", "FLOPs"]
    left = np.zeros(2)
    for name, col, p, f in FAM:
        vals = np.array([p, f]) * 100
        axs.barh([1, 0], vals, left=left, height=0.52, color=col,
                 edgecolor="white", lw=0.6, label=name)
        for k, (v, l0) in enumerate(zip(vals, left)):
            if v > 6:
                axs.text(l0 + v / 2, 1 - k, f"{v:.0f}%", ha="center",
                         va="center", fontsize=6.0, color="white")
        left = left + vals
    axs.set_yticks([1, 0])
    axs.set_yticklabels(rows, fontsize=6.2)
    axs.set_xlim(0, 100)
    axs.set_xticks([0, 25, 50, 75, 100])
    axs.tick_params(labelsize=6.0, length=2, width=0.6, pad=1.5)
    axs.set_xlabel("share of total (%)", fontsize=6.2, labelpad=1.0)
    axs.legend(fontsize=6.0, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.34), handlelength=1.0,
               columnspacing=1.0, borderaxespad=0.0)
    for sp in axs.spines.values():
        sp.set_linewidth(0.6)
    axs.set_facecolor("none")


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    titles(ax, 0.85, 6.96,
           "Tensor shapes and cost along a residual classifier",
           "Block height is spatial resolution, block width is channel "
           "count; late stages hold the parameters, early stages burn the "
           "compute")
    main_row(ax)
    budget_panel(fig, ax)
    share_panel(fig, ax)
    savefig(fig, "269_net_tensor_shapes.png")


if __name__ == "__main__":
    main()
