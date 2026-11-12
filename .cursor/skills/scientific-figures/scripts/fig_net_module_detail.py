"""Zoom from the whole backbone into one squeeze-and-excitation bottleneck.

The left card is the network at a glance with the studied stage highlighted;
two dashed guide lines open it into the right card, where the bottleneck is
drawn operator by operator: the 1x1 / 3x3 / 1x1 trunk, the squeeze-excite
branch that produces a per-channel gate, the element-wise product and the
identity residual add.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
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
LW = 0.9
DASH = (0, (4, 2.6))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def anchors(x, y, w, h):
    return dict(x=x, y=y, w=w, h=h, c=(x, y), l=(x - w / 2, y),
                r=(x + w / 2, y), t=(x, y + h / 2), b=(x, y - h / 2))


def block(ax, x, y, w, h, label="", color=ENC, sub=None, ec="white",
          text_color="white", fontsize=8, sub_color=None, lw=LW, ls="-",
          alpha=1.0, zorder=3, rounding=R, hatch=None, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=color,
        edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha, hatch=hatch,
        zorder=zorder))
    if label:
        ax.text(x, y + (0.105 if sub else 0.0), label, ha="center",
                va="center", color=text_color, fontsize=fontsize,
                fontweight=weight, zorder=zorder + 2, linespacing=1.35)
    if sub:
        ax.text(x, y - 0.115, sub, ha="center", va="center",
                color=sub_color or text_color, fontsize=6.5, alpha=0.92,
                zorder=zorder + 2)
    return anchors(x, y, w, h)


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.2, ls="-", ms=12, zorder=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def route(ax, pts, color=INK, lw=1.2, ls="-", ms=12, zorder=6):
    pts = [tuple(p) for p in pts]
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color, lw=lw, ls=ls, solid_capstyle="round",
                zorder=zorder)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=ms,
          zorder=zorder)


def op_circle(ax, x, y, sym="+", r=0.145, color=INK, fc="white", fontsize=9.5,
              zorder=7):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=color, lw=1.1,
                        zorder=zorder))
    ax.text(x, y + 0.005, sym, ha="center", va="center", fontsize=fontsize,
            color=color, zorder=zorder + 1)
    return dict(x=x, y=y, c=(x, y), l=(x - r, y), r=(x + r, y),
                t=(x, y + r), b=(x, y - r))


def tensor_label(ax, x, y, text, color=MUTED, fontsize=6.5, ha="center",
                 va="center", **kw):
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=fontsize,
                   **kw)


def panel(ax, x, y, w, h, fc=SOFT, ec=LINE, lw=0.8, ls="-", zorder=1,
          rounding=0.08):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))
    return anchors(x, y, w, h)


def titles(ax, x, y, main, sub=None):
    ax.text(x, y, main, fontsize=13.5, fontweight="bold", color=INK,
            ha="left", va="center")
    if sub:
        ax.text(x, y - 0.32, sub, fontsize=8.6, color=MUTED, ha="left",
                va="center")


# ---- figure ----------------------------------------------------------------
W, H = 13.4, 6.9
PY, PH = 3.90, 3.70
LX, LW_ = 2.30, 3.70
RX, RW = 8.80, 8.70
SPINE = 3.85
SEY = 2.52


def overview(ax):
    panel(ax, LX, PY, LW_, PH, fc="#fbfcfd", ec="#d5dce4")
    ax.text(LX - LW_ / 2 + 0.18, 5.55, "(a)  Backbone at a glance",
            fontsize=8.6, fontweight="bold", color=INK, ha="left",
            va="center")
    names = [("stem  conv 7×7", ENC), ("stage 1  ×3", ENC),
             ("stage 2  ×4", ATT), ("stage 3  ×6", ENC),
             ("head  GAP + FC", DEC)]
    ys = [2.66, 3.28, 3.90, 4.52, 5.14]
    boxes = []
    for (name, col), y in zip(names, ys):
        boxes.append(block(ax, LX, y, 2.06, 0.44, name, col, fontsize=7.6))
    for a, b in zip(boxes[:-1], boxes[1:]):
        arrow(ax, (LX, a["y"] + 0.25), (LX, b["y"] - 0.25), color=MUTED,
              lw=1.0, ms=10)
    tgt = boxes[2]
    ax.add_patch(FancyBboxPatch(
        (LX - 1.16, tgt["y"] - 0.34), 2.32, 0.68,
        boxstyle="round,pad=0,rounding_size=0.07", facecolor="none",
        edgecolor=ATT, linewidth=1.2, linestyle=DASH, zorder=8))
    ax.text(LX, 2.24, "one stage = 4 stacked bottlenecks", fontsize=6.8,
            color=MUTED, ha="center", va="center")
    for y0, y1 in [(tgt["y"] + 0.34, PY + PH / 2), (tgt["y"] - 0.34,
                                                    PY - PH / 2)]:
        ax.plot([LX + 1.16, RX - RW / 2], [y0, y1], color=ATT, lw=0.9,
                ls=DASH, zorder=2)


def detail(ax):
    panel(ax, RX, PY, RW, PH, fc="#fdfaf6", ec="#e6d4bd")
    ax.text(RX - RW / 2 + 0.18, 5.55,
            "(b)  Squeeze-and-excitation bottleneck, expanded", fontsize=8.6,
            fontweight="bold", color=INK, ha="left", va="center")

    ax.plot([4.82], [SPINE], marker="o", ms=4.2, color=INK, zorder=7)
    ax.text(4.82, SPINE - 0.26, "$x$", fontsize=8.6, color=INK, ha="center",
            va="center")
    trunk = []
    for x, lab in [(5.52, "1×1 conv\nC → C/4"), (6.62, "3×3 conv\nC/4 → C/4"),
                   (7.72, "1×1 conv\nC/4 → C")]:
        trunk.append(block(ax, x, SPINE, 0.82, 1.00, lab, ENC, fontsize=6.8))
    arrow(ax, (4.86, SPINE), (trunk[0]["l"][0] - 0.04, SPINE), ms=11)
    for a, b in zip(trunk[:-1], trunk[1:]):
        arrow(ax, (a["r"][0] + 0.04, SPINE), (b["l"][0] - 0.04, SPINE),
              ms=11)
    for b, t in zip(trunk, ["H×W×C/4", "H×W×C/4", "H×W×C"]):
        tensor_label(ax, b["x"], SPINE + 0.66, t)

    split = (8.32, SPINE)
    arrow(ax, (trunk[-1]["r"][0] + 0.04, SPINE), (split[0] - 0.02, SPINE),
          ms=11)
    ax.plot([split[0]], [SPINE], marker="o", ms=3.8, color=INK, zorder=7)

    mul = op_circle(ax, 11.30, SPINE, "×")
    add = op_circle(ax, 12.10, SPINE, "+")
    ax.plot([split[0], mul["l"][0] - 0.03], [SPINE, SPINE], color=INK,
            lw=1.2, zorder=6)
    arrow(ax, (mul["l"][0] - 0.20, SPINE), (mul["l"][0] - 0.03, SPINE),
          ms=11)
    arrow(ax, (mul["r"][0] + 0.03, SPINE), (add["l"][0] - 0.03, SPINE),
          ms=11)
    arrow(ax, (add["r"][0] + 0.03, SPINE), (12.86, SPINE), ms=11)
    ax.plot([12.92], [SPINE], marker="o", ms=4.2, color=INK, zorder=7)
    ax.text(12.92, SPINE - 0.26, "$y$", fontsize=8.6, color=INK, ha="center",
            va="center")
    tensor_label(ax, 11.30, SPINE - 0.32, "scale")
    tensor_label(ax, 12.10, SPINE - 0.32, "add")

    # squeeze-and-excitation branch
    se = []
    for x, lab in [(8.90, "global\navg-pool"), (9.90, "FC  C → C/16\nReLU"),
                   (10.90, "FC  C/16 → C\nsigmoid")]:
        se.append(block(ax, x, SEY, 0.86, 0.74, lab, ATT, fontsize=6.4))
    route(ax, [(split[0], SPINE - 0.04), (split[0], SEY),
               (se[0]["l"][0] - 0.04, SEY)], color=ATT, lw=1.1, ms=11)
    for a, b in zip(se[:-1], se[1:]):
        arrow(ax, (a["r"][0] + 0.04, SEY), (b["l"][0] - 0.04, SEY),
              color=ATT, lw=1.1, ms=11)
    route(ax, [(se[-1]["r"][0] + 0.04, SEY), (11.30, SEY),
               (11.30, mul["b"][1] - 0.04)], color=ATT, lw=1.1, ms=11)
    tensor_label(ax, 8.90, SEY - 0.52, "1×1×C")
    tensor_label(ax, 9.90, SEY - 0.52, "1×1×C/16")
    tensor_label(ax, 10.90, SEY - 0.52, "1×1×C  gate ∈ (0,1)")
    ax.text(8.42, SEY + 0.62, "squeeze-and-excitation branch", fontsize=7.0,
            color=ATT, ha="left", va="center", fontweight="bold")

    # identity residual
    route(ax, [(4.82, SPINE + 0.05), (4.82, 5.02), (12.10, 5.02),
               (12.10, add["t"][1] + 0.03)], color=DEC, lw=1.2, ms=11)
    ax.text(8.46, 5.14, "identity shortcut  (projection when C changes)",
            fontsize=7.0, color=DEC, ha="center", va="center")


def legend(ax):
    panel(ax, 6.72, 1.32, 12.70, 0.82)
    y = 1.32
    items = [("circ+", "element-wise add"),
             ("circx", "channel-wise multiply"),
             ("solid", "forward tensor flow"),
             ("dash", "zoom / callout"),
             ("text", "C = 256,  H×W = 28×28,  reduction r = 16")]
    xs = [0.85, 3.15, 5.60, 8.05, 10.20]
    for (kind, txt), x in zip(items, xs):
        if kind == "circ+":
            op_circle(ax, x, y, "+", r=0.13, fontsize=8.5, zorder=5)
        elif kind == "circx":
            op_circle(ax, x, y, "×", r=0.13, fontsize=8.5, zorder=5)
        elif kind == "solid":
            arrow(ax, (x - 0.13, y), (x + 0.29, y), color=INK, lw=1.2, ms=11,
                  zorder=5)
        elif kind == "dash":
            arrow(ax, (x - 0.13, y), (x + 0.29, y), color=ATT, lw=1.1,
                  ls=DASH, ms=11, zorder=5)
        if kind == "text":
            ax.text(x, y, txt, ha="left", va="center", fontsize=7.2,
                    color=INK, zorder=5)
        else:
            ax.text(x + 0.42, y, txt, ha="left", va="center", fontsize=7.2,
                    color=INK, zorder=5)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    titles(ax, 0.45, 6.58,
           "From backbone to operator: one bottleneck under the magnifier",
           "The highlighted stage on the left is expanded on the right, "
           "down to the element-wise operators and the channel gate")
    overview(ax)
    detail(ax)
    legend(ax)
    savefig(fig, "270_net_module_detail.png")


if __name__ == "__main__":
    main()
