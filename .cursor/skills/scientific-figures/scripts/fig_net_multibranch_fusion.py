"""Two-modality network with an explicit cross-attention fusion module.

A camera branch and a LiDAR branch run in parallel with matched stage
geometry, then meet in the highlighted fusion block where the two feature
streams exchange queries (the crossing arrows are the point of the figure).
A gated sum feeds one shared decoder that serves two task heads.  Bottom
strip: symbol legend plus a reduced-size ablation panel over fusion
strategies.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
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
                fontweight=weight, zorder=zorder + 2)
    if sub:
        ax.text(x, y - 0.115, sub, ha="center", va="center",
                color=sub_color or text_color, fontsize=6.5, alpha=0.92,
                zorder=zorder + 2)
    return anchors(x, y, w, h)


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.2, ls="-", ms=12, zorder=6,
          alpha=1.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color, alpha=alpha,
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


def op_circle(ax, x, y, sym="+", r=0.135, color=INK, fc="white", fontsize=9,
              zorder=7):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=color, lw=1.0,
                        zorder=zorder))
    ax.text(x, y + 0.004, sym, ha="center", va="center", fontsize=fontsize,
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
W, H = 13.2, 7.4
YA, YB = 5.60, 2.70          # camera row / LiDAR row
BX = [2.45, 3.75, 5.05]      # shared stage columns  (same x in both rows)
BW, BH = 1.16, 0.74

CAM = [("Conv stem", "/4 · 64", "128×128×64"),
       ("Stage 2", "/8 · 128", "64×64×128"),
       ("Stage 3", "/16 · 256", "32×32×256")]
LID = [("Voxelise", "0.2 m grid", "256×256×32"),
       ("Sparse conv", "/8 · 128", "64×64×128"),
       ("BEV flatten", "/16 · 256", "32×32×256")]


def branch(ax, y, items, color, src_label, src_sub, tag):
    src = block(ax, 1.05, y, 1.24, 0.86, src_label, DATA, sub=src_sub,
                fontsize=8)
    prev = src
    for name, sub, dim in items:
        b = block(ax, BX[items.index((name, sub, dim))], y, BW, BH, name,
                  color, sub=sub, fontsize=7.8)
        arrow(ax, (prev["r"][0] + 0.04, y), (b["l"][0] - 0.04, y),
              color=MUTED, lw=1.1, ms=11)
        tensor_label(ax, b["x"], y - BH / 2 - 0.17, dim)
        prev = b
    ax.text(0.42, y + 0.72, tag, fontsize=8.4, fontweight="bold",
            color=color, ha="left", va="center")
    return prev


def fusion(ax):
    panel(ax, 7.98, 4.15, 2.98, 2.62, fc="#fdf6ee", ec=ATT, ls=DASH, lw=1.2)
    ax.text(6.60, 5.28, "Cross-modal fusion", fontsize=8.6,
            fontweight="bold", color=ATT, ha="left", va="center")

    up = block(ax, 7.62, 4.78, 1.70, 0.66, "Cross-Attention", ATT,
               sub="Q: camera   K,V: LiDAR", fontsize=7.6)
    dn = block(ax, 7.62, 3.52, 1.70, 0.66, "Cross-Attention", ATT,
               sub="Q: LiDAR   K,V: camera", fontsize=7.6)
    gate = block(ax, 8.94, 4.15, 0.56, 1.86, "", ATT)
    ax.text(8.94, 4.68, "Gated", ha="center", va="center", color="white",
            fontsize=7.4, rotation=90, zorder=6)
    ax.text(8.94, 3.64, "sum", ha="center", va="center", color="white",
            fontsize=7.4, rotation=90, zorder=6)
    op_circle(ax, 8.94, 4.15, "×", r=0.12, color=ATT, fontsize=8)
    arrow(ax, (up["r"][0] + 0.04, up["y"]), (gate["l"][0] - 0.04, 4.52),
          color=INK, lw=1.1, ms=11)
    arrow(ax, (dn["r"][0] + 0.04, dn["y"]), (gate["l"][0] - 0.04, 3.78),
          color=INK, lw=1.1, ms=11)
    return up, dn, gate


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    titles(ax, 0.36, 7.02,
           "Two-branch camera–LiDAR network with cross-attention fusion",
           "Both branches keep the same stage geometry; the fusion block "
           "swaps queries between modalities before a single shared decoder")

    for x in [1.05] + BX:
        ax.plot([x, x], [3.16, 5.14], color="#e1e6ec", lw=0.9, ls=(0, (2, 3)),
                zorder=0)
    ax.text(3.05, 4.15, "identical stage geometry in both branches",
            fontsize=7.2, color=MUTED, ha="center", va="center", zorder=2,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor=LINE, linewidth=0.7))

    a_end = branch(ax, YA, CAM, ENC, "RGB image", "3×512×512",
                   "Camera branch")
    b_end = branch(ax, YB, LID, AUX, "LiDAR sweep", "N×4 points",
                   "LiDAR branch")

    up, dn, gate = fusion(ax)

    # crossing streams: each modality feeds both attention directions
    bus_a, bus_b = 6.05, 6.05
    for src, col, y0 in [(a_end, ENC, YA), (b_end, AUX, YB)]:
        arrow(ax, (src["r"][0] + 0.04, y0), (bus_a - 0.02, y0), color=col,
              lw=1.2, ms=11)
        ax.plot([bus_a], [y0], marker="o", ms=3.8, color=col, zorder=7)
    for col, y0, dy in [(ENC, YA, 0.16), (AUX, YB, -0.16)]:
        arrow(ax, (bus_a, y0), (up["l"][0] - 0.04, up["y"] + dy), color=col,
              lw=1.2, ms=11)
        arrow(ax, (bus_b, y0), (dn["l"][0] - 0.04, dn["y"] + dy), color=col,
              lw=1.2, ms=11)

    # shared decoder + heads
    dec = block(ax, 10.52, 4.15, 1.32, 1.36, "Shared\ndecoder", DEC,
                fontsize=8.2)
    ax.text(10.52, 3.72, "FPN, 3 levels", ha="center", va="center",
            color="white", fontsize=6.5, zorder=6)
    arrow(ax, (gate["r"][0] + 0.04, 4.15), (dec["l"][0] - 0.04, 4.15))
    h1 = block(ax, 12.28, 4.82, 1.24, 0.66, "Semantic map", DEC,
               sub="19 classes", fontsize=7.6)
    h2 = block(ax, 12.28, 3.48, 1.24, 0.66, "3-D boxes", DEC,
               sub="7-DoF + score", fontsize=7.6)
    for h in (h1, h2):
        arrow(ax, (dec["r"][0] + 0.04, 4.15), (h["l"][0] - 0.04, h["y"]),
              lw=1.1, ms=11)

    # ---- bottom strips ----------------------------------------------------
    panel(ax, 3.28, 1.16, 5.76, 1.38)
    ax.text(0.52, 1.66, "Symbols", fontsize=8.0, fontweight="bold",
            color=INK, ha="left", va="center", zorder=4)
    items = [(ENC, "-", "camera feature stream"),
             (AUX, "-", "LiDAR feature stream"),
             (ATT, DASH, "fusion module boundary"),
             (INK, "-", "shared / task-agnostic path")]
    for k, (col, ls, txt) in enumerate(items):
        xx = 0.58 + 2.86 * (k % 2)
        yy = 1.28 - 0.42 * (k // 2)
        arrow(ax, (xx, yy), (xx + 0.42, yy), color=col, ls=ls, lw=1.2,
              ms=11, zorder=4)
        ax.text(xx + 0.54, yy, txt, ha="left", va="center", fontsize=7.2,
                color=INK, zorder=4)

    panel(ax, 9.68, 1.16, 6.34, 1.38)
    ax.text(6.66, 1.66, "Fusion-strategy ablation", fontsize=8.0,
            fontweight="bold", color=INK, ha="left", va="center", zorder=4)
    axb = fig.add_axes([6.85 / W, 0.66 / H, 2.90 / W, 0.86 / H])
    names = ["camera only", "LiDAR only", "concat", "cross-attn"]
    vals = [58.2, 63.7, 66.1, 69.4]
    cols = [ENC, AUX, MUTED, ATT]
    y = np.arange(4)[::-1]
    axb.barh(y, vals, height=0.66, color=cols, edgecolor="white", lw=0.5)
    for yy, v in zip(y, vals):
        axb.text(v + 0.6, yy, f"{v:.1f}", va="center", ha="left",
                 fontsize=6.0, color=MUTED)
    axb.set_yticks(y)
    axb.set_yticklabels(names, fontsize=6.0)
    axb.set_xlim(50, 76)
    axb.set_xticks([50, 60, 70])
    axb.tick_params(labelsize=6.0, length=2, width=0.6, pad=1.5)
    axb.set_xlabel("mAP (%)", fontsize=6.2, labelpad=1.0)
    for sp in axb.spines.values():
        sp.set_linewidth(0.6)
    axb.set_facecolor("none")
    notes = ["queries exchanged once per stage",
             "gate: sigmoid over channel statistics",
             "branches frozen for the first 5 epochs"]
    for k, t in enumerate(notes):
        ax.text(10.10, 1.44 - 0.30 * k, "•  " + t, fontsize=6.8, color=INK,
                ha="left", va="center", zorder=4)

    savefig(fig, "265_net_multibranch_fusion.png")


if __name__ == "__main__":
    main()
