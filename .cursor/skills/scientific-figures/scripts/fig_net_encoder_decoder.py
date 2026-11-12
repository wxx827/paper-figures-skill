"""U-Net style encoder-decoder architecture with skip connections.

A staircase layout: the contracting path steps down to the left-of-centre
bottleneck, the expanding path steps back up on the right, and dashed arcs
carry the concatenated skip tensors across at matching resolution.  Every
node is a stacked feature-map cube whose footprint shrinks and whose depth
grows with the channel count, and each level is annotated with its tensor
shape.  A light legend strip at the bottom decodes the arrow vocabulary.

All geometry is expressed in inches on an equal-aspect canvas, so corner
radii, arrow heads and gaps are identical everywhere in the batch.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pubstyle import set_pub_style, PALETTES, savefig

# ---- shared architecture-diagram vocabulary -------------------------------
ENC = "#3b5b92"    # encoder blue
DEC = "#2e7d5b"    # decoder green
ATT = "#c9a227"    # attention / core module orange
LOSS = "#b23a48"   # loss red
DATA = "#8c8c8c"   # data / input grey
AUX = "#8e6bbf"    # auxiliary module purple
INK = "#24303c"
MUTED = "#6f7783"
SOFT = "#f2f4f7"
LINE = "#dde2e8"

R = 0.055          # uniform corner radius (inches)
LW = 0.9           # uniform box stroke width
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
    """Rounded module box with centred label and optional small subtitle."""
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
    """Connection arrow; endpoints are exact (no shrink)."""
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def route(ax, pts, color=INK, lw=1.2, ls="-", ms=12, zorder=6):
    """Poly-line routing with an arrow head on the final segment."""
    pts = [tuple(p) for p in pts]
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color, lw=lw, ls=ls, solid_capstyle="round",
                zorder=zorder)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=ms,
          zorder=zorder)


def feature_map(ax, x, y, w, h, n_layers=3, color=ENC, depth=0.055,
                alpha0=0.42, ec="white", lw=0.7, zorder=3):
    """Stacked feature-map cube (C x H x W): offset slabs, front one solid."""
    for i in range(n_layers):
        k = n_layers - 1 - i
        a = alpha0 + (1.0 - alpha0) * (i / max(n_layers - 1, 1))
        ax.add_patch(Rectangle((x - w / 2 + k * depth, y - h / 2 + k * depth),
                               w, h, facecolor=color, alpha=a, edgecolor=ec,
                               linewidth=lw, zorder=zorder + i))
    a = anchors(x, y, w, h)
    a["top"] = y + h / 2 + (n_layers - 1) * depth
    return a


def skip_connection(ax, p0, p1, rad=-0.08, color=MUTED, label=None, lw=1.0,
                    ms=11, fontsize=6.5, zorder=5):
    arrow(ax, p0, p1, color=color, rad=rad, lw=lw, ls=DASH, ms=ms,
          zorder=zorder)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        ax.text(mx + 0.5 * rad * dy, my - 0.5 * rad * dx, label, ha="center",
                va="center", fontsize=fontsize, color=color, zorder=zorder + 2,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none"))


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
W, H = 13.3, 7.35

LEVELS = [
    dict(y=5.92, s=0.95, n=3, dim="256×256×64"),
    dict(y=4.60, s=0.78, n=4, dim="128×128×128"),
    dict(y=3.32, s=0.63, n=5, dim="64×64×256"),
    dict(y=2.08, s=0.50, n=6, dim="32×32×512"),
]
ENC_X = [1.85, 3.35, 4.85]
BOT_X = 6.60
DEC_X = [8.35, 9.85, 11.35]


def op_label(ax, p0, p1, text, color, dx=0.13, dy=0.12, ha="left"):
    ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy, text,
            ha=ha, va="center", fontsize=6.5, color=color, zorder=8)


def stage_caption(ax, x, y, s, name, dim, color):
    ax.text(x, y - s / 2 - 0.17, name, ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=color)
    tensor_label(ax, x, y - s / 2 - 0.33, dim)


def param_inset(fig):
    """Small auxiliary panel: parameter budget per stage."""
    ax = fig.add_axes([9.05 / W, 0.72 / H, 3.30 / W, 1.02 / H])
    names = ["E1", "E2", "E3", "Bneck", "D3", "D2", "D1"]
    vals = [0.04, 0.30, 1.18, 4.72, 2.36, 0.59, 0.15]
    cols = [ENC] * 3 + [ATT] + [DEC] * 3
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.66, color=cols, edgecolor="white", lw=0.5)
    for yy, v in zip(y, vals):
        ax.text(v + 0.12, yy, f"{v:.2f}", va="center", ha="left",
                fontsize=6.0, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.0)
    ax.set_xlim(0, 6.1)
    ax.set_xticks([0, 2, 4, 6])
    ax.tick_params(labelsize=6.0, length=2, width=0.6, pad=1.5)
    ax.set_xlabel("parameters (M)", fontsize=6.2, labelpad=1.5)
    ax.set_title("Stage budget · 9.34 M total", fontsize=7.0,
                 fontweight="bold", color=INK, pad=3, loc="left")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    ax.set_facecolor("none")


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    titles(ax, 0.42, 6.95,
           "Encoder–decoder segmentation network with skip connections",
           "Contracting path halves the spatial grid and doubles the width; "
           "skips re-inject full-resolution detail into the decoder")

    # input / output
    src = block(ax, 0.72, LEVELS[0]["y"], 0.66, 0.82, "Input", DATA,
                sub="3×256×256")
    dst = block(ax, 12.62, LEVELS[0]["y"], 0.72, 0.82, "Logits", AUX,
                sub="K×256×256")

    enc, dec = [], []
    for i, x in enumerate(ENC_X):
        lv = LEVELS[i]
        a = feature_map(ax, x, lv["y"], lv["s"], lv["s"], lv["n"], ENC)
        stage_caption(ax, x, lv["y"], lv["s"], f"E{i + 1}", lv["dim"], ENC)
        enc.append(a)

    lv = LEVELS[3]
    bot = feature_map(ax, BOT_X, lv["y"], lv["s"], lv["s"], lv["n"], ATT)
    stage_caption(ax, BOT_X, lv["y"], lv["s"], "Bottleneck", lv["dim"], ATT)

    for j, x in enumerate(DEC_X):
        i = 2 - j
        lv = LEVELS[i]
        a = feature_map(ax, x, lv["y"], lv["s"], lv["s"], lv["n"], DEC)
        stage_caption(ax, x, lv["y"], lv["s"], f"D{i + 1}", lv["dim"], DEC)
        dec.append(a)
    dec = dec[::-1]          # dec[i] matches level i

    # input -> E1 ,  D1 -> logits
    arrow(ax, (src["r"][0] + 0.04, src["y"]),
          (enc[0]["l"][0] - 0.04, enc[0]["y"]), color=INK)
    arrow(ax, (dec[0]["r"][0] + 0.04, dec[0]["y"]),
          (dst["l"][0] - 0.04, dst["y"]), color=INK)

    # contracting path
    chain = enc + [bot]
    for i in range(3):
        a, b = chain[i], chain[i + 1]
        p0 = (a["x"] + a["w"] * 0.28, a["y"] - a["h"] / 2 - 0.03)
        p1 = (b["x"] - b["w"] * 0.34, b["y"] + b["h"] / 2 + 0.05)
        arrow(ax, p0, p1, color=ENC, lw=1.2)
        op_label(ax, p0, p1, "max-pool /2", ENC, dx=0.10, dy=0.10)

    # expanding path
    chain = [bot] + dec[::-1]
    for i in range(3):
        a, b = chain[i], chain[i + 1]
        p0 = (a["x"] + a["w"] * 0.30, a["y"] + a["h"] / 2 + 0.03)
        p1 = (b["x"] - b["w"] * 0.32, b["y"] - b["h"] / 2 - 0.05)
        arrow(ax, p0, p1, color=DEC, lw=1.2)
        op_label(ax, p0, p1, "up-conv ×2", DEC, dx=-0.05, dy=0.18,
                 ha="right")

    # skips
    for i, rad in zip(range(3), (-0.050, -0.070, -0.105)):
        lab = "skip connection · channel concat" if i == 0 else "concat"
        skip_connection(ax, (enc[i]["r"][0] + 0.05, enc[i]["y"]),
                        (dec[i]["l"][0] - 0.05, dec[i]["y"]), rad=rad,
                        label=lab)

    # legend block (fills the empty lower-left triangle)
    panel(ax, 2.72, 1.28, 4.55, 1.62)
    ax.text(0.62, 1.90, "Arrow vocabulary", fontsize=8.0, fontweight="bold",
            color=INK, ha="left", va="center", zorder=4)
    items = [(ENC, "-", "Down-stage: 2× conv 3×3, then max-pool 2×2"),
             (DEC, "-", "Up-stage: transposed conv 2×2, then 2× conv 3×3"),
             (MUTED, DASH, "Skip: concatenate encoder features"),
             (None, None, "Cube stack: depth ∝ channel count C")]
    for k, (col, ls, txt) in enumerate(items):
        yy = 1.60 - 0.28 * k
        if col is None:
            for m in range(3):
                ax.add_patch(Rectangle((0.66 + 0.035 * m, yy - 0.10 + 0.035 * m),
                                       0.17, 0.17, facecolor=ATT,
                                       alpha=0.45 + 0.27 * m,
                                       edgecolor="white", lw=0.6, zorder=4))
        else:
            arrow(ax, (0.66, yy), (1.06, yy), color=col, ls=ls, lw=1.2,
                  ms=11, zorder=4)
        ax.text(1.18, yy, txt, ha="left", va="center", fontsize=7.0,
                color=INK, zorder=4)

    panel(ax, 10.72, 1.28, 4.30, 1.62)
    param_inset(fig)
    savefig(fig, "263_net_encoder_decoder.png")


if __name__ == "__main__":
    main()
