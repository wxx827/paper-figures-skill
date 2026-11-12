"""Three architectures drawn with one shared module vocabulary.

Baseline, prior work and the proposed design occupy identical slot columns,
so the eye only has to resolve the two boxes that actually differ; those are
ringed in dashed red and carry a one-line note.  A narrow side panel keeps
the accuracy and size comparison at roughly a fifth of the main width so it
never competes with the diagrams.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
          text_color="white", fontsize=7.6, sub_color=None, lw=LW, ls="-",
          alpha=1.0, zorder=3, rounding=R):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=color,
        edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder))
    if label:
        ax.text(x, y + (0.095 if sub else 0.0), label, ha="center",
                va="center", color=text_color, fontsize=fontsize,
                zorder=zorder + 2)
    if sub:
        ax.text(x, y - 0.105, sub, ha="center", va="center",
                color=sub_color or text_color, fontsize=6.4, alpha=0.94,
                zorder=zorder + 2)
    return anchors(x, y, w, h)


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
    return anchors(x, y, w, h)


def ring(ax, b, pad=0.11, color=LOSS, lw=1.3):
    ax.add_patch(FancyBboxPatch(
        (b["x"] - b["w"] / 2 - pad, b["y"] - b["h"] / 2 - pad),
        b["w"] + 2 * pad, b["h"] + 2 * pad,
        boxstyle="round,pad=0,rounding_size=0.09", facecolor="none",
        edgecolor=color, linewidth=lw, linestyle=DASH, zorder=8))


def tensor_label(ax, x, y, text, color=MUTED, fontsize=6.5, ha="center",
                 va="center", **kw):
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=fontsize,
                   **kw)


def titles(ax, x, y, main, sub=None):
    ax.text(x, y, main, fontsize=13.5, fontweight="bold", color=INK,
            ha="left", va="center")
    if sub:
        ax.text(x, y - 0.32, sub, fontsize=8.6, color=MUTED, ha="left",
                va="center")


# ---- figure ----------------------------------------------------------------
W, H = 13.4, 7.4
ROWS = [5.72, 4.10, 2.48]
BANDX, BANDW = 5.15, 9.40
SLOTS = [(1.55, 0.90), (3.05, 1.26), (4.55, 1.16), (6.28, 1.62),
         (8.14, 1.30), (9.38, 0.74)]
MH = 0.62

VARIANTS = [
    dict(tag="Baseline", name="ResNet-50 + FPN detector",
         score="38.2 AP   ·   41.5 M   ·   4.1 GFLOPs",
         mods=[("images", DATA, None), ("Backbone", ENC, "ResNet-50"),
               ("FPN neck", AUX, "P3–P7"), (None, None, None),
               ("Conv head", DEC, "4 conv layers"), ("boxes", DATA, None)],
         rings=[], notes=[]),
    dict(tag="Prior work", name="self-attention neck (Zhang et al.)",
         score="41.6 AP   ·   52.3 M   ·   5.8 GFLOPs",
         mods=[("images", DATA, None), ("Backbone", ENC, "ResNet-50"),
               ("FPN neck", AUX, "P3–P7"),
               ("Self-attention neck", ATT, "6 layers, per level"),
               ("Conv head", DEC, "4 conv layers"), ("boxes", DATA, None)],
         rings=[3], notes=[(3, "attention runs per level → +10.8 M params")]),
    dict(tag="Ours", name="cross-scale gated attention + light head",
         score="43.4 AP   ·   44.8 M   ·   4.6 GFLOPs",
         mods=[("images", DATA, None), ("Backbone", ENC, "ResNet-50"),
               ("FPN neck", AUX, "P3–P7"),
               ("Cross-scale gated attn.", ATT, "2 layers, shared"),
               ("Light head", DEC, "2 conv layers"), ("boxes", DATA, None)],
         rings=[3, 4],
         notes=[(3, "one shared block across levels"),
                (4, "half the head depth")]),
]


def draw_variant(ax, y, spec):
    panel(ax, BANDX, y, BANDW, 1.50, fc="#fbfcfd", ec="#d9e0e7")
    ax.text(BANDX - BANDW / 2 + 0.18, y + 0.54, spec["tag"], fontsize=8.6,
            fontweight="bold", color=INK, ha="left", va="center")
    ax.text(BANDX - BANDW / 2 + 1.02, y + 0.54, "·  " + spec["name"],
            fontsize=7.4, color=MUTED, ha="left", va="center")
    ax.text(BANDX + BANDW / 2 - 0.18, y + 0.54, spec["score"], fontsize=7.4,
            color=INK, ha="right", va="center")

    ym = y - 0.06
    boxes = []
    for (x, w), (name, col, sub) in zip(SLOTS, spec["mods"]):
        if name is None:
            b = block(ax, x, ym, w, MH, "no enhancer", "white", ec="#c3ccd5",
                      ls=DASH, text_color=MUTED, fontsize=7.2, lw=1.0)
        else:
            b = block(ax, x, ym, w, MH, name, col, sub=sub, fontsize=7.4)
        boxes.append(b)
    for a, b in zip(boxes[:-1], boxes[1:]):
        arrow(ax, (a["r"][0] + 0.04, ym), (b["l"][0] - 0.04, ym),
              color="#8c97a3", lw=1.1, ms=11)
    for k in spec["rings"]:
        ring(ax, boxes[k])
    for k, txt in spec["notes"]:
        ax.text(boxes[k]["x"], y - 0.64, txt, fontsize=6.5, color=LOSS,
                ha="center", va="center")


def side_panel(fig, ax):
    panel(ax, 11.66, 4.10, 3.04, 4.74)
    ax.text(10.24, 6.18, "Accuracy vs. size", fontsize=8.4,
            fontweight="bold", color=INK, ha="left", va="center", zorder=4)
    names = ["Baseline", "Prior", "Ours"]
    cols = [DATA, ENC, ATT]
    y = np.arange(3)[::-1]

    axa = fig.add_axes([10.42 / W, 4.60 / H, 2.32 / W, 1.24 / H])
    vals = [38.2, 41.6, 43.4]
    axa.barh(y, vals, height=0.62, color=cols, edgecolor="white", lw=0.5)
    for yy, v in zip(y, vals):
        axa.text(v + 0.25, yy, f"{v:.1f}", va="center", ha="left",
                 fontsize=6.0, color=MUTED)
    axa.set_xlim(34, 47)
    axa.set_xticks([35, 40, 45])
    axa.set_xlabel("AP (%)", fontsize=6.2, labelpad=1.0)

    axp = fig.add_axes([10.42 / W, 2.56 / H, 2.32 / W, 1.24 / H])
    vals = [41.5, 52.3, 44.8]
    axp.barh(y, vals, height=0.62, color=cols, edgecolor="white", lw=0.5)
    for yy, v in zip(y, vals):
        axp.text(v + 0.8, yy, f"{v:.1f}", va="center", ha="left",
                 fontsize=6.0, color=MUTED)
    axp.set_xlim(0, 66)
    axp.set_xticks([0, 25, 50])
    axp.set_xlabel("parameters (M)", fontsize=6.2, labelpad=1.0)

    for a in (axa, axp):
        a.set_yticks(y)
        a.set_yticklabels(names, fontsize=6.0)
        a.tick_params(labelsize=6.0, length=2, width=0.6, pad=1.5)
        for sp in a.spines.values():
            sp.set_linewidth(0.6)
        a.set_facecolor("none")
    ax.text(10.24, 4.16, "+5.2 AP over baseline at +3.3 M parameters",
            fontsize=6.6, color=INK, ha="left", va="center", zorder=4)


def legend(ax):
    panel(ax, 6.72, 1.24, 12.70, 0.80)
    items = [(DATA, "data in / out"), (ENC, "backbone"), (AUX, "neck"),
             (ATT, "enhancer module"), (DEC, "task head")]
    x = 0.80
    for col, txt in items:
        block(ax, x, 1.24, 0.42, 0.26, "", col, zorder=4)
        ax.text(x + 0.30, 1.24, txt, ha="left", va="center", fontsize=7.2,
                color=INK, zorder=4)
        x += 1.95
    ax.add_patch(FancyBboxPatch(
        (x - 0.21, 1.11), 0.42, 0.26,
        boxstyle="round,pad=0,rounding_size=0.05", facecolor="none",
        edgecolor=LOSS, linewidth=1.3, linestyle=DASH, zorder=4))
    ax.text(x + 0.30, 1.24, "differs from the baseline", ha="left",
            va="center", fontsize=7.2, color=INK, zorder=4)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    titles(ax, 0.45, 7.10,
           "Three detectors, one module vocabulary: what actually changes",
           "Slot columns are identical across the rows, so the dashed red "
           "rings isolate the only two design decisions under study")
    for y, spec in zip(ROWS, VARIANTS):
        draw_variant(ax, y, spec)
    side_panel(fig, ax)
    legend(ax)
    savefig(fig, "271_net_comparison_arch.png")


if __name__ == "__main__":
    main()
