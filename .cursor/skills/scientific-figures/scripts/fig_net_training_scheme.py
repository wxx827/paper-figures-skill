"""Three-stage training schedule with explicit freeze states.

Each stage repeats the same four-module column, so the only thing that
changes between panels is which modules are trainable: frozen ones are
hatched, desaturated and carry a padlock, trainable ones stay solid.  The
red loss box hangs off the head and the dashed red path shows how far the
gradient actually travels, with a cross where it is cut.  The bottom axis
turns the three stages into an epoch budget.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Rectangle
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


def frost(color, amount=0.68):
    r, g, b = mpl.colors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


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
        ax.text(x, y + (0.095 if sub else 0.0), label, ha="center",
                va="center", color=text_color, fontsize=fontsize,
                fontweight=weight, zorder=zorder + 2)
    if sub:
        ax.text(x, y - 0.105, sub, ha="center", va="center",
                color=sub_color or text_color, fontsize=6.5, alpha=0.95,
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


def panel(ax, x, y, w, h, fc=SOFT, ec=LINE, lw=0.8, ls="-", zorder=1,
          rounding=0.08):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))
    return anchors(x, y, w, h)


def lock(ax, x, y, s=0.15, color=INK, zorder=8):
    """Padlock glyph drawn from primitives (no icon font)."""
    ax.add_patch(Rectangle((x - 0.5 * s, y - 0.46 * s), s, 0.62 * s,
                           facecolor=color, edgecolor="none", zorder=zorder))
    ax.add_patch(Arc((x, y + 0.16 * s), 0.62 * s, 0.62 * s, theta1=0,
                     theta2=180, lw=1.1, color=color, zorder=zorder))


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
W, H = 13.4, 7.7
CXS = [2.38, 6.70, 11.02]
PW, PH = 4.06, 3.70
PY = 4.90
ROWY = [3.54, 4.28, 5.02, 5.76]          # data, backbone, neck, head
LOSSY = 6.30
MW, MH = 1.92, 0.48
GX = 1.28                                 # gradient lane offset from centre

STAGES = [
    dict(name="Self-supervised pre-training", accent=ENC,
         data="unlabeled frames · 1.2 M",
         mods=[("Backbone  ResNet-50", ENC, True),
               ("Projector  MLP 2048→128", AUX, True),
               ("Contrastive head", DEC, True)],
         loss="InfoNCE", grad_to=1,
         foot="epochs 0–120   ·   lr 1e-3 cosine   ·   batch 256"),
    dict(name="Supervised fine-tuning", accent=ATT,
         data="labeled frames · 20 k",
         mods=[("Backbone  ResNet-50", ENC, False),
               ("Projector  MLP 2048→128", AUX, False),
               ("Task head  linear", DEC, True)],
         loss="Cross-entropy", grad_to=3,
         foot="epochs 120–230   ·   lr 5e-4 step   ·   batch 128"),
    dict(name="Joint refinement", accent=DEC,
         data="labeled + pseudo-labeled",
         mods=[("Backbone  ResNet-50", ENC, True),
               ("Projector  MLP 2048→128", AUX, True),
               ("Task head  linear", DEC, True)],
         loss="CE + consistency", grad_to=1,
         foot="epochs 230–300   ·   lr 5e-5 flat   ·   batch 128"),
]


def stage_panel(ax, cx, idx, spec):
    panel(ax, cx, PY, PW, PH, fc="#fbfcfd", ec="#d5dce4")
    x0 = cx - PW / 2 + 0.16
    ax.add_patch(FancyBboxPatch(
        (x0, PY + PH / 2 - 0.42), 0.30, 0.24,
        boxstyle="round,pad=0,rounding_size=0.05", facecolor=spec["accent"],
        edgecolor="none", zorder=4))
    ax.text(x0 + 0.15, PY + PH / 2 - 0.30, str(idx + 1), ha="center",
            va="center", color="white", fontsize=7.6, fontweight="bold",
            zorder=5)
    ax.text(x0 + 0.40, PY + PH / 2 - 0.30, spec["name"], ha="left",
            va="center", fontsize=8.6, fontweight="bold", color=INK)

    mx = cx - 0.42
    data = block(ax, mx, ROWY[0], MW, MH, spec["data"], DATA, fontsize=7.4)
    mods = [data]
    for k, (name, col, train) in enumerate(spec["mods"]):
        if train:
            b = block(ax, mx, ROWY[k + 1], MW, MH, name, col, fontsize=7.4)
        else:
            b = block(ax, mx, ROWY[k + 1], MW, MH, name, frost(col),
                      ec=col, text_color=INK, fontsize=7.4, hatch="////",
                      lw=0.9)
            lock(ax, mx + MW / 2 - 0.19, ROWY[k + 1], 0.17, color=col)
        mods.append(b)
    for a, b in zip(mods[:-1], mods[1:]):
        arrow(ax, (mx, a["y"] + MH / 2 + 0.03), (mx, b["y"] - MH / 2 - 0.03),
              color=INK, lw=1.2, ms=11)

    lb = block(ax, cx + GX, LOSSY, 1.24, MH, spec["loss"], LOSS,
               fontsize=7.4)
    route(ax, [(mx, ROWY[3] + MH / 2 + 0.03), (mx, LOSSY),
               (lb["l"][0] - 0.03, LOSSY)], color=INK, lw=1.2, ms=11)

    gy = ROWY[spec["grad_to"]]
    route(ax, [(cx + GX, LOSSY - MH / 2 - 0.03), (cx + GX, gy),
               (mx + MW / 2 + 0.04, gy)], color=LOSS, lw=1.1, ls=DASH, ms=11)
    if LOSSY - gy > 1.0:
        ax.text(cx + GX + 0.13, (LOSSY + gy) / 2, "gradient", rotation=90,
                fontsize=6.5, color=LOSS, ha="center", va="center")
    if spec["grad_to"] > 1:
        ys = gy - 0.34
        ax.plot([cx + GX, cx + GX], [gy - 0.05, ys + 0.12], color=LOSS,
                lw=1.0, ls=DASH, alpha=0.45, zorder=6)
        ax.plot([cx + GX - 0.11, cx + GX + 0.11], [ys + 0.09, ys - 0.09],
                color=LOSS, lw=1.5, zorder=8)
        ax.plot([cx + GX - 0.11, cx + GX + 0.11], [ys - 0.09, ys + 0.09],
                color=LOSS, lw=1.5, zorder=8)
        ax.text(cx + GX, ys - 0.20, "no gradient", fontsize=6.5, color=LOSS,
                ha="center", va="center")

    ax.text(cx - PW / 2 + 0.16, PY - PH / 2 - 0.22, spec["foot"],
            fontsize=7.0, color=MUTED, ha="left", va="center")


def timeline(ax):
    x0, x1 = 0.55, 12.85
    ax.text(x0, 2.54, "Epoch budget  ·  300 epochs end to end", fontsize=8.6,
            fontweight="bold", color=INK, ha="left", va="center")

    def ex(e):
        return x0 + (x1 - x0) * e / 300.0

    spans = [(0, 120, ENC, "stage 1  ·  120 epochs"),
             (120, 230, ATT, "stage 2  ·  110 epochs"),
             (230, 300, DEC, "stage 3  ·  70 epochs")]
    for a, b, col, lab in spans:
        ax.add_patch(FancyBboxPatch(
            (ex(a) + 0.03, 1.98), ex(b) - ex(a) - 0.06, 0.34,
            boxstyle="round,pad=0,rounding_size=0.05", facecolor=col,
            edgecolor="white", linewidth=0.9, zorder=3))
        ax.text((ex(a) + ex(b)) / 2, 2.15, lab, ha="center", va="center",
                color="white", fontsize=7.4, zorder=5)
    ax.plot([x0, x1], [1.82, 1.82], color=MUTED, lw=1.0)
    for e in range(0, 301, 50):
        ax.plot([ex(e), ex(e)], [1.82, 1.72], color=MUTED, lw=1.0)
        ax.text(ex(e), 1.56, str(e), ha="center", va="center", fontsize=6.8,
                color=MUTED)


def legend(ax):
    panel(ax, 6.70, 0.96, 12.30, 0.74)
    y = 0.96
    x = 0.80
    block(ax, x, y, 0.42, 0.26, "", ENC, zorder=4)
    ax.text(x + 0.30, y, "trainable module", ha="left", va="center",
            fontsize=7.2, color=INK, zorder=4)

    x = 3.05
    block(ax, x, y, 0.42, 0.26, "", frost(ENC), ec=ENC, hatch="////",
          zorder=4)
    lock(ax, x + 0.34, y, 0.16, color=ENC)
    ax.text(x + 0.50, y, "frozen module (padlock)", ha="left", va="center",
            fontsize=7.2, color=INK, zorder=4)

    x = 5.90
    arrow(ax, (x, y), (x + 0.42, y), color=INK, lw=1.2, ms=11, zorder=4)
    ax.text(x + 0.54, y, "forward pass", ha="left", va="center",
            fontsize=7.2, color=INK, zorder=4)

    x = 8.05
    arrow(ax, (x, y), (x + 0.42, y), color=LOSS, lw=1.1, ls=DASH, ms=11,
          zorder=4)
    ax.text(x + 0.54, y, "gradient path", ha="left", va="center",
            fontsize=7.2, color=INK, zorder=4)

    x = 10.30
    ax.plot([x - 0.11, x + 0.11], [y + 0.09, y - 0.09], color=LOSS, lw=1.4,
            zorder=4)
    ax.plot([x - 0.11, x + 0.11], [y - 0.09, y + 0.09], color=LOSS, lw=1.4,
            zorder=4)
    ax.text(x + 0.24, y, "gradient is cut here", ha="left", va="center",
            fontsize=7.2, color=INK, zorder=4)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    titles(ax, 0.55, 7.34,
           "Three-stage training schedule: what is frozen, what is learned",
           "The module column is identical in every stage; only the freeze "
           "pattern, the loss and the learning rate change")
    for k, (cx, spec) in enumerate(zip(CXS, STAGES)):
        stage_panel(ax, cx, k, spec)
    timeline(ax)
    legend(ax)
    savefig(fig, "268_net_training_scheme.png")


if __name__ == "__main__":
    main()
