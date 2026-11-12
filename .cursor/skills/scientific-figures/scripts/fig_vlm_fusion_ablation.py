"""Figure 349 - Ablation over the vision-language fusion strategy.

(a) schematic of the three fusion designs, drawn from patch and token chips.
(b) the accuracy each design reaches on three downstream tasks.  (c) the
attention each design puts on the queried object, all three on one colour
scale.  The subordinate row underneath trades the numbers off against
parameter count and latency.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from pubstyle import set_pub_style, savefig, lighten
from fig_vlm_image_text_align import (make_scene, attention_map, inch_canvas,
                                      draw_photo, text_card, panel_tag,
                                      bg_block, C_INK, C_SUB, C_FAINT, MONO,
                                      H, W)

STRATEGIES = [("Early fusion", "#808080"), ("Late fusion", "#3d6b8e"),
              ("Cross-attention", "#b23a48")]
TASKS = ["VQA\naccuracy", "Retrieval\nR@1", "Grounding\naccuracy"]
SCORES = np.array([[68.4, 45.1, 62.7],      # early
                   [71.2, 52.3, 66.9],      # late
                   [75.9, 57.8, 73.4]])     # cross-attention
PARAMS = [118.0, 94.0, 132.0]
LATENCY = [41.0, 23.0, 35.0]
ENTROPY = [3.42, 2.71, 1.86]
V_COL, L_COL = "#3b5b92", "#2e7d5b"
FIGSIZE = (10.4, 6.6)


def arrow(ax, p0, p1, color="#9a9a9a", lw=0.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5,
                                 linewidth=lw, color=color, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def tokens(ax, x, y, n, color, tag):
    s, g = 0.105, 0.028
    for i in range(n):
        ax.add_patch(Rectangle((x, y + i * (s + g)), s, s,
                               facecolor=lighten(color, 0.62),
                               edgecolor=color, linewidth=0.6, zorder=3))
    ax.text(x - 0.06, y + (n * (s + g) - g) / 2.0, tag, ha="right",
            va="center", fontsize=6.4, color=color, fontweight="bold", **MONO)
    return y + (n * (s + g) - g) / 2.0, x + s


def draw_arch(ax, y0, kind, col):
    """One fusion schematic inside a 3.43 x 1.05 in slot."""
    vy, vx = tokens(ax, 0.30, y0 + 0.02, 4, V_COL, "V")
    ly, lx = tokens(ax, 0.30, y0 + 0.60, 3, L_COL, "L")
    head_x = 2.72
    if kind == "early":
        text_card(ax, 0.78, y0 + 0.06, 0.92, 0.82,
                  "joint\ntransformer", fc="#f2f2f2", ec="#b8b8b8", fs=6.4,
                  mono=True, ha="center", color=C_INK, lw=0.7)
        arrow(ax, (vx + 0.05, vy), (0.75, y0 + 0.30))
        arrow(ax, (lx + 0.05, ly), (0.75, y0 + 0.62))
        arrow(ax, (1.70, y0 + 0.47), (head_x - 0.06, y0 + 0.47))
        ax.text(2.21, y0 + 0.37, "concat at\nlayer 0", ha="center",
                va="bottom", fontsize=6.0, color=C_FAINT, linespacing=1.2)
    elif kind == "late":
        text_card(ax, 0.78, y0 + 0.04, 0.72, 0.34, "vision enc",
                  fc=lighten(V_COL, 0.88), ec=V_COL, fs=6.2, mono=True,
                  ha="center", color=V_COL, lw=0.7)
        text_card(ax, 0.78, y0 + 0.58, 0.72, 0.34, "text enc",
                  fc=lighten(L_COL, 0.88), ec=L_COL, fs=6.2, mono=True,
                  ha="center", color=L_COL, lw=0.7)
        arrow(ax, (vx + 0.05, vy), (0.75, y0 + 0.21))
        arrow(ax, (lx + 0.05, ly), (0.75, y0 + 0.75))
        ax.add_patch(plt.Circle((1.86, y0 + 0.48), 0.115, facecolor="white",
                                edgecolor="#8a8a8a", linewidth=0.8, zorder=4))
        ax.text(1.86, y0 + 0.48, "$\\cdot$", ha="center", va="center",
                fontsize=10, color=C_INK, zorder=5)
        arrow(ax, (1.52, y0 + 0.21), (1.79, y0 + 0.42))
        arrow(ax, (1.52, y0 + 0.75), (1.79, y0 + 0.55))
        arrow(ax, (1.99, y0 + 0.48), (head_x - 0.06, y0 + 0.48))
        ax.text(2.36, y0 + 0.38, "dot product\nonly", ha="center",
                va="bottom", fontsize=6.0, color=C_FAINT, linespacing=1.2)
    else:
        text_card(ax, 0.78, y0 + 0.04, 0.62, 0.34, "vision enc",
                  fc=lighten(V_COL, 0.88), ec=V_COL, fs=6.2, mono=True,
                  ha="center", color=V_COL, lw=0.7)
        text_card(ax, 0.78, y0 + 0.58, 0.62, 0.34, "text enc",
                  fc=lighten(L_COL, 0.88), ec=L_COL, fs=6.2, mono=True,
                  ha="center", color=L_COL, lw=0.7)
        arrow(ax, (vx + 0.05, vy), (0.75, y0 + 0.21))
        arrow(ax, (lx + 0.05, ly), (0.75, y0 + 0.75))
        text_card(ax, 1.62, y0 + 0.06, 0.86, 0.82, "cross-\nattention",
                  fc=lighten(col, 0.86), ec=col, fs=6.4, mono=True,
                  ha="center", color=col, weight="bold", lw=0.9)
        arrow(ax, (1.42, y0 + 0.21), (1.59, y0 + 0.34), color=V_COL)
        arrow(ax, (1.42, y0 + 0.75), (1.59, y0 + 0.62), color=L_COL)
        arrow(ax, (2.48, y0 + 0.47), (head_x - 0.06, y0 + 0.47))
        ax.text(2.05, y0 + 0.96, "token-level exchange", ha="center",
                va="top", fontsize=6.0, color=C_FAINT)
    text_card(ax, head_x, y0 + 0.30, 0.44, 0.34, "head", fc="#f2f2f2",
              ec="#b8b8b8", fs=6.2, mono=True, ha="center", color=C_INK,
              lw=0.7)


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=FIGSIZE)
    bg_block(fig, 0.020, 0.030, 0.980, 0.246)

    # ---- (a) architectures ------------------------------------------------
    axa, aw, ah = inch_canvas(fig, [0.035, 0.290, 0.330, 0.590])
    for i, ((name, col), kind) in enumerate(zip(STRATEGIES,
                                                ("early", "late", "cross"))):
        y0 = 0.30 + i * 1.24
        axa.text(0.02, y0 - 0.10, f"{name}", ha="left", va="bottom",
                 fontsize=8.5, fontweight="bold", color=col)
        if i == 2:
            axa.add_patch(Rectangle((0.0, y0 - 0.16), aw, 1.18,
                                    facecolor=lighten(col, 0.94),
                                    edgecolor="none", zorder=0))
        draw_arch(axa, y0, kind, col)

    # ---- (b) downstream accuracy -----------------------------------------
    axb = fig.add_axes([0.450, 0.330, 0.215, 0.550])
    axb.set_facecolor("none")
    x = np.arange(len(TASKS))
    for i, (name, col) in enumerate(STRATEGIES):
        axb.bar(x + (i - 1) * 0.27, SCORES[i], width=0.25, color=col,
                edgecolor="white", linewidth=0.5, label=name, zorder=3)
        for xi, v in zip(x, SCORES[i]):
            axb.text(xi + (i - 1) * 0.27, v + 1.1, f"{v:.1f}", ha="center",
                     va="bottom", fontsize=6.0,
                     color=col if i == 2 else C_SUB,
                     fontweight="bold" if i == 2 else "normal", rotation=90)
    axb.set_xticks(x)
    axb.set_xticklabels(TASKS, fontsize=6.8, linespacing=1.3)
    axb.set_ylim(0, 96)
    axb.set_ylabel("score (%)", fontsize=8.0)
    axb.tick_params(labelsize=6.8, length=2.2, width=0.6)
    axb.set_title("Downstream accuracy", fontsize=9.0, fontweight="bold",
                  pad=4.0)
    axb.spines["left"].set_linewidth(0.6)
    axb.spines["bottom"].set_linewidth(0.6)

    # ---- (c) attention on the queried object ------------------------------
    axc, cw, chh = inch_canvas(fig, [0.735, 0.330, 0.245, 0.550])
    photo, boxes = make_scene(71, "street")
    tgt = boxes["red_car"]
    wide = [(0, 0, W - 1, H - 1)]
    raw = [attention_map(wide + [tgt], [1.0, 0.55], spread=1.9, floor=0.30,
                         seed=5) * 0.42,
           attention_map([tgt], [1.0], spread=1.9, floor=0.16, seed=5) * 0.66,
           attention_map([tgt], [1.0], spread=0.85, floor=0.06, seed=5) * 1.0]
    vmax = max(m.max() for m in raw)
    tw = 1.28
    th = tw * H / W
    for i, ((name, col), a) in enumerate(zip(STRATEGIES, raw)):
        y = 0.28 + i * 1.15
        axc.text(0.02, y - 0.05, name, ha="left", va="bottom", fontsize=7.0,
                 color=col, fontweight="bold" if i == 2 else "normal")
        draw_photo(axc, photo, 0.02, y, tw)
        n = np.clip(a / vmax, 0, 1)
        im = axc.imshow(n, extent=(0.02, 0.02 + tw, y + th, y),
                        origin="upper", cmap=plt.get_cmap("magma"), vmin=0,
                        vmax=1, zorder=4, alpha=0.10 + 0.55 * n,
                        interpolation="bilinear")
        axc.text(0.02 + tw + 0.10, y + th / 2.0 - 0.10,
                 f"entropy\n{ENTROPY[i]:.2f}", ha="left", va="center",
                 fontsize=6.4, color=C_SUB, linespacing=1.35, **MONO)
    cax = fig.add_axes([0.755, 0.292, 0.145, 0.013])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, pad=1.2,
                      color="#333333")
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_label("attention on query object (-)", fontsize=7.0,
                 color="#1a1a1a", labelpad=1.5)

    # ---- subordinate: cost ------------------------------------------------
    names = [n for n, _c in STRATEGIES]
    cols = [c for _n, c in STRATEGIES]
    ypos = np.arange(3)[::-1]
    for rect, vals, lab, unit in (
            ([0.075, 0.078, 0.330, 0.150], PARAMS, "Parameters", "M"),
            ([0.560, 0.078, 0.330, 0.150], LATENCY, "Inference latency",
             "ms")):
        axd = fig.add_axes(rect)
        axd.set_facecolor("none")
        axd.barh(ypos, vals, height=0.58, color=cols, edgecolor="white",
                 linewidth=0.5)
        for y, v, c in zip(ypos, vals, cols):
            axd.text(v + max(vals) * 0.02, y, f"{v:.0f}", va="center",
                     ha="left", fontsize=6.4, color=c, **MONO)
        axd.set_yticks(ypos)
        axd.set_yticklabels(names, fontsize=6.8)
        axd.set_xlim(0, max(vals) * 1.22)
        axd.tick_params(labelsize=6.4, length=2.0, width=0.6)
        axd.set_xlabel(f"{lab.lower()} ({unit})", fontsize=7.0)
        axd.set_title(lab, fontsize=8.5, fontweight="bold", pad=3.5)
        axd.spines["left"].set_linewidth(0.6)
        axd.spines["bottom"].set_linewidth(0.6)
        if rect[0] < 0.5:
            axe = axd
        else:
            axf = axd

    for a, lab in ((axa, "(a)"), (axb, "(b)"), (axc, "(c)"), (axe, "(d)"),
                   (axf, "(e)")):
        panel_tag(fig, a, lab)
    savefig(fig, "349_vlm_fusion_ablation.png")


if __name__ == "__main__":
    main()
