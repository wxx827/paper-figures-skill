"""Figure 352 - Failure modes of the vision-language model.

One row per systematic failure mode: the image, the question with the wrong
answer and the ground truth, the attention map that produced it, and a grey
commentary card naming the cause.  The subordinate row underneath carries the
share of each failure mode, the per-mode accuracy against the baseline and the
shared attention scale.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig
from fig_vlm_image_text_align import (make_scene, attention_map, inch_canvas,
                                      draw_photo, text_card, region_box,
                                      panel_tag, bg_block, cross, BG_QUESTION,
                                      BG_ERROR, BG_GT, BG_NOTE, C_BAD, C_GT,
                                      C_INK, C_SUB, C_FAINT, MONO, H, W)

MODES = [
    dict(name="Spatial relation confusion", color="#3d6b8e", kind="street",
         seed=71, q="Is the blue car left of the red car?",
         a="no", gt="yes", peak=0.38, look=["blue_car", "red_car"],
         w=[1.0, 0.9], spread=1.9,
         why="Both vehicles fall inside one pooled region, so the\n"
             "left / right decision is taken from the language prior\n"
             "instead of the layout of the scene."),
    dict(name="Counting error", color="#c9a227", kind="table", seed=203,
         q="How many apples are on the plate?", a="three", gt="four",
         peak=0.31, look=["apple1", "apple2", "apple3"], w=[1.0, 0.95, 0.9],
         spread=1.0,
         why="Attention saturates after the third instance; the fourth\n"
             "apple never receives enough mass to be counted, a failure\n"
             "that grows with the number of objects."),
    dict(name="Fine-grained attribute", color="#8e6bbf", kind="beach",
         seed=11, q="What colour is the rider's jacket?", a="black",
         gt="navy blue", peak=0.44, look=["man"], w=[1.0], spread=1.6,
         why="The colour is read off a 16 x 16 patch average that mixes\n"
             "the jacket with the shadowed horse behind it, so nearby\n"
             "hues collapse onto one label."),
]
SHARE = [31.0, 24.0, 22.0, 23.0]
SHARE_LAB = ["Spatial", "Counting", "Attribute", "Other"]
SHARE_COL = ["#3d6b8e", "#c9a227", "#8e6bbf", "#c4c4c4"]
BASE_ACC = [48.2, 41.6, 55.3]
OURS_ACC = [63.9, 57.4, 68.1]
ACCENT = "#b23a48"
FIGSIZE = (11.0, 6.8)


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=FIGSIZE)
    bg_block(fig, 0.020, 0.030, 0.980, 0.285)

    ax, cw, ch = inch_canvas(fig, [0.035, 0.300, 0.945, 0.590])
    gut, iw, qw, aw2, nw = 0.10, 1.48, 2.45, 1.48, 3.10
    ih = iw * H / W
    xs = [gut, gut + iw + 0.12, gut + iw + qw + 0.24,
          gut + iw + qw + aw2 + 0.40]
    for x, w, name in zip(xs, (iw, qw, aw2, nw),
                          ("Image", "Question and answer", "Attention map",
                           "Failure analysis")):
        ax.text(x + w / 2.0, 0.20, name, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=C_INK)

    raw, photos, boxsets = [], [], []
    for m in MODES:
        photo, boxes = make_scene(m["seed"], m["kind"])
        photos.append(photo)
        boxsets.append(boxes)
        raw.append(attention_map([boxes[k] for k in m["look"]], m["w"],
                                 spread=m["spread"], floor=0.10, seed=9)
                   * m["peak"])
    vmax = max(r.max() for r in raw)

    for i, m in enumerate(MODES):
        y = 0.34 + i * (ih + 0.13)
        draw_photo(ax, photos[i], xs[0], y, iw)
        for k in m["look"]:
            region_box(ax, boxsets[i][k], xs[0], y, iw, ih, color=C_GT,
                       dashed=True, lw=1.0)

        cy = y + (ih - 0.80) / 2.0
        text_card(ax, xs[1], cy, qw, 0.24, f"Q: {m['q']}", fc=BG_QUESTION,
                  ec="#cfcfcf", fs=6.8, mono=True, pad=0.10, color=C_INK)
        text_card(ax, xs[1], cy + 0.28, qw, 0.24, f"A: {m['a']}",
                  fc=BG_ERROR, ec="#e0a9a5", fs=6.8, mono=True, pad=0.10,
                  color=C_INK, weight="bold")
        cross(ax, xs[1] + qw - 0.16, cy + 0.40, s=0.050, lw=1.4, zorder=6)
        text_card(ax, xs[1], cy + 0.56, qw, 0.24, f"GT: {m['gt']}", fc=BG_GT,
                  ec="#a8ccb4", fs=6.8, mono=True, pad=0.10, color=C_INK)

        draw_photo(ax, photos[i], xs[2], y, iw)
        n = np.clip(raw[i] / vmax, 0, 1)
        im = ax.imshow(n, extent=(xs[2], xs[2] + iw, y + ih, y),
                       origin="upper", cmap=plt.get_cmap("magma"), vmin=0,
                       vmax=1, zorder=4, alpha=0.10 + 0.58 * n,
                       interpolation="bilinear")

        text_card(ax, xs[3], y, nw, ih, "", fc=BG_NOTE, ec="#dcdcdc", lw=0.6,
                  fs=6.0, rounding=0.06)
        ax.add_patch(Rectangle((xs[3] + 0.10, y + 0.14), 0.10, 0.10,
                               facecolor=m["color"], edgecolor="none",
                               zorder=5))
        ax.text(xs[3] + 0.26, y + 0.19, m["name"], ha="left", va="center",
                fontsize=7.6, fontweight="bold", color=m["color"], zorder=5)
        ax.text(xs[3] + 0.12, y + 0.38, m["why"], ha="left", va="top",
                fontsize=6.8, color=C_SUB, linespacing=1.45, zorder=5)

    # ---- subordinate: share of each failure mode -------------------------
    axd = fig.add_axes([0.060, 0.048, 0.170, 0.230])
    axd.set_facecolor("none")
    wedges, _t = axd.pie(SHARE, colors=SHARE_COL, startangle=90,
                         counterclock=False, radius=0.70,
                         wedgeprops=dict(width=0.30, edgecolor="white",
                                         linewidth=0.9))
    axd.text(0, 0, "1 284\nerrors", ha="center", va="center", fontsize=6.8,
             color=C_SUB, linespacing=1.3, **MONO)
    for wgt, lab, sh, col in zip(wedges, SHARE_LAB, SHARE, SHARE_COL):
        ang = np.deg2rad((wgt.theta1 + wgt.theta2) / 2.0)
        axd.text(0.92 * np.cos(ang), 0.92 * np.sin(ang), f"{lab}\n{sh:.0f}%",
                 ha="center", va="center", fontsize=6.4, color=col,
                 linespacing=1.25, fontweight="bold")
    axd.set_xlim(-1.25, 1.25)
    axd.set_ylim(-1.15, 1.15)
    axd.set_title("Share of failure modes", fontsize=8.5, fontweight="bold",
                  pad=3.0)

    # ---- subordinate: per-mode accuracy -----------------------------------
    axb = fig.add_axes([0.365, 0.080, 0.270, 0.170])
    axb.set_facecolor("none")
    x = np.arange(3)
    axb.bar(x - 0.19, BASE_ACC, width=0.36, color="#b8b8b8",
            edgecolor="white", linewidth=0.5, label="baseline")
    axb.bar(x + 0.19, OURS_ACC, width=0.36, color=ACCENT, edgecolor="white",
            linewidth=0.5, label="Ours")
    for xi, (b, o) in enumerate(zip(BASE_ACC, OURS_ACC)):
        axb.text(xi - 0.19, b + 1.4, f"{b:.1f}", ha="center", va="bottom",
                 fontsize=6.0, color=C_SUB)
        axb.text(xi + 0.19, o + 1.4, f"{o:.1f}", ha="center", va="bottom",
                 fontsize=6.0, color=ACCENT, fontweight="bold")
    axb.set_xticks(x)
    axb.set_xticklabels(SHARE_LAB[:3], fontsize=6.8)
    axb.set_ylim(0, 88)
    axb.set_ylabel("accuracy (%)", fontsize=7.0)
    axb.tick_params(labelsize=6.5, length=2.0, width=0.6)
    axb.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2,
               handlelength=1.1, borderaxespad=0.1, columnspacing=1.0)
    axb.set_title("Accuracy on the three hard modes", fontsize=8.5,
                  fontweight="bold", pad=3.5)
    axb.spines["left"].set_linewidth(0.6)
    axb.spines["bottom"].set_linewidth(0.6)

    cax = fig.add_axes([0.735, 0.120, 0.190, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, pad=1.2,
                      color="#333333")
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_label("attention weight, shared across rows (-)", fontsize=7.0,
                 labelpad=1.5)
    fig.text(0.735, 0.215, "Reading the rows", fontsize=8.5,
             fontweight="bold", ha="left", va="bottom", color=C_INK)
    fig.text(0.735, 0.160,
             "green dashed boxes mark the objects the question is about;\n"
             "the attention column shares one scale, so the maps are\n"
             "directly comparable between failure modes.", fontsize=6.8,
             color=C_SUB, ha="left", va="bottom", linespacing=1.45)

    for a, lab in ((ax, "(a)"), (axd, "(b)"), (axb, "(c)")):
        panel_tag(fig, a, lab, dx=0.018 if a is ax else 0.030)
    savefig(fig, "352_vlm_failure_analysis.png")


if __name__ == "__main__":
    main()
