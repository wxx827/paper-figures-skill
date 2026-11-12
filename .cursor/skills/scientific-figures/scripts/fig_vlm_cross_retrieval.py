"""Figure 344 - Cross-modal retrieval, both directions.

Top: two text queries, each with its top-5 retrieved images, correct hits in
green frames and distractors in red.  Bottom: one image query with its top-5
retrieved captions, drawn as bars whose length encodes cosine similarity.
The synthetic scenes come from the shared generator in figure 343.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from pubstyle import set_pub_style, savefig
from fig_vlm_image_text_align import (make_scene, inch_canvas, draw_photo,
                                      text_card, panel_tag, tick, cross,
                                      BG_ANSWER, BG_GT, BG_ERROR, C_OK, C_BAD,
                                      C_INK, C_SUB, C_FAINT, MONO, H, W)

# (query, [(scene kind, seed, similarity, is correct)])
T2I = [
    ("a brown horse walking\non the sandy beach",
     [("beach", 21, 0.412, True), ("beach", 33, 0.389, True),
      ("park", 44, 0.351, False), ("beach", 52, 0.334, True),
      ("street", 61, 0.298, False)]),
    ("a red car driving down\na city street",
     [("street", 71, 0.438, True), ("table", 82, 0.401, False),
      ("street", 93, 0.377, True), ("street", 104, 0.342, True),
      ("room", 115, 0.311, False)]),
]

I2T = [
    ("a brown dog running on the grass in a park", 0.451, True),
    ("a dog plays on a green lawn beside tall trees", 0.427, True),
    ("two people sitting on a park bench at noon", 0.390, False),
    ("a dog on the lawn with a person standing nearby", 0.362, True),
    ("a cat resting on a wooden bench outdoors", 0.318, False),
]


def frame(ax, x, y, w, h, ok, lw=1.6):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none",
                           edgecolor=C_OK if ok else C_BAD, linewidth=lw,
                           zorder=8))


def main():
    set_pub_style(base_font=8)
    figsize = (9.8, 5.9)
    fig = plt.figure(figsize=figsize)

    # ---- (a) text -> image ------------------------------------------------
    axa, cw, chh = inch_canvas(fig, [0.045, 0.395, 0.935, 0.505])
    qw, tw = 2.15, 1.20
    th = tw * H / W
    x0 = 2.55
    gapx = (cw - x0 - 5 * tw) / 4.0
    for k in range(5):
        axa.text(x0 + k * (tw + gapx) + tw / 2.0, 0.14, f"rank {k + 1}",
                 ha="center", va="bottom", fontsize=6.6, color=C_FAINT)

    for r, (query, hits) in enumerate(T2I):
        ry = 0.30 + r * 1.42
        text_card(axa, 0.02, ry + (th - 0.56) / 2.0, qw, 0.56, query,
                  fc=BG_ANSWER, ec="#9dbad6", fs=7.2, mono=True, ha="left",
                  pad=0.12, color=C_INK)
        axa.add_patch(FancyArrowPatch((0.02 + qw + 0.06, ry + th / 2.0),
                                      (x0 - 0.10, ry + th / 2.0),
                                      arrowstyle="-|>", mutation_scale=7,
                                      linewidth=1.0, color="#8a8a8a",
                                      zorder=5))
        for k, (kind, seed, sim, ok) in enumerate(hits):
            img, _b = make_scene(seed, kind)
            x = x0 + k * (tw + gapx)
            draw_photo(axa, img, x, ry, tw, ec="#ffffff", lw=0.0)
            frame(axa, x, ry, tw, th, ok)
            axa.add_patch(Circle((x + tw - 0.155, ry + 0.155), 0.105,
                                 facecolor="white", edgecolor=
                                 C_OK if ok else C_BAD, linewidth=0.8,
                                 alpha=0.92, zorder=9))
            if ok:
                tick(axa, x + tw - 0.155, ry + 0.155, s=0.055, lw=1.3,
                     zorder=10)
            else:
                cross(axa, x + tw - 0.155, ry + 0.155, s=0.048, lw=1.3,
                      zorder=10)
            axa.text(x + tw / 2.0, ry + th + 0.055, f"{sim:.3f}",
                     ha="center", va="top", fontsize=6.8,
                     color=C_OK if ok else C_BAD,
                     fontweight="bold" if k == 0 else "normal", **MONO)

    # ---- (b) image -> text ------------------------------------------------
    axb, cwb, chb = inch_canvas(fig, [0.045, 0.050, 0.935, 0.290])
    qimg, _qb = make_scene(44, "park")
    iw = 1.60
    ih = iw * H / W
    draw_photo(axb, qimg, 0.02, 0.30, iw, ec="#7a7a7a", lw=1.0)
    axb.text(0.02 + iw / 2.0, 0.30 + ih + 0.06, "image query", ha="center",
             va="top", fontsize=6.8, color=C_FAINT)
    axb.add_patch(FancyArrowPatch((0.02 + iw + 0.08, 0.30 + ih / 2.0),
                                  (2.02, 0.30 + ih / 2.0),
                                  arrowstyle="-|>", mutation_scale=7,
                                  linewidth=1.0, color="#8a8a8a", zorder=5))
    bx, bh, pitch = 2.20, 0.215, 0.250
    axb.text(bx, 0.16, "retrieved captions, bar length $\\propto$ cosine "
             "similarity", ha="left", va="bottom", fontsize=6.6, color=C_FAINT)
    for k, (cap, sim, ok) in enumerate(I2T):
        y = 0.30 + k * pitch
        bw = 3.05 + 7.4 * (sim - 0.30)
        text_card(axb, bx, y, bw, bh, cap, fc=BG_GT if ok else BG_ERROR,
                  ec=C_OK if ok else C_BAD, fs=6.8, mono=True, pad=0.12,
                  lw=1.0, color=C_INK)
        axb.text(bx + bw + 0.10, y + bh / 2.0, f"{sim:.3f}", ha="left",
                 va="center", fontsize=6.8, color=C_OK if ok else C_BAD,
                 fontweight="bold" if k == 0 else "normal", **MONO)
        axb.text(bx - 0.10, y + bh / 2.0, f"{k + 1}", ha="right", va="center",
                 fontsize=6.6, color=C_FAINT, **MONO)

    for a, lab, title in ((axa, "(a)", "Text $\\rightarrow$ image retrieval"),
                          (axb, "(b)", "Image $\\rightarrow$ text retrieval")):
        panel_tag(fig, a, lab)
        p = a.get_position()
        fig.text(p.x0 + 0.028, p.y1 + 0.010, title, fontsize=8.5,
                 fontweight="bold", ha="left", va="bottom", color=C_INK)

    savefig(fig, "344_vlm_cross_retrieval.png")


if __name__ == "__main__":
    main()
