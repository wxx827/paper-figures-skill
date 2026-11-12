"""Figure 346 - Where the decoder looks while it writes the caption.

(a) the input photograph next to the attention coverage summed over all
decoding steps, with the caption the decoder produced.  (b) one frame per
generated word, each showing that step's attention map over the image on a
shared colour scale, with the attention peak ringed.  (c) the finished caption
and, as a subordinate panel, the peak attention weight per step.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from pubstyle import set_pub_style, savefig, lighten
from fig_vlm_image_text_align import (make_scene, attention_map, inch_canvas,
                                      draw_photo, text_card, panel_tag,
                                      bg_block, C_PERSON, C_OBJECT, C_ACTION,
                                      C_SCENE, C_INK, C_SUB, C_FAINT, MONO,
                                      H, W)

WORDS = ["A", "man", "riding", "a", "horse", "on", "the", "beach"]
PEAKS = [0.09, 0.34, 0.22, 0.11, 0.41, 0.18, 0.10, 0.27]
GROUP = {1: C_PERSON, 2: C_ACTION, 4: C_OBJECT, 7: C_SCENE}
FIGSIZE = (10.4, 6.4)


def step_maps(boxes):
    """Raw (un-normalised) attention field per generated word."""
    wide = [(0, 0, W - 1, H - 1)]
    man, horse = boxes["man"], boxes["horse"]
    between = ((man[0] + horse[0]) / 2, (man[1] + horse[1]) / 2,
               (man[2] + horse[2]) / 2, (man[3] + horse[3]) / 2)
    ground = (horse[0] - 26, horse[3] - 12, horse[2] + 26, H - 6)
    spec = [(wide, [1.0], 2.4), ([man], [1.0], 0.9), ([between], [1.0], 1.1),
            (wide, [1.0], 2.2), ([horse], [1.0], 0.85),
            ([ground], [1.0], 1.0), (wide, [1.0], 2.0),
            ([boxes["sand"], boxes["sea"]], [1.0, 0.55], 1.0)]
    out = []
    for (bx, wt, sp), pk in zip(spec, PEAKS):
        out.append(attention_map(bx, wt, spread=sp, floor=0.10, seed=3) * pk)
    return out


def to_fig(rect, canvas_w, canvas_h, x, y, w, h):
    """Inch coordinates inside an inch_canvas -> figure fractions."""
    return [rect[0] + x / canvas_w * rect[2],
            rect[1] + (1 - (y + h) / canvas_h) * rect[3],
            w / canvas_w * rect[2], h / canvas_h * rect[3]]


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=FIGSIZE)
    photo, boxes = make_scene(11, "beach")
    maps = step_maps(boxes)
    vmax = max(m.max() for m in maps)
    cover = np.clip(sum(maps) / sum(maps).max(), 0, 1)
    cmap = plt.get_cmap("magma")

    def overlay(ax, a, x, y, w, hh, vmx):
        n = np.clip(a / vmx, 0, 1)
        return ax.imshow(n, extent=(x, x + w, y + hh, y), origin="upper",
                         cmap=cmap, vmin=0, vmax=1.0, zorder=4,
                         alpha=0.10 + 0.62 * n, interpolation="bilinear")

    # ---- (a) input, coverage, caption ------------------------------------
    ra = [0.045, 0.590, 0.935, 0.310]
    axa, aw, ah = inch_canvas(fig, ra)
    pw = 2.00
    ph = pw * H / W
    draw_photo(axa, photo, 0.04, 0.30, pw)
    axa.text(0.04 + pw / 2, 0.24, "input image", ha="center", va="bottom",
             fontsize=7.0, color=C_SUB)
    draw_photo(axa, photo, 2.52, 0.30, pw)
    im = overlay(axa, cover, 2.52, 0.30, pw, ph, 1.0)
    axa.text(2.52 + pw / 2, 0.24,
             "attention coverage  $\\sum_t \\alpha_t$", ha="center",
             va="bottom", fontsize=7.0, color=C_SUB)

    cax = fig.add_axes(to_fig(ra, aw, ah, 2.52 + pw + 0.12, 0.30, 0.10, ph))
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333",
                      pad=1.5)
    cb.set_label("normalised attention weight (-)", fontsize=7.0,
                 color=C_INK, labelpad=2.0)
    cb.set_ticks([0, 0.5, 1.0])

    axa.text(5.30, 0.24, "generated caption, greedy decoding", ha="left",
             va="bottom", fontsize=7.0, color=C_SUB)
    cw_, chh_, gapx, gapy = 1.02, 0.30, 0.09, 0.22
    for i, word in enumerate(WORDS):
        r, c = divmod(i, 4)
        x = 5.30 + c * (cw_ + gapx)
        y = 0.34 + r * (chh_ + gapy)
        col = GROUP.get(i)
        text_card(axa, x, y, cw_, chh_, word,
                  fc=lighten(col, 0.86) if col else "#f4f4f4",
                  ec=col if col else "#d8d8d8", fs=8.0, mono=True,
                  ha="center", color=col if col else C_SUB,
                  weight="bold" if col else "normal", lw=0.8)
        if c < 3 and i < len(WORDS) - 1:
            axa.add_patch(FancyArrowPatch((x + cw_ + 0.012, y + chh_ / 2),
                                          (x + cw_ + gapx - 0.012,
                                           y + chh_ / 2),
                                          arrowstyle="-|>", mutation_scale=5,
                                          linewidth=0.8, color="#a0a0a0",
                                          zorder=5))

    # ---- (b) per-word attention strip ------------------------------------
    rb = [0.045, 0.305, 0.935, 0.245]
    axb, bw, bh = inch_canvas(fig, rb)
    fw = 1.13
    fh = fw * H / W
    gap = (bw - 8 * fw) / 7.0
    for i, word in enumerate(WORDS):
        x = i * (fw + gap)
        col = GROUP.get(i)
        text_card(axb, x, 0.0, fw, 0.26, word,
                  fc=lighten(col, 0.86) if col else "#f4f4f4",
                  ec=col if col else "#d8d8d8", fs=7.6, mono=True,
                  ha="center", color=col if col else C_SUB,
                  weight="bold" if col else "normal", lw=0.8)
        draw_photo(axb, photo, x, 0.34, fw)
        overlay(axb, maps[i], x, 0.34, fw, fh, vmax)
        j, k = np.unravel_index(np.argmax(maps[i]), maps[i].shape)
        axb.add_patch(Circle((x + (k + 0.5) / W * fw,
                              0.34 + (j + 0.5) / H * fh), 0.055,
                             facecolor="none", edgecolor="white",
                             linewidth=1.0, zorder=8))
        axb.text(x + fw / 2, 0.34 + fh + 0.055,
                 f"peak $\\alpha$ = {PEAKS[i]:.2f}", ha="center", va="top",
                 fontsize=6.4, color=C_SUB, **MONO)

    # ---- (c) finished caption + subordinate peak curve -------------------
    bg_block(fig, 0.020, 0.030, 0.980, 0.272)
    rc = [0.045, 0.060, 0.500, 0.170]
    axc, ccw, cch = inch_canvas(fig, rc)
    axc.text(0.02, 0.16, "finished caption", ha="left", va="bottom",
             fontsize=7.0, color=C_SUB)
    text_card(axc, 0.02, 0.26, ccw - 0.06, 0.42,
              "A man riding a horse on the beach .", fc="white",
              ec="#c8c8c8", fs=9.0, mono=True, ha="center", color=C_INK,
              lw=0.8)
    axc.text(0.02, 0.86,
             "attention moves rider $\\rightarrow$ horse $\\rightarrow$ "
             "background as the sentence unfolds", ha="left", va="top",
             fontsize=6.8, color=C_FAINT)

    axd = fig.add_axes([0.650, 0.078, 0.280, 0.145])
    axd.set_facecolor("none")
    xs = np.arange(1, len(WORDS) + 1)
    axd.plot(xs, PEAKS, color="#3b5b92", lw=1.4, marker="o", ms=3.4,
             mfc="white", mew=1.0, zorder=3)
    for i, (x, v) in enumerate(zip(xs, PEAKS)):
        if i in GROUP:
            axd.plot([x], [v], marker="o", ms=4.4, color=GROUP[i], zorder=4)
    axd.axhline(0.15, color="#808080", lw=0.7, ls=(0, (3, 2)), zorder=0)
    axd.text(8.35, 0.15, "content-word\nthreshold", fontsize=6.0,
             color=C_FAINT, va="center", ha="left", linespacing=1.2)
    axd.set_xticks(xs)
    axd.set_xticklabels(WORDS, fontsize=6.4, rotation=35, ha="right", **MONO)
    axd.set_ylim(0, 0.48)
    axd.set_yticks([0, 0.2, 0.4])
    axd.tick_params(labelsize=6.4, length=2.0, width=0.6)
    axd.set_ylabel("peak $\\alpha$ (-)", fontsize=7.0)
    axd.set_title("Peak attention weight per step", fontsize=8.5,
                  fontweight="bold", pad=3.5)
    axd.spines["left"].set_linewidth(0.6)
    axd.spines["bottom"].set_linewidth(0.6)

    for a, lab in ((axa, "(a)"), (axb, "(b)"), (axc, "(c)"), (axd, "(d)")):
        panel_tag(fig, a, lab)
    savefig(fig, "346_vlm_caption_attention.png")


if __name__ == "__main__":
    main()
