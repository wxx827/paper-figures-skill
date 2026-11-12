"""Figure 348 - Visual grounding of referring expressions.

Rows are samples, columns are image, referring expression, baseline
prediction, ours and ground truth.  The expression is broken into word chips
so the head noun and the relation word can be highlighted; predictions are
drawn as solid blue boxes against the dashed green ground truth, each tile
carrying its IoU.  The ours column sits on a tinted background.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from fig_vlm_image_text_align import (make_scene, inch_canvas, draw_photo,
                                      text_card, region_box, panel_tag,
                                      C_PRED, C_GT, C_OBJECT, C_SCENE, C_INK,
                                      C_SUB, C_FAINT, MONO, H, W)

SAMPLES = [
    dict(kind="beach", seed=11, target="horse",
         words=["the", "man", "riding", "the", "brown", "horse"],
         key={4: C_OBJECT, 5: C_OBJECT},
         base=(0.42, -0.30, 1.34), ours=(0.03, 0.02, 1.06),
         iou=(0.42, 0.88)),
    dict(kind="street", seed=71, target="red_car",
         words=["the", "red", "car", "closest", "to", "the", "camera"],
         key={1: C_OBJECT, 2: C_OBJECT, 3: C_SCENE},
         base=(-0.95, -0.55, 1.55), ours=(0.04, -0.03, 1.08),
         iou=(0.31, 0.91)),
    dict(kind="park", seed=44, target="dog",
         words=["the", "dog", "on", "the", "grass", "left", "of", "the",
                "person"],
         key={1: C_OBJECT, 5: C_SCENE},
         base=(1.30, -0.10, 1.40), ours=(-0.05, 0.03, 1.05),
         iou=(0.38, 0.84)),
]
COLS = ["Image", "Referring expression", "Baseline", "Ours", "Ground truth"]
FIGSIZE = (11.0, 6.4)


def jitter(b, dx, dy, sc):
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    hw, hh = (b[2] - b[0]) / 2.0 * sc, (b[3] - b[1]) / 2.0 * sc
    cx += dx * hw
    cy += dy * hh
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def chips(ax, words, key, x, y, w, fs=6.8):
    """Lay the referring expression out as wrapping word chips."""
    cx, cy, hgt = x, y, 0.235
    for i, word in enumerate(words):
        cw = 0.075 * len(word) + 0.16
        if cx + cw > x + w:
            cx, cy = x, cy + hgt + 0.075
        col = key.get(i)
        text_card(ax, cx, cy, cw, hgt, word,
                  fc="#f6ece9" if col == C_OBJECT else
                  ("#f8f2e0" if col else "#f2f2f2"),
                  ec=col if col else "#d8d8d8", fs=fs, mono=True,
                  ha="center", color=col if col else C_SUB,
                  weight="bold" if col else "normal", lw=0.8, rounding=0.04)
        cx += cw + 0.06
    return cy + hgt


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=FIGSIZE)
    ax, cw, ch = inch_canvas(fig, [0.040, 0.045, 0.945, 0.855])

    gut, iw, ew, gap = 0.34, 1.72, 2.55, 0.13
    ih = iw * H / W
    xs = [gut, gut + iw + gap, gut + iw + gap + ew + gap,
          gut + 2 * iw + ew + 3 * gap, gut + 3 * iw + ew + 4 * gap]
    row_y = [0.36 + i * (ih + 0.20) for i in range(3)]

    ax.add_patch(Rectangle((xs[3] - 0.07, 0.10), iw + 0.14,
                           row_y[-1] + ih + 0.16 - 0.10,
                           facecolor="#eaf3ee", edgecolor="none", zorder=0))

    for j, name in enumerate(COLS):
        wdt = ew if j == 1 else iw
        ax.text(xs[j] + wdt / 2.0, 0.26, name, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color="#1f6a47" if name == "Ours" else C_INK)

    for i, s in enumerate(SAMPLES):
        img, boxes = make_scene(s["seed"], s["kind"])
        y = row_y[i]
        gt = boxes[s["target"]]
        ax.text(gut - 0.12, y + ih / 2.0, "ABC"[i], ha="right", va="center",
                fontsize=8.0, fontweight="bold", color=C_FAINT)

        draw_photo(ax, img, xs[0], y, iw)
        chips(ax, s["words"], s["key"], xs[1], y + 0.10, ew)

        for k, (key, iou) in enumerate((("base", s["iou"][0]),
                                        ("ours", s["iou"][1]))):
            x = xs[2 + k]
            draw_photo(ax, img, x, y, iw)
            region_box(ax, gt, x, y, iw, ih, color=C_GT, dashed=True, lw=1.2)
            region_box(ax, jitter(gt, *s[key]), x, y, iw, ih, color=C_PRED,
                       lw=1.5)
            ax.text(x + iw - 0.06, y + ih - 0.10, f"IoU {iou:.2f}",
                    ha="right", va="bottom", fontsize=6.4, color="white",
                    zorder=11, fontweight="bold" if k == 1 else "normal",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="#1a1a1acc", edgecolor="none"),
                    **MONO)
        draw_photo(ax, img, xs[4], y, iw)
        region_box(ax, gt, xs[4], y, iw, ih, color=C_GT, lw=1.6)

    ly = row_y[-1] + ih + 0.32
    handles = [Line2D([], [], color=C_PRED, lw=1.6, label="prediction"),
               Line2D([], [], color=C_GT, lw=1.4, ls=(0, (2.6, 1.6)),
                      label="ground truth"),
               Line2D([], [], color=C_OBJECT, lw=1.4,
                      label="head noun / attribute chip"),
               Line2D([], [], color=C_SCENE, lw=1.4,
                      label="spatial relation chip")]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(gut / cw, 1.0 - ly / ch), ncol=4,
                    frameon=False, fontsize=7.0, handlelength=1.5,
                    columnspacing=1.8, handletextpad=0.5)
    leg.set_clip_on(False)

    panel_tag(fig, ax, "(a)", dx=0.018)
    savefig(fig, "348_vlm_visual_grounding.png")


if __name__ == "__main__":
    main()
