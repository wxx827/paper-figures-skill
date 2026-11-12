"""Figure 345 - Visual question answering, qualitative examples.

A 3 x 2 grid of cells.  Each cell stacks the image, the question in a grey
card, the model answer in a blue card (red when wrong) and the ground truth in
a green card, with a geometric tick or cross.  Coloured boxes on the image mark
the region the model attends to and, when the answer is wrong, the region it
should have used.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig
from fig_vlm_image_text_align import (make_scene, inch_canvas, draw_photo,
                                      text_card, panel_tag, tick, cross,
                                      region_box, BG_QUESTION, BG_ANSWER,
                                      BG_GT, BG_ERROR, C_PRED, C_GT, C_INK,
                                      C_SUB, H, W)

EXAMPLES = [
    dict(kind="table", seed=203, q="How many apples are on the plate?",
         a="four", gt="four", ok=True, look="apples"),
    dict(kind="beach", seed=11, q="What is the man riding on the beach?",
         a="a horse", gt="a horse", ok=True, look="horse"),
    dict(kind="street", seed=71, q="What colour is the closest car?",
         a="red", gt="red", ok=True, look="red_car"),
    dict(kind="park", seed=44, q="What is the dog doing on the lawn?",
         a="sitting", gt="running", ok=False, look="bench", truth="dog"),
    dict(kind="room", seed=310, q="Where is the cat in this room?",
         a="on the floor", gt="on the sofa", ok=False,
         look=(24, 128, 92, 162), truth="cat"),
    dict(kind="street", seed=93, q="Is the traffic light showing green?",
         a="yes", gt="yes", ok=True, look="traffic_light"),
]

FIGSIZE = (10.4, 7.6)
CELL_W, CELL_H = 2.95, 3.08


def draw_cell(fig, rect, ex):
    ax, cw, ch = inch_canvas(fig, rect)
    img, boxes = make_scene(ex["seed"], ex["kind"])
    iw = cw
    ih = iw * H / W
    draw_photo(ax, img, 0.0, 0.02, iw)

    def region(key, color, dashed=False):
        b = boxes[key] if isinstance(key, str) else key
        region_box(ax, b, 0.0, 0.02, iw, ih, color=color, dashed=dashed,
                   lw=1.4)

    region(ex["look"], C_PRED)
    if "truth" in ex:
        region(ex["truth"], C_GT, dashed=True)

    y = 0.02 + ih + 0.08
    text_card(ax, 0.0, y, cw, 0.24, f"Q: {ex['q']}", fc=BG_QUESTION,
              ec="#cfcfcf", fs=6.8, mono=True, pad=0.10, color=C_INK)
    y += 0.265
    text_card(ax, 0.0, y, cw, 0.24, f"A: {ex['a']}",
              fc=BG_ANSWER if ex["ok"] else BG_ERROR,
              ec="#9dbad6" if ex["ok"] else "#e0a9a5", fs=6.8, mono=True,
              pad=0.10, color=C_INK, weight="bold")
    if ex["ok"]:
        tick(ax, cw - 0.16, y + 0.12, s=0.058, lw=1.4, zorder=6)
    else:
        cross(ax, cw - 0.16, y + 0.12, s=0.050, lw=1.4, zorder=6)
    y += 0.265
    text_card(ax, 0.0, y, cw, 0.24, f"GT: {ex['gt']}", fc=BG_GT, ec="#a8ccb4",
              fs=6.8, mono=True, pad=0.10, color=C_INK)
    return ax


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=FIGSIZE)
    wf, hf = CELL_W / FIGSIZE[0], CELL_H / FIGSIZE[1]
    gx, gy = 0.30 / FIGSIZE[0], 0.30 / FIGSIZE[1]
    top = 0.900
    axes = []
    for i, ex in enumerate(EXAMPLES):
        r, c = divmod(i, 3)
        rect = [0.045 + c * (wf + gx), top - hf - r * (hf + gy), wf, hf]
        axes.append(draw_cell(fig, rect, ex))
    for ax, lab in zip(axes, "(a) (b) (c) (d) (e) (f)".split()):
        panel_tag(fig, ax, lab)

    savefig(fig, "345_vlm_vqa_examples.png")


if __name__ == "__main__":
    main()
