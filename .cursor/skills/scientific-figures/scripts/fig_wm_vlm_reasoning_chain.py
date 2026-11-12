"""Figure 1173 - A four-step reasoning chain, each step citing a real crop.

One head-mounted frame of a lab bench on the left; on the right, four step
cards inside a single rounded container.  Every card carries its step number,
the sentence the model produced, and a thumbnail of the image region that step
refers to - and the thumbnail is a *crop of the same raster* shown on the
left, taken around the same anchor the box on the left is drawn from.  A crop
therefore cannot drift away from the box it is supposed to magnify: they are
two views of one array.

The last card is the answer, styled apart with a pale blue fill.  The recap
under the frame counts its numbers off ``STEPS`` rather than restating them.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon

from pubstyle import set_pub_style, savefig
from _wm_scenes import build

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
DEEP = "#2f4a75"
TEAL = "#2a7f7a"
VIOLET = "#7b3294"
STEEL = "#3d6b8e"

CONTAINER = "#f2f2f2"
ANSWER_FC, ANSWER_EC = "#e3ebf4", "#9fb3cc"

QUESTION = ("Which item has to be moved before the tray can be pulled "
            "straight out towards the operator?")

# (colour, anchor to cite, crop widening, sentence)
STEPS = [
    (DEEP, "tray", 1.00,
     "Locate the tray: it lies across the middle of the bench with its long "
     "edge facing the operator."),
    (TEAL, "probe", 1.15,
     "The probe rests on the tray's right half, so pulling the tray would "
     "drag the probe with it."),
    (VIOLET, "module", 1.20,
     "The module box stands clear at the left and never crosses the "
     "pull-out path."),
    (STEEL, "probe", 2.30,
     "Answer: move the probe first. The tray is then unobstructed along the "
     "whole pull-out path."),
]

W, H = 9.0, 6.2
PH, PW = 205, 286                       # main frame raster
CH_, CW_ = 96, 130                      # crop thumbnail raster target ratio

CX0, CY0, CWD, CHT = 0.26, 0.30, 8.48, 5.30      # container
IX0, IY0, IWD, IHT = 0.46, 2.30, 3.46, 2.48      # main frame
SX0, SWD = 4.14, 4.60                            # step column
SHT, SGAP = 0.92, 0.12


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return fig, ax


def box(ax, x0, y0, w, h, fc="white", ec=HAIR, lw=0.8, r=0.06, zorder=2):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)
    return p


def photo(ax, rgb, x0, y0, w, h, ec="#9a9a9a", lw=0.7, r=0.04, zorder=4):
    clip = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor="none", edgecolor=ec, linewidth=lw,
                          zorder=zorder + 4)
    ax.add_patch(clip)
    im = ax.imshow(np.clip(rgb, 0, 1), extent=[x0, x0 + w, y0, y0 + h],
                   zorder=zorder, interpolation="bilinear")
    im.set_clip_path(clip)
    return clip


def step_badge(ax, x, y, n, color, r=0.135, zorder=8):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white",
                        linewidth=1.0, zorder=zorder))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=7.6,
            fontweight="bold", color="white", zorder=zorder + 1)


def crop_window(anchor, widen, ratio):
    """Pixel window around an anchor, at the thumbnail's aspect ratio."""
    hw = max(anchor["rx"] * 1.55 * widen, anchor["ry"] * 1.55 * widen / ratio)
    hh = hw * ratio
    cx = float(np.clip(anchor["cx"], hw, PW - hw))
    cy = float(np.clip(anchor["cy"], hh, PH - hh))
    x0, x1 = int(round(cx - hw)), int(round(cx + hw))
    y0, y1 = int(round(cy - hh)), int(round(cy + hh))
    return max(x0, 0), min(x1, PW), max(y0, 0), min(y1, PH)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    sc = build("ego", seed=1578, shape=(PH, PW), room="lab",
               cam=dict(exposure=1.00, contrast=1.14, sat=0.78,
                        wb=(0.960, 1.000, 1.068), dof=1.7, sharpen=1.15))
    rgbf = np.clip(sc["rgb"], 0, 1)
    anc = {a["name"]: a for a in sc["anchors"]}

    # ---- container + question --------------------------------------------
    box(ax, CX0, CY0, CWD, CHT, fc=CONTAINER, ec="#e2e2e2", lw=0.9, r=0.14,
        zorder=1)
    qh = 0.46
    qy = CY0 + CHT - 0.20 - qh
    box(ax, CX0 + 0.20, qy, CWD - 0.40, qh, fc="white", ec=HAIR, lw=0.8,
        r=0.06, zorder=2)
    box(ax, CX0 + 0.32, qy + 0.11, 0.24, 0.24, fc=STEEL, ec="none", r=0.05,
        zorder=3)
    ax.text(CX0 + 0.44, qy + 0.23, "Q", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color="white", zorder=4)
    ax.text(CX0 + 0.68, qy + 0.23, QUESTION, ha="left", va="center",
            fontsize=8.4, color=INK, style="italic", zorder=3)

    # ---- main frame with one box per cited region -------------------------
    clip = photo(ax, rgbf, IX0, IY0, IWD, IHT)
    seen = {}
    geo = []
    for k, (col, name, widen, _txt) in enumerate(STEPS):
        if name in seen:                       # step 4 re-cites step 2's box
            continue
        seen[name] = k
        a = anc[name]
        bw = 2.0 * a["rx"] * 0.92 / PW * IWD
        bh = 2.0 * a["ry"] * 0.92 / PH * IHT
        cx = IX0 + a["cx"] / PW * IWD
        cy = IY0 + (1.0 - a["cy"] / PH) * IHT
        geo.append((k, col, cx - bw / 2, cy - bh / 2, bw, bh))
    for k, col, bx, by, bw, bh in geo:
        rect = Rectangle((bx, by), bw, bh, facecolor="none", edgecolor=col,
                         linewidth=1.5, zorder=8)
        ax.add_patch(rect)
        rect.set_clip_path(clip)
    for k, col, bx, by, bw, bh in geo:
        # The badge perches on the box's top-left corner, but these boxes
        # overlap; if that corner falls inside a *different* region's box the
        # badge drops to the bottom-left instead, so a number never sits
        # inside somebody else's rectangle.
        px, py = bx, by + bh
        if any(k2 != k and x2 <= px <= x2 + w2 and y2 <= py <= y2 + h2
               for k2, _c2, x2, y2, w2, h2 in geo):
            py = by
        step_badge(ax, px, py, k + 1, col, r=0.105, zorder=10)
    ax.text(IX0 + IWD / 2, IY0 - 0.19,
            "input frame  ·  head-mounted camera, lab bench", ha="center",
            va="center", fontsize=6.8, color=MUTED, zorder=3)

    # ---- recap under the frame -------------------------------------------
    ry0 = 0.74
    rh = IY0 - 0.36 - ry0
    box(ax, IX0, ry0, IWD, rh, fc="white", ec=HAIR, lw=0.8, r=0.07, zorder=2)
    ax.text(IX0 + 0.16, ry0 + rh - 0.20, "chain summary", ha="left",
            va="center", fontsize=7.6, fontweight="bold", color=INK,
            zorder=3)
    cited = []
    for k, (col, name, _w, _t) in enumerate(STEPS):
        if name not in cited:
            cited.append(name)
    ax.text(IX0 + 0.16, ry0 + rh - 0.48,
            f"{len(STEPS)} steps  ·  {len(cited)} distinct regions cited  ·  "
            f"step {len(STEPS)} returns to region "
            f"{list(seen).index(STEPS[-1][1]) + 1}",
            ha="left", va="center", fontsize=7.0, color=SUB, zorder=3)
    lx = IX0 + 0.16
    for j, name in enumerate(cited):
        col = STEPS[j][0]
        ax.add_patch(Circle((lx + 0.07, ry0 + 0.44), 0.072, facecolor=col,
                            edgecolor="none", zorder=3))
        ax.text(lx + 0.19, ry0 + 0.44, name, ha="left", va="center",
                fontsize=7.0, color=SUB, zorder=3)
        lx += 0.19 + 0.062 * len(name) + 0.36
    ax.text(IX0 + 0.16, ry0 + 0.19,
            "every crop on the right is cut from this same frame",
            ha="left", va="center", fontsize=6.6, color=MUTED, zorder=3)

    # ---- step cards -------------------------------------------------------
    ratio = CH_ / CW_
    top = qy - 0.16
    for k, (col, name, widen, txt) in enumerate(STEPS):
        y0 = top - (k + 1) * SHT - k * SGAP
        last = (k == len(STEPS) - 1)
        box(ax, SX0, y0, SWD, SHT,
            fc=ANSWER_FC if last else "white",
            ec=ANSWER_EC if last else HAIR,
            lw=1.2 if last else 0.8, r=0.07, zorder=2)
        step_badge(ax, SX0 + 0.30, y0 + SHT / 2, k + 1, col)
        ax.text(SX0 + 0.52, y0 + SHT / 2, textwrap.fill(txt, 44),
                ha="left", va="center", fontsize=7.3, color=INK,
                linespacing=1.42, zorder=3,
                fontweight="bold" if last else "normal")

        # crop of the very same raster, boxed in the very same colour
        x0p, x1p, y0p, y1p = crop_window(anc[name], widen, ratio)
        crop = rgbf[y0p:y1p, x0p:x1p]
        tw, th = 0.96, 0.96 * ratio
        tx, ty = SX0 + SWD - 0.14 - tw, y0 + SHT - 0.05 - th
        cclip = photo(ax, crop, tx, ty, tw, th, ec="white", lw=0.9, r=0.03,
                      zorder=5)
        a = anc[name]
        bx = tx + (a["cx"] - x0p - a["rx"] * 0.92) / (x1p - x0p) * tw
        by = ty + (1.0 - (a["cy"] - y0p + a["ry"] * 0.92) / (y1p - y0p)) * th
        # the box inside a crop is always the *region's* colour, not the
        # card's: step 4 re-cites region 2, so its crop must be teal like the
        # legend swatch, or the caption "region 2" points at a colour the
        # legend never introduced
        rcol = STEPS[seen[name]][0]
        r2 = Rectangle((bx, by), 2 * a["rx"] * 0.92 / (x1p - x0p) * tw,
                       2 * a["ry"] * 0.92 / (y1p - y0p) * th,
                       facecolor="none", edgecolor=rcol, linewidth=1.3,
                       zorder=9)
        ax.add_patch(r2)
        r2.set_clip_path(cclip)
        ax.text(tx + tw / 2, ty - 0.10, f"region {seen[name] + 1}",
                ha="center", va="center", fontsize=6.3, color=MUTED,
                zorder=6)

        if not last:                              # chevron down to the next
            cxs = SX0 + 0.30
            ymid = y0 - SGAP / 2
            ax.add_patch(Polygon([(cxs - 0.075, ymid + 0.035),
                                  (cxs + 0.075, ymid + 0.035),
                                  (cxs, ymid - 0.045)], closed=True,
                                 facecolor="#9a9a9a", edgecolor="none",
                                 zorder=4))


    savefig(fig, "1173_wm_vlm_reasoning_chain.png")


if __name__ == "__main__":
    main()
