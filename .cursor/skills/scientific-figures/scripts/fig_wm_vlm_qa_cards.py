"""Figure 1172 - Six question-answering cases, Ours against a baseline.

A 3 x 2 card grid.  Each card carries one procedural frame, the question with
its ground truth, and the two models' answers; a correct answer sits on a pale
blue strip behind a drawn tick, a wrong one on pale grey behind a drawn cross.
No red / green pair anywhere - it is the commonest colour-blind collision and
the batch's warm-area budget cannot afford the red either.

Six different rooms rather than six seeds of one: a kitchen and a workbench
seen from a head-mounted camera, a desk from a short lens, a bookshelf head
on, a dash camera on a motorway and a living room.  Two of the ground truths
are counted off the scene generator's own anchor list (`shelves` on the shelf
frame, `vehicle` anchors on the road frame), so the answer keyed as correct is
correct about the pixels that were actually drawn.

The footer tallies both models off ``CASES``; edit a verdict there and the
score follows.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, savefig
from _wm_scenes import build

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
DEEP = "#2f4a75"                       # the one emphasis colour: correct
STEEL = "#3d6b8e"

OK_FC, OK_EC = "#e3ebf4", "#9fb3cc"
NO_FC, NO_EC = "#ececec", "#cbcbcb"
NO_MARK = "#8a8a8a"

CARD_FC, CARD_EC = "#fafafa", "#d5d5d5"

# (scene kind, scene kwargs, camera overrides, seed, caption,
#  question, ground truth, ours answer, ours ok, baseline answer, base ok)
CASES = [
    ("ego", dict(room="kitchen"), {}, 3111, "head-mounted · kitchen",
     "Which side of the pan does the kettle stand on?", "right",
     "to the right of the pan", True, "to the left of the pan", False),
    ("tabletop", dict(), dict(wb=(0.962, 1.000, 1.058), sat=0.78,
                              contrast=1.14), 3112, "short lens · desk",
     "Which object on the desk stands furthest away?", "the bottle",
     "the green bottle at the back", True, "the open laptop", False),
    ("shelf", dict(shelves=3, plan=17), dict(wb=(0.950, 1.000, 1.070),
                                             sat=0.66, contrast=1.10),
     3113, "frontal · bookshelf",
     "How many shelves of this cabinet carry books?", "three",
     "three shelves", True, "two shelves", False),
    ("ego", dict(room="workbench"), {},      3114, "head-mounted · workbench",
     "Which tool lies closest to the head-mounted camera?", "the drill",
     "the cordless drill", True, "the cordless drill", True),
    ("road", dict(horizon=0.44, vpoint=0.52, block=15),
     dict(exposure=0.98), 3115, "dash camera · motorway",
     "How many vehicles are ahead of the ego car?", "three",
     "two vehicles", False, "three vehicles", True),
    ("ego", dict(room="living"), {}, 3116, "head-mounted · living room",
     "Is the cup to the left or the right of the tray?", "right",
     "on the right of the tray", True, "on the left of the tray", False),
]

W, H = 9.6, 6.8
MX, GAP = 0.34, 0.20
TOP, BOT = 6.16, 0.44
CW = (W - 2 * MX - 2 * GAP) / 3.0
CH = (TOP - BOT - GAP) / 2.0
PH, PW = 140, 286                      # thumbnail raster, matches the frame


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return fig, ax


def box(ax, x0, y0, w, h, fc="white", ec=HAIR, lw=0.8, r=0.055, zorder=2):
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


def tick_mark(ax, x, y, s=0.055, color=DEEP, lw=1.7, zorder=9):
    ax.plot([x - s, x - s * 0.22, x + s], [y + 0.02, y - s * 0.80,
                                           y + s * 0.92],
            color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder)


def cross_mark(ax, x, y, s=0.048, color=NO_MARK, lw=1.7, zorder=9):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)


def truth_of(kind, prm, anchors):
    """Ground truths that are countable are counted, not asserted."""
    if kind == "shelf":
        return f"{['zero', 'one', 'two', 'three', 'four'][prm['shelves']]}"
    if kind == "road":
        n = sum(1 for a in anchors if a["name"] == "vehicle")
        return f"{['zero', 'one', 'two', 'three', 'four'][n]}"
    return None


def draw_card(ax, k, case):
    kind, prm, cam, seed, cap, q, gt, a_ours, ok_ours, a_base, ok_base = case
    col, row = k % 3, k // 3
    x0 = MX + col * (CW + GAP)
    y0 = TOP - CH - row * (CH + GAP)

    box(ax, x0, y0, CW, CH, fc=CARD_FC, ec=CARD_EC, lw=0.8, r=0.07, zorder=2)

    sc = build(kind, seed=seed, shape=(PH, PW), cam=cam or None, **prm)
    counted = truth_of(kind, prm, sc["anchors"])
    if counted is not None:
        assert counted == gt, f"{kind}: drew {counted}, card claims {gt}"

    ph, pw = 1.22, CW - 0.20
    px, py = x0 + 0.10, y0 + CH - 0.09 - ph
    photo(ax, sc["rgb"], px, py, pw, ph)
    ax.text(px + 0.06, py + ph - 0.11, cap, ha="left", va="center",
            fontsize=6.2, color="#f2f2f2", zorder=10,
            bbox=dict(boxstyle="round,pad=0.26", facecolor="#2b2f33",
                      edgecolor="none", alpha=0.80))

    # -- question bar -------------------------------------------------------
    qh = 0.52
    qy = py - 0.05 - qh
    box(ax, x0 + 0.10, qy, CW - 0.20, qh, fc="white", ec=HAIR, lw=0.7,
        r=0.05, zorder=3)
    box(ax, x0 + 0.17, qy + qh - 0.29, 0.225, 0.225, fc=STEEL, ec="none",
        r=0.045, zorder=4)
    ax.text(x0 + 0.2825, qy + qh - 0.1775, "Q", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="white", zorder=5)
    ax.text(x0 + 0.45, qy + qh - 0.09, textwrap.fill(q, 40), ha="left",
            va="top", fontsize=6.9, color=INK, style="italic",
            linespacing=1.36, zorder=5)
    ax.text(x0 + CW - 0.17, qy + 0.085, f"ground truth: {gt}", ha="right",
            va="center", fontsize=6.1, color=MUTED, zorder=5)

    # -- the two answers ----------------------------------------------------
    ah = 0.33
    for j, (name, answer, ok) in enumerate((("Ours", a_ours, ok_ours),
                                            ("Baseline", a_base, ok_base))):
        ay = qy - 0.06 - (j + 1) * ah - j * 0.04
        fc, ec = (OK_FC, OK_EC) if ok else (NO_FC, NO_EC)
        box(ax, x0 + 0.10, ay, CW - 0.20, ah, fc=fc, ec=ec,
            lw=1.1 if ok else 0.8, r=0.05, zorder=3)
        ax.text(x0 + 0.19, ay + ah / 2, name, ha="left", va="center",
                fontsize=7.0, fontweight="bold",
                color=DEEP if ok else "#5f5f5f", zorder=5)
        ax.text(x0 + 0.72, ay + ah / 2, answer, ha="left", va="center",
                fontsize=6.9, color=INK if ok else SUB, style="italic",
                zorder=5)
        if ok:
            tick_mark(ax, x0 + CW - 0.24, ay + ah / 2)
        else:
            cross_mark(ax, x0 + CW - 0.24, ay + ah / 2)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    for k, case in enumerate(CASES):
        draw_card(ax, k, case)

    n = len(CASES)
    n_ours = sum(1 for c in CASES if c[8])
    n_base = sum(1 for c in CASES if c[10])

    miss = [c[4].split("·")[-1].strip() for c in CASES if not c[8]]
    lx = MX
    tick_mark(ax, lx + 0.07, 0.20)
    ax.text(lx + 0.20, 0.20, f"Ours  {n_ours} / {n} correct", ha="left",
            va="center", fontsize=7.4, color=INK)
    cross_mark(ax, lx + 1.66, 0.20)
    ax.text(lx + 1.79, 0.20, f"Baseline  {n_base} / {n} correct", ha="left",
            va="center", fontsize=7.4, color=SUB)
    ax.text(W - MX, 0.20,
            f"the {n - n_ours} of {n} that Ours misses is a counting "
            f"question, on the {', '.join(miss)} frame",
            ha="right", va="center", fontsize=6.5, color=MUTED)

    savefig(fig, "1172_wm_vlm_qa_cards.png")


if __name__ == "__main__":
    main()
