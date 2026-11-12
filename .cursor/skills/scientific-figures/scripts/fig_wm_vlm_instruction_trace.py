"""Figure 1175 - One instruction, four stages, and where the time went.

Top: the instruction.  Middle: parse, ground, plan and execute, each card
carrying the artefact that stage produced - the tagged phrase chips, the frame
with the target and the reference boxed, the top-down waypoint plan, and the
same frame with the executed path drawn over it.  Bottom: a proportional
timeline; a dashed drop line ties each card to the slice of time it consumed.

Two things are derived rather than asserted.  The waypoints are computed once
from the mug and notebook anchors and are then drawn in *both* the plan panel
and the execute overlay, so the plan and the execution cannot disagree.  And
the timeline is laid out from ``STAGES``: the total is their sum, the segment
widths are their fractions of it, and the footer prints the addition so a
reader can check it in one pass.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Rectangle, Circle, Polygon,
                                FancyArrowPatch)

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
CARD = "#f5f5f5"
PALE = "#e8edf3"

INSTRUCTION = ("Move the mug to the left of the monitor, keeping it clear "
               "of the keyboard.")
N_WP = 5

# (stage name, colour, seconds, caption)
STAGES = [
    ("parse", DEEP, 0.42,
     "The sentence is tagged into an action, a target, a spatial relation "
     "and a reference."),
    ("ground", TEAL, 1.16,
     "Target and reference are matched to two boxes in the current frame."),
    ("plan", VIOLET, 0.68,
     f"{N_WP} waypoints lift the mug over the keyboard and set it down left "
     f"of the monitor."),
    ("execute", STEEL, 2.34,
     "The arm follows the plan; the dashed ring is where the mug ends up."),
]

CHIPS = [("action", "move"), ("target", "the mug"),
         ("relation", "to the left of"), ("reference", "the monitor")]

W, H = 11.0, 4.8
PH, PW = 132, 250                       # frame raster
MX = 0.32
CGAP = 0.26
CWD = (W - 2 * MX - 3 * CGAP) / 4.0
CY0, CHT = 1.60, 2.06
TWD, THT = CWD - 0.24, 1.06             # artefact panel inside a card


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


def photo(ax, rgb, x0, y0, w, h, ec="#9a9a9a", lw=0.7, r=0.03, zorder=4):
    clip = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor="none", edgecolor=ec, linewidth=lw,
                          zorder=zorder + 5)
    ax.add_patch(clip)
    im = ax.imshow(np.clip(rgb, 0, 1), extent=[x0, x0 + w, y0, y0 + h],
                   zorder=zorder, interpolation="bilinear")
    im.set_clip_path(clip)
    return clip


def waypoints(src, dst, n=N_WP):
    """A lift-carry-place arc between two image points, in frame fractions."""
    t = np.linspace(0.0, 1.0, n)
    x = src[0] + (dst[0] - src[0]) * t
    y = src[1] + (dst[1] - src[1]) * t - 0.30 * np.sin(np.pi * t)
    return np.stack([x, y], axis=1)


def draw_parse(ax, x0, y0, w, h, col):
    box(ax, x0, y0, w, h, fc="white", ec=HAIR, lw=0.7, r=0.04, zorder=4)
    ch = 0.195
    for j, (role, word) in enumerate(CHIPS):
        cy = y0 + h - 0.14 - j * (ch + 0.055) - ch / 2
        ax.text(x0 + 0.10, cy, role, ha="left", va="center", fontsize=6.2,
                color=MUTED, zorder=6)
        cx = x0 + 0.62
        box(ax, cx, cy - ch / 2, w - 0.72, ch, fc=PALE, ec=col, lw=0.9,
            r=0.035, zorder=5)
        ax.text(cx + 0.09, cy, word, ha="left", va="center", fontsize=6.8,
                color=INK, fontweight="bold", zorder=6)


def draw_plan(ax, x0, y0, w, h, col, wp):
    box(ax, x0, y0, w, h, fc="#fbfbfc", ec=HAIR, lw=0.7, r=0.04, zorder=4)
    for g in range(1, 6):                       # top-down floor grid
        gx = x0 + g * w / 6.0
        ax.plot([gx, gx], [y0 + 0.05, y0 + h - 0.05], color="#e6e6e6",
                lw=0.5, zorder=5)
    for g in range(1, 4):
        gy = y0 + g * h / 4.0
        ax.plot([x0 + 0.05, x0 + w - 0.05], [gy, gy], color="#e6e6e6",
                lw=0.5, zorder=5)
    px = x0 + 0.10 + wp[:, 0] * (w - 0.20)
    py = y0 + 0.12 + (1.0 - wp[:, 1]) * (h - 0.30)
    ax.plot(px, py, color=col, lw=1.4, ls=(0, (4.0, 2.2)), zorder=7)
    for j in range(len(px)):
        ax.add_patch(Circle((px[j], py[j]), 0.038, facecolor="white",
                            edgecolor=col, linewidth=1.1, zorder=8))
    ax.add_patch(Circle((px[0], py[0]), 0.050, facecolor=col,
                        edgecolor="white", linewidth=0.8, zorder=9))
    ax.add_patch(Circle((px[-1], py[-1]), 0.072, facecolor="none",
                        edgecolor=col, linewidth=1.3, ls=(0, (2.4, 1.6)),
                        zorder=9))
    ax.text(px[0], py[0] - 0.13, "start", ha="center", va="top",
            fontsize=6.0, color=MUTED, zorder=9)
    ax.text(px[-1], py[-1] + 0.14, "goal", ha="center", va="bottom",
            fontsize=6.0, color=col, zorder=9)
    ax.text(x0 + w - 0.08, y0 + 0.09, "side view along the bench",
            ha="right", va="bottom", fontsize=6.0, color=MUTED, zorder=9)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    sc = build("ego", seed=1580, shape=(PH, PW), room="desk",
               cam=dict(exposure=1.01, contrast=1.13, sat=0.78,
                        wb=(0.960, 1.000, 1.066), dof=1.6, sharpen=1.15))
    rgbf = np.clip(sc["rgb"], 0, 1)
    anc = {a["name"]: a for a in sc["anchors"]}

    mug, ref = anc["mug"], anc["monitor"]
    src = (mug["cx"] / PW, mug["cy"] / PH)
    # left of the reference, but still on the bench: keep the mug's own height
    dst = ((ref["cx"] - 1.30 * ref["rx"]) / PW, mug["cy"] / PH)
    WP = waypoints(src, dst)

    # ---- instruction banner ----------------------------------------------
    bh = 0.46
    by = 4.26 - bh
    box(ax, MX, by, W - 2 * MX, bh, fc="white", ec=HAIR, lw=0.9, r=0.07,
        zorder=2)
    box(ax, MX + 0.12, by + 0.09, 0.86, 0.28, fc=DEEP, ec="none", r=0.05,
        zorder=3)
    ax.text(MX + 0.55, by + 0.23, "instruction", ha="center", va="center",
            fontsize=6.8, fontweight="bold", color="white", zorder=4)
    ax.text(MX + 1.12, by + 0.23, "\u201c" + INSTRUCTION + "\u201d",
            ha="left", va="center", fontsize=9.0, color=INK, style="italic",
            zorder=3)
    ax.text(W - MX - 0.14, by + 0.23, "single utterance, no follow-up",
            ha="right", va="center", fontsize=6.6, color=MUTED, zorder=3)

    # ---- four stage cards -------------------------------------------------
    total = sum(s[2] for s in STAGES)
    centres = []
    for k, (name, col, secs, cap) in enumerate(STAGES):
        x0 = MX + k * (CWD + CGAP)
        centres.append(x0 + CWD / 2)
        box(ax, x0, CY0, CWD, CHT, fc=CARD, ec="#e4e4e4", lw=0.9, r=0.08,
            zorder=1)
        box(ax, x0, CY0 + CHT - 0.30, CWD, 0.30, fc=col, ec="none", r=0.08,
            zorder=2)
        ax.add_patch(Rectangle((x0, CY0 + CHT - 0.30), CWD, 0.10,
                               facecolor=col, edgecolor="none", zorder=2))
        ax.text(x0 + 0.14, CY0 + CHT - 0.15, f"{k + 1}", ha="left",
                va="center", fontsize=7.0, fontweight="bold",
                color="#d6dee8", zorder=4)
        ax.text(x0 + 0.34, CY0 + CHT - 0.15, name, ha="left", va="center",
                fontsize=8.2, fontweight="bold", color="white", zorder=4)
        ax.text(x0 + CWD - 0.14, CY0 + CHT - 0.15, f"{secs:.2f} s",
                ha="right", va="center", fontsize=6.8, color="#e2e8ef",
                zorder=4)

        tx, ty = x0 + 0.12, CY0 + CHT - 0.30 - 0.10 - THT
        if k == 0:
            draw_parse(ax, tx, ty, TWD, THT, col)
        elif k == 2:
            draw_plan(ax, tx, ty, TWD, THT, col, WP)
        else:
            clip = photo(ax, rgbf, tx, ty, TWD, THT)
            if k == 1:
                for a, lab, c in ((mug, "target", col),
                                  (ref, "reference", "#8fa7bd")):
                    bw = 2.0 * a["rx"] * 0.90 / PW * TWD
                    bhh = 2.0 * a["ry"] * 0.90 / PH * THT
                    cx = tx + a["cx"] / PW * TWD
                    cy = ty + (1.0 - a["cy"] / PH) * THT
                    r = Rectangle((cx - bw / 2, cy - bhh / 2), bw, bhh,
                                  facecolor="none", edgecolor=c,
                                  linewidth=1.4, zorder=10)
                    ax.add_patch(r)
                    r.set_clip_path(clip)
                    # the monitor sits high enough that a caption above its
                    # box straddles the photo's top border; when there is no
                    # headroom the caption goes under the box instead
                    ly = cy + bhh / 2 + 0.055
                    if ly + 0.055 > ty + THT:
                        ly = cy - bhh / 2 - 0.055
                    t = ax.text(cx, ly, lab, ha="center",
                                va="center", fontsize=6.0, color=c,
                                zorder=11,
                                bbox=dict(facecolor="white", edgecolor="none",
                                          pad=1.1, alpha=0.92))
                    t.set_clip_on(False)
            else:
                px = tx + WP[:, 0] * TWD
                py = ty + (1.0 - WP[:, 1]) * THT
                ln, = ax.plot(px, py, color="white", lw=2.4, zorder=9,
                              alpha=0.75)
                ln.set_clip_path(clip)
                ln2, = ax.plot(px, py, color=col, lw=1.5,
                               ls=(0, (4.0, 2.2)), zorder=10)
                ln2.set_clip_path(clip)
                head = FancyArrowPatch((px[-2], py[-2]), (px[-1], py[-1]),
                                       arrowstyle="-|>", mutation_scale=8,
                                       color=col, linewidth=1.5, shrinkA=0,
                                       shrinkB=0, zorder=11)
                ax.add_patch(head)
                head.set_clip_path(clip)
                gr = Circle((px[-1], py[-1]), 0.085, facecolor="none",
                            edgecolor=col, linewidth=1.3, ls=(0, (2.4, 1.6)),
                            zorder=11)
                ax.add_patch(gr)
                gr.set_clip_path(clip)
            ax.text(tx + TWD - 0.06, ty + 0.07,
                    "frame  t = 0.00 s" if k == 1 else
                    f"frame  t = {total:.2f} s", ha="right", va="bottom",
                    fontsize=6.0, color="#f0f0f0", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#2b2f33",
                              edgecolor="none", alpha=0.78))

        ax.text(x0 + 0.12, ty - 0.10, textwrap.fill(cap, 40), ha="left",
                va="top", fontsize=6.7, color=SUB, linespacing=1.40,
                zorder=4)

        if k < len(STAGES) - 1:
            mid = x0 + CWD + CGAP / 2
            ax.add_patch(Polygon([(mid - 0.055, CY0 + CHT / 2 + 0.085),
                                  (mid - 0.055, CY0 + CHT / 2 - 0.085),
                                  (mid + 0.062, CY0 + CHT / 2)], closed=True,
                                 facecolor="#9a9a9a", edgecolor="none",
                                 zorder=5))

    # ---- proportional timeline -------------------------------------------
    ty0, thh = 0.74, 0.34
    tx0, twd = MX, W - 2 * MX
    acc = 0.0
    for k, (name, col, secs, cap) in enumerate(STAGES):
        sx = tx0 + acc / total * twd
        sw = secs / total * twd
        ax.add_patch(FancyBboxPatch(
            (sx + 0.008, ty0), sw - 0.016, thh,
            boxstyle="round,pad=0,rounding_size=0.035", facecolor=col,
            edgecolor="none", zorder=3))
        ax.text(sx + sw / 2, ty0 + thh / 2, f"{secs:.2f} s", ha="center",
                va="center", fontsize=6.8, fontweight="bold", color="white",
                zorder=5)
        ax.plot([centres[k], centres[k]], [CY0 - 0.06, ty0 + thh + 0.06],
                color="#bfbfbf", lw=0.7, ls=(0, (2.2, 2.0)), zorder=2)
        ax.text(sx + sw / 2, ty0 + thh + 0.10, name, ha="center",
                va="bottom", fontsize=6.6, color=col, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                          alpha=0.94))
        acc += secs

    ax.plot([tx0, tx0 + twd], [ty0 - 0.09, ty0 - 0.09], color="#8a8a8a",
            lw=0.8, zorder=3)
    for t in (0.0, 1.0, 2.0, 3.0, 4.0, total):
        gx = tx0 + t / total * twd
        ax.plot([gx, gx], [ty0 - 0.09, ty0 - 0.15], color="#8a8a8a", lw=0.8,
                zorder=3)
        ax.text(gx, ty0 - 0.21, f"{t:.2f}" if t == total else f"{t:.0f}",
                ha="center", va="top", fontsize=6.4, color=SUB, zorder=3)
    ax.text(tx0, ty0 - 0.40, "wall-clock time since the utterance (s)",
            ha="left", va="center", fontsize=6.8, color=SUB)
    ax.text(tx0 + twd, ty0 - 0.40,
            "  +  ".join(f"{s[2]:.2f}" for s in STAGES)
            + f"  =  {total:.2f} s end to end", ha="right", va="center",
            fontsize=6.8, color=MUTED)


    savefig(fig, "1175_wm_vlm_instruction_trace.png")


if __name__ == "__main__":
    main()
