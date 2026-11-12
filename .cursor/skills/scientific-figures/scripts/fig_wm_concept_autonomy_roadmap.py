"""Capability roadmap L0-L5 for world models, riding an ascending ribbon.

Two gradient trapezoids do the rhetorical work: a deep-blue banner across the
top that thins as it moves right, and a pale ribbon along the bottom that
thickens as it moves right.  The cards sit on the pale ribbon, so their feet
climb; their heights grow with the level, so their heads climb faster.  The
reader gets the sense of ascent before reading a single word, which is the
whole point of a roadmap figure.

Each card is the house two-tone block -- deep header naming the level, white
body listing what that level buys and what it still cannot do -- closed by a
pinned footer carrying the one number that separates the levels, the rollout
horizon the model stays trustworthy over.  A status pill above each card says
how much of the level is actually deployed today.

Levels and horizons are an illustrative synthesis, not a measured benchmark.
"""
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from pubstyle import COLORS, darken, savefig, set_pub_style
from figkit import inch_canvas

FW, FH = 11.0, 6.2
ROUND = 0.070

# The house blue ladder: the cards deepen as the capability level rises, so
# the ramp carries the ordering even before the reader gets to the labels.
LADDER = ["#8fb0d6", "#7ba0cb", "#5b83b5", "#4a72a4", "#3d6294", "#2f4f7f"]

PHASES = {"Done": "#2e7d5b", "Testing": "#c9a227", "Research": "#8e6bbf"}

# Level name (two lines), phase, horizon label, horizon bar fraction, bullets.
#
# Bullet counts run 4, 5, 6, 7, 8, 9 and that is load-bearing, not incidental.
# The cards get taller with the level (that ascent is the whole figure), so a
# flat bullet count would leave the short cards looking padded and the tall
# ones hollow.  Growing the list in step with the height keeps the white space
# below the last bullet at a near-constant ~0.45 in on every card.  If you add
# or remove a bullet, add or remove one from its neighbours too.
LEVELS = [
    ("Passive\nPerception", "Done", "--", 0.04, [
        "encodes frames to latents",
        "reconstruction only",
        "no action conditioning",
        "no forward model at all"]),
    ("One-step\nPrediction", "Done", "0.1 s", 0.14, [
        "next-frame dynamics head",
        "teacher forcing at train",
        "trained on logged pairs",
        "no multi-step credit",
        "error compounds fast"]),
    ("Short-horizon\nRollout", "Done", "2 s", 0.30, [
        "stable for 10-30 steps",
        "latent state stays bounded",
        "good enough for MPC",
        "needs frequent regrounding",
        "drifts on novel objects",
        "no reward model yet"]),
    ("Long-horizon\nImagination", "Testing", "20 s", 0.50, [
        "policy trained in dream",
        "value learned on rollouts",
        "keeps objects persistent",
        "handles occlusion gaps",
        "reward head co-trained",
        "needs replay to stay sharp",
        "still one fixed domain"]),
    ("Counterfactual\nReasoning", "Testing", "2 min", 0.72, [
        "answers what-if queries",
        "intervenes on objects",
        "swaps agent intentions",
        "separates cause from cue",
        "supports do-calculus tests",
        "wants action-labelled data",
        "expensive to evaluate",
        "brittle off-distribution"]),
    ("Open-world\nGeneralisation", "Research", "open", 1.00, [
        "unseen scenes and physics",
        "no per-domain re-fitting",
        "composes novel objects",
        "infers unseen dynamics",
        "calibrated uncertainty",
        "self-corrects from error",
        "transfers across robots",
        "one model, many bodies",
        "not demonstrated yet"]),
]

X0, X1 = 0.30, 10.70                 # ribbon span
CARD_W, PITCH = 1.66, 1.74
CX = [1.15 + PITCH * i for i in range(6)]
HEAD_H, FOOT_H = 0.58, 0.72
TOP = [3.55 + 0.30 * i for i in range(6)]
BAND_TOP, BAND_BL, BAND_BR = 6.15, 5.05, 5.55     # deep banner
RIB_BOT, RIB_TL, RIB_TR = 0.34, 0.62, 1.05        # pale ribbon


def lerp_x(x, y_left, y_right):
    return y_left + (y_right - y_left) * (x - X0) / (X1 - X0)


def gradient_band(ax, verts, c0, c1, zorder):
    """Horizontal ramp painted over the bounding box, clipped to a polygon."""
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    cmap = LinearSegmentedColormap.from_list("ramp", [c0, c1])
    im = ax.imshow(np.linspace(0.0, 1.0, 512).reshape(1, -1),
                   extent=[min(xs), max(xs), min(ys), max(ys)], cmap=cmap,
                   interpolation="bilinear", zorder=zorder)
    clip = Polygon(verts, closed=True, facecolor="none", edgecolor="none",
                   zorder=zorder)
    ax.add_patch(clip)
    im.set_clip_path(clip)
    return im


def ink_on(color):
    """Legible text colour for a header strip of the given blue."""
    r, g, b = mpl.colors.to_rgb(color)
    return "#1f2c3a" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.56 else "white"


def pill(ax, cx, cy, text, color, w, h=0.23, z=6):
    ax.add_patch(FancyBboxPatch((cx - w / 2.0, cy - h / 2.0), w, h,
                                boxstyle="round,pad=0,rounding_size=0.105",
                                facecolor=color, edgecolor="none",
                                mutation_aspect=1.0, zorder=z))
    ax.text(cx, cy, text, ha="center", va="center", color="white",
            fontsize=6.9, fontweight="bold", zorder=z + 1)


def level_card(ax, cx, y0, top, head, bullets, horizon, frac, color, z=4):
    w = CARD_W
    h = top - y0
    x0 = cx - w / 2.0
    box = f"round,pad=0,rounding_size={ROUND}"
    ax.add_patch(FancyBboxPatch((x0 + 0.035, y0 - 0.042), w, h, boxstyle=box,
                                facecolor="#d5dce4", edgecolor="none",
                                mutation_aspect=1.0, zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=box,
                                facecolor="white", edgecolor="none",
                                mutation_aspect=1.0, zorder=z))
    hy = top - HEAD_H
    ax.add_patch(FancyBboxPatch((x0, hy), w, HEAD_H, boxstyle=box,
                                facecolor=color, edgecolor="none",
                                mutation_aspect=1.0, zorder=z + 1))
    ax.add_patch(Rectangle((x0, hy), w, ROUND, facecolor=color,
                           edgecolor="none", zorder=z + 2))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=box, facecolor="none",
                                edgecolor=color, linewidth=0.9,
                                mutation_aspect=1.0, zorder=z + 3))
    ax.text(cx, hy + HEAD_H / 2.0, head, ha="center", va="center",
            color=ink_on(color), fontsize=8.6, fontweight="bold",
            linespacing=1.32, zorder=z + 4)

    by = hy - 0.30
    for b in bullets:
        ax.add_patch(Rectangle((x0 + 0.16, by - 0.026), 0.052, 0.052,
                               facecolor=color, edgecolor="none",
                               zorder=z + 4))
        ax.text(x0 + 0.27, by, b, ha="left", va="center", fontsize=6.5,
                color="#454e5a", zorder=z + 4)
        by -= 0.26

    fy = y0 + FOOT_H
    ax.plot([x0 + 0.16, x0 + w - 0.16], [fy, fy], color="#dde2e8", lw=0.8,
            zorder=z + 4)
    ax.text(x0 + 0.16, fy - 0.15, "ROLLOUT HORIZON", ha="left", va="center",
            fontsize=6.0, fontweight="bold", color="#96a0ac", zorder=z + 4)
    ax.text(x0 + 0.16, fy - 0.40, horizon, ha="left", va="center",
            fontsize=9.0, fontweight="bold", color=darken(color, 0.10),
            zorder=z + 4)
    bar_w = w - 0.32
    ax.add_patch(Rectangle((x0 + 0.16, y0 + 0.14), bar_w, 0.072,
                           facecolor="#e7ebf0", edgecolor="none", zorder=z + 4))
    ax.add_patch(Rectangle((x0 + 0.16, y0 + 0.14), bar_w * frac, 0.072,
                           facecolor=color, edgecolor="none", zorder=z + 5))


def main():
    set_pub_style()
    fig, ax = inch_canvas(FW, FH)

    gradient_band(ax, [(X0, BAND_TOP), (X1, BAND_TOP), (X1, BAND_BR),
                       (X0, BAND_BL)], "#22394f", "#4b78ad", zorder=1)
    gradient_band(ax, [(X0, RIB_TL), (X1, RIB_TR), (X1, RIB_BOT),
                       (X0, RIB_BOT)], "#eef3f9", "#8fb0d2", zorder=1)

    ax.text(0.62, 5.72, "Roadmap to autonomous world modelling",
            ha="left", va="center", fontsize=12.4, fontweight="bold",
            color="white", zorder=3)
    ax.text(0.62, 5.36, "Six capability levels, and how far the field has "
            "actually got", ha="left", va="center", fontsize=8.2,
            color="#cfdcea", zorder=3)

    # the pills climb faster than the banner falls; the guide makes that read
    pill_y = [t + 0.24 for t in TOP]
    ax.plot(CX, pill_y, color="#b9c4d1", lw=1.0, ls=(0, (3.6, 2.4)), zorder=3)

    cols = LADDER
    for i, (name, phase, horizon, frac, bullets) in enumerate(LEVELS):
        y0 = lerp_x(CX[i], RIB_TL, RIB_TR) - 0.02
        level_card(ax, CX[i], y0, TOP[i], name, bullets, horizon, frac,
                   cols[i])
        pw = 0.30 + 0.088 * len(phase)
        pill(ax, CX[i], pill_y[i], phase.upper(), PHASES[phase], pw)
        ax.text(CX[i], 0.15, f"LEVEL {i}", ha="center", va="center",
                fontsize=8.4, fontweight="bold", color=COLORS["ink"], zorder=3)

    ax.text(X1 - 0.12, 5.86, "Horizons are indicative orders of\n"
            "magnitude, not benchmark numbers", ha="right", va="center",
            fontsize=6.8, color="#a9bdd2", linespacing=1.55, zorder=3)

    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    savefig(fig, "1083_wm_concept_autonomy_roadmap.png")


if __name__ == "__main__":
    main()
