"""Antipodal grasp-pose candidates ranked on eight household objects.

Each tile is one object synthesised from primitive geometry in a 100 mm frame.
Every candidate is drawn the way a parallel-jaw gripper actually closes: two
finger strokes along the approach direction, joined by the palm bar, separated
by the commanded jaw opening.  Candidates share one quality colour scale so a
weak grasp on the wrench cannot look like a strong grasp on the mug, and the
top-ranked pose per object is redrawn heavier with a white core.  The numbers
in each tile - the top score and the predicted success - are reduced from the
same score array that colours the strokes.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge

from pubstyle import set_pub_style, get_palette, savefig
from figkit import image_axes, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"

OBJ_FACE = "#c8bcae"      # neutral grey-brown object body
OBJ_EDGE = "#8a7f74"
OBJ_DARK = "#b0a396"

# Sequential green trimmed at the pale end: the first two stops of the house
# ramp are almost white and a 0.3-quality stroke drawn in them is invisible.
Q_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "grasp_q", get_palette("cb_sequential_green", 9)[3:])
Q_LO, Q_HI = 0.30, 1.00

FRAME = 100.0             # tile side, mm
FINGER = 17.0             # finger stroke length, mm
JAW_REF = 34.0            # jaw opening the ranker prefers, mm


# --------------------------------------------------------------------------
# object primitives
# --------------------------------------------------------------------------
def rot_rect(cx, cy, w, h, ang=0.0):
    a = np.deg2rad(ang)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                    [w / 2, h / 2], [-w / 2, h / 2]])
    return pts @ r.T + np.array([cx, cy])


def poly(ax, pts, fc=OBJ_FACE, ec=OBJ_EDGE, lw=0.8, z=2, alpha=1.0):
    ax.add_patch(Polygon(np.asarray(pts), closed=True, facecolor=fc,
                         edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha))


def draw_mug(ax):
    ax.add_patch(Wedge((70, 52), 15.0, -75, 75, width=5.5, facecolor="none",
                       edgecolor=OBJ_EDGE, linewidth=2.6, zorder=2))
    poly(ax, rot_rect(50, 50, 38, 46))
    poly(ax, rot_rect(50, 70, 38, 6), fc=OBJ_DARK)


def draw_box(ax):
    poly(ax, rot_rect(50, 50, 34, 58, 18))
    poly(ax, rot_rect(50, 50, 34, 4, 18), fc=OBJ_DARK, lw=0.6)


def draw_bottle(ax):
    poly(ax, rot_rect(50, 38, 30, 30))
    poly(ax, [(35, 53), (65, 53), (57, 66), (43, 66)])
    poly(ax, rot_rect(50, 72, 14, 14))
    poly(ax, rot_rect(50, 81, 17, 6), fc=OBJ_DARK)


def draw_ball(ax):
    ax.add_patch(Circle((50, 50), 25, facecolor=OBJ_FACE, edgecolor=OBJ_EDGE,
                        linewidth=0.8, zorder=2))
    ax.add_patch(Wedge((50, 50), 25, 200, 340, width=3.0, facecolor=OBJ_DARK,
                       edgecolor="none", zorder=3))


def draw_wrench(ax):
    poly(ax, rot_rect(50, 50, 12, 62, -20))
    for sy in (+1, -1):
        cx = 50 + sy * 0.34 * 62 * np.sin(np.deg2rad(20))
        cy = 50 + sy * 0.34 * 62 * np.cos(np.deg2rad(20))
        ax.add_patch(Wedge((cx, cy), 11.5, 70 - 20 + (0 if sy > 0 else 180),
                           290 - 20 + (0 if sy > 0 else 180), width=5.0,
                           facecolor=OBJ_FACE, edgecolor=OBJ_EDGE,
                           linewidth=0.8, zorder=3))


def draw_bowl(ax):
    ax.add_patch(Wedge((50, 62), 30, 180, 360, facecolor=OBJ_FACE,
                       edgecolor=OBJ_EDGE, linewidth=0.8, zorder=2))
    poly(ax, rot_rect(50, 62, 60, 5), fc=OBJ_DARK, lw=0.6)


def draw_can(ax):
    poly(ax, rot_rect(50, 48, 34, 50))
    poly(ax, rot_rect(50, 71, 34, 5), fc=OBJ_DARK, lw=0.6)


def draw_driver(ax):
    poly(ax, rot_rect(50, 32, 22, 34, -12))
    poly(ax, rot_rect(50, 62, 7, 34, -12), fc=OBJ_DARK)
    poly(ax, rot_rect(50, 79, 10, 5, -12), fc=OBJ_EDGE)


# name, draw function, candidate grasps (cx, cy, angle deg, jaw mm, fit 0-1)
OBJECTS = [
    ("Mug", draw_mug, [(50, 60, 0, 40, 0.94), (70, 52, 88, 20, 0.63),
                       (50, 40, 34, 52, 0.44), (62, 68, 12, 46, 0.36)]),
    ("Cereal box", draw_box, [(50, 58, 18, 36, 0.92), (50, 38, 18, 36, 0.83),
                              (50, 50, 108, 60, 0.30),
                              (44, 66, 62, 44, 0.41)]),
    ("Bottle", draw_bottle, [(50, 72, 0, 18, 0.96), (50, 38, 0, 32, 0.78),
                             (50, 59, 0, 42, 0.47), (56, 46, 40, 44, 0.34)]),
    ("Ball", draw_ball, [(50, 46, 0, 50, 0.74), (50, 50, 74, 50, 0.71),
                         (50, 62, 26, 42, 0.55), (36, 40, 40, 34, 0.36)]),
    ("Wrench", draw_wrench, [(50, 50, 70, 14, 0.93), (36, 30, 70, 26, 0.58),
                             (64, 70, 70, 26, 0.55), (50, 50, 160, 66, 0.28)]),
    ("Bowl", draw_bowl, [(50, 60, 0, 64, 0.55), (22, 60, 92, 12, 0.86),
                         (50, 46, 0, 44, 0.62), (34, 44, 40, 50, 0.38)]),
    ("Can", draw_can, [(50, 56, 0, 36, 0.95), (50, 34, 0, 36, 0.88),
                       (50, 71, 0, 38, 0.60), (58, 44, 46, 48, 0.42)]),
    ("Screwdriver", draw_driver, [(50, 32, 78, 24, 0.91),
                                  (50, 64, 78, 10, 0.66),
                                  (50, 48, 78, 20, 0.57),
                                  (44, 74, 30, 30, 0.33)]),
]


def score(cands, rng):
    """Rank candidates from geometry: antipodal fit, offset, jaw opening."""
    out = []
    for cx, cy, _, jaw, fit in cands:
        d = float(np.hypot(cx - 50.0, cy - 50.0))
        q = (0.30 + 0.66 * fit - 0.0032 * d - 0.0042 * abs(jaw - JAW_REF)
             + rng.normal(0.0, 0.018))
        out.append(float(np.clip(q, Q_LO, Q_HI)))
    return np.array(out)


def draw_grasp(ax, cx, cy, ang, jaw, color, lw=1.15, alpha=1.0, z=5,
               best=False):
    """Parallel-jaw pose: two finger strokes plus the palm bar joining them."""
    a = np.deg2rad(ang)
    ux, uy = np.cos(a), np.sin(a)             # jaw closing direction
    px, py = -uy, ux                          # approach direction
    hw, hl = jaw / 2.0, FINGER / 2.0
    for s in (+1.0, -1.0):
        fx, fy = cx + s * hw * ux, cy + s * hw * uy
        if best:
            ax.plot([fx - px * hl, fx + px * hl], [fy - py * hl, fy + py * hl],
                    color="white", lw=lw + 1.6, solid_capstyle="round",
                    zorder=z - 1)
        ax.plot([fx - px * hl, fx + px * hl], [fy - py * hl, fy + py * hl],
                color=color, lw=lw, alpha=alpha, solid_capstyle="round",
                zorder=z)
    bx0, by0 = cx - hw * ux - px * hl, cy - hw * uy - py * hl
    bx1, by1 = cx + hw * ux - px * hl, cy + hw * uy - py * hl
    if best:
        ax.plot([bx0, bx1], [by0, by1], color="white", lw=lw + 1.6,
                solid_capstyle="round", zorder=z - 1)
    ax.plot([bx0, bx1], [by0, by1], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=z)


def main():
    set_pub_style()
    rng = np.random.default_rng(7)
    norm = mpl.colors.Normalize(Q_LO, Q_HI)

    fig, axes = plt.subplots(2, 4, figsize=(10.4, 6.4))
    tops, succs = [], []

    for ax, (name, drawer, cands) in zip(axes.ravel(), OBJECTS):
        image_axes(ax, color="#cccccc", lw=0.6)
        # Frame wider than tall: with equal aspect a square window would make
        # the tiles as tall as the row pitch and the two rows would touch.
        ax.set_xlim(-9.0, FRAME + 9.0)
        ax.set_ylim(0, FRAME)
        drawer(ax)

        q = score(cands, rng)
        order = np.argsort(q)
        for k in order:
            cx, cy, ang, jaw, _ = cands[k]
            is_best = (k == order[-1])
            draw_grasp(ax, cx, cy, ang, jaw, Q_CMAP(norm(q[k])),
                       lw=2.1 if is_best else 1.0,
                       alpha=1.0 if is_best else 0.80,
                       z=7 if is_best else 5, best=is_best)

        top = float(q.max())
        succ = 100.0 * float(np.sort(q)[-2:].mean())
        tops.append(top)
        succs.append(succ)
        ax.set_title(name, fontsize=8.5, fontweight="bold", color=INK, pad=3.0)
        ax.text(-5.0, 4.0, f"$q^*$ = {top:.2f}   success {succ:.0f} %",
                fontsize=6.4, color=SUB, ha="left", va="bottom")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=Q_CMAP)
    fig.tight_layout(rect=[0, 0.125, 1, 0.945], h_pad=2.6, w_pad=2.2)
    # Equal aspect shrinks the tiles inside their slots; letters are placed
    # against the box that is actually drawn, not the nominal one.
    fig.canvas.draw()
    for ax in axes.ravel():
        ax.apply_aspect()
    panel_labels_inches(fig, axes, dx=-0.16, dy=0.06)

    cax = fig.add_axes([0.35, 0.088, 0.30, 0.016])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.5, color="#333333")
    cb.set_label("grasp quality score $q$ (-)", fontsize=6.8, color=INK)

    n_c = sum(len(o[2]) for o in OBJECTS)
    savefig(fig, "582_rob_grasp_poses.png")


if __name__ == "__main__":
    main()
