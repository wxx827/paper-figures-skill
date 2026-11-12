"""Latent imagination: one executed path, five imagined branches, one choice.

The background contours are the model's own value estimate over a
two-dimensional slice of latent space -- kept pale on purpose, because they
are the stage and not the actor.  The solid dark line is what actually
happened; from the current state the agent unrolls its world model into five
dashed futures, each closing on a bubble that carries the return the model
predicts for it.  Exactly one branch is drawn in the accent colour: the
argmax, the one the planner will commit to.  That is the entire mechanism of
a Dreamer-style agent in one picture.

Drawn on an inch canvas so the reward bubbles are round circles rather than
ellipses (DESIGN_SYSTEM section 12.8) and so every gap is an inch on paper.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyBboxPatch

from pubstyle import COLORS, savefig, set_pub_style
from figkit import arrow, halo_text, inch_canvas

FW, FH = 10.4, 6.4
PX0, PX1, PY0, PY1 = 0.55, 9.95, 0.72, 5.35      # value-landscape panel

REAL = "#22364f"           # what happened: the darkest rung
IMAG = "#5b83b5"           # what the model imagines: a mid rung
PICK = "#b23a48"           # the single accent: the branch the planner takes
FIELD = LinearSegmentedColormap.from_list("field", ["#ffffff", "#cddfee"])

BRANCH = (4.30, 3.05)      # state the agent is planning from
# reach, launch angle (deg), curl over the rollout (deg), predicted return.
# Launch angle + curl is the *exit* angle: keeping those five values well
# separated is what stops two branches landing on top of each other.
ROLLOUTS = [
    (3.10, -36.0, -4.0, 0.31),
    (3.55, -14.0, -4.0, 0.74),
    (3.55, 8.0, -2.0, 1.28),
    (3.15, 26.0, 6.0, 0.42),
    (2.40, 48.0, 8.0, -0.19),
]
# Centre x, centre y, amplitude, width.  COUPLED TO ``ROLLOUTS``: every bump
# is parked on a branch endpoint so the landscape agrees with the return
# printed in that bubble -- the +1.28 branch climbs the peak, the -0.19 branch
# runs into the trough.  Change a return and you must move the matching bump,
# or the figure quietly contradicts its own numbers.
BUMPS = [(8.00, 3.45, 1.00, 2.10), (7.70, 2.00, 0.30, 1.30),
         (5.90, 5.15, -0.55, 1.45), (6.60, 0.95, -0.30, 1.40),
         (1.60, 1.30, 0.28, 1.90)]


def value_field(nx=260, ny=150):
    x = np.linspace(PX0, PX1, nx)
    y = np.linspace(PY0, PY1, ny)
    X, Y = np.meshgrid(x, y)
    Z = 0.10 * (X - PX0) / (PX1 - PX0)
    for cx, cy, a, s in BUMPS:
        Z = Z + a * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * s ** 2)))
    return X, Y, Z


def executed_path(n=140):
    t = np.linspace(0.0, 1.0, n)
    x = 1.05 + (BRANCH[0] - 1.05) * t
    y = 1.28 + (BRANCH[1] - 1.28) * t + 0.34 * np.sin(2.6 * np.pi * t) * (1 - t)
    return x, y


def imagined_path(reach, a0, curl, n=110):
    t = np.linspace(0.0, 1.0, n)
    a = np.radians(a0 + curl * t)
    r = reach * t
    return BRANCH[0] + r * np.cos(a), BRANCH[1] + r * np.sin(a)


def main():
    set_pub_style()
    fig, ax = inch_canvas(FW, FH)

    # ---- the value landscape, clipped to a rounded panel --------------------
    frame = FancyBboxPatch((PX0, PY0), PX1 - PX0, PY1 - PY0,
                           boxstyle="round,pad=0,rounding_size=0.10",
                           facecolor="#fbfcfd", edgecolor="#c3ccd6",
                           linewidth=0.9, mutation_aspect=1.0, zorder=1)
    ax.add_patch(frame)
    X, Y, Z = value_field()
    cf = ax.contourf(X, Y, Z, levels=14, cmap=FIELD, zorder=1.5)
    cl = ax.contour(X, Y, Z, levels=14, colors="#9cb8d3", linewidths=0.5,
                    zorder=1.6)
    for artist in (cf, cl):
        artist.set_clip_path(frame)
    ax.add_patch(FancyBboxPatch((PX0, PY0), PX1 - PX0, PY1 - PY0,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="none", edgecolor="#c3ccd6",
                                linewidth=0.9, mutation_aspect=1.0, zorder=7))

    # ---- what actually happened --------------------------------------------
    ex, ey = executed_path()
    ax.plot(ex, ey, color=REAL, lw=2.2, zorder=5, solid_capstyle="round")
    ax.plot(ex[::28], ey[::28], marker="o", ls="none", ms=4.2,
            markerfacecolor="white", markeredgecolor=REAL, markeredgewidth=1.1,
            zorder=6)
    halo_text(ax, ex[46], ey[46] - 0.34, "executed trajectory", ha="center",
              va="top", fontsize=7.4, color=REAL, halo="white", halo_lw=3.0,
              zorder=6, fontweight="bold")

    # ---- five imagined futures ---------------------------------------------
    best = int(np.argmax([r[3] for r in ROLLOUTS]))
    for k, (reach, a0, curl, ret) in enumerate(ROLLOUTS):
        top = k == best
        col = PICK if top else IMAG
        bx, by = imagined_path(reach, a0, curl)
        ax.plot(bx, by, color=col, lw=1.9 if top else 1.25,
                ls=(0, (4.2, 2.2)), alpha=1.0 if top else 0.92, zorder=4,
                solid_capstyle="round")
        ax.plot(bx[18::22], by[18::22], marker="o", ls="none", ms=2.6,
                color=col, alpha=0.85, zorder=4)
        ex_, ey_ = bx[-1], by[-1]
        ax.add_patch(Circle((ex_, ey_), 0.215, facecolor="white",
                            edgecolor=col, linewidth=1.5 if top else 1.1,
                            zorder=6))
        ax.text(ex_, ey_, f"{ret:+.2f}", ha="center", va="center",
                fontsize=7.0, fontweight="bold", color=col, zorder=7)

    # the chosen branch, called out from clear space to its lower right
    bx, by = imagined_path(*ROLLOUTS[best][:3])
    arrow(ax, (bx[-1] + 1.12, by[-1] - 0.62), (bx[-1] + 0.22, by[-1] - 0.16),
          color=PICK, lw=1.1, ms=9, zorder=7, rad=-0.20)
    halo_text(ax, PX1 - 0.22, by[-1] - 0.76,
              "planner commits to\n$\\arg\\max_a\\ \\hat{R}(a)$", ha="right",
              va="top", fontsize=7.2, color=PICK, halo="white", halo_lw=3.0,
              zorder=7, linespacing=1.55)

    # ---- the branch point ---------------------------------------------------
    ax.add_patch(Circle(BRANCH, 0.34, facecolor="none", edgecolor=REAL,
                        linewidth=0.8, alpha=0.35, zorder=5))
    ax.add_patch(Circle(BRANCH, 0.115, facecolor=REAL, edgecolor="white",
                        linewidth=1.2, zorder=6))
    halo_text(ax, BRANCH[0] - 0.20, BRANCH[1] + 0.42,
              "current latent state $z_t$", ha="right", va="bottom",
              fontsize=7.6, color=REAL, halo="white", halo_lw=3.0, zorder=7,
              fontweight="bold")

    # ---- key ----------------------------------------------------------------
    kx, ky, kw, kh = 0.86, 3.92, 2.58, 1.28
    ax.add_patch(FancyBboxPatch((kx, ky), kw, kh,
                                boxstyle="round,pad=0,rounding_size=0.075",
                                facecolor="white", edgecolor="#c3ccd6",
                                linewidth=0.8, alpha=0.94,
                                mutation_aspect=1.0, zorder=8))
    rows = [(REAL, "-", "executed, one real step"),
            (IMAG, (0, (4.2, 2.2)), "imagined, H = 15 steps"),
            (PICK, (0, (4.2, 2.2)), "branch with the best return"),
            (None, None, "predicted return $\\hat{R}$")]
    for i, (col, ls, lab) in enumerate(rows):
        yy = ky + kh - 0.28 - 0.27 * i
        if col is None:
            ax.add_patch(Circle((kx + 0.385, yy), 0.115, facecolor="white",
                                edgecolor=IMAG, linewidth=1.1, zorder=9))
        else:
            ax.plot([kx + 0.17, kx + 0.60], [yy, yy], color=col, lw=1.9,
                    ls=ls, zorder=9, solid_capstyle="round")
        ax.text(kx + 0.72, yy, lab, ha="left", va="center", fontsize=6.9,
                color="#454e5a", zorder=9)

    # ---- landscape legend, latent axes and captions -------------------------
    halo_text(ax, PX1 - 0.24, PY0 + 0.30,
              "contours: learned value $\\hat{V}(z)$", ha="right", va="bottom",
              fontsize=7.0, color="#3d6294", halo="white", halo_lw=3.0,
              zorder=7)
    ax.annotate("", xy=(PX0 + 0.98, PY0 + 0.26), xytext=(PX0 + 0.26,
                PY0 + 0.26), arrowprops=dict(arrowstyle="-|>", lw=0.9,
                color="#8a97a6", shrinkA=0, shrinkB=0), zorder=7)
    ax.annotate("", xy=(PX0 + 0.26, PY0 + 0.98), xytext=(PX0 + 0.26,
                PY0 + 0.26), arrowprops=dict(arrowstyle="-|>", lw=0.9,
                color="#8a97a6", shrinkA=0, shrinkB=0), zorder=7)
    ax.text(PX0 + 1.06, PY0 + 0.26, "$z_1$", ha="left", va="center",
            fontsize=7.4, color="#8a97a6", zorder=7)
    ax.text(PX0 + 0.26, PY0 + 1.06, "$z_2$", ha="center", va="bottom",
            fontsize=7.4, color="#8a97a6", zorder=7)

    ax.text(0.55, 6.08, "Planning by imagination in latent space",
            ha="left", va="center", fontsize=11.0, fontweight="bold",
            color=COLORS["ink"])
    ax.text(0.55, 5.78, "From the current state the world model unrolls five "
            "candidate futures; only the best-scoring one is executed",
            ha="left", va="center", fontsize=8.0, color=COLORS["sub"])
    ax.text(0.55, 0.30, "Illustrative landscape and returns: the geometry is "
            "the message, the numbers are not measurements",
            ha="left", va="center", fontsize=6.6, color=COLORS["muted"])

    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    savefig(fig, "1084_wm_concept_imagination.png")


if __name__ == "__main__":
    main()
