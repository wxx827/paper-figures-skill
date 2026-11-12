"""Reachable workspace of a two-link planar arm, described three ways.

The same kinematics drives all three panels and all three share one metric
window, so the linkage sketch, the reachable cross-section and the
manipulability field can be read on top of one another.  Panel (a) overlays
joint-space samples as translucent linkages with one pose solid.  Panel (b)
fills the set of Cartesian points that satisfy the inverse kinematics inside
the joint limits and paints the near-singular rim, where the determinant of
the Jacobian collapses, in house red.  Panel (c) draws the velocity
manipulability ellipse at sample points: the ellipses flatten into needles
exactly where panel (b) turns red.  Reach, area and the singular fraction are
all counted from the sampled grid.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"        # robot / model
RED = "#b23a48"         # singular, out of limits
GREY = "#808080"
FILL = "#c3d2e6"        # reachable set

L1, L2 = 0.42, 0.34                       # link lengths, m
Q1_LIM = (np.deg2rad(-25.0), np.deg2rad(205.0))
Q2_LIM = (np.deg2rad(-172.0), np.deg2rad(-4.0))    # elbow-down branch
XLIM = (-0.88, 0.88)
YLIM = (-0.74, 0.96)
SEQ_CMAP = get_cmap("cb_sequential_blue")


def fk(q1, q2):
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return x, y


def ik(x, y):
    """Elbow-down inverse kinematics; NaN where the point is unreachable."""
    c2 = (x ** 2 + y ** 2 - L1 ** 2 - L2 ** 2) / (2.0 * L1 * L2)
    ok = np.abs(c2) <= 1.0
    c2 = np.where(ok, c2, np.nan)
    q2 = -np.arccos(np.clip(c2, -1.0, 1.0))
    q1 = (np.arctan2(y, x)
          - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2)))
    q1 = np.where(ok, q1, np.nan)
    return q1, q2


def in_limits(q1, q2):
    return ((q1 >= Q1_LIM[0]) & (q1 <= Q1_LIM[1])
            & (q2 >= Q2_LIM[0]) & (q2 <= Q2_LIM[1]))


def jacobian(q1, q2):
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def grid_fields(n=260):
    x = np.linspace(XLIM[0], XLIM[1], n)
    y = np.linspace(YLIM[0], YLIM[1], int(n * 0.8))
    xx, yy = np.meshgrid(x, y)
    q1, q2 = ik(xx, yy)
    reach = np.isfinite(q1) & in_limits(q1, q2)
    w = L1 * L2 * np.abs(np.sin(q2))          # |det J|, m^2 / rad^2
    w = np.where(reach, w, np.nan)
    return x, y, reach.astype(float), w


def arm_panel(ax, rng):
    poses = [(np.deg2rad(a), np.deg2rad(b)) for a, b in
             [(10, -40), (35, -95), (72, -130), (108, -60),
              (150, -30), (178, -100)]]
    hero = (np.deg2rad(62.0), np.deg2rad(-72.0))

    ax.add_patch(Polygon([(-0.10, -0.13), (0.10, -0.13), (0.055, 0.0),
                          (-0.055, 0.0)], closed=True, facecolor="#3a3a3a",
                         edgecolor="none", zorder=3))
    for q1, q2 in poses:
        ex, ey = L1 * np.cos(q1), L1 * np.sin(q1)
        tx, ty = fk(q1, q2)
        ax.plot([0, ex, tx], [0, ey, ty], color=BLUE, lw=2.6, alpha=0.30,
                solid_capstyle="round", zorder=4)
        ax.plot([tx], [ty], marker="o", ms=3.0, color=BLUE, alpha=0.40,
                zorder=4)

    q1, q2 = hero
    ex, ey = L1 * np.cos(q1), L1 * np.sin(q1)
    tx, ty = fk(q1, q2)
    ax.plot([0, ex], [0, ey], color=BLUE, lw=3.4, solid_capstyle="round",
            zorder=6)
    ax.plot([ex, tx], [ey, ty], color=BLUE, lw=3.4, solid_capstyle="round",
            zorder=6)
    for px, py in [(0.0, 0.0), (ex, ey)]:
        ax.plot([px], [py], marker="o", ms=6.0, mfc="white", mec=INK,
                mew=1.0, zorder=7)
    ax.plot([tx], [ty], marker="o", ms=5.0, color=RED, zorder=7)

    ax.plot([0, 0.22], [0, 0], color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.plot([ex, ex + 0.18 * np.cos(q1)], [ey, ey + 0.18 * np.sin(q1)],
            color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(0.20, 0.055, r"$q_1$ = " + f"{np.rad2deg(q1):.0f}" + r"$^\circ$",
            fontsize=7.0, color=INK, ha="left", va="bottom")
    ax.text(ex + 0.03, ey + 0.11, r"$q_2$ = " + f"{np.rad2deg(q2):.0f}"
            + r"$^\circ$", fontsize=7.0, color=INK, ha="left", va="bottom")
    ax.text(tx + 0.03, ty, "tool centre point", fontsize=6.8, color=RED,
            ha="left", va="center")
    ax.text(-0.84, -0.70,
            f"$L_1$ = {L1 * 1e3:.0f} mm,  $L_2$ = {L2 * 1e3:.0f} mm\n"
            f"{len(poses)} sampled configurations", fontsize=6.6, color=SUB,
            ha="left", va="bottom", linespacing=1.35)
    ax.set_title("Linkage and joint samples", fontsize=8.5,
                 fontweight="bold", color=INK)
    _ = rng


def reach_panel(ax, x, y, reach, w):
    ax.contourf(x, y, reach, levels=[0.5, 1.5], colors=[FILL], alpha=0.85,
                zorder=2)
    ax.contour(x, y, reach, levels=[0.5], colors=[BLUE], linewidths=1.1,
               zorder=4)

    thr = 0.50 * np.nanmax(w)
    sing = np.where(np.isfinite(w) & (w < thr), 1.0, 0.0)
    ax.contourf(x, y, sing, levels=[0.5, 1.5], colors=[RED], alpha=0.55,
                zorder=3)

    frac = float(np.nansum(sing) / max(np.sum(reach), 1.0))
    cell = (x[1] - x[0]) * (y[1] - y[0])
    area = float(np.sum(reach) * cell)

    ax.plot([0], [0], marker="s", ms=4.0, color="#3a3a3a", zorder=6)
    ax.annotate("near-singular rim,\n" + r"$\det J \rightarrow 0$",
                xy=(0.0, L1 + L2 - 0.020), xytext=(0.30, 0.90), ha="left",
                va="top", fontsize=6.8, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.annotate("inner void,\nelbow fully folded", xy=(0.02, L1 - L2 - 0.01),
                xytext=(-0.83, 0.26), ha="left", va="center", fontsize=6.8,
                color=RED, arrowprops=dict(arrowstyle="-", color=RED,
                                           lw=0.7))
    ax.text(-0.84, -0.70,
            f"reach {(L1 + L2) * 1e3:.0f} mm,  area {area:.3f} m$^2$\n"
            f"near-singular share {frac * 100:.0f} %", fontsize=6.6,
            color=SUB, ha="left", va="bottom", linespacing=1.35)
    ax.set_title("Reachable cross-section", fontsize=8.5,
                 fontweight="bold", color=INK)
    return area, frac


def ellipse_panel(fig, ax, x, y, reach, w):
    ax.contour(x, y, reach, levels=[0.5], colors=[GREY], linewidths=0.8,
               zorder=2)
    xs = np.linspace(-0.66, 0.66, 9)
    ys = np.linspace(-0.24, 0.80, 7)
    norm = mpl.colors.Normalize(0.0, float(np.nanmax(w)))
    scale = 0.072

    n_drawn = 0
    for px in xs:
        for py in ys:
            q1, q2 = ik(px, py)
            if not np.isfinite(q1) or not in_limits(q1, q2):
                continue
            j = jacobian(q1, q2)
            u, sv, _ = np.linalg.svd(j)
            ang = np.rad2deg(np.arctan2(u[1, 0], u[0, 0]))
            wi = float(sv[0] * sv[1])
            ax.add_patch(Ellipse((px, py), 2 * scale * sv[0],
                                 2 * scale * sv[1], angle=ang,
                                 facecolor=SEQ_CMAP(norm(wi)), alpha=0.85,
                                 edgecolor=BLUE, linewidth=0.5, zorder=5))
            n_drawn += 1

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=SEQ_CMAP)
    safe_colorbar(fig, sm, ax, label=r"$w=\sqrt{\det JJ^{T}}$"
                  " (m$^2$/rad$^2$)", width=0.10, pad=0.06, labelsize=6.5)
    ax.text(-0.84, -0.70,
            f"{n_drawn} sample poses\nellipse axes are singular values of $J$",
            fontsize=6.6, color=SUB, ha="left", va="bottom", linespacing=1.35)
    ax.set_title("Manipulability field", fontsize=8.5,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(21)
    x, y, reach, w = grid_fields()

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.4))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xlabel("$x$ (m)")
        ax.grid(lw=0.4, alpha=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("$y$ (m)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    arm_panel(axes[0], rng)
    area, frac = reach_panel(axes[1], x, y, reach, w)
    ellipse_panel(fig, axes[2], x, y, reach, w)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], w_pad=2.2)
    fig.canvas.draw()
    for ax in axes:
        ax.apply_aspect()
    panel_labels_inches(fig, axes, dx=-0.34, dy=0.08)
    savefig(fig, "583_rob_workspace_reach.png")


if __name__ == "__main__":
    main()
