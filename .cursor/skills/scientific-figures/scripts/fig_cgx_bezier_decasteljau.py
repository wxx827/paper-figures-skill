"""A cubic Bezier built by de Casteljau, with the basis and the convergence.

Main panel: one cubic curve and the recursion that produces it.  At the marked
parameter the three levels of linear interpolation are drawn as they are
computed - four control points to three, three to two, two to the point on the
curve - and the second-level segment is repeated at twenty other parameters so
the family of tangent lines that envelopes the curve is visible.  The curve
itself is the last point of the same recursion evaluated densely, never a
spline drawn by other means.

Auxiliary (both at 58 % of the main panel width): the four cubic Bernstein
weights against the parameter, read at the same marked value, which are the
coefficients the recursion is implicitly forming; and the deviation of the
control polygon from the curve after repeated subdivision at the midpoint,
which falls by roughly a factor of four per level.

Illustrative geometry, no physical units.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
BLUE = "#3b5b92"
DEEP = "#134a76"
RUST = "#c0603f"
ACCENT = "#b23a48"
TEAL = "#2e7d5b"
GOLD = "#c9a227"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 1.02, 5.54, 4.44)
AUX_T_RECT = (7.62, 3.56, 3.16, 1.90)
AUX_B_RECT = (7.62, 1.02, 3.16, 1.90)
ASPECT = MAIN_RECT[2] / MAIN_RECT[3]

CTRL = np.array([[0.45, 0.85],
                 [1.35, 4.35],
                 [5.05, 0.35],
                 [6.25, 3.25]])
T_MARK = 0.62
LEVEL_COL = [DEEP, TEAL, RUST]


def decasteljau(ctrl, t):
    """All intermediate levels of the recursion at parameter ``t``."""
    levels = [np.asarray(ctrl, dtype=float)]
    cur = levels[0]
    while len(cur) > 1:
        cur = (1.0 - t) * cur[:-1] + t * cur[1:]
        levels.append(cur)
    return levels


def curve(ctrl, ts):
    return np.array([decasteljau(ctrl, t)[-1][0] for t in ts])


def subdivide(ctrl, t=0.5):
    """Split one Bezier into two that share the same trace."""
    lv = decasteljau(ctrl, t)
    left = np.array([l[0] for l in lv])
    right = np.array([l[-1] for l in reversed(lv)])
    return left, right


def polygon_deviation(pieces, dense):
    """Largest distance from any control vertex to a densely sampled curve."""
    verts = np.vstack(pieces)
    worst = 0.0
    for v in verts:
        worst = max(worst, float(np.linalg.norm(dense - v, axis=1).min()))
    return worst


def polyline_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def main():
    set_pub_style(base_font=8)

    ts = np.linspace(0.0, 1.0, 601)
    curv = curve(CTRL, ts)
    lv = decasteljau(CTRL, T_MARK)
    p_on = lv[-1][0]
    tangent = lv[-2][1] - lv[-2][0]
    arc = polyline_length(curv)
    chord = float(np.linalg.norm(CTRL[-1] - CTRL[0]))
    poly_len = polyline_length(CTRL)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the recursion, drawn ----------------------------------------
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5, alpha=0.5, zorder=0)

    for t in np.linspace(0.03, 0.97, 21):
        seg = decasteljau(CTRL, t)[2]
        ax.plot(seg[:, 0], seg[:, 1], color=lighten(GOLD, 0.42), lw=0.7,
                zorder=1)

    ax.plot(CTRL[:, 0], CTRL[:, 1], color=FAINT, lw=1.0, ls=(0, (4.0, 2.4)),
            marker="s", ms=5.4, mfc="white", mec=FAINT, mew=1.1, zorder=4)
    for k, p in enumerate(CTRL):
        ax.text(p[0], p[1] + 0.16, f"P{k}", ha="center", va="bottom",
                fontsize=7.2, color=SUB, fontweight="bold")

    ax.plot(curv[:, 0], curv[:, 1], color=BLUE, lw=2.2, zorder=5,
            solid_capstyle="round")

    for lev in (1, 2):
        seg = lv[lev]
        ax.plot(seg[:, 0], seg[:, 1], color=LEVEL_COL[lev], lw=1.3,
                zorder=6 + lev)
        ax.plot(seg[:, 0], seg[:, 1], ls="none", marker="o", ms=5.0,
                mfc=LEVEL_COL[lev], mec="white", mew=0.9, zorder=8 + lev)
    ax.plot(lv[0][:, 0], lv[0][:, 1], ls="none", marker="s", ms=5.4,
            mfc="white", mec=FAINT, mew=1.1, zorder=6)
    ax.plot([p_on[0]], [p_on[1]], marker="o", ms=8.0, mfc=ACCENT, mec="white",
            mew=1.2, zorder=12)

    tn = tangent / np.linalg.norm(tangent)
    ax.annotate("", xy=tuple(p_on + 1.05 * tn), xytext=tuple(p_on),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2,
                                shrinkA=0, shrinkB=0), zorder=11)
    ax.text(*(p_on + 1.10 * tn + np.array([0.05, 0.46])),
            "tangent = level-2 segment,\n3 x its length is the derivative",
            ha="left", va="bottom", fontsize=6.3, color=ACCENT,
            linespacing=1.4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                      pad=1.2))
    ax.text(p_on[0] + 0.12, p_on[1] - 0.14,
            f"B(t = {T_MARK:.2f}) = ({p_on[0]:.2f}, {p_on[1]:.2f})",
            ha="left", va="top", fontsize=6.6, color=ACCENT,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                      pad=1.2), zorder=12)

    allp = np.vstack([curv, CTRL])
    xc = 0.5 * (allp[:, 0].min() + allp[:, 0].max())
    yc = 0.5 * (allp[:, 1].min() + allp[:, 1].max())
    half_y = 0.5 * max(np.ptp(allp[:, 1]), np.ptp(allp[:, 0]) / ASPECT) * 1.44
    yc = yc + 0.12 * half_y          # clear top-right corner for the legend
    ax.set_xlim(xc - half_y * ASPECT, xc + half_y * ASPECT)
    ax.set_ylim(yc - half_y, yc + half_y)
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel("x (dimensionless length units)", fontsize=8.5)
    ax.set_ylabel("y (dimensionless length units)", fontsize=8.5)
    ax.set_title("de Casteljau: three rounds of linear interpolation put one "
                 "point on the curve", fontsize=9.0, fontweight="bold",
                 pad=5.0)

    handles = [
        Line2D([], [], color=FAINT, lw=1.0, ls=(0, (4.0, 2.4)), marker="s",
               ms=4.4, mfc="white", mec=FAINT,
               label=f"control polygon, 4 points, {poly_len:.2f} units long"),
        Line2D([], [], color=BLUE, lw=2.2,
               label=f"cubic Bezier, arc length {arc:.2f} units "
                     f"(chord {chord:.2f})"),
        Line2D([], [], color=LEVEL_COL[1], lw=1.3, marker="o", ms=4.2,
               mfc=LEVEL_COL[1], mec="white",
               label="level 1: 3 interpolated points"),
        Line2D([], [], color=LEVEL_COL[2], lw=1.3, marker="o", ms=4.2,
               mfc=LEVEL_COL[2], mec="white",
               label="level 2: 2 points, their segment is tangent"),
        Line2D([], [], color=lighten(GOLD, 0.42), lw=0.9,
               label="the same level-2 segment at 21 parameters"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True,
                    fontsize=6.5, handlelength=1.7, labelspacing=0.30,
                    borderpad=0.30, framealpha=0.92, edgecolor="#d5d5d5",
                    facecolor="white", bbox_to_anchor=(0.996, 0.996))
    leg.get_frame().set_linewidth(0.6)

    # ---- aux (b): the Bernstein weights ------------------------------------
    tb = np.linspace(0.0, 1.0, 401)
    basis = np.array([(1 - tb) ** 3, 3 * tb * (1 - tb) ** 2,
                      3 * tb ** 2 * (1 - tb), tb ** 3])
    wmark = np.array([(1 - T_MARK) ** 3, 3 * T_MARK * (1 - T_MARK) ** 2,
                      3 * T_MARK ** 2 * (1 - T_MARK), T_MARK ** 3])
    cols = [DEEP, TEAL, GOLD, RUST]
    ax_t.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    for k in range(4):
        ax_t.plot(tb, basis[k], color=cols[k], lw=1.4, zorder=4,
                  label=f"B{k},3   weight {wmark[k]:.3f}")
        ax_t.plot([T_MARK], [wmark[k]], marker="o", ms=4.0, mfc=cols[k],
                  mec="white", mew=0.8, ls="none", zorder=6)
    ax_t.axvline(T_MARK, color=ACCENT, lw=1.0, ls=(0, (3.0, 2.2)), zorder=3)
    ax_t.text(T_MARK - 0.02, 1.03, f"t = {T_MARK:.2f}, weights sum to "
              f"{wmark.sum():.3f}", ha="right", va="bottom", fontsize=6.1,
              color=ACCENT)
    ax_t.set_xlim(0.0, 1.0)
    ax_t.set_ylim(0.0, 1.42)
    ax_t.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_t.set_yticks([0.0, 0.5, 1.0])
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("parameter t (dimensionless, 0 to 1)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("basis weight (dimensionless)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("The same point as a weighted sum of P0..P3", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.legend(loc="upper center", frameon=False, fontsize=6.0, ncol=2,
                handlelength=1.2, labelspacing=0.22, columnspacing=1.0,
                borderpad=0.10, bbox_to_anchor=(0.50, 0.995))

    # ---- aux (c): subdivision convergence ----------------------------------
    dense = curve(CTRL, np.linspace(0.0, 1.0, 12001))
    pieces = [CTRL]
    levels, devs, counts = [], [], []
    for k in range(6):
        devs.append(polygon_deviation(pieces, dense))
        levels.append(k)
        counts.append(len(pieces))
        nxt = []
        for pc in pieces:
            a, b = subdivide(pc)
            nxt.extend([a, b])
        pieces = nxt
    devs = np.array(devs)
    ratio = devs[:-1] / devs[1:]

    ax_b.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_b.plot(levels, devs, color=BLUE, lw=1.4, marker="o", ms=3.6,
              mfc="white", mew=1.0, zorder=5,
              label="measured control-polygon deviation")
    ax_b.plot(levels, devs[0] * 4.0 ** (-np.array(levels, dtype=float)),
              color=ACCENT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=4,
              label="reference: quartered each level")
    ax_b.set_yscale("log")
    for k in (0, 2, 5):
        ax_b.text(levels[k], devs[k] * 1.70, f"{counts[k]} piece"
                  f"{'s' if counts[k] > 1 else ''}", ha="center", va="bottom",
                  fontsize=6.0, color=SUB)
    ax_b.set_xlim(-0.35, 5.5)
    ax_b.set_ylim(devs.min() * 0.30, devs.max() * 6.0)
    ax_b.set_xticks(levels)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("midpoint subdivisions (levels)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel("max deviation (length units)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title(f"Polygon converges to the curve, mean ratio "
                   f"{ratio.mean():.1f} x per level", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_b.legend(loc="lower left", frameon=False, fontsize=6.0,
                handlelength=1.3, labelspacing=0.24, borderpad=0.10,
                bbox_to_anchor=(-0.012, -0.015))

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "974_cgx_bezier_decasteljau.png")


if __name__ == "__main__":
    main()
