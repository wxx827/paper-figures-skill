"""Two-variable production LP: feasible polytope, objective contours, optimum.

Everything drawn here is solved, not sketched.  The feasible vertices come
from intersecting every pair of constraint boundaries and keeping the points
that satisfy all five inequalities; the optimum comes from
``scipy.optimize.linprog`` running on the same coefficient arrays that draw
the constraint lines.  The slack panel reports ``b - A x*`` per row, and the
sensitivity panel re-solves the programme over a grid of objective
coefficients so the vertex-switching breakpoints are measured rather than
asserted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as PolyPatch
from scipy.optimize import linprog

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, inch_axes, panel_labels_inches, text_card

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"

W, H = 10.4, 6.4

# max  c . x   s.t.  A x <= b,  x >= 0.   Rows are shop-floor hour budgets.
OBJ = np.array([3.0, 2.0])
A = np.array([[2.0, 1.0],
              [2.0, 3.0],
              [3.0, 1.0]])
B = np.array([18.0, 42.0, 24.0])
CON_NAME = ["machining", "assembly", "finishing"]
CON_COL = [BLUE, GREEN, GOLD]
VERT_COL = [BLUE, GREEN, GOLD, PURPLE, "#c0603f"]

FOOT = ("Feasible vertices are enumerated by intersecting every pair of "
        "constraint boundaries and keeping the points that satisfy all five "
        "inequalities; the optimum is returned by scipy.optimize.linprog on "
        "the same coefficient arrays that draw the lines. Slack is b - A x* "
        "per row, and the sensitivity sweep re-solves the programme for 401 "
        "values of the objective coefficient on x1.")


def full_system():
    """Stack the resource rows with the two non-negativity rows."""
    a = np.vstack([A, np.array([[-1.0, 0.0], [0.0, -1.0]])])
    b = np.concatenate([B, np.array([0.0, 0.0])])
    return a, b


def feasible_vertices():
    a, b = full_system()
    n = len(b)
    keep = []
    for i in range(n):
        for j in range(i + 1, n):
            m = a[[i, j]]
            if abs(np.linalg.det(m)) < 1e-9:
                continue
            p = np.linalg.solve(m, b[[i, j]])
            if np.all(a @ p <= b + 1e-7):
                if not any(np.allclose(p, q, atol=1e-6) for q in keep):
                    keep.append(p)
    v = np.array(keep)
    ctr = v.mean(axis=0)
    ang = np.arctan2(v[:, 1] - ctr[1], v[:, 0] - ctr[0])
    return v[np.argsort(ang)]


def solve(c1, c2):
    r = linprog(np.array([-c1, -c2]), A_ub=A, b_ub=B,
                bounds=[(0.0, None), (0.0, None)], method="highs")
    return r.x


def sensitivity(verts, c1_grid):
    """Optimal value and active vertex as the x1 coefficient is swept."""
    z = np.empty(c1_grid.size)
    idx = np.empty(c1_grid.size, dtype=int)
    for k, c1 in enumerate(c1_grid):
        x = solve(c1, OBJ[1])
        z[k] = c1 * x[0] + OBJ[1] * x[1]
        idx[k] = int(np.argmin(np.linalg.norm(verts - x, axis=1)))
    return z, idx


def runs(idx, grid):
    """Contiguous [lo, hi] coefficient intervals over which a vertex wins."""
    out = []
    start = 0
    for k in range(1, idx.size + 1):
        if k == idx.size or idx[k] != idx[start]:
            out.append((grid[start], grid[k - 1], int(idx[start])))
            start = k
    return out


def infeasible_strip(ax, a, b, color, depth):
    """Hatched band on the side of a boundary that violates the constraint."""
    na = float(np.linalg.norm(a))
    u = a / na
    d = np.array([-a[1], a[0]]) / na
    p0 = b * a / na ** 2
    tail = np.array([p0 - 90.0 * d, p0 + 90.0 * d])
    poly = np.vstack([tail, (tail + u * depth)[::-1]])
    ax.add_patch(PolyPatch(poly, closed=True, facecolor="none",
                           edgecolor=color, hatch="////", linewidth=0.0,
                           alpha=0.50, zorder=1))


def boundary_line(ax, a, b, color, xlim, ylim, label):
    na = float(np.linalg.norm(a))
    d = np.array([-a[1], a[0]]) / na
    p0 = b * a / na ** 2
    t = np.linspace(-90.0, 90.0, 2)
    pts = p0[None, :] + t[:, None] * d[None, :]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5, zorder=4, label=label)


def draw_main(ax, verts, xopt, zopt):
    xlim = (-0.45, 12.6)
    ylim = (-0.70, 17.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axisbelow(True)

    ax.add_patch(PolyPatch(verts, closed=True,
                           facecolor=lighten(GREEN, 0.80), edgecolor=GREEN,
                           linewidth=1.0, zorder=2))

    for a, b, col, name in zip(A, B, CON_COL, CON_NAME):
        infeasible_strip(ax, a, b, col, 0.62)
        boundary_line(ax, a, b, col, xlim, ylim,
                      rf"{a[0]:g}$x_1$ + {a[1]:g}$x_2$ $\leq$ {b:g}  ({name})")

    levels = np.arange(12.0, 49.0, 6.0)
    for k in levels:
        xs = np.array(xlim)
        ys = (k - OBJ[0] * xs) / OBJ[1]
        ax.plot(xs, ys, color=GREY, lw=0.7, ls=(0, (5, 3)), zorder=3)
    ax.plot(np.array(xlim), (zopt - OBJ[0] * np.array(xlim)) / OBJ[1],
            color=RED, lw=1.1, ls=(0, (5, 2.4)), zorder=5)

    step = 2.0 * 6.0 * OBJ / float(OBJ @ OBJ)
    base = np.array([6.9, 9.2])
    ax.annotate("", xy=tuple(base + step), xytext=tuple(base),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4,
                                mutation_scale=13, shrinkA=0, shrinkB=0),
                zorder=6)
    ax.text(base[0] + step[0] + 0.25, base[1] + step[1] + 0.10,
            "improving\ndirection", ha="left", va="center", fontsize=6.6,
            color=RED, linespacing=1.35)
    label_off = {(0, 0): (0.34, 0.40), (8, 0): (0.34, 0.40),
                 (6, 6): (0.38, 0.34), (0, 14): (0.38, 0.34)}
    for p in verts:
        opt = np.allclose(p, xopt, atol=1e-6)
        ax.plot([p[0]], [p[1]], marker="o", ms=8.0 if opt else 4.6,
                mfc=RED if opt else "white", mec=RED if opt else RULE,
                mew=1.1, zorder=7)
        if opt:
            continue
        key = tuple(int(round(v)) for v in p)
        dx, dy = label_off.get(key, (0.34, 0.40))
        ax.text(p[0] + dx, p[1] + dy, f"({key[0]:d}, {key[1]:d})",
                ha="left", va="bottom", fontsize=6.4, color=SUB, zorder=8)

    ax.annotate(f"optimum ({xopt[0]:.0f}, {xopt[1]:.0f})\n"
                f"$z^*$ = {zopt:.0f} units",
                xy=tuple(xopt), xytext=(xopt[0] + 2.2, xopt[1] + 2.6),
                fontsize=6.8, color=RED, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                shrinkA=0, shrinkB=3))

    ax.set_xlabel("product A output $x_1$ (units per shift)", fontsize=8.0)
    ax.set_ylabel("product B output $x_2$ (units per shift)", fontsize=8.0)
    ax.set_title("Feasible polytope with the objective contour family",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(labelsize=7.2)
    style_axes(ax)

    handles = [Patch(facecolor=lighten(GREEN, 0.80), edgecolor=GREEN,
                     label="feasible region")]
    handles += [Line2D([], [], color=c, lw=1.5, label=lab)
                for c, lab in zip(CON_COL, [
                    rf"{a[0]:g}$x_1$+{a[1]:g}$x_2$ $\leq$ {b:g} ({n}, h)"
                    for a, b, n in zip(A, B, CON_NAME)])]
    handles += [Line2D([], [], color=GREY, lw=0.7, ls=(0, (5, 3)),
                       label="objective contour"),
                Line2D([], [], color=RED, lw=0.0, marker="o", ms=6.5,
                       label="optimal vertex")]
    ax.legend(handles=handles, loc="upper right", fontsize=6.5,
              frameon=True, framealpha=0.92, facecolor="white",
              edgecolor=HAIR, borderpad=0.45, handlelength=1.7,
              labelspacing=0.42)


def draw_slack(ax, xopt):
    used = A @ xopt
    slack = B - used
    y = np.arange(len(B))[::-1]
    ax.barh(y, used, height=0.52, color=[lighten(c, 0.35) for c in CON_COL],
            edgecolor="white", linewidth=0.6, zorder=3)
    ax.barh(y, slack, left=used, height=0.52, color="#e2e2e2",
            edgecolor="white", linewidth=0.6, zorder=3)
    for yy, u, s in zip(y, used, slack):
        if s < 1e-6:
            ax.text(u + 0.5, yy, "binding", ha="left", va="center",
                    fontsize=6.2, color=RED, fontweight="bold")
        else:
            ax.text(u + s + 0.5, yy, f"{s:.0f} h idle", ha="left",
                    va="center", fontsize=6.2, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n(cap {b:.0f} h)" for n, b in zip(CON_NAME, B)],
                       fontsize=6.5)
    ax.set_xlim(0.0, float(B.max()) * 1.30)
    ax.set_xlabel("resource consumed at $x^*$ (h)", fontsize=7.2)
    n_bind = int((slack < 1e-6).sum())
    ax.set_title(f"Slack per constraint: {n_bind:d} of {len(B):d} rows bind",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    style_axes(ax)


def draw_sensitivity(ax, verts, grid, z, idx):
    segs = runs(idx, grid)
    for lo, hi, vi in segs:
        ax.axvspan(lo, hi, color=lighten(VERT_COL[vi % len(VERT_COL)], 0.82),
                   zorder=0, linewidth=0.0)
        p = [int(round(v)) for v in verts[vi]]
        ax.text(0.5 * (lo + hi), 0.055, f"({p[0]:d},{p[1]:d})",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.0, color=SUB)
    ax.plot(grid, z, color=INK, lw=1.5, zorder=4)
    for lo, hi, _ in segs[1:]:
        ax.axvline(lo, color=HAIR, lw=0.7, ls=(0, (3, 2.2)), zorder=1)
    ax.axvline(OBJ[0], color=RED, lw=1.1, ls=(0, (4, 2.2)), zorder=5)
    ax.plot([OBJ[0]], [np.interp(OBJ[0], grid, z)], marker="o", ms=4.6,
            mfc=RED, mec="white", mew=0.8, zorder=6)
    ax.text(OBJ[0] + 0.16, ax.get_ylim()[1], f"nominal $c_1$ = {OBJ[0]:g}",
            ha="left", va="top", fontsize=6.2, color=RED)
    ax.set_xlim(float(grid[0]), float(grid[-1]))
    ax.set_xlabel("objective coefficient $c_1$ (units per item)", fontsize=7.2)
    ax.set_ylabel("optimal profit $z^*$ (units)", fontsize=7.2)
    ax.set_title(f"Optimal vertex switches {len(segs) - 1} times",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    style_axes(ax)


def draw_model_card(fig, xopt, zopt):
    ax = inch_axes(fig, (0.80, 0.90, 5.30, 0.86), aspect=None)
    ax.set_xlim(0.0, 5.30)
    ax.set_ylim(0.0, 0.86)
    left = (rf"maximise   $z$ = {OBJ[0]:g}$x_1$ + {OBJ[1]:g}$x_2$"
            "\n"
            rf"subject to  {A[0, 0]:g}$x_1$ + {A[0, 1]:g}$x_2$ "
            rf"$\leq$ {B[0]:g}"
            "\n"
            rf"                {A[1, 0]:g}$x_1$ + {A[1, 1]:g}$x_2$ "
            rf"$\leq$ {B[1]:g}")
    right = (rf"                {A[2, 0]:g}$x_1$ + {A[2, 1]:g}$x_2$ "
             rf"$\leq$ {B[2]:g}"
             "\n"
             r"                $x_1$, $x_2$ $\geq$ 0"
             "\n"
             rf"solution   $x^*$ = ({xopt[0]:.0f}, {xopt[1]:.0f}), "
             rf"$z^*$ = {zopt:.0f} units")
    text_card(ax, 0.0, 0.0, 2.62, 0.86, left, fs=6.8, ha="left",
              pad=0.11, linespacing=1.50,
              title="Linear programme (hours per shift)", title_fs=7.0)
    text_card(ax, 2.72, 0.0, 2.58, 0.86, right, fs=6.8, ha="left",
              pad=0.11, linespacing=1.50, fc="white",
              title="Remaining rows and the solved optimum", title_fs=7.0)


def main():
    set_pub_style()
    verts = feasible_vertices()
    xopt = solve(OBJ[0], OBJ[1])
    zopt = float(OBJ @ xopt)
    grid = np.linspace(0.0, 8.0, 401)
    z, idx = sensitivity(verts, grid)

    fig = plt.figure(figsize=(W, H))
    ax_main = axes_inches(fig, (0.80, 2.22, 5.30, 3.40))
    ax_top = axes_inches(fig, (7.05, 4.30, 3.08, 1.32))
    ax_bot = axes_inches(fig, (7.05, 2.22, 3.08, 1.32))

    draw_main(ax_main, verts, xopt, zopt)
    draw_slack(ax_top, xopt)
    draw_sensitivity(ax_bot, verts, grid, z, idx)
    draw_model_card(fig, xopt, zopt)

    panel_labels_inches(fig, [ax_main, ax_top, ax_bot], dx=-0.56, dy=0.08)

    savefig(fig, "858_orx_lp_feasible_region.png")


if __name__ == "__main__":
    main()
