"""A signed distance field and the contour marching squares pulls out of it.

Main panel: the field of a shape built from three discs and a rounded box,
evaluated analytically on a fine grid and shown as a diverging map with the
interior negative.  Iso-distance curves at fixed offsets are drawn on top, and
two extractions of the zero level set are compared: a hand-written marching
squares run on a coarse 20 x 16 lattice, whose corners are marked, against the
same routine on a lattice eight times finer.

Auxiliary (both at 58 % of the main panel width): how far the extracted
vertices sit from the true surface as the lattice is refined, measured by
reading the analytic field at each extracted vertex, next to the segment count
the same runs produce; and the field along one horizontal scan line with the
numerical gradient magnitude beside it, which stays near unity because a
distance field must satisfy the eikonal condition.

Illustrative geometry, no physical units.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, get_cmap
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

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 1.02, 5.54, 4.44)
AUX_T_RECT = (7.62, 3.56, 3.16, 1.90)
AUX_B_RECT = (7.62, 1.02, 3.16, 1.90)
ASPECT = MAIN_RECT[2] / MAIN_RECT[3]

HALF_Y = 2.40
HALF_X = HALF_Y * ASPECT
COARSE = (20, 16)
SCAN_Y = -0.28

# marching-squares edge pairs, corners ordered bl, br, tr, tl
EDGES = {0: [], 1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
         5: [(3, 2), (0, 1)], 6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)],
         9: [(2, 0)], 10: [(3, 0), (1, 2)], 11: [(2, 1)], 12: [(1, 3)],
         13: [(1, 0)], 14: [(0, 3)], 15: []}


def sd_disc(x, y, cx, cy, r):
    return np.hypot(x - cx, y - cy) - r


def sd_round_box(x, y, cx, cy, hx, hy, rad):
    dx = np.abs(x - cx) - (hx - rad)
    dy = np.abs(y - cy) - (hy - rad)
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside - rad


def field(x, y):
    """Signed distance: discs unioned, one disc subtracted, box unioned."""
    a = sd_disc(x, y, -0.92, -0.18, 1.16)
    b = sd_disc(x, y, 0.72, 0.52, 0.94)
    c = sd_disc(x, y, 0.10, -0.62, 0.56)
    d = sd_round_box(x, y, 1.72, -1.05, 0.62, 0.38, 0.18)
    return np.minimum(np.maximum(np.minimum(a, b), -c), d)


def marching_squares(xs, ys, f, level=0.0):
    """Zero-level segments of ``f`` on the lattice ``xs`` x ``ys``."""
    segs = []
    g = f - level
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            v = (float(g[j, i]), float(g[j, i + 1]),
                 float(g[j + 1, i + 1]), float(g[j + 1, i]))
            case = 0
            for k in range(4):
                if v[k] > 0.0:
                    case |= 1 << k
            if case in (0, 15):
                continue
            corner = ((xs[i], ys[j]), (xs[i + 1], ys[j]),
                      (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1]))

            def cross(e):
                p, q = corner[e], corner[(e + 1) % 4]
                fp, fq = v[e], v[(e + 1) % 4]
                t = fp / (fp - fq)
                return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

            for e0, e1 in EDGES[int(case)]:
                segs.append((cross(e0), cross(e1)))
    return segs


def extraction_error(n_y):
    """Mean and worst |field| at the vertices a run of the extractor emits."""
    ny = int(n_y)
    nx = int(round(ny * ASPECT))
    xs = np.linspace(-HALF_X, HALF_X, nx + 1)
    ys = np.linspace(-HALF_Y, HALF_Y, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    segs = marching_squares(xs, ys, field(xx, yy))
    if not segs:
        return np.nan, np.nan, 0, 0.0
    v = np.array(segs).reshape(-1, 2)
    err = np.abs(field(v[:, 0], v[:, 1]))
    h = float(ys[1] - ys[0])
    return float(err.mean()), float(err.max()), len(segs), h


def main():
    set_pub_style(base_font=8)

    nx_f, ny_f = 720, 578
    xs_f = np.linspace(-HALF_X, HALF_X, nx_f)
    ys_f = np.linspace(-HALF_Y, HALF_Y, ny_f)
    xx_f, yy_f = np.meshgrid(xs_f, ys_f)
    phi = field(xx_f, yy_f)

    xs_c = np.linspace(-HALF_X, HALF_X, COARSE[0] + 1)
    ys_c = np.linspace(-HALF_Y, HALF_Y, COARSE[1] + 1)
    xx_c, yy_c = np.meshgrid(xs_c, ys_c)
    phi_c = field(xx_c, yy_c)
    segs_c = marching_squares(xs_c, ys_c, phi_c)

    fine = (COARSE[0] * 8, COARSE[1] * 8)
    xs_m = np.linspace(-HALF_X, HALF_X, fine[0] + 1)
    ys_m = np.linspace(-HALF_Y, HALF_Y, fine[1] + 1)
    xx_m, yy_m = np.meshgrid(xs_m, ys_m)
    segs_f = marching_squares(xs_m, ys_m, field(xx_m, yy_m))

    v_c = np.array(segs_c).reshape(-1, 2)
    err_c = np.abs(field(v_c[:, 0], v_c[:, 1]))
    h_c = float(ys_c[1] - ys_c[0])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the field, its offsets and two extractions ------------------
    ax.set_aspect("equal")
    vmax = 1.60
    cmap = get_cmap("cb_diverging")
    im = ax.imshow(phi, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
                   extent=(-HALF_X, HALF_X, -HALF_Y, HALF_Y), aspect="equal",
                   interpolation="bilinear", zorder=1)
    levels = [-0.9, -0.6, -0.3, 0.3, 0.6, 0.9, 1.2, 1.5]
    cs = ax.contour(xs_f, ys_f, phi, levels=levels, colors="#ffffff",
                    linewidths=0.6, alpha=0.75, zorder=2)
    ax.clabel(cs, levels[::2], fmt="%.1f", fontsize=6.0, inline=True,
              colors="#ffffff")

    ax.plot(xx_c.ravel(), yy_c.ravel(), ls="none", marker="+", ms=2.6,
            color="#4a4a4a", alpha=0.55, mew=0.5, zorder=3)
    ax.add_collection(LineCollection(segs_f, colors=INK, linewidths=1.5,
                                     zorder=5))
    ax.add_collection(LineCollection(segs_c, colors=ACCENT, linewidths=1.6,
                                     zorder=6))
    ax.plot(v_c[:, 0], v_c[:, 1], ls="none", marker="o", ms=2.6, mfc=ACCENT,
            mec="none", zorder=7)

    cax = ax.inset_axes([1.012, 0.0, 0.022, 1.0])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label("signed distance to the surface (length units,\nnegative "
                 "inside)", fontsize=7.0, color=INK, linespacing=1.4)
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.5)
    cb.outline.set_linewidth(0.6)

    ax.set_xlim(-HALF_X, HALF_X)
    ax.set_ylim(-HALF_Y, HALF_Y)
    ax.axhline(SCAN_Y, color="#ffffff", lw=1.0, ls=(0, (5.0, 3.0)), zorder=4)
    ax.text(-HALF_X + 0.10, SCAN_Y + 0.07, "scan line of panel (c)",
            ha="left", va="bottom", fontsize=6.3, color="#ffffff", zorder=8)
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel("x (dimensionless length units)", fontsize=8.5)
    ax.set_ylabel("y (dimensionless length units)", fontsize=8.5)
    ax.set_title("Signed distance field with iso-distance offsets and two "
                 "marching-squares extractions", fontsize=9.0,
                 fontweight="bold", pad=5.0)

    handles = [
        Line2D([], [], color=INK, lw=1.5,
               label=f"extracted at {fine[0]} x {fine[1]} cells: "
                     f"{len(segs_f)} segments"),
        Line2D([], [], color=ACCENT, lw=1.6, marker="o", ms=3.0, mfc=ACCENT,
               mec="none",
               label=f"extracted at {COARSE[0]} x {COARSE[1]} cells: "
                     f"{len(segs_c)} segments"),
        Line2D([], [], color="#4a4a4a", lw=0.0, marker="+", ms=4.0, mew=0.7,
               ls="none", label=f"coarse lattice corners, spacing "
                                f"{h_c:.2f} units"),
        Line2D([], [], color="#ffffff", lw=0.6,
               label="iso-distance offsets, 0.3 units apart"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=True,
                    fontsize=6.5, handlelength=1.7, labelspacing=0.30,
                    borderpad=0.30, framealpha=0.90, edgecolor="#d5d5d5",
                    facecolor="white", bbox_to_anchor=(0.004, 0.004))
    leg.get_frame().set_linewidth(0.6)

    # ---- aux (b): refinement study -----------------------------------------
    nys = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128])
    stats = [extraction_error(n) for n in nys]
    mean_e = np.array([s[0] for s in stats])
    max_e = np.array([s[1] for s in stats])
    n_seg = np.array([s[2] for s in stats])
    hs = np.array([s[3] for s in stats])

    ax_t.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_t.plot(hs, max_e, color=RUST, lw=1.3, marker="s", ms=3.2, mfc="white",
              mew=0.9, zorder=5, label="worst vertex")
    ax_t.plot(hs, mean_e, color=BLUE, lw=1.3, marker="o", ms=3.2, mfc="white",
              mew=0.9, zorder=5, label="mean vertex")
    ref = mean_e[-1] * (hs / hs[-1]) ** 2
    ax_t.plot(hs, ref, color=ACCENT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=4,
              label="reference slope, cell size squared")
    ax_t.plot([h_c], [err_c.mean()], marker="*", ms=9.0, mfc=ACCENT,
              mec="white", mew=0.7, ls="none", zorder=6,
              label=f"panel (a) coarse run, mean {err_c.mean():.3f} units")
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlim(hs.min() * 0.72, hs.max() * 1.40)
    ax_t.set_ylim(mean_e.min() * 0.28, max_e.max() * 12.0)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("cell size (length units)", fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("distance to surface (units)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title(f"Extraction error falls with the lattice "
                   f"({n_seg[0]} to {n_seg[-1]} segments)", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.legend(loc="upper left", frameon=False, fontsize=6.0,
                handlelength=1.3, labelspacing=0.24, borderpad=0.10,
                bbox_to_anchor=(-0.012, 1.045))

    # ---- aux (c): the field along a scan line, and its gradient ------------
    xs_s = np.linspace(-HALF_X, HALF_X, 1200)
    phi_s = field(xs_s, np.full_like(xs_s, SCAN_Y))
    gx = np.gradient(phi_s, xs_s)
    dy = 2e-4
    gy = (field(xs_s, np.full_like(xs_s, SCAN_Y + dy))
          - field(xs_s, np.full_like(xs_s, SCAN_Y - dy))) / (2 * dy)
    gmag = np.hypot(gx, gy)
    crossings = xs_s[:-1][np.signbit(phi_s[:-1]) != np.signbit(phi_s[1:])]

    ax_b.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_b.axhline(0.0, color=FAINT, lw=0.8, zorder=2)
    ax_b.fill_between(xs_s, 0.0, phi_s, where=phi_s < 0,
                      facecolor="#cfe1ee", edgecolor="none", zorder=1)
    ax_b.plot(xs_s, phi_s, color=DEEP, lw=1.4, zorder=5)
    for x0 in crossings:
        ax_b.plot([x0], [0.0], marker="o", ms=3.6, mfc=ACCENT, mec="white",
                  mew=0.7, zorder=6)
    ax_b.text(-HALF_X + 0.08, -1.06, "surface crossings, where the extractor "
              "places a vertex", ha="left", va="bottom", fontsize=6.0,
              color=ACCENT)
    ax_b.set_xlim(-HALF_X, HALF_X)
    ax_b.set_ylim(-1.15, 2.60)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("x along the scan line (length units)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel("signed distance (units)", fontsize=7.6, labelpad=2.0)

    ax_g = ax_b.twinx()
    ax_g.plot(xs_s, gmag, color=RUST, lw=1.0, alpha=0.9, zorder=4)
    ax_g.axhline(1.0, color=RUST, lw=0.7, ls=(0, (3.0, 2.2)), zorder=3)
    ax_g.text(HALF_X - 0.08, 1.05, "eikonal target |grad| = 1", ha="right",
              va="bottom", fontsize=6.0, color=RUST)
    ax_g.set_ylim(0.0, 1.36)
    ax_g.set_yticks([0.0, 0.5, 1.0])
    ax_g.tick_params(labelsize=6.6, colors=RUST)
    ax_g.set_ylabel("|grad| (dimensionless)", fontsize=7.6, color=RUST,
                    labelpad=2.0)
    ax_g.spines["top"].set_visible(False)
    ax_g.spines["right"].set_color(RUST)

    med = float(np.median(gmag))
    ax_b.set_title(f"{len(crossings)} zero crossings, median |grad| = "
                   f"{med:.3f}", fontsize=8.4, fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "975_cgx_sdf_marching_squares.png")


if __name__ == "__main__":
    main()
