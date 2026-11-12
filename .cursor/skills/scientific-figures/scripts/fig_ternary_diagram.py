"""Ternary isothermal section of the Fe–Cr–Ni system at 1000 °C.

The equilateral triangle, its ticks and its grid are built from an explicit
barycentric-to-cartesian transform.  Three phase fields are filled and
separated by boundary lines, iso-hardness contours are overlaid on a
Delaunay triangulation of the composition simplex, and commercial stainless
grades are plotted at their nominal compositions.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from pubstyle import set_pub_style, PALETTES, savefig

SQ3 = np.sqrt(3.0) / 2.0

# phase boundaries as (Cr, Ni) polylines
B_AG = [(0.15, 0.000), (0.30, 0.020), (0.45, 0.075), (0.60, 0.135),
        (0.70, 0.185), (0.78, 0.220)]
B_G = [(0.03, 0.000), (0.10, 0.025), (0.18, 0.060), (0.28, 0.115),
       (0.38, 0.185), (0.48, 0.275), (0.54, 0.350), (0.58, 0.420)]

GRADES = [("430", 0.16, 0.004, 7, 3), ("2205", 0.22, 0.055, 9, -3),
          ("304", 0.18, 0.086, -8, 3), ("316L", 0.17, 0.122, 8, 4),
          ("310S", 0.25, 0.200, 8, 1), ("alloy 600", 0.16, 0.760, 9, 0)]


def tri2xy(b, c):
    """(Cr, Ni) fractions -> cartesian; Fe = 1 - Cr - Ni."""
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    return b + 0.5 * c, SQ3 * c


def hardness(b, c):
    return 130.0 + 190.0 * b + 70.0 * c + 300.0 * b * c


def edge(p0, p1, n=40):
    """Points along a triangle edge in (Cr, Ni) space."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return list(map(tuple, np.asarray(p0) * (1 - t) + np.asarray(p1) * t))


def main():
    set_pub_style(base_font=9)

    fig, ax = plt.subplots(figsize=(7.2, 5.9))
    ax.set_aspect("equal")
    ax.axis("off")

    # --- phase fields ------------------------------------------------------
    alpha = B_AG + edge((0.78, 0.22), (1.0, 0.0)) + [(0.15, 0.0)]
    two = B_AG + edge((0.78, 0.22), (0.58, 0.42)) + B_G[::-1]
    gamma = B_G + edge((0.58, 0.42), (0.0, 1.0)) + [(0.0, 0.0)]
    fields = [(alpha, "#c3d2e3", "$\\alpha$ (bcc)", 0.72, 0.055),
              (two, "#ebdcbe", "$\\alpha$ + $\\gamma$", 0.435, 0.135),
              (gamma, "#d3e2cd", "$\\gamma$ (fcc)", 0.175, 0.470)]
    for poly, col, lab, lb, lc in fields:
        pb, pc = zip(*poly)
        px, py = tri2xy(np.array(pb), np.array(pc))
        ax.add_patch(Polygon(np.column_stack([px, py]), closed=True,
                             facecolor=col, edgecolor="none", zorder=2))
        lx, ly = tri2xy(lb, lc)
        ax.text(lx, ly, lab, fontsize=9, fontweight="bold", color="#33404d",
                ha="center", va="center", zorder=9, style="italic",
                path_effects=[pe.withStroke(linewidth=2.4,
                                            foreground="white")])

    # --- internal grid -----------------------------------------------------
    for v in np.arange(0.2, 0.81, 0.2):
        gx, gy = tri2xy(np.array([v, v]), np.array([0.0, 1.0 - v]))
        ax.plot(gx, gy, color="#ffffff", lw=0.6, alpha=0.85, zorder=3)
        gx, gy = tri2xy(np.array([0.0, 1.0 - v]), np.array([v, v]))
        ax.plot(gx, gy, color="#ffffff", lw=0.6, alpha=0.85, zorder=3)
        gx, gy = tri2xy(np.array([1.0 - v, 0.0]), np.array([0.0, 1.0 - v]))
        ax.plot(gx, gy, color="#ffffff", lw=0.6, alpha=0.85, zorder=3)

    # --- iso-hardness contours --------------------------------------------
    n = 130
    bs, cs_ = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            bs.append(i / n)
            cs_.append(j / n)
    bs, cs_ = np.array(bs), np.array(cs_)
    gx, gy = tri2xy(bs, cs_)
    triang = mtri.Triangulation(gx, gy)
    cmap = LinearSegmentedColormap.from_list(
        "hv", ["#e0a558", "#d1782e", "#ad5219", "#833a10", "#552208"])
    levels = np.arange(175, 301, 25)
    cs = ax.tricontour(triang, hardness(bs, cs_), levels=levels, cmap=cmap,
                       linewidths=0.9, zorder=4)
    # label each contour at a hand-picked interior point (Ni = 40 or 58 at.%)
    manual = []
    for i, lv in enumerate(levels):
        c_ni = 0.40 if i % 2 == 0 else 0.58
        b_cr = (lv - 130.0 - 70.0 * c_ni) / (190.0 + 300.0 * c_ni)
        manual.append(tri2xy(b_cr, c_ni))
    ax.clabel(cs, inline=True, fontsize=6.0, fmt="%d", inline_spacing=3,
              manual=manual)

    # --- phase boundaries --------------------------------------------------
    for poly in (B_AG, B_G):
        pb, pc = zip(*poly)
        px, py = tri2xy(np.array(pb), np.array(pc))
        ax.plot(px, py, color="#2b2b2b", lw=1.3, zorder=6,
                solid_capstyle="round")

    # --- triangle frame, ticks, axis titles --------------------------------
    corners = np.array([tri2xy(0.0, 0.0), tri2xy(1.0, 0.0), tri2xy(0.0, 1.0)])
    ax.add_patch(Polygon(corners, closed=True, facecolor="none",
                         edgecolor="#333333", lw=1.0, zorder=7))
    for v in np.arange(0.0, 1.01, 0.1):
        big = abs(v * 10 - round(v * 10)) < 1e-6 and round(v * 10) % 2 == 0
        ln = 0.020 if big else 0.012
        # bottom edge -> Cr
        x, y = tri2xy(v, 0.0)
        ax.plot([x, x - 0.5 * ln], [y, y - SQ3 * ln], color="#333333",
                lw=0.7, zorder=7)
        # right edge -> Ni
        x, y = tri2xy(1.0 - v, v)
        ax.plot([x, x + ln], [y, y], color="#333333", lw=0.7, zorder=7)
        # left edge -> Fe
        x, y = tri2xy(0.0, 1.0 - v)
        ax.plot([x, x - 0.5 * ln], [y, y + SQ3 * ln], color="#333333",
                lw=0.7, zorder=7)
        if not big:
            continue
        pct = f"{v * 100:.0f}"
        x, y = tri2xy(v, 0.0)
        ax.text(x - 0.028, y - 0.048, pct, fontsize=6.6, ha="center",
                va="center", color="#333333")
        x, y = tri2xy(1.0 - v, v)
        ax.text(x + 0.040, y, pct, fontsize=6.6, ha="left", va="center",
                color="#333333")
        x, y = tri2xy(0.0, 1.0 - v)
        ax.text(x - 0.042, y + 0.024, pct, fontsize=6.6, ha="right",
                va="center", color="#333333")

    ax.text(0.5, -0.115, "Cr (at.%)", fontsize=9, ha="center", va="center",
            fontweight="bold", color="#1a1a1a")
    ax.text(0.815, 0.50, "Ni (at.%)", fontsize=9, ha="center", va="center",
            rotation=-60, fontweight="bold", color="#1a1a1a")
    ax.text(0.185, 0.50, "Fe (at.%)", fontsize=9, ha="center", va="center",
            rotation=60, fontweight="bold", color="#1a1a1a")
    for (bx, cy), lab, dx, dy in ((tri2xy(0.0, 0.0), "Fe", -0.062, -0.052),
                                  (tri2xy(1.0, 0.0), "Cr", 0.062, -0.052),
                                  (tri2xy(0.0, 1.0), "Ni", 0.0, 0.058)):
        ax.text(bx + dx, cy + dy, lab, fontsize=10, fontweight="bold",
                ha="center", va="center", color="#1a1a1a")

    # --- measured alloys ---------------------------------------------------
    for lab, b, c, dx, dy in GRADES:
        x, y = tri2xy(b, c)
        ax.plot(x, y, marker="D", ms=4.0, mfc="white", mec="#8a2f2f",
                mew=1.0, ls="none", zorder=8)
        ax.annotate(lab, xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.5,
                    color="#8a2f2f", fontweight="bold", zorder=9,
                    ha="left" if dx > 0 else "right",
                    va="bottom" if dy > 0 else "top",
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground="white")])

    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.18, 1.04)

    cax = fig.add_axes([0.855, 0.21, 0.020, 0.54])
    cb = fig.colorbar(cs, cax=cax)
    cb.set_label("Vickers hardness (HV)", fontsize=7.8)
    cb.ax.tick_params(labelsize=6.6, width=0.6, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#555555")

    ax.set_title("Fe–Cr–Ni isothermal section at 1000 °C with iso-hardness "
                 "contours", fontsize=10.5, pad=6)
    fig.subplots_adjust(left=0.005, right=0.83, top=0.94, bottom=0.045)
    savefig(fig, "125_ternary_diagram.png")


if __name__ == "__main__":
    main()
