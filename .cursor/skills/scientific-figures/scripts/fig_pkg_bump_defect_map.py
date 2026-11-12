"""Flip-chip bump inspection map for one die, read as a defect-class map.

Main panel: the full solder-bump array of a single die on an equal-aspect
canvas, every bump coloured by the class the inspection step assigned it.  Two
spatial signatures are deliberately separable by eye - a non-wet ring that
tightens toward the corners, where the distance from the die neutral point is
largest, and a void cluster sitting on the die centre where package warpage
lifts the substrate away during reflow.  Dashed guides mark the two zones so
no leader line has to cross the array.  The two narrow right-hand panels turn
the same class vector into the views a yield engineer acts on: a Pareto of the
defect classes with its cumulative share, and the defect rate against distance
from the neutral point.

Every number quoted is read back off the plotted arrays - the yield from the
class counts, the Pareto cumulative from the sorted counts, the radial rates
from the same binning that draws the bars.  Values are illustrative only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from pubstyle import set_pub_style, savefig, style_axes, lighten, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

GREEN = "#2e7d5b"
RED = "#b23a48"
GOLD = "#c9a227"
BLUE = "#3b5b92"
PURPLE = "#8e6bbf"
PASS = lighten(GREEN, 0.58)

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.62, 0.74, 5.30, 4.68)
AUX_T_RECT = (7.10, 3.47, 2.90, 1.95)
AUX_B_RECT = (7.10, 0.74, 2.90, 1.95)

NX, NY = 28, 24
PITCH = 0.16                      # mm, bump pitch
R_BUMP = 0.060                    # mm, printed bump radius
RING_IN = 0.62                    # mm, inner edge of the non-wet ring guide
R_CLUSTER = 0.78                  # mm, void-cluster guide radius
CLASSES = [
    ("Pass", PASS),
    ("Non-wet", RED),
    ("Void (HiP)", GOLD),
    ("Bridge", BLUE),
    ("Missing", PURPLE),
]


def build_array(seed=9061):
    """Bump centres in mm, distance from neutral point, and one class each."""
    rng = np.random.default_rng(seed)
    x = (np.arange(NX) + 0.5) * PITCH
    y = (np.arange(NY) + 0.5) * PITCH
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()
    cx, cy = NX * PITCH / 2.0, NY * PITCH / 2.0
    dnp = np.hypot(xx - cx, yy - cy)

    d_n = dnp / dnp.max()
    p_nonwet = 0.006 + 0.62 * np.clip((d_n - 0.52) / 0.48, 0.0, 1.0) ** 2.0
    p_void = 0.62 * np.exp(-(dnp / 0.58) ** 2)
    p_bridge = 0.006 + 0.34 * np.exp(
        -(((xx - 3.28) / 0.26) ** 2 + ((yy - 2.90) / 0.26) ** 2))
    p_missing = 0.006

    u = rng.random((4, xx.size))
    cls = np.zeros(xx.size, dtype=int)
    cls[u[3] < p_missing] = 4
    cls[u[2] < p_bridge] = 3
    cls[u[1] < p_void] = 2
    cls[u[0] < p_nonwet] = 1
    return xx, yy, dnp, cls


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches any axis label or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        for art in (a.xaxis.label, a.yaxis.label, a.title):
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:40])


def main():
    set_pub_style(base_font=8)
    xx, yy, dnp, cls = build_array()
    n = cls.size
    die_w, die_h = NX * PITCH, NY * PITCH

    counts = np.array([int((cls == k).sum()) for k in range(len(CLASSES))])
    yield_pct = 100.0 * counts[0] / n
    n_bad = n - counts[0]
    dppm = 1e6 * counts / n

    order = np.argsort(counts[1:])[::-1] + 1
    par_counts = counts[order]
    par_ppm = dppm[order]
    par_cum = 100.0 * np.cumsum(par_counts) / par_counts.sum()
    par_names = [CLASSES[k][0] for k in order]
    par_cols = [CLASSES[k][1] for k in order]

    edges = np.linspace(0.0, dnp.max() * 1.0001, 9)
    mid = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(dnp, edges) - 1, 0, edges.size - 2)
    rate = np.array([100.0 * np.mean(cls[idx == b] != 0)
                     for b in range(edges.size - 1)])
    pop = np.array([int((idx == b).sum()) for b in range(edges.size - 1)])
    die_rate = 100.0 - yield_pct

    # ring / cluster populations, straight off the same geometry the guides use
    in_ring = (xx < RING_IN) | (xx > die_w - RING_IN) | \
              (yy < RING_IN) | (yy > die_h - RING_IN)
    in_cl = np.hypot(xx - die_w / 2.0, yy - die_h / 2.0) <= R_CLUSTER
    ring_rate = 100.0 * np.mean(cls[in_ring] == 1)
    cl_rate = 100.0 * np.mean(cls[in_cl] == 2)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_p = axes_inches(fig, AUX_T_RECT)
    ax_r = axes_inches(fig, AUX_B_RECT)

    # ---- main: the bump array ---------------------------------------------
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.30, die_w + 0.30)
    ax.set_ylim(-1.62, die_h + 0.62)

    ax.add_patch(Rectangle((-0.13, -0.13), die_w + 0.26, die_h + 0.26,
                           facecolor="#fafbfc", edgecolor="#9aa3ab",
                           lw=0.9, zorder=1))
    patches = [Circle((px, py), R_BUMP) for px, py in zip(xx, yy)]
    col = PatchCollection(patches, match_original=False, zorder=3)
    col.set_facecolor([CLASSES[k][1] for k in cls])
    col.set_edgecolor("white")
    col.set_linewidth(0.35)
    ax.add_collection(col)

    ax.add_patch(Rectangle((RING_IN, RING_IN), die_w - 2 * RING_IN,
                           die_h - 2 * RING_IN, facecolor="none",
                           edgecolor=RED, lw=0.9, ls=(0, (3.4, 2.2)),
                           zorder=6))
    ax.add_patch(Circle((die_w / 2.0, die_h / 2.0), R_CLUSTER,
                        facecolor="none", edgecolor=GOLD, lw=0.9,
                        ls=(0, (3.4, 2.2)), zorder=6))

    ax.text(-0.30, die_h + 0.22,
            f"first-pass bump yield {yield_pct:.1f} %\n"
            f"{n_bad} defective bumps, {1e6 * n_bad / n:,.0f} ppm",
            ha="left", va="bottom", fontsize=7.4, color=INK, linespacing=1.55,
            fontweight="bold")
    ax.text(die_w / 2.0, -0.30,
            f"die {die_w:.2f} x {die_h:.2f} mm   |   {NX} x {NY} array on a "
            f"{PITCH * 1000:.0f} $\\mu$m pitch   |   {n} bumps/die, 100 % "
            f"optical inspection",
            ha="center", va="top", fontsize=6.9, color=SUB)

    ax.text(-0.30, -0.66,
            f"- - -  outer band, > {RING_IN * 1000:.0f} $\\mu$m from the die "
            f"edge inward:\n      non-wet rate {ring_rate:.1f} % over "
            f"{int(in_ring.sum())} bumps",
            ha="left", va="top", fontsize=6.5, color=RED, linespacing=1.55)
    ax.text(die_w + 0.30, -0.66,
            f"- - -  centre disc, r $\\leq$ {R_CLUSTER * 1000:.0f} "
            f"$\\mu$m:\n      void rate {cl_rate:.1f} % over "
            f"{int(in_cl.sum())} bumps",
            ha="right", va="top", fontsize=6.5, color=GOLD, linespacing=1.55)

    handles = [Line2D([], [], marker="o", ls="none", ms=5.2, mfc=c,
                      mec="white", mew=0.5, label=f"{lab} ({counts[i]})")
               for i, (lab, c) in enumerate(CLASSES)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
              ncol=3, frameon=False, fontsize=6.9, handletextpad=0.35,
              columnspacing=1.30, labelspacing=0.45)

    # ---- aux (b): Pareto of defect classes ---------------------------------
    style_axes(ax_p)
    ax_p.set_axisbelow(True)
    ax_p.grid(axis="y", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    xs = np.arange(par_ppm.size)
    ax_p.bar(xs, par_ppm, width=0.62, color=par_cols, edgecolor="white",
             linewidth=0.6, zorder=3)
    for xi, v, c in zip(xs, par_ppm, par_counts):
        ax_p.text(xi, v * 1.03 + par_ppm.max() * 0.012, f"{c}", ha="center",
                  va="bottom", fontsize=6.2, color=SUB)
    ax_p.set_xticks(xs)
    ax_p.set_xticklabels(par_names, fontsize=6.6)
    ax_p.set_ylim(0, par_ppm.max() * 1.32)
    ax_p.tick_params(labelsize=6.8)
    ax_p.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1000.0:.0f}k" if v else "0"))
    ax_p.set_ylabel("defective bumps (ppm)", fontsize=7.6)
    ax_p.set_title("Two classes carry the loss", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    ax_c = ax_p.twinx()
    ax_c.plot(xs, par_cum, color=INK, lw=1.1, marker="o", ms=3.2,
              mfc="white", mec=INK, mew=1.0, zorder=6)
    ax_c.set_ylim(0, 108)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_ylabel("cumulative share (%)", fontsize=7.6)
    ax_c.spines["top"].set_visible(False)
    ax_c.text(0.97, 0.40, f"top two classes\n= {par_cum[1]:.0f} % of defects",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    # ---- aux (c): defect rate against distance from neutral point ----------
    style_axes(ax_r)
    ax_r.set_axisbelow(True)
    ax_r.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_r.bar(mid * 1000.0, rate, width=(edges[1] - edges[0]) * 830.0,
             color=lighten(BLUE, 0.74), edgecolor="white", linewidth=0.5,
             zorder=2)
    ax_r.plot(mid * 1000.0, rate, color=BLUE, lw=1.5, marker="o", ms=3.2,
              mfc="white", mec=BLUE, mew=1.1, zorder=5)
    ax_r.axhline(die_rate, color=RED, lw=0.9, ls=(0, (3.2, 2.0)), zorder=4)
    ax_r.text(mid[-1] * 1000.0, die_rate * 1.06,
              f"die average {die_rate:.1f} %", ha="right", va="bottom",
              fontsize=6.4, color=RED)
    ax_r.set_xlim(0, dnp.max() * 1000.0 * 1.06)
    ax_r.set_ylim(0, rate.max() * 1.40)
    ax_r.tick_params(labelsize=6.8)
    ax_r.set_xlabel("distance from die neutral point ($\\mu$m)", fontsize=7.6,
                    labelpad=1.5)
    ax_r.set_ylabel("defect rate (%)", fontsize=7.6)
    ax_r.set_title("Centre and corners both bite", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_r.text(0.035, 0.95,
              f"outermost ring {rate[-1]:.1f} % over {pop[-1]} bumps,\n"
              f"{rate[-1] / max(rate[3:5].mean(), 1e-9):.1f}x the quiet "
              f"mid-radius band at {rate[3:5].mean():.1f} %",
              transform=ax_r.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    panel_labels_inches(fig, [ax, ax_p, ax_r], dx=-0.42, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 906")
    check_layout(fig, foot, [ax_p, ax_r, ax_c])
    savefig(fig, "906_pkg_bump_defect_map.png")


if __name__ == "__main__":
    main()
