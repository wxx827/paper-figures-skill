"""Decay of grid turbulence along a wind-tunnel working section.

Main panel: the turbulence intensity measured by a single hot wire on the
tunnel centreline, plotted against distance from the biplanar grid in mesh
lengths, on logarithmic axes because the decay is a power law.  Three grid
meshes are shown; each is fitted over its own far field and the decay exponent
printed beside it is the fitted value, not the value that generated the data.
An inset draws the grid itself, with the solidity computed from the bar width
and mesh drawn there.

The companion panels are the other two things a grid is characterised by: the
integral length scale, which grows as the intensity falls because the surviving
eddies are the large ones, and the return to isotropy, which the small mesh
reaches soonest because it has the most mesh lengths of tunnel to do it in.

Illustrative synthetic data: an ideal power-law decay sampled at discrete
stations with hot-wire scatter added.
"""
import textwrap

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
GOLD = "#c9a227"

FIG_W, FIG_H = 10.6, 6.4
MAIN_RECT = (0.66, 0.86, 5.26, 4.42)
AUX_T_RECT = (7.34, 3.52, 2.98, 1.86)
AUX_B_RECT = (7.34, 0.82, 2.98, 1.86)

U_INF = 20.0            # m/s, tunnel speed for every run
X_WORK = 2.00           # m, streamwise station of the model
X_MIN, X_MAX = 0.34, 3.30       # m, traversable range behind the grid
X0_M = 4.0              # virtual origin, in mesh lengths
BAR_RATIO = 0.22        # bar width divided by mesh length

CASES = [dict(mesh=12.7, amp=0.0390, n=1.30, c=BLUE),
         dict(mesh=25.4, amp=0.0455, n=1.24, c=GREEN),
         dict(mesh=50.8, amp=0.0520, n=1.18, c=RED)]


def station_data(case, rng, n_pt=26):
    """Hot-wire stations behind one grid: x/M, Tu in %, and length scale."""
    x = np.linspace(X_MIN, X_MAX, n_pt)
    xm = x / (case["mesh"] * 1e-3)
    s = np.clip(xm - X0_M, 1.0, None)
    var = case["amp"] * s ** (-case["n"])
    tu = 100.0 * np.sqrt(var) * (1.0 + rng.normal(0.0, 0.016, n_pt))
    scale = 0.205 * case["mesh"] * s ** 0.42 * (1.0 + rng.normal(0.0, 0.022,
                                                                n_pt))
    iso = ((1.0 + (0.150 + 0.0042 * case["mesh"])
            * np.exp(-s / (30.0 + 0.95 * case["mesh"])))
           * (1.0 + rng.normal(0.0, 0.004, n_pt)))
    return x, xm, tu, scale, iso


def fit_exponent(xm, tu, keep_frac=45.0):
    """Decay exponent n from a straight-line fit of the far-field points."""
    k = xm > np.percentile(xm, keep_frac)
    slope, icpt = np.polyfit(np.log(xm[k]), np.log(tu[k]), 1)
    return -2.0 * slope, slope, icpt, k


def draw_grid_inset(ax, mesh, bars=3):
    """Schematic biplanar grid; returns the solidity of what is drawn."""
    d = BAR_RATIO
    span = bars + d
    ax.add_patch(Rectangle((0.0, 0.0), span, span, facecolor="white",
                           edgecolor="#9aa3ab", lw=0.7))
    for i in range(bars + 1):
        ax.add_patch(Rectangle((i - d / 2.0 + d / 2.0, 0.0), d, span,
                               facecolor="#4a4a4a", edgecolor="none"))
        ax.add_patch(Rectangle((0.0, i - d / 2.0 + d / 2.0), span, d,
                               facecolor="#4a4a4a", edgecolor="none"))
    ax.annotate("", xy=(d / 2.0 + d, 0.42), xytext=(d / 2.0 + 1.0 + d, 0.42),
                arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=0.8,
                                shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(d / 2.0 + 0.5 + d, 0.50, "M", fontsize=6.2, color=RED,
            ha="center", va="bottom")
    ax.set_xlim(-0.08, span + 0.08)
    ax.set_ylim(-0.08, span + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    return 2.0 * d - d ** 2


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(60413)
    runs = [(cs, *station_data(cs, rng)) for cs in CASES]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_l = axes_inches(fig, AUX_T_RECT)
    ax_i = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_l, ax_i):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, which="both", zorder=0)
        style_axes(a)

    # ---- main: the decay law -----------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    notes, work = [], []
    for cs, x, xm, tu, _, _ in runs:
        n_fit, slope, icpt, k = fit_exponent(xm, tu)
        xf = np.array([xm[k].min() * 0.80, xm.max() * 1.22])
        ax.plot(xf, np.exp(icpt) * xf ** slope, color=cs["c"], lw=1.2,
                alpha=0.75, zorder=4)
        ax.plot(xm, tu, ls="none", marker="o", ms=3.8, mfc="white",
                mec=cs["c"], mew=1.15, zorder=6,
                label=f"M = {cs['mesh']:.1f} mm")
        j = int(np.argmin(np.abs(x - X_WORK)))
        ax.plot([xm[j]], [tu[j]], marker="o", ms=5.2, color=cs["c"], zorder=8)
        notes.append(f"M = {cs['mesh']:4.1f} mm: fitted n = {n_fit:.2f}")
        work.append((cs, xm[j], tu[j]))

    ref_x = np.array([28.0, 95.0])
    ref_y = 5.9 * (ref_x / ref_x[0]) ** (-0.60)
    ax.plot(ref_x, ref_y, color=INK, lw=1.3, ls=(0, (5.0, 2.4)), zorder=7)
    ax.text(ref_x[0] * 0.42, ref_y[0] * 1.16,
            "reference decay\n$u'/U \\propto (x/M)^{-0.60}$, i.e. n = 1.20",
            fontsize=6.7, color=INK, ha="left", va="bottom",
            linespacing=1.45, zorder=9)
    for cs, xj, tj in work:
        ax.annotate(f"{tj:.2f} %", xy=(xj, tj), xytext=(xj * 0.80, tj * 0.80),
                    fontsize=6.5, color=cs["c"], ha="right", va="top",
                    zorder=9, arrowprops=dict(arrowstyle="-", color=cs["c"],
                                              lw=0.7, shrinkA=1, shrinkB=3))
    ax.text(0.020, 0.030,
            "filled circles: the model station, x = "
            f"{X_WORK:.2f} m\n" + "\n".join(notes) + "\nfits use the "
            "downstream 55 % of each traverse, clear of the virtual origin",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=INK, linespacing=1.50, zorder=11,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))

    ax.set_xlim(5.0, 330.0)
    ax.set_ylim(0.30, 9.0)
    ax.tick_params(labelsize=7.2, which="both")
    ax.set_xlabel("distance from the grid in mesh lengths  x/M  "
                  "(dimensionless)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("turbulence intensity  $u'/U_\\infty$  (%)", fontsize=8.4)
    ax.set_title(f"Grid turbulence decaying at $U_\\infty$ = {U_INF:.0f} m/s, "
                 "three biplanar grids", fontsize=9.0, fontweight="bold",
                 pad=5.0)
    ax.legend(loc="lower left", frameon=False, fontsize=6.9, handlelength=1.2,
              labelspacing=0.30, bbox_to_anchor=(0.020, 0.245))

    axg = ax.inset_axes([0.700, 0.615, 0.280, 0.345])
    sigma = draw_grid_inset(axg, CASES[1]["mesh"])
    axg.set_title(f"biplanar grid, solidity {sigma:.2f}", fontsize=6.5,
                  color=SUB, pad=2.0)

    # ---- aux (b): integral length scale ------------------------------------
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    grow = []
    for cs, x, xm, _, scale, _ in runs:
        ax_l.plot(xm, scale, ls="none", marker="s", ms=3.2, mfc="white",
                  mec=cs["c"], mew=1.1, zorder=6)
        p = np.polyfit(np.log(xm), np.log(scale), 1)
        xf = np.array([xm.min() * 0.85, xm.max() * 1.18])
        ax_l.plot(xf, np.exp(p[1]) * xf ** p[0], color=cs["c"], lw=1.3,
                  zorder=5)
        grow.append(p[0])
    ax_l.set_xlim(5.0, 330.0)
    ax_l.tick_params(labelsize=6.8, which="both")
    ax_l.set_xlabel("x/M (dimensionless)", fontsize=7.5, labelpad=1.5)
    ax_l.set_ylabel("integral length scale $L_x$ (mm)", fontsize=7.5)
    ax_l.set_title("The surviving eddies are the big ones", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_l.text(0.975, 0.055,
              "fitted growth exponents, small to large mesh:\n"
              + ", ".join(f"{g:.2f}" for g in grow),
              transform=ax_l.transAxes, ha="right", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)

    # ---- aux (c): return to isotropy ---------------------------------------
    ax_i.axhline(1.0, color=INK, lw=1.0, ls=(0, (4.0, 2.2)), zorder=3)
    ax_i.text(0.030, 1.006, "isotropic, $u'/v'$ = 1",
              transform=ax_i.get_yaxis_transform(), ha="left", va="bottom",
              fontsize=6.4, color=INK)
    ends = []
    for cs, x, xm, _, _, iso in runs:
        ax_i.plot(xm, iso, color=cs["c"], lw=1.5, marker="o", ms=2.6,
                  mfc="white", mew=0.9, zorder=6,
                  label=f"M = {cs['mesh']:.1f} mm")
        ends.append((cs["mesh"], float(iso[-1])))
    ax_i.set_xscale("log")
    ax_i.set_xlim(5.0, 330.0)
    ax_i.set_ylim(0.895, 1.44)
    ax_i.tick_params(labelsize=6.8, which="both")
    ax_i.set_xlabel("x/M (dimensionless)", fontsize=7.5, labelpad=1.5)
    ax_i.set_ylabel("anisotropy ratio $u'/v'$ (dimensionless)", fontsize=7.5)
    ax_i.set_title("The small mesh gets there first", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_i.text(0.975, 0.945,
              "at the last station:\n"
              + ",  ".join(f"{m:.1f} mm gives {v:.2f}" for m, v in ends),
              transform=ax_i.transAxes, ha="right", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)
    ax_i.legend(loc="lower left", frameon=False, fontsize=6.3, ncol=3,
                columnspacing=0.9,
                handlelength=1.3, labelspacing=0.24, borderaxespad=0.30)

    panel_labels_inches(fig, [ax, ax_l, ax_i], dx=-0.60, dy=0.10)
    foot = add_footer(fig, "Synthetic data, seed=60413")
    check_layout(fig, foot, [ax, ax_l, ax_i])
    savefig(fig, "934_wt_turbulence_decay.png")


if __name__ == "__main__":
    main()
