"""Gaussian plume from an elevated stack: footprint, decay and plume rise.

The main panel is the ground-level concentration footprint in plan view, drawn
on an equal-aspect axes because a dispersion footprint is a statement about
where the material lands and a stretched map would move it.  The stack sits at
the origin, the wind blows along +x, and the ground-level maximum is not
asserted: it is found by taking the argmax of the very field that is
contoured, so the distance printed on the panel and the contour pattern cannot
disagree.

The companion panels are the two reductions a dispersion assessment needs.
The centreline decay compares six Pasquill-Gifford stability classes on one
log-log axis - unstable air brings the peak in close and dilutes it fast,
stable air pushes it kilometres downwind - and the plume-rise panel puts
height on the vertical axis and shows how buoyant rise, capped by the Briggs
final-rise formula, sets the effective release height that every curve above
was computed with.

Illustrative synthetic source: one 60 m stack, Briggs open-country dispersion
coefficients and Briggs buoyant plume rise.
"""
import textwrap

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, style_axes, get_cmap, add_footer
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
BROWN = "#8c5a3c"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.62, 1.02, 5.35, 4.15)
AUX_T_RECT = (7.45, 3.42, 3.10, 1.80)
AUX_B_RECT = (7.45, 1.02, 3.10, 1.80)

Q_SRC = 12.0            # g/s, mass emission rate of the tracer gas
U_WIND = 4.0            # m/s, mean wind speed at stack top
H_STACK = 60.0          # m, physical stack height above ground
D_STACK = 2.4           # m, stack exit diameter
V_EXIT = 15.0           # m/s, stack exit velocity
T_EXIT = 420.0          # K, stack exit temperature
T_AMB = 288.0           # K, ambient temperature
DTHDZ = 0.020           # K/m, potential-temperature gradient, stable classes
G = 9.81                # m/s^2

MAIN_CLASS = "C"
CLASSES = ["A", "B", "C", "D", "E", "F"]
CLASS_NAME = {"A": "A very unstable", "B": "B unstable", "C": "C slightly unstable",
              "D": "D neutral", "E": "E slightly stable", "F": "F stable"}
CLASS_COL = {"A": GOLD, "B": BROWN, "C": RED, "D": BLUE, "E": GREEN,
             "F": PURPLE}
STABLE = ("E", "F")


def sigmas(x, cls):
    """Briggs open-country dispersion coefficients, metres, for x in metres."""
    x = np.maximum(np.asarray(x, dtype=float), 1.0)
    ay = {"A": 0.22, "B": 0.16, "C": 0.11, "D": 0.08, "E": 0.06, "F": 0.04}[cls]
    sy = ay * x / np.sqrt(1.0 + 1e-4 * x)
    if cls == "A":
        sz = 0.20 * x
    elif cls == "B":
        sz = 0.12 * x
    elif cls == "C":
        sz = 0.08 * x / np.sqrt(1.0 + 2e-4 * x)
    elif cls == "D":
        sz = 0.06 * x / np.sqrt(1.0 + 1.5e-3 * x)
    elif cls == "E":
        sz = 0.03 * x / (1.0 + 3e-4 * x)
    else:
        sz = 0.016 * x / (1.0 + 3e-4 * x)
    return sy, sz


def buoyancy_flux():
    """Briggs buoyancy flux parameter F, m^4 s^-3."""
    return (G * V_EXIT * D_STACK ** 2 / 4.0) * (T_EXIT - T_AMB) / T_EXIT


def plume_rise(x, cls):
    """Transitional Briggs rise capped at the final rise, metres."""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    fb = buoyancy_flux()
    trans = 1.6 * fb ** (1.0 / 3.0) * x ** (2.0 / 3.0) / U_WIND
    if cls in STABLE:
        s = G / T_AMB * DTHDZ
        final = 2.6 * (fb / (U_WIND * s)) ** (1.0 / 3.0)
    else:
        xf = 49.0 * fb ** 0.625 if fb < 55.0 else 119.0 * fb ** 0.4
        final = 1.6 * fb ** (1.0 / 3.0) * xf ** (2.0 / 3.0) / U_WIND
    return np.minimum(trans, final)


def effective_height(cls):
    """Stack height plus the final plume rise, metres, read off the curve."""
    return H_STACK + float(plume_rise(np.linspace(0.0, 4.0e4, 4001), cls).max())


def ground_conc(x, y, cls):
    """Ground-level concentration with full reflection, ug m^-3."""
    sy, sz = sigmas(x, cls)
    h = effective_height(cls)
    c = (Q_SRC / (np.pi * U_WIND * sy * sz)
         * np.exp(-0.5 * (y / sy) ** 2) * np.exp(-0.5 * (h / sz) ** 2))
    return c * 1.0e6


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
    rng = np.random.default_rng(1008)

    # ---- the field the main panel contours ---------------------------------
    xg = np.linspace(20.0, 3150.0, 620)
    yg = np.linspace(-1320.0, 1320.0, 520)
    X, Y = np.meshgrid(xg, yg)
    C = ground_conc(X, Y, MAIN_CLASS)
    k = np.unravel_index(np.argmax(C), C.shape)
    c_max, x_max = float(C[k]), float(X[k])
    h_eff = effective_height(MAIN_CLASS)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_d = axes_inches(fig, AUX_B_RECT)
    for a in (ax_c, ax_d):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the ground-level footprint ----------------------------------
    ax.set_aspect("equal")
    levels = np.linspace(0.0, c_max, 24)
    im = ax.contourf(X / 1e3, Y / 1e3, C, levels=levels,
                     cmap=get_cmap("cb_sequential_warm"), zorder=2)
    show = np.array([0.10, 0.25, 0.50, 0.75, 0.95]) * c_max
    cs = ax.contour(X / 1e3, Y / 1e3, C, levels=show, colors="#4a3a30",
                    linewidths=0.6, alpha=0.85, zorder=4)
    ax.clabel(cs, fmt="%.1f", fontsize=6, inline=True, inline_spacing=4)

    ax.axhline(0.0, color="#5a4a40", lw=0.7, ls=(0, (5.0, 3.0)), alpha=0.7,
               zorder=3)
    ax.plot([x_max / 1e3], [0.0], marker="o", ms=5.4, mfc="white", mec=RED,
            mew=1.5, zorder=8)
    halo_text(ax, x_max / 1e3, 0.075,
              f"ground-level maximum {c_max:.1f} $\\mu g/m^3$",
              ha="center", va="bottom", fontsize=6.5, color=RED, halo="white",
              halo_lw=2.4, zorder=9)

    rx = rng.uniform(0.35, 2.95, 7) * 1e3
    ry = rng.uniform(-1.05, 1.05, 7) * 1e3
    rc = ground_conc(rx, ry, MAIN_CLASS)
    ax.plot(rx / 1e3, ry / 1e3, ls="none", marker="s", ms=3.2, mfc="white",
            mec=INK, mew=0.8, zorder=8)
    j_hot = int(np.argmax(rc))
    halo_text(ax, rx[j_hot] / 1e3, ry[j_hot] / 1e3 - 0.055,
              f"worst receptor {rc[j_hot]:.1f} $\\mu g/m^3$", ha="center",
              va="top", fontsize=6.3, color=INK, halo="white", halo_lw=2.2,
              zorder=9)

    ax.add_patch(Rectangle((-0.055, -0.030), 0.055, 0.060, facecolor="#3a3a3a",
                           edgecolor="white", lw=0.7, zorder=8))
    halo_text(ax, -0.075, 0.0, "stack", ha="right", va="center", fontsize=6.6,
              color=INK, halo="white", halo_lw=2.2, zorder=9)
    ax.annotate("", xy=(0.42, 1.12), xytext=(0.06, 1.12),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2,
                                shrinkA=0, shrinkB=0), zorder=9)
    ax.text(0.24, 1.155, f"wind {U_WIND:.0f} m/s", ha="center", va="bottom",
            fontsize=6.8, color=INK, zorder=9)

    ax.set_xlim(-0.25, 3.15)
    ax.set_ylim(-1.32, 1.32)
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("downwind distance x (km)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("crosswind distance y (km)", fontsize=8.4)
    ax.set_title("Ground-level concentration footprint, Pasquill class "
                 f"{MAIN_CLASS}, {Q_SRC:.0f} g/s release", fontsize=9.0,
                 fontweight="bold", pad=5.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sy_at, sz_at = sigmas(np.array([x_max]), MAIN_CLASS)
    ax.text(0.012, 0.020,
            f"effective release height H = {h_eff:.0f} m "
            f"({H_STACK:.0f} m stack + {h_eff - H_STACK:.0f} m buoyant rise)\n"
            f"peak lands {x_max / 1e3:.2f} km downwind, where "
            f"$\\sigma_y$ = {sy_at[0]:.0f} m and $\\sigma_z$ = "
            f"{sz_at[0]:.0f} m\nfull reflection at the ground; open squares "
            "are seven receptor sites, contour labels in $\\mu g/m^3$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            color=INK, linespacing=1.50, zorder=12,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))
    safe_colorbar(fig, im, ax, label="conc. ($\\mu g/m^3$)",
                  width=0.13, pad=0.09, shrink=0.86, labelsize=7.0,
                  ticks=np.round(np.linspace(0.0, c_max, 6), 1))

    # ---- aux (b): centreline decay by stability class -----------------------
    xs = np.logspace(np.log10(150.0), np.log10(4.0e4), 420)
    peaks = {}
    for cls in CLASSES:
        cc = ground_conc(xs, 0.0, cls)
        peaks[cls] = (float(xs[np.argmax(cc)]), float(cc.max()))
        lw = 1.9 if cls == MAIN_CLASS else 1.15
        ax_c.plot(xs / 1e3, cc, color=CLASS_COL[cls], lw=lw, zorder=6,
                  label=CLASS_NAME[cls])
        ax_c.plot([peaks[cls][0] / 1e3], [peaks[cls][1]], marker="o", ms=2.9,
                  mfc="white", mec=CLASS_COL[cls], mew=1.0, zorder=7)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(0.15, 40.0)
    ax_c.set_ylim(0.006, 900.0)
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_xlabel("downwind distance x (km)", fontsize=7.5, labelpad=1.5)
    ax_c.set_ylabel("centreline conc. ($\\mu g/m^3$)", fontsize=7.5)
    ax_c.set_title("Stability decides where the peak lands", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_c.text(0.030, 0.055,
              f"class A peaks at {peaks['A'][0] / 1e3:.2f} km, class E at "
              f"{peaks['E'][0] / 1e3:.1f} km;\nopen circles are the maxima of "
              "the plotted curves",
              transform=ax_c.transAxes, ha="left", va="bottom", fontsize=6.3,
              color=SUB, linespacing=1.45)
    ax_c.legend(loc="upper left", frameon=False, fontsize=6.0, ncol=2,
                handlelength=1.3, columnspacing=0.9, labelspacing=0.20,
                borderaxespad=0.20)

    # ---- aux (c): plume rise, height on the vertical axis -------------------
    xr = np.linspace(0.0, 2000.0, 400)
    for cls, ls in (("D", "-"), ("F", (0, (4.4, 2.0)))):
        z = H_STACK + plume_rise(xr, cls)
        ax_d.plot(xr / 1e3, z, color=CLASS_COL[cls], lw=1.8, ls=ls, zorder=6)
        zf = float(z.max())
        ax_d.axhline(zf, color=CLASS_COL[cls], lw=0.7, ls=(0, (1.6, 1.8)),
                     alpha=0.85, zorder=3)
        ax_d.text(1.985, zf + 3.0, f"H = {zf:.0f} m, class {cls}", ha="right",
                  va="bottom", fontsize=6.4, color=CLASS_COL[cls])
    ax_d.add_patch(Rectangle((-0.03, 0.0), 0.06, H_STACK, facecolor="#c8c8c8",
                             edgecolor="#5a5a5a", lw=0.7, zorder=5))
    ax_d.axhline(H_STACK, color=FAINT, lw=0.8, ls=(0, (3.0, 2.2)), zorder=2)
    ax_d.text(0.075, H_STACK - 6.0, f"stack top {H_STACK:.0f} m", ha="left",
              va="top", fontsize=6.4, color=SUB)
    ax_d.set_xlim(-0.05, 2.0)
    ax_d.set_ylim(0.0, 235.0)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_xlabel("downwind distance x (km)", fontsize=7.5, labelpad=1.5)
    ax_d.set_ylabel("plume centreline height (m)", fontsize=7.5)
    ax_d.set_title("Buoyant rise sets the release height", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_d.text(0.975, 0.055,
              f"buoyancy flux F = {buoyancy_flux():.0f} $m^4/s^3$; stable air "
              "caps\nthe rise early, which is why class F stays aloft",
              transform=ax_d.transAxes, ha="right", va="bottom", fontsize=6.3,
              color=SUB, linespacing=1.45)
    ax_d.legend(handles=[Line2D([], [], color=BLUE, lw=1.8,
                                label="neutral (D)"),
                         Line2D([], [], color=PURPLE, lw=1.8,
                                ls=(0, (4.4, 2.0)), label="stable (F)")],
                loc="center right", frameon=False, fontsize=6.4,
                handlelength=1.7, labelspacing=0.26, borderaxespad=0.45)

    panel_labels_inches(fig, [ax, ax_c, ax_d], dx=-0.52, dy=0.10)
    foot = add_footer(fig, "Synthetic data, seed=1008")
    check_layout(fig, foot, [ax, ax_c, ax_d])
    savefig(fig, "1008_aqm_gaussian_plume.png")


if __name__ == "__main__":
    main()
