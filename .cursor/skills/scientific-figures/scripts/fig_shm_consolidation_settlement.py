"""Terzaghi consolidation of a doubly drained clay under a surcharge fill.

Main panel: excess pore pressure isochrones through the profile at seven
elapsed times, drawn against a depth axis that is positive downward, with the
two sand drainage boundaries and the initial pressure distribution shown for
reference.  Auxiliary (both at 58 % of the main panel width): the settlement
time curve on a logarithmic time axis with t50 and t90 read off it, and the
settlement each metre of clay contributes at one year and at the end of
primary consolidation.

The initial excess pressure is the 2:1 spread of the fill load, so it decays
with depth; the isochrones are the general Fourier series solution for that
distribution, the coefficients being integrated numerically from the same
array that is plotted at t = 0.  Settlement is the compression integral
m_v times (u0 - u) through the clay, taken by trapezoid on the plotted
isochrones, so the curve, the sublayer bars and the ultimate settlement all
come from one field.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_cmap, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
TEAL = "#2a7f7a"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"

SAND = lighten("#c9a227", 0.62)
CLAY = lighten("#8c6d4f", 0.58)
FILL = lighten("#7a6a5d", 0.45)

Z_TOP, Z_BOT = 2.0, 10.0        # clay layer, metres below ground surface
Z_MAX = 11.6
HD = 0.5 * (Z_BOT - Z_TOP)      # drainage path, double drainage
CV = 3.5 / 365.0                # m^2/day
MV = 4.2e-4                     # m^2/kN
Q_FILL, B_FILL, L_FILL = 110.0, 24.0, 40.0
TIMES = np.array([30.0, 90.0, 180.0, 365.0, 730.0, 1460.0, 2920.0])
NTERM = 80

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def initial_pressure(z):
    """2:1 load spread under a rectangular fill, in kPa at depth z."""
    return Q_FILL * B_FILL * L_FILL / ((B_FILL + z) * (L_FILL + z))


def isochrone(zc, u0, t):
    """Excess pore pressure at time ``t`` (days) for the given u0 profile."""
    zz = (zc - Z_TOP) / HD                     # 0 at top, 2 at base
    tf = CV * t / HD ** 2
    u = np.zeros_like(zc)
    for m in range(NTERM):
        mm = (2 * m + 1) * np.pi / 2.0
        coef = np.trapezoid(u0 * np.sin(mm * zz), zz)
        u += coef * np.sin(mm * zz) * np.exp(-mm ** 2 * tf)
    return u


def settlement(zc, u0, t):
    """Compression of the clay at time ``t``, in millimetres."""
    return 1000.0 * MV * np.trapezoid(u0 - isochrone(zc, u0, t), zc)


def sublayer_settlement(zc, u0, t, edges):
    """Settlement contributed by each 1 m sublayer, in millimetres."""
    du = u0 - isochrone(zc, u0, t)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (zc >= a) & (zc <= b)
        out.append(1000.0 * MV * np.trapezoid(du[m], zc[m]))
    return np.array(out)


def main_panel(ax, zc, u0, curves, cmap, norm):
    for z0, z1, fc in ((0.0, Z_TOP, SAND), (Z_TOP, Z_BOT, CLAY),
                       (Z_BOT, Z_MAX, SAND)):
        ax.axhspan(z0, z1, facecolor=fc, alpha=0.55, lw=0.0, zorder=1)
    for x, z, name, ha in ((1.5, 1.0, "medium sand (drain)", "left"),
                           (1.5, 10.8, "medium sand (drain)", "left"),
                           (105.0, 9.2, "soft clay", "right")):
        ax.text(x, z, name, fontsize=6.4, color=SUB, ha=ha, va="center",
                zorder=6)
    for zb in (Z_TOP, Z_BOT):
        ax.axhline(zb, color="#8a8a8a", lw=0.9, zorder=4)

    ax.plot(u0, zc, color=INK, lw=1.6, ls=(0, (4.0, 2.0)), zorder=7,
            label="t = 0 (2:1 load spread)")
    # One label per isochrone, each at its own depth, so a family of curves
    # that all bunch toward the drained faces cannot stack its labels.
    z_lab = np.linspace(Z_TOP + 0.7, Z_BOT - 0.7, len(TIMES))
    for t, u, zl in zip(TIMES, curves, z_lab):
        c = cmap(norm(t))
        ax.plot(u, zc, color=c, lw=1.5, zorder=6)
        ax.text(float(np.interp(zl, zc, u)) + 1.6, zl, "{:.0f} d".format(t),
                color=c, fontsize=6.1, ha="left", va="center", zorder=8)

    ax.set_xlim(-3.0, 108.0)
    ax.set_ylim(Z_MAX, -0.35)
    ax.set_xlabel("excess pore water pressure (kPa)")
    ax.set_ylabel("depth below ground surface (m), positive downward")
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Pore pressure isochrones, drained top and bottom",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    ax.legend(loc="upper right", fontsize=6.4, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.6, borderaxespad=0.4)

    cax = ax.inset_axes([0.56, 0.10, 0.36, 0.030])
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label("elapsed time (days)", fontsize=6.2, color=INK)
    cb.set_ticks([30.0, 100.0, 365.0, 1000.0, 2920.0])
    cb.ax.set_xticklabels(["30", "100", "365", "1000", "2920"])
    cb.ax.tick_params(labelsize=6, width=0.6, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")


def curve_panel(ax, tg, sg, s_ult, t50, t90):
    ax.plot(tg, sg, color=TEAL, lw=1.8, zorder=5)
    ax.axhline(s_ult, color=MUTED, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    ax.text(tg[0] * 1.15, s_ult - 6.0, "ultimate {:.0f} mm".format(s_ult),
            fontsize=6.0, color=SUB, ha="left", va="bottom", zorder=6)
    for t, frac, col in ((t50, 0.5, BLUE), (t90, 0.9, RED)):
        ax.plot([t, t], [0.0, frac * s_ult], color=col, lw=0.8,
                ls=(0, (2.5, 1.8)), zorder=4)
        ax.plot([tg[0], t], [frac * s_ult] * 2, color=col, lw=0.8,
                ls=(0, (2.5, 1.8)), zorder=4)
        ax.plot([t], [frac * s_ult], marker="o", ms=3.4, mfc=col,
                mec="white", mew=0.6, zorder=7)
        ax.text(t * 1.08, frac * s_ult - 8.0,
                "t{:.0f} = {:.0f} d".format(100 * frac, t), fontsize=6.0,
                color=col, ha="left", va="bottom", zorder=7)
    ax.set_xscale("log")
    ax.set_xlim(tg[0], tg[-1])
    ax.set_ylim(s_ult * 1.12, 0.0)
    ax.set_xlabel("time since loading (days, log scale)", fontsize=6.8)
    ax.set_ylabel("settlement (mm)", fontsize=6.8)
    ax.grid(which="both", lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Settlement plotted positive downward", fontsize=8.0,
                 fontweight="bold", color=INK)


def sublayer_panel(ax, edges, s_year, s_end):
    mid = 0.5 * (edges[:-1] + edges[1:])
    h = 0.72 * (edges[1] - edges[0])
    ax.barh(mid, s_end, height=h, facecolor=lighten(TEAL, 0.55),
            edgecolor=TEAL, linewidth=0.7, zorder=3,
            label="end of primary")
    ax.barh(mid, s_year, height=h * 0.55, facecolor=ORANGE,
            edgecolor="white", linewidth=0.5, zorder=4, label="at 365 days")
    for y, a, b in zip(mid, s_year, s_end):
        ax.text(b + 0.8, y, "{:.0f} %".format(100.0 * a / b), fontsize=6,
                color=SUB, ha="left", va="center", zorder=6)
    ax.set_ylim(Z_BOT, Z_TOP)
    ax.set_xlim(0.0, s_end.max() * 1.30)
    ax.set_xlabel("settlement of a 1 m sublayer (mm)", fontsize=6.8)
    ax.set_ylabel("depth (m), positive downward", fontsize=6.8)
    ax.grid(axis="x", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.2, borderaxespad=0.3, labelspacing=0.24)
    ax.set_title("Drains work inward from both faces", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    np.random.default_rng(8127)

    zc = np.linspace(Z_TOP, Z_BOT, 401)
    u0 = initial_pressure(zc)
    curves = [isochrone(zc, u0, t) for t in TIMES]

    tg = np.logspace(np.log10(10.0), np.log10(6000.0), 220)
    sg = np.array([settlement(zc, u0, t) for t in tg])
    s_ult = 1000.0 * MV * np.trapezoid(u0, zc)
    t50 = float(np.interp(0.5 * s_ult, sg, tg))
    t90 = float(np.interp(0.9 * s_ult, sg, tg))

    edges = np.arange(Z_TOP, Z_BOT + 0.01, 1.0)
    s_year = sublayer_settlement(zc, u0, 365.0, edges)
    s_end = sublayer_settlement(zc, u0, 40000.0, edges)
    assert abs(s_end.sum() - s_ult) < 1.0, "sublayers must sum to ultimate"
    assert np.all(np.diff(sg) >= -1e-9), "settlement must be monotonic"

    # The pale end of the sequential ramp is almost white on paper, so the
    # first isochrone would vanish; keep the top three quarters of it.
    base = get_cmap("cb_sequential_blue")
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "clay_time", base(np.linspace(0.42, 1.0, 256)))
    norm = mpl.colors.LogNorm(vmin=TIMES[0], vmax=TIMES[-1])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_curve = axes_inches(fig, AUX_T)
    ax_sub = axes_inches(fig, AUX_B)

    main_panel(ax_main, zc, u0, curves, cmap, norm)
    curve_panel(ax_curve, tg, sg, s_ult, t50, t90)
    sublayer_panel(ax_sub, edges, s_year, s_end)

    u_full = np.concatenate([[0.0], u0, [0.0]])


    panel_labels_inches(fig, [ax_main, ax_curve, ax_sub], dx=-0.62, dy=0.14)
    savefig(fig, "985_shm_consolidation_settlement.png")


if __name__ == "__main__":
    main()
