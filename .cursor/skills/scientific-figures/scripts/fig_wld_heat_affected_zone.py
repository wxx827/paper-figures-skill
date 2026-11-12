"""Metallurgical zones of an arc weld, and the hardness they leave behind.

The cross-section and the hardness traverse under it are one argument in two
panels: both are functions of the same peak-temperature field, plotted on the
same transverse axis, so the coarse-grain hardness peak and the intercritical
soft zone line up with the bands that cause them instead of being asserted to.
The two panels on the right supply the mechanism - the thermal cycle seen at
three distances from the fusion line, and where those cycles fall on the
continuous-cooling diagram.  Every cooling time and every hardness extreme is
measured from the curves as drawn.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, lighten, darken, savefig
from figkit import (add_scale_bar, axes_inches, halo_text,
                    panel_labels_inches)

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
GREY = "#808080"
HAIR = "#cccccc"

# Cross-section geometry, mm
X_HALF, PLATE_T = 16.0, 9.0
CAP_W, CAP_H = 6.2, 1.8
Y_TOP = 2.6

# Peak-temperature field of the pass: an elliptical decay about the arc root.
T_AMB, T_ARC = 25.0, 2900.0
A_ELL, B_ELL, Y_SRC = 7.5, 6.0, -0.5

T_FZ, T_CG, T_FG, T_AC1 = 1500.0, 1100.0, 900.0, 730.0
ZONE_T = [T_AC1, T_FG, T_CG, T_FZ]
ZONE_NAMES = ["Base metal", "Intercritical HAZ", "Fine-grain HAZ",
              "Coarse-grain HAZ", "Fusion zone"]
ZONE_FILL = ["#f0f0f0", lighten(GOLD, 0.60), lighten(GREEN, 0.58),
             lighten(BLUE, 0.55), lighten(RED, 0.38)]
ZONE_HATCH = ["", "..", "//", "\\\\", ""]

# Rykalin thick-section thermal cycle
Q_LINE = 900.0          # J/mm, arc energy per unit length
LAMBDA = 0.030          # W / (mm K)
DIFF = 5.6              # mm^2 / s
R_PROBE = [5.2, 6.2, 7.6]     # mm from the arc axis
CYCLE_COL = [RED, BLUE, GREEN]
# Arc energies compared on the CCT, each sampled at its own fusion boundary
# so the peak temperature is held fixed and only the cooling rate changes.
Q_SWEEP = [250.0, 1000.0, 5000.0]
Q_COL = ["#e69a54", "#cc722f", "#66300c"]
T_PROBE_PEAK = 1350.0
MS_TEMP, MF_TEMP = 410.0, 250.0


def peak_temperature(x, y):
    u = np.sqrt((x / A_ELL) ** 2 + ((y - Y_SRC) / B_ELL) ** 2)
    return T_AMB + (T_ARC - T_AMB) * np.exp(-1.35 * u ** 1.7)


def cap_height(x):
    return np.where(np.abs(x) < CAP_W,
                    CAP_H * (1.0 - (x / CAP_W) ** 2), -1.0)


def section_field(nx=641, ny=301):
    x = np.linspace(-X_HALF, X_HALF, nx)
    y = np.linspace(-PLATE_T, Y_TOP, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    tp = peak_temperature(X, Y)
    solid = (Y <= 0.0) | (Y <= cap_height(X))
    return x, y, np.where(solid, tp, np.nan)


def hardness(tp):
    """Vickers from peak temperature: soft intercritical dip, hard CGHAZ."""
    knots_t = [25.0, 600.0, 730.0, 800.0, 880.0, 950.0, 1100.0, 1250.0,
               1450.0, 1500.0, 2000.0, T_ARC]
    knots_h = [196.0, 196.0, 192.0, 176.0, 188.0, 210.0, 236.0, 268.0,
               258.0, 250.0, 243.0, 240.0]
    return np.interp(tp, knots_t, knots_h)


def thermal_cycle(r, t, q=Q_LINE):
    return T_AMB + q / (2.0 * np.pi * LAMBDA * t) * np.exp(
        -r ** 2 / (4.0 * DIFF * t))


def radius_for_peak(q, t_peak=T_PROBE_PEAK):
    """Distance at which arc energy ``q`` just reaches ``t_peak``."""
    return np.sqrt(np.exp(-1.0) * 4.0 * DIFF * q /
                   (2.0 * np.pi * LAMBDA * (t_peak - T_AMB)))


def cooling_time_85(t, temp):
    peak = int(np.argmax(temp))
    tc, cc = t[peak:], temp[peak:]
    t8 = np.interp(-800.0, -cc, tc)
    t5 = np.interp(-500.0, -cc, tc)
    return t8, t5, t5 - t8


def cct_start(temp, shift=1.0):
    return shift * 14.0 * 10.0 ** (2.2e-5 * (temp - 620.0) ** 2)


def section_panel(ax, x, y, tp):
    levels = [0.0] + ZONE_T + [T_ARC + 10.0]
    cs = ax.contourf(x, y, tp, levels=levels, colors=ZONE_FILL,
                     hatches=ZONE_HATCH, zorder=2)
    cs.set_edgecolor("#9a9a9a")
    cs.set_linewidth(0.35)
    ax.contour(x, y, tp, levels=ZONE_T, colors=[RULE], linewidths=0.7,
               zorder=3)

    xs = np.linspace(-X_HALF, X_HALF, 801)
    top = np.maximum(cap_height(xs), 0.0)
    ax.plot(xs, top, color=RULE, lw=1.1, zorder=5)
    ax.plot([-X_HALF, X_HALF], [-PLATE_T, -PLATE_T], color=RULE, lw=1.1,
            zorder=5)
    for sx in (-X_HALF, X_HALF):
        ax.plot([sx, sx], [-PLATE_T, 0.0], color=RULE, lw=1.1, zorder=5)

    row = np.argmin(np.abs(y + 2.0))
    tprow = tp[row]
    for temp, txt, dy in [(T_FZ, "fusion boundary, " + f"{T_FZ:.0f}" +
                           r" $^{\circ}$C", 2.15),
                          (T_AC1, r"HAZ outer edge, $Ac_1$ = " +
                           f"{T_AC1:.0f}" + r" $^{\circ}$C", 1.10)]:
        left = x[np.nanargmin(np.abs(np.where(x < 0.0, tprow, np.nan) - temp))]
        ax.annotate(txt, xy=(left, -2.0), xytext=(-X_HALF + 0.4, dy),
                    fontsize=6.6, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7),
                    zorder=8)
    ax.text(X_HALF - 0.4, -PLATE_T + 0.5,
            f"plate {PLATE_T:.0f} mm, cap {2.0 * CAP_W:.1f} mm wide",
            ha="right", va="bottom", fontsize=6.4, color=SUB, zorder=8)
    add_scale_bar(ax, 5.0, label="5 mm", loc=(0.035, 0.10), height=0.045,
                  fontsize=6.2, color="#3a3a3a")

    ax.set_aspect("equal")
    ax.set_xlim(-X_HALF, X_HALF)
    ax.set_ylim(-PLATE_T, Y_TOP)
    ax.set_ylabel("depth (mm)", fontsize=7.8)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=6.8)
    ax.set_title("Four metallurgical zones between the weld metal and the "
                 "plate", fontsize=9.0, fontweight="bold", color=INK)
    handles = [Patch(facecolor=ZONE_FILL[i], edgecolor="#9a9a9a",
                     linewidth=0.5, hatch=ZONE_HATCH[i], label=ZONE_NAMES[i])
               for i in range(5)]
    ax.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
              fontsize=6.8, bbox_to_anchor=(0.5, -0.015), borderaxespad=0.0,
              handlelength=1.5, columnspacing=1.1)


def hardness_panel(ax, x, seed=4):
    rng = np.random.default_rng(seed)
    tp = peak_temperature(x, -1.0)
    hv = hardness(tp)

    bounds = [0.0] + ZONE_T
    for i in range(5):
        lo = bounds[i]
        hi = ZONE_T[i] if i < 4 else T_ARC + 10.0
        m = (tp >= lo) & (tp < hi)
        if m.any():
            ax.fill_between(x, 150.0, 290.0, where=m, color=ZONE_FILL[i],
                            alpha=0.75, linewidth=0.0, zorder=0)
    xm = np.arange(-15.0, 15.01, 0.9)
    hm = hardness(peak_temperature(xm, -1.0)) + rng.normal(0.0, 3.4, xm.size)
    ax.plot(x, hv, color=INK, lw=1.5, zorder=5)
    ax.plot(xm, hm, ls="none", marker="o", ms=3.0, mfc="white", mec=RULE,
            mew=0.8, zorder=6)

    bm = float(hv[0])
    ax.axhline(bm, color=GREY, lw=0.8, ls=(0, (3.0, 2.0)), zorder=3)
    i_hi = int(np.argmax(hv))
    right = x > 0.0
    i_lo = int(np.arange(x.size)[right][np.argmin(hv[right])])
    ax.annotate(f"CGHAZ peak {hv[i_hi]:.0f} HV", xy=(x[i_hi], hv[i_hi]),
                xytext=(x[i_hi] - 5.2, hv[i_hi] + 12.0), fontsize=6.6,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.annotate(f"soft zone {hv[i_lo]:.0f} HV, "
                f"{100.0 * (bm - hv[i_lo]) / bm:.0f} % below base metal",
                xy=(x[i_lo], hv[i_lo]), xytext=(x[i_lo] + 1.2, 158.0),
                fontsize=6.6, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.text(-X_HALF + 0.3, bm + 2.0, f"base metal {bm:.0f} HV", fontsize=6.3,
            color=SUB, va="bottom")

    ax.set_xlim(-X_HALF, X_HALF)
    ax.set_ylim(150.0, 290.0)
    ax.set_xlabel("transverse distance from weld centreline (mm)",
                  fontsize=7.8)
    ax.set_ylabel("hardness (HV1)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.grid(axis="y", lw=0.5, alpha=0.6)


def cycle_panel(ax, cycles):
    for (r, t, temp, t8, t5, dt), col in zip(cycles, CYCLE_COL):
        ax.semilogx(t, temp, color=col, lw=1.5, zorder=4,
                    label=f"r = {r:.1f} mm, " + r"$T_{peak}$ = " +
                          f"{temp.max():.0f}" + r" $^{\circ}$C")
    for lev in (800.0, 500.0):
        ax.axhline(lev, color=GREY, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
        ax.text(0.22, lev + 22.0, f"{lev:.0f}" + r" $^{\circ}$C",
                fontsize=6.2, color=SUB, va="bottom")
    r, t, temp, t8, t5, dt = cycles[1]
    ax.annotate("", xy=(t5, 650.0), xytext=(t8, 650.0),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.9))
    halo_text(ax, np.sqrt(t8 * t5), 672.0,
              r"$\Delta t_{8/5}$ = " + f"{dt:.1f} s", ha="center",
              va="bottom", fontsize=6.6, color=BLUE, halo="white",
              halo_lw=2.4, zorder=8)
    ax.set_xlim(0.2, 200.0)
    ax.set_ylim(0.0, 1650.0)
    ax.set_xlabel("time from arc passage (s)", fontsize=7.4)
    ax.set_ylabel(r"temperature ($^{\circ}$C)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=6.3, loc="upper right",
              borderaxespad=0.2, handlelength=1.4, labelspacing=0.3)
    spread = max(c[5] for c in cycles) / min(c[5] for c in cycles)
    ax.set_title("Peak temperature falls, " + r"$\Delta t_{8/5}$" +
                 f" moves only {100.0 * (spread - 1.0):.0f} %",
                 fontsize=8.2, fontweight="bold", color=INK)


def product_of(tc, cc):
    """Which constituent a cooling path ends in, read off the CCT curves."""
    probe = np.linspace(MS_TEMP + 5.0, 820.0, 200)
    t_path = np.interp(-probe, -cc, tc)
    crossed = t_path >= cct_start(probe)
    if not crossed.any():
        return "martensite"
    return "ferrite + pearlite" if probe[crossed].max() > 560.0 else "bainite"


def cct_panel(ax, sweep):
    temp = np.linspace(MS_TEMP, 860.0, 200)
    ax.plot(cct_start(temp), temp, color=RULE, lw=1.2, zorder=4)
    ax.plot(cct_start(temp, 3.6), temp, color=RULE, lw=1.0,
            ls=(0, (3.4, 2.0)), zorder=4)
    ax.fill_betweenx(temp, cct_start(temp), cct_start(temp, 3.6),
                     color="#f0f0f0", zorder=1)
    for lev, txt in [(MS_TEMP, r"$M_s$"), (MF_TEMP, r"$M_f$")]:
        ax.axhline(lev, color=RULE, lw=0.9, ls=(0, (5.0, 2.4)), zorder=3)
        ax.text(0.115, lev + 12.0, txt, fontsize=6.6, color=RULE,
                va="bottom")

    products = []
    for i, ((q, t, tt, t8, t5, dt), col) in enumerate(zip(sweep, Q_COL)):
        peak = int(np.argmax(tt))
        tc, cc = t[peak:], tt[peak:]
        keep = (cc >= 215.0) & (cc <= 860.0)
        prod = product_of(tc, cc)
        products.append(prod)
        ax.plot(tc[keep], cc[keep], color=col, lw=1.5, zorder=6)
        # Endpoint-style key: a boxed legend would land on the curves it
        # describes in a panel this small.
        halo_text(ax, 0.125, 575.0 - 42.0 * i, f"{q:.0f} J/mm, " +
                  r"$\Delta t_{8/5}$ = " + f"{dt:.0f} s", fontsize=6.2,
                  color=col, ha="left", va="center", zorder=8, halo="white",
                  halo_lw=2.4)

    for tx, ty, txt in [(300.0, 760.0, "F + P"), (120.0, 470.0, "B"),
                        (0.30, 315.0, "M")]:
        halo_text(ax, tx, ty, txt, fontsize=7.4, color=SUB, ha="center",
                  va="center", fontweight="bold", halo="white", halo_lw=2.4,
                  zorder=8)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1000.0)
    ax.set_ylim(200.0, 880.0)
    ax.set_xlabel("time (s)", fontsize=7.4)
    ax.set_ylabel(r"temperature ($^{\circ}$C)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Arc energy decides the constituent", fontsize=8.2,
                 fontweight="bold", color=INK)
    return products


def main():
    set_pub_style()
    x, y, tp = section_field()
    t = np.logspace(np.log10(0.15), np.log10(400.0), 1400)
    cycles = []
    for r in R_PROBE:
        temp = thermal_cycle(r, t)
        t8, t5, dt = cooling_time_85(t, temp)
        cycles.append((r, t, temp, t8, t5, dt))

    tq = np.logspace(np.log10(0.05), np.log10(3000.0), 2000)
    sweep = []
    for q in Q_SWEEP:
        temp = thermal_cycle(radius_for_peak(q), tq, q)
        t8, t5, dt = cooling_time_85(tq, temp)
        sweep.append((q, tq, temp, t8, t5, dt))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_sec = axes_inches(fig, (0.78, 2.85, 6.00, 2.156))
    ax_hv = axes_inches(fig, (0.78, 1.05, 6.00, 1.55))
    ax_cyc = axes_inches(fig, (7.42, 3.28, 2.48, 1.72))
    ax_cct = axes_inches(fig, (7.42, 1.05, 2.48, 1.72))

    section_panel(ax_sec, x, y, tp)
    hardness_panel(ax_hv, np.linspace(-X_HALF, X_HALF, 1281))
    cycle_panel(ax_cyc, cycles)
    products = cct_panel(ax_cct, sweep)

    hv_peak = hardness(peak_temperature(
        np.linspace(-X_HALF, X_HALF, 1281), -1.0)).max()
    n_mart = sum(p == "martensite" for p in products)
    panel_labels_inches(fig, [ax_sec, ax_hv, ax_cyc, ax_cct], dx=-0.52,
                        dy=0.10)
    savefig(fig, "748_wld_heat_affected_zone.png")


if __name__ == "__main__":
    main()
