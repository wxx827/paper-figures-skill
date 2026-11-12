"""Torsional behaviour of a strain-wave joint: loops, slope, and one spectrum.

The main panel is a torque-windup test run at three amplitudes.  A single
torque-dependent stiffness law is integrated to give the backbone curve and a
reversal term opens it into a loop, so the three loops are nested by
construction rather than drawn separately; the enclosed area of each is the
shoelace area of the polygon actually plotted, converted to millijoules, and
the lost motion is read off the same arrays where they cross zero torque.  The
upper side panel differentiates the largest loading branch to expose the three
stiffness regions the loop only implies, and the lower one leaves the quasi
static world entirely: the amplitude spectrum of the transmission error, in
which the second harmonic of the wave generator dwarfs the once-per-revolution
eccentricity term.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3d6b8e"
STEEL = "#6b7f95"
GREEN = "#2e7d5b"
GOLD = "#c9a227"

SEED = 1064
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)

K1, K2, K3 = 1.55, 2.65, 3.60   # stiffness regions, N.m per arcmin
T_K1, T_K2 = 26.0, 74.0         # region boundaries, N.m
W_K1, W_K2 = 7.0, 12.0          # blending widths, N.m
LOST = 0.62                     # lost motion at zero torque, arcmin
T_REV = 16.0                    # reversal width of the friction term, N.m
AMPS = (45.0, 85.0, 125.0)      # test amplitudes, N.m
RATIO = 100                     # gear ratio of the strain-wave unit
ARCMIN_RAD = 2.908882e-4        # rad per arcmin

BOX = dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="none",
           alpha=0.88)


def backbone(torque):
    """Deflection in arcmin from a torque-dependent stiffness, integrated."""
    a = np.abs(torque)
    k = (K1 + (K2 - K1) / (1.0 + np.exp(-(a - T_K1) / W_K1))
         + (K3 - K2) / (1.0 + np.exp(-(a - T_K2) / W_K2)))
    theta = np.concatenate(([0.0], np.cumsum(
        0.5 * (1.0 / k[1:] + 1.0 / k[:-1]) * np.diff(torque))))
    return theta - np.interp(0.0, torque, theta), k


def loop(amp, n=900):
    """One closed torque-deflection loop at the requested amplitude."""
    up = np.linspace(-amp, amp, n)
    down = up[::-1]
    th_up, k_up = backbone(up)
    th_dn, _k = backbone(down)
    th_up = th_up - 0.5 * LOST * np.tanh((up + amp) / T_REV)
    th_dn = th_dn + 0.5 * LOST * np.tanh((amp - down) / T_REV)
    return up, th_up, down, th_dn, k_up


def shoelace(x, y):
    """Enclosed area of the closed polygon, in the plotted units."""
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def transmission_error(rng, n=8192, revs=4.0):
    """Output-side transmission error over several revolutions, in arcsec."""
    ang = np.linspace(0.0, 360.0 * revs, n, endpoint=False)
    rad = np.deg2rad(ang)
    te = (11.5 * np.sin(rad + 0.7)                    # eccentricity, order 1
          + 5.2 * np.sin(2.0 * rad + 1.9)             # ovality, order 2
          + 3.1 * np.sin(4.0 * rad - 0.4)             # order 4
          + 24.0 * np.sin(2.0 * RATIO * rad + 0.3)    # 2 per wave generator
          + 6.0 * np.sin(4.0 * RATIO * rad - 1.1)     # 4 per wave generator
          + rng.normal(0.0, 1.1, n))
    spec = np.abs(np.fft.rfft(te)) * 2.0 / n
    order = np.fft.rfftfreq(n, d=1.0 / n) / revs
    return ang, te, order, spec


def main_panel(ax, loops, areas, lost_meas):
    cols = (STEEL, BLUE, ORANGE)
    for (amp, (up, th_up, down, th_dn, _k)), col in zip(loops.items(), cols):
        ax.plot(up, th_up, color=col, lw=1.5, zorder=6)
        ax.plot(down, th_dn, color=col, lw=1.5, zorder=6)
        if amp == max(loops):
            ax.fill(np.concatenate([up, down]),
                    np.concatenate([th_up, th_dn]), color=ORANGE, alpha=0.13,
                    lw=0, zorder=3)
            i = int(0.72 * up.size)
            ax.annotate("", xy=(up[i + 30], th_up[i + 30]),
                        xytext=(up[i], th_up[i]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2),
                        zorder=7)
            j = int(0.72 * down.size)
            ax.annotate("", xy=(down[j + 30], th_dn[j + 30]),
                        xytext=(down[j], th_dn[j]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2),
                        zorder=7)

    amp_max = max(loops)
    up, th_up, down, th_dn, _k = loops[amp_max]
    ax.axhline(0.0, color="#d5d9de", lw=0.7, zorder=1)
    ax.axvline(0.0, color="#d5d9de", lw=0.7, zorder=1)

    lo = np.interp(0.0, up, th_up)
    hi = np.interp(0.0, down[::-1], th_dn[::-1])
    ax.plot([0.0, 0.0], [lo, hi], color=RED, lw=2.2, zorder=9,
            solid_capstyle="butt")
    ax.annotate("lost motion at zero torque\n{:.2f} arcmin  "
                "({:.0f} $\\mu$rad at the output)"
                .format(lost_meas, lost_meas * ARCMIN_RAD * 1e6),
                xy=(0.0, 0.5 * (lo + hi)), xytext=(0.055, 0.905),
                textcoords="axes fraction", ha="left", va="center",
                fontsize=6.5, color=RED, zorder=11, linespacing=1.34,
                bbox=BOX, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                          shrinkA=2, shrinkB=3))

    for (amp, col, area) in zip(loops, cols, areas):
        u, t_u, _d, _t_d, _k = loops[amp]
        ax.text(amp + 2.5, t_u[-1], "$\\pm${:.0f} N$\\cdot$m\n{:.1f} mJ per "
                "cycle".format(amp, area), ha="left", va="center",
                fontsize=6.2, color=col, zorder=10, linespacing=1.32,
                bbox=BOX)

    ax.set_xlim(-165.0, 176.0)
    ax.set_ylim(-42.0, 42.0)
    ax.set_xlabel("torque applied at the output flange (N$\\cdot$m)")
    ax.set_ylabel("torsional deflection (arcmin)")
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color=ORANGE, lw=1.5, label="loading / unloading path"),
        Line2D([], [], color=RED, lw=2.2, label="lost motion"),
        Line2D([], [], color=ORANGE, lw=6, alpha=0.25,
               label="dissipated per cycle")],
        loc="lower right", fontsize=6.2, frameon=True, facecolor="white",
        edgecolor="#dddddd", framealpha=0.95, handlelength=1.8,
        borderaxespad=0.4, labelspacing=0.26)
    ax.set_title("Three windup cycles on the same joint, nested by "
                 "construction", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def stiffness_panel(ax, loops):
    amp = max(loops)
    up, th_up, _d, _t, _k = loops[amp]
    keep = up > -amp + 3.0 * T_REV
    t_loc = up[keep]
    k_loc = np.gradient(t_loc, th_up[keep])
    ax.plot(t_loc, k_loc, color=BLUE, lw=1.6, zorder=6)
    for lvl, name, col in ((K1, "$K_1$", STEEL), (K2, "$K_2$", GREEN),
                           (K3, "$K_3$", GOLD)):
        ax.axhline(lvl, color=col, lw=0.8, ls=(0, (3.2, 2.2)), zorder=4)
        ax.text(t_loc[-1], lvl + 0.06, "{} = {:.2f}".format(name, lvl),
                ha="right", va="bottom", fontsize=6.1, color=col, zorder=8,
                bbox=BOX)
    for edge in (T_K1, T_K2):
        ax.axvline(edge, color="#c9ced4", lw=0.7, zorder=2)
    ax.set_xlim(t_loc.min(), t_loc.max())
    ax.set_ylim(1.2, 4.1)
    ax.set_xlabel("torque (N$\\cdot$m)", fontsize=6.8)
    ax.set_ylabel("local stiffness\n(N$\\cdot$m per arcmin)", fontsize=6.8,
                  linespacing=1.3)
    ax.tick_params(labelsize=6.6)
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Slope of the loading branch, differentiated", fontsize=8.0,
                 fontweight="bold", color=INK)


def spectrum_panel(ax, order, spec, te):
    keep = (order >= 0.5) & (order <= 600.0)
    o, s = order[keep], spec[keep]
    ax.plot(o, s, color=STEEL, lw=0.7, zorder=4)
    peaks = [(1.0, "1 x output\n(eccentricity)", BLUE, 1.0, 12.0),
             (2.0, "2 x output", GREEN, 1.0, -2.0),
             (2.0 * RATIO, "2 x wave generator", ORANGE, -1.0, 2.0)]
    for ordv, name, col, sgn, dy in peaks:
        i = int(np.argmin(np.abs(o - ordv)))
        ax.plot([o[i], o[i]], [0.0, s[i]], color=col, lw=1.6, zorder=6)
        ax.plot(o[i], s[i], marker="o", ms=3.4, color=col, zorder=7)
        ax.text(o[i] * (1.55 if sgn > 0 else 0.62), s[i] + dy,
                "{}\n{:.1f} arcsec".format(name, s[i]),
                ha="left" if sgn > 0 else "right", va="bottom", fontsize=6.1,
                color=col, zorder=9, linespacing=1.3, bbox=BOX)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 600.0)
    ax.set_ylim(0.0, 31.0)
    ax.set_xlabel("order (cycles per output revolution)", fontsize=6.8)
    ax.set_ylabel("transmission error\namplitude (arcsec)", fontsize=6.8,
                  linespacing=1.3)
    ax.tick_params(labelsize=6.6)
    ax.text(0.60, 29.0, "RMS transmission error {:.1f} arcsec"
            .format(float(np.std(te))), ha="left", va="top", fontsize=6.2,
            color=INK, zorder=9, bbox=BOX)
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Where the error sits in the order domain", fontsize=8.0,
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
    rng = np.random.default_rng(SEED)

    loops = {amp: loop(amp) for amp in AMPS}
    areas = []
    for amp in AMPS:
        up, th_up, down, th_dn, _k = loops[amp]
        x = np.concatenate([up, down])
        y = np.concatenate([th_up, th_dn])
        areas.append(shoelace(x, y) * ARCMIN_RAD * 1000.0)   # mJ per cycle
    assert areas[0] < areas[-1], "a bigger cycle must dissipate more"

    up, th_up, down, th_dn, _k = loops[max(AMPS)]
    lost_meas = float(np.interp(0.0, down[::-1], th_dn[::-1])
                      - np.interp(0.0, up, th_up))

    ang, te, order, spec = transmission_error(rng)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_k = axes_inches(fig, AUX_T)
    ax_s = axes_inches(fig, AUX_B)

    main_panel(ax_main, loops, areas, lost_meas)
    stiffness_panel(ax_k, loops)
    spectrum_panel(ax_s, order, spec, te)



    panel_labels_inches(fig, [ax_main, ax_k, ax_s], dx=-0.66, dy=0.14)
    savefig(fig, "1064_ric_harmonic_hysteresis.png")


if __name__ == "__main__":
    main()
