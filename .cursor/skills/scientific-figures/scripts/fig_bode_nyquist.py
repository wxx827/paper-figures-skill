"""Frequency-response pair for an open-loop control system.

Left column: Bode diagram - magnitude in dB over a logarithmic frequency
axis (a) above the phase in degrees (b) - with the gain crossover, the phase
crossover and the resulting gain and phase margins constructed graphically.

Right panel (c): the matching Nyquist locus with the unit circle, the
critical -1 point, the low-frequency asymptote, arrows for increasing
frequency and the same two margins read off the plot.

Physics: one single plant, the type-1 loop transfer function
L(s) = K / [s (0.5 s + 1)(0.1 s + 1)], is evaluated once and shared by all
three panels, so magnitude, phase and locus are mutually consistent; the
crossover frequencies are found by root-finding on that same function.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from scipy.optimize import brentq
from pubstyle import set_pub_style, savefig, add_panel_label

K = 4.0
T1, T2 = 0.5, 0.1
BLUE, RED, GOLD, GREY = "#1f5c8b", "#b23a48", "#d18f16", "#8f979e"


def loop(w):
    s = 1j * np.asarray(w, dtype=float)
    return K / (s * (T1 * s + 1.0) * (T2 * s + 1.0))


def mag_db(w):
    return 20.0 * np.log10(np.abs(loop(w)))


def phase_deg(w):
    return -90.0 - np.degrees(np.arctan(T1 * w)) \
        - np.degrees(np.arctan(T2 * w))


def main():
    set_pub_style(base_font=8)
    w = np.logspace(-1, 2.3, 1400)
    wc = brentq(mag_db, 0.05, 100.0)                 # gain crossover
    wp = brentq(lambda x: phase_deg(x) + 180.0, 0.5, 200.0)   # phase xover
    pm = 180.0 + phase_deg(wc)
    gm_db = -mag_db(wp)
    gm_lin = 10 ** (gm_db / 20.0)

    fig = plt.figure(figsize=(7.2, 4.15))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1.0],
                          height_ratios=[1.0, 0.88], left=0.075, right=0.985,
                          top=0.935, bottom=0.115, wspace=0.30, hspace=0.16)
    axm = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[1, 0], sharex=axm)
    axn = fig.add_subplot(gs[:, 1])

    # ---- (a) magnitude ---------------------------------------------------
    axm.semilogx(w, mag_db(w), color=BLUE, lw=1.7, zorder=4)
    axm.axhline(0, color="#9aa3ad", lw=0.8, ls=(0, (4, 2)), zorder=2)
    axm.axvline(wc, color=GOLD, lw=0.9, ls=(0, (2, 2)), zorder=2)
    axm.axvline(wp, color=RED, lw=0.9, ls=(0, (2, 2)), zorder=2)
    axm.annotate("", xy=(wp, 0.0), xytext=(wp, -gm_db),
                 arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color=RED,
                                 mutation_scale=7), zorder=6)
    axm.text(wp * 1.15, -gm_db / 2.0, "GM = %.1f dB" % gm_db, color=RED,
             fontsize=6.8, fontweight="bold", ha="left", va="center")
    axm.text(wc * 0.9, 32, r"$\omega_c$ = %.2f rad s$^{-1}$" % wc,
             color="#8a5c07", fontsize=6.5, ha="right", va="center")
    axm.plot([wc], [0.0], marker="o", ms=4.0, mfc=GOLD, mec="#12161b",
             mew=0.7, zorder=7)
    axm.set_ylim(-62, 44)
    axm.set_yticks([-60, -40, -20, 0, 20, 40])
    axm.set_ylabel("magnitude (dB)")
    axm.grid(True, which="both", lw=0.45, color="#e8e8e8")
    axm.tick_params(labelbottom=False, length=2.4)
    add_panel_label(axm, "(a)", x=-0.105, y=1.01, size=9.5)
    axm.set_title(r"$L(s) = 4\,/\,[\,s\,(0.5s+1)(0.1s+1)\,]$",
                  fontsize=7.6, color="#33404d", pad=3)

    # ---- (b) phase -------------------------------------------------------
    axp.semilogx(w, phase_deg(w), color=BLUE, lw=1.7, zorder=4)
    axp.axhline(-180, color="#9aa3ad", lw=0.8, ls=(0, (4, 2)), zorder=2)
    axp.axvline(wc, color=GOLD, lw=0.9, ls=(0, (2, 2)), zorder=2)
    axp.axvline(wp, color=RED, lw=0.9, ls=(0, (2, 2)), zorder=2)
    axp.annotate("", xy=(wc, -180.0), xytext=(wc, phase_deg(wc)),
                 arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color=GOLD,
                                 mutation_scale=7), zorder=6)
    axp.text(wc * 0.80, -168, r"PM = %.1f$^\circ$" % pm, color="#8a5c07",
             fontsize=6.8, fontweight="bold", ha="right", va="center")
    axp.plot([wc], [phase_deg(wc)], marker="o", ms=4.0, mfc=GOLD,
             mec="#12161b", mew=0.7, zorder=7)
    axp.plot([wp], [-180.0], marker="o", ms=4.0, mfc=RED, mec="#12161b",
             mew=0.7, zorder=7)
    axp.text(wp * 1.15, -195, r"$\omega_{180}$ = %.2f rad s$^{-1}$" % wp,
             color=RED, fontsize=6.5, ha="left", va="center")
    axp.set_ylim(-268, -78)
    axp.set_yticks([-270, -225, -180, -135, -90])
    axp.set_ylabel(r"phase ($^\circ$)")
    axp.set_xlabel(r"frequency  $\omega$  (rad s$^{-1}$)")
    axp.grid(True, which="both", lw=0.45, color="#e8e8e8")
    axp.tick_params(length=2.4)
    add_panel_label(axp, "(b)", x=-0.105, y=1.015, size=9.5)

    # ---- (c) Nyquist -----------------------------------------------------
    wn = np.logspace(-0.72, 2.6, 4000)
    g = loop(wn)
    axn.plot(g.real, g.imag, color=BLUE, lw=1.7, zorder=5,
             label=r"$L(j\omega)$,  $\omega > 0$")
    axn.plot(g.real, -g.imag, color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=3,
             label=r"mirror,  $\omega < 0$")
    axn.add_patch(Circle((0, 0), 1.0, fill=False, ec="#b9c0c6", lw=0.8,
                         ls=(0, (3, 2)), zorder=2))
    axn.axhline(0, color="#d4d4d4", lw=0.7, zorder=1)
    axn.axvline(0, color="#d4d4d4", lw=0.7, zorder=1)

    for target in (1.75, 0.95, 0.42, 0.14):
        i = int(np.argmin(np.abs(np.abs(g) - target)))
        axn.annotate("", xy=(g.real[i + 8], g.imag[i + 8]),
                     xytext=(g.real[i], g.imag[i]), zorder=7,
                     arrowprops=dict(arrowstyle="-|>", lw=0.7, color=BLUE,
                                     mutation_scale=9))
    axn.text(-1.62, -1.62, r"increasing $\omega$", fontsize=6.3,
             color=BLUE, ha="left", va="center", rotation=52)

    axn.plot([-1], [0], marker="x", ms=7.0, color=RED, mew=1.8, zorder=8)
    axn.text(-1.0, 0.13, "$-1$", color=RED, fontsize=7.2, fontweight="bold",
             ha="center", va="bottom")
    axn.plot([-1.0 / gm_lin], [0.0], marker="o", ms=4.4, mfc=RED,
             mec="#12161b", mew=0.7, zorder=8)
    axn.annotate(r"$1/\mathrm{GM}$ = %.2f" % (1.0 / gm_lin),
                 xy=(-1.0 / gm_lin, 0.0), xytext=(0.10, -1.02),
                 fontsize=6.5, color=RED, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=RED,
                                 shrinkA=1, shrinkB=4))
    gc = loop(wc)
    axn.plot([gc.real], [gc.imag], marker="o", ms=4.4, mfc=GOLD,
             mec="#12161b", mew=0.7, zorder=8)
    axn.plot([0, gc.real], [0, gc.imag], color=GOLD, lw=0.8, zorder=4)
    axn.add_patch(Arc((0, 0), 1.24, 1.24, theta1=180.0,
                      theta2=180.0 + pm, color=GOLD, lw=1.1, zorder=6))
    axn.annotate(r"PM = %.1f$^\circ$" % pm, xy=(-0.62, -0.28),
                 xytext=(-2.10, -0.98), fontsize=6.8, fontweight="bold",
                 color="#8a5c07", ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#8a5c07",
                                 shrinkA=1, shrinkB=3))
    axn.text(-2.12, 1.02, "stable closed loop:\nno encirclement of $-1$",
             fontsize=6.4, color="#33404d", ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#c9c9c9",
                       lw=0.5, alpha=0.94))

    axn.set_xlim(-2.25, 0.60)
    axn.set_ylim(-2.05, 1.15)
    axn.set_aspect("equal")
    axn.set_xlabel(r"Re $\{L(j\omega)\}$")
    axn.set_ylabel(r"Im $\{L(j\omega)\}$")
    axn.grid(True, lw=0.45, color="#f0f0f0")
    lg = axn.legend(loc="upper right", fontsize=6.2, handlelength=1.8,
                    borderpad=0.3, labelspacing=0.28, frameon=True,
                    framealpha=0.92)
    lg.get_frame().set_linewidth(0.5)
    for s in axn.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)
    axn.tick_params(length=2.4)
    add_panel_label(axn, "(c)", x=-0.14, y=1.005, size=9.5)

    savefig(fig, "99_bode_nyquist.png")


if __name__ == "__main__":
    main()
