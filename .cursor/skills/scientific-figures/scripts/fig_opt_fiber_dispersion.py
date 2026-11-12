"""Chromatic dispersion in single-mode fibre and what it costs a link.

The main panel is the claim: total dispersion is the sum of a material term
fixed by silica and a waveguide term the designer controls, so moving the
zero-dispersion wavelength is a fibre-design decision, not a material one.
Every zero-dispersion wavelength printed on the panel is found by sign change
on the curve that is drawn.  The two smaller panels turn that curve into link
budget: how fast the pulse spreads with distance, and how far the eye stays
open, before and after a compensating span.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

RED = "#b23a48"        # total dispersion / uncompensated link
BLUE = "#3b5b92"       # material term / compensated link
GOLD = "#c9a227"       # waveguide term
PURPLE = "#8e6bbf"     # dispersion-shifted fibre
GREEN = "#2e7d5b"      # non-zero dispersion-shifted fibre
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

LAM = np.linspace(1.20, 1.65, 500)      # wavelength in um
C_BAND = (1.530, 1.565)
D_SOURCE_NM = 0.12                      # source linewidth in nm
T0_PS = 20.0                            # launched pulse FWHM in ps
BITRATE_GBPS = 10.0
L_KM = np.linspace(0.0, 200.0, 400)
D_RESIDUAL = 0.9                        # ps/(nm km) after compensation


def d_material(lam):
    return 122.0 * (1.0 - 1.322 / lam)


def d_waveguide(lam, strength=1.0, offset=0.0):
    return -(1.0 + 4.0 * (lam - 1.25)) * strength - offset


def zero_wavelength(lam, d):
    s = np.sign(d)
    k = np.nonzero(np.diff(s) != 0)[0]
    if k.size == 0:
        return float("nan")
    i = int(k[0])
    return float(lam[i] - d[i] * (lam[i + 1] - lam[i]) / (d[i + 1] - d[i]))


def fibre_totals():
    mat = d_material(LAM)
    wg_smf = d_waveguide(LAM)
    fibres = [
        ("standard SMF", mat + wg_smf, RED, 1.9),
        ("dispersion shifted", mat + d_waveguide(LAM, 8.0, 0.3), PURPLE, 1.2),
        ("non-zero DSF", mat + d_waveguide(LAM, 7.0, 0.5), GREEN, 1.2),
    ]
    return mat, wg_smf, fibres


def main_panel(ax):
    mat, wg, fibres = fibre_totals()
    ax.axhline(0.0, color=GREY, lw=0.8, zorder=2)
    ax.axvspan(*C_BAND, color="#f0f0f0", zorder=1)
    ax.text(np.mean(C_BAND), -29.0, "C band", ha="center", va="bottom",
            fontsize=6.6, color=SUB)

    ax.plot(LAM, mat, color=BLUE, lw=1.3, ls=(0, (5, 2.6)), zorder=3,
            label="material dispersion")
    ax.plot(LAM, wg, color=GOLD, lw=1.3, ls=(0, (5, 2.6)), zorder=3,
            label="waveguide dispersion")

    zdws = {}
    for name, tot, color, lw in fibres:
        ax.plot(LAM, tot, color=color, lw=lw, zorder=4,
                label=f"total, {name}")
        z = zero_wavelength(LAM, tot)
        zdws[name] = z
        ax.plot([z], [0.0], marker="o", ms=4.0, mfc="white", mec=color,
                mew=1.1, zorder=6)

    z_smf = zdws["standard SMF"]
    ax.annotate(r"$\lambda_0$ = " + f"{z_smf * 1e3:.0f} nm",
                xy=(z_smf, 0.0), xytext=(z_smf - 0.012, -17.0), ha="center",
                fontsize=7.0, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.annotate(r"$\lambda_0$ = " + f"{zdws['dispersion shifted'] * 1e3:.0f}"
                " nm", xy=(zdws["dispersion shifted"], 0.0),
                xytext=(zdws["dispersion shifted"] - 0.010, -13.0),
                ha="right", fontsize=7.0, color=PURPLE,
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.7))

    d_c = float(np.interp(1.550, LAM, fibres[0][1]))
    ax.annotate(f"D = {d_c:.1f} " + r"ps nm$^{-1}$ km$^{-1}$" + "\nat 1550 nm",
                xy=(1.550, d_c), xytext=(1.430, 25.0), ha="left",
                fontsize=7.0, color=RED, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))

    ax.set_xlabel(r"wavelength $\lambda$ ($\mathrm{\mu m}$)")
    ax.set_ylabel(r"dispersion parameter $D$ (ps nm$^{-1}$ km$^{-1}$)")
    ax.set_xlim(LAM[0], LAM[-1])
    ax.set_ylim(-30.0, 30.0)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.legend(fontsize=7.0, frameon=False, loc="upper left",
              handlelength=1.7, labelspacing=0.30, borderaxespad=0.3)
    ax.set_title("Waveguide design, not glass chemistry, places the zero "
                 "crossing", fontsize=9.0, fontweight="bold", color=INK)
    return z_smf, d_c


def spread(d_par):
    dt = np.abs(d_par) * D_SOURCE_NM * L_KM
    return np.sqrt(T0_PS ** 2 + dt ** 2)


def broadening_panel(ax, d_c):
    t_bit = 1e3 / BITRATE_GBPS
    unc, comp = spread(d_c), spread(D_RESIDUAL)
    ax.axhline(t_bit, color=GREY, lw=0.9, ls=(0, (4, 2.6)), zorder=2)
    ax.text(L_KM[-1], t_bit * 1.06, f"{BITRATE_GBPS:.0f} Gb/s bit slot "
            f"{t_bit:.0f} ps", ha="right", va="bottom", fontsize=6.4,
            color=SUB)
    ax.plot(L_KM, unc, color=RED, lw=1.6, zorder=4)
    ax.plot(L_KM, comp, color=BLUE, lw=1.6, zorder=4)

    reach = float(np.interp(t_bit, unc, L_KM))
    ax.plot([reach], [t_bit], marker="o", ms=4.0, mfc="white", mec=RED,
            mew=1.1, zorder=5)
    ax.annotate(f"uncompensated\nreach {reach:.0f} km", xy=(reach, t_bit),
                xytext=(reach + 14.0, t_bit * 2.6), ha="left", fontsize=6.6,
                color=RED, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.text(L_KM[-1] * 0.98, comp[-1] * 0.72,
            f"after compensation, D = {D_RESIDUAL:.1f}", ha="right",
            va="top", fontsize=6.6, color=BLUE)

    ax.set_xlabel("fibre length $L$ (km)", fontsize=7.4)
    ax.set_ylabel("output width (ps)", fontsize=7.4)
    ax.set_xlim(0.0, L_KM[-1])
    ax.set_yscale("log")
    ax.set_ylim(10.0, 900.0)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_title("Pulse spreading against distance", fontsize=8.2,
                 fontweight="bold", color=INK)
    return t_bit


def eye_panel(ax, d_c, t_bit):
    def opening(d_par):
        dt = np.abs(d_par) * D_SOURCE_NM * L_KM
        return 1.0 / (1.0 + (dt / (0.60 * t_bit)) ** 2)

    unc, comp = opening(d_c), opening(D_RESIDUAL)
    ax.fill_between(L_KM, 0.0, unc, color=RED, alpha=0.14, linewidth=0.0)
    ax.plot(L_KM, unc, color=RED, lw=1.6, zorder=4)
    ax.plot(L_KM, comp, color=BLUE, lw=1.6, zorder=4)
    ax.axhline(0.5, color=GREY, lw=0.9, ls=(0, (4, 2.6)), zorder=2)

    l50 = float(np.interp(-0.5, -unc, L_KM))
    ax.plot([l50], [0.5], marker="o", ms=4.0, mfc="white", mec=RED, mew=1.1,
            zorder=5)
    ax.text(l50 + 5.0, 0.56, f"half closed at {l50:.0f} km", ha="left",
            va="bottom", fontsize=6.6, color=RED)
    ax.text(L_KM[-1] * 0.98, comp[-1] - 0.06,
            f"{comp[-1] * 100:.0f} % open at {L_KM[-1]:.0f} km", ha="right",
            va="top", fontsize=6.6, color=BLUE)
    ax.text(2.0, 0.52, "50 % criterion", ha="left", va="bottom",
            fontsize=6.4, color=SUB)

    ax.set_xlabel("fibre length $L$ (km)", fontsize=7.4)
    ax.set_ylabel("eye opening (-)", fontsize=7.4)
    ax.set_xlim(0.0, L_KM[-1])
    ax.set_ylim(0.0, 1.08)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_title("Eye opening before and after compensation", fontsize=8.2,
                 fontweight="bold", color=INK)
    return l50


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.32, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_broad = fig.add_subplot(gs[0, 1])
    ax_eye = fig.add_subplot(gs[1, 1])

    z_smf, d_c = main_panel(ax_main)
    t_bit = broadening_panel(ax_broad, d_c)
    l50 = eye_panel(ax_eye, d_c, t_bit)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_broad, ax_eye], dx=-0.46, dy=0.12)
    savefig(fig, "699_opt_fiber_dispersion.png")


if __name__ == "__main__":
    main()
