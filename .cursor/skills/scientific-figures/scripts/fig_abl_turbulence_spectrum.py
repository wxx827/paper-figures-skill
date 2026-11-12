"""Turbulence spectra of the along-wind component at four measurement heights.

The main panel is a log-log velocity spectrum with the three classical
ranges marked: the energy-containing range where the boundary layer stores
its variance, the inertial subrange that follows the minus five thirds law,
and the dissipation range where viscosity finally removes the energy.  A
free power-law fit over the inertial band supplies both the slope printed on
the panel and the dissipation rate derived from its level, so neither number
is quoted from elsewhere.

The companion panels come from the same spectra: turbulence intensity is the
square root of the integrated spectrum divided by the mean speed, and the
Davenport coherence shows how quickly two heights stop seeing the same eddy.
"""
import textwrap

import numpy as np

import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_palette
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
GRID = "#d9d9d9"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"

USTAR = 0.45          # friction velocity (m/s)
U_REF, Z_REF = 9.0, 100.0
ALPHA = 0.14          # shear exponent used for the mean profile (-)
F_ETA = 8.0           # start of the viscous roll-off (Hz)
ALPHA_U = 0.26        # one-sided Kolmogorov constant for the u component (-)
DECAY = 12.0          # Davenport coherence decay constant (-)
HEIGHTS = [20.0, 50.0, 100.0, 200.0]
SEPS = [20.0, 40.0, 80.0]
F_LO, F_HI = 1e-3, 3e1
INERTIAL = (0.05, 3.0)
ENERGY_TO = 0.05


def u_mean(z):
    return U_REF * (z / Z_REF) ** ALPHA


def kaimal(f, z):
    """Neutral Kaimal spectrum of the u component with a viscous roll-off."""
    u = u_mean(z)
    n = f * z / u
    s = USTAR ** 2 * 105.0 * (z / u) / (1.0 + 33.0 * n) ** (5.0 / 3.0)
    return s * np.exp(-(f / F_ETA) ** (4.0 / 3.0))


def coherence(f, sep, u=U_REF):
    return np.exp(-DECAY * f * sep / u)


def main():
    set_pub_style()
    rng = np.random.default_rng(719)

    f = np.logspace(np.log10(F_LO), np.log10(F_HI), 520)
    cols = get_palette("cb_sequential_blue", 7)[2:6]

    # band-averaged spectral estimates scatter about the model
    spec = {}
    for zz in HEIGHTS:
        jitter = np.exp(rng.normal(0.0, 0.13, f.size))
        spec[zz] = kaimal(f, zz) * jitter

    # free power-law fit over the inertial band of the reference height
    band = (f >= INERTIAL[0]) & (f <= INERTIAL[1])
    slope, icpt = np.polyfit(np.log10(f[band]), np.log10(spec[Z_REF][band]), 1)
    level = 10.0 ** icpt                       # spectral density at 1 Hz
    eps = (level / (ALPHA_U * (U_REF / (2.0 * np.pi)) ** (2.0 / 3.0))) ** 1.5

    # peak of the variance-preserving spectrum, read off the plotted curve
    fs = f * spec[Z_REF]
    f_peak = float(f[int(np.argmax(fs))])

    # turbulence intensity by integrating each spectrum
    zg = np.linspace(10.0, 220.0, 120)
    ti_curve = np.array([100.0 * np.sqrt(np.trapezoid(kaimal(f, zz), f))
                         / u_mean(zz) for zz in zg])
    ti_pts = np.array([100.0 * np.sqrt(np.trapezoid(spec[zz], f)) / u_mean(zz)
                       for zz in HEIGHTS])

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.88, 0.82, 5.02, 4.68))
    ax1 = axes_inches(fig, (6.72, 3.30, 3.00, 2.20))
    ax2 = axes_inches(fig, (6.72, 0.82, 3.00, 1.95))

    # ---- main panel: the spectra ------------------------------------------
    slo, shi = 2e-6, 2e3
    bands = [(F_LO, ENERGY_TO, "#f0f0f0", "energy-containing\nrange"),
             (ENERGY_TO, INERTIAL[1], "#ffffff", "inertial subrange"),
             (INERTIAL[1], F_HI, "#f0f0f0", "dissipation\nrange")]
    for x0, x1, fc, lab in bands:
        if fc != "#ffffff":
            ax.axvspan(x0, x1, facecolor=fc, edgecolor="none", zorder=0)
        ax.text(np.sqrt(x0 * x1), 0.975, lab, fontsize=6.8, color=SUB,
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                linespacing=1.25, zorder=3)
    for xv in (ENERGY_TO, INERTIAL[1]):
        ax.axvline(xv, color=GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)

    for zz, c in zip(HEIGHTS, cols):
        ax.plot(f, spec[zz], color=c, lw=1.35, zorder=4,
                label=f"$z$ = {zz:.0f} m, $U$ = {u_mean(zz):.1f} m/s")

    ref = 9.0 * level * f ** (-5.0 / 3.0)
    m = (f >= 0.02) & (f <= 6.0)
    ax.plot(f[m], ref[m], color=GREY, lw=1.1, ls=(0, (5, 2)), zorder=5)
    ax.text(0.30, 9.0 * level * 0.30 ** (-5.0 / 3.0) * 1.9, "$f^{-5/3}$",
            fontsize=8.0, color=GREY, ha="center", va="bottom")

    ax.plot([f_peak], [spec[Z_REF][int(np.argmax(fs))] * 2.6], marker="v",
            ms=5.0, color=RED, zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(F_LO, F_HI)
    ax.set_ylim(slo, shi)
    ax.set_xlabel("Frequency $f$ (Hz)", fontsize=8.0)
    ax.set_ylabel("Spectral density $S_u$ ((m/s)$^2$/Hz)", fontsize=8.0)
    ax.set_title("Along-wind velocity spectra at four heights", fontsize=9.0,
                 fontweight="bold")
    ax.tick_params(labelsize=7.2)
    ax.grid(True, which="major", color=GRID, lw=0.55, alpha=0.8)
    ax.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=7.0)

    ax.text(0.975, 0.815,
            f"fit over {INERTIAL[0]:g}-{INERTIAL[1]:g} Hz at $z$ = "
            f"{Z_REF:.0f} m\nslope = {slope:.2f}, "
            f"$\\varepsilon$ = {eps:.4f} m$^2$/s$^3$\n"
            f"red marker: peak of $f\\,S_u$ at {f_peak * 1e3:.1f} mHz",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=INK, linespacing=1.5, zorder=8,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))

    # ---- aux 1: turbulence intensity with height ---------------------------
    ax1.plot(ti_curve, zg, color=INK, lw=1.4, zorder=3)
    for zz, c, ti in zip(HEIGHTS, cols, ti_pts):
        ax1.plot([ti], [zz], marker="o", ms=4.4, mfc=c, mec=INK, mew=0.7,
                 zorder=5)
        ax1.text(ti + 0.22, zz, f"{ti:.1f} %", fontsize=6.4, color=INK,
                 ha="left", va="center")
    ax1.set_xlim(float(ti_curve.min()) - 1.2, float(ti_curve.max()) + 1.6)
    ax1.set_ylim(0.0, 230.0)
    ax1.set_xlabel("Turbulence intensity $I_u$ (%)", fontsize=7.2)
    ax1.set_ylabel("Height $z$ (m)", fontsize=7.2)
    ax1.set_title("Intensity falls with height", fontsize=8.5,
                  fontweight="bold")
    ax1.tick_params(labelsize=6.5)
    ax1.grid(True, color=GRID, lw=0.5, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.text(0.975, 0.95, "markers: integrated\nspectra of panel (a)",
             transform=ax1.transAxes, ha="right", va="top", fontsize=6.4,
             color=SUB, linespacing=1.3)

    # ---- aux 2: vertical coherence ----------------------------------------
    ramp = get_palette("cb_sequential_blue", 6)[2:5][::-1]
    ax2.axhline(0.5, color=GREY, lw=0.8, ls=(0, (4, 2)), zorder=2)
    for sep, c in zip(SEPS, ramp):
        ax2.plot(f, coherence(f, sep), color=c, lw=1.4, zorder=4,
                 label=f"{sep:.0f} m")
        f_half = np.log(2.0) * U_REF / (DECAY * sep)
        ax2.plot([f_half], [0.5], marker="o", ms=3.4, mfc="white", mec=c,
                 mew=0.9, zorder=5)
        ax2.text(f_half, 0.44, f"{f_half * 1e3:.0f}", fontsize=6.2, color=c,
                 ha="center", va="top")
    ax2.text(F_LO * 1.4, 0.05, "half-coherence frequency labelled in mHz",
             fontsize=6.4, color=SUB, ha="left", va="bottom")
    ax2.set_xscale("log")
    ax2.set_xlim(F_LO, F_HI)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("Frequency $f$ (Hz)", fontsize=7.2)
    ax2.set_ylabel("Coherence $\\gamma^2$ (-)", fontsize=7.2)
    ax2.set_title("Vertical coherence between two levels", fontsize=8.5,
                  fontweight="bold")
    ax2.tick_params(labelsize=6.5)
    ax2.grid(True, which="major", color=GRID, lw=0.5, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", frameon=False, fontsize=6.5, ncol=3,
               handlelength=1.2, columnspacing=0.9, handletextpad=0.4,
               title="separation $\\Delta z$")
    ax2.get_legend().get_title().set_fontsize(6.5)

    panel_labels_inches(fig, [ax, ax1, ax2], dx=-0.50, dy=0.10)


    savefig(fig, "719_abl_turbulence_spectrum.png")


if __name__ == "__main__":
    main()
