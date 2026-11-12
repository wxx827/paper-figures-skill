"""Neutron flux spectra of three reactor concepts over ten decades of energy.

The main panel is the claim: the same total flux redistributed by moderation,
plotted per unit lethargy on a log-log grid so that the thermal peak, the 1/E
slowing-down plateau and the fission source are all visible at once, with the
three energy regions shaded behind the curves.  The thermal fractions quoted
in the legend are integrated from the curves that are drawn.  The two smaller
panels supply the consequence: the resonance structure the slowing-down region
has to pass through, and the one-group cross sections each spectrum produces
when the plotted microscopic data are folded with the plotted flux.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
BLUE = "#3b5b92"        # thermal PWR
RED = "#b23a48"         # fast reactor, the emphasis case
GREEN = "#2e7d5b"       # graphite-moderated
GOLD = "#c9a227"        # U-235 fission cross section
PURPLE = "#8e6bbf"      # U-238 capture cross section
BAND_A = "#f0f0f0"
BAND_B = "#f5f5f5"

E_MIN, E_MAX = 1.0e-3, 2.0e7          # eV
E_THERMAL = 0.625                     # thermal cut-off, eV
E_FAST = 1.0e5                        # fast onset, eV
TOTAL_FLUX = 1.0e14                   # n/(cm^2 s), common normalisation
KT = 0.0253                           # eV, 20 C Maxwellian temperature

# U-238 capture resonances: (energy eV, peak barn, width eV)
RESONANCES = [(6.67, 9300.0, 0.030), (20.9, 3400.0, 0.045),
              (36.7, 2600.0, 0.060), (66.0, 1900.0, 0.075),
              (102.6, 1100.0, 0.095), (192.0, 700.0, 0.150),
              (461.0, 420.0, 0.30), (1140.0, 260.0, 0.60)]


def energy_grid(n=3000):
    return np.logspace(np.log10(E_MIN), np.log10(E_MAX), n)


def maxwellian(e, kt):
    m = (e / kt ** 2) * np.exp(-e / kt)
    return e * m / (e * m).max()


def watt(e):
    x = e / 1.0e6
    w = np.sqrt(x) * np.exp(-x / 1.29)
    return e * w / (e * w).max()


def plateau(e, lo, hi):
    return 1.0 / ((1.0 + (lo / e) ** 2) * (1.0 + (e / hi) ** 1.6))


def spectra(e):
    """Per-lethargy flux for three concepts, all at the same total flux."""
    out = {
        "PWR, light water": 1.00 * maxwellian(e, KT)
        + 0.30 * plateau(e, 8.0 * KT, 3.0e5) + 0.55 * watt(e),
        "graphite moderated": 1.35 * maxwellian(e, 1.18 * KT)
        + 0.16 * plateau(e, 9.0 * KT, 2.0e5) + 0.40 * watt(e),
        "sodium-cooled fast": 0.55 * plateau(e, 3.0e3, 4.0e5) + 1.00 * watt(e),
    }
    leth = np.log(e)
    for k, v in out.items():
        out[k] = v * TOTAL_FLUX / np.trapezoid(v, leth)
    return out


def cross_sections(e):
    """1/v backbone plus resolved resonances, in barn."""
    u235 = 580.0 * np.sqrt(0.0253 / e)
    for e0, pk, wd in [(0.29, 190.0, 0.06), (1.14, 90.0, 0.10),
                       (2.03, 70.0, 0.12), (8.8, 45.0, 0.30)]:
        u235 += pk / (1.0 + ((e - e0) / (wd * np.sqrt(e0))) ** 2)
    u235 += 1.30 / (1.0 + (3.0e4 / e) ** 1.1)          # smooth fast tail

    u238 = 2.7 * np.sqrt(0.0253 / e)
    for e0, pk, wd in RESONANCES:
        u238 += pk / (1.0 + ((e - e0) / (wd * np.sqrt(e0))) ** 2)
    u238 += 0.12 / (1.0 + (2.0e4 / e) ** 1.1)
    return {"U-235 fission": u235, "U-238 capture": u238}


def one_group(e, flux, sig):
    leth = np.log(e)
    return np.trapezoid(sig * flux, leth) / np.trapezoid(flux, leth)


def spectrum_panel(ax, e, flux):
    for (lo, hi), fc, name in [
            ((E_MIN, E_THERMAL), BAND_A, "thermal"),
            ((E_THERMAL, E_FAST), BAND_B, "resonance / slowing down"),
            ((E_FAST, E_MAX), BAND_A, "fast")]:
        ax.axvspan(lo, hi, facecolor=fc, edgecolor="none", zorder=0)
        ax.text(np.sqrt(lo * hi), 3.4e14, name, ha="center", va="top",
                fontsize=7.0, color=SUB, zorder=3)
    for x in (E_THERMAL, E_FAST):
        ax.axvline(x, color="#cccccc", lw=0.7, zorder=1)

    leth = np.log(e)
    for (name, phi), color in zip(flux.items(), [BLUE, GREEN, RED]):
        th = np.trapezoid(phi[e <= E_THERMAL], leth[e <= E_THERMAL])
        frac = th / np.trapezoid(phi, leth)
        ax.loglog(e, phi, color=color, lw=1.7, zorder=4,
                  label=f"{name}  ({frac * 100:.0f} % thermal)")

    pwr = flux["PWR, light water"]
    e_pk = float(e[np.argmax(np.where(e < E_THERMAL, pwr, 0.0))])
    ax.annotate(f"Maxwellian peak at {e_pk * 1e3:.0f} meV",
                xy=(e_pk, pwr.max()), xytext=(2.0e-3, 9.0e13), ha="left",
                va="center", fontsize=7.0, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    fast = flux["sodium-cooled fast"]
    e_f = float(e[np.argmax(fast)])
    ax.annotate(f"fission source peak at {e_f / 1e6:.2f} MeV",
                xy=(e_f, fast.max()), xytext=(2.0e3, 3.0e13), ha="left",
                va="center", fontsize=7.0, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))

    ax.set_xlim(E_MIN, E_MAX)
    ax.set_ylim(1.0e10, 6.0e14)
    ax.set_xlabel("neutron energy (eV)")
    ax.set_ylabel(r"flux per unit lethargy  $\Phi$  (n/(cm$^2$$\cdot$s))")
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=7.0, frameon=False, loc="lower left", handlelength=1.8,
              borderaxespad=0.3)
    ax.set_title("Same total flux, three moderation regimes", fontsize=9.0,
                 fontweight="bold", color=INK)


def cross_section_panel(ax, e, sig):
    for (lo, hi), fc in [((E_MIN, E_THERMAL), BAND_A),
                         ((E_THERMAL, E_FAST), BAND_B), ((E_FAST, E_MAX), BAND_A)]:
        ax.axvspan(lo, hi, facecolor=fc, edgecolor="none", zorder=0)
    for (name, s), color in zip(sig.items(), [GOLD, PURPLE]):
        ax.loglog(e, s, color=color, lw=1.2, zorder=4, label=name)

    u238 = sig["U-238 capture"]
    k = int(np.argmax(u238))
    ax.annotate(f"{u238[k] / 1e3:.1f} kbarn at {e[k]:.1f} eV",
                xy=(e[k], u238[k]), xytext=(2.5e-3, 2.6e4), ha="left",
                va="center", fontsize=6.5, color=PURPLE,
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.7))

    ax.set_xlim(E_MIN, E_MAX)
    ax.set_ylim(1.0e-3, 1.0e5)
    ax.set_xlabel("neutron energy (eV)", fontsize=7.4)
    ax.set_ylabel(r"$\sigma$ (barn)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", handlelength=1.5,
              borderaxespad=0.2)
    ax.set_title("Resonances block the slowing-down path", fontsize=8.2,
                 fontweight="bold", color=INK)


def group_panel(ax, e, flux, sig):
    names = list(sig)
    xs = np.arange(len(names), dtype=float)
    width = 0.26
    for i, ((label, phi), color) in enumerate(zip(flux.items(),
                                                  [BLUE, GREEN, RED])):
        vals = [one_group(e, phi, sig[n]) for n in names]
        ax.bar(xs + (i - 1) * width, vals, width * 0.88, color=color,
               edgecolor="white", linewidth=0.5, zorder=3,
               label=label.split(",")[0])
        for x, v in zip(xs + (i - 1) * width, vals):
            ax.text(x, v * 1.18, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.0, color=SUB)

    ax.set_yscale("log")
    ax.set_ylim(0.05, 4.0e3)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7.0)
    ax.set_ylabel(r"one-group $\bar{\sigma}$ (barn)", fontsize=7.4)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=1,
              handlelength=1.2, borderaxespad=0.2)
    ax.set_title("Spectrum-averaged cross sections", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    e = energy_grid()
    flux = spectra(e)
    sig = cross_sections(e)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], left=0.075, right=0.985,
                          top=0.895, bottom=0.125, wspace=0.30, hspace=0.62)
    ax_sp = fig.add_subplot(gs[:, 0])
    ax_xs = fig.add_subplot(gs[0, 1])
    ax_gr = fig.add_subplot(gs[1, 1])

    spectrum_panel(ax_sp, e, flux)
    cross_section_panel(ax_xs, e, sig)
    group_panel(ax_gr, e, flux, sig)

    ratio = (one_group(e, flux["PWR, light water"], sig["U-235 fission"]) /
             one_group(e, flux["sodium-cooled fast"], sig["U-235 fission"]))
    panel_labels_inches(fig, [ax_sp, ax_xs, ax_gr], dx=-0.44, dy=0.12)
    savefig(fig, "655_nuc_neutron_spectrum.png")


if __name__ == "__main__":
    main()
