"""An absorption spectrometer: the bench on the left, what it returns on the
right.

The main panel is the claim: every element between the lamp and the detector
is there to remove one specific ambiguity - collimation removes angle, the
chopper and reference arm remove lamp drift, the monochromator removes
wavelength ambiguity - and the beam is the only thing that touches them all.
The two smaller panels are what the bench produces: a raw spectrum whose
sloping baseline hides two of the four bands until it is subtracted, and the
integration time the same bench needs to reach a given signal-to-noise ratio.
Peak positions, band depths and the fitted noise exponent are all measured
from the plotted arrays.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy.signal import find_peaks

from pubstyle import set_pub_style, savefig
from figkit import (inch_axes, axes_inches, block, arrow, text_card,
                    panel_labels_inches)

RED = "#b23a48"        # the optical beam and the corrected spectrum
BLUE = "#3b5b92"       # optical elements, raw trace
GOLD = "#c9a227"       # dispersing element, baseline
GREY = "#808080"
DARK = "#3a3a3a"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

MW, MH = 5.35, 4.85
LAM = np.linspace(400.0, 800.0, 900)      # nm
BANDS = [(452.0, 0.62, 14.0), (521.0, 0.28, 11.0),
         (633.0, 0.46, 17.0), (712.0, 0.19, 9.0)]
T_INT = np.logspace(-2.0, 1.4, 60)        # integration time in s


def beam(ax, p0, p1, lw=1.6, head=True):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>" if head else "-", color=RED,
                                lw=lw, mutation_scale=10,
                                shrinkA=0, shrinkB=0), zorder=5)


def bench_panel(ax):
    y = 3.62
    ax.add_patch(Circle((0.52, y), 0.20, facecolor=GOLD, alpha=0.35,
                        edgecolor=GOLD, lw=1.1, zorder=6))
    ax.text(0.52, y + 0.30, "xenon\nlamp", ha="center", va="bottom",
            fontsize=6.8, color=SUB, linespacing=1.35)

    ax.add_patch(Polygon([(1.24, y - 0.30), (1.40, y - 0.24),
                          (1.40, y + 0.24), (1.24, y + 0.30)], closed=True,
                         facecolor=BLUE, alpha=0.30, edgecolor=BLUE, lw=1.0,
                         zorder=6))
    ax.text(1.32, y - 0.36, "collimator", ha="center", va="top",
            fontsize=6.8, color=BLUE)

    ax.add_patch(Rectangle((2.02, y - 0.30), 0.10, 0.60, facecolor=DARK,
                           edgecolor="none", zorder=6))
    ax.add_patch(Circle((2.07, y + 0.42), 0.13, facecolor="none",
                        edgecolor=DARK, lw=1.0, zorder=6))
    ax.text(2.07, y + 0.60, "chopper", ha="center", va="bottom",
            fontsize=6.8, color=SUB)

    ax.add_patch(Rectangle((2.72, y - 0.34), 0.52, 0.68, facecolor=BLUE,
                           alpha=0.16, edgecolor=BLUE, lw=1.0, zorder=6))
    ax.text(2.98, y - 0.40, "sample\ncuvette 10 mm", ha="center", va="top",
            fontsize=6.8, color=BLUE, linespacing=1.35)

    beam(ax, (0.74, y), (1.22, y), head=False)
    beam(ax, (1.42, y), (2.00, y), head=False)
    beam(ax, (2.14, y), (2.70, y), head=False)
    beam(ax, (3.26, y), (4.05, y))

    mono = block(ax, 4.48, y, 0.86, 0.92, color=BLUE, rounding=0.06,
                 zorder=6)
    ax.add_patch(Polygon([(4.30, y - 0.20), (4.66, y - 0.20), (4.48, y + 0.22)],
                         closed=True, facecolor="white", alpha=0.90,
                         edgecolor="white", lw=0.8, zorder=7))
    ax.text(4.48, y + 0.54, "monochromator", ha="center", va="bottom",
            fontsize=6.8, color=BLUE)
    ax.text(4.48, y - 0.54, "grating 1200 lines/mm", ha="center", va="top",
            fontsize=6.6, color=SUB)

    y2 = 1.98
    beam(ax, (mono["b"][0], mono["b"][1]), (4.48, y2 + 0.30))
    det = block(ax, 4.48, y2, 0.80, 0.44, label="detector", color=DARK,
                fontsize=6.8, rounding=0.05, zorder=6)
    arrow(ax, det["l"], (3.30, y2), color=SUB, lw=1.0, gap=0.05, zorder=6)
    text_card(ax, 1.90, y2 - 0.26, 1.32, 0.52,
              "lock-in amplifier\nand ADC", fc="#f5f5f5", ec=HAIR, fs=6.8,
              color=SUB, ha="center", zorder=4)

    text_card(ax, 0.28, 1.02, 2.92, 0.60,
              "Lock-in detection at the chopper frequency rejects room\n"
              "light and low-frequency lamp drift before the ADC.",
              fc="#f5f5f5", ec=HAIR, fs=6.8, color=SUB, ha="center",
              zorder=4)
    ax.plot([0.32, 0.86], [0.62, 0.62], color=RED, lw=1.6, zorder=5)
    ax.text(0.94, 0.62, "beam path", ha="left", va="center", fontsize=6.8,
            color=RED)
    ax.plot([0.32, 0.86], [0.34, 0.34], color=SUB, lw=1.0, zorder=5)
    ax.text(0.94, 0.34, "electrical signal", ha="left", va="center",
            fontsize=6.8, color=SUB)

    ax.set_title("Every element removes one ambiguity from the measurement",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)


def spectra(seed=3):
    rng = np.random.default_rng(seed)
    absorb = np.zeros_like(LAM)
    for c, a, w in BANDS:
        absorb += a * np.exp(-0.5 * ((LAM - c) / w) ** 2)
    drift = 0.55 * np.exp(-(LAM - 400.0) / 260.0) + 0.10 * (LAM - 400.0) / 400.0
    noise = rng.normal(0.0, 0.006, LAM.size)
    raw = absorb + drift + noise
    base = np.polyval(np.polyfit(LAM, drift, 3), LAM)
    return raw, base, raw - base


def spectrum_panel(ax):
    raw, base, corr = spectra()
    ax.plot(LAM, raw, color=BLUE, lw=1.2, zorder=3, label="raw signal")
    ax.plot(LAM, base, color=GOLD, lw=1.1, ls=(0, (4, 2.6)), zorder=4,
            label="fitted baseline")
    ax.plot(LAM, corr, color=RED, lw=1.4, zorder=5, label="after correction")

    pk, _ = find_peaks(corr, prominence=0.08, distance=60)
    for i in pk:
        ax.annotate(f"{LAM[i]:.0f} nm", xy=(LAM[i], corr[i]),
                    xytext=(LAM[i], corr[i] + 0.10), ha="center",
                    va="bottom", fontsize=6.2, color=RED,
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))

    ax.set_xlabel(r"wavelength $\lambda$ (nm)", fontsize=7.4)
    ax.set_ylabel("absorbance (-)", fontsize=7.4)
    ax.set_xlim(LAM[0], LAM[-1])
    ax.set_ylim(-0.08, 1.32)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=3,
              handlelength=1.5, columnspacing=1.0, borderaxespad=0.15,
              bbox_to_anchor=(0.50, 1.03))
    ax.set_title(f"Baseline removal recovers all {len(pk)} bands",
                 fontsize=8.2, fontweight="bold", color=INK)
    return len(pk)


def snr_panel(ax, seed=21):
    rng = np.random.default_rng(seed)
    snr = 34.0 * np.sqrt(T_INT) * (1.0 + rng.normal(0.0, 0.05, T_INT.size))
    dark = 34.0 * np.sqrt(T_INT) / np.sqrt(1.0 + T_INT / 1.6)

    k, b = np.polyfit(np.log10(T_INT), np.log10(snr), 1)
    ax.loglog(T_INT, snr, ls="none", marker="o", ms=2.6, mfc="white",
              mec=BLUE, mew=0.8, zorder=4)
    ax.loglog(T_INT, 10.0 ** (b + k * np.log10(T_INT)), color=RED, lw=1.5,
              zorder=5)
    ax.loglog(T_INT, dark, color=GOLD, lw=1.2, ls=(0, (4, 2.6)), zorder=4)
    ax.axhline(100.0, color=GREY, lw=0.9, ls=(0, (2.4, 2.0)), zorder=3)

    t100 = float(10.0 ** ((np.log10(100.0) - b) / k))
    ax.plot([t100], [100.0], marker="o", ms=4.0, mfc="white", mec=RED,
            mew=1.1, zorder=6)
    ax.annotate(f"SNR 100 at {t100:.2f} s", xy=(t100, 100.0),
                xytext=(t100 * 0.10, 330.0), ha="left", fontsize=6.6,
                color=RED, arrowprops=dict(arrowstyle="-", color=RED,
                                           lw=0.7))
    ax.text(0.011, 800.0, f"shot-noise fit, slope {k:.2f}", ha="left",
            va="top", fontsize=6.6, color=INK)
    ax.text(22.0, dark[-1] * 0.72, "with dark current", ha="right", va="top",
            fontsize=6.6, color=GOLD)

    ax.set_xlabel("integration time $t$ (s)", fontsize=7.4)
    ax.set_ylabel("signal-to-noise ratio (-)", fontsize=7.4)
    ax.set_xlim(T_INT[0], T_INT[-1])
    ax.set_ylim(2.0, 1000.0)
    ax.tick_params(labelsize=6.5)
    ax.grid(which="both", lw=0.4, alpha=0.5)
    ax.set_title("Averaging buys precision as the square root", fontsize=8.2,
                 fontweight="bold", color=INK)
    return k, t100


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 6.4))
    ax_bench = inch_axes(fig, (0.30, 0.55, MW, MH))
    ax_spec = axes_inches(fig, (6.60, 3.60, 3.12, 1.62))
    ax_snr = axes_inches(fig, (6.60, 0.92, 3.12, 1.62))

    bench_panel(ax_bench)
    n_bands = spectrum_panel(ax_spec)
    k, t100 = snr_panel(ax_snr)

    panel_labels_inches(fig, [ax_bench, ax_spec, ax_snr], dx=-0.26, dy=0.14)
    savefig(fig, "701_opt_spectroscopy_setup.png")


if __name__ == "__main__":
    main()
