"""Equal-loudness contours with the weighting curves and loudness growth.

The main panel is the ISO 226 family: sound pressure level against frequency
for a set of fixed loudness levels, anchored at 1 kHz where phon and dB SPL
coincide by definition.  The contours are flattest in the 2-5 kHz ear-canal
resonance and splay apart below 200 Hz, which is the whole reason a level
meter needs a weighting network at all - so the inverted A and C responses,
aligned at the same 1 kHz anchor, are drawn over the family for comparison.

The subordinate panels carry the two consequences: how loudness in sones grows
once the level is above threshold, and how a single narrow-band masker lifts
the threshold asymmetrically around itself.  Every printed number is measured
from the plotted arrays.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_palette
from figkit import panel_labels_inches

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

# ISO 226:2003 reference tables (29 third-octave bands, 20 Hz - 12.5 kHz).
ISO_F = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
                  315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                  3150, 4000, 5000, 6300, 8000, 10000, 12500], dtype=float)
ISO_AF = np.array([0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367,
                   0.349, 0.330, 0.315, 0.301, 0.288, 0.276, 0.267, 0.259,
                   0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242,
                   0.242, 0.245, 0.254, 0.271, 0.301])
ISO_LU = np.array([-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1,
                   -6.2, -4.5, -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0,
                   -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2,
                   -10.7, -3.1])
ISO_TF = np.array([78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1,
                   17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5,
                   1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3])

PHONS = [20.0, 40.0, 60.0, 80.0, 100.0]
F_REF = 1000.0
PAIN_SPL = 130.0


def iso226(phon):
    """Sound pressure level (dB SPL) of an equal-loudness contour."""
    af = 4.47e-3 * (10.0 ** (0.025 * phon) - 1.15) + \
        (0.4 * 10.0 ** (((ISO_TF + ISO_LU) / 10.0) - 9.0)) ** ISO_AF
    return (10.0 / ISO_AF) * np.log10(af) - ISO_LU + 94.0


def weighting(f, kind="A"):
    """IEC 61672 frequency weighting in dB, normalised to 0 dB at 1 kHz."""
    f2 = f ** 2
    c = (12194.0 ** 2 * f2) / ((f2 + 20.6 ** 2) * (f2 + 12194.0 ** 2))
    if kind == "C":
        return 20.0 * np.log10(c) + 0.062
    a = c * f2 / np.sqrt((f2 + 107.65265 ** 2) * (f2 + 737.86223 ** 2))
    return 20.0 * np.log10(a) + 2.0


def loudness_growth():
    phon = np.linspace(2.0, 110.0, 300)
    sone = np.where(phon >= 40.0, 2.0 ** ((phon - 40.0) / 10.0),
                    (phon / 40.0) ** 2.86 * 1.0)
    return phon, sone


def masking(seed=11):
    f = np.logspace(np.log10(120.0), np.log10(9000.0), 400)
    thr = np.interp(np.log10(f), np.log10(ISO_F), ISO_TF)
    fm, lm = 1000.0, 80.0
    z = np.log2(f / fm)
    lo_slope, hi_slope = 27.0, 12.0          # dB per octave, low / high side
    pattern = lm - 8.0 - np.where(z < 0.0, -lo_slope * z, hi_slope * z)
    rng = np.random.default_rng(seed)
    knots = np.linspace(np.log10(f[0]), np.log10(f[-1]), 12)
    pattern = pattern + 1.1 * np.interp(np.log10(f), knots,
                                        rng.normal(size=knots.size))
    return f, thr, np.maximum(pattern, thr), fm, lm


def main_panel(ax):
    cols = get_palette("cb_sequential_blue", 7)[2:7]
    curves = {}
    for phon, col in zip(PHONS, cols):
        spl = iso226(phon)
        curves[phon] = spl
        ax.plot(ISO_F, spl, color=col, lw=1.7, zorder=5)
        ax.text(ISO_F[-1] * 1.06, spl[-1], f"{phon:.0f}", ha="left",
                va="center", fontsize=6.8, color=col)
    ax.text(ISO_F[-1] * 1.06, 118.0, "phon", ha="left", va="center",
            fontsize=6.8, color=SUB)

    thr = iso226(0.0)
    ax.plot(ISO_F, thr, color=GREY, lw=1.2, ls=(0, (4, 2.4)), zorder=5)
    ax.fill_between(ISO_F, -12.0, thr, color="#f0f0f0", zorder=1)
    ax.text(70.0, 12.0, "inaudible", ha="left", va="center", fontsize=6.6,
            color=MUTED)
    ax.text(2100.0, 3.5, "hearing threshold (0 phon)", ha="left",
            va="bottom", fontsize=6.8, color=SUB)

    pain = np.full_like(ISO_F, PAIN_SPL)
    ax.plot(ISO_F, pain, color=RED, lw=1.1, ls=(0, (5, 2.4)), zorder=5)
    ax.text(60.0, PAIN_SPL + 2.0, "pain threshold", ha="left", va="bottom",
            fontsize=6.8, color=RED)

    fw = np.logspace(np.log10(20.0), np.log10(12500.0), 300)
    for kind, col, ls in [("A", GREEN, (0, (2.6, 2.0))),
                          ("C", RED, (0, (1.4, 1.6)))]:
        ax.plot(fw, 40.0 - weighting(fw, kind), color=col, lw=1.2, ls=ls,
                alpha=0.9, zorder=4)
    ax.text(24.0, 40.0 - weighting(np.array([24.0]), "A")[0] + 2.0,
            "inverted A", ha="left", va="bottom", fontsize=6.8, color=GREEN)
    ax.text(24.0, 40.0 - weighting(np.array([24.0]), "C")[0] - 6.0,
            "inverted C", ha="left", va="top", fontsize=6.8, color=RED)

    ax.plot([F_REF] * len(PHONS), [curves[p][17] for p in PHONS], marker="o",
            ms=3.6, ls="none", color=INK, mfc="white", mew=1.0, zorder=7)
    ax.axvline(F_REF, color=GREY, lw=0.7, ls=(0, (1.6, 2.0)), zorder=2)
    ax.text(F_REF * 1.08, 127.0, "1 kHz anchor: phon = dB SPL", ha="left",
            va="top", fontsize=6.8, color=SUB)

    i100 = int(np.argmin(np.abs(ISO_F - 100.0)))
    lift = curves[40.0][i100] - curves[40.0][17]
    ax.annotate("", xy=(100.0, curves[40.0][i100]), xytext=(100.0, 40.0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8,
                                shrinkA=1.0, shrinkB=1.0), zorder=7)
    ax.text(105.0, (curves[40.0][i100] + 40.0) / 2.0, f"+{lift:.0f} dB",
            ha="left", va="center", fontsize=6.8, color=INK,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.88), zorder=8)

    ax.set_xscale("log")
    ax.set_xlim(20.0, 12500.0)
    ax.set_ylim(-12.0, 142.0)
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k",
                        "10k"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(np.arange(0.0, 141.0, 20.0))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("sound pressure level (dB SPL)")
    ax.grid(lw=0.5, alpha=0.7)
    ax.set_title(f"Equal loudness costs {lift:.0f} dB more at 100 Hz than at "
                 f"1 kHz", fontsize=9.0, fontweight="bold", color=INK)
    return curves, lift


def growth_panel(ax):
    phon, sone = loudness_growth()
    ax.semilogy(phon, sone, color=BLUE, lw=1.7)
    ax.plot([40.0], [1.0], marker="o", ms=3.8, color=BLUE, mfc="white",
            mew=1.1)
    ax.text(42.0, 1.0, "1 sone = 40 phon", ha="left", va="center",
            fontsize=6.6, color=BLUE)
    i50, i60 = int(np.argmin(np.abs(phon - 50.0))), \
        int(np.argmin(np.abs(phon - 60.0)))
    ax.plot(phon[[i50, i60]], sone[[i50, i60]], marker="o", ms=3.2, lw=0.0,
            color=GREY, mfc="white", mew=1.0)
    ax.plot([phon[i50], phon[i50]], [sone[i50], sone[i60]], color=GREY,
            lw=0.7, ls=(0, (2.2, 2.0)))
    ax.text(3.0, 6.0, f"+10 phon\n= {sone[i60] / sone[i50]:.1f} x loudness",
            ha="left", va="bottom", fontsize=6.5, color=SUB, linespacing=1.3)
    ax.set_xlim(0.0, 110.0)
    ax.set_ylim(0.02, 200.0)
    ax.set_xlabel("loudness level (phon)", fontsize=7.4)
    ax.set_ylabel("loudness (sone)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Loudness growth function", fontsize=8.2, fontweight="bold",
                 color=INK)


def masking_panel(ax):
    f, thr, pattern, fm, lm = masking()
    ax.fill_between(f, thr, pattern, color=RED, alpha=0.14, linewidth=0.0)
    ax.plot(f, thr, color=GREY, lw=1.1, ls=(0, (4, 2.4)),
            label="quiet threshold")
    ax.plot(f, pattern, color=RED, lw=1.5, label="masked threshold")
    ax.plot([fm, fm], [thr.min() - 6.0, lm], color=BLUE, lw=2.2, zorder=5)
    ax.plot([fm], [lm], marker="v", ms=4.4, color=BLUE, zorder=5)
    ax.text(fm * 1.10, lm, f"masker {lm:.0f} dB SPL", ha="left", va="center",
            fontsize=6.5, color=BLUE)

    hi = pattern[np.argmin(np.abs(f - 2000.0))]
    lo = pattern[np.argmin(np.abs(f - 500.0))]
    ax.text(0.97, 0.42,
            f"upward spread {hi - lo:+.0f} dB (0.5 to 2 kHz)",
            transform=ax.transAxes, ha="right", va="center", fontsize=6.5,
            color=SUB)
    ax.set_xscale("log")
    ax.set_xlim(120.0, 9000.0)
    ax.set_ylim(-12.0, 100.0)
    ax.set_xticks([200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["200", "500", "1k", "2k", "5k"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("frequency (Hz)", fontsize=7.4)
    ax.set_ylabel("threshold (dB SPL)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", handlelength=1.5,
              labelspacing=0.28, borderaxespad=0.2)
    ax.set_title("Masking pattern of a 1 kHz tone", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.66)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_gr = fig.add_subplot(gs[0, 1])
    ax_mk = fig.add_subplot(gs[1, 1])

    curves, lift = main_panel(ax_main)
    growth_panel(ax_gr)
    masking_panel(ax_mk)

    fig.tight_layout(rect=[0, 0.035, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_gr, ax_mk], dx=-0.46, dy=0.12)
    savefig(fig, "728_sen_loudness_contours.png")


if __name__ == "__main__":
    main()
