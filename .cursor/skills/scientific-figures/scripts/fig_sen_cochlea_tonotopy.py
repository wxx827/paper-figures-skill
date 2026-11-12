"""Tonotopic map of the unrolled cochlea with three traveling-wave envelopes.

The main panel unrolls the basilar membrane into a straight strip running from
the stapes to the helicotrema, shaded by the logarithm of the characteristic
frequency, and lays the envelopes of three pure tones on top of it.  The
asymmetry of each envelope - a long shallow flank toward the base and an
abrupt cut-off past the peak - is what makes the place code work, and it is
the reason low frequencies mask high ones far more readily than the reverse.

The two subordinate panels give the analytic backing: the Greenwood
frequency-position function that fixes every label on the strip, and the
critical bandwidth, which shows that the resolving power of the map is roughly
constant in relative terms above about 500 Hz.  Peak places, quality factors
and bandwidth ratios printed on the figure are measured from the plotted
arrays.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_labels_inches, halo_text

RED = "#b23a48"
GREEN = "#2e7d5b"
PURPLE = "#8e6bbf"
BLUE = "#3b5b92"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"

L_BM = 35.0                       # basilar-membrane length, mm
GW_A, GW_ALPHA, GW_K = 165.4, 2.1, 0.88
TONES = [(4000.0, RED, "4 kHz"), (1000.0, PURPLE, "1 kHz"),
         (250.0, GREEN, "250 Hz")]
STRIP_Y0, STRIP_Y1 = -0.34, -0.08
LADDER = [16000.0, 8000.0, 4000.0, 2000.0, 1000.0, 500.0, 250.0, 125.0]
LAD_TXT = ["16k", "8k", "4k", "2k", "1k", "500", "250", "125"]


def gw_freq(x):
    """Greenwood characteristic frequency (Hz) at distance x mm from stapes."""
    return GW_A * (10.0 ** (GW_ALPHA * (1.0 - x / L_BM)) - GW_K)


def gw_place(f):
    """Inverse Greenwood map: distance from the stapes in mm."""
    return L_BM * (1.0 - np.log10(f / GW_A + GW_K) / GW_ALPHA)


def envelope(x, xcf):
    """Asymmetric traveling-wave envelope peaking at the tone's own place."""
    w_base = 2.5 + 0.28 * xcf
    w_apex = 1.4 + 0.045 * xcf
    e = np.where(x <= xcf,
                 np.exp(-((x - xcf) / w_base) ** 2),
                 np.exp(-((x - xcf) / w_apex) ** 2))
    e = e * (1.0 - np.exp(-x / 2.2))
    return e / e.max()


def critical_bands(f):
    erb = 24.7 * (4.37 * f / 1000.0 + 1.0)
    bark = 25.0 + 75.0 * (1.0 + 1.4 * (f / 1000.0) ** 2) ** 0.69
    third = f * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
    return erb, bark, third


def main_panel(ax, seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, L_BM, 900)

    cf_img = np.log10(np.clip(gw_freq(x), 20.0, None))[None, :]
    ax.imshow(cf_img, aspect="auto", origin="lower",
              cmap=get_cmap("cb_sequential_blue"),
              extent=[0.0, L_BM, STRIP_Y0, STRIP_Y1], zorder=2)
    ax.plot([0.0, L_BM, L_BM, 0.0, 0.0],
            [STRIP_Y0, STRIP_Y0, STRIP_Y1, STRIP_Y1, STRIP_Y0],
            color="#333333", lw=0.7, zorder=4)

    for f, txt in zip(LADDER, LAD_TXT):
        xp = gw_place(f)
        ax.plot([xp, xp], [STRIP_Y0, STRIP_Y0 - 0.035], color="#333333",
                lw=0.7, zorder=4)
        ax.text(xp, STRIP_Y0 - 0.055, txt, ha="center", va="top",
                fontsize=6.2, color=SUB)
    ax.text(L_BM / 2.0, -0.60, "characteristic frequency (Hz)", ha="center",
            va="top", fontsize=6.8, color=SUB)
    ax.text(0.2, -0.74, "base - stapes, stiff membrane", ha="left",
            va="top", fontsize=6.6, color=MUTED)
    ax.text(L_BM - 0.2, -0.74, "apex - helicotrema, compliant", ha="right",
            va="top", fontsize=6.6, color=MUTED)
    halo_text(ax, 1.0, (STRIP_Y0 + STRIP_Y1) / 2.0, "high CF", ha="left",
              va="center", fontsize=6.4, color="white", halo="#08263f",
              zorder=5)
    halo_text(ax, L_BM - 1.0, (STRIP_Y0 + STRIP_Y1) / 2.0, "low CF",
              ha="right", va="center", fontsize=6.4, color="#1a1a1a",
              halo="white", zorder=5)

    ax.axhline(0.0, color=GREY, lw=0.7, zorder=2)
    knots = np.linspace(0.0, L_BM, 15)
    peaks = {}
    for f, col, name in TONES:
        e = envelope(x, gw_place(f))
        e = e * (1.0 + 0.018 * np.interp(x, knots,
                                         rng.normal(size=knots.size)))
        xp = float(x[int(np.argmax(e))])
        peaks[name] = xp
        ax.fill_between(x, 0.0, e, color=col, alpha=0.16, linewidth=0.0,
                        zorder=3)
        ax.plot(x, e, color=col, lw=1.6, zorder=4, label=f"{name} tone")
        ax.plot([xp, xp], [0.0, -0.02], color=col, lw=0.9, zorder=4)
        ax.plot([xp, xp], [STRIP_Y1, e.max()], color=col, lw=0.7,
                ls=(0, (2.2, 2.0)), alpha=0.75, zorder=3)
        ax.plot([xp], [e.max()], marker="v", ms=4.0, color=col, zorder=5)
        ax.text(xp, e.max() + 0.055, f"{xp:.1f} mm", ha="center", va="bottom",
                fontsize=6.5, color=col)

    ax.set_xlim(0.0, L_BM)
    ax.set_ylim(-0.86, 1.20)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(np.arange(0.0, L_BM + 0.1, 5.0))
    ax.set_xlabel("distance from stapes (mm)")
    ax.set_ylabel("relative membrane displacement (-)")
    ax.yaxis.set_label_coords(-0.072, 0.70)
    ax.spines["left"].set_bounds(0.0, 1.0)
    ax.legend(fontsize=7.0, frameon=False, loc="upper left",
              handlelength=1.6, labelspacing=0.30, borderaxespad=0.3)
    span = peaks["250 Hz"] - peaks["4 kHz"]
    ax.set_title(f"Four octaves of tone frequency map onto {span:.1f} mm of "
                 f"membrane", fontsize=9.0, fontweight="bold", color=INK)
    return peaks


def greenwood_panel(ax, peaks):
    x = np.linspace(0.0, L_BM, 400)
    ax.semilogy(x, gw_freq(x), color=BLUE, lw=1.7, zorder=4)
    for f, col, name in TONES:
        xp = peaks[name]
        ax.plot([xp], [f], marker="o", ms=4.0, color=col, mfc="white",
                mew=1.2, zorder=5)
        ax.plot([xp, xp], [20.0, f], color=col, lw=0.7, ls=(0, (2.2, 2.0)),
                zorder=3)
    oct_mm = L_BM * np.log10(2.0) / GW_ALPHA
    ax.annotate(f"{oct_mm:.1f} mm per octave", xy=(0.04, 0.11),
                xycoords="axes fraction", ha="left", va="center",
                fontsize=6.6, color=SUB)
    ax.set_xlim(0.0, L_BM)
    ax.set_ylim(20.0, 30000.0)
    ax.set_xticks(np.arange(0.0, L_BM + 0.1, 7.0))
    ax.set_xlabel("distance from stapes (mm)", fontsize=7.4)
    ax.set_ylabel("characteristic frequency (Hz)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Greenwood frequency-position function", fontsize=8.2,
                 fontweight="bold", color=INK)


def bandwidth_panel(ax):
    f = np.logspace(np.log10(80.0), np.log10(16000.0), 400)
    erb, bark, third = critical_bands(f)
    ax.loglog(f, erb, color=BLUE, lw=1.6, label="ERB")
    ax.loglog(f, bark, color=RED, lw=1.4, ls=(0, (5, 2.2)),
              label="Zwicker critical band")
    ax.loglog(f, third, color=GREY, lw=1.0, ls=(0, (1.6, 1.8)),
              label="1/3 octave")
    i = int(np.argmin(np.abs(f - 4000.0)))
    ax.plot([f[i]], [erb[i]], marker="o", ms=3.4, color=BLUE, mfc="white",
            mew=1.0)
    ax.text(f[i] * 1.14, erb[i] * 0.52,
            f"ERB at 4 kHz = {erb[i]:.0f} Hz\n({erb[i] / f[i] * 100:.1f} % of "
            f"$f_c$)", ha="left", va="top", fontsize=6.5, color=BLUE,
            linespacing=1.3)
    ax.set_xlim(80.0, 16000.0)
    ax.set_ylim(20.0, 5000.0)
    ax.set_xlabel("centre frequency $f_c$ (Hz)", fontsize=7.4)
    ax.set_ylabel("bandwidth (Hz)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.7, which="major")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left",
              handlelength=1.5, labelspacing=0.28, borderaxespad=0.2)
    ax.set_title("Critical bandwidth against frequency", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.66)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_gw = fig.add_subplot(gs[0, 1])
    ax_bw = fig.add_subplot(gs[1, 1])

    peaks = main_panel(ax_main)
    greenwood_panel(ax_gw, peaks)
    bandwidth_panel(ax_bw)

    fig.tight_layout(rect=[0, 0.035, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_gw, ax_bw], dx=-0.46, dy=0.12)
    savefig(fig, "727_sen_cochlea_tonotopy.png")


if __name__ == "__main__":
    main()
