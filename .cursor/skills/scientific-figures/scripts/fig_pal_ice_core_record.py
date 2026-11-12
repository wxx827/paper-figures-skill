"""Eight glacial cycles read off a single ice core, four proxies at a time.

The main block is four stacked records on one shared age axis, drawn the way
palaeoclimate stratigraphy is always drawn: age increases to the left, so the
present day sits at the right-hand edge and the reader walks backwards in
time.  Marine isotope stages tint the background - warm odd stages in house
red, cold even stages in house blue - so the cyclicity is visible before any
curve is read.  The small square on the right carries the supporting evidence:
every pairwise correlation between the four proxies, computed from exactly the
series that are plotted.

Numbers in the annotations (glacial-interglacial amplitudes, the correlation
coefficients, the LGM values) are all measured at run time from the synthetic
records, so changing the seed changes them.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, lighten, savefig
from figkit import axes_inches, panel_labels_inches, safe_colorbar

RED = "#b23a48"       # warm / interglacial
BLUE = "#3b5b92"      # cold / glacial
GOLD = "#c9a227"      # methane
OCHRE = "#c0603f"     # dust
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
SOLID = "#3a3a3a"

PERMIL = "\u2030"
AGE_MAX = 800.0
LGM = 21.0

WARM_BAND = lighten(RED, 0.90)
COLD_BAND = lighten(BLUE, 0.92)

# Marine isotope stage boundaries (ka BP); odd stages are the warm ones.
MIS = [(0, 14, 1), (14, 29, 2), (29, 57, 3), (57, 71, 4), (71, 130, 5),
       (130, 191, 6), (191, 243, 7), (243, 300, 8), (300, 337, 9),
       (337, 374, 10), (374, 424, 11), (424, 478, 12), (478, 533, 13),
       (533, 563, 14), (563, 621, 15), (621, 676, 16), (676, 712, 17),
       (712, 761, 18), (761, 790, 19), (790, 800, 20)]


def _red(rng, n, width):
    """Unit-amplitude reddened noise: white noise through a Gaussian kernel."""
    ker = np.exp(-0.5 * (np.arange(-4 * width, 4 * width + 1) / width) ** 2)
    ker /= ker.sum()
    out = np.convolve(rng.normal(0.0, 1.0, n), ker, mode="same")
    return out / np.abs(out).max()


def records(seed=20):
    """Orbitally paced synthetic ice-core proxies on a common age scale.

    Each proxy is the shared orbital forcing plus its own reddened noise and
    its own lag, so the cross-proxy correlations that panel (e) reports are
    high but not degenerate - which is the point of showing them at all.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, AGE_MAX, 2400)

    # Skewed 100 ka sawtooth (slow build-up, fast termination) plus obliquity
    # and precession.
    p100 = np.cos(2 * np.pi * t / 100.0 + 0.90 * np.sin(2 * np.pi * t / 100.0))
    p41 = np.cos(2 * np.pi * t / 41.0 + 0.6)
    p23 = np.cos(2 * np.pi * t / 23.0 + 1.3)

    warm = p100 + 0.28 * p41 + 0.16 * p23 + 0.16 * _red(rng, t.size, 9.0)
    warm /= np.abs(warm).max()

    def lagged(ka, local, width=7.0):
        """Forcing delayed by ``ka`` and blurred by proxy-specific noise."""
        y = np.interp(t + ka, t, warm, left=warm[0], right=warm[-1])
        y = y + local * _red(rng, t.size, width)
        return y / np.abs(y).max()

    w_ice = lagged(0.0, 0.30, 6.0)
    w_co2 = lagged(1.5, 0.24, 11.0)
    w_ch4 = lagged(0.4, 0.34, 5.0)
    w_dst = lagged(2.2, 0.40, 5.0)
    w_sea = lagged(5.5, 0.22, 14.0)

    d18o = -54.4 + 4.3 * w_ice + 0.15 * rng.normal(size=t.size)
    co2 = 232.0 + 43.0 * w_co2 + 2.4 * rng.normal(size=t.size)
    ch4 = 515.0 + 165.0 * w_ch4 + 10.0 * rng.normal(size=t.size)
    dust = (0.32 + 7.4 * np.exp(-2.5 * (w_dst + 1.0))) * \
           (1.0 + 0.09 * rng.normal(size=t.size))
    sea = -58.0 + 58.0 * w_sea + 2.0 * rng.normal(size=t.size)
    return t, dict(d18o=d18o, co2=co2, ch4=ch4, dust=dust, sea=sea)


def stage_bands(ax, label_stages=False, ymax_frac=0.97):
    for a0, a1, num in MIS:
        ax.axvspan(a0, a1, facecolor=WARM_BAND if num % 2 else COLD_BAND,
                   edgecolor="none", zorder=0)
    if label_stages:
        lo, hi = ax.get_ylim()
        y = lo + ymax_frac * (hi - lo)
        for a0, a1, num in MIS:
            if num % 2 == 0 or (a1 - a0) < 13.0:
                continue
            ax.text(0.5 * (a0 + a1), y, str(num), ha="center", va="top",
                    fontsize=6.0, color=SUB, zorder=5)


def strip(ax, bottom=False):
    ax.set_xlim(0.0, AGE_MAX)
    ax.invert_xaxis()
    ax.tick_params(labelsize=7.0)
    ax.axvline(LGM, color=GREY, lw=0.6, ls=(0, (3, 2.2)), zorder=4)
    if bottom:
        ax.set_xlabel("age (ka BP), increasing to the left", fontsize=8.0)
    else:
        ax.tick_params(labelbottom=False)


def panel_d18o(ax, t, d18o):
    stage_bands(ax)
    mid = float(d18o.mean())
    ax.fill_between(t, mid, d18o, where=d18o >= mid, color=RED, alpha=0.40,
                    linewidth=0.0, interpolate=True, zorder=2)
    ax.fill_between(t, mid, d18o, where=d18o < mid, color=BLUE, alpha=0.40,
                    linewidth=0.0, interpolate=True, zorder=2)
    ax.plot(t, d18o, color=SOLID, lw=0.65, zorder=3)
    ax.set_ylim(d18o.min() - 0.8, d18o.max() + 2.4)
    ax.set_ylabel("$\\delta^{18}O$ of ice\n(%s VSMOW)" % PERMIL, fontsize=7.6)
    strip(ax)
    stage_bands(ax, label_stages=True)

    warm_top = float(np.percentile(d18o, 96))
    cold_bot = float(np.percentile(d18o, 4))
    ax.annotate("Holocene", xy=(5.0, d18o[t <= 6.0].mean()),
                xytext=(88.0, warm_top + 0.9), ha="right", va="center",
                fontsize=6.8, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    ax.annotate("last glacial maximum", xy=(LGM, d18o[np.argmin(abs(t - LGM))]),
                xytext=(255.0, cold_bot - 0.1), ha="right", va="center",
                fontsize=6.8, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    return float(np.percentile(d18o, 96) - np.percentile(d18o, 4))


def panel_gases(ax, t, co2, ch4):
    stage_bands(ax)
    ax.plot(t, co2, color=RED, lw=0.9, zorder=3)
    ax.set_ylabel("$CO_2$\n(ppm)", fontsize=7.6, color=RED)
    ax.tick_params(axis="y", colors=RED)
    ax.set_ylim(co2.min() - 12, co2.max() + 12)

    axr = ax.twinx()
    axr.plot(t, ch4, color=GOLD, lw=0.8, alpha=0.95, zorder=2)
    axr.set_ylabel("$CH_4$\n(ppb)", fontsize=7.6, color=GOLD)
    axr.tick_params(axis="y", colors=GOLD, labelsize=7.0)
    axr.set_ylim(ch4.min() - 60, ch4.max() + 60)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(GOLD)
    axr.set_xlim(0.0, AGE_MAX)
    axr.invert_xaxis()
    strip(ax)
    return float(co2.max() - co2.min())


def panel_dust(ax, t, dust):
    stage_bands(ax)
    ax.fill_between(t, 0.0, dust, color=OCHRE, alpha=0.45, linewidth=0.0,
                    zorder=2)
    ax.plot(t, dust, color=OCHRE, lw=0.7, zorder=3)
    ax.set_ylim(0.0, dust.max() * 1.18)
    ax.set_ylabel("dust flux\n(mg m$^{-2}$ yr$^{-1}$)", fontsize=7.6)
    strip(ax)
    peak = float(dust.max())
    ax.text(AGE_MAX - 12.0, dust.max() * 1.06,
            "glacial dust peaks reach %.1f mg m$^{-2}$ yr$^{-1}$" % peak,
            ha="left", va="top", fontsize=6.6, color=SUB)


def panel_sea(ax, t, sea):
    stage_bands(ax)
    ax.fill_between(t, sea.min() - 12, sea, color=BLUE, alpha=0.30,
                    linewidth=0.0, zorder=2)
    ax.plot(t, sea, color=BLUE, lw=0.85, zorder=3)
    ax.axhline(0.0, color=GREY, lw=0.7, ls=(0, (2.4, 2.0)), zorder=3)
    ax.set_ylim(sea.min() - 12, 24.0)
    ax.set_ylabel("relative sea level\n(m)", fontsize=7.6)
    strip(ax, bottom=True)
    ax.set_xticks(np.arange(0, 801, 100))
    ax.text(AGE_MAX - 12.0, 16.0, "present-day datum", ha="left", va="top",
            fontsize=6.6, color=MUTED)


def panel_corr(fig, ax, series):
    keys = ["d18o", "co2", "ch4", "dust", "sea"]
    names = ["$\\delta^{18}O$", "$CO_2$", "$CH_4$", "dust", "sea level"]
    mat = np.corrcoef(np.vstack([series[k] for k in keys]))
    im = ax.imshow(mat, cmap=get_cmap("cb_diverging"), vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(names, fontsize=6.8, rotation=32, ha="right")
    ax.set_yticklabels(names, fontsize=6.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#cccccc")
    for i in range(len(keys)):
        for j in range(len(keys)):
            r = mat[i, j]
            ax.text(j, i, ("%.2f" % r).replace("-0.", "-."),
                    ha="center", va="center", fontsize=6.2,
                    color="white" if abs(r) > 0.62 else INK)
    ax.set_title("Pairwise proxy coherence", fontsize=8.5, fontweight="bold",
                 color=INK, pad=6)
    safe_colorbar(fig, im, ax, label="Pearson r (-)", width=0.10, pad=0.07,
                  labelsize=6.5, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    return mat


def main():
    set_pub_style()
    t, s = records()

    fig = plt.figure(figsize=(10.4, 6.4))
    x0, w = 0.86, 5.92
    h, gap = 1.025, 0.26
    tops = [5.50 - i * (h + gap) for i in range(4)]
    axes = [axes_inches(fig, (x0, top - h, w, h)) for top in tops]

    amp = panel_d18o(axes[0], t, s["d18o"])
    dco2 = panel_gases(axes[1], t, s["co2"], s["ch4"])
    panel_dust(axes[2], t, s["dust"])
    panel_sea(axes[3], t, s["sea"])

    ax_corr = axes_inches(fig, (7.62, 2.05, 2.15, 2.15))
    mat = panel_corr(fig, ax_corr, s)


    panel_labels_inches(fig, axes + [ax_corr], dx=-0.70, dy=0.02)
    savefig(fig, "750_pal_ice_core_record.png")


if __name__ == "__main__":
    main()
