"""Zipf's law, Heaps' law and the frequency spectrum of one synthetic corpus.

A single sampled corpus feeds all three panels, so the numbers cannot drift
apart: tokens are drawn from a Zipf-Mandelbrot population, and everything the
figure claims - the fitted exponent, the Heaps exponent, the hapax share, the
slope of the frequency spectrum - is measured afterwards from that one sample
with ``np.polyfit``.  The main panel is the rank-frequency law itself; the tail
where the empirical curve peels away from the fit is shaded rather than hidden,
because that departure is a property of every finite corpus.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"

FW, FH = 10.4, 6.4
MAIN = (0.86, 0.84, 4.98, 4.56)
AUX_X, AUX_W, AUX_H = 7.12, 2.90, 1.98
AUX_TOP = (AUX_X, 3.42, AUX_W, AUX_H)
AUX_BOT = (AUX_X, 0.84, AUX_W, AUX_H)

V_TYPES = 20000
N_TOKENS = 400000
ALPHA = 1.07
BETA_M = 2.7            # Mandelbrot offset
FIT_LO, FIT_HI = 5, 2000
HEAPS_LO = 1000
# conventional English function words, pinned to the ranks they label
TOP_WORDS = [("the", 0), ("of", 1), ("and", 2), ("a", 4), ("is", 7)]


def corpus(seed=805):
    """Draw one corpus and return its rank-frequency and first-use profile."""
    rng = np.random.default_rng(seed)
    r = np.arange(1, V_TYPES + 1, dtype=float)
    p = 1.0 / (r + BETA_M) ** ALPHA
    p /= p.sum()
    stream = rng.choice(V_TYPES, size=N_TOKENS, p=p)
    counts = np.bincount(stream, minlength=V_TYPES)
    freq = np.sort(counts)[::-1]
    freq = freq[freq > 0]
    _, first = np.unique(stream, return_index=True)
    return freq, np.sort(first)


def heaps(first, n_pts=60):
    n = np.unique(np.round(np.logspace(2.0, np.log10(N_TOKENS), n_pts)))
    v = np.searchsorted(first, n, side="right").astype(float)
    ok = v > 1
    return n[ok], v[ok]


def spectrum(freq, n_bins=14):
    edges = np.logspace(0.0, np.log10(freq.max() * 1.05), n_bins + 1)
    cnt, _ = np.histogram(freq, bins=edges)
    width = np.diff(edges)
    mid = np.sqrt(edges[:-1] * edges[1:])
    dens = cnt / width
    ok = cnt > 0
    return mid[ok], dens[ok]


def main_panel(ax, freq):
    rank = np.arange(1, freq.size + 1, dtype=float)
    fit_m = (rank >= FIT_LO) & (rank <= FIT_HI)
    slope, icpt = np.polyfit(np.log10(rank[fit_m]), np.log10(freq[fit_m]), 1)
    tail = int(rank[freq <= 1][0]) if np.any(freq <= 1) else int(rank[-1])

    ax.axvspan(tail, rank[-1] * 1.4, color=GREY, alpha=0.10, lw=0.0, zorder=1)
    ax.scatter(rank, freq, s=4.5, color=BLUE, alpha=0.32, linewidths=0.0,
               zorder=3, rasterized=True, label="sampled types")
    xf = np.logspace(np.log10(FIT_LO * 0.4), np.log10(rank[-1] * 1.3), 100)
    ax.plot(xf, 10.0 ** (icpt + slope * np.log10(xf)), color=RED, lw=1.6,
            zorder=5, label="fit  $\\log_{10} f = %.2f %+.2f \\log_{10} r$"
            % (icpt, slope))
    ax.plot(xf, 10.0 ** (icpt + -1.0 * np.log10(xf)), color=GREY, lw=1.0,
            ls=(0, (4, 2.6)), zorder=4, label="slope $-1$ reference")

    for w, i in TOP_WORDS:
        ax.annotate(w, xy=(rank[i], freq[i] * 1.5),
                    xytext=(rank[i], freq[i] * 3.1), ha="center", va="bottom",
                    fontsize=7.2, color=INK, fontstyle="italic",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    idx = [i for _, i in TOP_WORDS]
    ax.scatter(rank[idx], freq[idx], s=14.0, facecolor="white",
               edgecolor=BLUE, linewidths=0.9, zorder=6)

    hapax = int((freq == 1).sum())
    ax.text(0.985, 0.60,
            "shaded: finite-corpus tail from rank %s\n"
            "%s hapax legomena, %.0f %% of the %s types"
            % (f"{tail:,}", f"{hapax:,}", 100.0 * hapax / freq.size,
               f"{freq.size:,}"),
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
            color=SUB, linespacing=1.5)
    ax.legend(loc="lower left", frameon=False, fontsize=7.2,
              handlelength=1.9, borderaxespad=0.35, labelspacing=0.42,
              markerscale=2.0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, rank[-1] * 1.5)
    ax.set_ylim(0.55, freq.max() * 3.2)
    ax.set_xlabel("word rank $r$ (-)")
    ax.set_ylabel("token frequency $f$ (tokens)")
    ax.grid(which="major", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Rank and frequency stay proportional over three decades",
                 fontsize=9.0, fontweight="bold", color=INK)
    return slope, hapax, tail


def heaps_panel(ax, n, v):
    m = n >= HEAPS_LO
    b, a = np.polyfit(np.log10(n[m]), np.log10(v[m]), 1)
    ax.loglog(n, v, marker="o", ms=2.6, lw=0.0, color=BLUE, alpha=0.75,
              mew=0.0)
    ax.loglog(n[m], 10.0 ** (a + b * np.log10(n[m])), color=RED, lw=1.4)
    ax.loglog(n, v[0] * n / n[0], color=GREY, lw=0.9, ls=(0, (4, 2.6)))
    ax.text(0.05, 0.94,
            r"$V = K\,N^{\beta}$,  $\beta$ = %.2f  ($N \geq $ %s)"
            % (b, f"{HEAPS_LO:,}"),
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            color=RED)
    ax.text(0.97, 0.06, "linear growth\nreference", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=GREY,
            linespacing=1.4)
    ax.set_xlabel("corpus size $N$ (tokens)", fontsize=7.4)
    ax.set_ylabel("vocabulary $V$ (types)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Heaps' law: vocabulary grows sub-linearly", fontsize=8.2,
                 fontweight="bold", color=INK)
    return b


def spectrum_panel(ax, mid, dens):
    s, a = np.polyfit(np.log10(mid), np.log10(dens), 1)
    ax.loglog(mid, dens, marker="s", ms=3.0, lw=1.1, color=GOLD,
              mfc="white", mew=0.9)
    ax.loglog(mid, 10.0 ** (a + s * np.log10(mid)), color=RED, lw=1.2,
              ls=(0, (3.2, 2.0)))
    ax.text(0.05, 0.10, "log-binned, slope %.2f" % s, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.0, color=RED)
    ax.set_xlabel("token frequency $f$ (tokens)", fontsize=7.4)
    ax.set_ylabel("types per unit $f$ (types)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Frequency spectrum of the same corpus", fontsize=8.2,
                 fontweight="bold", color=INK)
    return s


def main():
    set_pub_style()
    freq, first = corpus()
    n, v = heaps(first)
    mid, dens = spectrum(freq)

    fig = plt.figure(figsize=(FW, FH))
    ax_main = fig.add_axes([MAIN[0] / FW, MAIN[1] / FH,
                            MAIN[2] / FW, MAIN[3] / FH])
    ax_heap = fig.add_axes([AUX_TOP[0] / FW, AUX_TOP[1] / FH,
                            AUX_TOP[2] / FW, AUX_TOP[3] / FH])
    ax_spec = fig.add_axes([AUX_BOT[0] / FW, AUX_BOT[1] / FH,
                            AUX_BOT[2] / FW, AUX_BOT[3] / FH])

    slope, hapax, tail = main_panel(ax_main, freq)
    beta = heaps_panel(ax_heap, n, v)
    spectrum_panel(ax_spec, mid, dens)

    panel_labels_inches(fig, [ax_main, ax_heap, ax_spec], dx=-0.56, dy=0.10)
    savefig(fig, "805_lng_zipf_law.png")


if __name__ == "__main__":
    main()
