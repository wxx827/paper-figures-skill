"""Figure 468 - Corpus statistics of a synthetic n-gram inventory.

Three views of the same generated vocabulary: the Zipf rank-frequency law on
log-log axes with a slope fitted per n-gram order, the joint distribution of
word length and word frequency, and the twenty most frequent bigrams.

Canvas aspect is 0.44, outside the 0.55-0.80 band of DESIGN_SYSTEM section
1.2; this is exception E8, a single-row comparison grid with three columns,
where forcing the band would squeeze every panel into a narrow strip.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pubstyle import (set_pub_style, savefig, get_cmap, lighten, darken,
                      annotate_box, letter_panels)

# ---------------------------------------------------------------------------
# Shared NLP-batch vocabulary (464-473)
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d9d9d9"
ACCENT = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"

MONO = {"fontfamily": "monospace"}

ORDERS = [("unigrams", 1.02, BLUE), ("bigrams", 1.20, ACCENT),
          ("trigrams", 1.38, GREEN)]
N_TYPES = 6000
BIGRAMS = ["of the", "in the", "to the", "on the", "and the", "for the",
           "at the", "with the", "from the", "that the", "by the", "it is",
           "this is", "as a", "such as", "there is", "has been", "can be",
           "based on", "in order"]


def zipf(rank, alpha, total, rng, jitter=0.10):
    """Rank-frequency counts following ``f ~ r^-alpha``, with log-noise."""
    f = rank.astype(float) ** (-alpha)
    f = f * np.exp(jitter * rng.standard_normal(f.size))
    return f / f.sum() * total


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(468)
    rank = np.arange(1, N_TYPES + 1)

    counts = {name: zipf(rank, a, 3.4e6, rng) for name, a, _c in ORDERS}
    # longer words are rarer: length grows with the log of the rank
    length = np.clip(np.round(2.2 + 1.05 * np.log(rank) +
                              1.15 * rng.standard_normal(rank.size)),
                     1, 17).astype(int)
    uni = counts["unigrams"]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.6))
    ax0, ax1, ax2 = axes

    # ---- (a) Zipf law ------------------------------------------------------
    fits = {}
    for (name, _a, col) in ORDERS:
        c = counts[name]
        ax0.loglog(rank, c, lw=1.4, color=col, alpha=0.9, zorder=3,
                   label=name)
        s, b = np.polyfit(np.log10(rank), np.log10(c), 1)
        fits[name] = s
        ax0.loglog(rank, 10.0 ** (b + s * np.log10(rank)), lw=0.9,
                   color=darken(col, 0.25), ls=(0, (3.2, 1.8)), zorder=4)
    ax0.legend(loc="upper right", fontsize=7.0, frameon=False,
               handlelength=1.4, borderaxespad=0.2)
    ax0.set_xlabel("frequency rank (-)", fontsize=8.0)
    ax0.set_ylabel("occurrences in the corpus (-)", fontsize=8.0)
    ax0.tick_params(labelsize=7.0, length=2.5)
    ax0.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    ax0.set_title("Zipf law holds over three decades", fontsize=9.0,
                  fontweight="bold")
    annotate_box(ax0, "fitted slopes\n" + "\n".join(
        f"{n:<9s} {fits[n]:+.2f}" for n, _a, _c in ORDERS),
        loc="lower left", fontsize=6.6, **MONO)

    # ---- (b) length vs frequency ------------------------------------------
    xj = length + 0.34 * rng.standard_normal(length.size)
    yl = np.log10(uni)
    hh, xe, ye = np.histogram2d(xj, yl, bins=(46, 46))
    ix = np.clip(np.searchsorted(xe, xj) - 1, 0, hh.shape[0] - 1)
    iy = np.clip(np.searchsorted(ye, yl) - 1, 0, hh.shape[1] - 1)
    dens = hh[ix, iy]
    cmap = get_cmap("cb_sequential_blue")
    order = np.argsort(dens)
    ax1.scatter(xj[order], uni[order], c=dens[order], cmap=cmap, s=6.0,
                linewidth=0.0, zorder=3, vmin=0.0, vmax=dens.max())
    ax1.set_yscale("log")

    lens = np.arange(1, 18)
    med = np.array([np.median(uni[length == L]) if (length == L).any()
                    else np.nan for L in lens])
    ax1.plot(lens, med, lw=1.5, color=ACCENT, zorder=5,
             marker="o", ms=3.0, mfc="white", mec=ACCENT, mew=1.0)
    k = int(np.isfinite(med).sum() // 2)
    ax1.text(lens[k], med[k] * 3.0, "median per length", ha="center",
             va="bottom", fontsize=6.8, color=ACCENT, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                       edgecolor="none", alpha=0.85))
    ax1.set_xlim(0.2, 17.8)
    ax1.set_xlabel("word length (characters)", fontsize=8.0)
    ax1.set_ylabel("occurrences in the corpus (-)", fontsize=8.0)
    ax1.tick_params(labelsize=7.0, length=2.5)
    ax1.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    r = float(np.corrcoef(length, yl)[0, 1])
    ax1.set_title(f"Longer words are rarer, r = {r:+.2f}", fontsize=9.0,
                  fontweight="bold")

    # inline density key, so the panel keeps its full width
    cax = ax1.inset_axes([0.55, 0.90, 0.40, 0.038])
    cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap,
               aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#333333")
    ax1.text(0.55, 0.888, "low", transform=ax1.transAxes, fontsize=6.2,
             ha="left", va="top", color=MUTED)
    ax1.text(0.95, 0.888, "high", transform=ax1.transAxes, fontsize=6.2,
             ha="right", va="top", color=MUTED)
    ax1.text(0.75, 0.948, "point density (-)", transform=ax1.transAxes,
             fontsize=6.2, ha="center", va="bottom", color=MUTED)

    # ---- (c) top-20 bigrams ------------------------------------------------
    bi = counts["bigrams"][:len(BIGRAMS)]
    y = np.arange(len(BIGRAMS))[::-1]
    cols = [lighten(BLUE, 0.52 * i / (len(BIGRAMS) - 1))
            for i in range(len(BIGRAMS))]
    ax2.barh(y, bi, height=0.70, color=cols, edgecolor="white",
             linewidth=0.4, zorder=3)
    for yy, v in zip(y, bi):
        ax2.text(v * 1.04, yy, f"{v / 1e3:.1f}k", va="center", ha="left",
                 fontsize=6.0, color=SUB, **MONO)
    ax2.set_yticks(y)
    ax2.set_yticklabels(BIGRAMS, fontsize=6.4, **MONO)
    ax2.set_xlim(0, bi.max() * 1.26)
    ax2.set_xlabel("occurrences in the corpus (-)", fontsize=8.0)
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax2.xaxis.get_offset_text().set_fontsize(6.5)
    ax2.tick_params(labelsize=7.0, length=2.5)
    ax2.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax2.set_title("Top-20 bigrams are all function words", fontsize=9.0,
                  fontweight="bold")
    share = bi.sum() / counts["bigrams"].sum()
    ax2.text(bi.max() * 1.24, y[-1] - 0.55,
             f"top 20 of {N_TYPES:,} types\ncarry {share:.1%} of all tokens",
             ha="right", va="bottom", fontsize=6.6, color=MUTED,
             linespacing=1.35)

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    letter_panels(axes, x=-0.16, y=1.04)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94], w_pad=2.6)
    savefig(fig, "468_nlp_ngram_stats.png")


if __name__ == "__main__":
    main()
