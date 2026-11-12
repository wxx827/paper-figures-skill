"""Dot plot / Cleveland plot: group comparison with 95% confidence intervals.

Horizontal dot plot with mean markers and thin CI whiskers for six services in
three protection zones - a cleaner alternative to grouped bars. Everything is
put on a common z-score scale so the rows are comparable, alternating bands
separate the metrics, and a dashed zero line marks the regional mean.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pubstyle import set_pub_style, PALETTES, savefig, style_axes

# (metric, native unit, zone means as z-scores: core / buffer / outside)
METRICS = [
    ("Water yield",       "mm yr$^{-1}$",          [0.42, 0.18, -0.55]),
    ("Carbon storage",    "Mg C ha$^{-1}$",        [0.88, 0.35, -0.92]),
    ("Habitat quality",   "index",                 [1.05, 0.44, -1.20]),
    ("Soil conservation", "t ha$^{-1}$ yr$^{-1}$", [0.12, 0.05, -0.09]),
    ("Net primary prod.", "g C m$^{-2}$ yr$^{-1}$", [0.61, 0.52, -0.31]),
    ("Air quality",       "$\\mu$g m$^{-3}$",      [-0.28, 0.14, 0.66]),
]
GROUPS = ["Reserve core", "Buffer zone", "Outside"]
COLORS = [PALETTES["cb_safe"][0], PALETTES["cb_safe"][2],
          PALETTES["cb_safe"][1]]
MARKERS = ["o", "s", "^"]
N = 45


def main():
    set_pub_style()
    rng = np.random.default_rng(401)
    n_m, n_g = len(METRICS), len(GROUPS)
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    step = n_g + 1.4
    y_base = np.arange(n_m) * step

    for i in range(n_m):
        if i % 2 == 0:
            ax.axhspan(y_base[i] - 1.0, y_base[i] + n_g, color="#f4f4f4",
                       zorder=0, lw=0)
    ax.axvline(0, color="#999999", ls="--", lw=0.9, zorder=1)

    for g, (grp, col, mk) in enumerate(zip(GROUPS, COLORS, MARKERS)):
        means, los, his = [], [], []
        for _, _, zs in METRICS:
            sample = rng.normal(zs[g], 0.85, N)
            m = sample.mean()
            half = stats.t.ppf(0.975, N - 1) * sample.std(ddof=1) / np.sqrt(N)
            means.append(m)
            los.append(half)
            his.append(half)
        ypos = y_base + g
        ax.errorbar(means, ypos, xerr=[los, his], fmt=mk, color=col, ms=6,
                    capsize=2.6, capthick=0.9, elinewidth=1.0, label=grp,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    ax.set_yticks(y_base + (n_g - 1) / 2)
    ax.set_yticklabels([f"{m}\n({u})" for m, u, _ in METRICS], fontsize=8,
                       linespacing=1.5)
    ax.set_ylim(y_base[-1] + n_g, y_base[0] - 1.0)
    ax.set_xlabel("Standardised service score (z relative to the regional mean)",
                  fontsize=9)
    ax.set_title("Ecosystem services by protection zone", fontsize=10.5,
                 fontweight="bold", loc="left", x=-0.20, pad=10)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    ax.tick_params(axis="y", length=0)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=8.5, frameon=False, handletextpad=0.4)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    savefig(fig, "44_dotplot.png")


if __name__ == "__main__":
    main()
