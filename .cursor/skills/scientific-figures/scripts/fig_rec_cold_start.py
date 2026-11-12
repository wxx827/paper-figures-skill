"""Figure 475 - Cold-start accuracy of six recommendation strategies.

Small multiples, one strategy per panel: accuracy against the number of
interactions a user has already produced.  Every panel repeats all six curves
in light grey and highlights its own, so a reader can place each strategy in
the family without cross-referencing a legend, and the cold-start bins are
shaded across all six.

The y axis is computed once over every curve and every band, then set on all
six panels - the whole point of the figure is that content-based ranking wins
on the left and collaborative filtering wins on the right, which is invisible
the moment a panel is allowed to autoscale.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten
from figkit import panel_labels_inches, shared_limits

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID = "#cccccc", "#d9d9d9"

BINS = ["0", "1-2", "3-5", "6-10", "11-25", "26-60", "> 60"]
COLD = 3                      # bins 0-2 count as the cold-start regime
N_SEED = 4200                 # evaluation users per bin

# Baselines walk a grey ramp; the proposed model is the only coloured curve.
STRATEGIES = [
    ("Random",        "#b3b3b3", 1.3, (0.018, 0.022, 0.90)),
    ("Popularity",    "#9a9a9a", 1.3, (0.086, 0.104, 0.75)),
    ("Content-based", "#808080", 1.3, (0.152, 0.203, 0.55)),
    ("Collaborative", "#5a5a5a", 1.3, (0.021, 0.338, 0.16)),
    ("Hybrid",        "#3a3a3a", 1.3, (0.128, 0.312, 0.34)),
    ("Ours",          "#3b5b92", 2.0, (0.186, 0.381, 0.42)),
]


def curves(rng):
    """NDCG@10 per interaction bin, with a bootstrap standard error."""
    k = np.arange(len(BINS))
    out = {}
    for name, _c, _lw, (lo, hi, rate) in STRATEGIES:
        # Saturating growth from the cold value ``lo`` toward the warm value
        # ``hi``; ``rate`` sets how many interactions the strategy needs.
        mu = lo + (hi - lo) * (1.0 - np.exp(-rate * k))
        mu = mu * np.exp(rng.normal(0.0, 0.020, k.size))
        se = mu / np.sqrt(N_SEED * 0.04) + 0.0026
        out[name] = (mu, se)
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(475)
    data = curves(rng)
    k = np.arange(len(BINS))

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4), sharex=True,
                             sharey=True)
    lo, hi = shared_limits(
        [m + 1.96 * s for m, s in data.values()] +
        [m - 1.96 * s for m, s in data.values()], pad=0.09)
    lo = min(lo, 0.0)

    for ax, (name, color, lw, _p) in zip(axes.ravel(), STRATEGIES):
        ax.axvspan(-0.5, COLD - 0.5, facecolor="#f0f0f0", edgecolor="none",
                   zorder=0)
        for other, ocol, _olw, _op in STRATEGIES:
            if other == name:
                continue
            ax.plot(k, data[other][0], lw=0.8, color="#d5d5d5", zorder=2)

        mu, se = data[name]
        ax.fill_between(k, mu - 1.96 * se, mu + 1.96 * se,
                        facecolor=lighten(color, 0.68), edgecolor="none",
                        zorder=3)
        ax.plot(k, mu, lw=lw, color=color, marker="o", ms=3.2, mfc="white",
                mew=1.0, mec=color, zorder=4)

        cold = float(mu[:COLD].mean())
        warm = float(mu[COLD:].mean())
        ax.text(0.035, 0.955,
                f"cold {cold:.3f}   warm {warm:.3f}\n"
                f"gap {warm - cold:+.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                color=SUB, linespacing=1.35, zorder=6)
        ax.set_title(name, fontsize=9.0, fontweight="bold",
                     color=INK if name != "Ours" else color, pad=3.0)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.5, len(BINS) - 0.5)
        ax.set_xticks(k)
        ax.set_xticklabels(BINS, rotation=30, ha="right")
        ax.tick_params(labelsize=6.8, length=2.5)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0, 0].text(0.035, 0.20, "cold-start bins", transform=axes[0, 0].transAxes,
                    ha="left", va="center", fontsize=6.4, color=MUTED,
                    fontstyle="italic", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor=HAIR, linewidth=0.5, alpha=0.94))
    for ax in axes[:, 0]:
        ax.set_ylabel("NDCG@10 (-)", fontsize=8.0)
    for ax in axes[1, :]:
        ax.set_xlabel("prior interactions per user (-)", fontsize=8.0)

    ours = data["Ours"][0]
    best_base = np.max([data[n][0] for n, _c, _l, _p in STRATEGIES[:-1]], axis=0)
    lift = float((ours[:COLD] / best_base[:COLD]).mean() - 1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, axes, dx=-0.46, dy=0.08)
    savefig(fig, "475_rec_cold_start.png")


if __name__ == "__main__":
    main()
