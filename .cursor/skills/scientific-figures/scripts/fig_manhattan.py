"""Manhattan plot: genome-wide association -log10(p) by chromosome.

Two traits, six chromosomes. The bulk of the markers are drawn from the null
(uniform p-values), so the carpet sits below -log10(p) = 4 as it does in a real
scan, and a handful of causal loci appear as LD-shaped peaks: a lead SNP with
neighbouring markers whose signal decays with genetic distance. Both the
genome-wide (5e-8) and suggestive (1e-5) thresholds are drawn.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

CHR_LEN = np.array([248, 242, 198, 190, 181, 170], dtype=float)   # Mb
N_SNP = 6000
BAND = ["#2e5f8a", "#8fb0d6"]
GENES = ["PSBA", "RBCL", "NRT2.1", "AMT1", "GS1", "FTSH", "LHCB2", "SUS4"]
SIG, SUGG = 5e-8, 1e-5


def simulate(rng, n_loci):
    """Null-dominated scan with a few LD-shaped peaks."""
    weights = CHR_LEN / CHR_LEN.sum()
    counts = rng.multinomial(N_SNP, weights)
    chrom = np.repeat(np.arange(len(CHR_LEN)), counts)
    pos = np.concatenate([np.sort(rng.uniform(0, L, c))
                          for L, c in zip(CHR_LEN, counts)])
    y = -np.log10(rng.uniform(0, 1, N_SNP))

    hits = []
    for _ in range(n_loci):
        c = rng.integers(0, len(CHR_LEN))
        centre = rng.uniform(20, CHR_LEN[c] - 20)
        peak = rng.uniform(7.8, 15.5)
        m = chrom == c
        d = np.abs(pos - centre)
        blk = m & (d < 2.2)
        y[blk] = np.maximum(y[blk], peak * np.exp(-(d[blk] / 0.55) ** 2)
                            + rng.normal(0, 0.35, blk.sum()))
        lead = np.where(blk)[0]
        if lead.size:
            hits.append(lead[np.argmax(y[lead])])
    return chrom, pos, np.clip(y, 0, None), hits


def manhattan(ax, rng, idx, title, n_loci):
    chrom, pos, y, hits = simulate(rng, n_loci)
    offset = np.concatenate([[0], np.cumsum(CHR_LEN + 12)[:-1]])
    x = pos + offset[chrom]

    for c in range(len(CHR_LEN)):
        m = chrom == c
        if c % 2 == 0:
            ax.axvspan(offset[c] - 6, offset[c] + CHR_LEN[c] + 6,
                       color="#f5f5f5", zorder=0, lw=0)
        ax.scatter(x[m], y[m], s=5.5, color=BAND[c % 2], alpha=0.75,
                   edgecolors="none", rasterized=True, zorder=2)

    ax.axhline(-np.log10(SIG), color="#b23a48", ls="--", lw=0.9, zorder=3)
    ax.axhline(-np.log10(SUGG), color="#b0b0b0", ls=":", lw=0.9, zorder=3)

    used = []
    for k, i in enumerate(sorted(hits, key=lambda j: -y[j])):
        if any(abs(x[i] - ux) < 90 for ux in used):
            continue
        used.append(x[i])
        ax.annotate(GENES[(idx * 4 + k) % len(GENES)], (x[i], y[i]),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=6.8, fontstyle="italic", color="#333333",
                    zorder=6)

    ax.set_xlim(-14, offset[-1] + CHR_LEN[-1] + 14)
    ax.set_ylim(0, max(y.max() * 1.18, 10))
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=8.5)
    ax.set_xticks(offset + CHR_LEN / 2)
    ax.set_xticklabels([str(i + 1) for i in range(len(CHR_LEN))], fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    style_axes(ax)
    add_panel_label(ax, f"({'ab'[idx]})", x=-0.055, y=1.0, size=11)


def main():
    set_pub_style()
    rng = np.random.default_rng(451)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    manhattan(axes[0], rng, 0, "Trait A: ecosystem productivity", 4)
    manhattan(axes[1], rng, 1, "Trait B: nitrogen-use efficiency", 3)
    axes[1].set_xlabel("Chromosome", fontsize=9)

    handles = [Line2D([], [], color="#b23a48", ls="--", lw=0.9,
                      label="Genome-wide significance ($p = 5\\times10^{-8}$)"),
               Line2D([], [], color="#b0b0b0", ls=":", lw=0.9,
                      label="Suggestive ($p = 1\\times10^{-5}$)")]
    fig.tight_layout(rect=[0, 0.03, 1, 0.955], h_pad=2.2)
    savefig(fig, "45_manhattan.png")


if __name__ == "__main__":
    main()
