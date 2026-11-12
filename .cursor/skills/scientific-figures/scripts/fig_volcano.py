"""Volcano plot: log2 fold change vs -log10(p) with significance thresholds.

Two panels (two treatments vs the same control). Fold changes and p-values come
from a two-group differential-expression model - a null majority plus a sparse
set of truly regulated genes, each with its own standard error - so the point
cloud shows the characteristic volcano flanks instead of a rectangular blob.
The horizontal cut is the Benjamini-Hochberg 5% FDR threshold.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

DOWN, UP = PALETTES["blue_red"]
NS = "#c4c4c4"
FC_CUT, Q_CUT = 1.0, 0.05
SYMBOLS = ["SLC2A1", "HIF1A", "VEGFA", "MYC", "CDKN1A", "TP53", "EGFR",
           "FOXO3", "MMP9", "IL6", "STAT3", "PTEN", "BCL2", "CASP3",
           "NFKB1", "SOD2", "GAPDH", "ACTB", "RPS6", "MTOR", "AKT1",
           "JUN", "FOS", "ATF4", "XBP1", "HSPA5", "DDIT3", "PPARG"]


def simulate(rng, n=2500, frac_de=0.09, effect_sd=1.15, df=8):
    """Two-group contrast with n_rep = 5 per group -> t-distributed statistic."""
    is_de = rng.random(n) < frac_de
    true_fc = np.where(is_de, rng.normal(0, effect_sd, n), 0.0)
    se = 0.16 + 0.5 * rng.gamma(1.6, 0.14, n)        # gene-wise precision
    fc = true_fc + rng.normal(0, se)
    p = 2 * t_dist.sf(np.abs(fc / se), df)
    return fc, p


def bh_threshold(p, q=Q_CUT):
    m = p.size
    order = np.sort(p)
    ok = order <= q * np.arange(1, m + 1) / m
    return order[ok].max() if ok.any() else 0.0


def place_labels(ax, xs, ys, names, xlim, ylim, min_dx=0.085, min_dy=0.055):
    used = []
    for x, y, nm in zip(xs, ys, names):
        fx = (x - xlim[0]) / (xlim[1] - xlim[0])
        fy = (y - ylim[0]) / (ylim[1] - ylim[0])
        if any(abs(fx - ux) < min_dx and abs(fy - uy) < min_dy
               for ux, uy in used):
            continue
        used.append((fx, fy))
        ax.annotate(nm, (x, y), textcoords="offset points",
                    xytext=(4 if x > 0 else -4, 3), fontsize=6.5,
                    ha="left" if x > 0 else "right", color="#333333",
                    zorder=6)


def volcano(ax, rng, idx, title):
    fc, p = simulate(rng)
    y = -np.log10(p)
    p_cut = bh_threshold(p)
    y_cut = -np.log10(p_cut) if p_cut > 0 else np.inf

    up = (fc > FC_CUT) & (p <= p_cut)
    dn = (fc < -FC_CUT) & (p <= p_cut)
    ns = ~(up | dn)

    xlim = (-4.4, 4.4)
    ylim = (0, max(y.max() * 1.12, 6))
    ax.axhline(y_cut, color="#888888", ls="--", lw=0.8, zorder=2)
    for v in (-FC_CUT, FC_CUT):
        ax.axvline(v, color="#888888", ls=":", lw=0.8, zorder=2)

    ax.scatter(fc[ns], y[ns], s=7, color=NS, alpha=0.45, edgecolors="none",
               zorder=3)
    ax.scatter(fc[dn], y[dn], s=13, color=DOWN, alpha=0.8, linewidths=0.25,
               edgecolors="white", zorder=4,
               label=f"Down-regulated (n = {dn.sum()})")
    ax.scatter(fc[up], y[up], s=13, color=UP, alpha=0.8, linewidths=0.25,
               edgecolors="white", zorder=4,
               label=f"Up-regulated (n = {up.sum()})")

    sig = np.where(up | dn)[0]
    rank = sig[np.argsort(-(y[sig] * np.abs(fc[sig])))][:14]
    place_labels(ax, fc[rank], y[rank],
                 [SYMBOLS[i % len(SYMBOLS)] for i in range(len(rank))],
                 xlim, ylim)

    ax.text(xlim[1] * 0.985, y_cut, f" BH FDR {Q_CUT:.0%}  (p = {p_cut:.1e})",
            ha="right", va="bottom", fontsize=6.8, color="#777777")
    ax.text(FC_CUT, ylim[1] * 0.985, f" |log$_2$FC| = {FC_CUT:.0f}",
            ha="left", va="top", fontsize=6.8, color="#777777")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"log$_2$ fold change (treatment / control)", fontsize=8.5)
    ax.set_ylabel(r"$-\log_{10}$ (p-value)", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    leg = ax.legend(fontsize=7, frameon=True, loc="upper left",
                    facecolor="white", edgecolor="#cccccc", framealpha=0.92,
                    handletextpad=0.4, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)
    ax.grid(lw=0.5, alpha=0.3)
    style_axes(ax)
    add_panel_label(ax, f"({'ab'[idx]})", x=-0.11, y=1.02, size=11)


def main():
    set_pub_style()
    rng = np.random.default_rng(341)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.0))
    volcano(axes[0], rng, 0, "Treatment A vs control")
    volcano(axes[1], rng, 1, "Treatment B vs control")
    fig.tight_layout(rect=[0, 0.035, 1, 1], w_pad=3)
    savefig(fig, "38_volcano.png")


if __name__ == "__main__":
    main()
