"""Microbiome composition: stacked relative abundance with alpha diversity.

One stacked bar per sample with a fixed taxon order, samples blocked by
treatment group and flagged by a top colour band. A shared taxon legend sits
on the right, and the lower panel shares the same x-axis so each Shannon
diversity violin sits directly beneath its group.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

TAXA = ["Bacteroides", "Prevotella", "Faecalibacterium", "Ruminococcus",
        "Bifidobacterium", "Akkermansia", "Blautia", "Lactobacillus",
        "Escherichia", "Clostridium", "Other taxa"]
TAXA_C = ["#3f5d80", "#7fa3c3", "#bfd3e3", "#6b8f5e", "#a9c48f", "#c9a227",
          "#e3c98a", "#8c6b4f", "#b23a48", "#d99494", "#bcbcbc"]

GROUPS = ["Healthy control", "Antibiotic", "Antibiotic + probiotic", "Recovery"]
GROUP_C = ["#2a7f7a", "#c0603f", "#5b6e9e", "#7a9a53"]
N_PER = 15

# Dirichlet concentration per group: dysbiosis rises then partially resolves
ALPHAS = np.array([
    [7.0, 4.0, 6.5, 3.5, 2.5, 2.0, 3.0, 1.2, 0.5, 1.5, 1.6],
    [3.0, 1.0, 1.2, 1.0, 0.6, 0.4, 0.9, 2.6, 6.5, 3.2, 2.4],
    [4.5, 1.8, 3.0, 1.6, 4.5, 1.1, 1.8, 5.0, 2.2, 2.0, 2.0],
    [6.0, 3.2, 5.2, 2.8, 2.2, 1.7, 2.6, 1.6, 1.1, 1.8, 1.8],
])


def shannon(p):
    q = np.clip(p, 1e-12, None)
    return -np.sum(q * np.log(q), axis=1)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(88)

    comps, groups = [], []
    for g, alpha in enumerate(ALPHAS):
        c = rng.dirichlet(alpha * 2.2, N_PER)
        c = c[np.argsort(c[:, 0])]          # tidy gradient inside each block
        comps.append(c)
        groups.append(np.full(N_PER, g))
    comp = np.vstack(comps)
    group = np.concatenate(groups)
    n = comp.shape[0]
    div = shannon(comp)

    fig = plt.figure(figsize=(9.6, 6.3))
    gs = fig.add_gridspec(3, 1, left=0.075, right=0.775, top=0.845,
                          bottom=0.095, height_ratios=[0.045, 1.0, 0.52],
                          hspace=0.13)

    # ---- group band --------------------------------------------------------
    axb = fig.add_subplot(gs[0])
    for g in range(len(GROUPS)):
        idx = np.where(group == g)[0]
        axb.barh(0, idx.size, left=idx[0], height=1.0, color=GROUP_C[g],
                 edgecolor="white", lw=1.2)
        axb.text(idx.mean() + 0.5, 0, GROUPS[g], ha="center", va="center",
                 fontsize=7.4, fontweight="bold", color="white")
    axb.set_xlim(-0.5, n - 0.5)
    axb.set_ylim(-0.5, 0.5)
    axb.axis("off")

    # ---- stacked composition ----------------------------------------------
    ax = fig.add_subplot(gs[1])
    bottom = np.zeros(n)
    x = np.arange(n)
    for k, (name, c) in enumerate(zip(TAXA, TAXA_C)):
        ax.bar(x, comp[:, k], bottom=bottom, width=0.86, color=c,
               edgecolor="white", lw=0.28, label=name)
        bottom += comp[:, k]
    for g in range(1, len(GROUPS)):
        ax.axvline(np.where(group == g)[0][0] - 0.5, color="#333333", lw=0.8)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("Relative abundance (%)")
    ax.set_xticks([])
    style_axes(ax)
    ax.spines["bottom"].set_visible(False)
    add_panel_label(ax, "(a)", x=-0.068, y=1.10)

    # ---- alpha diversity, aligned to the same x-axis -----------------------
    axd = fig.add_subplot(gs[2], sharex=ax)
    centres, data = [], []
    for g in range(len(GROUPS)):
        idx = np.where(group == g)[0]
        centres.append(idx.mean())
        data.append(div[idx])
    parts = axd.violinplot(data, positions=centres, widths=N_PER * 0.74,
                           showextrema=False, showmedians=False,
                           bw_method=0.55)
    for body, c in zip(parts["bodies"], GROUP_C):
        body.set_facecolor(c); body.set_alpha(0.28)
        body.set_edgecolor(c); body.set_linewidth(0.9)
    for g, (ctr, vals) in enumerate(zip(centres, data)):
        jitter = ctr + rng.uniform(-1.6, 1.6, vals.size)
        axd.scatter(jitter, vals, s=9, color=GROUP_C[g], alpha=0.85, lw=0.3,
                    edgecolor="white", zorder=3)
        med = np.median(vals)
        axd.plot([ctr - 2.6, ctr + 2.6], [med, med], color=GROUP_C[g], lw=1.8,
                 zorder=4)
    kw_p = stats.kruskal(*data).pvalue
    axd.text(0.995, 0.06, f"Kruskal-Wallis $P$ = {kw_p:.1e}",
             transform=axd.transAxes, ha="right", va="bottom", fontsize=7.2,
             fontweight="bold", color="#1a1a1a")
    axd.set_ylabel("Shannon index")
    axd.set_xlabel(f"Gut microbiome samples (n = {n}), "
                   "ordered by Bacteroides abundance within group")
    axd.set_xticks([])
    style_axes(axd)
    axd.spines["bottom"].set_visible(False)
    add_panel_label(axd, "(b)", x=-0.068, y=1.02)

    # ---- taxon legend ------------------------------------------------------
    axl = fig.add_axes([0.79, 0.30, 0.20, 0.52]); axl.axis("off")
    # reversed so the legend order matches the top-to-bottom stacking order
    handles = [Patch(facecolor=c, edgecolor="white", lw=0.4, label=t)
               for t, c in zip(TAXA[::-1], TAXA_C[::-1])]
    leg = axl.legend(handles=handles, title="Genus", loc="upper left",
                     bbox_to_anchor=(0.0, 1.0), fontsize=7.2,
                     handlelength=1.1, handleheight=1.0, labelspacing=0.42,
                     borderpad=0.0, handletextpad=0.6)
    leg.get_title().set_fontsize(7.8)
    leg.get_title().set_fontweight("bold")
    leg._legend_box.align = "left"

    savefig(fig, "76_stacked_composition.png")


if __name__ == "__main__":
    main()
