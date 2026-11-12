"""GO enrichment bubble plot, facetted by ontology.

Classic over-representation figure: pathways on y ranked by significance,
GeneRatio on x, bubble area = gene count, bubble colour = -log10(FDR). A
narrow companion panel on the right shows fold enrichment as bars, and the
three GO branches (BP / MF / CC) are stacked as facets sharing one colourbar
and one bubble-size key.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

CMAP = LinearSegmentedColormap.from_list(
    "enrich", ["#fbe6d4", "#f2b880", "#d97b53", "#a83c46", "#5f1f33"])
BAR_C = "#7d95b3"

FACETS = [
    ("Biological Process", "#2a7f7a", [
        "Response to type I interferon",
        "Extracellular matrix organization",
        "Regulation of inflammatory response",
        "Cell cycle G2/M phase transition",
        "Leukocyte chemotaxis",
        "Fatty acid beta-oxidation",
        "Response to oxidative stress",
        "Epithelial cell proliferation",
    ]),
    ("Molecular Function", "#c0603f", [
        "Cytokine receptor binding",
        "Extracellular matrix structural constituent",
        "Protein serine/threonine kinase activity",
        "Oxidoreductase activity, NAD(P)H",
        "Heme binding",
        "Transcription coactivator binding",
    ]),
    ("Cellular Component", "#5b6e9e", [
        "Collagen-containing extracellular matrix",
        "Mitochondrial inner membrane",
        "Chromosome, centromeric region",
        "Endoplasmic reticulum lumen",
        "Focal adhesion",
    ]),
]


def synth(rng, n):
    """Ranked terms: significance, gene ratio and fold enrichment decay
    together, as in a real over-representation table."""
    rank = np.arange(n)
    nlp = np.sort(rng.uniform(2.0, 17.0, n))[::-1] * (1 - 0.015 * rank)
    ratio = 0.07 + 0.30 * (nlp / nlp.max()) ** 0.85 * rng.uniform(0.82, 1.12, n)
    count = np.round(ratio * rng.uniform(110, 330, n)).astype(int)
    fold = 1.4 + 4.4 * (nlp / nlp.max()) * rng.uniform(0.7, 1.2, n)
    return nlp, np.clip(ratio, 0.05, 0.42), np.clip(count, 8, 95), fold


def size_from_count(c):
    return 14 + (np.asarray(c, dtype=float) - 6) * 3.1


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(11)

    data = [synth(rng, len(terms)) for _, _, terms in FACETS]
    all_nlp = np.concatenate([d[0] for d in data])
    norm = Normalize(vmin=np.floor(all_nlp.min()), vmax=np.ceil(all_nlp.max()))
    fold_max = max(d[3].max() for d in data) * 1.30

    fig = plt.figure(figsize=(8.8, 7.6))
    gs = fig.add_gridspec(
        3, 2, left=0.275, right=0.795, top=0.885, bottom=0.075,
        width_ratios=[1.0, 0.30],
        height_ratios=[len(f[2]) for f in FACETS],
        wspace=0.055, hspace=0.30)

    for fi, ((fname, fcolor, terms), (nlp, ratio, count, fold)) in enumerate(
            zip(FACETS, data)):
        n = len(terms)
        y = np.arange(n)[::-1]

        ax = fig.add_subplot(gs[fi, 0])
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e6e8ec", lw=0.6)
        for yy in y:
            ax.plot([0, 1], [yy, yy], transform=ax.get_yaxis_transform(),
                    color="#eceef1", lw=0.5, zorder=0)
        sc = ax.scatter(ratio, y, s=size_from_count(count), c=nlp, cmap=CMAP,
                        norm=norm, edgecolor="#3a3a3a", linewidth=0.45,
                        zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(terms, fontsize=7.2)
        ax.set_ylim(-0.75, n - 0.25)
        ax.set_xlim(0.045, 0.44)
        ax.tick_params(axis="y", length=0)
        style_axes(ax)
        ax.spines["left"].set_visible(False)
        add_panel_label(ax, f"({chr(97 + fi)})", x=-0.395, y=1.0, size=10)
        ax.set_title(fname, fontsize=8.8, fontweight="bold", color=fcolor,
                     loc="left", pad=4)
        if fi == len(FACETS) - 1:
            ax.set_xlabel("Gene ratio (DE genes / term size)")
        else:
            ax.tick_params(axis="x", labelbottom=False)

        axb = fig.add_subplot(gs[fi, 1])
        axb.barh(y, fold, height=0.55, color=BAR_C, edgecolor="#33475e",
                 lw=0.45)
        for yy, fv in zip(y, fold):
            axb.text(fv + fold_max * 0.035, yy, f"{fv:.1f}", va="center",
                     ha="left", fontsize=6.2, color="#33475e")
        axb.set_ylim(-0.75, n - 0.25)
        axb.set_xlim(0, fold_max)
        axb.set_yticks([])
        axb.axvline(1, color="#999999", ls=(0, (3, 2)), lw=0.6)
        style_axes(axb)
        axb.spines["left"].set_visible(False)
        if fi == len(FACETS) - 1:
            axb.set_xlabel("Fold\nenrichment", fontsize=7.5)
            axb.set_xticks([0, 2, 4, 6])
        else:
            axb.tick_params(axis="x", labelbottom=False)
            axb.set_xticks([0, 2, 4, 6])

    # ---- shared colourbar --------------------------------------------------
    cax = fig.add_axes([0.845, 0.585, 0.017, 0.235])
    cb = fig.colorbar(sc, cax=cax)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(length=2.5, labelsize=7)
    cb.ax.set_title("-log$_{10}$\nFDR", fontsize=7.8, fontweight="bold",
                    pad=6, loc="left")

    # ---- shared bubble-size key -------------------------------------------
    axk = fig.add_axes([0.828, 0.24, 0.16, 0.24]); axk.axis("off")
    keys = [10, 30, 55, 85]
    handles = [Line2D([], [], marker="o", ls="", mfc="#e2e5ea",
                      mec="#3a3a3a", mew=0.45,
                      ms=np.sqrt(size_from_count(k)), label=str(k))
               for k in keys]
    leg = axk.legend(handles=handles, title="Gene count", loc="upper left",
                     bbox_to_anchor=(0.0, 1.0), fontsize=7,
                     labelspacing=1.05, handletextpad=0.9, borderpad=0.0)
    leg.get_title().set_fontsize(7.8)
    leg.get_title().set_fontweight("bold")
    leg._legend_box.align = "left"

    savefig(fig, "73_enrichment_bubble.png")


if __name__ == "__main__":
    main()
