"""Figure 471 - Document clustering under three text representations.

The same 320 documents, embedded three ways and projected to two dimensions.
Each projection is standardised to unit variance before plotting and all three
panels are then given one shared x and y range, computed once over the three
point sets - without that, matplotlib would stretch the worst representation
to fill its frame and TF-IDF would look as tidy as Sentence-BERT.

ARI and NMI come from k-means run on the plotted coordinates against the true
labels, and the dendrogram on the right is Ward linkage over the twelve
sub-topic centroids of the Sentence-BERT space.

Canvas aspect is 0.44, outside the 0.55-0.80 band of DESIGN_SYSTEM section
1.2; this is exception E8, a single-row comparison grid with four columns.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from pubstyle import set_pub_style, savefig, darken, annotate_box
from figkit import shared_limits, panel_labels_inches

# ---------------------------------------------------------------------------
# Shared NLP-batch vocabulary (464-473)
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d9d9d9"

CLASSES = ["sports", "politics", "technology", "health"]
CLASS_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227"]
SUBTOPICS = {
    "sports": ["football", "tennis", "cycling"],
    "politics": ["elections", "budget", "treaty"],
    "technology": ["ai models", "chips", "privacy"],
    "health": ["vaccines", "nutrition", "sleep"],
}
REPS = [("TF-IDF", 1.55), ("Doc2Vec", 1.02), ("Sentence-BERT", 0.60)]
DIM = 24
N_PER_SUB = 27


def build(rng):
    """Latent sub-topic structure, then one embedding per representation."""
    centres = rng.standard_normal((len(CLASSES), DIM)) * 1.9
    subs, labels, sub_names = [], [], []
    for ci, cls in enumerate(CLASSES):
        for name in SUBTOPICS[cls]:
            subs.append(centres[ci] + 0.72 * rng.standard_normal(DIM))
            sub_names.append(f"{cls} / {name}")
            labels += [ci] * N_PER_SUB
    subs = np.vstack(subs)
    labels = np.array(labels)

    projections = {}
    for name, sigma in REPS:
        X = np.repeat(subs, N_PER_SUB, axis=0)
        X = X + sigma * rng.standard_normal(X.shape)
        Xc = X - X.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
        P = Xc @ vt[:2].T
        projections[name] = P / P.std()          # common arbitrary scale
    return projections, labels, subs, sub_names


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(471)
    proj, labels, subs, sub_names = build(rng)
    n_docs = labels.size

    fig = plt.figure(figsize=(10.4, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.74],
                          left=0.055, right=0.845, top=0.815, bottom=0.135,
                          wspace=0.24)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axd = fig.add_subplot(gs[0, 3])

    xlim = shared_limits([p[:, 0] for p in proj.values()], pad=0.08)
    ylim = shared_limits([p[:, 1] for p in proj.values()], pad=0.08)

    for ax, (name, _s) in zip(axes, REPS):
        P = proj[name]
        for ci, cls in enumerate(CLASSES):
            m = labels == ci
            ax.scatter(P[m, 0], P[m, 1], s=11, color=CLASS_COLORS[ci],
                       alpha=0.80, edgecolor="white", linewidth=0.25,
                       zorder=3, label=cls if ax is axes[0] else None)
        km = KMeans(n_clusters=len(CLASSES), n_init=10,
                    random_state=0).fit_predict(P)
        ari = adjusted_rand_score(labels, km)
        nmi = normalized_mutual_info_score(labels, km)
        annotate_box(ax, f"ARI {ari:.3f}\nNMI {nmi:.3f}", loc="lower right",
                     fontsize=6.6, fontfamily="monospace")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("component 1 (s.d.)", fontsize=8.0)
        ax.tick_params(labelsize=7.0, length=2.5)
        ax.grid(color=GRID, linewidth=0.6, zorder=0)
        ax.set_title(name, fontsize=9.0, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("component 2 (s.d.)", fontsize=8.0)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # ---- (d) Ward dendrogram over the sub-topic centroids -----------------
    # colour every link by the class of the leaves underneath it, so the
    # dendrogram uses exactly the same hue per topic as the scatters do
    Z = linkage(subs, method="ward")
    n_leaf = subs.shape[0]
    members = {i: {CLASSES.index(s.split(" / ")[0])}
               for i, s in enumerate(sub_names)}
    for k, row in enumerate(Z):
        members[n_leaf + k] = members[int(row[0])] | members[int(row[1])]

    def link_colour(node):
        m = members[node]
        return (darken(CLASS_COLORS[next(iter(m))], 0.05) if len(m) == 1
                else "#9a9a9a")

    dendrogram(Z, ax=axd, orientation="left", labels=sub_names,
               link_color_func=link_colour, leaf_font_size=6.2)
    for lab in axd.get_ymajorticklabels():
        cls = lab.get_text().split(" / ")[0]
        lab.set_color(darken(CLASS_COLORS[CLASSES.index(cls)], 0.18))
    axd.tick_params(axis="y", length=0, pad=1.5)
    axd.tick_params(axis="x", labelsize=6.5, length=2.2)
    axd.set_xlabel("Ward linkage distance (-)", fontsize=7.0)
    axd.set_title("Sub-topic hierarchy", fontsize=8.5, fontweight="bold",
                  pad=3.0)
    axd.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    axd.yaxis.set_ticks_position("right")
    for s in ("top", "right", "left"):
        axd.spines[s].set_visible(False)

    handles, lab = axes[0].get_legend_handles_labels()

    best = max(REPS, key=lambda r: adjusted_rand_score(
        labels, KMeans(n_clusters=len(CLASSES), n_init=10,
                       random_state=0).fit_predict(proj[r[0]])))[0]
    panel_labels_inches(fig, axes + [axd], dx=-0.42, dy=0.08)
    savefig(fig, "471_nlp_text_clustering.png")


if __name__ == "__main__":
    main()
