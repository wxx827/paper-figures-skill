"""Clustered heatmap: correlation matrix with side dendrograms.

A seaborn-clustermap look built manually with scipy hierarchical clustering.
The matrix is the Pearson correlation of 120 simulated observations with a
three-block structure, so the dendrogram recovers real clusters instead of
noise.  Cluster membership is repeated as a colour ribbon next to each axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from pubstyle import set_pub_style, PALETTES, savefig

CMAP = LinearSegmentedColormap.from_list("rdbu", PALETTES["div_rdbu"][::-1])
CLUSTER_COLORS = ["#2e7d5b", "#c9772f", "#3d6b8e"]
N_OBS = 120

BLOCKS = [
    (["NDVI", "LAI", "ET", "RUNOFF"], 0.86),        # water-energy block
    (["TEM", "SLP", "ELEV", "ALBEDO"], 0.80),       # terrain-climate block
    (["POP", "GDP", "SOIL", "PRE"], 0.78),          # pressure block
]


def simulate(rng):
    cols, labels = [], []
    for members, load in BLOCKS:
        latent = rng.normal(0, 1, N_OBS)
        for k, name in enumerate(members):
            sign = -1.0 if k == len(members) - 1 else 1.0
            cols.append(sign * load * latent
                        + np.sqrt(1 - load ** 2) * rng.normal(0, 1, N_OBS))
            labels.append(name)
    return np.corrcoef(np.column_stack(cols), rowvar=False), labels


def main():
    set_pub_style()
    rng = np.random.default_rng(51)
    M, labels = simulate(rng)
    n = len(labels)

    dist = squareform(np.clip(1.0 - M, 0, 2), checks=False)
    Z = linkage(dist, method="average")
    clusters = fcluster(Z, t=3, criterion="maxclust")

    fig = plt.figure(figsize=(8.4, 8.0))
    gs = GridSpec(3, 3, width_ratios=[0.20, 0.035, 1],
                  height_ratios=[0.20, 0.035, 1], wspace=0.03, hspace=0.03,
                  left=0.055, right=0.90, top=0.90, bottom=0.10)

    ax_top = fig.add_subplot(gs[0, 2])
    dn = dendrogram(Z, ax=ax_top, color_threshold=0,
                    above_threshold_color="#666666", no_labels=True)
    ax_top.axis("off")
    order = dn["leaves"]

    ax_left = fig.add_subplot(gs[2, 0])
    dendrogram(Z, ax=ax_left, orientation="left", color_threshold=0,
               above_threshold_color="#666666", no_labels=True)
    ax_left.invert_yaxis()          # match the row order of the heatmap
    ax_left.axis("off")

    Mo = M[np.ix_(order, order)]
    lab_o = [labels[i] for i in order]
    cl_o = clusters[order]

    # cluster ribbons
    ax_cr = fig.add_subplot(gs[2, 1])
    ax_cr.imshow(np.array([[c] for c in cl_o]), aspect="auto",
                 cmap=LinearSegmentedColormap.from_list("cl", CLUSTER_COLORS),
                 vmin=1, vmax=3)
    ax_cr.set_xticks([])
    ax_cr.set_yticks([])
    ax_cc = fig.add_subplot(gs[1, 2])
    ax_cc.imshow(np.array([cl_o]), aspect="auto",
                 cmap=LinearSegmentedColormap.from_list("cl", CLUSTER_COLORS),
                 vmin=1, vmax=3)
    ax_cc.set_xticks([])
    ax_cc.set_yticks([])
    for a in (ax_cr, ax_cc):
        for s in a.spines.values():
            s.set_visible(False)

    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(Mo, cmap=CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(lab_o, rotation=90, fontsize=7.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(lab_o, fontsize=7.5)
    ax.yaxis.tick_right()               # keep clear of the cluster ribbon
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.7)
    ax.tick_params(which="both", length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = Mo[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center",
                    va="center", fontsize=6.0,
                    color="white" if abs(v) > 0.62 else "#333333")

    cax = fig.add_axes([0.925, 0.10, 0.018, 0.42])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson correlation coefficient, r", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")

    handles = [Patch(facecolor=CLUSTER_COLORS[k], label=f"Cluster {k + 1}")
               for k in range(3)]

    savefig(fig, "14_clustermap.png")


if __name__ == "__main__":
    main()
