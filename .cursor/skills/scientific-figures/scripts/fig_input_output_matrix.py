"""Inter-sectoral input-output matrix with clustered ordering.

The heatmap shows intermediate flows from supplying sectors (rows) to using
sectors (columns) on a logarithmic colour scale.  Sectors are reordered by
hierarchical clustering of their trade profiles, so supply blocks appear as
contiguous squares; the own-sector diagonal is outlined, the largest flows
are labelled, and marginal bars give total sales (right) and total
intermediate purchases (top).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist

from pubstyle import set_pub_style, PALETTES, savefig, style_axes

SECTORS = [
    "Agriculture", "Mining", "Food processing", "Textiles",
    "Wood & paper", "Chemicals", "Basic metals", "Machinery",
    "Electronics", "Utilities", "Construction", "Wholesale trade",
    "Transport", "Finance & ins.", "Business services",
]
BLOCK = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2])


def build_table(rng):
    """Synthetic intermediate-flow table, billion USD."""
    n = len(SECTORS)
    scale_out = np.exp(rng.normal(0.0, 0.55, n)) * 60.0
    scale_in = np.exp(rng.normal(0.0, 0.50, n)) * 60.0
    affinity = np.where(BLOCK[:, None] == BLOCK[None, :], 1.0, 0.28)
    upstream = np.array([1.5, 1.6, 0.8, 0.7, 0.9, 1.4, 1.5, 0.9,
                         0.9, 1.3, 0.4, 0.8, 1.0, 0.9, 1.1])
    z = (np.outer(scale_out * upstream, scale_in) / 1000.0 * affinity
         * np.exp(rng.normal(0, 0.45, (n, n))))
    z += np.diag(np.diag(z) * 2.4)                       # own-sector use
    z[BLOCK == 2, :] *= 1.25                             # services sell wide
    z[:, BLOCK == 2] *= 0.85
    return np.clip(z, 0.06, None)


def cluster_order(z):
    prof = np.log(z + 0.05)
    prof = (prof - prof.mean(axis=1, keepdims=True)) / prof.std(axis=1,
                                                                keepdims=True)
    feat = np.hstack([prof, prof.T])
    link = linkage(pdist(feat, metric="euclidean"), method="average")
    return leaves_list(link), fcluster(link, 3, criterion="maxclust")


def main():
    set_pub_style()
    rng = np.random.default_rng(19)
    z = build_table(rng)
    order, clust = cluster_order(z)
    z = z[np.ix_(order, order)]
    names = [SECTORS[i] for i in order]
    clust = clust[order]
    n = len(names)

    cmap = LinearSegmentedColormap.from_list(
        "io", PALETTES["cb_sequential_blue"], N=256)
    norm = LogNorm(vmin=max(z.min(), 0.06), vmax=z.max())

    fig = plt.figure(figsize=(7.4, 6.4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 0.20, 0.032],
                  height_ratios=[0.20, 1.0], wspace=0.045, hspace=0.045,
                  left=0.175, right=0.945, top=0.925, bottom=0.185)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    cax = fig.add_subplot(gs[1, 2])

    im = ax.imshow(z, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")

    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="#8c2f2f", lw=1.0, zorder=4))

    # cluster block separators
    bounds = np.where(np.diff(clust) != 0)[0] + 0.5
    for b in bounds:
        ax.axhline(b, color="white", lw=1.6)
        ax.axvline(b, color="white", lw=1.6)

    # label the largest own-sector and cross-sector flows
    ranked = np.dstack(np.unravel_index(np.argsort(z, axis=None)[::-1],
                                        z.shape))[0]
    diag = [(i, j) for i, j in ranked if i == j][:3]
    off = [(i, j) for i, j in ranked if i != j][:5]
    for i, j in diag + off:
        val = z[i, j]
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.2,
                color="white" if norm(val) > 0.62 else "#1a1a1a",
                fontweight="bold", zorder=5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.6,
                       rotation_mode="anchor")
    ax.set_yticklabels(names, fontsize=6.6)
    ax.set_xlabel("Using sector (intermediate demand)", labelpad=4)
    ax.set_ylabel("Supplying sector (intermediate sales)", labelpad=4)
    ax.tick_params(length=1.8)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_edgecolor("#333333")

    col_tot = z.sum(axis=0)
    row_tot = z.sum(axis=1)
    ax_top.bar(np.arange(n), col_tot, width=0.72, color="#7f9db9",
               edgecolor="white", lw=0.4)
    ax_top.set_ylabel("Purchases\n(bn USD)", fontsize=6.8, linespacing=1.3)
    ax_top.tick_params(axis="x", labelbottom=False, length=0)
    ax_top.tick_params(axis="y", labelsize=6.2)
    ax_top.set_ylim(0, col_tot.max() * 1.18)
    ax_top.set_yticks([0, int(round(col_tot.max() / 100.0) * 50),
                       int(round(col_tot.max() / 100.0) * 100)])
    ax_top.grid(axis="y", lw=0.4, color="#ececec")
    ax_top.set_axisbelow(True)
    style_axes(ax_top)

    ax_right.barh(np.arange(n), row_tot, height=0.72, color="#c39b6b",
                  edgecolor="white", lw=0.4)
    ax_right.set_xlabel("Sales\n(bn USD)", fontsize=6.8, linespacing=1.3)
    ax_right.tick_params(axis="y", labelleft=False, length=0)
    ax_right.tick_params(axis="x", labelsize=6.2)
    ax_right.set_xlim(0, row_tot.max() * 1.18)
    ax_right.set_xticks([0, int(round(row_tot.max() / 100.0) * 100)])
    ax_right.grid(axis="x", lw=0.4, color="#ececec")
    ax_right.set_axisbelow(True)
    style_axes(ax_right)

    # colour tick labels by supply block and legend them in the free corner
    block_names = {0: "Primary & energy", 1: "Manufacturing",
                   2: "Services & construction"}
    block_cols = {0: "#5b8c5a", 1: "#a8624a", 2: "#4a6b8f"}
    cl2block = {}
    for c in np.unique(clust):
        members = BLOCK[np.array(order)[clust == c]]
        cl2block[c] = int(np.bincount(members).argmax())
    for tick, c in zip(ax.get_xticklabels(), clust):
        tick.set_color(block_cols[cl2block[c]])
    for tick, c in zip(ax.get_yticklabels(), clust):
        tick.set_color(block_cols[cl2block[c]])

    ax_leg = fig.add_subplot(gs[0, 1:])
    ax_leg.axis("off")
    seen = []
    for c in np.unique(clust):
        b = cl2block[c]
        if b not in seen:
            seen.append(b)
    handles = [Rectangle((0, 0), 1, 1, fc=block_cols[b], ec="none",
                         label=block_names[b]) for b in sorted(seen)]
    ax_leg.legend(handles=handles, loc="center left", fontsize=6.4,
                  frameon=False, handlelength=0.9, handleheight=0.9,
                  labelspacing=0.55, borderaxespad=0.0,
                  bbox_to_anchor=(-0.05, 0.42), title="Supply block",
                  title_fontsize=6.6)

    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.2)
    cb.set_label("Intermediate flow (billion USD, log scale)", fontsize=7)

    ax.text(0.0, 1.30, "Input-output table, 15 sectors, 2024",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            ha="left", va="bottom")
    ax.text(0.0, 1.235,
            "Rows and columns reordered by hierarchical clustering; white "
            "rules separate supply blocks, red squares mark own-sector use.",
            transform=ax.transAxes, fontsize=6.8, color="#6f6f6f",
            ha="left", va="bottom")
    savefig(fig, "107_input_output_matrix.png")


if __name__ == "__main__":
    main()
