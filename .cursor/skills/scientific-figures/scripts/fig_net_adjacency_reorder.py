"""Adjacency matrix before and after a data-driven reordering.

Left: the weighted adjacency in the order the nodes arrived, which is the
order most data sets are stored in and shows nothing at all.  Right: the same
matrix under an average-linkage leaf order computed from Jaccard distances
between adjacency rows, where the block-diagonal structure appears.  Nothing
about the graph changed - only the permutation.

The reordering is *derived* from the matrix, never from the planted labels,
and the modularity bars score the partition each ordering implies (four equal
contiguous chunks), so the comparison is between two orderings rather than
between an ordering and the answer key.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist

from pubstyle import (CATEGORICAL, COLORS, get_palette, savefig,
                      set_pub_style)
from figkit import panel_labels_inches, safe_colorbar

SEED = 31
COMM_SIZES = [26, 24, 22, 18]
N_BLOCKS = len(COMM_SIZES)


def planted_partition(sizes, p_in, p_out, seed=0, spread=1.6):
    """Degree-heterogeneous stochastic block model; returns (A, labels).

    The propensities are renormalised within each block so no block comes out
    accidentally sparse, and the symmetric draw is a mirrored upper triangle:
    ``(u + u.T) / 2 < p`` would deliver 2p^2, less than half the density asked.
    """
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    n = labels.size
    theta = rng.gamma(spread, 1.0 / spread, n)
    for c in range(len(sizes)):
        theta[labels == c] /= theta[labels == c].mean()
    same = labels[:, None] == labels[None, :]
    p = np.clip(np.where(same, p_in, p_out) * np.outer(theta, theta), 0, 1)
    u = np.triu(rng.random((n, n)), 1)
    u = u + u.T + np.eye(n)
    A = (u < p).astype(float)
    for v in np.flatnonzero(A.sum(1) == 0):        # no isolates: a zero row
        peers = np.flatnonzero((labels == labels[v]) & (np.arange(n) != v))
        w = rng.choice(peers)                      # has no Jaccard neighbours
        A[v, w] = A[w, v] = 1.0                    # and derails the linkage
    return A, labels


def modularity(A, labels):
    m2 = A.sum()
    q = 0.0
    for c in np.unique(labels):
        idx = labels == c
        q += A[np.ix_(idx, idx)].sum() / m2 - (A[idx].sum() / m2) ** 2
    return q


def contiguous_blocks(n, k):
    """Cut 0..n-1 into k equal contiguous chunks - the partition an ordering
    implies if you simply slice the matrix along the diagonal."""
    return np.repeat(np.arange(k), np.diff(np.round(
        np.linspace(0, n, k + 1)).astype(int)))


def diagonal_profile(W, order):
    """Fraction of edge weight within band |i - j| <= d, as a function of d."""
    M = W[np.ix_(order, order)]
    n = M.shape[0]
    i, j = np.indices(M.shape)
    band = np.abs(i - j)
    total = M.sum()
    d = np.arange(n)
    return d / (n - 1.0), np.array([M[band <= t].sum() for t in d]) / total


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    A, labels = planted_partition(COMM_SIZES, p_in=0.34, p_out=0.020,
                                  seed=SEED)
    n = A.shape[0]
    # weights make the shared colour scale carry information; a 0/1 matrix
    # would need no colorbar at all
    w = np.triu(rng.gamma(2.6, 0.42, (n, n)), 1)
    W = A * (w + w.T)

    raw_order = rng.permutation(n)                     # "as collected"
    # Jaccard between adjacency rows ("do these two nodes know the same
    # people"), Ward linkage.  Average linkage on the same distances chains
    # into one 62-node blob plus singletons - the classic chaining failure.
    link = linkage(pdist(A, metric="jaccard"), method="ward")
    clust_order = leaves_list(link)
    detected = fcluster(link, N_BLOCKS, criterion="maxclust") - 1

    slices = contiguous_blocks(n, N_BLOCKS)
    q_raw = modularity(A, slices[np.argsort(raw_order)])
    q_clu = modularity(A, slices[np.argsort(clust_order)])

    vmax = np.percentile(W[W > 0], 99)
    norm = mpl.colors.Normalize(0.0, vmax)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cb_sequential_blue_deep", ["white"] + get_palette(
            "cb_sequential_blue")[1:])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.62],
                  height_ratios=[1.0, 0.55], hspace=0.40, wspace=0.62)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[0, 2])
    ax_p = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2])

    # ---- (a) / (b) the same matrix under two permutations ----------------
    for ax, order, title in ((ax_a, raw_order, "Order of arrival"),
                             (ax_b, clust_order,
                              "Ward leaf order, 4 blocks detected")):
        im = ax.imshow(W[np.ix_(order, order)], cmap=cmap, norm=norm,
                       interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xticks([0, n // 2, n - 1])
        ax.set_yticks([0, n // 2, n - 1])
        ax.tick_params(labelsize=6.8, length=2.5)
        ax.set_xlabel("node index in this ordering (-)", fontsize=7.5)
        ax.set_ylabel("node index in this ordering (-)", fontsize=7.5)
        ax.set_title(title, fontsize=8.8, loc="left", pad=4)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_color(COLORS["rule"])

    # block boundaries on the clustered matrix only
    bounds = np.flatnonzero(np.diff(detected[clust_order])) + 0.5
    for b in bounds:
        ax_b.axhline(b, color=COLORS["accent"], lw=0.8, ls="--")
        ax_b.axvline(b, color=COLORS["accent"], lw=0.8, ls="--")
    safe_colorbar(fig, im, ax_b, side="right", width=0.12, pad=0.08,
                  label="edge weight (-)")

    # ---- (c) modularity of the implied contiguous partition --------------
    bars = ax_q.bar([0, 1], [q_raw, q_clu], width=0.54,
                    color=[COLORS["grid"], COLORS["accent"]],
                    edgecolor="white", linewidth=0.7)
    for b, v in zip(bars, (q_raw, q_clu)):
        ax_q.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:+.3f}",
                  ha="center", va="bottom", fontsize=7.0,
                  color=COLORS["ink"], fontweight="bold")
    ax_q.axhline(0.0, color=COLORS["baseline"], lw=0.7)
    ax_q.set_xticks([0, 1])
    ax_q.set_xticklabels(["arrival\norder", "leaf\norder"], fontsize=6.8)
    ax_q.set_ylim(min(0.0, q_raw * 2.2) - 0.02, q_clu * 1.32)
    ax_q.set_ylabel("modularity Q (-)", fontsize=7.5)
    ax_q.tick_params(labelsize=6.8)
    ax_q.set_title("Q of the implied 4 chunks", fontsize=8.5, loc="left",
                   pad=4)

    # ---- (d) density between detected blocks -----------------------------
    dens = np.zeros((N_BLOCKS, N_BLOCKS))
    for i in range(N_BLOCKS):
        for j in range(N_BLOCKS):
            bi, bj = detected == i, detected == j
            pairs = bi.sum() * bj.sum() - (bi.sum() if i == j else 0)
            dens[i, j] = A[np.ix_(bi, bj)].sum() / max(pairs, 1)
    im_d = ax_d.imshow(dens, cmap=cmap,
                       norm=mpl.colors.Normalize(0.0, dens.max()))
    for i in range(N_BLOCKS):
        for j in range(N_BLOCKS):
            ax_d.text(j, i, f"{dens[i, j]:.2f}", ha="center", va="center",
                      fontsize=6.0,
                      color="white" if dens[i, j] > 0.55 * dens.max()
                      else COLORS["ink"])
    ax_d.set_xticks(range(N_BLOCKS))
    ax_d.set_yticks(range(N_BLOCKS))
    ax_d.set_xticklabels([f"B{i + 1}" for i in range(N_BLOCKS)], fontsize=6.6)
    ax_d.set_yticklabels([f"B{i + 1}" for i in range(N_BLOCKS)], fontsize=6.6)
    ax_d.tick_params(length=0)
    ax_d.set_title("Block link density (-)", fontsize=8.5, loc="left", pad=4)
    for s in ax_d.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(COLORS["rule"])

    # ---- (e) how tightly each ordering packs weight to the diagonal ------
    for order, name, col in ((raw_order, "arrival order", COLORS["baseline"]),
                             (clust_order, "leaf order", CATEGORICAL[0])):
        d, f = diagonal_profile(W, order)
        ax_p.plot(d, f, color=col, lw=1.6, label=name)
    ax_p.set_xlim(0, 1)
    ax_p.set_ylim(0, 1.02)
    ax_p.set_xlabel("bandwidth |i - j| as a fraction of n (-)", fontsize=7.5)
    ax_p.set_ylabel("edge weight inside band (-)", fontsize=7.5)
    ax_p.tick_params(labelsize=6.8)
    ax_p.legend(loc="lower right", frameon=False, fontsize=7.0)
    ax_p.set_title("Reordering pulls weight onto the diagonal",
                   fontsize=8.5, loc="left", pad=4)

    fig.tight_layout(rect=[0, 0.01, 1, 0.93], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_a, ax_b, ax_q, ax_p, ax_d],
                        labels=["(a)", "(b)", "(c)", "(e)", "(d)"],
                        dx=-0.40, dy=0.10)

    savefig(fig, "367_net_adjacency_reorder.png")


if __name__ == "__main__":
    main()
