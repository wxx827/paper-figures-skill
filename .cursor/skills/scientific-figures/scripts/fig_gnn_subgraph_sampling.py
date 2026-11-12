"""Two ways to fit a graph into GPU memory, drawn on one fixed layout.

The same 140-node, four-cluster graph and the same force-directed coordinates
appear in all three panels, so what changes between them is only which part
of the graph a training step has to hold.

GraphSAGE keeps a fan-out tree around a handful of seed nodes: cheap, but the
retained subgraph is scattered and most edges of the batch nodes are thrown
away.  Cluster-GCN keeps whole partitions instead: the induced subgraph is
dense and edge-complete inside a cluster, at the cost of dropping every
inter-cluster edge.  The counts and the activation footprint under each panel
are computed from the retained sets, not quoted.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from pubstyle import set_pub_style, savefig, add_panel_label, annotate_box

CLUSTER_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227"]   # house 1-4
INK, SUB, MUTED, RULE = "#1a1a1a", "#555555", "#777777", "#333333"
FADE_N, FADE_E = "#d2d6dc", "#c8ccd2"

N_PER, N_CLUSTER = 35, 4
P_IN, P_OUT = 0.115, 0.006
FANOUT = (5, 3)               # GraphSAGE neighbours sampled per hop
N_SEEDS = 5
KEEP_CLUSTERS = (0, 2)        # the Cluster-GCN mini-batch
HID, LAYERS, BYTES = 256, 3, 4
SEED = 33


def block_model(rng):
    labels = np.repeat(np.arange(N_CLUSTER), N_PER)
    n = labels.size
    prob = np.where(labels[:, None] == labels[None, :], P_IN, P_OUT)
    adj = np.triu((rng.random((n, n)) < prob).astype(float), 1)
    adj = adj + adj.T
    for i in np.flatnonzero(adj.sum(1) == 0):
        peers = np.flatnonzero((labels == labels[i]) & (np.arange(n) != i))
        j = rng.choice(peers)
        adj[i, j] = adj[j, i] = 1.0
    return adj, labels


def spring_layout(adj, rng, iters=600):
    """Fruchterman-Reingold in plain numpy (no networkx on this machine)."""
    n = adj.shape[0]
    pos = rng.normal(0, 0.6, (n, 2))
    k, temp = 1.5 / np.sqrt(n), 0.20
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
        np.fill_diagonal(dist, 1e9)
        disp = ((k ** 2 / dist ** 2)[:, :, None] * delta
                - (dist / k * adj)[:, :, None] * delta).sum(1) - 0.04 * pos
        step = np.linalg.norm(disp, axis=1, keepdims=True)
        pos += disp / np.maximum(step, 1e-9) * np.minimum(step, temp)
        temp *= 0.995
    pos -= pos.mean(0)
    return pos / np.abs(pos).max()


def graphsage_batch(adj, rng):
    """Seeds plus a sampled fan-out tree; returns kept nodes and edges."""
    n = adj.shape[0]
    seeds = rng.choice(n, N_SEEDS, replace=False)
    frontier, nodes, edges = list(seeds), set(seeds.tolist()), set()
    for fan in FANOUT:
        nxt = []
        for u in frontier:
            nb = np.flatnonzero(adj[u] > 0)
            take = rng.choice(nb, min(fan, nb.size), replace=False)
            for w in take:
                edges.add((min(u, int(w)), max(u, int(w))))
                if int(w) not in nodes:
                    nodes.add(int(w))
                    nxt.append(int(w))
        frontier = nxt
    return np.array(sorted(seeds)), np.array(sorted(nodes)), edges


def cluster_batch(adj, labels):
    """Whole partitions: every intra-cluster edge survives, the rest is cut."""
    nodes = np.flatnonzero(np.isin(labels, KEEP_CLUSTERS))
    keep = set(nodes.tolist())
    eu, ev = np.nonzero(np.triu(adj, 1))
    edges = {(int(a), int(b)) for a, b in zip(eu, ev)
             if a in keep and b in keep and labels[a] == labels[b]}
    return nodes, edges


def footprint(n_nodes):
    """Peak activation bytes for an L-layer, HID-wide model on n_nodes."""
    return n_nodes * HID * LAYERS * BYTES / 1024.0


def draw(ax, pos, labels, all_edges, keep_nodes, keep_edges, cut_edges=(),
         seeds=None, hulls=False):
    kn = np.zeros(len(pos), bool)
    kn[keep_nodes] = True
    for a, b in all_edges:
        if (a, b) in keep_edges:
            continue
        ls = ":" if (a, b) in cut_edges else "-"
        ax.plot(pos[[a, b], 0], pos[[a, b], 1], color=FADE_E, lw=0.5,
                alpha=0.55, ls=ls, zorder=1)
    for a, b in keep_edges:
        ax.plot(pos[[a, b], 0], pos[[a, b], 1], color=RULE, lw=0.95,
                alpha=0.40, zorder=3, solid_capstyle="round")
    if hulls:
        for c in range(N_CLUSTER):
            p = pos[labels == c]
            h = ConvexHull(p)
            loop = np.append(h.vertices, h.vertices[0])
            on = c in KEEP_CLUSTERS
            ax.plot(p[loop, 0], p[loop, 1], lw=0.9, ls=(0, (4, 2.4)),
                    color=CLUSTER_COLORS[c] if on else "#b9bec6",
                    alpha=0.95 if on else 0.7, zorder=2)
    ax.scatter(pos[~kn, 0], pos[~kn, 1], s=16, c=FADE_N, edgecolors="white",
               linewidths=0.6, zorder=4)
    for c in range(N_CLUSTER):
        m = kn & (labels == c)
        ax.scatter(pos[m, 0], pos[m, 1], s=28, c=CLUSTER_COLORS[c],
                   edgecolors="white", linewidths=0.6, zorder=5)
    if seeds is not None:
        ax.scatter(pos[seeds, 0], pos[seeds, 1], s=86, facecolors="none",
                   edgecolors=RULE, linewidths=1.3, zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.14, 1.14)
    ax.set_ylim(-1.14, 1.14)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    adj, labels = block_model(rng)
    n = labels.size
    pos = spring_layout(adj, np.random.default_rng(SEED + 1))

    eu, ev = np.nonzero(np.triu(adj, 1))
    all_edges = [(int(a), int(b)) for a, b in zip(eu, ev)]
    n_e = len(all_edges)

    seeds, sage_nodes, sage_edges = graphsage_batch(
        adj, np.random.default_rng(SEED + 2))
    clus_nodes, clus_edges = cluster_batch(adj, labels)
    cut = {e for e in all_edges
           if labels[e[0]] != labels[e[1]]
           and labels[e[0]] in KEEP_CLUSTERS and labels[e[1]] in KEEP_CLUSTERS}

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 4.4))
    full = np.arange(n)
    draw(axes[0], pos, labels, all_edges, full, set(all_edges))
    draw(axes[1], pos, labels, all_edges, sage_nodes, sage_edges, seeds=seeds)
    draw(axes[2], pos, labels, all_edges, clus_nodes, clus_edges,
         cut_edges=cut, hulls=True)

    titles = ["Full graph (one big batch)",
              f"GraphSAGE, fan-out {FANOUT[0]}–{FANOUT[1]}",
              f"Cluster-GCN, {len(KEEP_CLUSTERS)} of {N_CLUSTER} partitions"]
    subs = ["every node and edge resident",
            f"{N_SEEDS} seed nodes, {len(FANOUT)} hops",
            "partitions kept whole, cross edges cut"]
    kept = [(full, set(all_edges)), (sage_nodes, sage_edges),
            (clus_nodes, clus_edges)]
    for ax, t, s, (kn, ke), lab in zip(axes, titles, subs, kept,
                                       ["(a)", "(b)", "(c)"]):
        ax.text(0.5, 1.075, t, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.8, fontweight="bold", color=INK)
        ax.text(0.5, 1.022, s, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.8, color=MUTED)
        annotate_box(
            ax,
            f"nodes {len(kn)}/{n}  ({100 * len(kn) / n:.0f}%)\n"
            f"edges {len(ke)}/{n_e}  ({100 * len(ke) / n_e:.0f}%)\n"
            f"activations {footprint(len(kn)):.0f} KB",
            loc="lower center", fontsize=6.6)
        add_panel_label(ax, lab, x=-0.03, y=1.075)

    handles = [Line2D([], [], marker="o", ls="", ms=4.4, mfc=c, mec="white",
                      mew=0.6, label=f"partition {i + 1}")
               for i, c in enumerate(CLUSTER_COLORS)]
    handles += [
        Line2D([], [], marker="o", ls="", ms=3.4, mfc=FADE_N, mec="white",
               mew=0.5, label="not in batch"),
        Line2D([], [], marker="o", ls="", ms=7.2, mfc="none", mec=RULE,
               mew=1.3, label="seed node"),
        Line2D([], [], color=RULE, lw=1.1, alpha=0.5, label="edge in batch"),
        Line2D([], [], color=FADE_E, lw=1.1, ls=":", label="edge cut"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.008), ncol=8, frameon=False,
               fontsize=7.0)

    fig.tight_layout(rect=[0, 0.085, 1, 0.885], w_pad=1.6)
    savefig(fig, "448_gnn_subgraph_sampling.png")


if __name__ == "__main__":
    main()
