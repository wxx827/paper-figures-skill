"""Node embeddings tightening layer by layer, on one shared coordinate frame.

A three-community stochastic block model is propagated through three
symmetric-normalised GCN layers.  Panel (a) is the raw input feature space
with the graph drawn on top; (b)-(d) are the same nodes after one, two and
three rounds of neighbourhood averaging.  Points are coloured by the true
community, which the model never sees.

Every layer is mean-centred and rescaled to unit RMS radius before plotting -
a GNN's absolute embedding scale is set by the learned weight norm and says
nothing - so a single frame is shared by all four panels and the only thing
that changes across them is cluster *shape*.  The silhouette score printed in
each panel is computed from the very coordinates being drawn.
"""
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label, annotate_box

CLASS_COLORS = ["#b23a48", "#3b5b92", "#2e7d5b"]     # house序位 1, 2, 3
CLASS_NAMES = ["community A", "community B", "community C"]
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"

N_PER = 40
P_IN, P_OUT = 0.130, 0.004
SEED = 12


def block_model(rng):
    labels = np.repeat(np.arange(3), N_PER)
    n = labels.size
    same = labels[:, None] == labels[None, :]
    prob = np.where(same, P_IN, P_OUT)
    draw = rng.random((n, n))
    adj = np.triu((draw < prob).astype(float), 1)
    adj = adj + adj.T
    # nobody is allowed to be isolated: wire strays to a same-community peer
    for i in np.flatnonzero(adj.sum(1) == 0):
        peers = np.flatnonzero((labels == labels[i]) & (np.arange(n) != i))
        j = rng.choice(peers)
        adj[i, j] = adj[j, i] = 1.0
    return adj, labels


def propagate(adj, x0, n_layers=3):
    """Symmetric-normalised averaging with self-loops, the GCN core."""
    a_hat = adj + np.eye(adj.shape[0])
    d = 1.0 / np.sqrt(a_hat.sum(1))
    a_hat = a_hat * d[:, None] * d[None, :]
    out = [x0]
    for _ in range(n_layers):
        out.append(a_hat @ out[-1])
    return out


def standardise(x):
    """Centre, then scale to unit RMS radius (embedding scale is arbitrary)."""
    x = x - x.mean(0)
    return x / np.sqrt((x ** 2).sum(1).mean())


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    adj, labels = block_model(rng)
    n = labels.size

    centres = np.stack([[np.cos(a), np.sin(a)]
                        for a in np.radians([90, 210, 330])])
    x0 = 0.62 * centres[labels] + rng.normal(0, 0.92, (n, 2))
    coords = [standardise(x) for x in propagate(adj, x0, 3)]

    eu, ev = np.triu_indices(n, 1)
    keep = adj[eu, ev] > 0
    eu, ev = eu[keep], ev[keep]

    r = 1.08 * max(np.abs(c).max() for c in coords)
    sils = [silhouette_score(c, labels) for c in coords]

    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.9))
    titles = ["Input graph (layer 0)", "After layer 1", "After layer 2",
              "After layer 3"]
    for ax, c, t, s in zip(axes, coords, titles, sils):
        ax.plot(np.stack([c[eu, 0], c[ev, 0], np.full(eu.size, np.nan)]),
                np.stack([c[eu, 1], c[ev, 1], np.full(eu.size, np.nan)]),
                color="#8a8a8a", lw=0.45, alpha=0.22, zorder=1)
        for k in range(3):
            m = labels == k
            ax.scatter(c[m, 0], c[m, 1], s=13, c=CLASS_COLORS[k],
                       edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_title(t, fontsize=8.8, fontweight="bold", color=INK)
        ax.set_xlabel("embedding dim 1 (-)", fontsize=8.0)
        ax.tick_params(labelsize=7.0)
        annotate_box(ax, f"silhouette {s:+.2f}", loc="lower right",
                     fontsize=6.8)
    axes[0].set_ylabel("embedding dim 2 (-)", fontsize=8.0)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    for ax, lab in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(ax, lab)

    handles = [Line2D([], [], marker="o", ls="", ms=4.6, mfc=col,
                      mec="white", mew=0.6, label=name)
               for col, name in zip(CLASS_COLORS, CLASS_NAMES)]
    handles.append(Line2D([], [], color="#8a8a8a", lw=0.8, alpha=0.5,
                          label="graph edge"))

    fig.tight_layout(rect=[0, 0.115, 1, 0.885], w_pad=1.4)
    savefig(fig, "445_gnn_node_embedding.png")


if __name__ == "__main__":
    main()
