"""Graph-level classification: eight molecule-like graphs and the scoreboard.

Each tile is one test graph drawn with its own force-directed layout, tinted
by its true topology class.  The caption under a tile is what the classifier
said and how sure it was; tiles it got wrong are ringed in the house accent
and tinted.  Because the accent has to be reserved for the errors, the three
classes take house qualitative positions 2-4 (blue / green / gold) rather than
starting at red.

The confusion matrix on the right is computed over the whole held-out split,
not over the eight tiles - eight examples cannot support a rate - and the
tiles are drawn from that split, six correct and two wrong, so the panel and
the grid describe the same model.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label

CLASS_NAMES = ["acyclic", "monocyclic", "polycyclic"]
CLASS_COLORS = ["#3b5b92", "#2e7d5b", "#c9a227"]     # house序位 2, 3, 4
ACCENT = "#b23a48"                                    # reserved for errors
INK, SUB, MUTED, HAIR, RULE = "#1a1a1a", "#555555", "#777777", "#cccccc", "#333333"

# cycle rank per class, so a tile's drawing always matches its true label
RANKS = {0: [0], 1: [1], 2: [3]}
N_GRAPHS, FEAT_NOISE = 300, 0.62
N_SHOW, N_WRONG_SHOWN = 8, 2
SEED = 71


def random_mol(rng, cls):
    """A random tree, then extra bonds to reach this class's cycle rank."""
    n = int(rng.integers(9, 13))
    edges = {(int(rng.integers(0, i)), i) for i in range(1, n)}
    extra = int(rng.choice(RANKS[cls]))
    tries = 0
    while len(edges) < n - 1 + extra and tries < 400:
        tries += 1
        a, b = sorted(rng.choice(n, 2, replace=False))
        if a != b and (a, b) not in edges:
            edges.add((int(a), int(b)))
    adj = np.zeros((n, n))
    for a, b in edges:
        adj[a, b] = adj[b, a] = 1.0
    return adj


def features(adj):
    n = adj.shape[0]
    e = adj.sum() / 2.0
    deg = adj.sum(1)
    tri = np.trace(adj @ adj @ adj) / 6.0
    return [n, e, e - n + 1.0, deg.mean(), deg.max(), tri]


def spring_layout(adj, rng, iters=320):
    """Fruchterman-Reingold in plain numpy (no networkx on this machine)."""
    n = adj.shape[0]
    pos = rng.normal(0, 0.5, (n, 2))
    k, temp = 1.9 / np.sqrt(n), 0.30
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
        np.fill_diagonal(dist, 1e9)
        disp = ((k ** 2 / dist ** 2)[:, :, None] * delta
                - (dist / k * adj)[:, :, None] * delta).sum(1) - 0.05 * pos
        step = np.linalg.norm(disp, axis=1, keepdims=True)
        pos += disp / np.maximum(step, 1e-9) * np.minimum(step, temp)
        temp *= 0.992
    pos -= pos.mean(0)
    return pos / np.maximum(np.abs(pos).max(), 1e-9)


def draw_tile(ax, adj, rng, color, wrong):
    pos = spring_layout(adj, rng)
    if wrong:
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=ACCENT, alpha=0.055, edgecolor="none",
                               zorder=0))
    eu, ev = np.nonzero(np.triu(adj, 1))
    for a, b in zip(eu, ev):
        ax.plot(pos[[a, b], 0], pos[[a, b], 1], color=RULE, lw=1.0,
                alpha=0.40, zorder=2, solid_capstyle="round")
    ax.scatter(pos[:, 0], pos[:, 1], s=34, c=color, edgecolors="white",
               linewidths=0.6, zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.7 if wrong else 0.7)
        s.set_color(ACCENT if wrong else HAIR)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    labels = rng.integers(0, 3, N_GRAPHS)
    graphs = [random_mol(rng, int(c)) for c in labels]
    x = np.array([features(a) for a in graphs], dtype=float)
    x = (x - x.mean(0)) / x.std(0)
    x = x + rng.normal(0, FEAT_NOISE, x.shape)       # measurement noise

    idx = rng.permutation(N_GRAPHS)
    tr, te = idx[:N_GRAPHS // 2], idx[N_GRAPHS // 2:]
    lda = LinearDiscriminantAnalysis().fit(x[tr], labels[tr])
    prob = lda.predict_proba(x[te])
    pred = prob.argmax(1)
    conf = prob.max(1)
    true = labels[te]
    acc = float((pred == true).mean())

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1

    # two correct tiles per class so the grid is not dominated by one label
    wrong = np.flatnonzero(pred != true)
    per_class = (N_SHOW - N_WRONG_SHOWN) // 3
    picked = [rng.choice(np.flatnonzero((pred == true) & (true == c)),
                         per_class, replace=False) for c in range(3)]
    picked.append(rng.choice(wrong, N_SHOW - 3 * per_class, replace=False))
    pick = np.concatenate(picked)
    pick = pick[rng.permutation(N_SHOW)]

    fig = plt.figure(figsize=(10.0, 5.6))
    # column 4 is an empty gutter so the confusion panel's label and ticks do
    # not collide with the last tile column
    gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 0.16, 1.05],
                          wspace=0.20, hspace=0.55, left=0.035, right=0.965,
                          top=0.845, bottom=0.115)
    tiles = []
    for k, j in enumerate(pick):
        r, c = divmod(k, 4)
        ax = fig.add_subplot(gs[2 * r:2 * r + 2, c])
        t, p = int(true[j]), int(pred[j])
        draw_tile(ax, graphs[te[j]], np.random.default_rng(SEED + 10 + k),
                  CLASS_COLORS[t], t != p)
        ax.set_title(f"true: {CLASS_NAMES[t]}", fontsize=7.6,
                     fontweight="bold", color=CLASS_COLORS[t], pad=3)
        ax.set_xlabel(f"pred {CLASS_NAMES[p]}  ·  {conf[j]:.2f}",
                      fontsize=6.8, labelpad=2.5,
                      color=ACCENT if t != p else SUB)
        tiles.append(ax)

    ax_cm = fig.add_subplot(gs[1:3, 5])
    im = ax_cm.imshow(cm, cmap=get_cmap("cb_sequential_blue"), aspect="auto",
                      vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(3):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=7.4,
                       color="white" if cm[i, j] > 0.6 * cm.max() else INK,
                       fontweight="bold" if i == j else "normal")
    ax_cm.set_xticks(range(3))
    ax_cm.set_xticklabels(CLASS_NAMES, fontsize=6.8, rotation=28, ha="right")
    ax_cm.set_yticks(range(3))
    ax_cm.set_yticklabels(CLASS_NAMES, fontsize=6.8)
    for i, lab in enumerate(ax_cm.get_xticklabels()):
        lab.set_color(CLASS_COLORS[i])
    for i, lab in enumerate(ax_cm.get_yticklabels()):
        lab.set_color(CLASS_COLORS[i])
    ax_cm.tick_params(length=2.0)
    ax_cm.set_xlabel("predicted class (-)", fontsize=8.0, labelpad=2)
    ax_cm.set_ylabel("true class (-)", fontsize=8.0)
    ax_cm.set_title(f"Confusion, held-out\nn = {len(te)} graphs",
                    fontsize=8.5, fontweight="bold", color=INK, pad=5)

    add_panel_label(tiles[0], "(a)", x=-0.16, y=1.12)
    add_panel_label(ax_cm, "(b)", x=-0.40, y=1.10)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    wt, wp = np.unravel_index(off.argmax(), off.shape)
    savefig(fig, "451_gnn_graph_classification.png")


if __name__ == "__main__":
    main()
