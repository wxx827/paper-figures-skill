"""Six centrality measures on one network, one layout, one colour scale.

Every panel draws the *same* embedding at the *same* axis limits with the same
marker sizes, so the only thing that changes between panels is hue.  That is
the whole point of the figure: a reader comparing "which nodes matter" across
measures must not also be comparing six different pictures.

Each measure lives on its own units, so the colour scale is min-max rescaled
*within* each measure and the shared bar is labelled as such; the raw range is
printed in every panel so the rescaling cannot hide a flat measure.  The three
highest-scoring nodes of each measure are ringed and labelled.

Betweenness is Brandes' algorithm, closeness comes from all-pairs BFS,
eigenvector and PageRank from power iteration, k-core from peeling - all
numpy / collections, no networkx in this environment.
"""
from collections import deque

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pubstyle import COLORS, get_palette, savefig, set_pub_style
from figkit import panel_labels_inches, safe_colorbar

SEED = 23
COMM_SIZES = [30, 26, 24]


# --------------------------------------------------------------------------
# graph, layout
# --------------------------------------------------------------------------
def planted_partition(sizes, p_in, p_out, seed=0, spread=1.6):
    """Degree-heterogeneous stochastic block model; returns (A, labels).

    Two details that are easy to get wrong.  The propensities are renormalised
    *within* each block, otherwise a block that happens to draw small gammas
    comes out three times sparser than its neighbours and lays out as a
    straggling chain rather than a cluster.  And the symmetric draw is a
    mirrored upper triangle, not ``(u + u.T) / 2`` - the average of two
    uniforms is below p with probability 2p^2, so that shortcut silently
    delivers less than half the density asked for.
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
    return connect_components((u < p).astype(float), rng), labels


def connect_components(A, rng):
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in np.flatnonzero(A[v] > 0):
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(np.array(comp))
    if len(comps) > 1:
        comps.sort(key=len, reverse=True)
        for c in comps[1:]:
            u, v = rng.choice(comps[0]), rng.choice(c)
            A[u, v] = A[v, u] = 1.0
    return A


def cluster_init(labels, seed=0, ring=0.62, blob=0.135):
    rng = np.random.default_rng(seed)
    n_comm = int(labels.max()) + 1
    ang = np.linspace(0, 2 * np.pi, n_comm, endpoint=False) + np.pi / 6
    centres = ring * np.column_stack([np.cos(ang), np.sin(ang)])
    pos = np.empty((labels.size, 2))
    for c in range(n_comm):
        m = labels == c
        r = blob * np.sqrt(m.sum() / 25.0) * np.sqrt(rng.random(m.sum()))
        a = rng.uniform(0, 2 * np.pi, m.sum())
        pos[m] = centres[c] + np.column_stack([r * np.cos(a), r * np.sin(a)])
    return pos


def rescale(pos, rms=0.5):
    pos = pos - pos.mean(0)
    return pos * (rms / (np.sqrt((pos ** 2).sum(1).mean()) + 1e-12))


def spring_layout(A, labels, seed=0, iters=500, k_scale=1.24, t0=0.035,
                  corral=9.0, corral_slack=0.95, gravity=1.35):
    """Community-disc start, Fruchterman-Reingold refinement, fixed frame."""
    n = A.shape[0]
    n_comm = int(labels.max()) + 1
    pos = rescale(cluster_init(labels, seed=seed))
    k = k_scale / np.sqrt(n)
    w = (A.sum(1) + 1.0)
    wij = np.outer(w / w.mean(), w / w.mean())
    r_max = np.array([corral_slack * k * np.sqrt((labels == c).sum() / np.pi)
                      for c in range(n_comm)])[labels]
    t = t0
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1))
        np.fill_diagonal(dist, 1e9)          # finite: avoids 0 * inf -> NaN
        unit = delta / dist[..., None]
        force = (wij * k ** 2 / dist)[..., None] * unit
        force -= (A * dist / k)[..., None] * unit
        force = force.sum(1)
        cen = np.stack([pos[labels == c].mean(0) for c in range(n_comm)])
        radial = pos - cen[labels]
        r = np.linalg.norm(radial, axis=1)[:, None] + 1e-12
        force -= corral * np.maximum(r - r_max[:, None], 0.0) * radial / r
        force -= gravity * pos
        mag = np.sqrt((force ** 2).sum(-1))[:, None] + 1e-12
        pos = rescale(pos + force / mag * np.minimum(mag, t))
        t *= 0.992
    return pos


# --------------------------------------------------------------------------
# centralities
# --------------------------------------------------------------------------
def adjacency_list(A):
    return [np.flatnonzero(A[v] > 0).tolist() for v in range(A.shape[0])]


def betweenness(A):
    """Brandes, unweighted and undirected."""
    adj = adjacency_list(A)
    n = A.shape[0]
    bc = np.zeros(n)
    for s in range(n):
        stack, pred = [], [[] for _ in range(n)]
        sigma = np.zeros(n)
        dist = np.full(n, -1)
        sigma[s], dist[s] = 1.0, 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
    return bc / ((n - 1) * (n - 2))          # undirected, normalised


def closeness(A):
    adj = adjacency_list(A)
    n = A.shape[0]
    out = np.zeros(n)
    for s in range(n):
        dist = np.full(n, -1)
        dist[s] = 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        reach = dist[dist > 0]
        out[s] = reach.size / reach.sum() * (reach.size / (n - 1.0))
    return out


def eigenvector(A, iters=400):
    x = np.ones(A.shape[0])
    for _ in range(iters):
        x = A @ x
        x /= np.linalg.norm(x) + 1e-12
    return np.abs(x)


def pagerank(A, damping=0.85, iters=400):
    n = A.shape[0]
    out = np.maximum(A.sum(1), 1e-12)
    x = np.full(n, 1.0 / n)
    for _ in range(iters):
        x = (1 - damping) / n + damping * (A.T @ (x / out))
    return x / x.sum()


def k_core(A):
    deg = A.sum(1).copy()
    alive = np.ones(A.shape[0], dtype=bool)
    core = np.zeros(A.shape[0])
    k = 0
    while alive.any():
        k = max(k, int(deg[alive].min()))
        peel = alive & (deg <= k)
        while peel.any():
            core[peel] = k
            alive[peel] = False
            deg -= A[peel].sum(0)
            peel = alive & (deg <= k)
    return core


# --------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)

    A, labels = planted_partition(COMM_SIZES, p_in=0.230, p_out=0.004,
                                  seed=SEED)
    pos = spring_layout(A, labels, seed=SEED)

    measures = [
        ("Degree", A.sum(1)),
        ("Betweenness", betweenness(A)),
        ("Closeness", closeness(A)),
        ("Eigenvector", eigenvector(A)),
        ("PageRank", pagerank(A)),
        ("k-core index", k_core(A)),
    ]

    # Drop the palest stop of the house blue ramp: on a white page with white
    # node rims, cmap(0.0) is indistinguishable from no marker at all, and the
    # lower half of every measure would simply disappear.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cb_sequential_blue_deep", get_palette("cb_sequential_blue")[1:])
    norm = mpl.colors.Normalize(0.0, 1.0)      # every panel is rescaled to 0-1

    iu, ju = np.triu_indices_from(A, k=1)
    edges = [(i, j) for i, j in zip(iu[A[iu, ju] > 0], ju[A[iu, ju] > 0])]

    # one set of limits, derived once, applied to all six panels
    span = 1.06 * np.abs(pos).max()
    lim = (-span, span)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    for ax, (name, raw) in zip(axes.ravel(), measures):
        val = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        for i, j in edges:
            ax.plot(pos[[i, j], 0], pos[[i, j], 1], color=COLORS["baseline"],
                    lw=0.55, alpha=0.45, zorder=1, solid_capstyle="round")
        ax.scatter(pos[:, 0], pos[:, 1], s=32, c=val, cmap=cmap, norm=norm,
                   edgecolor="white", linewidth=0.6, zorder=3)

        top3 = np.argsort(raw)[-3:][::-1]
        ax.scatter(pos[top3, 0], pos[top3, 1], s=110, facecolor="none",
                   edgecolor=COLORS["accent"], linewidth=1.0, zorder=4)
        # The top three often sit inside one cluster, so the offsets are
        # staggered by rank instead of all hanging below the marker.
        offsets = [(0, -9, "center", "top"), (0, 9, "center", "bottom"),
                   (10, 0, "left", "center")]
        for rank, (v, (ox, oy, ha, va)) in enumerate(zip(top3, offsets), 1):
            ax.annotate(f"#{rank} n{v:02d}", xy=pos[v], xytext=(ox, oy),
                        textcoords="offset points", ha=ha, va=va,
                        fontsize=6.2, color=COLORS["ink"], zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.2,
                                                    foreground="white")])

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_title(name, fontsize=9.0, loc="center", pad=3,
                     fontweight="bold", color=COLORS["ink"])
        ax.text(0.5, -0.055, f"raw range {raw.min():.3g} to {raw.max():.3g}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.4,
                color=COLORS["muted"])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.tight_layout(rect=[0, 0.11, 1, 0.94], h_pad=2.6, w_pad=2.6)
    safe_colorbar(fig, sm, axes[1, 1], side="bottom", width=0.13, pad=0.30,
                  shrink=1.35,
                  label="centrality, min-max rescaled within measure (-)")
    panel_labels_inches(fig, axes, dx=-0.05, dy=0.03)

    savefig(fig, "366_net_centrality_compare.png")


if __name__ == "__main__":
    main()
