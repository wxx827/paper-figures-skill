"""Percolation of a scale-free network under random failure and two attacks.

The main panel is the survival curve: the largest connected component, as a
fraction of the original node count, against the fraction of nodes removed.
Random failure is an ensemble of 24 draws (band = one standard deviation);
the two attacks are deterministic and recomputed after every removal, so a
hub that only becomes central once its neighbours are gone is still found.

The threshold marks are not eyeballed off the survival curve.  Each dashed
line sits at the peak of that attack's second-largest component, which is the
standard finite-size estimator of the percolation threshold and the only place
on the curve where "the network breaks here" has a defensible meaning.

The three thumbnails freeze all three strategies at the same removal fraction
on one shared layout: same node positions, same limits, so the difference
between them is the damage and nothing else.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pubstyle import CATEGORICAL, COLORS, lighten, savefig, set_pub_style
from figkit import halo_text, panel_label_inches

SEED = 41
N_NODE = 130
M_ATTACH = 2          # edges added per new node in the growth model
N_RANDOM = 24         # random-failure realisations
F_MAX = 0.70          # removal fraction covered by the curves
F_SHOW = 0.18         # removal fraction drawn in the thumbnails

ATTACKS = [
    ("random", "Random failure", CATEGORICAL[1]),
    ("degree", "Degree attack", CATEGORICAL[3]),
    ("between", "Betweenness attack", COLORS["accent"]),
]


# --------------------------------------------------------------------------
def ba_graph(n, m, seed=0):
    """Preferential attachment by repeated-node sampling (Barabasi-Albert)."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=np.int8)
    targets = list(range(m + 1))
    for i, j in [(a, b) for a in targets for b in targets if a < b]:
        A[i, j] = A[j, i] = 1
    repeated = [v for v in targets for _ in range(m)]
    for v in range(m + 1, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(repeated[rng.integers(0, len(repeated))])
        for w in chosen:
            A[v, w] = A[w, v] = 1
        repeated.extend(chosen)
        repeated.extend([v] * m)
    return A


def adjacency_lists(A):
    return [np.flatnonzero(row).tolist() for row in A]


def component_sizes(adj, alive):
    """Sizes of the connected components among the alive nodes, descending."""
    seen = np.zeros(len(adj), dtype=bool)
    sizes = []
    for s in np.flatnonzero(alive):
        if seen[s]:
            continue
        stack, size = [s], 0
        seen[s] = True
        while stack:
            v = stack.pop()
            size += 1
            for w in adj[v]:
                if alive[w] and not seen[w]:
                    seen[w] = True
                    stack.append(w)
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def betweenness(adj, alive):
    """Brandes' algorithm on the induced subgraph of the alive nodes."""
    n = len(adj)
    cb = np.zeros(n)
    nodes = np.flatnonzero(alive)
    for s in nodes:
        stack = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n)
        dist = np.full(n, -1)
        sigma[s] = 1.0
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            stack.append(v)
            for w in adj[v]:
                if not alive[w]:
                    continue
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
                cb[w] += delta[w]
    return cb


def dismantle(adj, order=None, rule=None, n_remove=0):
    """Remove nodes one at a time; return (LCC sizes, second-component sizes).

    ``rule`` is recomputed on the surviving subgraph before every removal.  A
    ranking taken once on the intact network is a much weaker attack: it keeps
    deleting nodes that were hubs of a component that has already fallen off,
    and the curve it produces flatters the network.
    """
    n = len(adj)
    alive = np.ones(n, dtype=bool)
    s1, s2 = [], []
    sizes = component_sizes(adj, alive)
    s1.append(sizes[0])
    s2.append(sizes[1] if len(sizes) > 1 else 0)
    for step in range(n_remove):
        if rule is None:
            v = order[step]
        else:
            score = rule(adj, alive)
            score = np.where(alive, score, -np.inf)
            v = int(np.argmax(score))
        alive[v] = False
        sizes = component_sizes(adj, alive)
        s1.append(sizes[0] if sizes else 0)
        s2.append(sizes[1] if len(sizes) > 1 else 0)
    return np.array(s1, dtype=float), np.array(s2, dtype=float), alive


def degree_rule(adj, alive):
    return np.array([sum(1 for w in nb if alive[w]) for nb in adj],
                    dtype=float)


def spring_layout(A, seed=0, iters=700, k_scale=2.05, t0=0.05, gravity=2.4):
    """Fruchterman-Reingold with degree-scaled repulsion and firm gravity.

    A hub-and-spoke graph laid out with weak gravity throws its degree-one
    leaves far out on long spokes; the fixed-RMS renormalisation then squeezes
    the core into a dot, and the thumbnails below become grey dust.  Gravity
    keeps the leaves in frame so the core can use the space.
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    ang = rng.permutation(n) / n * 2 * np.pi
    pos = 0.5 * np.column_stack([np.cos(ang), np.sin(ang)])
    pos += 0.03 * rng.standard_normal((n, 2))
    k = k_scale / np.sqrt(n)
    w = A.sum(1) + 1.0
    w = w / w.mean()
    wij = np.outer(w, w)
    t = t0
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1))
        np.fill_diagonal(dist, 1e9)
        unit = delta / dist[..., None]
        force = (wij * k ** 2 / dist)[..., None] * unit
        force -= (A * dist / k)[..., None] * unit
        force = force.sum(1) - gravity * pos
        mag = np.sqrt((force ** 2).sum(-1))[:, None] + 1e-12
        pos = pos + force / mag * np.minimum(mag, t)
        pos -= pos.mean(0)
        pos *= 0.5 / (np.sqrt((pos ** 2).sum(1).mean()) + 1e-12)
        t *= 0.994
    return pos


# --------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    A = ba_graph(N_NODE, M_ATTACH, seed=SEED)
    adj = adjacency_lists(A)
    deg = A.sum(1)
    pos = spring_layout(A, seed=SEED)

    n_rem = int(F_MAX * N_NODE)
    frac = np.arange(n_rem + 1) / N_NODE
    i_show = int(F_SHOW * N_NODE)

    # ---- random failure ensemble ------------------------------------------
    r1 = np.empty((N_RANDOM, n_rem + 1))
    r2 = np.empty((N_RANDOM, n_rem + 1))
    alive_rand = None
    for r in range(N_RANDOM):
        order = rng.permutation(N_NODE)
        a, b, _ = dismantle(adj, order=order, n_remove=n_rem)
        r1[r], r2[r] = a, b
        if r == 0:                       # the realisation drawn as a thumbnail
            alive_rand = np.ones(N_NODE, dtype=bool)
            alive_rand[order[:i_show]] = False

    # ---- targeted attacks --------------------------------------------------
    d1, d2, _ = dismantle(adj, rule=degree_rule, n_remove=n_rem)
    b1, b2, _ = dismantle(adj, rule=betweenness, n_remove=n_rem)

    # replay the two attacks to get the alive mask at the snapshot fraction
    def alive_at(rule, k):
        alive = np.ones(N_NODE, dtype=bool)
        for _ in range(k):
            score = np.where(alive, rule(adj, alive), -np.inf)
            alive[int(np.argmax(score))] = False
        return alive

    alive_deg = alive_at(degree_rule, i_show)
    alive_bet = alive_at(betweenness, i_show)

    curves = {"random": r1.mean(0) / N_NODE,
              "degree": d1 / N_NODE,
              "between": b1 / N_NODE}
    second = {"random": r2.mean(0), "degree": d2, "between": b2}
    alive_map = {"random": alive_rand, "degree": alive_deg,
                 "between": alive_bet}
    f_c = {k: frac[int(np.argmax(v))] for k, v in second.items()}

    fig = plt.figure(figsize=(10.4, 6.2))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.0, 0.50],
                  hspace=0.30, wspace=0.06)
    ax = fig.add_subplot(gs[:, 0])
    ax_t = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    # ---- (a) survival curves ----------------------------------------------
    band_lo = np.percentile(r1 / N_NODE, 16, axis=0)
    band_hi = np.percentile(r1 / N_NODE, 84, axis=0)
    ax.fill_between(frac, band_lo, band_hi, color=lighten(CATEGORICAL[1], 0.78),
                    lw=0, zorder=1)
    for key, label, col in ATTACKS:
        ax.plot(frac, curves[key], color=col, lw=2.0 if key != "random" else 1.8,
                zorder=4, label=label +
                (f" (n = {N_RANDOM})" if key == "random" else ""))
        ax.axvline(f_c[key], color=col, lw=0.9, ls=(0, (4, 2)), zorder=2,
                   alpha=0.85)

    ymax = 1.06
    for k, (key, label, col) in enumerate(ATTACKS):
        ax.text(f_c[key], ymax - 0.055 * k, f" $f_c$ = {f_c[key]:.2f}",
                ha="left", va="top", fontsize=6.8, color=col)
    ax.axvline(F_SHOW, color=COLORS["rule"], lw=0.8, ls=":", zorder=3)
    halo_text(ax, F_SHOW + 0.006, 0.40, f"snapshots at f = {F_SHOW:.2f}",
              rotation=90, ha="left", va="bottom", fontsize=6.8,
              color=COLORS["sub"], halo="white", halo_lw=2.6, zorder=6)

    ax.set_xlim(0, F_MAX)
    ax.set_ylim(0, ymax + 0.02)
    ax.set_xlabel("fraction of nodes removed  f (-)", fontsize=8.5)
    ax.set_ylabel("largest connected component / N (-)", fontsize=8.5)
    ax.tick_params(labelsize=7.2)
    ax.grid(lw=0.5, color=COLORS["grid"], alpha=0.55)
    ax.set_title("Survival of the giant component", fontsize=9.5, loc="left",
                 pad=4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), fontsize=7.4,
              handlelength=1.8, borderaxespad=0.5)
    ax.text(0.0, -0.105, f"scale-free graph · N = {N_NODE} · "
            f"E = {int(A.sum() / 2)} · max degree {int(deg.max())} · "
            r"$f_c$ from the peak of the second-largest component",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
            color=COLORS["sub"])

    # ---- (b) remnant thumbnails, one shared layout ------------------------
    iu, ju = np.triu_indices(N_NODE, 1)
    emask = A[iu, ju] > 0
    ei, ej = iu[emask], ju[emask]
    half = 0.62 * max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]))
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    for axt, (key, label, col) in zip(ax_t, ATTACKS):
        alive = alive_map[key]
        # giant component of the remnant, so the survivors can be told apart
        seen = np.zeros(N_NODE, dtype=bool)
        best = np.zeros(N_NODE, dtype=bool)
        for s in np.flatnonzero(alive):
            if seen[s]:
                continue
            stack, comp = [s], []
            seen[s] = True
            while stack:
                v = stack.pop()
                comp.append(v)
                for w in adj[v]:
                    if alive[w] and not seen[w]:
                        seen[w] = True
                        stack.append(w)
            if len(comp) > best.sum():
                best = np.zeros(N_NODE, dtype=bool)
                best[comp] = True

        for u, v in zip(ei, ej):
            if not (alive[u] and alive[v]):
                axt.plot(pos[[u, v], 0], pos[[u, v], 1],
                         color=COLORS["hairline"], lw=0.35, alpha=0.45,
                         zorder=1)
            else:
                inside = best[u] and best[v]
                axt.plot(pos[[u, v], 0], pos[[u, v], 1],
                         color=col if inside else COLORS["baseline"],
                         lw=0.55, alpha=0.30 if inside else 0.22, zorder=2)
        dead = ~alive
        axt.scatter(pos[dead, 0], pos[dead, 1], s=3.2, color=COLORS["grid"],
                    linewidth=0, zorder=3)
        frag = alive & ~best
        axt.scatter(pos[frag, 0], pos[frag, 1], s=4.5 + 1.5 * deg[frag],
                    color=COLORS["baseline"], edgecolor="white",
                    linewidth=0.6, zorder=4)
        axt.scatter(pos[best, 0], pos[best, 1], s=4.5 + 1.5 * deg[best],
                    color=col, edgecolor="white", linewidth=0.6, zorder=5)
        axt.set_aspect("equal")
        axt.axis("off")
        axt.set_xlim(cx - half, cx + half)
        axt.set_ylim(cy - half, cy + half)
        axt.set_title(label, fontsize=8.0, loc="left", pad=2, color=col)
        axt.text(0.99, 0.02, f"LCC {best.sum() / N_NODE:.0%} of N · "
                 f"{int(alive.sum() - best.sum())} stranded",
                 transform=axt.transAxes, ha="right", va="bottom",
                 fontsize=6.4, color=COLORS["sub"])

    fig.tight_layout(rect=[0, 0.02, 1, 0.925], h_pad=1.6, w_pad=2.0)
    panel_label_inches(fig, ax, "(a)", dx=-0.46, dy=0.12)
    panel_label_inches(fig, ax_t[0], "(b)", dx=-0.16, dy=0.10)

    savefig(fig, "372_net_robustness.png")


if __name__ == "__main__":
    main()
