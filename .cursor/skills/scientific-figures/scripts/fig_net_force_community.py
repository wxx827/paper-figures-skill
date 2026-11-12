"""Force-directed layout of a modular network with community hulls.

The main panel is a Fruchterman-Reingold embedding of a planted-partition
graph: hue encodes the detected community, marker area encodes degree, and a
smoothed convex hull tints each community so the block structure survives at
contact-sheet size.  Subordinate panels on the right carry the community sizes
and the modularity of the partition against a size-matched random relabelling.

No networkx anywhere: the layout, the hulls and the modularity are a few dozen
lines of numpy each.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull

from pubstyle import CATEGORICAL, COLORS, lighten, savefig, set_pub_style
from figkit import halo_text, panel_labels_inches

SEED = 11
COMM_NAMES = ["Core services", "Sensing", "Storage", "Analytics"]
COMM_SIZES = [34, 30, 26, 22]


# --------------------------------------------------------------------------
# graph generation / layout / metrics  (numpy only)
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
    """Attach every stray component to the giant one with a single edge."""
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
            u = rng.choice(comps[0])
            v = rng.choice(c)
            A[u, v] = A[v, u] = 1.0
    return A


def cluster_init(labels, seed=0, ring=0.62, blob=0.135):
    """Seed each community as a filled disc on a ring.

    Two starts were tried before this one and both fail in a way that is worth
    recording.  A uniform random start lets the attraction term win before the
    blocks find opposite sides of the canvas, and the graph folds into a
    stringy chain.  A Fiedler-pair (spectral) start separates the blocks
    correctly but lays each one out along a 1-D eigenvector curve, so every
    community renders as a banana that no amount of FR relaxation rounds out.
    Starting from discs gives the refinement nothing to undo.
    """
    rng = np.random.default_rng(seed)
    n_comm = int(labels.max()) + 1
    ang = np.linspace(0, 2 * np.pi, n_comm, endpoint=False) + np.pi / 4
    centres = ring * np.column_stack([np.cos(ang), np.sin(ang)])
    pos = np.empty((labels.size, 2))
    for c in range(n_comm):
        m = labels == c
        r = blob * np.sqrt(m.sum() / 25.0) * np.sqrt(rng.random(m.sum()))
        a = rng.uniform(0, 2 * np.pi, m.sum())
        pos[m] = centres[c] + np.column_stack([r * np.cos(a), r * np.sin(a)])
    return pos


def rescale(pos, rms=0.5):
    """Centre and scale to a fixed RMS radius.

    Max-abs normalisation would hand the whole frame to a single outlier and
    then squeeze everyone else against it, which is what turns a spring layout
    into a ring of nodes hugging the border.
    """
    pos = pos - pos.mean(0)
    r = np.sqrt((pos ** 2).sum(1).mean()) + 1e-12
    return pos * (rms / r)


def spring_layout(A, labels, seed=0, iters=500, k_scale=1.24, t0=0.035,
                  corral=9.0, corral_slack=1.30, gravity=1.1):
    """Community-disc start, then Fruchterman-Reingold refinement.

    The layout is renormalised to a fixed RMS radius on *every* iteration.
    Without it the repulsion budget is spent inflating the whole drawing -
    clusters drift apart, each one stays a hard little pellet, and the final
    rescale shrinks those pellets back to invisibility.  Pinning the scale
    forces the step budget to be spent on local structure instead.
    """
    n = A.shape[0]
    n_comm = int(labels.max()) + 1
    pos = rescale(cluster_init(labels, seed=seed))
    k = k_scale / np.sqrt(n)
    # Degree-scaled repulsion (the ForceAtlas2 trick).  A hub attracts along
    # every one of its edges at once, so with uniform repulsion it collapses
    # its whole neighbourhood into a pellet; scaling repulsion by degree makes
    # a hub push as hard as it pulls and the neighbourhood opens up.
    w = (A.sum(1) + 1.0)
    w = w / w.mean()
    wij = np.outer(w, w)
    # radius each community needs if its members sit k apart, plus slack
    r_max = np.array([corral_slack * k * np.sqrt((labels == c).sum() / np.pi)
                      for c in range(n_comm)])[labels]
    t = t0
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1))
        # large-but-finite diagonal: kills self-repulsion without seeding the
        # 0 * inf = NaN that np.inf would inject into the attraction term
        np.fill_diagonal(dist, 1e9)
        unit = delta / dist[..., None]
        force = (wij * k ** 2 / dist)[..., None] * unit           # repulsion
        force -= (A * dist / k)[..., None] * unit                 # attraction
        force = force.sum(1)
        # Soft corral: nothing inside the community's own radius, a stiff pull
        # back outside it.  A plain pull-to-centroid also keeps the clusters
        # together but fights the repulsion everywhere, and the equilibrium of
        # those two is a hollow ring of nodes around an empty middle.
        cen = np.stack([pos[labels == c].mean(0) for c in range(n_comm)])
        radial = pos - cen[labels]
        r = np.linalg.norm(radial, axis=1)[:, None] + 1e-12
        force -= corral * np.maximum(r - r_max[:, None], 0.0) * radial / r
        # weak global gravity: only ~14 edges cross between blocks, so without
        # it the mutual repulsion parks each community in its own corner and
        # leaves a hole in the middle of the canvas
        force -= gravity * pos
        mag = np.sqrt((force ** 2).sum(-1))[:, None] + 1e-12
        pos = rescale(pos + force / mag * np.minimum(mag, t))
        t *= 0.992
    return pos


def modularity(A, labels):
    m2 = A.sum()
    q = 0.0
    for c in np.unique(labels):
        idx = labels == c
        q += A[np.ix_(idx, idx)].sum() / m2 - (A[idx].sum() / m2) ** 2
    return q


def smooth_hull(points, pad=0.05, n_out=240, keep=0.82):
    """Rounded outline around a point cloud, for a community 'blob'.

    ``keep`` drops the farthest few members before hulling: one node that has
    drifted across the canvas would otherwise stretch the blob into a spike
    that reads as structure rather than as an outlier.
    """
    if len(points) < 3:
        return None
    r = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    points = points[r <= np.quantile(r, keep)]
    if len(points) < 3:
        return None
    hull = ConvexHull(points)
    ring = points[hull.vertices]
    centre = ring.mean(0)
    ring = centre + (ring - centre) * 1.0 + pad * (
        (ring - centre) / (np.linalg.norm(ring - centre, axis=1)[:, None] + 1e-9))
    closed = np.vstack([ring, ring[:1]])
    try:
        tck, _ = splprep(closed.T, s=0.0, per=True, k=3)
        u = np.linspace(0, 1, n_out)
        return np.column_stack(splev(u, tck))
    except Exception:
        return closed


def fit_limits(ax, pos, box_aspect, pad=0.16):
    """Set equal-aspect limits that fill a panel of the given w/h ratio."""
    x0, x1 = pos[:, 0].min(), pos[:, 0].max()
    y0, y1 = pos[:, 1].min(), pos[:, 1].max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = (x1 - x0) * (1 + pad), (y1 - y0) * (1 + pad)
    if w / h < box_aspect:
        w = h * box_aspect
    else:
        h = w / box_aspect
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)


# --------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    A, labels = planted_partition(COMM_SIZES, p_in=0.190, p_out=0.0035,
                                  seed=SEED)
    pos = spring_layout(A, labels, seed=SEED)
    deg = A.sum(1)
    n_comm = len(COMM_SIZES)
    cols = CATEGORICAL[:n_comm]

    q_found = modularity(A, labels)
    shuffled = rng.permutation(labels)
    q_null = modularity(A, shuffled)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.50],
                  height_ratios=[1.0, 0.62], hspace=0.42, wspace=0.14)
    ax = fig.add_subplot(gs[:, 0])
    ax_size = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[1, 1])

    # ---- community hulls (behind everything) -----------------------------
    hulls = {}
    for c in range(n_comm):
        ring = smooth_hull(pos[labels == c])
        if ring is None:
            continue
        hulls[c] = ring
        ax.add_patch(Polygon(ring, closed=True, facecolor=lighten(cols[c], 0.82),
                             edgecolor=lighten(cols[c], 0.45), linewidth=0.7,
                             zorder=0))

    # ---- edges -----------------------------------------------------------
    iu, ju = np.triu_indices_from(A, k=1)
    mask = A[iu, ju] > 0
    for i, j in zip(iu[mask], ju[mask]):
        inside = labels[i] == labels[j]
        ax.plot(pos[[i, j], 0], pos[[i, j], 1],
                color=cols[labels[i]] if inside else COLORS["baseline"],
                lw=0.60 if inside else 0.85,
                alpha=0.34 if inside else 0.38, zorder=1,
                solid_capstyle="round")

    # ---- nodes -----------------------------------------------------------
    size = 9.0 + 5.2 * deg
    for c in range(n_comm):
        m = labels == c
        ax.scatter(pos[m, 0], pos[m, 1], s=size[m], color=cols[c],
                   edgecolor="white", linewidth=0.6, zorder=3)

    # ---- hub label: single highest-degree node per community -------------
    for c in range(n_comm):
        idx = np.flatnonzero(labels == c)
        v = idx[np.argmax(deg[idx])]
        off = np.sqrt(size[v]) / 2.0 + 4.5                 # points, clear rim
        ax.annotate(f"n{v:03d} · k={int(deg[v])}", xy=pos[v],
                    xytext=(0, -off), textcoords="offset points",
                    ha="center", va="top", fontsize=6.4,
                    color=COLORS["ink"], zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground="white")])

    # ---- community name tags above each hull -----------------------------
    for c in range(n_comm):
        ring = hulls.get(c, pos[labels == c])
        halo_text(ax, ring[:, 0].mean(), ring[:, 1].max() + 0.03,
                  COMM_NAMES[c], ha="center", va="bottom", fontsize=7.8,
                  fontweight="bold", color=cols[c], halo="white",
                  halo_lw=2.8, zorder=7)

    ax.set_aspect("equal")
    ax.axis("off")
    fit_limits(ax, pos, box_aspect=1.18, pad=0.13)
    ax.text(0.0, 0.0, f"{A.shape[0]} nodes · {int(A.sum() / 2)} edges · "
            f"mean degree {deg.mean():.1f}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.0, color=COLORS["sub"])

    # ---- (b) community sizes --------------------------------------------
    order = np.argsort(COMM_SIZES)
    ypos = np.arange(n_comm)
    ax_size.barh(ypos, [COMM_SIZES[i] for i in order], height=0.62,
                 color=[cols[i] for i in order], edgecolor="white",
                 linewidth=0.7)
    for y, i in zip(ypos, order):
        internal = A[np.ix_(labels == i, labels == i)].sum() / 2.0
        ax_size.text(COMM_SIZES[i] + 0.8, y,
                     f"{COMM_SIZES[i]} nodes · {int(internal)} links",
                     va="center", ha="left", fontsize=6.5,
                     color=COLORS["sub"])
    ax_size.set_yticks(ypos)
    ax_size.set_yticklabels([COMM_NAMES[i] for i in order], fontsize=7.0)
    ax_size.set_xlim(0, max(COMM_SIZES) * 1.62)
    ax_size.set_xlabel("community size (nodes)", fontsize=7.5)
    ax_size.tick_params(labelsize=6.8)
    ax_size.set_title("Community sizes", fontsize=8.5, loc="left", pad=4)

    # ---- (c) modularity vs random relabelling ---------------------------
    bars = ax_q.bar([0, 1], [q_found, q_null], width=0.52,
                    color=[COLORS["accent"], COLORS["grid"]],
                    edgecolor="white", linewidth=0.7)
    for b, v in zip(bars, (q_found, q_null)):
        ax_q.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=7.0,
                  color=COLORS["ink"], fontweight="bold")
    ax_q.set_xticks([0, 1])
    ax_q.set_xticklabels(["planted\npartition", "shuffled\nlabels"],
                         fontsize=6.8)
    ax_q.set_ylim(min(0.0, q_null * 1.6), q_found * 1.30)
    ax_q.axhline(0.0, color=COLORS["baseline"], lw=0.7)
    ax_q.set_ylabel("modularity Q (-)", fontsize=7.5)
    ax_q.tick_params(labelsize=6.8)
    ax_q.set_title("Partition quality", fontsize=8.5, loc="left", pad=4)

    fig.tight_layout(rect=[0, 0.01, 1, 0.915], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax, ax_size, ax_q], dx=-0.10, dy=0.18)

    savefig(fig, "364_net_force_community.png")


if __name__ == "__main__":
    main()
