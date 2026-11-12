"""A temporal network read three ways: raster, snapshots, and time series.

Panel (a) is the contact raster - one row per edge, one column per time step,
a mark where that edge carries a contact.  Rows are grouped by community and
sorted within a group by mean activation time, which turns the three
overlapping activity waves into three visible blocks instead of static.

Panel (c) freezes five windows of that raster back into network drawings.  All
five share one node layout computed on the aggregate graph, so a node that
appears to move between snapshots would be a bug, not a finding; only the
edges and the node sizes change.

Panel (b) tracks the whole run.  All three curves are fractions of their own
maximum possible value, which is what lets them share a single 0-1 axis
instead of needing a second y axis nobody reads correctly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pubstyle import CATEGORICAL, COLORS, savefig, set_pub_style
from figkit import halo_text, panel_label_inches

SEED = 23
N_NODE = 27
T = 132
COMM_SIZES = [9, 9, 9]
GROUP_NAMES = ["Community A", "Community B", "Community C", "Bridges"]
WIN_CENTRES = [14, 43, 68, 96, 120]
WIN_HALF = 7


def planted_partition(sizes, p_in, p_out, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    n = labels.size
    same = labels[:, None] == labels[None, :]
    p = np.where(same, p_in, p_out)
    u = np.triu(rng.random((n, n)), 1)
    A = ((u + u.T) < p) & ~np.eye(n, dtype=bool)
    return A.astype(float), labels


def activity(edges, groups, seed=0):
    """Bursty on/off series per edge under a group-specific activity wave.

    Independent Bernoulli draws per time step give a raster of uniform speckle
    - statistically fine, visually useless, because no burst survives to be
    seen at print size.  Carrying the previous state forward with probability
    ``persist`` turns the same marginal rate into runs, which is what real
    contact data looks like and what the snapshots below need in order to
    differ from one another.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    centres = {0: 0.20 * T, 1: 0.52 * T, 2: 0.82 * T}
    act = np.zeros((len(edges), T), dtype=bool)
    for e, g in enumerate(groups):
        if g < 3:
            wave = np.exp(-0.5 * ((t - centres[g]) / (0.16 * T)) ** 2)
            rate = 0.05 + 0.62 * wave
        else:                                   # bridges: low, always on
            rate = np.full(T, 0.13) + 0.05 * np.sin(2 * np.pi * t / 34.0)
        rate = rate * rng.uniform(0.55, 1.35)
        on = False
        for i in range(T):
            p = rate[i] if not on else 0.30 + 0.45 * rate[i]
            on = rng.random() < p
            act[e, i] = on
    return act


def components_fraction(n, edge_list, active):
    """Size of the largest connected component / n, from the active edges."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (u, v), on in zip(edge_list, active):
        if on:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    roots = np.array([find(i) for i in range(n)])
    return np.bincount(roots).max() / n


def spring_layout(A, seed=0, iters=460, k_scale=1.30, t0=0.045, gravity=1.4):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    ang = rng.permutation(n) / n * 2 * np.pi
    pos = 0.5 * np.column_stack([np.cos(ang), np.sin(ang)])
    pos += 0.03 * rng.standard_normal((n, 2))
    k = k_scale / np.sqrt(n)
    t = t0
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1))
        np.fill_diagonal(dist, 1e9)
        unit = delta / dist[..., None]
        force = (k ** 2 / dist)[..., None] * unit
        force -= (A * dist / k)[..., None] * unit
        force = force.sum(1) - gravity * pos
        mag = np.sqrt((force ** 2).sum(-1))[:, None] + 1e-12
        pos = pos + force / mag * np.minimum(mag, t)
        pos -= pos.mean(0)
        pos *= 0.5 / (np.sqrt((pos ** 2).sum(1).mean()) + 1e-12)
        t *= 0.993
    return pos


def main():
    set_pub_style(base_font=8)

    A, labels = planted_partition(COMM_SIZES, p_in=0.42, p_out=0.055,
                                  seed=SEED)
    iu, ju = np.triu_indices(N_NODE, 1)
    m = A[iu, ju] > 0
    edges = list(zip(iu[m], ju[m]))
    groups = np.array([labels[u] if labels[u] == labels[v] else 3
                       for u, v in edges])
    n_edge = len(edges)

    act = activity(edges, groups, seed=SEED)
    pos = spring_layout(A, seed=SEED)

    # raster row order: group block, then mean activation time inside it
    mean_t = np.array([(np.arange(T)[a].mean() if a.any() else T / 2)
                       for a in act])
    order = np.lexsort((mean_t, groups))
    act_s, groups_s = act[order], groups[order]

    cols = [CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[3], COLORS["baseline"]]

    # ---- time series (all three are fractions of their own maximum) -------
    frac_edges = act.sum(0) / n_edge
    lcc = np.array([components_fraction(N_NODE, edges, act[:, i])
                    for i in range(T)])
    deg_active = np.zeros((N_NODE, T))
    for e, (u, v) in enumerate(edges):
        deg_active[u] += act[e]
        deg_active[v] += act[e]
    frac_nodes = (deg_active > 0).sum(0) / N_NODE

    fig = plt.figure(figsize=(10.6, 6.8))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.57],
                  height_ratios=[1.0, 0.62], hspace=0.40, wspace=0.20)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    sub = gs[1, :].subgridspec(1, len(WIN_CENTRES), wspace=0.05)
    ax_s = [fig.add_subplot(sub[0, i]) for i in range(len(WIN_CENTRES))]

    # ---- (a) contact raster ------------------------------------------------
    rgb = np.ones((n_edge, T, 3))
    for e in range(n_edge):
        c = np.array(plt.matplotlib.colors.to_rgb(cols[groups_s[e]]))
        rgb[e][act_s[e]] = c
    ax_r.imshow(rgb, aspect="auto", interpolation="nearest",
                extent=(-0.5, T - 0.5, n_edge - 0.5, -0.5), zorder=2)

    bounds = np.flatnonzero(np.diff(groups_s)) + 0.5
    for b in bounds:
        ax_r.axhline(b, color=COLORS["rule"], lw=0.7, zorder=4)
    starts = np.r_[0, bounds + 0.5]
    ends = np.r_[bounds - 0.5, n_edge - 1]
    for g, (s, e) in enumerate(zip(starts, ends)):
        ax_r.text(-3.2, (s + e) / 2.0, GROUP_NAMES[g], rotation=90,
                  ha="right", va="center", fontsize=6.5, color=cols[g])

    for w, c in enumerate(WIN_CENTRES):
        ax_r.axvspan(c - WIN_HALF, c + WIN_HALF, color=COLORS["ink"],
                     alpha=0.085, lw=0, zorder=3)
        halo_text(ax_r, c, -0.02 * n_edge, f"W{w + 1}", ha="center",
                  va="bottom", fontsize=6.6, color=COLORS["ink"],
                  halo="white", halo_lw=2.4, zorder=6)

    ax_r.set_xlim(-0.5, T - 0.5)
    ax_r.set_ylim(n_edge - 0.5, -3.5)
    ax_r.set_xlabel("time step", fontsize=8.0)
    ax_r.set_ylabel(f"edge (n = {n_edge}, grouped)", fontsize=8.0, labelpad=18)
    ax_r.set_yticks([])
    ax_r.tick_params(labelsize=7.0)
    ax_r.set_title("Contact raster: when each edge is active", fontsize=9.0,
                   loc="left", pad=4)
    for side in ("top", "right"):
        ax_r.spines[side].set_visible(False)
    density = act.mean()
    ax_r.text(1.0, -0.175, f"{int(act.sum())} contacts · raster density "
              f"{density:.2f}", transform=ax_r.transAxes, ha="right",
              va="top", fontsize=6.9, color=COLORS["sub"])

    # ---- (b) metrics over time --------------------------------------------
    for w, c in enumerate(WIN_CENTRES):
        ax_m.axvspan(c - WIN_HALF, c + WIN_HALF, color=COLORS["ink"],
                     alpha=0.085, lw=0, zorder=0)
    ax_m.plot(np.arange(T), lcc, color=COLORS["ink"], lw=1.5, zorder=4,
              label="largest component / N")
    ax_m.plot(np.arange(T), frac_nodes, color=CATEGORICAL[1], lw=1.2,
              zorder=3, label="nodes with a contact / N")
    ax_m.plot(np.arange(T), frac_edges, color=COLORS["accent"], lw=1.2,
              zorder=3, label="edges active / E")
    ax_m.set_xlim(0, T - 1)
    ax_m.set_ylim(0, 1.30)
    ax_m.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_m.set_xlabel("time step", fontsize=7.5)
    ax_m.set_ylabel("fraction (-)", fontsize=7.5)
    ax_m.tick_params(labelsize=6.8)
    ax_m.set_title("Structure over time, one shared 0-1 scale", fontsize=8.5,
                   loc="left", pad=4)
    ax_m.legend(loc="upper left", fontsize=6.5, handlelength=1.6, ncol=1,
                borderaxespad=0.3)
    ax_m.grid(axis="y", lw=0.5, color=COLORS["grid"], alpha=0.55)
    peak = int(np.argmax(lcc))
    ax_m.annotate(f"peak connectivity t = {peak}", xy=(peak, lcc[peak]),
                  xytext=(peak + 6, 0.10), fontsize=6.4, color=COLORS["sub"],
                  arrowprops=dict(arrowstyle="-", lw=0.6,
                                  color=COLORS["baseline"]))

    # ---- (c) five snapshots on one shared layout --------------------------
    span_x = pos[:, 0].max() - pos[:, 0].min()
    span_y = pos[:, 1].max() - pos[:, 1].min()
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    half = 0.60 * max(span_x, span_y)
    # An edge counts as present in a window when it is active for more than
    # ``ON_FRAC`` of it.  "Active at least once" would light up almost every
    # edge in every window - the snapshots would all look identical and the
    # three waves visible in the raster would silently disappear.
    ON_FRAC = 0.34
    for w, (axw, c) in enumerate(zip(ax_s, WIN_CENTRES)):
        sl = slice(c - WIN_HALF, c + WIN_HALF + 1)
        share = act[:, sl].mean(axis=1)
        on = share > ON_FRAC
        for e, (u, v) in enumerate(edges):          # aggregate skeleton
            axw.plot(pos[[u, v], 0], pos[[u, v], 1], color=COLORS["hairline"],
                     lw=0.45, alpha=0.55, zorder=1)
        for e, (u, v) in enumerate(edges):
            if on[e]:
                axw.plot(pos[[u, v], 0], pos[[u, v], 1],
                         color=cols[groups[e]], lw=0.55 + 1.3 * share[e],
                         alpha=0.30 + 0.30 * share[e], zorder=2)
        k = np.zeros(N_NODE)
        for e, (u, v) in enumerate(edges):
            if on[e]:
                k[u] += 1
                k[v] += 1
        for g in range(3):
            sel = labels == g
            axw.scatter(pos[sel, 0], pos[sel, 1], s=7 + 3.6 * k[sel],
                        color=np.where(k[sel] > 0, cols[g],
                                       COLORS["grid"]).tolist(),
                        edgecolor="white", linewidth=0.6, zorder=3)
        axw.set_aspect("equal")
        axw.axis("off")
        axw.set_xlim(cx - half, cx + half)
        axw.set_ylim(cy - half, cy + half)
        axw.set_title(f"W{w + 1}   t = {c - WIN_HALF}-{c + WIN_HALF}",
                      fontsize=7.6, pad=2)
        axw.text(0.5, 0.005, f"{int(on.sum())} edges on > {ON_FRAC:.0%}",
                 ha="center",
                 va="bottom", transform=axw.transAxes, fontsize=6.4,
                 color=COLORS["sub"])

    handles = [Line2D([], [], color=cols[g], lw=1.4, label=GROUP_NAMES[g])
               for g in range(4)]
    handles.append(Line2D([], [], color=COLORS["hairline"], lw=1.0,
                          label="inactive (aggregate)"))
    ax_s[0].legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(2.6, -0.055), ncol=5, fontsize=6.8,
                   handlelength=1.8, columnspacing=1.6, borderaxespad=0.0)

    fig.tight_layout(rect=[0, 0.045, 1, 0.925], h_pad=2.6, w_pad=2.0)
    panel_label_inches(fig, ax_r, "(a)", dx=-0.44, dy=0.12)
    panel_label_inches(fig, ax_m, "(b)", dx=-0.42, dy=0.12)
    panel_label_inches(fig, ax_s[0], "(c)", dx=-0.16, dy=0.10)

    savefig(fig, "370_net_temporal.png")


if __name__ == "__main__":
    main()
