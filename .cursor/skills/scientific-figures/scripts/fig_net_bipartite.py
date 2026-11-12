"""A two-mode pollination network and the one-mode projection it implies.

The main panel is the bipartite graph itself: pollinator species on the left,
plant species on the right, one line per observed visit.  Both columns are
ordered by barycentric sweeps rather than by name, which is what keeps the
modular block structure visible instead of hiding it behind crossing edges.

The subordinate panels answer the question the two-mode drawing always
provokes: what does the network look like once one side is projected away, and
how differently connected are the two node classes.  The projection is drawn
with a weighted spring layout - edge width is the number of plants two
pollinators share - and the degree panel puts both classes on one shared axis
so that "plants are visited by more partners than pollinators visit" can
actually be read off the figure.

No networkx: adjacency, ordering, projection and layout are numpy.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pubstyle import (CATEGORICAL, COLORS, lighten, savefig, set_pub_style)
from figkit import halo_text, panel_label_inches, shared_limits

SEED = 19
N_POL = 16            # pollinator species (left column)
N_PLA = 22            # plant species (right column)
N_MOD = 3             # planted modules shared by both classes

COL_POL = CATEGORICAL[1]      # house blue
COL_PLA = CATEGORICAL[2]      # house green


# --------------------------------------------------------------------------
# two-mode graph
# --------------------------------------------------------------------------
def bipartite_graph(n_a, n_b, n_mod, seed=0, p_in=0.55, p_out=0.055):
    """Modular *and* nested incidence matrix, with no isolated node.

    Pure modularity gives three disconnected blocks, which is not what a real
    interaction network looks like; pure nestedness gives one solid triangle
    with no blocks to see.  Multiplying a block term by a per-node generality
    term produces both at once: specialists stay inside their module, the few
    generalists reach across, and the projection below has something to show.
    """
    rng = np.random.default_rng(seed)
    lab_a = np.sort(rng.integers(0, n_mod, n_a))
    lab_b = np.sort(rng.integers(0, n_mod, n_b))
    # generality: a handful of broad species, most of them narrow
    g_a = np.sort(rng.gamma(1.6, 1.0, n_a))[rng.permutation(n_a)]
    g_b = np.sort(rng.gamma(1.6, 1.0, n_b))[rng.permutation(n_b)]
    g_a, g_b = g_a / g_a.mean(), g_b / g_b.mean()
    same = lab_a[:, None] == lab_b[None, :]
    p = np.clip(np.where(same, p_in, p_out) * np.outer(g_a, g_b), 0, 1)
    B = (rng.random((n_a, n_b)) < p).astype(float)
    # every species must have at least one partner, else the drawing shows a
    # node floating with no explanation
    for i in np.flatnonzero(B.sum(1) == 0):
        cand = np.flatnonzero(lab_b == lab_a[i])
        B[i, rng.choice(cand if cand.size else np.arange(n_b))] = 1.0
    for j in np.flatnonzero(B.sum(0) == 0):
        cand = np.flatnonzero(lab_a == lab_b[j])
        B[rng.choice(cand if cand.size else np.arange(n_a)), j] = 1.0
    return B, lab_a, lab_b


def barycentre_order(B, sweeps=40):
    """Alternating barycentre sweeps - the classic crossing-reduction heuristic.

    Ordering both columns by module label alone still leaves the inter-module
    edges sweeping across the whole panel.  Repeatedly placing each node at the
    mean height of its partners settles into an ordering where the modules
    stack in a compatible sequence on both sides.
    """
    n_a, n_b = B.shape
    ya = np.linspace(0, 1, n_a)
    yb = np.linspace(0, 1, n_b)
    for _ in range(sweeps):
        yb = np.where(B.sum(0) > 0, (B * ya[:, None]).sum(0) /
                      np.maximum(B.sum(0), 1), yb)
        yb = np.argsort(np.argsort(yb)) / max(n_b - 1, 1)
        ya = np.where(B.sum(1) > 0, (B * yb[None, :]).sum(1) /
                      np.maximum(B.sum(1), 1), ya)
        ya = np.argsort(np.argsort(ya)) / max(n_a - 1, 1)
    return np.argsort(ya), np.argsort(yb)


def spring_layout(W, seed=0, iters=420, k_scale=1.45, t0=0.05, gravity=4.5):
    """Weighted Fruchterman-Reingold, rescaled to fixed RMS every step.

    Gravity is unusually strong here because the projection has a few isolated
    nodes: with nothing pulling on them they ride the repulsion out to the rim
    and, once the layout is renormalised to a fixed RMS radius, squeeze the
    connected core into an illegible knot in the middle.
    """
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    ang = rng.permutation(n) / n * 2 * np.pi
    pos = 0.5 * np.column_stack([np.cos(ang), np.sin(ang)])
    pos += 0.02 * rng.standard_normal((n, 2))
    k = k_scale / np.sqrt(n)
    w = W.sum(1) + 1.0
    w = w / w.mean()
    wij = np.outer(w, w)
    t = t0
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1))
        np.fill_diagonal(dist, 1e9)
        unit = delta / dist[..., None]
        force = (wij * k ** 2 / dist)[..., None] * unit
        force -= (W * dist / k)[..., None] * unit
        force = force.sum(1) - gravity * pos
        mag = np.sqrt((force ** 2).sum(-1))[:, None] + 1e-12
        pos = pos + force / mag * np.minimum(mag, t)
        pos -= pos.mean(0)
        pos *= 0.5 / (np.sqrt((pos ** 2).sum(1).mean()) + 1e-12)
        t *= 0.993
    return pos


def fit_limits(ax, pos, box_aspect, pad=0.16):
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

    B, lab_a, lab_b = bipartite_graph(N_POL, N_PLA, N_MOD, seed=SEED)
    oa, ob = barycentre_order(B)
    B = B[np.ix_(oa, ob)]
    lab_a, lab_b = lab_a[oa], lab_b[ob]

    deg_a, deg_b = B.sum(1), B.sum(0)
    n_edges = int(B.sum())
    connectance = n_edges / (N_POL * N_PLA)

    xa, xb = 0.0, 1.0
    ya = np.linspace(0, 1.55, N_POL)
    yb = np.linspace(0, 1.55, N_PLA)

    # one-mode projection onto the pollinators
    P = B @ B.T
    np.fill_diagonal(P, 0.0)
    P_cut = np.where(P >= 2, P, 0.0)
    pos = spring_layout(P_cut, seed=SEED)

    fig = plt.figure(figsize=(10.4, 6.3))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.56],
                  height_ratios=[1.0, 0.66], hspace=0.34, wspace=0.06)
    ax = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- (a) the two-mode graph -------------------------------------------
    mod_cols = [lighten(CATEGORICAL[i], 0.25) for i in range(N_MOD)]
    for i in range(N_POL):
        for j in np.flatnonzero(B[i] > 0):
            within = lab_a[i] == lab_b[j]
            ax.plot([xa, xb], [ya[i], yb[j]],
                    color=mod_cols[lab_a[i]] if within else COLORS["baseline"],
                    lw=0.75 if within else 0.9,
                    alpha=0.30 if within else 0.34, zorder=1,
                    solid_capstyle="round")

    ax.scatter(np.full(N_POL, xa), ya, s=13 + 5.0 * deg_a, color=COL_POL,
               edgecolor="white", linewidth=0.6, zorder=4)
    ax.scatter(np.full(N_PLA, xb), yb, s=13 + 5.0 * deg_b, color=COL_PLA,
               edgecolor="white", linewidth=0.6, zorder=4)

    for i in range(N_POL):
        ax.text(xa - 0.055, ya[i], f"P{i + 1:02d}", ha="right", va="center",
                fontsize=6.1, color=COLORS["sub"], zorder=5)
    for j in range(N_PLA):
        ax.text(xb + 0.055, yb[j], f"V{j + 1:02d}", ha="left", va="center",
                fontsize=6.1, color=COLORS["sub"], zorder=5)

    top_a = int(np.argmax(deg_a))
    top_b = int(np.argmax(deg_b))
    halo_text(ax, xa + 0.035, ya[top_a], f"generalist, {int(deg_a[top_a])} partners",
              ha="left", va="center", fontsize=6.4, color=COLORS["ink"],
              halo="white", halo_lw=2.6, zorder=6)
    halo_text(ax, xb - 0.035, yb[top_b], f"{int(deg_b[top_b])} visitors",
              ha="right", va="center", fontsize=6.4, color=COLORS["ink"],
              halo="white", halo_lw=2.6, zorder=6)

    halo_text(ax, xa, ya.max() + 0.11,
              f"Pollinators\n(mode A, n = {N_POL})", ha="center", va="bottom",
              fontsize=7.8, fontweight="bold", color=COL_POL, halo="white",
              halo_lw=2.8, zorder=7)
    halo_text(ax, xb, yb.max() + 0.11,
              f"Plants\n(mode B, n = {N_PLA})", ha="center", va="bottom",
              fontsize=7.8, fontweight="bold", color=COL_PLA, halo="white",
              halo_lw=2.8, zorder=7)

    ax.set_aspect("equal")
    ax.axis("off")
    fit_limits(ax, np.column_stack([np.r_[np.full(N_POL, xa),
                                          np.full(N_PLA, xb)],
                                    np.r_[ya, yb]]),
               box_aspect=0.92, pad=0.14)
    _y0, _y1 = ax.get_ylim()
    ax.set_ylim(_y0 - 0.20, _y1)          # strip of clear space for the legend
    ax.text(0.02, -0.01, f"{n_edges} observed visits · connectance "
            f"{connectance:.2f} · marker area $\\propto$ degree",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.9,
            color=COLORS["sub"])

    leg = [Line2D([], [], color=mod_cols[0], lw=1.2,
                  label="within-module visit"),
           Line2D([], [], color=COLORS["baseline"], lw=1.2,
                  label="cross-module visit")]
    ax.legend(handles=leg, loc="lower left", fontsize=6.8, handlelength=1.8,
              borderaxespad=0.2)

    # ---- (b) one-mode projection ------------------------------------------
    iu, ju = np.triu_indices(N_POL, 1)
    wmask = P_cut[iu, ju] > 0
    wmax = P_cut.max() if P_cut.max() > 0 else 1.0
    for i, j in zip(iu[wmask], ju[wmask]):
        ax_p.plot(pos[[i, j], 0], pos[[i, j], 1], color=COLORS["baseline"],
                  lw=0.35 + 1.5 * P_cut[i, j] / wmax,
                  alpha=0.18 + 0.17 * P_cut[i, j] / wmax, zorder=1,
                  solid_capstyle="round")
    for m in range(N_MOD):
        sel = lab_a == m
        ax_p.scatter(pos[sel, 0], pos[sel, 1], s=14 + 5.5 * deg_a[sel],
                     color=mod_cols[m], edgecolor="white", linewidth=0.6,
                     zorder=3)
    ax_p.set_aspect("equal")
    ax_p.axis("off")
    fit_limits(ax_p, pos, box_aspect=1.35, pad=0.20)
    ax_p.set_title("Pollinator projection\n(link = $\\geq$ 2 shared plants)",
                   fontsize=8.2, loc="left", pad=3)
    n_proj = int(wmask.sum())
    ax_p.text(0.0, 0.0, f"{n_proj} projected links · node colour = module",
              transform=ax_p.transAxes, ha="left", va="bottom",
              fontsize=6.5, color=COLORS["sub"])

    # ---- (c) degree distribution, one shared axis for both classes --------
    lo, hi = shared_limits([deg_a, deg_b], pad=0.0, lo=0.0)
    bins = np.arange(0, np.ceil(hi) + 2) - 0.5
    ha_, _ = np.histogram(deg_a, bins=bins)
    hb_, _ = np.histogram(deg_b, bins=bins)
    centres = 0.5 * (bins[1:] + bins[:-1])
    ax_d.bar(centres - 0.20, ha_, width=0.38, color=COL_POL,
             edgecolor="white", linewidth=0.6, label="pollinators")
    ax_d.bar(centres + 0.20, hb_, width=0.38, color=COL_PLA,
             edgecolor="white", linewidth=0.6, label="plants")
    ax_d.axvline(deg_a.mean(), color=COL_POL, lw=1.0, ls="--", zorder=4)
    ax_d.axvline(deg_b.mean(), color=COL_PLA, lw=1.0, ls="--", zorder=4)
    ax_d.text(deg_a.mean(), max(ha_.max(), hb_.max()) * 1.06,
              f"mean {deg_a.mean():.1f}", ha="center", va="bottom",
              fontsize=6.3, color=COL_POL)
    ax_d.text(deg_b.mean(), max(ha_.max(), hb_.max()) * 1.20,
              f"mean {deg_b.mean():.1f}", ha="center", va="bottom",
              fontsize=6.3, color=COL_PLA)
    ax_d.set_xlim(-0.6, np.ceil(hi) + 0.6)
    ax_d.set_ylim(0, max(ha_.max(), hb_.max()) * 1.42)
    ax_d.set_xlabel("degree (number of partners)", fontsize=7.5)
    ax_d.set_ylabel("species count", fontsize=7.5)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_title("Degree of both modes, one scale", fontsize=8.2,
                   loc="left", pad=3)
    ax_d.legend(loc="upper right", fontsize=6.8, handlelength=1.4,
                borderaxespad=0.3)
    ax_d.grid(axis="y", lw=0.5, color=COLORS["grid"], alpha=0.55)

    fig.tight_layout(rect=[0, 0.01, 1, 0.925], h_pad=2.2, w_pad=2.2)
    # (b) and (c) need a wider offset: an equal-aspect axes shrinks at draw
    # time, so its title starts to the right of the nominal box edge.
    panel_label_inches(fig, ax, "(a)", dx=-0.02, dy=0.14)
    panel_label_inches(fig, ax_p, "(b)", dx=-0.44, dy=0.14)
    panel_label_inches(fig, ax_d, "(c)", dx=-0.44, dy=0.14)

    savefig(fig, "369_net_bipartite.png")


if __name__ == "__main__":
    main()
