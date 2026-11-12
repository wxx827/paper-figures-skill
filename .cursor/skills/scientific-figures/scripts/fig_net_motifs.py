"""Motif census of a spatial network against a degree-preserving null model.

The top row draws the six connected three- and four-node subgraphs that are
counted, and colours each icon by the verdict reached below it: over-
represented, under-represented, or indistinguishable from the null.

Panel (b) puts the observed census next to the null ensemble on one shared log
axis - motif counts span three orders of magnitude, so a linear axis would
show the wedge bar and nothing else.  Panel (c) converts the same numbers into
z-scores, which is the quantity the significance claim is actually made on.

Counting is exact and induced (every four-node subset is classified by its
edge count and degree sequence, fully vectorised), and the null is 24
degree-preserving double-edge-swap rewirings of the same graph, so the
comparison holds the degree sequence fixed and varies only the wiring.
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from pubstyle import CATEGORICAL, COLORS, savefig, set_pub_style
from figkit import panel_label_inches

SEED = 31
N_NODE = 40
RADIUS = 0.235          # geometric connection radius
N_NULL = 24
Z_CRIT = 1.96

# key, printable name, node coordinates, edge list  (icon geometry)
MOTIFS = [
    ("wedge", "Wedge\n(open triad)",
     [(-0.9, -0.5), (0.0, 0.55), (0.9, -0.5)], [(0, 1), (1, 2)]),
    ("triangle", "Triangle\n(closed triad)",
     [(-0.9, -0.5), (0.0, 0.55), (0.9, -0.5)], [(0, 1), (1, 2), (0, 2)]),
    ("path4", "4-path",
     [(-1.0, -0.45), (-0.35, 0.5), (0.35, -0.5), (1.0, 0.45)],
     [(0, 1), (1, 2), (2, 3)]),
    ("star4", "4-star\n(claw)",
     [(0.0, 0.15), (-0.95, -0.5), (0.0, 1.0), (0.95, -0.5)],
     [(0, 1), (0, 2), (0, 3)]),
    ("cycle4", "4-cycle",
     [(-0.8, 0.6), (0.8, 0.6), (0.8, -0.6), (-0.8, -0.6)],
     [(0, 1), (1, 2), (2, 3), (3, 0)]),
    ("k4", "4-clique",
     [(-0.8, 0.6), (0.8, 0.6), (0.8, -0.6), (-0.8, -0.6)],
     [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]),
]


# --------------------------------------------------------------------------
def geometric_graph(n, r, seed=0):
    """Random geometric graph - high clustering, so triangles are plentiful."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    A = ((d < r) & ~np.eye(n, dtype=bool)).astype(np.int8)
    return A, pts


def rewire(A, rng, mult=14):
    """Degree-preserving double-edge swaps.

    The null has to hold the degree sequence fixed.  An Erdos-Renyi null with
    the same edge count would call every motif significant, because it gets
    the *degrees* wrong long before it gets the wiring wrong - the z-scores
    would then measure degree heterogeneity, not structure.
    """
    B = A.copy()
    iu, ju = np.triu_indices(B.shape[0], 1)
    m = B[iu, ju] > 0
    e = np.column_stack([iu[m], ju[m]])
    n_e = len(e)
    for _ in range(mult * n_e):
        i, j = rng.integers(0, n_e, 2)
        if i == j:
            continue
        u, v = e[i]
        x, y = e[j]
        if rng.random() < 0.5:
            x, y = y, x
        if len({u, v, x, y}) < 4 or B[u, y] or B[x, v]:
            continue
        B[u, v] = B[v, u] = 0
        B[x, y] = B[y, x] = 0
        B[u, y] = B[y, u] = 1
        B[x, v] = B[v, x] = 1
        e[i] = (u, y)
        e[j] = (x, v)
    return B


def census(A, c3, c4):
    """Induced counts of the six motifs, vectorised over node subsets."""
    a = A.astype(np.int16)
    # --- 3-node subsets ---
    e3 = (a[c3[:, 0], c3[:, 1]] + a[c3[:, 0], c3[:, 2]] +
          a[c3[:, 1], c3[:, 2]])
    out = {"wedge": int((e3 == 2).sum()), "triangle": int((e3 == 3).sum())}
    # --- 4-node subsets ---
    p = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    ee = np.stack([a[c4[:, i], c4[:, j]] for i, j in p])
    e4 = ee.sum(0)
    deg = np.stack([ee[0] + ee[1] + ee[2],      # node 0
                    ee[0] + ee[3] + ee[4],      # node 1
                    ee[1] + ee[3] + ee[5],      # node 2
                    ee[2] + ee[4] + ee[5]])     # node 3
    dmax = deg.max(0)
    out["path4"] = int(((e4 == 3) & (dmax == 2)).sum())
    out["star4"] = int(((e4 == 3) & (dmax == 3)).sum())
    out["cycle4"] = int(((e4 == 4) & (dmax == 2)).sum())
    out["k4"] = int((e4 == 6).sum())
    return out


# --------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    A, _ = geometric_graph(N_NODE, RADIUS, seed=SEED)
    c3 = np.array(list(itertools.combinations(range(N_NODE), 3)))
    c4 = np.array(list(itertools.combinations(range(N_NODE), 4)))

    obs = census(A, c3, c4)
    null = {k: np.empty(N_NULL) for k in obs}
    for r in range(N_NULL):
        cc = census(rewire(A, rng), c3, c4)
        for k in obs:
            null[k][r] = cc[k]

    keys = [m[0] for m in MOTIFS]
    o = np.array([obs[k] for k in keys], dtype=float)
    mu = np.array([null[k].mean() for k in keys])
    sd = np.array([null[k].std(ddof=1) for k in keys])
    z = (o - mu) / np.maximum(sd, 1e-9)

    def verdict_colour(zz):
        if zz > Z_CRIT:
            return COLORS["accent"]
        if zz < -Z_CRIT:
            return CATEGORICAL[1]
        return COLORS["baseline"]

    vcols = [verdict_colour(zz) for zz in z]
    deg = A.sum(1)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.46, 1.0], hspace=0.42)
    top = gs[0].subgridspec(1, 6, wspace=0.10)
    bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 0.58], wspace=0.24)
    ax_i = [fig.add_subplot(top[0, i]) for i in range(6)]
    ax_c = fig.add_subplot(bot[0, 0])
    ax_z = fig.add_subplot(bot[0, 1])

    # ---- (a) motif icons ---------------------------------------------------
    for k, (axm, (key, name, pts, es)) in enumerate(zip(ax_i, MOTIFS)):
        pts = np.array(pts, dtype=float)
        for u, v in es:
            axm.plot(pts[[u, v], 0], pts[[u, v], 1], color=vcols[k], lw=1.5,
                     alpha=0.65, zorder=1, solid_capstyle="round")
        axm.scatter(pts[:, 0], pts[:, 1], s=58, color=vcols[k],
                    edgecolor="white", linewidth=0.6, zorder=3)
        axm.set_aspect("equal")
        axm.axis("off")
        axm.set_xlim(-1.45, 1.45)
        axm.set_ylim(-1.20, 1.45)
        axm.text(0.5, -0.06, name, transform=axm.transAxes, ha="center",
                 va="top", fontsize=7.4, color=COLORS["ink"],
                 fontweight="bold", linespacing=1.25)
        axm.text(0.5, 1.02, f"z = {z[k]:+.1f}", transform=axm.transAxes,
                 ha="center", va="bottom", fontsize=7.0, color=vcols[k])

    # ---- (b) observed vs null, shared log axis ----------------------------
    x = np.arange(6)
    ax_c.bar(x - 0.19, o, width=0.36, color=vcols, edgecolor="white",
             linewidth=0.7, label="observed", zorder=3)
    ax_c.bar(x + 0.19, mu, width=0.36, color=COLORS["grid"],
             edgecolor="white", linewidth=0.7, zorder=3,
             label=f"rewired null (mean of {N_NULL})")
    ax_c.errorbar(x + 0.19, mu, yerr=sd, fmt="none", ecolor=COLORS["rule"],
                  elinewidth=0.8, capsize=2.0, zorder=4)
    ax_c.set_yscale("log")
    ax_c.set_ylim(0.6, max(o.max(), mu.max()) * 7.0)
    for xi, oi, mi in zip(x, o, mu):
        ax_c.text(xi - 0.19, oi * 1.22, f"{int(oi)}", ha="center",
                  va="bottom", fontsize=6.4, color=COLORS["ink"])
        ax_c.text(xi + 0.19, mi * 1.22, f"{mi:.0f}", ha="center", va="bottom",
                  fontsize=6.4, color=COLORS["sub"])
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([m[1] for m in MOTIFS], fontsize=7.0,
                         linespacing=1.2)
    ax_c.set_ylabel("induced subgraph count (log)", fontsize=8.0)
    ax_c.tick_params(labelsize=7.0)
    ax_c.set_title("Census against a degree-preserving null", fontsize=9.0,
                   loc="left", pad=4)
    ax_c.legend(loc="upper right", fontsize=7.0, handlelength=1.5,
                borderaxespad=0.3, ncol=2)
    ax_c.grid(axis="y", lw=0.5, color=COLORS["grid"], alpha=0.55)
    ax_c.text(0.0, -0.235, f"random geometric graph · N = {N_NODE} · "
              f"E = {int(A.sum() / 2)} · mean degree {deg.mean():.1f} · "
              f"null holds every degree fixed", transform=ax_c.transAxes,
              ha="left", va="top", fontsize=6.9, color=COLORS["sub"])

    # ---- (c) z-scores ------------------------------------------------------
    ax_z.axhspan(-Z_CRIT, Z_CRIT, color=COLORS["card"], zorder=0)
    ax_z.axhline(0.0, color=COLORS["rule"], lw=0.8, zorder=2)
    for s in (-Z_CRIT, Z_CRIT):
        ax_z.axhline(s, color=COLORS["baseline"], lw=0.7, ls="--", zorder=2)
    ax_z.bar(x, z, width=0.56, color=vcols, edgecolor="white", linewidth=0.7,
             zorder=3)
    span = max(abs(z).max(), Z_CRIT) * 1.42
    for xi, zi in zip(x, z):
        ax_z.text(xi, zi + np.sign(zi) * span * 0.045, f"{zi:+.1f}",
                  ha="center", va="bottom" if zi >= 0 else "top",
                  fontsize=6.5, color=COLORS["ink"])
    ax_z.set_ylim(-span, span)
    ax_z.set_xticks(x)
    ax_z.set_xticklabels([m[1].split("\n")[0] for m in MOTIFS], fontsize=6.8,
                         rotation=28, ha="right")
    ax_z.set_ylabel("z-score vs null (-)", fontsize=8.0)
    ax_z.tick_params(labelsize=7.0)
    ax_z.set_title("Significance of each motif", fontsize=9.0, loc="left",
                   pad=4)
    ax_z.text(0.985, 0.055, f"shaded band: |z| < {Z_CRIT}",
              transform=ax_z.transAxes, ha="right", va="bottom",
              fontsize=6.5, color=COLORS["sub"])

    leg = [Line2D([], [], color=COLORS["accent"], lw=3.0,
                  label="over-represented"),
           Line2D([], [], color=CATEGORICAL[1], lw=3.0,
                  label="under-represented"),
           Line2D([], [], color=COLORS["baseline"], lw=3.0,
                  label="not significant")]
    ax_i[0].legend(handles=leg, loc="upper center", bbox_to_anchor=(3.35, 1.42),
                   ncol=3, fontsize=7.2, handlelength=1.6, columnspacing=1.8,
                   borderaxespad=0.0)

    fig.tight_layout(rect=[0, 0.015, 1, 0.905], h_pad=2.6, w_pad=2.0)
    panel_label_inches(fig, ax_i[0], "(a)", dx=-0.10, dy=0.30)
    panel_label_inches(fig, ax_c, "(b)", dx=-0.46, dy=0.12)
    panel_label_inches(fig, ax_z, "(c)", dx=-0.46, dy=0.12)

    savefig(fig, "371_net_motifs.png")


if __name__ == "__main__":
    main()
