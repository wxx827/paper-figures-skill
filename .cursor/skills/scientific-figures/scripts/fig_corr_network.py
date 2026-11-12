"""Correlation network: nodes sized by strength, edges weighted by |r|.

The correlation matrix comes from a three-factor generative model (greenness,
heat/dryness, urbanisation), so the graph has real block structure instead of
random links. Node coordinates are produced by a Fruchterman-Reingold layout in
which the attraction of an edge scales with |r|, which pulls each factor's
variables into its own cluster. Labels are pushed radially outward from the
layout centre so they never sit on a node or on the title.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig

NODES = ["NDVI", "NPP", "SOC", "LST", "PRE", "ET", "POP", "GDP", "SLP"]
LONG = {"NDVI": "greenness", "NPP": "productivity", "SOC": "soil carbon",
        "LST": "surface temp.", "PRE": "precipitation",
        "ET": "evapotranspiration", "POP": "population", "GDP": "economy",
        "SLP": "slope"}
CAT = [0, 0, 0, 1, 1, 1, 2, 2, 2]
CAT_NAMES = ["Vegetation and carbon", "Climate and water",
             "Human pressure and terrain"]
CAT_COL = ["#2e7d5b", "#3b5b92", "#c0603f"]
POS_C, NEG_C = "#2a8f86", "#b23a48"
THR = 0.45
#            greenness  heat  urbanisation
LOADINGS = np.array([
    [0.85, -0.35, -0.30],   # NDVI
    [0.88, -0.20, -0.15],   # NPP
    [0.70, -0.10, -0.45],   # SOC
    [-0.30, 0.90, 0.35],    # LST
    [0.35, -0.80, -0.05],   # PRE
    [0.62, 0.45, -0.10],    # ET
    [-0.20, 0.25, 0.88],    # POP
    [-0.15, 0.20, 0.86],    # GDP
    [0.30, -0.15, -0.55],   # SLP
])


def corr_matrix(rng, n_obs=400):
    f = rng.normal(0, 1, (3, n_obs))
    unique = np.sqrt(np.clip(1 - (LOADINGS ** 2).sum(1), 0.05, None))
    x = LOADINGS @ f + unique[:, None] * rng.normal(0, 1, (len(NODES), n_obs))
    return np.corrcoef(x)


def fr_layout(w, rng, iters=400):
    """Fruchterman-Reingold with edge weights as attraction multipliers."""
    n = w.shape[0]
    pos = rng.normal(0, 0.35, (n, 2))
    k = np.sqrt(1.0 / n)
    temp = 0.20
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, np.inf)
        unit = delta / dist[:, :, None]
        rep = (k ** 2 / dist)[:, :, None] * unit
        finite = np.where(np.isinf(dist), 0.0, dist)
        att = (finite / k * w)[:, :, None] * unit
        disp = (rep - att).sum(axis=1)
        length = np.linalg.norm(disp, axis=1)
        length[length == 0] = 1.0
        pos += disp / length[:, None] * np.minimum(length, temp)[:, None]
        temp *= 0.985
    pos -= pos.mean(0)
    # rotate the principal axis onto x, then stretch each axis independently,
    # so the graph fills the canvas instead of running off one diagonal
    _, _, vt = np.linalg.svd(pos, full_matrices=False)
    pos = pos @ vt.T
    pos -= (pos.max(0) + pos.min(0)) / 2
    return pos / (np.abs(pos).max(0) + 1e-9)


def main():
    set_pub_style()
    rng = np.random.default_rng(481)
    c = corr_matrix(rng)
    n = len(NODES)
    adj = np.abs(c) * (np.abs(c) >= THR)
    np.fill_diagonal(adj, 0.0)
    pos = fr_layout(adj, rng)

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    for i in range(n):
        for j in range(i + 1, n):
            r = c[i, j]
            if abs(r) < THR:
                continue
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=POS_C if r > 0 else NEG_C,
                    lw=0.6 + 3.4 * (abs(r) - THR) / (1 - THR),
                    alpha=0.30 + 0.45 * (abs(r) - THR) / (1 - THR), zorder=1,
                    solid_capstyle="round")

    strength = adj.sum(1)
    sizes = 130 + 520 * strength / strength.max()
    for i, nm in enumerate(NODES):
        ax.scatter(*pos[i], s=sizes[i], color=CAT_COL[CAT[i]],
                   edgecolors="white", linewidths=1.3, zorder=3)
        d = pos[i] / (np.linalg.norm(pos[i]) + 1e-9)
        off = pos[i] + d * 0.20
        ax.text(off[0], off[1], nm, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#222222", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.30, 1.30)
    ax.axis("off")
    ax.set_title(f"Correlation network of environmental variables "
                 f"(|r| $\\geq$ {THR:.2f})", fontsize=10.5, fontweight="bold",
                 pad=6)

    handles = [Line2D([], [], marker="o", color="none", markersize=8,
                      markerfacecolor=CAT_COL[g], markeredgecolor="white",
                      label=CAT_NAMES[g]) for g in range(3)]
    handles += [Line2D([], [], color=POS_C, lw=2.4, label="Positive r"),
                Line2D([], [], color=NEG_C, lw=2.4, label="Negative r"),
                Line2D([], [], color="#999999", lw=0.8, label="|r| = 0.45"),
                Line2D([], [], color="#999999", lw=4.0, label="|r| = 1.00")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=4, fontsize=8, frameon=False, handletextpad=0.5,
              columnspacing=1.6, labelspacing=0.6)

    key = ", ".join(f"{k} = {v}" for k, v in list(LONG.items())[:5])
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    savefig(fig, "48_corr_network.png")


if __name__ == "__main__":
    main()
