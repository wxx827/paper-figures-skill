"""A genus co-occurrence network that is derived, not drawn.

Abundances for 55 genera across 80 faecal samples are generated from four
latent community drivers plus one opposing gradient, then pushed through a
multinomial so the table is compositional.  Nothing about the modules is
declared to the plot: the network is built by centred-log-ratio transforming
the relative abundances, taking every pairwise Spearman correlation, keeping
the pairs that survive a Benjamini-Hochberg correction at the stated effect
size, and the modules are recovered afterwards by average-linkage clustering
on the correlation distance.

Nodes are then parked on a fixed module-grouped layout rather than a
force-directed one, so that every genus stays visible; the grouping the layout
uses is the recovered partition.  Because that partition is measured and not
assumed, the modularity Q in the main panel, the degree distribution in (b)
and the internal-edge share in (c) are all read back off the same adjacency
matrix, and a module that does not really hold together cannot look tidy here:
its bar in (c) simply falls.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
POS_EDGE = "#a8b4c2"
NEG_EDGE = "#d98032"
MODULE_COLS = ("#b23a48", "#3b5b92", "#2e7d5b", "#8e6bbf")

N_SAMPLES = 80
DEPTH = 26000          # reads per sample
N_LATENT = 4
RHO_CUT = 0.70
FDR_CUT = 0.01
N_MODULES = 4
# Module discs on a wide rectangle so the point cloud fills the panel, with
# the lower row dropped far enough to leave a labelling band under each disc.
MODULE_CENTERS = ((-1.05, 0.40), (1.05, 0.40), (-1.05, -0.62), (1.05, -0.62))
R_MODULE = 0.30
Y_LO, Y_HI = -1.20, 1.30

GENERA = [
    "Bacteroides", "Prevotella", "Faecalibacterium", "Roseburia",
    "Ruminococcus", "Blautia", "Akkermansia", "Bifidobacterium",
    "Lactobacillus", "Escherichia", "Klebsiella", "Enterococcus",
    "Streptococcus", "Veillonella", "Fusobacterium", "Parabacteroides",
    "Alistipes", "Coprococcus", "Dorea", "Eubacterium", "Odoribacter",
    "Sutterella", "Haemophilus", "Collinsella", "Megasphaera",
    "Anaerostipes", "Butyrivibrio", "Clostridium", "Desulfovibrio",
    "Bilophila", "Barnesiella", "Butyricimonas", "Christensenella",
    "Coprobacillus", "Holdemania", "Lachnospira", "Mitsuokella",
    "Oscillospira", "Paraprevotella", "Peptoniphilus",
    "Phascolarctobacterium", "Pseudobutyrivibrio", "Rothia", "Slackia",
    "Subdoligranulum", "Turicibacter", "Actinomyces", "Aggregatibacter",
    "Anaerotruncus", "Atopobium", "Catenibacterium", "Citrobacter",
    "Dialister", "Eggerthella", "Gemella",
]
N_TAXA = len(GENERA)

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.85, 1.24, 5.55, 4.22)
AUX_T = (7.44, 3.60, 3.18, 1.86)
AUX_B = (7.44, 1.24, 3.18, 1.72)


def build_table(rng):
    """Counts driven by four latent factors and one opposing gradient."""
    driver = rng.integers(0, N_LATENT, size=N_TAXA)
    latent = rng.normal(0.0, 1.0, size=(N_SAMPLES, N_LATENT))
    gradient = rng.normal(0.0, 1.0, size=N_SAMPLES)

    load = rng.uniform(0.85, 1.55, size=N_TAXA)
    load[rng.random(N_TAXA) < 0.14] *= -1.0       # a few anti-correlated taxa
    grad_load = np.where(driver == 0, 0.75,
                         np.where(driver == 2, -0.75, 0.0))
    base = rng.lognormal(0.0, 1.15, size=N_TAXA)
    noise = rng.uniform(0.35, 0.85, size=N_TAXA)

    log_ab = (np.log(base)[None, :]
              + load[None, :] * latent[:, driver]
              + grad_load[None, :] * gradient[:, None]
              + rng.normal(0.0, 1.0, size=(N_SAMPLES, N_TAXA)) * noise)
    p = np.exp(log_ab)
    p = p / p.sum(axis=1, keepdims=True)
    counts = np.vstack([rng.multinomial(DEPTH, row) for row in p])
    return counts.astype(float)


def clr(rel, pseudo=0.5 / DEPTH):
    """Centred log ratio, the standard fix for compositional correlation."""
    x = np.log(rel + pseudo)
    return x - x.mean(axis=1, keepdims=True)


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def correlation_network(mat):
    """Spearman rho, BH-corrected, thresholded into a signed adjacency."""
    rho, pval = stats.spearmanr(mat)
    iu = np.triu_indices(N_TAXA, k=1)
    fdr_flat = bh_fdr(pval[iu])
    fdr = np.zeros_like(rho)
    fdr[iu] = fdr_flat
    fdr = fdr + fdr.T
    np.fill_diagonal(fdr, 1.0)

    keep = (np.abs(rho) >= RHO_CUT) & (fdr < FDR_CUT)
    np.fill_diagonal(keep, False)
    signed = np.where(keep, rho, 0.0)
    return rho, signed


def find_modules(rho):
    """Average-linkage clustering on 1 - rho, cut into N_MODULES groups."""
    dist = np.clip(1.0 - rho, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    link = hierarchy.linkage(squareform(dist, checks=False), method="average")
    return hierarchy.fcluster(link, N_MODULES, criterion="maxclust") - 1


def modularity(adj, comm):
    """Newman Q of the recovered partition on the unsigned adjacency."""
    a = np.abs(adj)
    m = a.sum() / 2.0
    if m <= 0.0:
        return 0.0
    deg = a.sum(axis=1)
    q = 0.0
    for c in np.unique(comm):
        mask = comm == c
        q += (a[np.ix_(mask, mask)].sum() / (2.0 * m)
              - (deg[mask].sum() / (2.0 * m)) ** 2)
    return q


def module_layout(comm, deg, centers=MODULE_CENTERS, r_mod=R_MODULE):
    """Modules on a ring, members on a phyllotaxis disc with hubs innermost.

    A force-directed layout would place these nodes too, but its output is
    seed-dependent and it happily overlaps nodes, which makes the degree and
    module statistics beside it harder to read off the picture.  Fixing the
    geometry keeps every node visible and legible; the grouping itself is
    still the one recovered from the correlations, not an input.
    """
    pos = np.zeros((N_TAXA, 2))
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for c in range(N_MODULES):
        idx = np.where(comm == c)[0]
        idx = idx[np.argsort(-deg[idx])]
        cx, cy = centers[c]
        n = max(len(idx), 1)
        for r, t in enumerate(idx):
            rad = r_mod * np.sqrt((r + 0.35) / n)
            a = r * golden
            pos[t] = (cx + rad * np.cos(a), cy + rad * np.sin(a))
    return pos


def network_panel(ax, pos, adj, comm, deg, mean_pc, q):
    iu = np.triu_indices(N_TAXA, k=1)
    for i, j in zip(*iu):
        w = adj[i, j]
        if w == 0.0:
            continue
        neg = w < 0.0
        ax.plot(pos[[i, j], 0], pos[[i, j], 1],
                color=NEG_EDGE if neg else POS_EDGE,
                lw=0.35 + 1.5 * (abs(w) - RHO_CUT) / (1.0 - RHO_CUT),
                ls=(0, (2.6, 1.8)) if neg else "-",
                alpha=0.85 if neg else 0.60, zorder=2)

    size = 12.0 + 78.0 * np.sqrt(mean_pc / mean_pc.max())
    for c in range(N_MODULES):
        m = comm == c
        ax.scatter(pos[m, 0], pos[m, 1], s=size[m],
                   facecolor=lighten(MODULE_COLS[c], 0.30),
                   edgecolor=MODULE_COLS[c], linewidth=0.8, zorder=4,
                   label="module M{:d}  ({:d} genera)".format(c + 1,
                                                              int(m.sum())))

    hubs = []
    for c in range(N_MODULES):
        idx = np.where(comm == c)[0]
        h = int(idx[np.argmax(deg[idx])])
        hubs.append(h)
        cx, cy = MODULE_CENTERS[c]
        # The phyllotaxis puts the highest-degree member at the disc centre,
        # so one ring and one caption per disc are unambiguous without a
        # leader line.
        t = ax.text(cx, cy + R_MODULE + 0.05,
                    "M{:d} hub: {:s}\nk = {:d} edges".format(
                        c + 1, GENERA[h], int(deg[h])),
                    ha="center", va="bottom", fontsize=6.2, color=INK,
                    linespacing=1.30, zorder=7)
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        ax.scatter([pos[h, 0]], [pos[h, 1]], s=size[h] + 58.0,
                   facecolor="none", edgecolor=INK, linewidth=0.9, zorder=8)

    n_edge = int((adj != 0.0).sum() // 2)
    n_neg = int((adj < 0.0).sum() // 2)
    dens = 2.0 * n_edge / (N_TAXA * (N_TAXA - 1.0))
    ax.text(0.015, 0.978,
            "{:d} genera, {:d} edges (density {:.3f})\n"
            "{:d} positive, {:d} negative\n"
            "modularity Q = {:.3f} over {:d} modules"
            .format(N_TAXA, n_edge, dens, n_edge - n_neg, n_neg, q,
                    N_MODULES),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.93), zorder=8)
    ax.text(0.985, 0.020, "node area = mean relative abundance (%)\n"
            "black ring = highest-degree genus of that module",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.0, color=MUTED, linespacing=1.4, zorder=8)
    ax.text(0.015, 0.020, "solid grey edge:  $\\rho$ > 0\n"
            "dashed orange edge:  $\\rho$ < 0", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.0, color=SUB, linespacing=1.4,
            zorder=8)

    leg = ax.legend(loc="upper right", fontsize=6.6, handletextpad=0.40,
                    borderpad=0.40, labelspacing=0.32, scatterpoints=1,
                    frameon=True, framealpha=0.93, edgecolor=HAIR)
    leg.get_frame().set_linewidth(0.6)
    for h in leg.legend_handles:
        h.set_sizes([22.0])

    half = 0.5 * (Y_HI - Y_LO) * MAIN[2] / MAIN[3]
    ax.set_xlim(-half, half)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Edges are Spearman $|\\rho| \\geq$ {:.2f} at FDR < {:.2f} "
                 "across {:d} samples".format(RHO_CUT, FDR_CUT, N_SAMPLES),
                 fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)
    return hubs


def degree_panel(ax, deg):
    top = int(deg.max())
    bins = np.arange(-0.5, top + 1.5, 1.0)
    ax.hist(deg, bins=bins, color=lighten(BLUE, 0.42), edgecolor=BLUE,
            linewidth=0.6, zorder=3)
    ax.axvline(deg.mean(), color=ORANGE, lw=1.1, ls=(0, (3.4, 2.2)), zorder=4)
    ax.text(deg.mean() + 0.5, 0.94, "mean k = {:.1f}".format(deg.mean()),
            transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=6.1, color=ORANGE)
    ax.text(0.985, 0.80, "{:d} of {:d} genera unconnected"
            .format(int((deg == 0).sum()), N_TAXA), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=SUB)

    ax.set_xlim(-0.7, top + 0.9)
    ax.set_xlabel("node degree k (number of edges)", fontsize=7.0)
    ax.set_ylabel("genera (count)", fontsize=7.0)
    ax.tick_params(labelsize=6.3)
    ax.grid(axis="y", lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Degree distribution of the network beside it", fontsize=8.0,
                 fontweight="bold", color=INK, pad=5.0)


def module_panel(ax, adj, comm, mean_pc):
    y = np.arange(N_MODULES)[::-1]
    for yy, c in zip(y, range(N_MODULES)):
        m = comm == c
        inside = int((adj[np.ix_(m, m)] != 0.0).sum() // 2)
        # Row sum counts every endpoint owned by a member, so an internal
        # edge is counted twice and an external one once.
        external = int((adj[m] != 0.0).sum()) - 2 * inside
        share = 100.0 * inside / max(inside + external, 1)
        ax.barh(yy, share, height=0.58, color=lighten(MODULE_COLS[c], 0.34),
                edgecolor=MODULE_COLS[c], linewidth=0.7, zorder=3)
        ax.text(share + 1.6, yy, "{:.0f} %".format(share), ha="left",
                va="center", fontsize=6.2, color=INK, zorder=5)
        ax.text(0.985, yy, "{:d} genera, {:.1f} % of reads"
                .format(int(m.sum()), float(mean_pc[m].sum())),
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6, color=SUB, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(["M{:d}".format(c + 1) for c in range(N_MODULES)],
                       fontsize=6.8)
    ax.set_ylim(y.min() - 0.60, y.max() + 0.64)
    ax.set_xlim(0.0, 118.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("edges kept inside the module (% of that module's edges)",
                  fontsize=7.0)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("How self-contained each recovered module is", fontsize=8.0,
                 fontweight="bold", color=INK, pad=5.0)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(9622)
    counts = build_table(rng)
    rel = counts / counts.sum(axis=1, keepdims=True)
    assert np.allclose(rel.sum(axis=1), 1.0), "abundance vectors must sum to 1"

    rho, adj = correlation_network(clr(rel))
    comm = find_modules(rho)
    deg = (adj != 0.0).sum(axis=1).astype(float)
    q = modularity(adj, comm)
    mean_pc = rel.mean(axis=0) * 100.0
    pos = module_layout(comm, deg)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_deg = axes_inches(fig, AUX_T)
    ax_mod = axes_inches(fig, AUX_B)

    hubs = network_panel(ax_main, pos, adj, comm, deg, mean_pc, q)
    degree_panel(ax_deg, deg)
    module_panel(ax_mod, adj, comm, mean_pc)

    n_edge = int((adj != 0.0).sum() // 2)
    n_neg = int((adj < 0.0).sum() // 2)
    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=-0.50, dy=0.14)
    panel_labels_inches(fig, [ax_deg, ax_mod], labels=["(b)", "(c)"],
                        dx=-0.46, dy=0.14)
    savefig(fig, "962_mgn_cooccurrence_net.png")


if __name__ == "__main__":
    main()
