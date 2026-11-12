"""Predicted functional pathways across three cohorts, from one table.

Per-sample abundances for 24 KEGG-style pathways are drawn so that each
sample's profile sums to unity, which is what a gene-family prediction such as
PICRUSt actually returns: a composition, not a set of independent counts.

The heat map shows the row z-score of the log abundance, so a row says where a
sample sits relative to the other samples for that pathway and nothing else -
the raw scale lives in the stacked bars of panel (c), where each cohort's five
category shares are the means of those same profiles and therefore add to
exactly 100 %.  Panel (b) ranks the pathways by a Kruskal-Wallis test across
the three cohorts with Benjamini-Hochberg correction, so a pathway is called
out there only if its row in the heat map really does separate.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from pubstyle import set_pub_style, savefig, get_cmap, lighten
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

GROUPS = ("Lean", "Obese", "Post-bypass")
GROUP_COLS = ("#3b5b92", "#d98032", "#2e7d5b")
NS = (15, 15, 15)
FDR_CUT = 0.05
N_TOP = 8

CATEGORIES = ("Carbohydrate", "Amino acid", "Energy", "Transport", "Repair")
CAT_COLS = ("#b23a48", "#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf")
CAT_FULL = ("Carbohydrate metabolism", "Amino acid metabolism",
            "Energy metabolism", "Membrane transport",
            "Replication and repair")

# (short label, category index) - short enough to fit the label gutter
PATHWAYS = [
    ("Glycolysis", 0), ("Pentose phosphate", 0), ("Starch and sucrose", 0),
    ("Butanoate metab.", 0), ("Propanoate metab.", 0), ("Galactose metab.", 0),
    ("Ala/Asp/Glu metab.", 1), ("Gly/Ser/Thr metab.", 1),
    ("Lysine biosynthesis", 1), ("Lysine degradation", 1),
    ("Phenylalanine metab.", 1), ("Tryptophan metab.", 1),
    ("Oxidative phosphoryl.", 2), ("Methane metab.", 2),
    ("Nitrogen metab.", 2), ("Sulfur metab.", 2), ("Carbon fixation", 2),
    ("ABC transporters", 3), ("PTS sugar uptake", 3),
    ("Secretion systems", 3),
    ("DNA replication", 4), ("Mismatch repair", 4),
    ("Homologous recomb.", 4), ("Nucleotide exc. repair", 4),
]
N_PATH = len(PATHWAYS)
CAT_OF = np.array([c for _, c in PATHWAYS])

FIG_W, FIG_H = 11.0, 6.5
MAIN = (2.02, 1.24, 4.38, 4.22)
AUX_T = (8.08, 3.60, 2.54, 1.86)
AUX_B = (8.08, 1.24, 2.54, 1.72)
BAND_X0, BAND_X1 = -1.15, -0.30      # category strip, in column units
BAND_Y0, BAND_Y1 = -1.05, -0.30      # cohort strip, in row units


def build_table(rng):
    """Compositional pathway profiles for three cohorts."""
    base = rng.lognormal(0.0, 0.72, size=N_PATH)
    base = base / base.sum()
    shift = np.zeros((3, N_PATH))
    idx = rng.permutation(N_PATH)
    shift[1, idx[:6]] = rng.uniform(0.30, 0.62, 6)
    shift[1, idx[6:11]] = -rng.uniform(0.28, 0.55, 5)
    shift[2, idx[:6]] = 0.34 * shift[1, idx[:6]]
    shift[2, idx[6:11]] = 0.30 * shift[1, idx[6:11]]
    shift[2, idx[11:15]] = rng.uniform(0.25, 0.50, 4)
    disp = rng.uniform(0.08, 0.17, size=N_PATH)

    rows, labels = [], []
    for g, n in enumerate(NS):
        for _ in range(n):
            p = np.exp(np.log(base) + shift[g] + rng.normal(0.0, disp))
            rows.append(p / p.sum())
            labels.append(g)
    return np.asarray(rows), np.asarray(labels)


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def order_columns(z, labels):
    """Cluster samples inside each cohort so blocks read cleanly."""
    out = []
    for g in range(len(GROUPS)):
        idx = np.where(labels == g)[0]
        link = hierarchy.linkage(pdist(z[:, idx].T, metric="correlation"),
                                 method="average")
        out.append(idx[hierarchy.leaves_list(link)])
    return np.concatenate(out)


def heatmap_panel(fig, ax, z, labels, order):
    cmap = get_cmap("cb_diverging")
    vmax = float(np.abs(z).max())
    im = ax.imshow(z[:, order], cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest",
                   extent=[0.0, len(order), float(N_PATH), 0.0], zorder=2)

    # cohort strip above the map
    start = 0
    for g, n in enumerate(NS):
        ax.add_patch(Rectangle((start, BAND_Y0), n, BAND_Y1 - BAND_Y0,
                               facecolor=GROUP_COLS[g], edgecolor="white",
                               linewidth=0.8, clip_on=False, zorder=4))
        ax.text(start + n / 2.0, (BAND_Y0 + BAND_Y1) / 2.0,
                "{:s}  (n = {:d})".format(GROUPS[g], n), ha="center",
                va="center", fontsize=6.4, color="white", fontweight="bold",
                clip_on=False, zorder=5)
        if start > 0:
            ax.axvline(start, color="white", lw=1.4, zorder=3)
        start += n

    # category strip to the left of the map
    for c in range(len(CATEGORIES)):
        rows = np.where(CAT_OF == c)[0]
        r0, r1 = float(rows.min()), float(rows.max() + 1)
        ax.add_patch(Rectangle((BAND_X0, r0), BAND_X1 - BAND_X0, r1 - r0,
                               facecolor=CAT_COLS[c], edgecolor="white",
                               linewidth=0.8, clip_on=False, zorder=4))
        ax.text((BAND_X0 + BAND_X1) / 2.0, (r0 + r1) / 2.0, CATEGORIES[c],
                ha="center", va="center", rotation=90, fontsize=6,
                color="white", fontweight="bold", clip_on=False, zorder=5)

    ax.set_xlim(BAND_X0 - 0.05, len(order))
    ax.set_ylim(float(N_PATH) + 0.10, BAND_Y0 - 0.10)
    ax.set_yticks(np.arange(N_PATH) + 0.5)
    ax.set_yticklabels([p for p, _ in PATHWAYS], fontsize=6.0)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=2.0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.set_xlabel("{:d} samples, ordered inside each cohort by profile "
                  "similarity".format(len(order)), fontsize=7.4)
    ax.set_title("Row z-score of predicted pathway abundance",
                 fontsize=9.0, fontweight="bold", color=INK, pad=7.0)

    cb = safe_colorbar(fig, im, ax, side="right", width=0.11, pad=0.09,
                       shrink=0.80, labelsize=6.2)
    cb.set_label("row z-score of $\\log_{10}$ abundance (SD)", fontsize=6.4,
                 color=INK, labelpad=3.0)
    return vmax


def ranking_panel(ax, rel, labels, fdr, hstat):
    top = np.argsort(fdr)[:N_TOP]
    top = top[np.argsort(fdr[top])[::-1]]
    y = np.arange(len(top))
    val = -np.log10(np.clip(fdr[top], 1.0e-12, None))

    for yy, j, v in zip(y, top, val):
        col = CAT_COLS[CAT_OF[j]]
        ax.barh(yy, v, height=0.62, color=lighten(col, 0.34), edgecolor=col,
                linewidth=0.6, zorder=3)
        ax.text(v + 0.18, yy, "H = {:.1f}".format(hstat[j]), ha="left",
                va="center", fontsize=6, color=MUTED, zorder=5)

    ax.axvline(-np.log10(FDR_CUT), color="#333333", lw=0.7,
               ls=(0, (3.4, 2.2)), zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([PATHWAYS[j][0] for j in top], fontsize=6.0)
    ax.set_ylim(-0.68, len(top) - 0.32)
    ax.set_xlim(0.0, float(val.max()) * 1.30)
    ax.set_xlabel("$-\\log_{10}$ FDR (-)", fontsize=7.0)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-np.log10(FDR_CUT) + 0.10, -0.58, "FDR = {:.2f}".format(FDR_CUT),
            ha="left", va="center", fontsize=6, color=MUTED)
    ax.set_title("Pathways ranked by Kruskal-Wallis across the three cohorts",
                 fontsize=8.0, fontweight="bold", color=INK, pad=5.0)
    return top


def composition_panel(ax, rel, labels):
    """Category shares per cohort; each bar is a partition of 100 %."""
    y = np.arange(len(GROUPS))[::-1]
    for yy, g in zip(y, range(len(GROUPS))):
        prof = rel[labels == g].mean(axis=0)
        share = np.array([100.0 * prof[CAT_OF == c].sum()
                          for c in range(len(CATEGORIES))])
        assert abs(share.sum() - 100.0) < 1.0e-9, "stack must total 100 %"
        left = 0.0
        for c, v in enumerate(share):
            ax.barh(yy, v, left=left, height=0.58, color=CAT_COLS[c],
                    edgecolor="white", linewidth=0.8, zorder=3)
            if v >= 8.0:
                ax.text(left + v / 2.0, yy, "{:.0f}".format(v), ha="center",
                        va="center", fontsize=6.0, color="white",
                        fontweight="bold", zorder=5)
            left += v

    ax.set_yticks(y)
    ax.set_yticklabels(GROUPS, fontsize=6.8)
    ax.set_ylim(y.min() - 0.62, y.max() + 0.66)
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of predicted gene families (%); colours as in the "
                  "category strip of (a)", fontsize=7.0)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Category composition of each cohort, bars total 100 %",
                 fontsize=8.0, fontweight="bold", color=INK, pad=5.0)


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
    rng = np.random.default_rng(9633)
    rel, labels = build_table(rng)
    assert np.allclose(rel.sum(axis=1), 1.0), "profiles must sum to 1"

    log_ab = np.log10(rel).T                      # pathways x samples
    z = (log_ab - log_ab.mean(axis=1, keepdims=True)) / \
        log_ab.std(axis=1, keepdims=True)
    order = order_columns(z, labels)

    hstat = np.empty(N_PATH)
    pval = np.empty(N_PATH)
    for j in range(N_PATH):
        groups = [rel[labels == g, j] for g in range(len(GROUPS))]
        hstat[j], pval[j] = stats.kruskal(*groups)
    fdr = bh_fdr(pval)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_map = axes_inches(fig, MAIN)
    ax_rank = axes_inches(fig, AUX_T)
    ax_comp = axes_inches(fig, AUX_B)

    heatmap_panel(fig, ax_map, z, labels, order)
    top = ranking_panel(ax_rank, rel, labels, fdr, hstat)
    composition_panel(ax_comp, rel, labels)

    n_sig = int((fdr < FDR_CUT).sum())
    panel_labels_inches(fig, [ax_map], labels=["(a)"], dx=-0.50, dy=0.14)
    panel_labels_inches(fig, [ax_rank, ax_comp], labels=["(b)", "(c)"],
                        dx=-0.46, dy=0.14)
    savefig(fig, "963_mgn_pathway_heatmap.png")


if __name__ == "__main__":
    main()
