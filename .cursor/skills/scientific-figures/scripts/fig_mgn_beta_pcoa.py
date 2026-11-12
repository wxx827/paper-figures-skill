"""Beta diversity of three gut cohorts: one distance matrix, read three ways.

Per-sample genus counts are drawn from a multinomial at a fixed sequencing
depth, so each relative-abundance vector sums to unity and the Bray-Curtis
matrix built from them is a genuine dissimilarity matrix rather than a set of
plotted coordinates dressed up as one.

Everything downstream is that matrix.  The ordination is a classical principal
coordinates analysis: double-centre the squared distances, take the
eigendecomposition, scale the eigenvectors by the square root of their
eigenvalues.  The percentages on the axes are those eigenvalues, the scree
bars are the same eigenvalues in full, and the broken-stick line is computed
from their count.  The confidence ellipses come from the 2x2 sample covariance
of the points inside them, and the PERMANOVA F is a real permutation test on
the distance matrix, so the p value quoted in the main panel cannot drift away
from the picture it annotates.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
GREEN = "#2e7d5b"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

GROUPS = ("Control", "Antibiotic", "Recovery")
COLS = (BLUE, ORANGE, GREEN)
NS = (18, 18, 16)
DEPTH = 28000          # reads per sample
N_TAXA = 42
N_PERM = 999
CONF = 0.95

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.85, 1.24, 5.55, 4.22)
AUX_T = (7.44, 3.60, 3.18, 1.86)
AUX_B = (7.44, 1.24, 3.18, 1.72)


def build_table(rng):
    """Per-sample genus counts for three cohorts, plus the group labels."""
    base = rng.lognormal(mean=0.0, sigma=1.30, size=N_TAXA)
    idx = rng.permutation(N_TAXA)
    shift = np.zeros((3, N_TAXA))
    shift[1, idx[:12]] = rng.uniform(0.95, 1.90, 12)
    shift[1, idx[12:22]] = -rng.uniform(0.90, 1.75, 10)
    shift[2, idx[:12]] = 0.46 * shift[1, idx[:12]]
    shift[2, idx[12:22]] = 0.38 * shift[1, idx[12:22]]
    shift[2, idx[22:30]] = rng.uniform(0.50, 1.15, 8)
    disp = rng.uniform(0.42, 0.95, size=N_TAXA)

    rows, labels = [], []
    for g, n in enumerate(NS):
        for _ in range(n):
            p = np.exp(np.log(base) + shift[g] + rng.normal(0.0, disp))
            p = p / p.sum()
            rows.append(rng.multinomial(DEPTH, p))
            labels.append(g)
    return np.asarray(rows, dtype=float), np.asarray(labels)


def bray_curtis(rel):
    """Bray-Curtis dissimilarity between every pair of abundance vectors."""
    n = rel.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        num = np.abs(rel[i] - rel).sum(axis=1)
        den = (rel[i] + rel).sum(axis=1)
        d[i] = num / den
    np.fill_diagonal(d, 0.0)
    return 0.5 * (d + d.T)


def pcoa(d):
    """Classical principal coordinates: Gower centring then eigendecomposition."""
    n = d.shape[0]
    a = -0.5 * d ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    g = j @ a @ j
    g = 0.5 * (g + g.T)
    w, v = np.linalg.eigh(g)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    pos = w > 1.0e-9
    coords = v[:, pos] * np.sqrt(w[pos])
    expl = 100.0 * w[pos] / w[pos].sum()
    return coords, expl


def permanova(d, labels, rng, n_perm=N_PERM):
    """Anderson's pseudo-F on the distance matrix, with a permutation p."""
    n = len(labels)
    keys = np.unique(labels)
    a = len(keys)
    d2 = d ** 2
    ss_t = d2.sum() / (2.0 * n)

    def ss_within(lab):
        s = 0.0
        for k in keys:
            m = lab == k
            s += d2[np.ix_(m, m)].sum() / (2.0 * m.sum())
        return s

    sw = ss_within(labels)
    f_obs = ((ss_t - sw) / (a - 1.0)) / (sw / (n - a))
    lab = labels.copy()
    hits = 1
    for _ in range(n_perm):
        rng.shuffle(lab)
        s = ss_within(lab)
        f = ((ss_t - s) / (a - 1.0)) / (s / (n - a))
        if f >= f_obs:
            hits += 1
    return f_obs, hits / (n_perm + 1.0), (ss_t - sw) / ss_t


def broken_stick(p):
    """Expected eigenvalue share, in per cent, under the null of no structure."""
    inv = 1.0 / np.arange(1, p + 1)
    return 100.0 * np.array([inv[k:].sum() / p for k in range(p)])


def ellipse_from_cov(ax, x, y, color):
    """Confidence ellipse from the real 2x2 covariance of the points shown."""
    cov = np.cov(x, y)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    ang = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    scale = stats.chi2.ppf(CONF, 2)
    width, height = 2.0 * np.sqrt(scale * np.maximum(w, 0.0))
    ax.add_patch(Ellipse((x.mean(), y.mean()), width, height, angle=ang,
                         facecolor=lighten(color, 0.86), edgecolor=color,
                         linewidth=0.9, alpha=0.55, zorder=2))
    return width * height * np.pi / 4.0


def ordination_panel(ax, coords, expl, labels, fstat, pval, r2):
    areas = []
    for g, (name, col, n) in enumerate(zip(GROUPS, COLS, NS)):
        m = labels == g
        x, y = coords[m, 0], coords[m, 1]
        areas.append(ellipse_from_cov(ax, x, y, col))
        ax.scatter(x, y, s=26.0, facecolor=lighten(col, 0.32), edgecolor=col,
                   linewidth=0.7, zorder=4,
                   label="{:s}  (n = {:d})".format(name, n))
        ax.plot([x.mean()], [y.mean()], marker="P", ms=7.0, mfc=col,
                mec="white", mew=0.8, zorder=6)

    ax.axhline(0.0, color=HAIR, lw=0.6, zorder=1)
    ax.axvline(0.0, color=HAIR, lw=0.6, zorder=1)
    ax.set_xlabel("PCoA axis 1  ({:.1f} % of variation)".format(expl[0]),
                  fontsize=7.4)
    ax.set_ylabel("PCoA axis 2  ({:.1f} % of variation)".format(expl[1]),
                  fontsize=7.4)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    floor = pval <= 1.0 / (N_PERM + 1.0)
    pt = "p = {:.3f}{:s}".format(pval, ", the permutation floor" if floor
                                 else "")
    ax.text(0.015, 0.975,
            "PERMANOVA on Bray-Curtis\n"
            "pseudo-F = {:.2f},  $R^2$ = {:.3f}\n{:s}\n({:,d} permutations)"
            .format(fstat, r2, pt, N_PERM),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.93), zorder=8)
    ax.text(0.985, 0.022, "ellipse = {:.0f} % confidence region from the "
            "sample covariance".format(100.0 * CONF), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.0, color=MUTED, zorder=8)
    leg = ax.legend(loc="upper right", fontsize=6.8, handletextpad=0.45,
                    borderpad=0.4, labelspacing=0.35, frameon=True,
                    framealpha=0.93, edgecolor=HAIR)
    leg.get_frame().set_linewidth(0.6)
    ax.set_title("Principal coordinates of {:d} faecal samples on the "
                 "Bray-Curtis matrix".format(len(labels)), fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)
    return areas


def scree_panel(ax, expl):
    n_show = 8
    v = expl[:n_show]
    x = np.arange(n_show)
    bs = broken_stick(len(expl))[:n_show]

    ax.bar(x, v, width=0.66, color=lighten(BLUE, 0.42), edgecolor=BLUE,
           linewidth=0.6, zorder=3)
    ax.step(np.concatenate([x - 0.5, [x[-1] + 0.5]]),
            np.concatenate([bs, bs[-1:]]), where="post", color=ORANGE,
            lw=1.1, ls=(0, (3.4, 2.2)), zorder=4)
    keep = int(np.sum(v > bs))
    for i in range(min(keep, 3)):
        ax.text(x[i], v[i] + 0.55, "{:.1f} %".format(v[i]), ha="center",
                va="bottom", fontsize=6, color=SUB)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x])
    ax.set_xlim(-0.62, n_show - 0.38)
    ax.set_ylim(0.0, float(v.max()) * 1.30)
    ax.set_xlabel("PCoA axis (-)", fontsize=7.0)
    ax.set_ylabel("variation explained (%)", fontsize=7.0)
    ax.tick_params(labelsize=6.3)
    ax.grid(axis="y", lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.985, 0.955, "dashed: broken-stick null", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=ORANGE)
    ax.text(0.985, 0.815, "{:d} axes exceed it; first two hold {:.1f} %"
            .format(keep, float(v[:2].sum())), transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=SUB)
    ax.set_title("Eigenvalue spectrum of the same matrix", fontsize=8.0,
                 fontweight="bold", color=INK, pad=5.0)


def dispersion_panel(ax, d, labels):
    """Within-group and between-group Bray-Curtis distances, median and IQR."""
    iu = np.triu_indices(d.shape[0], k=1)
    same = labels[iu[0]] == labels[iu[1]]
    rows = []
    for g, name in enumerate(GROUPS):
        m = same & (labels[iu[0]] == g)
        rows.append((name, d[iu][m], COLS[g]))
    rows.append(("Between groups", d[iu][~same], "#777777"))

    y = np.arange(len(rows))[::-1]
    for yy, (name, v, col) in zip(y, rows):
        q1, med, q3 = np.percentile(v, [25.0, 50.0, 75.0])
        ax.plot([q1, q3], [yy, yy], color=col, lw=2.0,
                solid_capstyle="round", zorder=3)
        ax.plot([med], [yy], marker="o", ms=4.6, mfc="white", mec=col,
                mew=1.3, zorder=4)
        ax.text(0.985, yy + 0.30, "median {:.3f}   n = {:,d} pairs"
                .format(med, len(v)), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6, color=SUB)

    within = [r[1] for r in rows[:3]]
    h, p_disp = stats.kruskal(*within)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_ylim(y.min() - 0.62, y.max() + 0.68)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Bray-Curtis dissimilarity (-)", fontsize=7.0)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.015, 0.045, "Kruskal-Wallis on within-group spread: "
            "H = {:.2f}, p = {:.3f}".format(h, p_disp),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0,
            color=MUTED)
    ax.set_title("Dispersion within each cohort versus between cohorts",
                 fontsize=8.0, fontweight="bold", color=INK, pad=5.0)
    return p_disp


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
    rng = np.random.default_rng(9612)
    counts, labels = build_table(rng)
    rel = counts / counts.sum(axis=1, keepdims=True)
    assert np.allclose(rel.sum(axis=1), 1.0), "abundance vectors must sum to 1"

    d = bray_curtis(rel)
    coords, expl = pcoa(d)
    fstat, pval, r2 = permanova(d, labels, rng)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_scree = axes_inches(fig, AUX_T)
    ax_disp = axes_inches(fig, AUX_B)

    ordination_panel(ax_main, coords, expl, labels, fstat, pval, r2)
    scree_panel(ax_scree, expl)
    p_disp = dispersion_panel(ax_disp, d, labels)

    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=-0.50, dy=0.14)
    panel_labels_inches(fig, [ax_scree, ax_disp], labels=["(b)", "(c)"],
                        dx=-0.46, dy=0.14)
    savefig(fig, "961_mgn_beta_pcoa.png")


if __name__ == "__main__":
    main()
