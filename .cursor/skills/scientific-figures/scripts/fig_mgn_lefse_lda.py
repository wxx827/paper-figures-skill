"""Differential abundance of gut genera between two cohorts, three ways.

One count table drives the whole page.  Per-sample genus counts are drawn
from a multinomial at a realistic sequencing depth, so relative abundances
sum to unity by construction and prevalence (the fraction of samples with a
non-zero count) is a property of the table rather than a separate invention.

Every number shown is computed from that table: the two-sided Mann-Whitney U
test gives the p value, Benjamini-Hochberg gives the FDR, the rank-biserial
correlation gives the effect size, and the LDA score is the base-10 logarithm
of the between-cohort separation of class means expressed in parts per
million, which is the scale LEfSe reports.  The main panel therefore cannot
disagree with the volcano beside it - a genus is drawn as a bar if and only
if it clears both thresholds marked on the volcano.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
ORANGE = "#d98032"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

GROUPS = ("Healthy", "Colitis")
N_A, N_B = 24, 26
DEPTH = 32000          # reads per sample
PSEUDO = 1.0e-5        # relative-abundance pseudocount for the log ratio
FDR_CUT = 0.05
LDA_CUT = 2.0

GENERA = [
    "Bacteroides", "Prevotella", "Faecalibacterium", "Roseburia",
    "Ruminococcus", "Blautia", "Akkermansia", "Bifidobacterium",
    "Lactobacillus", "Escherichia", "Klebsiella", "Enterococcus",
    "Streptococcus", "Veillonella", "Fusobacterium", "Parabacteroides",
    "Alistipes", "Coprococcus", "Dorea", "Eubacterium", "Odoribacter",
    "Sutterella", "Haemophilus", "Collinsella", "Megasphaera",
    "Anaerostipes",
]
# genus index -> log-scale shift applied to the Colitis cohort
SHIFTS = {2: -1.35, 3: -1.10, 5: -0.55, 6: -1.25, 17: -0.80, 25: -0.70,
          20: -0.60, 9: 1.55, 10: 1.40, 11: 1.05, 12: 0.85, 14: 1.30,
          13: 0.95, 22: 0.75}

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.85, 1.24, 5.55, 4.22)
AUX_T = (7.44, 3.60, 3.18, 1.86)
AUX_B = (7.44, 1.24, 3.18, 1.72)


def build_table(rng):
    """Per-sample genus counts for both cohorts, plus the group labels."""
    n_tx = len(GENERA)
    # Community-level template: a few dominant genera, a long tail.
    base = np.sort(rng.lognormal(mean=0.0, sigma=1.35, size=n_tx))[::-1]
    base = base[rng.permutation(n_tx)]
    disp = rng.uniform(0.55, 1.15, size=n_tx)

    rows, labels = [], []
    for gi, n in enumerate((N_A, N_B)):
        shift = np.zeros(n_tx)
        if gi == 1:
            for k, v in SHIFTS.items():
                shift[k] = v
        for _ in range(n):
            logp = np.log(base) + shift + rng.normal(0.0, disp)
            p = np.exp(logp)
            p = p / p.sum()
            rows.append(rng.multinomial(DEPTH, p))
            labels.append(gi)
    return np.asarray(rows, dtype=float), np.asarray(labels)


def rank_biserial(a, b):
    """Effect size that matches the Mann-Whitney U actually reported."""
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def test_table(rel, labels):
    """Mann-Whitney U, BH FDR, LDA score and log2 ratio for every genus."""
    a_mask, b_mask = labels == 0, labels == 1
    n_tx = rel.shape[1]
    pval = np.empty(n_tx)
    delta = np.empty(n_tx)
    lda = np.empty(n_tx)
    lfc = np.empty(n_tx)
    for j in range(n_tx):
        a, b = rel[a_mask, j], rel[b_mask, j]
        pval[j] = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        delta[j] = rank_biserial(b, a)
        # LDA score: class-mean separation in parts per million, logged.
        sep = abs(b.mean() - a.mean()) * 1.0e6
        lda[j] = np.log10(1.0 + sep)
        lfc[j] = np.log2((b.mean() + PSEUDO) / (a.mean() + PSEUDO))
    return pval, bh_fdr(pval), delta, lda, lfc


def lda_panel(ax, sel, lda, lfc, fdr):
    """Diverging LDA bars, signed by the cohort each genus is enriched in."""
    signed = np.where(lfc[sel] > 0.0, lda[sel], -lda[sel])
    order = np.argsort(signed)
    idx = sel[order]
    val = signed[order]
    y = np.arange(len(idx))

    for yy, v, j in zip(y, val, idx):
        col = ORANGE if v > 0 else BLUE
        ax.barh(yy, v, height=0.66, color=lighten(col, 0.18),
                edgecolor=col, linewidth=0.6, zorder=3)
        pad = 0.13 if v > 0 else -0.13
        ax.text(v + pad, yy, GENERA[j], ha="left" if v > 0 else "right",
                va="center", fontsize=6.4, color=INK, zorder=5)
        inner = -0.13 if v > 0 else 0.13
        ax.text(inner, yy, "FDR {:.1e}".format(fdr[j]),
                ha="right" if v > 0 else "left", va="center", fontsize=6,
                color=MUTED, zorder=5)

    lim = float(np.abs(val).max()) * 1.56
    for s in (-1.0, 1.0):
        ax.axvline(s * LDA_CUT, color=HAIR, lw=0.7, ls=(0, (3.4, 2.2)),
                   zorder=2)
    ax.axvline(0.0, color="#333333", lw=0.9, zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.9, len(idx) - 0.2)
    ax.set_yticks([])
    ticks = np.arange(-np.floor(lim), np.floor(lim) + 0.1, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["{:.0f}".format(abs(t)) for t in ticks])
    ax.set_xlabel("LDA score, $\\log_{10}$ of class-mean separation (ppm)",
                  fontsize=7.4)
    ax.tick_params(labelsize=6.4)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.text(-lim * 0.97, len(idx) - 0.55,
            "enriched in {:s}  (n = {:d})".format(GROUPS[0], N_A), ha="left",
            va="center", fontsize=7.2, color=BLUE, fontweight="bold")
    ax.text(lim * 0.97, len(idx) - 0.55,
            "enriched in {:s}  (n = {:d})".format(GROUPS[1], N_B), ha="right",
            va="center", fontsize=7.2, color=ORANGE, fontweight="bold")
    ax.text(LDA_CUT, -0.72, "threshold  LDA = {:.1f}".format(LDA_CUT),
            ha="left", va="center", fontsize=6, color=MUTED)
    ax.set_title("Genera passing FDR < {:.2f} and LDA > {:.1f}, one bar per "
                 "genus".format(FDR_CUT, LDA_CUT), fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)
    return idx


def volcano_panel(ax, rel, lfc, fdr, lda, sel):
    mean_pc = rel.mean(axis=0) * 100.0
    y = -np.log10(np.clip(fdr, 1.0e-12, None))
    hit = np.zeros(len(lfc), dtype=bool)
    hit[sel] = True
    size = 6.0 + 46.0 * np.sqrt(mean_pc / mean_pc.max())

    ax.scatter(lfc[~hit], y[~hit], s=size[~hit], facecolor="#d6d6d6",
               edgecolor="#9a9a9a", linewidth=0.4, zorder=3)
    for mask, col in ((hit & (lfc < 0), BLUE), (hit & (lfc > 0), ORANGE)):
        ax.scatter(lfc[mask], y[mask], s=size[mask],
                   facecolor=lighten(col, 0.35), edgecolor=col, linewidth=0.7,
                   zorder=4)
    ax.axhline(-np.log10(FDR_CUT), color="#333333", lw=0.7,
               ls=(0, (3.4, 2.2)), zorder=2)
    ax.axvline(0.0, color=HAIR, lw=0.6, zorder=1)

    top = sel[np.argsort(-lda[sel])][:3]
    for j in top:
        ax.annotate(GENERA[j], (lfc[j], y[j]),
                    xytext=(4.0 if lfc[j] > 0 else -4.0, 3.0),
                    textcoords="offset points", fontsize=6, color=SUB,
                    ha="left" if lfc[j] > 0 else "right", va="bottom")

    xr = float(np.abs(lfc).max()) * 1.22
    ax.set_xlim(-xr, xr)
    ax.set_ylim(-0.25, float(y.max()) * 1.24)
    ax.set_xlabel("$\\log_2$ ratio of mean relative abundance "
                  "(Colitis / Healthy)", fontsize=7.0)
    ax.set_ylabel("$-\\log_{10}$ FDR (-)", fontsize=7.0)
    ax.tick_params(labelsize=6.3)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.015, 0.965, "{:d} of {:d} genera significant"
            .format(int(hit.sum()), len(GENERA)), transform=ax.transAxes,
            ha="left", va="top", fontsize=6.2, color=SUB)
    ax.text(0.985, 0.965, "area = mean abundance", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.0, color=MUTED)
    ax.set_title("Volcano of the same tests, dashed line at FDR = "
                 "{:.2f}".format(FDR_CUT), fontsize=8.0, fontweight="bold",
                 color=INK, pad=5.0)


def profile_panel(ax, rel, labels, idx, fdr, delta):
    """Group medians with IQR for the strongest genera, in per cent."""
    show = list(idx[::-1])[:5]
    a_mask, b_mask = labels == 0, labels == 1
    y = np.arange(len(show))[::-1]

    for yy, j in zip(y, show):
        for mask, col, off in ((a_mask, BLUE, 0.17), (b_mask, ORANGE, -0.17)):
            v = rel[mask, j] * 100.0
            q1, med, q3 = np.percentile(v, [25.0, 50.0, 75.0])
            ax.plot([q1, q3], [yy + off, yy + off], color=col, lw=1.5,
                    solid_capstyle="round", zorder=3)
            ax.plot([med], [yy + off], marker="o", ms=4.2, mfc="white",
                    mec=col, mew=1.2, zorder=4)
        star = ("***" if fdr[j] < 1e-3 else "**" if fdr[j] < 1e-2 else "*")
        ax.text(0.985, yy, "{:s}  d = {:+.2f}".format(star, delta[j]),
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6, color=SUB)

    ax.set_yticks(y)
    ax.set_yticklabels([GENERA[j] for j in show], fontsize=6.4)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.62)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 60.0)
    ax.set_xlabel("relative abundance (%), median and IQR", fontsize=7.0)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.5, alpha=0.4, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Five strongest genera; d is the rank-biserial effect size",
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
    rng = np.random.default_rng(9601)
    counts, labels = build_table(rng)
    rel = counts / counts.sum(axis=1, keepdims=True)
    assert np.allclose(rel.sum(axis=1), 1.0), "relative abundances must sum to 1"

    pval, fdr, delta, lda, lfc = test_table(rel, labels)
    sel = np.where((fdr < FDR_CUT) & (lda > LDA_CUT))[0]
    prev = 100.0 * np.mean(counts > 0.0, axis=0)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_vol = axes_inches(fig, AUX_T)
    ax_prof = axes_inches(fig, AUX_B)

    idx = lda_panel(ax_main, sel, lda, lfc, fdr)
    volcano_panel(ax_vol, rel, lfc, fdr, lda, sel)
    profile_panel(ax_prof, rel, labels, idx, fdr, delta)

    up = int(np.sum(lfc[sel] > 0.0))
    panel_labels_inches(fig, [ax_main], labels=["(a)"], dx=-0.50, dy=0.14)
    panel_labels_inches(fig, [ax_vol, ax_prof], labels=["(b)", "(c)"],
                        dx=-0.46, dy=0.14)
    savefig(fig, "960_mgn_lefse_lda.png")


if __name__ == "__main__":
    main()
