"""Quality-control dashboard for a four-sample single-cell run.

Six peer panels: the three per-cell distributions that decide the filter
(genes, UMI, mitochondrial fraction), the joint UMI-gene scatter that shows
what the filter actually removes, the retained-cell tally and the doublet
score.  The three distribution panels measure different quantities, so they
deliberately do *not* share a y axis - each title names its unit instead.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import (add_colorbar, darken, get_cmap, letter_panels, lighten,
                      savefig, set_pub_style)

SAMPLES = ["S1 healthy", "S2 healthy", "S3 tumour", "S4 tumour"]
BASE, ACCENT = "#3b5b92", "#b23a48"
# S4 is the one sample that fails QC: the single emphasis object in the figure.
SAMPLE_COLORS = [BASE, BASE, BASE, ACCENT]

MIN_GENES, MAX_GENES = 250, 6500
MIN_UMI, MAX_MITO, MAX_DOUBLET = 500, 12.0, 0.30


def synth_run(rng):
    """Four libraries; the last one is over-digested and mito-heavy."""
    qual = [1.00, 0.94, 0.88, 0.62]
    mito_lvl = [3.2, 3.8, 5.6, 13.5]
    n_cells = [1850, 1620, 1740, 1980]
    out = []
    for q, ml, n in zip(qual, mito_lvl, n_cells):
        genes = rng.lognormal(np.log(1900 * q), 0.52, n)
        umi = genes * rng.lognormal(np.log(2.6), 0.30, n)
        mito = rng.gamma(2.2, ml / 2.2, n) * rng.lognormal(0, 0.25, n)
        dbl = np.clip(rng.beta(1.7, 12.0, n) + 0.35 * (rng.random(n) < 0.045),
                      0, 1)
        out.append(dict(genes=genes, umi=umi, mito=np.clip(mito, 0, 60),
                        doublet=dbl))
    return out


def violin(ax, data, colors, log=False):
    """House violin: pale body, sample-coloured edge, jitter and median dot."""
    parts = ax.violinplot(data, positions=np.arange(len(data)), widths=0.80,
                          showextrema=False, showmedians=False)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(lighten(col, 0.72))
        body.set_edgecolor(col)
        body.set_linewidth(0.8)
        body.set_alpha(1.0)
    rng = np.random.default_rng(7)
    for i, (d, col) in enumerate(zip(data, colors)):
        sub = rng.choice(d, size=min(260, d.size), replace=False)
        ax.scatter(i + rng.normal(0, 0.062, sub.size), sub, s=1.1,
                   color=darken(col, 0.25), alpha=0.22, linewidths=0.0,
                   zorder=3, rasterized=True)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color="#333333", lw=1.0, zorder=4)
        ax.plot([i], [med], marker="o", ms=3.2, mfc="white", mec="#333333",
                mew=0.8, zorder=5, linestyle="none")
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels([s.split()[0] for s in SAMPLES], fontsize=7.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    if log:
        ax.set_yscale("log")


def hline(ax, y, text):
    ax.axhline(y, color=ACCENT, lw=0.9, ls="--", zorder=6)
    ax.text(0.985, y, text, transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=6.2, color=ACCENT)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(533)
    run = synth_run(rng)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    (a, b, c), (d, e, f) = axes

    # ---- (a) genes per cell -------------------------------------------------
    violin(a, [s["genes"] for s in run], SAMPLE_COLORS)
    hline(a, MIN_GENES, f"min {MIN_GENES:,} genes")
    hline(a, MAX_GENES, f"max {MAX_GENES:,} genes")
    a.set_ylim(0, 9200)                       # crop the sparse KDE tail
    a.set_ylabel("genes detected per cell (genes)", fontsize=8.0)
    a.set_title("Gene complexity (genes per cell)", fontsize=8.5,
                fontweight="bold")

    # ---- (b) UMI per cell ---------------------------------------------------
    violin(b, [s["umi"] for s in run], SAMPLE_COLORS, log=True)
    hline(b, MIN_UMI, f"min {MIN_UMI:,} UMI")
    b.set_ylabel("UMI per cell (counts, log scale)", fontsize=8.0)
    b.set_title("Library size (UMI per cell)", fontsize=8.5, fontweight="bold")

    # ---- (c) mitochondrial fraction -----------------------------------------
    violin(c, [s["mito"] for s in run], SAMPLE_COLORS)
    hline(c, MAX_MITO, f"max {MAX_MITO:.0f}% mito")
    c.set_ylim(0, 42)
    c.set_ylabel("mitochondrial reads (% of UMI)", fontsize=8.0)
    c.set_title("Cell stress (% mito)", fontsize=8.5, fontweight="bold")

    # ---- (d) joint UMI-gene scatter -----------------------------------------
    umi = np.concatenate([s["umi"] for s in run])
    gen = np.concatenate([s["genes"] for s in run])
    mito = np.concatenate([s["mito"] for s in run])
    dbl = np.concatenate([s["doublet"] for s in run])
    order = rng.permutation(umi.size)
    sc = d.scatter(umi[order], gen[order], c=mito[order], s=1.6,
                   cmap=get_cmap("cb_sequential_warm"), vmin=0, vmax=25,
                   linewidths=0.0, alpha=0.75, rasterized=True)
    d.set_xscale("log")
    d.set_yscale("log")
    d.set_xlim(d.get_xlim())                  # freeze before adding the box
    d.add_patch(Rectangle((MIN_UMI, MIN_GENES), d.get_xlim()[1] - MIN_UMI,
                          MAX_GENES - MIN_GENES, facecolor="none",
                          edgecolor="#333333", lw=1.0, ls="--", zorder=5))
    keep = ((umi >= MIN_UMI) & (gen >= MIN_GENES) & (gen <= MAX_GENES)
            & (mito <= MAX_MITO) & (dbl <= MAX_DOUBLET))
    d.text(0.035, 0.965, f"{100 * keep.mean():.1f}% of cells inside the "
                         f"filter box", transform=d.transAxes, ha="left",
           va="top", fontsize=6.6, color="#333333")
    d.set_xlabel("UMI per cell (counts)", fontsize=8.0)
    d.set_ylabel("genes per cell (genes)", fontsize=8.0)
    d.set_title("Joint UMI-gene filter", fontsize=8.5, fontweight="bold")
    d.tick_params(labelsize=7.0)
    add_colorbar(fig, sc, d, label="mito content (%)", extend="max")

    # ---- (e) cells before and after -----------------------------------------
    before = np.array([s["genes"].size for s in run])
    after = np.array([int(np.sum((s["umi"] >= MIN_UMI)
                                 & (s["genes"] >= MIN_GENES)
                                 & (s["genes"] <= MAX_GENES)
                                 & (s["mito"] <= MAX_MITO)
                                 & (s["doublet"] <= MAX_DOUBLET)))
                      for s in run])
    x = np.arange(len(run))
    e.bar(x - 0.19, before, width=0.36, color="#cccccc", edgecolor="none",
          label="before filtering")
    e.bar(x + 0.19, after, width=0.36, color=SAMPLE_COLORS, edgecolor="none")
    for xi, (bf, af) in enumerate(zip(before, after)):
        e.text(xi + 0.19, af + 45, f"{100 * af / bf:.0f}%", ha="center",
               va="bottom", fontsize=6.5, color="#333333")
    e.set_xticks(x)
    e.set_xticklabels([s.split()[0] for s in SAMPLES], fontsize=7.0)
    e.set_ylim(0, before.max() * 1.18)
    e.set_ylabel("cells (n)", fontsize=8.0)
    e.set_title("Cells retained per sample (cells)", fontsize=8.5,
                fontweight="bold")
    e.tick_params(labelsize=7.0)
    e.legend(handles=[Line2D([], [], color="#cccccc", lw=5, label="before"),
                      Line2D([], [], color=BASE, lw=5, label="after (pass)"),
                      Line2D([], [], color=ACCENT, lw=5,
                             label="after (flagged sample)")],
             loc="upper right", fontsize=6.6, frameon=False, handlelength=1.1)

    # ---- (f) doublet score ---------------------------------------------------
    bins = np.linspace(0, 1.0, 46)
    cnt, edges = np.histogram(dbl, bins=bins)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    cols = [ACCENT if x0 > MAX_DOUBLET else BASE for x0 in ctr]
    f.bar(ctr, cnt, width=edges[1] - edges[0], color=cols, edgecolor="none")
    f.axvline(MAX_DOUBLET, color="#333333", lw=0.9, ls="--")
    f.set_yscale("log")
    f.set_xlim(0, 1.0)
    f.set_xlabel("doublet score (-)", fontsize=8.0)
    f.set_ylabel("cells (n, log scale)", fontsize=8.0)
    f.set_title("Doublet score per cell (-)", fontsize=8.5, fontweight="bold")
    f.tick_params(labelsize=7.0)
    f.text(0.97, 0.93, f"{100 * np.mean(dbl > MAX_DOUBLET):.1f}% predicted "
                       f"doublets\nthreshold {MAX_DOUBLET:.2f}",
           transform=f.transAxes, ha="right", va="top", fontsize=6.6,
           color="#333333", linespacing=1.3)

    letter_panels(axes)
    fig.tight_layout(rect=[0, 0.035, 1, 0.945], h_pad=2.6, w_pad=2.6)
    savefig(fig, "533_sc_qc_metrics.png")


if __name__ == "__main__":
    main()
