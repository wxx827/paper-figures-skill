"""Minimum inhibitory concentrations for one isolate collection, read three
ways.

The main panel is the raw antibiogram: fourteen antimicrobials against thirty
isolates, every cell shaded by its MIC on the doubling-dilution ladder the
assay actually produces, with a cross on every cell that sits at or above the
clinical breakpoint for that drug.  Both axes are reordered by hierarchical
clustering, so the two multidrug phenotypes in the collection separate on their
own instead of being asserted; the class band on the left shows that the blocks
that emerge are cross-resistance within drug classes.

The subordinate panels turn the same matrix into the two numbers a
surveillance report needs: the per-drug resistance rate with its confidence
interval, and how the resistant classes actually co-occur within isolates.
Every count comes from the plotted matrix.  Synthetic data; no real isolate
collection is represented.
"""
import textwrap
from itertools import combinations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from pubstyle import set_pub_style, get_palette, savefig, despine
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
HAIR = "#cccccc"
BEIGE = "#efe9dc"

# Drug class band.  Six semantic colours, all from the house supplementary
# list; the two-letter codes are expanded in the footnote.
CLASS_COLOR = {"BL": "#3b5b92", "TE": "#2e7d5b", "AG": "#c9a227",
               "MA": "#8e6bbf", "FQ": "#c0603f", "PH": "#2a7f7a"}
CLASS_NAME = {"BL": "beta-lactam", "TE": "tetracycline",
              "AG": "aminoglycoside", "MA": "macrolide",
              "FQ": "fluoroquinolone", "PH": "phenicol"}

# name, class, wild-type MIC exponent (log2 ug/mL), breakpoint exponent, and
# the share of the class-level resistance shift this agent actually takes -
# protected or later-generation members of a class take less of it, which is
# what spreads the per-drug resistance rates apart.
DRUGS = [
    ("Ampicillin", "BL", -2, 3, 1.05), ("Amoxi-clavulanate", "BL", -2, 3, 0.74),
    ("Ceftiofur", "BL", -3, 2, 0.58), ("Tetracycline", "TE", -2, 3, 1.06),
    ("Doxycycline", "TE", -2, 2, 0.72), ("Gentamicin", "AG", -2, 3, 0.64),
    ("Neomycin", "AG", -1, 4, 0.98), ("Spectinomycin", "AG", 2, 6, 1.06),
    ("Tylosin", "MA", 0, 4, 1.06), ("Tilmicosin", "MA", 0, 4, 0.88),
    ("Enrofloxacin", "FQ", -3, 1, 1.06), ("Marbofloxacin", "FQ", -3, 2, 0.74),
    ("Florfenicol", "PH", -1, 3, 0.86), ("Chloramphenicol", "PH", 0, 4, 1.02),
]

# Three latent phenotypes: wild type, and two multidrug patterns that load on
# different classes.  Values are log2 shifts applied to the wild-type MIC.
PHENOTYPES = [
    ("wild type", 12, {"BL": 0.0, "TE": 0.0, "AG": 0.0, "MA": 0.0,
                       "FQ": 0.0, "PH": 0.0}),
    ("MDR-A", 10, {"BL": 6.4, "TE": 6.2, "AG": 0.8, "MA": 0.6,
                   "FQ": 0.5, "PH": 3.6}),
    ("MDR-B", 8, {"BL": 0.6, "TE": 0.8, "AG": 5.6, "MA": 5.8,
                  "FQ": 7.0, "PH": 0.6}),
]

LOG2_LO, LOG2_HI = -3, 7          # 0.125 to 128 ug/mL


def mic_matrix(rng):
    exps, bps, classes, names, frac = [], [], [], [], []
    for nm, cl, wt, bp, fr in DRUGS:
        names.append(nm)
        classes.append(cl)
        exps.append(wt)
        bps.append(bp)
        frac.append(fr)
    exps = np.array(exps, dtype=float)
    bps = np.array(bps, dtype=float)
    frac = np.array(frac, dtype=float)

    cols, pheno = [], []
    for label, k, shift in PHENOTYPES:
        for _ in range(k):
            s = np.array([shift[c] for c in classes], dtype=float) * frac
            # Isolate-level scaling: not every member of a phenotype carries
            # the full resistance load.
            v = exps + s * rng.uniform(0.66, 1.16) + rng.normal(0.0, 0.70,
                                                                exps.size)
            cols.append(np.clip(np.rint(v), LOG2_LO, LOG2_HI))
            pheno.append(label)
    m = np.column_stack(cols)
    return m, bps, np.array(classes), names, pheno


def reorder(m):
    r = leaves_list(linkage(pdist(m, metric="euclidean"), method="average"))
    c = leaves_list(linkage(pdist(m.T, metric="euclidean"),
                            method="average"))
    return r, c


def wilson(k, n, z=1.96):
    p = k / n
    d = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, ctr - h), min(1.0, ctr + h)


def contrast(rgba, cut=0.58):
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < cut else INK


def heat_panel(fig, ax, ax_band, m, bps, classes, names, res):
    steps = np.arange(LOG2_LO, LOG2_HI + 1)
    cmap = ListedColormap(get_palette("cb_sequential_warm", steps.size))
    norm = BoundaryNorm(np.arange(LOG2_LO - 0.5, LOG2_HI + 0.6, 1.0),
                        cmap.N)
    im = ax.imshow(m, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")

    n_drug, n_iso = m.shape
    for i in range(n_drug):
        for j in range(n_iso):
            if res[i, j]:
                ax.plot([j], [i], marker="x", ms=3.2, mew=1.0,
                        color=contrast(cmap(norm(m[i, j]))), zorder=4)

    ax.set_xticks(np.arange(-0.5, n_iso, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_drug, 1.0), minor=True)
    ax.grid(which="minor", color="white", lw=0.45)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(np.arange(0, n_iso, 5))
    ax.set_xticklabels([f"{i + 1}" for i in range(0, n_iso, 5)], fontsize=6.6)
    ax.set_yticks([])
    ax.set_xlabel(f"isolate, hierarchically clustered (n={n_iso})",
                  fontsize=8.0)
    ax.set_title("Cross-resistance sorts the collection into two blocks",
                 fontsize=9.0, fontweight="bold", color=INK)
    for s in ax.spines.values():
        s.set_visible(False)

    for i, cl in enumerate(classes):
        fc = CLASS_COLOR[cl]
        ax_band.add_patch(Rectangle((0.0, i - 0.5), 1.0, 1.0, facecolor=fc,
                                    edgecolor="white", lw=0.5))
        ax_band.text(0.5, i, cl, ha="center", va="center", fontsize=6.0,
                     color=contrast(mpl.colors.to_rgba(fc)))
    ax_band.set_xlim(0.0, 1.0)
    ax_band.set_ylim(n_drug - 0.5, -0.5)
    ax_band.set_xticks([])
    ax_band.set_yticks(np.arange(n_drug))
    ax_band.set_yticklabels(
        [f"{names[i]} ($\\geq${2.0 ** bps[i]:g})" for i in range(n_drug)],
        fontsize=6.5)
    ax_band.tick_params(length=0, pad=2.0)
    for s in ax_band.spines.values():
        s.set_visible(False)
    ax_band.set_ylabel("antimicrobial (breakpoint, $\\mu g/mL$)",
                       fontsize=8.0)

    cb = safe_colorbar(fig, im, ax, label="MIC ($\\mu g/mL$), "
                                          "doubling dilutions",
                       width=0.12, pad=0.10, labelsize=6.6,
                       ticks=steps.astype(float))
    cb.ax.set_yticklabels([f"{2.0 ** s:g}" for s in steps], fontsize=6.3)
    return cmap, norm


def rate_panel(ax, names, classes, res):
    n_iso = res.shape[1]
    p, lo, hi = [], [], []
    for row in res:
        a, b, c = wilson(int(row.sum()), n_iso)
        p.append(100.0 * a)
        lo.append(100.0 * (a - b))
        hi.append(100.0 * (c - a))
    p = np.array(p)
    order = np.argsort(p)
    y = np.arange(len(order))
    cols = [CLASS_COLOR[classes[i]] for i in order]
    ax.barh(y, p[order], height=0.66, color=cols, edgecolor="white", lw=0.5,
            zorder=3)
    ax.errorbar(p[order], y, xerr=[np.array(lo)[order], np.array(hi)[order]],
                fmt="none", ecolor=RULE, elinewidth=0.8, capsize=1.8,
                capthick=0.8, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=6.0)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(0.0, 118.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("isolates resistant (%)", fontsize=7.6)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.tick_params(axis="y", length=0)
    despine(ax)
    ax.grid(axis="x", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(0.985, 0.03, f"bars coloured by class\nwhiskers 95 % Wilson, "
                         f"n={n_iso}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.2, color=SUB, linespacing=1.5)
    ax.set_title("Resistance rate by drug", fontsize=8.4, fontweight="bold",
                 color=INK)
    return p, order


def mdr_sets(classes, res):
    """Which classes each isolate is resistant to (any drug in the class)."""
    out = []
    for j in range(res.shape[1]):
        s = {cl for i, cl in enumerate(classes) if res[i, j]}
        out.append(frozenset(s))
    return out


def upset_panels(ax_bar, ax_dot, classes, res, top=7):
    sets = mdr_sets(classes, res)
    uniq = {}
    for s in sets:
        uniq[s] = uniq.get(s, 0) + 1
    combos = sorted(uniq.items(), key=lambda kv: (-kv[1], len(kv[0])))[:top]
    order_cl = ["BL", "TE", "PH", "AG", "MA", "FQ"]
    x = np.arange(len(combos))
    counts = np.array([c for _, c in combos])
    n_cls = np.array([len(s) for s, _ in combos])
    cols = [RED if k >= 3 else BLUE for k in n_cls]
    ax_bar.bar(x, counts, width=0.62, color=cols, edgecolor="white", lw=0.6,
               zorder=3)
    for xi, v in zip(x, counts):
        ax_bar.text(xi, v + 0.12, f"{v}", ha="center", va="bottom",
                    fontsize=6.2, color=INK)
    ax_bar.set_xlim(-0.7, len(combos) - 0.3)
    ax_bar.set_ylim(0.0, float(counts.max()) * 1.32)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("isolates (n)", fontsize=7.6)
    ax_bar.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True,
                                                          nbins=4))
    ax_bar.tick_params(axis="y", labelsize=6.8)
    despine(ax_bar)
    ax_bar.grid(axis="y", lw=0.45, alpha=0.55)
    ax_bar.set_axisbelow(True)
    mdr = int(sum(c for s, c in uniq.items() if len(s) >= 3))
    ax_bar.text(0.985, 0.95, f"red = MDR ($\\geq$3 classes): {mdr} isolates",
                transform=ax_bar.transAxes, ha="right", va="top",
                fontsize=6.3, color=RED)
    ax_bar.set_title("Resistance phenotypes co-occur by class", fontsize=8.4,
                     fontweight="bold", color=INK)

    for r, cl in enumerate(order_cl):
        ax_dot.axhline(r, color=BEIGE, lw=5.0, zorder=1)
    for xi, (s, _) in enumerate(combos):
        rows = [r for r, cl in enumerate(order_cl) if cl in s]
        if len(rows) > 1:
            ax_dot.plot([xi, xi], [min(rows), max(rows)], color=RULE, lw=1.0,
                        zorder=3)
        for r, cl in enumerate(order_cl):
            on = cl in s
            ax_dot.plot([xi], [r], marker="o", ms=4.0,
                        mfc=CLASS_COLOR[cl] if on else "white",
                        mec=RULE if on else HAIR, mew=0.7, zorder=4)
    ax_dot.set_xlim(-0.7, len(combos) - 0.3)
    ax_dot.set_ylim(len(order_cl) - 0.5, -0.5)
    ax_dot.set_xticks([])
    ax_dot.set_yticks(np.arange(len(order_cl)))
    ax_dot.set_yticklabels(order_cl, fontsize=6.2)
    ax_dot.tick_params(length=0)
    ax_dot.set_xlabel("resistance phenotype, most frequent first",
                      fontsize=7.6)
    for s in ax_dot.spines.values():
        s.set_visible(False)
    return counts, mdr


def main():
    set_pub_style()
    rng = np.random.default_rng(20843)
    m, bps, classes, names, _ = mic_matrix(rng)
    r_ord, c_ord = reorder(m)
    m = m[np.ix_(r_ord, c_ord)]
    bps = bps[r_ord]
    classes = classes[r_ord]
    names = [names[i] for i in r_ord]
    res = m >= bps[:, None]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_band = axes_inches(fig, (1.52, 0.95, 0.22, 4.16))
    ax_heat = axes_inches(fig, (1.78, 0.95, 4.02, 4.16))
    ax_rate = axes_inches(fig, (7.62, 3.44, 2.36, 1.96))
    ax_ubar = axes_inches(fig, (7.62, 1.76, 2.36, 1.10))
    ax_udot = axes_inches(fig, (7.62, 0.95, 2.36, 0.76))

    heat_panel(fig, ax_heat, ax_band, m, bps, classes, names, res)
    rates, _ = rate_panel(ax_rate, names, classes, res)
    counts, mdr = upset_panels(ax_ubar, ax_udot, classes, res)

    n_iso = m.shape[1]
    band = ", ".join(f"{k} {v}" for k, v in CLASS_NAME.items())
    panel_labels_inches(fig, [ax_band, ax_rate, ax_ubar], dx=-1.02, dy=0.14)
    savefig(fig, "843_vet_amr_resistance_panel.png")


if __name__ == "__main__":
    main()
