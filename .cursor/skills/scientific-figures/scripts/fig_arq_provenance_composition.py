"""Obsidian provenance: four sources, one element space, fourteen unknowns.

The main panel is the working diagram of a sourcing study.  Two element
ratios that survive weathering - Ba/Zr and Rb/Sr - are plotted against each
other, each geological source occupies its own ellipse, and the artefacts of
unknown origin are dropped into the same space as open stars.  Reading a
star's position against the ellipses is exactly how the assignment is made by
eye, so the panel shows the evidence rather than the conclusion.

The number attached to that conclusion is measured, not asserted.  A linear
discriminant analysis is fitted to all eight measured elements - not to the
two ratios on display - and its accuracy is the mean of a five-fold
stratified cross-validation, so every specimen is classified by a model that
never saw it.  The supporting panels open the same model from two other
sides: the discriminant scores show how much of the between-source variance
each axis carries, and the standardised loadings show which elements the two
axes are actually built from.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pubstyle import set_pub_style, savefig, CATEGORICAL, add_footer
from figkit import (axes_inches, halo_text, panel_labels_inches,
                    unclip_legend)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"
ACCENT = "#b23a48"

FW, FH = 10.4, 6.4          # DESIGN_SYSTEM section 1.1, "wide" tier

# Eight elements from a portable XRF run: six trace, two major.
ELEMENTS = ["Rb", "Sr", "Y", "Zr", "Nb", "Ba", "Fe2O3", "TiO2"]
EL_UNITS = ["ppm", "ppm", "ppm", "ppm", "ppm", "ppm", "wt%", "wt%"]
I_RB, I_SR, I_ZR, I_BA = 0, 1, 3, 5

# Source name, outcrop note, reference specimens, mean composition.
SOURCES = [
    ("Source A", "north rhyolite dome", 44,
     [165.0, 88.0, 34.0, 195.0, 18.0, 640.0, 1.55, 0.16]),
    ("Source B", "valley ignimbrite", 38,
     [132.0, 100.0, 27.0, 148.0, 13.0, 810.0, 2.05, 0.22]),
    ("Source C", "east coulee", 31,
     [205.0, 52.0, 41.0, 235.0, 24.0, 700.0, 1.15, 0.11]),
    ("Source D", "coastal flow", 25,
     [118.0, 190.0, 22.0, 120.0, 10.0, 950.0, 2.60, 0.28]),
]
# Within-source spread, as a standard deviation of log10 concentration.
SD = np.array([0.055, 0.075, 0.050, 0.048, 0.062, 0.052, 0.058, 0.066])
# True origin of the fourteen artefacts, in the order they are drawn.
UNKNOWN_TRUE = [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 3]


def draw_group(rng, mean_log, n, spread=1.0):
    """Log-normal specimens with a source-specific correlation structure.

    A single latent factor (crystal fractionation, in the geochemical story)
    is given a different loading vector per source, so the groups come out as
    ellipses with genuinely different elongations and orientations rather
    than as four copies of one blob.
    """
    load = rng.normal(0.0, 1.0, mean_log.size)
    f = rng.normal(0.0, 1.0, n)
    e = rng.normal(0.0, 1.0, (n, mean_log.size))
    return (mean_log + spread * (f[:, None] * load * SD * 0.85
                                 + e * SD * 0.80))


def cov_ellipse(ax, x, y, color, chi2=5.991, zorder=2):
    """95 % concentration ellipse of a bivariate sample (2 df chi-square)."""
    cxy = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cxy)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * np.sqrt(chi2 * np.clip(vals, 1e-12, None))
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang,
                         facecolor=color, alpha=0.11, edgecolor=color,
                         lw=0.9, ls=(0, (4.0, 2.2)), zorder=zorder))
    return w, h


def main():
    set_pub_style()
    rng = np.random.default_rng(20482)

    names = [s[0] for s in SOURCES]
    notes = [s[1] for s in SOURCES]
    counts = [s[2] for s in SOURCES]
    means = np.log10(np.array([s[3] for s in SOURCES], dtype=float))
    col = CATEGORICAL[:4]

    X = np.vstack([draw_group(rng, means[k], counts[k])
                   for k in range(len(SOURCES))])
    y = np.concatenate([np.full(counts[k], k) for k in range(len(SOURCES))])

    # Artefacts: same sources, but weathered surfaces scatter a little more.
    U = np.vstack([draw_group(rng, means[k], 1, spread=1.05)
                   for k in UNKNOWN_TRUE])
    u_true = np.array(UNKNOWN_TRUE)

    # ---- the measured number ---------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    scores = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=cv)
    acc, acc_sd = float(scores.mean()), float(scores.std())

    lda = LinearDiscriminantAnalysis(store_covariance=True).fit(X, y)
    u_pred = lda.predict(U)
    u_post = lda.predict_proba(U).max(axis=1)
    hit = int(np.sum(u_pred == u_true))

    Z = lda.transform(X)
    ZU = lda.transform(U)
    evr = getattr(lda, "explained_variance_ratio_", None)
    if evr is None:
        evr = np.array([np.nan, np.nan])

    # Standardised loadings: raw scalings carry the units of each element, so
    # they are rescaled by the pooled within-source standard deviation before
    # the two axes are compared.
    w_sd = np.sqrt(np.diag(lda.covariance_))
    load = lda.scalings_[:, :2] * w_sd[:, None]
    load = load / np.abs(load).max(axis=0)

    # Display space: two ratios that are robust to surface alteration.
    def ratios(A):
        return A[:, I_BA] - A[:, I_ZR], A[:, I_RB] - A[:, I_SR]

    rx, ry = ratios(X)
    ux, uy = ratios(U)

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, (0.82, 1.55, 5.30, 4.05))
    ax_d = axes_inches(fig, (6.98, 3.55, 2.90, 2.05))
    ax_l = axes_inches(fig, (6.98, 1.55, 2.90, 1.42))

    # ---- (a) the element-ratio space --------------------------------------
    # Group letters are pushed radially outward from the centre of the whole
    # cloud, so a label can never land on top of a neighbouring ellipse.
    cent = np.array([[rx[y == k].mean(), ry[y == k].mean()]
                     for k in range(len(SOURCES))])
    gmid = cent.mean(axis=0)
    span = np.array([np.ptp(rx), np.ptp(ry)])
    for k in range(len(SOURCES)):
        m = y == k
        cov_ellipse(ax_m, rx[m], ry[m], col[k])
        ax_m.scatter(rx[m], ry[m], s=15.0, facecolor=col[k],
                     edgecolor="white", linewidths=0.45, alpha=0.92,
                     zorder=4)
        hx = np.sqrt(5.991 * np.var(rx[m], ddof=1))
        hy = np.sqrt(5.991 * np.var(ry[m], ddof=1))
        d = (cent[k] - gmid) / span
        d = d / max(float(np.hypot(*d)), 1e-9)
        halo_text(ax_m, cent[k, 0] + d[0] * (hx + 0.035 * span[0]),
                  cent[k, 1] + d[1] * (hy + 0.035 * span[1]),
                  names[k].split()[-1], ha="center", va="center",
                  fontsize=8.6, fontweight="bold", color=col[k],
                  halo="white", halo_lw=2.6, zorder=7)
    for j in range(U.shape[0]):
        ax_m.scatter(ux[j], uy[j], s=52.0, marker="*",
                     facecolor="white", edgecolor=col[u_pred[j]],
                     linewidths=1.0, zorder=6)
    ax_m.set_xlabel("log$_{10}$ ( Ba / Zr ),  ppm / ppm", fontsize=8.0)
    ax_m.set_ylabel("log$_{10}$ ( Rb / Sr ),  ppm / ppm", fontsize=8.0)
    ax_m.tick_params(labelsize=7.0, length=2.6)
    ax_m.grid(lw=0.5, color=HAIR, alpha=0.7)
    ax_m.set_axisbelow(True)
    for s in ("top", "right"):
        ax_m.spines[s].set_visible(False)
    ax_m.set_title(f"Reference collection, n = {X.shape[0]} specimens; "
                   f"{U.shape[0]} artefacts of unknown origin",
                   fontsize=9.0, fontweight="bold", loc="left")
    ax_m.text(0.985, 0.975,
              f"LDA on {len(ELEMENTS)} elements\n"
              f"5-fold CV accuracy {acc:.1%} $\\pm$ {acc_sd:.1%}\n"
              f"{hit} of {U.shape[0]} artefacts recovered, "
              f"mean posterior {u_post.mean():.2f}",
              transform=ax_m.transAxes, ha="right", va="top", fontsize=6.8,
              color=INK, linespacing=1.45, zorder=8,
              bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                        edgecolor=HAIR, linewidth=0.6, alpha=0.93))
    leg = ax_m.legend(handles=[
        Line2D([0], [0], ls="none", marker="o", ms=4.2, color=col[k],
               mec="white", mew=0.5,
               label=f"{names[k]} - {notes[k]} (n = {counts[k]})")
        for k in range(len(SOURCES))] + [
        Line2D([0], [0], ls="none", marker="*", ms=7.0, color="white",
               mec=RULE, mew=0.9, label="artefact, colour = assignment"),
        Line2D([0], [0], color=RULE, lw=0.9, ls=(0, (4.0, 2.2)),
               label="95 % concentration ellipse")],
        loc="lower left", fontsize=6.4, frameon=True, framealpha=0.90,
        edgecolor=HAIR, facecolor="white", handletextpad=0.4,
        borderpad=0.35, labelspacing=0.3)
    unclip_legend(leg)

    # ---- (b) the discriminant space ---------------------------------------
    for k in range(len(SOURCES)):
        m = y == k
        ax_d.scatter(Z[m, 0], Z[m, 1], s=11.0, facecolor=col[k],
                     edgecolor="white", linewidths=0.35, alpha=0.9, zorder=3)
    ax_d.scatter(ZU[:, 0], ZU[:, 1], s=34.0, marker="*", facecolor="white",
                 edgecolor=RULE, linewidths=0.8, zorder=5)
    ax_d.axhline(0.0, color=HAIR, lw=0.6, zorder=1)
    ax_d.axvline(0.0, color=HAIR, lw=0.6, zorder=1)
    ax_d.set_xlabel(f"LD1  ({evr[0]:.0%} of between-source variance)",
                    fontsize=8.0)
    ax_d.set_ylabel(f"LD2  ({evr[1]:.0%})", fontsize=8.0)
    ax_d.tick_params(labelsize=7.0, length=2.6)
    ax_d.set_title("Sources separate on two discriminant axes",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_d.grid(lw=0.5, color=HAIR, alpha=0.7)
    ax_d.set_axisbelow(True)
    for s in ("top", "right"):
        ax_d.spines[s].set_visible(False)

    # ---- (c) what the axes are made of ------------------------------------
    ypos = np.arange(len(ELEMENTS))[::-1]
    ax_l.barh(ypos + 0.19, load[:, 0], height=0.34, color=CATEGORICAL[1],
              edgecolor="white", linewidth=0.4, label="LD1", zorder=3)
    ax_l.barh(ypos - 0.19, load[:, 1], height=0.34, color=CATEGORICAL[3],
              edgecolor="white", linewidth=0.4, label="LD2", zorder=3)
    ax_l.axvline(0.0, color=RULE, lw=0.7, zorder=4)
    ax_l.set_yticks(ypos)
    ax_l.set_yticklabels([f"{e} ({u})" for e, u in zip(ELEMENTS, EL_UNITS)],
                         fontsize=6.2)
    ax_l.set_xlim(-1.18, 1.18)
    ax_l.set_ylim(-0.72, len(ELEMENTS) - 0.28)
    ax_l.set_xlabel("standardised loading (max = 1)", fontsize=8.0)
    ax_l.tick_params(axis="x", labelsize=7.0, length=2.6)
    ax_l.tick_params(axis="y", length=0.0)
    top3 = np.argsort(np.abs(load).max(axis=1))[::-1][:3]
    ax_l.set_title(f"{ELEMENTS[top3[0]]}, {ELEMENTS[top3[1]]} and "
                   f"{ELEMENTS[top3[2]]} carry the discrimination",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_l.grid(axis="x", lw=0.5, color=HAIR, alpha=0.7)
    ax_l.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax_l.spines[s].set_visible(False)
    ax_l.legend(loc="lower right", fontsize=6.6, frameon=False,
                handlelength=1.1, borderpad=0.1, labelspacing=0.25,
                ncol=2, columnspacing=0.9, handletextpad=0.4)

    panel_labels_inches(fig, [ax_m, ax_d, ax_l], dx=-0.36, dy=0.16)


    # Footnote vs. axis labels: measured, not eyeballed.
    fig.canvas.draw()
    note = add_footer(fig, "Synthetic data, seed=20482")
    nb = note.get_window_extent()
    clash = [lab.get_text()
             for ax in (ax_m, ax_d, ax_l)
             for lab in (ax.xaxis.label, ax.yaxis.label)
             if lab.get_text() and nb.overlaps(lab.get_window_extent())]
    print("footnote overlap:", clash if clash else "none")

    print(f"LDA 5-fold CV accuracy {acc:.4f} +- {acc_sd:.4f} "
          f"(folds {np.round(scores, 3).tolist()})")
    print(f"explained variance ratio {np.round(evr[:3], 4).tolist()}")
    print(f"unknowns recovered {hit}/{U.shape[0]}, "
          f"mean posterior {u_post.mean():.3f}, min {u_post.min():.3f}")
    savefig(fig, "892_arq_provenance_composition.png")


if __name__ == "__main__":
    main()
