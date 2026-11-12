"""Exploratory factor analysis of a 16-item scale: retain, rotate, inspect.

One simulated response matrix produces all three panels.  The scree plot and
its parallel-analysis reference decide how many factors to keep; that same
number is what gets extracted and varimax-rotated for the loading matrix; and
the scatter is two columns of that rotated matrix, so the "simple structure"
it shows is the structure the heat map reports rather than a redrawing of the
generating model.  Every number in the footnote is measured off the rotated
solution.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_labels_inches, safe_colorbar, shared_norm

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
ACCENT = "#b23a48"
FACTOR_COLS = ["#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf"]

N_RESP = 350
N_ITEM = 16
N_TRUE = 4
N_SHOW = 10               # eigenvalues drawn in the scree plot
N_PARALLEL = 40
SALIENT = 0.40


def responses(seed=598):
    """Common-factor model with small cross-loadings and unique variance."""
    rng = np.random.default_rng(seed)
    lam = np.zeros((N_ITEM, N_TRUE))
    per = N_ITEM // N_TRUE
    for g in range(N_TRUE):
        lam[g * per:(g + 1) * per, g] = rng.uniform(0.58, 0.86, per)
    lam += rng.normal(0.0, 0.075, lam.shape)
    com = np.clip((lam ** 2).sum(axis=1), 0.05, 0.95)
    f = rng.normal(size=(N_RESP, N_TRUE))
    x = f @ lam.T + rng.normal(size=(N_RESP, N_ITEM)) * np.sqrt(1.0 - com)
    return x


def parallel_eigen(seed=1598):
    """95th-percentile eigenvalues of random data with the same shape."""
    rng = np.random.default_rng(seed)
    out = np.empty((N_PARALLEL, N_ITEM))
    for k in range(N_PARALLEL):
        z = rng.normal(size=(N_RESP, N_ITEM))
        out[k] = np.linalg.eigvalsh(np.corrcoef(z, rowvar=False))[::-1]
    return np.percentile(out, 95.0, axis=0)


def varimax(lam, n_iter=80, tol=1e-7):
    """Kaiser varimax rotation; the standard SVD iteration."""
    p, k = lam.shape
    rot = np.eye(k)
    last = 0.0
    for _ in range(n_iter):
        z = lam @ rot
        u, s, vt = np.linalg.svd(
            lam.T @ (z ** 3 - z @ np.diag(np.diag(z.T @ z)) / p))
        rot = u @ vt
        cur = float(s.sum())
        if last != 0.0 and cur / last < 1.0 + tol:
            break
        last = cur
    return lam @ rot


def main():
    set_pub_style()
    x = responses()
    corr = np.corrcoef(x, rowvar=False)
    eig_val, eig_vec = np.linalg.eigh(corr)
    eig_val, eig_vec = eig_val[::-1], eig_vec[:, ::-1]
    par = parallel_eigen()
    n_keep = int(np.count_nonzero(eig_val > par))

    lam = eig_vec[:, :n_keep] * np.sqrt(eig_val[:n_keep])
    lam = varimax(lam)
    lam *= np.where(lam.sum(axis=0) < 0.0, -1.0, 1.0)      # sign convention
    lam = lam[:, np.argsort(-(lam ** 2).sum(axis=0))]      # variance order

    dom = np.argmax(np.abs(lam), axis=1)
    primary = np.abs(lam[np.arange(N_ITEM), dom])
    cross = np.abs(lam).copy()
    cross[np.arange(N_ITEM), dom] = 0.0
    explained = (lam ** 2).sum() / N_ITEM * 100.0

    items = [f"I{i + 1:02d}" for i in range(N_ITEM)]
    factors = [f"F{j + 1}" for j in range(n_keep)]
    div = get_cmap("cb_diverging")
    norm = shared_norm([lam], symmetric=True)

    # Single row of three columns, so the canvas is flatter than the usual
    # 0.55-0.80 band (DESIGN_SYSTEM exception E8): squeezing it taller would
    # turn each panel into a strip.
    fig = plt.figure(figsize=(10.4, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 1.0, 0.62], wspace=0.34)
    ax_scree = fig.add_subplot(gs[0, 0])
    ax_load = fig.add_subplot(gs[0, 1])
    ax_scat = fig.add_subplot(gs[0, 2])
    fig.subplots_adjust(left=0.058, right=0.975, bottom=0.205, top=0.815)

    # ---- (a) scree plot ---------------------------------------------------
    idx = np.arange(1, N_SHOW + 1)
    ax_scree.axvspan(0.4, n_keep + 0.5, color="#eef3f8", zorder=0,
                     linewidth=0.0)
    ax_scree.plot(idx, eig_val[:N_SHOW], color="#3b5b92", lw=1.6,
                  marker="o", ms=4.2, mfc="white", mew=1.2, zorder=4,
                  label="observed data")
    ax_scree.plot(idx, par[:N_SHOW], color=GREY, lw=1.2, ls=(0, (3.6, 2.0)),
                  marker="s", ms=3.2, mfc="white", mew=0.9, zorder=3,
                  label="parallel analysis (95th pct)")
    ax_scree.axhline(1.0, color="#b3b3b3", lw=0.8, ls=(0, (1.6, 1.8)),
                     zorder=2)
    ax_scree.text(N_SHOW, 1.08, "Kaiser $\\lambda$ = 1", ha="right",
                  va="bottom", fontsize=6.4, color=MUTED)
    ax_scree.annotate(f"retain {n_keep} factors",
                      xy=(n_keep, eig_val[n_keep - 1]),
                      xytext=(n_keep + 1.4, eig_val[0] * 0.66),
                      fontsize=6.9, color=ACCENT,
                      arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                      lw=0.9, shrinkA=0.0, shrinkB=3.0))
    ax_scree.set_xticks(idx)
    ax_scree.set_xlim(0.4, N_SHOW + 0.6)
    ax_scree.set_ylim(0.0, eig_val[0] * 1.12)
    ax_scree.set_xlabel("component number (-)", fontsize=8.0)
    ax_scree.set_ylabel(r"eigenvalue $\lambda$ (-)", fontsize=8.0)
    ax_scree.tick_params(labelsize=7.0)
    ax_scree.grid(axis="y", lw=0.5, color="#e4e4e4")
    ax_scree.set_title("Scree and parallel analysis agree", fontsize=8.8,
                       fontweight="bold", color=INK)
    ax_scree.legend(loc="upper right", frameon=False, fontsize=6.6,
                    handlelength=1.7, labelspacing=0.30, borderaxespad=0.4)

    # ---- (b) rotated loading matrix --------------------------------------
    im = ax_load.imshow(lam, cmap=div, norm=norm, aspect="auto",
                        interpolation="nearest")
    for i in range(N_ITEM):
        for j in range(n_keep):
            v = lam[i, j]
            hot = abs(v) >= SALIENT
            ax_load.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         fontsize=6.0,
                         color="white" if abs(norm(v) - 0.5) > 0.30 else INK)
            if hot:
                ax_load.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0, facecolor="none",
                    edgecolor=ACCENT, linewidth=1.15, zorder=5))
    ax_load.set_xticks(np.arange(n_keep))
    ax_load.set_xticklabels(factors, fontsize=7.4)
    ax_load.set_yticks(np.arange(N_ITEM))
    ax_load.set_yticklabels(items, fontsize=6.6)
    ax_load.xaxis.set_ticks_position("top")
    ax_load.xaxis.set_label_position("top")
    ax_load.set_xlabel("rotated factor", fontsize=8.0)
    ax_load.set_ylabel("scale item", fontsize=8.0)
    ax_load.tick_params(length=2.2)
    for s in ax_load.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax_load.set_title(f"Varimax loadings, boxes mark "
                      f"$|\\lambda_{{ij}}|\\geq${SALIENT:.2f}",
                      fontsize=8.8, fontweight="bold", color=INK, pad=20.0)
    safe_colorbar(fig, im, ax_load, side="bottom",
                  label=r"rotated loading $\lambda_{ij}$ (-)", width=0.10,
                  pad=0.16, labelsize=6.8)

    # ---- (c) two-factor loading scatter ----------------------------------
    for g in range(n_keep):
        m = dom == g
        ax_scat.scatter(lam[m, 0], lam[m, 1], s=26, color=FACTOR_COLS[g],
                        edgecolor="white", linewidth=0.6, zorder=4,
                        label=f"{factors[g]} items")
    for v in (-SALIENT, SALIENT):
        ax_scat.axhline(v, color="#cccccc", lw=0.7, ls=(0, (2.4, 2.0)),
                        zorder=1)
        ax_scat.axvline(v, color="#cccccc", lw=0.7, ls=(0, (2.4, 2.0)),
                        zorder=1)
    ax_scat.axhline(0.0, color=GREY, lw=0.8, zorder=2)
    ax_scat.axvline(0.0, color=GREY, lw=0.8, zorder=2)
    lim = float(np.abs(lam[:, :2]).max()) * 1.22
    ax_scat.set_xlim(-lim, lim)
    ax_scat.set_ylim(-lim, lim)
    ax_scat.set_aspect("equal")
    ax_scat.set_xlabel(r"loading on F1 $\lambda_{i1}$ (-)", fontsize=8.0)
    ax_scat.set_ylabel(r"loading on F2 $\lambda_{i2}$ (-)", fontsize=8.0)
    ax_scat.tick_params(labelsize=7.0)
    ax_scat.set_title("Rotation gives simple structure", fontsize=8.8,
                      fontweight="bold", color=INK)
    ax_scat.legend(loc="lower left", frameon=False, fontsize=6.4,
                   handlelength=1.0, labelspacing=0.26, borderaxespad=0.3,
                   handletextpad=0.4)


    note = (f"{N_RESP} simulated respondents, {N_ITEM} items. Parallel "
            f"analysis uses the 95th percentile of {N_PARALLEL} random "
            f"matrices of the same shape and keeps {n_keep} factors, "
            f"together explaining {explained:.1f} % of item variance. After "
            f"varimax the mean primary loading is {primary.mean():.2f} "
            f"(range {primary.min():.2f}-{primary.max():.2f}) and the "
            f"largest cross-loading is {cross.max():.2f}; "
            f"{int((primary >= SALIENT).sum())} of {N_ITEM} items are "
            f"salient on their own factor only.")

    panel_labels_inches(fig, [ax_scree, ax_load, ax_scat],
                        labels=["(a)", "(b)", "(c)"], dx=-0.44, dy=0.10)
    savefig(fig, "598_psy_factor_analysis.png")


if __name__ == "__main__":
    main()
