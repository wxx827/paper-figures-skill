"""Bubble matrix: correlation grid where colour is r and circle area is |r|.

Two panels built from one simulated dataset: the Pearson matrix and the partial
correlation matrix derived from its precision matrix, so panel (b) really is
panel (a) with the shared variance removed. Cells whose correlation is not
significant at the 5% level are crossed out. Layout is driven by
``subplots_adjust`` plus a manually placed colorbar axis - ``tight_layout``
cannot handle a figure-level colorbar and used to emit a warning here.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label

VARS = ["TEM", "PRE", "NDVI", "ET", "NPP", "POP"]
N_OBS = 180
CMAP = LinearSegmentedColormap.from_list("corr", PALETTES["div_rdbu"][::-1],
                                         N=256)
NORM = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
MAX_AREA = 430.0
#              heat  water  human
LOADINGS = np.array([
    [0.92, -0.25, 0.20],    # TEM
    [-0.35, 0.88, -0.10],   # PRE
    [-0.40, 0.72, -0.35],   # NDVI
    [0.45, 0.68, -0.12],    # ET
    [-0.30, 0.66, -0.30],   # NPP
    [0.25, -0.15, 0.90],    # POP
])


def simulate(rng):
    f = rng.normal(0, 1, (3, N_OBS))
    uniq = np.sqrt(np.clip(1 - (LOADINGS ** 2).sum(1), 0.05, None))
    x = LOADINGS @ f + uniq[:, None] * rng.normal(0, 1, (len(VARS), N_OBS))
    pearson = np.corrcoef(x)
    prec = np.linalg.inv(pearson)
    dg = np.sqrt(np.diag(prec))
    partial = -prec / np.outer(dg, dg)
    np.fill_diagonal(partial, 1.0)
    return pearson, partial


def bubble_matrix(ax, idx, mat, title, df):
    n = len(VARS)
    crit = stats.t.ppf(0.975, df)
    r_crit = crit / np.sqrt(crit ** 2 + df)
    for i in range(n):
        for j in range(n):
            r = mat[i, j]
            ax.scatter(j, i, s=MAX_AREA * abs(r), c=[r], cmap=CMAP, norm=NORM,
                       edgecolors="white", linewidths=0.6, zorder=3)
            if i != j and abs(r) < r_crit:
                ax.plot(j, i, marker="x", ms=3.2, color="#8a8a8a",
                        markeredgewidth=0.8, zorder=4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(VARS, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(VARS, fontsize=8)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#ededed", lw=0.6)
    ax.tick_params(which="minor", length=0)
    add_panel_label(ax, f"({'ab'[idx]})", x=-0.16, y=1.0, size=11)


def main():
    set_pub_style()
    rng = np.random.default_rng(521)
    pearson, partial = simulate(rng)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4))
    bubble_matrix(axes[0], 0, pearson, "Pearson correlation", N_OBS - 2)
    bubble_matrix(axes[1], 1, partial, "Partial correlation",
                  N_OBS - len(VARS))

    fig.subplots_adjust(left=0.07, right=0.86, top=0.88, bottom=0.20,
                        wspace=0.28)
    cax = fig.add_axes([0.885, 0.24, 0.015, 0.56])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Correlation coefficient r", fontsize=8)
    cb.ax.tick_params(labelsize=7, width=0.7, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    handles = [Line2D([], [], marker="o", color="none",
                      markerfacecolor="#bbbbbb", markeredgecolor="white",
                      markersize=np.sqrt(MAX_AREA * v), label=f"|r| = {v:.1f}")
               for v in (0.2, 0.5, 1.0)]
    handles.append(Line2D([], [], marker="x", color="#8a8a8a", ls="none",
                          markersize=4, markeredgewidth=0.9,
                          label="not significant (p > 0.05)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.47, 0.02), handletextpad=0.8,
               columnspacing=2.0)
    fig.text(0.47, 0.115, f"n = {N_OBS} observations; circle area scales with "
             "|r|, colour with its sign", ha="center", fontsize=7.5,
             color="#555555")
    savefig(fig, "53_bubble_matrix.png")


if __name__ == "__main__":
    main()
