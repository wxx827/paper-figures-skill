"""Corrplot-style correlation matrix + collinearity screening.

Three panels:
  (a) full correlation matrix of 19 indicators drawn as area-scaled circles
  (b) number of strongly correlated partners (|r| >= 0.7) per indicator,
      drawn as area-scaled circles on the same diverging map as (a)/(c)
  (c) the low-collinearity subset kept by a greedy screen, annotated with r
      values and significance stars

The correlation matrix comes from a simulated block-structured data set, so
it is a genuine (positive semi-definite) correlation matrix rather than a
random symmetric array.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, PALETTES, savefig, add_panel_label

CMAP = LinearSegmentedColormap.from_list("rdbu", PALETTES["div_rdbu"][::-1])
THRESH = 0.7
N_OBS = 180

# four thematic blocks -> realistic within-block collinearity
BLOCKS = [
    (["MAT", "MAP", "PET", "AI", "FRS"], 0.85),
    (["NDVI", "LAI", "GPP", "NPP", "FVC"], 0.90),
    (["SOC", "TN", "pH", "CLAY", "BD"], 0.75),
    (["POP", "GDP", "NTL", "ROAD"], 0.88),
]
LABELS = [v for blk, _ in BLOCKS for v in blk]


def simulate_corr(rng):
    cols = []
    for members, load in BLOCKS:
        latent = rng.normal(0, 1, N_OBS)
        for k, _ in enumerate(members):
            sign = -1.0 if k == len(members) - 1 else 1.0
            cols.append(sign * load * latent
                        + np.sqrt(1 - load ** 2) * rng.normal(0, 1, N_OBS))
    cross = rng.normal(0, 1, N_OBS)
    data = np.column_stack(cols) + 0.30 * cross[:, None]
    return np.corrcoef(data, rowvar=False)


def pvalues(C, n=N_OBS):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(C) * np.sqrt((n - 2) / np.clip(1 - C ** 2, 1e-12, None))
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(t, n - 2)
    except Exception:                                    # pragma: no cover
        p = np.where(t > 3, 0.0005, 0.2)
    return np.clip(p, 0, 1)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def draw_matrix(ax, C, P, labels, numbers=False):
    n = len(labels)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.invert_yaxis()
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#ededed", lw=0.4)
    ax.grid(which="major", visible=False)
    fs = 6.5 if n < 12 else 5.8
    ax.set_xticklabels(labels, fontsize=fs, rotation=90)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(n):
        for j in range(n):
            r = C[i, j]
            if i == j:
                ax.add_patch(plt.Circle((j, i), 0.30, facecolor="#f0f0f0",
                                        edgecolor="#bdbdbd", lw=0.4))
                continue
            if numbers and j > i:
                ax.text(j, i, f"{r:.2f}", ha="center", va="center",
                        fontsize=6.0, color=CMAP((r + 1) / 2)
                        if abs(r) > 0.35 else "#666666")
                st = stars(P[i, j])
                if st:
                    ax.text(j, i - 0.32, st, ha="center", va="center",
                            fontsize=6.0, color="#555555")
                continue
            ax.add_patch(plt.Circle((j, i), 0.46 * np.sqrt(abs(r)),
                                    facecolor=CMAP((r + 1) / 2),
                                    edgecolor="white", lw=0.25))


def draw_partner_dots(ax, C, labels):
    """Per-indicator partner count, drawn with the same circle vocabulary as
    the two matrix panels (area-scaled discs on the shared diverging map)."""
    n = len(labels)
    off = C - np.eye(n)
    counts = (np.abs(off) >= THRESH).sum(axis=1)
    strong = np.where(np.abs(off) >= THRESH, off, 0.0)
    mean_sign = strong.sum(axis=1) / np.maximum(counts, 1)
    order = np.argsort(counts)

    ax.set_anchor("N")
    ax.set_xlim(-0.45, counts.max() + 0.95)
    ax.set_ylim(-0.5, n - 0.5)
    for k, i in enumerate(order):
        c = counts[i]
        if c == 0:
            ax.scatter(0, k, s=26, facecolor="#f0f0f0", edgecolor="#bdbdbd",
                       lw=0.4, zorder=3)
            continue
        ax.plot([0, c], [k, k], color="#dcdcdc", lw=0.8, zorder=2)
        ax.scatter(c, k, s=26 + 24 * c,
                   facecolor=CMAP(0.88) if mean_sign[i] > 0 else CMAP(0.12),
                   edgecolor="white", lw=0.5, zorder=3)
        ax.text(c + 0.30, k, str(c), va="center", fontsize=6,
                color="#333333")

    ax.set_yticks(range(n))
    ax.set_yticklabels([labels[i] for i in order], fontsize=6.0)
    ax.set_xticks(range(0, counts.max() + 1))
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", axis="y", color="#ededed", lw=0.4)
    ax.grid(which="major", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Partners with |r| $\\geq$ {THRESH}", fontsize=8)
    ax.tick_params(length=0, pad=1.5, labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc=CMAP(0.88), mec="white",
               mew=0.5, ms=5, label="mostly positive"),
        Line2D([], [], marker="o", ls="none", mfc=CMAP(0.12), mec="white",
               mew=0.5, ms=5, label="mostly negative"),
        Line2D([], [], marker="o", ls="none", mfc="#f0f0f0", mec="#bdbdbd",
               mew=0.4, ms=4, label="none"),
    ], fontsize=6.5, loc="lower right", frameon=False, handletextpad=0.4,
        borderaxespad=0.2)


def greedy_screen(C, labels):
    keep = list(range(len(labels)))
    while True:
        sub = np.abs(C[np.ix_(keep, keep)]) - np.eye(len(keep))
        counts = (sub >= THRESH).sum(axis=1)
        if counts.max() == 0:
            break
        keep.pop(int(np.argmax(counts)))
    return keep


def main():
    set_pub_style()
    rng = np.random.default_rng(5)
    C = simulate_corr(rng)
    P = pvalues(C)
    keep = greedy_screen(C, LABELS)
    sel = [LABELS[i] for i in keep]

    fig = plt.figure(figsize=(11.0, 5.0))
    gs = GridSpec(1, 3, width_ratios=[1.25, 0.58, 1.0], wspace=0.34,
                  left=0.055, right=0.985, top=0.86, bottom=0.20)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    draw_matrix(ax_a, C, P, LABELS)
    draw_partner_dots(ax_b, C, LABELS)
    draw_matrix(ax_c, C[np.ix_(keep, keep)], P[np.ix_(keep, keep)], sel,
                numbers=True)

    ax_a.set_title("All 19 indicators\n(circle area $\\propto$ |r|)",
                   fontsize=8.5, fontweight="bold", pad=6)
    ax_b.set_title("Collinearity screening", fontsize=8.5, fontweight="bold",
                   pad=6)
    ax_c.set_title(f"Retained subset (all |r| < {THRESH})\n"
                   "circles below, r above the diagonal",
                   fontsize=8.5, fontweight="bold", pad=6)
    # equalise the *absolute* label offset across unequal-width columns
    for ax, lab, lx in zip([ax_a, ax_b, ax_c], "abc", [-0.16, -0.35, -0.20]):
        add_panel_label(ax, f"({lab})", x=lx, y=1.02, size=10)

    cax = fig.add_axes([0.36, 0.075, 0.28, 0.018])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap=CMAP),
                      cax=cax, orientation="horizontal",
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson correlation coefficient, r", fontsize=7.5,
                 labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")

    savefig(fig, "05_corrplot.png")


if __name__ == "__main__":
    main()
