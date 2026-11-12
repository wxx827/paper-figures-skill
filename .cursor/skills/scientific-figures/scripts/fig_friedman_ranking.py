"""Friedman test and Nemenyi post-hoc comparison of the five metaheuristics.

Panel (a) ranks the algorithms by their mean Friedman rank over the eight
benchmark functions and draws the critical difference as an error bar.
Panel (b) is the classic Demsar critical-difference diagram: algorithms whose
mean ranks differ by less than CD are joined by a thick clique bar. Panel (c)
gives the full algorithm-by-function rank matrix behind both summaries.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import rankdata, friedmanchisquare
from pubstyle import set_pub_style, savefig, add_panel_label, get_cmap

ALGOS = ["ISOA", "SOA", "PSO", "SSA", "GWO"]
COLORS = {"ISOA": "#b23a48", "SOA": "#3b5b92", "PSO": "#2e7d5b",
          "SSA": "#8e6bbf", "GWO": "#e0812f"}

FKEYS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
FNAMES = {"F1": "Sphere", "F2": "Schwefel 2.22", "F3": "Rosenbrock",
          "F4": "Rastrigin", "F5": "Ackley", "F6": "Griewank",
          "F7": "Schaffer", "F8": "Drop-wave"}

# Mean final error above the optimum, identical to the convergence figures.
ENDS = {
    "F1": {"ISOA": 2.1e-95, "SOA": 5.6e-42, "PSO": 3.4e-9,
           "SSA": 8.8e-24, "GWO": 1.7e-50},
    "F2": {"ISOA": 3.7e-52, "SOA": 1.2e-26, "PSO": 6.1e-5,
           "SSA": 4.4e-14, "GWO": 9.3e-30},
    "F3": {"ISOA": 2.67e+1, "SOA": 2.84e+1, "PSO": 4.13e+1,
           "SSA": 2.91e+1, "GWO": 2.55e+1},
    "F4": {"ISOA": 2.4e-13, "SOA": 7.6e-6, "PSO": 3.9e+1,
           "SSA": 1.1e+1, "GWO": 3.4e+0},
    "F5": {"ISOA": 8.9e-16, "SOA": 4.4e-14, "PSO": 2.3e+0,
           "SSA": 1.9e-7, "GWO": 1.5e-13},
    "F6": {"ISOA": 3.3e-16, "SOA": 2.7e-3, "PSO": 1.9e-2,
           "SSA": 1.4e-8, "GWO": 4.2e-3},
    "F7": {"ISOA": 3.6e-16, "SOA": 9.7e-11, "PSO": 1.2e-16,
           "SSA": 5.2e-12, "GWO": 3.1e-9},
    "F8": {"ISOA": 1.0e-5, "SOA": 2.2e-4, "PSO": 6.4e-2,
           "SSA": 1.45e-2, "GWO": 3.2e-2},
}

# Studentised range statistic / sqrt(2) at alpha = 0.05, indexed by k.
Q_ALPHA = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164}


def rank_matrix():
    """ranks[f, a] with 1 = best on that function."""
    return np.array([rankdata([ENDS[f][a] for a in ALGOS]) for f in FKEYS])


def panel_bars(ax, ranks, cd):
    avg = ranks.mean(axis=0)
    sem = ranks.std(axis=0, ddof=1) / np.sqrt(len(FKEYS))
    order = np.argsort(avg)
    y = np.arange(len(ALGOS))[::-1]

    for yi, j in zip(y, order):
        a = ALGOS[j]
        ax.barh(yi, avg[j], height=0.60, color=COLORS[a],
                alpha=0.85 if a == "ISOA" else 0.62,
                edgecolor=COLORS[a], linewidth=1.6 if a == "ISOA" else 0.8,
                zorder=3)
        ax.errorbar(avg[j], yi, xerr=sem[j], fmt="none", ecolor="#333333",
                    elinewidth=0.9, capsize=2.6, capthick=0.9, zorder=5)
        ax.text(avg[j] + sem[j] + 0.16, yi, f"{avg[j]:.3f}", va="center",
                ha="left", fontsize=7.6, color="#1a1a1a",
                fontweight="bold" if a == "ISOA" else "normal", zorder=6)

    best = avg[order[0]]
    ax.axvline(best, color="#666666", lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.axvspan(best, best + cd, color="#bbbbbb", alpha=0.20, lw=0, zorder=1)
    ax.text(best + cd / 2.0, -0.30,
            f"critical difference  CD = {cd:.3f}", ha="center", va="center",
            fontsize=7.0, color="#444444",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#d5d5d5", lw=0.5, alpha=0.92))

    ax.set_yticks(y, [ALGOS[j] for j in order], fontsize=8.4)
    for tick, j in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[ALGOS[j]])
        if ALGOS[j] == "ISOA":
            tick.set_fontweight("bold")
    ax.set_xlim(0, len(ALGOS) + 0.45)
    ax.set_ylim(-0.72, len(ALGOS) - 0.42)
    ax.set_xlabel("Mean Friedman rank over the eight benchmarks "
                  "(1 = best)", fontsize=8.2, labelpad=2)
    ax.tick_params(axis="x", labelsize=7.4, length=2.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", lw=0.45, color="#ececec", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Mean rank with standard error", fontsize=9,
                 fontweight="bold", loc="left", pad=4)


def cliques(avg_sorted, cd):
    """Maximal groups of adjacent algorithms within CD of each other."""
    n = len(avg_sorted)
    groups = []
    for i in range(n):
        j = i
        while j + 1 < n and avg_sorted[j + 1] - avg_sorted[i] < cd:
            j += 1
        if j > i:
            groups.append((i, j))
    return [g for g in groups
            if not any(g != h and h[0] <= g[0] and g[1] <= h[1]
                       for h in groups)]


def panel_cd(ax, ranks, cd, stat, pval):
    avg = ranks.mean(axis=0)
    order = np.argsort(avg)
    names = [ALGOS[j] for j in order]
    vals = avg[order]

    lo, hi = 1.0, float(len(ALGOS))
    pad = 0.45
    ax.set_xlim(lo - pad - 0.62, hi + pad + 0.62)
    ax.set_ylim(-0.92, 0.78)
    ax.axis("off")

    # ---- rank axis --------------------------------------------------------
    ax.plot([lo, hi], [0, 0], color="#333333", lw=1.1, zorder=4)
    for r in np.arange(lo, hi + 0.01, 0.5):
        big = abs(r - round(r)) < 1e-9
        ax.plot([r, r], [0, 0.055 if big else 0.032], color="#333333",
                lw=1.0 if big else 0.7, zorder=4)
        if big:
            ax.text(r, 0.085, f"{int(r)}", ha="center", va="bottom",
                    fontsize=7.6, color="#333333")

    # ---- critical-difference ruler ---------------------------------------
    ax.plot([lo, lo + cd], [0.38, 0.38], color="#b02020", lw=1.5, zorder=5)
    for x in (lo, lo + cd):
        ax.plot([x, x], [0.34, 0.42], color="#b02020", lw=1.5, zorder=5)
    ax.text(lo + cd / 2.0, 0.455, f"CD = {cd:.3f}", ha="center", va="bottom",
            fontsize=7.8, color="#b02020", fontweight="bold")

    # ---- branches ---------------------------------------------------------
    n = len(names)
    left = list(range(0, (n + 1) // 2))
    for i, (nm, v) in enumerate(zip(names, vals)):
        is_left = i in left
        level = -0.24 - 0.155 * (i if is_left else (n - 1 - i))
        edge = lo - pad if is_left else hi + pad
        ax.plot([v, v], [0, level], color=COLORS[nm], lw=1.3, zorder=4)
        ax.plot([v, edge], [level, level], color=COLORS[nm], lw=1.3, zorder=4)
        ax.text(edge + (-0.06 if is_left else 0.06), level,
                f"{nm}  ({v:.2f})", ha="right" if is_left else "left",
                va="center", fontsize=8.0, color=COLORS[nm],
                fontweight="bold" if nm == "ISOA" else "normal")

    # ---- clique bars ------------------------------------------------------
    for gi, (i, j) in enumerate(cliques(vals, cd)):
        y = -0.075 - 0.072 * gi
        ax.plot([vals[i] - 0.045, vals[j] + 0.045], [y, y], color="#222222",
                lw=3.6, solid_capstyle="round", zorder=6)

    ax.text((lo + hi) / 2.0, -0.86, "algorithms joined by a bar are not "
            "significantly different at $\\alpha$ = 0.05", ha="center",
            va="center", fontsize=7.2, color="#5a5a5a", style="italic")
    ax.set_title(f"Nemenyi post-hoc critical-difference diagram      "
                 f"Friedman $\\chi^2$ = {stat:.3f},  $p$ = {pval:.2e}",
                 fontsize=9, fontweight="bold", loc="left", pad=6)


def panel_heat(ax, ranks):
    avg = ranks.mean(axis=0)
    order = np.argsort(avg)
    mat = np.column_stack([ranks[:, order].T, avg[order]])
    im = ax.imshow(mat[:, :-1], cmap=get_cmap("cb_diverging"), vmin=1, vmax=len(ALGOS),
                   aspect="auto")

    for r in range(len(ALGOS)):
        for c in range(len(FKEYS)):
            v = mat[r, c]
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=8.2,
                    color="white" if v >= 4 or v <= 1 else "#1a1a1a",
                    fontweight="bold" if v == 1 else "normal")
        ax.text(len(FKEYS) + 0.42, r, f"{mat[r, -1]:.3f}", ha="center",
                va="center", fontsize=8.2, color="#1a1a1a",
                fontweight="bold" if order[r] == 0 else "normal")
        ax.text(len(FKEYS) + 1.55, r, f"#{r + 1}", ha="center", va="center",
                fontsize=8.2, color=COLORS[ALGOS[order[r]]],
                fontweight="bold")

    ax.set_xticks(range(len(FKEYS)),
                  [f"${k[0]}_{k[1]}$\n{FNAMES[k]}" for k in FKEYS],
                  fontsize=7.2)
    ax.set_yticks(range(len(ALGOS)), [ALGOS[j] for j in order], fontsize=8.4)
    for tick, j in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[ALGOS[j]])
        if ALGOS[j] == "ISOA":
            tick.set_fontweight("bold")
    ax.set_xlim(-0.5, len(FKEYS) + 2.05)
    ax.text(len(FKEYS) + 0.42, -0.72, "mean\nrank", ha="center", va="center",
            fontsize=7.4, color="#333333", fontweight="bold")
    ax.text(len(FKEYS) + 1.55, -0.72, "overall", ha="center", va="center",
            fontsize=7.4, color="#333333", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Rank of every algorithm on every benchmark function",
                 fontsize=9, fontweight="bold", loc="left", pad=22)
    return im


def main():
    set_pub_style()
    ranks = rank_matrix()
    k, n = len(ALGOS), len(FKEYS)
    cd = Q_ALPHA[k] * np.sqrt(k * (k + 1) / (6.0 * n))
    stat, pval = friedmanchisquare(
        *[[ENDS[f][a] for f in FKEYS] for a in ALGOS])

    fig = plt.figure(figsize=(12.6, 8.8))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.42],
                  height_ratios=[1.0, 0.92], left=0.072, right=0.972,
                  top=0.878, bottom=0.128, hspace=0.46, wspace=0.20)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, :])

    panel_bars(axa, ranks, cd)
    panel_cd(axb, ranks, cd, stat, pval)
    im = panel_heat(axc, ranks)
    add_panel_label(axa, "(a)", x=-0.145, y=1.03, size=11)
    add_panel_label(axb, "(b)", x=-0.045, y=1.03, size=11)
    add_panel_label(axc, "(c)", x=-0.062, y=1.06, size=11)

    cax = fig.add_axes([0.742, 0.048, 0.185, 0.013])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      ticks=range(1, len(ALGOS) + 1))
    cb.set_label("rank on a single function (1 = best)", fontsize=7.0,
                 labelpad=1.5)
    cb.ax.tick_params(labelsize=6.6, length=2, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    savefig(fig, "150_friedman_ranking.png")


if __name__ == "__main__":
    main()
