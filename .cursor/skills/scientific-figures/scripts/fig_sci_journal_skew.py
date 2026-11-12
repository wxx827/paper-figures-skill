"""Why a journal average says so little: the skew inside one volume.

Main panel: every research paper published in one year by three fictional
journals, laid out on a logarithmic citation axis - a linear axis would pile
nine tenths of each journal on top of the left spine.  A kernel density on
log citations sits above each row, the individual papers are jittered below
it, and the median and the arithmetic mean (the number an impact factor
reports) are marked separately so the gap between them is the message.
Uncited papers cannot be drawn on a log axis, so they are collected as a
labelled block at the left edge instead of being dropped.

Auxiliary (both at 58 % of the main panel width): Lorenz curves of citations
within each journal with the Gini coefficient of each, and the mean-median
pair per journal as bars with the ratio between them.

Every quoted number comes from the sampled arrays: means, medians, the share
of papers below the mean, the uncited share, the top-decile share read off
the cumulative curve, and the Gini computed from the sorted vector.
Fictional journals, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
DEEP = "#2b4470"
RUST = "#c0603f"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (1.02, 0.90, 5.40, 4.58)
AUX_T_RECT = (7.64, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.64, 0.86, 3.14, 2.00)

# name, short name, papers in the volume, lognormal mu and sigma of the rate
JOURNALS = [
    ("Kaleva Reviews", "Kaleva", 210, 2.55, 1.15, DEEP),
    ("J. of Orsan Physics", "Orsan", 340, 1.55, 1.05, lighten(BLUE, 0.32)),
    ("Vantry Letters", "Vantry", 480, 0.95, 1.30, RUST),
]
XLO, XHI = 0.55, 900.0


def volume(rng, n, mu, sigma):
    """Citations received by each paper: a Poisson draw on a lognormal rate."""
    return rng.poisson(rng.lognormal(mu, sigma, n))


def gini(x):
    """Gini coefficient of a non-negative vector, from the sorted values."""
    s = np.sort(np.asarray(x, dtype=float))
    n = s.size
    if s.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * s).sum()) / (n * s.sum()) - (n + 1.0) / n)


def lorenz(x):
    """Cumulative share of papers and of citations, least cited first."""
    s = np.sort(np.asarray(x, dtype=float))
    p = np.arange(1, s.size + 1) / s.size
    c = np.cumsum(s) / s.sum()
    return np.concatenate([[0.0], p]), np.concatenate([[0.0], c])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(991)

    rows = []
    for name, short, n, mu, sigma, col in JOURNALS:
        c = volume(rng, n, mu, sigma)
        pos = c[c > 0]
        rows.append(dict(
            name=name, short=short, n=n, c=c, pos=pos, col=col,
            mean=float(c.mean()), median=float(np.median(c)),
            below=100.0 * float((c < c.mean()).mean()),
            uncited=100.0 * float((c == 0).mean()),
            g=gini(c)))

    flag = rows[0]
    desc = np.sort(flag["c"])[::-1]
    top10 = 100.0 * desc[:max(int(round(0.10 * desc.size)), 1)].sum() / desc.sum()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: one row per journal ------------------------------------------
    ax.set_xscale("log")
    ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.6, zorder=0)
    grid_x = np.logspace(np.log10(0.9), np.log10(XHI), 400)

    for i, r in enumerate(rows):
        base = float(len(rows) - 1 - i)     # first journal on the top row
        kde = gaussian_kde(np.log10(r["pos"]), bw_method=0.32)
        dens = kde(np.log10(grid_x))
        dens = 0.62 * dens / dens.max()
        ax.fill_between(grid_x, base, base + dens, facecolor=r["col"],
                        alpha=0.28, edgecolor="none", zorder=2)
        ax.plot(grid_x, base + dens, color=r["col"], lw=1.2, zorder=4)

        jit = rng.uniform(-0.30, -0.06, r["pos"].size)
        ax.scatter(r["pos"], base + jit, s=3.4, facecolor=r["col"],
                   edgecolor="none", alpha=0.42, zorder=3)

        n_unc = int((r["c"] == 0).sum())
        if n_unc:
            ax.scatter(np.full(n_unc, XLO * 1.18),
                       base + rng.uniform(-0.30, -0.06, n_unc), s=3.4,
                       color=FAINT, alpha=0.55, zorder=3, marker="|")

        ax.plot([r["median"]], [base - 0.185], marker="o", ms=5.2,
                mfc="white", mec=r["col"], mew=1.5, zorder=8)
        ax.plot([r["mean"]], [base - 0.185], marker="D", ms=5.0,
                mfc="white", mec=ACCENT, mew=1.5, zorder=8)
        ax.plot([r["median"], r["mean"]], [base - 0.185] * 2, lw=1.1,
                color=ACCENT, alpha=0.75, zorder=7)
        ax.text(np.sqrt(r["median"] * r["mean"]), base - 0.30,
                f"median {r['median']:.0f} -> mean {r['mean']:.1f} "
                f"citations/paper",
                ha="center", va="top", fontsize=6.2, color=SUB)
        ax.text(XHI * 0.82, base + 0.30,
                f"n = {r['n']} papers\n{r['below']:.0f} % below the mean\n"
                f"{r['uncited']:.0f} % uncited",
                ha="right", va="center", fontsize=6.3, color=SUB,
                linespacing=1.5)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows][::-1], fontsize=8.0)
    ax.set_ylim(-0.62, len(rows) - 0.18)
    ax.set_xlim(XLO, XHI)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_xlabel("citations received by a single paper (citations, "
                  "log scale)", fontsize=8.5)
    ax.set_title("One publication year of three fictional journals, every "
                 "paper plotted", fontsize=9.0, fontweight="bold", pad=5.0)
    ax.text(XLO * 1.30, -0.40, "uncited papers,\nplotted at the left edge",
            ha="left", va="top", fontsize=6.2, color=MUTED, linespacing=1.5)
    hl = [Line2D([], [], marker="o", ls="none", ms=5.2, mfc="white",
                 mec=SUB, mew=1.5, label="median paper"),
          Line2D([], [], marker="D", ls="none", ms=5.0, mfc="white",
                 mec=ACCENT, mew=1.5,
                 label="arithmetic mean = the impact factor")]
    leg = ax.legend(handles=hl, loc="lower right", frameon=False,
                    fontsize=6.6, labelspacing=0.45, handletextpad=0.7,
                    borderpad=0.2, bbox_to_anchor=(1.008, 0.045))
    for t in leg.get_texts():
        t.set_color(SUB)

    # ---- aux (b): Lorenz curves ---------------------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_t.plot([0, 1], [0, 1], color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)),
              zorder=2)
    for r in rows:
        p, c = lorenz(r["c"])
        ax_t.plot(100.0 * p, 100.0 * c, color=r["col"], lw=1.5, zorder=5)
    ax_t.set_xlim(0, 100)
    ax_t.set_ylim(0, 100)
    ax_t.set_xticks([0, 25, 50, 75, 100])
    ax_t.set_yticks([0, 25, 50, 75, 100])
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("share of papers, least cited first (%)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("share of citations (%)", fontsize=7.6)
    ax_t.set_title("A tenth of the papers, most of the citations",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_t.plot([90, 90], [0, 100.0 - top10], color=ACCENT, lw=0.9,
              ls=(0, (2.4, 1.8)), zorder=6)
    ax_t.plot([90], [100.0 - top10], marker="o", ms=3.6, mfc="white",
              mec=ACCENT, mew=1.2, zorder=8)
    ax_t.text(0.035, 0.955,
              f"{rows[0]['name']}: top 10 % of papers\ntake {top10:.0f} % of "
              f"the volume's citations",
              transform=ax_t.transAxes, ha="left", va="top", fontsize=6.1,
              color=ACCENT, linespacing=1.5)
    hg = [Line2D([], [], color=r["col"], lw=1.5,
                 label=f"{r['short']}, Gini {r['g']:.2f}")
          for r in rows]
    lg = ax_t.legend(handles=hg, loc="lower right", frameon=False,
                     fontsize=6.1, handlelength=1.7, labelspacing=0.30,
                     borderpad=0.15, bbox_to_anchor=(1.010, -0.020))
    for t in lg.get_texts():
        t.set_color(SUB)

    # ---- aux (c): mean against median ---------------------------------------
    ax_b.grid(axis="y", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    x = np.arange(len(rows), dtype=float)
    w = 0.34
    ax_b.bar(x - w / 2, [r["median"] for r in rows], width=w,
             facecolor=lighten(BLUE, 0.30), edgecolor=DEEP, linewidth=0.7,
             zorder=4, label="median citations/paper")
    ax_b.bar(x + w / 2, [r["mean"] for r in rows], width=w,
             facecolor=lighten(ACCENT, 0.42), edgecolor=ACCENT, linewidth=0.7,
             zorder=4, label="mean = impact factor")
    for xi, r in zip(x, rows):
        ax_b.text(xi + w / 2, r["mean"] + 0.5,
                  f"x{r['mean'] / max(r['median'], 0.5):.1f}", ha="center",
                  va="bottom", fontsize=6.2, color=ACCENT)
        ax_b.text(xi - w / 2, r["median"] + 0.5, f"{r['median']:.0f}",
                  ha="center", va="bottom", fontsize=6.2, color=DEEP)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([r["short"] for r in rows], fontsize=6.8)
    ax_b.set_ylim(0, max(r["mean"] for r in rows) * 1.32)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_ylabel("citations per paper", fontsize=7.6)
    ax_b.set_title("The gap the single number hides", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    lb = ax_b.legend(loc="upper right", frameon=False, fontsize=6.2,
                     handlelength=1.4, labelspacing=0.32, borderpad=0.15,
                     bbox_to_anchor=(1.012, 1.035))
    for t in lb.get_texts():
        t.set_color(SUB)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.74, dy=0.12)
    savefig(fig, "991_sci_journal_skew.png")


if __name__ == "__main__":
    main()
