"""Replication: the same experiments, run again with more subjects.

Main panel: forty fictional findings as slope lines between the effect size
their original study reported and the effect size a larger replication
found.  A line is drawn in the house red when the replication misses
significance in the original direction, in blue when it holds; the two heavy
markers are the medians of each column, so the collapse of the middle of the
distribution is visible without reading a single number.

Auxiliary (both at 58 % of the main panel width): replication effect against
original effect for the full simulated pool with the identity line and a
least-squares shrinkage slope; and the replication success rate split by how
convincing the original p value looked, which is the practical question -
does a p of 0.001 buy more than a p of 0.04.

The mechanism is in the simulation, not in the numbers: more than half of
the findings have no true effect at all, and an original study is only
"published" once it reaches p <= 0.05, so the published effect sizes are
selected upward and the replications regress toward the truth.  Every quoted
statistic is computed from the simulated pairs.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
DEEP = "#2b4470"
RUST = "#c0603f"
GOLD = "#c9a227"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (1.00, 0.90, 5.42, 4.58)
AUX_T_RECT = (7.64, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.64, 0.86, 3.14, 2.00)

N_ORIG, N_REP = 25, 60          # subjects per group
N_PAIRS = 520                   # simulated original / replication pairs
N_SHOWN = 40                    # pairs drawn as slope lines
P_NULL = 0.55                   # share of findings with no true effect
ALPHA = 0.05


def one_study(rng, n, d_true):
    """Observed Cohen's d and two-sided p of one two-sample experiment."""
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(d_true, 1.0, n)
    sp = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    d = (b.mean() - a.mean()) / sp
    t = d * np.sqrt(n / 2.0)
    return d, float(2.0 * stats.t.sf(abs(t), 2 * n - 2))


def published_original(rng, d_true):
    """Keep running the experiment until it is significant and positive."""
    for _ in range(400):
        d, p = one_study(rng, N_ORIG, d_true)
        if p <= ALPHA and d > 0:
            return d, p
    return d, p


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(994)

    d_true = np.where(rng.random(N_PAIRS) < P_NULL, 0.0,
                      np.abs(rng.normal(0.42, 0.18, N_PAIRS)))
    do, po, dr, pr = [], [], [], []
    for dt in d_true:
        d1, p1 = published_original(rng, dt)
        d2, p2 = one_study(rng, N_REP, dt)
        do.append(d1)
        po.append(p1)
        dr.append(d2)
        pr.append(p2)
    do, po, dr, pr = map(np.asarray, (do, po, dr, pr))
    ok = (pr <= ALPHA) & (dr > 0)

    sub = slice(0, N_SHOWN)
    s_ok = ok[sub]
    med_o = float(np.median(do[sub]))
    med_r = float(np.median(dr[sub]))
    shrink_shown = 100.0 * (1.0 - med_r / med_o)
    rate_shown = 100.0 * s_ok.mean()
    rate_all = 100.0 * ok.mean()
    slope = float(np.polyfit(do, dr, 1)[0])
    inter = float(np.polyfit(do, dr, 1)[1])

    edges = [0.0, 0.001, 0.01, 0.05]
    labels = ["p < 0.001", "0.001 - 0.01", "0.01 - 0.05"]
    rates, ns = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (po > lo) & (po <= hi)
        ns.append(int(m.sum()))
        rates.append(100.0 * ok[m].mean() if m.any() else np.nan)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the slope graph ----------------------------------------------
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.55, zorder=0)
    ax.axhline(0.0, color=FAINT, lw=0.9, zorder=2)
    for x in (0.0, 1.0):
        ax.axvline(x, color="#e2e2e2", lw=0.8, zorder=1)

    for i in range(N_SHOWN):
        col = DEEP if ok[i] else ACCENT
        ax.plot([0.0, 1.0], [do[i], dr[i]], color=col, lw=1.0,
                alpha=0.60 if ok[i] else 0.75, zorder=4)
        ax.plot([0.0], [do[i]], marker="o", ms=3.2, mfc=col, mec="white",
                mew=0.5, zorder=6)
        ax.plot([1.0], [dr[i]], marker="o", ms=3.2, mfc=col, mec="white",
                mew=0.5, zorder=6)

    ax.plot([0.0, 1.0], [med_o, med_r], color=GOLD, lw=2.6, zorder=8)
    for x, v in ((0.0, med_o), (1.0, med_r)):
        ax.plot([x], [v], marker="D", ms=7.0, mfc="white", mec=GOLD, mew=2.0,
                zorder=9)
    ax.text(-0.06, med_o, f"median d = {med_o:.2f}", ha="right", va="center",
            fontsize=7.0, color="#8a6d1f", fontweight="bold")
    ax.text(1.06, med_r, f"median d = {med_r:.2f}", ha="left", va="center",
            fontsize=7.0, color="#8a6d1f", fontweight="bold")

    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(-0.72, 1.85)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([f"original study\n(n = {N_ORIG} per group)",
                        f"direct replication\n(n = {N_REP} per group)"],
                       fontsize=8.0)
    ax.set_ylabel("observed effect size (Cohen's d)", fontsize=8.5)
    ax.set_title(f"{N_SHOWN} fictional findings and their replications",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    ax.text(-0.58, 1.78,
            f"{int(s_ok.sum())} of {N_SHOWN} replications are significant in "
            f"the original direction ({rate_shown:.0f} %)\n"
            f"the median published effect shrinks by {shrink_shown:.0f} % "
            f"from d = {med_o:.2f} to d = {med_r:.2f}\n"
            f"{int((dr[sub] < 0).sum())} replications land on the wrong side "
            f"of zero",
            ha="left", va="top", fontsize=6.8, color=SUB, linespacing=1.6)
    hl = [Line2D([], [], color=DEEP, lw=1.4, marker="o", ms=3.4, mfc=DEEP,
                 mec="white", label="replicated (p <= 0.05, same direction)"),
          Line2D([], [], color=ACCENT, lw=1.4, marker="o", ms=3.4,
                 mfc=ACCENT, mec="white", label="failed to replicate"),
          Line2D([], [], color=GOLD, lw=2.6, marker="D", ms=5.0, mfc="white",
                 mec=GOLD, mew=1.6, label="median of each column")]
    leg = ax.legend(handles=hl, loc="lower center", frameon=False,
                    fontsize=6.7, labelspacing=0.45, handlelength=2.2,
                    borderpad=0.2, bbox_to_anchor=(0.50, -0.018))
    for t in leg.get_texts():
        t.set_color(SUB)

    # ---- aux (b): replication against original ------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    lim = (-0.55, 1.75)
    ax_t.plot(lim, lim, color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=3)
    ax_t.axhline(0.0, color="#e2e2e2", lw=0.8, zorder=2)
    ax_t.scatter(do[~ok], dr[~ok], s=6.0, facecolor=lighten(ACCENT, 0.40),
                 edgecolor="none", alpha=0.75, zorder=4)
    ax_t.scatter(do[ok], dr[ok], s=6.0, facecolor=lighten(BLUE, 0.30),
                 edgecolor="none", alpha=0.80, zorder=5)
    xg = np.linspace(lim[0], lim[1], 40)
    ax_t.plot(xg, inter + slope * xg, color=DEEP, lw=1.5, zorder=7)
    ax_t.set_xlim(*lim)
    ax_t.set_ylim(*lim)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("original effect size (Cohen's d)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("replication (Cohen's d)", fontsize=7.6)
    ax_t.set_title("Replications regress toward zero", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.text(0.030, 0.955,
              f"fitted slope {slope:.2f} (identity = 1.00)\n"
              f"{rate_all:.0f} % of {N_PAIRS} pairs replicate\n"
              f"dashed line: equal effect sizes",
              transform=ax_t.transAxes, ha="left", va="top", fontsize=6.1,
              color=SUB, linespacing=1.5)

    # ---- aux (c): success by original p -------------------------------------
    ax_b.grid(axis="y", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    x = np.arange(len(rates), dtype=float)
    cols = [DEEP, lighten(BLUE, 0.25), lighten(ACCENT, 0.20)]
    ax_b.bar(x, rates, width=0.58, facecolor=[lighten(c, 0.45) for c in cols],
             edgecolor=cols, linewidth=0.8, zorder=4)
    for xi, r, n in zip(x, rates, ns):
        ax_b.text(xi, r + 1.6, f"{r:.0f} %", ha="center", va="bottom",
                  fontsize=6.6, color=INK)
        ax_b.text(xi, 5.0, f"n = {n} pairs", ha="center", va="bottom",
                  fontsize=6.1, color=INK, zorder=8)
    ax_b.axhline(rate_all, color=SUB, lw=0.9, ls=(0, (2.6, 2.0)), zorder=6)
    ax_b.text(len(rates) - 0.52, rate_all + 1.5,
              f"pool average {rate_all:.0f} %", ha="right", va="bottom",
              fontsize=6.1, color=SUB)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=6.8)
    ax_b.set_xlim(-0.62, len(rates) - 0.38)
    ax_b.set_ylim(0, 100)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("p value reported by the original study", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel("replications that hold (%)", fontsize=7.6)
    ax_b.set_title("A smaller original p does buy something", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.72, dy=0.12)
    savefig(fig, "994_sci_replication.png")


if __name__ == "__main__":
    main()
