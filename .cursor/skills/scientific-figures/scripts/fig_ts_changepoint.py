"""Offline change-point detection by greedy binary segmentation.

(a) the series with its fitted piecewise mean, the detected change points and
their uncertainty bands; (b) the residual cost as segments are added, whose
elbow selects K; (c) the resulting level ladder with 95 % confidence
intervals.  K is chosen in code from the cost curve, then used to draw (a) -
the three panels cannot disagree.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_labels_inches

SIG_C = "#3a3a3a"      # observed series
LVL_C = "#3b5b92"      # house blue: fitted level
CP_C = "#b23a48"       # house red: the change points
GREY = "#808080"
N = 300
K_MAX = 8
TRUE_CP = [64, 132, 198, 254]
TRUE_MU = [18.0, 23.0, 19.5, 24.5, 21.0]
SIGMA = 3.4


def synth(rng):
    y = np.empty(N)
    edges = [0] + TRUE_CP + [N]
    for mu, a, b in zip(TRUE_MU, edges[:-1], edges[1:]):
        y[a:b] = mu
    return y + rng.normal(0.0, SIGMA, N)


def seg_cost(y, a, b):
    s = y[a:b]
    return float(np.sum((s - s.mean()) ** 2)) if b > a else 0.0


def best_split(y, a, b, min_len=12):
    best, best_i = None, None
    for i in range(a + min_len, b - min_len + 1):
        c = seg_cost(y, a, i) + seg_cost(y, i, b)
        if best is None or c < best:
            best, best_i = c, i
    return best_i, best


def binary_segmentation(y, k_max):
    """Greedy splits; returns the cost after each added change point."""
    bounds = [(0, y.size)]
    cps, costs = [], [seg_cost(y, 0, y.size)]
    for _ in range(k_max):
        gains = []
        for a, b in bounds:
            i, c = best_split(y, a, b)
            if i is not None:
                gains.append((seg_cost(y, a, b) - c, i, a, b))
        if not gains:
            break
        gain, i, a, b = max(gains)
        bounds.remove((a, b))
        bounds += [(a, i), (i, b)]
        cps.append(i)
        costs.append(costs[-1] - gain)
    return cps, np.array(costs)


def pick_k(costs):
    """Knee of the cost curve: the point furthest below the end-to-end chord."""
    k = np.arange(costs.size, dtype=float)
    xn = (k - k[0]) / (k[-1] - k[0])
    yn = (costs - costs[-1]) / (costs[0] - costs[-1])
    return int(np.argmin(xn + yn))


def segment_stats(y, cps):
    edges = [0] + sorted(cps) + [y.size]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = y[a:b]
        ci = 1.96 * s.std(ddof=1) / np.sqrt(s.size)
        out.append((a, b, s.mean(), ci))
    return out


def cp_band(y, a, b, sd, chi2_95=3.841):
    """Profile-cost confidence set for one change point inside [a, b)."""
    grid = np.arange(a + 3, b - 2)
    c = np.array([seg_cost(y, a, i) + seg_cost(y, i, b) for i in grid])
    ok = (c - c.min()) <= chi2_95 * sd ** 2
    return float(grid[ok].min()), float(grid[ok].max())


def panel_series(ax, y, segs, cps, bands):
    ax.plot(np.arange(y.size), y, color=SIG_C, lw=0.7, alpha=0.85, zorder=2)
    lo, hi = y.min(), y.max()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.30 * (hi - lo))
    top = ax.get_ylim()[1]
    for a, b, mu, _ in segs:
        ax.plot([a, b - 1], [mu, mu], color=LVL_C, lw=2.4, zorder=5,
                solid_capstyle="butt")
        ax.text((a + b) / 2.0, mu + 0.055 * (hi - lo), f"{mu:.1f}",
                ha="center", va="bottom", fontsize=6.6, color=LVL_C,
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec="none", alpha=0.85))
    for c, (blo, bhi) in zip(cps, bands):
        ax.axvspan(blo, bhi, color=CP_C, alpha=0.16, lw=0, zorder=1)
        ax.axvline(c, color=CP_C, lw=1.0, ls=(0, (2.6, 1.8)), zorder=6)
        ax.text(c + 2.5, top - 0.055 * (top - lo), f"$\\tau$={c}", ha="left",
                va="top", fontsize=6.6, color=CP_C)
    ax.set_xlim(-3, y.size + 2)
    ax.set_xlabel("Sample index (-)")
    ax.set_ylabel("Level (mm)")
    ax.legend(handles=[
        Line2D([], [], color=SIG_C, lw=0.9, label="Observed"),
        Line2D([], [], color=LVL_C, lw=2.2, label="Segment mean"),
        Line2D([], [], color=CP_C, lw=1.0, ls=(0, (2.6, 1.8)),
               label="Change point $\\tau_k$"),
        Patch(facecolor=CP_C, alpha=0.16, edgecolor="none",
              label="95% location band")],
        loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4, fontsize=7,
        handlelength=1.6, columnspacing=1.8, borderaxespad=0.0)


def panel_elbow(ax, costs, k_sel):
    k = np.arange(costs.size)
    ax.plot(k, costs, color=LVL_C, lw=1.3, marker="o", ms=3.2, mfc="white",
            mew=0.85, zorder=3)
    ax.plot([k_sel], [costs[k_sel]], marker="o", ms=6.0, mfc=CP_C,
            mec="white", mew=0.9, zorder=5)
    ax.annotate(f"elbow at $K$ = {k_sel}", xy=(k_sel, costs[k_sel]),
                xytext=(k_sel + 1.15, costs[k_sel] + 0.30 * np.ptp(costs)),
                fontsize=6.8, color=CP_C, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=CP_C,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xlabel("Number of change points $K$ (-)", fontsize=7.5)
    ax.set_ylabel("Residual cost (mm$^2$)", fontsize=7.5)
    ax.set_xticks(k)
    ax.tick_params(labelsize=6.5)
    ax.set_ylim(0, costs.max() * 1.08)


def panel_levels(ax, segs):
    x = np.arange(1, len(segs) + 1)
    mu = np.array([s[2] for s in segs])
    ci = np.array([s[3] for s in segs])
    ax.errorbar(x, mu, yerr=ci, ls="none", marker="o", ms=4.0, color=LVL_C,
                mfc="white", mew=1.0, elinewidth=1.0, capsize=2.5, zorder=4)
    ax.step(np.r_[x - 0.5, x[-1] + 0.5], np.r_[mu, mu[-1]], where="post",
            color=LVL_C, lw=0.8, alpha=0.45, zorder=3)
    for xi, m, c in zip(x, mu, ci):
        ax.text(xi + 0.14, m, f"{m:.1f}", ha="left", va="center",
                fontsize=6.5, color="#555555")
    ax.set_xlim(0.4, len(segs) + 0.75)
    ax.set_xticks(x)
    ax.set_xlabel("Segment (-)", fontsize=7.5)
    ax.set_ylabel("Segment mean (mm)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    clash = any(mu[i] + ci[i] > mu[i + 1] - ci[i + 1] and
                mu[i] - ci[i] < mu[i + 1] + ci[i + 1]
                for i in range(mu.size - 1))
    ax.set_title("Adjacent 95% CIs overlap" if clash
                 else "No two adjacent 95% CIs overlap",
                 fontsize=8.5, fontweight="bold")


def main():
    set_pub_style()
    rng = np.random.default_rng(15)
    y = synth(rng)

    cps_all, costs = binary_segmentation(y, K_MAX)
    k_sel = pick_k(costs)
    cps = sorted(cps_all[:k_sel])
    segs = segment_stats(y, cps)

    # Location uncertainty from the profile cost around each change point.
    sd = np.sqrt(costs[k_sel] / (y.size - k_sel - 1))
    edges = [0] + cps + [y.size]
    bands = [cp_band(y, edges[i], edges[i + 2], sd)
             for i in range(len(cps))]

    fig = plt.figure(figsize=(7.4, 4.8))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.62],
                  height_ratios=[1.0, 0.55], hspace=0.52, wspace=0.26,
                  left=0.082, right=0.982, top=0.855, bottom=0.115)
    ax = fig.add_subplot(gs[0, :])
    bx = fig.add_subplot(gs[1, 0])
    cx = fig.add_subplot(gs[1, 1])

    panel_series(ax, y, segs, cps, bands)
    panel_elbow(bx, costs, k_sel)
    panel_levels(cx, segs)
    for a in (ax, bx, cx):
        a.grid(axis="y", lw=0.5, color="#f0f0f0")
        a.set_axisbelow(True)
        style_axes(a)

    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.62, dy=0.24)
    panel_labels_inches(fig, [bx, cx], labels=["(b)", "(c)"], dx=-0.62,
                        dy=0.10)
    widths = [int(hi - lo) for lo, hi in bands]
    savefig(fig, "359_ts_changepoint.png")


if __name__ == "__main__":
    main()
