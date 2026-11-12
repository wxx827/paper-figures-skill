"""Subjective workload (NASA-TLX) across four control-room task conditions.

Main panel: for every condition a pair of stacked bars, the left one giving
each subscale an equal 1/6 share (raw TLX) and the right one weighting it by
the source weight that the pairwise-comparison procedure returned, so the
re-ranking of the subscales is visible rather than asserted.  The weighted
total is drawn through the pairs as a line.

Auxiliary (both at 58 % of the main panel width): the distribution of
participant-level weighted totals per condition with paired-test brackets;
and the inverted-U between workload and task performance, with a quadratic
fit and its turning point.

Every number is computed from the simulated response table: the weights are
mean wins over the 15 pairwise comparisons each participant makes, the
weighted total is the dot product of weights with raw subscale scores, and
the brackets come from paired t-tests on the plotted samples.  Fictional
study, illustrative values only.
"""
import itertools
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from pubstyle import (set_pub_style, savefig, style_axes, lighten, sig_stars,
                      add_sig_bracket)
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
ACCENT = "#b23a48"
BLUE = "#3b5b92"
TEAL = "#2e7d5b"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.86, 0.86, 5.54, 4.62)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

DIMS = ["mental demand", "physical demand", "temporal demand",
        "performance", "effort", "frustration"]
DIM_COLORS = ["#3b5b92", "#84b6d6", "#2e7d5b", "#c9a227", "#c0603f",
              "#8e6bbf"]
# latent importance driving the pairwise comparisons (arbitrary logit scale)
IMPORTANCE = [1.35, -1.05, 0.95, 0.20, 0.75, -0.35]

CONDITIONS = ["nominal\nmonitoring", "single\nfault", "dual fault\n+ alarms",
              "dual fault\n+ comms load"]
# per-condition mean of each raw subscale, 0-100 scale
RAW_MEANS = np.array([
    [32, 18, 26, 24, 30, 16],
    [55, 27, 52, 38, 54, 34],
    [72, 35, 74, 55, 71, 58],
    [84, 46, 86, 62, 83, 69],
], dtype=float)
N_SUBJ = 18


def tlx_weights(rng):
    """Weights as mean wins over the 15 pairwise comparisons per participant."""
    pairs = list(itertools.combinations(range(len(DIMS)), 2))
    wins = np.zeros((N_SUBJ, len(DIMS)))
    for s in range(N_SUBJ):
        bias = rng.normal(0.0, 0.55, len(DIMS))
        for i, j in pairs:
            d = (IMPORTANCE[i] + bias[i]) - (IMPORTANCE[j] + bias[j])
            if rng.random() < 1.0 / (1.0 + np.exp(-d)):
                wins[s, i] += 1
            else:
                wins[s, j] += 1
    return wins.mean(axis=0) / len(pairs), wins, len(pairs)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(950)

    w, wins, n_pairs = tlx_weights(rng)

    # participant-level raw subscales, clipped to the 0-100 response scale
    raw = np.clip(RAW_MEANS[None, :, :]
                  + rng.normal(0.0, 8.5, (N_SUBJ, len(CONDITIONS), len(DIMS)))
                  + rng.normal(0.0, 6.0, (N_SUBJ, 1, 1)), 0.0, 100.0)
    raw_mean = raw.mean(axis=0)
    equal_share = raw_mean / len(DIMS)
    weighted_share = raw_mean * w[None, :]
    raw_total = equal_share.sum(axis=1)
    wt_total = weighted_share.sum(axis=1)
    subj_total = raw @ w                      # (participants, conditions)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: raw vs weighted stacks --------------------------------------
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    pos = np.arange(len(CONDITIONS), dtype=float)
    off, bw = 0.205, 0.36
    for k in range(len(DIMS)):
        ax.bar(pos - off, equal_share[:, k], width=bw,
               bottom=equal_share[:, :k].sum(axis=1),
               color=lighten(DIM_COLORS[k], 0.42), edgecolor="white",
               linewidth=0.6, zorder=3)
        ax.bar(pos + off, weighted_share[:, k], width=bw,
               bottom=weighted_share[:, :k].sum(axis=1),
               color=DIM_COLORS[k], edgecolor="white", linewidth=0.6,
               zorder=3)
    ax.plot(pos + off, wt_total, color=ACCENT, lw=1.5, marker="o", ms=4.2,
            mfc="white", mec=ACCENT, mew=1.2, zorder=6)
    for k in range(len(CONDITIONS)):
        ax.text(pos[k] - off, raw_total[k] + 1.6, f"{raw_total[k]:.1f}",
                ha="center", va="bottom", fontsize=6.5, color=SUB)
        ax.text(pos[k] + off, wt_total[k] + 4.6, f"{wt_total[k]:.1f}",
                ha="center", va="bottom", fontsize=6.8, color=ACCENT,
                fontweight="bold")
        ax.text(pos[k] - off, -2.0, "raw", ha="center", va="top",
                fontsize=6.2, color=MUTED)
        ax.text(pos[k] + off, -2.0, "weighted", ha="center", va="top",
                fontsize=6.2, color=MUTED)
    ax.set_xticks(pos)
    ax.set_xticklabels(CONDITIONS, fontsize=7.4, linespacing=1.35)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.set_xlim(-0.62, len(CONDITIONS) - 0.38)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("workload contribution (TLX score, 0-100)", fontsize=8.5)
    ax.set_title(f"NASA-TLX subscale composition, n={N_SUBJ} operators",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    handles = [Patch(facecolor=DIM_COLORS[k], edgecolor="white", linewidth=0.6,
                     label=f"{DIMS[k]}  (w = {w[k]:.3f})")
               for k in range(len(DIMS))]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False,
                    fontsize=6.8, handlelength=1.2, handleheight=0.9,
                    labelspacing=0.30, borderpad=0.15, ncol=2,
                    columnspacing=1.1, bbox_to_anchor=(-0.006, 1.004))
    ax.add_artist(leg)
    ax.text(3.42, 88.0,
            f"left bar: equal 1/6 = {1 / len(DIMS):.3f} share per subscale\n"
            f"right bar: source weights from {n_pairs} pairwise\n"
            f"comparisons per participant (weights sum to {w.sum():.2f})\n"
            f"red line: weighted total workload",
            ha="right", va="top", fontsize=6.7, color=SUB, linespacing=1.55)

    # ---- aux (b): distribution of weighted totals --------------------------
    ax_t.grid(axis="y", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    bp = ax_t.boxplot([subj_total[:, k] for k in range(len(CONDITIONS))],
                      positions=pos, widths=0.52, patch_artist=True,
                      showfliers=False, zorder=3,
                      medianprops=dict(color=INK, lw=1.2),
                      whiskerprops=dict(color=FAINT, lw=0.8),
                      capprops=dict(color=FAINT, lw=0.8),
                      boxprops=dict(edgecolor=FAINT, lw=0.8))
    for k, box in enumerate(bp["boxes"]):
        box.set_facecolor(lighten(BLUE, 0.30 + 0.16 * k))
    jx = rng.normal(0.0, 0.055, subj_total.shape)
    for k in range(len(CONDITIONS)):
        ax_t.plot(pos[k] + jx[:, k], subj_total[:, k], ls="none", marker="o",
                  ms=2.0, mfc=INK, mec="none", alpha=0.45, zorder=5)
    for k in range(len(CONDITIONS) - 1):
        t, p = stats.ttest_rel(subj_total[:, k], subj_total[:, k + 1])
        top = max(subj_total[:, k].max(), subj_total[:, k + 1].max())
        add_sig_bracket(ax_t, pos[k], pos[k + 1], top + 3.5 + 5.0 * (k % 2),
                        text=sig_stars(p), color=SUB, lw=0.8, fontsize=7.0)
    ax_t.set_xticks(pos)
    ax_t.set_xticklabels(["nominal", "1 fault", "2 faults", "+ comms"],
                         fontsize=6.5)
    ax_t.set_xlim(-0.62, len(CONDITIONS) - 0.38)
    ax_t.set_ylim(0.0, 108.0)
    ax_t.set_yticks([0, 25, 50, 75, 100])
    ax_t.tick_params(axis="y", labelsize=6.6)
    ax_t.set_ylabel("weighted TLX (score)", fontsize=7.6, labelpad=2.0)
    ax_t.set_title("Weighted totals per participant", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_t.text(-0.58, 3.0, "paired t-tests, adjacent conditions; "
              "*** p < 0.001", ha="left", va="bottom", fontsize=6.1,
              color=MUTED)

    # ---- aux (c): workload against performance -----------------------------
    load = subj_total.ravel()
    perf = (98.0 - 0.0118 * (load - 46.0) ** 2
            + rng.normal(0.0, 4.2, load.size))
    perf = np.clip(perf, 20.0, 100.0)
    co = np.polyfit(load, perf, 2)
    xs = np.linspace(load.min() - 2.0, load.max() + 2.0, 200)
    peak = -co[1] / (2.0 * co[0])
    peak_y = np.polyval(co, peak)
    ax_b.grid(color=GRID, lw=0.5, alpha=0.65, zorder=0)
    for k in range(len(CONDITIONS)):
        sel = slice(k, None, len(CONDITIONS))
        ax_b.plot(load.reshape(N_SUBJ, -1)[:, k],
                  perf.reshape(N_SUBJ, -1)[:, k], ls="none", marker="o",
                  ms=2.8, mfc=lighten(BLUE, 0.55 - 0.15 * k), mec="white",
                  mew=0.3, zorder=4)
    ax_b.plot(xs, np.polyval(co, xs), color=ACCENT, lw=1.6, zorder=6)
    ax_b.axvline(peak, color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=3)
    ax_b.plot([peak], [peak_y], marker="v", ms=5.0, mfc=ACCENT, mec="white",
              mew=0.7, zorder=7)
    ax_b.text(peak + 2.0, 64.0,
              f"optimum at {peak:.0f} score\n({peak_y:.0f} % correct)",
              ha="left", va="bottom", fontsize=6.2, color=ACCENT,
              linespacing=1.4)
    ax_b.set_xlim(0.0, 105.0)
    ax_b.set_ylim(58.0, 110.0)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_yticks([60, 75, 90, 105])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("weighted TLX (score)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("task performance (%)", fontsize=7.6, labelpad=2.0)
    ax_b.set_title("Inverted-U: too little load hurts too", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_b.text(103.0, 60.0, f"quadratic fit, n={load.size} "
              "participant-conditions", ha="right", va="bottom",
              fontsize=6.1, color=MUTED)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.62, dy=0.12)
    savefig(fig, "950_hfe_workload_tlx.png")


if __name__ == "__main__":
    main()
