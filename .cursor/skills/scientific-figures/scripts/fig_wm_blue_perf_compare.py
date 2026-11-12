"""Wm blue perf compare: grouped bars for six methods on three benchmarks.

Six world-model agents evaluated on three benchmarks.  The five baselines take
five rungs of the batch blue ladder, ordered light-to-dark, and Ours is the one
warm bar.  Every group carries a dark grey dashed rule at the strongest
baseline of that group, so the reader can see at a glance that only the warm
bar clears it.  The rule is read off the plotted means rather than typed in, so
it can never drift below a bar it is supposed to top.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
ACCENT = "#b5544a"
INK = "#22272b"
SOTA_C = "#4d4d4d"

METHODS = ["PlaNet", "Dreamer-V2", "IRIS", "TD-MPC2", "STORM", "Ours"]
METHOD_COLORS = [BLUE[5], BLUE[4], BLUE[3], BLUE[2], BLUE[1], ACCENT]
DATASETS = ["RLBench-18", "Meta-World MT-50", "Ego-Sim Hard"]
N_RUNS = 6
LABEL_H = 1.0        # height of a 6 pt value label, in y-axis points


def gen_data(seed=2519):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_blue_perf_compare.csv")
        runs = df.pivot_table(index=["method", "dataset"], columns="run",
                              values="success")
        return runs.mean(axis=1), runs.std(axis=1, ddof=1)
    """
    rng = np.random.default_rng(seed)
    # Per-benchmark difficulty offset plus a per-method skill profile.
    base = np.array([76.0, 70.0, 54.0])
    skill = np.array([-9.5, -5.2, -2.4, -0.6, 1.4, 5.6])
    shape = (len(METHODS), len(DATASETS))
    truth = base[None, :] + skill[:, None] + rng.normal(0.0, 1.5, shape)
    spread = rng.uniform(0.8, 2.4, shape)
    # Draw the individual evaluation runs and reduce them here, so the bars
    # and whiskers really are the mean and s.d. of N_RUNS numbers -- which is
    # what the caption tells the reader they are.
    runs = rng.normal(truth[..., None], spread[..., None], shape + (N_RUNS,))
    return runs.mean(axis=-1), runs.std(axis=-1, ddof=1)


def main():
    set_pub_style()
    mean, sd = gen_data()
    n_m, n_d = mean.shape

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    group_w = 0.84
    bar_w = group_w / n_m
    centres = np.arange(n_d, dtype=float)

    sota = mean[:-1].max(axis=0)   # strongest baseline, Ours excluded

    for m in range(n_m):
        off = -group_w / 2 + bar_w * (m + 0.5)
        ax.bar(centres + off, mean[m], width=bar_w * 0.92,
               color=METHOD_COLORS[m], edgecolor="white", linewidth=0.5,
               zorder=3, label=METHODS[m])
        ax.errorbar(centres + off, mean[m], yerr=sd[m], fmt="none",
                    ecolor="#4d4d4d", elinewidth=0.8, capsize=2.0,
                    capthick=0.8, zorder=5)
        for c, mu, s, lvl in zip(centres + off, mean[m], sd[m], sota):
            y = mu + s + 0.7
            # A halo alone is not enough where the rule lands mid-glyph: it
            # still reads as a strike-through, so lift any label whose box
            # (y .. y + LABEL_H, since va="bottom") straddles the rule.
            if y - 0.15 < lvl < y + LABEL_H:
                y = lvl + 0.25
            ax.text(c, y, f"{mu:.1f}", ha="center", va="bottom",
                    fontsize=6.0, color=INK, zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground="white")])

    for c, lvl in zip(centres, sota):
        ax.plot([c - group_w / 2 - 0.03, c + group_w / 2 + 0.03], [lvl, lvl],
                ls=(0, (5, 3)), color=SOTA_C, lw=1.0, zorder=6)
        ax.text(c + group_w / 2 + 0.06, lvl, f"{lvl:.1f}", ha="left",
                va="center", fontsize=6.0, color=SOTA_C, zorder=8)

    lo = 40.0
    hi = (mean + sd).max() + 6.0
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.62, n_d - 0.38)
    ax.set_xticks(centres)
    ax.set_xticklabels(DATASETS, fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel("Success rate (%)", fontsize=8.5)
    ax.grid(axis="y", lw=0.5, color="#e2e6ea", zorder=0)
    style_axes(ax)

    handles = [Patch(facecolor=c, edgecolor="white", linewidth=0.5, label=m)
               for m, c in zip(METHODS, METHOD_COLORS)]
    handles.append(Line2D([], [], ls=(0, (5, 3)), color="#4d4d4d", lw=1.0,
                          label="Best baseline"))
    ax.legend(handles=handles, loc="upper center", ncol=7, fontsize=8.0,
              frameon=False, bbox_to_anchor=(0.5, 1.075), columnspacing=1.8,
              handlelength=1.6)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    savefig(fig, "1114_wm_blue_perf_compare.png")


if __name__ == "__main__":
    main()
