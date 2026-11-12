"""Load imbalance across 48 ranks and what partitioning does about it.

The main panel sorts the per-rank compute time of one solver step. Because
the step ends at a barrier, the whole job runs at the pace of the slowest
rank, so the part of each bar that rises above the mean is the excess that
forces everyone else to wait; it is drawn in house red on top of the blue
useful work. The two side panels are the evidence: imbalance grows with rank
count at a rate set by the partitioner, and the achievable efficiency at the
largest configuration follows directly from that ratio.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter
from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

WORK = "#3b5b92"       # house blue, useful compute
WASTE = "#b23a48"      # house red, the single emphasis object
IDEAL = "#808080"      # grey dashed reference
GOLD = "#c9a227"
GREEN = "#2e7d5b"
SUB = "#555555"
MUTED = "#777777"

NRANK = 48
PROCS = np.array([8, 16, 32, 48, 64, 128, 256, 512], dtype=float)
# Growth rate relative to the block partitioner, whose rate is pinned to the
# measured imbalance so the curve passes through the point it explains.
STRATEGIES = [("Block", WORK, 1.00), ("Cyclic", GOLD, 0.55),
              ("Graph (k-way)", GREEN, 0.22)]


def main():
    set_pub_style()
    rng = np.random.default_rng(5671)

    # ---- per-rank step time ----------------------------------------------
    t = rng.lognormal(np.log(96.0), 0.105, NRANK)
    t[rng.choice(NRANK, 4, replace=False)] *= rng.uniform(1.18, 1.42, 4)
    t = np.sort(t)[::-1]
    mean_t, max_t = float(t.mean()), float(t.max())
    imbalance = max_t / mean_t
    wait = float((max_t - t).sum())
    wait_frac = wait / (NRANK * max_t)

    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.86], hspace=0.62, wspace=0.30)
    ax = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) sorted per-rank time ----------------------------------------
    x = np.arange(NRANK)
    ax.bar(x, np.minimum(t, mean_t), width=0.84, color=WORK, edgecolor="none",
           zorder=2)
    ax.bar(x, np.maximum(t - mean_t, 0.0), bottom=np.minimum(t, mean_t),
           width=0.84, color=WASTE, edgecolor="none", zorder=2)
    ax.axhline(mean_t, color=IDEAL, ls="--", lw=1.0, zorder=3)
    ax.axhline(max_t, color=WASTE, ls=":", lw=1.0, zorder=3)
    ax.text(NRANK - 0.5, mean_t, f" mean {mean_t:.1f} ms", ha="right",
            va="bottom", fontsize=6.8, color=SUB)
    ax.text(NRANK - 0.5, max_t, f" max {max_t:.1f} ms", ha="right",
            va="bottom", fontsize=6.8, color=WASTE)

    ax.set_xlim(-1.0, NRANK)
    ax.set_ylim(0.0, max_t * 1.45)
    ax.set_xticks(np.arange(0, NRANK + 1, 8))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("rank, sorted by step time (-)", fontsize=8.0)
    ax.set_ylabel("compute time per step (ms)", fontsize=8.0)
    ax.set_title("Per-rank step time before the barrier", fontsize=8.8,
                 fontweight="bold")
    ax.grid(axis="y", lw=0.5, alpha=0.45)

    handles = [Patch(facecolor=WORK, label="Work up to the mean"),
               Patch(facecolor=WASTE, label="Excess above the mean"),
               Line2D([], [], color=IDEAL, ls="--", lw=1.0,
                      label="Mean step time"),
               Line2D([], [], color=WASTE, ls=":", lw=1.0,
                      label="Slowest rank sets the barrier")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, fontsize=7.0, frameon=False, handlelength=1.7,
              labelspacing=0.30, columnspacing=1.4)
    ax.text(0.985, 0.875,
            f"{100.0 * wait_frac:.0f} % of the allocated rank-time\n"
            f"is spent waiting at the barrier",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.8, color=WASTE, linespacing=1.35)

    # ---- (b) imbalance vs rank count, three partitioners ------------------
    base = (imbalance - 1.0) / np.log2(NRANK / PROCS[0])
    curves = {}
    for name, col, rel in STRATEGIES:
        y = 1.0 + rel * base * np.log2(PROCS / PROCS[0])
        curves[name] = y
        ax_b.plot(PROCS, y, marker="o", ms=3.0, lw=1.2, color=col,
                  mfc="white", mew=0.8)
        ax_b.text(PROCS[-1] * 1.16, y[-1], name, ha="left", va="center",
                  fontsize=6.3, color=col, clip_on=False)
    ax_b.axhline(1.0, color=IDEAL, ls="--", lw=0.9)
    ax_b.axvline(NRANK, color=GOLD, ls=":", lw=0.8)
    ax_b.plot([NRANK], [imbalance], marker="D", ms=4.6, color=WASTE,
              mec="white", mew=0.7, ls="none", zorder=5)
    ax_b.text(NRANK * 0.82, imbalance + 0.06, f"measured {imbalance:.2f}",
              ha="right", va="bottom", fontsize=6.3, color=WASTE)

    ax_b.set_xscale("log", base=2)
    ax_b.set_xlim(7.0, 640.0)
    ax_b.set_xticks([8, 32, 128, 512])
    ax_b.set_xticklabels(["8", "32", "128", "512"])
    ax_b.xaxis.set_minor_locator(FixedLocator(PROCS))
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    ax_b.tick_params(labelsize=6.5, length=2.4)
    ax_b.tick_params(which="minor", length=1.6)
    ax_b.set_xlabel("ranks (-)", fontsize=7.2)
    ax_b.set_ylabel("imbalance max/mean (-)", fontsize=7.2)
    ax_b.set_title("Imbalance vs rank count", fontsize=8.5,
                   fontweight="bold")
    ax_b.grid(lw=0.45, alpha=0.45)

    # ---- (c) efficiency ceiling implied by (b) ----------------------------
    big = int(np.flatnonzero(PROCS == 512)[0])
    names = [s[0] for s in STRATEGIES]
    cols = [s[1] for s in STRATEGIES]
    eff = np.array([100.0 / curves[n][big] for n in names])
    ax_c.bar(np.arange(len(names)), eff, width=0.62, color=cols,
             edgecolor="none")
    for i, v in enumerate(eff):
        ax_c.text(i, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                  fontsize=6.3, color=SUB)
    ax_c.axhline(100.0, color=IDEAL, ls="--", lw=0.9)
    ax_c.set_xticks(np.arange(len(names)))
    ax_c.set_xticklabels(["Block", "Cyclic", "Graph"], fontsize=6.5)
    ax_c.set_ylim(0.0, 118.0)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.tick_params(labelsize=6.5, length=2.4)
    ax_c.set_ylabel("efficiency ceiling (%)", fontsize=7.2)
    ax_c.set_title(f"Ceiling at {PROCS[big]:.0f} ranks", fontsize=8.5,
                   fontweight="bold")
    ax_c.grid(axis="y", lw=0.45, alpha=0.45)


    fig.tight_layout(rect=[0, 0.085, 1, 0.945], w_pad=2.0, h_pad=2.0)
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.135))
    panel_labels_inches(fig, [ax, ax_b, ax_c], dx=-0.42, dy=0.06)
    savefig(fig, "567_hpc_load_balance.png")


if __name__ == "__main__":
    main()
