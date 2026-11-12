"""Figure 644 - Per-node resource occupancy across an edge cluster.

Six edge nodes, one shared day, one shared 0-100 % axis.  Each small multiple
stacks the three workload classes that consume the node's CPU, so the top of
the stack is the node's total CPU utilisation, and overlays memory and
bandwidth occupancy as lines on the same percentage scale.  Letting any panel
autoscale would make a quiet node look as busy as a saturated one, which is
precisely the comparison the figure exists to support.  The right-hand panel
reduces the same six series to the two cluster-level numbers an operator
watches: how scattered the free capacity is, and how often a placement is
refused.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import panel_labels_inches, panel_label_inches

ACCENT = "#b23a48"          # house red - overload only
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
OVERLOAD = 90.0             # % CPU at which the scheduler stops placing work
CLASSES = ["inference", "pre-processing", "system"]
NODES = [("edge-01", "factory floor", 0.92, 8.5),
         ("edge-02", "warehouse", 0.74, 11.0),
         ("edge-03", "street cabinet", 1.06, 17.5),
         ("edge-04", "campus rooftop", 0.68, 13.5),
         ("edge-05", "metro station", 1.14, 18.5),
         ("edge-06", "depot yard", 0.58, 9.0)]


def cluster(rng, n=289):
    """Return (hours, cpu stacks, memory, bandwidth) for the six nodes."""
    t = np.linspace(0.0, 24.0, n)
    stacks, mem, bw = [], [], []
    for _, _, gain, peak in NODES:
        diurnal = np.exp(-0.5 * ((t - peak) / 3.4) ** 2)
        second = 0.42 * np.exp(-0.5 * ((t - peak + 7.5) / 2.6) ** 2)
        drift = np.cumsum(rng.normal(0.0, 1.0, n))
        drift = 5.0 * (drift - drift.mean()) / (np.abs(drift).max() + 1e-9)
        infer = np.clip(gain * (60.0 * (diurnal + second)) + drift, 2.0, None)
        prep = np.clip(0.36 * infer + rng.normal(0.0, 1.4, n) + 4.0, 1.0, None)
        sysd = np.clip(6.5 + 1.6 * np.sin(t / 1.7) + rng.normal(0, 0.5, n),
                       2.0, None)
        total = infer + prep + sysd
        scale = np.clip(total / 99.0, 1.0, None)       # a node cannot exceed
        stacks.append(np.vstack([infer, prep, sysd]) / scale)     # 100 % CPU
        mem.append(np.clip(28.0 + 0.62 * total / scale +
                           rng.normal(0.0, 1.1, n), 5.0, 99.0))
        bw.append(np.clip(9.0 + 0.78 * infer / scale +
                          rng.normal(0.0, 1.6, n), 3.0, 99.0))
    return t, np.array(stacks), np.array(mem), np.array(bw)


def main():
    set_pub_style()
    rng = np.random.default_rng(644)
    t, stacks, mem, bw = cluster(rng)
    cpu = stacks.sum(axis=1)                       # (6, n) total CPU per node
    hot = cpu > OVERLOAD
    step = t[1] - t[0]

    fills = get_palette("cb_sequential_blue", 5)[1:4]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.40],
                          left=0.055, right=0.962, top=0.815, bottom=0.165,
                          wspace=0.24)
    gsl = gs[0, 0].subgridspec(2, 3, wspace=0.16, hspace=0.46)
    axes = np.array([[fig.add_subplot(gsl[r, c]) for c in range(3)]
                     for r in range(2)])
    ax_r = fig.add_subplot(gs[0, 1])

    for k, (name, site, _, _) in enumerate(NODES):
        ax = axes[k // 3, k % 3]
        ax.stackplot(t, stacks[k], colors=fills, edgecolor="white",
                     linewidth=0.35, zorder=3)
        ax.plot(t, mem[k], lw=1.0, color=GOLD, zorder=5)
        ax.plot(t, bw[k], lw=1.0, color=PURPLE, ls=(0, (2.8, 1.6)), zorder=5)
        ax.axhline(OVERLOAD, color=ACCENT, lw=0.8, ls=(0, (2.6, 1.8)),
                   zorder=6)
        ax.fill_between(t, 0.0, 100.0, where=hot[k], facecolor=ACCENT,
                        alpha=0.14, edgecolor="none", zorder=2)

        mins = hot[k].sum() * step * 60.0
        if mins > 0:
            ax.text(0.03, 0.985, "%.0f min over %.0f %%" % (mins, OVERLOAD),
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6.2, color=ACCENT, fontweight="bold", zorder=8)

        ax.set_xlim(0.0, 24.0)
        ax.set_ylim(0.0, 100.0)                    # one scale for all six
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=6.5, length=2.5)
        ax.set_title("%s  %s" % (name, site), fontsize=7.8,
                     fontweight="bold", color=COLORS["ink"], pad=3.0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if k % 3:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("resource utilisation (%)", fontsize=7.4)
        if k // 3 == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time of day (h)", fontsize=7.4)

    # ---- right: what the cluster looks like from the scheduler ------------
    free = 100.0 - cpu
    frag = 100.0 * (1.0 - free.max(axis=0) / np.maximum(free.sum(axis=0), 1e-9))
    reject = 100.0 * hot.mean(axis=0)
    win = 13
    ker = np.ones(win) / win
    frag_s = np.convolve(np.pad(frag, win // 2, mode="edge"), ker, "valid")
    rej_s = np.convolve(np.pad(reject, win // 2, mode="edge"), ker, "valid")

    ax_r.plot(t, frag_s, lw=1.4, color="#3b5b92", zorder=4,
              label="free-capacity fragmentation")
    ax_r.set_xlim(0.0, 24.0)
    ax_r.set_ylim(frag_s.min() - 5.0, frag_s.max() + 9.0)
    ax_r.set_xticks([0, 6, 12, 18, 24])
    ax_r.set_xlabel("time of day (h)", fontsize=7.4)
    ax_r.set_ylabel("fragmentation of free capacity (%)", fontsize=7.4,
                    color="#3b5b92")
    ax_r.tick_params(axis="y", labelsize=6.5, colors="#3b5b92")
    ax_r.tick_params(axis="x", labelsize=6.5)
    ax_r.grid(lw=0.45, alpha=0.45)
    ax_r.set_axisbelow(True)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["left"].set_color("#3b5b92")

    ax_r2 = ax_r.twinx()
    ax_r2.fill_between(t, 0.0, rej_s, facecolor=ACCENT, alpha=0.16,
                       edgecolor="none", zorder=2)
    ax_r2.plot(t, rej_s, lw=1.4, color=ACCENT, zorder=4,
               label="nodes refusing placement")
    ax_r2.set_ylim(0.0, 100.0)
    ax_r2.set_ylabel("nodes refusing placement (%)", fontsize=7.4,
                     color=ACCENT)
    ax_r2.tick_params(axis="y", labelsize=6.5, colors=ACCENT)
    ax_r2.spines["top"].set_visible(False)
    ax_r2.spines["right"].set_color(ACCENT)
    ax_r2.spines["left"].set_color("#3b5b92")

    peak_h = t[int(np.argmax(rej_s))]
    ax_r2.text(0.03, 0.035, "peak refusal %.0f %% of nodes at %02d:%02d"
               % (rej_s.max(), int(peak_h), int(peak_h % 1 * 60)),
               transform=ax_r2.transAxes, fontsize=6.4, color=ACCENT,
               ha="left", va="bottom", zorder=6)
    ax_r.set_title("Cluster view of the same day", fontsize=8.5,
                   fontweight="bold", pad=3.5)

    handles = ([Patch(facecolor=c, edgecolor="white", label=l)
                for c, l in zip(fills, CLASSES)] +
               [Line2D([], [], color=GOLD, lw=1.2, label="memory"),
                Line2D([], [], color=PURPLE, lw=1.2, ls=(0, (2.8, 1.6)),
                       label="bandwidth"),
                Line2D([], [], color=ACCENT, lw=1.0, ls=(0, (2.6, 1.8)),
                       label="overload threshold %.0f %%" % OVERLOAD)])

    panel_labels_inches(fig, axes, dx=-0.26, dy=0.07)
    panel_label_inches(fig, ax_r, "(g)", dx=-0.40, dy=0.07)

    busiest = int(np.argmax(hot.sum(axis=1)))

    savefig(fig, "644_edg_resource_allocation.png")


if __name__ == "__main__":
    main()
