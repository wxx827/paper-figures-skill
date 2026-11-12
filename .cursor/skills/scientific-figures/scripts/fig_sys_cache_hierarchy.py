"""Pointer-chase latency across a working-set sweep, level by level.

The main panel is the claim: average access latency is flat while the working
set fits a cache and steps up as soon as it spills, so the curve is a direct
read-out of the memory hierarchy.  One inclusion model generates everything on
the page - the plateau latencies quoted next to each shaded band are sampled
from the plotted curve rather than typed in, and the same per-level hit
probabilities are stacked in the first subordinate panel.  The second
subordinate panel pairs each level's latency with its bandwidth, which is the
trade the hierarchy exists to make.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_palette, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
RED = "#b23a48"

KB, MB, GB = 1.0, 1024.0, 1024.0 * 1024.0

# name, capacity (KB), hit latency (ns), streaming bandwidth (GB/s)
LEVELS = [
    ("L1", 32 * KB, 1.1, 940.0),
    ("L2", 1 * MB, 4.4, 372.0),
    ("L3", 32 * MB, 17.5, 205.0),
    ("DRAM", 8 * GB, 88.0, 116.0),
]
SHARP = 3.2                      # how abruptly a level spills
LEV_COLOR = dict(zip([l[0] for l in LEVELS],
                     get_palette("cb_sequential_blue", 6)[5:1:-1]))


def hit_probabilities(size_kb):
    """Per-level hit probability under a simple inclusive-capacity model."""
    miss = [1.0 / (1.0 + (cap / size_kb) ** SHARP) for _, cap, _, _ in LEVELS]
    probs, carry = [], np.ones_like(size_kb)
    for i, m in enumerate(miss):
        if i == len(miss) - 1:
            probs.append(carry)
        else:
            probs.append(carry * (1.0 - m))
            carry = carry * m
    return np.array(probs)


def sweep(seed=11, n=180):
    size = np.logspace(np.log10(4.0), np.log10(4.0 * GB), n)
    probs = hit_probabilities(size)
    lat = np.array([l[2] for l in LEVELS])
    clean = (probs * lat[:, None]).sum(axis=0)
    rng = np.random.default_rng(seed)
    noisy = clean * (1.0 + rng.normal(0.0, 0.028, size.size))
    return size, probs, clean, noisy


def size_ticks():
    vals = [4, 32, 256, 1 * MB, 8 * MB, 64 * MB, 512 * MB, 4 * GB]
    labs = []
    for v in vals:
        if v < MB:
            labs.append(f"{v:.0f} KB")
        elif v < GB:
            labs.append(f"{v / MB:.0f} MB")
        else:
            labs.append(f"{v / GB:.0f} GB")
    return vals, labs


def main_panel(ax, size, clean, noisy):
    edges = [size[0]] + [cap for _, cap, _, _ in LEVELS[:-1]] + [size[-1]]
    for i, (name, cap, _, _) in enumerate(LEVELS):
        ax.axvspan(edges[i], edges[i + 1], color=LEV_COLOR[name], alpha=0.10,
                   linewidth=0.0, zorder=0)

    ax.plot(size, noisy, marker="o", ms=2.4, lw=0.0, color=GREY, alpha=0.75,
            zorder=3)
    ax.plot(size, clean, lw=1.9, color="#3b5b92", zorder=4)

    quoted = []
    for i, (name, cap, _, _) in enumerate(LEVELS):
        probe = cap / 3.2 if name != "DRAM" else size[-1] * 0.75
        lat_here = float(np.interp(np.log10(probe), np.log10(size), clean))
        quoted.append(lat_here)
        if name != "DRAM":
            ax.axvline(cap, color=LEV_COLOR[name], lw=0.9, ls=(0, (3, 2.2)),
                       zorder=2)
            tag = f"{cap / MB:.0f} MB" if cap >= MB else f"{cap:.0f} KB"
            ax.text(cap * 1.12, 1.28, f"{name} capacity {tag}", rotation=90,
                    ha="left", va="bottom", fontsize=6.4,
                    color=LEV_COLOR[name])
        ax.text(probe, lat_here * 1.30, f"{name}\n{lat_here:.1f} ns",
                ha="center", va="bottom", fontsize=7.0, color=LEV_COLOR[name],
                fontweight="bold", linespacing=1.3, zorder=6)

    ratio = quoted[-1] / quoted[0]
    ax.annotate(f"DRAM is {ratio:.0f}x the L1 latency",
                xy=(size[-1] * 0.55, quoted[-1]),
                xytext=(2.5 * MB, quoted[-1] * 1.55), ha="center",
                va="bottom", fontsize=7.2, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8))

    vals, labs = size_ticks()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(size[0], size[-1])
    ax.set_ylim(0.8, 240.0)
    ax.set_xticks(vals)
    ax.set_xticklabels(labs, fontsize=7.0)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "200"],
                       fontsize=7.0)
    ax.set_xlabel("working-set size (KB / MB / GB, log scale)")
    ax.set_ylabel("average access latency (ns, log scale)")
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_title(f"Each spill costs roughly one step: "
                 f"{quoted[0]:.1f} ns to {quoted[-1]:.0f} ns", fontsize=9.0,
                 fontweight="bold", color=INK)
    return quoted, ratio


def stack_panel(ax, size, probs):
    ax.stackplot(size, probs * 100.0,
                 colors=[LEV_COLOR[l[0]] for l in LEVELS], linewidth=0.0)
    for i, (name, cap, _, _) in enumerate(LEVELS):
        owns = probs[i] > 0.5
        if not owns.any():
            continue
        xs = size[owns]
        xlab = float(np.sqrt(xs[0] * xs[-1]))
        j = int(np.argmin(np.abs(size - xlab)))
        y = 100.0 * (probs[:i, j].sum() + probs[i][j] / 2.0)
        ax.text(xlab, y, name, ha="center", va="center", fontsize=6.6,
                color="white", fontweight="bold")
    vals, labs = size_ticks()
    ax.set_xscale("log")
    ax.set_xlim(size[0], size[-1])
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(vals[::2])
    ax.set_xticklabels(labs[::2], fontsize=6.5)
    ax.set_xlabel("working-set size (log scale)", fontsize=7.4)
    ax.set_ylabel("accesses served (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Which level answers the access", fontsize=8.2,
                 fontweight="bold", color=INK)


def tradeoff_panel(ax, quoted):
    bw = np.array([l[3] for l in LEVELS])
    cap = np.array([l[1] for l in LEVELS])
    lat = np.array(quoted)
    ax.plot(lat, bw, lw=0.9, color=GREY, ls=(0, (3, 2.2)), zorder=2)
    for i, (name, _, _, _) in enumerate(LEVELS):
        ax.scatter([lat[i]], [bw[i]], s=26.0 + 9.0 * np.log10(cap[i]),
                   color=LEV_COLOR[name], edgecolor="white", linewidth=0.7,
                   zorder=4)
        ax.text(lat[i] * 1.16, bw[i] * 1.05, f"{name}\n{bw[i]:.0f} GB/s",
                ha="left", va="bottom", fontsize=6.3, color=SUB,
                linespacing=1.3)
    slope = float(np.polyfit(np.log10(lat), np.log10(bw), 1)[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 260.0)
    ax.set_ylim(70.0, 2200.0)
    ax.set_xlabel("hit latency (ns, log scale)", fontsize=7.4)
    ax.set_ylabel("bandwidth (GB/s, log)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.set_title(f"Latency buys bandwidth (slope {slope:.2f})", fontsize=8.2,
                 fontweight="bold", color=INK)
    return slope


def main():
    set_pub_style()
    size, probs, clean, noisy = sweep()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_hit = fig.add_subplot(gs[0, 1])
    ax_bw = fig.add_subplot(gs[1, 1])

    quoted, ratio = main_panel(ax_main, size, clean, noisy)
    stack_panel(ax_hit, size, probs)
    slope = tradeoff_panel(ax_bw, quoted)

    fig.tight_layout(rect=[0, 0.055, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_hit, ax_bw], dx=-0.46, dy=0.12)
    savefig(fig, "782_sys_cache_hierarchy.png")


if __name__ == "__main__":
    main()
