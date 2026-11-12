"""Roofline bound for a kernel suite, with the ridge point read off the model.

The main panel is the claim: every kernel sits under a bent ceiling whose
slanted part is a memory bandwidth and whose flat part is a peak issue rate,
so the distance from a marker to the roof directly above it is the headroom
that optimisation can still recover.  Three bandwidth roofs (L1, L2, DRAM) and
three compute ceilings (scalar, SIMD, SIMD with FMA) are drawn, the ridge point
is computed from the two limits that meet there rather than typed in, and each
kernel is labelled with the fraction of its own attainable roof that it
reaches.  The two subordinate panels carry the evidence: the same fractions as
a ranked bar chart, and the displacement each kernel made in the plane after
cache blocking and vectorisation.
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
GOLD = "#c9a227"

# Machine model of one socket.  Bandwidths in GB/s, peaks in GFLOP/s.
BW = {"L1": 940.0, "L2": 372.0, "DRAM": 116.0}
PEAK = {"scalar": 62.0, "SIMD": 248.0, "SIMD + FMA": 496.0}

BLUES = get_palette("cb_sequential_blue", 7)
BW_COLOR = {"L1": BLUES[6], "L2": BLUES[5], "DRAM": BLUES[3]}
CEIL_COLOR = {"scalar": "#8a8a8a", "SIMD": "#555555", "SIMD + FMA": "#333333"}

# (name, intensity in FLOP/Byte, fraction of its attainable roof, label side)
KERNELS = [
    ("SpMV", 0.11, 0.62, "below"),
    ("7-pt stencil", 0.38, 0.71, "right"),
    ("SGEMV", 0.24, 0.44, "below"),
    ("FFT 3D", 1.5, 0.55, "below"),
    ("conv 3x3", 6.2, 0.61, "below"),
    ("SGEMM", 14.0, 0.83, "below"),
    ("N-body", 42.0, 0.74, "below"),
]

# Optimisation moves: kernel -> (intensity gain, fraction after tuning)
TUNED = {
    "SpMV": (2.3, 0.70),
    "7-pt stencil": (3.4, 0.84),
    "SGEMM": (1.15, 0.93),
}

I_LO, I_HI = 0.03, 200.0


def attainable(intensity, bw=None, peak=None):
    """Roof height at a given intensity: bandwidth ramp, then compute ceiling."""
    bw = BW["DRAM"] if bw is None else bw
    peak = PEAK["SIMD + FMA"] if peak is None else peak
    return np.minimum(bw * np.asarray(intensity, dtype=float), peak)


def kernel_table(seed=7):
    rng = np.random.default_rng(seed)
    names = [k[0] for k in KERNELS]
    inten = np.array([k[1] for k in KERNELS])
    frac = np.array([k[2] for k in KERNELS])
    frac = np.clip(frac * (1.0 + rng.normal(0.0, 0.035, frac.size)), 0.05, 0.99)
    roof = attainable(inten)
    return names, inten, roof * frac, roof


def main_panel(ax, names, inten, perf, roof):
    grid = np.logspace(np.log10(I_LO), np.log10(I_HI), 600)
    peak = PEAK["SIMD + FMA"]

    # Shaded envelope under the DRAM roof: everything reachable on this socket.
    env = attainable(grid)
    ax.fill_between(grid, 1.0, env, color="#f5f5f5", linewidth=0.0, zorder=0)

    for tag, bw in BW.items():
        ramp = grid[bw * grid <= peak * 1.0]
        if ramp.size:
            ax.plot(ramp, bw * ramp, color=BW_COLOR[tag], lw=1.9, zorder=4,
                    solid_capstyle="round")
            xa = ramp[int({"DRAM": 0.30}.get(tag, 0.42) * ramp.size)]
            ax.text(xa, bw * xa * 1.12, f"{tag}  {bw:.0f} GB/s", rotation=34,
                    rotation_mode="anchor", transform_rotates_text=True,
                    ha="center", va="bottom", fontsize=6.6,
                    color=BW_COLOR[tag])

    for tag, pk in PEAK.items():
        x0 = pk / BW["DRAM"] if tag == "SIMD + FMA" else pk / BW["L2"]
        ax.plot([x0, I_HI], [pk, pk], color=CEIL_COLOR[tag], lw=1.5,
                ls=(0, (5, 2.4)) if tag != "SIMD + FMA" else "-", zorder=4)
        ax.text(I_HI * 0.88, pk * 1.09, f"{tag}  {pk:.0f} GFLOP/s", ha="right",
                va="bottom", fontsize=6.6, color=CEIL_COLOR[tag])

    ridge = peak / BW["DRAM"]
    ax.plot([ridge], [peak], marker="o", ms=5.0, mfc="white", mec=GOLD,
            mew=1.4, zorder=7)
    ax.annotate(f"DRAM ridge point\n{ridge:.2f} FLOP/Byte",
                xy=(ridge, peak), xytext=(ridge * 0.16, peak * 2.0),
                ha="center", va="bottom", fontsize=6.8, color=GOLD,
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.7))
    ax.axvline(ridge, color=GOLD, lw=0.7, ls=(0, (1.6, 2.2)), zorder=2)

    util = perf / roof
    worst = int(np.argmin(util))
    for i, name in enumerate(names):
        emph = i == worst
        color = RED if emph else "#333333"
        ax.plot([inten[i]], [perf[i]], marker="o", ms=5.2 if emph else 4.4,
                mfc=color if emph else "white", mec=color, mew=1.2, zorder=8)
        ax.plot([inten[i], inten[i]], [perf[i], roof[i]], color=color, lw=0.7,
                ls=(0, (1.4, 1.8)), alpha=0.85, zorder=6)
        if KERNELS[i][3] == "right":
            tx, ty, ha, va = inten[i] * 1.45, perf[i], "left", "center"
        else:
            tx, ty, ha, va = inten[i], perf[i] * 0.52, "center", "top"
        ax.text(tx, ty, f"{name}\n{util[i] * 100:.0f} % of roof", ha=ha,
                va=va, fontsize=6.4, color=color, linespacing=1.3, zorder=9)

    ax.text(0.038, 1.20, "memory bound", ha="left", va="bottom", fontsize=7.0,
            color=MUTED, style="italic")
    ax.text(I_HI * 0.80, 1.20, "compute bound", ha="right", va="bottom",
            fontsize=7.0, color=MUTED, style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(I_LO, I_HI)
    ax.set_ylim(1.0, peak * 3.4)
    ax.set_xlabel("arithmetic intensity (FLOP/Byte, log scale)")
    ax.set_ylabel("attainable performance (GFLOP/s, log scale)")
    ax.grid(which="major", lw=0.5, alpha=0.6)
    ax.set_title(f"{names[worst]} leaves the most headroom: "
                 f"{util[worst] * 100:.0f} % of its roof", fontsize=9.0,
                 fontweight="bold", color=INK)
    return util, worst


def util_panel(ax, names, util, worst):
    order = np.argsort(util)
    y = np.arange(len(names))
    cols = [RED if order[i] == worst else "#3b5b92" for i in range(len(names))]
    ax.barh(y, util[order] * 100.0, height=0.66, color=cols, edgecolor="white",
            linewidth=0.6, zorder=3)
    for i, v in enumerate(util[order] * 100.0):
        ax.text(v + 1.6, y[i], f"{v:.0f}", va="center", ha="left",
                fontsize=6.2, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels([names[j] for j in order], fontsize=6.5)
    ax.set_xlim(0.0, 108.0)
    ax.set_xlabel("attainable-roof utilisation (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("Distance to the roof, ranked", fontsize=8.2,
                 fontweight="bold", color=INK)


def move_panel(ax, names, inten, perf):
    grid = np.logspace(np.log10(0.05), np.log10(60.0), 300)
    ax.plot(grid, attainable(grid), color=BW_COLOR["DRAM"], lw=1.3, zorder=3)

    gains = []
    for name, (gain, frac_after) in TUNED.items():
        i = names.index(name)
        x0, y0 = inten[i], perf[i]
        x1 = x0 * gain
        y1 = attainable(x1) * frac_after
        gains.append(y1 / y0)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#3b5b92",
                                    lw=1.0, shrinkA=1.5, shrinkB=1.5))
        ax.plot([x0], [y0], marker="o", ms=3.4, color=GREY, zorder=5)
        ax.plot([x1], [y1], marker="o", ms=3.8, color="#3b5b92", zorder=5)
        ax.text(x1 * 1.20, y1 * 0.90, f"{name}  {y1 / y0:.1f}x", ha="left",
                va="top", fontsize=6.2, color=SUB)

    gm = float(np.exp(np.mean(np.log(np.array(gains)))))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 200.0)
    ax.set_ylim(3.0, 1400.0)
    ax.set_xlabel("arithmetic intensity (FLOP/Byte, log)", fontsize=7.4)
    ax.set_ylabel("performance (GFLOP/s, log)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(which="major", lw=0.5, alpha=0.6)
    ax.set_title(f"Blocking and SIMD move kernels right and up "
                 f"(geo. mean {gm:.1f}x)", fontsize=8.2, fontweight="bold",
                 color=INK)
    return gm


def main():
    set_pub_style()
    names, inten, perf, roof = kernel_table()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_util = fig.add_subplot(gs[0, 1])
    ax_move = fig.add_subplot(gs[1, 1])

    util, worst = main_panel(ax_main, names, inten, perf, roof)
    util_panel(ax_util, names, util, worst)
    gm = move_panel(ax_move, names, inten, perf)

    fig.tight_layout(rect=[0, 0.055, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_util, ax_move], dx=-0.44, dy=0.12)
    savefig(fig, "780_sys_roofline_model.png")


if __name__ == "__main__":
    main()
