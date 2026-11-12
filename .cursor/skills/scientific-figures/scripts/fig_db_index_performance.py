"""Figure 622 - Index strategy: lookup latency, maintenance cost, break-even.

One latency model is shared by all three panels.  Panel (a) sweeps table size
and reports point-lookup latency for five access strategies; the crossings
with the sequential-scan curve are found numerically from the plotted arrays,
so the shaded break-even band is exactly where the indexes start to pay.
Panel (b) prices the same indexes on the write side, and panel (c) mixes the
two into a total cost per operation, whose lowest envelope gives the
read-share intervals in which each strategy is the right default.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, COLORS
from figkit import panel_label_inches

SEQ = "#b23a48"             # house red - the strategy that does not scale
BTREE = "#3b5b92"           # house blue - the default index
HASH = "#c9a227"
COMPOSITE = "#8e6bbf"
COVER = "#2e7d5b"           # house green - the tuned choice

NAMES = ["no index (seq scan)", "B-tree", "hash", "composite", "covering"]
COLS = [SEQ, BTREE, HASH, COMPOSITE, COVER]
RANGE_LIFT = [1.00, 1.40, None, 1.28, 1.30]   # None: falls back to seq scan
WRITE_US = [600.0, 950.0, 880.0, 1550.0, 2300.0]
SPACE_PCT = [0.0, 18.0, 12.0, 31.0, 46.0]
N_REF = 1.0e7               # table size used by panels (b) and (c)
Q_RANGE = 0.30              # share of reads that are range scans


def point_latency(n):
    """Point-lookup latency in microseconds for each strategy."""
    lg = np.log2(n)
    return [50.0 + 0.024 * n, 90.0 * lg, 280.0 + 4.0 * lg, 65.0 * lg,
            45.0 * lg]


def crossing(x, y1, y2):
    """Where two curves cross, interpolated in log-x; None if they never do."""
    d = np.asarray(y1) - np.asarray(y2)
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    lx = np.log10(x)
    t = d[i] / (d[i] - d[i + 1])
    return float(10.0 ** (lx[i] + t * (lx[i + 1] - lx[i])))


def main():
    set_pub_style(base_font=8)

    n = np.logspace(3, 8, 400)
    lat = point_latency(n)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.062, right=0.965, top=0.885, bottom=0.125,
                          wspace=0.32, hspace=0.56)
    ax = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) latency versus table size ------------------------------------
    xo = [crossing(n, lat[0], lat[i]) for i in range(1, 5)]
    xo = [v for v in xo if v is not None]
    ax.axvspan(min(xo), max(xo), facecolor=SEQ, alpha=0.075, edgecolor="none",
               zorder=1)
    for k, (name, col) in enumerate(zip(NAMES, COLS)):
        ax.plot(n, lat[k], color=col, lw=1.9 if k in (0, 4) else 1.4,
                label=name, zorder=5 if k in (0, 4) else 4)
    for k, xc in zip(range(1, 5), xo):
        yc = np.interp(np.log10(xc), np.log10(n), lat[0])
        ax.plot([xc], [yc], marker="o", ms=3.8, mfc="white", mec=COLS[k],
                mew=1.1, zorder=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(n[0], n[-1])
    ax.set_ylim(30.0, 6.0e6)
    ax.set_xlabel("table size (rows)", fontsize=8.0)
    ax.set_ylabel(r"point-lookup latency ($\mu$s)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(True, which="major", lw=0.45, alpha=0.5)
    ax.grid(True, which="minor", lw=0.3, alpha=0.28)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("break-even band\n%.0fk to %.0fk rows"
                % (min(xo) / 1e3, max(xo) / 1e3),
                xy=(np.sqrt(min(xo) * max(xo)), 1.1e6),
                xytext=(3.0e5, 2.2e6), fontsize=6.8, color=SEQ, ha="left",
                va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=SEQ))
    gain = lat[0][-1] / lat[4][-1]
    ax.text(0.975, 0.035, "covering index is %.0fx faster at %.0e rows"
            % (gain, n[-1]), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.8, color=COVER)
    ax.legend(loc="upper left", frameon=False, fontsize=7.0,
              handlelength=1.7, handletextpad=0.5, labelspacing=0.38,
              title="access strategy", title_fontsize=7.0)
    ax.set_title("Lookup latency across five orders of table size",
                 fontsize=8.8, fontweight="bold", pad=4.0)

    # ---- (b) maintenance cost versus space --------------------------------
    ref = point_latency(N_REF)
    speed = [ref[0] / v for v in ref]
    amp = [w / WRITE_US[0] for w in WRITE_US]
    off = [(0.10, 2.6, "left"), (-0.12, 3.4, "right"),
           (-0.14, -1.8, "right"), (0.12, 2.6, "left"), (-0.14, 3.4, "right")]
    for k in range(5):
        ax_t.scatter([amp[k]], [SPACE_PCT[k]], s=16.0 * np.log10(speed[k] + 1.0)
                     * 10.0 + 18.0, facecolor=COLS[k], alpha=0.85,
                     edgecolor="white", lw=0.6, zorder=4)
        dx, dy, ha = off[k]
        ax_t.text(amp[k] + dx, SPACE_PCT[k] + dy, NAMES[k].split(" (")[0],
                  fontsize=6.4, color=COLS[k], ha=ha, va="bottom", zorder=6)
    ax_t.plot(amp, SPACE_PCT, color="#cccccc", lw=0.8, ls=(0, (3.5, 2.0)),
              zorder=2)
    ax_t.set_xlim(0.75, 4.55)
    ax_t.set_ylim(-4.0, 60.0)
    ax_t.set_xlabel("write amplification (x)", fontsize=7.4)
    ax_t.set_ylabel("space overhead (% of table)", fontsize=7.4)
    ax_t.tick_params(labelsize=6.6)
    ax_t.grid(True, lw=0.45, alpha=0.5)
    ax_t.set_axisbelow(True)
    for s in ("top", "right"):
        ax_t.spines[s].set_visible(False)
    ax_t.text(0.975, 0.045, "bubble area scales with read speed-up",
              transform=ax_t.transAxes, ha="right", va="bottom",
              fontsize=6.2, color=COLORS["muted"])
    ax_t.set_title("Write cost against space overhead", fontsize=8.5,
                   fontweight="bold", pad=3.5)

    # ---- (c) total cost versus read share ---------------------------------
    read_mix = []
    for k in range(5):
        pt = ref[k]
        rng_lat = ref[0] if RANGE_LIFT[k] is None else RANGE_LIFT[k] * pt
        read_mix.append((1.0 - Q_RANGE) * pt + Q_RANGE * rng_lat)
    r = np.linspace(0.0, 1.0, 400)
    cost = np.array([r * read_mix[k] + (1.0 - r) * WRITE_US[k]
                     for k in range(5)])
    win = np.argmin(cost, axis=0)

    for k in range(5):
        ax_c.plot(100 * r, cost[k], color=COLS[k], lw=1.5, zorder=4)
    edges = [0] + list(np.where(np.diff(win) != 0)[0] + 1) + [r.size]
    for a, b in zip(edges[:-1], edges[1:]):
        k = win[a]
        ax_c.axvspan(100 * r[a], 100 * r[b - 1], ymin=0.0, ymax=0.062,
                     facecolor=COLS[k], alpha=0.90, edgecolor="none",
                     zorder=6)
        if r[b - 1] - r[a] > 0.05:
            ax_c.text(50 * (r[a] + r[b - 1]), cost.min() * 1.30,
                      NAMES[k].split(" (")[0], ha="center", va="bottom",
                      fontsize=6.2, color=COLS[k], zorder=7)
    for e in edges[1:-1]:
        if r[e] < 0.02:
            continue
        ax_c.axvline(100 * r[e], color=COLORS["baseline"], lw=0.7,
                     ls=(0, (3.0, 2.0)), zorder=3)
        ax_c.text(100 * r[e], cost.max() * 0.85, "%.0f%%" % (100 * r[e]),
                  ha="center", va="top", fontsize=6.2,
                  color=COLORS["sub"], zorder=7)

    ax_c.set_yscale("log")
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(cost.min() * 0.80, cost.max() * 1.6)
    ax_c.set_xlabel("read share of the workload (%)", fontsize=7.4)
    ax_c.set_ylabel(r"cost per operation ($\mu$s)", fontsize=7.4)
    ax_c.tick_params(labelsize=6.6)
    ax_c.grid(True, lw=0.45, alpha=0.45)
    ax_c.set_axisbelow(True)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)
    ax_c.set_title("Cheapest strategy by read share", fontsize=8.5,
                   fontweight="bold", pad=3.5)

    for a, lab in ((ax, "(a)"), (ax_t, "(b)"), (ax_c, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.42, dy=0.09)

    flip = 100 * r[edges[-2]]

    savefig(fig, "622_db_index_performance.png")


if __name__ == "__main__":
    main()
