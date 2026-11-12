"""Request-latency tail for three service configurations, read as a CCDF.

The main panel is the claim: the mean and even the median say almost nothing
about a request-serving system, because the interesting mass lives four orders
of magnitude down the complementary CDF.  Plotting the survival function on log
axes turns that tail into a straight-ish line whose height at p99 and p99.9 is
what a service-level objective is actually written against.  Every quoted
number is a quantile of the plotted sample, and the tail-to-median ratio in the
title is computed from them.  The two subordinate panels carry the evidence:
how the percentile band moves through the day, and where the queue knee sits on
the throughput-latency curve.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
BLUE = "#3b5b92"
RED = "#b23a48"

N = 120_000
QUANTS = [(0.50, "p50"), (0.95, "p95"), (0.99, "p99"), (0.999, "p99.9")]
CONFIGS = [
    ("baseline", GREY, 1.00, 0.055),
    ("tuned allocator", BLUE, 0.74, 0.022),
    ("with GC pauses", RED, 1.06, 0.140),
]


def latencies(seed=17):
    """Log-normal service time plus a heavy-tailed stall component."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, _, scale, tail_p in CONFIGS:
        body = rng.lognormal(np.log(8.5 * scale), 0.42, N)
        stall = rng.pareto(1.35, N) * 26.0 * scale
        hit = rng.random(N) < tail_p
        out[name] = body + hit * stall
    return out


def ccdf(x, n_pts=800):
    xs = np.sort(x)
    n = xs.size
    j = np.unique(np.round(np.logspace(0, np.log10(n), n_pts)).astype(int))
    j = j[(j >= 1) & (j <= n)]
    idx = n - j                      # positions counted from the tail
    return xs[idx], j / float(n)


def timeline(seed=4, n=260):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 60.0          # minutes
    load = 1.0 + 0.34 * np.sin(2 * np.pi * t / 3.6) + 0.16 * np.sin(t * 2.2)
    p50 = 7.4 * load * (1.0 + rng.normal(0.0, 0.035, n))
    p95 = p50 * (2.6 + 0.9 * np.clip(load - 1.0, 0, None) * 3.0)
    p99 = p95 * (1.9 + 1.4 * np.clip(load - 1.15, 0, None) * 4.0)
    burst = np.exp(-0.5 * ((t - 2.6) / 0.16) ** 2)
    return t, p50, p95 * (1 + 0.5 * burst), p99 * (1 + 2.4 * burst)


def knee_curve(seed=9, n=26):
    rng = np.random.default_rng(seed)
    cap = 42_000.0                   # requests per second at saturation
    rho = np.linspace(0.08, 0.955, n)
    p99 = 26.0 / (1.0 - rho) ** 0.92
    p99 = p99 * (1.0 + rng.normal(0.0, 0.02, n))
    return rho * cap, p99, cap


def main_panel(ax, data):
    marks = {}
    for name, color, _, _ in CONFIGS:
        x, y = ccdf(data[name])
        ax.plot(x, y, lw=1.8, color=color, zorder=4, label=name)
        marks[name] = [float(np.quantile(data[name], q)) for q, _ in QUANTS]

    for q, tag in QUANTS:
        ax.axhline(1.0 - q, color="#cccccc", lw=0.7, ls=(0, (2.4, 2.0)),
                   zorder=1)
        ax.text(1.05e3, 1.0 - q, tag, ha="right", va="bottom", fontsize=6.6,
                color=MUTED)

    ref = marks["baseline"]
    for (q, tag), v in zip(QUANTS, ref):
        ax.plot([v, v], [4.0e-5, 1.0 - q], color=GREY, lw=0.7,
                ls=(0, (1.5, 1.8)), zorder=2)
        ax.text(2.15, (1.0 - q) * 0.86, f"baseline {tag} = {v:,.0f} ms",
                ha="left", va="top", fontsize=6.6, color=GREY)

    ratio = ref[-1] / ref[0]
    worst = marks["with GC pauses"][-1]
    ax.text(1.05e3, 0.34,
            f"with GC pauses\np99.9 = {worst:,.0f} ms, "
            f"{worst / ref[-1]:.1f}x the baseline tail",
            ha="right", va="top", fontsize=7.0, color=RED, linespacing=1.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2.0, 1.1e3)
    ax.set_ylim(4.0e-5, 1.05)
    ax.set_xlabel("request latency (ms, log scale)")
    ax.set_ylabel("fraction of requests slower than x (-, log scale)")
    ax.grid(axis="x", lw=0.5, alpha=0.55)
    ax.legend(loc="lower left", fontsize=7.2, frameon=False, handlelength=1.8,
              borderaxespad=0.6)
    ax.set_title(f"Baseline p99.9 is {ratio:.0f}x its own median",
                 fontsize=9.0, fontweight="bold", color=INK)
    return marks, ratio


def band_panel(ax, t, p50, p95, p99):
    ax.fill_between(t, p50, p99, color=BLUE, alpha=0.14, linewidth=0.0,
                    zorder=2)
    ax.fill_between(t, p50, p95, color=BLUE, alpha=0.28, linewidth=0.0,
                    zorder=3)
    ax.plot(t, p50, lw=1.4, color=BLUE, zorder=5)
    for lab, series in (("p99", p99), ("p95", p95), ("p50", p50)):
        ax.text(t[-1] * 0.995, series[-1], lab, ha="right", va="bottom",
                fontsize=6.4, color=BLUE)
    j = int(np.argmax(p99))
    ax.plot([t[j]], [p99[j]], marker="o", ms=3.6, color=RED, zorder=6)
    ax.text(t[j] + 0.10, p99[j], f"peak p99 {p99[j]:.0f} ms", ha="left",
            va="center", fontsize=6.5, color=RED)
    ax.set_yscale("log")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(*shared_limits([p50, p99], pad=0.14, log=True))
    ax.set_xlabel("time (min)", fontsize=7.4)
    ax.set_ylabel("latency (ms, log)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Percentile band over one window", fontsize=8.2,
                 fontweight="bold", color=INK)


def knee_panel(ax, thr, p99, cap):
    ax.plot(thr / 1e3, p99, marker="o", ms=2.8, lw=1.5, color=BLUE,
            mfc="white", mew=0.9, zorder=4)
    base = float(p99[0])
    k = int(np.flatnonzero(p99 > 3.0 * base)[0])
    ax.axvline(thr[k] / 1e3, color=RED, lw=0.9, ls=(0, (3, 2.2)), zorder=3)
    ax.text(thr[k] / 1e3 - 0.8, p99.max() * 0.72,
            f"knee at {thr[k] / 1e3:.0f} k req/s\n"
            f"({100.0 * thr[k] / cap:.0f} % of capacity)", ha="right",
            va="top", fontsize=6.5, color=RED, linespacing=1.35)
    ax.set_yscale("log")
    ax.set_xlim(0.0, cap / 1e3 * 1.02)
    ax.set_xlabel("offered load (k req/s)", fontsize=7.4)
    ax.set_ylabel("p99 latency (ms, log)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Throughput buys latency until it does not", fontsize=8.2,
                 fontweight="bold", color=INK)
    return thr[k]


def main():
    set_pub_style()
    data = latencies()
    t, p50, p95, p99 = timeline()
    thr, p99_load, cap = knee_curve()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_band = fig.add_subplot(gs[0, 1])
    ax_knee = fig.add_subplot(gs[1, 1])

    marks, ratio = main_panel(ax_main, data)
    band_panel(ax_band, t, p50, p95, p99)
    knee = knee_panel(ax_knee, thr, p99_load, cap)

    base = marks["baseline"]
    tuned = marks["tuned allocator"]
    fig.tight_layout(rect=[0, 0.058, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_band, ax_knee], dx=-0.50, dy=0.12)
    savefig(fig, "784_sys_latency_tail.png")


if __name__ == "__main__":
    main()
