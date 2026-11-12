"""Figure 623 - Twenty-four hour workload profile against a latency SLA.

Throughput drives everything.  A diurnal QPS curve plus one incident is fed
through a queueing delay model to produce the P50 / P95 / P99 latency ladder
in panel (a); the red bands are the intervals where the plotted P99 curve is
actually above the SLA line, found by scanning that array rather than drawn
by hand.  Panel (b) shares the time axis and shows what the traffic was made
of.  Panels (c) and (d) pool per-request latencies drawn from the same
per-bin distributions, so the quantiles marked on the CDF are the quantiles
of the sample that is plotted.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import panel_label_inches

QPS_C = "#2e7d5b"           # house green - throughput
LAD = ["#9cc3dd", "#3b5b92", "#12324f"]     # P50 / P95 / P99 ladder
ACCENT = "#b23a48"          # house red - SLA line and every breach
SLA_MS = 120.0
CAPACITY = 4800.0           # QPS at which the service saturates
MIX = ["point read", "range scan", "insert", "update"]


def build(rng, n=288):
    t = np.linspace(0.0, 24.0, n)
    diurnal = 0.5 - 0.5 * np.cos(2 * np.pi * (t - 4.5) / 24.0)
    qps = 900.0 + 2500.0 * diurnal ** 1.4
    qps += 1450.0 * np.exp(-0.5 * ((t - 14.3) / 0.42) ** 2)      # incident
    qps += 1300.0 * np.exp(-0.5 * ((t - 20.1) / 0.55) ** 2)      # batch job
    qps = np.maximum(qps + rng.normal(0.0, 55.0, n), 60.0)

    util = np.clip(qps / CAPACITY, 0.02, 0.90)
    p50 = 3.1 / (1.0 - util)
    p95 = p50 * (2.2 + 1.9 * util)
    p99 = p50 * (3.4 + 6.8 * util ** 2)
    p50 *= 1.0 + rng.normal(0.0, 0.02, n)
    p95 *= 1.0 + rng.normal(0.0, 0.025, n)
    p99 *= 1.0 + rng.normal(0.0, 0.03, n)

    share = np.stack([
        0.46 - 0.10 * diurnal, 0.17 + 0.05 * diurnal,
        0.21 + 0.04 * diurnal, 0.16 + 0.01 * diurnal])
    share = share / share.sum(axis=0, keepdims=True)
    return t, qps, p50, p95, p99, share


def pooled_latency(rng, qps, p50, p99):
    """Per-request sample whose per-bin median and 99th match the curves."""
    out = []
    for q, m, hi in zip(qps, p50, p99):
        k = max(int(q / 45.0), 8)
        sig = max(np.log(hi / m) / 2.3263, 0.05)
        out.append(rng.lognormal(np.log(m), sig, k))
    return np.concatenate(out)


def runs(mask):
    """Contiguous True intervals of a boolean array as (start, stop) pairs."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(623)
    t, qps, p50, p95, p99, share = build(rng)
    sample = pooled_latency(rng, qps, p50, p99)
    breach = runs(p99 > SLA_MS)
    breach_min = sum(t[b] - t[a] for a, b in breach) * 60.0

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.058, right=0.945, top=0.885, bottom=0.125,
                          wspace=0.34, hspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_c = fig.add_subplot(gs[0, 1])
    ax_z = fig.add_subplot(gs[1, 1])

    # ---- (a) throughput and latency percentiles ---------------------------
    for a, b in breach:
        for axx in (ax, ax_m):
            axx.axvspan(t[a], t[b], facecolor=ACCENT, alpha=0.11,
                        edgecolor="none", zorder=1)
    ax.fill_between(t, 0.0, qps, facecolor=QPS_C, alpha=0.16, lw=0.0,
                    zorder=2)
    ax.plot(t, qps, color=QPS_C, lw=1.3, zorder=4)
    ax.set_ylabel("throughput (QPS)", fontsize=8.0, color=QPS_C)
    ax.tick_params(axis="y", labelsize=7.0, colors=QPS_C)
    ax.tick_params(axis="x", labelsize=7.0, labelbottom=False)
    ax.set_ylim(0, qps.max() * 1.30)
    ax.set_xlim(0, 24)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color(QPS_C)

    axl = ax.twinx()
    for y, c, lab in zip((p50, p95, p99), LAD, ("P50", "P95", "P99")):
        axl.plot(t, y, color=c, lw=1.5, zorder=6, label=lab)
    axl.legend(loc="upper left", frameon=False, fontsize=6.8, ncol=3,
               handlelength=1.5, handletextpad=0.4, columnspacing=1.1,
               borderpad=0.2)
    axl.axhline(SLA_MS, color=ACCENT, lw=1.1, ls=(0, (4.5, 2.2)), zorder=7)
    axl.text(0.35, SLA_MS * 1.06, "P99 SLA %.0f ms" % SLA_MS, fontsize=6.6,
             color=ACCENT, va="bottom", ha="left", zorder=8)
    axl.set_ylabel("latency (ms)", fontsize=8.0)
    axl.tick_params(labelsize=7.0)
    axl.set_ylim(0, p99.max() * 1.22)
    axl.spines["top"].set_visible(False)
    axl.annotate("peak P99 %.0f ms" % p99.max(),
                 xy=(t[int(np.argmax(p99))], p99.max()),
                 xytext=(t[int(np.argmax(p99))] - 4.4, p99.max() * 0.99),
                 fontsize=6.6, color=ACCENT, ha="right", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT))
    ax.set_title("Peak %.0f QPS, %.0f min above the SLA in 24 h"
                 % (qps.max(), breach_min), fontsize=8.8, fontweight="bold",
                 pad=4.0)

    # ---- (b) query mix ------------------------------------------------------
    ramp = get_palette("cb_sequential_blue", 6)[1:5]
    ax_m.stackplot(t, 100.0 * share, colors=ramp, edgecolor="white",
                   linewidth=0.4, zorder=2)
    mids = np.cumsum(100 * share[:, -1]) - 0.5 * 100 * share[:, -1]
    for lab, y, c in zip(MIX, mids, ramp):
        ax_m.text(23.6, y, lab, ha="right", va="center", fontsize=6.4,
                  color="white" if c in ramp[2:] else "#1a1a1a",
                  fontfamily="monospace", zorder=5)
    ax_m.set_ylim(0, 100)
    ax_m.set_xlim(0, 24)
    ax_m.set_xticks(np.arange(0, 25, 3))
    ax_m.set_xlabel("time of day (h)", fontsize=8.0)
    ax_m.set_ylabel("query mix (%)", fontsize=7.6)
    ax_m.tick_params(labelsize=6.8)
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)

    # ---- (c) latency CDF ---------------------------------------------------
    xs = np.sort(sample)
    cdf = np.arange(1, xs.size + 1) / xs.size
    q50, q95, q99 = np.percentile(sample, [50, 95, 99])
    ax_c.plot(xs, cdf, color=LAD[2], lw=1.6, zorder=4)
    for q, c, lab in zip((q50, q95, q99), LAD, ("P50", "P95", "P99")):
        ax_c.plot([q, q], [0, np.interp(q, xs, cdf)], color=c, lw=0.9,
                  ls=(0, (3.0, 1.8)), zorder=3)
        ax_c.text(q + q99 * 0.022, 0.045, "%s %.0f ms" % (lab, q),
                  rotation=90, fontsize=6.2, color=c, ha="left",
                  va="bottom", zorder=5)
    ax_c.axvline(SLA_MS, color=ACCENT, lw=1.0, ls=(0, (4.5, 2.2)), zorder=4)
    ax_c.set_xlim(0, q99 * 1.45)
    ax_c.set_ylim(0, 1.02)
    ax_c.set_xlabel("request latency (ms)", fontsize=7.4)
    ax_c.set_ylabel("cumulative share (-)", fontsize=7.4)
    ax_c.tick_params(labelsize=6.6)
    ax_c.grid(True, lw=0.45, alpha=0.5)
    ax_c.set_axisbelow(True)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)
    ax_c.set_title("Pooled latency CDF, n = %s requests"
                   % format(xs.size, ","), fontsize=8.5, fontweight="bold",
                   pad=3.5)

    # ---- (d) tail magnifier -------------------------------------------------
    ax_z.plot(xs, cdf, color=LAD[2], lw=1.6, zorder=4)
    over = float((sample > SLA_MS).mean())
    ax_z.axvspan(SLA_MS, xs.max() * 1.3, facecolor=ACCENT, alpha=0.07,
                 edgecolor="none", zorder=1)
    ax_z.axvline(SLA_MS, color=ACCENT, lw=1.0, ls=(0, (4.5, 2.2)), zorder=4)
    ax_z.set_xscale("log")
    lo_z, hi_z = q95 * 0.85, xs.max() * 1.15
    ax_z.set_xlim(lo_z, hi_z)
    ax_z.set_ylim(0.94, 1.002)
    ticks = [v for v in (40, 60, 100, 150, 250, 400, 700, 1200)
             if lo_z <= v <= hi_z]
    ax_z.set_xticks(ticks)
    ax_z.set_xticklabels([str(v) for v in ticks])
    ax_z.set_xticks([], minor=True)
    ax_z.set_xlabel("request latency (ms)", fontsize=7.4)
    ax_z.set_ylabel("cumulative share (-)", fontsize=7.4)
    ax_z.tick_params(labelsize=6.6)
    ax_z.grid(True, which="both", lw=0.4, alpha=0.45)
    ax_z.set_axisbelow(True)
    for s in ("top", "right"):
        ax_z.spines[s].set_visible(False)
    ax_z.text(SLA_MS * 1.10, 0.947, "%.2f%% of requests\nbreach the SLA"
              % (100 * over), fontsize=6.4, color=ACCENT, ha="left",
              va="bottom", linespacing=1.35, zorder=6)
    ax_z.set_title("Tail magnified above the 94th percentile", fontsize=8.5,
                   fontweight="bold", pad=3.5)

    for a, lab in ((ax, "(a)"), (ax_m, "(b)"), (ax_c, "(c)"), (ax_z, "(d)")):
        panel_label_inches(fig, a, lab, dx=-0.42, dy=0.09)



    savefig(fig, "623_db_workload_profile.png")


if __name__ == "__main__":
    main()
