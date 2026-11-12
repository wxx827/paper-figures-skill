"""Figure 643 - End-to-end latency budget split into five stages.

One sample bank drives the whole figure: for every workload each of the five
pipeline stages is drawn 4000 times from a log-normal whose median and spread
are set per stage, and every bar, box, quantile and count is reduced from that
same bank.  The stacked bars in (a) are the mean budget, the boxes in (b) show
which stage owns the variance, and the survival curves in (c) show what the
mean hides: the tail, which is what an SLA is actually violated by.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import panel_label_inches

ACCENT = "#b23a48"          # house red - budget overruns only
SLA_MS = 100.0
N_DRAW = 4000

STAGES = ["sensing", "transport", "queueing", "compute", "backhaul"]
# workload -> mean milliseconds in each of the five stages
BUDGET = {
    "AGV motion control": [1.5, 3.2, 2.4, 6.8, 1.1],
    "AR overlay render": [2.4, 6.5, 4.8, 18.2, 2.0],
    "smart camera detect": [4.0, 9.5, 7.2, 26.5, 3.1],
    "voice front-end": [3.2, 12.0, 9.5, 31.0, 6.4],
    "vibration analytics": [2.0, 14.5, 18.0, 42.0, 8.5],
    "cloud video re-ID": [5.5, 28.0, 34.0, 46.0, 22.0],
    "OTA model sync": [1.2, 62.0, 41.0, 55.0, 28.0],
}
# log-normal shape per stage: queueing and transport are the heavy-tailed ones
SIGMA = [0.16, 0.42, 0.68, 0.22, 0.38]
TAIL_SHOW = ["AR overlay render", "vibration analytics", "OTA model sync"]


def draw_bank(rng):
    """{workload: (N_DRAW, 5) array of stage latencies in ms}."""
    bank = {}
    for name, means in BUDGET.items():
        cols = []
        for mu, sg in zip(means, SIGMA):
            med = mu / np.exp(0.5 * sg ** 2)          # keep the mean on target
            cols.append(rng.lognormal(np.log(med), sg, N_DRAW))
        bank[name] = np.column_stack(cols)
    return bank


def main():
    set_pub_style()
    rng = np.random.default_rng(43)
    bank = draw_bank(rng)

    cols = get_palette("cb_sequential_blue", 6)[1:]
    names = sorted(bank, key=lambda k: bank[k].sum(axis=1).mean())
    mean_stage = np.array([bank[n].mean(axis=0) for n in names])
    totals = np.array([bank[n].sum(axis=1) for n in names])
    mean_tot = totals.mean(axis=1)
    p99 = np.percentile(totals, 99, axis=1)
    over = mean_tot > SLA_MS

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.135, right=0.968, top=0.885, bottom=0.145,
                          wspace=0.40, hspace=0.62)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) the mean budget, stage by stage ------------------------------
    yy = np.arange(len(names))
    left = np.zeros(len(names))
    for k, stage in enumerate(STAGES):
        ax_a.barh(yy, mean_stage[:, k], left=left, height=0.62,
                  color=cols[k], edgecolor="white", linewidth=0.6,
                  label=stage, zorder=3)
        left = left + mean_stage[:, k]

    for i in range(len(names)):
        ax_a.plot([p99[i]], [yy[i]], marker="D", ms=3.4, mfc="white",
                  mec=COLORS["rule"], mew=0.8, zorder=6)
        ax_a.text(p99[i] + 5.0, yy[i], "%.0f / %.0f ms"
                  % (mean_tot[i], p99[i]), va="center", ha="left",
                  fontsize=6.2, color=ACCENT if over[i] else COLORS["sub"],
                  zorder=6)

    ax_a.axvline(SLA_MS, color=ACCENT, lw=1.1, ls=(0, (3.4, 1.8)), zorder=5)
    ax_a.text(SLA_MS - 3.0, len(names) - 0.35, "SLA %.0f ms" % SLA_MS,
              ha="right", va="center", fontsize=6.8, color=ACCENT,
              fontweight="bold", zorder=6)
    for i in np.flatnonzero(over):
        ax_a.axhspan(yy[i] - 0.44, yy[i] + 0.44, xmin=0.0, xmax=1.0,
                     facecolor=ACCENT, alpha=0.07, zorder=1)

    ax_a.set_yticks(yy)
    ax_a.set_yticklabels(names, fontsize=7.2)
    for i, t in enumerate(ax_a.get_yticklabels()):
        t.set_color(ACCENT if over[i] else COLORS["ink"])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.0, p99.max() * 1.30)
    ax_a.set_xlabel("mean end-to-end latency (ms)", fontsize=8.0)
    ax_a.tick_params(axis="x", labelsize=7.0)
    ax_a.grid(axis="x", lw=0.45, alpha=0.5)
    ax_a.set_axisbelow(True)
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)
    ax_a.legend(fontsize=7.0, frameon=False, loc="upper right", ncol=1,
                handlelength=1.5, borderpad=0.2, labelspacing=0.42,
                title="pipeline stage (mean / P99 totals at right)",
                title_fontsize=7.0)
    ax_a.set_title("%d of %d workloads overrun the budget in the mean"
                   % (int(over.sum()), over.size), fontsize=8.8,
                   fontweight="bold", pad=4.0)

    # ---- (b) which stage owns the spread ----------------------------------
    pooled = [np.concatenate([bank[n][:, k] for n in names])
              for k in range(len(STAGES))]
    bp = ax_b.boxplot(pooled, orientation="horizontal", widths=0.58,
                      patch_artist=True, showfliers=False,
                      medianprops=dict(color="#1a1a1a", lw=1.0),
                      whiskerprops=dict(color=COLORS["rule"], lw=0.7),
                      capprops=dict(color=COLORS["rule"], lw=0.7),
                      boxprops=dict(lw=0.6, edgecolor="white"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
    ax_b.set_yticks(np.arange(1, len(STAGES) + 1))
    ax_b.set_yticklabels(STAGES, fontsize=6.5)
    ax_b.invert_yaxis()
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.25, 400.0)
    ax_b.set_xlabel("per-stage latency (ms)", fontsize=7.4)
    ax_b.tick_params(labelsize=6.5)
    ax_b.grid(axis="x", lw=0.45, alpha=0.5)
    ax_b.set_axisbelow(True)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    worst = int(np.argmax([np.percentile(p, 99) / np.mean(p)
                           for p in pooled]))
    ax_b.set_title("%s has the widest spread" % STAGES[worst], fontsize=8.5,
                   fontweight="bold", pad=3.5)

    # ---- (c) the tail the mean hides --------------------------------------
    q = np.linspace(0.50, 0.9995, 400)
    tail_base = ["#3b5b92", "#c9a227", "#2e7d5b"]
    for j, name in enumerate(TAIL_SHOW):
        i = names.index(name)
        col = ACCENT if over[i] else tail_base[j]
        ax_c.plot(np.quantile(totals[i], q), 1.0 - q, lw=1.3, color=col,
                  zorder=4, label=name)
        ax_c.plot([p99[i]], [0.01], marker="o", ms=3.8, mfc="white", mec=col,
                  mew=1.0, zorder=5)
    ax_c.axhline(0.01, color=COLORS["baseline"], lw=0.8, ls=(0, (2.2, 1.6)),
                 zorder=2)
    ax_c.text(0.6, 0.0118, "P99", fontsize=6.2, color=COLORS["muted"],
              ha="left", va="bottom")
    ax_c.axvline(SLA_MS, color=ACCENT, lw=1.0, ls=(0, (3.4, 1.8)), zorder=3)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(0.5, 900.0)
    ax_c.set_ylim(4e-4, 0.7)
    ax_c.set_xlabel("end-to-end latency (ms)", fontsize=7.4)
    ax_c.set_ylabel("P(latency > x) (-)", fontsize=7.4)
    ax_c.tick_params(labelsize=6.5)
    ax_c.grid(lw=0.45, alpha=0.45)
    ax_c.set_axisbelow(True)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)
    ax_c.legend(fontsize=6.2, frameon=False, loc="lower left",
                handlelength=1.4, borderpad=0.2, labelspacing=0.32)
    breach = totals[names.index(TAIL_SHOW[1])] > SLA_MS
    ax_c.set_title("SLA missed on %.1f %% of requests"
                   % (100.0 * breach.mean()), fontsize=8.5,
                   fontweight="bold", pad=3.5)

    for a, lab, dx in ((ax_a, "(a)", -0.98), (ax_b, "(b)", -0.62),
                       (ax_c, "(c)", -0.62)):
        panel_label_inches(fig, a, lab, dx=dx, dy=0.09)

    ratio = (p99 / mean_tot).max()

    savefig(fig, "643_edg_latency_budget.png")


if __name__ == "__main__":
    main()
