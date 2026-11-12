"""EIP-1559 base fee chasing block utilisation through two congestion events.

The main panel is a closed loop rather than two independent curves: demand
sets block utilisation, the protocol moves the base fee by at most 12.5% per
block toward the 50% gas target, and the new fee feeds straight back into
demand.  Forty-four minutes of blocks are drawn as a step line so the
per-block adjustment is visible; the red bands are exactly the stretches
where utilisation stayed above the congestion threshold while the fee was
still catching up.

Auxiliary panels come off the same run: waiting time against the priority
bid with binned 10th/50th/90th percentile curves over the cloud, and the
split of what users actually paid into burned base fee, validator tip and
calldata.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, panel_labels_inches

W, H = 10.4, 6.4

BLUE = "#3b5b92"        # base fee
GREEN = "#2e7d5b"       # block utilisation
RED = "#b23a48"         # congestion / burned fee
GOLD = "#c9a227"        # validator tip
GREY = "#8a8a8a"        # calldata, secondary

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

N_BLOCKS = 220          # 44 min at 12 s per block
BLOCK_S = 12.0
TARGET = 0.50           # EIP-1559 gas target
MAX_STEP = 0.125        # protocol cap on the per-block base fee change
ELASTIC = 0.18          # how strongly demand backs off as the fee rises
CONGESTED = 0.92
BASE0 = 14.0

SHOCKS = [(7.5, 17.5, 1.70), (27.0, 36.0, 2.30)]    # start, end (min), size


def _ramp(t, c, k):
    return 1.0 / (1.0 + np.exp(-(t - c) / k))


def simulate(rng):
    t = np.arange(N_BLOCKS) * BLOCK_S / 60.0
    noise = rng.normal(0.0, 1.0, N_BLOCKS)
    for _ in range(14):
        noise = np.convolve(noise, np.ones(5) / 5.0, mode="same")
    noise /= np.abs(noise).max() or 1.0

    demand = 0.60 + 0.16 * noise
    for t0, t1, amp in SHOCKS:
        demand += amp * (_ramp(t, t0, 0.32) - _ramp(t, t1, 0.75))

    base = np.empty(N_BLOCKS)
    util = np.empty(N_BLOCKS)
    b = BASE0
    for i in range(N_BLOCKS):
        u = demand[i] / (1.0 + ELASTIC * (b / BASE0 - 1.0))
        util[i] = float(np.clip(u, 0.03, 1.0))
        base[i] = b
        b = max(1.0, b * (1.0 + MAX_STEP * (util[i] - TARGET) / TARGET))
    tip = 0.9 + 150.0 * np.clip(util - 0.50, 0.0, None) ** 1.5 + \
        rng.uniform(0.0, 1.5, N_BLOCKS)
    return t, base, util, tip


def congestion_spans(t, util, min_blocks=3):
    hot = util > CONGESTED
    spans, i = [], 0
    while i < hot.size:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < hot.size and hot[j]:
            j += 1
        if j - i >= min_blocks:
            spans.append((t[i], t[j - 1]))
        i = j
    return spans


def panel_main(ax, t, base, util, spans):
    for k, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color=RED, alpha=0.15, lw=0.0, zorder=0,
                   label=f"congested (> {100 * CONGESTED:.0f}% full)"
                   if k == 0 else None)

    ax.plot(t, base, lw=1.25, color=BLUE, zorder=4, drawstyle="steps-post",
            label="base fee (12.5% per-block cap)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0.0, base.max() * 1.30)
    ax.set_xlabel("time (min)", fontsize=8.0)
    ax.set_ylabel("EIP-1559 base fee (gwei)", fontsize=8.0, color=BLUE)
    ax.tick_params(axis="y", colors=BLUE, labelsize=7.2)
    ax.tick_params(axis="x", labelsize=7.2)

    axu = ax.twinx()
    axu.fill_between(t, 0.0, 100.0 * util, color=GREEN, alpha=0.20, lw=0.0,
                     zorder=1)
    axu.plot(t, 100.0 * util, lw=0.8, color=GREEN, alpha=0.85, zorder=3,
             label="block utilisation")
    axu.axhline(100.0 * TARGET, lw=0.9, ls="--", color=RULE, zorder=2)
    axu.text(t[-1], 100.0 * TARGET + 1.6, "50% gas target", ha="right",
             va="bottom", fontsize=6.6, color=RULE)
    axu.set_ylim(0.0, 232.0)
    axu.set_yticks([0, 25, 50, 75, 100])
    axu.set_ylabel("block utilisation (%)", fontsize=8.0, color=GREEN)
    axu.tick_params(axis="y", colors=GREEN, labelsize=7.2)
    ax.spines["top"].set_visible(False)
    axu.spines["top"].set_visible(False)
    axu.spines["right"].set_color(GREEN)
    ax.spines["left"].set_color(BLUE)
    axu.grid(False)

    pk = int(np.argmax(base))
    ax.annotate(f"peak {base[pk]:.0f} gwei",
                xy=(t[pk], base[pk]), xytext=(t[pk] - 3.4, base[pk] * 1.16),
                ha="right", fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=RULE,
                                shrinkA=1.0, shrinkB=2.0))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axu.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.0, ncol=1,
              loc="upper left", bbox_to_anchor=(0.015, 0.99))
    ax.set_title("The fee spike lags the queue it is meant to clear",
                 fontsize=9.0, fontweight="bold")
    return pk


def panel_wait(ax, rng):
    n = 520
    bid = np.exp(rng.normal(0.35, 0.85, n))          # priority fee, gwei
    wait = 1.0 + 17.0 * np.exp(-bid / 1.9) + rng.gamma(1.7, 0.95, n)
    ax.scatter(bid, wait, s=5.0, color=GREY, alpha=0.42, lw=0.0, zorder=2)

    edges = np.geomspace(bid.min(), bid.max(), 9)
    ctr, qs = [], {10: [], 50: [], 90: []}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (bid >= a) & (bid < b)
        if m.sum() < 6:
            continue
        ctr.append(np.sqrt(a * b))
        for q in qs:
            qs[q].append(np.percentile(wait[m], q))
    for q, (ls, c) in {10: (":", GREEN), 50: ("-", BLUE),
                       90: ("--", RED)}.items():
        ax.plot(ctr, qs[q], ls=ls, lw=1.2, color=c, zorder=4, label=f"p{q}")
    ax.set_xscale("log")
    ax.set_ylim(0.0, max(qs[90]) * 1.30)
    ax.set_xlabel("priority bid (gwei)", fontsize=8.0)
    ax.set_ylabel("inclusion delay (blocks)", fontsize=8.0)
    ax.set_title("Paying more buys inclusion", fontsize=8.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=6.6, ncol=3, loc="upper right",
              columnspacing=1.0, handlelength=1.6)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)
    return float(np.median(wait)), float(np.median(bid))


def panel_split(ax, t, base, tip, rng):
    edges = np.linspace(0.0, np.ceil(t[-1]), 5)
    ctr = np.arange(edges.size - 1)
    calldata = 3.5 + 2.4 * rng.random(ctr.size)
    b_h = np.array([base[(t >= a) & (t < b)].mean()
                    for a, b in zip(edges[:-1], edges[1:])])
    t_h = np.array([tip[(t >= a) & (t < b)].mean()
                    for a, b in zip(edges[:-1], edges[1:])])

    ax.bar(ctr, b_h, width=0.62, color=RED, edgecolor="white", linewidth=0.6,
           label="burned base fee")
    ax.bar(ctr, t_h, bottom=b_h, width=0.62, color=GOLD, edgecolor="white",
           linewidth=0.6, label="validator tip")
    ax.bar(ctr, calldata, bottom=b_h + t_h, width=0.62, color=GREY,
           edgecolor="white", linewidth=0.6, label="calldata")
    total = b_h + t_h + calldata
    burned = 100.0 * b_h.sum() / total.sum()
    ax.set_xticks(ctr)
    ax.set_xticklabels([f"{a:.0f}-{b:.0f}"
                        for a, b in zip(edges[:-1], edges[1:])])
    ax.set_xlabel("window within the run (min)", fontsize=8.0)
    ax.set_ylabel("mean cost per unit gas (gwei)", fontsize=8.0)
    ax.set_title("Most of the bill is burned", fontsize=8.5,
                 fontweight="bold")
    ax.set_ylim(0.0, total.max() * 1.42)
    ax.legend(frameon=False, fontsize=6.6, ncol=1, loc="upper left",
              handlelength=1.4, labelspacing=0.35)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)
    return burned


def main():
    set_pub_style()
    rng = np.random.default_rng(674)
    t, base, util, tip = simulate(rng)
    spans = congestion_spans(t, util)

    fig = plt.figure(figsize=(W, H))
    ax_m = axes_inches(fig, (0.80, 0.72, 5.78, 4.52))
    ax_w = axes_inches(fig, (7.72, 3.28, 2.34, 1.96))
    ax_s = axes_inches(fig, (7.72, 0.72, 2.34, 1.96))

    pk = panel_main(ax_m, t, base, util, spans)
    med_wait, med_bid = panel_wait(ax_w, rng)
    burned = panel_split(ax_s, t, base, tip, rng)

    hot = sum(b - a for a, b in spans)

    panel_labels_inches(fig, [ax_m], ["(a)"], dx=-0.60, dy=0.30)
    panel_labels_inches(fig, [ax_w, ax_s], ["(b)", "(c)"], dx=-0.60, dy=0.30)
    savefig(fig, "674_bch_gas_fee_dynamics.png")


if __name__ == "__main__":
    main()
