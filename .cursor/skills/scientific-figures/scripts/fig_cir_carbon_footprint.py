"""Product carbon footprint: where it comes from and what it costs to remove.

The main panel is a waterfall, the only chart type that shows a total and its
components at the same time, with the end-of-life recycling credit drawn as a
genuine downward step rather than a smaller positive bar.  A cumulative line
runs through the top of every step so the reader can follow the running
balance instead of adding the labels up.  The auxiliary panels answer the two
questions a waterfall always provokes: how does this total compare, and what
would it cost to cut it.

Synthetic data.  The net footprint, every share, the abatement totals and the
weighted average abatement cost are all recomputed from the two tables at the
top of the file.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, darken
from figkit import panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
UP = "#a1907c"            # emitting stage, warm neutral
HOT = "#c0603f"           # the largest emitting stage
DOWN = "#2e7d5b"          # recycling credit / negative-cost measure
TOTAL = "#3a3a3a"
ACCENT = "#b23a48"

# life-cycle step, kg CO2-eq per functional unit
STEPS = [("Raw materials", 8.40), ("Component manufacture", 5.20),
         ("Assembly", 2.10), ("Logistics", 1.60),
         ("Use phase, 8 years", 11.30), ("End-of-life treatment", 0.90),
         ("Recycling credit", -4.70)]
# comparator, kg CO2-eq per functional unit
PEERS = [("Market incumbent", 31.5), ("Sector average", 28.4),
         ("Best in class", 19.2)]
# abatement measure, potential (kt CO2-eq/a), marginal cost (USD/t CO2-eq)
MACC = [("LED and controls retrofit", 3.2, -48.0),
        ("Variable-speed drives", 2.4, -22.0),
        ("Waste heat recovery", 4.1, -6.0),
        ("Recycled feedstock share", 5.6, 14.0),
        ("Rail modal shift", 2.9, 38.0),
        ("On-site solar PPA", 6.3, 55.0),
        ("Green hydrogen process heat", 3.4, 128.0)]
CARBON_PRICE = 65.0       # USD/t CO2-eq


def main():
    set_pub_style(base_font=9)

    labels = [s[0] for s in STEPS]
    vals = np.array([s[1] for s in STEPS])
    cum = np.cumsum(vals)
    starts = cum - vals
    net = float(cum[-1])
    gross = float(vals[vals > 0].sum())
    credit = float(-vals[vals < 0].sum())

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0],
                          left=0.062, right=0.975, top=0.845, bottom=0.185,
                          wspace=0.28, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
    ax_cmp = fig.add_subplot(gs[0, 1])
    ax_mac = fig.add_subplot(gs[1, 1])

    # ---- (a) waterfall -----------------------------------------------------
    hot = int(np.argmax(vals))
    x = np.arange(len(STEPS) + 1)
    for i, v in enumerate(vals):
        col = DOWN if v < 0 else (HOT if i == hot else UP)
        ax.bar(i, v, bottom=starts[i], width=0.60, color=col,
               edgecolor="white", linewidth=0.6, zorder=3)
        top = starts[i] + max(v, 0.0)
        ax.text(i, top + 0.55, f"{v:+.1f}", ha="center", va="bottom",
                fontsize=6.8, fontweight="bold", color=darken(col, 0.35),
                zorder=6)
        if i < len(vals) - 1:
            ax.plot([i + 0.30, i + 1 - 0.30], [cum[i]] * 2, lw=0.7,
                    ls=(0, (3, 2)), color="#9a9a9a", zorder=2)
    ax.bar(len(STEPS), net, width=0.60, color=TOTAL, edgecolor="white",
           linewidth=0.6, zorder=3)
    ax.text(len(STEPS), net + 0.55, f"{net:.1f}", ha="center", va="bottom",
            fontsize=7.4, fontweight="bold", color=TOTAL, zorder=6)

    ax.plot(np.arange(len(STEPS)), cum, lw=1.1, color=RULE, marker="D",
            ms=3.2, mfc="white", mec=RULE, mew=0.9, zorder=7,
            label="Cumulative footprint")
    ax.plot([len(STEPS) - 1, len(STEPS)], [cum[-1], net], lw=1.1, color=RULE,
            zorder=7)
    ax.plot([len(STEPS)], [net], marker="D", ms=3.6, mfc=RULE, mec="white",
            mew=0.8, zorder=8)

    ax.axhline(0.0, color=RULE, lw=1.0, zorder=5)
    ax.set_xlim(-0.66, len(STEPS) + 0.66)
    ax.set_ylim(-1.5, max(cum.max(), net) * 1.30)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(l, 11) for l in labels] +
                       ["Net\nfootprint"], fontsize=6.6, linespacing=1.4)
    ax.set_ylabel("Carbon footprint (kg $CO_2$-eq per unit)")
    ax.set_xlabel("Life-cycle step, cradle to grave")
    ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.005), frameon=False,
              fontsize=6.8, handlelength=1.8)
    ax.set_title("Waterfall of a single functional unit, cradle to grave",
                 fontsize=9.0, fontweight="bold", loc="left", pad=5)
    ax.text(0.985, 0.965,
            f"gross emissions {gross:.1f} kg $CO_2$-eq\n"
            f"recycling credit -{credit:.1f} kg $CO_2$-eq "
            f"({100.0 * credit / gross:.0f}% of gross)\n"
            f"largest step: {labels[hot].lower()}, "
            f"{100.0 * vals[hot] / gross:.0f}% of gross",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=SUB, linespacing=1.6)

    # ---- (b) benchmark comparison -----------------------------------------
    bars = [("This product", net, ACCENT)] + \
           [(n, v, "#b3b3b3") for n, v in PEERS]
    bars.sort(key=lambda b: b[1])
    yb = np.arange(len(bars))[::-1]
    for k, (n, v, c) in enumerate(bars):
        ax_cmp.barh(yb[k], v, height=0.60, color=c, edgecolor="white",
                    linewidth=0.5, zorder=3)
        ax_cmp.text(v + 0.6, yb[k], f"{v:.1f}", va="center", ha="left",
                    fontsize=6.4, color=ACCENT if n == "This product" else SUB)
        if n != "This product":
            ax_cmp.text(0.99, yb[k], f"{100.0 * (v - net) / net:+.0f}%",
                        transform=ax_cmp.get_yaxis_transform(), va="center",
                        ha="right", fontsize=6.4, color=SUB)
    ax_cmp.axvline(net, color=ACCENT, lw=0.9, ls=(0, (3, 2)), zorder=5)
    ax_cmp.set_xlim(0.0, max(v for _, v, _ in bars) * 1.28)
    ax_cmp.set_ylim(-0.7, len(bars) - 0.3)
    ax_cmp.set_yticks(yb)
    ax_cmp.set_yticklabels([b[0] for b in bars], fontsize=6.4)
    ax_cmp.set_xlabel("Carbon footprint (kg $CO_2$-eq per unit)", fontsize=8.0)
    ax_cmp.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax_cmp)
    ax_cmp.tick_params(axis="x", labelsize=6.8)
    ax_cmp.tick_params(axis="y", length=0)
    ax_cmp.spines["left"].set_visible(False)
    ax_cmp.set_title("Benchmark against comparable products", fontsize=8.5,
                     fontweight="bold", loc="left", pad=4)

    # ---- (c) marginal abatement cost curve --------------------------------
    order = sorted(MACC, key=lambda m: m[2])
    widths = np.array([m[1] for m in order])
    costs = np.array([m[2] for m in order])
    left = np.concatenate([[0.0], np.cumsum(widths)[:-1]])
    total_ab = float(widths.sum())
    viable = costs <= CARBON_PRICE
    ab_viable = float(widths[viable].sum())
    wavg = float(np.sum(widths * costs) / total_ab)

    for l, w, c in zip(left, widths, costs):
        col = DOWN if c < 0 else (UP if c <= CARBON_PRICE else HOT)
        ax_mac.bar(l + w / 2.0, c, width=w * 0.985, color=col,
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax_mac.axhline(0.0, color=RULE, lw=1.0, zorder=5)
    ax_mac.axhline(CARBON_PRICE, color=ACCENT, lw=1.0, ls=(0, (4, 2)),
                   zorder=5)
    ax_mac.text(total_ab * 0.012, CARBON_PRICE + 5.0,
                f"carbon price {CARBON_PRICE:.0f} USD/t $CO_2$", ha="left",
                va="bottom", fontsize=6.4, color=ACCENT)
    ax_mac.text(left[0] + widths[0] / 2.0, costs[0] - 6.0,
                f"cheapest: {order[0][0]}, {costs[0]:+.0f} USD/t",
                ha="left", va="top", fontsize=6.2, color=darken(DOWN, 0.25))
    ax_mac.set_xlim(0.0, total_ab)
    ax_mac.set_ylim(costs.min() * 1.90, costs.max() * 1.32)
    ax_mac.set_xlabel("Cumulative abatement (kt $CO_2$-eq per year)",
                      fontsize=8.0)
    ax_mac.set_ylabel("Marginal cost\n(USD/t $CO_2$-eq)", fontsize=8.0,
                      linespacing=1.4)
    ax_mac.tick_params(labelsize=6.8)
    ax_mac.grid(axis="y", color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax_mac)
    ax_mac.set_title("Marginal abatement cost curve, site level", fontsize=8.5,
                     fontweight="bold", loc="left", pad=4)
    ax_mac.text(0.015, 0.965,
                f"{ab_viable:.1f} of {total_ab:.1f} kt/a clears the carbon "
                f"price\nvolume-weighted average {wavg:+.0f} USD/t $CO_2$-eq",
                transform=ax_mac.transAxes, ha="left", va="top",
                fontsize=6.4, color=SUB, linespacing=1.55)

    panel_labels_inches(fig, [ax, ax_cmp, ax_mac], dx=-0.46, dy=0.09)

    best = bars[0]

    print(f"gross={gross:.2f}  credit={credit:.2f}  net={net:.2f} kg CO2-eq")
    print(f"abatement total={total_ab:.1f} kt  viable={ab_viable:.1f} kt  "
          f"weighted avg cost={wavg:.1f} USD/t")
    savefig(fig, "629_cir_carbon_footprint.png")


if __name__ == "__main__":
    main()
