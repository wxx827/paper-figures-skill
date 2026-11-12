"""Continuous-review (s, S) inventory policy over one simulated season.

The main panel is the on-hand sawtooth: demand draws it down, a replenishment
order fires whenever the inventory position crosses the reorder point s, and
the receipt lifts stock back toward the order-up-to level S one lead time
later.  Days that finish below zero are backorders and are filled in house
red, which is the only place the figure uses that colour.  The two auxiliary
panels, held to ~58 % of the main width, show why the safety stock is where
it is: the service-level / holding-cost trade-off, and what the buffer is
actually made of for each ABC class.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"

T_DAYS = 120
S_LOW = 500.0      # reorder point s (units)
S_HIGH = 1500.0    # order-up-to level S (units)
UNIT_HOLD = 0.62   # USD per unit per day
SURGE = (58, 68)   # promotion window (days)


def simulate(rng):
    """Daily (s, S) run; returns on-hand, orders placed and receipts."""
    demand = rng.gamma(6.0, 11.5, T_DAYS)
    # A ten-day promotion doubles offtake; the policy parameters were tuned
    # for the base rate, so this is where the buffer runs out.
    demand[SURGE[0]:SURGE[1]] *= 1.95
    on_hand = np.zeros(T_DAYS)
    pos = S_HIGH
    stock = S_HIGH
    pipeline = {}
    placed, received = [], []
    for t in range(T_DAYS):
        if t in pipeline:
            stock += pipeline[t]
            received.append((t, pipeline.pop(t)))
        stock -= demand[t]
        pos -= demand[t]
        if pos <= S_LOW and not pipeline:
            qty = S_HIGH - pos
            lead = int(rng.integers(5, 10))
            pipeline[min(t + lead, T_DAYS - 1)] = qty
            pos += qty
            placed.append((t, lead, qty))
        on_hand[t] = stock
    return demand, on_hand, placed, received


def main():
    set_pub_style()
    rng = np.random.default_rng(559)
    demand, on_hand, placed, received = simulate(rng)
    t = np.arange(T_DAYS)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.90],
                          wspace=0.22, hspace=0.45)
    axm = fig.add_subplot(gs[:, 0])
    axs = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------ (a) sawtooth
    axm.axvspan(SURGE[0], SURGE[1], color="#f0f0f0", zorder=0)
    axm.text(0.5 * (SURGE[0] + SURGE[1]), S_HIGH * 1.02, "promotion",
             fontsize=6.6, color=MUTED, ha="center", va="bottom", zorder=2)
    axm.axhline(0.0, color="#808080", lw=0.8, zorder=2)
    axm.fill_between(t, np.minimum(on_hand, 0.0), 0.0, color=RED, alpha=0.30,
                     linewidth=0.0, zorder=3)
    axm.plot(t, on_hand, color=BLUE, lw=1.35, zorder=5,
             label="on-hand inventory")
    axm.axhline(S_HIGH, color=GREEN, lw=1.0, ls="--", zorder=4)
    axm.axhline(S_LOW, color=GOLD, lw=1.0, ls="--", zorder=4)
    label_box = dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                     alpha=0.88)
    axm.text(2.0, S_HIGH + 26, "order-up-to level S = %.0f units" % S_HIGH,
             fontsize=6.8, color=GREEN, ha="left", va="bottom",
             bbox=label_box, zorder=6)
    axm.text(2.0, S_LOW + 26, "reorder point s = %.0f units" % S_LOW,
             fontsize=6.8, color=GOLD, ha="left", va="bottom",
             bbox=label_box, zorder=6)

    for k, (t0, lead, qty) in enumerate(placed):
        axm.plot([t0], [on_hand[t0]], marker="v", ms=4.2, mfc=GOLD,
                 mec="white", mew=0.6, zorder=7)
        if k == 1:
            y = S_HIGH * 1.10
            for tx in (t0, t0 + lead):
                axm.plot([tx, tx], [on_hand[tx], y], color=RULE, lw=0.6,
                         ls=(0, (2.2, 1.8)), zorder=6)
            axm.annotate("", xy=(t0 + lead, y), xytext=(t0, y),
                         arrowprops=dict(arrowstyle="<|-|>", lw=0.9,
                                         color=RULE, mutation_scale=8),
                         zorder=8)
            axm.text(t0 + lead / 2.0, y + 34,
                     "replenishment lead time  L = %d days" % lead,
                     fontsize=6.8, color=RULE, ha="center", va="bottom",
                     zorder=8)
    for t1, qty in received:
        axm.plot([t1], [on_hand[t1]], marker="^", ms=4.2, mfc=GREEN,
                 mec="white", mew=0.6, zorder=7)

    short = np.clip(-on_hand, 0.0, None)
    stockout_days = int((on_hand < 0).sum())
    fill_rate = 100.0 * (1.0 - short.sum() / demand.sum())
    worst = int(np.argmin(on_hand))
    axm.annotate("stockout: %d days, %.0f units backordered"
                 % (stockout_days, short.sum()),
                 xy=(worst, on_hand[worst]),
                 xytext=(SURGE[0] - 3.0, -140.0),
                 fontsize=6.8, color=RED, ha="right", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=RED,
                                 shrinkA=1, shrinkB=3), zorder=8)

    axm.set_xlim(0, T_DAYS - 1)
    axm.set_ylim(min(on_hand.min() - 130.0, -210.0), S_HIGH * 1.30)
    axm.set_xlabel("time (days)", fontsize=8.0)
    axm.set_ylabel("inventory position (units)", fontsize=8.0)
    axm.tick_params(labelsize=7.0)
    axm.grid(axis="y", color="#e9e7e2", lw=0.55)
    axm.set_axisbelow(True)
    for sp in ("top", "right"):
        axm.spines[sp].set_visible(False)
    axm.text(0.988, 0.982,
             "mean daily demand %.0f units\nfill rate %.1f%%\n"
             "average on-hand %.0f units"
             % (demand.mean(), fill_rate, np.clip(on_hand, 0, None).mean()),
             transform=axm.transAxes, fontsize=6.8, color=INK, ha="right",
             va="top", linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#cccccc",
                       lw=0.6, alpha=0.92))
    axm.set_title("On-hand trajectory under the (s, S) policy", fontsize=9.0,
                  fontweight="bold", color=INK, pad=5)

    # ------------------------------------- (b) service level vs cost
    z = np.linspace(0.0, 3.2, 160)
    mean_lead = float(np.mean([p[1] for p in placed]))
    sigma_l = float(demand.std() * np.sqrt(mean_lead))
    service = 100.0 * norm.cdf(z)
    cycle_stock = 0.5 * float(np.mean([p[2] for p in placed]))
    holding = UNIT_HOLD * (cycle_stock + z * sigma_l) * 7.0 / 1000.0
    axs.plot(service, holding, color=BLUE, lw=1.6, zorder=3)
    for target in (90.0, 95.0, 99.0):
        zt = norm.ppf(target / 100.0)
        ct = UNIT_HOLD * (cycle_stock + zt * sigma_l) * 7.0 / 1000.0
        axs.plot([target], [ct], marker="o", ms=4.2,
                 mfc=RED if target == 95.0 else "white", mec=RED, mew=1.0,
                 zorder=5)
        axs.annotate("%.0f%% -> %.2f kUSD/week" % (target, ct),
                     xy=(target, ct), xytext=(target - 1.6, ct + 0.16),
                     fontsize=6.5, color=INK, ha="right", va="bottom")
    axs.set_xlim(50, 100.3)
    axs.set_xlabel("cycle service level (%)", fontsize=8.0)
    axs.set_ylabel("holding cost (kUSD/week)", fontsize=8.0)
    axs.tick_params(labelsize=6.8)
    axs.grid(color="#e9e7e2", lw=0.55)
    axs.set_axisbelow(True)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)
    axs.set_title("Service level costs more than linearly", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    # ------------------------------------- (c) safety-stock decomposition
    classes = ["class A", "class B", "class C"]
    demand_var = np.array([196.0, 128.0, 74.0])
    lead_var = np.array([142.0, 71.0, 30.0])
    forecast_bias = np.array([58.0, 44.0, 26.0])
    parts = [("demand variability", demand_var, BLUE),
             ("lead-time variability", lead_var, GREEN),
             ("forecast bias buffer", forecast_bias, PURPLE)]
    ypos = np.arange(len(classes))[::-1]
    left = np.zeros(len(classes))
    for name, vals, col in parts:
        axb.barh(ypos, vals, left=left, height=0.52, color=col,
                 edgecolor="white", linewidth=0.6, label=name, zorder=3)
        left = left + vals
    for y, tot in zip(ypos, left):
        axb.text(tot + 12.0, y, "%.0f units" % tot, fontsize=6.6, color=INK,
                 va="center", ha="left", zorder=4)
    axb.set_yticks(ypos)
    axb.set_yticklabels(classes, fontsize=6.8)
    axb.set_xlim(0, left.max() * 1.42)
    axb.set_xlabel("safety stock (units)", fontsize=8.0)
    axb.tick_params(labelsize=6.8)
    axb.legend(fontsize=6.5, frameon=False, loc="lower right",
               labelspacing=0.32, handlelength=1.4)
    axb.grid(axis="x", color="#e9e7e2", lw=0.55)
    axb.set_axisbelow(True)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.set_title("What the safety stock is made of", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [axm, axs, axb], dx=-0.42, dy=0.14)
    savefig(fig, "559_scm_inventory_policy.png")


if __name__ == "__main__":
    main()
