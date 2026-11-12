"""Bullwhip amplification across four echelons on one shared order axis.

Each tier forecasts with a short moving average and orders up to a
lead-time-scaled base stock, which is the textbook mechanism that turns a
mild consumer step into a swinging factory schedule.  The four stacked rows
therefore only mean something if they share a y range: the whole claim is
that the same physical quantity - orders in units per week - swings wider
the further upstream you look, and per-panel autoscaling would erase exactly
that.  The range is computed once over all five series and applied to every
row.  The auxiliary column quantifies the same effect as a variance
amplification ratio and as the response lag behind consumer demand.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_palette
from figkit import shared_limits, panel_labels_inches, axes_inches

RED = "#b23a48"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
BASE = "#8a8a8a"

N_WEEKS = 60
STEP_WEEK = 16
TIERS = ["retailer", "distributor", "manufacturer", "supplier"]
MA_WINDOW = 4          # forecast window (weeks)
LEAD = 2               # order lead time per echelon (weeks)


def echelon_orders(inflow):
    """Order-up-to response of one echelon to the demand it receives."""
    n = len(inflow)
    fc = np.empty(n)
    for t in range(n):
        lo = max(0, t - MA_WINDOW + 1)
        fc[t] = inflow[lo:t + 1].mean()
    base = (LEAD + 1.0) * fc
    out = inflow + np.diff(np.concatenate([[base[0]], base]))
    return np.clip(out, 0.0, None)


def main():
    set_pub_style()
    rng = np.random.default_rng(562)

    t = np.arange(N_WEEKS)
    consumer = 100.0 + 20.0 * (t >= STEP_WEEK) + rng.normal(0.0, 3.4, N_WEEKS)

    series, inflow = [], consumer
    for _ in TIERS:
        inflow = echelon_orders(inflow)
        series.append(inflow)

    var0 = consumer.var()
    amp = np.array([s.var() / var0 for s in series])
    peak_ratio = np.array([s.max() / consumer.mean() for s in series])
    colors = get_palette("cb_sequential_blue", 7)[3:7]

    # Inch-accurate placement: tight_layout cannot keep a four-row column and
    # a two-row column apart without either crushing the rows or opening a
    # gutter between the auxiliary panels.
    fig = plt.figure(figsize=(10.4, 6.4))
    row_h, row_gap, top = 1.15, 0.10, 5.62
    axes = [axes_inches(fig, (0.95, top - (i + 1) * row_h - i * row_gap,
                              5.35, row_h)) for i in range(4)]
    axa = axes_inches(fig, (7.00, 3.57, 3.10, 2.05))
    axl = axes_inches(fig, (7.00, 0.72, 3.10, 2.05))

    # One range for every echelon row - this is the entire point of the
    # figure, so it is computed over all five series before anything is drawn.
    lo, hi = shared_limits([consumer] + series, pad=0.08)

    for i, (ax, name, s, col) in enumerate(zip(axes, TIERS, series, colors)):
        ax.axvline(STEP_WEEK, color=RULE, lw=0.7, ls=":", zorder=2)
        ax.plot(t, consumer, color=BASE, lw=0.8, alpha=0.75, zorder=3,
                label="consumer demand" if i == 0 else None)
        ax.plot(t, s, color=col, lw=1.45, zorder=4,
                label="echelon orders" if i == 0 else None)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, N_WEEKS - 1)
        ax.tick_params(labelsize=6.8, labelbottom=(i == 3))
        ax.grid(axis="y", color="#eceae6", lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(0.010, 0.93, "%s   peak %.0f units/week   x%.1f variance"
                % (name, s.max(), amp[i]),
                transform=ax.transAxes, fontsize=7.0, color=col,
                fontweight="bold", ha="left", va="top", zorder=6)
    axes[3].set_xlabel("time (weeks)", fontsize=8.0)
    axes[0].text(STEP_WEEK + 0.8, lo + 0.07 * (hi - lo),
                 "consumer step +20 %", fontsize=6.6, color=RULE,
                 ha="left", va="bottom")
    axes[0].legend(fontsize=6.6, frameon=False, loc="lower right", ncol=2,
                   labelspacing=0.3, handlelength=1.5, columnspacing=1.1)
    axes[0].set_title("Downstream to upstream, one shared order axis",
                      fontsize=9.0, fontweight="bold", color=INK, pad=5)
    fig.text(0.014, 0.49, "order rate (units/week)", rotation=90,
             fontsize=8.0, color=INK, ha="center", va="center")

    # ------------------------------------------- (e) variance amplification
    ypos = np.arange(4)[::-1]
    axa.barh(ypos, amp, height=0.56, color=colors, edgecolor="white",
             linewidth=0.6, zorder=3)
    axa.axvline(1.0, color=RED, lw=1.0, ls="--", zorder=4,
                label="no amplification")
    for y, a in zip(ypos, amp):
        axa.text(a + 0.06 * amp.max(), y, "x%.1f" % a, fontsize=6.8,
                 color=INK, va="center", ha="left", zorder=5)
    axa.set_yticks(ypos)
    axa.set_yticklabels(TIERS, fontsize=6.8)
    axa.set_xlim(0, amp.max() * 1.30)
    axa.set_xlabel("order variance / consumer variance (-)", fontsize=8.0)
    axa.tick_params(labelsize=6.8)
    axa.legend(fontsize=6.5, frameon=False, loc="upper right",
               handlelength=1.4)
    axa.grid(axis="x", color="#eceae6", lw=0.5)
    axa.set_axisbelow(True)
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)
    axa.set_title("Variance amplification by echelon", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    # --------------------------------------------- (f) peak capacity call
    axl.barh(ypos, peak_ratio, height=0.56, color=colors, edgecolor="white",
             linewidth=0.6, zorder=3)
    axl.axvline(1.0, color=RED, lw=1.0, ls="--", zorder=4,
                label="mean consumer demand")
    for y, pr in zip(ypos, peak_ratio):
        axl.text(pr + 0.04 * peak_ratio.max(), y, "x%.2f" % pr, fontsize=6.8,
                 color=INK, va="center", ha="left", zorder=5)
    axl.set_yticks(ypos)
    axl.set_yticklabels(TIERS, fontsize=6.8)
    axl.set_xlim(0, peak_ratio.max() * 1.28)
    axl.set_xlabel("peak order / mean consumer demand (-)", fontsize=8.0)
    axl.tick_params(labelsize=6.8)
    axl.legend(fontsize=6.5, frameon=False, loc="upper right",
               handlelength=1.4)
    axl.grid(axis="x", color="#eceae6", lw=0.5)
    axl.set_axisbelow(True)
    for sp in ("top", "right"):
        axl.spines[sp].set_visible(False)
    axl.set_title("Upstream capacity must cover the peak", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    panel_labels_inches(fig, axes + [axa, axl], dx=-0.50, dy=0.05)
    savefig(fig, "562_scm_bullwhip_effect.png")


if __name__ == "__main__":
    main()
