"""Disruption response and recovery for three supply-chain resilience plans.

The main panel is the classic resilience triangle: service holds at the
nominal rate, a tier-1 plant outage knocks it down, and each mitigation plan
climbs back at its own pace.  Time-to-recover is measured off the curve
rather than asserted, and the lost service integrated between the nominal
line and the unmitigated response is filled in house red so the cost of
doing nothing has an area, not just a number.  The auxiliary column places
the same disruption in the portfolio: a probability-by-impact risk matrix
and a criticality ranking of the nodes that would cause it.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, lighten
from figkit import panel_labels_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"

T_END = 70
T_HIT = 10.0
NOMINAL = 100.0     # % of planned service rate

PLANS = [
    ("no mitigation", BLUE, 0.34, 16.0),
    ("+ safety stock", GOLD, 0.58, 11.0),
    ("+ dual sourcing", GREEN, 0.79, 6.5),
]


def response(t, floor, tau):
    """Service rate over time: flat, drop at T_HIT, exponential recovery."""
    y = np.full_like(t, NOMINAL, dtype=float)
    after = t >= T_HIT
    dt = t[after] - T_HIT
    y[after] = NOMINAL * (floor + (1.0 - floor) * (1.0 - np.exp(-dt / tau)))
    return y


def main():
    set_pub_style()
    rng = np.random.default_rng(563)
    t = np.linspace(0.0, T_END, 561)

    curves, ttr, loss = [], [], []
    for name, col, floor, tau in PLANS:
        clean = response(t, floor, tau)
        curves.append(np.clip(clean + rng.normal(0.0, 0.55, t.size),
                              0.0, NOMINAL + 1.2))
        # Metrics come off the noise-free response so a lucky sample cannot
        # declare the chain recovered a week early.
        ok = np.where((t > T_HIT) & (clean >= 0.98 * NOMINAL))[0]
        ttr.append(t[ok[0]] - T_HIT if ok.size else np.nan)
        loss.append(np.trapezoid(np.clip(NOMINAL - clean, 0.0, None), t))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.92],
                          wspace=0.22, hspace=0.46)
    axm = fig.add_subplot(gs[:, 0])
    axr = fig.add_subplot(gs[0, 1])
    axv = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------ (a) recovery curves
    axm.axhline(NOMINAL, color="#808080", lw=0.9, ls="--", zorder=3)
    axm.text(T_END - 0.6, NOMINAL + 1.0, "nominal service rate",
             fontsize=6.8, color="#555555", ha="right", va="bottom")
    axm.fill_between(t, curves[0], NOMINAL, where=curves[0] < NOMINAL,
                     color=RED, alpha=0.22, linewidth=0.0, zorder=2)
    axm.axvline(T_HIT, color=RED, lw=1.0, zorder=4)
    axm.text(T_HIT + 0.7, 14.0, "tier-1 plant outage", fontsize=6.8,
             color=RED, ha="left", va="center", rotation=90)

    for (name, col, floor, tau), y, r, ls_ in zip(PLANS, curves, ttr, loss):
        axm.plot(t, y, color=col, lw=1.5, zorder=5,
                 label="%s  (TTR %.0f days)" % (name, r))
        axm.plot([T_HIT + r], [np.interp(T_HIT + r, t, y)], marker="o",
                 ms=4.0, mfc="white", mec=col, mew=1.1, zorder=7)

    y_arrow = 30.0
    axm.annotate("", xy=(T_HIT + ttr[0], y_arrow), xytext=(T_HIT, y_arrow),
                 arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color=BLUE,
                                 mutation_scale=8), zorder=8)
    axm.text(T_HIT + ttr[0] / 2.0, y_arrow + 2.0,
             "TTR without mitigation = %.0f days" % ttr[0], fontsize=6.8,
             color=BLUE, ha="center", va="bottom", zorder=8)
    axm.annotate("lost service = %.0f %%-days\n(%.0f %%-days avoided by "
                 "dual sourcing)" % (loss[0], loss[0] - loss[2]),
                 xy=(T_HIT + 6.0, 62.0), xytext=(T_HIT + 21.0, 46.0),
                 fontsize=6.8, color=RED, ha="left", va="top",
                 linespacing=1.25, zorder=8,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=RED,
                                 shrinkA=2, shrinkB=3))

    axm.set_xlim(0, T_END)
    axm.set_ylim(0, 116)
    axm.set_xlabel("time since disruption window opened (days)", fontsize=8.0)
    axm.set_ylabel("order fill rate (% of plan)", fontsize=8.0)
    axm.tick_params(labelsize=7.0)
    axm.grid(axis="y", color="#e9e7e2", lw=0.55)
    axm.set_axisbelow(True)
    for sp in ("top", "right"):
        axm.spines[sp].set_visible(False)
    axm.legend(fontsize=6.8, frameon=False, loc="lower right",
               labelspacing=0.35, handlelength=1.7)
    axm.set_title("Recovery paths under three resilience plans",
                  fontsize=9.0, fontweight="bold", color=INK, pad=5)

    # --------------------------------------------------- (b) risk matrix
    risks = [("tier-1 plant outage", 22.0, 780.0, 2.0, 26.0, "left"),
             ("single-source chip", 12.0, 910.0, 2.0, 26.0, "left"),
             ("quality recall", 9.0, 520.0, 2.0, 26.0, "left"),
             ("port congestion", 47.0, 300.0, 0.0, 32.0, "center"),
             ("carrier strike", 33.0, 165.0, -2.0, 0.0, "right"),
             ("demand spike", 60.0, 120.0, -2.0, 14.0, "right")]
    names = [r[0] for r in risks]
    prob = np.array([r[1] for r in risks])
    impact = np.array([r[2] for r in risks])
    expo = prob / 100.0 * impact
    p_thr, i_thr = 30.0, 400.0

    axr.add_patch(plt.Rectangle((p_thr, i_thr), 70.0 - p_thr, 1050.0 - i_thr,
                                facecolor=lighten(RED, 0.86),
                                edgecolor="none", zorder=0))
    axr.axvline(p_thr, color="#bbbbbb", lw=0.8, ls="--", zorder=1)
    axr.axhline(i_thr, color="#bbbbbb", lw=0.8, ls="--", zorder=1)
    axr.text(69.0, 1030.0, "escalate", fontsize=6.5, color=RED, ha="right",
             va="top", zorder=2)
    axr.scatter(prob, impact, s=16.0 + 0.10 * expo, facecolor=RED,
                edgecolor="white", linewidth=0.6, zorder=4)
    for nm, p, im, dx, dy, ha in risks:
        axr.text(p + dx, im + dy, nm, fontsize=6.3, color=INK, ha=ha,
                 va="bottom" if dy >= 0 else "top", zorder=5)
    top = int(np.argmax(expo))
    axr.text(0.015, 0.035, "highest exposure:\n%s, %.0f kUSD/yr"
             % (names[top], expo[top]), transform=axr.transAxes,
             fontsize=6.5, color=RED, ha="left", va="bottom", zorder=6,
             linespacing=1.25)
    axr.set_xlim(0, 70)
    axr.set_ylim(0, 1050)
    axr.set_xlabel("annual probability (%)", fontsize=8.0)
    axr.set_ylabel("impact (kUSD)", fontsize=8.0)
    axr.tick_params(labelsize=6.8)
    for sp in ("top", "right"):
        axr.spines[sp].set_visible(False)
    axr.set_title("Risk matrix, marker area = expected loss", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    # -------------------------------------------- (c) node vulnerability
    nodes = ["tier-1 plant P2", "port hub", "chip supplier", "DC3",
             "carrier fleet", "DC1"]
    score = np.array([0.92, 0.74, 0.68, 0.41, 0.33, 0.21])
    ypos = np.arange(len(nodes))[::-1]
    cols = [RED if i == 0 else "#9aa9c0" for i in range(len(nodes))]
    axv.barh(ypos, score, height=0.55, color=cols, edgecolor="white",
             linewidth=0.6, zorder=3)
    for y, s in zip(ypos, score):
        axv.text(s + 0.015, y, "%.2f" % s, fontsize=6.7, color=INK,
                 va="center", ha="left", zorder=4)
    axv.set_yticks(ypos)
    axv.set_yticklabels(nodes, fontsize=6.8)
    axv.set_xlim(0, 1.12)
    axv.set_xlabel("criticality score (-)", fontsize=8.0)
    axv.tick_params(labelsize=6.8)
    axv.grid(axis="x", color="#e9e7e2", lw=0.55)
    axv.set_axisbelow(True)
    for sp in ("top", "right"):
        axv.spines[sp].set_visible(False)
    axv.set_title("Which node hurts most if it fails", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [axm, axr, axv], dx=-0.44, dy=0.14)
    savefig(fig, "563_scm_resilience_disruption.png")


if __name__ == "__main__":
    main()
