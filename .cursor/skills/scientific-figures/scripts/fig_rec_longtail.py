"""Figure 474 - Long-tail exposure under a recommender feedback loop.

Main panel: items sorted by popularity on a log-log exposure curve, with the
catalogue split into head / torso / tail bands.  Two curves share the panel -
the demand-matched exposure a catalogue "deserves" and the exposure a
popularity-amplifying ranker actually serves - and the gap between them in the
tail is filled, because that gap is the Matthew effect.  Right column: the
Lorenz curve behind the Gini coefficient, the catalogue share needed to cover
50 / 80 / 95 % of impressions, and a metric card.

One synthetic catalogue is drawn; every number printed on the figure (band
shares, Gini, coverage, never-shown share) is a reduction of that sample.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, inch_axes, panel_label_inches, text_card

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID = "#cccccc", "#d9d9d9"

SERVED = "#3b5b92"      # what the ranker actually shows
DEMAND = "#808080"      # what interest alone would give
DEFICIT = "#b23a48"     # the single emphasis: tail exposure lost

N_ITEMS = 20000
TOTAL = 4.2e8           # impressions served over the window
HEAD_END, TORSO_END = 200, 4000     # 1 % and 20 % of the catalogue


def gini(x):
    """Gini coefficient of a non-negative exposure vector."""
    s = np.sort(np.asarray(x, float))
    n = s.size
    c = np.cumsum(s)
    return float((n + 1 - 2.0 * (c / c[-1]).sum()) / n)


def lorenz(x):
    """Cumulative item share vs cumulative impression share, poorest first."""
    s = np.sort(np.asarray(x, float))
    c = np.concatenate([[0.0], np.cumsum(s) / s.sum()])
    return np.linspace(0.0, 1.0, c.size), c


def coverage_needed(x, frac):
    """Catalogue share (%) whose impressions add up to ``frac`` of the total."""
    s = np.sort(np.asarray(x, float))[::-1]
    c = np.cumsum(s) / s.sum()
    return 100.0 * (np.searchsorted(c, frac) + 1) / s.size


def simulate(rng):
    """Demand-matched and feedback-amplified exposure for one catalogue."""
    rank = np.arange(1, N_ITEMS + 1)
    base = rank ** -0.82 * np.exp(rng.normal(0.0, 0.34, N_ITEMS))
    demand = np.sort(base)[::-1]
    demand = demand / demand.sum() * TOTAL
    # Popularity amplification (gamma > 1) plus candidate-generation truncation:
    # items past the retrieval horizon are never scored at all.
    served = demand ** 1.42 / (1.0 + (rank / 4800.0) ** 4.8)
    served = served / served.sum() * TOTAL
    return rank, demand, served


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(474)
    rank, demand, served = simulate(rng)

    cuts = [(1, HEAD_END, "head"), (HEAD_END, TORSO_END, "torso"),
            (TORSO_END, N_ITEMS, "tail")]
    shares = [(served[a - 1:b].sum() / TOTAL, demand[a - 1:b].sum() / TOTAL)
              for a, b, _ in cuts]
    g_served, g_demand = gini(served), gini(demand)
    never = 100.0 * float((served < 1.0).mean())

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.72, 0.66, 5.15, 4.80))

    # ---- (a) exposure curve ------------------------------------------------
    for (a, b, name), (fs, fd), tone in zip(cuts, shares,
                                            ("#f0f0f0", "#f5f5f5", "#f0f0f0")):
        ax.axvspan(a, b, facecolor=tone, edgecolor="none", zorder=0)
        ax.text(np.sqrt(a * b), 0.985, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.2, color=SUB,
                fontweight="bold")
        ax.text(np.sqrt(a * b), 0.935,
                f"{100 * fs:.1f} % served\n{100 * fd:.1f} % demand",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.6, color=MUTED, linespacing=1.35)

    lack = served < demand
    ax.fill_between(rank, served, demand, where=lack, interpolate=True,
                    facecolor=lighten(DEFICIT, 0.76), edgecolor="none",
                    zorder=2, label="tail exposure deficit")
    ax.plot(rank, demand, lw=1.3, color=DEMAND, ls=(0, (4.0, 2.0)), zorder=4,
            label="demand-matched exposure")
    ax.plot(rank, served, lw=1.7, color=SERVED, zorder=5,
            label="served exposure (feedback loop)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, N_ITEMS)
    ax.set_ylim(max(served[served > 0].min() * 0.35, 1.0), demand[0] * 6.0)
    ax.set_xlabel("item rank by popularity (-)", fontsize=8.0)
    ax.set_ylabel("impressions per item (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(which="major", color=GRID, linewidth=0.6)
    ax.grid(which="minor", color=GRID, linewidth=0.35, alpha=0.55)
    ax.set_title("Exposure per item, catalogue sorted by popularity",
                 fontsize=9.0, fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.0, frameon=False,
              handlelength=1.6, borderaxespad=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    lost = float(demand[lack].sum() - served[lack].sum()) / TOTAL
    ax.annotate(f"{100 * lost:.1f} % of all impressions\nre-routed out of the tail",
                xy=(9000, served[8999]), xycoords="data",
                xytext=(1300, served[8999] * 0.10), textcoords="data",
                fontsize=6.8, color=INK, ha="left", va="center",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-|>", color=DEFICIT, lw=0.9,
                                shrinkA=2, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                          edgecolor=HAIR, linewidth=0.6, alpha=0.94))

    # ---- (b) Lorenz curve --------------------------------------------------
    axl = axes_inches(fig, (7.05, 3.16, 2.85, 1.72))
    xs, cs = lorenz(served)
    xd, cd = lorenz(demand)
    axl.fill_between(xs, xs, cs, facecolor=lighten(SERVED, 0.80),
                     edgecolor="none", zorder=2)
    axl.plot([0, 1], [0, 1], lw=0.9, color=FAINT, ls=(0, (3.0, 2.0)), zorder=3)
    axl.plot(xd, cd, lw=1.2, color=DEMAND, ls=(0, (4.0, 2.0)), zorder=4)
    axl.plot(xs, cs, lw=1.5, color=SERVED, zorder=5)
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.set_xticks([0, 0.5, 1.0])
    axl.set_yticks([0, 0.5, 1.0])
    axl.set_xlabel("cumulative catalogue share (-)", fontsize=7.0)
    axl.set_ylabel("cumulative\nimpressions (-)", fontsize=7.0, linespacing=1.2)
    axl.tick_params(labelsize=6.5, length=2.2)
    axl.set_title("Lorenz curve of exposure", fontsize=8.5, fontweight="bold",
                  pad=3.0)
    axl.text(0.055, 0.90, f"Gini {g_served:.3f}", fontsize=7.0, color=SERVED,
             fontweight="bold", ha="left", va="top")
    axl.text(0.055, 0.75, f"demand {g_demand:.3f}", fontsize=6.6, color=SUB,
             ha="left", va="top")
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)

    # ---- (c) catalogue share needed ---------------------------------------
    axc = axes_inches(fig, (7.05, 0.66, 2.85, 1.44))
    fracs = (0.50, 0.80, 0.95)
    cov_s = [coverage_needed(served, f) for f in fracs]
    cov_d = [coverage_needed(demand, f) for f in fracs]
    xpos = np.arange(len(fracs))
    axc.bar(xpos - 0.19, cov_s, width=0.36, color=SERVED, edgecolor="white",
            linewidth=0.5, zorder=3, label="served")
    axc.bar(xpos + 0.19, cov_d, width=0.36, color="#b3b3b3", edgecolor="white",
            linewidth=0.5, zorder=3, label="demand-matched")
    fmt = lambda v: f"{v:.2f}" if v < 1.0 else f"{v:.1f}"
    for x, v in zip(xpos - 0.19, cov_s):
        axc.text(x, v + 1.4, fmt(v), ha="center", va="bottom",
                 fontsize=6.2, color=SERVED)
    for x, v in zip(xpos + 0.19, cov_d):
        axc.text(x, v + 1.4, fmt(v), ha="center", va="bottom",
                 fontsize=6.2, color=SUB)
    axc.set_xticks(xpos)
    axc.set_xticklabels([f"{100 * f:.0f} %" for f in fracs])
    axc.set_ylim(0, max(cov_d) * 1.28)
    axc.set_xlabel("impressions covered (%)", fontsize=7.0)
    axc.set_ylabel("catalogue\nneeded (%)", fontsize=7.0, linespacing=1.2)
    axc.tick_params(labelsize=6.5, length=2.2)
    axc.set_title("Coverage cost of each impression tier", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    axc.legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=1.2,
               borderaxespad=0.3)
    axc.grid(axis="y", color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    # ---- metric card -------------------------------------------------------
    axk = inch_axes(fig, (6.90, 5.02, 3.20, 0.92))
    text_card(axk, 0.05, 0.02, 3.10, 0.86,
              f"Gini  {g_served:.3f}   (demand {g_demand:.3f})\n"
              f"catalogue coverage  {100 - never:.1f} %   "
              f"never shown  {never:.1f} %",
              fs=7.0, color=INK, ha="left", pad=0.14, rounding=0.06,
              title="Concentration audit", title_fs=7.6, title_color=SUB)

    panel_label_inches(fig, ax, "(a)", dx=-0.52, dy=0.10)
    panel_label_inches(fig, axl, "(b)", dx=-0.52, dy=0.10)
    panel_label_inches(fig, axc, "(c)", dx=-0.52, dy=0.10)
    savefig(fig, "474_rec_longtail.png")


if __name__ == "__main__":
    main()
