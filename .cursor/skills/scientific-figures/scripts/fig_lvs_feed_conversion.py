"""Feed conversion over the growing period, and what it costs.

Feed conversion ratio is not a constant of an animal: it worsens steadily with
age, because maintenance scales with metabolic body size while daily gain is
already past its peak.  The main panel shows that rise for three rations, with
pen-level error bars, over a stacked area giving the cumulative feed the
animals have eaten to get there and how its composition shifts from
protein-rich starter to cereal-heavy finisher.

The subordinate panels close the loop from biology to money: the pooled
intake-versus-gain regression whose slope *is* the whole-period conversion
ratio, and the feed cost of a kilogram of gain once ration price is applied -
the ration with the best conversion is not automatically the cheapest.

Intake is built from a maintenance term and a marginal cost of gain, so every
conversion ratio on the page is a quotient of two plotted arrays.  Synthetic
data; no real trial is represented.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"

# Cumulative-intake stack: one hue, three shades, so it reads as a single
# quantity split three ways rather than three competing series.
FEED_SHADES = ["#f0e8d8", "#dfd0b1", "#c6ae83"]
FEED_PARTS = ["cereal fraction", "protein meal fraction", "premix fraction"]

AGE_LO, AGE_HI = 42.0, 168.0
STAGE_EDGES = np.arange(AGE_LO, AGE_HI + 1.0, 21.0)
N_PEN = 6

# name, mature weight A (kg), rate k (1/d), marginal feed per kg gain,
# ration price (currency units per kg), colour
RATIONS = [
    ("control", 175.0, 0.0134, 1.500, 1.00, BLUE),
    ("high-lysine", 179.0, 0.0143, 1.348, 1.15, GREEN),
    ("enzyme-supplemented", 176.0, 0.0139, 1.420, 1.06, GOLD),
]
B_GOMP = 4.0
MAINT = 0.030        # kg feed per day per kg^0.75 of body weight


def trajectory(A, k, marg, t):
    """Body weight, daily gain and daily intake for one pen."""
    w = A * np.exp(-B_GOMP * np.exp(-k * t))
    adg = np.gradient(w, t)
    intake = MAINT * np.power(w, 0.75) + marg * adg
    return w, adg, intake


def pen_set(rng, ration, t):
    """``N_PEN`` pens of one ration, each with its own growth and appetite."""
    _, A, k, marg, price, _ = ration
    out = []
    for _ in range(N_PEN):
        out.append(trajectory(A * rng.normal(1.0, 0.022),
                              k * rng.normal(1.0, 0.030),
                              marg * rng.normal(1.0, 0.026), t))
    return out


def stage_fcr(t, adg, intake):
    """Feed conversion ratio per stage: feed eaten over weight put on."""
    vals = []
    for a, b in zip(STAGE_EDGES[:-1], STAGE_EDGES[1:]):
        m = (t >= a) & (t < b)
        vals.append(float(np.trapezoid(intake[m], t[m])
                          / np.trapezoid(adg[m], t[m])))
    return np.array(vals)


def part_shares(t):
    """Cereal / protein / premix shares of the ration, by age."""
    u = (t - AGE_LO) / (AGE_HI - AGE_LO)
    cereal = 0.552 + 0.168 * u
    premix = 0.082 - 0.024 * u
    return np.vstack([cereal, 1.0 - cereal - premix, premix])


def main_panel(ax, t, pens, mids):
    axr = ax.twinx()
    ref = pens[0]
    cum = np.cumsum(np.mean([p[2] for p in ref], axis=0)) * (t[1] - t[0])
    shares = part_shares(t)
    stack = np.cumsum(shares * cum[None, :], axis=0)
    lower = np.zeros_like(cum)
    patches = []
    for row, shade, name in zip(stack, FEED_SHADES, FEED_PARTS):
        axr.fill_between(t, lower, row, facecolor=shade, edgecolor="white",
                         lw=0.5, zorder=1)
        patches.append(plt.Rectangle((0, 0), 1, 1, facecolor=shade,
                                     edgecolor="white", lw=0.5))
        lower = row
    axr.set_ylim(0.0, float(cum[-1]) * 1.45)
    axr.set_ylabel("cumulative feed intake, control ration (kg)",
                   fontsize=8.0, color=SUB)
    axr.tick_params(labelsize=7.2, colors=SUB)
    axr.spines["top"].set_visible(False)

    handles, labels = [], []
    stats = []
    for (name, _, _, _, price, col), group in zip(RATIONS, pens):
        f = np.array([stage_fcr(t, p[1], p[2]) for p in group])
        mu, se = f.mean(axis=0), f.std(axis=0, ddof=1) / np.sqrt(N_PEN)
        ln = ax.errorbar(mids, mu, yerr=se, color=col, lw=1.8, marker="o",
                         ms=4.2, mfc="white", mew=1.2, capsize=2.6,
                         capthick=0.9, elinewidth=0.9, zorder=5)
        handles.append(ln)
        labels.append(f"{name}  ({mu[0]:.2f} to {mu[-1]:.2f} kg/kg)")
        stats.append((name, mu, se, price, col))

    ax.set_xlim(AGE_LO - 4.0, AGE_HI + 4.0)
    ax.set_ylim(1.4, 4.3)
    ax.set_xticks(STAGE_EDGES)
    ax.set_xlabel("age (d)", fontsize=8.0)
    ax.set_ylabel("feed conversion ratio (kg feed / kg gain)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_zorder(axr.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_title("Conversion worsens with age as maintenance takes a "
                 "bigger share", fontsize=9.0, fontweight="bold", color=INK)

    leg = ax.legend(handles + patches, labels + FEED_PARTS, fontsize=6.8,
                    frameon=True, framealpha=0.94, facecolor="white",
                    edgecolor="#cccccc", loc="upper left", handlelength=1.9,
                    borderpad=0.42, borderaxespad=0.5, ncol=1)
    leg.set_zorder(9)

    rise = 100.0 * (stats[0][1][-1] - stats[0][1][0]) / stats[0][1][0]
    ax.annotate(f"+{rise:.0f} % from first to last stage",
                xy=(mids[-1], stats[0][1][-1]),
                xytext=(mids[2] + 4.0, stats[0][1][-1] + 0.30), fontsize=6.8,
                color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=RULE, lw=0.9,
                                mutation_scale=7, shrinkA=2.0, shrinkB=4.0))
    return stats, float(cum[-1]), rise


def regression_panel(ax, t, pens):
    gains, intakes = [], []
    for (name, _, _, _, _, col), group in zip(RATIONS, pens):
        g = np.array([p[0][-1] - p[0][0] for p in group])
        f = np.array([np.trapezoid(p[2], t) for p in group])
        ax.scatter(g, f, s=22, facecolor="white", edgecolor=col, lw=1.2,
                   zorder=5, label=name)
        gains.append(g)
        intakes.append(f)
    g = np.concatenate(gains)
    f = np.concatenate(intakes)
    slope, icept = np.polyfit(g, f, 1)
    r = float(np.corrcoef(g, f)[0, 1])
    ratio = float(f.sum() / g.sum())
    xs = np.linspace(g.min() * 0.982, g.max() * 1.018, 50)
    ax.plot(xs, slope * xs + icept, color=RED, lw=1.5, zorder=4)
    # The fitted slope answers "what does one extra kg of gain cost?"; the
    # line through the origin is the whole-period ratio the main panel plots.
    # They differ by the maintenance feed, which the intercept absorbs.
    ax.plot(xs, ratio * xs, color=BASE, lw=1.2, ls=(0, (4.5, 2.4)), zorder=3)
    ax.text(0.035, 0.955, f"fitted slope = {slope:.2f} kg feed per extra "
                          f"kg gain\nratio of totals = {ratio:.2f} kg feed / "
                          f"kg gain\nefficiency = {1.0 / ratio:.3f} kg gain / "
                          f"kg feed\n$R^2$ = {r ** 2:.3f},  n={g.size} pens",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color=INK, linespacing=1.55)
    ax.set_xlabel("period weight gain (kg)", fontsize=7.6)
    ax.set_ylabel("period feed intake (kg)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(lw=0.45, alpha=0.5)
    ax.set_axisbelow(True)
    h, lab = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], color=RED, lw=1.5),
          plt.Line2D([], [], color=BASE, lw=1.2, ls=(0, (4.5, 2.4)))]
    lab += ["least-squares fit", "ratio of totals, through origin"]
    ax.legend(h, lab, fontsize=6.2, frameon=False, loc="lower right",
              handletextpad=0.3, borderaxespad=0.3, handlelength=1.7)
    ax.set_title("Marginal and whole-period conversion differ", fontsize=8.4,
                 fontweight="bold", color=INK)
    return slope, ratio, r ** 2


def cost_panel(ax, t, pens):
    names, mu, se, cols = [], [], [], []
    for (name, _, _, _, price, col), group in zip(RATIONS, pens):
        c = np.array([price * np.trapezoid(p[2], t) / (p[0][-1] - p[0][0])
                      for p in group])
        names.append(name.replace("-", "-\n"))
        mu.append(float(c.mean()))
        se.append(float(c.std(ddof=1) / np.sqrt(N_PEN)))
        cols.append(col)
    mu = np.array(mu)
    se = np.array(se)
    best = int(np.argmin(mu))
    x = np.arange(mu.size)
    # Bars keep their ration colour so the mapping holds across all three
    # panels; the winner is called out with the one accent instead.
    edges = ["white"] * mu.size
    widths = [0.7] * mu.size
    edges[best], widths[best] = RED, 1.8
    ax.bar(x, mu, width=0.56, color=cols, edgecolor=edges, lw=widths,
           zorder=3)
    ax.errorbar(x, mu, yerr=se, fmt="none", ecolor=RULE, elinewidth=0.9,
                capsize=2.6, capthick=0.9, zorder=5)
    for xi, v, e in zip(x, mu, se):
        ax.text(xi, v + e + 0.035 * mu.max(), f"{v:.2f}", ha="center",
                va="bottom", fontsize=6.5, color=INK)
    ax.set_ylim(0.0, float(mu.max() + se.max()) * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("feed cost per kg gain (CU/kg)", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.965, f"red outline = cheapest gain "
                        f"({RATIONS[best][0]})",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
            color=RED)
    ax.set_title("Best conversion is not the cheapest gain", fontsize=8.4,
                 fontweight="bold", color=INK)
    return names, mu, best


def main():
    set_pub_style()
    rng = np.random.default_rng(20844)
    t = np.linspace(AGE_LO, AGE_HI, 505)
    pens = [pen_set(rng, r, t) for r in RATIONS]
    mids = 0.5 * (STAGE_EDGES[:-1] + STAGE_EDGES[1:])

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 1.06, 4.94, 4.44))
    ax_reg = axes_inches(fig, (7.24, 3.54, 2.70, 1.96))
    ax_cost = axes_inches(fig, (7.24, 1.06, 2.70, 1.90))

    stats, cum_end, rise = main_panel(ax_main, t, pens, mids)
    slope, ratio, r2 = regression_panel(ax_reg, t, pens)
    _, cost, best = cost_panel(ax_cost, t, pens)

    panel_labels_inches(fig, [ax_main, ax_reg, ax_cost], dx=-0.54, dy=0.14)
    savefig(fig, "844_lvs_feed_conversion.png")


if __name__ == "__main__":
    main()
