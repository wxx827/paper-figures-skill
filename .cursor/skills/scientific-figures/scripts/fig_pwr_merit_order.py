"""Merit-order supply stack: the clearing price is solved, never asserted.

Sixteen generating units are sorted by short-run marginal cost, so the top
edge of the stack is the system supply curve.  A slightly elastic demand curve
crosses it inside one unit's block; that unit is the marginal one and its
marginal cost is the clearing price.  Everything quoted elsewhere in the
figure is read back from the same two curves: the price-formation staircase in
the upper side panel re-solves the intersection for a sweep of demand levels,
and the rent bars in the lower one are the dispatched megawatts of each fuel
multiplied by the gap between the clearing price and that unit's own cost.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from pubstyle import set_pub_style, savefig, lighten

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"

FUEL_COLOR = {
    "solar": "#6bbf8a",
    "wind": "#2a7f7a",
    "hydro": "#3b5b92",
    "nuclear": "#4a4f5c",
    "coal": "#7a6a5d",
    "CCGT": "#a89078",
    "OCGT": "#c4a98d",
    "oil": "#8c6f5a",
}
FUEL_ORDER = ["solar", "wind", "hydro", "nuclear", "coal", "CCGT", "OCGT",
              "oil"]

# name, fuel, capacity (MW), short-run marginal cost in USD/MWh
UNITS = [
    ("PV portfolio", "solar", 1800.0, 0.0),
    ("Wind portfolio", "wind", 2600.0, 0.5),
    ("Hydro cascade", "hydro", 1400.0, 6.5),
    ("Nuclear 1", "nuclear", 1200.0, 9.5),
    ("Nuclear 2", "nuclear", 1200.0, 10.2),
    ("Nuclear 3", "nuclear", 1200.0, 11.0),
    ("Coal A", "coal", 900.0, 28.0),
    ("Coal B", "coal", 900.0, 32.0),
    ("Coal C", "coal", 1100.0, 36.0),
    ("CCGT 1", "CCGT", 800.0, 44.0),
    ("CCGT 2", "CCGT", 800.0, 48.0),
    ("CCGT 3", "CCGT", 700.0, 55.0),
    ("CCGT 4", "CCGT", 600.0, 62.0),
    ("OCGT 1", "OCGT", 500.0, 88.0),
    ("OCGT 2", "OCGT", 450.0, 102.0),
    ("Oil peaker", "oil", 350.0, 155.0),
]

D0 = 15200.0        # demand intercept at zero price, MW
ELAST = 8.0         # MW withdrawn per \\$/MWh, a deliberately shallow slope
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)


def mw(x):
    """Thin-space thousands separator, so the text keeps its own commas."""
    return "{:,.0f}".format(x).replace(",", " ")


def stack():
    """Merit-ordered units with their cumulative capacity edges."""
    order = sorted(range(len(UNITS)), key=lambda i: UNITS[i][3])
    lo = 0.0
    rows = []
    for i in order:
        name, fuel, cap, mc = UNITS[i]
        rows.append((name, fuel, cap, mc, lo, lo + cap))
        lo += cap
    return rows


def demand_at(price, d0):
    return d0 - ELAST * price


def clear(rows, d0):
    """Intersect the step supply curve with the demand line."""
    for (_n, _f, _c, mc, lo, hi) in rows:
        q = demand_at(mc, d0)
        if q <= hi + 1e-9:                       # demand met inside this block
            if q >= lo:
                return mc, q                     # price set by this unit
            return mc, q                         # demand below the block start
    mc = rows[-1][3]
    return mc, min(demand_at(mc, d0), rows[-1][5])


def supply_curve(rows, n=1400):
    """Sampled step supply so fills and the curve cannot disagree."""
    q = np.linspace(0.0, rows[-1][5], n)
    p = np.zeros_like(q)
    for (_n, _f, _c, mc, lo, hi) in rows:
        p[(q >= lo) & (q <= hi)] = mc
    return q, p


def main_panel(ax, rows, price, qstar, qs, ps):
    for (name, fuel, cap, mc, lo, hi) in rows:
        col = FUEL_COLOR[fuel]
        served = float(np.clip(qstar - lo, 0.0, cap))
        if served > 1e-6:
            ax.add_patch(Rectangle((lo, 0.0), served, mc, facecolor=col,
                                   edgecolor="white", linewidth=0.7,
                                   alpha=0.94, zorder=3))
        if cap - served > 1e-6:
            ax.add_patch(Rectangle((lo + served, 0.0), cap - served, mc,
                                   facecolor=lighten(col, 0.55),
                                   edgecolor="white", linewidth=0.7,
                                   alpha=0.95, zorder=3))

    inside = qs <= qstar
    ax.fill_between(qs[inside], ps[inside], price, step=None,
                    facecolor="#e8d9c8", alpha=0.55, zorder=4, lw=0)
    ax.plot(qs, ps, color=INK, lw=1.25, zorder=6)

    pp = np.linspace(0.0, 170.0, 200)
    ax.plot(demand_at(pp, D0), pp, color=ORANGE, lw=1.9, zorder=7)
    ax.plot([qstar], [price], marker="o", ms=6.2, mfc=RED, mec="white",
            mew=1.1, zorder=9)

    ax.axhline(price, color=RED, lw=0.9, ls=(0, (4.5, 2.4)), zorder=5)
    ax.axvline(qstar, color=RED, lw=0.9, ls=(0, (4.5, 2.4)), zorder=5)
    ax.text(220.0, price + 3.0, "clearing price {:.0f} \\$/MWh".format(price),
            ha="left", va="bottom", fontsize=7.0, color=RED,
            fontweight="bold", zorder=10)
    ax.text(qstar - 220.0, 166.0, "cleared volume\n{:,.0f} MW"
            .format(qstar).replace(",", " "), ha="right", va="top",
            fontsize=6.8, color=RED, linespacing=1.32, zorder=10)

    marg = [r for r in rows if r[4] <= qstar <= r[5]]
    if marg:
        name, fuel, cap, mc, lo, hi = marg[0]
        ax.annotate("marginal unit: {:s}\n{:.0f} MW at {:.0f} \\$/MWh"
                    .format(name, cap, mc), xy=(0.5 * (lo + hi), mc),
                    xytext=(qstar - 1500.0, mc + 46.0), fontsize=6.8,
                    color=INK, ha="right", va="bottom", linespacing=1.32,
                    zorder=10,
                    arrowprops=dict(arrowstyle="-", color=SUB, lw=0.8,
                                    shrinkA=2, shrinkB=3))

    ax.text(3300.0, 40.0, "infra-marginal rent\n(price minus own cost)",
            ha="center", va="center", fontsize=6.6, color="#8a6f4f",
            fontweight="bold", linespacing=1.32, zorder=8)

    handles = [Patch(facecolor=FUEL_COLOR[f], edgecolor="white", lw=0.7,
                     label=f) for f in FUEL_ORDER]
    handles += [
        Line2D([], [], color=INK, lw=1.25, label="supply curve (merit order)"),
        Line2D([], [], color=ORANGE, lw=1.9, label="demand, 8 MW per \\$/MWh"),
        Patch(facecolor="#f0e5d8", edgecolor="#cbb79f", lw=0.6,
              label="uncommitted capacity"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.5, ncol=2, columnspacing=1.0,
              borderaxespad=0.3, labelspacing=0.26)

    ax.set_xlim(0.0, rows[-1][5])
    ax.set_ylim(0.0, 172.0)
    ax.set_xlabel("cumulative available capacity (MW)")
    ax.set_ylabel("short-run marginal cost (\\$/MWh)")
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Day-ahead supply stack of a 16-unit system, single hour",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def formation_panel(ax, rows, price, qstar):
    d0 = np.linspace(6500.0, 17200.0, 500)
    pr = np.array([clear(rows, d)[0] for d in d0])
    qq = np.array([clear(rows, d)[1] for d in d0])

    ax.plot(qq, pr, color=ORANGE, lw=1.6, zorder=5)
    ax.plot([qstar], [price], marker="o", ms=5.4, mfc=RED, mec="white",
            mew=0.9, zorder=7)
    ax.annotate("today {:,.0f} MW\n{:.0f} \\$/MWh".format(qstar, price)
                .replace(",", " "), xy=(qstar, price),
                xytext=(qstar - 900.0, price + 34.0), fontsize=6.2,
                color=RED, ha="right", va="bottom", linespacing=1.30,
                zorder=8, arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                          shrinkA=2, shrinkB=3))
    peak = qq[-1], pr[-1]
    ax.text(6500.0, 168.0,
            "the last {:.0f} % of demand\nmultiplies the price by {:.1f}"
            .format(100.0 * (peak[0] / qstar - 1.0), peak[1] / price),
            ha="left", va="top", fontsize=6.1, color=SUB, linespacing=1.32,
            zorder=8)

    ax.set_xlim(6200.0, 16800.0)
    ax.set_ylim(0.0, 172.0)
    ax.set_xlabel("cleared volume (MW)", fontsize=6.8)
    ax.set_ylabel("clearing price (\\$/MWh)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Price formation as demand grows", fontsize=8.0,
                 fontweight="bold", color=INK)


def rent_panel(ax, rows, price, qstar):
    rent, served = {}, {}
    for (_n, fuel, cap, mc, lo, _hi) in rows:
        q = float(np.clip(qstar - lo, 0.0, cap))
        if q <= 0.0:
            continue
        served[fuel] = served.get(fuel, 0.0) + q
        rent[fuel] = rent.get(fuel, 0.0) + q * (price - mc) / 1000.0

    fuels = [f for f in FUEL_ORDER if f in rent]
    y = np.arange(len(fuels))[::-1]
    vals = [rent[f] for f in fuels]
    ax.barh(y, vals, height=0.62, color=[FUEL_COLOR[f] for f in fuels],
            edgecolor="white", linewidth=0.7, zorder=4)
    for yi, f, v in zip(y, fuels, vals):
        ax.text(v + 2.0, yi, "{:.0f} k\\$/h  ({:,.0f} MW)".format(
            v, served[f]).replace(",", " "), ha="left", va="center",
            fontsize=6, color=SUB, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(fuels, fontsize=6.3)
    ax.set_xlim(0.0, max(vals) * 1.52)
    ax.set_ylim(-0.65, len(fuels) - 0.35)
    ax.set_xlabel("infra-marginal rent at the clearing price (k\\$ per hour)",
                  fontsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Who earns the rent this hour", fontsize=8.0,
                 fontweight="bold", color=INK)
    return sum(vals)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rows = stack()
    price, qstar = clear(rows, D0)

    dispatched = sum(float(np.clip(qstar - lo, 0.0, cap))
                     for (_n, _f, cap, _mc, lo, _hi) in rows)
    assert abs(dispatched - qstar) < 1e-6, "dispatch must equal cleared volume"
    assert abs(demand_at(price, D0) - qstar) < 1e-6, "off the demand curve"

    qs, ps = supply_curve(rows)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_form = axes_inches(fig, AUX_T)
    ax_rent = axes_inches(fig, AUX_B)

    main_panel(ax_main, rows, price, qstar, qs, ps)
    formation_panel(ax_form, rows, price, qstar)
    total_rent = rent_panel(ax_rent, rows, price, qstar)

    cap_tot = rows[-1][5]


    panel_labels_inches(fig, [ax_main, ax_form, ax_rent], dx=-0.62, dy=0.14)
    savefig(fig, "966_pwr_merit_order.png")


if __name__ == "__main__":
    main()
