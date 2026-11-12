"""Peak shaving with the payback drawn in, because the energy has to go back.

The evening price signal drives a price-elastic reduction, three contracted
programmes add their own blocks on top, and the sum of those four arrays is
the load reduction shown in the main panel.  Three quarters of the reduced
energy returns overnight as a payback bump, so the shaved and refilled areas
are computed from the same arrays and their ratio is asserted rather than
drawn by eye.  The scatter panel recovers the elasticity that generated the
price-responsive block by regressing hourly load change on price change, and
the bars split the peak megawatts and the shifted megawatt hours by programme.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
TEAL = "#2a7f7a"
PURPLE = "#7b5ea7"
SLATE = "#4a4f5c"
TAN = "#a89078"

ELAST = 0.022            # own-price elasticity of demand, dimensionless
PAYBACK = 0.75           # share of the shaved energy that returns later
P_REF = 42.0             # reference price, USD/MWh

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

PROGRAMS = [("critical peak pricing", ORANGE),
            ("direct load control, air conditioning", TEAL),
            ("industrial curtailment", TAN),
            ("smart EV charging", PURPLE)]


def day(rng):
    """Baseline load in MW and the day-ahead price in USD/MWh."""
    h = np.arange(24) + 0.5
    base = (4050.0
            + 780.0 * np.exp(-0.5 * ((h - 9.5) / 2.4) ** 2)
            + 1280.0 * np.exp(-0.5 * ((h - 19.0) / 2.1) ** 2)
            - 700.0 * np.exp(-0.5 * ((h - 4.0) / 3.1) ** 2)
            + rng.normal(0.0, 22.0, 24))
    price = (P_REF
             + 96.0 * np.exp(-0.5 * ((h - 19.0) / 1.5) ** 2)
             + 14.0 * np.exp(-0.5 * ((h - 9.0) / 2.2) ** 2)
             - 12.0 * np.exp(-0.5 * ((h - 3.5) / 3.0) ** 2)
             + rng.normal(0.0, 1.4, 24))
    return base, price


def response(base, price, rng):
    """Four reduction blocks in MW, plus the overnight payback."""
    h = np.arange(24) + 0.5
    dp = price / P_REF - 1.0
    cpp = ELAST * base * np.clip(dp, 0.0, None) \
        * (1.0 + rng.normal(0.0, 0.06, 24))
    cpp = np.clip(cpp, 0.0, None)

    window = ((h >= 17.0) & (h <= 21.0)).astype(float)
    dlc = 165.0 * window * np.exp(-0.5 * ((h - 19.0) / 1.9) ** 2)
    ind = 100.0 * ((h >= 18.0) & (h <= 20.0)).astype(float)
    ev = 80.0 * ((h >= 17.0) & (h <= 22.0)).astype(float)

    blocks = [cpp, dlc, ind, ev]
    red = sum(blocks)

    shape = np.exp(-0.5 * ((h - 2.0) / 2.6) ** 2) \
        + np.exp(-0.5 * ((h - 24.5) / 1.8) ** 2)
    shape = shape * (h < 6.0)
    pay = shape * (PAYBACK * red.sum() / shape.sum())
    return blocks, red, pay


def main_panel(ax, base, post, red, pay, price):
    x = np.arange(24) + 0.5
    ax.fill_between(x, post, base, where=base > post, facecolor="#f3ddd2",
                    edgecolor="none", zorder=2)
    ax.fill_between(x, base, post, where=post > base, facecolor="#e3dcef",
                    edgecolor="none", zorder=2)
    ax.plot(x, base, color=SLATE, lw=1.5, ls=(0, (4.0, 2.2)), zorder=5)
    ax.plot(x, post, color=ORANGE, lw=2.2, zorder=6)

    pk = int(np.argmax(base))
    ax.annotate("peak shaved by {:.0f} MW ({:.1f} %)"
                .format(red[pk], 100.0 * red[pk] / base[pk]),
                xy=(x[pk], 0.5 * (base[pk] + post[pk])),
                xytext=(x[pk] - 1.2, base[pk] + 480.0), fontsize=6.9,
                color=ORANGE, fontweight="bold", ha="right", va="bottom",
                zorder=10, arrowprops=dict(arrowstyle="-", color=ORANGE,
                                           lw=0.8, shrinkA=2, shrinkB=3))
    ib = int(np.argmax(pay))
    ax.annotate("payback fills the valley:\n{:.0f} MWh returns overnight"
                .format(pay.sum()), xy=(x[ib], 0.5 * (base[ib] + post[ib])),
                xytext=(x[ib] + 0.9, 3150.0), fontsize=6.6, color=PURPLE,
                ha="left", va="top", linespacing=1.32, zorder=10,
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.8,
                                shrinkA=2, shrinkB=3))

    axp = ax.twinx()
    axp.plot(x, price, color=RED, lw=1.3, ls=(0, (2.5, 1.8)), zorder=7)
    axp.set_ylim(0.0, 320.0)
    axp.set_ylabel("day-ahead price (\\$/MWh)", color=RED, fontsize=8.2)
    axp.tick_params(axis="y", colors=RED, labelsize=7.0)
    axp.set_yticks([0, 40, 80, 120])
    axp.spines["top"].set_visible(False)
    axp.spines["right"].set_color(RED)
    axp.spines["left"].set_visible(False)

    handles = [Line2D([], [], color=SLATE, lw=1.5, ls=(0, (4.0, 2.2)),
                      label="baseline load"),
               Line2D([], [], color=ORANGE, lw=2.2,
                      label="load after demand response"),
               Patch(facecolor="#f3ddd2", label="energy shaved"),
               Patch(facecolor="#e3dcef", label="energy paid back"),
               Line2D([], [], color=RED, lw=1.3, ls=(0, (2.5, 1.8)),
                      label="day-ahead price, right axis")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.4, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.8, borderaxespad=0.35, labelspacing=0.28)

    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(3000.0, 6100.0)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlabel("hour of day (hours)")
    ax.set_ylabel("load (MW)")
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    for side in ("top",):
        ax.spines[side].set_visible(False)
    ax.set_title("A summer weekday before and after the programmes respond",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def elasticity_panel(ax, base, price, cpp):
    dp = 100.0 * (price / P_REF - 1.0)
    dl = -100.0 * cpp / base
    keep = dp > 1.0
    slope, icept = np.polyfit(dp[keep], dl[keep], 1)
    xs = np.linspace(0.0, dp.max() * 1.05, 50)

    ax.axhline(0.0, color="#999999", lw=0.7, zorder=2)
    ax.scatter(dp[keep], dl[keep], s=17.0, facecolor=ORANGE,
               edgecolor="white", linewidth=0.5, zorder=5)
    ax.scatter(dp[~keep], dl[~keep], s=13.0, facecolor="#c9c9c9",
               edgecolor="white", linewidth=0.4, zorder=4)
    ax.plot(xs, icept + slope * xs, color=INK, lw=1.3, zorder=6)
    ax.text(0.03, 0.06, "fitted elasticity {:.3f}\ngenerated with {:.3f}"
            .format(-slope, ELAST), transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.2, color=SUB, linespacing=1.32, zorder=8)

    ax.set_xlim(-45.0, dp.max() * 1.10)
    ax.set_xlabel("price change from reference (per cent)", fontsize=6.8)
    ax.set_ylabel("load change (per cent)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Recovering the price response", fontsize=8.0,
                 fontweight="bold", color=INK)
    return -slope


def program_panel(ax, blocks, base):
    pk = int(np.argmax(base))
    peak = np.array([b[pk] for b in blocks])
    energy = np.array([b.sum() for b in blocks])
    y = np.arange(len(blocks))[::-1]

    ax.barh(y, peak, height=0.60, color=[c for _n, c in PROGRAMS],
            edgecolor="white", linewidth=0.7, zorder=4)
    for yi, p, e in zip(y, peak, energy):
        ax.text(p + 8.0, yi, "{:.0f} MW at the peak, {:.0f} MWh shifted"
                .format(p, e), ha="left", va="center", fontsize=6,
                color=SUB, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(["CPP", "DLC A/C", "industrial", "smart EV"],
                       fontsize=6.3)
    ax.set_xlim(0.0, peak.max() * 2.35)
    ax.set_ylim(-0.68, len(blocks) - 0.32)
    ax.set_xlabel("reduction at the system peak hour (MW)", fontsize=6.6)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Who delivers the shaved megawatts", fontsize=8.0,
                 fontweight="bold", color=INK)
    return peak, energy


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
    rng = np.random.default_rng(90210711)
    base, price = day(rng)
    blocks, red, pay = response(base, price, rng)
    post = base - red + pay

    assert np.allclose(sum(blocks), red, atol=1e-9), "programmes must sum"
    assert abs(pay.sum() - PAYBACK * red.sum()) < 1e-6, "payback share"
    assert np.allclose(post, base - red + pay, atol=1e-9), "load identity"
    assert np.all(red >= 0.0) and np.all(pay >= 0.0), "signs"

    pk = int(np.argmax(base))
    new_peak = float(post.max())

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_el = axes_inches(fig, AUX_T)
    ax_pr = axes_inches(fig, AUX_B)

    main_panel(ax_main, base, post, red, pay, price)
    fitted = elasticity_panel(ax_el, base, price, blocks[0])
    peak, energy = program_panel(ax_pr, blocks, base)



    panel_labels_inches(fig, [ax_main, ax_el, ax_pr], dx=-0.62, dy=0.14)
    savefig(fig, "971_pwr_demand_response.png")


if __name__ == "__main__":
    main()
