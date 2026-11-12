"""Who gains from the transfer, in dollars and as a share of what they have.

Main panel: the average benefit received by each income decile as bars on
the left axis, against the same benefit expressed as a share of that
decile's income on the right axis.  The bars barely move - the scheme pays
roughly the same amount to everyone - while the line falls by more than an
order of magnitude, and that divergence is the whole point: a flat transfer
is strongly progressive in relative terms even when it looks neutral in
absolute terms.  The Kakwani progressivity index is quoted on the panel.

Auxiliary (both at 58 % of the main panel width): Lorenz curves before and
after the transfer with both Gini coefficients; and the benefit
concentration curve against the pre-transfer Lorenz curve, where the area
between the two is the progressivity index itself.

Every coefficient is integrated off a plotted curve: the Ginis and the
concentration index are one minus twice the trapezoidal area under the
Lorenz and concentration curves as drawn, Kakwani is the difference of the
two, and Reynolds-Smolensky is the difference of the two Ginis.  The script
asserts that both Lorenz curves are monotone, stay under the diagonal, and
that the post-transfer curve lies above the pre-transfer one, so the
plotted geometry and the quoted numbers cannot disagree.  Fictional country,
illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten, darken
from figkit import axes_inches, panel_labels_inches

DEEP = "#2b4470"
BLUE = "#3b5b92"
TEAL = "#2e7d5b"
GOLD = "#c9a227"
RUST = "#c0603f"
ACCENT = "#b23a48"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.98, 0.86, 5.36, 4.62)
AUX_T_RECT = (7.74, 3.48, 3.04, 2.00)
AUX_B_RECT = (7.74, 0.86, 3.04, 2.00)

N_HH = 24000
# the whole transfer package rather than one scheme: a single small rebate
# moves the Lorenz curve by less than a line width, which makes panel (b)
# unreadable and the comparison pointless
BASE = 6800.0         # flat part of the transfer, USD per capita per year
TAPER = 0.045         # top-up per USD of income
CAP = 30000.0         # income above which the top-up stops growing
NGRID = 401


def build_households(seed=929):
    """Household income and transfer received, both in USD per year."""
    rng = np.random.default_rng(seed)
    inc = np.exp(rng.normal(np.log(34000.0), 0.80, N_HH))
    inc = np.clip(inc, 4200.0, 900000.0)
    ben = (BASE + TAPER * np.minimum(inc, CAP)
           + rng.normal(0.0, 260.0, N_HH))
    return inc, np.clip(ben, 1500.0, None)


def lorenz(values, weights_order=None, n=NGRID):
    """(p, L) on a fixed grid; ordered by ``weights_order`` when given."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v if weights_order is None else weights_order,
                       kind="stable")
    cum = np.cumsum(v[order])
    cum = np.concatenate([[0.0], cum]) / cum[-1]
    p_obs = np.linspace(0.0, 1.0, cum.size)
    p = np.linspace(0.0, 1.0, n)
    return p, np.interp(p, p_obs, cum)


def gini_from_curve(p, curve):
    """1 - 2 * area under the curve as drawn; the plot defines the number."""
    return float(1.0 - 2.0 * np.trapezoid(curve, p))


def nice_ticks(hi, n=5):
    """Round tick values from 0 up to ``hi``, so a spine can stop at hi."""
    t = mticker.MaxNLocator(n, steps=[1, 2, 2.5, 5, 10]).tick_values(0.0, hi)
    return [v for v in t if 0.0 <= v <= hi * 1.001]


def main():
    set_pub_style(base_font=8)
    inc, ben = build_households()

    edges = np.quantile(inc, np.linspace(0.0, 1.0, 11))
    edges[-1] += 1.0
    dec = np.clip(np.digitize(inc, edges[1:-1]), 0, 9)
    inc_d = np.array([inc[dec == k].mean() for k in range(10)])
    ben_d = np.array([ben[dec == k].mean() for k in range(10)])
    rel_d = 100.0 * ben_d / inc_d
    assert np.all(np.diff(inc_d) > 0.0)

    p, l_pre = lorenz(inc)
    _p, l_post = lorenz(inc + ben)
    _p, conc = lorenz(ben, weights_order=inc)
    for curve in (l_pre, l_post):
        assert np.all(np.diff(curve) >= -1e-12), "Lorenz curve must not fall"
        assert np.all(curve <= p + 1e-9), "Lorenz curve must stay under y=x"
    assert np.all(l_post >= l_pre - 1e-9), "the transfer must not raise Gini"

    g_pre = gini_from_curve(p, l_pre)
    g_post = gini_from_curve(p, l_post)
    c_ben = gini_from_curve(p, conc)
    kakwani = g_pre - c_ben
    rs = g_pre - g_post
    assert g_post < g_pre and rs > 0.0

    bottom_half = 100.0 * float(np.interp(0.5, p, conc))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_l = axes_inches(fig, AUX_T_RECT)
    ax_c = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_l, ax_c):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: absolute benefit as bars, relative benefit as a line ---------
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7, zorder=0)
    xk = np.arange(10)
    ax.bar(xk, ben_d, width=0.66, color=lighten(TEAL, 0.40), edgecolor=TEAL,
           linewidth=0.9, zorder=3)
    bar_top = float(ben_d.max()) * 1.06
    for k in xk:
        # low inside the bar: near the top the label would fight the % line
        ax.text(k, 0.60 * ben_d[k], f"{ben_d[k]:,.0f}", ha="center",
                va="center", fontsize=6.1, color=darken(TEAL, 0.28),
                zorder=5)
    ax.axhline(ben.mean(), color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=4)
    ax.text(-0.60, ben.mean() + 0.016 * bar_top,
            f"mean benefit {ben.mean():,.0f}", ha="left", va="bottom",
            fontsize=6.2, color=SUB, zorder=5)

    ax.set_xticks(xk)
    ax.set_xticklabels([f"D{k + 1}" for k in xk], fontsize=7.4)
    ax.set_xlim(-0.72, 9.72)
    ax.set_ylim(0.0, bar_top * 2.02)
    ax.set_yticks(nice_ticks(bar_top))
    ax.spines["left"].set_bounds(0.0, bar_top)
    ax.set_xlabel("household income decile (D1 poorest, D10 richest)",
                  fontsize=8.5)
    ax.set_ylabel("benefit received (USD per capita per year)", fontsize=8.5,
                  color=darken(TEAL, 0.16))
    ax.tick_params(axis="y", colors=darken(TEAL, 0.16))
    ax.set_title(f"Incidence of the Aldrune household transfer package, "
                 f"n={N_HH:,} households", fontsize=9.0, fontweight="bold",
                 pad=5.0)

    ax2 = ax.twinx()
    ax2.plot(xk, rel_d, color=DEEP, lw=2.0, zorder=7, solid_capstyle="round")
    ax2.plot(xk, rel_d, ls="none", marker="o", ms=4.4, mfc="white", mec=DEEP,
             mew=1.4, zorder=8)
    for k in (0, 4, 9):
        ax2.text(k + (0.26 if k < 9 else -0.26),
                 rel_d[k] + 0.045 * float(rel_d.max()),
                 f"{rel_d[k]:.1f} %", ha="left" if k < 9 else "right",
                 va="bottom", fontsize=6.4, color=DEEP, zorder=9)
    rel_top = float(rel_d.max()) * 1.07
    ax2.set_xlim(-0.72, 9.72)
    ax2.set_ylim(0.0, rel_top * 2.28)
    ax2.set_yticks(nice_ticks(rel_top))
    ax2.spines["right"].set_bounds(0.0, rel_top)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylabel("benefit as a share of household income (%)",
                   fontsize=8.5, color=DEEP)
    ax2.tick_params(axis="y", colors=DEEP, length=3)
    ax2.set_zorder(6)
    ax2.patch.set_visible(False)

    ax.text(0.002, 0.995,
            f"Kakwani progressivity index {kakwani:+.3f}   |   "
            f"Reynolds-Smolensky {rs:+.3f}   |   "
            f"Gini {g_pre:.3f} -> {g_post:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            color=SUB)
    ax.text(0.002, 0.923,
            f"the bars vary by only {ben_d.max() / ben_d.min():.2f}x across "
            f"the distribution, but the same money is worth "
            f"{rel_d[0] / rel_d[-1]:.0f}x more\nto the poorest decile than to "
            f"the richest; the bottom half of households collect "
            f"{bottom_half:.1f} % of the total outlay",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
            color=ACCENT, linespacing=1.55)

    hh = [Line2D([], [], color=TEAL, lw=5.0, alpha=0.55,
                 label="benefit in USD per capita (left axis)"),
          Line2D([], [], color=DEEP, lw=2.0, marker="o", ms=4.0, mfc="white",
                 mew=1.2, label="benefit as % of income (right axis)")]
    ax.legend(handles=hh, loc="upper right", frameon=False, fontsize=7.0,
              handlelength=2.2, labelspacing=0.40, borderpad=0.15,
              bbox_to_anchor=(1.004, 0.845))

    # ---- aux (b): Lorenz curves before and after ----------------------------
    ax_l.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_l.plot([0, 1], [0, 1], color=FAINT, lw=0.9, ls=(0, (2.6, 2.0)),
              zorder=2)
    ax_l.fill_between(p, l_pre, l_post, facecolor=lighten(TEAL, 0.74),
                      linewidth=0.0, zorder=3)
    ax_l.plot(p, l_pre, color=RUST, lw=1.7, zorder=5)
    ax_l.plot(p, l_post, color=TEAL, lw=1.7, zorder=6)
    i50 = int(np.argmin(np.abs(p - 0.5)))
    ax_l.annotate("", xy=(0.5, l_post[i50]), xytext=(0.5, l_pre[i50]),
                  arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color=ACCENT,
                                  shrinkA=0, shrinkB=0), zorder=8)
    ax_l.text(0.465, 0.5 * (l_pre[i50] + l_post[i50]) + 0.055,
              f"+{100 * (l_post[i50] - l_pre[i50]):.1f} pp of income\n"
              f"to the poorer half",
              ha="right", va="bottom", fontsize=6.1, color=ACCENT,
              linespacing=1.45, zorder=8)
    ax_l.set_xlim(0.0, 1.0)
    ax_l.set_ylim(0.0, 1.0)
    ax_l.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_l.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_l.tick_params(labelsize=6.6)
    ax_l.set_aspect("equal")
    ax_l.set_xlabel("cumulative share of households", fontsize=7.6,
                    labelpad=1.5)
    ax_l.set_ylabel("cumulative share of income", fontsize=7.6)
    ax_l.set_title("The transfer moves the Lorenz curve up", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_l.text(0.965, 0.045,
              f"Gini before {g_pre:.3f}\nGini after  {g_post:.3f}\n"
              f"shaded gap = {rs:.3f} Gini points",
              transform=ax_l.transAxes, ha="right", va="bottom", fontsize=6.1,
              color=SUB, linespacing=1.55)
    ax_l.text(0.045, 0.955, "before transfer", transform=ax_l.transAxes,
              ha="left", va="top", fontsize=6.2, color=RUST)
    ax_l.text(0.045, 0.878, "after transfer", transform=ax_l.transAxes,
              ha="left", va="top", fontsize=6.2, color=darken(TEAL, 0.10))

    # ---- aux (c): benefit concentration against the Lorenz curve ------------
    ax_c.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_c.plot([0, 1], [0, 1], color=FAINT, lw=0.9, ls=(0, (2.6, 2.0)),
              zorder=2)
    ax_c.fill_between(p, l_pre, conc, facecolor=lighten(GOLD, 0.74),
                      linewidth=0.0, zorder=3)
    ax_c.plot(p, l_pre, color=RUST, lw=1.7, zorder=5)
    ax_c.plot(p, conc, color=GOLD, lw=1.9, zorder=6)
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_c.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_c.tick_params(labelsize=6.6)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("cumulative share of households, poorest first",
                    fontsize=7.6, labelpad=1.5)
    ax_c.set_ylabel("cumulative share received", fontsize=7.6)
    ax_c.set_title("Benefits spread far more evenly than income",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_c.text(0.965, 0.045,
              f"concentration index {c_ben:+.3f}\nincome Gini {g_pre:.3f}\n"
              f"shaded area x 2 = Kakwani {kakwani:.3f}",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=6.1,
              color=SUB, linespacing=1.55)
    ax_c.text(0.045, 0.955, "benefit concentration curve",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=6.2,
              color=darken(GOLD, 0.18))
    ax_c.text(0.045, 0.878, "income Lorenz curve", transform=ax_c.transAxes,
              ha="left", va="top", fontsize=6.2, color=RUST)

    panel_labels_inches(fig, [ax, ax_l, ax_c], dx=-0.72, dy=0.12)
    savefig(fig, "929_pcy_equity_incidence.png")


if __name__ == "__main__":
    main()
