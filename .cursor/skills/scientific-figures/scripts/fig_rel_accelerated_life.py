"""Accelerated life test read as an Arrhenius extrapolation.

Main panel: failure times from four temperature cells on an Arrhenius grid -
1000/T along the bottom, junction temperature along the top, life on a log
axis - each cell drawn as a violin of its own failure distribution over the
raw points, with the least-squares line through every failure and a
confidence band that widens as the line leaves the tested range.  The
activation energy comes from the slope of that line, and the acceleration
factor from the same fit evaluated at the use temperature.

The right-hand panels are the two things a qualification report has to show
next: the cumulative failure curves of the four cells, and how fast the
uncertainty on the extrapolated life grows once the use condition sits below
the coldest cell that was actually run.

Everything is fitted, not asserted: the activation energy, the acceleration
factor, the extrapolated life and the widening band all come out of the same
regression over the plotted points.  Values are illustrative only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from pubstyle import set_pub_style, savefig, style_axes, lighten, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
PURPLE = "#8e6bbf"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.86, 0.80, 5.18, 4.32)
AUX_T_RECT = (7.16, 3.50, 2.88, 1.88)
AUX_B_RECT = (7.16, 0.80, 2.88, 1.88)

K_B = 8.617333e-5                  # eV/K
CELLS = [(150.0, 18, RED), (135.0, 18, GOLD), (120.0, 18, GREEN),
         (105.0, 18, BLUE)]
EA_TRUE, A_TRUE = 0.62, 8.3e-5     # eV and hours, generator only
T_USE = 55.0                       # degC, field use condition
SIGMA_LN = 0.34                    # within-cell lognormal scatter


def kelvin(c):
    return np.asarray(c, dtype=float) + 273.15


def draw_cells(seed=9101):
    """Failure times in hours for every cell, plus the pooled regression."""
    rng = np.random.default_rng(seed)
    data = {}
    for temp_c, n, col in CELLS:
        med = A_TRUE * np.exp(EA_TRUE / (K_B * kelvin(temp_c)))
        data[temp_c] = med * np.exp(rng.normal(0.0, SIGMA_LN, n))
    x = np.concatenate([np.full(len(v), 1000.0 / kelvin(t))
                        for t, v in data.items()])
    y = np.log(np.concatenate(list(data.values())))
    return data, x, y


def band(x0, x, m, b, s, xbar, sxx, tcrit):
    """Two-sided confidence band on the fitted line, in hours."""
    se = s * np.sqrt(1.0 / x.size + (x0 - xbar) ** 2 / sxx)
    mu = m * x0 + b
    return np.exp(mu - tcrit * se), np.exp(mu + tcrit * se), np.exp(mu)


def violin(ax, xc, vals, color, half=0.026):
    """Vertical violin of log-spaced values, centred on ``xc``."""
    lg = np.log10(vals)
    kde = stats.gaussian_kde(lg, bw_method=0.55)
    grid = np.linspace(lg.min() - 0.28, lg.max() + 0.28, 200)
    dens = kde(grid)
    w = half * dens / dens.max()
    ax.fill_betweenx(10.0 ** grid, xc - w, xc + w, color=lighten(color, 0.60),
                     edgecolor=color, linewidth=0.7, zorder=4)


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches any axis label or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        for art in (a.xaxis.label, a.yaxis.label, a.title):
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:40])


def main():
    set_pub_style(base_font=8)
    data, x, y = draw_cells()

    m, b = np.polyfit(x, y, 1)
    resid = y - (m * x + b)
    s = float(np.sqrt((resid ** 2).sum() / (x.size - 2)))
    xbar, sxx = float(x.mean()), float(((x - x.mean()) ** 2).sum())
    tcrit = float(stats.t.ppf(0.975, x.size - 2))

    ea = m * K_B * 1000.0
    x_use = 1000.0 / kelvin(T_USE)
    lo_use, hi_use, life_use = band(x_use, x, m, b, s, xbar, sxx, tcrit)
    x_hot = 1000.0 / kelvin(CELLS[0][0])
    af = float(np.exp(m * (x_use - x_hot)))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_u = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_c, ax_u):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main: the Arrhenius fit -------------------------------------------
    ax.grid(which="both", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax.set_yscale("log")
    xs = np.linspace(1000.0 / kelvin(162.0), x_use + 0.035, 260)
    lo, hi, mid = band(xs, x, m, b, s, xbar, sxx, tcrit)
    tested = xs <= 1000.0 / kelvin(CELLS[-1][0])
    ax.fill_between(xs, lo, hi, color="#e9edf3", zorder=2)
    ax.fill_between(xs[tested], lo[tested], hi[tested], color="#d7dfe9",
                    zorder=3)
    ax.plot(xs, mid, color=INK, lw=1.5, zorder=6)
    ax.plot(xs[~tested], mid[~tested], color=INK, lw=1.5,
            ls=(0, (4.0, 2.2)), zorder=7)

    rng = np.random.default_rng(4)
    for temp_c, n, col in CELLS:
        xc = 1000.0 / kelvin(temp_c)
        violin(ax, xc, data[temp_c], col)
        ax.plot(xc + rng.uniform(-0.012, 0.012, n), data[temp_c], ls="none",
                marker="o", ms=2.6, mfc=col, mec="white", mew=0.4, zorder=6)
        ax.text(xc, data[temp_c].max() * 2.4, f"{temp_c:.0f} $^\\circ$C",
                ha="center", va="bottom", fontsize=6.8, color=col)

    ax.axvline(x_use, color=PURPLE, lw=1.0, ls=(0, (3.0, 2.0)), zorder=5)
    ax.plot([x_use], [life_use], marker="D", ms=5.0, mfc=PURPLE, mec="white",
            mew=0.9, zorder=9)
    ax.errorbar([x_use], [life_use], yerr=[[life_use - lo_use],
                                           [hi_use - life_use]],
                color=PURPLE, lw=1.1, capsize=3.0, zorder=8)
    ax.annotate(f"use condition {T_USE:.0f} $^\\circ$C:\n"
                f"{life_use / 8760.0:.0f} years "
                f"({lo_use / 8760.0:.0f} - {hi_use / 8760.0:.0f} at 95 %)",
                xy=(x_use, hi_use), xytext=(x_use - 0.012, hi_use * 4.5),
                fontsize=6.9, color=PURPLE, linespacing=1.45, ha="right",
                va="bottom",
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=PURPLE,
                                shrinkA=2, shrinkB=3))

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(200.0, 4.0e6)
    ax.set_xlabel("1000 / T  (1/K)", fontsize=8.4)
    ax.set_ylabel("time to failure (hours, log scale)", fontsize=8.4)
    ax.set_title("One straight line in 1/T, and a band that pays for the "
                 "extrapolation", fontsize=9.0, fontweight="bold", pad=16.0)
    sec = ax.secondary_xaxis("top", functions=(lambda v: 1000.0 / v - 273.15,
                                               lambda c: 1000.0 / (c + 273.15)))
    sec.set_xticks([160, 140, 120, 100, 80, 60])
    sec.set_xlabel("junction temperature ($^\\circ$C)", fontsize=7.6,
                   labelpad=1.5)
    sec.tick_params(labelsize=6.8)

    ax.text(0.022, 0.055,
            f"fitted slope gives $E_a$ = {ea:.2f} eV\n"
            f"acceleration factor {CELLS[0][0]:.0f} $^\\circ$C -> "
            f"{T_USE:.0f} $^\\circ$C: AF = {af:,.0f}x\n"
            f"{x.size} failures over {len(CELLS)} cells, within-cell "
            f"scatter {s:.2f} in ln(hours)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0,
            color=INK, linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.94))
    ax.legend(handles=[
        Line2D([], [], color=INK, lw=1.5, label="fit over all failures"),
        Line2D([], [], color=INK, lw=1.5, ls=(0, (4.0, 2.2)),
               label="extrapolation"),
        Line2D([], [], color="#d7dfe9", lw=6.0, label="95 % band, tested"),
        Line2D([], [], color="#e9edf3", lw=6.0, label="95 % band, beyond")],
        loc="lower right", frameon=False, fontsize=6.8, handlelength=1.9,
        labelspacing=0.34, borderaxespad=0.35)

    # ---- aux (b): cumulative failure per cell ------------------------------
    ax_c.grid(which="both", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_c.set_xscale("log")
    for temp_c, n, col in CELLS:
        v = np.sort(data[temp_c])
        f = 100.0 * (np.arange(1, n + 1) - 0.3) / (n + 0.4)   # median ranks
        ax_c.step(v, f, where="post", color=col, lw=1.4, zorder=5)
        ax_c.plot(v, f, ls="none", marker="o", ms=2.4, mfc=col, mec="white",
                  mew=0.4, zorder=6)
    ax_c.axhline(50.0, color=FAINT, lw=0.7, ls=(0, (2.4, 2.0)), zorder=3)
    ax_c.set_xlim(400.0, 4.0e5)
    ax_c.set_ylim(0.0, 104.0)
    ax_c.set_yticks([0, 25, 50, 75, 100])
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_xlabel("time on test (hours, log scale)", fontsize=7.5,
                    labelpad=1.5)
    ax_c.set_ylabel("cumulative failures (%)", fontsize=7.5)
    ax_c.set_title("Four cells, one shape, four positions", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_c.legend(handles=[Line2D([], [], color=c, lw=1.4,
                                label=f"{t:.0f} $^\\circ$C")
                         for t, _, c in CELLS],
                loc="upper left", frameon=False, fontsize=6.4,
                handlelength=1.5, labelspacing=0.28, borderaxespad=0.25)
    med_hot = float(np.median(data[CELLS[0][0]]))
    med_cold = float(np.median(data[CELLS[-1][0]]))
    ax_c.text(0.975, 0.055,
              f"medians {med_hot:,.0f} h and {med_cold:,.0f} h:\n"
              f"{med_cold / med_hot:.0f}x for {CELLS[0][0] - CELLS[-1][0]:.0f}"
              f" $^\\circ$C of stress",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)

    # ---- aux (c): the price of extrapolating -------------------------------
    ax_u.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
    temps = np.linspace(40.0, 150.0, 200)
    xg = 1000.0 / kelvin(temps)
    lo_g, hi_g, _ = band(xg, x, m, b, s, xbar, sxx, tcrit)
    factor = hi_g / lo_g
    ax_u.plot(temps, factor, color=INK, lw=1.7, zorder=5)
    ax_u.axvspan(CELLS[-1][0], CELLS[0][0], color="#e6ebf1", zorder=1)
    ax_u.text(0.5 * (CELLS[-1][0] + CELLS[0][0]), 0.955, "tested range",
              transform=ax_u.get_xaxis_transform(), ha="center", va="top",
              fontsize=6.4, color=SUB)
    f_use = float(np.interp(T_USE, temps, factor))
    ax_u.plot([T_USE], [f_use], marker="D", ms=5.0, mfc=PURPLE, mec="white",
              mew=0.9, zorder=7)
    ax_u.annotate(f"at {T_USE:.0f} $^\\circ$C the 95 % interval\n"
                  f"already spans {f_use:.1f}x",
                  xy=(T_USE, f_use), xytext=(0.34, 0.70),
                  textcoords="axes fraction", fontsize=6.4, color=PURPLE,
                  linespacing=1.45, ha="left", va="bottom",
                  arrowprops=dict(arrowstyle="-|>", lw=0.8, color=PURPLE,
                                  shrinkA=2, shrinkB=3))
    ax_u.set_xlim(temps.min(), temps.max())
    ax_u.set_ylim(1.0, factor.max() * 1.14)
    ax_u.tick_params(labelsize=6.8)
    ax_u.set_xlabel("use temperature the fit is read at ($^\\circ$C)",
                    fontsize=7.5, labelpad=1.5)
    ax_u.set_ylabel("width of the 95 % interval (x)", fontsize=7.5)
    ax_u.set_title("Uncertainty grows away from the cells", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_c, ax_u], dx=-0.62, dy=0.30)
    foot = add_footer(fig, "Synthetic data, seed=4")
    check_layout(fig, foot, [ax, ax_c, ax_u])
    savefig(fig, "910_rel_accelerated_life.png")


if __name__ == "__main__":
    main()
