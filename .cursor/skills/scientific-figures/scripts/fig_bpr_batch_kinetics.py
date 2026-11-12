"""Batch fermentation time course with the four classical growth phases.

The main panel integrates a Monod / Luedeking-Piret system for biomass,
substrate and product on one time axis, so the three curves cannot disagree:
the substrate is exhausted, the biomass plateau and the end of product
accumulation all fall on the same instant because the same state vector
produces them.  Background bands mark lag, exponential, stationary and
decline, each boundary read off the integrated solution rather than typed in.
The two supporting panels use the same colours: the observed specific growth
rate, which is flat only while substrate is in excess, and the yield
coefficients recovered as slopes of biomass and product against consumed
substrate.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

GREEN = "#2e7d5b"        # biomass
BLUE = "#3b5b92"         # substrate
RED = "#b23a48"          # product
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

MU_MAX = 0.42            # h^-1
KS = 0.35                # g/L
Y_XS = 0.48              # g biomass / g substrate
Y_PS = 0.40              # g product / g substrate
ALPHA = 0.18             # growth-associated product formation, g/g
BETA = 0.030             # non-growth product formation, g/(g h)
MS = 0.014               # maintenance coefficient, g/(g h)
KD_MAX = 0.080           # h^-1
K_LAG = 0.55             # h^-1, adaptation rate out of lag
K_STARVE = 0.40          # h^-1, onset rate of starvation-driven death
S0, X0, P0 = 22.0, 0.15, 0.0
T_END = 26.0

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 0.95, 4.62, 4.55)
TOP = (7.82, 3.61, 2.68, 1.89)
BOT = (7.82, 0.95, 2.68, 1.89)


def simulate():
    """Integrate X / S / P with a lag variable and a starvation variable."""

    def rhs(_t, y):
        x, s, _p, q, d = y
        s = max(s, 0.0)
        f = s / (KS + s)
        mu = MU_MAX * f * q
        qp = ALPHA * mu + BETA * f
        qs = mu / Y_XS + qp / Y_PS + MS * f
        starve = 1.0 / (1.0 + (s / 0.05) ** 2)
        return [(mu - KD_MAX * d) * x, -qs * x, qp * x,
                K_LAG * (1.0 - q), K_STARVE * (starve - d)]

    t = np.linspace(0.0, T_END, 2601)
    sol = solve_ivp(rhs, (0.0, T_END), [X0, S0, P0, 0.05, 0.0], t_eval=t,
                    rtol=1e-9, atol=1e-11, method="LSODA")
    return sol.t, sol.y[0], np.maximum(sol.y[1], 0.0), sol.y[2]


def phase_bounds(t, x, s):
    mu_obs = np.gradient(np.log(x), t)
    mu_peak = float(mu_obs.max())
    lag_end = float(t[np.argmax(mu_obs >= 0.90 * mu_peak)])
    exhaust = float(t[np.argmax(s <= 0.02 * S0)])
    i_max = int(np.argmax(x))
    after = np.arange(i_max, t.size)
    drop = after[x[after] <= 0.99 * x[i_max]]
    stat_end = float(t[drop[0]]) if drop.size else T_END
    return mu_obs, mu_peak, lag_end, exhaust, stat_end


def main_panel(ax, t, x, s, p, marks, bounds, rng):
    mu_obs, mu_peak, lag_end, exhaust, stat_end = bounds
    edges = [0.0, lag_end, exhaust, stat_end, T_END]
    names = ["lag", "exponential", "stationary", "decline"]
    tints = [lighten(GREY, 0.90), lighten(GREEN, 0.90),
             lighten(BLUE, 0.90), lighten(RED, 0.90)]
    for (a, b), name, tint in zip(zip(edges[:-1], edges[1:]), names, tints):
        ax.axvspan(a, b, facecolor=tint, edgecolor="none", zorder=0)
        narrow = (b - a) < 2.6
        ax.text((a + b) / 2.0, 0.985 if not narrow else 0.960,
                name, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=6.8, color=SUB,
                rotation=90 if narrow else 0)
        if b < T_END:
            ax.axvline(b, color=HAIR, lw=0.7, zorder=1)

    # Axes stack: bands (ax) < substrate < product < biomass, because
    # matplotlib sorts whole axes by zorder - an opaque band on a lower axes
    # would otherwise hide every twin drawn after it.
    ax_s = ax.twinx()
    ax_p = ax.twinx()
    ax_x = ax.twinx()
    ax_p.spines["right"].set_position(("outward", 44))
    for k, axis in enumerate([ax_s, ax_p, ax_x], start=2):
        axis.set_zorder(k)
        axis.patch.set_visible(False)
    ax.set_zorder(1)

    ax_x.plot(t, x, color=GREEN, lw=1.9, zorder=5)
    ax_s.plot(t, s, color=BLUE, lw=1.9, zorder=5)
    ax_p.plot(t, p, color=RED, lw=1.9, zorder=5)

    for axis, series, color in [(ax_x, x, GREEN), (ax_s, s, BLUE),
                                (ax_p, p, RED)]:
        vals = np.interp(marks, t, series)
        noise = rng.normal(0.0, 0.022, marks.size) * max(series.max(), 1e-9)
        axis.plot(marks, vals + noise, ls="none", marker="o", ms=3.0,
                  mfc="white", mec=color, mew=0.9, zorder=6)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, x.max() * 1.42)
    ax_x.set_ylim(*ax.get_ylim())
    ax_s.set_ylim(0.0, S0 * 1.42)
    ax_p.set_ylim(0.0, p.max() * 1.42)

    ax.set_xlabel("fermentation time (h)")
    ax.set_ylabel("biomass X (g/L)", color=GREEN)
    ax_s.set_ylabel("substrate S (g/L)", color=BLUE)
    ax_p.set_ylabel("product P (g/L)", color=RED)
    ax.tick_params(axis="y", colors=GREEN)
    ax.spines["left"].set_color(GREEN)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for axis, color in [(ax_s, BLUE), (ax_p, RED)]:
        axis.spines["top"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_color(color)
        axis.tick_params(axis="y", colors=color)
    ax_x.spines["top"].set_visible(False)
    ax_x.spines["right"].set_visible(False)
    ax_x.set_yticks([])

    x_end = float(np.interp(exhaust, t, x))
    p_end = float(np.interp(exhaust, t, p))
    ax_x.axvline(exhaust, color=INK, lw=1.0, ls=(0, (4, 2.4)), zorder=7)
    ax_x.annotate(
        "S exhausted at t = {:.1f} h\nX plateau {:.2f} g/L,  P stops at "
        "{:.2f} g/L".format(exhaust, x.max(), p.max()),
        xy=(exhaust, x_end), xytext=(exhaust + 1.4, x.max() * 1.33),
        ha="left", va="top", fontsize=7.0, color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax_x.plot([exhaust], [x_end], marker="o", ms=4.4, color=GREEN, zorder=8)
    ax_p.plot([exhaust], [p_end], marker="o", ms=4.4, color=RED, zorder=8)

    handles = [Line2D([], [], color=GREEN, lw=1.9, label="biomass X (g/L)"),
               Line2D([], [], color=BLUE, lw=1.9, label="substrate S (g/L)"),
               Line2D([], [], color=RED, lw=1.9, label="product P (g/L)"),
               Line2D([], [], color=GREY, lw=0.0, marker="o", ms=3.0,
                      mfc="white", mec=GREY, label="offline samples")]
    ax_x.legend(handles=handles, loc="upper left", fontsize=7.0,
                frameon=False, bbox_to_anchor=(0.020, 0.815),
                handlelength=1.5, borderaxespad=0.0, labelspacing=0.32)
    ax.set_title("Depletion, growth arrest and product plateau coincide",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    return exhaust


def rate_panel(ax, t, mu_obs, bounds):
    _mu, mu_peak, lag_end, exhaust, _stat = bounds
    win = (t >= lag_end) & (t <= exhaust - 0.6)
    plateau = float(mu_obs[win].mean())
    ax.axvspan(lag_end, exhaust, facecolor=lighten(GREEN, 0.90),
               edgecolor="none", zorder=0)
    ax.plot(t, mu_obs, color=GREEN, lw=1.7, zorder=4)
    ax.axhline(plateau, color=INK, lw=0.9, ls=(0, (3.4, 2.2)), zorder=3)
    ax.axhline(0.0, color=GREY, lw=0.7, zorder=2)
    ax.text(T_END - 0.4, plateau + 0.018,
            r"exponential plateau $\mu$ = {:.3f} h$^{{-1}}$".format(plateau),
            ha="right", va="bottom", fontsize=6.8, color=INK)
    ax.text(exhaust + 0.7, mu_peak * 0.16,
            "S depleted:\nnet rate turns negative", ha="left", va="bottom",
            fontsize=6.6, color=SUB)
    ax.set_xlim(0.0, T_END)
    ax.set_ylim(mu_obs.min() * 1.30, mu_peak * 1.32)
    ax.set_xlabel("time (h)", fontsize=7.6)
    ax.set_ylabel(r"$\mu$ = d(ln X)/dt (h$^{-1}$)", fontsize=7.6,
                  color=GREEN)
    ax.tick_params(labelsize=6.8)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Observed specific growth rate", fontsize=8.4,
                 fontweight="bold", color=INK)
    return plateau


def yield_panel(ax, t, x, s, p, bounds):
    _mu, _peak, lag_end, exhaust, _stat = bounds
    win = (t >= lag_end) & (t <= exhaust - 0.6)
    used = S0 - s
    exp_xs = float(np.polyfit(used[win], x[win], 1)[0])
    exp_ps = float(np.polyfit(used[win], p[win], 1)[0])

    groups = [r"$Y_{X/S}$", r"$Y_{P/S}$"]
    app_vals = [exp_xs, exp_ps]
    sto_vals = [Y_XS, Y_PS]
    pos = np.arange(2, dtype=float)
    wid = 0.34
    for off, vals, hatched, name in [
            (-wid / 2, app_vals, False, "apparent, slope on consumed S"),
            (+wid / 2, sto_vals, True, "stoichiometric model coefficient")]:
        for k, (px, v, c) in enumerate(zip(pos + off, vals, [GREEN, RED])):
            ax.bar(px, v, width=wid, color="white" if hatched else c,
                   edgecolor=c, linewidth=0.9,
                   hatch="////" if hatched else None, zorder=3,
                   label=name if k == 0 else None)
            ax.text(px, v + 0.010, "{:.3f}".format(v), ha="center",
                    va="bottom", fontsize=6.4, color=c)
        if hatched:
            for px, v, c in zip(pos + off, vals, [GREEN, RED]):
                ax.plot([px - wid / 2, px + wid / 2], [v, v], color=c,
                        lw=1.0, zorder=4)

    ax.set_xticks(pos)
    ax.set_xticklabels(groups, fontsize=8.0)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, max(app_vals + sto_vals) * 1.42)
    ax.set_ylabel("yield coefficient (g/g)", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper center",
              handlelength=1.3, borderaxespad=0.15, ncol=1)
    ax.set_title("Apparent yields fall short of stoichiometry",
                 fontsize=8.4, fontweight="bold", color=INK)
    return exp_xs, exp_ps


def main():
    set_pub_style()
    rng = np.random.default_rng(4)
    t, x, s, p = simulate()
    bounds = phase_bounds(t, x, s)
    marks = np.arange(1.0, T_END, 2.0)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_rate = axes_inches(fig, TOP)
    ax_yield = axes_inches(fig, BOT)

    exhaust = main_panel(ax_main, t, x, s, p, marks, bounds, rng)
    plateau = rate_panel(ax_rate, t, bounds[0], bounds)
    y_xs, y_ps = yield_panel(ax_yield, t, x, s, p, bounds)

    panel_labels_inches(fig, [ax_main, ax_rate, ax_yield], dx=-0.62, dy=0.14)
    savefig(fig, "876_bpr_batch_kinetics.png")


if __name__ == "__main__":
    main()
