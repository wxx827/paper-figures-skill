"""Disease progress: three cultivars, two growth models, one weather record.

The main panel is the epidemic itself - scouted disease index against days
after inoculation, with a logistic and a Gompertz curve fitted to the same
points so the reader can see which shape the epidemic actually has.  Every
number quoted comes out of those fits or out of the plotted points: the rate
r is the fitted logistic parameter, AUDPC is trapezoidal on the observations,
and the R-squared decides which model the title names.  Panel (b) is the
driver, sharing the main time axis, and panel (c) collapses the epidemic to
the single number a breeding programme compares cultivars on.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_label_inches, shared_limits

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
DEEP_RED = "#8c2f2f"
BLUE = "#3b5b92"
EARTH = "#c0603f"
GREY = "#808080"

W, H = 10.4, 6.4
DAYS = 70.0
SCOUT_EVERY = 5.0
N_REP = 4
DEG = r"$^{\circ}$C"
AUDPC_UNIT = r"%$\cdot$d"

# (cultivar, plateau %, logistic rate 1/d, mid-point d, colour)
CULTIVARS = [
    ("cv. Suweon (susceptible)", 88.0, 0.132, 31.0, RED),
    ("cv. Milyang (moderate)", 60.0, 0.096, 39.0, GOLD),
    ("cv. Tetep (resistant)", 23.0, 0.072, 45.0, GREEN),
]
RH_INFECT = 90.0                 # leaf-wetness proxy, % relative humidity
T_INFECT = (18.0, 27.0)          # infection temperature window, deg C


def logistic(t, ymax, r, t50):
    return ymax / (1.0 + np.exp(-r * (t - t50)))


def gompertz(t, ymax, k, ti):
    return ymax * np.exp(-np.exp(-k * (t - ti)))


def r_squared(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def observe(rng):
    t = np.arange(0.0, DAYS + 0.1, SCOUT_EVERY)
    out = []
    for name, ymax, r, t50, col in CULTIVARS:
        truth = logistic(t, ymax, r, t50)
        reps = np.clip(truth[None, :] + rng.normal(0.0, 2.6 + 0.045 * truth,
                                                   (N_REP, t.size)),
                       0.0, 100.0)
        out.append((name, col, reps.mean(axis=0),
                    reps.std(axis=0, ddof=1) / np.sqrt(N_REP)))
    return t, out


def weather(rng):
    d = np.arange(0.0, DAYS + 0.1, 1.0)
    temp = (23.6 + 4.2 * np.sin(2.0 * np.pi * (d - 8.0) / 26.0) +
            rng.normal(0.0, 1.5, d.size))
    rh = np.clip(82.0 + 12.0 * np.sin(2.0 * np.pi * (d - 3.0) / 11.0) +
                 rng.normal(0.0, 5.5, d.size), 45.0, 100.0)
    ev = (rh >= RH_INFECT) & (temp >= T_INFECT[0]) & (temp <= T_INFECT[1])
    return d, temp, rh, ev


# ------------------------------------------------------------------- panels
def progress_panel(ax, t, obs, dense):
    stats_rows = []
    for (name, col, y, se) in obs:
        p0_l = [max(y.max(), 5.0), 0.12, 35.0]
        pl, _ = curve_fit(logistic, t, y, p0=p0_l, maxfev=20000)
        pg, _ = curve_fit(gompertz, t, y, p0=[p0_l[0], 0.08, 30.0],
                          maxfev=20000)
        r2l = r_squared(y, logistic(t, *pl))
        r2g = r_squared(y, gompertz(t, *pg))
        audpc = float(np.trapezoid(y, t))

        ax.errorbar(t, y, yerr=se, ls="none", marker="o", ms=3.6, mfc="white",
                    mew=1.0, mec=col, ecolor=col, elinewidth=0.8, capsize=1.8,
                    zorder=6)
        ax.plot(dense, logistic(dense, *pl), color=col, lw=1.8, zorder=5)
        ax.plot(dense, gompertz(dense, *pg), color=col, lw=1.1,
                ls=(0, (4.0, 2.2)), zorder=5)
        stats_rows.append((name, col, pl[1], audpc, r2l, r2g))

    ax.text(0.015, 0.600, "logistic rate and AUDPC (%s)" % AUDPC_UNIT,
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6,
            color=SUB)
    for i, (name, col, r, audpc, r2l, r2g) in enumerate(stats_rows):
        ax.text(0.015, 0.528 - 0.070 * i,
                "%s:  $r$ = %.3f /d,  %.0f"
                % (name.split(" (")[0].replace("cv. ", ""), r, audpc),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
                color=col)

    handles = [Line2D([], [], color=c, lw=1.8, marker="o", ms=3.6,
                      mfc="white", mew=1.0, mec=c, label=n)
               for n, _, _, _, c in CULTIVARS]
    handles += [Line2D([], [], color=GREY, lw=1.6, label="logistic fit"),
                Line2D([], [], color=GREY, lw=1.1, ls=(0, (4.0, 2.2)),
                       label="Gompertz fit")]
    ax.legend(handles=handles, fontsize=7.0, frameon=False, loc="upper left",
              bbox_to_anchor=(0.012, 1.015), handlelength=1.8)

    ax.set_xlim(-1.5, DAYS + 1.5)
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("days after inoculation (d)", fontsize=8.0)
    ax.set_ylabel("disease index (%)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    best = "logistic" if np.mean([s[4] for s in stats_rows]) >= \
        np.mean([s[5] for s in stats_rows]) else "Gompertz"
    ax.set_title("The %s model fits all three cultivars "
                 "($R^2$ = %.3f mean)" % (best,
                 np.mean([s[4] if best == "logistic" else s[5]
                          for s in stats_rows])),
                 fontsize=9.0, fontweight="bold", color=INK)
    return stats_rows, best


def weather_panel(ax, d, temp, rh, ev):
    for lim in T_INFECT:
        ax.axhline(lim, color=EARTH, lw=0.7, ls=(0, (2.4, 2.0)), zorder=1)
    ax.plot(d, temp, color=EARTH, lw=1.3, zorder=4, label="mean temperature")
    ax.set_ylabel("temperature (%s)" % DEG, color=EARTH, fontsize=7.4)
    ax.set_ylim(10.0, 35.0)
    ax.set_yticks([15, 20, 25, 30])
    ax.tick_params(axis="y", colors=EARTH, labelsize=6.5)
    ax.spines["left"].set_color(EARTH)

    ax2 = ax.twinx()
    ax2.plot(d, rh, color=BLUE, lw=1.1, alpha=0.9, zorder=3,
             label="relative humidity")
    ax2.axhline(RH_INFECT, color=BLUE, lw=0.8, ls=(0, (3.2, 2.0)), zorder=2)
    ax2.set_ylabel("relative humidity (%)", color=BLUE, fontsize=7.4)
    ax2.set_ylim(36.0, 122.0)
    ax2.set_yticks([50, 70, 90])
    ax2.tick_params(axis="y", colors=BLUE, labelsize=6.5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(BLUE)

    ax.plot(d[ev], np.full(int(np.count_nonzero(ev)), 33.4), ls="none",
            marker="|", ms=6.0, mew=1.2, color=DEEP_RED, zorder=6,
            label="infection event (n = %d)" % int(np.count_nonzero(ev)))
    ax.set_xlim(-1.5, DAYS + 1.5)
    ax.set_xlabel("days after inoculation (d)", fontsize=7.4)
    ax.tick_params(axis="x", labelsize=6.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.4, frameon=False,
              loc="lower left", bbox_to_anchor=(0.0, -0.03), ncol=1,
              handlelength=1.3)
    ax.set_title("Wet spells inside the %.0f - %.0f %s window drive it"
                 % (T_INFECT[0], T_INFECT[1], DEG), fontsize=8.4,
                 fontweight="bold", color=INK)
    return int(np.count_nonzero(ev))


def audpc_panel(ax, stats_rows):
    names = [s[0].split(" (")[0].replace("cv. ", "") for s in stats_rows]
    vals = [s[3] for s in stats_rows]
    cols = [s[1] for s in stats_rows]
    x = np.arange(len(vals), dtype=float)
    ax.bar(x, vals, width=0.58, color=cols, edgecolor="white", lw=0.8,
           zorder=3)
    top = shared_limits([vals], pad=0.0)[1]
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.025 * top, "%.0f" % v, ha="center", va="bottom",
                fontsize=6.8, color=INK)
    worst = max(vals)
    for xi, v in zip(x[1:], vals[1:]):
        ax.text(xi, v * 0.5, "-%.0f %%" % (100.0 * (1.0 - v / worst)),
                ha="center", va="center", fontsize=6.6, color="white",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylim(0.0, top * 1.20)
    ax.set_ylabel("AUDPC (%s)" % AUDPC_UNIT, fontsize=7.4)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.6)
    ax.set_title("Resistance cuts the epidemic by %.0f %%"
                 % (100.0 * (1.0 - min(vals) / worst)), fontsize=8.4,
                 fontweight="bold", color=INK)
    return worst, min(vals)


def main():
    set_pub_style()
    rng = np.random.default_rng(7412)
    t, obs = observe(rng)
    d, temp, rh, ev = weather(rng)
    dense = np.linspace(0.0, DAYS, 400)

    fig = plt.figure(figsize=(W, H))
    ax_main = axes_inches(fig, (0.92, 1.00, 4.88, 4.12))
    ax_wx = axes_inches(fig, (7.10, 3.52, 2.80, 1.72))
    ax_au = axes_inches(fig, (7.10, 0.95, 2.80, 1.62))

    stats_rows, best = progress_panel(ax_main, t, obs, dense)
    n_ev = weather_panel(ax_wx, d, temp, rh, ev)
    worst, bestv = audpc_panel(ax_au, stats_rows)

    fig.text(0.016, 6.14 / H,
             "Same %d infection events, %.1f-fold difference in AUDPC "
             "between the susceptible and the resistant cultivar"
             % (n_ev, worst / bestv),
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.016, 0.085 / H, textwrap.fill(
             "Points are %d replicate plots (mean +/- SE); solid and dashed "
             "curves are logistic and Gompertz fits to those points, and the "
             "%s form wins on $R^2$. AUDPC is trapezoidal over the %.0f-day "
             "record. Infection events are days with RH >= %.0f %% inside "
             "the %.0f - %.0f %s window."
             % (N_REP, best, DAYS, RH_INFECT, T_INFECT[0], T_INFECT[1], DEG),
             195),
             ha="left", va="bottom", fontsize=6.5, color=MUTED,
             linespacing=1.5)

    panel_label_inches(fig, ax_main, "(a)", dx=-0.72, dy=0.16)
    panel_label_inches(fig, ax_wx, "(b)", dx=-0.62, dy=0.16)
    panel_label_inches(fig, ax_au, "(c)", dx=-0.62, dy=0.16)

    savefig(fig, "741_pla_disease_progress.png")


if __name__ == "__main__":
    main()
