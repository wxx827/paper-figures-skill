"""Modal split: twenty years of share evolution, trip lengths and switching.

Panel (a) is a directly-labelled stacked area of the five main modes, so
no separate legend is needed anywhere in the figure.  The two shorter
panels on the right add the distributions behind the shares - trip-length
violins on a log axis - and where the switching happened, as a 2015 to
2025 mode-transition matrix.  Mode colours are the series-wide key reused
by the other transport figures.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde
from pubstyle import set_pub_style, savefig

# ---- series-wide mode colour key -----------------------------------------
MODES = ["Walk", "Cycle", "Bus", "Metro", "Car"]
MCOL = {"Walk": "#8a7aa3", "Cycle": "#6f9464", "Bus": "#d3a04c",
        "Metro": "#40708f", "Car": "#b06a52"}
INK = "#2b2b2b"
GREY = "#8c8c8c"

CTRL_YEARS = np.array([2005, 2010, 2015, 2020, 2025])
CTRL = {                       # % of daily trips at the control years
    "Walk": [29.0, 26.5, 24.0, 23.0, 22.0],
    "Cycle": [13.0, 10.0, 8.5, 11.0, 14.5],
    "Bus": [24.0, 22.0, 19.5, 17.0, 15.5],
    "Metro": [6.0, 10.5, 16.0, 20.0, 23.5],
    "Car": [28.0, 31.0, 32.0, 29.0, 24.5],
}
DIST = {                       # log-normal trip length, km
    "Walk": (0.72, 0.50), "Cycle": (2.20, 0.55), "Bus": (4.60, 0.58),
    "Metro": (8.60, 0.52), "Car": (7.20, 0.82),
}


def panel_tag(fig, ax, tag, title, dx=-0.44, dy=0.14):
    pos = ax.get_position()
    fig.text(pos.x0 + dx / fig.get_figwidth(),
             pos.y1 + dy / fig.get_figheight(), tag, fontsize=10.5,
             fontweight="bold", ha="left", va="baseline", color=INK)
    fig.text(pos.x0 + (dx + 0.35) / fig.get_figwidth(),
             pos.y1 + dy / fig.get_figheight(), title, fontsize=8.8,
             fontweight="bold", ha="left", va="baseline", color="#33474f")


def main():
    set_pub_style()
    rng = np.random.default_rng(5)

    yr = np.linspace(2005, 2025, 241)
    share = np.vstack([PchipInterpolator(CTRL_YEARS, CTRL[m])(yr)
                       for m in MODES])
    share = 100.0 * share / share.sum(axis=0)

    fig = plt.figure(figsize=(10.6, 6.05))
    axa = fig.add_axes([0.052, 0.115, 0.452, 0.775])
    axv = fig.add_axes([0.588, 0.585, 0.330, 0.305])
    axm = fig.add_axes([0.588, 0.115, 0.330, 0.305])
    cax = fig.add_axes([0.930, 0.115, 0.011, 0.305])

    # ------------------------------------------------- (a) stacked shares
    axa.stackplot(yr, share, colors=[MCOL[m] for m in MODES], lw=0.0,
                  zorder=2)
    cum = np.cumsum(share, axis=0)
    lo = np.vstack([np.zeros_like(yr), cum[:-1]])
    for i in range(1, len(MODES)):
        axa.plot(yr, cum[i - 1], color="white", lw=1.0, zorder=3)

    for i, m in enumerate(MODES):
        mid = 0.5 * (lo[i, -1] + cum[i, -1])
        axa.text(2024.6, mid, f"{m}\n{share[i, -1]:.0f}%", ha="right",
                 va="center", fontsize=7.6, fontweight="bold", color="white",
                 linespacing=1.25, zorder=6)
        mid0 = 0.5 * (lo[i, 0] + cum[i, 0])
        axa.text(2005.35, mid0, f"{share[i, 0]:.0f}%", ha="left", va="center",
                 fontsize=6.8, color="white", zorder=6, alpha=0.92)

    # headroom above 100 % keeps the event call-outs inside the axes
    for xv, lab in [(2013, "metro line 3 opens"),
                    (2018, "bike-share rollout")]:
        axa.plot([xv, xv], [0, 100], color="white", lw=0.9, ls=(0, (3, 2)),
                 zorder=5, alpha=0.85)
        axa.plot([xv, xv], [100, 103.5], color="#b8bcc0", lw=0.7, zorder=5)
        axa.text(xv, 105.0, lab, fontsize=6.8, color="#5a6570", ha="center",
                 va="bottom", zorder=6)
    axa.set_xlim(2005, 2025)
    axa.set_ylim(0, 116)
    axa.set_yticks(np.arange(0, 101, 20))
    axa.set_xticks(np.arange(2005, 2026, 5))
    axa.set_xlabel("year", labelpad=2)
    axa.set_ylabel("share of daily trips  (%)", labelpad=3)
    axa.tick_params(length=2.6)
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)

    # ------------------------------------------------ (b) length violins
    grid = np.linspace(np.log10(0.15), np.log10(70.0), 220)
    for i, m in enumerate(MODES):
        med, sg = DIST[m]
        d = rng.lognormal(np.log(med), sg, 4000)
        lg = np.log10(d)
        kde = gaussian_kde(lg, bw_method=0.28)
        dens = kde(grid)
        dens = 0.40 * dens / dens.max()
        axv.fill_betweenx(grid, i - dens, i + dens, facecolor=MCOL[m],
                          alpha=0.55, lw=0.0, zorder=2)
        axv.plot(i - dens, grid, color=MCOL[m], lw=0.8, zorder=3)
        axv.plot(i + dens, grid, color=MCOL[m], lw=0.8, zorder=3)
        q1, q2, q3 = np.percentile(lg, [25, 50, 75])
        axv.plot([i, i], [q1, q3], color="white", lw=2.6, zorder=4,
                 solid_capstyle="round")
        axv.plot([i, i], [q1, q3], color=INK, lw=1.0, zorder=5,
                 solid_capstyle="round")
        axv.plot([i], [q2], marker="o", ms=3.2, mfc="white", mec=INK,
                 mew=0.9, zorder=6)
        axv.text(i, np.log10(88.0), f"{10 ** q2:.1f}", ha="center",
                 va="center", fontsize=6.8, fontweight="bold", color=MCOL[m])
    axv.text(-1.02, np.log10(88.0), "median km", ha="left", va="center",
             fontsize=6.4, color=GREY)
    axv.axhline(np.log10(46.0), color="#e2e2e2", lw=0.6, zorder=1)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20]
    axv.set_yticks(np.log10(ticks))
    axv.set_yticklabels([str(t) for t in ticks], fontsize=6.8)
    axv.set_ylim(np.log10(0.15), np.log10(150.0))
    axv.set_xlim(-1.15, len(MODES) - 0.4)
    axv.set_xticks(range(len(MODES)))
    axv.set_xticklabels(MODES, fontsize=7.2)
    for lab, m in zip(axv.get_xticklabels(), MODES):
        lab.set_color(MCOL[m]); lab.set_fontweight("bold")
    axv.set_ylabel("trip length  (km)", labelpad=2, fontsize=7.6)
    axv.tick_params(length=2.4)
    axv.grid(axis="y", color="#ededed", lw=0.5, zorder=0)
    for sp in ("top", "right"):
        axv.spines[sp].set_visible(False)

    # ------------------------------------------- (c) 2015 -> 2025 switching
    stay = np.array([0.72, 0.63, 0.55, 0.83, 0.61])
    pull = np.array([0.10, 0.16, 0.09, 0.42, 0.23])     # attractiveness
    T = np.zeros((5, 5))
    for i in range(5):
        w = pull.copy()
        w[i] = 0.0
        w = w / w.sum() * (1.0 - stay[i])
        w[i] = stay[i]
        T[i] = w
    T = 100.0 * T

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "sw", ["#f7f6f3", "#dfe6ea", "#b5c8d3", "#7fa2b6", "#4c7893",
               "#2b5270"])
    im = axm.imshow(T, cmap=cmap, vmin=0, vmax=85, origin="upper",
                    extent=(0, 5, 5, 0), interpolation="nearest", zorder=2)
    for i in range(5):
        for j in range(5):
            axm.text(j + 0.5, i + 0.5, f"{T[i, j]:.0f}", ha="center",
                     va="center", fontsize=6.6,
                     fontweight="bold" if i == j else "normal",
                     color="white" if T[i, j] > 42 else "#3a3a3a", zorder=4)
    for k in range(1, 5):
        axm.axhline(k, color="white", lw=0.8, zorder=3)
        axm.axvline(k, color="white", lw=0.8, zorder=3)
    axm.set_xticks(np.arange(5) + 0.5)
    axm.set_yticks(np.arange(5) + 0.5)
    axm.set_xticklabels(MODES, fontsize=7.0)
    axm.set_yticklabels(MODES, fontsize=7.0)
    for lab, m in zip(axm.get_xticklabels(), MODES):
        lab.set_color(MCOL[m]); lab.set_fontweight("bold")
    for lab, m in zip(axm.get_yticklabels(), MODES):
        lab.set_color(MCOL[m]); lab.set_fontweight("bold")
    axm.tick_params(length=0, pad=2.0)
    axm.set_xlabel("main mode in 2025", fontsize=7.6, labelpad=3)
    axm.set_ylabel("main mode in 2015", fontsize=7.6, labelpad=3)
    axm.set_aspect("auto")
    for sp in axm.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7)
        sp.set_edgecolor("#b9b5ac")

    cb = fig.colorbar(im, cax=cax, ticks=[0, 20, 40, 60, 80])
    cb.set_label("% of 2015 users", fontsize=7.0, labelpad=3)
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.7)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#555555")

    panel_tag(fig, axa, "(a)", "Modal split, 2005-2025")
    panel_tag(fig, axv, "(b)", "Trip-length distribution")
    panel_tag(fig, axm, "(c)", "Mode switching, 2015 to 2025")

    savefig(fig, "207_mode_share.png")


if __name__ == "__main__":
    main()
