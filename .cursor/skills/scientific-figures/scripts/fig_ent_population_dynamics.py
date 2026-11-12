"""Pest population dynamics: three generations a year, one lagging predator.

The main panel is the monitoring record a control decision is actually made
from - four seasons of larval density, the generation boundaries that split
each season into three cohorts, and the economic threshold with every week
that breached it shaded.  The predator series on the right axis is not drawn
independently: it is the prey series passed through a delayed recruitment
kernel, so the phase lag printed on the panel is measured back out of the two
plotted arrays by cross-correlation rather than asserted.  Panel (b) puts the
four seasons in the context of the whole record, and panel (c) is the spatial
snapshot behind the single worst week.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_label_inches, safe_colorbar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
RED = "#b23a48"
DEEP_RED = "#8c2f2f"
BLUE = "#3b5b92"
GREY = "#808080"

# House-red sequential ramp, monotone in lightness (DESIGN_SYSTEM 2.2).
RED_RAMP = ["#faf2f3", "#eecdd2", "#dda3ac", "#c67785", "#a9505f", "#7d2c39"]
PEST_CMAP = mpl.colors.LinearSegmentedColormap.from_list("house_red", RED_RAMP)

W, H = 10.4, 6.4
WPY = 52                        # weeks per year
N_SHOWN = 4                     # seasons drawn in the main panel
N_RECORD = 12                   # seasons in the monitoring archive
FIRST_SHOWN = 2019
THRESHOLD = 3.0                 # economic threshold, individuals/plant
GEN_EDGES = (18.0, 26.0, 34.0, 42.0)     # week of year
GEN_PEAKS = (22.0, 30.0, 38.0)
GEN_WIDTH = (2.5, 2.8, 3.1)
WINTER = 0.10                   # off-season carry-over, individuals/plant
DEG = r"$^{\circ}$C"
DEGD = r"$^{\circ}$C$\cdot$d"

GRID = (12, 18)                 # field snapshot, rows x cols
PLOT_M = 8.0                    # side of one sampling plot, m


def season_scale(dd):
    """Multiplier turning a season's accumulated heat into peak pressure."""
    return (np.asarray(dd, dtype=float) / 1500.0) ** 2.4


def build_series(dd_shown):
    t = np.arange(0.0, N_SHOWN * WPY + 1.0)
    dens = np.full(t.size, WINTER)
    for yi, dd in enumerate(dd_shown):
        woy = t - yi * WPY
        for k, (c, wd) in enumerate(zip(GEN_PEAKS, GEN_WIDTH)):
            amp = 1.55 * (1.0 + 0.62 * k) * season_scale(dd)
            dens += amp * np.exp(-0.5 * ((woy - c) / wd) ** 2)
    kern = np.exp(-0.5 * ((np.arange(0.0, 12.0) - 3.2) / 2.1) ** 2)
    kern /= kern.sum()
    enemy = 0.16 + 0.34 * np.convolve(dens, kern)[:t.size]
    return t, dens, enemy


def measured_lag(dens, enemy, max_lag=12):
    a = dens - dens.mean()
    b = enemy - enemy.mean()
    best, best_r = 0, -2.0
    for lag in range(max_lag + 1):
        x = a[:a.size - lag] if lag else a
        y = b[lag:]
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best_r:
            best, best_r = lag, r
    return best, best_r


def field_snapshot(rng, mean_target):
    rows, cols = GRID
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    fld = np.zeros((rows, cols))
    for _ in range(5):
        cy, cx = rng.uniform(0, rows), rng.uniform(0, cols)
        sy, sx = rng.uniform(1.6, 3.4), rng.uniform(2.0, 4.6)
        fld += rng.uniform(0.6, 1.0) * np.exp(
            -0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    fld += 0.16 * rng.random((rows, cols))
    return fld * (mean_target / fld.mean())


# ------------------------------------------------------------------- panels
def main_panel(ax, t, dens, enemy, dd_shown):
    ax.axhspan(0.0, THRESHOLD, facecolor="#f5f5f5", edgecolor="none",
               zorder=0)
    over = dens >= THRESHOLD
    ax.fill_between(t, THRESHOLD, dens, where=over, interpolate=True,
                    facecolor=RED, alpha=0.20, linewidth=0.0, zorder=2)

    for yi in range(N_SHOWN):
        for e in GEN_EDGES:
            ax.axvline(yi * WPY + e, color=HAIR, lw=0.7, ls=(0, (1.8, 2.0)),
                       zorder=1)
    for k, c in enumerate(GEN_PEAKS):
        ax.text(c, 0.42, "G%d" % (k + 1), ha="center", va="center",
                fontsize=6.4, color=MUTED, zorder=4,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="#f5f5f5",
                          edgecolor="none"))

    ax.axhline(THRESHOLD, color=DEEP_RED, lw=1.0, ls=(0, (4.5, 2.4)),
               zorder=5)
    ax.text(1.5, THRESHOLD - 0.14, "threshold %.1f" % THRESHOLD, ha="left",
            va="top", fontsize=6.6, color=DEEP_RED, zorder=6)

    ax.plot(t, dens, color=RED, lw=1.7, zorder=6, label="larval density")
    n_over = int(np.count_nonzero(over))
    ax.plot(t[over], dens[over], ls="none", marker="o", ms=2.4, color=DEEP_RED,
            zorder=7, label="week above threshold (n = %d)" % n_over)

    ax2 = ax.twinx()
    ax2.plot(t, enemy, color=BLUE, lw=1.5, zorder=5, label="predator density")
    ax2.set_ylabel("predator density (individuals/plant)", color=BLUE,
                   fontsize=8.0)
    ax2.set_ylim(0.0, float(enemy.max()) * 1.45)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(BLUE)
    ax2.tick_params(axis="y", colors=BLUE, labelsize=7.0)

    lag, lag_r = measured_lag(dens, enemy)
    peak_i = int(np.argmax(dens))
    ax.annotate("predator peak trails prey by %d weeks\n"
                "(cross-correlation $r$ = %.2f)" % (lag, lag_r),
                xy=(t[peak_i] + lag, dens[peak_i] * 0.52),
                xytext=(t[peak_i] + 13.0, dens[peak_i] * 0.92),
                ha="left", va="center", fontsize=6.8, color=BLUE,
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))

    ax.set_xticks([yi * WPY + WPY / 2.0 for yi in range(N_SHOWN)])
    ax.set_xticklabels([str(FIRST_SHOWN + yi) for yi in range(N_SHOWN)],
                       fontsize=7.2)
    ax.set_xticks([yi * WPY for yi in range(N_SHOWN + 1)], minor=True)
    ax.set_xlim(0.0, float(t[-1]))
    ax.set_ylim(0.0, float(dens.max()) * 1.20)
    ax.set_xlabel("monitoring season (week resolution)", fontsize=8.0)
    ax.set_ylabel("larval density (individuals/plant)", color=RED,
                  fontsize=8.0)
    ax.tick_params(axis="y", colors=RED, labelsize=7.0)
    ax.spines["left"].set_color(RED)
    ax.grid(axis="y", lw=0.5, alpha=0.6)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, frameon=False, loc="upper left",
              bbox_to_anchor=(0.012, 1.015), handlelength=1.6)
    ax.set_title("Four seasons, %d generations, %d weeks over threshold"
                 % (N_SHOWN * len(GEN_PEAKS), n_over), fontsize=9.0,
                 fontweight="bold", color=INK)
    return n_over, lag, peak_i


def scatter_panel(ax, dd_all, peaks_all, n_shown):
    old = slice(0, len(dd_all) - n_shown)
    new = slice(len(dd_all) - n_shown, None)
    fit = stats.linregress(dd_all, peaks_all)
    r, p = stats.pearsonr(dd_all, peaks_all)
    xg = np.linspace(min(dd_all) - 40.0, max(dd_all) + 40.0, 50)
    ax.plot(xg, fit.intercept + fit.slope * xg, color=GREY, lw=1.0,
            ls=(0, (4.0, 2.2)), zorder=3)
    ax.plot(dd_all[old], peaks_all[old], ls="none", marker="o", ms=4.0,
            mfc="white", mew=1.0, mec=GREY, zorder=4,
            label="archive seasons")
    ax.plot(dd_all[new], peaks_all[new], ls="none", marker="o", ms=4.6,
            color=RED, zorder=5, label="seasons in (a)")
    ax.axhline(THRESHOLD, color=DEEP_RED, lw=0.8, ls=(0, (3.0, 2.0)),
               zorder=2)

    ax.set_xlabel("season heat sum (%s)" % DEGD, fontsize=7.4)
    ax.set_ylabel("annual peak (ind./plant)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ptxt = "$p$ < 0.001" if p < 0.001 else "$p$ = %.3f" % p
    ax.text(0.035, 0.96, "$r$ = %.2f, %s, n = %d" % (r, ptxt, len(dd_all)),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
            color=INK)
    ax.legend(fontsize=6.4, frameon=False, loc="lower right",
              bbox_to_anchor=(1.0, -0.02), handlelength=1.2)
    ax.set_title("Warm seasons peak higher", fontsize=8.4,
                 fontweight="bold", color=INK)
    return r


def map_panel(fig, ax, fld, week_label):
    im = ax.imshow(fld, cmap=PEST_CMAP, origin="lower", interpolation="nearest",
                   vmin=0.0, vmax=float(fld.max()))
    ax.set_aspect("equal")
    rows, cols = fld.shape
    hot = fld >= THRESHOLD
    ax.contour(hot.astype(float), levels=[0.5], colors=[DEEP_RED],
               linewidths=0.9, zorder=4)
    ax.set_xticks([0, cols // 2, cols - 1])
    ax.set_xticklabels(["0", "%.0f" % (PLOT_M * (cols // 2)),
                        "%.0f" % (PLOT_M * (cols - 1))], fontsize=6.4)
    ax.set_yticks([0, rows - 1])
    ax.set_yticklabels(["0", "%.0f" % (PLOT_M * (rows - 1))], fontsize=6.4)
    ax.set_xlabel("field easting (m)", fontsize=7.4)
    ax.set_ylabel("northing (m)", fontsize=7.4)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.text(0.5, 1.02, "%.0f %% of plots over threshold in %s"
            % (100.0 * hot.mean(), week_label), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.6, color=DEEP_RED)
    safe_colorbar(fig, im, ax, label="density (individuals/plant)",
                  width=0.11, pad=0.07)
    ax.set_title("Infestation is patchy, not uniform", fontsize=8.4,
                 fontweight="bold", color=INK, pad=13.0)


def main():
    set_pub_style()
    rng = np.random.default_rng(7392)

    dd_all = np.sort(rng.normal(1500.0, 195.0, N_RECORD))
    dd_all = dd_all[rng.permutation(N_RECORD)]
    dd_shown = dd_all[-N_SHOWN:]

    t, dens, enemy = build_series(dd_shown)
    peaks_shown = np.array([dens[(t >= yi * WPY) & (t < (yi + 1) * WPY)].max()
                            for yi in range(N_SHOWN)])
    peaks_old = (1.55 * (1.0 + 0.62 * 2) * season_scale(dd_all[:-N_SHOWN]) +
                 WINTER + rng.normal(0.0, 0.42, N_RECORD - N_SHOWN))
    peaks_all = np.concatenate([peaks_old, peaks_shown])

    fig = plt.figure(figsize=(W, H))
    ax_main = axes_inches(fig, (0.88, 1.00, 4.92, 4.12))
    ax_sc = axes_inches(fig, (7.10, 3.52, 2.80, 1.72))
    ax_map = axes_inches(fig, (7.10, 0.95, 2.40, 1.60))

    n_over, lag, peak_i = main_panel(ax_main, t, dens, enemy, dd_shown)
    r = scatter_panel(ax_sc, dd_all, peaks_all, N_SHOWN)
    worst_year = FIRST_SHOWN + int(peak_i // WPY)
    fld = field_snapshot(rng, float(dens[peak_i]) * 0.78)
    map_panel(fig, ax_map, fld, "%d week %d" % (worst_year,
                                                int(peak_i % WPY)))

    fig.text(0.016, 6.14 / H,
             "Every season breaks the threshold, and the predator response "
             "arrives %d weeks too late to stop it" % lag,
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.016, 0.085 / H, textwrap.fill(
             "Predator density in (a) is the larval series convolved with a "
             "delayed recruitment kernel, so the lag is measured back out of "
             "the plotted curves. Peaks in (b) for the four highlighted "
             "seasons are read off (a); the heat-sum correlation is "
             "$r$ = %.2f over %d seasons." % (r, N_RECORD), 195),
             ha="left", va="bottom", fontsize=6.5, color=MUTED,
             linespacing=1.5)

    panel_label_inches(fig, ax_main, "(a)", dx=-0.68, dy=0.16)
    panel_label_inches(fig, ax_sc, "(b)", dx=-0.62, dy=0.16)
    panel_label_inches(fig, ax_map, "(c)", dx=-0.62, dy=0.16)

    savefig(fig, "739_ent_population_dynamics.png")


if __name__ == "__main__":
    main()
