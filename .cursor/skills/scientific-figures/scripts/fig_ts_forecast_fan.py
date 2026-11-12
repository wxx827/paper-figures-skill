"""Probabilistic forecast fan with an interval-growth companion panel.

The main panel carries 60 months of history, the median path over an 18-month
horizon and nested 50 / 80 / 95 % predictive intervals drawn as one hue at
three lightness steps - never a rainbow, so the nesting reads as depth.  The
narrow right-hand panel shows the same intervals as widths against horizon,
which is the part of the story a fan hides.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten, darken
from figkit import panel_labels_inches

HIST_C = "#3a3a3a"     # observed history
BLUE = "#3b5b92"       # house blue: the model
GREY = "#808080"
N_HIST = 46
H_MAX = 18
LEVELS = [(95, 1.960), (80, 1.282), (50, 0.674)]
SHADES = (0.74, 0.52, 0.28)     # one hue, three lightness steps


def synth_history(rng, n):
    t = np.arange(n, dtype=float)
    trend = 96.0 + 0.42 * t + 5.0 * np.sin(2 * np.pi * t / 46.0)
    season = 6.2 * np.sin(2 * np.pi * t / 12.0 - 0.6)
    return trend + season + rng.normal(0.0, 2.1, n)


def forecast_paths(n, h):
    """Median path plus the sigma_h that widens like sqrt(h)."""
    t = np.arange(n, n + h, dtype=float)
    trend = 96.0 + 0.42 * t + 5.0 * np.sin(2 * np.pi * t / 46.0)
    season = 6.2 * np.sin(2 * np.pi * t / 12.0 - 0.6)
    steps = np.arange(1, h + 1, dtype=float)
    sigma = 2.35 * np.sqrt(steps)
    return t, trend + season, sigma, steps


def panel_fan(ax, t_h, y_h, t_f, med, sigma):
    shades = [lighten(BLUE, a) for a in SHADES]
    for (lvl, z), col in zip(LEVELS, shades):
        ax.fill_between(t_f, med - z * sigma, med + z * sigma, color=col,
                        lw=0, zorder=2, label=f"{lvl}% interval")
    ax.plot(t_h, y_h, color=HIST_C, lw=1.15, zorder=5, label="Observed")
    ax.plot([t_h[-1], t_f[0]], [y_h[-1], med[0]], color=BLUE, lw=1.4,
            ls=(0, (3.4, 2.0)), zorder=6)
    ax.plot(t_f, med, color=BLUE, lw=1.5, ls=(0, (3.4, 2.0)), zorder=6,
            label="Median forecast")

    ax.axvline(t_h[-1], color=GREY, lw=0.9, ls=(0, (1.6, 2.0)), zorder=4)
    ax.text(t_f[-1] + 0.9, med[-1], r"$\hat{y}_{t+h}$", ha="left",
            va="center", fontsize=8.2, color=BLUE, zorder=7)
    ax.set_xlabel("Month index (-)")
    ax.set_ylabel("Shipments (kt)")
    ax.set_xlim(-1.5, t_f[-1] + 5.5)
    z = LEVELS[0][1]
    lo = min(y_h.min(), (med - z * sigma).min())
    hi = max(y_h.max(), (med + z * sigma).max())
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.11 * (hi - lo))


def panel_width(ax, steps, sigma):
    shades = [darken(lighten(BLUE, a), 0.10) for a in (0.55, 0.32, 0.08)]
    marks = ["o", "s", "^"]
    for (lvl, z), col, mk in zip(LEVELS, shades, marks):
        w = 2.0 * z * sigma
        ax.plot(steps, w, color=col, lw=1.25, marker=mk, ms=2.9,
                mfc="white", mew=0.8, label=f"{lvl}%")
    ax.set_xlabel("Forecast horizon $h$ (months)", fontsize=7.5)
    ax.set_ylabel("Interval width (kt)", fontsize=7.5)
    ax.set_xlim(0.0, steps[-1] + 0.8)
    ax.set_ylim(0.0, 2.0 * LEVELS[0][1] * sigma[-1] * 1.16)
    ax.set_xticks([1, 6, 12, 18])
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.5,
              labelspacing=0.28, borderaxespad=0.25)
    ax.set_title("Width grows as $\\sqrt{h}$", fontsize=8.5,
                 fontweight="bold")


def main():
    set_pub_style()
    rng = np.random.default_rng(19)
    y_h = synth_history(rng, N_HIST)
    t_h = np.arange(N_HIST, dtype=float)
    t_f, med, sigma, steps = forecast_paths(N_HIST, H_MAX)

    fig = plt.figure(figsize=(7.6, 4.3))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.46], wspace=0.235,
                  left=0.072, right=0.985, top=0.865, bottom=0.135)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])

    panel_fan(ax, t_h, y_h, t_f, med, sigma)
    panel_width(bx, steps, sigma)
    for a in (ax, bx):
        a.grid(axis="y", lw=0.5, color="#f0f0f0")
        a.set_axisbelow(True)
        style_axes(a)

    lo, hi = ax.get_ylim()
    ax.text(t_h[-1] - 1.0, hi - 0.02 * (hi - lo), "training data ",
            ha="right", va="top", fontsize=6.8, color="#777777")
    ax.text(t_h[-1] + 1.0, hi - 0.02 * (hi - lo), " forecast horizon",
            ha="left", va="top", fontsize=6.8, color=BLUE)

    handles = [Line2D([], [], color=HIST_C, lw=1.3, label="Observed"),
               Line2D([], [], color=BLUE, lw=1.5, ls=(0, (3.4, 2.0)),
                      label=r"Median $\hat{y}_{t+h}$")]
    handles += [Patch(facecolor=lighten(BLUE, a), edgecolor="none",
                      label=f"{lvl}% predictive interval")
                for (lvl, _), a in zip(LEVELS, SHADES)]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              handlelength=1.7, labelspacing=0.32, borderaxespad=0.3)

    panel_labels_inches(fig, [ax, bx], dx=-0.52, dy=0.10)
    savefig(fig, "356_ts_forecast_fan.png")


if __name__ == "__main__":
    main()
