"""Emulsion instability read from the droplet size distribution it produces.

Five storage times each get a volume-weighted droplet distribution built from
one Ostwald-ripening law - the cube of the mean diameter grows linearly in
time - plus a slowly widening coarse shoulder that stands for flocculation.
D[4,3] and the span are then computed as moments and percentiles of exactly
those plotted curves, so the arrow tracking the mode across the panel and the
numbers printed beside it are the same measurement.  The creaming index of
each day is drawn on the right-hand axis at that day's own D[4,3], which is
the honest way to put a time series on a diameter abscissa and makes the
coupling between coarsening and phase separation visible in one panel.  The
Turbiscan-style map is the same creaming history resolved in height, and the
zeta-potential curve locates the isoelectric point by interpolating its own
zero crossing.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import (set_pub_style, savefig, darken, get_cmap, get_palette)
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

DAYS = np.array([0.0, 3.0, 7.0, 14.0, 28.0])
D0_UM = 0.62                 # volume-mode diameter at day 0
RIPEN = 0.0165               # um^3 per day, Ostwald cube-law slope
SIG0, SIG_RATE = 0.255, 0.0072   # lognormal width and its drift, per day
FLOC_RATE = 0.0135           # volume fraction moving into the coarse shoulder
FLOC_SHIFT = 3.4             # coarse shoulder sits this many times the mode
TUBE_MM = 42.0               # height of the Turbiscan cell
BINS_PER_DECADE = 24.0       # so the ordinate reads as volume per size bin
PH = np.linspace(2.0, 9.0, 200)
IEP_GUESS = 4.6

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)


def distribution(d, day):
    """Volume-weighted droplet distribution after ``day`` days of storage."""
    mode = (D0_UM ** 3 + RIPEN * day) ** (1.0 / 3.0)
    sig = SIG0 + SIG_RATE * day
    floc = min(FLOC_RATE * day, 0.42)
    ln = np.log(d)
    fine = np.exp(-0.5 * ((ln - np.log(mode)) / sig) ** 2) / sig
    coarse = np.exp(-0.5 * ((ln - np.log(mode * FLOC_SHIFT)) /
                            (1.7 * sig)) ** 2) / (1.7 * sig)
    v = (1.0 - floc) * fine + floc * coarse
    return (100.0 * np.log(10.0) / BINS_PER_DECADE) * v / np.trapezoid(v, ln)


def moments(d, v):
    """D[4,3], D10 / D50 / D90 and the span of a plotted distribution.

    The distribution is volume weighted, so its own integral is already the
    volume fraction; D[4,3] is therefore the first moment of that curve and
    the percentiles come from its cumulative integral.
    """
    ln = np.log(d)
    w = v / np.trapezoid(v, ln)
    d43 = float(np.trapezoid(w * d, ln))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) *
                                           np.diff(ln))])
    cum /= cum[-1]
    d10, d50, d90 = np.interp([0.10, 0.50, 0.90], cum, d)
    return d43, float(d10), float(d50), float(d90), float((d90 - d10) / d50)


def creaming(day):
    """Creaming index in % as a saturating function of storage time."""
    return 34.0 * (1.0 - np.exp(-day / 11.0))


def transmission(height, day):
    """Turbiscan transmission: a clarified layer eating down from the top."""
    front = TUBE_MM * (1.0 - 0.40 * (1.0 - np.exp(-day / 11.0)))
    return 2.0 + 84.0 / (1.0 + np.exp(-(height - front) / 1.5))


def zeta(ph):
    """Zeta potential in mV; a smooth titration curve through the IEP."""
    return 46.0 * np.tanh((IEP_GUESS - ph) / 1.85) - 3.5 * (ph - IEP_GUESS)


def main_panel(ax, d, dists, stats, colours):
    ax_c = ax.twinx()
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax_c.set_zorder(1)

    modes = [d[int(np.argmax(v))] for v in dists]
    peaks = [float(v.max()) for v in dists]
    pmax = max(peaks)

    for i, day in enumerate(DAYS):
        ax.plot(d, dists[i], color=colours[i], lw=1.6, zorder=5,
                label="day {:.0f}".format(day))
        ax.fill_between(d, 0.0, dists[i], color=colours[i], alpha=0.10,
                        lw=0.0, zorder=3)

    d43 = [s[0] for s in stats]
    y_arr = pmax * 1.07
    ax.annotate("", xy=(d43[-1], y_arr), xytext=(d43[0], y_arr), zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=SUB, lw=1.0,
                                mutation_scale=9))
    ax.text(np.sqrt(d43[0] * d43[-1]), y_arr + 0.025 * pmax,
            "$D[4,3]$ {:.2f} $\\rightarrow$ {:.2f} $\\mu m$".format(
                d43[0], d43[-1]),
            ha="center", va="bottom", fontsize=6.8, color=SUB, zorder=9)

    span = [s[4] for s in stats]
    ci = creaming(DAYS)
    ax_c.plot(d43, ci, color=RED, lw=1.2, ls=(0, (4.5, 2.0)), zorder=6)
    ax_c.plot(d43, ci, marker="s", ms=4.4, mfc=RED, mec="white", mew=0.8,
              ls="none", zorder=7)
    ax_c.text(d43[-1] * 1.06, ci[-1], "creaming index,\nplotted at each "
              "day's $D[4,3]$", ha="left", va="center", fontsize=6.4,
              color=RED, linespacing=1.30, zorder=8)

    for row, i in enumerate((0, len(DAYS) - 1)):
        ax.text(d[0] * 1.30, pmax * (1.40 - 0.085 * row),
                "day {:.0f}:  $D[4,3]$ = {:.2f} $\\mu m$,  span = "
                "{:.2f}".format(DAYS[i], d43[i], span[i]),
                ha="left", va="top", fontsize=6.8,
                color=darken(colours[i], 0.45), zorder=9)

    ax.set_xscale("log")
    ax.set_xlim(0.10, 30.0)
    ax.set_ylim(0.0, pmax * 1.44)
    ax_c.set_ylim(0.0, 96.0)
    ax.set_xlabel("droplet diameter ($\\mu m$)")
    ax.set_ylabel("volume in size bin (%, {:.0f} bins per decade)".format(
        BINS_PER_DECADE))
    ax_c.set_ylabel("creaming index (%)", color=RED)
    ax_c.tick_params(axis="y", colors=RED, labelsize=6.8)
    ax_c.spines["right"].set_color(RED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["left"].set_visible(False)
    ax.grid(lw=0.5, alpha=0.45)
    ax.legend(loc="upper right", fontsize=6.8, frameon=False,
              bbox_to_anchor=(0.995, 0.995), handlelength=1.6,
              borderaxespad=0.0, labelspacing=0.28)
    ax.set_title("Ripening moves the mode right and grows a flocculated "
                 "shoulder", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def turbiscan_panel(fig, ax):
    day = np.linspace(0.0, DAYS[-1], 160)
    hgt = np.linspace(0.0, TUBE_MM, 120)
    grid = transmission(hgt[:, None], day[None, :])
    mesh = ax.pcolormesh(day, hgt, grid, cmap=get_cmap("cb_sequential_blue"),
                         vmin=0.0, vmax=90.0, shading="gouraud", zorder=2)
    front = TUBE_MM * (1.0 - 0.40 * (1.0 - np.exp(-day / 11.0)))
    ax.plot(day, front, color=RED, lw=1.2, ls=(0, (4.0, 1.8)), zorder=4)
    ax.text(DAYS[-1] * 0.97, TUBE_MM * 0.94, "clarified top layer",
            ha="right", va="top", fontsize=6.2, color="white", zorder=5)
    ax.annotate("50 % transmission front", xy=(20.0, np.interp(20.0, day,
                                                               front)),
                xytext=(-4.0, -12.0), textcoords="offset points",
                ha="right", va="top", fontsize=6.0, color=RED, zorder=5)
    cax = ax.inset_axes([1.035, 0.0, 0.045, 1.0])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("transmission (%)", fontsize=6.6, color=INK)
    cb.ax.tick_params(labelsize=6.0, width=0.7, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(SUB)
    ax.set_xlim(0.0, DAYS[-1])
    ax.set_ylim(0.0, TUBE_MM)
    ax.set_xlabel("storage time (days)", fontsize=7.4)
    ax.set_ylabel("height in cell (mm)", fontsize=7.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Turbiscan transmission map", fontsize=8.0,
                 fontweight="bold", color=INK)


def zeta_panel(ax, iep):
    z = zeta(PH)
    ax.axhspan(-30.0, 30.0, color="#f5f5f5", zorder=1)
    ax.text(8.9, 0.0, "|$\\zeta$| < 30 mV:\naggregation likely", ha="right",
            va="center", fontsize=6.0, color=MUTED, linespacing=1.25,
            zorder=3)
    ax.axhline(0.0, color=SUB, lw=0.8, zorder=3)
    ax.plot(PH, z, color=GREEN, lw=1.7, zorder=5)
    ax.plot([iep], [0.0], marker="o", ms=5.0, mfc=RED, mec="white", mew=0.8,
            zorder=7)
    ax.annotate("isoelectric point\npH = {:.2f}".format(iep), xy=(iep, 0.0),
                xytext=(7.0, -16.0), textcoords="offset points", ha="left",
                va="top", fontsize=6.4, color=RED, linespacing=1.30, zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.90),
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.set_xlim(2.0, 9.0)
    ax.set_ylim(-62.0, 52.0)
    ax.set_xlabel("pH (-)", fontsize=7.4)
    ax.set_ylabel("zeta potential (mV)", fontsize=7.4)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Charge disappears where the emulsion does", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    """Report any in-view tick or axis label the footnote would sit on."""
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
    np.random.default_rng(916)
    colours = get_palette("cb_sequential_blue", 8)[3:8]

    d = np.geomspace(0.08, 40.0, 900)
    dists = [distribution(d, day) for day in DAYS]
    stats = [moments(d, v) for v in dists]
    z = zeta(PH)
    k = int(np.flatnonzero(z <= 0.0)[0])
    iep = float(np.interp(0.0, [z[k], z[k - 1]], [PH[k], PH[k - 1]]))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_tb = axes_inches(fig, AUX_T)
    ax_z = axes_inches(fig, AUX_B)

    main_panel(ax_main, d, dists, stats, colours)
    turbiscan_panel(fig, ax_tb)
    zeta_panel(ax_z, iep)

    panel_labels_inches(fig, [ax_main, ax_tb, ax_z], dx=-0.62, dy=0.14)
    savefig(fig, "916_fud_emulsion_stability.png")


if __name__ == "__main__":
    main()
