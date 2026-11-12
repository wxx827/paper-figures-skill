"""Figure 615 - Five centuries of eruptions at a single composite volcano.

The catalogue is generated once and then measured, never described twice.  A
repose interval is drawn from a log-normal whose median grows with the size of
the preceding eruption, the explosivity index comes from a log-linear
size-frequency law and the eruptive style is sampled from a style matrix
conditioned on that index - so the bubble colours in panel (a) and the bars in
panel (b) are two views of one list of events.

The cumulative curve, the fitted b value and the unrest lead time quoted in
panel (c) are all computed from the arrays that are plotted, which is the only
way the numbers can survive a change of random seed.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import matplotlib as mpl

from pubstyle import set_pub_style, savefig, get_palette, COLORS, darken
from figkit import panel_label_inches, unclip_legend

T0, T1 = 1500.0, 2025.0
VEI_LEVELS = np.arange(1, 7)
# Log-linear size-frequency law: each VEI step is about 10^-0.42 as common.
VEI_B = 0.32
STYLES = ["effusive", "strombolian", "vulcanian", "plinian"]
STYLE_COLORS = ["#3b5b92", "#2e7d5b", "#c9a227", "#b23a48"]
# P(style | VEI), one row per VEI level.
STYLE_MATRIX = np.array([
    [0.70, 0.28, 0.02, 0.00],
    [0.30, 0.52, 0.16, 0.02],
    [0.10, 0.35, 0.45, 0.10],
    [0.02, 0.14, 0.52, 0.32],
    [0.00, 0.04, 0.36, 0.60],
    [0.00, 0.00, 0.18, 0.82],
])
LEAD_DAYS = 180                # length of the precursor window
BACKGROUND_EQ = 2.0            # background seismicity, events per day


def bubble_size(vei):
    """Marker area for an explosivity index; VEI is logarithmic already."""
    return 3.2 * np.asarray(vei, dtype=float) ** 2.4


def build_catalogue(rng):
    """Years, explosivity indices and styles for one synthetic volcano."""
    w = 10.0 ** (-VEI_B * VEI_LEVELS)
    w = w / w.sum()
    year, years, veis, styles = T0, [], [], []
    while True:
        v = int(rng.choice(VEI_LEVELS, p=w))
        year += rng.lognormal(np.log(3.0 + 2.4 * (v - 1)), 0.62)
        if year >= T1:
            break
        years.append(year)
        veis.append(v)
        styles.append(int(rng.choice(len(STYLES), p=STYLE_MATRIX[v - 1])))
    return np.array(years), np.array(veis), np.array(styles)


def precursors(rng, days):
    """Seismicity, deformation rate and SO2 flux over the run-up window."""
    eq = BACKGROUND_EQ + 34.0 * np.exp(days / 20.0)
    eq = rng.poisson(np.clip(eq, 0.2, None)).astype(float)
    defo = 0.18 + 9.0 * np.exp(days / 26.0) + rng.normal(0.0, 0.22, days.size)
    so2 = (120.0 + 2400.0 * np.exp(days / 13.0)
           + 380.0 * (days > -46.0) + rng.normal(0.0, 55.0, days.size))
    return eq, np.clip(defo, 0.0, None), np.clip(so2, 0.0, None)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(6150)
    years, veis, styles = build_catalogue(rng)
    repose = np.diff(np.concatenate([[T0], years]))
    cum = np.arange(1, years.size + 1)
    big = veis >= 4

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.62, 1.0],
                          height_ratios=[1.0, 0.58],
                          left=0.058, right=0.945, top=0.880, bottom=0.120,
                          wspace=0.26, hspace=0.46)
    ax = fig.add_subplot(gs[0, :])
    ax_h = fig.add_subplot(gs[1, 0])
    sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1], hspace=0.30)
    ax_p = [fig.add_subplot(sub[i, 0]) for i in range(3)]

    # ---- (a) the event timeline --------------------------------------------
    for k, name in enumerate(STYLES):
        sel = styles == k
        ax.scatter(years[sel], repose[sel], s=bubble_size(veis[sel]),
                   facecolor=STYLE_COLORS[k], edgecolor="white",
                   linewidth=0.45, alpha=0.88, zorder=4, label=name)
    ax.set_yscale("log")
    # The bottom decade is left clear on purpose: the marker-size key lives
    # there, and a key drawn over the data cannot be read.
    ax.set_ylim(0.38, max(repose.max() * 3.4, 10.0))
    ax.set_xlim(T0, T1)
    ax.set_xlabel("year (CE)", fontsize=8.0)
    ax.set_ylabel("repose interval before event (years)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="y", lw=0.5, alpha=0.55)

    for j in np.nonzero(big)[0]:
        ax.annotate(f"VEI {veis[j]}", xy=(years[j], repose[j]),
                    xytext=(0.0, 11.0 + 0.9 * bubble_size(veis[j]) ** 0.5),
                    textcoords="offset points", fontsize=6.2,
                    color=darken(STYLE_COLORS[styles[j]], 0.25),
                    ha="center", va="bottom")

    ax_c = ax.twinx()
    ax_c.step(years, cum, where="post", color="#1a1a1a", lw=1.3, zorder=5)
    ax_c.set_ylim(0, cum[-1] * 1.06)
    ax_c.set_ylabel("cumulative eruptions (-)", fontsize=8.0)
    ax_c.tick_params(labelsize=7.0)
    ax_c.spines["top"].set_visible(False)
    rate_early = cum[years < 1800].size / (1800.0 - T0) * 100.0
    rate_late = (cum[-1] - cum[years < 1800].size) / (T1 - 1800.0) * 100.0
    ax.text(0.997, 1.005, f"{rate_early:.1f} events per century before 1800, "
            f"{rate_late:.1f} after", transform=ax.transAxes, fontsize=6.4,
            color=COLORS["sub"], ha="right", va="bottom")

    key_vei = (int(veis.min()), int(np.median(veis)), int(veis.max()))
    size_key = [Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor="#8a8a8a", markeredgecolor="white",
                       markeredgewidth=0.45,
                       markersize=np.sqrt(bubble_size(v)), label=f"VEI {v}")
                for v in sorted(set(key_vei))]
    # Explicit handles: the automatic ones would inherit whatever bubble size
    # each style happened to draw first, which reads as a size key.
    style_key = [Line2D([0], [0], marker="o", linestyle="none", markersize=4.4,
                        markerfacecolor=c, markeredgecolor="white",
                        markeredgewidth=0.45, label=n)
                 for n, c in zip(STYLES, STYLE_COLORS)]
    leg = ax.legend(handles=style_key, loc="upper left",
                    bbox_to_anchor=(0.005, 0.99), ncol=4, frameon=False,
                    fontsize=6.8, handletextpad=0.4, columnspacing=1.1)
    ax.add_artist(leg)
    unclip_legend(leg)
    ax.legend(handles=size_key, loc="lower left", bbox_to_anchor=(0.005, 0.01),
              ncol=3, frameon=False, fontsize=6.8, handletextpad=0.6,
              columnspacing=1.6, labelspacing=0.4, borderpad=0.2)
    ax.set_title("Eruption record, bubble area scaled by explosivity index",
                 fontsize=8.8, fontweight="bold", pad=4.0)

    # ---- (b) size-frequency ------------------------------------------------
    counts = np.array([(veis == v).sum() for v in VEI_LEVELS])
    ok = counts > 0
    fit = np.polyfit(VEI_LEVELS[ok], np.log10(counts[ok]), 1)
    ax_h.bar(VEI_LEVELS, np.maximum(counts, 0.1), width=0.68,
             color=[STYLE_COLORS[int(np.argmax(STYLE_MATRIX[v - 1]))]
                    for v in VEI_LEVELS], alpha=0.80, edgecolor="white",
             linewidth=0.5, zorder=3)
    ax_h.plot(VEI_LEVELS, 10.0 ** np.polyval(fit, VEI_LEVELS), color="#1a1a1a",
              lw=1.2, ls=(0, (4, 2.2)), zorder=5)
    for v, c in zip(VEI_LEVELS, counts):
        if c:
            ax_h.text(v, c * 1.18, f"{c}", ha="center", va="bottom",
                      fontsize=6.2, color=COLORS["sub"])
    ax_h.set_yscale("log")
    ax_h.set_ylim(0.55, counts.max() * 3.2)
    ax_h.set_xticks(VEI_LEVELS)
    ax_h.set_xlabel("volcanic explosivity index (VEI)", fontsize=7.6)
    ax_h.set_ylabel("events (-)", fontsize=7.6)
    ax_h.tick_params(labelsize=6.6)
    ax_h.grid(axis="y", lw=0.5, alpha=0.55)
    ax_h.text(0.97, 0.94, f"slope {fit[0]:+.2f} per VEI unit",
              transform=ax_h.transAxes, fontsize=6.6, color="#1a1a1a",
              ha="right", va="top")
    ax_h.set_title("Size-frequency distribution", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    for s in ("top", "right"):
        ax_h.spines[s].set_visible(False)

    # ---- (c) precursor signals ---------------------------------------------
    days = np.arange(-LEAD_DAYS, 1, 1.0)
    eq, defo, so2 = precursors(rng, days)
    # A single Poisson spike is not unrest; require the 7-day mean to stay
    # above three times background from the crossing onwards.
    smooth = np.convolve(eq, np.ones(7) / 7.0, mode="same")
    quiet = np.nonzero(smooth <= 3.0 * BACKGROUND_EQ)[0]
    k0 = int(quiet[-1]) + 1 if quiet.size and quiet[-1] + 1 < days.size else 0
    lead = int(-days[k0])
    cols = get_palette("cb_sequential_blue", 7)[2:5][::-1]
    series = ((eq, "EQ (d$^{-1}$)"),
              (defo, "uplift (mm d$^{-1}$)"),
              (so2, "SO$_2$ (t d$^{-1}$)"))
    for k, ((val, name), col) in enumerate(zip(series, cols)):
        a = ax_p[k]
        a.fill_between(days, 0.0, val, facecolor=col, alpha=0.28, lw=0.0,
                       zorder=2)
        a.plot(days, val, color=col, lw=1.0, zorder=3)
        a.axvline(-lead, color=COLORS["baseline"], lw=0.7, ls=(0, (3, 2.2)),
                  zorder=4)
        a.axvline(0.0, color="#b23a48", lw=1.1, zorder=5)
        a.set_xlim(-LEAD_DAYS, 6)
        a.set_ylim(0.0, val.max() * 1.30)
        a.set_ylabel(name, fontsize=6.2, labelpad=2.0)
        a.tick_params(labelsize=6.2, labelbottom=(k == 2))
        a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(2, prune="upper"))
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax_p[0].text(-lead - 4.0, ax_p[0].get_ylim()[1] * 0.96,
                 f"unrest onset, {lead} d before", fontsize=6.2,
                 color=COLORS["sub"], ha="right", va="top")
    ax_p[0].text(2.0, ax_p[0].get_ylim()[1] * 0.96, "eruption", fontsize=6.2,
                 color="#b23a48", ha="left", va="top", rotation=90)
    ax_p[2].set_xlabel("days relative to eruption onset", fontsize=7.6)
    ax_p[0].set_title("Precursor signals before onset", fontsize=8.5,
                      fontweight="bold", pad=3.5)

    panel_label_inches(fig, ax, "(a)", dx=-0.44, dy=0.09)
    panel_label_inches(fig, ax_h, "(b)", dx=-0.44, dy=0.09)
    panel_label_inches(fig, ax_p[0], "(c)", dx=-0.44, dy=0.09)


    savefig(fig, "615_vol_eruption_history.png")


if __name__ == "__main__":
    main()
