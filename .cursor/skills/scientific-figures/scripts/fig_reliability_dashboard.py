"""Distribution-reliability dashboard for one utility year.

(a) the four headline indices as KPI cards, each carrying a 12-month
sparkline and a year-on-year change coloured by whether the movement is an
improvement, not by its sign - SAIFI rising is bad, ASAI rising is good.
(b) a Pareto chart of interruption causes, (c) monthly SAIDI over five
years, (d) the feeder ranking against the 100-minute target.

The indices are kept internally consistent (SAIDI = SAIFI x CAIDI, and the
year-on-year deltas multiply out), so the four cards cannot contradict each
other.  Panel labels are offset by a fixed number of inches, which keeps them
aligned across rows of very different heights.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_colorbar)

INK = "#333333"
GREY = "#8d959c"
GOOD = "#4a8f68"
BAD = "#b23a48"
SLATE = "#7ba5bd"
SLATE_D = "#3a6c8c"
CARD_F = "#f7f8f9"
CARD_E = "#e3e7ea"

# name -> (value, unit, year-on-year %, lower_is_better)
KPIS = [
    ("SAIDI", 92.4, "min / customer", -8.3, True),
    ("SAIFI", 1.12, "events / customer", -4.1, True),
    ("CAIDI", 82.5, "min / event", -4.4, True),
    ("ASAI", 99.982, "% availability", +0.003, False),
]
CAUSES = [("Vegetation", 412), ("Equipment", 318), ("Weather", 265),
          ("Animal", 148), ("Vehicle", 96), ("Human error", 54),
          ("Unknown", 37)]
YEARS = [2020, 2021, 2022, 2023, 2024]
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
TARGET = 100.0
NFEED = 14


def kpi_card(axk, name, value, unit, delta, lower_better, spark, rng):
    axk.set_xlim(0, 1)
    axk.set_ylim(0, 1)
    axk.axis("off")
    axk.add_patch(FancyBboxPatch(
        (0.012, 0.02), 0.976, 0.96, transform=axk.transAxes,
        boxstyle="round,pad=0.0,rounding_size=0.055", facecolor=CARD_F,
        edgecolor=CARD_E, lw=0.8, zorder=0, clip_on=False))

    axk.text(0.065, 0.90, name, fontsize=7.2, color=GREY, fontweight="bold",
             ha="left", va="top", zorder=3)
    txt = f"{value:.3f}" if name == "ASAI" else f"{value:g}"
    axk.text(0.065, 0.735, txt, fontsize=15, color=INK, fontweight="bold",
             ha="left", va="top", zorder=3)
    axk.text(0.065, 0.415, unit, fontsize=6.2, color=GREY, ha="left",
             va="top", zorder=3)

    improving = (delta < 0) if lower_better else (delta > 0)
    col = GOOD if improving else BAD
    rot = 0.0 if delta > 0 else 180.0
    axk.plot([0.735], [0.875], marker=(3, 0, rot), markersize=5.0,
             color=col, linestyle="none", zorder=3, clip_on=False)
    axk.text(0.785, 0.875, f"{delta:+.3g}%" if abs(delta) < 0.1
             else f"{delta:+.1f}%", fontsize=6.6, color=col,
             fontweight="bold", ha="left", va="center", zorder=3)
    axk.text(0.935, 0.685, "vs. last year", fontsize=6.0, color=GREY,
             ha="right", va="center", zorder=3)

    sy = 0.085 + 0.215 * (spark - spark.min()) / np.ptp(spark)
    sx = np.linspace(0.075, 0.925, spark.size)
    axk.fill_between(sx, 0.078, sy, color=SLATE, alpha=0.20, lw=0, zorder=2)
    axk.plot(sx, sy, color=SLATE_D, lw=1.0, zorder=3)
    axk.plot([sx[-1]], [sy[-1]], marker="o", ms=2.6, color=SLATE_D,
             mec="white", mew=0.6, zorder=4, linestyle="none")


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(17)

    fig = plt.figure(figsize=(7.6, 6.15))
    gs = fig.add_gridspec(3, 1, left=0.078, right=0.975, top=0.925,
                          bottom=0.062, hspace=0.44,
                          height_ratios=[0.40, 1.0, 0.58])
    grow = gs[0].subgridspec(1, 4, wspace=0.10)
    mrow = gs[1].subgridspec(1, 2, wspace=0.30, width_ratios=[1.0, 0.86])
    axk = [fig.add_subplot(grow[i]) for i in range(4)]
    axb = fig.add_subplot(mrow[0])
    axc = fig.add_subplot(mrow[1])
    axd = fig.add_subplot(gs[2])

    # ---- (a) KPI cards -----------------------------------------------------
    # the sparkline is min-max normalised inside the card, so it is built in
    # arbitrary units: a unit ramp in the direction of travel plus noise.
    # Scaling real values instead made ASAI (a 0.003 % move) pure hash.
    for a, (name, val, unit, dlt, lower) in zip(axk, KPIS):
        trend = np.linspace(0.0, np.sign(dlt), 12) + 0.22 * rng.normal(0, 1,
                                                                       12)
        kpi_card(a, name, val, unit, dlt, lower, trend, rng)

    # ---- (b) cause Pareto --------------------------------------------------
    labs = [c for c, _ in CAUSES]
    cnt = np.array([n for _, n in CAUSES], dtype=float)
    cum = 100.0 * np.cumsum(cnt) / cnt.sum()
    xx = np.arange(len(labs))
    axb.bar(xx, cnt, width=0.66, color=SLATE, lw=0, zorder=3)
    for x, c in zip(xx, cnt):
        axb.text(x, c + cnt.max() * 0.025, f"{c:.0f}", fontsize=6.2,
                 color=INK, ha="center", va="bottom", zorder=4)
    axb.set_xticks(xx)
    axb.set_xticklabels(labs, fontsize=6.6, rotation=28, ha="right",
                        rotation_mode="anchor")
    axb.set_ylim(0, cnt.max() * 1.30)
    axb.set_ylabel("Interruptions", fontsize=7.4)
    axb.grid(axis="y", lw=0.4, alpha=0.5)
    style_axes(axb)

    axb2 = axb.twinx()
    axb2.plot(xx, cum, color=SLATE_D, lw=1.3, marker="o", ms=3.0,
              mec="white", mew=0.6, zorder=5)
    axb2.axhline(80, color=BAD, lw=0.8, ls=(0, (4, 2.2)), zorder=4)
    axb2.set_ylim(0, 108)
    axb2.set_yticks([0, 25, 50, 75, 100])
    axb2.set_ylabel("Cumulative (%)", fontsize=7.4, color=SLATE_D)
    axb2.tick_params(axis="y", colors=SLATE_D, labelsize=7)
    axb2.spines["top"].set_visible(False)
    axb2.spines["right"].set_color(SLATE_D)
    axb2.spines["right"].set_linewidth(0.8)
    # parked in the wedge between the flattening curve and the tail bars,
    # the one region of this panel that carries no ink
    k = int(np.argmax(cum >= 80))
    axb.text(len(labs) - 0.4, cnt.max() * 0.42, f"top {k + 1} causes\n"
             f"= {cum[k]:.0f}% of interruptions", fontsize=6.3, color=BAD,
             ha="right", va="bottom", linespacing=1.4, zorder=6)
    axb.set_title("Interruption causes", fontsize=8.4, fontweight="bold",
                  loc="left", x=0.0)

    # ---- (c) monthly SAIDI heat map ---------------------------------------
    seas = np.array([1.05, 0.92, 0.85, 0.80, 0.95, 1.25, 1.55, 1.45, 1.10,
                     0.88, 0.92, 1.12])
    base = np.array([15.2, 13.9, 12.4, 11.1, 9.9])[:, None]
    grid = base * seas[None, :] * (1.0 + 0.14 * rng.normal(0, 1, (5, 12)))
    grid = np.clip(grid, 2.0, None)
    cmap = LinearSegmentedColormap.from_list(
        "saidi", PALETTES["cb_sequential_warm"], N=256)
    norm = Normalize(0, np.ceil(grid.max() / 5) * 5)
    mesh = axc.pcolormesh(np.arange(13), np.arange(6), grid, cmap=cmap,
                          norm=norm, shading="flat", edgecolors="white",
                          linewidth=0.6)
    axc.set_xticks(np.arange(12) + 0.5)
    axc.set_xticklabels(MONTHS, fontsize=6.6)
    axc.set_yticks(np.arange(5) + 0.5)
    axc.set_yticklabels([str(y) for y in YEARS], fontsize=6.8)
    axc.tick_params(length=0)
    axc.invert_yaxis()
    axc.set_title("Monthly SAIDI by year", fontsize=8.4, fontweight="bold",
                  loc="left", x=0.0)
    for s in axc.spines.values():
        s.set_visible(False)
    add_colorbar(fig, mesh, axc, label="SAIDI (minutes)")
    axc.text(0.0, -0.135, f"annual total fell "
             f"{grid[0].sum():.0f} \u2192 {grid[-1].sum():.0f} minutes "
             f"({100 * (grid[-1].sum() / grid[0].sum() - 1):+.0f}%)",
             transform=axc.transAxes, fontsize=6.3, color=GREY, ha="left",
             va="top")

    # ---- (d) feeder ranking ------------------------------------------------
    saidi = np.sort(28.0 + 165.0 * rng.beta(1.9, 2.3, NFEED))[::-1]
    fx = np.arange(NFEED)
    cols = [BAD if v > TARGET else SLATE for v in saidi]
    axd.bar(fx, saidi, width=0.64, color=cols, lw=0, zorder=3)
    axd.axhline(TARGET, color=INK, lw=0.9, ls=(0, (4, 2.2)), zorder=4)
    nbad = int((saidi > TARGET).sum())
    axd.text(NFEED - 0.4, TARGET + 6, f"target {TARGET:.0f} min  \u00b7  "
             f"{nbad} of {NFEED} feeders above it", fontsize=6.4, color=INK,
             ha="right", va="bottom", zorder=6)
    for x, v in zip(fx[:3], saidi[:3]):
        axd.text(x, v + 5, f"{v:.0f}", fontsize=6.2, color=BAD,
                 ha="center", va="bottom", zorder=4)
    axd.set_xticks(fx)
    axd.set_xticklabels([f"F{i + 1:02d}" for i in fx], fontsize=6.5)
    axd.set_xlim(-0.7, NFEED - 0.3)
    axd.set_ylim(0, saidi.max() * 1.24)
    axd.set_ylabel("SAIDI (minutes)", fontsize=7.4)
    axd.set_xlabel("Feeder, ranked worst to best", fontsize=7.4)
    axd.grid(axis="y", lw=0.4, alpha=0.5)
    axd.tick_params(axis="x", length=0)
    style_axes(axd)


    fw, fh = fig.get_size_inches()
    for a, lab in ((axk[0], "(a)"), (axb, "(b)"), (axc, "(c)"),
                   (axd, "(d)")):
        bb = a.get_position()
        a.text(-0.44 / (bb.width * fw), 1.0 + 0.055 / (bb.height * fh), lab,
               transform=a.transAxes, ha="left", va="bottom",
               fontweight="bold", fontsize=10)

    print(f"SAIDI={KPIS[0][1]}  SAIFI x CAIDI = "
          f"{KPIS[1][1] * KPIS[2][1]:.1f} minutes")
    print(f"causes total={cnt.sum():.0f}  top-3 share={cum[2]:.0f}%")
    print(f"feeders above target: {nbad}/{NFEED}  "
          f"worst={saidi[0]:.0f} min  best={saidi[-1]:.0f} min")
    savefig(fig, "202_reliability_dashboard.png")


if __name__ == "__main__":
    main()
