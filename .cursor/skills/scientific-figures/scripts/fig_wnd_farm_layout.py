"""Wind farm layout: per-turbine yield, wake shadows and the cost of a row.

The main panel is the farm plan drawn on an equal-aspect axes with a scale
bar: 42 machines on a staggered 7D x 5D grid, each coloured by its modelled
annual yield and trailing the wake shadow it casts under the prevailing
westerly.  Positions that lose more than the upper-quartile share of their
free-standing yield are ringed, and the inset rose gives the direction
distribution the layout was drawn for.

One companion panel shows how the loss deepens row by row into the farm;
the other re-runs the same wake model on three candidate layouts, so the
array efficiencies compared there are computed, not quoted.
"""
import textwrap

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from pubstyle import set_pub_style, savefig, get_cmap, darken
from figkit import (axes_inches, panel_labels_inches, safe_colorbar,
                    add_scale_bar, halo_text)

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
GRID = "#d9d9d9"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"

DIAM = 126.0
U_INF = 9.0
CT = 0.80
KSTAR = 0.052
AEP_FREE = 23.0          # GWh/a of a free-standing machine (see figure 715)
YIELD_EXP = 2.2          # yield scales as (u / U_inf) ** YIELD_EXP
N_COL, N_ROW = 6, 7
DX, DY = 7.0 * DIAM, 5.0 * DIAM
X0, Y0 = 620.0, 900.0
XLIM = (0.0, 6300.0)
YLIM = (0.0, 5232.0)
WD = 270.0               # prevailing direction the wind blows from (deg)


def layout(kind):
    """Turbine coordinates (m) for the three candidate arrangements."""
    dx = 9.0 * DIAM if kind == "stretched" else DX
    pts = []
    for i in range(N_COL):
        for j in range(N_ROW):
            off = 0.5 * DY if (kind == "staggered" and i % 2) else 0.0
            pts.append((X0 + dx * i, Y0 + DY * j + off))
    return np.asarray(pts)


def sigma_m(xd):
    """Gaussian wake width (m) at downstream distance x / D."""
    beta = 0.5 * (1.0 + np.sqrt(1.0 - CT)) / np.sqrt(1.0 - CT)
    return (KSTAR * np.maximum(xd, 1.6) + 0.2 * np.sqrt(beta)) * DIAM


def wake_speeds(pts, wd_deg):
    """Hub speed at every turbine for one direction, deficits added in quadrature."""
    th = np.deg2rad(wd_deg)
    down = np.array([-np.sin(th), -np.cos(th)])     # unit vector downwind
    cross = np.array([down[1], -down[0]])
    s, n = pts @ down, pts @ cross
    dx = s[:, None] - s[None, :]                    # positive: j is upwind of i
    dy = np.abs(n[:, None] - n[None, :])
    sg = sigma_m(dx / DIAM)
    c = 1.0 - np.sqrt(np.maximum(1.0 - CT / (8.0 * (sg / DIAM) ** 2), 1e-6))
    d = np.where(dx > 0.5 * DIAM, c * np.exp(-0.5 * (dy / sg) ** 2), 0.0)
    return U_INF * (1.0 - np.sqrt((d ** 2).sum(axis=1)))


def yields(pts, sectors):
    """Yield per turbine (GWh/a), averaged over the direction distribution."""
    aep = np.zeros(pts.shape[0])
    for wd, f in sectors:
        aep += f * AEP_FREE * (wake_speeds(pts, wd) / U_INF) ** YIELD_EXP
    return aep


def wake_polygon(x, y, length=9.0 * DIAM, n=24):
    """Downwind shadow of one rotor under a westerly: the two-sigma envelope."""
    xs = np.linspace(0.0, length, n)
    half = np.maximum(0.5 * DIAM, 2.0 * sigma_m(xs / DIAM))
    pts = ([(x + a, y + b) for a, b in zip(xs, half)] +
           [(x + a, y - b) for a, b in zip(xs[::-1], half[::-1])])
    return Polygon(pts, closed=True, facecolor=BLUE, edgecolor="none",
                   alpha=0.11, zorder=2)


def wind_rose(rng, n_sector=12):
    """Sector frequencies (%) for a westerly-dominated site."""
    ang = np.deg2rad(np.arange(n_sector) * 360.0 / n_sector)
    mu = np.deg2rad(WD)
    f = np.exp(2.6 * np.cos(ang - mu)) + 0.35 * np.exp(1.4 * np.cos(ang - mu + 2.4))
    f = f * (1.0 + 0.05 * rng.standard_normal(n_sector))
    return ang, 100.0 * f / f.sum()


def main():
    set_pub_style()
    rng = np.random.default_rng(717)

    ang, freq = wind_rose(rng)
    sectors = list(zip(np.rad2deg(ang), freq / 100.0))

    pts = layout("staggered")
    aep = yields(pts, sectors)
    loss = 100.0 * (1.0 - aep / AEP_FREE)
    col_of = np.repeat(np.arange(N_COL), N_ROW)

    farm_aep = float(aep.sum())
    ideal_aep = AEP_FREE * pts.shape[0]
    eff_plot = 100.0 * farm_aep / ideal_aep
    q75 = float(np.percentile(loss, 75.0))
    hit = loss >= q75
    worst = int(np.argmax(loss))

    row_mean = np.array([loss[col_of == i].mean() for i in range(N_COL)])
    row_sd = np.array([loss[col_of == i].std() for i in range(N_COL)])

    schemes = [("Aligned\n7D x 5D", "aligned"),
               ("Staggered\n7D x 5D", "staggered"),
               ("Stretched\n9D x 5D", "stretched")]
    eff = np.asarray([100.0 * yields(layout(k), sectors).sum() / ideal_aep
                      for _, k in schemes])

    fig = plt.figure(figsize=(10.4, 6.0))
    ax = axes_inches(fig, (0.88, 0.82, 4.60, 3.82))
    ax1 = axes_inches(fig, (6.90, 2.90, 2.70, 1.74))
    ax2 = axes_inches(fig, (6.90, 0.82, 2.70, 1.55))

    # ---- main panel: the farm plan ---------------------------------------
    for (x, y) in pts:
        ax.add_patch(wake_polygon(x, y))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=aep, s=46, marker="o",
                    cmap=get_cmap("cb_sequential_blue"), edgecolor="#333333",
                    linewidths=0.5, zorder=5)
    ax.scatter(pts[hit, 0], pts[hit, 1], s=118, facecolor="none",
               edgecolor=RED, linewidths=0.9, zorder=6)
    halo_text(ax, pts[worst, 0] + 155.0, pts[worst, 1] - 265.0,
              f"worst {loss[worst]:.1f} %", fontsize=6.5, color=RED,
              halo="white", halo_lw=2.0, ha="left", va="top", zorder=8)

    ax.annotate("", xy=(1180.0, 5130.0), xytext=(320.0, 5130.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK,
                                mutation_scale=10))
    ax.text(1260.0, 5130.0, f"prevailing wind, {WD:.0f} deg", fontsize=6.8,
            color=INK, ha="left", va="center")
    ax.text(2040.0, 790.0,
            f"n = {pts.shape[0]} turbines, 6 MW each\n"
            f"{farm_aep:.0f} GWh/a total, array efficiency {eff_plot:.1f} %\n"
            f"red ring: loss $\\geq$ {q75:.1f} % (upper quartile)",
            fontsize=6.6, color=INK, ha="left", va="top", linespacing=1.4,
            zorder=8, bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                                edgecolor="#cccccc", linewidth=0.6, alpha=0.94))

    add_scale_bar(ax, 1000.0, "1 km", loc=(0.035, 0.045), height=0.016,
                  color="#333333", alt="white", text_color=INK, halo="white",
                  fontsize=6.4, zorder=7)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)", fontsize=8.0)
    ax.set_ylabel("Northing (m)", fontsize=8.0)
    ax.set_title("Farm plan with wake shadows and per-turbine yield",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(labelsize=7.2)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#8a8a8a")

    # wind rose inset, on an opaque card so the shadows do not read through
    ax.add_patch(Rectangle((0.795, 0.020), 0.195, 0.265,
                           transform=ax.transAxes, facecolor="white",
                           edgecolor="#cccccc", linewidth=0.6, zorder=6))
    axr = ax.inset_axes([0.822, 0.038, 0.142, 0.190], projection="polar")
    axr.set_zorder(7)
    bar_col = [darken(BLUE, 0.35) if abs(np.rad2deg(a) - WD) < 1.0 else BLUE
               for a in ang]
    axr.bar(0.5 * np.pi - ang, freq, width=2 * np.pi / len(ang) * 0.86,
            color=bar_col, edgecolor="white", linewidth=0.3)
    axr.set_xticks(0.5 * np.pi - np.deg2rad([0.0, 90.0, 180.0, 270.0]))
    axr.set_xticklabels(["N", "E", "S", "W"], fontsize=6.0, color=SUB)
    axr.set_yticks([])
    axr.tick_params(pad=-2.0)
    axr.grid(color="#e0e0e0", lw=0.4)
    axr.spines["polar"].set_color("#cccccc")
    axr.spines["polar"].set_linewidth(0.5)
    ax.text(0.892, 0.292, "wind rose, sector share", transform=ax.transAxes,
            fontsize=6.2, color=SUB, ha="center", va="bottom", zorder=7)

    safe_colorbar(fig, sc, ax, label="annual yield (GWh/a)", width=0.11,
                  pad=0.09, labelsize=6.6)

    # ---- aux 1: loss by row into the farm ---------------------------------
    rows = np.arange(1, N_COL + 1)
    ax1.bar(rows, row_mean, width=0.62, color=RED, alpha=0.85,
            edgecolor="white", linewidth=0.4, zorder=3)
    ax1.errorbar(rows, row_mean, yerr=row_sd, fmt="none", ecolor=INK,
                 elinewidth=0.8, capsize=2.2, zorder=4)
    for r, m in zip(rows, row_mean):
        ax1.text(r, m + row_sd[r - 1] + 0.4, f"{m:.1f}", fontsize=6.2,
                 color=INK, ha="center", va="bottom")
    ax1.set_xticks(rows)
    ax1.set_xlim(0.4, N_COL + 0.6)
    ax1.set_ylim(0.0, float(row_mean.max() + row_sd.max()) * 1.28)
    ax1.set_xlabel("Row from the upwind edge (-)", fontsize=7.2)
    ax1.set_ylabel("Wake loss (%)", fontsize=7.2)
    ax1.set_title("Loss deepens over the first rows", fontsize=8.5,
                  fontweight="bold")
    ax1.tick_params(labelsize=6.5)
    ax1.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.8)
    ax1.set_axisbelow(True)

    # ---- aux 2: candidate layouts -----------------------------------------
    names = [n for n, _ in schemes]
    cols = [BLUE if k == "staggered" else "#b3b3b3" for _, k in schemes]
    ax2.bar(np.arange(len(eff)), eff, width=0.58, color=cols,
            edgecolor="white", linewidth=0.4, zorder=3)
    for i, v in enumerate(eff):
        ax2.text(i, v + 0.35, f"{v:.1f} %", fontsize=6.5, color=INK,
                 ha="center", va="bottom")
    ax2.set_xticks(np.arange(len(eff)))
    ax2.set_xticklabels(names, fontsize=6.5, linespacing=1.3)
    ax2.set_ylim(0.0, float(eff.max()) * 1.16)
    ax2.set_ylabel("Array efficiency (%)", fontsize=7.2)
    ax2.set_title("Same wake model, three layouts", fontsize=8.5,
                  fontweight="bold")
    ax2.tick_params(labelsize=6.5)
    ax2.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.text(0.98, 0.06, "plotted layout in blue", transform=ax2.transAxes,
             fontsize=6.4, color=SUB, ha="right", va="bottom")

    panel_labels_inches(fig, [ax, ax1, ax2], dx=-0.52, dy=0.10)


    savefig(fig, "717_wnd_farm_layout.png")


if __name__ == "__main__":
    main()
