"""Ground-settlement monitoring around a deep excavation.

One settlement basin drives the whole figure: the contour map is the modelled
trough, and each monitoring point's time series is scaled by the value of that
same field at its own coordinates, so the map and the curves cannot disagree.
Settlement is plotted positive downward.  Alert and alarm thresholds are ruled
across both time panels, and the worst point is picked automatically from the
final readings and called out in both views.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pubstyle import (set_pub_style, savefig, add_panel_label, get_cmap,
                      style_axes)
from figkit import safe_colorbar

PIT = (95.0, 70.0, 70.0, 56.0)         # x0, y0, w, h in metres
SMAX = 38.0                            # mm at the trough axis
POINTS = [
    ("P1", 82.0, 98.0, "#b23a48"),
    ("P2", 178.0, 92.0, "#3b5b92"),
    ("P3", 132.0, 44.0, "#2e7d5b"),
    ("P4", 214.0, 150.0, "#c9a227"),
    ("P5", 46.0, 172.0, "#8e6bbf"),
]
ALERT, ALARM = 18.0, 28.0              # mm
RATE_ALARM = 0.45                      # mm/day
STAGES = [(40.0, 150.0, "excavation"), (150.0, 214.0, "base slab"),
          (214.0, 300.0, "backfill")]
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"


def basin(x, y):
    """Settlement magnitude (mm) of the trough around the excavation."""
    x0, y0, w, h = PIT
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    dx = np.maximum(np.abs(x - cx) - w / 2.0, 0.0)
    dy = np.maximum(np.abs(y - cy) - h / 2.0, 0.0)
    d = np.hypot(dx, dy)
    lobe = 1.0 + 0.22 * np.cos(np.arctan2(y - cy, x - cx) - 0.6)
    return SMAX * lobe * np.exp(-(d / 58.0) ** 1.6)


def history(t, s_final, rng):
    """Staged consolidation curve reaching ``s_final`` mm."""
    exc = 0.60 * (1.0 - np.exp(-np.maximum(t - 40.0, 0.0) / 42.0))
    slab = 0.28 * (1.0 - np.exp(-np.maximum(t - 150.0, 0.0) / 34.0))
    creep = 0.12 * (1.0 - np.exp(-np.maximum(t - 214.0, 0.0) / 70.0))
    s = s_final * (exc + slab + creep)
    return s + rng.normal(0.0, 0.28, t.size) * (t > 0)


def main():
    set_pub_style(**{"hatch.linewidth": 0.45})
    rng = np.random.default_rng(41008)

    t = np.arange(0.0, 361.0, 5.0)
    series, finals = {}, {}
    for name, px, py, col in POINTS:
        s = history(t, float(basin(np.array(px), np.array(py))), rng)
        series[name] = s
        finals[name] = s[-1]
    worst = max(finals, key=finals.get)
    wx, wy = next((p[1], p[2]) for p in POINTS if p[0] == worst)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.66],
                          height_ratios=[1.0, 0.60], left=0.058, right=0.955,
                          top=0.860, bottom=0.150, hspace=0.24, wspace=0.20)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0], sharex=ax_s)
    ax_m = fig.add_subplot(gs[:, 1])

    # ---- (a) settlement histories ----------------------------------------
    for (t0, t1, lab), sh in zip(STAGES, ("#f5f5f5", "#ececec", "#f5f5f5")):
        for ax in (ax_s, ax_r):
            ax.axvspan(t0, t1, color=sh, lw=0, zorder=0)
        ax_s.text(0.5 * (t0 + t1), 0.985, lab,
                  transform=ax_s.get_xaxis_transform(), ha="center",
                  va="top", fontsize=6.4, color=MUTED)

    for name, px, py, col in POINTS:
        ax_s.plot(t, series[name], color=col, lw=1.6, zorder=4,
                  label=f"{name} ({finals[name]:.1f} mm)")
    for lvl, lab, col in ((ALERT, "alert", "#c9a227"),
                          (ALARM, "alarm", "#b23a48")):
        ax_s.axhline(lvl, color=col, lw=1.0, ls=(0, (5, 2.2)), zorder=3)
        ax_s.text(358.0, lvl - 0.9, f"{lab} {lvl:.0f} mm", ha="right",
                  va="bottom", fontsize=6.4, color=col, fontweight="bold",
                  zorder=5)
    ax_s.plot([t[-1]], [finals[worst]], marker="o", ms=5.4, mfc="white",
              mec="#b23a48", mew=1.4, zorder=6)
    ax_s.annotate(f"{worst} exceeds the alarm level\n"
                  f"{finals[worst]:.1f} mm at day {t[-1]:.0f}",
                  xy=(t[-1], finals[worst]),
                  xytext=(202.0, finals[worst] * 1.11),
                  fontsize=6.9, color=INK, ha="left", va="center",
                  linespacing=1.3, zorder=7,
                  bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                            edgecolor="#cccccc", linewidth=0.6, alpha=0.94),
                  arrowprops=dict(arrowstyle="->", color=RULE, lw=0.8,
                                  connectionstyle="arc3,rad=0.18"))
    ax_s.set_ylim(max(finals.values()) * 1.22, -1.5)
    ax_s.set_ylabel("Settlement (mm, positive down)")
    ax_s.tick_params(labelbottom=False)
    ax_s.grid(color="#ececec", lw=0.5)
    ax_s.set_axisbelow(True)
    style_axes(ax_s)
    ax_s.set_title("Cumulative settlement at five monitoring points",
                   fontsize=8.5, fontweight="bold")

    # ---- (b) settlement rate ---------------------------------------------
    k = np.ones(5) / 5.0
    for name, px, py, col in POINTS:
        sm = np.convolve(series[name], k, mode="same")
        sm[:2], sm[-2:] = series[name][:2], series[name][-2:]
        ax_r.plot(t, np.gradient(sm, t), color=col, lw=1.4, zorder=4)
    ax_r.axhline(RATE_ALARM, color="#b23a48", lw=1.0, ls=(0, (5, 2.2)),
                 zorder=3)
    ax_r.text(358.0, RATE_ALARM + 0.015, f"rate alarm {RATE_ALARM:.2f} "
              "mm d$^{-1}$", ha="right", va="bottom", fontsize=6.4,
              color="#b23a48", fontweight="bold")
    ax_r.axhline(0.0, color="#808080", lw=0.7, zorder=2)
    ax_r.set_xlim(0.0, 360.0)
    ax_r.set_xlabel("Days since start of construction (d)")
    ax_r.set_ylabel("Rate (mm d$^{-1}$)", fontsize=7.6)
    ax_r.tick_params(labelsize=6.8)
    ax_r.grid(color="#ececec", lw=0.5)
    ax_r.set_axisbelow(True)
    style_axes(ax_r)

    # ---- (c) settlement contour map --------------------------------------
    gx = np.linspace(0.0, 270.0, 260)
    gy = np.linspace(0.0, 210.0, 210)
    GX, GY = np.meshgrid(gx, gy)
    Z = basin(GX, GY)
    levels = np.arange(0.0, SMAX * 1.25, 4.0)
    cf = ax_m.contourf(GX, GY, Z, levels=levels,
                       cmap=get_cmap("cb_sequential_warm"), extend="max")
    cf.set_edgecolor("face")
    cl = ax_m.contour(GX, GY, Z, levels=levels[1::2], colors="#8c5a2b",
                      linewidths=0.5)
    ax_m.clabel(cl, fmt="%.0f", fontsize=6.0, inline=True)

    x0, y0, w, h = PIT
    ax_m.add_patch(Rectangle((x0, y0), w, h, facecolor="white",
                             edgecolor=INK, linewidth=1.1, hatch="//",
                             zorder=5))
    ax_m.text(x0 + w / 2, y0 + h / 2, "excavation", ha="center", va="center",
              fontsize=6.8, color=INK, fontweight="bold", zorder=6)
    for name, px, py, col in POINTS:
        ax_m.plot([px], [py], marker="s", ms=5.0, mfc=col, mec="white",
                  mew=0.8, zorder=7)
        ax_m.text(px + 6.0, py + 5.0, name, fontsize=6.6, color=INK,
                  fontweight="bold", zorder=7)
    ax_m.plot([wx], [wy], marker="o", ms=11.0, mfc="none", mec="#b23a48",
              mew=1.4, zorder=8)
    ax_m.annotate(f"worst point {worst}\n{finals[worst]:.1f} mm",
                  xy=(wx, wy), xytext=(186.0, 22.0), fontsize=6.6,
                  color=INK, ha="left", va="center", linespacing=1.3,
                  zorder=9,
                  bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                            edgecolor="#cccccc", linewidth=0.6, alpha=0.94),
                  arrowprops=dict(arrowstyle="->", color=RULE, lw=0.8,
                                  connectionstyle="arc3,rad=0.24"))
    ax_m.set_aspect("equal")
    ax_m.set_xlim(0.0, 270.0)
    ax_m.set_ylim(0.0, 210.0)
    ax_m.set_xlabel("Local easting (m)", fontsize=7.6)
    ax_m.set_ylabel("Local northing (m)", fontsize=7.6)
    ax_m.tick_params(labelsize=6.8)
    for s in ax_m.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax_m.set_title("Modelled settlement trough at day 360",
                   fontsize=8.5, fontweight="bold")
    safe_colorbar(fig, cf, ax_m, label="settlement (mm)", side="bottom",
                  width=0.11, pad=0.34, shrink=0.86)

    handles, labels = ax_s.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.34, 0.008), ncol=5, frameon=False,
               fontsize=7.0, handlelength=1.7, columnspacing=1.5)

    add_panel_label(ax_s, "(a)", x=-0.075, y=1.045)
    add_panel_label(ax_r, "(b)", x=-0.075, y=1.045)
    add_panel_label(ax_m, "(c)", x=-0.115, y=1.020)
    savefig(fig, "410_geo_settlement_monitor.png")


if __name__ == "__main__":
    main()
