"""Die-level thermal simulation on the floorplan of figure 498: a spread
power map turned into a junction-temperature field with isotherms and block
outlines, plus the diagonal temperature profile and per-block peaks."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig, style_axes, get_cmap, lighten
from figkit import image_axes, panel_labels_inches, halo_text, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
BASE = "#808080"

DIE_W, DIE_H = 5200.0, 4400.0     # um, same die as figure 498
PX = 20.0                          # simulation pitch, um
T_AMB = 45.0                       # lid-side coolant baseline, degC
RTH = 95.0                         # degC per W/mm^2 of spread power
SPREAD = 340.0                     # lateral heat-spreading length, um
TJ_MAX = 100.0                     # junction temperature limit, degC

CLASS_COLOR = {"cpu": BLUE, "gpu": lighten(BLUE, 0.30), "sram": GREEN,
               "io": GOLD, "pll": PURPLE}

# name, x0, y0, w, h (um), class, power density (W/mm^2) - floorplan of 498
BLOCKS = [
    ("CPU 0", 380, 3260, 940, 760, "cpu", 0.81),
    ("CPU 1", 1410, 3260, 940, 760, "cpu", 0.75),
    ("CPU 2", 380, 2410, 940, 760, "cpu", 0.71),
    ("CPU 3", 1410, 2410, 940, 760, "cpu", 0.84),
    ("L3 SRAM", 2440, 2410, 900, 1610, "sram", 0.19),
    ("GPU", 3430, 2410, 1390, 1610, "gpu", 0.59),
    ("L2 SRAM A", 380, 1430, 1180, 800, "sram", 0.15),
    ("L2 SRAM B", 1650, 1430, 1180, 800, "sram", 0.15),
    ("NoC", 2920, 1430, 1900, 800, "cpu", 0.32),
    ("PLL", 380, 380, 560, 900, "pll", 0.43),
    ("DDR PHY", 1030, 380, 1800, 900, "io", 0.28),
    ("SerDes PHY", 2920, 380, 1900, 900, "io", 0.38),
]


def thermal_field():
    """Power density map, Gaussian-spread into a junction-temperature field."""
    nx, ny = int(DIE_W / PX), int(DIE_H / PX)
    p = np.zeros((ny, nx))
    for _n, x0, y0, w, h, _k, pd in BLOCKS:
        i0, i1 = int(y0 / PX), int((y0 + h) / PX)
        j0, j1 = int(x0 / PX), int((x0 + w) / PX)
        p[i0:i1, j0:j1] = pd
    t = T_AMB + RTH * gaussian_filter(p, SPREAD / PX, mode="nearest")
    return p, t


def block_peaks(t):
    out = []
    for name, x0, y0, w, h, kind, pd in BLOCKS:
        i0, i1 = int(y0 / PX), int((y0 + h) / PX)
        j0, j1 = int(x0 / PX), int((x0 + w) / PX)
        sub = t[i0:i1, j0:j1]
        out.append((name, float(sub.max()), CLASS_COLOR[kind],
                    pd * w * h / 1e6))
    return out


def map_panel(fig, ax, t, tmax):
    ext = [0, DIE_W / 1000.0, 0, DIE_H / 1000.0]        # mm
    im = ax.imshow(t, origin="lower", extent=ext, cmap=get_cmap(
        "cb_sequential_warm"), vmin=T_AMB, vmax=tmax, interpolation="bilinear",
        zorder=1)
    xs = np.linspace(0, DIE_W / 1000.0, t.shape[1])
    ys = np.linspace(0, DIE_H / 1000.0, t.shape[0])
    cs = ax.contour(xs, ys, t, levels=np.arange(50, int(tmax) + 1, 8),
                    colors="#3a3a3a", linewidths=0.5, alpha=0.55, zorder=3)
    ax.clabel(cs, fmt="%d", fontsize=6.0, inline=True, colors="#3a3a3a")
    ax.contour(xs, ys, t, levels=[TJ_MAX], colors=[RED], linewidths=1.3,
               zorder=4)

    for name, x0, y0, w, h, _k, _pd in BLOCKS:
        ax.add_patch(Rectangle((x0 / 1000.0, y0 / 1000.0), w / 1000.0,
                               h / 1000.0, facecolor="none",
                               edgecolor="white", linewidth=0.7, alpha=0.85,
                               zorder=5))
        halo_text(ax, (x0 + w / 2) / 1000.0, (y0 + h / 2) / 1000.0, name,
                  ha="center", va="center", fontsize=6.3, zorder=6)
    ax.add_patch(Rectangle((0, 0), DIE_W / 1000.0, DIE_H / 1000.0,
                           facecolor="none", edgecolor="#333333",
                           linewidth=0.9, zorder=7))

    ax.plot([0, DIE_W / 1000.0], [0, DIE_H / 1000.0], color="#1a1a1a",
            lw=0.9, ls=(0, (4, 2.2)), zorder=8)
    halo_text(ax, 0.30, 0.20, "profile A\u2013A'", ha="left", va="bottom",
              fontsize=6.4, rotation=40, zorder=9)

    ax.set_xlim(-0.06, DIE_W / 1000.0 + 0.06)
    ax.set_ylim(-0.06, DIE_H / 1000.0 + 0.06)
    image_axes(ax, color="#8a8a8a", lw=0.6)
    ax.set_title(f"Junction temperature over the die  \u00b7  isotherms every "
                 f"8 \u00b0C", fontsize=9.0, fontweight="bold", pad=5)
    safe_colorbar(fig, im, ax, label="junction temperature $T_j$ (\u00b0C)",
                  width=0.12, pad=0.08)
    return im


def profile_panel(ax, t, tlim):
    n = 420
    rr = np.linspace(0, t.shape[0] - 1, n)
    cc = np.linspace(0, t.shape[1] - 1, n)
    prof = map_coordinates(t, [rr, cc], order=1)
    d = np.linspace(0, np.hypot(DIE_W, DIE_H) / 1000.0, n)

    diag_len = np.hypot(DIE_W, DIE_H)
    for _name, x0, y0, w, h, kind, _pd in BLOCKS:
        s0 = max(x0 / DIE_W, y0 / DIE_H)
        s1 = min((x0 + w) / DIE_W, (y0 + h) / DIE_H)
        if s1 > s0:
            ax.axvspan(s0 * diag_len / 1000.0, s1 * diag_len / 1000.0,
                       color=CLASS_COLOR[kind], alpha=0.16, lw=0, zorder=0)
    ax.plot(d, prof, color="#8c2f2f", lw=1.7, zorder=4)
    ax.axhline(TJ_MAX, color=RED, lw=0.9, ls=(0, (4, 2.5)), zorder=3)
    ax.text(d[0], TJ_MAX + 0.8, f"$T_j$ limit {TJ_MAX:.0f} \u00b0C",
            ha="left", va="bottom", fontsize=6.3, color=RED)
    ax.axhline(T_AMB, color=BASE, lw=0.8, ls=(0, (2.2, 2.0)), zorder=3)
    ax.text(0.45 * d[-1], T_AMB + 0.8, f"coolant {T_AMB:.0f} \u00b0C",
            ha="center", va="bottom", fontsize=6.3, color=SUB)
    k = int(np.argmax(prof))
    ax.plot([d[k]], [prof[k]], marker="v", ms=5.0, color=RED, zorder=6)
    ax.annotate(f"{prof[k]:.1f} \u00b0C at {d[k]:.2f} mm",
                xy=(d[k], prof[k] + 1.2), xytext=(d[k], tlim[1] - 2.0),
                ha="center", va="top", fontsize=6.3, color=INK,
                arrowprops=dict(arrowstyle="-", color="#8a8a8a", lw=0.6))
    ax.set_xlim(0, d[-1])
    ax.set_ylim(*tlim)
    ax.set_xlabel("Distance along diagonal A\u2013A' (mm)", fontsize=7.5)
    ax.set_ylabel("Temperature (\u00b0C)", fontsize=7.5)
    ax.set_title("Diagonal temperature profile", fontsize=8.5,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)
    return prof


def peaks_panel(ax, peaks, tlim):
    peaks = sorted(peaks, key=lambda r: r[1])
    y = np.arange(len(peaks))
    for yi, (_n, tp, col, _p) in zip(y, peaks):
        over = tp > TJ_MAX
        ax.plot([T_AMB, tp], [yi, yi], color=col, lw=1.5, alpha=0.85,
                zorder=3, solid_capstyle="round")
        ax.plot([tp], [yi], marker="o", ms=4.6, mfc=RED if over else col,
                mec="white", mew=0.7, zorder=4)
        ax.text(tp + 0.9, yi, f"{tp:.1f}", ha="left", va="center",
                fontsize=6.1, color=RED if over else INK,
                fontweight="bold" if over else "normal", zorder=5)
    ax.axvline(TJ_MAX, color=RED, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
    ax.axvline(T_AMB, color=BASE, lw=0.8, ls=(0, (2.2, 2.0)), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in peaks], fontsize=6.3)
    ax.set_xlim(*tlim)
    ax.set_ylim(-0.7, len(peaks) - 0.3)
    ax.set_xlabel("Peak block temperature (\u00b0C), stems from coolant",
                  fontsize=7.5)
    ax.set_title("Per-block peak temperature", fontsize=8.5,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style()
    p, t = thermal_field()
    peaks = block_peaks(t)
    tmax = float(t.max())
    tlim = (T_AMB - 3.0, tmax + 6.0)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.55,
                          left=0.035, right=0.905, top=0.885, bottom=0.105)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_prof = fig.add_subplot(gs[0, 1])
    ax_peak = fig.add_subplot(gs[1, 1])

    map_panel(fig, ax_main, t, tmax)
    profile_panel(ax_prof, t, tlim)
    peaks_panel(ax_peak, peaks, tlim)

    total_w = sum(r[3] for r in peaks)
    hot = [r for r in peaks if r[1] > TJ_MAX]

    panel_labels_inches(fig, [ax_main, ax_prof, ax_peak], dx=-0.34, dy=0.08)
    savefig(fig, "502_semi_thermal_map.png")


if __name__ == "__main__":
    main()
