"""Package warpage: a saddle-shaped displacement field and where it comes from.

Main panel: out-of-plane displacement over the 15 x 15 mm footprint of a
moulded package on an equal-aspect canvas, drawn with a diverging map whose
neutral tone is pinned to zero displacement so the two lobes of the saddle
read as opposite signs rather than as "more" and "less".  Labelled isolines
and the two diagonal cut lines A-A' and B-B' are overlaid.

The right-hand panels are the two sections a warpage measurement report
always carries: the profiles along those two diagonals, which curve in
opposite directions because the shape is a saddle, and the warpage-versus-
temperature loop through a reflow pass, whose heating and cooling branches do
not retrace because the mould compound relaxes above its glass transition.

The profiles are interpolated out of the same field array that is imaged, and
every quoted number - peak-to-valley, coplanarity margin, residual warpage -
is measured from those arrays.  Values are illustrative only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig, style_axes, get_cmap, add_footer
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (0.42, 0.74, 5.30, 4.68)
AUX_T_RECT = (7.16, 3.50, 2.88, 1.88)
AUX_B_RECT = (7.16, 0.80, 2.88, 1.88)

SIDE = 15.0                        # mm, package footprint
N = 241                            # samples per side
SPEC = 100.0                       # um, coplanarity limit at room temperature
T_PEAK = 260.0                     # degC, peak reflow temperature
T_G = 125.0                        # degC, mould-compound glass transition


def warpage_field(seed=9081):
    """Saddle-dominated out-of-plane displacement in um on a square grid."""
    rng = np.random.default_rng(seed)
    g = np.linspace(-1.0, 1.0, N)
    u, v = np.meshgrid(g, g)
    w = (46.0 * u * v + 13.0 * (u ** 2 + v ** 2 - 0.62)
         + 5.0 * (u ** 2 - v ** 2))
    w = w + gaussian_filter(rng.normal(0.0, 9.0, w.shape), 14.0)
    return u * SIDE / 2.0, v * SIDE / 2.0, w


def diagonal(w, anti=False):
    """Sample the field along a corner-to-corner diagonal; returns (s, w)."""
    k = np.linspace(0.0, N - 1.0, 320)
    rows = (N - 1.0 - k) if anti else k
    prof = map_coordinates(w, np.vstack([rows, k]), order=1, mode="nearest")
    s = np.linspace(-np.sqrt(2.0) * SIDE / 2.0, np.sqrt(2.0) * SIDE / 2.0,
                    k.size)
    return s, prof


def reflow_loop():
    """Warpage of the A-A' diagonal against temperature, heating and cooling."""
    t_up = np.linspace(25.0, T_PEAK, 240)
    soft = 1.0 / (1.0 + np.exp(-(t_up - T_G) / 16.0))
    w_up = 68.0 - 0.150 * (t_up - 25.0) - 62.0 * soft
    t_dn = np.linspace(T_PEAK, 25.0, 240)
    soft_d = 1.0 / (1.0 + np.exp(-(t_dn - (T_G + 26.0)) / 20.0))
    w_dn = w_up[-1] + (0.226 * (T_PEAK - t_dn)) * (1.0 - 0.42 * soft_d)
    return t_up, w_up, t_dn, w_dn


def path_arrow(ax, x, y, i, color, ms=9):
    """Direction arrow riding on a plotted path at sample ``i``."""
    ax.add_patch(FancyArrowPatch((x[i], y[i]), (x[i + 3], y[i + 3]),
                                 arrowstyle="-|>", mutation_scale=ms, lw=0.0,
                                 color=color, shrinkA=0, shrinkB=0, zorder=8))


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches any axis label or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        for art in (a.xaxis.label, a.yaxis.label, a.title):
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:40])


def main():
    set_pub_style(base_font=8)
    xx, yy, w = warpage_field()
    pv = float(w.max() - w.min())
    s_a, prof_a = diagonal(w, anti=False)
    s_b, prof_b = diagonal(w, anti=True)
    t_up, w_up, t_dn, w_dn = reflow_loop()
    residual = float(w_dn[-1])
    opened = float(w_up[0] - w_dn[-1])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_p = axes_inches(fig, AUX_T_RECT)
    ax_h = axes_inches(fig, AUX_B_RECT)
    for a in (ax_p, ax_h):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the displacement field --------------------------------------
    ax.set_aspect("equal")
    norm = Normalize(-max(abs(w.min()), abs(w.max())),
                     max(abs(w.min()), abs(w.max())))
    cmap = get_cmap("cb_diverging")
    im = ax.pcolormesh(xx, yy, w, cmap=cmap, norm=norm, shading="gouraud",
                       zorder=2, rasterized=True)
    cs = ax.contour(xx, yy, w, levels=np.arange(-60, 81, 20), colors="#33333388",
                    linewidths=0.6, zorder=4)
    ax.clabel(cs, fmt="%.0f", fontsize=6.0, inline=True, inline_spacing=3)
    ax.contour(xx, yy, w, levels=[0.0], colors=[INK], linewidths=1.2,
               zorder=5)
    ax.add_patch(Rectangle((-SIDE / 2, -SIDE / 2), SIDE, SIDE,
                           facecolor="none", edgecolor="#9aa3ab", lw=0.9,
                           zorder=6))

    h = SIDE / 2.0
    ax.plot([-h, h], [-h, h], color=RED, lw=1.3, ls=(0, (4.0, 2.2)), zorder=7)
    ax.plot([-h, h], [h, -h], color=BLUE, lw=1.3, ls=(0, (4.0, 2.2)),
            zorder=7)
    for (px, py, lab, col, ha, va) in [
            (-h, -h, "A", RED, "right", "top"),
            (h, h, "A'", RED, "left", "bottom"),
            (-h, h, "B", BLUE, "right", "bottom"),
            (h, -h, "B'", BLUE, "left", "top")]:
        ax.text(px + (0.30 if ha == "left" else -0.30),
                py + (0.30 if va == "bottom" else -0.30), lab, color=col,
                fontsize=8.0, fontweight="bold", ha=ha, va=va, zorder=8)

    i_hi = np.unravel_index(np.argmax(w), w.shape)
    i_lo = np.unravel_index(np.argmin(w), w.shape)
    for idx, lab, col in ((i_hi, f"+{w.max():.0f}", RED),
                          (i_lo, f"{w.min():.0f}", BLUE)):
        px, py = float(xx[idx]), float(yy[idx])
        ox, oy = -0.55 * np.sign(px), -0.55 * np.sign(py)
        ax.plot([px], [py], marker="o", ms=4.2, mfc="white", mec=INK,
                mew=1.1, zorder=9)
        halo_text(ax, px + ox, py + oy, f"{lab} $\\mu$m",
                  ha="right" if ox < 0 else "left",
                  va="top" if oy < 0 else "bottom", fontsize=6.8, color=col,
                  halo="white", halo_lw=2.2, zorder=10)
    ax.set_xlim(-h - 0.55, h + 0.55)
    ax.set_ylim(-h - 0.55, h + 0.55)
    ax.set_xticks(np.arange(-6, 7, 3))
    ax.set_yticks(np.arange(-6, 7, 3))
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("package x (mm)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("package y (mm)", fontsize=8.4)
    ax.set_title("Out-of-plane displacement at 25 $^\\circ$C, "
                 f"{SIDE:.0f} x {SIDE:.0f} mm moulded package",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.030, 0.030,
            f"peak-to-valley {pv:.0f} $\\mu$m\n"
            f"coplanarity limit {SPEC:.0f} $\\mu$m ({SPEC - pv:+.0f} "
            f"$\\mu$m margin)\nheavy black line = zero displacement",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8,
            color=INK, linespacing=1.50, zorder=11,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))

    safe_colorbar(fig, im, ax, label="out-of-plane displacement ($\\mu$m)",
                  width=0.13, pad=0.10, shrink=0.86, labelsize=7.0,
                  ticks=np.arange(-60, 81, 20))

    # ---- aux (b): the two diagonal profiles --------------------------------
    ax_p.axhline(0.0, color=FAINT, lw=0.7, zorder=1)
    ax_p.plot(s_a, prof_a, color=RED, lw=1.7, zorder=5)
    ax_p.plot(s_b, prof_b, color=BLUE, lw=1.7, zorder=5)
    ax_p.fill_between(s_a, prof_a, prof_b, color="#ededed", zorder=2)
    ax_p.set_xlim(s_a.min(), s_a.max())
    ax_p.set_ylim(min(prof_a.min(), prof_b.min()) - 22.0,
                  max(prof_a.max(), prof_b.max()) + 34.0)
    ax_p.tick_params(labelsize=6.8)
    ax_p.set_xlabel("position along the diagonal (mm)", fontsize=7.5,
                    labelpad=1.5)
    ax_p.set_ylabel("displacement ($\\mu$m)", fontsize=7.5)
    ax_p.set_title("A saddle: the diagonals curve apart", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    mid = prof_a.size // 2
    drop_a = float(prof_a[[0, -1]].mean() - prof_a[mid])
    lift_b = float(prof_b[mid] - prof_b[[0, -1]].mean())
    ax_p.text(0.035, 0.95,
              f"A-A' sits {drop_a:.0f} $\\mu$m higher at its ends than at "
              f"the centre,\nB-B' runs {lift_b:.0f} $\\mu$m the other way "
              f"over the same span",
              transform=ax_p.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)
    ax_p.legend(handles=[Line2D([], [], color=RED, lw=1.7, label="A-A'"),
                         Line2D([], [], color=BLUE, lw=1.7, label="B-B'")],
                loc="lower right", frameon=False, fontsize=6.6,
                handlelength=1.7, labelspacing=0.30, borderaxespad=0.20)

    # ---- aux (c): the reflow hysteresis loop -------------------------------
    ax_h.axhline(0.0, color=FAINT, lw=0.7, zorder=1)
    ax_h.axvline(T_G, color=GOLD, lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    ax_h.text(T_G + 4.0, 0.965, f"$T_g$ {T_G:.0f} $^\\circ$C",
              transform=ax_h.get_xaxis_transform(), ha="left", va="top",
              fontsize=6.4, color=GOLD)
    ax_h.plot(t_up, w_up, color=RED, lw=1.7, zorder=5)
    ax_h.plot(t_dn, w_dn, color=BLUE, lw=1.7, zorder=5)
    for i in (60, 150, 205):
        path_arrow(ax_h, t_up, w_up, i, RED, ms=11)
        path_arrow(ax_h, t_dn, w_dn, i, BLUE, ms=11)
    ax_h.plot([25.0, 25.0], [w_up[0], w_dn[-1]], color=INK, lw=0.8,
              ls=(0, (2.2, 1.8)), zorder=6)
    ax_h.plot([25.0, 25.0], [w_up[0], w_dn[-1]], ls="none", marker="o",
              ms=3.4, mfc="white", mec=INK, mew=1.1, zorder=7)
    ax_h.set_xlim(10.0, T_PEAK + 14.0)
    ax_h.set_ylim(min(w_up.min(), w_dn.min()) - 26.0, w_up[0] + 34.0)
    ax_h.tick_params(labelsize=6.8)
    ax_h.set_xlabel("package temperature ($^\\circ$C)", fontsize=7.5,
                    labelpad=1.5)
    ax_h.set_ylabel("A-A' warpage ($\\mu$m)", fontsize=7.5)
    ax_h.set_title("Reflow leaves a permanent set", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_h.text(0.035, 0.055,
              f"the branches do not retrace:\nback at 25 $^\\circ$C the part "
              f"holds {residual:.0f} $\\mu$m,\n{opened:.0f} $\\mu$m below "
              f"where it started",
              transform=ax_h.transAxes, ha="left", va="bottom", fontsize=6.4,
              color=SUB, linespacing=1.45)
    ax_h.axvline(T_PEAK, color=FAINT, lw=0.8, ls=(0, (3.0, 2.0)), zorder=2)
    ax_h.text(T_PEAK - 4.0, 0.965, f"peak reflow {T_PEAK:.0f} $^\\circ$C",
              transform=ax_h.get_xaxis_transform(), ha="right", va="top",
              fontsize=6.4, color=SUB)

    panel_labels_inches(fig, [ax, ax_p, ax_h], dx=-0.52, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 908")
    check_layout(fig, foot, [ax, ax_p, ax_h])
    savefig(fig, "908_pkg_warpage_surface.png")


if __name__ == "__main__":
    main()
