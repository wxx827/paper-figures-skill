"""Two hundred returns to one taught pose, drawn on an equal-aspect field.

The cloud is not round and it does not stay still: a correlated tool-frame
noise sets its shape and a saturating thermal expansion walks its centre away
from the taught point over the first hour, so the figure has to be metrically
honest to be readable at all - every panel is in micrometres and the main
field has equal axis scales, otherwise the confidence ellipses would be
arbitrary shapes.  The two ellipses are covariance ellipses of the cold and
the soaked subsets, the repeatability radius is the mean distance to a
barycentre plus three standard deviations of those distances, and the drift
quoted in the side panel is the distance between the two barycentres already
drawn on the main field.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3d6b8e"
STEEL = "#6b7f95"
GREEN = "#2e7d5b"

SEED = 1063
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)

N_CYCLE = 200               # returns to the taught pose
CYCLE_S = 27.0              # cycle time, s
TAU_MIN = 21.0              # thermal time constant, min
DRIFT_UM = 118.0            # saturated barycentre drift, um
DRIFT_DEG = 21.0            # direction of the drift in the tool XY plane
DT_RISE = 15.5              # saturated gearbox temperature rise, deg C
T_START = 23.5              # gearbox temperature at the cold start, deg C
COLD_N = 25                 # cycles counted as cold
SOAK_N = 70                 # last cycles counted as thermally soaked
CHI2_95 = 5.991             # two degrees of freedom

BOX = dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="none",
           alpha=0.88)
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "ric_soak", ["#2f5f86", "#8aa3b6", "#c9b9a4", "#c0603f", "#7f2e2e"])


def landings(rng):
    """Correlated tool-frame scatter riding on a saturating thermal drift."""
    k = np.arange(N_CYCLE)
    t_min = k * CYCLE_S / 60.0
    soak = 1.0 - np.exp(-t_min / TAU_MIN)

    ang = np.deg2rad(DRIFT_DEG)
    drift = np.stack([DRIFT_UM * soak * np.cos(ang),
                      DRIFT_UM * soak * np.sin(ang)], axis=1)

    a = np.deg2rad(34.0)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    noise = rng.normal(0.0, 1.0, (N_CYCLE, 2)) * np.array([17.5, 8.4])
    xy = drift + noise @ rot.T
    z = rng.normal(0.0, 11.2, N_CYCLE) + 26.0 * soak
    temp = T_START + DT_RISE * soak
    return k, t_min, xy, z, temp


def cov_ellipse(pts):
    """Barycentre, 95 % covariance ellipse geometry and RP radius, all in um."""
    c = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    w, h = 2.0 * np.sqrt(CHI2_95 * vals)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    d = np.linalg.norm(pts - c, axis=1)
    return c, w, h, ang, float(d.mean() + 3.0 * d.std(ddof=1))


def fit_limits(x, y, aspect, pad=1.16):
    """Symmetric limits that fill the panel without distorting the scale."""
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    sx, sy = (x.max() - x.min()) * pad, (y.max() - y.min()) * pad
    sx = max(sx, sy * aspect)
    sy = max(sy, sx / aspect)
    return (cx - sx / 2, cx + sx / 2), (cy - sy / 2, cy + sy / 2)


def main_panel(fig, ax, t_min, xy, cold, soaked):
    ar = MAIN[2] / MAIN[3]
    (x0, x1), (y0, y1) = fit_limits(xy[:, 0], xy[:, 1], ar)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")

    ax.axhline(0.0, color="#d5d9de", lw=0.7, zorder=1)
    ax.axvline(0.0, color="#d5d9de", lw=0.7, zorder=1)
    ax.plot(0.0, 0.0, marker="+", ms=11, mew=1.4, color=INK, zorder=8)
    ax.text(4.0, -9.0, "taught pose", ha="left", va="top", fontsize=6.4,
            color=INK, zorder=9, bbox=BOX)

    sc = ax.scatter(xy[:, 0], xy[:, 1], c=t_min, cmap=CMAP, s=13.0,
                    linewidths=0.35, edgecolors="white", zorder=6)

    for mask, col, name, anchor, ha in (
            (cold, BLUE, "first {} cycles (cold)".format(COLD_N),
             (0.030, 0.615), "left"),
            (soaked, ORANGE, "last {} cycles (soaked)".format(SOAK_N),
             (0.965, 0.905), "right")):
        c, w, h, ang, rp = cov_ellipse(xy[mask])
        ax.add_patch(Ellipse(c, w, h, angle=ang, facecolor="none",
                             edgecolor=col, lw=1.5, zorder=7))
        ax.add_patch(Circle(c, rp, facecolor="none", edgecolor=col, lw=0.9,
                            ls=(0, (3.6, 2.4)), zorder=7))
        ax.plot(*c, marker="o", ms=5.0, color=col, mec="white", mew=0.9,
                zorder=9)
        ax.annotate("{}\nRP = {:.1f} $\\mu$m".format(name, rp), xy=tuple(c),
                    xytext=anchor, textcoords="axes fraction", ha=ha,
                    va="center", fontsize=6.3, color=col, zorder=10,
                    linespacing=1.34, bbox=BOX,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                    shrinkA=2, shrinkB=4))

    c_cold, _w, _h, _a, _r = cov_ellipse(xy[cold])
    c_soak, _w, _h, _a, _r = cov_ellipse(xy[soaked])
    d = float(np.linalg.norm(c_soak - c_cold))
    ax.annotate("", xy=tuple(c_soak), xytext=tuple(c_cold),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                                shrinkA=5, shrinkB=5), zorder=8)
    mid = 0.5 * (c_cold + c_soak)
    ax.text(mid[0], mid[1] + 12.0,
            "barycentre walks {:.0f} $\\mu$m".format(d), ha="center",
            va="bottom", fontsize=6.6, color=RED, zorder=10, bbox=BOX)

    ax.add_patch(mpl.patches.FancyBboxPatch(
        (0.700, 0.020), 0.292, 0.115, transform=ax.transAxes,
        boxstyle="round,pad=0,rounding_size=0.012", facecolor="white",
        edgecolor="#e2e2e2", lw=0.6, alpha=0.92, zorder=9))
    cax = ax.inset_axes([0.735, 0.075, 0.235, 0.028], zorder=10)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("elapsed time (min)", fontsize=6.3, color=INK, labelpad=2.0)
    cb.ax.tick_params(labelsize=6.0, width=0.6, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#666666")

    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=1.5, label="95 % covariance ellipse"),
        Line2D([], [], color=BLUE, lw=0.9, ls=(0, (3.6, 2.4)),
               label="RP = mean + 3 SD radius"),
        Line2D([], [], marker="o", ls="none", color=STEEL, ms=5.0,
               label="subset barycentre")],
        loc="upper left", fontsize=6.2, frameon=True, facecolor="white",
        edgecolor="#dddddd", framealpha=0.95, handlelength=1.8,
        borderaxespad=0.35, labelspacing=0.26)

    ax.set_xlabel("deviation along tool X ($\\mu$m)")
    ax.set_ylabel("deviation along tool Y ($\\mu$m)")
    ax.grid(lw=0.5, alpha=0.28)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Where the tool centre point actually lands, cycle after "
                 "cycle", fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    return d


def drift_panel(ax, t_min, xy, z, temp):
    r = np.linalg.norm(np.column_stack([xy, z]), axis=1)
    win = 15
    ker = np.ones(win) / win
    r_s = np.convolve(r, ker, mode="valid")
    t_s = np.convolve(t_min, ker, mode="valid")
    ax.plot(t_min, r, color="#c3ccd4", lw=0.7, zorder=3)
    ax.plot(t_s, r_s, color=ORANGE, lw=1.7, zorder=6)
    ax.set_xlabel("elapsed time (min)", fontsize=6.8)
    ax.set_ylabel("distance from the taught\npose ($\\mu$m)", fontsize=6.8,
                  color=ORANGE, linespacing=1.3)
    ax.tick_params(axis="y", labelsize=6.6, colors=ORANGE)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.set_xlim(0.0, t_min[-1])

    axr = ax.twinx()
    axr.plot(t_min, temp, color=BLUE, lw=1.3, ls=(0, (4.0, 2.2)), zorder=5)
    axr.set_ylabel("gearbox ($^\\circ$C)", fontsize=6.8, color=BLUE)
    axr.tick_params(axis="y", labelsize=6.6, colors=BLUE)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_color(BLUE)
    axr.spines["left"].set_visible(False)

    ax.text(t_min[-1] * 0.97, r_s.min(),
            "{:.1f} $^\\circ$C rise over {:.0f} min".format(
                temp[-1] - temp[0], t_min[-1]), ha="right", va="bottom",
            fontsize=6.1, color=SUB, zorder=9, bbox=BOX)
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("The drift is the warm-up, not the noise", fontsize=8.0,
                 fontweight="bold", color=INK)


def axis_panel(ax, res, sigmas):
    names = ["X", "Y", "Z"]
    cols = [BLUE, STEEL, GREEN]
    edges = np.linspace(-46.0, 46.0, 25)
    centres = 0.5 * (edges[1:] + edges[:-1])
    for i, (name, col) in enumerate(zip(names, cols)):
        h, _ = np.histogram(res[:, i], bins=edges, density=True)
        base = 2.0 - i
        ax.fill_between(centres, base, base + h / h.max() * 0.78, color=col,
                        alpha=0.30, lw=0, zorder=3)
        ax.plot(centres, base + h / h.max() * 0.78, color=col, lw=1.2,
                zorder=5)
        ax.plot([base * 0 - 3.0 * sigmas[i], 3.0 * sigmas[i]],
                [base - 0.05, base - 0.05], color=col, lw=1.3, zorder=6,
                solid_capstyle="butt")
        for s in (-3.0, 3.0):
            ax.plot([s * sigmas[i]] * 2, [base - 0.14, base + 0.04],
                    color=col, lw=1.0, zorder=6)
        ax.text(-44.0, base + 0.60,
                "{}   SD {:.1f} $\\mu$m,  $\\pm$3 SD {:.0f} $\\mu$m"
                .format(name, sigmas[i], 3.0 * sigmas[i]), ha="left",
                va="center", fontsize=6.2, color=col, zorder=8, bbox=BOX)
    ax.axvline(0.0, color="#c3c8cf", lw=0.8, zorder=2)
    ax.set_xlim(-46.0, 46.0)
    ax.set_ylim(-0.30, 3.05)
    ax.set_yticks([])
    ax.set_xlabel("residual after removing the drift ($\\mu$m)", fontsize=6.8)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.grid(axis="x", lw=0.5, alpha=0.32)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Per-axis spread once the drift is taken out", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
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
    rng = np.random.default_rng(SEED)
    k, t_min, xy, z, temp = landings(rng)

    cold = k < COLD_N
    soaked = k >= N_CYCLE - SOAK_N

    soak = 1.0 - np.exp(-t_min / TAU_MIN)
    trend = np.column_stack([
        np.polyval(np.polyfit(soak, xy[:, 0], 1), soak),
        np.polyval(np.polyfit(soak, xy[:, 1], 1), soak),
        np.polyval(np.polyfit(soak, z, 1), soak)])
    res = np.column_stack([xy, z]) - trend
    sigmas = res.std(axis=0, ddof=1)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_dr = axes_inches(fig, AUX_T)
    ax_ax = axes_inches(fig, AUX_B)

    drift = main_panel(fig, ax_main, t_min, xy, cold, soaked)
    drift_panel(ax_dr, t_min, xy, z, temp)
    axis_panel(ax_ax, res, sigmas)

    _c, _w, _h, _a, rp_cold = cov_ellipse(xy[cold])
    _c, _w, _h, _a, rp_soak = cov_ellipse(xy[soaked])



    panel_labels_inches(fig, [ax_main, ax_dr, ax_ax], dx=-0.66, dy=0.14)
    savefig(fig, "1063_ric_pose_repeatability.png")


if __name__ == "__main__":
    main()
