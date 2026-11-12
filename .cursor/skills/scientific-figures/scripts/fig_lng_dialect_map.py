"""Dialect geography: a hexagonal survey grid, one isogloss and its gradient.

The main panel is a simplified survey region tiled with equal-area hexagons,
each shaded by the share of informants who used the innovative second-person
plural pronoun.  Isoglosses are contours of the underlying preference surface:
a heavy line at the fifty-percent crossing that separates the two dialect
areas, and dashed lines at the quarter and three-quarter levels that bound the
transition belt.  The supporting panels turn the picture into numbers - the
south-to-north transect through the belt, fitted with a logistic whose midpoint
and ten-to-ninety width are read off the fit, and the mean usage of the three
areas the isoglosses define.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, get_cmap, get_palette, savefig
from figkit import panel_labels_inches, safe_colorbar, halo_text, add_scale_bar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
RED = "#b23a48"
SEQ = get_cmap("cb_sequential_blue")

FW, FH = 10.4, 6.4
MAIN = (0.74, 0.84, 5.00, 4.56)
AUX_X, AUX_W, AUX_H = 7.12, 2.90, 1.98
AUX_TOP = (AUX_X, 3.42, AUX_W, AUX_H)
AUX_BOT = (AUX_X, 0.84, AUX_W, AUX_H)

R_HEX = 9.0                     # hexagon circumradius, km
N_COL, N_ROW = 23, 24
CX, CY = 170.0, 154.0           # centre of the survey region, km
TRANSECT_X, TRANSECT_HALF = 168.0, 26.0
NEW, OLD = "yous", "y'all"


def surface(x, y):
    """Smooth preference surface for the innovative variant, in percent."""
    y0 = 150.0 + 34.0 * np.sin(2.0 * np.pi * x / 420.0) - 0.10 * (x - CX)
    return 100.0 / (1.0 + np.exp(-(y - y0) / 26.0))


def coast(theta):
    return (152.0 + 22.0 * np.cos(2.0 * theta + 0.6)
            + 13.0 * np.cos(3.0 * theta - 1.1))


def inside(x, y):
    dx, dy = x - CX, y - CY
    return np.hypot(dx, dy) <= coast(np.arctan2(dy, dx))


def cells(seed=806):
    rng = np.random.default_rng(seed)
    dx, dy = np.sqrt(3.0) * R_HEX, 1.5 * R_HEX
    xs, ys = [], []
    for row in range(N_ROW):
        for col in range(N_COL):
            xs.append(col * dx + (dx / 2.0 if row % 2 else 0.0))
            ys.append(row * dy)
    x, y = np.asarray(xs), np.asarray(ys)
    keep = inside(x, y)
    x, y = x[keep], y[keep]
    smooth = surface(x, y)
    obs = np.clip(smooth + rng.normal(0.0, 5.2, x.size), 0.0, 100.0)
    return x, y, smooth, obs


def logistic(y, lo, hi, mid, w):
    return lo + (hi - lo) / (1.0 + np.exp(-(y - mid) / w))


def map_panel(fig, ax, x, y, smooth, obs):
    patches = [RegularPolygon((xi, yi), numVertices=6, radius=R_HEX,
                              orientation=0.0) for xi, yi in zip(x, y)]
    pc = PatchCollection(patches, cmap=SEQ, edgecolor="white", linewidth=0.35,
                         zorder=2)
    pc.set_array(obs)
    pc.set_clim(0.0, 100.0)
    ax.add_collection(pc)

    th = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(CX + coast(th) * np.cos(th), CY + coast(th) * np.sin(th),
            color=GREY, lw=1.0, zorder=5)

    gx = np.linspace(-10.0, 360.0, 300)
    gy = np.linspace(-10.0, 330.0, 300)
    GX, GY = np.meshgrid(gx, gy)
    field = np.where(inside(GX, GY), surface(GX, GY), np.nan)
    # the belt boundaries need opposite ink: 25 % runs over pale cells,
    # 75 % over the darkest ones
    ax.contour(GX, GY, field, levels=[25.0], colors=["#333333"],
               linewidths=0.9, linestyles=[(0, (3.6, 2.4))], zorder=6)
    ax.contour(GX, GY, field, levels=[75.0], colors=["#ffffff"],
               linewidths=1.0, linestyles=[(0, (3.6, 2.4))], zorder=6)
    ax.contour(GX, GY, field, levels=[50.0], colors=[INK], linewidths=2.0,
               zorder=7)

    xa = 74.0
    ya = 150.0 + 34.0 * np.sin(2.0 * np.pi * xa / 420.0) - 0.10 * (xa - CX)
    slope = (34.0 * np.cos(2.0 * np.pi * xa / 420.0) * 2.0 * np.pi / 420.0
             - 0.10)
    halo_text(ax, xa, ya + 7.0, "%s / %s isogloss  (50 %%)" % (NEW, OLD),
              ha="center", va="bottom", fontsize=7.2, color=INK,
              halo="white", halo_lw=2.6,
              rotation=np.degrees(np.arctan(slope)),
              rotation_mode="anchor", zorder=9)
    halo_text(ax, 236.0, 268.0, "%s area" % NEW, ha="center", va="center",
              fontsize=8.0, color="white", halo="#08263f", halo_lw=2.2,
              fontweight="bold", zorder=9)
    halo_text(ax, 116.0, 44.0, "%s area" % OLD, ha="center", va="center",
              fontsize=8.0, color=INK, halo="white", halo_lw=2.4,
              fontweight="bold", zorder=9)
    halo_text(ax, 300.0, 176.0, "transition\nbelt", ha="center", va="center",
              fontsize=7.0, color=INK, halo="white", halo_lw=2.4,
              linespacing=1.35, zorder=9)

    ax.plot([TRANSECT_X, TRANSECT_X], [12.0, 300.0], color=RED, lw=1.1,
            ls=(0, (4.0, 2.4)), zorder=8)
    halo_text(ax, TRANSECT_X + 6.0, 300.0, "transect", ha="left", va="top",
              fontsize=6.8, color=RED, halo="white", halo_lw=2.2, zorder=9)

    ax.set_xlim(-12.0, 360.0)
    ax.set_ylim(-12.0, 328.0)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (km)")
    ax.set_ylabel("northing (km)")
    add_scale_bar(ax, 100.0, "100 km", loc=(0.035, 0.045), height=0.016,
                  color="#333333", text_color=INK, halo="white", fontsize=6.5)
    ax.set_title("Survey hexagons shaded by use of the innovative variant",
                 fontsize=9.0, fontweight="bold", color=INK)
    safe_colorbar(fig, pc, ax, label="informants using %s (%%)" % NEW,
                  width=0.11, pad=0.10, labelsize=6.8,
                  ticks=[0, 25, 50, 75, 100])


def transect_panel(ax, x, y, obs):
    m = np.abs(x - TRANSECT_X) <= TRANSECT_HALF
    ty, tv = y[m], obs[m]
    p, _ = curve_fit(logistic, ty, tv, p0=[2.0, 98.0, 150.0, 25.0],
                     maxfev=20000)
    yy = np.linspace(ty.min(), ty.max(), 200)
    ax.scatter(ty, tv, s=11.0, color="#2e74a8", alpha=0.75, linewidths=0.0,
               zorder=3)
    ax.plot(yy, logistic(yy, *p), color=RED, lw=1.6, zorder=4)
    width = np.log(81.0) * abs(p[3])
    ax.axvline(p[2], color=GREY, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)
    ax.axhline(50.0, color=GREY, lw=0.8, ls=(0, (2.4, 2.0)), zorder=2)
    ax.axvspan(p[2] - width / 2.0, p[2] + width / 2.0, color=GREY, alpha=0.12,
               lw=0.0, zorder=1)
    ax.text(0.04, 0.94, "midpoint %.0f km\n10-90 %% width %.0f km"
            % (p[2], width), transform=ax.transAxes, ha="left", va="top",
            fontsize=6.8, color=RED, linespacing=1.5)
    ax.set_xlabel("northing along transect (km)", fontsize=7.4)
    ax.set_ylabel("use of %s (%%)" % NEW, fontsize=7.4)
    ax.set_ylim(-5.0, 108.0)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("South-to-north transect through the belt", fontsize=8.2,
                 fontweight="bold", color=INK)
    return p[2], width, int(m.sum())


def area_panel(ax, smooth, obs):
    names = ["%s area" % OLD, "transition", "%s area" % NEW]
    masks = [smooth < 25.0, (smooth >= 25.0) & (smooth <= 75.0),
             smooth > 75.0]
    mus = [obs[m].mean() for m in masks]
    sds = [obs[m].std() for m in masks]
    ns = [int(m.sum()) for m in masks]
    cols = get_palette("cb_sequential_blue", 5)[1:4]
    ax.barh(np.arange(3), mus, height=0.56, color=cols, alpha=0.95,
            edgecolor="white", linewidth=0.8, xerr=sds, capsize=2.0,
            error_kw=dict(elinewidth=0.7, ecolor=GREY))
    for i, (mu, sd, n) in enumerate(zip(mus, sds, ns)):
        ax.text(mu + sd + 3.5, i, "%.0f %%  (n=%d)" % (mu, n), ha="left",
                va="center", fontsize=6.6, color=INK)
    ax.axvline(50.0, color=GREY, lw=0.8, ls=(0, (2.4, 2.0)), zorder=1)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(names, fontsize=6.8)
    ax.set_xlim(0.0, 128.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("mean use of %s (%%)" % NEW, fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Usage by isogloss-defined area", fontsize=8.2,
                 fontweight="bold", color=INK)
    return ns


def main():
    set_pub_style()
    x, y, smooth, obs = cells()

    fig = plt.figure(figsize=(FW, FH))
    ax_map = fig.add_axes([MAIN[0] / FW, MAIN[1] / FH,
                           MAIN[2] / FW, MAIN[3] / FH])
    ax_tr = fig.add_axes([AUX_TOP[0] / FW, AUX_TOP[1] / FH,
                          AUX_TOP[2] / FW, AUX_TOP[3] / FH])
    ax_ar = fig.add_axes([AUX_BOT[0] / FW, AUX_BOT[1] / FH,
                          AUX_BOT[2] / FW, AUX_BOT[3] / FH])

    map_panel(fig, ax_map, x, y, smooth, obs)
    mid, width, n_tr = transect_panel(ax_tr, x, y, obs)
    area_panel(ax_ar, smooth, obs)

    panel_labels_inches(fig, [ax_map, ax_tr, ax_ar], dx=-0.52, dy=0.10)
    savefig(fig, "806_lng_dialect_map.png")


if __name__ == "__main__":
    main()
