"""Shaded-relief terrain figure built from a synthetic fractal DEM.

Multi-octave Gaussian noise plus a curved ridge produces the elevation model;
a 315 deg / 45 deg light source gives the hillshade, over which a translucent
hypsometric colour ramp and 200 m contours are drawn. Side panels add a slope
map and a hypsometric curve, and an inset shows the A-A' elevation transect.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, map_coordinates
from pubstyle import set_pub_style, savefig

TERRAIN = LinearSegmentedColormap.from_list(
    "terrain_soft", ["#6f8a6b", "#94a374", "#b7af83", "#cbb894", "#bda694",
                     "#ece8e1"])
NY, NX = 320, 400
DX = 0.1  # km per pixel -> 40 x 32 km domain


def build_dem(rng):
    z = np.zeros((NY, NX))
    for sigma, amp in [(60, 1.0), (30, 0.62), (15, 0.40), (8, 0.26),
                       (4, 0.17), (2, 0.10), (1, 0.05)]:
        o = gaussian_filter(rng.normal(0, 1, (NY, NX)), sigma)
        z += amp * (o - o.mean()) / o.std()
    yy, xx = np.mgrid[0:NY, 0:NX]
    # curved mountain range with a rough crest
    ridge_x = 0.40 * NX + 0.15 * NX * np.sin(2.2 * np.pi * yy / NY)
    crest = np.exp(-((xx - ridge_x) ** 2) / (2 * (0.085 * NX) ** 2))
    rough = gaussian_filter(rng.normal(0, 1, (NY, NX)), 4)
    z += 3.2 * crest * (1 + 0.45 * rough / rough.std())
    # secondary massif
    z += 1.7 * np.exp(-(((xx - 0.78 * NX) ** 2 + (yy - 0.26 * NY) ** 2)
                        / (2 * (0.10 * NX) ** 2)))
    # basin / plain in the east
    z -= 2.0 * np.clip((xx - 0.82 * NX) / (0.18 * NX), 0, 1) ** 1.4
    z = gaussian_filter(z, 0.8)
    z = (z - z.min()) / np.ptp(z)
    return 2450 * z ** 1.35


def main():
    set_pub_style()
    rng = np.random.default_rng(2024)
    z = build_dem(rng)
    extent = (0, NX * DX, 0, NY * DX)

    ls = LightSource(azdeg=315, altdeg=45)
    norm = Normalize(0, 2450)
    rgb = ls.shade(z, cmap=TERRAIN, blend_mode="soft", vert_exag=2.4,
                   dx=DX * 1000, dy=DX * 1000, norm=norm)
    gy, gx = np.gradient(z, DX * 1000)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    fig = plt.figure(figsize=(10.0, 6.8))
    ax = fig.add_axes([0.015, 0.055, 0.705, 0.865])
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="bilinear")
    im = ScalarMappable(norm=norm, cmap=TERRAIN)
    xs = np.linspace(extent[0], extent[1], NX)
    ys = np.linspace(extent[2], extent[3], NY)
    ax.contour(xs, ys, z, levels=np.arange(200, 2450, 200), colors="#3a3a3a",
               linewidths=0.2, alpha=0.28)
    cs = ax.contour(xs, ys, z, levels=np.arange(500, 2450, 500),
                    colors="#2b2b2b", linewidths=0.55, alpha=0.65)
    ax.clabel(cs, levels=[1000, 2000], fmt="%d m", fontsize=6.0,
              inline_spacing=3)

    # transect A - A'
    p0, p1 = (2.5, 5.0), (36.5, 27.0)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#b23a48", lw=1.2,
            ls=(0, (5, 2)), zorder=5)
    for (px, py), lab in [(p0, "A"), (p1, "A$'$")]:
        ax.plot(px, py, "o", ms=4, color="#b23a48", mec="white", mew=0.8,
                zorder=6)
        ax.text(px, py + 0.9, lab, ha="center", fontsize=8, fontweight="bold",
                color="#b23a48", zorder=6)

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#555555"); s.set_linewidth(0.8)
    ax.text(0.0, 1.008, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")
    ax.text(0.042, 1.011, "Shaded relief and hypsometry "
            "(illumination 315$\\degree$, 45$\\degree$)",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", color="#3f5a4a")
    ax.text(0.985, 1.011, "max %d m  |  mean slope %.1f$\\degree$"
            % (z.max(), slope.mean()), transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom", color="#666")

    # elevation colourbar inside the map
    ax.add_patch(Rectangle((0.018, 0.888), 0.315, 0.098, transform=ax.transAxes,
                           facecolor="white", alpha=0.85, edgecolor="none",
                           zorder=6))
    cax = ax.inset_axes([0.035, 0.935, 0.28, 0.022], zorder=9)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("elevation (m a.s.l.)", fontsize=6.6, labelpad=1.5)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor("#555")
    cax.set_facecolor("white")

    ax.annotate("N", xy=(0.965, 0.93), xytext=(0.965, 0.855),
                xycoords="axes fraction", ha="center", fontsize=7.5,
                color="#222", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.2))
    ax.plot([0.035, 0.285], [0.045, 0.045], transform=ax.transAxes,
            color="#222", lw=2.2, solid_capstyle="butt")
    ax.text(0.16, 0.062, "10 km", transform=ax.transAxes, fontsize=6.2,
            ha="center", color="#222", fontweight="bold")

    # ---- profile inset (bottom-right of the map)
    ax.add_patch(Rectangle((0.455, 0.028), 0.525, 0.315, transform=ax.transAxes,
                           facecolor="white", alpha=0.86, edgecolor="none",
                           zorder=6))
    axp = ax.inset_axes([0.505, 0.075, 0.44, 0.235], zorder=9)
    npt = 400
    rr = np.linspace(p0[1] / DX, p1[1] / DX, npt)
    cc = np.linspace(p0[0] / DX, p1[0] / DX, npt)
    prof = map_coordinates(z, [rr, cc], order=1)
    dist = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) * np.linspace(0, 1, npt)
    axp.fill_between(dist, 0, prof, color="#8d9c86", alpha=0.55, lw=0)
    axp.plot(dist, prof, color="#33463c", lw=1.1)
    axp.set_xlim(0, dist[-1]); axp.set_ylim(0, 2600)
    axp.set_xlabel("distance along A-A$'$ (km)", fontsize=6.4, labelpad=1)
    axp.set_ylabel("m", fontsize=6.4, labelpad=1)
    axp.tick_params(labelsize=6.0, length=2)
    axp.set_yticks([0, 1000, 2000])
    axp.set_facecolor("white")
    axp.patch.set_alpha(1.0)
    for sp in axp.spines.values():
        sp.set_color("#777777")
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    axp.text(0.02, 0.88, "A", transform=axp.transAxes, fontsize=7,
             fontweight="bold", color="#b23a48")
    axp.text(0.97, 0.88, "A$'$", transform=axp.transAxes, fontsize=7,
             ha="right", fontweight="bold", color="#b23a48")

    # ---- slope map
    axs = fig.add_axes([0.755, 0.545, 0.205, 0.375])
    slope_cmap = LinearSegmentedColormap.from_list(
        "slope_soft", ["#f6f3ec", "#dccdb0", "#b8926a", "#7d5a44", "#3b2a22"])
    ims = axs.imshow(slope, cmap=slope_cmap, origin="lower", extent=extent,
                     interpolation="bilinear", vmin=0,
                     vmax=np.quantile(slope, 0.98))
    axs.set_xticks([]); axs.set_yticks([])
    for s in axs.spines.values():
        s.set_color("#555555"); s.set_linewidth(0.8); s.set_visible(True)
    axs.text(0.0, 1.02, "(b)", transform=axs.transAxes, fontsize=10,
             fontweight="bold", va="bottom")
    axs.text(0.13, 1.025, "Slope", transform=axs.transAxes, fontsize=9,
             fontweight="bold", va="bottom", color="#3f5a4a")
    cax2 = fig.add_axes([0.972, 0.545, 0.012, 0.375])
    cb2 = fig.colorbar(ims, cax=cax2, extend="max")
    cb2.set_label("slope (degrees)", fontsize=6.6, labelpad=2)
    cb2.ax.tick_params(labelsize=6, length=2)
    cb2.outline.set_linewidth(0.5); cb2.outline.set_edgecolor("#555")

    # ---- hypsometric curve
    axh = fig.add_axes([0.795, 0.11, 0.175, 0.30])
    srt = np.sort(z.ravel())[::-1]
    frac = np.arange(1, srt.size + 1) / srt.size * 100
    axh.plot(frac, srt, color="#3f6b57", lw=1.6)
    axh.fill_between(frac, 0, srt, color="#3f6b57", alpha=0.14, lw=0)
    for q, c in [(50, "#b23a48")]:
        med = np.interp(q, frac, srt)
        axh.plot([0, q], [med, med], color=c, lw=0.8, ls=(0, (3, 2)))
        axh.plot([q, q], [0, med], color=c, lw=0.8, ls=(0, (3, 2)))
        axh.text(q + 3, med + 90, "median %d m" % med, fontsize=6.2,
                 color=c, fontweight="bold")
    axh.set_xlim(0, 100); axh.set_ylim(0, 2500)
    axh.set_xlabel("area above elevation (%)", fontsize=7, labelpad=1.5)
    axh.set_ylabel("elevation (m)", fontsize=7, labelpad=1.5)
    axh.tick_params(labelsize=6.2, length=2.5)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
    axh.text(0.0, 1.03, "(c)", transform=axh.transAxes, fontsize=10,
             fontweight="bold", va="bottom")
    axh.text(0.16, 1.035, "Hypsometry", transform=axh.transAxes, fontsize=9,
             fontweight="bold", va="bottom", color="#3f5a4a")

    savefig(fig, "60_terrain_hillshade.png")


if __name__ == "__main__":
    main()
