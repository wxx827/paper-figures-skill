"""Land-surface temperature retrieval and its surface-property controls.

Panel (a) retrieved LST over a synthetic city-region scene (100 m pixels)
with a scale bar and north arrow.  Panels (b)-(c) density scatter of LST
against NDVI (the classic negative-sloping triangle with dry / wet edges)
and against impervious-surface fraction.  Panel (d) LST distribution by
land-cover class.

The scene is generated from a physically motivated surface-energy proxy
    LST = Ta + dT * (1 - Fv) * dryness + urban excess - lapse * elevation
so that the LST-NDVI relation is negative by construction.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

NY, NX = 260, 200          # 100 m pixels -> 26 km x 20 km
EXT = [0.0, 20.0, 0.0, 26.0]


def norm01(a):
    a = np.asarray(a, dtype=float)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def fractal_noise(rng, shape, sigmas=(24, 10, 4, 1.5), weights=(1.0, 0.55, 0.28, 0.12)):
    """Multi-scale smoothed noise -> spatially auto-correlated field."""
    out = np.zeros(shape)
    for s, w in zip(sigmas, weights):
        out += w * gaussian_filter(rng.normal(size=shape), s, mode="reflect")
    return norm01(out)


def build_scene(rng):
    yy, xx = np.mgrid[0:NY, 0:NX]
    xk = xx * 0.1                      # km
    yk = yy * 0.1

    # ---- terrain: forested hills along the northern rim -----------------
    elev = (0.40 * fractal_noise(rng, (NY, NX), (26, 11, 4), (1.0, 0.5, 0.2))
            + 0.60 * norm01(yk + 0.22 * xk))
    elev = norm01(gaussian_filter(elev, 2))

    # ---- hydrography: meandering river + reservoir ----------------------
    river_x = 5.6 + 2.2 * np.sin(yk / 3.4) + 0.9 * np.sin(yk / 1.25 + 0.8)
    width = 0.20 + 0.08 * np.sin(yk / 5.0)
    water = np.abs(xk - river_x) < width
    lake = ((xk - 15.0) ** 2 / 2.3 ** 2 + (yk - 6.6) ** 2 / 1.2 ** 2) < 1.0
    lake &= fractal_noise(rng, (NY, NX), (6, 2), (1.0, 0.4)) > 0.26
    water |= lake

    # ---- urban core + satellite town ------------------------------------
    core = np.exp(-(((xk - 9.6) / 3.0) ** 2 + ((yk - 13.0) / 2.9) ** 2))
    town = np.exp(-(((xk - 15.4) / 1.4) ** 2 + ((yk - 19.8) / 1.1) ** 2))
    urban_raw = (core + 0.85 * town
                 + 0.42 * (fractal_noise(rng, (NY, NX), (7, 2.5), (1.0, 0.5)) - 0.45))
    isa = np.clip((urban_raw - 0.30) * 1.9, 0.0, 1.0)
    isa = gaussian_filter(isa, 1.0)
    isa[water] = 0.0

    # ---- forest on the higher ground (fixed area fraction) --------------
    thr = np.quantile(elev, 0.70)
    forest = (elev > thr) & (isa < 0.12) & (~water)

    # ---- agricultural parcels (rectangular field blocks) ----------------
    bh, bw = 8, 10
    block = (yy // bh) * 512 + (xx // bw)
    nblk = block.max() + 1
    blk_ndvi = rng.uniform(0.34, 0.86, nblk)
    blk_fallow = rng.random(nblk) < 0.22
    parcel_ndvi = blk_ndvi[block]
    fallow = blk_fallow[block]

    # ---- land-cover label ------------------------------------------------
    # 0 water, 1 forest, 2 cropland, 3 grass/bare, 4 suburban, 5 urban core
    lc = np.full((NY, NX), 2, dtype=int)
    lc[fallow] = 3
    lc[forest] = 1
    lc[isa > 0.16] = 4
    lc[isa > 0.55] = 5
    lc[water] = 0

    # ---- NDVI ------------------------------------------------------------
    ndvi = np.where(lc == 2, parcel_ndvi, 0.30)
    ndvi = np.where(lc == 1, 0.83, ndvi)
    ndvi = np.where(lc == 3, 0.22, ndvi)
    ndvi = np.where(lc == 0, 0.02, ndvi)
    ndvi = ndvi * (1.0 - 0.82 * isa) + 0.06 * isa
    ndvi += 0.045 * (fractal_noise(rng, (NY, NX), (5, 1.5), (1.0, 0.6)) - 0.5)
    ndvi = gaussian_filter(ndvi, 0.7)
    ndvi[water] = 0.02 + 0.02 * rng.normal(size=water.sum()).clip(-1, 1)
    ndvi = np.clip(ndvi, -0.05, 0.92)

    # ---- LST from a surface-energy proxy --------------------------------
    # Fv -> 1 collapses the heating term, which produces the classic
    # triangular LST-NDVI feature space (wide dry range at low NDVI).
    fv = np.clip((ndvi - 0.05) / (0.86 - 0.05), 0, 1) ** 2
    dryness = norm01(gaussian_filter(fractal_noise(rng, (NY, NX), (16, 6), (1.0, 0.5)), 2))
    # riparian corridor stays moist regardless of cover -> populates the flat
    # wet edge of the feature space at low NDVI as well
    dist_km = distance_transform_edt(~water) * 0.1
    riparian = np.exp(-dist_km / 1.8)
    dryness = np.clip(0.05 + 1.10 * dryness + 0.10 * isa
                      - 0.35 * forest - 0.60 * riparian, 0.02, 1.0)

    lst = (25.4
           + 15.8 * (1.0 - fv) * dryness
           + 1.6 * isa
           - 1.3 * elev
           + rng.normal(0, 0.30, (NY, NX)))
    lst = gaussian_filter(lst, 0.6)
    lst[water] = 21.4 + rng.normal(0, 0.35, water.sum())
    return dict(lst=lst, ndvi=ndvi, isa=isa, lc=lc, water=water, elev=elev)


def add_scalebar(ax, x0, y0, length, label):
    fx = [pe.withStroke(linewidth=2.2, foreground="black")]
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=2.4,
            solid_capstyle="butt", zorder=7, path_effects=fx)
    for xt in (x0, x0 + length):
        ax.plot([xt, xt], [y0 - 0.32, y0 + 0.32], color="white", lw=1.3,
                zorder=7, path_effects=fx)
    ax.text(x0 + length / 2, y0 + 0.45, label, color="white", ha="center",
            va="bottom", fontsize=6.6, fontweight="bold", zorder=7,
            path_effects=fx)


def add_north(ax, x, y):
    fx = [pe.withStroke(linewidth=2.0, foreground="black")]
    ax.annotate("", xy=(x, y + 1.5), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.3,
                                mutation_scale=8), zorder=7)
    ax.text(x, y + 1.7, "N", color="white", ha="center", va="bottom",
            fontsize=7, fontweight="bold", zorder=7, path_effects=fx)


def frame_raster(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_edgecolor("#333333")


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(20240184)
    sc = build_scene(rng)
    lst, ndvi, isa, lc, water = (sc["lst"], sc["ndvi"], sc["isa"],
                                 sc["lc"], sc["water"])

    fig = plt.figure(figsize=(7.4, 5.0))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.78, 1.0, 1.0],
                  height_ratios=[1.0, 0.80], hspace=0.55, wspace=0.40)

    # ---- (a) LST map ------------------------------------------------------
    axm = fig.add_subplot(gs[:, 0])
    vmin, vmax = np.percentile(lst, [0.5, 99.5])
    im = axm.imshow(lst, origin="lower", extent=EXT, cmap="inferno",
                    vmin=vmin, vmax=vmax, interpolation="bilinear")
    frame_raster(axm)
    add_scalebar(axm, 1.2, 1.4, 5.0, "5 km")
    add_north(axm, 18.4, 1.4)
    axm.set_title("Retrieved LST", fontsize=8.5)
    add_panel_label(axm, "(a)", x=-0.03, y=1.005, size=10)
    # axm is set_aspect("equal") via frame_raster, so fig.colorbar(ax=axm)
    # would size the bar from the pre-shrink box and strand it off the map.
    cb = fig.colorbar(im, cax=axm.inset_axes([1.030, 0.0, 0.032, 1.0]))
    cb.set_label("LST (\u00b0C)", fontsize=7.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.5)
    axm.text(0.025, 0.982,
             "Landsat-like TIRS, 10:30 LT\n"
             f"UHI intensity $\\bf{{+{lst[lc == 5].mean() - lst[lc == 1].mean():.1f}}}$"
             " \u00b0C (urban \u2212 forest)",
             transform=axm.transAxes, ha="left", va="top", fontsize=6.3,
             color="white", linespacing=1.25,
             path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

    # ---- shared scatter styling ------------------------------------------
    dens_cmap = LinearSegmentedColormap.from_list(
        "dens", ["#ffffff"] + PALETTES["cb_sequential_blue"][1:])
    land = ~water
    nd_l, lst_l, isa_l = ndvi[land], lst[land], isa[land]

    # ---- (b) LST vs NDVI triangle ---------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    hb = axb.hexbin(nd_l, lst_l, gridsize=44, cmap=dens_cmap, mincnt=1,
                    linewidths=0.0, bins="log")
    k, b0 = np.polyfit(nd_l, lst_l, 1)
    r = np.corrcoef(nd_l, lst_l)[0, 1]
    xg = np.linspace(nd_l.min(), nd_l.max(), 50)
    axb.plot(xg, k * xg + b0, color="#b23a48", lw=1.5, zorder=5)

    # dry / wet edges from binned extremes
    bins = np.linspace(0.05, 0.85, 17)
    ctr, hi, lo = [], [], []
    for i in range(bins.size - 1):
        m = (nd_l >= bins[i]) & (nd_l < bins[i + 1])
        if m.sum() > 60:
            ctr.append(0.5 * (bins[i] + bins[i + 1]))
            hi.append(np.percentile(lst_l[m], 99))
            lo.append(np.percentile(lst_l[m], 1))
    ctr, hi, lo = map(np.asarray, (ctr, hi, lo))
    kd, bd = np.polyfit(ctr, hi, 1)
    kw, bw_ = np.polyfit(ctr, lo, 1)
    axb.plot(xg, kd * xg + bd, color="#8c2f2f", lw=1.0, ls="--", zorder=5)
    axb.plot(xg, kw * xg + bw_, color="#2166ac", lw=1.0, ls="--", zorder=5)
    axb.text(0.06, kd * 0.06 + bd + 0.4, "dry edge", fontsize=6.2,
             color="#8c2f2f", va="bottom", fontweight="bold")
    axb.text(0.62, kw * 0.62 + bw_ - 0.6, "wet edge", fontsize=6.2,
             color="#2166ac", va="top", fontweight="bold")
    axb.text(0.97, 0.96,
             f"$\\bf{{slope}}$ = {k:.1f} \u00b0C / NDVI\n"
             f"$r$ = {r:.2f},  $R^2$ = {r ** 2:.2f}",
             transform=axb.transAxes, ha="right", va="top", fontsize=6.4,
             linespacing=1.25,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                       lw=0.5, alpha=0.94))
    axb.set_xlabel("NDVI (dimensionless)")
    axb.set_ylabel("LST (\u00b0C)")
    axb.set_title("LST\u2013NDVI space", fontsize=8.5, loc="right")
    style_axes(axb)
    add_panel_label(axb, "(b)", x=-0.30, y=1.02, size=10)

    # ---- (c) LST vs impervious fraction ----------------------------------
    axc = fig.add_subplot(gs[0, 2])
    m = isa_l > 0.005
    axc.hexbin(isa_l[m], lst_l[m], gridsize=40, cmap=dens_cmap, mincnt=1,
               linewidths=0.0, bins="log")
    k2, b2 = np.polyfit(isa_l[m], lst_l[m], 1)
    r2 = np.corrcoef(isa_l[m], lst_l[m])[0, 1]
    xg2 = np.linspace(0, 1, 40)
    axc.plot(xg2, k2 * xg2 + b2, color="#b23a48", lw=1.5, zorder=5)
    axc.text(0.03, 0.96,
             f"$\\bf{{slope}}$ = $+${k2:.1f} \u00b0C\n$R^2$ = {r2 ** 2:.2f}",
             transform=axc.transAxes, ha="left", va="top", fontsize=6.4,
             linespacing=1.25,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc",
                       lw=0.5, alpha=0.94))
    axc.set_xlabel("Impervious fraction (\u2013)")
    axc.set_ylabel("LST (\u00b0C)")
    axc.set_title("LST\u2013imperviousness", fontsize=8.5, loc="right")
    style_axes(axc)
    add_panel_label(axc, "(c)", x=-0.30, y=1.02, size=10)

    cbx = fig.add_axes([0.972, 0.60, 0.011, 0.26])
    cb2 = fig.colorbar(hb, cax=cbx)
    cb2.set_label("pixel count (log)", fontsize=6.4)
    cb2.outline.set_linewidth(0.6)
    cb2.ax.tick_params(labelsize=6.0, width=0.6, length=2.0)

    # ---- (d) LST by land-cover class -------------------------------------
    axd = fig.add_subplot(gs[1, 1:])
    axd.set_facecolor("#f7f7f4")
    names = ["Water", "Forest", "Cropland", "Grass /\nbare", "Suburban",
             "Urban\ncore"]
    cols = ["#2166ac", "#0b3d2e", "#6bbf8a", "#c9a227", "#d98c5f", "#8c2f2f"]
    data = [lst[lc == i].ravel() for i in range(6)]
    bp = axd.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                     medianprops=dict(color="white", lw=1.1),
                     whiskerprops=dict(lw=0.7, color="#333333"),
                     capprops=dict(lw=0.7, color="#333333"),
                     boxprops=dict(lw=0.6, edgecolor="#333333"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.88)
    for i, d in enumerate(data):
        axd.text(i + 1, np.percentile(d, 75) + 0.5, f"{np.median(d):.1f}",
                 ha="center", va="bottom", fontsize=6.1, fontweight="bold",
                 color="#1a1a1a")
    axd.set_xticks(range(1, 7))
    axd.set_xticklabels(names, fontsize=6.6)
    axd.set_ylabel("LST (\u00b0C)")
    axd.set_title("LST by land-cover class", fontsize=8.5)
    style_axes(axd)
    axd.grid(axis="y", lw=0.5, alpha=0.6)
    add_panel_label(axd, "(d)", x=-0.115, y=1.03, size=10)

    # (b), (c) and (d) all read "LST (deg C)", so they share one scale; letting
    # each autoscale made the land-only hexbins in (b)/(c) look as cold as the
    # water class in (d).
    lst_lim = (19.0, float(np.percentile(lst, 99.9)) + 3.0)
    for a in (axb, axc, axd):
        a.set_ylim(*lst_lim)

    fig.subplots_adjust(top=0.905, bottom=0.095, left=0.030, right=0.945)
    savefig(fig, "184_rs_lst_retrieval.png")


if __name__ == "__main__":
    main()
