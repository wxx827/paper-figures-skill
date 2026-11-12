"""Figure 394 - Within-field yield map from a combine yield monitor.

A synthetic 1.5 x 1.5 km farm block: parcel mosaic, a centre-pivot circle, a
meandering creek and a farm track.  The main panel carries the interpolated
grain-yield raster with parcel boundaries and the combine swath trace on top;
the two subordinate panels give the yield histogram and the management-zone
summary derived from the very same raster.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, distance_transform_edt

from pubstyle import set_pub_style, savefig, add_panel_label
from figkit import add_scale_bar, safe_colorbar, halo_text

N = 300                 # raster side, px
MPP = 5.0               # ground sampling distance, m
SEED = 41

# Yield ramp: low = drought red, mid = gold, high = crop green.  Built from
# house earth / green hues so it sits next to the rest of the library, and
# deliberately not matplotlib's RdYlGn.
YIELD_CMAP = LinearSegmentedColormap.from_list(
    "yield_gyr", ["#8c2f2f", "#c0603f", "#d99a3f", "#c9a227", "#a8c07a",
                  "#6bbf8a", "#2e7d5b", "#134a34"])
NOCROP = "#ddd8cc"      # farm track, steading
WATER = "#7fa3bd"       # creek
ZONE_COLORS = ["#c0603f", "#c9a227", "#2e7d5b"]
ZONE_NAMES = ["Low", "Medium", "High"]


# ---------------------------------------------------------------- generators
def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(5, 13, 30), weights=(0.48, 0.32, 0.20)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _polyline_dist(n, pts):
    """Euclidean distance in pixels to a rasterised polyline."""
    m = np.zeros((n, n), bool)
    pts = np.asarray(pts, float)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        k = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 2
        ii = np.clip(np.round(np.linspace(y0, y1, k)).astype(int), 0, n - 1)
        jj = np.clip(np.round(np.linspace(x0, x1, k)).astype(int), 0, n - 1)
        m[ii, jj] = True
    return distance_transform_edt(~m)


def build_farm(n=N, seed=SEED):
    """Parcel mosaic + swath phase + non-crop mask for one farm block."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    # Four management blocks, each ploughed on its own bearing.
    anchors = np.array([[0.22, 0.24], [0.76, 0.19],
                        [0.26, 0.76], [0.79, 0.71]]) * n
    dist = np.stack([np.hypot(xx - a[0], yy - a[1]) + 15 * _noise(rng, n, 26)
                     for a in anchors])
    reg = np.argmin(dist, axis=0)

    fid = np.zeros((n, n), int)
    swath = np.zeros((n, n))
    for r, (ang, cu, cv) in enumerate([(13.0, 64, 48), (-28.0, 52, 58),
                                       (37.0, 47, 55), (4.0, 70, 44)]):
        th = np.deg2rad(ang)
        u = xx * np.cos(th) + yy * np.sin(th) + 3.0 * _noise(rng, n, 15)
        v = -xx * np.sin(th) + yy * np.cos(th) + 3.0 * _noise(rng, n, 15)
        f = (np.floor(u / cu).astype(int) * 211
             + np.floor(v / cv).astype(int) * 37 + r * 1013)
        fid = np.where(reg == r, f, fid)
        swath = np.where(reg == r, v, swath)

    # Centre-pivot circle: its own field, its own circular swath pattern.
    pcx, pcy, prr = 0.585 * n, 0.415 * n, 0.135 * n
    rad = np.hypot(xx - pcx, yy - pcy)
    pivot = rad < prr * (1 + 0.012 * _noise(rng, n, 3))
    fid = np.where(pivot, 90210, fid)
    swath = np.where(pivot, rad, swath)

    # Creek, farm track and steading are not cropped.
    ty = np.linspace(0, n - 1, 220)
    tx = (0.155 * n + 0.075 * n * np.sin(2 * np.pi * ty / n * 0.95)
          + 0.030 * n * np.sin(2 * np.pi * ty / n * 2.4 + 0.9))
    d_creek = _polyline_dist(n, np.c_[tx, ty])
    creek = d_creek <= (1.5 + 0.6 * _noise(rng, n, 22))

    track = np.zeros((n, n), bool)
    for pl in [[(0.0, 0.86 * n), (0.42 * n, 0.66 * n), (n - 1, 0.58 * n)],
               [(0.42 * n, 0.66 * n), (0.48 * n, n - 1)]]:
        track |= _polyline_dist(n, pl) <= 1.6
    stead = (np.hypot(xx - 0.435 * n, yy - 0.655 * n)
             < 0.030 * n * (1 + 0.25 * _noise(rng, n, 3)))

    nocrop = creek | track | stead
    return dict(fid=fid, swath=swath, nocrop=nocrop, d_creek=d_creek,
                creek=creek, pivot=pivot, xx=xx, yy=yy, rng=rng)


def yield_raster(farm, n=N):
    """Grain yield in t/ha, masked outside the cropped area."""
    rng = farm["rng"]
    xx, yy = farm["xx"], farm["yy"]

    y = 8.9 + 1.55 * _fbm(rng, n, (7, 18, 38))

    # Waterlogged strip along the creek, and a droughty gravel knoll.
    y -= 4.2 * np.exp(-(farm["d_creek"] / 13.0) ** 2)
    y -= 2.6 * np.exp(-(((xx - 0.83 * n) ** 2 + (yy - 0.31 * n) ** 2)
                        / (2 * (0.085 * n) ** 2)))
    # Irrigated pivot runs above the rain-fed average.
    y += 1.7 * farm["pivot"]

    # Headland compaction: the outer two turns of every parcel yield less.
    bnd = np.zeros((n, n), bool)
    fid = farm["fid"]
    bnd[:, :-1] |= fid[:, :-1] != fid[:, 1:]
    bnd[:-1, :] |= fid[:-1, :] != fid[1:, :]
    d_edge = distance_transform_edt(~bnd)
    y -= 2.1 * np.exp(-d_edge / 3.2)

    y += 0.22 * _noise(rng, n, 1.0)
    y = np.clip(y, 1.6, 13.4)
    return np.ma.masked_where(farm["nocrop"], y), bnd, d_edge


# --------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    farm = build_farm()
    ygrid, bnd, d_edge = yield_raster(farm)
    valid = ~ygrid.mask
    yv = np.asarray(ygrid[valid]).ravel()

    px_ha = (MPP ** 2) / 1e4                       # hectares per pixel
    area_ha = valid.sum() * px_ha
    mean_y = yv.mean()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.62],
                          left=0.045, right=0.885, top=0.905, bottom=0.085,
                          wspace=0.30, hspace=0.42)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_zone = fig.add_subplot(gs[1, 1])

    # ---- (a) yield map -----------------------------------------------------
    YIELD_CMAP.set_bad(NOCROP)
    im = ax_map.imshow(ygrid, cmap=YIELD_CMAP, vmin=2.0, vmax=13.0,
                       interpolation="bilinear")
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for s in ax_map.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#333333")

    # Combine swath trace: the header pass lines, 5 px = 25 m apart.
    swath = farm["swath"]
    trk = (np.mod(swath, 5.0) < 0.95) & valid
    ax_map.imshow(np.ma.masked_where(~trk, np.ones_like(swath)),
                  cmap=LinearSegmentedColormap.from_list(
                      "trk", ["#2b2b2b", "#2b2b2b"]),
                  vmin=0, vmax=1, alpha=0.17, interpolation="nearest")

    # Parcel boundaries, then the creek on top in its own water tone.
    ax_map.imshow(np.ma.masked_where(~bnd, np.ones_like(swath)),
                  cmap=LinearSegmentedColormap.from_list(
                      "bnd", ["#fbf8f0", "#fbf8f0"]),
                  vmin=0, vmax=1, alpha=0.92, interpolation="nearest")
    ax_map.imshow(np.ma.masked_where(~farm["creek"], np.ones_like(swath)),
                  cmap=LinearSegmentedColormap.from_list(
                      "creek", [WATER, WATER]),
                  vmin=0, vmax=1, interpolation="nearest")

    add_scale_bar(ax_map, 500.0 / MPP, "500 m", loc=(0.035, 0.048),
                  height=0.016, fontsize=6.4)
    halo_text(ax_map, 0.972, 0.965, "N \u25b2", transform=ax_map.transAxes,
              ha="right", va="top", fontsize=7.0, fontweight="bold")
    halo_text(ax_map, 0.028, 0.972,
              f"combine yield monitor  |  {MPP:.0f} m grid  |  "
              f"{area_ha:.0f} ha harvested",
              transform=ax_map.transAxes, ha="left", va="top", fontsize=6.4)
    ax_map.set_title("Grain yield, parcel boundaries and combine swaths",
                     fontsize=9.0, fontweight="bold")

    safe_colorbar(fig, im, ax_map, label="grain yield (t/ha)",
                  width=0.13, pad=0.09,
                  ticks=[2, 4, 6, 8, 10, 12])

    leg = [Line2D([0], [0], color="#e2dccd", lw=1.8, label="parcel boundary"),
           Line2D([0], [0], color="#2b2b2b", lw=1.0, alpha=0.45,
                  label="combine swath (25 m)"),
           Line2D([0], [0], color=WATER, lw=2.6, label="creek"),
           Line2D([0], [0], color=NOCROP, lw=2.6, label="track / steading")]
    ax_map.legend(handles=leg, loc="lower right", fontsize=6.6,
                  frameon=True, framealpha=0.9, facecolor="white",
                  edgecolor="#cccccc", borderpad=0.4,
                  bbox_to_anchor=(1.005, -0.008))

    # ---- (b) yield histogram ----------------------------------------------
    edges = np.linspace(2.0, 13.0, 45)
    cent = 0.5 * (edges[:-1] + edges[1:])
    cnt, _ = np.histogram(yv, bins=edges)
    ax_hist.bar(cent, cnt * px_ha, width=np.diff(edges) * 0.92,
                color="#6bbf8a", edgecolor="#2e7d5b", linewidth=0.4)
    ax_hist.set_ylim(0, (cnt * px_ha).max() * 1.32)
    ax_hist.axvline(mean_y, color="#b23a48", lw=1.2, ls="--", zorder=4)
    ax_hist.text(mean_y - 0.18, 0.965, f"field mean\n{mean_y:.2f} t/ha",
                 transform=ax_hist.get_xaxis_transform(), fontsize=6.6,
                 color="#b23a48", va="top", ha="right", linespacing=1.3)
    ax_hist.set_xlabel("grain yield (t/ha)", fontsize=8.0)
    ax_hist.set_ylabel("area (ha)", fontsize=8.0)
    ax_hist.set_xlim(2.0, 13.0)
    ax_hist.tick_params(labelsize=6.8)
    ax_hist.set_title("Yield distribution", fontsize=8.5, fontweight="bold")
    ax_hist.text(0.985, 0.965,
                 f"n = {yv.size:,} cells\nCV = {yv.std() / mean_y:.1%}\n"
                 f"p10-p90 = {np.percentile(yv, 10):.1f}-"
                 f"{np.percentile(yv, 90):.1f} t/ha",
                 transform=ax_hist.transAxes, ha="right", va="top",
                 fontsize=6.6, color="#555555", linespacing=1.4)

    # ---- (c) management zones ---------------------------------------------
    q = np.percentile(yv, [33.3, 66.7])
    zid = np.digitize(yv, q)
    means = [yv[zid == k].mean() for k in range(3)]
    sds = [yv[zid == k].std() for k in range(3)]
    shares = [100.0 * (zid == k).mean() for k in range(3)]

    xs = np.arange(3)
    ax_zone.bar(xs, means, width=0.62, color=ZONE_COLORS,
                edgecolor="white", linewidth=0.7,
                yerr=sds, error_kw=dict(ecolor="#333333", lw=0.8, capsize=2.5))
    for x, m, s, sh in zip(xs, means, sds, shares):
        ax_zone.text(x, m + s + 0.28, f"{m:.1f}\n{sh:.0f}% area", ha="center",
                     va="bottom", fontsize=6.6, color="#333333",
                     linespacing=1.25)
    ax_zone.axhline(mean_y, color="#808080", lw=0.8, ls="--", zorder=1)
    ax_zone.set_xlim(-0.82, 2.52)
    ax_zone.text(-0.78, mean_y + 0.12, "field mean", ha="left", va="bottom",
                 fontsize=6.2, color="#777777")
    ax_zone.set_xticks(xs)
    ax_zone.set_xticklabels([f"{z}\n(< {q[0]:.1f})" if i == 0 else
                             (f"{z}\n({q[0]:.1f}-{q[1]:.1f})" if i == 1 else
                              f"{z}\n(> {q[1]:.1f})")
                             for i, z in enumerate(ZONE_NAMES)], fontsize=6.6)
    ax_zone.set_ylabel("mean yield (t/ha)", fontsize=8.0)
    ax_zone.set_ylim(0, max(np.add(means, sds)) * 1.30)
    ax_zone.tick_params(axis="y", labelsize=6.8)
    ax_zone.set_title("Tercile management zones", fontsize=8.5,
                      fontweight="bold")

    for ax in (ax_hist, ax_zone):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    add_panel_label(ax_map, "(a)", x=-0.055, y=1.02)
    add_panel_label(ax_hist, "(b)", x=-0.155, y=1.02)
    add_panel_label(ax_zone, "(c)", x=-0.155, y=1.02)


    savefig(fig, "394_agr_yield_map.png")


if __name__ == "__main__":
    main()
