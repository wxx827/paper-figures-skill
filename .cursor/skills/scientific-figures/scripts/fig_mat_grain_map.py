"""Figure 384 - Polycrystal grain map with log-normal grain-size statistics.

A Laguerre (power-diagram) tessellation with a smoothly warped metric gives a
grain-boundary network that reads like an etched micrograph rather than a
Voronoi diagram: cells vary in size, boundaries wander, triple junctions sit at
sensible angles.  Grains are tinted by their own equivalent diameter, so the
map and the histogram beside it show the same quantity twice.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from scipy.ndimage import binary_dilation, gaussian_filter

from pubstyle import (set_pub_style, savefig, get_cmap, annotate_box,
                      style_axes, COLORS, CATEGORICAL)
from figkit import add_scale_bar, image_axes, safe_colorbar, panel_labels_inches

UM_PER_PX = 0.22          # microscope calibration of the synthetic frame
SHAPE = (360, 460)        # rows, cols


# ---------------------------------------------------------------------------
# Synthetic polycrystal
# ---------------------------------------------------------------------------
def smooth_noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def laguerre_grains(shape=SHAPE, n_grains=190, seed=4, relax=2,
                    sigma_r=0.34, warp=4.2, warp_sigma=8.5):
    """Weighted Voronoi labels; returns (labels, seed points)."""
    ny, nx = shape
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.random(n_grains) * nx, rng.random(n_grains) * ny]
    radii = np.exp(sigma_r * rng.standard_normal(n_grains))
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    wx = (xx + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    wy = (yy + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    scale = (nx * ny / n_grains) ** 0.5
    w = (radii * 0.55 * scale) ** 2

    lab = np.zeros(wx.size, dtype=np.int32)
    for _ in range(relax + 1):
        best = np.full(wx.size, np.inf)
        for k in range(n_grains):
            d = (wx - pts[k, 0]) ** 2 + (wy - pts[k, 1]) ** 2 - w[k]
            m = d < best
            best[m] = d[m]
            lab[m] = k
        cnt = np.bincount(lab, minlength=n_grains).astype(float)
        cnt[cnt == 0] = 1.0
        pts = np.c_[np.bincount(lab, weights=wx, minlength=n_grains) / cnt,
                    np.bincount(lab, weights=wy, minlength=n_grains) / cnt]
    return lab.reshape(shape), pts


def boundary_mask(lab, width=1):
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    if width > 1:
        b = binary_dilation(b, np.ones((width, width), bool))
    return b


def grain_diameters(lab, um_per_px=UM_PER_PX):
    """Equivalent circle diameter per label, and a border-contact flag."""
    n = int(lab.max()) + 1
    area_px = np.bincount(lab.ravel(), minlength=n).astype(float)
    d = 2.0 * np.sqrt(np.maximum(area_px, 0) / np.pi) * um_per_px
    edge = np.zeros(n, bool)
    for ids in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        edge[np.unique(ids)] = True
    return d, edge


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    lab, _ = laguerre_grains()
    diam, edge = grain_diameters(lab)
    interior = ~edge & (np.bincount(lab.ravel(), minlength=diam.size) > 0)

    d_int = diam[interior]
    d_mean = float(d_int.mean())
    d_med = float(np.median(d_int))
    # ASTM E112 grain-size number from the mean equivalent diameter (mm).
    astm_g = -6.6438 * np.log10(d_mean * 1e-3) - 3.293
    shape_, _, scale_ = stats.lognorm.fit(d_int, floc=0.0)

    # Truncate the ramp: a near-black top end would swallow the black grain
    # boundaries drawn over it.
    base = get_cmap("cb_sequential_blue")
    cmap = ListedColormap(base(np.linspace(0.04, 0.80, 256)), "grain_blue")
    vmin, vmax = np.percentile(d_int, [2, 98])
    field = np.clip((diam[lab] - vmin) / (vmax - vmin), 0, 1)
    rgb = cmap(field)[..., :3]
    rgb[boundary_mask(lab, 1)] = np.array([0.10, 0.10, 0.10])
    n_seen = int(np.unique(lab).size)

    fig = plt.figure(figsize=(7.6, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.60], wspace=0.20,
                          left=0.045, right=0.975, top=0.885, bottom=0.175)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    # ---- (a) grain map ----------------------------------------------------
    im = ax_map.imshow(rgb, interpolation="nearest")
    im.set_cmap(cmap)
    im.set_clim(vmin, vmax)
    image_axes(ax_map, color=COLORS["rule"], lw=0.7)
    ax_map.set_title(
        f"Electropolished + etched section, {n_seen} grains resolved",
        fontsize=8.5, fontweight="bold", pad=3.5)
    add_scale_bar(ax_map, 20.0 / UM_PER_PX, label="20 \u00b5m",
                  loc=(0.030, 0.050), height=0.022, fontsize=6.5)
    safe_colorbar(fig, im, ax_map, side="bottom", width=0.10, pad=0.055,
                  shrink=0.86,
                  label="grain equivalent diameter (\u00b5m)")

    # ---- (b) size distribution -------------------------------------------
    bins = np.linspace(0, np.percentile(d_int, 99.5) * 1.12, 17)
    ax_hist.hist(d_int, bins=bins, density=True, color="#9cc3dd",
                 edgecolor=COLORS["rule"], linewidth=0.5, zorder=2)
    xg = np.linspace(bins[0] + 1e-3, bins[-1], 300)
    ax_hist.plot(xg, stats.lognorm.pdf(xg, shape_, 0.0, scale_),
                 color=CATEGORICAL[0], lw=1.6, zorder=4,
                 label=f"log-normal fit ($\\sigma$ = {shape_:.2f})")
    ax_hist.axvline(d_mean, color=COLORS["rule"], lw=0.9, ls="--", zorder=3,
                    label=f"mean = {d_mean:.2f} \u00b5m")
    ax_hist.set_xlabel("equivalent diameter (\u00b5m)", fontsize=8.0)
    ax_hist.set_ylabel("probability density (\u00b5m$^{-1}$)", fontsize=8.0)
    ax_hist.set_xlim(bins[0], bins[-1])
    ax_hist.tick_params(labelsize=7.0, length=2.5)
    ax_hist.grid(axis="y", linewidth=0.5, alpha=0.7)
    style_axes(ax_hist)
    ax_hist.legend(loc="upper right", frameon=False, fontsize=7.0,
                   handlelength=1.4, borderaxespad=0.2)
    ax_hist.set_title("Grain-size distribution", fontsize=8.5,
                      fontweight="bold", pad=3.5)
    annotate_box(ax_hist,
                 f"n = {d_int.size} interior grains\n"
                 f"median = {d_med:.2f} \u00b5m\n"
                 f"ASTM E112  G = {astm_g:.1f}",
                 loc="center right", fontsize=6.8)

    panel_labels_inches(fig, [ax_map, ax_hist], dx=-0.26, dy=0.09)

    savefig(fig, "384_mat_grain_map.png")


if __name__ == "__main__":
    main()
