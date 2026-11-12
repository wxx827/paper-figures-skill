"""Figure 387 - Porosity analysis of a sintered section, four linked views.

Raw optical micrograph, the binary pore mask thresholded out of it, the same
pores recoloured by equivalent diameter, and the size distribution with its
cumulative pore-area curve.  Porosity, counts and percentiles are all measured
on the mask that panel (b) shows, so the four panels cannot drift apart.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation, gaussian_filter, label as cc_label

from pubstyle import (set_pub_style, savefig, get_cmap, annotate_box,
                      style_axes, COLORS, CATEGORICAL)
from figkit import add_scale_bar, image_axes, safe_colorbar, panel_labels_inches

UM_PER_PX = 0.30
SHAPE = (330, 440)


def smooth_noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def laguerre_grains(shape, n_grains, seed, relax=2, sigma_r=0.30,
                    warp=4.0, warp_sigma=8.5):
    ny, nx = shape
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.random(n_grains) * nx, rng.random(n_grains) * ny]
    radii = np.exp(sigma_r * rng.standard_normal(n_grains))
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    wx = (xx + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    wy = (yy + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    w = (radii * 0.55 * (nx * ny / n_grains) ** 0.5) ** 2
    lab = np.zeros(wx.size, dtype=np.int32)
    for _ in range(relax + 1):
        best = np.full(wx.size, np.inf)
        for k in range(n_grains):
            d = (wx - pts[k, 0]) ** 2 + (wy - pts[k, 1]) ** 2 - w[k]
            m = d < best
            best[m] = d[m]
            lab[m] = k
        cnt = np.maximum(np.bincount(lab, minlength=n_grains), 1).astype(float)
        pts = np.c_[np.bincount(lab, weights=wx, minlength=n_grains) / cnt,
                    np.bincount(lab, weights=wy, minlength=n_grains) / cnt]
    return lab.reshape(shape)


def boundary_mask(lab, width=1):
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return binary_dilation(b, np.ones((width, width), bool)) if width > 1 else b


def build_scene(seed=9, n_pores=190):
    rng = np.random.default_rng(seed)
    ny, nx = SHAPE
    lab = laguerre_grains(SHAPE, 210, seed=seed + 1)
    bnd = boundary_mask(lab, 1)

    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    wx = xx + 2.6 * smooth_noise(rng, SHAPE, 4.5)
    wy = yy + 2.6 * smooth_noise(rng, SHAPE, 4.5)

    cx = rng.random(n_pores) * nx
    cy = rng.random(n_pores) * ny
    rr = np.clip(np.exp(0.62 * rng.standard_normal(n_pores)) * 3.4, 1.6, 16.0)
    # A handful of coarse, lobed pores: clusters of overlapping discs.
    for _ in range(9):
        k = rng.integers(n_pores)
        for _ in range(rng.integers(2, 5)):
            cx = np.append(cx, cx[k] + rng.normal(0, 4.5))
            cy = np.append(cy, cy[k] + rng.normal(0, 4.5))
            rr = np.append(rr, rr[k] * rng.uniform(0.55, 1.0))

    pore = np.zeros(SHAPE, bool)
    for k in range(cx.size):
        pore |= (wx - cx[k]) ** 2 + (wy - cy[k]) ** 2 < rr[k] ** 2

    grey = 0.64 + 0.055 * (rng.random(211) - 0.5)[lab]
    grey -= 0.135 * bnd
    tex = (0.55 * smooth_noise(rng, SHAPE, 0.9)
           + 0.45 * smooth_noise(rng, SHAPE, 3.2))
    grey *= 1.0 + 0.040 * tex
    rim = binary_dilation(pore, np.ones((3, 3), bool)) & ~pore
    grey[rim] += 0.14                                   # polishing edge flare
    grey[pore] = 0.14 + 0.05 * tex[pore]
    inner = gaussian_filter(pore.astype(float), 2.0)
    grey = np.where(pore, grey * (0.75 + 0.45 * (1 - inner)), grey)
    yg, xg = np.mgrid[0:ny, 0:nx] / (ny - 1.0) - 0.5
    grey *= 1.0 - 0.10 * (xg ** 2 + yg ** 2) / 0.5
    grey += 0.010 * rng.standard_normal(SHAPE)
    return np.clip(grey, 0.0, 1.0), pore


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    grey, pore = build_scene()
    cc, _ = cc_label(pore)
    counts = np.bincount(cc.ravel())
    keep = counts >= 4                      # reject specks below 4 px
    keep[0] = False
    pore = keep[cc]
    cc, n_pore = cc_label(pore)
    area_px = np.bincount(cc.ravel())[1:]
    area_um2 = area_px * UM_PER_PX ** 2
    diam = 2.0 * np.sqrt(area_um2 / np.pi)
    porosity = 100.0 * pore.mean()

    order = np.argsort(diam)
    cum = 100.0 * np.cumsum(area_um2[order]) / area_um2.sum()
    d50 = float(np.interp(50.0, cum, diam[order]))

    dmap = np.full(SHAPE, np.nan)
    dmap[pore] = np.concatenate([[np.nan], diam])[cc[pore]]
    vmin, vmax = float(diam.min()), float(np.percentile(diam, 97))
    cmap_d = ListedColormap(get_cmap("cb_sequential_warm")(
        np.linspace(0.26, 0.96, 256)), "pore_warm")
    cmap_d.set_bad(COLORS["card_alt"])

    fig = plt.figure(figsize=(7.8, 6.1))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.24,
                          left=0.045, right=0.965, top=0.905, bottom=0.075)
    ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)]
                   for r in range(2)])
    ax_raw, ax_bin, ax_size, ax_dist = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    # ---- (a) raw ----------------------------------------------------------
    ax_raw.imshow(grey, cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest")
    image_axes(ax_raw, color=COLORS["rule"], lw=0.7)
    ax_raw.set_title("Polished section, bright field", fontsize=8.5,
                     fontweight="bold", pad=3.0)
    add_scale_bar(ax_raw, 30.0 / UM_PER_PX, label="30 \u00b5m",
                  loc=(0.035, 0.050), height=0.024, fontsize=6.5)

    # ---- (b) binary -------------------------------------------------------
    ax_bin.imshow(np.where(pore, 1.0, 0.0), cmap=ListedColormap(
        [COLORS["card"], COLORS["solid"]]), vmin=0, vmax=1,
        interpolation="nearest")
    image_axes(ax_bin, color=COLORS["rule"], lw=0.7)
    ax_bin.set_title("Otsu threshold, pores in black", fontsize=8.5,
                     fontweight="bold", pad=3.0)
    add_scale_bar(ax_bin, 30.0 / UM_PER_PX, label="30 \u00b5m",
                  loc=(0.035, 0.050), height=0.024, fontsize=6.5,
                  color="#111111", alt="white")
    annotate_box(ax_bin, f"porosity = {porosity:.2f} %\n{n_pore} pores",
                 loc="upper right", fontsize=7.0)

    # ---- (c) diameter-coded ----------------------------------------------
    im = ax_size.imshow(np.ma.masked_invalid(dmap), cmap=cmap_d, vmin=vmin,
                        vmax=vmax, interpolation="nearest")
    image_axes(ax_size, color=COLORS["rule"], lw=0.7)
    ax_size.set_title("Pores coloured by equivalent diameter", fontsize=8.5,
                      fontweight="bold", pad=3.0)
    add_scale_bar(ax_size, 30.0 / UM_PER_PX, label="30 \u00b5m",
                  loc=(0.035, 0.050), height=0.024, fontsize=6.5,
                  color="#111111", alt="white")
    safe_colorbar(fig, im, ax_size, side="right", width=0.10, pad=0.05,
                  label="pore diameter (\u00b5m)", labelsize=6.6)

    # ---- (d) distribution -------------------------------------------------
    bins = np.linspace(0, np.percentile(diam, 99) * 1.15, 20)
    ax_dist.hist(diam, bins=bins, color="#9cc3dd", edgecolor=COLORS["rule"],
                 linewidth=0.5, zorder=2)
    ax_dist.set_xlabel("pore equivalent diameter (\u00b5m)", fontsize=8.0)
    ax_dist.set_ylabel("pore count", fontsize=8.0)
    ax_dist.set_xlim(bins[0], bins[-1])
    ax_dist.tick_params(labelsize=7.0, length=2.5)
    ax_dist.grid(axis="y", linewidth=0.5, alpha=0.7)
    style_axes(ax_dist)

    ax_c = ax_dist.twinx()
    ax_c.plot(diam[order], cum, color=CATEGORICAL[0], lw=1.6, zorder=4)
    ax_c.axhline(50, color=COLORS["baseline"], lw=0.7, ls=":", zorder=3)
    ax_c.plot([d50], [50.0], marker="o", ms=4.0, color=CATEGORICAL[0],
              zorder=5)
    ax_c.annotate(f"$d_{{50}}$ = {d50:.1f} \u00b5m", xy=(d50, 50),
                  xytext=(d50 + 0.07 * bins[-1], 30), fontsize=6.8,
                  color=CATEGORICAL[0],
                  arrowprops=dict(arrowstyle="-", lw=0.7,
                                  color=CATEGORICAL[0]))
    ax_c.set_ylim(0, 104)
    ax_c.set_ylabel("cumulative pore area (%)", fontsize=8.0,
                    color=CATEGORICAL[0])
    ax_c.tick_params(labelsize=7.0, length=2.5, colors=CATEGORICAL[0])
    ax_c.spines["right"].set_visible(True)
    ax_c.spines["right"].set_color(CATEGORICAL[0])
    ax_c.spines["top"].set_visible(False)
    ax_dist.set_title("Size distribution and cumulative area", fontsize=8.5,
                      fontweight="bold", pad=3.0)
    annotate_box(ax_dist,
                 f"n = {n_pore}\nmean = {diam.mean():.1f} \u00b5m\n"
                 f"max = {diam.max():.1f} \u00b5m",
                 loc="lower right", fontsize=6.8)

    panel_labels_inches(fig, [ax_raw, ax_bin, ax_size, ax_dist],
                        dx=-0.26, dy=0.07)

    savefig(fig, "387_mat_porosity_analysis.png")


if __name__ == "__main__":
    main()
