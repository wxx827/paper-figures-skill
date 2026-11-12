"""Atmospheric correction: TOA to BOA reflectance.

Top row (four co-registered rasters): TOA false-free RGB, BOA RGB, the red-band
reflectance difference and the retrieved aerosol optical depth at 550 nm.
Bottom row (two analysis panels): TOA vs BOA spectra for four surface types
and validation against field spectrometer measurements before / after
correction.

Radiative-transfer forward model used to build the TOA scene:
    rho_TOA(lam) = rho_path(lam, AOD) + T(lam, AOD) * rho_BOA(lam)
with path radiance decreasing and transmittance increasing with wavelength,
so the haze signature is strongest in the blue band.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, PALETTES, savefig, style_axes, add_panel_label

N = 180                                  # 30 m pixels -> 5.4 km square
WL = np.array([490.0, 560.0, 665.0, 842.0, 1610.0, 2190.0])
BAND_NAMES = ["B2\n490", "B3\n560", "B4\n665", "B8\n842", "B11\n1610",
              "B12\n2190"]
# path-radiance and transmittance coefficients per band (dimensionless)
A_PATH = np.array([0.150, 0.108, 0.074, 0.032, 0.010, 0.005])
K_EXT = np.array([0.62, 0.44, 0.31, 0.17, 0.06, 0.035])

ENDMEMBERS = {
    "Water": [0.045, 0.035, 0.022, 0.012, 0.006, 0.004],
    "Forest": [0.028, 0.055, 0.032, 0.420, 0.160, 0.065],
    "Vigorous crop": [0.035, 0.070, 0.045, 0.480, 0.220, 0.095],
    "Senescent crop": [0.075, 0.120, 0.170, 0.280, 0.320, 0.240],
    "Bare soil": [0.095, 0.145, 0.205, 0.280, 0.360, 0.290],
    "Built-up": [0.110, 0.135, 0.160, 0.190, 0.240, 0.210],
}
CLASS_ORDER = list(ENDMEMBERS)


def norm01(a):
    a = np.asarray(a, float)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def fractal_noise(rng, shape, sigmas=(20, 8, 3), weights=(1.0, 0.5, 0.22)):
    out = np.zeros(shape)
    for s, w in zip(sigmas, weights):
        out += w * gaussian_filter(rng.normal(size=shape), s, mode="reflect")
    return norm01(out)


def build_surface(rng):
    yy, xx = np.mgrid[0:N, 0:N]
    elev = fractal_noise(rng, (N, N), (24, 9, 3))

    river = np.abs(xx - (34 + 16 * np.sin(yy / 26.0) + 6 * np.sin(yy / 9.0))) < 2.2
    lake = ((xx - 138.0) ** 2 / 20.0 ** 2 + (yy - 40.0) ** 2 / 11.0 ** 2) < 1.0
    water = river | lake

    urban = np.exp(-(((xx - 118.0) / 26.0) ** 2 + ((yy - 128.0) / 20.0) ** 2))
    urban = (urban + 0.4 * (fractal_noise(rng, (N, N), (6, 2)) - 0.45)) > 0.45

    forest = (elev > np.quantile(elev, 0.76)) & (~water) & (~urban)

    block = (yy // 11) * 512 + (xx // 14)
    nb = block.max() + 1
    kind = rng.choice([2, 3, 4], size=nb, p=[0.55, 0.27, 0.18])
    lc = kind[block]
    lc[forest] = 1
    lc[urban] = 5
    lc[water] = 0

    # per-band reflectance cube with parcel-level and pixel-level variability
    cube = np.zeros((N, N, WL.size))
    gain = 1.0 + 0.12 * rng.normal(size=nb)[block][..., None]
    for i, name in enumerate(CLASS_ORDER):
        m = lc == i
        cube[m] = np.asarray(ENDMEMBERS[name])
    cube *= np.clip(gain, 0.7, 1.35)
    cube += 0.010 * (fractal_noise(rng, (N, N), (4, 1.2))[..., None] - 0.5)
    cube += rng.normal(0, 0.004, cube.shape)
    for k in range(WL.size):
        cube[..., k] = gaussian_filter(cube[..., k], 0.6)
    return np.clip(cube, 0.001, 0.95), lc


def forward_toa(boa, aod):
    """rho_TOA = rho_path + T * rho_BOA, band-dependent."""
    path = aod[..., None] * A_PATH[None, None, :]
    trans = np.exp(-K_EXT[None, None, :] * aod[..., None] * 1.55)
    return path + trans * boa


def stretch_rgb(cube, lo, hi, gamma=1.25):
    rgb = np.dstack([cube[..., 2], cube[..., 1], cube[..., 0]])
    out = (rgb - lo) / (hi - lo)
    return np.clip(out, 0, 1) ** (1.0 / gamma)


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
    rng = np.random.default_rng(20240185)

    boa, lc = build_surface(rng)
    aod = 0.12 + 0.42 * fractal_noise(rng, (N, N), (26, 10), (1.0, 0.35))
    aod += 0.26 * np.exp(-(((np.mgrid[0:N, 0:N][1] - 120.0) / 40.0) ** 2
                           + ((np.mgrid[0:N, 0:N][0] - 132.0) / 34.0) ** 2))
    aod = gaussian_filter(np.clip(aod, 0.08, 0.80), 3.0)
    toa = forward_toa(boa, aod)

    lo = np.percentile(np.dstack([boa[..., 2], boa[..., 1], boa[..., 0]]),
                       1.5, axis=(0, 1))
    hi = np.percentile(np.dstack([boa[..., 2], boa[..., 1], boa[..., 0]]),
                       99.0, axis=(0, 1))
    rgb_boa = stretch_rgb(boa, lo, hi)
    rgb_toa = stretch_rgb(toa, lo, hi)     # identical stretch -> haze visible

    fig = plt.figure(figsize=(7.4, 5.0))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[0.84, 1.0],
                  hspace=0.36, wspace=0.24)

    # ---- (a)-(b) TOA / BOA RGB -------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(rgb_toa, origin="lower", interpolation="bilinear")
    frame_raster(ax_a)
    ax_a.set_title("TOA reflectance", fontsize=8.0)
    add_panel_label(ax_a, "(a)", x=-0.04, y=1.03, size=10)
    ax_a.text(0.5, -0.055, "uncorrected, hazy", transform=ax_a.transAxes,
              ha="center", va="top", fontsize=6.5, color="#555555")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(rgb_boa, origin="lower", interpolation="bilinear")
    frame_raster(ax_b)
    ax_b.set_title("BOA reflectance", fontsize=8.0)
    add_panel_label(ax_b, "(b)", x=-0.04, y=1.03, size=10)
    ax_b.text(0.5, -0.055, "6S-corrected", transform=ax_b.transAxes,
              ha="center", va="top", fontsize=6.5, color="#555555")

    # ---- (c) red-band difference -----------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    diff = toa[..., 2] - boa[..., 2]
    im_c = ax_c.imshow(diff, origin="lower", cmap="RdPu",
                       vmin=0, vmax=np.percentile(diff, 99.5),
                       interpolation="bilinear")
    frame_raster(ax_c)
    ax_c.set_title("TOA \u2212 BOA", fontsize=8.0)
    add_panel_label(ax_c, "(c)", x=-0.04, y=1.03, size=10)

    # ---- (d) AOD map ------------------------------------------------------
    ax_d = fig.add_subplot(gs[0, 3])
    im_d = ax_d.imshow(aod, origin="lower", cmap="YlOrBr",
                       vmin=0.1, vmax=0.8, interpolation="bilinear")
    frame_raster(ax_d)
    ax_d.set_title("Aerosol loading", fontsize=8.0)
    add_panel_label(ax_d, "(d)", x=-0.04, y=1.03, size=10)

    # ---- (e) TOA vs BOA spectra ------------------------------------------
    ax_e = fig.add_subplot(gs[1, :2])
    targets = ["Water", "Forest", "Senescent crop", "Built-up"]
    cols = ["#2166ac", "#0b3d2e", "#c9a227", "#8c2f2f"]
    aod_ref = 0.62
    xb = np.arange(WL.size)
    for name, c in zip(targets, cols):
        rb = np.asarray(ENDMEMBERS[name])
        rt = aod_ref * A_PATH + np.exp(-K_EXT * aod_ref * 1.55) * rb
        ax_e.plot(xb, rb, color=c, lw=1.5, marker="o", ms=3.2,
                  mec="white", mew=0.5, zorder=4)
        ax_e.plot(xb, rt, color=c, lw=1.1, ls="--", marker="s", ms=2.8,
                  alpha=0.85, zorder=3)
    ax_e.set_xticks(xb)
    ax_e.set_xticklabels(BAND_NAMES, fontsize=6.2)
    ax_e.set_xlabel("Sentinel-2 band / central wavelength (nm)", labelpad=1.5)
    ax_e.set_ylabel("Reflectance (\u2013)")
    ax_e.set_ylim(0, 0.56)
    ax_e.set_xlim(-0.35, WL.size - 0.65)
    style_axes(ax_e)
    ax_e.grid(axis="y", lw=0.5, alpha=0.55)
    ax_e.axvspan(-0.35, 1.5, color="#4a90d9", alpha=0.06, lw=0)
    ax_e.text(0.05, 0.545, "largest haze effect\nin the visible",
              fontsize=6.2, color="#3b5b92", va="top", linespacing=1.2)
    h1 = [Line2D([], [], color=c, lw=1.5, marker="o", ms=3.0, label=n)
          for n, c in zip(targets, cols)]
    h2 = [Line2D([], [], color="#555555", lw=1.5, label="BOA (surface)"),
          Line2D([], [], color="#555555", lw=1.1, ls="--",
                 label=f"TOA (AOD = {aod_ref:.2f})")]
    lg1 = ax_e.legend(handles=h1, loc="upper right", fontsize=6.3,
                      frameon=False, handlelength=1.3, labelspacing=0.22,
                      borderaxespad=0.2)
    ax_e.add_artist(lg1)
    lg1.set_clip_on(False)
    lg2 = ax_e.legend(handles=h2, loc="upper left",
                      bbox_to_anchor=(0.015, 0.79), fontsize=6.3,
                      frameon=False, handlelength=1.6, labelspacing=0.22,
                      borderaxespad=0.0)
    lg2.set_clip_on(False)
    add_panel_label(ax_e, "(e)", x=-0.10, y=1.03, size=10)

    # ---- (f) validation against field spectra ----------------------------
    ax_f = fig.add_subplot(gs[1, 2:])
    n_pts = 95
    field = np.concatenate([
        rng.uniform(0.02, 0.10, n_pts // 3),
        rng.uniform(0.10, 0.26, n_pts // 3),
        rng.uniform(0.26, 0.42, n_pts - 2 * (n_pts // 3))])
    aod_s = rng.uniform(0.15, 0.85, n_pts)
    # residual retrieval error (BRDF, adjacency, water-vapour) is what keeps
    # R^2 realistic; the TOA points carry the extra AOD-driven scatter
    resid = field * rng.normal(0, 0.17, n_pts) + rng.normal(0, 0.034, n_pts)
    sat_toa = (aod_s * A_PATH[2] * 1.9
               + np.exp(-K_EXT[2] * aod_s * 1.55) * field
               + resid + rng.normal(0, 0.018, n_pts))
    sat_boa = field * rng.normal(1.0, 0.05, n_pts) + resid * 0.92

    def stats(x, y):
        k, b = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        bias = float(np.mean(y - x))
        return k, b, r2, rmse, bias

    st_t = stats(field, sat_toa)
    st_b = stats(field, sat_boa)
    lim = [0.0, 0.50]
    ax_f.plot(lim, lim, color="#555555", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax_f.scatter(field, sat_toa, s=13, facecolor="#d98c5f", edgecolor="#8c4a12",
                 linewidth=0.35, alpha=0.85, zorder=3, label="before correction (TOA)")
    ax_f.scatter(field, sat_boa, s=13, facecolor="#2e7d5b", edgecolor="#0b3d2e",
                 linewidth=0.35, alpha=0.9, marker="^", zorder=4,
                 label="after correction (BOA)")
    xg = np.linspace(*lim, 20)
    ax_f.plot(xg, st_t[0] * xg + st_t[1], color="#8c4a12", lw=1.3, zorder=5)
    ax_f.plot(xg, st_b[0] * xg + st_b[1], color="#0b3d2e", lw=1.3, zorder=5)
    ax_f.set_xlim(lim)
    ax_f.set_ylim(lim)
    ax_f.set_aspect("equal")
    ax_f.set_xlabel("Field spectrometer reflectance (\u2013)", labelpad=1.5)
    ax_f.set_ylabel("Satellite reflectance (\u2013)")
    style_axes(ax_f)
    ax_f.text(0.035, 0.975,
              f"$\\bf{{TOA}}$  slope {st_t[0]:.2f}, $R^2$ {st_t[2]:.2f}\n"
              f"        RMSE {st_t[3]:.3f}, bias $+${st_t[4]:.3f}\n"
              f"$\\bf{{BOA}}$  slope {st_b[0]:.2f}, $R^2$ {st_b[2]:.2f}\n"
              f"        RMSE {st_b[3]:.3f}, bias {st_b[4]:+.3f}",
              transform=ax_f.transAxes, ha="left", va="top", fontsize=6.2,
              linespacing=1.35,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                        lw=0.5, alpha=0.95))
    ax_f.legend(loc="lower right", fontsize=6.2, frameon=False,
                handletextpad=0.35, borderaxespad=0.3, labelspacing=0.25)
    add_panel_label(ax_f, "(f)", x=-0.10, y=1.03, size=10)

    fig.subplots_adjust(top=0.885, bottom=0.085, left=0.055, right=0.985)

    # colorbars for (c) and (d), placed under the drawn (equal-aspect) boxes
    fig.canvas.draw()
    for ax, im, lab in ((ax_c, im_c, "$\\Delta\\rho_{665}$ (\u2013)"),
                        (ax_d, im_d, "AOD$_{550}$ (\u2013)")):
        p = ax.get_position()
        cax = fig.add_axes([p.x0 + 0.10 * p.width, p.y0 - 0.030,
                            0.80 * p.width, 0.014])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=6.4, labelpad=1.5)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(labelsize=6.0, width=0.6, length=2.0, pad=1.2)
        cb.locator = plt.MaxNLocator(4)
        cb.update_ticks()

    savefig(fig, "185_rs_atmos_correction.png")


if __name__ == "__main__":
    main()
