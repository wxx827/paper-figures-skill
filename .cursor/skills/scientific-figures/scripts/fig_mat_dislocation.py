"""Figure 388 - Dislocation cell structure imaged in weak-beam dark field.

Bright curvilinear defects on a dark foil: cell walls are the tangled boundary
network of a coarse Laguerre partition, broken into segments so they read as
individual dislocations, plus free dislocations meandering through the cell
interiors.  The density plotted on the right is measured from the drawn line
length, not typed in.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, gaussian_filter

from pubstyle import (set_pub_style, savefig, annotate_box, style_axes,
                      COLORS, CATEGORICAL)
from figkit import (add_scale_bar, add_zoom_inset, halo_text, image_axes,
                    panel_labels_inches, roi_rect, ZOOM, ZOOM_ALT)

NM_PER_PX = 5.0                     # TEM magnification of the synthetic foil
SHAPE = (340, 440)
FOIL_NM = 150.0                     # foil thickness used for the density


def smooth_noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def laguerre_cells(shape, n_cells, seed, relax=2, sigma_r=0.34,
                   warp=6.0, warp_sigma=11.0):
    ny, nx = shape
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.random(n_cells) * nx, rng.random(n_cells) * ny]
    radii = np.exp(sigma_r * rng.standard_normal(n_cells))
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    wx = (xx + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    wy = (yy + warp * smooth_noise(rng, shape, warp_sigma)).ravel()
    w = (radii * 0.55 * (nx * ny / n_cells) ** 0.5) ** 2
    lab = np.zeros(wx.size, dtype=np.int32)
    for _ in range(relax + 1):
        best = np.full(wx.size, np.inf)
        for k in range(n_cells):
            d = (wx - pts[k, 0]) ** 2 + (wy - pts[k, 1]) ** 2 - w[k]
            m = d < best
            best[m] = d[m]
            lab[m] = k
        cnt = np.maximum(np.bincount(lab, minlength=n_cells), 1).astype(float)
        pts = np.c_[np.bincount(lab, weights=wx, minlength=n_cells) / cnt,
                    np.bincount(lab, weights=wy, minlength=n_cells) / cnt]
    return lab.reshape(shape)


def draw_curve(mask, xs, ys):
    """Rasterise a polyline, dropping anything outside the frame.

    Clipping the indices instead would smear the excursion into a straight
    bright line along the image border - a very visible artefact.
    """
    ny, nx = mask.shape
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = int(np.hypot(np.diff(xs), np.diff(ys)).sum() * 2) + 4
    t = np.linspace(0, 1, xs.size)
    tt = np.linspace(0, 1, n)
    jj = np.round(np.interp(tt, t, xs)).astype(int)
    ii = np.round(np.interp(tt, t, ys)).astype(int)
    ok = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
    mask[ii[ok], jj[ok]] = True


PILEUP_XY = (108.0, 236.0)          # obstacle the pile-up stacks against


def build_foil(seed=12):
    rng = np.random.default_rng(seed)
    ny, nx = SHAPE
    cells = laguerre_cells(SHAPE, 17, seed=seed)

    wall = np.zeros(SHAPE, bool)
    wall[:, :-1] |= cells[:, :-1] != cells[:, 1:]
    wall[:-1, :] |= cells[:-1, :] != cells[1:, :]
    wall = binary_dilation(wall, np.ones((3, 3), bool))
    # Roughen the wall so it reads as a tangle of dislocations, not a contour.
    chop = smooth_noise(rng, SHAPE, 1.3) + 0.6 * smooth_noise(rng, SHAPE, 0.7)
    lines = wall & (chop > -1.05)
    lines |= binary_dilation(wall, np.ones((9, 9), bool)) & (chop > 1.25)

    # Free dislocations inside the cells: smoothed random walks.
    free = np.zeros(SHAPE, bool)
    for _ in range(24):
        x0, y0 = rng.random() * nx, rng.random() * ny
        ang = rng.random() * 2 * np.pi
        step = rng.uniform(4.0, 9.0)
        xs, ys = [x0], [y0]
        for _ in range(rng.integers(8, 16)):
            ang += rng.normal(0, 0.55)
            xs.append(xs[-1] + step * np.cos(ang))
            ys.append(ys[-1] + step * np.sin(ang))
        draw_curve(free, xs, ys)

    # A pile-up: parallel loops banked against a wall, spacing widening away
    # from the obstacle.
    pile = np.zeros(SHAPE, bool)
    ang = 0.30
    for i, off in enumerate(np.cumsum(np.linspace(3.4, 9.5, 7))):
        t = np.linspace(-26 + 1.4 * i, 26 - 1.4 * i, 60)
        bow = (0.014 + 0.004 * i) * t ** 2          # loops bow out downstream
        a = ang + rng.normal(0, 0.05)
        xs = (PILEUP_XY[0] + (off + bow) * np.cos(a) - t * np.sin(a)
              + 1.7 * np.sin(t / 11.0 + i))
        ys = (PILEUP_XY[1] + (off + bow) * np.sin(a) + t * np.cos(a)
              + 1.5 * np.cos(t / 8.0 + i))
        draw_curve(pile, xs, ys)

    line_px = lines | free | pile
    inten = (1.0 * lines + 0.82 * binary_dilation(free, np.ones((2, 2), bool))
             + 0.95 * binary_dilation(pile, np.ones((2, 2), bool)))
    img = gaussian_filter(np.clip(inten, 0, 1.2), 0.85)
    img = img / (img.max() + 1e-9)
    img = 0.14 + 0.82 * img ** 0.75
    # Diffuse bend contours and foil-thickness fringes.
    img += 0.10 * smooth_noise(rng, SHAPE, 34.0)
    img += 0.035 * np.sin(np.mgrid[0:ny, 0:nx][1] / 26.0
                          + 2.0 * smooth_noise(rng, SHAPE, 40.0))
    img += 0.030 * rng.standard_normal(SHAPE)
    return np.clip(img, 0.0, 1.0), line_px, cells


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    img, line_px, cells = build_foil()

    # Line length / (area x thickness) -> dislocation density in m^-2.
    length_nm = line_px.sum() * NM_PER_PX / 2.0     # lines are ~2 px wide
    vol_nm3 = SHAPE[0] * SHAPE[1] * NM_PER_PX ** 2 * FOIL_NM
    rho = length_nm / vol_nm3 * 1e18                # nm^-2 -> m^-2
    eps_img = 0.085

    eps = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.085, 0.12, 0.17, 0.24])
    rho_curve = 4e13 + (rho - 4e13) / (1 - np.exp(-eps_img / 0.06)) * \
        (1 - np.exp(-eps / 0.06))
    rng = np.random.default_rng(2)
    rho_meas = rho_curve * np.exp(0.10 * rng.standard_normal(eps.size))
    err = 0.13 * rho_meas

    fig = plt.figure(figsize=(7.8, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.58], wspace=0.20,
                          left=0.035, right=0.975, top=0.880, bottom=0.135)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_rho = fig.add_subplot(gs[0, 1])

    # ---- (a) micrograph ---------------------------------------------------
    ax_img.imshow(img, cmap="gray", vmin=0.05, vmax=1.0,
                  interpolation="nearest")
    image_axes(ax_img, color=COLORS["rule"], lw=0.7)
    ax_img.set_title("Weak-beam dark field, g/3g, 200 kV", fontsize=8.5,
                     fontweight="bold", pad=3.5)
    box = (218, 34, 92, 72)
    add_zoom_inset(ax_img, img, box_xywh=box,
                   inset_loc=(0.615, 0.030, 0.355, 0.355), color=ZOOM,
                   cmap="gray", vmin=0.05, vmax=1.0, connect=True)
    halo_text(ax_img, 0.795, 0.400, "cell wall junction, x3.4",
              transform=ax_img.transAxes, ha="center", va="bottom",
              fontsize=6.4, color="white")
    px, py = PILEUP_XY
    roi_rect(ax_img, (px - 14, py - 34, 82, 68), color=ZOOM_ALT, lw=1.15)
    halo_text(ax_img, px - 14, py - 38, "pile-up", fontsize=6.4,
              color="white", ha="left", va="bottom")
    add_scale_bar(ax_img, 500.0 / NM_PER_PX, label="500 nm",
                  loc=(0.035, 0.905), height=0.024, fontsize=6.5)

    # ---- (b) density vs strain -------------------------------------------
    ax_rho.errorbar(eps, rho_meas, yerr=err, fmt="o", ms=3.4,
                    color=COLORS["rule"], mfc="white", mew=0.9, lw=0.9,
                    capsize=2.0, zorder=3, label="TEM foils")
    xg = np.linspace(0, 0.26, 200)
    ax_rho.plot(xg, 4e13 + (rho - 4e13) / (1 - np.exp(-eps_img / 0.06))
                * (1 - np.exp(-xg / 0.06)), color=CATEGORICAL[1], lw=1.5,
                zorder=2, label="Kocks-Mecking fit")
    ax_rho.plot([eps_img], [rho], marker="D", ms=5.0,
                color=COLORS["accent"], zorder=5, label="this foil")
    ax_rho.axvline(eps_img, color=COLORS["accent"], lw=0.8, ls="--", zorder=1)
    ax_rho.set_xlabel("plastic strain (-)", fontsize=8.0)
    ax_rho.set_ylabel(r"dislocation density (m$^{-2}$)", fontsize=8.0)
    ax_rho.set_xlim(-0.012, 0.262)
    ax_rho.set_ylim(0, max(rho_meas.max() + err.max(), rho) * 1.16)
    ax_rho.tick_params(labelsize=7.0, length=2.5)
    ax_rho.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_rho.yaxis.get_offset_text().set_fontsize(6.5)
    ax_rho.grid(axis="y", linewidth=0.5, alpha=0.7)
    style_axes(ax_rho)
    ax_rho.legend(loc="lower right", frameon=False, fontsize=7.0,
                  handlelength=1.4, borderaxespad=0.3)
    ax_rho.set_title("Density measured from line length", fontsize=8.5,
                     fontweight="bold", pad=3.5)
    annotate_box(ax_rho,
                 f"$\\rho$ = {rho:.2e} m$^{{-2}}$\n"
                 f"cell size = {np.sqrt(SHAPE[0] * SHAPE[1] / 17) * NM_PER_PX:.0f} nm\n"
                 f"foil t = {FOIL_NM:.0f} nm",
                 loc="lower left", fontsize=6.6)

    panel_labels_inches(fig, [ax_img, ax_rho], dx=-0.24, dy=0.08)

    savefig(fig, "388_mat_dislocation.png")


if __name__ == "__main__":
    main()
