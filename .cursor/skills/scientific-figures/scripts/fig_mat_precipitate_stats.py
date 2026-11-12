"""Figure 391 - Precipitate coarsening over three ageing times.

Top row: HAADF-style fields at 2 h, 24 h and 200 h, same magnification and the
same grain boundary running through each, with the precipitate-free zone
widening as ageing proceeds.  Bottom row: the radii of exactly those particles,
on one shared pair of axes, against the LSW distribution.  The coarsening
exponent quoted at the end is fitted to the three measured means.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import (set_pub_style, savefig, annotate_box, style_axes,
                      COLORS, CATEGORICAL)
from figkit import (add_scale_bar, apply_shared_limits, image_axes,
                    panel_labels_inches)

NM_PER_PX = 1.0
SHAPE = (380, 400)
AREA_FRAC = 0.145
TIMES_H = np.array([2.0, 24.0, 200.0])
R0_CUBE, K_COARSE = 23.2, 20.4          # nm^3, nm^3/h


def smooth_noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def lsw_pdf(u):
    """Lifshitz-Slyozov-Wagner size distribution, u = r / r_mean."""
    u = np.asarray(u, float)
    out = np.zeros_like(u)
    m = (u > 0) & (u < 1.5)
    v = u[m]
    out[m] = (v ** 2 * (3.0 / (3.0 + v)) ** (7.0 / 3.0)
              * (1.5 / (1.5 - v)) ** (11.0 / 3.0)
              * np.exp(-v / (1.5 - v)))
    return out


def sample_lsw(rng, n, r_mean):
    u = np.linspace(1e-4, 1.4999, 4000)
    p = lsw_pdf(u)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, u) * r_mean / np.trapezoid(
        u * p, u) * np.trapezoid(p, u)


def build_field(r_mean, pfz_nm, seed):
    """Return (image, radii) for one ageing state."""
    rng = np.random.default_rng(seed)
    ny, nx = SHAPE
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

    # One wandering grain boundary across the field.
    gb_y = (0.30 * ny + 0.16 * ny * np.sin(np.arange(nx) / 130.0)
            + 9.0 * smooth_noise(rng, (1, nx), 22.0).ravel())
    d_gb = np.abs(yy - gb_y[None, :])

    n_part = int(AREA_FRAC * nx * ny / (np.pi * r_mean ** 2))
    radii, cx, cy = [], [], []
    while len(radii) < n_part:
        x = rng.random() * nx
        y = rng.random() * ny
        if abs(y - gb_y[int(np.clip(x, 0, nx - 1))]) < pfz_nm:
            continue
        radii.append(0.0)
        cx.append(x)
        cy.append(y)
    radii = sample_lsw(rng, len(radii), r_mean)
    cx, cy = np.array(cx), np.array(cy)

    img = 0.17 + 0.055 * smooth_noise(rng, SHAPE, 1.1)
    img += 0.030 * smooth_noise(rng, SHAPE, 6.0)
    part = np.zeros(SHAPE)
    for x0, y0, r in zip(cx, cy, radii):
        i0, i1 = int(max(y0 - r - 3, 0)), int(min(y0 + r + 4, ny))
        j0, j1 = int(max(x0 - r - 3, 0)), int(min(x0 + r + 4, nx))
        if i1 <= i0 or j1 <= j0:
            continue
        d = np.hypot(yy[i0:i1, j0:j1] - y0, xx[i0:i1, j0:j1] - x0)
        part[i0:i1, j0:j1] = np.maximum(
            part[i0:i1, j0:j1], np.clip((r + 0.9 - d) / 1.8, 0.0, 1.0))
    # Coarse boundary precipitates on the grain boundary itself.
    for _ in range(int(nx / (5.0 * r_mean))):
        x0 = rng.random() * nx
        y0 = gb_y[int(np.clip(x0, 0, nx - 1))] + rng.normal(0, 1.5)
        r = r_mean * rng.uniform(1.4, 2.4)
        i0, i1 = int(max(y0 - r - 3, 0)), int(min(y0 + r + 4, ny))
        j0, j1 = int(max(x0 - r - 3, 0)), int(min(x0 + r + 4, nx))
        d = np.hypot(yy[i0:i1, j0:j1] - y0, xx[i0:i1, j0:j1] - x0)
        part[i0:i1, j0:j1] = np.maximum(
            part[i0:i1, j0:j1], np.clip((r - d) / 1.8, 0.0, 1.0))

    img += 0.68 * gaussian_filter(part, 0.7) * (0.86 + 0.22 * smooth_noise(
        rng, SHAPE, 3.0))
    img += 0.30 * np.exp(-(d_gb / 2.1) ** 2)          # boundary contrast
    img += 0.020 * rng.standard_normal(SHAPE)
    return np.clip(img, 0.0, 1.0), radii


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    r_mean = (R0_CUBE + K_COARSE * TIMES_H) ** (1.0 / 3.0)
    pfz = 6.0 + 1.9 * r_mean

    fields, radii = [], []
    for i, (rm, pz) in enumerate(zip(r_mean, pfz)):
        img, r = build_field(rm, pz, seed=20 + i)
        fields.append(img)
        radii.append(r)

    meas_mean = np.array([r.mean() for r in radii])
    dens = np.array([r.size / (SHAPE[0] * SHAPE[1] * NM_PER_PX ** 2 * 1e-6)
                     for r in radii])                  # particles per um^2
    expo = np.polyfit(np.log(TIMES_H), np.log(meas_mean), 1)[0]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.60], wspace=0.13,
                          hspace=0.22, left=0.045, right=0.985, top=0.885,
                          bottom=0.085)
    ax_i = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_h = [fig.add_subplot(gs[1, c]) for c in range(3)]

    for c in range(3):
        ax_i[c].imshow(fields[c], cmap="gray", vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        image_axes(ax_i[c], color=COLORS["rule"], lw=0.7)
        ax_i[c].set_title(f"{TIMES_H[c]:.0f} h at 180 \u00b0C", fontsize=8.5,
                          fontweight="bold", pad=3.5)
        add_scale_bar(ax_i[c], 100.0 / NM_PER_PX, label="100 nm",
                      loc=(0.035, 0.045), height=0.022, fontsize=6.5)
        ax_i[c].text(0.975, 0.035, f"PFZ \u2248 {pfz[c]:.0f} nm",
                     transform=ax_i[c].transAxes, ha="right", va="bottom",
                     fontsize=6.5, color="white")

    edges = np.linspace(0, max(r.max() for r in radii) * 1.04, 26)
    ug = np.linspace(1e-3, 1.4999, 400)
    u_bar = np.trapezoid(ug * lsw_pdf(ug), ug) / np.trapezoid(lsw_pdf(ug), ug)
    tops = []
    for c in range(3):
        h, _, _ = ax_h[c].hist(radii[c], bins=edges, density=True,
                               color="#9cc3dd", edgecolor=COLORS["rule"],
                               linewidth=0.4, zorder=2)
        rg = ug * meas_mean[c] / u_bar
        pdf = lsw_pdf(ug) / np.trapezoid(lsw_pdf(ug), rg)
        tops.append(max(h.max(), pdf.max()))
        ax_h[c].plot(rg, pdf, color=CATEGORICAL[0], lw=1.5, zorder=4)
        ax_h[c].axvline(meas_mean[c], color=COLORS["rule"], lw=0.9, ls="--",
                        zorder=3)
        ax_h[c].set_xlabel("precipitate radius (nm)", fontsize=7.8)
        ax_h[c].tick_params(labelsize=6.8, length=2.5)
        ax_h[c].grid(axis="y", linewidth=0.5, alpha=0.7)
        style_axes(ax_h[c])
        annotate_box(ax_h[c],
                     f"$\\bar r$ = {meas_mean[c]:.1f} nm\n"
                     f"n = {radii[c].size}\n"
                     f"$N_A$ = {dens[c]:.0f} \u00b5m$^{{-2}}$",
                     loc="upper right", fontsize=6.6)
        if c == 0:
            ax_h[c].set_ylabel("probability density (nm$^{-1}$)",
                               fontsize=7.8)
            ax_h[c].text(0.03, 0.55, "LSW", transform=ax_h[c].transAxes,
                         fontsize=6.8, color=CATEGORICAL[0], ha="left")
        else:
            ax_h[c].tick_params(labelleft=False)

    apply_shared_limits(ax_h, [np.array([0.0, edges[-1]])], axis="x", pad=0.0)
    for a in ax_h:
        a.set_ylim(0, max(tops) * 1.12)

    panel_labels_inches(fig, ax_i + ax_h, dx=-0.24, dy=0.08)

    savefig(fig, "391_mat_precipitate_stats.png")


if __name__ == "__main__":
    main()
