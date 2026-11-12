"""Figure 378 - Deformable registration of a follow-up scan onto a baseline.

The two difference panels are the argument, so they share one symmetric colour
scale: without that, the residual after registration would autoscale itself
back up to full contrast and the figure would show no improvement at all.  The
last panel visualises the recovered transform as a warped grid over the
displacement magnitude.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import (image_axes, corner_stamp, shared_norm, safe_colorbar,
                    panel_labels_inches)

N = 224
MM_PX = 220.0 / N


# ---------------------------------------------------------------------------
# Synthetic brain phantom (layered ellipses + gyral folding, as in figure 253)
# ---------------------------------------------------------------------------
def _sm(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _sym(a):
    return 0.5 * (a + a[:, ::-1])


def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return (xx - (n - 1) / 2) / (n / 2), (yy - (n - 1) / 2) / (n / 2)


def _rad(x, y, cx, cy, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    xr = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    yr = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return np.hypot(xr / rx, yr / ry)


def brain_slice(seed=12):
    rng = np.random.default_rng(seed)
    x, y = _grid(N)
    wob = 1.0 + 0.015 * _sym(_sm(rng, N, 9))
    rr = _rad(x, y, 0.0, 0.0, 0.80, 0.92) * wob

    img = np.full((N, N), 0.02)
    img = np.where(rr <= 1.000, 0.42, img)
    img = np.where(rr <= 0.962, 0.09, img)
    img = np.where(rr <= 0.900, 0.13, img)
    brain = rr <= 0.875
    img = np.where(brain, 0.44 + 0.035 * _sym(_sm(rng, N, 3)), img)
    rw = _rad(x, y, 0.0, 0.03, 0.62, 0.74) * wob
    img = np.where(brain & (rw <= 1.0), 0.68 + 0.030 * _sm(rng, N, 5), img)

    g1, g2 = _sym(_sm(rng, N, 7.0)), _sym(_sm(rng, N, 15.0))
    sulci = (np.sin(8.0 * g1 + 4.2 * g2) > 0.60) & brain & (rr > 0.56)
    img = np.where(sulci, 0.14, img)
    for vx, vy, vrx, vry, va in [(0.115, -0.02, 0.055, 0.22, -13),
                                 (-0.115, -0.02, 0.055, 0.22, 13),
                                 (0.0, 0.10, 0.030, 0.10, 0)]:
        img = np.where(_rad(x, y, vx, vy, vrx, vry, va) * wob <= 1.0, 0.07, img)
    img = np.where((np.abs(x) < 0.010) & brain, 0.24, img)
    img = gaussian_filter(img, 0.8)
    return np.clip(img, 0, 1), brain


YY, XX = np.mgrid[0:N, 0:N].astype(float)


def warp(img, u, v):
    """Resample ``img`` at ``p + (u, v)``; u is along x, v along y."""
    return map_coordinates(img, [YY + v, XX + u], order=1, mode="nearest")


def transform(rng, rot_deg=4.2, tx=5.0, ty=-3.5, elastic=4.0):
    """Rigid part plus a smooth elastic part, in pixels."""
    cx = cy = (N - 1) / 2.0
    t = np.deg2rad(rot_deg)
    dx, dy = XX - cx, YY - cy
    u = (np.cos(t) - 1.0) * dx - np.sin(t) * dy + tx
    v = np.sin(t) * dx + (np.cos(t) - 1.0) * dy + ty
    u = u + elastic * _sm(rng, N, 24)
    v = v + elastic * _sm(rng, N, 24)
    return u, v


def rmse(a, b, mask):
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def main():
    set_pub_style(base_font=8)
    fixed, brain = brain_slice()
    rng = np.random.default_rng(23)
    u, v = transform(rng)

    moving = warp(fixed, u, v)
    moving = np.clip(moving * 1.06 - 0.015 + 0.012 * rng.standard_normal(
        (N, N)), 0, 1)                                   # scanner drift + noise

    # the estimate recovers most, but not all, of the true deformation
    u_est = 0.945 * u + 0.55 * _sm(rng, N, 30)
    v_est = 0.945 * v + 0.55 * _sm(rng, N, 30)
    registered = warp(moving, -u_est, -v_est)

    d_before = fixed - moving
    d_after = fixed - registered
    r_before, r_after = rmse(fixed, moving, brain), rmse(fixed, registered,
                                                         brain)
    mag = np.hypot(u, v) * MM_PX
    err = np.hypot(u - u_est, v - v_est) * MM_PX

    dnorm = shared_norm([d_before[brain], d_after[brain]], robust=(1, 99),
                        symmetric=True)
    dcmap = get_cmap("cb_diverging")

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 3, wspace=0.045, hspace=0.150,
                          left=0.045, right=0.945, top=0.900, bottom=0.150)
    axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    for ax in axs:
        image_axes(ax, color="#d5d5d5", lw=0.6)

    grey = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
    axs[0].imshow(fixed, **grey)
    axs[0].set_title("Fixed  (baseline scan)", fontsize=8.8, pad=3.5)
    axs[1].imshow(moving, **grey)
    axs[1].set_title("Moving  (follow-up scan)", fontsize=8.8, pad=3.5)
    axs[2].imshow(registered, **grey)
    axs[2].set_title("Moving after registration", fontsize=8.8, pad=3.5,
                     fontweight="bold")
    corner_stamp(axs[1], f"applied 4.2 deg + {np.hypot(5.0, 3.5) * MM_PX:.1f} "
                         f"mm, up to {mag[brain].max():.1f} mm total",
                 loc="lower left", fontsize=6.4)

    im = axs[3].imshow(np.where(brain, d_before, np.nan), cmap=dcmap,
                       norm=dnorm, interpolation="bilinear")
    axs[3].imshow(fixed, cmap="gray", vmin=0, vmax=1, alpha=0.20, zorder=0)
    axs[3].set_title("Difference before", fontsize=8.8, pad=3.5)
    corner_stamp(axs[3], f"RMSE {r_before:.4f} a.u.", loc="lower left",
                 fontsize=6.8, color="#1a1a1a", halo="white")

    axs[4].imshow(np.where(brain, d_after, np.nan), cmap=dcmap, norm=dnorm,
                  interpolation="bilinear")
    axs[4].imshow(fixed, cmap="gray", vmin=0, vmax=1, alpha=0.20, zorder=0)
    axs[4].set_title("Difference after", fontsize=8.8, pad=3.5,
                     fontweight="bold")
    corner_stamp(axs[4], f"RMSE {r_after:.4f} a.u.  "
                         f"({100 * (1 - r_after / r_before):.0f} % lower)",
                 loc="lower left", fontsize=6.8, color="#1a1a1a", halo="white")
    safe_colorbar(fig, im, axs[4], label="fixed - moving (a.u.)",
                  side="bottom", width=0.10, pad=0.055)

    # -- recovered deformation, drawn as a warped grid -----------------------
    imm = axs[5].imshow(np.where(brain, mag, np.nan),
                        cmap=get_cmap("cb_sequential_blue"),
                        vmin=0.0, vmax=float(np.nanmax(mag[brain])),
                        interpolation="bilinear")
    axs[5].imshow(fixed, cmap="gray", vmin=0, vmax=1, alpha=0.25, zorder=0)
    step = 14
    for j in range(0, N, step):
        xs = np.arange(0, N, 2.0)
        ys = np.full_like(xs, float(j))
        axs[5].plot(xs + map_coordinates(u_est, [ys, xs], order=1),
                    ys + map_coordinates(v_est, [ys, xs], order=1),
                    color="#333333", lw=0.55, alpha=0.85, zorder=4)
        ys2 = np.arange(0, N, 2.0)
        xs2 = np.full_like(ys2, float(j))
        axs[5].plot(xs2 + map_coordinates(u_est, [ys2, xs2], order=1),
                    ys2 + map_coordinates(v_est, [ys2, xs2], order=1),
                    color="#333333", lw=0.55, alpha=0.85, zorder=4)
    axs[5].set_xlim(-0.5, N - 0.5)
    axs[5].set_ylim(N - 0.5, -0.5)
    axs[5].set_title("Recovered deformation field", fontsize=8.8, pad=3.5)
    corner_stamp(axs[5], f"field error  mean {err[brain].mean():.2f} mm, "
                         f"p95 {np.percentile(err[brain], 95):.2f} mm",
                 loc="lower left", fontsize=6.4)
    safe_colorbar(fig, imm, axs[5], label="displacement (mm)",
                  side="bottom", width=0.10, pad=0.055)

    for ax in axs:
        ax.apply_aspect()
    panel_labels_inches(fig, axs, dx=-0.16, dy=0.10)


    savefig(fig, "378_med_registration.png")


if __name__ == "__main__":
    main()
