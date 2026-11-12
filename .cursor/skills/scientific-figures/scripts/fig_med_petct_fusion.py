"""Figure 380 - FDG PET/CT: anatomy, function, fusion and the SUV read-out.

The four tiles are one acquisition seen four ways.  ``magma`` stands in for the
classic hot-metal PET ramp: same dark-to-bright warm progression, but
perceptually uniform, so equal steps in SUV are equal steps in brightness
(DESIGN_SYSTEM.md section 2.4; figure 253 made the same substitution).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, COLORS
from figkit import image_axes, axes_inches, corner_stamp, safe_colorbar

N = 224
MM_PX = 220.0 / N
PET_CMAP = "magma"                 # perceptually uniform stand-in for hot
ACT_PER_SUV = 3.2                  # kBq/mL per SUV unit, for the PET tile
SUV_THRESHOLD = 4.0                # hot-spot read-out level for this study


# ---------------------------------------------------------------------------
# Synthetic head phantom (layered ellipses + gyral folding, as in figure 253)
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


# uptake blobs: (cx, cy, sx, sy, amplitude)
UPTAKE = [(0.00, -0.34, 0.26, 0.15, 0.62), (0.34, 0.06, 0.17, 0.24, 0.55),
          (-0.34, 0.06, 0.17, 0.24, 0.55), (0.00, 0.44, 0.30, 0.15, 0.50),
          (0.30, -0.30, 0.085, 0.080, 1.00)]      # the lesion


def petct(seed=15):
    rng = np.random.default_rng(seed)
    x, y = _grid(N)
    wob = 1.0 + 0.014 * _sym(_sm(rng, N, 9))
    rr = _rad(x, y, 0.0, 0.0, 0.80, 0.92) * wob

    ct = np.full((N, N), 0.03)
    ct = np.where(rr <= 1.000, 0.40, ct)
    ct = np.where(rr <= 0.962, 0.92, ct)
    ct = np.where(rr <= 0.900, 0.28, ct)
    brain = rr <= 0.875
    ct = np.where(brain, 0.46 + 0.025 * _sym(_sm(rng, N, 3)), ct)
    rw = _rad(x, y, 0.0, 0.03, 0.62, 0.74) * wob
    ct = np.where(brain & (rw <= 1.0), 0.40 + 0.020 * _sm(rng, N, 5), ct)
    g1, g2 = _sym(_sm(rng, N, 7.0)), _sym(_sm(rng, N, 15.0))
    sulci = (np.sin(8.0 * g1 + 4.2 * g2) > 0.62) & brain & (rr > 0.58)
    ct = np.where(sulci, 0.30, ct)
    for vx, vy, vrx, vry, va in [(0.115, -0.02, 0.055, 0.22, -13),
                                 (-0.115, -0.02, 0.055, 0.22, 13)]:
        ct = np.where(_rad(x, y, vx, vy, vrx, vry, va) * wob <= 1.0, 0.26, ct)
    ct = np.clip(gaussian_filter(ct, 0.9) + 0.008 * rng.standard_normal(
        (N, N)), 0, 1)

    pet = np.zeros((N, N))
    for bx, by, sx, sy, amp in UPTAKE:
        pet += amp * np.exp(-0.5 * (((x - bx) / sx) ** 2 + ((y - by) / sy) ** 2))
    pet *= 0.78 + 0.40 * _sm(rng, N, 12)
    head_soft = gaussian_filter(brain.astype(float), 3.0)
    pet = gaussian_filter(pet, 4.2) * head_soft
    pet = np.clip(pet + 0.035 * _sm(rng, N, 2.5) * head_soft, 0, None)
    pet /= pet.max()
    suv = 0.45 + 9.2 * pet                       # body-weight normalised SUV
    return ct, suv * (pet > 0.012), brain


def main():
    set_pub_style(base_font=8)
    ct, suv, brain = petct()
    act = suv * ACT_PER_SUV
    suv_max = float(suv.max())
    iy, ix = np.unravel_index(np.argmax(suv), suv.shape)

    fig = plt.figure(figsize=(10.4, 4.15))
    tile, gap = 2.34, 0.075
    x0 = (10.4 - (4 * tile + 3 * gap)) / 2.0
    axs = [axes_inches(fig, (x0 + i * (tile + gap), 0.86, tile, tile))
           for i in range(4)]
    for ax in axs:
        image_axes(ax, color="#d5d5d5", lw=0.6)

    # (a) CT
    axs[0].imshow(ct, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    axs[0].set_title("CT", fontsize=9.0, fontweight="bold", pad=3.5)
    corner_stamp(axs[0], "WW 400 / WL +50 HU", loc="lower left", fontsize=6.6)

    # (b) PET
    act_top = float(np.percentile(act[act > 0], 99.4))
    im_pet = axs[1].imshow(act, cmap=PET_CMAP, vmin=0.0, vmax=act_top,
                           interpolation="bilinear")
    axs[1].set_title("PET  (FDG, 60 min p.i.)", fontsize=9.0,
                     fontweight="bold", pad=3.5)
    corner_stamp(axs[1], "attenuation corrected", loc="lower left",
                 fontsize=6.6)
    safe_colorbar(fig, im_pet, axs[1], label="activity (kBq/mL)",
                  side="bottom", width=0.10, pad=0.055, extend="max")

    # (c) fusion
    alpha_max = 0.62
    heat = plt.get_cmap(PET_CMAP)(suv / suv_max)[..., :3]
    a = (np.clip(suv / suv_max, 0, 1) ** 1.15 * alpha_max)[..., None]
    fused = np.clip(np.dstack([ct] * 3) * (1 - a) + heat * a, 0, 1)
    axs[2].imshow(fused, interpolation="bilinear")
    axs[2].set_title("Fusion", fontsize=9.0, fontweight="bold", pad=3.5)
    corner_stamp(axs[2], f"opacity ramps 0 to {alpha_max:.2f} with SUV",
                 loc="lower left", fontsize=6.6)

    # (d) quantitative SUV map, thresholded at the clinical read-out level
    shown = np.where(suv >= SUV_THRESHOLD, suv, np.nan)
    axs[3].imshow(ct, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    im_suv = axs[3].imshow(shown, cmap=PET_CMAP, vmin=SUV_THRESHOLD,
                           vmax=suv_max, interpolation="bilinear", alpha=0.92)
    cs = axs[3].contour(suv, levels=[SUV_THRESHOLD, 8.0], colors="white",
                        linewidths=[0.7, 1.0], linestyles=["--", "-"],
                        zorder=4)
    axs[3].clabel(cs, fmt="%.1f", fontsize=6.0, inline=True)
    axs[3].plot([ix], [iy], marker="+", ms=11, mew=1.3, color="#b23a48",
                zorder=6)
    axs[3].annotate(f"SUVmax {suv_max:.1f} g/mL", xy=(ix, iy),
                    xytext=(0.04, 0.87), textcoords="axes fraction",
                    fontsize=6.8, color="#b23a48", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#b23a48"))
    axs[3].set_title("SUV map", fontsize=9.0, fontweight="bold", pad=3.5)
    corner_stamp(axs[3], f"shown above SUV {SUV_THRESHOLD:.1f}",
                 loc="lower left", fontsize=6.6)
    safe_colorbar(fig, im_suv, axs[3], label="SUV (g/mL)", side="bottom",
                  width=0.10, pad=0.055)

    lesion_suv = suv[max(iy - 5, 0):iy + 6, max(ix - 5, 0):ix + 6]

    savefig(fig, "380_med_petct_fusion.png")


if __name__ == "__main__":
    main()
