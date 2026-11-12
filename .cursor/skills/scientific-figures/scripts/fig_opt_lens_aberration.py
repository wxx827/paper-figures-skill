"""Spherical aberration: where the rays land, what it does to the spot.

The ray trace on the left is the claim: a single-element lens has no focus,
it has a focal *interval*, because the marginal zone bends light harder than
the paraxial zone.  The longitudinal aberration, the best-focus offset and
the blur radius quoted on it are all measured from the traced ray heights.
The two smaller panels are the consequence in image space - the point spread
function with and without the aberrated wavefront, rendered on one shared
scale so the Strehl loss is visible rather than normalised away, and the
Zernike decomposition of the wavefront that produced it.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, image_axes, panel_labels_inches

RED = "#b23a48"        # rays
BLUE = "#3b5b92"       # lens glass
GOLD = "#c9a227"       # aberration
GREY = "#808080"       # ideal / reference
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RAMP = get_cmap("cb_sequential_blue")

FW, FH = 10.4, 6.4
X_LENS = 15.0          # lens plane, mm
F0 = 48.0              # paraxial focal length, mm
SEMI = 25.0            # clear semi-aperture, mm
K_SA = 3.2e-4          # spherical-aberration coefficient, 1/mm^2
X_MAX = 70.0
RAY_H = np.array([4.0, 9.0, 14.0, 18.0, 21.5, 25.0])

# Zernike wavefront, RMS coefficients in waves at 587.6 nm.
ZERN = [("$Z_4$ defocus", 0.052), ("$Z_5$ astig 0", -0.031),
        ("$Z_6$ astig 45", 0.018), ("$Z_7$ coma y", 0.041),
        ("$Z_8$ coma x", -0.066), ("$Z_9$ trefoil", 0.022),
        ("$Z_{11}$ spherical", 0.148)]


def focal(h):
    """Zonal focal length in mm for a ray entering at height ``h`` (mm)."""
    return F0 / (1.0 + K_SA * h ** 2)


def ray_height(h, x):
    xf = X_LENS + focal(h)
    return h * (xf - x) / (xf - X_LENS)


def best_focus():
    x = np.linspace(X_LENS + focal(SEMI) - 6.0, X_LENS + F0 + 3.0, 900)
    rms = np.array([np.sqrt(np.mean(ray_height(RAY_H, xi) ** 2)) for xi in x])
    k = int(np.argmin(rms))
    return float(x[k]), float(rms[k])


def lens_shape(n=90):
    y = np.linspace(-SEMI - 1.0, SEMI + 1.0, n)
    sag = 3.4 * (1.0 - (y / (SEMI + 1.0)) ** 2)
    left = np.column_stack([X_LENS - sag, y])
    right = np.column_stack([X_LENS + sag, y])[::-1]
    return np.vstack([left, right])


def ray_panel(ax, box_aspect):
    y_max = X_MAX / box_aspect / 2.0
    ax.set_xlim(0.0, X_MAX)
    ax.set_ylim(-y_max, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    x_par = X_LENS + F0
    x_marg = X_LENS + focal(SEMI)
    x_best, r_best = best_focus()
    lsa = x_par - x_marg

    ax.plot([0.0, X_MAX], [0.0, 0.0], color=GREY, lw=0.7,
            ls=(0, (7, 2.5, 1.4, 2.5)), zorder=2)
    ax.add_patch(Polygon(lens_shape(), closed=True, facecolor=BLUE,
                         alpha=0.20, edgecolor=BLUE, linewidth=1.0,
                         zorder=3))

    for h in RAY_H:
        for s in (+1.0, -1.0):
            ax.plot([0.0, X_LENS], [s * h, s * h], color=RED, lw=0.9,
                    alpha=0.9, zorder=4)
            ax.plot([X_LENS, X_MAX], [s * h, s * ray_height(h, X_MAX)],
                    color=RED, lw=0.9, alpha=0.9, zorder=4)

    for xv, col, lab in [(x_marg, GOLD, "marginal focus"),
                         (x_par, GREY, "paraxial focus")]:
        ax.plot([xv, xv], [-y_max * 0.62, y_max * 0.62], color=col, lw=0.9,
                ls=(0, (3, 2.2)), zorder=5)
    ax.plot([x_best, x_best], [-y_max * 0.62, y_max * 0.62], color=INK,
            lw=0.9, ls=(0, (1.4, 1.8)), zorder=5)

    ax.annotate("", xy=(x_par, -y_max * 0.74), xytext=(x_marg, -y_max * 0.74),
                arrowprops=dict(arrowstyle="<|-|>", color=GOLD, lw=0.9,
                                mutation_scale=7), zorder=6)
    ax.text((x_par + x_marg) / 2.0, -y_max * 0.80,
            f"longitudinal SA = {lsa:.1f} mm", ha="center", va="top",
            fontsize=7.0, color=GOLD)

    ax.text(x_marg - 0.8, y_max * 0.66, "marginal", ha="right", va="bottom",
            fontsize=6.8, color=GOLD)
    ax.text(x_par + 0.8, y_max * 0.66, "paraxial", ha="left", va="bottom",
            fontsize=6.8, color=SUB)
    ax.annotate(f"circle of least confusion\nRMS blur {r_best:.2f} mm",
                xy=(x_best, r_best), xytext=(x_best - 17.0, y_max * 0.40),
                ha="center", fontsize=6.8, color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
    ax.text(X_LENS, SEMI + 3.2, f"$f_0$ = {F0:.0f} mm,  semi-aperture "
            f"{SEMI:.0f} mm", ha="center", va="bottom", fontsize=6.8,
            color=BLUE)

    x0, wbar = 2.5, 20.0
    ax.plot([x0, x0 + wbar], [-y_max * 0.93] * 2, color=SUB, lw=1.2,
            solid_capstyle="butt", zorder=6)
    for xt in (x0, x0 + wbar):
        ax.plot([xt, xt], [-y_max * 0.93 - 0.9, -y_max * 0.93 + 0.9],
                color=SUB, lw=1.0, zorder=6)
    ax.text(x0 + wbar / 2.0, -y_max * 0.93 + 1.4, f"{wbar:.0f} mm",
            ha="center", va="bottom", fontsize=6.5, color=SUB)

    ax.set_title(f"Marginal rays cross the axis {lsa:.1f} mm early",
                 fontsize=9.0, fontweight="bold", color=INK, pad=4.0)
    return lsa, r_best


def psf(coeffs=None, n=256, crop=40, pad=4.0):
    # The pupil fills only 1/pad of the array so the transform is sampled
    # finely enough to resolve the Airy rings instead of a two-pixel dot.
    u = np.linspace(-pad, pad, n)
    X, Y = np.meshgrid(u, u)
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    pup = (r <= 1.0).astype(float)
    w = np.zeros_like(r)
    if coeffs:
        c = dict(coeffs)
        w += c.get("defocus", 0.0) * np.sqrt(3) * (2 * r ** 2 - 1)
        w += c.get("astig", 0.0) * np.sqrt(6) * r ** 2 * np.cos(2 * th)
        w += c.get("coma", 0.0) * np.sqrt(8) * (3 * r ** 3 - 2 * r) * np.cos(th)
        w += c.get("spher", 0.0) * np.sqrt(5) * (6 * r ** 4 - 6 * r ** 2 + 1)
    field = pup * np.exp(2j * np.pi * w)
    img = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
    c0 = n // 2
    return img[c0 - crop:c0 + crop, c0 - crop:c0 + crop]


def psf_panels(fig, axes):
    ideal = psf()
    aber = psf([("defocus", 0.052), ("astig", -0.031), ("coma", -0.066),
                ("spher", 0.148)])
    peak = ideal.max()
    ideal, aber = ideal / peak, aber / peak
    strehl = float(aber.max())
    norm = mpl.colors.PowerNorm(gamma=0.42, vmin=0.0, vmax=1.0)

    im = None
    for ax, img, name in [(axes[0], ideal, "Diffraction limited"),
                          (axes[1], aber, "With aberration")]:
        im = ax.imshow(img, cmap=RAMP, norm=norm, interpolation="bilinear")
        image_axes(ax, color="#8a8a8a", lw=0.6)
        ax.set_title(name, fontsize=8.2, fontweight="bold", color=INK,
                     pad=3.0)
    axes[1].text(0.5, -0.055, f"Strehl ratio {strehl:.2f}",
                 transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=6.8, color=GOLD)
    axes[0].text(0.5, -0.055, "peak = 1.00 by definition",
                 transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=6.8, color=SUB)

    bb1 = axes[1].get_position()
    cax = fig.add_axes([bb1.x1 + 0.14 / FW, bb1.y0, 0.13 / FW, bb1.height])
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, 0.05, 0.2, 0.5, 1.0])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.5, width=0.7, length=2.5, color="#333333")
    cb.set_label("intensity / ideal peak (-)", fontsize=6.5, color=INK)
    return strehl


def zernike_panel(ax):
    names = [n for n, _ in ZERN]
    vals = np.array([v for _, v in ZERN])
    rms = float(np.sqrt(np.sum(vals ** 2)))
    pos = np.arange(len(vals))

    ax.barh(pos, vals, height=0.62, color=GOLD, alpha=0.90,
            edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0.0, color=SUB, lw=0.8, zorder=4)
    for p, v in zip(pos, vals):
        ax.text(v + (0.006 if v >= 0 else -0.006), p, f"{v:+.3f}",
                ha="left" if v >= 0 else "right", va="center", fontsize=6.2,
                color=INK)

    ax.set_yticks(pos)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.135, 0.205)
    ax.set_xlabel(r"coefficient (waves RMS at 587.6 nm)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.6)
    ax.set_title(f"Wavefront RMS {rms:.3f} waves = " + r"$\lambda$/"
                 + f"{1.0 / rms:.1f}", fontsize=8.2, fontweight="bold",
                 color=INK, pad=4.0)
    return rms


def main():
    set_pub_style()
    fig = plt.figure(figsize=(FW, FH))

    ray_rect = (0.58, 0.58, 5.30, 4.90)
    ax_ray = axes_inches(fig, ray_rect)
    ax_p0 = axes_inches(fig, (6.42, 3.72, 1.42, 1.42))
    ax_p1 = axes_inches(fig, (8.02, 3.72, 1.42, 1.42))
    ax_z = axes_inches(fig, (7.00, 1.02, 2.45, 1.84))

    lsa, r_best = ray_panel(ax_ray, ray_rect[2] / ray_rect[3])
    strehl = psf_panels(fig, [ax_p0, ax_p1])
    rms = zernike_panel(ax_z)

    panel_labels_inches(fig, [ax_ray, ax_p0, ax_z], dx=-0.34, dy=0.14)
    savefig(fig, "698_opt_lens_aberration.png")


if __name__ == "__main__":
    main()
