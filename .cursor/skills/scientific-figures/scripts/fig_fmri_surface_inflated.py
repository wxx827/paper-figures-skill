"""Figure 539 - Vertexwise map on four inflated cortical surface views.

Both hemispheres, lateral and medial, rendered as inflated ovals with the
usual binary curvature shading underneath (dark = sulcus, light = gyrus) and a
thresholded z map on top.  All four views are painted through one shared
``Normalize`` and share one colour bar: with four separate scales the same
tint would mean a different effect size in each quadrant, which is the single
most effective way for a surface figure to mislead.

Parcel outlines come from a nearest-seed partition of the same grid, so the
labels sit inside the regions they name.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

NX, NY = 400, 250              # rendering grid per view
RY = 0.56                      # half-height of the inflated outline
XL, YL = 1.12, 0.68            # displayed half-extent of each view
Z_THRESH = 1.6                 # display threshold on the z map

# Seeds are given in a canonical frame with anterior on the left; each view
# flips x as needed.  Only the named seeds get a text label.
LATERAL = [
    (-0.72, 0.20, "Middle frontal"), (-0.62, -0.24, ""),
    (-0.10, 0.30, "Precentral"), (0.22, 0.30, ""),
    (0.00, -0.30, "Superior temporal"), (0.56, -0.06, "Supramarginal"),
    (0.86, 0.18, "Lateral occipital"), (-0.28, -0.02, ""), (0.36, 0.04, ""),
    (0.48, -0.34, ""),
]
MEDIAL = [
    (-0.68, 0.12, "Medial prefrontal"), (-0.12, 0.32, "Paracentral"),
    (0.44, 0.28, "Precuneus"), (0.88, -0.02, "Cuneus"),
    (0.12, -0.28, "Posterior cingulate"), (-0.38, -0.26, ""),
    (0.64, -0.34, "Lingual"), (0.10, 0.06, ""), (-0.06, -0.06, ""),
]
VIEWS = [
    ("Left hemisphere \u00b7 lateral", LATERAL, +1, 3),
    ("Left hemisphere \u00b7 medial", MEDIAL, -1, 11),
    ("Right hemisphere \u00b7 lateral", LATERAL, -1, 23),
    ("Right hemisphere \u00b7 medial", MEDIAL, +1, 31),
]

# Truncated house warm ramp: the palest two stops vanish against the
# curvature shading, so the overlay starts where it is still separable.
ACT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "surf_warm", get_palette("cb_sequential_warm")[2:])


def outline_radius(theta):
    """Lobed radius that reads as an inflated hemisphere, not as an ellipse."""
    return (1.0 + 0.045 * np.cos(theta) - 0.075 * np.cos(2 * theta) +
            0.030 * np.sin(3 * theta) - 0.020 * np.cos(4 * theta))


def view_fields(seed, flip):
    """Curvature, z map and parcel id on the display grid for one view."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-XL - 0.06, XL + 0.06, NX)
    y = np.linspace(-YL - 0.04, YL + 0.04, NY)
    X, Y = np.meshgrid(x, y)

    th = np.arctan2(Y / RY, X)
    r = np.hypot(X, Y / RY)
    mask = r <= outline_radius(th)

    def field(sig):
        a = gaussian_filter(rng.standard_normal((NY, NX)), sig)
        return a / (a.std() + 1e-12)

    curv = np.sin(3.1 * field((9.0, 13.0)) + 1.5 * field((17.0, 24.0)))
    z = 1.9 * field((17.0, 24.0)) + 0.7 * field((9.0, 12.0))
    return X, Y, mask, curv, z


def parcel_ids(X, Y, seeds, flip):
    pts = np.array([[flip * sx, sy] for sx, sy, _ in seeds])
    d = ((X[..., None] - pts[:, 0]) ** 2 +
         ((Y[..., None] - pts[:, 1]) / 0.85) ** 2)
    return np.argmin(d, axis=-1)


def main():
    set_pub_style(base_font=8)

    fields = []
    for _, seeds, flip, seed in VIEWS:
        X, Y, mask, curv, z = view_fields(seed, flip)
        pid = parcel_ids(X, Y, seeds, flip)
        fields.append((X, Y, mask, curv, z, pid))

    zmax = max(np.nanmax(np.where(f[2], f[4], np.nan)) for f in fields)
    norm = mpl.colors.Normalize(vmin=Z_THRESH, vmax=float(np.ceil(zmax * 2)
                                                          / 2))

    FW, FH = 10.4, 6.4
    top_band, bot_band = 0.58, 0.48
    gx, gy = 0.34, 0.40
    vh = (FH - top_band - bot_band - gy) / 2.0
    vw = vh * XL / YL
    grid_w = 2 * vw + gx
    x0 = max((FW - 0.95 - grid_w) / 2.0, 0.30)
    y_bot = bot_band

    fig = plt.figure(figsize=(FW, FH))
    axes, im = [], None
    for k, ((title, seeds, flip, _), fl) in enumerate(zip(VIEWS, fields)):
        X, Y, mask, curv, z, pid = fl
        r, c = divmod(k, 2)
        ax = axes_inches(fig, (x0 + c * (vw + gx),
                               y_bot + (1 - r) * (vh + gy), vw, vh))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-XL, XL)
        ax.set_ylim(-YL, YL)
        ext = (X.min(), X.max(), Y.min(), Y.max())

        shade = np.where(curv > 0.0, 0.60, 0.80)
        ax.imshow(np.where(mask, shade, np.nan), cmap="gray", vmin=0.0,
                  vmax=1.0, origin="lower", extent=ext,
                  interpolation="bilinear", zorder=1)

        a = np.clip((z - Z_THRESH) / (0.55 * (norm.vmax - Z_THRESH)), 0, 1)
        a = np.where(mask, a, 0.0) * 0.94
        img = ax.imshow(np.where(mask, z, np.nan), cmap=ACT_CMAP, norm=norm,
                        alpha=a, origin="lower", extent=ext,
                        interpolation="bilinear", zorder=2)
        im = im or img

        for p in range(len(seeds)):
            ax.contour(X, Y, np.where(mask, (pid == p).astype(float), np.nan),
                       levels=[0.5], colors="#f2f2f2", linewidths=0.45,
                       alpha=0.80, zorder=3)
        th = np.linspace(0, 2 * np.pi, 400)
        rr = outline_radius(th)
        ax.plot(rr * np.cos(th), RY * rr * np.sin(th), color="#555555",
                lw=0.9, zorder=5)

        for sx, sy, name in seeds:
            if name:
                halo_text(ax, flip * sx, sy, name, fontsize=6.2, ha="center",
                          va="center", color="#1a1a1a", halo="#ffffff",
                          halo_lw=1.9, zorder=6)
        ax.set_title(title, fontsize=8.6, fontweight="bold", pad=3)
        for lab, sgn in (("A", -1.0), ("P", 1.0)):
            ax.text(flip * sgn * (XL - 0.02), -YL + 0.03, lab, fontsize=6.4,
                    color=COLORS["muted"], ha="center", va="bottom")
        axes.append(ax)

    span = fig.add_axes([(x0 + grid_w) / FW, y_bot / FH, 0.02 / FW,
                         (2 * vh + gy) / FH])
    span.set_axis_off()
    span.patch.set_visible(False)
    cb = safe_colorbar(fig, im, span, label="$z$-score (-)", width=0.12,
                       pad=0.16, shrink=0.66,
                       ticks=np.arange(2.0, norm.vmax + 0.01, 1.0))
    cb.ax.text(-0.2, -0.045, f"thresholded at\n$z$ > {Z_THRESH:.1f}",
               transform=cb.ax.transAxes, ha="left", va="top", fontsize=6.2,
               color=COLORS["muted"], linespacing=1.45)

    panel_labels_inches(fig, axes, dx=-0.34, dy=0.20)

    savefig(fig, "539_fmri_surface_inflated.png")


if __name__ == "__main__":
    main()
