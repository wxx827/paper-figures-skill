"""Figure 534 - Thresholded group fMRI activation on twelve axial slices.

The montage is one logical panel: twelve axial levels through a synthetic
2 mm-isotropic statistical parametric map, warm suprathreshold clusters
composited on the grey-scale anatomy.  Every tile is rendered through the
*same* ``Normalize`` and the *same* grey window, so cluster intensity is
comparable across levels - the one thing a per-slice autoscale would destroy.
Cluster coordinates, peak t and extent in the footer are measured from the
volume at run time, never typed in.

The statistical volume is coarse (2 mm, as an EPI would be) while the anatomy
is rendered on a 3x finer in-plane grid, which is also what a real overlay
looks like: crisp T1 underneath, smooth functional blobs on top.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label as cc_label, zoom

from pubstyle import set_pub_style, savefig, COLORS
from figkit import (axes_inches, image_axes, corner_stamp, add_scale_bar,
                    safe_colorbar, shared_norm)

# ---- acquisition geometry --------------------------------------------------
VOX = 2.0                                  # isotropic voxel size (mm)
NX, NY, NZ = 92, 110, 66                   # L-R, A-P, I-S voxel counts
ORIGIN = (-92.0, -118.0, -60.0)            # left/posterior/inferior corner, mm
UP = 3                                     # in-plane upsampling for anatomy
BRAIN_C = (0.0, -18.0, 14.0)               # ellipsoid centre (mm)
BRAIN_R = (68.0, 88.0, 56.0)               # ellipsoid semi-axes (mm)
XLIM = (-82.0, 82.0)                       # displayed field of view (mm)
YLIM = (-118.0, 84.0)

SLICES_MM = list(range(-18, 49, 6))        # twelve axial levels, z in mm

# Synthetic effect foci: (x, y, z) mm in template space, peak t, extent sigma mm
FOCI = [
    (-40.0, -20.0, 52.0, 5.6, 10.0),       # left precentral / M1 hand knob
    (40.0, -22.0, 52.0, 3.9, 8.0),         # right precentral
    (-52.0, 10.0, 20.0, 4.8, 9.0),         # left inferior frontal gyrus
    (4.0, 8.0, 46.0, 4.4, 8.5),            # supplementary motor area
    (-6.0, -84.0, 6.0, 6.1, 11.0),         # occipital / V1-V2
    (-38.0, -60.0, 42.0, 4.1, 8.0),        # left intraparietal sulcus
    (46.0, -50.0, 22.0, 3.6, 7.5),         # right temporoparietal junction
    (-28.0, -6.0, 6.0, 3.5, 6.5),          # left putamen
]

T_THRESH = 3.10                            # cluster-forming height threshold
K_EXTENT = 25                              # cluster extent threshold (voxels)

# Truncated magma: the perceptually uniform stand-in this library uses for the
# banned `hot` PET/activation ramp (DESIGN_SYSTEM 2.4).  The low end is cut off
# so the faintest suprathreshold voxel is still separable from grey anatomy.
ACT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "act_warm", plt.get_cmap("magma")(np.linspace(0.34, 0.94, 256)))


def _sm(rng, shape, sigma):
    """Unit-variance smooth Gaussian random field."""
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _axes_mm():
    x = ORIGIN[0] + (np.arange(NX) + 0.5) * VOX
    y = ORIGIN[1] + (np.arange(NY) + 0.5) * VOX
    z = ORIGIN[2] + (np.arange(NZ) + 0.5) * VOX
    return x, y, z


def _radius(X, Y, zmm, c, r, wob=1.0):
    return np.sqrt(((X - c[0]) / r[0]) ** 2 + ((Y - c[1]) / r[1]) ** 2 +
                   ((zmm - c[2]) / r[2]) ** 2) * wob


def slice_anatomy(zmm, fields, iz, rng):
    """One axial T1-like slice on the fine in-plane grid, display units 0-1."""
    wob, na, nb = (zoom(f[iz], UP, order=1) for f in fields)
    fine = VOX / UP
    xf = ORIGIN[0] + (np.arange(NX * UP) + 0.5) * fine
    yf = ORIGIN[1] + (np.arange(NY * UP) + 0.5) * fine
    X, Y = np.meshgrid(xf, yf)

    rr = _radius(X, Y, zmm, BRAIN_C, BRAIN_R, 1.0 + 0.028 * wob)
    a = np.full(X.shape, 0.025)
    a[rr <= 1.135] = 0.34                                   # scalp
    a[rr <= 1.078] = 0.86                                   # skull vault
    a[rr <= 1.028] = 0.13                                   # subarachnoid CSF
    brain = rr <= 1.0
    a[brain] = 0.52                                         # cortical grey
    rw = _radius(X, Y, zmm, BRAIN_C,
                 (BRAIN_R[0] * 0.72, BRAIN_R[1] * 0.68, BRAIN_R[2] * 0.78))
    core = brain & (rw <= 1.0)
    a[core] = 0.78                                          # white matter
    a[core] -= 0.05 * np.clip(nb, -1.2, 1.2)[core]          # WM inhomogeneity

    # Gyral folding: a band-pass phase field, confined to an irregular outer
    # ribbon so the deep white matter stays smooth instead of turning marbled.
    ribbon = brain & (rw > 0.80 + 0.16 * np.clip(na, -1.5, 1.5))
    a[(np.sin(3.6 * na + 1.7 * nb) > 0.50) & ribbon] = 0.26  # sulcal CSF
    a[(np.abs(X) < 1.5) & brain] = 0.20                      # interhemispheric

    for sx in (-1.0, 1.0):                                   # lateral ventricle
        v = _radius(X, Y, zmm, (sx * 12.0, -26.0, 16.0), (5.5, 27.0, 14.0))
        a[v <= 1.0] = 0.08
    head = rr <= 1.135
    a[head] *= 1.0 + 0.055 * wob[head]                       # receive bias
    a[brain] += 0.030 * np.clip(nb, -2.0, 2.0)[brain]        # tissue texture
    a = gaussian_filter(a, 1.15) + 0.016 * rng.standard_normal(X.shape)
    return np.clip(a, 0.0, 1.0)


def statistic_map(rng, brain):
    """Voxelwise t statistic: fixed foci plus a smooth null field."""
    x, y, z = _axes_mm()
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    t = np.zeros(X.shape)
    for fx, fy, fz, amp, sig in FOCI:
        d2 = (X - fx) ** 2 + (Y - fy) ** 2 + (Z - fz) ** 2
        t += amp * np.exp(-0.5 * d2 / sig ** 2)
    t += 0.95 * _sm(rng, X.shape, 2.8)
    return t * brain


def surviving_clusters(t):
    """Cluster-extent-corrected mask plus a peak table sorted by extent."""
    x, y, z = _axes_mm()
    lab, n = cc_label(t > T_THRESH)
    keep = np.zeros(t.shape, dtype=bool)
    table = []
    for i in range(1, n + 1):
        m = lab == i
        k = int(m.sum())
        if k < K_EXTENT:
            continue
        keep |= m
        iz, iy, ix = np.unravel_index(np.argmax(np.where(m, t, -np.inf)),
                                      t.shape)
        table.append(dict(k=k, t=float(t[iz, iy, ix]),
                          xyz=(x[ix], y[iy], z[iz])))
    table.sort(key=lambda d: -d["k"])
    return keep, table


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(11)

    x, y, z = _axes_mm()
    Z3, Y3, X3 = np.meshgrid(z, y, x, indexing="ij")
    brain = np.sqrt(((X3 - BRAIN_C[0]) / BRAIN_R[0]) ** 2 +
                    ((Y3 - BRAIN_C[1]) / BRAIN_R[1]) ** 2 +
                    ((Z3 - BRAIN_C[2]) / BRAIN_R[2]) ** 2) <= 0.99

    fields = [_sm(rng, (NZ, NY, NX), 6.0), _sm(rng, (NZ, NY, NX), 3.2),
              _sm(rng, (NZ, NY, NX), 6.5)]
    tvol = statistic_map(rng, brain)
    keep, clusters = surviving_clusters(tvol)
    tshow = np.where(keep, tvol, np.nan)

    # One norm for all twelve tiles (DESIGN_SYSTEM 13.1).
    norm = shared_norm([tvol[keep]], vmin=T_THRESH)
    extent = (ORIGIN[0], ORIGIN[0] + NX * VOX,
              ORIGIN[1], ORIGIN[1] + NY * VOX)

    # ---- inch-accurate montage geometry -----------------------------------
    FW, FH = 7.2, 6.6
    left, top_band, bot_band, cb_res = 0.26, 0.62, 0.92, 1.02
    gx, gy = 0.07, 0.15
    tile_h = (FH - top_band - bot_band - 2 * gy) / 3.0
    tile_w = tile_h * (XLIM[1] - XLIM[0]) / (YLIM[1] - YLIM[0])
    grid_w = 4 * tile_w + 3 * gx
    x_left = left + max((FW - cb_res - left - grid_w) / 2.0, 0.0)
    grid_h = 3 * tile_h + 2 * gy
    y_bot = bot_band

    fig = plt.figure(figsize=(FW, FH))
    im = None
    for i, zmm in enumerate(SLICES_MM):
        r, c = divmod(i, 4)
        ax = axes_inches(fig, (x_left + c * (tile_w + gx),
                               y_bot + (2 - r) * (tile_h + gy),
                               tile_w, tile_h))
        image_axes(ax, color="#cccccc", lw=0.5)
        iz = int(round((zmm - ORIGIN[2]) / VOX - 0.5))
        ax.imshow(slice_anatomy(zmm, fields, iz, rng), cmap="gray",
                  vmin=0.0, vmax=1.0, origin="lower", extent=extent,
                  interpolation="bilinear", zorder=1)
        img = ax.imshow(tshow[iz], cmap=ACT_CMAP, norm=norm, origin="lower",
                        extent=extent, interpolation="bilinear", zorder=2)
        im = im or img
        if np.nanmax(tvol[iz]) > T_THRESH:
            ax.contour(x, y, np.where(keep[iz], tvol[iz], 0.0),
                       levels=[T_THRESH], colors="white", linewidths=0.35,
                       alpha=0.75, zorder=3)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        corner_stamp(ax, f"z = {zmm:+d} mm", loc="lower left", fontsize=6.6)
        if i == 0:
            corner_stamp(ax, "L", loc="upper left", fontsize=6.6,
                         weight="bold")
            corner_stamp(ax, "R", loc="upper right", fontsize=6.6,
                         weight="bold")
            add_scale_bar(ax, 50.0, label="50 mm", fontsize=6.2,
                          loc=(0.055, 0.125), height=0.020)

    # ---- one shared colour bar, pinned beside the whole montage -----------
    span = fig.add_axes([(x_left + grid_w) / FW, y_bot / FH,
                         0.02 / FW, grid_h / FH])
    span.set_axis_off()
    span.patch.set_visible(False)
    tmax = float(np.nanmax(tshow))
    ticks = np.arange(3.5, tmax + 0.01, 0.5)
    cb = safe_colorbar(fig, im, span, label="$t$-value (-)", width=0.12,
                       pad=0.10, shrink=0.74, ticks=ticks)
    cb.ax.text(-0.2, -0.035, f"height $t$ > {T_THRESH:.2f}\n"
                             f"extent $k_E \\geq$ {K_EXTENT} voxels\n"
                             f"cluster $p_{{FWE}}$ < 0.05",
               transform=cb.ax.transAxes, ha="left", va="top", fontsize=6.2,
               color=COLORS["muted"], linespacing=1.45)

    fig.text(x_left / FW, (FH - 0.40) / FH,
             "Group SPM, random-effects contrast task > rest, "
             f"{VOX:.0f} mm isotropic, MNI axial levels",
             fontsize=6.8, color=COLORS["sub"], ha="left", va="center")

    rows = ["Largest clusters (peak $t$, MNI coordinate, extent):"]
    for j, cl in enumerate(clusters[:3], 1):
        cx, cy, cz = cl["xyz"]
        rows.append(f"    {j}.   $t$ = {cl['t']:.2f}      "
                    f"({cx:+.0f}, {cy:+.0f}, {cz:+.0f}) mm      "
                    f"$k_E$ = {cl['k']} voxels")
    fig.text(x_left / FW, 0.235 / FH, "\n".join(rows), fontsize=6.5,
             color=COLORS["ink"], ha="left", va="bottom", linespacing=1.55)
    fig.text(x_left / FW, 0.085 / FH,
             f"Warm overlay shows suprathreshold voxels only "
             f"({T_THRESH:.2f} \u2264 $t$ \u2264 {tmax:.2f}); the grey-scale "
             "anatomy uses one display window across the montage.",
             fontsize=6.5, color=COLORS["muted"], ha="left", va="bottom")

    savefig(fig, "534_fmri_activation_map.png")


if __name__ == "__main__":
    main()
