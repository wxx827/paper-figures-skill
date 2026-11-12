"""Hyperspectral data cube rendered as a pseudo-3D block.

Main panel: the cube itself. The top face is a false-colour composite of the
scene, the two exposed side faces are wavelength-by-space slices, and the
left edge carries the wavelength ruler so the reader can see that the third
axis of the block really is spectral.

Auxiliary panels (58 % of the main panel): reflectance spectra of four
representative pixels marked on the top face, and the PCA explained-variance
spectrum of the same cube.

The cube is synthesised by mixing four endmember spectra with smooth
abundance fields, so spatial structure and spectral structure are consistent
with one another instead of being independent noise.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, style_axes, get_cmap
from figkit import axes_inches, panel_labels_inches, halo_text

NX = NY = 96                 # spatial samples
NB = 120                     # spectral bands
WL0, WL1 = 400.0, 2500.0     # nm
GSD = 30.0                   # m ground sample distance

ENDMEMBERS = [
    ("Canopy", "#2e7d5b", [
        (400, 0.03), (550, 0.09), (670, 0.03), (720, 0.24), (780, 0.46),
        (1000, 0.47), (1300, 0.40), (1450, 0.17), (1650, 0.29), (1950, 0.07),
        (2200, 0.16), (2500, 0.08)]),
    ("Dry soil", "#c0603f", [
        (400, 0.08), (600, 0.15), (800, 0.23), (1000, 0.28), (1300, 0.31),
        (1450, 0.23), (1650, 0.35), (1950, 0.22), (2200, 0.28), (2500, 0.26)]),
    ("Water", "#3b5b92", [
        (400, 0.05), (500, 0.05), (600, 0.03), (700, 0.01), (900, 0.006),
        (1300, 0.004), (1650, 0.003), (1950, 0.002), (2500, 0.002)]),
    ("Roof", "#b23a48", [
        (400, 0.12), (600, 0.16), (800, 0.19), (1000, 0.21), (1300, 0.23),
        (1450, 0.18), (1650, 0.25), (1950, 0.18), (2200, 0.23), (2500, 0.20)]),
]

VAPOUR = [(1338, 1452), (1798, 1962)]

# Pseudo-3D block geometry, in the main panel's data units.
BW, BH = 3.05, 2.25          # top-face width, cube height (wavelength)
DX, DY = 0.98, 0.74          # depth vector of the top face


def field(rng, sigma):
    z = gaussian_filter(rng.normal(size=(NY, NX)), sigma, mode="reflect")
    return (z - z.mean()) / (z.std() + 1e-12)


def build_cube(rng, wl):
    """Mix four endmembers with smooth abundance fields."""
    specs = np.stack([
        np.clip(PchipInterpolator(np.array(a)[:, 0], np.array(a)[:, 1])(wl),
                0.0, 1.0)
        for _, _, a in ENDMEMBERS])

    yy, xx = np.mgrid[0:NY, 0:NX] / float(NX)
    logits = np.stack([
        1.9 * field(rng, 9.0) + 1.1 - 1.4 * yy,
        1.7 * field(rng, 7.0) + 0.9 * yy,
        4.2 * np.exp(-((xx - 0.30 - 0.10 * np.sin(6.0 * yy)) / 0.035) ** 2)
        - 1.2 + 0.6 * field(rng, 5.0),
        2.0 * field(rng, 3.5) - 0.9,
    ])
    abund = np.exp(logits - logits.max(axis=0, keepdims=True))
    abund /= abund.sum(axis=0, keepdims=True)

    shade = 1.0 + 0.10 * field(rng, 12.0)
    cube = np.tensordot(specs, abund, axes=(0, 0)) * shade[None, :, :]
    cube += 0.004 * rng.normal(size=cube.shape)
    return np.clip(cube, 0.0, 1.2), abund, specs


def stretch(a, lo=2.0, hi=98.0):
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / (p1 - p0 + 1e-12), 0.0, 1.0)


def face_transform(ax, a, b, c, d, tx, ty):
    """Affine mapping of the unit square onto one face of the block."""
    m = np.array([[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]])
    return mtransforms.Affine2D(matrix=m) + ax.transData


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(757)

    wl = np.linspace(WL0, WL1, NB)
    cube, abund, specs = build_cube(rng, wl)

    # ---- false-colour composite (NIR / red / green) -----------------------
    idx = [int(np.argmin(np.abs(wl - v))) for v in (860.0, 650.0, 550.0)]
    # One stretch shared by all three channels.  Stretching each band to its
    # own percentiles throws away the relative brightness that makes a
    # false-colour composite readable (vegetation red, water dark, soil tan)
    # and turns the scene into saturated magenta and cyan.
    rgb = stretch(np.dstack([cube[i] for i in idx]), 1.0, 99.0) ** 0.78
    lum = rgb.mean(axis=2, keepdims=True)
    rgb = np.clip(0.82 * rgb + 0.18 * lum, 0.0, 1.0)

    # ---- slices exposed on the two visible side faces ---------------------
    row, col = int(0.22 * NY), int(0.78 * NX)
    front = cube[:, row, :]          # bands x along-track columns
    right = cube[:, :, col]          # bands x across-track rows

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax = axes_inches(fig, (0.42, 0.72, 5.90, 4.78))
    ax_spec = axes_inches(fig, (7.02, 3.66, 2.98, 1.84))
    ax_pca = axes_inches(fig, (7.02, 0.98, 2.98, 1.72))

    # ---- (a) the cube -----------------------------------------------------
    cmap_slice = get_cmap("cb_sequential_blue")
    vmin, vmax = float(np.percentile(cube, 1)), float(np.percentile(cube, 99))

    ax.imshow(rgb, extent=(0, 1, 0, 1), origin="lower", aspect="auto",
              interpolation="bilinear", zorder=3,
              transform=face_transform(ax, BW, DX, 0.0, DY, 0.0, 0.0))
    im = ax.imshow(front, extent=(0, 1, 0, 1), origin="upper", aspect="auto",
                   interpolation="bilinear", cmap=cmap_slice,
                   vmin=vmin, vmax=vmax, zorder=3,
                   transform=face_transform(ax, BW, 0.0, 0.0, BH, 0.0, -BH))
    ax.imshow(right, extent=(0, 1, 0, 1), origin="upper", aspect="auto",
              interpolation="bilinear", cmap=cmap_slice,
              vmin=vmin, vmax=vmax, zorder=3,
              transform=face_transform(ax, DX, 0.0, DY, BH, BW, -BH))

    A = (0.0, 0.0)
    B = (BW, 0.0)
    C = (BW + DX, DY)
    D = (DX, DY)
    for p, q in ((A, B), (B, C), (C, D), (D, A),
                 (A, (0.0, -BH)), (B, (BW, -BH)), (C, (BW + DX, DY - BH)),
                 ((0.0, -BH), (BW, -BH)), ((BW, -BH), (BW + DX, DY - BH))):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#333333", lw=0.9,
                zorder=5, solid_capstyle="round")

    # wavelength ruler down the front-left edge
    for w in (500, 1000, 1500, 2000, 2500):
        yv = -BH * (w - WL0) / (WL1 - WL0)
        ax.plot([-0.055, 0.0], [yv, yv], color="#333333", lw=0.8, zorder=6)
        ax.text(-0.10, yv, "%d" % w, ha="right", va="center", fontsize=6.3,
                color="#333333")
    ax.text(-0.62, -BH / 2.0, "Wavelength (nm)", rotation=90, ha="center",
            va="center", fontsize=7.6, color="#1a1a1a")

    # spatial extent callouts
    ax.annotate("", xy=(0.0, DY + 0.30), xytext=(BW, DY + 0.30),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#555555"))
    ax.text(BW / 2.0, DY + 0.36, "%d px = %.2f km" % (NX, NX * GSD / 1000.0),
            ha="center", va="bottom", fontsize=6.6, color="#555555")
    ax.annotate("", xy=(BW + DX + 0.20, DY - 0.03),
                xytext=(BW + 0.20, -0.03),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#555555"))
    ax.text(BW + DX / 2.0 + 0.42, DY / 2.0 - 0.16, "%d px" % NY,
            ha="left", va="center", fontsize=6.6, color="#555555",
            rotation=37)

    halo_text(ax, 0.10, DY * 0.55 + 0.04,
              "false colour  R 860 / G 650 / B 550 nm", fontsize=6.4,
              ha="left", va="center", zorder=8)
    ax.text(BW + DX + 0.28, -BH + 0.06,
            "%d bands\n%.1f nm sampling" % (NB, (WL1 - WL0) / (NB - 1)),
            ha="left", va="bottom", fontsize=6.6, color="#555555",
            linespacing=1.35)

    # marker pixels, one per endmember, drawn on the sheared top face
    picks = []
    for e, (name, col, _) in enumerate(ENDMEMBERS):
        iy, ix = np.unravel_index(int(np.argmax(abund[e])), (NY, NX))
        picks.append((name, col, iy, ix))
        u, v = (ix + 0.5) / NX, 1.0 - (iy + 0.5) / NY
        px, py = BW * u + DX * v, DY * v
        ax.plot([px], [py], marker="o", ms=4.4, mfc=col, mec="white",
                mew=0.9, zorder=7)
        halo_text(ax, px + 0.07, py + 0.05, "P%d" % (e + 1), fontsize=6.3,
                  fontweight="bold", ha="left", va="bottom", zorder=8)

    cax = ax.inset_axes([0.02, 0.035, 0.26, 0.030])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("slice reflectance (-)", fontsize=6.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.2)

    ax.set_xlim(-0.95, BW + DX + 1.45)
    ax.set_ylim(-BH - 1.05, DY + 0.78)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Reflectance cube: %d x %d pixels x %d bands"
                 % (NX, NY, NB), fontsize=9.0, fontweight="bold", pad=2.0)

    # ---- (b) representative pixel spectra ---------------------------------
    for lo, hi in VAPOUR:
        ax_spec.axvspan(lo, hi, facecolor="#cccccc", alpha=0.6, lw=0, zorder=0)
    for k, (name, col, iy, ix) in enumerate(picks):
        ax_spec.plot(wl, 100.0 * cube[:, iy, ix], color=col, lw=1.15,
                     zorder=3, label="P%d %s" % (k + 1, name.lower()))
    ax_spec.set_xlim(WL0, WL1)
    ax_spec.set_ylim(0, 100.0 * float(cube.max()) * 1.10)
    ax_spec.set_xlabel("Wavelength (nm)", fontsize=7.4, labelpad=1.0)
    ax_spec.set_ylabel("Reflectance (%)", fontsize=7.4)
    ax_spec.set_title("Spectra of the four marked pixels", fontsize=8.5,
                      fontweight="bold")
    ax_spec.tick_params(labelsize=6.6)
    style_axes(ax_spec)
    ax_spec.grid(axis="y", lw=0.5, alpha=0.45)
    ax_spec.legend(loc="upper right", fontsize=6.4, frameon=False,
                   labelspacing=0.24, handlelength=1.3)

    # ---- (c) PCA explained variance ---------------------------------------
    X = cube.reshape(NB, -1).T
    X = X[rng.choice(X.shape[0], 4000, replace=False)]
    X = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    evr = 100.0 * sv ** 2 / np.sum(sv ** 2)
    k = 8
    xs = np.arange(1, k + 1)
    ax_pca.bar(xs, evr[:k], width=0.62, color="#5f9dc6", edgecolor="white",
               linewidth=0.4, zorder=3)
    ax_pca.set_yscale("log")
    ax_pca.set_ylim(max(evr[k - 1] * 0.35, 1e-3), 190.0)
    ax_pca.set_xticks(xs)
    ax_pca.set_xlabel("Principal component index (-)", fontsize=7.4,
                      labelpad=1.0)
    ax_pca.set_ylabel("Explained variance (%)", fontsize=7.4)
    ax_pca.tick_params(labelsize=6.6)
    style_axes(ax_pca)
    ax_pca.grid(axis="y", lw=0.5, alpha=0.45)

    ax_cum = ax_pca.twinx()
    ax_cum.plot(xs, np.cumsum(evr[:k]), color="#b23a48", lw=1.3, marker="o",
                ms=3.0, zorder=4)
    ax_cum.set_ylim(0.0, 106.0)
    ax_cum.set_ylabel("Cumulative (%)", fontsize=7.4, color="#b23a48")
    ax_cum.tick_params(labelsize=6.6, colors="#b23a48")
    ax_cum.spines["top"].set_visible(False)
    ax_cum.spines["right"].set_color("#b23a48")
    ax_pca.set_title("Spectral dimensionality is low", fontsize=8.5,
                     fontweight="bold")
    n95 = int(np.searchsorted(np.cumsum(evr), 95.0) + 1)
    ax_pca.text(0.98, 0.30, "%d PCs reach 95 %% of variance" % n95,
                transform=ax_pca.transAxes, ha="right", va="bottom",
                fontsize=6.3, color="#555555")

    panel_labels_inches(fig, [ax, ax_spec, ax_pca], dx=-0.10, dy=0.14)


    savefig(fig, "757_rs_hyperspectral_cube.png")


if __name__ == "__main__":
    main()
