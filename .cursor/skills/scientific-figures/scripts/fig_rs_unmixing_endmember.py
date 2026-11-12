"""Linear spectral unmixing of a three-endmember scene.

Main panel: the abundance map rendered as a barycentric colour blend, so a
pixel's hue *is* its mixture. A ternary key in the corner decodes the blend
and the three purest pixels are marked where they sit in the scene.

Auxiliary panels (58 % of the main panel): the three endmember spectra with
one mixed pixel and its reconstruction, the simplex in the plane of the first
two principal components with the data convex hull, and the distribution of
per-pixel reconstruction residuals.

Abundances are drawn first and the cube is synthesised from them, so the
recovered abundances can be checked against the truth - the recovery error
printed on the figure is that comparison, not a fixed string.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, panel_labels_inches, image_axes, halo_text

N = 180                       # pixels per side
GSD = 20.0                    # m
NB = 90                       # bands
WL0, WL1 = 400.0, 2500.0

ENDMEMBERS = [
    ("Vegetation", "#2e7d5b", [
        (400, 0.03), (550, 0.09), (670, 0.03), (720, 0.24), (780, 0.47),
        (1000, 0.48), (1300, 0.40), (1450, 0.17), (1650, 0.29), (1950, 0.07),
        (2200, 0.16), (2500, 0.08)]),
    ("Bare soil", "#c0603f", [
        (400, 0.08), (600, 0.15), (800, 0.24), (1000, 0.29), (1300, 0.32),
        (1450, 0.23), (1650, 0.36), (1950, 0.22), (2200, 0.29), (2500, 0.27)]),
    ("Water", "#3b5b92", [
        (400, 0.05), (500, 0.05), (600, 0.03), (700, 0.012), (900, 0.007),
        (1300, 0.004), (1650, 0.003), (1950, 0.002), (2500, 0.002)]),
]
NOISE = 0.006                 # additive sensor noise, reflectance units


def field(rng, sigma):
    z = gaussian_filter(rng.normal(size=(N, N)), sigma, mode="reflect")
    return (z - z.mean()) / (z.std() + 1e-12)


def rgb_of(hexcol):
    return np.array([int(hexcol[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])


def build(rng, wl):
    specs = np.stack([
        np.clip(PchipInterpolator(np.array(a)[:, 0], np.array(a)[:, 1])(wl),
                0.0, 1.0) for _, _, a in ENDMEMBERS])       # (3, NB)

    yy, xx = np.mgrid[0:N, 0:N] / float(N)
    logits = np.stack([
        2.6 * field(rng, 10.0) + 1.0 - 1.1 * yy,
        2.2 * field(rng, 8.0) + 0.6 * yy + 0.3,
        5.0 * np.exp(-((xx - 0.26 - 0.13 * np.sin(5.5 * yy)) / 0.055) ** 2)
        + 3.4 * np.exp(-(((xx - 0.74) / 0.11) ** 2 + ((yy - 0.24) / 0.07) ** 2))
        - 1.9,
    ])
    # mild sharpening: enough that a few pure pixels reach the simplex
    # vertices, not so much that every pixel collapses onto one endmember
    abund = np.exp(0.85 * (logits - logits.max(axis=0, keepdims=True)))
    abund /= abund.sum(axis=0, keepdims=True)

    cube = np.tensordot(specs, abund, axes=(0, 0))
    cube += NOISE * rng.normal(size=cube.shape)
    return specs, abund, cube


def unmix(cube, specs):
    """Least-squares abundances with a sum-to-one constraint by augmentation."""
    nb = specs.shape[1]
    E = np.vstack([specs.T, 40.0 * np.ones((1, specs.shape[0]))])
    Y = np.vstack([cube.reshape(nb, -1), 40.0 * np.ones((1, N * N))])
    a, *_ = np.linalg.lstsq(E, Y, rcond=None)
    a = np.clip(a, 0.0, None)
    a /= a.sum(axis=0, keepdims=True) + 1e-12
    resid = Y[:nb] - specs.T @ a
    rmse = np.sqrt((resid ** 2).mean(axis=0))
    return a.reshape(3, N, N), rmse.reshape(N, N)


def blend(abund):
    cols = np.stack([rgb_of(c) for _, c, _ in ENDMEMBERS])      # (3, 3)
    img = np.tensordot(abund, cols, axes=(0, 0))
    return np.clip(img * 1.30, 0.0, 1.0)


def ternary_key(ax_parent, loc):
    """Barycentric colour triangle that decodes the abundance blend."""
    cols = np.stack([rgb_of(c) for _, c, _ in ENDMEMBERS])
    v = np.array([[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]])
    n = 190
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    d = ((v[1, 1] - v[2, 1]) * (v[0, 0] - v[2, 0])
         + (v[2, 0] - v[1, 0]) * (v[0, 1] - v[2, 1]))
    l0 = ((v[1, 1] - v[2, 1]) * (gx - v[2, 0])
          + (v[2, 0] - v[1, 0]) * (gy - v[2, 1])) / d
    l1 = ((v[2, 1] - v[0, 1]) * (gx - v[2, 0])
          + (v[0, 0] - v[2, 0]) * (gy - v[2, 1])) / d
    l2 = 1.0 - l0 - l1
    lam = np.stack([l0, l1, l2])
    inside = (lam >= -0.004).all(axis=0)
    rgba = np.zeros((n, n, 4))
    rgba[..., :3] = np.clip(np.tensordot(lam, cols, axes=(0, 0)) * 1.30,
                            0.0, 1.0)
    rgba[..., 3] = inside.astype(float)

    axk = ax_parent.inset_axes(loc)
    axk.imshow(rgba, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
               interpolation="bilinear")
    axk.set_xticks([])
    axk.set_yticks([])
    axk.set_zorder(7)
    axk.patch.set_visible(False)
    for s in axk.spines.values():
        s.set_visible(False)
    for (vx, vy), (name, _, _), (hx, hy, ha, va) in zip(
            v, ENDMEMBERS, [(0.5, 1.03, "center", "bottom"),
                            (-0.02, -0.06, "right", "top"),
                            (1.02, -0.06, "left", "top")]):
        halo_text(axk, hx, hy, name.split()[0], fontsize=6.1, ha=ha, va=va,
                  zorder=9, fontweight="bold")
    return axk


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(760)

    wl = np.linspace(WL0, WL1, NB)
    specs, truth, cube = build(rng, wl)
    est, rmse = unmix(cube, specs)
    rec_err = float(np.abs(est - truth).mean())

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax = axes_inches(fig, (0.52, 1.18, 4.24, 4.24))
    ax_spec = axes_inches(fig, (5.42, 3.16, 2.50, 2.26))
    ax_sim = axes_inches(fig, (8.28, 3.16, 1.98, 2.26))
    ax_res = axes_inches(fig, (5.42, 0.95, 4.84, 1.52))

    # ---- (a) abundance blend ---------------------------------------------
    km = N * GSD / 1000.0
    ax.imshow(blend(est), extent=[0, km, 0, km], origin="lower",
              interpolation="bilinear")
    image_axes(ax, color="#333333", lw=0.7)
    ax.set_title("Fractional abundance rendered as a colour mixture",
                 fontsize=9.0, fontweight="bold")

    pures = []
    for e, (name, col, _) in enumerate(ENDMEMBERS):
        iy, ix = np.unravel_index(int(np.argmax(est[e])), (N, N))
        pures.append((e, name, col, iy, ix))
        px, py = (ix + 0.5) * GSD / 1000.0, (iy + 0.5) * GSD / 1000.0
        ax.plot([px], [py], marker="o", ms=5.0, mfc="none", mec="white",
                mew=1.6, zorder=6)
        ax.plot([px], [py], marker="o", ms=5.0, mfc="none", mec="#1a1a1a",
                mew=0.8, zorder=7)
        ha, dx = ("left", 0.10) if px < 0.55 * km else ("right", -0.10)
        va, dy = ("bottom", 0.08) if py < 0.90 * km else ("top", -0.08)
        halo_text(ax, px + dx, py + dy, "EM%d %s (%.0f %%)"
                  % (e + 1, name.lower(), 100.0 * est[e, iy, ix]),
                  fontsize=6.3, ha=ha, va=va, zorder=8, clip_on=True)

    # mixed pixel used in panel (b): the most evenly mixed pixel in the scene
    iy_m, ix_m = np.unravel_index(int(np.argmax(-(est ** 2).sum(axis=0))),
                                  (N, N))
    ax.plot([(ix_m + 0.5) * GSD / 1000.0], [(iy_m + 0.5) * GSD / 1000.0],
            marker="s", ms=5.0, mfc="none", mec="#1a1a1a", mew=1.1, zorder=7)
    mx, my = (ix_m + 0.5) * GSD / 1000.0, (iy_m + 0.5) * GSD / 1000.0
    ha, dx = ("left", 0.10) if mx < 0.55 * km else ("right", -0.10)
    halo_text(ax, mx + dx, my + 0.08, "M mixed pixel", fontsize=6.3, ha=ha,
              va="bottom", zorder=8, clip_on=True)

    ternary_key(ax, [0.045, 0.085, 0.235, 0.235])
    ax.text(0.985, 0.02, "%.0f m pixels, %.1f x %.1f km"
            % (GSD, km, km), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.3, color="white",
            bbox=dict(boxstyle="round,pad=0.24", fc="#3a3a3a", ec="none",
                      alpha=0.72))

    # ---- (b) endmember spectra --------------------------------------------
    for e, (name, col, _) in enumerate(ENDMEMBERS):
        ax_spec.plot(wl, 100.0 * specs[e], color=col, lw=1.4, zorder=3,
                     label="EM%d %s" % (e + 1, name.lower()))
    mix = cube[:, iy_m, ix_m]
    fit = specs.T @ est[:, iy_m, ix_m]
    ax_spec.plot(wl, 100.0 * mix, color="#555555", lw=0.0, marker="o", ms=1.6,
                 zorder=4, label="M observed")
    ax_spec.plot(wl, 100.0 * fit, color="#1a1a1a", lw=1.0, ls=(0, (4, 2)),
                 zorder=5, label="M reconstructed")
    ax_spec.set_xlim(WL0, WL1)
    ax_spec.set_ylim(0, 100.0 * specs.max() * 1.18)
    ax_spec.set_xlabel("Wavelength (nm)", fontsize=7.4, labelpad=1.0)
    ax_spec.set_ylabel("Reflectance (%)", fontsize=7.4)
    ax_spec.set_title("Endmembers and one mixed pixel", fontsize=8.5,
                      fontweight="bold")
    ax_spec.tick_params(labelsize=6.6)
    style_axes(ax_spec)
    ax_spec.grid(axis="y", lw=0.5, alpha=0.45)
    ax_spec.legend(loc="upper right", fontsize=6.3, frameon=False,
                   labelspacing=0.22, handlelength=1.3)
    ax_spec.text(0.02, 0.98, "M = %s %% EM1/EM2/EM3"
                 % " / ".join("%.0f" % (100.0 * est[e, iy_m, ix_m])
                              for e in range(3)),
                 transform=ax_spec.transAxes, ha="left", va="top",
                 fontsize=6.2, color="#555555")

    # ---- (c) simplex in PC space ------------------------------------------
    flat = cube.reshape(NB, -1).T
    mu = flat.mean(axis=0, keepdims=True)
    sub = flat[rng.choice(flat.shape[0], 1400, replace=False)]
    _, _, vt = np.linalg.svd(flat[::7] - mu, full_matrices=False)
    P = vt[:2].T
    pts = (sub - mu) @ P
    vert = (specs - mu) @ P

    a_sub = np.linalg.lstsq(specs.T, sub.T, rcond=None)[0]
    dom = np.argmax(np.clip(a_sub, 0.0, None), axis=0)
    for e, (name, col, _) in enumerate(ENDMEMBERS):
        m = dom == e
        ax_sim.plot(pts[m, 0], pts[m, 1], ls="none", marker="o", ms=1.9,
                    mfc=col, mec="none", alpha=0.40, zorder=2)
    hull = ConvexHull(pts)
    loop = np.append(hull.vertices, hull.vertices[0])
    ax_sim.plot(pts[loop, 0], pts[loop, 1], color="#777777", lw=0.8,
                ls=(0, (4, 2)), zorder=3, label="Data convex hull")
    tri = np.append(np.arange(3), 0)
    ax_sim.plot(vert[tri, 0], vert[tri, 1], color="#1a1a1a", lw=1.1,
                zorder=5, label="Endmember simplex")
    for e, (name, col, _) in enumerate(ENDMEMBERS):
        ax_sim.plot([vert[e, 0]], [vert[e, 1]], marker="o", ms=5.2, mfc=col,
                    mec="#1a1a1a", mew=0.9, zorder=6)
        ax_sim.annotate("EM%d" % (e + 1), (vert[e, 0], vert[e, 1]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6.3, fontweight="bold", color="#1a1a1a")
    ax_sim.set_xlabel("PC 1 (-)", fontsize=7.4, labelpad=1.0)
    ax_sim.set_ylabel("PC 2 (-)", fontsize=7.4)
    ax_sim.set_title("Pixels fill the simplex", fontsize=8.5,
                     fontweight="bold")
    ax_sim.tick_params(labelsize=6.6)
    style_axes(ax_sim)
    ax_sim.legend(loc="lower left", fontsize=6.2, frameon=False,
                  handlelength=1.3, labelspacing=0.22,
                  bbox_to_anchor=(-0.02, -0.02))

    # ---- (d) reconstruction residuals -------------------------------------
    r = 100.0 * rmse.ravel()
    ax_res.hist(r, bins=64, color="#9cc3dd", edgecolor="white",
                linewidth=0.25, zorder=3)
    ax_res.axvline(np.median(r), color="#b23a48", lw=1.2, ls="--", zorder=4)
    ax_res.axvline(100.0 * NOISE, color="#333333", lw=0.9, ls=(0, (3, 2)),
                   zorder=4)
    ax_res.annotate("sensor noise floor %.1f %%" % (100.0 * NOISE),
                    xy=(100.0 * NOISE, ax_res.get_ylim()[1] * 0.80),
                    xytext=(10, 4), textcoords="offset points",
                    fontsize=6.3, color="#333333")
    ax_res.annotate("median %.2f %%" % np.median(r),
                    xy=(np.median(r), ax_res.get_ylim()[1] * 0.52),
                    xytext=(10, 4), textcoords="offset points",
                    fontsize=6.3, color="#b23a48")
    ax_res.set_xlim(0, np.percentile(r, 99.7) * 1.05)
    ax_res.set_xlabel("Per-pixel reconstruction RMSE (% reflectance)",
                      fontsize=7.4, labelpad=1.0)
    ax_res.set_ylabel("Pixels (-)", fontsize=7.4)
    ax_res.set_title("Residuals sit at the noise floor", fontsize=8.5,
                     fontweight="bold")
    ax_res.tick_params(labelsize=6.6)
    style_axes(ax_res)
    ax_res.grid(axis="y", lw=0.5, alpha=0.45)
    ax_res.text(0.015, 0.94, "mean |abundance error| = %.1f %% over "
                "%d pixels" % (100.0 * rec_err, N * N),
                transform=ax_res.transAxes, ha="left", va="top",
                fontsize=6.3, color="#555555")

    panel_labels_inches(fig, [ax, ax_spec, ax_sim, ax_res], dx=-0.20,
                        dy=0.14)


    savefig(fig, "760_rs_unmixing_endmember.png")


if __name__ == "__main__":
    main()
