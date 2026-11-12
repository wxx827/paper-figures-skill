"""One model decimated four times, with the error that buys each reduction.

The four tiles are nested subsets of a single vertex pool in a fixed shuffled
order, so the coarse meshes really are decimations of the fine one rather than
four independent samplings, and all four share the camera, the coordinate box
and the height colour scale - the only thing that changes across the row is
how many triangles are left.

Both auxiliary panels measure rather than assert.  The error curve evaluates
each decimated mesh against a dense reference surface at the same 4000 probe
points, and the normal deviation is the angle between every triangle's own
normal and the analytic surface normal at its centroid, so the boxes widen for
exactly the meshes whose silhouettes have visibly gone faceted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from pubstyle import set_pub_style, savefig, get_cmap, get_palette
from figkit import shared_norm, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GUIDE = "#808080"
GRID = "#d9d9d9"
ACCENT = "#b23a48"

R_MM = 60.0                  # model radius, millimetres
BOUNDARY_N = 28              # rim vertices, never removed
LEVELS = [640, 230, 84, 36]  # vertices kept, in decimation order
ELEV, AZIM = 26.0, -58.0
N_PROBE = 4000


def height(x, y):
    """Smooth bumpy dome in millimetres; analytic so normals are exact."""
    r2 = (x ** 2 + y ** 2) / R_MM ** 2
    dome = 26.0 * np.exp(-1.6 * r2)
    ridge = 7.5 * np.sin(2.4 * x / R_MM + 0.6) * np.cos(1.9 * y / R_MM)
    lobe = 5.0 * np.exp(-9.0 * ((x / R_MM - 0.42) ** 2 +
                                (y / R_MM + 0.34) ** 2))
    return dome + ridge * np.exp(-0.9 * r2) + lobe


def height_grad(x, y):
    """Analytic gradient, used for the exact surface normal."""
    h = 1e-4 * R_MM
    fx = (height(x + h, y) - height(x - h, y)) / (2.0 * h)
    fy = (height(x, y + h) - height(x, y - h)) / (2.0 * h)
    return fx, fy


def disc_points(n, rng):
    """Sunflower lattice on the disc, jittered - even spacing, no clumps."""
    k = np.arange(1, n + 1)
    r = R_MM * np.sqrt(k / (n + 0.5))
    a = k * np.pi * (3.0 - np.sqrt(5.0))
    x = r * np.cos(a) + rng.normal(0.0, 0.9, n)
    y = r * np.sin(a) + rng.normal(0.0, 0.9, n)
    s = np.hypot(x, y)
    over = s > R_MM
    x[over] *= R_MM / s[over]
    y[over] *= R_MM / s[over]
    return x, y


def ring(n):
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return R_MM * np.cos(a), R_MM * np.sin(a)


def build_pool(rng):
    """One vertex pool in a fixed priority order; level k keeps a prefix."""
    bx, by = ring(BOUNDARY_N)
    ix, iy = disc_points(max(LEVELS) - BOUNDARY_N, rng)
    perm = rng.permutation(len(ix))
    # Rim vertices are never removed, interior ones leave in shuffled order,
    # which is what a decimation queue does to a surface with a boundary.
    return (np.concatenate([bx, ix[perm]]),
            np.concatenate([by, iy[perm]]))


def coarse(px, py, n_keep):
    """Prefix of the pool, triangulated - so the levels are truly nested."""
    x, y = px[:n_keep], py[:n_keep]
    z = height(x, y)
    return mtri.Triangulation(x, y), x, y, z


def normal_deviation(tri, x, y, z):
    """Angle (deg) between each facet normal and the true surface normal."""
    t = tri.triangles
    p = np.stack([x, y, z], axis=1)
    n = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    n *= np.sign(n[:, 2:3])
    cx, cy = x[t].mean(axis=1), y[t].mean(axis=1)
    fx, fy = height_grad(cx, cy)
    m = np.stack([-fx, -fy, np.ones_like(fx)], axis=1)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(np.sum(n * m, axis=1), -1.0, 1.0)))


def main():
    set_pub_style()
    rng = np.random.default_rng(6500)
    px, py = build_pool(rng)

    # Dense probe set for the error metric; the same points for every level.
    qx, qy = disc_points(N_PROBE, np.random.default_rng(6501))
    qx, qy = qx * 0.985, qy * 0.985
    qz = height(qx, qy)

    meshes, faces, rms, hausdorff, devs = [], [], [], [], []
    for n_keep in LEVELS:
        tri, x, y, z = coarse(px, py, n_keep)
        interp = mtri.LinearTriInterpolator(tri, z)
        zi = interp(qx, qy)
        err = np.abs(np.ma.filled(zi, np.nan) - qz)
        err = err[np.isfinite(err)]
        meshes.append((tri, x, y, z))
        faces.append(len(tri.triangles))
        rms.append(float(np.sqrt(np.mean(err ** 2))))
        hausdorff.append(float(np.max(err)))
        devs.append(normal_deviation(tri, x, y, z))

    cmap = get_cmap("cb_sequential_blue")
    norm = shared_norm([m[3] for m in meshes])
    lvl_cols = get_palette("cb_sequential_blue", 7)[3:6] + [ACCENT]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.60],
                          left=0.045, right=0.975, top=0.905, bottom=0.135,
                          hspace=0.34)
    gst = gs[0].subgridspec(1, 4, wspace=0.02)
    gsb = gs[1].subgridspec(1, 2, width_ratios=[1.0, 0.86], wspace=0.24)

    tiles = []
    for k, (tri, x, y, z) in enumerate(meshes):
        ax = fig.add_subplot(gst[0, k], projection="3d")
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_visible(False)
        ax.set_proj_type("ortho")
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_box_aspect((1.0, 1.0, 0.62), zoom=1.42)
        ax.set_xlim(-R_MM, R_MM)
        ax.set_ylim(-R_MM, R_MM)
        ax.set_zlim(-4.0, 40.0)

        surf = ax.plot_trisurf(tri, z, cmap=cmap, norm=norm, linewidth=0.28,
                               edgecolor="white", antialiased=True,
                               shade=False)
        surf.set_alpha(1.0)

        ax.text2D(0.5, 0.955, f"{faces[k]} triangles", transform=ax.transAxes,
                  ha="center", va="top", fontsize=8.5, fontweight="bold",
                  color=INK)
        ax.text2D(0.5, 0.090,
                  f"RMS {rms[k]:.2f} mm   max {hausdorff[k]:.2f} mm",
                  transform=ax.transAxes, ha="center", va="center",
                  fontsize=6.8, color=SUB)
        ax.text2D(0.5, 0.030,
                  f"{len(x)} vertices, median normal error "
                  f"{np.median(devs[k]):.1f} deg",
                  transform=ax.transAxes, ha="center", va="center",
                  fontsize=6.4, color=MUTED)
        tiles.append(ax)

    # ---- aux 1: error against face count, log-log --------------------------
    axe = fig.add_subplot(gsb[0, 0])
    f = np.array(faces, dtype=float)
    axe.plot(f, hausdorff, color=ACCENT, lw=1.5, marker="o", ms=4.2,
             label="one-sided Hausdorff error")
    axe.plot(f, rms, color="#3b5b92", lw=1.5, marker="s", ms=4.0,
             label="RMS error")
    fit = np.polyfit(np.log10(f), np.log10(rms), 1)
    slope = fit[0]
    axe.plot(f, 10.0 ** np.polyval(fit, np.log10(f)), color=GUIDE, lw=0.9,
             ls=(0, (3.2, 2.2)), zorder=1)
    axe.text(0.03, 0.05, f"grey: RMS power fit, slope {slope:.2f} per decade",
             transform=axe.transAxes, ha="left", va="bottom", fontsize=6.4,
             color=GUIDE)
    for k in range(len(f)):
        axe.annotate(f"{faces[k]}", xy=(f[k], hausdorff[k]),
                     xytext=(0.0, 7.0), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6.2,
                     color=lvl_cols[k])
    axe.set_xscale("log")
    axe.set_yscale("log")
    axe.set_xlabel("triangles retained (-)", fontsize=8.0)
    axe.set_ylabel("geometric error (mm)", fontsize=8.0)
    axe.set_title("Error grows about linearly as faces are halved",
                  fontsize=8.5, fontweight="bold", color=INK, pad=4)
    axe.tick_params(labelsize=6.6)
    axe.grid(True, which="both", color=GRID, lw=0.5)
    axe.legend(loc="upper right", frameon=False, fontsize=6.8)

    # ---- aux 2: normal deviation distribution ------------------------------
    axn = fig.add_subplot(gsb[0, 1])
    bp = axn.boxplot(devs, positions=np.arange(len(devs)), widths=0.56,
                     showfliers=False, patch_artist=True,
                     medianprops=dict(color=INK, lw=1.0),
                     whiskerprops=dict(color=RULE, lw=0.8),
                     capprops=dict(color=RULE, lw=0.8))
    for patch, col in zip(bp["boxes"], lvl_cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_edgecolor(RULE)
        patch.set_linewidth(0.7)
    for k, d in enumerate(devs):
        axn.scatter([k], [np.mean(d)], s=13, color=lvl_cols[k], zorder=5,
                    edgecolors="white", linewidths=0.5)
    axn.set_xticks(np.arange(len(devs)))
    axn.set_xticklabels([f"{n}" for n in faces], fontsize=6.6)
    axn.set_xlabel("triangles retained (-)", fontsize=8.0)
    axn.set_ylabel("normal deviation (deg)", fontsize=8.0)
    axn.set_title("Facet normals drift from the true surface",
                  fontsize=8.5, fontweight="bold", color=INK, pad=4)
    axn.tick_params(labelsize=6.6)
    axn.grid(True, axis="y", color=GRID, lw=0.5)
    axn.text(0.02, 0.96, "dots: mean, boxes: quartiles",
             transform=axn.transAxes, ha="left", va="top", fontsize=6.4,
             color=MUTED)


    panel_labels_inches(fig, tiles, dx=0.34, dy=-0.30)
    panel_labels_inches(fig, [axe, axn], labels=["(e)", "(f)"], dx=-0.04,
                        dy=0.26)
    savefig(fig, "650_cg_mesh_simplification.png")


if __name__ == "__main__":
    main()
