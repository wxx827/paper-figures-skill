"""Three reconstruction methods compared on three structurally different scenes.

Rows are scenes -- a carved sculpture (dense closed surface), an office corner
(planar point cloud) and a rock outcrop (rough height field) -- so the three
rows exercise genuinely different geometry, sampling density and rendering
mode.  Camera pose, axis range and colour range are shared inside a row and
deliberately differ between rows.  Each cell reports the Chamfer distance
measured on the array it draws, and the column with the lowest mean Chamfer
distance is the one that gets the highlight card.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, get_cmap, savefig

SEED = 20261527
METHODS = ["Voxel TSDF", "Neural SDF", "Ours"]
SCENES = ["Carved sculpture", "Office corner", "Rock outcrop"]
VIEWS = [(20.0, -58.0), (26.0, -72.0), (34.0, -44.0)]
SURF_CMAP = get_cmap("cb_sequential_blue")
ACCENT = "#8c2f2f"


# ------------------------------------------------------------------- scene 1
def sculpture(rng):
    """Closed organic surface: a carved figure with sharp ridges."""
    u = np.linspace(0.04, np.pi - 0.04, 68)
    v = np.linspace(0.0, 2.0 * np.pi, 104)
    U, V = np.meshgrid(u, v, indexing="ij")
    R = (1.0
         + 0.26 * np.abs(np.sin(2.5 * V)) ** 1.6 * np.sin(U)
         - 0.20 * np.exp(-((U - 1.95) / 0.22) ** 2)
         + 0.13 * np.sin(4.0 * U) * np.cos(V + 0.6)
         + 0.07 * np.cos(6.0 * V - 0.4))
    return U, V, R


def sculpture_xyz(U, V, R):
    return np.stack([R * np.sin(U) * np.cos(V), R * np.sin(U) * np.sin(V),
                     1.22 * R * np.cos(U)], axis=-1)


# ------------------------------------------------------------------- scene 2
def office(rng):
    """Planar point cloud plus, per point, the axis its plane is normal to."""
    def plane(n, o, e1, e2):
        a = rng.uniform(0, 1, (n, 1))
        b = rng.uniform(0, 1, (n, 1))
        return np.asarray(o) + a * np.asarray(e1) + b * np.asarray(e2)

    specs = [(6200, (-1.8, -1.4, 0.0), (3.6, 0, 0), (0, 2.8, 0), 2),
             (4000, (-1.8, 1.4, 0.0), (3.6, 0, 0), (0, 0, 2.0), 1),
             (3100, (-1.8, -1.4, 0.0), (0, 2.8, 0), (0, 0, 2.0), 0),
             (1700, (-0.9, 0.15, 0.74), (1.9, 0, 0), (0, 0.75, 0), 2),
             (600, (-0.9, 0.15, 0.0), (1.9, 0, 0), (0, 0, 0.74), 1),
             (800, (-0.15, 0.86, 0.78), (0.85, 0, 0), (0, 0, 0.52), 1),
             (1300, (0.95, -1.35, 0.0), (0.65, 0, 0), (0, 0, 1.15), 1),
             (760, (0.95, -1.35, 1.15), (0.65, 0, 0), (0, 0.55, 0), 2)]
    P = np.vstack([plane(n, o, e1, e2) for n, o, e1, e2, _ in specs])
    axis = np.concatenate([np.full(n, a) for n, _, _, _, a in specs])
    return P, axis


# ------------------------------------------------------------------- scene 3
def outcrop(rng):
    """Rough height field with a ridge line and a talus slope."""
    x = np.linspace(-2.4, 2.4, 78)
    y = np.linspace(-1.8, 1.8, 62)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = (0.62 * np.exp(-((X + 0.5) / 1.05) ** 2 - ((Y - 0.15) / 0.75) ** 2)
         + 0.48 * np.exp(-((X - 1.15) / 0.55) ** 2 - ((Y + 0.6) / 0.5) ** 2)
         + 0.22 * np.sin(2.1 * X + 0.4) * np.exp(-(Y / 1.6) ** 2)
         + 0.10 * np.cos(3.4 * Y - 0.8))
    Z += 0.055 * gaussian_filter(rng.normal(0, 1, Z.shape), 1.1)
    return X, Y, Z


# ---------------------------------------------------------------- degradation
def voxelise(A, step):
    return np.round(A / step) * step


def style3d(ax, elev, azim, xlim, ylim, zlim, box):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect(box)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    # ---- scene 1: surface variants ----------------------------------------
    U1, V1, R1 = sculpture(rng)
    gt1 = sculpture_xyz(U1, V1, R1)
    var1 = [sculpture_xyz(U1, V1, voxelise(R1, 0.085)),
            sculpture_xyz(U1, V1, gaussian_filter(R1, (2.4, 3.0),
                                                  mode=("nearest", "wrap"))),
            sculpture_xyz(U1, V1, R1 + rng.normal(0, 0.012, R1.shape))]

    # ---- scene 2: point-cloud variants ------------------------------------
    gt2, ax2 = office(rng)
    drift = gaussian_filter(rng.normal(0, 1, (gt2.shape[0], 3)), (26, 0))
    # marching cubes on a coarse grid keeps the sampling density but leaves a
    # half-voxel staircase across each plane, so the walls band
    block, amp = 0.24, 0.038
    stepped = gt2.copy()
    for a in (0, 1, 2):
        m = ax2 == a
        u, w = [gt2[m, k] for k in range(3) if k != a]
        parity = (np.floor(u / block) + np.floor(w / block)) % 2
        stepped[m, a] += amp * (2.0 * parity - 1.0)
    var2 = [stepped,
            gt2 + 0.026 * drift / max(drift.std(), 1e-9),
            gt2 + rng.normal(0, 0.013, gt2.shape)]

    # ---- scene 3: height-field variants -----------------------------------
    X3, Y3, Z3 = outcrop(rng)
    var3 = [voxelise(Z3, 0.115),
            gaussian_filter(Z3, 3.2, mode="nearest"),
            Z3 + rng.normal(0, 0.014, Z3.shape)]

    # Chamfer distance, in centimetres, read straight off the drawn arrays
    cd = np.array([
        [np.linalg.norm(v.reshape(-1, 3) - gt1.reshape(-1, 3),
                        axis=1).mean() * 100 for v in var1],
        [np.linalg.norm(v - gt2, axis=1).mean() * 100 for v in var2],
        [np.abs(v - Z3).mean() * 100 for v in var3]])
    best = int(np.argmin(cd.mean(axis=0)))

    fig = plt.figure(figsize=(11.2, 6.4))
    X0, Y0, CW, CH = 0.088, 0.058, 0.298, 0.278

    fig.patches.append(plt.Rectangle(
        (X0 + best * CW - 0.004, Y0 - 0.012), CW + 0.008, 3 * CH + 0.052,
        transform=fig.transFigure, facecolor="#f2f5f8", edgecolor="#d6dfe6",
        linewidth=0.7, zorder=-1))

    for j, name in enumerate(METHODS):
        fig.text(X0 + j * CW + CW / 2, 0.905, name, fontsize=9.4,
                 ha="center", va="bottom",
                 fontweight="bold" if j == best else "normal",
                 color=ACCENT if j == best else "#1a1a1a")
    for i, name in enumerate(SCENES):
        fig.text(0.052, Y0 + (2 - i) * CH + CH / 2, name, fontsize=8.8,
                 ha="center", va="center", rotation=90, fontweight="bold")

    zn1 = plt.Normalize(-1.45, 1.45)
    zn2 = plt.Normalize(-0.25, 2.05)
    zn3 = plt.Normalize(float(Z3.min()) - 0.25, float(Z3.max()) + 0.05)

    for j in range(3):
        # --- row 1: shaded closed surface ----------------------------------
        ax = fig.add_axes([X0 + j * CW - 0.032, Y0 + 2 * CH - 0.034,
                           CW + 0.064, CH + 0.082], projection="3d")
        P = var1[j]
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], rstride=1, cstride=1,
                        facecolors=SURF_CMAP(0.20 + 0.62 * zn1(P[..., 2])),
                        shade=True, edgecolor="none", antialiased=True)
        style3d(ax, *VIEWS[0], (-1.45, 1.45), (-1.45, 1.45), (-1.60, 1.60),
                (1, 1, 1.28))

        # --- row 2: planar point cloud -------------------------------------
        ax = fig.add_axes([X0 + j * CW - 0.032, Y0 + CH - 0.034,
                           CW + 0.064, CH + 0.082], projection="3d")
        Q = var2[j]
        ca, sa = np.cos(np.deg2rad(VIEWS[1][1])), np.sin(np.deg2rad(
            VIEWS[1][1]))
        o = np.argsort(Q[:, 0] * ca + Q[:, 1] * sa)
        ax.scatter(Q[o, 0], Q[o, 1], Q[o, 2], s=2.1,
                   c=SURF_CMAP(0.26 + 0.64 * zn2(Q[o, 2])), marker=".",
                   linewidths=0, depthshade=False, alpha=0.80,
                   rasterized=True)
        style3d(ax, *VIEWS[1], (-1.95, 1.95), (-1.55, 1.55), (0.0, 2.10),
                (1, 0.82, 0.60))

        # --- row 3: rough height field -------------------------------------
        ax = fig.add_axes([X0 + j * CW - 0.032, Y0 - 0.034,
                           CW + 0.064, CH + 0.082], projection="3d")
        Zv = var3[j]
        ax.plot_surface(X3, Y3, Zv, rstride=1, cstride=1,
                        facecolors=SURF_CMAP(0.18 + 0.66 * zn3(Zv)),
                        shade=True, edgecolor="none", antialiased=True)
        style3d(ax, *VIEWS[2], (-2.45, 2.45), (-1.85, 1.85), (-0.30, 1.15),
                (1, 0.78, 0.42))

    for i in range(3):
        for j in range(3):
            fig.text(X0 + j * CW + CW - 0.022, Y0 + (2 - i) * CH + 0.016,
                     f"CD {cd[i, j]:.2f} cm", fontsize=7.4, ha="right",
                     va="bottom", fontweight="bold" if j == best else "normal",
                     color=ACCENT if j == best else "#444444",
                     bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                               edgecolor="#dcdcdc", linewidth=0.5,
                               alpha=0.88))

    savefig(fig, "1122_wm_rec_method_compare.png")


if __name__ == "__main__":
    main()
