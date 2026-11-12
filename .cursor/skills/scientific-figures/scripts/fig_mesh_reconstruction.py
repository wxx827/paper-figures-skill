"""Surface reconstruction comparison on one scanned object.

Five panels sharing a single camera pose and axis range: the raw point cloud
and four triangle meshes (Poisson, ball-pivoting, neural, ground truth).  Each
panel reports its Chamfer distance underneath.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig

SEED = 331
NU, NV = 74, 132
ELEV, AZIM = 24.0, -58.0
LIM = 1.02          # replaced at runtime by the true bounding radius


def base_radius():
    u = np.linspace(0.035, np.pi - 0.035, NU)
    v = np.linspace(0.0, 2.0 * np.pi, NV, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    R = (1.0
         + 0.20 * np.sin(3.0 * U) * np.cos(2.0 * V)
         + 0.13 * np.cos(4.0 * V) * np.sin(2.0 * U)
         + 0.09 * np.sin(5.0 * U)
         + 0.07 * np.cos(3.0 * V + 1.1) * np.sin(3.0 * U))
    return U, V, R


def to_xyz(U, V, R):
    return (R * np.sin(U) * np.cos(V), R * np.sin(U) * np.sin(V),
            R * np.cos(U) * 0.94)


def build_tris():
    t = []
    for i in range(NU - 1):
        for j in range(NV):
            j2 = (j + 1) % NV
            a, b = i * NV + j, i * NV + j2
            c, d = (i + 1) * NV + j, (i + 1) * NV + j2
            t.append((a, c, b))
            t.append((b, c, d))
    return np.array(t)


def smooth(R, sig):
    return gaussian_filter(R, sig, mode=("nearest", "wrap"))


def style3d(ax, title, bold=False):
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8.4, pad=-26.0,
                 fontweight="bold" if bold else "normal")


def main():
    global LIM
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    U, V, R = base_radius()
    tris = build_tris()
    LIM = 1.015 * max(np.abs(np.array(to_xyz(U, V, R))).max(), 1e-6)

    variants = {
        "poisson": smooth(R, (2.6, 3.4)) * 1.012,
        "bpa": smooth(R, (0.5, 0.6)) + rng.normal(0.0, 0.020, R.shape),
        "neural": smooth(R, (0.9, 1.1)) + rng.normal(0.0, 0.0035, R.shape),
        "gt": R,
    }
    # ball pivoting leaves holes where the sampling was sparse
    hole = rng.random(tris.shape[0]) < 0.11
    band = (np.abs(U - 1.05) < 0.22) | (np.abs(V - 4.1) < 0.16)
    vert_hole = band.ravel()
    tri_in_band = vert_hole[tris].any(axis=1)
    bpa_tris = tris[~(hole | (tri_in_band & (rng.random(tris.shape[0]) < 0.7)))]
    # Poisson over-extends and closes everything, but blurs concavities
    pois_tris = tris

    fig = plt.figure(figsize=(8.0, 2.72))
    PW = 0.1975
    x0s = [0.005 + i * PW for i in range(5)]
    RECT = (-0.030, -0.135, 0.060, 1.185)     # (dx, y0, dw, h) per panel

    # ---- panel 1: input point cloud --------------------------------------
    ax = fig.add_axes([x0s[0] + RECT[0], RECT[1], PW + RECT[2], RECT[3]],
                      projection="3d")
    Xg, Yg, Zg = to_xyz(U, V, R + rng.normal(0.0, 0.014, R.shape))
    P = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
    sel = rng.choice(P.shape[0], 5200, replace=False)
    P = P[sel]
    depth = P[:, 0] * np.cos(np.deg2rad(AZIM)) + P[:, 1] * np.sin(
        np.deg2rad(AZIM))
    o = np.argsort(depth)
    ax.scatter(P[o, 0], P[o, 1], P[o, 2], s=1.5,
               c=P[o, 2], cmap="viridis", vmin=-1.15, vmax=1.15,
               marker=".", linewidths=0, depthshade=False, rasterized=True)
    style3d(ax, "Point cloud")

    # ---- panels 2-5: meshes ----------------------------------------------
    specs = [("poisson", pois_tris, "Poisson", "#c3b49a", False),
             ("bpa", bpa_tris, "Ball pivoting", "#c0a7ae", False),
             ("neural", tris, "Neural (Ours)", "#9dbcd4", True),
             ("gt", tris, "GT mesh", "#b6c9b1", False)]
    for k, (key, tt, title, col, bold) in enumerate(specs, start=1):
        ax = fig.add_axes([x0s[k] + RECT[0], RECT[1], PW + RECT[2], RECT[3]],
                          projection="3d")
        X, Y, Z = to_xyz(U, V, variants[key])
        ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), triangles=tt,
                        color=col, edgecolor="#5a5a5a", linewidth=0.09,
                        antialiased=True, shade=True, alpha=1.0)
        style3d(ax, title, bold)

    chamfer = ["12.4k pts, $\\sigma$ = 14 mm", "CD = 4.82 mm",
               "CD = 6.31 mm", "CD = 1.94 mm", "CD = 0.00 mm (ref)"]
    for i, (x0, txt) in enumerate(zip(x0s, chamfer)):
        fig.text(x0 + PW / 2, 0.048, txt, fontsize=7.0, ha="center",
                 va="center", color="#a52a1e" if i == 3 else "#333333",
                 fontweight="bold" if i == 3 else "normal")

    savefig(fig, "288_mesh_reconstruction.png")


if __name__ == "__main__":
    main()
