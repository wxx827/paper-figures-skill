"""Progressive TSDF fusion of an indoor scene.

Five snapshots of the same volumetric reconstruction, integrated from 10, 50,
200, 800 and all 1 400 RGB-D frames.  Every panel shares one camera pose, one
axis range and one colour range, so the only thing that changes across the row
is how much of the room has been carved out.  The strip underneath tracks
surface completeness against the number of fused frames.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig

SEED = 20261524
ELEV, AZIM = 26.0, -58.0
SNAPS = [10, 50, 200, 800, 1400]
N_FRAMES = 1400
MESH_CMAP = get_cmap("cb_sequential_blue")
XL, YL, ZL = (-3.15, 3.15), (-2.35, 2.35), (0.0, 2.75)


# ------------------------------------------------------------------ geometry
def quad_grid(p0, e1, e2, n1, n2, rng, rough=0.0):
    """Triangulated parallelogram patch with optional surface roughness."""
    u = np.linspace(0.0, 1.0, n1 + 1)
    v = np.linspace(0.0, 1.0, n2 + 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    p0, e1, e2 = map(np.asarray, (p0, e1, e2))
    pts = p0 + U[..., None] * e1 + V[..., None] * e2
    if rough:
        nrm = np.cross(e1, e2)
        nrm = nrm / np.linalg.norm(nrm)
        pts = pts + nrm * rng.normal(0.0, rough, U.shape)[..., None]
    tris = []
    for i in range(n1):
        for j in range(n2):
            a = i * (n2 + 1) + j
            b, c, d = a + 1, a + (n2 + 1), a + (n2 + 2)
            tris += [(a, c, b), (b, c, d)]
    return pts.reshape(-1, 3), np.array(tris, dtype=int)


def box_faces(rng, c, s, n=4, rough=0.004):
    """Four sides plus the top of an axis-aligned box."""
    cx, cy, cz = c
    sx, sy, sz = s
    specs = [((cx - sx / 2, cy - sy / 2, 0.0), (sx, 0, 0), (0, 0, sz)),
             ((cx - sx / 2, cy + sy / 2, 0.0), (sx, 0, 0), (0, 0, sz)),
             ((cx - sx / 2, cy - sy / 2, 0.0), (0, sy, 0), (0, 0, sz)),
             ((cx + sx / 2, cy - sy / 2, 0.0), (0, sy, 0), (0, 0, sz)),
             ((cx - sx / 2, cy - sy / 2, sz), (sx, 0, 0), (0, sy, 0))]
    return [quad_grid(p, e1, e2, n, n, rng, rough) for p, e1, e2 in specs]


def build_room(rng):
    parts = [
        quad_grid((-3.0, -2.2, 0.0), (6.0, 0, 0), (0, 4.4, 0), 26, 20, rng,
                  0.004),                                        # floor
        quad_grid((-3.0, 2.2, 0.0), (6.0, 0, 0), (0, 0, 2.6), 26, 12, rng,
                  0.004),                                        # back wall
        quad_grid((-3.0, -2.2, 0.0), (0, 4.4, 0), (0, 0, 2.6), 20, 12, rng,
                  0.004),                                        # left wall
        quad_grid((3.0, -2.2, 0.0), (0, 4.4, 0), (0, 0, 2.6), 20, 12, rng,
                  0.004),                                        # right wall
    ]
    parts += box_faces(rng, (0.95, -0.55, 0.0), (1.75, 0.95, 0.74), 5)
    parts += box_faces(rng, (-1.75, 1.10, 0.0), (1.55, 0.85, 0.52), 5)
    parts += box_faces(rng, (-2.35, -1.35, 0.0), (0.55, 0.55, 1.15), 4)

    verts, tris, off = [], [], 0
    for v, t in parts:
        verts.append(v)
        tris.append(t + off)
        off += v.shape[0]
    return np.vstack(verts), np.vstack(tris)


def observation_frame(V, T, rng):
    """Frame index at which each triangle first enters the TSDF volume."""
    ctr = V[T].mean(axis=1)
    t = np.linspace(0.0, 1.0, 240)
    # hand-held sweep: start near the table, spiral out to the walls
    path = np.column_stack([
        1.30 * np.cos(6.4 * t) * (0.25 + 0.75 * t),
        1.10 * np.sin(6.4 * t) * (0.25 + 0.75 * t) - 0.2,
        1.35 + 0.18 * np.sin(9.0 * t)])
    d = np.linalg.norm(ctr[:, None, :] - path[None, :, :], axis=2)
    seen = d < 2.45
    first = np.where(seen.any(axis=1), seen.argmax(axis=1),
                     path.shape[0] - 1).astype(float)
    # ragged fusion front: neighbouring voxels do not fuse in lock step
    first += rng.normal(0.0, 11.0, first.size)
    rank = np.argsort(np.argsort(first)) / (first.size - 1.0)
    return np.clip(N_FRAMES * rank ** 3, 1.0, N_FRAMES)


def style3d(ax):
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_zlim(*ZL)
    ax.set_box_aspect((1.0, 0.74, 0.52))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    V, T = build_room(rng)
    tframe = observation_frame(V, T, rng)

    zc = V[T][:, :, 2].mean(axis=1)
    vmin, vmax = -1.55, 2.72                    # shared colour range

    fig = plt.figure(figsize=(11.2, 5.7))
    PW = 0.196
    x0s = [0.004 + i * PW for i in range(5)]

    grow = np.array([(tframe <= f).mean() for f in SNAPS]) * 100.0
    for k, (x0, f) in enumerate(zip(x0s, SNAPS)):
        ax = fig.add_axes([x0 - 0.018, 0.320, PW + 0.036, 0.590],
                          projection="3d")
        m = tframe <= f
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=T[m],
                        cmap=MESH_CMAP, norm=plt.Normalize(vmin, vmax),
                        array=zc[m], edgecolor="#4a5a68", linewidth=0.055,
                        antialiased=True, shade=False, alpha=1.0)
        style3d(ax)
        lab = f"{f} frames" if f < N_FRAMES else f"all {N_FRAMES} frames"
        fig.text(x0 + PW / 2, 0.360, lab, fontsize=8.4, ha="center",
                 va="top", fontweight="bold",
                 color="#8c2f2f" if k == 4 else "#1a1a1a")
        fig.text(x0 + PW / 2, 0.318, f"{grow[k]:.1f} % complete",
                 fontsize=7.4, ha="center", va="top", color="#555555")


    # ---- completeness curve ------------------------------------------------
    axc = fig.add_axes([0.085, 0.085, 0.830, 0.160])
    ff = np.unique(np.round(np.logspace(0, np.log10(N_FRAMES), 220)))
    cc = np.array([(tframe <= f).mean() for f in ff]) * 100.0
    axc.plot(ff, cc, color="#2e74a8", lw=1.6, zorder=3)
    axc.fill_between(ff, 0, cc, color="#cfe1ee", alpha=0.7, zorder=2)
    for k, (f, g) in enumerate(zip(SNAPS, grow)):
        axc.plot([f, f], [0, g], color="#9aa3ab", lw=0.6, ls=(0, (2.4, 1.8)),
                 zorder=4)
        axc.plot(f, g, marker="o", ms=4.0, mfc="white",
                 mec="#8c2f2f" if k == 4 else "#2e74a8", mew=1.1, ls="none",
                 zorder=5)
    axc.set_xscale("log")
    axc.set_xlim(8, N_FRAMES * 1.15)
    axc.set_ylim(0, 104)
    axc.set_xticks([10, 50, 200, 800, 1400])
    axc.set_xticklabels(["10", "50", "200", "800", "1400"])
    axc.set_yticks([0, 50, 100])
    axc.set_xlabel("fused RGB-D frames", fontsize=7.2, labelpad=1.5)
    axc.set_ylabel("complete (%)", fontsize=7.2, labelpad=2.0)
    axc.tick_params(labelsize=6.6, length=2.2)
    axc.grid(axis="y", lw=0.4, color="#e6e6e6")
    axc.set_axisbelow(True)

    savefig(fig, "1119_wm_rec_tsdf_progress.png")


if __name__ == "__main__":
    main()
