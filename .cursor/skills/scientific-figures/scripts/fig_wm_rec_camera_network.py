"""Recovered camera network around a sparse structure-from-motion cloud.

The large 3-D view shows the sparse point cloud of a small courtyard together
with the ring of recovered camera poses; every frustum is tinted by the mean
reprojection error of the bundle-adjusted view.  The subordinate right column
carries the top-down camera layout and the error distribution, both keyed to
the same colour ramp.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pubstyle import set_pub_style, savefig

SEED = 20261522
ELEV, AZIM = 24.0, -62.0
LIM = 15.5
ZTOP = 10.0
# same ramp as 1523, which encodes the same quantity in the same family
ERR_CMAP = plt.get_cmap("cividis")
HEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "sfm_height", ["#c6ccd2", "#8d99a4", "#4f6070", "#1e2c38"])


# ------------------------------------------------------------------ geometry
def face_samples(rng, c, s, yaw, n, jitter=0.035):
    """Points sampled on the four walls plus the roof of an oriented box."""
    h = np.asarray(s, float) / 2.0
    area = np.array([s[1] * s[2], s[1] * s[2], s[0] * s[2], s[0] * s[2],
                     s[0] * s[1] * 0.8])
    counts = rng.multinomial(n, area / area.sum())
    specs = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1)]
    out = []
    for (axis, sgn), k in zip(specs, counts):
        q = rng.uniform(-1, 1, (k, 3)) * h
        q[:, axis] = sgn * h[axis]
        out.append(q)
    q = np.vstack(out)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                  [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return q @ R.T + np.asarray(c) + rng.normal(0.0, jitter, q.shape)


def build_cloud(rng):
    """Sparse cloud of a courtyard: three blocks, a plinth, trees, ground."""
    blocks = [((0.0, 8.2, 4.2), (13.0, 4.6, 8.4), 0.00),
              ((-8.4, -1.0, 3.3), (4.2, 12.5, 6.6), 0.06),
              ((7.6, -3.6, 2.7), (5.4, 7.4, 5.4), 0.22)]
    pts = [face_samples(rng, c, s, y, n)
           for (c, s, y), n in zip(blocks, (3400, 2400, 2000))]
    pts.append(face_samples(rng, (0.4, -0.6, 1.35), (1.9, 1.9, 2.7), 0.4, 340))

    for cx, cy in [(-3.6, 3.9), (4.6, 4.2), (-2.4, -6.4), (6.0, 6.4)]:
        d = rng.normal(0.0, 1.0, (240, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d *= rng.uniform(0.72, 1.0, (240, 1)) ** 0.5 * 1.85
        pts.append(d + np.array([cx, cy, 4.1]))
        pts.append(face_samples(rng, (cx, cy, 1.2), (0.4, 0.4, 2.4), 0.0, 40))

    # ground returns thin out with distance from the courtyard centre
    ng = 4200
    a = rng.uniform(0, 2 * np.pi, ng)
    r = 13.8 * np.sqrt(rng.uniform(0.004, 1.0, ng))
    keep = rng.random(ng) < np.clip(1.15 - 0.055 * r, 0.25, 1.0)
    g = np.column_stack([r[keep] * np.cos(a[keep]), r[keep] * np.sin(a[keep]),
                         rng.normal(0.0, 0.045, keep.sum())])
    pts.append(g)
    return np.vstack(pts), blocks


def block_wire(ax, blocks, color="#d8dcdf", lw=0.4, z=4):
    """Barely-there wireframe so the block silhouettes stay readable."""
    k = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    for c, s, yaw in blocks:
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                      [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        v = (k * (np.asarray(s) / 2.0)) @ R.T + np.asarray(c)
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                     (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            ax.plot(*zip(v[i], v[j]), color=color, lw=lw, zorder=z)


def look_at(C, target, up=np.array([0.0, 0.0, 1.0])):
    f = target - C
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return np.column_stack([r, np.cross(r, f), f])


def draw_frustum(ax, C, R, color, w=1.05, h=0.76, d=1.85, lw=0.8, z=30):
    v = C + np.array([[w, h, d], [-w, h, d], [-w, -h, d], [w, -h, d]]) @ R.T
    for p in v:
        ax.plot(*zip(C, p), color=color, lw=lw * 0.8, zorder=z, alpha=0.95)
    ring = np.vstack([v, v[:1]])
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, lw=lw, zorder=z)
    pc = Poly3DCollection([v], facecolor=color, edgecolor="none", alpha=0.16)
    pc.set_zorder(z - 1)
    ax.add_collection3d(pc)
    ax.plot([C[0]], [C[1]], [C[2]], marker="o", ms=2.4, color=color,
            mec="white", mew=0.35, zorder=z + 1)


def pane_style(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#dcdcdc")
        axis.pane.set_linewidth(0.5)
        axis._axinfo["grid"].update(color="#ededed", linewidth=0.4)


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    X, blocks = build_cloud(rng)
    inbox = ((np.abs(X[:, 0]) <= LIM) & (np.abs(X[:, 1]) <= LIM)
             & (X[:, 2] >= -0.4) & (X[:, 2] <= ZTOP))
    X = X[inbox]

    n_cam = 16
    phi = np.linspace(0, 2 * np.pi, n_cam, endpoint=False) + 0.22
    rad = 13.8 + rng.normal(0.0, 0.45, n_cam)
    hgt = 4.6 + 2.5 * np.sin(2.0 * phi + 0.6) + rng.normal(0.0, 0.35, n_cam)
    cams = np.column_stack([rad * np.cos(phi), rad * np.sin(phi), hgt])
    target = np.array([0.0, 1.0, 3.4])
    Rs = [look_at(C, target) for C in cams]

    # per-view mean reprojection error, worst on the two grazing viewpoints
    err = 0.42 + 0.30 * np.abs(np.sin(1.5 * phi + 0.9)) \
        + np.abs(rng.normal(0.0, 0.09, n_cam))
    err[[4, 11]] += 0.46
    vmin, vmax = 0.30, 1.45
    norm = Normalize(vmin, vmax)

    fig = plt.figure(figsize=(11.0, 5.9))

    # ---- main 3-D panel ---------------------------------------------------
    ax = fig.add_axes([-0.048, -0.020, 0.790, 1.050], projection="3d")
    ax.computed_zorder = False
    ax.patch.set_alpha(0.0)
    block_wire(ax, blocks)

    ca, sa = np.cos(np.deg2rad(AZIM)), np.sin(np.deg2rad(AZIM))
    depth = X[:, 0] * ca + X[:, 1] * sa
    order = np.argsort(depth)                # far first, near painted on top
    Xo, do = X[order], depth[order]
    dn = (do - do.min()) / np.ptp(do)        # 0 = far, 1 = near the viewer
    hn = np.clip(Xo[:, 2] / (ZTOP * 0.86), 0.0, 1.0)
    rgba = HEIGHT_CMAP(hn)
    rgba[:, 3] = np.clip(0.34 + 0.46 * dn, 0.30, 0.86)
    ax.scatter(Xo[:, 0], Xo[:, 1], Xo[:, 2], s=np.clip(1.6 + 2.1 * dn, 1.4,
               3.8), c=rgba, marker=".", linewidths=0, depthshade=False,
               zorder=5, rasterized=True)

    # capture trajectory through the recovered centres
    loop = np.vstack([cams, cams[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#9aa3ab", lw=0.7,
            ls=(0, (3.4, 2.2)), zorder=24)
    for C, R, e in zip(cams, Rs, err):
        draw_frustum(ax, C, R, ERR_CMAP(norm(e)), w=0.80, h=0.58, d=1.45)

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(-0.4, ZTOP)
    ax.set_box_aspect((1.0, 1.0, 0.52))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("X (m)", labelpad=-5, fontsize=7.2)
    ax.set_ylabel("Y (m)", labelpad=-5, fontsize=7.2)
    ax.set_zlabel("Z (m)", labelpad=-7, fontsize=7.2)
    ax.set_xticks([-14, 0, 14])
    ax.set_yticks([-14, 0, 14])
    ax.set_zticks([0, 5, 10])
    ax.tick_params(axis="both", pad=-3.5, labelsize=6.4)
    ax.tick_params(axis="z", pad=-1.5, labelsize=6.4)
    pane_style(ax)


    handles = [Line2D([], [], color="#4f6070", marker=".", ls="none", ms=5,
                      label="Triangulated point"),
               Line2D([], [], color="#3b6f8f", lw=0.9,
                      label="Camera frustum"),
               Line2D([], [], color="#9aa3ab", lw=0.7, ls=(0, (3.4, 2.2)),
                      label="Capture trajectory")]
    leg = fig.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.030, 0.300), frameon=True,
                     framealpha=0.94, edgecolor="#cccccc", fontsize=6.8,
                     handlelength=1.8, labelspacing=0.32, borderpad=0.42)
    leg.get_frame().set_linewidth(0.55)

    # ---- subordinate right column -----------------------------------------
    fig.patches.append(plt.Rectangle(
        (0.700, 0.035), 0.292, 0.905, transform=fig.transFigure,
        facecolor="#f4f4f5", edgecolor="#dedede", linewidth=0.6, zorder=-1))

    axb = fig.add_axes([0.714, 0.480, 0.264, 0.410])
    axb.scatter(X[:, 0], X[:, 1], s=0.6,
                c=HEIGHT_CMAP(np.clip(X[:, 2] / (ZTOP * 0.86), 0.0, 1.0)),
                marker=".", linewidths=0, alpha=0.75, rasterized=True,
                zorder=2)
    axb.plot(loop[:, 0], loop[:, 1], color="#aeb5bc", lw=0.7,
             ls=(0, (3.4, 2.2)), zorder=3)
    for C, R, e in zip(cams, Rs, err):
        f = R[:, 2] * 2.6
        axb.plot([C[0], C[0] + f[0]], [C[1], C[1] + f[1]],
                 color=ERR_CMAP(norm(e)), lw=0.8, zorder=4)
        axb.plot(C[0], C[1], marker="o", ms=3.4, mfc=ERR_CMAP(norm(e)),
                 mec="white", mew=0.5, ls="none", zorder=5)
    axb.set_xlim(-LIM - 1.0, LIM + 1.0)
    axb.set_ylim(-LIM - 1.0, LIM + 1.0)
    axb.set_aspect("equal")
    axb.set_xticks([])
    axb.set_yticks([])
    for sp in axb.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_color("#a8a8a8")
    axb.set_title("Top-down camera layout", fontsize=7.6, pad=3.0)
    axb.plot([-14.5, -4.5], [-15.0, -15.0], color="#555555", lw=1.0, zorder=6)
    axb.text(-9.5, -14.2, "10 m", fontsize=6.0, color="#555555", ha="center",
             va="bottom", zorder=6)

    axh = fig.add_axes([0.744, 0.170, 0.212, 0.215])
    e_all = np.concatenate([err[:, None] + rng.normal(0.0, 0.085, (n_cam, 45))
                            ], axis=1).ravel()
    e_all = np.clip(e_all, vmin, vmax)
    edges = np.linspace(vmin, vmax, 24)
    cnt, _ = np.histogram(e_all, bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    axh.bar(ctr, cnt, width=(edges[1] - edges[0]) * 0.92,
            color=ERR_CMAP(norm(ctr)), edgecolor="white", linewidth=0.3)
    rms = float(np.sqrt(np.mean(e_all ** 2)))
    axh.axvline(rms, color="#b23a48", lw=1.1, ls=(0, (3, 1.6)), zorder=4)
    axh.text(rms + 0.04, axh.get_ylim()[1] * 0.92, f"RMS {rms:.2f} px",
             fontsize=6.2, color="#b23a48", va="top")
    axh.set_xlim(vmin, vmax)
    axh.set_xlabel("reprojection error (px)", fontsize=6.8, labelpad=1.2)
    axh.set_ylabel("observations", fontsize=6.8, labelpad=1.5)
    axh.tick_params(labelsize=6.2, length=2.0)
    axh.set_title("Residual distribution", fontsize=7.6, pad=3.0)
    axh.grid(axis="y", lw=0.4, color="#e2e2e2")
    axh.set_axisbelow(True)

    cax = fig.add_axes([0.744, 0.083, 0.212, 0.019])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=ERR_CMAP), cax=cax,
                      orientation="horizontal", ticks=[0.4, 0.8, 1.2])
    cb.set_label("per-view mean reprojection error (px)", fontsize=6.6,
                 color="#1a1a1a", labelpad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.2, color="#333333")

    savefig(fig, "1117_wm_rec_camera_network.png")


if __name__ == "__main__":
    main()
