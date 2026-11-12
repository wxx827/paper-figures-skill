"""Multi-view geometry: camera poses, triangulated structure, epipolar lines.

Large 3-D panel with five posed camera frustums, the triangulated feature
cloud and viewing rays; the narrow right column carries two subordinate
panels -- an epipolar-geometry schematic and the reprojection-error
distribution.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrowPatch

from pubstyle import set_pub_style, savefig

SEED = 20260812
CAMCOL = ["#1f6fb4", "#2e8b57", "#c9701f", "#8e4bbf", "#b23a48"]


# ------------------------------------------------------------------ structure
def build_structure(rng):
    """Sparse feature cloud sampled from a little building plus a car."""
    pts, kind = [], []

    def face_samples(c, s, yaw, n, jitter=0.02):
        h = np.asarray(s) / 2.0
        out = []
        area = [s[1] * s[2], s[1] * s[2], s[0] * s[2], s[0] * s[2],
                s[0] * s[1]]
        area = np.array(area) / np.sum(area)
        counts = rng.multinomial(n, area)
        specs = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1)]
        for (axis, sgn), k in zip(specs, counts):
            q = rng.uniform(-1, 1, (k, 3)) * h
            q[:, axis] = sgn * h[axis]
            out.append(q)
        q = np.vstack(out)
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        return q @ R.T + np.asarray(c) + rng.normal(0, jitter, (q.shape[0], 3))

    shell = ((0.0, 0.0, 3.9), (7.6, 6.0, 7.8), 0.12)
    pts.append(face_samples(shell[0], shell[1], shell[2], 190))
    kind += ["structure"] * pts[-1].shape[0]
    pts.append(face_samples((6.8, -4.2, 0.80), (4.5, 1.95, 1.60), -0.30, 46))
    kind += ["car"] * pts[-1].shape[0]
    pts.append(face_samples((-5.6, 4.4, 1.7), (0.5, 0.5, 3.4), 0.0, 18))
    kind += ["pole"] * pts[-1].shape[0]
    P = np.vstack(pts)
    return P, np.array(kind), shell


def look_at(C, target, up=np.array([0.0, 0.0, 1.0])):
    f = target - C
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.column_stack([r, u, f])          # world <- camera


def frustum(C, R, w=1.15, h=0.86, d=2.0):
    k = np.array([[w, h, d], [-w, h, d], [-w, -h, d], [w, -h, d]])
    return C + k @ R.T


def draw_frustum(ax, C, R, color, z=30, lw=1.15, label=None):
    v = frustum(C, R)
    for p in v:
        ax.plot(*zip(C, p), color=color, lw=lw * 0.8, zorder=z, alpha=0.95)
    ring = np.vstack([v, v[:1]])
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, lw=lw, zorder=z)
    ax.add_collection3d(_poly([v], color, 0.13, z - 1))
    ax.plot([C[0]], [C[1]], [C[2]], marker="o", ms=3.0, color=color,
            zorder=z + 1)
    # small up-vector tick so the roll of each pose is readable
    top = 0.5 * (v[0] + v[1])
    ax.plot(*zip(top, top + R[:, 1] * 0.55), color=color, lw=lw, zorder=z)
    if label:
        ax.text(C[0], C[1], C[2] + 1.45, label, fontsize=6.8, color=color,
                ha="center", va="bottom", fontweight="bold", zorder=z + 2)


def _wire(ax, c, s, yaw):
    """Faint wireframe of the observed building so the shape reads."""
    h = np.asarray(s) / 2.0
    k = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float) * h
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    v = k @ R.T + np.asarray(c)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*zip(v[i], v[j]), color="#b8b8b8", lw=0.5, zorder=3)


def _poly(verts, color, alpha, z):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    pc = Poly3DCollection(verts, facecolor=color, edgecolor="none",
                          alpha=alpha)
    pc.set_zorder(z)
    return pc


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    X, kind, shell = build_structure(rng)
    centroid = np.array([0.0, 0.0, 3.6])

    # positions chosen so the five poses stay separated *in projection*
    cams = [np.array(p) for p in [(-11.0, -1.3, 4.6), (-9.5, 4.7, 7.8),
                                  (-6.5, 10.4, 5.2), (10.0, -6.6, 4.4),
                                  (9.5, 3.6, 6.8)]]
    czo = [30, 31, 4, 33, 34]            # C3 sits behind the structure
    Rs = [look_at(C, centroid) for C in cams]

    LIM = 12.5
    # mplot3d does not clip scatter against the axis limits, so features that
    # fall outside the box would be drawn floating beside it
    inbox = ((np.abs(X[:, 0]) <= LIM) & (np.abs(X[:, 1]) <= LIM)
             & (X[:, 2] >= 0.0) & (X[:, 2] <= 10.0))
    X, kind = X[inbox], kind[inbox]

    fig = plt.figure(figsize=(7.7, 4.35))
    ax = fig.add_axes([-0.040, -0.100, 0.755, 1.200], projection="3d")
    ax.computed_zorder = False
    # a 3-D axes keeps an opaque background patch that would white out the
    # grey backing plate it overlaps
    ax.patch.set_alpha(0.0)

    # ground grid for spatial reference
    for g in np.arange(-LIM, LIM + 0.1, 2.5):
        ax.plot([g, g], [-LIM, LIM], [0, 0], color="#e4e4e4", lw=0.4, zorder=1)
        ax.plot([-LIM, LIM], [g, g], [0, 0], color="#e4e4e4", lw=0.4, zorder=1)
    _wire(ax, *shell)

    # rays from the two anchor cameras through a subset of features
    sel = rng.choice(X.shape[0], 30, replace=False)
    for ci, alpha in ((1, 0.34), (3, 0.28)):
        C = cams[ci]
        for i in sel:
            ax.plot([C[0], X[i, 0]], [C[1], X[i, 1]], [C[2], X[i, 2]],
                    color=CAMCOL[ci], lw=0.35, alpha=alpha,
                    ls=(0, (2.6, 2.2)), zorder=6)

    order = {"structure": ("#2b2b2b", 6.5), "car": ("#3b5b92", 7.5),
             "pole": ("#4f8f4f", 7.0)}
    for k, (col, s) in order.items():
        m = kind == k
        ax.scatter(X[m, 0], X[m, 1], X[m, 2], s=s, c=col, marker=".",
                   linewidths=0, depthshade=False, zorder=20, alpha=0.9)
    # ground-plane shadow of the structure, for depth cueing
    ax.scatter(X[:, 0], X[:, 1], np.zeros(X.shape[0]), s=2.0, c="#c9c9c9",
               marker=".", linewidths=0, depthshade=False, zorder=2)

    for i, (C, R) in enumerate(zip(cams, Rs)):
        draw_frustum(ax, C, R, CAMCOL[i], z=czo[i], label=f"C$_{i+1}$")

    # baseline between the two anchor views
    ax.plot(*zip(cams[1], cams[3]), color="#666666", lw=0.9,
            ls=(0, (4, 2.2)), zorder=29)
    mid = 0.5 * (cams[1] + cams[3])
    ax.text(mid[0], mid[1], mid[2] + 0.55, "baseline", fontsize=6.5,
            color="#555555", ha="center", va="bottom", zorder=32)

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_zlim(0, 10)
    ax.set_box_aspect((1.0, 1.0, 0.46))
    ax.view_init(elev=26, azim=-58)
    ax.set_xlabel("X (m)", labelpad=-5, fontsize=7.2)
    ax.set_ylabel("Y (m)", labelpad=-5, fontsize=7.2)
    ax.set_zlabel("Z (m)", labelpad=-7, fontsize=7.2)
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    ax.set_zticks([0, 5, 10])
    ax.tick_params(axis="both", pad=-3.5, labelsize=6.4)
    ax.tick_params(axis="z", pad=-2.0, labelsize=6.4)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor("#d6d6d6")
        a.pane.set_linewidth(0.5)
        a._axinfo["grid"].update(color="#ebebeb", linewidth=0.4)


    handles = [Line2D([], [], color="#2b2b2b", marker=".", ls="none", ms=5,
                      label="Triangulated point"),
               Line2D([], [], color="#888888", lw=0.6, ls=(0, (2.6, 2.2)),
                      label="Viewing ray"),
               Line2D([], [], color="#666666", lw=0.9, ls=(0, (4, 2.2)),
                      label="Baseline C$_2$C$_4$")]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=min(len(handles), 5), frameon=False)
    leg.get_frame().set_linewidth(0.55)


    # ---------------- subordinate right column ----------------------------
    fig.patches.append(plt.Rectangle(
        (0.700, 0.040), 0.293, 0.895, transform=fig.transFigure,
        facecolor="#f4f4f5", edgecolor="#dedede", linewidth=0.6, zorder=-1))

    axe = fig.add_axes([0.712, 0.560, 0.272, 0.300])
    _epipolar(axe, rng)

    axr = fig.add_axes([0.749, 0.130, 0.212, 0.245])
    _errhist(axr, rng)

    savefig(fig, "286_multiview_geometry.png")


def _epipolar(ax, rng):
    """Two image planes, matched points and the pencil of epipolar lines."""
    W, Hh, GAP = 1.00, 0.80, 0.20
    ax.set_xlim(-0.02, 2 * W + GAP + 0.02)
    ax.set_ylim(-0.06, (2 * W + GAP + 0.04) / 1.66 - 0.06)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Epipolar constraint", fontsize=7.4, pad=2.0)

    rects = []
    for x0, lab, col in ((0.0, "view $I$", CAMCOL[1]),
                         (W + GAP, "view $I'$", CAMCOL[3])):
        r = plt.Rectangle((x0, 0.0), W, Hh, facecolor="white", edgecolor=col,
                          linewidth=0.9, zorder=2)
        ax.add_patch(r)
        rects.append(r)
        ax.text(x0 + W / 2, Hh + 0.035, lab, fontsize=6.6, ha="center",
                va="bottom", color=col)

    pts = np.array([[0.26, 0.60], [0.48, 0.34], [0.74, 0.54], [0.62, 0.16],
                    [0.36, 0.22]])
    ep = np.array([W + GAP + 0.13, 0.42])       # epipole inside view I'
    for i, p in enumerate(pts):
        ax.plot(p[0], p[1], marker="o", ms=3.0, mfc=CAMCOL[1], mec="white",
                mew=0.45, zorder=6)
        ax.text(p[0] + 0.035, p[1] + 0.02, f"$x_{i+1}$", fontsize=6.0,
                color="#333333", zorder=6)
        q = np.array([W + GAP + 0.10 + 0.86 * p[0],
                      p[1] + 0.07 * (p[0] - 0.5)])
        d = q - ep
        d /= np.linalg.norm(d)
        seg = np.array([ep - d * 0.40, ep + d * 2.40])
        ln, = ax.plot(seg[:, 0], seg[:, 1], color="#b23a48", lw=0.65,
                      alpha=0.9, zorder=4)
        ln.set_clip_path(rects[1])
        ax.plot(q[0], q[1], marker="o", ms=3.0, mfc=CAMCOL[3], mec="white",
                mew=0.45, zorder=6)
    ax.plot(ep[0], ep[1], marker="o", ms=3.4, mfc="#b23a48", mec="white",
            mew=0.5, zorder=7)
    ax.text(ep[0], ep[1] - 0.06, "$e'$", fontsize=6.2, color="#b23a48",
            ha="center", va="top", zorder=7)
    ax.text(2 * W + GAP - 0.02, -0.03, "$l' = Fx$", fontsize=6.4,
            color="#b23a48", ha="right", va="top", zorder=7)
    ax.add_patch(FancyArrowPatch((W + 0.025, 0.44), (W + GAP - 0.025, 0.44),
                                 arrowstyle="-|>", mutation_scale=7.0,
                                 linewidth=1.3, color="#777777"))
    ax.text(W + GAP / 2, 0.475, "$F$", fontsize=6.6, color="#555555",
            ha="center", va="bottom")


def _errhist(ax, rng):
    e = np.abs(rng.normal(0, 0.36, 4000)) + np.abs(rng.normal(0, 0.22, 4000))
    e = e[e < 1.8]
    ax.hist(e, bins=26, color="#7fa8cc", edgecolor="white", linewidth=0.35)
    rms = float(np.sqrt(np.mean(e ** 2)))
    ax.axvline(rms, color="#b23a48", lw=1.1, ls=(0, (3, 1.6)))
    ax.text(rms + 0.05, ax.get_ylim()[1] * 0.86, f"RMS {rms:.2f} px",
            fontsize=6.2, color="#b23a48", va="top")
    ax.set_xlabel("reprojection error (px)", fontsize=6.6, labelpad=1.2)
    ax.set_ylabel("tracks", fontsize=6.6, labelpad=1.5)
    ax.tick_params(labelsize=6.0, length=2.0)
    ax.set_title("Bundle-adjustment residuals", fontsize=7.4, pad=3.0)
    ax.grid(axis="y", lw=0.4, color="#e0e0e0")
    ax.set_axisbelow(True)


if __name__ == "__main__":
    main()
