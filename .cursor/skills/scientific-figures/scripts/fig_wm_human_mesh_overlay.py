"""Figure 1162 - Human mesh recovery: input, overlay and a rotated view.

Four poses, each shown three ways: the capture-volume frame that went in, the
recovered surface drawn back onto it, and the same surface seen from 90
degrees away, where the depth the network guessed is finally visible.

The mesh is a real one.  Every limb is a tapered tube of quads built in
metres, the head is a sphere, and each quad is projected through the *same*
camera that rendered the person into the frame - so the overlay lands on the
body rather than near it.  Back faces are culled and the rest painted far to
near, which is why the surface reads as a solid instead of a tangle.

The two things that must agree do agree by construction: the frame is
rendered from the ground-truth joints, the mesh is fitted to the predicted
joints, and MPJPE is the mean distance between those two arrays.  A pose that
looks further off has a bigger number, always.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch

from pubstyle import set_pub_style, savefig
from figkit import image_axes
from _wm_scenes import build, human_pose, project, body_radii, BONES, grey

H, W = 190, 136
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
DEEP, MID, PALE = "#20455f", "#2e74a8", "#cfe1ee"
FACE = np.array([0.545, 0.690, 0.800])          # mesh fill, cool blue
EDGE = "#27506e"
CARD = "#eef1f5"

DZ = 2.95
PROJ = dict(f=1.30, cam_y=1.02, cam_z=-DZ, cy=0.435, aspect=W / H)
LIGHT = np.array([-0.46, 0.66, -0.60])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

# (pose kind, phase, caption, per-joint noise sigma in metres, scene seed)
POSES = [("walk", 0.18, "walking", 0.0300, 6701),
         ("reach", 0.00, "reaching up", 0.0385, 6702),
         ("crouch", 0.00, "crouching", 0.0530, 6703),
         ("turn", 0.00, "turning away", 0.0445, 6704)]


# ---- mesh construction (metres) -------------------------------------------
def tube(p0, p1, r0, r1, nr=8, na=2):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        return []
    axis = axis / L
    tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else \
        np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    out = []
    for i in range(na):
        t0, t1 = i / na, (i + 1) / na
        c0, c1 = p0 + axis * L * t0, p0 + axis * L * t1
        q0, q1 = r0 + (r1 - r0) * t0, r0 + (r1 - r0) * t1
        for j in range(nr):
            a0, a1 = 2 * np.pi * j / nr, 2 * np.pi * (j + 1) / nr
            d0 = np.cos(a0) * u + np.sin(a0) * v
            d1 = np.cos(a1) * u + np.sin(a1) * v
            out.append(np.array([c0 + q0 * d0, c0 + q0 * d1,
                                 c1 + q1 * d1, c1 + q1 * d0]))
    return out


def sphere(c, r, nu=10, nv=6):
    c = np.asarray(c, float)
    def s(th, ph):
        return c + r * np.array([np.sin(th) * np.cos(ph), np.cos(th),
                                 np.sin(th) * np.sin(ph)])
    out = []
    for i in range(nv):
        th0, th1 = np.pi * i / nv, np.pi * (i + 1) / nv
        for j in range(nu):
            p0, p1 = 2 * np.pi * j / nu, 2 * np.pi * (j + 1) / nu
            out.append(np.array([s(th0, p0), s(th0, p1), s(th1, p1),
                                 s(th1, p0)]))
    return out


def body_mesh(P):
    quads = []
    for (a, b, r) in BONES:
        quads += tube(P[a], P[b], r * 1.02, r * 0.84, nr=8, na=2)
    quads += sphere(P[3] + np.array([0.0, 0.055, 0.0]), 0.108, 10, 6)
    return quads


def draw_mesh(ax, quads, yaw, alpha=0.66, lw=0.30):
    """Painter's algorithm with back-face culling, in tile pixel space."""
    prepared = []
    for q in quads:
        p2, dz = project(q, yaw=yaw, **PROJ)
        p2 = p2 * np.array([W, H])
        area = 0.0
        for k in range(4):
            x0, y0 = p2[k]
            x1, y1 = p2[(k + 1) % 4]
            area += x0 * y1 - x1 * y0
        if area >= 0:                       # back face: never seen
            continue
        n = np.cross(q[1] - q[0], q[2] - q[0])
        nn = np.linalg.norm(n)
        sh = 0.62 + 0.42 * max(0.0, float(np.dot(n / nn, LIGHT))) if nn > 0 \
            else 1.0
        prepared.append((float(dz.mean()), p2, sh))
    for _, p2, sh in sorted(prepared, key=lambda r: -r[0]):
        ax.add_patch(Polygon(p2, closed=True,
                             facecolor=np.clip(FACE * sh, 0, 1),
                             edgecolor=EDGE, lw=lw, alpha=alpha, zorder=4,
                             joinstyle="round"))


def tile(fig, gs, r, c):
    ax = fig.add_subplot(gs[r, c])
    image_axes(ax, color="#9a9a9a", lw=0.55)
    ax.set_aspect("auto")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    return ax


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1567)

    data = []
    for kind, phase, cap, sig, seed in POSES:
        gt = human_pose(kind, phase)
        # extremities are always the worst-recovered joints; the head sits
        # with the pelvis and the hips in the well-recovered group, because a
        # head that floats off the rendered head by two thirds of its own
        # diameter reads as a broken overlay, not as a 53 mm mean error
        scale = np.ones(18)
        scale[[5, 6, 8, 9]] = 1.55
        scale[[11, 12, 13, 15, 16, 17]] = 1.30
        scale[[0, 1, 3, 10, 14]] = 0.45
        pred = gt + rng.normal(0.0, sig, (18, 3)) * scale[:, None]
        mpjpe = float(np.linalg.norm(pred - gt, axis=1).mean() * 1000.0)
        pts_gt, _ = project(gt, yaw=0.0, **PROJ)
        sc = build("studio", seed=seed, shape=(H, W), body=pts_gt,
                   radii=body_radii(DZ), head_r=0.108 * PROJ["f"] / DZ,
                   horizon=0.420, shirt="#4c5460", trousers="#383f49",
                   cam=dict(exposure=1.03 + 0.015 * (seed % 3),
                            contrast=1.16 + 0.012 * (seed % 4)))
        data.append(dict(cap=cap, gt=gt, pred=pred, mpjpe=mpjpe, sc=sc,
                         mesh=body_mesh(pred)))

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 0.30, 1, 1, 1],
                          left=0.022, right=0.992, top=0.828, bottom=0.078,
                          wspace=0.045, hspace=0.315)

    cols = ["input frame", "mesh overlay", "side view  (yaw 90\u00b0)"]
    for k, d in enumerate(data):
        r, cbase = k // 2, (k % 2) * 4

        ax = tile(fig, gs, r, cbase)
        ax.imshow(d["sc"]["rgb"], interpolation="bilinear", aspect="auto")

        ax = tile(fig, gs, r, cbase + 1)
        ax.imshow(grey(d["sc"]["rgb"], lo=0.215, hi=0.720), cmap="gray",
                  vmin=0, vmax=1, interpolation="bilinear", aspect="auto")
        draw_mesh(ax, d["mesh"], yaw=0.0)
        jp, _ = project(d["pred"], yaw=0.0, **PROJ)
        ax.scatter(jp[:, 0] * W, jp[:, 1] * H, s=7.0, facecolors="#ffffff",
                   edgecolors="#14202c", linewidths=0.5, zorder=8)

        ax = tile(fig, gs, r, cbase + 2)
        ax.add_patch(FancyBboxPatch((0, 0), W, H, boxstyle="square,pad=0",
                                    facecolor=CARD, edgecolor="none",
                                    zorder=0))
        # after a 90-degree yaw the screen's horizontal axis is world Z, so
        # the ground line has to be laid along Z or it collapses to a spike
        floor, _ = project(np.array([[0.0, 0.0, -1.4], [0.0, 0.0, 1.4]]),
                           yaw=np.pi / 2, **PROJ)
        ax.plot(floor[:, 0] * W, floor[:, 1] * H, color="#b6c0c9", lw=0.9,
                zorder=1)
        draw_mesh(ax, d["mesh"], yaw=np.pi / 2, alpha=0.90, lw=0.32)

        # group header and the metric, centred over the three tiles
        p0 = fig.axes[k * 3].get_position()
        p2 = fig.axes[k * 3 + 2].get_position()
        xc = 0.5 * (p0.x0 + p2.x1)
        fig.text(p0.x0, p2.y1 + 0.016,
                 f"Pose {'ABCD'[k]}  \u00b7  {d['cap']}", fontsize=8.6,
                 fontweight="bold", color=INK, ha="left", va="bottom")
        fig.text(p2.x1, p2.y1 + 0.018, f"MPJPE {d['mpjpe']:.1f} mm",
                 fontsize=7.2, color=DEEP, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.26", facecolor=PALE,
                           edgecolor="#9fb4c6", lw=0.55))
        for j in range(3):
            pj = fig.axes[k * 3 + j].get_position()
            fig.text(pj.x0 + pj.width * 0.5, pj.y0 - 0.012, cols[j],
                     fontsize=6.6, color=SUB, ha="center", va="top")
        _ = xc

    savefig(fig, "1162_wm_human_mesh_overlay.png")


if __name__ == "__main__":
    main()
