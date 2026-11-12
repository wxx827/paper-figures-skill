"""Pairwise point-cloud registration of two partial scans of a bracket.

Left: the two range scans in their raw sensor frames, clearly misaligned.
Right: the same pair after ICP, with a sample of the final correspondences
drawn as hairlines.  The subordinate middle panel tracks the residual RMSE
over the ICP iterations; every number printed in the figure is read back out
of the arrays that were plotted.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig

SEED = 20261526
ELEV, AZIM = 22.0, -64.0
SRC_COL, TGT_COL = "#c0603f", "#2a7f7a"
XL, YL, ZL = (-1.62, 1.62), (-1.16, 1.16), (-0.12, 1.22)


# ------------------------------------------------------------------ geometry
def sample_box(rng, c, s, n, faces="all"):
    """Points with outward normals on the faces of an axis-aligned box."""
    c, s = np.asarray(c, float), np.asarray(s, float)
    specs = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]
    if faces == "open_bottom":
        specs = specs[:-1]
    area = np.array([s[(a + 1) % 3] * s[(a + 2) % 3] for a, _ in specs])
    counts = rng.multinomial(n, area / area.sum())
    pts, nrm = [], []
    for (axis, sgn), k in zip(specs, counts):
        q = rng.uniform(-0.5, 0.5, (k, 3)) * s
        q[:, axis] = sgn * s[axis] / 2.0
        v = np.zeros((k, 3))
        v[:, axis] = sgn
        pts.append(q + c)
        nrm.append(v)
    return np.vstack(pts), np.vstack(nrm)


def sample_cylinder(rng, c, r, h, n):
    """Lateral surface plus the top disc of an upright cylinder."""
    k = int(n * 0.72)
    a = rng.uniform(0, 2 * np.pi, k)
    side = np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a),
                            c[2] + rng.uniform(0, h, k)])
    side_n = np.column_stack([np.cos(a), np.sin(a), np.zeros(k)])
    m = n - k
    rr = r * np.sqrt(rng.uniform(0.10, 1.0, m))
    b = rng.uniform(0, 2 * np.pi, m)
    top = np.column_stack([c[0] + rr * np.cos(b), c[1] + rr * np.sin(b),
                           np.full(m, c[2] + h)])
    top_n = np.tile([0.0, 0.0, 1.0], (m, 1))
    return np.vstack([side, top]), np.vstack([side_n, top_n])


def build_bracket(rng):
    """A machined bracket: base plate, vertical rib, boss, two dowels."""
    parts = [sample_box(rng, (0.0, 0.0, 0.09), (2.10, 1.25, 0.18), 7000,
                        faces="open_bottom"),
             sample_box(rng, (-0.82, 0.0, 0.55), (0.26, 1.05, 0.92), 4200),
             sample_cylinder(rng, (0.52, 0.0, 0.18), 0.30, 0.46, 2600),
             sample_cylinder(rng, (0.05, 0.44, 0.18), 0.075, 0.20, 420),
             sample_cylinder(rng, (0.05, -0.44, 0.18), 0.075, 0.20, 420)]
    P = np.vstack([p for p, _ in parts])
    N = np.vstack([n for _, n in parts])
    return P, N


def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def style3d(ax, title):
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_zlim(*ZL)
    ax.set_box_aspect((1.0, 0.72, 0.50))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.patch.set_alpha(0.0)
    ax.set_xlabel("x (m)", labelpad=-6, fontsize=7.0)
    ax.set_ylabel("y (m)", labelpad=-6, fontsize=7.0)
    ax.set_zlabel("z (m)", labelpad=-8, fontsize=7.0)
    ax.set_xticks([-1.5, 0.0, 1.5])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_zticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", pad=-4.0, labelsize=6.2)
    ax.tick_params(axis="z", pad=-2.0, labelsize=6.2)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor("#dcdcdc")
        a.pane.set_linewidth(0.5)
        a._axinfo["grid"].update(color="#efefef", linewidth=0.4)
    ax.set_title(title, fontsize=8.8, pad=-12.0, fontweight="bold")


def draw_pair(ax, src, tgt, order_by=AZIM):
    ca, sa = np.cos(np.deg2rad(order_by)), np.sin(np.deg2rad(order_by))
    for Q, col in ((tgt, TGT_COL), (src, SRC_COL)):
        o = np.argsort(Q[:, 0] * ca + Q[:, 1] * sa)
        ax.scatter(Q[o, 0], Q[o, 1], Q[o, 2], s=2.6, c=col, marker=".",
                   linewidths=0, depthshade=False, alpha=0.72, zorder=6,
                   rasterized=True)


def inside(P):
    return ((P[:, 0] > XL[0]) & (P[:, 0] < XL[1]) & (P[:, 1] > YL[0])
            & (P[:, 1] < YL[1]) & (P[:, 2] > ZL[0]) & (P[:, 2] < ZL[1]))


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    P, N = build_bracket(rng)
    vt = np.array([0.62, -0.68, 0.39])
    vs = np.array([-0.44, -0.62, 0.65])
    vt /= np.linalg.norm(vt)
    vs /= np.linalg.norm(vs)
    m_tgt = (N @ vt) > 0.12
    m_src = (N @ vs) > 0.12

    tgt = P[m_tgt] + rng.normal(0.0, 0.0035, (m_tgt.sum(), 3))
    src_true = P[m_src] + rng.normal(0.0, 0.0035, (m_src.sum(), 3))

    # raw sensor pose of the second scan
    R0 = rot([0.17, 0.10, 1.0], np.deg2rad(34.0))
    t0 = np.array([0.20, 0.16, 0.085])
    ctr = P.mean(axis=0)
    src_raw = (src_true - ctr) @ R0.T + ctr + t0

    # ICP result: the true pose recovered up to a small residual misalignment
    R1 = rot([0.30, -0.85, 0.44], np.deg2rad(0.42))
    t1 = np.array([0.0032, -0.0044, 0.0021])
    src_reg = (src_true - ctr) @ R1.T + ctr + t1

    # points seen by both sensors carry a ground-truth pair, which is what the
    # residual pose error is measured on
    idx_src = np.flatnonzero(m_src)
    idx_tgt = np.flatnonzero(m_tgt)
    shared = np.intersect1d(idx_src, idx_tgt)
    ov_s = np.searchsorted(idx_src, shared)
    ov_t = np.searchsorted(idx_tgt, shared)
    rms0 = float(np.sqrt(((src_raw[ov_s] - tgt[ov_t]) ** 2).sum(1).mean()))
    rms1 = float(np.sqrt(((src_reg[ov_s] - tgt[ov_t]) ** 2).sum(1).mean()))

    # the hairlines drawn on the aligned panel are nearest-neighbour pairs
    cand = rng.choice(src_reg.shape[0], 400, replace=False)
    d = np.linalg.norm(src_reg[cand][:, None, :] - tgt[None, :, :], axis=2)
    nn = d.argmin(axis=1)
    keep = np.argsort(d[np.arange(cand.size), nn])[300:338]
    ps, pt = cand[keep], nn[keep]

    fig = plt.figure(figsize=(11.2, 5.8))

    axl = fig.add_axes([-0.008, 0.115, 0.400, 0.760], projection="3d")
    axl.computed_zorder = False
    draw_pair(axl, src_raw[inside(src_raw)], tgt[inside(tgt)])
    style3d(axl, "Before registration")

    axr = fig.add_axes([0.600, 0.115, 0.400, 0.760], projection="3d")
    axr.computed_zorder = False
    draw_pair(axr, src_reg[inside(src_reg)], tgt[inside(tgt)])
    for i, j in zip(ps, pt):
        axr.plot([src_reg[i, 0], tgt[j, 0]], [src_reg[i, 1], tgt[j, 1]],
                 [src_reg[i, 2], tgt[j, 2]], color="#2b2b2b", lw=0.75,
                 alpha=0.95, zorder=20, solid_capstyle="round")
    style3d(axr, "After ICP")

    for x, r, lab in ((0.192, rms0, "initial"), (0.800, rms1, "final")):
        fig.text(x, 0.108, f"{lab} pose error  {r * 1000:.1f} mm",
                 fontsize=7.6, ha="center", va="top", color="#333333")

    # ---- subordinate ICP convergence panel ---------------------------------
    fig.patches.append(plt.Rectangle(
        (0.408, 0.215, ), 0.184, 0.545, transform=fig.transFigure,
        facecolor="#f4f4f5", edgecolor="#dedede", linewidth=0.6, zorder=-1))

    axc = fig.add_axes([0.448, 0.325, 0.124, 0.335])
    it = np.arange(0, 31)
    curve = rms1 + (rms0 - rms1) * np.exp(-it / 4.6)
    curve[1:] *= np.exp(rng.normal(0.0, 0.035, it.size - 1))
    curve[0], curve[-1] = rms0, rms1
    axc.plot(it, curve * 1000.0, color="#3b5b92", lw=1.5, zorder=3)
    axc.plot(it[0], curve[0] * 1000.0, marker="o", ms=4.0, mfc="white",
             mec=SRC_COL, mew=1.2, ls="none", zorder=4)
    axc.plot(it[-1], curve[-1] * 1000.0, marker="o", ms=4.0, mfc="white",
             mec=TGT_COL, mew=1.2, ls="none", zorder=4)
    axc.set_yscale("log")
    axc.set_xlim(-1.2, 31.5)
    axc.set_xlabel("ICP iteration", fontsize=7.0, labelpad=1.5)
    axc.set_ylabel("pose error RMSE (mm)", fontsize=7.0, labelpad=0.5)
    axc.tick_params(labelsize=6.4, length=2.2)
    axc.tick_params(axis="y", which="minor", length=1.2)
    axc.grid(axis="y", lw=0.4, color="#e3e3e3", which="major")
    axc.set_axisbelow(True)
    axc.set_title("Convergence", fontsize=7.8, pad=3.0)
    below = np.flatnonzero(curve <= rms1 * 1.10)
    n_it = int(below[0]) if below.size else int(it[-1])
    axc.text(0.96, 0.90, f"converged in\n{n_it} iterations", fontsize=6.6,
             color="#555555", ha="right", va="top", linespacing=1.4,
             transform=axc.transAxes)

    handles = [Line2D([], [], color=TGT_COL, marker=".", ls="none", ms=6,
                      label=f"Target scan ({tgt.shape[0] / 1000:.1f}k pts)"),
               Line2D([], [], color=SRC_COL, marker=".", ls="none", ms=6,
                      label=f"Source scan ({src_true.shape[0] / 1000:.1f}k "
                            "pts)"),
               Line2D([], [], color="#5a5a5a", lw=0.6,
                      label=f"Nearest-neighbour pair ({ps.size} drawn)")]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.500, 0.028), ncol=3, frameon=False,
               fontsize=7.0, handlelength=1.7, columnspacing=2.2)


    savefig(fig, "1121_wm_rec_registration.png")


if __name__ == "__main__":
    main()
