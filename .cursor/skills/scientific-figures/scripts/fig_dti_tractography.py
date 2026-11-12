"""Figure 537 - Direction-encoded tractography of three major white-matter bundles.

Streamlines are coloured by the *local* orientation of each segment, red for
left-right, green for anterior-posterior, blue for inferior-superior.  That RGB
convention is a hard domain standard - a reader of a diffusion figure decodes
colour as direction before reading any label - so it is kept here even though
the library otherwise bans saturated primaries.  Everything else on the page
follows the house palette.

The subordinate bars report fractional anisotropy (simulated across subjects)
and tract volume measured by voxelising the streamlines that are actually
drawn, so the numbers cannot drift away from the geometry.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pubstyle import set_pub_style, savefig, COLORS
from figkit import axes_inches, panel_labels_inches

BRAIN_C = np.array([0.0, -14.0, 12.0])       # ellipsoid centre (mm)
BRAIN_R = np.array([72.0, 94.0, 60.0])       # semi-axes (mm)
VOX = 2.0                                     # voxel size for volumetry (mm)
TUBE_R = 3.0                                  # nominal streamline radius (mm)

# name, control polygon (mm), streamline count, spread (mm), mirrored?
BUNDLES = [
    ("Corpus callosum",
     [(-42, 0, 14), (-16, 0, 32), (0, 0, 36), (16, 0, 32), (42, 0, 14)],
     34, 4.0, False),
    ("Corticospinal tract",
     [(8, -16, -34), (14, -20, -6), (28, -20, 30), (38, -18, 56)],
     26, 3.4, True),
    ("Arcuate fasciculus",
     [(-48, 40, 4), (-40, 26, 44), (-44, -34, 44), (-56, -48, -14)],
     24, 3.2, True),
]
FA_TRUE = {"Corpus callosum": 0.61, "Corticospinal tract": 0.55,
           "Arcuate fasciculus": 0.47}
N_SUBJ = 24


def bezier(ctrl, t):
    """De Casteljau evaluation of one Bezier curve at parameters ``t``."""
    pts = [np.asarray(c, dtype=float) for c in ctrl]
    while len(pts) > 1:
        pts = [(1 - t)[:, None] * a + t[:, None] * b
               for a, b in zip(pts[:-1], pts[1:])]
    return pts[0]


def make_bundle(name, ctrl, n_lines, spread, mirror, rng, n_pts=110):
    """A fan of jittered streamlines; corpus callosum also fans front-to-back."""
    lines = []
    t = np.linspace(0.0, 1.0, n_pts)
    for k in range(n_lines):
        c = np.array(ctrl, dtype=float)
        if name.startswith("Corpus"):
            yk = np.linspace(-52.0, 40.0, n_lines)[k]
            c[:, 1] += yk
            c[0, 1] += 0.30 * yk                    # radiating ends
            c[-1, 1] += 0.30 * yk
            c[:, 2] -= 0.055 * abs(yk)
        c += rng.normal(0.0, spread, c.shape)
        p = bezier(c, t)
        p += 0.45 * rng.standard_normal((1, 3)) * np.sin(
            np.linspace(0, np.pi, n_pts))[:, None]
        lines.append(p)
        if mirror:
            q = p.copy()
            q[:, 0] *= -1.0
            lines.append(q)
    return lines


def direction_colors(p):
    """Segment colours from |unit tangent| - the RGB diffusion convention."""
    d = np.diff(p, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    return np.clip(np.abs(d) * 0.94 + 0.03, 0.0, 1.0)


def tract_volume(lines):
    """Volume in cm^3 of the voxels the drawn streamlines pass through."""
    pts = np.vstack(lines)
    offs = np.vstack([np.zeros(3), np.eye(3) * TUBE_R, -np.eye(3) * TUBE_R])
    grid = np.vstack([np.floor((pts + o) / VOX) for o in offs]).astype(np.int64)
    n_vox = np.unique(grid, axis=0).shape[0]
    return n_vox * VOX ** 3 / 1000.0


def draw_brain(ax, rng):
    """Faint ellipsoid shell plus three silhouette rings for context."""
    u = np.linspace(0, 2 * np.pi, 56)
    v = np.linspace(0, np.pi, 30)
    x = BRAIN_C[0] + BRAIN_R[0] * np.outer(np.cos(u), np.sin(v))
    y = BRAIN_C[1] + BRAIN_R[1] * np.outer(np.sin(u), np.sin(v))
    z = BRAIN_C[2] + BRAIN_R[2] * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#8a8a8a", alpha=0.040, linewidth=0,
                    antialiased=True, shade=False, zorder=0)
    a = np.linspace(0, 2 * np.pi, 200)
    rings = [(BRAIN_C[0] + BRAIN_R[0] * np.cos(a),
              BRAIN_C[1] + BRAIN_R[1] * np.sin(a),
              np.full_like(a, BRAIN_C[2])),
             (BRAIN_C[0] + BRAIN_R[0] * np.cos(a),
              np.full_like(a, BRAIN_C[1]),
              BRAIN_C[2] + BRAIN_R[2] * np.sin(a)),
             (np.full_like(a, BRAIN_C[0]),
              BRAIN_C[1] + BRAIN_R[1] * np.cos(a),
              BRAIN_C[2] + BRAIN_R[2] * np.sin(a))]
    for rx, ry, rz in rings:
        ax.plot(rx, ry, rz, color="#9a9a9a", lw=0.55, alpha=0.55, zorder=1)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(17)

    bundles = {}
    for name, ctrl, n, spread, mirror in BUNDLES:
        bundles[name] = make_bundle(name, ctrl, n, spread, mirror, rng)

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))

    # ---- (a) 3-D streamlines ----------------------------------------------
    ax = axes_inches(fig, (0.10, 0.44, 6.55, 5.10), projection="3d")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    draw_brain(ax, rng)

    mean_rgb = {}
    for name, lines in bundles.items():
        cols = []
        for p in lines:
            c = direction_colors(p)
            segs = np.stack([p[:-1], p[1:]], axis=1)
            ax.add_collection3d(Line3DCollection(
                segs, colors=c, linewidths=0.62, alpha=0.88, zorder=4))
            cols.append(c)
        mean_rgb[name] = np.clip(np.vstack(cols).mean(axis=0) * 1.05, 0, 1)

    lo = BRAIN_C - BRAIN_R * 1.02
    hi = BRAIN_C + BRAIN_R * 1.02
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(BRAIN_R * 2), zoom=1.52)
    ax.view_init(elev=16, azim=-64)

    for tx, ty, tz, lab in [(lo[0] * 1.12, BRAIN_C[1], BRAIN_C[2], "left"),
                            (hi[0] * 1.12, BRAIN_C[1], BRAIN_C[2], "right"),
                            (0, BRAIN_C[1], hi[2] * 1.14, "superior")]:
        ax.text(tx, ty, tz, lab, fontsize=6.6, color=COLORS["muted"],
                ha="center", va="center", zorder=8)

    key = [("L \u2013 R", "#d02020"), ("A \u2013 P", "#12a012"),
           ("I \u2013 S", "#1a3fd0")]
    fig.text(0.036, 0.855, "Oblique view from right, posterior and above",
             fontsize=6.8, color=COLORS["muted"], ha="left", va="center")
    fig.text(0.036, 0.150, "Segment colour = local fibre orientation",
             fontsize=7.0, color=COLORS["sub"], ha="left", va="center")
    for i, (lab, col) in enumerate(key):
        fig.text(0.036 + 0.052 * i, 0.108, lab, fontsize=7.6, color=col,
                 fontweight="bold", ha="left", va="center")

    # ---- (b) fractional anisotropy, (c) tract volume ----------------------
    names = [b[0] for b in BUNDLES]
    fa = {n: np.clip(rng.normal(FA_TRUE[n], 0.035, N_SUBJ), 0.05, 0.95)
          for n in names}
    vol = {n: tract_volume(bundles[n]) for n in names}
    bar_c = [mean_rgb[n] for n in names]
    ypos = np.arange(len(names))[::-1]

    axb = axes_inches(fig, (7.62, 3.28, 2.26, 1.90))
    axb.barh(ypos, [fa[n].mean() for n in names], height=0.56, color=bar_c,
             lw=0.5, edgecolor="white")
    axb.errorbar([fa[n].mean() for n in names], ypos,
                 xerr=[fa[n].std(ddof=1) for n in names], fmt="none",
                 ecolor=COLORS["sub"], elinewidth=0.8, capsize=2.0)
    for k, n in enumerate(names):
        axb.text(fa[n].mean() + fa[n].std(ddof=1) + 0.012, ypos[k],
                 f"{fa[n].mean():.2f}", fontsize=6.5, va="center",
                 color=COLORS["ink"])
    axb.set_xlim(0.0, 0.78)
    axb.set_xlabel("Fractional anisotropy (-)", fontsize=7.6)
    axb.set_title(f"Mean FA \u00b1 SD, n = {N_SUBJ} subjects", fontsize=8.2,
                  fontweight="bold", pad=4)

    axc = axes_inches(fig, (7.62, 0.72, 2.26, 1.90))
    vals = [vol[n] for n in names]
    axc.barh(ypos, vals, height=0.56, color=bar_c, lw=0.5, edgecolor="white")
    for k, v in enumerate(vals):
        axc.text(v + max(vals) * 0.02, ypos[k], f"{v:.1f}", fontsize=6.5,
                 va="center", color=COLORS["ink"])
    axc.set_xlim(0.0, max(vals) * 1.22)
    axc.set_xlabel("Tract volume (cm$^3$)", fontsize=7.6)
    axc.set_title("Volume of the drawn streamlines", fontsize=8.2,
                  fontweight="bold", pad=4)

    for a in (axb, axc):
        a.set_yticks(ypos)
        a.set_yticklabels(names, fontsize=7.0)
        a.tick_params(labelsize=6.8, length=2.2)
        a.set_ylim(-0.6, len(names) - 0.4)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    panel_labels_inches(fig, [ax, axb, axc], dx=-0.46, dy=0.24)
    n_lines = sum(len(v) for v in bundles.values())

    savefig(fig, "537_dti_tractography.png")


if __name__ == "__main__":
    main()
