"""Structure-from-motion feature tracks: epipolar matches and 3-D structure.

Left: a stereo pair of two street facades with the pencil of epipolar lines
and the putative correspondences, inliers in green and rejected matches in
red.  Right: the structure those matches triangulate to -- two quasi-planar
facade clusters at different depths, the street plane between them and a
sparse far wall glimpsed through the gap -- coloured by reprojection error.
Window panes carry visibly fewer points than the surrounding masonry because
untextured glass gives no matchable features.  A slim stacked bar underneath
reports the track-length mix.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import affine_transform, gaussian_filter

from pubstyle import set_pub_style, get_cmap, savefig

SEED = 20261523
IH, IW = 148, 214
GAP = 26.0
ELEV, AZIM = 20.0, -66.0
TRACK_CMAP = get_cmap("cb_sequential_blue")
ERR_CMAP = plt.get_cmap("cividis")
INLIER, OUTLIER = "#2e7d5b", "#b23a48"

# One layout drives both halves of the figure: the image blocks below and the
# triangulated facade clusters in ``build_tracks`` describe the same street.
#   image: (x0, x1, roof_row, tone, win_pitch_x, win_pitch_y, win_w, win_h)
IMG_BLOCKS = [(84, 126, 74, 0.82, 13, 16, 7, 9),      # far wall, seen in gap
              (8, 90, 36, 0.60, 19, 22, 11, 13),      # near-left facade
              (118, 206, 52, 0.70, 19, 22, 11, 13)]   # near-right facade
#   world: quasi-planar clusters, the left one closer to the camera pair
FACADES = [dict(x=(-7.2, -1.4), z=(0.15, 5.60), y=4.20,
                win=(0.86, 1.06), pitch=(1.48, 1.62), phase=(0.34, 0.52)),
           dict(x=(0.40, 7.40), z=(0.15, 4.60), y=6.80,
                win=(0.80, 0.96), pitch=(1.40, 1.50), phase=(0.28, 0.44))]


def track_color(length):
    """Track length 2..7 -> the darker two thirds of the blue ramp."""
    t = (np.asarray(length, float) - 2.0) / 5.0
    return TRACK_CMAP(0.38 + 0.62 * np.clip(t, 0.0, 1.0))


def clip_segment(p, d, x0, x1, y0, y1):
    """Liang-Barsky clip of the ray ``p + t d`` against an image rectangle."""
    t0, t1 = -1e9, 1e9
    for num, den in ((x0 - p[0], d[0]), (p[0] - x1, -d[0]),
                     (y0 - p[1], d[1]), (p[1] - y1, -d[1])):
        if abs(den) < 1e-12:
            if num > 0:
                return None
            continue
        t = num / den
        if den > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 >= t1:
        return None
    return np.array([p + t0 * d, p + t1 * d])


# -------------------------------------------------------------------- images
def render_facade(rng):
    """Two near facades with a gap, a far wall behind it, and the street."""
    yy, xx = np.mgrid[0:IH, 0:IW].astype(float)
    img = 0.92 - 0.17 * (yy / IH)
    for x0, x1, top, val, sx, sy, ww, wh in IMG_BLOCKS:
        m = (xx >= x0) & (xx < x1) & (yy >= top)
        img[m] = val
        for wy in range(top + 11, IH - 26, sy):
            for wx in range(x0 + 8, x1 - ww - 2, sx):
                w = (xx >= wx) & (xx < wx + ww) & (yy >= wy) & (yy < wy + wh)
                img[w] = val - 0.26
                img[(xx >= wx) & (xx < wx + ww) & (yy >= wy)
                    & (yy < wy + 3)] = val + 0.14
    img[yy >= IH - 22] = 0.42 + 0.10 * ((IH - yy[yy >= IH - 22]) / 22.0)
    img += 0.055 * gaussian_filter(rng.normal(0, 1, img.shape), 2.2)
    img += 0.022 * rng.normal(0, 1, img.shape)
    return np.clip(gaussian_filter(img, 0.55), 0.04, 1.0)


def warp(img, M, off):
    return affine_transform(img, M, offset=off, order=1, mode="reflect")


# ------------------------------------------------------------------ geometry
def look_at(C, target, up=np.array([0.0, 0.0, 1.0])):
    f = target - C
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return np.column_stack([r, np.cross(r, f), f])


def draw_frustum(ax, C, R, color, w=0.95, h=0.68, d=1.7, lw=0.85, z=30):
    v = C + np.array([[w, h, d], [-w, h, d], [-w, -h, d], [w, -h, d]]) @ R.T
    for p in v:
        ax.plot(*zip(C, p), color=color, lw=lw * 0.8, zorder=z)
    ring = np.vstack([v, v[:1]])
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, lw=lw, zorder=z)
    pc = Poly3DCollection([v], facecolor=color, edgecolor="none", alpha=0.15)
    pc.set_zorder(z - 1)
    ax.add_collection3d(pc)
    ax.plot([C[0]], [C[1]], [C[2]], marker="o", ms=2.8, color=color,
            mec="white", mew=0.4, zorder=z + 1)


def facade_cluster(rng, spec, n_cand):
    """Quasi-planar cluster whose density follows the window lattice.

    Glass panes match badly and stay sparse, the frames around them are the
    richest source of features, plain masonry sits in between.  Returns the
    points plus, per point, the glass flag and a 0..1 silhouette-edge weight.
    """
    x = rng.uniform(*spec["x"], n_cand)
    z = rng.uniform(*spec["z"], n_cand)
    px, pz = spec["pitch"]
    ww, wh = spec["win"]
    u = np.mod(x - spec["x"][0] - spec["phase"][0], px)
    v = np.mod(z - spec["z"][0] - spec["phase"][1], pz)
    du = np.maximum(np.abs(u - px / 2.0) - ww / 2.0, 0.0)
    dv = np.maximum(np.abs(v - pz / 2.0) - wh / 2.0, 0.0)
    d = np.hypot(du, dv)
    glass = d <= 0.0
    frame = (~glass) & (d < 0.13)
    p = np.where(glass, 0.14, np.where(frame, 1.0, 0.48))
    keep = rng.random(n_cand) < p
    x, z, glass = x[keep], z[keep], glass[keep]

    # thin shell around the plane: tight along the normal, loose in plane
    y = spec["y"] + rng.normal(0.0, 0.022, x.size)
    x = x + rng.normal(0.0, 0.012, x.size)
    z = z + rng.normal(0.0, 0.012, x.size)

    ex = np.minimum(x - spec["x"][0], spec["x"][1] - x)
    ez = np.minimum(z - spec["z"][0], spec["z"][1] - z)
    edge = np.clip(1.0 - np.minimum(ex, ez) / 0.28, 0.0, 1.0)
    return np.column_stack([x, y, z]), glass, edge


def build_tracks(rng, cams):
    """Structure implied by the image pair, with per-point reprojection error.

    Two facade clusters at different depths, the street between them and the
    far wall that shows through the gap.  The error field is derived from the
    geometry: distance to the pair, silhouette edges and untextured glass.
    """
    pts, glass, edge, kind = [], [], [], []
    for spec, n in zip(FACADES, (8200, 8200)):
        P, g, e = facade_cluster(rng, spec, n)
        pts.append(P)
        glass.append(g)
        edge.append(e)
        kind.append(np.full(P.shape[0], 0))

    # street plane between the two facades, sparse and low-texture
    n_g = 420
    G = np.column_stack([rng.uniform(-7.4, 7.4, n_g),
                         rng.uniform(0.4, 7.0, n_g),
                         rng.normal(0.0, 0.022, n_g)])
    pts.append(G)
    glass.append(np.zeros(n_g, bool))
    edge.append(np.zeros(n_g))
    kind.append(np.full(n_g, 1))

    # far wall glimpsed through the gap: few points, all of them poor
    n_f = 260
    F = np.column_stack([rng.uniform(-1.9, 1.1, n_f),
                         10.2 + rng.normal(0.0, 0.10, n_f),
                         rng.uniform(0.3, 3.4, n_f)])
    pts.append(F)
    glass.append(np.zeros(n_f, bool))
    edge.append(np.zeros(n_f))
    kind.append(np.full(n_f, 2))

    P = np.vstack(pts)
    glass = np.concatenate(glass)
    edge = np.concatenate(edge)
    kind = np.concatenate(kind)

    mid = cams.mean(axis=0)
    d = np.linalg.norm(P - mid, axis=1)
    err = (0.21 + 0.026 * (d - d.min())
           + 0.26 * edge + 0.34 * glass
           + 0.11 * (kind == 1) + 0.20 * (kind == 2)
           + np.abs(rng.normal(0.0, 0.055, P.shape[0])))

    # track length follows the error ranking, spread evenly over 2..7 views
    r = np.argsort(np.argsort(err)) / (err.size - 1.0)
    length = np.clip(np.round(6.7 - 5.2 * r + rng.normal(0.0, 0.55, r.size)),
                     2, 7)
    return P, length, err


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    img1 = render_facade(rng)
    th = np.deg2rad(3.4)
    M = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * 0.955
    ctr = np.array([IH / 2.0, IW / 2.0])
    off = ctr - M @ (ctr + np.array([4.0, 17.0]))
    img2 = warp(img1, M, off)

    # keypoints picked in view 2, mapped back through the same warp
    n_in, n_out = 17, 5
    p2 = np.column_stack([rng.uniform(14, IH - 16, n_in + n_out),
                          rng.uniform(14, IW - 16, n_in + n_out)])
    p1 = (M @ p2.T).T + off
    p1[n_in:] += rng.normal(0.0, 26.0, (n_out, 2))        # mismatches
    p1 = np.clip(p1, [6, 6], [IH - 7, IW - 7])

    fig = plt.figure(figsize=(11.2, 5.6))

    # ---- image pair with epipolar geometry --------------------------------
    axi = fig.add_axes([0.028, 0.320, 0.552, 0.440])
    axi.imshow(img1, cmap="gray", vmin=0.0, vmax=1.0, origin="upper",
               extent=[0, IW, 0, IH], interpolation="bilinear", zorder=2)
    axi.imshow(img2, cmap="gray", vmin=0.0, vmax=1.0, origin="upper",
               extent=[IW + GAP, 2 * IW + GAP, 0, IH],
               interpolation="bilinear", zorder=2)
    for x0, lab in ((0.0, "view $I_k$"), (IW + GAP, "view $I_{k+1}$")):
        axi.add_patch(plt.Rectangle((x0, 0), IW, IH, facecolor="none",
                                    edgecolor="#555555", linewidth=0.8,
                                    zorder=6))
        axi.text(x0 + IW / 2, IH + 4, lab, fontsize=7.4, ha="center",
                 va="bottom", color="#333333")

    epi = np.array([2 * IW + GAP + 120.0, 0.52 * IH])     # epipole, off-image
    for i in range(n_in + n_out):
        q = np.array([IW + GAP + p2[i, 1], IH - p2[i, 0]])
        d = q - epi
        d /= np.linalg.norm(d)
        seg = clip_segment(epi, d, IW + GAP, 2 * IW + GAP, 0.0, IH)
        if seg is not None:
            axi.plot(seg[:, 0], seg[:, 1], color="#8fb4cf", lw=0.5,
                     alpha=0.85, zorder=3)

    for i in range(n_in + n_out):
        good = i < n_in
        col = INLIER if good else OUTLIER
        a = np.array([p1[i, 1], IH - p1[i, 0]])
        b = np.array([IW + GAP + p2[i, 1], IH - p2[i, 0]])
        axi.plot([a[0], b[0]], [a[1], b[1]], color=col,
                 lw=0.45 if good else 0.55, alpha=0.75 if good else 0.9,
                 zorder=4 if good else 5)
        for p in (a, b):
            axi.plot(p[0], p[1], marker="o", ms=2.6, mfc=col, mec="white",
                     mew=0.45, ls="none", zorder=7)
    axi.set_xlim(-4, 2 * IW + GAP + 4)
    axi.set_ylim(-6, IH + 16)
    axi.set_aspect("equal")
    axi.axis("off")
    axi.text(2 * IW + GAP, -2, "$l' = F\\,x$", fontsize=7.0, color="#5b83a0",
             ha="right", va="top")

    handles = [Line2D([], [], color=INLIER, lw=0.9, marker="o", ms=3.0,
                      mec="white", mew=0.4, label=f"Inlier match ({n_in})"),
               Line2D([], [], color=OUTLIER, lw=0.9, marker="o", ms=3.0,
                      mec="white", mew=0.4, label=f"Rejected match ({n_out})"),
               Line2D([], [], color="#8fb4cf", lw=0.6,
                      label="Epipolar line $Fx$")]
    axi.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.02),
               ncol=3, frameon=False, fontsize=6.9, handlelength=1.9,
               columnspacing=1.6)

    # ---- 3-D tracks and baseline ------------------------------------------
    ax3 = fig.add_axes([0.586, 0.215, 0.408, 0.720], projection="3d")
    ax3.computed_zorder = False
    ax3.patch.set_alpha(0.0)

    # the recovered pair translates along the street, matching the mostly
    # horizontal parallax between I_k and I_{k+1}
    cams = np.array([[-1.15, -2.55, 1.75], [0.95, -2.45, 1.82]])
    tgt = np.array([-0.1, 5.2, 2.5])
    P, tlen, perr = build_tracks(rng, cams)
    ca, sa = np.cos(np.deg2rad(AZIM)), np.sin(np.deg2rad(AZIM))
    o = np.argsort(P[:, 0] * ca + P[:, 1] * sa)
    P, tlen, perr = P[o], tlen[o], perr[o]

    for g in np.arange(-7.0, 7.1, 3.5):
        ax3.plot([g, g], [0.4, 7.0], [0, 0], color="#efefef", lw=0.4,
                 zorder=2)
    for g in np.arange(0.4, 7.1, 1.65):
        ax3.plot([-7.4, 7.4], [g, g], [0, 0], color="#efefef", lw=0.4,
                 zorder=2)

    evmin, evmax = np.percentile(perr, [2.0, 97.0])
    enorm = Normalize(evmin, evmax)
    ax3.scatter(P[:, 0], P[:, 1], P[:, 2], s=3.6, c=ERR_CMAP(enorm(perr)),
                marker=".", linewidths=0, depthshade=False, alpha=0.92,
                zorder=10, rasterized=True)
    for C in cams:
        draw_frustum(ax3, C, look_at(C, tgt), "#8c4a12", w=0.62, h=0.45,
                     d=1.15)
    ax3.plot(cams[:, 0], cams[:, 1], cams[:, 2], color="#666666", lw=0.9,
             ls=(0, (3.6, 2.2)), zorder=28)
    mid = cams.mean(axis=0)
    base = float(np.linalg.norm(cams[1] - cams[0]))
    ax3.text(mid[0], mid[1] - 0.35, mid[2] - 0.30, f"baseline  {base:.2f} m",
             fontsize=6.6, color="#333333", ha="center", va="top", zorder=30,
             bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                       edgecolor="none", alpha=0.82))
    # viewing rays from both centres through a handful of tracks
    for C in cams:
        for i in rng.choice(P.shape[0], 12, replace=False):
            ax3.plot([C[0], P[i, 0]], [C[1], P[i, 1]], [C[2], P[i, 2]],
                     color="#c2c2c2", lw=0.3, alpha=0.5,
                     ls=(0, (2.4, 2.0)), zorder=6)

    ax3.set_xlim(-7.8, 7.8)
    ax3.set_ylim(-3.4, 11.0)
    ax3.set_zlim(0.0, 6.0)
    ax3.set_box_aspect((1.0, 0.86, 0.46))
    ax3.view_init(elev=ELEV, azim=AZIM)
    ax3.set_xlabel("X (m)", labelpad=-5, fontsize=7.2)
    ax3.set_ylabel("Y (m)", labelpad=-4, fontsize=7.2)
    ax3.set_zlabel("Z (m)", labelpad=-7, fontsize=7.2)
    ax3.set_xticks([-6, 0, 6])
    ax3.set_yticks([0, 5, 10])
    ax3.set_zticks([0, 3, 6])
    ax3.tick_params(axis="both", pad=-3.5, labelsize=6.4)
    ax3.tick_params(axis="z", pad=-1.5, labelsize=6.4)
    for a in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor("#dcdcdc")
        a.pane.set_linewidth(0.5)
        a._axinfo["grid"].update(color="#ededed", linewidth=0.4)

    for x, txt in ((0.304, "Epipolar verification of putative "
                           "correspondences"),
                   (0.790, "Triangulated structure and recovered baseline")):
        fig.text(x, 0.815, txt, fontsize=8.8, ha="center", va="bottom")

    cax = fig.add_axes([0.622, 0.268, 0.148, 0.016])
    cb = fig.colorbar(ScalarMappable(norm=enorm, cmap=ERR_CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_label("reprojection error (px)", fontsize=6.6, color="#1a1a1a",
                 labelpad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.2, color="#333333")
    # round the ticks inward: a tick outside [evmin, evmax] makes matplotlib
    # widen the bar and leave a blank sliver where no colour is defined
    tlo = np.ceil(evmin * 10.0) / 10.0
    thi = np.floor(evmax * 10.0) / 10.0
    cb.set_ticks(np.round(np.linspace(tlo, thi, 3), 2))
    cb.ax.set_xlim(evmin, evmax)

    # ---- track-length mix --------------------------------------------------
    axt = fig.add_axes([0.062, 0.115, 0.855, 0.070])
    axt.set_facecolor("#f6f6f7")
    labels = ["2 views", "3 views", "4 views", "5 views", "$\\geq$ 6 views"]
    buckets = [tlen == 2, tlen == 3, tlen == 4, tlen == 5, tlen >= 6]
    share = np.array([b.mean() for b in buckets]) * 100.0
    left = 0.0
    for i, (lab, s) in enumerate(zip(labels, share)):
        col = track_color(2.0 + i * 1.25)
        axt.barh(0, s, left=left, height=0.62, color=col, edgecolor="white",
                 linewidth=0.9)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        axt.text(left + s / 2.0, 0, f"{lab}\n{s:.0f}%", ha="center",
                 va="center", fontsize=6.6, linespacing=1.35,
                 color="#1a1a1a" if lum > 0.6 else "white")
        left += s
    axt.set_xlim(0, 100)
    axt.set_ylim(-0.55, 0.55)
    axt.axis("off")
    axt.text(-0.5, 0, "track\nlength", ha="right", va="center", fontsize=6.8,
             color="#333333", linespacing=1.35)
    axt.text(100.6, 0, f"{P.shape[0]:,} tracks\n"
             f"mean {tlen.mean():.1f} views",
             ha="left", va="center", fontsize=6.8, color="#555555",
             linespacing=1.35)


    savefig(fig, "1118_wm_rec_sfm_tracks.png")


if __name__ == "__main__":
    main()
