"""3-D LiDAR point-cloud scene overview with 3-D object detections.

A synthetic urban street is ray-cast from a roof-mounted 40-channel LiDAR so
the cloud shows the characteristic ring texture and occlusion shadows.  The
large 3-D view carries height-coloured points plus oriented 3-D boxes; the
narrow right column holds two subordinate panels (BEV projection, height
histogram).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig

SEED = 20250312
SENSOR = np.array([0.0, 0.0, 1.90])
RMAX = 44.0

SEM = {"ground": "#9aa0a6", "building": "#a9835f", "vegetation": "#4f8f4f",
       "vehicle": "#3b5b92", "pedestrian": "#c9a227", "other": "#8e6bbf"}


# ------------------------------------------------------------------ geometry
def build_scene():
    """Return (boxes, spheres, targets) describing a city block."""
    boxes, spheres = [], []

    def box(cx, cy, cz, sx, sy, sz, yaw, label):
        boxes.append(dict(c=np.array([cx, cy, cz]),
                          s=np.array([sx, sy, sz]), yaw=yaw, label=label))

    # --- building rows on both sides of the road ---------------------------
    left = [(-11, 14.5, 20, 9, 13.5), (10, 15.5, 18, 11, 17.5),
            (29, 14.0, 16, 8, 10.5)]
    right = [(-8, -15.0, 18, 10, 11.0), (12, -14.5, 20, 9, 16.0),
             (31, -15.5, 16, 11, 12.5)]
    # cross-street block that closes the vista at the far end
    left += [(45, 4.0, 14, 30, 15.0)]
    for cx, cy, sx, sy, h in left + right:
        box(cx, cy, h / 2, sx, sy, h, 0.0, "building")

    # --- parked / moving vehicles on the road ------------------------------
    cars = [(7.0, 3.2, 0.04), (16.5, 3.4, -0.02), (26.0, 3.0, 0.05),
            (4.0, -3.3, 3.10), (13.5, -3.5, 3.14), (23.0, -3.2, 3.12),
            (33.0, 3.1, 0.0), (-6.0, -3.4, 3.13)]
    targets = []
    for cx, cy, yaw in cars:
        sx, sy, sz = 4.5, 1.95, 1.62
        box(cx, cy, sz / 2, sx, sy, sz, yaw, "vehicle")
        targets.append(dict(c=np.array([cx, cy, sz / 2]),
                            s=np.array([sx, sy, sz]), yaw=yaw, cls="vehicle"))
    # a bus, to break the size monotony
    box(31.0, -3.3, 1.55, 8.5, 2.5, 3.10, 3.14, "vehicle")
    targets.append(dict(c=np.array([31.0, -3.3, 1.55]),
                        s=np.array([8.5, 2.5, 3.10]), yaw=3.14, cls="vehicle"))

    # --- pedestrians on the sidewalk ---------------------------------------
    for cx, cy in [(10.0, 8.4), (11.1, 9.0), (20.0, -8.6), (29.0, 8.8)]:
        box(cx, cy, 0.88, 0.62, 0.55, 1.76, 0.0, "pedestrian")
        targets.append(dict(c=np.array([cx, cy, 0.88]),
                            s=np.array([0.80, 0.75, 1.80]), yaw=0.0,
                            cls="pedestrian"))

    # --- street trees: trunk box + foliage sphere --------------------------
    for cx, cy in [(1.0, 9.6), (12.0, 9.6), (23.0, 9.6), (34.0, 9.6),
                   (-4.0, -9.6), (7.0, -9.6), (18.0, -9.6), (29.0, -9.6)]:
        box(cx, cy, 1.5, 0.42, 0.42, 3.0, 0.0, "vegetation")
        spheres.append(dict(c=np.array([cx, cy, 4.5]), r=2.05,
                            label="vegetation"))
    return boxes, spheres, targets


def _ray_box(o, D, b):
    c, s = np.cos(-b["yaw"]), np.sin(-b["yaw"])
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    oo = R @ (o - b["c"])
    DD = D @ R.T
    DD = np.where(np.abs(DD) < 1e-9, 1e-9, DD)
    h = b["s"] / 2.0
    t1 = (-h - oo) / DD
    t2 = (h - oo) / DD
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmax > 0.0)
    t = np.where(tmin > 0.0, tmin, tmax)
    return np.where(hit & (t > 0.0), t, np.inf)


def _ray_sphere(o, D, sp):
    L = o - sp["c"]
    b = 2.0 * (D @ L)
    cc = L @ L - sp["r"] ** 2
    disc = b * b - 4.0 * cc
    ok = disc > 0.0
    t = (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0
    return np.where(ok & (t > 0.0), t, np.inf)


def scan(boxes, spheres, n_ch=64, n_az=1800, seed=SEED):
    """Simulate a spinning LiDAR; returns points, class ids and range."""
    rng = np.random.default_rng(seed)
    el = np.deg2rad(np.linspace(-25.0, 21.0, n_ch))
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    E, A = np.meshgrid(el, az, indexing="ij")
    E = E.ravel() + rng.normal(0.0, 0.0012, E.size)
    A = A.ravel()
    D = np.column_stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A),
                         np.sin(E)])

    labels = list(SEM)
    t_best = np.full(D.shape[0], np.inf)
    cls = np.zeros(D.shape[0], dtype=np.int8)

    dz = np.where(np.abs(D[:, 2]) < 1e-9, 1e-9, D[:, 2])
    tg = (0.0 - SENSOR[2]) / dz
    tg = np.where((D[:, 2] < -1e-4) & (tg > 0.0), tg, np.inf)
    upd = tg < t_best
    t_best[upd], cls[upd] = tg[upd], labels.index("ground")

    for b in boxes:
        t = _ray_box(SENSOR, D, b)
        upd = t < t_best
        t_best[upd], cls[upd] = t[upd], labels.index(b["label"])
    for sp in spheres:                       # porous foliage: jitter the hit
        t = _ray_sphere(SENSOR, D, sp)
        t = t * (1.0 + rng.normal(0.0, 0.045, t.size))
        keep = rng.random(t.size) < 0.72     # some rays pass through leaves
        t = np.where(keep, t, np.inf)
        upd = t < t_best
        t_best[upd], cls[upd] = t[upd], labels.index("vegetation")

    ok = np.isfinite(t_best) & (t_best < RMAX)
    t_best, D, cls = t_best[ok], D[ok], cls[ok]
    P = SENSOR + D * t_best[:, None]
    P += rng.normal(0.0, 0.022, P.shape)                 # range noise
    # gentle road camber / undulation on the ground returns
    g = cls == 0
    P[g, 2] += (0.055 * np.sin(P[g, 0] * 0.09)
                + 0.045 * np.cos(P[g, 1] * 0.16) - 0.010 * np.abs(P[g, 1]))
    r = np.linalg.norm(P - SENSOR, axis=1)
    return P, cls, r


def clip_fov(P, cls, r, xlim, ylim):
    m = ((P[:, 0] > xlim[0]) & (P[:, 0] < xlim[1])
         & (P[:, 1] > ylim[0]) & (P[:, 1] < ylim[1]))
    return P[m], cls[m], r[m]


def box_corners(c, s, yaw):
    h = s / 2.0
    k = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    v = k * h
    co, si = np.cos(yaw), np.sin(yaw)
    R = np.array([[co, -si, 0.0], [si, co, 0.0], [0.0, 0.0, 1.0]])
    return v @ R.T + c


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_box3d(ax, c, s, yaw, color, lw=1.4, z=40):
    v = box_corners(c, s, yaw)
    for i, j in EDGES:
        ax.plot(*zip(v[i], v[j]), color=color, lw=lw, zorder=z,
                solid_capstyle="round")
    # heading marker: a short spear out of the front face
    d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    front = c + d * s[0] / 2.0
    tip = front + d * max(1.0, s[0] * 0.28)
    ax.plot(*zip(front, tip), color=color, lw=lw, zorder=z)
    ax.plot([tip[0]], [tip[1]], [tip[2]], marker="o", ms=2.0, color=color,
            zorder=z)


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    boxes, spheres, targets = build_scene()
    P, cls, rng_ = scan(boxes, spheres)

    XL, YL, ZL = (-15.0, 39.0), (-20.0, 20.0), (0.0, 17.0)
    P, cls, rng_ = clip_fov(P, cls, rng_, XL, YL)
    # mplot3d does not clip scatter against the axis limits, so anything above
    # ZL would be drawn floating outside the box frame
    zk = (P[:, 2] >= ZL[0]) & (P[:, 2] <= ZL[1])
    P, cls, rng_ = P[zk], cls[zk], rng_[zk]

    # far points get thinned and shrunk: mimics real density fall-off
    g = np.random.default_rng(SEED + 1)
    keep = g.random(P.shape[0]) < np.clip(1.10 - 0.010 * rng_, 0.45, 1.0)
    P, cls, rng_ = P[keep], cls[keep], rng_[keep]
    order = np.argsort(-rng_)                # far first, near painted on top
    P, cls, rng_ = P[order], cls[order], rng_[order]

    ZC = 13.0
    hz = np.clip(P[:, 2], 0.0, ZC)
    norm = Normalize(-1.0, ZC)
    size = np.clip(3.2 - 0.048 * rng_, 0.55, 3.2)
    alpha = np.clip(0.95 - 0.0075 * rng_, 0.50, 0.95)
    rgba = plt.get_cmap("viridis")(norm(hz))
    rgba[:, 3] = alpha

    fig = plt.figure(figsize=(7.8, 4.15))
    ax = fig.add_axes([-0.060, -0.090, 0.865, 1.135], projection="3d")
    ax.computed_zorder = False
    # a 3-D axes keeps an opaque background patch, which would punch a white
    # hole in the grey backing plate it overlaps
    ax.patch.set_alpha(0.0)

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=size, c=rgba, marker=".",
               linewidths=0, depthshade=False, zorder=5, rasterized=True)

    for t in targets:
        col = SEM[t["cls"]]
        pad = np.array([0.30, 0.30, 0.22])
        draw_box3d(ax, t["c"], t["s"] + pad, t["yaw"], col,
                   lw=1.5 if t["cls"] == "vehicle" else 1.2, z=40)

    # ego sensor marker
    ax.plot([SENSOR[0]], [SENSOR[1]], [SENSOR[2]], marker="^", ms=7,
            mfc="#b23a48", mec="#5c1113", mew=0.7, ls="none", zorder=45)
    ax.plot([0, 0], [0, 0], [0, SENSOR[2]], color="#b23a48", lw=0.8,
            alpha=0.8, zorder=45)
    ax.plot([0, 0], [0, 0], [SENSOR[2], 9.0], color="#b23a48", lw=0.6,
            ls=(0, (2, 1.6)), alpha=0.8, zorder=45)
    ax.text(0.0, 0.0, 9.6, "ego LiDAR", fontsize=6.6, color="#8c2c38",
            ha="center", va="bottom", zorder=46,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.82))

    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_zlim(*ZL)
    ax.set_box_aspect((1.0, 0.80, 0.46))
    ax.view_init(elev=25, azim=-66)
    ax.set_xlabel("x  forward (m)", labelpad=-3, fontsize=7.5)
    ax.set_ylabel("y  lateral (m)", labelpad=-4, fontsize=7.5)
    ax.set_zlabel("z (m)", labelpad=-7, fontsize=7.5)
    ax.set_xticks([-10, 5, 20, 35])
    ax.set_yticks([-20, 0, 20])
    ax.set_zticks([0, 8, 16])
    ax.tick_params(axis="both", pad=-3.5, labelsize=6.5)
    ax.tick_params(axis="z", pad=-2.0, labelsize=6.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#d0d0d0")
        axis.pane.set_linewidth(0.5)
        axis._axinfo["grid"].update(color="#e2e2e2", linewidth=0.45)

    # --- colorbar for the height ramp --------------------------------------
    cax = fig.add_axes([0.055, 0.105, 0.145, 0.019])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=cax,
                      orientation="horizontal", ticks=[0, 6, 12])
    cb.set_label("point height z (m)", fontsize=7.0, color="#1a1a1a",
                 labelpad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")

    handles = [
        Line2D([], [], color=SEM["vehicle"], lw=1.5, label="Vehicle box"),
        Line2D([], [], color=SEM["pedestrian"], lw=1.2, label="Pedestrian box"),
        Line2D([], [], color="#b23a48", marker="^", ls="none", ms=5,
               mec="#5c1113", mew=0.6, label="Ego LiDAR"),
    ]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=min(len(handles), 5), frameon=False)
    leg.get_frame().set_linewidth(0.55)

    # --- subordinate right column (grey backing plate) ---------------------
    fig.patches.append(plt.Rectangle(
        (0.752, 0.030), 0.243, 0.900, transform=fig.transFigure,
        facecolor="#f4f4f5", edgecolor="#dedede", linewidth=0.6, zorder=-1))

    axb = fig.add_axes([0.798, 0.455, 0.182, 0.400])
    axb.scatter(P[:, 1], P[:, 0], s=0.45, c=rgba, marker=".", linewidths=0,
                rasterized=True)
    for t in targets:
        v = box_corners(t["c"], t["s"] + 0.3, t["yaw"])[:4]
        axb.plot(np.r_[v[:, 1], v[0, 1]], np.r_[v[:, 0], v[0, 0]],
                 color=SEM[t["cls"]], lw=0.7)
    axb.set_xlim(YL[1], YL[0])
    axb.set_ylim(*XL)
    axb.set_aspect("equal")
    axb.set_xticks([])
    axb.set_yticks([])
    for sp in axb.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_color("#999999")
    axb.set_title("BEV projection", fontsize=7.2, pad=2.5)
    axb.text(0.04, 0.02, "20 m", transform=axb.transAxes, fontsize=6.0,
             color="#555555", ha="left", va="bottom")
    axb.plot([YL[1] - 1.5, YL[1] - 21.5], [XL[0] + 3.6, XL[0] + 3.6],
             color="#555555", lw=1.0)

    axh = fig.add_axes([0.798, 0.135, 0.182, 0.215])
    edges = np.linspace(0.0, ZC, 27)
    cnt, _ = np.histogram(P[:, 2], bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    axh.barh(ctr, cnt / 1000.0, height=0.40,
             color=plt.get_cmap("viridis")(norm(ctr)), edgecolor="none")
    axh.set_ylim(0, ZC)
    axh.set_yticks([0, 6, 12])
    axh.set_xlabel("points ($\\times10^3$)", fontsize=6.6, labelpad=1)
    axh.set_ylabel("z (m)", fontsize=6.6, labelpad=1)
    axh.tick_params(labelsize=6.0, length=2.0)
    axh.set_title("Height histogram", fontsize=7.2, pad=2.5)
    axh.grid(axis="x", lw=0.4, color="#e0e0e0")
    axh.set_axisbelow(True)


    print(f"points={P.shape[0]}  boxes={len(targets)}")
    savefig(fig, "283_pointcloud_scene.png")


if __name__ == "__main__":
    main()
