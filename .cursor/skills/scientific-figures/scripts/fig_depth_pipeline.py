"""Monocular depth-to-BEV pipeline, three samples x five stages.

Each stage sits in its own dashed rounded frame -- Input View, Depth
Estimation, Point Cloud, Reprojection, BEV Heat Map -- with thick arrows in
between and the final BEV column emphasised in red.  Every stage is derived
from the same ray-traced street render, so the five thumbnails of a row are
genuinely consistent with each other.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.cm import ScalarMappable

from pubstyle import set_pub_style, savefig

SEED = 913
DMAX = 55.0
SKY = np.array([0.78, 0.85, 0.92])

STAGES = ["Input View", "Depth Estimation", "Point Cloud", "Reprojection",
          "BEV Heat Map"]


# ---------------------------------------------------------------- scene defs
def make_scene(rng, style):
    boxes, spheres = [], []

    def box(cx, cy, cz, sx, sy, sz, yaw, col, kind="box"):
        boxes.append(dict(c=np.array([cx, cy, cz]), s=np.array([sx, sy, sz]),
                          yaw=yaw, col=np.array(
                              plt.matplotlib.colors.to_rgb(col)), kind=kind))

    facade = ["#c3a684", "#b09a86", "#c9b49b", "#a89279", "#bda78c"]
    if style == 0:
        rows = [(10, 17.0, 18, 10, 10.0), (32, 17.5, 18, 11, 13.5),
                (8, -17.0, 18, 10, 12.0), (30, -17.5, 18, 11, 9.0)]
        cars = [(13.0, 3.3, 0.03), (24.0, 3.1, -0.03), (16.0, -3.4, 3.12),
                (25.0, -3.3, 3.14), (33.0, -3.4, 3.13)]
        trees = [(6, 9.8), (18, 9.8), (30, 9.8), (12, -9.8), (26, -9.8)]
    elif style == 1:
        rows = [(12, 15.0, 22, 9, 14.0), (36, 16.0, 16, 10, 8.5),
                (14, -16.0, 20, 10, 9.5), (34, -16.5, 16, 10, 12.5)]
        cars = [(9.0, 3.2, 0.05), (19.5, 3.4, 0.02), (28.0, 3.2, -0.04),
                (15.0, -3.3, 3.10), (26.0, -3.5, 3.14)]
        trees = [(9, 9.8), (21, 9.8), (33, 9.8), (7, -9.8), (19, -9.8),
                 (31, -9.8)]
    else:
        rows = [(9, 16.0, 16, 10, 11.5), (28, 16.5, 20, 11, 15.0),
                (11, -16.5, 20, 11, 8.0), (33, -16.0, 16, 10, 13.0)]
        cars = [(11.5, 3.4, -0.02), (26.0, 3.2, 0.04), (14.0, -3.3, 3.13),
                (22.0, -3.5, 3.11), (30.0, -3.2, 3.14)]
        trees = [(4, 9.8), (16, 9.8), (28, 9.8), (10, -9.8), (22, -9.8),
                 (34, -9.8)]

    for i, (cx, cy, sx, sy, h) in enumerate(rows):
        box(cx, cy, h / 2, sx, sy, h, 0.0, facade[i % len(facade)])
    box(62.0, 3.0, 6.5, 14, 44, 13.0, 0.0, "#b6a893")

    paint = ["#cbced3", "#9b4046", "#6b7280", "#e8e8e8", "#5c6f8c", "#a7adb3"]
    for i, (cx, cy, yaw) in enumerate(cars):
        box(cx, cy, 0.79, 4.5, 1.95, 1.58, yaw, paint[i % len(paint)],
            kind="car")
    for cx, cy in trees:
        box(cx, cy, 1.6, 0.38, 0.38, 3.2, 0.0, "#6b4b32")
        spheres.append(dict(c=np.array([cx, cy, 4.8]), r=2.15,
                            col=np.array([0.29, 0.50, 0.28])))
    for cx, cy in [(rng.uniform(8, 28), 8.7), (rng.uniform(8, 28), -8.7)]:
        box(cx, cy, 0.88, 0.60, 0.55, 1.76, 0.0, "#d98b6a", kind="ped")
    return boxes, spheres


# ------------------------------------------------------------- ray-tracing
def _slab(o, D, b):
    c, s = np.cos(-b["yaw"]), np.sin(-b["yaw"])
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    oo, DD = R @ (o - b["c"]), D @ R.T
    DD = np.where(np.abs(DD) < 1e-9, 1e-9, DD)
    h = b["s"] / 2.0
    t1, t2 = (-h - oo) / DD, (h - oo) / DD
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmax > 0.0)
    t = np.where(tmin > 0.0, tmin, tmax)
    return np.where(hit & (t > 0.0), t, np.inf), R


def render(cam, W, H, foc, boxes, spheres, rng):
    """Ray-trace the scene -> (rgb, depth, world points, valid mask)."""
    u = (np.arange(W) + 0.5 - W / 2.0)
    v = (np.arange(H) + 0.5 - H * 0.46)
    U, V = np.meshgrid(u, v)
    D = np.stack([np.full(U.shape, foc), -U, -V], axis=-1)
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    Df = D.reshape(-1, 3)

    t = np.full(Df.shape[0], np.inf)
    prim = np.full(Df.shape[0], -1)
    dz = np.where(np.abs(Df[:, 2]) < 1e-9, 1e-9, Df[:, 2])
    tg = (0.0 - cam[2]) / dz
    tg = np.where((Df[:, 2] < -1e-4) & (tg > 0), tg, np.inf)
    upd = tg < t
    t[upd], prim[upd] = tg[upd], 0
    rots = []
    for i, b in enumerate(boxes):
        tb, R = _slab(cam, Df, b)
        rots.append(R)
        upd = tb < t
        t[upd], prim[upd] = tb[upd], i + 1
    off = len(boxes) + 1
    for j, sp in enumerate(spheres):
        L = cam - sp["c"]
        bq = 2.0 * (Df @ L)
        disc = bq * bq - 4.0 * (L @ L - sp["r"] ** 2)
        ts = (-bq - np.sqrt(np.maximum(disc, 0.0))) / 2.0
        ts = np.where((disc > 0) & (ts > 0), ts, np.inf)
        ts = ts * (1.0 + rng.normal(0, 0.010, ts.size))   # leafy silhouette
        upd = ts < t
        t[upd], prim[upd] = ts[upd], off + j

    valid = np.isfinite(t) & (t < DMAX)
    P = cam + Df * np.where(valid, t, DMAX)[:, None]

    col = np.zeros((Df.shape[0], 3))
    nrm = np.zeros((Df.shape[0], 3))
    g = prim == 0
    tile = ((np.floor(P[:, 0] / 3.0) + np.floor(P[:, 1] / 3.0)) % 2)
    road = np.abs(P[:, 1]) < 8.0
    base = np.where(road[:, None], np.array([0.44, 0.46, 0.48]),
                    np.array([0.60, 0.59, 0.55]))
    base = base * (0.94 + 0.06 * tile)[:, None]
    lane = (np.abs(np.abs(P[:, 1]) - 4.0) < 0.16) | (np.abs(P[:, 1]) < 0.18)
    base = np.where((lane & road)[:, None], np.array([0.84, 0.82, 0.60]), base)
    col[g], nrm[g] = base[g], np.array([0.0, 0.0, 1.0])

    for i, b in enumerate(boxes):
        m = prim == i + 1
        if not m.any():
            continue
        q = (P[m] - b["c"]) @ rots[i].T
        f = np.abs(q) / (b["s"] / 2.0)
        ax_ = np.argmax(f, axis=1)
        n = np.zeros_like(q)
        n[np.arange(q.shape[0]), ax_] = np.sign(q[np.arange(q.shape[0]), ax_])
        nrm[m] = n @ rots[i]
        c = np.tile(b["col"], (m.sum(), 1))
        if b["kind"] == "box":                       # window bands on facades
            band = ((np.floor(P[m][:, 2] / 1.9) % 2) == 1) & (ax_ != 2)
            c = np.where(band[:, None], c * 0.72, c)
        col[m] = c
    for j, sp in enumerate(spheres):
        m = prim == off + j
        if not m.any():
            continue
        n = P[m] - sp["c"]
        nrm[m] = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        col[m] = sp["col"] * (0.80 + 0.40 * rng.random((m.sum(), 1)))

    light = np.array([0.35, 0.42, 0.84])
    light /= np.linalg.norm(light)
    lam = 0.66 + 0.40 * np.clip(nrm @ light, 0.0, 1.0)
    rgb = np.clip(col * lam[:, None], 0.0, 1.0)
    fog = np.clip((t - 18.0) / 70.0, 0.0, 0.48)[:, None]
    rgb = rgb * (1.0 - fog) + SKY * fog
    rgb[~valid] = SKY
    return (rgb.reshape(H, W, 3), t.reshape(H, W), P.reshape(H, W, 3),
            valid.reshape(H, W))


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    FW, FH = 7.9, 4.95
    L, R = 0.026, 0.012
    gapx, gapy = 0.0215, 0.020
    top, bot = 0.898, 0.072
    cw = (1.0 - L - R - 4 * gapx) / 5.0
    rh = (top - bot - 2 * gapy) / 3.0
    padx, pady = 0.0085, 0.0135
    aw, ah = cw - 2 * padx, rh - 2 * pady
    AR = (aw * FW) / (ah * FH)
    W = 250
    H = int(round(W / AR))

    fig = plt.figure(figsize=(FW, FH))

    def cell(r, c):
        return (L + c * (cw + gapx), top - (r + 1) * rh - r * gapy, cw, rh)

    dcm = LinearSegmentedColormap.from_list(
        "depth", ["#f7f4b0", "#f0a02e", "#c9432f", "#7b2a6b", "#2a2a72",
                  "#101033"])
    bevcm = LinearSegmentedColormap.from_list(
        "bev", ["#fff1c9", "#ffcf78", "#f7963a", "#d94f34", "#8c1d18"])
    bevcm.set_bad("#e6e6e6")

    for r in range(3):
        cam = np.array([0.0, rng.uniform(-0.8, 0.8), 1.55 + 0.25 * r])
        boxes, spheres = make_scene(rng, r)
        rgb, dep, P, valid = render(cam, W, H, 0.86 * W, boxes, spheres, rng)

        x0, y0, _, _ = cell(r, 0)
        ax = fig.add_axes([x0 + padx, y0 + pady, aw, ah])
        ax.imshow(rgb, interpolation="bilinear")
        _clean(ax)

        x0, y0, _, _ = cell(r, 1)
        ax = fig.add_axes([x0 + padx, y0 + pady, aw, ah])
        dshow = np.ma.masked_where(~valid, dep)
        dcm.set_bad("#e9ecef")
        ax.imshow(dshow, cmap=dcm, vmin=3.0, vmax=DMAX,
                  interpolation="bilinear")
        _clean(ax)

        # ---- stage 3: sparse 3-D cloud lifted from the depth map ----------
        x0, y0, _, _ = cell(r, 2)
        ax = fig.add_axes([x0 + padx - 0.033, y0 + pady - 0.064,
                           aw + 0.066, ah + 0.128], projection="3d")
        # a 3-D axes keeps an opaque background patch even with the axis off,
        # which would white out the dashed stage frame underneath
        ax.patch.set_alpha(0.0)
        step = 2
        Q = P[::step, ::step][valid[::step, ::step]]
        C = rgb[::step, ::step][valid[::step, ::step]]
        keep = (Q[:, 0] < 31.0) & (np.abs(Q[:, 1]) < 13.5) & (Q[:, 2] < 10.0)
        Q, C = Q[keep], np.clip(C[keep] * 1.04, 0, 1)
        o = np.argsort(-Q[:, 0])
        ax.scatter(Q[o, 0], Q[o, 1], Q[o, 2], s=1.2, c=C[o], marker=".",
                   linewidths=0, depthshade=False, rasterized=True)
        ax.set_xlim(0, 31)
        ax.set_ylim(-13.5, 13.5)
        ax.set_zlim(0, 10)
        ax.set_box_aspect((1.0, 0.95, 0.46))
        ax.view_init(elev=20, azim=-70)
        ax.set_axis_off()

        # ---- stage 4: sparse reprojection onto the image ------------------
        x0, y0, _, _ = cell(r, 3)
        ax = fig.add_axes([x0 + padx, y0 + pady, aw, ah])
        ax.imshow(rgb * 0.38 + 0.62, interpolation="bilinear")
        rows_, cols_ = np.mgrid[3:H:7, 0:W:3]           # LiDAR-like scanlines
        mm = valid[rows_, cols_]
        ax.scatter(cols_[mm], rows_[mm], s=1.7, marker=".", linewidths=0,
                   c=dep[rows_, cols_][mm], cmap=dcm, vmin=3.0, vmax=DMAX,
                   alpha=0.95)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        _clean(ax)

        # ---- stage 5: BEV height / occupancy heat map ---------------------
        x0, y0, _, _ = cell(r, 4)
        ax = fig.add_axes([x0 + padx, y0 + pady, aw, ah])
        fwd = (0.0, 36.0)
        lat = 0.5 * (fwd[1] - fwd[0]) / AR
        Pv = P[valid]
        nb = 58
        Hh, _, _ = np.histogram2d(
            Pv[:, 0], Pv[:, 1], bins=[nb, int(nb / AR) if AR < 1 else nb],
            range=[[fwd[0], fwd[1]], [-lat, lat]],
            weights=np.clip(Pv[:, 2], 0.0, 8.0))
        Cc, _, _ = np.histogram2d(
            Pv[:, 0], Pv[:, 1], bins=[nb, int(nb / AR) if AR < 1 else nb],
            range=[[fwd[0], fwd[1]], [-lat, lat]])
        M = np.ma.masked_where(Cc < 1, np.divide(Hh, np.maximum(Cc, 1.0)))
        ax.imshow(M.T, origin="lower", cmap=bevcm, vmin=0.0, vmax=7.0,
                  extent=[fwd[0], fwd[1], -lat, lat], aspect="auto",
                  interpolation="nearest")
        ax.set_xlim(*fwd)
        ax.set_ylim(lat, -lat)
        _clean(ax)
        ax.plot([0.8], [0.0], marker=">", ms=4.5, mfc="#ffffff",
                mec="#222222", mew=0.7, zorder=5)
        for rr in (10, 20, 30):
            th = np.linspace(-1.05, 1.05, 60)
            ax.plot(rr * np.cos(th), rr * np.sin(th), color="#8a8a8a",
                    lw=0.4, ls=(0, (2.4, 1.8)), zorder=4)

        fig.text(L - 0.010, y0 + rh / 2.0, f"Sample {r + 1}", fontsize=7.6,
                 rotation=90, va="center", ha="right", color="#444444")

    # ---- dashed stage frames, headers and arrows --------------------------
    for c in range(5):
        last = c == 4
        ec = "#b23a48" if last else "#9a9a9a"
        for r in range(3):
            x0, y0, w, h = cell(r, c)
            fig.patches.append(FancyBboxPatch(
                (x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                transform=fig.transFigure, facecolor="none", edgecolor=ec,
                linewidth=1.05 if last else 0.85,
                linestyle=(0, (3.2, 2.0)), zorder=6))
            if c < 4:
                xa = x0 + w
                fig.patches.append(FancyArrowPatch(
                    (xa + 0.0035, y0 + h / 2), (xa + gapx - 0.0035, y0 + h / 2),
                    transform=fig.transFigure,                     arrowstyle="-|>",
                    mutation_scale=10.5, linewidth=2.1, color="#5b6470",
                    zorder=7))
        x0, _, w, _ = cell(0, c)


    cax = fig.add_axes([L + cw + gapx + 0.020, 0.033, 0.115, 0.014])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(3, DMAX), cmap=dcm),
                      cax=cax, orientation="horizontal", ticks=[5, 30, 55])
    cb.set_label("depth (m)", fontsize=7.0, color="#1a1a1a", labelpad=1.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")

    cax2 = fig.add_axes([L + 4 * (cw + gapx) + 0.030, 0.033, 0.115, 0.014])
    cb2 = fig.colorbar(ScalarMappable(norm=Normalize(0, 7), cmap=bevcm),
                       cax=cax2, orientation="horizontal", ticks=[0, 3.5, 7])
    cb2.set_label("BEV mean height (m)", fontsize=7.0, color="#1a1a1a",
                  labelpad=1.2)
    cb2.outline.set_linewidth(0.6)
    cb2.outline.set_edgecolor("#333333")
    cb2.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, color="#333333")


    savefig(fig, "285_depth_pipeline.png")


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


if __name__ == "__main__":
    main()
