"""BEV 3-D detection comparison: camera view, LiDAR BEV and three box sets.

2 x 3 grid -- Camera view | LiDAR BEV | GT boxes / Baseline | Ours | Error.
All BEV panels share one grid, one range and one scale bar; missed objects are
dashed red boxes and false positives are red crosses.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig

SEED = 4711
CAM = np.array([0.0, 0.0, 1.65])
SENSOR = np.array([0.0, 0.0, 1.90])
RMAX = 46.0

SEM = {"ground": "#9aa0a6", "building": "#a9835f", "vegetation": "#4f8f4f",
       "vehicle": "#3b5b92", "pedestrian": "#c9a227", "other": "#8e6bbf"}

AR = 1.108                                   # panel aspect (w / h)
FWD = (-3.0, 37.0)                           # BEV forward range (m)
LAT = 0.5 * (FWD[1] - FWD[0]) * AR           # BEV half-width (m)
IMW, IMH = 1108.0, 1000.0
FOC, CX, CY = 620.0, IMW / 2.0, IMH * 0.46


# ------------------------------------------------------------------ geometry
def build_scene():
    boxes, spheres, targets = [], [], []

    def box(cx, cy, cz, sx, sy, sz, yaw, label):
        boxes.append(dict(c=np.array([cx, cy, cz]), s=np.array([sx, sy, sz]),
                          yaw=yaw, label=label))

    for cx, cy, sx, sy, h in [(9, 19.0, 16, 12, 9.5), (28, 19.5, 16, 13, 12.0),
                              (7, -19.0, 16, 12, 11.0), (26, -19.5, 16, 13, 8.5),
                              (66, 3.0, 14, 44, 12.0)]:
        box(cx, cy, h / 2, sx, sy, h, 0.0, "building")

    cars = [(12.0, 3.3, 0.03, 4.6, 1.95, 1.58), (21.0, 3.1, -0.03, 4.4, 1.92, 1.52),
            (30.5, 3.4, 0.02, 4.5, 1.95, 1.60), (9.5, -3.4, 3.12, 4.5, 1.95, 1.55),
            (18.5, -3.2, 3.14, 4.7, 2.00, 1.62), (26.5, -3.5, 3.11, 4.4, 1.90, 1.50),
            (13.5, 7.4, 0.00, 4.3, 1.88, 1.48)]
    body = ["#c9ccd1", "#8b2f34", "#3a3f46", "#e6e6e6", "#4a5a72", "#9aa0a6",
            "#b8b2a4"]
    for (cx, cy, yaw, sx, sy, sz), col in zip(cars, body):
        box(cx, cy, sz / 2, sx, sy, sz, yaw, "vehicle")
        boxes[-1]["paint"] = col
        targets.append(dict(c=np.array([cx, cy]), s=np.array([sx, sy]),
                            yaw=yaw, cls="vehicle"))
    box(33.0, -3.3, 1.55, 8.0, 2.5, 3.10, 3.14, "vehicle")
    boxes[-1]["paint"] = "#d8d3c6"
    targets.append(dict(c=np.array([33.0, -3.3]), s=np.array([8.0, 2.5]),
                        yaw=3.14, cls="vehicle"))

    for cx, cy in [(9.0, 8.6), (10.2, 9.2), (23.0, -8.4), (16.0, 8.9)]:
        box(cx, cy, 0.88, 0.62, 0.55, 1.76, 0.0, "pedestrian")
        targets.append(dict(c=np.array([cx, cy]), s=np.array([0.80, 0.75]),
                            yaw=0.0, cls="pedestrian"))
    for cx, cy in [(5.0, -8.8), (19.0, 9.1)]:
        box(cx, cy, 0.72, 1.65, 0.55, 1.45, 0.0, "other")
        targets.append(dict(c=np.array([cx, cy]), s=np.array([1.80, 0.80]),
                            yaw=0.0, cls="cyclist"))

    for cx, cy in [(4.0, 9.8), (15.0, 9.8), (26.0, 9.8), (37.0, 9.8),
                   (2.0, -9.8), (13.0, -9.8), (24.0, -9.8), (35.0, -9.8)]:
        box(cx, cy, 1.6, 0.40, 0.40, 3.2, 0.0, "vegetation")
        spheres.append(dict(c=np.array([cx, cy, 4.7]), r=2.1))
    return boxes, spheres, targets


def _ray_box(o, D, b):
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
    return np.where(hit & (t > 0.0), t, np.inf)


def _ray_sphere(o, D, sp):
    L = o - sp["c"]
    b = 2.0 * (D @ L)
    disc = b * b - 4.0 * (L @ L - sp["r"] ** 2)
    t = (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0
    return np.where((disc > 0.0) & (t > 0.0), t, np.inf)


def scan(boxes, spheres, n_ch=56, n_az=1500, seed=SEED):
    rng = np.random.default_rng(seed)
    el = np.deg2rad(np.linspace(-24.0, 18.0, n_ch))
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    E, A = np.meshgrid(el, az, indexing="ij")
    E, A = E.ravel(), A.ravel()
    D = np.column_stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A),
                         np.sin(E)])
    t = np.full(D.shape[0], np.inf)
    dz = np.where(np.abs(D[:, 2]) < 1e-9, 1e-9, D[:, 2])
    tg = (0.0 - SENSOR[2]) / dz
    t = np.minimum(t, np.where((D[:, 2] < -1e-4) & (tg > 0), tg, np.inf))
    for b in boxes:
        t = np.minimum(t, _ray_box(SENSOR, D, b))
    for sp in spheres:
        ts = _ray_sphere(SENSOR, D, sp) * (1.0 + rng.normal(0, 0.05, t.size))
        ts = np.where(rng.random(t.size) < 0.7, ts, np.inf)
        t = np.minimum(t, ts)
    ok = np.isfinite(t) & (t < RMAX)
    P = SENSOR + D[ok] * t[ok][:, None] + rng.normal(0, 0.025, (ok.sum(), 3))
    return P


def corners2d(c, s, yaw):
    h = s / 2.0
    k = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * h
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return k @ R.T + c


def corners3d(c, s, yaw):
    h = s / 2.0
    k = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float) * h
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0],
                  [0, 0, 1.0]])
    return k @ R.T + c


FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
         (2, 3, 7, 6), (3, 0, 4, 7)]


def project(pts):
    q = np.atleast_2d(pts) - CAM
    d = np.maximum(q[:, 0], 1e-3)
    return np.column_stack([CX - FOC * q[:, 1] / d, CY - FOC * q[:, 2] / d]), \
        q[:, 0]


# ------------------------------------------------------- stylised camera view
def draw_camera(ax, boxes, spheres, targets):
    sky = np.linspace(0.0, 1.0, 256)[:, None]
    ax.imshow(sky, extent=[0, IMW, CY, 0], aspect="auto", zorder=1,
              cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                  "sky", ["#7fa6cd", "#a9c6e2", "#d8e5f0"]),
              interpolation="bilinear")
    ground = np.linspace(0.0, 1.0, 256)[:, None]
    ax.imshow(ground, extent=[0, IMW, IMH, CY], aspect="auto", zorder=1,
              cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                  "gnd", ["#b9b6ae", "#8f8d87"]), interpolation="bilinear")
    # road surface + lane markings
    road = np.array([[0.0, -8.0, 0.0], [0.0, 8.0, 0.0],
                     [90.0, 8.0, 0.0], [90.0, -8.0, 0.0]])
    uv, _ = project(road)
    ax.add_patch(Polygon(uv, closed=True, facecolor="#7c8085",
                         edgecolor="none", zorder=2))
    for lane_y in (-8.0, 8.0, -4.0, 4.0):
        q = np.array([[0.0, lane_y, 0.0], [90.0, lane_y, 0.0]])
        uv, _ = project(q)
        ax.plot(uv[:, 0], uv[:, 1], color="#dcdcdc", lw=1.0, zorder=3,
                alpha=0.75)
    for x0 in np.arange(2.0, 70.0, 6.0):
        q = np.array([[x0, -0.18, 0.0], [x0 + 3.0, -0.18, 0.0],
                      [x0 + 3.0, 0.18, 0.0], [x0, 0.18, 0.0]])
        uv, _ = project(q)
        ax.add_patch(Polygon(uv, closed=True, facecolor="#e3dfae",
                             edgecolor="none", zorder=3))
    for sy in (-8.6, 8.6):                       # sidewalk kerbs
        q = np.array([[0.0, sy, 0.0], [90.0, sy, 0.0],
                      [90.0, sy * 1.55, 0.16], [0.0, sy * 1.55, 0.16]])
        uv, _ = project(q)
        ax.add_patch(Polygon(uv, closed=True, facecolor="#a8a59f",
                             edgecolor="none", zorder=2))

    prims = []
    for b in boxes:
        base = b.get("paint") or {"building": "#bb9c7b", "vehicle": "#9aa0a6",
                                  "pedestrian": "#d98b6a",
                                  "vegetation": "#6b4b32",
                                  "other": "#7d8ea3"}[b["label"]]
        v = corners3d(b["c"], b["s"], b["yaw"])
        for f in FACES:
            quad = v[list(f)]
            uv, d = project(quad)
            if d.min() < 1.0:
                continue
            n = np.cross(quad[1] - quad[0], quad[2] - quad[0])
            n = n / (np.linalg.norm(n) + 1e-9)
            sh = 0.62 + 0.38 * abs(float(n @ np.array([0.42, 0.35, 0.84])))
            col = tuple(np.clip(np.array(plt.matplotlib.colors.to_rgb(base))
                                * sh, 0, 1))
            prims.append((d.mean(), uv, col))
    for sp in spheres:
        uv, d = project(sp["c"][None, :])
        if d[0] < 1.0:
            continue
        r = FOC * sp["r"] / d[0]
        th = np.linspace(0, 2 * np.pi, 40)
        poly = np.column_stack([uv[0, 0] + r * np.cos(th) * 1.05,
                                uv[0, 1] + r * np.sin(th) * 0.92])
        prims.append((d[0], poly, (0.31, 0.53, 0.30)))
    for d, poly, col in sorted(prims, key=lambda p: -p[0]):
        ax.add_patch(Polygon(poly, closed=True, facecolor=col,
                             edgecolor=(0, 0, 0, 0.16), linewidth=0.3,
                             zorder=4 + 1e-4 * (200.0 - d)))

    # aerial haze band around the horizon line
    haze = np.exp(-0.5 * ((np.linspace(0, IMH, 220) - CY) / 105.0) ** 2)
    rgba = np.zeros((220, 1, 4))
    rgba[..., :3] = 1.0
    rgba[..., 3] = 0.42 * haze[:, None]
    ax.imshow(rgba, extent=[0, IMW, IMH, 0], aspect="auto", zorder=20,
              interpolation="bilinear")

    # projected 3-D detection boxes
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    hcls = {"vehicle": 1.6, "pedestrian": 1.8, "cyclist": 1.5}
    for t in targets:
        if t["c"][0] < 5.0 or t["c"][0] > 36.0:
            continue
        c3 = np.array([t["c"][0], t["c"][1], hcls[t["cls"]] / 2 + 0.06])
        s3 = np.array([t["s"][0] + 0.2, t["s"][1] + 0.2, hcls[t["cls"]] + 0.2])
        v = corners3d(c3, s3, t["yaw"])
        uv, d = project(v)
        if d.min() < 1.5:
            continue
        col = {"vehicle": SEM["vehicle"], "pedestrian": SEM["pedestrian"],
               "cyclist": SEM["other"]}[t["cls"]]
        for i, j in edges:
            ax.plot(uv[[i, j], 0], uv[[i, j], 1], color=col, lw=1.0,
                    zorder=30, solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=1.9, foreground="white",
                                            alpha=0.55), pe.Normal()])
    ax.set_xlim(0, IMW)
    ax.set_ylim(IMH, 0)
    ax.set_aspect("equal")


# ------------------------------------------------------------- BEV utilities
def bev_frame(ax, title, bold=False):
    ax.set_xlim(LAT, -LAT)
    ax.set_ylim(*FWD)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#8a8a8a")
    ax.set_title(title, fontsize=8.0, pad=2.6,
                 fontweight="bold" if bold else "normal")


def bev_grid(ax):
    for r in (10, 20, 30):
        th = np.linspace(-np.pi / 2, np.pi / 2, 120)
        ax.plot(r * np.sin(th), r * np.cos(th), color="#c8c8c8", lw=0.4,
                ls=(0, (3, 2)), zorder=1)
    ax.plot([0, 0], [0, FWD[1]], color="#c8c8c8", lw=0.4, ls=(0, (3, 2)),
            zorder=1)
    ax.plot([0], [0], marker="^", ms=4.5, mfc="#b23a48", mec="#5c1113",
            mew=0.6, zorder=8)


def draw_bev_box(ax, c, s, yaw, color, lw=1.0, ls="-", fill=0.0, z=6):
    v = corners2d(np.asarray(c, float), np.asarray(s, float), yaw)
    ax.add_patch(Polygon(v[:, ::-1], closed=True, facecolor=color if fill else
                         "none", alpha=fill if fill else 1.0,
                         edgecolor="none" if fill else color, linewidth=0,
                         zorder=z - 1))
    ax.add_patch(Polygon(v[:, ::-1], closed=True, facecolor="none",
                         edgecolor=color, linewidth=lw, linestyle=ls,
                         zorder=z))
    d = np.array([np.cos(yaw), np.sin(yaw)])
    front = np.asarray(c, float) + d * s[0] / 2.0
    tip = front + d * 1.35
    ax.plot([front[1], tip[1]], [front[0], tip[0]], color=color, lw=lw,
            zorder=z, solid_capstyle="round")


# -------------------------------------------------------------------- figure
def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    boxes, spheres, targets = build_scene()
    P = scan(boxes, spheres)

    inview = [t for t in targets if FWD[0] < t["c"][0] < FWD[1]
              and abs(t["c"][1]) < LAT]

    def perturb(pos_sd, yaw_sd, size_sd, drop, extra_fp, seed):
        r = np.random.default_rng(seed)
        det, missed = [], []
        for t in inview:
            hard = (t["cls"] != "vehicle") or t["c"][0] > 28.0
            if r.random() < (drop * (2.1 if hard else 1.0)):
                missed.append(t)
                continue
            det.append(dict(c=t["c"] + r.normal(0, pos_sd, 2),
                            s=t["s"] * (1.0 + r.normal(0, size_sd, 2)),
                            yaw=t["yaw"] + r.normal(0, yaw_sd), cls=t["cls"]))
        fps = []
        for _ in range(extra_fp):
            fx = r.uniform(12.0, 34.0)
            fy = r.choice([-1.0, 1.0]) * r.uniform(7.5, 13.0)
            fps.append(dict(c=np.array([fx, fy]), s=np.array([4.3, 1.9]),
                            yaw=r.normal(0.0, 0.25) + (0 if fy < 0 else 3.14),
                            cls="vehicle"))
        return det, missed, fps

    base_det, base_miss, base_fp = perturb(0.55, 0.095, 0.10, 0.16, 2, SEED + 1)
    ours_det, ours_miss, ours_fp = perturb(0.17, 0.030, 0.035, 0.045, 0,
                                           SEED + 7)

    fig = plt.figure(figsize=(7.4, 5.45))
    L, B, W, H = 0.037, 0.128, 0.307, 0.372
    gapx, gapy = 0.006, 0.033
    axes = []
    for r_ in range(2):
        for c_ in range(3):
            axes.append(fig.add_axes([L + c_ * (W + gapx),
                                      B + (1 - r_) * (H + gapy), W, H]))
    (a_cam, a_bev, a_gt, a_base, a_ours, a_err) = axes

    draw_camera(a_cam, boxes, spheres, targets)
    a_cam.set_xticks([])
    a_cam.set_yticks([])
    for sp in a_cam.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#8a8a8a")
    a_cam.set_title("Camera view", fontsize=8.0, pad=2.6)

    # --- LiDAR BEV density -------------------------------------------------
    ny = 260
    nx = int(round(ny * AR))
    Hh, ye, xe = np.histogram2d(
        P[:, 0], P[:, 1], bins=[ny, nx],
        range=[[FWD[0], FWD[1]], [-LAT, LAT]])
    a_bev.imshow(np.log1p(Hh), origin="lower", cmap="bone_r",
                 extent=[LAT, -LAT, FWD[0], FWD[1]], vmin=0.0,
                 vmax=np.percentile(np.log1p(Hh[Hh > 0]), 96), zorder=2,
                 interpolation="nearest")
    bev_frame(a_bev, "LiDAR BEV density")
    bev_grid(a_bev)

    faint = np.log1p(Hh)
    fmax = np.percentile(faint[faint > 0], 96)
    for ax, title, bold in ((a_gt, "Ground truth", False),
                            (a_base, "Baseline (PointPillars)", False),
                            (a_ours, "Ours", True),
                            (a_err, "Error map (baseline)", False)):
        ax.imshow(faint, origin="lower", cmap="bone_r",
                  extent=[LAT, -LAT, FWD[0], FWD[1]], vmin=0.0, vmax=fmax,
                  alpha=0.30, zorder=2, interpolation="nearest")
        bev_frame(ax, title, bold)
        bev_grid(ax)

    ccol = {"vehicle": SEM["vehicle"], "pedestrian": SEM["pedestrian"],
            "cyclist": SEM["other"]}
    for t in inview:
        draw_bev_box(a_gt, t["c"], t["s"], t["yaw"], ccol[t["cls"]], lw=1.1,
                     fill=0.16)
    for d in base_det:
        draw_bev_box(a_base, d["c"], d["s"], d["yaw"], ccol[d["cls"]], lw=1.0,
                     fill=0.14)
    for d in base_fp:
        draw_bev_box(a_base, d["c"], d["s"], d["yaw"], "#b23a48", lw=1.0)
    for d in ours_det:
        draw_bev_box(a_ours, d["c"], d["s"], d["yaw"], ccol[d["cls"]], lw=1.0,
                     fill=0.14)

    # --- error panel: GT ghost + baseline hits + miss / FP marks -----------
    for t in inview:
        draw_bev_box(a_err, t["c"], t["s"], t["yaw"], "#9a9a9a", lw=0.7, z=4)
    for d in base_det:
        draw_bev_box(a_err, d["c"], d["s"], d["yaw"], "#2e8b57", lw=1.1, z=6)
    for t in base_miss:
        draw_bev_box(a_err, t["c"], t["s"], t["yaw"], "#b23a48", lw=1.3,
                     ls=(0, (2.2, 1.3)), z=7)
        a_err.plot([t["c"][1]], [t["c"][0]], marker="o", ms=6.5, mfc="none",
                   mec="#b23a48", mew=0.9, zorder=7)
    for d in base_fp:
        a_err.plot([d["c"][1]], [d["c"][0]], marker="x", ms=6.0, mew=1.6,
                   color="#b23a48", zorder=8)

    a_bev.plot([LAT - 1.5, LAT - 11.5], [FWD[0] + 2.0, FWD[0] + 2.0],
               color="#333333", lw=1.2, zorder=9)
    a_bev.text(LAT - 6.5, FWD[0] + 2.9, "10 m", fontsize=6.2, ha="center",
               va="bottom", color="#333333", zorder=9)

    # --- shared legend row -------------------------------------------------
    handles = [
        Line2D([], [], color=SEM["vehicle"], lw=1.6, label="Car"),
        Line2D([], [], color=SEM["pedestrian"], lw=1.6, label="Pedestrian"),
        Line2D([], [], color=SEM["other"], lw=1.6, label="Cyclist"),
        Line2D([], [], color="#9a9a9a", lw=1.0, label="GT (reference)"),
        Line2D([], [], color="#2e8b57", lw=1.4, label="True positive"),
        Line2D([], [], color="#b23a48", lw=1.3, ls=(0, (2.4, 1.5)),
               label="Missed detection"),
        Line2D([], [], color="#b23a48", marker="x", ls="none", ms=5, mew=1.4,
               label="False positive"),
    ]
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 0.024), ncol=7, frameon=False,
                     fontsize=7.0, handlelength=1.7, columnspacing=1.25,
                     handletextpad=0.45)
    for t in leg.get_texts():
        t.set_color("#1a1a1a")


    print(f"pts={P.shape[0]} gt={len(inview)} base={len(base_det)} "
          f"miss={len(base_miss)} ours={len(ours_det)}")
    savefig(fig, "284_bev_detection.png")


if __name__ == "__main__":
    main()
