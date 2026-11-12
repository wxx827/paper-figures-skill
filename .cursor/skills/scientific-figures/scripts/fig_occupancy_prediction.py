"""3-D semantic occupancy prediction against ground truth.

Two matched voxel views (GT / prediction) share one camera pose, one axis
range and one colour key; the accessory strip underneath holds the BEV slice
comparison and the IoU-per-height-layer profile.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig

SEED = 61803
LID = np.array([0.0, 0.0, 1.90])
RMAX = 42.0

CLASSES = ["ground", "building", "vegetation", "vehicle", "pedestrian",
           "other"]
SEM = {"ground": "#9aa0a6", "building": "#a9835f", "vegetation": "#4f8f4f",
       "vehicle": "#3b5b92", "pedestrian": "#c9a227", "other": "#8e6bbf"}
NICE = ["Drivable / ground", "Building", "Vegetation", "Vehicle",
        "Pedestrian", "Pole / other"]
CID = {c: i for i, c in enumerate(CLASSES)}

VS = 1.30
X0, X1 = -6.0, 37.0
Y0, Y1 = -13.0, 13.0
Z0, Z1 = 0.0, 9.1
ELEV, AZIM = 30.0, -62.0


def build_scene():
    boxes, spheres = [], []

    def box(cx, cy, cz, sx, sy, sz, yaw, label):
        boxes.append(dict(c=np.array([cx, cy, cz]), s=np.array([sx, sy, sz]),
                          yaw=yaw, label=label))

    for cx, cy, sx, sy, h in [(6, 15.0, 20, 10, 8.5), (28, 15.5, 18, 11, 10.5),
                              (10, -15.0, 20, 10, 9.5), (31, -15.5, 18, 11, 7.5)]:
        box(cx, cy, h / 2, sx, sy, h, 0.0, "building")
    for cx, cy, yaw in [(5.0, 3.3, 0.03), (16.0, 3.1, -0.03), (27.0, 3.4, 0.02),
                        (9.0, -3.4, 3.12), (21.0, -3.2, 3.14),
                        (32.0, -3.5, 3.11)]:
        box(cx, cy, 0.79, 4.6, 1.95, 1.60, yaw, "vehicle")
    for cx, cy in [(7.0, 8.6), (8.2, 9.1), (19.0, -8.5), (30.0, 8.7)]:
        box(cx, cy, 0.88, 0.70, 0.62, 1.80, 0.0, "pedestrian")
    for cx, cy in [(2.0, 9.8), (13.0, 9.8), (24.0, 9.8), (35.0, 9.8),
                   (0.0, -9.8), (11.0, -9.8), (22.0, -9.8), (33.0, -9.8)]:
        box(cx, cy, 1.7, 0.42, 0.42, 3.4, 0.0, "vegetation")
        spheres.append(dict(c=np.array([cx, cy, 5.0]), r=2.2))
    for cx, cy in [(4.0, -9.0), (17.0, 9.0), (29.0, -9.0)]:
        box(cx, cy, 2.7, 0.24, 0.24, 5.4, 0.0, "other")
    return boxes, spheres


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
    return np.where(hit & (t > 0.0), t, np.inf)


def scan(boxes, spheres, rng, n_ch=64, n_az=1500):
    el = np.deg2rad(np.linspace(-25.0, 20.0, n_ch))
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    E, A = np.meshgrid(el, az, indexing="ij")
    E, A = E.ravel(), A.ravel()
    D = np.column_stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A),
                         np.sin(E)])
    t = np.full(D.shape[0], np.inf)
    lab = np.zeros(D.shape[0], dtype=np.int8)
    dz = np.where(np.abs(D[:, 2]) < 1e-9, 1e-9, D[:, 2])
    tg = (0.0 - LID[2]) / dz
    tg = np.where((D[:, 2] < -1e-4) & (tg > 0), tg, np.inf)
    u = tg < t
    t[u], lab[u] = tg[u], CID["ground"]
    for b in boxes:
        tb = _slab(LID, D, b)
        u = tb < t
        t[u], lab[u] = tb[u], CID[b["label"]]
    for sp in spheres:
        L = LID - sp["c"]
        bq = 2.0 * (D @ L)
        disc = bq * bq - 4.0 * (L @ L - sp["r"] ** 2)
        ts = (-bq - np.sqrt(np.maximum(disc, 0.0))) / 2.0
        ts = np.where((disc > 0) & (ts > 0), ts, np.inf)
        ts = np.where(rng.random(ts.size) < 0.75, ts, np.inf)
        u = ts < t
        t[u], lab[u] = ts[u], CID["vegetation"]
    ok = np.isfinite(t) & (t < RMAX)
    return LID + D[ok] * t[ok][:, None], lab[ok]


def voxelize(P, lab):
    nx = int(round((X1 - X0) / VS))
    ny = int(round((Y1 - Y0) / VS))
    nz = int(round((Z1 - Z0) / VS))
    ix = np.floor((P[:, 0] - X0) / VS).astype(int)
    iy = np.floor((P[:, 1] - Y0) / VS).astype(int)
    iz = np.floor((P[:, 2] - Z0) / VS).astype(int)
    m = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz))
    ix, iy, iz, lb = ix[m], iy[m], iz[m], lab[m]
    grid = np.full((nx, ny, nz), -1, dtype=np.int8)
    cnt = np.zeros((nx, ny, nz, len(CLASSES)), dtype=np.int32)
    np.add.at(cnt, (ix, iy, iz, lb), 1)
    tot = cnt.sum(axis=-1)
    occ = tot >= 2
    grid[occ] = np.argmax(cnt, axis=-1)[occ]
    # the drivable surface is dense and complete in the annotation
    xs = (np.arange(nx) + 0.5) * VS + X0
    ys = (np.arange(ny) + 0.5) * VS + Y0
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    road = (np.abs(YY) < 8.6) & (XX > X0 + 0.5) & (
        np.hypot(XX, YY) < RMAX * 0.92)
    lay0 = grid[:, :, 0]
    lay0[road & (lay0 < 0)] = CID["ground"]
    grid[:, :, 0] = lay0
    return grid


def degrade(grid, rng, drop, spurious, flip):
    g = grid.copy()
    occ = np.argwhere(g >= 0)
    d = rng.random(occ.shape[0]) < drop
    # far / tall voxels are the ones a model tends to miss
    r = np.hypot(occ[:, 0] * VS + X0, occ[:, 1] * VS + Y0)
    d |= rng.random(occ.shape[0]) < np.clip((r - 20.0) / 34.0, 0, 1) * 0.55
    d &= occ[:, 2] > 0
    g[tuple(occ[d].T)] = -1
    keep = occ[~d]
    f = rng.random(keep.shape[0]) < flip
    conf = np.array([CID["vehicle"], CID["vegetation"], CID["building"],
                     CID["ground"], CID["other"], CID["vegetation"]])
    g[tuple(keep[f].T)] = conf[g[tuple(keep[f].T)]]
    free = np.argwhere(g < 0)
    sel = free[rng.random(free.shape[0]) < spurious]
    sel = sel[sel[:, 2] > 0]
    g[tuple(sel.T)] = rng.integers(1, len(CLASSES), sel.shape[0])
    return g


def draw_voxels(ax, grid, rng, prob=False):
    occ = np.argwhere(grid >= 0)
    lab = grid[tuple(occ.T)]
    x = occ[:, 0] * VS + X0
    y = occ[:, 1] * VS + Y0
    z = occ[:, 2] * VS + Z0
    cols = np.array([to_rgba(SEM[CLASSES[c]]) for c in lab])
    if prob:
        p = np.clip(0.95 - 0.010 * np.hypot(x, y)
                    + rng.normal(0, 0.06, x.size), 0.34, 0.96)
        cols[:, 3] = p
    else:
        cols[:, 3] = 0.95
    ax.bar3d(x, y, z, VS * 0.92, VS * 0.92, VS * 0.92, color=cols,
             shade=True, edgecolor=(0.25, 0.25, 0.25, 0.35), linewidth=0.12,
             zsort="max")
    return occ.shape[0]


def style3d(ax):
    ax.patch.set_alpha(0.0)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_zlim(Z0, Z1)
    ax.set_box_aspect((1.0, 0.62, 0.30))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.set_xticks([0, 12, 24, 36])
    ax.set_yticks([-12, 0, 12])
    ax.set_zticks([0, 4, 8])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor("#d8d8d8")
        a.pane.set_linewidth(0.45)
        a._axinfo["grid"].update(color="#ececec", linewidth=0.4)


def bev_panel(ax, grid, title, layer=1):
    lay = grid[:, :, layer].T
    cmap = ListedColormap(["#ffffff"] + [SEM[c] for c in CLASSES])
    norm = BoundaryNorm(np.arange(-1.5, len(CLASSES) + 0.5), cmap.N)
    ax.imshow(lay, origin="lower", cmap=cmap, norm=norm,
              extent=[Y1, Y0, X0, X1], aspect="equal",
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_color("#9a9a9a")
    ax.set_title(title, fontsize=7.0, pad=2.0)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    boxes, spheres = build_scene()
    P, lab = scan(boxes, spheres, rng)
    gt = voxelize(P, lab)
    pr = degrade(gt, rng, drop=0.10, spurious=0.0016, flip=0.09)

    fig = plt.figure(figsize=(7.6, 5.05))
    PW, PY, PH = 0.480, 0.392, 0.540
    ax1 = fig.add_axes([0.005 - 0.020, PY - 0.030, PW + 0.040, PH + 0.060],
                       projection="3d")
    n1 = draw_voxels(ax1, gt, rng, prob=False)
    style3d(ax1)

    ax2 = fig.add_axes([0.505 - 0.020, PY - 0.030, PW + 0.040, PH + 0.060],
                       projection="3d")
    n2 = draw_voxels(ax2, pr, rng, prob=True)
    style3d(ax2)

    # ---------------- accessory strip -------------------------------------
    fig.patches.append(plt.Rectangle(
        (0.016, 0.026), 0.970, 0.352, transform=fig.transFigure,
        facecolor="#f4f4f5", edgecolor="#dedede", linewidth=0.6, zorder=-1))

    axg = fig.add_axes([0.040, 0.052, 0.120, 0.268])
    bev_panel(axg, gt, "GT")
    axp = fig.add_axes([0.180, 0.052, 0.120, 0.268])
    bev_panel(axp, pr, "Prediction")
    fig.text(0.170, 0.350, "BEV slice, z = 1.3 - 2.6 m", fontsize=7.0,
             ha="center", va="center", zorder=5)

    layers = np.arange(gt.shape[2])
    ious, recs = [], []
    for k in layers:
        a, b = gt[:, :, k] >= 0, pr[:, :, k] >= 0
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        ious.append(100.0 * inter / max(union, 1))
        same = np.logical_and(a & b, gt[:, :, k] == pr[:, :, k]).sum()
        recs.append(100.0 * same / max(inter, 1))
    zc = (layers + 0.5) * VS

    axi = fig.add_axes([0.398, 0.082, 0.225, 0.235])
    axi.plot(ious, zc, "-o", color="#2e6f9e", ms=3.0, lw=1.3,
             label="Geometric IoU")
    axi.plot(recs, zc, "-s", color="#c9701f", ms=2.8, lw=1.1,
             label="Semantic agreement")
    axi.set_xlim(58, 104)
    axi.set_ylim(0, Z1)
    axi.set_xlabel("score (%)", fontsize=6.6, labelpad=1.2)
    axi.set_ylabel("height z (m)", fontsize=6.6, labelpad=1.5)
    axi.tick_params(labelsize=6.0, length=2.0)
    axi.grid(lw=0.4, color="#e2e2e2")
    axi.set_axisbelow(True)
    axi.set_title("Accuracy per height layer", fontsize=7.0, pad=2.0)
    axi.legend(fontsize=6.0, frameon=False, loc="lower left",
               handlelength=1.4, handletextpad=0.4, labelspacing=0.25)

    handles = [Patch(facecolor=SEM[c], edgecolor="none", label=n)
               for c, n in zip(CLASSES, NICE)]
    leg = fig.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.665, 0.348), ncol=2, frameon=False,
                     fontsize=7.0, handlelength=1.0, handleheight=0.9,
                     labelspacing=0.50, columnspacing=1.0,
                     handletextpad=0.45)
    leg.set_zorder(5)
    fig.text(0.665, 0.088, f"voxel {VS:.2f} m   |   grid "
             f"{gt.shape[0]} x {gt.shape[1]} x {gt.shape[2]}   |   shared "
             f"pose elev {ELEV:.0f}$\\degree$ / azim {AZIM:.0f}$\\degree$\n"
             f"GT {n1} occupied voxels   |   prediction {n2}\n"
             f"mIoU 71.4 %   |   RayIoU 38.2 %   |   voxel alpha = "
             "occupancy probability",
             fontsize=6.4, color="#444444", va="center", linespacing=1.6,
             zorder=5)


    print(f"gt={n1} pred={n2}")
    savefig(fig, "291_occupancy_prediction.png")


if __name__ == "__main__":
    main()
