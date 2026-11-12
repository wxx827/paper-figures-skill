"""Point-cloud semantic segmentation: Input / GT / Baseline / Ours.

The four 3-D panels are locked to one camera pose and one set of axis limits
so the comparison is meaningful; the shared class legend and the per-class IoU
bars live in a shrunken accessory strip along the bottom.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig

SEED = 5150
SENSOR = np.array([0.0, 0.0, 1.90])
RMAX = 37.0

CLASSES = ["ground", "building", "vegetation", "vehicle", "pedestrian",
           "other"]
SEM = {"ground": "#9aa0a6", "building": "#a9835f", "vegetation": "#4f8f4f",
       "vehicle": "#3b5b92", "pedestrian": "#c9a227", "other": "#8e6bbf"}
NICE = {"ground": "Ground", "building": "Building", "vegetation": "Vegetation",
        "vehicle": "Vehicle", "pedestrian": "Pedestrian", "other": "Pole /\nother"}
CID = {c: i for i, c in enumerate(CLASSES)}

# fixed view + limits, shared by all four panels
XL, YL, ZL = (-14.0, 34.0), (-21.0, 21.0), (0.0, 15.0)
ELEV, AZIM = 26.0, -62.0


def build_scene():
    boxes, spheres = [], []

    def box(cx, cy, cz, sx, sy, sz, yaw, label):
        boxes.append(dict(c=np.array([cx, cy, cz]), s=np.array([sx, sy, sz]),
                          yaw=yaw, label=label))

    for cx, cy, sx, sy, h in [(-8, 18.0, 18, 10, 9.0), (13, 18.5, 18, 11, 12.0),
                              (31, 17.5, 14, 9, 8.0), (-6, -18.0, 18, 10, 10.5),
                              (15, -18.5, 18, 11, 7.5), (40, 3.0, 12, 34, 12.0)]:
        box(cx, cy, h / 2, sx, sy, h, 0.0, "building")
    for cx, cy, yaw in [(6.0, 3.2, 0.04), (17.0, 3.4, -0.03), (27.0, 3.1, 0.02),
                        (2.0, -3.4, 3.12), (12.5, -3.3, 3.14),
                        (23.0, -3.5, 3.11), (-8.0, 3.2, 0.0)]:
        box(cx, cy, 0.79, 4.5, 1.95, 1.58, yaw, "vehicle")
    for cx, cy in [(9.0, 8.5), (10.1, 9.1), (20.0, -8.5), (28.0, 8.8),
                   (4.0, -8.7)]:
        box(cx, cy, 0.88, 0.62, 0.55, 1.76, 0.0, "pedestrian")
    for cx, cy in [(0.0, 9.7), (11.0, 9.7), (22.0, 9.7), (33.0, 9.7),
                   (-5.0, -9.7), (6.0, -9.7), (17.0, -9.7), (28.0, -9.7)]:
        box(cx, cy, 1.6, 0.40, 0.40, 3.2, 0.0, "vegetation")
        spheres.append(dict(c=np.array([cx, cy, 4.8]), r=2.15))
    for cx, cy in [(3.0, 9.0), (15.5, -9.0), (25.0, 9.0)]:   # lamp posts
        box(cx, cy, 2.6, 0.22, 0.22, 5.2, 0.0, "other")
    return boxes, spheres


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


def scan(boxes, spheres, n_ch=60, n_az=1500, seed=SEED):
    rng = np.random.default_rng(seed)
    el = np.deg2rad(np.linspace(-25.0, 20.0, n_ch))
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    E, A = np.meshgrid(el, az, indexing="ij")
    E, A = E.ravel(), A.ravel()
    D = np.column_stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A),
                         np.sin(E)])
    t = np.full(D.shape[0], np.inf)
    lab = np.zeros(D.shape[0], dtype=np.int8)
    dz = np.where(np.abs(D[:, 2]) < 1e-9, 1e-9, D[:, 2])
    tg = (0.0 - SENSOR[2]) / dz
    tg = np.where((D[:, 2] < -1e-4) & (tg > 0), tg, np.inf)
    u = tg < t
    t[u], lab[u] = tg[u], CID["ground"]
    for b in boxes:
        tb = _ray_box(SENSOR, D, b)
        u = tb < t
        t[u], lab[u] = tb[u], CID[b["label"]]
    for sp in spheres:
        L = SENSOR - sp["c"]
        bq = 2.0 * (D @ L)
        disc = bq * bq - 4.0 * (L @ L - sp["r"] ** 2)
        ts = (-bq - np.sqrt(np.maximum(disc, 0.0))) / 2.0
        ts = np.where((disc > 0) & (ts > 0), ts, np.inf)
        ts = ts * (1.0 + rng.normal(0, 0.05, ts.size))
        ts = np.where(rng.random(ts.size) < 0.70, ts, np.inf)
        u = ts < t
        t[u], lab[u] = ts[u], CID["vegetation"]
    ok = np.isfinite(t) & (t < RMAX)
    P = SENSOR + D[ok] * t[ok][:, None] + rng.normal(0, 0.022, (ok.sum(), 3))
    return P, lab[ok], t[ok]


CONFUSE = {CID["ground"]: CID["vehicle"], CID["building"]: CID["vegetation"],
           CID["vegetation"]: CID["building"], CID["vehicle"]: CID["ground"],
           CID["pedestrian"]: CID["other"], CID["other"]: CID["vegetation"]}


def corrupt(P, lab, n_blob, radius, seed, far_boost=0.0):
    """Spatially coherent label errors -- blobs, plus range-dependent noise."""
    rng = np.random.default_rng(seed)
    out = lab.copy()
    idx = rng.choice(P.shape[0], n_blob, replace=False)
    for i in idx:
        d = np.linalg.norm(P - P[i], axis=1)
        m = d < radius * rng.uniform(0.7, 1.3)
        for c, c2 in CONFUSE.items():
            mm = m & (out == c)
            out[mm] = c2
    if far_boost > 0:
        r = np.linalg.norm(P - SENSOR, axis=1)
        p = np.clip((r - 20.0) / 20.0, 0.0, 1.0) * far_boost
        flip = rng.random(P.shape[0]) < p
        for c, c2 in CONFUSE.items():
            mm = flip & (out == c)
            out[mm] = c2
    return out


def style3d(ax, title, bold=False):
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_zlim(*ZL)
    ax.set_box_aspect((1.0, 0.72, 0.52))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xticks([-10, 5, 20, 32])
    ax.set_yticks([-18, 0, 18])
    ax.set_zticks([0, 7, 14])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor("#d8d8d8")
        a.pane.set_linewidth(0.45)
        a._axinfo["grid"].update(color="#ececec", linewidth=0.4)
    ax.set_title(title, fontsize=8.6, pad=-14.0,
                 fontweight="bold" if bold else "normal")


def main():
    set_pub_style(base_font=8)
    boxes, spheres = build_scene()
    P, gt, rng_ = scan(boxes, spheres)

    g = np.random.default_rng(SEED + 3)
    # the z floor matters as much as the walls: mplot3d does not clip scatter
    # against the axis limits, so points below ZL[0] float under the box
    inbox = ((P[:, 0] > XL[0]) & (P[:, 0] < XL[1]) & (P[:, 1] > YL[0])
             & (P[:, 1] < YL[1]) & (P[:, 2] > ZL[0]) & (P[:, 2] < ZL[1]))
    P, gt, rng_ = P[inbox], gt[inbox], rng_[inbox]
    keep = g.random(P.shape[0]) < np.clip(1.05 - 0.011 * rng_, 0.45, 1.0)
    P, gt, rng_ = P[keep], gt[keep], rng_[keep]
    order = np.argsort(-rng_)
    P, gt, rng_ = P[order], gt[order], rng_[order]

    base = corrupt(P, gt, 40, 3.2, SEED + 11, far_boost=0.34)
    ours = corrupt(P, gt, 7, 1.2, SEED + 29, far_boost=0.04)

    size = np.clip(3.0 - 0.048 * rng_, 0.55, 3.0)
    alpha = np.clip(0.95 - 0.008 * rng_, 0.50, 0.95)

    fig = plt.figure(figsize=(7.9, 3.60))
    PW, PH, Y0 = 0.2455, 0.575, 0.322
    panels = [("Input (intensity)", None, False),
              ("Ground truth", gt, False),
              ("Baseline (SPVCNN)", base, False),
              ("Ours", ours, True)]
    for i, (title, lab, bold) in enumerate(panels):
        ax = fig.add_axes([0.004 + i * PW - 0.026, Y0 - 0.030,
                           PW + 0.052, PH + 0.060], projection="3d")
        # a 3-D axes keeps an opaque background patch that would clip a white
        # notch out of the grey accessory strip below
        ax.patch.set_alpha(0.0)
        if lab is None:
            refl = np.array([0.34, 0.52, 0.24, 0.78, 0.58, 0.66])[gt]
            inten = np.clip(refl + g.normal(0, 0.07, P.shape[0])
                            - 0.0045 * rng_, 0.02, 1.0)
            rgba = plt.get_cmap("cividis")(inten)
        else:
            rgba = np.array([plt.matplotlib.colors.to_rgba(SEM[CLASSES[c]])
                             for c in lab])
        rgba[:, 3] = alpha
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=size, c=rgba, marker=".",
                   linewidths=0, depthshade=False, rasterized=True)
        style3d(ax, title, bold)

    # ---------------- accessory strip -------------------------------------
    fig.patches.append(plt.Rectangle(
        (0.018, 0.026), 0.966, 0.272, transform=fig.transFigure,
        facecolor="#f4f4f5", edgecolor="#dedede", linewidth=0.6, zorder=-1))

    handles = [Patch(facecolor=SEM[c], edgecolor="none",
                     label=NICE[c].replace("\n", " ")) for c in CLASSES]
    leg = fig.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.036, 0.286), ncol=2, frameon=False,
                     fontsize=7.0, handlelength=1.0, handleheight=0.9,
                     labelspacing=0.42, columnspacing=1.0,
                     handletextpad=0.45)
    leg.set_zorder(5)

    iou_b = np.array([93.1, 84.6, 71.3, 79.8, 41.2, 48.5])
    iou_o = np.array([96.4, 91.8, 83.7, 90.2, 62.9, 66.1])
    axb = fig.add_axes([0.335, 0.082, 0.400, 0.168])
    x = np.arange(len(CLASSES))
    axb.bar(x - 0.20, iou_b, width=0.38, color="#b9c4d0", edgecolor="#8b98a6",
            linewidth=0.4, label="Baseline")
    axb.bar(x + 0.20, iou_o, width=0.38, color=[SEM[c] for c in CLASSES],
            edgecolor="#4a4a4a", linewidth=0.4, label="Ours")
    for xi, (a, b) in enumerate(zip(iou_b, iou_o)):
        axb.text(xi + 0.20, b + 1.8, f"{b:.0f}", ha="center", va="bottom",
                 fontsize=6.0, color="#1a1a1a", fontweight="bold")
    axb.set_xticks(x)
    axb.set_xticklabels([NICE[c] for c in CLASSES], fontsize=6.0,
                        linespacing=0.95)
    axb.set_ylim(0, 108)
    axb.set_yticks([0, 50, 100])
    axb.tick_params(labelsize=6.0, length=2.0)
    axb.set_ylabel("IoU (%)", fontsize=6.4, labelpad=1.5)
    axb.grid(axis="y", lw=0.4, color="#e0e0e0")
    axb.set_axisbelow(True)
    axb.set_title("Per-class IoU", fontsize=7.0, pad=2.0)
    axb.legend(loc="lower right", bbox_to_anchor=(1.005, 0.98), ncol=2,
               fontsize=6.0, frameon=False, handlelength=1.0,
               handletextpad=0.4, columnspacing=0.9)

    axm = fig.add_axes([0.795, 0.082, 0.170, 0.168])
    mi = [np.mean(iou_b), np.mean(iou_o)]
    axm.bar([0, 1], mi, width=0.52, color=["#b9c4d0", "#2e6f9e"],
            edgecolor="#4a4a4a", linewidth=0.4)
    for i, v in enumerate(mi):
        axm.text(i, v + 1.6, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=6.4, fontweight="bold" if i else "normal")
    axm.set_xticks([0, 1])
    axm.set_xticklabels(["Baseline", "Ours"], fontsize=6.2)
    axm.set_ylim(0, 100)
    axm.set_yticks([0, 50, 100])
    axm.tick_params(labelsize=6.0, length=2.0)
    axm.set_title("mIoU (%)", fontsize=7.0, pad=2.0)
    axm.grid(axis="y", lw=0.4, color="#e0e0e0")
    axm.set_axisbelow(True)


    print(f"points={P.shape[0]}")
    savefig(fig, "287_pointcloud_segmentation.png")


if __name__ == "__main__":
    main()
