"""Figure 256 - Object detection, qualitative comparison grid.

Two synthetic driving scenes (rows) against ground truth and four detectors
(columns).  Every prediction carries a class / confidence chip; missed ground
truth is redrawn as a red dashed box and false positives are struck through.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, PALETTES, savefig

H, W = 176, 256
MISS_RED = "#b23a48"
FP_ORANGE = "#c9a227"
DET_COLORS = {"car": "#3b5b92", "person": "#8e6bbf", "sign": "#2e7d5b"}


def image_axes(ax, color="#b8b8b8", lw=0.6):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def _sm(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _hex(c):
    return np.array(plt.matplotlib.colors.to_rgb(c))


# ---------------------------------------------------------------------------
# Synthetic driving scene + ground-truth boxes
# ---------------------------------------------------------------------------
def driving_scene(seed, car_depths=(0.86, 0.44), ped_depths=(0.66, 0.34)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    horizon = 66
    t = np.clip((yy - horizon) / (H - horizon), 0, 1)
    left = 112 + (2 - 112) * t
    right = 144 + (254 - 144) * t

    img = np.zeros((H, W, 3))
    g = np.clip(yy / horizon, 0, 1)[..., None]
    img = _hex("#5c8cc4") * (1 - g) + _hex("#dbe8f2") * g
    haze = np.clip(_sm(rng, (H, W), 10.0) * 0.45 + 0.2, 0, 1)[..., None]
    img = img * (1 - 0.25 * haze) + 0.25 * haze

    ground = yy >= horizon
    img = np.where(ground[..., None],
                   _hex("#7d8a6b") * (1 + 0.14 * _sm(rng, (H, W), 3))[..., None],
                   img)
    road = ground & (xx >= left) & (xx <= right)
    img = np.where(road[..., None],
                   _hex("#4c4e55") * (1 + 0.07 * _sm(rng, (H, W), 2))[..., None],
                   img)
    walk = ground & (((xx < left) & (xx > left - 12 - 30 * t))
                     | ((xx > right) & (xx < right + 12 + 30 * t)))
    img = np.where(walk[..., None],
                   _hex("#aaa49b") * (1 + 0.08 * _sm(rng, (H, W), 2))[..., None],
                   img)
    lane = road & (np.abs(xx - (left + right) / 2) < 0.9 + 1.5 * t) & \
        (((yy * (0.7 + 1.3 * t)) % 24) < 12)
    img = np.where(lane[..., None], _hex("#ddd5bc"), img)

    # distant skyline
    x = 0
    while x < W:
        bw = int(rng.integers(16, 40))
        top = horizon - int(rng.integers(12, 46))
        blk = (yy >= top) & (yy < horizon) & (xx >= x) & (xx < x + bw)
        img = np.where(blk[..., None],
                       _hex("#93897f") * (0.82 + 0.3 * rng.random()), img)
        x += bw + int(rng.integers(2, 7))

    boxes = []
    # ---- cars
    for ci, depth in enumerate(car_depths):
        cy = horizon + depth * (H - horizon)
        sc = 0.22 + 1.05 * depth
        cw, ch = 38 * sc, 22 * sc
        lane_i = -1 if ci % 2 == 0 else 1
        mid = (112 + (2 - 112) * depth + 144 + (254 - 144) * depth) / 2
        cx = mid + lane_i * 0.26 * (254 - 2) * depth * 0.5
        cx = float(np.clip(cx, cw / 2 + 2, W - cw / 2 - 2))
        col = _hex(["#b8452f", "#2f5f9e", "#c9a227", "#8b8f96"]
                   [int(rng.integers(0, 4))])
        body = (np.abs(xx - cx) < cw / 2) & (yy > cy - ch) & (yy < cy)
        roof = (np.abs(xx - cx) < 0.60 * cw / 2) & (yy > cy - 1.45 * ch) & \
               (yy <= cy - ch)
        img = np.where((body | roof)[..., None],
                       col * (1 + 0.10 * _sm(rng, (H, W), 1.4))[..., None], img)
        win = roof & (np.abs(xx - cx) < 0.50 * cw / 2) & (yy > cy - 1.36 * ch)
        img = np.where(win[..., None], _hex("#20242b"), img)
        for sgn in (-1, 1):
            wh = ((xx - (cx + sgn * 0.34 * cw)) ** 2 / (0.10 * cw) ** 2
                  + (yy - cy) ** 2 / (0.16 * ch) ** 2) < 1
            img = np.where(wh[..., None], _hex("#191b20"), img)
        boxes.append(("car", cx - cw / 2 - 1, cy - 1.5 * ch - 1,
                      cx + cw / 2 + 1, cy + 1))

    # ---- pedestrians on the sidewalk
    for k, depth in enumerate(ped_depths):
        py = horizon + depth * (H - horizon)
        sc = 0.30 + 0.95 * depth
        ph, pw = 44 * sc, 6.0 * sc
        side = -1 if k == 0 else 1
        edge = (112 + (2 - 112) * depth) if side < 0 else \
               (144 + (254 - 144) * depth)
        px = float(np.clip(edge + side * (10 + 18 * depth), 24, W - 24))
        torso = (np.abs(xx - px) < pw) & (yy > py - ph) & (yy < py)
        head = ((xx - px) ** 2 + (yy - (py - ph - pw)) ** 2) < (pw * 0.95) ** 2
        img = np.where(torso[..., None], _hex("#31343f"), img)
        img = np.where(head[..., None], _hex("#c39a7a"), img)
        boxes.append(("person", px - pw - 1.5, py - ph - 2.4 * pw,
                      px + pw + 1.5, py + 1))

    # ---- traffic sign
    sx, sy, sh = W - 26.0, horizon + 40.0, 38.0
    pole = (np.abs(xx - sx) < 1.4) & (yy > sy - sh) & (yy < sy)
    plate = ((xx - sx) ** 2 / 8.5 ** 2 + (yy - (sy - sh - 6)) ** 2 / 8.5 ** 2) < 1
    img = np.where(pole[..., None], _hex("#6f7379"), img)
    img = np.where(plate[..., None], _hex("#c93b32"), img)
    boxes.append(("sign", sx - 10, sy - sh - 16, sx + 10, sy - sh + 5))

    vign = 1 - 0.15 * (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) / 0.5)
    img = img * vign[..., None]
    img = np.clip(img + 0.010 * rng.standard_normal(img.shape), 0, 1)
    img = np.dstack([gaussian_filter(img[..., k], 0.45) for k in range(3)])
    return np.clip(img, 0, 1), boxes


# ---------------------------------------------------------------------------
# Detector simulation
# ---------------------------------------------------------------------------
def detect(boxes, rng, jitter=2.0, miss=0.0, fp=0, conf=(0.72, 0.93),
           min_area=0.0):
    dets, missed = [], []
    for (cls, x0, y0, x1, y1) in boxes:
        area = (x1 - x0) * (y1 - y0)
        p_miss = miss * (2.2 if area < min_area else 1.0)
        if rng.random() < p_miss:
            missed.append((cls, x0, y0, x1, y1))
            continue
        j = jitter * (1.0 + 0.5 * rng.standard_normal())
        dets.append((cls,
                     x0 + j * rng.standard_normal(), y0 + j * rng.standard_normal(),
                     x1 + j * rng.standard_normal(), y1 + j * rng.standard_normal(),
                     float(rng.uniform(*conf))))
    for _ in range(fp):
        cls = ["car", "person", "sign"][int(rng.integers(0, 3))]
        w = rng.uniform(16, 34)
        h = w * rng.uniform(0.5, 1.5)
        x0 = rng.uniform(30, W - w - 30)
        y0 = rng.uniform(74, H - h - 14)
        dets.append((cls, x0, y0, x0 + w, y0 + h, float(rng.uniform(0.31, 0.58))))
    return dets, missed


def draw_box(ax, cls, x0, y0, x1, y1, conf=None, color=None, dashed=False,
             lw=1.0, chip=True, strike=False):
    col = color or DET_COLORS[cls]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=col, linewidth=lw, zorder=5,
                           linestyle=(0, (2.2, 1.5)) if dashed else "solid"))
    if strike:
        ax.plot([x0, x1], [y0, y1], color=MISS_RED, lw=1.1, zorder=6)
        ax.plot([x0, x1], [y1, y0], color=MISS_RED, lw=1.1, zorder=6)
    if chip:
        txt = cls if conf is None else f"{cls} {conf:.2f}"
        # keep the chip inside the tile: flip anchoring near the borders
        right = x0 > W - 76
        below = y0 < 12
        ax.text(x1 - 0.5 if right else x0 + 0.5,
                y1 + 1.0 if below else y0 - 1.0, txt, fontsize=6.0,
                color="white", ha="right" if right else "left",
                va="top" if below else "bottom", zorder=7, clip_on=True,
                bbox=dict(boxstyle="round,pad=0.14", facecolor=col,
                          edgecolor="none"))


METHODS = [
    ("Faster R-CNN", dict(jitter=3.4, miss=0.30, fp=2, conf=(0.55, 0.84),
                          min_area=260), 34.6),
    ("YOLOv8", dict(jitter=2.6, miss=0.22, fp=2, conf=(0.60, 0.88),
                    min_area=260), 39.1),
    ("DETR", dict(jitter=1.8, miss=0.14, fp=1, conf=(0.66, 0.91),
                  min_area=260), 43.8),
    ("Ours", dict(jitter=0.7, miss=0.0, fp=0, conf=(0.86, 0.98),
                  min_area=260), 49.7),
]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(17)
    col_names = ["Ground truth"] + [m for m, _, _ in METHODS]
    ncol = len(col_names)
    rows = ["Daytime", "Dense traffic"]

    fig = plt.figure(figsize=(10.4, 3.84))
    gs = fig.add_gridspec(2, ncol, wspace=0.030, hspace=0.030,
                          left=0.048, right=0.996, top=0.888, bottom=0.185)

    axes = np.empty((2, ncol), object)
    scene_kw = [dict(car_depths=(0.86, 0.40), ped_depths=(0.64, 0.30)),
                dict(car_depths=(0.92, 0.56, 0.28), ped_depths=(0.72, 0.38))]
    for i in range(2):
        img, gt = driving_scene(seed=101 + 13 * i, **scene_kw[i])
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax)
            ax.imshow(img, interpolation="bilinear")
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            if j == 0:
                for (cls, x0, y0, x1, y1) in gt:
                    draw_box(ax, cls, x0, y0, x1, y1, lw=1.1)
            else:
                dets, missed = detect(gt, rng, **METHODS[j - 1][1])
                for (cls, x0, y0, x1, y1, c) in dets:
                    fp = c < 0.60 and METHODS[j - 1][0] != "Ours"
                    draw_box(ax, cls, x0, y0, x1, y1, conf=c,
                             color=FP_ORANGE if fp else None, strike=fp,
                             lw=0.95)
                for (cls, x0, y0, x1, y1) in missed:
                    draw_box(ax, cls, x0, y0, x1, y1, color=MISS_RED,
                             dashed=True, chip=False, lw=1.1)
                    ax.text(x0 + 0.5, y1 + 1.5, "missed", fontsize=6.0,
                            color="white", ha="left", va="top", zorder=7,
                            clip_on=True,
                            bbox=dict(boxstyle="round,pad=0.14",
                                      facecolor=MISS_RED, edgecolor="none"))
                ap = METHODS[j - 1][2] - 3.2 * i
                ax.text(0.978, 0.955, f"AP {ap:.1f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.2, color="white",
                        fontweight="bold" if j == ncol - 1 else "normal",
                        zorder=8,
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="#000000aa", edgecolor="none"))
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=7.8, labelpad=3.0)
            if i == 1:
                bold = col_names[j] == "Ours"
                ax.set_xlabel(col_names[j], fontsize=8.0, labelpad=3.2,
                              fontweight="bold" if bold else "normal",
                              color="#0b3d2e" if bold else "#1a1a1a")

    p0, p1 = axes[0, ncol - 1].get_position(), axes[1, ncol - 1].get_position()
    fig.add_artist(Rectangle((p0.x0 - 0.005, p1.y0 - 0.072), p0.width + 0.010,
                             p0.y1 - p1.y0 + 0.082, transform=fig.transFigure,
                             facecolor="#e8f2ec", edgecolor="none", zorder=-5))

    handles = [Patch(facecolor=DET_COLORS[c], edgecolor="none", label=c)
               for c in ("car", "person", "sign")]
    handles += [
        plt.Line2D([], [], color=MISS_RED, lw=1.2, linestyle=(0, (2.2, 1.5)),
                   label="missed detection"),
        plt.Line2D([], [], color=MISS_RED, lw=1.2, marker="x", markersize=5,
                   linestyle="none", label="false positive"),
    ]
    leg = fig.legend(handles=handles, ncol=5, loc="lower center",
                     bbox_to_anchor=(0.5, 0.004), frameon=False, fontsize=6.8,
                     handlelength=1.2, columnspacing=1.5, handletextpad=0.45)
    leg.set_zorder(10)

    savefig(fig, "256_cv_detection_compare.png")


if __name__ == "__main__":
    main()
