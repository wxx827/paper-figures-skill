"""Figure 255 - Semantic segmentation, qualitative comparison matrix.

Three synthetic street scenes (rows) against image, ground truth and five
predictors (columns).  Predictions are translucent class overlays; the Ours
column additionally circles the regions that the baselines get wrong.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Patch
from scipy.ndimage import gaussian_filter, map_coordinates

from pubstyle import set_pub_style, PALETTES, savefig

H, W = 176, 256
SKY, BUILD, ROAD, WALK, VEG, CAR, PERSON = range(7)
CLASS_NAMES = ["Sky", "Building", "Road", "Sidewalk", "Vegetation",
               "Car", "Person"]
CLASS_COLORS = ["#8ab6dd", "#a99a8c", "#6f7078", "#d2c9bf", "#5f9e4f",
                "#c0603f", "#d1478a"]


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
# Synthetic street scene: labels + a matching "photograph"
# ---------------------------------------------------------------------------
def street_scene(seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    horizon = 68 + int(rng.integers(-5, 6))

    lab = np.full((H, W), WALK, np.int16)
    lab[:horizon] = SKY
    shade = np.ones((H, W))

    # ---- buildings along the horizon
    x = 0
    while x < W:
        bw = int(rng.integers(22, 46))
        top = horizon - int(rng.integers(24, 58))
        keep = rng.random() > 0.22
        if keep:
            lab[max(top, 0):horizon + 3, x:min(x + bw, W)] = BUILD
            shade[max(top, 0):horizon + 3, x:min(x + bw, W)] = \
                0.78 + 0.34 * rng.random()
        x += bw + int(rng.integers(1, 5))

    # ---- road as a perspective trapezoid, sidewalk on both flanks
    t = np.clip((yy - horizon) / (H - horizon), 0, 1)
    left = 116 + (14 - 116) * t
    right = 140 + (244 - 140) * t
    road = (yy >= horizon) & (xx >= left) & (xx <= right)
    lab[road] = ROAD
    walk_w = 14 + 40 * t
    veg_band = (yy >= horizon) & ((xx < left - walk_w)
                                  | (xx > right + walk_w))
    lab[veg_band] = VEG

    # ---- roadside trees: crowns deliberately rise above the horizon so the
    # class boundaries a segmenter has to get right are actually visible
    trees = []
    for k in range(5):
        side = -1 if k % 2 == 0 else 1
        tt = float(rng.uniform(0.10, 0.85))
        ty = horizon + tt * (H - horizon)
        edge = (116 + (14 - 116) * tt) if side < 0 else (140 + (244 - 140) * tt)
        tx = edge + side * (18 + 46 * tt)
        r = 7 + 20 * tt
        if -r < tx < W + r:
            trees.append((tx, ty, r))
    for (tx, ty, r) in trees:
        cyc = ty - 2.1 * r
        crown = ((xx - tx) / r) ** 2 + ((yy - cyc) / (0.92 * r)) ** 2 < \
            1 + 0.30 * _sm(rng, (H, W), 2.0)
        lab[crown] = VEG
        trunk = (np.abs(xx - tx) < max(1.2, 0.13 * r)) & (yy > cyc) & \
                (yy < ty)
        lab[trunk] = VEG

    # ---- cars receding down the road
    cars = []
    for cy_t in rng.choice(np.linspace(0.16, 0.92, 9), size=4, replace=False):
        cy = horizon + cy_t * (H - horizon)
        sc = 0.20 + 1.0 * cy_t
        cw, ch = 34 * sc, 20 * sc
        lane = rng.choice([-1, 1])
        cx = (left[int(cy), 0] + right[int(cy), 0]) / 2 + lane * 0.22 * \
            (right[int(cy), 0] - left[int(cy), 0])
        body = (np.abs(xx - cx) < cw / 2) & (yy > cy - ch) & (yy < cy)
        roofm = (np.abs(xx - cx) < 0.62 * cw / 2) & (yy > cy - 1.42 * ch) & \
                (yy <= cy - ch)
        lab[body | roofm] = CAR
        cars.append((cx, cy, cw, ch, _hex(["#b8452f", "#2f5f9e", "#c9a227",
                                           "#7a7f88"][rng.integers(0, 4)])))

    # ---- one pedestrian on the sidewalk (the hard, small object)
    py = H - 22
    px = float(left[py, 0]) - 22
    ph = 40.0
    person = ((np.abs(xx - px) < 4.6) & (yy > py - ph) & (yy < py)) | \
             (((xx - px) ** 2 + (yy - (py - ph - 4.2)) ** 2) < 4.6 ** 2)
    lab[person] = PERSON

    # ---- render the photograph -------------------------------------------
    photo = np.zeros((H, W, 3))
    sky_top, sky_bot = _hex("#4f83bd"), _hex("#d8e7f2")
    g = np.clip(yy / max(horizon, 1), 0, 1)[..., None]
    photo = sky_top * (1 - g) + sky_bot * g
    cloud = np.clip(_sm(rng, (H, W), 9.0) * 0.5 + 0.15, 0, 1)[..., None]
    photo = photo * (1 - 0.28 * cloud) + 0.28 * cloud

    tex = {BUILD: 0.10, ROAD: 0.05, WALK: 0.07, VEG: 0.30, CAR: 0.05,
           PERSON: 0.05}
    base = {BUILD: "#8f8880", ROAD: "#4d4f56", WALK: "#b3ada4",
            VEG: "#3d6b32", CAR: "#8a8a8a", PERSON: "#2f313a"}
    fine = _sm(rng, (H, W), 1.1)
    med = _sm(rng, (H, W), 3.5)
    for cls, col in base.items():
        m = lab == cls
        if not m.any():
            continue
        amp = tex[cls]
        f = (1 + amp * (0.6 * fine + 0.4 * med))[..., None]
        photo = np.where(m[..., None], _hex(col) * f * shade[..., None], photo)

    for (cx, cy, cw, ch, col) in cars:                # per-instance car paint
        m = (lab == CAR) & (np.abs(xx - cx) < cw) & (yy > cy - 1.6 * ch) & \
            (yy < cy + 1)
        photo = np.where(m[..., None], col * (1 + 0.12 * fine)[..., None],
                         photo)

    win = ((xx % 9 < 5) & (yy % 11 < 6) & (lab == BUILD) & (yy < horizon - 4))
    photo = np.where(win[..., None],
                     photo * 0.55 + 0.45 * _hex("#e8d9a8") *
                     (0.4 + 0.6 * (_sm(rng, (H, W), 1.0) > 0))[..., None],
                     photo)
    lane = (lab == ROAD) & (np.abs(xx - (left + right) / 2) < 1.1 + 1.6 * t) \
        & (((yy * (0.6 + 1.4 * t)) % 22) < 11)
    photo = np.where(lane[..., None], _hex("#ded6bd"), photo)

    vign = 1 - 0.16 * (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) / 0.5)
    photo = photo * vign[..., None]
    photo = np.clip(photo + 0.012 * rng.standard_normal(photo.shape), 0, 1)
    photo = np.dstack([gaussian_filter(photo[..., k], 0.5) for k in range(3)])
    return np.clip(photo, 0, 1), lab, (px, py, ph), cars


# ---------------------------------------------------------------------------
# Prediction simulation: warp the label field, then confuse a few classes
# ---------------------------------------------------------------------------
def predict(gt, rng, warp=4.0, scale=16, lose_person=0.0, walk_to_road=0.0,
            veg_to_build=0.0, car_shrink=0.0, hallu=0.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dy = warp * _sm(rng, (H, W), scale)
    dx = warp * _sm(rng, (H, W), scale)
    pred = map_coordinates(gt, [yy + dy, xx + dx], order=0, mode="nearest")
    if lose_person:
        m = (pred == PERSON) & (_sm(rng, (H, W), 3.0) < lose_person * 2.4)
        pred[m] = WALK
    if walk_to_road:
        m = (pred == WALK) & (_sm(rng, (H, W), 11.0) > 1.2 - walk_to_road * 2)
        pred[m] = ROAD
    if veg_to_build:
        m = (pred == VEG) & (_sm(rng, (H, W), 9.0) > 1.3 - veg_to_build * 2)
        pred[m] = BUILD
    if car_shrink:
        m = (pred == CAR) & (_sm(rng, (H, W), 2.0) < car_shrink * 2.2)
        pred[m] = ROAD
    if hallu:
        for (src, dst, sg) in ((ROAD, WALK, 7.0), (BUILD, VEG, 6.0),
                               (VEG, BUILD, 5.0), (WALK, VEG, 8.0),
                               (SKY, BUILD, 9.0)):
            m = (pred == src) & (_sm(rng, (H, W), sg) > 1.75 - hallu * 2.0)
            pred[m] = dst
    return pred


METHODS = [
    ("FCN-8s", dict(warp=9.5, scale=11, lose_person=0.95, walk_to_road=0.52,
                    veg_to_build=0.42, car_shrink=0.45, hallu=0.85), 61.4),
    ("U-Net", dict(warp=6.8, scale=13, lose_person=0.60, walk_to_road=0.36,
                   veg_to_build=0.28, car_shrink=0.28, hallu=0.58), 68.9),
    ("DeepLabv3+", dict(warp=4.4, scale=16, lose_person=0.34,
                        walk_to_road=0.22, veg_to_build=0.17,
                        car_shrink=0.16, hallu=0.36), 74.6),
    ("Mask2Former", dict(warp=2.4, scale=19, lose_person=0.18,
                         walk_to_road=0.11, veg_to_build=0.09,
                         car_shrink=0.08, hallu=0.18), 79.2),
    ("Ours", dict(warp=0.7, scale=22, lose_person=0.0, walk_to_road=0.02,
                  veg_to_build=0.02, car_shrink=0.0, hallu=0.0), 83.5),
]


def overlay(photo, lab, alpha=0.52):
    col = np.zeros_like(photo)
    for c, hexc in enumerate(CLASS_COLORS):
        col = np.where((lab == c)[..., None], _hex(hexc), col)
    return np.clip(photo * (1 - alpha) + col * alpha, 0, 1)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(3)
    col_names = ["Image", "Ground truth"] + [m for m, _, _ in METHODS]
    ncol = len(col_names)
    rows = ["Scene A", "Scene B", "Scene C"]

    fig = plt.figure(figsize=(8.4, 3.30))
    gs = fig.add_gridspec(3, ncol, wspace=0.030, hspace=0.030,
                          left=0.040, right=0.996, top=0.905, bottom=0.170)

    axes = np.empty((3, ncol), object)
    for i, rname in enumerate(rows):
        photo, gt, person, cars = street_scene(seed=31 + 9 * i)
        tiles = [photo, overlay(photo, gt)]
        for _, kw, _m in METHODS:
            tiles.append(overlay(photo, predict(gt, rng, **kw)))
        for j, im in enumerate(tiles):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax)
            ax.imshow(im, interpolation="bilinear")
            if j >= 2:
                miou = METHODS[j - 2][2] - 2.1 * i + 1.0 * (i == 1)
                ax.text(0.975, 0.955, f"{miou:.1f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.0, color="white",
                        fontweight="bold" if j == ncol - 1 else "normal",
                        zorder=8,
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="#000000aa", edgecolor="none"))
            if j == ncol - 1:                 # circle the recovered regions
                px, py, ph = person
                for (ex, ey, ew, eh) in [(px, py - ph / 2, 30, ph + 16),
                                         (cars[-1][0], cars[-1][1] -
                                          cars[-1][3] / 2,
                                          cars[-1][2] * 1.7,
                                          cars[-1][3] * 2.6)]:
                    ax.add_patch(Ellipse((ex, ey), ew, eh, facecolor="none",
                                         edgecolor="white", lw=0.9,
                                         linestyle=(0, (2.4, 1.6)), zorder=7))
            if j == 0:
                ax.set_ylabel(rname, fontsize=7.8, labelpad=3.0)
            if i == 2:
                bold = col_names[j] == "Ours"
                ax.set_xlabel(col_names[j], fontsize=8.0, labelpad=3.2,
                              fontweight="bold" if bold else "normal",
                              color="#0b3d2e" if bold else "#1a1a1a")

    p0 = axes[0, ncol - 1].get_position()
    p1 = axes[2, ncol - 1].get_position()
    fig.add_artist(Rectangle((p0.x0 - 0.005, p1.y0 - 0.062), p0.width + 0.010,
                             p0.y1 - p1.y0 + 0.072, transform=fig.transFigure,
                             facecolor="#e8f2ec", edgecolor="none", zorder=-5))

    handles = [Patch(facecolor=c, edgecolor="#ffffff", linewidth=0.4, label=n)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    leg = fig.legend(handles=handles, ncol=7, loc="lower center",
                     bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=7.0,
                     handlelength=1.1, handleheight=0.9, columnspacing=1.4,
                     handletextpad=0.45)
    leg.set_zorder(10)

    savefig(fig, "255_cv_segmentation_compare.png")


if __name__ == "__main__":
    main()
