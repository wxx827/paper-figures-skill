"""Figure 541 - Nuclear instance segmentation compared across three methods.

Rows are three 20x fields with deliberately different nuclear morphology
(tumour nest, stromal margin, lymphoid aggregate); columns are the H&E tile,
the ground-truth instance map and three predictions.  Every instance gets its
own colour so that a *merge* - two touching nuclei predicted as one blob, the
classic U-Net failure - is visible as a single colour spanning two nuclei
rather than as a subtle boundary difference.

Dice and AJI are computed at run time from the rasterised masks themselves,
so the numbers printed on each tile cannot drift away from what is drawn.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from pubstyle import set_pub_style, savefig, darken, lighten
from figkit import image_axes, corner_stamp, background_card, halo_text

OD_H = np.array([0.650, 0.704, 0.286])
OD_E = np.array([0.092, 0.954, 0.283])

# Arbitrary instance-identity colours: these encode *no* semantics, they only
# have to separate neighbouring objects, so the house qualitative order is
# extended with its own supplementary tones rather than a rainbow map.
INSTANCE = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf",
            "#c0603f", "#2a7f7a", "#7b3294"]
OURS = "#3b5b92"
TILE_EDGE = "#b0b0b0"
MPP = 0.25                                  # 40x scan, micrometres per pixel
TH, TW = 360, 424                           # tile = 90 x 106 micrometres


# ---------------------------------------------------------------------------
# Synthetic H&E with matching instance labels
# ---------------------------------------------------------------------------
def _noise(rng, shape, sigma):
    a = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return a / (a.std() + 1e-9)


def _fbm(rng, shape, scales=(1.2, 4.0, 12.0), weights=(0.44, 0.34, 0.22)):
    out = sum(w * _noise(rng, shape, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def he_rgb(ch, ce, desat=0.10):
    od = (np.clip(ch, 0, None)[..., None] * OD_H
          + np.clip(ce, 0, None)[..., None] * OD_E)
    rgb = np.exp(-od)
    lum = rgb @ np.array([0.30, 0.59, 0.11])
    return np.clip(rgb * (1.0 - desat) + lum[..., None] * desat, 0, 1)


def field_maps(kind, shape, rng):
    """Tumour / lymphocyte probability for one of the three field types."""
    hh, ww = shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(float)
    f = _fbm(rng, shape, (6, 16, 34), (0.44, 0.34, 0.22))
    if kind == "tumour":
        band = np.exp(-0.5 * ((yy - 0.78 * hh + 0.30 * (xx - 0.5 * ww))
                              / (0.12 * hh)) ** 2)
        p_t = np.clip(0.92 + 0.30 * f - 0.85 * band, 0, 1)
        p_l = np.clip(0.10 * f, 0, 1)
    elif kind == "stroma":
        p_t = np.clip(0.26 + 0.55 * f, 0, 1)
        p_l = np.clip(0.30 * _fbm(rng, shape, (4, 10), (.6, .4)) - 0.05, 0, 1)
    else:                                              # lymphoid aggregate
        blob = np.exp(-0.5 * (((yy - 0.44 * hh) / (0.30 * hh)) ** 2
                              + ((xx - 0.46 * ww) / (0.32 * ww)) ** 2))
        p_l = np.clip((blob + 0.30 * f - 0.42) / 0.30, 0, 1)
        p_t = np.clip(0.42 * (1.0 - p_l) + 0.25 * f, 0, 1)
    return p_t, p_l


def make_nuclei(rng, shape, p_t, p_l):
    """Rejection-sampled nucleus parameters honouring the density fields."""
    hh, ww = shape
    dens = np.clip(1100 + 3600 * p_t + 8600 * p_l, 60, 12000)
    dens_px = dens * (MPP / 1000.0) ** 2
    r_px = np.clip(2.25 + 1.85 * p_t - 0.55 * p_l, 1.1, None) / MPP
    peak = float(dens_px.max())
    n_try = int(np.clip(26.0 * peak * hh * ww, 200, 90000))
    ys = rng.integers(0, hh, n_try)
    xs = rng.integers(0, ww, n_try)
    keep = rng.random(n_try) < (dens_px[ys, xs] / peak)
    occ = np.zeros(shape, bool)
    items = []
    for y, x in zip(ys[keep], xs[keep]):
        r = r_px[y, x] * rng.uniform(0.74, 1.34)
        rr = int(max(1, round(0.92 * r)))
        y0, y1 = max(0, y - rr), min(hh, y + rr + 1)
        x0, x1 = max(0, x - rr), min(ww, x + rr + 1)
        if occ[y0:y1, x0:x1].any():
            continue
        occ[y0:y1, x0:x1] = True
        e = rng.uniform(1.0, 1.12 + 1.05 * (1 - p_t[y, x]) * (1 - p_l[y, x]))
        items.append(dict(cy=float(y), cx=float(x), a=r * e, b=r / e,
                          th=rng.uniform(0, np.pi), ph=rng.uniform(0, 6.28),
                          amp=0.88 + 0.42 * p_t[y, x] + 0.85 * p_l[y, x]))
    return items


def _ellipse_patch(shape, it, scale=1.0):
    """Local (slice, distance) pair for one wobbly ellipse."""
    hh, ww = shape
    a, b = it["a"] * scale, it["b"] * scale
    rad = int(np.ceil(1.9 * max(a, b))) + 2
    y0, y1 = max(0, int(it["cy"]) - rad), min(hh, int(it["cy"]) + rad + 1)
    x0, x1 = max(0, int(it["cx"]) - rad), min(ww, int(it["cx"]) + rad + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dy, dx = yy - it["cy"], xx - it["cx"]
    ct, st = np.cos(it["th"]), np.sin(it["th"])
    u = (dx * ct + dy * st) / max(a, 1e-6)
    v = (-dx * st + dy * ct) / max(b, 1e-6)
    ang = np.arctan2(v, u)
    wob = 1.0 + 0.14 * np.sin(3 * ang + it["ph"]) + 0.08 * np.sin(5 * ang
                                                                  - it["ph"])
    return (slice(y0, y1), slice(x0, x1)), np.hypot(u, v) / wob


def rasterize(shape, items, ids=None, scale=1.0):
    lab = np.zeros(shape, np.int32)
    ids = range(1, len(items) + 1) if ids is None else ids
    for it, k in zip(items, ids):
        got = _ellipse_patch(shape, it, scale)
        if got is None:
            continue
        sl, d = got
        lab[sl] = np.where(d <= 1.0, k, lab[sl])
    return lab


def render_he(shape, items, p_t, p_l, rng):
    ce = (0.74 - 0.20 * p_t - 0.26 * p_l)
    ce *= 1.0 + 0.24 * _fbm(rng, shape, (1.0, 3.5, 10.0), (.42, .34, .24))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    fib = np.sin((0.83 * xx + 0.56 * yy) / 4.5 + 2.4 * _noise(rng, shape, 7.0))
    ce += 0.10 * fib * (1.0 - p_t) * (1.0 - p_l)
    ch = 0.07 + 0.06 * p_t
    for it in items:
        got = _ellipse_patch(shape, it, 1.0)
        if got is None:
            continue
        sl, d = got
        ch[sl] += it["amp"] * np.clip(1.18 - d * d, 0, 1) ** 0.55
        ce[sl] += 0.13 * np.clip(1.9 - d * d, 0, 1) ** 1.6
    ch *= 1.0 + 0.30 * _fbm(rng, shape, (0.7, 2.0), (.6, .4))
    rgb = he_rgb(ch, ce)
    rgb = np.dstack([gaussian_filter(rgb[..., c], 0.45) for c in range(3)])
    return np.clip(rgb + 0.008 * rng.standard_normal(rgb.shape), 0, 1)


# ---------------------------------------------------------------------------
# Predicted instance maps: three characteristic failure profiles
# ---------------------------------------------------------------------------
def predict(shape, items, rng, drop=0.10, merge=0.16, shift=0.9, scale=1.06,
            wob=0.9, fp=5):
    """Perturb the ground-truth objects into a plausible prediction."""
    n = len(items)
    ids = np.arange(1, n + 1)
    if n > 2 and merge > 0:
        pts = np.array([[it["cy"], it["cx"]] for it in items])
        _, nb = cKDTree(pts).query(pts, k=2)
        pick = rng.random(n) < merge
        ids[pick] = ids[nb[pick, 1]]           # share the neighbour's identity
    keep = rng.random(n) >= drop
    out, oids = [], []
    for it, k, ok in zip(items, ids, keep):
        if not ok:
            continue
        j = dict(it)
        j["cy"] += shift * rng.standard_normal()
        j["cx"] += shift * rng.standard_normal()
        j["a"] *= scale * rng.uniform(0.90, 1.12)
        j["b"] *= scale * rng.uniform(0.90, 1.12)
        j["ph"] += wob * rng.standard_normal()
        out.append(j)
        oids.append(int(k))
    nxt = n + 1
    for _ in range(fp):                        # spurious detections
        out.append(dict(cy=rng.uniform(6, shape[0] - 6),
                        cx=rng.uniform(6, shape[1] - 6),
                        a=rng.uniform(3.5, 7.0), b=rng.uniform(3.0, 6.0),
                        th=rng.uniform(0, np.pi), ph=rng.uniform(0, 6.28),
                        amp=1.0))
        oids.append(nxt)
        nxt += 1
    lab = rasterize(shape, out, oids)
    return relabel(lab)


def relabel(lab):
    ids = np.unique(lab)
    ids = ids[ids > 0]
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    lut[ids] = np.arange(1, ids.size + 1)
    return lut[lab]


def dice(gt, pred):
    a, b = gt > 0, pred > 0
    return 2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def aji(gt, pred):
    """Aggregated Jaccard index (Kumar et al. 2017), contingency-table form."""
    ng, npd = int(gt.max()), int(pred.max())
    if ng == 0 or npd == 0:
        return 0.0
    hist = np.bincount((gt.ravel().astype(np.int64) * (npd + 1)
                        + pred.ravel()),
                       minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
    ag, ap = hist.sum(1), hist.sum(0)
    inter = hist[1:, 1:]
    union = ag[1:, None] + ap[None, 1:] - inter
    iou = inter / np.maximum(union, 1)
    best = iou.argmax(1)
    bestv = iou.max(1)
    used = np.zeros(npd, bool)
    c = u = 0
    for i in range(ng):
        if bestv[i] <= 0:
            u += ag[i + 1]
            continue
        j = best[i]
        c += inter[i, j]
        u += union[i, j]
        used[j] = True
    u += ap[1:][~used].sum()
    return float(c) / max(u, 1)


def overlay(base, lab, fill=0.40, edge_lw=True):
    """Composite per-instance colour fill and a darker outline onto the tile."""
    n = int(lab.max())
    lut = np.zeros((n + 1, 3))
    dark = np.zeros((n + 1, 3))
    for k in range(1, n + 1):
        c = INSTANCE[(k * 3) % len(INSTANCE)]
        lut[k] = to_rgb(lighten(c, 0.26))
        dark[k] = to_rgb(darken(c, 0.40))
    out = base.copy()
    m = lab > 0
    out[m] = base[m] * (1 - fill) + lut[lab[m]] * fill
    if edge_lw:
        b = np.zeros(lab.shape, bool)
        b[:-1, :] |= lab[:-1, :] != lab[1:, :]
        b[1:, :] |= lab[:-1, :] != lab[1:, :]
        b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
        b &= m
        out[b] = dark[lab[b]]
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
ROWS = [("tumour", "Tumour nest"), ("stroma", "Stromal margin"),
        ("lymph", "Lymphoid aggregate")]
METHODS = [("U-Net", dict(drop=0.14, merge=0.26, shift=1.5, scale=1.10,
                          wob=1.5, fp=9)),
           ("HoVer-Net", dict(drop=0.08, merge=0.11, shift=1.0, scale=1.05,
                              wob=1.0, fp=5)),
           ("Ours", dict(drop=0.03, merge=0.03, shift=0.55, scale=1.01,
                         wob=0.5, fp=2))]
COLS = ["H&E tile"] + ["Ground truth"] + [m for m, _ in METHODS]


def main():
    set_pub_style(base_font=8)
    shape = (TH, TW)
    tiles, gts, preds, stats = [], [], [], []
    for r, (kind, _) in enumerate(ROWS):
        rng = np.random.default_rng(200 + 7 * r)
        p_t, p_l = field_maps(kind, shape, rng)
        items = make_nuclei(rng, shape, p_t, p_l)
        tiles.append(render_he(shape, items, p_t, p_l, rng))
        gt = relabel(rasterize(shape, items))
        gts.append(gt)
        preds.append([predict(shape, items, np.random.default_rng(900 + 11 * r
                                                                  + 3 * k),
                              **kw) for k, (_, kw) in enumerate(METHODS)])
        areas = np.bincount(gt.ravel())[1:] * MPP ** 2
        stats.append((int(gt.max()), float(areas.mean())))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 5, wspace=0.05, hspace=0.09, left=0.062,
                          right=0.988, top=0.884, bottom=0.105)
    axes = np.empty((3, 5), object)
    for r in range(3):
        for c in range(5):
            ax = fig.add_subplot(gs[r, c])
            axes[r, c] = ax
            ours = c == 4
            image_axes(ax, color=OURS if ours else TILE_EDGE,
                       lw=1.1 if ours else 0.55)
            if c == 0:
                img = tiles[r]
            elif c == 1:
                img = overlay(tiles[r], gts[r], fill=0.58)
            else:
                img = overlay(tiles[r], preds[r][c - 2], fill=0.48)
            ax.imshow(img, interpolation="bilinear", zorder=1)
            if r == 0:
                ax.set_title(COLS[c], fontsize=8.4, pad=3.4,
                             fontweight="bold" if ours else "normal",
                             color="#1a1a1a")
            if c == 0:
                ax.set_ylabel(ROWS[r][1], fontsize=8.0, labelpad=3.4,
                              color="#1a1a1a")
            if c >= 2:
                d = dice(gts[r], preds[r][c - 2])
                a = aji(gts[r], preds[r][c - 2])
                corner_stamp(ax, f"Dice {d:.3f}   AJI {a:.3f}",
                             loc="lower left", fontsize=6.4,
                             weight="bold" if ours else "normal")
            if c == 1:
                corner_stamp(ax, f"{stats[r][0]} instances", loc="lower left",
                             fontsize=6.4)

    for ax in axes.ravel():
        ax.apply_aspect()
    p0, p1 = axes[0, 4].get_position(), axes[2, 4].get_position()
    background_card(fig, (p0.x0 - 0.012, p1.y0 - 0.030,
                          p0.width + 0.024, p0.y1 - p1.y0 + 0.058),
                    facecolor="#eaeff6", zorder=-5, radius=0.006)

    halo_text(axes[0, 0], 0.030, 0.955, "40x  ·  0.25 µm/px",
              transform=axes[0, 0].transAxes, ha="left", va="top",
              fontsize=6.4, color="#1a1a1a", halo="white", halo_lw=2.3)

    counts = "     ".join(
        f"{name}: {n} nuclei, mean area {a:.1f} µm²"
        for (_, name), (n, a) in zip(ROWS, stats))
    savefig(fig, "541_path_nuclei_segmentation.png")


if __name__ == "__main__":
    main()
