"""Figure 281 - Nearest-neighbour retrieval in the learned embedding.

Each row starts from a query (blue frame) and shows the five closest gallery
items by cosine similarity.  Correct classes are framed green, confusions red.
The subordinate strip on the right reports precision@5 for the same rows, with
its bars aligned to the image rows.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, style_axes

OK_C, BAD_C, Q_C = "#2e7d5b", "#b23a48", "#3b5b92"
N = 64
CLASSES = ["mug", "butterfly", "boat", "bird", "car"]


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(0.8, 2.2, 5.5), weights=(0.45, 0.33, 0.22)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _ellipse(yy, xx, cx, cy, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return (u / max(rx, 1e-6)) ** 2 + (v / max(ry, 1e-6)) ** 2 <= 1.0


def _tri(yy, xx, p0, p1, p2, n):
    pts = [np.array(p) * n for p in (p0, p1, p2)]
    for sign in (1, -1):
        inside = np.ones(yy.shape, bool)
        for a, b in zip(pts, pts[1:] + pts[:1]):
            side = ((b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0]))
            inside &= sign * side >= 0
        if inside.any():
            return inside
    return np.zeros(yy.shape, bool)


def _soft(mask, sigma=0.7):
    return np.clip(gaussian_filter(mask.astype(float), sigma), 0, 1)


def _paint(img, mask, value, texture):
    return img * (1 - mask) + mask * np.clip(value + texture, 0.02, 0.99)


def render(cls, seed, n=N):
    """One instance of a class, with pose, scale and lighting jitter."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    tex = 0.030 * _fbm(rng, n)
    s = rng.uniform(0.86, 1.14)                    # scale jitter
    ox = rng.uniform(-0.05, 0.05)                  # position jitter
    oy = rng.uniform(-0.04, 0.04)
    tone = rng.uniform(-0.10, 0.10)

    bg = 0.62 + tone + 0.24 * _fbm(rng, n, (4, 10, 20))
    for _ in range(4):
        bx, by = rng.uniform(0, n, 2)
        rr = rng.uniform(0.10, 0.26) * n
        bg += rng.uniform(-0.18, 0.18) * np.exp(
            -(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * rr ** 2)))
    img = gaussian_filter(bg, 2.6)

    def E(cx, cy, rx, ry, ang=0.0):
        return _ellipse(yy, xx, (cx + ox) * n, (cy + oy) * n,
                        rx * s * n, ry * s * n, ang)

    if cls == "mug":
        handle = (E(0.70, 0.53, 0.105, 0.088)
                  & ~E(0.70, 0.53, 0.060, 0.046))
        img = _paint(img, _soft(handle), 0.28, tex)
        body = (np.abs(xx - (0.47 + ox) * n) < 0.165 * s * n) \
            & (yy > (0.375 + oy) * n) & (yy < (0.665 + oy) * n)
        body |= E(0.47, 0.665, 0.165, 0.050)
        img = _paint(img, _soft(body), 0.88 - 0.5 * (rng.random() < 0.35),
                     tex * 0.8)
        img = _paint(img, _soft(E(0.47, 0.375, 0.165, 0.055)), 0.62, tex * 0.6)
        img = _paint(img, _soft(E(0.47, 0.378, 0.128, 0.038)), 0.22, tex)

    elif cls == "butterfly":
        wings = np.zeros((n, n), bool)
        for sx in (-1, 1):
            wings |= E(0.50 + sx * 0.150, 0.42, 0.150, 0.112, sx * 28)
            wings |= E(0.50 + sx * 0.112, 0.60, 0.098, 0.082, -sx * 22)
        img = _paint(img, _soft(wings), 0.24, tex * 1.5)
        for sx in (-1, 1):
            img = _paint(img, _soft(E(0.50 + sx * 0.150, 0.405, 0.048, 0.040)),
                         0.86, tex * 0.5)
        img = _paint(img, _soft(E(0.50, 0.50, 0.022, 0.145)), 0.12, tex * 0.6)

    elif cls == "boat":
        hull = _tri(yy, xx, (0.30 + ox, 0.64 + oy), (0.70 + ox, 0.64 + oy),
                    (0.56 + ox, 0.74 + oy), n)
        img = _paint(img, _soft(yy > (0.66 + oy) * n, 1.2), 0.44, tex * 1.2)
        sail = _tri(yy, xx, (0.51 + ox, 0.19 + oy), (0.51 + ox, 0.63 + oy),
                    (0.76 + ox, 0.63 + oy), n)
        jib = _tri(yy, xx, (0.49 + ox, 0.23 + oy), (0.49 + ox, 0.63 + oy),
                   (0.30 + ox, 0.63 + oy), n)
        img = _paint(img, _soft(sail), 0.93, tex * 0.7)
        img = _paint(img, _soft(jib), 0.79, tex * 0.8)
        img = _paint(img, _soft(hull), 0.18, tex)

    elif cls == "bird":
        sub = E(0.50, 0.54, 0.150, 0.205, 14)
        sub |= E(0.60, 0.30, 0.085, 0.080)
        sub |= E(0.46, 0.53, 0.080, 0.150, 24)
        sub |= E(0.38, 0.80, 0.052, 0.130, 22)
        img = _paint(img, _soft(sub), 0.28, tex * 1.7)
        img = _paint(img, _soft(E(0.685, 0.325, 0.050, 0.026, -14)), 0.72,
                     tex * 0.5)
        img = _paint(img, _soft(E(0.622, 0.278, 0.018, 0.016)), 0.08, tex * 0.3)

    else:                                            # car
        img = _paint(img, _soft(yy > (0.70 + oy) * n, 1.2), 0.32, tex * 0.9)
        chassis = (np.abs(xx - (0.50 + ox) * n) < 0.300 * s * n) \
            & (yy > (0.545 + oy) * n) & (yy < (0.685 + oy) * n)
        chassis |= E(0.50, 0.61, 0.320, 0.085)
        cabin = E(0.50, 0.525, 0.175, 0.085) & (yy < (0.578 + oy) * n)
        img = _paint(img, _soft(chassis | cabin), 0.22, tex * 1.1)
        img = _paint(img, _soft(E(0.50, 0.53, 0.140, 0.058)
                                & (yy < (0.548 + oy) * n)), 0.68, tex * 0.6)
        for wx in (-0.185, 0.185):
            img = _paint(img, _soft(E(0.50 + wx, 0.690, 0.060, 0.056)), 0.08,
                         tex * 0.4)

    img += 0.010 * rng.standard_normal((n, n))
    lo, hi = np.percentile(img, [1.5, 98.5])
    return np.clip((img - lo) / (hi - lo), 0, 1)


# (query class, [(retrieved class, similarity), ...])
ROWS = [
    ("mug", [("mug", 0.961), ("mug", 0.943), ("mug", 0.918),
             ("mug", 0.902), ("mug", 0.884)]),
    ("butterfly", [("butterfly", 0.948), ("butterfly", 0.930),
                   ("butterfly", 0.905), ("bird", 0.871),
                   ("butterfly", 0.858)]),
    ("boat", [("boat", 0.955), ("boat", 0.934), ("boat", 0.911),
              ("boat", 0.893), ("boat", 0.870)]),
    ("bird", [("bird", 0.929), ("bird", 0.902), ("butterfly", 0.884),
              ("bird", 0.866), ("boat", 0.841)]),
    ("car", [("car", 0.958), ("car", 0.936), ("car", 0.907),
             ("boat", 0.879), ("car", 0.861)]),
]


def framed(ax, color, lw):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor=color, linewidth=lw, zorder=8,
                           clip_on=False))


def main():
    set_pub_style(base_font=8)
    rows = len(ROWS)
    cols = 6

    fig = plt.figure(figsize=(7.2, 5.05))
    left, right = 0.048, 0.735
    ws, hs = 0.05, 0.05
    tile = (right - left) * 7.2 / (cols + (cols - 1) * ws)
    gh = tile * (rows + (rows - 1) * hs) / 5.05
    gtop = 0.858
    gbot = gtop - gh

    gs = fig.add_gridspec(rows, cols, left=left, right=right, top=gtop,
                          bottom=gbot, wspace=ws, hspace=hs)

    prec = []
    for r, (qcls, hits) in enumerate(ROWS):
        for c in range(cols):
            ax = fig.add_subplot(gs[r, c])
            if c == 0:
                ax.imshow(render(qcls, 700 + 31 * r), cmap="gray",
                          vmin=0, vmax=1, interpolation="bilinear")
                framed(ax, Q_C, 1.5)
            else:
                rcls, sim = hits[c - 1]
                ax.imshow(render(rcls, 7000 + 91 * r + 13 * c), cmap="gray",
                          vmin=0, vmax=1, interpolation="bilinear")
                good = rcls == qcls
                framed(ax, OK_C if good else BAD_C, 1.1)
                ax.text(0.5, 0.035, f"{sim:.3f}", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=6.0, zorder=9,
                        color="#1a1a1a" if good else "#8c2f2f",
                        bbox=dict(boxstyle="round,pad=0.16",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.86))
            if r == 0:
                ax.set_title("Query" if c == 0 else f"Rank {c}",
                             fontsize=7.4, pad=3.6,
                             fontweight="bold" if c == 0 else "normal",
                             color=Q_C if c == 0 else "#1a1a1a")
            if c == 0:
                ax.set_ylabel(qcls, fontsize=7.2, fontweight="bold",
                              labelpad=4)
        prec.append(sum(h[0] == qcls for h in hits) / 5.0)

    axb = fig.add_axes([0.792, gbot, 0.112, gh])
    style_axes(axb)
    span = rows + (rows - 1) * hs
    centres = [((i * (1 + hs)) + 0.5) / span for i in range(rows)]
    bar_h = 0.34 / span
    for yc, p in zip(centres, prec):
        axb.barh(yc, p, height=bar_h, color=OK_C if p == 1.0 else "#c9a227",
                 edgecolor="white", linewidth=0.5)
        axb.text(p + 0.03, yc, f"{p:.2f}", ha="left", va="center",
                 fontsize=6.3, color="#4a4a4a")
    axb.set_ylim(1, 0)
    axb.set_xlim(0, 1.30)
    axb.set_yticks([])
    axb.set_xticks([0, 0.5, 1.0])
    axb.tick_params(labelsize=6.2)
    axb.spines["left"].set_visible(False)
    axb.set_xlabel("precision@5", fontsize=6.9, labelpad=2.0)
    axb.set_title("retrieval quality", fontsize=7.0, color="#4a4a4a",
                  pad=3.6)
    axb.text(0.0, -0.125, f"mean P@5 = {np.mean(prec):.2f}",
             transform=axb.transAxes, ha="left", va="top", fontsize=6.4,
             fontweight="bold", color="#3a3a3a")

    fig.text(left, gbot - 0.048,
             "blue frame = query,   green = same class,   red = confusion;   "
             "numbers are cosine similarity to the query",
             fontsize=6.6, color="#5a5a5a", ha="left", va="top")


    savefig(fig, "281_prototype_retrieval.png")


if __name__ == "__main__":
    main()
