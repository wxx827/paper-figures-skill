"""UAV bird's-eye-view perception pipeline, three flight samples deep.

Five stages run left to right — oblique frame, monocular depth, unprojected
points, inverse-perspective RGB, learned BEV occupancy — and every stage
sits in an identically sized dashed frame so the three sample rows line up
into clean columns.  The same synthetic city scene drives all five renders
of a row, so a reader can follow one building through the whole chain; the
final column is boxed in red as the product of the pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pubstyle import set_pub_style, PALETTES, savefig, get_cmap

# occupancy is a plain unipolar score, so it gets a house sequential ramp
# rather than a full-spectrum one
BEV_CMAP = get_cmap("cb_sequential_warm")

ENC = "#3b5b92"
DEC = "#2e7d5b"
ATT = "#c9a227"
LOSS = "#b23a48"
DATA = "#8c8c8c"
AUX = "#8e6bbf"
INK = "#24303c"
MUTED = "#6f7783"
SOFT = "#f2f4f7"
LINE = "#dde2e8"

R = 0.055
DASH = (0, (4, 2.6))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.2, ls="-", ms=12, zorder=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def frame(ax, x, y, w, h, color=MUTED, lw=1.1, ls=DASH, fc="white",
          zorder=2, rounding=0.10):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc,
        edgecolor=color, linewidth=lw, linestyle=ls, zorder=zorder))


def panel(ax, x, y, w, h, fc=SOFT, ec=LINE, lw=0.8, ls="-", zorder=1,
          rounding=0.08):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))


def tensor_label(ax, x, y, text, color=MUTED, fontsize=6.5, ha="center",
                 va="center", **kw):
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=fontsize,
                   **kw)


def titles(ax, x, y, main, sub=None):
    ax.text(x, y, main, fontsize=13.5, fontweight="bold", color=INK,
            ha="left", va="center")
    if sub:
        ax.text(x, y - 0.32, sub, fontsize=8.6, color=MUTED, ha="left",
                va="center")


# ---- scene synthesis -------------------------------------------------------
N = 104


def make_scene(rng):
    blds = []
    for _ in range(9):
        w = rng.uniform(0.10, 0.20)
        h = rng.uniform(0.10, 0.20)
        x0 = rng.uniform(0.04, 0.96 - w)
        y0 = rng.uniform(0.04, 0.96 - h)
        blds.append((x0, x0 + w, y0, y0 + h, rng.uniform(0.35, 1.0)))
    trees = [(rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95),
              rng.uniform(0.022, 0.040)) for _ in range(14)]
    return dict(blds=blds, trees=trees)


def _grid():
    yy, xx = np.mgrid[0:N, 0:N] / (N - 1.0)
    return xx, yy


def _rect(img, x0, x1, y0, y1, col):
    i0, i1 = int(y0 * N), int(y1 * N)
    j0, j1 = int(x0 * N), int(x1 * N)
    img[i0:i1, j0:j1] = col


def _disc(img, cx, cy, r, col):
    xx, yy = _grid()
    img[np.hypot(xx - cx, yy - cy) <= r] = col


def aerial(scene, rng):
    img = np.full((N, N, 3), 0.0)
    img[:] = np.array([0.44, 0.43, 0.41])
    for band in (0.30, 0.70):
        _rect(img, 0.0, 1.0, band - 0.035, band + 0.035, [0.30, 0.30, 0.31])
        _rect(img, band - 0.035, band + 0.035, 0.0, 1.0, [0.30, 0.30, 0.31])
    for k, (x0, x1, y0, y1, hh) in enumerate(scene["blds"]):
        tone = 0.55 + 0.30 * hh
        _rect(img, x0 + 0.012, x1 + 0.012, y0 + 0.012, y1 + 0.012,
              [0.20, 0.20, 0.22])
        _rect(img, x0, x1, y0, y1,
              [tone, tone * (0.93 + 0.05 * (k % 2)), tone * 0.86])
    for cx, cy, r in scene["trees"]:
        _disc(img, cx, cy, r, [0.24, 0.42, 0.26])
    img += rng.normal(0, 0.018, img.shape)
    return np.clip(img, 0, 1)


def depth_field(scene):
    xx, yy = _grid()
    d = 0.86 - 0.16 * yy + 0.05 * np.sin(4.0 * xx)
    for x0, x1, y0, y1, hh in scene["blds"]:
        m = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        d[m] = 0.86 - 0.52 * hh
    for cx, cy, r in scene["trees"]:
        m = np.hypot(xx - cx, yy - cy) <= r
        d[m] = 0.60
    return d


def cloud_points(scene, rng):
    xs, ys, zs = [], [], []
    for x0, x1, y0, y1, hh in scene["blds"]:
        n = int(160 * (x1 - x0) * (y1 - y0) / 0.02)
        u = rng.uniform(x0, x1, n)
        v = rng.uniform(y0, y1, n)
        xs.append(u)
        ys.append(v)
        zs.append(np.full(n, hh) + rng.normal(0, 0.03, n))
    n = 1500
    xs.append(rng.uniform(0, 1, n))
    ys.append(rng.uniform(0, 1, n))
    zs.append(np.abs(rng.normal(0.10, 0.04, n)))
    for cx, cy, r in scene["trees"]:
        m = 40
        a = rng.uniform(0, 2 * np.pi, m)
        rr = r * np.sqrt(rng.uniform(0, 1, m))
        xs.append(cx + rr * np.cos(a))
        ys.append(cy + rr * np.sin(a))
        zs.append(rng.uniform(0.25, 0.45, m))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))


def project(img):
    out = np.full_like(img, 0.07)
    for i in range(N):
        t = i / (N - 1.0)
        half = 0.19 + 0.60 * t
        u = (np.arange(N) / (N - 1.0) - 0.5) / half + 0.5
        ok = (u >= 0) & (u < 1)
        si = int(np.clip(t * 0.94 * N, 0, N - 1))
        out[i][ok] = img[si][(u[ok] * (N - 1)).astype(int)]
    return out


def bev_field(scene, rng):
    xx, yy = _grid()
    f = 0.06 + 0.10 * np.exp(-((yy - 0.5) ** 2) / 0.18)
    for x0, x1, y0, y1, hh in scene["blds"]:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = 0.055 + 0.05 * hh
        f += (0.35 + 0.65 * hh) * np.exp(-(((xx - cx) ** 2 +
                                            (yy - cy) ** 2) / (2 * s ** 2)))
    f += 0.12 * rng.random((N, N))
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(f, 1.4)


# ---- layout ----------------------------------------------------------------
W, H = 12.9, 8.2
COLS = [1.475 + 2.470 * k for k in range(5)]
ROWS = [5.95, 4.20, 2.45]
FW, FH = 1.85, 1.61
IW, IH = 1.69, 1.45

STAGES = [("UAV Perspective", "oblique camera frame", DATA),
          ("Depth Estimation", "monocular metric depth", ENC),
          ("Point Cloud", "unprojected 3-D points", AUX),
          ("RGB Projection", "inverse perspective map", DEC),
          ("BEV Heat Map", "learned occupancy", LOSS)]
SHAPES = ["3 × 1080 × 1920", "1 × 1080 × 1920", "≈ 180 k points",
          "3 × 200 × 200", "C × 200 × 200"]


def img_axes(fig, x, y):
    axi = fig.add_axes([(x - IW / 2) / W, (y - IH / 2) / H, IW / W, IH / H])
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_visible(False)
    return axi


def draw_row(fig, ax, y, seed):
    rng = np.random.default_rng(seed)
    scene = make_scene(rng)
    rgb = aerial(scene, rng)

    axi = img_axes(fig, COLS[0], y)
    axi.imshow(rgb, interpolation="bilinear")

    axi = img_axes(fig, COLS[1], y)
    axi.imshow(depth_field(scene), cmap="plasma", interpolation="bilinear")

    axi = img_axes(fig, COLS[2], y)
    px, py, pz = cloud_points(scene, rng)
    axi.set_facecolor("#101821")
    axi.scatter(px, py, c=pz, s=0.62, cmap="viridis", vmin=0.0, vmax=1.05,
                linewidths=0)
    axi.set_xlim(0, 1)
    axi.set_ylim(1, 0)

    axi = img_axes(fig, COLS[3], y)
    axi.imshow(project(rgb), interpolation="bilinear")

    axi = img_axes(fig, COLS[4], y)
    axi.imshow(bev_field(scene, rng), cmap=BEV_CMAP, interpolation="bilinear")

    for k, x in enumerate(COLS):
        col = LOSS if k == 4 else "#b6bfc9"
        frame(ax, x, y, FW, FH, color=col, lw=1.3 if k == 4 else 1.0)
        if k < 4:
            arrow(ax, (x + FW / 2 + 0.05, y), (COLS[k + 1] - FW / 2 - 0.05, y),
                  color="#8f9aa6", lw=2.2, ms=17)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    titles(ax, 0.30, 7.88,
           "UAV bird's-eye-view perception pipeline, stage by stage",
           "Three flight samples share the five-stage column layout; the red "
           "frame marks the product delivered to the planner")

    for k, ((name, sub, col), x) in enumerate(zip(STAGES, COLS)):
        x0 = x - FW / 2
        ax.add_patch(FancyBboxPatch(
            (x0, 7.06), 0.30, 0.24,
            boxstyle=f"round,pad=0,rounding_size=0.05", facecolor=col,
            edgecolor="none", zorder=4))
        ax.text(x0 + 0.15, 7.18, str(k + 1), ha="center", va="center",
                color="white", fontsize=7.6, fontweight="bold", zorder=5)
        ax.text(x0 + 0.40, 7.18, name, ha="left", va="center", fontsize=8.6,
                fontweight="bold", color=INK)
        ax.text(x0 + 0.02, 6.92, sub, ha="left", va="center", fontsize=6.8,
                color=MUTED)
        tensor_label(ax, x, 1.52, SHAPES[k])

    for k, (y, seed) in enumerate(zip(ROWS, (5, 19, 41))):
        draw_row(fig, ax, y, seed)
        ax.text(0.28, y, f"scene {k + 1}", rotation=90, ha="center",
                va="center", fontsize=7.4, color=MUTED)

    # colorbar for the BEV product only
    axc = fig.add_axes([(COLS[4] - IW / 2) / W, 1.06 / H, IW / W, 0.13 / H])
    axc.imshow(np.linspace(0, 1, 256)[None, :], cmap=BEV_CMAP, aspect="auto")
    axc.set_yticks([])
    axc.set_xticks([0, 255])
    axc.set_xticklabels(["low", "high"], fontsize=6.0)
    axc.tick_params(length=0, pad=1.5)
    for sp in axc.spines.values():
        sp.set_linewidth(0.6)
        sp.set_edgecolor("#b9c2cc")
    ax.text(COLS[4], 1.30, "occupancy score", fontsize=6.5, color=MUTED,
            ha="center", va="center")

    panel(ax, 4.52, 1.02, 7.35, 0.80)
    notes = ["Depth is metric and monocular; no LiDAR is used in this path.",
             "Points are unprojected with the pose recorded at capture time.",
             "The BEV grid is 0.25 m per cell over a 50 m × 50 m window."]
    for k, t in enumerate(notes):
        ax.text(1.02, 1.24 - 0.26 * k, "•  " + t, fontsize=7.0, color=INK,
                ha="left", va="center", zorder=4)

    savefig(fig, "267_net_uav_bev_pipeline.png")


if __name__ == "__main__":
    main()
