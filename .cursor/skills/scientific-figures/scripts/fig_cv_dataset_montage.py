"""Figure 261 - Dataset sample montage with a subordinate statistics column.

A 6 x 8 thumbnail grid shows the six capture conditions of a synthetic driving
corpus.  The class-count bars and the summary card on the right are drawn at a
fraction of the montage width so that they read as supporting material.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from pubstyle import set_pub_style, PALETTES, savefig, style_axes

T = 72          # thumbnail edge in pixels
NROW, NCOL = 6, 8

CONDITIONS = [
    ("Clear day", "#3f7fbf", dict(sky=("#3d76bd", "#cfe0ef"), gnd="#5a6b45",
                                  light=1.00, veil=0.00, tint="#ffffff")),
    ("Dusk", "#d98032", dict(sky=("#8c3f5a", "#f0a860"), gnd="#4a4432",
                             light=0.82, veil=0.05, tint="#ffb877")),
    ("Night", "#2f3b57", dict(sky=("#0b1226", "#22304d"), gnd="#1c2028",
                              light=0.42, veil=0.02, tint="#8ea8d8")),
    ("Rain", "#4c78a8", dict(sky=("#5a6b7a", "#a9b4bd"), gnd="#3f4a3c",
                             light=0.74, veil=0.16, tint="#c8d4dc")),
    ("Fog", "#9aa5ad", dict(sky=("#b8bfc4", "#dfe3e6"), gnd="#8d9186",
                            light=0.90, veil=0.42, tint="#e8ecee")),
    ("Snow", "#7fa8c9", dict(sky=("#8fa6bd", "#e6eef5"), gnd="#d8dee2",
                             light=0.96, veil=0.20, tint="#ffffff")),
]
COUNTS = [4180, 1740, 2260, 1620, 1180, 1500]


def _hex(c):
    return np.array(plt.matplotlib.colors.to_rgb(c))


def image_axes(ax, color="#ffffff", lw=0.5):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def thumbnail(cond, seed):
    """Tiny procedural street view under one capture condition."""
    p = CONDITIONS[cond][2]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:T, 0:T].astype(float)
    hor = int(rng.integers(28, 38))

    g = np.clip(yy / hor, 0, 1)[..., None]
    img = _hex(p["sky"][0]) * (1 - g) + _hex(p["sky"][1]) * g
    img = np.where((yy >= hor)[..., None],
                   _hex(p["gnd"]) * (1 + 0.12 * gaussian_filter(
                       rng.standard_normal((T, T)), 1.6))[..., None], img)

    # skyline
    x = 0
    while x < T:
        bw = int(rng.integers(5, 13))
        top = hor - int(rng.integers(4, 22))
        blk = (yy >= top) & (yy < hor + 1) & (xx >= x) & (xx < x + bw)
        base = _hex("#5a5f66") if cond != 2 else _hex("#171c2a")
        img = np.where(blk[..., None], base * (0.8 + 0.45 * rng.random()), img)
        if cond == 2 and rng.random() < 0.8:                # lit windows
            win = blk & ((xx % 3) < 1) & ((yy % 4) < 2)
            img = np.where(win[..., None], _hex("#f3d98a"), img)
        x += bw + int(rng.integers(1, 4))

    # road wedge + vehicles
    t = np.clip((yy - hor) / (T - hor), 0, 1)
    left, right = 30 + (2 - 30) * t, 42 + (70 - 42) * t
    road = (yy >= hor) & (xx >= left) & (xx <= right)
    img = np.where(road[..., None], _hex("#3c3f46") if cond != 5
                   else _hex("#b9c2c8"), img)
    lane = road & (np.abs(xx - (left + right) / 2) < 0.6 + 0.8 * t) & \
        ((yy * (0.8 + 1.4 * t) % 9) < 4)
    img = np.where(lane[..., None], _hex("#d8d2bd"), img)

    for _ in range(int(rng.integers(1, 4))):
        d = rng.uniform(0.25, 0.95)
        cy = hor + d * (T - hor)
        cw, ch = 10 * (0.3 + d), 6 * (0.3 + d)
        cx = float(np.clip((30 + (2 - 30) * d + 42 + (70 - 42) * d) / 2
                           + rng.uniform(-1, 1) * 9 * d, cw, T - cw))
        car = (np.abs(xx - cx) < cw / 2) & (yy > cy - ch) & (yy < cy)
        col = _hex(["#b8452f", "#2f5f9e", "#c9a227", "#8b8f96"]
                   [int(rng.integers(0, 4))])
        img = np.where(car[..., None], col, img)
        if cond == 2:                                       # head lights
            hl = np.exp(-(((xx - cx) ** 2 + (yy - cy + ch * 0.25) ** 2)
                          / (2 * (0.42 * cw) ** 2)))
            img = np.clip(img + 0.85 * hl[..., None] * _hex("#ffe08a"), 0, 1)

    for _ in range(int(rng.integers(1, 4))):
        tx = rng.uniform(4, T - 4)
        ty = hor + rng.uniform(2, 16)
        r = rng.uniform(3.5, 7.0)
        crown = ((xx - tx) / r) ** 2 + ((yy - (ty - 1.8 * r)) / r) ** 2 < 1
        img = np.where(crown[..., None],
                       _hex("#2c4a26") if cond != 5 else _hex("#5c6b5a"), img)

    if cond == 3:                                           # rain streaks
        for _ in range(90):
            x0, y0 = rng.uniform(0, T), rng.uniform(0, T)
            ln = rng.uniform(4, 10)
            ii = np.clip((y0 + np.arange(ln)).astype(int), 0, T - 1)
            jj = np.clip((x0 + 0.30 * np.arange(ln)).astype(int), 0, T - 1)
            img[ii, jj] = np.clip(img[ii, jj] + 0.30, 0, 1)
    if cond == 5:                                           # falling snow
        m = rng.random((T, T)) < 0.020
        img = np.where(m[..., None], _hex("#ffffff"), img)

    img = img * p["light"]
    img = img * (1 - p["veil"]) + p["veil"] * _hex(p["tint"])
    img = np.clip(img * (1 + 0.05 * gaussian_filter(
        rng.standard_normal((T, T)), 1.2))[..., None], 0, 1)
    return np.clip(np.dstack([gaussian_filter(img[..., k], 0.35)
                              for k in range(3)]), 0, 1)


def main():
    set_pub_style(base_font=8)

    fig = plt.figure(figsize=(7.6, 4.05))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.55, 1.0], wspace=0.175,
                             left=0.062, right=0.988, top=0.892, bottom=0.055)
    grid = outer[0, 0].subgridspec(NROW, NCOL, wspace=0.035, hspace=0.035)
    side = outer[0, 1].subgridspec(2, 1, height_ratios=[1.55, 1.0],
                                   hspace=0.30)

    first_axes = []
    for i in range(NROW):
        for j in range(NCOL):
            ax = fig.add_subplot(grid[i, j])
            image_axes(ax)
            ax.imshow(thumbnail(i, seed=1000 * i + 7 * j + 3),
                      interpolation="bilinear")
            ax.text(0.045, 0.045, f"{'CDNRFS'[i]}-{j + 1:02d}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=6.0, color="white", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.16",
                              facecolor="#00000088", edgecolor="none"))
            if j == 0:
                first_axes.append(ax)
                ax.set_ylabel(CONDITIONS[i][0], fontsize=7.2, labelpad=2.5)

    # ---- subordinate panel 1: per-condition sample counts
    axb = fig.add_subplot(side[0, 0])
    ypos = np.arange(NROW)[::-1]
    cols = [c for _, c, _ in CONDITIONS]
    axb.barh(ypos, COUNTS, height=0.66, color=cols, edgecolor="white",
             linewidth=0.5)
    for y, v in zip(ypos, COUNTS):
        axb.text(v + 90, y, f"{v:,}", va="center", ha="left", fontsize=6.0,
                 color="#4a4a4a")
    axb.set_yticks(ypos)
    axb.set_yticklabels([n for n, _, _ in CONDITIONS], fontsize=6.2)
    axb.set_xlim(0, max(COUNTS) * 1.30)
    axb.set_xlabel("frames per condition", fontsize=6.4, labelpad=1.5)
    axb.tick_params(axis="x", labelsize=6.0, length=2.0)
    axb.tick_params(axis="y", length=0)
    style_axes(axb)
    axb.spines["left"].set_linewidth(0.6)
    axb.spines["bottom"].set_linewidth(0.6)
    axb.set_title("Corpus composition", fontsize=7.0, color="#3c3c3c",
                  pad=3.0, loc="left")

    # ---- subordinate panel 2: summary card on a grey block
    axc = fig.add_subplot(side[1, 0])
    axc.set_xticks([])
    axc.set_yticks([])
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    for s in axc.spines.values():
        s.set_visible(False)
    axc.add_patch(FancyBboxPatch(
        (0.02, 0.04), 0.96, 0.92,
        boxstyle="round,pad=0.012,rounding_size=0.06", facecolor="#f2f2f0",
        edgecolor="#dcdcd8", linewidth=0.7, transform=axc.transAxes,
        clip_on=False, zorder=0))
    axc.text(0.075, 0.885, "Dataset at a glance", fontsize=7.0,
             fontweight="bold", color="#3c3c3c", va="top", ha="left")
    rows = [("frames", "12,480"), ("cities", "8"), ("resolution", "2048x1024"),
            ("classes", "19"), ("pixel-level labels", "5,000"),
            ("mean objects / frame", "17.4")]
    for k, (lab, val) in enumerate(rows):
        y = 0.735 - 0.125 * k
        axc.text(0.075, y, lab, fontsize=6.0, color="#6e6e6e", va="center",
                 ha="left")
        axc.text(0.925, y, val, fontsize=6.2, color="#1a1a1a", va="center",
                 ha="right", fontweight="bold")

    savefig(fig, "261_cv_dataset_montage.png")


if __name__ == "__main__":
    main()
