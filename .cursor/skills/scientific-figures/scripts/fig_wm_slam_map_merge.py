"""Multi-robot map merging: three partial maps fused into one global map.

Three robots sweep overlapping strips of the same building.  Each partial map
is rendered on the *same* world extent, so the unobserved remainder stays grey
and the three panels are directly comparable.  The merged map on the right is
the union of the three observation masks; the two overlap bands are shaded and
labelled with the intersection-over-union of the neighbouring masks, computed
from the grids that are actually drawn.

Single-hue blue batch (BRIEF_WM_BLUE.md section 2): robot identity is carried
by depth on the one ladder, never by a second hue.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
CYAN_ARROW = "#1c86b8"
UNSEEN = "#e4e7ea"

W, H = 30.0, 18.0
RES = 0.15
T = 0.42
ROBOTS = ["Robot A", "Robot B", "Robot C"]
WINDOWS = [(0.0, 12.5), (9.5, 21.5), (18.0, 30.0)]
DEPTH = [BLUE[1], BLUE[2], BLUE[3]]


def wall_rects():
    R = []

    def hwall(y, x0, x1, gaps=()):
        cuts = [x0]
        for g0, g1 in gaps:
            cuts += [g0, g1]
        cuts.append(x1)
        for a, b in zip(cuts[0::2], cuts[1::2]):
            if b - a > 1e-6:
                R.append((a, y - T / 2, b - a, T))

    def vwall(x, y0, y1, gaps=()):
        cuts = [y0]
        for g0, g1 in gaps:
            cuts += [g0, g1]
        cuts.append(y1)
        for a, b in zip(cuts[0::2], cuts[1::2]):
            if b - a > 1e-6:
                R.append((x - T / 2, a, T, b - a))

    hwall(T / 2, 0.0, W)
    hwall(H - T / 2, 0.0, W)
    vwall(T / 2, 0.0, H)
    vwall(W - T / 2, 0.0, H)
    hwall(7.4, T, W - T, gaps=[(4.2, 5.6), (13.6, 15.0), (24.0, 25.4)])
    hwall(11.0, T, W - T, gaps=[(2.6, 4.0), (11.0, 12.4), (21.0, 22.4)])
    for x in (6.4, 12.6, 19.4, 25.2):
        vwall(x, T, 7.4)
    for x in (4.6, 10.2, 16.8, 23.0, 27.4):
        vwall(x, 11.0, H - T)
    vwall(15.4, 7.4, 11.0)
    return R


RECTS = wall_rects()


def occupancy():
    nx, ny = int(round(W / RES)), int(round(H / RES))
    xs = (np.arange(nx) + 0.5) * RES
    ys = (np.arange(ny) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    m = np.zeros_like(X, bool)
    for x0, y0, w, h in RECTS:
        m |= (X >= x0) & (X <= x0 + w) & (Y >= y0) & (Y <= y0 + h)
    return m, X


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(63)

    walls, X = occupancy()
    ny, nx = walls.shape
    masks = []
    for x0, x1 in WINDOWS:
        m = (X >= x0) & (X < x1)
        # ragged sensing frontier rather than a ruler-straight cut
        jag = (np.sin(np.linspace(0, 7.0, ny)) * 0.55 +
               rng.normal(0, 0.18, ny))
        for r in range(ny):
            lo = int(round((x0 + (jag[r] if x0 > 0 else 0.0)) / RES))
            hi = int(round((x1 + (jag[r] if x1 < W else 0.0)) / RES))
            m[r, :] = False
            m[r, max(lo, 0):min(hi, nx)] = True
        masks.append(m)

    merged = masks[0] | masks[1] | masks[2]

    def iou(a, b):
        return float((a & b).sum()) / float((a | b).sum())

    iou_ab, iou_bc = iou(masks[0], masks[1]), iou(masks[1], masks[2])
    cover = [100.0 * m.mean() for m in masks]
    cover_all = 100.0 * merged.mean()

    ext = [0, W, 0, H]

    def draw(ax, mask, wall_col, free_col):
        img = np.zeros(walls.shape + (3,))
        img[:] = mpl.colors.to_rgb(UNSEEN)
        img[mask & ~walls] = mpl.colors.to_rgb(free_col)
        img[mask & walls] = mpl.colors.to_rgb(wall_col)
        ax.imshow(img, extent=ext, origin="lower", interpolation="nearest",
                  zorder=1)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.9)
            sp.set_color(INK)

    fig = plt.figure(figsize=(10.4, 5.4))
    bottom, top = 0.075, 0.845
    gh = top - bottom
    gw = gh * (5.4 / 10.4) * (W / H)
    gx = 0.964 - gw
    ph = (gh - 2 * 0.0241) / 3.0
    pw = ph * (5.4 / 10.4) * (W / H)
    px = 0.024

    axG = fig.add_axes([gx, bottom, gw, gh])
    subs = []
    for i in range(3):
        y = bottom + (2 - i) * (ph + 0.0241)
        axp = fig.add_axes([px, y, pw, ph])
        subs.append(axp)
        draw(axp, masks[i], DEPTH[i], BLUE[7])
        axp.text(0.022, 0.955, "%s   %.0f%% of the building"
                 % (ROBOTS[i], cover[i]), transform=axp.transAxes,
                 fontsize=6.6, fontweight="bold", color=DEPTH[i], va="top",
                 zorder=6,
                 bbox=dict(boxstyle="round,pad=0.24", fc="white",
                           ec="#c3ccd4", lw=0.5, alpha=0.95))
        add_panel_label(axp, "(%s)" % "abc"[i], x=-0.082, y=0.84, size=9.5)

    # ---- merged map -------------------------------------------------------
    draw(axG, merged, BLUE[0], BLUE[7])
    xs = (np.arange(nx) + 0.5) * RES
    ys = (np.arange(ny) + 0.5) * RES
    bands = [(masks[0] & masks[1], "A $\\cap$ B", iou_ab, 0.955),
             (masks[1] & masks[2], "B $\\cap$ C", iou_bc, 0.075)]
    for inter, name, val, ty in bands:
        rgba = np.zeros(walls.shape + (4,))
        rgba[..., :3] = mpl.colors.to_rgb(BLUE[4])
        rgba[..., 3] = np.where(inter, 0.40, 0.0)
        axG.imshow(rgba, extent=ext, origin="lower",
                   interpolation="nearest", zorder=4)
        axG.contour(xs, ys, inter.astype(float), levels=[0.5],
                    colors=[BLUE[2]], linewidths=0.9, zorder=5)
        cx = float(xs[inter.any(axis=0)].mean())
        axG.text(cx, ty * H, "%s\noverlap IoU %.3f" % (name, val),
                 ha="center",
                 va="top" if ty > 0.5 else "bottom", fontsize=6.8,
                 fontweight="bold", color=BLUE[1], linespacing=1.3, zorder=7,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white",
                           ec="#c3ccd4", lw=0.6, alpha=0.96))
    axG.set_title("Merged global map   -   %.1f%% of the building covered"
                  % cover_all, fontsize=8.8, fontweight="bold", color=INK,
                  pad=3.5)
    add_panel_label(axG, "(d)", x=-0.028, y=1.020, size=10)

    # ---- feeder arrows ----------------------------------------------------
    for i, axp in enumerate(subs):
        p0 = axp.get_position()
        p1 = axG.get_position()
        y0 = p0.y0 + p0.height / 2.0
        y1 = p1.y0 + p1.height * (0.78 - 0.28 * i)
        arr = FancyArrowPatch((p0.x1 + 0.006, y0), (p1.x0 - 0.007, y1),
                              transform=fig.transFigure,
                              connectionstyle="arc3,rad=%.2f"
                                              % (0.10 * (1 - i)),
                              arrowstyle="-|>", mutation_scale=9,
                              color=CYAN_ARROW, lw=1.4, zorder=5)
        fig.add_artist(arr)
    mx = (subs[1].get_position().x1 + axG.get_position().x0) / 2.0
    fig.text(mx, bottom + gh * 0.045, "align\n+ fuse", ha="center",
             va="center", fontsize=7.0, fontweight="bold", color=CYAN_ARROW,
             linespacing=1.3,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c3ccd4",
                       lw=0.6, alpha=0.96))

    handles = [
        Line2D([], [], marker="s", ls="none", ms=6.0, mfc=BLUE[0],
               mec="none", label="wall (merged)"),
        Line2D([], [], marker="s", ls="none", ms=6.0, mfc=BLUE[7],
               mec="#c3ccd4", label="observed free"),
        Line2D([], [], marker="s", ls="none", ms=6.0, mfc=UNSEEN,
               mec="#c3ccd4", label="not observed by this robot"),
        Line2D([], [], marker="s", ls="none", ms=6.0, mfc=BLUE[5],
               mec=BLUE[2], label="overlap band"),
    ]
    savefig(fig, "1140_wm_slam_map_merge.png")


if __name__ == "__main__":
    main()
