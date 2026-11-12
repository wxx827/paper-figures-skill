"""Occupancy-grid mapping: ground truth, lidar rays, log-odds map, cost map.

A 360-beam range finder with 8 m maximum range and 3 cm Gaussian noise is
simulated by ray marching against the analytic scene.  Six poses along the
corridor are fused with the standard inverse-sensor-model log-odds update,
so the recovered map genuinely contains unobserved (grey) pockets in the
shadow of every obstacle.  Panel (d) turns the thresholded belief into a
navigation cost map by Euclidean distance transform and exponential decay.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt
from pubstyle import set_pub_style, savefig, add_panel_label, add_colorbar

W, H = 20.0, 14.0
RES = 0.10
RECTS = [(3.0, 0.0, 1.4, 5.2), (3.0, 7.6, 1.4, 6.4),
         (7.2, 3.2, 1.4, 10.8), (11.0, 0.0, 1.4, 6.0),
         (15.0, 5.0, 4.2, 1.3), (16.6, 8.6, 1.2, 2.6)]
CIRCLES = [(5.6, 11.2, 0.95), (13.4, 10.2, 1.25), (9.9, 1.3, 0.8)]
START = np.array([1.3, 1.5])
GOAL = np.array([18.7, 12.6])

POSES = np.array([[1.6, 2.2], [2.1, 6.4], [5.9, 7.4], [9.6, 2.6],
                  [13.0, 6.6], [17.0, 12.0]])
MAX_RANGE = 8.0
N_BEAMS = 1080
SIGMA_R = 0.03
L_FREE, L_OCC, L_CLIP = -1.40, 0.90, 7.0
ROB_R = 0.35

OBS_C = "#3a3a3a"
S_C = "#1f8a4c"
G_C = "#b23a48"
ROB_C = "#0072b2"


def occupied(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.zeros(np.broadcast(x, y).shape, bool)
    for x0, y0, w, h in RECTS:
        m |= (x >= x0) & (x <= x0 + w) & (y >= y0) & (y <= y0 + h)
    for cx, cy, r in CIRCLES:
        m |= (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    return m | (x < 0) | (x > W) | (y < 0) | (y > H)


def scan(pose, rng, n_beams=N_BEAMS, step=RES / 2.0):
    """Ray-march a full 360 deg scan; return angles, ranges, hit flags."""
    ang = np.linspace(0.0, 2 * np.pi, n_beams, endpoint=False)
    t = np.arange(step, MAX_RANGE + step, step)
    X = pose[0] + np.cos(ang)[:, None] * t[None, :]
    Y = pose[1] + np.sin(ang)[:, None] * t[None, :]
    hit = occupied(X, Y)
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), len(t) - 1)
    rng_m = t[first] + np.where(any_hit, rng.normal(0, SIGMA_R, n_beams), 0.0)
    return ang, np.clip(rng_m, step, MAX_RANGE), any_hit


def log_odds_map(poses, seed=1):
    rng = np.random.default_rng(seed)
    nx, ny = int(round(W / RES)), int(round(H / RES))
    L = np.zeros((ny, nx))
    step = RES / 2.0
    for p in poses:
        ang, r, hit = scan(p, rng, step=step)
        free_i, free_j = [], []
        for a, rr in zip(ang, r):
            t = np.arange(step, max(rr - RES, step), step)
            if t.size == 0:
                continue
            free_i.append(((p[1] + np.sin(a) * t) / RES).astype(int))
            free_j.append(((p[0] + np.cos(a) * t) / RES).astype(int))
        fi = np.clip(np.concatenate(free_i), 0, ny - 1)
        fj = np.clip(np.concatenate(free_j), 0, nx - 1)
        flat = np.unique(fi * nx + fj)          # one free update per scan
        L.ravel()[flat] += L_FREE
        hi = np.clip(((p[1] + np.sin(ang[hit]) * r[hit]) / RES).astype(int),
                     0, ny - 1)
        hj = np.clip(((p[0] + np.cos(ang[hit]) * r[hit]) / RES).astype(int),
                     0, nx - 1)
        np.add.at(L, (hi, hj), L_OCC)
        np.clip(L, -L_CLIP, L_CLIP, out=L)
    return L


def frame(ax, row, col):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.set_xticks(np.arange(0, W + 1, 5))
    ax.set_yticks(np.arange(0, H + 1, 5))
    ax.tick_params(length=2.4, labelsize=6.6)
    if row == 1:
        ax.set_xlabel("x (m)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("y (m)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_yticklabels([])


def main():
    set_pub_style(base_font=8)
    nx, ny = int(round(W / RES)), int(round(H / RES))
    xs, ys = (np.arange(nx) + 0.5) * RES, (np.arange(ny) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    truth = occupied(X, Y)
    ext = [0, W, 0, H]

    fig, axes = plt.subplots(2, 2, figsize=(9.9, 6.85))
    fig.subplots_adjust(left=0.055, right=0.955, top=0.855, bottom=0.075,
                        wspace=0.10, hspace=0.13)

    # (a) ground truth ------------------------------------------------------
    ax = axes[0, 0]
    ax.imshow(truth, extent=ext, origin="lower", interpolation="nearest",
              cmap=mpl.colors.ListedColormap(["white", OBS_C]), vmin=0, vmax=1)
    for p in POSES:
        ax.add_patch(plt.Circle(p, MAX_RANGE, fc="none", ec=ROB_C, lw=0.4,
                                alpha=0.55, ls=(0, (2.5, 2.5)), zorder=2))
    ax.plot(POSES[:, 0], POSES[:, 1], ls="none", marker="o", ms=4.2,
            mfc=ROB_C, mec="white", mew=0.7, zorder=6)
    for k, p in enumerate(POSES):
        ax.text(p[0] + 0.35, p[1] + 0.35, "%d" % (k + 1), fontsize=6.2,
                color=ROB_C, fontweight="bold", zorder=7)
    ax.plot(*START, marker="^", ms=8, mfc=S_C, mec="white", mew=0.9,
            ls="none", zorder=8)
    ax.plot(*GOAL, marker="*", ms=13, mfc=G_C, mec="white", mew=0.9,
            ls="none", zorder=8)
    frame(ax, 0, 0)
    ax.set_title("Ground-truth workspace and sensing poses", fontsize=8.6,
                 fontweight="bold", pad=3)

    # (b) one lidar scan ----------------------------------------------------
    ax = axes[0, 1]
    ax.imshow(truth, extent=ext, origin="lower", interpolation="nearest",
              cmap=mpl.colors.ListedColormap(["#f4f4f4", OBS_C]),
              vmin=0, vmax=1)
    pose = POSES[2]
    rng = np.random.default_rng(11)
    ang, r, hit = scan(pose, rng)
    sub = slice(None, None, 12)
    ex, ey = pose[0] + np.cos(ang) * r, pose[1] + np.sin(ang) * r
    for a, rr, hh, xe, ye in zip(ang[sub], r[sub], hit[sub], ex[sub], ey[sub]):
        ax.plot([pose[0], xe], [pose[1], ye], color="#4a9fe0",
                lw=0.42, alpha=0.8 if hh else 0.35,
                ls="-" if hh else (0, (1.5, 1.5)), zorder=3)
    ax.plot(ex[hit], ey[hit], ls="none", marker="o", ms=1.5, mfc=G_C,
            mec="none", zorder=6)
    ax.add_patch(plt.Circle(pose, MAX_RANGE, fc="none", ec=ROB_C, lw=0.7,
                            ls=(0, (3.0, 2.0)), zorder=4))
    ax.plot(*pose, marker="o", ms=6.5, mfc=ROB_C, mec="white", mew=0.9,
            ls="none", zorder=8)
    ax.annotate("", xy=(pose[0] + 1.2, pose[1]), xytext=tuple(pose),
                arrowprops=dict(arrowstyle="-|>", color=ROB_C, lw=1.1,
                                mutation_scale=7), zorder=8)
    frame(ax, 0, 1)
    ax.set_title("Simulated 1080-beam range scan at pose 3", fontsize=8.6,
                 fontweight="bold", pad=3)
    ax.text(0.982, 0.035, "1080 beams, 0.33$^\\circ$ spacing "
            "(every 12th drawn)\nmax range 8.0 m, $\\sigma_r$ = 3 cm\n"
            "mean return %.2f m, longest %.2f m" % (r.mean(), r.max()),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.4, linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (c) log-odds occupancy map -------------------------------------------
    ax = axes[1, 0]
    L = log_odds_map(POSES)
    p_occ = 1.0 - 1.0 / (1.0 + np.exp(L))
    im = ax.imshow(p_occ, extent=ext, origin="lower", interpolation="nearest",
                   cmap="gray_r", vmin=0, vmax=1)
    frame(ax, 1, 0)
    ax.set_title("Fused log-odds occupancy belief (6 scans)", fontsize=8.6,
                 fontweight="bold", pad=3)
    add_colorbar(fig, im, ax,
                 label="P(occupied):  0 free / 0.5 unknown / 1 occupied",
                 fraction=0.036, pad=0.02, aspect=16,
                 ticks=[0, 0.25, 0.5, 0.75, 1])
    unknown = np.mean(np.abs(L) < 0.15)
    ax.text(0.982, 0.035, "$l_{\\mathrm{free}}$ = %.2f, "
            "$l_{\\mathrm{occ}}$ = %+.2f per scan\n"
            "%.0f%% of the arena still unobserved"
            % (L_FREE, L_OCC, 100 * unknown),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    # (d) inflated cost map -------------------------------------------------
    ax = axes[1, 1]
    lethal = p_occ > 0.65
    unknown = np.abs(L) < 0.35
    d = distance_transform_edt(~lethal) * RES
    cost = np.exp(-2.60 * np.clip(d - ROB_R, 0, None))
    cost[d <= ROB_R] = 1.0
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "costmap", ["#f7fbff", "#d9e8f5", "#9ec9e2", "#f6c391", "#c0603f",
                    "#8c2f2f"])
    im = ax.imshow(cost, extent=ext, origin="lower", interpolation="nearest",
                   cmap=cmap, vmin=0, vmax=1)
    ax.contour(X, Y, d, levels=[ROB_R], colors="#7f2704", linewidths=0.6)
    ax.imshow(np.where(unknown & ~lethal, 1.0, np.nan), extent=ext,
              origin="lower", interpolation="nearest",
              cmap=mpl.colors.ListedColormap(["#b9b9b9"]), vmin=0, vmax=1)
    ax.imshow(np.where(lethal, 1.0, np.nan), extent=ext, origin="lower",
              interpolation="nearest",
              cmap=mpl.colors.ListedColormap([OBS_C]), vmin=0, vmax=1)
    frame(ax, 1, 1)
    ax.set_title("Inflated navigation cost map", fontsize=8.6,
                 fontweight="bold", pad=3)
    cb = add_colorbar(fig, im, ax, label="traversal cost (-)", fraction=0.036,
                      pad=0.02, aspect=16, ticks=[0, 0.5, 1])
    ax.text(0.982, 0.035, "lethal cells: P > 0.65\ninscribed radius "
            "%.2f m (dark outline)\ncost = $e^{-2.6\\,(d - r)}$, "
            "grey = unobserved" % ROB_R,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.4, linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#bbbbbb",
                      lw=0.6, alpha=0.94))

    for k, ax in enumerate(axes.ravel()):
        add_panel_label(ax, "(%s)" % "abcd"[k], x=-0.05, y=1.005, size=10)

    handles = [Line2D([], [], marker="o", ls="none", ms=5.5, mfc=ROB_C,
                      mec="white", label="sensing pose"),
               Line2D([], [], color="#5aa9e6", lw=0.9, label="lidar beam"),
               Line2D([], [], marker="o", ls="none", ms=3.6, mfc=G_C,
                      mec="none", label="range return"),
               Line2D([], [], marker="^", ls="none", ms=6.5, mfc=S_C,
                      mec="white", label="start S"),
               Line2D([], [], marker="*", ls="none", ms=10, mfc=G_C,
                      mec="white", label="goal G"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc=OBS_C,
                      mec="none", label="true obstacle")]
    savefig(fig, "166_occupancy_grid.png")


if __name__ == "__main__":
    main()
