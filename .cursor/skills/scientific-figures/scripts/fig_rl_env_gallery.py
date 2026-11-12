"""Eight benchmark environments drawn as schematics, with their interfaces.

Everything is primitives - no screenshots - so the tiles share the house palette
and one geometry frame (12 x 9 units, equal aspect).  The caption under each
tile carries the numbers a reader actually needs to compare tasks: observation
dimension, action space and episode horizon.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Arc, Circle, FancyBboxPatch, Polygon,
                                Rectangle, Wedge)

from pubstyle import set_pub_style, savefig
from figkit import panel_label_inches

W, H = 12.0, 9.0
INK = "#3a3a3a"
SOFT = "#e8edf2"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
GREY = "#8a8a8a"

ENVS = [
    ("Inverted pendulum", "obs 4-D  ·  act 1-D continuous  ·  horizon 500"),
    ("Mountain car", "obs 2-D  ·  act 3 discrete  ·  horizon 200"),
    ("Reacher arm (2-link)", "obs 11-D  ·  act 2-D continuous  ·  horizon 150"),
    ("Grid world", "obs 25-D one-hot  ·  act 4 discrete  ·  horizon 100"),
    ("Maze navigation", "obs 7-D lidar  ·  act 4 discrete  ·  horizon 400"),
    ("Hopper (continuous)", "obs 11-D  ·  act 3-D continuous  ·  horizon 1000"),
    ("Multi-agent field", "obs 4x18-D  ·  act 4x2-D cont.  ·  horizon 300"),
    ("Atari-style bricks", "obs 84x84x4 px  ·  act 4 discrete  ·  horizon 2700"),
]


def frame(ax):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(GREY)
    return ax


def ground(ax, y=1.4, x0=0.6, x1=W - 0.6):
    ax.plot([x0, x1], [y, y], color=INK, lw=1.2, zorder=2)
    for x in np.arange(x0, x1, 0.62):
        ax.plot([x, x - 0.32], [y, y - 0.36], color=GREY, lw=0.6, zorder=1)


def pendulum(ax):
    ground(ax)
    ax.add_patch(FancyBboxPatch((4.6, 1.5), 2.8, 1.05,
                                boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=BLUE, edgecolor="white", lw=0.8,
                                zorder=4))
    for cx in (5.35, 6.65):
        ax.add_patch(Circle((cx, 1.48), 0.30, facecolor="white",
                            edgecolor=INK, lw=0.8, zorder=5))
    th = np.deg2rad(70.0)
    tip = (6.0 + 4.1 * np.cos(th), 2.55 + 4.1 * np.sin(th))
    ax.plot([6.0, tip[0]], [2.55, tip[1]], color=INK, lw=2.4, zorder=5,
            solid_capstyle="round")
    ax.add_patch(Circle(tip, 0.45, facecolor=RED, edgecolor="white", lw=0.8,
                        zorder=6))
    ax.add_patch(Circle((6.0, 2.55), 0.20, facecolor="white", edgecolor=INK,
                        lw=0.8, zorder=6))
    ax.plot([6.0, 6.0], [2.55, 7.6], color=GREY, lw=0.7, ls=(0, (2.4, 1.8)),
            zorder=3)
    ax.add_patch(Arc((6.0, 2.55), 3.2, 3.2, theta1=70, theta2=90, color=GOLD,
                     lw=1.1, zorder=4))
    ax.text(6.35, 4.6, "$\\theta$", fontsize=7.0, color=GOLD, zorder=6)
    ax.annotate("", xy=(9.2, 2.0), xytext=(7.6, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))
    ax.text(8.4, 2.25, "$F$", fontsize=7.0, color=GREEN, ha="center")


def mountain_car(ax):
    x = np.linspace(0.6, W - 0.6, 260)
    y = 3.7 + 2.1 * np.sin(0.62 * (x - 1.4))
    ax.plot(x, y, color=INK, lw=1.4, zorder=3)
    ax.fill_between(x, 0.5, y, color=SOFT, lw=0, zorder=1)
    k = 118
    ang = np.arctan2(y[k + 1] - y[k - 1], x[k + 1] - x[k - 1])
    cx, cy = x[k], y[k] + 0.34
    car = FancyBboxPatch((cx - 0.62, cy - 0.22), 1.24, 0.52,
                         boxstyle="round,pad=0,rounding_size=0.14",
                         facecolor=BLUE, edgecolor="white", lw=0.8, zorder=5)
    car.set_transform(plt.matplotlib.transforms.Affine2D().rotate_around(
        cx, cy, ang) + ax.transData)
    ax.add_patch(car)
    gx = x[-14]
    ax.plot([gx, gx], [y[-14], y[-14] + 1.7], color=INK, lw=1.0, zorder=5)
    ax.add_patch(Polygon([[gx, y[-14] + 1.7], [gx - 1.1, y[-14] + 1.35],
                          [gx, y[-14] + 1.0]], facecolor=RED,
                         edgecolor="white", lw=0.6, zorder=5))
    ax.annotate("", xy=(x[k] + 1.5, y[k] + 1.2), xytext=(x[k] - 1.4,
                                                         y[k] - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(1.4, 7.7, "under-powered:\nmust rock back first", fontsize=6.2,
            color="#555555", va="top")


def reacher(ax):
    bx, by = 3.0, 2.2
    ax.add_patch(Arc((bx, by), 13.2, 13.2, theta1=10, theta2=88, color=GREY,
                     lw=0.7, ls=(0, (2.4, 1.8)), zorder=1))
    j1 = (bx + 3.4, by + 2.6)
    tip = (j1[0] + 2.6, j1[1] + 1.7)
    ax.add_patch(Rectangle((bx - 0.85, by - 0.75), 1.7, 0.75, facecolor=INK,
                           edgecolor="white", lw=0.7, zorder=4))
    for a, b, c in ((( bx, by), j1, BLUE), (j1, tip, "#5d8ac0")):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=5.0, zorder=4,
                solid_capstyle="round")
    for p in ((bx, by), j1):
        ax.add_patch(Circle(p, 0.34, facecolor="white", edgecolor=INK, lw=0.9,
                            zorder=6))
    ax.plot([tip[0]], [tip[1]], marker="o", ms=6.0, mfc=GOLD, mec="white",
            mew=0.9, zorder=6)
    ax.plot([9.4], [6.9], marker="*", ms=13, mfc=RED, mec="white", mew=0.8,
            zorder=6)
    ax.plot([tip[0], 9.4], [tip[1], 6.9], color=RED, lw=0.8,
            ls=(0, (2.2, 1.8)), zorder=3)
    ax.text(9.4, 6.15, "target", fontsize=6.4, color=RED, ha="center",
            va="top")
    ax.text(1.0, 8.4, "$q_1, q_2$ joint angles", fontsize=6.2,
            color="#555555", ha="left", va="top")


def grid_world(ax):
    nx, ny = 6, 5
    cw, ch = 9.0 / nx, 6.6 / ny
    x0, y0 = 1.5, 1.2
    walls = {(1, 1), (2, 3), (4, 2)}
    for i in range(nx):
        for j in range(ny):
            fc = INK if (i, j) in walls else "white"
            ax.add_patch(Rectangle((x0 + i * cw, y0 + j * ch), cw, ch,
                                   facecolor=fc, edgecolor="#c9c9c9", lw=0.6,
                                   zorder=2))
    ax.add_patch(Rectangle((x0 + 5 * cw, y0 + 4 * ch), cw, ch,
                           facecolor="#f1dcdf", edgecolor="#c9c9c9", lw=0.6,
                           zorder=3))
    ax.plot([x0 + 5.5 * cw], [y0 + 4.5 * ch], marker="*", ms=12, mfc=RED,
            mec="white", mew=0.8, zorder=5)
    ax.plot([x0 + 0.5 * cw], [y0 + 0.5 * ch], marker="o", ms=7, mfc=GREEN,
            mec="white", mew=0.9, zorder=5)
    ax_, ay_ = x0 + 2.5 * cw, y0 + 1.5 * ch
    ax.plot([ax_], [ay_], marker="o", ms=4.2, mfc="white", mec=BLUE, mew=0.9,
            zorder=5)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.annotate("", xy=(ax_ + dx * cw * 0.78, ay_ + dy * ch * 0.72),
                    xytext=(ax_ + dx * cw * 0.20, ay_ + dy * ch * 0.20),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9),
                    zorder=5)
    ax.text(x0, y0 + ny * ch + 0.35, "4 moves per cell", fontsize=6.2,
            color="#555555", va="bottom")


def bfs(wall, src, dst):
    """Shortest 4-connected path, so the drawn route cannot cross a wall."""
    ny, nx = wall.shape
    prev = {src: None}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        if cur == dst:
            break
        i, j = cur
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (i + di, j + dj)
            if (0 <= nb[0] < nx and 0 <= nb[1] < ny and nb not in prev
                    and not wall[nb[1], nb[0]]):
                prev[nb] = cur
                queue.append(nb)
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def maze(ax):
    nx, ny = 11, 8
    cw, ch = 10.0 / nx, 6.8 / ny
    x0, y0 = 1.0, 1.1
    wall = np.zeros((ny, nx), bool)
    wall[1:6, 2] = True
    wall[3:8, 5] = True
    wall[0:4, 8] = True
    wall[5, 6:10] = True
    wall[2, 3:5] = True
    for j in range(ny):
        for i in range(nx):
            if wall[j, i]:
                ax.add_patch(Rectangle((x0 + i * cw, y0 + j * ch), cw, ch,
                                       facecolor=INK, edgecolor="white",
                                       lw=0.5, zorder=3))
    ax.add_patch(Rectangle((x0, y0), nx * cw, ny * ch, facecolor="none",
                           edgecolor=GREY, lw=0.8, zorder=4))
    path = bfs(wall, (0, 0), (nx - 1, ny - 1))
    px = [x0 + (p[0] + 0.5) * cw for p in path]
    py = [y0 + (p[1] + 0.5) * ch for p in path]
    ax.plot(px, py, color=BLUE, lw=1.4, ls=(0, (3.0, 1.6)), zorder=5,
            solid_capstyle="round")
    ax.plot([px[0]], [py[0]], marker="o", ms=6.5, mfc=GREEN, mec="white",
            mew=0.9, zorder=6)
    ax.plot([px[-1]], [py[-1]], marker="*", ms=12, mfc=RED, mec="white",
            mew=0.8, zorder=6)
    mid = len(px) // 2
    for a in np.linspace(0, 2 * np.pi, 7, endpoint=False):
        ax.plot([px[mid], px[mid] + 0.62 * np.cos(a)],
                [py[mid], py[mid] + 0.62 * np.sin(a)], color=GOLD, lw=0.6,
                zorder=6)
    ax.text(x0, y0 + ny * ch + 0.30, "7-beam lidar observation", fontsize=6.2,
            color="#555555", va="bottom")


def hopper(ax):
    ground(ax)
    hip = np.array([5.4, 5.6])
    knee = np.array([6.3, 3.9])
    foot = np.array([5.6, 2.0])
    ax.add_patch(FancyBboxPatch((hip[0] - 0.55, hip[1]), 1.1, 2.1,
                                boxstyle="round,pad=0,rounding_size=0.24",
                                facecolor=BLUE, edgecolor="white", lw=0.8,
                                zorder=4))
    for a, b, c in ((hip, knee, "#4c74ab"), (knee, foot, "#5d8ac0")):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=4.4, zorder=4,
                solid_capstyle="round")
    ax.plot([foot[0] - 0.75, foot[0] + 0.95], [foot[1] - 0.55, foot[1] - 0.55],
            color=INK, lw=3.6, zorder=4, solid_capstyle="round")
    for p in (hip, knee, foot):
        ax.add_patch(Circle(p, 0.28, facecolor="white", edgecolor=INK, lw=0.9,
                            zorder=6))
        ax.add_patch(Circle(p, 0.60, facecolor="none", edgecolor=GOLD, lw=0.7,
                            ls=(0, (1.8, 1.6)), zorder=5))
    ax.annotate("", xy=(9.4, 4.6), xytext=(7.6, 4.6),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))
    ax.text(8.5, 4.9, "forward reward", fontsize=6.2, color=GREEN,
            ha="center")
    ax.text(1.1, 7.9, "3 torque-controlled joints", fontsize=6.2,
            color="#555555", va="top")


def multiagent(ax):
    ax.add_patch(Rectangle((0.8, 1.0), W - 1.6, H - 2.4, facecolor="#fafbfc",
                           edgecolor=GREY, lw=0.7, zorder=1))
    pos = np.array([[2.6, 2.6], [4.4, 5.9], [8.2, 3.0], [9.4, 6.2]])
    vel = np.array([[0.9, 0.7], [1.0, -0.5], [-0.8, 0.9], [-1.0, -0.6]])
    cols = [RED, "#c0603f", BLUE, "#3d6b8e"]
    for i in range(4):
        for j in range(i + 1, 4):
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color="#cfd6dd", lw=0.6, ls=(0, (1.8, 1.8)), zorder=2)
    for p, v, c, n in zip(pos, vel, cols, ("A1", "A2", "B1", "B2")):
        ax.add_patch(Circle(p, 0.42, facecolor=c, edgecolor="white", lw=0.8,
                            zorder=5))
        ax.annotate("", xy=p + v * 1.15, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0))
        ax.text(p[0], p[1] - 0.85, n, fontsize=6.0, color=c, ha="center",
                va="top", zorder=6)
    for gx, gy, c in ((6.2, 7.4, RED), (6.2, 1.9, BLUE)):
        ax.plot([gx], [gy], marker="*", ms=12, mfc=c, mec="white", mew=0.8,
                zorder=5)
    ax.text(6.2, 8.2, "shared arena, team-specific goals", fontsize=6.2,
            color="#555555", ha="center", va="center")


def atari(ax):
    ax.add_patch(Rectangle((1.2, 0.9), W - 2.4, H - 1.9, facecolor="#f2f4f6",
                           edgecolor=GREY, lw=0.7, zorder=1))
    cols = [RED, GOLD, GREEN, BLUE]
    bw, bh = 1.14, 0.46
    for r in range(4):
        for c in range(7):
            ax.add_patch(Rectangle((1.6 + c * (bw + 0.10),
                                    6.6 - r * (bh + 0.16)), bw, bh,
                                   facecolor=cols[r], edgecolor="white",
                                   lw=0.6, alpha=0.92, zorder=3))
    ax.add_patch(Rectangle((5.0, 1.5), 2.4, 0.34, facecolor=INK,
                           edgecolor="white", lw=0.6, zorder=4))
    ax.add_patch(Circle((6.9, 2.8), 0.26, facecolor=INK, edgecolor="white",
                        lw=0.6, zorder=4))
    ax.plot([6.9, 8.6, 5.3], [2.8, 4.6, 6.0], color=GREY, lw=0.7,
            ls=(0, (2.0, 1.8)), zorder=3)
    ax.text(1.6, 8.05, "frame stack $k=4$", fontsize=6.2, color="#555555",
            va="bottom")


DRAW = [pendulum, mountain_car, reacher, grid_world, maze, hopper,
        multiagent, atari]


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=(10.0, 5.6))
    gs = fig.add_gridspec(2, 4, left=0.030, right=0.975, top=0.875,
                          bottom=0.155, hspace=0.62, wspace=0.10)
    axes = []
    for k, (draw, (name, spec)) in enumerate(zip(DRAW, ENVS)):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        axes.append(frame(ax))
        draw(ax)
        ax.set_title(name, fontsize=8.4, fontweight="bold", pad=3.5)
        ax.text(0.5, -0.045, spec.replace("  ·  ", "\n"),
                transform=ax.transAxes, ha="center", va="top", fontsize=6.4,
                color="#555555", linespacing=1.45)
        panel_label_inches(fig, ax, f"({'abcdefgh'[k]})", dx=-0.16, dy=0.06)

    savefig(fig, "442_rl_env_gallery.png")


if __name__ == "__main__":
    main()
