"""Wm blue occupancy density: SLAM occupancy grid sharpening over three stages.

Reads like the classic robotics occupancy triptych -- unknown space in a muted
blue, swept free space near white, occupied cells in deep navy -- but with the
reference figure's electric cyan pulled back to the house ladder.  Each panel
gets a black leader line to the bold "Robot position" call-out and a corner
letter.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pubstyle import set_pub_style, savefig

# WM blue batch ladder (dark -> near white); see BRIEF_WM_BLUE.md section 2.
BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
UNKNOWN, FREE, OCCUPIED = BLUE[4], BLUE[7], BLUE[0]
GRID_CMAP = ListedColormap([UNKNOWN, FREE, OCCUPIED])
GRID_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

H, W = 230, 310
STAGES = ["t = 18 s,  620 scans", "t = 95 s,  3.4k scans",
          "t = 240 s,  9.1k scans"]
ROBOT_TAGS = ["A", "B", "C"]

# Corridor skeleton the robot drives along, in grid coordinates.
CORRIDORS = [
    ((20, 20), (290, 20)), ((290, 20), (290, 210)),
    ((290, 210), (20, 210)), ((20, 210), (20, 20)),
    ((20, 115), (290, 115)), ((115, 20), (115, 210)),
    ((200, 115), (200, 210)), ((60, 60), (60, 115)),
    ((60, 60), (100, 60)), ((150, 20), (150, 115)),
    ((150, 60), (240, 60)), ((240, 60), (240, 115)),
    ((230, 150), (280, 150)), ((230, 150), (230, 195)),
    ((45, 150), (100, 150)), ((45, 150), (45, 195)),
]
ROBOT_XY = [(115, 20), (150, 22), (114, 60)]


def _distance_field():
    """Euclidean distance from every cell to the corridor skeleton."""
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.full((H, W), 1e6)
    for (x0, y0), (x1, y1) in CORRIDORS:
        vx, vy = x1 - x0, y1 - y0
        seg = max(vx * vx + vy * vy, 1e-9)
        t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / seg, 0.0, 1.0)
        dist = np.minimum(dist, np.hypot(xx - (x0 + t * vx),
                                         yy - (y0 + t * vy)))
    return dist


def _smooth_noise(rng, amp, scale=14):
    """Low-frequency noise field, used to smudge the early-stage walls."""
    small = rng.normal(0, 1, (H // scale + 2, W // scale + 2))
    ys = np.linspace(0, small.shape[0] - 1, H)
    xs = np.linspace(0, small.shape[1] - 1, W)
    yi = np.clip(np.floor(ys).astype(int), 0, small.shape[0] - 2)
    xi = np.clip(np.floor(xs).astype(int), 0, small.shape[1] - 2)
    fy, fx = (ys - yi)[:, None], (xs - xi)[None, :]
    top = small[yi][:, xi] * (1 - fx) + small[yi][:, xi + 1] * fx
    bot = small[yi + 1][:, xi] * (1 - fx) + small[yi + 1][:, xi + 1] * fx
    return amp * (top * (1 - fy) + bot * fy)


def gen_data(seed=1514):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import numpy as np
        return np.load("data/occupancy_stages.npy")     # (stage, H, W) in 0/1/2
    """
    rng = np.random.default_rng(seed)
    dist = _distance_field()
    grids = []
    for smudge, dropout, clutter in ((3.4, 0.42, 0.055),
                                     (1.3, 0.13, 0.014),
                                     (0.35, 0.02, 0.002)):
        d = dist + _smooth_noise(rng, smudge) + rng.normal(0, 0.35 * smudge,
                                                           (H, W))
        grid = np.zeros((H, W), dtype=int)
        free = d < 7.0
        wall = (d >= 7.0) & (d < 9.6)
        grid[free] = 1
        grid[wall] = 2
        # Early sweeps leave holes in the free space and speckle in the rooms.
        holes = rng.random((H, W)) < dropout
        grid[free & holes] = 0
        speck = (rng.random((H, W)) < clutter) & free
        grid[speck] = 2
        grids.append(grid)
    return grids


def main():
    set_pub_style(base_font=8)
    grids = gen_data()

    fig = plt.figure(figsize=(10.4, 3.6))
    axes = [fig.add_axes([0.020 + i * 0.327, 0.045, 0.300, 0.795])
            for i in range(3)]

    for k, (ax, grid) in enumerate(zip(axes, grids)):
        ax.imshow(grid, cmap=GRID_CMAP, norm=GRID_NORM,
                  interpolation="nearest", aspect="auto", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.3)
            s.set_edgecolor(INK)

        rx, ry = ROBOT_XY[k]
        ax.scatter([rx], [ry], s=52, marker="o", color=BLUE[0],
                   edgecolors="black", linewidths=0.7, zorder=5)
        ax.annotate(f"Robot position ({ROBOT_TAGS[k]})", xy=(rx + 4, ry + 5),
                    xytext=(rx + 46, ry + 38), fontsize=7.6,
                    fontweight="bold", color="black", ha="left", va="center",
                    zorder=6,
                    bbox=dict(boxstyle="square,pad=0.22", facecolor="white",
                              edgecolor="none", alpha=0.80),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9,
                                    shrinkA=3, shrinkB=2))

        ax.set_title(STAGES[k], fontsize=8.5, fontweight="bold", color=INK,
                     pad=4)
        ax.text(0.975, 0.045, f"({'abc'[k]})", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10.5, fontweight="bold",
                color=INK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.85))


    savefig(fig, "1109_wm_blue_occupancy_density.png")


if __name__ == "__main__":
    main()
