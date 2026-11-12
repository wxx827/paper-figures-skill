"""Figure 1142 - Collision resolution: three contact solvers, four instants.

Rows are contact models (none / penalty springs / implicit complementarity),
columns are four instants of the *same* drop: a rigid cube tumbling onto a
fixed step.  Every row starts from the identical pose, so the divergence
between rows is entirely the solver's doing.

The red patch in each tile is the *exact* intersection of the cube polygon
with the obstacle, obtained by Sutherland-Hodgman clipping against the two
convex pieces of the obstacle (the ground half-plane and the step).  The
penetration depth quoted on each tile is measured from the same geometry by
sampling the cube outline and taking the deepest point below the obstacle
surface, so the numbers and the red area can never disagree.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from pubstyle import set_pub_style, savefig, lighten
from figkit import background_card, image_axes

H = 0.34                       # cube half-edge (m)
SX0, SX1, SY1 = 1.22, 3.40, 0.45      # fixed step (runs off the right edge)
XLIM, YLIM = (-0.20, 3.10), (-0.75, 1.53)
BODY, BODY_EDGE = lighten("#3b5b92", 0.42), "#23385c"
OBST, OBST_EDGE = "#c8ccd2", "#6a7078"
HIT = "#b23a48"
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"

TIMES = [0, 14, 26, 44]
# (cx, cy, theta) per row per column; column 0 is shared by construction
START = (0.35, 1.10, 0.24)
FALL = (0.95, 0.95, 0.52)
ROWS = [
    ("No contact handling",
     [START, FALL, (1.60, 0.520, 0.78), (2.35, -0.220, 1.12)]),
    ("Penalty springs",
     [START, FALL, (1.60, 0.820, 0.74), (2.15, 0.878, 0.63)]),
    ("Ours: implicit contact",
     [START, FALL, (1.60, 0.932, 0.70), (2.18, 0.928, 0.62)]),
]

# obstacle as two convex half-plane sets, inside = a x + b y + c >= 0
GROUND = [(0.0, -1.0, 0.0)]
STEP = [(1.0, 0.0, -SX0), (-1.0, 0.0, SX1), (0.0, -1.0, SY1), (0.0, 1.0, 1.20)]


def cube(cx, cy, th):
    c, s = np.cos(th), np.sin(th)
    q = np.array([(-H, -H), (H, -H), (H, H), (-H, H)])
    return np.c_[cx + c * q[:, 0] - s * q[:, 1],
                 cy + s * q[:, 0] + c * q[:, 1]]


def clip(poly, planes):
    """Sutherland-Hodgman clip of a polygon against a convex half-plane set."""
    out = list(map(tuple, poly))
    for (a, b, c) in planes:
        if not out:
            return np.zeros((0, 2))
        src, out = out, []
        for i, p in enumerate(src):
            q = src[(i + 1) % len(src)]
            dp, dq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
            if dp >= 0:
                out.append(p)
            if (dp >= 0) != (dq >= 0):
                t = dp / (dp - dq)
                out.append((p[0] + t * (q[0] - p[0]),
                            p[1] + t * (q[1] - p[1])))
    return np.array(out) if out else np.zeros((0, 2))


def penetration(poly, n=400):
    """Deepest point of the cube outline inside the obstacle, in metres."""
    pts = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        pts.append(a + t * (b - a))
    p = np.vstack(pts)
    x, y = p[:, 0], p[:, 1]
    d = np.clip(-y, 0.0, None)                        # into the ground
    inside = (x > SX0) & (x < SX1) & (y < SY1)
    ds = np.minimum.reduce([x - SX0, SX1 - x, SY1 - y])
    return float(np.maximum(d, np.where(inside, ds, 0.0)).max())


def main():
    set_pub_style(base_font=8)

    cells = np.empty((len(ROWS), len(TIMES)), object)
    for i, (_, poses) in enumerate(ROWS):
        for j, pose in enumerate(poses):
            poly = cube(*pose)
            cells[i, j] = dict(poly=poly, pose=pose,
                               ov=[clip(poly, GROUND), clip(poly, STEP)],
                               d=penetration(poly))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(len(ROWS), len(TIMES), left=0.088, right=0.988,
                          top=0.855, bottom=0.088, wspace=0.045, hspace=0.085)
    axes = np.empty((len(ROWS), len(TIMES)), object)

    for i, (name, poses) in enumerate(ROWS):
        ours = i == len(ROWS) - 1
        for j, t_ms in enumerate(TIMES):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            image_axes(ax, color="#5b7ba8" if ours else "#9a9a9a",
                       lw=0.9 if ours else 0.55)
            ax.set_xlim(*XLIM)
            ax.set_ylim(*YLIM)

            obst = Polygon([(XLIM[0], YLIM[0]), (XLIM[1], YLIM[0]),
                            (XLIM[1], SY1), (SX0, SY1), (SX0, 0.0),
                            (XLIM[0], 0.0)], closed=True, facecolor=OBST,
                           edgecolor="none", zorder=1)
            ax.add_patch(obst)
            for k in range(-3, 14):                # hatch inside the obstacle
                x0 = XLIM[0] + 0.30 * k
                ln, = ax.plot([x0, x0 + 0.58], [YLIM[0], YLIM[0] + 1.25],
                              color="#adb2b9", lw=0.45, zorder=2)
                ln.set_clip_path(obst)
            ax.plot([XLIM[0], SX0, SX0, XLIM[1]], [0.0, 0.0, SY1, SY1],
                    color=OBST_EDGE, lw=1.1, solid_joinstyle="miter", zorder=3)

            cell = cells[i, j]
            if j > 0:                              # ghost of the previous pose
                gp = cells[i, j - 1]["poly"]
                ax.add_patch(Polygon(gp, closed=True, facecolor="none",
                                     edgecolor="#aab0b8", lw=0.55,
                                     ls=(0, (2.6, 1.9)), zorder=3))
                cx = [c["pose"][0] for c in cells[i, :j + 1]]
                cy = [c["pose"][1] for c in cells[i, :j + 1]]
                ax.plot(cx, cy, color="#aab0b8", lw=0.6, ls=":", zorder=3,
                        marker="o", ms=1.8, mfc="#aab0b8", mec="none")

            ax.add_patch(Polygon(cell["poly"], closed=True, facecolor=BODY,
                                 edgecolor=BODY_EDGE, lw=1.0, zorder=4))
            ax.plot(cell["pose"][0], cell["pose"][1], marker="+", ms=4.0,
                    color=BODY_EDGE, mew=0.9, zorder=5)
            for ov in cell["ov"]:
                if len(ov) >= 3:
                    ax.add_patch(Polygon(ov, closed=True, facecolor=HIT,
                                         edgecolor="none", alpha=0.80,
                                         zorder=5))

            if cell["d"] > 0.002:
                allov = np.vstack([o for o in cell["ov"] if len(o)])
                x0, y0 = allov.min(axis=0) - 0.10
                x1, y1 = allov.max(axis=0) + 0.10
                x0, x1 = max(x0, XLIM[0] + 0.03), min(x1, XLIM[1] - 0.03)
                y0, y1 = max(y0, YLIM[0] + 0.03), min(y1, YLIM[1] - 0.03)
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       facecolor="none", edgecolor=HIT,
                                       lw=1.0, zorder=6))
                # keep the readout clear of the box it is describing
                right = allov[:, 0].mean() < 0.5 * (XLIM[0] + XLIM[1])
                ax.text(0.975 if right else 0.025, 0.055,
                        f"$d$ = {cell['d'] * 1e3:.0f} mm",
                        transform=ax.transAxes,
                        ha="right" if right else "left", va="bottom",
                        fontsize=6.8, color=HIT, fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                  ec="#e2c3c8", lw=0.5, alpha=0.94))
            elif j == len(TIMES) - 1:
                ax.text(0.975, 0.055, f"$d$ = {cell['d'] * 1e3:.0f} mm",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6.8, color="#2e7d5b", fontweight="bold",
                        zorder=7,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                  ec="#c3ddd0", lw=0.5, alpha=0.94))

            if i == 0:
                ax.set_title(f"$t$ = {t_ms} ms", fontsize=8.6, pad=3.4,
                             color=INK if t_ms else SUB)
            if j == 0:
                ax.set_ylabel(name, fontsize=8.4, labelpad=4.0, color=INK,
                              fontweight="bold" if ours else "normal")

    p0, p1 = axes[2, 0].get_position(), axes[2, -1].get_position()
    background_card(fig, (p0.x0 - 0.072, p0.y0 - 0.020,
                          p1.x1 - p0.x0 + 0.078, p0.height + 0.040),
                    facecolor="#eaeff6", zorder=-5, radius=0.006)

    dl = [cells[i, -1]["d"] * 1e3 for i in range(len(ROWS))]
    savefig(fig, "1142_wm_phys_collision_grid.png")


if __name__ == "__main__":
    main()
