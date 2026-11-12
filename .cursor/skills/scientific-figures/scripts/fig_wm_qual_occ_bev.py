"""Figure 1089 - BEV occupancy forecasting: GT vs ours vs a baseline.

Rows are ground truth, our forecast and a baseline forecast; columns are four
prediction horizons of the same intersection approach.  Every cell is a
discrete semantic class drawn from one shared blue-grey palette (two warm
accents only), and the ego box plus the planned trajectory are overlaid on
every tile so the reader can compare geometry, not just colour.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Polygon
from scipy.ndimage import binary_dilation, gaussian_filter

from pubstyle import set_pub_style, savefig
from figkit import image_axes

RES = 0.5                        # metres per cell
NX, NY = 104, 72                 # lateral x longitudinal
Y0 = -5.0                        # metre coordinate of row 0
HALF = 8.6                       # main-road half width (m)

FREE, DRIVE, LANE, VEH, PED = 0, 1, 2, 3, 4
CLASS_COL = ["#eff1f3",          # free / unobserved
             "#aebecd",          # drivable surface
             "#c9a227",          # lane marking      (warm accent 1)
             "#3b5b92",          # vehicle
             "#b23a48"]          # pedestrian        (warm accent 2)
CLASS_NAME = ["Unobserved", "Drivable", "Lane marking", "Vehicle", "Pedestrian"]
CMAP = ListedColormap(CLASS_COL)

HORIZONS = [0.5, 1.0, 2.0, 3.0]
ROWS = [("Ground truth", 0.00, 0.0, False),
        ("Ours", 0.13, 0.35, False),
        ("OccWorld-S", 0.55, 1.30, True)]

# (lateral offset from the road centre, world y at t=0, speed, length, width)
TRAFFIC = [(-4.2, 12.0, 4.5, 4.6, 1.9),
           (-4.4, 30.0, 6.0, 4.4, 1.9),
           (4.3, 26.0, -7.5, 4.8, 2.0),
           (4.2, 8.5, -6.5, 4.5, 1.9),
           (-11.6, 21.0, 0.0, 4.6, 2.0)]
# world-frame pedestrians and kerbside parking (absolute x, y)
PEDS = [(-8.0, 20.4, 1.2), (8.4, 23.0, -1.1)]
LOTS = [(-23.0, -11.5, 1.0, 15.0), (12.0, 23.5, 4.0, 17.0),
        (-23.0, -12.5, 27.0, 40.0)]
PARKED = [(-20.5, 4.0), (-20.5, 9.5), (-16.0, 12.5),
          (16.5, 7.0), (20.5, 12.0), (-19.0, 31.0), (-15.0, 35.5)]
EGO_SPEED = 5.0


def road_center(y):
    return 3.1 * np.sin((y + 6.0) / 26.0) - 0.9


def road_heading(y):
    return np.arctan(3.1 * np.cos((y + 6.0) / 26.0) / 26.0)


def grids():
    x = (np.arange(NX) - NX / 2 + 0.5) * RES
    y = Y0 + (np.arange(NY) + 0.5) * RES
    return np.meshgrid(x, y)


def rot_rect(XX, YY, cx, cy, length, width, th):
    dx, dy = XX - cx, YY - cy
    lon = dx * np.sin(th) + dy * np.cos(th)
    lat = dx * np.cos(th) - dy * np.sin(th)
    return (np.abs(lon) <= length / 2) & (np.abs(lat) <= width / 2)


def build_bev(t, jitter, wobble, degrade, seed):
    """Semantic BEV at horizon ``t`` seconds; ``jitter``/``wobble`` fake error."""
    rng = np.random.default_rng(seed)
    XX, SY = grids()
    # the view is ego-centred, so static structure slides towards the ego
    YY = SY + EGO_SPEED * t
    lab = np.full((NY, NX), FREE)

    cxy = road_center(YY)
    half = HALF
    if wobble > 0:
        n = gaussian_filter(rng.standard_normal((NY, NX)), (5.0, 7.0))
        half = HALF + wobble * n / (n.std() + 1e-9)
    main = np.abs(XX - cxy) <= half
    cross = (YY > 17.6) & (YY < 25.6)
    if wobble > 0:
        cross = (YY > 17.6 + 0.6 * wobble) & (YY < 25.6 - 0.5 * wobble)
    lots = np.zeros_like(main)
    for (lx0, lx1, ly0, ly1) in LOTS:
        lots |= (XX > lx0) & (XX < lx1) & (YY > ly0) & (YY < ly1)
    if wobble > 0.6:
        lots &= gaussian_filter(rng.standard_normal((NY, NX)), 3.0) > -0.30 * wobble
    lab[main | cross | lots] = DRIVE

    # ---- lane markings: solid centre line, dashed lane dividers
    for off, dashed in ((0.0, False), (-4.3, True), (4.3, True)):
        m = np.abs(XX - (cxy + off)) < 0.30 + 0.10 * jitter
        if dashed:
            m &= (YY % 6.0) < 3.2
        lab[m & main] = LANE
    lab[(np.abs(YY - 21.6) < 0.30) & cross] = LANE

    # ---- kerbside parking, then moving traffic; all in world coordinates
    lo, hi = Y0 + EGO_SPEED * t, Y0 + NY * RES + EGO_SPEED * t
    for (px, py) in PARKED:
        if not (lo + 1 < py < hi - 1):
            continue
        m = rot_rect(XX, YY, px, py, 4.4, 1.9, 0.0)
        lab[binary_dilation(m, iterations=1) if degrade else m] = VEH

    for k, (off, y0, spd, ln, wd) in enumerate(TRAFFIC):
        if degrade and k == 1 and t >= 2.0:      # baseline loses a far vehicle
            continue
        y = y0 + spd * t
        if not (lo + 1 < y < hi - 1):
            continue
        e = jitter * t
        y += rng.normal(0, e + 1e-6)
        x = road_center(y) + off + rng.normal(0, 0.6 * e + 1e-6)
        th = road_heading(y) + (np.pi if spd < 0 else 0.0)
        m = rot_rect(XX, YY, x, y, ln, wd, th)
        if degrade:
            m = binary_dilation(m, iterations=1 + int(t >= 2.0))
        lab[m] = VEH
    if degrade and t >= 2.0:                     # ... and hallucinates one
        m = rot_rect(XX, YY, -10.0, lo + 12.0, 4.2, 2.2, 0.1)
        lab[binary_dilation(m, iterations=1)] = VEH

    for (px, py, vx) in PEDS:
        if degrade and t >= 2.0:                 # baseline drops pedestrians
            continue
        x = px + vx * t + rng.normal(0, 0.5 * jitter * t + 1e-6)
        m = ((XX - x) ** 2 + (YY - py) ** 2) < (0.85 ** 2)
        lab[binary_dilation(m, iterations=1)] = PED
    return lab


def miou(pred, gt):
    vals = []
    for c in (DRIVE, LANE, VEH, PED):
        p, g = pred == c, gt == c
        union = float((p | g).sum())
        if union > 0:
            vals.append(float((p & g).sum()) / union)
    return float(np.mean(vals)) if vals else 0.0


def overlay_ego(ax):
    cx, cy = NX / 2 - 0.5, (0.0 - Y0) / RES - 0.5
    L, Wd = 4.7 / RES, 1.9 / RES
    ax.add_patch(Rectangle((cx - Wd / 2, cy - L / 2), Wd, L, facecolor="#f7f9fb",
                           edgecolor="#1a1a1a", lw=0.9, zorder=6))
    ax.add_patch(Polygon([[cx - Wd / 2, cy + L / 2], [cx + Wd / 2, cy + L / 2],
                          [cx, cy + L / 2 + 2.0]], closed=True,
                         facecolor="#1a1a1a", edgecolor="none", zorder=6))


def overlay_plan(ax, t):
    y = np.linspace(2.0, 29.0, 40)
    x = road_center(y + EGO_SPEED * t) + 4.3 * np.clip((y - 9.0) / 14.0, 0, 1)
    col = (x / RES) + NX / 2 - 0.5
    row = (y - Y0) / RES - 0.5
    ax.plot(col, row, color="#ffffff", lw=1.5, zorder=7, solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=2.7, foreground="#1a1a1a"),
                          pe.Normal()])
    ax.plot(col[::9], row[::9], ls="none", marker="o", ms=2.6,
            markerfacecolor="#ffffff", markeredgecolor="#1a1a1a", mew=0.6,
            zorder=8)


def main():
    set_pub_style(base_font=8)

    maps = np.empty((len(ROWS), len(HORIZONS)), object)
    for i, (_, jit, wob, deg) in enumerate(ROWS):
        for j, t in enumerate(HORIZONS):
            maps[i, j] = build_bev(t, jit, wob, deg, seed=900 + 17 * j)

    fig = plt.figure(figsize=(10.0, 6.6))
    gs = fig.add_gridspec(3, 4, wspace=0.04, hspace=0.10,
                          left=0.078, right=0.988, top=0.885, bottom=0.118)

    for i, (name, _, _, _) in enumerate(ROWS):
        for j, t in enumerate(HORIZONS):
            ax = fig.add_subplot(gs[i, j])
            image_axes(ax, color="#9a9a9a", lw=0.55)
            ax.set_aspect("auto")
            ax.imshow(maps[i, j], cmap=CMAP, vmin=-0.5, vmax=4.5,
                      origin="lower", interpolation="nearest", aspect="auto")
            overlay_ego(ax)
            overlay_plan(ax, t)
            if i == 0:
                ax.set_title(f"t + {t:.1f} s", fontsize=8.6, pad=3.6)
            else:
                ax.text(0.975, 0.965, f"mIoU {miou(maps[i, j], maps[0, j]):.2f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.3, color="#1a1a1a", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.22",
                                  facecolor="#ffffffcc", edgecolor="none"))
            if j == 0:
                ax.set_ylabel(name, fontsize=8.4, labelpad=3.6,
                              fontweight="bold" if name == "Ours" else "normal")

    handles = [Patch(facecolor=c, edgecolor="#9a9a9a", lw=0.4, label=n)
               for c, n in zip(CLASS_COL, CLASS_NAME)]
    handles += [
        Patch(facecolor="#f7f9fb", edgecolor="#1a1a1a", lw=0.8, label="Ego vehicle"),
        Line2D([0], [0], color="#ffffff", lw=1.5, marker="o", ms=2.8,
               markerfacecolor="#ffffff", markeredgecolor="#1a1a1a", mew=0.6,
               label="Planned trajectory",
               path_effects=[pe.Stroke(linewidth=2.7, foreground="#1a1a1a"),
                             pe.Normal()])]
    fig.legend(handles=handles, ncol=7, loc="lower center",
               bbox_to_anchor=(0.53, 0.028), frameon=False, fontsize=7.2,
               handlelength=1.3, handleheight=0.95, columnspacing=1.5,
               handletextpad=0.5)

    savefig(fig, "1089_wm_qual_occ_bev.png")


if __name__ == "__main__":
    main()
