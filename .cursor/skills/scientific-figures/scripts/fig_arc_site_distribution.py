"""Settlement pattern: sites on the terrain, sized by area and ranked by size.

The map is the argument and everything on it is derived from one synthetic
elevation grid: the drainage lines are traced by steepest descent down that
grid, so the rivers really do run downhill and really do join, and the sites
are drawn from a riverine preference over the same surface.  Thiessen cells
around the first-tier sites give the reader the territorial reading that a
scatter of dots cannot.

The two side panels test what the map suggests.  A rank-size plot on log-log
axes compares the observed hierarchy against a strict Zipf line fitted to the
same areas, and the period counts show that the pattern is not one moment but
the sum of five.  Period colours are the shared archaeological set used in
630 and 631.
"""
import textwrap
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi

from pubstyle import set_pub_style, savefig, darken
from figkit import add_scale_bar, axes_inches, halo_text, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"

PERIODS = ["Neolithic", "Bronze Age", "Iron Age", "Roman", "Medieval"]
PERIOD_COLOR = {
    "Neolithic": "#8e6bbf",
    "Bronze Age": "#3b5b92",
    "Iron Age": "#2e7d5b",
    "Roman": "#c9a227",
    "Medieval": "#b23a48",
}
PERIOD_SHORT = {"Neolithic": "Neo", "Bronze Age": "BA", "Iron Age": "IA",
                "Roman": "Rom", "Medieval": "Med"}
PERIOD_WEIGHT = [0.13, 0.18, 0.24, 0.28, 0.17]

# pale hypsometric ramp, monotone in lightness (DESIGN_SYSTEM section 2.2)
TERRAIN_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "terrain_pale", ["#f6f3ea", "#e9e3d2", "#dbd2ba", "#cabfa2", "#b6a988"])
WATER = "#6d8fae"

FW, FH = 10.4, 6.4
MAP_RECT = (0.60, 0.98, 5.55, 4.24)
KM_Y = 33.0
KM_X = KM_Y * MAP_RECT[2] / MAP_RECT[3]      # equal aspect fills the slot
NX, NY = 320, 190
N_TIER1 = 7


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def valley_axis(x):
    """Centre line of the trunk valley, km north of the map edge."""
    return 0.44 * KM_Y + 3.4 * np.sin(x / 8.5) + 1.6 * np.cos(x / 3.1)


def build_terrain(rng):
    x = np.linspace(0.0, KM_X, NX)
    y = np.linspace(0.0, KM_Y, NY)
    X, Y = np.meshgrid(x, y)
    rough = gaussian_filter(rng.normal(0.0, 1.0, (NY, NX)), 11.0)
    rough /= np.abs(rough).max()
    hills = gaussian_filter(rng.normal(0.0, 1.0, (NY, NX)), 4.5)
    hills /= np.abs(hills).max()
    z = 330.0 + 112.0 * rough + 15.0 * hills
    z -= 140.0 * np.exp(-((Y - valley_axis(X)) / 4.6) ** 2)    # trunk valley
    z -= 2.60 * X                                              # downstream
    return x, y, gaussian_filter(z, 3.4)


def river_seeds(z, k=8, sep=30):
    """Highest points, kept apart so the traces do not all share a valley."""
    order = np.argsort(z, axis=None)[::-1]
    picked = []
    for flat in order:
        i, j = np.unravel_index(flat, z.shape)
        if all((i - a) ** 2 + (j - b) ** 2 > sep ** 2 for a, b in picked):
            picked.append((int(i), int(j)))
        if len(picked) >= k:
            break
    return picked


def descend(z, i, j, n_max=900):
    """Steepest-descent trace: this is what makes the rivers plausible."""
    path = [(j, i)]
    ny, nx = z.shape
    for _ in range(n_max):
        best, bz = None, z[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < ny and 0 <= jj < nx and z[ii, jj] < bz:
                    bz, best = z[ii, jj], (ii, jj)
        if best is None:
            break
        i, j = best
        path.append((j, i))
    return np.asarray(path)


def trace_rivers(x, y, z):
    out = []
    for i, j in river_seeds(z):
        p = descend(z, i, j)
        if p.shape[0] > 30:
            out.append(np.column_stack([x[p[:, 0]], y[p[:, 1]]]))
    return out


def place_sites(rng, x, y, z, n=46):
    """Riverine preference plus a minority of upland sites."""
    xs, ys, upland = [], [], []
    for k in range(n):
        up = rng.random() < 0.20
        px = rng.uniform(1.5, KM_X - 1.5)
        py = (rng.uniform(1.5, KM_Y - 1.5) if up
              else valley_axis(px) + rng.normal(0.0, 3.1))
        xs.append(px)
        ys.append(float(np.clip(py, 1.2, KM_Y - 1.2)))
        upland.append(up)
    xs, ys = np.array(xs), np.array(ys)
    # Pareto tail: a settlement system has a few big places and many small
    area = 0.62 * rng.random(n) ** (-1.0 / 1.18)
    area = np.clip(area, 0.5, 62.0)
    period = rng.choice(PERIODS, size=n, p=PERIOD_WEIGHT)
    elev = np.array([z[int(np.clip(v / KM_Y * (NY - 1), 0, NY - 1)),
                       int(np.clip(u / KM_X * (NX - 1), 0, NX - 1))]
                     for u, v in zip(xs, ys)])
    return dict(x=xs, y=ys, area=area, period=period, elev=elev,
                upland=np.array(upland))


def bounded_voronoi(pts):
    """Mirror the points across all four edges so every cell is finite."""
    p = np.asarray(pts, dtype=float)
    mirrors = [p.copy() for _ in range(4)]
    mirrors[0][:, 0] = -p[:, 0]
    mirrors[1][:, 0] = 2.0 * KM_X - p[:, 0]
    mirrors[2][:, 1] = -p[:, 1]
    mirrors[3][:, 1] = 2.0 * KM_Y - p[:, 1]
    vor = Voronoi(np.vstack([p] + mirrors))
    cells = []
    for i in range(len(p)):
        reg = vor.regions[vor.point_region[i]]
        if reg and -1 not in reg:
            cells.append(vor.vertices[reg])
    return cells


def draw_map(ax, x, y, z, rivers, sites):
    ax.set_xlim(0.0, KM_X)
    ax.set_ylim(0.0, KM_Y)
    ax.set_aspect("equal")

    lo = np.floor(z.min() / 20.0) * 20.0
    levels = np.arange(lo, z.max() + 20.0, 20.0)
    cf = ax.contourf(x, y, z, levels=levels, cmap=TERRAIN_CMAP, zorder=1,
                     extend="both")
    cf.set_edgecolor("face")
    rasterize(cf)
    cs = ax.contour(x, y, z, levels=levels, colors="#a89f88",
                    linewidths=0.35, zorder=2)
    rasterize(cs)
    lab = ax.clabel(cs, levels=levels[3::8], inline=True, fontsize=6.0,
                    fmt=lambda v: f"{v:.0f} m")
    for t in lab:
        t.set_color("#7f7460")

    for riv in rivers:
        n = riv.shape[0]
        for a, b, lw in ((0.0, 0.5, 0.5), (0.45, 1.0, 0.9)):
            seg = riv[int(a * n):max(int(b * n), int(a * n) + 2)]
            ax.plot(seg[:, 0], seg[:, 1], color=WATER, lw=lw, alpha=0.95,
                    solid_capstyle="round", zorder=3)
    # trunk channel along the floor of the valley the terrain was built around
    xt = np.linspace(0.0, KM_X, 400)
    ax.plot(xt, valley_axis(xt), color=WATER, lw=1.9, alpha=0.95,
            solid_capstyle="round", zorder=3)

    # Thiessen territories of the first-tier centres
    tier1 = sites["area"] >= np.sort(sites["area"])[-N_TIER1]
    for cell in bounded_voronoi(np.column_stack([sites["x"][tier1],
                                                 sites["y"][tier1]])):
        ax.add_patch(Polygon(cell, closed=True, facecolor="none",
                             edgecolor="#6f6f6f", linewidth=0.65,
                             linestyle=(0, (4.0, 2.2)), zorder=4))

    smax = float(sites["area"].max())
    for p in PERIODS:
        m = sites["period"] == p
        ax.scatter(sites["x"][m], sites["y"][m],
                   s=8.0 + 78.0 * np.sqrt(sites["area"][m] / smax),
                   color=PERIOD_COLOR[p], alpha=0.88, edgecolors="white",
                   linewidths=0.55, zorder=6)
    ax.scatter(sites["x"][tier1], sites["y"][tier1],
               s=8.0 + 78.0 * np.sqrt(sites["area"][tier1] / smax),
               facecolors="none", edgecolors="#2b2b2b", linewidths=0.8,
               zorder=7)

    for i in np.argsort(-sites["area"])[:3]:
        px = float(sites["x"][i])
        ha = "left" if px < 7.0 else ("right" if px > KM_X - 7.0 else "center")
        dx = 0.9 if ha == "left" else (-0.9 if ha == "right" else 0.0)
        halo_text(ax, px + dx, sites["y"][i] + 1.05,
                  f"S{i + 1:02d}  {sites['area'][i]:.0f} ha", ha=ha,
                  va="bottom", fontsize=6.2, color="white", halo="#2b2b2b",
                  halo_lw=1.9, zorder=9)

    add_scale_bar(ax, 10.0, "10 km", loc=(0.655, 0.918), height=0.013,
                  fontsize=6.2, color="#3a3a3a", text_color=INK,
                  halo="white")
    ax.set_xlabel("Easting (km)", fontsize=8.0)
    ax.set_ylabel("Northing (km)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.set_title("Recorded sites, Upper Aln survey block", fontsize=8.5,
                 fontweight="bold")

    size_keys = [1.0, 10.0, 40.0]
    handles = [Line2D([0], [0], marker="o", ls="none", mfc="#bdbdbd",
                      mec="white", mew=0.5,
                      ms=np.sqrt(8.0 + 78.0 * np.sqrt(a / smax)),
                      label=f"{a:.0f} ha") for a in size_keys]
    leg = ax.legend(handles=handles, loc="lower right", frameon=True,
                    framealpha=0.9, facecolor="white", edgecolor=HAIR,
                    borderpad=0.4, fontsize=6.6, labelspacing=0.62,
                    handletextpad=0.6, title="site area")
    leg.get_title().set_fontsize(6.6)
    return tier1


def draw_ranksize(ax, sites):
    a = np.sort(sites["area"])[::-1]
    rank = np.arange(1, a.size + 1)
    slope, intercept = np.polyfit(np.log10(rank), np.log10(a), 1)
    ax.loglog(rank, a, marker="o", ms=3.4, ls="none", mfc="#3b5b92",
              mec="white", mew=0.4, zorder=4, label="observed")
    ax.loglog(rank, 10.0 ** (intercept + slope * np.log10(rank)),
              color="#b23a48", lw=1.2, zorder=5,
              label=f"fit, slope {slope:.2f}")
    ax.loglog(rank, a[0] / rank, color="#808080", lw=0.9, ls=(0, (4, 2)),
              zorder=3, label="strict Zipf, slope -1")
    ax.set_xlabel("rank (-)", fontsize=8.0)
    ax.set_ylabel("site area (ha)", fontsize=8.0)
    ax.tick_params(labelsize=6.8, which="both")
    ax.set_title("Rank-size hierarchy", fontsize=8.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6.6, loc="lower left",
              handlelength=1.6, labelspacing=0.3, borderpad=0.15)
    return slope


def draw_counts(ax, sites):
    counts = [int(np.sum(sites["period"] == p)) for p in PERIODS]
    ax.bar(np.arange(len(PERIODS)), counts,
           color=[PERIOD_COLOR[p] for p in PERIODS], width=0.66,
           edgecolor=[darken(PERIOD_COLOR[p], 0.28) for p in PERIODS],
           linewidth=0.7)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.4, str(c), ha="center", va="bottom", fontsize=6.2,
                color=SUB)
    ax.set_xticks(np.arange(len(PERIODS)))
    ax.set_xticklabels([PERIOD_SHORT[p] for p in PERIODS], fontsize=6.8)
    ax.set_ylim(0, max(counts) * 1.28)
    ax.set_ylabel("sites (n)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Sites per period", fontsize=8.0, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return counts


def main():
    set_pub_style()
    rng = np.random.default_rng(632)
    x, y, z = build_terrain(rng)
    rivers = trace_rivers(x, y, z)
    sites = place_sites(rng, x, y, z)

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAP_RECT)
    ax_r = axes_inches(fig, (7.00, 2.82, 3.00, 2.40))
    ax_c = axes_inches(fig, (7.00, 0.98, 3.00, 1.30))

    tier1 = draw_map(ax_m, x, y, z, rivers, sites)
    slope = draw_ranksize(ax_r, sites)
    counts = draw_counts(ax_c, sites)

    handles = [Line2D([0], [0], marker="o", ls="none", ms=4.6,
                      mfc=PERIOD_COLOR[p], mec="white", mew=0.5, label=p)
               for p in PERIODS]
    handles.append(Line2D([0], [0], color="#6f6f6f", lw=0.8,
                          ls=(0, (4.0, 2.2)), label="Thiessen territory"))
    handles.append(Line2D([0], [0], color=WATER, lw=1.4, label="drainage"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.335, 0.062), ncol=7, frameon=False,
               fontsize=6.8, handlelength=1.5, columnspacing=1.1)

    panel_labels_inches(fig, [ax_m, ax_r, ax_c], dx=-0.32, dy=0.20)

    top3 = np.sort(sites["area"])[::-1][:3].sum() / sites["area"].sum()

    print(f"n={sites['area'].size} tier1={int(tier1.sum())} "
          f"slope={slope:.3f} top3={top3:.3f} counts={counts}")
    savefig(fig, "632_arc_site_distribution.png")


if __name__ == "__main__":
    main()
