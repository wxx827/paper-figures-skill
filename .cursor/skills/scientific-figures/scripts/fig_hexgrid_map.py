"""Hexagonal binning of point observations over a synthetic landmass.

Point events (clustered mixture + background) are aggregated onto a hex grid;
the vertical colourbar is placed flush against a rotated count histogram so
the two share one axis, and a concentration (Lorenz) curve quantifies how
unevenly the events are distributed across cells.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from pubstyle import set_pub_style, savefig

CMAP = LinearSegmentedColormap.from_list(
    "ember", ["#f6f1e7", "#eed9b7", "#e0ac77", "#cd7d51", "#a9503f",
              "#74304a", "#3d1c33"])


def land_radius(theta):
    return (1.0 + 0.14 * np.cos(2 * theta + 1.1)
            + 0.10 * np.cos(3 * theta - 0.6)
            + 0.05 * np.cos(5 * theta + 2.4))


def inside(p):
    return np.hypot(p[:, 0], p[:, 1]) < land_radius(np.arctan2(p[:, 1], p[:, 0]))


def sample_points(rng, n_bg=6500):
    pts = []
    bg = rng.uniform(-1.35, 1.35, size=(int(n_bg * 2.6), 2))
    pts.append(bg[inside(bg)][:n_bg])
    clusters = [(-0.48, 0.42, 0.16, 2600), (0.55, -0.28, 0.13, 2100),
                (0.10, 0.72, 0.10, 1300), (-0.62, -0.48, 0.19, 1500),
                (0.78, 0.35, 0.09, 900), (-0.05, -0.15, 0.30, 2200)]
    for cx, cy, s, n in clusters:
        c = rng.normal([cx, cy], s, size=(int(n * 1.6), 2))
        pts.append(c[inside(c)][:n])
    return np.vstack(pts)


def main():
    set_pub_style()
    rng = np.random.default_rng(19)
    pts = sample_points(rng)
    t = np.linspace(0, 2 * np.pi, 500)
    coast = np.c_[np.cos(t), np.sin(t)] * land_radius(t)[:, None]

    fig = plt.figure(figsize=(10.4, 6.3))
    ax = fig.add_axes([0.015, 0.055, 0.585, 0.855])
    hb = ax.hexbin(pts[:, 0], pts[:, 1], gridsize=(30, 17), cmap=CMAP,
                   mincnt=1, edgecolors="white", linewidths=0.3,
                   extent=(-1.30, 1.30, -1.22, 1.28))
    counts = np.asarray(hb.get_array())
    vmax = np.quantile(counts, 0.99)
    hb.set_norm(Normalize(0, vmax))
    hb.set_clip_path(Polygon(coast, closed=True, transform=ax.transData))
    ax.plot(coast[:, 0], coast[:, 1], color="#3a3a3a", lw=1.0, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.32, 1.34)
    ax.set_ylim(-1.26, 1.32)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.0, 0.985, "(a)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")
    ax.text(0.048, 0.988, "Hex-binned event density", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", va="bottom", color="#74304a")
    ax.text(0.0, 0.955,
            "N = %s events   |   %d occupied cells   |   cell width 42 km"
            % (format(len(pts), ","), len(counts)),
            transform=ax.transAxes, fontsize=7, color="#666", va="bottom")

    top = np.argsort(counts)[-1]
    verts = hb.get_offsets()[top]
    ax.annotate("peak cell\n%d events" % counts[top], xy=tuple(verts),
                xytext=(verts[0] + 0.42, verts[1] + 0.40), fontsize=7.2,
                fontweight="bold", color="#3d1c33", ha="center",
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#d9cfc4",
                          lw=0.5, alpha=0.93),
                arrowprops=dict(arrowstyle="-", color="#3d1c33", lw=0.8,
                                shrinkA=2, shrinkB=2), zorder=6)
    ax.annotate("N", xy=(0.055, 0.90), xytext=(0.055, 0.80),
                xycoords="axes fraction", ha="center", fontsize=7, color="#333",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    ax.plot([0.03, 0.20], [0.035, 0.035], transform=ax.transAxes,
            color="#333", lw=1.8, solid_capstyle="butt")
    ax.text(0.115, 0.052, "200 km", transform=ax.transAxes, fontsize=6,
            ha="center", color="#333")

    # ---- colourbar + count histogram sharing the y axis
    cax = fig.add_axes([0.645, 0.545, 0.016, 0.345])
    cb = fig.colorbar(hb, cax=cax, extend="max")
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#777777")
    cax.set_ylabel("events per cell", fontsize=7, labelpad=3)
    cax.yaxis.set_label_position("left")
    cax.yaxis.set_ticks_position("left")

    axh = fig.add_axes([0.663, 0.545, 0.30, 0.345])
    nb = 22
    edges = np.linspace(0, vmax, nb + 1)
    cnt, _ = np.histogram(np.clip(counts, 0, vmax), bins=edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    axh.barh(ctr, cnt, height=(edges[1] - edges[0]) * 0.86,
             color=CMAP(ctr / vmax), edgecolor="white", linewidth=0.3)
    axh.set_ylim(0, vmax)
    axh.set_yticks([])
    axh.set_xlabel("number of cells", fontsize=7.5, labelpad=2)
    axh.tick_params(labelsize=6.5, length=2.5)
    axh.spines["left"].set_visible(False)
    med = np.median(counts)
    axh.axhline(med, color="#333", lw=0.8, ls=(0, (3, 2)))
    axh.text(axh.get_xlim()[1] * 0.97, med + vmax * 0.03,
             "median %d" % med, fontsize=6.5, ha="right", color="#333")
    axh.text(-0.30, 1.06, "(b)  Cell-count distribution",
             transform=axh.transAxes, fontsize=9, fontweight="bold",
             va="bottom")

    # ---- concentration curve
    axl = fig.add_axes([0.700, 0.105, 0.263, 0.315])
    srt = np.sort(counts)[::-1]
    fx = np.arange(1, len(srt) + 1) / len(srt)
    fy = np.cumsum(srt) / srt.sum()
    axl.plot(fx * 100, fy * 100, color="#a9503f", lw=1.8)
    axl.plot([0, 100], [0, 100], color="#999", lw=0.8, ls=(0, (3, 2)))
    axl.fill_between(fx * 100, fx * 100, fy * 100, color="#a9503f", alpha=0.12)
    i10 = int(0.10 * len(srt))
    axl.plot([10], [fy[i10] * 100], "o", ms=4.5, color="#3d1c33", zorder=5)
    axl.annotate("top 10%% of cells\nhold %.0f%% of events" % (fy[i10] * 100),
                 xy=(10, fy[i10] * 100), xytext=(26, 30), fontsize=6.8,
                 color="#3d1c33", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color="#3d1c33", lw=0.7))
    axl.set_xlim(0, 100); axl.set_ylim(0, 100)
    axl.set_xlabel("cells ranked by density (%)", fontsize=7.5, labelpad=2)
    axl.set_ylabel("cumulative events (%)", fontsize=7.5, labelpad=2)
    axl.tick_params(labelsize=6.5, length=2.5)
    axl.set_xticks([0, 25, 50, 75, 100])
    axl.set_yticks([0, 25, 50, 75, 100])
    axl.text(-0.22, 1.06, "(c)  Spatial concentration",
             transform=axl.transAxes, fontsize=9, fontweight="bold",
             va="bottom")

    savefig(fig, "57_hexgrid_map.png")


if __name__ == "__main__":
    main()
