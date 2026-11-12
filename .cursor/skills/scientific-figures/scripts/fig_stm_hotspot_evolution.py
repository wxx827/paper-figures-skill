"""Diurnal evolution of kernel-density hotspots over a synthetic city.

Eight three-hour slices of a pick-up density surface are rendered on one
shared colour scale, computed once over the whole stack, so a panel that
looks hotter really is hotter - autoscaling each slice would manufacture a
night-time hotspot out of noise.  The dashed outline in every panel is the
same absolute contour used to measure hotspot area, and the full-width
auxiliary panel underneath tracks that area against the mean intensity
inside it, both read straight off the rasters that are plotted.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import add_scale_bar, panel_labels_inches, shared_norm

W, H = 7.8, 6.0                      # city extent in km
NG = 190                             # raster resolution
INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
AREA_C = "#b23a48"                   # house red
INT_C = "#3b5b92"                    # house blue
FRAME = "#c9c6bf"

SLOTS = ["00:00-03:00", "03:00-06:00", "06:00-09:00", "09:00-12:00",
         "12:00-15:00", "15:00-18:00", "18:00-21:00", "21:00-24:00"]

# (x, y, sigma, name, per-slot weight) - a CBD core, two suburban centres,
# a nightlife strip and a transport interchange, each with its own rhythm.
CENTRES = [
    (3.55, 3.15, 0.85, "CBD core",
     [0.15, 0.10, 0.86, 1.00, 0.78, 0.92, 0.66, 0.28]),
    (1.55, 4.55, 0.62, "north-west estate",
     [0.22, 0.16, 0.62, 0.35, 0.30, 0.44, 0.78, 0.46]),
    (6.05, 1.70, 0.66, "south-east estate",
     [0.18, 0.14, 0.58, 0.32, 0.28, 0.40, 0.72, 0.40]),
    (5.55, 4.60, 0.52, "nightlife strip",
     [0.82, 0.40, 0.10, 0.14, 0.26, 0.34, 0.70, 1.00]),
    (2.35, 1.35, 0.58, "rail interchange",
     [0.20, 0.24, 0.94, 0.42, 0.38, 0.60, 0.88, 0.34]),
]


def density_stack(rng):
    """Eight smooth pick-up density rasters in events per square kilometre."""
    x = np.linspace(0, W, NG)
    y = np.linspace(0, H, int(NG * H / W))
    X, Y = np.meshgrid(x, y)
    base = 3.0 * np.exp(-(((X - 3.9) / 4.6) ** 2 + ((Y - 3.0) / 3.4) ** 2))
    stack = []
    for t in range(8):
        f = base.copy()
        for cx, cy, sg, _name, wts in CENTRES:
            amp = 96.0 * wts[t] * np.exp(rng.normal(0.0, 0.05))
            sgx = sg * (1.0 + 0.16 * np.sin(0.8 * t + cx))
            sgy = sg * (1.0 + 0.16 * np.cos(0.7 * t + cy))
            f = f + amp * np.exp(-(((X - cx) / sgx) ** 2
                                   + ((Y - cy) / sgy) ** 2))
        grain = rng.normal(0.0, 1.0, X.shape)
        for _ in range(6):                      # cheap smoothing of the grain
            grain = 0.25 * (np.roll(grain, 1, 0) + np.roll(grain, -1, 0)
                            + np.roll(grain, 1, 1) + np.roll(grain, -1, 1))
        stack.append(np.clip(f + 9.0 * grain, 0.0, None))
    return np.array(stack), (x, y)


def main():
    set_pub_style()
    rng = np.random.default_rng(481)
    stack, (gx, gy) = density_stack(rng)

    # One norm for all eight panels; the threshold that defines a "hotspot"
    # is a single absolute level, so areas are comparable across slots.
    norm = shared_norm(stack, robust=(1, 99.6))
    cmap = get_cmap("cb_sequential_warm")
    thr = 0.42 * norm.vmax
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    area = np.array([(f > thr).sum() * cell for f in stack])
    inten = np.array([f[f > thr].mean() if (f > thr).any() else 0.0
                      for f in stack])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 4, left=0.045, right=0.902, top=0.878,
                          bottom=0.088, hspace=0.34, wspace=0.09,
                          height_ratios=[1.0, 1.0, 0.62])
    maps = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    axc = fig.add_subplot(gs[2, :])

    for t, ax in enumerate(maps):
        im = ax.imshow(stack[t], origin="lower", extent=[0, W, 0, H],
                       cmap=cmap, norm=norm, interpolation="bilinear")
        ax.contour(gx, gy, stack[t], levels=[thr], colors="#3a3a3a",
                   linewidths=0.65, linestyles=[(0, (2.6, 1.8))])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.7)
            sp.set_edgecolor(FRAME)
        ax.set_title(SLOTS[t], fontsize=8.2, fontweight="bold", color=INK,
                     pad=3)
        ax.text(0.972, 0.045, f"{area[t]:.1f} km$^2$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.2, color="#3a3a3a")

    add_scale_bar(maps[0], 2.0, "2 km", loc=(0.045, 0.055), height=0.030,
                  color="#333333", text_color="#1a1a1a", halo="white",
                  fontsize=6.0)
    for cx, cy, _s, name, _w in CENTRES[:1]:
        maps[3].annotate(name, xy=(cx, cy), xytext=(cx - 1.05, cy + 1.65),
                         fontsize=6.2, color="#1a1a1a", ha="center",
                         arrowprops=dict(arrowstyle="-", lw=0.6,
                                         color="#3a3a3a"))

    # one colour bar for the whole stack, spanning both map rows
    for ax in (maps[3], maps[7]):
        ax.apply_aspect()
    p0 = maps[3].get_position()
    p1 = maps[7].get_position()
    cax = fig.add_axes([p0.x1 + 0.014, p1.y0, 0.0125, p0.y1 - p1.y0])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.5, color="#333333")
    cb.set_label("pick-up density (count km$^{-2}$)", fontsize=7.0,
                 color=INK)
    cb.ax.axhline(thr, color="#3a3a3a", lw=0.9, ls=(0, (2.2, 1.6)))

    # ----------------------------------------------- (i) area vs intensity
    xs = np.arange(8)
    axc.bar(xs, area, width=0.56, color=AREA_C, alpha=0.85,
            edgecolor="white", linewidth=0.6, zorder=3, label="hotspot area")
    axc.set_ylabel("hotspot area (km$^2$)", fontsize=7.6, color=AREA_C,
                   labelpad=2)
    axc.tick_params(axis="y", labelsize=6.8, colors=AREA_C, length=2.4)
    axc.set_ylim(0, area.max() * 1.32)
    axc.set_xlim(-0.62, 7.62)
    axc.set_xticks(xs)
    axc.set_xticklabels([s.split("-")[0] for s in SLOTS], fontsize=6.8)
    axc.set_xlabel("time of day (start of 3 h slot)", fontsize=7.6,
                   labelpad=2)
    axc.grid(axis="y", color="#ececec", lw=0.55, zorder=0)
    axc.spines["top"].set_visible(False)

    axi = axc.twinx()
    axi.plot(xs, inten, color=INT_C, lw=1.5, marker="o", ms=3.6,
             mfc="white", mec=INT_C, mew=1.0, zorder=5,
             label="mean intensity inside hotspot")
    axi.set_ylabel("mean intensity (count km$^{-2}$)", fontsize=7.6,
                   color=INT_C, labelpad=3)
    axi.tick_params(axis="y", labelsize=6.8, colors=INT_C, length=2.4)
    axi.set_ylim(0, inten.max() * 1.24)
    axi.spines["top"].set_visible(False)
    axi.spines["right"].set_visible(True)
    axi.spines["right"].set_color(INT_C)
    axi.spines["left"].set_color(AREA_C)

    peak = int(np.argmax(area))
    axc.text(0.010, 0.94, f"peak extent {area[peak]:.1f} km$^2$ "
             f"in {SLOTS[peak]}", transform=axc.transAxes, fontsize=6.8,
             color=INK, fontweight="bold", ha="left", va="top", zorder=6)
    axc.text(0.0, -0.40, f"hotspot = density above {thr:.0f} count km$^{{-2}}$"
             " (same absolute level in all eight panels)",
             transform=axc.transAxes, fontsize=6.3, color=MUTED, ha="left",
             va="top")

    panel_labels_inches(fig, maps + [axc], dx=-0.16, dy=0.22)

    savefig(fig, "481_stm_hotspot_evolution.png")


if __name__ == "__main__":
    main()
