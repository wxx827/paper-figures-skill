"""Surface current field of a synthetic shelf sea.

A non-divergent stream function (meandering shelf-break jet plus four Gaussian
vortices) is differentiated into a velocity field, shaded by current speed and
traced with streamlines whose width follows the speed. Land is masked by a
coastline polygon and an offshore island; eddies, the coastal jet and the
shelf break are called out, and a bar scale gives the horizontal distance.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle
from pubstyle import set_pub_style, savefig

C_DEEP = "#17384f"
C_MAIN = "#2c7a8c"
C_LIGHT = "#8fbdd0"
C_ACC = "#c0603f"
C_LAND = "#e2e0da"

LON0, LON1 = 118.0, 127.0
LAT0, LAT1 = 28.0, 34.0
ASPECT = 1.0 / np.cos(np.deg2rad(31.0))     # display ratio, lat over lon
KM_PER_DEG_LON = 111.32 * np.cos(np.deg2rad(31.0))

EDDIES = [
    # lon, lat, strength (+ cyclonic), radius in degrees
    dict(x=121.4, y=32.5, a=+1.85, r=0.80, kind="cyclonic"),
    dict(x=123.9, y=30.3, a=-2.35, r=0.98, kind="anticyclonic"),
    dict(x=125.7, y=32.8, a=+1.55, r=0.72, kind="cyclonic"),
    dict(x=120.7, y=29.1, a=-1.05, r=0.55, kind="anticyclonic"),
]


def coast_lon(lat):
    """Longitude of the coastline as a function of latitude."""
    return (118.95 + 0.42 * np.sin((lat - 28.0) * 1.45)
            + 0.22 * np.sin((lat - 28.0) * 3.1 + 0.6)
            - 0.10 * (lat - 31.0))


def land_mask(X, Y):
    mainland = X < coast_lon(Y)
    island = (((X - 123.05) / 0.42) ** 2 + ((Y - 33.05) / 0.26) ** 2) < 1.0
    islet = (((X - 121.85) / 0.17) ** 2 + ((Y - 28.62) / 0.12) ** 2) < 1.0
    return mainland | island | islet


def stream_function(X, Y):
    """Meandering jet + vortices, in arbitrary units."""
    ysb = 30.6 + 0.55 * np.sin(2 * np.pi * (X - LON0) / 6.2)
    psi = -0.52 * (Y - ysb) - 0.075 * (Y - ysb) ** 2
    psi -= 0.72 * np.exp(-((X - coast_lon(Y)) / 0.50) ** 2)   # coastal jet
    for e in EDDIES:
        psi += e["a"] * e["r"] * np.exp(
            -(((X - e["x"]) / e["r"]) ** 2 + ((Y - e["y"]) / e["r"]) ** 2))
    return psi


def main():
    set_pub_style()

    nx, ny = 420, 300
    x = np.linspace(LON0, LON1, nx)
    y = np.linspace(LAT0, LAT1, ny)
    X, Y = np.meshgrid(x, y)

    psi = stream_function(X, Y)
    dy = (y[1] - y[0]) * 111.32
    dx = (x[1] - x[0]) * KM_PER_DEG_LON
    u = -np.gradient(psi, axis=0) / dy
    v = np.gradient(psi, axis=1) / dx
    scale = 1.45 / np.nanpercentile(np.hypot(u, v), 99.0)
    u, v = u * scale, v * scale
    spd = np.hypot(u, v)

    land = land_mask(X, Y)
    um = np.ma.masked_where(land, u)
    vm = np.ma.masked_where(land, v)
    sm = np.ma.masked_where(land, spd)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "shelf", ["#f6fafc", "#e0eef4", "#c2dde9", "#9dc6d9", "#75aac7",
                  "#4d8bb0", "#2f6d95"])

    fig = plt.figure(figsize=(9.2, 6.3))
    ax = fig.add_axes([0.062, 0.088, 0.812, 0.775])
    cax = fig.add_axes([0.888, 0.215, 0.017, 0.520])

    levels = np.linspace(0, 1.5, 31)
    cf = ax.contourf(X, Y, sm, levels=levels, cmap=cmap, extend="max",
                     zorder=1)
    cf.set_edgecolor("face")
    ax.contour(X, Y, sm, levels=[0.9], colors="#ffffff", linewidths=0.5,
               alpha=0.6, zorder=2)

    lw = 0.22 + 1.15 * (sm / 1.5)
    ax.streamplot(x, y, um, vm, density=(1.28, 0.95), color="#1d4159",
                  linewidth=lw, arrowsize=0.7, arrowstyle="-|>",
                  minlength=0.22, zorder=4)

    # ---- land -------------------------------------------------------------
    ylan = np.linspace(LAT0 - 0.2, LAT1 + 0.2, 400)
    poly = np.column_stack([coast_lon(ylan), ylan])
    poly = np.vstack([[LON0 - 0.4, LAT0 - 0.2], poly,
                      [LON0 - 0.4, LAT1 + 0.2]])
    ax.add_patch(Polygon(poly, closed=True, facecolor=C_LAND,
                         edgecolor="#9a978e", linewidth=0.8, zorder=6))
    for cx, cy, w, h in ((123.05, 33.05, 0.84, 0.52), (121.85, 28.62, 0.34,
                                                       0.24)):
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor=C_LAND,
                             edgecolor="#9a978e", linewidth=0.8, zorder=6))
    ax.text(118.34, 31.6, "M A I N L A N D", rotation=90, ha="center",
            va="center", fontsize=7.4, color="#8d8a82", fontweight="bold",
            zorder=7)
    ax.text(123.05, 33.05, "Ao I.", ha="center", va="center", fontsize=6.8,
            color="#7f7c74", zorder=7)

    # ---- eddy call-outs ---------------------------------------------------
    for e, (lx, ly, ha) in zip(
            EDDIES,
            [(121.4, 33.62, "center"), (123.7, 28.75, "center"),
             (126.55, 33.72, "center"), (119.95, 28.40, "center")]):
        w = 2.05 * e["r"]
        ax.add_patch(Ellipse((e["x"], e["y"]), w, w / ASPECT, fill=False,
                             edgecolor="#2b4a5e", linewidth=0.8,
                             linestyle=(0, (3.2, 2.2)), alpha=0.85, zorder=5))
    lab = [("cyclonic eddy", 121.4, 33.55), ("anticyclonic eddy", 123.7, 28.9),
           ("cyclonic eddy", 125.75, 33.75)]
    for txt, lx, ly in lab:
        ax.text(lx, ly, txt, ha="center", va="center", fontsize=7.2,
                color="#25404f", zorder=8,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="#cfd8dd", linewidth=0.5, alpha=0.9))
    ax.annotate("", xy=(121.4, 33.28), xytext=(121.4, 33.42),
                arrowprops=dict(arrowstyle="-", color="#5c7a8a", lw=0.7),
                zorder=8)
    ax.annotate("", xy=(123.7, 29.35), xytext=(123.7, 29.05),
                arrowprops=dict(arrowstyle="-", color="#5c7a8a", lw=0.7),
                zorder=8)

    ax.annotate("coastal jet", xy=(119.55, 30.35), xytext=(120.9, 30.05),
                fontsize=7.4, color=C_ACC, ha="left", va="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=C_ACC, lw=0.9,
                                connectionstyle="arc3,rad=0.25"))
    ax.plot(x, 30.6 + 0.55 * np.sin(2 * np.pi * (x - LON0) / 6.2),
            color="#b0451f", lw=0.9, ls=(0, (5, 2.5)), alpha=0.5, zorder=5,
            clip_on=True)
    ax.text(126.82, 30.6 + 0.55 * np.sin(2 * np.pi * (126.82 - LON0) / 6.2)
            + 0.22, "shelf-break jet axis", ha="right", va="bottom",
            fontsize=7.0, color="#a8481f", zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.85))

    # ---- frame, graticule, scale bar --------------------------------------
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect(ASPECT)
    ax.set_xticks(np.arange(118, 127.1, 1.0))
    ax.set_yticks(np.arange(28, 34.1, 1.0))
    ax.set_xticklabels([f"{v:.0f}\u00b0E" for v in np.arange(118, 127.1, 1.0)])
    ax.set_yticklabels([f"{v:.0f}\u00b0N" for v in np.arange(28, 34.1, 1.0)])
    ax.tick_params(length=3, labelsize=7.8)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)
    ax.grid(color="#ffffff", lw=0.4, alpha=0.35, zorder=3)

    bar_km = 150.0
    bw = bar_km / KM_PER_DEG_LON
    bx, by = 126.62 - bw, 28.34
    ax.add_patch(Rectangle((bx - 0.20, by - 0.13), bw + 0.40, 0.50,
                           facecolor="white", edgecolor="#d0d5d8",
                           linewidth=0.5, alpha=0.9, zorder=8))
    ax.plot([bx, bx + bw], [by, by], color="#333333", lw=1.5, zorder=9,
            solid_capstyle="butt")
    for xx in (bx, bx + bw / 2, bx + bw):
        ax.plot([xx, xx], [by, by + 0.075], color="#333333", lw=0.9, zorder=9)
    ax.text(bx + bw / 2, by + 0.105, f"{bar_km:.0f} km", ha="center",
            va="bottom", fontsize=7.0, color="#333333", zorder=9)

    ax.annotate("N", xy=(118.45, 28.95), xytext=(118.45, 28.42),
                ha="center", va="center", fontsize=7.6, fontweight="bold",
                color="#3a3a3a", zorder=9,
                arrowprops=dict(arrowstyle="-|>", color="#3a3a3a", lw=1.0,
                                mutation_scale=8))

    cb = fig.colorbar(cf, cax=cax, ticks=np.arange(0, 1.51, 0.25))
    cb.set_label("Surface current speed (m s$^{-1}$)", fontsize=8.0,
                 labelpad=4)
    cb.ax.tick_params(labelsize=7.4, width=0.7, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cax.text(0.5, -0.055, "streamline width\nscales with speed",
             transform=cax.transAxes, ha="center", va="top", fontsize=6.8,
             color="#6b6b6b", linespacing=1.4)

    savefig(fig, "216_ocean_currents.png")


if __name__ == "__main__":
    main()
