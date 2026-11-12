"""Layered conceptual/mechanism diagram drawn entirely with patches.

Atmosphere - canopy - organic soil - mineral soil - bedrock are stacked bands;
fluxes between them are ``FancyArrow`` patches whose shaft width encodes the
magnitude of the flux. A side column carries quantitative bars, the pool
budget and an arrow-width key that is rescaled to match the main panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import (FancyArrow, FancyBboxPatch, Circle, Rectangle)
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

INK = "#22303f"
MUTED = "#6b7784"
GRID = "#dcd7cd"
DOWN = "#2e7d7b"     # downward / input flux
UP = "#b0563d"       # upward / release flux
SIDE = "#cf9f43"     # lateral transport
CANOPY = "#84a07c"
DEEP = "#26414f"
WOOD = "#8a7358"

SKY = LinearSegmentedColormap.from_list("sky", ["#c6d8e4", "#eef4f8"])
LBOX = dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none",
            alpha=0.74)

# name, x, direction, y_top, y_bottom, value, unit, colour
FLUXES = [
    ("Precipitation", 13, +1, 80, 56, 812, "mm yr$^{-1}$", DOWN),
    ("GPP", 36, +1, 80, 56, 1480, "g C m$^{-2}$", DOWN),
    ("$R_\\mathrm{eco}$", 59, -1, 80, 56, 1010, "g C m$^{-2}$", UP),
    ("Evapotranspiration", 83, -1, 80, 56, 640, "mm yr$^{-1}$", UP),
    ("Litterfall", 25, +1, 50, 33, 340, "g C m$^{-2}$", DOWN),
    ("Infiltration", 52, +1, 50, 33, 470, "mm yr$^{-1}$", DOWN),
    ("Soil CO$_2$ efflux", 79, -1, 50, 33, 620, "g C m$^{-2}$", UP),
    ("Percolation", 30, +1, 28, 11, 210, "mm yr$^{-1}$", DOWN),
    ("DOC leaching", 62, +1, 28, 11, 24, "g C m$^{-2}$", DOWN),
]
VMAX = 1480.0


def shaft_width(v):
    return 0.65 + 3.3 * np.sqrt(v / VMAX)


def square_units(fig, ax, ymax=100.0):
    fig.canvas.draw()
    bb = ax.get_position()
    w, h = fig.get_size_inches()
    xmax = ymax * (bb.width * w) / (bb.height * h)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)
    return xmax


def unit_in(fig, ax, span):
    return ax.get_position().height * fig.get_size_inches()[1] / span


def band(ax, xmax, y0, y1, fc, label, sub, tcol=INK):
    ax.add_patch(Rectangle((0, y0), xmax, y1 - y0, facecolor=fc,
                           edgecolor="none", zorder=1))
    ax.plot([0, xmax], [y1, y1], color="white", lw=1.2, zorder=6)
    ax.text(1.8, y1 - 3.0, label, fontsize=7.8, fontweight="bold", color=tcol,
            va="top", ha="left", zorder=9)
    ax.text(1.8, y1 - 7.2, sub, fontsize=6.2, color=tcol, alpha=0.8,
            va="top", ha="left", zorder=9)


def tree(ax, x, base, h, rng):
    ax.add_patch(Rectangle((x - 0.65, base), 1.3, 0.45 * h, facecolor=WOOD,
                           edgecolor="none", zorder=4))
    for dx, dy, r in [(-2.6, 0.55, 0.28), (2.6, 0.53, 0.26), (0, 0.78, 0.34)]:
        ax.add_patch(Circle((x + dx, base + dy * h), r * h, facecolor=CANOPY,
                            edgecolor="#5f7b58", lw=0.6, alpha=0.95, zorder=5))
    for k in range(3):
        t = np.linspace(0, 1, 30)
        xr = x + (k - 1) * 3.0 * t + rng.normal(0, 0.1, 30).cumsum() * 0.22
        ax.plot(xr, base - 11 * t ** 1.15, color=WOOD, lw=0.9 - 0.2 * k,
                alpha=0.7, zorder=3)


def cloud(ax, x, y, s):
    for dx, dy, r in [(-1.5, 0, 1.0), (0, 0.35, 1.35), (1.5, 0, 0.95),
                      (0.6, -0.12, 1.1)]:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s, facecolor="white",
                            edgecolor="none", alpha=0.85, zorder=2))


def sun(ax, x, y, r):
    for a in np.linspace(0, 2 * np.pi, 13)[:-1]:
        ax.plot([x + 1.35 * r * np.cos(a), x + 1.95 * r * np.cos(a)],
                [y + 1.35 * r * np.sin(a), y + 1.95 * r * np.sin(a)],
                color="#e0b055", lw=1.0, zorder=2)
    ax.add_patch(Circle((x, y), r, facecolor="#f0cd7e", edgecolor="#dcae4d",
                        lw=0.8, zorder=3))


def diagram(fig, ax, rng):
    xmax = square_units(fig, ax)
    ax.axis("off")
    ax.imshow(np.linspace(0, 1, 128).reshape(-1, 1), extent=[0, xmax, 62, 100],
              aspect="auto", cmap=SKY, zorder=0, interpolation="bilinear")

    sun(ax, xmax - 10, 90, 2.8)
    cloud(ax, 16, 90, 2.0)
    cloud(ax, 45, 94, 1.5)
    ax.text(1.8, 97.0, "ATMOSPHERE", fontsize=7.8, fontweight="bold",
            color=DEEP, va="top", zorder=9)
    ax.text(1.8, 92.8, "well-mixed boundary layer", fontsize=6.2, color=DEEP,
            alpha=0.8, va="top", zorder=9)

    band(ax, xmax, 40, 62, "#eaf0e3", "CANOPY", "LAI 4.6  |  height 18 m")
    band(ax, xmax, 20, 40, "#d7c6a4", "ORGANIC SOIL",
         "0-30 cm  |  C 6.2 kg m$^{-2}$")
    band(ax, xmax, 5, 20, "#b39c78", "MINERAL SOIL",
         "30-120 cm  |  C 9.8 kg m$^{-2}$")
    ax.add_patch(Rectangle((0, 0), xmax, 5, facecolor="#8f8a80",
                           edgecolor="none", zorder=1))
    ax.plot([0, xmax], [5, 5], color="white", lw=1.2, zorder=6)
    ax.text(1.8, 2.5, "BEDROCK", fontsize=6.6, fontweight="bold",
            color="white", va="center", zorder=9)

    for x in np.linspace(3, xmax - 3, 40):
        ax.plot([x, x], [40, 40 + rng.uniform(1.0, 2.6)], color="#7f9c6e",
                lw=0.7, alpha=0.8, zorder=4)
    dx = rng.uniform(0, xmax, 260)
    dy = rng.uniform(6, 39, 260)
    ax.scatter(dx, dy, s=1.4, color=WOOD, alpha=0.30, linewidths=0, zorder=2)

    for x, h in [(14, 13.0), (41, 16.0), (68, 12.0), (97, 14.5)]:
        tree(ax, x, 40, h, rng)

    for name, x, d, ytop, ybot, v, unit, col in FLUXES:
        w = shaft_width(v)
        y0, dy_ = (ytop, ybot - ytop) if d > 0 else (ybot, ytop - ybot)
        ax.add_patch(FancyArrow(x, y0, 0, dy_, width=w, head_width=2.9 * w,
                                head_length=4.0, length_includes_head=True,
                                facecolor=col, edgecolor="white", lw=0.6,
                                zorder=8, alpha=0.96))
        ax.text(x + 1.55 * w + 1.4, (ytop + ybot) / 2,
                f"{name}\n$\\bf{{{v:g}}}$ {unit}", fontsize=6.4, color=INK,
                va="center", ha="left", zorder=10, linespacing=1.45,
                bbox=LBOX)

    wr = shaft_width(130)
    ax.add_patch(FancyArrow(xmax - 27, 25, 23, 0, width=wr,
                            head_width=2.9 * wr, head_length=4.0,
                            length_includes_head=True, facecolor=SIDE,
                            edgecolor="white", lw=0.6, zorder=8))
    ax.text(xmax - 26, 31.5, "Lateral runoff\n$\\bf{130}$ mm yr$^{-1}$",
            fontsize=6.4, color=INK, va="center", ha="left", zorder=10,
            linespacing=1.45, bbox=LBOX)

    ax.set_title("Coupled carbon-water exchange across the "
                 "soil-plant-atmosphere continuum", fontsize=9.5,
                 fontweight="bold", color=INK, pad=6)


def panel_flux_bars(ax):
    rows = [(n, v, c) for n, _, _, _, _, v, _, c in FLUXES]
    rows.append(("Lateral runoff", 130, SIDE))
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.62, color=[r[2] for r in rows],
            edgecolor="white", linewidth=0.7)
    for yi, r in zip(y, rows):
        ax.text(r[1] + 42, yi, f"{r[1]:g}", va="center", fontsize=6.5,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.set_xlim(0, 1880)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_xlabel("Annual flux (g C m$^{-2}$ or mm)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Flux magnitudes", fontsize=8.5, fontweight="bold", color=INK)
    style_axes(ax)


def panel_pools(ax):
    pools = [("Foliage", 0.6, CANOPY), ("Wood", 7.4, "#5f7b58"),
             ("Litter", 1.9, SIDE), ("Organic soil", 6.2, "#b98f4e"),
             ("Mineral soil", 9.8, WOOD)]
    left = 0.0
    for i, (name, v, c) in enumerate(pools):
        ax.barh([0], [v], left=left, height=0.5, color=c, edgecolor="white",
                linewidth=0.8, zorder=3)
        if v > 1.5:
            ax.text(left + v / 2, 0, f"{v}", ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold", zorder=4)
        cx = left + v / 2
        yt, ytx = (-0.36, -0.42) if i % 2 == 0 else (-0.66, -0.72)
        ax.plot([cx, cx], [-0.27, yt], color="#a8a196", lw=0.6, zorder=2)
        ax.text(max(cx, 0.15), ytx, name, ha="left" if cx < 1.5 else "center",
                va="top", fontsize=6.0, color=INK)
        left += v
    ax.set_xlim(0, left * 1.01)
    ax.set_ylim(-1.12, 0.45)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel("Cumulative carbon stock (kg C m$^{-2}$)")
    ax.set_title("Ecosystem carbon budget", fontsize=8.5, fontweight="bold",
                 color=INK, pad=4)
    ax.spines["left"].set_visible(False)
    style_axes(ax)


def panel_key(fig, ax, ax_main):
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    scale = unit_in(fig, ax_main, 100.0) / unit_in(fig, ax, 10.0)
    ax.add_patch(FancyBboxPatch((0.05, 0.25), 9.9, 9.4,
                                boxstyle="round,pad=0,rounding_size=0.45",
                                facecolor="#f6f4ef", edgecolor="none"))
    ax.text(0.7, 9.05, "Shaft thickness $\\propto$ $\\sqrt{\\mathrm{flux}}$",
            fontsize=7, fontweight="bold", color=INK, va="center")
    top = 8.20
    for v in [1480, 500, 100]:
        t = shaft_width(v) * scale
        y = top - t / 2
        ax.add_patch(Rectangle((0.9, y - t / 2), 2.5, t, facecolor="#9aa7ae",
                               edgecolor="white", lw=0.5))
        ax.text(3.75, y, f"{v:g}", fontsize=6.6, va="center", color=INK)
        top -= t + 0.55
    ax.text(0.7, 1.95, "Direction", fontsize=7, fontweight="bold", color=INK,
            va="center")
    for i, (c, lab) in enumerate([(DOWN, "input"), (UP, "release"),
                                  (SIDE, "lateral")]):
        x = 0.75 + 3.10 * i
        ax.add_patch(Rectangle((x, 0.72), 0.55, 0.42, facecolor=c,
                               edgecolor="none"))
        ax.text(x + 0.72, 0.93, lab, fontsize=6.4, va="center", color=INK)


def main():
    set_pub_style()
    rng = np.random.default_rng(19)
    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.42, 1.0], left=0.035,
                          right=0.985, top=0.872, bottom=0.058, wspace=0.17)
    ax_main = fig.add_subplot(gs[0, 0])
    right = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1],
                                    height_ratios=[1.70, 0.78, 0.85],
                                    hspace=0.62)
    ax_b = fig.add_subplot(right[0])
    ax_c = fig.add_subplot(right[1])
    ax_d = fig.add_subplot(right[2])

    diagram(fig, ax_main, rng)
    panel_flux_bars(ax_b)
    panel_pools(ax_c)
    panel_key(fig, ax_d, ax_main)

    add_panel_label(ax_main, "(a)", x=-0.012, y=1.015)
    add_panel_label(ax_b, "(b)", x=-0.025, y=1.045)
    add_panel_label(ax_c, "(c)", x=-0.025, y=1.10)
    add_panel_label(ax_d, "(d)", x=-0.025, y=1.00)

    savefig(fig, "89_conceptual_diagram.png")


if __name__ == "__main__":
    main()
