"""Hero / cover-page composite.

A dominant regime map with a zoom callout anchors five supporting panels
(ridgeline, directional rose, effect sizes, site trajectories, driver
decomposition). A dashed ConnectionPatch ties the collapsing site in the
regime map to its trajectory two panels below, so the eye is led through the
argument instead of scanning panel by panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Circle
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

DEEP = "#26414f"
TEAL = "#2e7d7b"
SAGE = "#7d9a72"
SAND = "#cf9f43"
RUST = "#b0563d"
INK = "#22303f"
MUTED = "#6b7784"
GRID = "#dcd7cd"

FIELD = LinearSegmentedColormap.from_list(
    "field", ["#1c5654", "#2e7d7b", "#79ada6", "#c3d6ca", "#f2ece1",
              "#e6bb9c", "#cf8158", "#b0563d", "#7d3527"])

SITES = [("A", 0.26, 0.30, SAGE), ("B", 0.52, 0.46, SAND),
         ("C", 0.79, 0.72, RUST)]


def surface(X, Y):
    return (1 / (1 + np.exp(-6.2 * (1.35 * X + 0.95 * Y - 1.18)))
            + 0.055 * np.sin(7.5 * X) * np.cos(6.0 * Y))


def trajectory(x0, y0, x1, y1, bow=0.10):
    t = np.linspace(0, 1, 90)
    x = x0 + (x1 - x0) * t
    y = y0 + (y1 - y0) * t + bow * np.sin(np.pi * t)
    return x, y


def panel_hero(fig, ax):
    n = 320
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = surface(X, Y)
    im = ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1], cmap=FIELD,
                   aspect="auto", vmin=0, vmax=1.05,
                   interpolation="bilinear")
    ax.contour(X, Y, Z, levels=[0.1, 0.25, 0.75, 0.9], colors="white",
               linewidths=0.5, alpha=0.45)
    ax.contour(X, Y, Z, levels=[0.5], colors="white", linewidths=2.6)
    ax.contour(X, Y, Z, levels=[0.5], colors=INK, linewidths=1.0)

    ax.text(0.085, 0.155, "RESILIENT", fontsize=9.5, fontweight="bold",
            color="#eef4ec")
    ax.text(0.085, 0.098, "recovery within a decade", fontsize=6.8,
            color="#eef4ec", alpha=0.9)
    ax.text(0.985, 0.945, "COLLAPSE", fontsize=9.5, fontweight="bold",
            color="#f7e7de", ha="right")
    ax.text(0.985, 0.895, "no recovery on record", fontsize=6.8,
            color="#f7e7de", ha="right", alpha=0.92)
    ax.annotate("critical boundary\n$p = 0.5$", xy=(0.305, 0.815),
                xytext=(0.048, 0.585), fontsize=6.8, color=INK,
                linespacing=1.4, zorder=9,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="none", alpha=0.82),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))

    for name, sx, sy, c in SITES:
        tx, ty = trajectory(sx - 0.20, sy - 0.20, sx, sy,
                            bow=0.05 if name != "C" else 0.09)
        ax.plot(tx, ty, color="white", lw=3.4, alpha=0.95, zorder=4)
        ax.plot(tx, ty, color=c, lw=1.7, zorder=5)
        ax.annotate("", xy=(tx[-1], ty[-1]), xytext=(tx[-6], ty[-6]),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4),
                    zorder=6)
        ax.add_patch(Circle((sx, sy), 0.023, facecolor="white",
                            edgecolor=c, lw=1.8, zorder=7))
        ax.text(sx, sy, name, ha="center", va="center", fontsize=6.8,
                fontweight="bold", color=c, zorder=8)

    ax.add_patch(FancyBboxPatch((0.018, 0.855), 0.375, 0.125,
                                boxstyle="round,pad=0,rounding_size=0.018",
                                facecolor="white", alpha=0.80,
                                edgecolor="none", zorder=8))
    cax = ax.inset_axes([0.045, 0.878, 0.315, 0.028], zorder=10)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6, length=2, pad=1.5, colors=INK)
    cb.outline.set_linewidth(0.5)
    cax.set_title("Collapse probability", fontsize=6.6, pad=3, color=INK)

    zx0, zx1, zy0, zy1 = 0.60, 0.86, 0.56, 0.82
    axi = ax.inset_axes([0.615, 0.055, 0.345, 0.315])
    axi.imshow(Z, origin="lower", extent=[0, 1, 0, 1], cmap=FIELD,
               aspect="auto", vmin=0, vmax=1.05, interpolation="bilinear")
    axi.contour(X, Y, Z, levels=[0.5], colors="white", linewidths=2.0)
    axi.contour(X, Y, Z, levels=[0.5], colors=INK, linewidths=0.9)
    tx, ty = trajectory(0.79 - 0.20, 0.72 - 0.20, 0.79, 0.72, bow=0.09)
    axi.plot(tx, ty, color="white", lw=4.0, alpha=0.95)
    axi.plot(tx, ty, color=RUST, lw=2.0)
    axi.add_patch(Circle((0.79, 0.72), 0.010, facecolor="white",
                         edgecolor=RUST, lw=1.5, zorder=6))
    axi.set_xlim(zx0, zx1)
    axi.set_ylim(zy0, zy1)
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_color(INK)
        s.set_linewidth(1.0)
    axi.text(0.5, 0.055, "boundary crossing of site C",
             transform=axi.transAxes, ha="center", va="bottom", fontsize=6.3,
             color=INK, zorder=9,
             bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                       edgecolor="none", alpha=0.82))
    ax.indicate_inset_zoom(axi, edgecolor=INK, alpha=0.85, linewidth=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative forcing (normalised)")
    ax.set_ylabel("Intrinsic sensitivity (normalised)")
    ax.set_title("Where systems tip", loc="left", fontsize=10.5,
                 fontweight="bold", color=INK, pad=6)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.8)


def panel_ridge(ax, rng):
    years = [1990, 1998, 2006, 2014, 2022]
    cols = [SAGE, "#a9a95f", SAND, "#c07a41", RUST]
    grid = np.linspace(-3.2, 4.6, 400)
    off = 0.62
    for i, (yr, c) in enumerate(zip(years, cols)):
        mu = -0.55 + 0.42 * i
        sd = 0.62 + 0.10 * i
        d = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / sd
        d = d / d.max() * 0.90
        base = (len(years) - 1 - i) * off
        ax.fill_between(grid, base, base + d, color=c, alpha=0.80, lw=0,
                        zorder=10 - i)
        ax.plot(grid, base + d, color="white", lw=1.0, zorder=10 - i)
        ax.plot([grid[0], grid[-1]], [base, base], color="#c9c2b6", lw=0.6,
                zorder=1)
        ax.text(-3.35, base + 0.10, str(yr), ha="right", va="bottom",
                fontsize=7, color=INK, fontweight="bold")
        ax.scatter([mu], [base + 0.045], marker="|", s=42, color="white",
                   zorder=11 - i)
    ax.axvline(0, color=INK, lw=0.8, ls="--", zorder=20)
    ax.text(0.12, 3.60, "1990 baseline", fontsize=6.5, color=INK, va="top")
    ax.set_xlim(-3.4, 4.6)
    ax.set_ylim(-0.12, 3.72)
    ax.set_yticks([])
    ax.set_xlabel("Anomaly ($\\sigma$)")
    ax.spines["left"].set_visible(False)
    ax.set_title("The whole distribution moved", loc="left", fontsize=8.8,
                 fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def panel_rose(ax, rng):
    k = 12
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    vals = 0.35 + 0.55 * np.exp(-0.5 * ((np.arange(k) - 7.5) / 2.6) ** 2) \
        + rng.uniform(0, 0.08, k)
    cmap = LinearSegmentedColormap.from_list("r", ["#cfd9c9", SAGE, TEAL,
                                                   DEEP])
    ax.bar(ang, vals, width=2 * np.pi / k * 0.86, color=cmap(vals / vals.max()),
           edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(ang[::3])
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=7)
    ax.set_yticks([0.3, 0.6, 0.9])
    ax.set_yticklabels(["", "0.6", "0.9"], fontsize=6.0, color=MUTED)
    ax.set_ylim(0, 1.03)
    ax.grid(color=GRID, lw=0.6)
    ax.spines["polar"].set_color(GRID)
    ax.tick_params(axis="x", pad=-1.5)


def panel_effects(ax):
    names = ["Warming", "Drying", "Fire return", "Fragmentation",
             "Grazing", "Nitrogen"]
    est = np.array([0.62, 0.48, 0.31, 0.19, -0.12, -0.27])
    lo = est - np.array([0.11, 0.09, 0.10, 0.08, 0.09, 0.11])
    hi = est + np.array([0.10, 0.11, 0.09, 0.09, 0.08, 0.10])
    y = np.arange(len(names))[::-1]
    for yi, e, l, h in zip(y, est, lo, hi):
        c = RUST if e > 0 else TEAL
        ax.plot([l, h], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
        ax.plot([l, l], [yi - 0.14, yi + 0.14], color=c, lw=1.0)
        ax.plot([h, h], [yi - 0.14, yi + 0.14], color=c, lw=1.0)
        ax.scatter([e], [yi], s=28, color=c, zorder=5, edgecolor="white",
                   lw=0.8)
    ax.axvline(0, color=INK, lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(-0.45, 0.80)
    ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75])
    ax.set_xlabel("Standardised effect on tipping risk")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("What pushes systems over", loc="left", fontsize=8.8,
                 fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def panel_sites(ax, rng):
    t = np.arange(1990, 2025)
    ends = {}
    for name, _, _, c in SITES:
        if name == "A":
            y = 82 - 0.22 * (t - 1990) + 2.0 * np.sin(0.5 * (t - 1990))
        elif name == "B":
            y = 78 - 0.62 * (t - 1990) + 2.4 * np.sin(0.42 * (t - 1990))
        else:
            y = 76 - 0.55 * (t - 1990) - 26 / (1 + np.exp(-0.55 * (t - 2011))) \
                + 2.2 * np.sin(0.45 * (t - 1990))
        y = y + rng.normal(0, 0.7, t.size)
        ax.fill_between(t, y - 2.4, y + 2.4, color=c, alpha=0.13, lw=0)
        ax.plot(t, y, color=c, lw=1.7, zorder=4)
        ax.scatter([t[-1]], [y[-1]], s=22, color=c, zorder=6,
                   edgecolor="white", lw=0.8)
        ax.text(2025.4, y[-1], f"site {name}", fontsize=7.2, color=c,
                va="center", fontweight="bold")
        ends[name] = (t[0], y[0])
    ax.axvline(2011, color=INK, lw=0.8, ls=":", zorder=2)
    ax.annotate("boundary crossed", xy=(2011, 20), xytext=(2000.5, 16),
                fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                connectionstyle="arc3,rad=-0.22"))
    ax.set_xlim(1990, 2029)
    ax.set_ylim(10, 92)
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_xlabel("Year")
    ax.set_ylabel("Ecosystem state index")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Three sites, three fates", loc="left", fontsize=8.8,
                 fontweight="bold", color=INK, pad=6)
    style_axes(ax)
    return ends


def panel_drivers(ax):
    names = ["Soil moisture deficit", "Heat extremes", "Canopy loss",
             "Seed limitation", "Herbivory", "Nutrient input",
             "Management", "Connectivity"]
    vals = np.array([0.34, 0.27, 0.19, 0.11, -0.06, -0.09, -0.17, -0.24])
    y = np.arange(len(names))[::-1]
    cols = [RUST if v > 0 else TEAL for v in vals]
    ax.barh(y, vals, height=0.60, color=cols, edgecolor="white", linewidth=0.8)
    for yi, v in zip(y, vals):
        ax.text(v + (0.020 if v > 0 else -0.020), yi, f"{v:+.2f}",
                va="center", ha="left" if v > 0 else "right", fontsize=6.6,
                color=INK)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(-0.42, 0.50)
    ax.set_xticks([-0.3, -0.15, 0, 0.15, 0.3, 0.45])
    ax.set_xlabel("Contribution to the 2022 anomaly")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.30, 7.95, "destabilising", fontsize=6.8, color=RUST,
            fontweight="bold", ha="center", va="center")
    ax.text(-0.24, 7.95, "stabilising", fontsize=6.8, color=TEAL,
            fontweight="bold", ha="center", va="center")
    ax.set_ylim(-0.72, 8.42)
    ax.set_title("Driver decomposition", loc="left", fontsize=8.8,
                 fontweight="bold", color=INK, pad=6)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(2026)
    fig = plt.figure(figsize=(13.6, 8.9))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 0.80, 0.95],
                          width_ratios=[1.0, 1.0, 0.86, 1.14],
                          left=0.052, right=0.955, top=0.845, bottom=0.062,
                          hspace=0.50, wspace=0.46)
    ax_a = fig.add_subplot(gs[0:2, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 2], projection="polar")
    ax_d = fig.add_subplot(gs[1, 3])
    ax_e = fig.add_subplot(gs[2, 0:2])
    ax_f = fig.add_subplot(gs[2, 2:4])

    panel_hero(fig, ax_a)
    panel_ridge(ax_b, rng)
    panel_rose(ax_c, rng)
    panel_effects(ax_d)
    ends = panel_sites(ax_e, rng)
    panel_drivers(ax_f)

    cell = ax_c.get_position()
    ax_c.set_position([cell.x0 + 0.004, cell.y0 - 0.004, cell.width * 0.94,
                       cell.height * 0.90])
    fig.text(cell.x0 - 0.030, cell.y1 + 0.012, "Storm approach direction",
             fontsize=8.8, fontweight="bold", color=INK, ha="left",
             va="bottom")
    fig.text(cell.x0 - 0.060, cell.y1 + 0.038, "(c)", fontsize=11,
             fontweight="bold", color="black", ha="left", va="bottom")

    con = ConnectionPatch(
        xyA=(0.79, -0.052), coordsA=ax_a.transAxes,
        xyB=(2011, 91.0), coordsB=ax_e.transData,
        arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color=RUST,
        linestyle=(0, (4, 2.6)), connectionstyle="arc3,rad=0.22",
        zorder=30, alpha=0.95)
    fig.add_artist(con)
    ax_e.text(2013.6, 95.5, "site C, traced through time", fontsize=6.7,
              color=RUST, ha="left", va="bottom", style="italic",
              clip_on=False)

    for ax, lab, x, y in [(ax_a, "(a)", -0.075, 1.020),
                          (ax_b, "(b)", -0.075, 1.045),
                          (ax_d, "(d)", -0.235, 1.045),
                          (ax_e, "(e)", -0.075, 1.045),
                          (ax_f, "(f)", -0.185, 1.045)]:
        add_panel_label(ax, lab, x=x, y=y)

    savefig(fig, "94_hero_figure.png")


if __name__ == "__main__":
    main()
