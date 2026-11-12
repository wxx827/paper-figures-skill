"""Daylight factor across a deep floor plate, with depth decay and sun hours.

The main panel is the compliance claim: a daylight factor field computed from
the glazed facades and the central atrium, with the 2 % contour drawn heavy so
the reader can see exactly which part of the plate passes and which does not.
The compliant fraction quoted in the title is counted on the same grid that is
being contoured.  The right-hand panels are the supporting evidence: how fast
the factor decays with depth into the plate along three sections, and how the
annual sunlight exposure is distributed over the same footprint.
"""
import textwrap
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import add_scale_bar, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"
WALL = "#3a3a3a"
GLASS = "#3d6b8e"

PLATE_X, PLATE_Y = 24.0, 20.0        # floor plate, metres
CORE = (9.0, 8.0, 6.0, 4.0)          # service core: x, y, w, h
ATRIUM = (17.0, 8.5, 3.2, 4.5)       # daylit void
DF_LIMIT = 2.0                       # per cent, compliance threshold
LEVELS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0]
SEQ = get_cmap("cb_sequential_blue")
SUN = get_cmap("cb_sequential_warm")


def rasterize(artist):
    """3.10 mis-reports ContourSet rasterisation as ignored; it is not."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def rect_distance(x, y, rect):
    rx, ry, rw, rh = rect
    dx = np.maximum(np.maximum(rx - x, x - (rx + rw)), 0.0)
    dy = np.maximum(np.maximum(ry - y, y - (ry + rh)), 0.0)
    return np.hypot(dx, dy)


def inside(x, y, rect):
    rx, ry, rw, rh = rect
    return (x >= rx) & (x <= rx + rw) & (y >= ry) & (y <= ry + rh)


def daylight_field(n=260, seed=31):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, PLATE_X, n)
    y = np.linspace(0.0, PLATE_Y, int(n * PLATE_Y / PLATE_X))
    X, Y = np.meshgrid(x, y)

    # Mullion modulation: glazing comes in 3.0 m modules, so the factor dips
    # opposite every frame rather than being a smooth function of depth alone.
    mod = 0.80 + 0.20 * np.cos(2 * np.pi * X / 3.0) ** 2
    south = 10.2 * np.exp(-Y / 2.9) * mod
    north = 8.4 * np.exp(-(PLATE_Y - Y) / 2.9) * mod
    void = 6.4 * np.exp(-rect_distance(X, Y, ATRIUM) / 2.1)
    # The core casts an obstruction shadow into the plate behind it.
    shade = 1.0 - 0.42 * np.exp(-rect_distance(X, Y, CORE) / 2.6)

    df = (south + north + void) * shade
    df = df * (1.0 + rng.normal(0.0, 0.006, df.shape))
    df = np.clip(df, 0.02, None)
    df[inside(X, Y, CORE) | inside(X, Y, ATRIUM)] = np.nan
    return x, y, df


def sun_hours(nx=16, ny=13):
    x = np.linspace(0.6, PLATE_X - 0.6, nx)
    y = np.linspace(0.6, PLATE_Y - 0.6, ny)
    X, Y = np.meshgrid(x, y)
    hrs = (1420.0 * np.exp(-Y / 3.4) + 540.0 * np.exp(-(PLATE_Y - Y) / 3.0)
           + 380.0 * np.exp(-rect_distance(X, Y, ATRIUM) / 2.4))
    hrs[inside(X, Y, CORE) | inside(X, Y, ATRIUM)] = np.nan
    return x, y, hrs


def plan_panel(fig, ax, x, y, df):
    cs = ax.contourf(x, y, df, levels=LEVELS, cmap=SEQ, extend="max")
    rasterize(cs)
    cs.set_edgecolor("face")
    ax.contour(x, y, df, levels=LEVELS[1:-1], colors=["#ffffff"],
               linewidths=0.5, alpha=0.8)
    ax.contour(x, y, df, levels=[DF_LIMIT], colors=[ACCENT], linewidths=1.6)
    ax.plot([], [], color=ACCENT, lw=1.6,
            label=f"{DF_LIMIT:.0f} % daylight factor contour")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.01), frameon=False,
              fontsize=6.8, handlelength=1.6, borderaxespad=0.0)

    for rect, label, hatch in ((CORE, "Core", "////"),
                               (ATRIUM, "Atrium", "\\\\\\\\")):
        rx, ry, rw, rh = rect
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor="#f0f0f0",
                               edgecolor=WALL, linewidth=0.8, hatch=hatch,
                               zorder=4))
        ax.text(rx + rw / 2, ry + rh / 2, label, ha="center", va="center",
                fontsize=6.6, color=SUB, zorder=5,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="none", alpha=0.92))

    ax.add_patch(Rectangle((0.0, 0.0), PLATE_X, PLATE_Y, facecolor="none",
                           edgecolor=WALL, linewidth=1.1, zorder=4))
    for y0 in (0.0, PLATE_Y):
        for x0 in np.arange(0.6, PLATE_X - 0.5, 3.0):
            ax.plot([x0, x0 + 2.2], [y0, y0], color=GLASS, lw=2.2,
                    solid_capstyle="butt", zorder=5)

    ax.annotate("", xy=(22.6, 23.4), xytext=(22.6, 21.2),
                arrowprops=dict(arrowstyle="-|>", color=WALL, lw=1.0,
                                mutation_scale=9, shrinkA=0, shrinkB=0))
    ax.text(23.2, 23.2, "N", ha="left", va="top", fontsize=7.4, color=INK,
            fontweight="bold")
    ax.text(-1.1, PLATE_Y / 2.0, "plate depth", rotation=90, ha="center",
            va="center", fontsize=6.8, color=MUTED)
    ax.text(PLATE_X / 2.0, -1.4, "glazed south facade", ha="center",
            va="top", fontsize=6.8, color=GLASS)
    ax.text(PLATE_X / 2.0, PLATE_Y + 1.2, "glazed north facade", ha="center",
            va="bottom", fontsize=6.8, color=GLASS)
    add_scale_bar(ax, 5.0, "5 m", loc=(0.020, 0.020), height=0.016, n_seg=2,
                  color=WALL, alt="white", text_color=INK, halo="white",
                  fontsize=6.4)

    ax.set_xlim(-2.4, PLATE_X + 1.0)
    ax.set_ylim(-4.6, PLATE_Y + 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    return cs


def decay_panel(ax, x, y, df, sections):
    for (xs, name, color) in sections:
        col = int(np.argmin(np.abs(x - xs)))
        prof = df[:, col]
        ok = np.isfinite(prof)
        ax.plot(y[ok], prof[ok], lw=1.5, color=color, label=name)
    ax.axhline(DF_LIMIT, color=ACCENT, lw=0.9, ls=(0, (3.0, 2.2)))
    ax.text(0.4, DF_LIMIT * 0.86, f"{DF_LIMIT:.0f} % threshold", ha="left",
            va="top", fontsize=6.4, color=ACCENT)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 34.0)
    ax.set_xlim(0.0, PLATE_Y)
    ax.set_xlabel("distance from south facade (m)", fontsize=7.4)
    ax.set_ylabel("daylight factor (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="lower center", ncol=3,
              handlelength=1.3, columnspacing=0.9, borderaxespad=0.15)
    ax.set_title("Depth decay along three sections", fontsize=8.2,
                 fontweight="bold", color=INK)


def sun_panel(fig, ax, sx, sy, hrs):
    im = ax.imshow(hrs, origin="lower", cmap=SUN, interpolation="nearest",
                   extent=[0.0, PLATE_X, 0.0, PLATE_Y])
    for rect, hatch in ((CORE, "////"), (ATRIUM, "\\\\\\\\")):
        rx, ry, rw, rh = rect
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor="#f0f0f0",
                               edgecolor=WALL, linewidth=0.6, hatch=hatch,
                               zorder=3))
    ax.set_xlim(0.0, PLATE_X)
    ax.set_ylim(0.0, PLATE_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("plate width (m)", fontsize=7.4)
    ax.set_ylabel("plate depth (m)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Annual sunlight exposure grid", fontsize=8.2,
                 fontweight="bold", color=INK)
    return im


def main():
    set_pub_style()
    x, y, df = daylight_field()
    sx, sy, hrs = sun_hours()

    valid = np.isfinite(df)
    compliant = float(np.count_nonzero(df[valid] >= DF_LIMIT)) / valid.sum()
    net_area = PLATE_X * PLATE_Y - CORE[2] * CORE[3] - ATRIUM[2] * ATRIUM[3]
    median_df = float(np.nanmedian(df))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.40, hspace=0.62)
    ax_plan = fig.add_subplot(gs[:, 0])
    ax_dec = fig.add_subplot(gs[0, 1])
    ax_sun = fig.add_subplot(gs[1, 1])

    cs = plan_panel(fig, ax_plan, x, y, df)
    decay_panel(ax_dec, x, y, df, [(1.5, "window module", "#3b5b92"),
                                   (0.75, "mullion", "#8e6bbf"),
                                   (16.0, "beside atrium", "#2e7d5b")])
    im = sun_panel(fig, ax_sun, sx, sy, hrs)

    ax_plan.set_title(f"{compliant:.0%} of the net floor area meets the "
                      f"{DF_LIMIT:.0f} % daylight factor", fontsize=9.0,
                      fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0.040, 1, 0.945], h_pad=2.6, w_pad=2.6)
    ax_plan.apply_aspect()
    safe_colorbar(fig, cs, ax_plan, label="daylight factor (%)", width=0.11,
                  pad=0.08, shrink=0.70, labelsize=6.6, ticks=LEVELS[::2])
    safe_colorbar(fig, im, ax_sun, label="sunlight hours (h/a)", width=0.09,
                  pad=0.07, labelsize=6.5)
    panel_labels_inches(fig, [ax_plan, ax_dec, ax_sun], dx=-0.40, dy=0.12)
    savefig(fig, "590_bim_daylight_analysis.png")


if __name__ == "__main__":
    main()
