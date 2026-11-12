"""Finding the upwind source: a bivariate polar plot and two roses.

The main panel is the bivariate polar plot an air quality analyst reaches for
first.  Every hour of a year is dropped into a wind-direction by wind-speed
cell and the cell is coloured by the mean fine-particle concentration measured
in it, so the plot separates two things a plain rose confuses: a local source
that only shows up in light winds sits near the centre, while a distant source
that needs transport to arrive shows up as a lobe out at four to seven metres
per second.

The companion panels split the same hours the two conventional ways.  The
frequency rose says how often the wind blows from each sector, which is what
decides how much exposure a sector can possibly cause, and the concentration
rose says how dirty the air is when it does - together with the conditional
probability of being in the dirtiest quartile, which is the statistic that
actually points at a source.

Illustrative synthetic year of hourly monitoring data at one urban site.
"""
import textwrap

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, style_axes, get_cmap, add_footer
from figkit import axes_inches, panel_label_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
RED = "#b23a48"
GOLD = "#c9a227"
BROWN = "#8c5a3c"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.62, 0.95, 5.00, 4.32)
AUX_T_RECT = (7.52, 3.40, 2.90, 1.80)
AUX_B_RECT = (7.52, 0.98, 2.90, 1.80)

N_HOUR = 8760
N_SECT = 16                 # sectors used by both roses
N_POLAR = 24                # finer sectors for the bivariate plot
U_EDGES = np.array([0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 11.0, 13.0])
CALM_U = 0.5                # m/s, below this the vane is unreliable
SRC_BEAR = 240.0            # deg, bearing of the industrial source
SPEED_CLASS = np.array([0.5, 2.0, 4.0, 6.0, 9.0, 30.0])
SPEED_COL = ["#cfe1ee", "#9cc3dd", "#5f9dc6", "#3b7aa8", "#1f4e79"]


def synth_hours(rng):
    """Hourly wind vector and fine-particle concentration for one year."""
    pick = rng.choice(3, N_HOUR, p=[0.44, 0.28, 0.28])
    mu = np.deg2rad(np.array([315.0, 225.0, 90.0]))[pick]
    kappa = np.array([3.2, 4.5, 1.1])[pick]
    wd = np.rad2deg(rng.vonmises(mu, kappa)) % 360.0
    align = np.cos(np.deg2rad(wd - 315.0))
    ws = rng.weibull(2.05, N_HOUR) * (4.6 + 1.5 * align)
    ws = np.clip(ws, 0.05, 16.0)

    d = np.abs((wd - SRC_BEAR + 180.0) % 360.0 - 180.0)
    plume = 105.0 * np.exp(-0.5 * (d / 19.0) ** 2) \
        * np.exp(-0.5 * ((ws - 5.0) / 2.3) ** 2)
    d2 = np.abs((wd - 120.0 + 180.0) % 360.0 - 180.0)
    regional = 34.0 * np.exp(-0.5 * (d2 / 55.0) ** 2) / (1.0 + 0.22 * ws)
    stagnation = 26.0 * np.exp(-ws / 1.8)
    base = 9.0 + rng.gamma(2.0, 2.4, N_HOUR)
    pm = base + plume + regional + stagnation
    return wd, ws, pm


def sector_bins(wd, n):
    """Sector index with the first sector centred on north."""
    w = 360.0 / n
    return (np.floor((wd + w / 2.0) % 360.0 / w)).astype(int) % n


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1010)
    wd, ws, pm = synth_hours(rng)
    calm = ws < CALM_U
    ok = ~calm

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT, projection="polar")
    ax_r = axes_inches(fig, AUX_T_RECT, projection="polar")
    ax_d = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_r):
        a.set_theta_zero_location("N")
        a.set_theta_direction(-1)
    ax_d.set_axisbelow(True)
    style_axes(ax_d)

    # ---- main: mean concentration by direction and speed --------------------
    si = sector_bins(wd[ok], N_POLAR)
    ui = np.clip(np.digitize(ws[ok], U_EDGES) - 1, 0, len(U_EDGES) - 2)
    grid = np.full((len(U_EDGES) - 1, N_POLAR), np.nan)
    cnt = np.zeros_like(grid)
    for i in range(len(U_EDGES) - 1):
        for j in range(N_POLAR):
            m = (ui == i) & (si == j)
            cnt[i, j] = m.sum()
            if m.sum() >= 8:
                grid[i, j] = pm[ok][m].mean()
    k = np.unravel_index(np.nanargmax(grid), grid.shape)
    hot_u = 0.5 * (U_EDGES[k[0]] + U_EDGES[k[0] + 1])
    hot_d = k[1] * 360.0 / N_POLAR
    hot_c = float(grid[k])

    w = 2.0 * np.pi / N_POLAR
    th_edge = np.linspace(-w / 2.0, 2.0 * np.pi - w / 2.0, N_POLAR + 1)
    im = ax.pcolormesh(th_edge, U_EDGES, np.ma.masked_invalid(grid),
                       cmap=get_cmap("cb_sequential_warm"), shading="flat",
                       zorder=3)
    ax.plot(np.full(2, np.deg2rad(SRC_BEAR)), [0.0, U_EDGES[-1]], color=RED,
            lw=1.1, ls=(0, (4.0, 2.4)), zorder=6)
    ax.plot([np.deg2rad(hot_d)], [hot_u], marker="o", ms=5.6, mfc="none",
            mec=INK, mew=1.2, zorder=7)
    ax.set_rlim(0.0, U_EDGES[-1])
    ax.set_rticks([3.0, 6.0, 9.0, 12.0])
    ax.set_rlabel_position(112.0)
    ax.set_yticklabels(["3", "6", "9", "12 m/s"], fontsize=6.4, color=SUB)
    ax.set_xticks(np.deg2rad(np.arange(0.0, 360.0, 45.0)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                       fontsize=7.4)
    ax.tick_params(pad=1.0)
    ax.grid(color="#ffffff", lw=0.6, alpha=0.75)
    ax.set_title("Mean $PM_{2.5}$ by wind direction and speed, "
                 f"{int(ok.sum())} valid hours", fontsize=9.0,
                 fontweight="bold", pad=13.0)
    ax.text(1.02, 0.015,
            f"hottest cell: wind from {hot_d:.0f} deg\nat about "
            f"{hot_u:.1f} m/s, mean {hot_c:.0f} $\\mu g/m^3$\nthe dashed "
            f"radius is the source\nbearing at {SRC_BEAR:.0f} deg",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            color=INK, linespacing=1.50, zorder=9,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.92))
    safe_colorbar(fig, im, ax, label="mean $PM_{2.5}$ ($\\mu g/m^3$)",
                  width=0.13, pad=0.24, shrink=0.72, labelsize=7.0)

    # ---- aux (b): frequency rose --------------------------------------------
    sj = sector_bins(wd, N_SECT)
    wsec = 2.0 * np.pi / N_SECT
    theta = np.arange(N_SECT) * wsec
    bottom = np.zeros(N_SECT)
    for c in range(len(SPEED_CLASS) - 1):
        m = (ws >= SPEED_CLASS[c]) & (ws < SPEED_CLASS[c + 1])
        frac = np.array([100.0 * ((sj == j) & m).sum() / N_HOUR
                         for j in range(N_SECT)])
        ax_r.bar(theta, frac, width=wsec * 0.90, bottom=bottom,
                 color=SPEED_COL[c], edgecolor="white", linewidth=0.35,
                 zorder=4)
        bottom += frac
    ax_r.set_xticks(np.deg2rad(np.arange(0.0, 360.0, 90.0)))
    ax_r.set_xticklabels(["N", "E", "S", "W"], fontsize=6.8)
    ax_r.set_rlim(0.0, bottom.max() * 1.12)
    ax_r.set_rticks([4.0, 8.0, 12.0])
    ax_r.set_yticklabels([])
    ax_r.tick_params(pad=0.5)
    ax_r.grid(color=GRID, lw=0.5, alpha=0.9)
    ax_r.set_title("How often the wind blows", fontsize=8.5,
                   fontweight="bold", pad=6.0)
    for c in range(len(SPEED_CLASS) - 1):
        y = 0.97 - 0.085 * c
        ax_r.add_patch(Rectangle((1.03, y - 0.045), 0.085, 0.058,
                                 transform=ax_r.transAxes,
                                 facecolor=SPEED_COL[c], edgecolor="white",
                                 linewidth=0.4, clip_on=False, zorder=8))
        hi = SPEED_CLASS[c + 1]
        lab = (f"{SPEED_CLASS[c]:.0f}-{hi:.0f}" if hi < 20
               else f">{SPEED_CLASS[c]:.0f}")
        ax_r.text(1.14, y - 0.016, lab + (" m/s" if c == 0 else ""),
                  transform=ax_r.transAxes, ha="left", va="center",
                  fontsize=6, color=SUB, clip_on=False, zorder=8)
    ax_r.text(0.5, -0.11,
              f"rings at 4, 8 and 12 % of the year; calm hours "
              f"({CALM_U:.1f} m/s or less)\nare "
              f"{100.0 * calm.mean():.1f} % of the year and carry no "
              "direction",
              transform=ax_r.transAxes, ha="center", va="top", fontsize=6.2,
              color=SUB, linespacing=1.45)

    # ---- aux (c): concentration rose and conditional probability ------------
    q75 = float(np.percentile(pm, 75.0))
    mean_sec = np.array([pm[sj == j].mean() for j in range(N_SECT)])
    cpf = np.array([100.0 * ((pm > q75) & (sj == j)).sum()
                    / max((sj == j).sum(), 1) for j in range(N_SECT)])
    j_hot = int(np.argmax(mean_sec))
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW",
             "SW", "WSW", "W", "WNW", "NW", "NNW"]
    cols = [RED if v >= 0.75 * mean_sec.max() else BROWN if
            v >= 0.45 * mean_sec.max() else GREEN for v in mean_sec]
    ax_d.bar(np.arange(N_SECT), mean_sec, width=0.78, color=cols,
             edgecolor="white", linewidth=0.5, zorder=4)
    ax_d.axhline(pm.mean(), color=FAINT, lw=0.9, ls=(0, (3.4, 2.2)), zorder=3)
    ax_d.text(N_SECT - 0.6, pm.mean() + 1.2, f"site mean {pm.mean():.0f}",
              ha="right", va="bottom", fontsize=6.2, color=SUB)
    ax_d.set_xticks(np.arange(0, N_SECT, 2))
    ax_d.set_xticklabels(names[::2], fontsize=6.6)
    ax_d.set_xlim(-0.75, N_SECT - 0.25)
    ax_d.set_ylim(0.0, mean_sec.max() * 1.40)
    ax_d.tick_params(labelsize=6.8)
    ax_d.set_xlabel("wind direction sector (16 sectors)", fontsize=7.5,
                    labelpad=1.5)
    ax_d.set_ylabel("mean $PM_{2.5}$ ($\\mu g/m^3$)", fontsize=7.5)
    ax_d.set_title("How dirty the air is when it does", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_d.grid(axis="y", color=GRID, lw=0.55, alpha=0.75, zorder=0)

    ax_t = ax_d.twinx()
    ax_t.plot(np.arange(N_SECT), cpf, color=BLUE, lw=1.5, marker="o", ms=2.6,
              zorder=6)
    ax_t.set_ylim(0.0, 100.0)
    ax_t.set_ylabel("CPF above the 75th pct. (%)", fontsize=7.5, color=BLUE)
    ax_t.tick_params(labelsize=6.8, labelcolor=BLUE)
    ax_t.spines["top"].set_visible(False)
    ax_t.grid(False)
    ax_d.text(0.030, 0.960,
              f"the {names[j_hot]} sector averages "
              f"{mean_sec[j_hot]:.0f} $\\mu g/m^3$\nand spends "
              f"{cpf[j_hot]:.0f} % of its hours above\nthe {q75:.0f} "
              "$\\mu g/m^3$ upper quartile",
              transform=ax_d.transAxes, ha="left", va="top", fontsize=6.3,
              color=SUB, linespacing=1.45)
    ax_d.legend(handles=[Line2D([], [], color=BLUE, lw=1.5, marker="o",
                                ms=2.6, label="CPF")],
                loc="upper right", bbox_to_anchor=(1.0, 0.76), frameon=False,
                fontsize=6.2, handlelength=1.6, borderaxespad=0.30)

    panel_label_inches(fig, ax, "(a)", dx=-0.48, dy=0.32)
    panel_label_inches(fig, ax_r, "(b)", dx=-0.30, dy=0.16)
    panel_label_inches(fig, ax_d, "(c)", dx=-0.52, dy=0.10)
    foot = add_footer(fig, "Synthetic data, seed=1010")
    check_layout(fig, foot, [ax_d])
    savefig(fig, "1010_aqm_pollution_rose.png")


if __name__ == "__main__":
    main()
