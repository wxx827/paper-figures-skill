"""Survey footprint, exposure depth and the resulting source counts.

One quantity — the number of exposures stacked on a 6 x 6 deg tile — drives
every panel and the single shared colour bar: it paints the footprint in (a),
splits the limiting-magnitude histogram in (b), and separates the cumulative
number counts in (c).  Tiles within 11 deg of the galactic plane are dropped,
which is why the footprint carries a curved bite across it.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label

NAVY = "#1f3355"
INDIGO = "#5b4b8a"
GOLD = "#c9962f"
CYAN = "#3f7f95"
RUST = "#a8523a"
GREY = "#8b9099"

DEPTH_COLORS = ["#cfe0ec", "#9dc0d8", "#5d90b5", "#31608e", "#28356b"]
BOUNDS = [1, 3, 5, 9, 17, 33]
CLASS_LABELS = ["1-2", "3-4", "5-8", "9-16", "17-32"]

RA_G = np.radians(192.85948)
DEC_G = np.radians(27.12825)
L_NCP = np.radians(122.93192)

DEEP = [(42.0, 12.0, 9.0, 26), (196.0, 55.0, 8.0, 21),
        (305.0, -8.0, 8.5, 24)]


def equ2gal(ra, dec):
    sb = (np.sin(dec) * np.sin(DEC_G) +
          np.cos(dec) * np.cos(DEC_G) * np.cos(ra - RA_G))
    b = np.arcsin(np.clip(sb, -1, 1))
    y = np.cos(dec) * np.sin(ra - RA_G)
    x = (np.sin(dec) * np.cos(DEC_G) -
         np.cos(dec) * np.sin(DEC_G) * np.cos(ra - RA_G))
    return (L_NCP - np.arctan2(y, x)) % (2 * np.pi), b


def gal2equ(l, b):
    sd = (np.sin(b) * np.sin(DEC_G) +
          np.cos(b) * np.cos(DEC_G) * np.sin(l - L_NCP))
    dec = np.arcsin(np.clip(sd, -1, 1))
    y = np.cos(b) * np.cos(l - L_NCP)
    x = (np.sin(b) * np.cos(DEC_G) -
         np.cos(b) * np.sin(DEC_G) * np.sin(l - L_NCP))
    return (RA_G + np.arctan2(y, x)) % (2 * np.pi), dec


def build_tiles(rng, step=6.0):
    ra_e = np.arange(0.0, 360.0 + step, step)
    de_e = np.arange(-30.0, 84.0 + step, step)
    rc = 0.5 * (ra_e[1:] + ra_e[:-1])
    dc = 0.5 * (de_e[1:] + de_e[:-1])
    RA, DE = np.meshgrid(rc, dc)

    n = (3.6 + 1.7 * np.sin(np.radians(RA) * 2.0 + 0.4)
         + 1.2 * np.cos(np.radians(DE) * 3.0)
         + 0.9 * np.sin(np.radians(RA + 2 * DE))
         + rng.normal(0, 0.8, RA.shape))
    for ra0, de0, rad, peak in DEEP:
        d = np.hypot((RA - ra0) * np.cos(np.radians(DE)), DE - de0)
        n += (peak - 4) * np.exp(-(d / rad) ** 2)
    n = np.clip(np.round(n), 0, 32)

    _, b = equ2gal(np.radians(RA), np.radians(DE))
    n[np.abs(np.degrees(b)) < 11.0] = 0
    n[(DE < -24.0) & (RA > 120.0) & (RA < 300.0)] = 0
    n[n < 1] = 0
    return ra_e, de_e, RA, DE, np.ma.masked_where(n < 1, n)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(163)

    ra_e, de_e, RA, DE, N = build_tiles(rng)
    flat_n = N.compressed()
    m_lim = 24.15 + 1.25 * np.log10(flat_n) + rng.normal(0, 0.09,
                                                         flat_n.size)

    cmap = mpl.colors.ListedColormap(DEPTH_COLORS)
    cmap.set_bad("#f3f4f6")
    norm = mpl.colors.BoundaryNorm(BOUNDS, cmap.N)

    fig = plt.figure(figsize=(10.4, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], hspace=0.34,
                          wspace=0.20, left=0.062, right=0.888, top=0.872,
                          bottom=0.078)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) footprint ------------------------------------------------------
    ax_a.set_facecolor("#f3f4f6")
    mesh = ax_a.pcolormesh(ra_e, de_e, N, cmap=cmap, norm=norm,
                           edgecolors="white", linewidth=0.18, zorder=2)
    lg = np.linspace(0, 2 * np.pi, 3000)
    for bb, lw, ls in [(0.0, 1.6, (0, (5, 2.6))),
                       (np.radians(11), 0.8, (0, (2, 2.2))),
                       (np.radians(-11), 0.8, (0, (2, 2.2)))]:
        ra, dec = gal2equ(lg, np.full_like(lg, bb))
        x, y = np.degrees(ra), np.degrees(dec)
        cut = np.abs(np.diff(x)) > 180
        x = np.insert(x, np.where(cut)[0] + 1, np.nan)
        y = np.insert(y, np.where(cut)[0] + 1, np.nan)
        ax_a.plot(x, y, color=GOLD, lw=lw, ls=ls, zorder=5)
    for ra0, de0, rad, peak in DEEP:
        ax_a.add_patch(plt.Circle((ra0, de0), rad * 1.15, fc="none",
                                  ec=RUST, lw=1.0, ls=(0, (3, 2.2)),
                                  zorder=6))
        ax_a.text(ra0, de0 + rad * 1.15 + 1.6, "deep field", fontsize=6.6,
                  color=RUST, ha="center", va="bottom", zorder=7,
                  bbox=dict(boxstyle="round,pad=0.18", fc="white",
                            ec="none", alpha=0.85))
    ax_a.text(0.5, 0.055, "galactic plane exclusion  $|b| < 11^\\circ$",
              transform=ax_a.transAxes, ha="center", va="bottom",
              fontsize=6.8, color="#8a6b1f", zorder=8,
              bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#e2d4ae",
                        lw=0.5, alpha=0.94))

    area = flat_n.size * 36.0
    ax_a.text(0.006, 0.962, "%d tiles of $6^\\circ\\times6^\\circ$   "
              "%.0f deg$^2$\nmedian %d exposures,  median $m_{\\rm lim} = "
              "%.2f$ mag" % (flat_n.size, area, np.median(flat_n),
                             np.median(m_lim)), transform=ax_a.transAxes,
              ha="left", va="top", fontsize=6.9, linespacing=1.5, zorder=8,
              bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#cccccc",
                        lw=0.55, alpha=0.94))
    ax_a.set_xlim(360, 0)
    ax_a.set_ylim(-30, 84)
    ax_a.set_xticks(np.arange(0, 361, 30))
    ax_a.set_yticks(np.arange(-30, 85, 15))
    ax_a.set_xlabel("right ascension  $\\alpha$ (deg)   —  increasing to the "
                    "left", fontsize=8.2, labelpad=2)
    ax_a.set_ylabel("declination  $\\delta$ (deg)", fontsize=8.2, labelpad=3)
    ax_a.tick_params(labelsize=7.0, length=2.6)
    for s in ax_a.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#9aa0a6")
    ax_a.set_title("Tiled footprint coloured by the number of stacked "
                   "exposures", fontsize=8.8, fontweight="bold", pad=3)

    # ---- shared colour bar --------------------------------------------------
    p0, p1 = ax_a.get_position(), ax_b.get_position()
    cax = fig.add_axes([0.905, p1.y0 + 0.10 * (p0.y1 - p1.y0), 0.015,
                        0.78 * (p0.y1 - p1.y0)])
    cb = fig.colorbar(mesh, cax=cax, ticks=BOUNDS, spacing="uniform")
    cb.set_label("number of stacked exposures  $N_{\\rm exp}$   "
                 "(shared by all panels)", fontsize=7.4, labelpad=5)
    cb.ax.tick_params(labelsize=6.8, width=0.6, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # ---- (b) limiting-magnitude histogram, stacked by exposure class -------
    edges = np.arange(23.8, 26.35, 0.10)
    groups = [m_lim[(flat_n >= BOUNDS[i]) & (flat_n < BOUNDS[i + 1])]
              for i in range(5)]
    ax_b.hist(groups, bins=edges, stacked=True, color=DEPTH_COLORS,
              edgecolor="white", linewidth=0.25,
              label=["$N_{\\rm exp}$ %s" % s for s in CLASS_LABELS])
    med = float(np.median(m_lim))
    ax_b.axvline(med, color=RUST, lw=1.1, ls=(0, (4, 2.4)), zorder=5)
    ax_b.text(med + 0.04, ax_b.get_ylim()[1] * 0.94, "median %.2f" % med,
              fontsize=6.8, color=RUST, ha="left", va="top")
    ax_b.set_xlim(23.8, 26.3)
    ax_b.set_xlabel("$5\\sigma$ limiting magnitude  $m_{\\rm lim}$ (mag)",
                    fontsize=8.2, labelpad=2)
    ax_b.set_ylabel("tiles / 0.1 mag", fontsize=8.0, labelpad=3)
    ax_b.tick_params(labelsize=7.0, length=2.6)
    ax_b.grid(True, axis="y", lw=0.4, color="#eceef1")
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="upper right", fontsize=6.6, frameon=False,
                labelspacing=0.35, handlelength=1.4)
    ax_b.set_title("Depth distribution of the tiles", fontsize=8.8,
                   fontweight="bold", pad=3)

    # ---- (c) cumulative counts ----------------------------------------------
    m = np.linspace(18.0, 26.6, 500)
    dm = m[1] - m[0]
    diff = 0.55 * np.log(10) * 10 ** (0.55 * (m - 18.0)) * 24.0
    for i, col in enumerate(DEPTH_COLORS):
        if groups[i].size == 0:
            continue
        depth = float(np.median(groups[i]))
        comp = 1.0 / (1 + np.exp((m - depth) / 0.16))
        ax_c.plot(m, np.cumsum(diff * comp) * dm + 20.0, color=col, lw=1.6,
                  zorder=4)
        ax_c.axvline(depth, color=col, lw=0.6, ls=(0, (1.6, 2.0)), zorder=2)
    ax_c.plot(m, np.cumsum(diff) * dm + 20.0, color=GREY, lw=1.0,
              ls=(0, (5, 2.4)), zorder=3)
    ax_c.text(20.0, 10 ** (0.55 * (20.6 - 18.0)) * 24.0,
              "Euclidean slope\n$\\log N \\propto 0.55\\,m$", fontsize=6.8,
              color="#6f767e", ha="left", va="bottom", linespacing=1.4)
    ax_c.annotate("counts saturate at each\ncompleteness limit",
                  xy=(25.85, 3.1e5), xytext=(20.1, 5.2e5), fontsize=6.9,
                  color=NAVY, ha="left", va="center", linespacing=1.4,
                  arrowprops=dict(arrowstyle="->", lw=0.8, color=NAVY,
                                  shrinkB=3))
    ax_c.set_yscale("log")
    ax_c.set_xlim(18.0, 26.6)
    ax_c.set_ylim(20, 1.4e6)
    ax_c.set_xlabel("magnitude  $m$ (mag)", fontsize=8.2, labelpad=2)
    ax_c.set_ylabel("$N(<m)$  (sources deg$^{-2}$)", fontsize=8.0, labelpad=3)
    ax_c.tick_params(labelsize=7.0, length=2.6)
    ax_c.grid(True, lw=0.4, color="#eceef1")
    ax_c.set_axisbelow(True)
    ax_c.set_title("Cumulative source counts per exposure class",
                   fontsize=8.8, fontweight="bold", pad=3)

    for ax_, lab in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")]:
        p = ax_.get_position()
        add_panel_label(ax_, lab, x=-0.030 / p.width,
                        y=1.0 + 0.026 / p.height, size=10)

    handles = [Line2D([], [], color=GOLD, lw=1.6, ls=(0, (5, 2.6)),
                      label="galactic plane  $b = 0^\\circ$"),
               Line2D([], [], color=GOLD, lw=0.8, ls=(0, (2, 2.2)),
                      label="$b = \\pm11^\\circ$ exclusion"),
               Line2D([], [], color=RUST, lw=1.0, ls=(0, (3, 2.2)),
                      label="deep-drilling field"),
               Line2D([], [], color=GREY, lw=1.0, ls=(0, (5, 2.4)),
                      label="uncorrected Euclidean counts"),
               Line2D([], [], marker="s", ls="none", ms=6, mfc="#f3f4f6",
                      mec="#c8ccd1", mew=0.6, label="not observed")]
    savefig(fig, "242_survey_coverage.png")


if __name__ == "__main__":
    main()
