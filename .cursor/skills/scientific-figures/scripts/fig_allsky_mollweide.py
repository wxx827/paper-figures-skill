"""All-sky map in equatorial coordinates on a Mollweide projection.

The background is a synthetic diffuse-emission field built from a thin plus a
thick galactic disc, a longitudinal spiral modulation and a handful of
molecular clouds; it is evaluated in galactic coordinates and mapped back to
equatorial ones with the standard J2000 rotation, which is why the bright band
runs diagonally across the frame.  Point sources are overplotted with marker
area scaled by flux density, and the b = 0 great circle is traced explicitly.
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

SKY_COLORS = ["#f8fafb", "#dfeaf1", "#b7d0de", "#84aac6", "#4f7ea6",
              "#2e5583", "#213c63", "#182a47"]

RA_G = np.radians(192.85948)
DEC_G = np.radians(27.12825)
L_NCP = np.radians(122.93192)


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


CLOUDS = [(0.55, 0.05, 0.09, 1.5), (2.60, -0.10, 0.13, 0.9),
          (4.10, 0.08, 0.10, 1.1), (5.40, -0.22, 0.16, 0.6),
          (1.60, 0.30, 0.12, 0.5), (3.30, -0.35, 0.14, 0.45)]


def emission(l, b):
    i = np.exp(-np.abs(b) / np.radians(3.6))
    i += 0.34 * np.exp(-np.abs(b) / np.radians(15.0))
    i *= 1.0 + 0.45 * np.cos(l) + 0.18 * np.cos(2 * l + 0.7)
    for l0, b0, s, a in CLOUDS:
        dl = (l - l0 + np.pi) % (2 * np.pi) - np.pi
        i += a * np.exp(-(dl ** 2 + (b - b0) ** 2) / (2 * s ** 2))
    return i + 0.055


def screen(ra, dec):
    """Wrap RA to [-pi, pi) and flip it so that RA increases to the left."""
    return -(((ra + np.pi) % (2 * np.pi)) - np.pi), dec


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(37)

    fig = plt.figure(figsize=(10.4, 7.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.255], hspace=0.20,
                          left=0.055, right=0.885, top=0.905, bottom=0.075)
    ax = fig.add_subplot(gs[0], projection="mollweide")
    ax_b = fig.add_subplot(gs[1])

    # ---- diffuse background ------------------------------------------------
    nx, ny = 720, 360
    xe = np.linspace(-np.pi, np.pi, nx + 1)
    ye = np.linspace(-np.pi / 2, np.pi / 2, ny + 1)
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    XC, YC = np.meshgrid(xc, yc)
    L, B = equ2gal(-XC, YC)
    Z = np.log10(emission(L, B))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("sky", SKY_COLORS)
    mesh = ax.pcolormesh(xe, ye, Z, cmap=cmap, shading="flat",
                         vmin=np.percentile(Z, 1.5), vmax=Z.max(),
                         rasterized=True, zorder=1)

    # ---- galactic plane and its +-10 deg envelope --------------------------
    lg = np.linspace(0, 2 * np.pi, 1400)
    for bb, lw, ls, alpha in [(0.0, 1.5, (0, (5, 2.6)), 0.95),
                              (np.radians(10), 0.7, (0, (2, 2.4)), 0.6),
                              (np.radians(-10), 0.7, (0, (2, 2.4)), 0.6)]:
        ra, dec = gal2equ(lg, np.full_like(lg, bb))
        sx, sy = screen(ra, dec)
        cut = np.abs(np.diff(sx)) > np.pi
        sx = np.insert(sx.astype(float), np.where(cut)[0] + 1, np.nan)
        sy = np.insert(sy.astype(float), np.where(cut)[0] + 1, np.nan)
        ax.plot(sx, sy, color=GOLD, lw=lw, ls=ls, alpha=alpha, zorder=4)

    # galactic centre / anticentre / poles
    marks = [(0.0, 0.0, "GC", "o", 7.0), (np.pi, 0.0, "anti-centre", "o", 5.0),
             (0.0, np.pi / 2, "NGP", "+", 7.0),
             (0.0, -np.pi / 2, "SGP", "+", 7.0)]
    for l0, b0, name, mk, ms in marks:
        ra, dec = gal2equ(np.array([l0]), np.array([b0]))
        sx, sy = screen(ra, dec)
        ax.plot(sx, sy, marker=mk, ms=ms, ls="none", mfc="none",
                mec="#f2e2b8" if mk == "o" else GOLD, mew=1.2, zorder=6)
        ax.text(sx[0], sy[0] - 0.075, name, ha="center", va="top",
                fontsize=6.6, color="#2f4257", zorder=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.82))

    # ---- point sources ------------------------------------------------------
    n_plane, n_iso = 240, 150
    lp = rng.uniform(0, 2 * np.pi, n_plane)
    bp = rng.laplace(0.0, np.radians(7.0), n_plane)
    li = rng.uniform(0, 2 * np.pi, n_iso)
    bi = np.arcsin(rng.uniform(-1, 1, n_iso))
    ls_, bs = np.concatenate([lp, li]), np.concatenate([bp, bi])
    keep = np.abs(bs) < np.pi / 2 - 0.02
    ls_, bs = ls_[keep], bs[keep]
    flux = 10 ** rng.uniform(np.log10(0.04), np.log10(24.0), ls_.size)
    ra, dec = gal2equ(ls_, bs)
    sx, sy = screen(ra, dec)
    size = 3.0 + 34.0 * (np.log10(flux) - np.log10(0.04)) / \
        (np.log10(24.0) - np.log10(0.04))
    ax.scatter(sx, sy, s=size, facecolor="#f0c04a", edgecolor="#4a3512",
               linewidths=0.35, alpha=0.9, zorder=5)

    # ---- graticule and labels ----------------------------------------------
    ax.grid(True, lw=0.45, color="#9fb3c6", alpha=0.75)
    xt = np.arange(-150, 151, 30)
    ax.set_xticks(np.radians(xt))
    ax.set_xticklabels([])
    ax.set_yticks(np.radians(np.arange(-75, 76, 15)))
    ax.set_yticklabels([r"%+d$^\circ$" % v for v in np.arange(-75, 76, 15)])
    ax.tick_params(labelsize=6.6, colors="#3a3a3a")
    # right-ascension labels drawn by hand along the equator so that they
    # always sit above the background mesh
    for v in xt:
        ax.text(np.radians(v), -0.035, "%dh" % (((-v) % 360) / 15),
                ha="center", va="top", fontsize=6.5, color="#33475f",
                zorder=9, bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                    ec="none", alpha=0.80))
    ax.set_title("Equatorial Mollweide projection  |  right ascension "
                 "increases to the left", fontsize=8.8, fontweight="bold",
                 pad=9)
    ax.text(0.5, -0.075, "right ascension  $\\alpha$ (hours)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.2)
    ax.text(-0.055, 0.5, "declination  $\\delta$", transform=ax.transAxes,
            ha="center", va="center", rotation=90, fontsize=8.2)

    # corner legends (the ellipse leaves the box corners free)
    fl = [0.1, 1.0, 10.0]
    handles = [Line2D([], [], marker="o", ls="none",
                      ms=np.sqrt(3.0 + 34.0 * (np.log10(f) - np.log10(0.04)) /
                                 (np.log10(24.0) - np.log10(0.04))),
                      mfc="#f0c04a", mec="#4a3512", mew=0.35,
                      label="%g Jy" % f) for f in fl]
    leg1 = ax.legend(handles=handles, loc="lower left", fontsize=6.8,
                     frameon=False, title="flux density  $S_{1.4}$",
                     title_fontsize=7.0, labelspacing=0.55,
                     bbox_to_anchor=(-0.045, -0.055), handletextpad=0.8)
    ax.add_artist(leg1)
    h2 = [Line2D([], [], color=GOLD, lw=1.5, ls=(0, (5, 2.6)),
                 label="galactic plane  $b=0^\\circ$"),
          Line2D([], [], color=GOLD, lw=0.7, ls=(0, (2, 2.4)),
                 label="$b=\\pm10^\\circ$"),
          Line2D([], [], color="#9fb3c6", lw=0.6,
                 label="equatorial graticule  ($30^\\circ$)")]
    leg2 = ax.legend(handles=h2, loc="lower right", fontsize=6.8,
                     frameon=False, bbox_to_anchor=(1.055, -0.055),
                     labelspacing=0.5)
    for lg_ in (leg1, leg2):
        lg_.set_clip_on(False)
        for t in lg_.get_texts():
            t.set_color("#33475f")

    # ---- colourbar ----------------------------------------------------------
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([0.900, pos.y0 + 0.16 * pos.height, 0.016,
                        0.66 * pos.height])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("diffuse brightness  (log$_{10}$, arb. units)",
                 fontsize=7.2, labelpad=4)
    cb.ax.tick_params(labelsize=6.6, width=0.6, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # ---- demoted strip: latitude concentration ------------------------------
    ax_b.set_facecolor("#f4f5f7")
    edges = np.linspace(-90, 90, 37)
    cnt, _ = np.histogram(np.degrees(bs), bins=edges)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    ax_b.bar(ctr, cnt, width=4.6, color=NAVY, alpha=0.55, lw=0.5,
             edgecolor=NAVY)
    iso = np.cos(np.radians(ctr))
    iso = iso / iso.sum() * cnt.sum()
    ax_b.plot(ctr, iso, color=RUST, lw=1.2, ls=(0, (4, 2.4)),
              label="isotropic expectation  $\\propto\\cos b$")
    ax_b.axvspan(-10, 10, color=GOLD, alpha=0.16, lw=0, zorder=0)
    top = cnt.max() * 1.22
    ax_b.set_ylim(0, top)
    ax_b.annotate("$|b|<10^\\circ$ holds\n%.0f%% of the sample"
                  % (100 * np.mean(np.abs(np.degrees(bs)) < 10)),
                  xy=(10.5, top * 0.46), xytext=(30, top * 0.62),
                  fontsize=6.8, color="#8a6b1f", ha="left", va="center",
                  linespacing=1.4,
                  arrowprops=dict(arrowstyle="->", lw=0.7, color="#8a6b1f"))
    ax_b.set_xlim(-90, 90)
    ax_b.set_xticks(np.arange(-90, 91, 30))
    ax_b.set_xlabel("galactic latitude  $b$ (deg)", fontsize=7.6, labelpad=1.5)
    ax_b.set_ylabel("sources / bin", fontsize=7.4, labelpad=2)
    ax_b.tick_params(labelsize=6.6, length=2.2)
    ax_b.grid(True, axis="y", lw=0.4, color="#e3e5e8")
    ax_b.set_axisbelow(True)
    for s in ax_b.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_b.legend(loc="upper right", fontsize=6.6, frameon=False)
    ax_b.set_title("Supporting statistic  |  the %d plotted sources cluster "
                   "towards the galactic plane" % ls_.size, fontsize=7.2,
                   style="italic", color="#555555", pad=2.5)

    add_panel_label(ax, "(a)", x=-0.045, y=1.015, size=10)
    add_panel_label(ax_b, "(b)", x=-0.045, y=1.055, size=10)

    savefig(fig, "235_allsky_mollweide.png")


if __name__ == "__main__":
    main()
