"""All-sky source catalogue on a Mollweide projection of the equatorial sky.

The main panel is the claim: the diffuse foreground and the catalogue share a
single organising structure, the Galactic plane, which on this projection is a
great circle drawn as a sinusoidal curve rather than a straight line.  Galactic
source classes crowd onto it; extragalactic classes are excluded from it by
the same extinction that makes it bright.  All coordinates go through the
standard J2000 equatorial-to-Galactic rotation, so the plane, the background
band and the labelled regions are all consistent with one another.

Side panels quantify what the map shows: counts against Galactic latitude, and
the magnitude distribution with the bright end at the top as convention
requires.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GREY = "#808080"

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"

FORE_CMAP = get_cmap("cb_sequential_blue")

# Rows are the equatorial components of the Galactic x, y and z axes (J2000),
# so v_gal = R @ v_eq and v_eq = R.T @ v_gal.
R_EQ2GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                     [+0.4941094279, -0.4448296300, +0.7469822445],
                     [-0.8676661490, -0.1980763734, +0.4559837762]])

# (name, RA deg, Dec deg, label offset in deg)
REGIONS = [("Galactic centre", 266.42, -28.99, (0.0, -9.0)),
           ("LMC", 80.89, -69.76, (0.0, 7.5)),
           ("SMC", 13.19, -72.83, (0.0, 7.5)),
           ("Orion", 83.82, -5.39, (0.0, 7.5)),
           ("M31", 10.68, 41.27, (0.0, 7.5)),
           ("Virgo cluster", 187.71, 12.39, (0.0, 7.5)),
           ("Cygnus X", 308.30, 41.32, (0.0, 7.5))]


def wrap180(a):
    return ((np.asarray(a, dtype=float) + 180.0) % 360.0) - 180.0


def sph_to_vec(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                     np.sin(la)])


def vec_to_sph(v):
    lon = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    lat = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return lon, lat


def _rotate(mat, v):
    """Apply a 3x3 rotation to a stack of vectors of any trailing shape."""
    return np.einsum("ij,j...->i...", mat, v)


def eq_to_gal(ra, dec):
    return vec_to_sph(_rotate(R_EQ2GAL, sph_to_vec(ra, dec)))


def gal_to_eq(lon, lat):
    return vec_to_sph(_rotate(R_EQ2GAL.T, sph_to_vec(lon, lat)))


def proj(ra, dec):
    """Map coordinates: RA increases to the left, as the sky is drawn."""
    return np.radians(-wrap180(ra)), np.radians(dec)


def foreground(seed=63, nx=421, ny=211):
    """Smooth Galactic foreground on the plotting grid."""
    rng = np.random.default_rng(seed)
    x_deg = np.linspace(-180.0, 180.0, nx)
    y_deg = np.linspace(-89.0, 89.0, ny)
    xx, yy = np.meshgrid(x_deg, y_deg)
    lon, lat = eq_to_gal((-xx) % 360.0, yy)

    band = np.exp(-np.abs(lat) / 7.2)
    bulge = 0.90 * np.exp(-0.5 * ((wrap180(lon) / 21.0) ** 2
                                  + (lat / 12.5) ** 2))
    clumps = gaussian_filter(rng.normal(size=xx.shape), 5.5)
    clumps = clumps / np.abs(clumps).max()
    field = band * (1.0 + 0.55 * clumps) + bulge + 0.045 * (1.0 + clumps)
    return np.radians(x_deg), np.radians(y_deg), np.clip(field, 0.0, None)


def catalogue(seed=91):
    rng = np.random.default_rng(seed)
    specs = [("star-forming region", GREEN, 130, 3.6, None),
             ("supernova remnant", GOLD, 78, 5.5, None),
             ("galaxy cluster", BLUE, 105, None, 13.0),
             ("AGN / quasar", PURPLE, 150, None, 8.0)]
    out = []
    for name, color, n, b_scale, avoid in specs:
        if b_scale is not None:                      # disc population
            lat = np.clip(rng.laplace(0.0, b_scale, n), -46.0, 46.0)
            core = rng.normal(0.0, 58.0, n)
            wide = rng.uniform(-180.0, 180.0, n)
            pick = rng.random(n) < 0.62
            lon = wrap180(np.where(pick, core, wide))
            mag = 7.0 + 5.6 * rng.beta(2.1, 2.3, n)
        else:                                        # isotropic population
            lon = np.empty(0)
            lat = np.empty(0)
            while lon.size < n:
                cand_l = rng.uniform(-180.0, 180.0, 4 * n)
                cand_b = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 4 * n)))
                ok = np.abs(cand_b) > avoid
                lon = np.concatenate([lon, cand_l[ok]])[:n]
                lat = np.concatenate([lat, cand_b[ok]])[:n]
            mag = 10.0 + 4.4 * rng.beta(2.6, 1.7, n)
        ra, dec = gal_to_eq(lon, lat)
        out.append(dict(name=name, color=color, ra=ra, dec=dec, gl=lon,
                        gb=lat, mag=mag, galactic=b_scale is not None))
    return out


def map_panel(fig, ax, grid, cat):
    gx, gy, field = grid
    mesh = ax.pcolormesh(gx, gy, field, shading="auto", cmap=FORE_CMAP,
                         vmin=0.0, vmax=field.max() * 1.35, zorder=1,
                         rasterized=True)

    plane_l = np.linspace(0.0, 360.0, 900)
    pra, pdec = gal_to_eq(plane_l, np.zeros_like(plane_l))
    px, py = proj(pra, pdec)
    px = np.where(np.abs(np.diff(px, prepend=px[0])) > np.pi, np.nan, px)
    ax.plot(px, py, color=RED, lw=1.2, zorder=4)

    mags = np.concatenate([c["mag"] for c in cat])
    m_lo, m_hi = mags.min(), mags.max()
    for c in cat:
        x, y = proj(c["ra"], c["dec"])
        size = 3.0 + 26.0 * (1.0 - (c["mag"] - m_lo) / (m_hi - m_lo)) ** 1.7
        ax.scatter(x, y, s=size, c=c["color"], alpha=0.85, linewidths=0.25,
                   edgecolors="white", zorder=5)

    for name, ra, dec, (dra, ddec) in REGIONS:
        x, y = proj(ra, dec)
        ax.plot([x], [y], marker="*", ms=7.0, mfc="#f4f1ea", mec=RULE,
                mew=0.7, zorder=7)
        lx, ly = proj(ra + dra, dec + ddec)
        halo_text(ax, lx, ly, name, ha="center",
                  va="bottom" if ddec > 0 else "top", fontsize=6.2,
                  color="#f4f1ea", halo="#1a1a1a", halo_lw=1.6, zorder=8)

    ax.grid(lw=0.4, color="#9a9a9a", alpha=0.55)
    ax.set_xticks(np.radians([-120.0, -60.0, 0.0, 60.0, 120.0]))
    # GeoAxes puts the longitude labels on the equator, where they disappear
    # into the foreground band; haloed text placed by hand always reads.
    ax.set_xticklabels([])
    for ra_lab in (0, 60, 120, 240, 300):
        lx, ly = proj(float(ra_lab), 0.0)
        halo_text(ax, lx, ly - 0.055, f"{ra_lab}$^\\circ$", ha="center",
                  va="top", fontsize=6.2, color=RULE, halo="white",
                  halo_lw=1.8, zorder=6)
    ax.set_yticks(np.radians([-60.0, -30.0, 0.0, 30.0, 60.0]))
    ax.tick_params(labelsize=6.4, colors=RULE)
    # No set_xlabel here: on a GeoAxes it lands underneath the horizontal
    # colorbar.  The axis convention goes in the colorbar label instead.
    safe_colorbar(fig, mesh, ax,
                  label="relative foreground brightness (-);  RA increases "
                        "to the left, Dec labelled at the left edge",
                  side="bottom", width=0.10, pad=0.08, shrink=0.74,
                  labelsize=6.4)
    return m_lo, m_hi


def latitude_panel(ax, cat):
    edges = np.linspace(-90.0, 90.0, 25)
    gal = np.concatenate([c["gb"] for c in cat if c["galactic"]])
    ext = np.concatenate([c["gb"] for c in cat if not c["galactic"]])
    floor = 0.65
    centres = 0.5 * (edges[1:] + edges[:-1])
    for arr, color, name in [(gal, RED, "Galactic classes"),
                             (ext, BLUE, "extragalactic classes")]:
        # A log count axis is what makes both populations readable at once:
        # the Galactic peak is an order of magnitude above the isotropic
        # background, so on a linear axis one of the two is always a flat line.
        h = np.maximum(np.histogram(arr, bins=edges)[0], floor)
        ax.step(centres, h, where="mid", color=color, lw=1.4, zorder=3,
                label=name)
        ax.fill_between(centres, floor, h, step="mid", color=color,
                        alpha=0.16, linewidth=0.0, zorder=2)
    conc = float(np.mean(np.abs(gal) < 10.0))
    ax.axvspan(-10.0, 10.0, color=GREY, alpha=0.16, linewidth=0.0, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(floor, 260.0)
    ax.text(0.98, 0.98, f"Galactic: {conc * 100:.0f} % at "
                        r"$|b| < 10^\circ$",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=INK)
    ax.set_xlim(-90.0, 90.0)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("Galactic latitude $b$ (deg)", fontsize=7.4)
    ax.set_ylabel("sources per bin (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left",
              handlelength=1.4, borderaxespad=0.3)
    ax.set_title("Counts against Galactic latitude", fontsize=8.4,
                 fontweight="bold", color=INK)
    return conc


def magnitude_panel(ax, cat, m_lo, m_hi):
    mags = np.concatenate([c["mag"] for c in cat])
    edges = np.linspace(m_lo, m_hi, 22)
    h, _ = np.histogram(mags, bins=edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    ax.barh(centres, h, height=(edges[1] - edges[0]) * 0.85, color=BLUE,
            edgecolor=RULE, linewidth=0.4, zorder=3)
    med = float(np.median(mags))
    ax.axhline(med, color=RED, lw=1.1, ls=(0, (3.4, 2.2)), zorder=4)
    # Extra head-room on the count axis so the corner captions clear the bars.
    ax.set_xlim(0.0, h.max() * 1.22)
    ax.invert_yaxis()                       # brighter magnitudes at the top
    ax.text(0.97, 0.15, f"median {med:.1f} mag", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.6, color=RED)
    ax.text(0.97, 0.05, f"n = {mags.size}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.6, color=SUB)
    ax.set_xlabel("sources per bin (-)", fontsize=7.4)
    ax.set_ylabel("apparent magnitude (mag)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(axis="x", lw=0.45, alpha=0.6)
    ax.set_title("Magnitude distribution, bright end up", fontsize=8.4,
                 fontweight="bold", color=INK)
    return med


def main():
    set_pub_style()
    grid = foreground()
    cat = catalogue()

    fw, fh = 10.4, 6.4
    fig = plt.figure(figsize=(fw, fh))
    # The Mollweide box is exactly 2:1 so the projection fills it without the
    # GeoAxes aspect rule shrinking the box and stranding the colorbar.
    ax_map = axes_inches(fig, (0.58, 2.27, 6.30, 3.15), projection="mollweide")
    ax_lat = axes_inches(fig, (7.45, 3.57, 2.68, 1.85))
    ax_mag = axes_inches(fig, (7.45, 1.05, 2.68, 1.85))

    m_lo, m_hi = map_panel(fig, ax_map, grid, cat)
    conc = latitude_panel(ax_lat, cat)
    med = magnitude_panel(ax_mag, cat, m_lo, m_hi)

    n_tot = sum(c["ra"].size for c in cat)
    ax_map.set_title(
        f"{n_tot} sources in four classes over the Galactic foreground",
        fontsize=9.0, fontweight="bold", color=INK, pad=8.0)

    handles = [Line2D([], [], marker="o", ls="none", ms=4.4, mfc=c["color"],
                      mec="white", mew=0.4, label=c["name"]) for c in cat]
    handles.append(Line2D([], [], color=RED, lw=1.2, label="Galactic plane"))
    handles.append(Line2D([], [], marker="*", ls="none", ms=6.5,
                          mfc="#f4f1ea", mec=RULE, mew=0.7,
                          label="named region"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.358, 0.212),
               ncol=3, frameon=False, fontsize=6.9, handlelength=1.3,
               columnspacing=1.8, labelspacing=0.5)


    panel_labels_inches(fig, [ax_map, ax_lat, ax_mag], dx=-0.48, dy=0.34)
    savefig(fig, "803_ast_skymap_mollweide.png")


if __name__ == "__main__":
    main()
