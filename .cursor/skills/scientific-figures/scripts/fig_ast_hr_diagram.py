"""Hertzsprung-Russell diagram: three stellar populations on one plane.

The main panel is the claim: once stars are placed on the effective
temperature / luminosity plane they do not fill it, they collapse onto a
diagonal main sequence plus two detached islands.  Temperature runs
*decreasing* to the right and luminosity is logarithmic - the two conventions
that make an HR diagram an HR diagram.  Marker colour follows an approximate
stellar colour ramp keyed to effective temperature and marker area follows the
radius derived from the Stefan-Boltzmann law, so the iso-radius diagonals are
consistent with the points rather than decoration drawn on top of them.

Both side panels are measured from the same synthetic catalogue: the counts
per spectral class, and the mass-luminosity relation whose exponent is fitted
from the plotted main-sequence stars.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GREY = "#808080"

# Approximate stellar colour ramp, cool -> hot.  Deliberately a two-ended
# warm/cool interpolation (not a rainbow): red dwarfs are orange, the Sun is
# near-white, O stars are blue-white.
STAR_RAMP = ["#9c3f1c", "#c06a33", "#dfa25e", "#f0dcae", "#f4f1ea",
             "#dde5f3", "#b3c6e8", "#7f9fd8"]
STAR_CMAP = mpl.colors.LinearSegmentedColormap.from_list("stellar", STAR_RAMP)

# Region tints, from the house qualitative order.
TINT = {"main sequence": "#3b5b92",
        "red giant branch": "#b23a48",
        "white dwarfs": "#8e6bbf"}

T_SUN = 5772.0
T_LO, T_HI = 2600.0, 34000.0
L_LO, L_HI = 2.0e-5, 5.0e5

# Spectral class boundaries in K, hot -> cool.
CLASSES = [("O", 30000.0, 55000.0), ("B", 10000.0, 30000.0),
           ("A", 7500.0, 10000.0), ("F", 6000.0, 7500.0),
           ("G", 5200.0, 6000.0), ("K", 3700.0, 5200.0),
           ("M", 2400.0, 3700.0)]


def radius(lum, teff):
    """Stefan-Boltzmann radius in solar radii."""
    return np.sqrt(lum) / (teff / T_SUN) ** 2


def catalogue(seed=17):
    rng = np.random.default_rng(seed)

    # --- main sequence, Salpeter-weighted masses ---------------------------
    n_ms = 460
    a, m_lo, m_hi = 2.35, 0.24, 17.0
    u = rng.random(n_ms)
    p = 1.0 - a
    mass = (m_lo ** p + u * (m_hi ** p - m_lo ** p)) ** (1.0 / p)
    # A young OB association contributes the handful of O and early-B stars
    # that a Salpeter draw over a field volume would essentially never yield.
    mass = np.concatenate(
        [mass, 10.0 ** rng.uniform(np.log10(17.0), np.log10(33.0), 12)])
    lum_ms = mass ** 3.5 * 10.0 ** rng.normal(0.0, 0.075, mass.size)
    t_ms = T_SUN * mass ** 0.505 * 10.0 ** rng.normal(0.0, 0.013, mass.size)

    # --- red giant branch plus a red clump overdensity ---------------------
    n_rgb = 74
    log_l = rng.uniform(1.15, 3.35, n_rgb)
    t_rgb = 5150.0 - 430.0 * (log_l - 1.15) + rng.normal(0.0, 105.0, n_rgb)
    lum_rgb = 10.0 ** log_l
    n_clump = 26
    lum_clump = 10.0 ** rng.normal(1.72, 0.09, n_clump)
    t_clump = rng.normal(4780.0, 95.0, n_clump)
    lum_rgb = np.concatenate([lum_rgb, lum_clump])
    t_rgb = np.concatenate([t_rgb, t_clump])

    # --- white dwarfs: a cooling track at fixed degenerate radius ----------
    n_wd = 62
    r_wd = rng.normal(0.0132, 0.0016, n_wd)
    t_wd = 10.0 ** rng.uniform(np.log10(5800.0), np.log10(27000.0), n_wd)
    lum_wd = r_wd ** 2 * (t_wd / T_SUN) ** 4 * 10.0 ** rng.normal(0.0, 0.05,
                                                                  n_wd)
    return dict(ms=(t_ms, lum_ms, mass), rgb=(t_rgb, lum_rgb),
                wd=(t_wd, lum_wd))


def hull_patch(ax, teff, lum, color, grow=1.05):
    """Soft convex envelope around a population, drawn in log-log space."""
    pts = np.column_stack([np.log10(teff), np.log10(lum)])
    hull = ConvexHull(pts)
    ring = pts[hull.vertices]
    centre = ring.mean(axis=0)
    ring = centre + (ring - centre) * grow
    poly = Polygon(10.0 ** ring, closed=True, facecolor=color, alpha=0.10,
                   edgecolor=color, linewidth=0.7, linestyle=(0, (3.5, 2.4)),
                   zorder=1)
    ax.add_patch(poly)
    return 10.0 ** centre


def main_panel(ax, cat, norm):
    t_all = np.concatenate([cat["ms"][0], cat["rgb"][0], cat["wd"][0]])
    l_all = np.concatenate([cat["ms"][1], cat["rgb"][1], cat["wd"][1]])
    r_all = radius(l_all, t_all)

    # iso-radius diagonals, consistent with the marker areas
    t_line = np.logspace(np.log10(T_LO), np.log10(T_HI), 200)
    iso = [(0.01, 8000.0, 0.30, "top"), (1.0, 13000.0, 2.1, "bottom"),
           (100.0, 5200.0, 2.1, "bottom")]
    for r_iso, t_lab, off, va in iso:
        l_iso = r_iso ** 2 * (t_line / T_SUN) ** 4
        ax.plot(t_line, l_iso, color=GREY, lw=0.75, ls=(0, (5, 3)), zorder=2)
        ax.text(t_lab, r_iso ** 2 * (t_lab / T_SUN) ** 4 * off,
                f"{r_iso:g} $R_\\odot$", fontsize=6.4, color=GREY,
                ha="center", va=va, zorder=2)

    for key, name in [("ms", "main sequence"), ("rgb", "red giant branch"),
                      ("wd", "white dwarfs")]:
        t, l = cat[key][0], cat[key][1]
        hull_patch(ax, t, l, TINT[name])

    size = 2.2 + 21.0 * np.clip((np.log10(r_all) + 2.3) / 4.6, 0.0, 1.0) ** 1.5
    ax.scatter(t_all, l_all, s=size, c=np.log10(t_all), cmap=STAR_CMAP,
               norm=norm, linewidths=0.35, edgecolors="#4a4a4a", zorder=4)

    ax.plot([T_SUN], [1.0], marker="o", ms=6.2, mfc="#f4f1ea",
            mec=RULE, mew=1.0, zorder=6)
    ax.annotate("Sun,  5772 K", xy=(T_SUN, 1.0),
                xytext=(11500.0, 0.040), fontsize=7.0, color=INK,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.7))

    labels = {"main sequence": (14000.0, 2.0e3),
              "red giant branch": (3400.0, 3.0e4),
              "white dwarfs": (23000.0, 2.6e-4)}
    for name, (tx, ly) in labels.items():
        ax.text(tx, ly, name, fontsize=7.6, color=TINT[name],
                fontweight="bold", ha="center", va="center", zorder=6)

    for letter, t0, t1 in CLASSES:
        mid = np.sqrt(max(t0, T_LO) * min(t1, T_HI))
        if T_LO < mid < T_HI:
            ax.text(mid, L_HI * 0.26, letter, fontsize=7.2, color=MUTED,
                    ha="center", va="top", zorder=3)
    ax.text(T_HI * 0.985, L_HI * 0.80, "spectral class", fontsize=6.4,
            color=MUTED, ha="left", va="top", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(T_HI, T_LO)                       # temperature decreases right
    ax.set_ylim(L_LO, L_HI)
    ax.set_xticks([30000, 20000, 10000, 7000, 5000, 3500, 2800])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _p: f"{v:,.0f}".replace(",", " ")))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("effective temperature $T_{\\rm eff}$ (K)")
    ax.set_ylabel("luminosity $L$ ($L_\\odot$)")
    ax.tick_params(labelsize=7.2)
    ax.grid(lw=0.45, alpha=0.55)
    return r_all


def class_counts(cat):
    t_all = np.concatenate([cat["ms"][0], cat["rgb"][0], cat["wd"][0]])
    names, counts, cols = [], [], []
    for letter, t0, t1 in CLASSES:
        n = int(np.sum((t_all >= t0) & (t_all < t1)))
        names.append(letter)
        counts.append(n)
        cols.append(np.sqrt(max(t0, T_LO) * min(t1, T_HI)))
    return names, np.array(counts), np.array(cols)


def counts_panel(ax, cat, norm):
    names, counts, temps = class_counts(cat)
    x = np.arange(len(names))
    colors = [STAR_CMAP(norm(np.log10(t))) for t in temps]
    ax.bar(x, counts, width=0.72, color=colors, edgecolor=RULE,
           linewidth=0.55, zorder=3)
    for xi, c in zip(x, counts):
        ax.text(xi, c + 0.02 * counts.max(), f"{c:d}", ha="center",
                va="bottom", fontsize=6.2, color=SUB)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.0)
    ax.set_ylim(0.0, counts.max() * 1.22)
    ax.set_xlabel("spectral class (-)", fontsize=7.4)
    ax.set_ylabel("star count (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title("Luminosity function by spectral class", fontsize=8.4,
                 fontweight="bold", color=INK)
    return counts


def mass_lum_panel(ax, cat, norm):
    t_ms, l_ms, mass = cat["ms"]
    slope, icept = np.polyfit(np.log10(mass), np.log10(l_ms), 1)
    ax.scatter(mass, l_ms, s=6.0, c=np.log10(t_ms), cmap=STAR_CMAP, norm=norm,
               linewidths=0.3, edgecolors="#4a4a4a", zorder=3)
    m_grid = np.logspace(np.log10(mass.min()), np.log10(mass.max()), 120)
    ax.plot(m_grid, 10.0 ** (icept + slope * np.log10(m_grid)),
            color="#b23a48", lw=1.5, zorder=4)
    ax.text(0.05, 0.94,
            f"$L \\propto M^{{{slope:.2f}}}$   (n = {mass.size})",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            color="#b23a48")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mass $M$ ($M_\\odot$)", fontsize=7.4)
    ax.set_ylabel("$L$ ($L_\\odot$)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.grid(lw=0.45, alpha=0.6)
    ax.set_title("Main-sequence mass-luminosity relation", fontsize=8.4,
                 fontweight="bold", color=INK)
    return slope


def main():
    set_pub_style()
    cat = catalogue()
    norm = mpl.colors.Normalize(np.log10(T_LO), np.log10(T_HI))

    fw, fh = 10.4, 6.4
    fig = plt.figure(figsize=(fw, fh))
    ax_main = axes_inches(fig, (0.86, 0.80, 5.50, 4.66))
    ax_cnt = axes_inches(fig, (7.00, 3.30, 3.05, 2.16))
    ax_ml = axes_inches(fig, (7.00, 0.80, 3.05, 1.80))

    r_all = main_panel(ax_main, cat, norm)
    counts_panel(ax_cnt, cat, norm)
    slope = mass_lum_panel(ax_ml, cat, norm)

    n_tot = r_all.size
    f_ms = cat["ms"][0].size / n_tot
    r_span = r_all.max() / r_all.min()
    ax_main.set_title(
        f"{f_ms * 100:.0f} % of the {n_tot} catalogued stars lie on the main "
        f"sequence", fontsize=9.0, fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax_main, ax_cnt, ax_ml], dx=-0.66, dy=0.30)
    savefig(fig, "798_ast_hr_diagram.png")


if __name__ == "__main__":
    main()
