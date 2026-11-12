"""Stand structure of an uneven-aged mixed forest, read from the crown map.

The map is the argument: every crown is a circle whose radius comes from the
same crown-diameter allometry the stem list is built with, so the overlap the
reader sees *is* the canopy closure quoted in the box - that number is
measured by rasterising the crowns onto a 0.25 m grid, not asserted.  Stem
density and basal area come from the same stem list, and the species mix is
size-dependent, so the pioneer birches really are the small stems.

The two side panels test what the map suggests.  A 5 cm diameter distribution
carries a negative-exponential (reverse-J) fit whose De Liocourt quotient is
computed from the fitted decay rate, and the height-diameter panel fits one
Chapman-Richards curve across all species so the asymptotic height and its
R-squared are properties of the plotted trees.  Species colours are the shared
forest set used in 685 and 688.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, darken, annotate_box
from figkit import add_scale_bar, axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"

# Shared forest species set (identical hues in 684 / 685 / 688).
SPECIES = ["Picea abies", "Pinus sylvestris", "Betula pendula",
           "Quercus robur", "Fagus sylvatica"]
SPECIES_COLOR = {
    "Picea abies": "#2e7d5b",
    "Pinus sylvestris": "#b23a48",
    "Betula pendula": "#c9a227",
    "Quercus robur": "#3b5b92",
    "Fagus sylvatica": "#8e6bbf",
}
# Preferred diameter (cm) and spread per species: this is what makes the map
# ecologically legible - birch sits in the small classes, oak in the large.
SPECIES_PREF = {
    "Picea abies": (34.0, 20.0, 0.30),
    "Pinus sylvestris": (42.0, 22.0, 0.20),
    "Betula pendula": (14.0, 10.0, 0.22),
    "Quercus robur": (48.0, 26.0, 0.10),
    "Fagus sylvatica": (20.0, 14.0, 0.18),
}
SPECIES_HMAX = {"Picea abies": 35.0, "Pinus sylvestris": 32.0,
                "Betula pendula": 26.0, "Quercus robur": 30.0,
                "Fagus sylvatica": 33.0}

FW, FH = 10.4, 6.4
MAP_RECT = (0.62, 1.02, 5.42, 4.16)
PLOT_Y = 45.0
PLOT_X = PLOT_Y * MAP_RECT[2] / MAP_RECT[3]      # equal aspect fills the slot
N_QX, N_QY = 3, 3
DBH_MIN, DBH_MAX, DBH_BIN = 5.0, 70.0, 5.0
N_TARGET = 152


def crown_radius(dbh):
    """Crown radius (m) from diameter at breast height (cm)."""
    return 0.34 + 0.108 * np.asarray(dbh, dtype=float)


def chapman(d, hmax, k, p):
    """Chapman-Richards height (m) above the 1.3 m breast-height offset."""
    return 1.3 + hmax * (1.0 - np.exp(-k * d)) ** p


def place_trees(rng):
    """Clustered stems with crown-scale inhibition, then size-biased species."""
    centres = rng.uniform([0.0, 0.0], [PLOT_X, PLOT_Y], size=(9, 2))
    xs, ys, dbh = [], [], []
    for _ in range(40000):
        if len(xs) >= N_TARGET:
            break
        if rng.random() < 0.55:
            p = centres[rng.integers(len(centres))] + rng.normal(0.0, 6.2, 2)
        else:
            p = rng.uniform([0.0, 0.0], [PLOT_X, PLOT_Y])
        if not (0.6 <= p[0] <= PLOT_X - 0.6 and 0.6 <= p[1] <= PLOT_Y - 0.6):
            continue
        d = DBH_MIN + rng.exponential(11.5)
        if d > DBH_MAX:
            continue
        r = crown_radius(d)
        keep = True
        for xx, yy, dd in zip(xs, ys, dbh):
            lim = 0.42 * (r + crown_radius(dd))
            if (p[0] - xx) ** 2 + (p[1] - yy) ** 2 < lim * lim:
                keep = False
                break
        if keep:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            dbh.append(float(d))

    xs, ys, dbh = np.array(xs), np.array(ys), np.array(dbh)
    spp = []
    for d in dbh:
        w = np.array([b * np.exp(-0.5 * ((d - mu) / sg) ** 2)
                      for mu, sg, b in (SPECIES_PREF[s] for s in SPECIES)])
        spp.append(rng.choice(SPECIES, p=w / w.sum()))
    spp = np.array(spp)
    hmax = np.array([SPECIES_HMAX[s] for s in spp])
    h = 1.3 + hmax * (1.0 - np.exp(-0.052 * dbh)) ** 1.20
    h = h + rng.normal(0.0, 0.85, dbh.size)
    return dict(x=xs, y=ys, dbh=dbh, species=spp, h=np.clip(h, 2.0, None))


def canopy_closure(trees, step=0.25):
    """Fraction of the plot floor under at least one crown."""
    gx = np.arange(step / 2.0, PLOT_X, step)
    gy = np.arange(step / 2.0, PLOT_Y, step)
    GX, GY = np.meshgrid(gx, gy)
    cover = np.zeros(GX.shape, dtype=bool)
    for x, y, d in zip(trees["x"], trees["y"], trees["dbh"]):
        r = crown_radius(d)
        cover |= (GX - x) ** 2 + (GY - y) ** 2 <= r * r
    return float(cover.mean())


def draw_map(ax, trees, stats):
    ax.set_xlim(0.0, PLOT_X)
    ax.set_ylim(0.0, PLOT_Y)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), PLOT_X, PLOT_Y, facecolor="#f7f7f4",
                           edgecolor="none", zorder=0))

    qw, qh = PLOT_X / N_QX, PLOT_Y / N_QY
    for i in range(1, N_QX):
        ax.plot([i * qw, i * qw], [0, PLOT_Y], color="#9a9a9a", lw=0.7,
                ls=(0, (4.0, 2.4)), zorder=2)
    for j in range(1, N_QY):
        ax.plot([0, PLOT_X], [j * qh, j * qh], color="#9a9a9a", lw=0.7,
                ls=(0, (4.0, 2.4)), zorder=2)
    for j in range(N_QY):
        for i in range(N_QX):
            ax.text(i * qw + 0.5, j * qh + 0.5,
                    f"Q{j * N_QX + i + 1}", ha="left", va="bottom",
                    fontsize=6.0, color="#9a9a9a", zorder=3)

    order = np.argsort(trees["h"])
    for k in order:
        c = SPECIES_COLOR[trees["species"][k]]
        ax.add_patch(Circle((trees["x"][k], trees["y"][k]),
                            crown_radius(trees["dbh"][k]), facecolor=c,
                            edgecolor=darken(c, 0.30), linewidth=0.5,
                            alpha=0.52, zorder=4))
    ax.scatter(trees["x"], trees["y"], s=1.6, color="#3a3a3a", zorder=6)

    add_scale_bar(ax, 10.0, "10 m", loc=(0.700, 0.040), height=0.014,
                  fontsize=6.2, color="#3a3a3a", text_color=INK, halo="white")
    annotate_box(ax, "\n".join([
        f"canopy closure  {stats['closure']:.0%}",
        f"stem density  {stats['density']:.0f} trees/ha",
        f"basal area  {stats['ba']:.1f} $m^2$/ha",
        f"quadratic mean DBH  {stats['qmd']:.1f} cm"]),
        loc="upper left", fontsize=6.6, margin=0.022)

    ax.set_xlabel("easting within plot (m)", fontsize=8.0)
    ax.set_ylabel("northing within plot (m)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.set_title("Crown map with 3 x 3 quadrat grid, crowns scaled by DBH",
                 fontsize=8.5, fontweight="bold")


def draw_dbh(ax, trees):
    edges = np.arange(DBH_MIN, DBH_MAX + DBH_BIN, DBH_BIN)
    counts, _ = np.histogram(trees["dbh"], bins=edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(mid, counts, width=DBH_BIN * 0.86, color="#9cc3dd",
           edgecolor="#2e74a8", linewidth=0.7, zorder=3)

    ok = counts > 0
    slope, intercept = np.polyfit(mid[ok], np.log(counts[ok]), 1,
                                  w=np.sqrt(counts[ok]))
    b = -slope
    grid = np.linspace(DBH_MIN, DBH_MAX, 220)
    ax.plot(grid, np.exp(intercept - b * grid), color="#b23a48", lw=1.4,
            zorder=5, label="reverse-J fit")
    q = float(np.exp(b * DBH_BIN))
    ax.legend(frameon=False, fontsize=6.6, loc="upper right",
              handlelength=1.5, borderpad=0.15)
    annotate_box(ax, f"$N(D)\\propto e^{{-{b:.3f}D}}$\nDe Liocourt "
                     f"q = {q:.2f} per {DBH_BIN:.0f} cm",
                 loc="center right", fontsize=6.4, margin=0.035)
    ax.set_xlabel("DBH class (cm)", fontsize=7.8)
    ax.set_ylabel("stems (n)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_xlim(0.0, DBH_MAX + 2.0)
    ax.set_title("Reverse-J diameter distribution", fontsize=8.5,
                 fontweight="bold")
    return b, q


def draw_hd(ax, trees):
    for s in SPECIES:
        m = trees["species"] == s
        ax.scatter(trees["dbh"][m], trees["h"][m], s=8.0,
                   color=SPECIES_COLOR[s], alpha=0.85, edgecolors="white",
                   linewidths=0.3, zorder=4)
    par, _ = curve_fit(chapman, trees["dbh"], trees["h"],
                       p0=[32.0, 0.05, 1.2], maxfev=20000)
    grid = np.linspace(DBH_MIN, trees["dbh"].max(), 200)
    ax.plot(grid, chapman(grid, *par), color=RULE, lw=1.3, zorder=6)
    resid = trees["h"] - chapman(trees["dbh"], *par)
    r2 = 1.0 - resid.var() / trees["h"].var()
    ax.axhline(1.3 + par[0], color="#808080", lw=0.8, ls=(0, (4, 2.4)),
               zorder=3)
    ax.text(DBH_MAX + 1.0, 1.3 + par[0] - 0.8,
            f"asymptote {1.3 + par[0]:.1f} m", ha="right", va="top",
            fontsize=6.2, color=MUTED)
    ax.text(0.975, 0.055, f"Chapman-Richards, $R^2$ = {r2:.2f}, "
                          f"n = {trees['dbh'].size}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.4, color=SUB)
    ax.set_xlabel("DBH (cm)", fontsize=7.8)
    ax.set_ylabel("tree height (m)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_xlim(0.0, DBH_MAX + 2.0)
    ax.set_ylim(0.0, 1.3 + par[0] + 4.5)
    ax.set_title("Height-diameter relationship", fontsize=8.5,
                 fontweight="bold")
    return par, r2


def main():
    set_pub_style()
    rng = np.random.default_rng(684)
    trees = place_trees(rng)

    area_ha = PLOT_X * PLOT_Y / 10000.0
    ba = np.sum(np.pi * (trees["dbh"] / 200.0) ** 2) / area_ha
    stats = dict(closure=canopy_closure(trees),
                 density=trees["dbh"].size / area_ha,
                 ba=ba,
                 qmd=float(np.sqrt(np.mean(trees["dbh"] ** 2))))

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAP_RECT)
    ax_d = axes_inches(fig, (7.10, 3.28, 2.92, 1.90))
    ax_h = axes_inches(fig, (7.10, 1.02, 2.92, 1.62))

    draw_map(ax_m, trees, stats)
    b, q = draw_dbh(ax_d, trees)
    par, r2 = draw_hd(ax_h, trees)

    handles = [Line2D([0], [0], marker="o", ls="none", ms=4.4,
                      mfc=SPECIES_COLOR[s], mec="white", mew=0.5, label=s)
               for s in SPECIES]

    panel_labels_inches(fig, [ax_m, ax_d, ax_h], dx=-0.34, dy=0.20)

    big = float(np.mean(trees["dbh"] >= 40.0))

    print(f"n={trees['dbh'].size} closure={stats['closure']:.3f} "
          f"density={stats['density']:.1f} ba={stats['ba']:.2f} "
          f"b={b:.4f} q={q:.3f} hmax={1.3 + par[0]:.2f} r2={r2:.3f}")
    savefig(fig, "684_for_stand_structure.png")


if __name__ == "__main__":
    main()
