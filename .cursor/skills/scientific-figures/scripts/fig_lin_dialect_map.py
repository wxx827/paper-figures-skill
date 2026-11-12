"""Dialect areas defined by a bundle of isoglosses, not by a single line.

Every element on the map comes from one pair of latent fields: the areas are
regions of that field, the six isoglosses are contours of the same field with
an independent smooth perturbation added, and each survey point's variant for
each feature is read off at its own location.  That is what produces the
figure's actual claim - the boundaries are not drawn, they emerge where four
independently wandering isoglosses happen to run together.

Line style, not colour, separates the isoglosses, so they stay legible over
the tinted areas.  The four small maps repeat the plot one feature at a time
so the reader can check that a single feature is a much blunter instrument
than the bundle, and the dendrogram clusters ten localities on the six-feature
vector alone, with no spatial information.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform

from pubstyle import set_pub_style, savefig, darken
from figkit import add_scale_bar, axes_inches, halo_text, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"
ISO = "#2b2b2b"

REGIONS = ["Northern", "Midland", "South-western", "South-eastern"]
REGION_COLOR = {"Northern": "#3b5b92", "Midland": "#2e7d5b",
                "South-western": "#c9a227", "South-eastern": "#b23a48"}
VARIANT_COLOR = ("#3b5b92", "#b23a48")

# label, contour level in latent units, line style
FEATURES = [
    ("/u:/ fronting", 0.42, "-"),
    ("rhotic coda /r/", 0.42, (0, (4.5, 2.0))),
    ("3sg present -s", 0.42, (0, (1.4, 1.6))),
    ("definite article reduction", 0.42, (0, (5.5, 1.6, 1.2, 1.6))),
    ("'bairn' for 'child'", -0.42, (0, (3.2, 1.4, 1.2, 1.4))),
    ("past participle -en kept", -0.42, (0, (2.4, 1.4))),
]
MINI = [0, 2, 4, 5]              # features repeated as small maps

PLACES = ["Aldwick", "Brenholm", "Carsdale", "Dunmere", "Eastrey",
          "Fenwick", "Garholt", "Holbeck", "Ilfham", "Kirkby"]

FW, FH = 10.4, 6.4
MAP_RECT = (0.58, 0.98, 5.30, 4.18)
Y_KM = 40.0
X_KM = Y_KM * MAP_RECT[2] / MAP_RECT[3]
NX, NY = 340, 268


def build_fields(rng):
    x = np.linspace(0.0, X_KM, NX)
    y = np.linspace(0.0, Y_KM, NY)
    Xg, Yg = np.meshgrid(x, y)
    core = 0.55 * Y_KM + 4.6 * np.sin(Xg / 8.5) + 2.1 * np.cos(Xg / 3.4)
    u = (Yg - core) / 9.0
    v = (Xg - (0.50 * X_KM + 5.2 * np.sin(Yg / 8.0))) / 10.0
    eps = []
    for _ in FEATURES:
        e = gaussian_filter(rng.normal(0.0, 1.0, (NY, NX)), 26.0)
        eps.append(0.30 * e / np.abs(e).max())
    return x, y, u, v, eps


def region_field(u, v):
    lab = np.ones(u.shape, dtype=int)
    lab[u > 0.42] = 0
    lab[(u < -0.42) & (v < 0.0)] = 2
    lab[(u < -0.42) & (v >= 0.0)] = 3
    return lab


def at_points(x, y, field, px, py):
    i = np.clip(np.searchsorted(y, py) - 1, 0, y.size - 1)
    j = np.clip(np.searchsorted(x, px) - 1, 0, x.size - 1)
    return field[i, j]


def survey(rng, n=96):
    px = rng.uniform(1.2, X_KM - 1.2, n)
    py = rng.uniform(1.2, Y_KM - 1.2, n)
    return px, py


def variant_share(x, y, u, eps, px, py, rng):
    """Share of informants using variant B, from the fields the lines use.

    Keeping this continuous is what lets the dendrogram separate localities
    that a bare majority coding would collapse onto identical vectors.
    """
    out = []
    for k, (name, level, style) in enumerate(FEATURES):
        f = at_points(x, y, u + eps[k], px, py) - level
        p = 1.0 / (1.0 + np.exp(-f / 0.16))
        out.append(np.clip(p + rng.normal(0.0, 0.055, p.size), 0.0, 1.0))
    return np.array(out).T


def draw_map(ax, x, y, u, v, eps, px, py):
    lab = region_field(u, v)
    cmap = mpl.colors.ListedColormap([REGION_COLOR[r] for r in REGIONS])
    im = ax.imshow(lab, cmap=cmap, vmin=-0.5, vmax=3.5, alpha=0.26,
                   origin="lower", extent=[0.0, X_KM, 0.0, Y_KM],
                   interpolation="nearest", zorder=1)
    im.set_rasterized(True)
    for k, r in enumerate(REGIONS):
        cs = ax.contour(x, y, (lab == k).astype(float), levels=[0.5],
                        colors=[darken(REGION_COLOR[r], 0.25)],
                        linewidths=1.1, zorder=3)
        cs.set_rasterized(False)

    for k, (name, level, style) in enumerate(FEATURES):
        ax.contour(x, y, u + eps[k], levels=[level], colors=[ISO],
                   linestyles=[style], linewidths=1.0, zorder=5)

    ax.plot(px, py, marker="o", ls="none", ms=2.3, mfc="#f5f5f5",
            mec="#2b2b2b", mew=0.5, zorder=6)

    for k, r in enumerate(REGIONS):
        m = lab == k
        cy = float(y[np.where(m.any(1))[0]].mean())
        cx = float(x[np.where(m.any(0))[0]].mean())
        halo_text(ax, cx, cy, r, ha="center", va="center", fontsize=8.0,
                  fontweight="bold", color=darken(REGION_COLOR[r], 0.35),
                  halo="white", halo_lw=2.6, zorder=8)

    add_scale_bar(ax, 10.0, "10 km", loc=(0.035, 0.045), height=0.014,
                  fontsize=6.2, color="#3a3a3a", text_color=INK,
                  halo="white")
    ax.set_xlim(0.0, X_KM)
    ax.set_ylim(0.0, Y_KM)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)", fontsize=8.0)
    ax.set_ylabel("Northing (km)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.set_title("Dialect areas and the isogloss bundle that defines them",
                 fontsize=8.5, fontweight="bold")

    handles = [Line2D([0], [0], color=ISO, lw=1.0, ls=st, label=nm)
               for nm, lv, st in FEATURES]
    handles.append(Line2D([0], [0], marker="o", ls="none", ms=2.8,
                          mfc="#f5f5f5", mec="#2b2b2b", mew=0.5,
                          label=f"survey locality (n={px.size})"))
    leg = ax.legend(handles=handles, loc="upper left", frameon=True,
                    framealpha=0.9, facecolor="white", edgecolor=HAIR,
                    borderpad=0.4, fontsize=6.2, handlelength=2.6,
                    labelspacing=0.32, title="isoglosses")
    leg.get_title().set_fontsize(6.4)
    return lab


def draw_mini(ax, x, y, u, eps, px, py, share, k, lab_field):
    name, level, style = FEATURES[k]
    im = ax.imshow(lab_field, cmap=mpl.colors.ListedColormap(
                       [REGION_COLOR[r] for r in REGIONS]),
                   vmin=-0.5, vmax=3.5, alpha=0.13, origin="lower",
                   extent=[0.0, X_KM, 0.0, Y_KM], interpolation="nearest",
                   zorder=1)
    im.set_rasterized(True)
    ax.contour(x, y, u + eps[k], levels=[level], colors=[ISO],
               linestyles=[style], linewidths=1.0, zorder=4)
    maj = (share[:, k] > 0.5).astype(int)
    for val in (0, 1):
        m = maj == val
        ax.plot(px[m], py[m], marker="o", ls="none", ms=2.2,
                mfc=VARIANT_COLOR[val], mec="white", mew=0.3, zorder=5)
    ax.set_xlim(0.0, X_KM)
    ax.set_ylim(0.0, Y_KM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_title(f"{name}  {maj.mean():.0%} B", fontsize=6.4, color=INK,
                 pad=2.0)


def draw_dendro(ax, feat, region_of):
    z = linkage(pdist(feat, metric="euclidean"), method="average")
    dn = dendrogram(z, ax=ax, labels=PLACES, color_threshold=0.0,
                    above_threshold_color="#555555", leaf_font_size=6.2)
    ax.set_ylabel("dialect distance (-)", fontsize=7.4)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.tick_params(axis="x", length=0)
    for t in ax.get_xticklabels():
        t.set_color(darken(REGION_COLOR[region_of[t.get_text()]], 0.2))
        t.set_rotation(38)
        t.set_ha("right")
    ax.set_title("Localities clustered on the six features alone",
                 fontsize=8.0, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return dn


def main():
    set_pub_style()
    rng = np.random.default_rng(635)
    x, y, u, v, eps = build_fields(rng)
    px, py = survey(rng)
    share = variant_share(x, y, u, eps, px, py, rng)

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAP_RECT)
    mini_w = 1.45
    mini_h = mini_w * Y_KM / X_KM
    minis = [axes_inches(fig, (6.58 + c * 1.62, 5.16 - mini_h - r *
                               (mini_h + 0.30), mini_w, mini_h))
             for r in range(2) for c in range(2)]
    ax_d = axes_inches(fig, (6.66, 0.98, 2.92, 1.10))

    lab = draw_map(ax_m, x, y, u, v, eps, px, py)
    for ax, k in zip(minis, MINI):
        draw_mini(ax, x, y, u, eps, px, py, share, k, lab)

    # named localities, spread out and kept clear of the legend and scale bar
    free = ~(((px < 0.36 * X_KM) & (py > 0.76 * Y_KM)) |
             ((px < 0.32 * X_KM) & (py < 0.18 * Y_KM)))
    cand = np.where(free)[0]
    cand = cand[np.argsort(px[cand] + 0.75 * py[cand])]
    idx = cand[np.linspace(0, cand.size - 1, len(PLACES), dtype=int)]
    reg = at_points(x, y, lab, px[idx], py[idx])
    region_of = {PLACES[i]: REGIONS[int(reg[i])] for i in range(len(PLACES))}
    draw_dendro(ax_d, share[idx], region_of)
    for i, p in enumerate(PLACES):
        lx = float(px[idx[i]])
        ha = "left" if lx < 6.0 else ("right" if lx > X_KM - 6.0
                                      else "center")
        dx = 0.6 if ha == "left" else (-0.6 if ha == "right" else 0.0)
        halo_text(ax_m, lx + dx, py[idx[i]] + 0.8, p, ha=ha, va="bottom",
                  fontsize=6.0, color=INK, halo="white", halo_lw=2.0,
                  zorder=9)

    fig.text((6.58 + 0.5 * (1.62 + mini_w)) / FW, (5.16 + 0.26) / FH,
             "One feature at a time", ha="center", va="bottom",
             fontsize=8.0, fontweight="bold", color=INK)
    panel_labels_inches(fig, [ax_m, minis[0], ax_d], dx=-0.32, dy=0.20)

    # how tight is the bundle?  measured on the drawn contour levels
    spread = []
    for xi in range(0, NX, 6):
        ys = []
        for k in (0, 1, 2, 3):
            col = u[:, xi] + eps[k][:, xi]
            hit = np.where(np.diff(np.sign(col - FEATURES[k][1])))[0]
            if hit.size:
                ys.append(y[hit[0]])
        if len(ys) == 4:
            spread.append(max(ys) - min(ys))
    spread = np.array(spread)

    # do the feature vectors alone know about the areas?
    dmat = squareform(pdist(share[idx], metric="euclidean"))
    same = np.array([[region_of[a] == region_of[b] for b in PLACES]
                     for a in PLACES])
    off = ~np.eye(len(PLACES), dtype=bool)
    d_in = float(dmat[same & off].mean())
    d_out = float(dmat[~same & off].mean())


    print(f"n={px.size} bundle median={np.median(spread):.2f} km "
          f"max={spread.max():.2f} km  d_in={d_in:.3f} d_out={d_out:.3f}")
    savefig(fig, "635_lin_dialect_map.png")


if __name__ == "__main__":
    main()
