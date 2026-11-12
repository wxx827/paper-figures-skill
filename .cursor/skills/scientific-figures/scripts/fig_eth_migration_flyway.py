"""Loop migration on a schematic flyway: four tracked routes, two seasons.

The main panel is the claim: autumn and spring are not the same line walked
twice.  Each population bows west in one season and east in the other, so the
figure has to show direction and season, not just a corridor - hence the
paired colours and the mid-route arrowheads.  Route width is the number of
tracked individuals, so a thin line is a weakly sampled claim rather than a
weak flyway.  The subordinate panels turn the map into dates: the phenological
cascade of spring arrivals up the western route, and how long birds actually
sit at each stopover.  Landmasses are schematic; distances are metric only in
latitude.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import CubicSpline

from pubstyle import set_pub_style, get_palette, savefig
from figkit import axes_inches, panel_labels_inches, halo_text, add_scale_bar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
AUTUMN = "#b23a48"
SPRING = "#3b5b92"
LAND = "#f0f0f0"
COAST = "#b3b3b3"
SEA = "#eef2f7"           # lighten(house blue) - keeps land / sea separable

ROLE_COLOR = {"breeding": "#2e7d5b", "stopover": "#c9a227",
              "wintering": "#8e6bbf"}
KM_PER_DEG = 111.32

SITES = {
    "Taimyr": (30.0, 68.0, "breeding", "left"),
    "Lapland": (21.0, 66.0, "breeding", "right"),
    "Baltic": (17.0, 57.0, "stopover", "left"),
    "Wadden Sea": (7.0, 53.0, "stopover", "right"),
    "Iberia": (-6.0, 39.0, "stopover", "right"),
    "Nile delta": (31.0, 31.0, "stopover", "left"),
    "Sahel": (2.0, 14.0, "wintering", "right"),
    "Rift Valley": (36.0, 7.0, "wintering", "left"),
}

ROUTES = [
    ("Lapland Wadden Sea Iberia Sahel", -6.5, AUTUMN, 34, "autumn"),
    ("Sahel Iberia Wadden Sea Lapland", 6.5, SPRING, 29, "spring"),
    ("Taimyr Baltic Nile delta Rift Valley", -5.0, AUTUMN, 22, "autumn"),
    ("Rift Valley Nile delta Baltic Taimyr", 5.5, SPRING, 18, "spring"),
]

LANDS = [
    [(-13, 45), (-9, 40), (-4, 37), (3, 38), (9, 36), (15, 40), (21, 37),
     (27, 38), (34, 42), (40, 49), (42, 56), (37, 63), (30, 69), (22, 71),
     (14, 68), (6, 64), (0, 58), (-5, 54), (-11, 50)],
    [(-17, 26), (-12, 21), (-7, 15), (-2, 9), (6, 3), (15, 1), (24, 3),
     (33, 7), (40, 13), (44, 20), (41, 26), (36, 30), (29, 32), (21, 33),
     (13, 31), (5, 32), (-3, 31), (-11, 29)],
]


def smooth_ring(pts, n=320):
    p = np.asarray(pts, dtype=float)
    t = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.vstack([p, p[:1]]),
                                               axis=0).T))]
    cs = CubicSpline(t, np.vstack([p, p[:1]]), bc_type="periodic", axis=0)
    return cs(np.linspace(0.0, t[-1], n))


def bowed_path(names, bow, n=260):
    pts = [SITES[k][:2] for k in names]
    wp = []
    for a, b in zip(pts[:-1], pts[1:]):
        wp.append(a)
        wp.append(((a[0] + b[0]) / 2.0 + bow, (a[1] + b[1]) / 2.0))
    wp.append(pts[-1])
    p = np.asarray(wp, dtype=float)
    t = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    cs = CubicSpline(t, p, axis=0)
    return cs(np.linspace(0.0, t[-1], n))


def split_names(s):
    out, cur = [], []
    for w in s.split():
        cur.append(w)
        if " ".join(cur) in SITES:
            out.append(" ".join(cur))
            cur = []
    return out


def north_arrow(ax, x=0.955, y=0.855, size=0.070):
    ax.annotate("", xy=(x, y + size), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0,
                                mutation_scale=8, shrinkA=0, shrinkB=0))
    ax.text(x, y + size + 0.014, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.0, color=RULE)


def map_panel(ax, extent):
    ax.set_facecolor(SEA)
    for land in LANDS:
        ring = smooth_ring(land)
        ax.fill(ring[:, 0], ring[:, 1], facecolor=LAND, edgecolor=COAST,
                linewidth=0.8, zorder=1)
    ring = smooth_ring([(16, 34.6), (19, 33.4), (21, 34.4), (18, 35.6)])
    ax.fill(ring[:, 0], ring[:, 1], facecolor=LAND, edgecolor=COAST,
            linewidth=0.7, zorder=1)

    n_max = max(r[3] for r in ROUTES)
    lengths = {}
    for spec, bow, color, n_ind, season in ROUTES:
        names = split_names(spec)
        p = bowed_path(names, bow)
        lw = 0.9 + 3.1 * n_ind / n_max
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, alpha=0.85,
                solid_capstyle="round", zorder=4)
        km = float(np.sum(np.hypot(*np.diff(p, axis=0).T))) * KM_PER_DEG
        lengths[f"{names[0]}-{names[-1]} {season}"] = km
        for f in (0.30, 0.68):
            k = int(f * (p.shape[0] - 1))
            ax.add_patch(FancyArrowPatch(
                tuple(p[k]), tuple(p[k + 3]), arrowstyle="-|>",
                mutation_scale=9.0 + 5.0 * n_ind / n_max, color=color,
                lw=lw, shrinkA=0, shrinkB=0, zorder=6))

    for name, (lon, lat, role, side) in SITES.items():
        ms = 7.2 if role != "stopover" else 5.6
        ax.plot([lon], [lat], marker="o", ms=ms, mfc=ROLE_COLOR[role],
                mec="white", mew=1.0, zorder=7)
        dx = 1.6 if side == "left" else -1.6
        halo_text(ax, lon + dx, lat, name, ha="left" if side == "left"
                  else "right", va="center", fontsize=6.8, color=INK,
                  halo="white", halo_lw=2.0, zorder=8)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.grid(lw=0.4, alpha=0.5)
    ax.set_title("Autumn and spring corridors bow to opposite sides",
                 fontsize=9.0, fontweight="bold", color=INK)

    handles = [Line2D([], [], color=AUTUMN, lw=2.2, label="autumn passage"),
               Line2D([], [], color=SPRING, lw=2.2, label="spring passage")]
    handles += [Line2D([], [], color="none", marker="o", ms=5.4,
                       mfc=ROLE_COLOR[r], mec="white", mew=0.9, label=r)
                for r in ("breeding", "stopover", "wintering")]
    ax.legend(handles=handles, fontsize=6.5, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="#cccccc", borderpad=0.4,
              loc="lower left", handlelength=1.7,
              borderaxespad=0.5).set_zorder(9)
    ax.text(0.015, 0.985, "line width = tracked individuals\n"
                          f"{min(r[3] for r in ROUTES)} to {n_max} birds "
                          "per route", transform=ax.transAxes, ha="left",
            va="top", fontsize=6.4, color=SUB, zorder=9)
    add_scale_bar(ax, 1000.0 / KM_PER_DEG, "1000 km", loc=(0.40, 0.045),
                  height=0.014, text_color=INK, halo="white", fontsize=6.4)
    north_arrow(ax)
    return lengths


def phenology(seed=11):
    rng = np.random.default_rng(seed)
    spec = [("Iberia", 78.0, 7.0, 61), ("Wadden Sea", 99.0, 8.5, 74),
            ("Lapland", 124.0, 9.5, 55)]
    return {k: (rng.normal(mu, sd, n), mu) for k, mu, sd, n in spec}


def pheno_panel(ax, data):
    cols = get_palette("cb_sequential_blue", 6)[2:6:1][:3]
    cols = [cols[0], cols[1], cols[2]]
    grid = np.linspace(50.0, 155.0, 400)
    peak = 0.0
    for (name, (vals, _)), col in zip(data.items(), cols):
        sd = float(np.std(vals))
        h = 1.06 * sd * vals.size ** (-0.2)
        d = np.exp(-0.5 * ((grid[:, None] - vals[None, :]) / h) ** 2)
        dens = d.sum(axis=1) / (vals.size * h * np.sqrt(2.0 * np.pi))
        ax.fill_between(grid, 0.0, dens, color=col, alpha=0.30, lw=0.0)
        ax.plot(grid, dens, color=col, lw=1.5)
        m = float(np.mean(vals))
        ax.plot([m], [dens[np.argmin(np.abs(grid - m))]], marker="v", ms=3.4,
                color=col, mec="white", mew=0.6)
        ax.text(m, dens.max() * 1.05, f"{name}\nday {m:.0f}, n={vals.size}",
                ha="center", va="bottom", fontsize=6.2, color=col)
        peak = max(peak, float(dens.max()))
    ax.set_xlabel("arrival day of year (day)", fontsize=7.4)
    ax.set_ylabel("density (1/day)", fontsize=7.4)
    ax.set_xlim(50.0, 155.0)
    ax.set_ylim(0.0, peak * 1.34)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title("Spring arrivals cascade up the flyway", fontsize=8.2,
                 fontweight="bold", color=INK)


def stopover(seed=5):
    rng = np.random.default_rng(seed)
    spec = [("Iberia", 5.4, 1.9, 48), ("Wadden Sea", 18.5, 5.4, 66),
            ("Baltic", 11.2, 3.6, 39), ("Nile delta", 7.8, 2.7, 44)]
    return {k: rng.gamma((mu / sd) ** 2, sd ** 2 / mu, n)
            for k, mu, sd, n in spec}


def stopover_panel(ax, data):
    names = list(data)
    y = np.arange(len(names))[::-1]
    mu = np.array([float(np.mean(v)) for v in data.values()])
    sd = np.array([float(np.std(v)) for v in data.values()])
    ax.barh(y, mu, xerr=sd, height=0.60, color=ROLE_COLOR["stopover"],
            edgecolor="white", linewidth=0.6, zorder=3,
            error_kw=dict(ecolor=RULE, elinewidth=0.8, capsize=2.2))
    for yy, m, v in zip(y, mu, data.values()):
        ax.text(0.4, yy, f"n={v.size}", ha="left", va="center", fontsize=6.2,
                color="#3a3a3a", zorder=5)
        ax.text(m + sd[list(y).index(yy)] + 0.8, yy, f"{m:.1f} d", ha="left",
                va="center", fontsize=6.3, color=SUB, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.set_xlim(0.0, float((mu + sd).max()) * 1.30)
    ax.set_xlabel("stopover duration (day)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.45, alpha=0.6)
    ax.set_title("The Wadden Sea holds birds longest", fontsize=8.2,
                 fontweight="bold", color=INK)
    return dict(zip(names, mu))


def main():
    set_pub_style()
    extent = (-31.0, 46.0, 0.0, 73.0)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_map = axes_inches(fig, (0.86, 0.80, 5.00, 4.62))
    ax_phe = axes_inches(fig, (7.02, 3.40, 2.84, 2.02))
    ax_sto = axes_inches(fig, (7.02, 0.80, 2.84, 2.02))

    lengths = map_panel(ax_map, extent)
    pheno = phenology()
    pheno_panel(ax_phe, pheno)
    stop = stopover()
    means = stopover_panel(ax_sto, stop)

    longest = max(lengths.items(), key=lambda kv: kv[1])
    lag = (float(np.mean(pheno["Lapland"][0]))
           - float(np.mean(pheno["Iberia"][0])))
    panel_labels_inches(fig, [ax_map, ax_phe, ax_sto], dx=-0.52, dy=0.14)
    savefig(fig, "778_eth_migration_flyway.png")


if __name__ == "__main__":
    main()
