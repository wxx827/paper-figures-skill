"""Capacitated vehicle routing: nearest-neighbour construction plus 2-opt.

Customer coordinates and demands are drawn once from a seeded generator; the
distance matrix is Euclidean on those coordinates in kilometres.  Routes are
built by a capacity-aware nearest-neighbour sweep and then improved by 2-opt
until no improving reversal remains, so the tour lengths, the load profile
printed along each leg and the before/after comparison are all measured from
the tours that are drawn rather than assumed.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches, halo_text

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
VEH_COL = [BLUE, RED, GREEN, GOLD, PURPLE]

W, H = 10.4, 6.4
SEED = 20260813
N_CUST = 22
CAPACITY = 120.0          # units of demand per vehicle
BOX = (56.0, 44.0)        # km x km service area; panel aspect matches this

FOOT = ("Customer positions and demands come from one seeded generator; the "
        "distance matrix is Euclidean on those coordinates. Routes are built "
        "by a capacity-aware nearest-neighbour sweep and then improved with "
        "2-opt until no reversal shortens a tour, so every kilometre figure "
        "quoted here is measured from the tours actually drawn. The baseline "
        "in panel (c) keeps the same vehicle clusters but serves each one in "
        "order-book sequence.")


def instance(rng):
    depot = np.array([BOX[0] * 0.47, BOX[1] * 0.52])
    ang = rng.uniform(0.0, 2.0 * np.pi, N_CUST)
    rad = np.sqrt(rng.uniform(0.05, 1.0, N_CUST))
    pts = np.column_stack([depot[0] + rad * np.cos(ang) * BOX[0] * 0.44,
                           depot[1] + rad * np.sin(ang) * BOX[1] * 0.44])
    pts = np.clip(pts, [2.5, 2.5], [BOX[0] - 2.5, BOX[1] - 2.5])
    demand = np.round(rng.uniform(9.0, 34.0, N_CUST), 0)
    coords = np.vstack([depot, pts])
    return coords, demand


def dist_matrix(coords):
    d = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((d ** 2).sum(axis=-1))


def tour_length(route, D):
    idx = [0] + list(route) + [0]
    return float(sum(D[idx[k], idx[k + 1]] for k in range(len(idx) - 1)))


def nearest_neighbour(D, demand):
    unserved = set(range(1, len(demand) + 1))
    routes = []
    while unserved:
        route, load, cur = [], 0.0, 0
        while True:
            cand = [j for j in unserved
                    if load + demand[j - 1] <= CAPACITY + 1e-9]
            if not cand:
                break
            j = min(cand, key=lambda k: D[cur, k])
            route.append(j)
            load += demand[j - 1]
            unserved.discard(j)
            cur = j
        routes.append(route)
    return routes


def two_opt(route, D):
    best = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for k in range(i + 1, len(best)):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                if tour_length(cand, D) < tour_length(best, D) - 1e-9:
                    best, improved = cand, True
    return best


def cumulative(route, demand):
    return np.cumsum([demand[j - 1] for j in route])


def draw_map(ax, coords, demand, routes, D, total_after, total_before):
    ax.set_aspect("equal")
    ax.set_xlim(0.0, BOX[0])
    ax.set_ylim(0.0, BOX[1])
    ax.set_axisbelow(True)

    tags = []
    for v, route in enumerate(routes):
        col = VEH_COL[v % len(VEH_COL)]
        path = np.vstack([coords[0], coords[route], coords[0]])
        ax.plot(path[:, 0], path[:, 1], color=col, lw=1.35, alpha=0.92,
                solid_joinstyle="round", zorder=3)
        load = cumulative(route, demand)
        for k, j in enumerate(route):
            ax.plot([coords[j, 0]], [coords[j, 1]], marker="o",
                    ms=3.0 + 0.115 * demand[j - 1], mfc=lighten(col, 0.45),
                    mec=col, mew=0.8, zorder=5)
            if k % 2 and k != len(route) - 1:
                continue
            # Push the running-load tag radially outward so the tags thin out
            # where the routes converge on the depot.
            d = coords[j] - coords[0]
            u = d / max(float(np.hypot(*d)), 1e-9)
            tags.append((coords[j] + 1.7 * u, f"{load[k]:.0f}", col))

    placed = []
    for p, txt, col in tags:
        if any(np.hypot(*(p - q)) < 3.0 for q in placed):
            continue
        placed.append(p)
        halo_text(ax, p[0], p[1], txt, color=col, halo="white", halo_lw=2.0,
                  fontsize=6.0, ha="center", va="center", zorder=6)

    ax.plot([coords[0, 0]], [coords[0, 1]], marker="s", ms=10.0, mfc=INK,
            mec="white", mew=1.1, zorder=7)
    halo_text(ax, coords[0, 0], coords[0, 1] - 2.2, "depot", color=INK,
              halo="white", halo_lw=2.0, fontsize=6.8, ha="center", va="top",
              zorder=7)

    saved = 100.0 * (total_before - total_after) / total_before
    ax.text(0.985, 0.020,
            f"{len(routes):d} vehicles  |  {len(demand):d} customers\n"
            f"{total_after:.1f} km total  |  {saved:.1f}% below the\n"
            f"order-book sequence",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color=INK, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor=HAIR, linewidth=0.6, alpha=0.93))

    ax.set_xlabel("easting (km)", fontsize=8.0)
    ax.set_ylabel("northing (km)", fontsize=8.0)
    ax.set_title(f"{len(routes):d} capacity-feasible tours out of one depot",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(labelsize=7.2)
    style_axes(ax)

    handles = [Line2D([], [], color=VEH_COL[v % len(VEH_COL)], lw=1.6,
                      marker="o", ms=4.0,
                      label=f"vehicle {v + 1} ({tour_length(r, D):.0f} km)")
               for v, r in enumerate(routes)]
    handles.append(Line2D([], [], color="none", marker="s", ms=6.0, mfc=INK,
                          mec="white", label="depot"))
    ax.legend(handles=handles, loc="lower left", fontsize=6.4, frameon=True,
              framealpha=0.93, facecolor="white", edgecolor=HAIR,
              borderpad=0.42, handlelength=1.6, labelspacing=0.34)


def draw_loads(ax, routes, demand):
    loads = np.array([demand[[j - 1 for j in r]].sum() for r in routes])
    util = 100.0 * loads / CAPACITY
    x = np.arange(len(routes))
    cols = [VEH_COL[v % len(VEH_COL)] for v in range(len(routes))]
    ax.bar(x, util, width=0.56, color=[lighten(c, 0.40) for c in cols],
           edgecolor=cols, linewidth=0.9, zorder=3)
    ax.axhline(100.0, color=RED, lw=1.0, ls=(0, (4, 2.2)), zorder=4)
    ax.text(len(routes) - 0.52, 101.5, f"capacity {CAPACITY:.0f} units",
            ha="right", va="bottom", fontsize=6.2, color=RED)
    for xx, u, l in zip(x, util, loads):
        ax.text(xx, u - 4.0, f"{l:.0f}", ha="center", va="top", fontsize=6.2,
                color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"veh {v + 1}" for v in range(len(routes))],
                       fontsize=6.5)
    ax.set_ylim(0.0, 118.0)
    ax.set_ylabel("load utilisation (%)", fontsize=7.2)
    ax.set_title(f"Mean fill {util.mean():.0f}% of capacity", fontsize=8.5,
                 fontweight="bold")
    ax.tick_params(axis="y", labelsize=6.5)
    style_axes(ax)


def draw_gain(ax, routes_base, routes_2o, D):
    before = np.array([tour_length(r, D) for r in routes_base])
    after = np.array([tour_length(r, D) for r in routes_2o])
    x = np.arange(len(routes_2o))
    ax.bar(x - 0.19, before, width=0.34, color="#e2e2e2", edgecolor=GREY,
           linewidth=0.7, label="order-book sequence", zorder=3)
    ax.bar(x + 0.19, after, width=0.34, color=lighten(GREEN, 0.34),
           edgecolor=GREEN, linewidth=0.7, label="nearest neighbour + 2-opt",
           zorder=3)
    for xx, b, a in zip(x, before, after):
        ax.text(xx + 0.19, a + 1.2, f"-{100.0 * (b - a) / b:.0f}%",
                ha="center", va="bottom", fontsize=6.2, color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels([f"veh {v + 1}" for v in range(len(routes_2o))],
                       fontsize=6.5)
    ax.set_ylim(0.0, before.max() * 1.30)
    ax.set_ylabel("tour length (km)", fontsize=7.2)
    ax.set_title(f"Routing removes {before.sum() - after.sum():.0f} km "
                 f"overall", fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(loc="upper center", fontsize=6.3, frameon=False,
              handlelength=1.3, labelspacing=0.3, ncol=2, columnspacing=1.0)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    coords, demand = instance(rng)
    D = dist_matrix(coords)

    routes_nn = nearest_neighbour(D, demand)
    # Same clusters, but served in the order the orders arrived: the plan a
    # dispatcher gets for free, and the baseline the routing has to beat.
    routes_base = [sorted(r) for r in routes_nn]
    routes_2o = [two_opt(r, D) for r in routes_nn]
    total_before = sum(tour_length(r, D) for r in routes_base)
    total_after = sum(tour_length(r, D) for r in routes_2o)

    fig = plt.figure(figsize=(W, H))
    ax_map = axes_inches(fig, (0.82, 1.16, 5.30, 4.16))
    ax_top = axes_inches(fig, (7.10, 3.48, 3.02, 1.84))
    ax_bot = axes_inches(fig, (7.10, 1.16, 3.02, 1.84))

    draw_map(ax_map, coords, demand, routes_2o, D, total_after, total_before)
    draw_loads(ax_top, routes_2o, demand)
    draw_gain(ax_bot, routes_base, routes_2o, D)

    panel_labels_inches(fig, [ax_map, ax_top, ax_bot], dx=-0.58, dy=0.08)

    savefig(fig, "860_orx_vrp_routes.png")


if __name__ == "__main__":
    main()
