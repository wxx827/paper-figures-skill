"""Capacitated vehicle routing solution for one depot and 28 customers.

Customers are split into balanced angular sectors, sequenced nearest-first
and then cleaned up with 2-opt, so the drawn loops do not cross themselves -
crossing tours are the fastest way to make a routing figure look wrong.
Marker area encodes drop size, each closed loop gets one house colour, and
every route carries its own load factor and driven distance.  The auxiliary
column shows the fleet-size trade-off (total distance against the number of
routes, with capacity-infeasible fleet sizes hollow) and the resulting shift
schedule.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig
from figkit import add_scale_bar, panel_labels_inches

ROUTE_COLORS = ["#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf"]
RED = "#b23a48"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
LAND = "#fbfaf8"

EXT_X, EXT_Y = 62.0, 54.0
DEPOT = np.array([31.0, 27.0])
CAPACITY = 1200.0       # units per vehicle
SPEED = 42.0            # km/h
SERVICE_H = 0.22        # h per stop
SHIFT_H = 8.0


def two_opt(order, D):
    """Remove crossings from a closed tour that starts at ``order[0]``."""
    n = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c, d = order[j], order[(j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    return order


def build_routes(pts, demand, D, k):
    """Balanced sector split, nearest-neighbour seed, then 2-opt."""
    ang = np.arctan2(pts[1:, 1] - DEPOT[1], pts[1:, 0] - DEPOT[0])
    order = np.argsort(ang % (2 * np.pi)) + 1
    cum = np.cumsum(demand[order])
    grp = np.clip((cum / (cum[-1] + 1e-9) * k).astype(int), 0, k - 1)
    routes = []
    for g in range(k):
        members = list(order[grp == g])
        if not members:
            continue
        tour, cur, pool = [0], 0, list(members)
        while pool:
            nxt = min(pool, key=lambda m: D[cur, m])
            tour.append(nxt)
            pool.remove(nxt)
            cur = nxt
        routes.append(two_opt(tour, D))
    return routes


def tour_length(tour, D):
    return sum(D[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def main():
    set_pub_style()
    rng = np.random.default_rng(560)

    n_cust = 28
    cust = np.column_stack([rng.uniform(4.0, EXT_X - 4.0, n_cust),
                            rng.uniform(4.0, EXT_Y - 4.0, n_cust)])
    pts = np.vstack([DEPOT, cust])
    demand = np.concatenate([[0.0], rng.gamma(4.0, 26.0, n_cust) + 40.0])
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)

    sweep_k = np.arange(2, 8)
    sweep_dist, sweep_maxload = [], []
    for k in sweep_k:
        rs = build_routes(pts, demand, D, k)
        sweep_dist.append(sum(tour_length(r, D) for r in rs))
        sweep_maxload.append(max(demand[r[1:]].sum() for r in rs))
    sweep_dist = np.array(sweep_dist)
    sweep_maxload = np.array(sweep_maxload)
    feasible = sweep_maxload <= CAPACITY
    k_star = int(sweep_k[feasible][np.argmin(sweep_dist[feasible])])

    routes = build_routes(pts, demand, D, k_star)
    lengths = np.array([tour_length(r, D) for r in routes])
    loads = np.array([demand[r[1:]].sum() for r in routes])
    durations = lengths / SPEED + SERVICE_H * np.array(
        [len(r) - 1 for r in routes])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.90],
                          wspace=0.20, hspace=0.45)
    axm = fig.add_subplot(gs[:, 0])
    axp = fig.add_subplot(gs[0, 1])
    axg = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------- (a) map
    axm.set_facecolor(LAND)
    axm.set_xlim(0, EXT_X)
    axm.set_ylim(0, EXT_Y)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_edgecolor("#c9c6bf")

    dmax = demand[1:].max()
    colors = (ROUTE_COLORS * 3)[:len(routes)]
    for r, col in zip(routes, colors):
        loop = pts[r + [r[0]]]
        axm.plot(loop[:, 0], loop[:, 1], color=col, lw=1.35, alpha=0.92,
                 solid_capstyle="round", solid_joinstyle="round", zorder=3)
        axm.scatter(pts[r[1:], 0], pts[r[1:], 1],
                    s=10.0 + 34.0 * demand[r[1:]] / dmax, facecolor=col,
                    edgecolor="white", linewidth=0.55, zorder=5)

    axm.plot(DEPOT[0], DEPOT[1], marker="s", ms=10.0, mfc="#3a3a3a",
             mec="white", mew=1.0, zorder=9)
    axm.text(DEPOT[0], DEPOT[1] - 2.2, "depot", fontsize=6.8, color=INK,
             ha="center", va="top", zorder=9)

    add_scale_bar(axm, 10.0, "10 km", loc=(0.035, 0.045), height=0.018,
                  color="#3a3a3a", alt="white", fontsize=6.2,
                  text_color=INK, halo="white")

    handles = [Line2D([], [], color=c, lw=1.6, marker="o", ms=4.0,
                      mfc=c, mec="white",
                      label="route %d:  %.0f km,  %.0f%% full"
                            % (i + 1, lengths[i], 100.0 * loads[i] / CAPACITY))
               for i, c in enumerate(colors)]
    handles.append(Line2D([], [], marker="s", ms=6.0, mfc="#3a3a3a",
                          mec="white", ls="none", label="depot"))
    leg = axm.legend(handles=handles, loc="lower right",
                     bbox_to_anchor=(0.995, 0.010), fontsize=6.6,
                     labelspacing=0.40, handlelength=1.8, borderpad=0.45,
                     frameon=True, framealpha=0.92, facecolor="white",
                     edgecolor="#cccccc", title="marker area = drop size")
    leg.get_title().set_fontsize(6.6)
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(10)
    axm.set_title("Routes after sector split and 2-opt", fontsize=9.0,
                  fontweight="bold", color=INK, pad=5)

    # ------------------------------------------- (b) fleet-size trade-off
    axp.plot(sweep_k, sweep_dist, color="#8a8a8a", lw=1.0, ls="--", zorder=2)
    axp.scatter(sweep_k[feasible], sweep_dist[feasible], s=30,
                facecolor="#3b5b92", edgecolor="white", linewidth=0.6,
                zorder=4, label="capacity feasible")
    axp.scatter(sweep_k[~feasible], sweep_dist[~feasible], s=30,
                facecolor="none", edgecolor=RED, linewidth=1.0, zorder=4,
                label="capacity violated")
    axp.scatter([k_star], [sweep_dist[sweep_k == k_star]], s=78,
                facecolor="none", edgecolor="#2e7d5b", linewidth=1.4,
                zorder=5)
    axp.annotate("selected: %d routes\n%.0f km total"
                 % (k_star, sweep_dist[sweep_k == k_star][0]),
                 xy=(k_star, sweep_dist[sweep_k == k_star][0]),
                 xytext=(k_star + 0.30, sweep_dist.max() * 0.985),
                 fontsize=6.6, color=INK, ha="left", va="top",
                 linespacing=1.25,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=RULE,
                                 shrinkA=1, shrinkB=4))
    axp.set_xlabel("routes in the plan (count)", fontsize=8.0)
    axp.set_ylabel("total distance (km)", fontsize=8.0)
    axp.set_xticks(sweep_k)
    axp.tick_params(labelsize=6.8)
    axp.legend(fontsize=6.5, frameon=False, loc="lower right",
               labelspacing=0.32, handlelength=1.2)
    axp.grid(color="#e9e7e2", lw=0.55)
    axp.set_axisbelow(True)
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    axp.set_title("Fewer trucks, more kilometres", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    # ------------------------------------------------- (c) shift schedule
    ypos = np.arange(len(routes))[::-1]
    for y, r, col, dur in zip(ypos, routes, colors, durations):
        axg.barh(y, SHIFT_H, height=0.50, color="#efeeea", edgecolor="none",
                 zorder=2)
        axg.barh(y, dur, height=0.50, color=col, edgecolor="white",
                 linewidth=0.5, zorder=3)
        t = 0.0
        for i in range(1, len(r)):
            t += D[r[i - 1], r[i]] / SPEED + SERVICE_H
            axg.plot([t, t], [y - 0.25, y + 0.25], color="white", lw=0.7,
                     zorder=4)
        axg.text(dur + 0.12, y, "%.1f h" % dur, fontsize=6.6, color=INK,
                 va="center", ha="left", zorder=5)
    axg.axvline(SHIFT_H, color=RED, lw=1.0, ls="--", zorder=6)
    axg.text(SHIFT_H - 0.12, ypos.max() + 0.52, "shift limit %.0f h" % SHIFT_H,
             fontsize=6.5, color=RED, ha="right", va="bottom", zorder=6)
    axg.set_yticks(ypos)
    axg.set_yticklabels(["route %d" % (i + 1) for i in range(len(routes))],
                        fontsize=6.8)
    axg.set_xlim(0, SHIFT_H * 1.22)
    axg.set_ylim(-0.7, ypos.max() + 0.9)
    axg.set_xlabel("elapsed time from depot departure (h)", fontsize=8.0)
    axg.tick_params(labelsize=6.8)
    axg.grid(axis="x", color="#e9e7e2", lw=0.55)
    axg.set_axisbelow(True)
    for sp in ("top", "right"):
        axg.spines[sp].set_visible(False)
    axg.set_title("Every route fits inside one shift", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.6, w_pad=2.6)
    for ax in (axm, axp, axg):
        ax.apply_aspect()
    panel_labels_inches(fig, [axm, axp, axg], dx=-0.22, dy=0.16)
    savefig(fig, "560_scm_vehicle_routing.png")


if __name__ == "__main__":
    main()
