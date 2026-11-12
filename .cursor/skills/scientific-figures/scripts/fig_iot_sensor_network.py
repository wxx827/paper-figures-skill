"""Figure 669 - Wireless sensor deployment across a four-bay plant.

Main panel: the floor plan, drawn on an equal-aspect axes so the coverage
discs stay circular and distances on paper mean metres.  Sensor type is
carried by marker shape rather than by colour, which leaves colour free for
the one continuous quantity worth a scale here - the received power on each
uplink.  Routes are the parent edges of a breadth-first tree grown from the
three gateways, so the hop counts in panel (c) are the same routing the main
panel draws rather than an independent invention.

Right column: (b) the two things that take a node offline - a flat battery and
a collapsed report rate - with the nodes below the replacement threshold
called out; (c) how many nodes sit at each distance from a gateway.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from pubstyle import set_pub_style, savefig, get_cmap, lighten
from figkit import axes_inches, panel_label_inches, safe_colorbar

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"

NODE = "#3b5b92"       # house blue - gateways and healthy nodes
LOW = "#b23a48"        # house red  - the single emphasis: nodes to service
WALL = "#6f6259"       # neutral warm grey - built structure

# Marker per sensor class; shape, not hue, so the RSSI ramp stays unambiguous.
KINDS = [("vibration", "o"), ("temperature", "s"), ("pressure", "^")]
BAYS = [(2.0, 20.0, 20.0, 15.0), (24.0, 20.0, 20.0, 15.0),
        (2.0, 2.0, 20.0, 15.0), (24.0, 2.0, 20.0, 15.0)]
BAY_NAMES = ["Bay A  machining", "Bay B  assembly",
             "Bay C  heat treat", "Bay D  packing"]
GATEWAYS = np.array([[12.5, 18.5], [33.5, 18.5], [23.0, 30.0]])
RANGE_M = 9.5          # starting radio range for the routing graph
COVER_M = 3.0          # nominal sensing coverage radius
LOW_BATT = 25.0        # replacement threshold, percent


def hop_tree(pos, n_gw, reach):
    """Breadth-first routing tree from the gateways; returns hops and parents."""
    d = np.hypot(pos[:, None, 0] - pos[None, :, 0],
                 pos[:, None, 1] - pos[None, :, 1])
    adj = (d <= reach) & (d > 0.0)
    hops = np.full(pos.shape[0], -1)
    parent = np.full(pos.shape[0], -1)
    hops[:n_gw] = 0
    frontier = list(range(n_gw))
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(adj[u]):
                if hops[v] < 0 or hops[v] > hops[u] + 1:
                    hops[v], parent[v] = hops[u] + 1, u
                    nxt.append(int(v))
                elif hops[v] == hops[u] + 1 and parent[v] >= 0 \
                        and d[u, v] < d[parent[v], v]:
                    parent[v] = u
        frontier = nxt
    return hops, parent, d


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(669)

    # Sensors on a jittered grid inside each bay, so coverage stays readable.
    pts, kind = [], []
    for b, (x0, y0, w, h) in enumerate(BAYS):
        for i in range(6):
            gx = x0 + w * (0.16 + 0.68 * (i % 3) / 2.0)
            gy = y0 + h * (0.24 + 0.42 * (i // 3))
            pts.append([gx + rng.normal(0, 1.1), gy + rng.normal(0, 1.1)])
            kind.append((b + i) % len(KINDS))
    sens = np.array(pts)
    kind = np.array(kind)
    n_gw, n_s = len(GATEWAYS), len(sens)
    pos = np.vstack([GATEWAYS, sens])

    # Grow the radio range only as far as it takes to leave nobody stranded;
    # an orphan node would make the hop histogram silently wrong.
    reach = RANGE_M
    while True:
        hops, parent, dist = hop_tree(pos, n_gw, reach)
        if (hops[n_gw:] > 0).all():
            break
        reach += 0.5

    edges, rssi = [], []
    for v in range(n_gw, pos.shape[0]):
        u = parent[v]
        if u < 0:
            continue
        edges.append([pos[u], pos[v]])
        # Log-distance path loss with shadowing, in dBm.
        rssi.append(-40.0 - 22.0 * np.log10(max(dist[u, v], 1.0))
                    + rng.normal(0.0, 1.6))
    rssi = np.array(rssi)

    batt = np.clip(rng.beta(1.5, 1.0, n_s) * 100.0, 5.0, 99.0)
    rate = np.clip(99.5 - np.maximum(0.0, LOW_BATT - batt) * 1.55
                   - 1.7 * hops[n_gw:] + rng.normal(0.0, 1.4, n_s),
                   45.0, 100.0)
    weak = batt < LOW_BATT

    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- (a) floor plan ----------------------------------------------------
    axp = axes_inches(fig, (0.86, 0.76, 4.80, 4.66))
    axp.set_aspect("equal")
    axp.add_patch(Rectangle((0, 0), 46, 37, facecolor="white",
                            edgecolor=WALL, linewidth=1.2, zorder=1))
    for (x0, y0, w, h), name in zip(BAYS, BAY_NAMES):
        axp.add_patch(Rectangle((x0, y0), w, h, facecolor="#f5f5f5",
                                edgecolor=HAIR, linewidth=0.7, zorder=1))
        axp.text(x0 + 0.6, y0 + h - 0.5, name, ha="left", va="top",
                 fontsize=6.4, color=MUTED, zorder=2)

    for (x, y) in sens:
        axp.add_patch(Circle((x, y), COVER_M, facecolor=lighten(NODE, 0.92),
                             edgecolor=lighten(NODE, 0.70), linewidth=0.4,
                             zorder=2))

    # Not reversed: the darkest ink goes to the strongest link, so a pale
    # thread on the plan reads immediately as a marginal one.
    cmap = get_cmap("cb_sequential_blue")
    lc = LineCollection(edges, cmap=cmap, linewidths=1.0, zorder=3)
    lc.set_array(rssi)
    lc.set_clim(-82.0, -56.0)
    axp.add_collection(lc)

    for k, (name, mk) in enumerate(KINDS):
        m = kind == k
        axp.plot(sens[m, 0], sens[m, 1], ls="none", marker=mk, ms=4.2,
                 mfc="white", mec="#333333", mew=0.9, zorder=5)
    axp.plot(sens[weak, 0], sens[weak, 1], ls="none", marker="o", ms=8.0,
             mfc="none", mec=LOW, mew=1.2, zorder=6)
    axp.plot(GATEWAYS[:, 0], GATEWAYS[:, 1], ls="none", marker="D", ms=7.0,
             mfc=NODE, mec="white", mew=1.0, zorder=7)
    for i, (gx, gy) in enumerate(GATEWAYS):
        axp.text(gx, gy + 1.6, f"GW-{i + 1}", ha="center", va="bottom",
                 fontsize=6.4, color=NODE, fontweight="bold", zorder=7)

    axp.set_xlim(-1.0, 47.0)
    axp.set_ylim(-1.0, 44.0)
    axp.set_xticks(np.arange(0, 47, 10))
    axp.set_yticks(np.arange(0, 38, 10))
    axp.set_xlabel("plant east-west (m)", fontsize=8.0)
    axp.set_ylabel("plant north-south (m)", fontsize=8.0)
    axp.tick_params(labelsize=7.0, length=2.6)
    handles = [Line2D([], [], ls="none", marker=mk, ms=4.2, mfc="white",
                      mec="#333333", mew=0.9, label=name)
               for name, mk in KINDS]
    handles += [Line2D([], [], ls="none", marker="D", ms=6.0, mfc=NODE,
                       mec="white", label="gateway"),
                Line2D([], [], ls="none", marker="o", ms=7.0, mfc="none",
                       mec=LOW, mew=1.2, label=f"battery < {LOW_BATT:.0f} %")]
    axp.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.8,
               frameon=False, handletextpad=0.35, columnspacing=1.1,
               bbox_to_anchor=(0.5, 1.0))
    axp.set_title(f"{n_s} sensors, {n_gw} gateways, median uplink "
                  f"{np.median(rssi):.0f} dBm, {reach:.0f} m radio reach",
                  fontsize=9.0, fontweight="bold", pad=3.0)
    for s in axp.spines.values():
        s.set_visible(False)
    safe_colorbar(fig, lc, axp, label="uplink RSSI (dBm)", width=0.11,
                  pad=0.10, shrink=0.80)

    # ---- (b) battery against report rate -----------------------------------
    axb = axes_inches(fig, (7.15, 3.36, 2.95, 2.06))
    axb.axvspan(0.0, LOW_BATT, facecolor=lighten(LOW, 0.90), edgecolor="none",
                zorder=0)
    axb.plot(batt[~weak], rate[~weak], ls="none", marker="o", ms=3.6,
             mfc=NODE, mec="white", mew=0.5, zorder=3, label="in service")
    axb.plot(batt[weak], rate[weak], ls="none", marker="o", ms=5.0,
             mfc=LOW, mec="white", mew=0.6, zorder=4,
             label="replace battery")
    axb.axvline(LOW_BATT, color=LOW, lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    axb.set_xlim(0.0, 104.0)
    axb.set_ylim(rate.min() - 0.10 * np.ptp(rate), 102.0)
    axb.set_xlabel("battery remaining (%)", fontsize=7.4)
    axb.set_ylabel("report rate (%)", fontsize=7.4)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.grid(axis="y", color=GRID, lw=0.6)
    axb.legend(loc="lower right", fontsize=6.5, frameon=False,
               handletextpad=0.35, labelspacing=0.25)
    axb.set_title(f"{weak.sum()} of {n_s} nodes below threshold",
                  fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # ---- (c) hops to the nearest gateway -----------------------------------
    axh = axes_inches(fig, (7.15, 0.76, 2.95, 1.94))
    hs = hops[n_gw:]
    order = np.arange(1, hs.max() + 1)
    counts = np.array([(hs == k).sum() for k in order])
    shades = [lighten(NODE, f) for f in np.linspace(0.05, 0.60, order.size)]
    axh.bar(order, counts, width=0.62, color=shades, edgecolor="white",
            linewidth=0.8, zorder=3)
    for k, c in zip(order, counts):
        axh.text(k, c + 0.25, f"{c}", ha="center", va="bottom", fontsize=6.8,
                 color=INK)
    axh.set_xticks(order)
    axh.set_xlim(0.4, order[-1] + 0.6)
    axh.set_ylim(0.0, counts.max() * 1.22)
    axh.set_xlabel("hops to nearest gateway (-)", fontsize=7.4)
    axh.set_ylabel("nodes (count)", fontsize=7.4)
    axh.tick_params(labelsize=6.5, length=2.2)
    axh.grid(axis="y", color=GRID, lw=0.6)
    axh.set_title(f"Mean path {hs.mean():.2f} hops", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axh.spines[s].set_visible(False)

    for ax, lab in ((axp, "(a)"), (axb, "(b)"), (axh, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.12)
    savefig(fig, "669_iot_sensor_network.png")


if __name__ == "__main__":
    main()
