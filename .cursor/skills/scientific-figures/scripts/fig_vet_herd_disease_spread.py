"""Pen-level spread of an outbreak through one barn, and what the barrier did.

The main panel is a scale floor plan of a hundred pens.  Each pen is filled
with the day it broke down, so the outbreak reads as a wave leaving the index
pen: fast inside the quarantined block, then stalled at the gate line before
it leaks into the rest of the barn.  The faint web underneath is the contact
network the pathogen actually travels on - neighbouring pens, plus a handful
of long-range links from shared equipment - and the onset days are the
shortest-path times through exactly that graph, not a painted-on gradient.

The subordinate panels are the two views a veterinarian asks for next: the
epidemic curve, and whether risk really does fall off with distance from the
index case.  Both are counted from the same pen array.  Synthetic data; no
real holding is represented.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from pubstyle import set_pub_style, get_cmap, savefig, despine
from figkit import axes_inches, panel_labels_inches, safe_colorbar, add_scale_bar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
BEIGE = "#efe9dc"          # pen that never broke down
HAIR = "#cccccc"

N_COL, N_ROW = 10, 10
PEN_W, PEN_H = 2.8, 2.2    # m
GAP_X, GAP_Y = 0.35, 0.30  # m
AISLE = 2.6                # m, central service alley
AISLE_AFTER = 5            # alley sits between row 4 and row 5
QZ_COL, QZ_ROW = 5, 5      # quarantined block = columns 0-4, rows 0-4
INDEX_PEN = (2, 2)
HORIZON = 24.0             # d of observation
ONSET_CMAP = get_cmap("cb_sequential_warm")

# Mean transmission delay in days along each kind of contact, and the cost of
# getting past the gate.  The footnote quotes these constants, so the caption
# cannot drift away from the model.
W_WALL, W_ALLEY, W_EQUIP, W_GATE = 2.20, 4.20, 3.60, 9.00


def pen_grid():
    """Lower-left corner of every pen, in metres."""
    xs, ys, cols, rows = [], [], [], []
    for r in range(N_ROW):
        y = r * (PEN_H + GAP_Y) + (AISLE if r >= AISLE_AFTER else 0.0)
        for c in range(N_COL):
            xs.append(c * (PEN_W + GAP_X))
            ys.append(y)
            cols.append(c)
            rows.append(r)
    return (np.array(xs), np.array(ys), np.array(cols), np.array(rows))


def in_zone(c, r):
    return (c < QZ_COL) & (r < QZ_ROW)


def contact_edges(rng, cols, rows):
    """Neighbour contacts plus a few shared-equipment links, with delays."""
    idx = {(int(c), int(r)): i for i, (c, r) in enumerate(zip(cols, rows))}
    edges = []
    for (c, r), i in idx.items():
        if (c + 1, r) in idx:
            edges.append((i, idx[(c + 1, r)], W_WALL))
        if (c, r + 1) in idx:
            # Stepping over the central alley is a slower contact than
            # sharing a partition wall.
            w = W_ALLEY if r + 1 == AISLE_AFTER else W_WALL * 1.07
            edges.append((i, idx[(c, r + 1)], w))
    long_range = [((1, 1), (7, 2)), ((3, 3), (2, 8)), ((0, 4), (8, 7)),
                  ((4, 0), (9, 3)), ((2, 6), (8, 9))]
    for a, b in long_range:
        edges.append((idx[a], idx[b], W_EQUIP))
    out = []
    for i, j, w in edges:
        w = w * float(rng.uniform(0.78, 1.32))
        ci, ri = int(cols[i]), int(rows[i])
        cj, rj = int(cols[j]), int(rows[j])
        crosses = bool(in_zone(ci, ri) != in_zone(cj, rj))
        if crosses:
            w += W_GATE     # the gate line: passable, but expensive
        out.append((i, j, w, crosses))
    return out, long_range, idx


def onset_days(edges, n, src):
    i = np.array([e[0] for e in edges] + [e[1] for e in edges])
    j = np.array([e[1] for e in edges] + [e[0] for e in edges])
    w = np.array([e[2] for e in edges] * 2)
    g = coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    return dijkstra(g, indices=src)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def barn_panel(fig, ax, xs, ys, cols, rows, onset, edges, long_range, idx):
    seg = [[(xs[i] + PEN_W / 2, ys[i] + PEN_H / 2),
            (xs[j] + PEN_W / 2, ys[j] + PEN_H / 2)] for i, j, _, _ in edges]
    ax.add_collection(LineCollection(seg, colors=HAIR, linewidths=0.4,
                                     zorder=1))
    lr = [[(xs[idx[a]] + PEN_W / 2, ys[idx[a]] + PEN_H / 2),
           (xs[idx[b]] + PEN_W / 2, ys[idx[b]] + PEN_H / 2)]
          for a, b in long_range]
    ax.add_collection(LineCollection(lr, colors=[SUB], linewidths=0.7,
                                     linestyles=[(0, (3.0, 2.0))], zorder=2))

    hit = onset <= HORIZON
    for k in range(xs.size):
        fc = ONSET_CMAP(onset[k] / HORIZON) if hit[k] else BEIGE
        ax.add_patch(Rectangle((xs[k], ys[k]), PEN_W, PEN_H, facecolor=fc,
                               edgecolor="white", lw=0.6, zorder=3))

    src = idx[INDEX_PEN]
    ax.add_patch(Rectangle((xs[src], ys[src]), PEN_W, PEN_H, facecolor="none",
                           edgecolor=INK, lw=1.6, zorder=6))
    ax.annotate("index case", xy=(xs[src] + PEN_W / 2, ys[src]),
                xytext=(xs[src] + 6.4, ys[src] - 3.2), fontsize=6.8,
                color=INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                mutation_scale=7, shrinkA=1.0, shrinkB=2.0))

    qx = QZ_COL * (PEN_W + GAP_X) - GAP_X / 2
    qy = QZ_ROW * (PEN_H + GAP_Y) - GAP_Y / 2 + AISLE / 2
    ax.plot([-GAP_X / 2, qx, qx], [qy, qy, -GAP_Y / 2], color=GREEN, lw=2.0,
            ls=(0, (5.5, 2.4)), zorder=7, solid_capstyle="butt")
    ax.text(qx - 0.6, qy + 0.35, "quarantine gate line", fontsize=6.8,
            color=GREEN, ha="right", va="bottom", zorder=8)

    w_tot = (N_COL - 1) * (PEN_W + GAP_X) + PEN_W
    h_tot = (N_ROW - 1) * (PEN_H + GAP_Y) + PEN_H + AISLE
    ax.set_xlim(-2.2, w_tot + 2.2)
    ax.set_ylim(-4.6, h_tot + 2.0)
    ax.set_aspect("equal")
    ax.set_xlabel("barn easting (m)", fontsize=8.0)
    ax.set_ylabel("barn northing (m)", fontsize=8.0)
    ax.tick_params(labelsize=7.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("The wave stalls at the gate line before it leaks out",
                 fontsize=9.0, fontweight="bold", color=INK)
    add_scale_bar(ax, 10.0, "10 m", loc=(0.035, 0.045), height=0.014,
                  text_color=INK, halo="white", fontsize=6.4)

    n_in = int(hit[in_zone(cols, rows)].sum())
    n_out = int(hit[~in_zone(cols, rows)].sum())
    tot_in = int(in_zone(cols, rows).sum())
    tot_out = int((~in_zone(cols, rows)).sum())
    ax.text(0.985, 0.018, f"inside gate line: {n_in}/{tot_in} pens affected\n"
                          f"outside: {n_out}/{tot_out} pens affected\n"
                          f"beige = never affected by day {HORIZON:.0f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor=HAIR, linewidth=0.7), zorder=9)

    sm = plt.cm.ScalarMappable(cmap=ONSET_CMAP,
                               norm=plt.Normalize(0.0, HORIZON))
    sm.set_array([])
    safe_colorbar(fig, sm, ax, label="onset day after index case (d)",
                  width=0.11, pad=0.10, labelsize=6.7,
                  ticks=[0, 6, 12, 18, 24])
    return hit


def epi_panel(ax, onset, hit):
    days = np.arange(0, int(HORIZON) + 1)
    new = np.histogram(onset[hit], bins=np.arange(-0.5, HORIZON + 1.5, 1.0))[0]
    cum = np.cumsum(new)
    ax.bar(days, new, width=0.82, color=BLUE, edgecolor="white", lw=0.35,
           zorder=3)
    axr = ax.twinx()
    axr.plot(days, cum, color=RED, lw=1.7, zorder=5)
    axr.set_ylim(0.0, max(6.0, cum.max() * 1.16))
    axr.set_ylabel("cumulative pens (n)", fontsize=7.6, color=RED)
    axr.tick_params(labelsize=6.8, colors=RED)
    axr.spines["top"].set_visible(False)

    pk = int(days[int(np.argmax(new))])
    ax.axvline(pk, color=RULE, lw=0.8, ls=(0, (2.2, 2.0)), zorder=2)
    ax.text(pk - 0.7, 0.955, f"peak incidence day {pk}",
            transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=6.5, color=RULE)
    ax.set_xlim(-0.8, HORIZON + 0.8)
    ax.set_ylim(0.0, float(new.max()) * 1.30)
    ax.set_xlabel("day after index case (d)", fontsize=7.6)
    ax.set_ylabel("new pens (n/d)", fontsize=7.6, color=BLUE)
    ax.tick_params(labelsize=6.8)
    ax.tick_params(axis="y", colors=BLUE)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_title("Epidemic curve: bars new, line cumulative", fontsize=8.4,
                 fontweight="bold", color=INK)
    return pk, int(cum[-1])


def gradient_panel(ax, xs, ys, hit, idx):
    src = idx[INDEX_PEN]
    d = np.hypot(xs - xs[src], ys - ys[src])
    edges_m = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 30.0])
    ctr, rate, lo, hi, ns = [], [], [], [], []
    for a, b in zip(edges_m[:-1], edges_m[1:]):
        m = (d >= a) & (d < b)
        if not m.any():
            continue
        p, l, h = wilson(int(hit[m].sum()), int(m.sum()))
        ctr.append(0.5 * (a + b))
        rate.append(100.0 * p)
        lo.append(100.0 * (p - l))
        hi.append(100.0 * (h - p))
        ns.append(int(m.sum()))
    ctr = np.array(ctr)
    rate = np.array(rate)
    ax.bar(ctr, rate, width=5.0, color=BEIGE, edgecolor=MUTED, lw=0.7,
           zorder=2)
    ax.errorbar(ctr, rate, yerr=[lo, hi], fmt="o", ms=3.6, mfc="white",
                mec=RED, mew=1.2, ecolor=RED, elinewidth=1.0, capsize=2.4,
                capthick=1.0, zorder=5)
    ax.plot(ctr, rate, color=RED, lw=1.4, zorder=4)
    for x, r, n in zip(ctr, rate, ns):
        ax.text(x, 3.0, f"n={n}", ha="center", va="bottom", fontsize=6.0,
                color=SUB, rotation=90)
    ax.set_xlim(0.0, edges_m[-1])
    ax.set_ylim(0.0, 118.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("distance from index pen (m)", fontsize=7.6)
    ax.set_ylabel("pens affected (%)", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.965, "bars are ring attack rates, whiskers 95 % Wilson",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.4,
            color=SUB)
    ax.set_title("Risk falls off with distance", fontsize=8.4,
                 fontweight="bold", color=INK)
    return rate[0], rate[-1]


def main():
    set_pub_style()
    rng = np.random.default_rng(20842)
    xs, ys, cols, rows = pen_grid()
    edges, long_range, idx = contact_edges(rng, cols, rows)
    onset = onset_days(edges, xs.size, idx[INDEX_PEN])

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.82, 0.98, 4.72, 4.58))
    ax_epi = axes_inches(fig, (7.16, 3.54, 2.78, 2.02))
    ax_grad = axes_inches(fig, (7.16, 0.98, 2.78, 2.02))

    hit = barn_panel(fig, ax_main, xs, ys, cols, rows, onset, edges,
                     long_range, idx)
    pk, total = epi_panel(ax_epi, onset, hit)
    near, far = gradient_panel(ax_grad, xs, ys, hit, idx)

    panel_labels_inches(fig, [ax_main, ax_epi, ax_grad], dx=-0.50, dy=0.14)
    savefig(fig, "842_vet_herd_disease_spread.png")


if __name__ == "__main__":
    main()
