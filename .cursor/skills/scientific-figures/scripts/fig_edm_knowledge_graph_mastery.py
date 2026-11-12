"""Prerequisite graph with mastery painted on it, and where to go next.

The graph is a DAG by construction - every edge runs from a lower layer to a
strictly higher one - so "prerequisite" cannot quietly become a cycle.  Within
a layer the vertical order is not typed in: a barycentre sweep reorders each
layer against its neighbours and the number of edge crossings is counted
before and after, so the layout claim in the footnote is measured.  A concept
is recommended only when the graph says it is reachable, i.e. every
prerequisite is already above threshold while the concept itself is not, and
the three recommendations are ranked by how much of the rest of the graph they
unlock rather than by how close they are to being finished.
"""
import textwrap

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from pubstyle import set_pub_style, get_palette, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"
ACCENT = "#b23a48"

FIGW, FIGH = 10.4, 6.4
MAIN = (0.78, 1.05, 5.45, 4.35)
AUX_T = (7.05, 3.62, 3.00, 1.78)
AUX_B = (7.05, 1.05, 3.00, 1.78)

MASTERED = 0.70           # mastery threshold
N_WEEK = 16               # weeks of practice history in panel (c)

# Layered curriculum: index -> (short label, layer).  Edges may only go from a
# lower layer to a higher one, which is what makes the graph acyclic.
NODES = [
    ("integers", 0), ("fractions", 0), ("number line", 0),
    ("decimals", 1), ("ratios", 1), ("negative\nnumbers", 1),
    ("percentages", 2), ("linear\nexpressions", 2), ("coordinate\nplane", 2),
    ("proportional\nreasoning", 3), ("linear\nequations", 3), ("slope", 3),
    ("systems of\nequations", 4), ("linear\nfunctions", 4),
    ("quadratics", 5), ("function\ntransforms", 5),
]
EDGES = [
    (0, 3), (1, 3), (1, 4), (0, 5), (2, 5), (2, 8),
    (3, 6), (4, 6), (3, 7), (5, 7), (5, 8),
    (6, 9), (4, 9), (7, 10), (8, 11), (7, 11),
    (9, 12), (10, 12), (11, 13), (10, 13),
    (12, 14), (13, 14), (13, 15), (11, 15),
]


def layered_positions(nodes, edges, sweeps=6, y_lo=0.45, y_hi=2.95):
    """Assign (x, y) per node; barycentre sweeps thin out the crossings."""
    layers = sorted({lay for _, lay in nodes})
    members = {lay: [i for i, (_, l) in enumerate(nodes) if l == lay]
               for lay in layers}
    order = {lay: list(v) for lay, v in members.items()}
    up = {i: [] for i in range(len(nodes))}
    down = {i: [] for i in range(len(nodes))}
    for a, b in edges:
        down[a].append(b)
        up[b].append(a)

    def ys(lay):
        k = len(order[lay])
        if k == 1:
            return {order[lay][0]: (y_lo + y_hi) / 2.0}
        grid = np.linspace(y_hi, y_lo, k)
        return {n: float(grid[j]) for j, n in enumerate(order[lay])}

    pos = {}
    for _ in range(sweeps):
        pos = {}
        for lay in layers:
            pos.update(ys(lay))
        for direction in (1, -1):
            seq = layers[1:] if direction > 0 else layers[-2::-1]
            for lay in seq:
                nb = up if direction > 0 else down
                key = {n: (np.mean([pos[m] for m in nb[n]]) if nb[n]
                           else pos[n]) for n in order[lay]}
                order[lay] = sorted(order[lay], key=lambda n: -key[n])
                pos.update(ys(lay))
    x = {i: float(lay) for i, (_, lay) in enumerate(nodes)}
    return x, pos


def count_crossings(edges, x, y):
    """Pairs of edge segments that actually intersect on the page."""
    def seg(a, b):
        return np.array([x[a], y[a]]), np.array([x[b], y[b]])

    def side(p, q, r):
        return np.sign(np.cross(q - p, r - p))

    n = 0
    for i in range(len(edges)):
        p1, p2 = seg(*edges[i])
        for j in range(i + 1, len(edges)):
            if set(edges[i]) & set(edges[j]):
                continue
            q1, q2 = seg(*edges[j])
            if (side(p1, p2, q1) * side(p1, p2, q2) < 0 and
                    side(q1, q2, p1) * side(q1, q2, p2) < 0):
                n += 1
    return n


def descendants(edges, n_nodes):
    """Everything reachable downstream of each node."""
    down = {i: [] for i in range(n_nodes)}
    for a, b in edges:
        down[a].append(b)
    out = {}
    for i in range(n_nodes):
        seen, stack = set(), list(down[i])
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            stack.extend(down[k])
        out[i] = seen
    return out


def state(seed=904):
    """Mastery in 0..1 and practice counts, decaying with curriculum depth."""
    rng = np.random.default_rng(seed)
    lay = np.array([l for _, l in NODES], dtype=float)
    mastery = np.clip(0.97 - 0.145 * lay + rng.normal(0.0, 0.075, len(NODES)),
                      0.05, 0.99)
    attempts = np.clip(
        (58 - 7.5 * lay + rng.normal(0.0, 7.0, len(NODES))).round(), 6, 70)
    week = np.clip((1.6 + 2.3 * lay + rng.normal(0.0, 1.1, len(NODES))
                    ).round(), 1, N_WEEK)
    return mastery, attempts, week


def main():
    set_pub_style()
    mastery, attempts, week = state()
    n_nodes = len(NODES)
    labels = [nm for nm, _ in NODES]

    x, y = layered_positions(NODES, EDGES)
    y0 = {i: v for i, v in y.items()}
    x_raw = {i: float(l) for i, (_, l) in enumerate(NODES)}
    naive = {}
    for lay in sorted({l for _, l in NODES}):
        mem = [i for i, (_, l) in enumerate(NODES) if l == lay]
        grid = (np.linspace(2.95, 0.45, len(mem)) if len(mem) > 1
                else np.array([1.70]))
        naive.update({n: float(g) for n, g in zip(mem, grid)})
    cross_before = count_crossings(EDGES, x_raw, naive)
    cross_after = count_crossings(EDGES, x, y0)

    ok = mastery >= MASTERED
    prereq = {i: [a for a, b in EDGES if b == i] for i in range(n_nodes)}
    unlocked = np.array([(not ok[i]) and all(ok[p] for p in prereq[i])
                         for i in range(n_nodes)])
    desc = descendants(EDGES, n_nodes)
    rank = sorted(np.flatnonzero(unlocked),
                  key=lambda i: (-len(desc[i]), NODES[i][1], -mastery[i]))
    path = rank[:3]

    blues = get_palette("cb_sequential_blue", 256)
    greys = get_palette("neutral", 256)

    def fill(i):
        """Sequential blue once mastered, a grey ramp while it is not."""
        if ok[i]:
            t = np.clip((mastery[i] - MASTERED) / (1.0 - MASTERED), 0.0, 1.0)
            return blues[int(60 + 190 * t)]
        return greys[int(20 + 150 * np.clip(mastery[i] / MASTERED, 0, 1))]

    def on_fill(i):
        """Legible text colour for whatever fill that node ended up with."""
        r, g, b = mpl.colors.to_rgb(fill(i))
        return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax_g = axes_inches(fig, MAIN)
    ax_b = axes_inches(fig, AUX_T)
    ax_t = axes_inches(fig, AUX_B)

    # ---- main: the prerequisite DAG ---------------------------------------
    for a, b in EDGES:
        ax_g.add_patch(FancyArrowPatch(
            (x[a], y[a]), (x[b], y[b]), arrowstyle="-|>", mutation_scale=7.0,
            linewidth=0.7, color=HAIR if not ok[a] else "#b6c3d2",
            connectionstyle="arc3,rad=0.045", shrinkA=9.0, shrinkB=10.0,
            zorder=2))
    sizes = 70.0 + 5.2 * attempts
    # Marker radius in data units, so the label clears the disc it belongs to
    # whatever the practice count made of it.
    unit = 6.17 / MAIN[2]                       # data units per inch, x
    radius = np.sqrt(sizes / np.pi) / 72.0 * unit
    for i in range(n_nodes):
        ax_g.scatter([x[i]], [y[i]], s=sizes[i], facecolor=fill(i),
                     edgecolor="white", linewidth=0.9, zorder=4)
        ax_g.text(x[i], y[i], f"{100 * mastery[i]:.0f}", ha="center",
                  va="center", fontsize=6, zorder=6, color=on_fill(i))
        ax_g.text(x[i], y[i] - radius[i] - 0.075, labels[i], ha="center",
                  va="top", fontsize=6.0, color=INK if ok[i] else SUB,
                  linespacing=1.25, zorder=5)
    for i in np.flatnonzero(unlocked):
        ax_g.scatter([x[i]], [y[i]], s=sizes[i] + 130.0, facecolor="none",
                     edgecolor=ACCENT, linewidth=1.3, zorder=5)
    for k, i in enumerate(path):
        ax_g.text(x[i] - 0.155, y[i] + 0.155, f"{k + 1}", ha="center",
                  va="center", fontsize=6.4, color="white", zorder=7,
                  bbox=dict(boxstyle="circle,pad=0.20", facecolor=ACCENT,
                            edgecolor="white", linewidth=0.6))

    ax_g.set_xlim(-0.55, 5.62)
    ax_g.set_ylim(-0.72, 4.05)
    ax_g.axis("off")
    for lay in range(6):
        ax_g.text(lay, 3.90, f"layer {lay + 1}", ha="center", va="bottom",
                  fontsize=6.6, color=MUTED)
    ax_g.annotate("", xy=(5.30, 3.78), xytext=(-0.30, 3.78),
                  arrowprops=dict(arrowstyle="-|>", color=HAIR, lw=0.9))
    ax_g.text(-0.30, 3.68, "prerequisite depth: foundation to advanced",
              ha="left", va="top", fontsize=6.6, color=MUTED)

    # mastery ramp and size key, drawn in data coordinates
    for j in range(40):
        t = j / 39.0
        ax_g.add_patch(Rectangle((-0.45 + 0.026 * j, -0.40), 0.027, 0.10,
                                 facecolor=blues[int(60 + 190 * t)],
                                 edgecolor="none", zorder=4))
    ax_g.text(-0.45, -0.26, f"mastery {100 * MASTERED:.0f} %", ha="left",
              va="bottom", fontsize=6.0, color=MUTED)
    ax_g.text(0.60, -0.26, "100 %", ha="right", va="bottom", fontsize=6.0,
              color=MUTED)
    ax_g.add_patch(Rectangle((0.80, -0.40), 0.30, 0.10, facecolor=greys[110],
                             edgecolor="none", zorder=4))
    ax_g.text(1.16, -0.35, "below threshold", ha="left", va="center",
              fontsize=6.0, color=MUTED)
    for j, att in enumerate([15, 40, 65]):
        ax_g.scatter([3.00 + 0.44 * j], [-0.35], s=70.0 + 5.2 * att,
                     facecolor="#dfe6ee", edgecolor="white", linewidth=0.8,
                     zorder=4)
        ax_g.text(3.00 + 0.44 * j, -0.68, f"{att}", ha="center", va="bottom",
                  fontsize=6.0, color=MUTED)
    ax_g.text(4.36, -0.35, "practice attempts\n(count)", ha="left",
              va="center", fontsize=6.0, color=MUTED, linespacing=1.3)

    seq = "  ->  ".join(f"{k + 1} {labels[i]}".replace("\n", " ")
                        for k, i in enumerate(path))
    ax_g.text(-0.50, 3.22,
              f"recommended next, ranked by how much\neach one unlocks:  "
              f"{seq}",
              ha="left", va="bottom", fontsize=6.9, color=ACCENT,
              linespacing=1.45)
    ax_g.text(5.58, 3.22,
              f"{int(unlocked.sum())} concepts unlocked but not mastered "
              f"(red ring);\n{int((~ok).sum())} of {n_nodes} sit below the "
              f"{100 * MASTERED:.0f} % threshold in total",
              ha="right", va="bottom", fontsize=6.5, color=MUTED,
              linespacing=1.4)
    ax_g.set_title("A prerequisite DAG, painted with how far the learner has "
                   "actually got",
                   fontsize=9.0, fontweight="bold", color=INK, pad=6.0)

    # ---- aux (b): mastery ranking -----------------------------------------
    idx = np.argsort(mastery)
    ypos = np.arange(n_nodes)
    ax_b.barh(ypos, 100 * mastery[idx], height=0.66,
              color=[fill(int(i)) for i in idx], edgecolor="white",
              linewidth=0.4, zorder=3)
    ax_b.axvline(100 * MASTERED, color=ACCENT, lw=0.9, ls=(0, (3.2, 2.0)),
                 zorder=4)
    ax_b.text(100 * MASTERED + 1.5, -0.7,
              f"threshold {100 * MASTERED:.0f} %", ha="left", va="bottom",
              fontsize=6.2, color=ACCENT)
    for j, i in enumerate(idx):
        ax_b.text(1.5, j, labels[int(i)].replace("\n", " "), ha="left",
                  va="center", fontsize=6, color=on_fill(int(i)), zorder=5)
    ax_b.set_xlim(0.0, 100.0)
    ax_b.set_ylim(-0.8, n_nodes - 0.2)
    ax_b.set_yticks([])
    ax_b.set_xticks(np.arange(0, 101, 25))
    ax_b.set_xlabel("mastery (%)", fontsize=7.4)
    ax_b.tick_params(labelsize=6.6)
    ax_b.spines["left"].set_visible(False)
    ax_b.grid(axis="x", lw=0.5, color="#eeeeee")
    ax_b.set_title(f"{int((~ok).sum())} of {n_nodes} concepts below "
                   f"threshold", fontsize=8.5, fontweight="bold", color=INK)

    # ---- aux (c): mastery accumulated over the term -----------------------
    weeks = np.arange(0, N_WEEK + 1)
    cum = np.array([int(np.sum((week <= w) & ok)) for w in weeks])
    ax_t.fill_between(weeks, 0, cum, step="post", color="#cfe1ee", alpha=0.75,
                      lw=0.0, zorder=2)
    ax_t.step(weeks, cum, where="post", color="#2e74a8", lw=1.7, zorder=4)
    ax_t.axhline(n_nodes, color=GREY, lw=0.8, ls=(0, (3.2, 2.0)), zorder=3)
    ax_t.text(0.3, n_nodes - 0.35, f"all {n_nodes} concepts in the graph",
              ha="left", va="top", fontsize=6.2, color=GREY)
    ax_t.plot([N_WEEK], [cum[-1]], marker="o", ms=4.2, mfc="white",
              mec="#2e74a8", mew=1.2, zorder=5)
    ax_t.annotate(f"{cum[-1]} mastered by week {N_WEEK};\n"
                  f"{n_nodes - cum[-1]} still open",
                  xy=(N_WEEK, cum[-1]), xytext=(-6.0, -4.0),
                  textcoords="offset points", ha="right", va="top",
                  fontsize=6.3, color=SUB, linespacing=1.4)
    ax_t.set_xlim(0.0, N_WEEK)
    ax_t.set_ylim(0.0, n_nodes * 1.18)
    ax_t.set_xticks(np.arange(0, N_WEEK + 1, 4))
    ax_t.set_yticks(np.arange(0, n_nodes + 1, 4))
    ax_t.set_xlabel("week of the course (weeks)", fontsize=7.4)
    ax_t.set_ylabel("concepts mastered (count)", fontsize=7.4)
    ax_t.tick_params(labelsize=6.6)
    ax_t.grid(axis="y", lw=0.5, color="#eeeeee")
    ax_t.set_title("Mastery accumulates in steps, not smoothly",
                   fontsize=8.5, fontweight="bold", color=INK)


    note = (
        f"Directed acyclic prerequisite graph of {n_nodes} concepts and "
        f"{len(EDGES)} edges over {max(l for _, l in NODES) + 1} curriculum "
        f"layers; every edge runs strictly forward, so no cycle is possible. "
        f"Within-layer order comes from {6} barycentre sweeps, which cut the "
        f"drawn edge crossings from {cross_before} to {cross_after}. Node "
        f"fill encodes mastery (blue ramp above the "
        f"{100 * MASTERED:.0f} % threshold, grey ramp below it), node area "
        f"encodes practice attempts ({int(attempts.min())} to "
        f"{int(attempts.max())} per concept) and a red ring marks a concept "
        f"whose prerequisites are all mastered while it is not - "
        f"{int(unlocked.sum())} of the {int((~ok).sum())} unmastered "
        f"concepts qualify. The three recommendations are those unlocked "
        f"concepts ranked by the number of downstream concepts they release "
        f"({', '.join(str(len(desc[i])) for i in path)} respectively).")
    fig.text(0.5, 0.16 / FIGH, textwrap.fill(note, 195), ha="center",
             va="bottom", fontsize=6.5, color=MUTED, linespacing=1.5)

    panel_labels_inches(fig, [ax_g, ax_b, ax_t],
                        labels=["(a)", "(b)", "(c)"], dx=-0.56, dy=0.12)
    savefig(fig, "898_edm_knowledge_graph_mastery.png")


if __name__ == "__main__":
    main()
