"""MCTS search-tree growth at four simulation budgets, with the tree statistics.

One UCT run of 3200 simulations on a synthetic eight-ply decision tree is frozen
at 50 / 200 / 800 / 3200 simulations and drawn four times.  Node area encodes
the visit count N(s), node shade encodes the value estimate Q(s) on the batch
blue ladder, and the dark polyline is the principal variation -- the path a
reader gets by following argmax N(s) from the root.

The lower strip is not a second dataset: the branching factor and the maximum
depth are recomputed from the *same* tree after every simulation, and the four
ringed markers are exactly the four trees above.  Change the search and both
rows change together.  The branching factor is printed as the division that
produces it, so the reader can check it from the two integers next to it.

Single-hue blue ladder (BRIEF_WM_BLUE.md section 2); the one warm accent marks
the snapshot budgets and covers well under 1 % of the canvas.

Aspect: single row of four comparison panels plus a statistics strip
(DESIGN_SYSTEM section 1.2, exception E8) -- the delivered ratio is checked
against the 0.55-0.80 band in the catalog.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from pubstyle import set_pub_style, savefig

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"          # snapshot markers only
CARD = "#dfe7ef"            # luminance 0.90: safely below the 0.92 ink line
HAIR = "#aab7c4"

VAL_CMAP = LinearSegmentedColormap.from_list(
    "wm_mcts_val", [BLUE[6], BLUE[5], BLUE[4], BLUE[2], BLUE[1], BLUE[0]],
    N=256)

FW, FH = 10.0, 5.7
MAXD = 8                    # plies below the root
ROOT_ACTIONS = 6
SNAPS = (50, 200, 800, 3200)
C_UCT = 0.85


# ---------------------------------------------------------------------------
# A deterministic synthetic decision tree: every quantity below is a pure
# function of the action path, so the search is reproducible without storing
# the environment.
# ---------------------------------------------------------------------------
def _seed(path):
    s = 12345
    for a in path:
        s = (s * 7919 + a * 131 + 17) % 2147483647
    return s


def n_actions(path):
    if len(path) >= MAXD:
        return 0
    if not path:
        return ROOT_ACTIONS
    return int(np.random.default_rng(_seed(path)).integers(2, 5))


def edge_gain(path):
    """Reward increment collected by taking the last action of ``path``."""
    return float(np.random.default_rng(_seed(path) + 7).normal(0.045, 0.30))


def rollout(path, rng):
    g = sum(edge_gain(path[:k + 1]) for k in range(len(path)))
    d = len(path)
    while d < MAXD:
        d += 1
        g += float(rng.normal(0.045, 0.30))
    return float(np.clip(0.5 + 0.34 * g + rng.normal(0.0, 0.06), 0.0, 1.0))


def run_uct(n_sim, seed=7):
    """One UCT search; returns the node table plus a per-simulation history."""
    rng = np.random.default_rng(seed)
    nodes = [dict(parent=-1, path=(), depth=0, kids=[],
                  acts=n_actions(()), n=0, w=0.0)]
    hist = dict(links=[], internal=[], depth=[], nodes=[])
    snaps = {}
    for t in range(1, n_sim + 1):
        i = 0
        chain = [0]
        while True:
            nd = nodes[i]
            if len(nd["kids"]) < nd["acts"]:            # expand
                a = len(nd["kids"])
                child = dict(parent=i, path=nd["path"] + (a,),
                             depth=nd["depth"] + 1, kids=[],
                             acts=n_actions(nd["path"] + (a,)), n=0, w=0.0)
                nodes.append(child)
                nd["kids"].append(len(nodes) - 1)
                i = len(nodes) - 1
                chain.append(i)
                break
            if not nd["kids"]:                          # terminal ply
                break
            logN = np.log(max(nd["n"], 1))
            best, bs = nd["kids"][0], -1e18
            for k in nd["kids"]:
                kn = nodes[k]
                q = kn["w"] / kn["n"] if kn["n"] else 0.5
                s = q + C_UCT * np.sqrt(logN / max(kn["n"], 1))
                if s > bs:
                    bs, best = s, k
            i = best
            chain.append(i)
        v = rollout(nodes[i]["path"], rng)
        for k in chain:
            nodes[k]["n"] += 1
            nodes[k]["w"] += v

        links = sum(len(nd["kids"]) for nd in nodes)
        internal = sum(1 for nd in nodes if nd["kids"])
        hist["links"].append(links)
        hist["internal"].append(internal)
        hist["depth"].append(max(nd["depth"] for nd in nodes))
        hist["nodes"].append(len(nodes))
        if t in SNAPS:
            snaps[t] = dict(count=len(nodes),
                            n=np.array([nd["n"] for nd in nodes]),
                            q=np.array([nd["w"] / nd["n"] if nd["n"] else 0.5
                                        for nd in nodes]),
                            links=links, internal=internal,
                            depth=hist["depth"][-1])
    return nodes, hist, snaps


def radial_layout(nodes):
    """Angular slots sized by leaf count; radius is the ply index."""
    span = np.zeros(len(nodes), int)
    for i in range(len(nodes) - 1, -1, -1):
        span[i] = (sum(span[k] for k in nodes[i]["kids"])
                   if nodes[i]["kids"] else 1)
    ang = np.zeros(len(nodes))
    stack = [(0, 0.0, 2.0 * np.pi * 0.985)]
    while stack:
        i, a0, a1 = stack.pop()
        ang[i] = 0.5 * (a0 + a1)
        acc, tot = a0, float(span[i])
        for k in nodes[i]["kids"]:
            wdt = (a1 - a0) * span[k] / tot
            stack.append((k, acc, acc + wdt))
            acc += wdt
    r = np.array([nd["depth"] for nd in nodes], float) / MAXD
    return np.column_stack([r * np.sin(ang), r * np.cos(ang)])


def principal_variation(nodes, count, visits):
    """argmax-visit path through the nodes that exist in this snapshot."""
    path, i = [0], 0
    while True:
        kids = [k for k in nodes[i]["kids"] if k < count and visits[k] > 0]
        if not kids:
            return path
        i = max(kids, key=lambda k: visits[k])
        path.append(i)


def draw_tree(ax, nodes, pos, snap, budget, letter):
    ax.set_facecolor(CARD)
    ax.set_xlim(-1.14, 1.14)
    ax.set_ylim(-1.14, 1.14)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color(HAIR)

    cnt, vis, q = snap["count"], snap["n"], snap["q"]
    for k in range(1, MAXD + 1):
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.sin(th) * k / MAXD, np.cos(th) * k / MAXD, lw=0.45,
                color="#c6d2de", zorder=1)

    segs, lws = [], []
    for i in range(1, cnt):
        p = nodes[i]["parent"]
        segs.append([pos[p], pos[i]])
        lws.append(0.35 + 0.85 * np.sqrt(vis[i] / max(vis[0], 1)))
    ax.add_collection(LineCollection(segs, colors=BLUE[4], linewidths=lws,
                                     alpha=0.85, zorder=2))

    size = 2.0 + 46.0 * np.sqrt(vis[1:cnt] / max(vis[0], 1))
    ax.scatter(pos[1:cnt, 0], pos[1:cnt, 1], s=size, c=q[1:cnt],
               cmap=VAL_CMAP, vmin=0.28, vmax=0.72, linewidths=0.28,
               edgecolors="white", zorder=4)

    pv = principal_variation(nodes, cnt, vis)
    ax.plot(pos[pv, 0], pos[pv, 1], color=INK, lw=1.9, zorder=6,
            solid_capstyle="round")
    ax.plot(pos[pv[1:], 0], pos[pv[1:], 1], ls="none", marker="o", ms=3.0,
            mfc="white", mec=INK, mew=0.9, zorder=7)
    ax.add_patch(Circle((0, 0), 0.052, facecolor=BLUE[0], edgecolor="white",
                        linewidth=1.0, zorder=8))

    ax.text(0.5, 1.030, "%s  %d simulations" % (letter, budget),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6,
            fontweight="bold", color=INK)
    ax.text(0.5, -0.055, "%d nodes  ·  depth %d  ·  PV %d plies"
            % (cnt, snap["depth"], len(pv) - 1), transform=ax.transAxes,
            ha="center", va="top", fontsize=6.8, color="#4b5866")


def main():
    set_pub_style(base_font=8)
    nodes, hist, snaps = run_uct(SNAPS[-1])
    pos = radial_layout(nodes)

    sims = np.arange(1, SNAPS[-1] + 1)
    links = np.array(hist["links"], float)
    internal = np.array(hist["internal"], float)
    bf = links / internal
    depth = np.array(hist["depth"], float)

    # the four ringed markers below are the four trees above, by construction
    for t in SNAPS:
        assert snaps[t]["links"] == hist["links"][t - 1]
        assert snaps[t]["internal"] == hist["internal"][t - 1]

    n_links, n_int = int(links[-1]), int(internal[-1])
    bf_txt = "%.3f" % (n_links / n_int)
    assert abs(n_links / n_int - float(bf_txt)) < 5e-4, "branching not closed"

    fig = plt.figure(figsize=(FW, FH))
    pw = 2.24 / FW
    gap = 0.10 / FW
    left = (1.0 - 4 * pw - 3 * gap) / 2.0
    top_y, top_h = 2.62 / FH, 2.24 / FH
    bot_y, bot_h = 0.66 / FH, 1.45 / FH

    axes = [fig.add_axes([left + i * (pw + gap), top_y, pw, top_h])
            for i in range(4)]
    for ax, t, lab in zip(axes, SNAPS, ["(a)", "(b)", "(c)", "(d)"]):
        draw_tree(ax, nodes, pos, snaps[t], t, lab)

    axB = fig.add_axes([left, bot_y, 4 * pw + 3 * gap, bot_h])
    axB.set_facecolor("white")
    # the first nine simulations are still expanding the root's own children,
    # where the ratio is dominated by a single internal node; the curve starts
    # once the statistic means something
    s0 = 9
    axB.plot(sims[s0:], bf[s0:], color=BLUE[1], lw=1.5, zorder=4)
    axB.set_xscale("log")
    axB.set_xlim(sims[s0], SNAPS[-1] * 1.45)
    axB.set_ylim(1.4, max(bf[s0:].max(), 3.0) + 0.55)
    axB.set_xlabel("simulations executed (log scale, from %d)" % sims[s0],
                   fontsize=7.4, labelpad=1.6)
    axB.set_ylabel("branching factor $\\bar{b}$", fontsize=7.4, labelpad=2.0,
                   color=BLUE[1])
    axB.tick_params(labelsize=6.8, length=2.2, pad=1.4)
    axB.grid(True, lw=0.4, color="#e6ebf0")
    axB.set_axisbelow(True)
    for side in ("left", "bottom"):
        axB.spines[side].set_linewidth(0.7)
        axB.spines[side].set_color("#95a3b1")

    axD = axB.twinx()
    axD.plot(sims[s0:], depth[s0:], color=BLUE[3], lw=1.5, ls=(0, (4.0, 2.0)),
             zorder=3)
    axD.set_ylim(0, MAXD + 0.8)
    axD.set_yticks(range(0, MAXD + 1, 2))
    axD.set_ylabel("maximum depth (plies)", fontsize=7.4, labelpad=2.4,
                   color=BLUE[3])
    axD.tick_params(labelsize=6.8, length=2.2, pad=1.4)
    axD.spines["right"].set_visible(True)
    axD.spines["right"].set_linewidth(0.7)
    axD.spines["right"].set_color("#95a3b1")
    axD.spines["top"].set_visible(False)

    for t, lab in zip(SNAPS, ["(a)", "(b)", "(c)", "(d)"]):
        axB.plot([t], [bf[t - 1]], marker="o", ms=5.2, mfc="white",
                 mec=ACCENT, mew=1.3, ls="none", zorder=7)
        axB.text(t, 0.985, lab, transform=axB.get_xaxis_transform(),
                 ha="center", va="top", fontsize=6.8, color=ACCENT,
                 fontweight="bold", zorder=8,
                 bbox=dict(boxstyle="round,pad=0.16", fc="white",
                           ec="none", alpha=0.94))

    axB.text(0.988, 0.085, "at 3200 simulations   $\\bar{b}$ = %d child links"
             " / %d internal nodes = %s        max depth = %d plies"
             % (n_links, n_int, bf_txt, int(depth[-1])),
             transform=axB.transAxes, ha="right", va="bottom", fontsize=6.9,
             color=INK, zorder=9,
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#c3ccd4",
                       lw=0.6, alpha=0.96))
    axB.set_title("Tree statistics recomputed after every simulation of the "
                  "same run", fontsize=8.4, fontweight="bold", color=INK,
                  pad=3.0)

    # ---- shared value colourbar in the header -----------------------------
    cax = fig.add_axes([0.806, 0.948, 0.148, 0.015])
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=VAL_CMAP, norm=mpl.colors.Normalize(0.28, 0.72),
        orientation="horizontal", ticks=[0.30, 0.50, 0.70])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, color=INK, pad=1.2)
    cb.set_label("node value $Q(s)$", fontsize=6.6, color=INK, labelpad=1.6)

    handles = [
        Line2D([], [], color=BLUE[4], lw=1.1, label="expanded edge"),
        Line2D([], [], marker="o", ls="none", ms=4.4, mfc=BLUE[2],
               mec="white", label="node, area $\\propto$ visits $N(s)$"),
        Line2D([], [], color=INK, lw=1.9, marker="o", ms=3.4, mfc="white",
               mec=INK, label="principal variation (argmax $N$)"),
        Line2D([], [], color=BLUE[1], lw=1.5, label="branching factor"),
        Line2D([], [], color=BLUE[3], lw=1.5, ls=(0, (4.0, 2.0)),
               label="maximum depth"),
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc="white", mec=ACCENT,
               mew=1.3, label="snapshot budget"),
    ]

    savefig(fig, "1207_wm_plan_mcts_tree.png")


if __name__ == "__main__":
    main()
