"""One agent episode as a tool-call graph, with the same episode tallied.

A 170-step session is simulated over twelve tools arranged in five role
columns.  Every artist on this figure comes out of that one walk: the edge
widths are transition counts, the node radii are call counts, the node shade is
the mean latency of the calls that node actually served, and the dashed accent
self-loops are the retries a failing call triggered.  The two right-hand panels
re-tally the same episode, which is why the stacked bars add up to the total
printed in the header and the latency lollipops line up row-for-row with them.

The session total is computed from the *printed* per-tool means, so a reader
who multiplies a bar by its lollipop and sums the twelve products lands on the
number in the footer; the script asserts that this printed total stays within
half a percent of the exact one.

Single-hue blue ladder (BRIEF_WM_BLUE.md section 2); the one warm accent is
reserved for retries.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch

from pubstyle import set_pub_style, savefig

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
CARD = "#dfe7ef"
LIGHT = "#f4f8fb"
HAIR = "#aab7c4"

LAT_CMAP = LinearSegmentedColormap.from_list(
    "wm_lat", [BLUE[6], BLUE[5], BLUE[4], BLUE[3], BLUE[2], BLUE[1], BLUE[0]],
    N=256)

FW, FH = 10.4, 5.9
GW, GH = 11.8, 8.2                     # graph world, matched to the panel
N_STEPS = 170
MAX_RETRY = 2

ROLES = ["perceive", "retrieve", "reason", "act", "verify"]
COL_X = [1.35, 3.55, 5.90, 8.15, 10.45]
# name, column, y, base latency (s), failure probability
TOOLS = [
    ("camera.read", 0, 5.95, 0.12, 0.03),
    ("depth.scan", 0, 2.85, 0.21, 0.05),
    ("kb.search", 1, 7.00, 0.55, 0.08),
    ("doc.fetch", 1, 4.55, 0.86, 0.12),
    ("web.query", 1, 2.05, 1.42, 0.18),
    ("sim.rollout", 2, 6.60, 2.35, 0.06),
    ("cost.eval", 2, 4.20, 0.34, 0.02),
    ("code.run", 2, 1.70, 1.05, 0.14),
    ("arm.move", 3, 5.70, 1.60, 0.09),
    ("gripper.set", 3, 2.70, 0.28, 0.04),
    ("state.check", 4, 6.05, 0.19, 0.03),
    ("log.write", 4, 3.20, 0.07, 0.01),
]
SUCC = {
    0: {2: 2, 1: 3, 5: 2, 10: 1},
    1: {5: 3, 6: 2, 2: 1},
    2: {3: 3, 4: 2, 6: 2, 5: 1},
    3: {6: 2, 5: 2, 7: 1, 2: 1},
    4: {2: 2, 6: 1, 7: 1},
    5: {6: 4, 7: 1, 8: 2},
    6: {8: 3, 5: 2, 9: 1, 10: 1},
    7: {6: 2, 5: 1, 10: 1},
    8: {9: 3, 10: 2, 0: 1},
    9: {10: 3, 8: 1, 11: 1},
    10: {11: 2, 0: 2, 5: 1, 8: 1},
    11: {0: 3, 2: 1},
}


def simulate(seed=1616):
    rng = np.random.default_rng(seed)
    n = len(TOOLS)
    counts = np.zeros(n, int)
    retries = np.zeros(n, int)
    lat = [[] for _ in range(n)]
    edges = {}
    i = 0
    for _ in range(N_STEPS):
        _, _, _, base, pf = TOOLS[i]
        attempts = 1
        while attempts <= MAX_RETRY and rng.random() < pf:
            attempts += 1
        retries[i] += attempts - 1
        for _a in range(attempts):
            counts[i] += 1
            lat[i].append(base * float(np.exp(rng.normal(0.0, 0.26))))
        opts = list(SUCC[i].items())
        w = np.array([v for _, v in opts], float)
        j = opts[int(rng.choice(len(opts), p=w / w.sum()))][0]
        edges[(i, j)] = edges.get((i, j), 0) + 1
        i = j
    means = np.array([float(np.mean(v)) for v in lat])
    return counts, retries, means, edges


def main():
    set_pub_style(base_font=8)
    counts, retries, means, edges = simulate()
    n = len(TOOLS)
    names = [t[0] for t in TOOLS]

    total = int(counts.sum())
    n_retry = int(retries.sum())
    assert total == N_STEPS + n_retry, "call tally does not reconcile"

    shown = ["%.2f" % v for v in means]
    t_shown = float(sum(c * float(s) for c, s in zip(counts, shown)))
    t_true = float((counts * means).sum())
    assert abs(t_shown / t_true - 1.0) < 5e-3, "printed session time drifts"

    fig = plt.figure(figsize=(FW, FH))
    axG = fig.add_axes([0.52 / FW, 1.00 / FH, 5.90 / FW, 4.10 / FH])
    axB = fig.add_axes([7.62 / FW, 3.20 / FH, 2.36 / FW, 1.90 / FH])
    axL = fig.add_axes([7.62 / FW, 1.00 / FH, 2.36 / FW, 1.70 / FH])

    # ---- (a) the call graph ------------------------------------------------
    axG.set_facecolor(CARD)
    axG.set_xlim(0, GW)
    axG.set_ylim(0, GH)
    axG.set_aspect("equal")
    axG.set_xticks([])
    axG.set_yticks([])
    for sp in axG.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.9)
        sp.set_color("#7f8c99")

    pos = np.array([[COL_X[t[1]], t[2]] for t in TOOLS], float)
    rad = 0.20 + 0.34 * np.sqrt(counts / counts.max())
    lnorm = mpl.colors.Normalize(means.min(), means.max())

    # role names sit at the foot of the columns: the top strip is taken by the
    # colourbar and the call tally
    for c, x in enumerate(COL_X):
        axG.plot([x, x], [0.78, 7.45], color="#c6d2de", lw=0.7, zorder=1)
        axG.text(x, 0.42, ROLES[c], ha="center", va="center", fontsize=7.2,
                 color="#4b5866", fontweight="bold", zorder=2)

    emax = max(edges.values())
    for (i, j), w in sorted(edges.items(), key=lambda kv: kv[1]):
        d = pos[j] - pos[i]
        L = float(np.hypot(*d))
        u = d / L
        p0 = pos[i] + u * (rad[i] + 0.04)
        p1 = pos[j] - u * (rad[j] + 0.10)
        back = pos[j][0] < pos[i][0] - 0.01
        curve = 0.30 if back else 0.13
        axG.add_patch(FancyArrowPatch(
            tuple(p0), tuple(p1), arrowstyle="-|>",
            mutation_scale=6.5 + 4.0 * w / emax,
            linewidth=0.45 + 2.30 * w / emax,
            color=BLUE[2] if w >= 0.45 * emax else BLUE[4],
            connectionstyle="arc3,rad=%.2f" % curve, shrinkA=0, shrinkB=0,
            alpha=0.92, zorder=3))

    for i in range(n):
        if retries[i] == 0:
            continue
        x, y = pos[i]
        axG.add_patch(FancyArrowPatch(
            (x - 0.20, y + rad[i] * 0.86), (x + 0.20, y + rad[i] * 0.86),
            arrowstyle="-|>", mutation_scale=6.0,
            linewidth=0.6 + 1.5 * retries[i] / max(retries.max(), 1),
            color=ACCENT, linestyle=(0, (2.2, 1.4)),
            connectionstyle="arc3,rad=-2.6", shrinkA=0, shrinkB=0, zorder=6))

    for i in range(n):
        axG.add_patch(Circle(tuple(pos[i]), rad[i],
                             facecolor=LAT_CMAP(lnorm(means[i])),
                             edgecolor="white", linewidth=1.1, zorder=7))
        axG.text(pos[i][0], pos[i][1] - rad[i] - 0.12, names[i], ha="center",
                 va="top", fontsize=6.6, color=INK, zorder=8,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                           alpha=0.90))

    axG.text(0.982, 0.975, "%d tool calls  =  %d first attempts  +  %d "
             "retries" % (total, N_STEPS, n_retry), transform=axG.transAxes,
             ha="right", va="top", fontsize=7.0, color=INK, zorder=10,
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c3ccd4",
                       lw=0.6, alpha=0.96))
    axG.set_title("Tool-call graph of one episode: node area is call count, "
                  "shade is mean latency", fontsize=8.8, fontweight="bold",
                  color=INK, pad=4.0)

    cax = axG.inset_axes([0.032, 0.940, 0.215, 0.026], zorder=12)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=LAT_CMAP, norm=lnorm,
                                   orientation="horizontal",
                                   ticks=[means.min(), means.max()])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.set_xticklabels(["%.2f" % means.min(), "%.2f" % means.max()])
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, color=INK, pad=1.2)
    cb.set_label("mean latency (s)", fontsize=6.6, color=INK, labelpad=1.6)

    handles = [
        Line2D([], [], color=BLUE[2], lw=2.4, label="frequent transition"),
        Line2D([], [], color=BLUE[4], lw=0.9, label="occasional transition"),
        Line2D([], [], color=ACCENT, lw=1.4, ls=(0, (2.2, 1.4)),
               label="retry after a failed call"),
    ]
    leg = axG.legend(handles=handles, loc="lower right", fontsize=6.9,
                     bbox_to_anchor=(0.995, 0.108), frameon=True,
                     handlelength=2.1, labelspacing=0.36, borderpad=0.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#c3ccd4")
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_alpha(0.95)
    leg.set_zorder(11)

    # ---- (b) call counts, (c) mean latency --------------------------------
    order = np.argsort(counts)                 # barh draws bottom-up
    y = np.arange(n)
    first = counts - retries

    axB.set_facecolor(LIGHT)
    axB.barh(y, first[order], height=0.68, color=BLUE[2], edgecolor="white",
             linewidth=0.6, zorder=4)
    axB.barh(y, retries[order], height=0.68, left=first[order],
             color=BLUE[6], edgecolor=ACCENT, linewidth=0.7, hatch="///",
             zorder=4)
    for k, idx in enumerate(order):
        axB.text(counts[idx] + 0.9, k, "%d" % counts[idx], ha="left",
                 va="center", fontsize=6.6, color=INK, zorder=6)
    axB.set_yticks(y)
    axB.set_yticklabels([names[i] for i in order], fontsize=6.8)
    axB.set_xlim(0, counts.max() * 1.24)
    axB.set_ylim(-0.7, n - 0.3)
    axB.set_xlabel("calls in the episode", fontsize=7.4, labelpad=1.8)
    axB.tick_params(labelsize=6.8, length=2.2, pad=1.4)
    axB.grid(True, axis="x", lw=0.4, color="#dde5ee")
    axB.set_axisbelow(True)
    for side in ("left", "bottom"):
        axB.spines[side].set_linewidth(0.7)
        axB.spines[side].set_color("#95a3b1")
    axB.set_title("Calls per tool", fontsize=8.2, fontweight="bold",
                  color=INK, pad=3.0)
    lb = axB.legend(handles=[Patch(facecolor=BLUE[2], edgecolor="white",
                                   label="first attempt"),
                             Patch(facecolor=BLUE[6], edgecolor=ACCENT,
                                   hatch="///", label="retry")],
                    loc="lower right", fontsize=6.5, frameon=True,
                    handlelength=1.5, labelspacing=0.30, borderpad=0.4)
    lb.get_frame().set_facecolor("white")
    lb.get_frame().set_edgecolor("#c3ccd4")
    lb.get_frame().set_linewidth(0.6)

    axL.set_facecolor(LIGHT)
    for k, idx in enumerate(order):
        axL.plot([0, means[idx]], [k, k], color=BLUE[5], lw=2.0,
                 solid_capstyle="round", zorder=4)
        axL.plot([means[idx]], [k], marker="o", ms=4.6,
                 mfc=LAT_CMAP(lnorm(means[idx])), mec=BLUE[0], mew=0.8,
                 ls="none", zorder=5)
        axL.text(means[idx] + 0.075, k, shown[idx], ha="left", va="center",
                 fontsize=6.5, color=INK, zorder=6)
    axL.set_yticks(y)
    axL.set_yticklabels([])
    axL.set_xlim(0, means.max() * 1.34)
    axL.set_ylim(-0.7, n - 0.3)
    axL.set_xlabel("mean latency per call (s)", fontsize=7.4, labelpad=1.8)
    axL.tick_params(labelsize=6.8, length=2.2, pad=1.4)
    axL.grid(True, axis="x", lw=0.4, color="#dde5ee")
    axL.set_axisbelow(True)
    for side in ("left", "bottom"):
        axL.spines[side].set_linewidth(0.7)
        axL.spines[side].set_color("#95a3b1")
    axL.set_title("Mean latency, same row order", fontsize=8.2,
                  fontweight="bold", color=INK, pad=3.0)

    savefig(fig, "1211_wm_plan_tool_graph.png")


if __name__ == "__main__":
    main()
