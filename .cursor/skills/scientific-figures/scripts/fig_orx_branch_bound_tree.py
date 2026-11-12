"""Branch-and-bound search tree for a two-variable integer programme.

A depth-first branch-and-bound is actually executed: every node solves its own
LP relaxation with ``scipy.optimize.linprog`` under the accumulated variable
bounds, branches on the most fractional variable, and is pruned when the
relaxation is infeasible, integral, or bounded above by the incumbent.  The
tree drawing, the branch labels, the convergence curve and the nodes-per-depth
histogram are all read back out of the recorded search log, so re-running with
different coefficients redraws a different tree.
"""
import textwrap
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.optimize import linprog

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, inch_axes, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PALE = "#e2e2e2"

W, H = 10.4, 6.4

# maximise c . x  s.t.  A x <= b,  x >= 0 and integral
OBJ = np.array([5.0, 4.0])
A = np.array([[6.0, 4.0], [1.0, 2.0]])
B = np.array([25.0, 6.0])
VARS = ["x_1", "x_2"]
TOL = 1e-7

FOOT = ("Each node solves its own LP relaxation with scipy.optimize.linprog "
        "under the bounds inherited from its branch, then branches on the "
        "most fractional variable; nodes are pruned when the relaxation is "
        "infeasible or cannot beat the incumbent. The convergence panel plots "
        "the incumbent value and the best remaining relaxation bound after "
        "each node is processed, so the gap closes only when the search "
        "proves optimality.")


def relax(bounds):
    r = linprog(-OBJ, A_ub=A, b_ub=B, bounds=bounds, method="highs")
    if not r.success:
        return None, None
    return r.x, float(OBJ @ r.x)


def branch_and_bound():
    """Depth-first search; returns the node list and the per-node bound log."""
    root = dict(id=0, parent=None, depth=0, label=None,
                bounds=[(0.0, None), (0.0, None)])
    nodes = [root]
    stack = [root]
    incumbent = -np.inf
    best_x = None
    history = []

    while stack:
        node = stack.pop()
        x, z = relax(node["bounds"])
        node["x"] = x
        node["z"] = z
        if x is None:
            node["status"] = "infeasible"
        elif z <= incumbent + TOL:
            node["status"] = "bounded"
        else:
            frac = np.abs(x - np.round(x))
            if frac.max() <= 1e-6:
                node["status"] = "integer"
                incumbent = z
                best_x = np.round(x)
            else:
                node["status"] = "branched"
                j = int(np.argmax(frac))
                lo, hi = np.floor(x[j]), np.ceil(x[j])
                kids = []
                for side, (l, u) in (("le", (None, lo)), ("ge", (hi, None))):
                    bd = list(node["bounds"])
                    olo, ohi = bd[j]
                    if side == "le":
                        bd[j] = (olo, u if ohi is None else min(ohi, u))
                        lab = rf"${VARS[j]} \leq {lo:.0f}$"
                    else:
                        bd[j] = (max(olo, l), ohi)
                        lab = rf"${VARS[j]} \geq {hi:.0f}$"
                    kid = dict(id=len(nodes), parent=node["id"],
                               depth=node["depth"] + 1, label=lab, bounds=bd)
                    nodes.append(kid)
                    kids.append(kid)
                stack.extend(kids[::-1])
        node["order"] = len(history)
        open_bounds = [nodes[s["parent"]]["z"] for s in stack
                       if nodes[s["parent"]]["z"] is not None]
        upper = max(open_bounds) if open_bounds else incumbent
        upper = max(upper, incumbent)
        history.append((len(history) + 1, incumbent, upper))

    return nodes, np.array(history, dtype=float), incumbent, best_x


def layout(nodes, x_span, y_top, y_step):
    kids = {n["id"]: [] for n in nodes}
    for n in nodes:
        if n["parent"] is not None:
            kids[n["parent"]].append(n["id"])
    leaves = []

    def walk(i):
        if not kids[i]:
            leaves.append(i)
        for k in kids[i]:
            walk(k)

    walk(0)
    pitch = x_span / max(len(leaves), 1)
    pos = {}
    for s, i in enumerate(leaves):
        pos[i] = (s + 0.5) * pitch
    for n in sorted(nodes, key=lambda d: -d["depth"]):
        i = n["id"]
        if kids[i]:
            pos[i] = float(np.mean([pos[k] for k in kids[i]]))
    return {n["id"]: (pos[n["id"]], y_top - n["depth"] * y_step)
            for n in nodes}, kids


def optimal_path(nodes, best_z):
    for n in nodes:
        if n["status"] == "integer" and abs(n["z"] - best_z) < 1e-6:
            chain = [n["id"]]
            while nodes[chain[-1]]["parent"] is not None:
                chain.append(nodes[chain[-1]]["parent"])
            return set(chain), n["id"]
    return set(), None


FACE = {"branched": lighten(BLUE, 0.74), "integer": lighten(GREEN, 0.66),
        "infeasible": PALE, "bounded": PALE}
EDGE = {"branched": BLUE, "integer": GREEN, "infeasible": GREY,
        "bounded": GREY}


def draw_tree(ax, nodes, pos, kids, path, best_id, bw, bh):
    for n in nodes:
        i = n["id"]
        if n["parent"] is None:
            continue
        x0, y0 = pos[n["parent"]]
        x1, y1 = pos[i]
        on = i in path and n["parent"] in path
        ax.plot([x0, x0, x1, x1], [y0 - bh / 2, (y0 + y1) / 2,
                                   (y0 + y1) / 2, y1 + bh / 2],
                color=RED if on else HAIR, lw=1.7 if on else 0.9,
                solid_joinstyle="round", zorder=2 if on else 1)
        ax.text(x1 + (0.055 if x1 >= x0 else -0.055), (y0 + y1) / 2 + 0.045,
                n["label"], ha="left" if x1 >= x0 else "right", va="bottom",
                fontsize=6.2, color=RED if on else SUB, zorder=5)

    for n in nodes:
        i = n["id"]
        x, y = pos[i]
        st = n["status"]
        dead = st in ("infeasible", "bounded")
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=FACE[st], edgecolor=EDGE[st],
            linewidth=1.7 if i == best_id else 0.9, mutation_aspect=1.0,
            zorder=3))
        head = f"N{i}"
        if n["z"] is None:
            body = "LP infeasible"
        else:
            body = f"LP {n['z']:.2f}"
        ax.text(x, y + 0.115, head, ha="center", va="center", fontsize=6.4,
                color=SUB, zorder=6)
        ax.text(x, y - 0.015, body, ha="center", va="center", fontsize=6.8,
                color=INK, fontweight="bold" if i == best_id else "normal",
                zorder=6)
        if n["z"] is not None:
            tail = (f"({n['x'][0]:.2f}, {n['x'][1]:.2f})" if st == "branched"
                    else f"({n['x'][0]:.0f}, {n['x'][1]:.0f})")
        else:
            tail = "pruned"
        ax.text(x, y - 0.145, tail, ha="center", va="center", fontsize=6.0,
                color=MUTED, zorder=6)
        if dead:
            rx, ry = bw * 0.40, bh * 0.36
            for s in (+1, -1):
                ax.plot([x - rx, x + rx], [y - s * ry, y + s * ry],
                        color=GREY, lw=1.1, alpha=0.55, zorder=7,
                        solid_capstyle="round")


def draw_convergence(ax, hist, best_z):
    step, lb, ub = hist[:, 0], hist[:, 1], hist[:, 2]
    lb_plot = np.where(np.isfinite(lb), lb, np.nan)
    ax.fill_between(step, lb_plot, ub, color=lighten(GOLD, 0.78), zorder=1,
                    step="post", linewidth=0.0)
    ax.step(step, ub, where="post", color=BLUE, lw=1.5, zorder=3,
            label="best bound (upper)")
    ax.step(step, lb_plot, where="post", color=GREEN, lw=1.5, zorder=3,
            label="incumbent (lower)")
    ax.axhline(best_z, color=RED, lw=0.9, ls=(0, (4, 2.2)), zorder=2)
    ax.text(step[-1], best_z + 0.35, f"proven optimum {best_z:.0f} units",
            ha="right", va="bottom", fontsize=6.2, color=RED)
    ax.set_xlim(step[0], step[-1])
    ax.set_ylim(np.nanmin(lb_plot) - 1.6, ub.max() + 2.4)
    ax.set_xlabel("nodes processed (count)", fontsize=7.2)
    ax.set_ylabel("objective (units)", fontsize=7.2)
    ax.set_title("Gap closes to 0 after the last node", fontsize=8.5,
                 fontweight="bold")
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="lower right", fontsize=6.3, frameon=False,
              handlelength=1.6, labelspacing=0.3)
    style_axes(ax)


def draw_depths(ax, nodes):
    cnt = Counter(n["depth"] for n in nodes)
    pruned = Counter(n["depth"] for n in nodes
                     if n["status"] in ("infeasible", "bounded"))
    depths = sorted(cnt)
    live = [cnt[d] - pruned[d] for d in depths]
    dead = [pruned[d] for d in depths]
    ax.bar(depths, live, width=0.58, color=lighten(BLUE, 0.42),
           edgecolor="white", linewidth=0.6, label="explored", zorder=3)
    ax.bar(depths, dead, width=0.58, bottom=live, color=PALE,
           edgecolor="white", linewidth=0.6, label="pruned", zorder=3)
    for d in depths:
        ax.text(d, cnt[d] + 0.10, f"{cnt[d]:d}", ha="center", va="bottom",
                fontsize=6.2, color=SUB)
    ax.set_xticks(depths)
    ax.set_ylim(0.0, max(cnt.values()) + 1.0)
    ax.set_yticks(np.arange(0, max(cnt.values()) + 2, 1))
    ax.set_xlabel("tree depth (levels below root)", fontsize=7.2)
    ax.set_ylabel("nodes (count)", fontsize=7.2)
    ax.set_title(f"{len(nodes)} nodes over {max(depths) + 1} levels",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="upper right", fontsize=6.3, frameon=False,
              handlelength=1.4, labelspacing=0.3)
    style_axes(ax)


def main():
    set_pub_style()
    nodes, hist, best_z, best_x = branch_and_bound()
    path, best_id = optimal_path(nodes, best_z)

    fig = plt.figure(figsize=(W, H))
    ax_tree = inch_axes(fig, (0.42, 1.02, 6.00, 4.42), aspect=None)
    ax_tree.set_xlim(0.0, 6.00)
    ax_tree.set_ylim(0.0, 4.42)
    ax_top = axes_inches(fig, (7.30, 3.66, 2.85, 1.72))
    ax_bot = axes_inches(fig, (7.30, 1.32, 2.85, 1.72))

    pos, kids = layout(nodes, x_span=5.80, y_top=3.76, y_step=0.71)
    pos = {i: (x + 0.10, y) for i, (x, y) in pos.items()}
    draw_tree(ax_tree, nodes, pos, kids, path, best_id, bw=1.02, bh=0.44)

    ax_tree.text(0.10, 4.40,
                 "Depth-first search tree, branch condition on every edge",
                 ha="left", va="top", fontsize=9.0, fontweight="bold",
                 color=INK)
    ax_tree.text(0.10, 4.18,
                 rf"max  $5x_1 + 4x_2$   s.t.  $6x_1 + 4x_2 \leq 25$,  "
                 rf"$x_1 + 2x_2 \leq 6$,  $x_1, x_2 \in \mathbb{{Z}}_{{+}}$",
                 ha="left", va="top", fontsize=7.0, color=SUB)

    handles = [
        Patch(facecolor=FACE["branched"], edgecolor=EDGE["branched"],
              label="branched (fractional LP)"),
        Patch(facecolor=FACE["integer"], edgecolor=EDGE["integer"],
              label="integer feasible"),
        Patch(facecolor=PALE, edgecolor=GREY, label="pruned (crossed out)"),
        Line2D([], [], color=RED, lw=1.7, label="path to the optimum")]
    ax_tree.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.50, -0.010), fontsize=6.5,
                   frameon=False, ncol=4, handlelength=1.5,
                   columnspacing=1.6)

    draw_convergence(ax_top, hist, best_z)
    draw_depths(ax_bot, nodes)

    panel_labels_inches(fig, [ax_top, ax_bot], dx=-0.60, dy=0.08, start=1)
    panel_labels_inches(fig, [ax_tree], dx=-0.22, dy=0.02)

    savefig(fig, "859_orx_branch_bound_tree.png")


if __name__ == "__main__":
    main()
