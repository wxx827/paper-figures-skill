"""A five-level hierarchy, its shortcut links, and the shape of its pyramid.

The main panel is a tidy top-down tree: every parent sits at the mean x of its
children, so no branch crosses another and the depth of a node is readable
from its height alone.  Hue is the level, taken from a sequential ramp rather
than a categorical set, because depth is ordered and a qualitative palette
would suggest it is not.

Solid lines are the tree itself.  Dashed lines are the links that make the
structure a hierarchy in name only: lateral links between units in the same
level, and skip links that jump a level and bypass the chain of command.  The
subordinate panels count what the drawing can only suggest - how many nodes
each level holds, and how the shortcut links are distributed by the number of
levels they span.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import COLORS, darken, get_palette, savefig, set_pub_style
from figkit import halo_text, panel_label_inches

SEED = 5
LEVEL_NAMES = ["Root", "Division", "Group", "Team", "Unit"]
BRANCHING = [(3, 3), (2, 4), (2, 3), (1, 3)]     # (min, max) children per level
X_SPAN = 6.4
DY = 1.28
N_LATERAL = 9
N_SKIP = 6


def grow_tree(branching, seed=0):
    """Parent list plus per-node level for a random branching tree."""
    rng = np.random.default_rng(seed)
    parent = [-1]
    level = [0]
    frontier = [0]
    for lvl, (lo, hi) in enumerate(branching):
        nxt = []
        for p in frontier:
            for _ in range(int(rng.integers(lo, hi + 1))):
                parent.append(p)
                level.append(lvl + 1)
                nxt.append(len(parent) - 1)
        frontier = nxt
    return np.array(parent), np.array(level)


def tidy_layout(parent, level, x_span=1.0, dy=1.0):
    """Leaves evenly spaced, every parent centred over its own children.

    Spacing each level evenly and independently is the tempting shortcut and
    it produces a tree whose edges all cross: a node ends up above someone
    else's children.  Placing the leaves first and folding the means upward
    costs three lines and removes every crossing.
    """
    n = len(parent)
    children = [[] for _ in range(n)]
    for v in range(1, n):
        children[parent[v]].append(v)
    x = np.zeros(n)
    slot = [0.0]

    def place(v):
        if not children[v]:
            x[v] = slot[0]
            slot[0] += 1.0
            return
        for c in children[v]:
            place(c)
        x[v] = np.mean([x[c] for c in children[v]])

    place(0)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-9) * x_span - x_span / 2.0
    y = -level * dy
    return x, y, children


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    parent, level = grow_tree(BRANCHING, seed=SEED)
    x, y, children = tidy_layout(parent, level, x_span=X_SPAN, dy=DY)
    n = len(parent)
    n_lvl = int(level.max()) + 1
    sizes = np.bincount(level)

    # depth-ordered ramp: root darkest, leaves lightest
    ramp = get_palette("cb_sequential_blue", n=n_lvl + 2)[::-1][:n_lvl]

    # ---- shortcut links ----------------------------------------------------
    def ancestor_path(v):
        p = []
        while v != -1:
            p.append(v)
            v = parent[v]
        return set(p)

    lateral, skip = [], []
    deep = np.flatnonzero(level >= 2)
    tries = 0
    while len(lateral) < N_LATERAL and tries < 4000:
        tries += 1
        u, v = rng.choice(deep, 2, replace=False)
        if level[u] == level[v] and parent[u] != parent[v] and \
                (u, v) not in lateral and (v, u) not in lateral:
            lateral.append((int(u), int(v)))
    tries = 0
    while len(skip) < N_SKIP and tries < 4000:
        tries += 1
        u, v = rng.choice(np.flatnonzero(level >= 1), 2, replace=False)
        if abs(level[u] - level[v]) >= 2 and u not in ancestor_path(v) and \
                v not in ancestor_path(u) and (u, v) not in skip:
            skip.append((int(u), int(v)))

    spans = np.array([abs(level[u] - level[v]) for u, v in lateral + skip])
    span_vals = np.arange(0, spans.max() + 1)
    span_counts = np.array([(spans == s).sum() for s in span_vals])

    # ---- figure ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.1))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.52],
                  height_ratios=[1.0, 0.60], hspace=0.52, wspace=0.16)
    ax = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])

    # ---- (a) the tree ------------------------------------------------------
    for lvl in range(n_lvl):                       # level bands
        if lvl % 2 == 0:
            ax.add_patch(Rectangle((-X_SPAN / 2 - 0.60, -lvl * DY - DY / 2.4),
                                   X_SPAN + 1.20, DY / 1.2,
                                   facecolor=COLORS["card"], edgecolor="none",
                                   zorder=0))
    for v in range(1, n):                          # tree edges
        p = parent[v]
        ax.plot([x[p], x[v]], [y[p], y[v]], color=ramp[level[v]], lw=0.85,
                alpha=0.65, zorder=2, solid_capstyle="round")
    for u, v in lateral:                           # same-level shortcuts
        ax.plot([x[u], x[v]], [y[u], y[v]], color=COLORS["baseline"],
                lw=0.85, ls=(0, (3.2, 1.8)), alpha=0.55, zorder=3)
    for u, v in skip:                              # level-jumping shortcuts
        ax.plot([x[u], x[v]], [y[u], y[v]], color=COLORS["accent"], lw=0.9,
                ls=(0, (3.2, 1.8)), alpha=0.55, zorder=3)

    subtree = np.zeros(n)
    for v in range(n - 1, -1, -1):
        subtree[v] = 1 + sum(subtree[c] for c in children[v])
    for lvl in range(n_lvl):
        m = level == lvl
        ax.scatter(x[m], y[m], s=16 + 3.4 * subtree[m], color=ramp[lvl],
                   edgecolor="white", linewidth=0.6, zorder=5)
    for lvl in range(n_lvl):
        ax.text(-X_SPAN / 2 - 0.72, -lvl * DY,
                f"L{lvl}  {LEVEL_NAMES[lvl]}\nn = {sizes[lvl]}", ha="right",
                va="center", fontsize=6.8, color=darken(ramp[lvl], 0.30),
                fontweight="bold", linespacing=1.25)

    halo_text(ax, x[0], y[0] + 0.34, f"root · {int(subtree[0])} nodes below",
              ha="center", va="bottom", fontsize=7.0, color=COLORS["ink"],
              halo="white", halo_lw=2.6, zorder=7)
    widest = int(np.argmax([len(c) for c in children]))
    halo_text(ax, x[widest], y[widest] + 0.22,
              f"{len(children[widest])} children", ha="center", va="bottom",
              fontsize=6.4, color=COLORS["sub"], halo="white", halo_lw=2.4,
              zorder=7)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-X_SPAN / 2 - 2.05, X_SPAN / 2 + 0.45)
    ax.set_ylim(-(n_lvl - 1) * DY - 0.62, 0.92)
    ax.text(0.0, 0.012, f"{n} nodes · {n - 1} tree links · "
            f"{len(lateral) + len(skip)} shortcut links · "
            "marker area $\\propto$ subtree size", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.9, color=COLORS["sub"])

    leg = [Line2D([], [], color=ramp[2], lw=1.2, label="reporting line"),
           Line2D([], [], color=COLORS["baseline"], lw=1.2,
                  ls=(0, (3.2, 1.8)), label="lateral link (same level)"),
           Line2D([], [], color=COLORS["accent"], lw=1.2, ls=(0, (3.2, 1.8)),
                  label="skip link ($\\geq$ 2 levels)")]
    ax.legend(handles=leg, loc="upper right", fontsize=6.9, handlelength=2.0,
              borderaxespad=0.2)

    # ---- (b) level pyramid -------------------------------------------------
    ypos = -np.arange(n_lvl)
    ax_p.barh(ypos, sizes, height=0.62, left=-sizes / 2.0,
              color=ramp, edgecolor="white", linewidth=0.7, zorder=3)
    for yy, s, lvl in zip(ypos, sizes, range(n_lvl)):
        ax_p.text(s / 2.0 + sizes.max() * 0.045, yy, f"{s}", va="center",
                  ha="left", fontsize=6.8, color=darken(ramp[lvl], 0.30))
    ax_p.set_yticks(ypos)
    ax_p.set_yticklabels([f"L{i}" for i in range(n_lvl)], fontsize=7.0)
    ax_p.set_xlim(-sizes.max() * 0.78, sizes.max() * 0.78)
    ax_p.set_xticks([-20, -10, 0, 10, 20])
    ax_p.set_xticklabels(["20", "10", "0", "10", "20"], fontsize=6.8)
    ax_p.set_xlabel("nodes per level", fontsize=7.5)
    ax_p.tick_params(labelsize=6.8)
    ax_p.set_title("Level pyramid", fontsize=8.5, loc="left", pad=4)
    ax_p.grid(axis="x", lw=0.5, color=COLORS["grid"], alpha=0.55)
    ratios = sizes[1:] / sizes[:-1]
    ax_p.text(0.02, 0.97, "level ratios\n" +
              " · ".join(f"{r:.1f}x" for r in ratios),
              transform=ax_p.transAxes, ha="left", va="top",
              fontsize=6.4, color=COLORS["sub"], linespacing=1.3)

    # ---- (c) shortcut links by span ---------------------------------------
    bar_cols = [COLORS["baseline"] if s < 2 else COLORS["accent"]
                for s in span_vals]
    ax_s.bar(span_vals, span_counts, width=0.55, color=bar_cols,
             edgecolor="white", linewidth=0.7, zorder=3)
    for s, c in zip(span_vals, span_counts):
        if c:
            ax_s.text(s, c + span_counts.max() * 0.05, f"{c}", ha="center",
                      va="bottom", fontsize=6.6, color=COLORS["ink"])
    ax_s.set_xticks(span_vals)
    ax_s.set_xlim(-0.6, span_vals.max() + 0.6)
    ax_s.set_ylim(0, span_counts.max() * 1.38)
    ax_s.set_xlabel("levels spanned by the shortcut", fontsize=7.5)
    ax_s.set_ylabel("links", fontsize=7.5)
    ax_s.tick_params(labelsize=6.8)
    ax_s.set_title("Shortcuts that bypass the chain", fontsize=8.5,
                   loc="left", pad=4)
    ax_s.grid(axis="y", lw=0.5, color=COLORS["grid"], alpha=0.55)

    fig.tight_layout(rect=[0, 0.015, 1, 0.925], h_pad=2.4, w_pad=2.2)
    panel_label_inches(fig, ax, "(a)", dx=-0.02, dy=0.12)
    panel_label_inches(fig, ax_p, "(b)", dx=-0.42, dy=0.12)
    panel_label_inches(fig, ax_s, "(c)", dx=-0.42, dy=0.12)

    savefig(fig, "373_net_hierarchy.png")


if __name__ == "__main__":
    main()
