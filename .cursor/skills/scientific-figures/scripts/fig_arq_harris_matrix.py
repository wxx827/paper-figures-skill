"""Harris matrix: twenty-three contexts, one acyclic sequence, no guesswork.

A Harris matrix is a drawing of a partial order, so the drawing has to obey
the order rather than illustrate it.  The relationships are declared once as
a list of "earlier -> later" pairs; the row of every unit is then the longest
path to it from the base of the sequence, which is what puts contemporary
units side by side on one level and guarantees that every line in the figure
runs upward in time.  The edge list is checked with a topological sort before
anything is drawn, so a cycle - the one error a Harris matrix can never
survive - stops the script instead of reaching the page.

Column order inside each level is not the order the contexts were typed in.
Repeated barycentre sweeps move each unit toward the mean position of its
neighbours, and the printed crossing count is the measured improvement over
the declaration order.  Edges that skip a level are routed through a corridor
chosen to clear every box in between, which is why long relationships bend
rather than pass through unrelated contexts.  The side panels count what the
matrix contains: units per phase, split by the three kinds of unit, and the
relationships by type.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from pubstyle import set_pub_style, savefig, get_palette, darken, add_footer
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"
LINK = "#6f6a5f"           # relationship lines: a dark soil grey

FW, FH = 10.4, 6.4         # DESIGN_SYSTEM section 1.1, "wide" tier
PW, PH = 5.30, 4.28        # main panel, inches (its data units are inches)
BW, BH = 1.10, 0.22        # unit box
COL_W = 1.38               # column pitch
CX = 2.64                  # column grid centre
X_DEPTH = 0.60             # right edge of the depth column
CLEAR = 0.62               # corridor clearance from a box centre

# Low-saturation soil palette; deposits are earth, cuts are voids (white),
# built features are the grey-green of mortar and stone.
KIND = {
    "deposit":   dict(fc="#e6e1d1", ec="#8f856a", lw=0.8, ls="-"),
    "interface": dict(fc="#ffffff", ec="#8f856a", lw=0.8, ls=(0, (3.0, 1.6))),
    "feature":   dict(fc="#ccd2c2", ec="#5f6b55", lw=1.5, ls="-"),
}
KIND_LABEL = {"deposit": "deposit", "interface": "interface (cut)",
              "feature": "structural feature"}

PHASES = [
    ("Phase 1", "120 cal BC - cal AD 20"),
    ("Phase 2", "cal AD 20-130"),
    ("Phase 3", "cal AD 130-350"),
    ("Phase 4", "cal AD 350-540"),
]

# key, box caption (12 characters or fewer), kind, phase index
NODES = [
    ("natural",    "natural",      "deposit",   0),
    ("buried",     "buried soil",  "deposit",   0),
    ("ditchcut",   "ditch cut",    "interface", 0),
    ("quarrycut",  "quarry cut",   "interface", 0),
    ("ditchfill",  "ditch fill",   "deposit",   0),
    ("quarryfill", "quarry fill",  "deposit",   0),
    ("phcut",      "posthole cut", "interface", 0),
    ("packing",    "post packing", "feature",   0),
    ("dump",       "levelling",    "deposit",   0),
    ("floor",      "earth floor",  "deposit",   1),
    ("pitcut",     "pit cut",      "interface", 1),
    ("hearth",     "hearth",       "feature",   1),
    ("pitfill",    "pit fill",     "deposit",   1),
    ("yardmk",     "yard make-up", "deposit",   2),
    ("wallcut",    "wall cut",     "interface", 2),
    ("wall",       "wall footing", "feature",   2),
    ("wall_b",     "wall footing", "feature",   2),
    ("yard",       "yard surface", "deposit",   2),
    ("mortar",     "mortar floor", "deposit",   2),
    ("midden",     "midden",       "deposit",   2),
    ("demol",      "demolition",   "deposit",   2),
    ("robcut",     "robbing cut",  "interface", 3),
    ("plough",     "ploughsoil",   "deposit",   3),
]

# earlier -> later, with the relationship the excavator recorded
EDGES = [
    ("natural", "ditchcut", "cut"), ("natural", "buried", "over"),
    ("natural", "quarrycut", "cut"), ("ditchcut", "ditchfill", "fill"),
    ("quarrycut", "quarryfill", "fill"), ("buried", "phcut", "cut"),
    ("phcut", "packing", "fill"), ("ditchfill", "dump", "over"),
    ("quarryfill", "dump", "over"), ("packing", "floor", "over"),
    ("ditchfill", "floor", "over"), ("floor", "hearth", "over"),
    ("dump", "pitcut", "cut"), ("pitcut", "pitfill", "fill"),
    ("hearth", "yardmk", "over"), ("pitfill", "wallcut", "cut"),
    ("pitfill", "yardmk", "over"), ("wallcut", "wall", "fill"),
    ("wallcut", "wall_b", "fill"), ("wall", "mortar", "abut"),
    ("yardmk", "yard", "over"), ("yard", "midden", "over"),
    ("mortar", "demol", "over"), ("wall", "demol", "over"),
    ("wall_b", "demol", "over"), ("midden", "demol", "over"),
    ("demol", "robcut", "cut"), ("robcut", "plough", "over"),
    ("demol", "plough", "over"),
]
# One wall, recorded as two segments in two trenches: the Harris "=".
EQUAL = [("wall", "wall_b")]

REL_LABEL = {"over": "overlies / sealed by", "cut": "cuts / cut by",
             "fill": "fills / filled by", "abut": "abuts / bonded to",
             "same": "same unit (=)"}

DEPTH_TOP, DEPTH_BASE = 0.10, 2.05      # m below site datum


def topological_order(keys, edges):
    """Kahn's algorithm; raises if the declared relationships form a cycle."""
    indeg = {k: 0 for k in keys}
    succ = {k: [] for k in keys}
    for a, b, _ in edges:
        succ[a].append(b)
        indeg[b] += 1
    queue = [k for k in keys if indeg[k] == 0]
    out = []
    while queue:
        k = queue.pop(0)
        out.append(k)
        for m in succ[k]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(out) != len(keys):
        raise ValueError("stratigraphic relationships contain a cycle")
    return out


def longest_path_levels(keys, edges):
    """Level = length of the longest chain of contexts below a unit."""
    level = {k: 0 for k in keys}
    for k in topological_order(keys, edges):
        for a, b, _ in edges:
            if a == k and level[b] < level[k] + 1:
                level[b] = level[k] + 1
    return level


def column_x(i, k):
    """x of slot ``i`` of ``k`` units, kept symmetric about the grid centre."""
    return CX + (i - (k - 1) / 2.0) * COL_W


def order_levels(keys, edges, level, sweeps=10):
    """Barycentre sweeps: pull each unit toward the mean of its neighbours.

    The barycentre has to be taken over *x positions*, not over slot indices:
    levels hold different numbers of units, so slot 0 of a two-unit level and
    slot 0 of a three-unit level sit 0.69 in apart and an index-based average
    silently compares the wrong things.  Only neighbours exactly one level
    away are counted, and the best ordering seen is kept, so a sweep can
    never leave the layout worse than it found it.
    """
    nlev = max(level.values()) + 1
    order = [[k for k in keys if level[k] == lv] for lv in range(nlev)]
    nb_up = {k: [] for k in keys}
    nb_dn = {k: [] for k in keys}
    for a, b, _ in edges:
        if level[b] - level[a] == 1:
            nb_up[a].append(b)
            nb_dn[b].append(a)

    def xmap():
        return {k: column_x(i, len(lv))
                for lv in order for i, k in enumerate(lv)}

    best, best_c = [list(s) for s in order], count_crossings(order, level,
                                                             edges)
    for s in range(sweeps):
        down = s % 2 == 0
        nb = nb_dn if down else nb_up
        for lv in (range(1, nlev) if down else range(nlev - 2, -1, -1)):
            x = xmap()
            cur = order[lv]
            key = [(float(np.mean([x[n] for n in nb[k]])) if nb[k] else x[k],
                    i) for i, k in enumerate(cur)]
            order[lv] = [k for _, k in sorted(zip(key, cur))]
        c = count_crossings(order, level, edges)
        if c < best_c:
            best, best_c = [list(s2) for s2 in order], c
    order = best
    for a, b in EQUAL:                     # equated units are drawn adjacent
        seq = order[level[a]]
        if b in seq and a in seq:
            seq.remove(b)
            seq.insert(seq.index(a) + 1, b)
    return order


def count_crossings(order, level, edges):
    """Crossings between single-level edges, counted gap by gap."""
    pos = {k: column_x(i, len(lv)) for lv in order for i, k in enumerate(lv)}
    gaps = {}
    for a, b, _ in edges:
        if level[b] - level[a] == 1:
            gaps.setdefault(level[a], []).append((pos[a], pos[b]))
    total = 0
    for lst in gaps.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if (lst[i][0] - lst[j][0]) * (lst[i][1] - lst[j][1]) < 0:
                    total += 1
    return total


def main():
    set_pub_style()

    keys = [n[0] for n in NODES]
    desc = {n[0]: n[1] for n in NODES}
    kind = {n[0]: n[2] for n in NODES}
    phase = {n[0]: n[3] for n in NODES}

    level = longest_path_levels(keys, EDGES)
    nlev = max(level.values()) + 1
    raw_order = [[k for k in keys if level[k] == lv] for lv in range(nlev)]
    cross0 = count_crossings(raw_order, level, EDGES)
    order = order_levels(keys, EDGES, level, sweeps=8)
    cross1 = count_crossings(order, level, EDGES)

    row = (PH - BH) / (nlev - 1)
    gapsize = row - BH
    xpos, ypos = {}, {}
    for lv, seq in enumerate(order):
        for i, k in enumerate(seq):
            xpos[k] = column_x(i, len(seq))
            ypos[k] = BH / 2.0 + lv * row

    # SU numbers follow the drawing: 101 at the top, deepest number at the
    # base, so the labels cannot disagree with the sequence.
    su = {}
    for i, k in enumerate(sorted(keys, key=lambda n: (-level[n], xpos[n]))):
        su[k] = f"SU {101 + i}"

    depth = {lv: DEPTH_TOP + (nlev - 1 - lv) * (DEPTH_BASE - DEPTH_TOP)
             / (nlev - 1) for lv in range(nlev)}
    thick_cm = 100.0 * (DEPTH_BASE - DEPTH_TOP) / (nlev - 1)

    pcol = get_palette("cb_sequential_purple", 6)[2:6]

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, (0.80, 1.36, PW, PH))
    ax_m.set_xlim(0.0, PW)
    ax_m.set_ylim(0.0, PH)
    ax_m.set_aspect("equal")
    ax_m.axis("off")
    ax_p = axes_inches(fig, (6.98, 3.72, 2.90, 1.92))
    ax_r = axes_inches(fig, (6.98, 1.36, 2.90, 1.55))

    # ---- (a) phase bands and the depth column ----------------------------
    for pi, (pname, prange) in enumerate(PHASES):
        lvs = [level[k] for k in keys if phase[k] == pi]
        y0 = BH / 2.0 + min(lvs) * row - BH / 2.0 - 0.035
        y1 = BH / 2.0 + max(lvs) * row + BH / 2.0 + 0.035
        ax_m.add_patch(Rectangle((0.03, y0), 0.20, y1 - y0,
                                 facecolor=pcol[pi], alpha=0.55, lw=0,
                                 zorder=1))
        ax_m.text(0.13, 0.5 * (y0 + y1), pname, rotation=90, ha="center",
                  va="center", fontsize=6.2, color=darken(pcol[pi], 0.45),
                  fontweight="bold", zorder=2)
    for lv in range(nlev):
        ax_m.text(X_DEPTH, BH / 2.0 + lv * row, f"{depth[lv]:.2f}",
                  ha="right", va="center", fontsize=6.0, color=MUTED,
                  zorder=2)
    ax_m.text(X_DEPTH, -0.07, "m below datum", ha="right", va="top",
              fontsize=6.0, color=MUTED)

    # ---- (a) relationships, drawn under the boxes ------------------------
    used = []
    for a, b, _ in sorted(EDGES, key=lambda e: level[e[1]] - level[e[0]]):
        xa, xb = xpos[a], xpos[b]
        y0, y1 = ypos[a] + BH / 2.0, ypos[b] - BH / 2.0
        if level[b] - level[a] == 1:
            ym = 0.5 * (y0 + y1)
            path = [(xa, y0), (xa, ym), (xb, ym), (xb, y1)]
        else:
            ya = y0 + 0.45 * gapsize
            yb = y1 - 0.45 * gapsize
            block = [xpos[k] for k in keys
                     if level[a] < level[k] < level[b]]
            cand = np.linspace(0.66, PW - 0.16, 120)
            ok = np.array([min(abs(c - v) for v in block) >= CLEAR
                           if block else True for c in cand])
            cost = np.abs(cand - 0.5 * (xa + xb)) + np.where(ok, 0.0, 99.0)
            for uc, uy0, uy1 in used:                # keep corridors apart
                if not (yb < uy0 or ya > uy1):
                    cost += 3.0 * (np.abs(cand - uc) < 0.14)
            xc = float(cand[int(np.argmin(cost))])
            used.append((xc, ya, yb))
            path = [(xa, y0), (xa, ya), (xc, ya), (xc, yb), (xb, yb),
                    (xb, y1)]
        ax_m.plot([p[0] for p in path], [p[1] for p in path], color=LINK,
                  lw=0.8, solid_joinstyle="miter", solid_capstyle="butt",
                  zorder=3)
    for a, b in EQUAL:                     # the "=" between equated units
        xl, xr = sorted((xpos[a], xpos[b]))
        for dy in (-0.022, 0.022):
            ax_m.plot([xl + BW / 2.0, xr - BW / 2.0],
                      [ypos[a] + dy, ypos[b] + dy], color=LINK, lw=0.8,
                      zorder=3)

    # ---- (a) the units ----------------------------------------------------
    for k in keys:
        st = KIND[kind[k]]
        ax_m.add_patch(FancyBboxPatch(
            (xpos[k] - BW / 2.0, ypos[k] - BH / 2.0), BW, BH,
            boxstyle="round,pad=0,rounding_size=0.03", facecolor=st["fc"],
            edgecolor=st["ec"], linewidth=st["lw"], linestyle=st["ls"],
            mutation_aspect=1.0, zorder=5))
        ax_m.text(xpos[k] - BW / 2.0 + 0.055, ypos[k], su[k], ha="left",
                  va="center", fontsize=6.4, fontweight="bold", color=INK,
                  zorder=6)
        ax_m.text(xpos[k] + BW / 2.0 - 0.055, ypos[k], desc[k], ha="right",
                  va="center", fontsize=6.0, color=SUB, zorder=6)

    ax_m.annotate("", xy=(PW - 0.05, PH - 0.12), xytext=(PW - 0.05, 0.12),
                  arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    ax_m.text(PW - 0.05, 0.5 * PH, "earlier  to  later", rotation=90,
              ha="right", va="center", fontsize=6.2, color=MUTED)
    ax_m.text(0.0, PH + 0.06,
              f"Harris matrix, {len(keys)} units, "
              f"{len(EDGES) + len(EQUAL)} relationships",
              ha="left", va="bottom", fontsize=9.0, fontweight="bold",
              color=INK)

    # ---- (b) units per phase, split by kind -------------------------------
    kinds = ["deposit", "interface", "feature"]
    counts = np.array([[sum(1 for k in keys if phase[k] == pi
                            and kind[k] == kd) for kd in kinds]
                       for pi in range(len(PHASES))], dtype=float)
    yp = np.arange(len(PHASES))
    left = np.zeros(len(PHASES))
    for j, kd in enumerate(kinds):
        # Segments carry the box styles of panel (a), so the key at the foot
        # of the figure serves both panels.
        ax_p.barh(yp, counts[:, j], left=left, height=0.62,
                  color=KIND[kd]["fc"], edgecolor=KIND[kd]["ec"],
                  linewidth=0.7, linestyle=KIND[kd]["ls"], zorder=3)
        left += counts[:, j]
    for pi in range(len(PHASES)):
        ax_p.text(left[pi] + 0.16, yp[pi], f"n = {left[pi]:.0f}", ha="left",
                  va="center", fontsize=6.4, color=SUB)
    ax_p.set_yticks(yp)
    ax_p.set_yticklabels([f"{p}\n{r}" for p, r in PHASES], fontsize=6.0,
                         linespacing=1.35)
    for tick, c in zip(ax_p.get_yticklabels(), pcol):
        tick.set_color(darken(c, 0.45))
    ax_p.set_xlim(0.0, left.max() * 1.28)
    ax_p.set_ylim(-0.62, len(PHASES) - 0.38)
    ax_p.set_xlabel("units recorded (n)", fontsize=8.0)
    ax_p.tick_params(axis="x", labelsize=7.0, length=2.6)
    ax_p.tick_params(axis="y", length=0.0)
    ax_p.set_title("Phase 1 is the deepest and the busiest", fontsize=8.5,
                   fontweight="bold", loc="left")
    ax_p.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_p.grid(axis="x", lw=0.5, color=HAIR, alpha=0.7)
    ax_p.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax_p.spines[s].set_visible(False)

    # ---- (c) relationships by type ----------------------------------------
    rel = {t: 0 for t in REL_LABEL}
    for _, _, t in EDGES:
        rel[t] += 1
    rel["same"] = len(EQUAL)
    items = sorted(rel.items(), key=lambda kv: kv[1])
    yr = np.arange(len(items))
    vals = np.array([v for _, v in items], dtype=float)
    ax_r.barh(yr, vals, height=0.62, color="#b7ae94", edgecolor="white",
              linewidth=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax_r.text(v + 0.25, yr[i], f"{v:.0f}", ha="left", va="center",
                  fontsize=6.4, color=SUB)
    ax_r.set_yticks(yr)
    ax_r.set_yticklabels([REL_LABEL[t] for t, _ in items], fontsize=6.2)
    ax_r.set_xlim(0.0, vals.max() * 1.22)
    ax_r.set_ylim(-0.62, len(items) - 0.38)
    ax_r.set_xlabel("relationships recorded (n)", fontsize=8.0)
    ax_r.tick_params(axis="x", labelsize=7.0, length=2.6)
    ax_r.tick_params(axis="y", length=0.0)
    ax_r.set_title("Superposition carries most of the sequence",
                   fontsize=8.5, fontweight="bold", loc="left")
    ax_r.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_r.grid(axis="x", lw=0.5, color=HAIR, alpha=0.7)
    ax_r.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax_r.spines[s].set_visible(False)

    fig.legend(handles=[
        Patch(facecolor=KIND[k]["fc"], edgecolor=KIND[k]["ec"],
              linewidth=KIND[k]["lw"], linestyle=KIND[k]["ls"],
              label=KIND_LABEL[k]) for k in kinds] + [
        Line2D([0], [0], color=LINK, lw=0.8,
               label="stratigraphic relationship, earlier below"),
        Line2D([0], [0], color=LINK, lw=0.8, ls=(0, (1.0, 1.0)),
               label="equated units, one context in two trenches (=)")],
        loc="lower left", bbox_to_anchor=(0.030, 0.100), ncol=3,
        fontsize=6.4, frameon=False, handletextpad=0.5, columnspacing=1.6,
        labelspacing=0.4)

    panel_labels_inches(fig, [ax_m, ax_p, ax_r], dx=-0.36, dy=0.16)


    fig.canvas.draw()
    note = add_footer(fig, "Synthetic data, figure 893")
    nb = note.get_window_extent()
    clash = [lab.get_text()
             for ax in (ax_p, ax_r)
             for lab in (ax.xaxis.label, ax.yaxis.label)
             if lab.get_text() and nb.overlaps(lab.get_window_extent())]
    print("footnote overlap:", clash if clash else "none")

    print(f"{len(keys)} units, {len(EDGES)} relationships, {nlev} levels, "
          f"DAG verified by topological sort")
    print(f"crossings: declaration order {cross0} -> barycentre order "
          f"{cross1}")
    for lv in range(nlev - 1, -1, -1):
        print(f"  level {lv:2d}  depth {depth[lv]:.2f} m  "
              + ", ".join(f"{su[k]} {desc[k]}" for k in order[lv]))
    savefig(fig, "893_arq_harris_matrix.png")


if __name__ == "__main__":
    main()
