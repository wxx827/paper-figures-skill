"""Human-error fault tree for a valve line-up task, with cut-set importance.

Main panel: the tree is drawn from a node table with standard gate symbols -
the AND gate is a flat-bottomed shape with a semicircular top, the OR gate a
shield with a concave bottom and a pointed apex - and the probability in every
intermediate box is propagated from the basic events underneath it, AND gates
multiplying and OR gates taking one minus the product of the complements.  The
dominant minimal cut set is highlighted along its path to the top event.

Auxiliary (both at 58 % of the main panel width): the contribution of each
minimal cut set, and the Fussell-Vesely importance of each basic event.  Both
come from the cut sets that the tree structure generates, not from a list
typed alongside it.

Fictional plant and task, illustrative values only.
"""
import itertools
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, inch_axes, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#cccccc"
GRID = "#d5d5d5"
SOFT = "#f5f5f5"
BLUE = "#3b5b92"
TEAL = "#2e7d5b"
AMBER = "#c9a227"
RUST = "#c0603f"
ACCENT = "#b23a48"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.62, 0.72, 5.86, 4.72)
AUX_T_RECT = (7.62, 3.48, 3.16, 2.00)
AUX_B_RECT = (7.62, 0.86, 3.16, 2.00)

BASIC = {
    "B1": ("procedure step\nambiguous", 0.060),
    "B2": ("valve tag\nmislabelled", 0.040),
    "B3": ("time-pressure\nslip", 0.090),
    "B4": ("checker skips\nthe step", 0.120),
    "B5": ("check done by\nthe same person", 0.200),
    "B6": ("alarm suppressed\nin start-up mode", 0.070),
    "B7": ("indication not on\nthe main display", 0.150),
}
GATES = {
    "TOP": ("OR", ["E1", "E2"], "wrong valve line-up\nreleased to the plant"),
    "E1": ("AND", ["E3", "E4"], "line-up error made\nand not caught"),
    "E2": ("AND", ["B6", "B7"], "control room does\nnot detect it"),
    "E3": ("OR", ["B1", "B2", "B3"], "wrong valve\nselected"),
    "E4": ("AND", ["B4", "B5"], "independent check\nfails"),
}


def prob(node):
    """Bottom-up propagation: AND multiplies, OR uses 1 - prod(1 - p)."""
    if node in BASIC:
        return BASIC[node][1]
    kind, kids, _ = GATES[node]
    ps = [prob(k) for k in kids]
    if kind == "AND":
        return float(np.prod(ps))
    return float(1.0 - np.prod([1.0 - p for p in ps]))


def cut_sets(node):
    """Cut sets of a node, as a list of frozensets of basic events."""
    if node in BASIC:
        return [frozenset([node])]
    kind, kids, _ = GATES[node]
    parts = [cut_sets(k) for k in kids]
    if kind == "OR":
        out = [c for p in parts for c in p]
    else:
        out = [frozenset().union(*combo)
               for combo in itertools.product(*parts)]
    return minimal(out)


def minimal(sets):
    """Drop any cut set that contains another one."""
    keep = []
    for s in sets:
        if not any(o < s for o in sets if o != s):
            if s not in keep:
                keep.append(s)
    return keep


def cs_prob(cs):
    return float(np.prod([BASIC[b][1] for b in cs]))


def and_gate_poly(x, y, w=0.40, h=0.36, n=40):
    r = w / 2.0
    rect_h = max(h - r, 0.02)
    y0 = y - h / 2.0
    th = np.linspace(0.0, np.pi, n)
    arc = np.column_stack([x + r * np.cos(th), y0 + rect_h + r * np.sin(th)])
    return np.vstack([[[x - r, y0], [x + r, y0], [x + r, y0 + rect_h]],
                      arc[::-1] if arc[0, 0] > arc[-1, 0] else arc,
                      [[x - r, y0 + rect_h]]])


def _quad(p0, p1, p2, n=26):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) \
        + t ** 2 * np.array(p2)


def or_gate_poly(x, y, w=0.40, h=0.40):
    r = w / 2.0
    y0, y1 = y - h / 2.0, y + h / 2.0
    bottom = _quad((x - r, y0), (x, y0 + 0.30 * h), (x + r, y0))
    right = _quad((x + r, y0), (x + 0.86 * r, y0 + 0.62 * h), (x, y1))
    left = _quad((x, y1), (x - 0.86 * r, y0 + 0.62 * h), (x - r, y0))
    return np.vstack([bottom, right, left])


def main():
    set_pub_style(base_font=8)

    p_top = prob("TOP")
    mcs = cut_sets("TOP")
    mcs = sorted(mcs, key=cs_prob, reverse=True)
    mcs_p = np.array([cs_prob(c) for c in mcs])
    mcs_sum = float(mcs_p.sum())
    fv = {b: sum(cs_prob(c) for c in mcs if b in c) / mcs_sum for b in BASIC}
    dominant = set(mcs[0])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = inch_axes(fig, MAIN_RECT, aspect="equal")
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    W, H = MAIN_RECT[2], MAIN_RECT[3]
    ax.add_patch(FancyBboxPatch((0.02, 0.02), W - 0.04, H - 0.04,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor="none", edgecolor=HAIR, lw=0.8,
                                zorder=0))

    # node geometry, inches on the panel canvas
    box_y = {"TOP": 4.40, "E1": 3.52, "E2": 3.52, "E3": 2.62, "E4": 2.62}
    box_x = {"TOP": 2.95, "E1": 1.95, "E2": 4.62, "E3": 1.02, "E4": 2.88}
    box_w = {"TOP": 2.10, "E1": 1.66, "E2": 1.62, "E3": 1.30, "E4": 1.42}
    gate_y = {"TOP": 3.96, "E1": 3.08, "E2": 3.08, "E3": 2.18, "E4": 2.18}
    circ = {"B1": (0.42, 1.66), "B2": (1.02, 1.66), "B3": (1.62, 1.66),
            "B4": (2.58, 1.66), "B5": (3.18, 1.66),
            "B6": (4.22, 2.62), "B7": (5.02, 2.62)}
    box_h, r_b = 0.40, 0.215

    def node_top(n):
        return (circ[n][0], circ[n][1] + r_b) if n in BASIC else \
            (box_x[n], box_y[n] + box_h / 2.0)

    hot_nodes = dominant | {"E2", "TOP"}

    # gates and the wiring beneath them
    for name, (kind, kids, _) in GATES.items():
        gx, gy = box_x[name], gate_y[name]
        ec = ACCENT if name == "E2" else INK
        poly = (and_gate_poly(gx, gy) if kind == "AND"
                else or_gate_poly(gx, gy))
        ax.add_patch(Polygon(poly, closed=True,
                             facecolor=lighten(BLUE if kind == "AND" else
                                               TEAL, 0.80),
                             edgecolor=ec, lw=1.1, zorder=5))
        ax.text(gx, gy - 0.045, kind, ha="center", va="center", fontsize=6,
                color=INK, fontweight="bold", zorder=6)
        ax.plot([gx, gx], [gy + (0.20 if kind == "AND" else 0.20),
                           box_y[name] - box_h / 2.0], color=INK, lw=0.9,
                zorder=3)
        bus = gy - 0.30
        ax.plot([gx, gx], [gy - 0.18, bus], color=INK, lw=0.9, zorder=3)
        xs = [node_top(k)[0] for k in kids]
        ax.plot([min(xs), max(xs)], [bus, bus], color=INK, lw=0.9, zorder=3)
        for k in kids:
            kx, ky = node_top(k)
            lit = (k in dominant) or (k in hot_nodes and k != "TOP")
            lit = lit and name in hot_nodes
            ax.plot([kx, kx], [ky, bus],
                    color=ACCENT if lit else INK,
                    lw=1.6 if lit else 0.9, zorder=4 if lit else 3)

    # intermediate event boxes
    for name in ("TOP", "E1", "E2", "E3", "E4"):
        p = prob(name)
        hot = name in ("E2", "TOP")
        ax.add_patch(FancyBboxPatch(
            (box_x[name] - box_w[name] / 2.0, box_y[name] - box_h / 2.0),
            box_w[name], box_h,
            boxstyle="round,pad=0,rounding_size=0.045",
            facecolor=lighten(ACCENT, 0.90) if hot else SOFT,
            edgecolor=ACCENT if hot else "#9a9a9a",
            lw=1.3 if name == "TOP" else 0.9, mutation_aspect=1.0, zorder=6))
        ax.text(box_x[name], box_y[name] + 0.055, GATES[name][2],
                ha="center", va="center", fontsize=6.3, color=INK,
                linespacing=1.3, zorder=7,
                fontweight="bold" if name == "TOP" else "normal")
        ax.text(box_x[name], box_y[name] - 0.135, f"P = {p:.4g} per demand",
                ha="center", va="center", fontsize=6.0,
                color=ACCENT if hot else SUB, zorder=7)

    # basic events
    for name, (label, p) in BASIC.items():
        cx, cy = circ[name]
        hot = name in dominant
        ax.add_patch(Circle((cx, cy), r_b,
                            facecolor=lighten(ACCENT if hot else BLUE, 0.84),
                            edgecolor=ACCENT if hot else BLUE, lw=1.1,
                            zorder=6))
        ax.text(cx, cy, name, ha="center", va="center", fontsize=6.2,
                color=INK, fontweight="bold", zorder=7)
        ax.text(cx, cy - r_b - 0.07, f"{label}\np = {p:.3f}", ha="center",
                va="top", fontsize=6, color=SUB, linespacing=1.35,
                zorder=7)

    # legend for the gate symbols and the summary
    lx, ly = 3.62, 1.42
    ax.add_patch(Polygon(and_gate_poly(lx, ly, 0.30, 0.28), closed=True,
                         facecolor=lighten(BLUE, 0.80), edgecolor=INK,
                         lw=1.0, zorder=5))
    ax.text(lx + 0.22, ly, "AND gate: all inputs must occur,\n"
            "probabilities multiply", ha="left", va="center", fontsize=6.0,
            color=SUB, linespacing=1.4)
    ax.add_patch(Polygon(or_gate_poly(lx, ly - 0.46, 0.30, 0.30),
                         closed=True, facecolor=lighten(TEAL, 0.80),
                         edgecolor=INK, lw=1.0, zorder=5))
    ax.text(lx + 0.22, ly - 0.46, "OR gate: any input suffices,\n"
            "P = 1 - prod(1 - p)", ha="left", va="center", fontsize=6.0,
            color=SUB, linespacing=1.4)
    ax.text(lx - 0.22, ly - 0.92,
            f"top event  P = {p_top:.4f} per line-up  "
            f"({p_top * 1000:.1f} per 1000)\n"
            f"{len(mcs)} minimal cut sets, rare-event sum "
            f"{mcs_sum:.4f}\ndominant cut set "
            f"{{{', '.join(sorted(dominant))}}} = {mcs_p[0]:.4f} "
            f"({100 * mcs_p[0] / mcs_sum:.0f} % of the sum)",
            ha="left", va="top", fontsize=6.3, color=INK, linespacing=1.55)
    ax.text(0.16, 0.20, "basic-event probabilities are per demand; the tree "
            "is evaluated bottom-up", ha="left", va="bottom", fontsize=6,
            color=MUTED)

    # ---- aux (b): cut-set contributions ------------------------------------
    labs = [" + ".join(sorted(c)) for c in mcs]
    share = 100.0 * mcs_p / mcs_sum
    ypos = np.arange(len(mcs))[::-1]
    cols = [ACCENT if set(c) == dominant else BLUE for c in mcs]
    ax_t.grid(axis="x", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_t.barh(ypos, share, height=0.58, color=cols, edgecolor="white",
              linewidth=0.7, zorder=3)
    for k, y in enumerate(ypos):
        ax_t.text(share[k] + 1.4, y, f"{share[k]:.1f} %  "
                  f"(P = {mcs_p[k]:.5f})", ha="left", va="center",
                  fontsize=6.1, color=INK)
    ax_t.set_yticks(ypos)
    ax_t.set_yticklabels(labs, fontsize=6.4)
    ax_t.set_ylim(-0.7, len(mcs) - 0.3)
    ax_t.set_xlim(0.0, 108.0)
    ax_t.set_xticks([0, 25, 50, 75, 100])
    ax_t.tick_params(axis="x", labelsize=6.6)
    ax_t.set_xlabel("share of the rare-event cut-set sum (%)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_title("One cut set carries the risk", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    # ---- aux (c): Fussell-Vesely importance --------------------------------
    names = sorted(BASIC, key=lambda b: fv[b])
    vals = np.array([100.0 * fv[b] for b in names])
    ypos = np.arange(len(names))
    cols = [ACCENT if b in dominant else BLUE for b in names]
    ax_b.grid(axis="x", color=GRID, lw=0.5, alpha=0.65, zorder=0)
    ax_b.barh(ypos, vals, height=0.62, color=cols, edgecolor="white",
              linewidth=0.7, zorder=3)
    for k in ypos:
        ax_b.text(vals[k] + 1.4, k, f"{vals[k]:.1f} %", ha="left",
                  va="center", fontsize=6.1, color=INK)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(
        [f"{b}  {BASIC[b][0].replace(chr(10), ' ')}" for b in names],
        fontsize=6.0)
    ax_b.set_ylim(-0.7, len(names) - 0.3)
    ax_b.set_xlim(0.0, 96.0)
    ax_b.set_xticks([0, 25, 50, 75])
    ax_b.tick_params(axis="x", labelsize=6.6)
    ax_b.set_xlabel("Fussell-Vesely importance (%)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_title("Which basic events to fix first", fontsize=8.4,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.44, dy=0.10)
    savefig(fig, "953_hfe_error_tree.png")


if __name__ == "__main__":
    main()
