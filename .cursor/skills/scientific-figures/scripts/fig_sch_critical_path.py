"""Activity-on-node project network with a computed critical path.

Forward and backward passes are run over the precedence list: earliest start
and finish come from the forward pass, latest start and finish from the
backward pass anchored on the project duration, and total float is LS - ES.
Activities with zero float form the critical path, which is highlighted rather
than declared.  The resource histogram is accumulated from the same schedule,
first at early start and then after a greedy levelling pass that shifts each
non-critical activity inside its own float window to shave the peak.
"""
import itertools
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, inch_axes, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"

W, H = 10.4, 6.4

# name: (duration in days, crew size, predecessors)
ACTS = {
    "A": (4.0, 3, []),
    "B": (3.0, 2, ["A"]),
    "C": (6.0, 4, ["A"]),
    "F": (3.0, 6, ["A"]),
    "D": (2.0, 2, ["B"]),
    "E": (5.0, 3, ["C"]),
    "H": (2.0, 4, ["B"]),
    "G": (4.0, 2, ["D", "E"]),
    "J": (2.0, 2, ["E"]),
    "I": (3.0, 4, ["G", "H", "J", "F"]),
}
TITLE = {"A": "site prep", "B": "formwork", "C": "steel frame",
         "F": "cable pull", "D": "ducting", "E": "cladding",
         "H": "fit-out", "G": "glazing", "J": "balustrade",
         "I": "commission"}

FOOT = ("Earliest start and finish come from a forward pass over the "
        "precedence list, latest start and finish from a backward pass "
        "anchored on the computed project duration, and total float is "
        "LS - ES; the zero-float chain is the critical path. The resource "
        "histogram is accumulated from the same schedule before and after an "
        "exact search over whole-day shifts of the non-critical activities "
        "inside their own float windows, so the levelled plan keeps every "
        "precedence link and the same project duration.")


def topological():
    order, done = [], set()
    while len(order) < len(ACTS):
        for k, (_, _, pre) in ACTS.items():
            if k not in done and all(p in done for p in pre):
                order.append(k)
                done.add(k)
    return order


def cpm():
    order = topological()
    es, ef = {}, {}
    for k in order:
        d, _, pre = ACTS[k]
        es[k] = max([ef[p] for p in pre], default=0.0)
        ef[k] = es[k] + d
    dur = max(ef.values())
    succ = {k: [] for k in ACTS}
    for k, (_, _, pre) in ACTS.items():
        for p in pre:
            succ[p].append(k)
    lf, ls = {}, {}
    for k in reversed(order):
        lf[k] = min([ls[s] for s in succ[k]], default=dur)
        ls[k] = lf[k] - ACTS[k][0]
    float_ = {k: ls[k] - es[k] for k in ACTS}
    return order, es, ef, ls, lf, float_, dur, succ


def ranks(order):
    r = {}
    for k in order:
        pre = ACTS[k][2]
        r[k] = 0 if not pre else max(r[p] for p in pre) + 1
    return r


def resource_profile(start, horizon):
    """Daily crew count; every duration and float here is a whole day."""
    use = np.zeros(int(round(horizon)))
    for k, (d, crew, _) in ACTS.items():
        s = int(round(start[k]))
        use[s:s + int(round(d))] += crew
    return use


def level(es, lf, float_, order, horizon):
    """Exact search over whole-day delays inside each activity's own float.

    Every candidate is checked against the precedence list and against its
    late finish before it is scored, so the levelled plan is a real schedule
    of the same length, not just a redrawn histogram.  The score is the peak
    crew first and the sum of squared daily crew second, which is the textbook
    levelling objective.
    """
    free = [k for k in order if float_[k] > 1e-9]
    grids = [range(int(round(float_[k])) + 1) for k in free]
    best, best_cost = dict(es), None
    for combo in itertools.product(*grids):
        st = dict(es)
        for k, off in zip(free, combo):
            st[k] = es[k] + off
        ok = True
        for k in order:
            d, _, pre = ACTS[k]
            if st[k] + d > lf[k] + 1e-9:
                ok = False
                break
            if any(st[k] < st[p] + ACTS[p][0] - 1e-9 for p in pre):
                ok = False
                break
        if not ok:
            continue
        u = resource_profile(st, horizon)
        cost = (float(u.max()), float((u ** 2).sum()))
        if best_cost is None or cost < best_cost:
            best, best_cost = st, cost
    return best


def draw_network(ax, es, ef, ls, lf, float_, dur, rank, bw, bh, span_x, span_y,
                 y0):
    cols = {}
    for k in ACTS:
        cols.setdefault(rank[k], []).append(k)
    pos = {}
    n_col = max(cols) + 1
    n_row = max(len(v) for v in cols.values())
    pitch_x = span_x / n_col
    pitch_y = span_y / n_row
    mid = n_row // 2
    # Critical activities take the middle row of every column, so the red
    # chain reads as one horizontal spine instead of a zig-zag.
    row_pref = sorted(range(n_row), key=lambda r: (abs(r - mid), r))
    for c, keys in cols.items():
        keys.sort()
        ordered = ([k for k in keys if float_[k] <= 1e-9] +
                   [k for k in keys if float_[k] > 1e-9])
        slots = [None] * n_row
        for k in ordered:
            for r in row_pref:
                if slots[r] is None:
                    slots[r] = k
                    break
        for r, k in enumerate(slots):
            if k is None:
                continue
            pos[k] = (0.30 + (c + 0.5) * pitch_x,
                      y0 + span_y - (r + 0.5) * pitch_y)

    for k, (_, _, pre) in ACTS.items():
        for p in pre:
            crit = float_[k] <= 1e-9 and float_[p] <= 1e-9
            x0, y0p = pos[p]
            x1, y1 = pos[k]
            ax.add_patch(FancyArrowPatch(
                (x0 + bw / 2, y0p), (x1 - bw / 2, y1), arrowstyle="-|>",
                mutation_scale=8.5, linewidth=1.9 if crit else 0.75,
                color=RED if crit else HAIR, shrinkA=0, shrinkB=0,
                connectionstyle="arc3,rad=0.0", zorder=2 if crit else 1))

    for k in ACTS:
        d, crew, _ = ACTS[k]
        crit = float_[k] <= 1e-9
        x, y = pos[k]
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=lighten(RED, 0.80) if crit else "white",
            edgecolor=RED if crit else GREY,
            linewidth=1.4 if crit else 0.8, mutation_aspect=1.0, zorder=3)) 
        ax.plot([x - bw / 2 + 0.05, x + bw / 2 - 0.05],
                [y + 0.035, y + 0.035], color=HAIR, lw=0.5, zorder=4)
        ax.text(x, y + 0.175, f"{k}  {TITLE[k]}", ha="center", va="center",
                fontsize=6.4, color=INK, fontweight="bold" if crit else
                "normal", zorder=5)
        ax.text(x, y + 0.088, f"{d:.0f} d,  {crew:d} crew", ha="center",
                va="center", fontsize=6.0, color=MUTED, zorder=5)
        ax.text(x - bw / 2 + 0.07, y - 0.045,
                f"ES {es[k]:.0f}", ha="left", va="center", fontsize=6.0,
                color=SUB, zorder=5)
        ax.text(x + bw / 2 - 0.07, y - 0.045,
                f"EF {ef[k]:.0f}", ha="right", va="center", fontsize=6.0,
                color=SUB, zorder=5)
        ax.text(x - bw / 2 + 0.07, y - 0.155,
                f"LS {ls[k]:.0f}", ha="left", va="center", fontsize=6.0,
                color=SUB, zorder=5)
        ax.text(x + bw / 2 - 0.07, y - 0.155,
                f"LF {lf[k]:.0f}", ha="right", va="center", fontsize=6.0,
                color=SUB, zorder=5)
        if not crit:
            ax.text(x, y - bh / 2 - 0.075, f"float {float_[k]:.0f} d",
                    ha="center", va="top", fontsize=6.0, color=GOLD, zorder=5)
    return pos


def draw_float(ax, float_, order):
    keys = sorted(ACTS, key=lambda k: (float_[k], k))
    vals = [float_[k] for k in keys]
    y = np.arange(len(keys))[::-1]
    cols = [RED if v <= 1e-9 else lighten(GOLD, 0.30) for v in vals]
    ax.barh(y, np.maximum(vals, 0.0), height=0.60, color=cols,
            edgecolor="white", linewidth=0.5, zorder=3)
    for yy, k, v in zip(y, keys, vals):
        if v <= 1e-9:
            ax.text(0.12, yy, "critical", ha="left", va="center",
                    fontsize=6.0, color=RED, fontweight="bold")
        else:
            ax.text(v + 0.12, yy, f"{v:.0f}", ha="left", va="center",
                    fontsize=6.0, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=6.3)
    ax.set_xlim(0.0, max(vals) + 1.6)
    ax.set_xlabel("total float (days)", fontsize=7.2)
    n_crit = sum(1 for v in vals if v <= 1e-9)
    ax.set_title(f"{n_crit:d} of {len(keys):d} activities have zero float",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    style_axes(ax)


def draw_resource(ax, es, lev, dur):
    before = resource_profile(es, dur)
    after = resource_profile(lev, dur)
    t = np.arange(len(before) + 1, dtype=float)
    b = np.append(before, before[-1])
    a = np.append(after, after[-1])
    ax.fill_between(t, b, step="post", color="#e2e2e2", edgecolor=GREY,
                    linewidth=0.7, zorder=2, label="early-start plan")
    ax.step(t, a, where="post", color=GREEN, lw=1.5, zorder=4,
            label="after levelling")
    ax.axhline(before.max(), color=GREY, lw=0.8, ls=(0, (4, 2.2)), zorder=3)
    ax.axhline(after.max(), color=GREEN, lw=0.8, ls=(0, (4, 2.2)), zorder=3)
    ax.text(dur * 0.015, before.max() + 0.25, f"peak {before.max():.0f}",
            ha="left", va="bottom", fontsize=6.2, color=SUB)
    ax.text(dur * 0.015, after.max() + 0.25, f"peak {after.max():.0f}",
            ha="left", va="bottom", fontsize=6.2, color=GREEN)
    ax.set_xlim(0.0, dur)
    ax.set_ylim(0.0, before.max() * 1.30)
    ax.set_xlabel("project time (days)", fontsize=7.2)
    ax.set_ylabel("crew on site (people)", fontsize=7.2)
    cut = 100.0 * (before.max() - after.max()) / before.max()
    ax.set_title(f"Levelling shaves the crew peak by {cut:.0f}%",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="upper right", fontsize=6.3, frameon=False,
              handlelength=1.4, labelspacing=0.3)
    style_axes(ax)


def main():
    set_pub_style()
    order, es, ef, ls, lf, float_, dur, succ = cpm()
    rank = ranks(order)
    lev = level(es, lf, float_, order, dur)

    fig = plt.figure(figsize=(W, H))
    ax_net = inch_axes(fig, (0.34, 1.02, 6.14, 4.44), aspect=None)
    ax_net.set_xlim(0.0, 6.14)
    ax_net.set_ylim(0.0, 4.44)
    ax_top = axes_inches(fig, (7.34, 3.62, 2.80, 1.84))
    ax_bot = axes_inches(fig, (7.34, 1.14, 2.80, 1.66))

    draw_network(ax_net, es, ef, ls, lf, float_, dur, rank, bw=0.94, bh=0.52,
                 span_x=5.54, span_y=3.30, y0=0.62)

    ax_net.text(0.06, 4.38,
                "Activity-on-node network, critical chain drawn in red",
                ha="left", va="top", fontsize=9.0, fontweight="bold",
                color=INK)
    crit_chain = "-".join(k for k in order if float_[k] <= 1e-9)
    ax_net.text(0.06, 4.16,
                f"Critical path {crit_chain} fixes the project at "
                f"{dur:.0f} days; every other activity carries float",
                ha="left", va="top", fontsize=7.0, color=SUB)

    handles = [Patch(facecolor=lighten(RED, 0.80), edgecolor=RED,
                     label="critical activity (float 0 d)"),
               Patch(facecolor="white", edgecolor=GREY,
                     label="activity with float"),
               Line2D([], [], color=RED, lw=1.9, label="critical path"),
               Line2D([], [], color=HAIR, lw=0.9, label="precedence link")]
    ax_net.legend(handles=handles, loc="lower center",
                  bbox_to_anchor=(0.50, -0.012), fontsize=6.5, frameon=False,
                  ncol=4, handlelength=1.5, columnspacing=1.5)

    draw_float(ax_top, float_, order)
    draw_resource(ax_bot, es, lev, dur)

    panel_labels_inches(fig, [ax_top, ax_bot], dx=-0.62, dy=0.08, start=1)
    panel_labels_inches(fig, [ax_net], dx=-0.16, dy=0.02)

    savefig(fig, "862_sch_critical_path.png")


if __name__ == "__main__":
    main()
