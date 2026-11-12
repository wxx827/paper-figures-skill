"""Figure 1201 - Failure-mode taxonomy of a world-model driving stack.

One table drives the whole page.  ``CATS`` lists fourteen leaf modes with a
count each; every other number on the figure is derived from it:

    category count  = sum of its leaves          (drawn as the middle nodes)
    total           = sum of the categories      (drawn as the root)
    share           = category / total           (the bars on the right)

so the tree and the bars cannot disagree, and the four printed shares are
asserted to add up to 100.00 % before the file is written.  Node *area* is
proportional to the count, which is why the sizes are set through
``scatter(s=...)`` - a marker stays circular whatever the axes aspect is,
unlike a ``Circle`` patch (DESIGN_SYSTEM section 12.8).

Change a leaf count and everything follows; do not hand-type a category total.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pubstyle import savefig, set_pub_style

SEED = 1606  # original id; figure number is 1201

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#d5d5d5"
CARD, CARD_EDGE = "#e2e8ee", "#c3cdd6"
RULE = "#4a5560"

# (category, ladder colour, [(leaf mode, episodes), ...])
# A blue ladder, deepest first: the four categories are separated by depth,
# not by hue, so the figure spends nothing from the colour budget.
CATS = [
    ("Perception", "#0b3d63", [
        ("missed detection, occluded agent", 118),
        ("false positive on road spray", 74),
        ("depth error > 1.0 m beyond 40 m", 63),
        ("track identity switch", 41)]),
    ("Prediction", "#2e6f9e", [
        ("intent misread at a merge", 96),
        ("multi-modal collapse to one mode", 58),
        ("rollout drift > 3 s into the horizon", 47)]),
    ("Planning", "#5f9dc6", [
        ("no feasible manoeuvre found", 69),
        ("over-conservative stall", 52),
        ("chance constraint relaxed too far", 38),
        ("goal oscillation between lanes", 24)]),
    ("Execution", "#9cc3dd", [
        ("actuator saturation on steer", 45),
        ("control latency spike > 120 ms", 33),
        ("tracking overshoot in a curve", 22)]),
]

AREA = 3.05          # points^2 per episode - node area encodes frequency
XR, XC, XL = 0.058, 0.243, 0.470      # root / category / leaf node columns
ELBOW_A, ELBOW_B = 0.152, 0.372       # where each right-angle turn happens


def layout():
    """Leaf rows, category rows and the derived counts, top to bottom."""
    rows, y, groups = [], 0.0, []
    for gi, (name, col, leaves) in enumerate(CATS):
        first = y
        for (lab, n) in leaves:
            rows.append(dict(group=gi, label=lab, n=n, row=y))
            y += 1.0
        groups.append(dict(name=name, colour=col, first=first, last=y - 1.0,
                           n=sum(n for _, n in leaves), gi=gi))
        y += 0.85                                   # gap between categories
    span = y - 0.85 - 1.0
    for r in rows:
        r["y"] = 0.955 - 0.905 * (r["row"] / span)
    for g in groups:
        g["y"] = 0.955 - 0.905 * (0.5 * (g["first"] + g["last"]) / span)
        g["top"] = 0.955 - 0.905 * ((g["first"] - 0.46) / span)
        g["bot"] = 0.955 - 0.905 * ((g["last"] + 0.46) / span)
    return rows, groups


def radius_scale(fig, ax):
    """Axes x-units per point, so an elbow can stop at a node's edge."""
    fig.canvas.draw()
    bb = ax.get_window_extent()
    return (1.0 / 72.0) / (bb.width / fig.dpi)


def elbow(ax, x0, y0, x1, y1, xm, colour, lw):
    ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=colour, lw=lw,
            solid_joinstyle="miter", solid_capstyle="butt", zorder=2)


def main():
    set_pub_style(base_font=8.5)
    rng = np.random.default_rng(SEED)          # fixed seed; layout is exact
    rows, groups = layout()

    total = sum(g["n"] for g in groups)
    assert total == sum(r["n"] for r in rows), "leaves do not sum to the root"
    shares = [100.0 * g["n"] / total for g in groups]
    shown = [f"{s:.2f}" for s in shares]
    assert abs(sum(map(float, shown)) - 100.0) < 5e-3, \
        "the four printed shares do not add up to 100.00 %"

    fig = plt.figure(figsize=(10.6, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.315], left=0.012,
                          right=0.988, top=0.858, bottom=0.088, wspace=0.045)
    axt = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    for ax in (axt, axb):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

    u = radius_scale(fig, axt)                 # axes x-units per point

    # ---- group bands (they also carry the ink this sparse layout needs) ----
    for g in groups:
        axt.add_patch(FancyBboxPatch(
            (0.196, g["bot"]), 0.792, g["top"] - g["bot"],
            boxstyle="round,pad=0.0,rounding_size=0.010",
            facecolor=CARD, edgecolor=CARD_EDGE, lw=0.6, zorder=0))

    # ---- root -------------------------------------------------------------
    root_s = AREA * total * 0.42
    rr = np.sqrt(root_s / np.pi) * u
    axt.scatter([XR], [0.5], s=root_s, color="#08263f", edgecolors="white",
                linewidths=1.3, zorder=4)
    axt.text(XR, 0.5, f"{total}", ha="center", va="center", color="white",
             fontsize=9.4, fontweight="bold", zorder=5)
    axt.text(XR, 0.5 - rr - 0.016, "all logged\nsafety events", ha="center",
             va="top", fontsize=7.4, color=INK, linespacing=1.45, zorder=5)

    # ---- categories -------------------------------------------------------
    for g, sh in zip(groups, shown):
        s = AREA * g["n"] * 0.72
        rc = np.sqrt(s / np.pi) * u
        elbow(axt, XR + rr, 0.5, XC - rc, g["y"], ELBOW_A, RULE, 1.05)
        axt.scatter([XC], [g["y"]], s=s, color=g["colour"],
                    edgecolors="white", linewidths=1.1, zorder=4)
        axt.text(XC, g["y"], f"{g['n']}", ha="center", va="center",
                 color="white" if g["gi"] < 2 else "#0b2c47", fontsize=8.2,
                 fontweight="bold", zorder=5)
        axt.text(XC, g["y"] + rc + 0.014, g["name"], ha="center", va="bottom",
                 fontsize=8.8, fontweight="bold", color=INK, zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.3,
                           alpha=0.94))
        axt.text(XC, g["y"] - rc - 0.014, f"{sh} % of {total}", ha="center",
                 va="top", fontsize=7.0, color=SUB, zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.94))

        for r in [q for q in rows if q["group"] == g["gi"]]:
            sl = AREA * r["n"]
            rl = np.sqrt(sl / np.pi) * u
            elbow(axt, XC + rc, g["y"], XL - rl, r["y"], ELBOW_B,
                  g["colour"], 0.9)
            axt.scatter([XL], [r["y"]], s=sl, color=g["colour"],
                        edgecolors="white", linewidths=0.9, zorder=4)
            axt.text(XL + rl + 0.012, r["y"], r["label"], ha="left",
                     va="center", fontsize=7.6, color=INK, zorder=5)
            axt.text(0.982, r["y"], f"{r['n']}", ha="right", va="center",
                     fontsize=7.6, color=SUB, fontweight="bold", zorder=5)

    # ---- size key (three real markers, drawn to the same scale) -----------
    axt.text(0.080, 0.088, "node area $\\propto$ episodes", ha="center",
             va="bottom", fontsize=7.0, color=SUB, zorder=5)
    for i, n in enumerate((25, 60, 120)):
        x = 0.042 + 0.038 * i
        axt.scatter([x], [0.045], s=AREA * n, color="#8fb6d4",
                    edgecolors="white", linewidths=0.8, zorder=4)
        axt.text(x, 0.010, f"{n}", ha="center", va="bottom", fontsize=6.8,
                 color=SUB, zorder=5)

    # ---- right: share bars, aligned row-for-row with the category nodes ---
    axb.add_patch(FancyBboxPatch(
        (0.012, 0.075), 0.976, 0.905,
        boxstyle="round,pad=0.0,rounding_size=0.020", facecolor="white",
        edgecolor=CARD_EDGE, lw=0.7, zorder=0))
    axb.text(0.5, 0.995, f"Share of all {total} events  (%)", ha="center",
             va="bottom", fontsize=8.6, fontweight="bold", color=INK)
    bx0, bx1 = 0.075, 0.740
    smax = max(shares)
    for k in (0, 10, 20, 30, 40):
        gx = bx0 + (bx1 - bx0) * k / (smax * 1.06)
        if gx <= bx1:
            axb.plot([gx, gx], [0.098, 0.935], color=HAIR, lw=0.6, zorder=1)
    axb.text(bx0, 0.052, "grid lines every 10 %; bars share one scale",
             ha="left", va="center", fontsize=6.8, color=MUTED, zorder=3)
    for g, s, sh in zip(groups, shares, shown):
        w = (bx1 - bx0) * s / (smax * 1.06)
        h = 0.044
        axb.add_patch(FancyBboxPatch(
            (bx0, g["y"] - h), w, 2 * h,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            facecolor=g["colour"], edgecolor="white", lw=0.8, zorder=2))
        axb.text(bx0 + w + 0.020, g["y"], f"{sh} %", ha="left", va="center",
                 fontsize=8.0, fontweight="bold", color=INK, zorder=3)
        axb.text(bx0 + 0.008, g["y"] + h + 0.014,
                 f"{g['name']}    n = {g['n']}", ha="left", va="bottom",
                 fontsize=7.4, color=INK, zorder=3)

    # ---- headline + the arithmetic, printed rather than implied -----------
    lead = groups[int(np.argmax(shares))]
    assert rows[0]["n"] == max(r["n"] for r in rows), \
        "the footer names the wrong leading leaf mode"
    savefig(fig, "1201_wm_safe_failure_taxonomy.png")


if __name__ == "__main__":
    main()
