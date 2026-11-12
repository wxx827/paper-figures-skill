"""Central carbon metabolism: glycolysis, the TCA cycle and oxidative phosphorylation.

The left panel is a drawn pathway, not a plot: metabolites are rounded boxes,
enzymatic steps are arrows carrying the enzyme name, near-equilibrium steps get
a double head and the two committed, rate-limiting steps (PFK-1 and isocitrate
dehydrogenase) get a heavy red arrow.  Three background cards separate the
cytosolic, matrix and inner-membrane compartments so the reader can see where
each block of chemistry happens.

The right panel is the accounting that the diagram implies: a waterfall of the
ATP debits and credits of one glucose molecule.  Every number printed on the
diagram and on the bars comes from the same ``STEPS`` table, so the two panels
cannot drift apart.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import darken, lighten, savefig, set_pub_style
from figkit import (arrow, axes_inches, block, inch_axes, panel_card,
                    panel_labels_inches, route, text_card)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
BLUE, GREEN, RED, GOLD = "#3b5b92", "#2e7d5b", "#b23a48", "#c9a227"
SOLID = "#3a3a3a"

# ATP ledger for one glucose, aerobic, malate-aspartate shuttle.
# kind: "sub" = substrate-level phosphorylation, "ox" = oxidative, "cost".
STEPS = [
    ("Glycolysis\nATP formed", 4.0, "sub"),
    ("Glycolysis\nATP invested", -2.0, "cost"),
    ("2 NADH\ncytosolic", 5.0, "ox"),
    ("PDH\n2 NADH", 5.0, "ox"),
    ("TCA\n2 GTP", 2.0, "sub"),
    ("TCA\n6 NADH", 15.0, "ox"),
    ("TCA\n2 $FADH_2$", 3.0, "ox"),
]
KIND_COLOR = {"sub": BLUE, "ox": GREEN, "cost": RED}
KIND_NAME = {"sub": "substrate-level phosphorylation",
             "ox": "oxidative phosphorylation",
             "cost": "ATP invested"}


def box_edge(bx, target, gap=0.03):
    """Point where the ray from a box centre to ``target`` leaves the box.

    Arrows drawn centre-to-centre and then shrunk by a constant look ragged
    where boxes of different widths meet; solving for the actual rectangle
    edge makes every head land the same distance from the frame it points at.
    """
    cx, cy, w, h = bx["x"], bx["y"], bx["w"], bx["h"]
    dx, dy = target[0] - cx, target[1] - cy
    d = float(np.hypot(dx, dy))
    if d < 1e-9:
        return (cx, cy)
    tx = (w / 2.0) / abs(dx) if abs(dx) > 1e-9 else np.inf
    ty = (h / 2.0) / abs(dy) if abs(dy) > 1e-9 else np.inf
    t = min(tx, ty)
    return (cx + dx * t + dx / d * gap, cy + dy * t + dy / d * gap)


def metabolite(ax, x, y, w, h, label, base, fs=6.6):
    return block(ax, x, y, w, h, label, color=lighten(base, 0.80),
                 ec=base, text_color=INK, fontsize=fs, lw=0.9, rounding=0.055)


def step_arrow(ax, a, b, name=None, limiting=False, reversible=False,
               label_dy=-0.14, va="top", fs=6.2, rad=0.0):
    """Enzymatic step between two metabolite boxes, edge to edge."""
    p0 = box_edge(a, b["c"])
    p1 = box_edge(b, a["c"])
    arrow(ax, p0, p1, color=RED if limiting else RULE,
          lw=2.1 if limiting else 1.0, ms=13 if limiting else 10, rad=rad,
          style="<|-|>" if reversible else "-|>", zorder=6)
    if name:
        mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
        ax.text(mx, my + label_dy, name, ha="center", va=va, fontsize=fs,
                color=RED if limiting else SUB, zorder=7)


def draw_pathway(ax, w, h):
    """The three-compartment pathway map on an inch canvas."""
    panel_card(ax, 2.85, 4.21, 5.60, 1.18, fc="#f5f5f5", ec="none",
               zorder=-2, rounding=0.06)
    panel_card(ax, 2.85, 2.60, 5.60, 1.76, fc="#f0f0f0", ec="none",
               zorder=-2, rounding=0.06)
    panel_card(ax, 2.85, 0.82, 5.60, 1.52, fc="#f5f5f5", ec="none",
               zorder=-2, rounding=0.06)
    for txt, yy in (("Glycolysis  -  cytosol", 4.72),
                    ("Tricarboxylic acid cycle  -  mitochondrial matrix",
                     3.40),
                    ("Oxidative phosphorylation  -  inner membrane", 1.50)):
        ax.text(0.16, yy, txt, ha="left", va="top", fontsize=7.2,
                fontweight="bold", color=SUB, zorder=2)

    # ---- glycolysis --------------------------------------------------------
    gly_names = ["Glucose", "G6P", "F1,6BP", "G3P  (x2)", "Pyruvate  (x2)"]
    gx = 0.57 + 1.14 * np.arange(5)
    gly = [metabolite(ax, x, 4.02, 0.86, 0.28, n, BLUE)
           for x, n in zip(gx, gly_names)]
    step_arrow(ax, gly[0], gly[1], "hexokinase")
    step_arrow(ax, gly[1], gly[2], "PFK-1  rate-limiting", limiting=True)
    step_arrow(ax, gly[2], gly[3], "aldolase  (reversible)", reversible=True)
    step_arrow(ax, gly[3], gly[4], "GAPDH / pyruvate kinase")
    ax.text(gx[3] + 0.57, 4.24, "+2 NADH,  +4 ATP", ha="center", va="bottom",
            fontsize=6.0, color=GREEN, zorder=7)

    # ---- pyruvate -> acetyl-CoA -------------------------------------------
    acoa = metabolite(ax, 4.35, 3.10, 0.92, 0.26, "Acetyl-CoA", GREEN)
    route(ax, [(gx[4], 3.88), (gx[4], 3.10), (acoa["r"][0] + 0.03, 3.10)],
          color=RULE, lw=1.0, ms=10)
    ax.text(gx[4] - 0.08, 3.54, "PDH complex", ha="right", va="center",
            fontsize=6.2, color=SUB, zorder=7)

    # ---- TCA ring ----------------------------------------------------------
    ring_names = ["Citrate", "Isocitrate", r"$\alpha$-KG", "Succinyl-CoA",
                  "Malate", "OAA"]
    cx0, cy0, rx, ry = 1.95, 2.58, 0.92, 0.52
    ang = np.deg2rad(np.array([90.0, 30.0, -30.0, -90.0, -150.0, 150.0]))
    ring = [metabolite(ax, cx0 + rx * np.cos(a), cy0 + ry * np.sin(a),
                       0.70, 0.24, n, GREEN, fs=6.2)
            for a, n in zip(ang, ring_names)]
    for i in range(6):
        nxt = ring[(i + 1) % 6]
        step_arrow(ax, ring[i], nxt, rad=-0.18,
                   limiting=(i == 1),
                   name=None)
    step_arrow(ax, acoa, ring[0], "citrate synthase", label_dy=0.12,
               va="bottom")
    ax.text(3.30, 2.58, "IDH\nrate-limiting", ha="left", va="center",
            fontsize=6.0, color=RED, linespacing=1.3, zorder=7)
    text_card(ax, 4.00, 1.88, 1.60, 0.62,
              "3 NADH, 1 $FADH_2$\n1 GTP, 2 $CO_2$", fc="white", ec="#cccccc",
              fs=6.2, color=SUB, ha="center", title="per turn (x2)",
              title_fs=6.4, title_color=INK, zorder=4)

    # ---- oxidative phosphorylation ----------------------------------------
    ox_names = ["NADH / $FADH_2$", "Complexes I-IV", "$H^+$ gradient",
                "ATP synthase", "ATP + $H_2O$"]
    ox_x = 0.61 + 1.12 * np.arange(5)
    ox = [metabolite(ax, x, 1.00, 0.94, 0.28, n, GOLD)
          for x, n in zip(ox_x, ox_names)]
    for i, name in enumerate(["$e^-$ transfer", "proton pumping",
                              "chemiosmosis", "$F_1F_o$ rotation"]):
        step_arrow(ax, ox[i], ox[i + 1], name, fs=6.0)
    route(ax, [(cx0, ring[3]["b"][1] - 0.03), (cx0, 1.65), (ox_x[0], 1.65),
               (ox_x[0], ox[0]["t"][1] + 0.03)], color=GREEN, lw=1.0, ms=10)
    ax.text(2.85, 0.36, "about 2.5 ATP per NADH and 1.5 ATP per $FADH_2$",
            ha="center", va="bottom", fontsize=6.4, color=MUTED, zorder=3)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)


def draw_waterfall(ax, total, oxid):
    names = [s[0] for s in STEPS]
    vals = np.array([s[1] for s in STEPS], dtype=float)
    kinds = [s[2] for s in STEPS]
    start = np.concatenate([[0.0], np.cumsum(vals)])[:-1]

    xx = np.arange(len(vals) + 1, dtype=float)
    for i, (v, k) in enumerate(zip(vals, kinds)):
        ax.bar(xx[i], v, bottom=start[i], width=0.62,
               color=KIND_COLOR[k], edgecolor="white", lw=0.6, zorder=3)
        top = start[i] + v
        ax.text(xx[i], max(top, start[i]) + 0.55, f"{v:+.0f}", ha="center",
                va="bottom", fontsize=6.2, color=KIND_COLOR[k], zorder=5)
        ax.plot([xx[i] - 0.31, xx[i + 1] + 0.31], [top, top], lw=0.6,
                ls=(0, (2.5, 2.0)), color="#b3b3b3", zorder=2)
    ax.bar(xx[-1], total, width=0.62, color=SOLID, edgecolor="white", lw=0.6,
           zorder=3)
    ax.text(xx[-1], total + 0.55, f"{total:.0f}", ha="center", va="bottom",
            fontsize=7.0, fontweight="bold", color=SOLID, zorder=5)

    ax.axhline(0.0, color="#808080", lw=0.8, zorder=2)
    ax.set_xticks(xx)
    ax.set_xticklabels(names + ["Net yield"], fontsize=6.0, rotation=38.0,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("cumulative ATP yield (ATP per glucose)", fontsize=7.6)
    ax.set_ylim(-3.5, total * 1.18)
    ax.set_xlim(-0.7, len(vals) + 0.7)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax.set_title(f"{oxid / total:.0%} of the ATP is oxidative",
                 fontsize=8.5, fontweight="bold", pad=6)
    ax.legend(handles=[Line2D([0], [0], color=KIND_COLOR[k], lw=4.0,
                              solid_capstyle="butt", label=KIND_NAME[k])
                       for k in ("sub", "ox", "cost")],
              loc="upper left", frameon=False, fontsize=6.4,
              handlelength=1.5, labelspacing=0.30, borderaxespad=0.15)


def main():
    set_pub_style()
    np.random.default_rng(660)          # fixed seed; this figure is analytic

    vals = np.array([s[1] for s in STEPS], dtype=float)
    total = float(vals.sum())
    oxid = float(sum(v for (_, v, k) in STEPS if k == "ox"))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_p = inch_axes(fig, (0.42, 0.72, 5.70, 4.85))
    ax_w = axes_inches(fig, (7.10, 1.37, 2.95, 3.55))

    draw_pathway(ax_p, 5.70, 4.85)
    draw_waterfall(ax_w, total, oxid)


    panel_labels_inches(fig, [ax_p, ax_w], dx=-0.32, dy=0.12)

    print(f"ATP ledger: {[s[1] for s in STEPS]}  net {total:.0f} ATP/glucose")
    print(f"oxidative {oxid:.0f} ATP ({oxid / total:.1%}), "
          f"substrate-level net {total - oxid:.0f} ATP")
    savefig(fig, "660_nut_metabolic_pathway.png")


if __name__ == "__main__":
    main()
