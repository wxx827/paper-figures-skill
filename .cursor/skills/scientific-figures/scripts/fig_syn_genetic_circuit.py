"""Toggle-switch genetic circuit drawn with SBOL-style glyphs.

The main panel is one inch canvas: three transcription units share a common
backbone geometry (bent-arrow promoter, half-disc ribosome binding site,
arrow-shaped coding sequence, T-bar terminator), and the regulatory layer is
drawn on separate horizontal corridors so that no two edges ever cross.  Blunt
heads are repression, the pointed head is activation.  The truth table and the
state graph on the right are generated from the same boolean model that the
caption quotes, so the wiring in the diagram and the logic in the table cannot
drift apart.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge

from pubstyle import set_pub_style, savefig
from figkit import arrow, inch_axes, panel_label_inches

GREEN = "#2e7d5b"       # promoters, activation, reporter output
BLUE = "#3b5b92"        # coding sequences, engineered parts
GOLD = "#c9a227"        # ribosome binding sites
RED = "#b23a48"         # repression, OFF state
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"

W, H = 10.4, 6.4
MW, MH = 6.00, 5.40                     # main diagram canvas, inches
BB0, BB1 = 0.45, 4.05                   # DNA backbone extent
X_PROM, X_RBS, X_CDS, X_TERM = 0.90, 1.55, 2.35, 3.55
X_PROT = 5.05

ROWS = [
    dict(y=4.10, prom=r"$P_{BAD}$", gene="lacI", prot="LacI",
         cds_color=BLUE, unit="Unit 1  repressor arm"),
    dict(y=2.70, prom=r"$P_{lac}$", gene="tetR", prot="TetR",
         cds_color=BLUE, unit="Unit 2  repressor arm"),
    dict(y=1.30, prom=r"$P_{tet}$", gene="gfp", prot="GFP",
         cds_color=GREEN, unit="Unit 3  reporter"),
]


def circuit_state(iptg, atc):
    """Boolean model of the wiring drawn in the main panel."""
    laci_active = not iptg                 # IPTG inactivates LacI
    tetr_made = not laci_active            # LacI represses P_lac
    tetr_active = tetr_made and not atc    # aTc inactivates TetR
    gfp = not tetr_active                  # TetR represses P_tet
    return laci_active, tetr_active, gfp


# ---------------------------------------------------------------- SBOL glyphs
def promoter(ax, x, y, color=GREEN):
    ax.plot([x, x], [y, y + 0.42], color=color, lw=1.4, zorder=4,
            solid_capstyle="round")
    arrow(ax, (x, y + 0.42), (x + 0.48, y + 0.42), color=color, lw=1.4,
          ms=8, zorder=4)


def rbs(ax, x, y, color=GOLD):
    ax.add_patch(Wedge((x, y), 0.17, 0.0, 180.0, facecolor=color,
                       edgecolor="white", lw=0.7, zorder=4))


def cds(ax, x, y, label, color=BLUE, w=1.25, h=0.36, tip=0.24):
    pts = [(x - w / 2, y - h / 2), (x + w / 2 - tip, y - h / 2),
           (x + w / 2, y), (x + w / 2 - tip, y + h / 2),
           (x - w / 2, y + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="white",
                         lw=0.9, zorder=4))
    ax.text(x - tip / 2, y, label, ha="center", va="center", fontsize=7.6,
            color="white", style="italic", zorder=6)


def terminator(ax, x, y, color=SUB):
    ax.plot([x, x], [y, y + 0.30], color=color, lw=1.4, zorder=4)
    ax.plot([x - 0.17, x + 0.17], [y + 0.30, y + 0.30], color=color, lw=1.4,
            zorder=4)


def protein_node(ax, x, y, label, color):
    ax.add_patch(FancyBboxPatch(
        (x - 0.44, y - 0.22), 0.88, 0.44,
        boxstyle="round,pad=0,rounding_size=0.10", facecolor=color,
        edgecolor="white", lw=0.9, mutation_aspect=1.0, zorder=5))
    ax.text(x, y, label, ha="center", va="center", fontsize=7.6,
            color="white", zorder=6)


def repress(ax, pts, color=RED, half=0.15, lw=1.1):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw,
            solid_capstyle="round", solid_joinstyle="round", zorder=5)
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    dx, dy = xb - xa, yb - ya
    d = (dx * dx + dy * dy) ** 0.5 or 1.0
    px, py = -dy / d, dx / d
    ax.plot([xb - px * half, xb + px * half], [yb - py * half, yb + py * half],
            color=color, lw=1.5, solid_capstyle="butt", zorder=6)


# ------------------------------------------------------------------ main panel
def draw_circuit(ax):
    ax.text(0.10, 5.22, "Cross-repressing toggle switch with a fluorescent "
            "read-out", ha="left", va="center", fontsize=8.6,
            fontweight="bold", color=INK)

    # activation input feeding the top promoter
    ax.add_patch(FancyBboxPatch(
        (0.90, 4.72), 1.72, 0.40,
        boxstyle="round,pad=0,rounding_size=0.09", facecolor="white",
        edgecolor=GREEN, lw=0.9, mutation_aspect=1.0, zorder=5))
    ax.text(1.76, 4.92, "AraC + L-arabinose", ha="center", va="center",
            fontsize=6.8, color=GREEN, zorder=6)
    arrow(ax, (1.20, 4.72), (1.20, 4.60), color=GREEN, lw=1.1, ms=9, zorder=5)

    for row in ROWS:
        y = row["y"]
        ax.plot([BB0, BB1], [y, y], color=RULE, lw=1.1, zorder=3)
        promoter(ax, X_PROM, y)
        rbs(ax, X_RBS, y)
        cds(ax, X_CDS, y, row["gene"], color=row["cds_color"])
        terminator(ax, X_TERM, y)
        ax.text(X_PROM + 0.52, y + 0.50, row["prom"], ha="left", va="bottom",
                fontsize=7.0, color=GREEN)
        ax.text(BB0, y - 0.30, row["unit"], ha="left", va="top", fontsize=6.4,
                color=MUTED)
        arrow(ax, (BB1 + 0.10, y), (X_PROT - 0.56, y), color=MUTED, lw=0.9,
              ls=(0, (2.6, 1.8)), ms=8, zorder=4)
        protein_node(ax, X_PROT, y, row["prot"], row["cds_color"])

    ax.text(BB1 + 0.14, ROWS[0]["y"] + 0.16, "translation", ha="left",
            va="bottom", fontsize=6.2, color=MUTED)

    # LacI blocks P_lac (unit 2); corridor sits between units 1 and 2
    repress(ax, [(X_PROT, ROWS[0]["y"] - 0.22), (X_PROT, 3.45), (X_PROM, 3.45),
                 (X_PROM, 3.30)])
    # TetR blocks P_tet (unit 3); corridor sits between units 2 and 3
    repress(ax, [(X_PROT, ROWS[1]["y"] - 0.22), (X_PROT, 2.05), (X_PROM, 2.05),
                 (X_PROM, 1.90)])

    for row, ind in zip(ROWS[:2], ["IPTG", "aTc"]):
        y = row["y"]
        ax.text(5.98, y + 0.12, ind, ha="right", va="bottom", fontsize=7.0,
                color=RED)
        repress(ax, [(5.94, y), (5.62, y)], half=0.13)

    key = [(GREEN, "promoter"), (GOLD, "RBS"), (BLUE, "CDS"),
           (SUB, "terminator")]
    x = 0.48
    for color, name in key:
        ax.add_patch(Circle((x, 0.45), 0.075, facecolor=color,
                            edgecolor="white", lw=0.6, zorder=4))
        ax.text(x + 0.14, 0.45, name, ha="left", va="center", fontsize=6.6,
                color=SUB)
        x += 1.02
    repress(ax, [(4.62, 0.45), (4.92, 0.45)], half=0.10)
    ax.text(5.02, 0.45, "repression", ha="left", va="center", fontsize=6.6,
            color=SUB)
    arrow(ax, (4.62, 0.16), (4.92, 0.16), color=GREEN, lw=1.1, ms=9)
    ax.text(5.02, 0.16, "activation", ha="left", va="center", fontsize=6.6,
            color=SUB)


# ------------------------------------------------------------- truth table
def draw_table(ax, w, h):
    cols = ["IPTG", "aTc", "LacI", "TetR", "GFP"]
    combos = [(False, False), (True, False), (False, True), (True, True)]
    x0, y_top = 0.16, h - 0.62
    cw = (w - 2 * x0) / len(cols)
    rh = 0.44

    ax.text(0.10, h - 0.12, "Steady-state logic of the switch", ha="left",
            va="top", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(x0 + cw, h - 0.34, "inputs (inducers)", ha="center", va="top",
            fontsize=6.4, color=MUTED)
    ax.text(x0 + 3.5 * cw, h - 0.34, "outputs (proteins)", ha="center",
            va="top", fontsize=6.4, color=MUTED)

    for j, name in enumerate(cols):
        ax.text(x0 + (j + 0.5) * cw, y_top + 0.10, name, ha="center",
                va="bottom", fontsize=7.0, color=INK)
    ax.plot([x0, x0 + len(cols) * cw], [y_top + 0.04, y_top + 0.04],
            color=RULE, lw=0.9)
    ax.plot([x0 + 2 * cw, x0 + 2 * cw], [y_top + 0.16, y_top - 4 * rh - 0.06],
            color=HAIR, lw=0.8, ls=(0, (2.6, 2.0)))

    for i, (iptg, atc) in enumerate(combos):
        laci, tetr, gfp = circuit_state(iptg, atc)
        vals = [iptg, atc, laci, tetr, gfp]
        yc = y_top - (i + 0.5) * rh
        for j, v in enumerate(vals):
            xc = x0 + (j + 0.5) * cw
            fc = (BLUE if j < 2 else (GREEN if j == 4 else BLUE)) if v else SOFT
            ax.add_patch(FancyBboxPatch(
                (xc - cw * 0.40, yc - rh * 0.36), cw * 0.80, rh * 0.72,
                boxstyle="round,pad=0,rounding_size=0.05", facecolor=fc,
                edgecolor="white" if v else HAIR, lw=0.7,
                mutation_aspect=1.0, zorder=3))
            txt = ("+" if v else "-") if j < 2 else ("high" if v else "low")
            ax.text(xc, yc, txt, ha="center", va="center", fontsize=6.6,
                    color="white" if v else SUB, zorder=5)

    off = [c for c in combos if not circuit_state(*c)[2]]
    ax.text(0.10, y_top - 4 * rh - 0.30,
            f"Reporter is low in {len(off)} of {len(combos)} input states: "
            "only IPTG\nwithout aTc lets TetR accumulate and shut off "
            r"$P_{tet}$",
            ha="left", va="top", fontsize=6.6, color=SUB, linespacing=1.5)


# ---------------------------------------------------------- state transitions
def draw_states(ax, w, h):
    ax.text(0.10, h - 0.12, "Bistable memory between two attractors",
            ha="left", va="top", fontsize=8.6, fontweight="bold", color=INK)

    cx1, cx2, cy, r = 0.92, w - 0.92, h * 0.50 - 0.10, 0.52
    for cx, color, title, sub in [
            (cx1, GREEN, "State I", "GFP high\nTetR low"),
            (cx2, RED, "State II", "GFP low\nTetR high")]:
        ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=color,
                            lw=1.4, zorder=4))
        ax.text(cx, cy + 0.17, title, ha="center", va="center", fontsize=7.4,
                color=color, fontweight="bold", zorder=5)
        ax.text(cx, cy - 0.14, sub, ha="center", va="center", fontsize=6.6,
                color=SUB, linespacing=1.4, zorder=5)

    arrow(ax, (cx1 + r, cy + 0.18), (cx2 - r, cy + 0.18), color=RULE, lw=1.0,
          rad=-0.32, ms=9)
    arrow(ax, (cx2 - r, cy - 0.18), (cx1 + r, cy - 0.18), color=RULE, lw=1.0,
          rad=-0.32, ms=9)
    ax.text((cx1 + cx2) / 2.0, cy + 0.74, "IPTG pulse", ha="center",
            va="bottom", fontsize=6.8, color=INK)
    ax.text((cx1 + cx2) / 2.0, cy - 0.74, "aTc pulse", ha="center", va="top",
            fontsize=6.8, color=INK)

    for cx in (cx1, cx2):
        arrow(ax, (cx + 0.19, cy + 0.484), (cx - 0.19, cy + 0.484),
              color=MUTED, lw=0.9, rad=1.9, ms=7)
        ax.text(cx, cy + r + 0.30, "holds", ha="center", va="bottom",
                fontsize=6.4, color=MUTED)
    ax.text(w / 2.0, 0.10, "no inducer: both states persist for > 20 "
            "generations", ha="center", va="bottom", fontsize=6.6, color=MUTED)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(W, H))

    ax_main = inch_axes(fig, (0.28, 0.42, MW, MH))
    aw, ah_top, ah_bot = 3.45, 2.55, 2.35
    ax_tab = inch_axes(fig, (6.72, 3.24, aw, ah_top))
    ax_st = inch_axes(fig, (6.72, 0.42, aw, ah_bot))

    draw_circuit(ax_main)
    draw_table(ax_tab, aw, ah_top)
    draw_states(ax_st, aw, ah_bot)

    fig.text(0.027, 6.14 / H, "Mutual repression plus one activated input "
             "turns three transcription units into a two-state memory",
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.027, 0.11 / H,
             "Glyphs follow the SBOL visual standard; blunt-ended lines are "
             "repression and the pointed head is activation. Inducer "
             "concentrations are 1 mM IPTG and 100 ng/mL aTc.",
             ha="left", va="bottom", fontsize=6.5, color=MUTED)

    panel_label_inches(fig, ax_main, "(a)", dx=-0.16, dy=0.02)
    panel_label_inches(fig, ax_tab, "(b)", dx=-0.16, dy=0.02)
    panel_label_inches(fig, ax_st, "(c)", dx=-0.16, dy=0.02)

    savefig(fig, "708_syn_genetic_circuit.png")


if __name__ == "__main__":
    main()
