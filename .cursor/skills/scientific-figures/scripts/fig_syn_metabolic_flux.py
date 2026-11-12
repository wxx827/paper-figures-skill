"""Central-carbon flux map before and after strain engineering.

Every reaction is drawn twice on the same edge - a grey lane for the parental
strain and a coloured lane for the engineered one - with the lane width set by
the flux it carries, so a rerouted pathway is visible as a change in weight
rather than as a number the reader has to look up.  The flux table is
stoichiometrically balanced at every branch point, and the two supporting
panels (the split at the acetyl-CoA node and the yield comparison) are both
computed from that same table, so the pie, the bars and the line widths cannot
disagree.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge

from pubstyle import set_pub_style, lighten, savefig
from figkit import arrow, axes_inches, inch_axes, panel_label_inches

GREY = "#808080"        # parental / wild-type lane
BLUE = "#3b5b92"        # engineered lane
GREEN = "#2e7d5b"       # product-forming reaction
RED = "#b23a48"         # by-product / carbon loss
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"

W, H = 10.4, 6.4
MW, MH = 6.00, 5.40
UNIT = r"mmol/(gDW$\cdot$h)"

# name -> (x, y, display label, kind)
NODES = {
    "GLC": (1.05, 4.98, "Glucose", "in"),
    "G6P": (1.05, 4.28, "G6P", "met"),
    "F6P": (1.05, 3.58, "F6P", "met"),
    "G3P": (1.05, 2.88, "G3P", "met"),
    "PEP": (1.05, 2.10, "PEP", "met"),
    "PYR": (1.05, 1.30, "PYR", "met"),
    "6PG": (2.85, 4.28, "6PG", "met"),
    "RU5P": (2.85, 3.58, "Ru5P", "met"),
    "OAA": (2.85, 2.10, "OAA", "met"),
    "ACCOA": (2.85, 1.30, "AcCoA", "hub"),
    "TCA": (4.70, 2.10, "TCA cycle", "met"),
    "PROD": (4.70, 1.30, "Product", "prod"),
    "ACE": (4.70, 0.55, "Acetate", "loss"),
}

# (from, to, wild-type flux, engineered flux, lane colour, label spec)
# label spec is None, ("txt", text, t, offset) or ("num", t, offset)
EDGES = [
    ("GLC", "G6P", 10.0, 10.0, BLUE, ("txt", "uptake 10.0", 0.50, 0.62)),
    ("G6P", "F6P", 7.0, 3.5, BLUE, None),
    ("G6P", "6PG", 3.0, 6.5, BLUE, ("txt", "pentose phosphate", 0.50, 0.24)),
    ("6PG", "RU5P", 3.0, 6.5, BLUE, None),
    ("RU5P", "F6P", 2.0, 4.3, BLUE, None),
    ("F6P", "G3P", 9.0, 7.8, BLUE, None),
    ("G3P", "PEP", 9.0, 7.8, BLUE, None),
    ("PEP", "PYR", 6.5, 4.8, BLUE, None),
    ("PEP", "OAA", 2.5, 3.0, BLUE, None),
    ("PYR", "ACCOA", 6.5, 4.8, BLUE, None),
    ("OAA", "TCA", 2.5, 3.0, BLUE, None),
    ("ACCOA", "TCA", 2.2, 1.2, BLUE, ("num", 0.34, 0.26)),
    ("ACCOA", "PROD", 1.5, 3.4, GREEN, ("num", 0.50, -0.25)),
    ("ACCOA", "ACE", 2.8, 0.2, RED, ("num", 0.30, -0.22)),
]
SPLIT = [("TCA cycle", 2.2, 1.2, BLUE), ("Product", 1.5, 3.4, GREEN),
         ("Acetate", 2.8, 0.2, RED)]


def lw_of(flux):
    return 0.45 + 0.60 * flux


def node_box(name):
    x, y, label, kind = NODES[name]
    return dict(x=x, y=y, w=0.30 + 0.108 * len(label), h=0.32, label=label,
                kind=kind)


def draw_node(ax, name):
    b = node_box(name)
    fc, ec, tc = "white", RULE, INK
    if b["kind"] == "prod":
        fc, ec, tc = GREEN, GREEN, "white"
    elif b["kind"] == "loss":
        fc, ec, tc = RED, RED, "white"
    elif b["kind"] in ("hub", "in"):
        fc, ec = SOFT, RULE
    ax.add_patch(FancyBboxPatch(
        (b["x"] - b["w"] / 2, b["y"] - b["h"] / 2), b["w"], b["h"],
        boxstyle="round,pad=0,rounding_size=0.07", facecolor=fc, edgecolor=ec,
        lw=0.9, mutation_aspect=1.0, zorder=6))
    ax.text(b["x"], b["y"], b["label"], ha="center", va="center",
            fontsize=7.0, color=tc, zorder=7)


def gap_of(b, ux, uy, margin=0.09):
    tx = (b["w"] / 2) / abs(ux) if abs(ux) > 1e-9 else 1e9
    ty = (b["h"] / 2) / abs(uy) if abs(uy) > 1e-9 else 1e9
    return min(tx, ty) + margin


def draw_edge(ax, src, dst, f_wt, f_eng, color, spec):
    a, b = node_box(src), node_box(dst)
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    d = float(np.hypot(dx, dy))
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    off = 0.085
    ga, gb = gap_of(a, ux, uy), gap_of(b, ux, uy)

    for sign, flux, col in [(+1.0, f_wt, GREY), (-1.0, f_eng, color)]:
        lw = lw_of(flux)
        arrow(ax, (a["x"] + px * off * sign, a["y"] + py * off * sign),
              (b["x"] + px * off * sign, b["y"] + py * off * sign),
              color=col, lw=lw, ms=6.0 + 1.25 * lw, gap_a=ga, gap_b=gb,
              zorder=4 if sign > 0 else 5)

    if not spec:
        return
    kind = spec[0]
    t, shift = spec[-2], spec[-1]
    xm = a["x"] + ux * d * t + px * shift
    ym = a["y"] + uy * d * t + py * shift
    if kind == "num":
        ax.text(xm, ym, f"{f_wt:.1f} $\\rightarrow$ {f_eng:.1f}", ha="center",
                va="center", fontsize=6.4, color=color, zorder=8,
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                          edgecolor="none"))
    else:
        ax.text(xm, ym, spec[1], ha="center", va="center", fontsize=6.6,
                color=MUTED, zorder=8)


def width_key(ax, x0, y0):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), 1.66, 1.36, boxstyle="round,pad=0,rounding_size=0.06",
        facecolor="white", edgecolor=HAIR, lw=0.7, mutation_aspect=1.0,
        zorder=3))
    ax.text(x0 + 0.12, y0 + 1.24, "lane width = flux", ha="left", va="top",
            fontsize=6.8, color=INK, zorder=5)
    y = y0 + 0.96
    for flux in (10.0, 5.0, 1.0):
        ax.plot([x0 + 0.16, x0 + 0.62], [y, y], color=SUB, lw=lw_of(flux),
                solid_capstyle="butt", zorder=5)
        ax.text(x0 + 0.72, y, f"{flux:.0f}", ha="left", va="center",
                fontsize=6.4, color=SUB, zorder=5)
        y -= 0.24
    ax.text(x0 + 0.12, y + 0.02, UNIT, ha="left", va="center", fontsize=6.2,
            color=MUTED, zorder=5)


def lane_key(ax):
    items = [(GREY, "parental strain"), (GREEN, "product-forming step"),
             (BLUE, "engineered strain"), (RED, "by-product loss")]
    for i, (col, name) in enumerate(items):
        x0 = 0.18 + (i % 2) * 2.10
        y = 0.76 - (i // 2) * 0.26
        ax.plot([x0, x0 + 0.40], [y, y], color=col, lw=2.6,
                solid_capstyle="butt", zorder=5)
        ax.text(x0 + 0.50, y, name, ha="left", va="center", fontsize=6.6,
                color=SUB, zorder=5)


def draw_network(ax):
    ax.text(0.05, 5.32, "Flux redistribution in central carbon metabolism",
            ha="left", va="center", fontsize=8.6, fontweight="bold",
            color=INK)
    for src, dst, f_wt, f_eng, col, spec in EDGES:
        draw_edge(ax, src, dst, f_wt, f_eng, col, spec)
    for name in NODES:
        draw_node(ax, name)
    width_key(ax, 4.20, 3.62)
    lane_key(ax)

    up = [e for e in EDGES if e[3] > e[2] + 1e-9]
    ax.text(0.18, 0.20, f"{len(up)} of {len(EDGES)} reactions carry more flux "
            f"after engineering; every branch point stays balanced",
            ha="left", va="center", fontsize=6.6, color=MUTED)


def split_panel(ax, w, h):
    ax.text(0.06, h - 0.10, "Carbon split at the acetyl-CoA node", ha="left",
            va="top", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(0.06, h - 0.32, "centre value = total node turnover in " + UNIT,
            ha="left", va="top", fontsize=6.2, color=MUTED)

    r, cy = 0.56, 1.06
    for k, (cx, idx, name) in enumerate([(w * 0.29, 1, "parental"),
                                         (w * 0.71, 2, "engineered")]):
        vals = np.array([s[idx] for s in SPLIT], dtype=float)
        frac = vals / vals.sum()
        start = 90.0
        for (label, _, _, col), f in zip(SPLIT, frac):
            ext = 360.0 * f
            face = col if k else lighten(col, 0.55)
            ax.add_patch(Wedge((cx, cy), r, start - ext, start, width=0.28,
                               facecolor=face, edgecolor="white", lw=0.9,
                               zorder=4))
            mid = np.deg2rad(start - ext / 2.0)
            out = f < 0.14
            rr = r + 0.19 if out else r - 0.14
            ax.text(cx + rr * np.cos(mid), cy + rr * np.sin(mid),
                    f"{f * 100:.0f}%", ha="center", va="center", fontsize=6.4,
                    color=INK if out else ("white" if k else SUB), zorder=6)
            start -= ext
        ax.text(cx, cy, f"{vals.sum():.1f}", ha="center", va="center",
                fontsize=7.6, color=INK, zorder=6)
        ax.text(cx, cy - r - 0.14, name, ha="center", va="top", fontsize=7.0,
                color=INK)

    x = 0.16
    for label, _, _, col in SPLIT:
        ax.add_patch(FancyBboxPatch(
            (x, 0.06), 0.17, 0.14,
            boxstyle="round,pad=0,rounding_size=0.03", facecolor=col,
            edgecolor="white", lw=0.6, mutation_aspect=1.0, zorder=5))
        ax.text(x + 0.24, 0.13, label, ha="left", va="center", fontsize=6.5,
                color=SUB)
        x += 1.12


def yield_panel(ax):
    glc = [e for e in EDGES if e[0] == "GLC"][0]
    prod = [e for e in EDGES if e[1] == "PROD"][0]
    ace = [e for e in EDGES if e[1] == "ACE"][0]
    tca = [e for e in EDGES if e[1] == "TCA"]

    wt = np.array([prod[2] / glc[2], ace[2] / glc[2],
                   sum(e[2] for e in tca) / glc[2]])
    en = np.array([prod[3] / glc[3], ace[3] / glc[3],
                   sum(e[3] for e in tca) / glc[3]])
    names = [r"product $Y_{P/S}$", r"acetate $Y_{A/S}$", "TCA drain"]
    cols = [GREEN, RED, BLUE]

    y = np.arange(len(names))[::-1].astype(float)
    ax.barh(y + 0.19, wt, height=0.34, color=GREY, edgecolor="white", lw=0.6)
    for yi, v, c in zip(y, en, cols):
        ax.barh(yi - 0.19, v, height=0.34, color=c, edgecolor="white", lw=0.6)

    for yi, a, b in zip(y, wt, en):
        ax.text(max(a, b) + 0.016, yi, f"{(b / a - 1.0) * 100:+.0f}%",
                ha="left", va="center", fontsize=6.6,
                color=GREEN if b >= a else RED)
    ax.text(0.012, y[0] + 0.19, "parental", ha="left", va="center",
            fontsize=6.4, color="white")
    ax.text(0.012, y[0] - 0.19, "engineered", ha="left", va="center",
            fontsize=6.4, color="white")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.0)
    ax.set_xlabel("carbon yield on glucose (mol/mol)", fontsize=7.4)
    ax.set_xlim(0.0, max(wt.max(), en.max()) * 1.30)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("Yield shifts from by-product to product", fontsize=8.2,
                 fontweight="bold", color=INK)
    return wt, en


def main():
    set_pub_style()
    fig = plt.figure(figsize=(W, H))

    ax_net = inch_axes(fig, (0.28, 0.42, MW, MH))
    aw = 3.45
    ax_pie = inch_axes(fig, (6.72, 3.16, aw, 2.36))
    ax_bar = axes_inches(fig, (7.55, 0.80, 2.42, 1.70))

    draw_network(ax_net)
    split_panel(ax_pie, aw, 2.36)
    wt, en = yield_panel(ax_bar)

    fig.text(0.021, 6.16 / H,
             "Rerouting glucose through the pentose phosphate pathway more "
             "than doubles product flux and starves acetate overflow",
             ha="left", va="center", fontsize=11.0, fontweight="bold",
             color=INK)
    fig.text(0.021, 0.11 / H,
             f"Fluxes are steady-state estimates in {UNIT}, normalised to a "
             f"glucose uptake of 10 and balanced at every branch point. "
             f"Product yield rises from {wt[0]:.2f} to {en[0]:.2f} mol/mol.",
             ha="left", va="bottom", fontsize=6.5, color=MUTED)

    panel_label_inches(fig, ax_net, "(a)", dx=-0.16, dy=0.02)
    panel_label_inches(fig, ax_pie, "(b)", dx=-0.16, dy=0.02)
    panel_label_inches(fig, ax_bar, "(c)", dx=-0.99, dy=0.14)

    savefig(fig, "710_syn_metabolic_flux.png")


if __name__ == "__main__":
    main()
