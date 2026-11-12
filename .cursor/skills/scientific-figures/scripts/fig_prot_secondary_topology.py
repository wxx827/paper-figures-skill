"""TOPS-style topology cartoon of a Rossmann-like alpha/beta fold.

Main panel: secondary-structure elements as symbols - beta strands are arrows
in the sheet plane, alpha helices are cylinders - placed at their spatial
position in the fold and joined by the chain path, so the classical
right-handed beta-alpha-beta crossovers are visible.  Companion panel: the same
elements laid out along the sequence, which is what the cartoon abstracts away.

Positions are declared once in ELEMENTS and every symbol, connector, label and
sequence block is derived from that table.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from pubstyle import set_pub_style, savefig, lighten
from figkit import panel_label_inches, axes_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
H_RED, H_BLUE = "#b23a48", "#3b5b92"
COIL = "#9a9a9a"

L_SEQ = 160
# name, kind, first residue, last residue, x, y  (1-based, inclusive)
ELEMENTS = [
    (r"$\beta$1", "E", 8, 15, 0.0, 0.0),
    (r"$\alpha$1", "H", 22, 38, -0.85, 1.95),
    (r"$\beta$2", "E", 45, 51, -1.5, 0.0),
    (r"$\alpha$2", "H", 58, 74, -2.35, -1.95),
    (r"$\beta$3", "E", 80, 86, -3.0, 0.0),
    (r"$\alpha$3", "H", 93, 110, 1.05, 1.95),
    (r"$\beta$4", "E", 116, 122, 1.5, 0.0),
    (r"$\alpha$4", "H", 128, 142, 2.40, -1.95),
    (r"$\beta$5", "E", 148, 154, 3.0, 0.0),
]

SW, SBODY, SHEAD = 0.46, 0.98, 0.58      # strand arrow geometry
HR = 0.46                                # helix radius


def strand_arrow(ax, x, y, color, up=True):
    """Upward (parallel) or downward strand arrow, centred on (x, y)."""
    h = SBODY + SHEAD
    s = 1.0 if up else -1.0
    y0 = y - s * h / 2.0
    pts = [(x - SW / 2, y0), (x + SW / 2, y0),
           (x + SW / 2, y0 + s * SBODY), (x + SW * 0.95, y0 + s * SBODY),
           (x, y0 + s * h),
           (x - SW * 0.95, y0 + s * SBODY), (x - SW / 2, y0 + s * SBODY)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="white",
                         linewidth=0.9, zorder=5))
    return (x, y0), (x, y0 + s * h)


def helix_cylinder(ax, x, y, color):
    """Circle-plus-ring cylinder glyph, the TOPS symbol for a helix."""
    ax.add_patch(Circle((x, y), HR, facecolor=color, edgecolor="white",
                        linewidth=0.9, zorder=5))
    ax.add_patch(Circle((x, y), HR * 0.60, facecolor="none",
                        edgecolor=lighten(color, 0.55), linewidth=0.8,
                        zorder=6))
    return (x - HR, y), (x + HR, y)


def main():
    set_pub_style(base_font=8)

    FW, FH = 7.2, 4.5
    fig = plt.figure(figsize=(FW, FH))
    ax = axes_inches(fig, (0.22, 0.40, 4.42, 3.42), aspect="equal")
    ax.axis("off")
    axs = axes_inches(fig, (5.42, 0.95, 0.62, 2.10))

    # ---- sheet backdrop ----------------------------------------------------
    ax.add_patch(FancyBboxPatch((-3.62, -1.02), 7.24, 2.04,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#f5f5f5", edgecolor="#dcdcdc",
                                linewidth=0.8, ls=(0, (3.4, 2.2)),
                                mutation_aspect=1.0, zorder=1))

    # ---- symbols -----------------------------------------------------------
    anchors = {}
    for name, kind, a, b, x, y in ELEMENTS:
        if kind == "E":
            bot, top = strand_arrow(ax, x, y, H_BLUE, up=True)
            anchors[name] = dict(entry=bot, exit=top, xy=(x, y))
            ax.text(x, y - 0.02, name, fontsize=7.0, color="white",
                    fontweight="bold", ha="center", va="center", zorder=7,
                    rotation=90)
            ax.text(x, -(SBODY + SHEAD) / 2 - 0.24, f"{a}-{b}", fontsize=6.2,
                    color=MUTED, ha="center", va="top", zorder=4)
        else:
            left, right = helix_cylinder(ax, x, y, H_RED)
            anchors[name] = dict(entry=left, exit=right, xy=(x, y))
            ax.text(x, y, name, fontsize=7.0, color="white",
                    fontweight="bold", ha="center", va="center", zorder=7)
            dy = HR + 0.12
            ax.text(x, y + (dy if y > 0 else -dy), f"{a}-{b}", fontsize=6.2,
                    color=MUTED, ha="center",
                    va="bottom" if y > 0 else "top", zorder=4)

    # ---- chain path --------------------------------------------------------
    names = [e[0] for e in ELEMENTS]
    for k in range(len(names) - 1):
        p0 = anchors[names[k]]["exit"]
        p1 = anchors[names[k + 1]]["entry"]
        above = max(p0[1], p1[1]) > 0.6
        rad = 0.32 if above else -0.32
        if ELEMENTS[k][1] == "E" and ELEMENTS[k + 1][1] == "H":
            rad = 0.26 if above else -0.26
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-", lw=1.0, color=COIL,
                                    shrinkA=2.5, shrinkB=2.5,
                                    connectionstyle=f"arc3,rad={rad}"),
                    zorder=3)

    n_in = anchors[names[0]]["entry"]
    ax.annotate("", xy=n_in, xytext=(n_in[0] - 0.75, n_in[1] - 0.85),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COIL,
                                mutation_scale=8, shrinkA=1, shrinkB=2))
    ax.text(n_in[0] - 0.82, n_in[1] - 0.92, "N", fontsize=8.5,
            fontweight="bold", color=INK, ha="right", va="top")
    c_out = anchors[names[-1]]["exit"]
    ax.annotate("", xy=(c_out[0] + 0.72, c_out[1] + 0.60), xytext=c_out,
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=COIL,
                                mutation_scale=8, shrinkA=2, shrinkB=1))
    ax.text(c_out[0] + 0.78, c_out[1] + 0.66, "C", fontsize=8.5,
            fontweight="bold", color=INK, ha="left", va="bottom")

    ax.annotate(r"right-handed $\beta$-$\alpha$-$\beta$ crossover",
                xy=(-2.10, 1.30), xytext=(-4.35, 2.95), fontsize=6.8,
                color=SUB, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, mutation_scale=6,
                                color=SUB, shrinkA=1, shrinkB=3))
    ax.annotate("all strands point\nthe same way", xy=(1.50, 0.85),
                xytext=(2.55, 2.88), fontsize=6.8, color=H_BLUE,
                ha="center", va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-|>", lw=0.7, mutation_scale=6,
                                color=H_BLUE, shrinkA=1, shrinkB=3))

    ax.set_xlim(-4.55, 4.55)
    ax.set_ylim(-3.05, 3.35)
    ax.set_title("Topology cartoon: parallel sheet, strand order 3-2-1-4-5",
                 fontsize=9.0, fontweight="bold", pad=3.0)

    # ---- companion: the same elements along the sequence -------------------
    axs.add_patch(Rectangle((0.0, 0.5), 1.0, L_SEQ, facecolor="#eeeeee",
                            edgecolor="none", zorder=1))
    for name, kind, a, b, _, _ in ELEMENTS:
        col = H_BLUE if kind == "E" else H_RED
        axs.add_patch(Rectangle((0.0, a), 1.0, b - a + 1, facecolor=col,
                                edgecolor="white", linewidth=0.6, zorder=3))
        axs.text(1.22, 0.5 * (a + b), name, fontsize=6.4, color=col,
                 ha="left", va="center", zorder=4)
    axs.set_xlim(0.0, 1.0)
    axs.set_ylim(L_SEQ + 0.5, 0.5)
    axs.set_xticks([])
    axs.set_yticks(np.arange(20, L_SEQ + 1, 40))
    axs.tick_params(axis="y", labelsize=6.4, length=2.2)
    axs.set_ylabel("residue index (-)", fontsize=7.4, labelpad=2.0)
    for s in axs.spines.values():
        s.set_visible(False)
    axs.set_title("Sequence\ntrack", fontsize=8.5, fontweight="bold", pad=4.0)

    frac_h = sum(b - a + 1 for _, k, a, b, _, _ in ELEMENTS if k == "H")
    frac_e = sum(b - a + 1 for _, k, a, b, _, _ in ELEMENTS if k == "E")
    fig.text(5.30 / FW, 0.62 / FH,
             f"{L_SEQ} residues\n"
             r"$\alpha$" + f" helix  {100.0 * frac_h / L_SEQ:.0f}%\n"
             r"$\beta$" + f" strand {100.0 * frac_e / L_SEQ:.0f}%\n"
             f"coil    {100.0 * (L_SEQ - frac_h - frac_e) / L_SEQ:.0f}%",
             fontsize=6.5, color=MUTED, ha="left", va="top", linespacing=1.6)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=6.0, mfc=H_RED, mec="white",
               mew=0.8, label=r"$\alpha$ helix"),
        Line2D([], [], marker="^", ls="none", ms=6.2, mfc=H_BLUE, mec="white",
               mew=0.8, label=r"$\beta$ strand"),
        Line2D([], [], color=COIL, lw=1.0, label="connecting loop"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.34, 0.005), ncol=3, fontsize=7.0,
               frameon=False, handletextpad=0.5, columnspacing=1.6)

    panel_label_inches(fig, ax, "(a)", dx=-0.14, dy=0.10)
    panel_label_inches(fig, axs, "(b)", dx=-0.40, dy=0.28)
    savefig(fig, "526_prot_secondary_topology.png")


if __name__ == "__main__":
    main()
