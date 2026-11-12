"""Wm blue venn capability: what an embodied world model has to hold at once.

Three half-transparent discs from a single blue ladder -- no second hue, the
three steps carry the three sets.  The pairwise lenses are too small to letter
in place, so each one gets a hairline leader out to the margin; only the
triple intersection is labelled inside.  Drawn on an inch canvas with an equal
aspect so the discs are genuinely circular (DESIGN_SYSTEM.md section 12.8:
``Circle`` on a normal axes comes out as an ellipse).

Everything here is a layout constant; there is no data behind the figure.
"""
from matplotlib.patches import Circle
from pubstyle import set_pub_style, savefig
from figkit import inch_canvas

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
LEAD = "#8a939c"

W, H = 7.6, 5.6
OX, OY = 3.80, 2.55       # centroid of the three disc centres
R = 1.35
ALPHA = 0.46

# (name, centre, fill step, edge step, where the exclusive label sits).
#
# The three fills are *adjacent* ladder steps on purpose.  Alpha compositing
# only darkens an overlap when the disc painted on top is darker than what is
# already there, so three far-apart steps (say BLUE[1] / BLUE[3] / BLUE[5])
# make each intersection come out LIGHTER than the darkest disc -- the Venn
# then reads backwards, with the exclusive region looking "more" than the
# shared one.  Keeping the steps one apart bounds that effect; set identity
# is carried by the edge colour and the label colour instead of by fill depth.
SETS = [
    ("Perception", (OX - 0.78, OY + 0.45), 2, 0, (2.40, 3.62)),
    ("Prediction", (OX + 0.78, OY + 0.45), 3, 1, (5.20, 3.62)),
    ("Planning", (OX, OY - 0.90), 4, 2, (3.80, 0.85)),
]

# (label, lens anchor, elbow, text anchor, ha, accent?)
LENSES = [
    ("Tracking &\nfiltering", (3.80, 3.30), (3.80, 4.52), (3.80, 4.60),
     "center", False),
    ("Reactive\ncontrol", (2.93, 2.05), (1.63, 1.30), (1.14, 1.30),
     "right", False),
    ("Imagination", (4.67, 2.05), (5.97, 1.30), (6.46, 1.30), "left", True),
]


def leader(ax, p0, p1, p2):
    ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], color=LEAD,
            lw=0.8, solid_capstyle="round", zorder=6)
    ax.plot([p0[0]], [p0[1]], marker="o", ms=3.2, color=LEAD, zorder=7)


def main():
    set_pub_style()
    fig, ax = inch_canvas(W, H)

    ax.text(0.15, 5.38, "Capabilities an embodied world model must hold at "
            "once", fontsize=10.5, fontweight="bold", color=INK, ha="left",
            va="center")
    ax.text(0.15, 5.12, "One blue ladder, three steps: the sets are told "
            "apart by depth, never by hue",
            fontsize=7.2, color="#555555", ha="left", va="center")

    for name, centre, fill, edge, lab in SETS:
        ax.add_patch(Circle(centre, R, facecolor=BLUE[fill], alpha=ALPHA,
                            edgecolor=BLUE[edge], linewidth=1.1, zorder=3))
    for name, centre, fill, edge, lab in SETS:
        ax.text(lab[0], lab[1], name, ha="center", va="center",
                fontsize=10.0, fontweight="bold", color=BLUE[edge], zorder=8)

    for label, anchor, elbow, tip, ha, hot in LENSES:
        leader(ax, anchor, elbow, tip)
        ax.text(tip[0] + (0.10 if ha == "left" else
                          -0.10 if ha == "right" else 0.0),
                tip[1] + (0.10 if ha == "center" else 0.0), label, ha=ha,
                va="bottom" if ha == "center" else "center", fontsize=7.6,
                fontweight="bold", linespacing=1.25,
                color=ACCENT if hot else INK, zorder=8)

    ax.text(OX, OY, "Embodied\nWorld Model", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=BLUE[0], linespacing=1.3,
            zorder=8)

    ax.text(0.15, 0.12, "Region areas are schematic; only the containment "
            "relations are asserted.", fontsize=6.8, color="#777777",
            ha="left", va="center")

    savefig(fig, "1104_wm_blue_venn_capability.png")


if __name__ == "__main__":
    main()
