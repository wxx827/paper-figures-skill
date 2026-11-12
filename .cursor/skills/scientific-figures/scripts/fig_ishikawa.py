"""Ishikawa cause-and-effect diagram for the six manufacturing M-categories."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

from pubstyle import set_pub_style, savefig

INK = "#333333"
NAVY = "#2c4a6e"
BLUE = "#4c78a8"
PALE = "#dbe4ee"
GREY = "#7d868f"
ALERT = "#b23a48"

SPINE_Y = 50.0
RIB_DX, RIB_DY = 14.0, 30.0          # horizontal / vertical run of a big bone
ANCHORS = (28.0, 48.0, 68.0)         # where the big bones meet the spine
STUB = 3.0                            # length of a small-bone stub
BOX_W, BOX_H = 17.0, 6.6

# (category, [(cause, is_root_cause), ...])
TOP = [
    ("MAN", [("Inadequate training", False),
             ("Shift-handover gap", False),
             ("Operator fatigue", False)]),
    ("MACHINE", [("Spindle bearing wear", True),
                 ("Worn nozzle tip", False),
                 ("Clamp force drift", False),
                 ("Overdue PM interval", False)]),
    ("MATERIAL", [("Resin batch variation", True),
                  ("Moisture in pellets", False),
                  ("Unapproved supplier", False)]),
]
BOTTOM = [
    ("METHOD", [("Wrong cure profile", False),
                ("No work instruction", False),
                ("Manual re-torque step", False)]),
    ("MEASUREMENT", [("Gauge R&R above 30%", True),
                     ("Calibration overdue", False),
                     ("Ambiguous datum", False)]),
    ("ENVIRONMENT", [("Night humidity swing", False),
                     ("Conveyor vibration", False),
                     ("Draught at dock door", False)]),
]


def big_bone(ax, xa, up):
    """Draw one category rib and return its start / end points."""
    sign = 1 if up else -1
    x0, y0 = xa, SPINE_Y
    x1, y1 = xa - RIB_DX, SPINE_Y + sign * RIB_DY
    ax.plot([x0, x1], [y0, y1], color=BLUE, lw=1.9, zorder=3,
            solid_capstyle="round")
    return (x0, y0), (x1, y1)


def category_box(ax, cx, cy, label):
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0,rounding_size=1.4", facecolor=PALE,
        edgecolor=NAVY, linewidth=1.0, zorder=4))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=7.6,
            fontweight="bold", color=NAVY, zorder=5)


def small_bones(ax, p0, p1, causes):
    """Evenly spaced horizontal stubs along a rib, text on the open side."""
    (x0, y0), (x1, y1) = p0, p1
    n = len(causes)
    fracs = [0.26 + 0.70 * (i / max(n - 1, 1)) for i in range(n)]
    for f, (text, root) in zip(fracs, causes):
        bx = x0 + f * (x1 - x0)
        by = y0 + f * (y1 - y0)
        col = ALERT if root else GREY
        ax.plot([bx, bx + STUB], [by, by], color=col, lw=1.0, zorder=3,
                solid_capstyle="round")
        if root:
            ax.plot([bx + STUB], [by], "o", ms=3.4, color=ALERT, zorder=4)
        ax.text(bx + STUB + 1.5, by, text, ha="left", va="center",
                fontsize=6.2, color=ALERT if root else INK,
                fontweight="bold" if root else "normal", zorder=4)


def main():
    set_pub_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.855, bottom=0.06)
    ax.set_xlim(0, 100)
    ax.set_ylim(6, 94)
    ax.axis("off")

    # ---- fish tail and spine ----------------------------------------------
    ax.add_patch(Polygon([(1.5, 57.0), (7.0, SPINE_Y), (1.5, 43.0)],
                         closed=True, facecolor=PALE, edgecolor=NAVY,
                         linewidth=0.9, zorder=2))
    ax.add_patch(FancyArrowPatch(
        (6.2, SPINE_Y), (83.4, SPINE_Y), arrowstyle="-|>", mutation_scale=20,
        linewidth=2.2, color=NAVY, shrinkA=0, shrinkB=0, zorder=3))

    # ---- effect (fish head) -----------------------------------------------
    hx, hw = 83.8, 15.7
    ax.add_patch(FancyBboxPatch(
        (hx, 40.0), hw, 20.0, boxstyle="round,pad=0,rounding_size=1.6",
        facecolor=NAVY, edgecolor=NAVY, linewidth=1.0, zorder=4))
    cx = hx + hw / 2
    ax.text(cx, 55.4, "EFFECT", ha="center", va="center", fontsize=6.6,
            fontweight="bold", color="#b9cadd", zorder=5)
    ax.text(cx, 49.4, "Solder void rate\n1.9% at final test", ha="center",
            va="center", fontsize=8.0, fontweight="bold", color="white",
            linespacing=1.5, zorder=5)
    ax.text(cx, 43.0, "target $\\leq$ 0.5%", ha="center", va="center",
            fontsize=6.4, color="#b9cadd", zorder=5)

    # ---- six category ribs -------------------------------------------------
    for xa, (label, causes) in zip(ANCHORS, TOP):
        p0, p1 = big_bone(ax, xa, up=True)
        category_box(ax, p1[0], p1[1] + BOX_H / 2 + 0.9, label)
        small_bones(ax, p0, p1, causes)
    for xa, (label, causes) in zip(ANCHORS, BOTTOM):
        p0, p1 = big_bone(ax, xa, up=False)
        category_box(ax, p1[0], p1[1] - BOX_H / 2 - 0.9, label)
        small_bones(ax, p0, p1, causes)

    # ---- titles and key ----------------------------------------------------
    ax.plot([1.0], [9.0], "o", ms=3.4, color=ALERT, clip_on=False, zorder=5)
    ax.text(3.0, 9.0, "verified root cause \u2014 confirmed by 5-why and "
            "measurement data", ha="left", va="center", fontsize=6.5,
            color=ALERT, fontweight="bold")
    ax.text(99.0, 9.0, "grey stubs: candidate causes not yet confirmed",
            ha="right", va="center", fontsize=6.5, color=GREY)

    savefig(fig, "246_ishikawa.png")


if __name__ == "__main__":
    main()
