"""Open challenges in world modelling, matched to mitigations by weighted arcs.

Two columns of the house two-tone card -- challenges on the left in a muted
red, mitigations on the right in a muted green -- joined by arcs whose width
and opacity encode how much of the challenge that mitigation actually takes
off the table.  The arcs are deliberately the blue of the rest of this family:
they are the connective tissue, not a third category, and keeping them one
neutral hue is what stops a ten-edge bipartite graph turning into confetti.

The pairing is one-to-many on purpose.  A survey table would let each row have
exactly one fix; drawing it shows what a table hides, that every challenge also
leans on a mitigation belonging to a different row.

Association strengths are an illustrative editorial judgement, not a
meta-analysis.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from pubstyle import COLORS, savefig, set_pub_style
from figkit import inch_canvas

FW, FH = 10.4, 6.4
ROUND = 0.065

CHAL_HEAD, CHAL_BODY = "#b0525f", "#fbeef0"
MITI_HEAD, MITI_BODY = "#4f8f72", "#eef6f1"
LINK = "#3d6294"

CHALLENGES = [
    ("Compounding rollout error", "long horizons drift away"),
    ("Object permanence loss", "occluded things stop existing"),
    ("Physics violations", "interpenetration, free energy"),
    ("Sim-to-real gap", "policies fail on real hardware"),
    ("Evaluation is ad hoc", "no shared rollout protocol"),
]
MITIGATIONS = [
    ("Latent-space training", "predict features, not pixels"),
    ("Persistent slot memory", "objects keep their own token"),
    ("Physics-informed loss", "penalise impossible states"),
    ("Domain randomisation", "widen the training support"),
    ("Shared benchmarks", "fixed horizons, fixed seeds"),
]
# challenge index, mitigation index, association strength in [0, 1]
LINKS = [(0, 0, 0.90), (0, 2, 0.40),
         (1, 1, 0.95), (1, 0, 0.35),
         (2, 2, 0.85), (2, 1, 0.30),
         (3, 3, 0.90), (3, 0, 0.45),
         (4, 4, 0.85), (4, 3, 0.30)]

# The columns are deliberately wide and the arc lane deliberately narrow
# (2.20 in, about 21 % of the canvas).  A wide lane spanned by seven hairlines
# reads as a hole in the middle of the figure; shortening the chords makes the
# arcs steeper, denser and easier to follow, and hands the reclaimed width to
# the cards, which is where the content actually is.
CARD_W, CARD_H, HEAD_H = 3.70, 0.80, 0.34
LX0, RX0 = 0.40, 6.30
ROW_Y = [4.78 - 0.92 * i for i in range(5)]
BANNER_Y = 5.48


def two_tone_card(ax, x0, cy, head, body, head_c, body_c, z=4):
    y0 = cy - CARD_H / 2.0
    box = f"round,pad=0,rounding_size={ROUND}"
    ax.add_patch(FancyBboxPatch((x0 + 0.032, y0 - 0.038), CARD_W, CARD_H,
                                boxstyle=box, facecolor="#dfe3e8",
                                edgecolor="none", mutation_aspect=1.0,
                                zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x0, y0), CARD_W, CARD_H, boxstyle=box,
                                facecolor=body_c, edgecolor="none",
                                mutation_aspect=1.0, zorder=z))
    hy = y0 + CARD_H - HEAD_H
    ax.add_patch(FancyBboxPatch((x0, hy), CARD_W, HEAD_H, boxstyle=box,
                                facecolor=head_c, edgecolor="none",
                                mutation_aspect=1.0, zorder=z + 1))
    ax.add_patch(Rectangle((x0, hy), CARD_W, ROUND, facecolor=head_c,
                           edgecolor="none", zorder=z + 2))
    ax.add_patch(FancyBboxPatch((x0, y0), CARD_W, CARD_H, boxstyle=box,
                                facecolor="none", edgecolor=head_c,
                                linewidth=0.9, mutation_aspect=1.0,
                                zorder=z + 3))
    cx = x0 + CARD_W / 2.0
    ax.text(cx, hy + HEAD_H / 2.0, head, ha="center", va="center",
            color="white", fontsize=8.4, fontweight="bold", zorder=z + 4)
    ax.text(cx, y0 + (CARD_H - HEAD_H) / 2.0, body, ha="center", va="center",
            color="#4a4f56", fontsize=6.8, zorder=z + 4)


def banner(ax, x0, text, color):
    ax.add_patch(FancyBboxPatch((x0, BANNER_Y - 0.19), CARD_W, 0.38,
                                boxstyle="round,pad=0,rounding_size=0.09",
                                facecolor=color, edgecolor="none",
                                mutation_aspect=1.0, zorder=4))
    ax.text(x0 + CARD_W / 2.0, BANNER_Y, text, ha="center", va="center",
            color="white", fontsize=9.4, fontweight="bold", zorder=5)


def main():
    set_pub_style()
    fig, ax = inch_canvas(FW, FH)

    banner(ax, LX0, "CHALLENGES", CHAL_HEAD)
    banner(ax, RX0, "MITIGATIONS", MITI_HEAD)

    # ---- arcs first, so no edge is ever drawn over a card -------------------
    x_out, x_in = LX0 + CARD_W, RX0
    for ci, mi, s in sorted(LINKS, key=lambda t: t[2]):
        y0, y1 = ROW_Y[ci], ROW_Y[mi]
        rad = 0.0 if ci == mi else (-0.16 if y1 > y0 else 0.16)
        ax.add_patch(FancyArrowPatch(
            (x_out + 0.06, y0), (x_in - 0.10, y1), arrowstyle="-|>",
            mutation_scale=7.0 + 6.0 * s, linewidth=0.7 + 2.5 * s,
            color=LINK, alpha=0.28 + 0.52 * s, shrinkA=0, shrinkB=0,
            connectionstyle=f"arc3,rad={rad}", zorder=2))

    for i, (head, body) in enumerate(CHALLENGES):
        two_tone_card(ax, LX0, ROW_Y[i], head, body, CHAL_HEAD, CHAL_BODY)
    for i, (head, body) in enumerate(MITIGATIONS):
        two_tone_card(ax, RX0, ROW_Y[i], head, body, MITI_HEAD, MITI_BODY)

    # ---- what the arc widths mean, on the footer line where there is room --
    key_y = 0.36
    ax.text(7.02, key_y, "link strength", ha="right", va="center",
            fontsize=6.8, fontweight="bold", color=COLORS["sub"])
    for i, (s, lab) in enumerate([(0.30, "weak"), (0.60, "partial"),
                                  (0.90, "strong")]):
        x = 7.16 + 0.94 * i
        ax.plot([x, x + 0.36], [key_y, key_y], color=LINK, lw=0.7 + 2.5 * s,
                alpha=0.28 + 0.52 * s, solid_capstyle="round", zorder=3)
        ax.text(x + 0.44, key_y, lab, ha="left", va="center", fontsize=6.6,
                color=COLORS["sub"])

    ax.text(0.40, 6.18, "What is still hard, and what is being done about it",
            ha="left", va="center", fontsize=11.0, fontweight="bold",
            color=COLORS["ink"])
    ax.text(0.40, 5.90, "Every challenge has a leading mitigation on its own "
            "row plus a partial cross-link, so the fixes do not partition "
            "cleanly", ha="left", va="center", fontsize=8.0,
            color=COLORS["sub"])
    ax.text(0.40, key_y, "Arc width and opacity encode how much of a "
            "challenge the mitigation removes; strengths are an editorial "
            "judgement", ha="left", va="center", fontsize=6.6,
            color=COLORS["muted"])

    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    savefig(fig, "1086_wm_concept_challenges.png")


if __name__ == "__main__":
    main()
