"""Taxonomy of world models: one root, three paradigms, ten instantiations.

A survey taxonomy is only useful if the reader can see the *shape* of the
field at a glance, so the drawing does three things.  Every edge is an
orthogonal polyline -- a diagonal edge in a tree invites the reader to compare
slopes, which mean nothing here.  Each paradigm owns a hue and each of its
children is a slightly lighter shade of it, so membership survives being read
at thumbnail size, when the text has gone.  And every node is the same
two-tone card (saturated header naming the node, white body carrying one line
of what it actually does), so the three tiers differ only in size.

Drawn on an inch canvas: card sizes, gutters and elbow clearances are all in
inches on paper.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from pubstyle import COLORS, lighten, savefig, set_pub_style
from figkit import arrow, inch_canvas

FW, FH = 10.4, 6.4
ROUND = 0.062
BODY_FILL = "#eef3f9"

ROOT = ("World Model", "a learned internal simulator of an environment")

# Three rungs of the house blue ladder rather than three hues: the columns
# already separate the paradigms, so hue would only be saying it twice.
# rung, paradigm name, one-line gloss, [(leaf name, leaf gloss), ...]
BRANCHES = [
    ("#2f4f7f", "Generative", "synthesise future observations", [
        ("Video Diffusion", "denoise future frames"),
        ("Autoregressive Tokens", "next-token video rollout"),
        ("Masked Prediction", "infill held-out patches"),
        ("Flow Matching", "learn a transport field"),
    ]),
    ("#4a72a4", "Predictive", "forecast a compact latent state", [
        ("RSSM / Dreamer", "stochastic recurrent state"),
        ("Latent JEPA", "predict embeddings, not pixels"),
        ("Koopman Encoder", "linearise the dynamics"),
    ]),
    ("#6d97c6", "Simulation-based", "wrap an explicit physical model", [
        ("Differentiable Physics", "gradients through contact"),
        ("Neural Game Engine", "learned rendering loop"),
        ("Digital Twin", "calibrated to a real rig"),
    ]),
]

COL_X0 = [0.62, 3.78, 6.94]      # left edge of each paradigm column
BRANCH_W, BRANCH_H = 2.55, 0.62
LEAF_W, LEAF_H = 2.30, 0.56
ROOT_W, ROOT_H = 3.50, 0.66
TRUNK_DX = 0.30                  # trunk offset from the column's left edge
LEAF_DX = 0.58                   # leaf left edge, from the column's left edge

ROOT_Y = 5.08
BUS_Y = 4.46
BRANCH_Y = 3.88
LEAF_Y0 = 3.00
LEAF_PITCH = 0.70


def ink_on(color):
    """Legible text colour for a header strip of the given blue."""
    r, g, b = mpl.colors.to_rgb(color)
    return "#1f2c3a" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.56 else "white"


def two_tone_card(ax, cx, cy, w, h, head, body, color, head_h,
                  head_fs, body_fs, z=4):
    """Deep header strip over a pale-blue body; hairline outline, soft drop."""
    x0, y0 = cx - w / 2.0, cy - h / 2.0
    box = f"round,pad=0,rounding_size={ROUND}"
    ax.add_patch(FancyBboxPatch((x0 + 0.030, y0 - 0.036), w, h, boxstyle=box,
                                facecolor="#d2dceb", edgecolor="none",
                                mutation_aspect=1.0, zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=box,
                                facecolor=BODY_FILL, edgecolor="none",
                                mutation_aspect=1.0, zorder=z))
    hy = y0 + h - head_h
    ax.add_patch(FancyBboxPatch((x0, hy), w, head_h, boxstyle=box,
                                facecolor=color, edgecolor="none",
                                mutation_aspect=1.0, zorder=z + 1))
    ax.add_patch(Rectangle((x0, hy), w, ROUND, facecolor=color,
                           edgecolor="none", zorder=z + 2))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=box, facecolor="none",
                                edgecolor=color, linewidth=0.85,
                                mutation_aspect=1.0, zorder=z + 3))
    ax.text(cx, hy + head_h / 2.0, head, ha="center", va="center",
            color=ink_on(color), fontsize=head_fs, fontweight="bold",
            zorder=z + 4)
    ax.text(cx, y0 + (h - head_h) / 2.0, body, ha="center", va="center",
            color="#3a4756", fontsize=body_fs, zorder=z + 4)


def elbow(ax, trunk_x, x_to, y, color):
    """Horizontal leg off the shared trunk, with the head at the leaf edge."""
    ax.add_patch(Circle((trunk_x, y), 0.030, facecolor=color,
                        edgecolor="none", zorder=3))
    arrow(ax, (trunk_x, y), (x_to, y), color=color, lw=1.05, ms=7.5,
          zorder=3, gap_b=0.02)


def main():
    set_pub_style()
    fig, ax = inch_canvas(FW, FH)

    bxs = [x0 + BRANCH_W / 2.0 + 0.10 for x0 in COL_X0]
    root_x = bxs[1]              # the root sits exactly over the middle drop
    edge = "#5b6a7d"

    # ---- tier rails: they say what each row *is* without a legend -----------
    tiers = [(ROOT_Y, "ROOT"), (BRANCH_Y, "PARADIGM"),
             (LEAF_Y0 - 1.5 * LEAF_PITCH, "INSTANTIATION")]
    for y, lab in tiers:
        ax.text(0.30, y, lab, ha="left", va="center", fontsize=6.6,
                fontweight="bold", color="#9aa4b0", rotation=90,
                rotation_mode="anchor")

    # ---- root -> paradigms: one shared bus, three orthogonal drops ---------
    ax.plot([root_x, root_x], [ROOT_Y - ROOT_H / 2.0, BUS_Y], color=edge,
            lw=1.15, zorder=2, solid_capstyle="butt")
    ax.plot([min(bxs), max(bxs)], [BUS_Y, BUS_Y], color=edge, lw=1.15,
            zorder=2, solid_capstyle="round")
    for bx, (hue, *_rest) in zip(bxs, BRANCHES):
        ax.add_patch(Circle((bx, BUS_Y), 0.034, facecolor=edge,
                            edgecolor="none", zorder=3))
        arrow(ax, (bx, BUS_Y), (bx, BRANCH_Y + BRANCH_H / 2.0), color=hue,
              lw=1.15, ms=8.0, zorder=3, gap_b=0.02)

    two_tone_card(ax, root_x, ROOT_Y, ROOT_W, ROOT_H, ROOT[0], ROOT[1],
                  "#22364f", head_h=0.32, head_fs=9.8, body_fs=7.2)

    # ---- paradigms and their instantiations --------------------------------
    for x0, (hue, name, gloss, leaves) in zip(COL_X0, BRANCHES):
        cx = x0 + BRANCH_W / 2.0 + 0.10
        two_tone_card(ax, cx, BRANCH_Y, BRANCH_W, BRANCH_H, name, gloss, hue,
                      head_h=0.30, head_fs=9.0, body_fs=6.9)

        trunk_x = x0 + TRUNK_DX
        leaf_cx = x0 + LEAF_DX + LEAF_W / 2.0
        ys = [LEAF_Y0 - i * LEAF_PITCH for i in range(len(leaves))]
        ax.plot([trunk_x, trunk_x], [BRANCH_Y - BRANCH_H / 2.0, ys[-1]],
                color=hue, lw=1.05, zorder=2, solid_capstyle="butt")
        for i, ((lname, lgloss), y) in enumerate(zip(leaves, ys)):
            tint = lighten(hue, 0.08 + 0.06 * i)
            elbow(ax, trunk_x, x0 + LEAF_DX, y, hue)
            two_tone_card(ax, leaf_cx, y, LEAF_W, LEAF_H, lname, lgloss, tint,
                          head_h=0.25, head_fs=7.6, body_fs=6.4)

    # ---- titles -------------------------------------------------------------
    ax.text(0.30, 6.06, "A taxonomy of world models",
            ha="left", va="center", fontsize=11.0, fontweight="bold",
            color=COLORS["ink"])
    ax.text(0.30, 5.76, "Three paradigms separated by what the model is asked "
            "to predict; each column takes one rung of the blue ladder",
            ha="left", va="center", fontsize=8.0, color=COLORS["sub"])
    # A note parked in the gap the short branches leave, rather than a
    # footnote stranded under a metre of white.
    note_x, note_y = COL_X0[2] + LEAF_DX, 0.86
    ax.plot([note_x, note_x], [note_y - 0.30, note_y + 0.30],
            color="#c3cad3", lw=1.6, zorder=2, solid_capstyle="round")
    ax.text(note_x + 0.20, note_y,
            "Boundaries are porous: a diffusion rollout\n"
            "conditioned on a latent state sits in two\n"
            "columns at once. Read the tree as emphasis.",
            ha="left", va="center", fontsize=6.9, color=COLORS["sub"],
            linespacing=1.62, zorder=3)
    ax.text(0.62, 0.24, "Every edge is orthogonal by construction: "
            "no slope in this diagram carries meaning",
            ha="left", va="center", fontsize=6.6, color=COLORS["muted"])

    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    savefig(fig, "1082_wm_concept_taxonomy.png")


if __name__ == "__main__":
    main()
