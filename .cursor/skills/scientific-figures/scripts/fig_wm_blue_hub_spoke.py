"""Wm blue hub spoke: hub-and-spoke map of embodied world-model challenges.

A flat "IEEE Access" concept map: one deep-blue hub carrying the topic, four
mid-blue satellites carrying the failure modes, hairline grey connectors.
The whole figure is drawn on an inch canvas, so corner radii, gaps and text
padding are all in the same physical unit and stay put when the canvas is
resized.  Solid fills, white bold type, no gradient and no drop shadow --
that flatness is the point: it survives being shrunk to a contact-sheet cell.

There is no data here; the layout constants below *are* the figure.
"""
from matplotlib.patches import FancyBboxPatch
from pubstyle import set_pub_style, savefig
from figkit import inch_canvas, anchors

# ---- WM blue batch palette (BRIEF_WM_BLUE.md section 2) --------------------
BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"          # the one warm colour, < 5 % of the area
LINK = "#b9c0c7"            # hairline connectors

W, H = 7.6, 5.6
CX, CY = 3.80, 2.50         # hub centre
HUB_W, HUB_H = 2.60, 1.42
SPOKE_H = 1.00
R = 0.13                    # corner radius, shared by every block


def flat_block(ax, x, y, w, h, title, sub=None, color=BLUE[3],
               title_size=9.5, zorder=4):
    """Solid rounded block with white bold type -- the batch's 'blue B' look."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={R}", facecolor=color,
        edgecolor="none", zorder=zorder))
    ty = y + (0.17 if sub else 0.0)
    ax.text(x, ty, title, ha="center", va="center", color="white",
            fontsize=title_size, fontweight="bold", linespacing=1.28,
            zorder=zorder + 2)
    if sub:
        ax.text(x, y - 0.30, sub, ha="center", va="center", color="white",
                fontsize=6.7, alpha=0.90, zorder=zorder + 2)
    return anchors(x, y, w, h)


def link(ax, p0, p1, color=LINK, lw=1.15, zorder=2):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw,
            solid_capstyle="butt", zorder=zorder)


def main():
    set_pub_style()
    fig, ax = inch_canvas(W, H)

    ax.text(0.18, 5.38, "Major challenges of embodied world models",
            fontsize=10.5, fontweight="bold", color=INK, ha="left",
            va="center")
    ax.text(0.18, 5.10, "Recurring failure modes reported across "
            "manipulation, driving and navigation benchmarks",
            fontsize=7.2, color="#555555", ha="left", va="center")

    hub = flat_block(
        ax, CX, CY, HUB_W, HUB_H,
        "Major Challenges of\nEmbodied World Models",
        sub="each spoke compounds the others", color=BLUE[1],
        title_size=11.0)

    top = flat_block(ax, CX, 4.28, 2.30, SPOKE_H,
                     "Long-horizon\nConsistency",
                     sub="drift over 100+ step rollouts")
    bot = flat_block(ax, CX, 0.72, 2.30, SPOKE_H,
                     "Action\nGrounding",
                     sub="latent plans vs. joint torques")
    left = flat_block(ax, 1.05, CY, 1.90, SPOKE_H,
                      "Physical\nPlausibility",
                      sub="contacts, mass, friction")
    right = flat_block(ax, 6.55, CY, 1.90, SPOKE_H,
                       "Sim-to-Real\nGap",
                       sub="transfer without fine-tuning")

    link(ax, hub["t"], top["b"])
    link(ax, hub["b"], bot["t"])
    link(ax, hub["l"], left["r"])
    # The one warm accent: the spoke the survey calls least solved.
    link(ax, hub["r"], right["l"], color=ACCENT, lw=1.7)
    ax.text(right["x"], right["t"][1] + 0.10, "least solved of the four",
            fontsize=6.6, color=ACCENT, ha="center", va="bottom",
            style="italic")

    savefig(fig, "1099_wm_blue_hub_spoke.png")


if __name__ == "__main__":
    main()
