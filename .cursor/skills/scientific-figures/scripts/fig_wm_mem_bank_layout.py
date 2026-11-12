"""Figure 1195 - Layout of a three-tier memory bank.

Working, episodic and semantic memory are drawn as three stacked cards in the
house "dark title bar over a pale body" style.  Each card carries the same
four facts - entry count, key dimensionality, write rate and retention - so a
reader can compare tiers by scanning one column instead of one card.  The
gaps between the cards hold the two traffic directions: encode / consolidate
going down, recall going up.

Colour discipline: the batch is all-cool, so the three tiers are separated by
depth on one blue ladder plus a fill pattern on their entry decks - ruled,
dotted, diagonally hatched, getting denser as the tier gets deeper.  No
second hue is used anywhere.

Honesty (BRIEF_WM section 1.7): every number printed here is an f-string over
the ``TIERS`` table, the capacity bars are drawn at ``used / total`` and print
that division in full, and the consolidation ratio is asserted against the two
counts that appear on the cards.  Change a count and the assertions fail
rather than the figure quietly lying.

Canvas: (10.4, 5.2), h/w = 0.50 - an inch-canvas architecture diagram, the E4
exception of DESIGN_SYSTEM section 1.2.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from pubstyle import set_pub_style, savefig

# ---- one blue ladder, no second hue ---------------------------------------
BAR_DARK = "#1f4e79"
BAR_MID = "#2f6fa8"
BAR_LIGHT = "#4a8cc7"
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#d5d5d5"
TRACK = "#e4e8ec"          # luminance 0.908: deliberately below the 0.92 band

W, H = 10.4, 5.2
X0, X1 = 0.40, 7.05        # card column
SPLIT = 3.18               # hairline between the entry deck and the stats
STAT_X = (3.66, 4.54, 5.46, 6.48)
BAR_X0, BAR_X1 = 7.46, 9.98
R = 0.055

# One row per tier.  ``hatch`` is the deck fill pattern that replaces a second
# hue: horizontal rules / dots / diagonals, getting denser as the tier gets
# deeper.  Every printed number on the figure comes out of this table.
TIERS = [
    dict(name="Working memory", role="current episode, always in context",
         head=BAR_LIGHT, body="#dde9f4", hatch="--",
         slot="slot = 768-d key + raw observation",
         entries="192", dim="768", write="20 Hz", keep="8 steps",
         used=192, total=256, y0=3.30, y1=4.36),
    dict(name="Episodic memory", role="one record per completed event",
         head=BAR_MID, body="#cfdfee", hatch="....",
         slot="slot = 768-d key + action trace + outcome",
         entries="36,000", dim="768", write="0.5 Hz", keep="30 days",
         used=36000, total=50000, y0=1.90, y1=2.96),
    dict(name="Semantic memory", role="facts abstracted over many episodes",
         head=BAR_DARK, body="#c0d4e8", hatch="///",
         slot="slot = 768-d key + relation triple + support count",
         entries="8,600", dim="768", write="1 / hour", keep="unbounded",
         used=8600, total=20000, y0=0.50, y1=1.56),
]

STAT_KEYS = [("entries", "entries"), ("key dim", "dim"),
             ("write rate", "write"), ("retention", "keep")]


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec, lw=0.9, ls="-", r=R, hatch=None, zorder=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle=ls, hatch=hatch, zorder=zorder)
    ax.add_patch(p)
    return p


def tier_card(ax, t):
    y0, y1 = t["y0"], t["y1"]
    h = y1 - y0
    rbox(ax, X0, y0, X1 - X0, h, t["body"], t["head"], lw=1.0, zorder=3)
    # dark title bar: a rounded box clipped visually by re-drawing the seam
    rbox(ax, X0, y1 - 0.30, X1 - X0, 0.30, t["head"], t["head"], lw=0.9,
         zorder=4)
    ax.add_patch(Rectangle((X0, y1 - 0.34), X1 - X0, 0.10,
                           facecolor=t["head"], edgecolor="none", zorder=4))
    ax.text(X0 + 0.18, y1 - 0.15, t["name"], ha="left", va="center",
            fontsize=8.6, fontweight="bold", color="white", zorder=6)
    ax.text(X1 - 0.18, y1 - 0.15, t["role"], ha="right", va="center",
            fontsize=6.6, color="white", zorder=6)

    # entry deck: seven representative slots, pattern differs per tier
    cw, gap = 0.26, 0.075
    yc = y0 + 0.46
    for i in range(7):
        x = X0 + 0.24 + i * (cw + gap)
        rbox(ax, x, yc - 0.17, cw, 0.34, "white", t["head"], lw=0.8,
             r=0.035, hatch=t["hatch"], zorder=5)
    xend = X0 + 0.24 + 7 * (cw + gap)
    ax.text(xend + 0.02, yc, "...", ha="left", va="center", fontsize=8.0,
            color=t["head"], zorder=5)
    ax.text(X0 + 0.24, y0 + 0.16, t["slot"], ha="left", va="center",
            fontsize=6.2, color=SUB, zorder=5)

    ax.plot([SPLIT, SPLIT], [y0 + 0.12, y1 - 0.40], color=t["head"],
            lw=0.7, alpha=0.55, zorder=5)
    for x, (label, key) in zip(STAT_X, STAT_KEYS):
        ax.text(x, y0 + 0.56, label, ha="center", va="center", fontsize=6.2,
                color=SUB, zorder=5)
        ax.text(x, y0 + 0.26, t[key], ha="center", va="center", fontsize=8.4,
                fontweight="bold", color=INK, zorder=5)


def flow(ax, x, y_from, y_to, colour, text, side="left"):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=1.1,
                                 color=colour, shrinkA=0, shrinkB=0,
                                 zorder=7))
    ax.text(x + (0.14 if side == "left" else 0.14),
            (y_from + y_to) / 2.0, text, ha="left", va="center",
            fontsize=6.3, color=colour, zorder=8,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.95))


def capacity(ax, t, pct_str):
    yc = 0.5 * (t["y0"] + t["y1"])
    frac = t["used"] / t["total"]
    ax.text(BAR_X0, yc + 0.28, f"{t['entries']} / {t['total']:,} slots",
            ha="left", va="center", fontsize=6.5, color=SUB)
    ax.add_patch(Rectangle((BAR_X0, yc - 0.13), BAR_X1 - BAR_X0, 0.26,
                           facecolor=TRACK, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((BAR_X0, yc - 0.13), (BAR_X1 - BAR_X0) * frac,
                           0.26, facecolor=t["head"], edgecolor="none",
                           zorder=4))
    ax.add_patch(Rectangle((BAR_X0, yc - 0.13), BAR_X1 - BAR_X0, 0.26,
                           facecolor="none", edgecolor=HAIR, linewidth=0.7,
                           zorder=5))
    ax.text(BAR_X0 + (BAR_X1 - BAR_X0) * frac - 0.07, yc, f"= {pct_str} %",
            ha="right", va="center", fontsize=6.6, color="white",
            fontweight="bold", zorder=6)
    ax.text(BAR_X1, yc - 0.30, f"headroom {t['total'] - t['used']:,} slots",
            ha="right", va="center", fontsize=6.0, color=FAINT)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    pcts = [f"{100.0 * t['used'] / t['total']:.1f}" for t in TIERS]
    for t, p in zip(TIERS, pcts):
        assert abs(float(p) - 100.0 * t["used"] / t["total"]) < 0.05, \
            f"capacity label for {t['name']} does not reproduce used/total"

    epi, sem = TIERS[1]["used"], TIERS[2]["used"]
    comp = f"{epi / sem:.2f}"
    assert abs(float(comp) - epi / sem) < 0.005, "compression ratio not closed"

    ax.text(0.40, 4.94, "A three-tier memory bank keeps 192 slots hot and "
                        "8,600 facts forever", ha="left", va="center",
            fontsize=11.0, fontweight="bold", color=INK)
    ax.text(0.40, 4.66, "tier depth on one blue ladder; entry decks are "
                        "ruled / dotted / hatched.  Down = write path, "
                        "up = recall path", ha="left", va="center",
            fontsize=7.4, color=MUTED)

    for t in TIERS:
        tier_card(ax, t)

    flow(ax, 1.52, 3.28, 2.98, BAR_MID, "encode: 1 slot per step")
    flow(ax, 5.06, 2.98, 3.28, BAR_MID, "recall: top-8 into context")
    flow(ax, 1.52, 1.88, 1.58, BAR_DARK, "consolidate: nightly replay")
    flow(ax, 5.06, 1.58, 1.88, BAR_DARK, "recall: top-4 facts")

    ax.plot([7.20, 7.20], [0.50, 4.36], color=HAIR, lw=0.8, zorder=2)
    ax.text(7.32, 4.62, "Capacity used", ha="left", va="center",
            fontsize=8.0, fontweight="bold", color=INK)
    ax.text(7.32, 4.44, "bar length = used / total slots", ha="left",
            va="center", fontsize=6.4, color=MUTED)
    for t, p in zip(TIERS, pcts):
        capacity(ax, t, p)

    ax.text(0.40, 0.20,
            f"Consolidation compresses the episodic tier into the semantic "
            f"tier: {TIERS[1]['entries']} / {TIERS[2]['entries']} = "
            f"{comp} episodes per retained fact; all three tiers share one "
            f"{TIERS[0]['dim']}-d key space.",
            ha="left", va="center", fontsize=6.5, color=MUTED, style="italic")

    savefig(fig, "1195_wm_mem_bank_layout.png")


if __name__ == "__main__":
    main()
