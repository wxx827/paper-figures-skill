"""Figure 1171 - Caption grounding: noun phrases underlined in region colour.

Left is one procedural living-room frame from the shared scene library; four
regions are boxed on the *anchors the generator actually placed*, so no box
ever encloses empty wall.  Right is the generated caption, laid out token by
token with the renderer so a noun phrase can be underlined in the colour of
its region, and an elbow leader runs from the start of each underline, through
the gutter, back to the box it names.

Single source of truth: ``REGIONS`` drives the box, the tag disc, the
underline colour, the leader and the score chip at once.  Change one row and
all four move together; the mean grounding score in the footer is the mean of
the four scores printed on the frame, so a reader adding them up gets the
number that is written down.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

from pubstyle import set_pub_style, savefig
from _wm_scenes import build

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
CARD = "#f5f5f5"
HAIR = "#d5d5d5"

# Four cold tiers: deep blue / teal / violet / steel.  Warm hues are the
# batch's scarcest resource (section 1.8), and region boxes are exactly where
# a palette slides into orange if nobody is watching.
COLS = ["#2f4a75", "#2a7f7a", "#7b3294", "#3d6b8e"]

# (anchor name, colour, score, leader port, tag corner)
#   port "right" -> the leader meets the right edge at the box centre height
#   port "top"   -> it climbs above every other box and drops onto the top edge
REGIONS = [
    ("picture", COLS[0], 0.93, "top", "left"),
    ("window", COLS[1], 0.88, "right", "left"),
    ("sofa", COLS[2], 0.91, "right", "left"),
    ("lamp", COLS[3], 0.76, "right", "right"),
]

# Caption tokens.  ``gid`` indexes REGIONS; every grounded phrase is
# line-initial so its leader can leave to the left through empty space
# instead of striking out the words in front of it.
CAPTION = [
    [("A quiet living room, seen from a low front-facing camera.", None)],
    [("The framed picture", 0), ("hangs high on the left wall.", None)],
    [("The tall window", 1), ("on the right admits flat, cool daylight", None)],
    [("that grazes the seat cushions.", None)],
    [("The fabric sofa", 2), ("faces the lens, centred in the room;", None)],
    [("a closed book rests on its left armrest.", None)],
    [("The floor lamp", 3), ("stands beyond it, switched on.", None)],
]

W, H = 11.0, 4.8
PH, PW = 195, 214                      # scene raster
PX0, PY0, PWD, PHT = 0.34, 0.50, 4.12, 3.76
GUT = [4.53, 4.70, 4.87, 5.04]         # one vertical leader lane per region
TX0, TY0 = 5.30, 3.62                  # caption baseline origin
LEAD = 0.360
FS_CAP = 8.7


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return fig, ax


def card(ax, x0, y0, w, h, fc=CARD, ec="none", lw=0.8, r=0.09, zorder=1):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)
    return p


def photo(ax, rgb, x0, y0, w, h, ec="#9a9a9a", lw=0.8, r=0.05, zorder=4):
    clip = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor="none", edgecolor=ec, linewidth=lw,
                          zorder=zorder + 6)
    ax.add_patch(clip)
    im = ax.imshow(np.clip(rgb, 0, 1), extent=[x0, x0 + w, y0, y0 + h],
                   zorder=zorder, interpolation="bilinear")
    im.set_clip_path(clip)
    return clip


def halo_path(ax, xs, ys, color, lw=0.85, zorder=9):
    """A leader that stays legible over both the photo and the white gutter."""
    ax.plot(xs, ys, color="white", lw=lw + 1.5, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder, alpha=0.85)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder + 1)


def tag_disc(ax, x, y, n, color, r=0.088, zorder=12):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white",
                        linewidth=0.9, zorder=zorder))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=6.4,
            fontweight="bold", color="white", zorder=zorder + 1)


def token_width(ax, fig, s, fs, weight="normal"):
    t = ax.text(0, 0, s, fontsize=fs, fontweight=weight, alpha=0.0)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    p = ax.transData.inverted().transform([[0, 0], [bb.width, 0]])
    return float(p[1][0] - p[0][0])


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    fig.canvas.draw()                        # renderer for the text metrics

    # ---- scene ------------------------------------------------------------
    # The indoor preset is mildly tungsten; pulled cool and low-chroma so the
    # wooden floor stays out of the warm-area budget.
    sc = build("indoor", seed=1576, shape=(PH, PW), horizon=0.545,
               cam=dict(wb=(0.952, 1.000, 1.070), sat=0.80, exposure=0.97,
                        contrast=1.20, vignette=0.30, dof=2.4, sharpen=1.15,
                        kappa=0.031))
    anc = {a["name"]: a for a in sc["anchors"]}
    clip = photo(ax, sc["rgb"], PX0, PY0, PWD, PHT)

    def to_x(px):
        return PX0 + px / PW * PWD

    def to_y(py):
        return PY0 + (1.0 - py / PH) * PHT

    geo = {}
    for k, (name, col, score, port, corner) in enumerate(REGIONS):
        a = anc[name]
        bw = 2.0 * a["rx"] * 0.86 / PW * PWD
        bh = 2.0 * a["ry"] * 0.86 / PH * PHT
        cx, cy = to_x(a["cx"]), to_y(a["cy"])
        x0, y0 = cx - bw / 2, cy - bh / 2
        geo[k] = (x0, y0, bw, bh, cx, cy)
        rect = Rectangle((x0, y0), bw, bh, facecolor="none", edgecolor=col,
                         linewidth=1.5, zorder=11)
        ax.add_patch(rect)
        rect.set_clip_path(clip)
        tx = x0 + (0.0 if corner == "left" else bw)
        tag_disc(ax, tx, y0 + bh, k + 1, col)
        ax.text(cx, y0 - 0.085, f"{score:.2f}", ha="center", va="top",
                fontsize=6.4, color=col, zorder=12,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                          alpha=0.92))

    ax.text(PX0 + PWD / 2, PY0 - 0.30,
            "input frame with the four predicted regions",
            ha="center", va="center", fontsize=6.9, color=MUTED)

    # ---- caption card -----------------------------------------------------
    card(ax, 5.10, 0.50, 5.56, 3.76, fc=CARD, r=0.10)
    card(ax, 5.28, 3.82, 1.86, 0.28, fc="white", ec=HAIR, lw=0.7, r=0.06,
         zorder=2)
    ax.text(6.21, 3.96, "generated caption", ha="center", va="center",
            fontsize=6.9, color=SUB, zorder=3)
    ax.text(10.48, 3.96, "GroundCap-VL", ha="right", va="center",
            fontsize=8.0, fontweight="bold", style="italic", color=INK,
            zorder=3)

    space = token_width(ax, fig, "n n", FS_CAP) - \
        2 * token_width(ax, fig, "n", FS_CAP)

    spans = {}
    for row, line in enumerate(CAPTION):
        y = TY0 - row * LEAD
        x = TX0
        for text, gid in line:
            wgt = "normal" if gid is None else "bold"
            ax.text(x, y, text, ha="left", va="baseline", fontsize=FS_CAP,
                    color=INK if gid is None else COLS[gid], zorder=4,
                    fontweight=wgt)
            wdt = token_width(ax, fig, text, FS_CAP, wgt)
            if gid is not None:
                uy = y - 0.083
                ax.plot([x, x + wdt], [uy, uy], color=COLS[gid], lw=1.5,
                        solid_capstyle="butt", zorder=5)
                spans[gid] = (x, uy)
            x += wdt + space

    # ---- leaders: underline -> gutter lane -> box -------------------------
    for k, (name, col, score, port, corner) in enumerate(REGIONS):
        x0, y0, bw, bh, cx, cy = geo[k]
        sx, sy = spans[k]
        gx = GUT[k]
        if port == "right":
            pts = [(sx, sy), (gx, sy), (gx, cy), (x0 + bw, cy)]
        else:
            top = PY0 + PHT - 0.13
            pts = [(sx, sy), (gx, sy), (gx, top), (cx, top), (cx, y0 + bh)]
        halo_path(ax, [p[0] for p in pts], [p[1] for p in pts], col)
        ax.add_patch(Circle(pts[-1], 0.036, facecolor=col, edgecolor="white",
                            linewidth=0.6, zorder=12))

    # ---- region legend ----------------------------------------------------
    lx, ly = TX0, 1.02
    for k, (name, col, score, port, corner) in enumerate(REGIONS):
        tag_disc(ax, lx + 0.08, ly, k + 1, col, r=0.080, zorder=5)
        ax.text(lx + 0.24, ly, name, ha="left", va="center", fontsize=7.2,
                color=SUB, zorder=4)
        lx += 0.24 + token_width(ax, fig, name, 7.2) + 0.52

    # ---- footer: the mean is the mean of the four printed chips -----------
    scores = [r[2] for r in REGIONS]
    mean = sum(scores) / len(scores)
    ax.text(5.30, 0.72,
            f"{len(REGIONS)} noun phrases  ·  {len(REGIONS)} predicted "
            f"regions  ·  mean grounding score "
            f"{' + '.join(f'{s:.2f}' for s in scores)} = {sum(scores):.2f}, "
            f"/ {len(REGIONS)} = {mean:.2f}",
            ha="left", va="center", fontsize=6.6, color=MUTED, zorder=4)


    savefig(fig, "1171_wm_vlm_caption_grounding.png")


if __name__ == "__main__":
    main()
