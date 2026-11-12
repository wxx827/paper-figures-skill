"""Embodied-assistant teaser: user thought bubble -> VLM answer card.

Left half is deliberately hand-drawn: a flat silhouette of a person, a wooden
table carrying a jar and a folded cloth, and a scalloped thought bubble linked
to the head by three shrinking circles.  Right half is the assistant's reply,
a pale-gold card with a dashed rounded border holding the answer text and a
scene thumbnail with the predicted hand trajectory drawn on top in dashed red.

Every glyph is built from matplotlib primitives; the thumbnail is procedural
noise (multi-octave value noise + Gaussian blur + muted wood ramp + vignette).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Polygon, Ellipse)
from scipy.ndimage import gaussian_filter, zoom
from pubstyle import set_pub_style, savefig, lighten

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
DARK = "#3a3a3a"
GOLD = "#c9a227"
RED = "#b23a48"
BLUE = "#3b5b92"
WOOD = "#a9764c"
WOOD_DK = "#6d4c33"

CMAP_WOOD = LinearSegmentedColormap.from_list(
    "wm_wood", ["#3f3128", "#6d5238", "#a37b50", "#d3b083", "#f6ecdb"])

R_BOX = 0.07
CARD_FC = lighten(GOLD, 0.90)


# ---- canvas ----------------------------------------------------------------
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return fig, ax


# ---- procedural imagery ----------------------------------------------------
def value_noise(rng, ny, nx, base=3, octaves=4, decay=0.55):
    out = np.zeros((ny, nx))
    amp, tot = 1.0, 0.0
    for k in range(octaves):
        gh = max(2, base * 2 ** k)
        gw = max(2, int(round(gh * nx / ny)))
        small = rng.random((gh, gw))
        big = zoom(small, (ny / gh, nx / gw), order=3)[:ny, :nx]
        if big.shape != (ny, nx):
            big = np.pad(big, ((0, ny - big.shape[0]), (0, nx - big.shape[1])),
                         mode="edge")
        out += amp * big
        tot += amp
        amp *= decay
    out /= tot
    return (out - out.min()) / (np.ptp(out) + 1e-9)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def seg_dist(xx, yy, p0, p1):
    """Distance from every pixel to the segment p0-p1 (broadcasting grids)."""
    px, py = p1[0] - p0[0], p1[1] - p0[1]
    t = np.clip(((xx - p0[0]) * px + (yy - p0[1]) * py)
                / (px * px + py * py), 0, 1)
    return np.hypot(xx - (p0[0] + t * px), yy - (p0[1] + t * py))


def tabletop(rng, ny=180, nx=212):
    """Table scene built from geometry first, texture second.

    A slanted table edge splits the frame, a rounded-trapezoid glass jar with
    a darker lid stands left of centre and a folded cloth sits to its right.
    Wood grain, low-amplitude noise and a shallow depth of field are layered
    on afterwards, so the structure survives the blur instead of dissolving
    into it.  Coordinates are top-origin, matching ``imshow``.
    """
    yy = np.linspace(0, 1, ny)[:, None]
    xx = np.linspace(0, 1, nx)[None, :]

    # -- table edge: below the slanted line is the table top ----------------
    edge = 0.235 - 0.085 * xx
    on_table = _sig(55 * (yy - edge))
    field = (1 - on_table) * (0.30 + 0.10 * yy) + on_table * (0.60 + 0.14 * yy)

    # -- wood grain running parallel to that edge ---------------------------
    u = yy + 0.085 * xx
    grain = (np.sin(2 * np.pi * 9.0 * u + 2.2 * value_noise(rng, ny, nx,
                                                            base=2))
             + 0.45 * np.sin(2 * np.pi * 21.0 * u))
    field += on_table * 0.042 * grain
    field -= 0.11 * np.exp(-((yy - edge) / 0.013) ** 2)      # edge shadow

    # -- glass jar: rounded trapezoid + lid + highlight ---------------------
    jx, jtop, jbase = 0.310, 0.255, 0.660
    span = np.clip((yy - jtop) / (jbase - jtop), 0, 1)
    hw = 0.056 + 0.018 * span
    inside = _sig(90 * (yy - jtop)) * _sig(110 * (jbase - yy))
    body = _sig(110 * (hw - np.abs(xx - jx))) * inside
    field -= 0.22 * np.exp(-(((xx - jx) / 0.110) ** 2
                             + ((yy - (jbase + 0.014)) / 0.017) ** 2))
    field += 0.17 * body
    field += 0.24 * body * np.exp(-((xx - (jx - 0.022)) / 0.010) ** 2)
    field -= 0.12 * body * np.exp(-((xx - (jx + 0.030)) / 0.013) ** 2)
    field -= 0.20 * inside * np.exp(
        -((np.abs(xx - jx) - hw) / 0.007) ** 2)               # glass contour
    field -= 0.18 * body * np.exp(-((yy - jbase) / 0.014) ** 2)
    lid = _sig(55 * (1 - np.sqrt(((xx - jx) / 0.068) ** 2
                                 + ((yy - jtop) / 0.030) ** 2)))
    field -= 0.30 * lid
    field += 0.20 * lid * np.exp(-((yy - (jtop - 0.016)) / 0.007) ** 2)

    # -- folded cloth: irregular closed blob with a few creases -------------
    cx, cy, rx, ry = 0.700, 0.735, 0.145, 0.075
    dx, dy = (xx - cx) / rx, (yy - cy) / ry
    ang = np.arctan2(dy, dx)
    rr = np.hypot(dx, dy) / (1 + 0.20 * np.cos(3 * ang + 0.6)
                             + 0.11 * np.sin(5 * ang - 0.3))
    clo = _sig(40 * (1 - rr))
    field -= 0.16 * _sig(20 * (1 - rr * 0.82)) * _sig(60 * (yy - cy))
    field = field * (1 - clo) + clo * 0.94
    for p0, p1 in (((0.615, 0.712), (0.757, 0.752)),
                   ((0.639, 0.781), (0.787, 0.740)),
                   ((0.675, 0.690), (0.715, 0.776))):
        field -= 0.12 * clo * np.exp(-(seg_dist(xx, yy, p0, p1) / 0.007) ** 2)

    # -- texture, then depth of field (background soft, subjects crisp) -----
    field += 0.09 * (value_noise(rng, ny, nx, base=5) - 0.5)
    far = gaussian_filter(field, 2.6)
    near = gaussian_filter(field, 0.50)
    dof = _sig(24 * (edge + 0.045 - yy)) * (1 - body) * (1 - lid) * (1 - clo)
    field = far * dof + near * (1 - dof)
    return (field - field.min()) / (np.ptp(field) + 1e-9)


def colorize(field, cmap, vignette=0.32, gain=1.1):
    ny, nx = field.shape
    yy = np.linspace(-1, 1, ny)[:, None]
    xx = np.linspace(-1, 1, nx)[None, :]
    rgb = cmap(np.clip((field - 0.5) * gain + 0.5, 0, 1))[..., :3]
    vig = 1.0 - vignette * (xx ** 2 + yy ** 2) / 2.0
    return np.clip(rgb * vig[..., None], 0, 1)


# ---- primitives ------------------------------------------------------------
def box(ax, x0, y0, w, h, fc="white", ec=RULE, lw=0.9, ls="-", r=R_BOX,
        zorder=3, alpha=1.0):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def photo(ax, rgb, x0, y0, w, h, ec="white", lw=1.0, r=0.05, zorder=6):
    clip = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor="none", edgecolor=ec, linewidth=lw,
                          zorder=zorder + 2)
    ax.add_patch(clip)
    im = ax.imshow(rgb, extent=[x0, x0 + w, y0, y0 + h], zorder=zorder,
                   interpolation="bilinear")
    im.set_clip_path(clip)
    return clip


def bezier(p0, p1, p2, n=140):
    t = np.linspace(0, 1, n)[:, None]
    p = ((1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1)
         + t ** 2 * np.array(p2))
    return p[:, 0], p[:, 1]


# ---- hand-drawn scene ------------------------------------------------------
def person(ax, x, y_feet, s=1.0, color=DARK):
    """Flat silhouette: head, torso, one raised arm, two legs."""
    ax.add_patch(Circle((x, y_feet + 1.52 * s), 0.17 * s, facecolor=color,
                        edgecolor="none", zorder=6))
    ax.add_patch(FancyBboxPatch(
        (x - 0.17 * s, y_feet + 0.62 * s), 0.34 * s, 0.74 * s,
        boxstyle="round,pad=0,rounding_size=0.09", facecolor=color,
        edgecolor="none", zorder=6))
    for dx in (-0.085, 0.085):
        ax.add_patch(FancyBboxPatch(
            (x + dx * s - 0.065 * s, y_feet), 0.13 * s, 0.66 * s,
            boxstyle="round,pad=0,rounding_size=0.055", facecolor=color,
            edgecolor="none", zorder=5))
    ax.plot([x + 0.13 * s, x + 0.40 * s], [y_feet + 1.22 * s,
                                           y_feet + 0.86 * s],
            color=color, lw=3.4 * s, solid_capstyle="round", zorder=6)


def thought_bubble(ax, cx, cy, a, b, lines, fc="white", ec=DARK, lw=1.1,
                   fontsize=9.0):
    th = np.linspace(0, 2 * np.pi, 520)
    rr = 1.0 + 0.085 * np.cos(9 * th)
    pts = np.c_[cx + a * rr * np.cos(th), cy + b * rr * np.sin(th)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec,
                         linewidth=lw, joinstyle="round", zorder=5))
    ax.text(cx, cy, lines, ha="center", va="center", fontsize=fontsize,
            color=INK, linespacing=1.55, zorder=7)


def bubble_trail(ax, pts, color=DARK):
    for (x, y, r) in pts:
        ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color,
                            linewidth=1.0, zorder=5))


def table(ax, x0, x1, y_top, y_foot, depth=0.20):
    top = Polygon([(x0, y_top), (x1, y_top), (x1 + depth, y_top + depth),
                   (x0 + depth, y_top + depth)], closed=True,
                  facecolor=lighten(WOOD, 0.18), edgecolor=WOOD_DK,
                  linewidth=1.0, zorder=4)
    ax.add_patch(top)
    ax.add_patch(Polygon([(x0, y_top - 0.13), (x1, y_top - 0.13),
                          (x1, y_top), (x0, y_top)], closed=True,
                         facecolor=WOOD, edgecolor=WOOD_DK, linewidth=1.0,
                         zorder=5))
    for lx, off in ((x0 + 0.10, 0.0), (x1 - 0.20, 0.0),
                    (x0 + 0.10 + depth, depth), (x1 - 0.20 + depth, depth)):
        ax.add_patch(Polygon(
            [(lx, y_foot + off), (lx + 0.11, y_foot + off),
             (lx + 0.11, y_top - 0.10 + off), (lx, y_top - 0.10 + off)],
            closed=True, facecolor=WOOD if off == 0 else lighten(WOOD, 0.22),
            edgecolor=WOOD_DK, linewidth=0.9, zorder=3 if off else 4))


def jar(ax, cx, y_base, w=0.26, h=0.34):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y_base), w, h,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor="#dbe6ea", edgecolor=RULE, linewidth=0.9, zorder=7))
    ax.add_patch(FancyBboxPatch(
        (cx - w * 0.42, y_base + h - 0.02), w * 0.84, 0.09,
        boxstyle="round,pad=0,rounding_size=0.025",
        facecolor=lighten(GOLD, 0.35), edgecolor=RULE, linewidth=0.9,
        zorder=8))


def cloth(ax, cx, y_base, w=0.44, h=0.14):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y_base), w, h,
        boxstyle="round,pad=0,rounding_size=0.05", facecolor="#efe7da",
        edgecolor=RULE, linewidth=0.9, zorder=7))
    ax.plot([cx - w / 2 + 0.05, cx + w / 2 - 0.05],
            [y_base + h * 0.55, y_base + h * 0.55], color=MUTED, lw=0.8,
            zorder=8)


def robot_head(ax, cx, cy, r=0.21):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=GOLD,
                        linewidth=1.2, zorder=7))
    ax.add_patch(FancyBboxPatch(
        (cx - r * 0.52, cy - r * 0.40), r * 1.04, r * 0.80,
        boxstyle="round,pad=0,rounding_size=0.03", facecolor=lighten(BLUE, 0.6),
        edgecolor=BLUE, linewidth=0.9, zorder=8))
    for dx in (-0.20, 0.20):
        ax.add_patch(Circle((cx + dx * r, cy), r * 0.11, facecolor=BLUE,
                            edgecolor="none", zorder=9))
    ax.plot([cx, cx], [cy + r * 0.40, cy + r * 0.66], color=BLUE, lw=1.0,
            zorder=8)
    ax.add_patch(Circle((cx, cy + r * 0.70), r * 0.09, facecolor=BLUE,
                        edgecolor="none", zorder=9))


# ---- figure ----------------------------------------------------------------
W, H = 11.8, 3.7


def main():
    set_pub_style()
    rng = np.random.default_rng(475)
    fig, ax = canvas(W, H)

    # -- left: the user and the scene ---------------------------------------
    person(ax, 0.92, 0.74)
    bubble_trail(ax, [(1.24, 2.40, 0.052), (1.44, 2.51, 0.075),
                      (1.68, 2.64, 0.098)])
    thought_bubble(
        ax, 2.92, 2.98, 1.24, 0.50,
        "Ah, I need something to open\nthis slippery jar more easily...")
    table(ax, 2.06, 4.16, 1.62, 0.72)
    jar(ax, 2.86, 1.82)
    cloth(ax, 3.62, 1.82)
    ax.text(3.05, 0.46, "user + egocentric observation", ha="center",
            va="center", fontsize=7.2, color=MUTED, style="italic")

    # -- right: the assistant card ------------------------------------------
    cx0, cy0, cw, ch = 5.02, 0.36, 6.42, 2.66
    box(ax, cx0, cy0, cw, ch, fc=CARD_FC, ec=GOLD, lw=1.3,
        ls=(0, (6, 3)), r=0.14, zorder=3)
    ax.text(7.55, 3.30, "HandoverVLM", ha="right", va="center", fontsize=11.0,
            fontweight="bold", style="italic", color=INK)
    ax.text(7.63, 3.30, "Assistant", ha="left", va="center", fontsize=11.0,
            color=SUB)
    robot_head(ax, 11.16, 3.30)

    box(ax, cx0 + 0.30, cy0 + ch - 0.52, 1.62, 0.30, fc="white",
        ec=lighten(GOLD, 0.35), lw=0.9, r=0.06, zorder=5)
    ax.text(cx0 + 1.11, cy0 + ch - 0.37, "Assistant response", ha="center",
            va="center", fontsize=7.0, color=SUB, zorder=6)

    tx, ty = cx0 + 0.30, cy0 + ch - 0.76
    ax.text(tx, ty, "To open the slippery lid, first grab the\n"
                    "folded cloth to the right of the jar. Here\n"
                    "is a plausible trajectory for your right\n"
                    "hand, overlaid on the scene:",
            ha="left", va="top", fontsize=8.6, color=INK, linespacing=1.62,
            zorder=6)
    ax.text(tx, ty - 0.98, "<HAND> <HAND> ......... <HAND>.", ha="left",
            va="top", fontsize=8.6, color=RED, style="italic", zorder=6)
    ax.text(tx, cy0 + 0.30, "12 waypoints  ·  grasp affordance  ·  0.4 s",
            ha="left", va="center", fontsize=6.8, color=MUTED, zorder=6)

    # -- thumbnail with the predicted hand trajectory ------------------------
    px0, py0, pw, ph = 8.66, cy0 + 0.26, 2.52, 2.14
    thumb = colorize(tabletop(rng), CMAP_WOOD, vignette=0.20, gain=1.0)
    clip = photo(ax, thumb, px0, py0, pw, ph,
                 ec="white", lw=1.2, r=0.06, zorder=6)
    # from the cloth, bowing right, up to the mouth of the jar
    bx, by = bezier((px0 + 0.700 * pw, py0 + 0.265 * ph),
                    (px0 + 0.950 * pw, py0 + 0.560 * ph),
                    (px0 + 0.345 * pw, py0 + 0.735 * ph))
    ln, = ax.plot(bx, by, color=RED, lw=1.5, ls=(0, (4.5, 2.6)), zorder=9)
    ln.set_clip_path(clip)
    dot = Circle((bx[0], by[0]), 0.055, facecolor=RED, edgecolor="white",
                 linewidth=0.8, zorder=10)
    ax.add_patch(dot)
    head = FancyArrowPatch((bx[-6], by[-6]), (bx[-1], by[-1]),
                           arrowstyle="-|>", mutation_scale=11, color=RED,
                           linewidth=1.5, shrinkA=0, shrinkB=0, zorder=10)
    ax.add_patch(head)
    head.set_clip_path(clip)
    ax.text(px0 + pw / 2, py0 - 0.14, "predicted hand trajectory",
            ha="center", va="center", fontsize=6.8, color=MUTED)

    savefig(fig, "1070_wm_teaser_vla_assistant.png")


if __name__ == "__main__":
    main()
