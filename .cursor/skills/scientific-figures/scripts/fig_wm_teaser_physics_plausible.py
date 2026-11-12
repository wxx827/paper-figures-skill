"""Physical plausibility teaser: prior decomposition -> two rollout verdicts.

Left is the conditioning frame plus the text prompt that drives the rollout.
The centre is a dashed container holding three physics-prior cards, each with
a hand-drawn glyph (falling mass, impact, squashed body) and the constant it
contributes.  Right contrasts two rollouts of the same prompt: the baseline
row is boxed in red where the sequence breaks physics, the ours row is boxed
in green with a drawn tick.  A full-width bar carries the headline metrics.

Frames are procedural (multi-octave value noise + Gaussian blur + muted ramp
+ vignette) with a table slab and a bright ball placed per frame, so the
object visibly travels along each row.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Ellipse, Polygon)
from scipy.ndimage import gaussian_filter, zoom
from pubstyle import set_pub_style, savefig, lighten

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
PURPLE = "#8e6bbf"

CMAP_LAB = LinearSegmentedColormap.from_list(
    "wm_lab", ["#494440", "#6e6862", "#9a938a", "#c9c3ba", "#f0ede7"])

W, H = 12.0, 4.0
Y_CAP = 3.76
BAND_TOP = 3.58


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


def lab_scene(rng, ball=(0.30, 0.62), table_right=0.52, ny=72, nx=104):
    """Bench scene: a table slab on the left, one bright ball in flight.

    ``ball`` is (x, y) in bottom-origin frame coordinates.
    """
    yy = np.linspace(0, 1, ny)[:, None]      # 0 = top of the image
    xx = np.linspace(0, 1, nx)[None, :]
    field = np.where(yy < 0.48, 0.80 - 0.10 * yy, 0.58 + 0.14 * (yy - 0.48))
    field = field + 0.0 * xx
    field -= 0.20 * (_sig(50 * (0.62 - yy)) * _sig(50 * (yy - 0.52))
                     * _sig(45 * (table_right - xx)))
    bx, by = ball[0], 1.0 - ball[1]
    field += 0.34 * np.exp(-(((xx - bx) / 0.075) ** 2
                             + ((yy - by) / 0.105) ** 2))
    field += 0.12 * (value_noise(rng, ny, nx, base=4) - 0.5)
    field = gaussian_filter(field, 1.0)
    return (field - field.min()) / (np.ptp(field) + 1e-9)


def colorize(field, cmap, vignette=0.24, gain=1.08):
    ny, nx = field.shape
    yy = np.linspace(-1, 1, ny)[:, None]
    xx = np.linspace(-1, 1, nx)[None, :]
    rgb = cmap(np.clip((field - 0.5) * gain + 0.5, 0, 1))[..., :3]
    vig = 1.0 - vignette * (xx ** 2 + yy ** 2) / 2.0
    return np.clip(rgb * vig[..., None], 0, 1)


# ---- primitives ------------------------------------------------------------
def box(ax, x0, y0, w, h, fc="white", ec=RULE, lw=0.9, ls="-", r=0.07,
        zorder=3, alpha=1.0):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def photo(ax, rgb, x0, y0, w, h, ec="white", lw=1.0, r=0.05, zorder=5):
    clip = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor="none", edgecolor=ec, linewidth=lw,
                          zorder=zorder + 4)
    ax.add_patch(clip)
    im = ax.imshow(rgb, extent=[x0, x0 + w, y0, y0 + h], zorder=zorder,
                   interpolation="bilinear")
    im.set_clip_path(clip)
    return clip


def arrow(ax, p0, p1, color=RULE, lw=1.3, ms=12, zorder=9):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                        zorder=zorder)
    ax.add_patch(a)
    return a


def caption(ax, x, text):
    ax.text(x, Y_CAP, text, ha="center", va="center", fontsize=8.8,
            fontweight="bold", color=INK)


def tick_mark(ax, x, y, s=0.055, color=GREEN, lw=1.6, zorder=12):
    ax.plot([x - s, x - s * 0.25, x + s], [y, y - s * 0.85, y + s * 0.9],
            color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder)


def cross_mark(ax, x, y, s=0.048, color=RED, lw=1.6, zorder=12):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)


# ---- physics glyphs --------------------------------------------------------
def glyph_gravity(ax, cx, cy, s=0.17, color=PURPLE, zorder=8):
    ax.add_patch(Circle((cx, cy + 0.85 * s), 0.26 * s, facecolor=color,
                        edgecolor="none", zorder=zorder))
    arrow(ax, (cx, cy + 0.52 * s), (cx, cy - 0.55 * s), color=color, lw=1.0,
          ms=8, zorder=zorder)
    ax.plot([cx - s, cx + s], [cy - 0.85 * s, cy - 0.85 * s], color=color,
            lw=1.0, zorder=zorder)


def glyph_collision(ax, cx, cy, s=0.17, color=PURPLE, zorder=8):
    ax.add_patch(Circle((cx - 0.46 * s, cy), 0.40 * s, facecolor="none",
                        edgecolor=color, linewidth=1.1, zorder=zorder))
    ax.add_patch(Circle((cx + 0.46 * s, cy), 0.40 * s, facecolor=color,
                        edgecolor="none", alpha=0.55, zorder=zorder))
    for a in (-0.9, 0.0, 0.9):
        ax.plot([cx + 0.06 * s * np.cos(a), cx + 0.42 * s * np.cos(a)],
                [cy + 0.55 * s * np.sin(a) + 0.62 * s * np.cos(a) * 0,
                 cy + 0.95 * s * np.sin(a)], color=color, lw=0.8,
                zorder=zorder + 1)


def glyph_deform(ax, cx, cy, s=0.17, color=PURPLE, zorder=8):
    ax.add_patch(Ellipse((cx, cy - 0.18 * s), 1.7 * s, 0.85 * s,
                         facecolor="none", edgecolor=color, linewidth=1.1,
                         zorder=zorder))
    ax.plot([cx - s, cx + s], [cy - 0.68 * s, cy - 0.68 * s], color=color,
            lw=1.0, zorder=zorder)
    for dx in (-0.42, 0.42):
        arrow(ax, (cx + dx * s, cy + 0.90 * s), (cx + dx * s, cy + 0.34 * s),
              color=color, lw=0.9, ms=7, zorder=zorder)


# ---- panels ----------------------------------------------------------------
def draw_input(ax, rng):
    x0, w = 0.38, 1.90
    caption(ax, x0 + w / 2, "Conditioning frame")
    h = 1.38
    photo(ax, colorize(lab_scene(rng, ball=(0.34, 0.66)), CMAP_LAB), x0,
          BAND_TOP - h, w, h, ec="white", lw=1.1, r=0.05, zorder=5)
    box(ax, x0, 1.62, w, 0.40, fc=lighten(GOLD, 0.90),
        ec=lighten(GOLD, 0.42), lw=0.9, r=0.06, zorder=3)
    ax.text(x0 + w / 2, 1.82, "\"the mug rolls off the table\"", ha="center",
            va="center", fontsize=7.2, color=SUB, style="italic", zorder=5)
    return x0 + w


def draw_priors(ax):
    x0, x1 = 2.80, 5.60
    caption(ax, (x0 + x1) / 2, "Physics prior decomposition")
    box(ax, x0, 1.50, x1 - x0, BAND_TOP - 1.50, fc=lighten(PURPLE, 0.95),
        ec=lighten(PURPLE, 0.48), lw=1.1, ls=(0, (6, 3)), r=0.11, zorder=2)
    cards = [("Gravity", "$g$ = 9.81 m s$^{-2}$", glyph_gravity),
             ("Collision", "restitution $e$ = 0.42", glyph_collision),
             ("Deformation", "stiffness $E$ = 1.4 MPa", glyph_deform)]
    ch, gap = 0.54, 0.11
    top = BAND_TOP - 0.12
    for k, (name, sub, glyph) in enumerate(cards):
        y = top - (k + 1) * ch - k * gap
        box(ax, x0 + 0.16, y, (x1 - x0) - 0.32, ch, fc="white", ec=HAIR,
            lw=0.9, r=0.07, zorder=4)
        glyph(ax, x0 + 0.48, y + ch / 2)
        ax.text(x0 + 0.82, y + ch * 0.62, name, ha="left", va="center",
                fontsize=8.4, fontweight="bold", color=INK, zorder=8)
        ax.text(x0 + 0.82, y + ch * 0.28, sub, ha="left", va="center",
                fontsize=6.8, color=MUTED, zorder=8)
    return x0, x1


def draw_rollouts(ax, rng):
    lx0, lw_ = 6.30, 0.86
    tx0, tw, tgap = 7.20, 1.42, 0.10
    caption(ax, tx0 + (3 * tw + 2 * tgap) / 2, "Rollout under the same prompt")
    rows = [dict(y0=2.66, h=0.92, col=RED, ok=False, name="Baseline",
                 verdict="implausible", score="PhyScore 0.31",
                 balls=[(0.62, 0.66), (0.72, 0.68), (0.84, 0.67)],
                 notes=["mug hovers", "passes through", "no rebound"]),
            dict(y0=1.50, h=0.92, col=GREEN, ok=True, name="Ours",
                 verdict="plausible", score="PhyScore 0.89",
                 balls=[(0.60, 0.52), (0.72, 0.30), (0.84, 0.16)],
                 notes=["free fall", "impact", "settles"])]
    for row in rows:
        y0, h, col = row["y0"], row["h"], row["col"]
        ax.text(lx0, y0 + h * 0.70, row["name"], ha="left", va="center",
                fontsize=8.6, fontweight="bold", color=INK)
        ax.text(lx0, y0 + h * 0.44, row["verdict"], ha="left", va="center",
                fontsize=7.4, color=col, style="italic")
        ax.text(lx0, y0 + h * 0.18, row["score"], ha="left", va="center",
                fontsize=6.8, color=MUTED)
        for k, (bx, by) in enumerate(row["balls"]):
            x = tx0 + k * (tw + tgap)
            clip = photo(ax, colorize(lab_scene(rng, ball=(bx, by)), CMAP_LAB),
                         x, y0, tw, h, ec=col, lw=1.5, r=0.05, zorder=5)
            mark = FancyBboxPatch(
                (x + (bx - 0.13) * tw, y0 + (by - 0.20) * h), 0.26 * tw,
                0.40 * h, boxstyle="round,pad=0,rounding_size=0.03",
                facecolor="none", edgecolor=col, linewidth=1.1,
                linestyle=(0, (3.5, 2.2)) if not row["ok"] else "-",
                zorder=10)
            ax.add_patch(mark)
            mark.set_clip_path(clip)
            ax.text(x + tw / 2, y0 - 0.13, row["notes"][k], ha="center",
                    va="center", fontsize=6.6, color=col)
        if row["ok"]:
            tick_mark(ax, lx0 + 0.66, y0 + h * 0.70)
        else:
            cross_mark(ax, lx0 + 0.66, y0 + h * 0.70)
    return lx0, tx0 + 3 * tw + 2 * tgap


def draw_metrics(ax):
    y0, h = 0.28, 0.56
    box(ax, 0.38, y0, 11.28, h, fc="#f5f5f5", ec=HAIR, lw=0.8, r=0.07,
        zorder=2)
    ax.text(0.62, y0 + h / 2, "Physics evaluation", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=INK, zorder=4)
    items = [("plausibility", "0.31", "0.89"),
             ("penetration rate", "12.4 %", "2.1 %"),
             ("energy drift", "0.44 J", "0.06 J"),
             ("human preference", "22 %", "78 %")]
    cy = y0 + h / 2
    slot0, pitch = 2.26, 2.24
    for k, (name, before, after) in enumerate(items):
        sx = slot0 + k * pitch
        ax.text(sx, cy, name, ha="left", va="center", fontsize=6.8, color=SUB,
                zorder=4)
        ax.text(sx + 1.34, cy, before, ha="right", va="center", fontsize=7.2,
                color=RED, zorder=4)
        arrow(ax, (sx + 1.42, cy), (sx + 1.66, cy), color=MUTED, lw=0.9,
              ms=8, zorder=4)
        ax.text(sx + 1.74, cy, after, ha="left", va="center", fontsize=7.2,
                fontweight="bold", color=GREEN, zorder=4)


def main():
    set_pub_style()
    rng = np.random.default_rng(1479)
    fig, ax = canvas(W, H)

    in_right = draw_input(ax, rng)
    px0, px1 = draw_priors(ax)
    rx0, _ = draw_rollouts(ax, rng)

    arrow(ax, (in_right + 0.07, 2.62), (px0 - 0.07, 2.62), lw=1.3, ms=13)
    arrow(ax, (px1 + 0.07, 2.62), (rx0 - 0.12, 2.62), lw=1.3, ms=13)
    ax.text((px1 + rx0) / 2 - 0.03, 2.82, "simulate", ha="center",
            va="center", fontsize=6.8, color=MUTED)
    draw_metrics(ax)

    savefig(fig, "1074_wm_teaser_physics_plausible.png")


if __name__ == "__main__":
    main()
