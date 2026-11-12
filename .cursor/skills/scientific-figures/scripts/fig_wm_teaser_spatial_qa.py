"""Spatial reasoning QA teaser: tagged video frames -> question -> answers.

Top row is a four-frame clip of one indoor scene while the camera pans right;
three tracked objects are boxed in the house categorical colours and tagged
with numbered discs.  A legend names the tags, a question bar states the task
with its ground truth, and the bottom row scores four models as answer chips:
a drawn tick on a pale green card, a drawn cross on a pale red one.

The frames are procedural: the same soft-edged objects are painted into the
noise field at the tracked positions, so every box actually encloses
something visible instead of floating over flat texture.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
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

CMAP_ROOM = LinearSegmentedColormap.from_list(
    "wm_room", ["#4a4540", "#6f6961", "#9b948a", "#c8c2b8", "#efece6"])

# (name, colour, x0, y0, w, h, drift_x, brightness) in frame-normalised units
OBJECTS = [("sofa", BLUE, 0.070, 0.10, 0.32, 0.34, -0.010, -0.20),
           ("lamp", GOLD, 0.560, 0.30, 0.13, 0.44, -0.032, 0.26),
           ("mug", RED, 0.440, 0.13, 0.105, 0.17, 0.060, 0.22)]

W, H = 11.6, 4.05
FW, FH, FGAP = 2.60, 1.55, 0.16
FX0 = (W - (4 * FW + 3 * FGAP)) / 2
FY0 = 2.14


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


def soft_rect(xx, yy, x0, x1, y0, y1, k=60):
    return (_sig(k * (xx - x0)) * _sig(k * (x1 - xx))
            * _sig(k * (yy - y0)) * _sig(k * (y1 - yy)))


def room_shot(rng, boxes, ny=76, nx=128):
    """Indoor frame with the tracked objects painted in at their boxes."""
    yy = np.linspace(0, 1, ny)[:, None]      # 0 = top of the image
    xx = np.linspace(0, 1, nx)[None, :]
    field = np.where(yy < 0.52, 0.78 - 0.08 * yy, 0.60 + 0.16 * (yy - 0.52))
    field = field + 0.0 * xx
    field += 0.05 * _sig(40 * (yy - 0.52))
    for (x0, y0, w, h, amp) in boxes:
        # frame coords are bottom-origin; the image field is top-origin
        field += amp * soft_rect(xx, yy, x0, x0 + w, 1 - (y0 + h), 1 - y0)
    field += 0.13 * (value_noise(rng, ny, nx, base=4) - 0.5)
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


def tick_mark(ax, x, y, s=0.058, color=GREEN, lw=1.6, zorder=9):
    ax.plot([x - s, x - s * 0.25, x + s], [y, y - s * 0.85, y + s * 0.9],
            color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder)


def cross_mark(ax, x, y, s=0.050, color=RED, lw=1.6, zorder=9):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=lw,
            solid_capstyle="round", zorder=zorder)


def tag_disc(ax, x, y, n, color, r=0.078, zorder=10):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white",
                        linewidth=0.9, zorder=zorder))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=6.2,
            fontweight="bold", color="white", zorder=zorder + 1)


# ---- panels ----------------------------------------------------------------
def draw_frames(ax, rng):
    times = ["0.0 s", "0.6 s", "1.2 s", "1.8 s"]
    for k in range(4):
        fx = FX0 + k * (FW + FGAP)
        boxes = [(x0 + dx * k, y0, w, h, amp)
                 for (_, _, x0, y0, w, h, dx, amp) in OBJECTS]
        clip = photo(ax, colorize(room_shot(rng, boxes), CMAP_ROOM), fx, FY0,
                     FW, FH, ec="white", lw=1.1, r=0.05, zorder=5)
        for j, (_, col, x0, y0, w, h, dx, _) in enumerate(OBJECTS):
            bx = fx + (x0 + dx * k) * FW
            by = FY0 + y0 * FH
            rect = Rectangle((bx, by), w * FW, h * FH, facecolor="none",
                             edgecolor=col, linewidth=1.3, zorder=8)
            ax.add_patch(rect)
            rect.set_clip_path(clip)
            tag_disc(ax, bx, by + h * FH, j + 1, col)
        ax.text(fx + FW / 2, FY0 + FH + 0.15, f"frame {k + 1}  ·  $t$ = "
                f"{times[k]}", ha="center", va="center", fontsize=7.4,
                color=SUB)


def draw_legend(ax):
    x = FX0 + 0.02
    y = 1.98
    for j, (name, col, *_rest) in enumerate(OBJECTS):
        tag_disc(ax, x + 0.078, y, j + 1, col, r=0.078, zorder=6)
        ax.text(x + 0.20, y, name, ha="left", va="center", fontsize=7.2,
                color=SUB)
        x += 0.24 + 0.070 * len(name) + 0.42
    ax.text(FX0 + 4 * FW + 3 * FGAP, y, "camera pans right", ha="right",
            va="center", fontsize=7.2, color=MUTED, style="italic")


def draw_question(ax):
    x0 = FX0
    w = 4 * FW + 3 * FGAP
    y0, h = 1.30, 0.54
    box(ax, x0, y0, w, h, fc=lighten(BLUE, 0.94), ec=lighten(BLUE, 0.50),
        lw=1.0, r=0.09, zorder=3)
    box(ax, x0 + 0.16, y0 + 0.11, 0.32, 0.32, fc=BLUE, ec="none", r=0.06,
        zorder=5)
    ax.text(x0 + 0.32, y0 + 0.27, "Q", ha="center", va="center", fontsize=9.0,
            fontweight="bold", color="white", zorder=6)
    ax.text(x0 + 0.62, y0 + 0.27, "From the camera pose in the last frame, "
            "which object ends up closest to the mug?", ha="left",
            va="center", fontsize=8.6, color=INK, style="italic", zorder=5)
    ax.text(x0 + w - 0.18, y0 + 0.27, "ground truth:  sofa", ha="right",
            va="center", fontsize=7.2, color=MUTED, zorder=5)


def draw_answers(ax):
    models = [("Video-LLM A", "the lamp, it is nearer the camera", False),
              ("Video-LLM B", "cannot be determined", False),
              ("Video-LLM C", "the lamp", False),
              ("SpatialWM (ours)", "the sofa, 0.8 m to its left", True)]
    y0, h = 0.30, 0.76
    for k, (name, answer, ok) in enumerate(models):
        x = FX0 + k * (FW + FGAP)
        col = GREEN if ok else RED
        box(ax, x, y0, FW, h, fc=lighten(col, 0.90), ec=lighten(col, 0.45),
            lw=1.4 if ok else 1.0, r=0.08, zorder=3)
        ax.text(x + 0.20, y0 + 0.52, name, ha="left", va="center",
                fontsize=8.0, fontweight="bold", color=INK, zorder=5)
        ax.text(x + 0.20, y0 + 0.24, answer, ha="left", va="center",
                fontsize=7.0, color=SUB, style="italic", zorder=5)
        if ok:
            tick_mark(ax, x + FW - 0.24, y0 + 0.52, color=col)
        else:
            cross_mark(ax, x + FW - 0.24, y0 + 0.52, color=col)


def main():
    set_pub_style()
    rng = np.random.default_rng(1478)
    fig, ax = canvas(W, H)
    draw_frames(ax, rng)
    draw_legend(ax)
    draw_question(ax)
    draw_answers(ax)
    savefig(fig, "1073_wm_teaser_spatial_qa.png")


if __name__ == "__main__":
    main()
