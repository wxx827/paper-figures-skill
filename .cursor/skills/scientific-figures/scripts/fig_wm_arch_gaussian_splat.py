"""4D Gaussian splatting rendering pipeline.

Left to right: the 4D Gaussian cloud swept along time, the time marginal and
the view-dependent colour it feeds, the projection of the time-sliced 3D
Gaussians into planar Gaussians on the image plane, and the depth-sorted
alpha compositing that produces the rendered view.  The rendering equation is
typeset across the bottom with each symbol tinted to match the stage that
produces it.

Geometry is expressed in inches on an equal-aspect canvas so corner radii,
arrow heads and gaps are identical everywhere.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Ellipse,
                                Polygon)
from pubstyle import set_pub_style, savefig, get_cmap

BLUE = "#3b5b92"
ORANGE = "#c0603f"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
FAINT = "#8a8a8a"
LINE = "#d5d5d5"

R = 0.06
LW = 0.9
DASH = (0, (6, 3))
DOT = (0, (1.6, 1.8))
SEED = 1483  # original id; figure number is 1078


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---- procedural imagery ----------------------------------------------------
def _blur(a, sigma):
    if sigma <= 0:
        return a
    r = max(int(3 * sigma), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, p)
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, p)


def _value_noise(rng, n, octaves=(2, 4, 8, 16), weights=(1.0, .58, .34, .19)):
    out = np.zeros((n, n))
    for o, wt in zip(octaves, weights):
        g = rng.random((o, o))
        t = np.linspace(0, o - 1, n)
        i0 = np.floor(t).astype(int)
        i1 = np.minimum(i0 + 1, o - 1)
        f = t - i0
        top = g[i0][:, i0] * (1 - f)[None, :] + g[i0][:, i1] * f[None, :]
        bot = g[i1][:, i0] * (1 - f)[None, :] + g[i1][:, i1] * f[None, :]
        out += wt * (top * (1 - f)[:, None] + bot * f[:, None])
    return (out - out.min()) / max(np.ptp(out), 1e-9)


SCENE = LinearSegmentedColormap.from_list(
    "scene", ["#3a4560", "#5b7a97", "#9ab6c8", "#dfc9a4", "#f6f0e6"])


def synth_frame(seed, n=88, sigma=0.9, cmap=SCENE):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    field = 0.48 * _value_noise(rng, n) + 0.52 * (1.0 - yy) ** 1.5
    for cx, cy, s, a in ((0.42, 0.58, 0.14, 0.78), (0.70, 0.44, 0.09, 0.40)):
        field += a * np.exp(-(((xx - cx) / s) ** 2 + ((yy - cy) / s) ** 2))
    field = _blur(field, sigma)
    field = (field - field.min()) / max(np.ptp(field), 1e-9)
    return 0.84 * cmap(0.08 + 0.84 * field)[..., :3] + 0.16


def image_tile(ax, x, y, w, h, rgb, ec=RULE, lw=0.8, zorder=4, rounding=R):
    clip = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={rounding}",
                          facecolor="white", edgecolor="none", zorder=zorder)
    ax.add_patch(clip)
    im = ax.imshow(rgb, extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
                   origin="upper", zorder=zorder + 0.2,
                   interpolation="bilinear")
    im.set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor="none", edgecolor=ec, linewidth=lw,
                                zorder=zorder + 0.4))
    return dict(x=x, y=y, w=w, h=h)


# ---- containers / arrows ---------------------------------------------------
def container(ax, x0, x1, y0, y1, color, title=None, rounding=0.14, zorder=1):
    w, h = x1 - x0, y1 - y0
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor=color, alpha=0.04, edgecolor="none",
                                zorder=zorder))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor="none", edgecolor=color,
                                linewidth=1.0, linestyle=DASH,
                                zorder=zorder + 0.5))
    if title:
        ax.text(x0 + 0.10, y1 + 0.10, title, ha="left", va="bottom",
                fontsize=8.2, fontweight="bold", color=color, zorder=zorder + 1)
    return dict(x0=x0, x1=x1, y0=y0, y1=y1, cx=(x0 + x1) / 2)


def arrow(ax, p0, p1, color=RULE, rad=0.0, lw=1.0, ls="-", ms=12, zorder=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def caption(ax, x, y, main, sub=None, color=INK, size=7.4, sub_color=MUTED):
    ax.text(x, y, main, ha="center", va="top", fontsize=size,
            fontweight="bold", color=color, zorder=8)
    if sub:
        ax.text(x, y - 0.19, sub, ha="center", va="top", fontsize=7.0,
                color=sub_color, zorder=8)


# ---- figure ----------------------------------------------------------------
W, H = 12.4, 4.95
Y0, Y1 = 2.05, 4.10
MID = 3.18
A = (0.45, 3.25)
B = (3.55, 5.45)
C = (5.75, 8.35)
D = (8.65, 10.50)
OUT = 11.40

CLOUD = ["#9cc3dd", "#c1acd8", "#c48ba9", "#84b6d6", "#a4d7b8"]


def gaussian_cloud(ax):
    container(ax, *A, Y0, Y1, PURPLE, "4D Gaussian field")
    rng = np.random.default_rng(SEED)
    xs = np.linspace(0.78, 2.94, 220)
    spine = MID + 0.20 * np.sin((xs - 0.78) * 2.1)
    half = 0.36 + 0.10 * np.cos((xs - 0.78) * 1.4)
    ax.fill_between(xs, spine - half, spine + half, facecolor=BLUE,
                    alpha=0.07, zorder=2)
    ax.plot(xs, spine + half, color=BLUE, lw=0.9, zorder=3)
    ax.plot(xs, spine - half, color=BLUE, lw=0.9, zorder=3)
    for xe in (xs[0], xs[-1]):
        k = 0 if xe == xs[0] else -1
        ax.add_patch(Ellipse((xe, spine[k]), 0.13, 2 * half[k],
                             facecolor="none", edgecolor=BLUE, lw=0.9,
                             zorder=3))
    for i in range(26):
        u = rng.random()
        x = 0.82 + u * 2.06
        s = MID + 0.20 * np.sin((x - 0.78) * 2.1)
        hh = 0.34 + 0.10 * np.cos((x - 0.78) * 1.4)
        y = s + rng.normal(0, 0.34) * hh
        w = rng.uniform(0.13, 0.30)
        ax.add_patch(Ellipse((x, y), w, w * rng.uniform(0.45, 0.9),
                             angle=rng.uniform(0, 180),
                             facecolor=CLOUD[i % len(CLOUD)], alpha=0.45,
                             edgecolor="none", zorder=4))
    for x in (1.25, 2.05, 2.75):
        s = MID + 0.20 * np.sin((x - 0.78) * 2.1)
        for dx, dy in ((0.22, 0.16), (-0.20, 0.13), (0.03, -0.24)):
            ax.annotate("", xy=(x + dx, s + dy), xytext=(x, s),
                        arrowprops=dict(arrowstyle="-|>", color=FAINT,
                                        lw=0.7, linestyle=DOT,
                                        mutation_scale=7, shrinkA=0,
                                        shrinkB=0), zorder=5)
    ax.annotate("", xy=(2.86, 3.86), xytext=(0.86, 3.72),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1,
                                linestyle=(0, (5, 2.6)), mutation_scale=11,
                                shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=-0.10"), zorder=6)
    ax.text(2.90, 3.92, "Time", ha="right", va="bottom", fontsize=7.4,
            fontweight="bold", color=ORANGE, zorder=8)
    caption(ax, 1.85, 2.52, "4D Gaussians", r"$p(x,y,z,t)$", color=PURPLE)


def time_and_colour(ax):
    container(ax, *B, Y0, Y1, GOLD, "Time + appearance")
    x0, x1 = 3.72, 5.28
    base = 3.42
    xs = np.linspace(x0, x1, 200)
    ys = base + 0.56 * np.exp(-0.5 * ((xs - 4.54) / 0.26) ** 2)
    ax.plot(xs, ys, color=ORANGE, lw=1.6, zorder=5)
    ax.annotate("", xy=(x1 + 0.06, base), xytext=(x0 - 0.06, base),
                arrowprops=dict(arrowstyle="-|>", color=RULE, lw=0.8,
                                mutation_scale=10, shrinkA=0, shrinkB=0),
                zorder=4)
    ax.plot([4.54, 4.54], [base, base + 0.56], color=GOLD, lw=0.9,
            ls=DOT, zorder=5)
    ax.text(4.54, base - 0.09, "$t$", ha="center", va="top", fontsize=7.0,
            color=MUTED, zorder=8)
    ax.text(4.86, 3.86, "$p_i(t)$", ha="left", va="center", fontsize=8.6,
            fontweight="bold", color=GOLD, zorder=8)
    ax.text(4.50, 3.18, "marginal time weight", ha="center", va="center",
            fontsize=6.4, color=MUTED, zorder=8)

    bx0, bx1, by, bh = 3.76, 5.24, 2.78, 0.30
    plate = FancyBboxPatch((bx0, by - bh / 2), bx1 - bx0, bh,
                           boxstyle="round,pad=0,rounding_size=0.04",
                           facecolor="white", edgecolor="none", zorder=4)
    ax.add_patch(plate)
    im = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1),
                   extent=(bx0, bx1, by - bh / 2, by + bh / 2), origin="lower",
                   cmap=get_cmap("cb_diverging"), vmin=0, vmax=1,
                   zorder=4.2, interpolation="bilinear")
    im.set_clip_path(plate)
    ax.add_patch(FancyBboxPatch((bx0, by - bh / 2), bx1 - bx0, bh,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor="none", edgecolor=RULE,
                                linewidth=0.8, zorder=4.4))
    caption(ax, 4.50, 2.52, "$c(d_i,t)$", "view- and time-dependent colour",
            color=ORANGE, size=8.6)


def projection(ax):
    container(ax, *C, Y0, Y1, BLUE, "Time slice + projection")
    src = [(6.24, 3.44, 0.34, 0.20, 25), (6.52, 3.12, 0.30, 0.17, -35),
           (6.20, 2.86, 0.24, 0.15, 60)]
    for x, y, w, h, a in src:
        ax.add_patch(Ellipse((x, y), w, h, angle=a, facecolor=PURPLE,
                             alpha=0.30, edgecolor=PURPLE, lw=0.7, zorder=4))
        ax.add_patch(Ellipse((x, y), w * 0.55, h * 0.55, angle=a,
                             facecolor=PURPLE, alpha=0.45, edgecolor="none",
                             zorder=5))
    quad = [(7.52, 2.72), (8.22, 2.94), (8.22, 4.02), (7.52, 3.80)]
    ax.add_patch(Polygon(quad, closed=True, facecolor=BLUE, alpha=0.10,
                         edgecolor=BLUE, lw=1.0, zorder=4))
    dst = [(7.78, 3.62, 0.26, 0.15), (7.98, 3.34, 0.22, 0.13),
           (7.74, 3.12, 0.18, 0.11)]
    for (x, y, w, h), s in zip(dst, src):
        ax.add_patch(Ellipse((x, y), w, h, angle=s[4] * 0.5, facecolor=BLUE,
                             alpha=0.42, edgecolor=BLUE, lw=0.6, zorder=6))
        arrow(ax, (s[0] + s[2] / 2 + 0.03, s[1]), (x - w / 2 - 0.05, y),
              color=FAINT, lw=0.7, ls=DOT, ms=9, rad=0.10)
    ax.text(7.06, 3.74, r"$\pi(\cdot\,|\,t)$", ha="center", va="center",
            fontsize=7.6, color=MUTED, zorder=8)
    caption(ax, 6.35, 2.52, "3D Gaussians", r"$p(x,y,z\,|\,t)$", color=PURPLE)
    caption(ax, 7.87, 2.52, "planar Gaussians", r"$p_i(u,v\,|\,t)$",
            color=BLUE)


def compositing(ax):
    container(ax, *D, Y0, Y1, GREEN, "Sort + blend")
    rng = np.random.default_rng(SEED + 4)
    for i in range(4):
        x = 9.28 + 0.16 * i
        y = 3.72 - 0.30 * i
        quad = [(x - 0.42, y - 0.20), (x + 0.42, y - 0.08),
                (x + 0.42, y + 0.28), (x - 0.42, y + 0.16)]
        ax.add_patch(Polygon(quad, closed=True, facecolor="white",
                             edgecolor=GREEN, lw=0.8, alpha=0.96,
                             zorder=4 + i))
        for _ in range(3):
            ax.add_patch(Ellipse((x + rng.uniform(-0.22, 0.22),
                                  y + rng.uniform(-0.02, 0.14)),
                                 rng.uniform(0.14, 0.26),
                                 rng.uniform(0.08, 0.13),
                                 angle=rng.uniform(0, 180), facecolor=GREEN,
                                 alpha=0.30, edgecolor="none",
                                 zorder=4.5 + i))
        ax.text(x + 0.50, y + 0.06, f"$\\alpha_{i + 1}$", ha="left",
                va="center", fontsize=6.6, color=GREEN, zorder=9)
    ax.annotate("", xy=(8.84, 2.62), xytext=(8.84, 3.86),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.9,
                                mutation_scale=10, shrinkA=0, shrinkB=0),
                zorder=6)
    ax.text(8.76, 3.24, "near $\\to$ far", ha="center", va="center",
            fontsize=6.2, color=GREEN, rotation=90, zorder=8)
    caption(ax, 9.62, 2.52, "depth-sorted", r"$\alpha$-compositing",
            color=GREEN)


def rendering_equation(fig, ax):
    ax.add_patch(FancyBboxPatch((0.45, 0.50), W - 0.90, 1.22,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#f5f5f5", edgecolor=LINE,
                                linewidth=0.8, zorder=1))
    parts = [(r"$\hat{I}(u,v,t)\;=\;$", INK),
             (r"$\sum_{i=1}^{n}$", PURPLE),
             (r"$\;\alpha_i$", GREEN),
             (r"$\,p_i(u,v\,|\,t)$", BLUE),
             (r"$\,p_i(t)$", GOLD),
             (r"$\,c(d_i,t)$", ORANGE),
             (r"$\,T_i$", GREEN),
             (r"$,\qquad T_i=\prod_{j<i}(1-\alpha_j)$", INK)]
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    texts, widths, cur = [], [], 0.0
    for txt, col in parts:
        t = ax.text(cur, 1.36, txt, ha="left", va="center", fontsize=11.0,
                    color=col, zorder=9)
        bb = t.get_window_extent(renderer=renderer)
        wpix = inv.transform((bb.x1, bb.y0))[0] - inv.transform(
            (bb.x0, bb.y0))[0]
        texts.append(t)
        widths.append(wpix)
        cur += wpix
    start = W / 2.0 - cur / 2.0
    off = 0.0
    for t, wpix in zip(texts, widths):
        t.set_x(start + off)
        off += wpix
    ax.text(W / 2.0, 0.90, "each symbol is tinted to match the stage above "
                           "that produces it", ha="center", va="center",
            fontsize=6.8, color=MUTED, zorder=9)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    ax.text(0.45, 4.70, "4D Gaussian splatting: from the dynamic field to a "
                        "rendered view", ha="left", va="center", fontsize=12.0,
            fontweight="bold", color=INK)
    ax.text(0.45, 4.45, "time-sliced Gaussians are projected, sorted by depth "
                        "and blended front to back in one rasteriser pass",
            ha="left", va="center", fontsize=7.8, color=MUTED)

    gaussian_cloud(ax)
    time_and_colour(ax)
    projection(ax)
    compositing(ax)

    arrow(ax, (A[1], MID), (B[0], MID))
    arrow(ax, (B[1], MID), (C[0], MID))
    arrow(ax, (C[1], MID), (D[0], MID))
    arrow(ax, (D[1], MID), (OUT - 0.60, MID))

    image_tile(ax, OUT, MID, 1.20, 0.95, synth_frame(SEED))
    caption(ax, OUT, MID - 0.60, "Rendered view", r"$\hat{I}(u,v,t)$",
            color=INK)

    rendering_equation(fig, ax)
    savefig(fig, "1078_wm_arch_gaussian_splat.png")


if __name__ == "__main__":
    main()
