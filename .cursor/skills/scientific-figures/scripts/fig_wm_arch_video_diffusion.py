"""Two-stage conditional video diffusion pipeline.

A denoising-schedule band runs across the top: seven latent thumbnails fade
from pure noise to a clean frame over a solver axis.  Below it the pipeline
reads left to right -- a conditioning container (text / action / depth) whose
branches converge on the stacked DiT blocks of stage one, then a temporal
refiner plus VAE decoder in stage two, and finally the generated clip as a
film strip.

Geometry is expressed in inches on an equal-aspect canvas so corner radii,
arrow heads and gaps are identical everywhere.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pubstyle import set_pub_style, savefig

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
DARK = "#3a3a3a"

R = 0.06
LW = 0.9
DASH = (0, (6, 3))
SEED = 1481  # original id; figure number is 1076


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def anchors(x, y, w, h):
    return dict(x=x, y=y, w=w, h=h, c=(x, y), l=(x - w / 2, y),
                r=(x + w / 2, y), t=(x, y + h / 2), b=(x, y - h / 2))


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
    "scene", ["#33405a", "#4f7091", "#93b3c6", "#dcc9a6", "#f5efe4"])


def synth_frame(seed, n=72, t=0.0, sigma=0.9, cmap=SCENE):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    field = 0.50 * _value_noise(rng, n) + 0.50 * (1.0 - yy) ** 1.4
    cx, cy = 0.28 + 0.36 * t, 0.60
    field += 0.66 * np.exp(-(((xx - cx) / 0.15) ** 2 + ((yy - cy) / 0.20) ** 2))
    field = _blur(field, sigma)
    field = (field - field.min()) / max(np.ptp(field), 1e-9)
    return 0.82 * cmap(0.08 + 0.84 * field)[..., :3] + 0.18


def noisy_frame(seed, alpha, n=72):
    """Blend a clean frame with white noise; ``alpha`` = 1 is fully clean."""
    rng = np.random.default_rng(seed + 991)
    grain = rng.normal(0.52, 0.20, (n, n, 3))
    grain = np.clip(_blur(grain.mean(axis=2), 0.5)[..., None]
                    * 0.55 + grain * 0.45, 0, 1)
    clean = synth_frame(seed, n=n, t=0.45, sigma=0.9)
    a = float(np.clip(alpha, 0, 1)) ** 1.3
    return np.clip(a * clean + (1 - a) * grain, 0, 1)


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
    return anchors(x, y, w, h)


# ---- boxes, arrows, containers ---------------------------------------------
def grad_block(ax, x, y, w, h, label="", tint="#f6e6c8", ec=RULE, sub=None,
               fontsize=9.0, zorder=3, lw=LW, text_color=INK, label_dy=0.10,
               sub_dy=-0.13, sub_size=6.3, rounding=R):
    base = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={rounding}",
                          facecolor="white", edgecolor="none", zorder=zorder)
    ax.add_patch(base)
    cmap = LinearSegmentedColormap.from_list("g", ["white", tint])
    im = ax.imshow(np.linspace(0, 1, 160).reshape(-1, 1),
                   extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
                   origin="lower", cmap=cmap, vmin=0, vmax=1,
                   zorder=zorder + 0.2, interpolation="bilinear")
    im.set_clip_path(base)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor="none", edgecolor=ec, linewidth=lw,
                                zorder=zorder + 0.4))
    if label:
        ax.text(x, y + (label_dy if sub else 0.0), label, ha="center",
                va="center", fontsize=fontsize, fontweight="bold",
                color=text_color, zorder=zorder + 1)
    if sub:
        ax.text(x, y + sub_dy, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, zorder=zorder + 1)
    return anchors(x, y, w, h)


def container(ax, x, y, w, h, color, title=None, rounding=0.14, zorder=1):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=color,
        alpha=0.04, edgecolor="none", zorder=zorder))
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor="none",
        edgecolor=color, linewidth=1.0, linestyle=DASH, zorder=zorder + 0.5))
    if title:
        ax.text(x - w / 2 + 0.10, y + h / 2 + 0.10, title, ha="left",
                va="bottom", fontsize=8.2, fontweight="bold", color=color,
                zorder=zorder + 1)
    return anchors(x, y, w, h)


def arrow(ax, p0, p1, color=RULE, rad=0.0, lw=1.0, ls="-", ms=12, zorder=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def tag(ax, x, y, text, color=FAINT, fontsize=6.4, ha="center", va="center",
        mono=True, zorder=8, **kw):
    if mono:
        kw["family"] = "monospace"
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
                   zorder=zorder, **kw)


def block_stack(ax, x, y, w, h, n=4, dx=-0.065, dy=0.065, tint="#f6e0cf",
                ec=ORANGE, zorder=3):
    """Depth-stacked module: back copies offset up-left, front one detailed."""
    for i in range(n - 1, 0, -1):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + i * dx, y - h / 2 + i * dy), w, h,
            boxstyle=f"round,pad=0,rounding_size={R}", facecolor="white",
            edgecolor=ec, linewidth=0.7, alpha=0.85, zorder=zorder + 0.05 * i))
    return grad_block(ax, x, y, w, h, tint=tint, ec=ec, zorder=zorder + 1)


def film_strip(ax, x0, x1, yc, h, frames, zorder=4):
    """Black film strip: rounded bar, two perforation rows, inset frames."""
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h,
                                boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=DARK, edgecolor=RULE, linewidth=0.8,
                                zorder=zorder))
    step, s = 0.185, 0.082
    xs = np.arange(x0 + 0.075, x1 - 0.075 - s, step)
    for xx in xs:
        for yy in (yc + h / 2 - 0.145, yc - h / 2 + 0.063):
            ax.add_patch(Rectangle((xx, yy), s, s, facecolor="white",
                                   edgecolor="none", zorder=zorder + 1))
    n = len(frames)
    inner = (x1 - x0) - 0.24
    fw = (inner - 0.05 * (n - 1)) / n
    for i, rgb in enumerate(frames):
        cx = x0 + 0.12 + fw / 2 + i * (fw + 0.05)
        image_tile(ax, cx, yc, fw, h - 0.56, rgb, ec="#dddddd", lw=0.5,
                   zorder=zorder + 2, rounding=0.02)
    return anchors((x0 + x1) / 2, yc, x1 - x0, h)


# ---- figure ----------------------------------------------------------------
W, H = 12.4, 5.5
CY, CH = 2.28, 2.36
TOP = CY + CH / 2                     # 3.46
COND = dict(c=1.35, w=1.80)
S1 = dict(c=4.51, w=4.08)
S2 = dict(c=8.235, w=2.93)
STRIP = (9.90, 12.00)


def schedule_band(ax):
    xs = np.linspace(2.78, 9.42, 7)
    alphas = np.linspace(0.0, 1.0, 7)
    for x, a in zip(xs, alphas):
        image_tile(ax, x, 4.36, 0.62, 0.62, noisy_frame(SEED, a))
    ax.annotate("", xy=(9.76, 3.94), xytext=(2.44, 3.94),
                arrowprops=dict(arrowstyle="-|>", color=RULE, lw=0.9,
                                mutation_scale=12, shrinkA=0, shrinkB=0),
                zorder=6)
    for x in xs:
        ax.plot([x, x], [3.94, 4.02], color=FAINT, lw=0.7, zorder=6)
    tag(ax, 2.44, 3.86, "$t=T$  pure noise", color=MUTED, fontsize=6.6,
        ha="left", va="top", mono=False)
    tag(ax, 9.76, 3.86, "$t=0$  clean latents", color=MUTED, fontsize=6.6,
        ha="right", va="top", mono=False)
    ax.text(2.30, 4.47, "Denoising", ha="right", va="center", fontsize=7.8,
            fontweight="bold", color=INK)
    ax.text(2.30, 4.25, "schedule", ha="right", va="center", fontsize=7.8,
            fontweight="bold", color=INK)


def conditions(ax):
    container(ax, COND["c"], CY, COND["w"], CH, BLUE, "Conditions")
    rows = [("Text prompt", '"a robot lifts the red mug"', PURPLE, 2.92),
            ("Action  $a_{1:T}$", "7-DoF end-effector deltas", GOLD, 2.28),
            ("Depth / camera", "$K,\\;[R|t]$ per frame", GREEN, 1.64)]
    out = []
    for label, sub, col, y in rows:
        b = grad_block(ax, COND["c"], y, 1.52, 0.48, label, tint="#e8e8f0",
                       ec=col, sub=sub, fontsize=7.4, label_dy=0.09,
                       sub_dy=-0.12, sub_size=5.8)
        ax.add_patch(Rectangle((COND["c"] - 0.76, y - 0.24), 0.055, 0.48,
                               facecolor=col, edgecolor="none", zorder=5))
        out.append(b)
    return out


def stage_one(ax, cond):
    container(ax, S1["c"], CY, S1["w"], CH, ORANGE,
              "Stage 1 · base latent diffusion")
    zt = image_tile(ax, 3.30, 3.00, 0.76, 0.60, noisy_frame(SEED, 0.05))
    tag(ax, 3.30, 2.64, r"$z_T\sim\mathcal{N}(0,I)$", color=INK, fontsize=7.2,
        mono=False, va="top")

    st = block_stack(ax, 4.75, 2.40, 1.36, 1.10)
    ax.text(4.75, 2.79, "DiT block", ha="center", va="center", fontsize=8.4,
            fontweight="bold", color=INK, zorder=8)
    for i, name in enumerate(["Self-attention", "Cross-attention", "MLP"]):
        yy = 2.53 - 0.215 * i
        ax.add_patch(FancyBboxPatch((4.75 - 0.58, yy - 0.085), 1.16, 0.17,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor="white", edgecolor=ORANGE,
                                    linewidth=0.6, alpha=0.95, zorder=7))
        ax.text(4.75, yy, name, ha="center", va="center", fontsize=6.2,
                color=RULE, zorder=8)
    z0 = image_tile(ax, 6.05, 2.40, 0.66, 0.60, noisy_frame(SEED, 1.0))
    tag(ax, 6.05, 2.04, r"$\hat{z}_0$", color=INK, fontsize=7.6, mono=False,
        va="top")

    arrow(ax, (3.62, 2.84), (4.24, 2.62))
    arrow(ax, (5.45, 2.40), (5.70, 2.40))
    for b, xt in zip(cond, (4.28, 4.53, 4.78)):
        arrow(ax, (b["x"] + 0.76, b["y"]), (xt, 1.86), color=BLUE, lw=0.9,
              ms=11, rad=0.06)
    tag(ax, 2.62, 1.50, "cross-attention + AdaLN-Zero conditioning",
        color=BLUE, fontsize=6.3, ha="left", mono=False)
    tag(ax, 2.62, 1.32, "DiT $\\times$ 28 layers  ·  1.4 B parameters",
        color=MUTED, fontsize=6.3, ha="left", mono=False)
    return z0


def stage_two(ax, z0):
    container(ax, S2["c"], CY, S2["w"], CH, GREEN,
              "Stage 2 · refine + decode")
    ref = block_stack(ax, 7.75, 2.45, 1.10, 0.95, n=3, tint="#d7e7de",
                      ec=GREEN)
    ax.text(7.75, 2.58, "Temporal", ha="center", va="center", fontsize=7.8,
            fontweight="bold", color=INK, zorder=8)
    ax.text(7.75, 2.40, "refiner", ha="center", va="center", fontsize=7.8,
            fontweight="bold", color=INK, zorder=8)
    tag(ax, 7.75, 2.20, "$\\times$ 8 layers", color=MUTED, fontsize=6.2,
        mono=False, zorder=8)
    dec = grad_block(ax, 9.10, 2.45, 1.10, 0.95, "VAE", tint="#dbe4f2",
                     ec=BLUE, sub="decoder", fontsize=8.4, label_dy=0.11,
                     sub_dy=-0.09, sub_size=7.4)
    tag(ax, 7.75, 1.72, "4$\\times$ temporal upsample", color=MUTED,
        fontsize=6.4, mono=False)
    tag(ax, 9.10, 1.72, "8$\\times$ spatial upsample", color=MUTED,
        fontsize=6.4, mono=False)

    arrow(ax, (z0["x"] + 0.33, z0["y"]), (7.20, 2.45))
    tag(ax, 6.70, 2.62, "16x40x64", color=FAINT, fontsize=6.0)
    arrow(ax, (8.30, 2.45), (8.55, 2.45))
    return ref, dec


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    ax.text(0.45, 5.24, "Two-stage conditional video diffusion world model",
            ha="left", va="center", fontsize=12.0, fontweight="bold",
            color=INK)
    ax.text(0.45, 4.98, "a conditioned DiT denoises the base latents; a "
                        "temporal refiner and VAE decoder emit the clip",
            ha="left", va="center", fontsize=7.8, color=MUTED)

    schedule_band(ax)
    cond = conditions(ax)
    z0 = stage_one(ax, cond)
    _, dec = stage_two(ax, z0)

    frames = [synth_frame(SEED, t=0.10 + 0.26 * i, sigma=0.8)
              for i in range(4)]
    strip = film_strip(ax, STRIP[0], STRIP[1], 2.45, 1.05, frames)
    arrow(ax, (dec["x"] + 0.55, dec["y"]), (STRIP[0], 2.45))
    ax.text(strip["x"], 1.79, r"Generated video  $\hat{V}$", ha="center",
            va="top", fontsize=8.2, fontweight="bold", color=INK)
    tag(ax, strip["x"], 1.60, "49 frames @ 24 fps  ·  480x720", color=MUTED,
        fontsize=6.4, va="top", mono=False)

    # sampler footnote strip
    ax.add_patch(FancyBboxPatch((0.45, 0.20), W - 0.90, 0.62,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#f5f5f5", edgecolor=LINE,
                                linewidth=0.8, zorder=1))
    ax.text(0.95, 0.51, r"$\mathcal{L}=\mathbb{E}_{t,\epsilon}\;"
                        r"\|\,v_\theta(z_t,t,c)-v_t\,\|^2$", ha="left",
            va="center", fontsize=8.4, color=INK, zorder=4)
    ax.text(3.10, 0.51, "v-prediction objective", ha="left", va="center",
            fontsize=6.8, color=MUTED, zorder=4)
    ax.text(6.05, 0.51, "sampler: DPM-Solver++ (2M), 50 steps", ha="left",
            va="center", fontsize=6.8, color=MUTED, zorder=4)
    ax.text(9.15, 0.51, "classifier-free guidance $w=7.5$", ha="left",
            va="center", fontsize=6.8, color=MUTED, zorder=4)
    for x in (2.95, 5.90, 9.00):
        ax.plot([x, x], [0.34, 0.68], color=LINE, lw=0.8, zorder=3)

    savefig(fig, "1076_wm_arch_video_diffusion.png")


if __name__ == "__main__":
    main()
