"""Vision-language-action model stack with per-connection tensor shapes.

The observation is patchified and pushed through a frozen ViT and an MLP
projector; the projected patches, the tokenised instruction and a set of
action queries form the prompt of a decoder-only LLM; the queried hidden
states are de-tokenised into a 7-DoF end-effector command.  Every hop carries
its tensor shape in small monospace type set on a white plate so the label
never fights the dashed stage borders.

Geometry is expressed in inches on an equal-aspect canvas so corner radii,
arrow heads and gaps are identical everywhere.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
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
SEED = 1482  # original id; figure number is 1077


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
    "scene", ["#3a4560", "#5b7a97", "#9ab6c8", "#dfc9a4", "#f6f0e6"])


def synth_frame(seed, n=88, sigma=0.9, cmap=SCENE):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    field = 0.50 * _value_noise(rng, n) + 0.50 * (1.0 - yy) ** 1.5
    for cx, cy, s, a in ((0.34, 0.62, 0.13, 0.70), (0.68, 0.52, 0.10, 0.45)):
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


def route(ax, pts, color=RULE, lw=1.0, ms=12, zorder=6):
    pts = [tuple(p) for p in pts]
    ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color,
            lw=lw, solid_capstyle="round", solid_joinstyle="round",
            zorder=zorder)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ms=ms, zorder=zorder)


def shape_tag(ax, x, y, text, ha="center", fontsize=6.4, zorder=9):
    """Tensor shape on a white plate so it can sit over any border."""
    return ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
                   family="monospace", color=MUTED, zorder=zorder,
                   bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                             edgecolor="none"))


def block_stack(ax, x, y, w, h, n=4, dx=-0.065, dy=0.065, tint="#f6e0cf",
                ec=ORANGE, zorder=3, **kw):
    for i in range(n - 1, 0, -1):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + i * dx, y - h / 2 + i * dy), w, h,
            boxstyle=f"round,pad=0,rounding_size={R}", facecolor="white",
            edgecolor=ec, linewidth=0.7, alpha=0.85, zorder=zorder + 0.05 * i))
    a = grad_block(ax, x, y, w, h, tint=tint, ec=ec, zorder=zorder + 1, **kw)
    a["back_l"] = x - w / 2 + (n - 1) * dx
    return a


# ---- figure ----------------------------------------------------------------
W, H = 12.4, 4.70
CY, CH = 2.475, 2.35
BOT = CY - CH / 2                      # 1.30
C1 = dict(c=1.50, w=2.10)
C2 = dict(c=4.425, w=2.55)
C3 = dict(c=7.75, w=2.90)
C4 = dict(c=10.90, w=2.20)
LLM_X = 7.62


def observation(ax):
    container(ax, C1["c"], CY, C1["w"], CH, BLUE, "Observation + instruction")
    img = image_tile(ax, 1.50, 2.80, 1.35, 1.00, synth_frame(SEED))
    x0, x1 = 1.50 - 0.675, 1.50 + 0.675
    y0, y1 = 2.80 - 0.50, 2.80 + 0.50
    for gx in np.linspace(x0, x1, 7)[1:-1]:
        ax.plot([gx, gx], [y0, y1], color="white", lw=0.5, alpha=0.75,
                zorder=6)
    for gy in np.linspace(y0, y1, 6)[1:-1]:
        ax.plot([x0, x1], [gy, gy], color="white", lw=0.5, alpha=0.75,
                zorder=6)
    cw, ch = (x1 - x0) / 6, (y1 - y0) / 5
    ax.add_patch(Rectangle((x0 + 2 * cw, y0 + 2 * ch), cw, ch,
                           facecolor="none", edgecolor=ORANGE, lw=1.1,
                           zorder=7))
    ax.text(1.50, 2.22, "RGB observation  ·  patchify 14$\\times$14",
            ha="center", va="top", fontsize=6.6, color=MUTED, zorder=8)
    chip = grad_block(ax, 1.50, 1.72, 1.80, 0.62, "Instruction",
                      tint="#e6dff2", ec=PURPLE,
                      sub='"put the red mug on the shelf"', fontsize=7.4,
                      label_dy=0.13, sub_dy=-0.14, sub_size=5.9)
    return img, chip


def vision_encoder(ax):
    container(ax, C2["c"], CY, C2["w"], CH, ORANGE, "Vision encoder")
    vit = block_stack(ax, C2["c"], 2.85, 1.55, 1.10, n=4, label="ViT-L/14",
                      sub="SigLIP  ·  frozen", fontsize=9.2, label_dy=0.14,
                      sub_dy=-0.16, sub_size=6.4)
    proj = grad_block(ax, C2["c"], 1.75, 1.55, 0.44, "MLP projector",
                      tint="#f6e0cf", ec=ORANGE, fontsize=7.6)
    arrow(ax, (C2["c"], 2.30), (C2["c"], 1.97))
    shape_tag(ax, C2["c"] + 0.14, 2.14, "Bx256x1024", ha="left")
    ax.text(C2["c"], 1.45, "2-layer GELU  ·  1024 $\\to$ 4096", ha="center",
            va="top", fontsize=6.2, color=MUTED, zorder=8)
    return vit, proj


def llm_backbone(ax):
    container(ax, C3["c"], CY, C3["w"], CH, GREEN, "LLM backbone")
    ys = [2.26, 2.70, 3.02, 3.34]
    for i, y in enumerate(ys):
        grad_block(ax, LLM_X, y, 2.00, 0.26, f"Layer {1 if i == 0 else 29 + i}",
                   tint="#d7e7de", ec=GREEN, fontsize=6.4)
    ax.text(LLM_X, 2.48, "$\\vdots$", ha="center", va="center", fontsize=9.0,
            color=MUTED, zorder=8)
    ax.plot([8.76, 8.76], [2.13, 3.47], color=GREEN, lw=0.9, zorder=5)
    for yy in (2.13, 3.47):
        ax.plot([8.70, 8.76], [yy, yy], color=GREEN, lw=0.9, zorder=5)
    ax.text(8.92, 2.80, "$\\times$ 32 layers", ha="center", va="center",
            fontsize=6.4, color=GREEN, rotation=90, zorder=8)

    # prompt = vision patches + language tokens + action queries
    n, cw, gap = 13, 0.175, 0.015
    x0 = LLM_X - (n * (cw + gap) - gap) / 2.0
    cols = [BLUE] * 7 + [PURPLE] * 3 + [GOLD] * 3
    cx = []
    for i, col in enumerate(cols):
        x = x0 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, 1.75), cw, 0.20,
                                    boxstyle="round,pad=0,rounding_size=0.025",
                                    facecolor=col, alpha=0.55,
                                    edgecolor=col, linewidth=0.6, zorder=5))
        cx.append(x + cw / 2)
    for name, col, i in ((r"$\times$256 vision", BLUE, 3),
                         (r"$\times$24 text", PURPLE, 7),
                         (r"$\times$7 queries", GOLD, 11)):
        ax.text(cx[i], 1.60, name, ha="center", va="center", fontsize=6.0,
                color=col, zorder=8)
    arrow(ax, (cx[3], 1.95), (cx[3], 2.13), lw=0.9, ms=11)
    return dict(x0=x0, cx=cx, top=(LLM_X + 1.00, 3.34), bar_y=1.75)


def action_head(ax):
    container(ax, C4["c"], CY, C4["w"], CH, PURPLE, "Action head")
    det = grad_block(ax, C4["c"], 3.15, 1.90, 0.55, "Action de-tokenizer",
                     tint="#e6dff2", ec=PURPLE, sub="256 bins $\\to$ continuous",
                     fontsize=8.0, label_dy=0.12, sub_dy=-0.14, sub_size=6.2)
    arrow(ax, (C4["c"], 2.875), (C4["c"], 2.62))
    shape_tag(ax, C4["c"] + 0.16, 2.74, "Bx7x256", ha="left")

    vals = [0.34, -0.12, 0.21, -0.28, 0.06, 0.41, -0.45]
    names = ["$\\Delta x$", "$\\Delta y$", "$\\Delta z$", "$\\Delta$roll",
             "$\\Delta$pitch", "$\\Delta$yaw", "gripper"]
    cmap = get_cmap("cb_diverging")
    norm = Normalize(-0.6, 0.6)
    cw, gp = 0.25, 0.02
    x0 = C4["c"] - (7 * (cw + gp) - gp) / 2.0
    for i, (v, nm) in enumerate(zip(vals, names)):
        x = x0 + i * (cw + gp)
        rgb = cmap(norm(v))
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        ax.add_patch(Rectangle((x, 2.25), cw, 0.34, facecolor=rgb,
                               edgecolor=RULE, lw=0.6, zorder=5))
        ax.text(x + cw / 2, 2.42, f"{v:+.2f}".replace("0.", "."), ha="center",
                va="center", fontsize=6.0, family="monospace",
                color="white" if lum < 0.55 else INK, zorder=8)
        ax.text(x + cw / 2, 2.17, nm, ha="center", va="top", fontsize=6.0,
                color=RULE, rotation=90, zorder=8)
    ax.text(C4["c"], 1.70, "7-DoF end-effector delta + gripper", ha="center",
            va="top", fontsize=6.8, fontweight="bold", color=INK, zorder=8)
    ax.text(C4["c"], 1.55, "action chunk: 8 steps @ 15 Hz", ha="center",
            va="top", fontsize=6.2, color=MUTED, zorder=8)
    return det


def footnotes(ax):
    ax.add_patch(FancyBboxPatch((0.45, 0.15), W - 0.90, 0.57,
                                boxstyle="round,pad=0,rounding_size=0.09",
                                facecolor="#f5f5f5", edgecolor=LINE,
                                linewidth=0.8, zorder=1))
    notes = [(0.95, "frozen: ViT  ·  trained: projector, LoRA (r=32), head"),
             (5.05, "prompt length 256 + 24 + 7 = 287 tokens"),
             (8.85, "hidden width d = 4096  ·  7.1 B parameters")]
    for x, txt in notes:
        ax.text(x, 0.435, txt, ha="left", va="center", fontsize=6.8,
                color=MUTED, zorder=4)
    for x in (4.80, 8.60):
        ax.plot([x, x], [0.28, 0.59], color=LINE, lw=0.8, zorder=3)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    ax.text(0.45, 4.46, "Vision-language-action policy stack",
            ha="left", va="center", fontsize=12.0, fontweight="bold",
            color=INK)
    ax.text(0.45, 4.20, "patchified pixels and instruction tokens share one "
                        "decoder-only prompt; action queries are read out and "
                        "de-tokenised", ha="left", va="center", fontsize=7.8,
            color=MUTED)

    img, chip = observation(ax)
    vit, proj = vision_encoder(ax)
    llm = llm_backbone(ax)
    det = action_head(ax)

    arrow(ax, (img["r"][0], img["y"]), (vit["back_l"], 2.85))
    shape_tag(ax, 2.82, 3.00, "Bx3x224x224")

    arrow(ax, (proj["r"][0], proj["y"]), (llm["x0"] - 0.06, llm["bar_y"] + 0.10))
    shape_tag(ax, 5.80, 1.98, "Bx256x4096")

    route(ax, [(chip["r"][0], chip["y"]), (2.62, chip["y"]), (2.62, 1.02),
               (llm["cx"][9], 1.02), (llm["cx"][9], llm["bar_y"] - 0.02)],
          color=PURPLE, lw=0.9, ms=11)
    shape_tag(ax, 4.60, 1.02, "Bx24x4096")

    arrow(ax, (llm["top"][0], llm["top"][1]), (C4["c"] - 0.95, 3.15))
    shape_tag(ax, 9.35, 3.36, "Bx7x4096")

    footnotes(ax)
    savefig(fig, "1077_wm_arch_vla_stack.png")


if __name__ == "__main__":
    main()
