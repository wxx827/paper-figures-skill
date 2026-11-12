"""Retrieval-augmented embodied agent with memory write-back.

The episodic memory bank is drawn as a deck of stacked cards; a fan of thin
lines carries the top-k neighbours into the retriever, which conditions the
LLM planner, which emits sub-goals for the low-level policy acting in the
environment.  Two arcs close the loop underneath: a solid one returning the
observation and success signal to the planner, and a dashed one writing the
finished episode back into memory.

The environment thumbnail is synthesised from geometry rather than from blur:
a horizon splits the frame into two brightness zones, capsule and rounded-box
primitives form the arm and the manipulated object, low-amplitude multi-scale
noise supplies texture, and only the background is blurred so the subject
stays sharp.

Geometry is expressed in inches on an equal-aspect canvas so corner radii,
arrow heads and gaps are identical everywhere.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pubstyle import set_pub_style, savefig, lighten

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
SEED = 1485  # original id; figure number is 1080


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


def _blur3(a, sigma):
    return np.dstack([_blur(a[..., c], sigma) for c in range(a.shape[2])])


def _value_noise(rng, ny, nx, octaves=(3, 6, 12, 24),
                 weights=(1.0, .55, .32, .18)):
    out = np.zeros((ny, nx))
    for o, wt in zip(octaves, weights):
        g = rng.random((o, o))
        ty = np.linspace(0, o - 1, ny)
        tx = np.linspace(0, o - 1, nx)
        iy0 = np.floor(ty).astype(int)
        ix0 = np.floor(tx).astype(int)
        iy1 = np.minimum(iy0 + 1, o - 1)
        ix1 = np.minimum(ix0 + 1, o - 1)
        fy, fx = (ty - iy0)[:, None], (tx - ix0)[None, :]
        top = g[iy0][:, ix0] * (1 - fx) + g[iy0][:, ix1] * fx
        bot = g[iy1][:, ix0] * (1 - fx) + g[iy1][:, ix1] * fx
        out += wt * (top * (1 - fy) + bot * fy)
    return (out - out.min()) / max(np.ptp(out), 1e-9)


def _capsule(xx, yy, p0, p1, r):
    bx, by = p1[0] - p0[0], p1[1] - p0[1]
    px, py = xx - p0[0], yy - p0[1]
    t = np.clip((px * bx + py * by) / (bx * bx + by * by + 1e-9), 0.0, 1.0)
    return np.hypot(px - t * bx, py - t * by) - r


def _rounded_box(xx, yy, cx, cy, hw, hh, r):
    qx = np.abs(xx - cx) - (hw - r)
    qy = np.abs(yy - cy) - (hh - r)
    return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
            + np.minimum(np.maximum(qx, qy), 0.0) - r)


def _cover(sdf, soft):
    return np.clip(0.5 - sdf / soft, 0.0, 1.0)


AR = 1.44                      # thumbnail width / height


def robot_scene(seed, ny=124):
    """Table-top manipulation scene: geometry first, texture and depth second."""
    nx = int(round(ny * AR))
    rng = np.random.default_rng(seed)
    yi, xi = np.mgrid[0:ny, 0:nx]
    yy = yi / (ny - 1.0)
    xx = xi / (nx - 1.0) * AR
    soft = 2.4 / ny
    hz = 0.54

    wall = np.array([0.70, 0.745, 0.795])
    table = np.array([0.64, 0.555, 0.445])
    up = 0.80 + 0.30 * (yy / hz)
    dn = 1.06 - 0.34 * np.clip((yy - hz) / (1.0 - hz), 0, 1)
    bg = np.where((yy < hz)[..., None], wall * up[..., None],
                  table * dn[..., None])
    bg *= 1.0 - 0.34 * _cover(np.abs(yy - hz) - 0.005, soft)[..., None]

    shadow = np.exp(-(((xx - 1.06) / 0.17) ** 2 + ((yy - 0.782) / 0.048) ** 2))
    shadow += np.exp(-(((xx - 0.28) / 0.11) ** 2 + ((yy - 0.822) / 0.032) ** 2))
    bg *= (1.0 - 0.32 * np.clip(shadow, 0, 1))[..., None]

    arm = np.minimum.reduce([
        _rounded_box(xx, yy, 0.28, 0.795, 0.085, 0.030, 0.014),
        _capsule(xx, yy, (0.28, 0.79), (0.28, 0.40), 0.050),
        _capsule(xx, yy, (0.28, 0.40), (0.58, 0.28), 0.043),
        _capsule(xx, yy, (0.58, 0.28), (0.80, 0.50), 0.035),
        _capsule(xx, yy, (0.80, 0.50), (0.91, 0.56), 0.019),
        _capsule(xx, yy, (0.80, 0.50), (0.795, 0.64), 0.019)])
    joints = np.minimum.reduce([
        _capsule(xx, yy, (0.28, 0.40), (0.28, 0.40), 0.031),
        _capsule(xx, yy, (0.58, 0.28), (0.58, 0.28), 0.027),
        _capsule(xx, yy, (0.80, 0.50), (0.80, 0.50), 0.022)])
    box = _rounded_box(xx, yy, 1.06, 0.665, 0.105, 0.085, 0.022)
    m_arm, m_box = _cover(arm, soft), _cover(box, soft)
    m_joint = _cover(joints, soft)
    m_lid = m_box * _cover(yy - 0.612, soft)

    shade = (1.24 - 0.44 * yy)[..., None]
    slate = np.array([0.27, 0.31, 0.37])
    crate = np.array([0.60, 0.37, 0.30])
    fg = bg * (1 - m_box[..., None]) + crate * shade * m_box[..., None]
    fg = fg * (1 - m_lid[..., None]) + crate * 1.30 * shade * m_lid[..., None]
    fg = fg * (1 - m_arm[..., None]) + slate * shade * m_arm[..., None]
    fg = fg * (1 - m_joint[..., None]) + slate * 1.55 * shade \
        * m_joint[..., None]

    texture = 1.0 + 0.085 * (_value_noise(rng, ny, nx) - 0.5) * 2.0
    grain = 1.0 + 0.022 * rng.normal(0.0, 1.0, (ny, nx))
    tex = (texture * grain)[..., None]
    fg, bg = fg * tex, bg * tex

    m = _blur(np.clip(m_arm + m_box, 0, 1), 0.9)[..., None]
    out = _blur3(fg, 0.45) * m + _blur3(bg, 2.2) * (1.0 - m)
    r2 = ((xx - AR / 2) / AR) ** 2 + (yy - 0.5) ** 2
    out *= (1.0 - 0.40 * r2)[..., None]
    return np.clip(out, 0.0, 1.0)


def image_tile(ax, x, y, w, h, rgb, ec=RULE, lw=0.8, zorder=4):
    clip = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={R}",
                          facecolor="white", edgecolor="none", zorder=zorder)
    ax.add_patch(clip)
    im = ax.imshow(rgb, extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
                   origin="upper", zorder=zorder + 0.2,
                   interpolation="bilinear")
    im.set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={R}",
                                facecolor="none", edgecolor=ec, linewidth=lw,
                                zorder=zorder + 0.4))


# ---- boxes, arrows, containers ---------------------------------------------
def grad_block(ax, x, y, w, h, label="", color=BLUE, sub=None, fontsize=8.6,
               zorder=3, label_dy=0.10, sub_dy=-0.13, sub_size=6.3):
    tint = lighten(color, 0.80)
    base = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={R}",
                          facecolor="white", edgecolor="none", zorder=zorder)
    ax.add_patch(base)
    cmap = LinearSegmentedColormap.from_list("g", ["white", tint])
    im = ax.imshow(np.linspace(0, 1, 128).reshape(-1, 1),
                   extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
                   origin="lower", cmap=cmap, vmin=0, vmax=1,
                   zorder=zorder + 0.2, interpolation="bilinear")
    im.set_clip_path(base)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={R}",
                                facecolor="none", edgecolor=color,
                                linewidth=LW, zorder=zorder + 0.4))
    if label:
        ax.text(x, y + (label_dy if sub else 0.0), label, ha="center",
                va="center", fontsize=fontsize, fontweight="bold", color=INK,
                zorder=zorder + 1)
    if sub:
        ax.text(x, y + sub_dy, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, zorder=zorder + 1)
    return dict(x=x, y=y, w=w, h=h)


def chip(ax, x, y, w, text, color=BLUE, h=0.30, fontsize=6.5, zorder=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor="white", edgecolor=color,
                                linewidth=0.7, zorder=zorder))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=RULE, zorder=zorder + 1)


def container(ax, x0, x1, y0, y1, color, title, rounding=0.14, zorder=1):
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
    ax.text(x0 + 0.10, y1 + 0.10, title, ha="left", va="bottom", fontsize=8.2,
            fontweight="bold", color=color, zorder=zorder + 1)


def arrow(ax, p0, p1, color=RULE, rad=0.0, lw=1.0, ls="-", ms=12, zorder=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def plate_text(ax, x, y, text, color=MUTED, fontsize=6.8, weight="normal",
               pad=0.26, zorder=9):
    return ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                   color=color, fontweight=weight, zorder=zorder,
                   bbox=dict(boxstyle=f"round,pad={pad}", facecolor="white",
                             edgecolor="none"))


# ---- figure ----------------------------------------------------------------
W, H = 12.4, 4.50
Y0, Y1 = 1.10, 3.51
MAIN = 2.90
ROWS = (2.26, 1.87, 1.46)
MEM = (0.45, 2.85)
RET = (3.20, 5.30)
PLN = (5.65, 7.95)
EXE = (8.30, 10.20)
ENV = (10.55, 12.00)


def memory_bank(ax):
    container(ax, *MEM, Y0, Y1, PURPLE, "Episodic memory")
    cx, cy, cw, ch = 1.65, 2.78, 1.72, 0.86
    for i in (3, 2, 1):
        ax.add_patch(FancyBboxPatch(
            (cx - cw / 2 - 0.06 * i, cy - ch / 2 + 0.06 * i), cw, ch,
            boxstyle=f"round,pad=0,rounding_size={R}", facecolor="white",
            edgecolor=PURPLE, linewidth=0.7, alpha=0.9, zorder=3 + i))
    ax.add_patch(FancyBboxPatch((cx - cw / 2, cy - ch / 2), cw, ch,
                                boxstyle=f"round,pad=0,rounding_size={R}",
                                facecolor=lighten(PURPLE, 0.90),
                                edgecolor=PURPLE, linewidth=LW, zorder=7))
    rows = [("episode #4182", "success"), ("skill: open drawer", "24 steps"),
            ("key: text + image", "d=768")]
    for j, (left, right) in enumerate(rows):
        yy = cy + 0.24 - 0.24 * j
        ax.add_patch(Circle((cx - cw / 2 + 0.15, yy), 0.035, facecolor=PURPLE,
                            edgecolor="none", zorder=8))
        ax.text(cx - cw / 2 + 0.26, yy, left, ha="left", va="center",
                fontsize=6.2, color=INK, zorder=8)
        ax.text(cx + cw / 2 - 0.12, yy, right, ha="right", va="center",
                fontsize=6.0, family="monospace", color=MUTED, zorder=8)
    ax.text(cx, ROWS[0], "128k episodes  ·  written online", ha="center",
            va="center", fontsize=6.6, color=MUTED, zorder=8)
    chip(ax, cx, ROWS[1], cw, "FAISS IVF-PQ  ·  hit rate 78.4 %", color=PURPLE)
    chip(ax, cx, ROWS[2], cw, "+2.1k episodes written per day", color=PURPLE)
    return dict(x=cx, right=cx + cw / 2, y=cy, top=cy + ch / 2,
                bottom=cy - ch / 2)


def retriever(ax):
    container(ax, *RET, Y0, Y1, BLUE, "Retriever")
    grad_block(ax, 4.25, MAIN, 1.78, 0.52, "Top-$k$ ANN search", color=BLUE,
               sub="cosine  ·  $k=5$  ·  3.1 ms", fontsize=8.4, label_dy=0.11,
               sub_dy=-0.14)
    for y, txt in zip(ROWS, ["query = goal + current observation",
                             "cross-encoder re-rank",
                             "returns 5 exemplars + outcomes"]):
        chip(ax, 4.25, y, 1.78, txt, color=BLUE)


def planner(ax):
    container(ax, *PLN, Y0, Y1, GREEN, "Planner")
    grad_block(ax, 6.80, MAIN, 1.98, 0.52, "LLM planner", color=GREEN,
               sub="retrieved episodes in context", fontsize=8.6,
               label_dy=0.11, sub_dy=-0.14)
    for y, txt in zip(ROWS, ["1 · approach the drawer",
                             "2 · grasp the handle",
                             "3 · pull, then release"]):
        chip(ax, 6.80, y, 1.98, txt, color=GREEN)


def executor(ax):
    container(ax, *EXE, Y0, Y1, ORANGE, "Executor")
    grad_block(ax, 9.25, MAIN, 1.62, 0.52, "Low-level policy", color=ORANGE,
               sub="visuomotor  ·  50 Hz", fontsize=8.4, label_dy=0.11,
               sub_dy=-0.14)
    for y, txt in zip(ROWS, ["action $a_t$  ·  7-DoF", "success detector",
                             "safety limiter"]):
        chip(ax, 9.25, y, 1.62, txt, color=ORANGE)


def environment(ax):
    container(ax, *ENV, Y0, Y1, GOLD, "Environment")
    image_tile(ax, 11.275, 2.85, 1.24, 0.86, robot_scene(SEED))
    ax.text(11.275, ROWS[0], "Simulator / real robot", ha="center",
            va="center", fontsize=7.2, fontweight="bold", color=INK, zorder=8)
    chip(ax, 11.275, ROWS[1], 1.24, "32 randomised scenes", color=GOLD)
    chip(ax, 11.275, ROWS[2], 1.24, "$o_{t+1}$, reward, done", color=GOLD)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    ax.text(0.45, 4.11, "Retrieval-augmented embodied agent",
            ha="left", va="center", fontsize=12.0, fontweight="bold",
            color=INK)
    ax.text(0.45, 3.86, "retrieved episodes condition the planner; every "
                        "finished rollout is written back, so the memory grows "
                        "with experience", ha="left", va="center",
            fontsize=7.8, color=MUTED)

    mem = memory_bank(ax)
    retriever(ax)
    planner(ax)
    executor(ax)
    environment(ax)

    # fan of retrieved neighbours
    for yy in np.linspace(mem["bottom"] + 0.13, mem["top"] - 0.09, 5):
        ax.plot([mem["right"] + 0.03, 3.02], [yy, MAIN], color=PURPLE, lw=0.7,
                alpha=0.75, solid_capstyle="round", zorder=5)
    arrow(ax, (3.02, MAIN), (RET[0], MAIN), color=PURPLE, lw=1.1)
    plate_text(ax, 3.02, 3.10, "top-$k$", color=PURPLE, fontsize=6.2,
               pad=0.22)

    arrow(ax, (RET[1], MAIN), (PLN[0], MAIN), lw=1.2)
    arrow(ax, (PLN[1], MAIN), (EXE[0], MAIN), lw=1.2)
    arrow(ax, (EXE[1], MAIN), (ENV[0], MAIN), lw=1.2)

    # closed loops underneath
    arrow(ax, (11.28, Y0), (6.80, Y0), color=GREEN, lw=1.1, rad=-0.11, ms=13)
    plate_text(ax, 9.04, 0.855, r"observation + success signal $\rightarrow$ "
                                r"replan", color=GREEN, fontsize=6.8,
               weight="bold")
    arrow(ax, (11.05, Y0), (mem["x"], Y0), color=PURPLE, lw=1.1, ls=DASH,
          rad=-0.14, ms=13)
    plate_text(ax, 6.35, 0.442, "write back: new episode, action trace and "
                                "outcome label", color=PURPLE, fontsize=6.8,
               weight="bold")

    savefig(fig, "1080_wm_arch_memory_agent.png")


if __name__ == "__main__":
    main()
