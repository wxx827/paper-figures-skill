"""Recurrent state-space world model unrolled over three timesteps.

Each timestep sits in its own rounded dashed container: the observation is
encoded into the posterior stochastic state ``z_t`` while the deterministic
state ``h_t`` carries the prior; a decoder reconstructs the frame and a small
head predicts the reward.  Containers are chained by the recurrent transition
arrow with the action injected on it.  The bottom strip names the two training
objectives (latent KL, reconstruction / reward likelihood).

Geometry is expressed in inches on an equal-aspect canvas so corner radii,
arrow heads and gaps are identical everywhere.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pubstyle import set_pub_style, savefig

# ---- shared world-model diagram vocabulary ---------------------------------
BLUE = "#3b5b92"
ORANGE = "#c0603f"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
RED = "#b23a48"
INK = "#1a1a1a"
RULE = "#333333"
MUTED = "#777777"
FAINT = "#8a8a8a"
LINE = "#d5d5d5"

R = 0.06                      # uniform corner radius (inches)
LW = 0.9                      # uniform box stroke width
DASH = (0, (6, 3))
SEED = 1480  # original id; figure number is 1075


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
        f = (t - i0)
        top = g[i0][:, i0] * (1 - f)[None, :] + g[i0][:, i1] * f[None, :]
        bot = g[i1][:, i0] * (1 - f)[None, :] + g[i1][:, i1] * f[None, :]
        out += wt * (top * (1 - f)[:, None] + bot * f[:, None])
    return (out - out.min()) / max(np.ptp(out), 1e-9)


SCENE = LinearSegmentedColormap.from_list(
    "scene", ["#2c3a52", "#4a6b8a", "#8fb0c4", "#d9c9a8", "#f4ede0"])


def synth_frame(seed, n=72, t=0.0, sigma=1.0, cmap=SCENE):
    """Low-saturation synthetic 'video frame': multi-scale noise + blob."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    field = 0.52 * _value_noise(rng, n) + 0.48 * (1.0 - yy) ** 1.4
    cx, cy = 0.30 + 0.34 * t, 0.60
    field += 0.62 * np.exp(-(((xx - cx) / 0.15) ** 2 + ((yy - cy) / 0.20) ** 2))
    field = _blur(field, sigma)
    field = (field - field.min()) / max(np.ptp(field), 1e-9)
    rgb = cmap(0.08 + 0.84 * field)[..., :3]
    return 0.80 * rgb + 0.20


def image_tile(ax, x, y, w, h, rgb, ec=RULE, lw=0.8, zorder=4):
    """Frame thumbnail clipped to a rounded rectangle."""
    clip = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={R}",
                          facecolor="white", edgecolor="none", zorder=zorder)
    ax.add_patch(clip)
    im = ax.imshow(rgb, extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
                   origin="upper", zorder=zorder + 0.2, interpolation="bilinear")
    im.set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={R}",
                                facecolor="none", edgecolor=ec, linewidth=lw,
                                zorder=zorder + 0.4))
    return anchors(x, y, w, h)


# ---- boxes, arrows, containers ---------------------------------------------
def grad_block(ax, x, y, w, h, label, tint="#f6e6c8", ec=RULE, sub=None,
               fontsize=9.0, zorder=3, lw=LW, text_color=INK,
               label_dy=0.10, sub_dy=-0.13, sub_size=6.3):
    """Module box with a vertical tint->white gradient fill."""
    base = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={R}",
                          facecolor="white", edgecolor="none", zorder=zorder)
    ax.add_patch(base)
    cmap = LinearSegmentedColormap.from_list("g", ["white", tint])
    im = ax.imshow(np.linspace(0, 1, 160).reshape(-1, 1),
                   extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
                   origin="lower", cmap=cmap, vmin=0, vmax=1,
                   zorder=zorder + 0.2, interpolation="bilinear")
    im.set_clip_path(base)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={R}",
                                facecolor="none", edgecolor=ec, linewidth=lw,
                                zorder=zorder + 0.4))
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
        ax.text(x, y + h / 2 + 0.12, title, ha="center", va="bottom",
                fontsize=8.4, fontweight="bold", color=color, zorder=zorder + 1)
    return anchors(x, y, w, h)


def arrow(ax, p0, p1, color=RULE, rad=0.0, lw=1.0, ls="-", ms=12, zorder=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def tag(ax, x, y, text, color=FAINT, fontsize=6.4, ha="center", va="center",
        mono=True, zorder=8):
    kw = dict(family="monospace") if mono else {}
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
                   zorder=zorder, **kw)


# ---- figure ----------------------------------------------------------------
W, H = 12.4, 5.55
CX = [2.20, 6.25, 10.30]
CW, CH = 3.36, 3.86
CY = 2.82
STEPS = [("Step t", BLUE), ("Step t+1", ORANGE), ("Step t+2", GREEN)]
SUBS = ["t", "t+1", "t+2"]


def draw_step(ax, cx, k, color):
    """One unrolled timestep; returns the h / z anchors for cross-step wiring."""
    y = CY
    s = SUBS[k]
    container(ax, cx, y, CW, CH, color, STEPS[k][0])

    h = grad_block(ax, cx, y + 1.18, 1.30, 0.54, f"$h_{{{s}}}$",
                   tint="#dbe4f2", ec=BLUE, fontsize=9.8, sub="deterministic")
    z = grad_block(ax, cx, y + 0.28, 1.16, 0.50, f"$z_{{{s}}}$",
                   tint="#f6e0cf", ec=ORANGE, fontsize=9.8, sub="stochastic")
    enc = grad_block(ax, cx - 1.10, y - 0.62, 0.88, 0.38, "Encoder",
                     tint="#dbe4f2", ec=BLUE, fontsize=7.6)
    dec = grad_block(ax, cx + 1.10, y - 0.62, 0.88, 0.38, "Decoder",
                     tint="#d7e7de", ec=GREEN, fontsize=7.6)
    rew = grad_block(ax, cx, y - 0.62, 0.92, 0.38, "Reward", tint="#f4e6c2",
                     ec=GOLD, fontsize=7.6)

    obs = image_tile(ax, cx - 1.10, y - 1.30, 0.82, 0.60,
                     synth_frame(SEED + 7 * k, t=0.10 + 0.35 * k, sigma=0.8))
    rec = image_tile(ax, cx + 1.10, y - 1.30, 0.82, 0.60,
                     synth_frame(SEED + 7 * k, t=0.10 + 0.35 * k, sigma=1.7))
    ax.text(obs["x"], obs["y"] - 0.30 - 0.11, f"$o_{{{s}}}$", ha="center",
            va="top", fontsize=8.0, color=INK, zorder=8)
    ax.text(rec["x"], rec["y"] - 0.30 - 0.11, rf"$\hat{{o}}_{{{s}}}$",
            ha="center", va="top", fontsize=8.0, color=INK, zorder=8)
    ax.text(cx, rew["y"] - 0.19 - 0.20, rf"$\hat{{r}}_{{{s}}}$", ha="center",
            va="center", fontsize=8.0, color=GOLD, zorder=8)

    # wiring inside the step
    arrow(ax, (obs["x"], obs["y"] + 0.30), (enc["x"], enc["y"] - 0.19))
    arrow(ax, (enc["x"] + 0.20, enc["y"] + 0.19), (cx - 0.42, z["y"] - 0.25))
    arrow(ax, (cx - 0.52, h["y"] - 0.27), (cx - 0.52, z["y"] + 0.25))
    arrow(ax, (cx + 0.45, h["y"] - 0.27), (dec["x"] - 0.12, dec["y"] + 0.19))
    arrow(ax, (cx + 0.58, z["y"] - 0.16), (dec["x"] + 0.12, dec["y"] + 0.19))
    arrow(ax, (cx, z["y"] - 0.25), (rew["x"], rew["y"] + 0.19))
    arrow(ax, (dec["x"], dec["y"] - 0.19), (rec["x"], rec["y"] + 0.30))

    tag(ax, cx - 0.60, (h["y"] - 0.27 + z["y"] + 0.25) / 2, "prior",
        color=BLUE, fontsize=6.4, ha="right", mono=False)
    tag(ax, cx - 0.74, enc["y"] + 0.46, "posterior", color=ORANGE,
        fontsize=6.4, ha="right", mono=False)
    return h, z


def loss_strip(ax):
    ax.add_patch(FancyBboxPatch((0.45, 0.14), W - 0.90, 0.66,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="#f5f5f5", edgecolor=LINE,
                                linewidth=0.8, zorder=1))
    items = [
        (RED, DASH, r"$\mathcal{L}_{\mathrm{KL}}=\beta\,"
                    r"\mathrm{KL}[\,q(z_t\,|\,h_t,o_t)\;\|\;p(z_t\,|\,h_t)\,]$",
         "latent prior matching"),
        (GREEN, "-", r"$\mathcal{L}_{\mathrm{rec}}=\|\hat{o}_t-o_t\|^2"
                     r"-\log p(r_t\,|\,h_t,z_t)$",
         "frame reconstruction + reward likelihood"),
    ]
    for i, (col, ls, expr, note) in enumerate(items):
        x0 = 0.95 + i * 5.75
        ax.plot([x0, x0 + 0.42], [0.47, 0.47], color=col, lw=1.6, ls=ls,
                solid_capstyle="butt", zorder=4)
        ax.text(x0 + 0.56, 0.55, expr, ha="left", va="center", fontsize=8.2,
                color=INK, zorder=4)
        ax.text(x0 + 0.56, 0.30, note, ha="left", va="center", fontsize=6.8,
                color=MUTED, zorder=4)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)

    ax.text(0.45, 5.30, "Recurrent state-space world model, unrolled",
            ha="left", va="center", fontsize=12.0, fontweight="bold",
            color=INK)
    ax.text(0.45, 5.04, "posterior from observations, prior from the "
                        "deterministic path; both decode frames and rewards",
            ha="left", va="center", fontsize=7.8, color=MUTED)
    ax.text(11.95, 5.28, r"$h_{t+1}=f_\theta(h_t,\,z_t,\,a_t)$", ha="right",
            va="center", fontsize=9.0, color=BLUE)

    states = [draw_step(ax, cx, k, STEPS[k][1]) for k, cx in enumerate(CX)]

    # recurrent transition between steps, with the action injected on it
    for k in range(2):
        h0, z0 = states[k]
        h1, _ = states[k + 1]
        xa = (h0["r"][0] + h1["l"][0]) / 2.0
        arrow(ax, (h0["r"][0], h0["y"]), (xa - 0.19, h0["y"]), color=BLUE,
              lw=1.2)
        arrow(ax, (xa + 0.19, h0["y"]), (h1["l"][0], h1["y"]), color=BLUE,
              lw=1.2)
        ax.add_patch(Circle((xa, h0["y"]), 0.19, facecolor="white",
                            edgecolor=GOLD, linewidth=1.0, zorder=7))
        ax.text(xa, h0["y"] + 0.005, f"$a_{{{SUBS[k]}}}$", ha="center",
                va="center", fontsize=7.2, color=GOLD, zorder=8)
        tag(ax, xa, h0["y"] + 0.33, "action", color=GOLD, fontsize=6.2,
            mono=False)
        # stochastic state also feeds the next transition
        arrow(ax, (z0["x"] + 0.58, z0["y"] + 0.14),
              (h1["l"][0], h1["y"] - 0.13), color=ORANGE, lw=0.9, ls=DASH,
              rad=-0.16, ms=11)

    loss_strip(ax)
    savefig(fig, "1075_wm_arch_latent_dynamics.png")


if __name__ == "__main__":
    main()
