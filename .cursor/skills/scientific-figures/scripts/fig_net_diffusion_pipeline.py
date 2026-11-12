"""Denoising-diffusion pipeline: forward corruption over reverse sampling.

Both rows share the same timestep columns, so a vertical slice is one noise
level: the top row walks x_0 to x_T under q, the bottom row walks back under
p_theta with arrows reversed, and the wrap-around connector on the right
marks the pure-noise handover.  The shared denoiser is drawn once in the
band between the rows and pinned to one reverse step by a dashed callout.
The right column holds the noise schedule and the sampling recipe.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from pubstyle import set_pub_style, PALETTES, savefig

ENC = "#3b5b92"
DEC = "#2e7d5b"
ATT = "#c9a227"
LOSS = "#b23a48"
DATA = "#8c8c8c"
AUX = "#8e6bbf"
INK = "#24303c"
MUTED = "#6f7783"
SOFT = "#f2f4f7"
LINE = "#dde2e8"

R = 0.055
LW = 0.9
DASH = (0, (4, 2.6))


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


def block(ax, x, y, w, h, label="", color=ENC, sub=None, ec="white",
          text_color="white", fontsize=8, sub_color=None, lw=LW, ls="-",
          alpha=1.0, zorder=3, rounding=R, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=color,
        edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder))
    if label:
        ax.text(x, y + (0.105 if sub else 0.0), label, ha="center",
                va="center", color=text_color, fontsize=fontsize,
                fontweight=weight, zorder=zorder + 2)
    if sub:
        ax.text(x, y - 0.115, sub, ha="center", va="center",
                color=sub_color or text_color, fontsize=6.5, alpha=0.92,
                zorder=zorder + 2)
    return anchors(x, y, w, h)


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.2, ls="-", ms=12, zorder=6,
          alpha=1.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def route(ax, pts, color=INK, lw=1.2, ls="-", ms=12, zorder=6):
    pts = [tuple(p) for p in pts]
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color, lw=lw, ls=ls, solid_capstyle="round",
                zorder=zorder)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=ms,
          zorder=zorder)


def skip_connection(ax, p0, p1, rad=-0.18, color=MUTED, lw=1.0, ms=11,
                    zorder=5):
    arrow(ax, p0, p1, color=color, rad=rad, lw=lw, ls=DASH, ms=ms,
          zorder=zorder)


def tensor_label(ax, x, y, text, color=MUTED, fontsize=6.5, ha="center",
                 va="center", **kw):
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=fontsize,
                   **kw)


def panel(ax, x, y, w, h, fc=SOFT, ec=LINE, lw=0.8, ls="-", zorder=1,
          rounding=0.08):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))
    return anchors(x, y, w, h)


def titles(ax, x, y, main, sub=None):
    ax.text(x, y, main, fontsize=13.5, fontweight="bold", color=INK,
            ha="left", va="center")
    if sub:
        ax.text(x, y - 0.32, sub, fontsize=8.6, color=MUTED, ha="left",
                va="center")


# ---- figure ----------------------------------------------------------------
W, H = 13.4, 7.7
COLS = [1.05, 2.65, 4.25, 5.85, 7.45, 9.05]
TILE = 1.12
Y_FWD, Y_REV = 5.85, 2.54
STEPS = [0, 100, 300, 600, 850, 1000]
ABAR = [1.0, 0.90, 0.62, 0.29, 0.065, 0.004]


def base_image(n=72):
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    img = np.empty((n, n, 3))
    img[..., 0] = 0.42 + 0.42 * yy
    img[..., 1] = 0.60 + 0.30 * yy
    img[..., 2] = 0.88 - 0.06 * yy
    d = np.hypot(xx - 0.73, yy - 0.24)
    sun = np.exp(-(d / 0.11) ** 2)
    img += sun[..., None] * np.array([0.52, 0.34, -0.28])
    xr = np.linspace(0.0, 1.0, n)
    far = 0.55 + 0.09 * np.sin(6.5 * xr + 0.4) + 0.04 * np.sin(13.0 * xr)
    near = 0.72 + 0.10 * np.sin(4.0 * xr + 2.1)
    m = yy > far[None, :]
    img[m] = np.array([0.32, 0.44, 0.40])
    m = yy > near[None, :]
    img[m] = np.array([0.17, 0.30, 0.24])
    lake = yy > 0.86
    img[lake] = np.array([0.24, 0.40, 0.52])
    return np.clip(img, 0, 1)


def noisy(img, abar, rng):
    x0 = img * 2.0 - 1.0
    eps = rng.standard_normal(img.shape)
    xt = np.sqrt(abar) * x0 + np.sqrt(max(1.0 - abar, 0.0)) * eps
    return np.clip((xt + 1.0) / 2.0, 0, 1)


def tile(fig, x, y, s, img, edge=INK, lw=0.9):
    ax = fig.add_axes([(x - s / 2) / W, (y - s / 2) / H, s / W, s / H])
    ax.imshow(img, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(lw)
        sp.set_edgecolor(edge)
    return ax


def rows(fig, ax):
    img = base_image()
    rf = np.random.default_rng(7)
    rr = np.random.default_rng(23)
    for k, (x, ab) in enumerate(zip(COLS, ABAR)):
        tile(fig, x, Y_FWD, TILE, noisy(img, ab, rf), edge="#4a5560")
        tile(fig, x, Y_REV, TILE, noisy(img, ab, rr), edge="#4a5560")
        lab = "$x_0$" if k == 0 else ("$x_T$" if k == len(COLS) - 1
                                      else f"$x_{{{STEPS[k]}}}$")
        ax.text(x, Y_FWD + TILE / 2 + 0.16, lab, ha="center", va="center",
                fontsize=8.4, color=INK)
        ax.text(x, Y_REV - TILE / 2 - 0.18, lab, ha="center", va="center",
                fontsize=8.4, color=INK)

    for k in range(len(COLS) - 1):
        x0 = COLS[k] + TILE / 2
        x1 = COLS[k + 1] - TILE / 2
        arrow(ax, (x0 + 0.05, Y_FWD), (x1 - 0.05, Y_FWD), color=LOSS, lw=1.2)
        ax.text((x0 + x1) / 2, Y_FWD + 0.19, "$q$", ha="center", va="center",
                fontsize=7.4, color=LOSS)
        arrow(ax, (x1 - 0.05, Y_REV), (x0 + 0.05, Y_REV), color=ENC, lw=1.2)
        ax.text((x0 + x1) / 2, Y_REV + 0.19, r"$p_\theta$", ha="center",
                va="center", fontsize=7.4, color=ENC)

    ax.text(0.42, 6.82, r"Forward diffusion  $q(x_t|x_{t-1})$  ·  fixed "
            "Gaussian corruption", fontsize=9.0, fontweight="bold",
            color=LOSS, ha="left", va="center")
    ax.text(0.42, 3.26, r"Reverse sampling  $p_\theta(x_{t-1}|x_t)$",
            fontsize=9.0, fontweight="bold", color=ENC, ha="left",
            va="center")

    xr = COLS[-1] + TILE / 2 + 0.31
    route(ax, [(COLS[-1] + TILE / 2 + 0.05, Y_FWD), (xr, Y_FWD),
               (xr, Y_REV), (COLS[-1] + TILE / 2 + 0.05, Y_REV)],
          color=MUTED, lw=1.1, ms=11)
    ax.text(xr + 0.13, (Y_FWD + Y_REV) / 2, r"$x_T \sim \mathcal{N}(0, I)$",
            rotation=90, ha="center", va="center", fontsize=7.0, color=MUTED)


def denoiser(ax):
    panel(ax, 5.10, 4.19, 4.90, 1.62, fc="#fdf6ee", ec=ATT, ls=DASH, lw=1.2)
    ax.text(2.87, 4.80, r"Shared denoiser  $\epsilon_\theta(x_t, t)$",
            fontsize=8.6, fontweight="bold", color=ATT, ha="left",
            va="center")

    enc = block(ax, 4.07, 4.16, 0.86, 0.58, "Encoder", ENC, fontsize=7.4)
    mid = block(ax, 5.13, 4.16, 0.80, 0.58, "Mid attn", ATT, fontsize=7.4)
    dec = block(ax, 6.21, 4.16, 0.86, 0.58, "Decoder", DEC, fontsize=7.4)
    arrow(ax, (2.77, 4.16), (enc["l"][0] - 0.04, 4.16), ms=11)
    ax.text(3.13, 4.34, "$x_t$", fontsize=7.6, color=INK, ha="center",
            va="center")
    arrow(ax, (enc["r"][0] + 0.04, 4.16), (mid["l"][0] - 0.04, 4.16), ms=11)
    arrow(ax, (mid["r"][0] + 0.04, 4.16), (dec["l"][0] - 0.04, 4.16), ms=11)
    arrow(ax, (dec["r"][0] + 0.04, 4.16), (7.43, 4.16), ms=11)
    ax.text(7.21, 4.36, r"$\hat{\epsilon}$", fontsize=8.0, color=INK,
            ha="center", va="center")
    skip_connection(ax, (4.07, 4.47), (6.21, 4.47), rad=-0.17)

    temb = block(ax, 4.07, 3.68, 1.06, 0.32, "t embedding", AUX,
                 fontsize=6.8)
    route(ax, [(temb["r"][0] + 0.04, 3.68), (5.13, 3.68),
               (5.13, mid["b"][1] - 0.04)], color=AUX, lw=1.0, ms=10)

    xc = (COLS[2] + COLS[3]) / 2
    arrow(ax, (xc, 3.36), (xc, Y_REV + 0.20), color=ATT, lw=1.0, ls=DASH,
          ms=11)
    ax.text(xc - 0.12, 3.24, "applied at every reverse step", fontsize=6.8,
            color=ATT, ha="right", va="center")


def right_column(fig, ax):
    panel(ax, 11.86, 5.62, 2.72, 2.42)
    ax.text(10.60, 6.62, "Noise schedule", fontsize=8.4, fontweight="bold",
            color=INK, ha="left", va="center", zorder=4)
    axs = fig.add_axes([10.86 / W, 4.72 / H, 1.86 / W, 1.42 / H])
    t = np.linspace(0, 1000, 400)
    beta = 1e-4 + (0.02 - 1e-4) * t / 1000.0
    abar = np.exp(-np.cumsum(beta) * (t[1] - t[0]))
    axs.plot(t, abar, color=ENC, lw=1.4, label=r"$\bar{\alpha}_t$")
    axs.plot(t, beta / beta.max(), color=LOSS, lw=1.4,
             label=r"$\beta_t\,/\,\beta_T$")
    for k, s in enumerate(STEPS):
        axs.axvline(s, color="#c9d0d8", lw=0.6, ls=(0, (2, 2)), zorder=0)
    axs.set_xlim(0, 1000)
    axs.set_ylim(-0.03, 1.06)
    axs.set_xticks([0, 500, 1000])
    axs.set_yticks([0, 0.5, 1.0])
    axs.tick_params(labelsize=6.0, length=2, width=0.6, pad=1.5)
    axs.set_xlabel("timestep t", fontsize=6.2, labelpad=1.0)
    axs.legend(fontsize=6.4, loc="center right", handlelength=1.1,
               borderaxespad=0.3)
    for sp in axs.spines.values():
        sp.set_linewidth(0.6)
    axs.set_facecolor("none")

    panel(ax, 11.86, 3.04, 2.72, 2.42)
    ax.text(10.60, 4.04, "Recipe", fontsize=8.4, fontweight="bold",
            color=INK, ha="left", va="center", zorder=4)
    lines = [(r"$x_t=\sqrt{\bar{\alpha}_t}\,x_0+"
              r"\sqrt{1-\bar{\alpha}_t}\,\epsilon$", 8.6),
             (r"$\mathcal{L}=\|\epsilon-\epsilon_\theta(x_t,t)\|^2$", 8.6)]
    yy = 3.72
    for txt, fs in lines:
        ax.text(11.86, yy, txt, fontsize=fs, color=INK, ha="center",
                va="center", zorder=4)
        yy -= 0.42
    notes = ["T = 1000 training steps",
             "cosine schedule, v-parameterisation",
             "DDIM sampler, 50 steps",
             "classifier-free guidance w = 3.0"]
    for k, s in enumerate(notes):
        ax.text(10.60, 2.72 - 0.27 * k, "•  " + s, fontsize=6.8, color=INK,
                ha="left", va="center", zorder=4)


def timeline(ax):
    x0, x1 = 0.49, COLS[-1] + TILE / 2
    ax.plot([x0, x1], [1.44, 1.44], color=MUTED, lw=1.0)
    arrow(ax, (x1, 1.44), (x1 + 0.30, 1.44), color=MUTED, lw=1.0, ms=10)
    for x, s in zip(COLS, STEPS):
        ax.plot([x, x], [1.44, 1.32], color=MUTED, lw=1.0)
        ax.text(x, 1.18, f"t = {s}", ha="center", va="center", fontsize=6.8,
                color=MUTED)
    ax.text(x1 + 0.28, 1.60, "more noise", ha="right", va="center",
            fontsize=6.8, color=MUTED)
    ax.text(0.49, 0.90, "shared timestep axis — a vertical slice through "
            "both rows is one noise level", fontsize=7.2, color=INK,
            ha="left", va="center")


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    titles(ax, 0.42, 7.44,
           "Diffusion model: forward corruption and learned reverse sampling",
           "Columns are timesteps; the same denoiser network is reused at "
           "every reverse step, conditioned on the timestep embedding")
    rows(fig, ax)
    denoiser(ax)
    right_column(fig, ax)
    timeline(ax)
    savefig(fig, "266_net_diffusion_pipeline.png")


if __name__ == "__main__":
    main()
