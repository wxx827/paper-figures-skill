"""Figure 1090 - What the eight spatial attention heads look at, on one frame.

The frame is the shared library's ``tabletop`` scene: a short lens looking down
at a desk under tungsten light, with seven discrete objects.  A cluttered
tabletop makes a far better substrate for attention maps than an open road -
there are actually distinct things for the heads to lock onto - and it keeps
this figure from sharing a scene with 1492 / 1496 / 1497.

The maps live on a coarse 19 x 26 token grid and are drawn with bilinear
interpolation, which is what a real attention rollout looks like once it is
upsampled to pixel space.  Each head is anchored on an object the scene
generator actually placed, so the blobs sit on content rather than on
hand-typed coordinates.  All eight share a single colour range and a single
colour bar; the subordinate strip underneath reports each head's normalised
entropy, computed from the very distribution that is plotted above it.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import image_axes, shared_norm
from _wm_scenes import build, grey, octave_noise

H, W = 152, 204                  # frame pixels
TH, TW = 19, 26                  # token grid
HZ = 0.34                        # table / wall junction of the tabletop scene
ATT_CMAP = get_cmap("cb_sequential_blue")


def blob(ty, tx, sy, sx):
    yy, xx = np.mgrid[0:TH, 0:TW].astype(float)
    return np.exp(-0.5 * (((yy - ty * TH) / max(sy * TH, 0.4)) ** 2 +
                          ((xx - tx * TW) / max(sx * TW, 0.4)) ** 2))


def on(anchor, tight=1.0):
    """A token-grid blob centred on a scene object, sized from its extent."""
    return blob(anchor["cy"] / H, anchor["cx"] / W,
                tight * anchor["ry"] / H, tight * anchor["rx"] / W)


def head_maps(objs, seed=77):
    rng = np.random.default_rng(seed)
    v = np.mgrid[0:TH, 0:TW][0].astype(float) / TH

    # the floors are deliberately tiny: a head that only looks at one object
    # must come out with a genuinely low entropy, or the strip below says
    # nothing at all
    raw = [
        0.015 + 1.00 * np.clip((v - HZ) / (1 - HZ), 0, 1) ** 2.0,    # table
        0.006 + 1.00 * on(objs["laptop"], 0.55),                     # screen
        0.004 + 1.00 * on(objs["mug"], 0.50),                        # mug
        0.006 + 1.00 * on(objs["book"], 0.45),                       # book
        0.80 + 0.42 * octave_noise(rng, (TH, TW), base=1.6, octaves=2),
        0.008 + 1.00 * np.exp(-0.5 * ((v - HZ) / 0.045) ** 2),       # wall seam
        0.010 + 1.00 * (on(objs["bowl"], 0.45)
                        + 0.55 * blob(0.52, 0.030, 0.22, 0.035)),    # periphery
        0.003 + 1.00 * on(objs["pen"], 0.34),                        # pen
    ]
    maps, ents = [], []
    for a in raw:
        a = np.clip(gaussian_filter(a, 0.7), 1e-6, None)
        p = a / a.sum()
        maps.append(p)
        ents.append(float(-(p * np.log(p)).sum() / np.log(p.size)))
    return maps, ents


LOOKS = ["table surface", "laptop screen", "mug", "book",
         "diffuse / global", "wall seam", "left periphery", "pen"]


def main():
    set_pub_style(base_font=8)
    scene = build("tabletop", seed=515, shape=(H, W))
    # a touch more contrast than the default remap: the overlay eats local
    # contrast, and a washed-out substrate makes the blobs look unanchored
    frame = np.clip((grey(scene["rgb"], lo=0.06, hi=0.92) - 0.5) * 1.18 + 0.5,
                    0, 1)
    objs = {a["name"]: a for a in scene["anchors"]}
    maps, ents = head_maps(objs)
    # one range for all heads; the sharpest peaks clip, which is what keeps the
    # diffuse heads from disappearing under a single scale
    norm = shared_norm(maps, robust=(0.0, 98.6))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 4, wspace=0.04, hspace=0.085,
                          height_ratios=[1.0, 1.0, 0.56],
                          left=0.052, right=0.905, top=0.885, bottom=0.085)

    im = None
    for k, what in enumerate(LOOKS):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        image_axes(ax, color="#9a9a9a", lw=0.55)
        ax.set_aspect("auto")
        ax.imshow(frame, cmap="gray", vmin=0, vmax=1, aspect="auto",
                  interpolation="bilinear", zorder=1)
        alpha = np.clip(norm(maps[k]), 0, 1) ** 0.55 * 0.90
        im = ax.imshow(maps[k], cmap=ATT_CMAP, norm=norm, alpha=alpha,
                       extent=(0, W, H, 0), aspect="auto",
                       interpolation="bilinear", zorder=2)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.text(0.028, 0.955, f"Head {k + 1}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.4, fontweight="bold",
                color="#ffffff", zorder=9,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="#1a1a1acc",
                          edgecolor="none"))
        ax.text(0.972, 0.955, what, transform=ax.transAxes, ha="right",
                va="top", fontsize=6.2, color="#ffffff", zorder=9,
                path_effects=[pe.Stroke(linewidth=1.6, foreground="#1a1a1a"),
                              pe.Normal()])
        ax.text(0.028, 0.045, f"H = {ents[k]:.2f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.2, color="#1a1a1a", zorder=9,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#ffffffcc",
                          edgecolor="none"))

    # ---- shared colour bar for the attention weight
    cax = fig.add_axes([0.918, 0.336, 0.013, 0.549])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("Attention weight per token", fontsize=7.4, labelpad=4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.4, color="#333333")
    cb.set_ticks(np.linspace(norm.vmin, norm.vmax, 4))
    cb.ax.set_yticklabels([f"{t:.3f}" for t in
                           np.linspace(norm.vmin, norm.vmax, 4)])

    # ---- subordinate strip: normalised entropy of each head
    axb = fig.add_subplot(gs[2, :])
    lo, hi = min(ents), max(ents)
    cols = [ATT_CMAP(0.25 + 0.65 * (1 - (e - lo) / (hi - lo + 1e-9)))
            for e in ents]
    axb.bar(np.arange(8), ents, width=0.56, color=cols, edgecolor="#333333",
            linewidth=0.5, zorder=3)
    axb.axhline(float(np.mean(ents)), color="#808080", lw=0.9, ls="--", zorder=2)
    axb.text(7.62, float(np.mean(ents)) + 0.012, "mean", fontsize=6.4,
             color="#777777", ha="right", va="bottom")
    for k, e in enumerate(ents):
        axb.text(k, e + 0.012, f"{e:.2f}", ha="center", va="bottom",
                 fontsize=6.3, color="#333333")
    axb.set_xticks(np.arange(8))
    axb.set_xticklabels([f"Head {k + 1}" for k in range(8)], fontsize=7.2)
    axb.set_xlim(-0.62, 7.62)
    axb.set_ylim(0, min(1.06, hi + 0.13))
    axb.set_ylabel("Norm. entropy", fontsize=7.6)
    axb.tick_params(axis="y", labelsize=6.8)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.set_axisbelow(True)
    axb.grid(axis="y", lw=0.5, color="#e2e2e2")

    savefig(fig, "1090_wm_qual_attention_heads.png")


if __name__ == "__main__":
    main()
