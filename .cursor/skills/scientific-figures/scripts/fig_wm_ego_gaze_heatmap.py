"""Figure 1159 - Egocentric gaze prediction: three maps over four clips.

Rows are the ground-truth fixation density, our model and a baseline; columns
are four held-out first-person clips, each shot in a different room with its
own focal length, exposure and white balance so the grid does not read as one
scene re-seeded four times.

Everything a label claims is computed from the arrays that are actually drawn:

* the fixation points come from the scene generator's own ``anchors``, so a
  cross never lands on empty wall;
* the three density maps are built from those same points, normalised to
  "times uniform" (mean 1.0) and shown through one shared colour range;
* AUC-Judd is measured for every tile against the plotted fixations, and the
  per-row number on the right is the mean of that row's four tiles.  Change a
  map and the numbers move with it.

Colour: the frames are converted to a light grey so the cool blue density has
somewhere to sit, and the only markers are white-on-dark, keeping the warm
area of the figure at essentially zero.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import image_axes
from _wm_scenes import build, grey

H, W = 180, 210
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
HEAT_CMAP = get_cmap("cb_sequential_blue")

# (room, seed, camera overrides, clip caption, fixation targets, distractor)
# A fixation target is (anchor name, weight, offset in units of the anchor's
# own radius) - so the whole ground truth is anchored to real painted objects.
CLIPS = [
    dict(room="kitchen", seed=6401, label="Clip 1  ·  kitchen",
         cam=dict(exposure=1.05, wb=(0.958, 1.00, 1.088), dof=1.5,
                  contrast=1.12),
         fix=[("pan", 1.00, (0.00, -0.18)), ("board", 0.42, (-0.20, 0.05))],
         distract="kettle"),
    dict(room="desk", seed=6402, label="Clip 2  ·  office desk",
         cam=dict(exposure=0.99, wb=(0.972, 1.00, 1.062), dof=2.1,
                  contrast=1.08),
         fix=[("monitor", 1.00, (-0.16, 0.10)), ("keyboard", 0.38, (0.0, 0.0))],
         distract="mug"),
    dict(room="workbench", seed=6403, label="Clip 3  ·  workbench",
         cam=dict(exposure=1.04, wb=(0.950, 1.00, 1.080), dof=1.7,
                  contrast=1.14),
         fix=[("drill", 1.00, (0.22, -0.10))],
         distract="crate"),
    dict(room="living", seed=6404, label="Clip 4  ·  living room",
         cam=dict(exposure=0.97, wb=(0.968, 1.00, 1.070), dof=2.3,
                  contrast=1.06),
         fix=[("tray", 1.00, (0.10, -0.12)), ("cup", 0.44, (0.0, -0.10))],
         distract="vase"),
]

ROWS = [("Human fixations", "kernel density of the plotted dots"),
        ("Ours  ·  GLC-Ego", "global-local cross attention"),
        ("Baseline  ·  SlowFast-G", "two-stream, centre prior")]


def blob(cx, cy, sx, sy):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))


def anchor(anchors, name):
    for a in anchors:
        if a["name"] == name:
            return a
    return anchors[0]


def finish_map(m, prior=0.16):
    """Peak-normalise, add the broad prior every real saliency map has, then
    rescale so the mean is 1.0 - the map is then read as 'times uniform'."""
    m = m / max(m.max(), 1e-9)
    m = (1.0 - prior) * m + prior * blob(0.50 * W, 0.56 * H,
                                         0.40 * W, 0.36 * H)
    m = gaussian_filter(m, 1.4)
    return m / max(m.mean(), 1e-9)


def auc_judd(m, pts):
    """P(map at a fixation exceeds the map at a random pixel)."""
    flat = np.sort(m.ravel())
    n = flat.size
    vals = np.array([m[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]
                     for x, y in pts])
    return float(np.mean(np.searchsorted(flat, vals, side="left") / n))


def build_clip(clip):
    """One scene, the observers' fixations, and the three density maps.

    The ground-truth map is a kernel density estimate of the very dots the
    tiles plot, so "GT" is not a hand-drawn blob that happens to sit near
    them - it is those points.
    """
    sc = build("ego", seed=clip["seed"], shape=(H, W), room=clip["room"],
               cam=clip["cam"])
    anc = sc["anchors"]
    rng = np.random.default_rng(clip["seed"] + 77)

    centres, obs = [], []
    for name, wt, (ox, oy) in clip["fix"]:
        a = anchor(anc, name)
        cx, cy = a["cx"] + ox * a["rx"], a["cy"] + oy * a["ry"]
        centres.append((cx, cy))
        n = max(2, int(round(9 * wt)))
        obs.append(np.stack([rng.normal(cx, 0.040 * W, n),
                             rng.normal(cy, 0.046 * H, n)], axis=1))
    # every observer set has a few wandering fixations; they are what keeps
    # AUC away from an implausible 1.000
    obs.append(np.stack([rng.uniform(0.10, 0.90, 2) * W,
                         rng.uniform(0.20, 0.88, 2) * H], axis=1))
    obs = np.concatenate(obs)
    obs[:, 0] = np.clip(obs[:, 0], 3, W - 4)
    obs[:, 1] = np.clip(obs[:, 1], 3, H - 4)
    dis = anchor(anc, clip["distract"])

    gt = np.zeros((H, W))
    for (x, y) in obs:
        gt += blob(x, y, 0.045 * W, 0.052 * H)

    ours = np.zeros((H, W))
    for (cx, cy) in centres:
        ours += blob(cx + 0.022 * W, cy - 0.028 * H, 0.080 * W, 0.090 * H)
    ours += 0.42 * blob(dis["cx"], dis["cy"], 0.070 * W, 0.080 * H)

    base = 0.66 * blob(centres[0][0] - 0.055 * W, centres[0][1] + 0.060 * H,
                       0.150 * W, 0.165 * H)
    base += 0.90 * blob(0.50 * W, 0.52 * H, 0.200 * W, 0.195 * H)
    base += 0.60 * blob(dis["cx"], dis["cy"], 0.115 * W, 0.125 * H)

    maps = [finish_map(gt, 0.22), finish_map(ours, 0.18),
            finish_map(base, 0.24)]
    return sc, maps, obs, centres


def overlay(ax, base_grey, m, norm):
    ax.imshow(base_grey, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto",
              interpolation="bilinear", zorder=1)
    t = np.clip(np.asarray(norm(m)), 0, 1)
    rgba = HEAT_CMAP(t)
    rgba[..., 3] = t ** 1.25 * 0.94
    ax.imshow(rgba, aspect="auto", interpolation="bilinear", zorder=2)


def cross(ax, x, y, r=8.0):
    for lw, col in ((2.6, "#14202c"), (1.2, "#ffffff")):
        ax.plot([x - r, x + r], [y, y], color=col, lw=lw, zorder=7,
                solid_capstyle="butt")
        ax.plot([x, x], [y - r, y + r], color=col, lw=lw, zorder=7,
                solid_capstyle="butt")


def peak_ring(ax, m):
    py, px = np.unravel_index(int(np.argmax(m)), m.shape)
    ax.scatter([px], [py], s=52, facecolors="none", edgecolors="#1a1a1a",
               linewidths=1.9, zorder=6)
    ax.scatter([px], [py], s=52, facecolors="none", edgecolors="#ffffff",
               linewidths=0.9, zorder=7)
    return px, py


def main():
    set_pub_style(base_font=8)

    clips = [build_clip(c) for c in CLIPS]
    all_maps = np.concatenate([m.ravel() for _, maps, _, _ in clips
                               for m in maps])
    # one colour range for all twelve tiles; the square-root scale keeps the
    # broad baseline map readable next to a peaky ground truth
    norm = mpl.colors.PowerNorm(0.55, vmin=0.0,
                                vmax=float(np.percentile(all_maps, 99.6)))

    auc = np.array([[auc_judd(maps[r], obs) for _, maps, obs, _ in clips]
                    for r in range(3)])
    nfix = sum(len(obs) for _, _, obs, _ in clips)
    # Calculator closure: a reader can average the four chips on a row, so
    # the row summary is the mean of the *printed* values rather than of the
    # full-precision ones.  One row lands on a .0005 boundary where the two
    # disagree in the third decimal, which is exactly the rounding trap.
    row_mean = np.round(auc, 3).mean(axis=1)
    drift = np.abs(row_mean - auc.mean(axis=1)).max()
    assert drift < 1e-3, f"row summary drifted from the true mean by {drift}"
    print("AUC-Judd per row/clip:\n", np.round(auc, 3),
          "\nrow means (closed):", np.round(row_mean, 3),
          "\ntrue means:", np.round(auc.mean(axis=1), 4),
          "\nfixations:", nfix)

    fig = plt.figure(figsize=(9.6, 6.4))
    gs = fig.add_gridspec(3, 4, left=0.108, right=0.830, top=0.845,
                          bottom=0.032, wspace=0.030, hspace=0.055)

    for r in range(3):
        for c, (sc, maps, obs, centres) in enumerate(clips):
            ax = fig.add_subplot(gs[r, c])
            image_axes(ax, color="#9a9a9a", lw=0.55)
            ax.set_aspect("auto")
            overlay(ax, grey(sc["rgb"], lo=0.135, hi=0.800), maps[r], norm)
            if r > 0:
                peak_ring(ax, maps[r])
            ax.scatter(obs[:, 0], obs[:, 1], s=8.5, facecolors="#ffffff",
                       edgecolors="#14202c", linewidths=0.55, zorder=6)
            for (x, y) in centres:
                cross(ax, x, y)
            ax.text(0.976, 0.950, f"AUC {auc[r, c]:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.2, color="#20344a", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.24", facecolor="#ffffff",
                              edgecolor="#b9c6d3", linewidth=0.5,
                              alpha=0.90))
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            if r == 0:
                ax.set_title(CLIPS[c]["label"], fontsize=7.6, color=INK,
                             pad=3.4)

        pos = fig.axes[r * 4].get_position()
        yc = pos.y0 + pos.height * 0.5
        name, how = ROWS[r]
        if r == 0:
            how = f"kernel density of the {nfix} plotted dots"
        fig.text(0.040, yc, name, rotation=90, ha="center", va="center",
                 fontsize=8.2, fontweight="bold", color=INK)
        fig.text(0.078, yc, how, rotation=90, ha="center", va="center",
                 fontsize=6.3, color=MUTED)

    # ---- shared colour bar + the per-row mean AUC ------------------------
    top = fig.axes[0].get_position()
    bot = fig.axes[-1].get_position()
    cax = fig.add_axes([0.945, bot.y0, 0.017, top.y1 - bot.y0])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=HEAT_CMAP),
                      cax=cax, ticks=[0, 1, 3, 6, 10, 15])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.4, color="#333333")
    cb.set_label("fixation density  (x uniform, sqrt scale)", fontsize=7.0,
                 color=INK, labelpad=4.0)

    for r in range(3):
        pos = fig.axes[r * 4].get_position()
        yc = pos.y0 + pos.height * 0.5
        fig.text(0.842, yc + 0.028, "mean AUC-Judd", fontsize=6.4,
                 color=MUTED, ha="left", va="center")
        fig.text(0.842, yc - 0.014, f"{row_mean[r]:.3f}", fontsize=12.0,
                 color="#20344a" if r < 2 else MUTED, ha="left", va="center",
                 fontweight="bold")
        fig.text(0.842, yc - 0.055,
                 f"range {auc[r].min():.3f}-{auc[r].max():.3f}", fontsize=6.2,
                 color=MUTED, ha="left", va="center")

    savefig(fig, "1159_wm_ego_gaze_heatmap.png")


if __name__ == "__main__":
    main()
