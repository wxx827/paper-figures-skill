"""Figure 1200 - Long-horizon consistency of one eight-step rollout.

The strip is a single ``conveyor`` scene from the shared library evolved
through eight time steps - the same set, the same seed, the same camera, only
``t`` changes - so it really is one scene at eight moments rather than eight
seeds of one generator (BRIEF_WM section 1.6, fifth step).  The belt geometry
and the wall tone are overridden so this rollout does not reuse the parameters
any other figure in the library gave that kind.

The three curves underneath are *measured on those eight frames*, not invented
next to them:

* object permanence - how many of the four tracked parts still have their
  anchor inside the frame, over how many were inside at step 0;
* spatial relation - the share of left-right orderings among the parts that
  are visible in both step 0 and step k that still hold;
* attribute stability - one minus the mean absolute change of a fixed patch
  of pixels centred on each surviving part's anchor since step 0.

So the vertical inconsistency markers cannot land anywhere except where a
metric really falls: they are placed at the steps where permanence or relation
drops below one.  The magnifier hangs on the anchor of the part that is
crossing the frame boundary at the first such step, which is why the crop
shows an object half out of view rather than empty belt.

Canvas (11.0, 6.3).  The first cut was (11.0, 5.6), which ``bbox_inches
= "tight"`` delivered at h/w = 0.499 - out of the 0.55-0.80 band of
DESIGN_SYSTEM section 1.2, and not on any of E1-E8: the row of eight tiles
fixes the width but nothing here fixes the height, so the fix is the height,
not an exception.  The extra inch also puts the tiles back on their own
aspect (145 x 112 rendered into a 1.24 x 0.98 in box) instead of squashing
them 9 % to fit.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes
from figkit import image_axes, roi_rect
from _wm_scenes import build

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#d5d5d5"
EVENT = "#8a8a8a"
FLAG = "#12314f"

NSTEP = 8
T0, DT = 0.20, 0.16
SHAPE = (112, 145)
SEED = 7605
SCENE = dict(belt_top=0.40, belt_bot=0.86, wall="#464e57")
CAM = dict(exposure=1.04, vignette=0.21, contrast=1.10, sharpen=1.24)

METRICS = [
    ("Object permanence", "#12314f", "-", "D"),
    ("Spatial relation", "#2f6fa8", (0, (5.0, 2.2)), "o"),
    ("Attribute stability", "#5b93c8", (0, (1.9, 1.9)), "s"),
]
PATCH = (9, 13)                     # half-height, half-width of the patch


def parts_of(scene):
    return [a for a in scene["anchors"] if a["name"] == "part"]


def rollout():
    """Eight frames of one scene, plus the three metrics read off them."""
    h, w = SHAPE
    frames, anchors = [], []
    for k in range(NSTEP):
        sc = build("conveyor", seed=SEED, shape=SHAPE, t=T0 + k * DT,
                   cam=CAM, **SCENE)
        frames.append(sc["rgb"])
        anchors.append(parts_of(sc))
    n_parts = len(anchors[0])

    def inside(a):
        return 0.0 <= a["cx"] <= (w - 1.0)

    vis = np.array([[inside(a) for a in row] for row in anchors])
    base = vis[0]

    def patch(img, a):
        cy = int(np.clip(a["cy"], PATCH[0], h - PATCH[0] - 1))
        cx = int(np.clip(a["cx"], PATCH[1], w - PATCH[1] - 1))
        return img[cy - PATCH[0]:cy + PATCH[0],
                   cx - PATCH[1]:cx + PATCH[1]].mean(axis=(0, 1))

    ref = [patch(frames[0], a) if base[i] else None
           for i, a in enumerate(anchors[0])]

    perm, rel, attr = [], [], []
    for k in range(NSTEP):
        both = [i for i in range(n_parts) if base[i] and vis[k][i]]
        perm.append(len(both) / float(base.sum()))

        pairs = [(i, j) for ii, i in enumerate(both) for j in both[ii + 1:]]
        keep = sum(1 for i, j in pairs
                   if np.sign(anchors[k][i]["cx"] - anchors[k][j]["cx"])
                   == np.sign(anchors[0][i]["cx"] - anchors[0][j]["cx"]))
        rel.append(keep / float(len(pairs)) if pairs else 1.0)

        d = [float(np.abs(patch(frames[k], anchors[k][i]) - ref[i]).mean())
             for i in both]
        attr.append(float(np.clip(1.0 - np.mean(d) / 0.30, 0.0, 1.0))
                    if d else 1.0)

    return frames, anchors, vis, (np.array(perm), np.array(rel),
                                  np.array(attr))


def main():
    set_pub_style()
    frames, anchors, vis, (perm, rel, attr) = rollout()
    series = [perm, rel, attr]
    h, w = SHAPE

    # a step is flagged where the scene itself loses an object or scrambles an
    # ordering *relative to the step before it* - an event, not a level - so
    # the markers cannot drift away from the thing they claim to mark
    flags = [k for k in range(1, NSTEP)
             if perm[k] < perm[k - 1] - 1e-6 or rel[k] < rel[k - 1] - 1e-6]
    # magnify the ordering break: an object wraps back in at the left edge, so
    # the before / after crops show empty belt and then a new object, which is
    # a picture of the metric rather than a caption about it
    ev = [k for k in flags if rel[k] < rel[k - 1] - 1e-6][0]
    prev = ev - 1
    back = [i for i in range(len(anchors[ev]))
            if vis[ev][i] and not vis[prev][i]][0]
    ea = anchors[ev][back]

    fig = plt.figure(figsize=(11.0, 6.3))
    gs = fig.add_gridspec(2, NSTEP, height_ratios=[0.34, 1.0], hspace=0.26,
                          wspace=0.042, left=0.052, right=0.988, top=0.812,
                          bottom=0.118)
    tiles = [fig.add_subplot(gs[0, k]) for k in range(NSTEP)]
    ax = fig.add_subplot(gs[1, :])

    for k, axt in enumerate(tiles):
        flagged = k in flags
        image_axes(axt, color=FLAG if flagged else "#9a9a9a",
                   lw=1.5 if flagged else 0.6)
        axt.set_aspect("auto")
        axt.imshow(frames[k], interpolation="bilinear", aspect="auto")
        axt.set_title(f"step {k}", fontsize=7.0, pad=2.8,
                      color=FLAG if flagged else SUB,
                      fontweight="bold" if flagged else "normal")
    tiles[0].set_ylabel("Rollout", fontsize=7.0, labelpad=3.0)

    # one window, two steps: the same pixels before and after the event
    bw, bh = 44, 36
    box = (int(np.clip(ea["cx"] - bw / 2, 0, w - bw)),
           int(np.clip(ea["cy"] - bh / 2, 0, h - bh)), bw, bh)
    roi_rect(tiles[prev], box, color=FLAG, lw=1.2)
    roi_rect(tiles[ev], box, color=FLAG, lw=1.2)

    # x of step k must land under the centre of tile k: solve the affine map
    # from the two outer tile positions instead of assuming zero gutters
    p0, p7 = tiles[0].get_position(), tiles[-1].get_position()
    pa = ax.get_position()
    f0 = (p0.x0 + p0.width / 2 - pa.x0) / pa.width
    f7 = (p7.x0 + p7.width / 2 - pa.x0) / pa.width
    span = (NSTEP - 1) / (f7 - f0)
    ax.set_xlim(-f0 * span, -f0 * span + span)
    # the floor is well below the lowest measurement: the space under the
    # curves is where the magnifier lives, not a truncated axis
    ax.set_ylim(0.10, 1.06)

    xs = np.arange(NSTEP)
    ytop, ybot = 1.02, 0.360
    for k in flags:
        ax.plot([k, k], [ybot, ytop], color=EVENT, lw=0.9,
                linestyle=(0, (2.2, 2.0)), zorder=2)
    for (name, col, ls, mk), y in zip(METRICS, series):
        ax.plot(xs, y, color=col, lw=1.9, linestyle=ls, marker=mk,
                markersize=5.0, markerfacecolor="white", markeredgewidth=1.2,
                label=name, zorder=4, solid_capstyle="round")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}" for k in xs])
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Consistency against step 0")
    ax.grid(True, axis="y", color=HAIR, lw=0.55, alpha=0.85)
    style_axes(ax)

    # ---- before / after magnifier, parked in the corner the curves leave
    # empty; the two crops are the same pixel window in two consecutive frames
    iw, x_ins = 0.105, 0.062
    fw_in, fh_in = fig.get_size_inches()          # never the figsize literal
    ih = iw * (pa.width * fw_in) / (pa.height * fh_in) * (bh / float(bw))
    for n, k in enumerate((prev, ev)):
        axi = ax.inset_axes([x_ins + n * (iw + 0.010), 0.315, iw, ih])
        axi.imshow(frames[k][box[1]:box[1] + bh, box[0]:box[0] + bw],
                   interpolation="bilinear", aspect="auto")
        axi.set_xticks([])
        axi.set_yticks([])
        for s in axi.spines.values():
            s.set_visible(True)
            s.set_linewidth(1.5)
            s.set_color(FLAG)
        axi.set_zorder(7)
        axi.patch.set_visible(True)
        axi.set_title(f"step {k}", fontsize=6.2, pad=2.0, color=FLAG,
                      fontweight="bold")
    mag = (iw * pa.width) / ((bw / float(w)) * p0.width)
    ax.text(x_ins, 0.030,
            f"{mag:.1f}x crop of one fixed pixel window at the left edge of "
            f"the belt.  An object wraps back into it\n"
            f"between step {prev} and step {ev}; at step 0 that object was "
            f"the right-most of the four, so every\n"
            f"ordering it takes part in inverts and spatial relation falls "
            f"from {rel[prev]:.2f} to {rel[ev]:.2f}.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6,
            color=RULE, linespacing=1.62, zorder=8,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="none", alpha=0.95))

    add_panel_label(tiles[0], "(a)", x=-0.30, y=1.34)
    add_panel_label(ax, "(b)", x=-0.045, y=1.010)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=EVENT, lw=0.9,
                          linestyle=(0, (2.2, 2.0)),
                          label="step flagged inconsistent"))
    assert len(handles) == len(METRICS) + 1, \
        "legend must list the three metrics and the flag marker, nothing else"

    shown = [f"{v:.2f}" for v in perm]
    tot = sum(map(float, shown))
    # four decimals: 6.50 / 8 is exactly 0.8125, and a three-decimal print of a
    # tie leaves the reader's calculator one digit away from the page
    mean_s = f"{tot / NSTEP:.4f}"
    assert abs(float(mean_s) - tot / NSTEP) < 5e-5, \
        "the printed mean is not the printed sum divided by the step count"

    savefig(fig, "1200_wm_mem_longhorizon_consistency.png")


if __name__ == "__main__":
    main()
