"""Figure 1164 - Egocentric action timeline: four lanes and a closed IoU bar.

Four swimlanes over one 62 s kitchen clip - predicted segments, ground truth,
gaze stability and hand visibility - and underneath, per-class temporal IoU.

Everything is derived, in this order, from the ground-truth segmentation:

    GT segments -> gaze stability dips at every GT boundary
                -> hand visibility, one level per action class
                -> boundary jitter, scaled by 1 - stability at that boundary
                -> the three class flips, placed on the three GT segments
                   with the lowest mean stability
                -> predicted segments
                -> intersection / union / IoU per class

So the headline - errors cluster where gaze is unstable - is not a caption
laid over an unrelated picture; it is how the predicted lane was built.

Calculator closure (BRIEF_WM section 1.7): the IoU printed on each bar is
verifiable from the intersection and union printed beside it, at the
precision printed.  ``closed_ratio`` adds a decimal wherever two would not
reproduce the quotient, and ``main`` refuses to save if the dashed mean
disagrees with the mean of the five printed bars.

Every label that sits over a lane, a grid line or a leader carries a white
plate.  This family is nothing but lines, and text struck through by one has
been the most repeated defect of the batch.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from pubstyle import set_pub_style, savefig, style_axes, get_cmap
from figkit import background_card

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
RULE, DEEP = "#333333", "#20455f"
CARD = "#f2f4f6"
PLATE = dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92)

DUR = 62.0
DT = 0.005
GRID = np.arange(0.0, DUR, DT)

# Five cool hues: distinguishable without leaving the blue half of the wheel,
# so not one of them draws on the warm-area budget.
CLASSES = ["take", "open", "pour", "stir", "put down"]
CCOL = ["#16394f", "#2e74a8", "#7fb2d4", "#3f7f78", "#6a5f9e"]
CINK = ["#ffffff", "#ffffff", "#12354e", "#ffffff", "#ffffff"]
HAND_LEVEL = [0.86, 0.62, 0.94, 0.90, 0.48]      # hand in frame, per class

SEED = 1569  # original id; figure number is 1164


# ---------------------------------------------------------------------------
def gt_segments():
    rng = np.random.default_rng(SEED)
    segs, t, prev = [], 0.0, -1
    while t < DUR - 1.5:
        c = int(rng.integers(0, len(CLASSES)))
        while c == prev:
            c = int(rng.integers(0, len(CLASSES)))
        d = float(rng.uniform(2.6, 6.8))
        segs.append((t, min(t + d, DUR), c))
        t += d
        prev = c
    segs[-1] = (segs[-1][0], DUR, segs[-1][2])
    return segs


def labels_of(segs):
    lab = np.full(GRID.shape, -1, dtype=int)
    for s in segs:
        lab[(GRID >= s[0]) & (GRID < s[1])] = s[2]
    return lab


def stability(segs):
    """Gaze stability: steady mid-action, collapsing at every truth cut."""
    s = np.full(GRID.shape, 0.93)
    for s0 in segs[1:]:
        s -= 0.62 * np.exp(-0.5 * ((GRID - s0[0]) / 0.85) ** 2)
    s += 0.045 * np.sin(GRID / 2.7) + 0.030 * np.sin(GRID / 0.9 + 1.1)
    return np.clip(s, 0.03, 1.0)


def visibility(lab):
    v = np.array([HAND_LEVEL[c] if c >= 0 else 0.40 for c in lab])
    k = int(1.1 / DT)
    return np.clip(np.convolve(v, np.ones(k) / k, mode="same"), 0.0, 1.0)


def predict(segs, stab):
    """Perturb the truth the way a real model fails: boundaries slip where
    the gaze is unstable, and whole segments are relabelled there too."""
    rng = np.random.default_rng(SEED + 7)

    def at(t):
        return float(stab[min(int(t / DT), len(stab) - 1)])

    mean_stab = [float(stab[(GRID >= a) & (GRID < b)].mean())
                 for (a, b, _) in segs]
    flip = {int(i) for i in np.argsort(mean_stab)[:3]}

    cuts = [0.0]
    for s in segs[1:]:
        slip = rng.normal(0.0, 0.95) * (1.35 - at(s[0]))
        cuts.append(float(np.clip(s[0] + slip, 0.4, DUR - 0.4)))
    cuts.append(DUR)
    cuts = list(np.maximum.accumulate(cuts))

    out = []
    for i, (_, _, c) in enumerate(segs):
        cc = c
        if i in flip:
            alt = [k for k in range(len(CLASSES)) if k != c]
            cc = int(alt[int(rng.integers(0, len(alt)))])
        out.append((cuts[i], cuts[i + 1], cc, i in flip))
    return [s for s in out if s[1] - s[0] > 1e-6], flip


def closed_ratio(inter, union):
    """(intersection, union, ratio, decimals) such that the division closes."""
    v = round(inter / union, 3)
    for nd in (2, 3, 4):
        i, u = round(inter, nd), round(union, nd)
        if u > 0 and abs(i / u - v) <= 0.0005:
            return i, u, v, nd
    return round(inter, 4), round(union, 4), v, 4


def lane_bars(ax, y, segs, height=0.62, hatched=False):
    for s in segs:
        t0, t1, c = s[0], s[1], s[2]
        bad = hatched and s[3]
        ax.add_patch(Rectangle((t0, y - height / 2), t1 - t0, height,
                               facecolor=CCOL[c], edgecolor="#14202c",
                               lw=1.30 if bad else 0.55, zorder=3))
        if bad:
            # the hatch is stroked in the patch's edge colour, so it gets the
            # class's own contrast ink rather than a fixed one that would
            # vanish on either the darkest or the lightest class
            ax.add_patch(Rectangle((t0, y - height / 2), t1 - t0, height,
                                   facecolor="none", edgecolor=CINK[c],
                                   lw=0.0, hatch="////", zorder=4))
        if t1 - t0 > 3.4:
            ax.text(0.5 * (t0 + t1), y, CLASSES[c], ha="center", va="center",
                    fontsize=6.3, zorder=6,
                    color="#12354e" if bad else CINK[c],
                    bbox=PLATE if bad else None)


def strip(ax, y, series, cmap, height=0.62):
    ax.imshow(series[None, :], cmap=cmap, vmin=0.0, vmax=1.0,
              extent=(0.0, DUR, y + height / 2, y - height / 2),
              aspect="auto", interpolation="bilinear", zorder=3)
    ax.add_patch(Rectangle((0.0, y - height / 2), DUR, height,
                           facecolor="none", edgecolor="#9a9a9a", lw=0.55,
                           zorder=4))


def main():
    set_pub_style(base_font=8)

    GT = gt_segments()
    gt_lab = labels_of(GT)
    STAB = stability(GT)
    VIS = visibility(gt_lab)
    PR, FLIP = predict(GT, STAB)
    pr_lab = labels_of(PR)

    raw_i, raw_u, inter, union, iou, nd = [], [], [], [], [], []
    for c in range(len(CLASSES)):
        g, p = gt_lab == c, pr_lab == c
        ri, ru = float((g & p).sum()) * DT, float((g | p).sum()) * DT
        i, u, v, d = closed_ratio(ri, ru)
        raw_i.append(ri)
        raw_u.append(ru)
        inter.append(i)
        union.append(u)
        iou.append(v)
        nd.append(d)

    miou = float(np.mean(iou))                      # mean of the printed bars
    true_miou = float(np.mean([a / b for a, b in zip(raw_i, raw_u)]))
    assert abs(miou - true_miou) < 5e-4, \
        f"mean IoU {miou:.5f} does not close on the bars ({true_miou:.5f})"
    correct = float((gt_lab == pr_lab).mean())
    print("per-class IoU:", iou, "\ndecimals printed:", nd,
          "\nmIoU printed:", round(miou, 4), " true:", round(true_miou, 5),
          "\nframe accuracy:", round(correct, 4),
          "\nsegments:", len(GT), " flipped:", sorted(FLIP))

    fig = plt.figure(figsize=(10.4, 6.0))
    background_card(fig, (0.012, 0.300, 0.976, 0.548), facecolor=CARD,
                    radius=0.010, zorder=-5)
    background_card(fig, (0.012, 0.058, 0.976, 0.212), facecolor=CARD,
                    radius=0.010, zorder=-5)

    ax = fig.add_axes([0.152, 0.338, 0.824, 0.458])
    ax.patch.set_visible(False)
    for t in range(0, int(DUR) + 1, 10):
        ax.axvline(t, color="#d6dce2", lw=0.7, zorder=0)

    # a pale column behind each relabelled segment, so a reader can check the
    # headline against the gaze lane instead of taking it on trust
    for s in PR:
        if s[3]:
            ax.add_patch(Rectangle((s[0], -1.05), s[1] - s[0], 4.70,
                                   facecolor="#20455f", alpha=0.085,
                                   edgecolor="none", zorder=1))

    lane_bars(ax, 0.0, PR, hatched=True)
    lane_bars(ax, 1.0, GT)
    strip(ax, 2.0, STAB, get_cmap("cb_sequential_blue"))
    strip(ax, 3.0, VIS, get_cmap("cb_sequential_purple"))

    names = ["Predicted", "Ground truth", "Gaze stability", "Hand visibility"]
    hints = ["hatched = wrong class", f"{len(GT)} annotated segments",
             "dark = steady fixation", "dark = both hands in frame"]
    for k, (n, hnt) in enumerate(zip(names, hints)):
        ax.text(-0.013, k, n, transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=7.8, fontweight="bold", color=INK)
        ax.text(-0.013, k, hnt, transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=6.3, color=MUTED)

    # the worst boundary slip: measured, not chosen
    worst, wi = max((abs(PR[i][0] - GT[i][0]), i) for i in range(1, len(GT)))
    a, b = sorted([GT[wi][0], PR[wi][0]])
    ax.add_patch(FancyArrowPatch((a, 0.50), (b, 0.50), arrowstyle="<|-|>",
                                 mutation_scale=7, color=DEEP, lw=1.1,
                                 zorder=7))
    ax.annotate(f"largest boundary slip {worst:.2f} s",
                xy=(0.5 * (a + b), 0.44), xytext=(0.5 * (a + b), -0.72),
                ha="center", va="center", fontsize=6.4, color=DEEP, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=DEEP,
                                shrinkA=1.5, shrinkB=0.5),
                bbox=PLATE)

    ax.set_xlim(0.0, DUR)
    ax.set_ylim(3.70, -1.30)
    ax.set_yticks([])
    ax.set_xlabel("time in the clip (s)", fontsize=7.6, labelpad=2.4)
    ax.tick_params(labelsize=7.0)
    ax.spines["left"].set_visible(False)
    style_axes(ax)
    ax.text(0.0, 1.050, "(a)   Four aligned lanes over one 62 s clip",
            transform=ax.transAxes, fontsize=8.4, fontweight="bold",
            color=INK, ha="left", va="bottom")
    ax.text(1.0, 1.050,
            f"{len(FLIP)} of {len(GT)} segments given the wrong class;  "
            f"frame-level accuracy {100 * correct:.1f} % once the boundary "
            "slips are counted too", transform=ax.transAxes, fontsize=6.9,
            color=SUB, ha="right", va="bottom")

    # ---- (b) per-class IoU, with the division that produced it ------------
    axb = fig.add_axes([0.152, 0.096, 0.824, 0.140])
    axb.patch.set_visible(False)
    ypos = np.arange(len(CLASSES))[::-1]
    for c in range(len(CLASSES)):
        y = ypos[c]
        axb.barh(y, iou[c], height=0.60, color=CCOL[c], edgecolor="#14202c",
                 lw=0.55, zorder=3)
        axb.text(iou[c] + 0.014, y,
                 f"{inter[c]:.{nd[c]}f} s  /  {union[c]:.{nd[c]}f} s  =  "
                 f"{iou[c]:.3f}", va="center", ha="left", fontsize=6.5,
                 color=DEEP, zorder=5, bbox=PLATE)
        axb.text(-0.008, y, CLASSES[c], va="center", ha="right",
                 fontsize=7.0, color=INK)
    axb.axvline(miou, color=RULE, lw=1.0, ls=(0, (3.6, 2.2)), zorder=4)
    axb.text(miou, len(CLASSES) - 0.58, f"mean {miou:.3f}", ha="center",
             va="bottom", fontsize=6.8, color=RULE, zorder=6, bbox=PLATE)
    axb.set_xlim(0.0, 1.34)
    axb.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axb.set_ylim(-0.60, len(CLASSES) - 0.34)
    axb.set_yticks([])
    axb.set_xlabel("temporal IoU between the two segment lanes above",
                   fontsize=7.4, labelpad=2.2)
    axb.tick_params(labelsize=7.0)
    axb.spines["left"].set_visible(False)
    # the extra width is there for the division labels, not for data
    axb.spines["bottom"].set_bounds(0.0, 1.0)
    style_axes(axb)
    axb.text(0.0, 1.11, "(b)   Per-class overlap, with the division shown",
             transform=axb.transAxes, fontsize=8.4, fontweight="bold",
             color=INK, ha="left", va="bottom")

    # both banners were single lines of 10.3-10.8 in on a 10.4 in canvas, and
    # `bbox_inches="tight"` widened the export to match, taking the delivered
    # h/w to 0.546 - below the 0.55 floor the layout was designed for
    savefig(fig, "1164_wm_ego_action_timeline.png")


if __name__ == "__main__":
    main()
