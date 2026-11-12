"""Figure 1161 - Problem setup: from a head-mounted rig to two gaze tasks.

Read left to right.  (a) the capture rig and what it sees, (b) one recorded
clip with the scanpath drawn on it, (c) the two prediction problems that
scanpath defines.

The three zones sit on pale grey cards - a narrative strip made of line art
alone lands far below the batch's 20 percent ink floor, and the cards are
what carry it back into band.

Numbers are read off geometry rather than typed: the field of view is
measured from the wedge that is actually drawn, and the dwell total, the
saccade count and the mean saccade amplitude all come from the ``SCAN``
array that positions the circles.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (Polygon, Ellipse, FancyBboxPatch,
                                FancyArrowPatch, Circle)

from pubstyle import set_pub_style, savefig
from figkit import image_axes, background_card
from _wm_scenes import build

H, W = 190, 225
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
RULE = "#333333"
DEEP, MID, PALE = "#20455f", "#2e74a8", "#cfe1ee"
CARD = "#f2f4f6"

FOV_HALF = 0.720            # radians; the wedge below is drawn from this
FOV_DIR = -0.075            # optical axis, relative to horizontal

# Scanpath, in reading order.  ``anc`` entries hang off an object the
# renderer really placed; ``frac`` entries are frame fractions inside the
# aisle band, which the store backdrop fills edge to edge with stock, so they
# cannot land on blank wall either.
SCAN = [("frac", (0.175, 0.185), 240),
        ("frac", (0.470, 0.130), 330),
        ("anc", ("carton", (-0.05, 0.10)), 190),
        ("anc", ("can", (0.05, -0.15)), 470),
        ("frac", (0.845, 0.290), 210),
        ("anc", ("packet", (-0.20, -0.55)), 540)]


def head(ax, cx, cy, s):
    """A profile head with the rig on it; returns the camera position."""
    prof = np.array([
        (-0.60, -0.78), (-0.74, -0.34), (-0.72, 0.16), (-0.54, 0.60),
        (-0.20, 0.85), (0.14, 0.83), (0.40, 0.60), (0.50, 0.26),
        (0.56, 0.07), (0.67, -0.03), (0.57, -0.11), (0.55, -0.27),
        (0.44, -0.33), (0.47, -0.47), (0.31, -0.61), (0.03, -0.70),
        (-0.28, -0.74)])
    ax.add_patch(Polygon(np.stack([cx + s * prof[:, 0], cy + s * prof[:, 1]],
                                  axis=1), closed=True, facecolor="#c9ced4",
                         edgecolor=RULE, lw=1.0, zorder=3))
    ax.add_patch(Polygon(np.array([[cx - 0.34 * s, cy - 0.72 * s],
                                   [cx + 0.16 * s, cy - 0.70 * s],
                                   [cx + 0.22 * s, cy - 1.10 * s],
                                   [cx - 0.42 * s, cy - 1.10 * s]]),
                         closed=True, facecolor="#b7bdc4", edgecolor=RULE,
                         lw=0.9, zorder=2))
    ax.add_patch(Ellipse((cx - 0.09 * s, cy - 0.02 * s), 0.20 * s, 0.26 * s,
                         facecolor="#b3b9c0", edgecolor=RULE, lw=0.8,
                         zorder=4))
    # temple arm, lens, forward camera, eye camera
    ax.plot([cx - 0.16 * s, cx + 0.46 * s], [cy + 0.10 * s, cy + 0.17 * s],
            color=DEEP, lw=2.0, solid_capstyle="round", zorder=5)
    ax.add_patch(FancyBboxPatch((cx + 0.30 * s, cy + 0.05 * s), 0.24 * s,
                                0.16 * s,
                                boxstyle="round,pad=0,rounding_size=" +
                                         str(0.04 * s),
                                facecolor="#dbe8f2", edgecolor=DEEP, lw=1.0,
                                alpha=0.85, zorder=5))
    camx, camy = cx + 0.54 * s, cy + 0.22 * s
    ax.add_patch(Circle((camx, camy), 0.075 * s, facecolor=DEEP,
                        edgecolor="#ffffff", lw=0.8, zorder=7))
    ax.add_patch(Circle((cx + 0.31 * s, cy - 0.03 * s), 0.052 * s,
                        facecolor="#ffffff", edgecolor=DEEP, lw=0.9,
                        zorder=7))
    ax.plot([cx + 0.31 * s, cx + 0.19 * s], [cy - 0.03 * s, cy + 0.02 * s],
            color=DEEP, lw=0.8, ls=(0, (2.0, 1.6)), zorder=6)
    return camx, camy


def frustum(ax, camx, camy, reach):
    a0, a1 = FOV_DIR - FOV_HALF, FOV_DIR + FOV_HALF
    p0 = (camx + reach * np.cos(a0), camy + reach * np.sin(a0))
    p1 = (camx + reach * np.cos(a1), camy + reach * np.sin(a1))
    ax.add_patch(Polygon([(camx, camy), p0, p1], closed=True,
                         facecolor=PALE, edgecolor="none", alpha=0.75,
                         zorder=1))
    for p in (p0, p1):
        ax.plot([camx, p[0]], [camy, p[1]], color=MID, lw=0.9,
                ls=(0, (3.4, 2.2)), zorder=2)
    g = FOV_DIR - 0.30
    ax.add_patch(FancyArrowPatch(
        (camx, camy),
        (camx + 0.92 * reach * np.cos(g), camy + 0.92 * reach * np.sin(g)),
        arrowstyle="-|>", mutation_scale=8, color=DEEP, lw=1.3, zorder=6))
    return np.degrees(2 * FOV_HALF)


def mini_density(ax, x0, y0, w_, h_):
    ax.add_patch(FancyBboxPatch((x0, y0), w_, h_,
                                boxstyle="round,pad=0,rounding_size=" +
                                         str(0.10 * h_),
                                facecolor="#e6edf3", edgecolor="#9fb4c6",
                                lw=0.7, zorder=4))
    for k, rad in enumerate((0.40, 0.27, 0.15)):
        ax.add_patch(Ellipse((x0 + 0.62 * w_, y0 + 0.55 * h_), 2 * rad * w_,
                             2 * rad * h_ * 1.05,
                             facecolor=[PALE, "#8fb8d6", MID][k],
                             edgecolor="none", alpha=0.9, zorder=5 + k))


def mini_scan(ax, x0, y0, w_, h_):
    ax.add_patch(FancyBboxPatch((x0, y0), w_, h_,
                                boxstyle="round,pad=0,rounding_size=" +
                                         str(0.10 * h_),
                                facecolor="#e6edf3", edgecolor="#9fb4c6",
                                lw=0.7, zorder=4))
    px = np.array([0.20, 0.44, 0.66]) * w_ + x0
    py = np.array([0.30, 0.66, 0.40]) * h_ + y0
    ax.plot(px, py, color=MID, lw=1.0, zorder=5)
    ax.scatter(px, py, s=16, facecolors="#ffffff", edgecolors=DEEP,
               linewidths=0.9, zorder=6)
    ax.add_patch(FancyArrowPatch((px[-1], py[-1]),
                                 (x0 + 0.88 * w_, y0 + 0.72 * h_),
                                 arrowstyle="-|>", mutation_scale=7,
                                 color=DEEP, lw=1.1, ls=(0, (2.2, 1.6)),
                                 zorder=6))
    ax.add_patch(Circle((x0 + 0.88 * w_, y0 + 0.72 * h_), 0.085 * w_,
                        facecolor="none", edgecolor=DEEP, lw=0.9,
                        ls=(0, (1.8, 1.4)), zorder=7))


def task_card(ax, x0, y0, w_, h_, tag, title, body, glyph):
    ax.add_patch(FancyBboxPatch((x0, y0), w_, h_,
                                boxstyle="round,pad=0,rounding_size=" +
                                         str(0.055 * h_),
                                facecolor="#ffffff", edgecolor="#b9c6d3",
                                lw=0.9, zorder=3))
    ax.add_patch(FancyBboxPatch((x0, y0 + 0.70 * h_), w_, 0.30 * h_,
                                boxstyle="round,pad=0,rounding_size=" +
                                         str(0.055 * h_),
                                facecolor=DEEP, edgecolor="none", zorder=4))
    ax.add_patch(FancyBboxPatch((x0, y0 + 0.70 * h_), w_, 0.10 * h_,
                                boxstyle="square,pad=0", facecolor=DEEP,
                                edgecolor="none", zorder=4))
    ax.text(x0 + 0.035 * w_, y0 + 0.845 * h_, tag, fontsize=6.6,
            color="#a9c8dd", ha="left", va="center", zorder=6)
    ax.text(x0 + 0.175 * w_, y0 + 0.845 * h_, title, fontsize=8.6,
            fontweight="bold", color="#ffffff", ha="left", va="center",
            zorder=6)
    ax.text(x0 + 0.035 * w_, y0 + 0.52 * h_, body, fontsize=7.0, color=SUB,
            ha="left", va="top", zorder=6, linespacing=1.42)
    glyph(ax, x0 + 0.735 * w_, y0 + 0.085 * h_, 0.235 * w_, 0.30 * h_)


def main():
    set_pub_style(base_font=8)
    FW, FH = 11.0, 4.4

    sc = build("ego", seed=6601, shape=(H, W), room="store",
               cam=dict(exposure=1.05, wb=(0.962, 1.00, 1.078), dof=1.0,
                        contrast=1.17, vignette=0.30, sharpen=1.30))
    anc = {a["name"]: a for a in sc["anchors"]}
    pts = []
    for how, spec, _ in SCAN:
        if how == "frac":
            pts.append([spec[0] * W, spec[1] * H])
        else:
            n, (ox, oy) = spec
            pts.append([anc[n]["cx"] + ox * anc[n]["rx"],
                        anc[n]["cy"] + oy * anc[n]["ry"]])
    pts = np.array(pts)
    dwell = np.array([d for _, _, d in SCAN], dtype=float)
    amp = np.hypot(*(pts[1:] - pts[:-1]).T) / W * np.degrees(2 * FOV_HALF)

    fig = plt.figure(figsize=(FW, FH))
    rects = [(0.012, 0.075, 0.240, 0.720),
             (0.278, 0.075, 0.318, 0.720),
             (0.628, 0.075, 0.364, 0.720)]
    for r in rects:
        background_card(fig, r, facecolor=CARD, radius=0.010, zorder=-5)

    heads = ["(a)  Capture rig", "(b)  One recorded clip",
             "(c)  Two prediction tasks"]
    for (x0, y0, w_, h_), t in zip(rects, heads):
        fig.text(x0 + 0.010, y0 + h_ - 0.014, t, fontsize=8.4,
                 fontweight="bold", color=INK, ha="left", va="top")

    # ---- (a) rig ----------------------------------------------------------
    x0, y0, w_, h_ = rects[0]
    axa = fig.add_axes([x0 + 0.008, y0 + 0.030, w_ - 0.016, h_ - 0.105])
    ar = (w_ - 0.016) * FW / ((h_ - 0.105) * FH)
    axa.set_xlim(0, ar)
    axa.set_ylim(0, 1)
    axa.set_axis_off()
    axa.patch.set_visible(False)
    camx, camy = head(axa, 0.30 * ar, 0.56, 0.30)
    fov = frustum(axa, camx, camy, 0.62 * ar)
    axa.text(camx + 0.30 * ar, 0.905, f"{fov:.0f}\u00b0 field of view",
             fontsize=7.2, color=DEEP, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff",
                       edgecolor="#b9c6d3", lw=0.6))
    axa.annotate("scene camera", xy=(camx, camy), xytext=(0.30 * ar, 0.965),
                 fontsize=6.8, color=SUB, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED,
                                 shrinkA=1, shrinkB=3))
    axa.annotate("eye camera", xy=(0.30 * ar + 0.31 * 0.30, 0.56 - 0.009),
                 xytext=(0.135 * ar, 0.150), fontsize=6.8, color=SUB,
                 ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED,
                                 shrinkA=1, shrinkB=3))
    # the lower edge of the wedge runs straight through this label, so it
    # needs the same plate the field-of-view chip above it already has
    axa.text(camx + 0.34 * ar, 0.375, "gaze vector\n$g_t$", fontsize=7.2,
             color=DEEP, ha="center", va="center", linespacing=1.35, zorder=8,
             bbox=dict(boxstyle="round,pad=0.26", facecolor="#ffffff",
                       edgecolor="none", alpha=0.92))
    axa.text(0.36 * ar, 0.030,
             "scene camera 1080p @ 30 Hz\neye camera 120 Hz, both clocks\n"
             "aligned to the scene shutter", fontsize=6.4, color=MUTED,
             ha="left", va="bottom", linespacing=1.45)

    # ---- (b) clip ---------------------------------------------------------
    x0, y0, w_, h_ = rects[1]
    axb = fig.add_axes([x0 + 0.010, y0 + 0.030, w_ - 0.020, h_ - 0.105])
    image_axes(axb, color="#9a9a9a", lw=0.6)
    axb.set_aspect("auto")
    axb.imshow(sc["rgb"], interpolation="bilinear", aspect="auto")
    for k in range(len(pts) - 1):
        axb.add_patch(FancyArrowPatch(pts[k], pts[k + 1], arrowstyle="-|>",
                                      mutation_scale=8, color="#14202c",
                                      lw=1.9, ls="-", shrinkA=7, shrinkB=9,
                                      zorder=5))
        axb.add_patch(FancyArrowPatch(pts[k], pts[k + 1], arrowstyle="-|>",
                                      mutation_scale=7, color="#dbe8f2",
                                      lw=0.9, shrinkA=7, shrinkB=9,
                                      zorder=6))
    for k, (px, py) in enumerate(pts):
        r = 4.2 + 0.030 * dwell[k]
        last = k == len(pts) - 1
        axb.add_patch(Circle((px, py), r, facecolor="none",
                             edgecolor="#14202c", lw=2.4, zorder=7))
        axb.add_patch(Circle((px, py), r, facecolor="#ffffff", alpha=0.30,
                             edgecolor=PALE if last else "#ffffff", lw=1.1,
                             ls=(0, (2.2, 1.6)) if last else "-", zorder=8))
        axb.text(px, py, f"{k + 1}", fontsize=6.6, fontweight="bold",
                 color="#ffffff", ha="center", va="center", zorder=9)
    axb.set_xlim(0, W)
    axb.set_ylim(H, 0)
    axb.text(0.020, 0.965,
             f"{len(pts)} fixations  ·  {dwell.sum() / 1000:.2f} s dwell  ·  "
             f"{len(amp)} saccades, mean {amp.mean():.1f}\u00b0",
             transform=axb.transAxes, fontsize=6.3, color="#20344a",
             ha="left", va="top", zorder=10,
             bbox=dict(boxstyle="round,pad=0.26", facecolor="#ffffff",
                       edgecolor="#b9c6d3", lw=0.5, alpha=0.92))
    axb.text(0.020, 0.035, "ring area grows with dwell;  dashed ring is the "
             "target of task c-2", transform=axb.transAxes, fontsize=6.2,
             color="#20344a", ha="left", va="bottom", zorder=10,
             bbox=dict(boxstyle="round,pad=0.26", facecolor="#ffffff",
                       edgecolor="#b9c6d3", lw=0.5, alpha=0.92))

    # ---- (c) tasks --------------------------------------------------------
    x0, y0, w_, h_ = rects[2]
    axc = fig.add_axes([x0 + 0.010, y0 + 0.030, w_ - 0.020, h_ - 0.105])
    arc = (w_ - 0.020) * FW / ((h_ - 0.105) * FH)
    axc.set_xlim(0, arc)
    axc.set_ylim(0, 1)
    axc.set_axis_off()
    axc.patch.set_visible(False)
    task_card(axc, 0.02 * arc, 0.545, 0.96 * arc, 0.415, "c-1",
              "Gaze estimation",
              "Given the frames up to $t$, predict a\n"
              "fixation density over frame $t$.\n"
              "Scored by AUC-Judd against the\n"
              "pooled observer fixations.", mini_density)
    task_card(axc, 0.02 * arc, 0.055, 0.96 * arc, 0.415, "c-2",
              "Saccade prediction",
              "Given fixations 1..$k$, predict where\n"
              "fixation $k{+}1$ lands and when.\n"
              "Scored by angular error and by\n"
              "onset latency.", mini_scan)

    # ---- narrative arrows: one in, then a right-angle fork out ------------
    fig.add_artist(FancyArrowPatch((0.256, 0.435), (0.274, 0.435),
                                   transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=11,
                                   color=RULE, lw=1.1))
    bus = Line2D([0.601, 0.611, 0.611], [0.435, 0.435, 0.435],
                 transform=fig.transFigure, color=RULE, lw=1.1)
    fig.add_artist(bus)
    fig.add_artist(Line2D([0.611, 0.611], [0.268, 0.600],
                          transform=fig.transFigure, color=RULE, lw=1.1))
    for ytgt in (0.600, 0.268):
        fig.add_artist(FancyArrowPatch((0.611, ytgt), (0.626, ytgt),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=11,
                                       color=RULE, lw=1.1))

    savefig(fig, "1161_wm_ego_problem_setup.png")


if __name__ == "__main__":
    main()
