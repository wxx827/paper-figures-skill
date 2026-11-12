"""Four families of action space, drawn as spaces, then scored on benchmarks.

The upper row gives each family its own miniature of the *space itself* -- a
lattice of atomic choices, a bounded continuous region with a sampled policy, a
discrete switch feeding continuous sub-ranges, and a discrete set whose members
each carry a parameter vector.  The lower matrix scores the same four families
across eight benchmark suites, and the family shown in a card and the family
shown in a matrix row are tied together by one hatch pattern, not by a second
hue: the whole figure runs on a single blue ladder.

Every row of the matrix is built from scores that are multiples of five and
sums to a multiple of forty, so the row mean printed on the right is exact and
the reader can reproduce it from the eight integers next to it.  The addition
for one row is printed in full at the foot of the figure, and the script
asserts the closure before drawing.

Single-hue blue ladder (BRIEF_WM_BLUE.md section 2); the one warm accent marks
only the best cell of each row.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from pubstyle import set_pub_style, savefig
from figkit import inch_axes

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
LIGHT = "#f4f8fb"           # luminance 0.97: card bodies stay off the ink count
HAIR = "#aab7c4"

CELL_CMAP = LinearSegmentedColormap.from_list(
    "wm_suit", [BLUE[7], BLUE[6], BLUE[5], BLUE[3], BLUE[1], BLUE[0]], N=256)

FW, FH = 10.4, 6.4

# family name, hatch, one-line definition, cardinality note
FAMILIES = [
    ("Discrete", "", "a finite lattice of atomic choices",
     "$\\mathcal{A} = \\{a_1 \\ldots a_{30}\\}$"),
    ("Continuous", "....", "a bounded region of a real vector space",
     "$\\mathcal{A} \\subseteq \\mathbb{R}^{2}$, box-bounded"),
    ("Hybrid", "///", "a discrete switch over continuous sub-ranges",
     "$\\mathcal{A} = \\{1,2,3\\} \\times \\mathbb{R}$"),
    ("Parameterised", "xxx", "each discrete action carries its own parameters",
     "$\\mathcal{A} = \\bigcup_k \\{a_k\\} \\times \\mathcal{X}_k$"),
]

BENCH = ["GridWorld", "Atari-57", "StarCraft II", "MuJoCo",
         "Meta-World", "CARLA", "RoboSuite", "Minecraft"]
# suitability 0-100; every row is eight multiples of five summing to a
# multiple of forty, which is what makes the printed row mean exact
SCORES = np.array([
    [95, 95, 80, 35, 40, 45, 40, 90],       # discrete            -> 520 / 8
    [30, 25, 30, 95, 90, 90, 95, 25],       # continuous          -> 480 / 8
    [70, 65, 90, 75, 80, 90, 80, 90],       # hybrid              -> 640 / 8
    [65, 60, 95, 60, 75, 70, 70, 65],       # parameterised       -> 560 / 8
], dtype=int)
ADAPTER = SCORES < 50       # these need a discretisation / relaxation adapter


# ---------------------------------------------------------------------------
# The four miniatures.  Each takes a rectangle in figure inches.
# ---------------------------------------------------------------------------
def mini_discrete(ax, x, y, w, h, rng):
    nc, nr = 6, 5
    s = min(w / nc, h / nr) * 0.80
    gx = (w - nc * s) / (nc + 1)
    gy = (h - nr * s) / (nr + 1)
    q = rng.uniform(0.15, 1.0, (nr, nc))
    best = np.unravel_index(int(np.argmax(q)), q.shape)
    for r in range(nr):
        for c in range(nc):
            cx = x + gx + c * (s + gx)
            cy = y + h - gy - s - r * (s + gy)
            ax.add_patch(Rectangle((cx, cy), s, s,
                                   facecolor=CELL_CMAP(0.12 + 0.72 * q[r, c]),
                                   edgecolor="white", linewidth=0.7, zorder=4))
            if (r, c) == best:
                ax.add_patch(Rectangle((cx - 0.018, cy - 0.018), s + 0.036,
                                       s + 0.036, facecolor="none",
                                       edgecolor=ACCENT, linewidth=1.5,
                                       zorder=6))
    return "argmax over %d atomic actions" % (nr * nc)


def mini_continuous(ax, x, y, w, h, rng):
    frame = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0,rounding_size=0.05",
                           facecolor="white", edgecolor=HAIR, linewidth=0.8,
                           mutation_aspect=1.0, zorder=3)
    ax.add_patch(frame)
    gx, gy = np.meshgrid(np.linspace(x, x + w, 130),
                         np.linspace(y, y + h, 90))
    mu = (x + 0.60 * w, y + 0.56 * h)
    z = np.exp(-(((gx - mu[0]) / (0.24 * w)) ** 2 +
                 ((gy - mu[1]) / (0.30 * h)) ** 2))
    z += 0.45 * np.exp(-(((gx - x - 0.26 * w) / (0.17 * w)) ** 2 +
                         ((gy - y - 0.34 * h) / (0.22 * h)) ** 2))
    cf = ax.contourf(gx, gy, z, levels=9, cmap=CELL_CMAP, zorder=3.5)
    cl = ax.contour(gx, gy, z, levels=9, colors=BLUE[2], linewidths=0.35,
                    zorder=3.6)
    for art in (cf, cl):
        art.set_clip_path(frame)
    pts = rng.normal(0.0, 1.0, (44, 2))
    px = np.clip(mu[0] + 0.19 * w * pts[:, 0], x + 0.03, x + w - 0.03)
    py = np.clip(mu[1] + 0.24 * h * pts[:, 1], y + 0.03, y + h - 0.03)
    ax.plot(px, py, ls="none", marker="o", ms=2.0, mfc="white", mec=BLUE[0],
            mew=0.5, zorder=5)
    ax.plot([mu[0]], [mu[1]], marker="+", ms=7.5, mew=1.6, color=ACCENT,
            ls="none", zorder=6)
    return "%d samples from $\\pi(a|s)$" % len(px)


def mini_hybrid(ax, x, y, w, h, rng):
    n = 3
    cw = w / (n + 0.6)
    top = y + h - 0.20
    for k in range(n):
        cx = x + 0.30 * cw + k * cw * 1.10
        ax.add_patch(FancyBboxPatch(
            (cx, top - 0.20), cw * 0.80, 0.20,
            boxstyle="round,pad=0,rounding_size=0.035",
            facecolor=BLUE[1 + k], edgecolor="white", linewidth=0.7,
            mutation_aspect=1.0, zorder=5))
        ax.text(cx + cw * 0.40, top - 0.10, "$k$=%d" % (k + 1), ha="center",
                va="center", fontsize=6.2, color="white", zorder=6)
        by = y + 0.26
        ax.plot([cx + cw * 0.40, cx + cw * 0.40], [top - 0.20, by + 0.13],
                color=BLUE[2], lw=0.9, zorder=4)
        lo, hi = rng.uniform(0.05, 0.35), rng.uniform(0.62, 0.96)
        ax.add_patch(Rectangle((cx, by), cw * 0.80, 0.13, facecolor="white",
                               edgecolor=HAIR, linewidth=0.7, zorder=5))
        ax.add_patch(Rectangle((cx + lo * cw * 0.80, by),
                               (hi - lo) * cw * 0.80, 0.13,
                               facecolor=BLUE[3], edgecolor="none", zorder=6))
        ax.text(cx + cw * 0.40, by - 0.050, "[%.2f, %.2f]" % (lo, hi),
                ha="center", va="top", fontsize=6.0, color="#4b5866",
                zorder=6)
    return "one switch, %d continuous ranges" % n


def mini_parameterised(ax, x, y, w, h, rng):
    names = ["grasp", "push", "place", "wait"]
    rh = h / len(names)
    for i, nm in enumerate(names):
        cy = y + h - (i + 0.5) * rh
        ax.add_patch(FancyBboxPatch(
            (x + 0.02, cy - 0.085), 0.52, 0.17,
            boxstyle="round,pad=0,rounding_size=0.035",
            facecolor=BLUE[1] if i == 0 else BLUE[4], edgecolor="white",
            linewidth=0.7, mutation_aspect=1.0, zorder=5))
        ax.text(x + 0.28, cy, nm, ha="center", va="center", fontsize=6.0,
                color="white" if i == 0 else INK, zorder=6)
        tx0, tx1 = x + 0.64, x + w - 0.10
        ax.plot([tx0, tx1], [cy, cy], color=BLUE[5], lw=2.6,
                solid_capstyle="round", zorder=4)
        v = rng.uniform(0.18, 0.86)
        ax.plot([tx0, tx0 + v * (tx1 - tx0)], [cy, cy], color=BLUE[2], lw=2.6,
                solid_capstyle="round", zorder=5)
        ax.add_patch(Circle((tx0 + v * (tx1 - tx0), cy), 0.048,
                            facecolor="white",
                            edgecolor=ACCENT if i == 0 else BLUE[1],
                            linewidth=1.2, zorder=7))
    return "%d actions, one parameter each" % len(names)


MINIS = [mini_discrete, mini_continuous, mini_hybrid, mini_parameterised]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(1614)

    row_sum = SCORES.sum(axis=1)
    row_mean = row_sum / 8.0
    for s, m in zip(row_sum, row_mean):
        assert s % 8 == 0, "row sum must divide by eight for an exact mean"
        assert abs(float("%.1f" % m) - s / 8.0) < 1e-9, "row mean not closed"
    hyb = SCORES[2]
    add_txt = " + ".join(str(v) for v in hyb)
    assert sum(hyb) == row_sum[2] and row_sum[2] / 8 == row_mean[2]

    fig = plt.figure(figsize=(FW, FH))

    # ---- (a) the four spaces ----------------------------------------------
    axT = inch_axes(fig, (0.0, 3.34, FW, 2.62), absolute=True)
    cw, cg = 2.36, 0.16
    x0 = (FW - 4 * cw - 3 * cg) / 2.0
    cy0, ch = 3.44, 2.36
    for i, (name, hatch, blurb, card) in enumerate(FAMILIES):
        cx = x0 + i * (cw + cg)
        axT.add_patch(FancyBboxPatch(
            (cx, cy0), cw, ch, boxstyle="round,pad=0,rounding_size=0.065",
            facecolor=LIGHT, edgecolor=HAIR, linewidth=0.9,
            mutation_aspect=1.0, zorder=2))
        axT.add_patch(FancyBboxPatch(
            (cx, cy0 + ch - 0.34), cw, 0.34,
            boxstyle="round,pad=0,rounding_size=0.065", facecolor=BLUE[1],
            edgecolor=BLUE[1], linewidth=0.9, mutation_aspect=1.0, zorder=3))
        # the hatch lives in a swatch, not across the whole header: it has to
        # be readable next to the row-label swatch below, and white bold type
        # cannot sit on a hatch
        axT.add_patch(Rectangle((cx + 0.10, cy0 + ch - 0.275), 0.28, 0.21,
                                facecolor=BLUE[6], edgecolor=BLUE[0],
                                hatch=hatch, linewidth=0.7, zorder=4))
        axT.text(cx + 0.10 + 0.28 + (cw - 0.38) / 2.0, cy0 + ch - 0.17,
                 "(%s)  %s" % ("abcd"[i], name), ha="center", va="center",
                 fontsize=8.4, fontweight="bold", color="white", zorder=5)
        note = MINIS[i](axT, cx + 0.11, cy0 + 0.62, cw - 0.22, 1.32, rng)
        axT.text(cx + cw / 2.0, cy0 + 0.44, card, ha="center", va="center",
                 fontsize=7.2, color=BLUE[0], zorder=5)
        axT.text(cx + cw / 2.0, cy0 + 0.255, blurb, ha="center", va="center",
                 fontsize=6.7, color="#4b5866", zorder=5)
        axT.text(cx + cw / 2.0, cy0 + 0.105, note, ha="center", va="center",
                 fontsize=6.3, color="#6f7a85", style="italic", zorder=5)

    # ---- (b) suitability matrix -------------------------------------------
    axM = fig.add_axes([1.95 / FW, 0.90 / FH, 7.40 / FW, 1.90 / FH])
    axM.imshow(SCORES, cmap=CELL_CMAP, vmin=15, vmax=105, aspect="auto",
               extent=(-0.5, 7.5, 3.5, -0.5), interpolation="nearest",
               zorder=2)
    best_col = SCORES.argmax(axis=1)
    for r in range(4):
        for c in range(8):
            v = SCORES[r, c]
            axM.text(c, r, "%d" % v, ha="center", va="center", fontsize=7.4,
                     color="white" if v >= 62 else INK, zorder=6,
                     fontweight="bold" if c == best_col[r] else "normal")
            if ADAPTER[r, c]:
                axM.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                        facecolor="none", edgecolor=BLUE[2],
                                        hatch="\\\\\\", linewidth=0.0,
                                        zorder=3))
            if c == best_col[r]:
                axM.add_patch(Rectangle((c - 0.46, r - 0.46), 0.92, 0.92,
                                        facecolor="none", edgecolor=ACCENT,
                                        linewidth=1.4, zorder=7))
        axM.add_patch(Rectangle((-0.5, r - 0.5), 8, 1, facecolor="none",
                                edgecolor="white", linewidth=1.0, zorder=5))

    axM.set_xticks([])
    axM.set_yticks([])
    for sp in axM.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
        sp.set_color("#7f8c99")
    for c, nm in enumerate(BENCH):
        axM.text(c, -0.62, nm, ha="center", va="bottom", fontsize=7.0,
                 color=INK, clip_on=False)
    for r, (name, hatch, _, _) in enumerate(FAMILIES):
        axM.add_patch(Rectangle((-1.78, r - 0.19), 0.30, 0.38,
                                facecolor=BLUE[6], edgecolor=BLUE[1],
                                hatch=hatch, linewidth=0.8, zorder=6,
                                clip_on=False))
        axM.text(-1.38, r, name, ha="left", va="center", fontsize=7.6,
                 color=INK, clip_on=False)
        axM.text(8.16, r, "%.0f" % row_mean[r], ha="center", va="center",
                 fontsize=7.6, fontweight="bold", color=BLUE[0],
                 clip_on=False)
    axM.text(8.16, -0.62, "row mean", ha="center", va="bottom", fontsize=7.0,
             color=INK, clip_on=False)
    axM.text(-1.78, -0.62, "action-space family", ha="left", va="bottom",
             fontsize=7.0, color=INK, clip_on=False)
    axM.set_title("Suitability of each family on eight benchmark suites "
                  "(0-100; hatched cells need a discretisation or relaxation "
                  "adapter)", fontsize=8.6, fontweight="bold", color=INK,
                  pad=16.0)

    cax = fig.add_axes([1.95 / FW, 0.50 / FH, 1.70 / FW, 0.085 / FH])
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=CELL_CMAP, norm=mpl.colors.Normalize(15, 105),
        orientation="horizontal", ticks=[25, 60, 95])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(INK)
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, color=INK, pad=1.2)
    cb.set_label("suitability score", fontsize=6.6, color=INK, labelpad=1.6)

    n_adapt = int(ADAPTER.sum())
    fig.text(4.05 / FW, 0.545 / FH, "hybrid row:  %s = %d,   / 8 = %.0f"
             "        %d of %d cells hatched"
             % (add_txt, row_sum[2], row_mean[2], n_adapt, SCORES.size),
             fontsize=7.0, color=INK, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#c3ccd4",
                       lw=0.6, alpha=0.96))

    savefig(fig, "1209_wm_plan_action_taxonomy.png")


if __name__ == "__main__":
    main()
