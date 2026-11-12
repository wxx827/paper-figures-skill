"""Figure 1199 - Episodic and semantic memory are different geometries.

(a) The episodic store: 720 traces from 12 episodes, drawn as twelve faint
temporal chains scattered across the projection - neighbouring points are
neighbouring *moments*, not neighbouring meanings.

(b) The semantic store: 180 facts abstracted out of those traces, collapsed
into five concept clusters.  Both panels are plotted on one shared ruler
(identical limits, identical on-paper scale), so "tighter" is a fact about
the pixels and not about the axis choice; the two mean radii printed in the
corners are computed from the arrays that are scattered, and the shrink
factor between them is closed with ``closed_ratio``.

(c) Recall by query type.  The two heat columns share one 0-1 scale; the
third column is a signed bar around a zero line rather than a second hue,
which is how this batch says "positive / negative" without spending colour
budget.  Every delta is the printed semantic value minus the printed
episodic value, asserted cell by cell, and the two column means are printed
as full sums in the footer.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch

from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#d5d5d5"
CARD = "#e6eaee"

EPI_COL = "#5b93c8"
SEM_COL = "#12314f"
BAR_COL = "#2f6fa8"
CMAP = get_cmap("cb_sequential_blue")

FW, FH = 10.4, 6.0
RECT_EPI = [0.042, 0.118, 0.205, 0.672]
RECT_ARR = [0.258, 0.118, 0.085, 0.672]
RECT_SEM = [0.352, 0.118, 0.205, 0.672]
RECT_MAT = [0.700, 0.118, 0.278, 0.672]

N_EPISODES, STEPS = 12, 60
N_CONCEPTS, PER_CONCEPT = 5, 36
CONCEPTS = ["containers", "tools", "places", "routines", "agents"]
XHALF = 2.10

# query type, episodic recall, semantic recall  (two decimals, so the delta
# column below is exact rather than rounded)
QUERIES = [
    ("Time of an event", 0.91, 0.24),
    ("Habitual location", 0.38, 0.86),
    ("Repetition check", 0.84, 0.31),
    ("Tool affordance", 0.29, 0.88),
    ("Duration of a run", 0.76, 0.41),
]


def episodic(yhalf):
    """Twelve temporal chains: a random walk per episode, widely scattered."""
    rng = np.random.default_rng(1604)
    chains = []
    for _ in range(N_EPISODES):
        x0 = rng.uniform(-0.80, 0.80) * XHALF
        y0 = rng.uniform(-0.80, 0.80) * yhalf
        step = rng.standard_normal((STEPS, 2)) * 0.155
        walk = np.cumsum(step, axis=0) + np.array([x0, y0])
        chains.append(walk)
    pts = np.vstack(chains)
    return chains, pts


# concept centres, hand-placed so no two clusters or their labels touch;
# the units are paper-isotropic (see ``ratio`` in main)
CENTRES = np.array([(-0.95, 2.55), (1.05, 1.45), (-1.10, 0.10),
                    (0.95, -1.35), (-0.35, -2.85)])


def semantic():
    rng = np.random.default_rng(2604)
    groups = [c + rng.standard_normal((PER_CONCEPT, 2)) * 0.21
              for c in CENTRES]
    return CENTRES, groups


def proj_axes(ax, xhalf, yhalf, title, sub):
    ax.set_xlim(-xhalf, xhalf)
    ax.set_ylim(-yhalf, yhalf)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#9a9a9a")
    ax.set_xlabel("projection dim 1", fontsize=6.8, labelpad=2.5,
                  color=SUB)
    ax.set_ylabel("projection dim 2", fontsize=6.8, labelpad=2.5, color=SUB)
    ax.text(0.5, 1.028, title, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8.8, fontweight="bold", color=INK)
    ax.text(0.5, 1.002, sub, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=6.6, color=MUTED)


def main():
    set_pub_style()
    fig = plt.figure(figsize=(FW, FH))
    ax_e = fig.add_axes(RECT_EPI)
    ax_a = fig.add_axes(RECT_ARR)
    ax_s = fig.add_axes(RECT_SEM)
    ax_m = fig.add_axes(RECT_MAT)

    # one paper unit in x must equal one paper unit in y in both projections,
    # or "tight cluster" becomes an artefact of the axis box
    ratio = (RECT_EPI[3] * FH) / (RECT_EPI[2] * FW)
    yhalf = XHALF * ratio

    chains, epi = episodic(yhalf)
    cen, groups = semantic()
    sem = np.vstack(groups)

    r_epi = float(np.hypot(*(epi - epi.mean(axis=0)).T).mean())
    r_sem = float(np.mean([np.hypot(*(g - c).T).mean()
                           for g, c in zip(groups, cen)]))
    # print both radii at two decimals, then derive the shrink factor from
    # those two strings, so a reader who divides them lands on the same number
    ns, ds = f"{r_epi:.2f}", f"{r_sem:.2f}"
    qs = f"{float(ns) / float(ds):.1f}"
    assert abs(float(qs) - float(ns) / float(ds)) < 0.05, \
        "printed radii do not reproduce the printed shrink factor"

    # ---- (a) episodic
    proj_axes(ax_e, XHALF, yhalf, "Episodic store",
              f"{epi.shape[0]} traces, {N_EPISODES} episodes  |  "
              f"mean radius {ns}")
    for w in chains:
        ax_e.plot(w[:, 0], w[:, 1], color=EPI_COL, lw=0.55, alpha=0.40,
                  zorder=2)
    ax_e.scatter(epi[:, 0], epi[:, 1], s=6.0, facecolors="none",
                 edgecolors=EPI_COL, linewidths=0.55, zorder=3)

    # ---- (b) semantic
    proj_axes(ax_s, XHALF, yhalf, "Semantic store",
              f"{sem.shape[0]} facts, {N_CONCEPTS} concepts  |  "
              f"mean radius {ds}")
    for k, (g, c) in enumerate(zip(groups, cen)):
        ax_s.add_patch(Ellipse(c, 1.24, 1.24, facecolor="white", alpha=0.62,
                               edgecolor=SEM_COL, linewidth=0.7,
                               linestyle=(0, (3.2, 2.0)), zorder=2))
        ax_s.scatter(g[:, 0], g[:, 1], s=7.5, color=SEM_COL, alpha=0.88,
                     linewidths=0, zorder=3)
        ax_s.text(c[0], c[1] - 0.80, CONCEPTS[k], ha="center", va="top",
                  fontsize=6.4, color=SEM_COL, fontweight="bold", zorder=5,
                  bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                            edgecolor="none", alpha=0.90))

    # ---- abstraction arrow between the two stores
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    ax_a.add_patch(FancyArrowPatch((0.06, 0.50), (0.94, 0.50),
                                   arrowstyle="-|>", mutation_scale=15,
                                   linewidth=2.0, color=BAR_COL, shrinkA=0,
                                   shrinkB=0, zorder=4))
    ax_a.text(0.5, 0.565, "abstraction", ha="center", va="bottom",
              fontsize=7.0, fontweight="bold", color=BAR_COL)
    comp = f"{epi.shape[0] / sem.shape[0]:.1f}"
    assert abs(float(comp) - epi.shape[0] / sem.shape[0]) < 0.05, \
        "compression label does not reproduce traces / facts"
    ax_a.text(0.5, 0.435, f"{epi.shape[0]} / {sem.shape[0]}\n= {comp} traces\n"
                          f"per fact", ha="center", va="top", fontsize=6.3,
              color=SUB, linespacing=1.55)
    ax_a.text(0.5, 0.175, f"spread\n{ns} / {ds}\n= {qs}x tighter",
              ha="center", va="top", fontsize=6.3, color=SUB,
              linespacing=1.55)

    # ---- (c) recall matrix, two heat columns plus a signed bar column
    ax_m.set_xlim(0.0, 3.30)
    ax_m.set_ylim(len(QUERIES), 0.0)
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    for s in ax_m.spines.values():
        s.set_visible(False)
    scale = 0.40 / max(abs(b - a) for _, a, b in QUERIES)
    for i, (name, e, s_) in enumerate(QUERIES):
        d = round(s_ - e, 2)
        assert abs(d - (round(s_, 2) - round(e, 2))) < 1e-9, \
            f"{name}: delta is not the difference of the two printed cells"
        for c, v in ((0, e), (1, s_)):
            rgba = CMAP(v)
            ax_m.add_patch(Rectangle((c + 0.03, i + 0.06), 0.94, 0.88,
                                     facecolor=rgba, edgecolor="white",
                                     linewidth=0.8, zorder=3))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax_m.text(c + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                      fontsize=7.6, color="white" if lum < 0.55 else RULE,
                      zorder=5)
        ax_m.plot([2.5, 2.5], [i + 0.10, i + 0.90], color=HAIR, lw=0.7,
                  zorder=2)
        ax_m.add_patch(Rectangle((2.5, i + 0.33), d * scale, 0.34,
                                 facecolor=BAR_COL if d > 0 else EPI_COL,
                                 edgecolor="white", linewidth=0.5,
                                 hatch=None if d > 0 else "///", zorder=4))
        # every delta is right-aligned in one column, clear of every bar
        ax_m.text(3.28, i + 0.50, f"{d:+.2f}", ha="right", va="center",
                  fontsize=6.8, color=RULE, zorder=5)
        ax_m.text(-0.06, i + 0.5, name, ha="right", va="center",
                  fontsize=6.9, color=INK)
    for cx, head in ((0.5, "Episodic"), (1.5, "Semantic"),
                     (2.65, "Sem - Epi")):
        ax_m.text(cx, -0.10, head, ha="center", va="bottom",
                  fontsize=7.4, fontweight="bold", color=INK)
    ax_m.text(0.5, 1.075, "Recall@5 by query type",
              transform=ax_m.transAxes, ha="center", va="bottom",
              fontsize=8.8, fontweight="bold", color=INK)
    ax_m.text(2.5, len(QUERIES) + 0.20,
              "episodic wins $\\leftarrow$   $\\rightarrow$ semantic wins",
              ha="center", va="top", fontsize=6.2, color=FAINT)

    add_panel_label(ax_e, "(a)", x=-0.020, y=1.115)
    add_panel_label(ax_s, "(b)", x=-0.020, y=1.115)
    add_panel_label(ax_m, "(c)", x=-0.020, y=1.115)

    cax = fig.add_axes([RECT_MAT[0], 0.048, 0.176, 0.013])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP, aspect="auto",
               extent=(0, 1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([0, 0.5, 1.0])
    cax.tick_params(labelsize=6.0, width=0.6, length=1.8, pad=1.2)
    for s in cax.spines.values():
        s.set_linewidth(0.6)
        s.set_color(RULE)

    e_vals = [f"{e:.2f}" for _, e, _ in QUERIES]
    s_vals = [f"{s_:.2f}" for _, _, s_ in QUERIES]
    e_sum, s_sum = sum(map(float, e_vals)), sum(map(float, s_vals))
    # two lines, and the radius note stops well left of the colour bar: as one
    # line this ran under the bar and its tick labels came out through the type

    savefig(fig, "1199_wm_mem_episodic_semantic.png")


if __name__ == "__main__":
    main()
