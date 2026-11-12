"""Phylogeny with an aligned trait matrix.

Left: a rectangular phylogram drawn from line segments, with branch depths on
a substitution scale, bootstrap support at every internal node and dotted tip
leaders.  Right: the trait matrix, three z-scored continuous traits on a
shared diverging ramp plus two categorical traits, with binomials as
right-hand tick labels.  Tree and matrix are locked to one shared y-limit
constant so rows can never drift apart.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig


BRANCH = "#3f4a45"
INK = "#1a1a1a"
CAT_COLS = ["#2e7d5b", "#c08f3e", "#3d8b93"]
DIET_LEVELS = ["Insectivore", "Granivore", "Omnivore"]
NEST_LEVELS = ["Cavity", "Cup", "Ground"]

ECO_DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "eco_div", ["#8c5a2b", "#cfa877", "#f1eee5", "#8fc0a9", "#1b5e4b"])


def leaf(name):
    return ("leaf", name)


TREE = ("clade", 0.00, None, [
    ("clade", 0.30, 99, [
        ("clade", 0.62, 97, [leaf("Turdus merula"), leaf("Turdus philomelos")]),
        leaf("Erithacus rubecula")]),
    ("clade", 0.14, 92, [
        ("clade", 0.26, 88, [
            ("clade", 0.55, 79, [leaf("Sylvia atricapilla"),
                                 leaf("Phylloscopus collybita")]),
            ("clade", 0.58, 96, [leaf("Parus major"),
                                 leaf("Cyanistes caeruleus")])]),
        ("clade", 0.22, 84, [
            ("clade", 0.60, 95, [leaf("Fringilla coelebs"),
                                 leaf("Carduelis carduelis")]),
            ("clade", 0.34, 72, [
                ("clade", 0.52, 68, [leaf("Passer domesticus"),
                                     leaf("Sturnus vulgaris")]),
                ("clade", 0.44, 81, [
                    leaf("Troglodytes troglodytes"),
                    ("clade", 0.70, 89, [leaf("Certhia familiaris"),
                                         leaf("Sitta europaea")])])])])])])

#            species              mass(g) wing(mm) clutch diet nest
TRAITS = {
    "Turdus merula":         (100.0, 130.0, 4.5, 2, 1),
    "Turdus philomelos":     (70.0, 118.0, 4.6, 0, 1),
    "Erithacus rubecula":    (17.0, 72.0, 5.2, 0, 2),
    "Sylvia atricapilla":    (18.0, 74.0, 4.8, 2, 1),
    "Phylloscopus collybita": (8.0, 60.0, 5.6, 0, 2),
    "Parus major":           (18.0, 76.0, 8.5, 2, 0),
    "Cyanistes caeruleus":   (11.0, 64.0, 9.2, 0, 0),
    "Fringilla coelebs":     (22.0, 85.0, 4.7, 1, 1),
    "Carduelis carduelis":   (15.0, 76.0, 4.9, 1, 1),
    "Passer domesticus":     (30.0, 78.0, 4.4, 1, 0),
    "Sturnus vulgaris":      (80.0, 128.0, 5.0, 2, 0),
    "Troglodytes troglodytes": (9.0, 50.0, 6.4, 0, 2),
    "Certhia familiaris":    (9.0, 64.0, 5.8, 0, 0),
    "Sitta europaea":        (22.0, 86.0, 6.8, 2, 0),
}

CONT_LABELS = ["Body mass\n(log g)", "Wing length\n(mm)", "Clutch size\n(eggs)"]
DISC_LABELS = ["Diet guild", "Nest type"]
TIP_DEPTH = 1.0


def panel_tag(fig, ax, label, dx=-0.052, dy=0.018, size=10):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, label, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color=INK)


def collect_leaves(node, out):
    if node[0] == "leaf":
        out.append(node[1])
    else:
        for ch in node[3]:
            collect_leaves(ch, out)
    return out


def draw_tree(ax, node, leaf_y, segs, supports):
    """Recursively lay out a rectangular phylogram; returns the node's y."""
    if node[0] == "leaf":
        return leaf_y[node[1]], TIP_DEPTH
    _, depth, support, children = node
    ys = []
    for ch in children:
        y_ch, x_ch = draw_tree(ax, ch, leaf_y, segs, supports)
        segs.append(((depth, y_ch), (x_ch, y_ch)))
        ys.append(y_ch)
    segs.append(((depth, min(ys)), (depth, max(ys))))
    y_here = 0.5 * (min(ys) + max(ys))
    if support is not None:
        supports.append((depth, y_here, support))
    return y_here, depth


def main():
    set_pub_style(base_font=8)

    names = collect_leaves(TREE, [])
    n = len(names)
    leaf_y = {nm: (n - 1 - i) for i, nm in enumerate(names)}
    YLIM = (-1.9, n - 1 + 0.75)          # shared by tree and matrix

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.62],
                  height_ratios=[0.30, 1.0], hspace=0.10, wspace=0.035)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_tree = fig.add_subplot(gs[1, 0])
    ax_mat = fig.add_subplot(gs[1, 1])

    # ---- phylogram --------------------------------------------------------
    segs, supports = [], []
    draw_tree(ax_tree, TREE, leaf_y, segs, supports)
    root_y = 0.5 * (min(leaf_y.values()) + max(leaf_y.values()))
    segs.append(((-0.075, root_y), (0.0, root_y)))

    for (x0, y0), (x1, y1) in segs:
        ax_tree.plot([x0, x1], [y0, y1], color=BRANCH, lw=1.0,
                     solid_capstyle="round", zorder=3)

    for x, y, sup in supports:
        col = BRANCH if sup >= 90 else "#8d8d84"
        ax_tree.text(x + 0.014, y + 0.30, f"{sup}", fontsize=6.0, color=col,
                     ha="left", va="bottom", zorder=4)

    for nm in names:
        y = leaf_y[nm]
        ax_tree.plot([TIP_DEPTH, 1.135], [y, y], color="#c9c9c1", lw=0.55,
                     ls=(0, (1.2, 1.6)), zorder=2)
        ax_tree.scatter([TIP_DEPTH], [y], s=7, color=BRANCH, zorder=4)

    # substitution scale bar
    ax_tree.plot([0.72, 0.92], [-1.15, -1.15], color="#6f6f66", lw=1.0,
                 solid_capstyle="butt")
    for xt in (0.72, 0.92):
        ax_tree.plot([xt, xt], [-1.28, -1.02], color="#6f6f66", lw=1.0)
    ax_tree.text(0.82, -1.42, "0.2 substitutions/site", fontsize=6.2,
                 color="#6f6f66", ha="center", va="top")

    ax_tree.set_xlim(-0.10, 1.14)
    ax_tree.set_ylim(*YLIM)
    ax_tree.set_xticks([])
    ax_tree.set_yticks([])
    for s in ax_tree.spines.values():
        s.set_visible(False)
    ax_tree.set_title("Maximum-likelihood phylogeny (cytb + ND2, 14 taxa)",
                      fontsize=8.0, loc="left", pad=5)

    # ---- trait matrix -----------------------------------------------------
    raw = np.array([TRAITS[nm][:3] for nm in names], dtype=float)
    raw[:, 0] = np.log10(raw[:, 0])
    z = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    zlim = 2.1
    norm = mpl.colors.Normalize(-zlim, zlim)

    cont_x = [0.0, 1.0, 2.0]
    disc_x = [3.35, 4.35]
    cw, ch = 0.88, 0.80

    for i, nm in enumerate(names):
        y = leaf_y[nm]
        for j, xc in enumerate(cont_x):
            ax_mat.add_patch(Rectangle((xc - cw / 2, y - ch / 2), cw, ch,
                                       facecolor=ECO_DIV(norm(z[i, j])),
                                       edgecolor="white", linewidth=0.7,
                                       zorder=3))
        for j, xc in enumerate(disc_x):
            lvl = TRAITS[nm][3 + j]
            ax_mat.add_patch(Rectangle((xc - cw / 2, y - ch / 2), cw, ch,
                                       facecolor=CAT_COLS[lvl],
                                       edgecolor="white", linewidth=0.7,
                                       alpha=0.88, zorder=3))

    ax_mat.set_xlim(-0.60, disc_x[-1] + 0.60)
    ax_mat.set_ylim(*YLIM)
    ax_mat.set_yticks([leaf_y[nm] for nm in names])
    ax_mat.set_yticklabels(names, fontsize=6.6, fontstyle="italic")
    ax_mat.yaxis.set_ticks_position("right")
    ax_mat.yaxis.set_label_position("right")
    ax_mat.set_xticks(cont_x + disc_x)
    ax_mat.set_xticklabels(CONT_LABELS + DISC_LABELS, fontsize=6.4,
                           rotation=42, ha="left", linespacing=1.2)
    ax_mat.xaxis.set_ticks_position("top")
    ax_mat.xaxis.set_label_position("top")
    ax_mat.tick_params(length=0, pad=2.5)
    for s in ax_mat.spines.values():
        s.set_visible(False)
    ax_mat.axvline(2.68, color="#dcdcd4", lw=0.8, ymin=0.06, ymax=0.98)

    # ---- legend strip -----------------------------------------------------
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")

    for row, (title, levels) in enumerate([("Diet guild", DIET_LEVELS),
                                           ("Nest type", NEST_LEVELS)]):
        y = 0.93 - 0.245 * row
        ax_leg.text(0.0, y, title, fontsize=6.8, fontweight="bold",
                    va="center", color=INK)
        for k, lab in enumerate(levels):
            x = 0.235 + 0.255 * k
            ax_leg.add_patch(Rectangle((x, y - 0.058), 0.036, 0.116,
                                       facecolor=CAT_COLS[k], alpha=0.88,
                                       edgecolor="white", linewidth=0.6,
                                       clip_on=False))
            ax_leg.text(x + 0.052, y, lab, fontsize=6.5, va="center",
                        color="#33332f")

    ax_leg.text(0.0, 0.385, "Trait z-score", fontsize=6.8, fontweight="bold",
                va="center", color=INK)
    bx0, bw, by, bh = 0.235, 0.40, 0.327, 0.115
    nseg = 48
    for k in range(nseg):
        ax_leg.add_patch(Rectangle((bx0 + bw * k / nseg, by), bw / nseg + 0.002,
                                   bh, facecolor=ECO_DIV(k / (nseg - 1.0)),
                                   edgecolor="none", clip_on=False))
    ax_leg.add_patch(Rectangle((bx0, by), bw, bh, facecolor="none",
                               edgecolor="#9a9a92", linewidth=0.6,
                               clip_on=False))
    for frac, lab in ((0.0, "$-$2"), (0.5, "0"), (1.0, "+2")):
        ax_leg.text(bx0 + bw * frac, by - 0.055, lab, fontsize=6.2,
                    ha="center", va="top", color="#55554f")
    ax_leg.text(0.0, 0.07, "node labels = bootstrap support (%)",
                fontsize=6.2, va="center", color="#8d8d84", style="italic")

    fig.subplots_adjust(left=0.072, right=0.845, top=0.905, bottom=0.035)

    panel_tag(fig, ax_tree, "(a)", dx=-0.060, dy=0.012)
    panel_tag(fig, ax_mat, "(b)", dx=-0.032, dy=0.100)

    savefig(fig, "227_phylo_traits.png")


if __name__ == "__main__":
    main()
