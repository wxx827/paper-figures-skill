"""Language phylogeny with its lexical similarity matrix and node-age density.

The tree is drawn on a real time axis rather than as a schematic cladogram:
horizontal position is the divergence date, so a shallow split looks shallow.
Time runs backwards from the present at the right, which is the reading
direction of a family tree and lets every leaf label sit in the same column.
Shared innovations are written on the branch they define, because a
subgrouping without the evidence that motivates it is only a drawing.

The two supporting panels are computed from the same tree object as the
drawing: the similarity matrix decays with time since the most recent common
ancestor and is ordered by the leaf order, which is what makes the four
subfamily blocks appear, and the density panel shows when the splits happen.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import (axes_inches, panel_labels_inches, safe_colorbar,
                    shared_norm)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#cccccc"
STEM = "#555555"

GROUP_COLOR = {
    "Germanic": "#3b5b92",
    "Italic": "#b23a48",
    "Balto-Slavic": "#2e7d5b",
    "Indo-Iranian": "#c9a227",
}

# (name, age in years BP, children, group started here or None)
TREE = ("Proto-language", 6000, [
    ("Western", 4700, [
        ("Germanic", 2500, [
            ("West Germanic", 1450, [
                ("English", 0, [], None),
                ("Continental WG", 950, [
                    ("German", 0, [], None),
                    ("Dutch", 0, [], None),
                ], None),
            ], None),
            ("Swedish", 0, [], None),
        ], "Germanic"),
        ("Italic", 2750, [
            ("Romance", 1550, [
                ("Spanish", 0, [], None),
                ("Gallo-Italic", 1150, [
                    ("French", 0, [], None),
                    ("Italian", 0, [], None),
                ], None),
            ], None),
            ("Latin (attested)", 1900, [], None),
        ], "Italic"),
    ], None),
    ("Eastern", 4200, [
        ("Balto-Slavic", 3300, [
            ("Slavic", 1350, [
                ("Russian", 0, [], None),
                ("West Slavic", 900, [
                    ("Polish", 0, [], None),
                    ("Czech", 0, [], None),
                ], None),
            ], None),
            ("Lithuanian", 0, [], None),
        ], "Balto-Slavic"),
        ("Indo-Iranian", 3050, [
            ("Indic", 1700, [
                ("Hindi", 0, [], None),
                ("Bengali", 0, [], None),
            ], None),
            ("Persian", 0, [], None),
        ], "Indo-Iranian"),
    ], None),
], None)

# shared innovations, keyed by the node the branch leads into
INNOVATION = {
    "Germanic": "*p t k > f th h",
    "West Germanic": "consonant gemination",
    "Italic": "*bh dh gh > f",
    "Romance": "nominal case system lost",
    "Balto-Slavic": "*s > x after i u r k",
    "Slavic": "open-syllable restructuring",
    "Indo-Iranian": "*e *o *a merge to a",
    "Indic": "retroflex series",
}
CODE = {"English": "Eng", "German": "Ger", "Dutch": "Dut", "Swedish": "Swe",
        "Spanish": "Spa", "French": "Fra", "Italian": "Ita",
        "Latin (attested)": "Lat", "Russian": "Rus", "Polish": "Pol",
        "Czech": "Cze", "Lithuanian": "Lit", "Hindi": "Hin",
        "Bengali": "Ben", "Persian": "Per"}
DECAY = 4200.0                   # e-folding time of lexical retention, yr

FW, FH = 10.4, 6.4


def flatten(node, group=None, parent=None, out=None, order=None):
    name, age, kids, grp = node
    group = grp or group
    out = {} if out is None else out
    order = [] if order is None else order
    out[name] = dict(age=float(age), group=group, parent=parent,
                     kids=[k[0] for k in kids])
    if not kids:
        order.append(name)
    for k in kids:
        flatten(k, group, name, out, order)
    return out, order


def layout(nodes, leaves):
    y = {n: float(i) for i, n in enumerate(leaves)}
    for name in sorted(nodes, key=lambda n: nodes[n]["age"]):
        kids = nodes[name]["kids"]
        if kids:
            y[name] = float(np.mean([y[k] for k in kids]))
    return y


def mrca_age(nodes, a, b):
    def chain(n):
        c = []
        while n is not None:
            c.append(n)
            n = nodes[n]["parent"]
        return c
    ca, cb = chain(a), set(chain(b))
    for n in ca:
        if n in cb:
            return nodes[n]["age"]
    return max(nodes[n]["age"] for n in nodes)


def draw_tree(ax, nodes, leaves, y):
    for name, d in nodes.items():
        col = GROUP_COLOR.get(d["group"], STEM)
        lw = 1.9 if d["group"] else 2.3
        p = d["parent"]
        if p is not None:
            ax.plot([nodes[p]["age"], d["age"]], [y[name]] * 2, color=col,
                    lw=lw, solid_capstyle="round", zorder=4)
        if d["kids"]:
            ys = [y[k] for k in d["kids"]]
            ax.plot([d["age"]] * 2, [min(ys), max(ys)], color=col, lw=lw,
                    solid_capstyle="round", zorder=4)
            ax.plot([d["age"]], [y[name]], marker="o", ms=3.2, mfc="white",
                    mec=col, mew=1.0, zorder=6)

    for name, d in nodes.items():
        col = GROUP_COLOR.get(d["group"], STEM)
        if not d["kids"]:
            if d["age"] > 0.0:            # attested but with no living issue
                ax.plot([d["age"], 0.0], [y[name]] * 2, color=col, lw=0.8,
                        ls=(0, (1.6, 1.8)), zorder=3)
            ax.plot([d["age"]], [y[name]], marker="o", ms=4.0, mfc=col,
                    mec="white", mew=0.6, zorder=7)
            ax.text(-150.0, y[name], name, ha="left", va="center",
                    fontsize=7.2, color=INK)
        elif name in GROUP_COLOR:
            # below the node on screen: the branch above it carries the
            # innovation caption
            ax.text(d["age"], y[name] + 0.30, name, ha="center", va="top",
                    fontsize=7.4, fontweight="bold", color=GROUP_COLOR[name],
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                              edgecolor="none", alpha=0.88))

    for name, text in INNOVATION.items():
        d = nodes[name]
        col = GROUP_COLOR.get(d["group"], STEM)
        xm = 0.5 * (nodes[d["parent"]]["age"] + d["age"])
        ax.plot([xm], [y[name]], marker="|", ms=6.5, mew=1.1, color=col,
                zorder=6)
        ax.text(xm, y[name] - 0.10, text, ha="center", va="bottom",
                fontsize=6.0, color=SUB, zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.88))

    ax.axvline(0.0, color=HAIR, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(6350.0, -1450.0)
    ax.set_ylim(len(leaves) - 0.4, -0.9)
    ax.set_xticks([6000, 5000, 4000, 3000, 2000, 1000, 0])
    ax.set_yticks([])
    ax.set_xlabel("Divergence time (years BP, present at right)",
                  fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Subgrouping and the innovations that define it",
                 fontsize=8.5, fontweight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def similarity(nodes, leaves, rng):
    n = len(leaves)
    s = np.zeros((n, n))
    for i, a in enumerate(leaves):
        for j, b in enumerate(leaves):
            if i == j:
                s[i, j] = 100.0
            elif j > i:
                t = mrca_age(nodes, a, b)
                v = 100.0 * np.exp(-t / DECAY) + rng.normal(0.0, 2.2)
                s[i, j] = s[j, i] = float(np.clip(v, 2.0, 99.0))
    return s


def draw_matrix(fig, ax, s, leaves):
    norm = shared_norm([s])
    im = ax.imshow(s, cmap=get_cmap("cb_sequential_blue"), norm=norm,
                   aspect="equal", zorder=2)
    ax.set_xticks(np.arange(len(leaves)))
    ax.set_xticklabels([CODE[n] for n in leaves], fontsize=6.0, rotation=90)
    ax.set_yticks(np.arange(len(leaves)))
    ax.set_yticklabels([CODE[n] for n in leaves], fontsize=6.0)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("Lexical similarity, ordered by the tree", fontsize=8.0,
                 fontweight="bold")
    safe_colorbar(fig, im, ax, label="shared basic vocabulary (%)",
                  width=0.10, pad=0.06, labelsize=6.4)


def draw_ages(ax, ages):
    grid = np.linspace(0.0, 6400.0, 400)
    kde = gaussian_kde(ages, bw_method=0.34)
    d = kde(grid)
    ax.fill_between(grid, 0.0, d, color="#3b5b92", alpha=0.22, lw=0.0,
                    zorder=2)
    ax.plot(grid, d, color="#3b5b92", lw=1.3, zorder=4)
    for a in ages:
        ax.plot([a, a], [0.0, 0.045 * d.max()], color=STEM, lw=0.7, zorder=5)
    med = float(np.median(ages))
    ax.axvline(med, color="#b23a48", lw=1.0, ls=(0, (3.5, 2.0)), zorder=6)
    ax.text(med - 130.0, d.max() * 0.92, f"median {med:.0f} yr BP",
            ha="right", va="top", fontsize=6.2, color="#b23a48")
    ax.set_xlim(6400.0, 0.0)
    ax.set_ylim(0.0, d.max() * 1.22)
    ax.set_yticks([])
    ax.set_xlabel("node age (years BP)", fontsize=7.6)
    ax.set_ylabel("density (-)", fontsize=7.6)
    ax.tick_params(labelsize=6.6)
    ax.set_title(f"Ages of the {len(ages)} splits", fontsize=8.0,
                 fontweight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main():
    set_pub_style()
    rng = np.random.default_rng(634)
    nodes, leaves = flatten(TREE)
    y = layout(nodes, leaves)
    s = similarity(nodes, leaves, rng)
    ages = np.array([d["age"] for d in nodes.values() if d["kids"]])

    fig = plt.figure(figsize=(FW, FH))
    ax_t = axes_inches(fig, (0.42, 0.98, 5.45, 4.20))
    ax_m = axes_inches(fig, (6.90, 2.78, 2.40, 2.40))
    ax_d = axes_inches(fig, (6.90, 0.98, 2.40, 1.22))

    draw_tree(ax_t, nodes, leaves, y)
    draw_matrix(fig, ax_m, s, leaves)
    draw_ages(ax_d, ages)

    panel_labels_inches(fig, [ax_t, ax_m, ax_d], dx=-0.30, dy=0.20)

    # within-group versus between-group similarity, straight off the matrix
    grp = np.array([nodes[n]["group"] for n in leaves])
    same = np.array([[grp[i] == grp[j] for j in range(len(leaves))]
                     for i in range(len(leaves))])
    off = ~np.eye(len(leaves), dtype=bool)
    within = float(s[same & off].mean())
    between = float(s[~same & off].mean())


    print(f"leaves={len(leaves)} nodes={len(ages)} "
          f"within={within:.1f}% between={between:.1f}%")
    savefig(fig, "634_lin_language_tree.png")


if __name__ == "__main__":
    main()
