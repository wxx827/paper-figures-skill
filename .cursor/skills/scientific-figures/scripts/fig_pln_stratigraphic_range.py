"""Stratigraphic range chart of eighteen synthetic genera through the Phanerozoic.

Ranges are drawn as horizontal bars against a single geological time axis that
runs from old on the left to young on the right; the axis is inverted once, at
the top of the drawing code, and the period band, the event lines and the main
panel all inherit it, so the three can never disagree about which way time
flows.  Bar fill encodes relative abundance on one shared sequential ramp with
one colour bar - per-bar autoscaling would make a rare genus look as common as
an abundant one.

The two subordinate panels count what the main panel shows: first and last
appearances are binned per period straight out of the same range table, and
the duration histogram is the difference of the same two columns.  The period
band deliberately avoids the International Chronostratigraphic Chart colours,
which are far too saturated to sit under this palette.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
GREY = "#808080"
RED = "#b23a48"
BLUE = "#3b5b92"

FIG_W, FIG_H = 10.4, 6.4
SEED = 33
T_OLD, T_YOUNG = 541.0, 0.0

# Period name, abbreviation, base (Ma), top (Ma).  Quaternary is folded into
# the Neogene: 2.6 Myr is 0.02 in wide here and could not carry a label.
PERIODS = [
    ("Cambrian", "Cm", 541.0, 486.9),
    ("Ordovician", "O", 486.9, 443.1),
    ("Silurian", "S", 443.1, 419.2),
    ("Devonian", "D", 419.2, 358.9),
    ("Carboniferous", "C", 358.9, 298.9),
    ("Permian", "P", 298.9, 251.9),
    ("Triassic", "Tr", 251.9, 201.4),
    ("Jurassic", "J", 201.4, 145.0),
    ("Cretaceous", "K", 145.0, 66.0),
    ("Paleogene", "Pg", 66.0, 23.0),
    ("Neogene", "Ng", 23.0, 0.0),
]
BAND = ("#f0f0f0", "#e0e0e0")

# Mass extinctions: age (Ma) and the boundary shorthand used on the chart.
EVENTS = [(443.8, "O-S"), (372.0, "F-F"), (251.9, "P-Tr"),
          (201.4, "Tr-J"), (66.0, "K-Pg")]

# name, first appearance (Ma), last appearance (Ma), abundance
TAXA = [
    ("Genus A", 528.0, 471.0, 3.1),
    ("Genus B", 516.0, 443.8, 7.4),
    ("Genus C", 505.0, 372.0, 4.6),
    ("Genus D", 488.0, 443.8, 9.2),
    ("Genus E", 470.0, 419.0, 2.4),
    ("Genus F", 452.0, 251.9, 6.1),
    ("Genus G", 428.0, 372.0, 8.7),
    ("Genus H", 411.0, 298.0, 5.3),
    ("Genus I", 383.0, 251.9, 3.8),
    ("Genus J", 349.0, 251.9, 10.4),
    ("Genus K", 302.0, 214.0, 4.1),
    ("Genus L", 266.0, 201.4, 6.8),
    ("Genus M", 237.0, 201.4, 2.9),
    ("Genus N", 208.0, 96.0, 7.9),
    ("Genus O", 186.0, 66.0, 5.6),
    ("Genus P", 152.0, 66.0, 9.8),
    ("Genus Q", 118.0, 34.0, 3.4),
    ("Genus R", 61.0, 4.0, 6.3),
]


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)

    order = np.argsort([-t[1] for t in TAXA])
    names = [TAXA[i][0] for i in order]
    fad = np.array([TAXA[i][1] for i in order])
    lad = np.array([TAXA[i][2] for i in order])
    abund = np.array([TAXA[i][3] for i in order])
    n = len(names)
    err_f = rng.uniform(2.5, 9.0, n)
    err_l = rng.uniform(2.5, 9.0, n)
    dur = fad - lad

    cmap = get_cmap("cb_sequential_blue")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=abund.max() * 1.05)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, (1.05, 1.44, 4.85, 4.03))
    ax_b = axes_inches(fig, (1.05, 1.12, 4.85, 0.28))
    cax = axes_inches(fig, (6.08, 2.42, 0.105, 1.72))
    ax_c = axes_inches(fig, (7.20, 3.62, 2.90, 1.85))
    ax_d = axes_inches(fig, (7.20, 0.92, 2.90, 1.85))

    # ---- (a) range bars ---------------------------------------------------
    ypos = np.arange(n)
    head = 2.6                      # empty rows kept for the event labels
    ax.set_xlim(T_OLD, T_YOUNG)     # old on the left, young on the right
    ax.set_ylim(n - 0.4, -head)
    for i in range(0, n, 2):
        ax.axhspan(i - 0.5, i + 0.5, color="#fafafa", zorder=0)

    for age, lab in EVENTS:
        ax.axvline(age, color=RED, lw=0.9, alpha=0.80, zorder=2)
        ax_b.axvline(age, color=RED, lw=0.9, alpha=0.80, zorder=5)
        ax.text(age, -0.65, lab, fontsize=6.4, color=RED, ha="center",
                va="bottom", rotation=90.0, zorder=6)

    for i, (f, l, a) in enumerate(zip(fad, lad, abund)):
        ax.barh(i, f - l, left=l, height=0.52, color=cmap(norm(a)),
                edgecolor="white", linewidth=0.4, zorder=4)
        for x, e in ((f, err_f[i]), (l, err_l[i])):
            ax.plot([x - e, x + e], [i, i], color=GREY, lw=0.7, zorder=5)
            for c in (x - e, x + e):
                ax.plot([c, c], [i - 0.13, i + 0.13], color=GREY, lw=0.7,
                        zorder=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=6.6, style="italic")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Stratigraphic ranges of 18 genera, ordered by first "
                 "appearance", fontsize=9.0, fontweight="bold", color=INK)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="vertical")
    cb.set_label("relative abundance (specimens / sample)", fontsize=6.4,
                 color=SUB, labelpad=3.0)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(GREY)
    cb.ax.tick_params(labelsize=6.2, width=0.6, length=2.0, color=GREY)

    # ---- period band ------------------------------------------------------
    ax_b.set_xlim(T_OLD, T_YOUNG)
    ax_b.set_ylim(0.0, 1.0)
    for k, (full, ab, base, top) in enumerate(PERIODS):
        ax_b.axvspan(base, top, facecolor=BAND[k % 2], edgecolor="white",
                     linewidth=0.6, zorder=1)
        ax_b.text(0.5 * (base + top), 0.42, ab, fontsize=6.6, color=INK,
                  ha="center", va="center", zorder=6)
    ax_b.set_yticks([])
    ax_b.set_xlabel("geological time (Ma before present) -- older to the left",
                    fontsize=8.0)
    ax_b.tick_params(axis="x", labelsize=6.8)
    for s in ax_b.spines.values():
        s.set_visible(False)

    # ---- (b) originations and extinctions per period ---------------------
    edges = np.array([T_OLD] + [p[3] for p in PERIODS])
    orig, _ = np.histogram(fad, bins=edges[::-1])
    ext, _ = np.histogram(lad, bins=edges[::-1])
    orig, ext = orig[::-1], ext[::-1]
    yp = np.arange(len(PERIODS))
    ax_c.barh(yp, -orig, height=0.66, color=BLUE, edgecolor="white",
              linewidth=0.5)
    ax_c.barh(yp, ext, height=0.66, color=RED, edgecolor="white",
              linewidth=0.5)
    ax_c.axvline(0.0, color=INK, lw=0.7, zorder=4)
    m = max(orig.max(), ext.max()) + 1
    ax_c.set_xlim(-m, m)
    ax_c.set_ylim(len(PERIODS) - 0.4, -1.15)
    ax_c.text(-m * 0.52, -1.05, "first appearances", fontsize=6.4,
              color=BLUE, ha="center", va="top")
    ax_c.text(m * 0.52, -1.05, "last appearances", fontsize=6.4, color=RED,
              ha="center", va="top")
    ax_c.set_yticks(yp)
    ax_c.set_yticklabels([p[1] for p in PERIODS], fontsize=6.4)
    ax_c.set_xticks(np.arange(-m + 1, m, 1))
    ax_c.set_xticklabels([str(abs(v)) for v in np.arange(-m + 1, m, 1)])
    ax_c.tick_params(labelsize=6.4, length=2.0)
    ax_c.set_xlabel("taxa per period (n)", fontsize=7.4)
    ax_c.spines["left"].set_visible(False)
    ax_c.set_title("Turnover clusters at the event horizons", fontsize=8.5,
                   fontweight="bold", color=INK)

    # ---- (c) range duration ----------------------------------------------
    ax_d.hist(dur, bins=np.linspace(0.0, dur.max() * 1.08, 11), color=BLUE,
              alpha=0.80, edgecolor="white", linewidth=0.6)
    ax_d.axvline(np.median(dur), color=RED, lw=1.1, ls=(0, (2.8, 1.6)),
                 zorder=4)
    ax_d.text(np.median(dur) + 4.0, ax_d.get_ylim()[1] * 0.70,
              f"median {np.median(dur):.0f} Myr", fontsize=6.6, color=RED,
              ha="left", va="top")
    ax_d.set_xlabel("range duration (Myr)", fontsize=7.4)
    ax_d.set_ylabel("genera (n)", fontsize=7.4)
    ax_d.tick_params(labelsize=6.6)
    ax_d.text(0.97, 0.94,
              f"n = {n} genera   mean {dur.mean():.0f} Myr\n"
              f"longest {dur.max():.0f} Myr   shortest {dur.min():.0f} Myr",
              transform=ax_d.transAxes, ha="right", va="top", fontsize=6.4,
              color=SUB)
    ax_d.set_title("Most ranges are shorter than 100 Myr", fontsize=8.5,
                   fontweight="bold", color=INK)


    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.62, dy=0.10)
    panel_labels_inches(fig, [ax_c], labels=["(b)"], dx=-0.52, dy=0.10)
    panel_labels_inches(fig, [ax_d], labels=["(c)"], dx=-0.52, dy=0.10)
    savefig(fig, "825_pln_stratigraphic_range.png")


if __name__ == "__main__":
    main()
