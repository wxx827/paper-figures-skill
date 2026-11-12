"""Method family tree: three world-model lineages laid out along a year axis.

Each node is a representative method placed at its publication year, joined to
its parent by an S-curve in the family's own rung of the blue ladder; marker
area is proportional to the citation count, and the size key quotes three
counts that are actually present in the data.  One dashed grey link marks the
single cross-family borrowing, and the warm ring picks out the most-cited node
by ``argmax`` rather than by hand.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from pubstyle import COLORS, darken, savefig, set_pub_style

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"

FAMILIES = [("Latent dynamics models", BLUE[0]),
            ("Video generative world models", BLUE[2]),
            ("Embodied control world models", BLUE[4])]

# key, label, year, family index, y row, parent key, citations
NODES = [
    ("wm18", "World Models", 2018, 0, 9.65, None, 3120),
    ("planet", "PlaNet", 2019, 0, 9.65, "wm18", 2180),
    ("dreamer", "Dreamer", 2020, 0, 9.65, "planet", 2940),
    ("drmv2", "DreamerV2", 2021, 0, 9.65, "dreamer", 1760),
    ("drmv3", "DreamerV3", 2023, 0, 10.75, "drmv2", 1490),
    ("tdmpc2", "TD-MPC2", 2024, 0, 8.55, "drmv2", 640),
    ("vgpt", "VideoGPT", 2021, 1, 6.20, None, 1180),
    ("gaia", "GAIA-1", 2023, 1, 7.30, "vgpt", 810),
    ("genie", "Genie", 2024, 1, 5.10, "vgpt", 1040),
    ("soraw", "Sora-WM", 2025, 1, 7.30, "gaia", 430),
    ("genie2", "Genie-2", 2025, 1, 5.10, "genie", 360),
    ("navgen", "NavGen", 2026, 1, 6.20, "genie2", 120),
    ("muzero", "MuZero", 2020, 2, 2.05, None, 2460),
    ("iris", "IRIS", 2022, 2, 3.10, "muzero", 560),
    ("daydr", "DayDreamer", 2022, 2, 1.00, "muzero", 720),
    ("robodr", "RoboDreamer", 2024, 2, 1.00, "daydr", 380),
    ("embwm", "EmbodiedWM", 2025, 2, 2.05, "robodr", 210),
    ("wmagent", "WM-Agent", 2026, 2, 3.10, "embwm", 140),
]
# The one link that crosses families, drawn dashed and grey.
CROSS = ("vgpt", "iris")
# Row positions above are hand-set (see the module note); the bands are not --
# they are the span of whatever rows each family ended up on, so moving a node
# cannot leave it sitting outside its own band.
BAND_PAD = 0.55
BANDS = [(min(n[4] for n in NODES if n[3] == k) - BAND_PAD,
          max(n[4] for n in NODES if n[3] == k) + BAND_PAD)
         for k in range(len(FAMILIES))]
YLO, YHI = min(lo for lo, _ in BANDS), max(hi for _, hi in BANDS)


def gen_data(seed=1555):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_survey_family_tree.csv")
        return list(df.itertuples(index=False, name=None))
    """
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.16, 0.16, len(NODES))
    return [(k, lab, yr + float(dx), fam, y, par, cit)
            for (k, lab, yr, fam, y, par, cit), dx in zip(NODES, jitter)]


def scurve(ax, p0, p1, colour, lw=1.15, ls="-", zorder=2, alpha=0.9):
    """Horizontal-tangent cubic between two nodes."""
    xm = 0.5 * (p0[0] + p1[0])
    path = Path([p0, (xm, p0[1]), (xm, p1[1]), p1],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=colour, lw=lw,
                           linestyle=ls, zorder=zorder, alpha=alpha,
                           capstyle="round"))


def main():
    set_pub_style(base_font=8)
    nodes = gen_data()
    pos = {k: (x, y) for k, _, x, _, y, _, _ in nodes}
    cites = np.array([c for *_, c in nodes], dtype=float)
    top_i = int(np.argmax(cites))

    def area(c):
        return 34.0 + 340.0 * (c / cites.max())

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = fig.add_axes([0.052, 0.115, 0.700, 0.720])

    # ---- family bands -------------------------------------------------------
    for (lo, hi), (name, colour) in zip(BANDS, FAMILIES):
        ax.add_patch(Rectangle((2017.15, lo), 9.85, hi - lo,
                               facecolor="#f4f7fa", edgecolor="none",
                               zorder=0))
        ax.text(2017.32, hi - 0.22, name, ha="left", va="top", fontsize=7.2,
                color=darken(colour, 0.46), fontweight="bold", zorder=1)

    years = np.arange(2018, 2027)
    for yr in years:
        ax.plot([yr, yr], [YLO, YHI], color="#e2e7ec", lw=0.6, zorder=0)

    # ---- edges then nodes ---------------------------------------------------
    for key, _, x, fam, y, parent, _ in nodes:
        if parent is not None:
            scurve(ax, pos[parent], (x, y), FAMILIES[fam][1])
    scurve(ax, pos[CROSS[0]], pos[CROSS[1]], "#9aa4ad", lw=1.0, ls=(0, (3, 2)),
           zorder=2, alpha=0.95)

    for i, (key, label, x, fam, y, _, cit) in enumerate(nodes):
        colour = FAMILIES[fam][1]
        ax.scatter([x], [y], s=area(cit), color=colour, edgecolors="white",
                   linewidths=0.9, zorder=4)
        if i == top_i:
            ax.scatter([x], [y], s=area(cit) * 2.5, facecolors="none",
                       edgecolors=ACCENT, linewidths=1.3, zorder=5)
        r = np.sqrt(area(cit) / np.pi) / 72.0          # marker radius, inches
        dy = 0.30 + r * (YHI - YLO) / (0.720 * 6.4)
        ax.text(x, y + dy, label, ha="center", va="bottom", fontsize=7.2,
                color=INK, zorder=6)

    ax.annotate(f"most cited: {nodes[top_i][1]} ({int(cites[top_i]):d})",
                xy=pos[nodes[top_i][0]], xytext=(2021.6, 11.05),
                ha="left", va="center", fontsize=7.0, color=ACCENT, zorder=7,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8,
                                shrinkA=2.0, shrinkB=9.0))

    ax.set_xlim(2017.15, 2027.0)
    ax.set_ylim(YLO - 0.15, YHI + 0.25)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=7.4, color=INK)
    ax.set_yticks([])
    ax.set_xlabel("Year of first publication", fontsize=8.0, color=INK)
    ax.tick_params(axis="x", length=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # ---- keys ---------------------------------------------------------------
    fam_keys = [Line2D([], [], marker="o", linestyle="none", markersize=7.0,
                       markerfacecolor=c, markeredgecolor="white",
                       markeredgewidth=0.7,
                       label=f"{name} "
                             f"({sum(1 for n in nodes if n[3] == k):d} methods)")
                for k, (name, c) in enumerate(FAMILIES)]
    fam_keys.append(Line2D([], [], color="#9aa4ad", lw=1.0,
                           linestyle=(0, (3, 2)), label="cross-family reuse"))
    leg = fig.legend(handles=fam_keys, loc="upper left",
                     bbox_to_anchor=(0.772, 0.830), fontsize=7.4,
                     frameon=False, labelspacing=0.85, handletextpad=0.6)
    for txt in leg.get_texts():
        txt.set_color(INK)

    ticks = [int(cites.min()), int(np.median(cites)), int(cites.max())]
    size_keys = [Line2D([], [], marker="o", linestyle="none",
                        markersize=2.0 * np.sqrt(area(t) / np.pi),
                        markerfacecolor="#9dbad6", markeredgecolor="white",
                        markeredgewidth=0.7, label=f"{t:d} citations")
                 for t in ticks]
    leg2 = fig.legend(handles=size_keys, loc="upper left",
                      bbox_to_anchor=(0.772, 0.545), fontsize=7.4,
                      frameon=False, labelspacing=1.55, handletextpad=0.9)
    for txt in leg2.get_texts():
        txt.set_color(INK)
    fig.text(0.772, 0.575, "Marker area = citations", ha="left", va="bottom",
             fontsize=7.4, color=COLORS["sub"])

    # Lineage totals: the one thing the tree itself cannot show, since the
    # marker areas are per method.  Summed from the same citation column.
    fam_total = [sum(n[6] for n in nodes if n[3] == k)
                 for k in range(len(FAMILIES))]
    fig.text(0.772, 0.330, "Citations by lineage", ha="left", va="baseline",
             fontsize=7.4, fontweight="bold", color=INK)
    for row, (k, (name, colour)) in enumerate(zip(range(len(FAMILIES)),
                                                  FAMILIES)):
        yr_ = 0.292 - 0.032 * row
        fig.text(0.772, yr_, " ".join(name.split()[:2]), ha="left",
                 va="baseline", fontsize=7.2, color=darken(colour, 0.46))
        fig.text(0.972, yr_, f"{fam_total[k]:,d}", ha="right", va="baseline",
                 fontsize=7.2, color=INK)
    fig.add_artist(plt.Line2D([0.772, 0.972], [0.208, 0.208],
                              color="#c3ccd4", lw=0.7))
    fig.text(0.772, 0.186, f"All {len(nodes):d} methods", ha="left",
             va="baseline", fontsize=7.2, fontweight="bold", color=INK)
    fig.text(0.972, 0.186, f"{sum(fam_total):,d}", ha="right", va="baseline",
             fontsize=7.2, fontweight="bold", color=INK)

    n_roots = sum(1 for n in nodes if n[5] is None)

    savefig(fig, "1150_wm_survey_family_tree.png")


if __name__ == "__main__":
    main()
