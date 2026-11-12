"""Measured stratigraphic column of a synthetic Jurassic-Triassic succession.

The lithology column carries the message: every unit is filled with the
conventional ornament for its rock type (stipple for sandstone, ruling for
mudstone, brick for limestone, ellipses for conglomerate, solid for coal), so
the section can be read at a glance without consulting the colour key.  Depth
increases downward.  A chronostratigraphic ribbon, leader-lined facies
descriptions and a grain-size / cycle curve sit to the right; the fining-upward
cycles marked on that curve are detected from the grain-size index itself.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, lighten, darken

# ---- lithology ornaments (domain convention, DESIGN_SYSTEM section 2.2) ----
# face colours are derived from the house palette so the column sits in the
# same tonal family as the rest of the library; the hatch does the real work.
LITH = {
    "conglomerate": dict(fc=lighten("#c0603f", 0.52), hatch="oo",
                         grain=1.00, name="Conglomerate"),
    "sandstone":    dict(fc=lighten("#c9a227", 0.46), hatch="....",
                         grain=0.74, name="Sandstone"),
    "siltstone":    dict(fc=lighten("#8e6bbf", 0.72), hatch="--..",
                         grain=0.46, name="Siltstone"),
    "mudstone":     dict(fc=lighten("#2e7d5b", 0.70), hatch="----",
                         grain=0.20, name="Mudstone / shale"),
    "limestone":    dict(fc=lighten("#3b5b92", 0.66), hatch=None,
                         grain=0.56, name="Limestone (brick)"),
    "coal":         dict(fc="#3a3a3a", hatch=None,
                         grain=0.10, name="Coal"),
}

# top -> bottom: (lithology, thickness in m, facies description)
UNITS = [
    ("mudstone", 18, "Grey mudstone, rooted horizons"),
    ("sandstone", 26, "Fine sandstone, trough cross-beds"),
    ("siltstone", 14, "Laminated siltstone, plant debris"),
    ("coal", 4, "Bright-banded coal seam"),
    ("mudstone", 22, "Dark mudstone, siderite nodules"),
    ("sandstone", 34, "Medium sandstone, erosive base"),
    ("conglomerate", 12, "Pebbly channel lag"),
    ("siltstone", 20, "Silty interbeds, ripple lamination"),
    ("mudstone", 30, "Marine mudstone, ammonite debris"),
    ("limestone", 26, "Bioclastic crinoidal limestone"),
    ("mudstone", 18, "Calcareous mudstone"),
    ("limestone", 34, "Massive reefal limestone"),
    ("sandstone", 28, "Glauconitic sandstone"),
    ("siltstone", 22, "Bioturbated silty marl"),
    ("mudstone", 40, "Basinal shale, pyritic"),
    ("conglomerate", 16, "Basal conglomerate above unconformity"),
    ("sandstone", 36, "Arkosic red-bed sandstone"),
]

AGES = [
    (0, 130, "Upper Jurassic", lighten("#3b5b92", 0.62)),
    (130, 224, "Middle Jurassic", lighten("#2a7f7a", 0.62)),
    (224, 304, "Lower Jurassic", lighten("#c9a227", 0.62)),
    (304, 400, "Upper Triassic", lighten("#b23a48", 0.62)),
]

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"


def brick_ornament(ax, x0, x1, top, bot, color, dy=5.5, lw=0.5):
    """Staggered brick ruling, the conventional carbonate ornament."""
    clip = Rectangle((x0, top), x1 - x0, bot - top, transform=ax.transData)
    rows = np.arange(top + dy, bot - 1e-6, dy)
    segs = []
    for y in rows:
        segs.append(ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=3)[0])
    for k, y in enumerate(np.arange(top, bot - 1e-6, dy)):
        n = 3
        off = 0.5 if k % 2 else 0.0
        for j in range(n + 1):
            x = x0 + (x1 - x0) * (j + off) / n
            if x0 - 1e-9 <= x <= x1 + 1e-9:
                segs.append(ax.plot([x, x], [y, min(y + dy, bot)],
                                    color=color, lw=lw, zorder=3)[0])
    for s in segs:
        s.set_clip_path(clip)


def spread(vals, gap, lo, hi):
    """Push labels apart while preserving order, so leaders cannot cross."""
    y = [float(v) for v in vals]
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + gap)
    if y[-1] > hi:
        y[-1] = hi
        for i in range(len(y) - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - gap)
    y[0] = max(y[0], lo)
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + gap)
    return y


def main():
    set_pub_style(**{"hatch.linewidth": 0.45})

    tops = np.concatenate([[0.0], np.cumsum([u[1] for u in UNITS])])
    zmax = float(tops[-1])
    mids = 0.5 * (tops[:-1] + tops[1:])
    grain = np.array([LITH[u[0]]["grain"] for u in UNITS])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.06, 0.24, 0.96, 0.68],
                          left=0.058, right=0.945, top=0.885, bottom=0.135,
                          wspace=0.06)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1], sharey=ax_l)
    ax_d = fig.add_subplot(gs[0, 2], sharey=ax_l)
    ax_c = fig.add_subplot(gs[0, 3], sharey=ax_l)

    # ---- lithology column -------------------------------------------------
    for (lith, _, _), t0, t1, g in zip(UNITS, tops[:-1], tops[1:], grain):
        spec = LITH[lith]
        # width is keyed to grain size: the classic ragged column profile
        w = 0.34 + 0.60 * g
        ax_l.add_patch(Rectangle(
            (0.0, t0), w, t1 - t0, facecolor=spec["fc"],
            edgecolor=darken(spec["fc"], 0.42), linewidth=0.6,
            hatch=spec["hatch"], zorder=2))
        if lith == "limestone":
            brick_ornament(ax_l, 0.0, w, t0, t1, darken(spec["fc"], 0.45))

    ax_l.set_xlim(0.0, 1.0)
    ax_l.set_ylim(zmax, 0.0)
    ax_l.set_yticks(np.arange(0, zmax + 1, 50))
    ax_l.set_ylabel("Depth below datum (m)")
    ax_l.set_xticks([0.34, 0.63, 0.94])
    ax_l.set_xticklabels(["clay", "sand", "gravel"], fontsize=6.6,
                         color=MUTED)
    ax_l.tick_params(axis="x", length=2.0)
    ax_l.set_title("Lithology", fontsize=8.5, fontweight="bold")
    for s in ax_l.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)

    # unconformity: the wavy contact above the basal conglomerate
    i_unc = [i for i, u in enumerate(UNITS) if u[0] == "conglomerate"][-1]
    z_unc = tops[i_unc]
    xw = np.linspace(0.0, 1.0, 220)
    ax_l.plot(xw, z_unc + 3.4 * np.sin(xw * 26.0), color=RULE, lw=1.3,
              zorder=6)
    ax_l.annotate("regional unconformity", xy=(0.52, z_unc),
                  xytext=(0.30, z_unc - 26), fontsize=6.8, color=RULE,
                  ha="left", va="center",
                  arrowprops=dict(arrowstyle="->", color=RULE, lw=0.7,
                                  connectionstyle="arc3,rad=0.25"))

    # ---- chronostratigraphy ribbon ---------------------------------------
    for z0, z1, name, col in AGES:
        ax_a.add_patch(Rectangle((0, z0), 1, z1 - z0, facecolor=col,
                                 edgecolor="white", linewidth=0.9, zorder=2))
        ax_a.text(0.5, 0.5 * (z0 + z1), name, rotation=90, ha="center",
                  va="center", fontsize=7.0, color=darken(col, 0.62),
                  fontweight="bold", zorder=3)
    ax_a.set_xlim(0, 1)
    ax_a.set_xticks([])
    ax_a.tick_params(labelleft=False, left=False)
    ax_a.set_title("Age", fontsize=8.5, fontweight="bold")
    for s in ax_a.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)

    # ---- facies descriptions with non-crossing leaders --------------------
    ax_d.set_xlim(0, 1)
    ax_d.axis("off")
    ys = spread(mids, gap=zmax * 0.0295, lo=6.0, hi=zmax - 4.0)
    for (lith, _, desc), zm, y in zip(UNITS, mids, ys):
        ax_d.plot([0.0, 0.055, 0.085], [zm, y, y], color="#b3b3b3", lw=0.6,
                  clip_on=False, zorder=1)
        ax_d.plot([0.085], [y], marker="o", ms=2.1,
                  color=darken(LITH[lith]["fc"], 0.35), zorder=2,
                  clip_on=False)
        ax_d.text(0.105, y, desc, ha="left", va="center", fontsize=7.0,
                  color=INK, zorder=2)
    ax_d.set_title("Facies description", fontsize=8.5, fontweight="bold",
                   loc="left", x=0.085)

    # ---- grain-size / cycle curve ----------------------------------------
    zstep = np.repeat(tops, 2)[1:-1]
    gstep = np.repeat(grain, 2)
    ax_c.fill_betweenx(zstep, 0.0, gstep, color=lighten("#3b5b92", 0.80),
                       lw=0, zorder=1)
    ax_c.plot(gstep, zstep, color="#3b5b92", lw=1.2, zorder=3)
    for gx in (0.20, 0.46, 0.74, 1.00):
        ax_c.axvline(gx, color="#d5d5d5", lw=0.5, zorder=0)

    # fining-upward cycles detected from the grain-size index itself
    runs, i = [], 0
    while i < len(grain) - 1:
        j = i
        while j < len(grain) - 1 and grain[j] <= grain[j + 1]:
            j += 1
        if j - i >= 2:
            runs.append((i, j))
        i = max(j, i + 1)
    for k, (i0, i1) in enumerate(runs):
        z_bot, z_top = tops[i1 + 1], tops[i0]
        ax_c.annotate("", xy=(1.14, z_top), xytext=(1.14, z_bot),
                      annotation_clip=False,
                      arrowprops=dict(arrowstyle="-|>", color="#b23a48",
                                      lw=1.2, shrinkA=0, shrinkB=0))
        ax_c.text(1.20, 0.5 * (z_top + z_bot),
                  f"FU {z_bot - z_top:.0f} m", rotation=90, ha="left",
                  va="center", fontsize=6.4, color="#b23a48",
                  clip_on=False)
    ax_c.set_xlim(0.0, 1.06)
    ax_c.set_xticks([0.20, 0.74])
    ax_c.set_xticklabels(["clay", "sand"], fontsize=6.6, color=MUTED)
    ax_c.tick_params(labelleft=False, length=2.0)
    ax_c.set_title("Cycles", fontsize=8.5, fontweight="bold")
    for s in ax_c.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)

    # ---- shared lithology key --------------------------------------------
    handles = [Rectangle((0, 0), 1, 1, facecolor=v["fc"], hatch=v["hatch"],
                         edgecolor=darken(v["fc"], 0.42), linewidth=0.6,
                         label=v["name"])
               for v in LITH.values()]
    handles.append(Line2D([0], [0], color="#b23a48", lw=1.2,
                          label="fining-upward cycle (FU)"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.008), ncol=7, frameon=False,
               fontsize=7.0, handlelength=1.5, handleheight=1.15,
               columnspacing=1.4)

    n_fu = sum(t[1] - t[0] + 1 for t in runs)
    savefig(fig, "404_geo_stratigraphic_column.png")


if __name__ == "__main__":
    main()
