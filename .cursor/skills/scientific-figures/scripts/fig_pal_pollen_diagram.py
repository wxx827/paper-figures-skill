"""Percentage pollen diagram from a lake core, with CONISS-style zonation.

The classic palaeoecology plate: one narrow column per taxon, all sharing a
single downward-positive depth axis, each filled against a common 0-50 %
scale so that a small taxon really does look small.  Columns are ordered and
colour-banded by growth form - trees, shrubs, herbs - which is the grouping a
reader needs before any individual curve means anything.

To the right sit the two summary curves that a percentage diagram cannot
carry: total pollen concentration and microscopic charcoal.  The far-right
strip is the zonation, obtained by recursive stratigraphically constrained
splitting of the percentage matrix, so the zone boundaries drawn across every
column come from the data rather than from the eye.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, lighten, savefig
from figkit import axes_inches, inch_axes, panel_labels_inches

GREEN = "#2e7d5b"     # trees
GOLD = "#c9a227"      # shrubs
OCHRE = "#c0603f"     # herbs
BLUE = "#3b5b92"      # pollen concentration
RED = "#b23a48"       # charcoal
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
HAIR = "#cccccc"

DMAX = 850.0          # deepest sample, cm below sediment surface
PCT_MAX = 50.0        # shared percentage scale for every taxon column
N_ZONES = 4

GROUPS = [("trees", GREEN, ["Pinus", "Picea", "Betula", "Quercus"]),
          ("shrubs", GOLD, ["Juniperus", "Corylus", "Alnus"]),
          ("herbs", OCHRE, ["Poaceae", "Artemisia", "Cyperaceae"])]

# Geometry, all in inches on a 12.6 x 7.6 canvas.
FW, FH = 12.6, 7.6
COL_X0, COL_W, COL_GAP = 1.05, 0.70, 0.10
AX_Y0, AX_H = 1.00, 5.15
AUX_X0, AUX_W, AUX_GAP = 9.35, 1.00, 0.16
ZONE_X0, ZONE_W = 11.70, 0.50
HDR_Y = AX_Y0 + AX_H + 0.05
BAND_Y, BAND_H = 6.66, 0.16


def _ramp(z, z0, w):
    """1 for samples shallower than ``z0``, 0 for deeper ones."""
    return 1.0 / (1.0 + np.exp((z - z0) / w))


def _bump(z, c, s):
    return np.exp(-0.5 * ((z - c) / s) ** 2)


def core(seed=31):
    """Percentages, concentration and charcoal down one 850 cm core."""
    rng = np.random.default_rng(seed)
    z = np.arange(0.0, DMAX + 5.0, 5.0)

    raw = {
        "Pinus": 0.10 + 0.50 * _ramp(z, 600, 40) - 0.22 * _ramp(z, 155, 35),
        "Picea": 0.05 + 0.32 * _bump(z, 520, 95) - 0.03 * _ramp(z, 155, 35),
        "Betula": 0.05 + 0.58 * _bump(z, 560, 85) + 0.10 * _ramp(z, 660, 50),
        "Quercus": 0.03 + 0.52 * _ramp(z, 420, 45) - 0.42 * _ramp(z, 155, 40),
        "Juniperus": 0.02 + 0.44 * _bump(z, 645, 45),
        "Corylus": 0.03 + 0.52 * _bump(z, 370, 75) - 0.06 * _ramp(z, 155, 40),
        "Alnus": 0.03 + 0.38 * _ramp(z, 355, 50) - 0.20 * _ramp(z, 155, 40),
        "Poaceae": 0.06 + 0.46 * (1.0 - _ramp(z, 600, 50))
                   + 0.62 * _ramp(z, 155, 32),
        "Artemisia": 0.04 + 0.52 * (1.0 - _ramp(z, 620, 45))
                     + 0.34 * _ramp(z, 150, 30),
        "Cyperaceae": 0.05 + 0.32 * (1.0 - _ramp(z, 660, 60))
                      + 0.10 * _bump(z, 300, 130)
                      + 0.16 * _ramp(z, 150, 30),
    }
    names = [t for _, _, taxa in GROUPS for t in taxa]
    mat = np.vstack([np.clip(raw[t], 0.01, None) for t in names])
    mat = mat * (1.0 + 0.10 * rng.normal(size=mat.shape))
    pct = 100.0 * mat / mat.sum(axis=0, keepdims=True)

    conc = (0.55 + 6.4 * _bump(z, 430, 200) + 1.9 * _ramp(z, 170, 40)) * \
           (1.0 + 0.13 * rng.normal(size=z.size))
    char = (0.30 + 11.5 * _ramp(z, 165, 28) + 1.1 * _bump(z, 640, 40)) * \
           (1.0 + 0.22 * rng.normal(size=z.size))
    return z, names, pct, np.abs(conc), np.abs(char)


def coniss(mat, k=N_ZONES, min_size=7):
    """Stratigraphically constrained splitting: the zonation, from the data.

    Repeatedly cut the *contiguous* segment whose split buys the largest drop
    in within-zone sum of squares.  Keeping the segments contiguous is what
    makes this a stratigraphic zonation rather than an ordinary clustering -
    a zone has to be a depth interval.
    """
    x = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-9)

    def sse(a, b):
        blk = x[a:b]
        return float(((blk - blk.mean(axis=0)) ** 2).sum())

    n = x.shape[0]
    seg = [(0, n)]
    while len(seg) < k:
        best = None
        for i, (a, b) in enumerate(seg):
            if b - a < 2 * min_size:
                continue
            whole = sse(a, b)
            for s in range(a + min_size, b - min_size + 1):
                gain = whole - sse(a, s) - sse(s, b)
                if best is None or gain > best[0]:
                    best = (gain, i, s)
        if best is None:
            break
        _, i, s = best
        a, b = seg.pop(i)
        seg[i:i] = [(a, s), (s, b)]
    return sorted(seg)


def taxon_axes(fig, n):
    return [axes_inches(fig, (COL_X0 + i * (COL_W + COL_GAP), AX_Y0,
                              COL_W, AX_H)) for i in range(n)]


def style_column(ax, first=False):
    ax.set_ylim(DMAX, 0.0)
    ax.set_xlim(0.0, PCT_MAX)
    ax.set_xticks([0.0, 25.0])
    ax.set_xticklabels(["0", "25"], fontsize=6.2)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.tick_params(length=2.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if first:
        ax.set_ylabel("depth below sediment surface (cm)", fontsize=8.0)
        ax.set_yticks(np.arange(0, 851, 100))
    else:
        ax.set_yticks(np.arange(0, 851, 100))
        ax.tick_params(labelleft=False)


def draw_taxa(fig, z, names, pct, bounds):
    axes = taxon_axes(fig, len(names))
    colors = {t: c for _, c, taxa in GROUPS for t in taxa}
    for i, (ax, name) in enumerate(zip(axes, names)):
        col = colors[name]
        ax.fill_betweenx(z, 0.0, pct[i], color=col, alpha=0.50,
                         linewidth=0.0, zorder=2)
        ax.plot(pct[i], z, color=col, lw=0.7, zorder=3)
        for b in bounds:
            ax.axhline(b, color=GREY, lw=0.6, ls=(0, (3.0, 2.2)), zorder=4)
        style_column(ax, first=(i == 0))
    return axes


def aux_column(fig, x0, z, y, color, unit, bounds):
    ax = axes_inches(fig, (x0, AX_Y0, AUX_W, AX_H))
    ax.fill_betweenx(z, 0.0, y, color=color, alpha=0.30, linewidth=0.0,
                     zorder=2)
    ax.plot(y, z, color=color, lw=0.9, zorder=3)
    for b in bounds:
        ax.axhline(b, color=GREY, lw=0.6, ls=(0, (3.0, 2.2)), zorder=4)
    ax.set_ylim(DMAX, 0.0)
    ax.set_xlim(0.0, y.max() * 1.12)
    ax.set_yticks(np.arange(0, 851, 100))
    ax.tick_params(labelleft=False, labelsize=6.2, length=2.2)
    ax.set_xlabel(unit, fontsize=6.6, color=SUB, labelpad=1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def zone_strip(fig, segs, z, bounds):
    ax = axes_inches(fig, (ZONE_X0, AX_Y0, ZONE_W, AX_H))
    tints = [lighten(GREEN, 0.80), lighten(GOLD, 0.78), lighten(OCHRE, 0.80),
             lighten(BLUE, 0.82)]
    n = len(segs)
    for k, (a, b) in enumerate(segs):
        top = z[a]
        bot = z[b - 1] if b < z.size else DMAX
        ax.axhspan(top, bot, facecolor=tints[k % len(tints)],
                   edgecolor="none", zorder=1)
        ax.text(0.5, 0.5 * (top + bot), "PZ-%d" % (n - k), ha="center",
                va="center", fontsize=7.0, color=INK, fontweight="bold",
                zorder=3)
        ax.text(0.5, 0.5 * (top + bot) + 34.0,
                "%d-%d cm" % (round(top), round(bot)), ha="center",
                va="center", fontsize=6.0, color=SUB, zorder=3)
    for b in bounds:
        ax.axhline(b, color="#3a3a3a", lw=0.9, zorder=4)
    ax.set_ylim(DMAX, 0.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 851, 100))
    ax.tick_params(labelleft=False, length=2.2)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(HAIR)
    return ax


def headers(fig, names):
    """Rotated column captions and the growth-form banner above them."""
    ov = inch_axes(fig, (0.0, 0.0, FW, FH), aspect=None, absolute=True,
                   zorder=8)
    ov.patch.set_visible(False)

    for i, name in enumerate(names):
        x = COL_X0 + i * (COL_W + COL_GAP)
        ov.text(x + 0.08, HDR_Y, name, rotation=55, rotation_mode="anchor",
                ha="left", va="bottom", fontsize=7.0, style="italic",
                color=INK)
    ov.text(AUX_X0 + 0.08, HDR_Y, "pollen concentration", rotation=55,
            rotation_mode="anchor", ha="left", va="bottom", fontsize=7.0,
            color=BLUE)
    ov.text(AUX_X0 + AUX_W + AUX_GAP + 0.08, HDR_Y, "charcoal", rotation=55,
            rotation_mode="anchor", ha="left", va="bottom", fontsize=7.0,
            color=RED)
    ov.text(ZONE_X0 + 0.08, HDR_Y, "zones", rotation=55,
            rotation_mode="anchor", ha="left", va="bottom", fontsize=7.0,
            color=SUB)

    i = 0
    for label, color, taxa in GROUPS:
        x0 = COL_X0 + i * (COL_W + COL_GAP)
        wid = len(taxa) * COL_W + (len(taxa) - 1) * COL_GAP
        ov.add_patch(plt.Rectangle((x0, BAND_Y), wid, BAND_H,
                                   facecolor=color, edgecolor="none",
                                   zorder=3))
        ov.text(x0 + wid / 2.0, BAND_Y + BAND_H / 2.0, label, ha="center",
                va="center", fontsize=7.2, color="white", fontweight="bold",
                zorder=4)
        i += len(taxa)
    return ov


def main():
    set_pub_style()
    z, names, pct, conc, char = core()
    segs = coniss(pct.T)
    bounds = [z[a] for a, _ in segs[1:]]

    fig = plt.figure(figsize=(FW, FH))
    ax_taxa = draw_taxa(fig, z, names, pct, bounds)
    ax_conc = aux_column(fig, AUX_X0, z, conc, BLUE,
                         "$10^4$ grains cm$^{-3}$", bounds)
    ax_char = aux_column(fig, AUX_X0 + AUX_W + AUX_GAP, z, char, RED,
                         "particles cm$^{-2}$ yr$^{-1}$", bounds)
    ax_zone = zone_strip(fig, segs, z, bounds)
    headers(fig, names)

    tree_top = pct[:4, z <= 200.0].sum(axis=0).mean()
    tree_mid = pct[:4, (z > 250.0) & (z < 480.0)].sum(axis=0).mean()
    char_top = char[z <= 165.0].mean()
    char_deep = char[z > 300.0].mean()


    panel_labels_inches(fig, [ax_taxa[0], ax_conc, ax_char, ax_zone],
                        dx=-0.55, dy=0.72)
    savefig(fig, "752_pal_pollen_diagram.png")


if __name__ == "__main__":
    main()
