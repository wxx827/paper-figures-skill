"""Aroma GC-MS profile: mirrored total-ion chromatograms and the odour
activity they imply.

Twenty-two volatiles are placed on a retention-time axis with Gaussian peaks
whose widths grow with retention, so the two total-ion traces are a genuine
sum of the same compound table that feeds every number on the page.  The two
samples are mirrored about zero - fresh-pressed juice upward, thermally
processed juice downward - because a difference of two peaks is far easier to
read as a reflection than as two overlaid traces.  Concentrations come from
one response factor applied to the peak areas, and the odour activity value is
that concentration divided by the compound's orthonasal threshold; anything
above unity is marked on the trace with a filled dot, so the highlighted peaks
on the chromatogram and the bars in the ranking panel are the same quantity
read twice.  The radar aggregates the same OAVs by odour class on a log scale,
which is why processing shows up there as a swing from the fruity and green
sectors into the roasted, caramel and sulfury ones.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

GREEN = "#2e7d5b"        # fruity / fresh
LEAF = "#6bbf8a"         # green, grassy
TEAL = "#2a7f7a"         # citrus
PURPLE = "#8e6bbf"       # floral
OCHRE = "#c0603f"        # roasted
GOLD = "#c9a227"         # caramel / sweet
BLUE = "#3b5b92"         # fatty
RED = "#b23a48"          # sulfury / off-note
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

CLASS_COLOR = {"fruity": GREEN, "green": LEAF, "citrus": TEAL,
               "floral": PURPLE, "roasted": OCHRE, "caramel": GOLD,
               "fatty": BLUE, "sulfury": RED}
CLASS_ORDER = ["fruity", "green", "citrus", "floral", "roasted", "caramel",
               "fatty", "sulfury"]

# name, retention time (min), retention index, class, area A, area B,
# orthonasal threshold in water (ug/L)
COMPOUNDS = [
    ("ethyl acetate", 4.80, 888, "fruity", 120.0, 88.0, 5.0),
    ("ethyl butanoate", 8.60, 1035, "fruity", 210.0, 96.0, 1.0),
    ("ethyl hexanoate", 12.60, 1233, "fruity", 262.0, 132.0, 5.0),
    ("hexanal", 7.90, 1083, "green", 62.0, 208.0, 4.5),
    ("(E)-2-hexenal", 9.40, 1216, "green", 46.0, 128.0, 17.0),
    ("1-hexanol", 11.20, 1355, "green", 82.0, 118.0, 500.0),
    ("3-methylbutan-1-ol", 10.10, 1209, "fruity", 300.0, 322.0, 30000.0),
    ("limonene", 13.40, 1200, "citrus", 152.0, 138.0, 200.0),
    ("octanal", 12.95, 1290, "citrus", 26.0, 34.0, 0.7),
    ("linalool", 16.80, 1547, "floral", 96.0, 58.0, 6.0),
    ("2-phenylethanol", 22.50, 1906, "floral", 180.0, 152.0, 14000.0),
    ("beta-damascenone", 19.60, 1822, "floral", 12.0, 7.0, 0.05),
    ("furfural", 14.90, 1461, "roasted", 30.0, 182.0, 3000.0),
    ("5-methylfurfural", 16.10, 1570, "roasted", 15.0, 96.0, 1100.0),
    ("2-acetyl-1-pyrroline", 15.50, 1338, "roasted", 3.0, 9.4, 0.10),
    ("guaiacol", 21.00, 1860, "roasted", 8.0, 41.0, 3.0),
    ("furaneol", 23.80, 2030, "caramel", 35.0, 92.0, 10.0),
    ("vanillin", 26.50, 2560, "caramel", 22.0, 61.0, 20.0),
    ("nonanal", 15.20, 1391, "fatty", 40.0, 56.0, 1.0),
    ("(E,E)-2,4-decadienal", 20.40, 1810, "fatty", 10.0, 27.0, 0.07),
    ("methional", 13.90, 1450, "sulfury", 5.0, 23.0, 0.20),
    ("dimethyl trisulfide", 12.00, 1375, "sulfury", 2.0, 11.5, 0.10)]

# In retention order, so the three annotation rows below are filled round
# robin and no two captions on the same row can touch.
LABELLED = ["hexanal", "ethyl butanoate", "methional", "2-acetyl-1-pyrroline",
            "linalool", "beta-damascenone", "guaiacol", "furaneol"]

RF = 0.045               # ug/L of headspace per unit peak area
RT_LO, RT_HI = 3.0, 29.0
SAMPLE_A = "fresh-pressed"
SAMPLE_B = "pasteurised 92 C / 30 s"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.82, 3.61, 2.68, 1.89)
AUX_B = (7.82, 1.02, 2.68, 1.82)


def peak_width(rt):
    """Half-width of a capillary peak grows slowly with retention time."""
    return 0.048 + 0.0042 * rt


def trace(x, areas, rng):
    """Total-ion trace: Gaussian peaks on a drifting, noisy baseline."""
    y = np.zeros_like(x)
    for (_n, rt, _ri, _c, _a, _b, _t), area in zip(COMPOUNDS, areas):
        w = peak_width(rt)
        y += area / (w * np.sqrt(2.0 * np.pi)) * np.exp(
            -0.5 * ((x - rt) / w) ** 2)
    base = 24.0 + 12.0 * (x - RT_LO) / (RT_HI - RT_LO)
    return y + base + rng.normal(0.0, 2.2, x.size)


def main_panel(ax, x, ya, yb, oav_a, oav_b, scale):
    ax.axhline(0.0, color=SUB, lw=0.8, zorder=4)
    ax.fill_between(x, 0.0, ya, color=lighten(GREEN, 0.86), lw=0.0, zorder=2)
    ax.fill_between(x, 0.0, -yb, color=lighten(OCHRE, 0.86), lw=0.0, zorder=2)
    ax.plot(x, ya, color=GREEN, lw=1.15, zorder=5)
    ax.plot(x, -yb, color=OCHRE, lw=1.15, zorder=5)

    rows = (1.78, 1.56, 1.34)
    for k, (name, rt, ri, cls, _a, _b, _t) in enumerate(COMPOUNDS):
        top_a = np.interp(rt, x, ya)
        top_b = np.interp(rt, x, yb)
        col = CLASS_COLOR[cls]
        if oav_a[k] >= 1.0:
            ax.plot([rt], [top_a + 2.5], marker="o", ms=3.6, mfc=col,
                    mec="white", mew=0.6, zorder=7)
        if oav_b[k] >= 1.0:
            ax.plot([rt], [-top_b - 2.5], marker="o", ms=3.6, mfc=col,
                    mec="white", mew=0.6, zorder=7)
        if name not in LABELLED:
            continue
        y_txt = rows[LABELLED.index(name) % 3] * scale
        ax.plot([rt, rt], [top_a + 6.0, y_txt - 2.0], lw=0.55, color=HAIR,
                zorder=3)
        ax.text(rt, y_txt, "{:s}\nRI {:d}".format(name, ri), ha="center",
                va="bottom", fontsize=6, color=col, linespacing=1.20,
                zorder=8)

    ax.set_xlim(RT_LO, RT_HI)
    ax.set_ylim(-1.32 * scale, 2.10 * scale)
    ax.set_yticks([-scale, -0.5 * scale, 0.0, 0.5 * scale, scale])
    ax.set_yticklabels(["100", "50", "0", "50", "100"])
    ax.set_xlabel("retention time (min)")
    ax.set_ylabel("relative total-ion abundance (%)")
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(RT_HI - 0.3, 0.60 * scale, SAMPLE_A, ha="right", va="center",
            fontsize=7.4, color=GREEN, fontweight="bold")
    ax.text(RT_HI - 0.3, -0.85 * scale, SAMPLE_B, ha="right", va="center",
            fontsize=7.4, color=OCHRE, fontweight="bold")
    handles = [Line2D([], [], color=SUB, lw=0.0, marker="o", ms=4.2,
                      mfc=SUB, mec="white",
                      label="OAV > 1 (odour active), coloured by class")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.8, frameon=False,
              bbox_to_anchor=(0.012, 1.0), handlelength=1.2,
              borderaxespad=0.0)
    ax.set_title("Mirrored total-ion chromatograms of the same 22 volatiles",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def radar_panel(ax, cls_a, cls_b):
    n = len(CLASS_ORDER)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.concatenate([ang, ang[:1]])
    va = np.concatenate([cls_a, cls_a[:1]])
    vb = np.concatenate([cls_b, cls_b[:1]])
    ax.plot(ring, va, color=GREEN, lw=1.4, zorder=5)
    ax.fill(ring, va, color=GREEN, alpha=0.16, lw=0.0, zorder=3)
    ax.plot(ring, vb, color=OCHRE, lw=1.4, ls=(0, (4.0, 1.8)), zorder=5)
    ax.fill(ring, vb, color=OCHRE, alpha=0.16, lw=0.0, zorder=4)

    # Ticks first, limits last: set_yticks on a polar axes re-expands rmax to
    # cover the requested ticks, so a fixed ladder would shrink the polygons.
    top = float(max(va.max(), vb.max())) * 1.18
    ticks = [t for t in np.arange(0.3, top, 0.3)]
    ax.set_xticks(ang)
    ax.set_xticklabels(CLASS_ORDER, fontsize=6.2, color=SUB)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["{:.1f}".format(t) for t in ticks], fontsize=6,
                       color=MUTED)
    ax.set_ylim(0.0, top)
    ax.tick_params(axis="x", pad=0.5)
    ax.set_rlabel_position(104.0)
    ax.grid(lw=0.5, color=HAIR, alpha=0.9)
    ax.spines["polar"].set_color(HAIR)
    ax.spines["polar"].set_linewidth(0.7)
    ax.set_title(r"Odour load by class, $\log_{10}(1+\Sigma$ OAV$)$ (-)",
                 fontsize=8.0, fontweight="bold", color=INK, pad=9.0)


def rank_panel(ax, names, cls, oav_a, oav_b, n_top=10):
    score = np.maximum(oav_a, oav_b)
    keep = np.argsort(score)[::-1][:n_top][::-1]
    pos = np.arange(len(keep), dtype=float)
    for i, k in enumerate(keep):
        col = CLASS_COLOR[cls[k]]
        ax.barh(pos[i] + 0.20, oav_a[k], height=0.38, color=col,
                edgecolor="white", linewidth=0.4, zorder=4)
        ax.barh(pos[i] - 0.20, oav_b[k], height=0.38, color=lighten(col, 0.52),
                edgecolor="white", linewidth=0.4, hatch="///", zorder=4)
    ax.axvline(1.0, color=RED, lw=1.0, ls=(0, (3.5, 2.0)), zorder=6)
    ax.text(1.0, len(keep) - 0.30, "OAV = 1 perception threshold", ha="center",
            va="center", fontsize=6, color=RED, zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.92))
    ax.set_xscale("log")
    ax.set_xlim(0.05, float(score.max()) * 6.0)
    ax.set_ylim(-0.72, len(keep) + 0.05)
    ax.set_yticks(pos)
    ax.set_yticklabels([names[k] for k in keep], fontsize=6)
    ax.set_xlabel("odour activity value OAV (-)", fontsize=7.4)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Ten strongest odorants", fontsize=8.0, fontweight="bold",
                 color=INK, pad=9.5)
    ax.text(0.0, 1.012, "solid bar = fresh-pressed,   hatched bar = "
            "pasteurised", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.0, color=SUB)


def audit(fig, foot, axes):
    """Report any in-view tick or axis label the footnote would sit on."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(912)

    names = [c[0] for c in COMPOUNDS]
    cls = [c[3] for c in COMPOUNDS]
    area_a = np.array([c[4] for c in COMPOUNDS])
    area_b = np.array([c[5] for c in COMPOUNDS])
    thr = np.array([c[6] for c in COMPOUNDS])
    conc_a, conc_b = area_a * RF, area_b * RF
    oav_a, oav_b = conc_a / thr, conc_b / thr

    cls_a = np.array([np.log10(1.0 + sum(oav_a[i] for i in range(len(names))
                                         if cls[i] == c))
                      for c in CLASS_ORDER])
    cls_b = np.array([np.log10(1.0 + sum(oav_b[i] for i in range(len(names))
                                         if cls[i] == c))
                      for c in CLASS_ORDER])

    x = np.linspace(RT_LO, RT_HI, 6000)
    ya = trace(x, area_a, rng)
    yb = trace(x, area_b, rng)
    scale = float(max(ya.max(), yb.max()))
    ya, yb = 100.0 * ya / scale, 100.0 * yb / scale
    scale = 100.0

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_rad = axes_inches(fig, AUX_T, projection="polar")
    ax_rank = axes_inches(fig, AUX_B)

    main_panel(ax_main, x, ya, yb, oav_a, oav_b, scale)
    radar_panel(ax_rad, cls_a, cls_b)
    rank_panel(ax_rank, names, cls, oav_a, oav_b)

    n_act_a = int(np.sum(oav_a >= 1.0))
    n_act_b = int(np.sum(oav_b >= 1.0))
    top = int(np.argmax(oav_b / np.maximum(oav_a, 1e-9)))

    panel_labels_inches(fig, [ax_main, ax_rad, ax_rank], dx=-0.62, dy=0.14)
    savefig(fig, "912_fud_gcms_aroma_profile.png")


if __name__ == "__main__":
    main()
