"""What a headset covers of the visual field, and how sharp it has to be.

The main disc is a top-down map of the horizontal visual field: two monocular
fields are declared, and the binocular overlap, the two monocular crescents and
the blind sector behind the head are all derived from their intersection and
union, so no sector width in the figure is typed twice.  The headset wedge is
drawn on the same disc, which turns "110 degrees" into a visible fraction of
what the eyes already see.

The auxiliary panels answer the two questions the disc raises.  Acuity falls
off with eccentricity along one Weymouth-style law, and each headset's angular
pixel pitch is a horizontal line whose crossing with that law - solved, not
eyeballed - is the eccentricity beyond which its pixels stop being the limit.
The bar panel then puts horizontal and vertical coverage side by side against
the human field.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GUIDE = "#808080"
GRID = "#d9d9d9"
ACCENT = "#b23a48"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"
BLIND = "#e2e2e2"

# Monocular horizontal extents, signed degrees (positive = subject's right).
RIGHT_EYE = (-60.0, 104.0)
LEFT_EYE = (-104.0, 60.0)
HUMAN_V = 130.0

HEADSETS = [
    ("Headset A", 96.0, 92.0, 1200.0),
    ("Headset B", 110.0, 98.0, 2160.0),
    ("Headset C", 130.0, 110.0, 3648.0),
    ("Wide prototype", 180.0, 135.0, 3840.0),
]
SHOWN = 1                      # headset drawn on the field disc

# Weymouth-style acuity law: minimum resolvable detail against eccentricity.
MAR0, E2 = 0.72, 2.30
FOVEA_DEG = 5.0


def mar(ecc):
    """Minimum angle of resolution (arcmin) at an eccentricity in degrees."""
    return MAR0 * (1.0 + np.abs(ecc) / E2)


def mar_inverse(arcmin):
    """Eccentricity (deg) at which acuity has coarsened to ``arcmin``."""
    return E2 * (arcmin / MAR0 - 1.0)


def wedge(ax, lo, hi, r0, r1, **kw):
    a = np.deg2rad(np.linspace(lo, hi, max(int(abs(hi - lo)), 2) * 2))
    ax.fill_between(a, r0, r1, **kw)
    return a


def main():
    set_pub_style()

    binoc = (max(RIGHT_EYE[0], LEFT_EYE[0]), min(RIGHT_EYE[1], LEFT_EYE[1]))
    total = (min(RIGHT_EYE[0], LEFT_EYE[0]), max(RIGHT_EYE[1], LEFT_EYE[1]))
    w_binoc = binoc[1] - binoc[0]
    w_total = total[1] - total[0]
    w_blind = 360.0 - w_total
    mono_r = (binoc[1], total[1])
    mono_l = (total[0], binoc[0])

    name_s, h_s, v_s, px_s = HEADSETS[SHOWN]
    hmd = (-h_s / 2.0, h_s / 2.0)
    cover_binoc = 100.0 * min(h_s, w_binoc) / w_binoc
    cover_total = 100.0 * h_s / w_total

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.60],
                          height_ratios=[1.0, 0.80],
                          left=0.035, right=0.965, top=0.875, bottom=0.225,
                          wspace=0.24, hspace=0.58)

    # ---- (a) horizontal visual field, top-down ----------------------------
    ax = fig.add_subplot(gs[:, 0], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.24)
    ax.set_yticks([])
    ax.set_thetagrids(np.arange(0.0, 360.0, 30.0),
                      labels=["0", "30", "60", "90", "120", "150", "180",
                              "150", "120", "90", "60", "30"], fontsize=7.0)
    ax.tick_params(axis="x", pad=0.5)
    ax.grid(True, color=GRID, lw=0.5)
    ax.spines["polar"].set_color(RULE)
    ax.spines["polar"].set_linewidth(0.7)
    ax.set_axisbelow(True)

    wedge(ax, total[1], total[0] + 360.0, 0.0, 1.0, color=BLIND, lw=0,
          zorder=2)
    for lo, hi in (mono_l, mono_r):
        wedge(ax, lo, hi, 0.0, 1.0, color=GREEN, alpha=0.30, lw=0, zorder=3)
    wedge(ax, binoc[0], binoc[1], 0.0, 1.0, color=BLUE, alpha=0.34, lw=0,
          zorder=4)

    for b in (total[0], total[1], binoc[0], binoc[1]):
        ax.plot([np.deg2rad(b)] * 2, [0.0, 1.0], color="white", lw=1.0,
                zorder=5)

    a_h = np.deg2rad(np.linspace(hmd[0], hmd[1], 200))
    ax.plot(a_h, np.full_like(a_h, 1.10), color=ACCENT, lw=1.8, zorder=7)
    for b in hmd:
        ax.plot([np.deg2rad(b)] * 2, [0.0, 1.10], color=ACCENT, lw=1.3,
                zorder=7)
    ax.fill_between(a_h, 0.0, 1.10, color=ACCENT, alpha=0.07, lw=0, zorder=1)

    ax.text(np.deg2rad(0.0), 0.56, f"binocular\n{w_binoc:.0f} deg",
            ha="center", va="center", fontsize=7.4, color=INK,
            linespacing=1.35, zorder=8)
    for ang, lab in ((np.mean(mono_r), "right eye\nonly"),
                     (np.mean(mono_l), "left eye\nonly")):
        ax.text(np.deg2rad(ang), 0.60,
                f"{lab}\n{mono_r[1] - mono_r[0]:.0f} deg", ha="center",
                va="center", fontsize=6.8, color=INK, linespacing=1.30,
                zorder=8)
    ax.text(np.deg2rad(180.0), 0.58, f"blind\n{w_blind:.0f} deg", ha="center",
            va="center", fontsize=7.0, color=SUB, linespacing=1.35, zorder=8)
    ax.text(np.deg2rad(0.0), 0.90, f"{name_s}: {h_s:.0f} deg H", ha="center",
            va="center", fontsize=7.4, color=ACCENT, zorder=9)

    ax.set_title("Horizontal visual field, viewed from above", fontsize=9.0,
                 fontweight="bold", color=INK, pad=10)
    bb = ax.get_position()
    cx = 0.5 * (bb.x0 + bb.x1)
    fig.text(cx, 0.150,
             f"human field {w_total:.0f} deg H and {HUMAN_V:.0f} deg V",
             ha="center", va="bottom", fontsize=6.8, color=SUB)
    fig.text(cx, 0.120,
             f"{name_s} covers {cover_binoc:.0f} % of the binocular overlap, "
             f"{cover_total:.0f} % of the full field",
             ha="center", va="bottom", fontsize=6.8, color=SUB)
    fig.text(cx, 0.090, "sector radius carries no scale; rim angles are "
             "degrees from straight ahead",
             ha="center", va="bottom", fontsize=6.4, color=MUTED)

    # ---- (b) acuity against eccentricity ----------------------------------
    axa = fig.add_subplot(gs[0, 1])
    ecc_max = 30.0
    ecc = np.linspace(0.0, ecc_max, 400)
    axa.axvspan(0.0, FOVEA_DEG, color="#f0f0f0", zorder=0)
    axa.text(FOVEA_DEG + 0.6, 0.22, "fovea", ha="left", va="bottom",
             fontsize=6.4, color=MUTED)
    axa.plot(ecc, mar(ecc), color=INK, lw=1.6, zorder=5,
             label="human acuity limit")

    pitch_cols = [BLUE, ACCENT, GREEN]
    for (name, h_f, _v, px), col in zip(HEADSETS[:3], pitch_cols):
        pitch = h_f * 60.0 / px
        cross = mar_inverse(pitch)
        axa.axhline(pitch, color=col, lw=1.1, ls=(0, (3.2, 2.2)), zorder=4)
        axa.text(ecc_max - 0.6, pitch, f"{name}  {pitch:.2f} arcmin/pixel",
                 ha="right", va="bottom", fontsize=6.4, color=col)
        axa.plot([cross], [pitch], marker="o", ms=4.0, color=col, zorder=6,
                 mec="white", mew=0.6)
        axa.annotate(f"{cross:.0f} deg", xy=(cross, pitch), xytext=(0.0, -9.0),
                     textcoords="offset points", ha="center", va="top",
                     fontsize=6.2, color=col)
    axa.set_xlim(0.0, ecc_max)
    axa.set_ylim(0.0, mar(ecc_max) * 1.05)
    axa.set_xlabel("eccentricity from fixation (deg)", fontsize=8.0)
    axa.set_ylabel("resolvable detail (arcmin)", fontsize=8.0)
    axa.set_title("Where panel pixels stop being the limit", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)
    axa.tick_params(labelsize=6.6)
    axa.grid(True, axis="y", color=GRID, lw=0.5)
    axa.legend(loc="upper left", frameon=False, fontsize=6.8)

    # ---- (c) coverage bars -------------------------------------------------
    axb = fig.add_subplot(gs[1, 1])
    names = [h[0] for h in HEADSETS]
    hs = np.array([h[1] for h in HEADSETS])
    vs = np.array([h[2] for h in HEADSETS])
    y = np.arange(len(names))[::-1]
    axb.barh(y + 0.19, hs, height=0.34, color=BLUE, edgecolor="white",
             linewidth=0.5, zorder=3)
    axb.barh(y - 0.19, vs, height=0.34, color=GREEN, edgecolor="white",
             linewidth=0.5, zorder=3)
    axb.text(4.0, y[0] + 0.19, "horizontal", ha="left", va="center",
             fontsize=6.2, color="white", zorder=5)
    axb.text(4.0, y[0] - 0.19, "vertical", ha="left", va="center",
             fontsize=6.2, color="white", zorder=5)
    axb.axvline(w_total, color=GUIDE, lw=1.0, ls=(0, (3.2, 2.2)), zorder=5)
    axb.axvline(HUMAN_V, color=GUIDE, lw=1.0, ls=(0, (1.6, 1.8)), zorder=5)
    axb.text(w_total - 3.0, len(names) - 0.42, f"human H {w_total:.0f}",
             ha="right", va="top", fontsize=6.4, color=GUIDE)
    axb.text(HUMAN_V - 3.0, len(names) - 0.42, f"human V {HUMAN_V:.0f}",
             ha="right", va="top", fontsize=6.4, color=GUIDE)
    for yy, hv, vv in zip(y, hs, vs):
        axb.text(hv + 3.0, yy + 0.19, f"{hv:.0f}", ha="left", va="center",
                 fontsize=6.2, color=SUB)
        axb.text(vv + 3.0, yy - 0.19, f"{vv:.0f}", ha="left", va="center",
                 fontsize=6.2, color=SUB)
    axb.set_yticks(y)
    axb.set_yticklabels(names, fontsize=6.8)
    axb.set_xlim(0.0, max(w_total, hs.max()) * 1.16)
    axb.set_ylim(-0.66, len(names) - 0.34)
    axb.set_xlabel("field of view (deg)", fontsize=8.0)
    axb.set_title("No shipping headset reaches the human field",
                  fontsize=8.5, fontweight="bold", color=INK, pad=4)
    axb.tick_params(labelsize=6.6)
    axb.grid(True, axis="x", color=GRID, lw=0.5)


    panel_labels_inches(fig, [ax, axa, axb], dx=-0.02, dy=0.34)
    savefig(fig, "651_vr_field_of_view.png")


if __name__ == "__main__":
    main()
