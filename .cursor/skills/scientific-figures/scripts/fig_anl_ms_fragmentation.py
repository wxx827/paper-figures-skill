"""Electron-ionisation mass spectrum of a monochlorinated ester, read as a
fragmentation tree rather than as a list of sticks.

Twenty-one monoisotopic fragments are entered once, each with its carbon and
chlorine count.  Every satellite stick on the plot is then *generated* from
those counts by convolving the binomial 13C distribution with the binomial
37Cl distribution - the chlorine term contributing at mass offset two, which
is what produces the M/M+2 signature the inset exists to show.  Nothing about
the isotope pattern is typed in, so the comparison panel on the right is a
real check of the drawn sticks against theory and not a decorative pair of
bars.

Arcs join a precursor to its product and carry the neutral loss, because the
information a reader actually wants from an EI spectrum is which small
molecule left, not the absolute masses.  The lower right panel closes the
loop: every stick is assigned to one of four series and the summed ion
current of each series is stacked from its own members, so the widths there
add up to the heights here.
"""
import textwrap
from math import comb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

SERIES = {
    "M": (RED, "molecular ion and direct losses"),
    "ACYL": (BLUE, "chlorinated acylium series"),
    "ARO": (GREEN, "aromatic hydrocarbon series"),
    "ALI": (GOLD, "low-mass aliphatic ions"),
}
SERIES_ORDER = ["M", "ACYL", "ARO", "ALI"]

# m/z of the monoisotopic ion, relative intensity, carbon count, chlorine
# count, series, printed caption (None = m/z only, or nothing if weak)
FRAGMENTS = [
    (236, 62.0, 12, 1, "M", r"M$^{+\bullet}$"),
    (218, 27.0, 12, 1, "M", None),
    (208, 21.0, 11, 1, "M", None),
    (201, 14.0, 12, 0, "M", None),
    (193, 100.0, 10, 1, "M", "base peak"),
    (177, 26.0, 9, 1, "ACYL", None),
    (165, 54.0, 9, 1, "ACYL", None),
    (149, 38.0, 8, 1, "ACYL", None),
    (139, 45.0, 7, 1, "ACYL", None),
    (128, 18.0, 7, 1, "ACYL", None),
    (121, 30.0, 8, 0, "ARO", None),
    (111, 41.0, 6, 1, "ACYL", None),
    (105, 24.0, 8, 0, "ARO", None),
    (91, 58.0, 7, 0, "ARO", r"C$_7$H$_7^+$"),
    (77, 36.0, 6, 0, "ARO", r"C$_6$H$_5^+$"),
    (75, 16.0, 6, 0, "ARO", None),
    (65, 20.0, 5, 0, "ARO", None),
    (51, 12.0, 4, 0, "ARO", None),
    (43, 68.0, 2, 0, "ALI", r"CH$_3$CO$^+$"),
    (39, 15.0, 3, 0, "ALI", None),
    (29, 9.0, 2, 0, "ALI", None),
]

# precursor m/z, product m/z, neutral loss, apex height of the joining arc.
# Apex rises with the span so that arcs sharing the m/z 236 precursor nest
# instead of crossing.
ARCS = [
    (236, 193, r"$-$43 (C$_3$H$_7$)", 152.0),
    (236, 208, r"$-$28 (CO)", 130.0),
    (236, 218, r"$-$18 (H$_2$O)", 110.0),
    (193, 165, r"$-$28 (CO)", 132.0),
    (139, 111, r"$-$28 (CO)", 92.0),
    (111, 75, r"$-$36 (HCl)", 76.0),
]

P_C13 = 0.0107        # natural abundance of 13C
P_CL37 = 0.2424       # natural abundance of 37Cl, two mass units heavier
N_ISO = 4             # satellites tracked: M, M+1, M+2, M+3
MZ_LO, MZ_HI = 20.0, 262.0
LABEL_MIN = 24.0      # only sticks above this get an m/z caption
PARENT = 236
Y_TOP = 168.0         # headroom above 100 % for the arcs and the inset

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.85, 1.30, 5.55, 4.20)
AUX_T = (7.42, 3.62, 3.18, 1.88)
AUX_B = (7.42, 1.28, 3.18, 1.72)


def isotope_pattern(n_c, n_cl, n=N_ISO):
    """Relative heights of M, M+1, ... M+n-1 for one elemental composition."""
    carbon = np.array([comb(n_c, k) * P_C13 ** k * (1.0 - P_C13) ** (n_c - k)
                       if k <= n_c else 0.0 for k in range(n)])
    chlorine = np.zeros(n)
    for k in range(n_cl + 1):
        if 2 * k < n:
            chlorine[2 * k] = (comb(n_cl, k) * P_CL37 ** k
                               * (1.0 - P_CL37) ** (n_cl - k))
    conv = np.convolve(carbon, chlorine)[:n]
    return conv / conv[0]


def build_sticks(rng):
    """Every drawn stick, satellites included, plus the chemical background."""
    sticks = {}
    for mz, inten, n_c, n_cl, ser, _lab in FRAGMENTS:
        pat = isotope_pattern(n_c, n_cl)
        # A real instrument never reproduces theory exactly; a few percent of
        # relative error is what the comparison panel is there to expose.
        err = 1.0 + rng.normal(0.0, 0.035, N_ISO)
        err[0] = 1.0
        for k in range(N_ISO):
            h = inten * pat[k] * err[k]
            if h < 0.05:
                continue
            key = mz + k
            sticks[key] = (sticks.get(key, (0.0, ser))[0] + h, ser)

    grid = np.arange(int(MZ_LO), int(MZ_HI) + 1)
    noise = np.abs(rng.normal(0.0, 0.55, grid.size)) + 0.15
    top = max(v[0] for v in sticks.values())
    scaled = {k: (100.0 * v[0] / top, v[1]) for k, v in sticks.items()}
    return scaled, grid, noise


def main_panel(ax, sticks, grid, noise):
    ax.vlines(grid, 0.0, noise, color=HAIR, lw=0.7, zorder=2)
    for mz, (h, ser) in sorted(sticks.items()):
        ax.vlines(mz, 0.0, h, color=SERIES[ser][0], lw=1.15, zorder=4)

    for mz, _i, _c, _cl, ser, lab in FRAGMENTS:
        h = sticks[mz][0]
        if h < LABEL_MIN and lab is None:
            continue
        col = SERIES[ser][0]
        ax.text(mz, h + 2.4, "{:d}".format(mz), ha="center", va="bottom",
                fontsize=6.0, color=col, zorder=6)
        if lab:
            ax.text(mz, h + 10.0, lab, ha="center", va="bottom", fontsize=6.2,
                    color=col, fontweight="bold", zorder=6)

    for a, b, loss, apex in ARCS:
        arc(ax, a, sticks[a][0] + 4.0, b, sticks[b][0] + 4.0, apex, loss)

    ax.set_xlim(MZ_LO, MZ_HI)
    ax.set_ylim(0.0, Y_TOP)
    ax.set_yticks(np.arange(0.0, 101.0, 20.0))
    ax.set_xticks(np.arange(20.0, 261.0, 20.0))
    ax.set_xlabel("m/z (-)")
    ax.set_ylabel("relative abundance (%)")
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Electron-ionisation spectrum with the neutral losses that "
                 "link the peaks", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def arc(ax, x0, y0, x1, y1, apex, label):
    """Quadratic arc whose apex is exactly where the caption is printed.

    ``connectionstyle="arc3,rad=..."`` bulges by a fraction of the *display*
    distance, so the crest of the curve lands somewhere that depends on the
    axes limits and the caption drifts off it.  Solving the Bezier for a
    requested apex instead keeps label and curve locked together.
    """
    t = np.linspace(0.0, 1.0, 160)
    ctrl = 2.0 * apex - 0.5 * (y0 + y1)
    x = (1.0 - t) ** 2 * x0 + (1.0 - t) * t * (x0 + x1) + t ** 2 * x1
    y = (1.0 - t) ** 2 * y0 + 2.0 * (1.0 - t) * t * ctrl + t ** 2 * y1
    ax.plot(x, y, color=SUB, lw=0.7, zorder=7)
    ax.add_patch(FancyArrowPatch(
        (x[-8], y[-8]), (x[-1], y[-1]), arrowstyle="-|>", mutation_scale=7.0,
        lw=0.7, color=SUB, shrinkA=0.0, shrinkB=0.0, zorder=7))
    ax.text(0.5 * (x0 + x1), apex + 1.8, label, ha="center", va="bottom",
            fontsize=6.0, color=SUB, zorder=8,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.92))


def cluster_inset(ax, sticks):
    """Magnified molecular-ion cluster, paired to its crop by colour.

    ``indicate_inset_zoom`` would run two connector lines the full width of
    the spectrum to reach a tile parked over the empty low-mass end; the
    matching red frames say the same thing without drawing over the data.
    """
    lo, hi = PARENT - 3, PARENT + 5
    axi = ax.inset_axes([0.035, 0.600, 0.235, 0.330])
    for mz in range(lo, hi + 1):
        h = sticks.get(mz, (0.0, "M"))[0]
        if h <= 0.0:
            continue
        axi.vlines(mz, 0.0, h, color=RED, lw=1.7)
        if h > 1.0:
            axi.text(mz, h + 1.0, "{:d}".format(mz), ha="center", va="bottom",
                     fontsize=6, color=RED)
    axi.set_xlim(lo - 0.6, hi + 0.6)
    axi.set_ylim(0.0, sticks[PARENT][0] * 1.46)
    axi.set_xticks([PARENT, PARENT + 2])
    axi.tick_params(labelsize=6, length=2.0, pad=1.2)
    axi.set_yticks([])
    axi.set_title("M / M+2 cluster, one Cl (m/z {:d}-{:d})".format(lo, hi),
                  fontsize=6, color=RED, pad=2.4)
    axi.patch.set_facecolor("white")
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.9)
        s.set_color(RED)

    ax.add_patch(plt.Rectangle(
        (lo - 0.6, 0.0), (hi - lo) + 1.2, sticks[PARENT][0] * 1.46,
        facecolor="none", edgecolor=RED, linewidth=0.9, zorder=9))
    return axi


def isotope_panel(ax, sticks):
    theory = isotope_pattern(12, 1) * 100.0
    base = sticks[PARENT][0]
    meas = np.array([sticks.get(PARENT + k, (0.0, "M"))[0] / base * 100.0
                     for k in range(N_ISO)])
    pos = np.arange(N_ISO, dtype=float)
    ax.bar(pos - 0.19, theory, width=0.36, color="none", edgecolor=SUB,
           linewidth=0.9, linestyle=(0, (2.4, 1.6)), label="theoretical",
           zorder=4)
    ax.bar(pos + 0.19, meas, width=0.36, color=RED, edgecolor="white",
           linewidth=0.5, label="measured", zorder=4)
    for x, t, m in zip(pos, theory, meas):
        ax.text(x + 0.19, m + 2.0, "{:.1f}".format(m), ha="center",
                va="bottom", fontsize=6, color=RED, zorder=6)
    ax.set_xticks(pos)
    ax.set_xticklabels(["M", "M+1", "M+2", "M+3"], fontsize=6.6)
    ax.set_ylim(0.0, 122.0)
    ax.set_ylabel("abundance relative to M (%)", fontsize=7.2)
    ax.tick_params(axis="y", labelsize=6.4)
    ax.grid(axis="y", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=6.3, frameon=False, loc="upper right",
              handlelength=1.3, borderaxespad=0.2)
    ax.set_title(r"Molecular-ion isotope cluster, C$_{12}$H$_{13}$ClO$_2$",
                 fontsize=8.0, fontweight="bold", color=INK, pad=6.0)
    return theory, meas


def series_panel(ax, sticks):
    total = sum(h for h, _s in sticks.values())
    shares = {}
    for k, ser in enumerate(SERIES_ORDER):
        members = sorted([(mz, h) for mz, (h, s) in sticks.items()
                          if s == ser], key=lambda t: -t[1])
        left = 0.0
        col = SERIES[ser][0]
        for j, (mz, h) in enumerate(members):
            w = 100.0 * h / total
            ax.barh(k, w, left=left, height=0.62,
                    color=lighten(col, 0.10 + 0.62 * j / max(len(members) - 1,
                                                             1)),
                    edgecolor="white", linewidth=0.4, zorder=4)
            if w >= 5.0:
                ax.text(left + w / 2.0, k, "{:d}".format(mz), ha="center",
                        va="center", fontsize=6, color="white",
                        fontweight="bold", zorder=6)
            left += w
        shares[ser] = left
        ax.text(left + 1.2, k, "{:.0f}%".format(left), ha="left",
                va="center", fontsize=6.2, color=col, fontweight="bold",
                zorder=6)

    ax.set_yticks(np.arange(len(SERIES_ORDER)))
    ax.set_yticklabels([SERIES[s][1].split(" series")[0]
                        for s in SERIES_ORDER], fontsize=6.3)
    # The tick labels double as the colour key for the whole figure, which is
    # why the main panel carries no legend of its own.
    for lab, ser in zip(ax.get_yticklabels(), SERIES_ORDER):
        lab.set_color(SERIES[ser][0])
    ax.set_xlim(0.0, 46.0)
    ax.set_ylim(-0.68, len(SERIES_ORDER) - 0.20)
    ax.set_xlabel("share of summed ion current (%)", fontsize=7.2)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Ion current attributed to each fragment series",
                 fontsize=8.0, fontweight="bold", color=INK, pad=14.0)
    ax.text(0.0, 1.015, "segments are the individual m/z within a series; "
            "colours key the spectrum", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.0, color=SUB)
    return shares


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
    rng = np.random.default_rng(937)
    sticks, grid, noise = build_sticks(rng)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_iso = axes_inches(fig, AUX_T)
    ax_ser = axes_inches(fig, AUX_B)

    main_panel(ax_main, sticks, grid, noise)
    cluster_inset(ax_main, sticks)
    theory, meas = isotope_panel(ax_iso, sticks)
    shares = series_panel(ax_ser, sticks)

    dev = float(np.max(np.abs(meas - theory)))
    base_mz = max(sticks.items(), key=lambda t: t[1][0])[0]
    n_sticks = sum(1 for h, _s in sticks.values() if h >= 0.5)

    panel_labels_inches(fig, [ax_main, ax_iso, ax_ser], dx=-0.62, dy=0.14)
    savefig(fig, "937_anl_ms_fragmentation.png")


if __name__ == "__main__":
    main()
