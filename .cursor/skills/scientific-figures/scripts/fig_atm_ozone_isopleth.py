"""EKMA ozone isopleth: which precursor cut actually lowers peak ozone.

The classic ozone-isopleth diagram spans a VOC x NOx precursor plane, and the
only line on it that matters operationally is the ridge - the locus of NOx that
maximises ozone at each VOC level.  Above the ridge the chemistry is
VOC-limited and cutting NOx alone *raises* ozone; below it the chemistry is
NOx-limited and NOx control works.  Four monitoring supersites are placed on
the plane and the auxiliary panel re-evaluates the very same response surface
under a 30 % single-precursor cut, so the sign of every bar is derived from the
field that is contoured rather than asserted in the caption.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import panel_labels_inches, safe_colorbar

INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
ACCENT = "#b23a48"          # house red, reserved for the exceedance level
NOX_C = "#3b5b92"           # house blue  - NOx-only control
VOC_C = "#2e7d5b"           # house green - VOC-only control
RIDGE_C = "#3a3a3a"

VOC_LO, VOC_HI = 100.0, 2000.0      # ppbC
NOX_LO, NOX_HI = 2.0, 140.0         # ppb
STANDARD = 70.0                     # ppb, 8 h ozone design value

# (VOC ppbC, NOx ppb, site label, label offset in axes-data units)
SITES = [
    (450.0, 88.0, "urban core", (-40.0, 12.0), "right"),
    (720.0, 58.0, "downtown", (55.0, 9.0), "left"),
    (1140.0, 34.0, "suburban", (60.0, 8.0), "left"),
    (1560.0, 20.0, "rural downwind", (-45.0, 10.0), "right"),
]


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def nox_optimum(voc):
    """NOx level that maximises ozone at a given VOC - the ridge locus."""
    return 0.0781 * np.power(voc, 0.963)


def ozone(voc, nox):
    """Peak 8 h ozone (ppb) on the precursor plane.

    A smooth surrogate for a box-model EKMA run: production grows sub-linearly
    with VOC and peaks in NOx at ``nox_optimum``, falling away again as NOx
    titration takes over.
    """
    r = np.asarray(nox, dtype=float) / nox_optimum(np.asarray(voc, dtype=float))
    return 22.0 + 150.0 * np.power(np.asarray(voc, dtype=float) / 1200.0,
                                   0.50) * r * np.exp(1.0 - r)


def main():
    set_pub_style()
    rng = np.random.default_rng(552)
    # Site precursor pairs carry a small measurement jitter; every number
    # printed on the figure is recomputed from the jittered pair.
    sites = [(v * float(rng.normal(1.0, 0.035)),
              n * float(rng.normal(1.0, 0.045)), name, off, ha)
             for v, n, name, off, ha in SITES]

    voc = np.linspace(VOC_LO, VOC_HI, 420)
    nox = np.linspace(NOX_LO, NOX_HI, 400)
    V, N = np.meshgrid(voc, nox)
    O3 = ozone(V, N)

    # Ridge read off the plotted grid, not from the analytic formula, so it
    # cannot drift away from the contours if the surface is ever retuned.
    ridge_idx = np.argmax(O3, axis=0)
    ridge_nox = nox[ridge_idx]
    interior = (ridge_idx > 0) & (ridge_idx < nox.size - 1)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.58], left=0.062,
                          right=0.965, top=0.885, bottom=0.115, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    levels = np.arange(30.0, 205.0, 10.0)
    cf = ax.contourf(V, N, O3, levels=levels, cmap=get_cmap("cb_sequential_warm"),
                     extend="both")
    rasterize(cf)
    cf.set_edgecolor("face")
    # The exceedance level is drawn once, in red; leaving it in the grey set
    # too would stack two labels on the same line.
    grey = [lv for lv in levels[::2] if abs(lv - STANDARD) > 1e-6]
    cs = ax.contour(V, N, O3, levels=grey, colors="#6b6b6b", linewidths=0.5)
    ax.clabel(cs, cs.levels[::2], inline=True, fontsize=6.0, fmt="%.0f",
              colors="#3a3a3a")

    cse = ax.contour(V, N, O3, levels=[STANDARD], colors=[ACCENT],
                     linewidths=1.5)
    ax.clabel(cse, inline=True, fontsize=6.4, fmt=f"{STANDARD:.0f} ppb",
              colors=ACCENT)

    ax.plot(voc[interior], ridge_nox[interior], color=RIDGE_C, lw=1.5,
            ls=(0, (4.2, 2.0)), zorder=5)
    ax.annotate("ridge line", xy=(1780.0, float(nox_optimum(1780.0))),
                xytext=(1720.0, 132.0), fontsize=7.0, color=RIDGE_C,
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.92),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=RIDGE_C))

    box = dict(boxstyle="round,pad=0.30", facecolor="white",
               edgecolor="#cccccc", linewidth=0.6, alpha=0.90)
    ax.text(700.0, 125.0, "VOC-limited\ncutting $NO_x$ alone raises $O_3$",
            fontsize=7.2, color=INK, ha="center", va="center", bbox=box,
            zorder=7, linespacing=1.35)
    ax.text(1600.0, 64.0, "$NO_x$-limited\n$NO_x$ control is effective",
            fontsize=7.2, color=INK, ha="center", va="center", bbox=box,
            zorder=7, linespacing=1.35)

    # ------------------------------------------------------- site markers
    for vx, nx, name, (dx, dy), ha in sites:
        ax.plot([vx], [nx], marker="o", ms=5.4, mfc="white", mec=INK,
                mew=1.1, zorder=8)
        ax.text(vx + dx, nx + dy, f"{name}\n{ozone(vx, nx):.0f} ppb",
                fontsize=6.6, color=INK, ha=ha, va="bottom", zorder=8,
                linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="none", alpha=0.82))

    # Two control paths, each annotated with the ozone change it delivers.
    paths = [((sites[0][0], sites[0][1]),
              (sites[0][0] * 0.70, sites[0][1] * 0.48)),
             ((sites[2][0], sites[2][1]),
              (sites[2][0], sites[2][1] * 0.50))]
    for (v0, n0), (v1, n1) in paths:
        d = ozone(v1, n1) - ozone(v0, n0)
        ax.annotate("", xy=(v1, n1), xytext=(v0, n0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK,
                                    shrinkA=4.5, shrinkB=1.5,
                                    mutation_scale=9))
        ax.text(0.5 * (v0 + v1) + 42.0, 0.5 * (n0 + n1),
                f"{d:+.0f} ppb", fontsize=6.4, color=INK, ha="left",
                va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="none", alpha=0.85))

    ax.set_xlim(VOC_LO, VOC_HI)
    ax.set_ylim(NOX_LO, NOX_HI)
    ax.set_xlabel("VOC concentration (ppbC)", fontsize=8.0)
    ax.set_ylabel("$NO_x$ concentration (ppb)", fontsize=8.0)
    ax.set_title("Peak 8 h ozone on the precursor plane", fontsize=9.0,
                 fontweight="bold", color=INK)
    ax.tick_params(labelsize=7.0)

    safe_colorbar(fig, cf, ax, label="peak 8 h $O_3$ (ppb)", width=0.11,
                  pad=0.07, labelsize=7.0)

    # --------------------------------------- (b) single-precursor response
    cut = 0.30
    names, d_nox, d_voc = [], [], []
    for vx, nx, name, _off, _ha in sites:
        base = ozone(vx, nx)
        names.append(name)
        d_nox.append(ozone(vx, nx * (1.0 - cut)) - base)
        d_voc.append(ozone(vx * (1.0 - cut), nx) - base)
    d_nox = np.array(d_nox)
    d_voc = np.array(d_voc)

    y = np.arange(len(names))[::-1]
    h = 0.34
    axb.barh(y + h / 2 + 0.02, d_nox, height=h, color=NOX_C, alpha=0.90,
             edgecolor="white", linewidth=0.6, zorder=3,
             label=f"$NO_x$ cut {cut:.0%}")
    axb.barh(y - h / 2 - 0.02, d_voc, height=h, color=VOC_C, alpha=0.90,
             edgecolor="white", linewidth=0.6, zorder=3,
             label=f"VOC cut {cut:.0%}")
    axb.axvline(0.0, color="#808080", lw=0.9, zorder=4)

    span = float(max(np.abs(d_nox).max(), np.abs(d_voc).max()))
    for yy, val in zip(y + h / 2 + 0.02, d_nox):
        off = 0.035 * span * (1 if val >= 0 else -1)
        axb.text(val + off, yy, f"{val:+.1f}", fontsize=6.2, color=SUB,
                 va="center", ha="left" if val >= 0 else "right")
    for yy, val in zip(y - h / 2 - 0.02, d_voc):
        off = 0.035 * span * (1 if val >= 0 else -1)
        axb.text(val + off, yy, f"{val:+.1f}", fontsize=6.2, color=SUB,
                 va="center", ha="left" if val >= 0 else "right")

    axb.set_yticks(y)
    axb.set_yticklabels(names, fontsize=7.0)
    axb.set_xlim(-span * 1.38, span * 1.38)
    axb.set_ylim(-0.75, len(names) - 0.25)
    axb.set_xlabel("change in peak $O_3$ (ppb)", fontsize=7.5)
    axb.set_title("Response to a single-precursor cut", fontsize=8.5,
                  fontweight="bold", color=INK)
    axb.tick_params(labelsize=6.8)
    axb.grid(axis="x", color="#ececec", lw=0.55, zorder=0)
    axb.legend(loc="lower right", fontsize=6.8, frameon=False)
    axb.text(0.0, -0.145, "positive bar = disbenefit ($NO_x$ titration)",
             transform=axb.transAxes, fontsize=6.5, color=MUTED, ha="left",
             va="top")

    n_up = int(np.sum(d_nox > 0.0))
    panel_labels_inches(fig, [ax, axb], dx=-0.42, dy=0.14)

    savefig(fig, "552_atm_ozone_isopleth.png")


if __name__ == "__main__":
    main()
