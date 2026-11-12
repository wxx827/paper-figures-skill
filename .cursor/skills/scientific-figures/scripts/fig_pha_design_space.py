"""Design space as an intersection of constraints, not as a response surface.

Three critical quality attributes are modelled over the two process parameters
of a fluid-bed granulation.  Each one contributes a single acceptance boundary
and a lightly tinted acceptable side; the design space is the region where all
three tints overlap, computed as a boolean mask on the grid and filled solid so
that it cannot drift away from the boundaries drawn over it.  The normal
operating range sits inside it, and the control-strategy point at its centre is
the operating point every number in the two side panels is evaluated at: the
sensitivity bars are numerical derivatives there, and the capability curve is
the worst of the three attributes as the temperature is walked away from it.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
TEAL = "#2a7f7a"
RED = "#b23a48"
GREY = "#808080"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

T_LO, T_HI = 45.0, 75.0           # inlet air temperature, deg C
S_LO, S_HI = 8.0, 24.0            # binder spray rate, g/min
NOR_T = (58.0, 64.0)
NOR_S = (16.8, 19.0)
CPK_TARGET = 1.33

# batch-to-batch standard deviation of each attribute at a fixed setting
SIG_D50, SIG_LOD, SIG_DIS = 2.0, 0.030, 1.6


def d50(temp, spray):
    """Granule median diameter in micrometres."""
    u, v = spray - 16.0, temp - 60.0
    return 150.0 + 7.0 * u - 2.2 * v - 0.12 * u ** 2 - 0.015 * v ** 2


def lod(temp, spray):
    """Loss on drying (residual moisture) in per cent."""
    u, v = spray - 16.0, temp - 60.0
    return 2.05 + 0.115 * u - 0.048 * v + 0.0035 * u * v


def dissolution(temp, spray):
    """Dissolved at 30 min, per cent of label claim."""
    u, w = spray - 16.0, temp - 62.0
    return 93.0 - 0.70 * u - 0.30 * w - 0.12 * w ** 2


# name, function, limit, sense ("min" = limit is a floor), sigma, colour, unit
CQAS = [
    ("granule $d_{50}$", d50, 140.0, "min", SIG_D50, BLUE, "$\\mu$m"),
    ("loss on drying", lod, 2.50, "max", SIG_LOD, GREEN, "%"),
    ("dissolution at 30 min", dissolution, 82.0, "min", SIG_DIS, GOLD, "%"),
]

CONTROL = (0.5 * sum(NOR_T), 0.5 * sum(NOR_S))


def acceptable(fn, limit, sense, temp, spray):
    z = fn(temp, spray)
    return z >= limit if sense == "min" else z <= limit


def cpk(fn, limit, sense, sigma, temp, spray):
    z = fn(temp, spray)
    return ((z - limit) if sense == "min" else (limit - z)) / (3.0 * sigma)


def main_panel(ax, tg, sg, mask):
    TT, SS = np.meshgrid(tg, sg)

    for (name, fn, limit, sense, _s, col, unit) in CQAS:
        z = fn(TT, SS)
        ok = np.where(acceptable(fn, limit, sense, TT, SS), 1.0, 0.0)
        ax.contourf(TT, SS, ok, levels=[0.5, 1.5], colors=[col], alpha=0.11,
                    zorder=1)
        ax.contour(TT, SS, z, levels=[limit], colors=[col], linewidths=1.7,
                   zorder=5)

    ax.contourf(TT, SS, np.where(mask, 1.0, 0.0), levels=[0.5, 1.5],
                colors=[lighten(TEAL, 0.58)], zorder=2)
    ax.contour(TT, SS, np.where(mask, 1.0, 0.0), levels=[0.5],
               colors=[TEAL], linewidths=2.1, zorder=6)

    ax.add_patch(Rectangle((NOR_T[0], NOR_S[0]), NOR_T[1] - NOR_T[0],
                           NOR_S[1] - NOR_S[0], facecolor="none",
                           edgecolor=INK, linewidth=1.4, ls=(0, (4.0, 2.0)),
                           zorder=8))
    ax.text(NOR_T[1] + 0.4, NOR_S[1] + 0.10, "NOR", ha="left", va="bottom",
            fontsize=7.0, color=INK, fontweight="bold", zorder=9)
    ax.plot([CONTROL[0]], [CONTROL[1]], marker="*", ms=13.0, mfc=RED,
            mec="white", mew=1.0, zorder=10)
    ax.annotate("control strategy\n{:.0f} $^\\circ$C, {:.1f} g/min".format(
        *CONTROL), xy=CONTROL, xytext=(CONTROL[0] + 1.4, CONTROL[1] - 4.4),
        fontsize=6.8, color=RED, ha="left", va="top", linespacing=1.30,
        zorder=10, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, shrinkA=1,
                        shrinkB=5))

    ax.text(57.6, 14.2, "design space\n(all three attributes in "
                        "specification)", ha="center", va="top", fontsize=7.2,
            color=TEAL, fontweight="bold", linespacing=1.35, zorder=9,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.80))

    labels = [("$d_{50}$ = 140 $\\mu$m", 70.4, 17.2, BLUE, -18.0),
              ("LOD = 2.50 %", 55.0, 16.4, GREEN, 12.0),
              ("dissolution = 82 %", 70.6, 10.6, GOLD, 90.0)]
    for text, x, y, col, rot in labels:
        ax.text(x, y, text, ha="center", va="center", fontsize=6.6,
                color=col, rotation=rot, rotation_mode="anchor", zorder=9,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.88))

    handles = [Line2D([], [], color=col, lw=1.7,
                      label="{:s} limit {:s} {:g} {:s}".format(
                          name, "$\\geq$" if sense == "min" else "$\\leq$",
                          limit, unit))
               for (name, _f, limit, sense, _s, col, unit) in CQAS]
    handles += [
        Patch(facecolor=lighten(TEAL, 0.58), edgecolor=TEAL, lw=1.6,
              label="design space (intersection)"),
        Patch(facecolor="none", edgecolor=INK, lw=1.3, ls="--",
              label="normal operating range"),
        Line2D([], [], color="none", marker="*", ms=9.0, mfc=RED, mec="white",
               mew=0.8, label="control strategy set point"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.7, borderaxespad=0.3, labelspacing=0.30)

    ax.set_xlim(T_LO, T_HI)
    ax.set_ylim(S_LO, S_HI)
    ax.set_xlabel("inlet air temperature ($^\\circ$C)")
    ax.set_ylabel("binder spray rate (g/min)")
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Fluid-bed granulation, three critical quality attributes",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def sensitivity_panel(ax, sens):
    n = len(CQAS)
    y = np.arange(n)[::-1]
    h = 0.32
    for k, (fac, col, lab) in enumerate(
            ((0, "#5f9dc6", "temperature, $\\pm$ 3 $^\\circ$C"),
             (1, "#c0603f", "spray rate, $\\pm$ 1.1 g/min"))):
        vals = [sens[i][fac] for i in range(n)]
        ax.barh(y + (0.5 - k) * h, vals, height=h * 0.92, color=col,
                edgecolor="white", linewidth=0.7, zorder=4, label=lab)
        for yi, v in zip(y, vals):
            ax.text(v + 0.012, yi + (0.5 - k) * h, "{:.2f}".format(v),
                    ha="left", va="center", fontsize=6, color=SUB, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in CQAS], fontsize=6.3)
    ax.set_xlim(0.0, max(max(s) for s in sens) * 1.32)
    ax.set_ylim(-0.60, n - 0.40)
    ax.set_xlabel("fraction of the margin used over half the NOR "
                  "(dimensionless)", fontsize=6.6)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.0, frameon=False,
              handlelength=1.1, borderaxespad=0.25, labelspacing=0.22)
    ax.set_title("Sensitivity at the set point", fontsize=8.0,
                 fontweight="bold", color=INK)


def capability_panel(ax, tg, worst, t_win):
    ax.axhline(CPK_TARGET, color=RED, lw=1.1, ls=(0, (4.5, 2.2)), zorder=5)
    ax.text(T_LO + 0.6, CPK_TARGET + 0.05,
            "$C_{{pk}}$ = {:.2f} target".format(CPK_TARGET), ha="left",
            va="bottom", fontsize=6.1, color=RED, zorder=7)
    if np.isfinite(t_win[0]):
        ax.axvspan(t_win[0], t_win[1], color="#eaf2f1", zorder=0)
        ax.text(0.5 * (t_win[0] + t_win[1]), 0.06,
                "capable window {:.0f} - {:.0f} $^\\circ$C".format(*t_win),
                ha="center", va="bottom", fontsize=6.1, color=TEAL,
                fontweight="bold", zorder=6)
    for x in NOR_T:
        ax.axvline(x, color=INK, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    ax.text(0.5 * sum(NOR_T), 2.42, "NOR", ha="center", va="top",
            fontsize=6.2, color=INK, fontweight="bold", zorder=6)

    ax.plot(tg, worst, color=TEAL, lw=1.7, zorder=6)
    ax.plot([CONTROL[0]], [np.interp(CONTROL[0], tg, worst)], marker="*",
            ms=9.0, mfc=RED, mec="white", mew=0.8, zorder=8)
    ax.set_xlim(T_LO, T_HI)
    ax.set_ylim(0.0, 2.55)
    ax.set_xlabel("inlet air temperature ($^\\circ$C) at {:.1f} g/min"
                  .format(CONTROL[1]), fontsize=7.0)
    ax.set_ylabel("worst-attribute $C_{pk}$", fontsize=7.2)
    ax.grid(lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Capability as the set point drifts", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    """Warn if the footnote box overlaps an in-view tick or axis label."""
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

    tg = np.linspace(T_LO, T_HI, 601)
    sg = np.linspace(S_LO, S_HI, 601)
    TT, SS = np.meshgrid(tg, sg)
    mask = np.ones_like(TT, dtype=bool)
    for (_n, fn, limit, sense, _s, _c, _u) in CQAS:
        mask &= acceptable(fn, limit, sense, TT, SS)
    area = mask.mean() * (T_HI - T_LO) * (S_HI - S_LO)

    t0, s0 = CONTROL
    sens = []
    for (_n, fn, limit, sense, _sg, _c, _u) in CQAS:
        z0 = fn(t0, s0)
        margin = abs(z0 - limit)
        dt = abs(fn(t0 + 3.0, s0) - z0)
        ds = abs(fn(t0, s0 + 1.1) - z0)
        sens.append((dt / margin, ds / margin))

    tline = np.linspace(T_LO, T_HI, 900)
    worst = np.min([[cpk(fn, limit, sense, sig, t, s0) for t in tline]
                    for (_n, fn, limit, sense, sig, _c, _u) in CQAS], axis=0)
    good = tline[worst >= CPK_TARGET]
    t_win = (good.min(), good.max()) if good.size else (np.nan, np.nan)
    cpk0 = float(np.interp(t0, tline, worst))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_sen = axes_inches(fig, AUX_T)
    ax_cap = axes_inches(fig, AUX_B)

    main_panel(ax_main, tg, sg, mask)
    sensitivity_panel(ax_sen, sens)
    capability_panel(ax_cap, tline, worst, t_win)



    panel_labels_inches(fig, [ax_main, ax_sen, ax_cap], dx=-0.62, dy=0.14)
    savefig(fig, "946_pha_design_space.png")


if __name__ == "__main__":
    main()
