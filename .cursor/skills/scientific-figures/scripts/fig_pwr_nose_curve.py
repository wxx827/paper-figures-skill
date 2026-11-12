"""Nose curves from the two-bus quadratic, not from a sketch.

For a Thevenin source behind a reactance the receiving-end voltage solves
V**4 + (2QX - E**2) V**2 + X**2 (P**2 + Q**2) = 0, so both branches of every
PV curve here are the two roots of that quadratic and the nose is simply the
load at which the discriminant reaches zero.  Three cases are drawn - the
intact corridor, the same corridor after one circuit is lost, and that outage
with a switched capacitor bank supporting the reactive load.  The loading
margins in the lower panel are differences between those computed noses and
the operating point, and the feeder schematic is solved bus by bus with the
same equation at the post-contingency loading.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GREEN = "#2e7d5b"
PURPLE = "#7b5ea7"

SBASE = 100.0            # MVA, so 1.0 p.u. of power is 100 MW
E = 1.02                 # sending-end voltage, p.u.
P_OP = 105.0             # operating load, MW
V_CRIT = 0.95            # lowest acceptable steady-state voltage, p.u.
MARGIN_CRIT = 15.0       # required loading margin, per cent of the load

# label, series reactance (p.u.), load power factor, colour, dash
CASES = [
    ("intact corridor, 2 circuits, X = 0.22 p.u.", 0.22, 0.98, BLUE, "-"),
    ("N-1, one circuit out, X = 0.33 p.u.", 0.33, 0.98, ORANGE, "-"),
    ("N-1 with 40 MVAr capacitor, pf 1.00", 0.33, 1.00, GREEN, "-"),
]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

VMAP = mcolors.LinearSegmentedColormap.from_list(
    "house_volt", ["#8c2f3c", "#c0603f", "#e3d3bd", "#7fa3c4", "#2f4f80"])


def branches(p_mw, x, pf):
    """Upper and lower root of the two-bus voltage quadratic, in p.u."""
    p = np.asarray(p_mw, dtype=float) / SBASE
    q = p * np.tan(np.arccos(pf))
    b = E ** 2 - 2.0 * q * x
    disc = b ** 2 - 4.0 * (x ** 2) * (p ** 2 + q ** 2)
    ok = disc >= 0.0
    hi = np.full(p.shape, np.nan)
    lo = np.full(p.shape, np.nan)
    r = np.sqrt(np.where(ok, disc, 0.0))
    hi[ok] = np.sqrt(np.maximum((b[ok] + r[ok]) / 2.0, 0.0))
    lo[ok] = np.sqrt(np.maximum((b[ok] - r[ok]) / 2.0, 0.0))
    return hi, lo, ok


def nose(x, pf, hi_mw=400.0):
    """Maximum loadability, found on the same grid the curve is drawn on."""
    grid = np.linspace(0.0, hi_mw, 40001)
    _h, _l, ok = branches(grid, x, pf)
    p_max = grid[ok].max()
    v_max = branches(np.array([p_max]), x, pf)[0][0]
    return float(p_max), float(v_max)


def feeder(x_total, pf, p_total):
    """Radial four-load feeder solved bus by bus at a given total load."""
    share = np.array([0.18, 0.24, 0.30, 0.28])
    seg = np.array([0.30, 0.24, 0.24, 0.22]) * x_total
    v = [E]
    tail = np.cumsum(share[::-1])[::-1] * p_total     # downstream aggregate
    for k in range(4):
        p = tail[k] / SBASE
        q = p * np.tan(np.arccos(pf))
        b = v[-1] ** 2 - 2.0 * q * seg[k]
        disc = b ** 2 - 4.0 * (seg[k] ** 2) * (p ** 2 + q ** 2)
        v.append(float(np.sqrt(max((b + np.sqrt(max(disc, 0.0))) / 2.0, 0.0))))
    return np.array(v), share * p_total


def main_panel(ax, curves):
    grid = np.linspace(0.0, 400.0, 4001)
    for (lab, x, pf, col, ls), (p_max, v_max) in zip(CASES, curves):
        hi, lo, ok = branches(grid, x, pf)
        ax.plot(grid[ok], hi[ok], color=col, lw=1.9, ls=ls, zorder=6)
        ax.plot(grid[ok], lo[ok], color=col, lw=1.1, ls=(0, (3.0, 2.0)),
                alpha=0.75, zorder=5)
        ax.plot([p_max], [v_max], marker="o", ms=5.4, mfc=col, mec="white",
                mew=1.0, zorder=8)
        ax.text(p_max + 4.0, v_max, "{:.0f} MW".format(p_max), ha="left",
                va="center", fontsize=6.3, color=col, fontweight="bold",
                zorder=9)

    ax.axhline(V_CRIT, color=RED, lw=1.0, ls=(0, (4.5, 2.4)), zorder=4)
    ax.text(4.0, V_CRIT + 0.008, "0.95 p.u. steady-state voltage criterion",
            ha="left", va="bottom", fontsize=6.4, color=RED, zorder=7)
    ax.axvline(P_OP, color=INK, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    ax.text(P_OP - 4.0, 0.28, "operating load {:.0f} MW".format(P_OP),
            ha="right", va="bottom", fontsize=6.5, color=INK,
            fontweight="bold", rotation=90.0, zorder=7)

    p1 = curves[1][0]
    v_op = branches(np.array([P_OP]), CASES[1][1], CASES[1][2])[0][0]
    ax.annotate("", xy=(p1, 0.615), xytext=(P_OP, 0.615),
                arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=1.1,
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    ax.text(P_OP - 7.0, 0.600, "N-1 loading margin\n{:.0f} MW "
            "({:.0f} %)".format(p1 - P_OP, 100.0 * (p1 - P_OP) / P_OP),
            ha="right", va="bottom", fontsize=6.6, color=ORANGE,
            fontweight="bold", linespacing=1.30, zorder=9)
    ax.plot([P_OP], [v_op], marker="s", ms=5.0, mfc="white", mec=ORANGE,
            mew=1.2, zorder=9)

    ax.text(336.0, 0.88, "dashed lower branch:\nthe unstable root of the "
            "same quadratic", ha="right", va="top", fontsize=6.4, color=SUB,
            linespacing=1.34, zorder=8)

    handles = [Line2D([], [], color=c, lw=1.9, label=l)
               for (l, _x, _pf, c, _ls) in CASES]
    handles += [
        Line2D([], [], color=SUB, lw=1.1, ls=(0, (3.0, 2.0)),
               label="unstable lower branch"),
        Line2D([], [], color="none", marker="o", ms=5.0, mfc=SUB,
               mec="white", label="maximum loadability (nose)"),
        Line2D([], [], color="none", marker="s", ms=5.0, mfc="white",
               mec=ORANGE, mew=1.2, label="post-contingency operating point"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.3, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.8, borderaxespad=0.35, labelspacing=0.28)

    ax.set_xlim(0.0, 340.0)
    ax.set_ylim(0.20, 1.09)
    ax.set_xlabel("receiving-end load (MW)")
    ax.set_ylabel("receiving-end voltage (p.u.)")
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("PV curves of a load pocket fed through one corridor",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def feeder_panel(fig, ax, v, load_mw):
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.zeros(5)
    norm = mcolors.Normalize(vmin=0.90, vmax=1.03)

    for k in range(4):
        ax.plot(xs[k:k + 2], ys[k:k + 2], color="#6f6f6f", lw=2.2,
                solid_capstyle="round", zorder=2)
    for k in range(5):
        ax.add_patch(Circle((xs[k], ys[k]), 0.20, facecolor=VMAP(norm(v[k])),
                            edgecolor="white", linewidth=1.1, zorder=4))
        ax.text(xs[k], 0.30, "B{:d}".format(k + 1), ha="center", va="bottom",
                fontsize=6.2, color=INK, zorder=5)
        ax.text(xs[k], -0.32, "{:.3f}".format(v[k]), ha="center", va="top",
                fontsize=6.0, color=SUB if v[k] >= V_CRIT else RED,
                fontweight="normal" if v[k] >= V_CRIT else "bold", zorder=5)
    for k in range(4):
        ax.add_patch(Polygon([[xs[k + 1] - 0.16, -0.62],
                              [xs[k + 1] + 0.16, -0.62],
                              [xs[k + 1], -0.90]], closed=True,
                             facecolor=ORANGE, edgecolor="white",
                             linewidth=0.6, zorder=4))
        ax.text(xs[k + 1], -0.98, "{:.0f}".format(load_mw[k]), ha="center",
                va="top", fontsize=6.0, color=ORANGE, zorder=5)
    ax.text(xs[0], -0.70, "source", ha="center", va="top", fontsize=6.0,
            color=SUB, zorder=5)
    ax.text(2.0, 1.02, "load (MW) under each bus, voltage in p.u. below",
            ha="center", va="bottom", fontsize=6.1, color=MUTED, zorder=5)

    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-1.55, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Feeder voltages at the N-1 loading", fontsize=8.0,
                 fontweight="bold", color=INK)

    cax = ax.inset_axes([0.18, 0.035, 0.64, 0.055])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=VMAP), cax=cax,
                      orientation="horizontal")
    cb.set_label("bus voltage (p.u.)", fontsize=6.1, color=INK)
    cb.ax.tick_params(labelsize=6.0, width=0.6, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#555555")


def margin_panel(ax, curves):
    names = ["intact", "N-1", "N-1 + 40 MVAr"]
    cols = [c[3] for c in CASES]
    marg = np.array([100.0 * (p - P_OP) / P_OP for (p, _v) in curves])
    y = np.arange(3)[::-1]
    ax.barh(y, marg, height=0.58, color=cols, edgecolor="white",
            linewidth=0.7, zorder=4)
    for yi, m, (p, _v) in zip(y, marg, curves):
        ax.text(m + 2.0, yi, "{:.0f} %  ({:.0f} MW headroom)"
                .format(m, p - P_OP), ha="left", va="center", fontsize=6.1,
                color=SUB, zorder=6)
    ax.axvline(MARGIN_CRIT, color=RED, lw=1.2, ls=(0, (4.0, 2.2)), zorder=6)
    ax.text(MARGIN_CRIT + 1.5, -0.58, "{:.0f} % criterion"
            .format(MARGIN_CRIT), ha="left", va="bottom", fontsize=6.1,
            color=RED, fontweight="bold", zorder=7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlim(0.0, marg.max() * 1.55)
    ax.set_ylim(-0.72, 2.42)
    ax.set_xlabel("loading margin above {:.0f} MW (per cent)".format(P_OP),
                  fontsize=6.6)
    ax.grid(axis="x", lw=0.5, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Margin against the planning criterion", fontsize=8.0,
                 fontweight="bold", color=INK)
    return marg


def audit(fig, foot, axes):
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
    curves = [nose(x, pf) for (_l, x, pf, _c, _ls) in CASES]
    v_bus, load_mw = feeder(CASES[1][1], CASES[1][2], P_OP)

    assert curves[0][0] > curves[2][0] > curves[1][0], "case ordering"
    assert np.all(np.diff(v_bus) < 0.0), "voltage must fall along the feeder"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_net = axes_inches(fig, AUX_T)
    ax_mar = axes_inches(fig, AUX_B)

    main_panel(ax_main, curves)
    feeder_panel(fig, ax_net, v_bus, load_mw)
    marg = margin_panel(ax_mar, curves)



    panel_labels_inches(fig, [ax_main, ax_net, ax_mar], dx=-0.62, dy=0.14)
    savefig(fig, "968_pwr_nose_curve.png")


if __name__ == "__main__":
    main()
