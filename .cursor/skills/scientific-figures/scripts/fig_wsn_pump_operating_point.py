"""Pump duty point: the intersection is solved, and the efficiency follows it.

One centrifugal pump is drawn at five speeds through the affinity laws, so the
whole family comes from a single rated head-flow parabola.  The system curve is
a static lift plus a friction term, and each duty point is the root of
head(Q) - system(Q) found by bisection rather than read off the drawing.  The
efficiency islands are not decoration either: an iso-efficiency locus under the
affinity laws is itself a parabola through the origin, so each island is closed
between two such parabolas and the arcs of the fastest and slowest speed curve,
and the efficiency quoted at every duty point is the same rated efficiency
function evaluated at the affinity-scaled flow.  The lower side panel spends
that consistency on the question an operator actually asks: throttling to a
reduced flow against slowing the pump down, priced in kWh per cubic metre.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, lighten

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
DEEP = "#134a76"
GREEN = "#2e7d5b"
LEAF = "#6bbf8a"
RED = "#b23a48"
ORANGE = "#c0603f"

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

H0 = 58.0                  # shut-off head at rated speed, m
A_PUMP = 1.944e-5          # head-flow curve coefficient, m per (m3/h)^2
Q_BEP = 900.0              # best efficiency flow at rated speed, m3/h
ETA_MAX = 0.82             # efficiency at the best efficiency point
H_STAT = 26.0              # static lift of the delivery system, m
R_SYS = 2.006e-5           # system friction coefficient, m per (m3/h)^2
SPEEDS = np.array([1.00, 0.94, 0.88, 0.82, 0.76])
N_RATED = 1480.0           # rated shaft speed, rev/min
RHO_G = 1000.0 * 9.81      # water weight density, N/m3
Q_MAX = 1450.0             # flow axis limit, m3/h


def head(q, s=1.0):
    """Affinity-scaled pump head (m) for flow q (m3/h) at speed ratio s."""
    return s ** 2 * H0 - A_PUMP * np.asarray(q, dtype=float) ** 2


def eta(q, s=1.0):
    """Efficiency (-) from the rated curve, entered at the scaled flow."""
    x = np.asarray(q, dtype=float) / (s * Q_BEP)
    return np.clip(ETA_MAX * (2.0 * x - x * x), 0.0, 1.0)


def system(q, h_stat=H_STAT, r=R_SYS):
    return h_stat + r * np.asarray(q, dtype=float) ** 2


def duty(s, h_stat=H_STAT, r=R_SYS):
    """Root of head(Q, s) - system(Q); returns flow, head and efficiency."""
    f = lambda q: head(q, s) - system(q, h_stat, r)
    if f(1.0) <= 0.0:
        return 0.0, h_stat, 0.0
    hi = float(np.sqrt((s ** 2 * H0) / A_PUMP)) - 1.0
    q = brentq(f, 1.0, hi, xtol=1e-10, rtol=1e-14)
    return q, float(head(q, s)), float(eta(q, s))


def shaft_power(q, h, e):
    """Shaft power in kW from the plotted flow, head and efficiency."""
    return RHO_G * (np.asarray(q, dtype=float) / 3600.0) * h / \
        (np.maximum(e, 1e-6) * 1000.0)


def island(level):
    """Closed iso-efficiency island: two parabolas capped by the speed arcs."""
    root = np.sqrt(max(1.0 - level / ETA_MAX, 0.0))
    q_lo, q_hi = Q_BEP * (1.0 - root), Q_BEP * (1.0 + root)
    s_hi, s_lo = SPEEDS.max(), SPEEDS.min()
    xs, ys = [], []
    for s in (s_hi,):                                   # top arc, lo -> hi
        qq = np.linspace(s * q_lo, s * q_hi, 60)
        xs.append(qq)
        ys.append(head(qq, s))
    for q1 in (q_hi,):                                  # right parabola down
        k = head(q1) / q1 ** 2
        qq = np.linspace(s_hi * q1, s_lo * q1, 40)
        xs.append(qq)
        ys.append(k * qq ** 2)
    for s in (s_lo,):                                   # bottom arc, hi -> lo
        qq = np.linspace(s * q_hi, s * q_lo, 60)
        xs.append(qq)
        ys.append(head(qq, s))
    for q1 in (q_lo,):                                  # left parabola up
        k = head(q1) / q1 ** 2
        qq = np.linspace(s_lo * q1, s_hi * q1, 40)
        xs.append(qq)
        ys.append(k * qq ** 2)
    return np.concatenate(xs), np.concatenate(ys)


def control_study(flows):
    """Specific energy of throttling against variable speed, kWh per m3."""
    thr, vfd, spd = [], [], []
    for q in flows:
        h_t = float(head(q, 1.0))                       # valve absorbs the rest
        e_t = float(eta(q, 1.0))
        thr.append(shaft_power(q, h_t, e_t) / q)
        f = lambda s: head(q, s) - system(q)
        s = brentq(f, 0.20, 1.20, xtol=1e-12)
        h_v = float(system(q))
        e_v = float(eta(q, s))
        vfd.append(shaft_power(q, h_v, e_v) / q)
        spd.append(s)
    return np.array(thr), np.array(vfd), np.array(spd)


def main_panel(ax, pts):
    q = np.linspace(1.0, Q_MAX, 500)

    for lev, alpha in ((0.60, 0.16), (0.70, 0.22), (0.78, 0.30)):
        ix, iy = island(lev)
        ax.fill(ix, iy, facecolor=LEAF, alpha=alpha, edgecolor=GREEN,
                linewidth=0.6, zorder=2)
        tip = int(np.argmin(ix))
        ax.text(ix[tip] - 16.0, iy[tip] + 0.7, "{:.0f} %".format(100 * lev),
                fontsize=6.1, color=GREEN, ha="right", va="bottom", zorder=6)

    for k, s in enumerate(SPEEDS):
        col = lighten(DEEP, 0.13 * k)
        hh = head(q, s)
        m = hh > 0.0
        ax.plot(q[m], hh[m], color=col, lw=1.5 if k == 0 else 1.05, zorder=5)
        ax.text(16.0, s ** 2 * H0 + 0.5, "{:.0f} rev/min"
                .format(s * N_RATED), fontsize=6.1, color=col, ha="left",
                va="bottom", zorder=6)

    ax.plot(q, system(q), color=ORANGE, lw=1.8, zorder=6)
    ax.text(1210.0, system(1210.0) - 2.4,
            "system curve\n{:.0f} m static lift + friction".format(H_STAT),
            fontsize=6.4, color=ORANGE, ha="center", va="top",
            linespacing=1.30, zorder=7)

    qs = np.array([p[0] for p in pts])
    hs = np.array([p[1] for p in pts])
    ax.plot(qs, hs, color=RED, lw=0.9, ls=(0, (4.0, 2.2)), zorder=7)
    ax.plot(qs, hs, marker="o", ms=4.6, mfc=RED, mec="white", mew=0.9,
            ls="none", zorder=9)
    q0, h0, e0 = pts[0]
    rows = "\n".join(
        "{:>4.0f} rev/min   {:>4.0f} m3/h   {:>4.1f} m   {:>4.1f} %"
        .format(s * N_RATED, p[0], p[1], 100.0 * p[2])
        for s, p in zip(SPEEDS, pts))
    ax.text(Q_MAX - 22.0, 1.8, "solved duty points\n" + rows, fontsize=6.1,
            color=SUB, ha="right", va="bottom", linespacing=1.42,
            family="DejaVu Sans Mono", zorder=10,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="white",
                      edgecolor="#dddddd", linewidth=0.6, alpha=0.94))

    ax.axvline(Q_BEP, color=GREEN, lw=0.8, ls=(0, (1.6, 1.8)), zorder=4)
    ax.text(Q_BEP - 11.0, 2.0, "best efficiency flow {:.0f} m3/h"
            .format(Q_BEP), rotation=90, fontsize=6.1, color=GREEN,
            ha="left", va="bottom", zorder=6)

    handles = [
        Line2D([], [], color=DEEP, lw=1.5, label="pump head, rated speed"),
        Line2D([], [], color=lighten(DEEP, 0.4), lw=1.05,
               label="affinity-scaled speeds"),
        Line2D([], [], color=ORANGE, lw=1.8, label="system resistance curve"),
        Line2D([], [], color=RED, lw=0.9, ls=(0, (4.0, 2.2)), marker="o",
               ms=4.2, mfc=RED, mec="white", label="solved duty points"),
        Patch(facecolor=LEAF, alpha=0.35, edgecolor=GREEN, lw=0.6,
              label="iso-efficiency islands"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.7, borderaxespad=0.3, labelspacing=0.28)

    ax.set_xlim(0.0, Q_MAX)
    ax.set_ylim(0.0, 62.0)
    ax.set_xlabel("flow rate (m3/h)")
    ax.set_ylabel("total head (m H2O)")
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Head-flow family of one pump at five speeds against its "
                 "system", fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def curve_panel(ax, pts):
    q = np.linspace(20.0, 1250.0, 400)
    for s, col, lab, ha, dx in ((1.00, DEEP, "rated speed", "left", 60.0),
                                (0.88, lighten(DEEP, 0.38), "88 % speed",
                                 "right", -60.0)):
        e = eta(q, s)
        m = e > 0.02
        ax.plot(q[m], 100.0 * e[m], color=col, lw=1.4, zorder=5)
        ax.text(s * Q_BEP + dx, 100.0 * ETA_MAX + 1.8, lab, fontsize=6.1,
                color=col, ha=ha, va="bottom", zorder=6)
    for (qq, hh, ee), s in zip(pts, SPEEDS):
        if s in (1.00, 0.88):
            ax.plot([qq], [100.0 * ee], marker="o", ms=4.4, mfc=RED,
                    mec="white", mew=0.8, zorder=8)
    ax.set_xlim(0.0, 1250.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("flow rate (m3/h)", fontsize=6.8)
    ax.set_ylabel("efficiency (%)", fontsize=6.6, color=GREEN)
    ax.tick_params(axis="y", colors=GREEN)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax2 = ax.twinx()
    for s, col in ((1.00, ORANGE), (0.88, lighten(ORANGE, 0.42))):
        e = eta(q, s)
        m = (e > 0.05) & (head(q, s) > 0.0)
        ax2.plot(q[m], shaft_power(q[m], head(q[m], s), e[m]), color=col,
                 lw=1.2, ls=(0, (4.0, 2.0)), zorder=6)
    ax2.set_ylim(0.0, 240.0)
    ax2.set_ylabel("shaft power (kW)", fontsize=6.6, color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE, labelsize=6.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(ORANGE)
    ax.set_title("Solid: efficiency.  Dashed: shaft power", fontsize=8.0,
                 fontweight="bold", color=INK)


def energy_panel(ax, flows, thr, vfd, spd):
    ax.fill_between(flows, vfd, thr, facecolor="#f2c9ae", alpha=0.55,
                    edgecolor="none", zorder=3)
    ax.plot(flows, thr, color=ORANGE, lw=1.5, zorder=5)
    ax.plot(flows, vfd, color=BLUE, lw=1.5, zorder=6)
    save = 100.0 * (1.0 - vfd / thr)
    k = int(np.argmax(save))
    ax.annotate("{:.0f} % less energy\nat {:.0f} m3/h ({:.0f} rev/min)"
                .format(save[k], flows[k], spd[k] * N_RATED),
                xy=(flows[k], 0.5 * (thr[k] + vfd[k])),
                xytext=(flows[k] + 105.0, thr[k] - 0.012), fontsize=6.1,
                color=ORANGE, ha="left", va="top", linespacing=1.30,
                zorder=8, arrowprops=dict(arrowstyle="-", color=ORANGE,
                                          lw=0.7, shrinkA=2, shrinkB=3))
    ax.text(flows[-1], vfd[-1] - 0.012, "variable speed", fontsize=6.2,
            color=BLUE, ha="right", va="top", zorder=7)
    ax.text(flows[-1], thr[-1] + 0.010, "throttled at rated speed",
            fontsize=6.2, color=ORANGE, ha="right", va="bottom", zorder=7)
    ax.set_xlim(flows[0], flows[-1])
    ax.set_ylim(0.0, float(thr.max()) * 1.22)
    ax.set_xlabel("delivered flow (m3/h)", fontsize=6.8)
    ax.set_ylabel("specific energy (kWh/m3)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Cost of holding a reduced flow", fontsize=8.0,
                 fontweight="bold", color=INK)
    return save


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
    pts = [duty(s) for s in SPEEDS]
    for (q, h, e), s in zip(pts, SPEEDS):
        assert abs(head(q, s) - system(q)) < 1e-8, "duty point off the curves"
        assert abs(e - float(eta(q, s))) < 1e-12, "efficiency must match"
    q0, h0, e0 = pts[0]
    assert abs(float(eta(Q_BEP)) - ETA_MAX) < 1e-12, "BEP is not at Q_BEP"

    # Throttling can only ever reduce the flow, so the comparison lives
    # between the minimum turndown and the natural duty point.
    flows = np.linspace(380.0, pts[0][0] - 12.0, 60)
    thr, vfd, spd = control_study(flows)
    assert np.all(vfd <= thr + 1e-9), "throttling cannot beat variable speed"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_eff = axes_inches(fig, AUX_T)
    ax_nrg = axes_inches(fig, AUX_B)

    main_panel(ax_main, pts)
    curve_panel(ax_eff, pts)
    save = energy_panel(ax_nrg, flows, thr, vfd, spd)


    p0 = shaft_power(q0, h0, e0)

    panel_labels_inches(fig, [ax_main, ax_eff, ax_nrg], dx=-0.62, dy=0.14)
    savefig(fig, "1003_wsn_pump_operating_point.png")


if __name__ == "__main__":
    main()
