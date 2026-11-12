"""One UA, two flow arrangements, and the log-mean that separates them.

The same hardware - same conductance, same two streams, same inlets - is piped
first in counterflow and then in parallel flow.  Everything drawn comes out of
one integration: the local driving difference decays exponentially along the
tube, the temperature curves are that difference fed back through each stream's
capacity rate, and the duty is read from the plotted endpoints.  The log-mean
temperature difference is not quoted from a table, it is the horizontal line
that encloses the same area as the driving-difference curve above it, which is
what the "log mean" in LMTD literally means; multiplying it by UA reproduces the
duty the effectiveness relation gives, and that identity is asserted before the
figure is drawn.  Parallel flow loses because its two streams run into each
other: the driving difference collapses to a pinch at the exit, so the cold
stream can never approach the hot inlet as closely as counterflow lets it, and
counterflow is free to send the cold outlet out hotter than the hot outlet.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HOT = "#b23a48"
HOT_L = "#d59a9f"
EMBER = "#c0603f"
COLD = "#3b5b92"
COLD_L = "#9cc3dd"
DEEP = "#134a76"
GREY = "#8a8a8a"

DEG = r"$^\circ$C"

SEED = 1038
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

TH_IN = 120.0              # hot stream inlet, degC
TC_IN = 25.0               # cold stream inlet, degC
M_HOT, CP_HOT = 2.40, 4.18      # kg/s, kJ/(kg K)
M_COLD, CP_COLD = 3.60, 4.18
UA = 22.0                  # overall conductance, kW/K
L_TUBE = 6.0               # active length, m

C_HOT = M_HOT * CP_HOT     # kW/K
C_COLD = M_COLD * CP_COLD
C_MIN = min(C_HOT, C_COLD)
C_MAX = max(C_HOT, C_COLD)
C_R = C_MIN / C_MAX
NTU = UA / C_MIN
Q_MAX = C_MIN * (TH_IN - TC_IN)


def eps_counter(ntu, cr):
    ntu = np.asarray(ntu, dtype=float)
    if abs(cr - 1.0) < 1e-9:
        return ntu / (1.0 + ntu)
    e = np.exp(-ntu * (1.0 - cr))
    return (1.0 - e) / (1.0 - cr * e)


def eps_parallel(ntu, cr):
    ntu = np.asarray(ntu, dtype=float)
    return (1.0 - np.exp(-ntu * (1.0 + cr))) / (1.0 + cr)


def profiles(mode, n=481):
    """Temperature fields along the tube, from the exponential driving law."""
    x = np.linspace(0.0, L_TUBE, n)
    k = UA / L_TUBE                                  # kW/(K m)
    eps = float(eps_counter(NTU, C_R) if mode == "counter"
                else eps_parallel(NTU, C_R))
    q = eps * Q_MAX
    th_out = TH_IN - q / C_HOT
    tc_out = TC_IN + q / C_COLD
    if mode == "counter":
        a = k * (1.0 / C_HOT - 1.0 / C_COLD)
        dt0 = TH_IN - tc_out
    else:
        a = k * (1.0 / C_HOT + 1.0 / C_COLD)
        dt0 = TH_IN - TC_IN
    dt = dt0 * np.exp(-a * x)
    integral = dt0 * (1.0 - np.exp(-a * x)) / a       # exact, kW m / (kW/K)
    th = TH_IN - (k / C_HOT) * integral
    tc = th - dt if mode == "counter" else TC_IN + (k / C_COLD) * integral
    return dict(mode=mode, x=x, th=th, tc=tc, dt=dt, q=q, eps=eps,
                th_out=th_out, tc_out=tc_out)


def lmtd_from_curves(f):
    """Log-mean of the two terminal differences taken off the plotted arrays."""
    d1, d2 = float(f["dt"][0]), float(f["dt"][-1])
    return (d1 - d2) / np.log(d1 / d2)


def main_panel(ax, cf, pf, lm_c, lm_p):
    x = cf["x"]
    ax.fill_between(x, cf["tc"], cf["th"], color="#fbe9e2", edgecolor="none",
                    zorder=1)
    ax.plot(x, cf["th"], color=HOT, lw=2.1, zorder=7)
    ax.plot(x, cf["tc"], color=COLD, lw=2.1, zorder=7)
    ax.plot(pf["x"], pf["th"], color=HOT_L, lw=1.35, ls=(0, (4.6, 2.2)),
            zorder=6)
    ax.plot(pf["x"], pf["tc"], color=COLD_L, lw=1.35, ls=(0, (4.6, 2.2)),
            zorder=6)

    for xa, i, lab, ha, off in ((0.08, 0, 1, "left", 0.16),
                                (L_TUBE - 0.08, -1, 2, "right", -0.16)):
        lo, hi = float(cf["tc"][i]), float(cf["th"][i])
        ax.annotate("", xy=(xa, hi), xytext=(xa, lo),
                    arrowprops=dict(arrowstyle="<->", color=EMBER, lw=1.0,
                                    shrinkA=0, shrinkB=0), zorder=9)
        ax.text(xa + off, 0.5 * (lo + hi),
                r"$\Delta T_{%d}$ = %.1f K" % (lab, hi - lo), fontsize=6.6,
                color=EMBER, ha=ha, va="center", fontweight="bold",
                zorder=10)

    y_lo, y_hi = float(cf["th"][-1]), float(cf["tc"][0])
    ax.plot([2.90, L_TUBE], [y_lo, y_lo], color=GREY, lw=0.7,
            ls=(0, (1.6, 1.8)), zorder=4)
    ax.plot([0.0, 2.90], [y_hi, y_hi], color=GREY, lw=0.7, ls=(0, (1.6, 1.8)),
            zorder=4)
    ax.annotate("", xy=(2.90, y_hi), xytext=(2.90, y_lo),
                arrowprops=dict(arrowstyle="<->", color=SUB, lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=9)
    ax.text(3.02, 65.0, "cross %.1f K" % (y_hi - y_lo), fontsize=6.5,
            color=SUB, ha="left", va="bottom", fontweight="bold", zorder=10)
    ax.text(0.20, 12.2, "counterflow sends the cold outlet out hotter than "
            "the hot outlet;\nparallel flow can never cross its own streams",
            fontsize=6.4, color=SUB, ha="left", va="bottom", linespacing=1.36,
            zorder=10)

    i_p = int(np.argmin(np.abs(pf["x"] - (L_TUBE - 0.05))))
    ax.annotate("parallel flow pinches to %.1f K here:\nthe two streams have "
                "run into each other" % pf["dt"][i_p],
                xy=(pf["x"][i_p], 0.5 * (pf["th"][i_p] + pf["tc"][i_p])),
                xytext=(L_TUBE - 0.35, 96.0), fontsize=6.4, color="#a06a6a",
                ha="right", va="bottom", linespacing=1.34, zorder=10,
                arrowprops=dict(arrowstyle="-", color="#a06a6a", lw=0.8,
                                shrinkA=3, shrinkB=3))

    ax.annotate("", xy=(1.50, 134.0), xytext=(0.30, 134.0),
                arrowprops=dict(arrowstyle="-|>", color=HOT, lw=1.0,
                                shrinkA=0, shrinkB=0), zorder=8)
    ax.text(1.62, 134.0, "hot stream %.2f kg/s, %.2f kW/K"
            % (M_HOT, C_HOT), fontsize=6.4, color=HOT, ha="left",
            va="center", zorder=8)
    ax.annotate("", xy=(0.30, 126.0), xytext=(1.50, 126.0),
                arrowprops=dict(arrowstyle="-|>", color=COLD, lw=1.0,
                                shrinkA=0, shrinkB=0), zorder=8)
    ax.text(1.62, 126.0, "cold stream %.2f kg/s, %.2f kW/K (counterflow)"
            % (M_COLD, C_COLD), fontsize=6.4, color=COLD, ha="left",
            va="center", zorder=8)

    handles = [
        Line2D([], [], color=HOT, lw=2.0, label="hot, counterflow"),
        Line2D([], [], color=COLD, lw=2.0, label="cold, counterflow"),
        Line2D([], [], color=HOT_L, lw=1.3, ls=(0, (4.6, 2.2)),
               label="hot, parallel flow"),
        Line2D([], [], color=COLD_L, lw=1.3, ls=(0, (4.6, 2.2)),
               label="cold, parallel flow"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=6.3,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.94, handlelength=2.0, borderaxespad=0.4,
                    labelspacing=0.28, bbox_to_anchor=(1.0, 0.995))
    leg.set_zorder(12)

    ax.set_xlim(0.0, L_TUBE)
    ax.set_ylim(6.0, 141.0)
    ax.set_xlabel("position along the exchanger (m)")
    ax.set_ylabel("stream temperature (" + DEG + ")")
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Same UA = %.0f kW/K, same inlets: %.0f kW against %.0f kW"
                 % (UA, cf["q"], pf["q"]), fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def driving_panel(ax, cf, pf, lm_c, lm_p):
    ax.fill_between(pf["x"], pf["dt"], lm_p, where=pf["dt"] >= lm_p,
                    color="#f3ded4", edgecolor="none", zorder=2)
    for f, col, lm, name, xt, ha in (
            (cf, DEEP, lm_c, "counterflow", 5.90, "right"),
            (pf, EMBER, lm_p, "parallel flow", 0.10, "left")):
        ax.semilogy(f["x"], f["dt"], color=col, lw=1.6, zorder=6)
        ax.axhline(lm, color=col, lw=1.0, ls=(0, (4.0, 2.2)), zorder=5)
        ax.text(xt, lm * (1.16 if col is DEEP else 0.86),
                "LMTD %.1f K, %s" % (lm, name), fontsize=6.1, color=col,
                ha=ha, va="bottom" if col is DEEP else "top", zorder=8)
    ax.text(0.10, 1.78, "each dashed level encloses the\nsame area as its own "
            "curve - that\nis what the log mean is", fontsize=6.0, color=SUB,
            ha="left", va="bottom", linespacing=1.32, zorder=8)
    ax.set_xlim(0.0, L_TUBE)
    ax.set_ylim(1.6, 190.0)
    ax.set_xlabel("position along the exchanger (m)", fontsize=6.8)
    ax.set_ylabel("driving difference Th - Tc (K)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.40, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("The driving difference decays exponentially", fontsize=8.0,
                 fontweight="bold", color=INK)


def duty_panel(ax, cf, pf):
    ua = np.linspace(0.4, 66.0, 320)
    q_c = eps_counter(ua / C_MIN, C_R) * Q_MAX
    q_p = eps_parallel(ua / C_MIN, C_R) * Q_MAX
    q_p_lim = Q_MAX / (1.0 + C_R)
    ax.plot(ua, q_c, color=DEEP, lw=1.7, zorder=6)
    ax.plot(ua, q_p, color=EMBER, lw=1.7, zorder=6)
    ax.axhline(Q_MAX, color=GREY, lw=0.8, ls=(0, (3.4, 2.0)), zorder=4)
    ax.axhline(q_p_lim, color=EMBER, lw=0.8, ls=(0, (1.8, 1.8)), zorder=4)
    ax.text(65.0, Q_MAX - 16.0, "thermodynamic ceiling Cmin(Th,in - Tc,in) "
            "= %.0f kW" % Q_MAX, fontsize=6.0, color=SUB, ha="right",
            va="top", zorder=8)
    ax.text(65.0, q_p_lim - 16.0, "parallel-flow ceiling %.0f kW" % q_p_lim,
            fontsize=6.0, color=EMBER, ha="right", va="top", zorder=8)
    for f, col in ((cf, DEEP), (pf, EMBER)):
        ax.plot([UA], [f["q"]], marker="o", ms=4.4, mfc="white", mec=col,
                mew=1.3, zorder=9)
    ax.plot([UA, UA], [pf["q"], cf["q"]], color=SUB, lw=0.8,
            ls=(0, (1.6, 1.6)), zorder=5)
    ax.text(UA + 1.6, 0.5 * (pf["q"] + cf["q"]),
            "%.0f kW bought by\nreversing one stream" % (cf["q"] - pf["q"]),
            fontsize=6.1, color=INK, ha="left", va="center", linespacing=1.30,
            zorder=9)
    ax.text(2.0, cf["q"] + 26.0, "counterflow", fontsize=6.4, color=DEEP,
            ha="left", va="bottom", fontweight="bold", zorder=8)
    ax.text(2.0, pf["q"] - 30.0, "parallel flow", fontsize=6.4, color=EMBER,
            ha="left", va="top", fontweight="bold", zorder=8)
    ax.set_xlim(0.0, 66.0)
    ax.set_ylim(0.0, Q_MAX * 1.14)
    ax.set_xlabel("conductance UA (kW/K)", fontsize=6.8)
    ax.set_ylabel("duty Q (kW)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.40)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Buying area cannot rescue parallel flow", fontsize=8.0,
                 fontweight="bold", color=INK)
    return q_p_lim


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
    np.random.default_rng(SEED)

    cf = profiles("counter")
    pf = profiles("parallel")
    lm_c = lmtd_from_curves(cf)
    lm_p = lmtd_from_curves(pf)

    assert abs(cf["tc"][-1] - TC_IN) < 1e-6, \
        "the counterflow cold stream must enter at the far end"
    assert abs(pf["tc"][0] - TC_IN) < 1e-6, \
        "the parallel cold stream must enter at the near end"
    for f, lm in ((cf, lm_c), (pf, lm_p)):
        assert abs(UA * lm - f["q"]) < 1e-3 * f["q"], \
            "UA times the log-mean has to reproduce the duty"
        mean_dt = float(np.trapezoid(f["dt"], f["x"]) / L_TUBE)
        assert abs(mean_dt - lm) < 5e-3 * lm, \
            "the log-mean is the area-average of the driving difference"
    approach_c = TH_IN - cf["tc_out"]
    approach_p = TH_IN - pf["tc_out"]
    assert approach_p > approach_c, \
        "parallel flow must leave a wider cold-outlet approach than counterflow"
    assert cf["q"] > pf["q"], "counterflow must win at equal UA"
    assert cf["tc_out"] > cf["th_out"], "this design should cross"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_dt = axes_inches(fig, AUX_T)
    ax_q = axes_inches(fig, AUX_B)

    main_panel(ax_main, cf, pf, lm_c, lm_p)
    driving_panel(ax_dt, cf, pf, lm_c, lm_p)
    q_p_lim = duty_panel(ax_q, cf, pf)



    panel_labels_inches(fig, [ax_main, ax_dt, ax_q], dx=-0.62, dy=0.14)
    savefig(fig, "1038_ht_exchanger_lmtd.png")


if __name__ == "__main__":
    main()
