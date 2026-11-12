"""AVO analysis: angle gather, intercept-gradient crossplot, class catalogue.

The main panel is a wiggle / variable-area angle gather.  Every event is
generated from a two-term Shuey response, so the target trough at 1650 ms
brightens with offset because its gradient is negative, not because the
amplitudes were painted in.  Red dots mark the sample that the extraction
reads on each trace; the inset regresses those picks against sin-squared of
the incidence angle, and the intercept and gradient it reports are the
regression coefficients, divided by the wavelet peak so they are comparable
with reflection coefficients.

Companions are 57 % of the gather width: the intercept-gradient crossplot with
its fitted background trend and fluid quadrants, and the four canonical AVO
classes drawn on one shared amplitude scale.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches, shared_limits

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

PICK = "#b23a48"
TRACE = "#333333"
FLUID = {"Brine sand": "#3b5b92", "Oil sand": "#2e7d5b",
         "Gas sand": "#b23a48", "Shale": "#c9a227"}
CLASS_COL = ["#3b5b92", "#2e7d5b", "#b23a48", "#8e6bbf"]

T0, T1, DT = 1400.0, 2000.0, 2.0
N_ANG, ANG_LO, ANG_HI = 22, 2.0, 44.0
T_TARGET = 1650.0
FREQ = 30.0

# (time ms, intercept R0, gradient G)
EVENTS = [(1462.0, 0.085, -0.045), (1548.0, -0.055, 0.020),
          (T_TARGET, -0.115, -0.205), (1694.0, 0.100, 0.175),
          (1772.0, 0.070, -0.040), (1858.0, -0.050, -0.018),
          (1934.0, 0.078, 0.030)]

CLASSES = [("Class I", 0.115, -0.290), ("Class II", 0.012, -0.205),
           ("Class III", -0.135, -0.215), ("Class IV", -0.150, 0.105)]


def ricker(freq, dt, n=101):
    tt = (np.arange(n) - n // 2) * dt * 1e-3
    a = (np.pi * freq * tt) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def shuey(r0, g, theta_deg):
    s = np.sin(np.deg2rad(theta_deg)) ** 2
    return r0 + g * s


def main():
    set_pub_style()
    rng = np.random.default_rng(5533)

    t_ax = np.arange(T0, T1 + DT, DT)
    nt = t_ax.size
    ang = np.linspace(ANG_LO, ANG_HI, N_ANG)
    w = ricker(FREQ, DT)
    w_peak = float(w.max())

    refl = np.zeros((nt, N_ANG))
    for t_ev, r0, g in EVENTS:
        amp = shuey(r0, g, ang)
        # small residual moveout, as on a real NMO-corrected gather
        t_ev_a = t_ev + 9.0 * np.sin(np.deg2rad(ang)) ** 2
        for j in range(N_ANG):
            i = int(round((t_ev_a[j] - T0) / DT))
            if 0 <= i < nt:
                refl[i, j] += amp[j]
    gather = np.apply_along_axis(lambda c: np.convolve(c, w, mode="same"), 0,
                                 refl)
    gather = gather + rng.normal(0.0, 0.0035, gather.shape)

    # ---- extraction on the plotted traces -----------------------------------
    win = np.abs(t_ax - T_TARGET) < 26.0
    idx = np.where(win)[0]
    pick_i = np.array([idx[np.argmin(gather[idx, j])] for j in range(N_ANG)])
    pick_t = t_ax[pick_i]
    pick_a = gather[pick_i, np.arange(N_ANG)] / w_peak
    s2 = np.sin(np.deg2rad(ang)) ** 2
    grad, icpt = np.polyfit(s2, pick_a, 1)
    pred = grad * s2 + icpt
    r2 = 1.0 - np.sum((pick_a - pred) ** 2) / np.sum(
        (pick_a - pick_a.mean()) ** 2)

    # ---- crossplot populations ----------------------------------------------
    pops = {}
    for name, (mu, sd, gmu, gsd, on_trend) in {
            "Shale": (0.005, 0.038, 0.0, 0.028, True),
            "Brine sand": (0.030, 0.030, 0.0, 0.026, True),
            "Oil sand": (-0.060, 0.028, -0.130, 0.038, False),
            "Gas sand": (-0.120, 0.034, -0.225, 0.048, False)}.items():
        n = 220 if on_trend else 130
        r = rng.normal(mu, sd, n)
        g = (-1.35 * r + rng.normal(gmu, gsd, n) if on_trend
             else rng.normal(gmu, gsd, n))
        pops[name] = (r, g)
    bg_r = np.concatenate([pops["Shale"][0], pops["Brine sand"][0]])
    bg_g = np.concatenate([pops["Shale"][1], pops["Brine sand"][1]])
    b_slope, b_icpt = np.polyfit(bg_r, bg_g, 1)

    # ---- canvas ---------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.82, 1.06, 5.25, 4.22))
    ax_ig = axes_inches(fig, (6.95, 3.58, 3.00, 1.70))
    ax_cl = axes_inches(fig, (6.95, 1.06, 3.00, 1.70))

    # ---- angle gather ----------------------------------------------------------
    ax.set_ylim(T1, T0)
    d_ang = float(ang[1] - ang[0])
    ax.set_xlim(ANG_LO - 1.5 * d_ang, ANG_HI + 1.5 * d_ang)
    sc = 1.55 * d_ang / float(np.abs(gather).max())
    for j, a0 in enumerate(ang):
        xx = a0 + gather[:, j] * sc
        ax.plot(xx, t_ax, color=TRACE, lw=0.45, zorder=3)
        ax.fill_betweenx(t_ax, a0, xx, where=xx > a0, facecolor=TRACE,
                         alpha=0.82, lw=0, zorder=2)
        ax.fill_betweenx(t_ax, a0, xx, where=xx < a0, facecolor="#3b5b92",
                         alpha=0.28, lw=0, zorder=2)
    ax.axhline(T_TARGET, color=PICK, lw=0.8, ls=(0, (4, 2.4)), zorder=4)
    ax.plot(ang, pick_t, ls="none", marker="o", ms=2.6, mfc=PICK,
            mec="white", mew=0.4, zorder=6)
    ax.text(ANG_LO - 1.2 * d_ang, T_TARGET - 8.0, "target", color=PICK,
            fontsize=6.8, fontweight="bold", ha="left", va="bottom", zorder=6)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(ang[::2])
    ax.set_xticklabels(["%.0f" % a for a in ang[::2]], fontsize=6.6)
    ax.set_xlabel("Incidence angle (deg)", fontsize=8.0, labelpad=3.0)
    ax.set_ylabel("Two-way time (ms)", fontsize=8.0)
    ax.tick_params(axis="y", labelsize=7.2)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Target trough brightens with angle; shallower events do not",
                 fontsize=8.8, fontweight="bold", pad=22.0)

    # amplitude-versus-angle extraction, overlaid as an inset
    axi = ax.inset_axes([0.575, 0.055, 0.395, 0.255])
    axi.set_zorder(8)
    axi.patch.set_visible(True)
    axi.set_facecolor("white")
    axi.plot(s2, pick_a, ls="none", marker="o", ms=2.8, mfc=PICK,
             mec="white", mew=0.4)
    gs = np.linspace(s2.min(), s2.max(), 40)
    axi.plot(gs, grad * gs + icpt, color=RULE, lw=1.0, ls=(0, (4, 2)))
    axi.axhline(0.0, color=MUTED, lw=0.6)
    axi.set_xlabel(r"$\sin^2\theta$ (-)", fontsize=6.2, labelpad=1.5)
    axi.set_ylabel("amplitude (-)", fontsize=6.2, labelpad=1.5)
    axi.tick_params(labelsize=6.0, length=2.0, width=0.6, pad=1.5)
    for s in axi.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color(RULE)
    axi.text(0.96, 0.94, "R0 = %.3f,  G = %.3f\n$R^2$ = %.3f, n = %d"
             % (icpt, grad, r2, N_ANG), transform=axi.transAxes, ha="right",
             va="top", fontsize=6.0, color=INK, linespacing=1.4)

    # ---- intercept-gradient crossplot --------------------------------------------
    ax_ig.axhline(0.0, color=MUTED, lw=0.7)
    ax_ig.axvline(0.0, color=MUTED, lw=0.7)
    for name, (r, g) in pops.items():
        ax_ig.scatter(r, g, s=5, color=FLUID[name], alpha=0.55,
                      linewidths=0.0, label=name)
    gx = np.linspace(-0.24, 0.20, 40)
    ax_ig.plot(gx, b_slope * gx + b_icpt, color=RULE, lw=1.2, ls=(0, (4, 2)))
    ax_ig.fill_between(gx, b_slope * gx + b_icpt - 0.075,
                       b_slope * gx + b_icpt + 0.075, facecolor=MUTED,
                       alpha=0.13, lw=0)
    ax_ig.plot([icpt], [grad], marker="*", ms=11.0, mfc=PICK, mec="white",
               mew=0.7, zorder=7)
    tt = ax_ig.annotate("target", (icpt, grad), textcoords="offset points",
                        xytext=(7, -1), fontsize=6.4, color=PICK,
                        fontweight="bold", zorder=8, va="center")
    tt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    ax_ig.set_xlim(-0.26, 0.20)
    ax_ig.set_ylim(-0.40, 0.28)
    for xq, yq, lab in ((-0.245, -0.375, "Class III"),
                        (-0.245, 0.245, "Class IV"),
                        (0.185, -0.375, "Class I / II")):
        q = ax_ig.text(xq, yq, lab, fontsize=6.2, color=SUB,
                       ha="left" if xq < 0 else "right", va="bottom")
        q.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    ax_ig.set_xlabel("Intercept R0 (-)", fontsize=7.4)
    ax_ig.set_ylabel("Gradient G (-)", fontsize=7.4)
    ax_ig.tick_params(labelsize=6.6)
    ax_ig.grid(color=GRID, lw=0.4)
    ax_ig.set_axisbelow(True)
    ax_ig.legend(frameon=False, fontsize=6.2, loc="upper right",
                 labelspacing=0.28, handletextpad=0.4, markerscale=1.6)
    ax_ig.set_title("Hydrocarbon sands leave the background trend",
                    fontsize=8.5, fontweight="bold")

    # ---- forward-modelled AVO classes ----------------------------------------------
    th = np.linspace(0.0, 45.0, 120)
    curves = [shuey(r0, g, th) for _, r0, g in CLASSES]
    lo, hi = shared_limits(curves + [pick_a], pad=0.10)
    for (name, r0, g), c, col in zip(CLASSES, curves, CLASS_COL):
        ax_cl.plot(th, c, color=col, lw=1.4, label=name)
    ax_cl.plot(ang, pick_a, ls="none", marker="o", ms=3.0, mfc=PICK,
               mec="white", mew=0.4, zorder=6, label="Extracted target")
    ax_cl.axhline(0.0, color=MUTED, lw=0.7)
    ax_cl.set_ylim(lo, hi)
    ax_cl.set_xlim(0.0, 45.0)
    ax_cl.set_xlabel("Incidence angle (deg)", fontsize=7.4)
    ax_cl.set_ylabel("Reflection coefficient (-)", fontsize=7.4)
    ax_cl.tick_params(labelsize=6.6)
    ax_cl.grid(color=GRID, lw=0.4)
    ax_cl.set_axisbelow(True)
    ax_cl.legend(frameon=False, fontsize=6.2, loc="lower left", ncol=2,
                 labelspacing=0.26, columnspacing=1.0, handlelength=1.3,
                 handletextpad=0.5)
    ax_cl.set_title("Extracted target follows the Class III template",
                    fontsize=8.5, fontweight="bold")

    bright = abs(pick_a[-1] / pick_a[0])
    panel_labels_inches(fig, [ax, ax_ig, ax_cl], dx=-0.58, dy=0.34)
    savefig(fig, "814_seis_avo_gather.png")


if __name__ == "__main__":
    main()
