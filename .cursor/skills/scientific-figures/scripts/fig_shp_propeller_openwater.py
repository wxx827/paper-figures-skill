"""Open-water characteristics of a fixed-pitch propeller at three pitch ratios.

The main panel is the open-water diagram: thrust coefficient, ten times torque
coefficient and efficiency against advance coefficient.  The efficiency is not
a fourth model - it is J times KT divided by two pi KQ, evaluated on the two
coefficient arrays that are drawn in the same panel, so the curve cannot drift
away from the coefficients it is supposed to summarise.  The design point is
the argmax of that quotient, and the shaded wedge is the thrust loading above
which back cavitation would be expected.

The upper companion panel sweeps pitch ratio continuously and reports the best
efficiency each pitch can reach together with the advance coefficient where it
occurs.  The lower panel closes the loop with the hull: a propeller law puts
the required thrust on the same axis as the delivered thrust, and the crossing
found by interpolation is the operating point for a clean, a design and a
fouled hull.

Illustrative synthetic data in the style of a Wageningen B-series chart.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"
STEEL = "#5b6b7a"
BLUE = "#2e74a8"
PITCH_C = ["#9cc3dd", "#2e74a8", "#134a76"]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.06, 5.26, 4.34)
AUX_T = (7.46, 3.62, 3.05, 1.78)
AUX_B = (7.46, 1.06, 3.05, 1.72)

PITCH = [0.80, 1.00, 1.20]      # pitch-to-diameter ratio, dimensionless
PD_DESIGN = 1.00
K_CAV = 0.62                    # thrust-loading limit KT/J^2, dimensionless
LOADS = [("light trial", 0.150, GREEN),
         ("design service", 0.300, BLUE),
         ("heavy, fouled", 0.600, RED)]


def coefficients(j, pd):
    """Thrust and torque coefficients (both dimensionless) at pitch ratio pd."""
    kt0 = 0.330 * pd + 0.050
    j0t = 0.980 * pd + 0.020
    kq0 = 0.022 * pd ** 3 + 0.032
    j0q = j0t * 1.090
    kt = kt0 * (1.0 - np.clip(j / j0t, 0.0, None) ** 1.60)
    kq = kq0 * (1.0 - np.clip(j / j0q, 0.0, None) ** 1.90)
    return kt, kq


def efficiency(j, kt, kq):
    """Open-water efficiency from the plotted coefficients, dimensionless."""
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = j * kt / (2.0 * np.pi * kq)
    return np.where((kt > 0.0) & (kq > 0.0), eta, np.nan)


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)

    j = np.linspace(0.0, 1.40, 1401)
    curves = []
    for pd in PITCH:
        kt, kq = coefficients(j, pd)
        curves.append((pd, kt, kq, efficiency(j, kt, kq)))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_e = axes_inches(fig, AUX_T)
    ax_m = axes_inches(fig, AUX_B)
    for a in (ax, ax_e, ax_m):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)

    # ---- main: the open-water diagram --------------------------------------
    pd_d, kt_d, kq_d, eta_d = curves[PITCH.index(PD_DESIGN)]
    cav = K_CAV * j ** 2
    wedge = kt_d > cav
    ax.fill_between(j[wedge], cav[wedge], kt_d[wedge], color=RED, alpha=0.13,
                    zorder=2)
    j_cav = float(j[wedge][-1])
    seg_c = j[j <= j_cav * 1.40]
    ax.plot(seg_c, K_CAV * seg_c ** 2, color=RED, lw=1.0, ls=(0, (4.0, 2.2)),
            zorder=3)
    ax.text(0.045, 0.055,
            "back-cavitation wedge on P/D = %.2f:\nthrust loading $K_T/J^2$ "
            "> %.2f below J = %.3f" % (pd_d, K_CAV, j_cav),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4,
            color=RED, linespacing=1.45, zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                      pad=1.6))

    for (pd, kt, kq, eta), col in zip(curves, PITCH_C):
        ok = kt > 0.0
        ax.plot(j[ok], kt[ok], color=col, lw=2.0, zorder=6)
        ax.plot(j[kq > 0.0], (10.0 * kq)[kq > 0.0], color=col, lw=1.5,
                ls=(0, (5.0, 2.2)), zorder=6)
        ax.plot(j, eta, color=col, lw=1.5, ls=(0, (1.2, 1.6)), zorder=6)
        k = int(np.nanargmax(np.where(np.isfinite(eta), eta, -1.0)))
        ax.plot([j[k]], [eta[k]], marker="o", ms=4.2, mfc="white", mec=col,
                mew=1.3, zorder=9)

    ax.plot([], [], color=STEEL, lw=2.0, label="$K_T$ thrust coefficient")
    ax.plot([], [], color=STEEL, lw=1.5, ls=(0, (5.0, 2.2)),
            label="10 $K_Q$ torque coefficient")
    ax.plot([], [], color=STEEL, lw=1.5, ls=(0, (1.2, 1.6)),
            label="$\\eta_0 = J K_T / (2 \\pi K_Q)$")
    for pd, col in zip(PITCH, PITCH_C):
        ax.plot([], [], color=col, lw=2.0, label="P/D = %.2f" % pd)

    k_d = int(np.nanargmax(np.where(np.isfinite(eta_d), eta_d, -1.0)))
    ax.axvline(j[k_d], color=FAINT, lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    ax.annotate("design point on P/D = %.2f, taken from the plotted "
                "quotient:\nJ = %.3f, $K_T$ = %.4f, 10$K_Q$ = %.4f, "
                "$\\eta_0$ = %.3f"
                % (pd_d, j[k_d], kt_d[k_d], 10.0 * kq_d[k_d], eta_d[k_d]),
                xy=(j[k_d], eta_d[k_d]), xytext=(0.035, 0.985),
                textcoords="axes fraction", fontsize=6.8, color=INK,
                ha="left", va="top", linespacing=1.5, zorder=11,
                arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.9,
                                shrinkA=2, shrinkB=4))

    ax.set_xlim(0.0, 1.40)
    ax.set_ylim(0.0, 0.86)
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel("advance coefficient J = $V_A$/(nD) (dimensionless)",
                  fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("$K_T$, 10$K_Q$, $\\eta_0$ (dimensionless)", fontsize=8.4)
    ax.set_title("Open-water diagram at three pitch ratios", fontsize=9.2,
                 fontweight="bold", pad=5.0)
    ax.legend(loc="upper right", frameon=False, fontsize=6.9,
              handlelength=1.9, labelspacing=0.30, borderaxespad=0.35)

    # ---- aux (b): best efficiency against pitch ratio ----------------------
    pds = np.linspace(0.60, 1.45, 120)
    best_eta, best_j = [], []
    for pd in pds:
        kt, kq = coefficients(j, pd)
        eta = efficiency(j, kt, kq)
        k = int(np.nanargmax(np.where(np.isfinite(eta), eta, -1.0)))
        best_eta.append(eta[k])
        best_j.append(j[k])
    best_eta = np.array(best_eta)
    best_j = np.array(best_j)

    ax_e.plot(pds, best_eta, color=GREEN, lw=1.9, zorder=6)
    ax_e.set_xlim(0.60, 1.45)
    ax_e.set_ylim(float(best_eta.min()) * 0.94, float(best_eta.max()) * 1.10)
    ax_e.tick_params(labelsize=6.8)
    ax_e.set_xlabel("pitch ratio P/D (dimensionless)", fontsize=7.5,
                    labelpad=1.5)
    ax_e.set_ylabel("peak $\\eta_0$ (dimensionless)", fontsize=7.5,
                    color=GREEN)
    ax_e.tick_params(axis="y", labelcolor=GREEN)

    ax_e2 = ax_e.twinx()
    ax_e2.plot(pds, best_j, color=GOLD, lw=1.9, ls=(0, (5.0, 2.2)), zorder=6)
    ax_e2.set_ylim(float(best_j.min()) * 0.90, float(best_j.max()) * 1.12)
    ax_e2.spines["top"].set_visible(False)
    ax_e2.tick_params(labelsize=6.8, labelcolor=GOLD, length=3)
    ax_e2.set_ylabel("J at peak (dimensionless)", fontsize=7.5, color=GOLD)
    ax_e2.grid(False)

    for pd, col in zip(PITCH, PITCH_C):
        ax_e.plot([pd], [float(np.interp(pd, pds, best_eta))], marker="o",
                  ms=4.4, mfc="white", mec=col, mew=1.4, zorder=9)
    ax_e.text(0.035, 0.955,
              "coarser pitch buys %.3f of efficiency from P/D %.2f to %.2f,\n"
              "and moves the best J from %.2f to %.2f"
              % (best_eta[-1] - best_eta[0], pds[0], pds[-1], best_j[0],
                 best_j[-1]),
              transform=ax_e.transAxes, ha="left", va="top", fontsize=6.3,
              color=SUB, linespacing=1.45, zorder=10)
    ax_e.set_title("Pitch ratio sets the ceiling", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    # ---- aux (c): matching the propeller to the hull -----------------------
    ok = kt_d > 0.0
    ax_m.plot(j[ok], kt_d[ok], color=STEEL, lw=2.0, zorder=6,
              label="$K_T$ delivered, P/D = %.2f" % pd_d)
    for row, (name, c_load, col) in enumerate(LOADS):
        diff = kt_d - c_load * j ** 2
        i0 = int(np.where(np.diff(np.sign(diff)) != 0)[0][0])
        j_op = float(np.interp(0.0, [diff[i0 + 1], diff[i0]],
                               [j[i0 + 1], j[i0]]))
        kt_op = float(np.interp(j_op, j, kt_d))
        eta_op = float(np.interp(j_op, j[np.isfinite(eta_d)],
                                 eta_d[np.isfinite(eta_d)]))
        seg = j[j <= j_op * 1.18]
        ax_m.plot(seg, c_load * seg ** 2, color=col, lw=1.5,
                  ls=(0, (4.0, 2.0)), zorder=5)
        ax_m.plot([j_op], [kt_op], marker="o", ms=4.8, mfc="white", mec=col,
                  mew=1.4, zorder=9)
        ax_m.text(0.290, 0.955 - 0.088 * row,
                  "%-15s C = %.2f: J = %.3f, $\\eta_0$ = %.3f"
                  % (name, c_load, j_op, eta_op), transform=ax_m.transAxes,
                  ha="left", va="top", fontsize=6.1, color=col, zorder=10)
    ax_m.set_xlim(0.0, 1.10)
    ax_m.set_ylim(0.0, 0.42)
    ax_m.tick_params(labelsize=6.8)
    ax_m.set_xlabel("advance coefficient J (dimensionless)", fontsize=7.5,
                    labelpad=1.5)
    ax_m.set_ylabel("$K_T$ (dimensionless)", fontsize=7.5)
    ax_m.set_title("Load lines pick the operating point", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_m.text(0.985, 0.055, "dashed: $K_T = C J^2$ hull load lines",
              transform=ax_m.transAxes, ha="right", va="bottom",
              fontsize=6.2, color=MUTED, zorder=10)

    panel_labels_inches(fig, [ax, ax_e, ax_m], dx=-0.64, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 1000")
    check_layout(fig, foot, [ax, ax_e, ax_m])
    savefig(fig, "1000_shp_propeller_openwater.png")


if __name__ == "__main__":
    main()
