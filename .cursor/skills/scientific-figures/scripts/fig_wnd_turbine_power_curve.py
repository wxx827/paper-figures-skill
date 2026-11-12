"""Turbine power curve, thrust coefficient and the energy each wind bin buys.

The main panel carries the measured power curve of a synthetic 6 MW machine:
ten-minute bin means with scatter, the warranted curve supplied with the
turbine, and a fitted curve built from a rotor power coefficient.  The three
operating regions - below cut-in, partial load, full load - are shaded and
their boundaries are read back out of the fitted curve rather than typed in.
The thrust coefficient shares the panel on a right-hand axis.

The two companion panels multiply out to the annual yield: the Weibull speed
distribution of the site, and the product of power curve and frequency, which
is the energy each 1 m/s bin actually contributes over a year.
"""
import textwrap

import numpy as np
from scipy.special import gamma as gamma_fn

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
GRID = "#d9d9d9"

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"

RHO = 1.225
DIAM = 150.0
AREA = np.pi * (DIAM / 2.0) ** 2
P_RATED = 6.0e6          # W
U_IN, U_OUT = 3.0, 25.0
U_MAX = 27.0
WB_K, WB_A = 2.10, 8.60  # Weibull shape (-) and scale (m/s)
HOURS = 8760.0


CP_MAX = 0.470


def power_w(u):
    """Electrical power (W): cubic capture, soft-saturated at rated."""
    u = np.asarray(u, dtype=float)
    r = 0.5 * RHO * AREA * CP_MAX * u ** 3 / P_RATED
    n = 12.0
    p = P_RATED * r / (1.0 + r ** n) ** (1.0 / n)
    return np.where((u >= U_IN) & (u <= U_OUT), p, 0.0)


def cp_effective(u):
    """Power coefficient implied by the plotted curve: pitch-limited above rated."""
    u = np.asarray(u, dtype=float)
    denom = 0.5 * RHO * AREA * np.maximum(u, 1e-6) ** 3
    return np.where(u >= U_IN, power_w(u) / denom, 0.0)


def thrust_ct(u):
    """Thrust coefficient: near-constant below rated, shed above it."""
    u = np.asarray(u, dtype=float)
    ct = 0.86 * np.ones_like(u)
    ramp = np.clip((u - U_IN) / 1.6, 0.0, 1.0)
    shed = np.clip(11.0 / np.maximum(u, 1e-9), 0.0, 1.0) ** 2.6
    return np.where((u >= U_IN) & (u <= U_OUT), ct * ramp * shed, 0.0)


def weibull_pdf(u, k=WB_K, a=WB_A):
    return (k / a) * (u / a) ** (k - 1.0) * np.exp(-(u / a) ** k)


def weibull_cdf(u, k=WB_K, a=WB_A):
    return 1.0 - np.exp(-(u / a) ** k)


def main():
    set_pub_style()
    rng = np.random.default_rng(715)

    ug = np.linspace(0.0, U_MAX, 900)
    p_fit = power_w(ug) / 1e6                       # MW
    ct_fit = thrust_ct(ug)

    # rated speed read off the fitted curve, not typed in
    above = np.where(p_fit >= 0.995 * P_RATED / 1e6)[0]
    u_rated = float(ug[above[0]])

    # measured 10 min bin means: scatter grows with turbulence-driven spread
    u_meas = np.sort(rng.uniform(U_IN - 0.6, U_OUT + 0.4, 190))
    base = power_w(u_meas) / 1e6
    spread = 0.055 * P_RATED / 1e6 * np.exp(-((u_meas - 9.0) / 6.0) ** 2) + 0.04
    p_meas = np.clip(base + rng.normal(0.0, spread), 0.0, None)

    p_warr = 0.965 * p_fit                          # warranted curve

    # ---- annual energy: power curve x frequency --------------------------
    edges = np.arange(0.0, U_MAX + 1.0, 1.0)
    mid = 0.5 * (edges[:-1] + edges[1:])
    freq = weibull_cdf(edges[1:]) - weibull_cdf(edges[:-1])
    hrs = HOURS * freq
    e_bin = power_w(mid) / 1e6 * hrs / 1e3          # GWh/a
    aep = float(e_bin.sum())
    u_mean = WB_A * gamma_fn(1.0 + 1.0 / WB_K)
    cf = aep * 1e3 / (P_RATED / 1e6 * HOURS)
    i_peak = int(np.argmax(e_bin))

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.88, 0.82, 4.72, 4.68))
    ax1 = axes_inches(fig, (6.85, 3.30, 2.85, 2.20))
    ax2 = axes_inches(fig, (6.85, 0.82, 2.85, 1.95))

    # ---- main panel -------------------------------------------------------
    zones = [(0.0, U_IN, "#f0f0f0", "below\ncut-in"),
             (U_IN, u_rated, lighten(BLUE, 0.93), "partial load"),
             (u_rated, U_OUT, lighten(BLUE, 0.84), "full load"),
             (U_OUT, U_MAX, "#f0f0f0", "above\ncut-out")]
    for x0, x1, fc, lab in zones:
        ax.axvspan(x0, x1, facecolor=fc, edgecolor="none", zorder=0)
        ax.text(0.5 * (x0 + x1), 6.62, lab, ha="center", va="top",
                fontsize=6.6, color=SUB, linespacing=1.2, zorder=2)
    for xv, lab in ((U_IN, f"cut-in {U_IN:.0f} m/s"),
                    (u_rated, f"rated {u_rated:.1f} m/s"),
                    (U_OUT, f"cut-out {U_OUT:.0f} m/s")):
        ax.axvline(xv, color=GREY, lw=0.85, ls=(0, (4, 2)), zorder=1)
        ax.text(xv + 0.22, 0.14, lab, fontsize=6.6, color=SUB, rotation=90,
                ha="left", va="bottom", zorder=3)

    ax.plot(ug, p_warr, color=GREY, lw=1.3, ls=(0, (5, 2)), zorder=3)
    ax.plot(ug, p_fit, color=BLUE, lw=1.9, zorder=5)
    ax.scatter(u_meas, p_meas, s=6.5, facecolor="none", edgecolor=INK,
               linewidths=0.45, alpha=0.72, zorder=4)
    ax.axhline(P_RATED / 1e6, color=BLUE, lw=0.8, ls=(0, (2, 2)), alpha=0.7,
               zorder=2)
    ax.text(0.6, P_RATED / 1e6 + 0.08, f"rated {P_RATED / 1e6:.1f} MW",
            fontsize=6.8, color=BLUE, ha="left", va="bottom")

    ax_ct = ax.twinx()
    ax_ct.plot(ug, ct_fit, color=GOLD, lw=1.5, ls=(0, (1, 1.4)), zorder=6)
    ax_ct.set_ylim(0.0, 1.05)
    ax_ct.set_ylabel("Thrust coefficient $C_T$ (-)", fontsize=8.0, color=GOLD)
    ax_ct.tick_params(axis="y", labelsize=7.2, colors=GOLD)
    ax_ct.spines["right"].set_visible(True)
    ax_ct.spines["right"].set_color(GOLD)
    ax_ct.spines["top"].set_visible(False)

    ax.set_xlim(0.0, U_MAX)
    ax.set_ylim(0.0, 6.9)
    ax.set_xlabel("Hub-height wind speed $u$ (m/s)", fontsize=8.0)
    ax.set_ylabel("Electrical power $P$ (MW)", fontsize=8.0)
    ax.set_title("Power curve, operating regions and thrust coefficient",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(labelsize=7.2)
    ax.grid(True, axis="y", color=GRID, lw=0.55, alpha=0.8)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", ls="none", mfc="none", mec=INK, mew=0.45,
               ms=3.0, label=f"10 min bin means (n = {u_meas.size})"),
        Line2D([], [], color=BLUE, lw=1.9, label="Fitted curve, $C_P$ model"),
        Line2D([], [], color=GREY, lw=1.3, ls=(0, (5, 2)),
               label="Warranted curve"),
        Line2D([], [], color=GOLD, lw=1.5, ls=(0, (1, 1.4)),
               label="Thrust coefficient $C_T$ (right axis)"),
    ]
    ax.legend(handles=handles, loc="center right", frameon=False,
              fontsize=7.2, bbox_to_anchor=(0.995, 0.44))

    cp_max = float(cp_effective(ug).max())
    ax.text(0.985, 0.035,
            f"peak $C_P$ = {cp_max:.3f} (-)   "
            f"mean deviation from warranty = "
            f"{100.0 * (np.mean(p_meas) / np.mean(np.interp(u_meas, ug, p_warr)) - 1.0):+.1f} %",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color=INK, zorder=8,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.92))

    # ---- aux 1: Weibull speed distribution --------------------------------
    pdf = weibull_pdf(np.clip(ug, 1e-6, None))
    ax1.fill_between(ug, pdf, color=BLUE, alpha=0.20, lw=0.0, zorder=2)
    ax1.plot(ug, pdf, color=BLUE, lw=1.5, zorder=3)
    ax1.axvline(u_mean, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=4)
    ax1.text(u_mean + 0.4, 0.95, f"mean {u_mean:.2f} m/s",
             transform=ax1.get_xaxis_transform(), fontsize=6.5, color=INK,
             ha="left", va="top")
    for xv in (U_IN, u_rated, U_OUT):
        ax1.axvline(xv, color=GREY, lw=0.7, ls=(0, (3, 2)), alpha=0.8,
                    zorder=1)
    ax1.set_xlim(0.0, U_MAX)
    ax1.set_ylim(0.0, None)
    ax1.set_xlabel("Wind speed $u$ (m/s)", fontsize=7.2)
    ax1.set_ylabel("Frequency density (1/(m/s))", fontsize=7.2)
    ax1.set_title(f"Site distribution, Weibull $k$ = {WB_K:.2f}, "
                  f"$A$ = {WB_A:.2f} m/s", fontsize=8.5, fontweight="bold")
    ax1.tick_params(labelsize=6.5)
    ax1.grid(True, color=GRID, lw=0.5, alpha=0.8)
    ax1.set_axisbelow(True)

    # ---- aux 2: energy contribution per bin -------------------------------
    cols = [RED if i == i_peak else BLUE for i in range(mid.size)]
    ax2.bar(mid, e_bin, width=0.86, color=cols, edgecolor="white",
            linewidth=0.35, zorder=3)
    ax2.annotate(f"modal bin\n{edges[i_peak]:.0f}-{edges[i_peak + 1]:.0f} m/s\n"
                 f"{e_bin[i_peak]:.2f} GWh/a",
                 xy=(mid[i_peak], e_bin[i_peak]),
                 xytext=(mid[i_peak] + 3.2, e_bin[i_peak] * 0.80),
                 fontsize=6.5, color=RED, ha="left", va="top",
                 linespacing=1.3,
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=RED))
    for xv in (U_IN, u_rated, U_OUT):
        ax2.axvline(xv, color=GREY, lw=0.7, ls=(0, (3, 2)), alpha=0.8,
                    zorder=1)
    ax2.set_xlim(0.0, U_MAX)
    ax2.set_ylim(0.0, None)
    ax2.set_xlabel("Wind speed $u$ (m/s)", fontsize=7.2)
    ax2.set_ylabel("Energy yield (GWh/a)", fontsize=7.2)
    ax2.set_title(f"Annual yield by bin, total {aep:.1f} GWh/a",
                  fontsize=8.5, fontweight="bold")
    ax2.tick_params(labelsize=6.5)
    ax2.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.text(0.985, 0.94, f"capacity factor {cf:.3f} (-)",
             transform=ax2.transAxes, ha="right", va="top", fontsize=6.5,
             color=SUB)

    panel_labels_inches(fig, [ax, ax1, ax2], dx=-0.50, dy=0.10)


    savefig(fig, "715_wnd_turbine_power_curve.png")


if __name__ == "__main__":
    main()
