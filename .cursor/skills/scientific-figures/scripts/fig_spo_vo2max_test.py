"""Incremental ramp test to exhaustion: gas exchange, ventilation and lactate.

One 25 W/min ramp is simulated breath by breath and then read the way a
laboratory would read it.  The first ventilatory threshold is not scripted: it
is found by the V-slope method, a two-segment least-squares fit of carbon
dioxide output against oxygen uptake whose breakpoint is chosen by scanning
every candidate index.  The second threshold is the first breath at which the
smoothed respiratory exchange ratio reaches 1.00, and VO2max is the highest
30 s average with an explicit plateau test against the expected ramp slope.
The three stacked panels share one time axis so every threshold line falls on
the same abscissa in all of them.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import get_cmap, savefig, set_pub_style
from figkit import panel_labels_inches, safe_colorbar, text_card

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
BLUE, GREEN, RED, GOLD = "#3b5b92", "#2e7d5b", "#b23a48", "#c9a227"
SOLID = "#3a3a3a"

MASS = 72.0            # kg
DT = 1.0 / 6.0         # 10 s averaging bins (min)
T_END = 12.0           # min to exhaustion
P0, RAMP = 50.0, 25.0  # W and W/min
VO2_CAP = 47.0         # mL/(kg min) where the linear rise starts to bend
VO2_ASYM = 55.5        # asymptotic ceiling of the same curve


def smooth3(y):
    """Three-bin running mean, the standard breath-averaging window."""
    k = np.ones(3) / 3.0
    return np.convolve(np.pad(y, 1, mode="edge"), k, mode="valid")


def two_line_break(x, y, lo=0.22, hi=0.86):
    """V-slope breakpoint: the split that minimises the two-segment SSE."""
    n = x.size
    best = (np.inf, n // 2)
    for i in range(int(lo * n), int(hi * n)):
        sse = 0.0
        for sl in (slice(0, i), slice(i, n)):
            xa, ya = x[sl], y[sl]
            if xa.size < 5:
                sse = np.inf
                break
            sse += float(np.sum((ya - np.polyval(
                np.polyfit(xa, ya, 1), xa)) ** 2))
        if sse < best[0]:
            best = (sse, i)
    i = best[1]
    p1 = np.polyfit(x[:i], y[:i], 1)
    p2 = np.polyfit(x[i:], y[i:], 1)
    xb = float((p2[1] - p1[1]) / (p1[0] - p2[0]))
    return i, p1, p2, xb


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    set_pub_style()
    rng = np.random.default_rng(664)

    t = np.arange(0.0, T_END + DT / 2.0, DT)
    power = P0 + RAMP * t

    lin = 6.4 + 0.1470 * power
    vo2 = np.where(lin < VO2_CAP, lin,
                   VO2_ASYM - (VO2_ASYM - VO2_CAP)
                   * np.exp(-(lin - VO2_CAP) / (VO2_ASYM - VO2_CAP)))
    vo2 = smooth3(vo2 + rng.normal(0.0, 0.85, t.size))

    v1_ref, v2_ref = 0.56 * VO2_ASYM, 0.82 * VO2_ASYM
    rer = (0.845 + 0.105 * sigmoid((vo2 - v1_ref) / 2.6)
           + 0.165 * sigmoid((vo2 - v2_ref) / 1.9))
    rer = smooth3(rer + rng.normal(0.0, 0.010, t.size))
    vco2 = smooth3(rer * vo2)

    # ---- thresholds, all measured from the traces above --------------------
    i_v1, p1, p2, vo2_vt1 = two_line_break(vo2, vco2)
    t_vt1 = float(np.interp(vo2_vt1, vo2, t))
    above = np.where(rer >= 1.00)[0]
    i_v2 = int(above[0]) if above.size else t.size - 1
    t_vt2 = float(t[i_v2])
    vo2_vt2 = float(vo2[i_v2])

    tail = vo2[-3:]
    vo2max = float(tail.mean())
    d_obs = float(vo2[-1] - vo2[-7])          # last 60 s
    d_exp = 0.1470 * RAMP * 1.0
    plateau = d_obs < 0.5 * d_exp

    ve = smooth3((vco2 * MASS / 1000.0)
                 * (25.5 + 9.5 * sigmoid((t - t_vt2) / 0.55))
                 + rng.normal(0.0, 1.4, t.size))
    hr = smooth3(68.0 + 0.400 * (power - P0)
                 - 6.0 * sigmoid((t - 10.2) / 0.8)
                 + rng.normal(0.0, 1.6, t.size))
    hr_max = float(hr.max())

    t_la = np.arange(0.0, T_END + 0.1, 2.0)
    lactate = (0.90 + 0.21 * np.exp(0.3044 * t_la)
               + rng.normal(0.0, 0.10, t_la.size))
    la_vt1 = float(np.interp(t_vt1, t_la, lactate))
    la_vt2 = float(np.interp(t_vt2, t_la, lactate))
    p_vt1, p_vt2 = P0 + RAMP * t_vt1, P0 + RAMP * t_vt2
    p_peak = float(power[-1])

    # ---- layout ------------------------------------------------------------
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.60],
                          height_ratios=[1.0, 0.88, 0.88],
                          left=0.062, right=0.945, top=0.800, bottom=0.150,
                          wspace=0.40, hspace=0.24)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax_s = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1:, 1])

    def thresholds(ax):
        ax.axvline(t_vt1, color=GREEN, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
        ax.axvline(t_vt2, color=RED, lw=0.9, ls=(0, (4, 2.5)), zorder=2)

    # ---- (a) gas exchange --------------------------------------------------
    ax1.axvspan(t_vt2, t[-1], color=RED, alpha=0.055, zorder=0)
    ax1.axvspan(t_vt1, t_vt2, color=GOLD, alpha=0.055, zorder=0)
    ax1.axvspan(t[0], t_vt1, color=GREEN, alpha=0.055, zorder=0)
    ax1.plot(t, vo2, color=BLUE, lw=1.5, zorder=4)
    ax1.plot(t, vco2, color=GOLD, lw=1.5, zorder=4)
    ax1.axhline(vo2max, color=RED, lw=0.8, ls=":", zorder=3)
    ax1.plot([t[-2]], [vo2max], marker="o", ms=4.6, color=RED, mec="white",
             mew=0.7, zorder=6)
    ax1.annotate(f"$VO_{{2max}}$ {vo2max:.1f} mL/(kg$\\cdot$min)"
                 + ("\nplateau confirmed" if plateau else "\nno plateau"),
                 xy=(t[-2], vo2max), xytext=(t_vt1 + 0.25, vo2max + 5.0),
                 fontsize=6.4, color=RED, ha="left", va="bottom",
                 linespacing=1.3,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=RED,
                                 shrinkA=2, shrinkB=4))
    thresholds(ax1)
    ax1.set_ylabel("gas exchange\n(mL/(kg$\\cdot$min))", fontsize=7.6,
                   linespacing=1.3)
    ax1.set_ylim(0.0, max(vco2.max(), vo2.max()) * 1.34)
    ax1.tick_params(labelsize=6.8, labelbottom=False)
    ax1.grid(axis="y", lw=0.5, color="#f2f2f2", zorder=1)
    ax1.set_title("One ramp to exhaustion, read three ways", fontsize=9.0,
                  fontweight="bold", pad=16)
    ax1.legend(handles=[Line2D([0], [0], color=BLUE, lw=1.5,
                               label="$\\dot{V}O_2$"),
                        Line2D([0], [0], color=GOLD, lw=1.5,
                               label="$\\dot{V}CO_2$")],
               loc="upper left", frameon=False, fontsize=7.0, ncol=2,
               handlelength=1.6, borderaxespad=0.2)
    sec = ax1.secondary_xaxis(
        "top", functions=(lambda x: P0 + RAMP * x,
                          lambda p: (p - P0) / RAMP))
    sec.set_xlabel("external power (W)", fontsize=7.4, labelpad=1.5)
    sec.tick_params(labelsize=6.6)

    # ---- (b) ventilation and RER ------------------------------------------
    ax2.plot(t, ve, color=BLUE, lw=1.5, zorder=4)
    ax2.set_ylabel("$\\dot{V}_E$ (L/min)", fontsize=7.6, color=BLUE)
    ax2.tick_params(axis="y", labelsize=6.8, colors=BLUE)
    ax2.set_ylim(0.0, ve.max() * 1.30)
    ax2.tick_params(axis="x", labelsize=6.8, labelbottom=False)
    ax2.grid(axis="y", lw=0.5, color="#f2f2f2", zorder=1)
    ax2r = ax2.twinx()
    ax2r.plot(t, rer, color=GOLD, lw=1.4, zorder=4)
    ax2r.axhline(1.0, color=RULE, lw=0.7, ls=":", zorder=3)
    ax2r.set_ylabel("RER (-)", fontsize=7.6, color=GOLD)
    ax2r.tick_params(axis="y", labelsize=6.8, colors=GOLD)
    ax2r.set_ylim(0.72, 1.30)
    ax2r.spines["right"].set_visible(True)
    ax2r.spines["right"].set_color(GOLD)
    ax2r.spines["top"].set_visible(False)
    thresholds(ax2)
    for x, name, col, va in ((t_vt1, "VT1", GREEN, "bottom"),
                             (t_vt2, "VT2", RED, "bottom")):
        ax2.text(x - 0.10, 0.03, name, transform=ax2.get_xaxis_transform(),
                 ha="right", va=va, fontsize=6.6, fontweight="bold",
                 color=col, zorder=7)

    # ---- (c) heart rate and lactate ---------------------------------------
    ax3.plot(t, hr, color=BLUE, lw=1.5, zorder=4)
    ax3.set_ylabel("heart rate (bpm)", fontsize=7.6, color=BLUE)
    ax3.tick_params(axis="y", labelsize=6.8, colors=BLUE)
    ax3.set_ylim(50.0, hr.max() * 1.18)
    ax3.tick_params(axis="x", labelsize=6.8)
    ax3.grid(axis="y", lw=0.5, color="#f2f2f2", zorder=1)
    ax3.set_xlabel("time into the 25 W/min ramp (min)", fontsize=8.0)
    ax3.set_xlim(t[0], t[-1])
    ax3r = ax3.twinx()
    ax3r.plot(t_la, lactate, color=GOLD, lw=1.3, marker="o", ms=3.6,
              mfc="white", mec=GOLD, mew=0.9, zorder=5)
    ax3r.set_ylabel("blood lactate (mmol/L)", fontsize=7.6, color=GOLD)
    ax3r.tick_params(axis="y", labelsize=6.8, colors=GOLD)
    ax3r.set_ylim(0.0, lactate.max() * 1.28)
    ax3r.spines["right"].set_visible(True)
    ax3r.spines["right"].set_color(GOLD)
    ax3r.spines["top"].set_visible(False)
    thresholds(ax3)
    for x, la, col in ((t_vt1, la_vt1, GREEN), (t_vt2, la_vt2, RED)):
        ax3r.plot([x], [la], marker="D", ms=4.0, color=col, mec="white",
                  mew=0.6, zorder=7)
        ax3r.text(x + 0.12, la + 0.35, f"{la:.1f} mmol/L", ha="left",
                  va="bottom", fontsize=6.2, color=col, zorder=7)

    # ---- results card ------------------------------------------------------
    ax_s.axis("off")
    text_card(ax_s, 0.02, 0.06, 0.96, 0.86,
              f"$VO_{{2max}}$      {vo2max:.1f} mL/(kg$\\cdot$min)\n"
              f"peak power   {p_peak:.0f} W\n"
              f"HR peak      {hr_max:.0f} bpm\n"
              f"VT1          {vo2_vt1:.1f} mL/(kg$\\cdot$min), "
              f"{p_vt1:.0f} W\n"
              f"VT2          {vo2_vt2:.1f} mL/(kg$\\cdot$min), "
              f"{p_vt2:.0f} W\n"
              f"VT1 / VT2    {100 * vo2_vt1 / vo2max:.0f}% / "
              f"{100 * vo2_vt2 / vo2max:.0f}% of $VO_{{2max}}$",
              fc="#f5f5f5", ec="#cccccc", fs=6.6, color=SUB, ha="left",
              pad=0.05, title="Test summary", title_fs=7.6,
              title_color=INK, linespacing=1.65)

    # ---- (d) V-slope -------------------------------------------------------
    sc = ax_v.scatter(vo2, vco2, s=13, c=t, cmap=get_cmap("cb_sequential_blue"),
                      edgecolors="white", linewidths=0.35, zorder=4)
    xs1 = np.linspace(vo2.min(), vo2_vt1 + 3.0, 40)
    xs2 = np.linspace(vo2_vt1 - 3.0, vo2.max(), 40)
    ax_v.plot(xs1, np.polyval(p1, xs1), color=MUTED, lw=1.2, zorder=5)
    ax_v.plot(xs2, np.polyval(p2, xs2), color=RED, lw=1.2, zorder=5)
    ax_v.plot([vo2_vt1], [np.polyval(p1, vo2_vt1)], marker="o", ms=6.0,
              color=GREEN, mec="white", mew=1.0, zorder=7)
    ax_v.annotate(f"VT1  {vo2_vt1:.1f} mL/(kg$\\cdot$min)\n"
                  f"S1 = {p1[0]:.2f},  S2 = {p2[0]:.2f}",
                  xy=(vo2_vt1, np.polyval(p1, vo2_vt1)),
                  xytext=(vo2.min() + 1.0, vco2.max() * 0.92),
                  fontsize=6.4, color=GREEN, ha="left", va="top",
                  linespacing=1.35,
                  arrowprops=dict(arrowstyle="->", lw=0.8, color=GREEN,
                                  shrinkA=2, shrinkB=6))
    ax_v.set_xlabel("$\\dot{V}O_2$ (mL/(kg$\\cdot$min))", fontsize=7.6)
    ax_v.set_ylabel("$\\dot{V}CO_2$ (mL/(kg$\\cdot$min))", fontsize=7.6)
    ax_v.tick_params(labelsize=6.8)
    ax_v.grid(lw=0.5, color="#f4f4f4", zorder=0)
    ax_v.set_title("V-slope: the breakpoint is fitted,\nnot chosen",
                   fontsize=8.5, fontweight="bold", pad=5, linespacing=1.35)
    safe_colorbar(fig, sc, ax_v, label="time into ramp (min)", width=0.10,
                  pad=0.07, labelsize=6.3)

    panel_labels_inches(fig, [ax1, ax2, ax3, ax_v], dx=-0.46, dy=0.10)

    print(f"VO2max {vo2max:.2f} mL/(kg min), plateau={plateau} "
          f"(observed {d_obs:.2f} vs expected {d_exp:.2f} per 60 s)")
    print(f"VT1 {vo2_vt1:.2f} at {t_vt1:.2f} min / {p_vt1:.0f} W  "
          f"({100 * vo2_vt1 / vo2max:.1f}% VO2max), S1={p1[0]:.3f} "
          f"S2={p2[0]:.3f}")
    print(f"VT2 {vo2_vt2:.2f} at {t_vt2:.2f} min / {p_vt2:.0f} W  "
          f"({100 * vo2_vt2 / vo2max:.1f}% VO2max)")
    print(f"HR peak {hr_max:.0f} bpm, peak VE {ve.max():.0f} L/min, "
          f"peak lactate {lactate.max():.1f} mmol/L")
    savefig(fig, "664_spo_vo2max_test.png")


if __name__ == "__main__":
    main()
