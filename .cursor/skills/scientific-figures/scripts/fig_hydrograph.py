"""Storm hydrograph with an inverted hyetograph and baseflow separation.

The classic hydrology layout: a rainfall hyetograph hanging from the top of the
canvas (reversed y axis) sits directly above the discharge hydrograph, both on a
shared time axis. Direct runoff and baseflow are separated by a dashed line and
filled with two tones, the flood peak is marked, and the basin lag between the
centroid of effective rainfall and the peak is dimensioned with a double arrow.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, style_axes

# ---- shared hydro / ocean palette (used across figures 213-222) -----------
C_DEEP = "#17384f"    # deep navy
C_MAIN = "#2c7a8c"    # teal
C_LIGHT = "#8fbdd0"   # pale blue
C_ACC = "#c0603f"     # rust accent
C_WARM = "#d9a441"    # ochre accent
C_GREY = "#8a8a8a"

AREA = 385.0          # catchment area, km^2


def panel_tag(fig, ax, text, dx=-0.055, dy=0.010, size=10.5):
    """Panel label placed in *figure* coordinates so all tags line up."""
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color="#1a1a1a")


def build_event():
    """Hourly rainfall and a 0.1 h discharge series for a single storm."""
    hours = np.arange(0, 120, dtype=float)
    rain = np.zeros_like(hours)
    rain[3:6] = [1.1, 2.4, 1.3]                       # antecedent shower
    burst = [0.9, 2.8, 6.1, 9.4, 12.6, 10.2, 6.4,
             3.6, 2.1, 3.9, 6.8, 4.5, 2.4, 1.2, 0.5]  # main storm
    rain[16:16 + len(burst)] = burst

    # effective rainfall: initial abstraction then a constant loss rate
    loss = np.minimum(rain, 1.6)
    cum = np.cumsum(rain)
    loss[cum < 6.0] = rain[cum < 6.0]
    eff = np.clip(rain - loss, 0.0, None)

    t = np.arange(0, 120, 0.1)
    # gamma unit hydrograph, peak near 8 h
    k, theta = 3.4, 3.1
    tau = np.arange(0, 60, 0.1)
    uh = tau ** (k - 1) * np.exp(-tau / theta)
    uh /= uh.sum() * 0.1
    # 1 mm of runoff over AREA km2 in one hour -> AREA*1e6/1000/3600 m3/s
    gain = AREA * 1e6 / 1000.0 / 3600.0
    direct = np.zeros_like(t)
    for h, e in zip(hours, eff):
        if e <= 0:
            continue
        i0 = int(round(h / 0.1))
        seg = min(len(uh), len(t) - i0)
        direct[i0:i0 + seg] += gain * e * uh[:seg]

    rise = 1.0 / (1.0 + np.exp(-(t - 30.0) / 7.0))
    base = 34.0 + 26.0 * rise * np.exp(-np.maximum(t - 46.0, 0.0) / 110.0)
    base -= 5.0 * (1.0 - np.exp(-t / 260.0))
    return hours, rain, eff, t, base, base + direct


def main():
    set_pub_style()
    hours, rain, eff, t, base, q = build_event()

    fig = plt.figure(figsize=(9.4, 5.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.55],
                          left=0.085, right=0.905, top=0.845, bottom=0.115,
                          hspace=0.07)
    ax_r = fig.add_subplot(gs[0])
    ax_q = fig.add_subplot(gs[1], sharex=ax_r)

    # ---- (a) inverted hyetograph -----------------------------------------
    ax_r.bar(hours + 0.5, rain, width=0.92, color=C_LIGHT,
             edgecolor=C_MAIN, linewidth=0.45, zorder=3,
             label="Total rainfall")
    ax_r.bar(hours + 0.5, eff, width=0.92, color=C_DEEP,
             edgecolor="none", zorder=4, label="Effective rainfall")
    ax_r.set_ylim(15.5, 0)
    ax_r.set_yticks([0, 5, 10, 15])
    ax_r.set_ylabel("Rainfall\n(mm h$^{-1}$)", linespacing=1.35)
    ax_r.grid(axis="y", color="#e8e8e8", lw=0.5)
    ax_r.set_axisbelow(True)
    ax_r.spines["bottom"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.spines["top"].set_visible(True)
    ax_r.spines["top"].set_color("#333333")
    ax_r.xaxis.set_ticks_position("top")
    ax_r.tick_params(axis="x", labelbottom=False, labeltop=False, length=2.2)
    ax_r.tick_params(axis="y", length=3)

    p_tot, p_eff = rain.sum(), eff.sum()
    ax_r.text(0.985, 0.06,
              f"Event rainfall  {p_tot:.0f} mm      "
              f"Effective rainfall  {p_eff:.0f} mm",
              transform=ax_r.transAxes, ha="right", va="bottom",
              fontsize=7.6, color="#444444")
    leg_r = ax_r.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02),
                        ncol=2, fontsize=7.6, handlelength=1.3,
                        handleheight=0.9, columnspacing=1.4)
    leg_r.set_clip_on(False)

    # ---- (b) discharge hydrograph ----------------------------------------
    ax_q.fill_between(t, 0, base, color=C_MAIN, alpha=0.55, lw=0, zorder=2,
                      label="Baseflow")
    ax_q.fill_between(t, base, q, color=C_MAIN, alpha=0.18, lw=0, zorder=2,
                      label="Direct runoff")
    ax_q.plot(t, base, color=C_DEEP, lw=1.0, ls=(0, (4.5, 2.2)), zorder=5,
              label="Baseflow separation")
    ax_q.plot(t, q, color=C_DEEP, lw=1.8, zorder=6, label="Observed discharge")

    ax_q.set_xlim(0, 120)
    ax_q.set_ylim(0, 1.32 * q.max())
    ax_q.set_xticks(np.arange(0, 121, 12))
    ax_q.set_xlabel("Time since start of storm (h)")
    ax_q.set_ylabel("Discharge (m$^{3}$ s$^{-1}$)")
    ax_q.grid(axis="y", color="#eeeeee", lw=0.5)
    ax_q.set_axisbelow(True)
    style_axes(ax_q)

    # peak marker + callout
    ip = int(np.argmax(q))
    tp, qp = t[ip], q[ip]
    ax_q.plot([tp], [qp], marker="o", ms=5.0, mfc="white", mec=C_ACC,
              mew=1.5, zorder=8)
    ax_q.annotate(f"Flood peak\n$\\bf{{{qp:.0f}}}$ m$^{3}$ s$^{{-1}}$"
                  f"  at  $\\bf{{{tp:.0f}}}$ h",
                  xy=(tp + 0.8, qp), xytext=(tp + 14.0, qp * 0.94),
                  fontsize=7.8, color=C_ACC, ha="left", va="center",
                  linespacing=1.45,
                  arrowprops=dict(arrowstyle="-", color=C_ACC, lw=0.8,
                                  connectionstyle="arc3,rad=-0.22"))

    # centroid of effective rainfall -> basin lag
    tc = float((hours[eff > 0] + 0.5) @ eff[eff > 0] / eff.sum())
    for ax, col in ((ax_r, C_DEEP), (ax_q, C_DEEP)):
        ax.axvline(tc, color=col, lw=0.7, ls=(0, (2.5, 2.0)), zorder=1,
                   alpha=0.75)
    ax_q.axvline(tp, color=C_ACC, lw=0.7, ls=(0, (2.5, 2.0)), zorder=1,
                 alpha=0.8)
    ax_r.axvline(tp, color=C_ACC, lw=0.7, ls=(0, (2.5, 2.0)), zorder=1,
                 alpha=0.8)

    ylag = 1.16 * qp
    ax_q.annotate("", xy=(tc, ylag), xytext=(tp, ylag),
                  arrowprops=dict(arrowstyle="<|-|>", color="#4a4a4a",
                                  lw=0.8, mutation_scale=7,
                                  shrinkA=0, shrinkB=0))
    ax_q.text(0.5 * (tc + tp), ylag * 1.030,
              f"basin lag  $t_L$ = {tp - tc:.1f} h", ha="center", va="bottom",
              fontsize=7.8, color="#3a3a3a")
    ax_q.text(tc - 4.2, 0.56 * qp, "centroid of effective rainfall",
              ha="center", va="center", rotation=90, fontsize=7.2,
              color=C_DEEP)

    # limb labels
    ax_q.text(tp - 2.4, 0.56 * qp, "rising limb", ha="center", va="center",
              fontsize=7.2, color="#3c6274", rotation=76)
    ax_q.text(tp + 21, 0.34 * qp, "recession limb", ha="left", va="center",
              fontsize=7.2, color="#3c6274", rotation=-20)
    ax_q.annotate("baseflow", xy=(96, 24), xytext=(96, 0.20 * qp),
                  ha="center", va="bottom", fontsize=7.2, color="#38657a",
                  arrowprops=dict(arrowstyle="-", color="#7aa8bb", lw=0.7))

    # event water balance box
    vol = np.trapezoid(np.clip(q - base, 0, None), t) * 3600.0
    depth = vol / (AREA * 1e6) * 1000.0
    txt = (f"Catchment area   {AREA:.0f} km$^2$\n"
           f"Runoff depth      {depth:.0f} mm\n"
           f"Runoff coeff.      {depth / p_tot:.2f}\n"
           f"Time to peak      {tp:.0f} h")
    ax_q.text(0.985, 0.955, txt, transform=ax_q.transAxes, ha="right",
              va="top", fontsize=7.4, color="#333333", linespacing=1.55,
              bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                        edgecolor="#d5d5d5", linewidth=0.6, alpha=0.95))

    handles, labels = ax_q.get_legend_handles_labels()
    order = [3, 2, 1, 0]
    leg = ax_q.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc="center right", bbox_to_anchor=(0.995, 0.50),
                      fontsize=7.8, handlelength=1.7, labelspacing=0.45)
    leg.set_clip_on(False)


    panel_tag(fig, ax_r, "(a)")
    panel_tag(fig, ax_q, "(b)")
    savefig(fig, "213_hydrograph.png")


if __name__ == "__main__":
    main()
