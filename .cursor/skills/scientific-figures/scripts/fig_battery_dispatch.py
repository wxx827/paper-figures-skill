"""Price-arbitrage dispatch of a 100 MW / 400 MWh battery over one day.

Three stacked panels share the hour axis: (a) charge / discharge power with
charging drawn below zero, (b) state of charge against the operating band,
(c) the day-ahead price that drives the schedule.  The two buy windows and
the two sell windows are shaded straight through all three panels, so a
reader can trace "cheap hours -> tank fills -> expensive hours -> tank
empties" down a single column.

The state of charge is integrated from the power schedule with a one-way
efficiency, so the SoC trace and the power bars cannot disagree.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes

CHG = "#4a7f9e"          # charging (hydro blue)
DIS = "#cf8447"          # discharging (gas orange)
SOC_C = "#8478a6"        # state of charge (nuclear purple)
INK = "#333333"
GREY = "#8d959c"
BAND_C = "#e9f0f5"
BAND_D = "#f9efe4"

P_MAX, E_MAX = 100.0, 400.0       # MW, MWh
ETA = 0.922                       # one-way, ~0.85 round trip
SOC_LO, SOC_HI = 10.0, 95.0       # %
E0 = 80.0                         # MWh at 00:00

CHARGE_W = [(0.0, 5.0), (11.0, 15.0)]
DISCHARGE_W = [(7.0, 9.0), (17.5, 21.5)]


def win(t, a, b, r=0.30):
    """Trapezoidal window with ``r``-hour ramps."""
    return np.clip(np.minimum((t - a) / r, (b - t) / r), 0.0, 1.0)


def circ_gauss(t, mu, sigma):
    d = ((t - mu + 12.0) % 24.0) - 12.0
    return np.exp(-0.5 * (d / sigma) ** 2)


def build():
    t = np.linspace(0.0, 24.0, 97)
    rng = np.random.default_rng(31)

    price = (22.0 + 52.0 * circ_gauss(t, 8.0, 2.1)
             + 128.0 * circ_gauss(t, 19.3, 2.4)
             - 34.0 * circ_gauss(t, 13.0, 2.6))
    price += 3.0 * np.sin(2 * np.pi * t / 7.3) + rng.normal(0, 1.6, t.size)

    power = (-40.0 * win(t, *CHARGE_W[0]) - 55.0 * win(t, *CHARGE_W[1])
             + 40.0 * win(t, *DISCHARGE_W[0])
             + 70.0 * win(t, *DISCHARGE_W[1]))

    dt = t[1] - t[0]
    de = np.where(power < 0, -power * ETA, -power / ETA) * dt
    energy = E0 + np.concatenate([[0.0], np.cumsum(de[:-1])])
    return t, price, power, energy


def main():
    set_pub_style(base_font=8)
    t, price, power, energy = build()
    soc = 100.0 * energy / E_MAX
    dt = t[1] - t[0]

    chg = np.clip(-power, 0, None)
    dis = np.clip(power, 0, None)
    e_in, e_out = chg.sum() * dt, dis.sum() * dt
    buy = (chg * price).sum() * dt / max(e_in, 1e-9)
    sell = (dis * price).sum() * dt / max(e_out, 1e-9)
    margin = (dis * price).sum() * dt - (chg * price).sum() * dt

    fig = plt.figure(figsize=(7.2, 5.35))
    gs = fig.add_gridspec(3, 1, left=0.088, right=0.982, top=0.925,
                          bottom=0.135, hspace=0.16,
                          height_ratios=[1.0, 0.9, 0.9])
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=axa)
    axc = fig.add_subplot(gs[2], sharex=axa)
    axes = (axa, axb, axc)

    # arbitrage windows run through every panel
    for a in axes:
        for w in CHARGE_W:
            a.axvspan(*w, color=BAND_C, lw=0, zorder=0)
        for w in DISCHARGE_W:
            a.axvspan(*w, color=BAND_D, lw=0, zorder=0)

    # ---- (a) power ---------------------------------------------------------
    axa.bar(t, power, width=dt * 0.92, lw=0, zorder=3,
            color=np.where(power < 0, CHG, DIS))
    axa.axhline(0, color=INK, lw=0.8, zorder=4)
    axa.set_ylim(-72, 90)
    axa.set_yticks([-50, 0, 50])
    axa.set_ylabel("Power (MW)")
    # sign convention pinned to the right margin, clear of the band labels
    axa.text(0.994, 0.93, "discharge  +", transform=axa.transAxes,
             fontsize=6.3, color=DIS, ha="right", va="top")
    axa.text(0.994, 0.045, "charge  \u2212", transform=axa.transAxes,
             fontsize=6.3, color=CHG, ha="right", va="bottom")
    for w, lab in zip(CHARGE_W + DISCHARGE_W,
                      ["BUY", "BUY", "SELL", "SELL"]):
        axa.text(sum(w) / 2, 84.0, lab, fontsize=6.1, color=GREY,
                 ha="center", va="top", fontweight="bold")
    axa.set_title("Day-ahead arbitrage schedule of a 100 MW / 400 MWh "
                  "battery", fontsize=8.8, fontweight="bold", loc="left",
                  x=0.0)

    # ---- (b) state of charge ----------------------------------------------
    axb.axhspan(SOC_LO, SOC_HI, color="#f2f4f2", lw=0, zorder=1)
    for lvl, lab in ((SOC_HI, f"upper limit {SOC_HI:.0f}%"),
                     (SOC_LO, f"lower limit {SOC_LO:.0f}%")):
        axb.axhline(lvl, color="#a4544f", lw=0.8, ls=(0, (4, 2.2)), zorder=3)
        axb.text(23.85, lvl + (2.5 if lvl < 50 else -2.5), lab, fontsize=6.2,
                 color="#a4544f", ha="right",
                 va="bottom" if lvl < 50 else "top", zorder=5)
    axb.plot(t, soc, color=SOC_C, lw=1.8, zorder=4)
    axb.fill_between(t, 0, soc, color=SOC_C, alpha=0.10, lw=0, zorder=2)
    k = int(np.argmax(soc))
    axb.plot([t[k]], [soc[k]], marker="o", ms=3.6, color=SOC_C, mec="white",
             mew=0.7, zorder=6, linestyle="none")
    # short enough to sit entirely inside the SoC plateau, so the rising and
    # falling limbs never cut through it
    axb.text(16.2, 85.5, f"peak SoC {soc[k]:.0f}%", fontsize=6.3,
             color=SOC_C, ha="center", va="top", zorder=6)
    axb.text(0.30, 92.0,
             f"{P_MAX:.0f} MW / {E_MAX:.0f} MWh  \u00b7  "
             f"{100 * ETA ** 2:.0f}% round trip\n"
             f"Throughput {e_out:.0f} MWh/day  \u00b7  "
             f"{e_out / E_MAX:.2f} cycles\n"
             f"Gross margin ${margin / 1000:.1f}k/day",
             fontsize=6.4, color="#5d6a75", ha="left", va="top",
             linespacing=1.5, zorder=6)
    axb.set_ylim(0, 108)
    axb.set_yticks([0, 25, 50, 75, 100])
    axb.set_ylabel("State of charge (%)")

    # ---- (c) price ---------------------------------------------------------
    axc.fill_between(t, 0, price, color="#eceff1", lw=0, zorder=1)
    axc.plot(t, price, color="#4a5158", lw=1.5, zorder=4)
    for w, lvl, col, lab in ((CHARGE_W, buy, CHG, "mean buy"),
                             (DISCHARGE_W, sell, DIS, "mean sell")):
        for a, b in w:
            axc.plot([a, b], [lvl, lvl], color=col, lw=1.5, zorder=5,
                     solid_capstyle="butt")
        # label sits after the *first* window of each kind, where the price
        # trace is far away, instead of running off the right-hand edge
        axc.text(w[0][1] + 0.25, lvl, f"{lab} {lvl:.0f} $/MWh",
                 fontsize=6.3, color=col, ha="left", va="center", zorder=6)
    axc.text(0.006, 0.96, f"arbitrage spread {sell - buy:.0f} $/MWh",
             transform=axc.transAxes, fontsize=6.5, color=INK, ha="left",
             va="top", fontweight="bold", zorder=6)
    axc.set_ylim(-18, max(price) * 1.16)
    axc.set_yticks([0, 50, 100, 150])
    axc.set_ylabel("Price ($/MWh)")
    axc.set_xlabel("Hour of day")

    for i, a in enumerate(axes):
        a.set_xlim(0, 24)
        a.set_xticks(np.arange(0, 25, 3))
        a.grid(axis="y", lw=0.4, alpha=0.45)
        style_axes(a)
        if i < 2:
            a.tick_params(labelbottom=False)
    axc.set_xticklabels([f"{h:02d}" for h in range(0, 25, 3)])

    fw, fh = fig.get_size_inches()
    for a, lab in zip(axes, ["(a)", "(b)", "(c)"]):
        bb = a.get_position()
        a.text(-0.48 / (bb.width * fw), 1.0 + 0.045 / (bb.height * fh), lab,
               transform=a.transAxes, ha="left", va="bottom",
               fontweight="bold", fontsize=10)

    handles = [
        Patch(facecolor=CHG, label="Charging"),
        Patch(facecolor=DIS, label="Discharging"),
        Line2D([], [], color=SOC_C, lw=1.8, label="State of charge"),
        Line2D([], [], color="#4a5158", lw=1.5, label="Day-ahead price"),
        Patch(facecolor=BAND_C, label="Buy window"),
        Patch(facecolor=BAND_D, label="Sell window"),
        Line2D([], [], color="#a4544f", lw=0.8, ls=(0, (4, 2.2)),
               label="SoC limits"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.53, -0.012), handlelength=1.7,
               columnspacing=1.8, labelspacing=0.45)

    print(f"SoC range {soc.min():.1f}%\u2013{soc.max():.1f}%  "
          f"start {soc[0]:.1f}%  end {soc[-1]:.1f}%")
    print(f"charged {e_in:.0f} MWh @ {buy:.1f} $/MWh   "
          f"discharged {e_out:.0f} MWh @ {sell:.1f} $/MWh   "
          f"margin ${margin:,.0f}")
    savefig(fig, "197_battery_dispatch.png")


if __name__ == "__main__":
    main()
