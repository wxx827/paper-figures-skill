"""Hourly generation stack with the net-load "duck curve".

(a) stacked dispatch of six technologies over one day, with the served load
drawn on top of the stack and the net load (load minus wind and solar) shown
as a dashed curve; the evening solar-withdrawal ramp is shaded and measured.
(b) the same day rolled up into daily energy.

Layout note: panel (b) is a companion statistic, not a second main result,
so it is given ~28 % of the canvas width and sits on a pale card.  The stack
is built as a residual (gas balances the system), which guarantees the top of
the stack coincides exactly with the load curve.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
from scipy.ndimage import gaussian_filter1d

from pubstyle import set_pub_style, savefig, style_axes

FUEL = {
    "Nuclear": "#8478a6",
    "Coal":    "#5b5f64",
    "Hydro":   "#4a7f9e",
    "Wind":    "#4f9e91",
    "Solar":   "#dcb35c",
    "Gas":     "#cf8447",
}
ORDER = ["Nuclear", "Coal", "Hydro", "Wind", "Solar", "Gas"]
INK = "#333333"
GREY = "#8d959c"
LOAD_C = "#1f2a33"


def circ_gauss(t, mu, sigma):
    d = ((t - mu + 12.0) % 24.0) - 12.0
    return np.exp(-0.5 * (d / sigma) ** 2)


def build_day(seed=5):
    """15-minute dispatch of a 7.4 GW peak system, in MW."""
    t = np.linspace(0.0, 24.0, 97)
    rng = np.random.default_rng(seed)

    load = (4300.0 + 900.0 * circ_gauss(t, 9.0, 2.4)
            + 2600.0 * circ_gauss(t, 19.0, 2.9)
            + 1200.0 * circ_gauss(t, 14.5, 3.2))
    load *= 1.0 + 0.008 * gaussian_filter1d(rng.normal(0, 1, t.size), 3.0)
    frac = (load - load.min()) / np.ptp(load)

    clear = np.clip(np.sin(np.pi * (t - 6.0) / 12.6), 0.0, None) ** 1.25
    cloud = np.clip(1.0 - 0.22 * np.abs(
        gaussian_filter1d(rng.normal(0, 1, t.size), 2.5)), 0.6, 1.0)
    gen = {
        "Nuclear": np.full_like(t, 1200.0),
        "Coal": 900.0 + 700.0 * gaussian_filter1d(frac, 4.0),
        "Hydro": 400.0 + 500.0 * frac,
        # wind blows harder overnight and slackens through the afternoon
        "Wind": 900.0 + 320.0 * np.cos(2 * np.pi * (t - 3.0) / 24.0)
        + 110.0 * gaussian_filter1d(rng.normal(0, 1, t.size), 5.0),
        "Solar": 1750.0 * clear * cloud,
    }
    gen["Gas"] = load - sum(gen.values())          # gas balances the system
    return t, load, gen


def main():
    set_pub_style(base_font=8)
    t, load, gen = build_day()
    net = load - gen["Wind"] - gen["Solar"]

    # steepest three-hour net-load ramp
    k3 = 12                                        # 3 h at 15-min resolution
    ramp = net[k3:] - net[:-k3]
    j = int(np.argmax(ramp))
    r0, r1, rmw = t[j], t[j + k3], ramp[j]

    fig = plt.figure(figsize=(7.4, 4.15))
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, top=0.895,
                          bottom=0.185, wspace=0.055,
                          width_ratios=[1.0, 0.285])
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ---- (a) dispatch stack ----------------------------------------------
    ax.axvspan(r0, r1, color="#f4ece0", lw=0, zorder=0)
    ax.stackplot(t, [gen[k] for k in ORDER],
                 colors=[FUEL[k] for k in ORDER], lw=0.0, zorder=2,
                 alpha=0.94)
    ax.plot(t, load, color=LOAD_C, lw=1.7, zorder=5)
    ax.plot(t, net, color=LOAD_C, lw=1.4, ls=(0, (4.5, 2.2)), zorder=5)

    # both call-outs live in the white space above the load curve, never on
    # top of a filled band
    ib = int(np.argmin(net))
    ax.plot([t[ib]], [net[ib]], marker="o", ms=3.6, color=LOAD_C,
            mec="white", mew=0.7, zorder=6, linestyle="none")
    ax.annotate(f"net-load belly\n{net[ib]:,.0f} MW at {t[ib]:.0f}:00",
                xy=(t[ib], net[ib]), xytext=(0.9, 7150),
                fontsize=6.4, color=LOAD_C, ha="left", va="top",
                linespacing=1.35, zorder=7,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=LOAD_C,
                                connectionstyle="arc3,rad=0.20"))
    ax.annotate(f"solar withdrawal: {rmw:,.0f} MW\nnet-load ramp in 3 h "
                f"({r0:.0f}:00\u2013{r1:.0f}:00)",
                xy=((r0 + r1) / 2, 7480), xytext=((r0 + r1) / 2, 7660),
                fontsize=6.4, color="#9c5b3c", ha="center", va="bottom",
                linespacing=1.35, zorder=7,
                arrowprops=dict(arrowstyle="-[, widthB=1.75, lengthB=0.30",
                                lw=0.7, color="#9c5b3c"))

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 8600)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 3)])
    ax.set_yticks(np.arange(0, 8001, 2000))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Generation / load (MW)")
    ax.set_title("Daily dispatch stack and net load", fontsize=8.8,
                 fontweight="bold", loc="left", x=0.0)
    ax.grid(axis="y", lw=0.4, alpha=0.35, color="white")
    style_axes(ax)

    # ---- (b) daily energy roll-up (subordinate panel) ---------------------
    axb.add_patch(FancyBboxPatch(
        (-0.045, -0.055), 1.10, 1.115, transform=axb.transAxes,
        boxstyle="round,pad=0.0,rounding_size=0.035", facecolor="#f6f6f4",
        edgecolor="#e0e0dc", lw=0.7, zorder=0, clip_on=False))

    dt = t[1] - t[0]
    energy = {k: np.trapezoid(gen[k], dx=dt) / 1000.0 for k in ORDER}
    tot = sum(energy.values())
    keys = sorted(ORDER, key=lambda k: energy[k], reverse=True)
    ypos = np.arange(len(keys))[::-1]
    axb.barh(ypos, [energy[k] for k in keys], height=0.6,
             color=[FUEL[k] for k in keys], lw=0, zorder=3)
    for y, k in zip(ypos, keys):
        axb.text(energy[k] + tot * 0.012, y,
                 f"{100 * energy[k] / tot:.0f}%", va="center", ha="left",
                 fontsize=6.4, color=INK, zorder=4)
    axb.set_yticks(ypos)
    axb.set_yticklabels(keys, fontsize=7)
    axb.set_ylim(-1.75, 5.55)
    axb.set_xlim(0, max(energy.values()) * 1.30)
    axb.set_xticks([0, 10, 20, 30])
    axb.set_xlabel("Daily energy (GWh)", fontsize=7.2)
    axb.set_title("Energy mix", fontsize=8.0, fontweight="bold", loc="left",
                  x=0.0)
    vre = 100.0 * (energy["Wind"] + energy["Solar"]) / tot
    ren = vre + 100.0 * energy["Hydro"] / tot
    axb.text(0.0, -1.00, f"Total {tot:.0f} GWh\nVariable renewables "
             f"{vre:.0f}%\nAll renewables {ren:.0f}%", fontsize=6.4,
             color="#5d6a75", ha="left", va="center", linespacing=1.5,
             zorder=4)
    axb.grid(axis="x", lw=0.4, alpha=0.5)
    axb.tick_params(axis="y", length=0)
    axb.tick_params(axis="x", labelsize=6.6)
    style_axes(axb)

    fw, fh = fig.get_size_inches()
    for a, lab in ((ax, "(a)"), (axb, "(b)")):
        bb = a.get_position()
        a.text(-0.40 / (bb.width * fw), 1.0 + 0.075 / (bb.height * fh), lab,
               transform=a.transAxes, ha="left", va="bottom",
               fontweight="bold", fontsize=10)

    handles = [Patch(facecolor=FUEL[k], label=k) for k in ORDER]
    handles += [
        Line2D([], [], color=LOAD_C, lw=1.7, label="Served load"),
        Line2D([], [], color=LOAD_C, lw=1.4, ls=(0, (4.5, 2.2)),
               label="Net load (load \u2212 wind \u2212 solar)"),
        Patch(facecolor="#f4ece0", label="Evening ramp window"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.015), handlelength=1.6,
               columnspacing=1.5, labelspacing=0.45)

    print(f"peak load={load.max():,.0f} MW  min={load.min():,.0f} MW")
    print(f"net-load belly={net.min():,.0f} MW  max 3-h ramp={rmw:,.0f} MW "
          f"({r0:.2f}h -> {r1:.2f}h)")
    print(f"gas min={gen['Gas'].min():,.0f} MW  max={gen['Gas'].max():,.0f} MW"
          f"  daily energy={tot:.1f} GWh")
    savefig(fig, "195_renewable_stack.png")


if __name__ == "__main__":
    main()
