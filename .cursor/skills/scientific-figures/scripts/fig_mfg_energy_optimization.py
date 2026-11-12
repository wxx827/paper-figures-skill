"""Figure 683 - Plant energy against the production rhythm.

Main panel: hourly energy use stacked by equipment group, with the output
rate on the right axis and the three tariff windows laid in as neutral
background bands (a colour band here would fight the stack; the price is
printed in the band instead).

Right column at 58 %: (b) a waterfall from the baseline day to the optimised
day, (c) specific energy per equipment group before and after.

Every measure in (b) is an hourly delta array derived from the very profiles
drawn in (a), so the energy saving, the tariff-weighted cost saving and the
per-unit figures in (c) are three views of one calculation and cannot drift
apart (DESIGN_SYSTEM 13.2).
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, get_palette, lighten
from figkit import axes_inches, panel_label_inches

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"
BLUE, RED, GREEN = "#3b5b92", "#b23a48", "#2e7d5b"
GOLD, SOLID = "#c9a227", "#3a3a3a"

EQUIP = ["Furnace", "Press line", "Compressor", "Chiller", "Auxiliary"]
# hour ranges (inclusive start, exclusive end), tariff name, CNY per kWh
TARIFF = [(0, 8, "off-peak", 0.42), (8, 10, "shoulder", 0.78),
          (10, 12, "peak", 1.35), (12, 17, "shoulder", 0.78),
          (17, 21, "peak", 1.35), (21, 24, "off-peak", 0.42)]
BAND_FILL = {"off-peak": "#f5f5f5", "shoulder": "#ededed", "peak": "#dcdcdc"}


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(683)
    fig = plt.figure(figsize=(10.4, 6.4))

    h = np.arange(24) + 0.5
    price = np.zeros(24)
    for a, b, name, p in TARIFF:
        price[a:b] = p

    # two shifts of production, units per hour
    duty = (np.exp(-0.5 * ((h - 10.0) / 3.1) ** 2) +
            0.92 * np.exp(-0.5 * ((h - 19.0) / 2.7) ** 2))
    duty = duty / duty.max()
    output = np.round(46.0 * duty + 3.0 * rng.random(24), 1)

    # equipment profiles: standing load plus a production-coupled part
    standing = np.array([260.0, 55.0, 88.0, 120.0, 46.0])
    coupled = np.array([420.0, 380.0, 205.0, 150.0, 62.0])
    load = (standing[:, None] + coupled[:, None] * duty[None, :]) * \
        np.exp(rng.normal(0.0, 0.025, (len(EQUIP), 24)))     # kWh per hour

    base_kwh = float(load.sum())
    base_cost = float((load.sum(axis=0) * price).sum())
    peak_m = np.isin(np.arange(24), np.concatenate(
        [np.arange(a, b) for a, b, n, p in TARIFF if n == "peak"]))
    peak_share = 100.0 * float(load[:, peak_m].sum()) / base_kwh

    # ---- improvement measures, each an hourly delta on the same arrays ----
    idle = duty < 0.34
    deltas = {}
    d = np.zeros((len(EQUIP), 24))
    d[2] = 0.22 * load[2]
    deltas["Compressor VFD"] = d
    d = np.zeros((len(EQUIP), 24))
    d[1, idle] = 0.62 * load[1, idle]
    d[4, idle] = 0.30 * load[4, idle]
    deltas["Idle shutdown"] = d
    d = np.zeros((len(EQUIP), 24))
    d[3] = 0.14 * load[3]
    deltas["Chiller reset"] = d
    d = np.zeros((len(EQUIP), 24))
    d[0, peak_m] = 0.16 * load[0, peak_m]
    deltas["Furnace peak trim"] = d

    opt = load - sum(deltas.values())
    opt_kwh = float(opt.sum())
    opt_cost = float((opt.sum(axis=0) * price).sum())
    save_pct = 100.0 * (1.0 - opt_kwh / base_kwh)
    cost_pct = 100.0 * (1.0 - opt_cost / base_cost)

    # =====================================================================
    # (a) stacked energy against output
    # =====================================================================
    axm = axes_inches(fig, (0.88, 0.95, 4.92, 4.16))
    for a, b, name, p in TARIFF:
        axm.axvspan(a, b, facecolor=BAND_FILL[name], edgecolor="none",
                    zorder=0)
    for x, txt in ((3.6, "off-peak %.2f CNY/kWh" % 0.42),
                   (14.2, "shoulder %.2f" % 0.78),
                   (19.0, "peak %.2f" % 1.35)):
        axm.text(x, 0.992, txt, transform=axm.get_xaxis_transform(),
                 ha="center", va="top", fontsize=6.2, color=MUTED)

    ramp = get_palette("cb_sequential_blue", len(EQUIP) + 2)[1:-1][::-1]
    axm.stackplot(h, *load, colors=ramp, labels=EQUIP, edgecolor="white",
                  linewidth=0.4, zorder=2)
    axm.set_xlim(0.0, 24.0)
    axm.set_ylim(0.0, load.sum(axis=0).max() * 1.34)
    axm.set_xticks(np.arange(0, 25, 2))
    axm.set_xlabel("hour of day (h)", fontsize=8.0)
    axm.set_ylabel("energy use (kWh per hour)", fontsize=8.0)
    axm.tick_params(labelsize=7.0, length=2.6)
    axm.grid(axis="y", color=GRID, lw=0.55)
    axm.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=5,
               fontsize=6.8, frameon=False, handlelength=1.1,
               columnspacing=1.1, handletextpad=0.4)

    axo = axm.twinx()
    axo.plot(h, output, lw=1.7, color=GOLD, marker="o", ms=2.8, zorder=5)
    axo.set_ylim(0.0, output.max() * 1.34)
    axo.set_ylabel("production output (units/h)", fontsize=8.0, color=GOLD)
    axo.tick_params(labelsize=7.0, length=2.6, colors=GOLD)
    axo.spines["right"].set_color(GOLD)
    axo.spines["top"].set_visible(False)
    axo.annotate("output %.0f units/day" % output.sum(),
                 xy=(h[int(np.argmax(output))], output.max()),
                 xytext=(6.0, 4.0), textcoords="offset points", ha="left",
                 va="bottom", fontsize=6.6, color=GOLD)

    axm.text(0.4, 0.905, "baseline %.1f MWh/day, %.0f %% of it inside the "
             "peak windows" % (base_kwh / 1000.0, peak_share),
             transform=axm.get_xaxis_transform(), ha="left", va="top",
             fontsize=6.8, color=INK)
    for s in ("top",):
        axm.spines[s].set_visible(False)
    axm.set_title("Hourly energy by equipment group against the production "
                  "rhythm", fontsize=9.0, fontweight="bold", pad=17.0)

    # =====================================================================
    # (b) waterfall from baseline to optimised
    # =====================================================================
    axw = axes_inches(fig, (7.22, 3.44, 2.82, 1.76))
    # A saving waterfall rather than a total-energy one: at 9 % of a 30 MWh
    # day the four steps would be hairlines against the baseline bar, and
    # zooming the axis to rescue them would truncate that bar.  Building the
    # saving up from zero keeps every bar honest and readable.
    names = list(deltas)
    steps = np.array([float(deltas[k].sum()) for k in names])
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    labels = names + ["Total saving"]

    ypos = np.arange(len(labels))[::-1]
    for k, s in enumerate(steps):
        axw.barh(ypos[k], s, left=cum[k], height=0.52,
                 color=lighten(GREEN, 0.42), edgecolor=GREEN, linewidth=0.7,
                 zorder=3)
        axw.text(cum[k + 1] + cum[-1] * 0.018, ypos[k],
                 "%.0f  (%.1f %%)" % (s, 100.0 * s / base_kwh), ha="left",
                 va="center", fontsize=6.4, color=SUB)
        axw.plot([cum[k + 1], cum[k + 1]], [ypos[k] - 0.30,
                 ypos[k + 1] + 0.30], lw=0.6, color=HAIR, zorder=2)
    axw.barh(ypos[-1], cum[-1], height=0.62, color=GREEN, edgecolor="white",
             linewidth=0.7, zorder=3)
    axw.text(cum[-1] + cum[-1] * 0.018, ypos[-1],
             "%.0f  (%.1f %%)" % (cum[-1], save_pct), ha="left",
             va="center", fontsize=6.4, color=INK, fontweight="bold")
    axw.set_yticks(ypos)
    axw.set_yticklabels(labels, fontsize=6.5)
    axw.set_xlim(0.0, cum[-1] * 1.55)
    axw.set_xlabel("energy saved (kWh per day), baseline %.0f to %.0f"
                   % (base_kwh, opt_kwh), fontsize=7.4)
    axw.tick_params(labelsize=6.5, length=2.2)
    axw.grid(axis="x", color=GRID, lw=0.55)
    axw.set_title("Saving %.1f %% of energy and %.1f %% of cost "
                  "(%.0f CNY/day)" % (save_pct, cost_pct,
                                      base_cost - opt_cost),
                  fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axw.spines[s].set_visible(False)

    # =====================================================================
    # (c) specific energy per equipment group
    # =====================================================================
    axr = axes_inches(fig, (7.22, 0.95, 2.82, 1.76))
    units = float(output.sum())
    spec_b = load.sum(axis=1) / units
    spec_o = opt.sum(axis=1) / units
    order = np.argsort(spec_b)
    yp = np.arange(len(EQUIP))
    worst = int(order[-1])
    cols = [RED if order[k] == worst else lighten(BLUE, 0.35)
            for k in range(len(EQUIP))]
    axr.barh(yp, spec_b[order], height=0.58, color=cols, edgecolor="white",
             linewidth=0.7, zorder=3, label="baseline")
    axr.plot(spec_o[order], yp, ls="none", marker="|", ms=11.0, mew=2.0,
             color=GREEN, zorder=5, label="after optimisation")
    for k, i in enumerate(order):
        axr.text(spec_b[i] + spec_b.max() * 0.02, k, "%.2f" % spec_b[i],
                 ha="left", va="center", fontsize=6.4, color=SUB)
    axr.set_yticks(yp)
    axr.set_yticklabels([EQUIP[i] for i in order], fontsize=6.5)
    axr.set_xlim(0.0, spec_b.max() * 1.24)
    axr.set_xlabel("specific energy (kWh per unit)", fontsize=7.4)
    axr.tick_params(labelsize=6.5, length=2.2)
    axr.grid(axis="x", color=GRID, lw=0.55)
    axr.legend(loc="lower right", fontsize=6.5, frameon=False,
               handlelength=1.2, labelspacing=0.25)
    axr.set_title("%s dominates at %.2f kWh per unit"
                  % (EQUIP[worst], spec_b[worst]), fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)

    for ax, lab in ((axm, "(a)"), (axw, "(b)"), (axr, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.52, dy=0.14)
    savefig(fig, "683_mfg_energy_optimization.png")


if __name__ == "__main__":
    main()
