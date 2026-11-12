"""Levelised cost of electricity: composition today, trajectory since 2010.

(a) each technology's median LCOE broken into capital, fixed O&M, fuel and
carbon, with the P10-P90 range drawn as a whisker under the bar, sorted from
cheapest to most expensive median.
(b) the 2010-2024 learning curves on a log axis, where a constant percentage
decline plots as a straight line.

Colour discipline: panel (a) encodes *cost components* on a single slate ramp
plus one red accent for carbon, panel (b) encodes *technologies* with the
shared fuel palette.  The two encodings therefore never share a hue, so a
reader is never asked to hold two meanings for one colour.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes

# cost components: one slate ramp, carbon called out in red
COMP = [("Capital", "#3a6c8c"), ("Fixed O&M", "#7ba5bd"),
        ("Fuel", "#bcd2de"), ("Carbon", "#b23a48")]
INK = "#333333"
GREY = "#8d959c"

# technology -> (capital, O&M, fuel, carbon, P10, P90)
TECH = {
    "Solar PV (utility)": (28, 6, 0, 0, 24, 52),
    "Onshore wind":       (30, 10, 0, 0, 28, 58),
    "Hydro":              (44, 8, 0, 0, 34, 88),
    "Gas CCGT":           (16, 5, 38, 17, 52, 124),
    "Offshore wind":      (58, 18, 0, 0, 58, 110),
    "Coal":               (26, 9, 26, 42, 74, 142),
    "Nuclear":            (78, 22, 8, 0, 82, 168),
}

# technology -> (colour, 2010 LCOE, 2024 LCOE, curvature)
# curvature < 1 front-loads the decline, which is how the learning curves
# actually ran: steep early progress that flattens out
HIST = {
    "Solar PV":      ("#dcb35c", 380.0, 45.0, 0.72),
    "Offshore wind": ("#2f7a6f", 195.0, 78.0, 0.86),
    "Onshore wind":  ("#4f9e91", 105.0, 38.0, 0.80),
    "Gas CCGT":      ("#cf8447", 82.0, 76.0, 1.00),
    "Coal":          ("#5b5f64", 88.0, 103.0, 1.00),
}
YEARS = np.arange(2010, 2025)
LAB_SEP = 0.065          # minimum label spacing, in decades


def main():
    set_pub_style(base_font=8)
    order = sorted(TECH, key=lambda k: sum(TECH[k][:4]))
    ypos = np.arange(len(order))[::-1]

    fig = plt.figure(figsize=(7.6, 4.35))
    gs = fig.add_gridspec(1, 2, left=0.135, right=0.975, top=0.885,
                          bottom=0.195, wspace=0.30,
                          width_ratios=[1.0, 0.62])
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ---- (a) cost stack + range whisker ----------------------------------
    for y, name in zip(ypos, order):
        cap, om, fuel, carb, lo, hi = TECH[name]
        left = 0.0
        for val, (_, col) in zip((cap, om, fuel, carb), COMP):
            if val > 0:
                axa.barh(y + 0.14, val, left=left, height=0.44, color=col,
                         edgecolor="white", lw=0.6, zorder=3)
                left += val
        axa.text(left + 3.0, y + 0.14, f"{left:.0f}", fontsize=6.8,
                 color=INK, va="center", ha="left", zorder=4)
        yw = y - 0.26
        axa.plot([lo, hi], [yw, yw], color="#5d6a75", lw=0.9, zorder=4,
                 solid_capstyle="butt")
        for xc in (lo, hi):
            axa.plot([xc, xc], [yw - 0.10, yw + 0.10], color="#5d6a75",
                     lw=0.9, zorder=4)
        axa.plot([left], [yw], marker="D", ms=3.0, color="#5d6a75",
                 mec="white", mew=0.6, zorder=5, linestyle="none")

    axa.set_yticks(ypos)
    axa.set_yticklabels(order, fontsize=7.4)
    axa.set_ylim(-0.75, len(order) - 0.35)
    axa.set_xlim(0, 182)
    axa.set_xticks([0, 40, 80, 120, 160])
    axa.set_xlabel("Levelised cost of electricity ($/MWh)")
    axa.set_title("Cost composition, 2024", fontsize=8.8, fontweight="bold",
                  loc="left", x=0.0)
    axa.tick_params(axis="y", length=0)
    axa.grid(axis="x", lw=0.4, alpha=0.5)
    style_axes(axa)

    # component key parked in the empty top-right of the bar field
    handles = [Patch(facecolor=c, label=n) for n, c in COMP]
    handles.append(Line2D([], [], color="#5d6a75", lw=0.9, marker="D",
                          ms=3.0, mec="white", mew=0.6,
                          label="P10\u2013P90 range"))
    leg = axa.legend(handles=handles, loc="upper right",
                     bbox_to_anchor=(1.005, 1.02), frameon=True,
                     facecolor="white", edgecolor="#dcdcdc", framealpha=0.95,
                     fontsize=6.8, handlelength=1.5, labelspacing=0.42,
                     borderpad=0.5)
    leg.get_frame().set_linewidth(0.6)
    leg.set_clip_on(False)

    # ---- (b) learning curves ---------------------------------------------
    tt = (YEARS - YEARS[0]) / (YEARS[-1] - YEARS[0])
    for name, (col, y0, y1, k) in HIST.items():
        w = tt ** k
        y = np.exp(np.log(y0) * (1 - w) + np.log(y1) * w)
        axb.plot(YEARS, y, color=col, lw=1.6, zorder=4)
        axb.plot([YEARS[-1]], [y1], marker="o", ms=3.4, color=col,
                 mec="white", mew=0.7, zorder=5, linestyle="none")

    # offshore wind and gas CCGT end two dollars apart, so the end labels are
    # pushed apart in log space and tied back with hairline leaders
    prev = None
    for name, (col, y0, y1, _) in sorted(HIST.items(),
                                         key=lambda kv: -kv[1][2]):
        ylab = y1
        if prev is not None and np.log10(prev / ylab) < LAB_SEP:
            ylab = prev / 10 ** LAB_SEP
        prev = ylab
        if abs(ylab - y1) > 1e-9:
            axb.plot([YEARS[-1], YEARS[-1] + 0.45], [y1, ylab], color=col,
                     lw=0.6, zorder=5)
        axb.text(YEARS[-1] + 0.55, ylab,
                 f"{name}  {100 * (y1 - y0) / y0:+.0f}%", fontsize=6.6,
                 color=col, ha="left", va="center", zorder=6)

    axb.set_yscale("log")
    axb.set_xlim(2010, 2032.5)
    axb.set_ylim(27, 470)
    axb.set_yticks([40, 60, 100, 200, 400])
    axb.set_yticklabels(["40", "60", "100", "200", "400"])
    axb.set_yticks([], minor=True)
    axb.set_xticks([2010, 2014, 2018, 2022])
    axb.set_xlabel("Year")
    axb.set_ylabel("LCOE ($/MWh, log scale)")
    axb.set_title("Learning curves, 2010\u20132024", fontsize=8.8,
                  fontweight="bold", loc="left", x=0.0)
    axb.grid(axis="y", lw=0.4, alpha=0.5)
    style_axes(axb)
    axb.text(0.012, 0.035, "straight line = constant % decline per year",
             transform=axb.transAxes, fontsize=6.3, color=GREY, ha="left",
             va="bottom")

    fw, fh = fig.get_size_inches()
    for a, lab in ((axa, "(a)"), (axb, "(b)")):
        bb = a.get_position()
        a.text(-0.42 / (bb.width * fw), 1.0 + 0.055 / (bb.height * fh), lab,
               transform=a.transAxes, ha="left", va="bottom",
               fontweight="bold", fontsize=10)


    for name in order:
        print(f"  {name:20s} median={sum(TECH[name][:4]):3.0f} "
              f"range={TECH[name][4]:3.0f}\u2013{TECH[name][5]:3.0f} $/MWh")
    savefig(fig, "200_lcoe_compare.png")


if __name__ == "__main__":
    main()
