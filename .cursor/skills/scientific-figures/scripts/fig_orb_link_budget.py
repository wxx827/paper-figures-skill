"""X-band downlink budget, from transmitter terminal to link margin.

Every line of the waterfall is computed rather than tabulated: the two
antenna gains come from their diameters and the wavelength, the spreading
loss from the slant range at the working elevation, the gaseous and rain
terms from a specific attenuation along the same slant path, and the system
noise from the station noise temperature.  The running total is therefore
forced to end on the margin, and the side panels re-run the identical
function across elevation so the design point on each of them is the same
number the waterfall closes with.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
DEEP = "#22406e"
TEAL = "#2a7f7a"
GOLD = "#c9a227"
SLATE = "#4a4f5c"
PALE = "#dbe3ee"

C_LIGHT = 2.99792458e8     # m/s
R_E = 6378.14              # km
H_SAT = 705.0              # km
FREQ = 8.20e9              # Hz, X-band downlink
P_TX_W = 5.0               # W
LOSS_FEED = 1.4            # dB
D_TX = 0.30                # m
ETA_TX = 0.55
D_RX = 3.70                # m
ETA_RX = 0.60
T_SYS = 150.0              # K
LOSS_POINT = 0.5           # dB
LOSS_POL = 0.3             # dB
LOSS_IMPL = 1.5            # dB
EBN0_REQ = 6.5             # dB, 8PSK with the coding assumed
RATE_MBPS = 600.0
RAIN_DESIGN = 5.0          # mm/h at the design point
RAIN_HEAVY = 15.0          # mm/h for the stressed case
H_RAIN = 3.4               # km, effective rain height above the station
H_GAS = 4.2                # km, equivalent gaseous path height
GAMMA_GAS = 0.011          # dB/km at this frequency
RAIN_K, RAIN_A = 0.0188, 1.217
BW_MHZ = 375.0             # MHz of allocated bandwidth
SPEC_EFF = 3.0             # bit/s/Hz of the modulation and coding
MARGIN_KEEP = 3.0          # dB held back when sizing the rate
ELEV_DESIGN = 10.0         # deg

FIG_W, FIG_H = 11.0, 6.5
MAIN = (2.02, 1.02, 3.86, 4.48)
AUX_T = (7.62, 3.63, 2.60, 1.87)
AUX_B = (7.62, 1.02, 2.60, 1.80)


def wavelength():
    return C_LIGHT / FREQ


def dish_gain(diameter, eff):
    """Aperture gain (dBi) of a circular dish."""
    return 10.0 * np.log10(eff * (np.pi * diameter / wavelength()) ** 2)


def slant_range(elev_deg):
    """Range (km) to a circular orbit seen at a given elevation."""
    s = np.sin(np.radians(elev_deg))
    return np.sqrt((R_E * s) ** 2 + H_SAT ** 2 + 2.0 * R_E * H_SAT) - R_E * s


def path_loss(elev_deg):
    d_m = slant_range(elev_deg) * 1000.0
    return 20.0 * np.log10(4.0 * np.pi * d_m / wavelength())


def gas_loss(elev_deg):
    return GAMMA_GAS * H_GAS / np.sin(np.radians(elev_deg))


def rain_loss(elev_deg, rate_mm_h):
    if rate_mm_h <= 0.0:
        return 0.0 * np.asarray(elev_deg, dtype=float)
    gamma = RAIN_K * rate_mm_h ** RAIN_A
    return gamma * H_RAIN / np.sin(np.radians(elev_deg))


def g_over_t():
    return dish_gain(D_RX, ETA_RX) - 10.0 * np.log10(T_SYS)


def budget(elev_deg, rain_mm_h):
    """The waterfall as (label, delta dB, kind) plus the running total."""
    rows = [
        ("transmitter power, {:.0f} W".format(P_TX_W),
         10.0 * np.log10(P_TX_W), "gain"),
        ("feeder and filter loss", -LOSS_FEED, "loss"),
        ("spacecraft antenna, {:.2f} m".format(D_TX),
         dish_gain(D_TX, ETA_TX), "gain"),
        ("EIRP (dBW)", 0.0, "sub"),
        ("free-space loss at {:.0f} km".format(slant_range(elev_deg)),
         -path_loss(elev_deg), "loss"),
        ("gaseous absorption", -gas_loss(elev_deg), "loss"),
        ("rain at {:.0f} mm/h".format(rain_mm_h),
         -rain_loss(elev_deg, rain_mm_h), "loss"),
        ("antenna pointing loss", -LOSS_POINT, "loss"),
        ("polarisation mismatch", -LOSS_POL, "loss"),
        ("station G/T, {:.1f} m dish".format(D_RX), g_over_t(), "gain"),
        ("Boltzmann constant", 228.6, "gain"),
        ("carrier to noise density (dB-Hz)", 0.0, "sub"),
        ("data rate, {:.0f} Mbps".format(RATE_MBPS),
         -10.0 * np.log10(RATE_MBPS * 1e6), "loss"),
        ("energy per bit to noise (dB)", 0.0, "sub"),
        ("required Eb/N0", -EBN0_REQ, "loss"),
        ("implementation loss", -LOSS_IMPL, "loss"),
        ("link margin (dB)", 0.0, "sub"),
    ]
    total, out = 0.0, []
    for name, delta, kind in rows:
        if kind == "sub":
            out.append((name, total, total, kind))
        else:
            out.append((name, total, total + delta, kind))
            total += delta
    return out, total


def margin_at(elev_deg, rain_mm_h, rate_mbps=RATE_MBPS):
    """Link margin (dB) at one elevation, from the same terms as the table."""
    eirp = 10.0 * np.log10(P_TX_W) - LOSS_FEED + dish_gain(D_TX, ETA_TX)
    cn0 = (eirp - path_loss(elev_deg) - gas_loss(elev_deg)
           - rain_loss(elev_deg, rain_mm_h) - LOSS_POINT - LOSS_POL
           + g_over_t() + 228.6)
    return cn0 - 10.0 * np.log10(rate_mbps * 1e6) - EBN0_REQ - LOSS_IMPL


def rate_capacity(elev_deg, rain_mm_h):
    """Rate (Mbps) that still keeps the reserved margin, power limited."""
    eirp = 10.0 * np.log10(P_TX_W) - LOSS_FEED + dish_gain(D_TX, ETA_TX)
    cn0 = (eirp - path_loss(elev_deg) - gas_loss(elev_deg)
           - rain_loss(elev_deg, rain_mm_h) - LOSS_POINT - LOSS_POL
           + g_over_t() + 228.6)
    return 10.0 ** ((cn0 - EBN0_REQ - LOSS_IMPL - MARGIN_KEEP) / 10.0) / 1e6


def waterfall_panel(ax, rows, total):
    y = np.arange(len(rows))[::-1]
    colors = {"gain": BLUE, "loss": ORANGE, "sub": SLATE}
    for yi, (name, x0, x1, kind) in zip(y, rows):
        if kind == "sub":
            ax.plot([x0], [yi], marker="D", ms=5.4, mfc=GOLD, mec=INK,
                    mew=0.7, zorder=6)
            ax.plot([-190.0, x0], [yi, yi], color=SLATE, lw=0.6,
                    ls=(0, (2.4, 2.0)), alpha=0.7, zorder=3)
            ax.text(x0 + 6.0, yi, "{:+.1f}".format(x0), fontsize=6.4,
                    color=INK, fontweight="bold", ha="left", va="center",
                    zorder=7)
        else:
            ax.barh(yi, x1 - x0, left=x0, height=0.58,
                    color=colors[kind], edgecolor="white", linewidth=0.5,
                    zorder=5)
            side = 4.0 if x1 >= x0 else -4.0
            ax.text(max(x0, x1) + 4.0 if x1 >= x0 else min(x0, x1) - 4.0,
                    yi, "{:+.1f}".format(x1 - x0), fontsize=6.2,
                    color=colors[kind],
                    ha="left" if side > 0 else "right", va="center",
                    zorder=7)
    for yi in y[:-1]:
        ax.axhline(yi - 0.5, color="#e6e6e6", lw=0.4, zorder=1)
    ax.axvline(0.0, color="#9aa3ab", lw=0.8, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    for lab, row in zip(ax.get_yticklabels(), rows):
        if row[3] == "sub":
            lab.set_fontweight("bold")
            lab.set_color(INK)
        else:
            lab.set_color(SUB)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlim(-190.0, 130.0)
    ax.set_xlabel("running total (dB, reference given by each subtotal)")
    ax.tick_params(labelsize=6.6)
    ax.grid(axis="x", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("From {:.0f} W at the transmitter to {:.1f} dB of margin"
                 .format(P_TX_W, total), fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)
    ax.legend(handles=[Patch(facecolor=BLUE, edgecolor="none", label="gain"),
                       Patch(facecolor=ORANGE, edgecolor="none",
                             label="loss"),
                       Line2D([], [], color=GOLD, marker="D", ms=5.0, lw=0,
                              mec=INK, label="subtotal")],
              loc="lower right", fontsize=6.3, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.5, borderaxespad=0.4, labelspacing=0.26)


def margin_panel(ax, elev, design):
    clear = margin_at(elev, 0.0)
    light = margin_at(elev, RAIN_DESIGN)
    heavy = margin_at(elev, RAIN_HEAVY)
    ax.fill_between(elev, heavy, clear, color=PALE, alpha=0.7, zorder=2)
    ax.plot(elev, clear, color=BLUE, lw=1.5, zorder=6, label="clear sky")
    ax.plot(elev, light, color=TEAL, lw=1.3, ls=(0, (4.0, 2.0)), zorder=6,
            label="{:.0f} mm/h".format(RAIN_DESIGN))
    ax.plot(elev, heavy, color=RED, lw=1.3, zorder=6,
            label="{:.0f} mm/h".format(RAIN_HEAVY))
    ax.axhline(0.0, color=SLATE, lw=0.9, zorder=4)
    ax.axvline(ELEV_DESIGN, color=SUB, lw=0.8, ls=(0, (3.0, 2.0)), zorder=4)
    ax.plot([ELEV_DESIGN], [design], marker="o", ms=4.6, mfc=GOLD, mec=INK,
            mew=0.7, zorder=8)
    ax.annotate("design point\n{:.1f} dB at {:.0f} deg"
                .format(design, ELEV_DESIGN), xy=(ELEV_DESIGN, design),
                xytext=(24.0, design - 5.0), fontsize=6.1, color=INK,
                ha="left", va="center", linespacing=1.3, zorder=9,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(elev[0], 90.0)
    ax.set_ylim(min(heavy.min(), -2.0) - 1.5, clear.max() + 2.5)
    ax.set_xticks([5, 20, 40, 60, 80, 90])
    ax.set_xlabel("elevation angle (deg)", fontsize=6.8)
    ax.set_ylabel("link margin (dB)", fontsize=6.8)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, ncol=3, columnspacing=0.7,
              borderaxespad=0.25, labelspacing=0.2)
    ax.set_title("Margin against elevation and rain", fontsize=8.0,
                 fontweight="bold", color=INK)


def rate_panel(ax, elev):
    cap = BW_MHZ * SPEC_EFF
    clear = rate_capacity(elev, 0.0)
    heavy = rate_capacity(elev, RAIN_HEAVY)
    ax.plot(elev, clear, color=BLUE, lw=1.5, zorder=6, label="clear sky")
    ax.plot(elev, heavy, color=RED, lw=1.3, zorder=6,
            label="{:.0f} mm/h".format(RAIN_HEAVY))
    ax.axhline(cap, color=SLATE, lw=1.0, ls=(0, (4.0, 2.2)), zorder=5)
    ax.text(88.0, cap * 1.10, "{:.0f} MHz at {:.0f} bit/s/Hz caps the link "
            "at {:.0f} Mbps".format(BW_MHZ, SPEC_EFF, cap), fontsize=6.0,
            color=SLATE, ha="right", va="bottom")
    ax.axhline(RATE_MBPS, color=GOLD, lw=1.0, zorder=5)
    ax.text(88.0, RATE_MBPS * 0.60, "{:.0f} Mbps flown".format(RATE_MBPS),
            fontsize=6.0, color="#8a6d10", ha="right", va="top")
    cross = float(np.interp(RATE_MBPS, heavy, elev))
    ax.plot([cross], [RATE_MBPS], marker="o", ms=4.4, mfc=RED, mec="white",
            mew=0.8, zorder=8)
    ax.annotate("rain needs {:.0f} deg\nto hold the rate".format(cross),
                xy=(cross, RATE_MBPS), xytext=(cross + 6.0, RATE_MBPS * 0.30),
                fontsize=6.1, color=RED, ha="left", va="center",
                linespacing=1.3, zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_yscale("log")
    ax.set_xlim(elev[0], 90.0)
    ax.set_ylim(60.0, 4.0e4)
    ax.set_xticks([5, 20, 40, 60, 80, 90])
    ax.set_xlabel("elevation angle (deg)", fontsize=6.8)
    ax.set_ylabel("supportable rate (Mbps)", fontsize=6.8)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, ncol=2, columnspacing=0.7,
              borderaxespad=0.25, labelspacing=0.2)
    ax.set_title("What the power budget could carry", fontsize=8.0,
                 fontweight="bold", color=INK)


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
    rows, total = budget(ELEV_DESIGN, RAIN_DESIGN)
    design = margin_at(ELEV_DESIGN, RAIN_DESIGN)
    eirp = [r[1] for r in rows if r[0].startswith("EIRP")][0]
    cn0 = [r[1] for r in rows if r[0].startswith("carrier")][0]
    ebn0 = [r[1] for r in rows if r[0].startswith("energy")][0]

    steps = sum(r[2] - r[1] for r in rows)
    assert abs(steps - total) < 1e-9, "the waterfall must close on the total"
    assert abs(total - design) < 1e-9, "table and curve must agree"
    assert abs(cn0 - 10.0 * np.log10(RATE_MBPS * 1e6) - ebn0) < 1e-9
    assert margin_at(90.0, 0.0) > design, "zenith must beat the design point"

    elev = np.linspace(5.0, 90.0, 341)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_fall = axes_inches(fig, MAIN)
    ax_mar = axes_inches(fig, AUX_T)
    ax_rate = axes_inches(fig, AUX_B)

    waterfall_panel(ax_fall, rows, total)
    margin_panel(ax_mar, elev, design)
    rate_panel(ax_rate, elev)



    panel_labels_inches(fig, [ax_fall], labels=["(a)"], dx=-1.72, dy=0.14)
    panel_labels_inches(fig, [ax_mar, ax_rate], labels=["(b)", "(c)"],
                        dx=-0.62, dy=0.14)
    savefig(fig, "1023_orb_link_budget.png")


if __name__ == "__main__":
    main()
