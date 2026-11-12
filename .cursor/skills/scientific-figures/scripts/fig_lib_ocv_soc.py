"""Open-circuit voltage of an NMC811/graphite cell against state of charge.

(a) Charge and discharge OCV after a 4 h rest at each SOC, with the
    hysteresis band filled between the two curves.  Every millivolt quoted
    in the panel is the vertical gap of those two arrays.
(b) The same pair as dV/dSOC, the quantity a battery-management system
    uses to invert voltage into SOC.
(c) Mean hysteresis against SOC, with the 10-90 % window mean marked.

Illustrative synthetic OCV tables, not a specific commercial cell.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
INK = "#1a1a1a"
MUTED = "#777777"
SEED = 1213


def ocv_base(soc):
    """Smooth NMC811-like OCV, volts, SOC in 0-1."""
    x = np.clip(soc, 0.0, 1.0)
    return (3.42
            + 0.62 * x
            + 0.18 * np.tanh((x - 0.22) / 0.055)
            + 0.11 * np.tanh((x - 0.58) / 0.07)
            + 0.22 * np.tanh((x - 0.92) / 0.045))


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(SEED)
    soc = np.linspace(0.02, 0.98, 49)
    hyst = 0.012 + 0.018 * np.exp(-((soc - 0.35) / 0.28) ** 2)
    v_ch = ocv_base(soc) + 0.5 * hyst + rng.normal(0, 0.0012, soc.size)
    v_dc = ocv_base(soc) - 0.5 * hyst + rng.normal(0, 0.0012, soc.size)
    v_mean = 0.5 * (v_ch + v_dc)
    gap = v_ch - v_dc
    mid = (soc > 0.10) & (soc < 0.90)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.02, 5.35, 4.62))
    axb = axes_inches(fig, (7.30, 3.74, 2.90, 1.90))
    axc = axes_inches(fig, (7.30, 1.02, 2.90, 1.90))

    ax.fill_between(100 * soc, v_dc, v_ch, color=GOLD, alpha=0.22, lw=0,
                    zorder=1, label="hysteresis")
    ax.plot(100 * soc, v_ch, color=RED, lw=1.7, zorder=4, label="charge rest")
    ax.plot(100 * soc, v_dc, color=BLUE, lw=1.7, zorder=4,
            label="discharge rest")
    ax.plot(100 * soc, v_mean, color=INK, lw=0.9, ls=(0, (3.2, 1.8)),
            zorder=3, label="mean OCV")
    ax.set_xlabel("State of charge (%)")
    ax.set_ylabel("Open-circuit voltage (V)")
    ax.set_xlim(0, 100)
    ax.set_ylim(3.15, 4.45)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)
    ax.tick_params(labelsize=7.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.97, 0.06,
            "peak gap = {:.0f} mV\n10-90 % mean = {:.0f} mV".format(
                1e3 * gap.max(), 1e3 * gap[mid].mean()),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdcdc",
                      lw=0.6))

    dch = np.gradient(v_ch, soc)
    ddc = np.gradient(v_dc, soc)
    axb.plot(100 * soc, dch, color=RED, lw=1.4)
    axb.plot(100 * soc, ddc, color=BLUE, lw=1.4)
    axb.set_xlabel("State of charge (%)", fontsize=8, labelpad=1.5)
    axb.set_ylabel("d$V$/dSOC (V)", fontsize=8, labelpad=1.5)
    axb.set_xlim(0, 100)
    axb.tick_params(labelsize=6.5)
    axb.grid(color="#ededed", lw=0.5, zorder=0)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    axb.set_title("Local slope", fontsize=8.5, fontweight="bold", pad=4)

    axc.plot(100 * soc, 1e3 * gap, color=GOLD, lw=1.6)
    axc.axhline(1e3 * gap[mid].mean(), color=MUTED, lw=0.8,
                ls=(0, (2.6, 1.8)))
    axc.set_xlabel("State of charge (%)", fontsize=8, labelpad=1.5)
    axc.set_ylabel("Hysteresis (mV)", fontsize=8, labelpad=1.5)
    axc.set_xlim(0, 100)
    axc.set_ylim(0, 1e3 * gap.max() * 1.25)
    axc.tick_params(labelsize=6.5)
    axc.grid(color="#ededed", lw=0.5, zorder=0)
    for side in ("top", "right"):
        axc.spines[side].set_visible(False)
    axc.set_title("Charge minus discharge", fontsize=8.5, fontweight="bold",
                  pad=4)

    panel_labels_inches(fig, [ax, axb, axc], dx=-0.34, dy=0.10)
    savefig(fig, "1213_lib_ocv_soc.png")


if __name__ == "__main__":
    main()
