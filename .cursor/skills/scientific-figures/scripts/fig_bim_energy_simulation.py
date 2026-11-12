"""Whole-building energy simulation: 8760 hours, monthly end uses, EUI card.

The main panel is the raw simulation output folded into a day-by-hour matrix,
which is the only view in which the three schedules a designer actually tunes
are visible at once: the occupied band across the working day, the weekly
stripe of unoccupied weekends, and the summer cooling season swelling out of
the shoulder months.  The peak hour is located by argmax on the same series
that fills the matrix.  The two smaller panels carry the accounting: monthly
consumption split by end use, and the headline intensity metrics, all
integrated from the hourly series rather than quoted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_card, panel_labels_inches, safe_colorbar, use_inch_units

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"

END_USES = ["HVAC", "Lighting", "Equipment", "Hot water"]
USE_COLOR = {"HVAC": "#3b5b92", "Lighting": "#c9a227",
             "Equipment": "#2e7d5b", "Hot water": "#8e6bbf"}
RAMP = get_cmap("cb_sequential_warm")

FLOOR_AREA = 4200.0          # m2, conditioned
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DAY_PROFILE = np.array([
    0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.12, 0.35, 0.75, 0.95, 0.95, 0.95,
    0.72, 0.92, 0.95, 0.95, 0.90, 0.68, 0.38, 0.20, 0.10, 0.07, 0.05, 0.05])


def simulate(seed=17):
    """Hourly electric load by end use for one non-leap year, in kW."""
    rng = np.random.default_rng(seed)
    hours = np.arange(8760)
    doy, hod = hours // 24, hours % 24
    weekend = ((doy + 3) % 7) >= 5                       # Jan 1 is a Thursday

    occ = DAY_PROFILE[hod] * np.where(weekend, 0.16, 1.0)
    t_out = (14.5 - 12.0 * np.cos(2 * np.pi * (doy - 18) / 365.0)
             + 5.0 * np.sin(2 * np.pi * (hod - 9) / 24.0)
             + rng.normal(0.0, 1.1, hours.size))

    cooling = 6.5 * np.clip(t_out - 22.0, 0.0, None)
    heating = 3.8 * np.clip(16.0 - t_out, 0.0, None)
    hvac = (cooling + heating) * (0.30 + 0.70 * occ) + 5.5 * occ
    light = 26.0 * occ + 1.5
    equip = 34.0 * occ + 4.0
    dhw = 7.0 * occ * (1.0 + 0.9 * np.exp(-0.5 * ((hod - 8.0) / 1.6) ** 2))

    loads = {"HVAC": hvac, "Lighting": light, "Equipment": equip,
             "Hot water": dhw}
    jitter = 1.0 + rng.normal(0.0, 0.035, hours.size)
    return {k: np.clip(v * jitter, 0.0, None) for k, v in loads.items()}


def heat_panel(ax, total):
    grid = total.reshape(365, 24).T                       # hour x day
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=RAMP,
                   interpolation="nearest", extent=[0.0, 365.0, 0.0, 24.0])

    idx = int(np.argmax(total))
    day, hour = idx // 24, idx % 24
    month = int(np.searchsorted(np.cumsum(MONTH_DAYS), day + 1))
    dom = day + 1 - int(np.cumsum(MONTH_DAYS)[month - 1]) if month else day + 1
    ax.plot([day + 0.5], [hour + 0.5], marker="o", ms=7.0, mfc="none",
            mec=ACCENT, mew=1.3, zorder=5)
    ax.annotate(f"peak {total[idx]:.0f} kW\n{MONTHS[month]} {dom:.0f}, "
                f"{hour:02d}:00", xy=(day + 0.5, hour + 0.5),
                xytext=(day - 58.0, 21.6), ha="center", va="center",
                fontsize=6.8, color=ACCENT, zorder=6,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))

    starts = np.concatenate([[0], np.cumsum(MONTH_DAYS)])
    ax.set_xticks(starts[:-1] + MONTH_DAYS / 2.0)
    ax.set_xticklabels(MONTHS, fontsize=6.8)
    ax.set_xticks(starts, minor=True)
    ax.tick_params(axis="x", which="minor", length=2.2, color="#777777")
    ax.set_yticks(np.arange(0, 25, 4))
    ax.set_ylabel("hour of day (h)")
    ax.set_xlabel("day of year (-)")
    ax.set_xlim(0.0, 365.0)
    ax.set_title("Hourly electric load across the simulated year",
                 fontsize=9.0, fontweight="bold", color=INK)
    return im, idx


def monthly_panel(ax, loads):
    bounds = np.concatenate([[0], np.cumsum(MONTH_DAYS)]) * 24
    x = np.arange(12)
    bottom = np.zeros(12)
    for use in END_USES:
        mwh = np.array([loads[use][bounds[m]:bounds[m + 1]].sum() / 1000.0
                        for m in range(12)])
        ax.bar(x, mwh, bottom=bottom, width=0.74, color=USE_COLOR[use],
               edgecolor="white", linewidth=0.5, label=use)
        bottom += mwh
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in MONTHS], fontsize=6.5)
    ax.set_xlim(-0.7, 11.7)
    ax.set_ylabel("monthly energy (MWh)", fontsize=7.4)
    ax.set_xlabel("month (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), handlelength=1.2,
              columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0)
    ax.set_title("Monthly consumption by end use", fontsize=8.2,
                 fontweight="bold", color=INK)
    return bottom


def metric_panel(ax, cards):
    w, h = use_inch_units(ax)
    cw, ch = (w - 0.16) / 2.0, (h - 0.16) / 2.0
    for i, (value, unit, label, color) in enumerate(cards):
        cx = 0.04 + cw / 2.0 + (i % 2) * (cw + 0.08)
        cy = h - 0.04 - ch / 2.0 - (i // 2) * (ch + 0.08)
        panel_card(ax, cx, cy, cw, ch, fc="#f5f5f5", ec="#cccccc", lw=0.7,
                   rounding=0.05)
        ax.text(cx, cy + 0.10, value, ha="center", va="center", fontsize=12.0,
                color=color)
        ax.text(cx, cy - 0.10, unit, ha="center", va="center", fontsize=6.4,
                color=MUTED)
        ax.text(cx, cy - 0.24, label, ha="center", va="center", fontsize=6.8,
                color=SUB)


def main():
    set_pub_style()
    loads = simulate()
    total = sum(loads.values())
    annual = {k: v.sum() for k, v in loads.items()}
    annual_kwh = sum(annual.values())
    eui = annual_kwh / FLOOR_AREA
    peak = float(total.max())
    load_factor = total.mean() / peak
    hvac_share = annual["HVAC"] / annual_kwh

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.62], wspace=0.34, hspace=0.62)
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_month = fig.add_subplot(gs[0, 1])
    ax_card = fig.add_subplot(gs[1, 1])

    im, idx = heat_panel(ax_heat, total)
    monthly_panel(ax_month, loads)
    ax_card.set_title("Headline intensity metrics", fontsize=8.2,
                      fontweight="bold", color=INK)

    fig.tight_layout(rect=[0, 0.040, 1, 0.945], h_pad=2.6, w_pad=2.6)

    metric_panel(ax_card, [
        (f"{eui:.0f}", "kWh/m$^2$$\\cdot$a", "energy use intensity", INK),
        (f"{peak:.0f}", "kW", "peak electric load", ACCENT),
        (f"{load_factor * 100:.0f}", "%", "annual load factor", INK),
        (f"{hvac_share * 100:.0f}", "%", "HVAC share of energy",
         USE_COLOR["HVAC"])])
    safe_colorbar(fig, im, ax_heat, label="hourly load (kW)", width=0.11,
                  pad=0.09, labelsize=6.6)
    panel_labels_inches(fig, [ax_heat, ax_month, ax_card], dx=-0.40, dy=0.12)
    savefig(fig, "589_bim_energy_simulation.png")


if __name__ == "__main__":
    main()
