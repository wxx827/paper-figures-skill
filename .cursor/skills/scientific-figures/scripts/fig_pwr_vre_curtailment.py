"""Curtailment as the part of the resource that will not fit under the line.

A week of hourly wind and solar is generated once and then used everywhere.
The absorbable headroom is the load minus the must-run floor, so the hours
where the available renewable output rises above that line are curtailed by
definition, and the shaded wedge is the difference between the two arrays.
The diurnal panel is the same week folded onto 24 hours - it is where the
negative correlation between the two resources shows up - and the sweep
re-runs the whole accounting for larger fleets, once as it stands and once
with a 500 MW / 2000 MWh battery absorbing the surplus greedily.
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
TEAL = "#2a7f7a"
GREEN = "#6bbf8a"
PURPLE = "#7b5ea7"
SLATE = "#4a4f5c"

HOURS = 168
WIND_CAP = 2400.0        # MW installed
SOLAR_CAP = 1900.0       # MW installed
MUSTRUN = 1500.0         # MW of inflexible must-run generation
BAT_P, BAT_E = 1000.0, 4000.0     # MW and MWh
ETA = 0.90

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)


def week(rng):
    """One week of hourly load, wind and solar, all in MW."""
    t = np.arange(HOURS)
    hod = t % 24
    dow = t // 24
    weekend = np.isin(dow, [5, 6]).astype(float)

    load = (3900.0
            + 900.0 * np.exp(-0.5 * ((hod - 9.5) / 2.6) ** 2)
            + 1250.0 * np.exp(-0.5 * ((hod - 19.5) / 2.2) ** 2)
            - 620.0 * np.exp(-0.5 * ((hod - 4.0) / 3.2) ** 2)
            - 520.0 * weekend
            + rng.normal(0.0, 45.0, HOURS))

    solar = SOLAR_CAP * np.clip(np.sin(np.pi * (hod - 6.3) / 12.2),
                                0.0, None) ** 1.35
    cloud = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * (t + 30) / 61.0))
    solar = solar * np.clip(cloud + rng.normal(0.0, 0.05, HOURS), 0.0, 1.0)

    slow = (0.52 + 0.30 * np.sin(2 * np.pi * (t - 18) / 76.0)
            + 0.12 * np.sin(2 * np.pi * (t + 5) / 29.0))
    diurnal = 0.10 * np.cos(2 * np.pi * (hod - 3.0) / 24.0)   # windier at night
    wind = WIND_CAP * np.clip(slow + diurnal + rng.normal(0.0, 0.035, HOURS),
                              0.02, 0.97)
    return load, wind, solar


def account(load, wind, solar, scale=1.0, battery=False):
    """Curtailment hour by hour; returns MWh totals and the hourly arrays."""
    avail = scale * (wind + solar)
    head = np.maximum(load - MUSTRUN, 0.0)
    soc = 0.0
    used = np.zeros(HOURS)
    stored = np.zeros(HOURS)
    curt = np.zeros(HOURS)
    for k in range(HOURS):
        u = min(avail[k], head[k])
        surplus = avail[k] - u
        if battery and surplus > 0.0:
            take = min(surplus, BAT_P, (BAT_E - soc) / ETA)
            soc += take * ETA
            stored[k] = take
            surplus -= take
        elif battery and soc > 0.0:
            give = min(head[k] - u, BAT_P, soc)
            soc -= give
            u += give
        used[k] = u
        curt[k] = surplus
    return dict(avail=avail, used=used, stored=stored, curt=curt, head=head)


def main_panel(ax, day, load, wind, solar, acc):
    ax.fill_between(day, 0.0, wind, facecolor=TEAL, alpha=0.90,
                    edgecolor="white", linewidth=0.3, zorder=3)
    ax.fill_between(day, wind, wind + solar, facecolor=GREEN, alpha=0.92,
                    edgecolor="white", linewidth=0.3, zorder=3)
    ax.plot(day, acc["head"], color=ORANGE, lw=1.6, zorder=6)
    ax.plot(day, load, color=SLATE, lw=0.9, alpha=0.75, zorder=5)

    ax.fill_between(day, acc["head"], acc["avail"],
                    where=acc["avail"] > acc["head"], facecolor="none",
                    hatch="///", edgecolor=RED, linewidth=0.0, zorder=7)
    ax.plot(day, np.where(acc["avail"] > acc["head"], acc["avail"], np.nan),
            color=RED, lw=0.9, zorder=8)

    worst = int(np.argmax(acc["curt"]))
    ax.annotate("largest spill {:.0f} MW".format(acc["curt"][worst]),
                xy=(day[worst], 0.5 * (acc["head"][worst]
                                       + acc["avail"][worst])),
                xytext=(0.08, 5230.0), fontsize=6.7, color=RED,
                fontweight="bold", ha="left", va="top", zorder=10,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, shrinkA=2,
                                shrinkB=3))
    ax.text(0.08, 5820.0, "the {:.0f} MW must-run floor is what\npushes the "
            "orange line below the load".format(MUSTRUN), ha="left",
            va="top", fontsize=6.5, color=SUB, linespacing=1.32, zorder=9)

    handles = [Patch(facecolor=TEAL, alpha=0.90, label="wind, available"),
               Patch(facecolor=GREEN, alpha=0.92, label="solar, available"),
               Line2D([], [], color=ORANGE, lw=1.6,
                      label="headroom (load minus must-run)"),
               Line2D([], [], color=SLATE, lw=0.9, alpha=0.75,
                      label="system load"),
               Patch(facecolor="white", edgecolor=RED, hatch="///",
                     label="curtailed energy")]
    ax.legend(handles=handles, loc="upper right", fontsize=6.3, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.7, ncol=2, columnspacing=1.0,
              borderaxespad=0.35, labelspacing=0.28)

    ax.set_xlim(0.0, 7.0)
    ax.set_ylim(0.0, 5900.0)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""])
    ax.set_xlabel("day of week (168 hours at hourly resolution)")
    ax.set_ylabel("power (MW)")
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("One week of available wind and solar against what the "
                 "system can take", fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def diurnal_panel(ax, wind, solar, r):
    hod = np.arange(HOURS) % 24
    h = np.arange(24)
    mw = np.array([wind[hod == k].mean() for k in h])
    ms = np.array([solar[hod == k].mean() for k in h])
    sd = np.array([(wind + solar)[hod == k].std() for k in h])

    ax.fill_between(h, mw + ms - sd, mw + ms + sd, facecolor="#e2e6ea",
                    zorder=2)
    ax.plot(h, mw, color=TEAL, lw=1.7, zorder=5, label="wind")
    ax.plot(h, ms, color=GREEN, lw=1.7, zorder=5, label="solar")
    ax.plot(h, mw + ms, color=INK, lw=1.5, ls=(0, (3.5, 2.0)), zorder=6,
            label="combined")

    cv = 100.0 * (mw + ms).std() / (mw + ms).mean()
    ax.text(0.5, (mw + ms).max() * 1.19,
            "hourly correlation r = {:+.2f}\ncombined profile varies "
            "{:.0f} % about its mean".format(r, cv), ha="left", va="top",
            fontsize=6.2, color=SUB, linespacing=1.32, zorder=8)

    ax.set_xlim(0.0, 23.0)
    ax.set_ylim(0.0, (mw + ms).max() * 1.42)
    ax.set_xticks(np.arange(0, 24, 6))
    ax.set_xlabel("hour of day (hours)", fontsize=6.8)
    ax.set_ylabel("mean output (MW)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, ncol=3, columnspacing=0.7,
              borderaxespad=0.25)
    ax.set_title("The week folded onto one day", fontsize=8.0,
                 fontweight="bold", color=INK)
    return cv


def sweep_panel(ax, load, wind, solar, base_pct):
    scales = np.linspace(0.6, 2.4, 19)
    plain, batt = [], []
    for s in scales:
        a = account(load, wind, solar, scale=s)
        b = account(load, wind, solar, scale=s, battery=True)
        plain.append(100.0 * a["curt"].sum() / a["avail"].sum())
        batt.append(100.0 * b["curt"].sum() / b["avail"].sum())
    plain, batt = np.array(plain), np.array(batt)
    cap = scales * (WIND_CAP + SOLAR_CAP) / 1000.0

    ax.fill_between(cap, batt, plain, facecolor="#e6dff0", alpha=0.85,
                    zorder=3)
    ax.plot(cap, plain, color=ORANGE, lw=1.7, marker="o", ms=2.6, zorder=6,
            label="no storage")
    ax.plot(cap, batt, color=PURPLE, lw=1.7, marker="o", ms=2.6, zorder=7,
            label="{:.0f} MW / {:.0f} MWh battery".format(BAT_P, BAT_E))

    now = (WIND_CAP + SOLAR_CAP) / 1000.0
    ax.axvline(now, color=SUB, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    ax.plot([now], [base_pct], marker="o", ms=4.8, mfc=ORANGE, mec="white",
            mew=0.9, zorder=8)
    ax.annotate("{:.1f} GW today\n{:.1f} % spilled".format(now, base_pct),
                xy=(now, base_pct), xytext=(now + 0.35, plain.max() * 0.52),
                fontsize=6.1, color=SUB, ha="left", va="top",
                linespacing=1.30, zorder=9,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7, shrinkA=2,
                                shrinkB=3))

    ax.set_xlim(cap[0], cap[-1])
    ax.set_ylim(0.0, plain.max() * 1.12)
    ax.set_xlabel("installed wind plus solar (GW)", fontsize=6.8)
    ax.set_ylabel("energy curtailed (per cent)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.0, frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.92, handlelength=1.4,
              borderaxespad=0.25, labelspacing=0.24)
    ax.set_title("Curtailment as the fleet grows", fontsize=8.0,
                 fontweight="bold", color=INK)
    return plain, batt


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
    rng = np.random.default_rng(4180312)
    load, wind, solar = week(rng)
    acc = account(load, wind, solar)

    assert np.allclose(acc["used"] + acc["curt"], acc["avail"], atol=1e-9), \
        "used plus curtailed must equal the available resource"
    assert np.all(acc["used"] <= acc["head"] + 1e-9), "headroom violated"

    day = np.arange(HOURS) / 24.0
    r = float(np.corrcoef(wind, solar)[0, 1])
    base_pct = 100.0 * acc["curt"].sum() / acc["avail"].sum()
    cf_w = 100.0 * wind.mean() / WIND_CAP
    cf_s = 100.0 * solar.mean() / SOLAR_CAP

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_diu = axes_inches(fig, AUX_T)
    ax_swp = axes_inches(fig, AUX_B)

    main_panel(ax_main, day, load, wind, solar, acc)
    cv = diurnal_panel(ax_diu, wind, solar, r)
    plain, batt = sweep_panel(ax_swp, load, wind, solar, base_pct)



    panel_labels_inches(fig, [ax_main, ax_diu, ax_swp], dx=-0.62, dy=0.14)
    savefig(fig, "970_pwr_vre_curtailment.png")


if __name__ == "__main__":
    main()
