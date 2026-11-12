"""Thermal stress to coral mortality, on one shared 1998-2024 time axis.

The three stacked panels are a causal chain, which is why they must share an
x axis and be read downward.  Sea-surface temperature carries the bleaching
threshold (the maximum monthly mean climatology plus 1 degree C) and the red
fill marks every month that crosses it.  Degree heating weeks are accumulated
from exactly that exceedance over a rolling twelve-week window, so the DHW
curve is a transform of the panel above rather than a second opinion.
Bleaching cover and colony mortality are then driven by DHW alone.

Vertical rules mark the years whose peak DHW passed the 8 degree C-week
mortality line; they are located by argmax on the DHW series, not typed in.
The side panel takes the two worst of those events and compares how far down
the reef slope the bleaching reached - shallow colonies bleach almost
completely while the mesophotic zone is buffered.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import (axes_inches, panel_labels_inches, shared_limits,
                    halo_text)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
SEA = "#3b5b92"
HOT = "#b23a48"
DHW_FILL, DHW_EDGE = "#f4c088", "#cc722f"
COVER = "#c9a227"
MORT = "#b23a48"

FW, FH = 10.4, 6.4
COL_X, COL_W = 0.82, 5.26
ROW_H, ROW_GAP = 1.22, 0.22
ROW3 = (COL_X, 1.06, COL_W, ROW_H)
ROW2 = (COL_X, 1.06 + ROW_H + ROW_GAP, COL_W, ROW_H)
ROW1 = (COL_X, 1.06 + 2 * (ROW_H + ROW_GAP), COL_W, ROW_H)
AUX = (7.02, 1.86, 2.86, 2.44)

Y0, Y1 = 1998, 2024
WEEKS_PER_MONTH = 4.345
DHW_WATCH, DHW_ALERT = 4.0, 8.0
DEPTHS = np.array([2.0, 5.0, 8.0, 12.0, 16.0, 22.0, 30.0])
EVENT_AMP = {1998: 1.35, 2002: 0.72, 2006: 0.55, 2010: 1.02,
             2016: 1.58, 2020: 1.14, 2024: 1.28}


def build_series(rng):
    t = np.arange(Y0, Y1 + 1.0, 1.0 / 12.0)[:-1]
    seasonal = 1.70 * np.cos(2.0 * np.pi * (t - 0.17))
    trend = 0.0165 * (t - Y0)
    sst = 27.55 + seasonal + trend
    for yr, amp in EVENT_AMP.items():
        sst = sst + amp * np.exp(-0.5 * ((t - (yr + 0.24)) / 0.135) ** 2)
    sst = sst + rng.normal(0.0, 0.16, t.size)

    base = t < (Y0 + 10)                       # climatology baseline decade
    month = np.round((t - np.floor(t)) * 12.0).astype(int) % 12
    mmm = max(float(np.mean(sst[base & (month == m)])) for m in range(12))
    thresh = mmm + 1.0

    hotspot = np.clip(sst - mmm, 0.0, None)
    dhw = np.zeros_like(sst)
    for k in range(sst.size):
        lo = max(0, k - 2)
        dhw[k] = np.sum(hotspot[lo:k + 1]) * WEEKS_PER_MONTH

    cover = np.zeros_like(sst)
    mort = np.zeros_like(sst)
    for k in range(sst.size):
        target = 94.0 / (1.0 + np.exp(-(dhw[k] - 6.2) / 1.45))
        prev = cover[k - 1] if k else 0.0
        cover[k] = max(target, prev * 0.845)
        mtarget = 62.0 / (1.0 + np.exp(-(dhw[k] - 10.8) / 1.70))
        mort[k] = max(mtarget, (mort[k - 1] if k else 0.0) * 0.976)
    cover = np.clip(cover + rng.normal(0.0, 1.1, sst.size), 0.0, 100.0)
    mort = np.clip(mort + rng.normal(0.0, 0.8, sst.size), 0.0, 100.0)
    return dict(t=t, sst=sst, mmm=mmm, thresh=thresh, dhw=dhw, cover=cover,
                mort=mort)


def find_events(s):
    ev = []
    for yr in range(Y0, Y1 + 1):
        m = (s["t"] >= yr) & (s["t"] < yr + 1)
        if not np.any(m):
            continue
        peak = float(np.max(s["dhw"][m]))
        if peak >= DHW_ALERT:
            k = int(np.argmax(np.where(m, s["dhw"], -1.0)))
            ev.append(dict(year=yr, dhw=peak, cover=float(s["cover"][k]),
                           mort=float(np.max(s["mort"][m]))))
    return ev


def depth_profile(peak_cover, dhw, rng):
    """Bleached cover (%) down the reef slope for one event."""
    z_scale = 6.0 + 0.95 * dhw
    prof = peak_cover * np.exp(-((DEPTHS - 1.5) / z_scale) ** 1.35)
    err = 3.0 + 0.07 * prof
    return np.clip(prof + rng.normal(0.0, 1.4, DEPTHS.size), 0.0, 100.0), err


def draw_sst(ax, s, events):
    ax.plot(s["t"], s["sst"], color=SEA, lw=1.0, zorder=4)
    ax.fill_between(s["t"], s["thresh"], s["sst"],
                    where=s["sst"] > s["thresh"], color=HOT, alpha=0.55,
                    linewidth=0.0, interpolate=True, zorder=3)
    ax.axhline(s["thresh"], color=HOT, lw=1.0, ls=(0, (4.0, 2.2)), zorder=5)
    halo_text(ax, Y0 + 5.4, s["thresh"] + 0.14,
              "bleaching threshold  MMM + 1 = {:.1f} $\\degree$C"
              .format(s["thresh"]), fontsize=6.2, color=HOT, halo="white",
              halo_lw=2.4, ha="left", va="bottom", zorder=6)
    ax.set_ylim(s["sst"].min() - 0.35, s["sst"].max() + 0.55)
    ax.set_ylabel("SST ($\\degree$C)", fontsize=7.8)
    ax.set_title("Sea-surface temperature against the bleaching threshold",
                 fontsize=8.5, fontweight="bold")


def draw_dhw(ax, s, events):
    ax.fill_between(s["t"], 0.0, s["dhw"], color=DHW_FILL, linewidth=0.0,
                    zorder=3)
    ax.plot(s["t"], s["dhw"], color=DHW_EDGE, lw=0.9, zorder=4)
    for lev, col, lab in ((DHW_WATCH, "#a04c18", "alert level 1"),
                          (DHW_ALERT, HOT, "alert level 2")):
        ax.axhline(lev, color=col, lw=0.9, ls=(0, (4.0, 2.2)), zorder=5)
        halo_text(ax, Y0 + 5.4, lev + 0.35,
                  "{:s}  {:.0f} $\\degree$C-weeks".format(lab, lev),
                  fontsize=6.2, color=col, halo="white", halo_lw=2.4,
                  ha="left", va="bottom", zorder=6)
    ax.set_ylim(0.0, s["dhw"].max() * 1.30)
    for e in events:
        ax.text(e["year"] + 0.25, e["dhw"] + 0.6, str(e["year"]),
                fontsize=6.2, color=INK, va="bottom", zorder=7,
                ha="left" if e["year"] <= Y0 + 1 else "center")
    ax.set_ylabel("DHW ($\\degree$C-weeks)", fontsize=7.8)
    ax.set_title("Accumulated heat stress over a 12-week window",
                 fontsize=8.5, fontweight="bold")


def draw_impact(ax, s):
    ax.fill_between(s["t"], 0.0, s["cover"], color=COVER, alpha=0.30,
                    linewidth=0.0, zorder=2)
    ax.plot(s["t"], s["cover"], color=COVER, lw=1.2, zorder=4,
            label="bleached cover")
    ax.plot(s["t"], s["mort"], color=MORT, lw=1.4, zorder=5,
            label="colony mortality")
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", ncol=2,
              handlelength=1.5, borderpad=0.15, columnspacing=1.2)
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("reef area (%)", fontsize=7.8)
    ax.set_xlabel("year", fontsize=7.8)
    ax.set_title("Bleached cover and cumulative colony mortality",
                 fontsize=8.5, fontweight="bold")


def draw_depth(ax, s, events, rng):
    worst = sorted(events, key=lambda e: -e["dhw"])[:2]
    cols = [HOT, "#3d6b8e"]
    profiles = []
    for e, col in zip(worst, cols):
        prof, err = depth_profile(e["cover"], e["dhw"], rng)
        profiles.append(prof)
        ax.errorbar(prof, DEPTHS, xerr=err, color=col, lw=1.4, marker="o",
                    ms=3.6, mfc="white", mew=1.0, elinewidth=0.7, capsize=2.0,
                    zorder=4,
                    label="{:d}  ({:.1f} $\\degree$C-weeks)".format(
                        e["year"], e["dhw"]))
    lo, hi = shared_limits(profiles, pad=0.10, lo=0.0)
    ax.set_xlim(0.0, min(hi, 108.0))
    ax.axhspan(0.0, 10.0, color="#f0f0f0", zorder=0)
    ax.text(0.03, 0.975, "shallow reef flat", transform=ax.transAxes,
            fontsize=6.2, color=MUTED, ha="left", va="top", zorder=2)
    ax.text(0.03, 0.03, "mesophotic refuge", transform=ax.transAxes,
            fontsize=6.2, color=MUTED, ha="left", va="bottom", zorder=2)
    ax.set_ylim(33.0, 0.0)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right",
              handlelength=1.5, borderpad=0.15)
    ax.set_xlabel("bleached cover (%)", fontsize=7.8)
    ax.set_ylabel("depth (m)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Bleaching against depth, two worst events", fontsize=8.5,
                 fontweight="bold")
    return worst, profiles


def main():
    set_pub_style()
    rng = np.random.default_rng(692)
    s = build_series(rng)
    events = find_events(s)

    fig = plt.figure(figsize=(FW, FH))
    ax1 = axes_inches(fig, ROW1)
    ax2 = axes_inches(fig, ROW2)
    ax3 = axes_inches(fig, ROW3)
    axd = axes_inches(fig, AUX)

    draw_sst(ax1, s, events)
    draw_dhw(ax2, s, events)
    draw_impact(ax3, s)
    worst, profiles = draw_depth(axd, s, events, rng)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(Y0, Y1 + 1)
        ax.set_xticks(np.arange(Y0 + 2, Y1 + 1, 4))
        ax.tick_params(labelsize=6.8)
        for e in events:
            ax.axvline(e["year"] + 0.25, color="#9a9a9a", lw=0.8,
                       ls=(0, (2.6, 2.2)), zorder=1)
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    panel_labels_inches(fig, [ax1, ax2, ax3, axd], dx=-0.48, dy=0.14)

    shallow = profiles[0][0]
    deep = profiles[0][-1]

    print("mmm={:.2f} thresh={:.2f} events={:s} mort={:.1f}".format(
        s["mmm"], s["thresh"], str([e["year"] for e in events]),
        s["mort"][-1]))
    savefig(fig, "692_mar_coral_bleaching.png")


if __name__ == "__main__":
    main()
