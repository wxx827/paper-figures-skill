"""Figure 670 - Line balance against takt, and what it costs in output.

Main panel: one horizontal bar per station, split into the three things a part
spends its cycle on.  The takt line is the customer's demand rate, so any bar
crossing it is a station the line cannot sustain; the longest one is the
bottleneck and is banded in red, because it - not the average - sets the rate
of the whole line.

Right column: (b) the OEE factors measured over the same shifts, with the
world-class reference; (c) the output those factors actually produced, against
both the takt-implied capacity and the ceiling the bottleneck imposes.  The
gap called out in (c) is the difference between the two lines drawn there.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import axes_inches, panel_label_inches

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"

PROC = "#3b5b92"       # house blue - value-adding processing
WAIT = "#c9a227"       # house gold - queueing
TRAN = "#808080"       # neutral    - transfer / handling
GAP = "#b23a48"        # house red  - the single emphasis: bottleneck and gap
GREEN = "#2e7d5b"

STATIONS = ["OP-10 load", "OP-20 mill", "OP-30 drill", "OP-40 deburr",
            "OP-50 wash", "OP-60 gauge", "OP-70 assemble", "OP-80 pack"]
PROC_S = np.array([38.0, 54.0, 47.0, 33.0, 29.0, 22.0, 58.0, 31.0])
WAIT_S = np.array([6.0, 9.0, 7.0, 14.0, 5.0, 4.0, 18.0, 8.0])
TRAN_S = np.array([5.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 9.0])
TAKT_S = 62.0          # customer demand, seconds per part
N_SHIFT = 12
OEE_NAMES = ["Availability", "Performance", "Quality", "OEE"]
WORLD_CLASS = 85.0     # conventional world-class OEE, percent


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(670)

    proc = PROC_S * np.exp(rng.normal(0.0, 0.035, PROC_S.size))
    wait = WAIT_S * np.exp(rng.normal(0.0, 0.10, WAIT_S.size))
    tran = TRAN_S * np.exp(rng.normal(0.0, 0.06, TRAN_S.size))
    cycle = proc + wait + tran
    bott = int(np.argmax(cycle))

    avail = np.clip(rng.normal(0.912, 0.030, N_SHIFT), 0.70, 1.0)
    perf = np.clip(rng.normal(0.846, 0.045, N_SHIFT), 0.60, 1.0)
    qual = np.clip(rng.normal(0.981, 0.008, N_SHIFT), 0.90, 1.0)
    oee = avail * perf * qual

    cap_takt = 3600.0 / TAKT_S              # units/h the demand rate implies
    cap_bott = 3600.0 / cycle[bott]         # ceiling set by the slowest station
    output = cap_bott * oee
    gap = cap_takt - output.mean()

    fig = plt.figure(figsize=(10.4, 6.4))

    # ---- (a) station cycle time against takt -------------------------------
    axm = axes_inches(fig, (0.90, 0.76, 5.18, 4.66))
    ypos = np.arange(len(STATIONS))[::-1]
    axm.axhspan(ypos[bott] - 0.46, ypos[bott] + 0.46,
                facecolor=lighten(GAP, 0.90), edgecolor="none", zorder=0)

    left = np.zeros_like(cycle)
    for seg, col, name in ((proc, PROC, "processing"), (wait, WAIT, "waiting"),
                           (tran, TRAN, "transfer")):
        axm.barh(ypos, seg, left=left, height=0.58, color=col,
                 edgecolor="white", linewidth=0.8, label=name, zorder=3)
        left += seg

    axm.axvline(TAKT_S, color=GAP, lw=1.3, ls=(0, (5.0, 2.4)), zorder=5)
    axm.text(TAKT_S + 0.9, ypos[0] + 0.62, f"takt {TAKT_S:.0f} s", ha="left",
             va="bottom", fontsize=7.2, color=GAP, fontweight="bold")

    over = cycle > TAKT_S
    for i, (y, c, hot) in enumerate(zip(ypos, cycle, over)):
        note = (f"{c:.0f} s    bottleneck, {c - TAKT_S:+.0f} s vs takt"
                if i == bott else f"{c:.0f} s")
        axm.text(c + 1.4, y, note, ha="left", va="center", fontsize=7.0,
                 color=GAP if hot else INK,
                 fontweight="bold" if hot else "normal")

    axm.set_yticks(ypos)
    axm.set_yticklabels(STATIONS)
    for tick, hot in zip(axm.get_yticklabels(), over):
        tick.set_color(GAP if hot else INK)
    axm.set_xlim(0.0, cycle.max() * 1.38)
    axm.set_ylim(ypos[-1] - 0.85, ypos[0] + 0.95)
    axm.set_xlabel("cycle time per part (s)", fontsize=8.0)
    axm.tick_params(labelsize=7.0, length=2.6)
    axm.grid(axis="x", color=GRID, lw=0.6)
    axm.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
               fontsize=7.0, frameon=False, handlelength=1.3,
               columnspacing=1.4, handletextpad=0.45)
    axm.set_title(f"{int(over.sum())} of {len(STATIONS)} stations exceed takt",
                  fontsize=9.0, fontweight="bold", pad=20.0)
    for s in ("top", "right"):
        axm.spines[s].set_visible(False)

    # ---- (b) OEE factors ---------------------------------------------------
    axo = axes_inches(fig, (7.02, 3.36, 3.02, 2.06))
    vals = np.array([avail.mean(), perf.mean(), qual.mean(), oee.mean()]) * 100.0
    errs = np.array([avail.std(), perf.std(), qual.std(), oee.std()]) * 100.0
    cols = [lighten(PROC, 0.52), lighten(PROC, 0.34), lighten(PROC, 0.16),
            darken(PROC, 0.20)]
    x = np.arange(len(OEE_NAMES))
    axo.bar(x, vals, width=0.62, color=cols, edgecolor="white", linewidth=0.8,
            yerr=errs, error_kw=dict(ecolor=SUB, elinewidth=0.8, capsize=2.2),
            zorder=3)
    axo.axhline(WORLD_CLASS, color=GAP, lw=0.9, ls=(0, (3.0, 2.0)), zorder=4)
    axo.text(len(OEE_NAMES) - 0.62, WORLD_CLASS + 1.6,
             f"world class {WORLD_CLASS:.0f} %", ha="right", va="bottom",
             fontsize=6.4, color=GAP)
    for xi, v, e in zip(x, vals, errs):
        axo.text(xi, v + e + 2.2, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=6.8, color=INK)
    axo.set_xticks(x)
    axo.set_xticklabels(OEE_NAMES, rotation=12, ha="right")
    axo.set_xlim(-0.62, len(OEE_NAMES) - 0.38)
    axo.set_ylim(0.0, 112.0)
    axo.set_ylabel("rate over the window (%)", fontsize=7.4)
    axo.tick_params(labelsize=6.5, length=2.2)
    axo.grid(axis="y", color=GRID, lw=0.6)
    axo.set_title(f"OEE {vals[-1]:.1f} %, "
                  f"{WORLD_CLASS - vals[-1]:.1f} pp below world class",
                  fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axo.spines[s].set_visible(False)

    # ---- (c) output per shift ---------------------------------------------
    axc = axes_inches(fig, (7.02, 0.76, 3.02, 1.94))
    shifts = np.arange(1, N_SHIFT + 1)
    axc.fill_between(shifts, output, cap_takt, facecolor=lighten(GAP, 0.86),
                     edgecolor="none", zorder=1)
    axc.axhline(cap_takt, color=GAP, lw=1.1, ls=(0, (5.0, 2.4)), zorder=4,
                label=f"takt capacity {cap_takt:.0f}")
    axc.axhline(cap_bott, color=BASE, lw=1.0, ls=(0, (2.6, 2.0)), zorder=4,
                label=f"bottleneck ceiling {cap_bott:.0f}")
    axc.plot(shifts, output, lw=1.5, color=PROC, marker="o", ms=3.2,
             mfc="white", mec=PROC, mew=0.9, zorder=5,
             label=f"actual {output.mean():.0f}")
    axc.set_xlim(0.6, N_SHIFT + 0.4)
    axc.set_ylim(output.min() - 5.0, cap_takt + 5.0)
    axc.set_xticks(shifts[::2])
    axc.set_xlabel("shift (index)", fontsize=7.4)
    axc.set_ylabel("output (units/h)", fontsize=7.4)
    axc.tick_params(labelsize=6.5, length=2.2)
    axc.grid(axis="y", color=GRID, lw=0.6)
    axc.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3,
               fontsize=6.4, frameon=False, handlelength=1.3,
               columnspacing=0.9, handletextpad=0.4)
    axc.text(1.2, (cap_takt + cap_bott) / 2.0 + 1.0,
             f"{gap:.1f} units/h short of takt", ha="left", va="center",
             fontsize=6.6, color=GAP)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    for ax, lab in ((axm, "(a)"), (axo, "(b)"), (axc, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.72, dy=0.12)
    savefig(fig, "670_iot_production_takt.png")


if __name__ == "__main__":
    main()
