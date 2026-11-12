"""Ultrafiltration fouling: one resistance ledger drives every panel.

The rack runs at constant flux, so Darcy's law makes transmembrane pressure a
direct read-out of total resistance.  Resistance is tracked as three terms - a
fixed membrane term, a reversible cake that builds through each filtration
cycle, and an irreversible term that grows by the fraction of cake the backwash
fails to lift - and the sawtooth in the main panel is simply that ledger pushed
through Darcy.  The start-of-cycle points therefore *are* the irreversible
baseline, the chemical clean is a reset of one term, the permeability curve in
the upper side panel is the same pressures divided into the same flux, and the
stacked areas in the lower one are the ledger itself.  Fouling rates are least
squares fits to the baseline, before and after the clean.
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
BLUE = "#3b5b92"
DEEP = "#134a76"
PALE = "#9cc3dd"
GREEN = "#2e7d5b"
LEAF = "#6bbf8a"
BROWN = "#7a6a5d"
RED = "#b23a48"
ORANGE = "#c0603f"

SEED = 1006
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

FLUX_LMH = 55.0            # design flux held constant, L/(m2 h)
MU = 1.002e-3              # dynamic viscosity of water at 20 C, Pa s
R_MEM = 1.55e12            # clean membrane resistance, 1/m
CAKE_RATE = 2.45e12        # cake resistance built per hour of filtration, 1/m
IRREV_FRAC = 0.040         # cake fraction the backwash cannot lift
CYCLE_MIN = 30.0           # filtration time between backwashes, min
BW_MIN = 1.0               # backwash duration, min
N_CYCLE = 68               # cycles in the campaign
CIP_CYCLE = 44             # chemical clean after this many cycles
CIP_RECOVERY = 0.86        # irreversible resistance removed by the clean
TMP_ALARM = 55.0           # cycle-start pressure that calls for a clean, kPa


def flux_ms():
    return FLUX_LMH / 1000.0 / 3600.0


def tmp_kpa(resistance):
    """Darcy: constant flux turns resistance straight into pressure."""
    return MU * flux_ms() * np.asarray(resistance, dtype=float) / 1000.0


def campaign(rng):
    """Time, resistance components and TMP for the whole run."""
    t, rm, rrev, rirr, cyc = [], [], [], [], []
    clock = 0.0
    irr = 0.0
    cycle_h = CYCLE_MIN / 60.0
    for k in range(N_CYCLE):
        tt = np.linspace(0.0, cycle_h, 14)
        cake = CAKE_RATE * tt * (1.0 + 0.055 * rng.normal())
        t.append(clock + tt)
        rm.append(np.full(tt.size, R_MEM))
        rrev.append(cake)
        rirr.append(np.full(tt.size, irr))
        cyc.append(np.full(tt.size, k))
        irr += IRREV_FRAC * float(cake[-1])
        clock += cycle_h + BW_MIN / 60.0
        if k + 1 == CIP_CYCLE:
            irr *= (1.0 - CIP_RECOVERY)
            clock += 1.5                                  # clean-in-place, h
    t = np.concatenate(t)
    rm, rrev, rirr = (np.concatenate(a) for a in (rm, rrev, rirr))
    cyc = np.concatenate(cyc)
    return t, rm, rrev, rirr, cyc, tmp_kpa(rm + rrev + rirr)


def baseline(t, tmp, cyc):
    """Start-of-cycle pressure: the irreversible fouling trend."""
    first = np.array([np.flatnonzero(cyc == k)[0] for k in range(N_CYCLE)])
    return t[first], tmp[first]


def fit_rate(tb, pb, lo, hi):
    """Least-squares fouling rate over a window, kPa per hour."""
    m = (tb >= lo) & (tb <= hi)
    a, b = np.polyfit(tb[m], pb[m], 1)
    return float(a), float(b), tb[m]


def main_panel(ax, t, tmp, tb, pb, fits, t_cip):
    ax.axvspan(t_cip[0], t_cip[1], facecolor="#dfe9df", edgecolor="none",
               zorder=1)
    ax.axhline(TMP_ALARM, color=RED, lw=1.0, ls=(0, (4.5, 2.4)), zorder=4)
    ax.text(0.4, TMP_ALARM + 0.9,
            "cycle-start alarm {:.0f} kPa".format(TMP_ALARM), fontsize=6.3,
            color=RED, ha="left", va="bottom", zorder=7)

    ax.plot(t, tmp, color=DEEP, lw=0.75, zorder=5)
    ax.plot(tb, pb, marker="o", ms=2.9, mfc=ORANGE, mec="white", mew=0.4,
            ls="none", zorder=8)
    for (a, b, tw), col in zip(fits, (ORANGE, GREEN)):
        ax.plot(tw, a * tw + b, color=col, lw=1.5, ls=(0, (5.0, 2.2)),
                zorder=9)
        ax.text(tw.mean(), a * tw.mean() + b - 2.6,
                "{:+.2f} kPa/h".format(a), fontsize=6.4, color=col,
                ha="center", va="top", fontweight="bold", zorder=10)

    lo = int(np.flatnonzero(t <= t_cip[0])[-1])
    hi = int(np.flatnonzero(t >= t_cip[1])[0])
    ax.annotate("chemical clean removes {:.0f} % of the\nirreversible "
                "resistance, {:.1f} kPa recovered"
                .format(100.0 * CIP_RECOVERY, tmp[lo] - tmp[hi]),
                xy=(t_cip[1], tmp[hi]),
                xytext=(t_cip[1] - 1.5, tmp[hi] - 13.0), fontsize=6.5,
                color=GREEN, ha="right", va="top", linespacing=1.32,
                zorder=10, arrowprops=dict(arrowstyle="-", color=GREEN,
                                           lw=0.8, shrinkA=2, shrinkB=4))

    k0 = np.flatnonzero((t > 3.0) & (t < 5.0))
    ax.annotate("one cycle: {:.0f} min of filtration builds\n{:.1f} kPa of "
                "cake, backwash returns all but {:.0f} %"
                .format(CYCLE_MIN, tmp[k0].max() - tmp[k0].min(),
                        100.0 * IRREV_FRAC),
                xy=(t[k0][int(np.argmax(tmp[k0]))], tmp[k0].max()),
                xytext=(0.6, tmp.max() * 1.16),
                fontsize=6.5, color=DEEP, ha="left", va="bottom",
                linespacing=1.32, zorder=10,
                arrowprops=dict(arrowstyle="-", color=DEEP, lw=0.8,
                                shrinkA=2, shrinkB=4))

    handles = [
        Line2D([], [], color=DEEP, lw=0.9,
               label="transmembrane pressure, {:.0f} LMH".format(FLUX_LMH)),
        Line2D([], [], color="none", marker="o", mfc=ORANGE, mec="white",
               ms=4.0, label="start of each filtration cycle"),
        Line2D([], [], color=ORANGE, lw=1.5, ls=(0, (5.0, 2.2)),
               label="fouling rate, least squares"),
        Patch(facecolor="#dfe9df", edgecolor="none",
              label="clean-in-place, {:.1f} h".format(t_cip[1] - t_cip[0])),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.9, borderaxespad=0.3, labelspacing=0.28)

    ax.set_xlim(0.0, t.max())
    ax.set_ylim(0.0, tmp.max() * 1.30)
    ax.set_xlabel("elapsed operating time (h)")
    ax.set_ylabel("transmembrane pressure (kPa)")
    ax.grid(lw=0.5, alpha=0.32)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("{:d} filtration cycles at constant flux, one chemical clean"
                 .format(N_CYCLE), fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def permeability_panel(ax, tb, pb, t_cip):
    volume = FLUX_LMH * tb                       # permeate per membrane area
    perm = FLUX_LMH / (pb / 100.0)               # LMH per bar
    pre = tb <= t_cip[0]
    ax.plot(volume[pre], perm[pre], color=DEEP, lw=1.5, zorder=5)
    ax.plot(volume[~pre], perm[~pre], color=GREEN, lw=1.5, zorder=6)
    ax.plot([volume[pre][0]], [perm[pre][0]], marker="o", ms=4.2, mfc=PALE,
            mec=DEEP, mew=0.8, zorder=8)
    loss = 100.0 * (1.0 - perm[pre][-1] / perm[pre][0])
    ax.annotate("{:.0f} % of the clean permeability\nlost before the clean"
                .format(loss), xy=(volume[pre][-1], perm[pre][-1]),
                xytext=(volume[pre][-1] * 0.50, perm[pre][0] * 0.94),
                fontsize=6.1, color=DEEP, ha="left", va="top",
                linespacing=1.30, zorder=9,
                arrowprops=dict(arrowstyle="-", color=DEEP, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.text(volume[-1], perm[-1] + 6.0, "after the clean", fontsize=6.1,
            color=GREEN, ha="right", va="bottom", zorder=8)
    ax.set_xlim(0.0, volume.max() * 1.04)
    ax.set_ylim(0.0, perm.max() * 1.16)
    ax.set_xlabel("cumulative permeate (L per m2 of membrane)", fontsize=6.8)
    ax.set_ylabel("permeability (LMH/bar)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Permeability against throughput", fontsize=8.0,
                 fontweight="bold", color=INK)
    return loss


def ledger_panel(ax, t, rm, rrev, rirr):
    scale = 1e-12
    ax.stackplot(t, scale * rm, scale * rirr, scale * rrev,
                 colors=["#c9d3de", BROWN, LEAF], edgecolor="none",
                 zorder=3, alpha=0.95)
    ax.plot(t, scale * (rm + rirr + rrev), color=INK, lw=0.6, zorder=6)
    ax.text(t.max() * 0.03, scale * R_MEM * 0.45, "membrane", fontsize=6.0,
            color=SUB, ha="left", va="center", zorder=7)
    ax.text(t.max() * 0.52, scale * (R_MEM + rirr.max() * 0.45),
            "irreversible", fontsize=6.0, color="#ffffff", ha="center",
            va="center", zorder=7)
    ax.text(t.max() * 0.98, scale * (R_MEM + rirr[-1] + rrev.max() * 1.1),
            "reversible cake", fontsize=6.0, color=GREEN, ha="right",
            va="bottom", zorder=7)
    ax.set_xlim(0.0, t.max())
    ax.set_ylim(0.0, scale * (rm + rirr + rrev).max() * 1.22)
    ax.set_xlabel("elapsed operating time (h)", fontsize=6.8)
    ax.set_ylabel("resistance (1e12 per m)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("The resistance ledger behind the sawtooth", fontsize=8.0,
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
    rng = np.random.default_rng(SEED)
    t, rm, rrev, rirr, cyc, tmp = campaign(rng)

    assert np.allclose(tmp, tmp_kpa(rm + rrev + rirr), rtol=0, atol=1e-12), \
        "the plotted pressure must be Darcy applied to the ledger"
    pre = cyc < CIP_CYCLE
    assert np.all(np.diff(rirr[pre]) >= -1e-6), \
        "irreversible fouling cannot fall while the rack keeps running"
    assert rirr[cyc == CIP_CYCLE][0] < rirr[cyc == CIP_CYCLE - 1][0], \
        "the clean has to remove irreversible resistance"

    tb, pb = baseline(t, tmp, cyc)
    t_cip = (float(t[cyc == CIP_CYCLE - 1][-1]),
             float(t[cyc == CIP_CYCLE][0]))
    fits = [fit_rate(tb, pb, 1.0, t_cip[0]),
            fit_rate(tb, pb, t_cip[1], t.max())]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_perm = axes_inches(fig, AUX_T)
    ax_res = axes_inches(fig, AUX_B)

    main_panel(ax_main, t, tmp, tb, pb, fits, t_cip)
    loss = permeability_panel(ax_perm, tb, pb, t_cip)
    ledger_panel(ax_res, t, rm, rrev, rirr)


    thr = FLUX_LMH * float(tb[-1])

    panel_labels_inches(fig, [ax_main, ax_perm, ax_res], dx=-0.62, dy=0.14)
    savefig(fig, "1006_wsn_membrane_fouling.png")


if __name__ == "__main__":
    main()
