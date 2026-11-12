"""Night-flow leakage assessment: the loss is measured, not assumed.

A district metered area is logged for a week at fifteen-minute resolution.
The inflow trace is built from a diurnal consumption pattern plus a constant
background loss and a burst that starts midweek, and the minimum night flow is
then *read back* from that trace between 02:00 and 04:00 each night.  Legitimate
night use is subtracted from it to give the leakage rate, and the daily leakage
volume, the loss share and the post-repair recovery all follow from the same
series.  The upper side panel ranks six districts on infrastructure leakage
index computed from their own night flows; the lower one shows the pressure
signature the burst leaves on the two loggers that bracket it, since a burst is
a hydraulic event before it is an accounting one.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, lighten

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
DEEP = "#134a76"
PALE = "#9cc3dd"
GREEN = "#2e7d5b"
RED = "#b23a48"
ORANGE = "#c0603f"

SEED = 1004
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

DAYS = 7
STEP_H = 0.25                  # logger interval, h
N_PROP = 1850                  # properties served by the district
USE_PER_PROP = 0.72e-3         # legitimate night use per property, L/s
NIGHT_USE = USE_PER_PROP * N_PROP
BURST_DAY = 3.35               # burst starts partway through day 4
REPAIR_DAY = 5.55              # burst repaired on day 6
BURST_RATE = 4.6               # burst discharge, L/s
BACKGROUND = 2.9               # background leakage at the logged pressure, L/s
NIGHT_LO, NIGHT_HI = 2.0, 4.0  # minimum night flow window, h


def diurnal(hours):
    """Consumption pattern, L/s: it has to reach the night-use floor at 03:00.

    A plain sinusoid never gets close enough to zero, which would leave the
    night flow full of real consumption and inflate the leakage estimate.
    """
    h = hours % 24.0
    p = 0.5 * (1.0 - np.cos(2.0 * np.pi * (h - 3.0) / 24.0))
    return (NIGHT_USE + 12.5 * p ** 1.25
            + 3.2 * np.exp(-0.5 * ((h - 8.0) / 1.5) ** 2)
            + 3.6 * np.exp(-0.5 * ((h - 20.0) / 1.8) ** 2))


def inflow_series(rng):
    """District meter inflow at 15 min resolution, L/s."""
    hours = np.arange(0.0, DAYS * 24.0, STEP_H)
    use = diurnal(hours)
    weekend = (hours // 24.0) >= 5.0
    use = NIGHT_USE + (use - NIGHT_USE) * np.where(weekend, 0.90, 1.0)
    burst = np.where((hours >= BURST_DAY * 24.0) &
                     (hours < REPAIR_DAY * 24.0), BURST_RATE, 0.0)
    noise = rng.normal(0.0, 0.24, hours.size)
    return hours, use + BACKGROUND + burst + noise, use, burst


def night_flows(hours, flow):
    """Minimum night flow per day, taken from the plotted series itself."""
    day = (hours // 24.0).astype(int)
    mnf, when = [], []
    for d in range(DAYS):
        m = (day == d) & (hours % 24.0 >= NIGHT_LO) & \
            (hours % 24.0 < NIGHT_HI)
        idx = np.flatnonzero(m)
        k = idx[int(np.argmin(flow[idx]))]
        mnf.append(float(flow[k]))
        when.append(float(hours[k]))
    return np.array(mnf), np.array(when)


def district_table(mnf_logged):
    """Six districts; the first one is the district logged in the main panel."""
    names = ["DMA-01", "DMA-02", "DMA-03", "DMA-04", "DMA-05", "DMA-06"]
    props = np.array([N_PROP, 1240, 2310, 960, 1580, 2050])
    mains = np.array([14.2, 9.8, 18.6, 7.1, 12.4, 16.0])       # km
    press = np.array([38.0, 44.0, 33.0, 51.0, 41.0, 36.0])     # m H2O
    mnf = np.array([mnf_logged, 1.78, 6.83, 1.96, 3.33, 4.53])  # L/s
    night_use = USE_PER_PROP * props                           # L/s
    leak = mnf - night_use
    # Unavoidable annual real losses: the leakage-index denominator, in L/s.
    uarl = (18.0 * mains + 0.8 * props) * press / 86400.0
    return names, props, mains, press, mnf, leak, leak / uarl


def pressure_signature(rng):
    """Head at two loggers, one inside the burst zone and one upstream."""
    hours = np.arange(0.0, DAYS * 24.0, STEP_H)
    base = 41.5 - 0.055 * (diurnal(hours) - 11.0) ** 2 * 0.5
    up = base + 3.6 + rng.normal(0.0, 0.10, hours.size)
    drop = np.where((hours >= BURST_DAY * 24.0) &
                    (hours < REPAIR_DAY * 24.0), 4.3, 0.0)
    dn = base - drop + rng.normal(0.0, 0.12, hours.size)
    return hours, up, dn


def main_panel(ax, hours, flow, use, mnf, when, leak):
    for d in range(DAYS):
        ax.axvspan(d * 24.0 + NIGHT_LO, d * 24.0 + NIGHT_HI,
                   facecolor="#e8eef5", edgecolor="none", zorder=1)
    ax.axvspan(BURST_DAY * 24.0, REPAIR_DAY * 24.0, facecolor="#f6ded0",
               edgecolor="none", alpha=0.75, zorder=0)

    ax.fill_between(hours, 0.0, use, facecolor=PALE, alpha=0.55, lw=0,
                    zorder=2)
    ax.plot(hours, flow, color=DEEP, lw=0.95, zorder=5)
    ax.plot(hours, use, color=BLUE, lw=0.9, ls=(0, (4.0, 2.0)), zorder=4)

    ax.plot(when, mnf, marker="o", ms=5.0, mfc=RED, mec="white", mew=0.9,
            ls="none", zorder=8)
    ax.axhline(NIGHT_USE, color=GREEN, lw=1.0, ls=(0, (4.5, 2.4)), zorder=6)
    ax.text(2.0, NIGHT_USE + 0.35, "legitimate night use {:.2f} L/s"
            .format(NIGHT_USE), fontsize=6.2, color=GREEN, ha="left",
            va="bottom", zorder=7)

    for d, (m, w) in enumerate(zip(mnf, when)):
        ax.annotate("", xy=(w, m), xytext=(w, NIGHT_USE),
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=0.7,
                                    shrinkA=0, shrinkB=0), zorder=7)
        side = "left" if w < 12.0 else "right"           # keep it on the page
        ax.text(w + (1.4 if side == "left" else -1.4), m + 0.45,
                "{:.1f} L/s".format(leak[d]), fontsize=6.0, color=RED,
                ha=side, va="bottom", zorder=9)

    ax.text(0.5 * (BURST_DAY + REPAIR_DAY) * 24.0, 23.4,
            "burst open for {:.0f} h from day {:d}"
            .format((REPAIR_DAY - BURST_DAY) * 24.0, int(BURST_DAY) + 1),
            fontsize=6.6, color=ORANGE, ha="center", va="top",
            linespacing=1.32, zorder=9)
    ax.annotate("night flow rises {:.1f} L/s\nthe morning after the burst"
                .format(leak[4] - leak[0]), xy=(when[4], mnf[4]),
                xytext=(when[4] + 9.0, mnf[4] + 5.2), fontsize=6.3,
                color=RED, ha="left", va="bottom", linespacing=1.30,
                zorder=10, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8,
                                           shrinkA=2, shrinkB=4))

    handles = [
        Line2D([], [], color=DEEP, lw=1.1, label="district meter inflow"),
        Line2D([], [], color=BLUE, lw=0.9, ls=(0, (4.0, 2.0)),
               label="metered consumption component"),
        Line2D([], [], color="none", marker="o", mfc=RED, mec="white",
               ms=5.0, label="minimum night flow, 02:00-04:00"),
        Line2D([], [], color=GREEN, lw=1.0, ls=(0, (4.5, 2.4)),
               label="legitimate night use"),
        Patch(facecolor="#f6ded0", edgecolor="none", label="burst open"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.7, ncol=2, columnspacing=1.0,
              borderaxespad=0.3, labelspacing=0.26)

    ax.set_xlim(0.0, DAYS * 24.0)
    ax.set_ylim(0.0, 26.0)
    ax.set_xticks(np.arange(0.0, DAYS * 24.0 + 1.0, 12.0))
    ax.set_xticklabels(["day {:d}".format(int(h // 24) + 1) if h % 24 == 0
                        else "12:00"
                        for h in np.arange(0.0, DAYS * 24.0 + 1.0, 12.0)],
                       fontsize=6.2)
    ax.set_xlabel("time over a one-week logging campaign (h)")
    ax.set_ylabel("district meter flow (L/s)")
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("One week of district inflow at 15 min resolution, "
                 "night window shaded", fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def index_panel(ax, names, leak, ili, props):
    y = np.arange(len(names))[::-1]
    cols = [RED if v >= 4.0 else (ORANGE if v >= 2.0 else GREEN) for v in ili]
    ax.barh(y, ili, height=0.66, color=cols, edgecolor="white", linewidth=0.6,
            zorder=4)
    for yi, v, lk, p in zip(y, ili, leak, props):
        ax.text(v + 0.09, yi, "{:.1f} L/s over {:d} properties"
                .format(lk, int(p)), fontsize=6, color=SUB, ha="left",
                va="center", zorder=6)
    for xv, lab in ((2.0, "band B"), (4.0, "band C")):
        ax.axvline(xv, color=MUTED, lw=0.7, ls=(0, (2.2, 2.0)), zorder=5)
        ax.text(xv + 0.05, -1.00, lab, fontsize=6, color=MUTED,
                ha="left", va="bottom", zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.3)
    ax.set_xlim(0.0, float(ili.max()) * 1.62)
    ax.set_ylim(-1.1, len(names) - 0.3)
    ax.set_xlabel("infrastructure leakage index (-)", fontsize=6.8)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Districts ranked by leakage index", fontsize=8.0,
                 fontweight="bold", color=INK)


def signature_panel(ax, hours, up, dn):
    ax.axvspan(BURST_DAY * 24.0, REPAIR_DAY * 24.0, facecolor="#f6ded0",
               edgecolor="none", alpha=0.75, zorder=1)
    ax.plot(hours, up, color=BLUE, lw=0.9, zorder=5)
    ax.plot(hours, dn, color=ORANGE, lw=0.9, zorder=6)
    pre = (hours > 48.0) & (hours < BURST_DAY * 24.0)
    during = (hours > BURST_DAY * 24.0 + 4.0) & (hours < REPAIR_DAY * 24.0)
    drop = float(dn[pre].mean() - dn[during].mean())
    ax.annotate("logger 2 loses {:.1f} m H2O\nwhile the burst is open"
                .format(drop),
                xy=(0.5 * (BURST_DAY + REPAIR_DAY) * 24.0,
                    float(dn[during].mean())),
                xytext=(DAYS * 24.0 - 4.0, 33.4), fontsize=6.1,
                color=ORANGE, ha="right", va="top", linespacing=1.30,
                zorder=9, arrowprops=dict(arrowstyle="-", color=ORANGE,
                                          lw=0.7, shrinkA=2, shrinkB=3))
    ax.text(3.0, 48.4, "logger 1, upstream", fontsize=6.1, color=BLUE,
            ha="left", va="top", zorder=8)
    ax.text(3.0, 29.4, "logger 2, inside the burst zone", fontsize=6.1,
            color=ORANGE, ha="left", va="bottom", zorder=8)
    ax.set_xlim(0.0, DAYS * 24.0)
    ax.set_ylim(28.0, 49.0)
    ax.set_xticks(np.arange(0.0, DAYS * 24.0 + 1.0, 24.0))
    ax.set_xticklabels(["d{:d}".format(int(h // 24) + 1)
                        for h in np.arange(0.0, DAYS * 24.0 + 1.0, 24.0)],
                       fontsize=6.2)
    ax.set_xlabel("logging day", fontsize=6.8)
    ax.set_ylabel("pressure head (m H2O)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Pressure signature that locates the burst", fontsize=8.0,
                 fontweight="bold", color=INK)
    return drop


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
    hours, flow, use, burst = inflow_series(rng)
    mnf, when = night_flows(hours, flow)
    leak = mnf - NIGHT_USE
    assert np.all(leak > 0.0), "night flow must exceed legitimate night use"

    # Every headline number is an integral of the series that is drawn.
    total_in = float(np.trapezoid(flow, hours)) * 3.6            # m3 in a week
    total_use = float(np.trapezoid(use, hours)) * 3.6
    loss_share = 100.0 * (1.0 - total_use / total_in)
    daily_leak = leak * 86.4                                     # m3 per day
    burst_vol = float(np.trapezoid(burst, hours)) * 3.6

    names, props, mains, press, dma_mnf, dma_leak, ili = \
        district_table(float(mnf[0]))
    ph, up, dn = pressure_signature(rng)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_idx = axes_inches(fig, AUX_T)
    ax_sig = axes_inches(fig, AUX_B)

    main_panel(ax_main, hours, flow, use, mnf, when, leak)
    index_panel(ax_idx, names, dma_leak, ili, props)
    drop = signature_panel(ax_sig, ph, up, dn)



    panel_labels_inches(fig, [ax_main, ax_idx, ax_sig], dx=-0.62, dy=0.14)
    savefig(fig, "1004_wsn_leakage_dma.png")


if __name__ == "__main__":
    main()
