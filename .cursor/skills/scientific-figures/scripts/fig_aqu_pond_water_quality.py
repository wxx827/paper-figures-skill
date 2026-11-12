"""Ten days of pond water quality, with the pre-dawn oxygen sag it turns on.

Dissolved oxygen is integrated rather than drawn: dDO/dt = P(t) - R +
k (DO_sat - DO), with P driven by daylight and DO_sat taken from the
temperature trace on the row below, so the oxygen maximum lands in the
afternoon and the minimum at dawn because the model says so, not because a
sine wave was phased that way.  The aerator is a closed loop on that same
integration - it starts below 3.8 mg/L and stops above 5.0 - so the shaded
run-time ribbon is an output of the simulation and always lines up with the
sag that triggered it.  Where the loop still loses, DO drops under the
3.0 mg/L line and the gap between curve and threshold is filled in red.

pH tracks the same photosynthetic CO2 drawdown, and total ammonia nitrogen is
a second integration fed by two daily feeds and cleared by uptake and
nitrification, which is why it peaks overnight in antiphase with DO.  A slow
seasonal cooling runs underneath the whole window.

Panel (b) folds all ten days onto one 24 h clock - median with a 10-90
percentile envelope - and panel (c) counts, from the plotted arrays, how long
each parameter sat outside its suitable range.  Illustrative values; this is a
layout reference.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GREY = "#808080"
DO_C, TEMP_C, PH_C, TAN_C = "#2a7f7a", "#c0603f", "#8e6bbf", "#c9a227"
RISK, AER = "#b23a48", "#3b5b92"

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.04, 4.86, 4.16)
AUX1 = (6.62, 3.14, 2.78, 2.06)
AUX2 = (6.62, 1.04, 2.78, 1.44)

N_DAYS = 10
STEPS = 144                       # 10-minute sampling
DT = 1.0 / STEPS
DO_MIN_OK, TAN_MAX_OK = 3.0, 1.00
T_OK = (24.0, 31.0)
PH_OK = (6.5, 9.0)
AER_ON, AER_OFF = 3.3, 5.0


def build_pond(rng):
    """Integrate DO and ammonia over a ten-day window with a live aerator."""
    t = np.arange(0.0, N_DAYS, DT)
    n = t.size
    hour = (t % 1.0) * 24.0
    day = np.floor(t).astype(int)

    temp = (30.6 - 0.34 * t + 2.6 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
            + rng.normal(0.0, 0.06, n))
    do_sat = 468.0 / (31.6 + temp)
    light = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    cloud = 0.74 + 0.34 * rng.random(N_DAYS)
    p_max = 44.0 * cloud[day] * (1.0 - 0.012 * t)
    resp = 19.2
    k_ex = 2.0

    do = np.empty(n)
    on = np.zeros(n, dtype=bool)
    do[0] = 6.4
    running = False
    for i in range(n - 1):
        if do[i] < AER_ON:
            running = True
        elif do[i] > AER_OFF:
            running = False
        on[i] = running
        aer = 1.6 * (do_sat[i] - do[i]) if running else 0.0
        rate = (p_max[i] * light[i] - resp + k_ex * (do_sat[i] - do[i]) + aer)
        do[i + 1] = max(do[i] + rate * DT + rng.normal(0.0, 0.035), 0.30)
    on[-1] = running

    span = do.max() - do.min()
    ph = 7.48 + 1.70 * (do - do.min()) / span + rng.normal(0.0, 0.026, n)

    feed = (8.5 * np.exp(-0.5 * ((hour - 8.0) / 0.65) ** 2)
            + 6.4 * np.exp(-0.5 * ((hour - 17.0) / 0.65) ** 2))
    tan = np.empty(n)
    tan[0] = 0.80
    for i in range(n - 1):
        rate = feed[i] - (3.1 * light[i] + 0.50) * tan[i]
        tan[i + 1] = max(tan[i] + rate * DT + rng.normal(0.0, 0.004), 0.02)

    return dict(t=t, hour=hour, day=day, temp=temp, do=do, do_sat=do_sat,
                ph=ph, tan=tan, on=on)


def stacked_rows(fig, n_rows=4, gap=0.055):
    x0, y0, w, h = MAIN
    hh = (h - gap * (n_rows - 1)) / n_rows
    return [axes_inches(fig, (x0, y0 + (n_rows - 1 - i) * (hh + gap), w, hh))
            for i in range(n_rows)]


PRE_DAWN = (2.0, 6.0)


def shade_aerator(ax, d, lo, hi):
    """Aerator run-time as a baseline ribbon, so it never dims the traces."""
    band = lo + 0.055 * (hi - lo)
    ax.fill_between(d["t"], lo, band, where=d["on"], step="mid",
                    color=lighten(AER, 0.42), edgecolor="none", zorder=2)


def shade_pre_dawn(ax, d, lo, hi):
    pre = (d["hour"] >= PRE_DAWN[0]) & (d["hour"] < PRE_DAWN[1])
    ax.fill_between(d["t"], lo, hi, where=pre, step="mid",
                    color=lighten(RISK, 0.90), edgecolor="none", zorder=0)
    return pre


def draw_rows(axs, d):
    ax_do, ax_t, ax_ph, ax_n = axs

    lo, hi = 0.0, max(d["do"].max(), d["do_sat"].max()) * 1.34
    pre = shade_pre_dawn(ax_do, d, lo, hi)
    shade_aerator(ax_do, d, lo, hi)
    ax_do.plot(d["t"], d["do_sat"], color=GREY, lw=0.7, ls=(0, (3.0, 2.0)),
               zorder=3)
    ax_do.axhline(DO_MIN_OK, color=RISK, lw=0.8, ls=(0, (3.2, 2.0)), zorder=4)
    ax_do.fill_between(d["t"], d["do"], DO_MIN_OK, where=d["do"] < DO_MIN_OK,
                       color=RISK, edgecolor="none", alpha=0.85, zorder=5)
    ax_do.plot(d["t"], d["do"], color=DO_C, lw=1.05, zorder=6)
    ax_do.set_ylim(lo, hi)
    ax_do.set_ylabel("DO\n(mg/L)", fontsize=7.4, color=darken(DO_C, 0.12))
    ax_do.text(N_DAYS - 0.06, DO_MIN_OK + 0.62,
               "risk threshold {:.1f} mg/L".format(DO_MIN_OK), fontsize=6.1,
               color=RISK, ha="right", va="bottom", zorder=8)

    handles = [Patch(facecolor=lighten(RISK, 0.90), edgecolor="none",
                     label="pre-dawn risk window {:02d}:00-{:02d}:00"
                     .format(int(PRE_DAWN[0]), int(PRE_DAWN[1]))),
               Patch(facecolor=RISK, alpha=0.85, edgecolor="none",
                     label="DO under {:.0f} mg/L".format(DO_MIN_OK)),
               Patch(facecolor=lighten(AER, 0.42), edgecolor="none",
                     label="aerator running")]
    leg = ax_do.legend(handles=handles, loc="upper left", fontsize=6,
                       frameon=True, framealpha=0.93, borderpad=0.34,
                       handlelength=1.4, labelspacing=0.26, ncol=3,
                       columnspacing=0.9, handletextpad=0.42)
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(12)

    lo, hi = d["temp"].min() - 1.4, d["temp"].max() + 2.4
    shade_pre_dawn(ax_t, d, lo, hi)
    shade_aerator(ax_t, d, lo, hi)
    ax_t.axhline(T_OK[1], color=GREY, lw=0.7, ls=(0, (3.2, 2.0)), zorder=3)
    ax_t.text(N_DAYS - 0.06, T_OK[1] + 0.12,
              "suitable max {:.0f} $^{{\\circ}}$C".format(T_OK[1]),
              fontsize=6.1, color=MUTED, ha="right", va="bottom", zorder=8)
    ax_t.plot(d["t"], d["temp"], color=TEMP_C, lw=1.05, zorder=6)
    ax_t.set_ylim(lo, hi)
    ax_t.set_ylabel("temp\n($^{\\circ}$C)", fontsize=7.4,
                    color=darken(TEMP_C, 0.12))
    ax_t.text(0.004, 0.96, "water temperature: diel swing on a seasonal "
              "cooling of {:.1f} $^{{\\circ}}$C over the window"
              .format(d["temp"][:STEPS].mean() - d["temp"][-STEPS:].mean()),
              transform=ax_t.transAxes, fontsize=6.4, color=SUB, ha="left",
              va="top", zorder=8)

    lo, hi = d["ph"].min() - 0.22, d["ph"].max() + 0.42
    shade_pre_dawn(ax_ph, d, lo, hi)
    shade_aerator(ax_ph, d, lo, hi)
    ax_ph.axhline(PH_OK[1], color=GREY, lw=0.7, ls=(0, (3.2, 2.0)), zorder=3)
    ax_ph.text(N_DAYS - 0.06, PH_OK[1] + 0.03,
               "suitable max {:.1f}".format(PH_OK[1]), fontsize=6.1,
               color=MUTED, ha="right", va="bottom", zorder=8)
    ax_ph.plot(d["t"], d["ph"], color=PH_C, lw=1.05, zorder=6)
    ax_ph.set_ylim(lo, hi)
    ax_ph.set_ylabel("pH\n(-)", fontsize=7.4, color=darken(PH_C, 0.12))
    ax_ph.text(0.004, 0.96, "pH: CO$_2$ drawdown pushes it up with the "
               "oxygen peak", transform=ax_ph.transAxes, fontsize=6.4,
               color=SUB, ha="left", va="top", zorder=8)

    lo, hi = 0.0, d["tan"].max() * 1.46
    shade_pre_dawn(ax_n, d, lo, hi)
    shade_aerator(ax_n, d, lo, hi)
    ax_n.axhline(TAN_MAX_OK, color=GREY, lw=0.7, ls=(0, (3.2, 2.0)), zorder=3)
    ax_n.text(N_DAYS - 0.06, TAN_MAX_OK + 0.02,
              "suitable max {:.1f} mg/L".format(TAN_MAX_OK), fontsize=6.1,
              color=MUTED, ha="right", va="bottom", zorder=8)
    ax_n.plot(d["t"], d["tan"], color=TAN_C, lw=1.05, zorder=6)
    ax_n.set_ylim(lo, hi)
    ax_n.set_ylabel("TAN\n(mg/L)", fontsize=7.4, color=darken(TAN_C, 0.20))
    ax_n.text(0.004, 0.96, "total ammonia nitrogen: two feeds a day, cleared "
              "in daylight, peaks overnight", transform=ax_n.transAxes,
              fontsize=6.4, color=SUB, ha="left", va="top", zorder=8)

    for ax in axs:
        ax.set_xlim(0.0, N_DAYS)
        ax.set_xticks(np.arange(0, N_DAYS + 1, 1))
        ax.tick_params(labelsize=6.6)
        ax.grid(axis="x", color="#ededed", lw=0.5, zorder=1)
    for ax in axs[:-1]:
        ax.set_xticklabels([])
    axs[-1].set_xlabel("time (days from start of the monitoring window)",
                       fontsize=7.8)
    axs[0].set_title("Water quality on a shared clock: DO peaks in daylight "
                     "and bottoms before dawn", fontsize=8.5,
                     fontweight="bold")


def draw_envelope(ax, d):
    grid = d["do"].reshape(N_DAYS, STEPS)
    clock = np.arange(STEPS) * 24.0 / STEPS
    med = np.median(grid, axis=0)
    p10 = np.percentile(grid, 10, axis=0)
    p90 = np.percentile(grid, 90, axis=0)

    ax.axhspan(0.0, DO_MIN_OK, color=lighten(RISK, 0.88), zorder=0)
    ax.fill_between(clock, p10, p90, color=lighten(DO_C, 0.80),
                    edgecolor="none", zorder=3)
    ax.plot(clock, med, color=DO_C, lw=1.6, zorder=6)
    ax.axhline(DO_MIN_OK, color=RISK, lw=0.8, ls=(0, (3.2, 2.0)), zorder=4)

    i_lo, i_hi = int(np.argmin(med)), int(np.argmax(med))
    for i, tag, col in ((i_lo, "minimum", RISK), (i_hi, "maximum", DO_C)):
        ax.plot([clock[i]], [med[i]], marker="o", ms=4.6,
                color=darken(col, 0.12), markeredgecolor="white",
                markeredgewidth=0.7, zorder=8)
        ax.annotate("{:s} {:.1f} mg/L\nat {:02d}:{:02d}"
                    .format(tag, med[i], int(clock[i]),
                            int(round((clock[i] % 1.0) * 60.0))),
                    xy=(clock[i], med[i]),
                    xytext=(0, 13 if tag == "maximum" else -15),
                    textcoords="offset points", fontsize=6.2,
                    color=darken(col, 0.18), ha="center",
                    va="bottom" if tag == "maximum" else "top", zorder=9,
                    linespacing=1.30)

    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(0.0, p90.max() * 1.30)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("hour of day (h)", fontsize=7.6)
    ax.set_ylabel("dissolved oxygen (mg/L)", fontsize=7.6)
    ax.tick_params(labelsize=6.6)
    ax.set_title("Diel DO envelope, ten days folded on one clock",
                 fontsize=8.5, fontweight="bold")
    return med, clock[i_lo], clock[i_hi]


def draw_exceedance(ax, d):
    hours = DT * 24.0
    items = [("DO < {:.0f} mg/L".format(DO_MIN_OK), d["do"] < DO_MIN_OK, DO_C),
             ("temp > {:.0f} $^{{\\circ}}$C".format(T_OK[1]),
              d["temp"] > T_OK[1], TEMP_C),
             ("pH > {:.1f}".format(PH_OK[1]), d["ph"] > PH_OK[1], PH_C),
             ("TAN > {:.1f} mg/L".format(TAN_MAX_OK),
              d["tan"] > TAN_MAX_OK, TAN_C)]
    names = [it[0] for it in items]
    hrs = np.array([float(it[1].sum()) * hours for it in items])
    pct = hrs / (N_DAYS * 24.0) * 100.0
    y = np.arange(len(items))
    ax.barh(y, hrs, height=0.60, color=[lighten(it[2], 0.32) for it in items],
            edgecolor="white", linewidth=0.5, zorder=4)
    for yi, h, p in zip(y, hrs, pct):
        ax.text(h + hrs.max() * 0.022, yi, "{:.1f} h  ({:.1f}%)".format(h, p),
                fontsize=6.3, color=SUB, ha="left", va="center", zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(0.0, hrs.max() * 1.62)
    ax.set_xlabel("time outside the suitable range (h in {:d} days)"
                  .format(N_DAYS), fontsize=7.6)
    ax.tick_params(labelsize=6.6)
    ax.set_title("Exceedance duration by parameter", fontsize=8.5,
                 fontweight="bold")
    return hrs, pct


def main():
    set_pub_style()
    rng = np.random.default_rng(9040)
    d = build_pond(rng)

    fig = plt.figure(figsize=(FW, FH))
    rows = stacked_rows(fig)
    ax_e = axes_inches(fig, AUX1)
    ax_x = axes_inches(fig, AUX2)

    draw_rows(rows, d)
    med, h_lo, h_hi = draw_envelope(ax_e, d)
    hrs, pct = draw_exceedance(ax_x, d)

    panel_labels_inches(fig, [rows[0], ax_e, ax_x], dx=-0.46, dy=0.20)

    duty = float(d["on"].mean())

    print("do_min={:.2f} do_max={:.2f} duty={:.3f} hrs={:s}"
          .format(d["do"].min(), d["do"].max(), duty,
                  " ".join("{:.1f}".format(v) for v in hrs)))
    savefig(fig, "904_aqu_pond_water_quality.png")


if __name__ == "__main__":
    main()
