"""Obstructive apnoea excerpt on four channels, plus the whole-night burden.

The main panel stacks the four signals a scorer reads together on one time
axis, so that the causal order is visible rather than asserted: airflow stops
first, thoracic and abdominal effort continue and go paradoxical - which is
what makes the event obstructive rather than central - the oxygen saturation
only starts to fall once the circulation has carried the deoxygenated blood to
the finger probe, and the heart rate surges as the arousal terminates the
event.  The lag between airflow cessation and the saturation nadir is measured
from the plotted traces, not asserted.  The two smaller panels place the
excerpt in context: the apnoea-hypopnoea index split by stage and body
position, and the saturation distribution of the whole night.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten, darken
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"

FS = 25.0                       # Hz, detail window
WIN_S = 300.0                   # 5-min excerpt
WIN_START_H = 2.24              # clock hour where the excerpt begins
RESP_HZ = 0.25                  # 15 breaths per minute
SPO2_BASE = 96.4
DESAT_GAIN = 8.5                # percent saturation per unit kernel response
TST_H = 6.9                     # total sleep time of the simulated night

EVENTS = [(38.0, 61.0), (104.0, 130.0), (181.0, 204.0), (243.0, 268.0)]
STAGES = ["N1", "N2", "N3", "REM"]
POSITIONS = ["supine", "lateral"]
STAGE_H = {"N1": 0.55, "N2": 3.45, "N3": 1.15, "REM": 1.75}
POS_SHARE = {"supine": 0.58, "lateral": 0.42}
RATE = {("N1", "supine"): 30.0, ("N2", "supine"): 33.0,
        ("N3", "supine"): 11.0, ("REM", "supine"): 62.0,
        ("N1", "lateral"): 11.0, ("N2", "lateral"): 13.0,
        ("N3", "lateral"): 4.0, ("REM", "lateral"): 24.0}


def desat_kernel(dt, lag=7.0, tau=7.5, span=80.0):
    """Delayed, asymmetric impulse response from airway closure to the probe."""
    x = np.arange(0.0, span, dt)
    y = np.clip(x - lag, 0.0, None) ** 1.7 * np.exp(-np.clip(x - lag, 0.0, None)
                                                    / tau)
    return y / (y.sum() * dt)


def window_signals(seed=13):
    rng = np.random.default_rng(seed)
    dt = 1.0 / FS
    t = np.arange(0.0, WIN_S, dt)
    occl = np.zeros_like(t)
    for s, e in EVENTS:
        occl[(t >= s) & (t < e)] = 1.0

    phase = 2.0 * np.pi * RESP_HZ * t
    flow_env = np.ones_like(t)
    eff_env = np.ones_like(t)
    para = np.zeros_like(t)
    for s, e in EVENTS:
        m = (t >= s) & (t < e)
        flow_env[m] = 0.06
        eff_env[m] = np.linspace(1.05, 1.75, int(m.sum()))   # crescendo
        para[m] = 1.0
        r = (t >= e) & (t < e + 11.0)
        flow_env[r] = 1.70                                    # hyperpnoea
        eff_env[r] = 1.55
    smooth = np.array([0.25, 0.5, 0.25])
    for _ in range(6):
        flow_env = np.convolve(flow_env, smooth, mode="same")
        eff_env = np.convolve(eff_env, smooth, mode="same")

    flow = flow_env * np.sin(phase) + rng.normal(0.0, 0.02, t.size)
    thor = eff_env * np.sin(phase + 0.12) + rng.normal(0.0, 0.015, t.size)
    abdo = eff_env * np.sin(phase + 0.12 + np.pi * para) * 0.85 \
        + rng.normal(0.0, 0.015, t.size)

    k = desat_kernel(dt)
    resp = np.convolve(occl, k, mode="full")[:t.size] * dt
    spo2 = SPO2_BASE - DESAT_GAIN * resp + rng.normal(0.0, 0.06, t.size)

    brady = np.convolve(occl, np.ones(int(6.0 * FS)) / (6.0 * FS),
                        mode="same")
    surge = np.zeros_like(t)
    for s, e in EVENTS:
        m = t >= e
        surge[m] += np.exp(-(t[m] - e) / 6.5)
    hr = (57.0 + 2.4 * np.sin(phase - 0.6) - 6.5 * brady + 17.0 * surge
          + rng.normal(0.0, 0.35, t.size))
    return t, flow, thor, abdo, spo2, hr


def night(seed=23):
    """Per-cell event counts, the whole-night saturation trace and T90."""
    rng = np.random.default_rng(seed)
    hours, counts = {}, {}
    for st in STAGES:
        for po in POSITIONS:
            h = STAGE_H[st] * POS_SHARE[po]
            hours[(st, po)] = h
            counts[(st, po)] = int(rng.poisson(RATE[(st, po)] * h))
    total = sum(counts.values())

    dt = 1.0
    t = np.arange(0.0, TST_H * 3600.0, dt)
    occl = np.zeros_like(t)
    starts = np.sort(rng.uniform(0.0, t[-1] - 60.0, total))
    for s in starts:
        d = rng.uniform(14.0, 34.0)
        occl[(t >= s) & (t < s + d)] = 1.0
    k = desat_kernel(dt)
    resp = np.convolve(occl, k, mode="full")[:t.size] * dt
    spo2 = (SPO2_BASE + 0.5 * np.sin(2.0 * np.pi * t / 5400.0)
            - DESAT_GAIN * resp + rng.normal(0.0, 0.20, t.size))
    t90 = 100.0 * float((spo2 < 90.0).mean())
    return hours, counts, total, spo2, t90


def clock(sec):
    h = WIN_START_H + sec / 3600.0
    return "%02d:%02d" % (int(h) % 24, int(round((h % 1.0) * 60.0)) % 60)


def shade(ax, first=False):
    for i, (s, e) in enumerate(EVENTS):
        ax.axvspan(s, e, color=RED, alpha=0.12, lw=0.0, zorder=0)
        ax.axvline(s, color=RED, lw=0.6, ls=(0, (2.4, 2.0)), alpha=0.55,
                   zorder=1)
        ax.axvline(e, color=RED, lw=0.6, ls=(0, (2.4, 2.0)), alpha=0.55,
                   zorder=1)


def channel_axes(ax, ylabel, last=False):
    shade(ax)
    ax.set_xlim(0.0, WIN_S)
    ax.set_ylabel(ylabel, fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    ticks = np.arange(0.0, WIN_S + 1.0, 60.0)
    ax.set_xticks(ticks)
    if last:
        ax.set_xticklabels([clock(s) for s in ticks], fontsize=7.0)
        ax.set_xlabel("clock time (h:min)")
    else:
        ax.set_xticklabels([])


def main_panels(axes, t, flow, thor, abdo, spo2, hr):
    ax_f, ax_e, ax_s, ax_h = axes

    ax_f.plot(t, flow, color=RULE, lw=0.55)
    ax_f.set_ylim(-2.4, 2.9)
    channel_axes(ax_f, "nasal flow (L/s)")
    ax_f.text(0.5 * (EVENTS[0][0] + EVENTS[0][1]), 2.35, "obstructive apnoea",
              ha="center", va="bottom", fontsize=6.8, color=RED)
    ax_f.annotate("post-event hyperpnoea", xy=(EVENTS[2][1] + 4.0, 1.75),
                  xytext=(EVENTS[2][1] + 26.0, 2.45), ha="left", va="top",
                  fontsize=6.6, color=SUB,
                  arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7))
    ax_f.set_title("Airflow stops first, oxygen follows", fontsize=9.0,
                   fontweight="bold", color=INK)

    ax_e.plot(t, thor, color=BLUE, lw=0.55, label="thoracic")
    ax_e.plot(t, abdo, color=PURPLE, lw=0.55, label="abdominal")
    ax_e.set_ylim(-2.6, 3.1)
    channel_axes(ax_e, "excursion (mm)")
    ax_e.text(0.5 * (EVENTS[1][0] + EVENTS[1][1]), 2.55,
              "effort persists and goes paradoxical", ha="center",
              va="bottom", fontsize=6.6, color=BLUE)
    ax_e.legend(fontsize=6.4, frameon=False, loc="lower right", ncol=2,
                handlelength=1.2, borderaxespad=0.15, columnspacing=1.0)

    ax_s.plot(t, spo2, color=GREEN, lw=1.0)
    ax_s.axhline(90.0, color="#808080", lw=0.8, ls=(0, (3, 2.2)))
    ax_s.text(3.0, 90.25, "90 %", ha="left", va="bottom", fontsize=6.2,
              color=MUTED)
    lo, hi = spo2.min(), SPO2_BASE
    ax_s.set_ylim(lo - 1.6, hi + 1.6)
    channel_axes(ax_s, r"$SpO_2$ (%)")

    s0, e0 = EVENTS[1]
    seg = (t >= s0) & (t <= e0 + 60.0)
    i = int(np.flatnonzero(seg)[np.argmin(spo2[seg])])
    lag = t[i] - s0
    drop = SPO2_BASE - spo2[i]
    ax_s.annotate("", xy=(t[i], spo2[i] - 0.25), xytext=(s0, spo2[i] - 0.25),
                  arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8,
                                  shrinkA=0, shrinkB=0))
    ax_s.text(0.5 * (s0 + t[i]), spo2[i] - 0.50,
              "nadir lags airway closure by %.0f s" % lag, ha="center",
              va="top", fontsize=6.6, color=INK)
    ax_s.annotate(r"$\Delta SpO_2$ = %.1f %%" % (-drop),
                  xy=(t[i], spo2[i]), xytext=(t[i] + 22.0, spo2[i] + 1.0),
                  ha="left", va="bottom", fontsize=6.6, color=GREEN,
                  arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.7))

    ax_h.plot(t, hr, color=GOLD, lw=0.9)
    ax_h.set_ylim(hr.min() - 4.0, hr.max() + 8.0)
    channel_axes(ax_h, "heart rate (bpm)", last=True)
    for s, e in EVENTS:
        j = int(np.argmin(np.abs(t - (e + 3.0))))
        ax_h.annotate("", xy=(t[j], hr[j] + 1.0),
                      xytext=(t[j], hr[j] + 7.0),
                      arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8,
                                      mutation_scale=7, shrinkA=0, shrinkB=0))
    j0 = int(np.argmin(np.abs(t - (EVENTS[0][1] + 3.0))))
    rise = hr[j0] - np.median(hr[(t > EVENTS[0][0]) & (t < EVENTS[0][1])])
    ax_h.text(t[j0] + 5.0, hr[j0] + 7.0, "arousal, +%.0f bpm" % rise,
              ha="left", va="top", fontsize=6.6, color=RED)
    return lag, drop


def ahi_panel(ax, hours, counts, total):
    x = np.arange(len(STAGES), dtype=float)
    w = 0.36
    for k, (po, col) in enumerate(zip(POSITIONS, [RED, BLUE])):
        v = [counts[(st, po)] / hours[(st, po)] for st in STAGES]
        ax.bar(x + (k - 0.5) * w, v, width=w, color=col, edgecolor="white",
               linewidth=0.6, label=po, zorder=3)
        for xi, vi in zip(x + (k - 0.5) * w, v):
            ax.text(xi, vi + 1.6, "%.0f" % vi, ha="center", va="bottom",
                    fontsize=6.0, color=col)
    ahi = total / sum(hours.values())
    ax.axhline(ahi, color="#808080", lw=0.9, ls=(0, (3, 2.2)), zorder=4)
    ax.text(len(STAGES) - 0.55, 94.0,
            "dashed: night AHI %.1f events/h" % ahi, ha="right", va="top",
            fontsize=6.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=6.8)
    ax.set_ylim(0.0, 98.0)
    ax.set_xlabel("sleep stage (-)", fontsize=7.4)
    ax.set_ylabel("AHI (events/h)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left", ncol=2,
              handlelength=1.2, borderaxespad=0.2, columnspacing=1.0)
    ax.set_title("Event rate by stage and body position", fontsize=8.2,
                 fontweight="bold", color=INK)
    return ahi


def spo2_panel(ax, spo2, t90):
    bins = np.arange(80.0, 100.5, 0.5)
    ax.hist(spo2, bins=bins, color=lighten(GREEN, 0.42), edgecolor="white",
            linewidth=0.4, zorder=3)
    ax.hist(spo2[spo2 < 90.0], bins=bins, color=RED, edgecolor="white",
            linewidth=0.4, zorder=4)
    ax.axvline(90.0, color=RULE, lw=0.9, ls=(0, (3, 2.2)), zorder=5)
    med = float(np.median(spo2))
    ax.axvline(med, color=darken(GREEN, 0.25), lw=0.9, zorder=5)
    ax.text(med - 0.4, 0.96, "median %.1f %%" % med, transform=
            ax.get_xaxis_transform(), ha="right", va="top", fontsize=6.4,
            color=darken(GREEN, 0.25))
    ax.text(89.6, 0.62, "T90 = %.1f %% of sleep" % t90,
            transform=ax.get_xaxis_transform(), ha="right", va="center",
            fontsize=6.6, color=RED)
    ax.set_xlim(82.0, 100.0)
    ax.set_xlabel(r"$SpO_2$ (%)", fontsize=7.4)
    ax.set_ylabel("time (s per 0.5 % bin)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Whole-night saturation distribution", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    t, flow, thor, abdo, spo2, hr = window_signals()
    hours, counts, total, night_spo2, t90 = night()

    fig = plt.figure(figsize=(10.4, 6.4))
    h = 1.0875
    chans = [axes_inches(fig, (0.86, 1.02 + h * (3 - k), 4.96, h))
             for k in range(4)]
    ax_top = axes_inches(fig, (6.92, 3.505, 2.90, 1.865))
    ax_bot = axes_inches(fig, (6.92, 0.98, 2.90, 1.865))

    lag, drop = main_panels(chans, t, flow, thor, abdo, spo2, hr)
    ahi = ahi_panel(ax_top, hours, counts, total)
    spo2_panel(ax_bot, night_spo2, t90)

    note = ("Synthetic polysomnography; the excerpt is a %.0f s window "
            "beginning at %s, sampled at %.0f Hz, and the summary panels cover "
            "a %.1f h simulated night. Red bands mark scored obstructive "
            "apnoeas. The saturation drop, the nadir lag, the tachycardia "
            "rise, the per-cell AHI and T90 are all measured from the plotted "
            "arrays." % (WIN_S, clock(0.0), FS, TST_H))
    panel_labels_inches(fig, [chans[0], ax_top, ax_bot], dx=-0.56, dy=0.13)
    savefig(fig, "790_slp_apnea_events.png")


if __name__ == "__main__":
    main()
