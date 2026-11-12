"""Onset envelope with beat tracking, tempo curve, IOI modes and stability.

One event list drives the whole figure.  Beat times are produced by
integrating an instantaneous tempo that holds near 120 BPM and then slows
into a ritardando; eighth-note and occasional triplet subdivisions are added
between the beats, and the onset envelope is the sum of the resulting
exponentially decaying pulses.  The tempo curve is 60 / (beat interval), the
inter-onset-interval histogram is a first difference of the same event list,
and the stability panel is a rolling coefficient of variation of the beat
intervals, so all four panels move together when the seed changes.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#cccccc"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"

BPM0 = 120.0
BPM_END = 78.0
T_END = 16.0
T_RIT = 11.0
BEATS_PER_BAR = 4
FS = 500.0          # envelope sample rate, Hz
TAU = 0.055         # pulse decay constant, s


def tempo_at(t):
    """Instantaneous tempo in BPM: steady with drift, then a ritardando."""
    steady = BPM0 * (1.0 + 0.012 * np.sin(2.0 * np.pi * t / 6.5))
    if t <= T_RIT:
        return steady
    f = min((t - T_RIT) / (T_END - T_RIT), 1.0)
    return steady + (BPM_END - steady) * f ** 1.3


def beat_times(rng):
    beats, t = [], 0.0
    while t < T_END:
        beats.append(t)
        t += 60.0 / tempo_at(t) * (1.0 + 0.012 * rng.standard_normal())
    return np.array(beats)


def onset_events(beats, rng):
    """Beat onsets plus eighth and occasional triplet subdivisions."""
    times, amps, kinds = [], [], []
    for i, b in enumerate(beats):
        down = i % BEATS_PER_BAR == 0
        times.append(b)
        amps.append((1.0 if down else 0.72) * (0.9 + 0.2 * rng.random()))
        kinds.append("downbeat" if down else "beat")
        if i + 1 >= beats.size:
            continue
        step = beats[i + 1] - b
        u = rng.random()
        if u < 0.18:
            for f in (1.0 / 3.0, 2.0 / 3.0):
                times.append(b + f * step)
                amps.append(0.34 * (0.9 + 0.2 * rng.random()))
                kinds.append("triplet")
        elif u < 0.62:
            times.append(b + 0.5 * step)
            amps.append(0.46 * (0.9 + 0.2 * rng.random()))
            kinds.append("eighth")
        # the remaining beats carry no subdivision, which is what leaves a
        # quarter-note mode in the inter-onset-interval histogram
    order = np.argsort(times)
    return (np.array(times)[order], np.array(amps)[order],
            np.array(kinds)[order])


def envelope(times, amps, rng):
    t = np.arange(0.0, T_END, 1.0 / FS)
    env = 0.045 * np.abs(rng.standard_normal(t.size))
    for t0, a in zip(times, amps):
        m = t >= t0
        env[m] += a * np.exp(-(t[m] - t0) / TAU)
    k = np.ones(5) / 5.0
    return t, np.convolve(env, k, mode="same")


def rolling_cv(beats, win=8):
    ioi = np.diff(beats)
    mid = 0.5 * (beats[:-1] + beats[1:])
    cv, tc = [], []
    for i in range(ioi.size - win + 1):
        w = ioi[i:i + win]
        cv.append(100.0 * w.std(ddof=1) / w.mean())
        tc.append(mid[i + win // 2])
    return np.array(tc), np.array(cv)


def main():
    set_pub_style()
    rng = np.random.default_rng(849)
    beats = beat_times(rng)
    times, amps, kinds = onset_events(beats, rng)
    t, env = envelope(times, amps, rng)
    env = env / env.max()

    beat_ioi = np.diff(beats)
    beat_bpm = 60.0 / beat_ioi
    beat_mid = 0.5 * (beats[:-1] + beats[1:])
    all_ioi = np.diff(times)
    t_cv, cv = rolling_cv(beats)
    downbeats = beats[::BEATS_PER_BAR]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_env = axes_inches(fig, (0.80, 2.55, 5.22, 3.05))
    ax_bpm = axes_inches(fig, (0.80, 0.92, 5.22, 1.35))
    ax_ioi = axes_inches(fig, (7.05, 3.30, 3.05, 2.30))
    ax_cv = axes_inches(fig, (7.05, 0.92, 3.05, 1.90))

    # ---- onset envelope with beat grid ------------------------------------
    ax_env.axvspan(T_RIT, T_END, facecolor="#f5f5f5", edgecolor="none",
                   zorder=0)
    ax_env.fill_between(t, 0.0, env, color=BLUE, alpha=0.22, lw=0.0, zorder=1)
    ax_env.plot(t, env, color=BLUE, lw=0.7, zorder=2)
    for i, b in enumerate(beats):
        down = i % BEATS_PER_BAR == 0
        ax_env.axvline(b, color=RED if down else "#9a9a9a",
                       lw=1.3 if down else 0.6,
                       ls="-" if down else (0, (2, 2)), zorder=3,
                       alpha=1.0 if down else 0.85)
    for bar, b in enumerate(downbeats):
        ax_env.text(b, 1.10, "bar %d" % (bar + 1), ha="left", va="bottom",
                    fontsize=6.2, color=SUB)
    ax_env.text(0.5 * (T_RIT + T_END), 0.97, "ritardando", ha="center",
                va="center", fontsize=6.8, color=GOLD)
    ax_env.set_xlim(0.0, T_END)
    ax_env.set_ylim(0.0, 1.30)
    ax_env.set_ylabel("onset strength (a.u., peak-normalised)", fontsize=8.0)
    ax_env.tick_params(labelsize=7.0, labelbottom=False)
    for s in ("top", "right"):
        ax_env.spines[s].set_visible(False)
    ax_env.set_title("Onset envelope with tracked beats, %d beats in %d bars"
                     % (beats.size, downbeats.size), fontsize=9.0,
                     fontweight="bold", pad=4.0)

    # ---- tempo curve -------------------------------------------------------
    ax_bpm.axvspan(T_RIT, T_END, facecolor="#f5f5f5", edgecolor="none",
                   zorder=0)
    ax_bpm.step(beat_mid, beat_bpm, where="mid", color=GREEN, lw=1.0,
                zorder=2)
    ax_bpm.plot(beat_mid, beat_bpm, ls="none", marker="o", ms=2.4,
                color=GREEN, zorder=3)
    steady = beat_bpm[beat_mid <= T_RIT]
    ax_bpm.axhline(steady.mean(), color="#808080", lw=0.8, ls=(0, (4, 3)),
                   zorder=1)
    ax_bpm.annotate("%.1f BPM median before the ritardando"
                    % np.median(steady), xy=(0.4, steady.mean()),
                    xytext=(0.4, steady.mean() + 9.0), fontsize=6.5,
                    color=SUB)
    ax_bpm.annotate("-%.0f BPM over %.0f s"
                    % (beat_bpm[beat_mid <= T_RIT][-1] - beat_bpm[-1],
                       T_END - T_RIT),
                    xy=(beat_mid[-1], beat_bpm[-1]),
                    xytext=(T_RIT - 2.9, beat_bpm[-1] - 4.0), fontsize=6.5,
                    color=INK, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=MUTED))
    ax_bpm.set_xlim(0.0, T_END)
    ax_bpm.set_ylim(beat_bpm.min() - 12.0, beat_bpm.max() + 14.0)
    ax_bpm.set_xlabel("time (s)", fontsize=8.0)
    ax_bpm.set_ylabel("tempo (BPM)", fontsize=8.0)
    ax_bpm.tick_params(labelsize=7.0)
    for s in ("top", "right"):
        ax_bpm.spines[s].set_visible(False)

    # ---- inter-onset-interval histogram ------------------------------------
    bins = np.arange(0.0, 0.92, 0.012)
    counts, edges = np.histogram(all_ioi, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax_ioi.bar(centres, counts, width=0.011, color=BLUE, alpha=0.85,
               edgecolor="none")
    pk, _ = find_peaks(counts, height=max(2, 0.10 * counts.max()), distance=4)
    order = np.argsort(counts[pk])[::-1][:3]
    beat_med = float(np.median(beat_ioi))
    for j in sorted(pk[order]):
        ax_ioi.annotate("%.0f ms\n%.2f x beat" % (centres[j] * 1e3,
                                                  centres[j] / beat_med),
                        xy=(centres[j], counts[j]),
                        xytext=(centres[j], counts[j] + 0.08 * counts.max()),
                        ha="center", va="bottom", fontsize=6.2, color=SUB,
                        linespacing=1.35)
    ax_ioi.set_xlim(0.0, min(0.92, all_ioi.max() * 1.18))
    ax_ioi.set_ylim(0.0, counts.max() * 1.42)
    ax_ioi.set_xlabel("inter-onset interval (s)", fontsize=7.5)
    ax_ioi.set_ylabel("onset pair count (n=%d)" % all_ioi.size, fontsize=7.5)
    ax_ioi.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax_ioi.spines[s].set_visible(False)
    ax_ioi.set_title("IOI modes: beat and its subdivisions", fontsize=8.5,
                     fontweight="bold", pad=3.0)

    # ---- beat-interval stability -------------------------------------------
    ax_cv.axvspan(T_RIT, T_END, facecolor="#f5f5f5", edgecolor="none",
                  zorder=0)
    ax_cv.plot(t_cv, cv, color=RED, lw=1.2, zorder=2)
    ax_cv.plot(t_cv, cv, ls="none", marker="o", ms=2.2, color=RED, zorder=3)
    base = cv[t_cv <= T_RIT]
    ax_cv.axhline(base.mean(), color="#808080", lw=0.8, ls=(0, (4, 3)),
                  zorder=1)
    ax_cv.text(0.35, base.mean() + 0.35, "steady mean %.1f %%" % base.mean(),
               fontsize=6.4, color=SUB, va="bottom")
    ax_cv.set_xlim(0.0, T_END)
    ax_cv.set_ylim(0.0, cv.max() * 1.30)
    ax_cv.set_xlabel("time (s)", fontsize=7.5)
    ax_cv.set_ylabel("beat IOI CV (%), 8-beat window", fontsize=7.5)
    ax_cv.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax_cv.spines[s].set_visible(False)
    ax_cv.set_title("Beat stability drops in the ritardando", fontsize=8.5,
                    fontweight="bold", pad=3.0)

    panel_labels_inches(fig, [ax_env, ax_bpm, ax_ioi, ax_cv], dx=-0.52,
                        dy=0.12)


    savefig(fig, "849_mir_tempo_beat.png")


if __name__ == "__main__":
    main()
