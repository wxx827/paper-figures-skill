"""Borbely's two-process model run across two normal nights and one lost one.

The main panel is the model itself, integrated rather than sketched: homeostatic
sleep pressure S climbs toward its asymptote while the subject is awake and
decays exponentially while asleep, and the circadian process C supplies the two
oscillating thresholds it bounces between.  Sleep starts where S meets the
upper envelope and ends where it reaches the lower one, so the onset and wake
clock times printed on the figure are outputs of the simulation, not labels
placed by hand.  Blocking sleep for one night lets S run far above the envelope
it should have touched, and the recovery episode is correspondingly longer.
The two smaller panels show what that excess costs: subjective sleepiness and
psychomotor vigilance, both driven by the same S and C traces.
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
GOLD = "#c9a227"

START_H = 7.0            # the simulation begins at 07:00
T_END = 100.0
DT = 0.01
TAU_R = 18.2             # h, homeostatic build-up while awake
TAU_D = 4.2              # h, dissipation while asleep
S0 = 0.0934              # steady-state morning value of the same model
C_AMP = 0.060
H_UP_MEAN = 0.6844
H_LO_MEAN = 0.0634
C_PHASE = 4.0            # thresholds are lowest at 23:00
DEPRIV = (64.0, 88.0)    # sleep blocked from the third night into day four


def circadian(t):
    return np.cos(2.0 * np.pi * (np.asarray(t) - C_PHASE) / 24.0)


def h_up(t):
    return H_UP_MEAN + C_AMP * circadian(t)


def h_lo(t):
    return H_LO_MEAN + C_AMP * circadian(t)


def simulate():
    n = int(round(T_END / DT))
    t = np.arange(n) * DT
    s = np.empty(n)
    asleep = np.zeros(n, dtype=bool)
    s[0] = S0
    state = False
    for i in range(1, n):
        if state:
            if s[i - 1] <= h_lo(t[i - 1]):
                state = False
        else:
            blocked = DEPRIV[0] <= t[i - 1] < DEPRIV[1]
            if s[i - 1] >= h_up(t[i - 1]) and not blocked:
                state = True
        asleep[i - 1] = state
        if state:
            s[i] = s[i - 1] - DT * s[i - 1] / TAU_D
        else:
            s[i] = s[i - 1] + DT * (1.0 - s[i - 1]) / TAU_R
    asleep[-1] = state
    return t, s, asleep


def episodes(t, flag):
    out, i, n = [], 0, flag.size
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1]:
            j += 1
        out.append((t[i], t[j]))
        i = j + 1
    return out


def clock(t):
    h = START_H + t
    return "%02d:%02d" % (int(h) % 24, int(round((h % 1.0) * 60.0)) % 60)


def sleepiness_load(t, s):
    """Combined homeostatic and circadian drive, on one shared 0-1 scale."""
    hour = (START_H + np.asarray(t)) % 24.0
    circ = 0.5 + 0.5 * np.cos(2.0 * np.pi * (hour - 5.0) / 24.0)
    return 0.62 * (np.asarray(s) / 1.0) + 0.38 * circ


def performance(t, s, asleep, seed=29):
    """Sample the two wake episodes every two hours, eight virtual subjects."""
    rng = np.random.default_rng(seed)
    wake = episodes(t, ~asleep)
    picks = [wake[0], max(wake, key=lambda w: w[1] - w[0])]
    load_all = sleepiness_load(t, s)
    lo, hi = float(load_all.min()), float(load_all.max())
    out = {}
    names = ["baseline day", "extended wake"]
    for name, (a, b) in zip(names, picks):
        h = np.arange(0.0, (b - a) + 1e-9, 2.0)
        idx = np.searchsorted(t, a + h)
        idx = np.clip(idx, 0, t.size - 1)
        base = (load_all[idx] - lo) / (hi - lo)
        subj = base[None, :] + rng.normal(0.0, 0.045, (8, base.size))
        kss = np.clip(1.0 + 8.0 * subj, 1.0, 9.0)
        rt = 232.0 + 235.0 * np.clip(subj, 0.0, None) ** 1.6 \
            + rng.normal(0.0, 8.0, subj.shape)
        out[name] = (h, kss, rt)
    return out


def model_panel(ax, t, s, asleep):
    hu, hl = h_up(t), h_lo(t)
    ax.fill_between(t, hl, hu, color=lighten(GOLD, 0.72), linewidth=0.0,
                    zorder=1)
    for ep in episodes(t, asleep):
        ax.axvspan(ep[0], ep[1], color="#e2e2e2", lw=0.0, zorder=0)
    ax.plot(t, hu, color=GOLD, lw=1.1, ls=(0, (4, 2.4)), zorder=3)
    ax.plot(t, hl, color=darken(GOLD, 0.28), lw=1.1, ls=(0, (4, 2.4)),
            zorder=3)
    ax.plot(t, s, color=BLUE, lw=1.9, zorder=5)

    ax.axvspan(DEPRIV[0], DEPRIV[1], ymin=0.945, ymax=0.995, color=RED,
               alpha=0.22, lw=0.0, zorder=4)
    awake_h = max(b - a for a, b in episodes(t, ~asleep))
    ax.text(0.5 * (DEPRIV[0] + DEPRIV[1]), 1.164,
            "sleep deprivation, %.0f h of continuous wake" % awake_h,
            ha="center", va="center", fontsize=7.0, color=RED, zorder=6)

    eps = episodes(t, asleep)
    for k, (a, b) in enumerate(eps):
        for x in (a, b):
            ax.text(x, 0.995, clock(x), rotation=90, ha="center", va="bottom",
                    fontsize=6.2, color=SUB, zorder=6)
            ax.plot([x], [np.interp(x, t, s)], marker="o", ms=3.8, color=BLUE,
                    mfc="white", mew=1.0, zorder=7)
        ax.text(0.5 * (a + b), 0.035, "%.1f h" % (b - a), ha="center",
                va="bottom", fontsize=6.3, color=SUB, zorder=6)
    ax.text(0.5 * sum(eps[-1]), 0.135, "recovery", ha="center", va="bottom",
            fontsize=6.3, color=INK, zorder=6)

    ax.text(52.0, h_up(52.0) + 0.030, r"process C: upper threshold $H^+$",
            ha="left", va="bottom", fontsize=6.8, color=darken(GOLD, 0.18),
            zorder=6)
    ax.text(50.0, h_lo(50.0) - 0.030, r"lower threshold $H^-$", ha="left",
            va="top", fontsize=6.8, color=darken(GOLD, 0.28), zorder=6)
    ax.text(9.0, np.interp(9.0, t, s) + 0.035, "process S", ha="center",
            va="bottom", fontsize=7.2, color=BLUE, zorder=6)

    peak = float(s.max())
    ip = int(np.argmax(s))
    ax.annotate("S overshoots $H^+$ by %.2f" % (peak - h_up(t[ip])),
                xy=(t[ip], peak), xytext=(t[ip] - 6.0, peak + 0.105),
                ha="right", va="bottom", fontsize=6.8, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, 1.20)
    ticks = np.arange(0.0, T_END + 0.1, 8.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([clock(v) for v in ticks], fontsize=6.8)
    ax.set_xlabel("clock time over four days (h:min)")
    ax.set_ylabel("sleep pressure S (-)")
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Sleep starts at the upper envelope, ends at the lower",
                 fontsize=9.0, fontweight="bold", color=INK)
    return awake_h, peak


def kss_panel(ax, perf):
    for name, col in [("extended wake", RED), ("baseline day", BLUE)]:
        h, kss, _ = perf[name]
        mu, sd = kss.mean(axis=0), kss.std(axis=0)
        ax.fill_between(h, mu - sd, mu + sd, color=col, alpha=0.18,
                        linewidth=0.0, zorder=2)
        ax.plot(h, mu, color=col, lw=1.4, marker="o", ms=2.8, mfc="white",
                mew=0.8, zorder=3, label=name)
    ax.axhline(7.0, color="#808080", lw=0.8, ls=(0, (3, 2.2)), zorder=1)
    ax.text(0.4, 7.15, "sleepy ( KSS 7 )", ha="left", va="bottom",
            fontsize=6.2, color=MUTED)
    ax.set_xlim(0.0, max(h.max() for h, _, _ in perf.values()) + 0.5)
    ax.set_ylim(1.0, 9.6)
    ax.set_xlabel("time awake (h)", fontsize=7.4)
    ax.set_ylabel("KSS sleepiness (1-9)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left",
              handlelength=1.3, borderaxespad=0.2)
    ax.set_title("Subjective sleepiness", fontsize=8.2, fontweight="bold",
                 color=INK)


def pvt_panel(ax, perf):
    top = 0.0
    for name, col in [("extended wake", RED), ("baseline day", BLUE)]:
        h, _, rt = perf[name]
        mu, sd = rt.mean(axis=0), rt.std(axis=0)
        top = max(top, float((mu + sd).max()))
        ax.fill_between(h, mu - sd, mu + sd, color=col, alpha=0.18,
                        linewidth=0.0, zorder=2)
        ax.plot(h, mu, color=col, lw=1.4, marker="o", ms=2.8, mfc="white",
                mew=0.8, zorder=3, label=name)
    h_e, _, rt_e = perf["extended wake"]
    base = float(rt_e[:, np.argmin(np.abs(h_e - 8.0))].mean())
    late = float(rt_e[:, np.argmin(np.abs(h_e - 24.0))].mean())
    ax.set_ylim(200.0, top * 1.24)
    ax.annotate("+%.0f ms at 24 h awake" % (late - base),
                xy=(24.0, late), xytext=(38.0, top * 1.17), ha="right",
                va="top", fontsize=6.6, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    ax.set_xlim(0.0, max(h.max() for h, _, _ in perf.values()) + 0.5)
    ax.set_xlabel("time awake (h)", fontsize=7.4)
    ax.set_ylabel("PVT reaction time (ms)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left",
              handlelength=1.3, borderaxespad=0.2)
    ax.set_title("Psychomotor vigilance", fontsize=8.2, fontweight="bold",
                 color=INK)


def main():
    set_pub_style()
    t, s, asleep = simulate()
    perf = performance(t, s, asleep)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.84, 1.22, 5.10, 4.15))
    ax_top = axes_inches(fig, (6.92, 3.505, 2.90, 1.865))
    ax_bot = axes_inches(fig, (6.92, 0.98, 2.90, 1.865))

    awake_h, peak = model_panel(ax_main, t, s, asleep)
    kss_panel(ax_top, perf)
    pvt_panel(ax_bot, perf)

    eps = episodes(t, asleep)
    rec = eps[-1][1] - eps[-1][0]
    norm = eps[0][1] - eps[0][0]
    note = ("Two-process model integrated at %.2f h steps with build-up time "
            "constant %.1f h and dissipation time constant %.1f h; the "
            "thresholds are a single cosine of period 24 h offset to their two "
            "means. Sleep onset and offset are threshold crossings found "
            "during integration, and the sleepiness and vigilance panels are "
            "driven by the same S and C traces sampled every two hours across "
            "eight virtual subjects." % (DT, TAU_R, TAU_D))
    panel_labels_inches(fig, [ax_main, ax_top, ax_bot], dx=-0.54, dy=0.13)
    savefig(fig, "791_slp_two_process_model.png")


if __name__ == "__main__":
    main()
