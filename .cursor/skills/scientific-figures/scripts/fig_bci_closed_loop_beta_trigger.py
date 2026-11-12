"""Closed-loop stimulation: the biomarker decides when the lead is on.

Main panel: twelve seconds of one closed-loop run drawn as three offset rows
on one time base - the local field potential, the causal beta-band envelope
the controller actually sees against its threshold, and the delivered current
- with every stimulation epoch shaded straight through all three rows.
Auxiliary panels (both 58 per cent of the main panel width): the envelope
averaged on stimulation onset against the matched open-loop average, and a
sweep of the threshold showing what it buys in duty cycle and costs in
suppression.

The loop is simulated sample by sample: a first-order Butterworth band-pass
runs as a three-tap difference equation, a peak-hold follower turns it into an
envelope, a hysteresis rule with a refractory period switches the current on,
and the current relaxes the beta drive with its own time constant, so the
trace that triggers the stimulator is the trace the stimulator changes.  Duty
cycle, burst counts, onset-aligned averages and the suppression percentage are
all measured on the arrays that are plotted.  Synthetic signal, not a patient
recording.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter
from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GREY = "#b3b3b3"
BOX = dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="none",
           alpha=0.86)

FS = 1000.0                  # sampling rate, Hz
DUR = 10.0                   # seconds shown
BAND = (13.0, 30.0)          # beta band, Hz
F_BETA = 21.5                # carrier of the synthetic burst, Hz
TAU_ENV = 0.085              # peak-hold decay of the envelope follower, s
CTRL_DT = 0.020              # controller update period, s
MIN_ON = 0.150               # shortest stimulation epoch, s
REFRACTORY = 0.120           # dead time after an epoch, s
OFF_FRAC = 0.75              # release fraction of the trigger threshold
I_STIM = 2.4                 # delivered current, mA
S_MAX = 0.72                 # fractional beta drive removed at full effect
TAU_ON, TAU_OFF = 0.22, 0.38  # suppression build-up and recovery, s
THR_PCTL = 68.0              # threshold percentile of the open-loop envelope
THR_SWEEP = np.linspace(35.0, 92.0, 20)

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)

ROW_LFP, ROW_ENV, ROW_STIM = 210.0, 40.0, -70.0
ENV_GAIN, STIM_GAIN = 1.55, 14.0


def sources(rng):
    """Beta drive, carrier and broadband noise for one synthetic channel."""
    n = int(DUR * FS)
    t = np.arange(n) / FS
    drive = np.full(n, 4.0)
    for centre in rng.uniform(0.6, DUR - 0.6, 9):
        width = rng.uniform(0.16, 0.40)
        drive += rng.uniform(22.0, 50.0) * np.exp(
            -0.5 * ((t - centre) / width) ** 2)
    phase = 2.0 * np.pi * F_BETA * t + np.cumsum(
        rng.normal(0.0, 0.055, n))
    noise = np.convolve(rng.normal(0.0, 6.0, n + 40),
                        np.ones(12) / 12.0, mode="same")[:n] * 3.0
    return t, drive, np.cos(phase), noise


def run(t, drive, carrier, noise, thr, closed=True):
    """One causal pass of the controller; returns the arrays it produced."""
    b, a = butter(1, [BAND[0], BAND[1]], btype="band", fs=FS)
    n = t.size
    lfp = np.empty(n)
    env = np.empty(n)
    stim = np.zeros(n)
    x1 = x2 = y1 = y2 = 0.0
    e = 0.0
    s = 0.0
    alpha = float(np.exp(-1.0 / (TAU_ENV * FS)))
    ctrl_n = max(int(CTRL_DT * FS), 1)
    on = False
    t_on = t_off = -1e9
    for i in range(n):
        tgt = S_MAX if on else 0.0
        tau = TAU_ON if on else TAU_OFF
        s += (tgt - s) / (tau * FS)
        x = drive[i] * (1.0 - s) * carrier[i] + noise[i]
        yb = (b[0] * x + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2)
        x2, x1 = x1, x
        y2, y1 = y1, yb
        e = max(abs(yb), e * alpha)
        lfp[i] = x
        env[i] = e
        if closed and i % ctrl_n == 0:
            now = i / FS
            if on:
                if e < OFF_FRAC * thr and now - t_on >= MIN_ON:
                    on, t_off = False, now
            elif e > thr and now - t_off >= REFRACTORY:
                on, t_on = True, now
        stim[i] = I_STIM if on else 0.0
    return lfp, env, stim


def epochs(stim, t):
    """Stimulation on/off times in seconds, read off the delivered current."""
    on = stim > 0.0
    edge = np.diff(on.astype(int))
    starts = list(np.flatnonzero(edge == 1) + 1)
    stops = list(np.flatnonzero(edge == -1) + 1)
    if on[0]:
        starts.insert(0, 0)
    if on[-1]:
        stops.append(len(on) - 1)
    return [(t[i], t[j]) for i, j in zip(starts, stops)]


def trace_panel(ax, t, lfp, env, stim, thr, ep, stats):
    for t0, t1 in ep:
        ax.axvspan(t0, t1, facecolor=BLUE, alpha=0.11, edgecolor="none",
                   zorder=1)
    ax.plot(t, ROW_LFP + lfp, color="#4a4a4a", lw=0.4, zorder=4)
    ax.plot(t, ROW_ENV + ENV_GAIN * env, color=RED, lw=1.1, zorder=5)
    ax.fill_between(t, ROW_ENV, ROW_ENV + ENV_GAIN * env, facecolor=RED,
                    alpha=0.10, edgecolor="none", zorder=3)
    ax.plot([0.0, DUR], [ROW_ENV + ENV_GAIN * thr] * 2, color=ORANGE, lw=1.0,
            ls=(0, (4.0, 2.0)), zorder=6)
    ax.plot([0.0, DUR], [ROW_ENV + ENV_GAIN * OFF_FRAC * thr] * 2,
            color=ORANGE, lw=0.8, ls=(0, (1.6, 2.0)), zorder=6)
    ax.step(t, ROW_STIM + STIM_GAIN * stim, where="post", color=BLUE, lw=1.0,
            zorder=5)
    ax.fill_between(t, ROW_STIM, ROW_STIM + STIM_GAIN * stim, step="post",
                    facecolor=BLUE, alpha=0.20, edgecolor="none", zorder=3)
    for t0, _ in ep:
        ax.plot([t0], [ROW_ENV + ENV_GAIN * thr], marker="v", ms=3.6,
                color=ORANGE, zorder=8)

    ax.text(0.10, ROW_LFP + 78.0,
            "local field potential, band 1 to 100 Hz ($\\mu V$)",
            fontsize=6.6, color=INK, ha="left", va="bottom", bbox=BOX,
            zorder=9)
    ax.text(0.10, ROW_ENV + ENV_GAIN * env.max() + 7.0,
            "causal beta envelope, {:.0f} to {:.0f} Hz ($\\mu V$); "
            "trigger at {:.1f} $\\mu V$, release at {:.1f} $\\mu V$"
            .format(BAND[0], BAND[1], thr, OFF_FRAC * thr), fontsize=6.6,
            color=RED, ha="left", va="bottom", bbox=BOX, zorder=9)
    ax.text(0.10, ROW_STIM + STIM_GAIN * I_STIM + 4.0,
            "delivered current, 0 or {:.1f} mA at 130 Hz".format(I_STIM),
            fontsize=6.6, color=BLUE, ha="left", va="bottom", bbox=BOX,
            zorder=9)
    ax.text(DUR - 0.10, ROW_STIM - 22.0,
            "{:d} epochs, duty cycle {:.1f} %, median epoch {:.0f} ms"
            .format(stats["n_ep"], stats["duty"], stats["median_ms"]),
            fontsize=6.6, color=SUB, ha="right", va="bottom", bbox=BOX,
            zorder=9)

    # The traces run the full width, so the scale bar lives just outside the
    # frame rather than on top of the data.
    ax.plot([DUR + 0.10] * 2, [ROW_LFP - 50.0, ROW_LFP + 50.0], color=INK,
            lw=1.2, zorder=9, clip_on=False)
    ax.text(DUR + 0.20, ROW_LFP, "100 $\\mu V$", fontsize=6.0, color=INK,
            ha="center", va="center", rotation=90, zorder=9, clip_on=False)

    ax.set_xlim(0.0, DUR)
    ax.set_ylim(ROW_STIM - 34.0, ROW_LFP + 120.0)
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.set_ylabel("three views of one closed-loop run (offset for display)")
    ax.tick_params(width=0.8, length=3.0)
    for s_name in ("top", "right", "left"):
        ax.spines[s_name].set_visible(False)
    ax.set_title("The stimulator answers each beta burst and lets go when the "
                 "envelope falls", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def onset_panel(ax, env_c, env_o, ep, stats):
    pre, post = 0.45, 0.95
    npre, npost = int(pre * FS), int(post * FS)
    lag = (np.arange(-npre, npost) / FS) * 1e3
    seg_c, seg_o = [], []
    for t0, _ in ep:
        i = int(round(t0 * FS))
        if i - npre < 0 or i + npost > env_c.size:
            continue
        seg_c.append(env_c[i - npre:i + npost])
        seg_o.append(env_o[i - npre:i + npost])
    seg_c, seg_o = np.array(seg_c), np.array(seg_o)
    m_c, m_o = seg_c.mean(axis=0), seg_o.mean(axis=0)
    sem = seg_c.std(axis=0) / np.sqrt(seg_c.shape[0])
    ax.axvspan(0.0, post * 1e3, facecolor=BLUE, alpha=0.09, edgecolor="none",
               zorder=1)
    ax.fill_between(lag, m_c - sem, m_c + sem, facecolor=RED, alpha=0.16,
                    edgecolor="none", zorder=3)
    ax.plot(lag, m_c, color=RED, lw=1.5, zorder=6,
            label="closed loop (n = {:d})".format(seg_c.shape[0]))
    ax.plot(lag, m_o, color=GREY, lw=1.3, ls=(0, (4.0, 2.0)), zorder=5,
            label="same bursts, stimulator off")
    ax.axvline(0.0, color=INK, lw=0.8, zorder=4)
    k = int(np.argmin(m_c[npre:npre + int(0.7 * FS)])) + npre
    ax.annotate("trough {:.0f} ms after onset\n{:.0f} % below the open loop"
                .format(lag[k], 100.0 * (1.0 - m_c[k] / m_o[k])),
                xy=(lag[k], m_c[k]), xytext=(-6.0, 30.0),
                textcoords="offset points", fontsize=6, color=RED,
                ha="right", va="bottom", linespacing=1.4, bbox=BOX, zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkA=1, shrinkB=3))
    ax.set_xlim(lag[0], lag[-1])
    ax.set_ylim(0.0, max(m_o.max(), m_c.max()) * 1.42)
    ax.set_xlabel("time from stimulation onset (ms)", fontsize=6.8)
    ax.set_ylabel("beta envelope ($\\mu V$)", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(lw=0.5, alpha=0.28)
    for s_name in ("top", "right"):
        ax.spines[s_name].set_visible(False)
    ax.legend(loc="upper left", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Onset-aligned envelope", fontsize=8.0, fontweight="bold",
                 color=INK)


def sweep_panel(ax, sweep, stats, thr):
    thrs = np.array([s[0] for s in sweep])
    duty = np.array([s[1] for s in sweep])
    supp = np.array([s[2] for s in sweep])
    ax.plot(thrs, duty, color=BLUE, lw=1.5, zorder=6, label="duty cycle")
    ax.plot([thr], [stats["duty"]], marker="o", ms=5.0, mfc="white",
            mec=BLUE, mew=1.2, zorder=8)
    ax.set_xlim(thrs[0], thrs[-1])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("trigger threshold ($\\mu V$)", fontsize=6.8)
    ax.set_ylabel("time stimulating (%)", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(lw=0.5, alpha=0.28)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(thrs, supp, color=RED, lw=1.4, ls=(0, (4.0, 2.0)), zorder=6,
             label="beta suppression")
    ax2.plot([thr], [stats["supp"]], marker="o", ms=5.0, mfc="white",
             mec=RED, mew=1.2, zorder=8)
    ax2.set_ylim(0.0, 100.0)
    ax2.set_ylabel("beta suppression (%)", fontsize=6.5, color=RED)
    ax2.tick_params(labelsize=6.0, width=0.7, length=2.5, colors=RED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(RED)
    ax2.annotate("chosen setting\n{:.1f} $\\mu V$: {:.0f} % of the time on,\n"
                 "{:.0f} % suppression"
                 .format(thr, stats["duty"], stats["supp"]),
                 xy=(thr, stats["supp"]), xytext=(-14.0, 46.0),
                 textcoords="offset points", fontsize=6, color=INK,
                 ha="right", va="bottom", linespacing=1.45, bbox=BOX,
                 zorder=9,
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                 shrinkA=1, shrinkB=3))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Threshold sweep on the same {:.0f} s".format(DUR),
                 fontsize=8.0, fontweight="bold", color=INK)


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


def summarise(t, env_c, env_o, stim):
    ep = epochs(stim, t)
    dur_ms = np.array([(b - a) * 1e3 for a, b in ep]) if ep else np.array([0.])
    return ep, dict(n_ep=len(ep), duty=100.0 * float((stim > 0).mean()),
                    median_ms=float(np.median(dur_ms)),
                    supp=100.0 * (1.0 - float(env_c.mean() / env_o.mean())))


def main():
    set_pub_style()
    rng = np.random.default_rng(9137)
    t, drive, carrier, noise = sources(rng)

    _, env_open, _ = run(t, drive, carrier, noise, np.inf, closed=False)
    thr = float(np.percentile(env_open, THR_PCTL))

    lfp, env, stim = run(t, drive, carrier, noise, thr, closed=True)
    ep, stats = summarise(t, env, env_open, stim)

    sweep = []
    for pct in THR_SWEEP:
        th = float(np.percentile(env_open, pct))
        _, e_c, s_c = run(t, drive, carrier, noise, th, closed=True)
        sweep.append((th, 100.0 * float((s_c > 0).mean()),
                      100.0 * (1.0 - float(e_c.mean() / env_open.mean()))))

    print("threshold {:.1f} uV ({:.0f}th percentile of the open loop)"
          .format(thr, THR_PCTL))
    print("{:d} epochs, duty cycle {:.1f} %, median epoch {:.0f} ms, "
          "beta suppression {:.1f} %".format(stats["n_ep"], stats["duty"],
                                             stats["median_ms"],
                                             stats["supp"]))
    print("sweep: duty {:.1f} % to {:.1f} %, suppression {:.1f} % to {:.1f} %"
          .format(sweep[0][1], sweep[-1][1], sweep[0][2], sweep[-1][2]))
    assert stats["duty"] < 100.0, "the loop does switch off"
    assert env.mean() < env_open.mean(), "closing the loop lowers beta"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_t = axes_inches(fig, MAIN)
    ax_o = axes_inches(fig, AUX_T)
    ax_s = axes_inches(fig, AUX_B)

    trace_panel(ax_t, t, lfp, env, stim, thr, ep, stats)
    onset_panel(ax_o, env, env_open, ep, stats)
    sweep_panel(ax_s, sweep, stats, thr)



    panel_labels_inches(fig, [ax_t, ax_o, ax_s], dx=-0.62, dy=0.14)
    savefig(fig, "1058_bci_closed_loop_beta_trigger.png")


if __name__ == "__main__":
    main()
