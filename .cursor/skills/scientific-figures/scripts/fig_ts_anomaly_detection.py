"""Anomaly detection read as three stacked, x-aligned views of one series.

(a) the raw signal with the injected anomalies ringed, (b) the detector's
score against its alarm threshold and (c) a one-line outcome ribbon that
colours every flagged sample as a hit, a false alarm or a miss.  Precision,
recall and F1 printed on the figure are counted from the ribbon itself.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_labels_inches

SIG_C = "#3a3a3a"      # observed signal
SCORE_C = "#3b5b92"    # house blue: the detector
THR_C = "#b23a48"      # house red: anomalies and the threshold
TP_C = "#2e7d5b"       # hit
FP_C = "#c9a227"       # false alarm
FN_C = "#b23a48"       # miss
N = 260
THR = 2.8
EVENTS = [(38, 3, 3.9), (77, 2, -4.6), (112, 4, 5.4), (146, 2, 2.3),
          (171, 3, -5.8), (198, 2, 3.6), (223, 3, 4.4), (245, 2, -2.2)]


def synth_signal(rng, n):
    t = np.arange(n, dtype=float)
    base = (52.0 + 3.4 * np.sin(2 * np.pi * t / 62.0)
            + 1.1 * np.sin(2 * np.pi * t / 23.0) + 0.012 * t)
    y = base + rng.normal(0.0, 0.75, n)
    truth = np.zeros(n, dtype=bool)
    for start, width, amp in EVENTS:
        y[start:start + width] += amp
        truth[start:start + width] = True
    return t, y, truth


def robust_score(y, win=9):
    """Rolling-median residual scaled by a rolling MAD -> a robust z."""
    pad = win // 2
    yp = np.pad(y, pad, mode="edge")
    med = np.array([np.median(yp[i:i + win]) for i in range(y.size)])
    res = y - med
    mad = np.median(np.abs(res - np.median(res))) * 1.4826
    return np.abs(res) / max(mad, 1e-9)


def panel_signal(ax, t, y, truth):
    ax.plot(t, y, color=SIG_C, lw=0.95, zorder=3)
    ax.plot(t[truth], y[truth], ls="none", marker="o", ms=5.2, mfc="none",
            mec=THR_C, mew=1.0, zorder=4)
    ax.set_ylabel("Sensor value\n(bar)", linespacing=1.25)
    ax.legend(handles=[Line2D([], [], color=SIG_C, lw=1.2, label="Signal"),
                       Line2D([], [], ls="none", marker="o", ms=4.4,
                              mfc="none", mec=THR_C, mew=1.0,
                              label="Injected anomaly")],
              loc="upper left", ncol=2, fontsize=7, handlelength=1.4,
              columnspacing=1.2, borderaxespad=0.3)


def panel_score(ax, t, score, thr):
    ax.fill_between(t, 0.0, score, color=SCORE_C, alpha=0.13, lw=0, zorder=2)
    ax.plot(t, score, color=SCORE_C, lw=1.0, zorder=3)
    ax.axhline(thr, color=THR_C, lw=1.0, ls=(0, (3.0, 2.0)), zorder=4)
    ax.text(t[0] + 2.0, thr + 0.30, f"threshold $s^*$ = {thr:.1f}",
            ha="left", va="bottom", fontsize=6.8, color=THR_C)
    ax.set_ylabel("Anomaly score\n$s_t$ (-)", linespacing=1.25)
    ax.set_ylim(0.0, max(score.max() * 1.18, thr * 1.6))


def panel_ribbon(ax, truth, det):
    """Colour whole runs, so the ribbon counts the same events the legend does."""
    hits = false_a = misses = 0
    for a, b in runs(truth):
        if not det[a:b].any():
            ax.bar((a + b) / 2.0 - 0.5, 1.0, width=max(b - a, 2),
                   color=FN_C, edgecolor="none", zorder=3)
            misses += 1
    for a, b in runs(det):
        hit = truth[a:b].any()
        ax.bar((a + b) / 2.0 - 0.5, 1.0, width=max(b - a, 2),
               color=TP_C if hit else FP_C, edgecolor="none", zorder=4)
        hits += int(hit)
        false_a += int(not hit)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_ylabel("Outcome", labelpad=13)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Day index (-)")
    return hits, false_a, misses


def runs(mask):
    """Contiguous True runs as (start, stop) pairs - one event each."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def event_counts(truth, det):
    hits = sum(1 for a, b in runs(truth) if det[a:b].any())
    misses = len(runs(truth)) - hits
    false_a = sum(1 for a, b in runs(det) if not truth[a:b].any())
    return hits, false_a, misses


def main():
    set_pub_style()
    rng = np.random.default_rng(55)
    t, y, truth = synth_signal(rng, N)
    score = robust_score(y)
    det = score > THR

    fig = plt.figure(figsize=(7.4, 4.6))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 0.62, 0.15],
                  hspace=0.16, left=0.098, right=0.982, top=0.895,
                  bottom=0.205)
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[1], sharex=ax)
    cx = fig.add_subplot(gs[2], sharex=ax)

    panel_signal(ax, t, y, truth)
    panel_score(bx, t, score, THR)
    hits, false_a, misses = panel_ribbon(cx, truth, det)
    for a in (ax, bx):
        a.grid(axis="y", lw=0.5, color="#f0f0f0")
        a.set_axisbelow(True)
        plt.setp(a.get_xticklabels(), visible=False)
        a.tick_params(axis="x", length=0)
    for a in (ax, bx, cx):
        style_axes(a)
    cx.set_xlim(-2, N + 1)

    prec = hits / max(hits + false_a, 1)
    rec = hits / max(hits + misses, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    bx.text(0.995, 0.955,
            f"event-level precision {prec:.2f}   recall {rec:.2f}   "
            f"F1 {f1:.2f}",
            transform=bx.transAxes, ha="right", va="top", fontsize=6.8,
            color="#555555")

    fig.legend(handles=[Patch(facecolor=TP_C, edgecolor="none",
                              label=f"hit ({hits} of "
                                    f"{hits + misses} events)"),
                        Patch(facecolor=FP_C, edgecolor="none",
                              label=f"false alarm ({false_a})"),
                        Patch(facecolor=FN_C, edgecolor="none",
                              label=f"miss ({misses})")],
               loc="lower center", bbox_to_anchor=(0.53, 0.045), ncol=3,
               fontsize=7, handlelength=1.5, handleheight=0.9,
               columnspacing=2.2, frameon=False)

    panel_labels_inches(fig, [ax, bx, cx], dx=-0.86, dy=0.03)
    savefig(fig, "358_ts_anomaly_detection.png")


if __name__ == "__main__":
    main()
