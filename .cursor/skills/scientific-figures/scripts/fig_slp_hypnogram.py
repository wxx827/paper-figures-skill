"""Whole-night hypnogram with the stage budget and the transition structure.

The main panel is the signature plot of sleep medicine: a 30-s epoch staircase
scored W / REM / N1 / N2 / N3 top-down, so that "deeper" literally means lower
on the page.  Slow-wave sleep loads the first half of the night and REM bouts
lengthen toward morning; the REM episodes are the only thing highlighted with a
tinted band, because they are what the reader is meant to count.  Every number
on the figure - sleep onset latency, REM latency, awakening count, sleep
efficiency, the stage shares and the transition probabilities - is measured
from the same epoch array that draws the staircase.  The two smaller panels
carry the summary evidence: how the night divides between stages, and which
stage-to-stage moves the night is actually made of.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import (set_pub_style, savefig, lighten, darken, annotate_box,
                      get_cmap)
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

BLUE = "#3b5b92"
RED = "#b23a48"
OCHRE = "#c0603f"

# Stage order is the scoring convention, not an accident: index 0 sits at the
# top of the panel and index 4 at the bottom, so the y axis reads W -> N3.
NAMES = ["W", "REM", "N1", "N2", "N3"]
COL = [RED, OCHRE, lighten(BLUE, 0.46), BLUE, darken(BLUE, 0.34)]

EPOCH_MIN = 0.5                 # 30-s scoring epochs
EPOCH_H = EPOCH_MIN / 60.0
LIGHTS_OFF_H = 22.0             # clock hour of the first epoch
TIB_H = 9.0                     # time in bed


def hypnogram(seed=7):
    """Epoch-by-epoch stage codes for one night, front-loaded with N3."""
    rng = np.random.default_rng(seed)
    n = int(round(TIB_H / EPOCH_H))
    stg = np.zeros(n, dtype=int)
    i = 0

    def put(minutes, level):
        nonlocal i
        k = max(int(round(minutes / EPOCH_MIN)), 1)
        j = min(i + k, n)
        stg[i:j] = level
        i = j

    put(13.0 + rng.normal(0.0, 2.4), 0)          # sleep onset latency
    n3_plan = [34.0, 26.0, 13.0, 5.0, 0.0]
    rem_plan = [9.0, 17.0, 24.0, 29.0, 31.0]
    c = 0
    while i < n - 24:
        n3 = n3_plan[min(c, 4)]
        rem = rem_plan[min(c, 4)]
        put(rng.uniform(2.5, 5.0), 2)            # N1 doorway
        put(rng.uniform(26.0, 36.0), 3)
        if n3 > 0.0:
            put(n3 * rng.uniform(0.85, 1.15), 4)
            put(rng.uniform(18.0, 26.0), 3)
        else:
            put(rng.uniform(22.0, 30.0), 3)
        if c > 0 and rng.random() < 0.75:
            put(rng.uniform(1.5, 7.0), 0)        # arousal before REM
            put(rng.uniform(0.5, 1.5), 2)
        put(rem * rng.uniform(0.85, 1.15), 1)
        if rng.random() < 0.55:
            put(rng.uniform(1.0, 4.5), 0)
        c += 1

    final_wake = int(round(rng.uniform(6.0, 12.0) / EPOCH_MIN))
    stg[n - final_wake:] = 0
    return stg


def runs(a):
    """Contiguous constant-value runs as ``(start, stop, value)`` triples."""
    out = []
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        out.append((i, j + 1, int(a[i])))
        i = j + 1
    return out


def metrics(stg):
    """Every annotated quantity, computed from the plotted epoch array."""
    n = len(stg)
    asleep = stg != 0
    onset = int(np.argmax(asleep))
    last = int(n - 1 - np.argmax(asleep[::-1]))
    tst = float(asleep.sum()) * EPOCH_MIN
    rem_idx = np.flatnonzero(stg == 1)
    wake_runs = [r for r in runs(stg) if r[2] == 0 and onset < r[0] <= last]
    share = {NAMES[k]: float((stg == k).sum()) * EPOCH_MIN for k in range(5)}
    return dict(
        onset=onset, last=last,
        sol=onset * EPOCH_MIN,
        tst=tst, tib=n * EPOCH_MIN,
        eff=tst / (n * EPOCH_MIN),
        rem_lat=(int(rem_idx[0]) - onset) * EPOCH_MIN,
        waso=sum((r[1] - r[0]) for r in wake_runs) * EPOCH_MIN,
        n_wake=len(wake_runs),
        n_rem=len([r for r in runs(stg) if r[2] == 1]),
        minutes=share)


def transition_matrix(stg):
    """Bout-level stage-to-stage transition probabilities (self moves dropped)."""
    m = np.zeros((5, 5))
    seq = [r[2] for r in runs(stg)]
    for a, b in zip(seq[:-1], seq[1:]):
        m[a, b] += 1.0
    rows = m.sum(axis=1, keepdims=True)
    return m / np.where(rows > 0, rows, 1.0)


def clock_labels(hours):
    return ["%02d:00" % int(round(h) % 24) for h in hours]


def main_panel(ax, stg, m):
    t = np.arange(len(stg)) * EPOCH_H
    seg = runs(stg)

    for lv in range(5):
        ax.axhline(lv, color=GRID, lw=0.6, zorder=0)
    for a, b, lv in seg:
        if lv == 1:
            ax.axvspan(a * EPOCH_H, b * EPOCH_H, color=OCHRE, alpha=0.13,
                       lw=0.0, zorder=1)

    ax.step(np.append(t, TIB_H), np.append(stg, stg[-1]), where="post",
            color=RULE, lw=0.75, zorder=3)
    for a, b, lv in seg:
        ax.plot([a * EPOCH_H, b * EPOCH_H], [lv, lv], color=COL[lv],
                lw=3.0, solid_capstyle="butt", zorder=4)

    for a, b, lv in seg:
        if lv == 0 and m["onset"] < a <= m["last"]:
            ax.plot([(a + b) * 0.5 * EPOCH_H], [-0.30], marker="v", ms=3.0,
                    color=RED, zorder=5, clip_on=False)

    sol_h, rem_h = m["sol"] / 60.0, m["rem_lat"] / 60.0
    on_h = m["onset"] * EPOCH_H
    ax.annotate("", xy=(0.0, -0.62), xytext=(on_h, -0.62),
                arrowprops=dict(arrowstyle="<->", color=SUB, lw=0.8,
                                shrinkA=0, shrinkB=0))
    ax.text(on_h + 0.09, -0.62,
            "sleep onset latency %.1f min" % m["sol"], ha="left",
            va="center", fontsize=6.8, color=SUB)
    ax.annotate("", xy=(on_h, -0.95), xytext=(on_h + rem_h, -0.95),
                arrowprops=dict(arrowstyle="<->", color=OCHRE, lw=0.8,
                                shrinkA=0, shrinkB=0))
    ax.text(on_h + rem_h * 0.5, -1.02,
            "REM latency %.0f min" % m["rem_lat"], ha="center", va="bottom",
            fontsize=6.8, color=OCHRE)
    ax.text(TIB_H - 0.06, -1.02, "%d awakenings after onset" % m["n_wake"],
            ha="right", va="bottom", fontsize=6.8, color=RED)

    ax.set_ylim(4.62, -1.28)
    ax.set_xlim(0.0, TIB_H)
    ax.set_yticks(range(5))
    ax.set_yticklabels(NAMES)
    for lab, c in zip(ax.get_yticklabels(), COL):
        lab.set_color(c)
        lab.set_fontweight("bold")
    ticks = np.arange(0.0, TIB_H + 1e-9, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(clock_labels(LIGHTS_OFF_H + ticks), fontsize=7.0)
    ax.set_xlabel("clock time (h:min)")
    ax.set_ylabel("sleep stage (-)")
    ax.spines["left"].set_visible(True)

    annotate_box(ax, "\n".join([
        "time in bed %.0f min" % m["tib"],
        "total sleep time %.0f min" % m["tst"],
        "sleep efficiency %.1f %%" % (100.0 * m["eff"]),
        "WASO %.0f min" % m["waso"],
        "%d NREM-REM cycles" % m["n_rem"]]),
        loc="lower right", fontsize=6.6, color=INK)
    ax.set_title("Slow-wave sleep front-loads the night, REM bouts lengthen "
                 "toward morning", fontsize=9.0, fontweight="bold", color=INK)


def budget_panel(ax, m):
    total = m["tib"]
    left = 0.0
    for k in range(5):
        pct = 100.0 * m["minutes"][NAMES[k]] / total
        ax.barh(0.0, pct, left=left, height=0.44, color=COL[k],
                edgecolor="white", linewidth=0.8, zorder=3,
                label="%s  %.0f min" % (NAMES[k], m["minutes"][NAMES[k]]))
        y = 0.40 if k % 2 == 0 else 0.66
        ax.plot([left + pct / 2.0] * 2, [0.22, y - 0.03], lw=0.6,
                color=MUTED, zorder=2)
        ax.text(left + pct / 2.0, y, "%.1f %%" % pct, ha="center",
                va="bottom", fontsize=6.4, color=INK)
        left += pct

    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-1.42, 1.02)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("share of time in bed (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.40), ncol=3,
              fontsize=6.4, frameon=False, handlelength=1.1,
              handletextpad=0.45, columnspacing=1.0, labelspacing=0.30)
    ax.set_title("Stage budget of the night", fontsize=8.2,
                 fontweight="bold", color=INK)


def transition_panel(ax, tm):
    im = ax.imshow(tm, cmap=get_cmap("cb_sequential_blue"), aspect="auto",
                   vmin=0.0, vmax=1.0)
    for r in range(5):
        for c in range(5):
            if r == c:
                ax.text(c, r, "-", ha="center", va="center", fontsize=6.2,
                        color=MUTED)
                continue
            v = tm[r, c]
            ax.text(c, r, "%.2f" % v, ha="center", va="center", fontsize=6.0,
                    color="white" if v > 0.55 else INK)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(NAMES, fontsize=6.5)
    ax.set_yticklabels(NAMES, fontsize=6.5)
    for lab, c in zip(ax.get_xticklabels(), COL):
        lab.set_color(c)
    for lab, c in zip(ax.get_yticklabels(), COL):
        lab.set_color(c)
    ax.set_xlabel("stage entered (-)", fontsize=7.4)
    ax.set_ylabel("stage left (-)", fontsize=7.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_title("Bout transition probabilities", fontsize=8.2,
                 fontweight="bold", color=INK)
    return im


def main():
    set_pub_style()
    stg = hypnogram()
    m = metrics(stg)
    tm = transition_matrix(stg)

    fw, fh = 10.4, 6.4
    fig = plt.figure(figsize=(fw, fh))
    ax_main = axes_inches(fig, (0.80, 0.92, 5.30, 4.45))
    ax_top = axes_inches(fig, (7.00, 3.505, 2.85, 1.865))
    ax_bot = axes_inches(fig, (7.00, 0.92, 2.85, 1.865))

    main_panel(ax_main, stg, m)
    budget_panel(ax_top, m)
    transition_panel(ax_bot, tm)

    note = ("Synthetic 30-s epoch scoring over %.0f h in bed from %02d:00; "
            "stages are drawn on the clinical top-down axis so that depth "
            "increases downward, and the tinted bands mark REM episodes. "
            "Latencies, awakening count, sleep efficiency, stage shares and "
            "transition probabilities are all recomputed from the epoch array "
            "that draws the staircase." % (TIB_H, LIGHTS_OFF_H))
    panel_labels_inches(fig, [ax_main, ax_top, ax_bot], dx=-0.50, dy=0.13)
    savefig(fig, "786_slp_hypnogram.png")


if __name__ == "__main__":
    main()
