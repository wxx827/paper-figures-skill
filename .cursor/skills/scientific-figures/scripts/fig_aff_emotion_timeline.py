"""Continuous valence and arousal over a four-minute clip, with state segments.

Six simulated annotators track a scripted affect trajectory; the plotted lines
are their mean and the bands are plus or minus one standard deviation across
annotators.  The background colouring is not decoration: each frame is
assigned to a circumplex quadrant, runs shorter than the minimum segment
length are absorbed into their longer neighbour, and the resulting segment
list produces both the transition matrix and the duration shares on the
right, so the three panels always describe the same segmentation.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
GRID = "#cccccc"
V_COLOR = "#333333"
A_COLOR = "#8e6bbf"

STATES = ["Excited", "Content", "Sad", "Tense"]
STATE_COLOR = {"Excited": "#3b5b92", "Content": "#2e7d5b",
               "Sad": "#c0603f", "Tense": "#b23a48"}

T_END = 240.0
DT = 0.5
N_RATERS = 6
MIN_SEG = 10.0

KNOT_T = [0.0, 25.0, 50.0, 65.0, 85.0, 100.0, 120.0, 135.0, 150.0, 165.0,
          180.0, 200.0, 215.0, 230.0, 240.0]
KNOT_V = [0.30, 0.55, 0.25, -0.30, -0.55, -0.60, -0.45, -0.15, 0.25, 0.40,
          0.45, 0.55, 0.40, 0.25, 0.15]
KNOT_A = [0.20, 0.50, 0.62, 0.58, 0.35, -0.15, -0.45, -0.40, -0.35, -0.15,
          0.25, 0.50, -0.05, -0.35, -0.45]
TURN_WIN = 20.0     # s, half-window used to quote the size of a change


def annotator_tracks(rng):
    t = np.arange(0.0, T_END + DT, DT)
    base_v = PchipInterpolator(KNOT_T, KNOT_V)(t)
    base_a = PchipInterpolator(KNOT_T, KNOT_A)(t)
    k = np.ones(9) / 9.0
    out = np.empty((N_RATERS, t.size, 2))
    for r in range(N_RATERS):
        bias = rng.normal(0.0, 0.07, 2)
        for d, base in enumerate((base_v, base_a)):
            wobble = np.convolve(rng.standard_normal(t.size), k, mode="same")
            wobble = np.convolve(wobble, k, mode="same") * 0.55
            out[r, :, d] = np.clip(base + bias[d] + wobble, -1.0, 1.0)
    return t, out


def quadrant(v, a):
    if a >= 0.0:
        return "Excited" if v >= 0.0 else "Tense"
    return "Content" if v >= 0.0 else "Sad"


def segment(labels, dt, min_dur=MIN_SEG):
    """Run-length encode, then absorb runs shorter than ``min_dur``."""
    segs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segs.append([labels[start], start * dt, i * dt])
            start = i
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for k, s in enumerate(segs):
            if s[2] - s[1] >= min_dur:
                continue
            if k == 0:
                nb = 1
            elif k == len(segs) - 1:
                nb = len(segs) - 2
            else:
                prev = segs[k - 1][2] - segs[k - 1][1]
                nxt = segs[k + 1][2] - segs[k + 1][1]
                nb = k - 1 if prev >= nxt else k + 1
            if nb < k:
                segs[nb][2] = s[2]
            else:
                segs[nb][1] = s[1]
            segs.pop(k)
            changed = True
            break
    merged = [segs[0]]
    for s in segs[1:]:
        if s[0] == merged[-1][0]:
            merged[-1][2] = s[2]
        else:
            merged.append(s)
    return merged


def main():
    set_pub_style()
    rng = np.random.default_rng(851)
    t, tracks = annotator_tracks(rng)
    mean = tracks.mean(axis=0)
    sd = tracks.std(axis=0, ddof=1)

    labels = [quadrant(v, a) for v, a in mean]
    segs = segment(labels, DT)

    w = int(TURN_WIN / DT)
    speed = np.full(t.size, np.nan)
    speed[w:-w] = np.hypot(mean[2 * w:, 0] - mean[:-2 * w, 0],
                           mean[2 * w:, 1] - mean[:-2 * w, 1])
    turns = []
    for i in np.argsort(np.nan_to_num(speed, nan=-1.0))[::-1]:
        if all(abs(t[i] - t[j]) > 45.0 for j in turns):
            turns.append(int(i))
        if len(turns) == 3:
            break
    turns.sort()

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.85, 1.35, 5.17, 4.25))
    ax_rib = axes_inches(fig, (0.85, 0.92, 5.17, 0.26))
    ax_tr = axes_inches(fig, (7.00, 3.62, 2.60, 1.98))
    ax_dur = axes_inches(fig, (7.00, 0.92, 2.60, 1.72))

    # ---- affect timeline ---------------------------------------------------
    for name, t0, t1 in segs:
        ax.axvspan(t0, t1, facecolor=STATE_COLOR[name], alpha=0.11,
                   edgecolor="none", zorder=0)
        ax.axvline(t1, color="#b3b3b3", lw=0.6, zorder=1)
    ax.axhline(0.0, color=RULE, lw=0.8, zorder=2)
    for d, (col, lab) in enumerate(((V_COLOR, "valence"),
                                    (A_COLOR, "arousal"))):
        ax.fill_between(t, mean[:, d] - sd[:, d], mean[:, d] + sd[:, d],
                        color=col, alpha=0.16, lw=0.0, zorder=3)
        ax.plot(t, mean[:, d], color=col, lw=1.4, zorder=4,
                label="%s (mean of %d raters, +/- 1 SD)" % (lab, N_RATERS))

    for i in turns:
        dv = mean[i + w, 0] - mean[i - w, 0]
        da = mean[i + w, 1] - mean[i - w, 1]
        which = ("valence", dv) if abs(dv) >= abs(da) else ("arousal", da)
        top = mean[i, 1] >= mean[i, 0]
        y = (max(mean[i]) + 0.26) if top else (min(mean[i]) - 0.26)
        ax.annotate("t = %.0f s\n%s %+.2f in %.0f s"
                    % (t[i], which[0], which[1], 2 * TURN_WIN),
                    xy=(t[i], mean[i, 1] if top else mean[i, 0]),
                    xytext=(t[i], y), ha="center",
                    va="bottom" if top else "top", fontsize=6.4, color=INK,
                    linespacing=1.35, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED,
                                    mutation_scale=7, shrinkA=1.0,
                                    shrinkB=2.0))

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(-1.05, 1.15)
    ax.set_ylabel("rating (-, -1 to +1)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, labelbottom=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.005), handlelength=1.6,
              columnspacing=1.6)
    ax.set_title("Continuous affect over %.0f s, %d emotion segments detected"
                 % (T_END, len(segs)), fontsize=9.0, fontweight="bold",
                 pad=4.0)

    # ---- segment ribbon ----------------------------------------------------
    for name, t0, t1 in segs:
        ax_rib.axvspan(t0, t1, facecolor=STATE_COLOR[name], alpha=0.92,
                       edgecolor="white", lw=0.7)
        if t1 - t0 > 22.0:
            ax_rib.text(0.5 * (t0 + t1), 0.5, name, ha="center", va="center",
                        fontsize=6.2, color="white")
    ax_rib.set_xlim(0.0, T_END)
    ax_rib.set_ylim(0.0, 1.0)
    ax_rib.set_yticks([])
    ax_rib.set_xlabel("time (s)", fontsize=8.0)
    ax_rib.tick_params(labelsize=7.0)
    for s in ("top", "right", "left"):
        ax_rib.spines[s].set_visible(False)

    # ---- transition matrix -------------------------------------------------
    order = STATES
    mat = np.zeros((4, 4))
    for a, b in zip(segs[:-1], segs[1:]):
        mat[order.index(a[0]), order.index(b[0])] += 1.0
    im = ax_tr.imshow(mat, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                      vmax=max(mat.max(), 1.0), origin="upper")
    ax_tr.set_aspect("equal")
    for i in range(4):
        for j in range(4):
            ax_tr.text(j, i, "%d" % mat[i, j], ha="center", va="center",
                       fontsize=6.6,
                       color="white" if mat[i, j] > 0.55 * mat.max() else SUB)
    ax_tr.set_xticks(range(4))
    ax_tr.set_yticks(range(4))
    ax_tr.set_xticklabels(order, fontsize=6.4, rotation=30, ha="right")
    ax_tr.set_yticklabels(order, fontsize=6.4)
    for i, name in enumerate(order):
        ax_tr.get_yticklabels()[i].set_color(STATE_COLOR[name])
        ax_tr.get_xticklabels()[i].set_color(STATE_COLOR[name])
    ax_tr.set_xlabel("next state", fontsize=7.5)
    ax_tr.set_ylabel("current state", fontsize=7.5)
    ax_tr.tick_params(length=2.0)
    ax_tr.set_title("Segment transitions (n=%d)" % int(mat.sum()),
                    fontsize=8.5, fontweight="bold", pad=3.0)

    # ---- duration share ----------------------------------------------------
    share = {s: 0.0 for s in order}
    for name, t0, t1 in segs:
        share[name] += t1 - t0
    pct = np.array([100.0 * share[s] / T_END for s in order])
    ys = np.arange(4)[::-1]
    ax_dur.barh(ys, pct, height=0.62,
                color=[STATE_COLOR[s] for s in order], alpha=0.9,
                edgecolor="white", lw=0.5)
    for y, s, p in zip(ys, order, pct):
        ax_dur.text(p + 1.2, y, "%.1f %% (%.0f s)" % (p, share[s]),
                    va="center", ha="left", fontsize=6.3, color=SUB)
    ax_dur.set_yticks(ys)
    ax_dur.set_yticklabels(order, fontsize=6.6)
    for i, name in enumerate(order):
        ax_dur.get_yticklabels()[i].set_color(STATE_COLOR[name])
    ax_dur.set_xlim(0.0, pct.max() * 1.55)
    ax_dur.set_xlabel("share of clip duration (%)", fontsize=7.5)
    ax_dur.tick_params(labelsize=6.5, length=2.0)
    for s in ("top", "right"):
        ax_dur.spines[s].set_visible(False)
    ax_dur.set_title("Time spent per emotion state", fontsize=8.5,
                     fontweight="bold", pad=3.0)

    panel_labels_inches(fig, [ax, ax_tr, ax_dur], dx=-0.52, dy=0.12)


    savefig(fig, "851_aff_emotion_timeline.png")


if __name__ == "__main__":
    main()
