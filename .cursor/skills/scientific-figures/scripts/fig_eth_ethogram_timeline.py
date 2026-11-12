"""Ethogram of eight focal animals across a full 24 h cycle.

The main panel is the claim: the group is bimodally crepuscular, and the two
activity peaks are synchronised across individuals rather than being an
artefact of averaging - the state bands line up vertically, and the activity
trace underneath is the same data collapsed to one number per time bin.  The
two subordinate panels quantify what the bands only show qualitatively: how
the 24 h is divided between behaviours per animal, and how sticky each
behaviour is once entered.  Every percentage and probability is counted from
the plotted state sequences.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import axes_inches, panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

BEHAV = ["foraging", "resting", "travelling", "social", "vigilance"]
SHORT = ["forage", "rest", "travel", "social", "vigil"]
COLOR = ["#2e7d5b", "#3b5b92", "#c9a227", "#8e6bbf", "#b23a48"]
STICK = np.array([0.930, 0.962, 0.855, 0.805, 0.780])   # per-step persistence

N_IND = 8
DT_MIN = 5.0
N_BIN = int(24 * 60 / DT_MIN)
SUNRISE, SUNSET = 6.0, 19.0
ACTIVE = (0, 2)                    # foraging and travelling count as active


def drive(t, shift=0.0, gain=1.0):
    """Crepuscular activity drive on 0..1 for hour-of-day ``t``."""
    a = (0.85 * np.exp(-0.5 * ((t - 6.6 - shift) / 1.75) ** 2)
         + 0.92 * np.exp(-0.5 * ((t - 18.4 - shift) / 2.05) ** 2) + 0.16)
    return np.clip(gain * a, 0.0, 1.0)


def preference(a):
    p = np.stack([0.03 + 0.58 * a, 0.80 - 0.66 * a, 0.03 + 0.16 * a,
                  0.02 + 0.10 * a, 0.03 + 0.07 * a], axis=-1)
    return p / p.sum(axis=-1, keepdims=True)


def ethogram(seed=31):
    rng = np.random.default_rng(seed)
    t = np.arange(N_BIN) * DT_MIN / 60.0
    seqs = np.empty((N_IND, N_BIN), dtype=int)
    for i in range(N_IND):
        p = preference(drive(t, shift=rng.normal(0.0, 0.32),
                             gain=rng.uniform(0.88, 1.10)))
        s = int(rng.choice(5, p=p[0]))
        for k in range(N_BIN):
            if k and rng.random() < STICK[s]:
                seqs[i, k] = s
                continue
            s = int(rng.choice(5, p=p[k]))
            seqs[i, k] = s
    return t, seqs


def runs(seq):
    idx = np.flatnonzero(np.diff(seq)) + 1
    starts = np.r_[0, idx]
    ends = np.r_[idx, seq.size]
    return starts, ends, seq[starts]


def night_shade(ax):
    for x0, x1 in [(0.0, SUNRISE), (SUNSET, 24.0)]:
        ax.axvspan(x0, x1, facecolor="#f0f0f0", edgecolor="none", zorder=0)


def band_panel(ax, t, seqs):
    night_shade(ax)
    for i in range(N_IND):
        s, e, v = runs(seqs[i])
        for k in range(5):
            m = v == k
            if not m.any():
                continue
            spans = [(t[a], (b - a) * DT_MIN / 60.0)
                     for a, b in zip(s[m], e[m])]
            ax.broken_barh(spans, (N_IND - 1 - i - 0.38, 0.76),
                           facecolors=COLOR[k], edgecolor="none", zorder=3)
    ax.set_yticks(np.arange(N_IND))
    ax.set_yticklabels([f"A{N_IND - j:02d}" for j in range(N_IND)],
                       fontsize=7.0)
    ax.set_ylim(-0.62, N_IND - 0.38)
    ax.set_xlim(0.0, 24.0)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.tick_params(labelbottom=False, length=2.5)
    ax.set_ylabel("focal individual (-)")
    ax.text(SUNRISE / 2.0, N_IND - 0.52, "night", ha="center", va="top",
            fontsize=6.4, color=MUTED)
    ax.text((SUNSET + 24.0) / 2.0, N_IND - 0.52, "night", ha="center",
            va="top", fontsize=6.4, color=MUTED)
    ax.set_title("Both activity peaks are synchronised across the group",
                 fontsize=9.0, fontweight="bold", color=INK)


def activity_panel(ax, t, seqs):
    act = 100.0 * np.isin(seqs, ACTIVE).mean(axis=0)
    k = 7
    sm = np.convolve(np.r_[act[-k:], act, act[:k]], np.ones(k) / k,
                     mode="same")[k:-k]
    night_shade(ax)
    ax.fill_between(t, 0.0, sm, color="#3b5b92", alpha=0.20, linewidth=0.0,
                    zorder=2)
    ax.plot(t, sm, color="#3b5b92", lw=1.5, zorder=4)
    peaks = [float(t[np.argmax(sm[t < 12.0])]),
             float(t[12 <= t][np.argmax(sm[t >= 12.0])])]
    for p in peaks:
        ax.axvline(p, color="#b23a48", lw=0.9, ls=(0, (2.4, 2.0)), zorder=5)
        ax.text(p, 96.0, f"{int(p):02d}:{int(round((p % 1) * 60)):02d}",
                ha="center", va="top", fontsize=6.3, color="#b23a48")
    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 3)], fontsize=7.0)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(labelsize=7.0)
    ax.set_xlabel("time of day (h)")
    ax.set_ylabel("active (%)", fontsize=7.6)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    return peaks


def budget_panel(ax, seqs):
    frac = np.stack([100.0 * (seqs == k).mean(axis=1) for k in range(5)],
                    axis=1)
    y = np.arange(N_IND)[::-1]
    left = np.zeros(N_IND)
    for k in range(5):
        ax.barh(y, frac[:, k], left=left, height=0.68, color=COLOR[k],
                edgecolor="white", linewidth=0.5, zorder=3)
        left = left + frac[:, k]
    mean = frac.mean(axis=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"A{i + 1:02d}" for i in range(N_IND)], fontsize=6.5)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-0.7, N_IND - 0.3)
    ax.set_xlabel("time budget (% of 24 h)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.45, alpha=0.6)
    ax.set_title(f"Resting takes {mean[1]:.0f} % of the day",
                 fontsize=8.2, fontweight="bold", color=INK)
    return mean


def transition_panel(fig, ax, seqs, bin_min=10.0):
    step = int(bin_min / DT_MIN)
    coarse = np.stack([
        np.array([np.bincount(row[i:i + step], minlength=5).argmax()
                  for i in range(0, N_BIN, step)]) for row in seqs])
    m = np.zeros((5, 5))
    for row in coarse:
        for a, b in zip(row[:-1], row[1:]):
            m[a, b] += 1.0
    p = m / m.sum(axis=1, keepdims=True)

    im = ax.imshow(p, cmap=get_cmap("cb_sequential_blue"), aspect="auto",
                   vmin=0.0, vmax=1.0)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{p[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.0,
                    color="white" if p[i, j] > 0.55 else INK)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(SHORT, fontsize=6.4, rotation=28, ha="right")
    ax.set_yticklabels(SHORT, fontsize=6.4)
    ax.set_xlabel("state at $t$ + 10 min (-)", fontsize=7.4)
    ax.set_ylabel("state at $t$ (-)", fontsize=7.4)
    ax.tick_params(length=2.0)
    ax.set_title("Diagonal is behavioural persistence", fontsize=8.2,
                 fontweight="bold", color=INK)
    safe_colorbar(fig, im, ax, label="transition probability (-)", width=0.10,
                  pad=0.06, labelsize=6.4, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    return np.diag(p)


def main():
    set_pub_style()
    t, seqs = ethogram()

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_band = axes_inches(fig, (0.86, 2.05, 5.05, 3.32))
    ax_act = axes_inches(fig, (0.86, 0.82, 5.05, 0.92))
    ax_bud = axes_inches(fig, (7.00, 3.40, 2.62, 2.00))
    ax_trn = axes_inches(fig, (7.00, 0.82, 2.62, 1.88))

    band_panel(ax_band, t, seqs)
    peaks = activity_panel(ax_act, t, seqs)
    mean = budget_panel(ax_bud, seqs)
    diag = transition_panel(fig, ax_trn, seqs)

    panel_labels_inches(fig, [ax_band, ax_act, ax_bud, ax_trn], dx=-0.52,
                        dy=0.12)
    savefig(fig, "776_eth_ethogram_timeline.png")


if __name__ == "__main__":
    main()
