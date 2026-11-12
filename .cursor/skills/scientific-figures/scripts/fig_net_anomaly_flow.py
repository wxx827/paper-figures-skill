"""Figure 646 - One day of flow telemetry with three injected incidents.

The three stacked panels share one time axis on purpose: the point of the
figure is that the volume trace alone (a) misses the port scan entirely, that
the per-feature detector scores (b) are what actually separate the three
incidents, and that the alerts (c) therefore land in three visually different
patterns.  Every alert in (c) is emitted by thresholding the scores drawn in
(b), and every number in the two right-hand panels is a confusion-matrix
reduction of those same alerts against the shaded ground-truth windows.

Reference-gallery figure: the numbers are generated, the layout is the point.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pubstyle import set_pub_style, savefig, COLORS
from figkit import panel_label_inches

ACCENT = "#b23a48"          # house red - attack windows and misses
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"

THR = 0.62                  # detector threshold on every feature score
ATTACKS = [(6.20, 7.05, "SYN flood"),
           (13.50, 14.35, "DNS amplification"),
           (19.75, 21.00, "slow port scan")]
FEATURES = [("volume z-score", BLUE), ("flow-entropy drop", GREEN),
            ("SYN / ACK ratio", GOLD)]
# how strongly each incident drives each of the three features
RESPONSE = np.array([[0.95, 0.40, 0.88],      # SYN flood
                     [0.78, 0.62, 0.15],      # DNS amplification
                     [0.06, 0.92, 0.30]])     # slow port scan


def bump(t, t0, t1, sharp=1.0):
    """Smooth 0..1 window over [t0, t1]."""
    mid, half = 0.5 * (t0 + t1), 0.5 * (t1 - t0)
    return np.exp(-0.5 * (np.abs(t - mid) / (half * 0.62)) ** (2.0 * sharp))


def telemetry(rng, n=1441):
    t = np.linspace(0.0, 24.0, n)
    base = (24.0 + 16.0 * np.exp(-0.5 * ((t - 11.0) / 3.6) ** 2) +
            12.0 * np.exp(-0.5 * ((t - 20.5) / 2.4) ** 2))
    walk = np.cumsum(rng.normal(0.0, 1.0, n))
    walk = 2.4 * (walk - walk.mean()) / (np.abs(walk).max() + 1e-9)
    gbps = base + walk + rng.normal(0.0, 0.85, n)

    scores = np.zeros((3, n))
    for k in range(3):
        drift = np.cumsum(rng.normal(0.0, 1.0, n))
        drift = (drift - drift.mean()) / (np.abs(drift).max() + 1e-9)
        s = 0.16 + 0.10 * drift + np.abs(rng.normal(0.0, 0.075, n))
        # benign bursts: reconfigurations and backup windows that look odd
        for c in rng.integers(0, n, 16):
            s = s + rng.uniform(0.16, 0.62) * bump(
                t, t[c] - 0.06, t[c] + 0.06, sharp=1.4)
        scores[k] = np.clip(s, 0.0, 1.0)

    truth = np.zeros(n, dtype=bool)
    for a, (t0, t1, _) in enumerate(ATTACKS):
        w = bump(t, t0, t1, sharp=1.6 if a < 2 else 1.0)
        gbps = gbps + (34.0 if a == 0 else 19.0 if a == 1 else 1.2) * w
        for k in range(3):
            scores[k] = np.clip(scores[k] + RESPONSE[a, k] * w, 0.0, 1.0)
        truth |= (t >= t0) & (t <= t1)
    return t, gbps, scores, truth


def main():
    set_pub_style()
    rng = np.random.default_rng(646)
    t, gbps, scores, truth = telemetry(rng)

    fired = scores > THR                       # (3, n) per-detector firing
    pred = fired.any(axis=0)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    spec = tn / max(tn + fp, 1)

    delays = []
    for t0, t1, _ in ATTACKS:
        inside = np.flatnonzero(pred & (t >= t0) & (t <= t1))
        delays.append((t[inside[0]] - t0) * 60.0 if inside.size else np.nan)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.55],
                          left=0.062, right=0.965, top=0.885, bottom=0.155,
                          wspace=0.26)
    gsl = gs[0, 0].subgridspec(3, 1, hspace=0.18)
    gsr = gs[0, 1].subgridspec(2, 1, hspace=0.58)
    ax_a = fig.add_subplot(gsl[0])
    ax_b = fig.add_subplot(gsl[1], sharex=ax_a)
    ax_c = fig.add_subplot(gsl[2], sharex=ax_a)
    ax_d = fig.add_subplot(gsr[0])
    ax_e = fig.add_subplot(gsr[1])

    for ax in (ax_a, ax_b, ax_c):
        for t0, t1, _ in ATTACKS:
            ax.axvspan(t0, t1, facecolor=ACCENT, alpha=0.13, edgecolor="none",
                       zorder=1)
        ax.set_xlim(0.0, 24.0)
        ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
        ax.tick_params(labelsize=6.6, length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---- (a) volume ------------------------------------------------------
    ax_a.plot(t, gbps, lw=0.85, color=BLUE, zorder=4)
    ax_a.set_ylabel("offered load\n(Gbps)", fontsize=7.4, linespacing=1.25)
    ax_a.tick_params(labelbottom=False)        # never set_xticklabels([]) here
    for t0, t1, name in ATTACKS:
        ax_a.text(0.5 * (t0 + t1), 1.03, name, transform=
                  ax_a.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=6.4, color=ACCENT, fontweight="bold")
    ax_a.set_ylim(0.0, gbps.max() * 1.16)

    # ---- (b) detector scores ---------------------------------------------
    for k, (name, col) in enumerate(FEATURES):
        ax_b.plot(t, scores[k], lw=0.9, color=col, zorder=4, label=name)
    ax_b.axhline(THR, color=COLORS["rule"], lw=0.8, ls=(0, (2.6, 1.8)),
                 zorder=5)
    ax_b.text(0.15, THR + 0.03, "alert threshold %.2f" % THR, fontsize=6.2,
              color=COLORS["sub"], ha="left", va="bottom")
    ax_b.set_ylim(0.0, 1.30)
    ax_b.set_yticks([0.0, 0.5, 1.0])
    ax_b.set_ylabel("anomaly score\n(-)", fontsize=7.4, linespacing=1.25)
    ax_b.tick_params(labelbottom=False)
    ax_b.legend(fontsize=6.4, frameon=False, loc="upper left", ncol=3,
                handlelength=1.5, columnspacing=1.2, borderpad=0.15,
                bbox_to_anchor=(0.0, 1.04))

    # ---- (c) the alerts those scores emit --------------------------------
    for k, (name, col) in enumerate(FEATURES):
        idx = np.flatnonzero(fired[k])[::4]
        sev = scores[k][idx]
        ax_c.scatter(t[idx], np.full(idx.size, 2 - k) +
                     rng.uniform(-0.13, 0.13, idx.size),
                     s=6.0 + 26.0 * (sev - THR) / (1.0 - THR), c=col,
                     alpha=0.85, edgecolors="white", linewidths=0.25,
                     zorder=4)
    ax_c.set_ylim(-0.7, 2.7)
    ax_c.set_yticks([2, 1, 0])
    ax_c.set_yticklabels([f[0] for f in FEATURES], fontsize=6.4)
    for lab, (_, col) in zip(ax_c.get_yticklabels(), FEATURES):
        lab.set_color(col)
    ax_c.set_xlabel("time of day (h)", fontsize=8.0)
    ax_c.text(0.005, 0.06, "marker area scales with score above threshold",
              transform=ax_c.transAxes, fontsize=6.2, color=COLORS["muted"],
              ha="left", va="bottom")

    # ---- (d) score distribution, normal against attack --------------------
    comb = scores.max(axis=0)
    grid = np.linspace(0.0, 1.05, 320)
    for mask, col, name in ((~truth, BLUE, "benign minutes"),
                            (truth, ACCENT, "incident minutes")):
        kde = gaussian_kde(comb[mask], bw_method=0.28)
        d = kde(grid)
        ax_d.fill_between(grid, 0.0, d, facecolor=col, alpha=0.22,
                          edgecolor="none", zorder=3)
        ax_d.plot(grid, d, lw=1.2, color=col, zorder=4,
                  label="%s (n=%d)" % (name, int(mask.sum())))
    ax_d.axvline(THR, color=COLORS["rule"], lw=0.9, ls=(0, (2.6, 1.8)),
                 zorder=5)
    ax_d.set_xlim(0.0, 1.05)
    ax_d.set_ylim(0.0, ax_d.get_ylim()[1] * 1.22)
    ax_d.set_xlabel("strongest feature score (-)", fontsize=7.4)
    ax_d.set_ylabel("density (-)", fontsize=7.4)
    ax_d.tick_params(labelsize=6.5)
    ax_d.legend(fontsize=6.4, frameon=False, loc="upper center",
                handlelength=1.5, borderpad=0.2)
    for s in ("top", "right"):
        ax_d.spines[s].set_visible(False)
    ax_d.set_title("The overlap is where the false alarms live", fontsize=8.5,
                   fontweight="bold", pad=3.5)

    # ---- (e) confusion-matrix reduction -----------------------------------
    vals = np.array([prec, rec, f1, spec]) * 100.0
    labs = ["precision", "recall", "F1", "specificity"]
    cols = [BLUE, BLUE, GREEN, BLUE]
    ax_e.bar(np.arange(4), vals, width=0.58, color=cols, edgecolor="white",
             linewidth=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax_e.text(i, v + 2.0, "%.1f" % v, ha="center", va="bottom",
                  fontsize=6.4, color=COLORS["sub"])
    ax_e.set_xticks(np.arange(4))
    ax_e.set_xticklabels(labs, fontsize=6.5)
    ax_e.set_ylim(0.0, 112.0)
    ax_e.set_yticks([0, 25, 50, 75, 100])
    ax_e.set_ylabel("minute-level score (%)", fontsize=7.4)
    ax_e.tick_params(labelsize=6.5)
    ax_e.grid(axis="y", lw=0.45, alpha=0.5)
    ax_e.set_axisbelow(True)
    for s in ("top", "right"):
        ax_e.spines[s].set_visible(False)
    ax_e.set_title("Median detection delay %.0f min" % np.nanmedian(delays),
                   fontsize=8.5, fontweight="bold", pad=3.5)

    for a, lab, dx in ((ax_a, "(a)", -0.52), (ax_b, "(b)", -0.52),
                       (ax_c, "(c)", -0.52), (ax_d, "(d)", -0.44),
                       (ax_e, "(e)", -0.44)):
        panel_label_inches(fig, a, lab, dx=dx, dy=0.06)

    w_last = (t >= ATTACKS[2][0]) & (t <= ATTACKS[2][1])
    k_best = int(np.argmax(scores[:, w_last].max(axis=1)))

    savefig(fig, "646_net_anomaly_flow.png")


if __name__ == "__main__":
    main()
