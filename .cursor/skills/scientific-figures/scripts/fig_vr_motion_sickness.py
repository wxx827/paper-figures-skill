"""Discomfort as the integral of sensory conflict, not of any single burst.

The main panel keeps three quantities on their own axes rather than rescaling
them onto a shared one: linear acceleration in metres per second squared,
angular velocity in degrees per second, and a discomfort score that is the
running output of a leaky integrator driven by both.  That is why the red
bands do not line up with the tallest motion spikes - the score crosses its
threshold after sustained conflict, and the delay between the two is the point
of the figure.

The auxiliary panels are computed from one synthetic cohort rather than
asserted: correlations are Pearson coefficients with their own significance
stars, and each mitigation is a paired within-subject contrast with a
standard error and a paired t-test.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pubstyle import set_pub_style, savefig, sig_stars
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RULE = "#333333"
GUIDE = "#808080"
GRID = "#d9d9d9"
ACCENT = "#b23a48"

BLUE = "#3b5b92"
GREEN = "#2e7d5b"

T_END = 20.0          # session length, minutes
DT = 0.01
THRESHOLD = 6.0       # discomfort score at which a session is aborted
K_UP, K_DECAY = 6.8, 0.35
A_MAX, W_MAX = 3.5, 85.0

FACTORS = [
    ("motion-to-photon latency (ms)", "latency"),
    ("virtual turn rate (deg/s)", "turn"),
    ("displayed field of view (deg)", "fov"),
    ("session duration (min)", "duration"),
    ("frame rate (fps)", "fps"),
]
MITIGATIONS = [
    ("Static rest frame", 1.55),
    ("Vignette on turns", 1.20),
    ("Snap turning", 0.95),
    ("90 to 120 fps", 0.62),
    ("Seated play", 0.28),
]


def bursts(t, centres, widths, amps, freqs, rng):
    y = np.zeros_like(t)
    for c, w, a, f in zip(centres, widths, amps, freqs):
        env = np.exp(-0.5 * ((t - c) / w) ** 2)
        y += a * env * (0.62 + 0.38 * np.sin(2.0 * np.pi * f * (t - c)))
    return y + rng.normal(0.0, 0.014 * np.max(np.abs(y) + 1e-9), t.size)


def session(rng):
    t = np.arange(0.0, T_END + DT, DT)
    accel = bursts(t, [2.4, 6.1, 9.0, 12.6, 15.4, 18.2],
                   [0.72, 0.55, 0.95, 0.62, 1.10, 0.70],
                   [1.9, 2.6, 3.1, 2.2, 3.4, 2.8],
                   [0.55, 0.72, 0.61, 0.88, 0.47, 0.76], rng)
    omega = bursts(t, [3.1, 6.4, 8.6, 13.1, 15.9, 18.6],
                   [0.50, 0.80, 0.70, 0.95, 1.25, 0.55],
                   [38.0, 61.0, 47.0, 72.0, 80.0, 55.0],
                   [0.42, 0.63, 0.51, 0.38, 0.47, 0.70], rng)
    accel = np.clip(np.abs(accel), 0.0, A_MAX)
    omega = np.clip(np.abs(omega), 0.0, W_MAX)

    conflict = 0.55 * accel / A_MAX + 0.45 * omega / W_MAX
    score = np.zeros_like(t)
    for k in range(1, t.size):
        d = K_UP * conflict[k - 1] - K_DECAY * score[k - 1]
        score[k] = min(10.0, max(0.0, score[k - 1] + d * DT))
    return t, accel, omega, conflict, score


def over_segments(t, mask):
    """Contiguous [start, end] spans where ``mask`` is True."""
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)]


def cohort(rng, n=72):
    d = dict(
        latency=rng.uniform(8.0, 45.0, n),
        turn=rng.uniform(20.0, 90.0, n),
        fov=rng.uniform(90.0, 130.0, n),
        duration=rng.uniform(5.0, 30.0, n),
        fps=rng.choice([72.0, 90.0, 120.0], n),
    )
    score = (1.1 + 0.105 * d["latency"] + 0.030 * d["turn"]
             + 0.026 * d["fov"] + 0.052 * d["duration"]
             - 0.038 * d["fps"] + rng.normal(0.0, 0.85, n))
    d["score"] = np.clip(score, 0.0, 10.0)
    return d


def main():
    set_pub_style()
    rng = np.random.default_rng(6520)
    t, accel, omega, conflict, score = session(rng)

    over = score >= THRESHOLD
    spans = over_segments(t, over)
    t_first = t[np.argmax(over)] if over.any() else np.nan
    t_over = float(np.sum(over) * DT)
    peak = float(score.max())
    t_peak = float(t[int(np.argmax(score))])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.92],
                          left=0.058, right=0.965, top=0.880, bottom=0.150,
                          wspace=0.62, hspace=0.62)

    # ---- (a) motion and discomfort on their own axes ----------------------
    ax = fig.add_subplot(gs[:, 0])
    axd = ax.twinx()
    axw = ax.twinx()
    axw.spines["right"].set_position(("outward", 34))

    for lo, hi in spans:
        ax.axvspan(lo, hi, color=ACCENT, alpha=0.10, lw=0, zorder=0)

    ax.fill_between(t, 0.0, accel, color=BLUE, alpha=0.16, lw=0, zorder=2)
    ax.plot(t, accel, color=BLUE, lw=0.9, alpha=0.95, zorder=3)
    axw.plot(t, omega, color=GREEN, lw=1.0, alpha=0.85, zorder=2)
    axd.plot(t, score, color=ACCENT, lw=1.9, zorder=5)
    axd.axhline(THRESHOLD, color=GUIDE, lw=1.0, ls=(0, (3.2, 2.2)), zorder=4)
    axd.text(0.30, THRESHOLD + 0.12, f"abort threshold {THRESHOLD:.1f}",
             ha="left", va="bottom", fontsize=6.5, color=GUIDE)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(0.0, A_MAX * 1.05)
    axw.set_ylim(0.0, W_MAX * 1.05)
    axd.set_ylim(0.0, 10.4)
    ax.set_xlabel("session time (min)", fontsize=8.0)
    ax.set_ylabel(r"linear acceleration (m$\cdot$s$^{-2}$)", fontsize=8.0,
                  color=BLUE)
    axw.set_ylabel("angular velocity (deg/s)", fontsize=8.0, color=GREEN)
    axd.set_ylabel("discomfort score (-)", fontsize=8.0, color=ACCENT)
    for a_, col in ((ax, BLUE), (axw, GREEN), (axd, ACCENT)):
        a_.tick_params(axis="y", colors=col, labelsize=6.8)
    ax.tick_params(axis="x", labelsize=6.8)
    for a_, side, col in ((ax, "left", BLUE), (axw, "right", GREEN),
                          (axd, "right", ACCENT)):
        a_.spines[side].set_visible(True)
        a_.spines[side].set_color(col)
        a_.spines[side].set_linewidth(0.8)
    for a_ in (axd, axw):
        a_.spines["top"].set_visible(False)
        a_.spines["left"].set_visible(False)
        a_.spines["bottom"].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.5)

    axd.annotate(f"peak {peak:.1f} at {t_peak:.1f} min",
                 xy=(t_peak, peak), xytext=(-8.0, -10.0),
                 textcoords="offset points", ha="right", va="top",
                 fontsize=6.6, color=ACCENT)
    axd.annotate(f"first crossing {t_first:.1f} min", xy=(t_first, THRESHOLD),
                 xytext=(-8.0, 16.0), textcoords="offset points", ha="right",
                 va="bottom", fontsize=6.6, color=ACCENT,
                 arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7))
    ax.set_title("Discomfort lags the motion that causes it", fontsize=9.0,
                 fontweight="bold", color=INK, pad=6)
    ax.text(0.015, 0.965,
            f"{t_over:.1f} min of {T_END:.0f} above threshold "
            f"({100.0 * t_over / T_END:.0f} % of the session), "
            f"in {len(spans)} episodes",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color=SUB)

    # ---- (b) what correlates with sickness --------------------------------
    axc = fig.add_subplot(gs[0, 1])
    coh = cohort(rng)
    rs, ps = [], []
    for _lab, key in FACTORS:
        r, p = stats.pearsonr(coh[key], coh["score"])
        rs.append(float(r))
        ps.append(float(p))
    order = np.argsort(np.abs(rs))
    y = np.arange(len(FACTORS))
    for k, i in enumerate(order):
        col = ACCENT if rs[i] > 0 else BLUE
        axc.barh(k, rs[i], height=0.60, color=col, edgecolor="white",
                 linewidth=0.5, zorder=3)
        axc.text(rs[i] + 0.025 if rs[i] > 0 else 0.025, k,
                 f"{rs[i]:+.2f} {sig_stars(ps[i])}", ha="left", va="center",
                 fontsize=6.4, color=col)
    axc.axvline(0.0, color=RULE, lw=0.8)
    axc.set_yticks(y)
    axc.set_yticklabels([FACTORS[i][0] for i in order], fontsize=6.6)
    axc.set_xlim(-0.62, 0.82)
    axc.set_ylim(-0.62, len(FACTORS) - 0.38)
    axc.set_xlabel("Pearson r with discomfort score (-)", fontsize=8.0)
    axc.set_title("Latency dominates the cohort", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)
    axc.tick_params(labelsize=6.6)
    axc.grid(True, axis="x", color=GRID, lw=0.5)
    axc.text(0.02, 0.04, f"n = {len(coh['score'])} sessions",
             transform=axc.transAxes, ha="left", va="bottom", fontsize=6.4,
             color=MUTED)

    # ---- (c) what helps ----------------------------------------------------
    axm = fig.add_subplot(gs[1, 1])
    n_pair = 36
    means, sems, stars = [], [], []
    for _name, eff in MITIGATIONS:
        base = rng.normal(6.4, 1.15, n_pair)
        after = base - rng.normal(eff, 0.55, n_pair)
        delta = base - after
        means.append(float(delta.mean()))
        sems.append(float(delta.std(ddof=1) / np.sqrt(n_pair)))
        stars.append(sig_stars(float(stats.ttest_rel(base, after).pvalue)))
    xm = np.arange(len(MITIGATIONS))
    cols = [BLUE if m >= 0.5 else GUIDE for m in means]
    axm.bar(xm, means, width=0.62, color=cols, edgecolor="white",
            linewidth=0.5, yerr=sems, error_kw=dict(ecolor=RULE, lw=0.8,
                                                    capsize=2.2), zorder=3)
    for x_, m, s_, st in zip(xm, means, sems, stars):
        axm.text(x_, m + s_ + 0.07, f"{m:.2f} {st}", ha="center", va="bottom",
                 fontsize=6.4, color=SUB)
    axm.set_xticks(xm)
    axm.set_xticklabels([m[0] for m in MITIGATIONS], fontsize=6.4,
                        rotation=18.0, ha="right")
    axm.set_ylim(0.0, max(means) * 1.32)
    axm.set_ylabel("score reduction (-)", fontsize=8.0)
    axm.set_title("Rest frames beat frame rate", fontsize=8.5,
                  fontweight="bold", color=INK, pad=4)
    axm.tick_params(labelsize=6.6)
    axm.grid(True, axis="y", color=GRID, lw=0.5)
    axm.text(0.98, 0.95, f"paired, n = {n_pair} per arm, bars are SEM",
             transform=axm.transAxes, ha="right", va="top", fontsize=6.4,
             color=MUTED)


    panel_labels_inches(fig, [ax, axc, axm], dx=-0.40, dy=0.28)
    savefig(fig, "652_vr_motion_sickness.png")


if __name__ == "__main__":
    main()
