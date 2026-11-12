"""Usability test summary: task success against effort, SUS, and learning.

Twelve tasks are sampled once with a completion rate, a median time on task
and an error count; the main panel plots them against the acceptance window
the team agreed on (>= 90% complete in <= 90 s), so membership of the green
block is a geometric fact rather than a claim, and the bubbles that miss it
are the only red objects on the page.

The three auxiliary panels use the same cohort: 30 SUS questionnaires with
the 68-point industry mean drawn in, the five sub-scales of those same
questionnaires against the previous build, and five repeated attempts per
participant fitted with a power law whose exponent is printed from the fit.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, panel_labels_inches

W, H = 10.4, 6.4

BLUE = "#3b5b92"        # meets the acceptance window
RED = "#b23a48"         # misses it
GREEN = "#2e7d5b"       # target zone, current build
GREY = "#8a8a8a"        # previous build, secondary
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

TARGET_RATE = 90.0      # percent of participants completing the task
TARGET_TIME = 90.0      # seconds
SUS_BENCH = 68.0        # published industry mean
DIMS = ["Learnability", "Efficiency", "Memorability", "Error recovery",
        "Satisfaction"]


def sample_tasks(rng):
    n = 12
    time = np.exp(rng.normal(np.log(78.0), 0.62, n))
    rate = 100.0 - 26.0 * (time / 130.0) ** 1.25 - rng.uniform(0.0, 9.0, n)
    rate = np.clip(rate, 44.0, 100.0)
    errors = rng.poisson(np.clip((100.0 - rate) / 9.0, 0.2, None))
    return time, rate, errors


def panel_tasks(ax, time, rate, errors):
    ax.add_patch(Rectangle((0.0, TARGET_RATE), TARGET_TIME, 100.0 -
                           TARGET_RATE, facecolor=GREEN, alpha=0.15,
                           edgecolor=GREEN, linewidth=0.8, ls="--", zorder=0))
    ax.text(TARGET_TIME, 100.9,
            f"acceptance window: >= {TARGET_RATE:.0f}% complete "
            f"in <= {TARGET_TIME:.0f} s",
            ha="left", va="center", fontsize=6.6, color=GREEN, zorder=3)

    ok = (rate >= TARGET_RATE) & (time <= TARGET_TIME)
    size = 26.0 + 34.0 * errors
    ax.scatter(time[ok], rate[ok], s=size[ok], color=BLUE, alpha=0.85,
               edgecolors="white", linewidths=0.8, zorder=4)
    ax.scatter(time[~ok], rate[~ok], s=size[~ok], color=RED, alpha=0.85,
               edgecolors="white", linewidths=0.8, zorder=4)
    for i, (x, y) in enumerate(zip(time, rate), start=1):
        ax.annotate(f"T{i}", xy=(x, y), xytext=(0.0, -12.0 - 0.28 *
                    np.sqrt(size[i - 1])), textcoords="offset points",
                    ha="center", va="top", fontsize=6.4, color=SUB)

    for k in (0, 3, 6):
        ax.scatter([], [], s=26.0 + 34.0 * k, color=GREY, alpha=0.7,
                   edgecolors="white", linewidths=0.8,
                   label=f"{k} errors" if k else "no errors")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left",
              labelspacing=1.05, borderpad=0.9, handletextpad=1.1,
              title="bubble area", title_fontsize=6.8)
    ax.set_xlim(0.0, time.max() * 1.14)
    ax.set_ylim(38.0, 104.0)
    ax.set_xlabel("median time on task (s)", fontsize=8.0)
    ax.set_ylabel("participants completing the task (%)", fontsize=8.0)
    ax.set_title("Only the short tasks clear the acceptance window",
                 fontsize=9.0, fontweight="bold")
    ax.tick_params(labelsize=7.2)
    style_axes(ax)
    return ok


def panel_sus(ax, rng):
    sus = np.clip(rng.normal(74.5, 11.5, 30), 20.0, 100.0)
    ax.hist(sus, bins=np.arange(30.0, 105.0, 7.5), color=BLUE,
            edgecolor="white", linewidth=0.6)
    ax.axvline(SUS_BENCH, lw=1.0, ls="--", color=RED)
    ax.axvline(sus.mean(), lw=1.0, color=INK)
    ax.set_ylim(0.0, ax.get_ylim()[1] * 1.30)
    top = ax.get_ylim()[1]
    ax.text(SUS_BENCH - 1.5, top * 0.98, f"industry {SUS_BENCH:.0f}",
            ha="right", va="top", fontsize=6.2, color=RED, rotation=90.0)
    ax.text(sus.mean() + 1.5, top * 0.98, f"mean {sus.mean():.1f}",
            ha="left", va="top", fontsize=6.2, color=INK, rotation=90.0)
    ax.set_xlabel("SUS score (points)", fontsize=8.0)
    ax.set_ylabel(f"participants (n = {sus.size})", fontsize=8.0)
    ax.set_title("SUS beats the benchmark", fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.6)
    style_axes(ax)
    return sus


def panel_radar(ax, rng, sus):
    now = np.clip(sus.mean() + rng.normal(0.0, 7.0, len(DIMS)), 45.0, 96.0)
    was = np.clip(now - rng.uniform(4.0, 21.0, len(DIMS)), 30.0, 95.0)
    ang = np.linspace(0.0, 2.0 * np.pi, len(DIMS), endpoint=False)
    close = np.concatenate([ang, ang[:1]])
    for vals, color, lab, ls in ((was, GREY, "previous build", "--"),
                                 (now, GREEN, "current build", "-")):
        v = np.concatenate([vals, vals[:1]])
        ax.plot(close, v, lw=1.3, color=color, ls=ls, label=lab)
        ax.fill(close, v, color=color, alpha=0.16)
    ax.set_xticks(ang)
    ax.set_xticklabels(DIMS, fontsize=6.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=6.0, color=MUTED)
    ax.set_rlabel_position(198.0)
    ax.tick_params(pad=0.5)
    ax.grid(color="#d9d9d9", lw=0.6)
    ax.set_title("Sub-scales (points)", fontsize=8.5, fontweight="bold",
                 pad=10.0)
    ax.legend(frameon=False, fontsize=6.2, loc="lower center",
              bbox_to_anchor=(0.5, -0.34), ncol=2, columnspacing=1.0,
              handlelength=1.4)
    return now, was


def panel_learning(ax, rng):
    attempts = np.arange(1, 6)
    n_p = 18
    a0 = rng.uniform(120.0, 205.0, (n_p, 1))
    expo = -0.42
    curve = a0 * attempts ** expo * rng.lognormal(0.0, 0.11, (n_p,
                                                              attempts.size))
    mean = curve.mean(axis=0)
    lo = np.percentile(curve, 10, axis=0)
    hi = np.percentile(curve, 90, axis=0)
    ax.fill_between(attempts, lo, hi, color=BLUE, alpha=0.16, lw=0.0,
                    label="p10-p90 across participants")
    for row in curve[:8]:
        ax.plot(attempts, row, lw=0.5, color=GREY, alpha=0.55)
    ax.plot(attempts, mean, marker="o", ms=4.0, lw=1.6, color=BLUE,
            mec="white", mew=0.7, label="cohort mean")
    slope, icpt = np.polyfit(np.log(attempts), np.log(mean), 1)
    ax.plot(attempts, np.exp(icpt) * attempts ** slope, lw=1.0, ls="--",
            color=RED, label=f"power law, exponent {slope:.2f}")
    drop = 100.0 * (1.0 - mean[-1] / mean[0])
    ax.annotate(f"{drop:.0f}% faster by attempt {attempts[-1]}",
                xy=(attempts[-1], mean[-1]), xytext=(-8.0, 22.0),
                textcoords="offset points", ha="right", fontsize=6.8,
                color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=RULE,
                                shrinkA=1.0, shrinkB=3.0))
    ax.set_xticks(attempts)
    ax.set_xlabel("attempt (index)", fontsize=8.0)
    ax.set_ylabel("time on task (s)", fontsize=8.0)
    ax.set_title(f"Learning curve, {n_p} participants", fontsize=8.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=6.4, loc="upper right",
              labelspacing=0.32)
    ax.tick_params(labelsize=6.6)
    style_axes(ax)
    return slope, drop


def main():
    set_pub_style()
    rng = np.random.default_rng(676)
    time, rate, errors = sample_tasks(rng)

    fig = plt.figure(figsize=(W, H))
    ax_t = axes_inches(fig, (0.78, 0.72, 5.24, 4.52))
    ax_s = axes_inches(fig, (6.86, 3.62, 1.48, 1.62))
    ax_r = axes_inches(fig, (8.64, 3.62, 1.62, 1.62), projection="polar")
    ax_l = axes_inches(fig, (6.86, 0.72, 3.40, 1.86))

    ok = panel_tasks(ax_t, time, rate, errors)
    sus = panel_sus(ax_s, rng)
    now, was = panel_radar(ax_r, rng, sus)
    slope, drop = panel_learning(ax_l, rng)


    panel_labels_inches(fig, [ax_t, ax_s], ["(a)", "(b)"], dx=-0.60, dy=0.30)
    panel_labels_inches(fig, [ax_r], ["(c)"], dx=-0.14, dy=0.30)
    panel_labels_inches(fig, [ax_l], ["(d)"], dx=-0.60, dy=0.30)
    savefig(fig, "676_hci_usability_metrics.png")


if __name__ == "__main__":
    main()
