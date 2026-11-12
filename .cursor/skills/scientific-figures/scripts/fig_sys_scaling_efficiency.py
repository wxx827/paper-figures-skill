"""Strong and weak scaling of one solver, with the serial fraction fitted.

The main panel is the claim: the measured strong-scaling curve leaves the ideal
linear line early, and the Amdahl law fitted to its rising branch says why -
a small serial fraction caps the achievable speedup no matter how many cores
are thrown at it.  The serial fraction is obtained by regressing inverse
speedup on inverse core count over the plotted points, never assumed, and the
core count where parallel efficiency last holds 70 percent is located in the
same array.  The two subordinate panels carry the evidence: efficiency against
core count, and the weak-scaling step time that should stay flat but does not.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"
BLUE = "#3b5b92"
RED = "#b23a48"
GOLD = "#c9a227"
GREEN = "#2e7d5b"

CORES = 2 ** np.arange(0, 11)          # 1 .. 1024
SERIAL = 0.005                         # ground truth used to synthesise runs
COMM = 0.0004                          # per-doubling communication penalty
EFF_TARGET = 0.70
FIT_UPTO = 128                         # rising branch used for the Amdahl fit


def strong_scaling(seed=3):
    steps = np.log2(CORES)
    t = SERIAL + (1.0 - SERIAL) / CORES + COMM * steps ** 1.6 / 10.0
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, 0.012, CORES.size)
    jitter[0] = 0.0                    # the one-core baseline is the reference
    t = t * (1.0 + jitter)
    speedup = t[0] / t
    return speedup, speedup / CORES


def fit_amdahl(speedup):
    """Regress 1/S on 1/N over the rising branch; intercept is the serial part."""
    m = CORES <= FIT_UPTO
    x, y = 1.0 / CORES[m], 1.0 / speedup[m]
    slope, intercept = np.polyfit(x, y, 1)
    p = float(np.clip(intercept, 1e-4, 0.5))
    return p, 1.0 / p


def weak_scaling(seed=8):
    rng = np.random.default_rng(seed)
    base = 42.0                        # ms per step at one core
    t = base * (1.0 + 0.011 * np.log2(CORES) ** 1.25)
    return t * (1.0 + rng.normal(0.0, 0.012, CORES.size))


def main_panel(ax, speedup, eff, p, ceiling):
    grid = np.logspace(0, np.log10(CORES[-1] * 1.35), 260)
    ax.plot(grid, grid, color=GREY, lw=1.0, ls=(0, (4, 2.6)), zorder=2)
    ax.text(CORES[-1] * 0.55, CORES[-1] * 0.72, "ideal linear", rotation=41,
            rotation_mode="anchor", transform_rotates_text=True, ha="center",
            va="bottom", fontsize=6.8, color=GREY)

    amdahl = 1.0 / (p + (1.0 - p) / grid)
    ax.plot(grid, amdahl, color=GOLD, lw=1.4, ls=(0, (5, 2.4)), zorder=3)
    y_top = CORES[-1] * 1.5
    ax.axhline(ceiling, color=GOLD, lw=0.8, ls=(0, (1.6, 2.2)), zorder=2)
    ax.text(1.15, min(ceiling * 1.06, y_top * 0.80),
            f"Amdahl ceiling {ceiling:.0f}x at p = {p * 100:.2f} % serial",
            ha="left", va="bottom", fontsize=7.0, color=GOLD)

    ax.plot(CORES, speedup, marker="o", ms=4.0, lw=1.8, color=BLUE,
            mfc="white", mew=1.1, zorder=5, label="measured strong scaling")

    ok = np.flatnonzero(eff >= EFF_TARGET)
    knee = int(CORES[ok[-1]])
    ax.plot([knee], [speedup[ok[-1]]], marker="D", ms=6.0, mfc="white",
            mec=RED, mew=1.5, zorder=7)
    ax.annotate(f"efficiency drops below {EFF_TARGET * 100:.0f} % after "
                f"{knee} cores",
                xy=(knee, speedup[ok[-1]]), xytext=(knee * 0.30,
                                                    speedup[ok[-1]] * 3.6),
                ha="center", va="bottom", fontsize=7.2, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(0.85, CORES[-1] * 1.5)
    ax.set_ylim(0.85, CORES[-1] * 1.5)
    ax.set_xticks(CORES)
    ax.set_xticklabels([str(c) for c in CORES], fontsize=7.0)
    ax.set_yticks(CORES)
    ax.set_yticklabels([str(c) for c in CORES], fontsize=7.0)
    ax.set_xlabel("cores (log2 scale)")
    ax.set_ylabel("speedup over one core (x, log2 scale)")
    ax.grid(lw=0.5, alpha=0.6)
    ax.legend(loc="lower right", fontsize=7.2, frameon=False,
              handlelength=1.8, borderaxespad=0.6)
    ax.set_title(f"Amdahl allows {ceiling:.0f}x, communication stops the run "
                 f"at {speedup.max():.0f}x", fontsize=9.0, fontweight="bold",
                 color=INK)
    return knee


def eff_panel(ax, eff, knee):
    ax.plot(CORES, eff * 100.0, marker="o", ms=3.4, lw=1.5, color=BLUE,
            mfc="white", mew=1.0, zorder=4)
    ax.axhline(EFF_TARGET * 100.0, color=RED, lw=0.9, ls=(0, (3, 2.2)),
               zorder=3)
    ax.axvline(knee, color=RED, lw=0.8, ls=(0, (1.6, 2.0)), zorder=3)
    ax.text(1.15, EFF_TARGET * 100.0 + 2.5, f"{EFF_TARGET * 100:.0f} % target",
            ha="left", va="bottom", fontsize=6.5, color=RED)
    ax.text(knee * 0.85, 8.0, f"{knee} cores", ha="right", va="bottom",
            fontsize=6.5, color=RED)
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.85, CORES[-1] * 1.2)
    ax.set_ylim(0.0, 108.0)
    ax.set_xticks(CORES[::2])
    ax.set_xticklabels([str(c) for c in CORES[::2]], fontsize=6.5)
    ax.set_xlabel("cores (log2 scale)", fontsize=7.4)
    ax.set_ylabel("parallel efficiency (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Efficiency, the same data per core", fontsize=8.2,
                 fontweight="bold", color=INK)


def weak_panel(ax, t_weak):
    ax.plot(CORES, t_weak, marker="s", ms=3.4, lw=1.5, color=GREEN,
            mfc="white", mew=1.0, zorder=4)
    ax.axhline(t_weak[0], color=GREY, lw=0.9, ls=(0, (4, 2.4)), zorder=3)
    drift = t_weak[-1] / t_weak[0] - 1.0
    ax.text(1.15, t_weak.max() * 1.04,
            f"step time drifts +{drift * 100:.0f} % by {CORES[-1]} cores",
            ha="left", va="top", fontsize=6.8, color=GREEN)
    ax.text(1.15, t_weak[0] * 0.985, "constant-work ideal", ha="left",
            va="top", fontsize=6.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.85, CORES[-1] * 1.2)
    ax.set_ylim(t_weak.min() * 0.94, t_weak.max() * 1.08)
    ax.set_xticks(CORES[::2])
    ax.set_xticklabels([str(c) for c in CORES[::2]], fontsize=6.5)
    ax.set_xlabel("cores (log2 scale)", fontsize=7.4)
    ax.set_ylabel("time per step (ms)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Weak scaling: work per core held fixed", fontsize=8.2,
                 fontweight="bold", color=INK)
    return drift


def main():
    set_pub_style()
    speedup, eff = strong_scaling()
    p, ceiling = fit_amdahl(speedup)
    t_weak = weak_scaling()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_eff = fig.add_subplot(gs[0, 1])
    ax_weak = fig.add_subplot(gs[1, 1])

    knee = main_panel(ax_main, speedup, eff, p, ceiling)
    eff_panel(ax_eff, eff, knee)
    drift = weak_panel(ax_weak, t_weak)

    fig.tight_layout(rect=[0, 0.058, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_eff, ax_weak], dx=-0.46, dy=0.12)
    savefig(fig, "783_sys_scaling_efficiency.png")


if __name__ == "__main__":
    main()
