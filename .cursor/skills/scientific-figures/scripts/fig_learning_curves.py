"""2x3 learning-curve grid: multi-seed reward curves with variability bands.

Mimics RL / training figures: four methods per panel, each drawn as the mean
over five seeds with a +/-1 s.d. band, over a faint "failure" floor region and
a "target" ceiling tint. Shared legend on top, panel labels (a)-(f).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pubstyle import (set_pub_style, PALETTES, savefig, style_axes,
                      add_panel_label)

TRIALS = ["Trial 0", "Trial 1", "Trial 2", "Trial 3"]
COLORS = PALETTES["cb_safe"][:4]
MARKERS = ["o", "s", "^", "D"]
PANELS = ["Franka Emika Panda", "Unitree Go1", "Race Car",
          "Quadrotor", "Allegro Hand", "ANYmal"]
FAIL_C, GOAL_C = "#d9a7a7", "#9dc9ae"
N_SEEDS = 5


def seed_runs(rng, n_ep, floor, ceiling, rate, asym):
    """Five noisy runs of one method: saturating mean + AR(1) noise."""
    x = np.arange(n_ep)
    span = ceiling - floor
    runs = np.empty((N_SEEDS, n_ep))
    for s in range(N_SEEDS):
        r = rate * rng.normal(1.0, 0.18)
        a = asym * rng.normal(1.0, 0.05)
        mean = floor + span * a * (1 - np.exp(-max(r, 0.02) * x))
        # correlated exploration noise that shrinks as the policy converges
        sd = span * 0.075 * (1.25 - 0.7 * x / max(n_ep - 1, 1))
        e = np.zeros(n_ep)
        for i in range(1, n_ep):
            e[i] = 0.55 * e[i - 1] + rng.normal(0, 1)
        runs[s] = mean + e * sd
    return x, runs


def panel(ax, rng, idx, n_ep, ceiling, floor):
    span = ceiling - floor
    ax.axhspan(floor, floor + span * 0.12, color=FAIL_C, alpha=0.35, zorder=0,
               lw=0)
    ax.axhspan(ceiling - span * 0.08, ceiling, color=GOAL_C, alpha=0.35,
               zorder=0, lw=0)
    ax.grid(axis="y", lw=0.5, alpha=0.45)

    # per-panel learning rates / asymptotes: the ranking is not the same
    # everywhere, which is what real benchmark sweeps look like
    rates = 0.10 + 0.10 * rng.random(4)
    asyms = np.clip(rng.normal(0.87, 0.07, 4), 0.62, 0.99)
    best, best_final = 0, -np.inf
    for t, (name, col, mk) in enumerate(zip(TRIALS, COLORS, MARKERS)):
        x, runs = seed_runs(rng, n_ep, floor, ceiling, rates[t], asyms[t])
        mu, sd = runs.mean(0), runs.std(0)
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15, lw=0,
                        zorder=1)
        step = max(2, n_ep // 9)
        ax.plot(x, mu, color=col, lw=1.4, marker=mk, ms=3.4,
                markevery=(t, step), markeredgecolor="white", markeredgewidth=0.5,
                label=name, zorder=3 + t)
        fin = mu[-3:].mean()
        if fin > best_final:
            best, best_final = t, fin
            best_sd = sd[-3:].mean()

    ax.text(0.97, 0.05, f"best: {TRIALS[best]}\n{best_final:.1f} $\\pm$ "
            f"{best_sd:.1f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                      lw=0.6, alpha=0.92), zorder=6)

    ax.set_xlabel("Episode $n$", fontsize=8)
    ax.set_ylabel(r"Return  $\hat{J}(\pi_n)$", fontsize=8)
    ax.set_title(PANELS[idx], fontsize=9, fontweight="bold")
    ax.set_xlim(0, n_ep - 1)
    ax.set_ylim(floor - span * 0.06, ceiling + span * 0.06)
    ax.margins(x=0)
    style_axes(ax)
    add_panel_label(ax, f"({'abcdef'[idx]})", x=-0.16, y=1.02, size=10)


def main():
    set_pub_style()
    rng = np.random.default_rng(211)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    cfgs = [(25, 12, 5), (32, 30, 0), (50, 200, 0),
            (28, 15, 2), (40, 8, 1), (35, 100, 10)]
    for k, (ax, (n_ep, ceil, flr)) in enumerate(zip(axes.ravel(), cfgs)):
        panel(ax, rng, k, n_ep, ceil, flr)

    handles = [Line2D([], [], color=c, marker=m, ms=3.6, lw=1.4,
                      markeredgecolor="white", markeredgewidth=0.5, label=t)
               for t, c, m in zip(TRIALS, COLORS, MARKERS)]
    handles += [Patch(facecolor=GOAL_C, alpha=0.55, label="Target band"),
                Patch(facecolor=FAIL_C, alpha=0.55, label="Failure band")]
    fig.tight_layout(rect=[0, 0.02, 1, 0.945], h_pad=2.4, w_pad=2.4)
    savefig(fig, "28_learning_curves.png")


if __name__ == "__main__":
    main()
