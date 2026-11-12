"""Return curves for five RL algorithms on six continuous-control suites.

2 x 3 grid.  Every panel shows six seeds as hairlines, the seed median as the
thick line and the inter-quartile range as a band.  Raw return scales differ by
three orders of magnitude across suites, so the plotted quantity is the
expert-normalised return (random policy = 0, expert = 1); that makes one shared
y range honest, and each panel prints the raw return its 1.0 corresponds to.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label
from figkit import shared_limits

ALGOS = ["DQN", "PPO", "TD3", "SAC", "Ours"]
COL = {"DQN": "#8e6bbf", "PPO": "#808080", "TD3": "#c9a227",
       "SAC": "#2e7d5b", "Ours": "#3b5b92"}
LW = {"DQN": 1.3, "PPO": 1.3, "TD3": 1.3, "SAC": 1.3, "Ours": 2.0}

# suite -> (budget in millions of env steps, random return, expert return)
SUITES = [
    ("HalfCheetah-v4", 3.0, -280.0, 11800.0),
    ("Walker2d-v4", 3.0, 1.6, 5400.0),
    ("Ant-v4", 5.0, -60.0, 6900.0),
    ("Humanoid-v4", 8.0, 120.0, 9200.0),
    ("Hopper-v4", 2.0, 18.0, 3700.0),
    ("Quadruped-run", 4.0, 4.0, 940.0),
]

# algorithm -> (asymptote as a fraction of expert, learning-rate constant)
SPEC = {
    "DQN": (0.44, 0.55),
    "PPO": (0.62, 0.38),
    "TD3": (0.78, 0.75),
    "SAC": (0.85, 0.95),
    "Ours": (0.97, 1.85),
}
N_SEED = 6
GRID = "#eeeeee"


def suite_runs(rng, x, tau, wobble):
    """{algo: (n_seed, n_step) normalised returns} for one suite."""
    out = {}
    for algo in ALGOS:
        asym, rate = SPEC[algo]
        # per-suite difficulty tilt, so the ranking is not identical everywhere
        asym = float(np.clip(asym * rng.normal(1.0, 0.055), 0.05, 1.02))
        runs = np.empty((N_SEED, x.size))
        for s in range(N_SEED):
            r = rate * rng.normal(1.0, 0.17) / tau
            a = asym * rng.normal(1.0, 0.045)
            mean = a * (1.0 - np.exp(-max(r, 0.05) * x))
            e = np.zeros(x.size)
            for i in range(1, x.size):
                e[i] = 0.88 * e[i - 1] + rng.normal(0.0, 1.0)
            e = e / max(np.abs(e).max(), 1e-9)
            sd = wobble * (1.0 - 0.55 * x / x[-1])
            runs[s] = mean + e * sd
        out[algo] = runs
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4340)

    data, grids = [], []
    for name, budget, lo, hi in SUITES:
        x = np.linspace(0.0, budget, 240)
        tau = budget / 3.0
        data.append(suite_runs(rng, x, tau, wobble=0.085))
        grids.append(x)

    bands = []
    for runs in data:
        for algo in ALGOS:
            q1, q3 = np.percentile(runs[algo], [25, 75], axis=0)
            bands += [q1, q3]
    ylo, yhi = shared_limits(bands, pad=0.06)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    for k, (ax, (name, budget, lo, hi)) in enumerate(zip(axes.ravel(), SUITES)):
        x, runs = grids[k], data[k]
        ax.set_axisbelow(True)
        ax.grid(color=GRID, lw=0.7, zorder=0)
        ax.axhline(1.0, color="#808080", lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)

        for algo in ALGOS:
            r = runs[algo]
            hot = algo == "Ours"
            for s in range(N_SEED):
                ax.plot(x, r[s], color=COL[algo], lw=0.6, alpha=0.30,
                        zorder=2, solid_capstyle="round")
            q1, q3 = np.percentile(r, [25, 75], axis=0)
            ax.fill_between(x, q1, q3, color=COL[algo],
                            alpha=0.22 if hot else 0.14, lw=0,
                            zorder=4 if hot else 3)
            ax.plot(x, np.median(r, axis=0), color=COL[algo], lw=LW[algo],
                    zorder=8 if hot else 6, solid_capstyle="round")

        med_ours = np.median(runs["Ours"], axis=0)
        best_base = max(np.median(runs[a], axis=0)[-1]
                        for a in ALGOS if a != "Ours")
        ax.text(0.955, 0.055,
                f"Ours {med_ours[-1]:.2f}  ($+${med_ours[-1] - best_base:.2f})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.9,
                color=COL["Ours"], fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#d5e0ea",
                          lw=0.6, alpha=0.94))
        ax.text(0.04, 0.955, f"1.0 $=$ {hi:,.0f} return",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                color="#777777", zorder=9)

        ax.set_xlim(0, budget)
        ax.set_ylim(ylo, yhi)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 4, 5, 10]))
        ax.set_title(name, fontsize=8.6, fontweight="bold", pad=3.5)
        style_axes(ax)
        add_panel_label(ax, f"({'abcdef'[k]})", x=-0.12, y=1.05)
        if k % 3:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Expert-normalised return (-)", fontsize=8)
        if k // 3:
            ax.set_xlabel("Environment steps ($\\times 10^6$)", fontsize=8)

    handles = [Line2D([], [], color=COL[a], lw=LW[a], label=a) for a in ALGOS]
    handles += [
        Line2D([], [], color="#8a8a8a", lw=0.6, alpha=0.6,
               label=f"individual seed ($n={N_SEED}$)"),
        Patch(facecolor="#9fb6cb", alpha=0.5, lw=0, label="IQR across seeds"),
        Line2D([], [], color="#808080", lw=0.8, ls=(0, (3.0, 2.0)),
               label="expert level"),
    ]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=min(len(handles), 5), frameon=False)
    leg.get_texts()[4].set_fontweight("bold")

    fig.tight_layout(rect=[0, 0.025, 1, 0.925], h_pad=2.6, w_pad=2.6)
    savefig(fig, "434_rl_return_curves.png")


if __name__ == "__main__":
    main()
