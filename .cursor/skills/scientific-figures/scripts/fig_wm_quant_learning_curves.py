"""World-model learning curves: 3 x 4 task grid with an oversized shared legend.

Twelve embodied / control benchmarks, four agents, five seeds each.  The thick
line is the seed mean, the band is +/-1 s.d. and it narrows as the runs settle.
Raw returns differ by orders of magnitude across suites, so every panel plots
the normalised score (random policy = 0, reference expert = 1); that is what
makes one shared y range across all twelve panels honest.

Synthetic data lives in ``gen_runs`` only -- swap that one function for a real
loader and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from pubstyle import set_pub_style, savefig, style_axes
from figkit import shared_limits

METHODS = ["Ours (WM-8)", "DreamerV3", "IRIS", "PPO"]
# cool-blue family for the baselines, one warm accent reserved for Ours
COL = {"Ours (WM-8)": "#b23a48", "DreamerV3": "#3b5b92",
       "IRIS": "#2a7f7a", "PPO": "#8e6bbf"}
LW = {"Ours (WM-8)": 2.1, "DreamerV3": 1.8, "IRIS": 1.8, "PPO": 1.8}

# method -> (asymptote as a fraction of expert, learning-rate constant)
SPEC = {
    "Ours (WM-8)": (0.96, 2.05),
    "DreamerV3": (0.82, 1.35),
    "IRIS": (0.70, 0.85),
    "PPO": (0.48, 0.55),
}

# (task, env-step budget in millions, difficulty tilt)
TASKS = [
    ("Crafter", 5.0, 1.00),
    ("Minecraft Diamond", 100.0, 0.78),
    ("Walker Run", 2.0, 1.12),
    ("Quadruped Run", 4.0, 0.92),
    ("Cheetah Run", 2.0, 1.08),
    ("Humanoid Walk", 20.0, 0.68),
    ("Atari Boxing", 40.0, 1.05),
    ("Atari Krull", 40.0, 0.88),
    ("Meta-World Assembly", 8.0, 0.82),
    ("RoboDesk Push", 6.0, 0.95),
    ("Memory Maze 9x9", 50.0, 0.72),
    ("Habitat PointNav", 12.0, 1.02),
]

N_SEED = 5
GRID = "#ededed"


def ar_walk(rng, n, rho=0.90):
    """Unit-scaled autocorrelated walk: measured-looking, not analytic."""
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0.0, 1.0)
    return e / max(np.abs(e).max(), 1e-9)


def gen_runs(rng, x, tilt, wobble=0.105):
    """{method: (n_seed, n_step) normalised scores} for one task."""
    out = {}
    tau = x[-1] / 3.0
    for m in METHODS:
        asym, rate = SPEC[m]
        # per-task tilt on top of the global one, so the ranking is not
        # word-for-word identical in all twelve panels
        asym = float(np.clip(asym * tilt * rng.normal(1.0, 0.13), 0.04, 1.03))
        rate = rate * rng.normal(1.0, 0.28)
        # jitter shared by every seed: keeps the mean curve from going smooth
        common = ar_walk(rng, x.size) * wobble * 0.45
        runs = np.empty((N_SEED, x.size))
        for s in range(N_SEED):
            r = rate * rng.normal(1.0, 0.22) / tau
            a = asym * rng.normal(1.0, 0.085)
            mean = a * (1.0 - np.exp(-max(r, 0.04) * x))
            sd = wobble * (1.0 - 0.55 * x / x[-1])
            runs[s] = mean + common + ar_walk(rng, x.size) * sd
        out[m] = runs
    return out


def step_fmt(v, _pos):
    if v <= 0:
        return "0"
    return f"{v:g}M"


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4980)

    grids, data = [], []
    for _, budget, tilt in TASKS:
        x = np.linspace(0.0, budget, 260)
        grids.append(x)
        data.append(gen_runs(rng, x, tilt))

    # one shared y range for all twelve panels, computed before anything is drawn
    envelope = []
    for runs in data:
        for m in METHODS:
            mu = runs[m].mean(axis=0)
            sd = runs[m].std(axis=0, ddof=1)
            envelope += [mu - sd, mu + sd]
    ylo, yhi = shared_limits(envelope, pad=0.05)

    fig = plt.figure(figsize=(11.6, 6.4))
    gs = fig.add_gridspec(3, 4, left=0.052, right=0.808, top=0.885,
                          bottom=0.085, hspace=0.46, wspace=0.16)

    for k, (task, budget, _) in enumerate(TASKS):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        x, runs = grids[k], data[k]
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.axhline(1.0, color="#b3b3b3", lw=0.8, ls=(0, (3.0, 2.2)), zorder=1)

        for m in METHODS:
            hot = m == METHODS[0]
            mu = runs[m].mean(axis=0)
            sd = runs[m].std(axis=0, ddof=1)
            ax.fill_between(x, mu - sd, mu + sd, color=COL[m],
                            alpha=0.22 if hot else 0.18, lw=0,
                            zorder=3 if hot else 2)
            ax.plot(x, mu, color=COL[m], lw=LW[m], zorder=7 if hot else 5,
                    solid_capstyle="round")

        ax.set_xlim(0, budget)
        ax.set_ylim(ylo, yhi)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(step_fmt))
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(task, fontsize=9.0, fontweight="bold", pad=3.5)
        style_axes(ax)
        if k % 4:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Normalised score", fontsize=8.2)
        if k // 4 == 2:
            ax.set_xlabel("Env steps", fontsize=8.6)

    handles = [Line2D([], [], color=COL[m], lw=3.0, solid_capstyle="round",
                      label=m) for m in METHODS]
    handles += [
        Patch(facecolor="#9aa9bd", alpha=0.45, lw=0,
              label="$\\pm$1 s.d. ($n=5$)"),
        Line2D([], [], color="#b3b3b3", lw=1.0, ls=(0, (3.0, 2.2)),
               label="expert level"),
    ]
    leg = fig.legend(handles=handles, loc="center left",
                     bbox_to_anchor=(0.820, 0.5), frameon=False,
                     fontsize=10.5, handlelength=1.4, borderpad=0.6,
                     labelspacing=0.85, handletextpad=0.7)
    leg.get_texts()[0].set_fontweight("bold")
    for t in leg.get_texts()[4:]:
        t.set_fontsize(8.4)
        t.set_color("#555555")

    # The banners sit hard against the canvas edges and the legend is pulled
    # back inside it: nothing here overflows, but `bbox_inches="tight"` crops
    # unused margin as readily as it adds it, and the trimmed 11.42 x 6.19
    # export read 0.542 against a declared 0.552.
    savefig(fig, "1093_wm_quant_learning_curves.png")


if __name__ == "__main__":
    main()
