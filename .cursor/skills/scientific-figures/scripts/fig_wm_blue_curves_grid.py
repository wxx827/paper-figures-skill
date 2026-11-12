"""Wm blue curves grid: 4x4 learning-curve grid drawn in one blue ladder.

Sixteen control tasks, four agents each.  Every baseline is a different step of
the batch blue ladder -- no second hue anywhere -- and the single warm accent
marks "Ours".  Each curve is the mean over five seeds with a same-colour +/-1
s.d. band whose width shrinks as training converges.  The score is normalised,
so all sixteen panels share one y range computed over the whole grid.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, style_axes

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
ACCENT = "#b5544a"

METHODS = ["Random-shooting MPC", "Latent-imagination", "Model-free SAC",
           "Ours (unified world model)"]
# Three steps of the ladder, two rungs apart so neighbours stay separable at
# thumbnail size; the fourth series is the only warm colour in the figure.
METHOD_COLORS = [BLUE[4], BLUE[2], BLUE[0], ACCENT]

TASKS = ["Cartpole swing-up", "Reacher hard", "Cheetah run", "Walker walk",
         "Walker run", "Hopper hop", "Quadruped walk", "Finger turn",
         "Cup catch", "Pendulum swing", "Acrobot swing-up", "Fish swim",
         "Humanoid stand", "Manipulator lift", "Swimmer 6-link",
         "Point-mass hard"]
N_SEEDS = 5
N_STEPS = 110
X_MAX = 200.0          # million environment steps


_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def _word(k):
    """Spell small counts out, the way a paper title would."""
    return _WORDS[k] if k < len(_WORDS) else str(k)


def _leader_count(data, tail=8):
    """How many tasks the last series finishes ahead on, from the plotted means."""
    ours = len(METHODS) - 1
    return sum(int(np.argmax([mu[-tail:].mean() for mu, _ in panel]) == ours)
               for panel in data.values())


def _runs(rng, x, asym, rate):
    """Five seeds of one agent: saturating mean + AR(1) exploration noise."""
    out = np.empty((N_SEEDS, x.size))
    for s in range(N_SEEDS):
        a = asym * rng.normal(1.0, 0.055)
        r = max(rate * rng.normal(1.0, 0.22), 0.004)
        mean = a * (1.0 - np.exp(-r * x))
        e = np.zeros(x.size)
        for i in range(1, x.size):
            e[i] = 0.62 * e[i - 1] + rng.normal(0.0, 1.0)
        e /= max(np.abs(e).max(), 1e-9)
        # Band is wide while the policy still explores, tight once converged.
        sd = 0.11 * max(a, 0.12) * (1.35 - 0.85 * x / X_MAX)
        out[s] = mean + e * sd * 2.4
    return np.clip(out, -0.03, 1.12)


def gen_data(seed=2518):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import pandas as pd
        df = pd.read_csv("data/wm_blue_curves_grid.csv")
        return df.pivot_table(index="step", columns=["task", "method"],
                              values="score")
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, X_MAX, N_STEPS)
    data = {}
    for t, task in enumerate(TASKS):
        ceiling = rng.uniform(0.30, 0.98)
        # Baseline ordering is not the same in every task, which is what real
        # benchmark sweeps look like.
        factors = rng.permutation([0.55, 0.72, 0.88])
        asyms = list(ceiling * factors) + [min(ceiling * 1.02, 1.03)]
        # In two of the sixteen tasks a baseline still wins.
        if t in (5, 11):
            asyms[3] = ceiling * 0.66
        rates = 0.055 * rng.uniform(0.45, 1.9, 4)
        panel = []
        for m in range(4):
            runs = _runs(rng, x, asyms[m], rates[m])
            panel.append((runs.mean(0), runs.std(0)))
        data[task] = panel
    return x, data


def main():
    set_pub_style()
    x, data = gen_data()

    # Iron rule: sweep the whole grid before drawing, then use one y range.
    lo = min((mu - sd).min() for p in data.values() for mu, sd in p)
    hi = max((mu + sd).max() for p in data.values() for mu, sd in p)
    span = hi - lo
    ylim = (lo - 0.04 * span, hi + 0.06 * span)

    fig, axes = plt.subplots(4, 4, figsize=(10.4, 8.0), sharex=True,
                             sharey=True)
    for k, (ax, task) in enumerate(zip(axes.ravel(), TASKS)):
        ax.grid(axis="y", lw=0.5, color="#e6eaee", zorder=0)
        for m, (mu, sd) in enumerate(data[task]):
            col = METHOD_COLORS[m]
            ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.16, lw=0,
                            zorder=2 + m)
            ax.plot(x, mu, color=col, lw=1.6, zorder=6 + m,
                    solid_joinstyle="round")
        ax.set_title(task, fontsize=8.5, fontweight="bold", pad=3.5)
        ax.set_xlim(0, X_MAX)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.tick_params(labelsize=7.0)
        if k % 4 == 0:
            ax.set_ylabel("Normalised score", fontsize=8.0)
        if k >= 12:
            ax.set_xlabel("Env steps (M)", fontsize=8.0)
        style_axes(ax)

    handles = [Line2D([], [], color=c, lw=2.2, label=m)
               for m, c in zip(METHODS, METHOD_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.0,
               frameon=False, bbox_to_anchor=(0.5, -0.004),
               handlelength=2.0, columnspacing=2.4)
    # Counted from the curves that were just drawn, so the claim cannot go
    # stale if the data behind them changes.
    fig.tight_layout(rect=[0, 0.058, 1, 0.955], h_pad=1.6, w_pad=1.4)
    savefig(fig, "1113_wm_blue_curves_grid.png")


if __name__ == "__main__":
    main()
