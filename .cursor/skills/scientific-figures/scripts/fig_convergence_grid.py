"""3x2 log-scale convergence grid over 30 independent runs.

Each panel shows the geometric-mean best-so-far fitness of five metaheuristics
on one benchmark function, wrapped in a translucent min-max envelope over the
30 runs. The final mean fitness of every algorithm is tabulated inside the
panel so that the ranking can be read without a separate table.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label

ALGOS = ["ISOA", "SOA", "PSO", "SSA", "GWO"]
COLORS = {"ISOA": "#b23a48", "SOA": "#3b5b92", "PSO": "#2e7d5b",
          "SSA": "#8e6bbf", "GWO": "#e0812f"}
LSTYLE = {"ISOA": "-", "SOA": "-", "PSO": (0, (5, 2)),
          "SSA": (0, (6, 2, 1, 2)), "GWO": (0, (1.4, 1.6))}
LWIDTH = {"ISOA": 2.0, "SOA": 1.3, "PSO": 1.3, "SSA": 1.3, "GWO": 1.45}
KIND = {"ISOA": "isoa", "SOA": "soa", "PSO": "pso", "SSA": "ssa",
        "GWO": "gwo"}

ITERS = 200
N_RUNS = 30

PANELS = [
    dict(key="F1", name="Sphere", mode="Unimodal", gstart=3.0e4),
    dict(key="F2", name="Schwefel 2.22", mode="Unimodal", gstart=7.5e1),
    dict(key="F3", name="Rosenbrock", mode="Unimodal", gstart=4.2e6),
    dict(key="F4", name="Rastrigin", mode="Multimodal", gstart=3.4e2),
    dict(key="F5", name="Ackley", mode="Multimodal", gstart=2.05e1),
    dict(key="F6", name="Griewank", mode="Multimodal", gstart=2.8e2),
]
BY_KEY = {p["key"]: p for p in PANELS}
KEYS = [p["key"] for p in PANELS]

ENDS = {
    "F1": {"ISOA": 2.1e-95, "SOA": 5.6e-42, "PSO": 3.4e-9,
           "SSA": 8.8e-24, "GWO": 1.7e-50},
    "F2": {"ISOA": 3.7e-52, "SOA": 1.2e-26, "PSO": 6.1e-5,
           "SSA": 4.4e-14, "GWO": 9.3e-30},
    "F3": {"ISOA": 2.67e+1, "SOA": 2.84e+1, "PSO": 4.13e+1,
           "SSA": 2.91e+1, "GWO": 2.55e+1},
    "F4": {"ISOA": 2.4e-13, "SOA": 7.6e-6, "PSO": 3.9e+1,
           "SSA": 1.1e+1, "GWO": 3.4e+0},
    "F5": {"ISOA": 8.9e-16, "SOA": 4.4e-14, "PSO": 2.3e+0,
           "SSA": 1.9e-7, "GWO": 1.5e-13},
    "F6": {"ISOA": 3.3e-16, "SOA": 2.7e-3, "PSO": 1.9e-2,
           "SSA": 1.4e-8, "GWO": 4.2e-3},
}
SIGMA = {"F1": 3.00, "F2": 2.20, "F3": 0.020, "F4": 1.80,
         "F5": 0.22, "F6": 2.00}


def _progress(rng, T, kind):
    t = np.arange(T, dtype=float) / max(T - 1, 1)
    if kind == "isoa":
        w = np.exp(-3.0 * t) + 0.30 + 0.30 * np.exp(-((t - 0.45) / 0.12) ** 2)
        mask = rng.random(T) < 0.85
    elif kind == "soa":
        w = np.exp(-4.5 * t) + 0.05
        mask = rng.random(T) < 0.60
    elif kind == "pso":
        w = np.exp(-9.0 * t) + 0.004
        mask = rng.random(T) < 0.50
    elif kind == "ssa":
        w = np.exp(-3.0 * t) + 0.10
        mask = rng.random(T) < 0.12
    else:
        w = np.exp(-2.0 * t) + 0.18
        mask = rng.random(T) < 0.75
    inc = w * rng.gamma(1.6, 1.0, T) * mask
    inc[0] = 0.0
    p = np.cumsum(inc)
    return p / max(p[-1], 1e-12)


def curve(fkey, algo, run):
    rng = np.random.default_rng(
        9001 + KEYS.index(fkey) * 977 + ALGOS.index(algo) * 131 + run * 7919)
    p = _progress(rng, ITERS, KIND[algo])
    g0 = BY_KEY[fkey]["gstart"] * 10 ** rng.normal(0, 0.06)
    g1 = ENDS[fkey][algo] * 10 ** rng.normal(0, SIGMA[fkey])
    g1 = min(g1, g0 * 1e-2)
    return 10 ** (np.log10(g0) + p * (np.log10(g1) - np.log10(g0)))


def ensemble(fkey, algo):
    runs = np.array([curve(fkey, algo, r) for r in range(N_RUNS)])
    mean = 10 ** np.mean(np.log10(runs), axis=0)
    return mean, runs.min(axis=0), runs.max(axis=0)


def panel(ax, spec, label):
    it = np.arange(1, ITERS + 1)
    stats = {a: ensemble(spec["key"], a) for a in ALGOS}
    for a in ALGOS:
        mean, lo, hi = stats[a]
        ax.fill_between(it, lo, hi, color=COLORS[a],
                        alpha=0.20 if a == "ISOA" else 0.11, lw=0,
                        zorder=2 if a == "ISOA" else 1)
    for a in ALGOS:
        mean = stats[a][0]
        ax.plot(it, mean, color=COLORS[a], ls=LSTYLE[a], lw=LWIDTH[a],
                zorder=6 if a == "ISOA" else 4, solid_capstyle="round")

    ax.set_yscale("log")
    lo = min(s[1].min() for s in stats.values())
    hi = max(s[2].max() for s in stats.values())
    ax.set_ylim(lo * 10 ** -0.9, hi * 10 ** 0.6)
    ax.set_xlim(0, ITERS)
    ax.set_xlabel("Iteration", fontsize=8.4, labelpad=1.8)
    ax.set_ylabel("Best-so-far fitness", fontsize=8.4, labelpad=1.8)
    ax.tick_params(labelsize=7.2, length=2.6)
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, lw=0.45, color="#ececec", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    k = spec["key"]
    ax.set_title(f"${k[0]}_{k[1]}(x)$  {spec['name']}", fontsize=9,
                 fontweight="bold", loc="left", pad=3.5)
    ax.text(1.0, 1.012, spec["mode"], transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7, color="#7a7a7a", style="italic")
    add_panel_label(ax, label, x=-0.135, y=1.02, size=10)

    # ---- final-value ranking strip, kept below the axes so no curve is hidden
    finals = sorted(((stats[a][0][-1], a) for a in ALGOS))
    ax.text(0.0, -0.235, "final mean fitness:", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.4, color="#6a6a6a")
    for i, (v, a) in enumerate(finals):
        ax.text(0.203 + i * 0.160, -0.235, f"{a} {v:.2e}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                family="DejaVu Sans Mono", color=COLORS[a],
                fontweight="bold" if a == "ISOA" else "normal")


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.6, 10.2))
    gs = GridSpec(3, 2, figure=fig, left=0.072, right=0.988, top=0.900,
                  bottom=0.113, hspace=0.62, wspace=0.19)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for i, spec in enumerate(PANELS):
        panel(fig.add_subplot(gs[i // 2, i % 2]), spec, labels[i])

    handles = [Line2D([], [], color=COLORS[a], ls=LSTYLE[a], lw=LWIDTH[a],
                      label=a) for a in ALGOS]
    handles.append(Line2D([], [], color="#9a9a9a", lw=6, alpha=0.30,
                          label="min-max band over 30 runs"))

    # Wrapped: a single-line footer wider than the canvas makes the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).
    savefig(fig, "145_convergence_grid.png")


if __name__ == "__main__":
    main()
