"""3x2 box-plot panel of the final fitness reached in 30 independent runs.

Every panel compares the five algorithms on one benchmark function: boxes over
a logarithmic axis, the raw 30 runs jittered on top, a diamond for the mean,
and Wilcoxon rank-sum brackets that test each competitor against ISOA. The
win / tie / loss symbol (+ / = / -) follows the convention of the swarm
intelligence literature.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ranksums
from pubstyle import set_pub_style, savefig, add_panel_label

ALGOS = ["ISOA", "SOA", "PSO", "SSA", "GWO"]
COLORS = {"ISOA": "#b23a48", "SOA": "#3b5b92", "PSO": "#2e7d5b",
          "SSA": "#8e6bbf", "GWO": "#e0812f"}
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


def final_values(fkey, algo):
    out = []
    for run in range(N_RUNS):
        rng = np.random.default_rng(
            9001 + KEYS.index(fkey) * 977 + ALGOS.index(algo) * 131
            + run * 7919)
        _progress(rng, ITERS, KIND[algo])          # keep the seed stream aligned
        g0 = BY_KEY[fkey]["gstart"] * 10 ** rng.normal(0, 0.06)
        g1 = ENDS[fkey][algo] * 10 ** rng.normal(0, SIGMA[fkey])
        out.append(min(g1, g0 * 1e-2))
    return np.array(out)


def panel(ax, spec, label):
    data = {a: final_values(spec["key"], a) for a in ALGOS}
    rng = np.random.default_rng(4242 + KEYS.index(spec["key"]))
    pos = np.arange(1, len(ALGOS) + 1)

    bp = ax.boxplot([data[a] for a in ALGOS], positions=pos, widths=0.56,
                    patch_artist=True, showfliers=False, whis=1.5,
                    medianprops=dict(color="#1a1a1a", lw=1.1),
                    whiskerprops=dict(lw=0.8, color="#555555"),
                    capprops=dict(lw=0.8, color="#555555"))
    for a, box in zip(ALGOS, bp["boxes"]):
        box.set_facecolor(COLORS[a])
        box.set_alpha(0.32 if a != "ISOA" else 0.48)
        box.set_edgecolor(COLORS[a])
        box.set_linewidth(1.6 if a == "ISOA" else 0.9)

    for i, a in enumerate(ALGOS):
        v = data[a]
        x = pos[i] + rng.uniform(-0.17, 0.17, v.size)
        ax.scatter(x, v, s=5.0, color=COLORS[a], alpha=0.55, lw=0,
                   zorder=4)
        ax.scatter([pos[i]], [10 ** np.mean(np.log10(v))], marker="D", s=22,
                   facecolor="white", edgecolor=COLORS[a], linewidths=1.1,
                   zorder=6)

    ax.set_yscale("log")
    allv = np.concatenate([data[a] for a in ALGOS])
    lo, hi = allv.min(), allv.max()
    # reserve the top third of the panel for the significance brackets
    bot = np.log10(lo) - 0.06 * np.log10(hi / lo)
    ax.set_ylim(10 ** bot, 10 ** (bot + (np.log10(hi) - bot) / 0.655))
    ax.set_xlim(0.45, len(ALGOS) + 0.55)
    ax.set_xticks(pos, ALGOS, fontsize=7.6)
    for tick, a in zip(ax.get_xticklabels(), ALGOS):
        tick.set_color(COLORS[a])
        if a == "ISOA":
            tick.set_fontweight("bold")
    ax.set_ylabel("Final fitness (30 runs)", fontsize=8.2, labelpad=1.8)
    ax.tick_params(axis="y", labelsize=7.0, length=2.6)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, axis="y", lw=0.45, color="#ececec", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    k = spec["key"]
    ax.set_title(f"${k[0]}_{k[1]}(x)$  {spec['name']}", fontsize=9,
                 fontweight="bold", loc="left", pad=3.5)
    ax.text(1.0, 1.012, spec["mode"], transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7, color="#7a7a7a", style="italic")
    add_panel_label(ax, label, x=-0.13, y=1.02, size=10)

    # ---- Wilcoxon rank-sum of every competitor against ISOA --------------
    trans = ax.get_xaxis_transform()          # x in data, y in axes fraction
    wtl = [0, 0, 0]
    for i, a in enumerate(ALGOS[1:], start=1):
        p = ranksums(data["ISOA"], data[a]).pvalue
        if p >= 0.05:
            sym, col, wtl[1] = "=", "#7a7a7a", wtl[1] + 1
        elif np.median(data["ISOA"]) < np.median(data[a]):
            sym, col, wtl[0] = "+", "#b23a48", wtl[0] + 1
        else:
            sym, col, wtl[2] = "\u2212", "#2a6f9e", wtl[2] + 1
        y = 0.705 + (i - 1) * 0.083
        ax.plot([pos[0], pos[0], pos[i], pos[i]],
                [y - 0.020, y, y, y - 0.020], transform=trans, lw=0.8,
                color="#666666", zorder=5, solid_joinstyle="miter")
        ax.text((pos[0] + pos[i]) / 2.0, y + 0.006,
                f"{sym}   $p$ = {p:.1e}", transform=trans, ha="center",
                va="bottom", fontsize=6.0, color=col, zorder=5,
                fontweight="bold")

    ax.text(0.985, 0.022, f"ISOA vs. rest   +/=/-  =  "
            f"{wtl[0]}/{wtl[1]}/{wtl[2]}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.6, color="#333333",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#d5d5d5", linewidth=0.6, alpha=0.94))


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 10.4))
    gs = GridSpec(3, 2, figure=fig, left=0.072, right=0.988, top=0.888,
                  bottom=0.070, hspace=0.40, wspace=0.19)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for i, spec in enumerate(PANELS):
        panel(fig.add_subplot(gs[i // 2, i % 2]), spec, labels[i])

    handles = [Patch(facecolor=COLORS[a], edgecolor=COLORS[a], alpha=0.4,
                     label=a) for a in ALGOS]
    handles += [
        Line2D([], [], marker="D", ls="none", mfc="white", mec="#555555",
               mew=1.1, ms=5, label="geometric mean"),
        Line2D([], [], marker="o", ls="none", color="#8a8a8a", ms=3,
               label="individual run"),
        Line2D([], [], color="#666666", lw=0.9,
               label="Wilcoxon rank-sum vs. ISOA"),
    ]

    # Wrapped: a single-line footer wider than the canvas makes the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).
    savefig(fig, "146_algo_boxplot_stats.png")


if __name__ == "__main__":
    main()
