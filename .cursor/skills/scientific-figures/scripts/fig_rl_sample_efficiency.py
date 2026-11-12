"""How many environment steps each algorithm needs, and what that costs.

Left: steps to reach 80% of expert return, on a log axis because the spread is
two orders of magnitude; bars are seed medians with the seed IQR as the error
bar.  Right: the same runs as a trade-off cloud - reaching the target sooner is
not free, and bubble area shows the wall-clock compute that bought it.  The
staircase is the computed non-dominated frontier, not a drawn line.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, annotate_box
from figkit import panel_label_inches

# algorithm -> (median steps to target, final return, GPU-hours, family)
ALGOS = [
    ("DQN", 8.4e6, 0.52, 14.0, "value"),
    ("A3C", 5.6e6, 0.58, 9.5, "policy"),
    ("PPO", 3.2e6, 0.66, 7.2, "policy"),
    ("Rainbow", 2.4e6, 0.74, 21.0, "value"),
    ("TD3", 1.35e6, 0.79, 11.5, "value"),
    ("SAC", 9.2e5, 0.83, 12.8, "value"),
    ("DreamerV3", 3.1e5, 0.93, 33.0, "model-based"),
    ("Ours", 1.45e5, 0.91, 18.5, "ours"),
]
FAM_C = {"value": "#8e6bbf", "policy": "#808080", "model-based": "#2e7d5b",
         "ours": "#3b5b92"}
TARGET = 0.80
N_SEED = 8
GRID = "#eeeeee"


def seed_steps(rng):
    """Per-seed steps-to-target: lognormal around the nominal median."""
    out = {}
    for name, med, _, _, _ in ALGOS:
        s = 0.42 if name != "Ours" else 0.30
        out[name] = med * np.exp(rng.normal(0, s, N_SEED))
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4410)
    steps = seed_steps(rng)

    names = [a[0] for a in ALGOS]
    cols = [FAM_C[a[4]] for a in ALGOS]
    med = np.array([np.median(steps[n]) for n in names])
    q1 = np.array([np.percentile(steps[n], 25) for n in names])
    q3 = np.array([np.percentile(steps[n], 75) for n in names])
    perf = np.array([a[2] for a in ALGOS])
    gpuh = np.array([a[3] for a in ALGOS])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.78], left=0.115,
                          right=0.975, top=0.855, bottom=0.115, wspace=0.24)
    axb = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1])

    # ---- (a) steps to target ----------------------------------------------
    y = np.arange(len(ALGOS))
    axb.barh(y, med, height=0.62, color=cols, edgecolor="white", lw=0.7,
             zorder=3, log=True)
    axb.errorbar(med, y, xerr=[med - q1, q3 - med], fmt="none",
                 ecolor="#333333", elinewidth=0.9, capsize=2.5, capthick=0.9,
                 zorder=5)
    for k, n in enumerate(names):
        hot = n == "Ours"
        axb.plot(steps[n], np.full(N_SEED, k) + rng.uniform(-0.16, 0.16,
                                                            N_SEED),
                 ls="none", marker="o", ms=2.4, mfc="white", mec="#4a4a4a",
                 mew=0.5, alpha=0.85, zorder=6)
        axb.text(q3[k] * 1.35, k, f"{med[k] / 1e6:.2f}M" if med[k] >= 1e6
                 else f"{med[k] / 1e3:.0f}k", va="center", ha="left",
                 fontsize=7.0, color=cols[k] if hot else "#4a4a4a",
                 fontweight="bold" if hot else "normal", zorder=6)
    axb.set_yticks(y, names, fontsize=7.6)
    axb.get_yticklabels()[-1].set_fontweight("bold")
    axb.get_yticklabels()[-1].set_color(FAM_C["ours"])
    axb.invert_yaxis()
    axb.set_xscale("log")
    axb.set_xlim(3e4, 4e7)
    axb.set_xlabel(f"Environment steps to reach {TARGET:.0%} of expert return "
                   "(log scale)", fontsize=8)
    axb.set_title("Sample cost of the same target score", fontsize=8.8,
                  fontweight="bold", pad=4)
    axb.grid(axis="x", color=GRID, lw=0.7, which="major")
    axb.set_axisbelow(True)
    axb.tick_params(labelsize=7.0)
    best_base = med[:-1].min()
    annotate_box(axb, f"Ours needs {best_base / med[-1]:.1f}$\\times$ fewer "
                 f"steps than the\nbest baseline ({names[int(np.argmin(med[:-1]))]}), "
                 f"{med[0] / med[-1]:.0f}$\\times$ fewer than DQN",
                 loc="lower right", fontsize=6.8)
    style_axes(axb)

    # ---- (b) trade-off cloud ----------------------------------------------
    order = np.argsort(med)
    frontier, best = [], -np.inf
    for i in order:
        if perf[i] > best:
            frontier.append(i)
            best = perf[i]
    fx = np.array([med[i] for i in frontier])
    fy = np.array([perf[i] for i in frontier])
    sx, sy = [fx[0]], [fy[0]]
    for k in range(1, len(fx)):
        sx += [fx[k], fx[k]]
        sy += [fy[k - 1], fy[k]]
    axp.plot(sx, sy, color="#b23a48", lw=1.2, ls=(0, (3.4, 1.8)), zorder=4,
             label="non-dominated frontier")
    axp.scatter(med, perf, s=gpuh * 11.0, c=cols, alpha=0.85,
                edgecolors="white", linewidths=0.9, zorder=6)
    for k, n in enumerate(names):
        hot = n == "Ours"
        axp.annotate(n, xy=(med[k], perf[k]),
                     xytext=(0, 7 + 0.22 * np.sqrt(gpuh[k] * 11.0)),
                     textcoords="offset points", ha="center", fontsize=6.8,
                     color=cols[k], fontweight="bold" if hot else "normal",
                     zorder=7)
    axp.axhline(TARGET, color="#808080", lw=0.9, ls=(0, (3.0, 2.0)), zorder=2)
    axp.text(3.4e7, TARGET + 0.006, f"target {TARGET:.0%}", ha="right",
             va="bottom", fontsize=6.4, color="#777777")
    axp.annotate("", xy=(1.6e5, 0.505), xytext=(1.1e6, 0.505),
                 arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0))
    axp.text(4.2e5, 0.515, "better", ha="center", va="bottom", fontsize=6.8,
             color="#555555", style="italic")
    axp.set_xscale("log")
    axp.set_xlim(8e4, 3.4e7)
    axp.set_ylim(0.46, 0.99)
    axp.set_xlabel("Environment steps at the reported score (log scale)",
                   fontsize=8)
    axp.set_ylabel("Final return (normalised, 1.0 $=$ expert)", fontsize=8)
    axp.set_title("Sample cost against final score, bubble $=$ compute",
                  fontsize=8.8, fontweight="bold", pad=4)
    axp.grid(color=GRID, lw=0.7)
    axp.set_axisbelow(True)
    axp.tick_params(labelsize=7.0)
    style_axes(axp)

    bub = [Line2D([], [], ls="none", marker="o", mfc="#cfd8e2", mec="white",
                  ms=np.sqrt(g * 11.0) / 2.2, label=f"{g:.0f} h")
           for g in (10.0, 20.0, 33.0)]
    leg1 = axp.legend(handles=bub, loc="upper right", frameon=False,
                      fontsize=6.8, title="wall-clock (GPU-h)",
                      labelspacing=1.0, borderpad=0.6, handletextpad=1.0)
    leg1.get_title().set_fontsize(6.8)
    leg1.get_title().set_color("#555555")

    handles = [Line2D([], [], ls="none", marker="s", ms=6, mfc=c, mec="white",
                      label=f)
               for f, c in FAM_C.items()]
    handles += [Line2D([], [], color="#b23a48", lw=1.2, ls=(0, (3.4, 1.8)),
                       label="non-dominated frontier"),
                Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white",
                       mec="#4a4a4a", label=f"per-seed run (n$=${N_SEED})")]

    for ax, lab in ((axb, "(a)"), (axp, "(b)")):
        panel_label_inches(fig, ax, lab, dx=-0.62, dy=0.08)

    savefig(fig, "441_rl_sample_efficiency.png")


if __name__ == "__main__":
    main()
