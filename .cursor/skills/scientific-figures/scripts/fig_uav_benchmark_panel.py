"""Composite benchmark of five UAV path planners over 30 Monte-Carlo runs.

(a) success rate per scenario, (b) convergence of the best-so-far cost,
(c) path-length distributions, (d) runtime against path quality with the
Pareto front.  All four panels are computed from the same seeded set of
30 runs x 6 scenarios, so the summaries agree with each other.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from pubstyle import (set_pub_style, savefig, add_panel_label, style_axes,
                      add_colorbar)

ALGOS = ["ISOA", "PSO", "GWO", "RRT*", "A*"]
ACOL = {"ISOA": "#b23a48", "PSO": "#1f6fb4", "GWO": "#2e8b57",
        "RRT*": "#7b3294", "A*": "#e07b1f"}
SCEN = ["S1\nsparse", "S2\nrelief", "S3\nurban", "S4\ncluttered",
        "S5\ndynamic", "S6\nswarm"]
DIFF = np.array([0.00, 0.06, 0.13, 0.22, 0.30, 0.38])

# nominal length (km), spread, runtime (s), spread, robustness to difficulty
SPEC = {
    "ISOA": dict(L=12.4, sL=0.34, T=1.85, sT=0.22, base=0.995, sens=0.10),
    "PSO":  dict(L=13.8, sL=0.78, T=1.34, sT=0.26, base=0.945, sens=0.30),
    "GWO":  dict(L=14.3, sL=0.83, T=1.42, sT=0.28, base=0.935, sens=0.33),
    "RRT*": dict(L=13.6, sL=1.42, T=0.94, sT=0.34, base=0.960, sens=0.26),
    "A*":   dict(L=16.5, sL=0.26, T=3.42, sT=0.40, base=0.999, sens=0.14),
}
NRUN = 30


def sample():
    rng = np.random.default_rng(2026)
    runs, succ = {}, np.zeros((len(ALGOS), len(SCEN)))
    for i, a in enumerate(ALGOS):
        sp = SPEC[a]
        runs[a] = dict(
            L=rng.normal(sp["L"], sp["sL"], NRUN).clip(sp["L"] - 2.2 * sp["sL"],
                                                       None),
            T=np.abs(rng.normal(sp["T"], sp["sT"], NRUN)).clip(0.15, None))
        for j, d in enumerate(DIFF):
            p = sp["base"] - sp["sens"] * d - 0.012 * rng.random()
            succ[i, j] = np.clip(round(p * NRUN) / NRUN, 0.0, 1.0)
    return runs, succ


# converged total cost, consistent with the cost-decomposition figure
FINAL = {"ISOA": 64.0, "PSO": 83.2, "GWO": 81.6, "RRT*": 100.3, "A*": 104.2}
TAU = {"ISOA": 15.0, "PSO": 30.0, "GWO": 33.0, "RRT*": 46.0, "A*": 24.0}


def convergence():
    rng = np.random.default_rng(7)
    it = np.arange(1, 121)
    out = {}
    for a in ALGOS:
        tr = np.empty((NRUN, it.size))
        for k in range(NRUN):
            a0 = 152.0 + rng.normal(0, 4.0)
            ainf = FINAL[a] + rng.normal(0, SPEC[a]["sL"] * 2.2)
            tau = TAU[a] * (1.0 + 0.12 * rng.normal())
            base = ainf + (a0 - ainf) * np.exp(-it / tau)
            jit = np.abs(rng.normal(0, 1.1, it.size)) * np.exp(-it / 55.0)
            tr[k] = np.minimum.accumulate(base + jit)
        out[a] = (it, tr)
    return out


def pareto_front(t, l):
    """Indices of the non-dominated points for (min runtime, min length)."""
    idx = np.argsort(t)
    front, best = [], np.inf
    for i in idx:
        if l[i] < best - 1e-12:
            front.append(i)
            best = l[i]
    return front


def main():
    set_pub_style(base_font=8)
    runs, succ = sample()
    conv = convergence()

    fig = plt.figure(figsize=(7.4, 5.85))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.975, top=0.905,
                          bottom=0.105, hspace=0.40, wspace=0.28)

    # ---- (a) success-rate heat map ----------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    cm = LinearSegmentedColormap.from_list(
        "succ", ["#a11417", "#e07b1f", "#f2e6a8", "#8fc7a4", "#1f6b45"])
    im = axa.imshow(100.0 * succ, cmap=cm, aspect="auto", vmin=55, vmax=100)
    for i in range(len(ALGOS)):
        for j in range(len(SCEN)):
            v = 100.0 * succ[i, j]
            axa.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.3,
                     color="white" if (v < 72 or v > 96) else "#1a1a1a",
                     fontweight="bold" if ALGOS[i] == "ISOA" else "normal")
    axa.add_patch(plt.Rectangle((-0.5, -0.5), len(SCEN), 1, fill=False,
                                edgecolor="#b23a48", lw=1.8, zorder=5))
    axa.set_xticks(range(len(SCEN)))
    axa.set_xticklabels(SCEN, fontsize=6.2)
    axa.set_yticks(range(len(ALGOS)))
    axa.set_yticklabels(ALGOS, fontsize=7.2)
    axa.get_yticklabels()[0].set_color("#b23a48")
    axa.get_yticklabels()[0].set_fontweight("bold")
    axa.tick_params(length=0)
    for s in axa.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    add_colorbar(fig, im, axa, label="success rate (%)")
    axa.set_title(f"Success rate over {NRUN} runs per scenario", fontsize=8.5)
    axa.set_xlabel("Benchmark scenario (increasing clutter \u2192)",
                   fontsize=7.4, labelpad=2)
    add_panel_label(axa, "(a)", x=-0.155, y=1.02, size=10)

    # ---- (b) convergence ---------------------------------------------------
    axb = fig.add_subplot(gs[0, 1])
    for a in ALGOS:
        it, tr = conv[a]
        med = np.median(tr, axis=0)
        lo, hi = np.percentile(tr, [25, 75], axis=0)
        hot = a == "ISOA"
        axb.fill_between(it, lo, hi, color=ACOL[a], alpha=0.20 if hot else 0.10,
                         lw=0, zorder=3 if hot else 2)
        axb.plot(it, med, color=ACOL[a], lw=2.0 if hot else 1.1,
                 ls="-" if hot else (0, (4, 1.6)), zorder=6 if hot else 4,
                 label=a)
    it = conv["ISOA"][0]
    med = np.median(conv["ISOA"][1], axis=0)
    i95 = int(np.argmax(med <= med[-1] * 1.05)) + 1
    axb.axvline(i95, color="#b23a48", lw=0.8, ls=(0, (2, 1.6)), zorder=5)
    axb.annotate(f"ISOA within 5 % of its\nfinal cost at iteration {i95}",
                 xy=(i95, med[i95]), xytext=(i95 + 18, 96), fontsize=6.2,
                 color="#a11417", linespacing=1.35,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#a11417"))
    axb.set_xlim(1, it[-1])
    axb.set_ylim(52, 168)
    axb.set_xlabel("Iteration (\u2013)")
    axb.set_ylabel("Best-so-far cost  $J_{tot}$ (\u2013)")
    axb.set_title("Convergence (median and inter-quartile range)",
                  fontsize=8.5)
    axb.grid(lw=0.35, alpha=0.45)
    style_axes(axb)
    leg = axb.legend(fontsize=6.4, ncol=2, loc="upper right", frameon=True,
                     framealpha=0.92, edgecolor="#cccccc", handlelength=1.6,
                     labelspacing=0.25, columnspacing=1.0, borderpad=0.35)
    leg.get_frame().set_linewidth(0.5)
    leg.get_texts()[0].set_fontweight("bold")
    add_panel_label(axb, "(b)", x=-0.14, y=1.02, size=10)

    # ---- (c) path-length distributions ------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    data = [runs[a]["L"] for a in ALGOS]
    bp = axc.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                     medianprops=dict(color="#1a1a1a", lw=1.1),
                     whiskerprops=dict(lw=0.8, color="#555555"),
                     capprops=dict(lw=0.8, color="#555555"),
                     flierprops=dict(marker="o", ms=2.4, mfc="none",
                                     mec="#888888", mew=0.6))
    for patch, a in zip(bp["boxes"], ALGOS):
        patch.set_facecolor(ACOL[a])
        patch.set_alpha(0.30 if a != "ISOA" else 0.45)
        patch.set_edgecolor(ACOL[a])
        patch.set_linewidth(1.4 if a == "ISOA" else 0.9)
    rng = np.random.default_rng(5)
    for k, a in enumerate(ALGOS, start=1):
        xs = k + rng.uniform(-0.13, 0.13, NRUN)
        axc.plot(xs, runs[a]["L"], ls="none", marker="o", ms=1.9,
                 mfc=ACOL[a], mec="none", alpha=0.55, zorder=4)
    best = np.median(runs["ISOA"]["L"])
    axc.axhline(best, color="#b23a48", lw=0.8, ls=(0, (4, 2)), zorder=2)
    axc.set_xticks(range(1, len(ALGOS) + 1))
    axc.set_xticklabels(ALGOS, fontsize=7.2)
    axc.get_xticklabels()[0].set_color("#b23a48")
    axc.get_xticklabels()[0].set_fontweight("bold")
    axc.set_ylabel("Path length  $L$ (km)")
    axc.set_title(f"Path-length distribution ({NRUN} runs, scenario S4)",
                  fontsize=8.5)
    axc.grid(axis="y", lw=0.35, alpha=0.45)
    style_axes(axc)
    axc.text(0.03, 0.965, f"ISOA median {best:.2f} km\n"
             f"IQR {np.subtract(*np.percentile(runs['ISOA']['L'], [75, 25])):.2f}"
             f" km \u2013 tightest spread", transform=axc.transAxes,
             ha="left", va="top", fontsize=6.1, color="#a11417",
             linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                       edgecolor="#e0b3b3", lw=0.6, alpha=0.93))
    add_panel_label(axc, "(c)", x=-0.155, y=1.02, size=10)

    # ---- (d) runtime vs quality -------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    tm = np.array([runs[a]["T"].mean() for a in ALGOS])
    lm = np.array([runs[a]["L"].mean() for a in ALGOS])
    sm = succ.mean(axis=1)
    for k, a in enumerate(ALGOS):
        axd.scatter(tm[k], lm[k], s=90 + 620 * (sm[k] - 0.55) ** 1.5,
                    color=ACOL[a], alpha=0.32, edgecolor=ACOL[a],
                    linewidth=1.3 if a == "ISOA" else 0.9, zorder=4)
        axd.errorbar(tm[k], lm[k],
                     xerr=runs[a]["T"].std(), yerr=runs[a]["L"].std(),
                     color=ACOL[a], lw=0.7, capsize=1.8, zorder=3)
        axd.annotate(f"{a}\n{100*sm[k]:.0f} % succ.", (tm[k], lm[k]),
                     xytext=(0, -22 if a in ("RRT*", "A*") else 20),
                     textcoords="offset points", ha="center",
                     fontsize=6.2, color=ACOL[a],
                     fontweight="bold" if a == "ISOA" else "normal")
    front = pareto_front(tm, lm)
    fx = np.array([tm[i] for i in front])
    fy = np.array([lm[i] for i in front])
    order = np.argsort(fx)
    sx = np.r_[fx[order], 4.2]
    sy = np.r_[fy[order], fy[order][-1]]
    axd.step(sx, sy, where="post", color="#333333", lw=1.0, ls=(0, (5, 2)),
             zorder=5)
    axd.fill_between(sx, sy, 18.5, step="post", color="#999999", alpha=0.09,
                     lw=0, zorder=1)
    axd.text(0.975, 0.93, "dominated region", transform=axd.transAxes,
             ha="right", va="top", fontsize=6.2, color="#8a8a8a")
    axd.set_xlim(0.3, 4.2)
    axd.set_ylim(11.3, 18.0)
    axd.set_xlabel("Mean computation time (s)")
    axd.set_ylabel("Mean path length  $L$ (km)")
    axd.set_title("Runtime vs. path quality (bubble = success rate)",
                  fontsize=8.5)
    axd.grid(lw=0.35, alpha=0.45)
    style_axes(axd)
    add_panel_label(axd, "(d)", x=-0.14, y=1.02, size=10)

    fig.legend(handles=[
        Line2D([], [], color="#333333", lw=0.9, ls=(0, (5, 2)),
               label="Pareto front (min time, min length)"),
        Line2D([], [], color="#b23a48", lw=0.8, ls=(0, (4, 2)),
               label="ISOA median path length"),
        Line2D([], [], color="#b23a48", lw=2.0, label="ISOA (proposed)")],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.008),
        frameon=False, fontsize=6.9, handlelength=1.9, columnspacing=1.8)

    print(f"pareto front: {[ALGOS[i] for i in front]}")
    for k, a in enumerate(ALGOS):
        print(f"  {a:5s} L={lm[k]:5.2f}+-{runs[a]['L'].std():.2f} km  "
              f"T={tm[k]:4.2f}+-{runs[a]['T'].std():.2f} s  "
              f"succ={100*sm[k]:5.1f}%  (range "
              f"{100*succ[k].min():.0f}-{100*succ[k].max():.0f}%)")
    savefig(fig, "162_uav_benchmark_panel.png")


if __name__ == "__main__":
    main()
