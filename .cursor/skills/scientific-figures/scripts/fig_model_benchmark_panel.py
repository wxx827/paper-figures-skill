"""Composite model-benchmark panel: one figure that answers the whole comparison.

(a) Radar of six quality metrics, axis-wise min-max normalised so the shape
    differences are readable.
(b) Per-dataset rank heatmap over eight tabular benchmarks with average rank.
(c) Training cost against accuracy, bubble area proportional to parameter
    count, with the Pareto frontier traced.
(d) Mean AUROC with 95% CI over ten seeds and the individual seed results.

Every panel is derived from one latent skill vector, so the story is
internally consistent across the four views.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

MODELS = ["Logistic Reg.", "Random Forest", "LightGBM", "XGBoost",
          "FT-Transformer", "Deep Ensemble"]
COLORS = ["#4a6fa5", "#4a7c59", "#c39b3f", "#b0653a", "#8a5f9c", "#7d2f3d"]
SKILL = np.array([0.00, 0.50, 0.86, 0.90, 0.82, 1.00])
TRAIN_S = np.array([2.4, 46.0, 21.0, 63.0, 1720.0, 5240.0])
PARAMS_M = np.array([0.001, 11.8, 2.1, 3.4, 7.9, 38.5])
DATASETS = ["Adult", "Covertype", "Higgs", "Epsilon", "Jannis", "Aloi",
            "Helena", "Year"]
METRICS = ["AUROC", "AUPRC", "Accuracy", "Macro-F1", "Calibration\n(1 - ECE)",
           "Robustness\n(shifted test)"]
RANK_CMAP = LinearSegmentedColormap.from_list(
    "rank", ["#1f5b55", "#3f7f76", "#78a89f", "#b3ccc5", "#dde5e1",
             "#f2f0ea"])


def main():
    set_pub_style()
    rng = np.random.default_rng(20260812)
    nm, nd = len(MODELS), len(DATASETS)

    # ---- one latent skill vector drives every panel ----------------------
    base = rng.uniform(0.68, 0.80, nd)                   # dataset difficulty
    gain = rng.uniform(0.10, 0.19, nd)
    auc_ds = (base[None, :] + gain[None, :] * SKILL[:, None]
              + rng.normal(0, 0.017, (nm, nd)))          # model x dataset
    ranks = np.argsort(np.argsort(-auc_ds, axis=0), axis=0) + 1
    avg_rank = ranks.mean(axis=1)

    seeds = auc_ds.mean(axis=1)[:, None] + rng.normal(0, 0.0075, (nm, 10))
    mu = seeds.mean(axis=1)
    ci = 1.96 * seeds.std(axis=1, ddof=1) / np.sqrt(seeds.shape[1])

    # metric table: same ordering trend, metric-specific deviations
    dev = rng.normal(0, 0.16, (nm, len(METRICS)))
    dev[0, 4] += 1.15                                    # LR is well calibrated
    dev[1, 5] += 0.75                                    # RF is robust
    dev[4, 3] += 0.35
    raw = SKILL[:, None] * 1.6 + dev
    metrics = (raw - raw.min(axis=0)) / np.ptp(raw, axis=0) * 0.82 + 0.18

    fig = plt.figure(figsize=(11.2, 8.3))
    gs = GridSpec(2, 2, figure=fig, left=0.055, right=0.975, top=0.855,
                  bottom=0.115, width_ratios=[1.0, 1.22],
                  height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.24)
    ax_r = fig.add_subplot(gs[0, 0], projection="polar")
    ax_h = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])

    # ---- (a) radar -------------------------------------------------------
    ang = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    ring = np.concatenate([ang, ang[:1]])
    for m in range(nm):
        vals = np.concatenate([metrics[m], metrics[m, :1]])
        ax_r.plot(ring, vals, color=COLORS[m], lw=1.5, zorder=3)
        ax_r.fill(ring, vals, color=COLORS[m], alpha=0.055, zorder=2)
    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)
    ax_r.set_xticks(ang)
    ax_r.set_xticklabels(METRICS, fontsize=7.2)
    ax_r.tick_params(axis="x", pad=1.0)
    ax_r.set_ylim(0, 1.06)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["", "", "", ""])
    ax_r.grid(color="#cccccc", lw=0.5)
    ax_r.spines["polar"].set_linewidth(0.7)
    ax_r.spines["polar"].set_edgecolor("#bbbbbb")
    add_panel_label(ax_r, "(a)", x=-0.16, y=1.06, size=10)
    ax_r.set_title("Multi-metric profile\n"
                   "(axis-wise min-max normalised)", fontsize=8.6,
                   fontweight="bold", linespacing=1.45, pad=9)

    # ---- (b) rank heatmap ------------------------------------------------
    im = ax_h.imshow(ranks, cmap=RANK_CMAP, aspect="auto",
                     norm=Normalize(1, nm))
    for m in range(nm):
        for d in range(nd):
            ax_h.text(d, m, f"{ranks[m, d]}", ha="center", va="center",
                      fontsize=7.4, fontweight="bold",
                      color="white" if ranks[m, d] <= 2 else "#2b2b2b")
        ax_h.text(nd + 0.15, m, f"{avg_rank[m]:.2f}", ha="center",
                  va="center", fontsize=7.6, fontweight="bold",
                  color="#1a1a1a" if avg_rank[m] == avg_rank.min() else
                  "#4a4a4a")
    ax_h.text(nd + 0.15, -0.85, "Avg.\nrank", ha="center", va="center",
              fontsize=7.2, fontweight="bold", color="#1a1a1a",
              linespacing=1.35)
    ax_h.set_xlim(-0.5, nd + 0.75)
    ax_h.set_xticks(range(nd), DATASETS, fontsize=7.2, rotation=30,
                    ha="right", rotation_mode="anchor")
    ax_h.set_yticks(range(nm), MODELS, fontsize=7.4)
    for tick, c in zip(ax_h.get_yticklabels(), COLORS):
        tick.set_color(c)
        tick.set_fontweight("bold")
    ax_h.tick_params(length=0)
    ax_h.set_xticks(np.arange(-0.5, nd, 1), minor=True)
    ax_h.set_yticks(np.arange(-0.5, nm, 1), minor=True)
    ax_h.grid(which="minor", color="white", lw=0.9)
    ax_h.tick_params(which="minor", length=0)
    for s in ax_h.spines.values():
        s.set_visible(False)
    add_panel_label(ax_h, "(b)", x=-0.185, y=1.10, size=10)
    ax_h.set_title("Rank per benchmark dataset  (1 = best)", fontsize=8.6,
                   fontweight="bold", loc="left", pad=16)
    cb = fig.colorbar(im, ax=ax_h, fraction=0.026, pad=0.065, aspect=13)
    cb.set_ticks(range(1, nm + 1))
    cb.ax.invert_yaxis()
    cb.ax.tick_params(labelsize=6.6, length=2)
    cb.set_label("Rank", fontsize=7.2, labelpad=2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # ---- (c) cost vs accuracy -------------------------------------------
    sizes = 26 + 340 * (PARAMS_M / PARAMS_M.max()) ** 0.55
    order = np.argsort(TRAIN_S)
    front_x, front_y, best = [], [], -np.inf
    for i in order:
        if mu[i] > best:
            best = mu[i]
            front_x.append(TRAIN_S[i])
            front_y.append(mu[i])
    ax_s.step(front_x, front_y, where="post", color="#8c8c8c", lw=1.0,
              ls=(0, (4, 3)), zorder=1)
    ax_s.scatter(TRAIN_S, mu, s=sizes, c=COLORS, alpha=0.80,
                 edgecolors="white", linewidths=0.9, zorder=3)
    offs = [(0, 13), (0, -21), (0, -17), (0, 14), (0, -18), (0, 16)]
    for i, name in enumerate(MODELS):
        ax_s.annotate(name, (TRAIN_S[i], mu[i]), textcoords="offset points",
                      xytext=offs[i], ha="center", fontsize=7.0,
                      fontweight="bold", color=COLORS[i],
                      path_effects=[pe.withStroke(linewidth=2.0,
                                                  foreground="white")])
    ax_s.set_xscale("log")
    ax_s.set_xlabel("Training time per fold (s, log scale)", fontsize=8.0)
    ax_s.set_ylabel("Mean AUROC across 8 datasets", fontsize=8.0)
    ax_s.set_xlim(1.2, 16000)
    span = mu.max() - mu.min()
    ax_s.set_ylim(mu.min() - 0.30 * span, mu.max() + 0.26 * span)
    ax_s.tick_params(labelsize=7.2)
    ax_s.text(0.985, 0.045, "dashed line = Pareto frontier",
              transform=ax_s.transAxes, ha="right", va="bottom",
              fontsize=6.8, color="#7a7a7a", style="italic")
    bubble = [Line2D([], [], marker="o", ls="none", color="#9a9a9a",
                     alpha=0.55, markersize=np.sqrt(26 + 340 * (p / PARAMS_M.max()) ** 0.55) / 1.6,
                     label=f"{p:g} M") for p in (0.001, 3.4, 38.5)]
    ax_s.legend(handles=bubble, title="Parameters", loc="upper left",
                fontsize=6.6, title_fontsize=6.8, frameon=False,
                labelspacing=0.85, handletextpad=0.9, borderpad=0.2)
    style_axes(ax_s)
    add_panel_label(ax_s, "(c)", x=-0.135, y=1.03, size=10)
    ax_s.set_title("Accuracy versus training cost", fontsize=8.6,
                   fontweight="bold", loc="left", pad=4)

    # ---- (d) headline metric with CI ------------------------------------
    o = np.argsort(mu)
    ypos = np.arange(nm)
    ax_b.barh(ypos, mu[o] - 0.60, left=0.60, height=0.60,
              color=[COLORS[i] for i in o], alpha=0.88, edgecolor="white",
              linewidth=0.6, zorder=2)
    ax_b.errorbar(mu[o], ypos, xerr=ci[o], fmt="none", ecolor="#2b2b2b",
                  elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)
    for k, i in enumerate(o):
        ax_b.scatter(seeds[i], np.full(10, k) + rng.uniform(-0.16, 0.16, 10),
                     s=4.5, color="#1a1a1a", alpha=0.40, linewidths=0,
                     zorder=5)
        ax_b.text(mu[i] + ci[i] + 0.0035, k, f"{mu[i]:.4f}", ha="left",
                  va="center", fontsize=7.2, fontweight="bold",
                  color="#2b2b2b")
    ax_b.set_yticks(ypos, [MODELS[i] for i in o], fontsize=7.4)
    for tick, i in zip(ax_b.get_yticklabels(), o):
        tick.set_color(COLORS[i])
        tick.set_fontweight("bold")
    lo = mu.min() - 0.022
    ax_b.set_xlim(lo, mu.max() + 0.032)
    ax_b.set_ylim(-0.7, nm - 0.3)
    ax_b.set_xlabel("Mean AUROC  (95% CI over 10 seeds; dots = seeds; "
                    "axis truncated)", fontsize=7.6)
    ax_b.tick_params(labelsize=7.0, length=2.6)
    ax_b.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax_b.axvline(mu.max(), color="#333333", lw=0.7, ls=(0, (3, 2)), zorder=1)
    style_axes(ax_b)
    add_panel_label(ax_b, "(d)", x=-0.185, y=1.03, size=10)
    ax_b.set_title("Headline metric with uncertainty", fontsize=8.6,
                   fontweight="bold", loc="left", pad=4)

    handles = [Line2D([], [], color=c, lw=2.4, label=n)
               for n, c in zip(MODELS, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7.8,
               frameon=False, bbox_to_anchor=(0.5, 0.048), handlelength=1.9,
               columnspacing=2.4, handletextpad=0.6)

    savefig(fig, "70_model_benchmark_panel.png")


if __name__ == "__main__":
    main()
