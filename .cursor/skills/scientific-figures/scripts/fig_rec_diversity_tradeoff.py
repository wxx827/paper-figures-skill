"""Figure 478 - Accuracy against diversity as the re-ranking weight is swept.

Main panel: every configuration is one bubble, positioned by slate category
entropy and NDCG@10 and sized by novelty, with the lambda sweep of each
re-ranker joined into a curve and the Pareto front over all of them dashed on
top.  The "better" corner is marked, because on a trade-off plot the reader's
first question is always which direction is good.

Auxiliary panels: what the sweep costs the proposed method one lambda at a
time, and how the top-10 slate composition shifts at three representative
settings.  The composition bars are not drawn freehand - the tier weights are
solved so that their entropy equals the entropy plotted in the main panel.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_label_inches

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID = "#cccccc", "#d9d9d9"

OURS = "#3b5b92"
PARETO = "#b23a48"      # the single emphasis: the frontier itself
TIERS = ["head", "upper mid", "lower mid", "tail", "fresh"]
TIER_C = ["#3a3a3a", "#5a5a5a", "#8a8a8a", "#b3b3b3", "#d5d5d5"]

LAMBDAS = np.array([0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8])
H0 = 0.98               # entropy of the pure-relevance slate, in bits
# name, colour, peak accuracy, accuracy decay, entropy gain, novelty gain
METHODS = [
    ("MMR",        "#b3b3b3", 0.352, 0.115, 0.72, 0.30),
    ("DPP",        "#8a8a8a", 0.356, 0.086, 0.88, 0.42),
    ("Calibrated", "#5a5a5a", 0.359, 0.070, 0.80, 0.36),
    ("Ours",       OURS,      0.368, 0.041, 1.10, 0.58),
]


def sweep(rng):
    """One (entropy, NDCG, novelty) triple per method and lambda."""
    out = {}
    for name, col, top, decay, egain, ngain in METHODS:
        ent = H0 + egain * (1.0 - np.exp(-2.6 * LAMBDAS))
        ndcg = top - decay * LAMBDAS ** 1.35
        nov = 0.18 + ngain * LAMBDAS ** 0.85
        jitter = rng.normal(0.0, 0.0022, LAMBDAS.size)
        out[name] = (ent + jitter * 3.0, ndcg + jitter, nov, col)
    return out


def pareto_front(x, y):
    """Indices of the non-dominated points when both axes are better-larger."""
    order = np.argsort(-x)
    keep, best = [], -np.inf
    for i in order:
        if y[i] > best:
            keep.append(i)
            best = y[i]
    return np.array(keep[::-1])


def tier_mix(target_h, n_tier):
    """Geometric tier weights whose Shannon entropy matches ``target_h``."""
    i = np.arange(n_tier)
    betas = np.linspace(0.0, 9.0, 6001)
    ws = np.exp(-betas[:, None] * i[None, :])
    ws = ws / ws.sum(axis=1, keepdims=True)
    h = -(ws * np.log2(np.clip(ws, 1e-12, None))).sum(axis=1)
    return ws[int(np.argmin(np.abs(h - target_h)))]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(478)
    data = sweep(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.74, 0.66, 5.22, 4.82))

    # ---- (a) trade-off cloud ----------------------------------------------
    all_x, all_y = [], []
    for name, col, *_ in METHODS:
        ent, ndcg, nov, c = data[name]
        all_x.append(ent)
        all_y.append(ndcg)
        ours = name == "Ours"
        ax.plot(ent, ndcg, lw=2.0 if ours else 1.1, color=c,
                zorder=5 if ours else 3, solid_capstyle="round")
        ax.scatter(ent, ndcg, s=28 + 620 * nov ** 1.6,
                   facecolor=lighten(c, 0.55 if ours else 0.35),
                   edgecolor=c, linewidth=1.0 if ours else 0.7,
                   zorder=6 if ours else 4)
        if not ours:
            ax.text(ent[-1], ndcg[-1] - 0.011, name, ha="center", va="top",
                    fontsize=7.2, color=c, zorder=7)

    fx = np.concatenate(all_x)
    fy = np.concatenate(all_y)
    idx = pareto_front(fx, fy)
    ax.plot(fx[idx], fy[idx], lw=1.4, color=PARETO, ls=(0, (4.5, 2.2)),
            zorder=8)

    ax.set_xlim(fx.min() - 0.10, fx.max() + 0.13)
    ax.set_ylim(fy.min() - 0.032, fy.max() + 0.026)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Representative configurations, all read back off the swept arrays.
    ent_o, ndcg_o, nov_o, _ = data["Ours"]
    for li, dx, dy, ha, va in ((0, 0.012, 0.008, "left", "bottom"),
                               (3, 0.0, 0.009, "center", "bottom"),
                               (6, 0.018, -0.010, "right", "top")):
        ax.annotate(f"$\\lambda$ = {LAMBDAS[li]:.2f}\n"
                    f"H {ent_o[li]:.2f} bits | NDCG {ndcg_o[li]:.3f}",
                    xy=(ent_o[li], ndcg_o[li]),
                    xytext=(ent_o[li] + dx, ndcg_o[li] + dy),
                    fontsize=6.6, color=INK, ha=ha, va=va, linespacing=1.35,
                    zorder=9,
                    bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                              edgecolor=HAIR, linewidth=0.6, alpha=0.94))
    ax.text(ent_o[-1] + 0.012, ndcg_o[-1] + 0.007, "Ours", ha="right",
            va="bottom", fontsize=7.6, color=OURS, fontweight="bold",
            zorder=9)

    ax.set_xlabel("slate category entropy (bits)", fontsize=8.0)
    ax.set_ylabel("NDCG@10 (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_title("Every configuration of every re-ranker", fontsize=9.0,
                 fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.annotate("", xy=(x1 - 0.035 * (x1 - x0), y1 - 0.035 * (y1 - y0)),
                xytext=(x1 - 0.19 * (x1 - x0), y1 - 0.155 * (y1 - y0)),
                arrowprops=dict(arrowstyle="-|>", color="#3a3a3a", lw=1.3),
                zorder=9)
    ax.text(x1 - 0.045 * (x1 - x0), y1 - 0.165 * (y1 - y0), "better",
            ha="right", va="top", fontsize=7.4, color="#3a3a3a",
            fontweight="bold", zorder=9)

    size_h = [Line2D([], [], marker="o", ls="none", mfc="#d5d5d5",
                     mec="#5a5a5a", mew=0.7,
                     ms=np.sqrt(28 + 620 * v ** 1.6) * 0.72,
                     label=f"novelty {v:.2f}") for v in (0.20, 0.45, 0.70)]
    size_h.append(Line2D([], [], color=PARETO, lw=1.4, ls=(0, (4.5, 2.2)),
                         label=f"Pareto front ({idx.size} of {fx.size})"))
    leg = ax.legend(handles=size_h, loc="lower left", fontsize=7.0,
                    frameon=True, labelspacing=0.75, borderaxespad=0.5,
                    handletextpad=0.9)
    leg.set_zorder(10)

    # ---- (b) sweep of the proposed method ---------------------------------
    axb = axes_inches(fig, (7.10, 3.28, 2.62, 2.10))
    axb.plot(LAMBDAS, ndcg_o, lw=1.6, color=OURS, marker="o", ms=3.4,
             mfc="white", mew=1.0, zorder=4)
    axb.set_xlabel("diversity weight $\\lambda$ (-)", fontsize=7.0)
    axb.set_ylabel("NDCG@10 (-)", fontsize=7.0, color=OURS)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.tick_params(axis="y", colors=OURS)
    axb.grid(axis="y", color=GRID, linewidth=0.6)
    axb.set_title("Cost of one more bit of entropy", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    axt = axb.twinx()
    axt.plot(LAMBDAS, ent_o, lw=1.4, color="#5a5a5a", ls=(0, (4.0, 2.0)),
             marker="^", ms=3.4, mfc="white", mew=1.0, zorder=4)
    axt.set_ylabel("entropy (bits)", fontsize=7.0, color="#5a5a5a")
    axt.tick_params(labelsize=6.5, length=2.2, colors="#5a5a5a")
    axb.spines["top"].set_visible(False)
    axt.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(True)

    d_h = ent_o - ent_o[0]
    d_pct = 100.0 * (ndcg_o[0] - ndcg_o) / ndcg_o[0]
    eff = d_h[1:] / np.maximum(d_pct[1:], 1e-9)
    k = int(np.argmax(eff)) + 1
    axb.text(0.96, 0.60, f"peak efficiency\n$\\lambda$ = {LAMBDAS[k]:.2f}\n"
             f"{eff[k - 1]:.2f} bits / 1 % NDCG",
             transform=axb.transAxes, ha="right", va="center", fontsize=6.4,
             color=SUB, linespacing=1.4, zorder=6)

    # ---- (c) slate composition --------------------------------------------
    axc = axes_inches(fig, (7.10, 0.66, 2.62, 1.66))
    picks = [0, 3, 6]
    comp = np.array([tier_mix(ent_o[li], len(TIERS)) for li in picks])
    left = np.zeros(len(picks))
    ypos = np.arange(len(picks))[::-1]
    for c, (name, col) in enumerate(zip(TIERS, TIER_C)):
        axc.barh(ypos, 100 * comp[:, c], left=100 * left, height=0.56,
                 color=col, edgecolor="white", linewidth=0.6, zorder=3,
                 label=name)
        for r, y in enumerate(ypos):
            if comp[r, c] > 0.085:
                axc.text(100 * (left[r] + comp[r, c] / 2), y,
                         f"{100 * comp[r, c]:.0f}", ha="center", va="center",
                         fontsize=6.0, color="white" if c < 3 else INK,
                         zorder=5)
        left = left + comp[:, c]
    axc.set_yticks(ypos)
    axc.set_yticklabels([f"$\\lambda$ = {LAMBDAS[i]:.2f}" for i in picks],
                        fontsize=6.8)
    axc.set_xlim(0, 100)
    axc.set_xlabel("share of the top-10 slate (%)", fontsize=7.0)
    axc.tick_params(labelsize=6.5, length=2.2)
    axc.set_title("Slate composition by popularity tier", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    axc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
               fontsize=6.4, frameon=False, handlelength=1.0,
               columnspacing=0.9, handletextpad=0.4)
    for s in ("top", "right", "left"):
        axc.spines[s].set_visible(False)
    axc.tick_params(axis="y", length=0)

    on_front = sum(1 for i in idx if i >= 3 * LAMBDAS.size)
    held = (f"all {idx.size} Pareto points" if on_front == idx.size
            else f"{on_front} of {idx.size} Pareto points")
    panel_label_inches(fig, ax, "(a)", dx=-0.54, dy=0.10)
    panel_label_inches(fig, axb, "(b)", dx=-0.54, dy=0.10)
    panel_label_inches(fig, axc, "(c)", dx=-0.54, dy=0.10)
    savefig(fig, "478_rec_diversity_tradeoff.png")


if __name__ == "__main__":
    main()
