"""453 - Component ablation and depth behaviour of a message-passing network.

Left: which components each configuration switches on, drawn as geometric
ticks and crosses rather than glyphs.  Middle: the accuracy those
configurations reach, as the gain in points over the mean-aggregation baseline
- raw accuracy spans 70 to 95 percent across these five datasets, so a shared
colour scale on the raw numbers would hide every ablation effect; the gain is
the quantity that is actually comparable.  Right (subordinate): the depth
sweep of the two extreme configurations on ogbn-arxiv, which is where the
heat-map column for that dataset is read from, so the two panels cannot
disagree.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path

from pubstyle import (set_pub_style, savefig, style_axes, get_cmap, lighten,
                      add_panel_label)
from figkit import shared_norm, safe_colorbar, text_card

INK, RULE, SUB = "#1a1a1a", "#333333", "#555555"
MUTED, HAIR, GRID = "#777777", "#cccccc", "#e8e8e8"
OK, NO = "#2e7d5b", "#b3b3b3"
BASE_C, BEST_C = "#6f6f6f", "#b23a48"

CHECK = Path([(-1.00, 0.06), (-0.30, -0.66), (1.00, 0.72)],
             [Path.MOVETO, Path.LINETO, Path.LINETO])

# component columns, grouped
COLS = [("Mean", "agg"), ("Max", "agg"), ("Attention", "agg"),
        ("BatchNorm", "norm"), ("PairNorm", "norm"), ("Residual", "skip")]
GROUPS = [("Aggregation", 0, 3), ("Normalisation", 3, 5), ("Skip", 5, 6)]

# configuration -> components switched on
CONFIGS = [
    ("C1", ["Mean"]),
    ("C2", ["Max"]),
    ("C3", ["Attention"]),
    ("C4", ["Attention", "BatchNorm"]),
    ("C5", ["Attention", "PairNorm"]),
    ("C6", ["Mean", "PairNorm", "Residual"]),
    ("C7", ["Attention", "BatchNorm", "Residual"]),
    ("C8", ["Attention", "PairNorm", "Residual"]),
]

# component -> (accuracy effect in points, extra depth tolerance in log2 layers)
EFFECT = {"Mean": (0.00, 0.00), "Max": (-0.55, -0.15), "Attention": (0.95, 0.25),
          "BatchNorm": (0.50, 0.55), "PairNorm": (0.80, 1.35),
          "Residual": (1.10, 1.70)}

# dataset -> (baseline accuracy at 4 layers, effect scale, oversmoothing drop)
DATASETS = [("Cora", 81.4, 1.00, 6.2), ("CiteSeer", 70.9, 1.15, 7.4),
            ("PubMed", 78.7, 0.85, 5.1), ("ogbn-arxiv", 71.3, 1.25, 9.6),
            ("Reddit", 94.8, 0.55, 4.3)]
DEPTHS = np.array([2, 4, 8, 16, 32, 64])
L_REF = 4                     # depth the heat map is read at
N_SEED = 5


def cfg_terms(comps):
    gain = sum(EFFECT[c][0] for c in comps)
    tol = 1.05 + sum(EFFECT[c][1] for c in comps)
    return gain, tol


def accuracy(base, scale, drop, comps, depth, jitter=0.0):
    """Accuracy of one configuration at one depth: ceiling minus oversmoothing."""
    gain, tol = cfg_terms(comps)
    d = np.log2(np.asarray(depth, dtype=float))
    return base + scale * gain - drop / (1.0 + np.exp(-(d - tol - 1.55) / 0.62)) \
        + jitter


def sym_tick(ax, x, y, on):
    if on:
        ax.plot([x], [y], marker="o", ms=8.5, mfc=lighten(OK, 0.86),
                mec="none", ls="none", zorder=3)
        ax.plot([x], [y], marker=CHECK, ms=6.4, mec=OK, mew=1.6, mfc="none",
                ls="none", zorder=4)
    else:
        ax.plot([x], [y], marker="x", ms=3.6, mec=NO, mew=1.1, ls="none",
                zorder=3)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(453)

    names = [c[0] for c in CONFIGS]
    n_cfg, n_ds = len(CONFIGS), len(DATASETS)

    # per-(config, dataset) measurement noise, drawn once and reused everywhere
    jit = rng.normal(0.0, 0.13, (n_cfg, n_ds))

    acc4 = np.empty((n_cfg, n_ds))
    for i, (_, comps) in enumerate(CONFIGS):
        for j, (_, base, scale, drop) in enumerate(DATASETS):
            acc4[i, j] = accuracy(base, scale, drop, comps, L_REF, jit[i, j])
    delta = acc4 - acc4[0][None, :]

    norm = shared_norm([delta], symmetric=True)
    cmap = get_cmap("cb_diverging")

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.30, 0.84],
                          left=0.062, right=0.975, top=0.775, bottom=0.150,
                          wspace=0.34)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    gsr = gs[0, 2].subgridspec(2, 1, height_ratios=[0.56, 0.40], hspace=0.34)
    ax_d = fig.add_subplot(gsr[0])
    ax_t = fig.add_subplot(gsr[1])

    best = int(np.argmax(delta.mean(axis=1)))

    # ---- (a) component matrix ---------------------------------------------
    for i, (name, comps) in enumerate(CONFIGS):
        if i % 2 == 0:
            ax_c.add_patch(Rectangle((-0.5, i - 0.5), len(COLS), 1.0,
                                     facecolor="#f7f7f7", edgecolor="none",
                                     zorder=0))
        for j, (col, _) in enumerate(COLS):
            sym_tick(ax_c, j, i, col in comps)
    for _, j0, j1 in GROUPS[:-1]:
        ax_c.axvline(j1 - 0.5, color=HAIR, lw=0.8, zorder=2)
    for gname, j0, j1 in GROUPS:
        ax_c.text(0.5 * (j0 + j1 - 1), n_cfg - 0.30, gname, ha="center",
                  va="top", fontsize=7.0, color=SUB, fontweight="bold",
                  clip_on=False)
    ax_c.set_xlim(-0.5, len(COLS) - 0.5)
    ax_c.set_ylim(n_cfg - 0.5, -0.5)
    ax_c.set_xticks(range(len(COLS)))
    ax_c.set_xticklabels([c[0] for c in COLS], fontsize=6.4, rotation=40,
                         ha="left", rotation_mode="anchor")
    ax_c.xaxis.set_ticks_position("top")
    ax_c.set_yticks(range(n_cfg))
    ax_c.set_yticklabels(names, fontsize=7.2)
    ax_c.tick_params(length=0, pad=2.5)
    for lab in (ax_c.get_yticklabels()[best],):
        lab.set_color(BEST_C)
        lab.set_fontweight("bold")
    for s in ax_c.spines.values():
        s.set_visible(False)

    # ---- (b) accuracy gain heat map ---------------------------------------
    im = ax_h.imshow(delta, cmap=cmap, norm=norm, aspect="auto",
                     interpolation="nearest")
    ax_h.set_xticks(range(n_ds))
    ax_h.set_xticklabels([d[0] for d in DATASETS], fontsize=6.6, rotation=28,
                         ha="left", rotation_mode="anchor")
    ax_h.xaxis.set_ticks_position("top")
    ax_h.set_yticks(range(n_cfg))
    ax_h.set_yticklabels(names, fontsize=7.2)
    ax_h.tick_params(length=0, pad=2.5)
    ax_h.set_ylim(n_cfg - 0.5, -0.5)
    for s in ax_h.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    for i in range(n_cfg):
        for j in range(n_ds):
            v = delta[i, j]
            txt = f"{acc4[i, j]:.1f}" if i == 0 else f"{v:+.1f}"
            hot = abs(v) / max(abs(norm.vmin), abs(norm.vmax)) > 0.62
            ax_h.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                      color="white" if hot else INK,
                      fontweight="bold" if i == best else "normal")
        ax_h.axhline(i + 0.5, color="white", lw=0.7)
    for j in range(n_ds):
        ax_h.axvline(j + 0.5, color="white", lw=0.7)
        jb = int(np.argmax(delta[:, j]))
        ax_h.add_patch(Rectangle((j - 0.5, jb - 0.5), 1.0, 1.0, fill=False,
                                 edgecolor=BEST_C, lw=1.3, zorder=5))
    for ax, lab, title in ((ax_c, "(a)", "Components switched on"),
                           (ax_h, "(b)", "Accuracy gain over the C1 baseline "
                                         "(points)")):
        x0 = ax.get_position().x0
    cb = safe_colorbar(fig, im, ax_h, label="accuracy gain (points)",
                       side="bottom", width=0.10, pad=0.10, shrink=0.86,
                       labelsize=6.6)
    cb.ax.tick_params(labelsize=6.4)

    # ---- (c) depth sweep of the two extremes (subordinate) -----------------
    j_ar = [d[0] for d in DATASETS].index("ogbn-arxiv")
    _, base, scale, drop = DATASETS[j_ar]
    curves = {}
    for i in (0, best):
        comps = CONFIGS[i][1]
        mean = accuracy(base, scale, drop, comps, DEPTHS, jit[i, j_ar])
        seeds = mean[None, :] + rng.normal(0.0, 0.22, (N_SEED, DEPTHS.size))
        curves[i] = (mean, seeds)

    for i, col in ((0, BASE_C), (best, BEST_C)):
        mean, seeds = curves[i]
        sd = seeds.std(axis=0)
        ax_d.fill_between(DEPTHS, mean - sd, mean + sd, color=col, alpha=0.18,
                          lw=0, zorder=2)
        ax_d.plot(DEPTHS, mean, color=col, lw=1.7 if i == best else 1.3,
                  marker="o", ms=3.2, mfc="white", mec=col, mew=1.0, zorder=4)
        ax_d.text(DEPTHS[0] * 1.07, mean[0] - 0.35, names[i], ha="left",
                  va="top", fontsize=7.2, color=col, fontweight="bold")
    ax_d.axvline(L_REF, color=MUTED, lw=0.8, ls=(0, (3.0, 2.0)), zorder=1)
    ax_d.text(5.2, 0.30, f"{L_REF} layers =\nheat-map row",
              transform=ax_d.get_xaxis_transform(), ha="left", va="center",
              fontsize=6.2, color=MUTED, linespacing=1.2)
    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks(DEPTHS)
    ax_d.set_xticklabels([f"{d}" for d in DEPTHS], fontsize=6.6)
    ax_d.tick_params(axis="x", which="minor", bottom=False)
    ax_d.tick_params(labelsize=6.8, length=2.4)
    ax_d.set_xlabel("Message-passing layers (-)", fontsize=7.4, labelpad=2.0)
    ax_d.set_ylabel("Accuracy on ogbn-arxiv (%)", fontsize=7.4)
    ax_d.grid(color=GRID, lw=0.7, zorder=0)
    ax_d.set_axisbelow(True)
    style_axes(ax_d)
    ax_d.set_xlim(1.62, 82.0)
    ax_d.set_title(f"Depth sweep: {names[0]} vs {names[best]}", fontsize=8.0,
                   fontweight="bold", pad=3.0)
    add_panel_label(ax_d, "(c)", x=-0.20, y=1.06)
    for t in ax_d.texts:
        t.set_clip_on(False)

    m_base = curves[0][0]
    m_best = curves[best][0]
    k32 = int(np.where(DEPTHS == 32)[0][0])
    k4 = int(np.where(DEPTHS == L_REF)[0][0])
    ax_d.annotate("", xy=(32, m_best[k32]), xytext=(32, m_base[k32]),
                  arrowprops=dict(arrowstyle="<->", color=BEST_C, lw=0.9,
                                  shrinkA=1.0, shrinkB=1.0), zorder=6)
    ax_d.text(30, 0.5 * (m_best[k32] + m_base[k32]),
              f"{m_best[k32] - m_base[k32]:+.1f} pp\nat 32 layers", ha="right",
              va="center", fontsize=6.2, color=BEST_C, linespacing=1.2)

    # ---- (d) read-out card --------------------------------------------------
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")
    keep_b = 100.0 * m_best[k32] / m_best[k4]
    keep_a = 100.0 * m_base[k32] / m_base[k4]
    text_card(
        ax_t, 0.0, 0.02, 1.0, 0.96,
        f"{names[best]} = attention + PairNorm + residual\n"
        f"\u2022 mean gain over C1: {delta[best].mean():+.1f} pp\n"
        f"\u2022 best on {int((np.argmax(delta, axis=0) == best).sum())} of "
        f"{n_ds} datasets\n"
        f"\u2022 at 32 layers keeps {keep_b:.0f}% of its {L_REF}-layer\n"
        f"   accuracy, against {keep_a:.0f}% for {names[0]}\n"
        f"\u2022 PairNorm {delta[4].mean() - delta[2].mean():+.1f} pp, "
        f"residual {delta[best].mean() - delta[4].mean():+.1f} pp",
        fs=6.3, va="top", ha="left", pad=0.055, rounding=0.045,
        title="Read-out", title_fs=7.4, color=SUB, title_color=INK)
    ax_t.texts[-1].set_y(0.86)

    handles = [
        Line2D([], [], color=OK, marker=CHECK, ms=6.4, mew=1.6, mfc="none",
               ls="none", label="component enabled"),
        Line2D([], [], color=NO, marker="x", ms=3.6, mew=1.1, ls="none",
               label="component disabled"),
        Line2D([], [], color=BEST_C, lw=1.3, ls="-",
               label="best configuration per dataset"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.062, 0.006),
               ncol=3, frameon=False, fontsize=6.8, handlelength=1.6,
               columnspacing=1.6)

    savefig(fig, "453_gnn_depth_ablation.png")


if __name__ == "__main__":
    main()
